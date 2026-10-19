"""Rebuild a complete bilingual report from frozen prior and dated new evidence.

Usage: python render_completion.py EXPERIMENT [--pdf]
This is an offline operation. It never makes model or database requests.
"""
import argparse
import base64
import copy
import html
import json
import math
import os
import re
import logging
import shlex
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
import markdown
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as PlotPath
from sparkbench.completion import write_summary, quantile


BLUE, PALE, DARK, MUTED, GRID = '#0040FF', '#b9d4ff', '#10213d', '#506079', '#e5edf9'


def fmt(value, digits=2):
    return '—' if value is None else f'{value:,.{digits}f}'


def table(headers, rows):
    return '\n| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + ''.join('| ' + ' | '.join(str(v) for v in row) + ' |\n' for row in rows) + '\n'


def environment_section(inventory,lang):
    zh=lang=='zh';a,b=inventory['0'],inventory['1']
    if any(a[key]!=b[key] for key in ['gpu_model','driver_version','compute_capability','os','kernel','python','installed_sdk_packages']):
        raise ValueError('Node environment differences require an explicit two-column inventory')
    py=a['python'];pkg=py['packages'];sdk=a['installed_sdk_packages']
    rows=[
        ['GPU型号 / 架构' if zh else 'GPU model / architecture',a['gpu_model']+' / Blackwell; SM '+a['compute_capability']],
        ['GPU驱动' if zh else 'GPU driver',a['driver_version']],
        ['操作系统 / 内核' if zh else 'OS / kernel',a['os']+' / '+a['kernel']],
        ['CUDA Toolkit / nvcc',sdk['cuda-toolkit-13-0']+' / '+sdk['cuda-nvcc-13-0']],
        ['CUDA Runtime / PyTorch CUDA',sdk['cuda-cudart-13-0']+' / '+py['torch_cuda_build']],
        ['cuBLAS（系统安装包）' if zh else 'cuBLAS (system package)',sdk['libcublas-13-0']],
        ['NCCL / cuDNN',pkg['nvidia-nccl-cu13']+' / '+pkg['nvidia-cudnn-cu13']],
        ['Python / SGLang',py['python']+' / '+pkg['sglang']],
        ['PyTorch / Transformers',pkg['torch']+' / '+pkg['transformers']],
        ['FlashInfer / Triton',pkg['flashinfer-python']+' / '+pkg['triton']],
        ['safetensors',pkg['safetensors']],
        ['ModelOpt（节点安装版）' if zh else 'ModelOpt (installed)',pkg['nvidia-modelopt']],
        ['UCX（系统安装包）' if zh else 'UCX (system package)',sdk['libucx0']],
        ['Nsight Compute / Nsight Systems',sdk['nsight-compute-2025.3.1']+' / '+sdk['nsight-systems-2025.3.2']],
    ]
    args=list(a['launch_argv']);args[args.index('--node-rank')+1]='${NODE_RANK}'
    token=lambda value:'"'+value+'"' if '${' in value else shlex.quote(value)
    lines=[' '.join(token(value) for value in args[:3])];i=3
    while i<len(args):
        group=[args[i]];i+=1
        if i<len(args) and not args[i].startswith('--'):group.append(args[i]);i+=1
        lines.append('  '+' '.join(token(value) for value in group))
    command='export NCCL_SOCKET_IFNAME='+shlex.quote(a['launch_environment']['NCCL_SOCKET_IFNAME'])+'\nexport SGLANG_HOST_IP="${NODE_IP}"\n'+' \\\n'.join(lines)
    return (('### 运行环境、SDK与启动配置\n\n下表为2026-10-19核验的推理服务环境，两节点版本一致。系统SDK安装版本与服务Python环境中的运行库分列；硬件型号为双NVIDIA DGX Spark / GB10。\n' if zh else
             '### Runtime environment, SDKs and launch configuration\n\nThis is the inference-service environment verified on October 19, 2026, matching across both NVIDIA DGX Spark / GB10 nodes. System SDK packages and libraries in the service Python environment are identified separately.\n')+
            table(['组件／型号','节点0与节点1的版本'] if zh else ['Component / model','Version on both nodes'],rows)+
            ('权重配置记录的量化工具为ModelOpt 0.46.0，与节点当前安装的ModelOpt版本分别记录。完整清单见`protocols/environment.json`。\n\n**实际启动参数**（密钥、机器地址与安装路径使用变量）：\n\n' if zh else
             'The weight configuration records ModelOpt 0.46.0 at quantization time, separately from the installed version. The full inventory is in `protocols/environment.json`.\n\n**Deployed launch arguments**, with variables for credentials, addresses and installation paths:\n\n')+
            '```bash\n'+command+'\n```\n\n'+
            ('`NODE_RANK`在两节点分别为0和1，`NODE_IP`为本节点地址，`NODE0_IP`为节点0的双机互联地址；`SGLANG_PYTHON`指服务虚拟环境中的Python，`MODEL_PATH`指固定权重版本目录，`SGLANG_API_KEY`由部署者提供。命令请求NEXTN，当前引擎解析为EAGLE；上下文262,144、TP=2、预测步数3、草稿词元4。\n' if zh else
             '`NODE_RANK` is 0 or 1; `NODE_IP` is the local node address and `NODE0_IP` is node 0 on the interconnect. `SGLANG_PYTHON` selects the service virtual environment, `MODEL_PATH` selects the pinned weights, and the operator supplies `SGLANG_API_KEY`. The requested NEXTN algorithm resolves to EAGLE: context 262,144, TP=2, three prediction steps and four draft tokens.\n'))


class SVG:
    def __init__(self, height=500):
        matplotlib.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'infron-gb10-completion-v1',
                                    'font.family':['General Sans','PingFang SC','Noto Sans CJK SC','Noto Sans SC','Arial','DejaVu Sans']})
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        self.figure=Figure(figsize=(10.4,height/100),dpi=100,facecolor='white')
        self.axis=self.figure.add_axes([0,0,1,1]);self.axis.set_xlim(0,1040);self.axis.set_ylim(height,0);self.axis.axis('off')
        self.parts=[]

    def text(self, x, y, value, size=16, color=DARK, weight='normal', anchor='start'):
        self.axis.text(x,y,str(value),fontsize=size*.72,color=color,fontweight=weight,
                       ha={'start':'left','middle':'center','end':'right'}[anchor],va='baseline')

    def rect(self, x, y, w, h, color=BLUE):
        if w>0:
            self.axis.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=3',facecolor=color,edgecolor='none'))

    def line(self, x1, y1, x2, y2, color=GRID, dash=''):
        self.axis.plot([x1,x2],[y1,y2],color=color,linewidth=.72,linestyle=(0,(6,4)) if dash else '-',zorder=0)

    def write(self, path):
        for part in self.parts:
            commands=re.findall(r'([ML])([-\d.]+),([-\d.]+)',re.search(r'd="([^"]*)"',part)[1])
            vertices=[(float(x),float(y)) for _,x,y in commands]
            codes=[PlotPath.MOVETO if command=='M' else PlotPath.LINETO for command,_,_ in commands]
            if vertices:
                color=re.search(r'stroke="([^"]+)"',part)[1]
                self.axis.add_patch(PathPatch(PlotPath(vertices,codes),fill=False,edgecolor=color,linewidth=1.08))
        self.figure.savefig(path,format='svg',metadata={'Creator':'Infron SparkBench / Matplotlib','Date':None})
        path=Path(path)
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        self.figure.clear()


def charts(summary, telemetry, out, lang):
    zh = lang == 'zh'
    coverage = summary['coverage']
    s = SVG(410)
    rows = [('硬件与算子 / Hardware', 3740, 3740), ('双机集合通信 / Collectives', 840, 840),
            ('原生固定输出 / Native fixed output', 16, 16),
            ('10-19 公网长度 / Public lengths', coverage['length_conditions_complete'], 8),
            ('10-19 干扰结果 / Observed interference pairs', coverage['interference_pairs_observed'], 5),
            ('10-19 缓存配对 / Cache pairs', coverage['cache_pairs_qualified'], 3)]
    s.text(24, 32, '固定清单的有效证据覆盖' if zh else 'Qualified evidence for each fixed list', 22, weight='bold')
    for i, (label, n, total) in enumerate(rows):
        y = 72 + 49 * i
        s.text(24, y + 17, label.split(' / ')[0 if zh else 1], 16)
        s.rect(420, y, 440, 24, GRID); s.rect(420, y, 440 * n / total, 24)
        s.text(1000, y + 18, f'{n:,} / {total:,}', 18, BLUE, 'bold', 'end')
    s.text(24, 389, '各行分母独立；早期失败记录保留，日期之间不合并成功率。' if zh else 'Each row has its own denominator; earlier failures and dates remain separate.', 14, MUTED)
    s.write(out / f'completion_coverage.{lang}.svg')

    s = SVG(570)
    s.text(24, 32, '公网生成：输出上限与实际完成量' if zh else 'Public generation: output cap and realized output', 22, weight='bold')
    s.rect(610, 47, 18, 12, PALE); s.text(636, 59, '上限' if zh else 'Cap', 14)
    s.rect(754, 47, 18, 12); s.text(780, 59, '实际生成' if zh else 'Actual output', 14)
    x0, width = 280, 560
    scale = lambda value: max(0, math.log2(max(value, 1) / 512)) / 9 * width
    for tick in [1024, 4096, 16384, 65536, 262144]:
        x = x0 + scale(tick); s.line(x, 82, x, 510); s.text(x, 538, f'{tick // 1024}K', 14, MUTED, anchor='middle')
    for i, row in enumerate(summary['lengths']):
        y = 90 + 50 * i
        label = f"{(row.get('target_input_tokens') or 0)//1024}K → {row['output_cap']/1024:g}K"
        s.text(24, y + 22, label, 16)
        s.rect(x0, y, scale(row['output_cap']), 14, PALE)
        actual = row.get('origin_output_tokens')
        if row['qualified_response']:
            s.rect(x0, y + 17, scale(actual), 14)
            s.text(1000, y + 26, fmt(actual, 0), 17, BLUE, 'bold', 'end')
        else:
            s.text(1000, y + 26, '未验收' if zh else 'Pending', 16, MUTED, anchor='end')
    s.text(24, 561, '输入 → 输出上限；横轴为对数词元数量。每条件1次；自然终止。' if zh else 'Input → output cap; logarithmic token axis. One observation per condition; natural stopping.', 14, MUTED)
    s.write(out / f'completion_lengths.{lang}.svg')

    s = SVG(520)
    s.text(24, 32, '长输入加入后，短请求等待如何变化' if zh else 'Short-request waiting with additional long inputs', 22, weight='bold')
    qualified = [p for p in summary['interference'] if p['qualified']]
    maxvalue = max([p['arms'][m]['wait_s'][q] for p in qualified for m in ['isolated', 'mixed'] for q in ['median', 'p95']] + [1]) * 1.15
    for col, (q, title) in enumerate([('median', '中位数' if zh else 'Median'), ('p95', '95分位' if zh else '95th percentile')]):
        x0 = 140 + col * 480
        s.text(x0, 70, title + ('（秒）' if zh else ' (seconds)'), 18, weight='bold')
        for tick in range(5):
            x = x0 + tick / 4 * 320; s.line(x, 83, x, 440); s.text(x, 462, fmt(tick / 4 * maxvalue, 1), 13, MUTED, anchor='middle')
        for i, pair in enumerate(summary['interference']):
            y = 100 + i * 67
            s.text(x0 - 15, y + 22, str(i+1), 16, anchor='end')
            if pair['qualified']:
                for j, mode in enumerate(['isolated', 'mixed']):
                    value = pair['arms'][mode]['wait_s'][q]
                    s.rect(x0, y + j * 23, value / maxvalue * 320, 17, PALE if j == 0 else BLUE)
                    s.text(x0 + value / maxvalue * 320 + 6, y + j * 23 + 14, fmt(value, 2), 13)
            else:
                label=('含拒绝，不估计效应' if zh else 'Rejected arrivals; no effect estimate') if pair.get('observed') else ('完整配对待验收' if zh else 'Complete pair pending')
                s.text(x0, y + 25,label, 14, MUTED)
    s.rect(210, 490, 18, 12, PALE); s.text(236, 502, '仅短请求' if zh else 'Short only', 14)
    s.rect(530, 490, 18, 12); s.text(556, 502, '加入长请求' if zh else 'Additional long requests', 14)
    s.write(out / f'completion_interference.{lang}.svg')

    s = SVG(390)
    s.text(24, 32, '64K输入：零设计重叠的实际缓存词元比例' if zh else '64K inputs: measured reuse with zero designed overlap', 22, weight='bold')
    for i, pair in enumerate(summary['cache']):
        for j, role in enumerate(['seed', 'probe']):
            y = 68 + (2*i+j) * 43
            value = pair['arms'][role].get('cache_fraction')
            s.text(24, y + 18, f"{i+1} · " + (('首条' if role == 'seed' else '第二条') if zh else role), 16)
            s.rect(230, y, 610, 22, GRID)
            if value is not None:
                s.rect(230, y, value * 610, 22)
                s.text(1000, y + 18, fmt(value*100, 3)+'%', 18, BLUE, 'bold', 'end')
            else:
                s.text(1000, y + 18, '未归属' if zh else 'Unattributed', 16, MUTED, anchor='end')
    s.text(24, 365, '比例 = 可归属窗口的缓存词元增量 ÷ 输入词元增量；设计值0%不预设测量结果。' if zh else 'Ratio = attributed cached-token increment / input-token increment; a 0% design is not a measurement.', 14, MUTED)
    s.write(out / f'completion_cache.{lang}.svg')

    s = SVG(460)
    s.text(24, 32, '补测期间的双节点资源观测' if zh else 'Two-node resources during the dated measurements', 22, weight='bold')
    epochs = [r['epoch'] for r in telemetry]
    if epochs:
        begin, end = min(epochs), max(epochs)
        for panel, (name, lower, upper, title) in enumerate([('temperature_c', 30, 90, '温度 °C' if zh else 'Temperature °C'), ('gpu_util_pct', 0, 100, 'GPU利用率 %' if zh else 'GPU utilization %')]):
            top = 65 + panel * 175; bottom = top + 125
            s.text(24, top+10, title, 15)
            for tick in [lower, (lower+upper)/2, upper]:
                y = bottom - (tick-lower)/(upper-lower)*125; s.line(150, y, 995, y); s.text(138, y+5, fmt(tick, 0), 13, MUTED, anchor='end')
            if name == 'temperature_c':
                y = bottom - (85-lower)/(upper-lower)*125; s.line(150, y, 995, y, '#996b29', '6 4')
            for rank, color in [(0, BLUE), (1, '#658ac5')]:
                rows = sorted([r for r in telemetry if r['node']==rank], key=lambda r:r['epoch'])
                points=[]; previous=None
                for r in rows:
                    if r.get(name) is None:
                        previous=None;continue
                    x=150+(r['epoch']-begin)/max(end-begin,1)*845
                    y=bottom-(r[name]-lower)/(upper-lower)*125
                    command='M' if previous is None or r['epoch']-previous > 20 else 'L'
                    points.append(f'{command}{x:.2f},{y:.2f}');previous=r['epoch']
                s.parts.append(f'<path d="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
            for tick in range(5):
                s.text(150+tick*845/4, bottom+23, fmt((end-begin)/60*tick/4,1), 13, MUTED, anchor='middle')
        s.text(400, 433, '起始后的分钟数；蓝色：节点0；浅蓝：节点1' if zh else 'Minutes since start; blue: node 0; light blue: node 1', 14, MUTED)
    s.write(out / f'completion_resources.{lang}.svg')


def sections(summary, lang):
    zh = lang == 'zh'; c=summary['coverage']; bmap={b['id']:b for b in summary['blocks']}
    pairs=[p for p in summary['interference'] if p['qualified']]
    ratios=[p['mixed_over_isolated']['median'] for p in pairs]
    ratio_text = f"{fmt(min(ratios))}–{fmt(max(ratios))}" if ratios else '—'
    tails=[p['mixed_over_isolated']['p95'] for p in pairs]
    tail_text=f"{fmt(min(tails))}–{fmt(max(tails))}" if tails else '—'
    cache_values=[arm['cache_fraction'] for pair in summary['cache'] for arm in pair['arms'].values() if arm.get('cache_fraction') is not None]
    cache_range=f"{fmt(min(cache_values)*100,3)}–{fmt(max(cache_values)*100,3)}%" if cache_values else '—'
    max_public=max([r['origin_output_tokens'] for r in summary['lengths'] if r['qualified_response']]+[0])
    length_rows=[]
    for r in summary['lengths']:
        window=bmap.get(r['block_id'],{}).get('source_window',{})
        cache=window.get('cache_fraction')
        length_rows.append([fmt(r.get('origin_input_tokens'),0),fmt(r['output_cap'],0),f"**{fmt(r.get('origin_output_tokens'),0)}**",
                            fmt(r['first_visible_s']),fmt(r['e2e_s']),fmt(100*cache,2)+'%' if cache is not None else '—',r.get('finish_reason') or '—'])
    pair_rows=[]
    throughput_rows=[]
    rejection_rows=[]
    for pair in range(5):
        for mode in ['isolated','mixed']:
            rejected=[r for r in summary.get('rejections',[]) if r['pair']==pair and r['mode']==mode]
            if rejected:
                waits=[r['rejection_queue_s'] for r in rejected if r['rejection_queue_s'] is not None]
                rejection_rows.append([pair+1,('孤立' if mode=='isolated' else '混合') if zh else mode,
                                       f"{len(rejected)}/32",fmt(min(waits))+'–'+fmt(max(waits)),'未转发到模型' if zh else 'Not forwarded to model'])
    for p in summary['interference']:
        a,b=p['arms']['isolated'],p['arms']['mixed']
        pair_rows.append([str(p['pair']+1),f"{a.get('short_complete',0)}/32 → {b.get('short_complete',0)}/32",
                          fmt(a.get('wait_s',{}).get('median'))+' → **'+fmt(b.get('wait_s',{}).get('median'))+'**',
                          fmt(a.get('wait_s',{}).get('p95'))+' → **'+fmt(b.get('wait_s',{}).get('p95'))+'**',
                          '**'+fmt((p.get('mixed_over_isolated') or {}).get('median'))+'**',
                          str(a.get('local_rejections',0))+' / '+str(b.get('local_rejections',0)),
                          ('完整配对' if zh else 'Complete pair') if p['qualified'] else ('含拒绝；延迟仅描述完整响应' if zh else 'Rejections; latency describes completed responses only') if p.get('observed') else ('待验收' if zh else 'Pending')])
        for mode,arm in [('isolated',a),('mixed',b)]:
            block=bmap.get(arm.get('block_id'),{});window=block.get('source_window',{})
            fraction=window.get('cache_fraction')
            duration=block['end_epoch']-block['start_epoch'] if block.get('end_epoch') and block.get('start_epoch') else None
            throughput_rows.append([p['pair']+1,('孤立' if mode=='isolated' else '混合') if zh else mode,
                                    f"{block.get('offered') or 0} / {block.get('sent') or 0} / {block.get('completed') or 0}",
                                    fmt(duration,1),
                                    '**'+fmt(block.get('source_output_tokens_s'))+'**',
                                    fmt(arm.get('output_tokens',{}).get('median'),0),
                                    fmt(fraction*100,2)+'%' if fraction is not None else '—'])
    cache_rows=[]
    for p in summary['cache']:
        for role in ['seed','probe']:
            arm=p['arms'][role];delta=arm.get('delta',{});fraction=arm.get('cache_fraction')
            cache_rows.append([str(p['pair']+1),('首条' if role=='seed' else '第二条') if zh else role,
                               fmt(delta.get('input_tokens'),0),fmt(delta.get('cached_tokens'),0),
                               '**'+fmt(fraction*100,3)+'%**' if fraction is not None else '—',
                               ('有效' if arm['valid'] else '未验收') if zh else ('Valid' if arm['valid'] else 'Pending')])
    resource_rows=[]
    for rank,node in summary['nodes'].items():
        resource_rows.append([rank,fmt(node['temperature_c']['median'],1)+' / **'+fmt(node['temperature_c']['max'],1)+'**',
                              fmt(node['gpu_util_pct']['median'],1)+' / '+fmt(node['gpu_util_pct']['p95'],1),
                              fmt(node['gpu_clock_mhz']['median'],0),fmt(node['power_w']['median'],1),fmt(node['memory_available_gib']['min'],2)])
    coverage_rows=[['公网长度上限' if zh else 'Public output caps',f"**{c['length_conditions_complete']}/8**",'八个原未发送条件，各1条；自然终止。' if zh else 'One natural-stop observation for each of eight formerly unsent conditions.'],
                   ['长短请求干扰' if zh else 'Long/short interference',f"**{c['interference_pairs_observed']}/5**",f"全部到达与终态已记录；其中{c['interference_pairs_qualified']}对可估计完整配对效应，本地拒绝{c['local_rejections_observed']}条。" if zh else f"All arrivals and outcomes recorded; {c['interference_pairs_qualified']} pairs support a complete-pair effect estimate, with {c['local_rejections_observed']} local rejections retained."],
                   ['64K零设计重叠' if zh else '64K zero designed overlap',f"**{c['cache_pairs_qualified']}/3**",'每对2条；根据独占计数器窗口计算实际缓存比例。' if zh else 'Two requests per pair; actual reuse comes from attributable counter windows.']]
    f={}
    f['summary']=((f"### 2026-10-19 独立补测结论\n\n{table(['范围','有效证据','解释'],coverage_rows)}\n"
                  f"原生引擎已经验证最高 **260,096词元**固定输出；公网采用自然终止，八个输出上限条件不能替代固定长输出验证。完整干扰配对中，短请求首字等待的混合／孤立比值为：中位数 **{ratio_text}倍**，95分位 **{tail_text}倍**。各日期独立统计，早期失败记录保留。\n" if zh else
                  f"### Independent measurements on 2026-10-19\n\n{table(['Scope','Qualified evidence','Interpretation'],coverage_rows)}\n"
                  f"The native engine demonstrated fixed output up to **260,096 tokens**. Public requests stop naturally, so the eight output-cap conditions do not establish fixed long-output attainment. Across qualified pairs, mixed/isolated short-request waiting is **{ratio_text}×** at the median and **{tail_text}×** at p95. Dates remain separate and earlier failures are retained.\n"))
    f['protocol']=(("### 2026-10-19：长度、干扰与缓存补测方案\n\n两节点在专用窗口内测量，公网路径为 Infron → Cloudflare → 冻结请求转发入口 → SGLang。入口仅放行本批预注册载荷；两臂均经过同一转发路径。模型、量化与引擎启动参数保持归档配置，新增结果独立报告。\n\n"+
        table(['部分','固定数量','控制条件'],[['路由预检','1','32词元上限；核验实际路由和源站参数。'],['公网长度','8','原未发送的输入／输出组合；每条件1条。'],['干扰预热','320','5对×2臂×32条；每臂预热全部短输入，输出参数相同。'],['干扰正式短请求','320','5对×2臂×32条；配对输入和完整泊松到达序列相同，名义到达率0.12次/秒。'],['混合臂长请求','40','每对8条；32K输入、128输出上限，均匀散布于短请求窗口。'],['缓存对照','6','3个独立对，每对2条64K输入；早期唯一标记使设计前缀重叠为0%。']])+
        "总计 **695条逻辑请求**；预热、重试和正式测量分列。干扰臂交替先后顺序；客户端短／长工作协程分别最多4／2，总体授权上限8。完整保留32个短请求的到达时刻，不截断时间序列；队列上限32，出队时以5秒作为超时判定阈值；实际拒绝等待可以超过5秒，全部拒绝计入结果。温度达到85°C退避，低于75°C并稳定30秒后恢复。\n" if zh else
        "### 2026-10-19: length, interference and cache protocol\n\nBoth nodes are measured in a dedicated window. The public path is Infron → Cloudflare → frozen-request forwarding proxy → SGLang. Only registered payloads are admitted, and both arms use the same path. Model, quantization and engine launch configuration match the archived deployment; the new date remains a separate experiment.\n\n"+
        table(['Component','Fixed count','Controls'],[['Route preflight','1','32-token cap; verify actual route and source parameters.'],['Public lengths','8','One observation for each formerly unsent input/output condition.'],['Interference warmup','320','5 pairs × 2 arms × 32 short inputs; all inputs prewarmed with matching output parameters.'],['Measured short requests','320','5 × 2 × 32; identical prompts and full Poisson schedules within each pair, nominal rate 0.12/s.'],['Mixed-arm long requests','40','Eight per pair; 32K inputs, 128-token caps, uniformly distributed over the short-request horizon.'],['Cache controls','6','Three independent pairs of 64K inputs; early unique markers set designed prefix overlap to 0%.']])+
        "The plan fixes **695 logical requests**. Warmup, retries and measurements are separate. Arm order alternates; short/long client workers are limited to 4/2 within the authorized total ceiling 8. All 32 short arrivals are retained without truncation. The local queue holds at most 32 requests. A 5-second expiration threshold is checked on dequeue, so observed waiting before rejection can exceed 5 seconds; rejection remains an outcome. Thermal backoff starts at 85°C and resumes below 75°C after30 stable seconds.\n"))
    f['metrics']=(("### 补测指标的定义与纳入规则\n\n"+table(['指标','定义与边界'],[['从到达到首字的等待','首个可见正文到达客户端的时间 − 预定到达时间；包含客户端调度、排队和已核验重试。'],['首字等待中位数／95分位','每臂32条短请求的样本分位数，线性插值；95分位对少量尾部样本敏感，不报告p99。'],['干扰比值','同一配对的混合臂中位等待 ÷ 孤立臂中位等待；两臂全部短请求完整、载荷和到达匹配后才计算。'],['缓存词元比例','源站缓存词元增量 ÷ 输入词元增量；两端空闲、计数单调，完成数和输入／输出用量与本批逐请求记录一致。'],['实际生成量','源站原始usage的completion_tokens；公网usage、计费量和生成上限分别保存。'],['统计重复单位','5个配对区组；每组32条相关短请求不是32个独立实验。报告逐对结果与范围，不将全部请求当独立重复来缩窄置信区间。']])+"响应必须具有正常终态、流结束标志、有效源站用量及一致计数器窗口。缺失窗口显示不可用；0%命中只在有效分母上报告。\n" if zh else
        "### Additional metric definitions and inclusion rules\n\n"+table(['Metric','Definition and boundary'],[['Arrival-to-visible wait','Client receipt of first visible text minus scheduled arrival; includes client scheduling, queuing and verified retries.'],['Median/p95 waiting','Linear-interpolation sample quantiles of 32 short requests per arm. The p95 estimate is tail-sensitive; no p99 claim is made.'],['Interference ratio','Mixed-arm median wait divided by matched isolated-arm median wait, only after all short requests, payloads and arrival schedules match.'],['Cached-token fraction','Source cached-token increment / input-token increment. Both boundaries must be idle; monotonic request, input and output counters must reconcile with the block.'],['Realized generation','completion_tokens from the original source usage. Public usage, billed usage and requested output cap remain distinct.'],['Replication unit','Five paired blocks. The 32 correlated requests within a block are not32 independent experiments. Pair results and ranges are reported without a pooled pseudo-replication confidence interval.']])+"Responses require normal termination, a stream-end marker, source usage and a reconciled counter window. Missing windows are unavailable; zero reuse is reported only with a valid denominator.\n"))
    f['coverage']=(("### 2026-10-19 补测的独立分母\n\n" if zh else "### Independent denominators for the 2026-10-19 supplement\n\n")+table(['范围','有效／计划','定义'] if zh else ['Scope','Qualified/planned','Definition'],coverage_rows)+(
        f"本批已发送 **{c['selected_attempts_started']}** 条注册尝试、已取得 **{c['selected_source_complete']}** 条完整源站用量记录，另有 **{c['local_rejections_observed']}条**本地队列拒绝；完整观察的逻辑结果为 **{c['logical_outcomes_observed']}/695**。另保留独占入口启用前的4条探针及其他有界尝试，不混入当前比较。初始299条清单的历史完成率不回填。原来1条未完整记录属于32K输入／8K输出上限条件，该条件已在10月17日用相同输入和上限独立补测 **4/4** 完成；10月19日补充其余8个原未发送条件。对应关系保存在`protocols/gap-resolution.json`。\n" if zh else
        f"The selected attempts include **{c['selected_attempts_started']}** submitted requests, **{c['selected_source_complete']}** complete source-usage records and **{c['local_rejections_observed']} local queue rejections**: **{c['logical_outcomes_observed']}/695** observed logical outcomes. Four pre-isolation probes and any further bounded attempts remain separate. The historical 299-request completion rate is not backfilled. Its one incomplete response belonged to the 32K-input/8K-output-cap condition, independently repeated on October 17 with identical inputs and caps (**4/4** complete). October 19 covers the other eight formerly unsent conditions. The mapping is in `protocols/gap-resolution.json`.\n"))
    f['length']=(("### 2026-10-19 公网长度：上限与实际生成\n\n" if zh else "### 2026-10-19 public lengths: caps and actual generation\n\n")+(
        "输出上限只约束允许生成的最大词元数量。源站记录确认max_tokens透传，ignore_eos未透传；模型可提前自然结束。每条件只有1次观测，表中首字和总耗时是单请求记录，不能当作分布分位数。缓存状态随实测窗口列出。\n" if zh else
        "An output cap only limits the maximum permitted generation. Source records confirm max_tokens forwarding, while ignore_eos is not forwarded; the model can stop naturally. Each condition has one observation, so first-text and total times are individual measurements, not distribution quantiles. Measured cache state is shown explicitly.\n")+
        table(['实际输入','输出上限','实际输出','首字秒','总耗时秒','缓存词元比例','结束原因'] if zh else ['Actual input','Output cap','Actual output','First text s','Total s','Cached-token fraction','Finish'],length_rows)+f"\n![{'输出上限与实际生成' if zh else 'Output caps and realized generation'}](completion_lengths.{lang}.svg)\n\n"+(
        f"这组数据补齐 **{c['length_conditions_complete']}/8** 个公网条件的响应证据，本批自然输出的观测最大值为 **{max_public:,}词元**；固定输出合格数为 **{c['fixed_output_qualified']}**。观测最大值不是服务输出上限；原生引擎最高260,096词元的固定输出结果仍按独立路径解读。\n" if zh else
        f"This adds response evidence for **{c['length_conditions_complete']}/8** public conditions, with a maximum observed natural output of **{max_public:,} tokens**; **{c['fixed_output_qualified']}** qualify as fixed-output observations. The observed maximum is not a service limit. Native fixed generation up to 260,096 tokens remains a separate path.\n"))
    f['cache']=(("### 2026-10-19：64K零设计重叠的缓存核验\n\n设计0%表示两条输入在靠前位置使用不同标记，不表示缓存计数必然为零。此对照检验低复用输入的实际状态，与前述相同前缀立即复用实验共同构成两个端点；它不测TTL或长期缓存寿命。\n" if zh else
        "### 2026-10-19: cache verification at 64K with zero designed overlap\n\nA 0% design uses different markers early in the inputs; it does not assume zero measured reuse. This tests low-reuse inputs and complements the identical-prefix immediate-reuse experiment. It does not measure a TTL or long-term cache lifetime.\n")+
        table(['配对','请求','输入增量','缓存增量','实际比例','窗口'] if zh else ['Pair','Request','Input increment','Cached increment','Measured ratio','Window'],cache_rows)+f"\n![{'缓存计数器对照' if zh else 'Cache counter controls'}](completion_cache.{lang}.svg)\n\n"+(
        f"取得 **{c['cache_pairs_qualified']}/3** 个完整独立配对，已验收窗口的实际缓存词元比例为 **{cache_range}**。原来64K低复用窗口的缺失观测仍保留，本批提供可归属的新日期证据；两类计数窗口没有跨日期合并。\n" if zh else
        f"**{c['cache_pairs_qualified']}/3** independent pairs qualify, with measured cached-token fractions of **{cache_range}** in qualified windows. The earlier missing 64K low-reuse window remains missing; these are attributable observations from a new date and are not pooled across dates.\n"))
    f['interference']=(("### 2026-10-19：长请求对短请求等待的影响\n\n孤立臂只有32条短请求；混合臂保持相同短输入与到达时刻，并加入8条32K输入。正式测量前，两臂均预热全部短输入。下表只在完整配对上给出比值，同时保留所有拒绝；结果是该混合服务条件的整体影响，不能等同于纯预填充内核耗时。\n" if zh else
        "### 2026-10-19: effect of long requests on short-request waiting\n\nThe isolated arm has 32 short requests. The mixed arm preserves their prompts and scheduled arrivals and adds eight 32K inputs. Both arms prewarm every short input. Ratios require complete pairs, while all rejections remain reported. This measures the overall mixed-service effect, not pure prefill-kernel time.\n")+
        table(['配对','短请求完整数：孤立→混合','首字中位秒：孤立→混合','首字95分位秒：孤立→混合','中位比值','本地拒绝：孤立/混合','解释范围'] if zh else ['Pair','Complete short: isolated→mixed','Median wait s: isolated→mixed','p95 wait s: isolated→mixed','Median ratio','Rejections: isolated/mixed','Interpretation'],pair_rows)+f"\n![{'短请求配对等待' if zh else 'Paired short-request waiting'}](completion_interference.{lang}.svg)\n\n"+(
        f"已完整记录 **{c['interference_pairs_observed']}/5** 组到达与终态，其中 **{c['interference_pairs_qualified']}/5** 组可计算完整配对效应；中位等待比值范围 **{ratio_text}倍**，95分位比值范围 **{tail_text}倍**。这些范围描述区组之间的差异，不是置信区间。10月17日的3个不完整配对仍作为历史拒绝记录保留，未用于本表的效应估计。\n" if zh else
        f"All arrivals and outcomes are observed in **{c['interference_pairs_observed']}/5** pairs; **{c['interference_pairs_qualified']}/5** qualify for complete-pair effect estimates, with median-wait ratios of **{ratio_text}×** and p95 ratios of **{tail_text}×**. These are ranges across blocks, not confidence intervals. The three incomplete pairs from October 17 remain historical rejection evidence and are excluded from this effect estimate.\n")+
        (table(['配对','臂','短请求本地拒绝','拒绝时实际排队秒','执行状态'] if zh else ['Pair','Arm','Local short-request rejections','Observed queue wait at rejection s','Execution'],rejection_rows)+(
            '本地队列在出队时检查5秒阈值，因此记录到的实际等待可能更长。拒绝请求没有首字时间；含拒绝组的延迟统计只描述完成响应，不能作为完整配对效应。这些拒绝是预定准入规则下的观测结果，并非缺失测试记录。\n' if zh else
            'The local queue checks its 5-second threshold on dequeue, so actual waiting before rejection can be longer. Rejected requests have no first-text time. Latency summaries for an arm with rejections describe completed responses only and do not establish a complete-pair effect. These are observed admission outcomes, not missing test records.\n') if rejection_rows else '')+
        table(['配对','臂','到达／发送／完整','区组墙钟秒','区组输出词元/s','短请求输出中位词元','全臂缓存比例'] if zh else ['Pair','Arm','Offered/sent/complete','Block wall time s','Block output tokens/s','Median short output tokens','Whole-arm cache fraction'],throughput_rows)+(
        "区组吞吐以同一到达窗口内的源站实际输出总量除以从调度开始到全部响应结束的墙钟时间，包含到达间隔；它不代表饱和容量。全臂缓存比例按输入词元加权，混合臂包含长输入，不能直接当作短请求的命中率。\n" if zh else
        "Block throughput divides actual source output by wall time from scheduler start through the last response, including inter-arrival gaps; it is not saturated capacity. Whole-arm cache fractions are input-token weighted and include long prompts in mixed arms, so they are not short-request hit rates.\n"))
    f['resources']=(("### 2026-10-19：双节点温度与资源\n\n" if zh else "### 2026-10-19: two-node temperature and resources\n\n")+table(
        ['节点','温度中位/最高°C','GPU利用率中位/95分位%','GPU频率中位MHz','传感器功率中位W','最低可用内存GiB'] if zh else ['Node','Temperature median/max°C','GPU utilization median/p95%','Median GPU clock MHz','Median sensor power W','Minimum available memory GiB'],resource_rows)+f"\n![{'节点资源时间序列' if zh else 'Node resource time series'}](completion_resources.{lang}.svg)\n\n"+(
        "约每5秒采样一次；统计包含预热、负载、区组间排空与环境核验，因此利用率中位数不能代表饱和吞吐。超过20秒的采样间断不连线。85°C为实验退避阈值，不能据此推断驱动已经发出硬件降频信号；传感器功率不等于整机插座能耗。\n" if zh else
        "Samples are taken approximately every 5 seconds, including warmup, work, inter-block draining and environment verification. Median utilization therefore does not represent saturated throughput. Gaps longer than 20 seconds are not connected. 85°C is the experiment's backoff threshold, not evidence of a driver throttling signal. Sensor power is not whole-system wall energy.\n"))
    f['cost']=(("### 2026-10-19 补测实付API费用\n\n" if zh else "### Actual API charges for the 2026-10-19 supplement\n\n")+table(
        ['项目','实测／核对值'] if zh else ['Item','Observed/reconciled value'],[
            [('本补测批次已归档尝试的费用' if summary['billing']['complete'] else '已核对尝试的费用小计') if zh else ('Charges for archived attempts in this supplement' if summary['billing']['complete'] else 'Subtotal for attributed attempts'),f"**US${summary['cost_usd_all_attempts']:.6f}**"],
            ['已发物理尝试数' if zh else 'Submitted physical attempts',str(c['all_attempts_started'])],
            ['账单归属覆盖' if zh else 'Billing attribution coverage',f"{summary['billing']['attributed_attempts']}/{summary['billing']['started_attempts']}"],
            ['费用口径' if zh else 'Accounting boundary','包含预检、预热和保留尝试；按源站关联账单核对。' if zh else 'Includes preflight, warmup and retained attempts; reconciled with source-linked billing.']])+(
        "补测费用不是生产单价估计。前文1／3／5年持有成本与回报情景保持各自的负载、采购和利用率假设，不把本日期不同材料的吞吐混入旧情景。\n" if zh else
        "These experiment charges are not a production unit-price estimate. Earlier 1/3/5-year ownership and return scenarios retain their workload, capital and utilization assumptions; throughput from this different workload is not substituted.\n"))
    f['conclusion']=(("### 新证据、适用条件与后续工作\n\n" if zh else "### New evidence, applicability and next work\n\n")+table(
        ['维度','本次可支持的结论','后续工作'] if zh else ['Dimension','Supported conclusion','Next work'],[
            ['公网长输出' if zh else 'Public long output',f"{c['length_conditions_complete']}/8个上限条件已有响应证据；实际生成量单列。" if zh else f"{c['length_conditions_complete']}/8 output-cap conditions have response evidence; actual lengths are separate.",'如需固定长度公网容量，先验证强制生成参数的端到端透传。' if zh else 'Verify end-to-end fixed-generation parameter forwarding before a public fixed-length capacity study.'],
            ['混合服务' if zh else 'Mixed serving',f"{c['interference_pairs_qualified']}/5完整配对；中位等待比值{ratio_text}，95分位比值{tail_text}。" if zh else f"{c['interference_pairs_qualified']}/5 complete pairs; median-wait ratios {ratio_text}, p95 ratios {tail_text}.",'用同一到达序列对照长短请求分队或预填充预算，验证尾部等待能否改善；再扩展到达率与日期。' if zh else 'Use the same traces to compare separate short/long queues or prefill budgets, testing tail-latency improvement before extending rates and dates.'],
            ['前缀缓存' if zh else 'Prefix caching',f"{c['cache_pairs_qualified']}/3个64K低复用对照可归属。" if zh else f"{c['cache_pairs_qualified']}/3 attributable 64K low-reuse controls.",'结合既有复用／保留实验解释；增加跨时段竞争，不能由低复用对照推导TTL。' if zh else 'Interpret alongside reuse/retention tests; add cross-period competition rather than inferring a TTL from low-reuse controls.'],
            ['硬件对照与经济性' if zh else 'Hardware comparison and economics','GB10与H200按资源功能对照；硬件实测与规格、成本情景分列。' if zh else 'GB10/H200 resources are functionally aligned; measurements, specifications and cost scenarios remain distinct.','需要H200同负载、插座电表和真实利用率，才能给出实测推理性价比。' if zh else 'Same-workload H200 data, a wall meter and real utilization are needed for measured inference price/performance.']])+(
        "各项固定实验是否完成以第05节独立分母为准。物理DRAM字节计数、严格逐词元时间戳及H200同负载仍无实测证据，这些限制不由增加普通请求数解决。\n" if zh else
        "Completion is defined by the separate fixed denominators in section05. Physical DRAM-byte counts, exact token timestamps and a same-workload H200 comparison remain unmeasured; more ordinary requests do not resolve those instrument/resource limits.\n"))
    f['appendix']=(("### 2026-10-19：新增实验的代码与数据\n\n" if zh else "### 2026-10-19: additional experiment code and data\n\n")+table(
        ['路径','用途'] if zh else ['Path','Purpose'],[
            ['`../protocols/environment.json`','逐节点硬件型号、驱动、SDK版本及脱敏的完整启动参数。' if zh else 'Per-node hardware model, driver, SDK versions and complete redacted launch arguments.'],
            ['`../protocols/public/manifest.json` / `../protocols/cache/manifest.json`','冻结条件、完整到达偏移、数量与SHA256。' if zh else 'Frozen conditions, full arrival offsets, counts and SHA256.'],
            ['`../protocols/*/plans/*.jsonl`','逐请求输入材料和参数；保留原始语言。' if zh else 'Request inputs and parameters in their original language.'],
            ['`../data/requests.json`','逐尝试标量观测：状态、计时、用量、费用、输入哈希及纳入标记。' if zh else 'Attempt-level states, timing, usage, charges, prompt hashes and inclusion flags.'],
            ['`../data/arrivals.json`','全部到达事件与准入终态，包括未转发的本地拒绝。' if zh else 'All arrival and admission events, including local rejections that were never forwarded.'],
            ['`../data/blocks.json`','区组、独立尝试及前后源站计数器。' if zh else 'Blocks, independent attempts and boundary source counters.'],
            ['`../data/telemetry.json`','双节点温度、利用率、频率、传感器功率和可用内存。' if zh else 'Two-node temperature, utilization, clocks, sensor power and available memory.'],
            ['`../analysis/completion-summary.json`','可复算统计、配对资格和计数器校验原因。' if zh else 'Reproducible statistics, pair qualification and counter-validation reasons.'],
            ['`../analysis/build_report.py` / `../../../../src/sparkbench/completion.py`','离线报告重建与分析实现。' if zh else 'Offline report builder and analysis implementation.']])+(
        "私有原始测量归档：`workspace/export/qwen38_perf_p3_20261014_0905/evidence_completion_20261019/`；包含原始SSE、生成正文、逐请求回执及源站读数。GitHub发布冻结输入、标量证据和复算代码，不分发凭证、生产控制器或业务身份。\n" if zh else
        "Private raw archive: `workspace/export/qwen38_perf_p3_20261014_0905/evidence_completion_20261019/`, containing SSE, generated text, receipts and source snapshots. The GitHub distribution includes frozen inputs, scalar evidence and offline analysis, excluding credentials, production controllers and business identities.\n"))
    return f


def replace_md_section(text, title, replacement):
    pattern = r'(?m)^### ' + re.escape(title) + r'\n.*?(?=^### |^## |\Z)'
    return re.sub(pattern, lambda _: replacement+'\n', text, count=1, flags=re.S | re.M)


def insert_md(text, section, snippet, before=False):
    match=re.search(r'(?m)^## '+section+r' · .+$',text)
    if not match: raise ValueError('Missing section '+section)
    end=re.search(r'(?m)^## ',text[match.end():])
    index=match.end() if before else match.end()+end.start() if end else len(text)
    if not before and end:
        anchor=text.rfind('<a id="section-',match.end(),index)
        if anchor>=0 and re.fullmatch(r'<a id="section-\d+"></a>\s*',text[anchor:index]):
            index=anchor
    return text[:index]+'\n\n'+snippet+'\n\n'+text[index:]


def decorate_snippet(snippet, lang, out, slug):
    fragment=BeautifulSoup(markdown.markdown(snippet,extensions=['tables','fenced_code']), 'html.parser')
    for strong in fragment.select('strong'):strong['class']=['metric-value']
    for i,t in enumerate(list(fragment.select('table')),1):
        t['class']=['metric-table','result-table'];wrap=fragment.new_tag('div');wrap['class']=['table-wrap'];wrap['tabindex']='0';t.wrap(wrap)
        heading=t.find_previous(['h3','h2'])
        caption=fragment.new_tag('caption');caption['id']=f'table-caption-{slug}-{i}'
        label=fragment.new_tag('span');label['class']=['visual-label'];label.string='2026-10-19'
        title=fragment.new_tag('span');title['class']=['visual-title'];title.string=re.sub(r'^2026-10-19[:： ]*','',heading.get_text()) if heading else ('补测结果' if lang=='zh' else 'Supplementary results')
        caption.extend([label,title]);t.insert(0,caption)
        wrap['role']='region';wrap['aria-labelledby']=caption['id']
        for th in t.select('th'):th['scope']='col'
    for image in list(fragment.select('img')):
        path=out/image['src'];image['src']='data:image/svg+xml;base64,'+base64.b64encode(path.read_bytes()).decode();image['class']=['static-chart']
        figure=fragment.new_tag('figure');figure['class']=['figure'];figure['id']='figure-'+slug
        viewport=fragment.new_tag('div');viewport['class']=['chart-viewport'];viewport['tabindex']='0'
        image.wrap(viewport);viewport.wrap(figure)
        caption=fragment.new_tag('figcaption')
        label=fragment.new_tag('span');label['class']=['visual-label'];label.string='2026-10-19'
        title=fragment.new_tag('span');title['class']=['visual-title'];title.string=image.get('alt','')
        caption.extend([label,title]);figure.append(caption)
    return fragment


def put_html(soup, section, snippet, lang, out, slug, start=False):
    h=soup.find(id='section-'+section)
    fragment=decorate_snippet(snippet,lang,out,slug)
    if start:
        for node in reversed(list(fragment.contents)):h.insert_after(node)
    else:
        following=h.find_next('h2')
        if following:
            for node in list(fragment.contents):following.insert_before(node)
        else:
            for node in list(fragment.contents):h.parent.append(node)


def replace_html_subsection(soup, title, snippet, lang, out, slug):
    h=next((x for x in soup.find_all('h3') if x.get_text()==title),None)
    if h is None:raise ValueError('Missing heading '+title)
    following=h.next_sibling
    while following and getattr(following,'name',None) not in ('h2','h3'):
        current=following;following=current.next_sibling;current.extract()
    for node in list(decorate_snippet(snippet,lang,out,slug).contents):h.insert_before(node)
    h.extract()


def render(experiment, pdf=False):
    experiment=Path(experiment).resolve();base=experiment.parent/'2026-10-14_17-dual-spark'
    if not base.exists():
        # Private previews can name the immutable repository explicitly.
        base=Path(os.environ['SPARK_BASE_EXPERIMENT'])
    out=experiment/'reports';out.mkdir(parents=True,exist_ok=True)
    summary=write_summary(experiment,experiment/'analysis')
    telemetry=[r for r in json.loads((experiment/'data/telemetry.json').read_text()) if r.get('in_measurement_window',True)]
    c=summary['coverage']
    timestamp=max([r['epoch'] for r in telemetry]+[0]);stamp=datetime.fromtimestamp(timestamp,timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    for lang in ['zh','en']:
        zh=lang=='zh';charts(summary,telemetry,out,lang);fragments=sections(summary,lang)
        md=(base/f'reports/report.{lang}.md').read_text()
        soup=BeautifulSoup((base/f'reports/report.{lang}.html').read_text(),'html.parser')
        env_section=environment_section(json.loads((experiment/'protocols/environment.json').read_text()),lang)
        overview_title='测试领域、测试项与主要结论' if zh else 'Test domains, items and principal findings'
        overview=next((h for h in soup.find_all('h3') if h.get_text()==overview_title),None)
        if overview is None:
            # The overview is the first subsection of the study introduction.
            overview=soup.find(id='section-01').find_next('h3')
            overview_title=overview.get_text()
        for node in list(decorate_snippet(env_section,lang,out,'environment').contents):overview.insert_before(node)
        md=md.replace('### '+overview_title,env_section+'\n### '+overview_title,1)
        path_copy={
            '完整启动命令与解析后的参数见附录C。': '当前环境与完整启动命令见第1节；附录C保留部署参数的详细核验记录。',
            'The full launch command and parsed parameters are in Appendix C.': 'The current environment and complete launch command appear in Section 1; Appendix C retains the detailed deployment-parameter verification.',
            '以下为本机绝对磁盘路径。目录索引覆盖本轮测试、模型与题集来源及方案资料；逐文件清单记录每个文件的路径、大小、修改时间，以及源码文件的 SHA-256。':
            '以下保留原始测量工作区的路径索引。workspace/表示执行实验的工作区根目录，export/从该根目录展开；这些路径不表示GitHub包含全部原始文件。可下载的源码与数据以仓库的reference/、src/和experiments/为准。逐文件清单记录原始文件的路径、大小、修改时间及源码SHA-256。',
            'All paths below are absolute local disk paths. The index covers this experiment, model and dataset sources, and protocol materials. The complete inventory records each file’s path, size, modification time and, for source code, SHA-256.':
            'This preserves the original measurement-workspace index. workspace/ denotes the execution workspace root, and export/ is relative to that root; these paths do not mean every raw file is distributed on GitHub. Downloadable code and data are under reference/, src/ and experiments/ in the repository. The original inventory records paths, sizes, modification times and source-code SHA256 hashes.',
            '绝对磁盘路径':'原始工作区路径','磁盘绝对路径':'原始工作区路径',
            'Absolute disk path':'Original workspace path',
        }
        for before,after in path_copy.items():
            md=md.replace(before,after)
            for node in list(soup.find_all(string=lambda value:value and before in value)):
                node.replace_with(str(node).replace(before,after))
        # The representative two-node profiles supersede the early access probe.
        # The original dated archive retains that diagnostic without repeating
        # its obsolete availability conclusions in the current report.
        probe_header='| 节点 | 硬件计数器状态 |' if zh else '| Node | Hardware counter status |'
        probe_end='[本批逐条数据、条件汇总与保护状态]' if zh else '[Per-record data, condition summaries and guard states]'
        if probe_header in md:
            begin=md.index(probe_header);end=md.index(probe_end,begin)
            md=md[:begin]+md[end:]
        for t in list(soup.find_all('table')):
            headers=[th.get_text(strip=True) for th in t.find_all('th')]
            if any(label in headers for label in ['硬件计数器状态','诊断计数器','Hardware counter status','Diagnostic counter']):
                (t.parent if 'table-wrap' in t.parent.get('class',[]) else t).extract()
        for p in list(soup.find_all('p')):
            if p.get_text().startswith(('上述可用数值来自服务恢复','Available values come from one successful profile')):
                p.extract()
        old_counter=('计数器访问只作诊断，不改变系统权限；未取得的缓存命中率和活跃线程计数保留缺失。' if zh else
                     'Counter access is diagnostic without system permission changes; unavailable cache-hit and active-warp counters remain missing.')
        new_counter=('硬件计数器按后文的代表性核函数单独统计；物理DRAM字节字段不可用，不以零值替代。' if zh else
                     'Hardware counters are reported separately for the representative kernels below. Physical DRAM-byte fields are unavailable and are not replaced with zeros.')
        md=md.replace(old_counter,new_counter)
        for node in list(soup.find_all(string=lambda value:value and old_counter in value)):
            node.replace_with(str(node).replace(old_counter,new_counter))
        verification=json.loads((experiment/'protocols/configuration-verification.json').read_text())
        versions=verification['0']['versions']
        environment_text=f"PyTorch {versions['torch']} · Transformers {versions['transformers']}"
        for endpoint_table in soup.select('table'):
            if 'POST' in endpoint_table.get_text() and 'SGLang' in endpoint_table.get_text() and '/chat/completions' in endpoint_table.get_text():
                for label,value in [('软件环境' if zh else 'Software environment',environment_text),
                                    ('本次配置核验' if zh else 'Configuration verification','2026-10-19 · '+('双节点版本与配置／分词器文件散列一致' if zh else 'Matched node versions and configuration/tokenizer hashes'))]:
                    row=soup.new_tag('tr')
                    for cell in [label,value]:
                        td=soup.new_tag('td');td.string=cell;row.append(td)
                    endpoint_table.find('tbody').append(row)
                break
        lines=[];in_object_section=False;environment_inserted=False
        for line in md.splitlines():
            if line.startswith('## '):in_object_section=line.startswith('## 02 · ')
            lines.append(line)
            if in_object_section and not environment_inserted and line.startswith('|') and 'SGLang' in line and '0.5.21' in line:
                lines.append('| '+('软件环境' if zh else 'Software environment')+' | '+environment_text+' |')
                lines.append('| '+('本次配置核验' if zh else 'Configuration verification')+' | 2026-10-19 · '+('双节点版本与配置／分词器文件散列一致' if zh else 'Matched node versions and configuration/tokenizer hashes')+' |')
                environment_inserted=True
        md='\n'.join(lines)+'\n'
        old_length_title='扩展长度：覆盖、实际生成与重复性' if zh else 'Expanded lengths: coverage, realized output and repeatability'
        dated_length_title='2026-10-14–17 长度批次：覆盖与实际生成' if zh else 'October 14–17 length batches: coverage and realized generation'
        heading=next((h for h in soup.find_all('h3') if h.get_text()==old_length_title),None)
        if heading:heading.string=dated_length_title
        md=md.replace('### '+old_length_title,'### '+dated_length_title)
        # Remove obsolete undated coverage conclusions; retain original data rows.
        prefixes=['覆盖边界：','Coverage boundary:','后续优先补齐公网长度的未覆盖组合','Priority is to complete the uncovered public-length']
        for p in list(soup.find_all('p')):
            if any(p.get_text().startswith(prefix) for prefix in prefixes):p.extract()
        md='\n'.join(line for line in md.splitlines() if not any(line.startswith(prefix) for prefix in prefixes))+'\n'
        # Align summary claims with native and public measurement boundaries.
        for table_index, label in [(0,'输入／输出与业务形态' if zh else 'Lengths and business shapes'),(-1,'长输入与长输出' if zh else 'Long input and output')]:
            replacement=(f'原生固定输出最高 **260,096词元**；10月19日公网新增 **{c["length_conditions_complete"]}/8** 个条件。公网实际生成量按自然终止单列。' if zh else
                         f'Native fixed output reaches **260,096 tokens**; the October 19 supplement adds **{c["length_conditions_complete"]}/8** public conditions, with natural-stop lengths reported separately.')
            for tr in soup.select('tr'):
                cells=tr.find_all('td',recursive=False)
                if cells and cells[0].get_text().strip()==label:
                    cells[-2 if table_index==0 else 1].clear()
                    target=cells[-2 if table_index==0 else 1]
                    # The overview has three cells: the result belongs in its last cell.
                    if table_index==0:
                        target=cells[-1];target.clear()
                        cells[1].string='输入／输出长度、业务比例与实际生成量。' if zh else 'Input/output lengths, business ratios and realized generation.'
                    target.append(BeautifulSoup(markdown.markdown(replacement),'html.parser'))
                    if table_index==-1:cells[-1].string='原生固定生成与公网自然终止分列。' if zh else 'Native fixed generation and public natural stopping are separate.'
            lines=[]
            for line in md.splitlines():
                if line.startswith('|') and label in line and ('section-08' in line or line.startswith('| '+label+' |')):
                    cells=line.split('|')
                    if table_index==0:cells[-2]=replacement
                    else:cells[2]=replacement;cells[3]='原生与公网分列。' if zh else 'Native and public results remain separate.'
                    line='|'.join(cells)
                lines.append(line)
            md='\n'.join(lines)+'\n'
        # Clearly identify the dates of retained measurement sections.
        note=('10月14–17日的服务、硬件和引擎数据保留各自日期与分母。以下标注“2026-10-19”的结果为新独立实验；它们不改写历史成功率，也不与旧容量数据合并。' if zh else
              'Service, hardware and engine observations from October 14–17 retain their own dates and denominators. Results labeled 2026-10-19 are independent measurements; they do not rewrite historical success rates or pool with earlier capacity estimates.')
        put_html(soup,'08',note,lang,out,'date-scope',start=True);md=insert_md(md,'08',note,True)
        length_anchor='固定长输出：原生引擎的可达范围' if zh else 'Fixed long output: demonstrated native-engine range'
        heading=next(h for h in soup.find_all('h3') if h.get_text()==length_anchor)
        for node in list(decorate_snippet(fragments['length'],lang,out,'completion-length').contents):heading.insert_before(node)
        md=md.replace('### '+length_anchor,fragments['length']+'\n\n### '+length_anchor,1)
        for key,sec in [('summary','01'),('protocol','03'),('metrics','04'),('coverage','05'),('cache','08'),('resources','09'),('cost','11'),('appendix','15')]:
            put_html(soup,sec,fragments[key],lang,out,'completion-'+key,start=key=='coverage')
            md=insert_md(md,sec,fragments[key],before=key=='coverage')
        oldtitle='长请求对短请求的等待影响' if zh else 'Effect of long requests on short-request waiting'
        replace_html_subsection(soup,oldtitle,fragments['interference'],lang,out,'completion-interference')
        md=replace_md_section(md,oldtitle,fragments['interference'])
        oldtitle='最终覆盖与后续补测优先项' if zh else 'Final coverage and priorities for further measurement'
        replace_html_subsection(soup,oldtitle,fragments['conclusion'],lang,out,'completion-conclusion')
        md=replace_md_section(md,oldtitle,fragments['conclusion'])
        # Current coverage graphic; the earlier count table remains historical.
        figure=soup.find(id='figure-workflow')
        if figure:
            figure.find('img')['src']='data:image/svg+xml;base64,'+base64.b64encode((out/f'completion_coverage.{lang}.svg').read_bytes()).decode()
            figure.find('figcaption').clear();figure.find('figcaption').string=('固定清单的有效证据覆盖：各行日期、单位和分母独立。' if zh else 'Qualified evidence by fixed list: each row has a separate date, unit and denominator.')
        md=md.replace(f'](workflow.{lang}.svg)',f'](completion_coverage.{lang}.svg)')
        md=re.sub(r'(?m)^图 05-01.*$', '图05-01｜各固定清单的有效证据覆盖；各行分母独立。',md) if zh else re.sub(r'(?m)^Figure 05-01.*$', 'Figure05-01 | Qualified evidence by fixed list, with separate denominators.',md)
        h=soup.find(id='section-05');intro=next((p for p in h.find_next_siblings('p') if p.get_text().startswith(('本轮执行已结束','Execution has ended','Execution is complete'))),None)
        if intro:intro.string=note
        md=md.replace('本轮执行已结束，证据覆盖仍有明确缺口。', '固定清单按日期独立验收，原始缺失观测不回填。').replace('Execution has ended, with explicit evidence gaps.', 'Fixed lists are qualified by date; historical missing observations are not backfilled.')
        current_title='未执行与结论受限的项目' if zh else 'Unexecuted items and limits on conclusions'
        limits=('### 当前证据与结论边界\n\n' if zh else '### Current evidence and interpretation limits\n\n')+table(
            ['范围','当前证据','边界'] if zh else ['Scope','Current evidence','Boundary'],[
                ['公网长度' if zh else 'Public lengths',f"2026-10-19: {c['length_conditions_complete']}/8",'原299条历史清单保留；新日期独立证据。' if zh else 'Original 299-row outcomes retained; new date analyzed separately.'],
                ['混合干扰' if zh else 'Interference',(f"{c['interference_pairs_observed']}/5已观测；{c['interference_pairs_qualified']}/5可估计效应" if zh else f"{c['interference_pairs_observed']}/5 observed; {c['interference_pairs_qualified']}/5 effect-eligible"),'只对完整配对给出效应比值。' if zh else 'Effect ratios only for complete pairs.'],
                ['64K低复用窗口' if zh else '64K low-reuse windows',f"2026-10-19: {c['cache_pairs_qualified']}/3",'实际命中依赖计数器，不能只根据设计比例推断。' if zh else 'Actual reuse requires counters, not the design ratio alone.'],
                ['连续公网服务' if zh else 'Continuous public service','3,600.1s; 599/600','10月17日单独一小时；不外推全天可用性。' if zh else 'Separate October 17 hour; not all-day availability.'],
                ['H200、整机电费和物理计数器' if zh else 'H200, wall energy and physical counters','未实测' if zh else 'Not measured','缺少设备或仪器接口；保留官方规格与测量边界。' if zh else 'Equipment/instrument limits; retain specifications and measurement boundaries.']])
        replace_html_subsection(soup,current_title,limits,lang,out,'completion-limits');md=replace_md_section(md,current_title,limits)
        # Latest observation timestamp; no claim that unfinished previews are final.
        finished=c['logical_outcomes_observed']==695 and c['interference_pairs_observed']==5 and c['cache_pairs_qualified']==3
        if soup.select_one('.hero .status'):soup.select_one('.hero .status').string=('独立补测已完成' if zh else 'Independent supplement complete') if finished else ('测量进行中 · 未冻结预览' if zh else 'Measurements in progress · unfrozen preview')
        if soup.select_one('.hero .caption'):soup.select_one('.hero .caption').string=('数据截止：' if zh else 'Observations through: ')+stamp
        md=re.sub(r'(?m)^(数据快照（UTC）：|Data snapshot \(UTC\):).*$', ('数据快照（UTC）：' if zh else 'Data snapshot (UTC): ')+stamp,md)
        # Keep Markdown figures portable while reusing the immutable prior archive.
        def image_path(match):
            target=match.group(2)
            if target.startswith(('http:','https:','data:')) or (out/target).exists():return match.group(0)
            original=base/'reports'/target
            return f'![{match.group(1)}]({os.path.relpath(original,out)})' if original.exists() else match.group(0)
        md=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',image_path,md)
        # Non-image historical links also resolve relative to their original report.
        def link_path(match):
            target=match.group(2)
            if target.startswith(('#','http:','https:','data:')) or (out/target).exists():return match.group(0)
            original=(base/'reports'/target).resolve()
            return f'[{match.group(1)}]({os.path.relpath(original,out)})' if original.exists() else match.group(0)
        md=re.sub(r'(?<!!)\[([^\]]*)\]\(([^)]+)\)',link_path,md)
        style=soup.new_tag('style');style.string='.metric-value{color:#004AB9;font-weight:700}.table-wrap{max-width:100%;overflow-x:auto}.figure img{max-width:100%;height:auto} @media print{.table-wrap{overflow:visible}.figure{break-inside:avoid}table{font-size:9px}}';soup.head.append(style)
        for strong in soup.select('main strong'):strong['class']=list(set(strong.get('class',[])+['metric-value']))
        (out/f'report.{lang}.html').write_text(str(soup));(out/f'report.{lang}.md').write_text(md.rstrip()+'\n')
        if pdf:
            from weasyprint import HTML
            HTML(filename=str(out/f'report.{lang}.html')).write_pdf(str(out/f'report.{lang}.pdf'))
        print(json.dumps({'language':lang,'tables':len(soup.select('table')),'figures':len(soup.select('figure')),'final_measurement_coverage':finished}),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);parser.add_argument('--pdf',action='store_true');args=parser.parse_args()
    render(args.experiment,args.pdf)
