"""Present the frozen measurements as one thematic, bilingual study.

Only report presentation changes here. Request records, counter windows and
protocol manifests keep their original identities and denominators.
"""
import base64
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

from bs4 import BeautifulSoup
import markdown


def table(headers, rows):
    return '\n| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + ''.join('| ' + ' | '.join(map(str, row)) + ' |\n' for row in rows) + '\n'


class Document:
    def __init__(self, md, soup, out, lang):
        self.md, self.soup, self.out, self.lang = md, soup, Path(out), lang

    def fragment(self, text, slug):
        part = BeautifulSoup(markdown.markdown(text, extensions=['tables', 'fenced_code']), 'html.parser')
        for i, tab in enumerate(part.find_all('table'), 1):
            tab['class'] = ['metric-table', 'result-table']
            wrap = part.new_tag('div', attrs={'class': 'table-wrap', 'tabindex': '0', 'role': 'region'})
            tab.wrap(wrap)
            cap = part.new_tag('caption', id=f'table-caption-{slug}-{i}')
            heading = tab.find_previous(['h3', 'h2'])
            cap.string = heading.get_text() if heading else ('实验结果' if self.lang == 'zh' else 'Experimental results')
            tab.insert(0, cap); wrap['aria-labelledby'] = cap['id']
            for cell in tab.find_all('th'): cell['scope'] = 'col'
        for img in part.find_all('img'):
            path = self.out / img['src']
            img['src'] = 'data:image/svg+xml;base64,' + base64.b64encode(path.read_bytes()).decode()
            img['class'] = ['static-chart']
            identifier = 'figure-workflow' if 'coverage' in path.name else 'figure-' + slug
            fig = part.new_tag('figure', attrs={'class': 'figure', 'id': identifier})
            view = part.new_tag('div', attrs={'class': 'chart-viewport', 'tabindex': '0'})
            img.wrap(view); view.wrap(fig)
            cap = part.new_tag('figcaption'); cap.string = img.get('alt', ''); fig.append(cap)
        for item in part.select('strong'): item['class'] = ['metric-value']
        return list(part.contents)

    def subsection(self, title):
        heading = next((h for h in self.soup.find_all('h3') if h.get_text(strip=True) == title), None)
        if heading is None: raise ValueError('Missing subsection: ' + title)
        nodes = [heading]; node = heading.next_sibling
        while node is not None and getattr(node, 'name', None) not in ('h2', 'h3'):
            nodes.append(node); node = node.next_sibling
        pattern = r'(?ms)^### ' + re.escape(title) + r'\n.*?(?=^### |^## |^<a id="section-|\Z)'
        found = re.search(pattern, self.md)
        if not found: raise ValueError('Missing Markdown subsection: ' + title)
        self.md = self.md[:found.start()] + self.md[found.end():]
        for node in nodes: node.extract()
        return found.group(0), nodes

    def drop(self, title):
        self.subsection(title)

    def insert_before(self, title, block):
        text, nodes = block
        heading = next(h for h in self.soup.find_all('h3') if h.get_text(strip=True) == title)
        for node in nodes: heading.insert_before(node)
        self.md = self.md.replace('### ' + title + '\n', text.rstrip() + '\n\n### ' + title + '\n', 1)

    def append_section(self, section, block):
        text, nodes = block
        heading = self.soup.find(id='section-' + section)
        following = heading.find_next('h2')
        for node in nodes:
            if following: following.insert_before(node)
            else: heading.parent.append(node)
        match = re.search(r'(?m)^## ' + section + r' · .*$', self.md)
        end = re.search(r'(?m)^<a id="section-|^## ', self.md[match.end():])
        index = match.end() + end.start() if end else len(self.md)
        self.md = self.md[:index] + '\n' + text.rstrip() + '\n\n' + self.md[index:]

    def replace_section(self, section, text):
        heading = self.soup.find(id='section-' + section)
        node = heading.next_sibling
        while node is not None and getattr(node, 'name', None) != 'h2':
            following = node.next_sibling; node.extract(); node = following
        fragment = self.fragment(text, 'study-' + section)
        for item in reversed(fragment): heading.insert_after(item)
        match = re.search(r'(?m)^## ' + section + r' · .*$', self.md)
        end = re.search(r'(?m)^<a id="section-|^## ', self.md[match.end():])
        index = match.end() + end.start() if end else len(self.md)
        self.md = self.md[:match.end()] + '\n\n' + text.rstrip() + '\n\n' + self.md[index:]

    def rename(self, before, after):
        self.md = self.md.replace(before, after)
        for node in list(self.soup.find_all(string=lambda value: value and before in value)):
            if node.parent.name not in ('script', 'style'):
                node.replace_with(str(node).replace(before, after))

    def paragraph(self, prefix, replacement):
        candidates = [p for p in self.soup.find_all('p') if p.get_text().startswith(prefix.replace('**', ''))]
        for p in candidates:
            if replacement:
                for node in self.fragment(replacement, 'paragraph'): p.insert_before(node)
            p.extract()
        lines = []
        for line in self.md.splitlines():
            if line.replace('**', '').startswith(prefix.replace('**', '')):
                if replacement: lines.append(replacement)
            else: lines.append(line)
        self.md = '\n'.join(lines) + '\n'

    def replace_cell(self, label, column, value):
        for row in self.soup.select('tr'):
            cells = row.find_all('td', recursive=False)
            if cells and cells[0].get_text(strip=True) == label:
                cells[column].clear()
                fragment = BeautifulSoup(markdown.markdown(value), 'html.parser')
                for node in list(fragment.p.contents): cells[column].append(node)
        lines = []
        for line in self.md.splitlines():
            if line.startswith('|'):
                cells = line.split('|')
                first = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', cells[1]).strip()
                if first == label: cells[column + 1] = ' ' + value + ' '; line = '|'.join(cells)
            lines.append(line)
        self.md = '\n'.join(lines) + '\n'

    def add_overview_row(self, after_label, row):
        overview = next(t for t in self.soup.find_all('table') if t.find('a', href='#section-06') and t.find('a', href='#section-11'))
        fragment = BeautifulSoup(markdown.markdown(table(['a','b','c'], [row]), extensions=['tables']), 'html.parser')
        newrow = fragment.find('tbody').find('tr').extract()
        oldrow = next(tr for tr in overview.find_all('tr') if tr.find('td') and tr.find('td').get_text(strip=True) == after_label)
        oldrow.insert_after(newrow)
        lines = []
        for line in self.md.splitlines():
            lines.append(line)
            if line.startswith('| [') and '](' in line and after_label in line:
                lines.append('| ' + ' | '.join(row) + ' |')
        self.md = '\n'.join(lines) + '\n'


def coverage_text(summary, base, lang):
    zh = lang == 'zh'; c = summary['coverage']
    archived = json.loads((base / 'data/coverage.json').read_text())
    rows = []
    for record in archived['fixed_coverage']:
        rows.append([record['zh' if zh else 'en'], f"**{record['valid']:,}/{record['planned']:,}**",
                     '固定条件、固定轮次' if zh else 'Fixed conditions and repetitions'])
    rows += [
        ['公网大输出预算' if zh else 'Public large-output budgets', f"**{c['length_conditions_complete']}/8**", '各条件1条自然结束响应；输出上限与实际长度分列。' if zh else 'One natural-stop response per condition; cap and realized length reported separately.'],
        ['完整到达序列的混合负载' if zh else 'Mixed workload with complete arrival traces', '**5/5**', '**4/5**可估计完整配对效应；第5组混合臂**3/32**短请求本地拒绝。' if zh else '**4/5** support complete-pair effects; **3/32** short arrivals in mixed arm 5 are locally rejected.'],
        ['64K低复用前缀' if zh else '64K low-reuse prefixes', '**3/3**', '每对2条；6个源站计数器窗口全部可归属。' if zh else 'Two requests per pair; all six source-counter windows attributable.'],
    ]
    text = ('实验覆盖按测量单位分别报告：硬件轮次、通信试验、请求、配对和连续窗口具有不同分母。完成响应、达到输出上限和满足服务目标是不同结果。各组保留实际采样日期、配置和排除记录。\n\n### 固定条件与有效观测\n' if zh else
            'Coverage is reported by measurement unit: hardware rounds, collective trials, requests, pairs and continuous windows have different denominators. Response completion, output-length attainment and service-level attainment are distinct outcomes. Each cohort retains its sampling date, configuration and exclusions.\n\n### Fixed conditions and valid observations\n')
    text += table(['测试项','有效观测／计划','统计口径'] if zh else ['Test','Valid / planned','Interpretation'], rows)
    text += f"\n![{'分层实验覆盖与效应估计资格' if zh else 'Layered coverage and effect-estimation eligibility'}](completion_coverage.{lang}.svg)\n\n"
    text += ('### 公网请求分组与结果\n\n表中区分固定样本清单与条件调用上限；后者是停止约束，不是必须用满的请求数量。\n' if zh else
             '### Public request cohorts and outcomes\n\nFixed sample lists differ from conditional call ceilings; a ceiling is a stopping bound, not a mandatory sample count.\n')
    names = [('接口校准','API calibration'),('短题库与并发','Short-bank concurrency'),('输入／输出矩阵','Input/output matrix'),('思考预算配对','Reasoning-budget pairs'),('上下文与边界','Context and boundaries'),('前缀复用','Prefix reuse'),('业务与缓存','Business and cache'),('开放到达与混合负载','Open arrivals and mixtures'),('长度与持续窗口','Lengths and sustained windows')]
    rows = []
    for i, phase in enumerate(archived['initial_phases']):
        rows.append([names[i][0 if zh else 1], f"{phase['cap']:,} " + (('固定' if i < 7 else '上限') if zh else ('fixed' if i < 7 else 'ceiling')),
                     f"{phase['logical_sent']:,}", f"{phase['complete']:,}", f"{phase['logical_sent']-phase['complete']:,}"])
    text += table(['请求分组','设计数量','已发送逻辑请求','完整响应','未完整响应'] if zh else ['Cohort','Design count','Logical requests sent','Complete','Incomplete'], rows)
    text += ('上表来自10月14–17日的服务矩阵。另有独立并发参考、业务长度、混合调度与一小时窗口的**1,028条**完整公网响应，按其自身条件分析。原生引擎的**231条**正式请求与公网请求分开计量。\n\n' if zh else
             'The table describes the October 14–17 service matrix. A further **1,028 complete public responses** belong to separately defined concurrency references, business lengths, mixed scheduling and the one-hour window. The **231 formal native-engine requests** are measured separately from public requests.\n\n')
    text += ('10月19日受控长度、完整混合序列和低复用前缀实验共**695个逻辑请求**：**692条**完整源站响应和**3条**本地拒绝。其构成为1条校准、8条长度观测、320条预热、320条正式短请求、40条混合长请求及6条缓存请求。另归档4条入口隔离前的探针；**696次**物理已发尝试均有账单归属。\n\n' if zh else
             'Controlled lengths, complete mixed traces and low-reuse prefixes on October 19 comprise **695 logical outcomes**: **692 complete source responses** and **3 local rejections**. They include one calibration, eight length observations, 320 warmups, 320 formal short arrivals, 40 long arrivals and six cache requests. Four probes before ingress isolation are separately archived; billing is attributable for all **696 submitted physical attempts**.\n\n')
    text += ('### 排除记录与结论边界\n' if zh else '### Exclusions and limits of inference\n')
    exclusions = [
        ['长度清单' if zh else 'Length list', '299项中290条完整、1条不完整、8条未发送；32K/8K条件另有4条完整匹配观测，大输出预算组覆盖8个条件。' if zh else 'Of 299 entries, 290 completed, one was incomplete and eight were unsent; 32K/8K has four complete matched observations, and the large-budget cohort covers eight conditions.', '按条件索引可获得响应证据；不重写单次请求状态，不把输出预算当实际长度。' if zh else 'Response evidence is indexed by condition; request outcomes are unchanged and caps are not realized lengths.'],
        ['截断到达序列' if zh else 'Truncated arrival traces', '5对候选中3对执行、2对未进入；无完整配对效应。' if zh else 'Three of five candidate pairs executed; two were not admitted. No complete-pair effect estimate.', '不与完整到达序列的5对混合负载合并估计。' if zh else 'Excluded from effect estimates for the five complete-arrival-trace pairs.'],
        ['计数器缺失' if zh else 'Missing counters', '原始长度缓存18个窗口中17个有效；64K低复用组另有6个可归属窗口。' if zh else 'Seventeen of 18 length-cache windows are valid; the 64K low-reuse cohort has six attributable windows.', '缺失值不记作零命中；按请求和计数窗口分别归属。' if zh else 'Missing values are not zero hits; attribution remains request/window specific.'],
        ['公网固定长输出' if zh else 'Public fixed long output', '公网请求自然结束；原生固定输出最高260,096词元。' if zh else 'Public generation stops naturally; native fixed output reaches 260,096 tokens.', '原生结果不能替代公网定长容量。' if zh else 'Native observations do not establish public fixed-length capacity.'],
        ['仪器与设备范围' if zh else 'Instrumentation and hardware scope', 'H200同负载、插座电表、完整物理DRAM字节及精确逐词元时钟未实测。' if zh else 'No matched H200 run, wall-meter energy, complete physical DRAM bytes or exact per-token clocks.', '保留规格参照与估算，避免将其标作实测。' if zh else 'Specifications and estimates remain distinct from measurements.'],
    ]
    return text + table(['范围','观测记录','分析规则'] if zh else ['Scope','Observed records','Analysis rule'], exclusions)


def cost_text(summary, base, lang):
    old = json.loads((base / 'data/completion/summary.json').read_text())
    def find_cost(obj):
        if isinstance(obj, dict):
            if 'prior_actual_usd' in obj and 'new_actual_usd' in obj: return obj
            for value in obj.values():
                found = find_cost(value)
                if found: return found
        elif isinstance(obj, list):
            for value in obj:
                found = find_cost(value)
                if found: return found
    cost = find_cost(old)
    amounts = [Decimal(str(cost['prior_actual_usd'])), Decimal(str(cost['new_actual_usd'])), Decimal(str(summary['cost_usd_all_attempts']))]
    total = sum(amounts)
    zh = lang == 'zh'
    rows = [
        ['服务性能矩阵' if zh else 'Service performance matrix', f'**{amounts[0]:.6f}**', '10月14–17日；按原始请求账单归属。' if zh else 'October 14–17; reconciled with the request ledger.'],
        ['独立并发、业务长度与连续服务窗口' if zh else 'Concurrency references, business lengths and continuous service', f'**{amounts[1]:.6f}**', '10月17日；按公网请求账单核对。' if zh else 'October 17; reconciled with the public-request ledger.'],
        ['受控长度、混合调度与低复用前缀' if zh else 'Controlled lengths, mixed scheduling and low-reuse prefixes', f'**{amounts[2]:.6f}**', '10月19日；696/696已发物理尝试账单可归属，含预热与探针。' if zh else 'October 19; billing attributable for 696/696 submitted attempts, including warmups and probes.'],
        ['实验API费用合计' if zh else 'Total experimental API charges', f'**{total:.6f}**', '互不重叠的账单小计相加；不含设备、电费和人力。' if zh else 'Sum of disjoint ledger cohorts; excludes equipment, electricity and labor.'],
    ]
    return ('### 实验费用汇总\n\n' if zh else '### Experimental cost ledger\n\n') + table(['工作负载范围','实付USD','核算范围'] if zh else ['Workload scope','Paid USD','Accounting boundary'], rows) + (
        '实验总费用描述本研究的调用开销。生产单位成本与投资回报仍使用同一负载组的费用、生成量和吞吐，避免把不同输入／输出比例的费用混为一个生产单价。\n' if zh else
        'The total describes API expenditure for this study. Production unit costs and investment returns use matched workload charges, output volume and throughput; heterogeneous input/output ratios are not collapsed into a production unit price.\n')


def conclusion_text(summary, lang):
    zh = lang == 'zh'
    rows = [
        ['GPU计算' if zh else 'GPU compute', '8,192³稠密矩阵：BF16 **94.08–94.34**、FP8 **191.88–192.64 TFLOP/s**。' if zh else 'Dense 8,192³ GEMM: BF16 **94.08–94.34**, FP8 **191.88–192.64 TFLOP/s**.', '取决于精度、形状和实现路径；不直接等于推理吞吐。' if zh else 'Precision-, shape- and implementation-specific; not inference throughput.'],
        ['数据供给与通信' if zh else 'Memory supply and communication', '8 GiB连续复制**223–227 GB/s**；1 GiB AllReduce总线带宽**21.81 GB/s**。' if zh else '8 GiB streaming copy: **223–227 GB/s**; 1 GiB AllReduce bus bandwidth: **21.81 GB/s**.', '复制按读＋写逻辑字节计量；NCCL与RDMA主机缓冲区测量分列。' if zh else 'Copy counts logical reads plus writes; NCCL and host-buffer RDMA remain separate.'],
        ['引擎阶段' if zh else 'Engine stages', '单并发预填充**2,427–3,354词元/s**；固定256输出的解码**49.8–60.2词元/s**。' if zh else 'Concurrency-one prefill: **2,427–3,354 tokens/s**; decode with 256 outputs: **49.8–60.2 tokens/s**.', '原生合成负载；计时包含引擎调度边界。' if zh else 'Native synthetic workloads; timings include engine scheduling.'],
        ['公网并发' if zh else 'Public concurrency', '并发1→8：合计**45.1→158.9词元/s**，正文首字95分位**1.65→2.39秒**。' if zh else 'Concurrency 1→8: **45.1→158.9 tokens/s**, first-content p95 **1.65→2.39 s**.', '60条短提示库、同组完整区组；不是服务承诺。' if zh else 'The 60-prompt bank and matched complete blocks; not a service guarantee.'],
        ['长输入与长输出' if zh else 'Long input and output', '原生固定输出最高**260,096词元**；公网64K输入条件实际生成**27,018词元**。' if zh else 'Native fixed output reaches **260,096 tokens**; the public 64K-input observation generates **27,018 tokens**.', '公网按自然结束分析；实际输出与max_tokens分列。' if zh else 'Public generation stops naturally; realized output and max_tokens are separate.'],
        ['混合请求调度' if zh else 'Mixed-request scheduling', '5组完整到达序列中4组可估计效应：短请求中位等待**1.18–3.37倍**、95分位**4.54–14.68倍**。' if zh else 'Four of five complete-arrival-trace pairs support effects: short-request median wait **1.18–3.37×**, p95 **4.54–14.68×**.', '第5组混合臂3/32短请求本地拒绝；该组完成子集延迟不用于完整配对效应。' if zh else 'Mixed arm 5 locally rejects 3/32 short arrivals; completed-subset latency is excluded from complete-pair effects.'],
        ['前缀缓存' if zh else 'Prefix caching', '64K低复用6个窗口均为**0%**缓存词元；32K名义空闲300秒组为**91.7%**，8K周期复用6轮约**100%**。' if zh else 'All six 64K low-reuse windows have **0%** cached tokens; the 32K nominal 300-second idle cohort reaches **91.7%**, and six 8K periodic reuse rounds are about **100%**.', '不同实验控制内容、等待或竞争；不据此推导统一TTL或长期业务命中率。' if zh else 'Content, idle gaps and pressure differ by experiment; no universal TTL or long-term business hit rate follows.'],
        ['持续服务与热状态' if zh else 'Sustained service and thermal state', '公网连续**3,600.1秒**，600次到达中599完整、1本地拒绝；双节点计算／带宽／混合阶段各**60／60／120分钟**。' if zh else 'Public serving: **3,600.1 seconds**, 599 complete and one local rejection out of 600 arrivals; each node has **60/60/120-minute** compute/bandwidth/mixed phases.', '各阶段独立；混合阶段含温度退避，不能拼为连续4小时业务可靠性。' if zh else 'Separate phases; mixed work includes thermal backoff and does not establish four-hour continuous serving reliability.'],
        ['H200技术参照' if zh else 'H200 reference', 'H200标称本地内存带宽更高，设备互联、精度支持及功耗边界不同。' if zh else 'H200 specifies greater local-memory bandwidth and different device links, precision support and power boundaries.', '官方规格对照；无同模型、同负载的H200推理实测。' if zh else 'An official-specification reference without matched H200 inference measurements.'],
    ]
    text = ('本研究在统一的测量框架下给出双DGX Spark的计算、内存、通信、引擎与公网服务表现。部署选择应同时考察处理量、用户等待、缓存条件、热状态和经济假设。\n' if zh else
            'This study measures dual-DGX-Spark compute, memory, communication, engine behavior and public serving within one measurement framework. Deployment decisions require throughput, user waiting, cache conditions, thermals and economic assumptions together.\n')
    text += table(['维度','主要结论','适用条件'] if zh else ['Dimension','Principal finding','Scope'], rows)
    text += ('### 经济性与部署取舍\n' if zh else '### Economics and deployment trade-offs\n') + table(
        ['用途','情景结果','假设'] if zh else ['Use','Scenario result','Assumptions'], [
            ['自用' if zh else 'Self-use', '回本**5.48年**；五年ROI **−8.7%**。' if zh else '**5.48-year** payback; five-year ROI **−8.7%**.', '双机US$10,000；短提示并发8；75%有效生产比例；含估算电费，未计其他运营费。' if zh else 'US$10,000 hardware; short prompts at concurrency 8; 75% productive utilization; estimated electricity, other operating costs excluded.'],
            ['对外售卖' if zh else 'External sales', '回本**2.65年**；五年ROI **+88.5%**。' if zh else '**2.65-year** payback; five-year ROI **+88.5%**.', '75%售出比例；参考API综合价的2倍售价；不是实际收入。' if zh else '75% sold utilization; twice the blended API reference price; not observed revenue.'],
        ])
    text += ('### 局限与后续工作\n' if zh else '### Limitations and future work\n') + table(
        ['方向','下一步验证','验收依据'] if zh else ['Direction','Next investigation','Evidence required'], [
            ['混合服务调度' if zh else 'Mixed-serving scheduling', '比较长短请求分队与预填充预算。' if zh else 'Compare separate short/long queues and prefill budgets.', '相同完整到达序列；同时报告尾部等待、吞吐和本地拒绝。' if zh else 'Identical complete arrival traces; report tails, throughput and local rejections together.'],
            ['公网固定长输出' if zh else 'Public fixed long output', '核验强制生成参数的端到端透传。' if zh else 'Verify end-to-end forwarding of forced-generation controls.', '实际输出达到目标，保持输入与并发一致；自然结束另列。' if zh else 'Realized target lengths with matched inputs and concurrency; natural stopping remains separate.'],
            ['缓存与持续性' if zh else 'Caching and sustained behavior', '扩大业务竞争、独立日期与环境温度范围。' if zh else 'Extend competing business traffic, independent dates and ambient temperatures.', '可归属计数器窗口、重复区组、退避占时及达标吞吐。' if zh else 'Attributable counters, repeated blocks, cooling fraction and goodput.'],
            ['硬件机制与H200' if zh else 'Hardware mechanisms and H200', '取得物理内存流量与匹配的H200配置。' if zh else 'Obtain physical-memory traffic and matched H200 configurations.', '同模型、精度、长度、缓存及通信路径的可比观测。' if zh else 'Matched model, precision, lengths, cache state and communication paths.'],
            ['实际持有成本' if zh else 'Observed ownership cost', '以插座电表、真实利用率和运营费用替换估算。' if zh else 'Replace estimates with wall-meter energy, real utilization and operating expenses.', '重新计算1／3／5年现金流、回本时间与ROI。' if zh else 'Recompute 1/3/5-year cash flows, payback and ROI.'],
        ])
    return text


def normalize_captions(doc):
    """Number captions by their final section and mirror them in Markdown."""
    counts = {}; image_captions = {}
    for element in doc.soup.find('main').find_all(['table', 'figure']):
        heading = element.find_previous('h2')
        section = heading.get('id', 'section-01').removeprefix('section-') if heading else '01'
        kind = 'table' if element.name == 'table' else 'figure'
        key = (section, kind); counts[key] = counts.get(key, 0) + 1
        caption = element.find('caption' if kind == 'table' else 'figcaption')
        if caption is None:
            caption = doc.soup.new_tag('caption' if kind == 'table' else 'figcaption')
            element.insert(0, caption) if kind == 'table' else element.append(caption)
        title_node = caption.select_one('.visual-title')
        title = title_node.get_text(' ', strip=True) if title_node else caption.get_text(' ', strip=True)
        title = re.sub(r'^(?:(?:图|表|Figure|Table)\s*\d{2}[-–]\d{2}\s*[｜|·:：]?\s*|2026-10-19[:： ]*)', '', title)
        if not title:
            nearest = element.find_previous(['h3','h2']); title = nearest.get_text(' ',strip=True)
        label = (('表' if kind == 'table' else '图') if doc.lang == 'zh' else ('Table' if kind == 'table' else 'Figure')) + f' {section}-{counts[key]:02d}'
        caption.clear()
        number = doc.soup.new_tag('span', attrs={'class':'visual-label'}); number.string = label
        content = doc.soup.new_tag('span', attrs={'class':'visual-title'}); content.string = title
        caption.extend([number, content])
        if kind == 'figure':
            img = element.find('img')
            if img and img.get('src','').startswith('data:image/svg+xml;base64,'):
                digest = hashlib.sha256(base64.b64decode(img['src'].split(',',1)[1])).hexdigest()
                image_captions[digest] = label + '｜' + title
    lines = doc.md.splitlines(); result = []; i = 0
    while i < len(lines):
        line = lines[i]; result.append(line)
        found = re.fullmatch(r'!\[[^]]*\]\(([^)]+)\)',line.strip())
        if found and (doc.out / found[1]).is_file():
            digest = hashlib.sha256((doc.out / found[1]).read_bytes()).hexdigest()
            if digest in image_captions:
                j = i + 1
                while j < len(lines) and not lines[j].strip(): j += 1
                if j < len(lines) and re.match(r'^(?:图|Figure)\s*\d{2}[-–]\d{2}', lines[j]): i = j
                result += ['', image_captions[digest], '']
        i += 1
    doc.md = '\n'.join(result).rstrip() + '\n'


def integrate(md, soup, summary, lang, out, base):
    doc = Document(md, soup, out, lang); zh = lang == 'zh'
    pick = lambda a,b: a if zh else b
    # One abstract and one concluding synthesis cover the entire study.
    doc.drop(pick('2026-10-19 独立补测结论','Independent measurements on 2026-10-19'))
    doc.paragraph(pick('原生引擎固定输出补充结果：','Native fixed-output completion:'), '')
    if not zh: doc.paragraph('Additional native-engine result:', '')
    doc.replace_section('05', coverage_text(summary, base, lang))
    doc.replace_section('12', conclusion_text(summary, lang))
    doc.drop(pick('补充实验的实际API费用','Actual API charges for the additional experiments'))
    doc.drop(pick('2026-10-19 补测实付API费用','Actual API charges for the 2026-10-19 supplement'))
    costs = cost_text(summary, base, lang)
    doc.insert_before(pick('双机持有成本：采购与电费','Two-node ownership: purchase and electricity'), (costs, doc.fragment(costs,'study-cost')))

    # Place engine, serving and cache observations beside their corresponding
    # mechanisms; retain detailed block records in the statistical appendix.
    archive = doc.subsection(pick('独立公网区组、混合干扰与连续窗口','Independent public blocks, interference and continuous window'))
    doc.append_section('14', archive)
    moves = [
        (pick('固定长输出：原生引擎的可达范围','Fixed long output: demonstrated native-engine range'),pick('公网并发：吞吐与等待','Public concurrency: throughput and waiting')),
        (pick('多场景GPU剖析与测量开销','GPU profiling across conditions and measurement overhead'),pick('公网并发：吞吐与等待','Public concurrency: throughput and waiting')),
        (pick('2026-10-14–17 长度批次：覆盖与实际生成','October 14–17 length batches: coverage and realized generation'),pick('思考预算：正文等待与输出形态','Reasoning budget: visible waiting and output shape')),
        (pick('2026-10-19 公网长度：上限与实际生成','2026-10-19 public lengths: caps and actual generation'),pick('思考预算：正文等待与输出形态','Reasoning budget: visible waiting and output shape')),
        (pick('2026-10-19：长请求对短请求等待的影响','2026-10-19: effect of long requests on short-request waiting'),pick('业务形态与缓存比例：实际覆盖','Business shapes and cache ratios: observed coverage')),
    ]
    for source,target in moves: doc.insert_before(target, doc.subsection(source))

    names = [
        ('补充基准：长度、业务与缓存负载','性能负载：长度、业务与缓存','Complementary workloads: length, business and caching','Performance workloads: length, business and caching'),
        ('2026-10-19：长度、干扰与缓存补测方案','受控长度、混合请求与低复用前缀的实验设计','2026-10-19: length, interference and cache protocol','Controlled lengths, mixed requests and low-reuse prefixes'),
        ('补测指标的定义与纳入规则','准入、配对与计数器归属规则','Additional metric definitions and inclusion rules','Admission, pairing and counter-attribution rules'),
        ('FP64矩阵计算补充测量','FP64矩阵计算','Additional FP64 matrix measurements','FP64 matrix measurements'),
        ('2026-10-14–17 长度批次：覆盖与实际生成','长上下文与输入／输出比例：响应分布','October 14–17 length batches: coverage and realized generation','Long contexts and input/output ratios: response distributions'),
        ('2026-10-19 公网长度：上限与实际生成','大输出预算：上限与实际生成','2026-10-19 public lengths: caps and actual generation','Large-output budgets: caps and realized generation'),
        ('2026-10-19：64K零设计重叠的缓存核验','64K低复用前缀：实际缓存比例','2026-10-19: cache verification at 64K with zero designed overlap','64K low-reuse prefixes: measured cache fractions'),
        ('2026-10-19：长请求对短请求等待的影响','混合请求调度：长输入对短请求等待的影响','2026-10-19: effect of long requests on short-request waiting','Mixed-request scheduling: short-request waiting with long inputs'),
        ('2026-10-19：双节点温度与资源','受控公网负载期间的双节点资源','2026-10-19: two-node temperature and resources','Two-node resources under controlled public workloads'),
        ('独立公网区组、混合干扰与连续窗口','按运行区组分列的服务观测','Independent public blocks, interference and continuous window','Service observations by execution block'),
        ('业务／长度补充区组','业务长度区组','Business/length completion blocks','Business-length blocks'),
        ('附录B：补充图表与统计明细','附录B：详细图表与统计明细','Appendix B: supplementary figures and statistics','Appendix B: detailed figures and statistics'),
        ('硬件补测：代码、数据与复算路径','硬件实验：代码、数据与复算路径','Hardware completion: code, data and reproduction paths','Hardware experiments: code, data and reproduction paths'),
        ('独立补充实验：代码、原始证据与分析','固定长输出、剖析与连续服务：数据索引','Independent completion arm: code, raw evidence and analysis','Fixed output, profiling and continuous service: data index'),
        ('2026-10-19：新增实验的代码与数据','受控公网工作负载：代码与数据索引','2026-10-19: additional experiment code and data','Controlled public workloads: code and data index'),
    ]
    for a,b,c,d in names: doc.rename(a if zh else c, b if zh else d)

    doc.paragraph(pick('10月14–17日的服务、硬件和引擎数据保留','Service, hardware and engine observations from October 14–17'),
        pick('本节按引擎阶段、固定长输出、公网并发、长度、混合请求和前缀缓存展开。不同工作负载分别统计；每项比较使用其对应的输入、输出、缓存和采样窗口。',
             'This section proceeds from engine stages and fixed output to public concurrency, lengths, mixed requests and prefix caching. Workloads are analyzed separately, with comparisons tied to their inputs, outputs, cache state and sampling windows.'))
    doc.paragraph(pick('两节点在专用窗口内测量','Both nodes are measured in a dedicated window.'),
        pick('长度、完整混合到达序列和低复用前缀使用专用双节点窗口。请求路径为Infron → Cloudflare → 预注册载荷转发入口 → SGLang；两臂使用相同路径。采样日期为2026-10-19，启动参数见第1节。引擎原生请求、其他公网负载和硬件实验保留各自的配置快照与计时边界。',
             'Controlled lengths, complete mixed traces and low-reuse prefixes use a dedicated two-node window on October 19, 2026. Requests traverse Infron → Cloudflare → registered-payload forwarding proxy → SGLang; both arms use the same path. Launch arguments appear in Section 1. Native-engine requests, other public workloads and hardware tests retain their own configuration snapshots and timing boundaries.'))
    doc.rename(pick('原未发送的输入／输出组合；每条件1条。','One observation for each formerly unsent input/output condition.'),
               pick('8个输入／输出预算组合；每条件1条。','Eight input/output-budget combinations; one observation per condition.'))
    doc.paragraph(pick('这组数据补齐','This adds response evidence for'),
        pick('八个大输出预算条件均取得完整响应；本组自然输出最大观测量为**27,018词元**。这表示对应输入材料的实际生成结果，不是服务输出上限。固定输出的最高观测量**260,096词元**来自原生引擎，按其独立路径解释。',
             'All eight large-output-budget conditions have complete responses; this cohort\'s maximum natural output is **27,018 tokens**. It describes generation for these inputs, not a service output limit. The **260,096-token** fixed-output observation belongs to the native-engine path.'))
    doc.paragraph(pick('取得 **3/3** 个完整独立配对','**3/3** independent pairs qualify'),
        pick('三个独立配对、六个窗口均可归属，实际缓存词元比例均为**0%**。结果与靠前位置使用不同输入标记的低复用设计一致；即时复用、空闲保留和竞争实验分别回答不同的缓存问题。',
             'All three independent pairs and six windows are attributable, with **0%** measured cached tokens throughout. This agrees with the low-reuse construction using distinct early markers. Immediate reuse, idle retention and pressure experiments address separate cache questions.'))
    doc.paragraph(pick('已完整记录 **5/5** 组到达与终态','All arrivals and outcomes are observed in **5/5** pairs'),
        pick('五组到达与终态均已完整记录，其中**4/5**组可估计完整配对效应：中位等待比值**1.18–3.37倍**，95分位比值**4.54–14.68倍**。范围描述区组间差异，不是置信区间。截断到达序列的区组不纳入该效应估计，排除规则见第05节。',
             'All arrivals and outcomes are recorded for five pairs; **4/5** support complete-pair effects: median-wait ratios **1.18–3.37×** and p95 ratios **4.54–14.68×**. These are ranges across blocks, not confidence intervals. Truncated-arrival blocks are excluded under the rules in Section 05.'))
    doc.paragraph(pick('本批以独立时间轮次补齐硬件证据。','This segment adds independent time-round evidence:'),
        pick('基础矩阵为每节点107个条件、每条件10轮；扩展矩阵覆盖通用FP32乘加、随机指针追踪、共享内存、统一内存访问、CPU竞争、多执行流及受支持的低精度／2:4稀疏矩阵计算。每次计时前后验证结果，保护跳过和不支持项不进入性能分母。49条校准观测单独归档，不加入这些重复轮次。',
             'The base matrix has 107 conditions per node and ten repetitions per condition. The extended matrix covers scalar FP32 FMA, pointer chasing, shared/managed memory, CPU contention, multiple streams and supported low-precision/2:4 sparse GEMM. Validation precedes and follows timing; guard skips and unsupported cases are excluded. Forty-nine calibration observations remain separate from these repetitions.'))
    # Technical descriptions retain their meaning without revision history.
    replacements = [
        ('不新增长期保留或缓存容量结论','不推导长期保留或缓存容量结论','does not add long-retention','does not establish long-retention'),
        ('低思考预算补充在','低思考预算在','new anchors are separate cohorts','length anchors are separate cohorts'),
        ('补充的性能范围','覆盖的性能范围','Additional performance scope','Performance scope'),
        ('下文补充三种输入／并发条件及配对开销','下文报告三种输入／并发条件及配对开销','The following adds three input/concurrency conditions','The following reports three input/concurrency conditions'),
        ('新增重复区组','重复区组','New repeated blocks','Repeated blocks'),
        ('新增结果独立报告','按工作负载分别报告','the new date remains a separate experiment','workloads are analyzed separately'),
        ('新增请求与计时CSV','请求与计时CSV','Additional request/timing CSV','Request/timing CSV'),
        ('新增长度结果','长度结果','New length findings','Length findings'),
        ('以下补充图表','以下详细图表','These supplementary figures','These detailed figures'),
        ('补测期间','受控负载期间','during the dated measurements','under controlled workloads'),
        ('长度、干扰与缓存补测方案','长度、混合请求与缓存实验设计','Length, interference and cache protocol','Length, mixed-request and cache protocol'),
        ('补充实验已完成','缓存专项包含','The supplement has completed','The cache study contains'),
        ('补充区组','业务长度区组','Business/length completion block','Business-length block'),
        ('当前NCCL日志','NCCL测量日志','This separate historical host-buffer test','This separate host-buffer test'),
        ('独立历史主机缓冲区测量','独立主机缓冲区测量','all historical data','all workloads'),
        ('目前有 17/18','包含17/18','currently has 17/18','contains 17/18'),
        ('历史分段观测（5段，包含中断）','分段观测（5段，包含中断）','Historical segmented observation','Segmented observation'),
    ]
    for a,b,c,d in replacements: doc.rename(a if zh else c, b if zh else d)
    doc.paragraph(pick('混合负载采用独立预热和基线；','Mixed work uses an independent warmup and baseline.'),
        pick('混合负载采用独立预热和基线：温度达到**85°C**时暂停提交GPU工作，低于**75°C**持续**30秒**后恢复。执行时间、冷却时间与中断尝试分别记录。计算和带宽阶段采用达到**85°C**即停止的控制规则，因此各阶段的持续时长与温控边界分别报告。',
             'Mixed work uses independent warmup and baseline measurements: submissions pause at **85°C** and resume below **75°C** after **30 stable seconds**. Execution time, cooling and interrupted attempts are recorded separately. Compute and bandwidth phases stop at **85°C**, so durations and thermal-control boundaries are reported per phase.'))
    doc.paragraph(pick('原**638次**到达的持续阶段','The original **638'),
        pick('638次到达的观测分为五段，包含中断的总跨度为34,580.3秒，不能代表连续运行。连续服务结论采用独立的3,600.1秒窗口，按五分钟分组展示。',
             'The 638-arrival observation comprises five interrupted segments spanning 34,580.3 seconds; it does not establish continuous operation. Continuous-service conclusions use the separate 3,600.1-second window, summarized in five-minute intervals.'))

    doc.replace_cell(pick('输入／输出与业务形态','Lengths and business shapes'), 2,
        pick('原生固定输出最高**260,096词元**；公网64K输入条件自然生成**27,018词元**。各长度预算与实际生成量分别报告。',
             'Native fixed output reaches **260,096 tokens**; the public 64K-input observation naturally generates **27,018 tokens**. Budgets and realized lengths are separate.'))
    doc.replace_cell(pick('前缀缓存与保留','Prefix cache and retention'), 2,
        pick('64K低复用6个窗口均为**0%**；32K名义空闲300秒组为**91.7%**；8K周期复用6轮约**100%**。各值对应不同控制条件。',
             'All six 64K low-reuse windows have **0%** cached tokens; the 32K nominal 300-second idle cohort reaches **91.7%**; six 8K periodic reuse rounds are about **100%**. Conditions differ.'))
    doc.add_overview_row(pick('公网服务与到达负载','Public serving and arrivals'), [
        '[混合请求调度](#section-08)' if zh else '[Mixed-request scheduling](#section-08)',
        '相同短输入与完整到达序列；混合臂加入32K长输入。' if zh else 'Matched short prompts and complete arrival traces; 32K inputs added to mixed arms.',
        '4组完整配对的短请求中位等待为孤立臂的**1.18–3.37倍**，95分位**4.54–14.68倍**；第5组保留**3条**本地拒绝。' if zh else 'Four complete pairs: median short-request wait **1.18–3.37×** the isolated arm, p95 **4.54–14.68×**; pair 5 retains **three local rejections**.',
    ])
    doc.rename('### 新证据、适用条件与后续工作','### 适用条件与后续工作')
    if soup.select_one('.hero .status'): soup.select_one('.hero .status').string = pick('完整性能研究报告','System and inference performance study')
    if soup.select_one('.hero .caption'): soup.select_one('.hero .caption').string = pick('测量窗口：2026-10-14 至 2026-10-19','Measurement period: October 14–19, 2026')
    doc.md = re.sub(r'(?m)^(数据快照（UTC）：|Data snapshot \(UTC\):).*$', pick('测量窗口：2026-10-14 至 2026-10-19；各实验保留独立的采样日期与统计分母。','Measurement period: October 14–19, 2026; experiments retain their sampling dates and statistical denominators.'),doc.md)
    # The environment is a dated observation, not a claim that every network
    # binding remained identical throughout all hardware and service windows.
    doc.replace_cell(pick('推理配置核验','Inference configuration verification'),1,
        pick('10月14–17日的性能记录对应各自启动快照；10月19日受控公网实验使用第1节的启动参数。各组内比较保持相同路径，跨窗口不合并绝对吞吐。',
             'October 14–17 performance records retain their launch snapshots; controlled public experiments on October 19 use the arguments in Section 1. Paths are matched within comparisons; absolute throughput is not pooled across windows.'))
    normalize_captions(doc)
    for strong in soup.select('main strong'): strong['class'] = list(set(strong.get('class',[]) + ['metric-value']))
    return doc.md, soup
