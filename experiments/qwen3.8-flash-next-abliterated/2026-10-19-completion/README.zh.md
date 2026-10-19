[English](README.en.md) · [中文](README.zh.md)

# 2026-10-19：公网长度、混合干扰与64K缓存补测

本批使用相同的双DGX Spark和SGLang配置，补充早期缺失的公网条件、完整到达记录、长短请求配对及低复用缓存计数窗口。原始2026-10-14至17日实验保持不变，新的逐尝试记录、计数器窗口和结论单独归档。

当前有效证据：长度 **8/8** 条；干扰结果 **5/5** 对（其中**4/5**可估计完整配对效应）；缓存 **3/3** 对。计划共 **695** 条逻辑请求，包含预检和预热。

先读[结论](conclusions.zh.md)，再读[完整中文报告](reports/report.zh.md)。可下载[独立HTML](reports/report.zh.html)或[PDF](reports/report.zh.pdf)离线分享；报告同时保留历史硬件、引擎、H200技术对照及经济性章节。

## 离线复算与报告重建

在仓库根目录安装后执行：

```bash
sparkbench analyze-completion experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion --output outputs/completion
python -m pip install -e '.[reports]'
python experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/analysis/build_report.py experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion --pdf
```

复算仅读取已发布标量证据，不调用模型或数据库。报告重建从只读的2026-10-14_17-dual-spark报告继承历史章节，再加入本日期结果。更改产物后需按新实验归档；原SHA256清单用于验证发布快照。

## 如何复现测量条件

1. 先核对[模型与部署配置](experiment.json)，确保权重revision、引擎版本、TP、上下文及MTP参数相同。
2. 从`protocols/public/manifest.json`与`protocols/cache/manifest.json`读取完整条件；`plans/*.jsonl`含输入、输出上限、配对身份及原始材料，SHA256见`protocols/plan-hashes.json`。
3. 长度条件逐条运行并保存实际usage；不得将max_tokens当实际输出量。缓存条件先测首条，再测第二条，每条前后排空并采样源站计数器。
4. 干扰实验依次预热全部32条短输入，再重放冻结的完整到达偏移；两臂短输入和偏移完全相同，混合臂加8条长输入。保持4个短请求worker、2个长请求worker及32格队列；出队时检查5秒超时阈值，实际拒绝等待可能更长。
5. 实测保存所有到达、拒绝、失败、重试、首字时间与源站用量。两节点约5秒采样；85°C退避、低于75°C连续30秒恢复。任何服务改动须在新日期记录。

通用`sparkbench serve`可做长度单条件重测和接口校准；它的均匀到达调度不等于本实验冻结的泊松混合序列。完整配对复现须实现清单中的到达偏移及worker分组，并按本仓库分析器验收。本仓库不自动变更生产路由或执行临时入口隔离。

## 数据文件

`data/requests.json`保留逐尝试标量；`data/arrivals.json`保留到达与准入终态；`data/blocks.json`保留计数窗口；`data/telemetry.json`记录资源。`protocols/environment.json`记录逐节点硬件型号、驱动、SDK及脱敏启动参数。`analysis/completion-summary.json`保存纳入结果与失败原因。正文和凭证不作为测量标量发布；公开题目与合成输入保留各自来源语言。

## 单条件重测示例

将自己的API密钥放入`SPARKBENCH_API_KEY`环境变量；先核对配置里的端点、路由和模型。以下命令只预览1K输入／16K输出上限条件，不发送请求：

```bash
sparkbench serve --config experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/configs/length-i1024-o16384.json --workloads experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/protocols/public/plans/gc19_length_output_axis_i1024_o16384.jsonl --output outputs/new-length-1k-16k
```

执行时追加`--execute`，并使用新输出目录；配置中的max_tokens决定输出上限。ignore_eos只是请求参数，是否透传须在自己的端点验证。源站温控需由节点侧提供，普通公网客户端不能代替双节点保护。完整泊松调度算法见[历史arrival源码](../../../reference/2026-study/serving/arrival.py.txt)，本批完整偏移见冻结manifest。
