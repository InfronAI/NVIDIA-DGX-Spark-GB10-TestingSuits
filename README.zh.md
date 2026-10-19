[English](README.en.md)

# NVIDIA DGX Spark GB10 测试套件

面向推理工程师的**硬件—算子—双机通信—模型服务**实验项目。按模型和不可混淆的实验批次组织配置、测量表、统计口径与结论；既能直接阅读结果，也能在自己的设备上运行测试。

## 从这里开始

| 目的 | 入口 |
| --- | --- |
| 理解这次测出了什么 | [结果与适用条件](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/conclusions.zh.md) |
| 按模型查找实验 | [模型目录](models/README.zh.md) |
| 阅读完整报告 | [中文 Markdown](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/reports/report.zh.md) · [离线 HTML](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/reports/report.zh.html) · [PDF](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/reports/report.zh.pdf) |
| 复算已发布结果 | [快速开始](docs/quickstart.zh.md) |
| 在新设备／新模型上测试 | [复现指南](docs/reproduction.zh.md) · [新增模型与实验](docs/contributing.zh.md) |
| 理解指标与边界 | [测量方法](docs/methodology.zh.md) · [指标字典](docs/metrics.zh.md) |

HTML 下载后直接用浏览器打开；图表和交互计算内嵌。GitHub 文件预览不执行 HTML。

## 当前实验

| 模型 / 配置 | 实验 | 可支持的主要结论 |
| --- | --- | --- |
| Qwen3.8 Flash Next Abliterated / NVFP4 / SGLang 0.5.21 / 双节点 TP=2 / MTP 开启 | [2026-10-14_17-dual-spark](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/README.zh.md) | 同一短提示比较组：并发 1→8，输出 **45.1→158.9 token/s**，正文首字 p95 **1.65→2.39 s**。 |

**执行已结束，证据覆盖不完整。** 公网长度计划 299 次，290 次完整响应、8 次未发送、1 次不完整；5 组干扰候选只有3组执行，完整配对为0。原生引擎16条固定输出探针不补填公网缺口。H200是规格参照，没有同负载实测。

![同组容量与等待时间](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/reports/capacity.zh.svg)

## 可复用内容

- OpenAI 兼容流式客户端和 SGLang 原生 `/generate` 客户端；有界并发、独立固定到达、超时、逐尝试记录、明确的 Cloudflare HTTP 错误退避。
- 1K–256K 输入／输出设计矩阵，按输入＋输出不超过上下文筛选；固定题库、业务形态、前缀比例、保留／竞争与复用方案。
- 已验证的 GB10 CUDA/库级硬件算法：基础矩阵、低精度、内存访问、算子、CPU竞争、集合通信、计算通信重叠、持续运行。
- 服务端 Prometheus 窗口、引擎阶段与 GPU 轨迹的分层解释；离线表格复算、双语文档及完整报告。

历史算法在原硬件上运行过；新整理的适配层通过离线与模拟服务验证，**没有为本次发布重新占用生产 GPU**。参考源码与可执行入口的区别见[代码与数据范围](docs/data-and-provenance.zh.md)。

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
sparkbench list
sparkbench validate
sparkbench summarize experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark --output outputs/summary
python -m unittest discover -s tests -v
```

## 项目结构

```text
models/          模型身份、权重版本、实验索引
experiments/     每批实验的协议、测量表、结论、双语报告
src/sparkbench/  配置化请求、SSE解析、统计、监控与CLI
benchmarks/gb10/ 已测CUDA/库级算法和本地保护
configs/         可编辑示例；不含凭证
docs/            中英文方法、复现、指标和贡献指南
reference/       历史算法参考文本，不作为生产控制入口
provenance/      导入来源与文件散列
```

原始软件采用 [MIT](LICENSE)；模型、题集和第三方组件遵循各自条款，见 [NOTICE](NOTICE.md)。仓库不包含模型权重。
