[English](README.md) · **中文**

# NVIDIA DGX Spark GB10 测试套件

面向推理工程师的可复用性能实验项目。以 **硬件 → 算子与通信 → 推理引擎 → 用户可见服务** 为测量主线，按模型和实验批次保存方案、代码、数据、图表及结论。既可以离线阅读和复算已有结果，也可以在自己的端点或 GPU 上执行新的有界测试。

## 1. 实验背景与研究范围

DGX Spark 将 Blackwell GPU、Arm CPU 与统一内存集成在 GB10 平台中。实际推理性能同时取决于计算精度、内存访问、模型结构、双机通信、推理引擎调度及公网服务链路；设备规格中的峰值数字不能直接换算为模型生成速度或用户等待时间。

本项目用分层测量解释这些因素：先建立单机计算和内存基线，再测算子与双机通信，随后观察引擎预填充／解码，最后测量真实 API 的响应速度、吞吐、缓存复用及混合负载表现。每项结论关联具体配置、工作负载、计量单位和数据表。规格参照、实测结果和成本假设分开呈现。

| 当前实验对象 | 已记录配置 |
| --- | --- |
| 硬件 | 两台 NVIDIA DGX Spark／GB10；每台 128 GB LPDDR5x 统一内存；双节点张量并行 |
| 服务模型 | `qwen/qwen3.8-flash-next-abliterated`，通过 Infron 访问 |
| 权重来源 | [dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4/tree/be794b990578ef3031eccf9f28e675a289a09ee9)，固定 revision `be794b990578ef3031eccf9f28e675a289a09ee9` |
| 推理配置 | SGLang 0.5.21；NVFP4／`modelopt_fp4`；TP=2；MTP 开启；上下文总上限 262,144 token |
| 研究时间与记录 | 2026-10-14 至 2026-10-19；[完整研究报告](experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/reports/report.zh.md)；[测量记录](experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/README.zh.md) |
| 测量目标 | 服务性能及资源表现；题集作为输入材料，不评分答案正确性，不做模型能力排名 |

这是自定义工程研究，不是 MLPerf 认证提交。H200 在报告中是架构与规格参照，没有同负载实测；经济性部分为显式假设下的估算。

## 2. 测试方案

### 分层实验设计

| 层级／测试项 | 控制条件与方法 | 主要输出 |
| --- | --- | --- |
| GPU 核心计算 | 按精度、矩阵形状、规模和库实现分别测 GEMM；校验数值；区分稠密计算与稀疏峰值口径 | TFLOP/s、耗时、精度与有效条件 |
| GPU 内存访问 | 改变工作集、读／写／复制模式；分别报告逻辑字节，不将 GB10 统一内存称为 HBM | GB/s、规模响应曲线 |
| 独立算子与系统干扰 | 算子形状、并行度、CPU 竞争及持续运行条件 | 算子延迟、资源和温度随时间变化 |
| 双机通信与重叠 | 两个 rank 配对；按集合通信类型、消息大小和计算／通信组合测量 | 算法带宽、总线带宽、延迟、重叠收益 |
| 引擎内部 | 原生 SGLang 请求、固定输出探针、阶段计时、GPU 轨迹；剖析与非剖析结果分开 | 预填充、解码、原生缓存计数及阶段耗时 |
| 公网模型服务 | 完整流式响应；并发 1／2／4／8；闭环并发与独立到达分开；长短请求配对 | 首 token／正文首字时间、总耗时、吞吐、完成率、拒绝与错误 |
| 缓存与稳定性 | 前缀比例、重复访问、保留时间、竞争及周期复用；核对原生计数或完整计数器窗口 | 实际命中比例、等待时间变化、复用稳定性 |
| 成本与架构参照 | 双机采购 US$10,000；明确电价、利用率及价格假设；H200按资源功能对齐 | 1／3／5 年持有成本、回本条件及投资回报情景 |

### 工作负载与数据来源

- **固定输入题库：** MMLU-Pro、MATH-500、GSM8K、IFEval、HumanEval+、MBPP+，每类 10 条，共 60 条。用于覆盖知识、数学、指令及编程等输入形式；抽样清单、来源与许可见[数据与来源](docs/data-and-provenance.zh.md)及[实验协议目录](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/protocols/)。
- **长度矩阵：** 输入、输出各设 1K、2K、4K、8K、16K、32K、64K、128K、256K 九档，1K=1,024 token；按输入＋输出不超过 262,144 筛选，并保留模板开销与边界条件。完整 256K 输入不能再加正长度输出。输出上限、自然停止长度、强制固定长度是三个不同概念。
- **业务输入：** 八类业务形态覆盖不同输入／输出比例。完整报告逐项说明含义、使用背景、测量目的与输入输出示例。
- **缓存输入：** 共享前缀目标为 0%／25%／50%／75%／90%／100%；这些是内容设计比例，**不是实际缓存命中率**。保留、竞争、周期复用分别测试。
- **合成硬件输入：** 固定形状矩阵、内存工作集和通信消息。原生超长输出可能包含重复空白，只能支持容量／调度结论。

### 如何保证结论可核查

请求用于统计等待时间；完整连续测试块用于计算吞吐；双机同条件同轮次为一个通信试验；配对测试用于分析缓存和干扰。重试、拒绝、失败与缺失单独计数，历史记录不被后续成功覆盖。吞吐按同组完整块的总输出 token／总墙钟时间计算，不平均拼接互不相同的速率。

模型、权重 revision、引擎参数、随机种子、计划、实际输出和 SHA-256 清单均按批次保存。SSE 事件到达时间不是精确逐 token 的 GPU 时间；公网、原生引擎和硬件数据不能直接混合。少量重复不支持稳健的 p99 或普遍服务承诺。定义见[测量方法](docs/methodology.zh.md)和[指标字典](docs/metrics.zh.md)。

## 3. 按目的阅读

**完整测试报告**按GPU计算、内存、通信、引擎阶段、公网服务、缓存、持续性能与经济性组织。先读[摘要与主要结论](experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/reports/report.zh.md#section-01)，再沿对应章节查看定义、工作负载、图表和适用条件。


| 你希望了解什么 | 建议阅读顺序 |
| --- | --- |
| 快速判断部署表现 | [主要结论与适用条件](experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/conclusions.zh.md) → 完整报告的摘要、服务性能和总体结论 |
| 理解实验可信度 | [模型索引](models/README.zh.md) → [实验协议](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/protocols/) → [方法](docs/methodology.zh.md) → [指标](docs/metrics.zh.md) → 对应数据表 |
| 阅读完整图表和交互成本计算 | 下载[中文 HTML](experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/reports/report.zh.html)；英文版见[English HTML](experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/reports/report.en.html) |
| 分享或打印 | [中文 PDF](experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/reports/report.zh.pdf)／[English PDF](experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/reports/report.en.pdf)；[Release 下载](https://github.com/InfronAI/NVIDIA-DGX-Spark-GB10-TestingSuits/releases) |
| 在 GitHub 直接阅读 | [中文 Markdown](experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/reports/report.zh.md)／[English Markdown](experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/reports/report.en.md) |
| 复算已有结果 | 下面的离线复算步骤 → [快速开始](docs/quickstart.zh.md) |
| 重测或增加模型 | 下面的执行示例 → [复现指南](docs/reproduction.zh.md) → [贡献指南](docs/contributing.zh.md) |

HTML 是可独立分享的单文件，图表与计算脚本内嵌；下载后用浏览器打开。GitHub 文件页展示源码，不运行 HTML。只阅读报告不需要安装 Python、GPU、SSH 或 API 凭证。

**主要观测：** 同一短提示比较组中，并发1→8，输出吞吐**45.1→158.9词元/s**、正文首字95分位**1.65→2.39秒**；原生固定输出最高**260,096词元**。混合请求实验完整记录5组到达与终态，其中4组支持完整配对效应，第5组保留3条本地拒绝。全部样本分母、排除规则和结论边界见报告第05节。

![同组容量与等待时间](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/reports/capacity.zh.svg)

## 4. 下载与安装

可在公开仓库的 **Code → Download ZIP** 下载源码，解压后进入项目根目录；需要保留版本记录时使用 Git：

```bash
git clone https://github.com/InfronAI/NVIDIA-DGX-Spark-GB10-TestingSuits.git
cd NVIDIA-DGX-Spark-GB10-TestingSuits
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
sparkbench --help
```

要求 Python **3.10+**。基础包安装 HTTPX 和 NumPy，不安装 CUDA、驱动或模型权重。Windows 激活环境使用 `.venv\Scripts\Activate.ps1`；GPU 执行入口面向 Linux GB10。若要核对首次发布版，克隆后先执行 `git switch --detach v0.1.0` 再安装；`main` 包含后续更新。安装来源是本仓库，不能假设同名 PyPI 包具有相同代码。

可选报表依赖：`python -m pip install -e '.[reports]'`。已有 HTML／PDF 无需这些依赖；WeasyPrint 的本地系统库需按其平台要求另行安装。可移植 CLI 的 `summarize` 只生成复算表，不自动重建历史完整报告。

重建中文图表时，操作系统还需提供中文字体；绘图程序包含PingFang SC与Noto Sans CJK SC等回退字体。

## 5. 离线使用与复算

在仓库根目录运行：

```bash
sparkbench list
sparkbench validate
sparkbench summarize experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark --output outputs/summary
sparkbench matrix --context 262144 > outputs/length-matrix.json
sparkbench analyze-completion experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion --output outputs/completion
python -m unittest discover -s tests -v
```

| 命令 | 做什么／输出什么 |
| --- | --- |
| `list` | 列出模型及其已登记实验，帮助选择数据批次 |
| `validate` | 核对模型／实验关联、双语文档和产物 SHA-256；不调用 API |
| `summarize` | 从已发布连续测试块表复算容量，输出 `summary.json`、`capacity.csv`；不重新生成模型回答 |
| `analyze-completion` | 复算10月19日逐尝试用量、长度、完整干扰配对和缓存计数器窗口；输出JSON与CSV，不调用模型 |
| `matrix` | 输出上下文约束下的长度设计及排除项；矩阵不是已完成实验数 |
| 单元测试 | 验证配置、流式解析、统计及模拟服务等软件行为；不代表 GPU 性能验收 |

这里的“复算”验证保存数据能否得到相同统计结果；在新端点上“重测”会产生新请求、费用与不同环境下的新数据。完整历史服务操作、GPU 轨迹采集和源站对账并非一条 CLI 命令可以自动复现，范围见[数据与来源](docs/data-and-provenance.zh.md)。

## 6. 连接自己的端点执行

先复制配置，再填写自己的 `base_url` 和 `model`。例如 `base_url` 必须包含 OpenAI 兼容接口的 `/v1`，模型名须为该端点目录实际接受的名称；Infron 别名与原生 SGLang 的服务名可能不同。

```bash
cp configs/serving.example.json configs/serving.local.json
# 编辑 serving.local.json；默认6条输入、并发1、输出上限128 token。
# 在本机安全注入环境变量 SPARKBENCH_API_KEY，不把密钥写进配置或提交到Git。
sparkbench serve --config configs/serving.local.json --workloads configs/smoke-workloads.jsonl
# 上一步仅预览；确认端点和负载后，用全新的输出目录执行：
sparkbench serve --config configs/serving.local.json --workloads configs/smoke-workloads.jsonl --output outputs/my-serving-run --execute
```

`serve` 保存计划、逐请求／逐尝试回执及 `summary.json`，默认不保存完整提示和回答。独立固定到达示例见 `configs/arrival.example.json`；原生接口见 `configs/native.example.json`，其输入必须是与模型 tokenizer 匹配的 `input_ids`。这些入口不会自动配置网关路由、查询生产数据库或启动远端 profiler。

显式 Cloudflare HTTP 暂态错误执行有限重试；超时／断连的结果未知时不会盲目重放。输出目录放置 `STOP` 可停止新请求，已在途请求按有限超时排空。新测量使用新目录，不覆盖原始记录。字段说明和监控接入见[复现指南](docs/reproduction.zh.md)。

## 7. 在 GB10 上测硬件与双机通信

在已有匹配 CUDA／PyTorch 环境的 Linux GB10 节点安装本项目。NVFP4 还需要兼容的 FlashInfer／CUTLASS 后端；仓库不包含模型权重或这些系统安装器。历史实测算法与新适配入口的验证范围不同，不能把 CPU／模拟服务测试视为对所有 CUDA 环境的认证。

```bash
sparkbench hardware --mode base
# 预览后，在可独占的节点执行：
sparkbench hardware --mode base --output outputs/base-run --execute
sparkbench hardware --mode extensions --output outputs/extensions-run --execute
# 两台节点分别执行；将192.0.2.10替换为节点0测试网卡地址。
sparkbench hardware --mode collectives --rank 0 --master-addr 192.0.2.10 --output outputs/nccl-r0 --execute
sparkbench hardware --mode collectives --rank 1 --master-addr 192.0.2.10 --output outputs/nccl-r1 --execute
```

`overlap` 使用同样的双节点启动方式；`soak` 为持续运行入口。通信取决于安装环境和 `NCCL_*` 配置，必须结合 NCCL 日志确认实际传输路径。运行器不自动安装驱动、停止推理服务或调整功率／频率。保留至少 16 GiB 或 15% 内存余量；温度达到 85°C 退避，低于 75°C 连续 30 秒后恢复，冷却最多 600 秒；集合通信在阈值处终止本轮，再以新成对批次重试。`STOP_GPU` 停止后续提交。详细参数与可复现范围见[复现指南](docs/reproduction.zh.md)。

## 8. 项目结构、贡献与许可

```text
models/          模型身份、固定权重版本、实验索引
experiments/     每批实验的方案、数据表、结论及双语报告
src/sparkbench/  配置、流式客户端、统计、监控与CLI
benchmarks/gb10/ CUDA／库级测量算法及本地保护
configs/         不含凭证的配置与输入示例
docs/            中英文方法、指标、复现和贡献指南
reference/       历史实现参考文本，不作为生产控制入口
provenance/      导入来源与文件散列
outputs/         本机新实验输出，不纳入已发布历史数据
```

增加模型时新建模型身份清单和日期实验目录，登记环境、计划、原始／派生数据关系及适用结论，同时补全中英文文档和散列清单。提交前运行 `sparkbench validate` 和单元测试；不提交密钥、运行环境或模型权重。具体格式见[贡献指南](docs/contributing.zh.md)。

原创软件使用 [MIT](LICENSE)。模型、题集及第三方组件保留各自条款，见 [NOTICE](NOTICE.md)。
