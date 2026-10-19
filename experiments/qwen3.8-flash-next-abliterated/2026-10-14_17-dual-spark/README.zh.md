[English](README.en.md)

# 双 DGX Spark · Qwen NVFP4 实验

测量时间：2026年10月14—17日。部署：两台128GB统一内存GB10，SGLang0.5.21，TP=2，MTP已启用。执行状态：结束；有效证据覆盖：部分条件缺失。

1. 先读[结论与条件](conclusions.zh.md)。
2. 在[完整报告](reports/report.zh.md)中依次查看定义、方法、覆盖、结果和局限；可下载[HTML](reports/report.zh.html)或[PDF](reports/report.zh.pdf)。
3. 查阅[冻结测试方案](protocols/test-plan.zh.md)、[机器可读实验记录](experiment.json)和[带散列的文件清单](artifacts.json)。

| 数据目录 | 测量范围 | 不能混淆的边界 |
| --- | --- | --- |
| `data/service` | 公网流式请求、连续区组、业务与缓存、成本情景 | 端到端包含网络和排队；自然结束不等于固定输出长度 |
| `data/hardware` | 卸载模型后的GPU计算、内存、算子、双机通信及持续工作 | 库调用计时不必然等于单内核或理论峰值 |
| `data/engine` | 原生引擎预填充、解码与GPU阶段 | 进程内阶段和GPU区间分别计时 |
| `data/completion` | 16条长输出、9组剖析配对、缓存、1小时服务、FP64与计数器 | 独立日期批次，不回填历史失败率 |
| `data/coverage.json` | 最终有效覆盖与缺口 | 执行结束不等于证据齐全 |

测试题库：MMLU-Pro、MATH-500、GSM8K、IFEval、HumanEval+和MBPP+，每类10题，共60条；另有八种业务形态及长度／前缀设计。题目版本、抽样与内容散列见 `protocols/`。
