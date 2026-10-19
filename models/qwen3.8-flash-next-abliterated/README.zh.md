[English](README.en.md)

# Qwen3.8 Flash Next Abliterated (NVFP4)

| 字段 | 记录 |
| --- | --- |
| 权重 | [dealignai 模型仓库](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4) |
| 固定 revision | `be794b990578ef3031eccf9f28e675a289a09ee9` |
| 量化 | NVFP4，SGLang `modelopt_fp4` |
| 上下文 | 262,144 token，输入与输出合计 |
| 文本结构 | 48层，隐藏维度2560，512路由专家，每词元选10；36线性注意力层、12完整注意力层 |
| 参数量 | 本研究未独立核验总参数与激活参数数量 |
| 完整研究报告 | [双DGX Spark系统与推理性能](../../experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/reports/report.zh.md)；2026-10-14至19日 |

本项目只测推理系统性能，固定题目作为负载，不报告模型能力排名。模型名及消融定位不构成能力或安全表现结论。

测量记录：[硬件、引擎与服务矩阵](../../experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/README.zh.md) · [受控长度、混合调度与前缀缓存](../../experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/README.zh.md)。
