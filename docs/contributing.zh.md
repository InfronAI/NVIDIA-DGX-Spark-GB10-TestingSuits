[English](contributing.en.md)

# 新增模型与实验

1. 在`models/<模型标识>/model.json`登记权重仓库、完整revision、精度和上下文，并提供`README.zh.md`与`README.en.md`。
2. 每次实验新建`experiments/<模型标识>/<日期-引擎-精度-拓扑>/`。复制`templates/experiment.json`，填写硬件、引擎、配置与研究目标；时间、路径和模型身份不能复用旧值。
3. 执行前冻结负载、种子、重复次数、到达过程、时限和停止条件。明确哪些是固定样本数、哪些只是请求上限。
4. 原始尝试与有效分析样本分开；失败、拒绝、未执行、unsupported和缺失字段都保留。不要把重试成功回填为首次成功。
5. 增加双语结论、README、方案和报告。每项结论链接其数据表，标明工作负载、样本数、计时边界、未测条件与未来工作。
6. 为数据/报告生成`artifacts.json` SHA-256清单，更新模型实验索引，运行`python scripts/verify_repository.py`与离线测试。提出PR时附条件变化和验证范围。

只对同模型revision、精度、请求长度、并发、缓存、拓扑和计时边界相容的实验计算速度比。理论规格与实测分列，未经匹配的MTP或平台对照不推导因果收益。

## GitHub 自动检查

[工作流模板](../deploy/github-actions.validate.yml)包含离线测试、数据完整性检查和执行预览。本次上传凭证不含 `workflow` 权限，因此模板尚未在 GitHub Actions 中启用。仓库维护者可使用具有工作流写入权限的账号，将它保存为 `.github/workflows/validate.yml`；它不调用模型服务，也不运行 GPU 负载。发布前本地检查回执见 [publication-validation.json](../provenance/publication-validation.json)。

## 发布前脱敏检查

运行`python scripts/verify_repository.py`检查裸域名、URL、非示例内网地址、个人目录、SSH用户名和内嵌SVG。网络地址采用`example.invalid`、文档专用IP或环境变量；用户名采用`benchmark`等占位符。外部参考域名需明确列入允许清单；真实凭证和部署信息不得写入测试样例。
