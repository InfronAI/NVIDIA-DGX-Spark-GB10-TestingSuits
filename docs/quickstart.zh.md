[English](quickstart.en.md)

# 快速开始

需要Python3.10或更新版本。离线索引和汇总不需要GPU、SSH、数据库或API密钥。

```bash
git clone https://github.com/InfronAI/NVIDIA-DGX-Spark-GB10-TestingSuits.git
cd NVIDIA-DGX-Spark-GB10-TestingSuits
python -m venv .venv
. .venv/bin/activate
pip install -e .
sparkbench list
sparkbench validate
sparkbench summarize experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark --output outputs/summary
python -m unittest discover -s tests -v
```

`summary.json`和`capacity.csv`从发布的连续区组表复算。校验将检查模型与实验关联、双语文件及测量文件SHA-256；文件变化必须明确重新登记。

下面两条只显示设计，不发请求、不调用GPU：

```bash
sparkbench serve --config configs/serving.example.json --workloads configs/smoke-workloads.jsonl
sparkbench hardware --mode base
```

真实执行在同一命令上显式添加`--execute`，并使用新的`--output`目录。配置见[复现指南](reproduction.zh.md)。仓库已公开；离线阅读和克隆不需要API凭证。
