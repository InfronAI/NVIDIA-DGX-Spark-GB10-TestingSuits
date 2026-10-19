[中文](quickstart.zh.md)

# Quickstart

Python3.10+ is required. Offline indexes and summaries need no GPU, SSH, database or API credentials.

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

`summary.json` and `capacity.csv` are reconstructed from published continuous-block tables. Validation checks model/experiment identity, bilingual documents and artifact SHA-256; changed artifacts require explicit re-registration.

These commands preview plans without making requests or calling GPUs:

```bash
sparkbench serve --config configs/serving.example.json --workloads configs/smoke-workloads.jsonl
sparkbench hardware --mode base
```

To execute, add `--execute` and select a new `--output` directory. See [reproduction](reproduction.en.md). The repository is currently private; reading/cloning requires repository access.
