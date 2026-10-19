[中文](README.zh.md)

# NVIDIA DGX Spark GB10 Testing Suite

An evidence-first **hardware → operators → two-node communication → model-serving** study for inference engineers. Model and experiment indexes connect configurations, measurement tables, methods and conclusions. Read the findings, recompute published results, or run bounded workloads on your own deployment.

## Start here

| Goal | Entry point |
| --- | --- |
| Understand findings and limits | [Conclusions](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/conclusions.en.md) |
| Browse experiments by model | [Model index](models/README.en.md) |
| Read the full report | [English Markdown](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/reports/report.en.md) · [Offline HTML](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/reports/report.en.html) · [PDF](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/reports/report.en.pdf) |
| Recompute published results | [Quickstart](docs/quickstart.en.md) |
| Test another deployment/model | [Reproduction](docs/reproduction.en.md) · [Contributing experiments](docs/contributing.en.md) |
| Understand metrics and boundaries | [Methodology](docs/methodology.en.md) · [Metric dictionary](docs/metrics.en.md) |

Download HTML and open it in a browser; figures and interactive calculations are embedded. GitHub's file preview does not execute HTML.

## Current experiment

| Model / deployment | Experiment | Supported finding |
| --- | --- | --- |
| Qwen3.8 Flash Next Abliterated / NVFP4 / SGLang 0.5.21 / two nodes, TP=2 / MTP enabled | [2026-10-14_17-dual-spark](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/README.en.md) | Within the same short-prompt cohort, concurrency 1→8 gives **45.1→158.9 output tokens/s**, with visible-content latency p95 **1.65→2.39 s**. |

**Execution ended; evidence coverage is incomplete.** Of299 public-length cases,290 responses completed,8 were unsent and1 incomplete. Three of five interference candidates ran; zero pairs were fully observed. Sixteen native fixed-output probes do not fill public-path gaps. H200 is a specification reference, not a matched measured platform.

![Matched capacity and latency](experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark/reports/capacity.en.svg)

## Reusable components

- OpenAI-compatible SSE and native SGLang `/generate` clients: bounded concurrency, independent fixed arrivals, deadlines, per-attempt records and backoff for explicit Cloudflare HTTP failures.
- A1K–256K input/output design matrix filtered by total context, fixed task bank, business shapes, prefix overlap, retention, competition and periodic-reuse protocols.
- Hardware-validated GB10 CUDA/library algorithms: baseline GEMM, low precision, memory access, operators, CPU contention, collectives, compute/communication overlap and sustained work.
- Prometheus windows, engine-stage and GPU-trace interpretation, offline numerical reconstruction, bilingual documentation and full reports.

Historical algorithms ran on the original hardware. The portable adapters have offline and mock-service validation; **production GPUs were not rerun for this repository release**. See [code and evidence scope](docs/data-and-provenance.en.md).

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
sparkbench list
sparkbench validate
sparkbench summarize experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark --output outputs/summary
python -m unittest discover -s tests -v
```

## Layout

```text
models/          Model identity, weight revision, experiment index
experiments/     Per-run protocols, tables, conclusions, bilingual reports
src/sparkbench/  Configurable requests, SSE, statistics, monitoring, CLI
benchmarks/gb10/ Hardware-validated CUDA/library algorithms and local guards
configs/         Editable examples without credentials
docs/            Bilingual methods, reproduction, metrics, contribution guide
reference/       Historical implementation text, not production controllers
provenance/      Import sources and checksums
```

Original software uses [MIT](LICENSE). Models, datasets and third-party components retain their own terms; see [NOTICE](NOTICE.md). Model weights are not included.
