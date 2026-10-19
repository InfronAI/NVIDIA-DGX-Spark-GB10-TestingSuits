[English](README.en.md) · [中文](README.zh.md)

# Dual-DGX-Spark system and inference performance study

This study measures Qwen3.8 Flash Next Abliterated NVFP4 on two NVIDIA DGX Spark / GB10 nodes over October 14–19, 2026. The complete report covers GPU compute, memory, communication, engine prefill/decode, public serving, lengths, caching, mixed scheduling, sustained performance and economics.

Start with [findings](conclusions.en.md), then the [complete English report](reports/report.en.md). [English HTML](reports/report.en.html) / [中文HTML](reports/report.zh.html) are standalone offline documents; [English PDF](reports/report.en.pdf) / [中文PDF](reports/report.zh.pdf) are printable.

## Reading and data map

| Content | Location |
| --- | --- |
| Hardware model, driver, SDKs, model and launch command | Sections 01–02; `protocols/environment.json` |
| Benchmark sources, business examples and methods | Sections 03–04 |
| Denominators, rejections and exclusions | Section 05 |
| Hardware, communication, engine, serving, cache and sustained results | Sections 06–09 |
| H200 reference, economics and conclusions | Sections 10–12 |
| Metric details, complete figures, code and evidence index | Sections 13–15 |

The local `data/` and `protocols/` directories hold controlled public measurements from October 19: 695 logical outcomes, comprising 692 complete source responses and three local rejections, with eight length conditions, five mixed-request pairs and three low-reuse cache pairs. Other serving, hardware and native-engine records reside in the [October 14–17 data directory](../2026-10-14_17-dual-spark/README.en.md). The report organizes findings by topic while preserving workload-specific measurement units and denominators.

## Offline recomputation and report rebuilding

After installing from the repository root:

```bash
sparkbench analyze-completion experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion --output outputs/completion
python -m pip install -e '.[reports]'
python experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/analysis/build_report.py experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion --pdf
```

Recomputation reads published scalar evidence without model or database calls. Rebuilding reads both measurement directories and presents results by topic; `analysis/build_report.py` is the entry point and `analysis/integrate_report.py` defines the structure and synthesis. Changed artifacts require a new archive; the original SHA256 list validates the published snapshot.

## Reproducing measurement conditions

1. Match the [model and deployment manifest](experiment.json): weight revision, engine version, TP, context and MTP parameters.
2. Read `protocols/public/manifest.json` and `protocols/cache/manifest.json`. Their `plans/*.jsonl` files contain inputs, output caps and pair identities; hashes are in `protocols/plan-hashes.json`.
3. Run each length condition individually, recording actual usage. For cache controls, run seed then probe with drained source-counter boundaries around each request.
4. Before each interference arm, prewarm all 32 short inputs, then replay the full frozen arrival offsets. Short prompts and offsets match across arms; the mixed arm adds eight long inputs. Preserve 4 short workers, 2 long workers and a queue of 32; check the 5-second expiration threshold on dequeue, so actual waiting before rejection may be longer.
5. Retain all arrivals, rejections, failures, retries, first-text times and source usage. Sample both nodes approximately every 5 seconds; back off at 85°C and resume below 75°C for 30 seconds. Record service changes under a new date.

Generic `sparkbench serve` supports individual length conditions and API calibration. Its uniform-arrival scheduler is not the frozen Poisson mixed trace used here. Exact pair reproduction requires the manifest offsets and worker partitions, followed by analyzer qualification. This repository does not modify production routing or install ingress isolation automatically.

## Data files

`data/requests.json` retains attempt-level scalars; `data/arrivals.json` retains arrival and admission outcomes; `data/blocks.json` contains counter boundaries; `data/telemetry.json` records resources. `protocols/environment.json` records each node's hardware model, driver, SDKs and redacted launch arguments. `analysis/completion-summary.json` contains qualification and exclusion reasons. Generated bodies and credentials are excluded from the scalar distribution; task and synthetic inputs retain their source language.

## One-condition rerun example

Set your own API key in `SPARKBENCH_API_KEY` and verify the configured endpoint, route and model. This command previews the 1K-input/16K-output-cap condition without sending a request:

```bash
sparkbench serve --config experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/configs/length-i1024-o16384.json --workloads experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/protocols/public/plans/gc19_length_output_axis_i1024_o16384.jsonl --output outputs/new-length-1k-16k
```

Append `--execute` to run, using a new output directory. The config max_tokens sets the cap. Requesting ignore_eos does not establish forwarding; verify your endpoint. Node-side thermal controls remain required; a public API client does not implement two-node protection. The full Poisson scheduler is in the [historical arrival source](../../../reference/2026-study/serving/arrival.py.txt), with this batch's offsets in its frozen manifest.
