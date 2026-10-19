[English](README.en.md) · [中文](README.zh.md)

# 2026-10-19: public lengths, mixed interference and 64K cache supplement

The same dual-DGX-Spark and SGLang configuration fills missing public conditions, complete arrival histories, short/long pairs and low-reuse cache counter windows. The October 14–17 experiment stays unchanged; new attempts, counter windows and conclusions are archived separately.

Qualified evidence: **8/8** lengths, **5/5** observed interference pairs (**4/5** effect-eligible) and **3/3** cache pairs. The plan fixes **695** logical requests including preflight and warmup.

Start with [conclusions](conclusions.en.md), then the [full English report](reports/report.en.md). Download [standalone HTML](reports/report.en.html) or [PDF](reports/report.en.pdf) for offline sharing. Historical hardware, engine, H200 architecture and economic chapters remain in the complete report.

## Offline recomputation and report rebuilding

After installing from the repository root:

```bash
sparkbench analyze-completion experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion --output outputs/completion
python -m pip install -e '.[reports]'
python experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/analysis/build_report.py experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion --pdf
```

Recomputation reads published scalar evidence without model or database calls. Rebuilding inherits historical chapters from the immutable 2026-10-14_17-dual-spark report and inserts this dated experiment. Changed artifacts require a new archive; the original SHA256 list validates the published snapshot.

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
