[中文](README.zh.md)

# Dual DGX Spark · Qwen NVFP4 experiment

Measured October14–17,2026. Deployment: two GB10 nodes with128GB unified memory each, SGLang0.5.21,TP=2 and MTP enabled. Execution ended; some planned conditions lack valid evidence.

1. Start with [conclusions and scope](conclusions.en.md).
2. Read definitions, methods, coverage, results and limits in the [full report](reports/report.en.md); download [HTML](reports/report.en.html) or [PDF](reports/report.en.pdf).
3. Inspect the [frozen protocol](protocols/test-plan.en.md), [experiment record](experiment.json) and [checksummed artifact inventory](artifacts.json).

| Data | Measurement scope | Boundary |
| --- | --- | --- |
| `data/service` | Public streaming requests, continuous blocks, business/cache cases and cost scenarios | End-to-end includes network and queues; natural stopping is not fixed-length attainment |
| `data/hardware` | Unloaded-model compute, memory, operators, communication and sustained work | Library-call timing is not necessarily a single kernel or theoretical peak |
| `data/engine` | Native prefill/decode and GPU stage observations | Process-local durations and GPU intervals are distinct |
| `data/completion` |16 fixed outputs,9 profiling pairs, cache, public hour, FP64/counters | Independently dated; never backfills historical success rates |
| `data/coverage.json` | Final coverage and gaps | Execution termination is not complete evidence |

Fixed task bank:10 questions each from MMLU-Pro, MATH-500, GSM8K, IFEval, HumanEval+ and MBPP+,60 total. Eight business shapes and length/prefix designs provide additional workloads. Versions, sampling and content hashes are in `protocols/`.
