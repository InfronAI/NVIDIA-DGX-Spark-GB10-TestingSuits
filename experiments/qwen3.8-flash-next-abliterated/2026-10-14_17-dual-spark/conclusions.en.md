[中文](conclusions.zh.md)

# Findings and applicability

| Area | Observation | Supported interpretation and limits |
| --- | --- | --- |
| Short-prompt concurrency | Concurrency1/2/4/8:45.1/78.7/120.2/158.9 output tokens/s; visible-content latency p95:1.65/1.78/1.79/2.39s | Matched complete, uninterrupted blocks without extra client attempts. Not all-time availability or a universal service commitment. |
| Unified-memory access |8GiB read:110.04/110.09GB/s; write:192.63/192.87GB/s; copy read+write:223.41/226.95GB/s (nodes0/1) | Logical bytes. GB10 uses LPDDR5x unified memory, not HBM; not physical DRAM transaction counts. |
| NVFP4 compute | Fixed8192³ dense synthetic GEMM:157.38/148.02TFLOP/s | Validated fixed-scale synthetic inputs; not complete MoE inference speed. |
| Long output | Native maximum260,096 tokens at about61.61 decode tokens/s | Forced length includes repeated whitespace; scheduling/capacity probe, not useful-text quality. |
| Continuous service |3600.107s,600 arrivals,599 complete,1 local rejection | One-hour observation; not all-day availability. |
| GPU/communication coverage |3740 valid base/extension rounds;840 paired collective trials;120 overlap trials | Retain node, shape, precision and timing scope; two ranks are not independent replicates. |
| Public-length gap |299 planned,291 sent,290 complete,8 unsent,1 incomplete | Complete responses do not imply input/output target attainment; zero passed the strict joint fixed-length criterion. |
| Mixed-interference gap |5 candidates,3 executed,2 not admitted,0 fully observed pairs | Mixed arms include locally rejected short requests; no reliable fully paired interference multiplier. |
| H200/economics |H200 specifications only; US$10,000 capital with explicit utilization/price ROI scenarios | No matched H200 workload, wall-meter energy or realized operating revenue. |

Use measured concurrency as a starting range and validate service targets on real business traffic. The engine's48-running-request limit does not establish a48-concurrent latency SLO. Next priorities: missing public-length cells, interference windows that fit their arrival schedules, wall energy and a matched H200 experiment.

Recompute from the repository root with `sparkbench summarize experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark --output outputs/summary`. Full definitions and measurement tables: [report](reports/report.en.md), [data](data/).
