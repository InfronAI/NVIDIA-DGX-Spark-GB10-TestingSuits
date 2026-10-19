[English](conclusions.en.md) · [中文](conclusions.zh.md) · [Full report / 完整报告](reports/report.en.md)

# Findings and applicability

This study measures dual-DGX-Spark compute, memory, communication, engine behavior and public serving within one measurement framework. Deployment decisions require throughput, user waiting, cache conditions, thermals and economic assumptions together.

| Dimension | Principal finding | Scope |
| --- | --- | --- |
| GPU compute | Dense 8,192³ GEMM: BF16 **94.08–94.34**, FP8 **191.88–192.64 TFLOP/s**. | Precision-, shape- and implementation-specific; not inference throughput. |
| Memory supply and communication | 8 GiB streaming copy: **223–227 GB/s**; 1 GiB AllReduce bus bandwidth: **21.81 GB/s**. | Copy counts logical reads plus writes; NCCL and host-buffer RDMA remain separate. |
| Engine stages | Concurrency-one prefill: **2,427–3,354 tokens/s**; decode with 256 outputs: **49.8–60.2 tokens/s**. | Native synthetic workloads; timings include engine scheduling. |
| Public concurrency | Concurrency 1→8: **45.1→158.9 tokens/s**, first-content p95 **1.65→2.39 s**. | The 60-prompt bank and matched complete blocks; not a service guarantee. |
| Long input and output | Native fixed output reaches **260,096 tokens**; the public 64K-input observation generates **27,018 tokens**. | Public generation stops naturally; realized output and max_tokens are separate. |
| Mixed-request scheduling | Four of five complete-arrival-trace pairs support effects: short-request median wait **1.18–3.37×**, p95 **4.54–14.68×**. | Mixed arm 5 locally rejects 3/32 short arrivals; completed-subset latency is excluded from complete-pair effects. |
| Prefix caching | All six 64K low-reuse windows have **0%** cached tokens; the 32K nominal 300-second idle cohort reaches **91.7%**, and six 8K periodic reuse rounds are about **100%**. | Content, idle gaps and pressure differ by experiment; no universal TTL or long-term business hit rate follows. |
| Sustained service and thermal state | Public serving: **3,600.1 seconds**, 599 complete and one local rejection out of 600 arrivals; each node has **60/60/120-minute** compute/bandwidth/mixed phases. | Separate phases; mixed work includes thermal backoff and does not establish four-hour continuous serving reliability. |
| H200 reference | H200 specifies greater local-memory bandwidth and different device links, precision support and power boundaries. | An official-specification reference without matched H200 inference measurements. |

### Economics and deployment trade-offs

| Use | Scenario result | Assumptions |
| --- | --- | --- |
| Self-use | **5.48-year** payback; five-year ROI **−8.7%**. | US$10,000 hardware; short prompts at concurrency 8; 75% productive utilization; estimated electricity, other operating costs excluded. |
| External sales | **2.65-year** payback; five-year ROI **+88.5%**. | 75% sold utilization; twice the blended API reference price; not observed revenue. |

### Limitations and future work

| Direction | Next investigation | Evidence required |
| --- | --- | --- |
| Mixed-serving scheduling | Compare separate short/long queues and prefill budgets. | Identical complete arrival traces; report tails, throughput and local rejections together. |
| Public fixed long output | Verify end-to-end forwarding of forced-generation controls. | Realized target lengths with matched inputs and concurrency; natural stopping remains separate. |
| Caching and sustained behavior | Extend competing business traffic, independent dates and ambient temperatures. | Attributable counters, repeated blocks, cooling fraction and goodput. |
| Hardware mechanisms and H200 | Obtain physical-memory traffic and matched H200 configurations. | Matched model, precision, lengths, cache state and communication paths. |
| Observed ownership cost | Replace estimates with wall-meter energy, real utilization and operating expenses. | Recompute 1/3/5-year cash flows, payback and ROI. |
