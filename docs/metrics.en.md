[中文](metrics.zh.md)

# Metric dictionary

| Metric | Definition / unit | Boundary |
| --- | --- | --- |
| First visible content | Client send to first nonempty visible delta, seconds | Reasoning-only deltas excluded; not GPU prefill |
| First generated event | First content/reasoning/tool event, seconds | Events may contain multiple tokens |
| End-to-end latency | Send to stream end, seconds | Includes queues, network, engine and terminal transmission |
| Mean subsequent-token time | (end−first generated)/(generated count−1), seconds/token | Approximate mean, not a per-token latency distribution |
| Aggregate throughput | Sum generated tokens / sum continuous-block wall times | Not an average of per-request rates |
| Goodput | Requests/tokens meeting declared latency targets per second | Targets, arrival denominator and failures must be visible |
| Offered/admitted/complete | Separate arrival, sent and complete counts | Local rejection remains in offered denominator |
| Engine stages | Process-local queue, prefill and decode durations | No subtraction of cross-process absolute clocks |
| GPU stage activity | Union of matching device-event intervals per node | Do not sum nested events or ranks |
| MTP acceptance | Native accept length, verify count and acceptance rate | Enabled configuration and measured acceptance are separate |
| Cached-token fraction | Cached/input tokens or same-window counter delta ratio | Prefix overlap is not hit rate; reset windows invalid |
| GEMM rate |2MNK×calls/CUDA interval/10¹²,TFLOP/s | FMA counts2; dense and sparse work distinct |
| Logical memory bandwidth | Declared read/write bytes/device interval/10⁹,GB/s | Copy read+write=2N; not physical DRAM bytes |
| Dependent access | Serial chain time/accesses,ns | Not all-thread throughput |
| Collective algorithm bandwidth | Defined message bytes/slower rank's matched time | AllGather message size means full output |
| Collective bus bandwidth | Algorithm bandwidth×operation/rank factor | Not NIC line rate or automatic RDMA proof |
| Compute/communication overlap | Same-round serial/overlap plus hidden fraction | Match work and modes; exclude TCP control time |
| Power/energy | Sensor W and integration over valid adjacent intervals,J | GPU/SoC sensors are not wall-outlet energy |
| Temperature/clocks/utilization | °C,MHz,% with sampling coverage | Experimental thermal guard is not evidence of driver throttling |
| Paid cost/ROI | Reconciled charges;1/3/5-year scenario cash flow | Missing charges are not zero; do not deduct depreciation twice |

Missing values are empty/null, never fabricated zero. Every result needs conditions, sample count and timing scope.
