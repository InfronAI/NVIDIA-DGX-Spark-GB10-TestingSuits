### 2026-10-19: length, interference and cache protocol

Both nodes are measured in a dedicated window. The public path is Infron → Cloudflare → frozen-request forwarding proxy → SGLang. Only registered payloads are admitted, and both arms use the same path. Model, quantization and engine launch configuration match the archived deployment; the new date remains a separate experiment.


| Component | Fixed count | Controls |
| --- | --- | --- |
| Route preflight | 1 | 32-token cap; verify actual route and source parameters. |
| Public lengths | 8 | One observation for each formerly unsent input/output condition. |
| Interference warmup | 320 | 5 pairs × 2 arms × 32 short inputs; all inputs prewarmed with matching output parameters. |
| Measured short requests | 320 | 5 × 2 × 32; identical prompts and full Poisson schedules within each pair, nominal rate 0.12/s. |
| Mixed-arm long requests | 40 | Eight per pair; 32K inputs, 128-token caps, uniformly distributed over the short-request horizon. |
| Cache controls | 6 | Three independent pairs of 64K inputs; early unique markers set designed prefix overlap to 0%. |

The plan fixes **695 logical requests**. Warmup, retries and measurements are separate. Arm order alternates; short/long client workers are limited to 4/2 within the authorized total ceiling 8. All 32 short arrivals are retained without truncation. The local queue holds at most 32 requests. A 5-second expiration threshold is checked on dequeue, so observed waiting before rejection can exceed 5 seconds; rejection remains an outcome. Thermal backoff starts at 85°C and resumes below 75°C after30 stable seconds.

### Additional metric definitions and inclusion rules


| Metric | Definition and boundary |
| --- | --- |
| Arrival-to-visible wait | Client receipt of first visible text minus scheduled arrival; includes client scheduling, queuing and verified retries. |
| Median/p95 waiting | Linear-interpolation sample quantiles of 32 short requests per arm. The p95 estimate is tail-sensitive; no p99 claim is made. |
| Interference ratio | Mixed-arm median wait divided by matched isolated-arm median wait, only after all short requests, payloads and arrival schedules match. |
| Cached-token fraction | Source cached-token increment / input-token increment. Both boundaries must be idle; monotonic request, input and output counters must reconcile with the block. |
| Realized generation | completion_tokens from the original source usage. Public usage, billed usage and requested output cap remain distinct. |
| Replication unit | Five paired blocks. The 32 correlated requests within a block are not32 independent experiments. Pair results and ranges are reported without a pooled pseudo-replication confidence interval. |

Responses require normal termination, a stream-end marker, source usage and a reconciled counter window. Missing windows are unavailable; zero reuse is reported only with a valid denominator.
