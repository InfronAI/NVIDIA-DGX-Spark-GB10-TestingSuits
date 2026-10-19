[English](conclusions.en.md) · [中文](conclusions.zh.md)

### Independent measurements on 2026-10-19


| Scope | Qualified evidence | Interpretation |
| --- | --- | --- |
| Public output caps | **8/8** | One natural-stop observation for each of eight formerly unsent conditions. |
| Long/short interference | **5/5** | All arrivals and outcomes recorded; 4 pairs support a complete-pair effect estimate, with 3 local rejections retained. |
| 64K zero designed overlap | **3/3** | Two requests per pair; actual reuse comes from attributable counter windows. |


The native engine demonstrated fixed output up to **260,096 tokens**. Public requests stop naturally, so the eight output-cap conditions do not establish fixed long-output attainment. Across qualified pairs, mixed/isolated short-request waiting is **1.18–3.37×** at the median and **4.54–14.68×** at p95. Dates remain separate and earlier failures are retained.

### New evidence, applicability and next work


| Dimension | Supported conclusion | Next work |
| --- | --- | --- |
| Public long output | 8/8 output-cap conditions have response evidence; actual lengths are separate. | Verify end-to-end fixed-generation parameter forwarding before a public fixed-length capacity study. |
| Mixed serving | 4/5 complete pairs; median-wait ratios 1.18–3.37, p95 ratios 4.54–14.68. | Use the same traces to compare separate short/long queues or prefill budgets, testing tail-latency improvement before extending rates and dates. |
| Prefix caching | 3/3 attributable 64K low-reuse controls. | Interpret alongside reuse/retention tests; add cross-period competition rather than inferring a TTL from low-reuse controls. |
| Hardware comparison and economics | GB10/H200 resources are functionally aligned; measurements, specifications and cost scenarios remain distinct. | Same-workload H200 data, a wall meter and real utilization are needed for measured inference price/performance. |

Completion is defined by the separate fixed denominators in section05. Physical DRAM-byte counts, exact token timestamps and a same-workload H200 comparison remain unmeasured; more ordinary requests do not resolve those instrument/resource limits.
