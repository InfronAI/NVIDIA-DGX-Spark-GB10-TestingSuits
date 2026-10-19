[中文](reproduction.zh.md)

# Reproduction and new experiments

## Serving

Copy `configs/serving.example.json` and set your endpoint/model. Credentials come only from `SPARKBENCH_API_KEY`. Defaults:6 requests,concurrency1,128-token cap. JSONL rows require unique `id` and `messages`; the historical60-task bank is at the experiment's `protocols/protocol/workloads.jsonl`.

```bash
sparkbench serve --config configs/serving.example.json --workloads configs/smoke-workloads.jsonl
# Review the preview; execute into a new output directory.
sparkbench serve --config configs/serving.example.json --workloads configs/smoke-workloads.jsonl --output outputs/new-serving --execute
```

`arrival.example.json` uses independent fixed arrivals. Saturation records local rejection instead of silently converting to closed-loop queuing. Poisson arrivals and historical interference designs remain in reference implementations/protocols; this new fixed-arrival adapter does not claim exact reproduction of those schedules.

Only explicit Cloudflare HTTP failures receive bounded retries. Timeout/disconnect outcomes remain unknown and are not blindly replayed. Each attempt is retained; final success does not erase earlier failure. Deadlines never extend. `STOP` in the output directory stops new dispatch; in-flight calls drain under finite timeouts. Prompt/completion contents are not persisted by default; counts, timing, payload hashes and error categories are retained.

## Native engine and cache

`native.example.json` targets native SGLang `/generate`. JSONL must supply `input_ids` verified with the matching tokenizer; character count is not token count. `ignore_eos=true` is a forced-length workload. Actual token counts and engine phase fields are retained in `native_meta`. Non-streaming clients do not observe first-token arrival and never fabricate TTFT from total time.

Optional `metrics_url`/`counter_names` capture windows; credentials come from `SPARKBENCH_METRICS_KEY`. Track resets, label changes and unrelated concurrent traffic. Cache tests use identical complete prefixes and explicit seed/probe identity and waits. Profiling endpoints/trace exports vary by engine version: verify the installed version first. Historical automatic-stop trace validation is in `reference/2026-study/completion/`; the CLI does not automatically start remote profilers.

## GPU and two-node system

Execution requires dedicated Linux GB10 hardware and a verified installed CUDA/PyTorch environment. The suite does not install drivers, stop services, or change power/clock policy. Recorded library versions and kernel paths are in the full report. NVFP4 requires a matching FlashInfer/CUTLASS backend. Unsupported paths are recorded as unsupported, not zero throughput.

```bash
sparkbench hardware --mode base
sparkbench hardware --mode base --output outputs/base --execute
sparkbench hardware --mode extensions --output outputs/extensions --execute
sparkbench hardware --mode soak --output outputs/soak --duration-limit 36000 --execute
# Run on separate nodes; replace 192.0.2.10 with node0's test-network address.
sparkbench hardware --mode collectives --rank 0 --master-addr 192.0.2.10 --output outputs/nccl-r0 --execute
sparkbench hardware --mode collectives --rank 1 --master-addr 192.0.2.10 --output outputs/nccl-r1 --execute
```

`overlap` uses the same two-node launch pattern. `NCCL_*` environment determines communication and is recorded. Inspect NCCL logs to establish the actual transport. Single-node extensions do not automatically launch overlap. Historical FP64 and representative Nsight programs/protocols/data are included as references; they are not automatically executed by this CLI.

Every output directory must be new. Reserve at least16GiB or15% memory; cool at85°C and resume below75°C for30s, at most600s without extending the run. Collective arms terminate at the threshold; retry into a new paired run. `STOP_GPU` stops later submissions. Do not reuse historical service-stop coordinators on production.

This release validates CPU imports, configuration, SSE, retries, statistics, mock HTTP and offline reconstruction. It does not rerun the portable GPU adapters or certify arbitrary new CUDA environments.
