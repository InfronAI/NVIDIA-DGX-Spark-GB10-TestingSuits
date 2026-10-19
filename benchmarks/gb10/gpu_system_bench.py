"""First resident-model hardware batch: library compute, memory and operators.

No service mutation. Device intervals include any host-submission gaps between
library calls; they are not an instruction-throughput or profiler measurement.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

from gpu_system_guard import GIB, memory_check, sample_guard, utc, write_json


THERMAL_SETTINGS=None
THERMAL_CONTEXT='warmup'
THERMAL_WAIT_SECONDS=0.0

def work_guard(root):
    global THERMAL_WAIT_SECONDS
    if THERMAL_SETTINGS is None:return sample_guard(root)
    from thermal_backoff import wait_if_hot
    sample,waited=wait_if_hot(root,lambda:sample_guard(root),THERMAL_SETTINGS,THERMAL_CONTEXT)
    THERMAL_WAIT_SECONDS+=waited
    return sample

def emit(root, row):
    row = {'utc': utc(), 'node': os.uname().nodename, 'execution_pid': os.getpid(), **row}
    with (root / 'measurements.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n'); f.flush()
    write_json(root / 'measurement_latest.json', row)


def timed(torch, root, fn, minimum_s=1.0):
    wall_start=time.perf_counter(); backoff_start=THERMAL_WAIT_SECONDS
    work_guard(root)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    first = torch.cuda.Event(enable_timing=True); last = torch.cuda.Event(enable_timing=True)
    first.record(); fn(); last.record(); last.synchronize()
    one = max(first.elapsed_time(last) / 1000, .000001)
    batch = max(1, min(8192, math.ceil(.1 / one)))
    device_s = submit_s = 0.; calls = 0; started = time.perf_counter()
    while device_s < minimum_s:
        work_guard(root)
        first.record(); before = time.perf_counter()
        for _ in range(batch):
            fn()
        submit_s += time.perf_counter() - before
        last.record(); last.synchronize()
        device_s += first.elapsed_time(last) / 1000; calls += batch
    return {'cuda_interval_s': device_s, 'iterations': calls,
            'mean_device_interval_us': device_s / calls * 1e6,
            'host_submission_us_per_call': submit_s / calls * 1e6,
            'wall_s': time.perf_counter() - wall_start,
            'thermal_backoff_s': THERMAL_WAIT_SECONDS-backoff_start,
            'thermal_policy': THERMAL_SETTINGS,
            'timing_boundary': 'CUDA events around repeated library calls; host-submission gaps included; allocation/validation outside unless explicitly noted'}


def validation(actual, expected, tolerance):
    a = actual.detach().cpu().double(); b = expected.detach().cpu().double()
    err = a - b
    absolute = float(err.abs().max())
    normalized = float(err.square().sum().sqrt() / b.square().sum().sqrt().clamp_min(1e-12))
    finite = math.isfinite(absolute) and math.isfinite(normalized)
    return {'passed': finite and normalized <= tolerance,
            'sampled_elements': a.numel(), 'relative_l2_error': normalized if finite else None,
            'max_absolute_error': absolute if finite else None,
            'relative_l2_tolerance': tolerance, 'reference': 'CPU float64 on specified output subset'}


def cases():
    out = []
    for precision in ['fp32', 'tf32_allowed', 'fp16', 'bf16']:
        shapes = [(n, n, n) for n in [1024, 2048, 4096, 8192]]
        shapes += [(m, n, n) for m in [1, 8, 32, 128, 512] for n in [4096, 8192]]
        for m, n, k in shapes:
            out.append({'kind': 'gemm', 'precision': precision, 'shape': [m, n, k]})
    for mib in [256, 1024, 4096, 8192]:
        for kind in ['copy', 'triad']:
            for stride in [1, 2]:
                out.append({'kind': kind, 'working_set_bytes': mib * 1024 ** 2, 'stride': stride})
    for n in [2 ** k for k in range(10, 19)]:
        for kind in ['sum', 'softmax', 'rmsnorm']:
            out.append({'kind': kind, 'elements': n})
    for length in [128, 1024, 8192]:
        out.append({'kind': 'attention', 'query_length': length, 'kv_length': length})
    for length in [1024, 8192, 32768]:
        out.append({'kind': 'attention', 'query_length': 1, 'kv_length': length})
    out.extend([{'kind': 'graph_add', 'graph': x, 'elements': 1048576} for x in [False, True]])
    for c in out:
        c['condition_id'] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]
    return out


def construct(torch, c):
    kind = c['kind']; meta = {'dtype': 'float32', 'allocation': 'CUDA tensors in coherent system memory', 'sparsity': 'dense'}
    estimate = 256 * 1024 ** 2
    if kind == 'gemm':
        m, n, k = c['shape']; precision = c['precision']
        dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(precision, torch.float32)
        item = 2 if precision in ['fp16', 'bf16'] else 4
        estimate += (m*k + k*n + m*n) * item
    elif kind in ['copy', 'triad']:
        estimate += c['working_set_bytes']
    elif kind == 'attention':
        estimate += (2*c['query_length'] + 2*c['kv_length']) * 16 * 128 * 2
    else:
        estimate += c['elements'] * 32
    guard = memory_check(estimate)
    if not guard['allowed']:
        return None, None, {'state': 'skipped_memory_guard', 'memory_guard': guard}
    meta['memory_guard'] = guard
    if kind == 'gemm':
        torch.backends.cuda.matmul.allow_tf32 = precision == 'tf32_allowed'
        a = torch.randn(m, k, device='cuda', dtype=dtype); b = torch.randn(k, n, device='cuda', dtype=dtype)
        out = torch.empty(m, n, device='cuda', dtype=dtype)
        ref = a[:16].cpu().double() @ b[:, :16].cpu().double()
        fn = lambda: torch.mm(a, b, out=out)
        check = lambda: validation(out[:16, :16], ref, .03 if item == 2 else (.003 if precision == 'tf32_allowed' else 1e-5))
        meta.update(dtype=str(dtype), flops_per_call=2*m*n*k, backend='PyTorch matmul / cuBLAS selection; tensor instructions not profiler-verified')
    elif kind in ['copy', 'triad']:
        buffers = 2 if kind == 'copy' else 3
        n = c['working_set_bytes'] // (4 * buffers)
        a = torch.rand(n, device='cuda')[::c['stride']]
        b = torch.rand(n, device='cuda')[::c['stride']]
        out = b if kind == 'copy' else torch.empty(n, device='cuda')[::c['stride']]
        idx = torch.arange(256, device='cuda', dtype=torch.int64) * (len(a)-1) // 255
        ref = a[idx].cpu().double() if kind == 'copy' else a[idx].cpu().double() + 3*b[idx].cpu().double()
        fn = (lambda: out.copy_(a)) if kind == 'copy' else (lambda: torch.add(a, b, alpha=3, out=out))
        check = lambda: validation(out[idx], ref, 1e-6)
        meta.update(logical_bytes_per_call=a.numel()*4*buffers, backend='PyTorch copy/add', bandwidth_scope='logical bytes, not measured DRAM transactions', working_set_bytes=n*buffers*4)
    elif kind in ['sum', 'softmax', 'rmsnorm']:
        n = c['elements']; x = torch.randn(n, device='cuda'); cpu = x.cpu().double()
        if kind == 'sum':
            out = torch.empty((), device='cuda'); fn = lambda: torch.sum(x, dim=0, out=out); ref = cpu.sum()
        elif kind == 'softmax':
            out = torch.empty_like(x); fn = lambda: torch.softmax(x, dim=0, out=out); ref = torch.softmax(cpu, dim=0)
        else:
            out = [None]
            def fn():
                out[0] = torch.nn.functional.rms_norm(x, (n,), eps=1e-6)
            ref = cpu / (cpu.square().mean() + 1e-6).sqrt()
            meta['output_allocation'] = 'library-returned output allocation inside each timed call'
        check = lambda: validation(out[0] if isinstance(out, list) else out, ref, 1e-4)
        meta.update(backend='PyTorch standalone library operator; not necessarily a single kernel')
    elif kind == 'attention':
        from torch.nn.attention import sdpa_kernel, SDPBackend
        q = torch.randn(1, 16, c['query_length'], 128, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, 16, c['kv_length'], 128, device='cuda', dtype=torch.bfloat16); v = torch.randn_like(k)
        ref = torch.softmax(q[0, 0, :4].cpu().double() @ k[0, 0].cpu().double().T / math.sqrt(128), dim=-1) @ v[0, 0].cpu().double()
        out = [None]
        def fn():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out[0] = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        check = lambda: validation(out[0][0, 0, :4], ref, .03)
        meta.update(dtype='bfloat16', backend='PyTorch Flash SDPA, forced; no math fallback', heads=16, head_dim=128, causal=False, output_allocation='inside library call')
    elif kind == 'graph_add':
        x = torch.randn(c['elements'], device='cuda'); y = torch.randn_like(x); out = torch.empty_like(x)
        ref = x[:256].cpu().double() + y[:256].cpu().double()
        fn = lambda: torch.add(x, y, out=out)
        if c['graph']:
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for _ in range(5): fn()
            stream.synchronize(); graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): fn()
            fn = graph.replay
        check = lambda: validation(out[:256], ref, 1e-6)
        meta.update(backend='CUDA Graph replay' if c['graph'] else 'ordinary PyTorch launch')
    return fn, check, meta


def warmup(torch, root):
    c = {'kind': 'gemm', 'precision': 'bf16', 'shape': [2048, 2048, 2048]}
    fn, check, meta = construct(torch, c)
    if fn is None:
        raise RuntimeError('insufficient memory for controlled warmup')
    start = window = time.monotonic(); rates = []; temps = []; windows = []; stable = False
    while time.monotonic() - start < 900:
        d = timed(torch, root, fn, 1)
        rates.append(meta['flops_per_call'] * d['iterations'] / d['cuda_interval_s'])
        temps.append(work_guard(root)['gpu']['temperature_c'])
        if time.monotonic() - window >= 60:
            windows.append({'mean_flops_s': statistics.mean(rates), 'mean_temperature_c': statistics.mean(temps)})
            emit(root, {'kind': 'warmup_window', 'elapsed_s': time.monotonic()-start, **windows[-1]})
            rates = []; temps = []; window = time.monotonic()
        if time.monotonic()-start >= 300 and len(windows) >= 3:
            w = windows[-3:]; speeds = [x['mean_flops_s'] for x in w]; ts = [x['mean_temperature_c'] for x in w]
            stable = (max(speeds)-min(speeds))/statistics.mean(speeds) <= .03 and max(ts)-min(ts) <= 2
            if stable: break
    result = {'warmup_elapsed_s': time.monotonic()-start, 'steady_by_engineering_rule': stable, 'validation': check()}
    emit(root, {'kind': 'warmup_result', **result})
    if not result['validation']['passed']: raise RuntimeError('warmup numerical validation failed')
    del fn, check; gc.collect(); torch.cuda.empty_cache()
    return result


def run_single(torch, root):
    warm = warmup(torch, root)
    all_cases = cases(); terminal = {'ok': 0, 'skipped': 0}; completed=set()
    path=root/'measurements.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            prior=json.loads(line)
            if not prior.get('condition_id') or prior.get('suite')=='extensions':continue
            if prior.get('state')=='measured' and prior.get('validation',{}).get('passed'):
                completed.add((prior['condition_id'],prior['round']));terminal['ok']+=1
            elif prior.get('state')=='skipped_memory_guard':
                completed.add((prior['condition_id'],prior['round']));terminal['skipped']+=1
            elif prior.get('state')=='condition_failed' and prior.get('error')!='exclusive GPU lease absent or expired':
                raise RuntimeError('prior non-lease condition failure requires review')
    write_json(root/'single_resume_coverage.json',{'utc':utc(),'completed_keys_preserved':len(completed),'pid':os.getpid()})
    for repeat in range(10):
        ordered = list(all_cases); random.Random(20261015+repeat).shuffle(ordered)
        for c in ordered:
            if (c['condition_id'],repeat+1) in completed:continue
            sample_guard(root)
            torch.manual_seed(20261015 + repeat * 1000 + int(c['condition_id'][:6], 16))
            record = {**c, 'round': repeat+1, 'warmup_state': warm, 'start_utc': utc()}
            fn = check = None
            try:
                fn, check, meta = construct(torch, c); record.update(meta)
                if fn is None:
                    terminal['skipped'] += 1
                else:
                    fn(); torch.cuda.synchronize(); v = check()
                    if not v['passed']: raise RuntimeError('numerical validation failed before timing')
                    record.update(timed(torch, root, fn)); record['validation'] = check()
                    if not record['validation']['passed']: raise RuntimeError('numerical validation failed after timing')
                    record['state'] = 'measured'; terminal['ok'] += 1
                    if 'flops_per_call' in meta:
                        record['effective_tflops'] = meta['flops_per_call'] * record['iterations'] / record['cuda_interval_s'] / 1e12
                    if 'logical_bytes_per_call' in meta:
                        record['effective_gb_s'] = meta['logical_bytes_per_call'] * record['iterations'] / record['cuda_interval_s'] / 1e9
            except (RuntimeError, NotImplementedError) as exc:
                record.update(state='condition_failed', error=str(exc)[:600]); emit(root, record)
                # Unsupported kernels may be skipped; numeric/guard/OOM failures stop the suite.
                if any(x in str(exc).lower() for x in ['no available kernel', 'not supported', 'not implemented']):
                    terminal['skipped'] += 1
                else:
                    raise
            else:
                record['end_utc'] = utc(); emit(root, record)
            finally:
                fn = check = None; gc.collect(); torch.cuda.empty_cache()
    return {'rounds': 10, 'conditions': len(all_cases), **terminal}


def run_soak(torch, root):
    global THERMAL_CONTEXT
    THERMAL_CONTEXT='warmup'
    warm = warmup(torch, root)
    compute = {'kind': 'gemm', 'precision': 'bf16', 'shape': [2048, 2048, 2048]}
    bandwidth = {'kind': 'triad', 'working_set_bytes': 256*1024**2, 'stride': 1}
    summary = []
    from soak_protocol import schedule
    for phase, seconds in schedule(root):
        selected = [compute] if phase == 'compute' else [bandwidth] if phase == 'bandwidth' else [compute, bandwidth]
        funcs = [construct(torch, c) for c in selected]
        if any(fn is None for fn, check, meta in funcs): raise RuntimeError('soak working-set memory guard')
        # Establish a workload-specific five-minute reference before the
        # nominal 60/60/120-minute measurement, preserving every sample.
        THERMAL_CONTEXT=phase+'_baseline'
        baseline_start=time.monotonic(); baseline_index=0
        while time.monotonic()-baseline_start < 300:
            fn,check,meta=funcs[baseline_index % len(funcs)]
            d=timed(torch,root,fn,1);v=check()
            if not v['passed']:raise RuntimeError('soak baseline numerical validation failed')
            emit(root,{'kind':'soak_baseline','phase':phase,'work':selected[baseline_index % len(funcs)]['kind'],'elapsed_s':time.monotonic()-baseline_start,'validation':v,**d,**meta})
            baseline_index+=1
        THERMAL_CONTEXT=phase
        start = time.monotonic(); i = 0
        while time.monotonic() - start < seconds:
            fn, check, meta = funcs[i % len(funcs)]
            d = timed(torch, root, fn, 1); v = check()
            if not v['passed']: raise RuntimeError('soak numerical validation failed')
            emit(root, {'kind': 'sustained', 'phase': phase, 'work': selected[i % len(funcs)]['kind'], 'elapsed_s': time.monotonic()-start, 'warmup_state': warm, 'validation': v, **d, **meta})
            i += 1
        summary.append({'phase': phase, 'elapsed_s': time.monotonic()-start, 'windows': i})
        funcs = []; fn = check = None; gc.collect(); torch.cuda.empty_cache()
    return {'phases': summary}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--mode', choices=['single', 'soak'], required=True)
    args = ap.parse_args(); root = Path(args.root)
    global THERMAL_SETTINGS
    if args.mode=='soak':
        from thermal_backoff import policy
        THERMAL_SETTINGS=policy(root)
    sample_guard(root)
    try:
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.cuda.set_device(0)
        sample_guard(root)
        props = torch.cuda.get_device_properties(0)
        metadata = {'utc': utc(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'device': str(props), 'capability': list(torch.cuda.get_device_capability()),
                    'nccl': list(torch.cuda.nccl.version()), 'mode': args.mode,
                    'scope': 'resident model, inference drained; no server settings changed',
                    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        write_json(root / ('metadata_'+args.mode+'.json'), metadata)
        with torch.inference_mode():
            result = run_single(torch, root) if args.mode == 'single' else run_soak(torch, root)
        write_json(root / (args.mode+'_result.json'), {'state': 'complete', 'utc': utc(), **result})
    except BaseException as exc:
        write_json(root / (args.mode+'_result.json'), {'state': 'stopped', 'utc': utc(), 'error': str(exc)[:1000]})
        raise


if __name__ == '__main__':
    main()
