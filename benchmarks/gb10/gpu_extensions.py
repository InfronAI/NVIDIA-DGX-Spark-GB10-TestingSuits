"""Finite guarded hardware extensions. New native kernels and installed libraries.

No package, driver, service, power-policy or system profiler-permission changes.
"""
import argparse,ctypes as C,gc,hashlib,json,math,os,random,signal,subprocess,sys,time
from pathlib import Path
from gpu_system_guard import sample_guard,memory_check,utc,write_json
from gpu_system_bench import emit,timed,validation,construct,warmup


def conditions():
    rows=[]
    for threads in [128,256,512]:
        for mult in [1,4,8]:
            for chains in [1,8]:rows.append(dict(kind='scalar_fma',threads=threads,sm_multiplier=mult,chains=chains,loops=32768//chains))
    for exp in range(12,31):
        for hint in ['ca','cg']:rows.append(dict(kind='pointer_chase',address_span_bytes=2**exp,cache_hint=hint,steps=65536))
    for stride in [1,2,4,8,16,32]:rows.append(dict(kind='shared_memory',stride=stride,loops=4096))
    for mib in [256,1024,4096,8192]:
        for mode in ['read','write']:rows.append(dict(kind='native_bandwidth',mode=mode,working_set_bytes=mib*1024**2))
    for graph in [False,True]:rows.append(dict(kind='empty_launch',graph=graph))
    for n in [1,2,4,8]:
        for mode in ['serial','parallel']:rows.append(dict(kind='streams',streams=n,mode=mode,matrix_n=1024))
    for workers in [0,1,4,10,20]:
        for mode in (['none'] if workers==0 else ['memory','compute']):rows.append(dict(kind='cpu_contention',workers=workers,cpu_mode=mode,working_set_bytes=256*1024**2))
    rows.append(dict(kind='managed_access',working_set_bytes=256*1024**2))
    shapes=[(n,n,n) for n in [1024,2048,4096,8192]]+[(m,n,n) for m in [1,8,32,128,512] for n in [4096,8192]]
    for dtype in ['fp8_e4m3','nvfp4','fp16_sparse_2of4','bf16_sparse_2of4']:
        for shape in shapes:rows.append(dict(kind='low_precision',precision=dtype,shape=list(shape)))
    for r in rows:
        r['rounds']=10 if r['kind']=='empty_launch' or (r['kind']=='low_precision' and r['shape'] in [[4096]*3,[1,4096,4096],[128,4096,4096]]) else 5
        r['condition_id']=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()[:16]
    return rows


def library(root):
    nvcc='/usr/local/cuda/bin/nvcc'; cmd=[nvcc,'-O3','-std=c++17','-arch=sm_121','--shared','-Xcompiler','-fPIC','-lineinfo',str(root/'kernels.cu'),'-o',str(root/'kernels.so')]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    write_json(root/'native_build.json',dict(utc=utc(),argv=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    if p.returncode:raise RuntimeError('native kernel compilation failed')
    with (root/'kernels.sass.txt').open('w') as f:subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(root/'kernels.so')],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=30)
    sass=(root/'kernels.sass.txt').read_text();assert 'FFMA' in sass,'scalar FMA missing in compiled instructions'
    lib=C.CDLL(str(root/'kernels.so'));ptr=C.c_void_p;i=C.c_int;sz=C.c_size_t
    for name,args in {'launch_fma':[ptr,ptr,i,i,i,i,ptr],'launch_chase':[ptr,ptr,ptr,i,i,ptr],'launch_shared':[ptr,i,i,i,ptr],'launch_read':[ptr,ptr,sz,i,ptr],'launch_write':[ptr,sz,i,ptr],'launch_empty':[ptr],'managed_alloc':[C.POINTER(ptr),sz],'managed_free':[ptr],'fma_occupancy':[i,i,C.POINTER(i)]}.items():
        getattr(lib,name).argtypes=args;getattr(lib,name).restype=i
    probe_cmd=[nvcc,'-O3','-std=c++17','-arch=sm_121','-DPROBE_MAIN',str(root/'kernels.cu'),'-o',str(root/'counter_probe')]
    probe=subprocess.run(probe_cmd,capture_output=True,text=True,timeout=180)
    counter={'build_returncode':probe.returncode,'build_stderr':probe.stderr,'scope':'diagnostic only; not formal timing'}
    if probe.returncode==0:
        cmd=['/usr/local/cuda/bin/ncu','--csv','--metrics','sm__warps_active.avg.pct_of_peak_sustained_active,lts__t_sector_hit_rate.pct,dram__bytes_read.sum,dram__bytes_write.sum','--launch-count','1',str(root/'counter_probe')]
        try:
            probe=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
            counter.update(argv=cmd,returncode=probe.returncode,stdout=probe.stdout,stderr=probe.stderr)
        except subprocess.TimeoutExpired:counter['state']='diagnostic_timeout'
    write_json(root/'counter_probe.json',counter)
    return lib


def checked(code):
    if code:raise RuntimeError('CUDA runtime code '+str(code))


def invalid_harness_probe(row):
    # The installed sparse dispatch supports aten.t, not numpy_T (.T).
    # These exact failed API probes contain no valid timing and do not show
    # unsupported hardware. Raw rows remain intact; corrected .t() is retested.
    return (row.get('suite')=='extensions' and row.get('state')=='unsupported_probe'
            and row.get('precision','').endswith('sparse_2of4')
            and 'numpy_T.default' in row.get('error',''))


def invalid_legacy_managed_validation(row):
    return (row.get('suite')=='extensions' and row.get('kind')=='managed_access' and row.get('state')=='measured'
            and row.get('execution_pid') in [278387,3606624])


def completed_keys(root):
    """Resume audited environment interruptions without replaying valid rounds."""
    p=root/'measurements.jsonl';done=set()
    for line in p.read_text().splitlines() if p.exists() else []:
        row=json.loads(line)
        if row.get('suite')!='extensions':continue
        if invalid_legacy_managed_validation(row):continue
        key=(row['condition_id'],row['round'])
        if row.get('state')=='condition_failed':
            raise RuntimeError('failed extension measurement requires review before continuation')
        if row.get('state')=='measured':
            if not row.get('validation',{}).get('passed'):raise RuntimeError('invalid measured row')
            if key in done:raise RuntimeError('duplicate extension round')
            done.add(key)
    return done


def cpu_worker(root,core,mode):
    # A separate process keeps the memory streamer independent of the GIL.
    os.sched_setaffinity(0,{core})
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import numpy as np
    parent=os.getppid();a=np.ones(4*1024*1024,dtype=np.float32) if mode=='memory' else np.ones((64,64),dtype=np.float32)
    b=np.ones_like(a)
    while os.getppid()==parent:
        sample_guard(root)
        until=time.monotonic()+.1
        while time.monotonic()<until:
            if mode=='memory':np.add(a,1,out=b);np.add(b,-1,out=a)
            else:np.matmul(a,b,out=b);b.fill(1)


def make(torch,np,lib,root,c):
    kind=c['kind'];sm=torch.cuda.get_device_properties(0).multi_processor_count;stream=torch.cuda.current_stream().cuda_stream
    cleanup=lambda:None;meta={'allocation':'cuda tensors, coherent system RAM','timing_boundary':'CUDA events around repeated submissions, including host gaps','sparsity':'dense'}
    required=512*1024**2
    if kind in ['native_bandwidth','managed_access']:required+=c['working_set_bytes']
    if kind=='pointer_chase':required+=int(c['address_span_bytes']*1.25)
    if kind=='cpu_contention':required+=c['working_set_bytes']+c['workers']*32*1024**2
    if kind=='low_precision':
        m,n,k=c['shape'];required+=(m*k+n*k)*12+m*n*2
    guard=memory_check(required);meta['memory_guard']=guard
    if not guard['allowed']:return None,None,meta,cleanup
    if kind=='scalar_fma':
        blocks=sm*c['sm_multiplier'];nt=blocks*c['threads'];nc=c['chains'];loops=c['loops']
        out=torch.empty(nt*nc,device='cuda');cycles=torch.empty(nt,device='cuda',dtype=torch.int64)
        def fn():checked(lib.launch_fma(out.data_ptr(),cycles.data_ptr(),blocks,c['threads'],nc,loops,stream))
        x=np.array([[np.float32(.01*(j+1))+np.float32((i%17)*.001) for j in range(nc)] for i in range(32)],dtype=np.float64)
        for _ in range(loops):x=x*float(np.float32(1.000001))+float(np.float32(.000001))
        ref=torch.tensor(x.reshape(-1),dtype=torch.float64)
        def check():return validation(out[:32*nc],ref,.002)
        active=C.c_int();checked(lib.fma_occupancy(c['threads'],nc,C.byref(active)))
        meta.update(flops_per_call=2*nt*nc*loops,blocks=blocks,active_blocks_per_sm_limit=active.value,occupancy_scope='CUDA theoretical residency limit; not measured active occupancy',compiled_instruction='FFMA in archived SASS')
    elif kind=='pointer_chase':
        n=c['address_span_bytes']//128;order=np.arange(n,dtype=np.int32);np.random.default_rng(20261016).shuffle(order)
        p=torch.empty(c['address_span_bytes']//4,device='cuda',dtype=torch.int32)
        idx=torch.from_numpy(order.astype(np.int64)*32).cuda();vals=torch.from_numpy(np.roll(order,-1)).cuda();p[idx]=vals;del idx,vals
        state=torch.tensor([int(order[0])],device='cuda',dtype=torch.int32);cycles=torch.empty(1,device='cuda',dtype=torch.int64);calls=[0]
        def fn():
            checked(lib.launch_chase(p.data_ptr(),state.data_ptr(),cycles.data_ptr(),c['steps'],int(c['cache_hint']=='cg'),stream));calls[0]+=1
        # Traverse the entire cycle once before formal timing. The timed walk
        # carries state across calls instead of repeatedly touching a small prefix.
        for _ in range(math.ceil(n/c['steps'])):fn()
        torch.cuda.synchronize()
        def check():return {'passed':int(state.item())==int(order[(calls[0]*c['steps'])%n]),'reference':'seeded single random cycle; exact endpoint','sampled_elements':1,'last_device_cycles_per_access':float(cycles.item())/c['steps']}
        meta.update(accesses_per_call=c['steps'],cache_line_spacing_bytes=128,active_words=n,cache_hit_rate=None,cache_scope='address-span sweep with explicit PTX ca/cg hint; no inferred hit rate')
    elif kind=='shared_memory':
        out=torch.empty(sm*256,device='cuda');ref=torch.tensor([float(i%32+c['loops']) for i in range(256)])
        def fn():checked(lib.launch_shared(out.data_ptr(),sm,c['stride'],c['loops'],stream))
        check=lambda:validation(out[:256],ref,0)
        meta.update(logical_bytes_per_call=sm*256*c['loops']*8,memory_scope='logical shared-memory reads+writes; conflict count requires counters')
    elif kind=='native_bandwidth':
        n=c['working_set_bytes']//4;x=torch.ones(n,device='cuda');out=torch.empty(sm*256,device='cuda');blocks=sm
        if c['mode']=='read':
            def fn():checked(lib.launch_read(x.data_ptr(),out.data_ptr(),n,blocks,stream))
            ref=torch.tensor([max(0,(n-1-i)//(blocks*256)+1) for i in range(blocks*256)],dtype=torch.float64)
            check=lambda:validation(out,ref,0)
        else:
            def fn():checked(lib.launch_write(x.data_ptr(),n,blocks,stream))
            check=lambda:{'passed':bool((x[::max(1,n//1024)]==3).all().item()),'sampled_elements':len(x[::max(1,n//1024)])}
        meta.update(logical_bytes_per_call=n*4,bandwidth_scope='logical payload bytes, not physical DRAM transactions')
    elif kind=='empty_launch':
        def fn():checked(lib.launch_empty(stream))
        if c['graph']:
            graph=torch.cuda.CUDAGraph();side=torch.cuda.Stream()
            with torch.cuda.stream(side):checked(lib.launch_empty(side.cuda_stream))
            side.synchronize()
            with torch.cuda.graph(graph):checked(lib.launch_empty(torch.cuda.current_stream().cuda_stream))
            fn=graph.replay
        check=lambda:{'passed':True,'reference':'CUDA error and synchronization status; empty kernel has no numerical output'}
    elif kind=='streams':
        n=c['matrix_n'];num=c['streams'];xs=[torch.randn(n,n,device='cuda',dtype=torch.bfloat16) for _ in range(num)];ys=[torch.randn_like(x) for x in xs];outs=[torch.empty_like(x) for x in xs]
        refs=[x[:16].cpu().double()@y[:,:16].cpu().double() for x,y in zip(xs,ys)];streams=[torch.cuda.Stream() for _ in range(num)];current=torch.cuda.current_stream()
        def fn():
            if c['mode']=='serial':
                for x,y,o in zip(xs,ys,outs):torch.mm(x,y,out=o)
            else:
                for st,x,y,o in zip(streams,xs,ys,outs):
                    st.wait_stream(current)
                    with torch.cuda.stream(st):torch.mm(x,y,out=o)
                for st in streams:current.wait_stream(st)
        def check():
            vals=[validation(o[:16,:16],ref,.03) for o,ref in zip(outs,refs)]
            return {'passed':all(v['passed'] for v in vals),'components':vals}
        meta.update(flops_per_call=num*2*n**3,comparison='serial/parallel pairs hold total matrix count fixed; stream-count changes total work')
    elif kind=='cpu_contention':
        fn,check,base=construct(torch,{'kind':'triad','working_set_bytes':c['working_set_bytes'],'stride':1});meta.update(base)
        cores=sorted(os.sched_getaffinity(0))[:c['workers']];children=[]
        def cleanup():
            for p in children:
                if p.poll() is None:p.terminate()
            for p in children:
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        try:
            for core in cores:children.append(subprocess.Popen([sys.executable,__file__,'--root',str(root),'--cpu-worker',str(core),'--cpu-mode',c['cpu_mode']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
            for _ in range(30):sample_guard(root);time.sleep(.1)
            if any(p.poll() is not None for p in children):raise RuntimeError('CPU contender exited')
        except BaseException:cleanup();raise
        meta.update(cpu_affinity=cores,cpu_workers=len(children),cpu_worker_processes=True)
    elif kind=='managed_access':
        p=C.c_void_p();checked(lib.managed_alloc(C.byref(p),c['working_set_bytes']));n=c['working_set_bytes']//4
        host=np.ctypeslib.as_array(C.cast(p,C.POINTER(C.c_float)),shape=(n,));host.fill(1);cuda_ms=[]
        def fn():checked(lib.launch_write(p.value,n,sm,stream))
        for _ in range(2):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();fn();end.record();end.synchronize();cuda_ms.append(start.elapsed_time(end))
        cpu_ms=[]
        for _ in range(2):
            start=time.perf_counter();total=float(host.sum(dtype=np.float64));cpu_ms.append((time.perf_counter()-start)*1000)
        def check():
            torch.cuda.synchronize();observed=float(host.sum(dtype=np.float64))
            return {'passed':observed==3*n,'reference':'fresh full managed allocation float64 CPU sum after GPU write','checked_elements':n}
        cleanup=lambda:checked(lib.managed_free(p))
        meta.update(allocation='cudaMallocManaged; CPU initialized, GPU write first/repeat then CPU read first/repeat',gpu_first_repeat_ms=cuda_ms,cpu_first_repeat_ms=cpu_ms,logical_bytes_per_call=n*4)
    elif kind=='low_precision':
        m,n,k=c['shape'];out=torch.empty(m,n,device='cuda',dtype=torch.bfloat16)
        if c['precision']=='fp8_e4m3':
            a=torch.randn(m,k,device='cuda').to(torch.float8_e4m3fn);b=torch.randn(n,k,device='cuda').to(torch.float8_e4m3fn);scale=torch.ones((),device='cuda');ref=a[:16].float().cpu().double()@b[:16].float().cpu().double().T
            box=[None]
            def fn():box[0]=torch._scaled_mm(a,b.T,scale_a=scale,scale_b=scale,out_dtype=torch.bfloat16)
            check=lambda:validation(box[0][:16,:16],ref,.03)
            meta.update(backend='PyTorch _scaled_mm, FP8 E4M3 operands, BF16 output',reference='same quantized operands, CPU float64',output_allocation='inside timed library call')
        elif c['precision'].endswith('sparse_2of4'):
            dtype=torch.float16 if c['precision'].startswith('fp16') else torch.bfloat16
            a=torch.randn(m,k,device='cuda',dtype=dtype);b=torch.randn(n,k,device='cuda',dtype=dtype);b.reshape(n,k//4,4)[:,:,2:]=0
            ref=a[:16].cpu().double()@b[:16].cpu().double().T
            from torch.sparse import to_sparse_semi_structured
            before=time.perf_counter();sparse=to_sparse_semi_structured(b);torch.cuda.synchronize();meta['sparse_compression_wall_s']=time.perf_counter()-before
            box=[None]
            def fn():box[0]=torch.mm(a,sparse.t())
            check=lambda:validation(box[0][:16,:16],ref,.03)
            meta.update(backend=type(sparse).__name__,sparsity='structured 2:4',flop_scope='dense-equivalent 2MNK; nonzero arithmetic MNK',nonzero_flops_per_call=m*n*k,output_allocation='inside timed library call')
        else:
            from flashinfer import nvfp4_quantize,mm_fp4
            from flashinfer.quantization.fp4_quantization import SfLayout
            # Exact E2M1 inputs with a forced +/-6 in each 16-value block give
            # unity block scales. This isolates GEMM correctness from rounding.
            lut=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda',dtype=torch.bfloat16)
            a=lut[torch.randint(0,16,(m,k),device='cuda')];b=lut[torch.randint(0,16,(n,k),device='cuda')];a[:,::16]=6;b[:,::16]=6
            scale=torch.ones(1,device='cuda');before=time.perf_counter();aq,sa=nvfp4_quantize(a,scale,sfLayout=SfLayout.layout_128x4,do_shuffle=False);bq,sb=nvfp4_quantize(b,scale,sfLayout=SfLayout.layout_128x4,do_shuffle=False);torch.cuda.synchronize()
            meta['quantize_and_pack_wall_s']=time.perf_counter()-before
            def unpack(q):
                bits=torch.stack((q&15,q>>4),dim=-1).reshape(q.shape[0],-1).long();return lut[bits]
            if not bool((unpack(aq)==a).all().item() and (unpack(bq)==b).all().item()):raise RuntimeError('NVFP4 exact-input quantization validation failed')
            # All logical blocks have max(abs(x))=6 and the global scale is one.
            # Build the known unit-scale buffer explicitly, including padding:
            # padded quantizer storage is not a numerical output to validate.
            sa.fill_(56);sb.fill_(56);torch.cuda.synchronize()
            if not bool((sa==56).all().item() and (sb==56).all().item()):raise RuntimeError('NVFP4 explicit unity scales not established')
            meta['block_scale_construction']='all logical and padding E4M3 bytes set to 56 (unity), after exact packed-code check'
            ref=a[:16].cpu().double()@b[:16].cpu().double().T
            def fn():mm_fp4(aq,bq.T,sa,sb.T,scale,torch.bfloat16,out,backend='cutlass')
            check=lambda:validation(out[:16,:16],ref,.03)
            meta.update(backend='FlashInfer explicit CUTLASS NVFP4; unity block/global scales; BF16 output',reference='same exactly representable quantized operands, CPU float64',quantization_scope='controlled unity-scale operand distribution; not a quantization-quality or production GEMM benchmark')
        meta['flops_per_call']=2*m*n*k
    else:raise ValueError(kind)
    return fn,check,meta,cleanup


def run(root,rank):
    import numpy as np,torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
    sample_guard(root);lib=library(root)
    write_json(root/'extensions_environment.json',{'utc':utc(),'rank':rank,'torch':torch.__version__,'device':str(torch.cuda.get_device_properties(0)),'cpu_affinity':sorted(os.sched_getaffinity(0)),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    import importlib.metadata,shutil
    sources={};site=Path(torch.__file__).parent.parent
    for rel in ['flashinfer/gemm/gemm_base.py','flashinfer/quantization/fp4_quantization.py','torch/sparse/semi_structured.py']:
        source=site/rel
        if source.exists():
            dest=root/('source_'+rel.replace('/','_'));shutil.copyfile(source,dest);sources[rel]=hashlib.sha256(source.read_bytes()).hexdigest()
    versions={}
    for package in ['torch','flashinfer-python','nvidia-nccl-cu13','numpy']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    write_json(root/'extensions_source_versions.json',{'utc':utc(),'versions':versions,'sources':sources,'native_library_sha256':hashlib.sha256((root/'kernels.so').read_bytes()).hexdigest()})
    result=run_conditions(torch,np,lib,root)
    from gpu_overlap import run as run_overlap
    result['overlap']=run_overlap(root,rank)
    return result


def run_conditions(torch,np,lib,root):
    done=completed_keys(root)
    p=root/'measurements.jsonl'
    previous=[json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
    previous=[x for x in previous if x.get('suite')=='extensions']
    invalid=[x for x in previous if invalid_harness_probe(x) or invalid_legacy_managed_validation(x)]
    previous=[x for x in previous if not (invalid_harness_probe(x) or invalid_legacy_managed_validation(x))]
    unsupported_keys={(x['condition_id'],x['round']) for x in previous if x.get('state') in ['unsupported_probe','unsupported_after_probe']}
    blocked={x['condition_id']:x.get('error',x.get('reason','unsupported prior probe')) for x in previous if x.get('state') in ['unsupported_probe','unsupported_after_probe']}
    write_json(root/'extensions_resume_coverage.json',{'utc':utc(),'preserved_valid_rounds':len(done),'preserved_unsupported_rounds':len(unsupported_keys),'excluded_api_probes':invalid,'rule':'preserve validated and genuinely unsupported condition/round pairs; retry invalid sparse .T dispatch through installed .t API; numerical failures require review'})
    warm=warmup(torch,root);rows=conditions();count=len(done);profiled=set()
    with torch.inference_mode():
        for repeat in range(10):
            ordered=[c for c in rows if repeat<c['rounds']];random.Random(20261016+repeat).shuffle(ordered)
            for c in ordered:
                if (c['condition_id'],repeat+1) in done|unsupported_keys:continue
                sample_guard(root);row={**c,'round':repeat+1,'start_utc':utc(),'suite':'extensions'};fn=check=None;cleanup=lambda:None
                if c['condition_id'] in blocked:
                    emit(root,{**row,'state':'unsupported_after_probe','reason':blocked[c['condition_id']]});continue
                try:
                    torch.manual_seed(20261016+repeat);fn,check,meta,cleanup=make(torch,np,lib,root,c);row.update(meta)
                    if fn is None:row['state']='skipped_memory_guard'
                    else:
                        fn();torch.cuda.synchronize();v=check();row.update(validation=v,validation_phase='before_timing')
                        if not v['passed']:raise RuntimeError('extension numerical validation failed before timing')
                        if c['kind']=='low_precision' and c['precision'] not in profiled:
                            # Diagnostic trace is kept separate from unprofiled rates.
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                                fn();torch.cuda.synchronize()
                            prof.export_chrome_trace(str(root/('low_precision_'+c['precision']+'.trace.json')))
                            profiled.add(c['precision'])
                        d=timed(torch,root,fn,1);v=check();row.update(d,validation=v,validation_phase='after_timing')
                        if not v['passed']:raise RuntimeError('extension numerical validation failed after timing')
                        row.update(d,validation=v,state='measured');count+=1
                        if 'flops_per_call' in meta:row['effective_tflops']=meta['flops_per_call']*d['iterations']/d['cuda_interval_s']/1e12
                        if 'logical_bytes_per_call' in meta:row['effective_gb_s']=meta['logical_bytes_per_call']*d['iterations']/d['cuda_interval_s']/1e9
                        if 'accesses_per_call' in meta:row['dependent_access_ns']=d['mean_device_interval_us']*1000/meta['accesses_per_call']
                except (NotImplementedError,RuntimeError,ImportError) as exc:
                    is_unsupported=isinstance(exc,(ImportError,NotImplementedError)) or any(s in str(exc).lower() for s in ['not supported','unsupported','no available kernel','requires sm','not implemented'])
                    row.update(state='unsupported_probe' if is_unsupported else 'condition_failed',error=str(exc)[:1500]);emit(root,row)
                    if is_unsupported:blocked[c['condition_id']]=str(exc)[:1500]
                    else:raise
                else:row['end_utc']=utc();emit(root,row)
                finally:
                    cleanup();fn=check=None;gc.collect();torch.cuda.empty_cache()
    return {'validated':count,'conditions':len(rows),'unsupported_conditions':blocked,'planned_condition_rounds':sum(x['rounds'] for x in rows)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--rank',type=int,default=0);ap.add_argument('--cpu-worker',type=int);ap.add_argument('--cpu-mode');a=ap.parse_args();root=Path(a.root)
    if a.cpu_worker is not None:cpu_worker(root,a.cpu_worker,a.cpu_mode);return
    try:
        result=run(root,a.rank);write_json(root/'extensions_result.json',{'state':'complete','utc':utc(),**result})
    except BaseException as exc:write_json(root/'extensions_result.json',{'state':'stopped','utc':utc(),'error':str(exc)[:1500]});raise
if __name__=='__main__':main()
