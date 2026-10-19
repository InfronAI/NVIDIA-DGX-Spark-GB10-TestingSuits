"""Portable adapter for the archived, hardware-validated GB10 implementations."""
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def backend():
    root=Path(__file__).resolve().parents[2]/'benchmarks/gb10'
    if not root.is_dir():raise RuntimeError('Use an editable install from a full repository checkout')
    sys.path.insert(0,str(root))
    return root


def run(args):
    source=backend()
    base=importlib.import_module('gpu_system_bench')
    ext=importlib.import_module('gpu_extensions')
    counts={'base':len(base.cases())*10,'extensions':sum(x['rounds'] for x in ext.conditions()),
            'collectives':3*28*10,'overlap':120,'soak':'60 min compute + 60 min bandwidth + 120 min mixed, separate phases'}
    if not 1<=args.duration_limit<=172800:raise ValueError('Finite duration must be between 1 and 172800 seconds')
    if not 1<=args.master_port<=65534:raise ValueError('Master port outside range')
    if not args.execute:return {'state':'dry_run','mode':args.mode,'condition_rounds_per_node':counts[args.mode],
        'thermal_policy':'85 C pause; below 75 C for 30 s to resume; collective arms abort on thermal limit',
        'deadline_seconds':args.duration_limit,'requires':'Dedicated Linux GB10 GPU; existing CUDA/PyTorch environment'}
    if sys.platform!='linux':raise RuntimeError('Hardware execution requires Linux; dry-run works anywhere')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if torch.cuda.get_device_capability()!=(12,1):raise RuntimeError('This hardware suite targets GB10 sm_121')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    guard=importlib.import_module('gpu_system_guard')
    os.environ['SPARKBENCH_MASTER_ADDR']=args.master_addr
    os.environ['SPARKBENCH_MASTER_PORT']=str(args.master_port)
    guard.write_json(output/'session.json',{'execute':True,'mode':args.mode,'rank':args.rank,'started_epoch':time.time(),
        'deadline_epoch':time.time()+args.duration_limit,'thermal_pause_c':85,'thermal_resume_c':75})
    guard.write_json(output/'environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'device':str(torch.cuda.get_device_properties(0)),
        'capability':list(torch.cuda.get_device_capability()),'mode':args.mode,'rank':args.rank,
        'scope':'User-selected dedicated hardware window; service residency must be recorded separately'})
    shutil.copyfile(source/'kernels.cu',output/'kernels.cu')
    lock=Path('/tmp/infron-sparkbench-gpu0.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        torch.set_num_threads(1);torch.cuda.set_device(0)
        with torch.inference_mode():
            if args.mode=='base':result=base.run_single(torch,output)
            elif args.mode=='soak':result=base.run_soak(torch,output)
            elif args.mode=='extensions':
                import numpy as np
                library=ext.library(output)
                result=ext.run_conditions(torch,np,library,output)
            else:
                module=importlib.import_module('gpu_system_collectives' if args.mode=='collectives' else 'gpu_overlap')
                result=module.run(output,args.rank)
        receipt={'state':'execution_ended','mode':args.mode,'result':result}
        guard.write_json(output/'result.json',receipt);return receipt
    except BaseException as exc:
        guard.write_json(output/'result.json',{'state':'stopped','error_type':type(exc).__name__,'error':str(exc)[:1000]})
        raise
    finally:lock.close()
