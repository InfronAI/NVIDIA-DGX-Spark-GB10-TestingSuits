"""Paired compute/communication controls, fixed total work and rank count."""
import ctypes as C,gc,json,socket,time,os
from gpu_system_collectives import UniqueId,nccl_library,checked
from gpu_system_guard import sample_guard,memory_check,write_json,utc
from gpu_system_bench import emit,validation

class Rendezvous:
    def __init__(self,root,rank):
        until=time.monotonic()+1200
        if rank==0:
            s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind((os.environ.get('SPARKBENCH_MASTER_ADDR','127.0.0.1'),int(os.environ.get('SPARKBENCH_MASTER_PORT','18571'))+1));s.listen(1);s.settimeout(2)
            while True:
                sample_guard(root)
                try:self.sock,_=s.accept();break
                except socket.timeout:
                    if time.monotonic()>until:raise
            s.close()
        else:
            while True:
                sample_guard(root)
                try:self.sock=socket.create_connection((os.environ.get('SPARKBENCH_MASTER_ADDR','127.0.0.1'),int(os.environ.get('SPARKBENCH_MASTER_PORT','18571'))+1),timeout=2);break
                except OSError:
                    if time.monotonic()>until:raise
                    time.sleep(1)
        self.sock.settimeout(1200);self.file=self.sock.makefile('rwb',buffering=0)
    def exchange(self,x):
        self.file.write(json.dumps(x).encode()+b'\n');data=self.file.readline(100000)
        if not data:raise RuntimeError('overlap peer connection ended')
        return json.loads(data)

def run(root,rank):
    import torch,base64,random
    sample_guard(root);control=Rendezvous(root,rank);lib,path=nccl_library();uid=UniqueId()
    if rank==0:checked(lib,lib.ncclGetUniqueId(C.byref(uid)))
    peer=control.exchange({'uid':base64.b64encode(bytes(uid)).decode() if rank==0 else None})
    if rank==1:uid=UniqueId.from_buffer_copy(base64.b64decode(peer['uid']))
    comm=C.c_void_p();checked(lib,lib.ncclCommInitRank(C.byref(comm),2,uid,rank));count=0
    try:
        for repeat in range(5):
            for size in [8192,8388608,67108864]:
                for n in [2048,4096]:
                    guard=memory_check(size*2+n*n*6+512*1024**2);peer=control.exchange({'allowed':guard['allowed']})
                    if not guard['allowed'] or not peer['allowed']:
                        emit(root,dict(kind='compute_communication_overlap',round=repeat+1,message_bytes=size,matrix_n=n,state='skipped_memory_guard',memory_guard=guard));continue
                    src=torch.full((size//4,),rank+1.,device='cuda');dst=torch.empty_like(src);a=torch.randn(n,n,device='cuda',dtype=torch.bfloat16);b=torch.randn_like(a);out=torch.empty_like(a);ref=a[:16].cpu().double()@b[:,:16].cpu().double()
                    current=torch.cuda.current_stream();cs=torch.cuda.Stream();ns=torch.cuda.Stream()
                    modes=['compute_only','communication_only','serial','overlap'];random.Random(20261016+repeat).shuffle(modes)
                    for mode in modes:
                        def invoke():
                            if mode=='overlap':
                                cs.wait_stream(current);ns.wait_stream(current)
                                with torch.cuda.stream(cs):torch.mm(a,b,out=out)
                                checked(lib,lib.ncclAllReduce(src.data_ptr(),dst.data_ptr(),size//4,7,0,comm,C.c_void_p(ns.cuda_stream)))
                                current.wait_stream(cs);current.wait_stream(ns)
                            else:
                                if mode!='communication_only':torch.mm(a,b,out=out)
                                if mode!='compute_only':checked(lib,lib.ncclAllReduce(src.data_ptr(),dst.data_ptr(),size//4,7,0,comm,C.c_void_p(current.cuda_stream)))
                        key=[repeat,size,n,mode];assert control.exchange({'key':key})['key']==key
                        for _ in range(5):invoke()
                        torch.cuda.synchronize();first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True);elapsed=0.;iterations=0
                        while True:
                            sample_guard(root);control.exchange({'begin':iterations});first.record()
                            for _ in range(8):invoke()
                            last.record();last.synchronize();elapsed+=first.elapsed_time(last)/1000;iterations+=8
                            other=control.exchange({'elapsed':elapsed})
                            if min(elapsed,other['elapsed'])>=1:break
                        v={'passed':True}
                        if mode!='communication_only':v=validation(out[:16,:16],ref,.03)
                        if mode!='compute_only':v['collective_passed']=bool((dst==3).all().item());v['passed'] &= v['collective_passed']
                        other=control.exchange({'valid':v['passed']})
                        if not v['passed'] or not other['valid']:raise RuntimeError('overlap numerical validation failed')
                        emit(root,dict(kind='compute_communication_overlap',rank=rank,round=repeat+1,message_bytes=size,matrix_n=n,mode=mode,state='measured',iterations=iterations,cuda_interval_s=elapsed,mean_device_interval_us=elapsed/iterations*1e6,validation=v,timing_boundary='joined CUDA events; NCCL control exchange excluded; same compute and communication work for serial/overlap'))
                        count+=1
                    invoke=None;del src,dst,a,b,out,ref;gc.collect();torch.cuda.empty_cache()
        checked(lib,lib.ncclCommDestroy(comm))
    except BaseException:lib.ncclCommAbort(comm);raise
    finally:control.file.close();control.sock.close()
    return {'validated':count,'rounds':5}
