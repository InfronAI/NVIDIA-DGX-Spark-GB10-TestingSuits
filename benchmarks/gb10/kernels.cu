#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
// Explicit PTX keeps the scalar FMA workload independent of cuBLAS selection.
template<int CHAINS> __global__ void fma_core(float* out, unsigned long long* cyc, int loops) {
 int id=blockIdx.x*blockDim.x+threadIdx.x; float x[CHAINS];
 #pragma unroll
 for(int j=0;j<CHAINS;j++) x[j]=0.01f*(j+1)+(id%17)*0.001f;
 unsigned long long begin=clock64();
 for(int k=0;k<loops;k++) {
  #pragma unroll
  for(int j=0;j<CHAINS;j++) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x[j]) : "f"(1.000001f), "f"(0.000001f));
 }
 unsigned long long end=clock64();
 #pragma unroll
 for(int j=0;j<CHAINS;j++) out[id*CHAINS+j]=x[j];
 cyc[id]=end-begin;
}
__global__ void chase(const unsigned* p,unsigned* state,unsigned long long* cycles,int steps,int hint) {
 if(threadIdx.x || blockIdx.x)return;
 unsigned x=*state; unsigned long long begin=clock64();
 for(int j=0;j<steps;j++) {
  const unsigned* q=p+(size_t)x*32;
  if(hint) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(q) : "memory");
  else asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(q) : "memory");
 }
 *cycles=clock64()-begin; *state=x;
}
__global__ void shared_bank(float* out,int stride,int loops) {
 __shared__ volatile float a[8192];
 int tid=threadIdx.x, slot=(tid/32)*1024+(tid%32)*stride;
 a[slot]=float(tid%32); __syncthreads();
 for(int j=0;j<loops;j++)a[slot]=a[slot]+1.0f;
 out[blockIdx.x*blockDim.x+tid]=a[slot];
}
__global__ void stream_read(const float* a,float* out,size_t n) {
 size_t id=blockIdx.x*blockDim.x+threadIdx.x; float x=0;
 for(size_t i=id;i<n;i+=(size_t)gridDim.x*blockDim.x)x+=a[i];
 out[id]=x;
}
__global__ void stream_write(float* a,size_t n) {
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)a[i]=3.0f;
}
__global__ void empty_kernel() {asm volatile("");}
extern "C" int launch_fma(float* out,unsigned long long* cycles,int blocks,int threads,int chains,int loops,void* stream){
 if(chains==8)fma_core<8><<<blocks,threads,0,(cudaStream_t)stream>>>(out,cycles,loops);
 else fma_core<1><<<blocks,threads,0,(cudaStream_t)stream>>>(out,cycles,loops);
 return (int)cudaGetLastError();
}
extern "C" int launch_chase(const unsigned* p,unsigned* state,unsigned long long* cycles,int steps,int hint,void* stream){
 chase<<<1,32,0,(cudaStream_t)stream>>>(p,state,cycles,steps,hint);return (int)cudaGetLastError();
}
extern "C" int launch_shared(float* out,int blocks,int stride,int loops,void* stream){
 shared_bank<<<blocks,256,0,(cudaStream_t)stream>>>(out,stride,loops);return (int)cudaGetLastError();
}
extern "C" int launch_read(const float* a,float* out,size_t n,int blocks,void* stream){
 stream_read<<<blocks,256,0,(cudaStream_t)stream>>>(a,out,n);return (int)cudaGetLastError();
}
extern "C" int launch_write(float* a,size_t n,int blocks,void* stream){
 stream_write<<<blocks,256,0,(cudaStream_t)stream>>>(a,n);return (int)cudaGetLastError();
}
extern "C" int launch_empty(void* stream){empty_kernel<<<1,32,0,(cudaStream_t)stream>>>();return (int)cudaGetLastError();}
extern "C" int managed_alloc(void** p,size_t n){return (int)cudaMallocManaged(p,n);}
extern "C" int managed_free(void* p){return (int)cudaFree(p);}
extern "C" int fma_occupancy(int threads,int chains,int* active){
 return (int)(chains==8?cudaOccupancyMaxActiveBlocksPerMultiprocessor(active,fma_core<8>,threads,0):cudaOccupancyMaxActiveBlocksPerMultiprocessor(active,fma_core<1>,threads,0));
}
#ifdef PROBE_MAIN
#include <cstdio>
int main(){
 float* out=nullptr;unsigned long long* cyc=nullptr;
 if(cudaMalloc(&out,256*8*sizeof(float))!=cudaSuccess)return 2;
 if(cudaMalloc(&cyc,256*sizeof(unsigned long long))!=cudaSuccess)return 3;
 fma_core<8><<<1,256>>>(out,cyc,4096);
 int rc=(int)cudaDeviceSynchronize();cudaFree(out);cudaFree(cyc);std::printf("runtime_status=%d\n",rc);return rc;
}
#endif
