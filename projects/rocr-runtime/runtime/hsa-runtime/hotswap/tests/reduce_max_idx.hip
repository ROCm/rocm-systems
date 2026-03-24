#include <hip/hip_runtime.h>
extern "C" __global__ void reduce_max_idx(int* out_idx, float* out_val,
                                           const float* in, int n) {
  extern __shared__ float smem[];
  int* sidx = (int*)(smem + blockDim.x);
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  smem[tid] = (i < n) ? in[i] : -1e30f;
  sidx[tid] = i;
  __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s && smem[tid+s] > smem[tid]) {
      smem[tid] = smem[tid+s];
      sidx[tid] = sidx[tid+s];
    }
    __syncthreads();
  }
  if (tid == 0) { out_val[blockIdx.x] = smem[0]; out_idx[blockIdx.x] = sidx[0]; }
}
