#include <hip/hip_runtime.h>
// Block-level max reduction into out[blockIdx.x]
// Tests v_max_f32 in shared memory reduction tree
extern "C" __global__ void max_reduce(float* out, const float* in, int n) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  sdata[tid] = (i < n) ? in[i] : -1e30f;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}
