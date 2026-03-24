#include <hip/hip_runtime.h>
extern "C" __global__ void rmsnorm_kernel(float* out, const float* in, const float* weight, int N, float eps) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;
  int row = blockIdx.x;
  float val = (tid < N) ? in[row * N + tid] : 0.0f;
  smem[tid] = val * val;
  __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid+s];
    __syncthreads();
  }
  float rms = rsqrtf(smem[0] / (float)N + eps);
  if (tid < N) out[row * N + tid] = val * rms * weight[tid];
}
