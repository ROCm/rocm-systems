#include <hip/hip_runtime.h>
extern "C" __global__ void layernorm(float* out, const float* in,
                                      const float* gamma, const float* beta,
                                      int N, float eps) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;
  int i = blockIdx.x * N + tid;
  float val = (tid < N) ? in[i] : 0.0f;
  smem[tid] = val; __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid+s]; __syncthreads();
  }
  float mean = smem[0] / (float)N; __syncthreads();
  float diff = val - mean;
  smem[tid] = diff * diff; __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid+s]; __syncthreads();
  }
  float inv_std = rsqrtf(smem[0] / (float)N + eps);
  if (tid < N) out[i] = diff * inv_std * gamma[tid] + beta[tid];
}
