#include <hip/hip_runtime.h>
// Row-wise softmax with shared memory warp reduction.
// grid=(rows,1,1), block=(32,1,1), shared_mem=32*sizeof(float)
//
// Phase 1: stride over row, reduce max into smem[0..31], tree reduce → smem[0]
// Phase 2: compute exp(x-max) per element, write to out, reduce sum → smem[0]
// Phase 3: normalize out in-place by 1/sum
extern "C" __global__ void softmax_row(
    float* __restrict__ out, const float* __restrict__ in,
    int rows, int cols)
{
  extern __shared__ float smem[];  // 32 floats
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) return;

  const float* row_in = in + row * cols;
  float*       row_out = out + row * cols;

  // Phase 1: find row max
  float thread_max = -1e30f;
  for (int col = tid; col < cols; col += blockDim.x)
    thread_max = fmaxf(thread_max, row_in[col]);
  smem[tid] = thread_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      smem[tid] = fmaxf(smem[tid], smem[tid + s]);
    __syncthreads();
  }
  float row_max = smem[0];
  __syncthreads();

  // Phase 2: exp(x - max), store to out, accumulate sum
  float thread_sum = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    float e = __expf(row_in[col] - row_max);
    row_out[col] = e;
    thread_sum += e;
  }
  smem[tid] = thread_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      smem[tid] += smem[tid + s];
    __syncthreads();
  }
  float inv_sum = 1.0f / smem[0];
  __syncthreads();

  // Phase 3: normalize in-place
  for (int col = tid; col < cols; col += blockDim.x)
    row_out[col] *= inv_sum;
}
