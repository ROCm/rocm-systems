#include <hip/hip_runtime.h>
// Split-K SGEMM (two-kernel: compute partial sums + reduce)
//
// Pass 1 – matmul_splitk_compute
//   grid=(M, num_splits, 1)  block=(32, 1, 1)
//   Each block owns one output row (blockIdx.x) and one K-slice (blockIdx.y).
//   blockIdx.y exercises the TTMP9-derived workgroup-ID path that plain 1D
//   matmul never reaches, giving the transpiler a new TTMP taint chain.
//
// Pass 2 – matmul_splitk_reduce
//   grid=(M, 1, 1)  block=(32, 1, 1)
//   Sum workspace slices along the split dimension into C.

extern "C" __global__ void matmul_splitk_compute(
    float* __restrict__ workspace,     // [num_splits * M * N]
    const float* __restrict__ A,
    const float* __restrict__ B,
    int M, int N, int K, int num_splits)
{
  int row   = blockIdx.x;
  int split = blockIdx.y;
  int tid   = threadIdx.x;
  if (row >= M) return;

  int K_chunk = (K + num_splits - 1) / num_splits;
  int k_lo = split * K_chunk;
  int k_hi = k_lo + K_chunk;
  if (k_hi > K) k_hi = K;

  for (int col = tid; col < N; col += blockDim.x) {
    float acc = 0.0f;
    for (int k = k_lo; k < k_hi; k++)
      acc += A[row * K + k] * B[k * N + col];
    workspace[split * M * N + row * N + col] = acc;
  }
}

extern "C" __global__ void matmul_splitk_reduce(
    float* __restrict__ C,
    const float* __restrict__ workspace,   // [num_splits * M * N]
    int M, int N, int num_splits)
{
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= M) return;

  for (int col = tid; col < N; col += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < num_splits; s++)
      acc += workspace[s * M * N + row * N + col];
    C[row * N + col] = acc;
  }
}
