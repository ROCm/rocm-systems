#include <hip/hip_runtime.h>
// Split-K SGEMM pass 1: compute partial sums into workspace.
// grid=(M, num_splits, 1)  block=(32, 1, 1)
//
// blockIdx.y (split index) is derived from TTMP9 on GFX12 — a different TTMP
// taint chain than the 1D matmul, giving the transpiler new coverage.
extern "C" __global__ void matmul_splitk_compute(
    float* __restrict__ workspace,   // [num_splits * M * N]
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
