#include <hip/hip_runtime.h>
// Naive SGEMM: one block per output row, threads stride over columns
// Launch with grid=(M,1,1), block=(32,1,1); arbitrary M, N, K
// Stride loop exercises repeated memory access patterns across different K depths.
extern "C" __global__ void matmul(
    float* __restrict__ C, const float* __restrict__ A,
    const float* __restrict__ B, int M, int N, int K) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= M) return;
  for (int col = tid; col < N; col += blockDim.x) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
      sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}
