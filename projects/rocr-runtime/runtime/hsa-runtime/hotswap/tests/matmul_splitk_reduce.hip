#include <hip/hip_runtime.h>
// Split-K SGEMM pass 2: sum workspace slices into C.
// grid=(M, 1, 1)  block=(32, 1, 1)
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
