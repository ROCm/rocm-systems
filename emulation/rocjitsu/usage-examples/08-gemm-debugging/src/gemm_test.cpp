// GEMM debugging example
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Simple GEMM kernel (not optimized)
__global__ void simple_gemm(float *C, const float *A, const float *B,
                            int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

int main() {
  const int M = 16, N = 16, K = 16;
  std::vector<float> h_A(M * K, 1.0f);
  std::vector<float> h_B(K * N, 1.0f);
  std::vector<float> h_C(M * N);

  float *d_A, *d_B, *d_C;
  hipMalloc(&d_A, M * K * sizeof(float));
  hipMalloc(&d_B, K * N * sizeof(float));
  hipMalloc(&d_C, M * N * sizeof(float));

  hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);

  std::cout << "Running GEMM: C(" << M << "x" << N << ") = A(" << M << "x" << K
            << ") * B(" << K << "x" << N << ")" << std::endl;

  simple_gemm<<<grid, block>>>(d_C, d_A, d_B, M, N, K);
  hipDeviceSynchronize();

  hipMemcpy(h_C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost);

  // Verify: all elements should be K (since all inputs are 1.0)
  bool correct = true;
  for (int i = 0; i < M * N; ++i) {
    if (std::abs(h_C[i] - K) > 1e-5f) {
      correct = false;
      break;
    }
  }

  std::cout << "Result: " << (correct ? "PASSED" : "FAILED") << std::endl;

  hipFree(d_A); hipFree(d_B); hipFree(d_C);
  return correct ? 0 : 1;
}