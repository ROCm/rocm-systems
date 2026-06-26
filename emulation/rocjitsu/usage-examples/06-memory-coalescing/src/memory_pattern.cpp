// Memory coalescing example
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void strided_access(float *out, float *in, int N, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = in[i * stride];  // Strided (poor coalescing)
  }
}

__global__ void coalesced_access(float *out, float *in, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = in[i];  // Sequential (good coalescing)
  }
}

int main() {
  const int N = 1024;
  float *d_in, *d_out;

  hipMalloc(&d_in, N * 16 * sizeof(float));
  hipMalloc(&d_out, N * sizeof(float));

  std::cout << "Testing memory access patterns..." << std::endl;

  // Strided access
  strided_access<<<16, 64>>>(d_out, d_in, N, 16);
  hipDeviceSynchronize();
  std::cout << "Strided access: completed" << std::endl;

  // Coalesced access
  coalesced_access<<<16, 64>>>(d_out, d_in, N);
  hipDeviceSynchronize();
  std::cout << "Coalesced access: completed" << std::endl;

  hipFree(d_in);
  hipFree(d_out);
  return 0;
}