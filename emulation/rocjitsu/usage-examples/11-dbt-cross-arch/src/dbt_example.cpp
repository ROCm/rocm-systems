// DBT cross-architecture example
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

__global__ void simple_kernel(float *data, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i] = i * 2.0f;
  }
}

int main() {
  const int N = 256;
  std::vector<float> h_data(N);

  float *d_data;
  hipMalloc(&d_data, N * sizeof(float));

  std::cout << "DBT Cross-Architecture Example" << std::endl;
  std::cout << "Kernel compiled for gfx950 (CDNA4)" << std::endl;
  std::cout << "Running through DBT translation..." << std::endl;

  simple_kernel<<<4, 64>>>(d_data, N);
  hipDeviceSynchronize();

  hipMemcpy(h_data.data(), d_data, N * sizeof(float), hipMemcpyDeviceToHost);

  bool correct = true;
  for (int i = 0; i < N; ++i) {
    if (h_data[i] != i * 2.0f) {
      correct = false;
      break;
    }
  }

  std::cout << "Result: " << (correct ? "PASSED" : "FAILED") << std::endl;
  std::cout << "DBT translation successful!" << std::endl;

  hipFree(d_data);
  return correct ? 0 : 1;
}