// Occupancy analysis example
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void high_register_kernel(float *data, int N) {
  // Uses many registers
  float v1 = threadIdx.x, v2 = v1 * 2, v3 = v2 * 2;
  float v4 = v3 * 2, v5 = v4 * 2, v6 = v5 * 2;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) data[i] = v1 + v2 + v3 + v4 + v5 + v6;
}

__global__ void low_register_kernel(float *data, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) data[i] = threadIdx.x * 12.0f;
}

int main() {
  const int N = 1024;
  float *d_data;
  hipMalloc(&d_data, N * sizeof(float));

  std::cout << "Testing different kernel configurations..." << std::endl;

  // Test various block sizes
  for (int blockSize : {64, 128, 256, 512}) {
    int gridSize = (N + blockSize - 1) / blockSize;
    std::cout << "Block size " << blockSize << ": ";
    low_register_kernel<<<gridSize, blockSize>>>(d_data, N);
    hipDeviceSynchronize();
    std::cout << "OK" << std::endl;
  }

  hipFree(d_data);
  return 0;
}