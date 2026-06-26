// Fixed version with validation
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void safe_kernel(float *ptr, int N) {
  int i = threadIdx.x;
  // FIXED: Check pointer validity
  if (ptr != nullptr && i < N) {
    ptr[i] = i * 1.0f;
  }
}

int main() {
  float *d_ptr = nullptr;
  hipMalloc(&d_ptr, 64 * sizeof(float));
  std::cout << "Launching kernel with valid pointer..." << std::endl;
  safe_kernel<<<1, 64>>>(d_ptr, 64);
  hipDeviceSynchronize();
  std::cout << "Success! Kernel completed safely." << std::endl;
  hipFree(d_ptr);
  return 0;
}