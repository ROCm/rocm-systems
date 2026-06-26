// Kernel crash example - NULL pointer dereference
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void unsafe_kernel(float *ptr, int N) {
  int i = threadIdx.x;
  // BUG: No NULL check
  if (i < N) ptr[i] = i * 1.0f;
}

int main() {
  float *d_ptr = nullptr;  // Intentionally NULL
  std::cout << "Launching kernel with NULL pointer..." << std::endl;
  unsafe_kernel<<<1, 64>>>(d_ptr, 64);
  hipDeviceSynchronize();
  std::cout << "If you see this, rocjitsu caught the error!" << std::endl;
  return 0;
}