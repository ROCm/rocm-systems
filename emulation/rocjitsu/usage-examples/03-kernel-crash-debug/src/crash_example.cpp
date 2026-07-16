// Kernel crash example - NULL pointer dereference
// Demonstrates that rocjitsu may complete silently; host validation is required.
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <iostream>

__global__ void unsafe_kernel(float *ptr, int N) {
  int i = threadIdx.x;
  // BUG: No NULL check — undefined on real hardware
  if (i < N) ptr[i] = i * 1.0f;
}

int main() {
  float *d_ptr = nullptr;  // Intentionally NULL
  std::cout << "Invalid Pointer Example - NULL device pointer" << std::endl;
  std::cout << "  d_ptr = nullptr" << std::endl;
  std::cout << "  Launch: unsafe_kernel<<<1, 64>>>(d_ptr, 64)" << std::endl;
  std::cout << std::endl;

  std::cout << "Launching kernel with NULL pointer..." << std::endl;
  unsafe_kernel<<<1, 64>>>(d_ptr, 64);
  hipDeviceSynchronize();

  std::cout << std::endl;
  std::cout << "WARNING: hipDeviceSynchronize() returned successfully." << std::endl;
  std::cout << "  rocjitsu did NOT fault on the NULL pointer." << std::endl;
  std::cout << "  The emulator wrote through VA 0 into sparse backing memory." << std::endl;
  std::cout << "  On real hardware this is undefined — may fault, hang, or corrupt." << std::endl;
  std::cout << "  Fix: hipMalloc before launch + host-side validation (see crash_fixed.cpp)."
            << std::endl;

  return EXIT_FAILURE;
}
