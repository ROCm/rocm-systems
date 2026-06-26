// Multi-GPU collective example
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
  int deviceCount = 0;
  hipGetDeviceCount(&deviceCount);

  std::cout << "Multi-GPU Collective Example" << std::endl;
  std::cout << "Detected " << deviceCount << " GPU(s)" << std::endl;

  // Simple multi-device test
  for (int dev = 0; dev < deviceCount; ++dev) {
    hipSetDevice(dev);

    float *d_data;
    hipMalloc(&d_data, 1024 * sizeof(float));

    std::cout << "GPU " << dev << ": Allocated memory" << std::endl;

    hipFree(d_data);
  }

  std::cout << "Multi-GPU test completed" << std::endl;
  return 0;
}