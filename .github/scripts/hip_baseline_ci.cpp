// ROCM-32154 isolation test: proves HIP staging-buffer leaks originate in
// libamdhip64, not RCCL. hipMalloc+hipFree on each visible device triggers
// the same hsa_amd_memory_pool_allocate call sites as the RCCL CI failure.
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
  hipInit(0);
  int n = 0;
  hipGetDeviceCount(&n);
  printf("HIP baseline: %d device(s) visible\n", n);
  for (int dev = 0; dev < n; ++dev) {
    int* p = nullptr;
    hipError_t err = hipSetDevice(dev);
    if (err != hipSuccess) { printf("  dev%d: hipSetDevice failed: %s\n", dev, hipGetErrorString(err)); continue; }
    err = hipMalloc(&p, 4);
    if (err != hipSuccess) { printf("  dev%d: hipMalloc failed: %s\n", dev, hipGetErrorString(err)); continue; }
    hipFree(p);
    printf("  dev%d: hipMalloc+hipFree OK\n", dev);
  }
  printf("HIP baseline: done\n");
  return 0;
}
