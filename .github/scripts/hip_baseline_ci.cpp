// ROCM-32154 isolation test.
//
// Proves that the leaks reported in the CI failure originate in libamdhip64
// and libhsa-runtime64, not in RCCL. Exercises three allocation paths on
// every visible device with zero RCCL code in the process:
//
//   1. hipMalloc / hipFree
//      Triggers the hsa_amd_memory_pool_allocate staging-buffer allocation at
//      libamdhip64+0x5e6910 (4 MB indirect leak in CI report).
//
//   2. hipExtMallocWithFlags(..., hipDeviceMallocFinegrained) / hipFree
//      Triggers the fine-grained allocator at libamdhip64+0x5e5b14
//      (second 4 MB indirect leak in CI report, also seen via RCCL fillInfo).
//
//   3. hipSetDevice / hipInit (device context init)
//      Triggers hsa_amd_signal_create and AqlQueue::AqlQueue allocations in
//      libhsa-runtime64 (direct leaks in CI report from ROCr teardown order).
//
// Run with ASAN_OPTIONS=halt_on_error=0 so LSan fires at exit even if
// bad-frees occur mid-teardown (hardware topology determines which is seen).

#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  hipInit(0);

  int n = 0;
  hipGetDeviceCount(&n);
  printf("HIP baseline: %d device(s) visible\n", n);

  for (int dev = 0; dev < n; ++dev) {
    hipError_t err = hipSetDevice(dev);
    if (err != hipSuccess) {
      printf("  dev%d: hipSetDevice failed: %s\n", dev, hipGetErrorString(err));
      continue;
    }

    // Path 1: plain hipMalloc — triggers libamdhip64+0x5e6910 staging buffer.
    int* p = nullptr;
    err = hipMalloc(&p, 4);
    if (err == hipSuccess) {
      hipFree(p);
      printf("  dev%d: hipMalloc+hipFree OK\n", dev);
    } else {
      printf("  dev%d: hipMalloc failed: %s\n", dev, hipGetErrorString(err));
    }

    // Path 2: fine-grained alloc — triggers libamdhip64+0x5e5b14 allocator,
    // the same site reached via RCCL's fillInfo hipExtMallocWithFlags probe.
    int* fg = nullptr;
    err = hipExtMallocWithFlags((void**)&fg, 4, hipDeviceMallocFinegrained);
    if (err == hipSuccess) {
      hipFree(fg);
      printf("  dev%d: hipExtMallocWithFlags(finegrained)+hipFree OK\n", dev);
    } else {
      printf("  dev%d: hipExtMallocWithFlags(finegrained) not supported: %s\n",
             dev, hipGetErrorString(err));
    }
  }

  printf("HIP baseline: done\n");
  return 0;
}
