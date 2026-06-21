// EXPERIMENTAL test driver for the HSA_DOORBELL_POKE feature.
//
// Enqueues a large number of GPU kernels into a single HIP stream and then
// blocks the host in hipDeviceSynchronize while the GPU drains the backlog.
// While the host is blocked, the patched HSA runtime (HSA_DOORBELL_POKE=1)
// periodically re-rings the compute-queue doorbell. Run under rocprofv3.
//
// Build:
//   hipcc -O2 doorbell_poke_test.cpp -o doorbell_poke_test
//
// Run (with the patched libhsa-runtime64.so on the library path):
//   HSA_DOORBELL_POKE=1 HSA_DOORBELL_POKE_INTERVAL_US=10 \
//     rocprofv3 --hip-trace -- ./doorbell_poke_test

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(cmd)                                                                  \
  do {                                                                                  \
    hipError_t _e = (cmd);                                                              \
    if (_e != hipSuccess) {                                                             \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__,       \
              __LINE__);                                                                \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

// A deliberately non-trivial kernel so that 20000 dispatches take long enough
// (a couple of seconds) for the periodic doorbell poking to be clearly
// observable while the host is blocked in hipDeviceSynchronize.
__global__ void busy_kernel(uint64_t* out, uint32_t iters) {
  uint64_t acc = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    acc += i * 2654435761ull;
    acc ^= (acc << 13);
    acc += (acc >> 7);
  }
  if (acc == 0xdeadbeefull) {  // never true, keeps the work from being optimized away
    out[threadIdx.x] = acc;
  }
}

int main(int argc, char** argv) {
  int num_kernels = (argc > 1) ? std::atoi(argv[1]) : 20000;
  uint32_t iters = (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : 200000u;

  int device = 0;
  HIP_CHECK(hipSetDevice(device));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  printf("Device: %s (gcnArch=%s)\n", prop.name, prop.gcnArchName);

  uint64_t* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, sizeof(uint64_t) * 256));

  // Single stream: all kernels are serialized on one compute queue.
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  const dim3 grid(1);
  const dim3 block(64);

  printf("Enqueuing %d kernels (iters=%u) into a single stream...\n", num_kernels, iters);
  auto t_enqueue_start = std::chrono::steady_clock::now();
  for (int i = 0; i < num_kernels; ++i) {
    busy_kernel<<<grid, block, 0, stream>>>(d_out, iters);
  }
  HIP_CHECK(hipGetLastError());
  auto t_enqueue_end = std::chrono::steady_clock::now();
  double enqueue_ms =
      std::chrono::duration<double, std::milli>(t_enqueue_end - t_enqueue_start).count();
  printf("All kernels enqueued in %.1f ms. Calling hipDeviceSynchronize()...\n", enqueue_ms);
  fflush(stdout);

  auto t_sync_start = std::chrono::steady_clock::now();
  HIP_CHECK(hipDeviceSynchronize());
  auto t_sync_end = std::chrono::steady_clock::now();
  double sync_ms =
      std::chrono::duration<double, std::milli>(t_sync_end - t_sync_start).count();

  printf("hipDeviceSynchronize() returned after %.1f ms.\n", sync_ms);
  printf("Done.\n");

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_out));
  return 0;
}
