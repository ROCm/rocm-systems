// Test kernel for lds_conflict_fit.py.
//
// It has two LDS load sites with known bank behaviour:
//   site A: consecutive lanes read consecutive dwords -> no bank conflicts.
//   site B: lanes are 64 dwords (256 bytes) apart -> every lane hits the same
//           bank, the worst case (wave32: 31 extra cycles per instruction).
// Each block runs the two sites a different number of times, sweeping from
// "all A" in block 0 to "all B" in the last block. Over the run, the mix of
// A and B in any time window therefore changes, which is what makes the
// least-squares fit able to separate the two sites.
//
// Build with the sqtt-marker pass so every LDS access emits an address marker:
//   SQTT_TRACE_ADDRESSES=lds hipcc -O2 -g --offload-arch=<gfx> \
//     -fpass-plugin=<prefix>/lib/libsqtt-marker.so lds_conflict.hip -o lds_conflict
// Then capture with:
//   rocprofv3 --att -d captures -- ./lds_conflict 64 64 64
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                              \
  do {                                                                        \
    hipError_t e = (x);                                                       \
    if (e != hipSuccess) {                                                    \
      std::printf("%s failed: %s\n", #x, hipGetErrorString(e));               \
      return 1;                                                               \
    }                                                                         \
  } while (0)

constexpr int kThreads = 256;
// 64 dwords per thread, so site B's stride-64 pattern stays in range.
constexpr int kLdsWords = 64 * kThreads;

__global__ void lds_conflict(float *out, int iters_a, int iters_b) {
  __shared__ float lds[kLdsWords];
  const int t = threadIdx.x;

  // Fill LDS. These stores are conflict-free and show up as an lds_store site;
  // pass --loads-only to the fit to leave them out.
  for (int i = t; i < kLdsWords; i += kThreads)
    lds[i] = static_cast<float>(i);
  __syncthreads();

  // Block 0 runs only site A, the last block only site B, and blocks in
  // between run a linear mix of both.
  const int n = gridDim.x > 1 ? gridDim.x - 1 : 1;
  const int a = iters_a * (n - blockIdx.x) / n;
  const int b = iters_b * blockIdx.x / n;

  float acc = 0.f;
  // Site A: lane t reads dword t + i -> 32 different banks, no conflict.
  for (int i = 0; i < a; ++i)
    acc += lds[(t + i) % kLdsWords];
  // Site B: lane t reads dword 64 * t + i -> all lanes map to one bank.
  for (int i = 0; i < b; ++i)
    acc += lds[(t * 64 + i) % kLdsWords];

  // Keep the loads live so the compiler cannot drop them.
  out[blockIdx.x * kThreads + t] = acc;
}

int main(int argc, char **argv) {
  // Arguments: number of blocks, site A iterations, site B iterations.
  const int blocks = argc > 1 ? std::atoi(argv[1]) : 1024;
  const int iters_a = argc > 2 ? std::atoi(argv[2]) : 4096;
  const int iters_b = argc > 3 ? std::atoi(argv[3]) : 4096;

  float *out = nullptr;
  CHECK(hipMalloc(&out, sizeof(float) * blocks * kThreads));
  hipLaunchKernelGGL(lds_conflict, dim3(blocks), dim3(kThreads), 0, 0, out,
                     iters_a, iters_b);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());

  // Copy back and print a couple of values as a basic sanity check.
  std::vector<float> h(blocks * kThreads);
  CHECK(hipMemcpy(h.data(), out, sizeof(float) * h.size(), hipMemcpyDeviceToHost));
  std::printf("blocks=%d iters_a=%d iters_b=%d out[0]=%f out[last]=%f\n", blocks,
              iters_a, iters_b, h.front(), h.back());
  CHECK(hipFree(out));
  return 0;
}
