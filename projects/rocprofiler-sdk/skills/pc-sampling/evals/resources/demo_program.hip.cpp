// Build:
//   hipcc -O3 -g demo_program.hip.cpp -o demo_program
//
// Run:
//   ./demo_program [repeat]     # repeat defaults to 1 (~100 ms of kernel time)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err_ = (expr);                                                  \
    if (err_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP error '%s' at %s:%d: %s\n", #expr, __FILE__,   \
                   __LINE__, hipGetErrorString(err_));                         \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// Sized so each kernel is roughly a third of ~100 ms on gfx942.
constexpr int kN = 8192;
constexpr int kTransposeReps = 35;
constexpr int kBlock = 256;
constexpr int kDivThreads = 1 << 20;
constexpr int kDivIters = 58000;
constexpr int kAtomicThreads = 1 << 20;
constexpr int kAtomicBins = 4;
constexpr int kAtomicReps = 14;

__global__ __launch_bounds__(kBlock) void k_naive_transpose(const float* __restrict__ in,
                                                            float* __restrict__ out, int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= n || row >= n) return;
  out[static_cast<size_t>(col) * n + row] = in[static_cast<size_t>(row) * n + col] + 1.0f;
}

__global__ __launch_bounds__(kBlock) void k_integer_div(uint32_t* __restrict__ out,
                                                        uint32_t divisor, int iters) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t v = tid * 2u + 1u;
  uint32_t d = (divisor + tid) & 255u;
  for (int i = 0; i < iters; ++i) {
    v = (v * 1664525u + 1013904223u) / (d | 1u);
    d = (d + 2u) & 255u;
  }
  out[tid] = v;
}

__global__ __launch_bounds__(kBlock) void k_global_atomic_add(unsigned* __restrict__ hist,
                                                              int reps) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < reps; ++i) {
    atomicAdd(&hist[(tid + static_cast<uint32_t>(i)) & 3u], 1u);
  }
}

struct Timer {
  hipEvent_t start_ev, stop_ev;
  Timer() {
    HIP_CHECK(hipEventCreate(&start_ev));
    HIP_CHECK(hipEventCreate(&stop_ev));
  }
  ~Timer() {
    (void)hipEventDestroy(start_ev);
    (void)hipEventDestroy(stop_ev);
  }
  void start() { HIP_CHECK(hipEventRecord(start_ev)); }
  float stop() {
    HIP_CHECK(hipEventRecord(stop_ev));
    HIP_CHECK(hipEventSynchronize(stop_ev));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, start_ev, stop_ev));
    return ms;
  }
};

int main(int argc, char** argv) {
  const int repeat = (argc > 1) ? std::max(1, std::atoi(argv[1])) : 1;

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("Device: %s (%s), %d CUs\n", prop.name, prop.gcnArchName,
              prop.multiProcessorCount);

  const size_t elems = static_cast<size_t>(kN) * kN;
  float *d_in = nullptr, *d_out = nullptr;
  uint32_t* d_div = nullptr;
  unsigned* d_hist = nullptr;
  HIP_CHECK(hipMalloc(&d_in, elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_div, static_cast<size_t>(kDivThreads) * sizeof(uint32_t)));
  HIP_CHECK(hipMalloc(&d_hist, kAtomicBins * sizeof(unsigned)));
  HIP_CHECK(hipMemset(d_in, 0, elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_out, 0, elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_div, 0, static_cast<size_t>(kDivThreads) * sizeof(uint32_t)));
  HIP_CHECK(hipMemset(d_hist, 0, kAtomicBins * sizeof(unsigned)));
  HIP_CHECK(hipDeviceSynchronize());

  dim3 block(64, 4);
  dim3 grid(kN / block.x, kN / block.y);

  float transpose_ms = 0.f, div_ms = 0.f, atomic_ms = 0.f;
  Timer timer;
  for (int rep = 0; rep < repeat; ++rep) {
    timer.start();
    for (int i = 0; i < kTransposeReps; ++i) {
      if ((i & 1) == 0)
        k_naive_transpose<<<grid, block>>>(d_in, d_out, kN);
      else
        k_naive_transpose<<<grid, block>>>(d_out, d_in, kN);
    }
    transpose_ms += timer.stop();

    timer.start();
    k_integer_div<<<kDivThreads / kBlock, kBlock>>>(d_div, 1000003u, kDivIters);
    div_ms += timer.stop();

    timer.start();
    k_global_atomic_add<<<kAtomicThreads / kBlock, kBlock>>>(d_hist, kAtomicReps);
    atomic_ms += timer.stop();
  }
  HIP_CHECK(hipGetLastError());

  float sample[4] = {};
  uint32_t div_sample[4] = {};
  unsigned hist[kAtomicBins] = {};
  HIP_CHECK(hipMemcpy(sample, d_out, sizeof(sample), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(div_sample, d_div, sizeof(div_sample), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(hist, d_hist, sizeof(hist), hipMemcpyDeviceToHost));

  const float sum_ms = transpose_ms + div_ms + atomic_ms;
  std::printf("\n%-22s %10s %7s\n", "kernel", "time (ms)", "share");
  std::printf("%-22s %10.3f %6.1f%%\n", "k_naive_transpose", transpose_ms,
              100.f * transpose_ms / sum_ms);
  std::printf("%-22s %10.3f %6.1f%%\n", "k_integer_div", div_ms, 100.f * div_ms / sum_ms);
  std::printf("%-22s %10.3f %6.1f%%\n", "k_global_atomic_add", atomic_ms,
              100.f * atomic_ms / sum_ms);
  std::printf("%-22s %10.3f\n", "sum of kernels", sum_ms);
  std::printf("checksum out=[%.1f %.1f %.1f %.1f] div=[%u %u %u %u] atomic=[%u %u %u %u]\n",
              sample[0], sample[1], sample[2], sample[3], div_sample[0], div_sample[1],
              div_sample[2], div_sample[3], hist[0], hist[1], hist[2], hist[3]);

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_div));
  HIP_CHECK(hipFree(d_hist));
  return 0;
}
