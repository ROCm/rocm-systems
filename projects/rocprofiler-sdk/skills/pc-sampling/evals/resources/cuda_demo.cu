// cuda_demo.cu
//
// Build:
//   nvcc -O3 -g cuda_demo.cu -o cuda_demo
//
// Run:
//   ./cuda_demo [repeat]     # repeat defaults to 1

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err_ = (expr);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error '%s' at %s:%d: %s\n", #expr, __FILE__,  \
                   __LINE__, cudaGetErrorString(err_));                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

constexpr int kN = 8192;
constexpr int kTransposeReps = 36;
constexpr int kBlock = 256;
constexpr int kDivThreads = 1 << 20;
constexpr int kDivIters = 58000;
constexpr int kAtomicThreads = 1 << 20;
constexpr int kAtomicBins = 4;
constexpr int kAtomicReps = 14;

__global__ __launch_bounds__(kBlock) void k_naive_transpose(float* __restrict__ a,
                                                            float* __restrict__ b, int n,
                                                            int reps) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= n || row >= n) return;

  for (int i = 0; i < reps; ++i) {
    if ((i & 1) == 0) {
      b[static_cast<size_t>(col) * n + row] =
          a[static_cast<size_t>(row) * n + col] + 1.0f;
    } else {
      a[static_cast<size_t>(col) * n + row] =
          b[static_cast<size_t>(row) * n + col] + 1.0f;
    }
  }
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
  cudaEvent_t start_ev, stop_ev;
  Timer() {
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));
  }
  ~Timer() {
    (void)cudaEventDestroy(start_ev);
    (void)cudaEventDestroy(stop_ev);
  }
  void start() { CUDA_CHECK(cudaEventRecord(start_ev)); }
  float stop() {
    CUDA_CHECK(cudaEventRecord(stop_ev));
    CUDA_CHECK(cudaEventSynchronize(stop_ev));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
    return ms;
  }
};

int main(int argc, char** argv) {
  const int repeat = (argc > 1) ? std::max(1, std::atoi(argv[1])) : 1;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("Device: %s (SM %d.%d), %d SMs\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);

  const size_t elems = static_cast<size_t>(kN) * kN;
  float *d_in = nullptr, *d_out = nullptr;
  uint32_t* d_div = nullptr;
  unsigned* d_hist = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_div, static_cast<size_t>(kDivThreads) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_hist, kAtomicBins * sizeof(unsigned)));
  CUDA_CHECK(cudaMemset(d_in, 0, elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_out, 0, elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_div, 0, static_cast<size_t>(kDivThreads) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_hist, 0, kAtomicBins * sizeof(unsigned)));
  CUDA_CHECK(cudaDeviceSynchronize());

  dim3 block(64, 4);
  dim3 grid(kN / block.x, kN / block.y);

  float transpose_ms = 0.f, div_ms = 0.f, atomic_ms = 0.f;
  Timer timer;
  for (int rep = 0; rep < repeat; ++rep) {
    timer.start();
    k_naive_transpose<<<grid, block>>>(d_in, d_out, kN, kTransposeReps);
    transpose_ms += timer.stop();

    timer.start();
    k_integer_div<<<kDivThreads / kBlock, kBlock>>>(d_div, 1000003u, kDivIters);
    div_ms += timer.stop();

    timer.start();
    k_global_atomic_add<<<kAtomicThreads / kBlock, kBlock>>>(d_hist, kAtomicReps);
    atomic_ms += timer.stop();
  }
  CUDA_CHECK(cudaGetLastError());

  float sample[4] = {};
  uint32_t div_sample[4] = {};
  unsigned hist[kAtomicBins] = {};
  CUDA_CHECK(cudaMemcpy(sample, d_out, sizeof(sample), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(div_sample, d_div, sizeof(div_sample), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hist, d_hist, sizeof(hist), cudaMemcpyDeviceToHost));

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

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_div));
  CUDA_CHECK(cudaFree(d_hist));
  return 0;
}
