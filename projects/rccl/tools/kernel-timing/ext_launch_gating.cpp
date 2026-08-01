// Gating experiments for replacing RCCL's stream-marker kernel timing with
// events attached to the dispatch itself.
//
//  0  does hipExtModuleLaunchKernel take grid dims or global work size?
//  1  is it legal under stream capture, with and without events?
//  2  how do stream-marker timings compare with dispatch-attached ones?
//  3  what do the two extra marker packets cost per launch?

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#define HC(x)                                                                            \
  do {                                                                                   \
    hipError_t _e = (x);                                                                 \
    if (_e != hipSuccess) {                                                              \
      printf("    FAIL %s -> %s\n", #x, hipGetErrorString(_e));                          \
    }                                                                                    \
  } while (0)

__global__ void reportDims(unsigned* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = gridDim.x;
    out[1] = blockDim.x;
  }
}

__global__ void spin(uint64_t ticks) {
  uint64_t t0 = __builtin_amdgcn_s_memrealtime();
  while (__builtin_amdgcn_s_memrealtime() - t0 < ticks) {
  }
}

static double wallRateKHz() {
  int khz = 0;
  hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0);
  return khz ? khz : 25000.0;
}

template <typename T>
static void mkExtra(void** extra, T* arg, size_t* sz) {
  extra[0] = HIP_LAUNCH_PARAM_BUFFER_POINTER;
  extra[1] = arg;
  extra[2] = HIP_LAUNCH_PARAM_BUFFER_SIZE;
  extra[3] = sz;
  extra[4] = HIP_LAUNCH_PARAM_END;
}

static hipFunction_t fnSpin, fnDims;
static const unsigned kBlocks = 8, kThreads = 256;

// ---------------------------------------------------------------- test 0
static void testDims() {
  printf("\n[0] argument convention (grid dims vs global work size)\n");
  unsigned *d, h[2];
  HC(hipMalloc(&d, 2 * sizeof(unsigned)));
  void* extra[5];
  size_t sz = sizeof(unsigned*);
  mkExtra(extra, &d, &sz);

  HC(hipModuleLaunchKernel(fnDims, kBlocks, 1, 1, kThreads, 1, 1, 0, nullptr, nullptr, extra));
  HC(hipDeviceSynchronize());
  HC(hipMemcpy(h, d, sizeof h, hipMemcpyDeviceToHost));
  printf("    hipModuleLaunchKernel   (%u,%u) -> gridDim.x=%u blockDim.x=%u\n", kBlocks, kThreads,
         h[0], h[1]);

  h[0] = h[1] = 0;
  HC(hipMemcpy(d, h, sizeof h, hipMemcpyHostToDevice));
  HC(hipExtModuleLaunchKernel(fnDims, kBlocks, 1, 1, kThreads, 1, 1, 0, nullptr, nullptr, extra,
                              nullptr, nullptr, 0));
  HC(hipDeviceSynchronize());
  HC(hipMemcpy(h, d, sizeof h, hipMemcpyDeviceToHost));
  printf("    hipExtModuleLaunchKernel(%u,%u) -> gridDim.x=%u blockDim.x=%u   %s\n", kBlocks,
         kThreads, h[0], h[1],
         h[0] == kBlocks ? "(same convention: grid dims)" : "(DIFFERENT: global work size)");

  h[0] = h[1] = 0;
  HC(hipMemcpy(d, h, sizeof h, hipMemcpyHostToDevice));
  HC(hipExtModuleLaunchKernel(fnDims, kBlocks * kThreads, 1, 1, kThreads, 1, 1, 0, nullptr, nullptr,
                              extra, nullptr, nullptr, 0));
  HC(hipDeviceSynchronize());
  HC(hipMemcpy(h, d, sizeof h, hipMemcpyDeviceToHost));
  printf("    hipExtModuleLaunchKernel(%u,%u) -> gridDim.x=%u blockDim.x=%u   %s\n",
         kBlocks * kThreads, kThreads, h[0], h[1],
         h[0] == kBlocks ? "(global work size confirmed)" : "");
  HC(hipFree(d));
}

// ---------------------------------------------------------------- test 1
static void captureCase(const char* what, bool withEvents) {
  hipStream_t s;
  hipGraph_t g = nullptr;
  hipEvent_t e0 = nullptr, e1 = nullptr;
  HC(hipStreamCreate(&s));
  if (withEvents) {
    HC(hipEventCreate(&e0));
    HC(hipEventCreate(&e1));
  }
  uint64_t ticks = 1000;
  size_t sz = sizeof(ticks);
  void* extra[5];
  mkExtra(extra, &ticks, &sz);

  hipError_t beg = hipStreamBeginCapture(s, hipStreamCaptureModeGlobal);
  hipError_t lau = hipExtModuleLaunchKernel(fnSpin, kBlocks * kThreads, 1, 1, kThreads, 1, 1, 0, s,
                                            nullptr, extra, e0, e1, 0);
  hipError_t end = hipStreamEndCapture(s, &g);
  printf("    %-28s begin=%-14s launch=%-22s end=%-22s graph=%s\n", what, hipGetErrorName(beg),
         hipGetErrorName(lau), hipGetErrorName(end), g ? "yes" : "null");

  if (g && end == hipSuccess) {
    hipGraphExec_t ge = nullptr;
    hipError_t ins = hipGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    hipError_t run = ge ? hipGraphLaunch(ge, s) : hipErrorInvalidValue;
    hipError_t syn = hipStreamSynchronize(s);
    printf("      %-26s instantiate=%-14s launch=%-14s sync=%s\n", "replay", hipGetErrorName(ins),
           hipGetErrorName(run), hipGetErrorName(syn));
    if (withEvents && syn == hipSuccess) {
      float ms = -1;
      hipError_t el = hipEventElapsedTime(&ms, e0, e1);
      printf("      %-26s elapsed=%s (%.4f ms)\n", "events after graph replay",
             hipGetErrorName(el), ms);
    }
    if (ge) hipGraphExecDestroy(ge);
    hipGraphDestroy(g);
  }
  if (e0) hipEventDestroy(e0);
  if (e1) hipEventDestroy(e1);
  HC(hipStreamDestroy(s));
}

static void testCapture() {
  printf("\n[1] stream capture legality\n");
  captureCase("ext launch, no events", false);
  captureCase("ext launch, with events", true);

  // Baseline: what the plain module launch does under capture, for contrast.
  hipStream_t s;
  hipGraph_t g = nullptr;
  HC(hipStreamCreate(&s));
  uint64_t ticks = 1000;
  size_t sz = sizeof(ticks);
  void* extra[5];
  mkExtra(extra, &ticks, &sz);
  hipError_t beg = hipStreamBeginCapture(s, hipStreamCaptureModeGlobal);
  hipError_t lau =
      hipModuleLaunchKernel(fnSpin, kBlocks, 1, 1, kThreads, 1, 1, 0, s, nullptr, extra);
  hipError_t end = hipStreamEndCapture(s, &g);
  printf("    %-28s begin=%-14s launch=%-22s end=%-22s graph=%s\n", "plain module launch",
         hipGetErrorName(beg), hipGetErrorName(lau), hipGetErrorName(end), g ? "yes" : "null");
  if (g) hipGraphDestroy(g);
  HC(hipStreamDestroy(s));
}

// ---------------------------------------------------------------- test 2
static void testAccuracy() {
  printf("\n[2] measured kernel duration, marker events vs dispatch-attached events\n");
  const double kHz = wallRateKHz();
  printf("    wall clock %.0f kHz\n", kHz);
  printf("    %10s %14s %14s %14s\n", "target us", "marker us", "ext us", "marker-ext");

  hipStream_t s;
  HC(hipStreamCreate(&s));
  hipEvent_t m0, m1, x0, x1;
  HC(hipEventCreate(&m0));
  HC(hipEventCreate(&m1));
  HC(hipEventCreate(&x0));
  HC(hipEventCreate(&x1));

  for (double targetUs : {5.0, 20.0, 100.0, 1000.0}) {
    uint64_t ticks = (uint64_t)(targetUs * kHz / 1000.0);
    size_t sz = sizeof(ticks);
    void* extra[5];
    mkExtra(extra, &ticks, &sz);

    std::vector<double> mk, ex;
    for (int i = 0; i < 20; ++i) {
      HC(hipEventRecord(m0, s));
      HC(hipModuleLaunchKernel(fnSpin, kBlocks, 1, 1, kThreads, 1, 1, 0, s, nullptr, extra));
      HC(hipEventRecord(m1, s));
      HC(hipStreamSynchronize(s));
      float ms = 0;
      HC(hipEventElapsedTime(&ms, m0, m1));
      mk.push_back(ms * 1000.0);

      HC(hipExtModuleLaunchKernel(fnSpin, kBlocks * kThreads, 1, 1, kThreads, 1, 1, 0, s, nullptr,
                                  extra, x0, x1, 0));
      HC(hipStreamSynchronize(s));
      HC(hipEventElapsedTime(&ms, x0, x1));
      ex.push_back(ms * 1000.0);
    }
    std::sort(mk.begin(), mk.end());
    std::sort(ex.begin(), ex.end());
    double m = mk[mk.size() / 2], e = ex[ex.size() / 2];
    printf("    %10.1f %14.2f %14.2f %14.2f\n", targetUs, m, e, m - e);
  }
  HC(hipEventDestroy(m0));
  HC(hipEventDestroy(m1));
  HC(hipEventDestroy(x0));
  HC(hipEventDestroy(x1));
  HC(hipStreamDestroy(s));
}

// ---------------------------------------------------------------- test 3
static void testPerturbation() {
  printf("\n[3] launch cost, back-to-back dispatches of a short kernel\n");
  const int N = 5000;
  const double kHz = wallRateKHz();
  uint64_t ticks = (uint64_t)(5.0 * kHz / 1000.0);  // ~5 us of work
  size_t sz = sizeof(ticks);
  void* extra[5];
  mkExtra(extra, &ticks, &sz);

  hipStream_t s;
  HC(hipStreamCreate(&s));
  std::vector<hipEvent_t> pool(2 * N);
  for (auto& e : pool) HC(hipEventCreate(&e));

  auto run = [&](int mode) {
    HC(hipStreamSynchronize(s));
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
      if (mode == 0) {
        HC(hipModuleLaunchKernel(fnSpin, kBlocks, 1, 1, kThreads, 1, 1, 0, s, nullptr, extra));
      } else if (mode == 1) {
        HC(hipEventRecord(pool[2 * i], s));
        HC(hipModuleLaunchKernel(fnSpin, kBlocks, 1, 1, kThreads, 1, 1, 0, s, nullptr, extra));
        HC(hipEventRecord(pool[2 * i + 1], s));
      } else {
        HC(hipExtModuleLaunchKernel(fnSpin, kBlocks * kThreads, 1, 1, kThreads, 1, 1, 0, s, nullptr,
                                    extra, pool[2 * i], pool[2 * i + 1], 0));
      }
    }
    HC(hipStreamSynchronize(s));
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
  };

  run(0);  // warm
  double plain = std::min({run(0), run(0), run(0)});
  double marker = std::min({run(1), run(1), run(1)});
  double ext = std::min({run(2), run(2), run(2)});
  printf("    %-34s %8.3f us/launch\n", "plain launch, no timing", plain);
  printf("    %-34s %8.3f us/launch  (+%.3f)\n", "launch + 2 marker events", marker,
         marker - plain);
  printf("    %-34s %8.3f us/launch  (+%.3f)\n", "launch with attached events", ext, ext - plain);

  for (auto& e : pool) HC(hipEventDestroy(e));
  HC(hipStreamDestroy(s));
}

int main() {
  hipDeviceProp_t p;
  HC(hipGetDeviceProperties(&p, 0));
  printf("device: %s (%s)\n", p.name, p.gcnArchName);
  HC(hipGetFuncBySymbol(&fnSpin, (const void*)spin));
  HC(hipGetFuncBySymbol(&fnDims, (const void*)reportDims));
  testDims();
  testCapture();
  testAccuracy();
  testPerturbation();
  return 0;
}
