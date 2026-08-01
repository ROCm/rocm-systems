// Follow-ups to probe.cpp:
//  A  separate the cost of the ext launch path from the cost of the events
//  B  compare marker and dispatch-attached timing on the SAME launches, with a
//     busy stream, which is the case RCCL actually runs in

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
    if (_e != hipSuccess) printf("    FAIL %s -> %s\n", #x, hipGetErrorString(_e));       \
  } while (0)

__global__ void spin(uint64_t ticks) {
  uint64_t t0 = __builtin_amdgcn_s_memrealtime();
  while (__builtin_amdgcn_s_memrealtime() - t0 < ticks) {
  }
}

static hipFunction_t fnSpin;
static const unsigned kBlocks = 8, kThreads = 256;

static double wallRateKHz() {
  int khz = 0;
  hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0);
  return khz ? khz : 25000.0;
}

static void mkExtra(void** extra, uint64_t* arg, size_t* sz) {
  extra[0] = HIP_LAUNCH_PARAM_BUFFER_POINTER;
  extra[1] = arg;
  extra[2] = HIP_LAUNCH_PARAM_BUFFER_SIZE;
  extra[3] = sz;
  extra[4] = HIP_LAUNCH_PARAM_END;
}

// ---------------------------------------------------------------- A
static void testLaunchCost(double kernelUs) {
  const int N = 4000;
  const double kHz = wallRateKHz();
  uint64_t ticks = (uint64_t)(kernelUs * kHz / 1000.0);
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
      switch (mode) {
        case 0:
          HC(hipModuleLaunchKernel(fnSpin, kBlocks, 1, 1, kThreads, 1, 1, 0, s, nullptr, extra));
          break;
        case 1:  // ext path, no events: isolates the cost of the API itself
          HC(hipExtModuleLaunchKernel(fnSpin, kBlocks * kThreads, 1, 1, kThreads, 1, 1, 0, s,
                                      nullptr, extra, nullptr, nullptr, 0));
          break;
        case 2:  // today's RCCL: two marker packets around a plain launch
          HC(hipEventRecord(pool[2 * i], s));
          HC(hipModuleLaunchKernel(fnSpin, kBlocks, 1, 1, kThreads, 1, 1, 0, s, nullptr, extra));
          HC(hipEventRecord(pool[2 * i + 1], s));
          break;
        case 3:  // proposed: events attached to the dispatch
          HC(hipExtModuleLaunchKernel(fnSpin, kBlocks * kThreads, 1, 1, kThreads, 1, 1, 0, s,
                                      nullptr, extra, pool[2 * i], pool[2 * i + 1], 0));
          break;
      }
    }
    HC(hipStreamSynchronize(s));
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
  };

  run(0);
  double a = std::min({run(0), run(0), run(0)});
  double b = std::min({run(1), run(1), run(1)});
  double c = std::min({run(2), run(2), run(2)});
  double d = std::min({run(3), run(3), run(3)});
  printf("\n[A] wall time per launch, %.0f us kernel, %d back-to-back\n", kernelUs, N);
  printf("    %-36s %8.3f us\n", "plain module launch", a);
  printf("    %-36s %8.3f us  (%+.3f)\n", "ext launch, no events", b, b - a);
  printf("    %-36s %8.3f us  (%+.3f)\n", "plain + 2 marker events (today)", c, c - a);
  printf("    %-36s %8.3f us  (%+.3f)\n", "ext launch with events (proposed)", d, d - a);

  for (auto& e : pool) HC(hipEventDestroy(e));
  HC(hipStreamDestroy(s));
}

// ---------------------------------------------------------------- B
static void testAccuracyUnderLoad(double kernelUs) {
  const int N = 200;
  const double kHz = wallRateKHz();
  uint64_t ticks = (uint64_t)(kernelUs * kHz / 1000.0);
  size_t sz = sizeof(ticks);
  void* extra[5];
  mkExtra(extra, &ticks, &sz);

  hipStream_t s;
  HC(hipStreamCreate(&s));
  std::vector<hipEvent_t> m0(N), m1(N), x0(N), x1(N);
  for (int i = 0; i < N; ++i) {
    HC(hipEventCreate(&m0[i]));
    HC(hipEventCreate(&m1[i]));
    HC(hipEventCreate(&x0[i]));
    HC(hipEventCreate(&x1[i]));
  }

  // Both timing schemes on the same dispatches, queued back-to-back with no
  // synchronization, so the stream stays busy the way it does under a real
  // workload.
  for (int i = 0; i < N; ++i) {
    HC(hipEventRecord(m0[i], s));
    HC(hipExtModuleLaunchKernel(fnSpin, kBlocks * kThreads, 1, 1, kThreads, 1, 1, 0, s, nullptr,
                                extra, x0[i], x1[i], 0));
    HC(hipEventRecord(m1[i], s));
  }
  HC(hipStreamSynchronize(s));

  std::vector<double> mk, ex;
  for (int i = 0; i < N; ++i) {
    float a = 0, b = 0;
    HC(hipEventElapsedTime(&a, m0[i], m1[i]));
    HC(hipEventElapsedTime(&b, x0[i], x1[i]));
    mk.push_back(a * 1000.0);
    ex.push_back(b * 1000.0);
  }
  std::sort(mk.begin(), mk.end());
  std::sort(ex.begin(), ex.end());
  auto pct = [](std::vector<double>& v, double p) { return v[(size_t)(p * (v.size() - 1))]; };

  printf("\n[B] same %d dispatches, busy stream, %.0f us kernel\n", N, kernelUs);
  printf("    %-22s %9s %9s %9s %9s\n", "", "min", "median", "p90", "max");
  printf("    %-22s %9.2f %9.2f %9.2f %9.2f\n", "marker events us", mk.front(), pct(mk, .5),
         pct(mk, .9), mk.back());
  printf("    %-22s %9.2f %9.2f %9.2f %9.2f\n", "attached events us", ex.front(), pct(ex, .5),
         pct(ex, .9), ex.back());
  printf("    median overstatement by markers: %+.2f us (%+.1f%%)\n", pct(mk, .5) - pct(ex, .5),
         100.0 * (pct(mk, .5) - pct(ex, .5)) / pct(ex, .5));

  for (int i = 0; i < N; ++i) {
    HC(hipEventDestroy(m0[i]));
    HC(hipEventDestroy(m1[i]));
    HC(hipEventDestroy(x0[i]));
    HC(hipEventDestroy(x1[i]));
  }
  HC(hipStreamDestroy(s));
}

int main() {
  hipDeviceProp_t p;
  HC(hipGetDeviceProperties(&p, 0));
  printf("device: %s (%s)\n", p.name, p.gcnArchName);
  HC(hipGetFuncBySymbol(&fnSpin, (const void*)spin));
  for (double us : {5.0, 50.0}) testLaunchCost(us);
  for (double us : {5.0, 50.0}) testAccuracyUnderLoad(us);
  return 0;
}
