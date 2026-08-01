// Validate hacked absolute dispatch timestamps against rocprof's kernel trace.
//
// Prints, per launch, the absolute CLOCK_BOOTTIME start/end recovered from the
// stop event alongside hipEventElapsedTime, so the same run can be compared
// line-for-line with rocprofv3 --kernel-trace output.

#include "evtstamp.h"

#include <cstdio>
#include <vector>

#define HC(x)                                                                            \
  do {                                                                                   \
    hipError_t _e = (x);                                                                 \
    if (_e != hipSuccess) printf("FAIL %s -> %s\n", #x, hipGetErrorString(_e));           \
  } while (0)

__global__ void spin(uint64_t ticks) {
  uint64_t t0 = __builtin_amdgcn_s_memrealtime();
  while (__builtin_amdgcn_s_memrealtime() - t0 < ticks) {
  }
}

static hipFunction_t g_fn;
static hipStream_t g_stream;
static int g_khz;
static bool g_stopOnly = false;

static void launchSpin(double us, hipEvent_t x0, hipEvent_t x1) {
  uint64_t ticks = (uint64_t)(us * g_khz / 1000.0);
  size_t sz = sizeof(ticks);
  void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ticks, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                   HIP_LAUNCH_PARAM_END};
  HC(hipExtModuleLaunchKernel(g_fn, 8 * 256, 1, 1, 256, 1, 1, 0, g_stream, nullptr, extra,
                              g_stopOnly ? nullptr : x0, x1, 0));
  HC(hipStreamSynchronize(g_stream));
}

int main(int argc, char** argv) {
  int n = argc > 1 ? atoi(argv[1]) : 10;
  double us = argc > 2 ? atof(argv[2]) : 50.0;
  g_stopOnly = argc > 3 && atoi(argv[3]) != 0;
  printf("start event: %s\n", g_stopOnly ? "OMITTED (stop event only)" : "attached");

  HC(hipDeviceGetAttribute(&g_khz, hipDeviceAttributeWallClockRate, 0));
  HC(hipGetFuncBySymbol(&g_fn, (const void*)spin));
  HC(hipStreamCreate(&g_stream));

  evtstamp::Chain chain = evtstamp::discover(launchSpin);
  evtstamp::describe(chain);
  if (!chain.valid()) return 1;

  printf("\n%d launches of a %.0f us kernel\n", n, us);
  printf("%4s %20s %20s %12s %12s %10s %9s\n", "#", "start(boot ns)", "end(boot ns)", "hacked us",
         "hipElapsed", "delta", "bracket");
  std::vector<double> hacked, elapsed;
  int bracketed = 0;
  for (int i = 0; i < n; ++i) {
    hipEvent_t x0, x1;
    HC(hipEventCreate(&x0));
    HC(hipEventCreate(&x1));
    uint64_t before = evtstamp::bootNs();
    launchSpin(us, x0, x1);
    uint64_t after = evtstamp::bootNs();
    float ms = 0;
    if (!g_stopOnly) HC(hipEventElapsedTime(&ms, x0, x1));
    uint64_t s = 0, e = 0;
    bool ok = evtstamp::read(x1, chain, &s, &e);
    double h = ok ? (e - s) / 1000.0 : -1;
    // The dispatch must fall inside the host window around launch+sync; this is
    // what proves the values are absolute CLOCK_BOOTTIME, not just a delta.
    bool inWindow = ok && s >= before && e <= after;
    bracketed += inWindow;
    printf("%4d %20lu %20lu %12.3f %12.3f %10.3f %9s\n", i, (unsigned long)s, (unsigned long)e, h,
           ms * 1000.0, ms * 1000.0 - h, inWindow ? "ok" : "OUT");
    hacked.push_back(h);
    elapsed.push_back(ms * 1000.0);
    hipEventDestroy(x0);
    hipEventDestroy(x1);
  }
  printf("\n%d/%d dispatches fall inside their host CLOCK_BOOTTIME window\n", bracketed, n);

  auto med = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  printf("\nmedian hacked %.3f us, median hipEventElapsedTime %.3f us, overstatement %.3f us\n",
         med(hacked), med(elapsed), med(elapsed) - med(hacked));
  return 0;
}
