// Cost per launch of each timing scheme, measured back-to-back on a busy stream.
//
// Modes: no timing at all, stream-marker events (what RCCL does today), attached
// start+stop events, and attached stop event only (which the pointer-chase makes
// sufficient, since it carries both dispatch timestamps).

#include "evtstamp.h"

#include <algorithm>
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

enum Mode { NONE, MARKERS, BOTH, STOP_ONLY };
static const char* kModeName[] = {"no timing", "stream markers", "attached start+stop",
                                  "attached stop only"};

static void dispatch(uint64_t ticks, hipEvent_t a, hipEvent_t b, Mode m) {
  size_t sz = sizeof(ticks);
  void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ticks, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                   HIP_LAUNCH_PARAM_END};
  if (m == MARKERS) {
    HC(hipEventRecord(a, g_stream));
    HC(hipExtModuleLaunchKernel(g_fn, 8 * 256, 1, 1, 256, 1, 1, 0, g_stream, nullptr, extra,
                                nullptr, nullptr, 0));
    HC(hipEventRecord(b, g_stream));
  } else {
    HC(hipExtModuleLaunchKernel(g_fn, 8 * 256, 1, 1, 256, 1, 1, 0, g_stream, nullptr, extra,
                                m == BOTH ? a : nullptr, m == NONE ? nullptr : b, 0));
  }
}

int main(int argc, char** argv) {
  int n = argc > 1 ? atoi(argv[1]) : 1000;
  double us = argc > 2 ? atof(argv[2]) : 5.0;

  HC(hipDeviceGetAttribute(&g_khz, hipDeviceAttributeWallClockRate, 0));
  HC(hipGetFuncBySymbol(&g_fn, (const void*)spin));
  HC(hipStreamCreate(&g_stream));
  uint64_t ticks = (uint64_t)(us * g_khz / 1000.0);

  evtstamp::Chain chain = evtstamp::discover([&](double d, hipEvent_t x0, hipEvent_t x1) {
    dispatch((uint64_t)(d * g_khz / 1000.0), x0, x1, BOTH);
    HC(hipStreamSynchronize(g_stream));
  });
  evtstamp::describe(chain);

  printf("\n%d back-to-back launches of a %.0f us kernel\n\n", n, us);
  printf("%-22s %14s %14s %16s\n", "mode", "us/launch", "vs no timing", "kernel us (hack)");

  double baseline = 0;
  for (Mode m : {NONE, MARKERS, BOTH, STOP_ONLY}) {
    std::vector<hipEvent_t> a(n), b(n);
    for (int i = 0; i < n; ++i) {
      HC(hipEventCreate(&a[i]));
      HC(hipEventCreate(&b[i]));
    }
    dispatch(ticks, a[0], b[0], m);  // warm the path
    HC(hipStreamSynchronize(g_stream));

    uint64_t t0 = evtstamp::bootNs();
    for (int i = 0; i < n; ++i) dispatch(ticks, a[i], b[i], m);
    HC(hipStreamSynchronize(g_stream));
    uint64_t t1 = evtstamp::bootNs();

    double per = (t1 - t0) / 1000.0 / n;
    if (m == NONE) baseline = per;

    std::vector<double> k;
    if (m == BOTH || m == STOP_ONLY) {
      for (int i = 0; i < n; ++i) {
        uint64_t s, e;
        if (evtstamp::read(b[i], chain, &s, &e)) k.push_back((e - s) / 1000.0);
      }
      std::sort(k.begin(), k.end());
    }
    printf("%-22s %14.3f %14.3f %16s\n", kModeName[m], per, per - baseline,
           k.empty() ? "-" : [&] {
             static char buf[32];
             snprintf(buf, sizeof(buf), "%.3f", k[k.size() / 2]);
             return buf;
           }());

    for (int i = 0; i < n; ++i) {
      hipEventDestroy(a[i]);
      hipEventDestroy(b[i]);
    }
  }
  return 0;
}
