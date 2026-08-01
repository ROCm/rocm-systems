// Is the HIP event's timestamp really ROCr's, and can it be read through the
// documented HSA API instead of raw struct offsets?
//
// A ROCr signal handle is a pointer to amd_signal_t, so a candidate qword found
// in the event's object graph can be screened before use: it must be readable,
// its kind field must be AMD_SIGNAL_KIND_USER, and its start_ts/end_ts must be
// populated. Only then is it handed to hsa_amd_profiling_get_dispatch_time --
// passing a bogus handle would fault inside the runtime.

#include "evtstamp.h"

#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <cstdio>
#include <map>
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

static hsa_agent_t g_gpu;
static hsa_status_t findGpu(hsa_agent_t a, void*) {
  hsa_device_type_t t;
  hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
  if (t == HSA_DEVICE_TYPE_GPU) {
    g_gpu = a;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

// Collect every qword reachable from the event, with the path taken to it.
static void collect(uint64_t base, int depth, std::vector<int>& path, std::set<uint64_t>& seen,
                    std::map<uint64_t, std::vector<int>>& out) {
  if (depth > 3 || !seen.insert(base).second) return;
  uint64_t w[32];
  if (!evtstamp::detail::safeRead(base, w, sizeof(w))) return;
  for (int i = 0; i < 32; ++i) {
    path.push_back(i * 8);
    if (w[i] > 0x10000 && (w[i] & 7) == 0) {
      if (out.find(w[i]) == out.end()) out[w[i]] = path;
      collect(w[i], depth + 1, path, seen, out);
    }
    path.pop_back();
  }
}

int main(int argc, char** argv) {
  double us = argc > 1 ? atof(argv[1]) : 50.0;

  int khz = 0;
  HC(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
  hipFunction_t fn;
  HC(hipGetFuncBySymbol(&fn, (const void*)spin));
  hipStream_t s;
  HC(hipStreamCreate(&s));
  hsa_iterate_agents(findGpu, nullptr);

  uint64_t ticks = (uint64_t)(us * khz / 1000.0);
  size_t sz = sizeof(ticks);
  void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ticks, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                   HIP_LAUNCH_PARAM_END};

  hipEvent_t x1;
  HC(hipEventCreate(&x1));
  uint64_t before = evtstamp::bootNs();
  HC(hipExtModuleLaunchKernel(fn, 8 * 256, 1, 1, 256, 1, 1, 0, s, nullptr, extra, nullptr, x1, 0));
  HC(hipStreamSynchronize(s));
  uint64_t after = evtstamp::bootNs();

  std::map<uint64_t, std::vector<int>> cands;
  {
    std::vector<int> p;
    std::set<uint64_t> seen;
    collect((uint64_t)x1, 0, p, seen, cands);
  }
  printf("%zu pointer-like values reachable from the stop event\n", cands.size());

  printf("\ncandidates that look like an amd_signal_t (kind == USER, timestamps set):\n");
  for (auto& kv : cands) {
    uint64_t sig = kv.first;
    struct {
      int64_t kind;
      int64_t value;
      uint64_t mailbox;
      uint32_t id, rsvd;
      uint64_t start_ts, end_ts;
    } hdr;
    if (!evtstamp::detail::safeRead(sig, &hdr, sizeof(hdr))) continue;
    if (hdr.kind != AMD_SIGNAL_KIND_USER) continue;
    if (hdr.start_ts == 0 || hdr.end_ts <= hdr.start_ts) continue;

    printf("  signal @ %#lx  path event", (unsigned long)sig);
    for (int h : kv.second) printf(" -> *(+%d)", h);
    printf("\n    raw ticks   start=%lu end=%lu  span=%lu ticks\n", (unsigned long)hdr.start_ts,
           (unsigned long)hdr.end_ts, (unsigned long)(hdr.end_ts - hdr.start_ts));

    uint64_t cs = 0, ce = 0;
    hsa_amd_profiling_convert_tick_to_system_domain(g_gpu, hdr.start_ts, &cs);
    hsa_amd_profiling_convert_tick_to_system_domain(g_gpu, hdr.end_ts, &ce);
    printf("    converted   start=%lu end=%lu  span=%.3f us  (in host window: %s)\n",
           (unsigned long)cs, (unsigned long)ce, (ce - cs) / 1000.0,
           (cs >= before && ce <= after) ? "yes" : "NO");

    hsa_amd_profiling_dispatch_time_t t{};
    hsa_status_t st = hsa_amd_profiling_get_dispatch_time(g_gpu, {sig}, &t);
    printf("    hsa_amd_profiling_get_dispatch_time -> %s", st == HSA_STATUS_SUCCESS ? "ok" : "err");
    if (st == HSA_STATUS_SUCCESS)
      printf("  start=%lu end=%lu  span=%.3f us", (unsigned long)t.start, (unsigned long)t.end,
             (t.end - t.start) / 1000.0);
    printf("\n");
  }

  evtstamp::Chain cached;
  cached.hop = {88, 248};
  cached.leaf = 88;
  evtstamp::Chain sigChain;
  sigChain.hop = {88, 248};
  sigChain.leaf = 16;  // leaf read gives {signal, next qword}; only the first is used

  int n = argc > 2 ? atoi(argv[2]) : 20;
  printf("\n%d launches: ROCclr cached copy vs hsa_amd_profiling_get_dispatch_time\n\n", n);
  printf("%4s %14s %14s %14s %14s\n", "#", "cached us", "hsa us", "span diff ns", "start diff ns");
  std::vector<double> startDiff, spanDiff;
  for (int i = 0; i < n; ++i) {
    hipEvent_t e1;
    HC(hipEventCreate(&e1));
    HC(hipExtModuleLaunchKernel(fn, 8 * 256, 1, 1, 256, 1, 1, 0, s, nullptr, extra, nullptr, e1,
                                0));
    HC(hipStreamSynchronize(s));

    uint64_t cs = 0, ce = 0, sig = 0, dummy = 0;
    evtstamp::read(e1, cached, &cs, &ce);
    evtstamp::detail::follow(e1, sigChain, &sig, &dummy);
    hsa_amd_profiling_dispatch_time_t t{};
    hsa_status_t st = hsa_amd_profiling_get_dispatch_time(g_gpu, {sig}, &t);
    if (st != HSA_STATUS_SUCCESS) {
      printf("%4d  hsa_amd_profiling_get_dispatch_time failed\n", i);
      hipEventDestroy(e1);
      continue;
    }
    double sd = (double)((int64_t)cs - (int64_t)t.start);
    double pd = (double)((int64_t)(ce - cs) - (int64_t)(t.end - t.start));
    printf("%4d %14.3f %14.3f %14.0f %14.0f\n", i, (ce - cs) / 1000.0, (t.end - t.start) / 1000.0,
           pd, sd);
    startDiff.push_back(sd);
    spanDiff.push_back(pd);
    hipEventDestroy(e1);
  }
  if (!startDiff.empty()) {
    std::sort(startDiff.begin(), startDiff.end());
    std::sort(spanDiff.begin(), spanDiff.end());
    printf("\nstart offset: median %.0f ns, min %.0f, max %.0f\n", startDiff[startDiff.size() / 2],
           startDiff.front(), startDiff.back());
    printf("span  offset: median %.0f ns, min %.0f, max %.0f\n", spanDiff[spanDiff.size() / 2],
           spanDiff.front(), spanDiff.back());
  }

  // Durations are exact tick deltas, but absolute values depend on ROCr's
  // tick/system calibration. Re-convert one real dispatch's ticks over time to
  // see how far an absolute timestamp can move.
  hipEvent_t keep;
  HC(hipEventCreate(&keep));
  HC(hipExtModuleLaunchKernel(fn, 8 * 256, 1, 1, 256, 1, 1, 0, s, nullptr, extra, nullptr, keep,
                              0));
  HC(hipStreamSynchronize(s));
  uint64_t sig = 0, dummy = 0, rawStart = 0;
  if (evtstamp::detail::follow(keep, sigChain, &sig, &dummy) &&
      evtstamp::detail::safeRead(sig + 32, &rawStart, 8)) {
    printf("\nre-converting one real dispatch tick (%lu) over 2 s\n", (unsigned long)rawStart);
    uint64_t base = 0;
    for (int i = 0; i <= 10; ++i) {
      uint64_t v = 0;
      hsa_amd_profiling_convert_tick_to_system_domain(g_gpu, rawStart, &v);
      if (i == 0) base = v;
      printf("  t=%.1fs  %lu  drift %+ld ns\n", i * 0.2, (unsigned long)v,
             (long)((int64_t)v - (int64_t)base));
      usleep(200000);
    }
  }
  return 0;
}
