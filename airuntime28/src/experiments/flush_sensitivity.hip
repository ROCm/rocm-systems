// Is the flush big enough for "cold" to mean cold?
//
// Every cold measurement in this investigation depends on one constant, and that
// constant was chosen under a mistake. 256 MiB was picked believing GL2 was
// 4 MiB, which made it a 64x margin; against the measured ~96 MiB it is 2.7x. A
// 2.7x margin is not obviously enough, and if it is not, then "cold" copies have
// been starting with part of the source or destination still resident, which
// flatters or penalises variants unevenly depending on how they treat the cache.
//
// So sweep it. For each candidate flush size, measure the two things that would
// change if the flush were inadequate:
//
//   1. Cold dependent-load latency over a footprint that fits in GL2. An
//      inadequate flush leaves lines behind, and the latency drops toward the
//      warm figure. This is the sensitive probe - latency, not bandwidth.
//   2. The copy time and the non-temporal effect at a mid-range size. This is
//      what the flush constant is actually used for.
//
// Pick the smallest size at which both stop moving. If the latency figure is
// still falling at the largest flush tried, no size tested is adequate and that
// is itself the result.

#include <cstdio>
#include <string>
#include <vector>

#include "../common/check.h"
#include "../common/config.h"
#include "../common/geometry.h"
#include "../common/harness.h"
#include "../common/kernels.h"
#include "../common/stats.h"
#include "../common/variants.h"

using namespace bench;

int main(int argc, char** argv) {
  const int iters = static_cast<int>(argValue(argc, argv, "--iters", 20));
  const int warmup = static_cast<int>(argValue(argc, argv, "--warmup", 5));
  // A footprint comfortably inside measured GL2, so an adequate flush must evict
  // it and an inadequate one will not.
  const u64 probeBytes = static_cast<u64>(argValue(argc, argv, "--probe-mib", 32)) * kMiB;
  const u64 copyBytes = static_cast<u64>(argValue(argc, argv, "--copy-mib", 128)) * kMiB;

  Machine m = initMachine("flush sensitivity");
  std::printf("  probe/copy    : %llu MiB chase footprint, %llu MiB copy\n",
              static_cast<unsigned long long>(probeBytes / kMiB),
              static_cast<unsigned long long>(copyBytes / kMiB));
  std::printf("  iters/warmup  : %d / %d\n", iters, warmup);

  const u64 flushSizes[] = {128 * kMiB, 256 * kMiB, 512 * kMiB, 1 * kGiB, 2 * kGiB};

  u8* probe = nullptr;
  u8* src = nullptr;
  u8* dst = nullptr;
  HIP_CHECK(hipMalloc(&probe, probeBytes));
  HIP_CHECK(hipMalloc(&src, copyBytes));
  HIP_CHECK(hipMalloc(&dst, copyBytes));
  HIP_CHECK(hipMemset(src, 0xA5, copyBytes));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  EventTimer timer;
  Chase chase(probe, probeBytes);
  const u64 steps = std::min<u64>(probeBytes / 64, 20000);

  // Reference: the same walk with the previous dispatch having populated the
  // cache. Cold latency approaching this value means the flush is not working.
  double warmRef = 0.0;
  {
    u64* sink = nullptr;
    HIP_CHECK(hipMalloc(&sink, 64 * sizeof(u64)));
    std::vector<double> v;
    for (int i = 0; i < warmup + iters; ++i) {
      const double ms = timer.timeMs(
          stream, [&] { chase.enqueueSingleLane(stream, probe, steps, sink); },
          [&] { chase.enqueueSingleLane(stream, probe, steps, sink); });
      if (i >= warmup) v.push_back(ms * 1e6 / static_cast<double>(steps));
    }
    warmRef = median(v);
    HIP_CHECK(hipFree(sink));
  }
  std::printf("\n  warm reference (no flush, cache populated by previous dispatch): "
              "%.1f ns/hop\n\n",
              warmRef);

  std::printf("  %10s %8s %14s %10s %14s %30s\n", "flush", "xGL2", "cold_ns/hop", "vs warm",
              "plain-128_ms", "nt-store-128 [95% CI]");
  std::printf("  %s\n", std::string(92, '-').c_str());

  std::vector<double> coldByFlush;
  std::vector<double> ntByFlush;
  std::vector<double> copyByFlush;

  for (u64 flushBytes : flushSizes) {
    Scratch scratch(flushBytes);

    std::vector<double> lat;
    for (int i = 0; i < warmup + iters; ++i) {
      const double ms = timer.timeMs(
          stream, [&] { scratch.enqueueFlush(stream); },
          [&] { chase.enqueueSingleLane(stream, probe, steps, scratch.sink()); });
      if (i >= warmup) lat.push_back(ms * 1e6 / static_cast<double>(steps));
    }
    const double cold = median(lat);

    // The copy comparison uses the same shared harness as every other result, so
    // a difference between flush sizes cannot come from a difference in method.
    const std::vector<std::string> arms = {"plain-128", "nt-store-128"};
    const int variantOf[2] = {CopyPlain128, CopyNtStore128};
    const SlotPlan plan = SlotPlan::duplicated(arms);
    const SlotResults r = runSlots(plan, iters, warmup, [&](int slot) {
      const int v = variantOf[plan.arm[slot]];
      return timer.timeMs(
          stream, [&] { scratch.enqueueFlush(stream); },
          [&] { enqueueCopy(v, stream, src, dst, copyBytes, m.blitWg); });
    });
    const Delta nt = r.effect(1);
    char cell[64];
    formatDelta(cell, sizeof(cell), nt, r.isSignificant(nt));

    std::printf("  %8llu M %7.1fx %14.1f %9.2fx %14.4f %30s\n",
                static_cast<unsigned long long>(flushBytes / kMiB),
                static_cast<double>(flushBytes) / static_cast<double>(kGL2Bytes), cold,
                cold / warmRef, r.medianMs(r.firstSlotOf(0)), cell);
    std::fflush(stdout);

    char extra[64];
    std::snprintf(extra, sizeof(extra), "flush_mib=%llu",
                  static_cast<unsigned long long>(flushBytes / kMiB));
    emitRow("flush_sensitivity", "chase", "cold_ns_per_hop", cold, "ns", extra);
    emitRow("flush_sensitivity", "plain-128", "copy_ms", r.medianMs(r.firstSlotOf(0)), "ms", extra);
    emitRow("flush_sensitivity", "nt-store-128", "delta_pct", nt.median, "pct", extra);
    coldByFlush.push_back(cold);
    ntByFlush.push_back(nt.median);
    copyByFlush.push_back(r.medianMs(r.firstSlotOf(0)));
  }

  // Where does it stop moving? The smallest flush whose cold latency and copy
  // time are both within 2% of the largest flush's is the smallest adequate one.
  std::printf("\n  %10s %14s %14s\n", "flush", "cold_vs_max", "copy_vs_max");
  const double refCold = coldByFlush.back();
  const double refCopy = copyByFlush.back();
  int chosen = static_cast<int>(sizeof(flushSizes) / sizeof(flushSizes[0])) - 1;
  for (int i = static_cast<int>(coldByFlush.size()) - 1; i >= 0; --i) {
    const double dc = (coldByFlush[i] / refCold - 1.0) * 100.0;
    const double dk = (copyByFlush[i] / refCopy - 1.0) * 100.0;
    std::printf("  %8llu M %13.2f%% %13.2f%%\n",
                static_cast<unsigned long long>(flushSizes[i] / kMiB), dc, dk);
    if (std::abs(dc) <= 2.0 && std::abs(dk) <= 2.0) chosen = i;
  }
  std::printf("\n  smallest flush where both are within 2%% of the 2 GiB result: %llu MiB\n",
              static_cast<unsigned long long>(flushSizes[chosen] / kMiB));
  std::printf("  config.h currently uses %llu MiB.%s\n",
              static_cast<unsigned long long>(kFlushBytes / kMiB),
              flushSizes[chosen] <= kFlushBytes ? " Adequate." : " Smaller than the sweep wants.");
  emitRow("flush_sensitivity", "-", "smallest_adequate_mib",
          static_cast<double>(flushSizes[chosen] / kMiB), "MiB", "");

  // The finding this experiment exists to establish is that flush size does not
  // matter here, because the dispatch boundary has already invalidated GL2. That
  // is what to assert: a run where cold latency starts tracking flush size means
  // residency has appeared, which would overturn FINDING-gl2-residency.md and
  // make every flush constant in the suite load-bearing again.
  //
  // Asserting on the flush *choice* instead would be fragile - at a low iteration
  // count noise alone can push a size out of the 2% band and pick a larger one.
  double lo = coldByFlush[0], hi = coldByFlush[0];
  for (double c : coldByFlush) {
    lo = std::min(lo, c);
    hi = std::max(hi, c);
  }
  const double spreadPct = (hi / lo - 1.0) * 100.0;
  std::printf("  cold latency spread across a 16x range of flush sizes: %.2f%%\n", spreadPct);
  emitRow("flush_sensitivity", "-", "cold_latency_spread_pct", spreadPct, "pct", "");
  BENCH_ASSERT(spreadPct < 10.0,
               "cold latency varies by %.2f%% across flush sizes, so cache state now survives "
               "a dispatch and every flush constant in this suite needs re-picking",
               spreadPct);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(probe));
  HIP_CHECK(hipFree(src));
  HIP_CHECK(hipFree(dst));
  flushRows();
  return exitCode();
}
