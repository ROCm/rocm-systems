// A copy running alongside a cache-sensitive kernel.
//
// This is the only scenario where the hint has a mechanism to pay on this part.
// Cross-dispatch cache retention does not exist here (see cache_capacity), so a
// copy cannot leave anything useful behind for a later kernel. But a copy running
// *concurrently* with another kernel evicts that kernel's lines while it is still
// using them, and telling the cache not to retain the copy's writes stops that.
//
// The metric is therefore the victim kernel's time, not the copy's.
//
// This experiment replaces victim_sweep.hip and victim_ws_sweep.hip, which
// measured the same thing with 110 lines of difference between them and no way to
// tell which had produced the published table. Both are reproduced here: the
// sweep is the same eight working-set sizes against the same fixed 128 MiB copy.
//
// Working-set size is the axis because getting it wrong is what made the first
// version of this measurement report 1%. A 2 MiB victim was chosen when GL2 was
// believed to be 4 MiB; against the measured ~96 MiB it is trivially resident and
// a copy barely threatens it, so there was nothing for the hint to protect.

#include <cmath>
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

namespace {

// The victim occupies half the CUs; the copy has the rest to contend for cache
// with. Matches the published sweep.
constexpr int kVictimBlocks = 128;
constexpr int kVictimThreads = 256;

// Picks a pass count putting the victim near targetMs when it runs alone, so the
// contention window is long enough to average over and short enough to run many
// times. Calibrated per working-set size, since a larger set needs fewer passes
// for the same duration.
int calibratePasses(const u64x2* ws, u64 wsElems, u64* sink, hipStream_t stream, EventTimer& timer,
                    double targetMs) {
  int passes = 8;
  for (int attempt = 0; attempt < 12; ++attempt) {
    const double ms = timer.timeMs(stream, [&] {
      hipLaunchKernelGGL(sweepReadKernel, dim3(kVictimBlocks), dim3(kVictimThreads), 0, stream, ws,
                         wsElems, passes, sink);
    });
    if (ms > 0.7 * targetMs && ms < 1.4 * targetMs) break;
    const double scale = std::min(8.0, std::max(0.125, targetMs / std::max(ms, 1e-4)));
    const int next = static_cast<int>(std::ceil(passes * scale));
    if (next == passes) break;
    passes = std::max(1, next);
  }
  return passes;
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = static_cast<int>(argValue(argc, argv, "--iters", 25));
  const int warmup = static_cast<int>(argValue(argc, argv, "--warmup", 6));
  const u64 copySize = static_cast<u64>(argValue(argc, argv, "--copy-mib", 128)) * kMiB;
  const double targetMs = 5.0;

  Machine m = initMachine("concurrency: copy vs cache-sensitive victim");
  std::printf("  copy          : %llu MiB, repeated to cover the victim's whole run\n",
              static_cast<unsigned long long>(copySize / kMiB));
  std::printf("  victim        : %d WGs x %d threads sweeping its working set, ~%.0f ms alone\n",
              kVictimBlocks, kVictimThreads, targetMs);
  std::printf("  metric        : the VICTIM's time. Negative means the hint helped it.\n");
  std::printf("  iters/warmup  : %d / %d\n", iters, warmup);

  const std::vector<std::string> arms = {"plain-128", "nt-store-128", "asm-plain-128"};
  const int variantOf[] = {CopyPlain128, CopyNtStore128, CopyAsmPlain128};
  const SlotPlan plan = SlotPlan::duplicated(arms);

  const u64 wsSizes[] = {2 * kMiB,  8 * kMiB,  16 * kMiB, 32 * kMiB,
                         48 * kMiB, 64 * kMiB, 96 * kMiB, 128 * kMiB};
  const u64 maxWs = 128 * kMiB;

  u8* src = nullptr;
  u8* dst = nullptr;
  u64x2* ws = nullptr;
  HIP_CHECK(hipMalloc(&src, copySize));
  HIP_CHECK(hipMalloc(&dst, copySize));
  HIP_CHECK(hipMalloc(&ws, maxWs));
  HIP_CHECK(hipMemset(src, 0xA5, copySize));
  HIP_CHECK(hipMemset(dst, 0x00, copySize));
  HIP_CHECK(hipMemset(ws, 0x33, maxWs));

  Scratch scratch;
  hipStream_t sMain, sCopy;
  HIP_CHECK(hipStreamCreate(&sMain));
  HIP_CHECK(hipStreamCreate(&sCopy));
  EventTimer timer;

  std::printf("\n  %10s %8s %8s %11s %30s %30s %8s\n", "victim_ws", "passes", "copies",
              "alone_ms", "nt-store-128", "asm-plain-128 (control)", "res_lim");
  std::printf("  %s\n", std::string(112, '-').c_str());

  double bestGain = 0.0;
  u64 bestAt = 0;
  int significantGains = 0;
  double coarsestLimit = 0.0;
  for (u64 wsBytes : wsSizes) {
    const u64 wsElems = wsBytes / sizeof(u64x2);
    const int passes = calibratePasses(ws, wsElems, scratch.sink(), sMain, timer, targetMs);

    // How long the victim takes with nothing else running, and how many copies it
    // takes to keep it company for all of that.
    double aloneMs = 0.0;
    {
      std::vector<double> v;
      for (int i = 0; i < warmup + iters; ++i) {
        const double ms = timer.timeMs(sMain, [&] {
          hipLaunchKernelGGL(sweepReadKernel, dim3(kVictimBlocks), dim3(kVictimThreads), 0, sMain,
                             ws, wsElems, passes, scratch.sink());
        });
        if (i >= warmup) v.push_back(ms);
      }
      aloneMs = median(v);
    }
    double copyMs = 0.0;
    {
      std::vector<double> v;
      for (int i = 0; i < 8; ++i)
        v.push_back(timer.timeMs(
            sCopy, [&] { enqueueCopy(CopyPlain128, sCopy, src, dst, copySize, m.blitWg); }));
      copyMs = median(v);
    }
    const int nCopies =
        std::max(1, std::min(20000, static_cast<int>(std::ceil(1.3 * aloneMs /
                                                               std::max(copyMs, 1e-4)))));

    const SlotResults r = runSlots(plan, iters, warmup, [&](int slot) {
      const int v = variantOf[plan.arm[slot]];
      return timer.timeMs(
          sMain,
          [&] {
            scratch.enqueueFlush(sMain);
            // Two passes to bring the working set in before contention starts.
            // Given that nothing survives a dispatch boundary on this part this
            // is a no-op, but it is kept because the published sweep had it and
            // removing it would be an unforced change to the thing being
            // reproduced.
            hipLaunchKernelGGL(sweepReadKernel, dim3(kVictimBlocks), dim3(kVictimThreads), 0, sMain,
                               ws, wsElems, 2, scratch.sink());
            HIP_CHECK(hipStreamSynchronize(sMain));
          },
          [&] {
            hipLaunchKernelGGL(sweepReadKernel, dim3(kVictimBlocks), dim3(kVictimThreads), 0, sMain,
                               ws, wsElems, passes, scratch.sink());
            // Enqueued after the victim, so the copies begin a few microseconds
            // into a ~5 ms window. The asymmetry is identical for every slot.
            for (int k = 0; k < nCopies; ++k)
              enqueueCopy(v, sCopy, src, dst, copySize, m.blitWg);
          },
          [&] { HIP_CHECK(hipStreamSynchronize(sCopy)); });
    });

    const Delta nt = r.effect(1);
    const Delta ctl = r.effect(2);
    const bool ntSig = r.isSignificant(nt);
    char c1[64], c2[64];
    formatDelta(c1, sizeof(c1), nt, ntSig);
    formatDelta(c2, sizeof(c2), ctl, r.isSignificant(ctl));

    std::printf("  %8llu M %8d %8d %11.3f %30s %30s %6.2fpp\n",
                static_cast<unsigned long long>(wsBytes / kMiB), passes, nCopies, aloneMs, c1, c2,
                r.resolutionLimit());
    std::fflush(stdout);

    char extra[96];
    std::snprintf(extra, sizeof(extra), "ws_mib=%llu passes=%d copies=%d sig=%d",
                  static_cast<unsigned long long>(wsBytes / kMiB), passes, nCopies,
                  static_cast<int>(ntSig));
    emitRow("concurrency", "nt-store-128", "victim_delta_pct", nt.median, "pct", extra);
    emitRow("concurrency", "asm-plain-128", "victim_delta_pct", ctl.median, "pct", extra);
    emitRow("concurrency", "plain-128", "victim_alone_ms", aloneMs, "ms", extra);

    coarsestLimit = std::max(coarsestLimit, r.resolutionLimit());
    if (ntSig && nt.median < 0.0) ++significantGains;
    if (ntSig && nt.median < bestGain) {
      bestGain = nt.median;
      bestAt = wsBytes;
    }
  }

  std::printf("\n  best gain: %+.2f%% at a %llu MiB working set; %d of %d sizes significant\n",
              bestGain, static_cast<unsigned long long>(bestAt / kMiB), significantGains,
              static_cast<int>(sizeof(wsSizes) / sizeof(wsSizes[0])));
  std::printf("  The control arm is the same copy with a hand-written store carrying the\n");
  std::printf("  default hint. It should sit near zero: if it moves with the nt arm, what is\n");
  std::printf("  being measured is codegen rather than the temporal hint.\n");
  emitRow("concurrency", "nt-store-128", "best_gain_pct", bestGain, "pct", "");
  emitRow("concurrency", "nt-store-128", "best_gain_at_ws_mib", static_cast<double>(bestAt / kMiB),
          "MiB", "");

  // The published claim is a peak near -5% around 32 MiB and a shelf near -2.4%
  // above it. A future run that no longer sees a gain anywhere has invalidated
  // the only scenario in which this change pays, so say so loudly.
  //
  // Only when the run could have known, though. At a low iteration count the
  // resolution limit exceeds the effect and nothing is significant, which is a
  // property of the run rather than of the hardware; asserting there would make
  // every smoke test look like a regression.
  const double kNeededResolution = 2.0;  // pp, against a ~2.4% shelf
  if (coarsestLimit > kNeededResolution) {
    std::printf("\n  resolution limit reached %.2f pp, coarser than the %.1f pp needed to\n",
                coarsestLimit, kNeededResolution);
    std::printf("  check the published result. Not asserting; re-run with more iterations.\n");
  } else {
    BENCH_ASSERT(significantGains >= 4,
                 "only %d of 8 working-set sizes show a significant gain at a %.2f pp "
                 "resolution limit; the published concurrency benefit no longer reproduces",
                 significantGains, coarsestLimit);
    BENCH_ASSERT(bestGain < -1.5, "best concurrency gain is only %+.2f%%, against a published -5%%",
                 bestGain);
  }

  HIP_CHECK(hipStreamDestroy(sMain));
  HIP_CHECK(hipStreamDestroy(sCopy));
  HIP_CHECK(hipFree(src));
  HIP_CHECK(hipFree(dst));
  HIP_CHECK(hipFree(ws));
  flushRows();
  return exitCode();
}
