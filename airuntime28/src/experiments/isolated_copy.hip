// Isolated streaming copy: all nine variants, nothing else running.
//
// This replaces the block-ordered version of this measurement. The old one ran
// every iteration of one variant before starting the next, which credits each
// variant with whatever the machine happened to be doing during its block - and
// it produced the report's headline table while the report's own method section
// listed block ordering as a confounder. The numbers here come from the shared
// harness: one sample per arm per iteration, order reshuffled every iteration,
// every arm duplicated so the rig's own resolution is measured alongside the
// effect.
//
// Source and destination are both flushed out of cache before each sample, so
// this is the cold streaming case: the one that matters for a blit, and the one
// where a non-temporal store hint has the least to work with, since there is no
// resident data for it to protect.

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
  // remote/isa_check.sh reads the expectations from here so they have one
  // definition; see variants.h.
  if (argFlag(argc, argv, "--print-isa-expectations")) {
    printIsaExpectations();
    return 0;
  }

  const u64 size = static_cast<u64>(argValue(argc, argv, "--size-mib", 1024)) * kMiB;
  const int iters = static_cast<int>(argValue(argc, argv, "--iters", kDefaultIters));
  const int warmup = static_cast<int>(argValue(argc, argv, "--warmup", kDefaultWarmup));
  const bool cold = !argFlag(argc, argv, "--warm");

  Machine m = initMachine("isolated copy");
  std::printf("  copy size     : %llu MiB, %s\n",
              static_cast<unsigned long long>(size / kMiB), cold ? "cold (flushed)" : "warm");
  std::printf("  iters/warmup  : %d / %d, %d slots (every arm twice)\n\n", iters, warmup,
              2 * kNumVariants);

  // Correctness before performance. A variant that does not copy correctly has
  // nothing interesting to say about how fast it does not copy correctly.
  for (int v = 0; v < kNumVariants; ++v) verifyVariant(v, m.blitWg);
  std::printf("  byte-exactness: all %d variants verified (unaligned tail included)\n\n",
              kNumVariants);

  u8* src = nullptr;
  u8* dst = nullptr;
  HIP_CHECK(hipMalloc(&src, size));
  HIP_CHECK(hipMalloc(&dst, size));
  HIP_CHECK(hipMemset(src, 0xA5, size));
  HIP_CHECK(hipMemset(dst, 0x00, size));
  reportBuffers(src, dst);

  Scratch scratch;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  EventTimer timer;

  std::vector<std::string> armNames;
  for (int v = 0; v < kNumVariants; ++v) armNames.push_back(kVariants[v].name);
  const SlotPlan plan = SlotPlan::duplicated(armNames);

  const SlotResults r = runSlots(plan, iters, warmup, [&](int slot) {
    const int v = plan.arm[slot];
    return timer.timeMs(
        stream,
        [&] {
          if (cold) scratch.enqueueFlush(stream);
        },
        [&] { enqueueCopy(v, stream, src, dst, size, m.blitWg); });
  });

  printArmTable(r, armNames);

  std::printf("\n  %-20s %10s %10s\n", "arm", "GB/s", "vs plain-128");
  const double baseMs = r.medianMs(r.firstSlotOf(CopyPlain128));
  for (int v = 0; v < kNumVariants; ++v) {
    const double ms = r.medianMs(r.firstSlotOf(v));
    // Two buffer touches per byte moved: one read, one write.
    const double gbs = 2.0 * static_cast<double>(size) / (ms * 1e-3) / 1e9;
    std::printf("  %-20s %10.1f %+9.2f%%\n", kVariants[v].name, gbs, (ms / baseMs - 1.0) * 100.0);
  }

  // Each row below pairs a variant against the control that differs from it in
  // exactly one respect, which is not always plain-128. Comparing the hand-written
  // -asm variants against plain-128 conflates the temporal hint with the effect of
  // writing the store by hand, and that effect turns out to be larger than any
  // hint measured here.
  struct Comparison {
    const char* question;
    int arm;
    int against;
  };
  const Comparison comparisons[] = {
      {"NT store hint, 128-bit (shipped)", CopyNtStore128, CopyPlain128},
      {"also hinting the load", CopyNtBoth128, CopyNtStore128},
      {"hand-written store, default hint", CopyAsmPlain128, CopyPlain128},
      {"TH_STORE_NT_RT hint", CopyNtRtStore128, CopyAsmPlain128},
      {"NT store hint at 64-bit width", CopyNtStore64, CopyPlain64},
      {"NT store hint at 32-bit width", CopyPr2616Actual32, CopyPlain32},
      {"64-bit instead of 128-bit", CopyPlain64, CopyPlain128},
      {"32-bit instead of 128-bit", CopyPlain32, CopyPlain128},
  };

  std::printf("\n  Controlled comparisons: each row changes exactly one thing from its control\n");
  std::printf("  %-34s %-18s %-30s\n", "change", "measured against", "effect [95% CI]");
  std::printf("  %s\n", std::string(84, '-').c_str());
  for (const Comparison& c : comparisons) {
    const Delta d = r.effect(c.arm, c.against);
    char cell[80];
    formatDelta(cell, sizeof(cell), d, r.isSignificant(d));
    std::printf("  %-34s %-18s %-30s\n", c.question, kVariants[c.against].name, cell);
    char extra[96];
    std::snprintf(extra, sizeof(extra), "vs=%s sig=%d", kVariants[c.against].name,
                  static_cast<int>(r.isSignificant(d)));
    emitRow("isolated_copy_controlled", kVariants[c.arm].name, "delta_pct", d.median, "pct", extra);
  }
  std::printf("  negative is faster; '(noise)' means smaller than this run's %.2f pp\n"
              "  resolution limit, so no effect was found\n",
              r.resolutionLimit());

  std::printf("\n");
  for (int v = 0; v < kNumVariants; ++v) {
    const Delta d = r.effect(v);
    char extra[64];
    std::snprintf(extra, sizeof(extra), "ci=%.3f..%.3f sig=%d", d.lo, d.hi,
                  v == 0 ? 0 : static_cast<int>(r.isSignificant(d)));
    emitRow("isolated_copy", kVariants[v].name, "median_ms", r.medianMs(r.firstSlotOf(v)), "ms",
            extra);
    emitRow("isolated_copy", kVariants[v].name, "delta_pct", d.median, "pct", extra);
  }
  emitRow("isolated_copy", "-", "resolution_limit_pp", r.resolutionLimit(), "pp", "");

  // Sign checks on the results the report leans on. These are the conclusions a
  // future run must either reproduce or loudly contradict; a silent flip would
  // otherwise be published as if nothing had changed.
  const Delta plain64 = r.effect(CopyPlain64);
  const Delta plain32 = r.effect(CopyPlain32);
  BENCH_ASSERT(plain64.median > 10.0,
               "expected the 64-bit width penalty to stay large (>10%%), measured %+.2f%%",
               plain64.median);
  BENCH_ASSERT(plain32.median > 10.0,
               "expected the 32-bit width penalty to stay large (>10%%), measured %+.2f%%",
               plain32.median);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(src));
  HIP_CHECK(hipFree(dst));
  flushRows();
  return exitCode();
}
