// Where in the size range does the store hint do anything?
//
// A single headline size cannot answer that. Below a few MiB the dispatch itself
// dominates and no store hint can matter; in the middle band the copy is large
// enough to be memory-bound but small enough that its footprint interacts with
// GL2; at 1 GiB the copy is several times GL2 in each direction and cache policy
// has almost nothing left to influence. The interesting question is whether
// there is a band where the hint pays, and the answer only shows up as a curve.
//
// Every point uses the shared harness, so a difference between two sizes cannot
// come from a difference in method.

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
  const int iters = static_cast<int>(argValue(argc, argv, "--iters", 25));
  const int warmup = static_cast<int>(argValue(argc, argv, "--warmup", 6));

  Machine m = initMachine("size curve");
  std::printf("  iters/warmup  : %d / %d\n", iters, warmup);

  const std::vector<std::string> arms = {"plain-128", "nt-store-128", "nt-both-128",
                                         "ntrt-store-128"};
  const int variantOf[] = {CopyPlain128, CopyNtStore128, CopyNtBoth128, CopyNtRtStore128};
  const int nArms = static_cast<int>(arms.size());
  const SlotPlan plan = SlotPlan::duplicated(arms);

  // --sizes-mib 8,12,16,... replaces the default sweep, for probing a feature of
  // the curve at finer spacing than a full sweep would be worth running.
  std::vector<u64> sizes;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) != "--sizes-mib") continue;
    std::string list(argv[i + 1]);
    size_t pos = 0;
    while (pos <= list.size()) {
      const size_t comma = list.find(',', pos);
      const std::string tok = list.substr(pos, comma - pos);
      if (!tok.empty()) sizes.push_back(static_cast<u64>(std::atoll(tok.c_str())) * kMiB);
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }
  if (sizes.empty())
    sizes = {64 * kKiB,  256 * kKiB, 1 * kMiB,   4 * kMiB,   16 * kMiB, 32 * kMiB, 64 * kMiB,
             96 * kMiB,  128 * kMiB, 192 * kMiB, 256 * kMiB, 512 * kMiB, 1 * kGiB};

  u64 maxSize = 0;
  for (u64 s : sizes) maxSize = std::max(maxSize, s);

  u8* src = nullptr;
  u8* dst = nullptr;
  HIP_CHECK(hipMalloc(&src, maxSize));
  HIP_CHECK(hipMalloc(&dst, maxSize));
  HIP_CHECK(hipMemset(src, 0xA5, maxSize));
  // Both buffers are touched in full before any timing. Without this, the first
  // size to reach into a previously-untouched part of the destination pays a
  // first-write cost that belongs to the allocator, not to the copy.
  HIP_CHECK(hipMemset(dst, 0x00, maxSize));

  Scratch scratch;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  EventTimer timer;

  std::printf("\n  Each cell is the paired median difference against plain-128 at that size,\n");
  std::printf("  with a 95%% bootstrap interval. '(ns)' = not separable: the effect does not\n");
  std::printf("  exceed res_lim, this rig's own error measured from duplicate slots in the\n");
  std::printf("  same run, so no effect was found at that size.\n\n");
  std::printf("  %10s %11s %9s", "size", "plain_ms", "GB/s");
  for (int a = 1; a < nArms; ++a) std::printf(" %26s", arms[a].c_str());
  std::printf("  %8s\n", "res_lim");

  double bestGain = 0.0;
  u64 bestAt = 0;
  for (u64 size : sizes) {
    const SlotResults r = runSlots(plan, iters, warmup, [&](int slot) {
      const int v = variantOf[plan.arm[slot]];
      return timer.timeMs(
          stream, [&] { scratch.enqueueFlush(stream); },
          [&] { enqueueCopy(v, stream, src, dst, size, m.blitWg); });
    });

    const double baseMs = r.medianMs(r.firstSlotOf(0));
    const double gbs = 2.0 * static_cast<double>(size) / (baseMs * 1e-3) / 1e9;
    char label[24];
    if (size >= kMiB)
      std::snprintf(label, sizeof(label), "%lluMiB", static_cast<unsigned long long>(size / kMiB));
    else
      std::snprintf(label, sizeof(label), "%lluKiB", static_cast<unsigned long long>(size / kKiB));

    std::printf("  %10s %11.4f %9.0f", label, baseMs, gbs);
    for (int a = 1; a < nArms; ++a) {
      const Delta d = r.effect(a);
      const bool sig = r.isSignificant(d);
      char cell[64];
      formatDelta(cell, sizeof(cell), d, sig);
      std::printf(" %26s", cell);
      char extra[64];
      std::snprintf(extra, sizeof(extra), "size=%s sig=%d", label, static_cast<int>(sig));
      emitRow("size_curve", arms[a].c_str(), "delta_pct", d.median, "pct", extra);
      if (a == 1 && sig && d.median < bestGain) {
        bestGain = d.median;
        bestAt = size;
      }
    }
    std::printf("  %7.2fpp\n", r.resolutionLimit());
    std::fflush(stdout);

    char extra[64];
    std::snprintf(extra, sizeof(extra), "size=%s", label);
    emitRow("size_curve", "plain-128", "copy_ms", baseMs, "ms", extra);
    emitRow("size_curve", "plain-128", "bandwidth_rw", gbs, "GB/s", extra);
  }

  std::printf("\n  Absolute times between about 16 and 48 MiB are not a smooth function of\n");
  std::printf("  size: they sit on flat plateaus (~103 us for 16-20 MiB, ~56 us for 24-48 MiB)\n");
  std::printf("  independent of how much data is moved, and a run occasionally lands on the\n");
  std::printf("  faster plateau at a size that usually takes the slower one. Repeatability\n");
  std::printf("  within a run is under 1%%, so the paired columns are unaffected - both arms\n");
  std::printf("  sit on the same plateau - but the ms and GB/s columns in that band describe\n");
  std::printf("  a run rather than the hardware. Cause not established; see REPORT.md limits.\n");

  std::printf("\n  best significant nt-store-128 gain: %+.2f%% at %llu MiB\n", bestGain,
              static_cast<unsigned long long>(bestAt / kMiB));
  std::printf("  measured GL2 is %llu MiB, so a copy of S touches 2S and stops fitting\n",
              static_cast<unsigned long long>(kGL2Bytes / kMiB));
  std::printf("  above about %llu MiB.\n", static_cast<unsigned long long>(kGL2Bytes / kMiB / 2));
  emitRow("size_curve", "nt-store-128", "best_gain_pct", bestGain, "pct", "");
  emitRow("size_curve", "nt-store-128", "best_gain_at_mib", static_cast<double>(bestAt / kMiB),
          "MiB", "");

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(src));
  HIP_CHECK(hipFree(dst));
  flushRows();
  return exitCode();
}
