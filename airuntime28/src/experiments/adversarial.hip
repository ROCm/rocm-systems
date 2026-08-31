// Deliberate attempts to make the non-temporal store hint lose.
//
// The hint tells the cache not to retain what a copy writes. Every way that can
// cost something has the same shape: somebody wanted the copied data, or the
// space it displaced, soon afterwards. So each case below constructs a consumer
// with that property and times the consumer, not the copy.
//
// Footprint is the correction that forced this rerun. The original suite ran
// every case at 128 MiB, chosen when GL2 was believed to be 4 MiB. A 128 MiB copy
// touches 256 MiB, nearly three times the measured ~96 MiB GL2, so nothing it
// wrote could have stayed resident under any store policy - the cases meant to
// punish the hint could not have detected damage even if there were some, and
// "no case found" was therefore untested. Each case now runs at three footprints:
// half of GL2, all of GL2, and well past it.

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

constexpr int kStreams = 4;
constexpr int kPoolBuffers = 5;

struct Rig {
  Machine m;
  Scratch* scratch = nullptr;
  hipStream_t s[kStreams] = {};
  EventTimer* timer = nullptr;
  u8* pool[kPoolBuffers] = {};
  int iters = 15;
  int warmup = 4;
  int worseCases = 0;
  // Worst news the suite found, and the coarsest resolution it found it with.
  // Reported together because "no case found" is only as strong as the smallest
  // regression the suite could have detected.
  double mostAdverse = -1e9;
  std::string mostAdverseCase;
  double widestLimit = 0.0;
  std::string widestLimitCase;
};

Rig g;

void enqueueConsume(hipStream_t s, const u8* buf, u64 bytes) {
  hipLaunchKernelGGL(sweepReadKernel, dim3(1024), dim3(256), 0, s,
                     reinterpret_cast<const u64x2*>(buf), bytes / sizeof(u64x2), 1,
                     g.scratch->sink());
}

// One case: two variants, each in two slots, order reshuffled every iteration.
//
// setup() is untimed, body() is timed. Which of the two the copy goes in is the
// design of each case: where the copy is timed we are asking whether the hint
// makes the copy faster, and where the consumer is timed we are asking whether
// the hint made the consumer slower, which is the question that matters here.
template <typename Setup, typename Body>
void runCase(const char* name, const char* mechanism, u64 footprint, int plainVariant,
             int ntVariant, Setup setup, Body body) {
  const SlotPlan plan = SlotPlan::duplicated({"plain", "nt"});
  const int variantOf[2] = {plainVariant, ntVariant};

  const SlotResults r = runSlots(plan, g.iters, g.warmup, [&](int slot) {
    const int v = variantOf[plan.arm[slot]];
    return g.timer->timeMs(
        g.s[0],
        [&] {
          g.scratch->enqueueFlush(g.s[0]);
          setup(v);
          HIP_CHECK(hipStreamSynchronize(g.s[0]));
        },
        [&] { body(v); },
        [&] {
          for (int k = 0; k < kStreams; ++k) HIP_CHECK(hipStreamSynchronize(g.s[k]));
        });
  });

  const Delta eff = r.effect(1);
  const bool sig = r.isSignificant(eff);
  const char* verdict = !sig                 ? "no difference"
                        : (eff.median > 0.0) ? "*** NT IS WORSE ***"
                                             : "NT is better";
  if (sig && eff.median > 0.0) ++g.worseCases;

  char where[64];
  std::snprintf(where, sizeof(where), "%s @ %llu MiB", name,
                static_cast<unsigned long long>(footprint / kMiB));
  if (eff.median > g.mostAdverse) {
    g.mostAdverse = eff.median;
    g.mostAdverseCase = where;
  }
  if (r.resolutionLimit() > g.widestLimit) {
    g.widestLimit = r.resolutionLimit();
    g.widestLimitCase = where;
  }

  char cell[64];
  formatDelta(cell, sizeof(cell), eff, sig);
  std::printf("  %-14s %-44s %9.3f %-28s %7.2fpp  %s\n", name, mechanism,
              r.medianMs(r.firstSlotOf(0)), cell, r.resolutionLimit(), verdict);
  std::fflush(stdout);

  char extra[96];
  std::snprintf(extra, sizeof(extra), "footprint_mib=%llu sig=%d res_pp=%.2f",
                static_cast<unsigned long long>(footprint / kMiB), static_cast<int>(sig),
                r.resolutionLimit());
  emitRow("adversarial", name, "delta_pct", eff.median, "pct", extra);
  emitRow("adversarial", name, "baseline_ms", r.medianMs(r.firstSlotOf(0)), "ms", extra);
}

void runSuite(u64 fp) {
  std::printf("\n--- footprint %llu MiB per buffer (a copy touches %llu MiB = %.2fx GL2) ---\n",
              static_cast<unsigned long long>(fp / kMiB),
              static_cast<unsigned long long>(2 * fp / kMiB),
              2.0 * static_cast<double>(fp) / static_cast<double>(kGL2Bytes));
  std::printf("  %-14s %-44s %9s %-28s %9s  %s\n", "case", "mechanism that could hurt",
              "plain_ms", "nt vs plain [95% CI]", "res_limit", "verdict");
  std::printf("  %s\n", std::string(130, '-').c_str());

  auto noSetup = [](int) {};
  u8** p = g.pool;

  // The yardstick: copy timed, nothing else running. Every other case is read
  // against this.
  runCase("reference", "one copy timed, nothing else", fp, CopyPlain128, CopyNtStore128, noSetup,
          [&](int v) { enqueueCopy(v, g.s[0], p[0], p[1], fp, g.m.blitWg); });

  // The central case. The copy is untimed; only the reader of what it wrote is
  // timed. If the hint evicted the destination, this reader pays for it.
  runCase("consume-only", "reader of the destination (copy untimed)", fp, CopyPlain128,
          CopyNtStore128, [&](int v) { enqueueCopy(v, g.s[0], p[0], p[1], fp, g.m.blitWg); },
          [&](int) { enqueueConsume(g.s[0], p[1], fp); });

  // Reuse as a source rather than as a plain read.
  runCase("chain-second", "2nd copy, src just written by the 1st", fp, CopyPlain128,
          CopyNtStore128, [&](int v) { enqueueCopy(v, g.s[0], p[0], p[1], fp, g.m.blitWg); },
          [&](int v) { enqueueCopy(v, g.s[0], p[1], p[2], fp, g.m.blitWg); });

  // Control for chain-second: identical timed work reading a source nothing just
  // wrote. The gap between the two is the value of the reuse.
  runCase("chain-coldsrc", "same, src NOT just written (control)", fp, CopyPlain128,
          CopyNtStore128, [&](int v) { enqueueCopy(v, g.s[0], p[0], p[1], fp, g.m.blitWg); },
          [&](int v) { enqueueCopy(v, g.s[0], p[3], p[2], fp, g.m.blitWg); });

  // Fan-out: one source read repeatedly. An NT *load* should hurt here, and
  // nt-store-128 does not hint loads, so this case's job is to confirm that the
  // shipped variant leaves the source alone. Sized so source plus destinations
  // together still fit GL2 at the smallest footprint.
  runCase("fanout", "one source copied to 3 destinations", fp, CopyPlain128, CopyNtStore128,
          noSetup, [&](int v) {
            for (int k = 1; k <= 3; ++k)
              enqueueCopy(v, g.s[0], p[0], p[k], fp / 3, g.m.blitWg);
          });

  // Write-combining: if the hint defeats it, rewriting the same lines repeatedly
  // is where that shows up.
  runCase("overwrite", "same destination rewritten 6x", fp, CopyPlain128, CopyNtStore128, noSetup,
          [&](int v) {
            for (int k = 0; k < 6; ++k) enqueueCopy(v, g.s[0], p[0], p[1], fp, g.m.blitWg);
          });

  // The CLR host-copy staging path reuses one pinned buffer for every chunk, so
  // its destination is genuinely hot. Timed on the reader for the same reason as
  // consume-only.
  runCase("staging-reuse", "one staging buffer, 4 chunks, reader timed", fp, CopyPlain128,
          CopyNtStore128,
          [&](int v) {
            for (int k = 0; k < 4; ++k) enqueueCopy(v, g.s[0], p[0], p[1], fp / 4, g.m.blitWg);
          },
          [&](int) { enqueueConsume(g.s[0], p[1], fp / 4); });

  runCase("multistream", "4 concurrent copies on 4 streams", fp, CopyPlain128, CopyNtStore128,
          noSetup, [&](int v) {
            for (int k = 0; k < kStreams; ++k)
              enqueueCopy(v, g.s[k], p[0], p[1 + k], fp / 2, g.m.blitWg);
          });

  // The narrow fallback path, where the hint applies to 4-byte stores. Compared
  // against plain-32 so the comparison is hint-only, not hint-plus-width.
  runCase("unaligned", "32-bit fallback path, 4B NT stores", fp, CopyPlain32, CopyPr2616Actual32,
          noSetup,
          [&](int v) { enqueueCopy(v, g.s[0], p[0] + 4, p[1] + 4, fp - 8, g.m.blitWg); });
}

}  // namespace

int main(int argc, char** argv) {
  g.iters = static_cast<int>(argValue(argc, argv, "--iters", 15));
  g.warmup = static_cast<int>(argValue(argc, argv, "--warmup", 4));

  g.m = initMachine("adversarial search for cases where the hint loses");
  std::printf("  iters/warmup  : %d / %d, 4 slots per case (both arms twice)\n", g.iters,
              g.warmup);

  const u64 poolBuffer = 192 * kMiB;
  for (int i = 0; i < kPoolBuffers; ++i) {
    HIP_CHECK(hipMalloc(&g.pool[i], poolBuffer));
    HIP_CHECK(hipMemset(g.pool[i], 0xA5 + i, poolBuffer));
  }
  Scratch scratch;
  g.scratch = &scratch;
  for (int k = 0; k < kStreams; ++k) HIP_CHECK(hipStreamCreate(&g.s[k]));
  EventTimer timer;
  g.timer = &timer;

  // Half of GL2, all of GL2, and well past it. The first two are the footprints
  // at which the destination could plausibly have stayed resident, which is the
  // precondition for the hint to be able to cost anything.
  const u64 footprints[] = {kGL2Bytes / 4, kGL2Bytes / 2, 128 * kMiB};
  for (u64 fp : footprints) runSuite(fp);

  std::printf("\n  A verdict needs both: an interval excluding zero AND a magnitude above the\n");
  std::printf("  resolution limit, which is this rig measuring the same arm twice in the\n");
  std::printf("  same run. Positive means the non-temporal variant took longer.\n");

  std::printf("\n  cases where the hint is significantly worse : %d\n", g.worseCases);
  std::printf("  most adverse effect seen anywhere           : %+.2f%%  (%s)\n", g.mostAdverse,
              g.mostAdverseCase.c_str());
  std::printf("  coarsest resolution in the suite            : %.2f pp (%s)\n", g.widestLimit,
              g.widestLimitCase.c_str());
  std::printf("\n  So: no regression was found, and a regression smaller than %.2f pp could not\n",
              g.widestLimit);
  std::printf("  have been found in the least sensitive case. That is the claim's limit.\n");
  emitRow("adversarial", "-", "worse_cases", static_cast<double>(g.worseCases), "count", "");
  emitRow("adversarial", "-", "most_adverse_pct", g.mostAdverse, "pct",
          g.mostAdverseCase.c_str());
  emitRow("adversarial", "-", "coarsest_resolution_pp", g.widestLimit, "pp",
          g.widestLimitCase.c_str());

  BENCH_ASSERT(g.worseCases == 0,
               "%d adversarial case(s) now show the hint losing; the report's central "
               "'no case found' claim no longer holds",
               g.worseCases);

  for (int k = 0; k < kStreams; ++k) HIP_CHECK(hipStreamDestroy(g.s[k]));
  for (int i = 0; i < kPoolBuffers; ++i) HIP_CHECK(hipFree(g.pool[i]));
  flushRows();
  return exitCode();
}
