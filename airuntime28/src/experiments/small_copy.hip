// Small copies: is there anything to win below the memory-bound regime?
//
// Two things had to be separated here. At small sizes the dispatch costs more
// than the copy, so a difference between two variants can come from having two
// different kernel objects rather than from what they store - different code
// objects, different scratch, different instruction fetch. The runtime-mode arms
// below use ONE kernel object with the store selected by a kernel argument, which
// removes that difference entirely; the template arms are what production would
// actually ship. Reporting both says how much of any effect is the hint and how
// much is the second kernel.
//
// The sweep now runs to 64 MiB. The original stopped at 8 MiB and described that
// as "small enough to have stayed cached", which was reasoning from the driver's
// 4 MiB figure; against the measured ~96 MiB GL2, 8 MiB is nowhere near the
// interesting boundary.

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

// One kernel object, both store policies. Same grid-stride shape and end_ptr
// termination as the production kernel; the aligned/tail split is dropped because
// every size here is a multiple of 16.
__global__ void copyRuntimeMode(const u8* src, u8* dst, u64 end_ptr, u32 next_chunk,
                                u32 workgroup_size, int useNt) {
  u64 id = (static_cast<u64>(blockIdx.x) * workgroup_size + threadIdx.x);
  const u64x2* srcD = reinterpret_cast<const u64x2*>(src);
  u64x2* dstD = reinterpret_cast<u64x2*>(dst);
  if (useNt) {
    while (reinterpret_cast<u64>(&dstD[id]) < end_ptr) {
      __builtin_nontemporal_store(srcD[id], &dstD[id]);
      id += next_chunk;
    }
  } else {
    while (reinterpret_cast<u64>(&dstD[id]) < end_ptr) {
      dstD[id] = srcD[id];
      id += next_chunk;
    }
  }
}

int main(int argc, char** argv) {
  const int iters = static_cast<int>(argValue(argc, argv, "--iters", 30));
  const int warmup = static_cast<int>(argValue(argc, argv, "--warmup", 8));

  Machine m = initMachine("small copies");
  std::printf("  iters/warmup  : %d / %d\n", iters, warmup);

  const std::vector<std::string> arms = {"runtime-plain", "runtime-nt", "template-plain",
                                         "template-nt"};
  const SlotPlan plan = SlotPlan::duplicated(arms);

  const u64 sizes[] = {16 * kKiB, 64 * kKiB,  256 * kKiB, 1 * kMiB,  2 * kMiB,
                       4 * kMiB,  8 * kMiB,   16 * kMiB,  32 * kMiB, 64 * kMiB};
  const u64 maxSize = 64 * kMiB;

  u8* src = nullptr;
  u8* dst = nullptr;
  HIP_CHECK(hipMalloc(&src, maxSize));
  HIP_CHECK(hipMalloc(&dst, maxSize));
  HIP_CHECK(hipMemset(src, 0xA5, maxSize));
  HIP_CHECK(hipMemset(dst, 0x00, maxSize));

  Scratch scratch;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  EventTimer timer;

  // An empty dispatch on the same stream, for scale: any size whose copy time is
  // close to this is measuring the launch, not the copy.
  double emptyMs = 0.0;
  {
    std::vector<double> v;
    for (int i = 0; i < warmup + iters; ++i) {
      const double ms = timer.timeMs(stream, [&] {
        hipLaunchKernelGGL(copyRuntimeMode, dim3(1), dim3(64), 0, stream, src, dst,
                           reinterpret_cast<u64>(dst), 1u, 64u, 0);
      });
      if (i >= warmup) v.push_back(ms);
    }
    emptyMs = median(v);
  }
  std::printf("\n  a dispatch that copies nothing takes %.4f ms on this stream\n\n", emptyMs);

  std::printf("  %10s %10s %8s %28s %28s %8s\n", "size", "plain_ms", "vs_empty",
              "runtime-nt (hint only)", "template-nt (as shipped)", "res_lim");
  std::printf("  %s\n", std::string(100, '-').c_str());

  for (u64 size : sizes) {
    const SlotResults r = runSlots(plan, iters, warmup, [&](int slot) {
      const int arm = plan.arm[slot];
      return timer.timeMs(
          stream, [&] { scratch.enqueueFlush(stream); },
          [&] {
            if (arm < 2) {
              const Geometry gm = makeGeometry(kVariants[CopyPlain128], dst, size, m.blitWg);
              hipLaunchKernelGGL(copyRuntimeMode, gm.grid, gm.block, 0, stream, src, dst, gm.endPtr,
                                 gm.nextChunk, gm.workgroupSize, arm == 1 ? 1 : 0);
            } else {
              enqueueCopy(arm == 3 ? CopyNtStore128 : CopyPlain128, stream, src, dst, size,
                          m.blitWg);
            }
          });
    });

    const double baseMs = r.medianMs(r.firstSlotOf(0));
    const Delta runtimeNt = r.effect(1, 0);
    const Delta templateNt = r.effect(3, 2);
    char c1[64], c2[64];
    formatDelta(c1, sizeof(c1), runtimeNt, r.isSignificant(runtimeNt));
    formatDelta(c2, sizeof(c2), templateNt, r.isSignificant(templateNt));

    char label[24];
    if (size >= kMiB)
      std::snprintf(label, sizeof(label), "%lluMiB", static_cast<unsigned long long>(size / kMiB));
    else
      std::snprintf(label, sizeof(label), "%lluKiB", static_cast<unsigned long long>(size / kKiB));

    std::printf("  %10s %10.4f %7.1fx %28s %28s %6.2fpp\n", label, baseMs, baseMs / emptyMs, c1, c2,
                r.resolutionLimit());
    std::fflush(stdout);

    char extra[80];
    std::snprintf(extra, sizeof(extra), "size_kib=%llu vs_empty=%.1f",
                  static_cast<unsigned long long>(size / kKiB), baseMs / emptyMs);
    emitRow("small_copy", "runtime-nt", "delta_pct", runtimeNt.median, "pct", extra);
    emitRow("small_copy", "template-nt", "delta_pct", templateNt.median, "pct", extra);
    emitRow("small_copy", "runtime-plain", "copy_ms", baseMs, "ms", extra);
  }

  std::printf("\n  'vs_empty' is how many empty dispatches the copy costs. Below about 3x, the\n");
  std::printf("  measurement is dominated by launch overhead and no store policy can move it.\n");

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(src));
  HIP_CHECK(hipFree(dst));
  flushRows();
  return exitCode();
}
