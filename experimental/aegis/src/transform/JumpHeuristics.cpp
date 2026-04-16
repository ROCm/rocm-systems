#include "aegisbit/JumpHeuristics.h"
#include "aegisbit/RuntimeConfig.h"

namespace aegisbit {

bool shouldForceAllRelay(const InstrumentationPlan &Plan, size_t SiteCount,
                         uint64_t BaseAddr, uint64_t TextSectionSize) {
  const bool ForceRelay = RuntimeConfig::getInstance().Transform.ForceRelay;
  if (ForceRelay && Plan.Register == RegisterMode::ZeroSGPR)
    return true;
  return Plan.Register == RegisterMode::ZeroSGPR &&
         (SiteCount * 700 + (TextSectionSize - BaseAddr) > 131072);
}

bool shouldUseSwapPCSharedBody(const InstrumentationPlan &Plan,
                               uint64_t TextSectionSize,
                               uint64_t BaseAddr,
                               size_t SiteCount) {
  if (Plan.Register != RegisterMode::StandardScratch)
    return false;
  if (Plan.Payload != PayloadStrategy::OnGpuReduce)
    return false;
  if (Plan.Jump != JumpStrategy::SharedBody)
    return false;
  if (RuntimeConfig::getInstance().Transform.ForceSwapPC)
    return true;
  // The island is placed at alignIslandStart(TextSectionSize, Occupied), so
  // the worst-case s_call_b64 target (from a site at BaseAddr) is at least
  // TextSectionSize - BaseAddr bytes away, plus the island body itself. Using
  // the full post-kernel span -- not just this kernel's size -- is what lets
  // us correctly upgrade small kernels sitting near the start of a dense
  // multi-kernel .text.
  uint64_t EstIslandSize = SiteCount * 12 + 2048;
  uint64_t SpanToIslandEnd =
      (TextSectionSize > BaseAddr ? TextSectionSize - BaseAddr : 0) +
      EstIslandSize;
  return SpanToIslandEnd > 120 * 1024;
}

uint64_t computePreKernelPadSize(const InstrumentationPlan &Plan,
                                 uint64_t KernelStart, uint64_t KernelSize,
                                 uint64_t TextSectionSize) {
  constexpr uint64_t MinPreKernelSpace = 256 * 1024;
  if (Plan.Register != RegisterMode::ZeroSGPR)
    return 0;
  if (KernelStart >= MinPreKernelSpace)
    return 0;
  bool LargeKernel = KernelSize > 100 * 1024;
  bool DenseCodeObject = TextSectionSize > KernelStart + KernelSize + 4096;
  if (!LargeKernel && !DenseCodeObject)
    return 0;
  return ((MinPreKernelSpace - KernelStart) + 255) & ~255ULL;
}

bool resolveAdaptiveOverflow(const InstrumentationPlan &Plan,
                             IslandAllocator &Alloc,
                             uint64_t PatchSiteAbs,
                             uint32_t SiteIdx,
                             uint32_t &LastRetrySiteIdx,
                             uint32_t &RetryCount,
                             uint32_t MaxRetries,
                             bool &UseRelay,
                             int64_t &BranchToDword) {
  if (Plan.Register != RegisterMode::ZeroSGPR ||
      IslandAllocator::inBranchRange(BranchToDword))
    return false;

  bool Retry = Alloc.tryResolveOverflow(PatchSiteAbs, SiteIdx, LastRetrySiteIdx,
                                        RetryCount, MaxRetries);
  if (Retry)
    return true;

  UseRelay = true;
  BranchToDword = Alloc.branchDword(PatchSiteAbs);
  return false;
}

} // namespace aegisbit
