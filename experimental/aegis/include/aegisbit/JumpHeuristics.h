#ifndef AEGISBIT_JUMP_HEURISTICS_H
#define AEGISBIT_JUMP_HEURISTICS_H

#include "aegisbit/InstrumentationPlan.h"
#include "aegisbit/IslandAllocator.h"

namespace aegisbit {

bool shouldForceAllRelay(const InstrumentationPlan &Plan, size_t SiteCount,
                         uint64_t BaseAddr, uint64_t TextSectionSize);

bool shouldUseSwapPCSharedBody(const InstrumentationPlan &Plan,
                               uint64_t TextSectionSize,
                               uint64_t BaseAddr,
                               size_t SiteCount);

uint64_t computePreKernelPadSize(const InstrumentationPlan &Plan,
                                 uint64_t KernelStart, uint64_t KernelSize,
                                 uint64_t TextSectionSize = 0);

bool resolveAdaptiveOverflow(const InstrumentationPlan &Plan,
                             IslandAllocator &Alloc,
                             uint64_t PatchSiteAbs,
                             uint32_t SiteIdx,
                             uint32_t &LastRetrySiteIdx,
                             uint32_t &RetryCount,
                             uint32_t MaxRetries,
                             bool &UseRelay,
                             int64_t &BranchToDword);

} // namespace aegisbit

#endif // AEGISBIT_JUMP_HEURISTICS_H
