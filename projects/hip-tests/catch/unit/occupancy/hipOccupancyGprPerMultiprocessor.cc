/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
Unit_hipOccupancy_Positive_MonotoneInVgprCount - Validate that the occupancy reported by
hipOccupancyMaxActiveBlocksPerMultiprocessor never grows when a kernel uses more VGPRs.
Unit_hipOccupancy_Positive_EqualVgprCountEqualOccupancy - Validate that two kernels with the same
VGPR count, the same block size and no LDS are given the same occupancy.

The granule is not exposed through hipDeviceProp_t, so these check the ordering between kernels
rather than an expected block count; that ordering holds on every architecture.
*/

#include "occupancy_common.hh"

#include <algorithm>
#include <vector>

namespace {

constexpr int kBlockSize = 128;

// Clobbering vN pins the kernel at N + 1 VGPRs at any optimization level; computed register
// pressure is not usable here, it collapses to one count once locals spill to scratch at -O0.
#define OCCUPANCY_VGPR_KERNEL_NAMED(NAME, N)                                                       \
  __global__ void NAME(float* out) {                                                               \
    asm volatile("" ::: "v" #N);                                                                   \
    *out = 1.0f;                                                                                   \
  }

#define OCCUPANCY_VGPR_KERNEL(N) OCCUPANCY_VGPR_KERNEL_NAMED(vgprKernel##N, N)

// Straddles the 24 register granule of the 1536 VGPR gfx11/gfx12 parts.
#define OCCUPANCY_VGPR_KERNEL_LIST(X)                                                              \
  X(7) X(15) X(23) X(39) X(63) X(81) X(97) X(113) X(119) X(121) X(129) X(137) X(161) X(191)

OCCUPANCY_VGPR_KERNEL_LIST(OCCUPANCY_VGPR_KERNEL)

// A second kernel at the same clobber, and so the same register count, as vgprKernel97.
OCCUPANCY_VGPR_KERNEL_NAMED(vgprKernelTwin97, 97)

struct OccupancySample {
  int numRegs;
  int numBlocks;
};

OccupancySample Sample(const void* func) {
  hipFuncAttributes attributes;
  HIP_CHECK(hipFuncGetAttributes(&attributes, func));

  int numBlocks = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, func, kBlockSize, 0));

  INFO("numRegs " << attributes.numRegs << ", numBlocks " << numBlocks);
  REQUIRE(attributes.numRegs > 0);
  REQUIRE(attributes.sharedSizeBytes == 0);  // keeps LDS out of the comparison
  REQUIRE(numBlocks > 0);

  return {attributes.numRegs, numBlocks};
}

std::vector<OccupancySample> CollectSamples() {
#define OCCUPANCY_VGPR_SAMPLE(N) Sample(reinterpret_cast<const void*>(vgprKernel##N)),
  std::vector<OccupancySample> samples{OCCUPANCY_VGPR_KERNEL_LIST(OCCUPANCY_VGPR_SAMPLE)};
#undef OCCUPANCY_VGPR_SAMPLE

  std::sort(samples.begin(), samples.end(),
            [](const OccupancySample& a, const OccupancySample& b) {
              return a.numRegs < b.numRegs;
            });
  return samples;
}

}  // namespace

HIP_TEST_CASE(Unit_hipOccupancy_Positive_MonotoneInVgprCount) {
  const std::vector<OccupancySample> samples = CollectSamples();

  for (size_t i = 1; i < samples.size(); ++i) {
    const OccupancySample& lower = samples[i - 1];
    const OccupancySample& higher = samples[i];

    INFO("numRegs " << lower.numRegs << " -> " << higher.numRegs << " reported numBlocks "
                    << lower.numBlocks << " -> " << higher.numBlocks);
    REQUIRE(higher.numBlocks <= lower.numBlocks);
  }
}

HIP_TEST_CASE(Unit_hipOccupancy_Positive_EqualVgprCountEqualOccupancy) {
  // Two distinct kernels at the same register count: equal VGPRs, block size and LDS leaves
  // nothing for the occupancy to differ on.
  const OccupancySample first = Sample(reinterpret_cast<const void*>(vgprKernel97));
  const OccupancySample second = Sample(reinterpret_cast<const void*>(vgprKernelTwin97));

  INFO("numRegs " << first.numRegs << " and " << second.numRegs << ", numBlocks "
                  << first.numBlocks << " and " << second.numBlocks);
  REQUIRE(first.numRegs == second.numRegs);
  REQUIRE(first.numBlocks == second.numBlocks);
}
