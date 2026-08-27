// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"

namespace rocjitsu {
namespace {

namespace kd = rocr::llvm::amdhsa;

TEST(ConSanDescriptor, ResourceFactsShareTargetWaveAndAccumulatorSemantics) {
  kd::kernel_descriptor_t descriptor{};
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                  kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 3u);

  // gfx1250 is Wave32-only even when a producer omits the legacy Wave32 bit.
  EXPECT_EQ(descriptor_vgpr_granularity(descriptor, ROCJITSU_CODE_ARCH_CDNA5), 16u);
  EXPECT_EQ(descriptor_vgpr_allocation_count(descriptor, ROCJITSU_CODE_ARCH_CDNA5), 64u);

  // The same encoded count has distinct RDNA Wave64 and Wave32 meanings.
  EXPECT_EQ(descriptor_vgpr_allocation_count(descriptor, ROCJITSU_CODE_ARCH_RDNA4), 16u);
  AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                  kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 1u);
  EXPECT_EQ(descriptor_vgpr_allocation_count(descriptor, ROCJITSU_CODE_ARCH_RDNA4), 32u);

  // Only descriptor-partitioned CDNA targets interpret ACCUM_OFFSET as the
  // ordinary/accumulator boundary.
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
  EXPECT_EQ(descriptor_vgpr_allocation_count(descriptor, ROCJITSU_CODE_ARCH_CDNA4), 32u);
  EXPECT_EQ(descriptor_ordinary_vgpr_allocation_count(descriptor, ROCJITSU_CODE_ARCH_CDNA4), 16u);
  EXPECT_EQ(descriptor_ordinary_vgpr_allocation_count(descriptor, ROCJITSU_CODE_ARCH_CDNA5), 64u);

  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                  kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 15u);
  EXPECT_EQ(descriptor_sgpr_allocation_count(descriptor, ROCJITSU_CODE_ARCH_CDNA4),
            REGISTER_SET_MAX_SGPRS);
}

TEST(ConSanDescriptor, SgprGrowthPreservesTargetSpecialRegisterPlacement) {
  kd::kernel_descriptor_t cdna4{};
  EXPECT_TRUE(grow_descriptor_sgpr_allocation(cdna4, 20u, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(AMDHSA_BITS_GET(cdna4.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT),
            3u);
  EXPECT_EQ(descriptor_sgpr_allocation_count(cdna4, ROCJITSU_CODE_ARCH_CDNA4), 32u);

  kd::kernel_descriptor_t rdna4{};
  EXPECT_TRUE(grow_descriptor_sgpr_allocation(rdna4, 20u, ROCJITSU_CODE_ARCH_RDNA4));
  // RDNA exposes its complete fixed scalar file; this legacy allocation field
  // is neither needed nor changed.
  EXPECT_EQ(AMDHSA_BITS_GET(rdna4.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT),
            0u);
  EXPECT_EQ(descriptor_sgpr_allocation_count(rdna4, ROCJITSU_CODE_ARCH_RDNA4), 106u);
}

TEST(ConSanDescriptor, SgprGrowthRejectsUnrepresentableOrdinaryRegistersWithoutMutation) {
  kd::kernel_descriptor_t descriptor{};
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                  kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 1u);
  const uint32_t original_rsrc1 = descriptor.compute_pgm_rsrc1;

  EXPECT_FALSE(grow_descriptor_sgpr_allocation(descriptor, 0u, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(grow_descriptor_sgpr_allocation(descriptor, 103u, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(grow_descriptor_sgpr_allocation(descriptor, 16u, ROCJITSU_CODE_ARCH_INVALID));
  EXPECT_EQ(descriptor.compute_pgm_rsrc1, original_rsrc1);

  EXPECT_TRUE(grow_descriptor_sgpr_allocation(descriptor, 102u, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT),
            13u);
}

} // namespace
} // namespace rocjitsu
