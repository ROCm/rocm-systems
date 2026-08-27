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

TEST(ConSanDescriptor, VgprGrowthHonorsTargetBankingAndCallerAddressLimit) {
  kd::kernel_descriptor_t first_gfx1250_granule{};
  EXPECT_TRUE(grow_descriptor_vgpr_allocation(
      first_gfx1250_granule,
      {.required_ordinary_count = 17u, .maximum_ordinary_count = REGISTER_SET_MAX_VGPRS},
      ROCJITSU_CODE_ARCH_CDNA5));
  EXPECT_EQ(AMDHSA_BITS_GET(first_gfx1250_granule.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
            1u);

  kd::kernel_descriptor_t banked_gfx1250{};
  AMDHSA_BITS_SET(banked_gfx1250.compute_pgm_rsrc1,
                  kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 15u);
  EXPECT_TRUE(grow_descriptor_vgpr_allocation(
      banked_gfx1250,
      {.required_ordinary_count = 257u, .maximum_ordinary_count = REGISTER_SET_MAX_VGPRS},
      ROCJITSU_CODE_ARCH_CDNA5));
  EXPECT_EQ(AMDHSA_BITS_GET(banked_gfx1250.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
            16u);

  kd::kernel_descriptor_t caller_limited = banked_gfx1250;
  const uint32_t original_rsrc1 = caller_limited.compute_pgm_rsrc1;
  EXPECT_FALSE(grow_descriptor_vgpr_allocation(
      caller_limited, {.required_ordinary_count = 273u, .maximum_ordinary_count = 256u},
      ROCJITSU_CODE_ARCH_CDNA5));
  EXPECT_EQ(caller_limited.compute_pgm_rsrc1, original_rsrc1);

  kd::kernel_descriptor_t rdna4{};
  AMDHSA_BITS_SET(rdna4.kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                  1u);
  EXPECT_FALSE(grow_descriptor_vgpr_allocation(
      rdna4, {.required_ordinary_count = 257u, .maximum_ordinary_count = REGISTER_SET_MAX_VGPRS},
      ROCJITSU_CODE_ARCH_RDNA4));

  kd::kernel_descriptor_t invalid{};
  EXPECT_FALSE(grow_descriptor_vgpr_allocation(
      invalid, {.required_ordinary_count = 0u, .maximum_ordinary_count = 256u},
      ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(grow_descriptor_vgpr_allocation(
      invalid, {.required_ordinary_count = 1u, .maximum_ordinary_count = 256u},
      ROCJITSU_CODE_ARCH_INVALID));
  EXPECT_EQ(invalid.compute_pgm_rsrc1, 0u);
  EXPECT_EQ(invalid.compute_pgm_rsrc3, 0u);
}

TEST(ConSanDescriptor, VgprGrowthMovesOnlyProvenEmptyCdnaAccumulatorStorage) {
  kd::kernel_descriptor_t live_accumulator_bank{};
  AMDHSA_BITS_SET(live_accumulator_bank.compute_pgm_rsrc1,
                  kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 3u);
  AMDHSA_BITS_SET(live_accumulator_bank.compute_pgm_rsrc3,
                  kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
  const uint32_t original_rsrc1 = live_accumulator_bank.compute_pgm_rsrc1;
  const uint32_t original_rsrc3 = live_accumulator_bank.compute_pgm_rsrc3;

  EXPECT_FALSE(grow_descriptor_vgpr_allocation(
      live_accumulator_bank, {.required_ordinary_count = 20u, .maximum_ordinary_count = 256u},
      ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(live_accumulator_bank.compute_pgm_rsrc1, original_rsrc1);
  EXPECT_EQ(live_accumulator_bank.compute_pgm_rsrc3, original_rsrc3);

  EXPECT_TRUE(grow_descriptor_vgpr_allocation(live_accumulator_bank,
                                              {.required_ordinary_count = 20u,
                                               .maximum_ordinary_count = 256u,
                                               .accumulator_bank_is_proven_empty = true},
                                              ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(AMDHSA_BITS_GET(live_accumulator_bank.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
            3u);
  EXPECT_EQ(AMDHSA_BITS_GET(live_accumulator_bank.compute_pgm_rsrc3,
                            kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET),
            7u);
}

TEST(ConSanDescriptor, VgprGrowthUsesIntrinsicEmptyCdnaAccumulatorGap) {
  kd::kernel_descriptor_t within_gap{};
  AMDHSA_BITS_SET(within_gap.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
  EXPECT_TRUE(grow_descriptor_vgpr_allocation(
      within_gap, {.required_ordinary_count = 12u, .maximum_ordinary_count = 256u},
      ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(AMDHSA_BITS_GET(within_gap.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
            1u);
  EXPECT_EQ(
      AMDHSA_BITS_GET(within_gap.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET), 3u);

  kd::kernel_descriptor_t past_gap{};
  AMDHSA_BITS_SET(past_gap.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
  EXPECT_TRUE(grow_descriptor_vgpr_allocation(
      past_gap, {.required_ordinary_count = 20u, .maximum_ordinary_count = 256u},
      ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(AMDHSA_BITS_GET(past_gap.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
            2u);
  EXPECT_EQ(AMDHSA_BITS_GET(past_gap.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET),
            5u);
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
