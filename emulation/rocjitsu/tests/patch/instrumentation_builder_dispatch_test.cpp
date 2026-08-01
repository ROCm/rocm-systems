// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <gtest/gtest.h>

namespace rocjitsu {
namespace {

namespace ib = instrumentation;

TEST(InstrumentationBuilderDispatch, ScalarControlSelectsTargetBackend) {
  const auto legacy_hw_id = build_hwreg_imm(/*reg_id=*/4, /*offset=*/0, /*size_bits=*/6);
  const auto gfx12_hw_id = build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/10);
  ASSERT_TRUE(legacy_hw_id);
  ASSERT_TRUE(gfx12_hw_id);
  EXPECT_EQ(ib::build_s_getreg_b32(20, *legacy_hw_id, ROCJITSU_CODE_ARCH_CDNA3), 0xb8942804u);
  EXPECT_EQ(ib::build_s_getreg_b32(20, *legacy_hw_id, ROCJITSU_CODE_ARCH_CDNA4), 0xb8942804u);
  EXPECT_EQ(ib::build_s_getreg_b32(20, *gfx12_hw_id, ROCJITSU_CODE_ARCH_RDNA4), 0xb8944817u);
  EXPECT_EQ(ib::build_s_getreg_b32(20, *gfx12_hw_id, ROCJITSU_CODE_ARCH_GFX1250), 0xb8944817u);
  EXPECT_EQ(ib::build_s_mov_b64(20, 22, ROCJITSU_CODE_ARCH_CDNA3), 0xbe940116u);
  EXPECT_EQ(ib::build_s_mov_b64(20, 22, ROCJITSU_CODE_ARCH_CDNA4), 0xbe940116u);
  EXPECT_EQ(ib::build_s_mov_b64(20, 22, ROCJITSU_CODE_ARCH_RDNA4), 0xbe940116u);
  EXPECT_EQ(ib::build_s_and_saveexec_b64(20, 22, ROCJITSU_CODE_ARCH_CDNA4), 0xbe942016u);
  EXPECT_EQ(ib::build_s_and_saveexec_b64(20, 22, ROCJITSU_CODE_ARCH_RDNA4), 0xbe942116u);
  EXPECT_EQ(ib::build_s_andn2_b64(20, 22, 24, ROCJITSU_CODE_ARCH_CDNA4), 0x89941816u);
  EXPECT_EQ(ib::build_s_andn2_b64(20, 22, 24, ROCJITSU_CODE_ARCH_RDNA4), 0x91941816u);
  EXPECT_EQ(ib::build_s_and_b64(20, 22, 24, ROCJITSU_CODE_ARCH_CDNA4), 0x86941816u);
  EXPECT_EQ(ib::build_s_and_b64(20, 22, 24, ROCJITSU_CODE_ARCH_RDNA4), 0x8b941816u);
  EXPECT_EQ(ib::build_s_bcnt1_i32_b64(20, 22, ROCJITSU_CODE_ARCH_CDNA4), 0xbe940d16u);
  EXPECT_EQ(ib::build_s_bcnt1_i32_b64(20, 22, ROCJITSU_CODE_ARCH_RDNA4), 0xbe941916u);
  EXPECT_EQ(ib::build_s_bcnt1_i32_b64(20, 22, ROCJITSU_CODE_ARCH_GFX1250), 0xbe941916u);
  EXPECT_EQ(ib::build_s_cbranch_execz(1, ROCJITSU_CODE_ARCH_CDNA4), 0xbf880001u);
  EXPECT_EQ(ib::build_s_cbranch_execz(1, ROCJITSU_CODE_ARCH_RDNA4), 0xbfa50001u);
}

TEST(InstrumentationBuilderDispatch, VectorAndWaitSemanticsSelectTargetBackend) {
  EXPECT_EQ(ib::build_v_lshrrev_b32(10, scalar_positive_inline_u32(3), 3, ROCJITSU_CODE_ARCH_CDNA3),
            0x20140683u);
  EXPECT_EQ(ib::build_v_lshrrev_b32(10, scalar_positive_inline_u32(3), 3, ROCJITSU_CODE_ARCH_CDNA4),
            0x20140683u);
  EXPECT_EQ(ib::build_v_min_u32(10, vector_source_vgpr(2), 3, ROCJITSU_CODE_ARCH_CDNA3),
            0x1c140702u);
  EXPECT_EQ(ib::build_v_min_u32(10, vector_source_vgpr(2), 3, ROCJITSU_CODE_ARCH_CDNA4),
            0x1c140702u);
  EXPECT_EQ(ib::build_v_min_u32(10, vector_source_vgpr(2), 3, ROCJITSU_CODE_ARCH_RDNA4),
            0x26140702u);
  EXPECT_EQ(ib::build_v_min_u32(10, vector_source_vgpr(2), 3, ROCJITSU_CODE_ARCH_GFX1250),
            0x26140702u);
  EXPECT_EQ(ib::build_v_cmp_eq_u32_vcc(vector_source_vgpr(2), 3, ROCJITSU_CODE_ARCH_CDNA4),
            0x7d940702u);
  EXPECT_EQ(ib::build_v_writelane_b32(10, 20, 3, ROCJITSU_CODE_ARCH_CDNA3),
            (std::array<uint32_t, 2>{0xd28a000au, 0x00010614u}));
  EXPECT_EQ(ib::build_v_readlane_b32(20, 10, 3, ROCJITSU_CODE_ARCH_CDNA4),
            (std::array<uint32_t, 2>{0xd2890014u, 0x0001070au}));
  EXPECT_EQ(ib::build_s_wait_flat_load0(ROCJITSU_CODE_ARCH_CDNA4), 0xbf8c0070u);
  EXPECT_EQ(ib::build_s_wait_flat_store0(ROCJITSU_CODE_ARCH_CDNA4), 0xbf8c0070u);
  EXPECT_EQ(ib::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_CDNA4), 0xbf8c0f70u);
  EXPECT_EQ(ib::build_s_wait_global_store0(ROCJITSU_CODE_ARCH_CDNA4), 0xbf8c0f70u);
  EXPECT_EQ(ib::build_s_wait_lds0(ROCJITSU_CODE_ARCH_CDNA4), 0xbf8cc07fu);
  EXPECT_EQ(ib::build_s_wait_scalar_load0(ROCJITSU_CODE_ARCH_CDNA4), 0xbf8cc07fu);
  EXPECT_EQ(ib::build_s_wait_flat_load0(ROCJITSU_CODE_ARCH_CDNA3), 0xbf8c0070u);
  EXPECT_EQ(ib::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_CDNA3), 0xbf8c0f70u);
  EXPECT_EQ(ib::build_s_wait_lds0(ROCJITSU_CODE_ARCH_CDNA3), 0xbf8cc07fu);
  EXPECT_EQ(ib::build_s_wait_flat_load0(ROCJITSU_CODE_ARCH_RDNA3), 0xbf890007u);
  EXPECT_EQ(ib::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_RDNA3), 0xbf8903f7u);
  EXPECT_EQ(ib::build_s_wait_global_store0(ROCJITSU_CODE_ARCH_RDNA3), 0xbf8903f7u);
  EXPECT_EQ(ib::build_s_wait_lds0(ROCJITSU_CODE_ARCH_RDNA3), 0xbf89fc07u);
  EXPECT_EQ(ib::build_s_wait_scalar_load0(ROCJITSU_CODE_ARCH_RDNA3), 0xbf89fc07u);
  EXPECT_EQ(ib::build_s_wait_flat_load0(ROCJITSU_CODE_ARCH_RDNA4), 0xbfc80000u);
  EXPECT_EQ(ib::build_s_wait_flat_store0(ROCJITSU_CODE_ARCH_RDNA4), 0xbfc90000u);
  EXPECT_EQ(ib::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_RDNA4), 0xbfc00000u);
  EXPECT_EQ(ib::build_s_wait_global_store0(ROCJITSU_CODE_ARCH_RDNA4), 0xbfc10000u);
  EXPECT_EQ(ib::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_GFX1250), 0xbfc00000u);
  EXPECT_EQ(ib::build_s_wait_global_store0(ROCJITSU_CODE_ARCH_GFX1250), 0xbfc10000u);
  EXPECT_EQ(ib::build_s_wait_lds0(ROCJITSU_CODE_ARCH_RDNA4), 0xbfc60000u);
  EXPECT_EQ(ib::build_s_wait_scalar_load0(ROCJITSU_CODE_ARCH_RDNA4), 0xbfc70000u);
  EXPECT_EQ(ib::build_salu_dependency_delay(ROCJITSU_CODE_ARCH_CDNA4), 0xbf800000u);
  EXPECT_EQ(ib::build_s_wait_indirect_pc0(ROCJITSU_CODE_ARCH_CDNA4), 0xbf800000u);
  EXPECT_EQ(ib::build_s_trap(2, ROCJITSU_CODE_ARCH_CDNA4), 0xbf920002u);
}

TEST(InstrumentationBuilderDispatch, Gfx1100UsesRdna3InstructionForms) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA3;
  const auto add = build_rdna3_v_add_u32(6, vector_source_vgpr(7), 8, kArch);
  const auto ds_store = build_rdna3_ds_store_b32(0, 1, 0, kArch);
  const auto ds_load = build_rdna3_ds_load_b32(2, 0, 0, kArch);
  const auto flat_store = build_rdna3_flat_store_b32(0, 2, 0, kArch);
  const auto private_store = build_rdna3_address_free_scratch_store_b32(7, 4, kArch);
  const auto barrier = build_rdna3_s_barrier(kArch);
  ASSERT_TRUE(add && ds_store && ds_load && flat_store && private_store && barrier);
  EXPECT_TRUE(ib::is_admitted_arch(kArch));
  EXPECT_TRUE(ib::is_rdna_family_arch(kArch));
  EXPECT_FALSE(is_rdna4_family_arch(kArch));
  EXPECT_EQ(ib::build_s_trap(0, kArch), build_rdna3_s_trap(0, kArch));
  EXPECT_EQ(ib::build_s_mov_b64(20, 22, kArch), build_rdna3_s_mov_b64(20, 22, kArch));
  EXPECT_EQ(ib::build_v_add_u32(6, vector_source_vgpr(7), 8, kArch), (std::vector<uint32_t>{*add}));
  EXPECT_EQ(ib::build_ds_store_b32(0, 1, 0, kArch),
            (std::vector<uint32_t>(ds_store->begin(), ds_store->end())));
  EXPECT_EQ(ib::build_ds_load_b32(2, 0, 0, kArch),
            (std::vector<uint32_t>(ds_load->begin(), ds_load->end())));
  EXPECT_EQ(ib::build_flat_store_b32(0, 2, kArch),
            (std::vector<uint32_t>(flat_store->begin(), flat_store->end())));
  EXPECT_EQ(ib::build_private_store_b32(7, 4, kArch),
            (std::vector<uint32_t>(private_store->begin(), private_store->end())));
  EXPECT_EQ(ib::build_workgroup_barrier_only(kArch), (std::vector<uint32_t>{*barrier}));
}

TEST(InstrumentationBuilderDispatch, SaluToValuDependencyWaitSelectsTargetBackend) {
  const auto cdna3 = instrumentation::build_salu_to_valu_dependency_wait(ROCJITSU_CODE_ARCH_CDNA3);
  const auto cdna4 = instrumentation::build_salu_to_valu_dependency_wait(ROCJITSU_CODE_ARCH_CDNA4);
  const auto rdna4 = instrumentation::build_salu_to_valu_dependency_wait(ROCJITSU_CODE_ARCH_RDNA4);
  const auto gfx1250 =
      instrumentation::build_salu_to_valu_dependency_wait(ROCJITSU_CODE_ARCH_GFX1250);

  ASSERT_TRUE(cdna3);
  ASSERT_TRUE(cdna4);
  ASSERT_TRUE(rdna4);
  ASSERT_TRUE(gfx1250);
  EXPECT_EQ(*cdna3, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(*cdna4, *build_cdna4_salu_dependency_delay(ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(*rdna4, *build_s_wait_alu_sa_sdst0(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(*gfx1250, *build_s_wait_alu_sa_sdst0(ROCJITSU_CODE_ARCH_GFX1250));
}

TEST(InstrumentationBuilderDispatch, ValuToSaluDependencyWaitSelectsTargetBackend) {
  const auto cdna3 = instrumentation::build_valu_to_salu_dependency_wait(ROCJITSU_CODE_ARCH_CDNA3);
  const auto cdna4 = instrumentation::build_valu_to_salu_dependency_wait(ROCJITSU_CODE_ARCH_CDNA4);
  const auto rdna4 = instrumentation::build_valu_to_salu_dependency_wait(ROCJITSU_CODE_ARCH_RDNA4);
  const auto gfx1250 =
      instrumentation::build_valu_to_salu_dependency_wait(ROCJITSU_CODE_ARCH_GFX1250);

  ASSERT_TRUE(cdna3);
  ASSERT_TRUE(cdna4);
  ASSERT_TRUE(rdna4);
  ASSERT_TRUE(gfx1250);
  EXPECT_EQ(*cdna3, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_EQ(*cdna4, *build_cdna4_salu_dependency_delay(ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(*rdna4, *build_s_wait_alu_va_sdst0(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(*gfx1250, *build_s_wait_alu_va_sdst0(ROCJITSU_CODE_ARCH_GFX1250));
}

TEST(InstrumentationBuilderDispatch, UnsupportedArchitectureFailsClosed) {
  EXPECT_FALSE(ib::is_admitted_arch(ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_s_mov_b64(20, 22, ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_s_getreg_b32(20, 0x2804u, ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_s_cmp_lg_u32(20, 22, ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_s_cselect_b32(20, 22, 24, ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_s_cmp_lg_u32(20, 22, ROCJITSU_CODE_ARCH_INVALID));
  EXPECT_FALSE(ib::build_s_cselect_b32(20, 22, 24, ROCJITSU_CODE_ARCH_INVALID));
  EXPECT_FALSE(ib::build_valu_to_salu_dependency_wait(ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_s_and_saveexec_b64(20, 22, ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(
      ib::build_v_lshrrev_b32(10, scalar_positive_inline_u32(3), 3, ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_v_min_u32(10, vector_source_vgpr(2), 3, ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_s_wait_flat_load0(ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_s_wait_global_store0(ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_s_wait_scalar_load0(ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_salu_dependency_delay(ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_s_trap(2, ROCJITSU_CODE_ARCH_CDNA2));
}

TEST(InstrumentationBuilderDispatch, VariableLengthRecipesSelectTargetBackend) {
  const auto cdna_mov = ib::build_v_mov_b32_literal(10, 0x12345678u, ROCJITSU_CODE_ARCH_CDNA4);
  const auto rdna_mov = ib::build_v_mov_b32_literal(10, 0x12345678u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto cdna_add = ib::build_v_add_u32(10, vector_source_vgpr(2), 3, ROCJITSU_CODE_ARCH_CDNA4);
  const auto rdna_add = ib::build_v_add_u32(10, vector_source_vgpr(2), 3, ROCJITSU_CODE_ARCH_RDNA4);
  const auto cdna_literal_add =
      ib::build_v_add_u32_literal(10, 0x12345678u, 3, ROCJITSU_CODE_ARCH_CDNA4);
  const auto cdna_in_place_literal_add =
      ib::build_v_add_u32_literal(10, 11, 0x12345678u, 10, ROCJITSU_CODE_ARCH_CDNA4);
  const auto cdna3_literal_add =
      ib::build_v_add_u32_literal(10, 0x12345678u, 3, ROCJITSU_CODE_ARCH_CDNA3);
  const auto cdna3_literal_add_with_unused_temp =
      ib::build_v_add_u32_literal(10, 11, 0x12345678u, 3, ROCJITSU_CODE_ARCH_CDNA3);
  const auto rdna_literal_add =
      ib::build_v_add_u32_literal(10, 0x12345678u, 3, ROCJITSU_CODE_ARCH_RDNA4);
  const auto rdna_literal_add_with_unused_temp =
      ib::build_v_add_u32_literal(10, 11, 0x12345678u, 3, ROCJITSU_CODE_ARCH_RDNA4);
  const auto cdna_multiply =
      ib::build_v_mul_lo_u32_literal(10, 11, 0x85ebca6bu, 10, ROCJITSU_CODE_ARCH_CDNA4);
  const auto rdna_multiply =
      ib::build_v_mul_lo_u32_literal(10, 11, 0x85ebca6bu, 10, ROCJITSU_CODE_ARCH_RDNA4);
  const auto cdna_store = ib::build_flat_store_b32(2, 7, ROCJITSU_CODE_ARCH_CDNA4);
  const auto rdna_store = ib::build_flat_store_b32(2, 7, ROCJITSU_CODE_ARCH_RDNA4);
  const auto gfx1250_store = ib::build_flat_store_b32(2, 7, ROCJITSU_CODE_ARCH_GFX1250);
  const auto cdna_private = ib::build_private_store_b32(7, 4, ROCJITSU_CODE_ARCH_CDNA4);
  const auto rdna_private = ib::build_private_store_b32(7, 4, ROCJITSU_CODE_ARCH_RDNA4);
  const auto cdna_atomic =
      ib::build_flat_atomic_add_u32(2, 7, 8, true, 2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto rdna_atomic =
      ib::build_flat_atomic_add_u32(2, 7, 8, true, 2, ROCJITSU_CODE_ARCH_RDNA4);
  const auto gfx1250_atomic =
      ib::build_flat_atomic_add_u32(2, 7, 8, true, 2, ROCJITSU_CODE_ARCH_GFX1250);
  const auto cdna_barrier = ib::build_workgroup_barrier(ROCJITSU_CODE_ARCH_CDNA4);
  const auto rdna_barrier = ib::build_workgroup_barrier(ROCJITSU_CODE_ARCH_RDNA4);
  const auto cdna_barrier_only = ib::build_workgroup_barrier_only(ROCJITSU_CODE_ARCH_CDNA4);
  const auto rdna_barrier_only = ib::build_workgroup_barrier_only(ROCJITSU_CODE_ARCH_RDNA4);
  const auto gfx1250_lds64 = ib::build_ds_store_b64(/*vaddr=*/3, /*vdata=*/4, /*byte_offset=*/16,
                                                    ROCJITSU_CODE_ARCH_GFX1250);
  const auto gfx1250_lds128 = ib::build_ds_store_b128(
      /*vaddr=*/3, /*vdata=*/4, /*byte_offset=*/16, ROCJITSU_CODE_ARCH_GFX1250);
  const auto gfx1250_bounds = ib::build_v_cmp_gt_u32_literal_vcc(
      /*literal=*/13080u, /*vsrc1=*/3, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(cdna_mov && rdna_mov && cdna_add && rdna_add && cdna_literal_add &&
              cdna_in_place_literal_add && cdna3_literal_add &&
              cdna3_literal_add_with_unused_temp && rdna_literal_add &&
              rdna_literal_add_with_unused_temp && cdna_multiply && rdna_multiply && cdna_store &&
              rdna_store && gfx1250_store && cdna_private && rdna_private && cdna_atomic &&
              rdna_atomic && gfx1250_atomic && cdna_barrier && rdna_barrier && gfx1250_lds64 &&
              gfx1250_lds128 && gfx1250_bounds);

  EXPECT_EQ(*cdna_mov, (std::vector<uint32_t>{0x7e1402ffu, 0x12345678u}));
  EXPECT_EQ(cdna_mov->size(), 2u);
  EXPECT_EQ(rdna_mov->size(), 3u);
  EXPECT_EQ(*cdna_add, (std::vector<uint32_t>{0xd1ff000au, 0x02020702u}));
  EXPECT_EQ(rdna_add->size(), 1u);
  EXPECT_EQ(cdna_literal_add->size(), 4u);
  EXPECT_EQ(cdna_in_place_literal_add->size(), 4u);
  EXPECT_EQ(*cdna3_literal_add_with_unused_temp, *cdna3_literal_add);
  EXPECT_EQ(cdna3_literal_add->size(), 2u);
  EXPECT_EQ(*rdna_literal_add_with_unused_temp, *rdna_literal_add);
  EXPECT_EQ(rdna_literal_add->size(), 2u);
  EXPECT_EQ(cdna_multiply->size(), 4u);
  EXPECT_EQ(rdna_multiply->size(), 3u);
  EXPECT_EQ(cdna_store->size(), 2u);
  EXPECT_EQ(rdna_store->size(), 3u);
  EXPECT_EQ((*rdna_store)[0] & 0x7fu, kRdna4FlatNoSaddrEncoding);
  EXPECT_EQ((*gfx1250_store)[0] & 0x7fu, kGfx1250FlatNoSaddrEncoding);
  EXPECT_EQ(cdna_private->size(), 2u);
  EXPECT_EQ(rdna_private->size(), 3u);
  EXPECT_EQ(ib::build_s_wait_private_load0(ROCJITSU_CODE_ARCH_CDNA4), 0xbf8c0f70u);
  EXPECT_EQ(ib::build_s_wait_private_store0(ROCJITSU_CODE_ARCH_CDNA4), 0xbf8c0f70u);
  EXPECT_EQ(cdna_atomic->size(), 2u);
  EXPECT_EQ(rdna_atomic->size(), 3u);
  EXPECT_EQ((*rdna_atomic)[0] & 0x7fu, kRdna4FlatNoSaddrEncoding);
  EXPECT_EQ((*gfx1250_atomic)[0] & 0x7fu, kGfx1250FlatNoSaddrEncoding);
  EXPECT_EQ(*cdna_barrier, (std::vector<uint32_t>{0xbf8c0070u, 0xbf8a0000u}));
  EXPECT_EQ(rdna_barrier->size(), 3u);
  ASSERT_TRUE(cdna_barrier_only && rdna_barrier_only);
  EXPECT_EQ(*cdna_barrier_only, (std::vector<uint32_t>{0xbf8a0000u}));
  EXPECT_EQ(rdna_barrier_only->size(), 2u);
  EXPECT_EQ(*gfx1250_lds64, (std::vector<uint32_t>{0xD9340010u, 0x00000403u}));
  EXPECT_EQ(*gfx1250_lds128, (std::vector<uint32_t>{0xDB7C0010u, 0x00000403u}));
  EXPECT_EQ(*gfx1250_bounds, (std::vector<uint32_t>{0x7C9806FFu, 0x00003318u}));
}

TEST(InstrumentationBuilderDispatch, CompareSwapB64SelectsEveryAdmittedBackend) {
  const auto rdna = ib::build_flat_atomic_cmpswap_b64(8, 10, 10, true, 2, ROCJITSU_CODE_ARCH_RDNA4);
  const auto gfx1250 =
      ib::build_flat_atomic_cmpswap_b64(8, 10, 10, true, 2, ROCJITSU_CODE_ARCH_GFX1250);
  const auto cdna3 =
      ib::build_flat_atomic_cmpswap_b64(8, 10, 10, true, 2, ROCJITSU_CODE_ARCH_CDNA3);
  const auto cdna4 =
      ib::build_flat_atomic_cmpswap_b64(8, 10, 10, true, 2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto expected_rdna =
      build_flat_atomic_cmpswap_b64_vaddr_vsrc_vdst(8, 10, 10, true, 2, ROCJITSU_CODE_ARCH_RDNA4);
  const auto expected_gfx1250 =
      build_gfx1250_flat_atomic_cmpswap_b64(8, 10, 10, true, 2, ROCJITSU_CODE_ARCH_GFX1250);

  ASSERT_TRUE(rdna && gfx1250 && cdna3 && cdna4 && expected_rdna && expected_gfx1250);
  EXPECT_EQ(*rdna, std::vector<uint32_t>(expected_rdna->begin(), expected_rdna->end()));
  EXPECT_EQ(*gfx1250, std::vector<uint32_t>(expected_gfx1250->begin(), expected_gfx1250->end()));
  EXPECT_EQ(cdna3->size(), 2u);
  EXPECT_EQ(cdna4->size(), 2u);
  EXPECT_FALSE(ib::build_flat_atomic_cmpswap_b64(8, 11, 10, true, 2, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(ib::build_flat_atomic_cmpswap_b64(8, 11, 10, true, 2, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstrumentationBuilderDispatch, VariableLengthRecipesFailClosed) {
  EXPECT_FALSE(ib::build_v_add_u32_literal(3, 0x12345678u, 3, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(ib::build_v_add_u32_literal(3, 3, 0x12345678u, 3, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(ib::build_v_and_b32_literal(3, 7, 3, ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_v_mul_lo_u32_literal(3, 4, 7, 4, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(ib::build_v_add_u32(3, 4, 3, ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_flat_store_b32(3, 7, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(ib::build_flat_load_b32(2, 7, ROCJITSU_CODE_ARCH_CDNA4, 0x1000u));
  EXPECT_FALSE(ib::build_private_store_b32(7, 0x1000u, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(ib::build_flat_atomic_add_u32(2, 7, 8, true, 1, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(ib::build_workgroup_barrier(ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_workgroup_barrier_only(ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(ib::build_ds_store_b64(3, 4, 0, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(ib::build_v_cmp_gt_u32_literal_vcc(13080u, 3, ROCJITSU_CODE_ARCH_CDNA4));
}

} // namespace
} // namespace rocjitsu
