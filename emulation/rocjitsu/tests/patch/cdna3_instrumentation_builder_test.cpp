// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/cdna3_instrumentation_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string_view>

namespace rocjitsu {
namespace {

TEST(Cdna3InstrumentationBuilder, ScalarAndVectorEncodingsMatchLlvm) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA3;
  EXPECT_EQ(build_cdna3_s_mov_b64(20, 22, kArch), 0xbe940116u);
  EXPECT_EQ(build_cdna3_s_and_saveexec_b64(20, 22, kArch), 0xbe942016u);
  EXPECT_EQ(build_cdna3_s_andn2_b64(20, 22, 24, kArch), 0x89941816u);
  EXPECT_EQ(build_cdna3_s_and_b64(20, 22, 24, kArch), 0x86941816u);
  EXPECT_EQ(build_cdna3_s_bcnt1_i32_b64(20, 22, kArch), 0xbe940d16u);
  EXPECT_EQ(build_cdna3_s_xor_b64(20, 22, 24, kArch), 0x88941816u);
  EXPECT_EQ(build_cdna3_s_cselect_b32(20, scalar_positive_inline_u32(1),
                                      scalar_positive_inline_u32(0), kArch),
            0x85148081u);
  EXPECT_EQ(build_cdna3_s_cmp_lg_u32(20, scalar_positive_inline_u32(0), kArch), 0xbf078014u);
  EXPECT_EQ(build_cdna3_s_cmp_eq_u32(20, scalar_positive_inline_u32(0), kArch), 0xbf068014u);
  EXPECT_EQ(build_cdna3_v_lshrrev_b32(10, scalar_positive_inline_u32(3), 3, kArch), 0x20140683u);
  EXPECT_EQ(build_cdna3_v_lshlrev_b32(10, scalar_positive_inline_u32(3), 3, kArch), 0x24140683u);
  EXPECT_EQ(build_cdna3_v_and_b32(10, scalar_positive_inline_u32(7), 3, kArch), 0x26140687u);
  EXPECT_EQ(build_cdna3_v_xor_b32(10, scalar_positive_inline_u32(7), 3, kArch), 0x2a140687u);
  EXPECT_EQ(build_cdna3_v_add_u32(10, vector_source_vgpr(2), 3, kArch), 0x68140702u);
  EXPECT_EQ(build_cdna3_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), 3, kArch), 0x7d940680u);
  EXPECT_EQ(build_cdna3_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), 3, kArch), 0x7d9a0680u);
  EXPECT_EQ(build_cdna3_v_cmp_ne_u16_vcc(scalar_positive_inline_u32(0), 3, kArch), 0x7d5a0680u);
  EXPECT_EQ(build_cdna3_v_cmp_gt_u32_vcc(scalar_positive_inline_u32(0), 3, kArch), 0x7d980680u);
  EXPECT_EQ(build_cdna3_v_readfirstlane_b32(20, 8, kArch), 0x7e280508u);
  EXPECT_EQ(build_cdna3_v_mbcnt_lo_u32_b32(10, /*-1 inline constant=*/193u,
                                           scalar_positive_inline_u32(0), kArch),
            (std::array<uint32_t, 2>{0xd28c000au, 0x000100c1u}));
  EXPECT_EQ(build_cdna3_v_mbcnt_hi_u32_b32(10, /*-1 inline constant=*/193u, vector_source_vgpr(10),
                                           kArch),
            (std::array<uint32_t, 2>{0xd28d000au, 0x000214c1u}));
  EXPECT_EQ(build_cdna3_v_mov_b32_literal(10, 0x12345678u, kArch),
            (std::array<uint32_t, 2>{0x7e1402ffu, 0x12345678u}));
  EXPECT_EQ(build_cdna3_vop2_literal(cdna3::kVAndB32Vop2, 10, 0x12345678u, 3, kArch),
            (std::array<uint32_t, 2>{0x261406ffu, 0x12345678u}));
  EXPECT_EQ(build_cdna3_v_mul_lo_u32_literal(10, 11, 0x12345678u, 3, kArch),
            (std::vector<uint32_t>{0x7e1602ffu, 0x12345678u, 0xd285000au, 0x0002070bu}));
}

TEST(Cdna3InstrumentationBuilder, MemoryAndAtomicEncodingsMatchLlvmAndDecoder) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA3;
  const auto buffer_inv = build_cdna3_buffer_inv_sc1(kArch);
  const auto store = build_cdna3_flat_store_b32(10, 12, 4, kArch);
  const auto load = build_cdna3_flat_load_b32(10, 13, 4, kArch);
  const auto ds_store = build_cdna3_ds_store_b32(2, 3, 4, kArch);
  const auto ds_store64 = build_cdna3_ds_store_b64(2, 6, 4, kArch);
  const auto ds_store128 = build_cdna3_ds_store_b128(2, 8, 4, kArch);
  const auto ds_xchg = build_cdna3_ds_storexchg_rtn_b64(4, 2, 6, 4, kArch);
  const auto ds_xchg32 = build_cdna3_ds_storexchg_rtn_b32(4, 2, 6, 4, kArch);
  const auto ds_or = build_cdna3_ds_or_rtn_b32(4, 2, 6, 4, kArch);
  const auto ds_load = build_cdna3_ds_load_b32(4, 2, 4, kArch);
  const auto add = build_cdna3_flat_atomic_add_u32(2, 4, 5, true, 2, kArch);
  const auto bit_or = build_cdna3_flat_atomic_or_u32(2, 4, 5, true, 2, kArch);
  const auto cmp_swap = build_cdna3_flat_atomic_cmpswap_b32(2, 4, 6, true, 2, kArch);
  const auto cmp_swap64 = build_cdna3_flat_atomic_cmpswap_b64(2, 4, 6, true, 2, kArch);
  const auto swap64 = build_cdna3_flat_atomic_swap_b64(2, 4, 6, true, 2, kArch);
  const auto add64 = build_cdna3_flat_atomic_add_u64(2, 4, 6, true, 2, kArch);
  ASSERT_TRUE(buffer_inv && store && load && ds_store && ds_store64 && ds_store128 && ds_xchg &&
              ds_xchg32 && ds_or && ds_load && add && bit_or && cmp_swap && cmp_swap64 && swap64 &&
              add64);
  EXPECT_EQ(*buffer_inv, (std::array<uint32_t, 2>{0xe0a48000u, 0x00000000u}));
  EXPECT_EQ(*store, (std::array<uint32_t, 2>{0xdc700004u, 0x00000c0au}));
  EXPECT_EQ(*load, (std::array<uint32_t, 2>{0xdc500004u, 0x0d00000au}));
  EXPECT_EQ(*ds_store, (std::array<uint32_t, 2>{0xd81a0004u, 0x00000302u}));
  EXPECT_EQ(*ds_store64, (std::array<uint32_t, 2>{0xd89a0004u, 0x00000602u}));
  EXPECT_EQ(*ds_store128, (std::array<uint32_t, 2>{0xd9be0004u, 0x00000802u}));
  EXPECT_EQ(*ds_xchg, (std::array<uint32_t, 2>{0xd8da0004u, 0x04000602u}));
  EXPECT_EQ(*ds_xchg32, (std::array<uint32_t, 2>{0xd85a0004u, 0x04000602u}));
  EXPECT_EQ(*ds_or, (std::array<uint32_t, 2>{0xd8540004u, 0x04000602u}));
  EXPECT_EQ(*ds_load, (std::array<uint32_t, 2>{0xd86c0004u, 0x04000002u}));
  EXPECT_EQ(*add, (std::array<uint32_t, 2>{0xdd090000u, 0x05000402u}));
  EXPECT_EQ(*bit_or, (std::array<uint32_t, 2>{0xdd250000u, 0x05000402u}));
  EXPECT_EQ(*cmp_swap, (std::array<uint32_t, 2>{0xdd050000u, 0x06000402u}));
  EXPECT_EQ(*cmp_swap64, (std::array<uint32_t, 2>{0xdd850000u, 0x06000402u}));
  EXPECT_EQ(*swap64, (std::array<uint32_t, 2>{0xdd810000u, 0x06000402u}));
  EXPECT_EQ(*add64, (std::array<uint32_t, 2>{0xdd890000u, 0x06000402u}));

  auto decoder = Decoder::create(kArch);
  ASSERT_NE(decoder, nullptr);
  for (const auto &[words, mnemonic] :
       std::array<std::pair<const uint32_t *, std::string_view>, 15>{{
           {store->data(), "flat_store_dword"},
           {load->data(), "flat_load_dword"},
           {ds_store->data(), "ds_write_b32"},
           {ds_store64->data(), "ds_write_b64"},
           {ds_store128->data(), "ds_write_b128"},
           {ds_xchg->data(), "ds_wrxchg_rtn_b64"},
           {ds_xchg32->data(), "ds_wrxchg_rtn_b32"},
           {ds_or->data(), "ds_or_rtn_b32"},
           {ds_load->data(), "ds_read_b32"},
           {add->data(), "flat_atomic_add"},
           {bit_or->data(), "flat_atomic_or"},
           {cmp_swap->data(), "flat_atomic_cmpswap"},
           {cmp_swap64->data(), "flat_atomic_cmpswap_x2"},
           {swap64->data(), "flat_atomic_swap_x2"},
           {add64->data(), "flat_atomic_add_x2"},
       }}) {
    std::unique_ptr<Instruction> instruction(decoder->decode(words));
    ASSERT_NE(instruction, nullptr);
    EXPECT_EQ(std::string_view(instruction->mnemonic()), mnemonic);
  }
}

TEST(Cdna3InstrumentationBuilder, ScratchWaitAndAddressRecipesMatchLlvm) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA3;
  const auto store = build_cdna3_address_free_scratch_store_b32(7, 4, kArch);
  const auto load = build_cdna3_address_free_scratch_load_b32(7, 4, kArch);
  const auto saddr_store = build_cdna3_scratch_store_b32_saddr(10, 33, 4, kArch);
  const auto saddr_load = build_cdna3_scratch_load_b32_saddr(10, 33, 4, kArch);
  const auto add_offset = build_cdna3_v_add_u64_vgpr_offset(10, 12, kArch);
  const auto literal_offset = build_cdna3_v_add_u64_signed_i24(10, 0x7fffff, kArch);
  const auto negative_inline_offset = build_cdna3_v_add_u64_signed_i24(10, -4, kArch);
  const auto positive_inline_offset = build_cdna3_v_add_u64_signed_i24(10, 4, kArch);
  const auto scalar_load = build_cdna3_s_load_dword(20, 0, 4, kArch);
  ASSERT_TRUE(store && load && saddr_store && saddr_load && add_offset && literal_offset &&
              negative_inline_offset && positive_inline_offset && scalar_load);
  EXPECT_EQ(*store, (std::array<uint32_t, 2>{0xdc704004u, 0x007f0700u}));
  EXPECT_EQ(*load, (std::array<uint32_t, 2>{0xdc504004u, 0x077f0000u}));
  EXPECT_EQ(*saddr_store, (std::array<uint32_t, 2>{0xdc704004u, 0x00210a00u}));
  EXPECT_EQ(*saddr_load, (std::array<uint32_t, 2>{0xdc504004u, 0x0a210000u}));
  EXPECT_EQ(*add_offset, (std::vector<uint32_t>{0x3214150cu, 0x38161680u}));
  EXPECT_EQ(*literal_offset, (std::vector<uint32_t>{0x321414ffu, 0x007fffffu, 0x38161680u}));
  EXPECT_EQ(*negative_inline_offset, (std::vector<uint32_t>{0x321414c4u, 0x381616c1u}));
  EXPECT_EQ(*positive_inline_offset, (std::vector<uint32_t>{0x32141484u, 0x38161680u}));
  EXPECT_FALSE(build_cdna3_v_add_u64_signed_i24(10, 1 << 23, kArch));
  EXPECT_EQ(*scalar_load, (std::array<uint32_t, 2>{0xc0020500u, 0x00000004u}));
  EXPECT_EQ(build_cdna3_s_wait_vmcnt0(kArch), 0xbf8c0f70u);
  EXPECT_EQ(build_cdna3_s_wait_lgkmcnt0(kArch), 0xbf8cc07fu);
  EXPECT_EQ(build_cdna3_s_wait_vmcnt_lgkmcnt0(kArch), 0xbf8c0070u);
  EXPECT_EQ(build_cdna3_s_barrier(kArch), 0xbf8a0000u);
}

TEST(Cdna3InstrumentationBuilder, RejectsWrongArchitectureAndInvalidTuples) {
  EXPECT_FALSE(build_cdna3_s_mov_b64(20, 22, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_cdna3_v_lshrrev_b32(10, 3, 3, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_cdna3_flat_store_b32(255, 7, 0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_flat_atomic_add_u32(3, 4, 5, true, 2, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_flat_atomic_add_u32(2, 4, 5, true, 1, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_flat_atomic_cmpswap_b32(2, 5, 6, true, 2, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_flat_atomic_cmpswap_b32(2, 4, 6, false, 2, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_flat_atomic_swap_b64(2, 4, 6, false, 2, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_flat_atomic_add_u64(2, 4, 6, false, 2, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_ds_storexchg_rtn_b64(5, 2, 6, 4, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_ds_storexchg_rtn_b64(4, 2, 7, 4, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_flat_store_b32(2, 7, 0x1000, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_flat_load_b32(2, 7, 0x1000, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_address_free_scratch_store_b32(7, 2, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_address_free_scratch_load_b32(7, 2, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_s_mov_b64(20, kVopLiteralSource, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_s_and_saveexec_b64(20, kVopLiteralSource, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_v_mul_lo_u32_literal(10, 3, 0x12345678u, 3, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(build_cdna3_address_free_scratch_store_b32(7, kMaxCdnaAddressFreeScratchPrivateBytes,
                                                          ROCJITSU_CODE_ARCH_CDNA3));
}

} // namespace
} // namespace rocjitsu
