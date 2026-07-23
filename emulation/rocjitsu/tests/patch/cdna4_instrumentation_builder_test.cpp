// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/cdna4_instrumentation_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>

namespace rocjitsu {
namespace {

TEST(Cdna4InstrumentationBuilder, ScalarControlMatchesLlvmAndDecoder) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto mov_b64 = build_cdna4_s_mov_b64(20, 22, kArch);
  const auto save_exec = build_cdna4_s_and_saveexec_b64(20, 22, kArch);
  const auto andn2 = build_cdna4_s_andn2_b64(20, 22, 24, kArch);
  const auto and_b64 = build_cdna4_s_and_b64(20, 22, 24, kArch);
  const auto bcnt = build_cdna4_s_bcnt1_i32_b64(20, 22, kArch);
  const auto xor_b64 = build_cdna4_s_xor_b64(20, 22, 24, kArch);
  const auto save_scc = build_cdna4_s_cselect_b32(20, scalar_positive_inline_u32(1),
                                                  scalar_positive_inline_u32(0), kArch);
  const auto restore_scc = build_cdna4_s_cmp_lg_u32(20, scalar_positive_inline_u32(0), kArch);
  const auto cmp_eq = build_cdna4_s_cmp_eq_u32(20, scalar_positive_inline_u32(0), kArch);
  const auto scc0 = build_cdna4_s_cbranch_scc0(1, kArch);
  const auto scc1 = build_cdna4_s_cbranch_scc1(1, kArch);
  const auto vccz = build_cdna4_s_cbranch_vccz(1, kArch);
  const auto vccnz = build_cdna4_s_cbranch_vccnz(1, kArch);
  const auto execz = build_cdna4_s_cbranch_execz(1, kArch);
  const auto execnz = build_cdna4_s_cbranch_execnz(1, kArch);
  ASSERT_TRUE(mov_b64 && save_exec && andn2 && and_b64 && bcnt && xor_b64 && save_scc &&
              restore_scc && cmp_eq && scc0 && scc1 && vccz && vccnz && execz && execnz);

  const std::array<std::pair<uint32_t, std::string_view>, 15> cases = {{
      {*mov_b64, "s_mov_b64"},
      {*save_exec, "s_and_saveexec_b64"},
      {*andn2, "s_andn2_b64"},
      {*and_b64, "s_and_b64"},
      {*bcnt, "s_bcnt1_i32_b64"},
      {*xor_b64, "s_xor_b64"},
      {*save_scc, "s_cselect_b32"},
      {*restore_scc, "s_cmp_lg_u32"},
      {*cmp_eq, "s_cmp_eq_u32"},
      {*scc0, "s_cbranch_scc0"},
      {*scc1, "s_cbranch_scc1"},
      {*vccz, "s_cbranch_vccz"},
      {*vccnz, "s_cbranch_vccnz"},
      {*execz, "s_cbranch_execz"},
      {*execnz, "s_cbranch_execnz"},
  }};
  const std::array<uint32_t, 15> llvm_words = {
      0xbe940116u, 0xbe942016u, 0x89941816u, 0x86941816u, 0xbe940d16u,
      0x88941816u, 0x85148081u, 0xbf078014u, 0xbf068014u, 0xbf840001u,
      0xbf850001u, 0xbf860001u, 0xbf870001u, 0xbf880001u, 0xbf890001u,
  };
  auto decoder = Decoder::create(kArch);
  ASSERT_NE(decoder, nullptr);
  for (size_t i = 0; i < cases.size(); ++i) {
    EXPECT_EQ(cases[i].first, llvm_words[i]);
    std::unique_ptr<Instruction> inst(decoder->decode(&cases[i].first));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(std::string_view(inst->mnemonic()), cases[i].second);
  }
}

TEST(Cdna4InstrumentationBuilder, ScalarControlRejectsWrongArchAndInvalidOperands) {
  EXPECT_FALSE(build_cdna4_s_mov_b64(20, 22, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_cdna4_s_mov_b64(127, 22, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_cdna4_s_mov_b64(20, kVopLiteralSource, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_cdna4_s_and_saveexec_b64(20, kVopLiteralSource, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_cdna4_s_andn2_b64(20, 255, 24, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_cdna4_s_and_b64(20, 255, 24, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_cdna4_s_bcnt1_i32_b64(128, 22, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_cdna4_s_cselect_b32(128, 128, 128, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_cdna4_s_cbranch_scc0(1, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(Cdna4InstrumentationBuilder, VectorArithmeticMatchesLlvmAndDecoder) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto lshr = build_cdna4_v_lshrrev_b32(10, scalar_positive_inline_u32(3), 3, kArch);
  const auto lshl = build_cdna4_v_lshlrev_b32(10, scalar_positive_inline_u32(3), 3, kArch);
  const auto bit_and = build_cdna4_v_and_b32(10, scalar_positive_inline_u32(7), 3, kArch);
  const auto bit_xor = build_cdna4_v_xor_b32(10, scalar_positive_inline_u32(7), 3, kArch);
  const auto min_literal = build_cdna4_v_min_u32_literal(10, 0x12345678u, 3, kArch);
  const auto add = build_cdna4_v_add_u32(10, vector_source_vgpr(2), 3, kArch);
  const auto add_literal = build_cdna4_v_add_u32_literal(10, 0x12345678u, 3, kArch);
  const auto mad = build_cdna4_v_mad_u32_u24(10, 20, 3, 4, kArch);
  const auto multiply = build_cdna4_v_mul_lo_u32_literal(10, 11, 0x85ebca6bu, 10, kArch);
  ASSERT_TRUE(lshr && lshl && bit_and && bit_xor && min_literal && add && add_literal && mad &&
              multiply);

  EXPECT_EQ(*lshr, 0x20140683u);
  EXPECT_EQ(*lshl, 0x24140683u);
  EXPECT_EQ(*bit_and, 0x26140687u);
  EXPECT_EQ(*bit_xor, 0x2a140687u);
  EXPECT_EQ(*min_literal, (std::array<uint32_t, 2>{0x1c1406ffu, 0x12345678u}));
  EXPECT_EQ(*add, (std::array<uint32_t, 2>{0xd1ff000au, 0x02020702u}));
  EXPECT_EQ(*add_literal,
            (std::vector<uint32_t>{0x7e1402ffu, 0x12345678u, 0xd1ff000au, 0x0202070au}));
  EXPECT_EQ(*mad, (std::array<uint32_t, 2>{0xd1c3000au, 0x04120614u}));
  EXPECT_EQ(*multiply, (std::vector<uint32_t>{0x7e1602ffu, 0x85ebca6bu, 0xd285000au, 0x0002150bu}));

  auto decoder = Decoder::create(kArch);
  ASSERT_NE(decoder, nullptr);
  const auto expect_decode = [&](const uint32_t *words, std::string_view mnemonic) {
    std::unique_ptr<Instruction> inst(decoder->decode(words));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(std::string_view(inst->mnemonic()), mnemonic);
  };
  expect_decode(&*lshr, "v_lshrrev_b32_e32");
  expect_decode(&*lshl, "v_lshlrev_b32_e32");
  expect_decode(min_literal->data(), "v_min_u32_e32");
  expect_decode(add->data(), "v_add3_u32");
  expect_decode(mad->data(), "v_mad_u32_u24");
  expect_decode(multiply->data(), "v_mov_b32_e32");
  expect_decode(multiply->data() + 2, "v_mul_lo_u32");
}

TEST(Cdna4InstrumentationBuilder, VectorIdentityMatchesLlvmAndDecoder) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto readfirst = build_cdna4_v_readfirstlane_b32(20, 8, kArch);
  const auto mbcnt_lo =
      build_cdna4_v_mbcnt_lo_u32_b32(13, 0xc1, scalar_positive_inline_u32(0), kArch);
  const auto mbcnt_hi = build_cdna4_v_mbcnt_hi_u32_b32(13, 0xc1, vector_source_vgpr(13), kArch);
  const auto cmp_eq = build_cdna4_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), 3, kArch);
  const auto cmp_ne = build_cdna4_v_cmp_ne_u32_vcc(vector_source_vgpr(2), 3, kArch);
  const auto cmp_gt = build_cdna4_v_cmp_gt_u32_vcc(scalar_positive_inline_u32(7), 3, kArch);
  ASSERT_TRUE(readfirst && mbcnt_lo && mbcnt_hi && cmp_eq && cmp_ne && cmp_gt);
  EXPECT_EQ(*readfirst, 0x7e280508u);
  EXPECT_EQ(*mbcnt_lo, (std::array<uint32_t, 2>{0xd28c000du, 0x000100c1u}));
  EXPECT_EQ(*mbcnt_hi, (std::array<uint32_t, 2>{0xd28d000du, 0x00021ac1u}));
  EXPECT_EQ(*cmp_eq, 0x7d940680u);
  EXPECT_EQ(*cmp_ne, 0x7d9a0702u);
  EXPECT_EQ(*cmp_gt, 0x7d980687u);

  auto decoder = Decoder::create(kArch);
  ASSERT_NE(decoder, nullptr);
  for (const auto &[words, mnemonic] :
       std::array<std::pair<const uint32_t *, std::string_view>, 6>{{
           {&*readfirst, "v_readfirstlane_b32_e32"},
           {mbcnt_lo->data(), "v_mbcnt_lo_u32_b32"},
           {mbcnt_hi->data(), "v_mbcnt_hi_u32_b32"},
           {&*cmp_eq, "v_cmp_eq_u32_e32"},
           {&*cmp_ne, "v_cmp_ne_u32_e32"},
           {&*cmp_gt, "v_cmp_gt_u32_e32"},
       }}) {
    std::unique_ptr<Instruction> inst(decoder->decode(words));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(std::string_view(inst->mnemonic()), mnemonic);
  }
}

TEST(Cdna4InstrumentationBuilder, VectorAddressArithmeticMatchesLlvm) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto vgpr_offset = build_cdna4_v_add_u64_vgpr_offset(10, 12, kArch);
  const auto positive = build_cdna4_v_add_u64_signed_i24(10, 0x7fffff, kArch);
  const auto negative = build_cdna4_v_add_u64_signed_i24(10, -4, kArch);
  ASSERT_TRUE(vgpr_offset && positive && negative);
  EXPECT_EQ(*vgpr_offset, (std::vector<uint32_t>{0x3214150cu, 0x38161680u}));
  EXPECT_EQ(*positive, (std::vector<uint32_t>{0x321414ffu, 0x007fffffu, 0x38161680u}));
  EXPECT_EQ(*negative, (std::vector<uint32_t>{0x321414c4u, 0x381616c1u}));
}

TEST(Cdna4InstrumentationBuilder, VectorLiteralOverlapFailsClosed) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  EXPECT_FALSE(build_cdna4_v_add_u32_literal(10, 7, 10, kArch));
  EXPECT_FALSE(build_cdna4_v_mul_lo_u32_literal(10, 10, 7, 10, kArch));
  EXPECT_FALSE(build_cdna4_v_add_u32(10, 0, 3, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(Cdna4InstrumentationBuilder, SmemAndFlatPublicationMatchLlvmAndDecoder) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto smem = build_cdna4_s_load_dword(20, 0, 4, kArch);
  const auto store = build_cdna4_flat_store_b32(10, 12, 4, kArch);
  const auto load = build_cdna4_flat_load_b32(10, 13, 4, kArch);
  ASSERT_TRUE(smem && store && load);
  EXPECT_EQ(*smem, (std::array<uint32_t, 2>{0xc0020500u, 0x00000004u}));
  EXPECT_EQ(*store, (std::array<uint32_t, 2>{0xdc700004u, 0x00000c0au}));
  EXPECT_EQ(*load, (std::array<uint32_t, 2>{0xdc500004u, 0x0d00000au}));

  auto decoder = Decoder::create(kArch);
  ASSERT_NE(decoder, nullptr);
  for (const auto &[words, mnemonic] :
       std::array<std::pair<const uint32_t *, std::string_view>, 3>{{
           {smem->data(), "s_load_dword"},
           {store->data(), "flat_store_dword"},
           {load->data(), "flat_load_dword"},
       }}) {
    std::unique_ptr<Instruction> inst(decoder->decode(words));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(std::string_view(inst->mnemonic()), mnemonic);
  }
}

TEST(Cdna4InstrumentationBuilder, SmemAndFlatPublicationBoundsFailClosed) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  EXPECT_FALSE(build_cdna4_s_load_dword(102, 0, 4, kArch));
  EXPECT_FALSE(build_cdna4_s_load_dword(20, 1, 4, kArch));
  EXPECT_FALSE(build_cdna4_s_load_dword(20, 0, 2, kArch));
  EXPECT_FALSE(build_cdna4_s_load_dword(20, 0, 0x100000u, kArch));
  EXPECT_TRUE(build_cdna4_flat_store_b32(10, 12, 0xfff, kArch));
  EXPECT_FALSE(build_cdna4_flat_store_b32(3, 12, 0, kArch));
  EXPECT_FALSE(build_cdna4_flat_load_b32(10, 13, 0x1000, kArch));
  EXPECT_FALSE(build_cdna4_flat_load_b32(10, 13, 0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(Cdna4InstrumentationBuilder, DsAndFlatAtomicsMatchLlvmAndDecoder) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto ds_store = build_cdna4_ds_store_b32(2, 3, 4, kArch);
  const auto ds_xchg = build_cdna4_ds_storexchg_rtn_b64(4, 2, 6, 4, kArch);
  const auto add = build_cdna4_flat_atomic_add_u32(2, 4, 5, true, 2, kArch);
  const auto bit_or = build_cdna4_flat_atomic_or_u32(2, 4, 5, true, 2, kArch);
  const auto cmp_swap = build_cdna4_flat_atomic_cmpswap_b32(2, 4, 6, true, 2, kArch);
  const auto swap64 = build_cdna4_flat_atomic_swap_b64(2, 4, 6, true, 2, kArch);
  const auto add64 = build_cdna4_flat_atomic_add_u64(2, 4, 6, true, 2, kArch);
  ASSERT_TRUE(ds_store && ds_xchg && add && bit_or && cmp_swap && swap64 && add64);
  EXPECT_EQ(*ds_store, (std::array<uint32_t, 2>{0xd81a0004u, 0x00000302u}));
  EXPECT_EQ(*ds_xchg, (std::array<uint32_t, 2>{0xd8da0004u, 0x04000602u}));
  EXPECT_EQ(*add, (std::array<uint32_t, 2>{0xdd090000u, 0x05000402u}));
  EXPECT_EQ(*bit_or, (std::array<uint32_t, 2>{0xdd250000u, 0x05000402u}));
  EXPECT_EQ(*cmp_swap, (std::array<uint32_t, 2>{0xdd050000u, 0x06000402u}));
  EXPECT_EQ(*swap64, (std::array<uint32_t, 2>{0xdd810000u, 0x06000402u}));
  EXPECT_EQ(*add64, (std::array<uint32_t, 2>{0xdd890000u, 0x06000402u}));

  auto decoder = Decoder::create(kArch);
  ASSERT_NE(decoder, nullptr);
  for (const auto &[words, mnemonic] :
       std::array<std::pair<const uint32_t *, std::string_view>, 7>{{
           {ds_store->data(), "ds_write_b32"},
           {ds_xchg->data(), "ds_wrxchg_rtn_b64"},
           {add->data(), "flat_atomic_add"},
           {bit_or->data(), "flat_atomic_or"},
           {cmp_swap->data(), "flat_atomic_cmpswap"},
           {swap64->data(), "flat_atomic_swap_x2"},
           {add64->data(), "flat_atomic_add_x2"},
       }}) {
    std::unique_ptr<Instruction> inst(decoder->decode(words));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(std::string_view(inst->mnemonic()), mnemonic);
  }
}

TEST(Cdna4InstrumentationBuilder, DsAndFlatAtomicTuplesFailClosed) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  EXPECT_FALSE(build_cdna4_ds_storexchg_rtn_b64(5, 2, 6, 0, kArch));
  EXPECT_FALSE(build_cdna4_ds_storexchg_rtn_b64(4, 2, 7, 0, kArch));
  EXPECT_FALSE(build_cdna4_flat_atomic_add_u32(3, 4, 5, true, 2, kArch));
  EXPECT_FALSE(build_cdna4_flat_atomic_add_u32(2, 4, 5, true, 1, kArch));
  EXPECT_FALSE(build_cdna4_flat_atomic_cmpswap_b32(2, 5, 6, true, 2, kArch));
  EXPECT_FALSE(build_cdna4_flat_atomic_cmpswap_b32(2, 4, 6, false, 2, kArch));
  EXPECT_FALSE(build_cdna4_flat_atomic_swap_b64(2, 4, 7, true, 2, kArch));
  EXPECT_FALSE(build_cdna4_flat_atomic_add_u64(2, 4, 6, false, 2, kArch));
}

TEST(Cdna4InstrumentationBuilder, BuildAddressFreeScratchB32) {
  const auto store = build_cdna4_address_free_scratch_store_b32(
      /*vsrc=*/7, /*byte_offset=*/4, ROCJITSU_CODE_ARCH_CDNA4);
  const auto load = build_cdna4_address_free_scratch_load_b32(
      /*vdst=*/7, /*byte_offset=*/4, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(store);
  ASSERT_TRUE(load);
  EXPECT_EQ(*store, (std::array<uint32_t, 2>{0xdc704004u, 0x007f0700u}));
  EXPECT_EQ(*load, (std::array<uint32_t, 2>{0xdc504004u, 0x077f0000u}));

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> store_inst(decoder->decode(store->data()));
  std::unique_ptr<Instruction> load_inst(decoder->decode(load->data()));
  ASSERT_NE(store_inst, nullptr);
  ASSERT_NE(load_inst, nullptr);
  EXPECT_EQ(store_inst->mnemonic(), "scratch_store_dword");
  EXPECT_EQ(load_inst->mnemonic(), "scratch_load_dword");
  EXPECT_EQ(store_inst->size(), 8u);
  EXPECT_EQ(load_inst->size(), 8u);
}

TEST(Cdna4InstrumentationBuilder, AddressFreeScratchOffsetBoundary) {
  EXPECT_TRUE(build_cdna4_address_free_scratch_store_b32(255, kMaxCdnaAddressFreeScratchDwordOffset,
                                                         ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_TRUE(build_cdna4_address_free_scratch_load_b32(255, kMaxCdnaAddressFreeScratchDwordOffset,
                                                        ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_cdna4_address_free_scratch_store_b32(0, kMaxCdnaAddressFreeScratchPrivateBytes,
                                                          ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_cdna4_address_free_scratch_load_b32(0, kMaxCdnaAddressFreeScratchPrivateBytes,
                                                         ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_cdna4_address_free_scratch_store_b32(0, 2, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_cdna4_address_free_scratch_load_b32(0, 0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(Cdna4InstrumentationBuilder, BuildPrivateWait) {
  EXPECT_EQ(build_cdna4_s_wait_vmcnt0(ROCJITSU_CODE_ARCH_CDNA4), 0xbf8c0f70u);
  EXPECT_FALSE(build_cdna4_s_wait_vmcnt0(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(Cdna4InstrumentationBuilder, WaitBarrierAndCacheControlMatchLlvmAndDecoder) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto flat_wait = build_cdna4_s_wait_flat0(kArch);
  const auto lds_wait = build_cdna4_s_wait_lds0(kArch);
  const auto scalar_wait = build_cdna4_s_wait_scalar_load0(kArch);
  const auto barrier = build_cdna4_s_barrier(kArch);
  const auto waited_barrier = build_cdna4_s_barrier_with_memory_wait(kArch);
  const auto trap = build_cdna4_s_trap(2, kArch);
  const auto dependency_delay = build_cdna4_salu_dependency_delay(kArch);
  const auto icache_inv = build_cdna4_s_icache_inv(kArch);
  const auto dcache_inv = build_cdna4_s_dcache_inv(kArch);
  const auto dcache_wb = build_cdna4_s_dcache_wb(kArch);
  const auto dcache_inv_vol = build_cdna4_s_dcache_inv_vol(kArch);
  const auto buffer_inv = build_cdna4_buffer_inv_sc1(kArch);
  const auto dcache_wb_vol = build_cdna4_s_dcache_wb_vol(kArch);
  ASSERT_TRUE(flat_wait && lds_wait && scalar_wait && barrier && waited_barrier && trap &&
              dependency_delay && icache_inv && dcache_inv && dcache_wb && dcache_inv_vol &&
              buffer_inv && dcache_wb_vol);

  EXPECT_EQ(*flat_wait, 0xbf8c0070u);
  EXPECT_EQ(*lds_wait, 0xbf8cc07fu);
  EXPECT_EQ(*scalar_wait, *lds_wait);
  EXPECT_EQ(*barrier, 0xbf8a0000u);
  EXPECT_EQ(*waited_barrier, (std::array<uint32_t, 2>{0xbf8c0070u, 0xbf8a0000u}));
  EXPECT_EQ(*trap, 0xbf920002u);
  EXPECT_EQ(*dependency_delay, 0xbf800000u);
  EXPECT_EQ(*icache_inv, 0xbf930000u);
  EXPECT_EQ(*dcache_inv, (std::array<uint32_t, 2>{0xc0800000u, 0x00000000u}));
  EXPECT_EQ(*dcache_wb, (std::array<uint32_t, 2>{0xc0840000u, 0x00000000u}));
  EXPECT_EQ(*dcache_inv_vol, (std::array<uint32_t, 2>{0xc0880000u, 0x00000000u}));
  EXPECT_EQ(*buffer_inv, (std::array<uint32_t, 2>{0xe0a48000u, 0x00000000u}));
  EXPECT_EQ(*dcache_wb_vol, (std::array<uint32_t, 2>{0xc08c0000u, 0x00000000u}));

  auto decoder = Decoder::create(kArch);
  ASSERT_NE(decoder, nullptr);
  const auto expect_decode = [&](const uint32_t *words, std::string_view mnemonic, uint32_t size) {
    std::unique_ptr<Instruction> inst(decoder->decode(words));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(std::string_view(inst->mnemonic()), mnemonic);
    EXPECT_EQ(inst->size(), size);
  };
  expect_decode(&*flat_wait, "s_waitcnt", 4u);
  expect_decode(&*lds_wait, "s_waitcnt", 4u);
  expect_decode(&*barrier, "s_barrier", 4u);
  expect_decode(&*trap, "s_trap", 4u);
  expect_decode(&*dependency_delay, "s_nop", 4u);
  expect_decode(&*icache_inv, "s_icache_inv", 4u);
  expect_decode(dcache_inv->data(), "s_dcache_inv", 8u);
  expect_decode(dcache_wb->data(), "s_dcache_wb", 8u);
  expect_decode(dcache_inv_vol->data(), "s_dcache_inv_vol", 8u);
  expect_decode(buffer_inv->data(), "buffer_inv", 8u);
  expect_decode(dcache_wb_vol->data(), "s_dcache_wb_vol", 8u);
}

TEST(Cdna4InstrumentationBuilder, WaitBarrierAndCacheControlRejectWrongArchitecture) {
  constexpr rj_code_arch_t kWrongArch = ROCJITSU_CODE_ARCH_RDNA4;
  EXPECT_FALSE(build_cdna4_s_waitcnt(0, kWrongArch));
  EXPECT_FALSE(build_cdna4_s_wait_flat0(kWrongArch));
  EXPECT_FALSE(build_cdna4_s_wait_lds0(kWrongArch));
  EXPECT_FALSE(build_cdna4_s_wait_scalar_load0(kWrongArch));
  EXPECT_FALSE(build_cdna4_s_barrier(kWrongArch));
  EXPECT_FALSE(build_cdna4_s_barrier_with_memory_wait(kWrongArch));
  EXPECT_FALSE(build_cdna4_s_trap(2, kWrongArch));
  EXPECT_FALSE(build_cdna4_salu_dependency_delay(kWrongArch));
  EXPECT_FALSE(build_cdna4_s_icache_inv(kWrongArch));
  EXPECT_FALSE(build_cdna4_s_dcache_inv(kWrongArch));
  EXPECT_FALSE(build_cdna4_s_dcache_wb(kWrongArch));
  EXPECT_FALSE(build_cdna4_s_dcache_inv_vol(kWrongArch));
  EXPECT_FALSE(build_cdna4_buffer_inv_sc1(kWrongArch));
  EXPECT_FALSE(build_cdna4_s_dcache_wb_vol(kWrongArch));
}

} // namespace
} // namespace rocjitsu
