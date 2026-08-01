// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/rdna3_instrumentation_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string_view>

namespace rocjitsu {
namespace {

TEST(Rdna3InstrumentationBuilder, ScalarVectorAndWaitEncodingsMatchLlvm) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA3;
  EXPECT_EQ(build_rdna3_s_getreg_b32(20, 0xf817u, kArch), 0xb894f817u);
  EXPECT_EQ(build_rdna3_s_mov_b64(20, 22, kArch), 0xbe940116u);
  EXPECT_EQ(build_rdna3_s_and_saveexec_b64(20, 22, kArch), 0xbe942116u);
  EXPECT_EQ(build_rdna3_v_add_u32(6, vector_source_vgpr(7), 8, kArch), 0x4a0c1107u);
  EXPECT_EQ(build_rdna3_s_wait_vmcnt0(kArch), 0xbf8903f7u);
  EXPECT_EQ(build_rdna3_s_wait_lgkmcnt0(kArch), 0xbf89fc07u);
  EXPECT_EQ(build_rdna3_s_wait_vmcnt_lgkmcnt0(kArch), 0xbf890007u);
  EXPECT_EQ(build_rdna3_s_barrier(kArch), 0xbfbd0000u);
  EXPECT_EQ(build_rdna3_s_trap(2, kArch), 0xbf900002u);
}

TEST(Rdna3InstrumentationBuilder, MemoryAndAtomicEncodingsMatchLlvmAndDecoder) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA3;
  const auto scalar_load = build_rdna3_s_load_dword(20, 0, 0xffffcu, kArch);
  const auto flat_store = build_rdna3_flat_store_b32(4, 6, 0xfffu, kArch);
  const auto flat_load = build_rdna3_flat_load_b32(4, 7, 0xfffu, kArch);
  const auto scratch_store = build_rdna3_address_free_scratch_store_b32(7, 0xffcu, kArch);
  const auto scratch_load = build_rdna3_address_free_scratch_load_b32(7, 0xffcu, kArch);
  const auto ds_store = build_rdna3_ds_store_b32(2, 3, 4, kArch);
  const auto ds_store64 = build_rdna3_ds_store_b64(2, 6, 4, kArch);
  const auto ds_store128 = build_rdna3_ds_store_b128(2, 8, 4, kArch);
  const auto ds_load = build_rdna3_ds_load_b32(4, 2, 4, kArch);
  const auto add = build_rdna3_flat_atomic_add_u32(2, 4, 5, true, 2, kArch);
  const auto bit_or = build_rdna3_flat_atomic_or_u32(2, 4, 5, true, 2, kArch);
  const auto cmp_swap = build_rdna3_flat_atomic_cmpswap_b32(2, 4, 5, true, 2, kArch);
  const auto cmp_swap64 = build_rdna3_flat_atomic_cmpswap_b64(2, 8, 6, true, 2, kArch);
  const auto swap64 = build_rdna3_flat_atomic_swap_b64(2, 8, 6, true, 2, kArch);
  const auto add64 = build_rdna3_flat_atomic_add_u64(2, 8, 6, true, 2, kArch);
  ASSERT_TRUE(scalar_load && flat_store && flat_load && scratch_store && scratch_load && ds_store &&
              ds_store64 && ds_store128 && ds_load && add && bit_or && cmp_swap && cmp_swap64 &&
              swap64 && add64);

  EXPECT_EQ(*scalar_load, (std::array<uint32_t, 2>{0xf4000500u, 0xf80ffffcu}));
  EXPECT_EQ(*flat_store, (std::array<uint32_t, 2>{0xdc680fffu, 0x007c0604u}));
  EXPECT_EQ(*flat_load, (std::array<uint32_t, 2>{0xdc500fffu, 0x077c0004u}));
  EXPECT_EQ(*scratch_store, (std::array<uint32_t, 2>{0xdc690ffcu, 0x007c0700u}));
  EXPECT_EQ(*scratch_load, (std::array<uint32_t, 2>{0xdc510ffcu, 0x077c0000u}));
  EXPECT_EQ(*ds_store, (std::array<uint32_t, 2>{0xd8340004u, 0x00000302u}));
  EXPECT_EQ(*ds_store64, (std::array<uint32_t, 2>{0xd9340004u, 0x00000602u}));
  EXPECT_EQ(*ds_store128, (std::array<uint32_t, 2>{0xdb7c0004u, 0x00000802u}));
  EXPECT_EQ(*ds_load, (std::array<uint32_t, 2>{0xd8d80004u, 0x04000002u}));
  EXPECT_EQ(*add, (std::array<uint32_t, 2>{0xdcd44000u, 0x057c0402u}));
  EXPECT_EQ(*bit_or, (std::array<uint32_t, 2>{0xdcf44000u, 0x057c0402u}));
  EXPECT_EQ(*cmp_swap, (std::array<uint32_t, 2>{0xdcd04000u, 0x057c0402u}));
  EXPECT_EQ(*cmp_swap64, (std::array<uint32_t, 2>{0xdd084000u, 0x067c0802u}));
  EXPECT_EQ(*swap64, (std::array<uint32_t, 2>{0xdd044000u, 0x067c0802u}));
  EXPECT_EQ(*add64, (std::array<uint32_t, 2>{0xdd0c4000u, 0x067c0802u}));

  auto decoder = Decoder::create(kArch);
  ASSERT_NE(decoder, nullptr);
  for (const auto &[words, mnemonic] :
       std::array<std::pair<const uint32_t *, std::string_view>, 10>{{
           {flat_store->data(), "flat_store_b32"},
           {flat_load->data(), "flat_load_b32"},
           {ds_store->data(), "ds_store_b32"},
           {ds_store64->data(), "ds_store_b64"},
           {ds_store128->data(), "ds_store_b128"},
           {ds_load->data(), "ds_load_b32"},
           {add->data(), "flat_atomic_add_u32"},
           {bit_or->data(), "flat_atomic_or_b32"},
           {cmp_swap->data(), "flat_atomic_cmpswap_b32"},
           {add64->data(), "flat_atomic_add_u64"},
       }}) {
    std::unique_ptr<Instruction> instruction(decoder->decode(words));
    ASSERT_NE(instruction, nullptr);
    EXPECT_EQ(std::string_view(instruction->mnemonic()), mnemonic);
  }
}

TEST(Rdna3InstrumentationBuilder, RejectsWrongArchitectureAndOutOfRangeOperands) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA3;
  EXPECT_FALSE(build_rdna3_s_mov_b64(20, 22, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_rdna3_s_load_dword(20, 0, 0x100000u, kArch));
  EXPECT_FALSE(build_rdna3_s_load_dword(20, 0, 2u, kArch));
  EXPECT_FALSE(build_rdna3_flat_store_b32(4, 6, 0x1000u, kArch));
  EXPECT_FALSE(build_rdna3_flat_load_b32(4, 7, 0x1000u, kArch));
  EXPECT_FALSE(build_rdna3_address_free_scratch_store_b32(7, 0x1000u, kArch));
  EXPECT_FALSE(build_rdna3_address_free_scratch_load_b32(7, 0x1000u, kArch));
  EXPECT_FALSE(build_rdna3_address_free_scratch_store_b32(7, 2u, kArch));
  EXPECT_FALSE(build_rdna3_flat_atomic_add_u32(255, 4, 5, true, 2, kArch));
  EXPECT_FALSE(build_rdna3_flat_atomic_add_u32(2, 4, 5, true, 1, kArch));
  EXPECT_FALSE(build_rdna3_flat_atomic_cmpswap_b32(2, 255, 5, true, 2, kArch));
  EXPECT_FALSE(build_rdna3_flat_atomic_cmpswap_b32(2, 4, 5, false, 2, kArch));
  EXPECT_FALSE(build_rdna3_ds_store_b64(2, 255, 4, kArch));
  EXPECT_FALSE(build_rdna3_ds_store_b128(2, 253, 4, kArch));
}

} // namespace
} // namespace rocjitsu
