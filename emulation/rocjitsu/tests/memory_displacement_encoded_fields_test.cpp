// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/memory_displacement_encoded_fields.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace rocjitsu {
namespace {

constexpr std::array<rj_code_target_id_t, 4> kTargets = {
    ROCJITSU_CODE_TARGET_GFX1250, ROCJITSU_CODE_TARGET_GFX1251, ROCJITSU_CODE_TARGET_GFX1201,
    ROCJITSU_CODE_TARGET_GFX1200};

struct MemoryCase {
  std::array<uint32_t, 3> words;
  std::string_view mnemonic;
};

std::array<MemoryCase, 4> memory_cases(rj_code_target_id_t target) {
  if (target == ROCJITSU_CODE_TARGET_GFX1250 || target == ROCJITSU_CODE_TARGET_GFX1251) {
    return {{
        {.words = cdna5::build_vflat(cdna5::kFlatAtomicAddU32Vflat,
                                     {.saddr = amdgpu::kModernNullSelector,
                                      .vdst = 5,
                                      .scope = 2,
                                      .th = 1,
                                      .vsrc = 3,
                                      .vaddr = 2,
                                      .ioffset = 0xfffffc}),
         .mnemonic = "flat_atomic_add_u32"},
        {.words = cdna5::build_vglobal(cdna5::kGlobalAtomicAddU32Vglobal,
                                       {.saddr = amdgpu::kModernNullSelector,
                                        .vdst = 5,
                                        .scope = 2,
                                        .th = 1,
                                        .vsrc = 3,
                                        .vaddr = 2,
                                        .ioffset = 0xfffffc}),
         .mnemonic = "global_atomic_add_u32"},
        {.words = cdna5::build_vbuffer(cdna5::kBufferAtomicAddU32Vbuffer, {.vdata = 5,
                                                                           .rsrc = 4,
                                                                           .scope = 2,
                                                                           .th = 1,
                                                                           .offen = 1,
                                                                           .vaddr = 2,
                                                                           .ioffset = 0xfffffc}),
         .mnemonic = "buffer_atomic_add_u32"},
        {.words = cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                                       {.saddr = amdgpu::kModernNullSelector,
                                        .vdst = 5,
                                        .scope = 2,
                                        .vaddr = 2,
                                        .ioffset = 0xfffffc}),
         .mnemonic = "global_load_b32"},
    }};
  }
  return {{
      {.words =
           rdna4::build_vflat(rdna4::kFlatAtomicAddU32Vflat, {.saddr = amdgpu::kModernNullSelector,
                                                              .vdst = 5,
                                                              .scope = 2,
                                                              .th = 1,
                                                              .vsrc = 3,
                                                              .vaddr = 2,
                                                              .ioffset = 0xfffffc}),
       .mnemonic = "flat_atomic_add_u32"},
      {.words = rdna4::build_vglobal(rdna4::kGlobalAtomicAddU32Vglobal,
                                     {.saddr = amdgpu::kModernNullSelector,
                                      .vdst = 5,
                                      .scope = 2,
                                      .th = 1,
                                      .vsrc = 3,
                                      .vaddr = 2,
                                      .ioffset = 0xfffffc}),
       .mnemonic = "global_atomic_add_u32"},
      {.words = rdna4::build_vbuffer(rdna4::kBufferAtomicAddU32Vbuffer, {.vdata = 5,
                                                                         .rsrc = 4,
                                                                         .scope = 2,
                                                                         .th = 1,
                                                                         .offen = 1,
                                                                         .vaddr = 2,
                                                                         .ioffset = 0xfffffc}),
       .mnemonic = "buffer_atomic_add_u32"},
      {.words =
           rdna4::build_vglobal(rdna4::kGlobalLoadB32Vglobal, {.saddr = amdgpu::kModernNullSelector,
                                                               .vdst = 5,
                                                               .scope = 2,
                                                               .vaddr = 2,
                                                               .ioffset = 0xfffffc}),
       .mnemonic = "global_load_b32"},
  }};
}

bool only_offset_changed(const std::array<uint32_t, 3> &source,
                         const std::array<uint32_t, 3> &replacement) {
  // Every low byte of word two is an address operand, outside the immediate.
  return source[0] == replacement[0] && source[1] == replacement[1] &&
         (source[2] & 255u) == (replacement[2] & 255u) && replacement[2] / 256u == 4u;
}

TEST(MemoryDisplacementEncodedFields, PreservesAndDecodesEachConcreteTargetAndForm) {
  for (rj_code_target_id_t target : kTargets) {
    SCOPED_TRACE(static_cast<int>(target));
    for (const MemoryCase &fixture : memory_cases(target)) {
      SCOPED_TRACE(fixture.mnemonic);
      const std::array<uint32_t, 3> pristine = fixture.words;
      EXPECT_EQ(amdgpu::read_vmem_signed24_offset(target, pristine), -4);
      const std::optional<std::array<uint32_t, 3>> replacement =
          amdgpu::replace_vmem_signed24_offset(target, pristine, 4);
      ASSERT_TRUE(replacement);
      EXPECT_EQ(fixture.words, pristine);
      EXPECT_TRUE(only_offset_changed(pristine, *replacement));
      EXPECT_EQ((*replacement)[2], 0x00000402u);

      std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
      ASSERT_TRUE(decoder);
      DecodeResult before = decoder->decode_window(pristine);
      DecodeResult after = decoder->decode_window(*replacement);
      ASSERT_TRUE(before.succeeded());
      ASSERT_TRUE(after.succeeded());
      EXPECT_EQ(after.value()->size(), 12);
      EXPECT_EQ(after.value()->mnemonic(), fixture.mnemonic);
      ASSERT_EQ(after.value()->num_src_operands(), before.value()->num_src_operands());
      ASSERT_EQ(after.value()->num_dst_operands(), before.value()->num_dst_operands());
      for (int index = 0; index < before.value()->num_src_operands(); ++index)
        EXPECT_EQ(after.value()->src_operand(index)->name(),
                  before.value()->src_operand(index)->name());
      for (int index = 0; index < before.value()->num_dst_operands(); ++index)
        EXPECT_EQ(after.value()->dst_operand(index)->name(),
                  before.value()->dst_operand(index)->name());

      for (size_t word = 0; word < 3; ++word) {
        std::array<uint32_t, 3> corrupted = *replacement;
        corrupted[word] ^= 1u;
        EXPECT_FALSE(only_offset_changed(pristine, corrupted));
      }
    }
  }
}

TEST(MemoryDisplacementEncodedFields, SignedBoundariesNoOpAndRawUnalignedValues) {
  struct OffsetCase {
    int32_t value;
    uint32_t expected_word;
  };
  constexpr std::array<OffsetCase, 7> offsets = {{
      {.value = -0x800000, .expected_word = 0x800000a5},
      {.value = -4, .expected_word = 0xfffffca5},
      {.value = -1, .expected_word = 0xffffffa5},
      {.value = 0, .expected_word = 0x000000a5},
      {.value = 1, .expected_word = 0x000001a5},
      {.value = 4, .expected_word = 0x000004a5},
      {.value = 0x7fffff, .expected_word = 0x7fffffa5},
  }};
  for (rj_code_target_id_t target : kTargets) {
    for (uint32_t prefix : {0xec000000u, 0xee000000u, 0xc4000000u}) {
      // Raw field edits preserve reserved and unused bits as well as operands.
      const std::array<uint32_t, 3> source = {prefix | 0x00b5a53cu, 0xdeadbeefu, 0xfffffca5u};
      for (const OffsetCase &offset : offsets) {
        SCOPED_TRACE(offset.value);
        const std::optional<std::array<uint32_t, 3>> replacement =
            amdgpu::replace_vmem_signed24_offset(target, source, offset.value);
        ASSERT_TRUE(replacement);
        const std::array<uint32_t, 3> expected = {source[0], source[1], offset.expected_word};
        EXPECT_EQ(*replacement, expected);
        EXPECT_EQ(amdgpu::read_vmem_signed24_offset(target, expected), offset.value);
      }
    }
  }
}

TEST(MemoryDisplacementEncodedFields, Cdna5VectorScalingDoesNotScaleImmediate) {
  for (rj_code_target_id_t target : {ROCJITSU_CODE_TARGET_GFX1250, ROCJITSU_CODE_TARGET_GFX1251}) {
    for (uint8_t scale : {0, 1}) {
      for (uint32_t prefix : {0xec000000u, 0xee000000u}) {
        const std::array<uint32_t, 3> source = {
            prefix | 0x00154004u, 0x01980005u | (static_cast<uint32_t>(scale) << 16), 0xfffffc02u};
        const std::optional<std::array<uint32_t, 3>> replacement =
            amdgpu::replace_vmem_signed24_offset(target, source, 8);
        ASSERT_TRUE(replacement);
        const std::array<uint32_t, 3> expected = {source[0], source[1], 0x00000802u};
        EXPECT_EQ(*replacement, expected);
        EXPECT_EQ(((*replacement)[1] >> 16) & 1u, scale);
        EXPECT_EQ(amdgpu::read_vmem_signed24_offset(target, *replacement), 8);
      }
    }
  }
}

TEST(MemoryDisplacementEncodedFields, RejectsUnsupportedTargetFormatExtentAndRange) {
  constexpr std::array<uint32_t, 3> source = {0xee15407fu, 0x00980005u, 0xfffffc02u};
  for (rj_code_target_id_t target :
       {ROCJITSU_CODE_TARGET_GFX1100, ROCJITSU_CODE_TARGET_GFX942, ROCJITSU_CODE_TARGET_GFX950,
        static_cast<rj_code_target_id_t>(-1)}) {
    EXPECT_FALSE(amdgpu::read_vmem_signed24_offset(target, source));
    EXPECT_FALSE(amdgpu::replace_vmem_signed24_offset(target, source, 4));
  }
  for (rj_code_target_id_t target : kTargets) {
    for (size_t size = 0; size < 3; ++size) {
      EXPECT_FALSE(amdgpu::read_vmem_signed24_offset(target, std::span(source).first(size)));
      EXPECT_FALSE(amdgpu::replace_vmem_signed24_offset(target, std::span(source).first(size), 4));
    }
    constexpr std::array<uint32_t, 4> oversized = {source[0], source[1], source[2], 0};
    EXPECT_FALSE(amdgpu::read_vmem_signed24_offset(target, oversized));
    EXPECT_FALSE(amdgpu::replace_vmem_signed24_offset(target, oversized, 4));
    for (uint32_t prefix : {0xed000000u, 0xd0000000u, 0xd8000000u, 0x00000000u}) {
      const std::array<uint32_t, 3> other = {prefix, source[1], source[2]};
      EXPECT_FALSE(amdgpu::read_vmem_signed24_offset(target, other));
      EXPECT_FALSE(amdgpu::replace_vmem_signed24_offset(target, other, 4));
    }
    for (int64_t value : {int64_t{-0x800001}, int64_t{0x800000},
                          std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()})
      EXPECT_FALSE(amdgpu::replace_vmem_signed24_offset(target, source, value));
    constexpr std::array<uint32_t, 3> expected = {0xee15407fu, 0x00980005u, 0xfffffc02u};
    EXPECT_EQ(source, expected);
  }
}

} // namespace
} // namespace rocjitsu
