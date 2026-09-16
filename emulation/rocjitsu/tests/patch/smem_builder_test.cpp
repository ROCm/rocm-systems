// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file smem_builder_test.cpp
/// @brief Encoding contract for the scalar-memory builders, across all ten
///        AMDGPU targets.
///
/// The load is pinned by decoding it back through the same generated decoder the
/// simulator uses, not against a hand-computed word. A literal expected word
/// would restate the fields this builder has to get right (the halved SBASE, and
/// the two shapes of "immediate offset, no SGPR offset") rather than check them.
///
/// Decoding cannot check RDNA's SOFFSET, because the model builds its offset
/// operand from the immediate alone and ignores that field. It is asserted
/// against the raw encoded word instead, per generation.

#include "rocjitsu/code/builders/smem_builders.h"

#include "rocjitsu/code/builders/spill_builders.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/register_set.h"

#include "../decode_test_util.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace rocjitsu {
namespace {

// How a target spells "immediate offset, no SGPR offset". CDNA1-4 clear
// SOFFSET_EN and set IMM; RDNA has neither bit and must name NULL in SOFFSET.
enum class SmemOffsetShape { CdnaEnableBits, RdnaSoffsetNull };

struct SmemArch {
  rj_code_arch_t arch;
  const char *name; ///< For failure messages; the enum prints as an int.
  const char *load_mnemonic;
  uint32_t max_byte_offset;
  SmemOffsetShape shape;
  std::optional<uint8_t> soffset_null; ///< Set only for RdnaSoffsetNull.
  const char *wait_mnemonic;
};

// All ten AMDGPU targets. The per-target constants are spelled out rather than
// derived from the builder, so a generated table changing underneath us fails
// here instead of the test silently agreeing with the new value.
constexpr std::array<SmemArch, 10> kArchs{{
    {ROCJITSU_CODE_ARCH_CDNA1, "cdna1", "s_load_dwordx2", 0x0FFFFFu,
     SmemOffsetShape::CdnaEnableBits, std::nullopt, "s_waitcnt"},
    {ROCJITSU_CODE_ARCH_CDNA2, "cdna2", "s_load_dwordx2", 0x0FFFFFu,
     SmemOffsetShape::CdnaEnableBits, std::nullopt, "s_waitcnt"},
    {ROCJITSU_CODE_ARCH_CDNA3, "cdna3", "s_load_dwordx2", 0x0FFFFFu,
     SmemOffsetShape::CdnaEnableBits, std::nullopt, "s_waitcnt"},
    {ROCJITSU_CODE_ARCH_CDNA4, "cdna4", "s_load_dwordx2", 0x0FFFFFu,
     SmemOffsetShape::CdnaEnableBits, std::nullopt, "s_waitcnt"},
    // NULL is 125 on RDNA1/2 and 124 from RDNA3 on. A dispatch that hardcoded
    // one value would encode a real SGPR as the offset on the other half.
    {ROCJITSU_CODE_ARCH_RDNA1, "rdna1", "s_load_dwordx2", 0x0FFFFFu,
     SmemOffsetShape::RdnaSoffsetNull, 125, "s_waitcnt"},
    {ROCJITSU_CODE_ARCH_RDNA2, "rdna2", "s_load_dwordx2", 0x0FFFFFu,
     SmemOffsetShape::RdnaSoffsetNull, 125, "s_waitcnt"},
    {ROCJITSU_CODE_ARCH_RDNA3, "rdna3", "s_load_b64", 0x0FFFFFu, SmemOffsetShape::RdnaSoffsetNull,
     124, "s_waitcnt"},
    {ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", "s_load_b64", 0x0FFFFFu,
     SmemOffsetShape::RdnaSoffsetNull, 124, "s_waitcnt"},
    // GFX12 widens the immediate to 24 bits and splits the wait counters.
    {ROCJITSU_CODE_ARCH_RDNA4, "rdna4", "s_load_b64", 0x7FFFFFu, SmemOffsetShape::RdnaSoffsetNull,
     124, "s_wait_kmcnt"},
    {ROCJITSU_CODE_ARCH_CDNA5, "cdna5", "s_load_b64", 0x7FFFFFu, SmemOffsetShape::RdnaSoffsetNull,
     124, "s_wait_kmcnt"},
}};

constexpr std::array<rj_code_arch_t, 2> kNonAmdGpuArchs{ROCJITSU_CODE_ARCH_RV32I,
                                                        ROCJITSU_CODE_ARCH_RV64I};

// Decode the instruction starting at `words[0]`, or null if it did not decode.
// Two words are always passed so an 8-byte encoding has its second word.
std::unique_ptr<Instruction> decode_one(const std::array<uint32_t, 2> &words, rj_code_arch_t arch) {
  auto decoder = Decoder::create(arch);
  if (!decoder)
    return nullptr;
  return std::unique_ptr<Instruction>(decode_valid(*decoder, words.data()));
}

// SOFFSET occupies bits [63:57], the second word's high 7 bits, on every
// generation whose encoding carries it.
constexpr uint8_t encoded_soffset(const std::array<uint32_t, 2> &words) {
  return static_cast<uint8_t>((words[1] >> 25) & 0x7Fu);
}

// The builder's job is the three fields the encoding does not take literally:
// the destination pair, the address pair (encoded halved), and the immediate.
// Decoding recovers all three on every target.
TEST(SmemBuilder, LoadDwordx2RoundTripsThroughTheDecoderOnEveryTarget) {
  constexpr uint16_t kSdst = 4;
  constexpr uint16_t kSbase = 6;
  constexpr uint32_t kOffset = 0x1010;

  for (const SmemArch &a : kArchs) {
    const auto inst = decode_one(build_s_load_dwordx2(kSdst, kSbase, kOffset, a.arch), a.arch);
    ASSERT_NE(inst, nullptr) << a.name;
    EXPECT_EQ(inst->mnemonic(), std::string_view(a.load_mnemonic)) << a.name;

    ASSERT_GE(inst->num_dst_operands(), 1) << a.name;
    const auto dst_ref = inst->dst_operand(0)->to_register_ref();
    ASSERT_TRUE(dst_ref.has_value()) << a.name;
    EXPECT_EQ(dst_ref->cls, RegClass::SGPR) << a.name;
    EXPECT_EQ(dst_ref->index, kSdst) << a.name;
    EXPECT_EQ(dst_ref->width, 2) << a.name;

    // SBASE is encoded halved and the decoder doubles it back, so an unhalved
    // builder would name s12 here.
    ASSERT_GE(inst->num_src_operands(), 1) << a.name;
    const auto base_ref = inst->src_operand(0)->to_register_ref();
    ASSERT_TRUE(base_ref.has_value()) << a.name;
    EXPECT_EQ(base_ref->cls, RegClass::SGPR) << a.name;
    EXPECT_EQ(base_ref->index, kSbase) << a.name;
    EXPECT_EQ(base_ref->width, 2) << a.name;

    // The immediate. Without this the builder can drop its only variable
    // argument and every other assertion here still holds. On CDNA it also
    // covers the IMM bit: with IMM clear the same 21 bits decode as an SGPR
    // selector taking `offset & 0x7F` (shared/addr_calc_scalar.h reads it that
    // way on the address path, not just in disassembly), so the value would come
    // back as 0x10 rather than 0x1010.
    ASSERT_GE(inst->num_src_operands(), 2) << a.name;
    const Operand *offset = inst->src_operand(1);
    ASSERT_NE(offset, nullptr) << a.name;
    EXPECT_EQ(offset->encoding_value(), static_cast<int>(kOffset)) << a.name;
    EXPECT_FALSE(offset->to_register_ref().has_value())
        << a.name << ": immediate offset decoded as a register";
  }
}

// RDNA carries no SOFFSET_EN, so an unset SOFFSET field names s0 and the load
// silently adds whatever s0 holds. The decoder cannot catch this, since it builds
// the offset operand from the immediate alone. Assert the encoded word instead,
// and the generation's own NULL code, which moved between RDNA2 and RDNA3.
TEST(SmemBuilder, RdnaNamesItsOwnNullCodeInSoffset) {
  bool saw_125 = false;
  bool saw_124 = false;
  for (const SmemArch &a : kArchs) {
    if (a.shape != SmemOffsetShape::RdnaSoffsetNull)
      continue;
    const auto words = build_s_load_dwordx2(/*sdst=*/4, /*sbase=*/6, /*byte_offset=*/0x40, a.arch);
    ASSERT_TRUE(a.soffset_null.has_value()) << a.name;
    EXPECT_EQ(encoded_soffset(words), *a.soffset_null) << a.name;
    EXPECT_NE(encoded_soffset(words), 0) << a.name << ": SOFFSET 0 would name s0";
    saw_125 |= *a.soffset_null == 125;
    saw_124 |= *a.soffset_null == 124;
  }
  // Both codes are actually exercised, so a shared constant cannot pass this.
  EXPECT_TRUE(saw_125);
  EXPECT_TRUE(saw_124);
}

// Both operands name a 64-bit pair, so an odd base has no encoding. It must be
// refused rather than silently truncated to the even pair below it, which would
// load through, or into, the wrong register.
TEST(SmemBuilder, OddPairBaseIsRejectedOnEveryTarget) {
  for (const SmemArch &a : kArchs) {
    EXPECT_THROW((void)build_s_load_dwordx2(/*sdst=*/4, /*sbase=*/7, /*byte_offset=*/0, a.arch),
                 util::InvalidInst)
        << a.name << ": odd sbase";
    EXPECT_THROW((void)build_s_load_dwordx2(/*sdst=*/5, /*sbase=*/6, /*byte_offset=*/0, a.arch),
                 util::InvalidInst)
        << a.name << ": odd sdst";
  }
}

// The encoded fields are narrower than the uint16_t parameters and the generated
// set_field masks rather than rejects, so an out-of-range index would encode a
// different register: sbase 128 halves to 64, which a 6-bit SBASE truncates to 0,
// silently loading through s[0:1].
TEST(SmemBuilder, OutOfRangeRegisterIndexIsRejectedOnEveryTarget) {
  for (const SmemArch &a : kArchs) {
    EXPECT_NO_THROW(
        (void)build_s_load_dwordx2(kMaxSmemSdata - 1, kMaxSmemSbase, /*byte_offset=*/0, a.arch))
        << a.name;
    EXPECT_THROW(
        (void)build_s_load_dwordx2(/*sdst=*/4, kMaxSmemSbase + 2, /*byte_offset=*/0, a.arch),
        util::InvalidInst)
        << a.name << ": sbase past SBASE";
    EXPECT_THROW(
        (void)build_s_load_dwordx2(kMaxSmemSdata + 1, /*sbase=*/6, /*byte_offset=*/0, a.arch),
        util::InvalidInst)
        << a.name << ": sdst past SDATA";
  }
}

// Two immediate widths, both read as signed, so the bound is the signed-positive
// maximum: one bit below the field width. Pinning the exact values keeps a later
// "the field is 21 bits wide" simplification from doubling the accepted range.
TEST(SmemBuilder, OffsetIsBoundedByTheArchSignedImmediateRange) {
  for (const SmemArch &a : kArchs) {
    EXPECT_EQ(max_smem_byte_offset(a.arch), a.max_byte_offset) << a.name;
    EXPECT_NO_THROW((void)build_s_load_dwordx2(/*sdst=*/4, /*sbase=*/0, a.max_byte_offset, a.arch))
        << a.name;
    EXPECT_THROW((void)build_s_load_dwordx2(/*sdst=*/4, /*sbase=*/0, a.max_byte_offset + 1, a.arch),
                 util::InvalidInst)
        << a.name;

    // The offset an unsigned reading of the field would have admitted is
    // refused. Nothing in simulation distinguishes these, since the decoder
    // zero-extends, so this bound is the only guard against an offset that runs
    // backwards on hardware.
    const uint32_t unsigned_max = (a.max_byte_offset << 1) | 1u;
    EXPECT_THROW((void)build_s_load_dwordx2(/*sdst=*/4, /*sbase=*/0, unsigned_max, a.arch),
                 util::InvalidInst)
        << a.name;
  }

  // GFX12's field is wider, so an offset legal there is not legal earlier.
  EXPECT_THROW((void)build_s_load_dwordx2(4, 0, 0x100000u, ROCJITSU_CODE_ARCH_CDNA4),
               util::InvalidInst);
  EXPECT_NO_THROW((void)build_s_load_dwordx2(4, 0, 0x100000u, ROCJITSU_CODE_ARCH_RDNA4));
}

// The scalar-load wait is its own counter. On GFX12 it must be KMCNT: the
// LOADCNT wait in spill_builders.h orders VMEM and would leave a scalar load in
// flight. CDNA5 has no s_waitcnt opcode at all.
TEST(SmemBuilder, ScalarLoadWaitSelectsTheScalarCounter) {
  for (const SmemArch &a : kArchs) {
    const std::array<uint32_t, 2> wait{build_wait_scalar_loads_complete(a.arch), 0};
    const auto inst = decode_one(wait, a.arch);
    ASSERT_NE(inst, nullptr) << a.name;
    EXPECT_EQ(inst->mnemonic(), std::string_view(a.wait_mnemonic)) << a.name;
  }

  // Through RDNA3.5 the monolithic wait covers both counters; GFX12 splits them,
  // and only there is the distinction observable in the encoding.
  EXPECT_EQ(build_wait_scalar_loads_complete(ROCJITSU_CODE_ARCH_CDNA3),
            build_wait_loads_complete(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_NE(build_wait_scalar_loads_complete(ROCJITSU_CODE_ARCH_RDNA4),
            build_wait_loads_complete(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(SmemBuilder, NonAmdGpuArchsThrow) {
  for (const rj_code_arch_t arch : kNonAmdGpuArchs) {
    EXPECT_THROW((void)build_s_load_dwordx2(/*sdst=*/4, /*sbase=*/0, /*byte_offset=*/0, arch),
                 util::UnimplementedInst);
    EXPECT_THROW((void)build_wait_scalar_loads_complete(arch), util::UnimplementedInst);
    EXPECT_THROW((void)max_smem_byte_offset(arch), util::UnimplementedInst);
  }
}

} // namespace
} // namespace rocjitsu
