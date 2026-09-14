// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3p_pk_inline_constant_correctness_test.cpp
/// @brief Value-correctness check for VOP3P packed-16 sources fed by an inline
/// FLOAT constant (encoding values 240..248).
///
/// Packed F16 narrows inline float constants into the low half: inline 1.0
/// becomes 0x00003C00, so both results select that half with OPSEL. Packed BF16
/// preserves the FP32 pattern, 0x3F800000, and selects the upper half instead.
/// Treating BF16 like F16 makes correctly encoded instructions consume zero.
///
/// These tests assert absolute results, including integer and literal controls.
/// Literal values must remain verbatim even when their bits resemble an inline
/// source selector; narrowing is keyed on the selector, not the operand value.
///
/// Packed BF16 cases run on CDNA5 (gfx1250). F16 and integer controls run on
/// RDNA4, with additional CDNA coverage for 16-bit source operands. DOT cases
/// exercise their separate inline handling on RDNA3 and RDNA4.

#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/execute_shared.h"
#include "rocjitsu/isa/decode_result.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "util/simd.h"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 32; // RDNA4 / CDNA5 wave32
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t kSrcVgpr = 0; // src0 = v0, encoding value 256
constexpr uint32_t kDstVgpr = 4;
constexpr uint32_t DST_SENTINEL = 0xCDCDCDCDu;
constexpr uint32_t kExec = 0xA5A58001u;

// Source-field encodings: inline 1.0, inline integer 1, and the 32-bit literal
// escape.
constexpr uint16_t kInlineOneFloat = 242;
constexpr uint16_t kInlineInv2Pi = 248; // the last float selector
constexpr uint16_t kInlineZero = 128;
constexpr uint16_t kInlineOneInt = 129;
constexpr uint16_t kLiteral = 255;

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  explicit Fixture(rj_code_arch_t arch)
      : gpu_mem("vop3p_pk_inline_const_mem"), l2("vop3p_pk_inline_const_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3p_pk_inline_const", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(arch);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  // Seeds every lane of v0 through v2, runs the encoding, and
  // returns the per-lane destination.
  std::array<uint32_t, WF_SIZE> run(const uint32_t *words, uint32_t src0_value,
                                    uint32_t src1_value = 0, uint32_t src2_value = 0) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      cu->write_vgpr(vb + kSrcVgpr, lane, src0_value);
      cu->write_vgpr(vb + 1, lane, src1_value);
      cu->write_vgpr(vb + 2, lane, src2_value);
      cu->write_vgpr(vb + kDstVgpr, lane, DST_SENTINEL);
    }
    wf->set_exec(kExec);
    DecodeResult decoded = decoder->decode(words);
    EXPECT_TRUE(decoded.succeeded()) << "decode failed";
    if (decoded.failed())
      return {};
    std::unique_ptr<Instruction> inst = std::move(decoded).value();
    cu->execute_instruction(inst.get(), *wf);
    std::array<uint32_t, WF_SIZE> out{};
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vb + kDstVgpr, lane);
    return out;
  }
};

// Restores the process force-scalar gate on scope exit.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

void expect_all_active(const std::array<uint32_t, WF_SIZE> &out, uint32_t want, const char *label) {
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (kExec >> lane) & 1u;
    if (active)
      EXPECT_EQ(out[lane], want) << label << ": lane " << lane;
    else
      EXPECT_EQ(out[lane], DST_SENTINEL) << label << ": clobbered inactive lane " << lane;
  }
}

// v_pk_add_f16 v4, v0, 1.0 op_sel_hi:[1,0] with v0 = {1.0h, 1.0h}. Both halves
// must see 1.0h, so the result is {2.0h, 2.0h}. Before the inline-constant fix
// the packed halves read the low 16 bits of 0x3F800000 and the add was a no-op
// copy of 0x3C003C00.
TEST(Vop3pPkInlineConstantCorrectness, AddF16InlineOneReachesBothHalves) {
  const auto words =
      rdna4::build_vop3p(rdna4::kVPkAddF16Vop3p,
                         {.vdst = kDstVgpr, .src0 = 256, .src1 = kInlineOneFloat, .opsel_hi = 1});
  Fixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  const auto out = fx.run(words.data(), 0x3C003C00u);
  expect_all_active(out, 0x40004000u, "v_pk_add_f16 v4, v0, 1.0 op_sel_hi:[1,0]");
}

// CDNA5 ISA section 7.7.2: BF16 selects the upper half of the FP32 inline
// constant, whereas F16 selects the low half of a narrowed F16 constant.
TEST(Vop3pPkInlineConstantCorrectness, AddBf16InlineOneReachesBothHalves) {
  const std::array<uint32_t, 2> words = cdna5::build_vop3p(
      cdna5::kVPkAddBf16Vop3p,
      {.vdst = kDstVgpr, .opsel = 2, .src0 = 256, .src1 = kInlineOneFloat, .opsel_hi = 3});
  Fixture fx(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(fx.wf, nullptr);
  const std::array<uint32_t, WF_SIZE> out = fx.run(words.data(), 0x3F803F80u);
  expect_all_active(out, 0x40004000u, "v_pk_add_bf16 v4, v0, 1.0 op_sel:[0,1]");
}

TEST(Vop3pPkInlineConstantCorrectness, FmaBf16InlineHalfSelectionForEachSource) {
  // Both settings currently use scalar execution; retain coverage for future SIMD bodies.
  ForceScalarGuard guard;
  for (bool force_scalar : {false, true}) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(fx.wf, nullptr);
    for (uint32_t source = 0; source < 3; ++source) {
      for (uint8_t low_select : {0, 1}) {
        for (uint8_t high_select : {0, 1}) {
          SCOPED_TRACE(::testing::Message()
                       << "scalar=" << force_scalar << " source=" << source
                       << " low=" << unsigned(low_select) << " high=" << unsigned(high_select));
          std::array<uint16_t, 3> sources = {256, 257, 258};
          sources[source] = kInlineOneFloat;
          const uint8_t high_mask = (7u & ~(1u << source)) | (high_select << source);
          const std::array<uint32_t, 2> words = cdna5::build_vop3p(
              cdna5::kVPkFmaBf16Vop3p, {.vdst = kDstVgpr,
                                        .opsel = static_cast<uint8_t>(low_select << source),
                                        .opsel_hi_2 = static_cast<uint8_t>(high_mask >> 2),
                                        .src0 = sources[0],
                                        .src1 = sources[1],
                                        .src2 = sources[2],
                                        .opsel_hi = static_cast<uint8_t>(high_mask & 3)});
          // For v0=2, v1=3, v2=4, substituting 1 gives 7, 6, 7;
          // substituting the zero low half gives 4, 4, 6.
          constexpr std::array<uint16_t, 3> with_one = {0x40E0, 0x40C0, 0x40E0};
          constexpr std::array<uint16_t, 3> with_zero = {0x4080, 0x4080, 0x40C0};
          const uint32_t expected_low = low_select ? with_one[source] : with_zero[source];
          const uint32_t expected_high = high_select ? with_one[source] : with_zero[source];
          const std::array<uint32_t, WF_SIZE> out =
              fx.run(words.data(), 0x40004000, 0x40404040, 0x40804080);
          expect_all_active(out, expected_low | (expected_high << 16), "packed BF16 FMA inline");
        }
      }
    }
  }
}

TEST(Vop3pPkInlineConstantCorrectness, Bf16BinaryInlineHalfSelection) {
  ForceScalarGuard guard;
  struct Case {
    uint16_t opcode;
    uint32_t other;
    uint16_t with_one;
    uint16_t with_zero;
  };
  constexpr std::array<Case, 4> cases = {{
      {cdna5::kVPkAddBf16Vop3p, 0x40004000, 0x4040, 0x4000},
      {cdna5::kVPkMulBf16Vop3p, 0x40004000, 0x4000, 0x0000},
      {cdna5::kVPkMinNumBf16Vop3p, 0x40004000, 0x3F80, 0x0000},
      {cdna5::kVPkMaxNumBf16Vop3p, 0xC000C000, 0x3F80, 0x0000},
  }};
  for (bool force_scalar : {false, true}) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(fx.wf, nullptr);
    for (const Case &test : cases) {
      for (uint32_t source = 0; source < 2; ++source) {
        for (uint8_t low_select : {0, 1}) {
          for (uint8_t high_select : {0, 1}) {
            SCOPED_TRACE(::testing::Message()
                         << "opcode=" << test.opcode << " source=" << source
                         << " low=" << unsigned(low_select) << " high=" << unsigned(high_select));
            const std::array<uint32_t, 2> words = cdna5::build_vop3p(
                test.opcode, {.vdst = kDstVgpr,
                              .opsel = static_cast<uint8_t>(low_select << source),
                              .src0 = source == 0 ? kInlineOneFloat : uint16_t{256},
                              .src1 = source == 1 ? kInlineOneFloat : uint16_t{256},
                              .opsel_hi = static_cast<uint8_t>((3u & ~(1u << source)) |
                                                               (high_select << source))});
            const uint32_t expected_low = low_select ? test.with_one : test.with_zero;
            const uint32_t expected_high = high_select ? test.with_one : test.with_zero;
            const std::array<uint32_t, WF_SIZE> out = fx.run(words.data(), test.other);
            expect_all_active(out, expected_low | (expected_high << 16),
                              "packed BF16 binary inline");
          }
        }
      }
    }
  }
}

TEST(Vop3pPkInlineConstantCorrectness, FmaBf16RegisterAndLiteralPreserveBothHalves) {
  Fixture fx(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(fx.wf, nullptr);
  for (uint8_t select : {0, 1}) {
    for (uint16_t source : {uint16_t{258}, kLiteral}) {
      const std::array<uint32_t, 2> base =
          cdna5::build_vop3p(cdna5::kVPkFmaBf16Vop3p, {.vdst = kDstVgpr,
                                                       .opsel = static_cast<uint8_t>(select << 2),
                                                       .opsel_hi_2 = select,
                                                       .src0 = 256,
                                                       .src1 = 257,
                                                       .src2 = source,
                                                       .opsel_hi = 3});
      const std::array<uint32_t, 3> words = {base[0], base[1], 0x3F804000};
      const std::array<uint32_t, WF_SIZE> out =
          fx.run(words.data(), 0x40004000, 0x40404040, 0x3F804000);
      expect_all_active(out, select ? 0x40E040E0 : 0x41004100, "packed BF16 register/literal");
    }
  }
}

// A source declared 16 bits wide is a different case: CDNA2 builds every packed
// f16 source that way (and CDNA3 does for v_pk_min_f16 / v_pk_max_f16), so
// Operand::read_lane has already resolved the inline constant through the
// half-precision table and hands back 0x00003C00. Re-narrowing THAT would read
// the 16-bit pattern as an f32 denormal and flush it to zero, so the rewrite has
// to decline it -- pk16_src_needs_narrowing checks the operand width for exactly
// this reason. Same shape and same answer as the RDNA4 case above.
TEST(Vop3pPkInlineConstantCorrectness, AddF16InlineOneOnSixteenBitSourceOperand) {
  const auto words =
      cdna2::build_vop3p(cdna2::kVPkAddF16Vop3p,
                         {.vdst = kDstVgpr, .src0 = 256, .src1 = kInlineOneFloat, .op_sel_hi = 1});
  Fixture fx(ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  const auto out = fx.run(words.data(), 0x3C003C00u);
  expect_all_active(out, 0x40004000u, "cdna2 v_pk_add_f16 v4, v0, 1.0 op_sel_hi:[1,0]");
}

// v_dot2_f16_f16 is VOP3, not VOP3P, but src0/src1 are still 32-bit operands
// consumed as packed v2 halves -- only src2 goes through read_vop3_true16_src,
// which is already 16-bit -- so it needs the same narrowing.
//
// v_dot2_f16_f16 v4, v0, 1.0, 0 with v0 = {1.0h, 0} is 1.0*1.0 + 0*0 + 0 = 1.0h.
// Unnarrowed, src1 reads {lo = 0, hi = 0x3F80} and the answer is 0. The
// destination is a true16 half, so the low half of v4 is written and the high
// half keeps its sentinel.
TEST(Vop3pPkInlineConstantCorrectness, Dot2F16F16InlineOneNarrowsPackedHalves) {
  const auto words = rdna4::build_vop3(
      rdna4::kVDot2F16F16Vop3,
      {.vdst = kDstVgpr, .src0 = 256, .src1 = kInlineOneFloat, .src2 = kInlineZero});
  Fixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  const auto out = fx.run(words.data(), 0x00003C00u);
  expect_all_active(out, (DST_SENTINEL & 0xFFFF0000u) | 0x3C00u, "v_dot2_f16_f16 v4, v0, 1.0, 0");
}

// Control: a 32-bit literal whose VALUE is an inline-constant encoding value.
// IsaOperand stores a literal's value in encoding_value_, so a rewrite keyed on
// the operand rather than on the source-selector field would narrow 0xF2 here.
//
// The addend has to be chosen so the two behaviours differ. Narrowing gives
// f32_to_f16(bit_cast<float>(0xF2)) = 0 -- 0xF2 is an f32 denormal far below
// the f16 range -- so any seed for which adding 0x00F2 is a no-op passes
// either way. 0x0800 is 2^-13, where the f16 step is 2^-23 and the literal's
// 242 * 2^-24 lands exactly on 0x0879. Keyed correctly the result is
// 0x08790879; keyed on the operand it would stay 0x08000800.
TEST(Vop3pPkInlineConstantCorrectness, AddF16LiteralValuedLikeInlineConstantIsVerbatim) {
  const auto base = rdna4::build_vop3p(
      rdna4::kVPkAddF16Vop3p, {.vdst = kDstVgpr, .src0 = 256, .src1 = kLiteral, .opsel_hi = 1});
  const std::array words{base[0], base[1], uint32_t{kInlineOneFloat}};
  Fixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  // The literal is a half subnormal; preserve inputs for this source-decoding test.
  fx.wf->set_mode_raw(0xf0);
  const auto out = fx.run(words.data(), 0x08000800u);
  expect_all_active(out, 0x08790879u, "v_pk_add_f16 v4, v0, lit(0xF2) op_sel_hi:[1,0]");
}

// Control: inline INTEGER constants are not float constants. v_pk_add_i16 with
// inline 1 keeps reading 1 from the low half of the resolved value.
TEST(Vop3pPkInlineConstantCorrectness, AddI16InlineIntegerConstantIsUnchanged) {
  const auto words =
      rdna4::build_vop3p(rdna4::kVPkAddI16Vop3p,
                         {.vdst = kDstVgpr, .src0 = 256, .src1 = kInlineOneInt, .opsel_hi = 1});
  Fixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  const auto out = fx.run(words.data(), 0x00050005u);
  expect_all_active(out, 0x00060006u, "v_pk_add_i16 v4, v0, 1 op_sel_hi:[1,0]");
}

// Default packing (op_sel_hi = 3) is the one shape the SIMD fast path accepts.
// The inline-constant rewrite lives in the scalar body only, so the fast path
// has to decline an inline float source rather than diverge from it.
//
// This has to run on an arch that reaches the probe. RDNA4 and CDNA5 emit
// local v_pk_add_f16 bodies with no ROCJITSU_TRY_SIMD line at all, so both
// modes would take the same scalar loop and the comparison would be vacuous;
// CDNA4 delegates to execute_v_pk_add_f16_vop3p, which carries the probe.
//
// This shape also pins the HIGH half. An inline float constant occupies a
// packed-16 source as {lo = the 16-bit pattern, hi = 0}: the LLVM assembler
// inline-encodes the packed v2f16 immediate 0x00003C00 as "1.0" for
// v_pk_add_f16 but has to spill 0x3C003C00 to a 32-bit literal, which it would
// not do if the hardware replicated the constant into both halves. So with
// op_sel_hi:[1,1] and v0 = {1.0h, 1.0h} the low lane adds 1.0h and the HIGH
// lane adds +0.0h, giving {1.0h, 2.0h} = 0x3C004000.
TEST(Vop3pPkInlineConstantCorrectness, AddF16InlineOneDefaultPackingAgreesWithScalar) {
  ForceScalarGuard gate_guard;
  const auto words =
      cdna4::build_vop3p(cdna4::kVPkAddF16Vop3p,
                         {.vdst = kDstVgpr, .src0 = 256, .src1 = kInlineOneFloat, .op_sel_hi = 3});

  const auto run_mode = [&](bool force_scalar) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx(ROCJITSU_CODE_ARCH_CDNA4);
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    return fx.run(words.data(), 0x3C003C00u);
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (kExec >> lane) & 1u;
    EXPECT_EQ(scalar_out[lane], simd_out[lane])
        << "op_sel_hi:[1,1]: SIMD path diverged from scalar body at lane " << lane;
    if (active)
      EXPECT_EQ(scalar_out[lane], 0x3C004000u)
          << "op_sel_hi:[1,1]: inline 1.0 did not land in the low half only, at lane " << lane;
    else
      EXPECT_EQ(scalar_out[lane], DST_SENTINEL) << "clobbered inactive lane " << lane;
  }
}

// The narrowing has to round, not truncate: selector 248 is 1/(2*pi), whose f32
// form 0x3E22F983 sits between two f16 values. The emulator's own 16-bit inline
// table (scalar_operand_resolve.h) spells that constant 0x3118, so RNE is what
// agrees with the rest of the model; truncating would give 0x3117. Adding it to
// zero leaves the narrowed pattern intact in both halves.
TEST(Vop3pPkInlineConstantCorrectness, AddF16InlineInv2PiRoundsToNearestEven) {
  const auto words =
      rdna4::build_vop3p(rdna4::kVPkAddF16Vop3p,
                         {.vdst = kDstVgpr, .src0 = 256, .src1 = kInlineInv2Pi, .opsel_hi = 1});
  Fixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  const auto out = fx.run(words.data(), 0x00000000u);
  expect_all_active(out, 0x31183118u, "v_pk_add_f16 v4, v0, 0.15915494 op_sel_hi:[1,0]");
}

// src2 takes the same 32-bit read as src0/src1 on a three-source packed op.
// v_pk_fma_f16 v4, v0, v0, 1.0 op_sel_hi:[1,1,0] with v0 = {2.0h, 2.0h} is
// 2*2 + 1 in both halves. Leaving src2 unnarrowed adds 0 and gives 0x44004400.
TEST(Vop3pPkInlineConstantCorrectness, FmaF16InlineOneOnSrc2ReachesBothHalves) {
  const auto words = rdna4::build_vop3p(
      rdna4::kVPkFmaF16Vop3p,
      {.vdst = kDstVgpr, .src0 = 256, .src1 = 256, .src2 = kInlineOneFloat, .opsel_hi = 3});
  Fixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  const auto out = fx.run(words.data(), 0x40004000u);
  expect_all_active(out, 0x45004500u, "v_pk_fma_f16 v4, v0, v0, 1.0 op_sel_hi:[1,1,0]");
}

// v_dot2_f32_f16 v4, v0, 1.0, 0 with v0 = {1.0h, 0}. The dot is
// a.lo*b.lo + a.hi*b.hi + c, so the answer is 1.0f. Unnarrowed, src1 reads
// {lo = 0, hi = 0x3F80} and both products drop out, leaving 0.
//
// RDNA4 carries its own generated dot2 bodies, which have no SIMD probe.
TEST(Vop3pPkInlineConstantCorrectness, Dot2F32F16InlineOneNarrowsOnLocalBody) {
  const auto words = rdna4::build_vop3p(
      rdna4::kVDot2F32F16Vop3p,
      {.vdst = kDstVgpr, .src0 = 256, .src1 = kInlineOneFloat, .src2 = kInlineZero, .opsel_hi = 3});
  Fixture fx(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  const auto out = fx.run(words.data(), 0x00003C00u);
  expect_all_active(out, 0x3F800000u, "v_dot2_f32_f16 v4, v0, 1.0, 0");
}

// Same dot, packed BF16, on RDNA3 -- which calls the generated dot2 template
// every target except RDNA4 shares. bf16 1.0 is 0x3F80.
//
// The shared body is also the only one that carries the SIMD probe, and
// op_sel_hi:[1,1] is the packing try_execute_vop3p_dot_f16_simd accepts. The
// narrowing lives in the scalar body, so the fast path has to decline an inline
// float source rather than diverge from it; both modes run here.
TEST(Vop3pPkInlineConstantCorrectness, Dot2F32Bf16InlineOneNarrowsAndSimdAgrees) {
  ForceScalarGuard gate_guard;
  const auto words = rdna3::build_vop3p(rdna3::kVDot2F32Bf16Vop3p, {.vdst = kDstVgpr,
                                                                    .src0 = 256,
                                                                    .src1 = kInlineOneFloat,
                                                                    .src2 = kInlineZero,
                                                                    .op_sel_hi = 3});

  const auto run_mode = [&](bool force_scalar) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx(ROCJITSU_CODE_ARCH_RDNA3);
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    EXPECT_EQ(fx.wf->wf_size(), WF_SIZE) << "fixture assumes wave32";
    return fx.run(words.data(), 0x00003F80u);
  };

  const auto scalar_out = run_mode(/*force_scalar=*/true);
  const auto simd_out = run_mode(/*force_scalar=*/false);
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    EXPECT_EQ(scalar_out[lane], simd_out[lane])
        << "v_dot2_f32_bf16: SIMD path diverged from scalar body at lane " << lane;
    const bool active = (kExec >> lane) & 1u;
    if (active)
      EXPECT_EQ(scalar_out[lane], 0x3F800000u) << "v_dot2_f32_bf16 v4, v0, 1.0, 0: lane " << lane;
    else
      EXPECT_EQ(scalar_out[lane], DST_SENTINEL) << "clobbered inactive lane " << lane;
  }
}

} // namespace
