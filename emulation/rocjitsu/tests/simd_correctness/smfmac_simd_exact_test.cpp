// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file smfmac_simd_exact_test.cpp
/// @brief Decoded CDNA3/CDNA4 F32 SMFMAC SIMD-vs-scalar checks.

#include "mma_exact_test_support.h"

#include "../decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace mma_exact;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t A_OFF = 0;
constexpr uint32_t B_OFF = 32;
constexpr uint32_t INDEX_OFF = 64;
constexpr uint32_t DST_OFF = 96;
constexpr uint32_t STATE_REGS = 128;
constexpr uint32_t MAX_DST_REGS = 16;

// A nibble is (high two bits: second K position, low two bits: first K
// position). Exercise all six legal ordered 2-of-4 selector pairs.
constexpr std::array<uint32_t, 6> LEGAL_PAIRS{0x4u, 0x8u, 0xCu, 0x9u, 0xDu, 0xEu};

enum class Alias { None, A, B, Index };

struct SmfmacCase {
  rj_code_arch_t arch;
  uint16_t opcode;
  const char *name;
  Fmt a_fmt;
  Fmt b_fmt;
  uint8_t a_regs;
  uint8_t b_regs;
  uint8_t dst_regs;
};

#define C3(OP, NAME, A, B, D, AFMT, BFMT)                                                          \
  SmfmacCase { ROCJITSU_CODE_ARCH_CDNA3, cdna3::k##OP##Vop3pMfma, NAME, AFMT, BFMT, A, B, D }
#define C4(OP, NAME, A, B, D, AFMT, BFMT)                                                          \
  SmfmacCase { ROCJITSU_CODE_ARCH_CDNA4, cdna4::k##OP##Vop3pMfma, NAME, AFMT, BFMT, A, B, D }

// Every generated, implemented F32 SMFMAC form. The I32 forms currently
// report UnimplementedInstruction and are intentionally outside this suite.
constexpr std::array CDNA3_CASES{
    C3(VSmfmacF3216x16x32F16, "v_smfmac_f32_16x16x32_f16", 2, 4, 4, Fmt::F16, Fmt::F16),
    C3(VSmfmacF3232x32x16F16, "v_smfmac_f32_32x32x16_f16", 2, 4, 16, Fmt::F16, Fmt::F16),
    C3(VSmfmacF3216x16x32Bf16, "v_smfmac_f32_16x16x32_bf16", 2, 4, 4, Fmt::BF16, Fmt::BF16),
    C3(VSmfmacF3232x32x16Bf16, "v_smfmac_f32_32x32x16_bf16", 2, 4, 16, Fmt::BF16, Fmt::BF16),
    C3(VSmfmacF3216x16x64Bf8Bf8, "v_smfmac_f32_16x16x64_bf8_bf8", 2, 4, 4, Fmt::BF8, Fmt::BF8),
    C3(VSmfmacF3216x16x64Bf8Fp8, "v_smfmac_f32_16x16x64_bf8_fp8", 2, 4, 4, Fmt::BF8, Fmt::FP8),
    C3(VSmfmacF3216x16x64Fp8Bf8, "v_smfmac_f32_16x16x64_fp8_bf8", 2, 4, 4, Fmt::FP8, Fmt::BF8),
    C3(VSmfmacF3216x16x64Fp8Fp8, "v_smfmac_f32_16x16x64_fp8_fp8", 2, 4, 4, Fmt::FP8, Fmt::FP8),
    C3(VSmfmacF3232x32x32Bf8Bf8, "v_smfmac_f32_32x32x32_bf8_bf8", 2, 4, 16, Fmt::BF8, Fmt::BF8),
    C3(VSmfmacF3232x32x32Bf8Fp8, "v_smfmac_f32_32x32x32_bf8_fp8", 2, 4, 16, Fmt::BF8, Fmt::FP8),
    C3(VSmfmacF3232x32x32Fp8Bf8, "v_smfmac_f32_32x32x32_fp8_bf8", 2, 4, 16, Fmt::FP8, Fmt::BF8),
    C3(VSmfmacF3232x32x32Fp8Fp8, "v_smfmac_f32_32x32x32_fp8_fp8", 2, 4, 16, Fmt::FP8, Fmt::FP8),
};

constexpr std::array CDNA4_CASES{
    C4(VSmfmacF3216x16x64F16, "v_smfmac_f32_16x16x64_f16", 4, 8, 4, Fmt::F16, Fmt::F16),
    C4(VSmfmacF3232x32x32F16, "v_smfmac_f32_32x32x32_f16", 4, 8, 16, Fmt::F16, Fmt::F16),
    C4(VSmfmacF3216x16x64Bf16, "v_smfmac_f32_16x16x64_bf16", 4, 8, 4, Fmt::BF16, Fmt::BF16),
    C4(VSmfmacF3232x32x32Bf16, "v_smfmac_f32_32x32x32_bf16", 4, 8, 16, Fmt::BF16, Fmt::BF16),
    C4(VSmfmacF3216x16x128Bf8Bf8, "v_smfmac_f32_16x16x128_bf8_bf8", 4, 8, 4, Fmt::BF8, Fmt::BF8),
    C4(VSmfmacF3216x16x128Bf8Fp8, "v_smfmac_f32_16x16x128_bf8_fp8", 4, 8, 4, Fmt::BF8, Fmt::FP8),
    C4(VSmfmacF3216x16x128Fp8Bf8, "v_smfmac_f32_16x16x128_fp8_bf8", 4, 8, 4, Fmt::FP8, Fmt::BF8),
    C4(VSmfmacF3216x16x128Fp8Fp8, "v_smfmac_f32_16x16x128_fp8_fp8", 4, 8, 4, Fmt::FP8, Fmt::FP8),
    C4(VSmfmacF3232x32x64Bf8Bf8, "v_smfmac_f32_32x32x64_bf8_bf8", 4, 8, 16, Fmt::BF8, Fmt::BF8),
    C4(VSmfmacF3232x32x64Bf8Fp8, "v_smfmac_f32_32x32x64_bf8_fp8", 4, 8, 16, Fmt::BF8, Fmt::FP8),
    C4(VSmfmacF3232x32x64Fp8Bf8, "v_smfmac_f32_32x32x64_fp8_bf8", 4, 8, 16, Fmt::FP8, Fmt::BF8),
    C4(VSmfmacF3232x32x64Fp8Fp8, "v_smfmac_f32_32x32x64_fp8_fp8", 4, 8, 16, Fmt::FP8, Fmt::FP8),
    // gfx950 also exposes the CDNA3 shapes and encodings.
    C4(VSmfmacF3216x16x32F16, "v_smfmac_f32_16x16x32_f16", 2, 4, 4, Fmt::F16, Fmt::F16),
    C4(VSmfmacF3232x32x16F16, "v_smfmac_f32_32x32x16_f16", 2, 4, 16, Fmt::F16, Fmt::F16),
    C4(VSmfmacF3216x16x32Bf16, "v_smfmac_f32_16x16x32_bf16", 2, 4, 4, Fmt::BF16, Fmt::BF16),
    C4(VSmfmacF3232x32x16Bf16, "v_smfmac_f32_32x32x16_bf16", 2, 4, 16, Fmt::BF16, Fmt::BF16),
    C4(VSmfmacF3216x16x64Bf8Bf8, "v_smfmac_f32_16x16x64_bf8_bf8", 2, 4, 4, Fmt::BF8, Fmt::BF8),
    C4(VSmfmacF3216x16x64Bf8Fp8, "v_smfmac_f32_16x16x64_bf8_fp8", 2, 4, 4, Fmt::BF8, Fmt::FP8),
    C4(VSmfmacF3216x16x64Fp8Bf8, "v_smfmac_f32_16x16x64_fp8_bf8", 2, 4, 4, Fmt::FP8, Fmt::BF8),
    C4(VSmfmacF3216x16x64Fp8Fp8, "v_smfmac_f32_16x16x64_fp8_fp8", 2, 4, 4, Fmt::FP8, Fmt::FP8),
    C4(VSmfmacF3232x32x32Bf8Bf8, "v_smfmac_f32_32x32x32_bf8_bf8", 2, 4, 16, Fmt::BF8, Fmt::BF8),
    C4(VSmfmacF3232x32x32Bf8Fp8, "v_smfmac_f32_32x32x32_bf8_fp8", 2, 4, 16, Fmt::BF8, Fmt::FP8),
    C4(VSmfmacF3232x32x32Fp8Bf8, "v_smfmac_f32_32x32x32_fp8_bf8", 2, 4, 16, Fmt::FP8, Fmt::BF8),
    C4(VSmfmacF3232x32x32Fp8Fp8, "v_smfmac_f32_32x32x32_fp8_fp8", 2, 4, 16, Fmt::FP8, Fmt::FP8),
};

#undef C3
#undef C4

struct SmfmacFixture : ExactFixture {
  explicit SmfmacFixture(rj_code_arch_t arch) : ExactFixture(arch, WF_SIZE) {}
};

void restore_window(SmfmacFixture &fx, uint32_t off, const std::vector<uint32_t> &words) {
  const uint32_t regs = static_cast<uint32_t>(words.size() / WF_SIZE);
  for (uint32_t reg = 0; reg < regs; ++reg)
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      fx.cu->write_vgpr(fx.vbase + off + reg, lane,
                        words[static_cast<size_t>(reg) * WF_SIZE + lane]);
}

void restore(SmfmacFixture &fx, const std::vector<uint32_t> &words) {
  restore_window(fx, 0, words);
}

uint8_t f8_element(Fmt fmt, Mode mode, uint32_t i, std::mt19937 &rng, bool fnuz) {
  if (mode == Mode::NaN) {
    if (fnuz)
      return 0x80u;
    if (fmt == Fmt::FP8)
      return (i & 1) ? 0xFFu : 0x7Fu;
    return static_cast<uint8_t>((i & 1 ? 0x80u : 0u) | (0x7Du + i % 3));
  }
  if (mode == Mode::Inf)
    return static_cast<uint8_t>((i & 1 ? 0x80u : 0u) | 0x7Cu);
  if (mode == Mode::Denorm)
    return static_cast<uint8_t>((i & 1 ? 0x80u : 0u) | 0x01u);
  if (mode == Mode::MaxFinite)
    return fnuz ? static_cast<uint8_t>((i & 1 ? 0x80u : 0u) | 0x7Fu)
                : static_cast<uint8_t>((i & 1 ? 0x80u : 0u) | (fmt == Fmt::FP8 ? 0x7Eu : 0x7Bu));
  if (mode == Mode::Zeros)
    return 0;
  if (mode == Mode::SignedZero)
    return fnuz ? 0u : static_cast<uint8_t>((i & 1) ? 0x80u : 0u);

  float value = 0;
  if (mode == Mode::Cancel)
    value = (i & 1) ? -1.0f : 1.0f;
  else
    value = static_cast<float>(std::uniform_int_distribution<int>(-4, 4)(rng));
  if (fnuz)
    return fmt == Fmt::FP8 ? util::f32_to_fp8_e4m3_fnuz_rne(value)
                           : util::f32_to_bf8_e5m2_fnuz_rne(value);
  return fmt == Fmt::FP8 ? util::f32_to_fp8_e4m3_rne(value) : util::f32_to_bf8_e5m2_rne(value);
}

void seed_input(SmfmacFixture &fx, uint32_t off, uint32_t regs, Fmt fmt, Mode mode, uint32_t seed,
                bool fnuz) {
  if (fmt != Fmt::FP8 && fmt != Fmt::BF8) {
    fx.seed(off, regs, fmt, mode, seed);
    return;
  }
  std::mt19937 rng(seed);
  uint32_t i = 0;
  for (uint32_t reg = 0; reg < regs; ++reg)
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint32_t word = 0;
      for (uint32_t byte = 0; byte < 4; ++byte)
        word |= static_cast<uint32_t>(f8_element(fmt, mode, i++, rng, fnuz)) << (8 * byte);
      fx.cu->write_vgpr(fx.vbase + off + reg, lane, word);
    }
}

void seed_indices(SmfmacFixture &fx, uint32_t off, uint32_t phase) {
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    uint32_t word = 0;
    for (uint32_t nibble = 0; nibble < 8; ++nibble)
      word |= LEGAL_PAIRS[(lane + 5 * nibble + phase) % LEGAL_PAIRS.size()] << (4 * nibble);
    fx.cu->write_vgpr(fx.vbase + off, lane, word);
  }
}

std::unique_ptr<Instruction> decode(const SmfmacCase &test, uint32_t a_off, uint32_t b_off,
                                    uint32_t index_off, bool acc_dst = false,
                                    bool acc_sources = false) {
  const auto fields = cdna4::Vop3pMfmaBuilderFields{
      .vdst = DST_OFF,
      .acc_cd = static_cast<uint8_t>(acc_dst),
      .src0 = static_cast<uint16_t>(256 + a_off),
      .src1 = static_cast<uint16_t>(256 + b_off),
      .src2 = static_cast<uint16_t>(256 + index_off),
      .acc = static_cast<uint8_t>(acc_sources ? 3 : 0),
  };
  auto decoder = Decoder::create(test.arch);
  if (!decoder)
    return nullptr;
  if (test.arch == ROCJITSU_CODE_ARCH_CDNA3) {
    const auto words = cdna3::build_vop3p_mfma(test.opcode, {.vdst = fields.vdst,
                                                             .acc_cd = fields.acc_cd,
                                                             .src0 = fields.src0,
                                                             .src1 = fields.src1,
                                                             .src2 = fields.src2,
                                                             .acc = fields.acc});
    return std::unique_ptr<Instruction>(decode_valid(*decoder, words.data()));
  }
  const auto words = cdna4::build_vop3p_mfma(test.opcode, fields);
  return std::unique_ptr<Instruction>(decode_valid(*decoder, words.data()));
}

void compare_state(const SmfmacCase &test, const std::vector<uint32_t> &initial,
                   const std::vector<uint32_t> &scalar, const std::vector<uint32_t> &simd) {
  for (uint32_t reg = 0; reg < STATE_REGS; ++reg)
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const size_t i = static_cast<size_t>(reg) * WF_SIZE + lane;
      ASSERT_EQ(simd[i], scalar[i])
          << test.name << " reg=" << reg << " lane=" << lane << " scalar=0x" << std::hex
          << scalar[i] << " simd=0x" << simd[i];
      if (reg < DST_OFF || reg >= DST_OFF + test.dst_regs) {
        ASSERT_EQ(scalar[i], initial[i])
            << test.name << ": scalar path changed source/canary reg=" << reg << " lane=" << lane;
        ASSERT_EQ(simd[i], initial[i])
            << test.name << ": SIMD path changed source/canary reg=" << reg << " lane=" << lane;
      }
    }
}

void execute_both(SmfmacFixture &fx, const SmfmacCase &test, Instruction &instruction,
                  const std::vector<uint32_t> &initial) {
  ForceScalarGuard force_scalar_guard;
  restore(fx, initial);
  util::set_force_scalar_for_testing(true);
  const bool scalar_ok = fx.cu->execute_instruction(&instruction, *fx.wf).succeeded();
  const auto scalar = fx.snapshot(0, STATE_REGS);

  restore(fx, initial);
  util::set_force_scalar_for_testing(false);
  const bool simd_ok = fx.cu->execute_instruction(&instruction, *fx.wf).succeeded();
  const auto simd = fx.snapshot(0, STATE_REGS);

  ASSERT_TRUE(scalar_ok) << test.name << ": scalar execution failed";
  ASSERT_TRUE(simd_ok) << test.name << ": SIMD execution failed";
  compare_state(test, initial, scalar, simd);
}

void run_case(const SmfmacCase &test, Mode mode, uint32_t seed, uint32_t phase,
              Alias alias = Alias::None) {
  SmfmacFixture fx(test.arch);
  ASSERT_NE(fx.wf, nullptr);
  const uint32_t a_off = alias == Alias::A ? DST_OFF : A_OFF;
  const uint32_t b_off = alias == Alias::B ? DST_OFF : B_OFF;
  const uint32_t index_off = alias == Alias::Index ? DST_OFF : INDEX_OFF;

  restore(fx, std::vector<uint32_t>(static_cast<size_t>(STATE_REGS) * WF_SIZE, 0));
  // D is also C, so every case starts with a real accumulator. For overlap
  // cases, overwrite its initial words with the aliased source afterwards.
  fx.seed(DST_OFF, test.dst_regs, Fmt::F32, Mode::RandomInt, seed + 3);
  const bool fnuz = test.arch == ROCJITSU_CODE_ARCH_CDNA3;
  seed_input(fx, a_off, test.a_regs, test.a_fmt, mode, seed + 1, fnuz);
  seed_input(fx, b_off, test.b_regs, test.b_fmt, mode, seed + 2, fnuz);
  seed_indices(fx, index_off, phase);
  fx.seed_words(DST_OFF - 1, 1, seed + 4);
  fx.seed_words(DST_OFF + MAX_DST_REGS, 1, seed + 5);

  auto instruction = decode(test, a_off, b_off, index_off);
  ASSERT_NE(instruction, nullptr) << test.name;
  const auto initial = fx.snapshot(0, STATE_REGS);
  execute_both(fx, test, *instruction, initial);
}

void run_acc_case(const SmfmacCase &test, bool acc_sources, uint32_t seed) {
  ASSERT_EQ(test.arch, ROCJITSU_CODE_ARCH_CDNA3);
  SmfmacFixture fx(test.arch);
  ASSERT_NE(fx.wf, nullptr);
  restore(fx, std::vector<uint32_t>(static_cast<size_t>(STATE_REGS) * WF_SIZE, 0));
  restore_window(fx, amdgpu::ACC_VGPR_OFFSET,
                 std::vector<uint32_t>(static_cast<size_t>(STATE_REGS) * WF_SIZE, 0));

  const uint32_t src_bank = acc_sources ? amdgpu::ACC_VGPR_OFFSET : 0;
  seed_input(fx, src_bank + A_OFF, test.a_regs, test.a_fmt, Mode::RandomInt, seed + 1, true);
  seed_input(fx, src_bank + B_OFF, test.b_regs, test.b_fmt, Mode::RandomInt, seed + 2, true);
  // src2 is an ordinary VGPR operand; only A/B use the MFMA acc selector.
  seed_indices(fx, INDEX_OFF, seed);
  fx.seed(amdgpu::ACC_VGPR_OFFSET + DST_OFF, test.dst_regs, Fmt::F32, Mode::RandomInt, seed + 3);
  fx.seed_words(amdgpu::ACC_VGPR_OFFSET + DST_OFF - 1, 1, seed + 4);
  fx.seed_words(amdgpu::ACC_VGPR_OFFSET + DST_OFF + MAX_DST_REGS, 1, seed + 5);

  auto instruction = decode(test, A_OFF, B_OFF, INDEX_OFF, true, acc_sources);
  ASSERT_NE(instruction, nullptr) << test.name;
  const auto initial_v = fx.snapshot(0, STATE_REGS);
  const auto initial_acc = fx.snapshot(amdgpu::ACC_VGPR_OFFSET, STATE_REGS);
  ForceScalarGuard force_scalar_guard;
  restore(fx, initial_v);
  restore_window(fx, amdgpu::ACC_VGPR_OFFSET, initial_acc);
  util::set_force_scalar_for_testing(true);
  const bool scalar_ok = fx.cu->execute_instruction(instruction.get(), *fx.wf).succeeded();
  const auto scalar_v = fx.snapshot(0, STATE_REGS);
  const auto scalar_acc = fx.snapshot(amdgpu::ACC_VGPR_OFFSET, STATE_REGS);

  restore(fx, initial_v);
  restore_window(fx, amdgpu::ACC_VGPR_OFFSET, initial_acc);
  util::set_force_scalar_for_testing(false);
  const bool simd_ok = fx.cu->execute_instruction(instruction.get(), *fx.wf).succeeded();
  const auto simd_v = fx.snapshot(0, STATE_REGS);
  const auto simd_acc = fx.snapshot(amdgpu::ACC_VGPR_OFFSET, STATE_REGS);

  ASSERT_TRUE(scalar_ok) << test.name << ": AccVGPR scalar execution failed";
  ASSERT_TRUE(simd_ok) << test.name << ": AccVGPR SIMD execution failed";
  for (uint32_t reg = 0; reg < STATE_REGS; ++reg)
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const size_t i = static_cast<size_t>(reg) * WF_SIZE + lane;
      ASSERT_EQ(scalar_v[i], initial_v[i]) << test.name << ": scalar changed VGPR " << reg;
      ASSERT_EQ(simd_v[i], initial_v[i]) << test.name << ": SIMD changed VGPR " << reg;
      ASSERT_EQ(simd_acc[i], scalar_acc[i])
          << test.name << ": AccVGPR result mismatch at reg=" << reg << " lane=" << lane;
      if (reg < DST_OFF || reg >= DST_OFF + test.dst_regs) {
        ASSERT_EQ(scalar_acc[i], initial_acc[i])
            << test.name << ": scalar changed AccVGPR source/canary reg=" << reg;
        ASSERT_EQ(simd_acc[i], initial_acc[i])
            << test.name << ": SIMD changed AccVGPR source/canary reg=" << reg;
      }
    }
}

void run_nan_priority_case(const SmfmacCase &test, bool nan_in_b, uint32_t seed) {
  SmfmacFixture fx(test.arch);
  ASSERT_NE(fx.wf, nullptr);
  restore(fx, std::vector<uint32_t>(static_cast<size_t>(STATE_REGS) * WF_SIZE, 0));
  const bool fnuz = test.arch == ROCJITSU_CODE_ARCH_CDNA3;
  seed_input(fx, A_OFF, test.a_regs, test.a_fmt, Mode::Cancel, seed + 1, fnuz);
  seed_input(fx, B_OFF, test.b_regs, test.b_fmt, nan_in_b ? Mode::NaN : Mode::Cancel, seed + 2,
             fnuz);
  fx.seed(DST_OFF, test.dst_regs, Fmt::F32, Mode::RandomInt, seed + 3);
  if (!nan_in_b)
    for (uint32_t reg = 0; reg < test.dst_regs; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        // Signaling NaN with a payload: replay must follow tied C as well as
        // the packed A and B inputs.
        fx.cu->write_vgpr(fx.vbase + DST_OFF + reg, lane, 0x7FA12345u);
  seed_indices(fx, INDEX_OFF, seed);
  fx.seed_words(DST_OFF - 1, 1, seed + 4);
  fx.seed_words(DST_OFF + MAX_DST_REGS, 1, seed + 5);
  auto instruction = decode(test, A_OFF, B_OFF, INDEX_OFF);
  ASSERT_NE(instruction, nullptr) << test.name;
  const auto initial = fx.snapshot(0, STATE_REGS);
  execute_both(fx, test, *instruction, initial);
}

template <size_t N> void run_all_forms(const std::array<SmfmacCase, N> &cases) {
  constexpr std::array MODES{Mode::RandomInt, Mode::Zeros, Mode::SignedZero, Mode::Cancel,
                             Mode::NaN,       Mode::Inf,   Mode::Denorm,     Mode::MaxFinite};
  for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
    const auto &test = cases[case_index];
    for (size_t mode_index = 0; mode_index < MODES.size(); ++mode_index) {
      const Mode mode = MODES[mode_index];
      // FP8 has no infinity, and CDNA3 FNUZ BF8 has none either.
      if (mode == Mode::Inf && (test.a_fmt == Fmt::FP8 || test.b_fmt == Fmt::FP8 ||
                                (test.arch == ROCJITSU_CODE_ARCH_CDNA3 && test.a_fmt == Fmt::BF8)))
        continue;
      SCOPED_TRACE(::testing::Message() << test.name << " arch=" << static_cast<int>(test.arch)
                                        << " mode=" << static_cast<int>(mode));
      run_case(test, mode, 101 + 17 * case_index + mode_index, case_index + mode_index);
      if (testing::Test::HasFatalFailure())
        return;
    }
  }
}

template <size_t N> void run_all_aliases(const std::array<SmfmacCase, N> &cases) {
  constexpr std::array ALIASES{Alias::A, Alias::B, Alias::Index};
  for (size_t case_index = 0; case_index < cases.size(); ++case_index)
    for (size_t alias_index = 0; alias_index < ALIASES.size(); ++alias_index) {
      const auto &test = cases[case_index];
      SCOPED_TRACE(::testing::Message() << test.name << " arch=" << static_cast<int>(test.arch)
                                        << " alias=" << static_cast<int>(ALIASES[alias_index]));
      run_case(test, Mode::RandomInt, 701 + 17 * case_index + alias_index, case_index + alias_index,
               ALIASES[alias_index]);
      if (testing::Test::HasFatalFailure())
        return;
    }
}

template <size_t N> void run_all_nan_sources(const std::array<SmfmacCase, N> &cases) {
  for (size_t case_index = 0; case_index < cases.size(); ++case_index)
    for (bool nan_in_b : {true, false}) {
      const auto &test = cases[case_index];
      SCOPED_TRACE(::testing::Message() << test.name << " arch=" << static_cast<int>(test.arch)
                                        << (nan_in_b ? " NaN in B" : " signaling NaN in tied C"));
      run_nan_priority_case(test, nan_in_b, 1901 + 17 * case_index + !nan_in_b);
      if (testing::Test::HasFatalFailure())
        return;
    }
}

void check_fused_overflow_cancellation(const SmfmacCase &test, bool negate) {
  SmfmacFixture fx(test.arch);
  ASSERT_NE(fx.wf, nullptr);
  restore(fx, std::vector<uint32_t>(static_cast<size_t>(STATE_REGS) * WF_SIZE, 0));
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
    fx.cu->write_vgpr(fx.vbase + INDEX_OFF, lane, 0x44444444u);

  // One selected BF16 product is (+/-2^64) * 2^64. Fusing it with
  // (-/+FLT_MAX) yields +/-2^104. A rounded multiply would overflow to Inf.
  fx.cu->write_vgpr(fx.vbase + A_OFF, 0, negate ? 0x0000DF80u : 0x00005F80u);
  fx.cu->write_vgpr(fx.vbase + B_OFF, 0, 0x00005F80u);
  fx.cu->write_vgpr(fx.vbase + DST_OFF, 0, negate ? 0x7F7FFFFFu : 0xFF7FFFFFu);
  auto instruction = decode(test, A_OFF, B_OFF, INDEX_OFF);
  ASSERT_NE(instruction, nullptr);
  const auto initial = fx.snapshot(0, STATE_REGS);
  execute_both(fx, test, *instruction, initial);
  if (testing::Test::HasFatalFailure())
    return;
  const auto result = fx.snapshot(DST_OFF, test.dst_regs);
  EXPECT_EQ(result[0], negate ? 0xF3800000u : 0x73800000u)
      << test.name << ": SMFMAC must use fused accumulation";
}

} // namespace

TEST(SmfmacSimdExact, Cdna3DecodedForms) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "SMFMAC AVX-512 fast path requires 16-wide native SIMD";
  run_all_forms(CDNA3_CASES);
}

TEST(SmfmacSimdExact, Cdna4DecodedForms) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "SMFMAC AVX-512 fast path requires 16-wide native SIMD";
  run_all_forms(CDNA4_CASES);
}

TEST(SmfmacSimdExact, Cdna3DestinationAliasesEachSource) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "SMFMAC AVX-512 fast path requires 16-wide native SIMD";
  run_all_aliases(CDNA3_CASES);
}

TEST(SmfmacSimdExact, Cdna4DestinationAliasesEachSource) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "SMFMAC AVX-512 fast path requires 16-wide native SIMD";
  run_all_aliases(CDNA4_CASES);
}

TEST(SmfmacSimdExact, Cdna3AccVgprDestinationAndSources) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "SMFMAC AVX-512 fast path requires 16-wide native SIMD";
  for (size_t i = 0; i < CDNA3_CASES.size(); ++i)
    for (bool acc_sources : {false, true}) {
      const auto &test = CDNA3_CASES[i];
      SCOPED_TRACE(::testing::Message() << test.name << " acc_cd=1 acc_sources=" << acc_sources);
      run_acc_case(test, acc_sources, 1201 + 17 * i + acc_sources);
      if (testing::Test::HasFatalFailure())
        return;
    }
}

TEST(SmfmacSimdExact, Cdna3NaNSourcePriority) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "SMFMAC AVX-512 fast path requires 16-wide native SIMD";
  run_all_nan_sources(CDNA3_CASES);
}

TEST(SmfmacSimdExact, Cdna4NaNSourcePriority) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "SMFMAC AVX-512 fast path requires 16-wide native SIMD";
  run_all_nan_sources(CDNA4_CASES);
}

TEST(SmfmacSimdExact, Bf16OverflowCancellationRequiresFusion) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "SMFMAC AVX-512 fast path requires 16-wide native SIMD";
  for (const auto &test : {CDNA3_CASES[2], CDNA4_CASES[2]})
    for (bool negate : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << test.name << " arch=" << static_cast<int>(test.arch) << " negate=" << negate);
      check_fused_overflow_cancellation(test, negate);
    }
}
