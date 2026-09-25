// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file swmmac_k64_simd_exact_test.cpp
/// @brief Decoded SIMD-vs-scalar checks for the five gfx1250 K=64
/// F16/BF16 SWMMAC instructions.

#include "mma_exact_test_support.h"

#include "../decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace mma_exact;

constexpr uint32_t WF_SIZE = mma_test::WMMA_WF_SIZE;
constexpr uint32_t A_OFF = 0;
constexpr uint32_t B_OFF = 32;
constexpr uint32_t ACC_OFF = 64;
constexpr uint32_t INDEX_OFF = 96;
constexpr uint32_t STATE_REGS = 128;
constexpr uint32_t INPUT_A_REGS = 8;
constexpr uint32_t INPUT_B_REGS = 16;
constexpr uint32_t F32_ACC_REGS = 8;
constexpr uint32_t PACKED_ACC_REGS = 4;

// Each nibble stores the ordered selector pair for one 2:4 group. These are
// the six and only six legal index0 < index1 combinations.
constexpr std::array<uint32_t, 6> LEGAL_INDEX_PAIRS{
    0x4u, // (0,1)
    0x8u, // (0,2)
    0xCu, // (0,3)
    0x9u, // (1,2)
    0xDu, // (1,3)
    0xEu, // (2,3)
};

enum class OutputKind { F32, F16, BF16, BF16F32 };
enum class DstAlias { None, A, B, Index };

struct SwmmacCase {
  uint16_t opcode;
  const char *label;
  Fmt input_fmt;
  Fmt accumulator_fmt;
  OutputKind output_kind;
  uint32_t accumulator_regs;
  uint32_t output_regs;
};

constexpr std::array SWMMAC_CASES{
    SwmmacCase{cdna5::kVSwmmacF3216x16x64F16Vop3p, "v_swmmac_f32_16x16x64_f16", Fmt::F16, Fmt::F32,
               OutputKind::F32, F32_ACC_REGS, F32_ACC_REGS},
    SwmmacCase{cdna5::kVSwmmacF3216x16x64Bf16Vop3p, "v_swmmac_f32_16x16x64_bf16", Fmt::BF16,
               Fmt::F32, OutputKind::F32, F32_ACC_REGS, F32_ACC_REGS},
    SwmmacCase{cdna5::kVSwmmacF1616x16x64F16Vop3p, "v_swmmac_f16_16x16x64_f16", Fmt::F16, Fmt::F16,
               OutputKind::F16, PACKED_ACC_REGS, PACKED_ACC_REGS},
    SwmmacCase{cdna5::kVSwmmacBf1616x16x64Bf16Vop3p, "v_swmmac_bf16_16x16x64_bf16", Fmt::BF16,
               Fmt::BF16, OutputKind::BF16, PACKED_ACC_REGS, PACKED_ACC_REGS},
    // Unlike the ordinary BF16 result form, BF16F32 consumes eight F32 C
    // registers and produces four packed-BF16 D registers.
    SwmmacCase{cdna5::kVSwmmacBf16f3216x16x64Bf16Vop3p, "v_swmmac_bf16f32_16x16x64_bf16", Fmt::BF16,
               Fmt::F32, OutputKind::BF16F32, F32_ACC_REGS, PACKED_ACC_REGS},
};

struct SwmmacFixture : ExactFixture {
  SwmmacFixture() : ExactFixture(ROCJITSU_CODE_ARCH_CDNA5, WF_SIZE) {}
};

struct Observation {
  bool succeeded = false;
  std::vector<uint32_t> output;
  std::vector<uint32_t> tail;
  std::vector<uint32_t> canaries;
};

void restore(SwmmacFixture &fx, uint32_t off, const std::vector<uint32_t> &words) {
  ASSERT_EQ(words.size() % WF_SIZE, 0u);
  const uint32_t regs = static_cast<uint32_t>(words.size() / WF_SIZE);
  for (uint32_t reg = 0; reg < regs; ++reg)
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      fx.cu->write_vgpr(fx.vbase + off + reg, lane,
                        words[static_cast<size_t>(reg) * WF_SIZE + lane]);
}

void seed_legal_indices(SwmmacFixture &fx, uint32_t phase, uint32_t off = INDEX_OFF) {
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    uint32_t word = 0;
    for (uint32_t group = 0; group < 8; ++group) {
      const uint32_t pair = LEGAL_INDEX_PAIRS[(lane + group + phase) % LEGAL_INDEX_PAIRS.size()];
      word |= pair << (4 * group);
    }
    fx.cu->write_vgpr(fx.vbase + off, lane, word);
  }
}

void seed_pattern(SwmmacFixture &fx, uint32_t off, uint32_t regs, uint32_t salt) {
  for (uint32_t reg = 0; reg < regs; ++reg)
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      fx.cu->write_vgpr(fx.vbase + off + reg, lane, 0xA5000000u ^ (salt << 16) ^ (reg << 8) ^ lane);
}

void write_packed16(SwmmacFixture &fx, uint32_t base, const amdgpu::InputLoc &loc, uint16_t value) {
  const uint32_t reg = fx.vbase + base + loc.vgpr_offset;
  const uint32_t old = fx.cu->read_vgpr(reg, loc.lane);
  const uint32_t shift = loc.sub_element * 16;
  const uint32_t word = (old & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(value) << shift);
  fx.cu->write_vgpr(reg, loc.lane, word);
}

void write_packed8(SwmmacFixture &fx, uint32_t base, const amdgpu::InputLoc &loc, uint8_t value) {
  const uint32_t reg = fx.vbase + base + loc.vgpr_offset;
  const uint32_t old = fx.cu->read_vgpr(reg, loc.lane);
  const uint32_t shift = loc.sub_element * 8;
  const uint32_t word = (old & ~(0xFFu << shift)) | (static_cast<uint32_t>(value) << shift);
  fx.cu->write_vgpr(reg, loc.lane, word);
}

void clear_state(SwmmacFixture &fx) {
  restore(fx, 0, std::vector<uint32_t>(static_cast<size_t>(STATE_REGS) * WF_SIZE, 0));
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
    fx.cu->write_vgpr(fx.vbase + INDEX_OFF, lane, 0x44444444u);
  seed_pattern(fx, ACC_OFF - 1, 1, 0x51);
  seed_pattern(fx, ACC_OFF + F32_ACC_REGS, 1, 0x52);
}

Observation execute_from_state(SwmmacFixture &fx, Instruction &instruction,
                               const std::vector<uint32_t> &initial, const SwmmacCase &test,
                               bool force_scalar) {
  restore(fx, 0, initial);
  util::set_force_scalar_for_testing(force_scalar);
  Observation result;
  result.succeeded = fx.cu->execute_instruction(&instruction, *fx.wf).succeeded();
  result.output = fx.snapshot(ACC_OFF, test.output_regs);
  if (test.output_regs < F32_ACC_REGS)
    result.tail = fx.snapshot(ACC_OFF + test.output_regs, F32_ACC_REGS - test.output_regs);
  result.canaries = fx.snapshot(ACC_OFF - 1, 1);
  const auto after = fx.snapshot(ACC_OFF + F32_ACC_REGS, 1);
  result.canaries.insert(result.canaries.end(), after.begin(), after.end());
  return result;
}

void expect_unchanged_tail(const SwmmacCase &test, const Observation &result,
                           const std::vector<uint32_t> &initial) {
  if (test.output_regs == F32_ACC_REGS)
    return;
  const auto first = initial.begin() + static_cast<size_t>(ACC_OFF + test.output_regs) * WF_SIZE;
  const auto last = initial.begin() + static_cast<size_t>(ACC_OFF + F32_ACC_REGS) * WF_SIZE;
  const std::vector<uint32_t> expected(first, last);
  EXPECT_EQ(result.tail, expected)
      << test.label
      << (test.output_kind == OutputKind::BF16F32
              ? ": upper four F32 C registers must survive the packed-BF16 write"
              : ": packed output wrote past its four-register destination");
}

void expect_unchanged_canaries(const SwmmacCase &test, const Observation &result,
                               const std::vector<uint32_t> &initial) {
  std::vector<uint32_t> expected;
  const auto before = initial.begin() + static_cast<size_t>(ACC_OFF - 1) * WF_SIZE;
  expected.insert(expected.end(), before, before + WF_SIZE);
  const auto after = initial.begin() + static_cast<size_t>(ACC_OFF + F32_ACC_REGS) * WF_SIZE;
  expected.insert(expected.end(), after, after + WF_SIZE);
  EXPECT_EQ(result.canaries, expected) << test.label << ": destination canary changed";
}

void run_exact_case(const SwmmacCase &test, Mode mode, uint32_t seed, uint32_t metadata_phase,
                    DstAlias dst_alias = DstAlias::None) {
  SwmmacFixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);

  const uint32_t a_off = dst_alias == DstAlias::A ? ACC_OFF : A_OFF;
  const uint32_t b_off = dst_alias == DstAlias::B ? ACC_OFF : B_OFF;
  const uint32_t index_off = dst_alias == DstAlias::Index ? ACC_OFF : INDEX_OFF;
  if (dst_alias != DstAlias::None) {
    // Alias cases use the homogeneous packed-F16 form so the same words can
    // simultaneously represent C and whichever input overlaps D.
    ASSERT_EQ(test.input_fmt, Fmt::F16);
    ASSERT_EQ(test.accumulator_fmt, Fmt::F16);
  }

  if (dst_alias == DstAlias::A) {
    fx.seed(ACC_OFF, INPUT_A_REGS, Fmt::F16, mode, seed + 1);
  } else {
    fx.seed(A_OFF, INPUT_A_REGS, test.input_fmt, mode, seed + 1);
  }
  if (dst_alias == DstAlias::B) {
    fx.seed(ACC_OFF, INPUT_B_REGS, Fmt::F16, mode, seed + 2);
  } else {
    fx.seed(B_OFF, INPUT_B_REGS, test.input_fmt, mode, seed + 2);
  }
  if (dst_alias != DstAlias::A && dst_alias != DstAlias::B) {
    fx.seed(ACC_OFF, test.accumulator_regs, test.accumulator_fmt, Mode::RandomInt, seed + 3);
    if (test.accumulator_regs < F32_ACC_REGS)
      seed_pattern(fx, ACC_OFF + test.accumulator_regs, F32_ACC_REGS - test.accumulator_regs,
                   seed + 5);
  }
  seed_legal_indices(fx, metadata_phase, index_off);
  seed_pattern(fx, ACC_OFF - 1, 1, seed + 6);
  seed_pattern(fx, ACC_OFF + F32_ACC_REGS, 1, seed + 7);

  const auto words = cdna5::build_vop3p(
      test.opcode, {.vdst = ACC_OFF,
                    .opsel = static_cast<uint8_t>(dst_alias != DstAlias::None ? 4u : 0u),
                    .opsel_hi_2 = static_cast<uint8_t>(dst_alias != DstAlias::None ? 1u : 0u),
                    .src0 = static_cast<uint16_t>(256 + a_off),
                    .src1 = static_cast<uint16_t>(256 + b_off),
                    .src2 = static_cast<uint16_t>(256 + index_off)});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
  ASSERT_NE(instruction, nullptr);

  const auto initial = fx.snapshot(0, STATE_REGS);
  ForceScalarGuard force_scalar_guard;
  const auto scalar = execute_from_state(fx, *instruction, initial, test, true);
  const auto simd = execute_from_state(fx, *instruction, initial, test, false);

  ASSERT_TRUE(scalar.succeeded) << test.label << ": scalar execution failed";
  ASSERT_TRUE(simd.succeeded) << test.label << ": SIMD execution failed";
  EXPECT_EQ(simd.output, scalar.output)
      << test.label << " mode=" << static_cast<int>(mode)
      << ": decoded SIMD result is not bit-identical to forced scalar";
  expect_unchanged_tail(test, scalar, initial);
  expect_unchanged_tail(test, simd, initial);
  expect_unchanged_canaries(test, scalar, initial);
  expect_unchanged_canaries(test, simd, initial);
}

std::unique_ptr<Instruction> decode_case(const SwmmacCase &test) {
  const auto words = cdna5::build_vop3p(
      test.opcode,
      {.vdst = ACC_OFF, .src0 = 256 + A_OFF, .src1 = 256 + B_OFF, .src2 = 256 + INDEX_OFF});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  if (!decoder)
    return nullptr;
  return std::unique_ptr<Instruction>(decode_valid(*decoder, words.data()));
}

std::unique_ptr<Instruction> decode_bf16f32() { return decode_case(SWMMAC_CASES[4]); }

void expect_scalar_and_default(SwmmacFixture &fx, Instruction &instruction, const SwmmacCase &test,
                               const std::vector<uint32_t> &initial,
                               const std::vector<uint32_t> &expected, bool fp16_ovfl,
                               bool check_default = true) {
  ForceScalarGuard force_scalar_guard;
  fx.wf->set_mode_raw(fp16_ovfl ? amdgpu::Wavefront::FP16_OVFL_BIT : 0u);
  const auto scalar = execute_from_state(fx, instruction, initial, test, true);
  ASSERT_TRUE(scalar.succeeded) << test.label << ": scalar execution failed";
  EXPECT_EQ(scalar.output, expected) << test.label << ": scalar result";
  expect_unchanged_tail(test, scalar, initial);
  expect_unchanged_canaries(test, scalar, initial);

  if (!check_default)
    return;
  fx.wf->set_mode_raw(fp16_ovfl ? amdgpu::Wavefront::FP16_OVFL_BIT : 0u);
  const auto default_path = execute_from_state(fx, instruction, initial, test, false);
  ASSERT_TRUE(default_path.succeeded) << test.label << ": default execution failed";
  EXPECT_EQ(default_path.output, expected) << test.label << ": default result";
  expect_unchanged_tail(test, default_path, initial);
  expect_unchanged_canaries(test, default_path, initial);
}

uint32_t output_element(const Observation &observation, const SwmmacCase &test, uint32_t row,
                        uint32_t col) {
  if (test.output_kind == OutputKind::F32) {
    const auto out = amdgpu::gfx12_wmma_output_loc_32(WF_SIZE, 16, 16, row, col);
    return observation.output[static_cast<size_t>(out.reg) * WF_SIZE + out.lane];
  }
  const auto out = amdgpu::gfx12_wmma_output_loc_16(WF_SIZE, 16, 16, row, col);
  const uint32_t word = observation.output[static_cast<size_t>(out.reg) * WF_SIZE + out.lane];
  return (word >> (out.sub_element * 16)) & 0xFFFFu;
}

void expect_scalar_and_default_element(SwmmacFixture &fx, Instruction &instruction,
                                       const SwmmacCase &test, const std::vector<uint32_t> &initial,
                                       uint32_t expected, bool fp16_ovfl) {
  ForceScalarGuard force_scalar_guard;
  fx.wf->set_mode_raw(fp16_ovfl ? amdgpu::Wavefront::FP16_OVFL_BIT : 0u);
  const auto scalar = execute_from_state(fx, instruction, initial, test, true);
  ASSERT_TRUE(scalar.succeeded) << test.label << ": scalar execution failed";
  EXPECT_EQ(output_element(scalar, test, 0, 0), expected) << test.label << ": scalar D[0,0]";
  expect_unchanged_tail(test, scalar, initial);
  expect_unchanged_canaries(test, scalar, initial);

  fx.wf->set_mode_raw(fp16_ovfl ? amdgpu::Wavefront::FP16_OVFL_BIT : 0u);
  const auto default_path = execute_from_state(fx, instruction, initial, test, false);
  ASSERT_TRUE(default_path.succeeded) << test.label << ": default execution failed";
  EXPECT_EQ(default_path.output, scalar.output)
      << test.label << ": default result is not bit-identical to scalar";
  EXPECT_EQ(output_element(default_path, test, 0, 0), expected) << test.label << ": default D[0,0]";
  expect_unchanged_tail(test, default_path, initial);
  expect_unchanged_canaries(test, default_path, initial);
}

void seed_bf16f32_contract_state(SwmmacFixture &fx, std::vector<uint32_t> &expected) {
  fx.seed(A_OFF, INPUT_A_REGS, Fmt::BF16, Mode::Zeros, 0);
  fx.seed(B_OFF, INPUT_B_REGS, Fmt::BF16, Mode::Zeros, 0);
  seed_legal_indices(fx, 0);
  expected.assign(static_cast<size_t>(PACKED_ACC_REGS) * WF_SIZE, 0);
  std::vector<uint8_t> written(static_cast<size_t>(PACKED_ACC_REGS) * WF_SIZE, 0);
  constexpr std::array<uint32_t, 8> C_BITS{
      0x3F808000u, // halfway, retained BF16 LSB even: round down
      0x3F818000u, // halfway, retained BF16 LSB odd: round up
      0xBF808000u, // negative halfway, retained BF16 LSB even
      0xBF818000u, // negative halfway, retained BF16 LSB odd
      0x00008000u, // halfway to the smallest BF16 subnormal
      0x00018000u, // subnormal halfway with an odd retained LSB
      0x00808000u, // smallest-normal neighborhood, even retained LSB
      0x00818000u, // smallest-normal neighborhood, odd retained LSB
  };

  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      const float c = std::bit_cast<float>(C_BITS[(row * 16 + col) % C_BITS.size()]);
      const auto c_loc = amdgpu::wmma_output_loc_32(16, 16, row, col);
      fx.cu->write_vgpr(fx.vbase + ACC_OFF + c_loc.reg, c_loc.lane, std::bit_cast<uint32_t>(c));

      const auto d_loc = amdgpu::wmma_output_loc_16(16, 16, row, col);
      const size_t index = static_cast<size_t>(d_loc.reg) * WF_SIZE + d_loc.lane;
      const uint32_t shift = d_loc.sub_element * 16;
      expected[index] |= static_cast<uint32_t>(util::f32_to_bf16_rne(c)) << shift;
      written[index] |= static_cast<uint8_t>(1u << d_loc.sub_element);
    }
  }
  for (uint8_t mask : written)
    ASSERT_EQ(mask, 0x3u);
  seed_pattern(fx, ACC_OFF - 1, 1, 0x31);
  seed_pattern(fx, ACC_OFF + F32_ACC_REGS, 1, 0x32);
}

} // namespace

TEST(SwmmacK64SimdExact, AllDecodedFormsMatchScalarForLegalMetadata) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "the K=64 SWMMAC fast paths require 16-lane native SIMD";

  constexpr std::array MODES{Mode::RandomInt, Mode::Zeros, Mode::SignedZero, Mode::Cancel,
                             Mode::NaN,       Mode::Inf,   Mode::Denorm,     Mode::MaxFinite};
  uint32_t case_index = 0;
  for (const auto &test : SWMMAC_CASES) {
    for (uint32_t mode_index = 0; mode_index < MODES.size(); ++mode_index) {
      SCOPED_TRACE(::testing::Message()
                   << test.label << " mode=" << static_cast<int>(MODES[mode_index]));
      run_exact_case(test, MODES[mode_index], 100 + 17 * case_index + mode_index,
                     case_index + mode_index);
      if (testing::Test::HasFatalFailure())
        return;
    }
    ++case_index;
  }
}

TEST(SwmmacK64SimdExact, PackedF16SnapshotsAliasedInputsAndSupportsReuseHints) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "the K=64 SWMMAC fast paths require 16-lane native SIMD";

  const auto &test = SWMMAC_CASES[2];
  constexpr std::array ALIASES{DstAlias::A, DstAlias::B, DstAlias::Index};
  for (uint32_t i = 0; i < ALIASES.size(); ++i) {
    SCOPED_TRACE(::testing::Message() << "alias=" << static_cast<int>(ALIASES[i]));
    run_exact_case(test, Mode::RandomInt, 701 + i, 3 + i, ALIASES[i]);
  }
}

TEST(SwmmacK64SimdExact, Bf16f32ReadsEightF32CAndWritesFourPackedBf16D) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "the K=64 SWMMAC fast paths require 16-lane native SIMD";

  SwmmacFixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  std::vector<uint32_t> expected;
  seed_bf16f32_contract_state(fx, expected);
  auto instruction = decode_bf16f32();
  ASSERT_NE(instruction, nullptr);

  const auto initial = fx.snapshot(0, STATE_REGS);
  const auto &test = SWMMAC_CASES[4];
  ForceScalarGuard force_scalar_guard;
  const auto scalar = execute_from_state(fx, *instruction, initial, test, true);
  const auto simd = execute_from_state(fx, *instruction, initial, test, false);

  ASSERT_TRUE(scalar.succeeded);
  ASSERT_TRUE(simd.succeeded);
  EXPECT_EQ(scalar.output, expected);
  EXPECT_EQ(simd.output, expected);
  expect_unchanged_tail(test, scalar, initial);
  expect_unchanged_tail(test, simd, initial);
  expect_unchanged_canaries(test, scalar, initial);
  expect_unchanged_canaries(test, simd, initial);
}

TEST(SwmmacK64SimdExact, Bf16ResultRoundsHalfwayToEven) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "the K=64 SWMMAC fast paths require 16-lane native SIMD";

  SwmmacFixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  fx.seed(A_OFF, INPUT_A_REGS, Fmt::BF16, Mode::Zeros, 0);
  fx.seed(B_OFF, INPUT_B_REGS, Fmt::BF16, Mode::Zeros, 0);
  fx.seed(ACC_OFF, PACKED_ACC_REGS, Fmt::BF16, Mode::Zeros, 0);
  seed_pattern(fx, ACC_OFF + PACKED_ACC_REGS, F32_ACC_REGS - PACKED_ACC_REGS, 0x41);
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
    fx.cu->write_vgpr(fx.vbase + INDEX_OFF, lane, 0x44444444u);

  // 0x3f01 * 0x3f40 is exactly F32 0x3ec18000: halfway between BF16 0x3ec1
  // and 0x3ec2. The retained LSB is odd, so RNE must produce 0x3ec2.
  for (uint32_t row = 0; row < 16; ++row)
    write_packed16(fx, A_OFF, amdgpu::swmmac_a_input_loc(WF_SIZE, 16, 64, row, 0, 16), 0x3F01u);
  for (uint32_t col = 0; col < 16; ++col)
    write_packed16(fx, B_OFF, amdgpu::swmmac_b_input_loc(WF_SIZE, 16, 64, col, 0, 16), 0x3F40u);

  const auto &test = SWMMAC_CASES[3];
  const auto words = cdna5::build_vop3p(
      test.opcode,
      {.vdst = ACC_OFF, .src0 = 256 + A_OFF, .src1 = 256 + B_OFF, .src2 = 256 + INDEX_OFF});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
  ASSERT_NE(instruction, nullptr);

  const auto initial = fx.snapshot(0, STATE_REGS);
  ForceScalarGuard force_scalar_guard;
  const auto scalar = execute_from_state(fx, *instruction, initial, test, true);
  const auto simd = execute_from_state(fx, *instruction, initial, test, false);
  ASSERT_TRUE(scalar.succeeded);
  ASSERT_TRUE(simd.succeeded);
  const std::vector<uint32_t> expected(static_cast<size_t>(PACKED_ACC_REGS) * WF_SIZE, 0x3EC23EC2u);
  EXPECT_EQ(scalar.output, expected);
  EXPECT_EQ(simd.output, expected);
  expect_unchanged_tail(test, scalar, initial);
  expect_unchanged_tail(test, simd, initial);
}

TEST(SwmmacK64SimdExact, Bf16InputsUseFusedAccumulation) {
  struct FusedCase {
    size_t test_index;
    uint32_t initial_c;
    uint32_t expected;
  };
  constexpr std::array CASES{
      FusedCase{1, 0xFF7FFFFFu, 0x73800000u},
      FusedCase{3, 0x0000FF7Fu, 0x00007B80u},
      FusedCase{4, 0xFF7FFFFFu, 0x00007380u},
  };

  for (const auto &fused : CASES) {
    const auto &test = SWMMAC_CASES[fused.test_index];
    SCOPED_TRACE(test.label);
    SwmmacFixture fx;
    ASSERT_NE(fx.cu, nullptr);
    ASSERT_NE(fx.wf, nullptr);
    clear_state(fx);

    // BF16 0x5f80 is 2^64. A fused (2^64 * 2^64) - FLT_MAX
    // remains finite; split multiply/add first rounds the product to 2^128
    // and then produces +Inf.
    fx.cu->write_vgpr(fx.vbase + A_OFF, 0, 0x00005F80u);
    fx.cu->write_vgpr(fx.vbase + B_OFF, 0, 0x00005F80u);
    fx.cu->write_vgpr(fx.vbase + ACC_OFF, 0, fused.initial_c);

    auto instruction = decode_case(test);
    ASSERT_NE(instruction, nullptr);
    const auto initial = fx.snapshot(0, STATE_REGS);
    std::vector<uint32_t> expected(static_cast<size_t>(test.output_regs) * WF_SIZE, 0);
    expected[0] = fused.expected;
    expect_scalar_and_default(fx, *instruction, test, initial, expected, false);
  }
}

TEST(SwmmacK64SimdExact, SparseGatherMatchesIndependentPhysicalLayoutOracle) {
  for (const size_t test_index : {size_t{1}, size_t{3}, size_t{4}}) {
    const auto &test = SWMMAC_CASES[test_index];
    SCOPED_TRACE(test.label);
    SwmmacFixture fx;
    ASSERT_NE(fx.cu, nullptr);
    ASSERT_NE(fx.wf, nullptr);
    clear_state(fx);

    // Deliberately use raw physical locations rather than the layout helpers:
    // A reg5/lane5/high is row5, compressed-k19. Give its metadata location a
    // unique group-1 selector: index lane21 selects dense k39, while every
    // other group still selects (0,1). B reg11/lane7/high is k39, column7. The
    // only nonzero dot product is
    // therefore D[5,7] = 2 * 3 = 6.
    fx.cu->write_vgpr(fx.vbase + A_OFF + 5, 5, 0x40000000u);
    fx.cu->write_vgpr(fx.vbase + INDEX_OFF, 21, 0x444444C4u);
    fx.cu->write_vgpr(fx.vbase + B_OFF + 11, 7, 0x40400000u);

    auto instruction = decode_case(test);
    ASSERT_NE(instruction, nullptr);
    const auto initial = fx.snapshot(0, STATE_REGS);
    std::vector<uint32_t> expected(static_cast<size_t>(test.output_regs) * WF_SIZE, 0);
    if (test.output_kind == OutputKind::F32)
      expected[5 * WF_SIZE + 7] = 0x40C00000u;
    else
      expected[2 * WF_SIZE + 7] = 0x40C00000u;
    expect_scalar_and_default(fx, *instruction, test, initial, expected, false);
  }
}

TEST(SwmmacK64SimdExact, NaNPayloadPriorityIsBitExact) {
  struct NanCase {
    const char *label;
    uint16_t a0;
    uint16_t b0;
    uint16_t c;
    uint16_t a1;
    uint16_t b1;
    uint16_t expected;
  };
  constexpr std::array CASES{
      NanCase{"A qNaN wins over B and C", 0x7FC1u, 0x7FC2u, 0x7FC3u, 0, 0, 0x7FC1u},
      NanCase{"B qNaN wins over C", 0x3F80u, 0x7FC2u, 0x7FC3u, 0, 0, 0x7FC2u},
      NanCase{"C qNaN is retained", 0, 0, 0x7FC3u, 0, 0, 0x7FC3u},
      NanCase{"A sNaN is quieted", 0x7F81u, 0x3F80u, 0, 0, 0, 0x7FC1u},
      NanCase{"signed B qNaN is retained", 0x3F80u, 0xFFC4u, 0, 0, 0, 0xFFC4u},
      NanCase{"Inf times zero is canonical", 0x7F80u, 0, 0, 0, 0, 0xFFC0u},
      NanCase{"later B qNaN supersedes accumulated A qNaN", 0x7FC1u, 0x3F80u, 0, 0x3F80u, 0x7FC2u,
              0x7FC2u},
  };
  for (const auto &nan : CASES) {
    for (const size_t test_index : {size_t{1}, size_t{3}, size_t{4}}) {
      const auto &test = SWMMAC_CASES[test_index];
      SCOPED_TRACE(::testing::Message() << test.label << ": " << nan.label);
      SwmmacFixture fx;
      ASSERT_NE(fx.cu, nullptr);
      ASSERT_NE(fx.wf, nullptr);
      clear_state(fx);

      write_packed16(fx, A_OFF, amdgpu::swmmac_a_input_loc(WF_SIZE, 16, 64, 0, 0, 16), nan.a0);
      write_packed16(fx, B_OFF, amdgpu::swmmac_b_input_loc(WF_SIZE, 16, 64, 0, 0, 16), nan.b0);
      if (nan.a1 != 0 || nan.b1 != 0) {
        write_packed16(fx, A_OFF, amdgpu::swmmac_a_input_loc(WF_SIZE, 16, 64, 0, 1, 16), nan.a1);
        write_packed16(fx, B_OFF, amdgpu::swmmac_b_input_loc(WF_SIZE, 16, 64, 0, 1, 16), nan.b1);
      }
      fx.cu->write_vgpr(fx.vbase + ACC_OFF, 0,
                        test.accumulator_fmt == Fmt::F32 ? static_cast<uint32_t>(nan.c) << 16
                                                         : nan.c);

      auto instruction = decode_case(test);
      ASSERT_NE(instruction, nullptr);
      const auto initial = fx.snapshot(0, STATE_REGS);
      const uint32_t expected = test.output_kind == OutputKind::F32
                                    ? static_cast<uint32_t>(nan.expected) << 16
                                    : nan.expected;
      expect_scalar_and_default_element(fx, *instruction, test, initial, expected, false);
    }
  }
}

TEST(SwmmacK64SimdExact, F16NaNPayloadPriorityIsBitExact) {
  struct NanCase {
    const char *label;
    uint16_t a;
    uint16_t b;
    uint16_t c;
    uint32_t expected_f32;
    uint16_t expected_f16;
  };
  constexpr std::array CASES{
      NanCase{"A qNaN wins over B and C", 0x7E01u, 0x7E03u, 0x7E05u, 0x7FC02000u, 0x7E01u},
      NanCase{"B qNaN wins over C", 0x3C00u, 0x7E03u, 0x7E05u, 0x7FC06000u, 0x7E03u},
      NanCase{"C qNaN is retained", 0, 0, 0x7E05u, 0x7FC0A000u, 0x7E05u},
      NanCase{"A sNaN is quieted", 0x7C01u, 0x3C00u, 0, 0x7FC02000u, 0x7E01u},
      NanCase{"signed B qNaN is retained", 0x3C00u, 0xFE07u, 0, 0xFFC0E000u, 0xFE07u},
      NanCase{"Inf times zero is canonical", 0x7C00u, 0, 0, 0xFFC00000u, 0xFE01u},
  };

  for (const auto &nan : CASES) {
    for (const size_t test_index : {size_t{0}, size_t{2}}) {
      const auto &test = SWMMAC_CASES[test_index];
      SCOPED_TRACE(::testing::Message() << test.label << ": " << nan.label);
      SwmmacFixture fx;
      ASSERT_NE(fx.cu, nullptr);
      ASSERT_NE(fx.wf, nullptr);
      clear_state(fx);

      write_packed16(fx, A_OFF, amdgpu::swmmac_a_input_loc(WF_SIZE, 16, 64, 0, 0, 16), nan.a);
      write_packed16(fx, B_OFF, amdgpu::swmmac_b_input_loc(WF_SIZE, 16, 64, 0, 0, 16), nan.b);
      fx.cu->write_vgpr(fx.vbase + ACC_OFF, 0,
                        test.accumulator_fmt == Fmt::F32
                            ? std::bit_cast<uint32_t>(util::f16_to_f32(nan.c))
                            : nan.c);

      auto instruction = decode_case(test);
      ASSERT_NE(instruction, nullptr);
      const auto initial = fx.snapshot(0, STATE_REGS);
      const uint32_t expected =
          test.output_kind == OutputKind::F32 ? nan.expected_f32 : nan.expected_f16;
      expect_scalar_and_default_element(fx, *instruction, test, initial, expected, false);
    }
  }
}

TEST(SwmmacK64SimdExact, Bf16f32HonorsFp16OvflAndClampsInfinity) {
  SwmmacFixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  clear_state(fx);
  constexpr std::array C_BITS{
      0x7F7FFFFFu, // largest finite positive F32: RNE overflows BF16
      0xFF7FFFFFu, // largest finite negative F32: RNE overflows BF16
      0x7F800000u, // matrix FP16_OVFL clamps +Inf
      0xFF800000u, // matrix FP16_OVFL clamps -Inf
  };
  constexpr std::array CLEAR_RESULTS{uint16_t{0x7F80}, uint16_t{0xFF80}, uint16_t{0x7F80},
                                     uint16_t{0xFF80}};
  constexpr std::array SET_RESULTS{uint16_t{0x7F7F}, uint16_t{0xFF7F}, uint16_t{0x7F7F},
                                   uint16_t{0xFF7F}};
  std::vector<uint32_t> expected_clear(static_cast<size_t>(PACKED_ACC_REGS) * WF_SIZE, 0);
  std::vector<uint32_t> expected_set(static_cast<size_t>(PACKED_ACC_REGS) * WF_SIZE, 0);

  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      const size_t value_index = (row * 16 + col) % C_BITS.size();
      const auto c_loc = amdgpu::wmma_output_loc_32(16, 16, row, col);
      fx.cu->write_vgpr(fx.vbase + ACC_OFF + c_loc.reg, c_loc.lane, C_BITS[value_index]);
      const auto d_loc = amdgpu::wmma_output_loc_16(16, 16, row, col);
      const size_t word_index = static_cast<size_t>(d_loc.reg) * WF_SIZE + d_loc.lane;
      const uint32_t shift = d_loc.sub_element * 16;
      expected_clear[word_index] |= static_cast<uint32_t>(CLEAR_RESULTS[value_index]) << shift;
      expected_set[word_index] |= static_cast<uint32_t>(SET_RESULTS[value_index]) << shift;
    }
  }

  const auto &test = SWMMAC_CASES[4];
  auto instruction = decode_case(test);
  ASSERT_NE(instruction, nullptr);
  const auto initial = fx.snapshot(0, STATE_REGS);
  expect_scalar_and_default(fx, *instruction, test, initial, expected_clear, false);
  expect_scalar_and_default(fx, *instruction, test, initial, expected_set, true);
}

TEST(SwmmacK64SimdExact, Bf16f32ClampsFiniteArithmeticOverflow) {
  struct OverflowCase {
    const char *label;
    uint16_t a;
    uint16_t clear_result;
    uint16_t set_result;
  };
  constexpr std::array CASES{
      OverflowCase{"positive", 0x7F7Fu, 0x7F80u, 0x7F7Fu},
      OverflowCase{"negative", 0xFF7Fu, 0xFF80u, 0xFF7Fu},
  };
  const auto &test = SWMMAC_CASES[4];

  for (const auto &overflow : CASES) {
    SCOPED_TRACE(overflow.label);
    SwmmacFixture fx;
    ASSERT_NE(fx.cu, nullptr);
    ASSERT_NE(fx.wf, nullptr);
    clear_state(fx);
    write_packed16(fx, A_OFF, amdgpu::swmmac_a_input_loc(WF_SIZE, 16, 64, 0, 0, 16), overflow.a);
    write_packed16(fx, B_OFF, amdgpu::swmmac_b_input_loc(WF_SIZE, 16, 64, 0, 0, 16), 0x4000u);
    auto instruction = decode_case(test);
    ASSERT_NE(instruction, nullptr);
    const auto initial = fx.snapshot(0, STATE_REGS);
    expect_scalar_and_default_element(fx, *instruction, test, initial, overflow.clear_result,
                                      false);
    expect_scalar_and_default_element(fx, *instruction, test, initial, overflow.set_result, true);
  }
}

TEST(SwmmacK64SimdExact, PackedF16HonorsFp16OvflAndClampsInfinity) {
  struct OverflowCase {
    const char *label;
    uint16_t a;
    uint16_t b;
    uint16_t clear_result;
    uint16_t set_result;
  };
  constexpr std::array CASES{
      OverflowCase{"positive finite overflow", 0x7BFFu, 0x4000u, 0x7C00u, 0x7BFFu},
      OverflowCase{"negative finite overflow", 0xFBFFu, 0x4000u, 0xFC00u, 0xFBFFu},
      OverflowCase{"positive infinity", 0x7C00u, 0x3C00u, 0x7C00u, 0x7BFFu},
      OverflowCase{"negative infinity", 0xFC00u, 0x3C00u, 0xFC00u, 0xFBFFu},
  };
  const auto &test = SWMMAC_CASES[2];

  for (const auto &overflow : CASES) {
    SCOPED_TRACE(overflow.label);
    SwmmacFixture fx;
    ASSERT_NE(fx.cu, nullptr);
    ASSERT_NE(fx.wf, nullptr);
    clear_state(fx);
    write_packed16(fx, A_OFF, amdgpu::swmmac_a_input_loc(WF_SIZE, 16, 64, 0, 0, 16), overflow.a);
    write_packed16(fx, B_OFF, amdgpu::swmmac_b_input_loc(WF_SIZE, 16, 64, 0, 0, 16), overflow.b);
    auto instruction = decode_case(test);
    ASSERT_NE(instruction, nullptr);
    const auto initial = fx.snapshot(0, STATE_REGS);
    expect_scalar_and_default_element(fx, *instruction, test, initial, overflow.clear_result,
                                      false);
    expect_scalar_and_default_element(fx, *instruction, test, initial, overflow.set_result, true);
  }
}

TEST(SwmmacK64SimdExact, PackedBf16HonorsFp16OvflAndClampsInfinity) {
  struct OverflowCase {
    const char *label;
    uint16_t a;
    uint16_t b;
    uint16_t c;
    uint16_t clear_result;
    uint16_t set_result;
  };
  constexpr std::array CASES{
      // max BF16 + 2^60*2^59 is the exact RNE boundary between max BF16
      // and infinity. The odd retained LSB rounds toward infinity.
      OverflowCase{"positive finite overflow", 0x5D80u, 0x5D00u, 0x7F7Fu, 0x7F80u, 0x7F7Fu},
      OverflowCase{"negative finite overflow", 0xDD80u, 0x5D00u, 0xFF7Fu, 0xFF80u, 0xFF7Fu},
      OverflowCase{"positive arithmetic overflow", 0x7F7Fu, 0x4000u, 0, 0x7F80u, 0x7F7Fu},
      OverflowCase{"negative arithmetic overflow", 0xFF7Fu, 0x4000u, 0, 0xFF80u, 0xFF7Fu},
      OverflowCase{"positive infinity", 0, 0, 0x7F80u, 0x7F80u, 0x7F7Fu},
      OverflowCase{"negative infinity", 0, 0, 0xFF80u, 0xFF80u, 0xFF7Fu},
  };
  const auto &test = SWMMAC_CASES[3];

  for (const auto &overflow : CASES) {
    SCOPED_TRACE(overflow.label);
    SwmmacFixture fx;
    ASSERT_NE(fx.cu, nullptr);
    ASSERT_NE(fx.wf, nullptr);
    clear_state(fx);
    write_packed16(fx, A_OFF, amdgpu::swmmac_a_input_loc(WF_SIZE, 16, 64, 0, 0, 16), overflow.a);
    write_packed16(fx, B_OFF, amdgpu::swmmac_b_input_loc(WF_SIZE, 16, 64, 0, 0, 16), overflow.b);
    fx.cu->write_vgpr(fx.vbase + ACC_OFF, 0, overflow.c);
    auto instruction = decode_case(test);
    ASSERT_NE(instruction, nullptr);
    const auto initial = fx.snapshot(0, STATE_REGS);
    std::vector<uint32_t> expected(static_cast<size_t>(PACKED_ACC_REGS) * WF_SIZE, 0);
    expected[0] = overflow.clear_result;
    expect_scalar_and_default(fx, *instruction, test, initial, expected, false);
    expected[0] = overflow.set_result;
    expect_scalar_and_default(fx, *instruction, test, initial, expected, true);
  }
}

TEST(SwmmacSimdExact, K128DecodedF16HonorsFp16OvflAndClampsInfinity) {
  constexpr SwmmacCase test{
      cdna5::kVSwmmacF1616x16x128Bf8Bf8Vop3p,
      "v_swmmac_f16_16x16x128_bf8_bf8",
      Fmt::BF8,
      Fmt::F16,
      OutputKind::F16,
      PACKED_ACC_REGS,
      PACKED_ACC_REGS,
  };

  SwmmacFixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  clear_state(fx);
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    fx.cu->write_vgpr(fx.vbase + INDEX_OFF, lane, 0x44444444u);
    fx.cu->write_vgpr(fx.vbase + INDEX_OFF + 1, lane, 0x44444444u);
  }

  // BF8 +Inf * 1.0 produces +Inf in D[0,0]. The decoded instruction must
  // preserve infinity with MODE.FP16_OVFL clear and clamp it to MAX_F16 when
  // the mode bit is set.
  write_packed8(fx, A_OFF, amdgpu::swmmac_a_input_loc(WF_SIZE, 16, 128, 0, 0, 8), 0x7Cu);
  write_packed8(fx, B_OFF, amdgpu::swmmac_b_input_loc(WF_SIZE, 16, 128, 0, 0, 8), 0x3Cu);

  auto instruction = decode_case(test);
  ASSERT_NE(instruction, nullptr);
  const auto initial = fx.snapshot(0, STATE_REGS);
  expect_scalar_and_default_element(fx, *instruction, test, initial, 0x7C00u, false);
  expect_scalar_and_default_element(fx, *instruction, test, initial, 0x7BFFu, true);
}
