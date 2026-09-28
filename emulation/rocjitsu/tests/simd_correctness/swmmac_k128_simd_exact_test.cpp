// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file swmmac_k128_simd_exact_test.cpp
/// @brief Decoded scalar-vs-AVX-512 checks for gfx1250 K=128 FP8/BF8 SWMMAC.

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
constexpr uint32_t D_OFF = 64;
constexpr uint32_t INDEX_OFF = 96;
constexpr uint32_t STATE_REGS = 128;
constexpr uint32_t A_REGS = 8;
constexpr uint32_t B_REGS = 16;
constexpr uint32_t INDEX_REGS = 2;
constexpr uint32_t F32_D_REGS = 8;
constexpr uint32_t F16_D_REGS = 4;
constexpr std::array<uint32_t, 6> LEGAL_PAIRS{0x4u, 0x8u, 0xCu, 0x9u, 0xDu, 0xEu};

enum class DstAlias { None, A, B, Index };

struct SwmmacCase {
  uint16_t opcode;
  const char *name;
  Fmt a_fmt;
  Fmt b_fmt;
  bool f32_result;

  uint32_t dst_regs() const { return f32_result ? F32_D_REGS : F16_D_REGS; }
};

constexpr std::array CASES{
    SwmmacCase{cdna5::kVSwmmacF3216x16x128Fp8Fp8Vop3p, "v_swmmac_f32_16x16x128_fp8_fp8", Fmt::FP8,
               Fmt::FP8, true},
    SwmmacCase{cdna5::kVSwmmacF3216x16x128Fp8Bf8Vop3p, "v_swmmac_f32_16x16x128_fp8_bf8", Fmt::FP8,
               Fmt::BF8, true},
    SwmmacCase{cdna5::kVSwmmacF3216x16x128Bf8Fp8Vop3p, "v_swmmac_f32_16x16x128_bf8_fp8", Fmt::BF8,
               Fmt::FP8, true},
    SwmmacCase{cdna5::kVSwmmacF3216x16x128Bf8Bf8Vop3p, "v_swmmac_f32_16x16x128_bf8_bf8", Fmt::BF8,
               Fmt::BF8, true},
    SwmmacCase{cdna5::kVSwmmacF1616x16x128Fp8Fp8Vop3p, "v_swmmac_f16_16x16x128_fp8_fp8", Fmt::FP8,
               Fmt::FP8, false},
    SwmmacCase{cdna5::kVSwmmacF1616x16x128Fp8Bf8Vop3p, "v_swmmac_f16_16x16x128_fp8_bf8", Fmt::FP8,
               Fmt::BF8, false},
    SwmmacCase{cdna5::kVSwmmacF1616x16x128Bf8Fp8Vop3p, "v_swmmac_f16_16x16x128_bf8_fp8", Fmt::BF8,
               Fmt::FP8, false},
    SwmmacCase{cdna5::kVSwmmacF1616x16x128Bf8Bf8Vop3p, "v_swmmac_f16_16x16x128_bf8_bf8", Fmt::BF8,
               Fmt::BF8, false},
};

struct SwmmacFixture : ExactFixture {
  SwmmacFixture() : ExactFixture(ROCJITSU_CODE_ARCH_CDNA5, WF_SIZE) {}
};

struct Observation {
  bool succeeded;
  std::vector<uint32_t> state;
};

void restore(SwmmacFixture &fx, const std::vector<uint32_t> &words) {
  ASSERT_EQ(words.size(), static_cast<size_t>(STATE_REGS) * WF_SIZE);
  for (uint32_t reg = 0; reg < STATE_REGS; ++reg)
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      fx.cu->write_vgpr(fx.vbase + reg, lane, words[static_cast<size_t>(reg) * WF_SIZE + lane]);
}

void clear_state(SwmmacFixture &fx) {
  restore(fx, std::vector<uint32_t>(static_cast<size_t>(STATE_REGS) * WF_SIZE, 0));
}

void seed_indices(SwmmacFixture &fx, uint32_t off, uint32_t phase) {
  // A 32-entry index set occupies two VGPRs per lane. Exercise every legal
  // ordered 2:4 pair in each half, including the upper register.
  for (uint32_t reg = 0; reg < INDEX_REGS; ++reg)
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint32_t word = 0;
      for (uint32_t group = 0; group < 8; ++group)
        word |= LEGAL_PAIRS[(phase + lane + reg * 3 + group) % LEGAL_PAIRS.size()] << (4 * group);
      fx.cu->write_vgpr(fx.vbase + off + reg, lane, word);
    }
}

void seed_constant_indices(SwmmacFixture &fx) {
  for (uint32_t reg = 0; reg < INDEX_REGS; ++reg)
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      fx.cu->write_vgpr(fx.vbase + INDEX_OFF + reg, lane, 0x44444444u);
}

void write_byte(SwmmacFixture &fx, uint32_t off, uint32_t reg, uint32_t lane, uint32_t byte,
                uint8_t value) {
  const uint32_t address = fx.vbase + off + reg;
  const uint32_t shift = 8 * byte;
  const uint32_t old = fx.cu->read_vgpr(address, lane);
  fx.cu->write_vgpr(address, lane,
                    (old & ~(0xFFu << shift)) | (static_cast<uint32_t>(value) << shift));
}

std::unique_ptr<Instruction> decode_case(const SwmmacCase &test, uint32_t a_off = A_OFF,
                                         uint32_t b_off = B_OFF, uint32_t index_off = INDEX_OFF) {
  const auto words =
      cdna5::build_vop3p(test.opcode, {.vdst = D_OFF,
                                       .src0 = static_cast<uint16_t>(256 + a_off),
                                       .src1 = static_cast<uint16_t>(256 + b_off),
                                       .src2 = static_cast<uint16_t>(256 + index_off)});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  if (!decoder)
    return nullptr;
  return std::unique_ptr<Instruction>(decode_valid(*decoder, words.data()));
}

Observation execute_from_state(SwmmacFixture &fx, Instruction &instruction,
                               const std::vector<uint32_t> &initial, bool force_scalar) {
  restore(fx, initial);
  util::set_force_scalar_for_testing(force_scalar);
  const bool succeeded = fx.cu->execute_instruction(&instruction, *fx.wf).succeeded();
  return {succeeded, fx.snapshot(0, STATE_REGS)};
}

void expect_equal_and_bounded(const SwmmacCase &test, const std::vector<uint32_t> &initial,
                              const Observation &scalar, const Observation &simd) {
  ASSERT_TRUE(scalar.succeeded) << test.name << ": forced-scalar execution failed";
  ASSERT_TRUE(simd.succeeded) << test.name << ": default execution failed";
  for (uint32_t reg = 0; reg < STATE_REGS; ++reg)
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const size_t i = static_cast<size_t>(reg) * WF_SIZE + lane;
      ASSERT_EQ(simd.state[i], scalar.state[i])
          << test.name << ": SIMD mismatch at reg=" << reg << " lane=" << lane;
      if (reg < D_OFF || reg >= D_OFF + test.dst_regs()) {
        ASSERT_EQ(scalar.state[i], initial[i])
            << test.name << ": scalar changed source or canary reg=" << reg << " lane=" << lane;
        ASSERT_EQ(simd.state[i], initial[i])
            << test.name << ": SIMD changed source or canary reg=" << reg << " lane=" << lane;
      }
    }
}

void expect_output(const SwmmacCase &test, const Observation &observation,
                   const std::vector<uint32_t> &expected) {
  const auto first = observation.state.begin() + static_cast<size_t>(D_OFF) * WF_SIZE;
  const std::vector<uint32_t> actual(first, first + static_cast<size_t>(test.dst_regs()) * WF_SIZE);
  EXPECT_EQ(actual, expected) << test.name << ": independent expected output";
}

uint32_t first_output(const SwmmacCase &test, const Observation &observation) {
  const uint32_t word = observation.state[static_cast<size_t>(D_OFF) * WF_SIZE];
  return test.f32_result ? word : word & 0xFFFFu;
}

void run_case(const SwmmacCase &test, Mode mode, uint32_t seed, uint32_t phase,
              DstAlias alias = DstAlias::None) {
  SwmmacFixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  clear_state(fx);

  const uint32_t a_off = alias == DstAlias::A ? D_OFF : A_OFF;
  const uint32_t b_off = alias == DstAlias::B ? D_OFF : B_OFF;
  const uint32_t index_off = alias == DstAlias::Index ? D_OFF : INDEX_OFF;

  // The decoded instruction reads C from its tied D tuple. Seed it first so
  // source-overlap trials can replace the same physical words afterward.
  fx.seed(D_OFF, test.dst_regs(), test.f32_result ? Fmt::F32 : Fmt::F16, Mode::RandomInt, seed + 3);
  fx.seed(a_off, A_REGS, test.a_fmt, mode, seed + 1);
  fx.seed(b_off, B_REGS, test.b_fmt, mode, seed + 2);
  seed_indices(fx, index_off, phase);
  fx.seed_words(D_OFF - 1, 1, seed + 4);
  fx.seed_words(D_OFF + B_REGS, 1, seed + 5);

  auto instruction = decode_case(test, a_off, b_off, index_off);
  ASSERT_NE(instruction, nullptr) << test.name;
  const auto initial = fx.snapshot(0, STATE_REGS);
  ForceScalarGuard guard;
  const auto scalar = execute_from_state(fx, *instruction, initial, true);
  const auto simd = execute_from_state(fx, *instruction, initial, false);
  expect_equal_and_bounded(test, initial, scalar, simd);
}

uint8_t one(Fmt fmt) { return fmt == Fmt::FP8 ? 0x38u : 0x3Cu; }
uint8_t nan(Fmt fmt, bool negative) {
  const uint8_t positive = fmt == Fmt::FP8 ? 0x7Fu : 0x7Eu;
  return negative ? static_cast<uint8_t>(positive | 0x80u) : positive;
}

float decode_eight_bit(Fmt fmt, uint8_t raw) {
  return fmt == Fmt::FP8 ? util::fp8_e4m3_ocp_to_f32(raw) : util::bf8_e5m2_ocp_to_f32(raw);
}

uint32_t result_bits(const SwmmacCase &test, float value) {
  return test.f32_result ? std::bit_cast<uint32_t>(value) : util::f32_to_f16(value);
}

} // namespace

TEST(SwmmacK128SimdExact, AllEightDecodedFormsMatchForcedScalar) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "K=128 SWMMAC requires 16-lane native SIMD";

  constexpr std::array MODES{Mode::RandomInt, Mode::Zeros, Mode::SignedZero, Mode::Cancel,
                             Mode::NaN,       Mode::Inf,   Mode::Denorm,     Mode::MaxFinite};
  for (size_t case_index = 0; case_index < CASES.size(); ++case_index) {
    const auto &test = CASES[case_index];
    for (size_t mode_index = 0; mode_index < MODES.size(); ++mode_index) {
      const Mode mode = MODES[mode_index];
      if (mode == Mode::Inf && (!fmt_has_inf(test.a_fmt) || !fmt_has_inf(test.b_fmt)))
        continue;
      SCOPED_TRACE(::testing::Message() << test.name << " mode=" << static_cast<int>(mode));
      run_case(test, mode, 101 + 17 * case_index + mode_index, case_index + mode_index);
      if (testing::Test::HasFatalFailure())
        return;
    }
  }
}

TEST(SwmmacK128SimdExact, DestinationAliasesA_BAndBothMetadataWords) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "K=128 SWMMAC requires 16-lane native SIMD";

  constexpr std::array ALIASES{DstAlias::A, DstAlias::B, DstAlias::Index};
  for (size_t case_index = 0; case_index < CASES.size(); ++case_index)
    for (size_t alias_index = 0; alias_index < ALIASES.size(); ++alias_index) {
      const auto &test = CASES[case_index];
      SCOPED_TRACE(::testing::Message()
                   << test.name << " alias=" << static_cast<int>(ALIASES[alias_index]));
      run_case(test, Mode::RandomInt, 701 + 17 * case_index + alias_index, case_index + alias_index,
               ALIASES[alias_index]);
      if (testing::Test::HasFatalFailure())
        return;
    }
}

TEST(SwmmacK128SimdExact, TiedCIsReadFromD) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "K=128 SWMMAC requires 16-lane native SIMD";

  for (const auto &test : CASES) {
    SCOPED_TRACE(test.name);
    SwmmacFixture fx;
    ASSERT_NE(fx.cu, nullptr);
    ASSERT_NE(fx.wf, nullptr);
    clear_state(fx);
    seed_constant_indices(fx);
    const uint32_t c_word = test.f32_result ? 0x3F800000u : 0x3C003C00u;
    for (uint32_t reg = 0; reg < test.dst_regs(); ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        fx.cu->write_vgpr(fx.vbase + D_OFF + reg, lane, c_word);
    auto instruction = decode_case(test);
    ASSERT_NE(instruction, nullptr);
    const auto initial = fx.snapshot(0, STATE_REGS);
    ForceScalarGuard guard;
    const auto scalar = execute_from_state(fx, *instruction, initial, true);
    const auto simd = execute_from_state(fx, *instruction, initial, false);
    expect_equal_and_bounded(test, initial, scalar, simd);
    const std::vector<uint32_t> expected(static_cast<size_t>(test.dst_regs()) * WF_SIZE, c_word);
    expect_output(test, scalar, expected);
    expect_output(test, simd, expected);
  }
}

TEST(SwmmacK128SimdExact, UpperMetadataAndPhysicalGatherMatchIndependentOracle) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "K=128 SWMMAC requires 16-lane native SIMD";

  for (const auto &test : CASES) {
    SCOPED_TRACE(test.name);
    SwmmacFixture fx;
    ASSERT_NE(fx.cu, nullptr);
    ASSERT_NE(fx.wf, nullptr);
    clear_state(fx);
    seed_constant_indices(fx);

    // Raw physical locations, deliberately independent of the layout helpers:
    // A reg7/lane21/byte1 is row5, compressed K=61. Its selector is entry29
    // of index lane21, stored in the upper metadata register at nibble6.
    // Changing that pair to (0,3) selects dense K=123. B reg14/lane23/byte3
    // is dense K=123, column7. The only product is D[5,7] = 2 * 3 = 6.
    write_byte(fx, A_OFF, 7, 21, 1, 0x40u);
    write_byte(fx, B_OFF, 14, 23, 3, test.b_fmt == Fmt::FP8 ? 0x44u : 0x42u);
    const uint32_t old = fx.cu->read_vgpr(fx.vbase + INDEX_OFF + 1, 21);
    fx.cu->write_vgpr(fx.vbase + INDEX_OFF + 1, 21, (old & ~(0xFu << 24)) | (0xCu << 24));

    auto instruction = decode_case(test);
    ASSERT_NE(instruction, nullptr);
    const auto initial = fx.snapshot(0, STATE_REGS);
    ForceScalarGuard guard;
    const auto scalar = execute_from_state(fx, *instruction, initial, true);
    const auto simd = execute_from_state(fx, *instruction, initial, false);
    expect_equal_and_bounded(test, initial, scalar, simd);
    std::vector<uint32_t> expected(static_cast<size_t>(test.dst_regs()) * WF_SIZE, 0);
    // gfx1250 wave32: F32 row5/col7 is D reg5/lane7; packed F16 is
    // D reg2/lane7/high half. These coordinates are intentionally literal.
    if (test.f32_result)
      expected[5 * WF_SIZE + 7] = 0x40C00000u;
    else
      expected[2 * WF_SIZE + 7] = 0x46000000u;
    expect_output(test, scalar, expected);
    expect_output(test, simd, expected);
  }
}

TEST(SwmmacK128SimdExact, NaNSourcePriorityIsBitExact) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "K=128 SWMMAC requires 16-lane native SIMD";

  enum class Source { A, B, C };
  for (const auto &test : CASES)
    for (Source source : {Source::A, Source::B, Source::C}) {
      SCOPED_TRACE(::testing::Message() << test.name << " NaN source=" << static_cast<int>(source));
      SwmmacFixture fx;
      ASSERT_NE(fx.cu, nullptr);
      ASSERT_NE(fx.wf, nullptr);
      clear_state(fx);
      seed_constant_indices(fx);
      // Distinct signs and a C payload make A, then B, then C priority
      // observable after exceptional-lane replay.
      const uint8_t a = source == Source::A ? nan(test.a_fmt, true) : one(test.a_fmt);
      const uint8_t b = source == Source::A   ? nan(test.b_fmt, false)
                        : source == Source::B ? nan(test.b_fmt, true)
                                              : one(test.b_fmt);
      write_byte(fx, A_OFF, 0, 0, 0, a);
      write_byte(fx, B_OFF, 0, 0, 0, b);
      fx.cu->write_vgpr(fx.vbase + D_OFF, 0, test.f32_result ? 0x7FC12345u : 0x00007E05u);

      float selected = 0.0f;
      if (source == Source::A)
        selected = decode_eight_bit(test.a_fmt, a);
      else if (source == Source::B)
        selected = decode_eight_bit(test.b_fmt, b);
      else
        selected = test.f32_result ? std::bit_cast<float>(0x7FC12345u) : util::f16_to_f32(0x7E05u);
      selected = std::bit_cast<float>(std::bit_cast<uint32_t>(selected) | 0x00400000u);
      const uint32_t expected = result_bits(test, selected);

      auto instruction = decode_case(test);
      ASSERT_NE(instruction, nullptr);
      const auto initial = fx.snapshot(0, STATE_REGS);
      ForceScalarGuard guard;
      const auto scalar = execute_from_state(fx, *instruction, initial, true);
      const auto simd = execute_from_state(fx, *instruction, initial, false);
      expect_equal_and_bounded(test, initial, scalar, simd);
      EXPECT_EQ(first_output(test, scalar), expected) << test.name << ": scalar NaN payload";
      EXPECT_EQ(first_output(test, simd), expected) << test.name << ": SIMD NaN payload";
    }
}

TEST(SwmmacK128SimdExact, PackedF16RoundsHalfwayToEven) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "K=128 SWMMAC requires 16-lane native SIMD";

  for (size_t case_index = 4; case_index < CASES.size(); ++case_index) {
    const auto &test = CASES[case_index];
    SCOPED_TRACE(test.name);
    SwmmacFixture fx;
    ASSERT_NE(fx.cu, nullptr);
    ASSERT_NE(fx.wf, nullptr);
    clear_state(fx);
    seed_constant_indices(fx);

    // Row0 starts at 1.0 (even retained bit), row1 at the next half value
    // (odd retained bit). One product of 2^-3 * 2^-8 = 2^-11 puts each result
    // exactly halfway between adjacent F16 values. Both share D reg0/lane0.
    fx.cu->write_vgpr(fx.vbase + D_OFF, 0, 0x3C013C00u);
    const uint8_t a = test.a_fmt == Fmt::FP8 ? 0x20u : 0x30u;
    const uint8_t b = test.b_fmt == Fmt::FP8 ? 0x02u : 0x1Cu;
    write_byte(fx, A_OFF, 0, 0, 0, a);
    write_byte(fx, A_OFF, 0, 1, 0, a);
    write_byte(fx, B_OFF, 0, 0, 0, b);

    auto instruction = decode_case(test);
    ASSERT_NE(instruction, nullptr);
    const auto initial = fx.snapshot(0, STATE_REGS);
    ForceScalarGuard guard;
    const auto scalar = execute_from_state(fx, *instruction, initial, true);
    const auto simd = execute_from_state(fx, *instruction, initial, false);
    expect_equal_and_bounded(test, initial, scalar, simd);
    EXPECT_EQ(scalar.state[static_cast<size_t>(D_OFF) * WF_SIZE], 0x3C023C00u);
    EXPECT_EQ(simd.state[static_cast<size_t>(D_OFF) * WF_SIZE], 0x3C023C00u);
  }
}
