// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file wmma_simd_exact_test.cpp
/// @brief Expensive bit-exact SIMD-vs-scalar checks for every gfx1250
/// WMMA/SWMMAC execute kernel and shape (dense, sparse, packed16, i32, mixed,
/// scaled, specialized). Built only with -DRJ_ENABLE_EXPENSIVE_CHECKS=ON.
///
/// Same scheme as the MFMA suite: identical pre-seeded VGPR state through the
/// forced-scalar path then the SIMD path, dst compared word for word, inputs
/// rounding-free (see mma_exact_test_support.h). The WMMA kernels accumulate
/// in-place (dst == acc), so the accumulator window is reseeded before each
/// run.

#include "mma_exact_test_support.h"

namespace {

using namespace rocjitsu;
using namespace mma_exact;

constexpr uint32_t WF = 32;
constexpr uint32_t S0 = 0, S1 = 32, ACC = 64, INDEX = 96;
constexpr uint32_t IN_REGS = 16, ACC_REGS = 8, INDEX_REGS = 4;
constexpr uint32_t BF16_IN_REGS = 8, BF16_DST = 112, BF16_DST_REGS = 4;
constexpr uint32_t INDEX_KEY = 0;
constexpr uint32_t CONST_ONE = 0x3F800000u;
constexpr uint32_t SCALE_A = 100, SCALE_B = 104;

using WmmaF32SpecFn = void (*)(amdgpu::ComputeUnitCore &, uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t);
// The trailing bool is MODE.FP16_OVFL. These cases compare the SIMD fast path
// against the scalar reference, which the mode does not distinguish, so every
// call below leaves it clear.
using WmmaF16SpecFn = void (*)(amdgpu::ComputeUnitCore &, uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, bool);

struct WmmaFixture : ExactFixture {
  WmmaFixture() : ExactFixture(ROCJITSU_CODE_ARCH_CDNA5, WF) {}
};

// Drive one (kernel, fmt) case across all trial modes and both accumulator
// sources. dst == acc, so reseed_acc restores the window between runs.
void run_case(const char *label, Fmt fmt, Fmt acc_fmt,
              const std::function<void(WmmaFixture &, uint32_t)> &kernel) {
  WmmaFixture fx;
  ASSERT_NE(fx.wf, nullptr);
  for (auto [mode, seed] : trials_for(fmt)) {
    fx.seed(S0, IN_REGS, fmt, mode, seed + 1);
    fx.seed(S1, IN_REGS, fmt, mode, seed + 2);
    fx.seed_words(INDEX, INDEX_REGS, seed + 4); // any 2-bit index pattern is valid
    auto reseed_acc = [&] { fx.seed(ACC, ACC_REGS, acc_fmt, Mode::RandomInt, seed + 3); };
    for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, CONST_ONE}) {
      expect_bit_exact(label, mode, fx, reseed_acc, [&] { kernel(fx, const_acc); }, ACC, ACC_REGS);
      if (testing::Test::HasFatalFailure())
        return;
    }
  }
}

template <typename Ea, typename Eb>
void run_dense_f32(const char *label, Fmt fmt, uint32_t M, uint32_t N, uint32_t K, uint32_t bits,
                   Ea ea, Eb eb) {
  run_case(label, fmt, Fmt::F32, [=](WmmaFixture &fx, uint32_t const_acc) {
    amdgpu::exec_wmma_f32(*fx.cu, M, N, K, bits, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                          fx.vbase + ACC, ea, eb, const_acc);
  });
}

template <typename Ea, typename Eb>
void run_dense_f16(const char *label, Fmt fmt, uint32_t K, uint32_t bits, Ea ea, Eb eb) {
  run_case(label, fmt, Fmt::F16, [=](WmmaFixture &fx, uint32_t const_acc) {
    amdgpu::exec_wmma_f16(*fx.cu, 16, 16, K, bits, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                          fx.vbase + ACC, ea, eb, const_acc);
  });
}

template <typename Ea, typename Eb>
void run_sparse_f32(const char *label, Fmt fmt, uint32_t K, uint32_t bits, uint32_t index_entries,
                    Ea ea, Eb eb) {
  run_case(label, fmt, Fmt::F32, [=](WmmaFixture &fx, uint32_t const_acc) {
    amdgpu::exec_swmmac_f32(*fx.cu, 16, 16, K, bits, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                            fx.vbase + ACC, fx.vbase + INDEX, index_entries, INDEX_KEY, ea, eb,
                            const_acc);
  });
}

template <typename Ea, typename Eb>
void run_sparse_f16(const char *label, Fmt fmt, uint32_t K, uint32_t bits, uint32_t index_entries,
                    Ea ea, Eb eb) {
  run_case(label, fmt, Fmt::F16, [=](WmmaFixture &fx, uint32_t const_acc) {
    amdgpu::exec_swmmac_f16(*fx.cu, 16, 16, K, bits, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                            fx.vbase + ACC, fx.vbase + INDEX, index_entries, INDEX_KEY, ea, eb,
                            const_acc);
  });
}

} // namespace

// --- dense f32-out, f16 inputs (generic + both specialized kernels) ---
TEST(WmmaSimdExact, F32_f16) {
  SKIP_IF_NO_SIMD();
  run_dense_f32("wmma_f32_16x16x16_f16", Fmt::F16, 16, 16, 16, 16, amdgpu::extract_f16,
                amdgpu::extract_f16);
  run_dense_f32("wmma_f32_16x16x32_f16", Fmt::F16, 16, 16, 32, 16, amdgpu::extract_f16,
                amdgpu::extract_f16);
  run_case("wmma_f32_16x16x32_f16_spec", Fmt::F16, Fmt::F32, [](WmmaFixture &fx, uint32_t ca) {
    amdgpu::exec_wmma_f32_16x16x32_f16(*fx.cu, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                                       fx.vbase + ACC, ca);
  });
  run_case("wmma_f32_16x16x4_f32_spec", Fmt::F32, Fmt::F32, [](WmmaFixture &fx, uint32_t ca) {
    amdgpu::exec_wmma_f32_f32_spec<16, 16, 4>(*fx.cu, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                                              fx.vbase + ACC, ca);
  });
}

// --- dense f16-out (packed16, generic + specialized) ---
TEST(WmmaSimdExact, F16_f16) {
  SKIP_IF_NO_SIMD();
  run_dense_f16("wmma_f16_16x16x16_f16", Fmt::F16, 16, 16, amdgpu::extract_f16,
                amdgpu::extract_f16);
  run_dense_f16("wmma_f16_16x16x32_f16", Fmt::F16, 32, 16, amdgpu::extract_f16,
                amdgpu::extract_f16);
  run_case("wmma_f16_16x16x32_f16_spec", Fmt::F16, Fmt::F16, [](WmmaFixture &fx, uint32_t ca) {
    amdgpu::exec_wmma_f16_spec<16, 16, 32>(*fx.cu, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                                           fx.vbase + ACC, ca);
  });
}

// --- dense bf16 inputs: f32-out and packed bf16-out ---
TEST(WmmaSimdExact, Bf16) {
  SKIP_IF_NO_SIMD();
  run_dense_f32("wmma_f32_16x16x32_bf16", Fmt::BF16, 16, 16, 32, 16, amdgpu::extract_bf16,
                amdgpu::extract_bf16);
  run_case("wmma_bf16_16x16x32_bf16", Fmt::BF16, Fmt::BF16, [](WmmaFixture &fx, uint32_t ca) {
    amdgpu::exec_wmma_bf16(*fx.cu, 16, 16, 32, 16, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                           fx.vbase + ACC, amdgpu::extract_bf16, amdgpu::extract_bf16, ca);
  });
  run_case("wmma_f32_16x16x32_bf16_spec", Fmt::BF16, Fmt::F32, [](WmmaFixture &fx, uint32_t ca) {
    amdgpu::exec_wmma_f32_16x16x32_bf16(*fx.cu, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                                        fx.vbase + ACC, ca);
  });
  run_case("wmma_bf16_16x16x32_bf16_spec", Fmt::BF16, Fmt::BF16, [](WmmaFixture &fx, uint32_t ca) {
    amdgpu::exec_wmma_bf16_spec<16, 16, 32>(*fx.cu, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                                            fx.vbase + ACC, ca);
  });
  run_sparse_f32("swmmac_f32_16x16x64_bf16", Fmt::BF16, 64, 16, 16, amdgpu::extract_bf16,
                 amdgpu::extract_bf16);
  run_case("swmmac_bf16_16x16x64_bf16", Fmt::BF16, Fmt::BF16, [](WmmaFixture &fx, uint32_t ca) {
    amdgpu::exec_swmmac_bf16(*fx.cu, 16, 16, 64, 16, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                             fx.vbase + ACC, fx.vbase + INDEX, 16, INDEX_KEY, amdgpu::extract_bf16,
                             amdgpu::extract_bf16, ca);
  });
}

// The BF16F32 opcode is mixed-width: A/B are eight packed-bf16 VGPRs, C is
// eight f32 VGPRs, and D is four packed-bf16 VGPRs.  Exercise every BF16 edge
// corpus, both accumulator sources, and every C modifier against the scalar
// oracle with C and D deliberately kept separate.
TEST(WmmaSimdExact, Bf16F32Mixed) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "the BF16F32 fast path requires 16-lane native SIMD";

  WmmaFixture fx;
  ASSERT_NE(fx.wf, nullptr);
  for (auto [mode, seed] : trials_for(Fmt::BF16))
    for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, CONST_ONE})
      for (uint32_t c_modifier = 0; c_modifier < 4; ++c_modifier) {
        auto reseed = [&] {
          fx.seed(S0, BF16_IN_REGS, Fmt::BF16, mode, seed + 1);
          fx.seed(S1, BF16_IN_REGS, Fmt::BF16, mode, seed + 2);
          fx.seed(ACC, ACC_REGS, Fmt::F32, Mode::RandomInt, seed + 3);
          fx.seed_words(BF16_DST, BF16_DST_REGS, seed + 4);
        };
        expect_bit_exact(
            "wmma_bf16f32_16x16x32_bf16", mode, fx, reseed,
            [&] {
              amdgpu::exec_wmma_bf16f32_16x16x32_bf16(*fx.cu, fx.vbase + BF16_DST, fx.vbase + S0,
                                                      fx.vbase + S1, fx.vbase + ACC, const_acc,
                                                      c_modifier);
            },
            BF16_DST, BF16_DST_REGS);
        if (testing::Test::HasFatalFailure())
          return;
      }
}

// Every read must be snapshotted before the first four-register packed output
// write.  Cover full and partial destination overlap with each operand, plus
// shared A/C and B/C source regions whose words are interpreted at two widths.
TEST(WmmaSimdExact, Bf16F32MixedOverlap) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "the BF16F32 fast path requires 16-lane native SIMD";

  struct AliasCase {
    const char *label;
    uint32_t dst;
    uint32_t s0;
    uint32_t s1;
    uint32_t s2;
  };
  constexpr AliasCase cases[] = {
      {"dst_eq_s0", S0, S0, S1, ACC},        {"dst_partial_s0", S0 + 4, S0, S1, ACC},
      {"dst_eq_s1", S1, S0, S1, ACC},        {"dst_partial_s1", S1 + 4, S0, S1, ACC},
      {"dst_eq_acc", ACC, S0, S1, ACC},      {"dst_partial_acc", ACC + 4, S0, S1, ACC},
      {"s0_eq_acc", BF16_DST, ACC, S1, ACC}, {"s1_eq_acc", BF16_DST, S0, ACC, ACC},
  };

  for (const auto &alias : cases)
    for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, CONST_ONE}) {
      WmmaFixture fx;
      ASSERT_NE(fx.wf, nullptr);
      auto reseed = [&] {
        fx.seed(S0, BF16_IN_REGS, Fmt::BF16, Mode::RandomInt, 0x101);
        fx.seed(S1, BF16_IN_REGS, Fmt::BF16, Mode::RandomInt, 0x202);
        fx.seed(ACC, ACC_REGS, Fmt::F32, Mode::RandomInt, 0x303);
        if (alias.dst == BF16_DST)
          fx.seed_words(BF16_DST, BF16_DST_REGS, 0x404);
      };
      expect_bit_exact(
          alias.label, Mode::RandomInt, fx, reseed,
          [&] {
            amdgpu::exec_wmma_bf16f32_16x16x32_bf16(
                *fx.cu, fx.vbase + alias.dst, fx.vbase + alias.s0, fx.vbase + alias.s1,
                fx.vbase + alias.s2, const_acc, /*c_modifier=*/3);
          },
          alias.dst, BF16_DST_REGS);
      if (testing::Test::HasFatalFailure())
        return;
    }
}

// Distinct NaN payloads expose host FMA operand-selection differences that a
// canonical all-NaN corpus cannot detect.  Each mixed FMA step gives priority
// to its current A, then current B, then the running accumulator, and quiets a
// selected signaling NaN.
TEST(WmmaSimdExact, Bf16F32MixedNanPayloadPriority) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "the BF16F32 fast path requires 16-lane native SIMD";

  WmmaFixture fx;
  ASSERT_NE(fx.wf, nullptr);
  auto reseed = [&] {
    fx.seed(S0, BF16_IN_REGS, Fmt::BF16, Mode::Zeros, 0);
    fx.seed(S1, BF16_IN_REGS, Fmt::BF16, Mode::Zeros, 0);
    fx.seed(ACC, ACC_REGS, Fmt::F32, Mode::Zeros, 0);
    fx.seed_words(BF16_DST, BF16_DST_REGS, 0x4E414E);
    auto write_bf16 = [&](uint32_t base, const auto &loc, uint16_t value) {
      uint32_t word = fx.cu->read_vgpr(fx.vbase + base + loc.vgpr_offset, loc.lane);
      const uint32_t shift = 16 * loc.sub_element;
      word = (word & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(value) << shift);
      fx.cu->write_vgpr(fx.vbase + base + loc.vgpr_offset, loc.lane, word);
    };
    auto write_acc = [&](uint32_t row, uint32_t col, uint32_t value) {
      const auto loc = amdgpu::wmma_output_loc_32(16, 16, row, col);
      fx.cu->write_vgpr(fx.vbase + ACC + loc.reg, loc.lane, value);
    };

    write_bf16(S0, amdgpu::wmma_input_loc(16, 32, 0, 0, 16), 0x7FC1u);
    write_bf16(S1, amdgpu::wmma_input_loc(16, 32, 0, 0, 16), 0x7FC2u);
    write_acc(0, 0, 0x7FC30000u);

    write_bf16(S0, amdgpu::wmma_input_loc(16, 32, 1, 0, 16), 0x3F80u);
    write_bf16(S1, amdgpu::wmma_input_loc(16, 32, 1, 0, 16), 0x7FC2u);
    write_acc(1, 1, 0x7FC30000u);

    write_acc(2, 2, 0x7FC30000u);
    write_bf16(S0, amdgpu::wmma_input_loc(16, 32, 3, 1, 16), 0x7F81u);
    write_bf16(S1, amdgpu::wmma_input_loc(16, 32, 3, 1, 16), 0x3F80u);
    write_bf16(S0, amdgpu::wmma_input_loc(16, 32, 4, 2, 16), 0x3F80u);
    write_bf16(S1, amdgpu::wmma_input_loc(16, 32, 4, 2, 16), 0xFFC4u);
    write_bf16(S0, amdgpu::wmma_input_loc(16, 32, 5, 3, 16), 0x7F80u);

    write_bf16(S0, amdgpu::wmma_input_loc(16, 32, 6, 0, 16), 0x7FC1u);
    write_bf16(S0, amdgpu::wmma_input_loc(16, 32, 6, 31, 16), 0x3F80u);
    write_bf16(S1, amdgpu::wmma_input_loc(16, 32, 6, 31, 16), 0x7FC2u);
  };
  expect_bit_exact(
      "wmma_bf16f32_nan_payloads", Mode::NaN, fx, reseed,
      [&] {
        amdgpu::exec_wmma_bf16f32_16x16x32_bf16(*fx.cu, fx.vbase + BF16_DST, fx.vbase + S0,
                                                fx.vbase + S1, fx.vbase + ACC,
                                                amdgpu::ACC_FROM_VGPR, /*c_modifier=*/0);
      },
      BF16_DST, BF16_DST_REGS);
  ASSERT_FALSE(testing::Test::HasFatalFailure());

  auto expect_output = [&](uint32_t row, uint32_t col, uint16_t expected) {
    const auto out = amdgpu::wmma_output_loc_16(16, 16, row, col);
    const uint32_t word = fx.cu->read_vgpr(fx.vbase + BF16_DST + out.reg, out.lane);
    EXPECT_EQ(static_cast<uint16_t>(word >> (16 * out.sub_element)), expected)
        << "row=" << row << " col=" << col;
  };
  expect_output(0, 0, 0x7FC1u); // A before B and C.
  expect_output(1, 1, 0x7FC2u); // B before C.
  expect_output(2, 2, 0x7FC3u); // C when A and B are numeric.
  expect_output(3, 3, 0x7FC1u); // Selected signaling A is quieted.
  expect_output(4, 4, 0xFFC4u); // Sign and payload are preserved.
  expect_output(5, 5, 0xFFC0u); // Invalid Inf*0 has a stable NaN encoding.
  expect_output(6, 6, 0x7FC2u); // A later source NaN supersedes the accumulator.
}

// Zero products isolate the mixed output map and the BF16 truncation contract.
// Several values would round up under RNE, so this also catches use of the
// wrong pack operation.
TEST(WmmaSimdExact, Bf16F32MixedPackLayout) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "the BF16F32 fast path requires 16-lane native SIMD";

  WmmaFixture fx;
  ASSERT_NE(fx.wf, nullptr);
  fx.seed(S0, BF16_IN_REGS, Fmt::BF16, Mode::Zeros, 0);
  fx.seed(S1, BF16_IN_REGS, Fmt::BF16, Mode::Zeros, 0);
  fx.seed_words(BF16_DST, BF16_DST_REGS, 0xBAD5EED);
  constexpr uint32_t inputs[] = {0x3F800001u, 0x3F80FFFFu, 0x3F818000u,
                                 0xBF80FFFFu, 0x0080FFFFu, 0x7F7FFFFFu};
  for (uint32_t row = 0; row < 16; ++row)
    for (uint32_t col = 0; col < 16; ++col) {
      const auto out = amdgpu::wmma_output_loc_32(16, 16, row, col);
      fx.cu->write_vgpr(fx.vbase + ACC + out.reg, out.lane,
                        inputs[(row * 16 + col) % std::size(inputs)]);
    }

  ForceScalarGuard force_scalar_guard;
  util::set_force_scalar_for_testing(false);
  amdgpu::exec_wmma_bf16f32_16x16x32_bf16(*fx.cu, fx.vbase + BF16_DST, fx.vbase + S0, fx.vbase + S1,
                                          fx.vbase + ACC, amdgpu::ACC_FROM_VGPR, /*c_modifier=*/0);
  for (uint32_t row = 0; row < 16; ++row)
    for (uint32_t col = 0; col < 16; ++col) {
      const auto out = amdgpu::wmma_output_loc_16(16, 16, row, col);
      const uint32_t word = fx.cu->read_vgpr(fx.vbase + BF16_DST + out.reg, out.lane);
      const uint16_t actual = static_cast<uint16_t>(word >> (16 * out.sub_element));
      const uint16_t expected =
          static_cast<uint16_t>(inputs[(row * 16 + col) % std::size(inputs)] >> 16);
      EXPECT_EQ(actual, expected) << "row=" << row << " col=" << col;
    }
}

// Multiplication overflows in isolation, but the hardware-fused operation is
// finite after adding -FLT_MAX.  This witnesses the single-rounding contract
// in both the forced-scalar oracle and the vector path.
TEST(WmmaSimdExact, Bf16F32MixedFusedOverflow) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "the BF16F32 fast path requires 16-lane native SIMD";

  WmmaFixture fx;
  ASSERT_NE(fx.wf, nullptr);
  auto reseed = [&] {
    fx.seed(S0, BF16_IN_REGS, Fmt::BF16, Mode::Zeros, 0);
    fx.seed(S1, BF16_IN_REGS, Fmt::BF16, Mode::Zeros, 0);
    fx.seed(ACC, ACC_REGS, Fmt::F32, Mode::Zeros, 0);
    fx.seed_words(BF16_DST, BF16_DST_REGS, 0x5151);
    const auto a = amdgpu::wmma_input_loc(16, 32, /*row=*/0, /*k=*/0, 16);
    const auto b = amdgpu::wmma_input_loc(16, 32, /*col=*/0, /*k=*/0, 16);
    auto write_bf16 = [&](uint32_t base, const auto &loc, uint16_t value) {
      uint32_t word = fx.cu->read_vgpr(fx.vbase + base + loc.vgpr_offset, loc.lane);
      const uint32_t shift = 16 * loc.sub_element;
      word = (word & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(value) << shift);
      fx.cu->write_vgpr(fx.vbase + base + loc.vgpr_offset, loc.lane, word);
    };
    write_bf16(S0, a, 0x5F80u); // 2^64
    write_bf16(S1, b, 0x5F80u);
    const auto c = amdgpu::wmma_output_loc_32(16, 16, /*row=*/0, /*col=*/0);
    fx.cu->write_vgpr(fx.vbase + ACC + c.reg, c.lane, 0xFF7FFFFFu); // -FLT_MAX
  };
  expect_bit_exact(
      "wmma_bf16f32_fused_overflow", Mode::MaxFinite, fx, reseed,
      [&] {
        amdgpu::exec_wmma_bf16f32_16x16x32_bf16(*fx.cu, fx.vbase + BF16_DST, fx.vbase + S0,
                                                fx.vbase + S1, fx.vbase + ACC,
                                                amdgpu::ACC_FROM_VGPR, /*c_modifier=*/0);
      },
      BF16_DST, BF16_DST_REGS);
  ASSERT_FALSE(testing::Test::HasFatalFailure());
  const auto out = amdgpu::wmma_output_loc_16(16, 16, /*row=*/0, /*col=*/0);
  const uint32_t word = fx.cu->read_vgpr(fx.vbase + BF16_DST + out.reg, out.lane);
  EXPECT_EQ(static_cast<uint16_t>(word >> (16 * out.sub_element)), 0x7380u);
}

// --- dense fp8/bf8 inputs, all four A/B combos, K=64 and K=128, f32/f16 out ---
TEST(WmmaSimdExact, F8Dense) {
  SKIP_IF_NO_SIMD();
  for (uint32_t k : {64u, 128u}) {
    run_dense_f32("wmma_f32_fp8_fp8", Fmt::FP8, 16, 16, k, 8, amdgpu::extract_fp8,
                  amdgpu::extract_fp8);
    run_dense_f32("wmma_f32_fp8_bf8", Fmt::FP8, 16, 16, k, 8, amdgpu::extract_fp8,
                  amdgpu::extract_bf8);
    run_dense_f32("wmma_f32_bf8_fp8", Fmt::BF8, 16, 16, k, 8, amdgpu::extract_bf8,
                  amdgpu::extract_fp8);
    run_dense_f32("wmma_f32_bf8_bf8", Fmt::BF8, 16, 16, k, 8, amdgpu::extract_bf8,
                  amdgpu::extract_bf8);
    run_dense_f16("wmma_f16_fp8_fp8", Fmt::FP8, k, 8, amdgpu::extract_fp8, amdgpu::extract_fp8);
    run_dense_f16("wmma_f16_bf8_bf8", Fmt::BF8, k, 8, amdgpu::extract_bf8, amdgpu::extract_bf8);
  }
}

// --- specialized dense fp8/bf8 kernels (constexpr dims + LUT bulk convert):
// all four A/B format pairs, K=64 and K=128, f32 and f16 output. ---
TEST(WmmaSimdExact, F8SpecDense) {
  SKIP_IF_NO_SIMD();
  auto spec_f32 = [](WmmaF32SpecFn fn, Fmt fmt, const char *label) {
    run_case(label, fmt, Fmt::F32, [fn](WmmaFixture &fx, uint32_t ca) {
      fn(*fx.cu, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1, fx.vbase + ACC, ca, 0);
    });
  };
  auto spec_f16 = [](WmmaF16SpecFn fn, Fmt fmt, const char *label) {
    run_case(label, fmt, Fmt::F16, [fn](WmmaFixture &fx, uint32_t ca) {
      fn(*fx.cu, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1, fx.vbase + ACC, ca, false);
    });
  };
  spec_f32(amdgpu::exec_wmma_f32_f8_spec<16, 16, 64, true, true>, Fmt::FP8, "spec_f32_fp8_fp8_k64");
  spec_f32(amdgpu::exec_wmma_f32_f8_spec<16, 16, 64, true, false>, Fmt::FP8,
           "spec_f32_fp8_bf8_k64");
  spec_f32(amdgpu::exec_wmma_f32_f8_spec<16, 16, 64, false, true>, Fmt::BF8,
           "spec_f32_bf8_fp8_k64");
  spec_f32(amdgpu::exec_wmma_f32_f8_spec<16, 16, 64, false, false>, Fmt::BF8,
           "spec_f32_bf8_bf8_k64");
  spec_f32(amdgpu::exec_wmma_f32_f8_spec<16, 16, 128, true, true>, Fmt::FP8,
           "spec_f32_fp8_fp8_k128");
  spec_f32(amdgpu::exec_wmma_f32_f8_spec<16, 16, 128, true, false>, Fmt::FP8,
           "spec_f32_fp8_bf8_k128");
  spec_f32(amdgpu::exec_wmma_f32_f8_spec<16, 16, 128, false, true>, Fmt::BF8,
           "spec_f32_bf8_fp8_k128");
  spec_f32(amdgpu::exec_wmma_f32_f8_spec<16, 16, 128, false, false>, Fmt::BF8,
           "spec_f32_bf8_bf8_k128");
  spec_f16(amdgpu::exec_wmma_f16_f8_spec<16, 16, 64, true, true>, Fmt::FP8, "spec_f16_fp8_fp8_k64");
  spec_f16(amdgpu::exec_wmma_f16_f8_spec<16, 16, 64, true, false>, Fmt::FP8,
           "spec_f16_fp8_bf8_k64");
  spec_f16(amdgpu::exec_wmma_f16_f8_spec<16, 16, 64, false, true>, Fmt::BF8,
           "spec_f16_bf8_fp8_k64");
  spec_f16(amdgpu::exec_wmma_f16_f8_spec<16, 16, 64, false, false>, Fmt::BF8,
           "spec_f16_bf8_bf8_k64");
  spec_f16(amdgpu::exec_wmma_f16_f8_spec<16, 16, 128, true, true>, Fmt::FP8,
           "spec_f16_fp8_fp8_k128");
  spec_f16(amdgpu::exec_wmma_f16_f8_spec<16, 16, 128, true, false>, Fmt::FP8,
           "spec_f16_fp8_bf8_k128");
  spec_f16(amdgpu::exec_wmma_f16_f8_spec<16, 16, 128, false, true>, Fmt::BF8,
           "spec_f16_bf8_fp8_k128");
  spec_f16(amdgpu::exec_wmma_f16_f8_spec<16, 16, 128, false, false>, Fmt::BF8,
           "spec_f16_bf8_bf8_k128");
}

// --- sparse f16/fp8 SWMMAC ---
TEST(WmmaSimdExact, Sparse) {
  SKIP_IF_NO_SIMD();
  run_sparse_f32("swmmac_f32_16x16x32_f16", Fmt::F16, 32, 16, 16, amdgpu::extract_f16,
                 amdgpu::extract_f16);
  run_sparse_f32("swmmac_f32_16x16x64_f16", Fmt::F16, 64, 16, 16, amdgpu::extract_f16,
                 amdgpu::extract_f16);
  run_sparse_f16("swmmac_f16_16x16x64_f16", Fmt::F16, 64, 16, 16, amdgpu::extract_f16,
                 amdgpu::extract_f16);
  run_sparse_f32("swmmac_f32_16x16x128_fp8", Fmt::FP8, 128, 8, 32, amdgpu::extract_fp8,
                 amdgpu::extract_fp8);
  run_sparse_f32("swmmac_f32_16x16x128_bf8", Fmt::BF8, 128, 8, 32, amdgpu::extract_bf8,
                 amdgpu::extract_bf8);
  run_sparse_f16("swmmac_f16_16x16x128_fp8", Fmt::FP8, 128, 8, 32, amdgpu::extract_fp8,
                 amdgpu::extract_fp8);
  run_sparse_f16("swmmac_f16_16x16x128_bf8", Fmt::BF8, 128, 8, 32, amdgpu::extract_bf8,
                 amdgpu::extract_bf8);
}

// --- integer WMMA/SWMMAC, signed/unsigned, clamp on and off ---
TEST(WmmaSimdExact, I32) {
  SKIP_IF_NO_SIMD();
  run_case("wmma_i32_16x16x16_i8", Fmt::I8, Fmt::I8, [](WmmaFixture &fx, uint32_t ca) {
    amdgpu::exec_wmma_i32_i8(*fx.cu, 16, 16, 16, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                             fx.vbase + ACC, ca);
  });
  for (bool clamp : {false, true}) {
    run_case("wmma_i32_16x16x64_iu8", Fmt::I8, Fmt::I8, [clamp](WmmaFixture &fx, uint32_t ca) {
      amdgpu::exec_wmma_i32(*fx.cu, 16, 16, 64, 8, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                            fx.vbase + ACC, amdgpu::extract_i8, amdgpu::extract_u8, clamp, ca);
    });
    run_case("swmmac_i32_16x16x128_i8", Fmt::I8, Fmt::I8, [clamp](WmmaFixture &fx, uint32_t ca) {
      amdgpu::exec_swmmac_i32(*fx.cu, 16, 16, 128, 8, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                              fx.vbase + ACC, fx.vbase + INDEX, 32, INDEX_KEY, amdgpu::extract_i8,
                              amdgpu::extract_i8, clamp, ca);
    });
  }
  run_case("swmmac_i32_16x16x32_i8", Fmt::I8, Fmt::I8, [](WmmaFixture &fx, uint32_t ca) {
    amdgpu::exec_swmmac_i32_i8(*fx.cu, 16, 16, 32, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                               fx.vbase + ACC, fx.vbase + INDEX, 16, INDEX_KEY, ca);
  });
}

// --- specialized dense iu8 kernel: all sign combos, clamp on and off ---
TEST(WmmaSimdExact, I32Iu8Spec) {
  SKIP_IF_NO_SIMD();
  for (bool clamp : {false, true})
    for (bool a_signed : {false, true})
      for (bool b_signed : {false, true})
        run_case("wmma_i32_16x16x64_iu8_spec", Fmt::I8, Fmt::I8, [=](WmmaFixture &fx, uint32_t ca) {
          amdgpu::exec_wmma_i32_16x16x64_iu8(*fx.cu, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                                             fx.vbase + ACC, a_signed, b_signed, clamp, ca);
        });
}

// --- specialized dense iu8 kernel: saturation corners ---
// Accumulator seeded just below INT32_MAX / just above INT32_MIN so the i64
// add crosses the int32 boundary: with clamp it must saturate, without it
// wrap, identically to the scalar reference.
TEST(WmmaSimdExact, I32Iu8SpecSaturation) {
  SKIP_IF_NO_SIMD();
  struct Corner {
    uint32_t acc_word;
    Mode a_mode; // sum direction picked via A pattern: 0x7F = up, 0x80 = down
  };
  const Corner corners[] = {{0x7FFFFF00u, Mode::MaxFinite}, {0x80000100u, Mode::Denorm}};
  for (auto c : corners)
    for (bool clamp : {false, true}) {
      WmmaFixture fx;
      ASSERT_NE(fx.wf, nullptr);
      fx.seed(S0, IN_REGS, Fmt::I8, c.a_mode, 1);        // A: all 0x7F or all 0x80 (signed -128)
      fx.seed(S1, IN_REGS, Fmt::I8, Mode::MaxFinite, 2); // B: all 0x7F
      auto reseed_acc = [&] {
        for (uint32_t reg = 0; reg < ACC_REGS; ++reg)
          for (uint32_t lane = 0; lane < WF; ++lane)
            fx.cu->write_vgpr(fx.vbase + ACC + reg, lane, c.acc_word);
      };
      expect_bit_exact(
          "wmma_i32_16x16x64_iu8_spec_sat", c.a_mode, fx, reseed_acc,
          [&] {
            amdgpu::exec_wmma_i32_16x16x64_iu8(*fx.cu, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                                               fx.vbase + ACC, /*a_signed=*/true,
                                               /*b_signed=*/false, clamp, amdgpu::ACC_FROM_VGPR);
          },
          ACC, ACC_REGS);
      if (testing::Test::HasFatalFailure())
        return;
    }
}

// --- mixed-format dense (fp4/fp6/bf6 paths) and fp4 32x16 shape ---
TEST(WmmaSimdExact, MixedFmt) {
  SKIP_IF_NO_SIMD();
  run_dense_f32("wmma_f32_32x16x128_fp4", Fmt::RAW4, 32, 16, 128, 4, amdgpu::extract_fp4,
                amdgpu::extract_fp4);
  run_case("wmma_f32_mixed_fp4_fp4", Fmt::RAW4, Fmt::F32, [](WmmaFixture &fx, uint32_t ca) {
    amdgpu::exec_wmma_f32_mixed(*fx.cu, 16, 16, 128, 4, 4, fx.vbase + ACC, fx.vbase + S0,
                                fx.vbase + S1, fx.vbase + ACC, amdgpu::extract_fp4,
                                amdgpu::extract_fp4, ca);
  });
  run_case("wmma_f32_mixed_fp6_bf6", Fmt::RAW6, Fmt::F32, [](WmmaFixture &fx, uint32_t ca) {
    amdgpu::exec_wmma_f32_mixed(*fx.cu, 16, 16, 128, 6, 6, fx.vbase + ACC, fx.vbase + S0,
                                fx.vbase + S1, fx.vbase + ACC, amdgpu::extract_fp6,
                                amdgpu::extract_bf6, ca);
  });
  run_case("wmma_f32_mixed_fp8_fp6", Fmt::FP8, Fmt::F32, [](WmmaFixture &fx, uint32_t ca) {
    amdgpu::exec_wmma_f32_mixed(*fx.cu, 16, 16, 128, 8, 6, fx.vbase + ACC, fx.vbase + S0,
                                fx.vbase + S1, fx.vbase + ACC, amdgpu::extract_fp8,
                                amdgpu::extract_fp6, ca);
  });
}

// --- scaled mixed-format dense (e8m0 power-of-two scales stay exact) ---
// Scale bytes are constrained to [2^-15, 2^16]: the scalar path folds scales
// after the a*b product while the SIMD hoist folds them into A and B, so the
// two only agree bit-for-bit while no intermediate overflows or underflows.
TEST(WmmaSimdExact, ScaledMixed) {
  SKIP_IF_NO_SIMD();
  auto seed_scales = [](WmmaFixture &fx, uint32_t off, uint32_t seed) {
    std::mt19937 rng(seed);
    for (uint32_t reg = 0; reg < 4; ++reg)
      for (uint32_t lane = 0; lane < WF; ++lane) {
        uint32_t w = 0;
        for (uint32_t b = 0; b < 4; ++b)
          w |= (0x70u + (rng() & 0x1Fu)) << (b * 8); // e8m0 in [2^-15, 2^16]
        fx.cu->write_vgpr(fx.vbase + off + reg, lane, w);
      }
  };
  run_case("wmma_f32_scaled_fp8", Fmt::FP8, Fmt::F32, [&](WmmaFixture &fx, uint32_t ca) {
    seed_scales(fx, SCALE_A, 0xA5);
    seed_scales(fx, SCALE_B, 0x5A);
    auto sa = [&fx](uint32_t lane) { return fx.cu->read_vgpr(fx.vbase + SCALE_A, lane); };
    auto sb = [&fx](uint32_t lane) { return fx.cu->read_vgpr(fx.vbase + SCALE_B, lane); };
    amdgpu::exec_wmma_f32_scaled_mixed(*fx.cu, 16, 16, 128, 8, 8, fx.vbase + ACC, fx.vbase + S0,
                                       fx.vbase + S1, fx.vbase + ACC, amdgpu::extract_fp8,
                                       amdgpu::extract_fp8, ca, sa, sb, /*matrix_a_scale=*/0,
                                       /*matrix_b_scale=*/0, /*matrix_a_scale_fmt=*/0,
                                       /*matrix_b_scale_fmt=*/0);
  });
}
