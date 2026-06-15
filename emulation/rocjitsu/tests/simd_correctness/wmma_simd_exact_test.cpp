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
constexpr uint32_t INDEX_ENTRIES = 16, INDEX_KEY = 0;
constexpr uint32_t CONST_ONE = 0x3F800000u;
constexpr uint32_t SCALE_A = 100, SCALE_B = 104;

struct WmmaFixture : ExactFixture {
  WmmaFixture() : ExactFixture(ROCJITSU_CODE_ARCH_GFX1250, WF) {}
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
void run_sparse_f32(const char *label, Fmt fmt, uint32_t K, uint32_t bits, Ea ea, Eb eb) {
  run_case(label, fmt, Fmt::F32, [=](WmmaFixture &fx, uint32_t const_acc) {
    amdgpu::exec_swmmac_f32(*fx.cu, 16, 16, K, bits, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                            fx.vbase + ACC, fx.vbase + INDEX, INDEX_ENTRIES, INDEX_KEY, ea, eb,
                            const_acc);
  });
}

template <typename Ea, typename Eb>
void run_sparse_f16(const char *label, Fmt fmt, uint32_t K, uint32_t bits, Ea ea, Eb eb) {
  run_case(label, fmt, Fmt::F16, [=](WmmaFixture &fx, uint32_t const_acc) {
    amdgpu::exec_swmmac_f16(*fx.cu, 16, 16, K, bits, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                            fx.vbase + ACC, fx.vbase + INDEX, INDEX_ENTRIES, INDEX_KEY, ea, eb,
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
  run_sparse_f32("swmmac_f32_16x16x64_bf16", Fmt::BF16, 64, 16, amdgpu::extract_bf16,
                 amdgpu::extract_bf16);
  run_case("swmmac_bf16_16x16x64_bf16", Fmt::BF16, Fmt::BF16, [](WmmaFixture &fx, uint32_t ca) {
    amdgpu::exec_swmmac_bf16(*fx.cu, 16, 16, 64, 16, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1,
                             fx.vbase + ACC, fx.vbase + INDEX, INDEX_ENTRIES, INDEX_KEY,
                             amdgpu::extract_bf16, amdgpu::extract_bf16, ca);
  });
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
  auto spec_f32 = [](auto fn, Fmt fmt, const char *label) {
    run_case(label, fmt, Fmt::F32, [fn](WmmaFixture &fx, uint32_t ca) {
      fn(*fx.cu, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1, fx.vbase + ACC, ca);
    });
  };
  auto spec_f16 = [](auto fn, Fmt fmt, const char *label) {
    run_case(label, fmt, Fmt::F16, [fn](WmmaFixture &fx, uint32_t ca) {
      fn(*fx.cu, fx.vbase + ACC, fx.vbase + S0, fx.vbase + S1, fx.vbase + ACC, ca);
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
  run_sparse_f32("swmmac_f32_16x16x32_f16", Fmt::F16, 32, 16, amdgpu::extract_f16,
                 amdgpu::extract_f16);
  run_sparse_f32("swmmac_f32_16x16x64_f16", Fmt::F16, 64, 16, amdgpu::extract_f16,
                 amdgpu::extract_f16);
  run_sparse_f16("swmmac_f16_16x16x64_f16", Fmt::F16, 64, 16, amdgpu::extract_f16,
                 amdgpu::extract_f16);
  run_sparse_f32("swmmac_f32_16x16x128_fp8", Fmt::FP8, 128, 8, amdgpu::extract_fp8,
                 amdgpu::extract_fp8);
  run_sparse_f32("swmmac_f32_16x16x128_bf8", Fmt::BF8, 128, 8, amdgpu::extract_bf8,
                 amdgpu::extract_bf8);
  run_sparse_f16("swmmac_f16_16x16x128_fp8", Fmt::FP8, 128, 8, amdgpu::extract_fp8,
                 amdgpu::extract_fp8);
  run_sparse_f16("swmmac_f16_16x16x128_bf8", Fmt::BF8, 128, 8, amdgpu::extract_bf8,
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
                               fx.vbase + ACC, fx.vbase + INDEX, INDEX_ENTRIES, INDEX_KEY, ca);
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
