// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mfma_simd_exact_test.cpp
/// @brief Expensive bit-exact SIMD-vs-scalar checks for every CDNA4 MFMA
/// execute kernel and shape (generic, specialized, scaled, i8, f64).
/// Built only with -DRJ_ENABLE_EXPENSIVE_CHECKS=ON.
///
/// Each case runs identical pre-seeded VGPR state through the forced-scalar
/// path then the SIMD path and asserts the full dst window matches word for
/// word. Inputs are rounding-free (see mma_exact_test_support.h) so fused
/// (SIMD) and non-fused (scalar) accumulation cannot legitimately differ.

#include "mma_exact_test_support.h"

namespace {

using namespace rocjitsu;
using namespace mma_exact;

constexpr uint32_t WF = 64;
constexpr uint32_t S0 = 0, S1 = 64, ACC = 128, DST = 192, DST_REGS = 32;
constexpr uint32_t IN_REGS = 16, ACC_REGS = 32;
constexpr uint32_t CONST_ONE = 0x3F800000u;

struct MfmaFixture : ExactFixture {
  MfmaFixture() : ExactFixture(ROCJITSU_CODE_ARCH_CDNA4, WF) {}
};

// Drive one (kernel, fmt) case across all trial modes and both accumulator
// sources (VGPR window + 1.0f immediate).
void run_case(const char *label, Fmt fmt, Fmt acc_fmt,
              const std::function<void(MfmaFixture &, uint32_t)> &kernel) {
  MfmaFixture fx;
  ASSERT_NE(fx.wf, nullptr);
  for (auto [mode, seed] : trials_for(fmt)) {
    fx.seed(S0, IN_REGS, fmt, mode, seed + 1);
    fx.seed(S1, IN_REGS, fmt, mode, seed + 2);
    auto reseed_acc = [&] { fx.seed(ACC, ACC_REGS, acc_fmt, Mode::RandomInt, seed + 3); };
    for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, CONST_ONE}) {
      expect_bit_exact(label, mode, fx, reseed_acc, [&] { kernel(fx, const_acc); }, DST, DST_REGS);
      if (testing::Test::HasFatalFailure())
        return;
    }
  }
}

template <typename Ea, typename Eb>
void run_f32(const char *label, Fmt fmt, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
             uint32_t bits, Ea ea, Eb eb, uint32_t cbsz = 0, uint32_t abid = 0, uint32_t blgp = 0) {
  run_case(label, fmt, Fmt::F32, [=](MfmaFixture &fx, uint32_t const_acc) {
    amdgpu::exec_f32(*fx.cu, M, N, K, B, bits, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1,
                     fx.vbase + ACC, ea, eb, const_acc, cbsz, abid, blgp);
  });
}

void run_i8(const char *label, uint32_t M, uint32_t N, uint32_t K, uint32_t B) {
  run_case(label, Fmt::I8, Fmt::I8, [=](MfmaFixture &fx, uint32_t const_acc) {
    amdgpu::exec_i32_i8(*fx.cu, M, N, K, B, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1,
                        fx.vbase + ACC, const_acc);
  });
}

} // namespace

// --- exec_f32 generic, f16 inputs: every dispatched CDNA4 f16 shape ---
TEST(MfmaSimdExact, F32_f16_AllShapes) {
  SKIP_IF_NO_SIMD();
  const struct {
    uint32_t m, n, k, b;
  } shapes[] = {{16, 16, 32, 1}, {32, 32, 16, 1}, {32, 32, 8, 1}, {16, 16, 16, 1},
                {4, 4, 4, 16},   {16, 16, 4, 4},  {32, 32, 4, 2}};
  for (auto s : shapes)
    run_f32("f32_f16", Fmt::F16, s.m, s.n, s.k, s.b, 16, amdgpu::extract_f16, amdgpu::extract_f16);
}

// --- exec_f32 generic, bf16 inputs ---
TEST(MfmaSimdExact, F32_bf16_AllShapes) {
  SKIP_IF_NO_SIMD();
  const struct {
    uint32_t m, n, k, b;
  } shapes[] = {{16, 16, 32, 1}, {32, 32, 16, 1}, {4, 4, 4, 16}, {16, 16, 4, 4}, {32, 32, 4, 2}};
  for (auto s : shapes)
    run_f32("f32_bf16", Fmt::BF16, s.m, s.n, s.k, s.b, 16, amdgpu::extract_bf16,
            amdgpu::extract_bf16);
}

// --- exec_f32 generic, fp8/bf8 inputs incl. mixed A/B pairs ---
TEST(MfmaSimdExact, F32_f8_AllShapes) {
  SKIP_IF_NO_SIMD();
  const struct {
    uint32_t m, n, k;
  } shapes[] = {{16, 16, 32}, {32, 32, 16}, {16, 16, 128}, {32, 32, 64}};
  for (auto s : shapes) {
    run_f32("f32_fp8", Fmt::FP8, s.m, s.n, s.k, 1, 8, amdgpu::extract_fp8, amdgpu::extract_fp8);
    run_f32("f32_bf8", Fmt::BF8, s.m, s.n, s.k, 1, 8, amdgpu::extract_bf8, amdgpu::extract_bf8);
  }
  run_f32("f32_fp8_bf8", Fmt::FP8, 16, 16, 32, 1, 8, amdgpu::extract_fp8, amdgpu::extract_bf8);
  run_f32("f32_bf8_fp8", Fmt::BF8, 32, 32, 16, 1, 8, amdgpu::extract_bf8, amdgpu::extract_fp8);
}

// --- exec_f32 generic, f32 inputs (4x4 stays generic; bigger go specialized) ---
TEST(MfmaSimdExact, F32_f32_Generic4x4) {
  SKIP_IF_NO_SIMD();
  run_f32("f32_f32_4x4x1x16", Fmt::F32, 4, 4, 1, 16, 32, amdgpu::extract_f32, amdgpu::extract_f32);
  run_f32("f32_f32_4x4x4x16", Fmt::F32, 4, 4, 4, 16, 32, amdgpu::extract_f32, amdgpu::extract_f32);
}

// --- cbsz/abid/blgp lane-permutation paths (folded into the SIMD hoist) ---
TEST(MfmaSimdExact, F32_f16_Permuted) {
  SKIP_IF_NO_SIMD();
  run_f32("f32_f16_cbsz1", Fmt::F16, 16, 16, 4, 4, 16, amdgpu::extract_f16, amdgpu::extract_f16,
          /*cbsz=*/1, /*abid=*/0, /*blgp=*/0);
  run_f32("f32_f16_blgp1", Fmt::F16, 16, 16, 4, 4, 16, amdgpu::extract_f16, amdgpu::extract_f16,
          /*cbsz=*/0, /*abid=*/0, /*blgp=*/1);
}

// --- specialized f16 kernels (constexpr dims + F16C bulk convert) ---
TEST(MfmaSimdExact, F16Spec_AllShapes) {
  SKIP_IF_NO_SIMD();
  auto spec = [](auto fn, const char *label) {
    run_case(label, Fmt::F16, Fmt::F32, [fn](MfmaFixture &fx, uint32_t const_acc) {
      fn(*fx.cu, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1, fx.vbase + ACC, const_acc, 0, 0, 0);
    });
  };
  spec(amdgpu::exec_f32_mfma_f16_spec<16, 16, 32>, "spec_f16_16x16x32");
  spec(amdgpu::exec_f32_mfma_f16_spec<32, 32, 16>, "spec_f16_32x32x16");
  spec(amdgpu::exec_f32_mfma_f16_spec<32, 32, 8>, "spec_f16_32x32x8");
  spec(amdgpu::exec_f32_mfma_f16_spec<16, 16, 16>, "spec_f16_16x16x16");
  spec(amdgpu::exec_f32_mfma_f16_spec<32, 32, 4, 2>, "spec_f16_32x32x4x2");
  spec(amdgpu::exec_f32_mfma_f16_spec<16, 16, 4, 4>, "spec_f16_16x16x4x4");
}

// --- specialized bf16 kernels (constexpr dims + bf16 zero-extend bulk convert) ---
TEST(MfmaSimdExact, Bf16Spec_AllShapes) {
  SKIP_IF_NO_SIMD();
  auto spec = [](auto fn, const char *label) {
    run_case(label, Fmt::BF16, Fmt::F32, [fn](MfmaFixture &fx, uint32_t const_acc) {
      fn(*fx.cu, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1, fx.vbase + ACC, const_acc, 0, 0, 0);
    });
  };
  spec(amdgpu::exec_f32_mfma_bf16_spec<16, 16, 32>, "spec_bf16_16x16x32");
  spec(amdgpu::exec_f32_mfma_bf16_spec<32, 32, 16>, "spec_bf16_32x32x16");
  spec(amdgpu::exec_f32_mfma_bf16_spec<32, 32, 8>, "spec_bf16_32x32x8");
  spec(amdgpu::exec_f32_mfma_bf16_spec<16, 16, 16>, "spec_bf16_16x16x16");
  spec(amdgpu::exec_f32_mfma_bf16_spec<32, 32, 4, 2>, "spec_bf16_32x32x4x2");
  spec(amdgpu::exec_f32_mfma_bf16_spec<16, 16, 4, 4>, "spec_bf16_16x16x4x4");
}

// --- specialized fp8/bf8 kernels (constexpr dims + LUT bulk convert), all four
// A/B format pairs on both dense shapes. The trial modes drive NaN, Inf (bf8),
// denorm, and max-finite f8 codes through both paths. ---
TEST(MfmaSimdExact, F8Spec_AllShapes) {
  SKIP_IF_NO_SIMD();
  auto spec = [](auto fn, Fmt fmt, const char *label) {
    run_case(label, fmt, Fmt::F32, [fn](MfmaFixture &fx, uint32_t const_acc) {
      fn(*fx.cu, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1, fx.vbase + ACC, const_acc, 0, 0, 0);
    });
  };
  spec(amdgpu::exec_f32_mfma_f8_spec<16, 16, 32, true, true>, Fmt::FP8, "spec_fp8_fp8_16x16x32");
  spec(amdgpu::exec_f32_mfma_f8_spec<16, 16, 32, true, false>, Fmt::FP8, "spec_fp8_bf8_16x16x32");
  spec(amdgpu::exec_f32_mfma_f8_spec<16, 16, 32, false, true>, Fmt::BF8, "spec_bf8_fp8_16x16x32");
  spec(amdgpu::exec_f32_mfma_f8_spec<16, 16, 32, false, false>, Fmt::BF8, "spec_bf8_bf8_16x16x32");
  spec(amdgpu::exec_f32_mfma_f8_spec<32, 32, 16, true, true>, Fmt::FP8, "spec_fp8_fp8_32x32x16");
  spec(amdgpu::exec_f32_mfma_f8_spec<32, 32, 16, true, false>, Fmt::FP8, "spec_fp8_bf8_32x32x16");
  spec(amdgpu::exec_f32_mfma_f8_spec<32, 32, 16, false, true>, Fmt::BF8, "spec_bf8_fp8_32x32x16");
  spec(amdgpu::exec_f32_mfma_f8_spec<32, 32, 16, false, false>, Fmt::BF8, "spec_bf8_bf8_32x32x16");
}

// --- specialized f32 kernels ---
TEST(MfmaSimdExact, F32Spec_AllShapes) {
  SKIP_IF_NO_SIMD();
  auto spec = [](auto fn, const char *label) {
    run_case(label, Fmt::F32, Fmt::F32, [fn](MfmaFixture &fx, uint32_t const_acc) {
      fn(*fx.cu, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1, fx.vbase + ACC, const_acc, 0, 0, 0);
    });
  };
  spec(amdgpu::exec_f32_mfma_f32_spec<32, 32, 2, 1>, "spec_f32_32x32x2");
  spec(amdgpu::exec_f32_mfma_f32_spec<16, 16, 4, 1>, "spec_f32_16x16x4");
  spec(amdgpu::exec_f32_mfma_f32_spec<32, 32, 1, 2>, "spec_f32_32x32x1x2");
  spec(amdgpu::exec_f32_mfma_f32_spec<16, 16, 1, 4>, "spec_f32_16x16x1x4");
}

// --- MX-scaled fp8/bf8 (per-32-block e8m0 scales: power-of-two, exact) ---
TEST(MfmaSimdExact, F32Scaled) {
  SKIP_IF_NO_SIMD();
  constexpr uint32_t SCALE_A = 240, SCALE_B = 248;
  auto scaled = [&](const char *label, Fmt fmt, uint32_t m, uint32_t n, uint32_t k, auto ex) {
    run_case(label, fmt, Fmt::F32, [=](MfmaFixture &fx, uint32_t const_acc) {
      fx.seed_words(SCALE_A, 4, 0xA5);
      fx.seed_words(SCALE_B, 4, 0x5A);
      amdgpu::exec_f32_scaled(*fx.cu, m, n, k, 1, 8, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1,
                              fx.vbase + ACC, ex, ex, const_acc, 0, 0, 0, fx.vbase + SCALE_A,
                              fx.vbase + SCALE_B);
    });
  };
  scaled("scaled_fp8_16x16x128", Fmt::FP8, 16, 16, 128, amdgpu::extract_fp8);
  scaled("scaled_bf8_32x32x64", Fmt::BF8, 32, 32, 64, amdgpu::extract_bf8);
}

// --- integer i8 MFMA: exact MAC, every dispatched shape ---
TEST(MfmaSimdExact, I32_i8_AllShapes) {
  SKIP_IF_NO_SIMD();
  const struct {
    uint32_t m, n, k, b;
  } shapes[] = {{16, 16, 64, 1}, {32, 32, 32, 1}, {32, 32, 16, 1}, {16, 16, 32, 1},
                {32, 32, 4, 2},  {16, 16, 4, 4},  {4, 4, 4, 16}};
  for (auto s : shapes)
    run_i8("i32_i8", s.m, s.n, s.k, s.b);
}

// --- specialized i8 kernels (constexpr dims + i8 sign-extend bulk convert) ---
// The accumulator window is seeded with random words spanning the full int32
// range, so wrap-around near INT32_MAX/INT32_MIN is exercised on both paths.
TEST(MfmaSimdExact, I8Spec_AllShapes) {
  SKIP_IF_NO_SIMD();
  auto spec = [](auto fn, const char *label) {
    run_case(label, Fmt::I8, Fmt::I8, [fn](MfmaFixture &fx, uint32_t const_acc) {
      fn(*fx.cu, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1, fx.vbase + ACC, const_acc);
    });
  };
  spec(amdgpu::exec_i32_mfma_i8_spec<16, 16, 64>, "spec_i8_16x16x64");
  spec(amdgpu::exec_i32_mfma_i8_spec<32, 32, 32>, "spec_i8_32x32x32");
  spec(amdgpu::exec_i32_mfma_i8_spec<32, 32, 16>, "spec_i8_32x32x16");
  spec(amdgpu::exec_i32_mfma_i8_spec<16, 16, 32>, "spec_i8_16x16x32");
  spec(amdgpu::exec_i32_mfma_i8_spec<32, 32, 4, 2>, "spec_i8_32x32x4x2");
  spec(amdgpu::exec_i32_mfma_i8_spec<16, 16, 4, 4>, "spec_i8_16x16x4x4");
}

// --- f64 MFMA ---
TEST(MfmaSimdExact, F64_AllShapes) {
  SKIP_IF_NO_SIMD();
  if (util::native<double>::size() <= 1)
    GTEST_SKIP() << "no native f64 SIMD";
  const struct {
    uint32_t m, n, k, b;
  } shapes[] = {{16, 16, 4, 1}, {4, 4, 4, 4}};
  for (auto s : shapes)
    run_case("f64", Fmt::F64, Fmt::F64, [=](MfmaFixture &fx, uint32_t const_acc) {
      amdgpu::exec_f64(*fx.cu, s.m, s.n, s.k, s.b, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1,
                       fx.vbase + ACC, const_acc);
    });
}
