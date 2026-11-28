// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mfma_simd_exact_test.cpp
/// @brief Expensive bit-exact SIMD-vs-scalar checks for every CDNA4 MFMA
/// instruction family (public floating paths and remaining legacy i8/f64 helpers).
/// Built only with -DRJ_ENABLE_EXPENSIVE_CHECKS=ON.
///
/// Each case runs identical pre-seeded VGPR state through the forced-scalar
/// path then the SIMD path and asserts the full dst window matches word for
/// word. Inputs are rounding-free (see mma_exact_test_support.h) so fused
/// (SIMD) and non-fused (scalar) accumulation cannot legitimately differ.

#include "mfma_f32_test_dispatch.h"
#include "mfma_f64_test_dispatch.h"
#include "mfma_f8_test_dispatch.h"
#include "mfma_half_test_dispatch.h"
#include "mfma_integer_test_dispatch.h"
#include "mma_exact_test_support.h"
#include "rocjitsu/isa/isa_family.h"

namespace {

using namespace rocjitsu;
using namespace mma_exact;

constexpr uint32_t WF = 64;
constexpr uint32_t S0 = 0, S1 = 64, ACC = 128, DST = 192, DST_REGS = 32;
constexpr uint32_t IN_REGS = 16, ACC_REGS = 32;
constexpr uint32_t CONST_ONE = 0x3F800000u;

using MfmaFloatSpecFn = void (*)(amdgpu::ComputeUnitCore &, uint32_t, uint32_t, uint32_t, uint32_t,
                                 uint32_t, uint32_t, uint32_t, uint32_t);
using MfmaIntSpecFn = void (*)(amdgpu::ComputeUnitCore &, uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t);
struct MfmaFixture : ExactFixture {
  MfmaFixture() : ExactFixture(rocjitsu::IsaFamily::Cdna4, WF) {}
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
    auto reseed = [&] {
      // Clear scalar results so an omitted SIMD store cannot match them.
      fx.seed_words(DST, DST_REGS, seed + 4);
      fx.seed(ACC, ACC_REGS, acc_fmt, Mode::RandomInt, seed + 3);
    };
    for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, CONST_ONE}) {
      expect_bit_exact(label, mode, fx, reseed, [&] { kernel(fx, const_acc); }, DST, DST_REGS);
      if (testing::Test::HasFatalFailure()) {
        return;
      }
    }
  }
}

void run_i8(const char *label, uint32_t M, uint32_t N, uint32_t K, uint32_t B) {
  run_case(label, Fmt::I8, Fmt::I8, [=](MfmaFixture &fx, uint32_t const_acc) {
    rocjitsu::test::execute_mfma_i8(*fx.cu, M, N, K, B, fx.vbase + DST, fx.vbase + S0,
                                    fx.vbase + S1, fx.vbase + ACC, const_acc);
  });
}

void expect_fixture_bit_exact(const char *label, const std::function<void(MfmaFixture &)> &seed,
                              const std::function<void(MfmaFixture &)> &kernel, uint32_t dst,
                              uint32_t dst_regs) {
  ForceScalarGuard force_scalar_guard;
  MfmaFixture scalar_fixture;
  MfmaFixture simd_fixture;
  ASSERT_NE(scalar_fixture.wf, nullptr);
  ASSERT_NE(simd_fixture.wf, nullptr);
  seed(scalar_fixture);
  seed(simd_fixture);

  util::set_force_scalar_for_testing(true);
  kernel(scalar_fixture);
  util::set_force_scalar_for_testing(false);
  kernel(simd_fixture);

  std::vector<uint32_t> scalar = scalar_fixture.snapshot(dst, dst_regs);
  std::vector<uint32_t> simd = simd_fixture.snapshot(dst, dst_regs);
  for (size_t word = 0; word < scalar.size(); ++word) {
    ASSERT_EQ(scalar[word], simd[word])
        << label << ": SIMD diverges from scalar at word " << word << " (scalar=0x" << std::hex
        << scalar[word] << " simd=0x" << simd[word] << ")";
  }
}

} // namespace
// Source controls keep the same public SIMD path as unmodified inputs.
TEST(MfmaSimdExact, F16Permuted) {
  for (uint32_t controls : {1U, 1U << 7}) {
    run_case("f16 controls", Fmt::F16, Fmt::F32, [=](MfmaFixture &fx, uint32_t constant) {
      rocjitsu::test::execute_mfma_half_shape<false, 16, 16, 4, 4>(
          *fx.cu, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1, fx.vbase + ACC, constant,
          controls & 7, 0, controls >> 7);
    });
  }
}

// --- specialized f16 kernels (constexpr dims + F16C bulk convert) ---
TEST(MfmaSimdExact, F16Spec_AllShapes) {
  auto spec = [](MfmaFloatSpecFn fn, const char *label) {
    run_case(label, Fmt::F16, Fmt::F32, [fn](MfmaFixture &fx, uint32_t const_acc) {
      fn(*fx.cu, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1, fx.vbase + ACC, const_acc, 0, 0, 0);
    });
  };
  spec(rocjitsu::test::execute_mfma_half_shape<false, 16, 16, 32>, "spec_f16_16x16x32");
  spec(rocjitsu::test::execute_mfma_half_shape<false, 32, 32, 16>, "spec_f16_32x32x16");
  spec(rocjitsu::test::execute_mfma_half_shape<false, 32, 32, 8>, "spec_f16_32x32x8");
  spec(rocjitsu::test::execute_mfma_half_shape<false, 16, 16, 16>, "spec_f16_16x16x16");
  spec(rocjitsu::test::execute_mfma_half_shape<false, 32, 32, 4, 2>, "spec_f16_32x32x4x2");
  spec(rocjitsu::test::execute_mfma_half_shape<false, 16, 16, 4, 4>, "spec_f16_16x16x4x4");
}

// --- specialized bf16 kernels (constexpr dims + bf16 zero-extend bulk convert) ---
TEST(MfmaSimdExact, Bf16Spec_AllShapes) {
  auto spec = [](MfmaFloatSpecFn fn, const char *label) {
    run_case(label, Fmt::BF16, Fmt::F32, [fn](MfmaFixture &fx, uint32_t const_acc) {
      fn(*fx.cu, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1, fx.vbase + ACC, const_acc, 0, 0, 0);
    });
  };
  spec(rocjitsu::test::execute_mfma_half_shape<true, 16, 16, 32>, "spec_bf16_16x16x32");
  spec(rocjitsu::test::execute_mfma_half_shape<true, 32, 32, 16>, "spec_bf16_32x32x16");
  spec(rocjitsu::test::execute_mfma_half_shape<true, 32, 32, 8>, "spec_bf16_32x32x8");
  spec(rocjitsu::test::execute_mfma_half_shape<true, 16, 16, 16>, "spec_bf16_16x16x16");
  spec(rocjitsu::test::execute_mfma_half_shape<true, 32, 32, 4, 2>, "spec_bf16_32x32x4x2");
  spec(rocjitsu::test::execute_mfma_half_shape<true, 16, 16, 4, 4>, "spec_bf16_16x16x4x4");
}

// --- specialized f32 kernels ---
TEST(MfmaSimdExact, F32Spec_AllShapes) {
  auto spec = [](MfmaFloatSpecFn fn, const char *label) {
    run_case(label, Fmt::F32, Fmt::F32, [fn](MfmaFixture &fx, uint32_t const_acc) {
      fn(*fx.cu, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1, fx.vbase + ACC, const_acc, 0, 0, 0);
    });
  };
  spec(rocjitsu::test::execute_mfma_f32_shape<32, 32, 2, 1>, "spec_f32_32x32x2");
  spec(rocjitsu::test::execute_mfma_f32_shape<16, 16, 4, 1>, "spec_f32_16x16x4");
  spec(rocjitsu::test::execute_mfma_f32_shape<32, 32, 1, 2>, "spec_f32_32x32x1x2");
  spec(rocjitsu::test::execute_mfma_f32_shape<16, 16, 1, 4>, "spec_f32_16x16x1x4");
  spec(rocjitsu::test::execute_mfma_f32_shape<4, 4, 1, 16>, "f32_4x4x1x16");
}

TEST(MfmaSimdExact, F32SpecDestructiveSourceOverlap) {
  constexpr uint32_t W = static_cast<uint32_t>(util::RegisterBlock<float>::size());
  if (W <= 1 || 32 % W != 0) {
    GTEST_SKIP() << "test requires a supported native SIMD width";
  }
  constexpr uint32_t source_b = 0;
  constexpr uint32_t destination_and_source_a = 64;
  constexpr uint32_t accumulator = 128;

  for (bool materialize_overlap : {false, true}) {
    SCOPED_TRACE(materialize_overlap ? "materialized" : "logical zero");
    auto seed = [=](MfmaFixture &fx) {
      fx.seed(source_b, 1, Fmt::F32, Mode::RandomInt, 11);
      fx.seed(accumulator, 32, Fmt::F32, Mode::RandomInt, 22);
      if (materialize_overlap) {
        fx.seed(destination_and_source_a, 1, Fmt::F32, Mode::RandomInt, 33);
      }
    };
    auto kernel = [](MfmaFixture &fx) {
      rocjitsu::test::execute_mfma_f32_shape<32, 32, 1, 2>(
          *fx.cu, fx.vbase + destination_and_source_a, fx.vbase + destination_and_source_a,
          fx.vbase + source_b, fx.vbase + accumulator, amdgpu::ACC_FROM_VGPR, 0, 0, 0);
    };
    expect_fixture_bit_exact("f32 source overlap", seed, kernel, destination_and_source_a, 32);
  }
}

TEST(MfmaSimdExact, F16SpecDestructiveAccumulatorOverlap) {
  constexpr uint32_t source_a = 0;
  constexpr uint32_t source_b = 16;
  constexpr uint32_t accumulator = 48;
  constexpr uint32_t destination = 64;

  for (bool materialize_overlap : {false, true}) {
    SCOPED_TRACE(materialize_overlap ? "materialized" : "logical zero");
    auto seed = [=](MfmaFixture &fx) {
      fx.seed(source_a, 2, Fmt::F16, Mode::RandomInt, 44);
      fx.seed(source_b, 2, Fmt::F16, Mode::RandomInt, 55);
      fx.seed(accumulator, 16, Fmt::F32, Mode::RandomInt, 66);
      if (materialize_overlap) {
        fx.seed(destination, 16, Fmt::F32, Mode::RandomInt, 77);
      }
    };
    auto kernel = [](MfmaFixture &fx) {
      rocjitsu::test::execute_mfma_half_shape<false, 32, 32, 4, 2>(
          *fx.cu, fx.vbase + destination, fx.vbase + source_a, fx.vbase + source_b,
          fx.vbase + accumulator, amdgpu::ACC_FROM_VGPR, 0, 0, 0);
    };
    expect_fixture_bit_exact("f16 accumulator overlap", seed, kernel, destination, 32);
  }
}

TEST(MfmaSimdExact, Bf16SpecDestructiveAccumulatorOverlap) {
  constexpr uint32_t source_a = 0;
  constexpr uint32_t source_b = 16;
  constexpr uint32_t accumulator = 48;
  constexpr uint32_t destination = 64;

  for (bool materialize_overlap : {false, true}) {
    SCOPED_TRACE(materialize_overlap ? "materialized" : "logical zero");
    auto seed = [=](MfmaFixture &fx) {
      fx.seed(source_a, 2, Fmt::BF16, Mode::RandomInt, 122);
      fx.seed(source_b, 2, Fmt::BF16, Mode::RandomInt, 133);
      fx.seed(accumulator, 16, Fmt::F32, Mode::RandomInt, 144);
      if (materialize_overlap) {
        fx.seed(destination, 16, Fmt::F32, Mode::RandomInt, 155);
      }
    };
    auto kernel = [](MfmaFixture &fx) {
      rocjitsu::test::execute_mfma_half_shape<true, 32, 32, 4, 2>(
          *fx.cu, fx.vbase + destination, fx.vbase + source_a, fx.vbase + source_b,
          fx.vbase + accumulator, amdgpu::ACC_FROM_VGPR, 0, 0, 0);
    };
    expect_fixture_bit_exact("bf16 accumulator overlap", seed, kernel, destination, 32);
  }
}

// --- integer i8 MFMA: exact MAC, every dispatched shape ---
TEST(MfmaSimdExact, I32_i8_AllShapes) {
  const struct {
    uint32_t m, n, k, b;
  } shapes[] = {{16, 16, 64, 1}, {32, 32, 32, 1}, {32, 32, 16, 1}, {16, 16, 32, 1},
                {32, 32, 4, 2},  {16, 16, 4, 4},  {4, 4, 4, 16}};
  for (auto s : shapes) {
    run_i8("i32_i8", s.m, s.n, s.k, s.b);
  }
}

// --- f64 MFMA ---
TEST(MfmaSimdExact, F64_AllShapes) {
  const struct {
    uint32_t m, n, k, b;
  } shapes[] = {{16, 16, 4, 1}, {4, 4, 4, 4}};
  for (auto s : shapes) {
    run_case("f64", Fmt::F64, Fmt::F64, [=](MfmaFixture &fx, uint32_t const_acc) {
      rocjitsu::test::execute_mfma_f64_shape(*fx.cu, s.m, s.n, s.k, s.b, fx.vbase + DST,
                                             fx.vbase + S0, fx.vbase + S1, fx.vbase + ACC,
                                             const_acc);
    });
  }
}
