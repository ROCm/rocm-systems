// SPDX-License-Identifier: MIT

/// @file cdna3_mfma_test.cpp
/// @brief CDNA3 MFMA arithmetic against MI300X captures, and which instructions use it.

#include "fixtures/float_dot/cdna3_mfma_cases.h"
#include "rocjitsu/isa/arch/amdgpu/shared/cdna3_mfma.h"
#include "simd_correctness/mma_exact_test_support.h"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace mma_exact;
using amdgpu::cdna3_mfma_f8;
using amdgpu::cdna3_mfma_matmul;
using amdgpu::Cdna3MfmaArithmetic;
using amdgpu::kCdna3MfmaBf16;
using amdgpu::kCdna3MfmaF16;
using amdgpu::kCdna3MfmaXf32;

constexpr uint32_t WF = 64;
constexpr uint32_t S0 = 0, S1 = 64, ACC = 128, DST = 192;
constexpr uint32_t CONST_ONE = 0x3F800000u;
constexpr uint32_t kNanBits = 0xFFC00000u;

struct RecordedForm {
  const char *name;
  Cdna3MfmaArithmetic arithmetic;
  uint32_t K;
  std::span<const uint32_t> cases;
};

const RecordedForm kRecordedForms[] = {
    {"f16_32x32x8", kCdna3MfmaF16, 8, kCdna3MfmaCases_f16_32x32x8},
    {"f16_16x16x16", kCdna3MfmaF16, 16, kCdna3MfmaCases_f16_16x16x16},
    {"f16_32x32x4_2b", kCdna3MfmaF16, 4, kCdna3MfmaCases_f16_32x32x4_2b},
    {"f16_16x16x4_4b", kCdna3MfmaF16, 4, kCdna3MfmaCases_f16_16x16x4_4b},
    {"f16_4x4x4_16b", kCdna3MfmaF16, 4, kCdna3MfmaCases_f16_4x4x4_16b},
    {"bf16_32x32x8", kCdna3MfmaBf16, 8, kCdna3MfmaCases_bf16_32x32x8},
    {"bf16_16x16x16", kCdna3MfmaBf16, 16, kCdna3MfmaCases_bf16_16x16x16},
    {"bf16_32x32x4_2b", kCdna3MfmaBf16, 4, kCdna3MfmaCases_bf16_32x32x4_2b},
    {"bf16_16x16x4_4b", kCdna3MfmaBf16, 4, kCdna3MfmaCases_bf16_16x16x4_4b},
    {"bf16_4x4x4_16b", kCdna3MfmaBf16, 4, kCdna3MfmaCases_bf16_4x4x4_16b},
    {"xf32_32x32x4", kCdna3MfmaXf32, 4, kCdna3MfmaCases_xf32_32x32x4},
    {"xf32_16x16x8", kCdna3MfmaXf32, 8, kCdna3MfmaCases_xf32_16x16x8},
    {"fp8_32x32x16", cdna3_mfma_f8(true, true), 16, kCdna3MfmaCases_fp8_32x32x16},
    {"bf8_32x32x16", cdna3_mfma_f8(false, false), 16, kCdna3MfmaCases_bf8_32x32x16},
    {"fp8_bf8_32x32x16", cdna3_mfma_f8(true, false), 16, kCdna3MfmaCases_fp8_bf8_32x32x16},
    {"bf8_fp8_32x32x16", cdna3_mfma_f8(false, true), 16, kCdna3MfmaCases_bf8_fp8_32x32x16},
    {"fp8_fp8_16x16x32", cdna3_mfma_f8(true, true), 32, kCdna3MfmaCases_fp8_fp8_16x16x32},
    {"bf8_bf8_16x16x32", cdna3_mfma_f8(false, false), 32, kCdna3MfmaCases_bf8_bf8_16x16x32},
    {"fp8_bf8_16x16x32", cdna3_mfma_f8(true, false), 32, kCdna3MfmaCases_fp8_bf8_16x16x32},
    {"bf8_fp8_16x16x32", cdna3_mfma_f8(false, true), 32, kCdna3MfmaCases_bf8_fp8_16x16x32},
};

void check_recorded(const RecordedForm &form) {
  SCOPED_TRACE(form.name);
  const size_t case_words = 2 * form.K + 2;
  ASSERT_FALSE(form.cases.empty());
  ASSERT_EQ(form.cases.size() % case_words, 0u);
  for (size_t at = 0; at < form.cases.size(); at += case_words) {
    std::vector<float> a(form.K), b(form.K);
    for (uint32_t k = 0; k < form.K; ++k) {
      a[k] = std::bit_cast<float>(form.cases[at + k]);
      b[k] = std::bit_cast<float>(form.cases[at + form.K + k]);
    }
    float c = std::bit_cast<float>(form.cases[at + 2 * form.K]);
    cdna3_mfma_matmul(form.arithmetic, 1, 1, form.K, a.data(), b.data(), &c);
    EXPECT_EQ(std::bit_cast<uint32_t>(c), form.cases[at + 2 * form.K + 1])
        << "case " << at / case_words;
  }
}

float one_output(const Cdna3MfmaArithmetic &arithmetic, const std::vector<float> &a,
                 const std::vector<float> &b, float c) {
  cdna3_mfma_matmul(arithmetic, 1, 1, static_cast<uint32_t>(a.size()), a.data(), b.data(), &c);
  return c;
}

TEST(Cdna3Mfma, RecordedOutputs) {
  for (const RecordedForm &form : kRecordedForms)
    check_recorded(form);
}

TEST(Cdna3Mfma, RecordedOutputsDoNotDependOnHostRoundingMode) {
  const int saved = std::fegetround();
  for (int mode : {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(mode), 0);
    for (const RecordedForm &form : kRecordedForms)
      check_recorded(form);
  }
  std::fesetround(saved);
}

TEST(Cdna3Mfma, TruncatesSmallProductsBelowTheLargest) {
  // -1.5 * 2^-25 is below the 24 fraction bits kept under a product of 1.0, so truncation drops
  // it. Sequential f32 addition would round the sum to 1 - 2^-24.
  const std::vector<float> a{1.0f, 0x1.8p-13f, 0, 0, 0, 0, 0, 0};
  const std::vector<float> b{1.0f, -0x1p-12f, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(one_output(kCdna3MfmaF16, a, b, 0.0f), 1.0f);
}

TEST(Cdna3Mfma, RoundsTheAccumulatorDown) {
  // C is 25 bits below the product, so its unit lands below the 24 bits kept for C and rounds down:
  // a negative C takes a whole 2^-24 off, a positive C adds nothing.
  const std::vector<float> a{1.0f, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<float> b{1.0f, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(one_output(kCdna3MfmaF16, a, b, 0x1p-25f), 1.0f);
  EXPECT_EQ(one_output(kCdna3MfmaF16, a, b, -0x1p-25f), 1.0f - 0x1p-24f);
}

TEST(Cdna3Mfma, F8SumsEvenAndOddPositionsSeparately) {
  // 2^14 and -2^-14. In the same group the small product truncates to nothing. In separate groups
  // it keeps its own 24 bits, then rounds down when the groups are aligned: a whole unit comes off.
  const Cdna3MfmaArithmetic fp8 = cdna3_mfma_f8(true, true);
  std::vector<float> a(16, 0.0f), b(16, 0.0f);
  a[0] = 128.0f;
  b[0] = 128.0f;
  a[2] = 0x1p-7f;
  b[2] = -0x1p-7f;
  EXPECT_EQ(one_output(fp8, a, b, 0.0f), 16384.0f);
  std::swap(a[1], a[2]);
  std::swap(b[1], b[2]);
  EXPECT_EQ(one_output(fp8, a, b, 0.0f), 16384.0f - 0x1p-10f);
}

TEST(Cdna3Mfma, F8DropsAFarAccumulatorButNotAnInfiniteOne) {
  const Cdna3MfmaArithmetic bf8 = cdna3_mfma_f8(false, false);
  std::vector<float> a(16, 0.0f), b(16, 0.0f);
  a[0] = 32.0f;
  b[0] = 32.0f;
  EXPECT_EQ(one_output(bf8, a, b, -0x1p-30f), 1024.0f);
  a[0] = 32768.0f;
  b[0] = 32768.0f;
  EXPECT_EQ(one_output(bf8, a, b, INFINITY), INFINITY);
  EXPECT_EQ(one_output(bf8, a, b, -INFINITY), -INFINITY);
  EXPECT_EQ(std::bit_cast<uint32_t>(one_output(bf8, a, b, NAN)), kNanBits);
}

TEST(Cdna3Mfma, Xf32IgnoresTheLow13InputBits) {
  const float low_bits = std::bit_cast<float>(0x3F801FFFu);
  const std::vector<float> a{low_bits, 0, 0, 0};
  const std::vector<float> b{low_bits, 0, 0, 0};
  EXPECT_EQ(one_output(kCdna3MfmaXf32, a, b, 0.0f), 1.0f);
}

TEST(Cdna3Mfma, ProductsAt2To128CountAsInfinite) {
  const float big = 0x1p64f;
  const std::vector<float> a{big, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<float> b{big, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(one_output(kCdna3MfmaBf16, a, b, -0x1p127f), INFINITY);
  const std::vector<float> opposite_a{big, -big, 0, 0, 0, 0, 0, 0};
  const std::vector<float> opposite_b{big, big, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(std::bit_cast<uint32_t>(one_output(kCdna3MfmaBf16, opposite_a, opposite_b, 0.0f)),
            kNanBits);
}

TEST(Cdna3Mfma, EveryNanIsWrittenTheSameWay) {
  const std::vector<float> a{INFINITY, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<float> b{0.0f, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(std::bit_cast<uint32_t>(one_output(kCdna3MfmaF16, a, b, 1.0f)), kNanBits);
  EXPECT_EQ(std::bit_cast<uint32_t>(one_output(kCdna3MfmaF16, b, b, NAN)), kNanBits);
}

TEST(Cdna3Mfma, CancellingToZeroGivesPositiveZero) {
  const std::vector<float> a{1.0f, 1.0f, 0, 0, 0, 0, 0, 0};
  const std::vector<float> b{1.0f, -1.0f, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(std::bit_cast<uint32_t>(one_output(kCdna3MfmaF16, a, b, -0.0f)), 0u);
}

struct Shape {
  uint32_t M, N, K, blocks, a_bits, b_bits;
};

// Fills sources and accumulator with anything at all, or with nearby exponents so products round.
void seed_inputs(ExactFixture &fixture, bool narrow, uint32_t seed,
                 uint32_t (*narrow_source)(uint32_t), uint32_t source_regs, uint32_t acc_regs) {
  std::mt19937 rng(seed);
  for (uint32_t base : {S0, S1})
    for (uint32_t reg = 0; reg < source_regs; ++reg)
      for (uint32_t lane = 0; lane < WF; ++lane) {
        const uint32_t word = rng();
        fixture.cu->write_vgpr(fixture.vbase + base + reg, lane,
                               narrow ? narrow_source(word) : word);
      }
  for (uint32_t reg = 0; reg < acc_regs; ++reg)
    for (uint32_t lane = 0; lane < WF; ++lane) {
      const uint32_t word = rng();
      fixture.cu->write_vgpr(fixture.vbase + ACC + reg, lane,
                             narrow ? (word & 0x807FFFFFu) | 0x3F800000u : word);
    }
}

uint32_t narrow_f16(uint32_t word) { return (word & 0x83FF83FFu) | 0x3C003C00u; }
uint32_t narrow_bf16(uint32_t word) { return (word & 0x807F807Fu) | 0x3F803F80u; }
uint32_t narrow_f8(uint32_t word) { return (word & 0x87878787u) | 0x40404040u; }
uint32_t narrow_f32(uint32_t word) { return (word & 0x807FFFFFu) | 0x3F800000u; }

uint32_t source_register_count(const Shape &shape) {
  return std::max(shape.M, shape.N) * shape.K * shape.blocks *
             std::max(shape.a_bits, shape.b_bits) / (WF * 32) +
         1;
}

// Runs a kernel over registers, on its SIMD and forced-scalar paths, and separately works out each
// output from the same registers with cdna3_mfma_matmul.
template <typename ExtractA, typename ExtractB>
void check_dispatch(const char *label, const Shape &shape, const Cdna3MfmaArithmetic &arithmetic,
                    ExtractA extract_a, ExtractB extract_b, uint32_t (*narrow_source)(uint32_t),
                    const std::function<void(ExactFixture &, uint32_t)> &kernel) {
  const auto [M, N, K, blocks, a_bits, b_bits] = shape;
  const uint32_t output_regs = M * N * blocks / WF;
  for (bool narrow : {true, false}) {
    for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, CONST_ONE}) {
      ForceScalarGuard guard;
      ExactFixture fixture(ROCJITSU_CODE_ARCH_CDNA3, WF);
      ASSERT_NE(fixture.wf, nullptr);
      seed_inputs(fixture, narrow, 1234u + K, narrow_source, source_register_count(shape),
                  output_regs);

      std::vector<uint32_t> expected(static_cast<size_t>(output_regs) * WF);
      std::vector<float> a(M * K), b(K * N), c(M * N);
      for (uint32_t block = 0; block < blocks; ++block) {
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t k = 0; k < K; ++k)
            a[row * K + k] = extract_a(*fixture.cu, fixture.vbase + S0,
                                       amdgpu::input_loc(M, K, blocks, row, k, block, a_bits));
        for (uint32_t k = 0; k < K; ++k)
          for (uint32_t col = 0; col < N; ++col)
            b[k * N + col] = extract_b(*fixture.cu, fixture.vbase + S1,
                                       amdgpu::input_loc(N, K, blocks, col, k, block, b_bits));
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            const auto out = amdgpu::output_loc_32(M, N, row, col, block);
            c[row * N + col] = std::bit_cast<float>(
                const_acc != amdgpu::ACC_FROM_VGPR
                    ? const_acc
                    : fixture.cu->read_vgpr(fixture.vbase + ACC + out.reg, out.lane));
          }
        cdna3_mfma_matmul(arithmetic, M, N, K, a.data(), b.data(), c.data());
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            const auto out = amdgpu::output_loc_32(M, N, row, col, block);
            expected[static_cast<size_t>(out.reg) * WF + out.lane] =
                std::bit_cast<uint32_t>(c[row * N + col]);
          }
      }

      for (bool scalar : {false, true}) {
        util::set_force_scalar_for_testing(scalar);
        kernel(fixture, const_acc);
        EXPECT_EQ(fixture.snapshot(DST, output_regs), expected)
            << label << (scalar ? " scalar" : " simd") << (narrow ? " narrow" : " any bits")
            << (const_acc != amdgpu::ACC_FROM_VGPR ? " const" : " vgpr");
      }
    }
  }
}

// One FMA per product, in K order: what CDNA4, and CDNA3's plain f32 shapes, still do.
template <typename ExtractA, typename ExtractB>
void check_stays_fma(const char *label, rj_code_arch_t arch, const Shape &shape, ExtractA extract_a,
                     ExtractB extract_b, uint32_t (*narrow_source)(uint32_t),
                     const std::function<void(ExactFixture &, uint32_t)> &kernel) {
  const auto [M, N, K, blocks, a_bits, b_bits] = shape;
  const uint32_t output_regs = M * N * blocks / WF;
  ExactFixture fixture(arch, WF);
  ASSERT_NE(fixture.wf, nullptr);
  seed_inputs(fixture, true, 99u, narrow_source, source_register_count(shape), output_regs);
  std::vector<uint32_t> expected(static_cast<size_t>(output_regs) * WF);
  for (uint32_t block = 0; block < blocks; ++block)
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        const auto out = amdgpu::output_loc_32(M, N, row, col, block);
        float acc =
            std::bit_cast<float>(fixture.cu->read_vgpr(fixture.vbase + ACC + out.reg, out.lane));
        for (uint32_t k = 0; k < K; ++k)
          acc = std::fma(extract_a(*fixture.cu, fixture.vbase + S0,
                                   amdgpu::input_loc(M, K, blocks, row, k, block, a_bits)),
                         extract_b(*fixture.cu, fixture.vbase + S1,
                                   amdgpu::input_loc(N, K, blocks, col, k, block, b_bits)),
                         acc);
        expected[static_cast<size_t>(out.reg) * WF + out.lane] = std::bit_cast<uint32_t>(acc);
      }
  kernel(fixture, amdgpu::ACC_FROM_VGPR);
  EXPECT_EQ(fixture.snapshot(DST, output_regs), expected) << label;
}

using SpecFn = void (*)(amdgpu::ComputeUnitCore &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                        uint32_t, uint32_t, uint32_t);

// Binds one specialization of the generic spec templates to ComputeUnitCore.
std::function<void(ExactFixture &, uint32_t)> spec_kernel(SpecFn kernel) {
  return [kernel](ExactFixture &fixture, uint32_t const_acc) {
    kernel(*fixture.cu, fixture.vbase + DST, fixture.vbase + S0, fixture.vbase + S1,
           fixture.vbase + ACC, const_acc, 0, 0, 0);
  };
}

TEST(Cdna3MfmaDispatch, F16) {
  const auto extract = amdgpu::extract_f16;
  check_dispatch("f16 32x32x8", {32, 32, 8, 1, 16, 16}, kCdna3MfmaF16, extract, extract, narrow_f16,
                 spec_kernel(amdgpu::exec_f32_mfma_f16_spec<32, 32, 8>));
  check_dispatch("f16 16x16x16", {16, 16, 16, 1, 16, 16}, kCdna3MfmaF16, extract, extract,
                 narrow_f16, spec_kernel(amdgpu::exec_f32_mfma_f16_spec<16, 16, 16>));
  check_dispatch("f16 32x32x4 2b", {32, 32, 4, 2, 16, 16}, kCdna3MfmaF16, extract, extract,
                 narrow_f16, spec_kernel(amdgpu::exec_f32_mfma_f16_spec<32, 32, 4, 2>));
  check_dispatch("f16 16x16x4 4b", {16, 16, 4, 4, 16, 16}, kCdna3MfmaF16, extract, extract,
                 narrow_f16, spec_kernel(amdgpu::exec_f32_mfma_f16_spec<16, 16, 4, 4>));
  check_dispatch("f16 4x4x4 16b", {4, 4, 4, 16, 16, 16}, kCdna3MfmaF16, extract, extract,
                 narrow_f16, [=](ExactFixture &fixture, uint32_t const_acc) {
                   amdgpu::exec_f32(*fixture.cu, 4, 4, 4, 16, 16, fixture.vbase + DST,
                                    fixture.vbase + S0, fixture.vbase + S1, fixture.vbase + ACC,
                                    extract, extract, const_acc, 0, 0, 0);
                 });
}

TEST(Cdna3MfmaDispatch, Bf16) {
  const auto extract = amdgpu::extract_bf16;
  check_dispatch("bf16 32x32x8", {32, 32, 8, 1, 16, 16}, kCdna3MfmaBf16, extract, extract,
                 narrow_bf16, spec_kernel(amdgpu::exec_f32_mfma_bf16_spec<32, 32, 8>));
  check_dispatch("bf16 16x16x16", {16, 16, 16, 1, 16, 16}, kCdna3MfmaBf16, extract, extract,
                 narrow_bf16, spec_kernel(amdgpu::exec_f32_mfma_bf16_spec<16, 16, 16>));
  check_dispatch("bf16 16x16x4 4b", {16, 16, 4, 4, 16, 16}, kCdna3MfmaBf16, extract, extract,
                 narrow_bf16, spec_kernel(amdgpu::exec_f32_mfma_bf16_spec<16, 16, 4, 4>));
}

TEST(Cdna3MfmaDispatch, Fp8AndBf8) {
  const auto fp8 = amdgpu::extract_fp8_fnuz;
  const auto bf8 = amdgpu::extract_bf8_fnuz;
  using amdgpu::exec_f32_mfma_f8_spec;
  check_dispatch("fp8 fp8 16x16x32", {16, 16, 32, 1, 8, 8}, cdna3_mfma_f8(true, true), fp8, fp8,
                 narrow_f8, spec_kernel(exec_f32_mfma_f8_spec<16, 16, 32, true, true, true>));
  check_dispatch("bf8 bf8 32x32x16", {32, 32, 16, 1, 8, 8}, cdna3_mfma_f8(false, false), bf8, bf8,
                 narrow_f8, spec_kernel(exec_f32_mfma_f8_spec<32, 32, 16, false, false, true>));
  check_dispatch("fp8 bf8 16x16x32", {16, 16, 32, 1, 8, 8}, cdna3_mfma_f8(true, false), fp8, bf8,
                 narrow_f8, spec_kernel(exec_f32_mfma_f8_spec<16, 16, 32, true, false, true>));
  check_dispatch("bf8 fp8 32x32x16", {32, 32, 16, 1, 8, 8}, cdna3_mfma_f8(false, true), bf8, fp8,
                 narrow_f8, spec_kernel(exec_f32_mfma_f8_spec<32, 32, 16, false, true, true>));
}

TEST(Cdna3MfmaDispatch, Xf32) {
  const auto extract = amdgpu::extract_f32;
  check_dispatch("xf32 16x16x8", {16, 16, 8, 1, 32, 32}, kCdna3MfmaXf32, extract, extract,
                 narrow_f32, spec_kernel(amdgpu::exec_f32_mfma_f32_spec<16, 16, 8, 1>));
  check_dispatch("xf32 32x32x4", {32, 32, 4, 1, 32, 32}, kCdna3MfmaXf32, extract, extract,
                 narrow_f32, spec_kernel(amdgpu::exec_f32_mfma_f32_spec<32, 32, 4, 1>));
}

TEST(Cdna3MfmaDispatch, OtherChipsAndPlainF32StayFma) {
  const auto f16 = amdgpu::extract_f16;
  const auto bf16 = amdgpu::extract_bf16;
  const auto f32 = amdgpu::extract_f32;
  check_stays_fma("cdna4 f16", ROCJITSU_CODE_ARCH_CDNA4, {32, 32, 8, 1, 16, 16}, f16, f16,
                  narrow_f16, spec_kernel(amdgpu::exec_f32_mfma_f16_spec<32, 32, 8>));
  check_stays_fma("cdna4 bf16", ROCJITSU_CODE_ARCH_CDNA4, {16, 16, 16, 1, 16, 16}, bf16, bf16,
                  narrow_bf16, spec_kernel(amdgpu::exec_f32_mfma_bf16_spec<16, 16, 16>));
  check_stays_fma("cdna4 xf32", ROCJITSU_CODE_ARCH_CDNA4, {32, 32, 4, 1, 32, 32}, f32, f32,
                  narrow_f32, spec_kernel(amdgpu::exec_f32_mfma_f32_spec<32, 32, 4, 1>));
  check_stays_fma("cdna3 f32 32x32x2", ROCJITSU_CODE_ARCH_CDNA3, {32, 32, 2, 1, 32, 32}, f32, f32,
                  narrow_f32, spec_kernel(amdgpu::exec_f32_mfma_f32_spec<32, 32, 2, 1>));
  check_stays_fma("cdna3 f32 16x16x1 4b", ROCJITSU_CODE_ARCH_CDNA3, {16, 16, 1, 4, 32, 32}, f32,
                  f32, narrow_f32, spec_kernel(amdgpu::exec_f32_mfma_f32_spec<16, 16, 1, 4>));
}

} // namespace
