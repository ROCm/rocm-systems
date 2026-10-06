// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mma_exact_test_support.h"

#include <array>

namespace {

using namespace rocjitsu;
using namespace mma_exact;

TEST(MfmaScaledOperandExact, Fp4StagingMatchesGenericLayout) {
  for (uint32_t wave_size : {32u, 64u}) {
    ExactFixture fx(wave_size == 64 ? ROCJITSU_CODE_ARCH_CDNA4 : ROCJITSU_CODE_ARCH_CDNA5,
                    wave_size);
    ASSERT_NE(fx.wf, nullptr);
    constexpr uint32_t source = 16;
    const uint32_t registers = 4 * 64 / wave_size;
    fx.seed_words(source, registers, 0x9817);
    // Include every FP4 encoding, including both signs of zero.
    fx.cu->write_vgpr(fx.vbase + source, 0, 0x76543210u);
    fx.cu->write_vgpr(fx.vbase + source, 1, 0xfedcba98u);
    auto reads = amdgpu::RegisterAccess(*fx.cu).read_vgpr_region(
        fx.vbase + source, registers, amdgpu::mfma_full_lane_mask(wave_size));
    for (uint32_t dim : {16u, 32u}) {
      const uint32_t K = 2048 / dim;
      const uint32_t a_stride = K + 3;
      const uint32_t b_stride = dim + 3;
      constexpr float sentinel = 99.0f;
      std::vector<float> a(dim * a_stride, sentinel);
      std::vector<float> b(K * b_stride, sentinel);
      amdgpu::stage_mfma_scale_operand<false>(reads, dim, K, 4, a_stride, a.data(),
                                              amdgpu::extract_fp4);
      amdgpu::stage_mfma_scale_operand<true>(reads, dim, K, 4, b_stride, b.data(),
                                             amdgpu::extract_fp4);
      for (uint32_t index = 0; index < dim; ++index) {
        for (uint32_t k = 0; k < K; ++k) {
          SCOPED_TRACE(::testing::Message() << "wave=" << wave_size << " dim=" << dim
                                            << " index=" << index << " k=" << k);
          // The unchanged general GFX9 mapping is independent of packed staging.
          const auto loc =
              amdgpu::physicalize_loc(amdgpu::input_loc(dim, K, 1, index, k, 0, 4), wave_size);
          const auto expected =
              std::bit_cast<uint32_t>(amdgpu::extract_fp4(*fx.cu, fx.vbase + source, loc));
          ASSERT_EQ(std::bit_cast<uint32_t>(a[index * a_stride + k]), expected);
          ASSERT_EQ(std::bit_cast<uint32_t>(b[k * b_stride + index]), expected);
        }
        for (uint32_t k = K; k < a_stride; ++k)
          EXPECT_EQ(a[index * a_stride + k], sentinel);
      }
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t index = dim; index < b_stride; ++index)
          EXPECT_EQ(b[k * b_stride + index], sentinel);
    }
  }
}

template <typename ExtractA, typename ExtractB>
void compare_scaled_mfma(uint32_t dim, uint32_t destination, uint32_t seed, Fmt a_format,
                         Fmt b_format, ExtractA extract_a, ExtractB extract_b,
                         bool materialize_a = true) {
  ForceScalarGuard guard;
  ExactFixture scalar(ROCJITSU_CODE_ARCH_CDNA4, 64);
  ExactFixture simd(ROCJITSU_CODE_ARCH_CDNA4, 64);
  ASSERT_NE(scalar.wf, nullptr);
  ASSERT_NE(simd.wf, nullptr);
  constexpr uint32_t source_a = 0, source_b = 16, accumulator = 32;
  constexpr uint32_t scale_a = 96, scale_b = 97;
  const uint32_t K = 2048 / dim;
  const uint32_t a_bits = fmt_bits(a_format), b_bits = fmt_bits(b_format);
  const uint32_t accumulator_regs = dim * dim / 64;
  for (auto *fx : {&scalar, &simd}) {
    if (materialize_a)
      fx->seed(source_a, a_bits, a_format, Mode::RandomInt, seed);
    fx->seed(source_b, b_bits, b_format, Mode::RandomInt, seed + 1);
    fx->seed(accumulator, accumulator_regs, Fmt::F32, Mode::RandomInt, seed + 2);
    for (uint32_t lane = 0; lane < 64; ++lane) {
      uint32_t a_word = 0, b_word = 0;
      for (uint32_t byte = 0; byte < 4; ++byte) {
        a_word |= (125u + (lane + byte) % 5) << (byte * 8);
        b_word |= (125u + (lane * 3 + byte) % 5) << (byte * 8);
      }
      fx->cu->write_vgpr(fx->vbase + scale_a, lane, a_word);
      fx->cu->write_vgpr(fx->vbase + scale_b, lane, b_word);
    }
  }
  auto execute = [&](ExactFixture &fx) {
    amdgpu::exec_f32_scaled_mixed(
        *fx.cu, dim, dim, K, 1, a_bits, b_bits, fx.vbase + destination, fx.vbase + source_a,
        fx.vbase + source_b, fx.vbase + accumulator, extract_a, extract_b,
        seed & 1u ? amdgpu::ACC_FROM_VGPR : std::bit_cast<uint32_t>(1.0f), fx.vbase, 256 + scale_a,
        256 + scale_b, seed % 4, (seed + 1) % 4, seed % 4);
  };
  util::set_force_scalar_for_testing(true);
  execute(scalar);
  util::set_force_scalar_for_testing(false);
  execute(simd);
  const auto expected = scalar.snapshot(destination, accumulator_regs);
  const auto actual = simd.snapshot(destination, accumulator_regs);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index)
    ASSERT_EQ(actual[index], expected[index])
        << "dim=" << dim << " destination=" << destination << " seed=" << seed << " word=" << index;
}

TEST(MfmaScaledOperandExact, Fp4MatchesScalarWithSourceAndAccumulatorOverlap) {
  SKIP_IF_NO_SIMD();
  for (uint32_t dim : {16u, 32u})
    for (uint32_t destination : {0u, 16u, 32u, 64u})
      for (uint32_t seed : {1u, 22u})
        compare_scaled_mfma(dim, destination, seed, Fmt::RAW4, Fmt::RAW4, amdgpu::extract_fp4,
                            amdgpu::extract_fp4);
}

TEST(MfmaScaledOperandExact, Fp4LogicalZeroSourceOverlapMatchesScalar) {
  SKIP_IF_NO_SIMD();
  for (uint32_t dim : {16u, 32u})
    compare_scaled_mfma(dim, 0, 13, Fmt::RAW4, Fmt::RAW4, amdgpu::extract_fp4, amdgpu::extract_fp4,
                        /*materialize_a=*/false);
}

TEST(MfmaScaledOperandExact, Fp4MixedFormatsMatchScalar) {
  SKIP_IF_NO_SIMD();
  for (uint32_t dim : {16u, 32u}) {
    compare_scaled_mfma(dim, 64, 71, Fmt::RAW4, Fmt::RAW6, amdgpu::extract_fp4,
                        amdgpu::extract_fp6);
    compare_scaled_mfma(dim, 64, 82, Fmt::RAW6, Fmt::RAW4, amdgpu::extract_bf6,
                        amdgpu::extract_fp4);
    compare_scaled_mfma(dim, 64, 93, Fmt::RAW4, Fmt::FP8, amdgpu::extract_fp4, amdgpu::extract_fp8);
    compare_scaled_mfma(dim, 64, 104, Fmt::BF8, Fmt::RAW4, amdgpu::extract_bf8,
                        amdgpu::extract_fp4);
  }
}

} // namespace
