// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mma_fused_mac_test.cpp
/// @brief One matrix instruction returns one answer: the scalar reference and
/// the SIMD core of the MFMA kernels round their MACs the same way.
///
/// The kernels in mma_exec.h carry two implementations of every f32/f64 matrix
/// instruction -- a scalar reference and a stdx::simd core -- and the emulator
/// picks between them on RJ_FORCE_SCALAR, AVX-512 availability and the shape.
/// The SIMD core has always issued a fused FMA, matching the hardware's
/// single-rounding MACs; the scalar reference multiplied and then added, so
/// whenever the product itself rounds the two disagreed. Witnessing that needs
/// f32 or f64 inputs: an f16/bf16/fp8 product is exact in f32, so those formats
/// never showed it (which is why the existing bit-exact suites stayed green).
///
/// Both witnesses use the same construction. Every A element is a, every B
/// element is b, and their exact product a*b needs one more bit than the format
/// holds, so rounding it alone loses a low term d. C is seeded to -K*round(a*b),
/// which makes the non-fused path cancel to exactly +0 while the fused path
/// keeps the K accumulated copies of d -- the whole answer, not one low bit.

#include "mma_exact_test_support.h"

namespace {

using namespace rocjitsu;
using namespace mma_exact;

// --- v_mfma_f32_16x16x4_f32 (wave64) ---
//
// a = 1 + 2^-12, b = 1 + 2^-13. The exact product is 1 + 2^-12 + 2^-13 + 2^-25;
// rounding it to f32 on its own drops the 2^-25 (ulp(1.0) = 2^-23).
constexpr uint32_t F32_A = 0x3F800800u; // 1 + 2^-12
constexpr uint32_t F32_B = 0x3F800400u; // 1 + 2^-13
// C = -4 * round(a*b) = -(4 + 2^-10 + 2^-11), exact in f32.
constexpr uint32_t F32_C = 0xC0800C00u;
// Four fused MACs keep the 2^-25 term at every step and land on 4 * 2^-25.
constexpr uint32_t F32_FUSED = 0x33000000u; // 2^-25
// Four pre-rounded products cancel C exactly.
constexpr uint32_t F32_NON_FUSED = 0x00000000u;

// --- v_mfma_f64_16x16x4_f64 (wave64) ---
//
// a = b = 1 + 2^-27. The exact square is 1 + 2^-26 + 2^-54; rounding it to f64
// on its own drops the 2^-54 (ulp(1.0) = 2^-52).
constexpr uint64_t F64_A = 0x3FF0000002000000ull; // 1 + 2^-27
// C = -4 * round(a*a) = -(4 + 2^-24), exact in f64.
constexpr uint64_t F64_C = 0xC010000004000000ull;
constexpr uint64_t F64_FUSED = 0x3C90000000000000ull; // 4 * 2^-54 = 2^-52
constexpr uint64_t F64_NON_FUSED = 0ull;

void fill(ExactFixture &fx, uint32_t off, uint32_t regs, uint32_t word) {
  for (uint32_t reg = 0; reg < regs; ++reg)
    for (uint32_t lane = 0; lane < fx.wf_size; ++lane)
      fx.cu->write_vgpr(fx.vbase + off + reg, lane, word);
}

// 64-bit elements occupy (lo,hi) register pairs; every element gets `value`.
void fill64(ExactFixture &fx, uint32_t off, uint32_t regs, uint64_t value) {
  for (uint32_t reg = 0; reg + 1 < regs; reg += 2)
    for (uint32_t lane = 0; lane < fx.wf_size; ++lane) {
      fx.cu->write_vgpr(fx.vbase + off + reg, lane, static_cast<uint32_t>(value));
      fx.cu->write_vgpr(fx.vbase + off + reg + 1, lane, static_cast<uint32_t>(value >> 32));
    }
}

// Run `kernel` with the force-scalar gate at `scalar` and assert the dst window
// matches; `word_expect` maps a word index to its expected value so 64-bit
// results can alternate lo/hi.
void expect_dst(const char *label, bool scalar, ExactFixture &fx, const std::function<void()> &seed,
                const std::function<void()> &kernel, uint32_t dst_off, uint32_t dst_regs,
                const std::function<uint32_t(size_t)> &word_expect) {
  ForceScalarGuard guard;
  seed();
  util::set_force_scalar_for_testing(scalar);
  kernel();
  util::set_force_scalar_for_testing(false);
  auto got = fx.snapshot(dst_off, dst_regs);
  for (size_t word = 0; word < got.size(); ++word)
    ASSERT_EQ(got[word], word_expect(word))
        << label << (scalar ? " [scalar]" : " [simd]") << ": word " << std::dec << word << " is 0x"
        << std::hex << got[word] << ", want 0x" << word_expect(word);
}

constexpr uint32_t S0 = 0, S1 = 8, DST = 16;
constexpr uint32_t F32_IN_REGS = 2, F32_DST_REGS = 4;

TEST(MmaFusedMac, MfmaF32ScalarAndSimdAgreeOnARoundingProduct) {
  // Guard the witness: the two candidate answers must differ, or the test would
  // pass no matter which MAC the kernels use.
  ASSERT_NE(F32_FUSED, F32_NON_FUSED);
  auto expect_fused = [](size_t) { return F32_FUSED; };
  for (bool scalar : {true, false}) {
    ExactFixture fx(ROCJITSU_CODE_ARCH_CDNA4, mma_test::MFMA_WF_SIZE);
    ASSERT_NE(fx.wf, nullptr);
    auto seed = [&] {
      fill(fx, S0, F32_IN_REGS, F32_A);
      fill(fx, S1, F32_IN_REGS, F32_B);
      fill(fx, DST, F32_DST_REGS, 0);
    };
    // Generic executor.
    expect_dst(
        "exec_f32 16x16x4", scalar, fx, seed,
        [&] {
          amdgpu::exec_f32(*fx.cu, 16, 16, 4, 1, 32, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1,
                           fx.vbase + DST, amdgpu::extract_f32, amdgpu::extract_f32, F32_C);
        },
        DST, F32_DST_REGS, expect_fused);
    if (testing::Test::HasFatalFailure())
      return;
    // Specialized fast path for the same instruction.
    expect_dst(
        "exec_f32_mfma_f32_spec<16,16,4,1>", scalar, fx, seed,
        [&] {
          amdgpu::exec_f32_mfma_f32_spec<16, 16, 4, 1>(
              *fx.cu, fx.vbase + DST, fx.vbase + S0, fx.vbase + S1, fx.vbase + DST, F32_C, 0, 0, 0);
        },
        DST, F32_DST_REGS, expect_fused);
    if (testing::Test::HasFatalFailure())
      return;
  }
}

constexpr uint32_t F64_S0 = 0, F64_S1 = 8, F64_ACC = 16, F64_DST = 32;
constexpr uint32_t F64_IN_REGS = 2, F64_ACC_REGS = 8, F64_DST_REGS = 8;

TEST(MmaFusedMac, MfmaF64ScalarAndSimdAgreeOnARoundingProduct) {
  ASSERT_NE(F64_FUSED, F64_NON_FUSED);
  // Results land as (lo,hi) register pairs.
  auto expect_fused = [](size_t word) {
    const bool hi = ((word / mma_test::MFMA_WF_SIZE) & 1u) != 0;
    return static_cast<uint32_t>(hi ? (F64_FUSED >> 32) : F64_FUSED);
  };
  for (bool scalar : {true, false}) {
    ExactFixture fx(ROCJITSU_CODE_ARCH_CDNA4, mma_test::MFMA_WF_SIZE);
    ASSERT_NE(fx.wf, nullptr);
    expect_dst(
        "exec_f64 16x16x4", scalar, fx,
        [&] {
          fill64(fx, F64_S0, F64_IN_REGS, F64_A);
          fill64(fx, F64_S1, F64_IN_REGS, F64_A);
          fill64(fx, F64_ACC, F64_ACC_REGS, F64_C);
          fill(fx, F64_DST, F64_DST_REGS, 0);
        },
        [&] {
          amdgpu::exec_f64(*fx.cu, 16, 16, 4, 1, fx.vbase + F64_DST, fx.vbase + F64_S0,
                           fx.vbase + F64_S1, fx.vbase + F64_ACC);
        },
        F64_DST, F64_DST_REGS, expect_fused);
    if (testing::Test::HasFatalFailure())
      return;
  }
}

} // namespace
