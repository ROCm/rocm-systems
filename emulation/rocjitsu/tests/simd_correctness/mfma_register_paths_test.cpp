// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mfma_register_paths_test.cpp
/// @brief Wave64 MFMA register-snapshot paths against the forced-scalar reference.

#include "mma_exact_test_support.h"

namespace {

using namespace rocjitsu;
using namespace mma_exact;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t kSourceA = 0;
constexpr uint32_t kSourceB = 16;
constexpr uint32_t kAccumulator = 32;
constexpr uint32_t kDestination = 64;

enum class Overlap { None, Accumulator, SourceA };

struct Shape {
  Fmt input_format;
  Fmt accumulator_format;
  uint32_t a_regs;
  uint32_t b_regs;
  uint32_t d_regs;
};

/// Runs @p kernel forced-scalar and then normally over identical register state, for a
/// destination that is separate from, equal to, or overlapping the sources.
void expect_matches_scalar(
    const char *label, const Shape &shape, Overlap overlap, uint32_t seed,
    const std::function<void(ExactFixture &, uint32_t, uint32_t, uint32_t, uint32_t)> &kernel) {
  ExactFixture fixture(ROCJITSU_CODE_ARCH_CDNA4, WF_SIZE);
  ASSERT_NE(fixture.wf, nullptr);
  const uint32_t destination = overlap == Overlap::Accumulator ? kAccumulator
                               : overlap == Overlap::SourceA   ? kSourceA
                                                               : kDestination;
  // Random words first, so a destination word that is never written cannot match the
  // scalar result; the sources are seeded last, so an overlapping one keeps its data.
  auto reseed = [&] {
    fixture.seed_words(destination, shape.d_regs, seed + 1000);
    if (shape.accumulator_format == Fmt::I8)
      fixture.seed_words(kAccumulator, shape.d_regs, seed + 3);
    else
      fixture.seed(kAccumulator, shape.d_regs, shape.accumulator_format, Mode::RandomInt, seed + 3);
    fixture.seed(kSourceB, shape.b_regs, shape.input_format, Mode::RandomInt, seed + 2);
    fixture.seed(kSourceA, shape.a_regs, shape.input_format, Mode::RandomInt, seed + 1);
  };
  auto run = [&] {
    kernel(fixture, fixture.vbase + destination, fixture.vbase + kSourceA, fixture.vbase + kSourceB,
           fixture.vbase + kAccumulator);
  };
  expect_bit_exact(label, Mode::RandomInt, fixture, reseed, run, destination, shape.d_regs);
}

TEST(MfmaRegisterPathsTest, F64MatchesScalarForSeparateAndOverlappingDestinations) {
  SKIP_IF_NO_SIMD();
  struct Mfma {
    uint32_t m, n, k, batch, d_regs;
  };
  for (const Mfma mfma : {Mfma{16, 16, 4, 1, 8}, Mfma{4, 4, 4, 4, 2}})
    for (Overlap overlap : {Overlap::None, Overlap::Accumulator, Overlap::SourceA})
      for (uint32_t neg : {0u, 1u, 2u, 4u, 7u})
        for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, 0x40000000u}) {
          const Shape shape{Fmt::F64, Fmt::F64, 2, 2, mfma.d_regs};
          expect_matches_scalar(
              "f64 mfma", shape, overlap, 11 + neg,
              [&](ExactFixture &fx, uint32_t dst, uint32_t a, uint32_t b, uint32_t c) {
                amdgpu::exec_f64(*fx.cu, mfma.m, mfma.n, mfma.k, mfma.batch, dst, a, b, c,
                                 const_acc, neg);
              });
          if (::testing::Test::HasFatalFailure())
            return;
        }
}

TEST(MfmaRegisterPathsTest, Fp8ChunkedLayoutMatchesScalarWithLanePermutations) {
  SKIP_IF_NO_SIMD();
  struct Permutation {
    uint32_t cbsz, abid, blgp;
  };
  // 16x16x128 uses the chunked 16-element lane layout, which the specialized paths exclude.
  const Shape shape{Fmt::FP8, Fmt::F32, 8, 8, 4};
  for (Overlap overlap : {Overlap::None, Overlap::Accumulator, Overlap::SourceA})
    for (const Permutation p : {Permutation{0, 0, 0}, Permutation{2, 1, 0}, Permutation{0, 0, 3}})
      for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, 0x3f800000u}) {
        expect_matches_scalar(
            "fp8 16x16x128", shape, overlap, 21 + p.cbsz,
            [&](ExactFixture &fx, uint32_t dst, uint32_t a, uint32_t b, uint32_t c) {
              amdgpu::exec_f32(*fx.cu, 16, 16, 128, 1, 8, dst, a, b, c, amdgpu::extract_fp8,
                               amdgpu::extract_fp8, const_acc, p.cbsz, p.abid, p.blgp);
            });
        if (::testing::Test::HasFatalFailure())
          return;
      }
}

TEST(MfmaRegisterPathsTest, I8MatchesScalarForBothShapesAndLanePermutations) {
  SKIP_IF_NO_SIMD();
  struct Mfma {
    uint32_t m, n, k, d_regs;
  };
  struct Permutation {
    uint32_t cbsz, abid, blgp;
  };
  for (const Mfma mfma : {Mfma{16, 16, 64, 4}, Mfma{32, 32, 32, 16}})
    for (Overlap overlap : {Overlap::None, Overlap::Accumulator, Overlap::SourceA})
      for (const Permutation p : {Permutation{0, 0, 0}, Permutation{0, 0, 3}})
        for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, 7u}) {
          const Shape shape{Fmt::I8, Fmt::I8, 4, 4, mfma.d_regs};
          expect_matches_scalar(
              "i8 mfma", shape, overlap, 31 + p.blgp,
              [&](ExactFixture &fx, uint32_t dst, uint32_t a, uint32_t b, uint32_t c) {
                amdgpu::exec_i32_i8(*fx.cu, mfma.m, mfma.n, mfma.k, 1, dst, a, b, c, const_acc,
                                    p.cbsz, p.abid, p.blgp);
              });
          if (::testing::Test::HasFatalFailure())
            return;
        }
}

} // namespace
