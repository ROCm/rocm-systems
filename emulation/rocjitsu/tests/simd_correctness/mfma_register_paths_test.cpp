// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mfma_register_paths_test.cpp
/// @brief Wave64 MFMA register-snapshot paths against the forced-scalar reference.

#include "mma_exact_test_support.h"
#include "simdojo/components/register_file.h"
#include "simdojo/components/vector_reg.h"

#include <format>
#include <optional>
#include <string>

namespace {

using namespace rocjitsu;
using namespace mma_exact;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t kSourceA = 0;
constexpr uint32_t kSourceB = 16;
constexpr uint32_t kAccumulator = 32;
constexpr uint32_t kDestination = 64;

enum class Overlap { None, Accumulator, SourceA, SourceB };

using MfmaVgpr = simdojo::VectorReg<WF_SIZE, uint32_t>;
using MfmaLazyStorage = simdojo::detail::SoftwareLazyRegisterStorage<MfmaVgpr, 1024>;

struct Shape {
  Fmt input_format;
  Fmt accumulator_format;
  uint32_t a_regs;
  uint32_t b_regs;
  uint32_t d_regs;
  std::optional<Fmt> b_format = std::nullopt; ///< When B's format differs from A's.
};

/// Runs @p kernel forced-scalar and then normally over identical register state, for a
/// destination that is separate from, equal to, or overlapping the sources.
void expect_matches_scalar(
    const char *label, const Shape &shape, Overlap overlap, bool cross_chunks, uint32_t seed,
    const std::function<void(ExactFixture &, uint32_t, uint32_t, uint32_t, uint32_t)> &kernel) {
  ExactFixture fixture(ROCJITSU_CODE_ARCH_CDNA4, WF_SIZE);
  ASSERT_NE(fixture.wf, nullptr);
  const uint32_t offset = cross_chunks ? MfmaLazyStorage::registers_per_chunk() - 1 : 0;
  const uint32_t source_a = kSourceA + offset;
  const uint32_t source_b = kSourceB + offset;
  const uint32_t accumulator = kAccumulator + offset;
  const uint32_t destination = overlap == Overlap::Accumulator ? accumulator
                               : overlap == Overlap::SourceA   ? source_a
                               : overlap == Overlap::SourceB   ? source_b
                                                               : kDestination + offset;
  SCOPED_TRACE(static_cast<int>(overlap));
  SCOPED_TRACE(cross_chunks);
  if (cross_chunks) {
    const auto chunk = [&](uint32_t reg) {
      return (fixture.vbase + reg) / MfmaLazyStorage::registers_per_chunk();
    };
    ASSERT_NE(chunk(source_a), chunk(source_a + shape.a_regs - 1));
    ASSERT_NE(chunk(source_b), chunk(source_b + shape.b_regs - 1));
    ASSERT_NE(chunk(destination), chunk(destination + shape.d_regs - 1));
  }
  // Random words first, so a destination word that is never written cannot match the
  // scalar result; the sources are seeded last, so an overlapping one keeps its data.
  auto reseed = [&] {
    fixture.seed_words(destination, shape.d_regs, seed + 1000);
    if (shape.accumulator_format == Fmt::I8)
      fixture.seed_words(accumulator, shape.d_regs, seed + 3);
    else
      fixture.seed(accumulator, shape.d_regs, shape.accumulator_format, Mode::RandomInt, seed + 3);
    fixture.seed(source_b, shape.b_regs, shape.b_format.value_or(shape.input_format),
                 Mode::RandomInt, seed + 2);
    fixture.seed(source_a, shape.a_regs, shape.input_format, Mode::RandomInt, seed + 1);
  };
  auto run = [&] {
    kernel(fixture, fixture.vbase + destination, fixture.vbase + source_a, fixture.vbase + source_b,
           fixture.vbase + accumulator);
  };
  expect_bit_exact(label, Mode::RandomInt, fixture, reseed, run, destination, shape.d_regs);
}

// Every wave64 shape that the generated MFMA executors route through exec_f32_mixed,
// exec_i32_i8 and exec_f64 and that has at most 16 result registers.
TEST(MfmaRegisterPathsTest, RoutedShapesFitTheSnapshotPath) {
  SKIP_IF_NO_SIMD();
  struct Mfma {
    uint32_t m, n, k, batch;
  };
  for (const Mfma s : {Mfma{4, 4, 1, 16}, Mfma{4, 4, 2, 16}, Mfma{4, 4, 4, 16},
                       Mfma{16, 16, 128, 1}, Mfma{32, 32, 64, 1}})
    EXPECT_TRUE(amdgpu::mfma_regions_shape_fits<float>(s.m, s.n, s.k, s.batch))
        << s.m << "x" << s.n << "x" << s.k << "x" << s.batch;
  for (const Mfma s :
       {Mfma{16, 16, 64, 1}, Mfma{32, 32, 32, 1}, Mfma{32, 32, 16, 1}, Mfma{16, 16, 32, 1},
        Mfma{16, 16, 16, 1}, Mfma{32, 32, 8, 1}, Mfma{16, 16, 4, 4}, Mfma{4, 4, 4, 16}})
    EXPECT_TRUE(amdgpu::mfma_regions_shape_fits<int32_t>(s.m, s.n, s.k, s.batch))
        << s.m << "x" << s.n << "x" << s.k << "x" << s.batch;
  for (const Mfma s : {Mfma{16, 16, 4, 1}, Mfma{4, 4, 4, 4}})
    EXPECT_TRUE(amdgpu::mfma_regions_shape_fits<double>(s.m, s.n, s.k, s.batch))
        << s.m << "x" << s.n << "x" << s.k << "x" << s.batch;
  // 32 result registers exceed the snapshot; this shape keeps the existing path.
  EXPECT_FALSE(amdgpu::mfma_regions_shape_fits<int32_t>(32, 32, 4, 2));
}

TEST(MfmaRegisterPathsTest, F64MatchesScalarForSeparateAndOverlappingDestinations) {
  SKIP_IF_NO_SIMD();
  struct Mfma {
    uint32_t m, n, k, batch, d_regs;
  };
  for (const Mfma mfma : {Mfma{16, 16, 4, 1, 8}, Mfma{4, 4, 4, 4, 2}})
    for (bool cross_chunks : {false, true})
      for (Overlap overlap :
           {Overlap::None, Overlap::Accumulator, Overlap::SourceA, Overlap::SourceB})
        for (uint32_t neg : {0u, 1u, 2u, 4u, 7u})
          for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, 0x40000000u}) {
            const Shape shape{Fmt::F64, Fmt::F64, 2, 2, mfma.d_regs};
            expect_matches_scalar(
                "f64 mfma", shape, overlap, cross_chunks, 11 + neg,
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
  for (bool cross_chunks : {false, true})
    for (Overlap overlap :
         {Overlap::None, Overlap::Accumulator, Overlap::SourceA, Overlap::SourceB})
      for (const Permutation p : {Permutation{0, 0, 0}, Permutation{2, 1, 0}, Permutation{0, 0, 3}})
        for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, 0x3f800000u}) {
          expect_matches_scalar(
              "fp8 16x16x128", shape, overlap, cross_chunks, 21 + p.cbsz,
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
    for (bool cross_chunks : {false, true})
      for (Overlap overlap :
           {Overlap::None, Overlap::Accumulator, Overlap::SourceA, Overlap::SourceB})
        for (const Permutation p : {Permutation{0, 0, 0}, Permutation{0, 0, 3}})
          for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, 7u}) {
            const Shape shape{Fmt::I8, Fmt::I8, 4, 4, mfma.d_regs};
            expect_matches_scalar(
                "i8 mfma", shape, overlap, cross_chunks, 31 + p.blgp,
                [&](ExactFixture &fx, uint32_t dst, uint32_t a, uint32_t b, uint32_t c) {
                  amdgpu::exec_i32_i8(*fx.cu, mfma.m, mfma.n, mfma.k, 1, dst, a, b, c, const_acc,
                                      p.cbsz, p.abid, p.blgp);
                });
            if (::testing::Test::HasFatalFailure())
              return;
          }
}

TEST(MfmaRegisterPathsTest, F8F6F4MatchesScalarForEveryFormatPair) {
  SKIP_IF_NO_SIMD();
  // cbsz and blgp select the A and B formats: FP8, BF8, FP6, BF6, FP4.
  constexpr Fmt kFormats[] = {Fmt::FP8, Fmt::BF8, Fmt::RAW6, Fmt::RAW6, Fmt::RAW4};
  constexpr uint32_t kBits[] = {8, 8, 6, 6, 4};
  struct Mfma {
    uint32_t m, n, k, d_regs;
  };
  for (const Mfma mfma : {Mfma{16, 16, 128, 4}, Mfma{32, 32, 64, 16}})
    for (uint32_t a_fmt = 0; a_fmt < 5; ++a_fmt)
      for (uint32_t b_fmt = 0; b_fmt < 5; ++b_fmt)
        for (bool cross_chunks : {false, true})
          for (Overlap overlap :
               {Overlap::None, Overlap::Accumulator, Overlap::SourceA, Overlap::SourceB})
            for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, 0x3f800000u}) {
              const std::string label =
                  std::format("f8f6f4 {}x{}x{} a={} b={}", mfma.m, mfma.n, mfma.k, a_fmt, b_fmt);
              const Shape shape{kFormats[a_fmt], Fmt::F32,    kBits[a_fmt],
                                kBits[b_fmt],    mfma.d_regs, kFormats[b_fmt]};
              expect_matches_scalar(
                  label.c_str(), shape, overlap, cross_chunks, 41 + a_fmt * 5 + b_fmt,
                  [&](ExactFixture &fx, uint32_t dst, uint32_t a, uint32_t b, uint32_t c) {
                    ASSERT_TRUE(amdgpu::dispatch_matrix_fmt_pair(
                        a_fmt, b_fmt, [&](uint32_t a_bits, uint32_t b_bits, auto ea, auto eb) {
                          amdgpu::exec_f32_mixed(*fx.cu, mfma.m, mfma.n, mfma.k, 1, a_bits, b_bits,
                                                 dst, a, b, c, ea, eb, const_acc);
                        }));
                  });
              if (::testing::Test::HasFatalFailure())
                return;
            }
}

TEST(MfmaRegisterPathsTest, Bf16FourByFourMatchesScalarWithLanePermutations) {
  SKIP_IF_NO_SIMD();
  struct Permutation {
    uint32_t cbsz, abid, blgp;
  };
  // 4x4x2 with 16 blocks has one A and one B register.
  const Shape shape{Fmt::BF16, Fmt::F32, 1, 1, 4};
  for (Overlap overlap : {Overlap::None, Overlap::Accumulator, Overlap::SourceA, Overlap::SourceB})
    for (const Permutation p : {Permutation{0, 0, 0}, Permutation{2, 3, 0}, Permutation{4, 9, 5}})
      for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, 0x3f800000u}) {
        expect_matches_scalar(
            "bf16 4x4x2x16", shape, overlap, /*cross_chunks=*/false, 51 + p.cbsz,
            [&](ExactFixture &fx, uint32_t dst, uint32_t a, uint32_t b, uint32_t c) {
              amdgpu::exec_f32(*fx.cu, 4, 4, 2, 16, 16, dst, a, b, c, amdgpu::extract_bf16,
                               amdgpu::extract_bf16, const_acc, p.cbsz, p.abid, p.blgp);
            });
        if (::testing::Test::HasFatalFailure())
          return;
      }
}

TEST(MfmaRegisterPathsTest, I8SingleRegisterShapesMatchScalarWithLanePermutations) {
  SKIP_IF_NO_SIMD();
  struct Case {
    uint32_t m, n, k, batch, d_regs, cbsz, abid, blgp;
  };
  // Each shape reads one A and one B register; batched shapes also broadcast A blocks.
  for (const Case c : {Case{32, 32, 8, 1, 16, 0, 0, 3}, Case{16, 16, 16, 1, 4, 0, 0, 5},
                       Case{16, 16, 4, 4, 16, 0, 0, 0}, Case{16, 16, 4, 4, 16, 2, 3, 7},
                       Case{4, 4, 4, 16, 4, 0, 0, 0}, Case{4, 4, 4, 16, 4, 4, 9, 2},
                       Case{4, 4, 4, 16, 4, 1, 1, 6}})
    for (Overlap overlap :
         {Overlap::None, Overlap::Accumulator, Overlap::SourceA, Overlap::SourceB})
      for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, 7u}) {
        const std::string label = std::format("i8 {}x{}x{}x{} cbsz={} abid={} blgp={}", c.m, c.n,
                                              c.k, c.batch, c.cbsz, c.abid, c.blgp);
        const Shape shape{Fmt::I8, Fmt::I8, 1, 1, c.d_regs};
        expect_matches_scalar(
            label.c_str(), shape, overlap, /*cross_chunks=*/false, 61 + c.cbsz,
            [&](ExactFixture &fx, uint32_t dst, uint32_t a, uint32_t b, uint32_t acc) {
              amdgpu::exec_i32_i8(*fx.cu, c.m, c.n, c.k, c.batch, dst, a, b, acc, const_acc, c.cbsz,
                                  c.abid, c.blgp);
            });
        if (::testing::Test::HasFatalFailure())
          return;
      }
}

} // namespace
