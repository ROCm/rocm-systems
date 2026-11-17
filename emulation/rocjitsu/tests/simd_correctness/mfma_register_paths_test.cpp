// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mfma_register_paths_test.cpp
/// @brief Wave64 MFMA register-snapshot paths against the forced-scalar reference.

#include "mma_exact_test_support.h"
#include "simdojo/components/register_file.h"
#include "simdojo/components/vector_reg.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <optional>
#include <string>
#include <vector>

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
    uint32_t m, n, k, batch, bits;
  };
  for (const Mfma s : {Mfma{4, 4, 1, 16, 32}, Mfma{4, 4, 2, 16, 16}, Mfma{4, 4, 4, 16, 16}})
    EXPECT_TRUE(amdgpu::mfma_snapshot_shape_fits<float>(s.m, s.n, s.k, s.batch, s.bits, s.bits))
        << s.m << "x" << s.n << "x" << s.k << "x" << s.batch;
  // f8f6f4 selects the A and B widths independently.
  for (const Mfma s : {Mfma{16, 16, 128, 1, 0}, Mfma{32, 32, 64, 1, 0}})
    for (uint32_t a_bits : {8u, 6u, 4u})
      for (uint32_t b_bits : {8u, 6u, 4u})
        EXPECT_TRUE(amdgpu::mfma_snapshot_shape_fits<float>(s.m, s.n, s.k, s.batch, a_bits, b_bits))
            << s.m << "x" << s.n << "x" << s.k << " a_bits=" << a_bits << " b_bits=" << b_bits;
  for (const Mfma s : {Mfma{16, 16, 64, 1, 8}, Mfma{32, 32, 32, 1, 8}, Mfma{32, 32, 16, 1, 8},
                       Mfma{16, 16, 32, 1, 8}, Mfma{16, 16, 16, 1, 8}, Mfma{32, 32, 8, 1, 8},
                       Mfma{16, 16, 4, 4, 8}, Mfma{4, 4, 4, 16, 8}})
    EXPECT_TRUE(amdgpu::mfma_snapshot_shape_fits<int32_t>(s.m, s.n, s.k, s.batch, 8, 8))
        << s.m << "x" << s.n << "x" << s.k << "x" << s.batch;
  for (const Mfma s : {Mfma{16, 16, 4, 1, 64}, Mfma{4, 4, 4, 4, 64}})
    EXPECT_TRUE(amdgpu::mfma_snapshot_shape_fits<double>(s.m, s.n, s.k, s.batch, 64, 64))
        << s.m << "x" << s.n << "x" << s.k << "x" << s.batch;
  // 32 result registers exceed the snapshot; this shape keeps the generic path.
  EXPECT_FALSE(amdgpu::mfma_snapshot_shape_fits<int32_t>(32, 32, 4, 2, 8, 8));
}

// The snapshot path copies and publishes whole registers, so its layout predicates may only
// accept shapes whose elements use every lane of exactly the registers the snapshot copies.
TEST(MfmaRegisterPathsTest, LayoutPredicatesAcceptOnlyDenseLayouts) {
  constexpr uint32_t kDims[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
  constexpr uint32_t kBatches[] = {1, 2, 3, 4, 8, 16, 32, 64};
  uint32_t accepted = 0;
  for (uint32_t dim : kDims)
    for (uint32_t k : {1u, 2u, 3u, 4u, 8u, 12u, 16u, 32u, 48u, 64u, 128u, 256u})
      for (uint32_t batch : kBatches)
        for (uint32_t bits : {4u, 6u, 8u, 16u, 32u, 64u}) {
          if (!amdgpu::mfma_input_layout_is_dense(dim, k, batch, bits))
            continue;
          ++accepted;
          SCOPED_TRACE(std::format("input {}x{}x{} bits={}", dim, k, batch, bits));
          const uint32_t regs = amdgpu::mfma_dense_reg_count(uint64_t{dim} * k * batch, bits, 64);
          std::vector<bool> used(size_t{regs} * 64);
          for (uint32_t b = 0; b < batch; ++b)
            for (uint32_t i = 0; i < dim; ++i)
              for (uint32_t kk = 0; kk < k; ++kk) {
                const auto loc = amdgpu::input_loc(dim, k, batch, i, kk, b, bits);
                const uint32_t last_reg = loc.vgpr_offset + (loc.bit_offset + bits + 31) / 32 - 1;
                ASSERT_LT(loc.lane, 64u);
                ASSERT_LT(last_reg, regs);
                for (uint32_t reg = loc.vgpr_offset; reg <= last_reg; ++reg)
                  used[size_t{reg} * 64 + loc.lane] = true;
              }
          EXPECT_EQ(std::ranges::count(used, false), 0);
        }
  EXPECT_GT(accepted, 0u);
  accepted = 0;
  for (uint32_t m : kDims)
    for (uint32_t n : kDims)
      for (uint32_t batch : kBatches)
        for (uint32_t words_per_element : {1u, 2u}) {
          if (!amdgpu::mfma_output_layout_is_dense(m, n, batch, words_per_element))
            continue;
          ++accepted;
          SCOPED_TRACE(std::format("output {}x{}x{} words={}", m, n, batch, words_per_element));
          const uint64_t words = uint64_t{m} * n * batch * words_per_element;
          ASSERT_EQ(words % 64, 0u);
          std::vector<bool> written(words);
          for (uint32_t b = 0; b < batch; ++b)
            for (uint32_t i = 0; i < m; ++i)
              for (uint32_t j = 0; j < n; ++j) {
                const auto out = words_per_element == 1 ? amdgpu::output_loc_32(m, n, i, j, b)
                                                        : amdgpu::output_loc_64(m, n, i, j, b);
                ASSERT_LT(out.lane, 64u);
                for (uint32_t w = 0; w < words_per_element; ++w) {
                  const uint64_t word = uint64_t{out.reg + w} * 64 + out.lane;
                  ASSERT_LT(word, words);
                  ASSERT_FALSE(written[word]) << "reg " << out.reg + w << " lane " << out.lane;
                  written[word] = true;
                }
              }
        }
  EXPECT_GT(accepted, 0u);
}

// A direct exec_f32 call with M = N = K = B = 4 places its 64 outputs in lanes 0-15 of four
// registers, which the snapshot cannot publish; it must use the generic path.
TEST(MfmaRegisterPathsTest, SparseOutputLayoutMatchesScalar) {
  SKIP_IF_NO_SIMD();
  EXPECT_FALSE(amdgpu::mfma_snapshot_shape_fits<float>(4, 4, 4, 4, 32, 32));
  const Shape shape{Fmt::F32, Fmt::F32, 1, 1, 4};
  for (Overlap overlap : {Overlap::None, Overlap::Accumulator, Overlap::SourceA, Overlap::SourceB})
    for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, 0x3f800000u}) {
      expect_matches_scalar(
          "f32 4x4x4x4", shape, overlap, /*cross_chunks=*/false, 71,
          [&](ExactFixture &fx, uint32_t dst, uint32_t a, uint32_t b, uint32_t c) {
            amdgpu::exec_f32<4, 4, 4, 4>(*fx.cu, 32, dst, a, b, c, amdgpu::extract_f32,
                                         amdgpu::extract_f32, const_acc);
          });
      if (::testing::Test::HasFatalFailure())
        return;
    }
}

// An operand or destination that runs past the end of the wave's VGPR allocation reads zero
// from, and skips writes to, only the registers outside it.
TEST(MfmaRegisterPathsTest, OperandsCrossingTheAllocationEndMatchScalar) {
  SKIP_IF_NO_SIMD();
  ForceScalarGuard guard;
  ExactFixture fx(ROCJITSU_CODE_ARCH_CDNA4, WF_SIZE);
  ASSERT_NE(fx.wf, nullptr);
  const uint32_t end = fx.cu->vgpr_allocation_block_size();
  ASSERT_GE(end, 48u);
  struct Case {
    const char *name;
    uint32_t a, b, c, d;
  };
  // i8 16x16x64 has four registers in each operand and in the result.
  for (const Case c :
       {Case{"A", end - 2, 4, 8, 16}, Case{"B", 0, end - 2, 8, 16},
        Case{"accumulator", 0, 4, end - 2, 16}, Case{"destination", 0, 4, 8, end - 2}})
    for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, 5u}) {
      SCOPED_TRACE(std::format("{} const_acc={}", c.name, const_acc));
      std::vector<uint32_t> results[2];
      for (bool force_scalar : {true, false}) {
        fx.seed_words(0, 24, 81);
        fx.seed_words(end - 24, 24, 82);
        util::set_force_scalar_for_testing(force_scalar);
        amdgpu::exec_i32_i8(*fx.cu, 16, 16, 64, 1, fx.vbase + c.d, fx.vbase + c.a, fx.vbase + c.b,
                            fx.vbase + c.c, const_acc);
        results[force_scalar] = fx.snapshot(0, 24);
        const auto tail = fx.snapshot(end - 24, 24);
        results[force_scalar].insert(results[force_scalar].end(), tail.begin(), tail.end());
      }
      EXPECT_EQ(results[false], results[true]);
    }
}

// Integer accumulation wraps. On a host with eight int32 lanes the four-column shape also runs
// the matmul's scalar tail.
TEST(MfmaRegisterPathsTest, I8AccumulationWrapsOnBothPaths) {
  SKIP_IF_NO_SIMD();
  ForceScalarGuard guard;
  for (bool force_scalar : {true, false}) {
    SCOPED_TRACE(force_scalar);
    ExactFixture fx(ROCJITSU_CODE_ARCH_CDNA4, WF_SIZE);
    ASSERT_NE(fx.wf, nullptr);
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      fx.cu->write_vgpr(fx.vbase + kSourceA, lane, 0x01010101u);
      fx.cu->write_vgpr(fx.vbase + kSourceB, lane, 0x01010101u);
      for (uint32_t reg = 0; reg < 4; ++reg)
        fx.cu->write_vgpr(fx.vbase + kAccumulator + reg, lane, 0x7ffffffeu);
    }
    util::set_force_scalar_for_testing(force_scalar);
    amdgpu::exec_i32_i8(*fx.cu, 4, 4, 4, 16, fx.vbase + kDestination, fx.vbase + kSourceA,
                        fx.vbase + kSourceB, fx.vbase + kAccumulator);
    for (uint32_t word : fx.snapshot(kDestination, 4))
      ASSERT_EQ(word, 0x80000002u);
  }
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
                amdgpu::exec_f32<16, 16, 128, 1>(*fx.cu, 8, dst, a, b, c, amdgpu::extract_fp8,
                                                 amdgpu::extract_fp8, const_acc, p.cbsz, p.abid,
                                                 p.blgp);
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
                          if (mfma.m == 16)
                            amdgpu::exec_f32_mixed<16, 16, 128, 1>(*fx.cu, a_bits, b_bits, dst, a,
                                                                   b, c, ea, eb, const_acc);
                          else
                            amdgpu::exec_f32_mixed<32, 32, 64, 1>(*fx.cu, a_bits, b_bits, dst, a, b,
                                                                  c, ea, eb, const_acc);
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
              amdgpu::exec_f32<4, 4, 2, 16>(*fx.cu, 16, dst, a, b, c, amdgpu::extract_bf16,
                                            amdgpu::extract_bf16, const_acc, p.cbsz, p.abid,
                                            p.blgp);
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
