// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file floating_operation_test.cpp
/// @brief Source and output modifier stage order around a raw-bit operation.

#include "rocjitsu/isa/arch/amdgpu/shared/floating_operation.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/minmax.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_modifier.h"
#include "util/simd.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

namespace denorm = rocjitsu::amdgpu::input_denormal;
namespace fmt = rocjitsu::amdgpu::fp_format;
namespace mm = rocjitsu::amdgpu::minmax;
namespace om = rocjitsu::amdgpu::output_modifier;
namespace fp = rocjitsu::amdgpu::floating_operation;

// Exercise the common wrapper with explicit results, including every source bit.
// The same cases run on scalar values and SIMD lanes.
template <typename Fmt, typename V> void expect_modifier_stage_order() {
  using L = typename Fmt::Lane;
  SCOPED_TRACE(Fmt::kWidth);
  constexpr L one = (Fmt::kExponentMax >> 1) << Fmt::kMantissaBits;
  constexpr L half = one - Fmt::kMinNormal;
  constexpr L three_quarters = half | (Fmt::kMinNormal >> 1);
  constexpr L two = one + Fmt::kMinNormal;
  constexpr L three = two | (Fmt::kMinNormal >> 1);
  const auto expect_bits = [](V actual, L expected) {
    if constexpr (std::is_same_v<V, L>) {
      EXPECT_EQ(actual, expected);
    } else {
      for (std::size_t lane = 0; lane < V::size(); ++lane)
        EXPECT_EQ(actual[lane], expected);
    }
  };

  // Each source is modified and flushed in its original format. Keep OMOD
  // disabled here: its own subnormal handling would hide a missing input flush.
  for (uint32_t mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    for (unsigned source = 0; source < 3; ++source) {
      std::array<V, 3> values{V(L{0}), V(L{0}), V(L{0})};
      values[source] = V(L{1});
      const fp::WithModifiers<Fmt, mm::Operation<Fmt, mm::Min3Num>> negative{
          {0u, 1u << source}, {}, {denorm::Policy::make(mode)}};
      expect_bits(negative(values[0], values[1], values[2]), Fmt::kSign | L(mode & 1u));
      values[source] = V(Fmt::kSign | L{1});
      const fp::WithModifiers<Fmt, mm::Operation<Fmt, mm::Max3Num>> positive{
          {1u << source, 0u}, {}, {denorm::Policy::make(mode)}};
      expect_bits(positive(values[0], values[1], values[2]), L(mode & 1u));
    }
  }

  // ABS on src1 and NEG on src2: max(0, abs(-1), -2) = 1.
  const fp::WithModifiers<Fmt, mm::Operation<Fmt, mm::Max3Num>> three_sources{
      {2u, 4u}, {}, {denorm::Policy::make(3)}};
  expect_bits(three_sources(V(L{0}), V(Fmt::kSign | one), V(two)), one);
  const fp::WithModifiers<Fmt, mm::Operation<Fmt, mm::Med3Num>> median{
      {0u, 1u}, {}, {denorm::Policy::make(3)}};
  expect_bits(median(V(one), V(two), V(three)), two);

  // ABS precedes NEG, and each applies once: min(-abs(-0.75), 0.5) = -0.75.
  const fp::WithModifiers<Fmt, mm::Operation<Fmt, mm::MinNum>> source_order{
      {1u, 1u}, {}, {denorm::Policy::make(3)}};
  expect_bits(source_order(V(Fmt::kSign | three_quarters), V(half)), Fmt::kSign | three_quarters);

  // ABS -> min -> multiply by 2 -> clamp: min(abs(-0.75), 2) * 2 clamps to 1.
  // Clamping before OMOD would leave 1.5 instead.
  om::Policy output;
  output.omod = 1;
  output.clamp = true;
  const fp::WithModifiers<Fmt, mm::Operation<Fmt, mm::MinNum>> all_stages{
      {1u, 0u}, output, {denorm::Policy::make(3)}};
  expect_bits(all_stages(V(Fmt::kSign | three_quarters), V(two)), one);
}

TEST(FloatingOperationTest, ScalarModifierStageOrder) {
  expect_modifier_stage_order<fmt::F16, uint32_t>();
  expect_modifier_stage_order<fmt::F32, uint32_t>();
  expect_modifier_stage_order<fmt::F64, uint64_t>();
}

TEST(FloatingOperationTest, SimdModifierStageOrder) {
#if __has_include(<experimental/simd>)
  expect_modifier_stage_order<fmt::F16, util::native<uint32_t>>();
  expect_modifier_stage_order<fmt::F32, util::native<uint32_t>>();
#if UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS
  expect_modifier_stage_order<fmt::F64, util::stdx::fixed_size_simd<uint64_t, 1>>();
#else
  expect_modifier_stage_order<fmt::F64, util::native<uint64_t>>();
#endif
#else
  GTEST_SKIP() << "<experimental/simd> unavailable";
#endif
}

} // namespace
