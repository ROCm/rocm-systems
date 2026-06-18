// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_modifier_simd_test.cpp
/// @brief Bit-exactness of the in-vector VOP3 source/destination modifier
/// helpers against the scalar reference the generated bodies use. VOP3 ops
/// carry per-source abs/neg and per-instruction omod/clamp; the SIMD fast path
/// must apply them identically to the scalar lambda nest
/// (execute_shared.h: abs(std::fabs) -> neg(-x) per source, then omod(*2/*4/*0.5)
/// -> clamp(std::clamp(v,0,1)) on the result). These tests pin every modifier
/// combination over normal/NaN/Inf/denormal/signed-zero inputs.

#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"

#include "util/simd.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace {

using rocjitsu::amdgpu::apply_vop3_dst_mod_f32;
using rocjitsu::amdgpu::apply_vop3_dst_mod_f64;
using rocjitsu::amdgpu::apply_vop3_src_mod_f32;
using rocjitsu::amdgpu::apply_vop3_src_mod_f64;

#define SKIP_IF_NO_SIMD()                                                                          \
  if constexpr (!util::has_stdx_simd) {                                                            \
    GTEST_SKIP() << "<experimental/simd> unavailable; VOP3 modifier helpers skipped.";             \
    return;                                                                                        \
  }

// Curated bit patterns: ±normal, ±0, ±Inf, qNaN, sNaN, ±denormal, >1 and in [0,1].
constexpr std::array<uint32_t, 12> kPatterns = {
    0x3F000000u, // 0.5
    0x40490FDBu, // pi (~3.14, > 1)
    0xC0490FDBu, // -pi
    0x00000000u, // +0
    0x80000000u, // -0
    0x7F800000u, // +Inf
    0xFF800000u, // -Inf
    0x7FC00000u, // qNaN
    0x7F800001u, // sNaN
    0x00000001u, // +denormal
    0x80000001u, // -denormal
    0x3DCCCCCDu, // 0.1 (in [0,1])
};

constexpr std::array<uint64_t, 12> kF64SrcPatterns = {
    0x3FE0000000000000ull, // 0.5
    0x400921FB54442D18ull, // pi (> 1)
    0xC00921FB54442D18ull, // -pi
    0x0000000000000000ull, // +0
    0x8000000000000000ull, // -0
    0x7FF0000000000000ull, // +Inf
    0xFFF0000000000000ull, // -Inf
    0x7FF8000000000000ull, // qNaN
    0x7FF0000000000001ull, // sNaN
    0x0000000000000001ull, // +denormal
    0x8000000000000001ull, // -denormal
    0x3FB999999999999Aull, // 0.1 (in [0,1])
};

constexpr std::array<uint64_t, 11> kF64DstPatterns = {
    0x3FE0000000000000ull, // 0.5
    0x400921FB54442D18ull, // pi (> 1)
    0xC00921FB54442D18ull, // -pi
    0x0000000000000000ull, // +0
    0x8000000000000000ull, // -0
    0x7FF0000000000000ull, // +Inf
    0xFFF0000000000000ull, // -Inf
    0x7FF8000000000000ull, // qNaN
    0x0000000000000001ull, // +denormal
    0x8000000000000001ull, // -denormal
    0x3FB999999999999Aull, // 0.1 (in [0,1])
};

float ref_src(float x, bool abs, bool neg) {
  if (abs)
    x = std::fabs(x);
  if (neg)
    x = -x;
  return x;
}

double ref_src(double x, bool abs, bool neg) {
  if (abs)
    x = std::fabs(x);
  if (neg)
    x = -x;
  return x;
}

float ref_dst(float v, uint32_t omod, bool clamp) {
  if (omod == 1)
    v *= 2.0f;
  else if (omod == 2)
    v *= 4.0f;
  else if (omod == 3)
    v *= 0.5f;
  if (clamp)
    v = std::clamp(v, 0.0f, 1.0f);
  return v;
}

double ref_dst(double v, uint32_t omod, bool clamp) {
  if (omod == 1)
    v *= 2.0;
  else if (omod == 2)
    v *= 4.0;
  else if (omod == 3)
    v *= 0.5;
  if (clamp)
    v = std::clamp(v, 0.0, 1.0);
  return v;
}

TEST(Vop3ModifierSimd, SrcMod_AllAbsNegCombos_BitExact) {
  SKIP_IF_NO_SIMD();
  constexpr std::size_t W = util::native_width_v<float>;
  alignas(util::native<float>) float in[W];
  for (std::size_t i = 0; i < W; ++i)
    in[i] = std::bit_cast<float>(kPatterns[i % kPatterns.size()]);
  const util::native<float> v(in, util::stdx::element_aligned);

  for (uint32_t abs = 0; abs < 2; ++abs) {
    for (uint32_t neg = 0; neg < 2; ++neg) {
      // SrcIdx 0 reads abs/neg bit 0.
      auto out0 = apply_vop3_src_mod_f32<0>(v, abs ? 1u : 0u, neg ? 1u : 0u);
      alignas(util::native<float>) float o0[W];
      out0.copy_to(o0, util::stdx::element_aligned);
      // SrcIdx 1 reads abs/neg bit 1.
      auto out1 = apply_vop3_src_mod_f32<1>(v, abs ? 2u : 0u, neg ? 2u : 0u);
      alignas(util::native<float>) float o1[W];
      out1.copy_to(o1, util::stdx::element_aligned);
      for (std::size_t i = 0; i < W; ++i) {
        const float r = ref_src(in[i], abs != 0, neg != 0);
        EXPECT_EQ(std::bit_cast<uint32_t>(o0[i]), std::bit_cast<uint32_t>(r))
            << "src0 abs=" << abs << " neg=" << neg << " lane " << i;
        EXPECT_EQ(std::bit_cast<uint32_t>(o1[i]), std::bit_cast<uint32_t>(r))
            << "src1 abs=" << abs << " neg=" << neg << " lane " << i;
      }
    }
  }
}

TEST(Vop3ModifierSimd, SrcModF64_AllAbsNegCombos_BitExact) {
  SKIP_IF_NO_SIMD();
  constexpr std::size_t W = util::native_width64;
  alignas(util::native<double>) double in[W];
  for (std::size_t i = 0; i < W; ++i)
    in[i] = std::bit_cast<double>(kF64SrcPatterns[i % kF64SrcPatterns.size()]);
  const util::native<double> v(in, util::stdx::element_aligned);

  for (uint32_t abs = 0; abs < 2; ++abs) {
    for (uint32_t neg = 0; neg < 2; ++neg) {
      auto out0 = apply_vop3_src_mod_f64<0>(v, abs ? 1u : 0u, neg ? 1u : 0u);
      alignas(util::native<double>) double o0[W];
      out0.copy_to(o0, util::stdx::element_aligned);
      auto out1 = apply_vop3_src_mod_f64<1>(v, abs ? 2u : 0u, neg ? 2u : 0u);
      alignas(util::native<double>) double o1[W];
      out1.copy_to(o1, util::stdx::element_aligned);
      for (std::size_t i = 0; i < W; ++i) {
        const double r = ref_src(in[i], abs != 0, neg != 0);
        EXPECT_EQ(std::bit_cast<uint64_t>(o0[i]), std::bit_cast<uint64_t>(r))
            << "src0 abs=" << abs << " neg=" << neg << " lane " << i;
        EXPECT_EQ(std::bit_cast<uint64_t>(o1[i]), std::bit_cast<uint64_t>(r))
            << "src1 abs=" << abs << " neg=" << neg << " lane " << i;
      }
    }
  }
}

TEST(Vop3ModifierSimd, DstMod_AllOmodClampCombos_BitExact) {
  SKIP_IF_NO_SIMD();
  constexpr std::size_t W = util::native_width_v<float>;
  alignas(util::native<float>) float in[W];
  for (std::size_t i = 0; i < W; ++i)
    in[i] = std::bit_cast<float>(kPatterns[i % kPatterns.size()]);
  const util::native<float> v(in, util::stdx::element_aligned);

  for (uint32_t omod = 0; omod < 4; ++omod) {
    for (uint32_t clamp = 0; clamp < 2; ++clamp) {
      auto out = apply_vop3_dst_mod_f32(v, omod, clamp);
      alignas(util::native<float>) float o[W];
      out.copy_to(o, util::stdx::element_aligned);
      for (std::size_t i = 0; i < W; ++i) {
        const float r = ref_dst(in[i], omod, clamp != 0);
        EXPECT_EQ(std::bit_cast<uint32_t>(o[i]), std::bit_cast<uint32_t>(r))
            << "omod=" << omod << " clamp=" << clamp << " lane " << i;
      }
    }
  }
}

TEST(Vop3ModifierSimd, DstModF64_AllOmodClampCombos_BitExact) {
  SKIP_IF_NO_SIMD();
  constexpr std::size_t W = util::native_width64;
  alignas(util::native<double>) double in[W];
  for (std::size_t i = 0; i < W; ++i)
    in[i] = std::bit_cast<double>(kF64DstPatterns[i % kF64DstPatterns.size()]);
  const util::native<double> v(in, util::stdx::element_aligned);

  for (uint32_t omod = 0; omod < 4; ++omod) {
    for (uint32_t clamp = 0; clamp < 2; ++clamp) {
      auto out = apply_vop3_dst_mod_f64(v, omod, clamp);
      alignas(util::native<double>) double o[W];
      out.copy_to(o, util::stdx::element_aligned);
      for (std::size_t i = 0; i < W; ++i) {
        const double r = ref_dst(in[i], omod, clamp != 0);
        EXPECT_EQ(std::bit_cast<uint64_t>(o[i]), std::bit_cast<uint64_t>(r))
            << "omod=" << omod << " clamp=" << clamp << " lane " << i;
      }
    }
  }
}

} // namespace
