// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PORTABLE_MIXED_FMA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PORTABLE_MIXED_FMA_H_

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/abi.h"

namespace rocjitsu::amdgpu {

// The selected backend supplies a compatible or native double-lane ABI.
using MixedFmaAbi = simd_backend::mixed_fma_abi;
using MixedFmaDouble = util::stdx::simd<double, MixedFmaAbi>;
using MixedFmaBits = util::stdx::simd<uint64_t, MixedFmaAbi>;
inline constexpr std::size_t mixed_fma_width64 = MixedFmaDouble::size();

/// Narrow finite F64 lanes directly to F16, without an intermediate F32 rounding.
inline MixedFmaBits round_finite_f16_simd(MixedFmaDouble value, uint32_t round_mode,
                                          bool fp16_ovfl) {
  using U = MixedFmaBits;
  const U bits = std::bit_cast<U>(value);
  const U sign = (bits >> 48) & U(0x8000u);
  const U exponent = (bits >> 52) & U(0x7ffu);
  const U significand = (bits & U(0x000fffffffffffffull)) | U(1ull << 52);
  U shift(42);
  util::stdx::where(exponent < U(1009), shift) = U(1051) - exponent;
  util::stdx::where(exponent < U(998), shift) = U(53);
  const U remainder = significand & ((U(1) << shift) - U(1));
  U rounded = significand >> shift;
  util::stdx::where(exponent >= U(1009), rounded) += (exponent - U(1009)) << 10;
  const auto inexact = remainder != U(0);
  switch (round_mode & 3u) {
  case 0: {
    const U halfway = U(1) << (shift - U(1));
    const auto increment =
        (remainder > halfway) || ((remainder == halfway) && ((rounded & U(1)) != U(0)));
    util::stdx::where(increment && (exponent >= U(998)), rounded) += U(1);
    break;
  }
  case 1:
    util::stdx::where(inexact && (sign == U(0)), rounded) += U(1);
    break;
  case 2:
    util::stdx::where(inexact && (sign != U(0)), rounded) += U(1);
    break;
  default:
    break;
  }
  U overflow(0x7bffu);
  if (!fp16_ovfl) {
    if ((round_mode & 3u) == 0)
      overflow = U(0x7c00u);
    else if ((round_mode & 3u) == 1)
      util::stdx::where(sign == U(0), overflow) = U(0x7c00u);
    else if ((round_mode & 3u) == 2)
      util::stdx::where(sign != U(0), overflow) = U(0x7c00u);
  }
  util::stdx::where((exponent > U(1038)) || (rounded >= U(0x7c00u)), rounded) = overflow;
  util::stdx::where((bits & U(0x7fffffffffffffffull)) == U(0), rounded) = U(0);
  return sign | rounded;
}

/// MIX F16 keeps the exact F32 product and addition residual in F64 SIMD lanes.
/// Only nonfinite inputs need scalar NaN selection. Legacy MAD passes its
/// already rounded F32 result as c and uses the same vector narrowing policy.
template <bool Fused>
#if defined(__GNUC__) && !defined(__clang__)
[[gnu::optimize("rounding-math")]]
#endif
inline util::native<uint32_t>
mixed_fma_f16_simd(util::native<float> a, util::native<float> b, util::native<float> c,
                   uint32_t round_mode, bool clamp, bool fp16_ovfl, bool clamp_nan_to_zero) {
  using D = MixedFmaDouble;
  using U = MixedFmaBits;
  constexpr std::size_t W = util::native_width_v<float>;
  constexpr std::size_t WD = mixed_fma_width64;
  using NarrowFloat = util::stdx::fixed_size_simd<float, WD>;
  using NarrowWord = util::stdx::fixed_size_simd<uint32_t, WD>;
  alignas(util::native<float>) float av[W], bv[W], cv[W];
  alignas(util::native<uint32_t>) uint32_t out[W];
  a.copy_to(av, util::stdx::vector_aligned);
  b.copy_to(bv, util::stdx::vector_aligned);
  c.copy_to(cv, util::stdx::vector_aligned);
  fp_mode::ScopedEnvironment environment(0);
  for (std::size_t base = 0; base < W; base += WD) {
    const D da =
        util::stdx::static_simd_cast<D>(NarrowFloat(av + base, util::stdx::element_aligned));
    const D db =
        util::stdx::static_simd_cast<D>(NarrowFloat(bv + base, util::stdx::element_aligned));
    const D dc =
        util::stdx::static_simd_cast<D>(NarrowFloat(cv + base, util::stdx::element_aligned));
    D value = dc;
    auto special = util::stdx::isinf(dc) || util::stdx::isnan(dc);
    if constexpr (Fused) {
      const D product = da * db;
      value = product + dc;
      const D addend_virtual = value - product;
      const D residual = (product - (value - addend_virtual)) + (dc - addend_virtual);
      U bits = std::bit_cast<U>(value);
      const U error = std::bit_cast<U>(residual);
      const auto step = ((error & U(0x7fffffffffffffffull)) != U(0)) && ((bits & U(1)) == U(0));
      const auto same_sign = ((bits ^ error) >> 63) == U(0);
      util::stdx::where(step && same_sign, bits) += U(1);
      util::stdx::where(step && !same_sign, bits) -= U(1);
      const U product_sign = (std::bit_cast<U>(da) ^ std::bit_cast<U>(db)) & U(1ull << 63);
      const U addend = std::bit_cast<U>(dc);
      const auto matching_zeros = ((addend & U(0x7fffffffffffffffull)) == U(0)) &&
                                  ((addend & U(1ull << 63)) == product_sign);
      U zero_sign((round_mode & 3u) == 2 ? 1ull << 63 : 0);
      util::stdx::where(matching_zeros, zero_sign) = product_sign;
      util::stdx::where((bits & U(0x7fffffffffffffffull)) == U(0), bits) = zero_sign;
      value = std::bit_cast<D>(bits);
      special |= util::stdx::isinf(da) || util::stdx::isnan(da) || util::stdx::isinf(db) ||
                 util::stdx::isnan(db);
    }
    if (clamp) {
      util::stdx::where(value <= D(0), value) = D(0);
      util::stdx::where(value > D(1), value) = D(1);
    }
    const auto half = util::stdx::static_simd_cast<NarrowWord>(
        round_finite_f16_simd(value, round_mode, fp16_ovfl));
    half.copy_to(out + base, util::stdx::element_aligned);
    if (util::stdx::any_of(special))
      for (std::size_t lane = 0; lane < WD; ++lane)
        if (special[lane]) {
          const auto i = base + lane;
          if constexpr (Fused)
            out[i] = fp_mode::detail::fma_f32_to_f16_nearest_environment(
                av[i], bv[i], cv[i], round_mode, clamp, fp16_ovfl, clamp_nan_to_zero);
          else
            out[i] = pseudo_scalar::round_f16_result(cv[i], round_mode, 0, clamp, fp16_ovfl,
                                                     clamp_nan_to_zero);
        }
  }
  return util::native<uint32_t>(out, util::stdx::vector_aligned);
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PORTABLE_MIXED_FMA_H_
