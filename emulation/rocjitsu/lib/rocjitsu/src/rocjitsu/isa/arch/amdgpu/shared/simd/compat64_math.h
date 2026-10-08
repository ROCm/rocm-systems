// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_COMPAT64_MATH_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_COMPAT64_MATH_H_

#include "rocjitsu/isa/arch/amdgpu/shared/simd/abi.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/common.h"

namespace rocjitsu::amdgpu {

namespace simd_backend {
inline util::native<double> clamp_f64(util::native<double> v, bool clamp_nan_to_zero) {
  return util::map_native64_scalar<double>(v, [clamp_nan_to_zero](double x) {
    if (std::isnan(x))
      return clamp_nan_to_zero ? 0.0 : x;
    if (x <= 0.0)
      return 0.0;
    if (x > 1.0)
      return 1.0;
    return x;
  });
}
inline util::native<uint64_t> flush_f64_inputs(util::native<uint64_t> bits) {
  return util::map_native64_scalar<uint64_t>(bits, [](uint64_t lane) {
    const bool denormal =
        (lane & 0x7ff0000000000000ULL) == 0 && (lane & 0x000fffffffffffffULL) != 0;
    return denormal ? lane & 0x8000000000000000ULL : lane;
  });
}
inline util::native<uint64_t> finish_f64_omod(util::native<uint64_t> bits) {
  return util::map_native64_scalar<uint64_t>(bits, [](uint64_t lane) {
    const bool subnormal =
        (lane & 0x7ff0000000000000ULL) == 0 && (lane & 0x000fffffffffffffULL) != 0;
    if (subnormal)
      lane &= 0x8000000000000000ULL;
    return (lane & 0x7fffffffffffffffULL) == 0 ? 0 : lane;
  });
}
} // namespace simd_backend

template <typename T, typename CmpOp>
  requires(util::has_stdx_simd)
inline uint64_t cmp_bits64(util::native<T> a, util::native<T> b, CmpOp cmp_op) {
  constexpr std::size_t W = util::native_width64;
  alignas(64) T abuf[W];
  alignas(64) T bbuf[W];
  a.copy_to(abuf, util::stdx::element_aligned);
  b.copy_to(bbuf, util::stdx::element_aligned);
  uint64_t bits = 0;
  for (std::size_t i = 0; i < W; ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      using One = util::stdx::fixed_size_simd<T, 1>;
      const One av(&abuf[i], util::stdx::element_aligned);
      const One bv(&bbuf[i], util::stdx::element_aligned);
      if (cmp_op(av, bv)[0])
        bits |= uint64_t{1} << i;
    } else if (cmp_op(abuf[i], bbuf[i])) {
      bits |= uint64_t{1} << i;
    }
  }
  return bits;
}

template <typename CmpOp>
  requires(util::has_stdx_simd)
inline uint64_t cmp_class_f64_bits(util::native<uint64_t> s, util::narrow32<uint32_t> mask,
                                   CmpOp cmp_op) {
  constexpr std::size_t W = util::native_width64;
  alignas(64) uint64_t sbuf[W];
  alignas(64) uint32_t mbuf[W];
  s.copy_to(sbuf, util::stdx::element_aligned);
  mask.copy_to(mbuf, util::stdx::element_aligned);
  uint64_t bits = 0;
  for (std::size_t i = 0; i < W; ++i) {
    using One64 = util::stdx::fixed_size_simd<uint64_t, 1>;
    using One32 = util::stdx::fixed_size_simd<uint32_t, 1>;
    const One64 sv(&sbuf[i], util::stdx::element_aligned);
    const One32 mv(&mbuf[i], util::stdx::element_aligned);
    if (cmp_op(sv, mv)[0])
      bits |= uint64_t{1} << i;
  }
  return bits;
}

/// Vector exponent classification and exact normal-range scaling. Only lanes
/// requiring subnormal rounding, overflow or exceptional-value handling use
/// the scalar architectural primitive.
template <typename Float>
  requires(!(!simd_backend::native64_masks && sizeof(Float) == 8))
inline auto div_scale_simd(util::native<Float> value, util::native<Float> denominator,
                           util::native<Float> numerator, uint32_t rounding, uint32_t denorm) {
  using F = DivisionFormat<Float>;
  using Bits = typename F::Bits;
  using U = util::native<Bits>;
  using I = util::native<std::make_signed_t<Bits>>;
  U v = std::bit_cast<U>(value), d = std::bit_cast<U>(denominator), n = std::bit_cast<U>(numerator);
  auto exp = [](U x) {
    return util::stdx::static_simd_cast<I>((x & U(F::infinity)) >> F::fraction);
  };
  if (!(denorm & 1u)) {
    util::stdx::where((v & U(F::infinity)) == U(0), v) = v & U(F::sign);
    util::stdx::where((d & U(F::infinity)) == U(0), d) = d & U(F::sign);
    util::stdx::where((n & U(F::infinity)) == U(0), n) = n & U(F::sign);
  }
  I de = exp(d), ne = exp(n), delta = ne - de;
  I adjustment(0);
  // Apply conditions from last to first to retain the scalar rule priority.
  util::stdx::where(ne <= I(sizeof(Float) == 4 ? 24 : 53), adjustment) = I(F::scale);
  const auto denominator_selected =
      util::stdx::static_simd_cast<I>(v) == util::stdx::static_simd_cast<I>(d);
  auto post = delta <= I(-F::threshold);
  util::stdx::where(post && !denominator_selected, adjustment) = I(F::scale);
  util::stdx::where(post && denominator_selected, adjustment) = I(0);
  const auto large_denominator = de >= I(2 * F::bias - 1);
  util::stdx::where(large_denominator, adjustment) = I(-F::scale);
  util::stdx::where(large_denominator && post && !denominator_selected, adjustment) = I(0);
  util::stdx::where(de == I(0), adjustment) = I(F::scale);
  post = post && de != I(0);
  const auto large_delta = delta >= I(F::threshold);
  util::stdx::where(large_delta, adjustment) = I(0);
  util::stdx::where(large_delta && denominator_selected, adjustment) = I(F::scale);
  post = post || large_delta;
  I ve = exp(v), scaled_exp = ve + adjustment;
  U result = (v & U(~F::infinity)) | (util::stdx::static_simd_cast<U>(scaled_exp) << F::fraction);
  uint64_t post_bits = util::simd_mask_to_bits(post);
  for (std::size_t i = 0; i < U::size(); ++i) {
    if (ve[i] == 0 || ve[i] == int(F::infinity >> F::fraction) || scaled_exp[i] <= 0 ||
        scaled_exp[i] >= int(F::infinity >> F::fraction) || (d[i] & ~F::sign) == 0 ||
        (n[i] & ~F::sign) == 0) {
      const auto scalar =
          div_scale<Float>(value[i], denominator[i], numerator[i], rounding, denorm);
      result[i] = std::bit_cast<Bits>(scalar.value);
      post_bits = (post_bits & ~(uint64_t{1} << i)) | (uint64_t(scalar.post_scale) << i);
    }
  }
  return std::pair{std::bit_cast<util::native<Float>>(result), post_bits};
}

template <typename Float>
  requires(sizeof(Float) == 8)
inline auto div_scale_simd(util::native<Float> value, util::native<Float> denominator,
                           util::native<Float> numerator, uint32_t rounding, uint32_t denorm) {
  util::native<Float> result(0);
  uint64_t post_bits = 0;
  for (std::size_t lane = 0; lane < result.size(); ++lane) {
    const auto scalar =
        div_scale<Float>(value[lane], denominator[lane], numerator[lane], rounding, denorm);
    result[lane] = scalar.value;
    post_bits |= uint64_t{scalar.post_scale} << lane;
  }
  return std::pair{result, post_bits};
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_COMPAT64_MATH_H_
