// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PORTABLE_MATH_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PORTABLE_MATH_H_

#include "rocjitsu/isa/arch/amdgpu/shared/mixed_fma_simd.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/common.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/math64.h"

namespace rocjitsu::amdgpu {

/// In-vector VOP3 destination modifier, bit-exact with the scalar tail: `omod`
/// scales by an exact power of two (1->*2, 2->*4, 3->*0.5; IEEE-exact, no
/// rounding), then `clamp` saturates to [0,1]. The clamp uses ordered compares
/// (`v < 0`, `v > 1`), which are false for NaN, so NaN passes through unchanged —
/// using the selected architecture/MODE NaN policy. Instantiated for float and double;
/// the `_f32`/`_f64` wrappers below name the two lane types the VOP3 paths use.
template <typename T>
util::native<T> apply_vop3_dst_mod(util::native<T> v, uint32_t omod, uint32_t clamp,
                                   bool clamp_nan_to_zero) {
  if constexpr (std::is_same_v<T, float>) {
    if (omod != 0) {
      using U = util::native<uint32_t>;
      U bits = std::bit_cast<U>(v);
      util::stdx::where((bits & U(0x7fffffffu)) < U(0x00800000u), bits) = U(0);
      v = std::bit_cast<util::native<float>>(bits);
    }
  }
  const auto unscaled = v;
  if (omod == 1)
    v = v * T(2);
  else if (omod == 2)
    v = v * T(4);
  else if (omod == 3)
    v = v * T(0.5);
  if constexpr (std::is_same_v<T, float>) {
    using U = util::native<uint32_t>;
    U bits = std::bit_cast<U>(v);
    const U original = std::bit_cast<U>(unscaled);
    util::stdx::where((original & U(0x7fffffffu)) > U(0x7f800000u), bits) = original;
    if (omod == 3)
      util::stdx::where((original & U(0x7fffffffu)) < U(0x01000000u), bits) =
          original & U(0x80000000u);
    v = std::bit_cast<util::native<float>>(bits);
  }
  if (clamp) {
    if constexpr (std::is_same_v<T, double>) {
      v = simd_backend::clamp_f64(v, clamp_nan_to_zero);
    } else {
      if (clamp_nan_to_zero)
        util::stdx::where(util::stdx::isnan(v), v) = T(0);
      util::stdx::where(v <= T(0), v) = T(0);
      util::stdx::where(v > T(1), v) = T(1);
    }
  }
  if (omod != 0) {
    if constexpr (std::is_same_v<T, float>) {
      using U = util::native<uint32_t>;
      U bits = std::bit_cast<U>(v);
      const auto subnormal = ((bits & U(0x7f800000u)) == U(0)) && ((bits & U(0x007fffffu)) != U(0));
      // Existing zeros were canonicalized before scaling. Preserve the sign
      // of a newly underflowed result.
      util::stdx::where(subnormal, bits) = bits & U(0x80000000u);
      v = std::bit_cast<util::native<float>>(bits);
    } else {
      using U = util::native<uint64_t>;
      U bits = std::bit_cast<U>(v);
      bits = simd_backend::finish_f64_omod(bits);
      v = std::bit_cast<util::native<double>>(bits);
    }
  }
  return v;
}

/// @brief Execute a MODE-aware native batch of architectural F64 FMAs.
#if defined(__GNUC__) && !defined(__clang__)
// Keep hardware FMA inside the guest rounding environment when GCC inlines it.
[[gnu::optimize("rounding-math")]]
#endif
inline util::native<double> fma_f64_mode_simd(util::native<double> src0, util::native<double> src1,
                                              util::native<double> src2, uint32_t round_mode,
                                              uint32_t denorm_mode) {
  using U = util::native<uint64_t>;
  const U original0 = std::bit_cast<U>(src0);
  const U original1 = std::bit_cast<U>(src1);
  const U original2 = std::bit_cast<U>(src2);
  auto flush = [](util::native<double> value) {
    U bits = std::bit_cast<U>(value);
    bits = simd_backend::flush_f64_inputs(bits);
    return std::bit_cast<util::native<double>>(bits);
  };
  if ((denorm_mode & 1u) == 0) {
    src0 = flush(src0);
    src1 = flush(src1);
    src2 = flush(src2);
  }

  util::native<double> result;
  {
    fp_mode::ScopedEnvironment environment(round_mode);
    result = util::stdx::fma(src0, src1, src2);
  }
  if ((denorm_mode & 2u) == 0)
    result = flush(result);

  constexpr std::size_t W = util::native_width64;
  alignas(U) uint64_t a_bits[W];
  alignas(U) uint64_t b_bits[W];
  alignas(U) uint64_t c_bits[W];
  alignas(U) uint64_t result_bits[W];
  original0.copy_to(a_bits, util::stdx::vector_aligned);
  original1.copy_to(b_bits, util::stdx::vector_aligned);
  original2.copy_to(c_bits, util::stdx::vector_aligned);
  std::bit_cast<U>(result).copy_to(result_bits, util::stdx::vector_aligned);
  for (std::size_t i = 0; i < W; ++i) {
    if (std::isnan(std::bit_cast<double>(a_bits[i])) ||
        std::isnan(std::bit_cast<double>(b_bits[i])) ||
        std::isnan(std::bit_cast<double>(c_bits[i])))
      result_bits[i] = fp_mode::fma_f64(a_bits[i], b_bits[i], c_bits[i], round_mode, denorm_mode);
  }
  return std::bit_cast<util::native<double>>(U(result_bits, util::stdx::vector_aligned));
}

inline util::native<uint32_t> simd_sign_extend_u32(util::native<uint32_t> v, unsigned bits) {
  assert(bits >= 1 && bits <= 32 && "simd_sign_extend_u32 requires a 1..32 bit width");
  const uint32_t sign = uint32_t{1} << (bits - 1);
  const uint32_t mask = bits == 32 ? ~uint32_t{0} : ((uint32_t{1} << bits) - uint32_t{1});
  return ((v & mask) ^ sign) - sign;
}

inline util::native<uint32_t> simd_mul_i24_u32(util::native<uint32_t> lhs,
                                               util::native<uint32_t> rhs) {
  return simd_sign_extend_u32(lhs, 24) * simd_sign_extend_u32(rhs, 24);
}

inline util::native<uint32_t> simd_mad_i24_u32(util::native<uint32_t> lhs,
                                               util::native<uint32_t> rhs,
                                               util::native<uint32_t> addend) {
  return simd_mul_i24_u32(lhs, rhs) + addend;
}

inline util::native<uint32_t> simd_lshl_u32(util::native<uint32_t> v, util::native<uint32_t> sh) {
  return util::map_native_scalar<uint32_t>(
      v, sh, [](uint32_t value, uint32_t count) { return value << (count & 31u); });
}

inline util::native<uint32_t> simd_lshr_u32(util::native<uint32_t> v, util::native<uint32_t> sh) {
  return util::map_native_scalar<uint32_t>(
      v, sh, [](uint32_t value, uint32_t count) { return value >> (count & 31u); });
}

inline util::native<uint64_t> simd_lshl_u64(util::native<uint64_t> v, util::native<uint64_t> sh) {
  return util::map_native64_scalar<uint64_t>(
      v, sh, [](uint64_t value, uint64_t count) { return value << (count & 63u); });
}

inline util::native<uint64_t> simd_lshr_u64(util::native<uint64_t> v, util::native<uint64_t> sh) {
  return util::map_native64_scalar<uint64_t>(
      v, sh, [](uint64_t value, uint64_t count) { return value >> (count & 63u); });
}

inline util::native<uint64_t> simd_ashr_i64(util::native<uint64_t> v, util::native<uint64_t> sh) {
  return util::map_native64_scalar<uint64_t>(v, sh, [](uint64_t value, uint64_t count) {
    return static_cast<uint64_t>(static_cast<int64_t>(value) >> (count & 63u));
  });
}

inline util::native<uint32_t> simd_bfe_u32(util::native<uint32_t> src,
                                           util::native<uint32_t> offset,
                                           util::native<uint32_t> width) {
  return util::map_native_scalar<uint32_t>(
      src, offset, width, [](uint32_t value, uint32_t offset_value, uint32_t width_value) {
        const uint32_t off = offset_value & 31u;
        const uint32_t w = width_value & 31u;
        if (w == 0)
          return uint32_t{0};
        const uint32_t mask = (uint32_t{1} << w) - 1u;
        return (value >> off) & mask;
      });
}

inline util::native<uint32_t> simd_bfe_i32(util::native<uint32_t> src,
                                           util::native<uint32_t> offset,
                                           util::native<uint32_t> width) {
  return util::map_native_scalar<uint32_t>(
      src, offset, width, [](uint32_t value, uint32_t offset_value, uint32_t width_value) {
        const uint32_t off = offset_value & 31u;
        const uint32_t w = width_value & 31u;
        if (w == 0)
          return uint32_t{0};
        const uint32_t mask = (uint32_t{1} << w) - 1u;
        const uint32_t extracted = static_cast<uint32_t>(static_cast<int32_t>(value) >> off) & mask;
        const uint32_t signbit = uint32_t{1} << (w - 1u);
        return (extracted ^ signbit) - signbit;
      });
}

inline util::native<uint32_t> simd_bfm_b32(util::native<uint32_t> width,
                                           util::native<uint32_t> offset) {
  return util::map_native_scalar<uint32_t>(width, offset,
                                           [](uint32_t width_value, uint32_t offset_value) {
                                             return bfm_b32(width_value, offset_value);
                                           });
}

inline util::native<int32_t> simd_cvt_i32_f32(util::native<float32_t> s) {
  return util::map_native_convert_scalar<int32_t>(s, [](float32_t value) {
    if (std::isnan(value))
      return int32_t{0};
    if (value >= 2147483648.0f)
      return std::numeric_limits<int32_t>::max();
    if (value < -2147483648.0f)
      return std::numeric_limits<int32_t>::min();
    return static_cast<int32_t>(value);
  });
}

inline util::native<uint32_t> simd_cvt_u32_f32(util::native<float32_t> s) {
  return util::map_native_convert_scalar<uint32_t>(s, [](float32_t value) {
    if (std::isnan(value) || value < 0.0f)
      return uint32_t{0};
    if (value >= 4294967296.0f)
      return std::numeric_limits<uint32_t>::max();
    return static_cast<uint32_t>(value);
  });
}

inline util::native<uint32_t> simd_cvt_i16_f32_to_u32(util::native<float32_t> s) {
  return util::map_native_convert_scalar<uint32_t>(s, [](float32_t value) {
    if (std::isnan(value))
      return uint32_t{0};
    if (value >= 32768.0f)
      return static_cast<uint32_t>(static_cast<uint16_t>(std::numeric_limits<int16_t>::max()));
    if (value < -32768.0f)
      return static_cast<uint32_t>(static_cast<uint16_t>(std::numeric_limits<int16_t>::min()));
    return static_cast<uint32_t>(static_cast<uint16_t>(static_cast<int16_t>(value)));
  });
}

inline util::native<uint32_t> simd_cvt_u16_f32_to_u32(util::native<float32_t> s) {
  return util::map_native_convert_scalar<uint32_t>(s, [](float32_t value) {
    if (std::isnan(value) || value < 0.0f)
      return uint32_t{0};
    if (value >= 65536.0f)
      return static_cast<uint32_t>(std::numeric_limits<uint16_t>::max());
    return static_cast<uint32_t>(static_cast<uint16_t>(value));
  });
}

inline util::native<uint32_t> select_vop3_true16_src(util::native<uint32_t> value, uint32_t opsel,
                                                     uint32_t src_idx) {
  if (opsel & (1u << src_idx))
    value >>= 16;
  return value & util::broadcast<uint32_t>(0xffffu);
}

/// In-vector VOP3 source modifier (f32), bit-exact with the scalar lambda the
/// generated bodies emit per source: `abs` first (`std::fabs`), then `neg`
/// (`-x`). `abs`/`neg` are the raw VOP3 modifier fields; the bit for source
/// index `SrcIdx` selects whether the modifier applies. std::fabs clears the
/// sign bit and unary minus flips it (both NaN-payload preserving), so the
/// vector form is a pure sign-bit AND/XOR — bit-identical on every input.
template <unsigned SrcIdx>
util::native<float> apply_vop3_src_mod_f32(util::native<float> v, uint32_t abs, uint32_t neg) {
  using U = util::native<uint32_t>;
  U b = std::bit_cast<U>(v);
  if (abs & (1u << SrcIdx))
    b = b & 0x7FFFFFFFu;
  if (neg & (1u << SrcIdx))
    b = b ^ 0x80000000u;
  return std::bit_cast<util::native<float>>(b);
}

/// In-vector VOP3 source modifier (f64), the f64 counterpart of
/// apply_vop3_src_mod_f32: abs first (std::fabs = sign-bit clear), then neg
/// (unary minus = sign-bit flip). Both are sign-bit-only on IEEE binary64, so
/// the vector form is a pure AND/XOR — bit-identical incl. NaN payload,
/// matching the scalar lambda the f64 VOP3 bodies emit.
template <unsigned SrcIdx>
util::native<double> apply_vop3_src_mod_f64(util::native<double> v, uint32_t abs, uint32_t neg) {
  using U = util::native<uint64_t>;
  U b = std::bit_cast<U>(v);
  if (abs & (1u << SrcIdx))
    b = b & 0x7FFFFFFFFFFFFFFFull;
  if (neg & (1u << SrcIdx))
    b = b ^ 0x8000000000000000ull;
  return std::bit_cast<util::native<double>>(b);
}

inline util::native<double> apply_vop3_dst_mod_f64(util::native<double> v, uint32_t omod,
                                                   uint32_t clamp, bool clamp_nan_to_zero) {
  return apply_vop3_dst_mod<double>(v, omod, clamp, clamp_nan_to_zero);
}

inline util::native<float> apply_vop3_dst_mod_f32(util::native<float> v, uint32_t omod,
                                                  uint32_t clamp, bool clamp_nan_to_zero) {
  return apply_vop3_dst_mod<float>(v, omod, clamp, clamp_nan_to_zero);
}

inline util::native<uint32_t> finalize_omod_f16_bits_simd(util::native<uint32_t> value,
                                                          uint32_t omod) {
  if (omod == 0)
    return value;
  using U = util::native<uint32_t>;
  const auto subnormal = ((value & U(0x7c00u)) == U(0)) && ((value & U(0x03ffu)) != U(0));
  util::stdx::where(subnormal, value) = value & U(0x8000u);
  util::stdx::where((value & U(0x7fffu)) == U(0), value) = U(0);
  return value;
}

/// @brief Execute a native-width batch of architectural F16 fused multiply-adds.
/// @details The raw F16 operands remain in 32-bit SIMD lanes. Each native-width
/// batch is split into double-width chunks and retains an exact-sum residual
/// without an intermediate F32 rounding. The final
/// F16 rounding and output policy reuse the scalar architectural primitive.
inline util::native<uint32_t>
fma_f16_mode_simd(util::native<uint32_t> src0, util::native<uint32_t> src1,
                  util::native<uint32_t> src2, bool abs0, bool abs1, bool abs2, bool neg0,
                  bool neg1, bool neg2, uint32_t round_mode, uint32_t denorm_mode, uint32_t omod,
                  bool clamp, bool fp16_ovfl, bool clamp_nan_to_zero, bool quiet_nan) {
  fp_mode::ScopedEnvironment nearest_environment(0);
  constexpr std::size_t W32 = util::native_width_v<uint32_t>;
  constexpr std::size_t W64 = util::native_width64;
  static_assert(W32 % W64 == 0);
  alignas(util::native<uint32_t>) uint32_t raw0[W32];
  alignas(util::native<uint32_t>) uint32_t raw1[W32];
  alignas(util::native<uint32_t>) uint32_t raw2[W32];
  alignas(util::native<uint32_t>) uint32_t out[W32];
  src0.copy_to(raw0, util::stdx::vector_aligned);
  src1.copy_to(raw1, util::stdx::vector_aligned);
  src2.copy_to(raw2, util::stdx::vector_aligned);

  auto prepare = [denorm_mode](uint32_t raw, bool absolute, bool negate) {
    return fp_mode::detail::flush_input_f16(
        fp_mode::detail::modify_f16(static_cast<uint16_t>(raw), absolute, negate), denorm_mode);
  };
  for (std::size_t base = 0; base < W32; base += W64) {
    alignas(util::native<double>) double a_lanes[W64];
    alignas(util::native<double>) double b_lanes[W64];
    alignas(util::native<double>) double c_lanes[W64];
    for (std::size_t i = 0; i < W64; ++i) {
      a_lanes[i] = static_cast<double>(util::f16_to_f32(prepare(raw0[base + i], abs0, neg0)));
      b_lanes[i] = static_cast<double>(util::f16_to_f32(prepare(raw1[base + i], abs1, neg1)));
      c_lanes[i] = static_cast<double>(util::f16_to_f32(prepare(raw2[base + i], abs2, neg2)));
    }
    const util::native<double> a(a_lanes, util::stdx::vector_aligned);
    const util::native<double> b(b_lanes, util::stdx::vector_aligned);
    const util::native<double> c(c_lanes, util::stdx::vector_aligned);
    const util::native<double> product = a * b;
    const util::native<double> result = product + c;
    const util::native<double> addend_virtual = result - product;
    const util::native<double> residual =
        (product - (result - addend_virtual)) + (c - addend_virtual);
    alignas(util::native<double>) double result_lanes[W64];
    result.copy_to(result_lanes, util::stdx::vector_aligned);
    for (std::size_t i = 0; i < W64; ++i) {
      double value = fp_mode::detail::round_to_odd({result_lanes[i], residual[i]});
      if (value == 0.0) {
        // F16 products cannot underflow in double, so zero means exact
        // cancellation or zero operands. Apply the guest sign policy.
        const bool negative_product = std::signbit(a_lanes[i]) != std::signbit(b_lanes[i]);
        const bool matching_zeros =
            c_lanes[i] == 0.0 && negative_product == std::signbit(c_lanes[i]);
        value = (matching_zeros ? negative_product : round_mode == 2) ? -0.0 : 0.0;
      }
      const bool exceptional_input =
          !std::isfinite(a_lanes[i]) || !std::isfinite(b_lanes[i]) || !std::isfinite(c_lanes[i]);
      uint16_t rounded =
          exceptional_input ? fp_mode::fma_f16(static_cast<uint16_t>(raw0[base + i]),
                                               static_cast<uint16_t>(raw1[base + i]),
                                               static_cast<uint16_t>(raw2[base + i]), abs0, abs1,
                                               abs2, neg0, neg1, neg2, round_mode, denorm_mode,
                                               omod, clamp, fp16_ovfl, clamp_nan_to_zero, quiet_nan)
                            : fp_mode::finish_fma_f16(value, round_mode, denorm_mode, omod, clamp,
                                                      fp16_ovfl, clamp_nan_to_zero);
      out[base + i] = rounded;
    }
  }
  return util::native<uint32_t>(out, util::stdx::vector_aligned);
}

/// F32 FMA runs in the caller's guest-rounding environment and applies MODE
/// denormal controls explicitly. Exceptional lanes use scalar NaN selection
/// and minimum-normal boundary handling.
inline util::native<float> fma_f32_simd(util::native<float> a, util::native<float> b,
                                        util::native<float> c, const Wavefront &wf,
                                        uint32_t omod = 0, bool force_flush = false) {
  using U = util::native<uint32_t>;
  const uint32_t denorm_mode = force_flush ? 0 : wf.fp_denorm_mode_f32();
  if (!(denorm_mode & 1u)) {
    a = util::flush_denorm_f32_simd(a);
    b = util::flush_denorm_f32_simd(b);
    c = util::flush_denorm_f32_simd(c);
  }
  const bool flush_output = omod || !(denorm_mode & 2u);
  auto exceptional = (std::bit_cast<U>(a) & U(0x7fffffffu)) >= U(0x7f800000u) ||
                     (std::bit_cast<U>(b) & U(0x7fffffffu)) >= U(0x7f800000u) ||
                     (std::bit_cast<U>(c) & U(0x7fffffffu)) >= U(0x7f800000u);
  auto result = util::stdx::fma(a, b, c);
  if (flush_output)
    exceptional |= (std::bit_cast<U>(result) & U(0x7fffffffu)) == U(0x00800000u);
  if (util::stdx::any_of(exceptional))
    for (std::size_t i = 0; i < U::size(); ++i)
      if (exceptional[i])
        result[i] = fp_mode::fma_f32(a[i], b[i], c[i], wf.cu().arch(), wf.ieee_mode(), denorm_mode,
                                     omod != 0);
  // FMA's subnormal intermediate is flushed before OMOD can scale it normal.
  return flush_output ? util::flush_denorm_f32_simd(result) : result;
}

/// ADD and MUL share FMA's NaN and pre-packing tininess policy. SUB retains
/// direct host subtraction's policy. The caller establishes guest rounding.
template <fp_mode::Arithmetic operation>
inline util::native<float> binary_f32_simd(util::native<float> a, util::native<float> b,
                                           const Wavefront &wf, uint32_t omod = 0) {
  static_assert(operation == fp_mode::Arithmetic::ADD || operation == fp_mode::Arithmetic::SUB ||
                operation == fp_mode::Arithmetic::MUL);
  if constexpr (operation == fp_mode::Arithmetic::SUB) {
    // Preserve direct subtraction's existing host NaN policy. Rewriting this as
    // architectural ADD with a negated input would select different NaN bits.
    using U = util::native<uint32_t>;
    const uint32_t denorm_mode = wf.fp_denorm_mode_f32();
    if (!(denorm_mode & 1u)) {
      a = util::flush_denorm_f32_simd(a);
      b = util::flush_denorm_f32_simd(b);
    }
    auto result = a - b;
    const auto nan_input = (std::bit_cast<U>(a) & U(0x7fffffffu)) > U(0x7f800000u) ||
                           (std::bit_cast<U>(b) & U(0x7fffffffu)) > U(0x7f800000u);
    if (util::stdx::any_of(nan_input))
      for (std::size_t i = 0; i < U::size(); ++i)
        if (nan_input[i])
          result[i] = fp_mode::detail::evaluate_arithmetic<fp_mode::Arithmetic::SUB, float>(
              a[i], b[i], 0.0f);
    return (denorm_mode & 2u) ? result : util::flush_denorm_f32_simd(result);
  } else if constexpr (operation == fp_mode::Arithmetic::ADD)
    return fma_f32_simd(a, util::native<float>(1.0f), b, wf, omod);
  else {
    using U = util::native<uint32_t>;
    auto zero = std::bit_cast<util::native<float>>((std::bit_cast<U>(a) ^ std::bit_cast<U>(b)) &
                                                   U(0x80000000u));
    return fma_f32_simd(a, b, zero, wf, omod);
  }
}

/// DX9 accumulator and three-source forms share the scalar flushing, zero-product,
/// NaN and underflow policies.
inline util::native<float> fma_dx9_zero_f32_simd(util::native<float> a, util::native<float> b,
                                                 util::native<float> c, const Wavefront &wf) {
  using U = util::native<uint32_t>;
  fp_mode::ScopedEnvironment environment(wf.fp_round_mode_f32());
  a = util::flush_denorm_f32_simd(a);
  b = util::flush_denorm_f32_simd(b);
  const auto zero_product = (std::bit_cast<U>(a) & U(0x7fffffffu)) == U(0) ||
                            (std::bit_cast<U>(b) & U(0x7fffffffu)) == U(0);
  // DX9 supplies a positive zero product even when the other operand is
  // infinite or NaN. The addition still applies rounding and addend NaN rules.
  U a_bits = std::bit_cast<U>(a), b_bits = std::bit_cast<U>(b);
  util::stdx::where(zero_product, a_bits) = U(0);
  util::stdx::where(zero_product, b_bits) = U(0x3f800000u);
  a = std::bit_cast<util::native<float>>(a_bits);
  b = std::bit_cast<util::native<float>>(b_bits);
  return fma_f32_simd(a, b, c, wf, 0, true);
}

/// @brief Apply architectural F64 OMOD and CLAMP to a native batch.
inline util::native<double> finish_f64_mode_simd(util::native<double> value, uint32_t round_mode,
                                                 uint32_t omod, bool clamp,
                                                 bool clamp_nan_to_zero) {
  using U = util::native<uint64_t>;
  constexpr std::size_t W = util::native_width64;
  alignas(U) uint64_t bits[W];
  std::bit_cast<U>(value).copy_to(bits, util::stdx::vector_aligned);
  for (uint64_t &lane : bits)
    lane = fp_mode::finish_f64(lane, round_mode, omod, clamp, clamp_nan_to_zero);
  return std::bit_cast<util::native<double>>(U(bits, util::stdx::vector_aligned));
}

template <typename = void>
  requires(util::has_stdx_simd)
inline PkF32Halves read_pkf32_halves(const RegisterAccess::OperandReadPair32View &op,
                                     uint32_t lane_base) {
  return {op.template load_lo_native<float>(lane_base),
          op.template load_hi_native<float>(lane_base)};
}

/// In-vector f32 sign flip (neg modifier): XOR the sign bit. Bit-exact to the
/// scalar `x = -x` for all values incl. ±0 / ±Inf / NaN (payload preserved).
inline util::native<float> pkf32_neg(util::native<float> v, bool do_neg) {
  if (!do_neg)
    return v;
  return std::bit_cast<util::native<float>>(std::bit_cast<util::native<uint32_t>>(v) ^
                                            util::native<uint32_t>(0x80000000u));
}

/// Re-type a `simd_mask` (e.g. the result of a float comparison) to the mask
/// type of `native<To>`, so it can drive `util::stdx::where` on a `native<To>`
/// value. Needed by the clamp/NaN cvt-to-int functors, which compute masks in
/// the float domain but blend into an int result. Wraps the libstdc++
/// `__proposed` mask cast in one place; `<experimental/simd>` is libstdc++-only
/// so the dependency is acceptable.
template <typename To, typename Mask>
  requires(util::has_stdx_simd)
inline auto simd_mask_as(const Mask &m) {
  return util::stdx::__proposed::static_simd_cast<util::native<To>>(m);
}

/// @brief Apply V_CVT_F32_F16 policy before widening a SIMD batch.
inline util::native<float> cvt_f32_f16_mode_simd(util::native<uint32_t> raw, const Wavefront &wf) {
  using U = util::native<uint32_t>;
  raw &= U(0xffffu);
  const U magnitude = raw & U(0x7fffu);
  if (!(wf.fp_denorm_mode_f16_f64() & 1u))
    util::stdx::where(magnitude < U(0x0400u), raw) = raw & U(0x8000u);
  if (fp_mode::quiets_nan(wf.cu().arch(), wf.ieee_mode()))
    util::stdx::where(magnitude > U(0x7c00u), raw) = raw | U(0x0200u);
  return util::f16_to_f32_simd(raw);
}

/// @brief Apply the F16 special cases before the shared modifier/writeback path.
inline util::native<float> div_fixup_f16_promoted_simd(util::native<float> quotient,
                                                       util::native<float> denominator,
                                                       util::native<float> numerator,
                                                       uint32_t rounding, uint32_t denorm) {
  return util::native<float>([&](auto index) {
    return div_fixup_f16(static_cast<float>(quotient[index]),
                         static_cast<float>(denominator[index]),
                         static_cast<float>(numerator[index]), rounding, denorm);
  });
}

/// @brief Batch operand access through the shared FIXUP and guest OMOD helpers.
template <typename T>
inline util::native<T> div_fixup_simd(util::native<T> p, util::native<T> d, util::native<T> n,
                                      uint32_t rounding, uint32_t denorm, uint32_t omod) {
  util::native<T> result;
  for (std::size_t i = 0; i < util::native<T>::size(); ++i)
    result[i] = div_apply_omod(div_fixup(static_cast<T>(p[i]), static_cast<T>(d[i]),
                                         static_cast<T>(n[i]), rounding, denorm),
                               rounding, omod);
  return result;
}

inline util::native<float> fma_mix_mul_add(util::native<float> a, util::native<float> b,
                                           util::native<float> c) {
  return a * b + c;
}

/// SIMD arithmetic in F64 avoids the F32 intermediate rounding of packed F16.
/// Scalar final rounding is shared with the architectural implementation. Fused
/// F16 and BF16 sums retain a residual when their terms exceed F64 precision.
template <PackedFloatOp Op, bool Bf16, typename U>
inline U packed_float_half_simd(U a_bits, U b_bits, U c_bits, Wavefront &wf, bool clamp) {
  using D = util::native<double>;
  constexpr bool Select = Op == PackedFloatOp::MIN || Op == PackedFloatOp::MAX ||
                          Op == PackedFloatOp::MINIMUM || Op == PackedFloatOp::MAXIMUM;
  constexpr bool Minimum = Op == PackedFloatOp::MIN || Op == PackedFloatOp::MINIMUM;
  constexpr bool Propagate = Op == PackedFloatOp::MINIMUM || Op == PackedFloatOp::MAXIMUM;
  constexpr std::size_t W = D::size();
  static_assert(U::size() == W);
  auto widen = [&wf](uint32_t bits) {
    (void)wf;
    if constexpr (Bf16)
      return static_cast<double>(util::bf16_to_f32(static_cast<uint16_t>(bits)));
    else
      return static_cast<double>(util::f16_to_f32(fp_mode::detail::flush_input_f16(
          static_cast<uint16_t>(bits), wf.fp_denorm_mode_f16_f64())));
  };
  D a([&](auto i) { return widen(a_bits[i]); });
  D b([&](auto i) { return widen(b_bits[i]); });
  D c([&](auto i) { return widen(c_bits[i]); });
  D result(0.0), residual(0.0);
  if constexpr (Select) {
    // Integer ordering covers finite halves, infinities and signed zero. NaNs
    // are repaired below using the exact instruction-specific selection policy.
    auto order = [](U bits) {
      U key = bits ^ U(0x8000u);
      util::stdx::where((bits & U(0x8000u)) != U(0), key) = (~bits) & U(0xffffu);
      return key;
    };
    U selected = a_bits;
    if constexpr (Minimum)
      util::stdx::where(order(b_bits) < order(a_bits), selected) = b_bits;
    else
      util::stdx::where(order(b_bits) > order(a_bits), selected) = b_bits;
    result = D([&](auto i) { return widen(selected[i]); });
  } else if constexpr (!Bf16 && Op != PackedFloatOp::FMA) {
    if constexpr (Op == PackedFloatOp::ADD)
      result = a + b;
    else if constexpr (Op == PackedFloatOp::MUL)
      result = a * b;
  } else {
    D product = Op == PackedFloatOp::ADD ? a : a * b;
    D addend = Op == PackedFloatOp::ADD ? b : c;
    if constexpr (Op == PackedFloatOp::MUL)
      addend = D([&](auto i) { return std::copysign(0.0, product[i]); });
    result = product + addend;
    D addend_virtual = result - product;
    residual = (product - (result - addend_virtual)) + (addend - addend_virtual);
  }
  U output(0);
  for (std::size_t i = 0; i < W; ++i) {
    double value = result[i];
    const bool nan = std::isnan(a[i]) || std::isnan(b[i]);
    if constexpr (Select) {
      if (nan) {
        if constexpr (Propagate)
          value = std::numeric_limits<double>::quiet_NaN();
        else
          value = Minimum ? std::fmin(a[i], b[i]) : std::fmax(a[i], b[i]);
      }
    }
    if constexpr (Bf16) {
      uint16_t rounded;
      if constexpr (Select) {
        // Preserve signaling-NaN behavior of the scalar float operation;
        // widening to double can quiet the input before selection.
        rounded = nan ? fp_mode::packed_select_bf16(util::bf16_to_f32(a_bits[i]),
                                                    util::bf16_to_f32(b_bits[i]), Minimum)
                      : util::f32_to_bf16_rne(static_cast<float>(value));
      } else {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
            (Op == PackedFloatOp::FMA && !std::isfinite(c[i])) || value == 0.0) {
          const float first = static_cast<float>(a[i]);
          const float second = Op == PackedFloatOp::ADD ? 1.0f : static_cast<float>(b[i]);
          const float third = Op == PackedFloatOp::ADD   ? static_cast<float>(b[i])
                              : Op == PackedFloatOp::MUL ? std::copysign(0.0f, first * second)
                                                         : static_cast<float>(c[i]);
          rounded = fp_mode::detail::fma_f32_to_bf16_nearest_environment(first, second, third, 0,
                                                                         false, false);
        } else {
          rounded = fp_mode::detail::round_exact_to_bf16({value, residual[i]}, 0);
        }
        if (wf.fp16_ovfl() && (rounded & 0x7fffu) == 0x7f80u && std::isfinite(a[i]) &&
            std::isfinite(b[i]) && (Op != PackedFloatOp::FMA || std::isfinite(c[i])))
          rounded = static_cast<uint16_t>((rounded & 0x8000u) | 0x7f7fu);
      }
      output[i] = fp_mode::clamp_bf16(rounded, clamp, floating_clamp_nan_to_zero(wf));
    } else {
      if constexpr (Op != PackedFloatOp::FMA) {
        if (nan) {
          constexpr auto operation = Op == PackedFloatOp::ADD   ? fp_mode::PackedBinaryOp::ADD
                                     : Op == PackedFloatOp::MUL ? fp_mode::PackedBinaryOp::MUL
                                     : Op == PackedFloatOp::MIN ? fp_mode::PackedBinaryOp::MIN
                                     : Op == PackedFloatOp::MAX ? fp_mode::PackedBinaryOp::MAX
                                     : Op == PackedFloatOp::MINIMUM
                                         ? fp_mode::PackedBinaryOp::MINIMUM
                                         : fp_mode::PackedBinaryOp::MAXIMUM;
          output[i] = fp_mode::packed_binary_f16(
              operation, a_bits[i], b_bits[i], wf.fp_round_mode_f16_f64(),
              wf.fp_denorm_mode_f16_f64(), clamp, wf.fp16_ovfl(), floating_clamp_nan_to_zero(wf),
              wf.cu().arch(), wf.ieee_mode());
          continue;
        }
      }
      if constexpr (Op == PackedFloatOp::ADD) {
        if (value == 0.0 && std::signbit(a[i]) != std::signbit(b[i]))
          value = wf.fp_round_mode_f16_f64() == 2 ? -0.0 : 0.0;
      }
      if constexpr (Op == PackedFloatOp::FMA) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || !std::isfinite(c[i])) {
          output[i] = fp_mode::fma_f16(
              a_bits[i], b_bits[i], c_bits[i], false, false, false, false, false, false,
              wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64(), 0, clamp, wf.fp16_ovfl(),
              floating_clamp_nan_to_zero(wf), fp_mode::quiets_nan(wf.cu().arch(), wf.ieee_mode()));
          continue;
        }
        value = fp_mode::detail::round_to_odd({value, residual[i]});
        if (value == 0.0) {
          const bool product_negative = ((a_bits[i] ^ b_bits[i]) & 0x8000u) != 0;
          const bool matching_zeros = c[i] == 0.0 && product_negative == std::signbit(c[i]);
          value =
              (matching_zeros ? product_negative : wf.fp_round_mode_f16_f64() == 2) ? -0.0 : 0.0;
        }
        output[i] =
            fp_mode::finish_fma_f16(value, wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64(),
                                    0, clamp, wf.fp16_ovfl(), floating_clamp_nan_to_zero(wf));
        continue;
      }
      uint16_t rounded =
          pseudo_scalar::round_f16_result(value, wf.fp_round_mode_f16_f64(), 0, clamp,
                                          wf.fp16_ovfl(), floating_clamp_nan_to_zero(wf));
      if (!(wf.fp_denorm_mode_f16_f64() & 2u) && (rounded & 0x7c00u) == 0)
        rounded &= 0x8000u;
      output[i] = rounded;
    }
  }
  return output;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PORTABLE_MATH_H_
