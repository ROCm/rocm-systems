// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_SIMD_H_
#define UTIL_SIMD_H_

#include "util/bit.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#if __has_include(<experimental/simd>)
#include <experimental/simd>
#endif

namespace util {

/// Compile-time switch for `<experimental/simd>` availability. Callers
/// gate SIMD fast paths via `if constexpr (util::has_stdx_simd)` so the
/// branch and all downstream calls compile out when the header is
/// absent (e.g. older libstdc++).
inline constexpr bool has_stdx_simd =
#if __has_include(<experimental/simd>)
    true;
#else
    false;
#endif

/// Process-wide switch that callers check before taking the SIMD fast path.
/// Read ONCE from the `RJ_FORCE_SCALAR` env var at startup (unset/empty/"0" =>
/// false, any other value => true). Production code never writes it; a
/// test-only seam (util/simd_test_hooks.h) may override it in-process so a
/// single test can drive both the scalar and SIMD execute paths and compare.
///
/// Defined in util/src/simd.cpp; the backing global is hidden in that TU. e2e
/// runs force the scalar codepath by setting `RJ_FORCE_SCALAR` before launch,
/// without recompiling.
bool force_scalar();

#if __has_include(<experimental/simd>)
namespace stdx = std::experimental;

template <class T> using native = stdx::native_simd<T>;

/// Native-SIMD width measured in 32-bit lanes. Convenience constant.
template <class T> constexpr std::size_t native_width_v = native<T>::size();
#else
// Fallback definitions so `if constexpr (has_stdx_simd)` discarded
// branches compile out even in non-template callers (e.g. gtest TEST
// bodies), where the discarded branch is still fully type-checked and
// odr-used. They are never executed: the constexpr condition is `false`
// and callers skip the SIMD path first.
template <class T> struct native {
  static constexpr std::size_t size() { return 1; }
  constexpr T operator[](std::size_t) const { return T{}; }
};
template <class T> constexpr std::size_t native_width_v = native<T>::size();
template <class T> native<T> load(const uint32_t *) { return {}; }
template <class T> native<T> broadcast(uint32_t) { return {}; }
template <class T> void masked_store(uint32_t *, native<T>, uint64_t) {}
template <class T> void blit_to_buffer(uint32_t (&)[native<T>::size()], native<T>) {}
template <class T> native<T> load64(const uint32_t *, const uint32_t *) { return {}; }
template <class T> native<T> broadcast64(uint64_t) { return {}; }
template <class T> void masked_store64(uint32_t *, uint32_t *, native<T>, uint64_t) {}
template <class T> struct narrow32 {
  static constexpr std::size_t size() { return 1; }
  constexpr T operator[](std::size_t) const { return T{}; }
};
template <class T> narrow32<T> load_narrow(const uint32_t *) { return {}; }
template <class T> narrow32<T> broadcast_narrow(uint32_t) { return {}; }
template <class T> void masked_store_narrow(uint32_t *, narrow32<T>, uint64_t) {}
#endif

/// Native-SIMD width for 64-bit lane types (e.g. native<uint64_t>/<double>),
/// measured in 64-bit lanes — half of native_width_v<uint32_t> on the same
/// vector width (8 vs 16 on AVX-512). Defined in both branches: in the
/// no-`<experimental/simd>` fallback native<uint64_t>::size() is 1.
inline constexpr std::size_t native_width64 = native<uint64_t>::size();

#if __has_include(<experimental/simd>)
/// Load `native<T>` from contiguous uint32_t storage. T must be a
/// 32-bit trivially-copyable type.
template <class T> native<T> load(const uint32_t *p) {
  using Bits = stdx::native_simd<uint32_t>;
  using Val = native<T>;
  static_assert(sizeof(T) == sizeof(uint32_t));
  static_assert(sizeof(Val) == sizeof(Bits));
  Bits bits(p, stdx::element_aligned);
  if constexpr (std::is_same_v<T, uint32_t>)
    return bits;
  else
    return std::bit_cast<Val>(bits);
}

/// Broadcast `broadcast_bits` (bit-cast to T) to every lane of
/// `native<T>`. T must be a 32-bit trivially-copyable type.
template <class T> native<T> broadcast(uint32_t broadcast_bits) {
  using Val = native<T>;
  static_assert(sizeof(T) == sizeof(uint32_t));
  if constexpr (std::is_same_v<T, uint32_t>)
    return Val(broadcast_bits);
  else
    return Val(std::bit_cast<T>(broadcast_bits));
}

/// Store `v` into contiguous uint32_t storage at `dst`, blending in only
/// the lanes whose bit is set in `mask`. If `mask` covers the full SIMD
/// width, falls through to a straight contiguous store.
template <class T> void masked_store(uint32_t *dst, native<T> v, uint64_t mask) {
  using Bits = stdx::native_simd<uint32_t>;
  using Val = native<T>;
  static_assert(sizeof(T) == sizeof(uint32_t));
  static_assert(sizeof(Val) == sizeof(Bits));
  constexpr std::size_t W = Bits::size();
  const uint64_t full = util::mask<uint64_t>(static_cast<int>(W));
  Bits bits = [&] {
    if constexpr (std::is_same_v<T, uint32_t>)
      return v;
    else
      return std::bit_cast<Bits>(v);
  }();
  if ((mask & full) == full) {
    bits.copy_to(dst, stdx::element_aligned);
    return;
  }
  alignas(Bits) uint32_t buf[W];
  bits.copy_to(buf, stdx::vector_aligned);
  for (std::size_t i = 0; i < W; ++i)
    if (mask & (1ULL << i))
      dst[i] = buf[i];
}

/// Same as masked_store, but writes to a caller-supplied uint32_t buffer
/// instead of contiguous lane storage. For operands whose dst is not a
/// contiguous VGPR (rocjitsu falls back to write_lane_chunk in that case).
template <class T> void blit_to_buffer(uint32_t (&buf)[native<T>::size()], native<T> v) {
  using Bits = stdx::native_simd<uint32_t>;
  Bits bits = [&] {
    if constexpr (std::is_same_v<T, uint32_t>)
      return v;
    else
      return std::bit_cast<Bits>(v);
  }();
  bits.copy_to(buf, stdx::vector_aligned);
}

/// Load `native<T>` (T a 64-bit trivially-copyable type) from the split lo/hi
/// 32-bit lane arrays of an f64/i64 operand. A per-lane f64 lives as two 32-bit
/// VGPRs at the same lane index — `lo` points at reg N's lane storage, `hi` at
/// reg N+1's — so lane k's value is `(uint64_t)hi[k] << 32 | lo[k]`, matching
/// `Operand::read_lane64`. `native_width64` lanes are combined; this is a scalar
/// gather (the two arrays are not a single contiguous 64-bit load) chosen for
/// bit-exactness over raw throughput.
template <class T> native<T> load64(const uint32_t *lo, const uint32_t *hi) {
  static_assert(sizeof(T) == sizeof(uint64_t));
  using U64 = stdx::native_simd<uint64_t>;
  static_assert(sizeof(native<T>) == sizeof(U64));
  constexpr std::size_t W = U64::size();
  alignas(U64) uint64_t buf[W];
  for (std::size_t i = 0; i < W; ++i)
    buf[i] = (static_cast<uint64_t>(hi[i]) << 32) | lo[i];
  U64 bits(buf, stdx::vector_aligned);
  if constexpr (std::is_same_v<T, uint64_t>)
    return bits;
  else
    return std::bit_cast<native<T>>(bits);
}

/// Broadcast `broadcast_bits` (bit-cast to T) to every lane of `native<T>` for a
/// 64-bit lane type. The companion of `broadcast` for the f64/i64 path; used when
/// a source operand resolves to a scalar/immediate rather than per-lane storage.
template <class T> native<T> broadcast64(uint64_t broadcast_bits) {
  static_assert(sizeof(T) == sizeof(uint64_t));
  using Val = native<T>;
  if constexpr (std::is_same_v<T, uint64_t>)
    return Val(broadcast_bits);
  else
    return Val(std::bit_cast<T>(broadcast_bits));
}

/// Store `v` (a 64-bit lane type) into the split lo/hi 32-bit lane arrays,
/// blending in only the lanes whose bit is set in `mask`. Each active lane's
/// 64-bit value is split back into lo (low 32) and hi (high 32), mirroring
/// `Operand::write_lane64`. No full-mask contiguous fast path: lo/hi are
/// separate registers, so the store is always a per-lane scatter.
template <class T> void masked_store64(uint32_t *lo, uint32_t *hi, native<T> v, uint64_t mask) {
  static_assert(sizeof(T) == sizeof(uint64_t));
  using U64 = stdx::native_simd<uint64_t>;
  static_assert(sizeof(native<T>) == sizeof(U64));
  constexpr std::size_t W = U64::size();
  U64 bits = [&] {
    if constexpr (std::is_same_v<T, uint64_t>)
      return v;
    else
      return std::bit_cast<U64>(v);
  }();
  alignas(U64) uint64_t buf[W];
  bits.copy_to(buf, stdx::vector_aligned);
  for (std::size_t i = 0; i < W; ++i)
    if (mask & (1ULL << i)) {
      lo[i] = static_cast<uint32_t>(buf[i]);
      hi[i] = static_cast<uint32_t>(buf[i] >> 32);
    }
}

/// A `native_width64`-wide SIMD of a 32-bit lane type (e.g. narrow32<float>,
/// narrow32<int32_t>). Used by the mixed-width f64<->32-bit conversion glue: a
/// chunk of `native_width64` (8 on AVX-512) f64 lanes pairs with the same number
/// of 32-bit lanes, so the 32-bit side is a `fixed_size_simd` of that width — a
/// direct `static_simd_cast` bridges it to/from `native<double>` (also
/// `native_width64`-wide) with no bit_cast.
template <class T> using narrow32 = stdx::fixed_size_simd<T, native_width64>;

/// Load `narrow32<T>` (native_width64 lanes of a 32-bit type) from contiguous
/// uint32_t storage. The 32-bit-source counterpart of `load` for the cvt glue.
/// `fixed_size_simd` is not trivially copyable (so `std::bit_cast` of the whole
/// vector is ill-formed, unlike `native_simd`); the bit reinterpretation is done
/// per lane through the trivially-copyable scalar `T`.
template <class T> narrow32<T> load_narrow(const uint32_t *p) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  if constexpr (std::is_same_v<T, uint32_t>) {
    return narrow32<uint32_t>(p, stdx::element_aligned);
  } else {
    alignas(narrow32<T>) T buf[native_width64];
    for (std::size_t i = 0; i < native_width64; ++i)
      buf[i] = std::bit_cast<T>(p[i]);
    return narrow32<T>(buf, stdx::vector_aligned);
  }
}

/// Broadcast `broadcast_bits` (bit-cast to T) to every lane of `narrow32<T>`.
/// Companion of `broadcast` for the narrow (8-wide) cvt path; used when a source
/// operand resolves to a scalar/immediate rather than per-lane storage.
template <class T> narrow32<T> broadcast_narrow(uint32_t broadcast_bits) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  if constexpr (std::is_same_v<T, uint32_t>)
    return narrow32<T>(broadcast_bits);
  else
    return narrow32<T>(std::bit_cast<T>(broadcast_bits));
}

/// Store `v` (native_width64 lanes of a 32-bit type) into contiguous uint32_t
/// storage at `dst`, blending in only the lanes whose bit is set in `mask`. The
/// 32-bit-dst counterpart of `masked_store` for the f64-src cvt glue, which
/// writes only `native_width64` 32-bit lanes per chunk.
template <class T> void masked_store_narrow(uint32_t *dst, narrow32<T> v, uint64_t mask) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  constexpr std::size_t W = native_width64;
  const uint64_t full = util::mask<uint64_t>(static_cast<int>(W));
  alignas(narrow32<T>) T buf[W];
  v.copy_to(buf, stdx::vector_aligned);
  if ((mask & full) == full) {
    for (std::size_t i = 0; i < W; ++i)
      dst[i] = std::bit_cast<uint32_t>(buf[i]);
    return;
  }
  for (std::size_t i = 0; i < W; ++i)
    if (mask & (1ULL << i))
      dst[i] = std::bit_cast<uint32_t>(buf[i]);
}

/// Vectorized, bit-exact port of `f16_to_f32` (util/data_types.h). Each lane's
/// low 16 bits hold the f16; high bits are ignored. All branch selection is
/// done with `where`-masks in the uint32 domain. Bit-identical to the scalar
/// version for every input, including NaN payloads.
inline native<float> f16_to_f32_simd(native<uint32_t> v) {
  using U = native<uint32_t>;
  const U h = v & 0xFFFFu;
  const U sign = (h >> 15) & 1u;
  const U exp = (h >> 10) & 0x1Fu;
  const U mant = h & 0x3FFu;
  const U sign31 = sign << 31;

  // Normal (exp 1..30): rebias exponent 15 -> 127 (delta 112). Default path.
  U bits = sign31 | ((exp + 112u) << 23) | (mant << 13);

  // Inf / NaN (exp == 31): keep mantissa, set max exponent.
  stdx::where(exp == 31u, bits) = sign31 | 0x7F800000u | (mant << 13);

  // Zero (exp == 0 && mant == 0).
  stdx::where(exp == 0u && mant == 0u, bits) = sign31;

  // Denormal (exp == 0 && mant != 0): renormalize. p = index of the highest
  // set bit of mant; mant is in 1..1023 (< 2^24) so the int->float cast is
  // exact and p = floor(log2(mant)) reads straight out of the f32 exponent.
  const native<float> mf = stdx::static_simd_cast<native<float>>(mant);
  const U p = (std::bit_cast<U>(mf) >> 23) - 127u;
  // Scalar k = 10 - p shifts; exp_final = 1 - k = p - 9; exp field = p + 103.
  const U dn = sign31 | ((p + 103u) << 23) | (((mant << (10u - p)) & 0x3FFu) << 13);
  stdx::where(exp == 0u && mant != 0u, bits) = dn;

  return std::bit_cast<native<float>>(bits);
}

/// Vectorized, bit-exact port of `f32_to_f16` (util/data_types.h). Returns the
/// f16 bits in the low 16 bits of each lane (high bits zero). All conditions
/// are expressed as unsigned comparisons on the biased f32 exponent `fe`, so
/// every mask stays in the uint32 domain. The exponent rebias e = fe - 112
/// maps the scalar branches to: nan/inf fe==255, overflow fe>=143, normal
/// fe 113..142, denormal fe 102..112, flush fe<102.
inline native<uint32_t> f32_to_f16_simd(native<float> val) {
  using U = native<uint32_t>;
  const U f = std::bit_cast<U>(val);
  const U sign = (f >> 16) & 0x8000u;
  const U fe = (f >> 23) & 0xFFu;
  const U fm = f & 0x7FFFFFu;

  // Normal path (default; overwritten for the other ranges below).
  U m = fm >> 13;
  const U rb = (fm >> 12) & 1u;
  U sticky(0u);
  stdx::where((fm & 0xFFFu) != 0u, sticky) = 1u;
  m = m + (rb & (sticky | (m & 1u)));
  U exp_n = fe - 112u;
  const auto carry = (m > 0x3FFu); // mantissa rounded up into the exponent
  stdx::where(carry, m) = 0u;
  stdx::where(carry, exp_n) = exp_n + 1u;
  U out = sign | (exp_n << 10) | m;
  stdx::where(carry && exp_n >= 31u, out) = sign | 0x7C00u; // carry overflow -> Inf

  // Denormal (fe 102..112): m' = fm|0x800000, shift right by sh=126-fe (14..24)
  // with round-to-nearest-even.
  const U mm = fm | 0x800000u;
  // sh = 126 - fe is in [14,24] for the real denormal range (fe 102..112), but
  // lanes with fe < 102 (flushed to zero below) yield sh up to 126. Clamp to 31
  // so the shifts here never hit undefined/masked behaviour; those lanes are
  // overwritten by the `fe < 102` flush, so the clamped value is discarded.
  U sh = 126u - fe;
  stdx::where(sh > 31u, sh) = 31u;
  const U drb = (mm >> (sh - 1u)) & 1u;
  const U mask_lo = (1u << (sh - 1u)) - 1u;
  U dsticky(0u);
  stdx::where((mm & mask_lo) != 0u, dsticky) = 1u;
  U r = mm >> sh;
  r = r + (drb & (dsticky | (r & 1u)));
  stdx::where(fe <= 112u, out) = sign | r;

  // Flush to zero (fe < 102) and saturate to Inf (fe 143..254).
  stdx::where(fe < 102u, out) = sign;
  stdx::where(fe >= 143u && fe <= 254u, out) = sign | 0x7C00u;

  // Inf / NaN (fe == 255): NaN keeps payload MSBs and forces the low bit.
  stdx::where(fe == 255u, out) = sign | 0x7C00u;
  stdx::where(fe == 255u && fm != 0u, out) = sign | 0x7C00u | (fm >> 13) | 1u;

  return out;
}

/// Bit-exact SIMD rounding helpers (trunc / ceil / floor / round-to-nearest-even).
///
/// libstdc++'s `std::experimental::simd` rounding intrinsics are NOT bit-exact
/// against the scalar `std::trunc/ceil/floor/nearbyint` at every vector width:
/// at SSE width (native_simd<float> of size 4) they (a) drop the sign of a
/// zero-magnitude result (e.g. trunc(-0.3) yields +0.0 instead of -0.0) and
/// (b) mangle the payload/quiet bit of a NaN input instead of returning it
/// unchanged. The scalar generated bodies these SIMD fast paths must match use
/// the libc functions, which preserve sign-of-zero and pass NaNs through
/// verbatim. So we run the stdx intrinsic for the finite, nonzero-result lanes
/// and then repair the two edge cases by blend:
///   - NaN input lane: return the input bits unchanged.
///   - zero-magnitude result lane: copy the input's sign bit onto the +0 the
///     intrinsic produced (IEEE-754 round-to-integer keeps the operand sign on
///     a zero result, so this is exactly the sign of the *input*).
/// Bit-identical to the scalar reference at every native width.
template <class Float, class Round> native<Float> round_fixup_simd(native<Float> a, Round round) {
  using F = native<Float>;
  using U = std::conditional_t<sizeof(Float) == 4, native<uint32_t>, native<uint64_t>>;
  using Bits = typename U::value_type;
  static_assert(sizeof(Bits) == sizeof(Float));
  constexpr Bits kSign = Bits(1) << (sizeof(Bits) * 8 - 1);
  // Quiet bit = MSB of the mantissa (bit 22 for f32, bit 51 for f64).
  constexpr Bits kQuiet = Bits(1) << (sizeof(Float) == 4 ? 22 : 51);

  const U ai = std::bit_cast<U>(a);
  F r = round(a);
  // All blends use float-domain masks (simd_mask<Float>) to match the existing
  // f64 transcendental helpers' compile-tested pattern (== / isnan), keeping the
  // 64-bit-mask path off the libstdc++ AVX-512 `_S_to_bits<long long>` hazard.
  //
  // Zero-magnitude result inherits the input sign: round-to-integer of x in
  // (-1, 0] is -0.0 (scalar libc result), but the stdx intrinsic yields +0.0 at
  // narrow widths. `r == 0` matches both +0 and -0 lanes.
  const F signed_zero = std::bit_cast<F>(ai & U(kSign));
  stdx::where(r == F(Float(0)), r) = signed_zero;
  // NaN input -> canonical quiet NaN (sign + payload preserved, quiet bit set),
  // exactly what scalar `std::trunc/ceil/floor/nearbyint` produce: qNaN passes
  // through unchanged and sNaN is quieted. The stdx intrinsic instead clears the
  // quiet bit at narrow widths, so blend the bit-correct value in explicitly.
  const F quieted = std::bit_cast<F>(ai | U(kQuiet));
  stdx::where(stdx::isnan(a), r) = quieted;
  return r;
}

inline native<float> trunc_simd(native<float> a) {
  return round_fixup_simd<float>(a, [](native<float> x) { return stdx::trunc(x); });
}
inline native<float> ceil_simd(native<float> a) {
  return round_fixup_simd<float>(a, [](native<float> x) { return stdx::ceil(x); });
}
inline native<float> floor_simd(native<float> a) {
  return round_fixup_simd<float>(a, [](native<float> x) { return stdx::floor(x); });
}
inline native<float> rndne_simd(native<float> a) {
  return round_fixup_simd<float>(a, [](native<float> x) { return stdx::nearbyint(x); });
}

inline native<double> trunc_simd(native<double> a) {
  return round_fixup_simd<double>(a, [](native<double> x) { return stdx::trunc(x); });
}
inline native<double> ceil_simd(native<double> a) {
  return round_fixup_simd<double>(a, [](native<double> x) { return stdx::ceil(x); });
}
inline native<double> floor_simd(native<double> a) {
  return round_fixup_simd<double>(a, [](native<double> x) { return stdx::floor(x); });
}
inline native<double> rndne_simd(native<double> a) {
  return round_fixup_simd<double>(a, [](native<double> x) { return stdx::nearbyint(x); });
}

/// Scalar round-to-integer matching the round_fixup_simd / AMD-hardware NaN
/// behavior on every compiler. clang lowers std::floor/ceil/trunc to the
/// roundss/roundsd instruction, which quiets a signaling NaN (sign + payload
/// preserved, quiet bit set) — exactly what the SIMD fast paths and the GPU
/// produce. gcc's glibc floorf/floor/ceil/trunc instead pass an sNaN through
/// verbatim, so the generated scalar body would disagree with its own SIMD
/// fast path (and with hardware) under gcc. Quiet explicitly there; on clang
/// this is an idempotent no-op left to the compiler. (std::nearbyint already
/// quiets under glibc, so rndne needs no fixup.)
#if defined(__GNUC__) && !defined(__clang__)
template <class F> inline F quiet_snan_scalar(F a, F r) {
  using U = std::conditional_t<sizeof(F) == 4, uint32_t, uint64_t>;
  constexpr U kQuiet = U(1) << (sizeof(F) == 4 ? 22 : 51);
  return std::isnan(a) ? std::bit_cast<F>(std::bit_cast<U>(a) | kQuiet) : r;
}
#else
template <class F> inline F quiet_snan_scalar(F /*a*/, F r) { return r; }
#endif
inline float floor_scalar(float a) { return quiet_snan_scalar(a, std::floor(a)); }
inline double floor_scalar(double a) { return quiet_snan_scalar(a, std::floor(a)); }
inline float ceil_scalar(float a) { return quiet_snan_scalar(a, std::ceil(a)); }
inline double ceil_scalar(double a) { return quiet_snan_scalar(a, std::ceil(a)); }
inline float trunc_scalar(float a) { return quiet_snan_scalar(a, std::trunc(a)); }
inline double trunc_scalar(double a) { return quiet_snan_scalar(a, std::trunc(a)); }

/// Flush f32 denormals to sign-preserving zero (FTZ). Branchless vector port of
/// `amdgpu::transcendental::flush_denorm_f32`: a lane with biased exponent 0 and
/// nonzero mantissa becomes ±0 (sign preserved); every other lane (normal, Inf,
/// NaN, ±0) passes through unchanged. AMD transcendental micro-ops always run in
/// FTZ mode, so the f32 rcp/rsq/exp/log SIMD ports below funnel through this.
inline native<float> flush_denorm_f32_simd(native<float> v) {
  using U = native<uint32_t>;
  U b = std::bit_cast<U>(v);
  stdx::where(((b & 0x7F800000u) == 0u) && ((b & 0x007FFFFFu) != 0u), b) = b & 0x80000000u;
  return std::bit_cast<native<float>>(b);
}

/// Vector ports of `amdgpu::transcendental::*_f32`, mirroring the scalar
/// reference body bit-for-bit so the VOP1 SIMD fast path agrees with the
/// forced-scalar path on every lane. The ±0/Inf special cases fall out of IEEE
/// div/sqrt after the FTZ input flush; only NaN-input preservation (scalar
/// returns the input NaN unchanged) and the negative-domain canonical qNaN
/// (0x7FC00000) need explicit blends. The 16-bit (f16) ops reuse these on the
/// f16->f32 intermediate, matching the scalar `f32_to_f16(rcp_f32(f16_to_f32))`.
// Canonical positive quiet-NaN (f32), broadcast across the vector. Shared by
// the transcendental fast paths below, which blend it into out-of-domain
// lanes (negative sqrt/rsqrt, log of a negative) to match the scalar refs.
inline const native<float> kQNaN = std::bit_cast<native<float>>(native<uint32_t>(0x7FC00000u));

inline native<float> rcp_f32_simd(native<float> a) {
  native<float> x = flush_denorm_f32_simd(a);
  native<float> r = flush_denorm_f32_simd(native<float>(1.0f) / x);
  stdx::where(stdx::isnan(a), r) = a;
  return r;
}

inline native<float> rsq_f32_simd(native<float> a) {
  native<float> x = flush_denorm_f32_simd(a);
  native<float> r = flush_denorm_f32_simd(native<float>(1.0f) / stdx::sqrt(x));
  stdx::where(x < native<float>(0.0f), r) = kQNaN; // negatives incl -Inf -> qNaN
  stdx::where(stdx::isnan(a), r) = a;
  return r;
}

inline native<float> sqrt_f32_simd(native<float> a) {
  native<float> r = stdx::sqrt(a); // no FTZ flush: scalar sqrt_f32 keeps denormals
  stdx::where(a < native<float>(0.0f), r) = kQNaN;
  stdx::where(stdx::isnan(a), r) = a;
  return r;
}

inline native<float> log_f32_simd(native<float> a) {
  native<float> x = flush_denorm_f32_simd(a);
  native<float> r = stdx::log2(x); // input-flush only; scalar log_f32 has no out-flush
  stdx::where(x < native<float>(0.0f), r) = kQNaN;
  stdx::where(stdx::isnan(a), r) = a;
  return r;
}

inline native<float> exp_f32_simd(native<float> a) {
  native<float> x = flush_denorm_f32_simd(a);
  native<float> r = flush_denorm_f32_simd(stdx::exp2(x));
  stdx::where(stdx::isnan(a), r) = a;
  return r;
}

/// Branchless SWAR population count over a uint32 vector. Each lane holds
/// std::popcount(lane) in [0, 32], bit-identical to the scalar std::popcount.
/// Used by the bit-scan VOP1/VOP3 SIMD fast paths (bcnt / ffbh / ffbl / cls).
inline native<uint32_t> popcount_u32_simd(native<uint32_t> x) {
  using U = native<uint32_t>;
  x = x - ((x >> 1) & U(0x55555555u));
  x = (x & U(0x33333333u)) + ((x >> 2) & U(0x33333333u));
  x = (x + (x >> 4)) & U(0x0F0F0F0Fu);
  x = x + (x >> 8);
  x = x + (x >> 16);
  return x & U(0x3Fu);
}

/// Count leading zeros over a uint32 vector. RAW count: a zero input yields 32
/// (callers blend the AMD `0xFFFFFFFF`-on-zero sentinel themselves). Computed
/// as 32 - popcount(MSB-smear), so a lane with its MSB at bit b yields 31 - b.
inline native<uint32_t> clz_u32_simd(native<uint32_t> x) {
  using U = native<uint32_t>;
  U s = x;
  s = s | (s >> 1);
  s = s | (s >> 2);
  s = s | (s >> 4);
  s = s | (s >> 8);
  s = s | (s >> 16);
  return U(32u) - popcount_u32_simd(s);
}

/// Count trailing zeros over a uint32 vector. RAW count: a zero input yields 32
/// (callers blend the AMD `0xFFFFFFFF`-on-zero sentinel themselves). Computed as
/// popcount((x & -x) - 1): isolate the lowest set bit, then count the bits below.
inline native<uint32_t> ctz_u32_simd(native<uint32_t> x) {
  using U = native<uint32_t>;
  U lowbit = x & (~x + U(1u));
  return popcount_u32_simd(lowbit - U(1u));
}

/// Vector port of `util::fp8_e4m3_to_f32` over the low byte of each lane (E4M3:
/// 1 sign / 4 exp / 3 mantissa, bias 7, no Inf). Bit-identical to the scalar:
/// the denormal path `mant * 2^-9` is exact for mant in [0,7] and folds the ±0
/// case (mant==0 -> +0 | signbit). E4M3FN has no Inf and a single NaN encoding
/// (S.1111.111): exp==15 stays normal for mant 0..6 (max finite 448) and yields
/// a canonical qNaN only for mant==7. Used by the v_cvt_f32_fp8 SIMD fast path.
inline native<float> fp8_e4m3_to_f32_simd(native<uint32_t> v) {
  using U = native<uint32_t>;
  U b = v & U(0xFFu);
  U sign = (b >> 7) & U(1u);
  U exp = (b >> 3) & U(0xFu);
  U mant = b & U(0x7u);
  U signbit = sign << 31;
  U normal = signbit | ((exp + U(120u)) << 23) | (mant << 20); // exp + 127 - 7
  native<float> dn =
      stdx::static_simd_cast<native<float>>(mant) * native<float>(0.001953125f); // 2^-9
  U dnb = std::bit_cast<U>(dn) | signbit; // exp0: ±denormal, or ±0 when mant==0
  U out = normal;                         // exp15/mant<7 stay normal (E4M3FN, no Inf)
  stdx::where(exp == 0u, out) = dnb;
  stdx::where((exp == 15u) & (mant == 7u), out) =
      signbit | U(0x7FC00000u); // E4M3FN single NaN (S.1111.111)
  return std::bit_cast<native<float>>(out);
}

/// Vector port of `util::bf8_e5m2_to_f32` over the low byte of each lane (E5M2:
/// 1 sign / 5 exp / 2 mantissa, bias 15, has Inf). Bit-identical to the scalar:
/// the denormal path `mant * 2^-16` is exact for mant in [0,3] and folds the ±0
/// case; exp==31 yields ±Inf (mant==0) or a canonical qNaN (mant!=0). Used by
/// the v_cvt_f32_bf8 SIMD fast path.
inline native<float> bf8_e5m2_to_f32_simd(native<uint32_t> v) {
  using U = native<uint32_t>;
  U b = v & U(0xFFu);
  U sign = (b >> 7) & U(1u);
  U exp = (b >> 2) & U(0x1Fu);
  U mant = b & U(0x3u);
  U signbit = sign << 31;
  U normal = signbit | ((exp + U(112u)) << 23) | (mant << 21); // exp + 127 - 15
  U inf_nan = signbit | U(0x7F800000u);                        // exp31, mant0 -> ±Inf
  stdx::where(mant != 0u, inf_nan) = signbit | U(0x7FC00000u); // exp31, mant!=0 -> qNaN
  native<float> dn =
      stdx::static_simd_cast<native<float>>(mant) * native<float>(1.52587890625e-05f); // 2^-16
  U dnb = std::bit_cast<U>(dn) | signbit; // exp0: ±denormal, or ±0 when mant==0
  U out = normal;
  stdx::where(exp == 0u, out) = dnb;
  stdx::where(exp == 31u, out) = inf_nan;
  return std::bit_cast<native<float>>(out);
}

/// Vector ports of the IEEE-2019 maximum / minimum operations (the non-"num"
/// forms used by v_maximum_*/v_minimum_* on gfx1250/rdna4). Unlike fmax/fmin
/// these PROPAGATE NaN (any NaN input -> canonical qNaN) and order signed zeros
/// (-0 < +0): maximum of a ±0 tie is +0, minimum is -0. Bit-identical to the
/// scalar bodies:
///   if (isnan(a)||isnan(b)) return qNaN;
///   if (a==b)              return signbit(a) ? <tie> : <other>;
///   return a <cmp> b ? a : b;
template <typename V> V ieee_maximum_simd(V a, V b) {
  const auto nan = stdx::isnan(a) || stdx::isnan(b);
  const auto eq = (a == b);
  const auto sa = stdx::signbit(a); // true when a is negative (incl. -0)
  V res = b;                        // a < b (and the a==b,!sa case start)
  stdx::where(a > b, res) = a;
  stdx::where(eq && !sa, res) = a; // ±0 / equal tie: pick the +signed operand
  stdx::where(nan, res) = V(std::numeric_limits<typename V::value_type>::quiet_NaN());
  return res;
}
template <typename V> V ieee_minimum_simd(V a, V b) {
  const auto nan = stdx::isnan(a) || stdx::isnan(b);
  const auto eq = (a == b);
  const auto sa = stdx::signbit(a);
  V res = b; // a > b (and the a==b,!sa case)
  stdx::where(a < b, res) = a;
  stdx::where(eq && sa, res) = a; // ±0 / equal tie: pick the -signed operand
  stdx::where(nan, res) = V(std::numeric_limits<typename V::value_type>::quiet_NaN());
  return res;
}

/// Cubemap face ops (back v_cube{id,sc,tc}_f32). The scalar bodies select among
/// the three axes by an `else if` cascade on |x|,|y|,|z| with `>=` ties: the X
/// face wins ties, then Y, then Z. The vector forms apply the blends in reverse
/// priority (Z default, Y overwrite, X overwrite last) so X wins ties exactly as
/// scalar. Every branch is a bit-identical value copy / sign flip of the
/// (already abs/neg-applied) inputs — no FP arithmetic — so byte-identical. NaN
/// inputs make every `>=` mask false -> the Z-axis default, matching scalar.
/// (cubema = 2 * fmax(|x|,fmax(|y|,|z|)) is emitted inline at the call site.)
inline native<float> cube_id_f32_simd(native<float> x, native<float> y, native<float> z) {
  using F = native<float>;
  const F ax = stdx::abs(x), ay = stdx::abs(y), az = stdx::abs(z);
  const auto x_face = (ax >= ay) && (ax >= az);
  const auto y_face = (ay >= ax) && (ay >= az);
  F r = F(5.0f);
  stdx::where(z >= F(0.0f), r) = F(4.0f); // Z face: z>=0 ? 4 : 5
  F yv = F(3.0f);
  stdx::where(y >= F(0.0f), yv) = F(2.0f);
  stdx::where(y_face, r) = yv;
  F xv = F(1.0f);
  stdx::where(x >= F(0.0f), xv) = F(0.0f);
  stdx::where(x_face, r) = xv;
  return r;
}
inline native<float> cube_sc_f32_simd(native<float> x, native<float> y, native<float> z) {
  using F = native<float>;
  const F ax = stdx::abs(x), ay = stdx::abs(y), az = stdx::abs(z);
  const auto x_face = (ax >= ay) && (ax >= az);
  const auto y_face = (ay >= ax) && (ay >= az);
  F r = x;
  stdx::where(z >= F(0.0f), r) = -x; // Z face: z>=0 ? -x : x
  stdx::where(y_face, r) = x;
  F xv = -z;
  stdx::where(x >= F(0.0f), xv) = z; // X face: x>=0 ? z : -z
  stdx::where(x_face, r) = xv;
  return r;
}
inline native<float> cube_tc_f32_simd(native<float> x, native<float> y, native<float> z) {
  using F = native<float>;
  const F ax = stdx::abs(x), ay = stdx::abs(y), az = stdx::abs(z);
  const auto x_face = (ax >= ay) && (ax >= az);
  const auto y_face = (ay >= ax) && (ay >= az);
  F r = -y; // Z face and X face both return -y
  F yv = z;
  stdx::where(y >= F(0.0f), yv) = -z; // Y face: y>=0 ? -z : z
  stdx::where(y_face, r) = yv;
  stdx::where(x_face, r) = -y; // restore -y on ties where the Y mask also fired
  return r;
}

/// Normalized f32->int16 / ->uint16 pack-convert lanes (back v_cvt_pk[_]norm_*).
/// Scalar: `isnan(f) ? 0 : static_cast<intN>(clamp(f * K, lo, hi))`. The NaN->0
/// blend is done in the FLOAT domain (so the mask type matches) before the int
/// truncation, avoiding any float-mask -> int-mask conversion. Clamp keeps the
/// value in range so static_simd_cast (truncate-toward-zero) matches the scalar
/// cast; the caller masks &0xFFFF when packing. i16: K=32767, clamp
/// [-32768,32767]; u16: K=65535, clamp [0,65535].
inline native<int32_t> cvt_pknorm_i16_f32_simd(native<float> f) {
  native<float> p = f * native<float>(32767.0f);
  stdx::where(p < native<float>(-32768.0f), p) = native<float>(-32768.0f);
  stdx::where(p > native<float>(32767.0f), p) = native<float>(32767.0f);
  stdx::where(stdx::isnan(f), p) = native<float>(0.0f);
  return stdx::static_simd_cast<native<int32_t>>(p);
}
inline native<uint32_t> cvt_pknorm_u16_f32_simd(native<float> f) {
  native<float> p = f * native<float>(65535.0f);
  stdx::where(p < native<float>(0.0f), p) = native<float>(0.0f);
  stdx::where(p > native<float>(65535.0f), p) = native<float>(65535.0f);
  stdx::where(stdx::isnan(f), p) = native<float>(0.0f);
  return stdx::static_simd_cast<native<uint32_t>>(p);
}

/// Vector port of the f32 `std::frexp` mantissa over raw float bits. Returns the
/// significand m with |m| in [0.5, 1) such that input = m * 2^e (e via
/// frexp_exp_f32_simd). Normal lanes force the exponent field to 126 and keep
/// the mantissa; ±0 / Inf / NaN pass through unchanged (matching glibc frexp,
/// which the scalar body calls unguarded); denormals are renormalized via the
/// highest-set-bit index p = floor(log2(M)) (M < 2^24, so the int->float cast is
/// exact and p reads out of the f32 exponent field). Bit-identical to the scalar
/// v_frexp_mant_f32 body.
inline native<float> frexp_mant_f32_simd(native<uint32_t> v) {
  using U = native<uint32_t>;
  U sign = v & U(0x80000000u);
  U E = (v >> 23) & U(0xFFu);
  U M = v & U(0x7FFFFFu);
  U normal = sign | (U(126u) << 23) | M; // exponent forced to 2^-1
  // Denormal: p = index of highest set bit of M; mantissa <<= (23 - p), drop the
  // implicit leading 1, exponent field stays 126.
  const native<float> mf = stdx::static_simd_cast<native<float>>(M);
  const U p = (std::bit_cast<U>(mf) >> 23) - U(127u);
  U dn = sign | (U(126u) << 23) | ((M << (U(23u) - p)) & U(0x7FFFFFu));
  U out = normal;
  stdx::where(E == 0u, out) = v;                 // ±0 (M==0); overwritten if denormal
  stdx::where((E == 0u) && (M != 0u), out) = dn; // denormal -> renormalized
  stdx::where(E == 255u, out) = v;               // Inf passes through unchanged
  // frexp quiets signaling NaNs (sets the mantissa MSB) while preserving the
  // payload; qNaN inputs are unchanged by the idempotent OR.
  stdx::where((E == 255u) && (M != 0u), out) = v | U(0x00400000u);
  return std::bit_cast<native<float>>(out);
}

/// Vector port of the f32 `std::frexp` exponent over raw float bits, returned as
/// the raw bits of the int32 result. Normal lanes give E - 126; denormals give
/// p - 148 (p = floor(log2(M))); ±0 / Inf / NaN give 0 (the scalar body only
/// calls frexp on finite non-zero inputs, leaving exp = 0 otherwise). Negative
/// results carry the correct two's-complement bits. Bit-identical to the scalar
/// v_frexp_exp_i32_f32 body.
inline native<uint32_t> frexp_exp_f32_simd(native<uint32_t> v) {
  using U = native<uint32_t>;
  U E = (v >> 23) & U(0xFFu);
  U M = v & U(0x7FFFFFu);
  U normal = E - U(126u);
  const native<float> mf = stdx::static_simd_cast<native<float>>(M);
  const U p = (std::bit_cast<U>(mf) >> 23) - U(127u);
  U dn = p - U(148u);
  U out = normal;
  stdx::where(E == 0u, out) = U(0u);             // ±0 (M==0); overwritten if denormal
  stdx::where((E == 0u) && (M != 0u), out) = dn; // denormal exponent
  stdx::where(E == 255u, out) = U(0u);           // Inf / NaN -> 0
  return out;
}

/// 64-bit-lane port of the f64 `std::frexp` mantissa over native<double>. Same
/// structure as frexp_mant_f32_simd at f64 widths (11 exp bits bias 1023, 52
/// mantissa bits): normal lanes force the exponent field to 1022; denormals
/// renormalize via p = floor(log2(M)) read from double(M)'s exponent (M < 2^53,
/// exact); ±0 / Inf pass through; NaN is quieted (mantissa MSB set). Bit-identical
/// to the scalar v_frexp_mant_f64 body.
inline native<double> frexp_mant_f64_simd(native<double> x) {
  using U = native<uint64_t>;
  U v = std::bit_cast<U>(x);
  U sign = v & U(0x8000000000000000ull);
  U E = (v >> 52) & U(0x7FFull);
  U M = v & U(0xFFFFFFFFFFFFFull); // 52 mantissa bits
  U normal = sign | (U(1022ull) << 52) | M;
  const native<double> mf = stdx::static_simd_cast<native<double>>(M);
  const U p = (std::bit_cast<U>(mf) >> 52) - U(1023ull);
  U dn = sign | (U(1022ull) << 52) | ((M << (U(52ull) - p)) & U(0xFFFFFFFFFFFFFull));
  U out = normal;
  stdx::where(E == 0ull, out) = v;                   // ±0 (M==0); overwritten if denormal
  stdx::where((E == 0ull) && (M != 0ull), out) = dn; // denormal -> renormalized
  stdx::where(E == 2047ull, out) = v;                // Inf passes through unchanged
  stdx::where((E == 2047ull) && (M != 0ull), out) = v | U(0x0008000000000000ull); // quiet NaN
  return std::bit_cast<native<double>>(out);
}

/// 64-bit-lane port of the f64 `std::frexp` exponent. Returns each int32 result
/// in the low 32 bits of a native<uint64_t> lane (sign-extended), so the CVT
/// f64->b32 glue narrows it with one static_simd_cast. Normal lanes give
/// E - 1022; denormals p - 1073; ±0 / Inf / NaN give 0 (the scalar guards
/// frexp to finite non-zero inputs). Bit-identical to v_frexp_exp_i32_f64.
inline native<uint64_t> frexp_exp_f64_simd(native<double> x) {
  using U = native<uint64_t>;
  U v = std::bit_cast<U>(x);
  U E = (v >> 52) & U(0x7FFull);
  U M = v & U(0xFFFFFFFFFFFFFull);
  U normal = E - U(1022ull);
  const native<double> mf = stdx::static_simd_cast<native<double>>(M);
  const U p = (std::bit_cast<U>(mf) >> 52) - U(1023ull);
  U dn = p - U(1073ull);
  U out = normal;
  stdx::where(E == 0ull, out) = U(0ull);
  stdx::where((E == 0ull) && (M != 0ull), out) = dn;
  stdx::where(E == 2047ull, out) = U(0ull);
  return out;
}

/// Sum of the four per-byte absolute differences of two uint32 lanes (the core
/// of v_sad_u8 / v_sad_hi_u8). Each byte difference is in [0,255], so the sum is
/// in [0,1020]; bit-identical to the scalar byte loop.
inline native<uint32_t> sad_bytes_u32_simd(native<uint32_t> a, native<uint32_t> b) {
  using U = native<uint32_t>;
  U r(0u);
  for (int i = 0; i < 4; ++i) {
    U ai = (a >> (i * 8)) & U(0xFFu);
    U bi = (b >> (i * 8)) & U(0xFFu);
    U d = ai - bi;
    stdx::where(bi > ai, d) = bi - ai; // |ai - bi|
    r += d;
  }
  return r;
}

/// Masked per-byte SAD (v_msad_u8): like sad_bytes_u32_simd but a byte whose
/// reference (src0) value is zero contributes nothing. Bit-identical to scalar.
inline native<uint32_t> msad_bytes_u32_simd(native<uint32_t> a, native<uint32_t> b) {
  using U = native<uint32_t>;
  U r(0u);
  for (int i = 0; i < 4; ++i) {
    U ai = (a >> (i * 8)) & U(0xFFu);
    U bi = (b >> (i * 8)) & U(0xFFu);
    U d = ai - bi;
    stdx::where(bi > ai, d) = bi - ai;
    stdx::where(ai == 0u, d) = 0u; // skip masked-out (zero reference) bytes
    r += d;
  }
  return r;
}

/// Per-byte unsigned lerp (v_lerp_u8): out_byte = a + ((b - a) * c + 128) / 256,
/// computed independently per byte and repacked. The division is signed and
/// truncates toward zero (matching the scalar `int / 256`), so a negative
/// numerator gets the +1 floor->trunc correction. Bit-identical to the scalar.
inline native<uint32_t> lerp_u8_simd(native<uint32_t> a, native<uint32_t> b, native<uint32_t> c) {
  using U = native<uint32_t>;
  using I = native<int32_t>;
  U r(0u);
  for (int i = 0; i < 4; ++i) {
    const int sh = i * 8;
    I ab = stdx::static_simd_cast<I>((a >> sh) & U(0xFFu));
    I bb = stdx::static_simd_cast<I>((b >> sh) & U(0xFFu));
    I cb = stdx::static_simd_cast<I>((c >> sh) & U(0xFFu));
    I num = (bb - ab) * cb + I(128);
    I q = num >> 8; // arithmetic shift == floor division by 256
    stdx::where((num < 0) && ((num & I(255)) != 0), q) = q + I(1); // floor -> toward zero
    I res = ab + q;
    r = r | (stdx::static_simd_cast<U>(res) << sh);
  }
  return r;
}

/// Byte permute (v_perm_b32): build each output byte from a selector byte of
/// src2 indexing the 8 bytes of the {src0:src1} 64-bit source (src1 = low word).
/// Selector 0..7 picks a source byte; 0xD yields 0xFF; every other value yields
/// 0. Bit-identical to the scalar byte loop.
inline native<uint32_t> perm_b32_simd(native<uint32_t> a, native<uint32_t> b, native<uint32_t> c) {
  using U = native<uint32_t>;
  U srcbyte[8];
  for (int k = 0; k < 4; ++k)
    srcbyte[k] = (b >> (k * 8)) & U(0xFFu); // bytes 0..3 from the low word (src1)
  for (int k = 0; k < 4; ++k)
    srcbyte[k + 4] = (a >> (k * 8)) & U(0xFFu); // bytes 4..7 from the high word (src0)
  U r(0u);
  for (int i = 0; i < 4; ++i) {
    U sel = (c >> (i * 8)) & U(0xFFu);
    U byte(0u); // 0xC and all other selectors >= 8 -> 0
    for (int k = 0; k < 8; ++k)
      stdx::where(sel == U(static_cast<uint32_t>(k)), byte) = srcbyte[k];
    stdx::where(sel == U(0xDu), byte) = U(0xFFu); // 0xD -> 0xFF
    r = r | (byte << (i * 8));
  }
  return r;
}

/// High 32 bits of the 64-bit UNSIGNED product a*b (v_mul_hi_u32 semantics).
///
/// Computed purely with 32-bit-lane SIMD ops via the 16x16 partial-product
/// decomposition, deliberately avoiding `fixed_size_simd<uint64_t, N>`: clang +
/// libstdc++ miscompile the 64-bit-lane multiply/shift of an over-native-width
/// `fixed_size_simd` (the high half comes out wrong), so the int64-widening
/// approach diverges from the scalar `(uint64)a * b >> 32` at every SIMD width.
/// This decomposition uses only native<uint32_t> arithmetic, so it is
/// bit-identical to the scalar reference on every host. aHi/bHi/aLo/bLo are the
/// 16-bit halves; each 16x16 partial fits in 32 bits and the carry chain stays
/// below 2^32 (cross <= 3*0xFFFF), so no intermediate overflows.
inline native<uint32_t> mul_hi_u32_simd(native<uint32_t> a, native<uint32_t> b) {
  using U = native<uint32_t>;
  const U aLo = a & U(0xFFFFu), aHi = a >> 16;
  const U bLo = b & U(0xFFFFu), bHi = b >> 16;
  const U lolo = aLo * bLo;
  const U hilo = aHi * bLo;
  const U lohi = aLo * bHi;
  const U hihi = aHi * bHi;
  const U cross = (lolo >> 16) + (hilo & U(0xFFFFu)) + (lohi & U(0xFFFFu));
  return hihi + (hilo >> 16) + (lohi >> 16) + (cross >> 16);
}

/// High 32 bits of the 64-bit SIGNED product a*b (v_mul_hi_i32 semantics).
/// Derived from the unsigned high word with the standard signed correction
/// `hi_s = hi_u - (a<0 ? b : 0) - (b<0 ? a : 0)`. Same clang/libstdc++
/// motivation as `mul_hi_u32_simd`: no `fixed_size_simd<int64_t, N>`.
inline native<uint32_t> mul_hi_i32_simd(native<uint32_t> a, native<uint32_t> b) {
  using U = native<uint32_t>;
  U hi = mul_hi_u32_simd(a, b);
  // Sign correction in the uint domain (the negative test is the high bit, so
  // the where-mask type matches `hi` and avoids a signed/unsigned mask mismatch).
  stdx::where((a >> 31) != U(0u), hi) = hi - b;
  stdx::where((b >> 31) != U(0u), hi) = hi - a;
  return hi;
}
#endif // __has_include(<experimental/simd>)

#if !__has_include(<experimental/simd>)
// Fallback stub for non-template gtest callers (e.g. UtilSimd.FlushDenormF32),
// whose discarded `if constexpr (has_stdx_simd)` branch is still type-checked.
template <class T> native<T> flush_denorm_f32_simd(native<T>) { return {}; }
inline native<float> trunc_simd(native<float>) { return {}; }
inline native<float> ceil_simd(native<float>) { return {}; }
inline native<float> floor_simd(native<float>) { return {}; }
inline native<float> rndne_simd(native<float>) { return {}; }
inline native<double> trunc_simd(native<double>) { return {}; }
inline native<double> ceil_simd(native<double>) { return {}; }
inline native<double> floor_simd(native<double>) { return {}; }
inline native<double> rndne_simd(native<double>) { return {}; }
inline native<uint32_t> mul_hi_u32_simd(native<uint32_t>, native<uint32_t>) { return {}; }
inline native<uint32_t> mul_hi_i32_simd(native<uint32_t>, native<uint32_t>) { return {}; }
#endif

} // namespace util

#endif // UTIL_SIMD_H_
