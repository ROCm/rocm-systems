// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_SIMD_H_
#define UTIL_SIMD_H_

#include "util/bit.h"

#include <bit>
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

/// Process-wide, immutable switch that callers check before taking the
/// SIMD fast path. Read ONCE from the `RJ_FORCE_SCALAR` env var on first
/// call (unset/empty/"0" => false, any other value => true) and fixed for
/// the process lifetime. Returned by value: there is no setter and no
/// mutable reference, so production code cannot flip it at runtime and the
/// public API leaks no mutable global state.
///
/// A/B testing selects the mode by setting `RJ_FORCE_SCALAR` before launch
/// (the SIMD correctness suite is run twice and the per-case result dumps
/// diffed; see tests/simd_ab.h), and e2e runs force the scalar codepath the
/// same way without recompiling.
inline bool force_scalar() {
  static const bool flag = [] {
    const char *e = std::getenv("RJ_FORCE_SCALAR");
    return e && e[0] && !(e[0] == '0' && e[1] == '\0');
  }();
  return flag;
}

#if __has_include(<experimental/simd>)
namespace stdx = std::experimental;

template <class T> using native = stdx::native_simd<T>;

/// Native-SIMD width measured in 32-bit lanes. Convenience constant.
template <class T> inline constexpr std::size_t native_width_v = native<T>::size();
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
template <class T> inline constexpr std::size_t native_width_v = native<T>::size();
template <class T> inline native<T> load(const uint32_t *) { return {}; }
template <class T> inline native<T> broadcast(uint32_t) { return {}; }
template <class T> inline void masked_store(uint32_t *, native<T>, uint64_t) {}
template <class T> inline void blit_to_buffer(uint32_t (&)[native<T>::size()], native<T>) {}
template <class T> inline native<T> load64(const uint32_t *, const uint32_t *) { return {}; }
template <class T> inline native<T> broadcast64(uint64_t) { return {}; }
template <class T> inline void masked_store64(uint32_t *, uint32_t *, native<T>, uint64_t) {}
template <class T> struct narrow32 {
  static constexpr std::size_t size() { return 1; }
  constexpr T operator[](std::size_t) const { return T{}; }
};
template <class T> inline narrow32<T> load_narrow(const uint32_t *) { return {}; }
template <class T> inline narrow32<T> broadcast_narrow(uint32_t) { return {}; }
template <class T> inline void masked_store_narrow(uint32_t *, narrow32<T>, uint64_t) {}
#endif

/// Native-SIMD width for 64-bit lane types (e.g. native<uint64_t>/<double>),
/// measured in 64-bit lanes — half of native_width_v<uint32_t> on the same
/// vector width (8 vs 16 on AVX-512). Defined in both branches: in the
/// no-`<experimental/simd>` fallback native<uint64_t>::size() is 1.
inline constexpr std::size_t native_width64 = native<uint64_t>::size();

#if __has_include(<experimental/simd>)
/// Load `native<T>` from contiguous uint32_t storage. T must be a
/// 32-bit trivially-copyable type.
template <class T> inline native<T> load(const uint32_t *p) {
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
template <class T> inline native<T> broadcast(uint32_t broadcast_bits) {
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
template <class T> inline void masked_store(uint32_t *dst, native<T> v, uint64_t mask) {
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
template <class T> inline void blit_to_buffer(uint32_t (&buf)[native<T>::size()], native<T> v) {
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
template <class T> inline native<T> load64(const uint32_t *lo, const uint32_t *hi) {
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
template <class T> inline native<T> broadcast64(uint64_t broadcast_bits) {
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
template <class T>
inline void masked_store64(uint32_t *lo, uint32_t *hi, native<T> v, uint64_t mask) {
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
template <class T> inline narrow32<T> load_narrow(const uint32_t *p) {
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
template <class T> inline narrow32<T> broadcast_narrow(uint32_t broadcast_bits) {
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
template <class T> inline void masked_store_narrow(uint32_t *dst, narrow32<T> v, uint64_t mask) {
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
/// case (mant==0 -> +0 | signbit); exp==15 yields the max normal (mant==0) or a
/// canonical qNaN (mant!=0). Used by the v_cvt_f32_fp8 SIMD fast path.
inline native<float> fp8_e4m3_to_f32_simd(native<uint32_t> v) {
  using U = native<uint32_t>;
  U b = v & U(0xFFu);
  U sign = (b >> 7) & U(1u);
  U exp = (b >> 3) & U(0xFu);
  U mant = b & U(0x7u);
  U signbit = sign << 31;
  U normal = signbit | ((exp + U(120u)) << 23) | (mant << 20); // exp + 127 - 7
  U inf_nan = signbit | U(0x43800000u);                        // exp15, mant0 -> max normal
  stdx::where(mant != 0u, inf_nan) = signbit | U(0x7FC00000u); // exp15, mant!=0 -> qNaN
  native<float> dn =
      stdx::static_simd_cast<native<float>>(mant) * native<float>(0.001953125f); // 2^-9
  U dnb = std::bit_cast<U>(dn) | signbit; // exp0: ±denormal, or ±0 when mant==0
  U out = normal;
  stdx::where(exp == 0u, out) = dnb;
  stdx::where(exp == 15u, out) = inf_nan;
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
#endif // __has_include(<experimental/simd>)

#if !__has_include(<experimental/simd>)
// Fallback stub for non-template gtest callers (e.g. UtilSimd.FlushDenormF32),
// whose discarded `if constexpr (has_stdx_simd)` branch is still type-checked.
template <class T> inline native<T> flush_denorm_f32_simd(native<T>) { return {}; }
#endif

} // namespace util

#endif // UTIL_SIMD_H_
