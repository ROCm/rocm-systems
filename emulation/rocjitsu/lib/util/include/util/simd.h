// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// SIMD register loads, stores and mask bridges for VM operand transport.
// GPU instruction arithmetic and numerical conversions belong to rj-arith.

#ifndef UTIL_SIMD_H_
#define UTIL_SIMD_H_

#include "util/bit.h"
#include "util/host_target.h"
#include "util/simd_control.h"

#include <bit>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if RJ_HOST_HAS_STDX_SIMD
#include <experimental/simd>
#endif

namespace util {

/// Compile-time switch for `<experimental/simd>` availability. Callers
/// gate SIMD fast paths via `if constexpr (util::has_stdx_simd)` so the
/// branch and all downstream calls compile out when the header is
/// absent (e.g. older libstdc++).
inline constexpr bool has_stdx_simd = RJ_HOST_HAS_STDX_SIMD;

#if RJ_HOST_HAS_STDX_SIMD
namespace stdx = std::experimental;

template <class T> using native = stdx::native_simd<T>;

/// Native-SIMD width measured in 32-bit lanes. Convenience constant.
template <class T> constexpr std::size_t native_width_v = native<T>::size();

/// Pack a SIMD predicate into the low mask.size() bits. Bit i corresponds to
/// SIMD lane i. libstdc++ exposes a bitset bridge that maps native masks to
/// movemask-style instructions; retain a lane fallback for other TS
/// implementations.
template <class Mask> [[gnu::always_inline]] inline uint64_t simd_mask_to_bits(const Mask &mask) {
  constexpr std::size_t W = Mask::size();
  static_assert(W <= 64);
  constexpr bool broken_native64_width =
      RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS && W == native<uint64_t>::size();
  if constexpr (!broken_native64_width && requires { mask.__to_bitset().to_ullong(); }) {
    return static_cast<uint64_t>(mask.__to_bitset().to_ullong());
  } else {
    uint64_t bits = 0;
    for (std::size_t i = 0; i < W; ++i) {
      if (mask[i]) {
        bits |= uint64_t{1} << i;
      }
    }
    return bits;
  }
}

/// Expand the low Simd::size() bits into a SIMD predicate. High input bits are
/// ignored. The libstdc++ bitset bridge lowers to a broadcast, lane-bit test,
/// and vector compare instead of constructing the predicate lane by lane.
template <class Simd>
[[gnu::always_inline]] inline typename Simd::mask_type simd_mask_from_bits(uint64_t bits) {
  using Mask = typename Simd::mask_type;
  constexpr std::size_t W = Simd::size();
  static_assert(W <= 64);
  constexpr bool broken_native64_width =
      RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS && W == native<uint64_t>::size();
  if constexpr (!broken_native64_width && requires { Mask::__from_bitset(std::bitset<W>(bits)); }) {
    return Mask::__from_bitset(std::bitset<W>(bits));
  } else {
    Mask mask(false);
    for (std::size_t i = 0; i < W; ++i) {
      mask[i] = ((bits >> i) & 1u) != 0;
    }
    return mask;
  }
}

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
template <class Mask> uint64_t simd_mask_to_bits(const Mask &) { return 0; }
template <class Simd> bool simd_mask_from_bits(uint64_t) { return false; }
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

#if RJ_HOST_HAS_STDX_SIMD
/// Load `native<T>` from contiguous uint32_t storage. T must be a
/// 32-bit trivially-copyable type.
template <class T> native<T> load(const uint32_t *p) {
  using Bits = stdx::native_simd<uint32_t>;
  using Val = native<T>;
  static_assert(sizeof(T) == sizeof(uint32_t));
  static_assert(sizeof(Val) == sizeof(Bits));
  Bits bits(p, stdx::element_aligned);
  if constexpr (std::is_same_v<T, uint32_t>) {
    return bits;
  } else {
    return std::bit_cast<Val>(bits);
  }
}

/// Broadcast `broadcast_bits` (bit-cast to T) to every lane of
/// `native<T>`. T must be a 32-bit trivially-copyable type.
template <class T> native<T> broadcast(uint32_t broadcast_bits) {
  using Val = native<T>;
  static_assert(sizeof(T) == sizeof(uint32_t));
  if constexpr (std::is_same_v<T, uint32_t>) {
    return Val(broadcast_bits);
  } else {
    return Val(std::bit_cast<T>(broadcast_bits));
  }
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
    if constexpr (std::is_same_v<T, uint32_t>) {
      return v;
    } else {
      return std::bit_cast<Bits>(v);
    }
  }();
  if ((mask & full) == full) {
    bits.copy_to(dst, stdx::element_aligned);
    return;
  }
  alignas(Bits) uint32_t buf[W];
  bits.copy_to(buf, stdx::vector_aligned);
  for (std::size_t i = 0; i < W; ++i) {
    if (mask & (1ULL << i)) {
      dst[i] = buf[i];
    }
  }
}

/// Same as masked_store, but writes to a caller-supplied uint32_t buffer
/// instead of contiguous lane storage. For operands whose dst is not a
/// contiguous VGPR (rocjitsu falls back to write_lane_chunk in that case).
template <class T> void blit_to_buffer(uint32_t (&buf)[native<T>::size()], native<T> v) {
  using Bits = stdx::native_simd<uint32_t>;
  Bits bits = [&] {
    if constexpr (std::is_same_v<T, uint32_t>) {
      return v;
    } else {
      return std::bit_cast<Bits>(v);
    }
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
  for (std::size_t i = 0; i < W; ++i) {
    buf[i] = (static_cast<uint64_t>(hi[i]) << 32) | lo[i];
  }
  U64 bits(buf, stdx::vector_aligned);
  if constexpr (std::is_same_v<T, uint64_t>) {
    return bits;
  } else {
    return std::bit_cast<native<T>>(bits);
  }
}

/// Broadcast `broadcast_bits` (bit-cast to T) to every lane of `native<T>` for a
/// 64-bit lane type. The companion of `broadcast` for the f64/i64 path; used when
/// a source operand resolves to a scalar/immediate rather than per-lane storage.
template <class T> native<T> broadcast64(uint64_t broadcast_bits) {
  static_assert(sizeof(T) == sizeof(uint64_t));
  using Val = native<T>;
  if constexpr (std::is_same_v<T, uint64_t>) {
    return Val(broadcast_bits);
  } else {
    return Val(std::bit_cast<T>(broadcast_bits));
  }
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
    if constexpr (std::is_same_v<T, uint64_t>) {
      return v;
    } else {
      return std::bit_cast<U64>(v);
    }
  }();
  alignas(U64) uint64_t buf[W];
  bits.copy_to(buf, stdx::vector_aligned);
  for (std::size_t i = 0; i < W; ++i) {
    if (mask & (1ULL << i)) {
      lo[i] = static_cast<uint32_t>(buf[i]);
      hi[i] = static_cast<uint32_t>(buf[i] >> 32);
    }
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
    for (std::size_t i = 0; i < native_width64; ++i) {
      buf[i] = std::bit_cast<T>(p[i]);
    }
    return narrow32<T>(buf, stdx::vector_aligned);
  }
}

/// Broadcast `broadcast_bits` (bit-cast to T) to every lane of `narrow32<T>`.
/// Companion of `broadcast` for the narrow (8-wide) cvt path; used when a source
/// operand resolves to a scalar/immediate rather than per-lane storage.
template <class T> narrow32<T> broadcast_narrow(uint32_t broadcast_bits) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  if constexpr (std::is_same_v<T, uint32_t>) {
    return narrow32<T>(broadcast_bits);
  } else {
    return narrow32<T>(std::bit_cast<T>(broadcast_bits));
  }
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
    for (std::size_t i = 0; i < W; ++i) {
      dst[i] = std::bit_cast<uint32_t>(buf[i]);
    }
    return;
  }
  for (std::size_t i = 0; i < W; ++i) {
    if (mask & (1ULL << i)) {
      dst[i] = std::bit_cast<uint32_t>(buf[i]);
    }
  }
}

#endif // RJ_HOST_HAS_STDX_SIMD

} // namespace util

#endif // UTIL_SIMD_H_
