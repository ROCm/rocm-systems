// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_SIMD_NO_STDX_H_
#define UTIL_DETAIL_SIMD_NO_STDX_H_

#include "util/detail/simd/types.h"

namespace util {

// Fallback definitions so `if constexpr (has_stdx_simd)` discarded
// branches compile out even in non-template callers (e.g. gtest TEST
// bodies), where the discarded branch is still fully type-checked and
// odr-used. They are never executed: the constexpr condition is `false`
// and callers skip the SIMD path first.
template <class T> native<T> load(const uint32_t *) { return {}; }
template <class T> native<T> broadcast(uint32_t) { return {}; }
template <class Mask> uint64_t simd_mask_to_bits(const Mask &) { return 0; }
template <class Simd> bool simd_mask_from_bits(uint64_t) { return false; }
inline native<uint32_t> simd_u32_lanes_from_bits(uint64_t) { return {}; }
inline native<float> native_fma(native<float>, native<float>, native<float>) { return {}; }
template <class T> void masked_store(uint32_t *, native<T>, uint64_t) {}
template <class T> void blit_to_buffer(uint32_t (&)[native<T>::size()], native<T>) {}
template <class T> native<T> load64(const uint32_t *, const uint32_t *) { return {}; }
template <class T> native<T> broadcast64(uint64_t) { return {}; }
template <class T> void masked_store64(uint32_t *, uint32_t *, native<T>, uint64_t) {}
template <class T, class Fn> native<T> map_native_scalar(native<T>, native<T>, Fn) { return {}; }
template <class T, class Fn> native<T> map_native_scalar(native<T>, native<T>, native<T>, Fn) {
  return {};
}
template <class To, class From, class Fn> native<To> map_native_convert_scalar(native<From>, Fn) {
  return {};
}
template <class T, class Fn> native<T> map_native64_scalar(native<T>, Fn) { return {}; }
template <class T, class Fn> native<T> map_native64_scalar(native<T>, native<T>, Fn) { return {}; }
template <class T> narrow32<T> load_narrow(const uint32_t *) { return {}; }
template <class T> narrow32<T> broadcast_narrow(uint32_t) { return {}; }
template <class T> void masked_store_narrow(uint32_t *, narrow32<T>, uint64_t) {}

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

} // namespace util

#endif // UTIL_DETAIL_SIMD_NO_STDX_H_
