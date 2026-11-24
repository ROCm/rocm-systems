// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_LANES_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_LANES_H_

/// @file Per-lane address forms of GLOBAL and FLAT accesses, whole lane vectors at a time.
///
/// @details Every GLOBAL/FLAT address calculation is this loop:
///
///   for each lane l < wf_size:
///     if l is in EXEC: per_lane_addr[l] = bias + widen(vgpr[l])
///
/// where bias folds the wave-uniform terms (SGPR base, immediate offset) once.
/// Addresses are sums modulo 2^64, so folding them first changes no address.
/// The forms differ only in how they widen the VGPR value:
///
///   Level 1  lane_addresses_u32   bias + zero-extended 32-bit offset (SADDR forms)
///            lane_addresses_s32   bias + sign-extended 32-bit offset * scale (gfx1250 SADDR)
///            lane_addresses_u64   bias + 64-bit VGPR pair (no SADDR)
///            lanes_in_window      lanes that may fall in a 4 GiB window (FLAT private aperture)
///   Level 2  the scalar bodies, which are the loop above, and AVX2 bodies
///   Level 3  EXEC bits as lane masks
///
/// Inactive lanes are written as 0 instead of skipped: the AVX2 bodies load
/// every lane and clear the inactive ones with an AND, which costs less than
/// testing EXEC per lane or a masked store. Writing 0 matches skipping because
/// every caller passes a freshly constructed VectorMemState, whose
/// per_lane_addr is value-initialized. The scalar bodies write 0 as well, so
/// the two are interchangeable slot for slot.
///
/// The AVX2 bodies need a constant wave size to unroll; other wave sizes and
/// hosts without AVX2 run the scalar body.

#include "util/host_target.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <span>

#if RJ_HOST_HAS_AVX2
#include <immintrin.h>
#endif

namespace rocjitsu {
namespace amdgpu {
namespace addr_calc {

/// @brief Lane values of an absent VGPR operand.
inline constexpr std::array<uint32_t, 64> kZeroLanes{};

namespace detail {

inline void lane_addresses_u32_scalar(const uint32_t *offsets, uint64_t exec, uint32_t wf_size,
                                      uint64_t bias, uint64_t *out);
inline void lane_addresses_s32_scalar(const uint32_t *offsets, uint64_t exec, uint32_t wf_size,
                                      uint64_t bias, uint32_t scale, uint64_t *out);
inline void lane_addresses_u64_scalar(const uint32_t *lo, const uint32_t *hi, uint64_t exec,
                                      uint32_t wf_size, uint64_t bias, uint64_t *out);
inline uint64_t lanes_in_window_scalar(const uint64_t *addresses, uint64_t lanes, uint32_t wf_size,
                                       uint64_t keep, uint64_t base);
#if RJ_HOST_HAS_AVX2
template <uint32_t WF>
void lane_addresses_u32_avx2(const uint32_t *offsets, uint64_t exec, uint64_t bias, uint64_t *out);
template <uint32_t WF>
void lane_addresses_s32_avx2(const uint32_t *offsets, uint64_t exec, uint64_t bias, uint32_t scale,
                             uint64_t *out);
template <uint32_t WF>
void lane_addresses_u64_avx2(const uint32_t *lo, const uint32_t *hi, uint64_t exec, uint64_t bias,
                             uint64_t *out);
template <uint32_t WF>
uint64_t lanes_in_window_avx2(const uint64_t *addresses, uint64_t lanes, uint64_t keep,
                              uint64_t base);
#endif

/// Mask of the lanes below @p wf_size; EXEC may carry bits past the wave.
constexpr uint64_t wave_lanes(uint32_t wf_size) {
  return wf_size >= 64 ? ~uint64_t{0} : (uint64_t{1} << wf_size) - 1;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Level 1: the forms callers use. Each writes out[0, wf_size).
// ---------------------------------------------------------------------------

/// @brief out[l] = bias + offsets[l] (zero-extended) for lanes in @p exec, 0 for the others.
inline void lane_addresses_u32(std::span<const uint32_t> offsets, uint64_t exec, uint32_t wf_size,
                               uint64_t bias, uint64_t *out) {
  assert(offsets.size() >= wf_size);
#if RJ_HOST_HAS_AVX2
  if (wf_size == 64)
    return detail::lane_addresses_u32_avx2<64>(offsets.data(), exec, bias, out);
  if (wf_size == 32)
    return detail::lane_addresses_u32_avx2<32>(offsets.data(), exec, bias, out);
#endif
  detail::lane_addresses_u32_scalar(offsets.data(), exec, wf_size, bias, out);
}

/// @brief out[l] = bias + int32(offsets[l]) * scale for lanes in @p exec, 0 for the others.
/// @details The offset is sign-extended before it is scaled, so the multiply is signed.
inline void lane_addresses_s32(std::span<const uint32_t> offsets, uint64_t exec, uint32_t wf_size,
                               uint64_t bias, uint32_t scale, uint64_t *out) {
  assert(offsets.size() >= wf_size);
  assert(scale <= INT32_MAX);
#if RJ_HOST_HAS_AVX2
  if (wf_size == 64)
    return detail::lane_addresses_s32_avx2<64>(offsets.data(), exec, bias, scale, out);
  if (wf_size == 32)
    return detail::lane_addresses_s32_avx2<32>(offsets.data(), exec, bias, scale, out);
#endif
  detail::lane_addresses_s32_scalar(offsets.data(), exec, wf_size, bias, scale, out);
}

/// @brief out[l] = bias + (hi[l] << 32 | lo[l]) for lanes in @p exec, 0 for the others.
inline void lane_addresses_u64(std::span<const uint32_t> lo, std::span<const uint32_t> hi,
                               uint64_t exec, uint32_t wf_size, uint64_t bias, uint64_t *out) {
  assert(lo.size() >= wf_size && hi.size() >= wf_size);
#if RJ_HOST_HAS_AVX2
  if (wf_size == 64)
    return detail::lane_addresses_u64_avx2<64>(lo.data(), hi.data(), exec, bias, out);
  if (wf_size == 32)
    return detail::lane_addresses_u64_avx2<32>(lo.data(), hi.data(), exec, bias, out);
#endif
  detail::lane_addresses_u64_scalar(lo.data(), hi.data(), exec, wf_size, bias, out);
}

/// @brief Lanes of @p lanes whose (address & keep) - base, modulo 2^64, is below 2^32.
/// @details With keep = ~0 and base = hi << 32 these are exactly the lanes whose upper
/// address word is hi, the private-aperture test. Callers translate the lanes it
/// returns exactly, so a window that over-approximates their test is still exact.
inline uint64_t lanes_in_window(const uint64_t *addresses, uint64_t lanes, uint32_t wf_size,
                                uint64_t keep, uint64_t base) {
  lanes &= detail::wave_lanes(wf_size);
  if (!lanes)
    return 0;
#if RJ_HOST_HAS_AVX2
  if (wf_size == 64)
    return detail::lanes_in_window_avx2<64>(addresses, lanes, keep, base);
  if (wf_size == 32)
    return detail::lanes_in_window_avx2<32>(addresses, lanes, keep, base);
#endif
  return detail::lanes_in_window_scalar(addresses, lanes, wf_size, keep, base);
}

namespace detail {

// ---------------------------------------------------------------------------
// Level 2: scalar bodies. These are the reference loops.
// ---------------------------------------------------------------------------

inline bool lane_active(uint64_t exec, uint32_t lane) { return (exec >> lane) & 1; }

inline void lane_addresses_u32_scalar(const uint32_t *offsets, uint64_t exec, uint32_t wf_size,
                                      uint64_t bias, uint64_t *out) {
  for (uint32_t lane = 0; lane < wf_size; ++lane)
    out[lane] = lane_active(exec, lane) ? bias + offsets[lane] : 0;
}

inline void lane_addresses_s32_scalar(const uint32_t *offsets, uint64_t exec, uint32_t wf_size,
                                      uint64_t bias, uint32_t scale, uint64_t *out) {
  for (uint32_t lane = 0; lane < wf_size; ++lane) {
    const int64_t offset = int64_t{static_cast<int32_t>(offsets[lane])} * scale;
    out[lane] = lane_active(exec, lane) ? bias + static_cast<uint64_t>(offset) : 0;
  }
}

inline void lane_addresses_u64_scalar(const uint32_t *lo, const uint32_t *hi, uint64_t exec,
                                      uint32_t wf_size, uint64_t bias, uint64_t *out) {
  for (uint32_t lane = 0; lane < wf_size; ++lane)
    out[lane] = lane_active(exec, lane) ? bias + (uint64_t{hi[lane]} << 32 | lo[lane]) : 0;
}

inline uint64_t lanes_in_window_scalar(const uint64_t *addresses, uint64_t lanes, uint32_t wf_size,
                                       uint64_t keep, uint64_t base) {
  uint64_t hits = 0;
  for (uint32_t lane = 0; lane < wf_size; ++lane)
    if (((addresses[lane] & keep) - base) >> 32 == 0)
      hits |= uint64_t{1} << lane;
  return hits & lanes;
}

#if RJ_HOST_HAS_AVX2

// ---------------------------------------------------------------------------
// Level 3: EXEC as lane masks.
// ---------------------------------------------------------------------------

/// Bit l of a 64-lane mask in element l, so four lanes' masks are one AND and one compare.
alignas(32) inline constexpr std::array<uint64_t, 64> kLaneBits = [] {
  std::array<uint64_t, 64> bits{};
  for (uint32_t lane = 0; lane < 64; ++lane)
    bits[lane] = uint64_t{1} << lane;
  return bits;
}();

/// All-ones in the 64-bit elements of lanes lane..lane+3 that are set in @p exec.
inline __m256i lane_mask4(__m256i exec, uint32_t lane) {
  const __m256i bits = _mm256_load_si256(reinterpret_cast<const __m256i *>(&kLaneBits[lane]));
  return _mm256_cmpeq_epi64(_mm256_and_si256(exec, bits), bits);
}

/// All-ones in the 32-bit elements of lanes lane..lane+7 that are set in @p exec.
inline __m256i lane_mask8(uint64_t exec, uint32_t lane) {
  // Moves bit i of the 8-lane chunk into the sign bit of element i.
  const __m256i bit_to_sign = _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24);
  const __m256i chunk = _mm256_set1_epi32(static_cast<int>((exec >> lane) & 0xFF));
  return _mm256_srai_epi32(_mm256_sllv_epi32(chunk, bit_to_sign), 31);
}

inline __m128i load4(const uint32_t *lanes) {
  return _mm_loadu_si128(reinterpret_cast<const __m128i *>(lanes));
}

inline void store4(uint64_t *out, __m256i value) {
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(out), value);
}

// ---------------------------------------------------------------------------
// Level 2: AVX2 bodies, four 64-bit lanes per vector.
// ---------------------------------------------------------------------------

template <uint32_t WF>
void lane_addresses_u32_avx2(const uint32_t *offsets, uint64_t exec, uint64_t bias, uint64_t *out) {
  static_assert(WF % 4 == 0 && WF <= 64);
  const __m256i vexec = _mm256_set1_epi64x(static_cast<long long>(exec));
  const __m256i vbias = _mm256_set1_epi64x(static_cast<long long>(bias));
#pragma GCC unroll 16
  for (uint32_t lane = 0; lane < WF; lane += 4) {
    const __m256i sum = _mm256_add_epi64(_mm256_cvtepu32_epi64(load4(offsets + lane)), vbias);
    store4(out + lane, _mm256_and_si256(sum, lane_mask4(vexec, lane)));
  }
}

template <uint32_t WF>
void lane_addresses_s32_avx2(const uint32_t *offsets, uint64_t exec, uint64_t bias, uint32_t scale,
                             uint64_t *out) {
  static_assert(WF % 4 == 0 && WF <= 64);
  const __m256i vexec = _mm256_set1_epi64x(static_cast<long long>(exec));
  const __m256i vbias = _mm256_set1_epi64x(static_cast<long long>(bias));
  const __m256i vscale = _mm256_set1_epi64x(scale);
#pragma GCC unroll 16
  for (uint32_t lane = 0; lane < WF; lane += 4) {
    // VPMULDQ multiplies the low 32 bits of each element as signed values, so
    // the product is int32(offset) * scale widened, as the scalar body forms it.
    const __m256i offset = _mm256_mul_epi32(_mm256_cvtepu32_epi64(load4(offsets + lane)), vscale);
    const __m256i sum = _mm256_add_epi64(offset, vbias);
    store4(out + lane, _mm256_and_si256(sum, lane_mask4(vexec, lane)));
  }
}

template <uint32_t WF>
void lane_addresses_u64_avx2(const uint32_t *lo, const uint32_t *hi, uint64_t exec, uint64_t bias,
                             uint64_t *out) {
  static_assert(WF % 8 == 0 && WF <= 64);
  const __m256i vexec = _mm256_set1_epi64x(static_cast<long long>(exec));
  const __m256i vbias = _mm256_set1_epi64x(static_cast<long long>(bias));
#pragma GCC unroll 8
  for (uint32_t lane = 0; lane < WF; lane += 8) {
    const __m256i l = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lo + lane));
    const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(hi + lane));
    // Interleaving within 128-bit halves pairs lanes {0,1,4,5} and {2,3,6,7};
    // the cross-half permutes put them back in lane order.
    const __m256i even = _mm256_unpacklo_epi32(l, h);
    const __m256i odd = _mm256_unpackhi_epi32(l, h);
    const __m256i first = _mm256_permute2x128_si256(even, odd, 0x20);
    const __m256i second = _mm256_permute2x128_si256(even, odd, 0x31);
    store4(out + lane, _mm256_and_si256(_mm256_add_epi64(first, vbias), lane_mask4(vexec, lane)));
    store4(out + lane + 4,
           _mm256_and_si256(_mm256_add_epi64(second, vbias), lane_mask4(vexec, lane + 4)));
  }
}

template <uint32_t WF>
uint64_t lanes_in_window_avx2(const uint64_t *addresses, uint64_t lanes, uint64_t keep,
                              uint64_t base) {
  static_assert(WF % 4 == 0 && WF <= 64);
  const __m256i vkeep = _mm256_set1_epi64x(static_cast<long long>(keep));
  const __m256i vbase = _mm256_set1_epi64x(static_cast<long long>(base));
  const __m256i zero = _mm256_setzero_si256();
  uint64_t hits = 0;
#pragma GCC unroll 16
  for (uint32_t lane = 0; lane < WF; lane += 4) {
    const __m256i address = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(addresses + lane));
    const __m256i offset = _mm256_sub_epi64(_mm256_and_si256(address, vkeep), vbase);
    const __m256i inside = _mm256_cmpeq_epi64(_mm256_srli_epi64(offset, 32), zero);
    hits |= uint64_t(_mm256_movemask_pd(_mm256_castsi256_pd(inside))) << lane;
  }
  return hits & lanes;
}

#endif // RJ_HOST_HAS_AVX2

} // namespace detail

} // namespace addr_calc
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_LANES_H_
