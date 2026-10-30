// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file lds_writeback_avx2.h
/// @brief Isa::avx2 for lds_writeback.h: the same levels with 8-lane vectors.
///
/// @details Included by lds_writeback.h in AVX2 builds.
///
///   Level 1  by EXEC shape: plain or masked stores
///   Level 2  8 lanes of each destination register, per format
///   Level 3  EXEC masks and loads
///
/// Its 2.9 ns lead over scalar on the census mix: about 1 ns from the cells,
/// where 8-lane shuffles take a third of scalar's instructions on b128, b64,
/// u16 and Chunks (b128: 182 vs 579 per call) and scalar is 0.2-0.5 ns faster
/// on b32 Wave and FewLanes; the rest appears only on the mix, which is where
/// scalar's EXEC-shape branches mispredict (0.26 vs 0.03 per call).

#ifndef ROCJITSU_VM_AMDGPU_LDS_WRITEBACK_AVX2_H_
#define ROCJITSU_VM_AMDGPU_LDS_WRITEBACK_AVX2_H_

#include "rocjitsu/vm/amdgpu/lds_writeback.h"

#include <immintrin.h>

namespace rocjitsu {
namespace amdgpu {
namespace lds_writeback {
namespace avx2 {

enum class Stores { Plain, Masked };

template <DstFormat F, uint32_t Wave, Stores S>
[[gnu::always_inline]] inline void store_chunks(const Target<F, Wave> &t, uint64_t exec);
template <DstFormat F, uint32_t Wave>
[[gnu::always_inline]] inline void values(const Target<F, Wave> &t, uint32_t c, __m256i *v);
[[gnu::always_inline]] inline __m256i block(const uint8_t *src);
[[gnu::always_inline]] inline void unzip(const uint8_t *src, __m256i *v);
[[gnu::always_inline]] inline void transpose(const uint8_t *src, __m256i *v);
[[gnu::always_inline]] inline void gather(const uint8_t *src, __m256i *v);
template <DstFormat F>
[[gnu::always_inline]] inline __m256i extend(const uint8_t *src, const uint8_t *old, bool ecc);
[[gnu::always_inline]] inline __m256i exec_mask(uint64_t exec, uint32_t c);
[[gnu::always_inline]] inline __m256i load(const void *p);

// ---------------------------------------------------------------------------
// Level 1: by EXEC shape.
// ---------------------------------------------------------------------------

/// b32 always takes Masked: its EXEC is a full wave in 49.5% of the census and
/// FewLanes in 29.3%, so a branch on it mispredicts (mix 6.3 -> 7.9 ns with
/// one, though b32 Wave alone drops 3.8 -> 3.5). Other formats are a full wave
/// almost always, and masked stores there cost the mix 1.5 ns (7.8 -> 6.3 with
/// plain stores).
template <DstFormat F, uint32_t Wave> [[gnu::always_inline]] inline Stores choose(uint64_t exec) {
  if (kStride<F> != 4 && exec == kAll<Wave>)
    return Stores::Plain;
  return Stores::Masked;
}

template <DstFormat F, uint32_t Wave> void by_exec(const Target<F, Wave> &t, uint64_t exec) {
  switch (choose<F, Wave>(exec)) {
  case Stores::Plain:
    return store_chunks<F, Wave, Stores::Plain>(t, exec);
  case Stores::Masked:
    return store_chunks<F, Wave, Stores::Masked>(t, exec);
  }
}

/// Stores as a template parameter, not a branch in the loop: b64 full 8.1 ->
/// 5.8 ns.
template <DstFormat F, uint32_t Wave, Stores S>
void store_chunks(const Target<F, Wave> &t, uint64_t exec) {
  for (uint32_t c = 0; c < Wave / 8; ++c) {
    __m256i v[kRegs<F>];
    values<F, Wave>(t, c, v);
    for (uint32_t r = 0; r < kRegs<F>; ++r) {
      uint8_t *p = t.dst[r] + c * 32;
      if constexpr (S == Stores::Masked)
        _mm256_maskstore_epi32(reinterpret_cast<int *>(p), exec_mask(exec, c), v[r]);
      else
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), v[r]);
    }
  }
}

// ---------------------------------------------------------------------------
// Level 2: lanes 8c..8c+7 of each destination register, per format.
// ---------------------------------------------------------------------------

/// Chunk c is the first or second half of group c / 2.
template <DstFormat F, uint32_t Wave>
void values(const Target<F, Wave> &t, uint32_t c, __m256i *v) {
  const uint8_t *src = t.group[c / 2] + (c % 2) * 8 * kStride<F>;
  if constexpr (kStride<F> == 4)
    v[0] = block(src);
  else if constexpr (kRegs<F> == 2)
    unzip(src, v);
  else if constexpr (kRegs<F> == 4)
    transpose(src, v);
  else if constexpr (kRegs<F> == 3)
    gather(src, v);
  else
    v[0] = extend<F>(src, t.dst[0] + c * 32, t.ecc);
}

/// b32: lane-major data is, for one register, the register layout.
__m256i block(const uint8_t *src) { return load(src); }

/// b64: the even and odd dwords of two 4-lane loads.
///   a = │L0.r0 L0.r1 L1.r0 L1.r1│L2 .. L3 ..│   b = │L4 .. L5 ..│L6 .. L7 ..│
///   shufps 0x88 -> │L0 L1 L4 L5│L2 L3 L6 L7│ (r0);  vpermq 0xD8 -> │L0 .. L7│
void unzip(const uint8_t *src, __m256i *v) {
  const __m256 a = _mm256_castsi256_ps(load(src)), b = _mm256_castsi256_ps(load(src + 32));
  v[0] = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(a, b, 0x88)), 0xD8);
  v[1] = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(a, b, 0xDD)), 0xD8);
}

/// b128: lane j in the low half and lane j + 4 in the high half, so one 4x4
/// transpose per half gives lanes 0..7 in order with no cross-half shuffle.
///   x0 = │L0.r0 L0.r1 L0.r2 L0.r3│L4.r0 ..│  ...  x3 = │L3 ..│L7 ..│
///   v[r] = │L0.r L1.r L2.r L3.r│L4.r L5.r L6.r L7.r│
void transpose(const uint8_t *src, __m256i *v) {
  const auto *f = reinterpret_cast<const float *>(src);
  const __m256 x0 = _mm256_loadu2_m128(f + 16, f), x1 = _mm256_loadu2_m128(f + 20, f + 4);
  const __m256 x2 = _mm256_loadu2_m128(f + 24, f + 8), x3 = _mm256_loadu2_m128(f + 28, f + 12);
  const __m256 t0 = _mm256_unpacklo_ps(x0, x1), t1 = _mm256_unpacklo_ps(x2, x3);
  const __m256 t2 = _mm256_unpackhi_ps(x0, x1), t3 = _mm256_unpackhi_ps(x2, x3);
  v[0] = _mm256_castps_si256(_mm256_shuffle_ps(t0, t1, 0x44));
  v[1] = _mm256_castps_si256(_mm256_shuffle_ps(t0, t1, 0xEE));
  v[2] = _mm256_castps_si256(_mm256_shuffle_ps(t2, t3, 0x44));
  v[3] = _mm256_castps_si256(_mm256_shuffle_ps(t2, t3, 0xEE));
}

/// b96 is 0.07% of the census: a gather instead of a third transpose.
void gather(const uint8_t *src, __m256i *v) {
  const __m256i lane = _mm256_setr_epi32(0, 12, 24, 36, 48, 60, 72, 84);
  for (uint32_t r = 0; r < 3; ++r)
    v[r] = _mm256_i32gather_epi32(reinterpret_cast<const int *>(src + 4 * r), lane, 1);
}

/// u8, i8, u16, i16 and their d16 forms: the values of scalar::narrow.
template <DstFormat F> __m256i extend(const uint8_t *src, const uint8_t *old, bool ecc) {
  static_assert(F.num_elems == 1 && (F.elem_size == 1 || F.elem_size == 2));
  __m256i h;
  if constexpr (F.elem_size == 1) {
    const __m128i x = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src));
    h = F.sign_extend ? _mm256_cvtepi8_epi32(x) : _mm256_cvtepu8_epi32(x);
  } else {
    const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
    h = F.sign_extend ? _mm256_cvtepi16_epi32(x) : _mm256_cvtepu16_epi32(x);
  }
  if constexpr (kD16<F>) {
    const __m256i keep = ecc ? _mm256_setzero_si256() : load(old);
    h = F.d16_hi ? _mm256_blend_epi16(keep, _mm256_slli_epi32(h, 16), 0xAA)
                 : _mm256_blend_epi16(keep, h, 0x55);
  }
  return h;
}

// ---------------------------------------------------------------------------
// Level 3: EXEC masks and loads.
// ---------------------------------------------------------------------------

/// Lanes 8c..8c+7 of exec, each in its dword's sign bit (what maskstore reads).
__m256i exec_mask(uint64_t exec, uint32_t c) {
  const __m256i half = _mm256_set1_epi32(static_cast<int>(exec >> (32 * (c / 4))));
  const int s = 31 - 8 * static_cast<int>(c % 4);
  return _mm256_sllv_epi32(half,
                           _mm256_setr_epi32(s, s - 1, s - 2, s - 3, s - 4, s - 5, s - 6, s - 7));
}

__m256i load(const void *p) { return _mm256_loadu_si256(static_cast<const __m256i *>(p)); }

} // namespace avx2
} // namespace lds_writeback
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_LDS_WRITEBACK_AVX2_H_
