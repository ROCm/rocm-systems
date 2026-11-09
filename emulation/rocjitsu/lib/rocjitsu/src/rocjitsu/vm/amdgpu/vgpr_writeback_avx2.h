// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vgpr_writeback_avx2.h
/// @brief Isa::avx2 for vgpr_writeback.h: eight lanes at a time.

#ifndef ROCJITSU_VM_AMDGPU_VGPR_WRITEBACK_AVX2_H_
#define ROCJITSU_VM_AMDGPU_VGPR_WRITEBACK_AVX2_H_

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/vm/amdgpu/vgpr_writeback.h"

#include <immintrin.h>

namespace rocjitsu {
namespace amdgpu {
namespace vgpr_writeback {
namespace avx2 {

enum class WriteMode { Unmasked, Masked };

template <DstFormat F, uint32_t Wave, WriteMode M>
RJ_ALWAYS_INLINE inline void store_all_chunks(const Target<F, Wave> &t, uint64_t exec);
template <DstFormat F, uint32_t Wave>
RJ_ALWAYS_INLINE inline void chunk_values(const Target<F, Wave> &t, uint32_t c, __m256i *v);
RJ_ALWAYS_INLINE inline __m256i b32_value(const uint8_t *src);
RJ_ALWAYS_INLINE inline void b64_values(const uint8_t *src, __m256i *v);
RJ_ALWAYS_INLINE inline void b96_values(const uint8_t *src, __m256i *v);
template <uint32_t R> RJ_ALWAYS_INLINE inline __m256i b96_column(__m256i a, __m256i b, __m256i c);
RJ_ALWAYS_INLINE inline void b128_values(const uint8_t *src, __m256i *v);
template <DstFormat F>
RJ_ALWAYS_INLINE inline __m256i narrow_values(const uint8_t *src, const uint8_t *old, bool ecc);
RJ_ALWAYS_INLINE inline __m256i maskstore_mask(uint64_t exec, uint32_t c);
RJ_ALWAYS_INLINE inline __m256i load_unaligned(const void *p);

template <DstFormat F, uint32_t Wave>
RJ_ALWAYS_INLINE inline WriteMode pick_write_mode(uint64_t exec) {
  if constexpr (kStride<F> == 4) // only for b32; faster to mask than branch
    return WriteMode::Masked;
  return exec == prefix(Wave) ? WriteMode::Unmasked : WriteMode::Masked;
}

template <DstFormat F, uint32_t Wave> void by_exec(const Target<F, Wave> &t, uint64_t exec) {
  switch (pick_write_mode<F, Wave>(exec)) {
  case WriteMode::Unmasked:
    return store_all_chunks<F, Wave, WriteMode::Unmasked>(t, exec);
  case WriteMode::Masked:
    return store_all_chunks<F, Wave, WriteMode::Masked>(t, exec);
  }
}

/// The write mode is a template parameter, not a branch in the loop: a branch
/// there made full b64 waves much slower.
template <DstFormat F, uint32_t Wave, WriteMode M>
void store_all_chunks(const Target<F, Wave> &t, uint64_t exec) {
  for (uint32_t c = 0; c < Wave / 8; ++c) {
    __m256i v[kRegs<F>];
    chunk_values<F, Wave>(t, c, v);
    for (uint32_t r = 0; r < kRegs<F>; ++r) {
      uint8_t *p = t.dst[r] + c * 32;
      if constexpr (M == WriteMode::Masked)
        _mm256_maskstore_epi32(reinterpret_cast<int *>(p), maskstore_mask(exec, c), v[r]);
      else
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), v[r]);
    }
  }
}

template <DstFormat F, uint32_t Wave>
void chunk_values(const Target<F, Wave> &t, uint32_t c, __m256i *v) {
  const uint8_t *src = t.group[c / 2] + (c % 2) * 8 * kStride<F>;
  if constexpr (kStride<F> == 4)
    v[0] = b32_value(src);
  else if constexpr (kRegs<F> == 2)
    b64_values(src, v);
  else if constexpr (kRegs<F> == 3)
    b96_values(src, v);
  else if constexpr (kRegs<F> == 4)
    b128_values(src, v);
  else
    v[0] = narrow_values<F>(src, t.dst[0] + c * 32, t.ecc);
}

/// b32: lane-major data is, for one register, the register layout.
__m256i b32_value(const uint8_t *src) { return load_unaligned(src); }

/// b64: even dwords are register 0, odd dwords are register 1.
void b64_values(const uint8_t *src, __m256i *v) {
  const __m256 a = _mm256_castsi256_ps(load_unaligned(src));
  const __m256 b = _mm256_castsi256_ps(load_unaligned(src + 32));
  constexpr int kEven = 0x88;    // shufps: dwords 0 and 2 of each operand (r0)
  constexpr int kOdd = 0xDD;     // shufps: dwords 1 and 3 of each operand (r1)
  constexpr int kInOrder = 0xD8; // vpermq: qwords 0, 2, 1, 3
  v[0] = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(a, b, kEven)), kInOrder);
  v[1] = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(a, b, kOdd)), kInOrder);
}

/// b96: Three dwords per lane. The lanes wrap around the registers.
///   a │L0.r0 L0.r1 L0.r2 ... ... ... L2.r0 L2.r1│ <-- L2.r2 wrapped to the next register
///   b │L2.r2 ... ... ... ... ... ... L5.r0│
///   c │L5.r1 L5.r2 ... ... ... L7.r0 L7.r1 L7.r2│
///
/// Register 0 is │a0 a3 a6│b1 b4 b7│c2 c5│
/// @note A gather does it in one instruction and is much slower.
///
void b96_values(const uint8_t *src, __m256i *v) {
  const __m256i a = load_unaligned(src);
  const __m256i b = load_unaligned(src + 32);
  const __m256i c = load_unaligned(src + 64);
  v[0] = b96_column<0>(a, b, c);
  v[1] = b96_column<1>(a, b, c);
  v[2] = b96_column<2>(a, b, c);
}

template <uint32_t R> constexpr int b96_row_lanes(uint32_t row) {
  int mask = 0;
  for (uint32_t l = 0; l < 8; ++l)
    if ((3 * l + R) / 8 == row)
      mask |= 1 << l;
  return mask;
}

template <uint32_t R> __m256i b96_column(__m256i a, __m256i b, __m256i c) {
  const __m256i column =
      _mm256_setr_epi32((3 * 0 + R) % 8, (3 * 1 + R) % 8, (3 * 2 + R) % 8, (3 * 3 + R) % 8,
                        (3 * 4 + R) % 8, (3 * 5 + R) % 8, (3 * 6 + R) % 8, (3 * 7 + R) % 8);
  const __m256i pa = _mm256_permutevar8x32_epi32(a, column);
  const __m256i pb = _mm256_permutevar8x32_epi32(b, column);
  const __m256i pc = _mm256_permutevar8x32_epi32(c, column);
  return _mm256_blend_epi32(_mm256_blend_epi32(pa, pb, b96_row_lanes<R>(1)), pc,
                            b96_row_lanes<R>(2));
}

/// b128: Four dwords per lane. Lane j and lane j + 4 share a half.
///   x0 │L0.r0 L0.r1 L0.r2 L0.r3│L4.r0 ... ... L4.r3│
///   x1 │L1.r0 ... ... L1.r3│L5.r0 ... ... L5.r3│
///   x2 │L2.r0 ... ... L2.r3│L6.r0 ... ... L6.r3│
///   x3 │L3.r0 ... ... L3.r3│L7.r0 ... ... L7.r3│
///
/// Register r is │L0.r L1.r L2.r L3.r│L4.r L5.r L6.r L7.r│
void b128_values(const uint8_t *src, __m256i *v) {
  const auto *f = reinterpret_cast<const float *>(src);
  const __m256 x0 = _mm256_loadu2_m128(f + 16, f);
  const __m256 x1 = _mm256_loadu2_m128(f + 20, f + 4);
  const __m256 x2 = _mm256_loadu2_m128(f + 24, f + 8);
  const __m256 x3 = _mm256_loadu2_m128(f + 28, f + 12);
  const __m256 t0 = _mm256_unpacklo_ps(x0, x1), t1 = _mm256_unpacklo_ps(x2, x3);
  const __m256 t2 = _mm256_unpackhi_ps(x0, x1), t3 = _mm256_unpackhi_ps(x2, x3);
  constexpr int kLow = 0x44;  // low pair of each source
  constexpr int kHigh = 0xEE; // high pair of each source
  v[0] = _mm256_castps_si256(_mm256_shuffle_ps(t0, t1, kLow));
  v[1] = _mm256_castps_si256(_mm256_shuffle_ps(t0, t1, kHigh));
  v[2] = _mm256_castps_si256(_mm256_shuffle_ps(t2, t3, kLow));
  v[3] = _mm256_castps_si256(_mm256_shuffle_ps(t2, t3, kHigh));
}

/// u8, i8, u16, i16. One element per lane, widened into the dword.
///   u8 u16 i8 i16  │... ... h│
///   d16_hi         │h │old│ <-- SRAM ECC: │h │0│
///   d16_lo         │old│ h│ <-- SRAM ECC: │0 │h│
template <DstFormat F> __m256i narrow_values(const uint8_t *src, const uint8_t *old, bool ecc) {
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
    const __m256i keep = ecc ? _mm256_setzero_si256() : load_unaligned(old);
    constexpr int kHi = 0xAA;
    constexpr int kLo = 0x55;
    h = F.d16_hi ? _mm256_blend_epi16(keep, _mm256_slli_epi32(h, 16), kHi)
                 : _mm256_blend_epi16(keep, h, kLo);
  }
  return h;
}

/// maskstore reads the sign bit, so lane i's EXEC bit has to land on it.
__m256i maskstore_mask(uint64_t exec, uint32_t c) {
  const __m256i bits = _mm256_set1_epi32(static_cast<int>(exec >> (8 * c)));
  return _mm256_sllv_epi32(bits, _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
}

__m256i load_unaligned(const void *p) {
  return _mm256_loadu_si256(static_cast<const __m256i *>(p));
}

} // namespace avx2
} // namespace vgpr_writeback
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_VGPR_WRITEBACK_AVX2_H_
