// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file lds_writeback.h
/// @brief A DS load's writeback, specialized by opcode format and wave size.
///
/// @details Generated execute_impl() code describes each plain load with
/// constants, so the destination format is known at compile time, and a wave
/// is 32 or 64 lanes. With both fixed, every load is the same loop over 8-lane
/// chunks with a constant trip count: form each destination register's 8
/// dwords (a load for b32, a transpose for b64/b96/b128, an extension for
/// 8/16-bit), then store them.
///
/// Measured in a standalone comparison against the per-lane writeback over the
/// DS-pattern census (~/rj-lds-patterns/microbench/writeback, Ryzen 9 9950X):
/// 29.9 -> 6.4 ns per writeback on the census mix, 27.2 -> 4.3 on runs16.

#ifndef ROCJITSU_VM_AMDGPU_LDS_WRITEBACK_H_
#define ROCJITSU_VM_AMDGPU_LDS_WRITEBACK_H_

#include "util/host_target.h"

#include <bit>
#include <cstdint>
#include <cstring>

#if RJ_HOST_HAS_AVX2
#include <immintrin.h>
#endif

namespace rocjitsu {
namespace amdgpu {
namespace lds_writeback {

/// @brief What a DS load's data becomes in its destination VGPRs.
struct DstFormat {
  uint8_t elem_size = 4;
  uint8_t num_elems = 1;
  bool sign_extend = false; ///< i8 / i16.
  bool d16_hi = false;      ///< Writes the upper 16 bits.
  bool d16_lo = false;      ///< Writes the lower 16 bits.
};

/// @brief Destination VGPRs a load of format @p F writes.
template <DstFormat F> constexpr uint32_t kRegs = (F.elem_size * F.num_elems + 3) / 4;

#if RJ_HOST_HAS_AVX2
template <DstFormat F, uint32_t Wave> struct Writeback {
  static constexpr uint32_t kStride = F.elem_size * F.num_elems;
  static constexpr bool kD16 = F.d16_hi || F.d16_lo;
  static constexpr bool kB32 = F.elem_size == 4 && F.num_elems == 1;
  static constexpr uint64_t kAll = Wave == 64 ? ~uint64_t{0} : (uint64_t{1} << Wave) - 1;

  static void run(uint8_t *const *regs, bool sram_ecc, uint64_t exec, const uint8_t *staging) {
    // A local copy: the caller's array may alias the stores below.
    uint8_t *dst[kRegs<F>];
    for (uint32_t r = 0; r < kRegs<F>; ++r)
      dst[r] = regs[r];
    const bool ecc = kD16 && sram_ecc;

    // b32 always uses masked stores: its EXEC is full in 54% of the census and
    // <= 4 lanes in 29%, so a branch on it mispredicts (mix 6.3 -> 7.9 ns with
    // the branch, though b32 full alone drops 3.8 -> 3.5). Other formats have a
    // full EXEC almost always, and masked stores there cost the mix 1.5 ns
    // (7.8 -> 6.3 with plain stores).
    if (!kB32 && exec == kAll)
      store_chunks<false>(dst, exec, ecc, staging);
    else
      store_chunks<true>(dst, exec, ecc, staging);
  }

  template <bool Masked>
  static void store_chunks(uint8_t *const *dst, uint64_t exec, bool ecc, const uint8_t *staging) {
    for (uint32_t c = 0; c < Wave / 8; ++c) {
      __m256i v[kRegs<F>];
      chunk(staging + c * 8 * kStride, dst[0] + c * 32, ecc, v);
      for (uint32_t r = 0; r < kRegs<F>; ++r) {
        auto *p = dst[r] + c * 32;
        if constexpr (Masked)
          _mm256_maskstore_epi32(reinterpret_cast<int *>(p), exec_mask(exec, c), v[r]);
        else
          _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), v[r]);
      }
    }
  }

  /// old: the destination's current dwords, which d16 loads partly keep.
  static void chunk(const uint8_t *src, const uint8_t *old, bool ecc, __m256i *v) {
    if constexpr (F.elem_size >= 4)
      dwords(src, v);
    else
      v[0] = narrow(src, old, ecc);
  }

  static void dwords(const uint8_t *src, __m256i *v) {
    if constexpr (kRegs<F> == 1) {
      // Staging is lane-major, which for b32 is the register layout.
      v[0] = load(src);
    } else if constexpr (kRegs<F> == 2) {
      const __m256 a = _mm256_castsi256_ps(load(src)), b = _mm256_castsi256_ps(load(src + 32));
      v[0] = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(a, b, 0x88)), 0xD8);
      v[1] = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(a, b, 0xDD)), 0xD8);
    } else if constexpr (kRegs<F> == 4) {
      // Lane j in the low half, lane j + 4 in the high half, so one 4x4
      // transpose per half gives lanes 0..7 in order with no cross-half shuffle.
      const auto *f = reinterpret_cast<const float *>(src);
      const __m256 x0 = _mm256_loadu2_m128(f + 16, f), x1 = _mm256_loadu2_m128(f + 20, f + 4);
      const __m256 x2 = _mm256_loadu2_m128(f + 24, f + 8), x3 = _mm256_loadu2_m128(f + 28, f + 12);
      const __m256 t0 = _mm256_unpacklo_ps(x0, x1), t1 = _mm256_unpacklo_ps(x2, x3);
      const __m256 t2 = _mm256_unpackhi_ps(x0, x1), t3 = _mm256_unpackhi_ps(x2, x3);
      v[0] = _mm256_castps_si256(_mm256_shuffle_ps(t0, t1, 0x44));
      v[1] = _mm256_castps_si256(_mm256_shuffle_ps(t0, t1, 0xEE));
      v[2] = _mm256_castps_si256(_mm256_shuffle_ps(t2, t3, 0x44));
      v[3] = _mm256_castps_si256(_mm256_shuffle_ps(t2, t3, 0xEE));
    } else {
      // b96 is 0.05% of the census: a gather instead of a third transpose.
      static_assert(kRegs<F> == 3);
      const __m256i lane = _mm256_setr_epi32(0, 12, 24, 36, 48, 60, 72, 84);
      for (uint32_t r = 0; r < 3; ++r)
        v[r] = _mm256_i32gather_epi32(reinterpret_cast<const int *>(src + 4 * r), lane, 1);
    }
  }

  static __m256i narrow(const uint8_t *src, const uint8_t *old, bool ecc) {
    static_assert(F.num_elems == 1 && (F.elem_size == 1 || F.elem_size == 2));
    __m256i v;
    if constexpr (F.elem_size == 1) {
      const __m128i x = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src));
      v = F.sign_extend ? _mm256_cvtepi8_epi32(x) : _mm256_cvtepu8_epi32(x);
    } else {
      const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
      v = F.sign_extend ? _mm256_cvtepi16_epi32(x) : _mm256_cvtepu16_epi32(x);
    }
    if constexpr (kD16) {
      // SRAM ECC targets write the whole dword, zeroing the other half.
      const __m256i keep = ecc ? _mm256_setzero_si256() : load(old);
      v = F.d16_hi ? _mm256_blend_epi16(keep, _mm256_slli_epi32(v, 16), 0xAA)
                   : _mm256_blend_epi16(keep, v, 0x55);
    }
    return v;
  }

  /// Lanes 8c..8c+7 of exec, each in its dword's sign bit (what maskstore reads).
  static __m256i exec_mask(uint64_t exec, uint32_t c) {
    const __m256i half = _mm256_set1_epi32(static_cast<int>(exec >> (32 * (c / 4))));
    const int s = 31 - 8 * static_cast<int>(c % 4);
    return _mm256_sllv_epi32(half,
                             _mm256_setr_epi32(s, s - 1, s - 2, s - 3, s - 4, s - 5, s - 6, s - 7));
  }

  static __m256i load(const void *p) { return _mm256_loadu_si256(static_cast<const __m256i *>(p)); }
};
#endif

/// @brief Per-lane writeback, for hosts without AVX2.
template <DstFormat F>
void write_back_lanes(uint8_t *const *dst, bool sram_ecc, uint64_t exec, const uint8_t *staging) {
  constexpr uint32_t stride = F.elem_size * F.num_elems;
  constexpr uint32_t copy_size = F.elem_size < 4 ? F.elem_size : 4;
  for (uint64_t lanes = exec; lanes; lanes &= lanes - 1) {
    const uint32_t lane = std::countr_zero(lanes);
    for (uint32_t r = 0; r < kRegs<F>; ++r) {
      uint32_t value = 0;
      std::memcpy(&value, &staging[lane * stride + r * 4], copy_size);
      if constexpr (F.sign_extend && F.elem_size == 1)
        value = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(value)));
      if constexpr (F.sign_extend && F.elem_size == 2)
        value = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(value)));
      uint8_t *at = dst[r] + lane * 4;
      if constexpr (F.d16_hi || F.d16_lo) {
        uint32_t old = 0;
        if (!sram_ecc)
          std::memcpy(&old, at, 4);
        value =
            F.d16_hi ? (old & 0xFFFFu) | (value << 16) : (old & 0xFFFF0000u) | (value & 0xFFFFu);
      }
      std::memcpy(at, &value, 4);
    }
  }
}

/// @brief Write a load's lane-major staging to its destination VGPRs for the
/// lanes in @p exec, leaving every other lane unchanged.
/// @details @p dst holds the raw lanes of the kRegs<F> destination registers.
/// Wavefront::exec() has no lanes past the wave, which makes the wave size a
/// safe template parameter.
template <DstFormat F>
void write_back_as(uint8_t *const *dst, bool sram_ecc, uint32_t wave, uint64_t exec,
                   const uint8_t *staging) {
#if RJ_HOST_HAS_AVX2
  if (wave == 32)
    Writeback<F, 32>::run(dst, sram_ecc, exec, staging);
  else
    Writeback<F, 64>::run(dst, sram_ecc, exec, staging);
#else
  (void)wave;
  write_back_lanes<F>(dst, sram_ecc, exec, staging);
#endif
}

} // namespace lds_writeback
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_LDS_WRITEBACK_H_
