// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 loose dense FP16/BF16 WMMA kernels for gfx11/gfx12 and both wave sizes.
// Reuses decoded factors across eight-column FMA accumulation, applying operand
// modifiers and stage-specific narrowing for packed results before final stores.

#include "common/float_formats.h"
#include "common/simd.h"
#include "common/x86_64/half_x86_64_v3.h"
#include "rj_arith/rj_arith.h"
#include "wmma/layout.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Narrow eight FP32 lanes into low halfwords: BF16 truncates, FP16 rounds
// to nearest-even and optionally saturates finite overflow. Preserve subnormals.
template <bool Bf16> __m256i packed_narrow(__m256 value, bool saturate) {
  if constexpr (Bf16) {
    return _mm256_srli_epi32(_mm256_castps_si256(value), 16);
  } else {
    return rocjitsu::arith::half_narrow(value, saturate);
  }
}

// Widen eight low-halfword FP16/BF16 encodings into FP32 lanes.
template <bool Bf16> __m256 packed_widen(__m256i value) {
  if constexpr (Bf16) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(value, 16));
  } else {
    return rocjitsu::arith::half_input(value, 0, 0);
  }
}

// Compute loose FP16/BF16 A*B+C in the selected gfx layout and wave size.
// Modifiers must be validated; Modified=false requires no modifiers. Packed
// results use staged narrowing and preserve gfx11's unselected D half.
template <bool Bf16, bool Modified, int Lanes, bool Rdna3, bool Packed = false>
void wmma(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
          const uint32_t *const *b, const uint32_t *const *c, bool saturate = false) {
  const uint32_t a_sign = ((modifiers & RJ_ARITH_WMMA_NEG_LO_A) ? 0x8000U : 0) |
                          ((modifiers & RJ_ARITH_WMMA_NEG_HI_A) ? 0x80000000U : 0);
  // Decode B once for all output rows. Each input word holds two consecutive
  // K elements; columns occupy consecutive lanes within each half-wave.
  constexpr int Groups = Rdna3 ? Lanes / 16 : 1;
  constexpr int KPerLane = 256 / Lanes;
  alignas(32) float right[Groups][16][16];
  for (int group = 0; group < Groups; ++group) {
    for (int k = 0; k < 16; ++k) {
      for (int col = 0; col < 16; col += 8) {
        __m256i words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
            b[Rdna3 ? k / 2 : (k % KPerLane) / 2] + 16 * (Rdna3 ? group : k / KPerLane) + col));
        if constexpr (Modified) {
          __m256i signs = _mm256_set1_epi32(((modifiers & RJ_ARITH_WMMA_NEG_LO_B) ? 0x8000 : 0) |
                                            ((modifiers & RJ_ARITH_WMMA_NEG_HI_B) ? INT32_MIN : 0));
          words = _mm256_xor_si256(words, signs);
        }
        if constexpr (Bf16) {
          // BF16 widens exactly by placing its bits in FP32's high half.
          words = k % 2 ? _mm256_and_si256(words, _mm256_set1_epi32(-65536))
                        : _mm256_slli_epi32(words, 16);
          _mm256_store_ps(right[group][k] + col, _mm256_castsi256_ps(words));
        } else {
          words = k % 2 ? _mm256_srli_epi32(words, 16)
                        : _mm256_and_si256(words, _mm256_set1_epi32(65535));
          __m128i halves =
              _mm_packus_epi32(_mm256_castsi256_si128(words), _mm256_extracti128_si256(words, 1));
          _mm256_store_ps(right[group][k] + col, _mm256_cvtph_ps(halves));
        }
      }
    }
  }

  uint32_t result[256 / Lanes][Lanes];
  for (int row = 0; row < 16; ++row) {
    // GFX11 interleaves rows across replicated groups. GFX12 blocks rows,
    // with Wave64's upper half holding the second four-row block.
    int group = Rdna3 ? row % Groups : 0;
    int reg = Rdna3 ? row / Groups : row % KPerLane;
    int lane = Rdna3 ? 16 * group : 16 * (row / 8) + (Lanes == 64 ? 32 * ((row / 4) % 2) : 0);
    int source_reg = Packed && !Rdna3 ? reg / 2 : reg;
    // Read eight C values at a column offset, widening the selected half if packed.
    auto read_acc = [&](int offset) {
      auto words =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[source_reg] + lane + offset));
      if constexpr (Packed) {
        int shift = Rdna3 ? (modifiers & RJ_ARITH_WMMA_HIGH_C_D ? 16 : 0) : 16 * (reg % 2);
        words = _mm256_and_si256(_mm256_srl_epi32(words, _mm_cvtsi32_si128(shift)),
                                 _mm256_set1_epi32(65535));
        return packed_widen<Bf16>(words);
      } else {
        return _mm256_castsi256_ps(words);
      }
    };
    __m256 low = read_acc(0), high = read_acc(8);
    if constexpr (Modified) {
      __m256 keep = _mm256_castsi256_ps(
          _mm256_set1_epi32((modifiers & RJ_ARITH_WMMA_ABS_C) ? INT32_MAX : -1));
      __m256 sign =
          _mm256_castsi256_ps(_mm256_set1_epi32((modifiers & RJ_ARITH_WMMA_NEG_C) ? INT32_MIN : 0));
      low = _mm256_xor_ps(_mm256_and_ps(low, keep), sign);
      high = _mm256_xor_ps(_mm256_and_ps(high, keep), sign);
    }
    for (int step = 0; step < 16; ++step) {
      int k = Packed && !Rdna3 && Lanes == 32 ? rocjitsu::arith::wmma_gfx12_w32_k(step) : step;
      uint32_t word =
          a[Rdna3 ? k / 2 : (k % KPerLane) / 2][row + 16 * (Rdna3 ? group : k / KPerLane)];
      if constexpr (Modified) {
        word ^= a_sign;
      }
      uint16_t bits = uint16_t(word >> (16 * (k % 2)));
      __m256 left = _mm256_set1_ps(Bf16 ? rocjitsu::arith::bf16_to_float(bits) : _cvtsh_ss(bits));
      low = _mm256_fmadd_ps(left, _mm256_load_ps(right[group][k]), low);
      high = _mm256_fmadd_ps(left, _mm256_load_ps(right[group][k] + 8), high);
      // gfx11 narrows every DOT2; gfx12 every DOT4, so the order matters.
      if constexpr (Packed) {
        if ((step + 1) % (Rdna3 ? 2 : 4) == 0) {
          low = packed_widen<Bf16>(packed_narrow<Bf16>(low, saturate));
          high = packed_widen<Bf16>(packed_narrow<Bf16>(high, saturate));
        }
      }
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[reg] + lane),
                        Packed ? packed_narrow<Bf16>(low, saturate) : _mm256_castps_si256(low));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[reg] + lane + 8),
                        Packed ? packed_narrow<Bf16>(high, saturate) : _mm256_castps_si256(high));
  }

  // Stage all outputs before any stores, since D may overlap A, B or C.
  constexpr int DestinationRegs = 256 / Lanes / (Packed && !Rdna3 ? 2 : 1);
  for (int reg = 0; reg < DestinationRegs; ++reg) {
    for (int lane = 0; lane < Lanes; lane += 8) {
      int source_reg = Packed && !Rdna3 ? reg * 2 : reg;
      auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[source_reg] + lane));
      // gfx11 replaces one half of each D word; gfx12 combines two output rows.
      if constexpr (Packed && Rdna3) {
        int shift = modifiers & RJ_ARITH_WMMA_HIGH_C_D ? 16 : 0;
        auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d[reg] + lane));
        value = _mm256_or_si256(_mm256_sll_epi32(value, _mm_cvtsi32_si128(shift)),
                                _mm256_and_si256(old, _mm256_set1_epi32(shift ? 65535 : -65536)));
      } else if constexpr (Packed) {
        auto high =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[source_reg + 1] + lane));
        value = _mm256_or_si256(value, _mm256_slli_epi32(high, 16));
      }
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(d[reg] + lane), value);
    }
  }
}

} // namespace

namespace rocjitsu::arith {

// Compute loose FP16/BF16 A*B+C with FP32 output and validated modifiers.
// D may overlap inputs; Lanes and Rdna3 select the physical register layout.
template <bool Bf16, int Lanes, bool Rdna3>
void wmma_f32_x86_64_v3(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c) {
  if (modifiers) {
    wmma<Bf16, true, Lanes, Rdna3>(modifiers, d, a, b, c);
  } else {
    wmma<Bf16, false, Lanes, Rdna3>(0, d, a, b, c);
  }
}

template void wmma_f32_x86_64_v3<false, 32, false>(uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void wmma_f32_x86_64_v3<false, 32, true>(uint32_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void wmma_f32_x86_64_v3<false, 64, false>(uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void wmma_f32_x86_64_v3<false, 64, true>(uint32_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void wmma_f32_x86_64_v3<true, 32, false>(uint32_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void wmma_f32_x86_64_v3<true, 32, true>(uint32_t, uint32_t *const *,
                                                 const uint32_t *const *, const uint32_t *const *,
                                                 const uint32_t *const *);
template void wmma_f32_x86_64_v3<true, 64, false>(uint32_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void wmma_f32_x86_64_v3<true, 64, true>(uint32_t, uint32_t *const *,
                                                 const uint32_t *const *, const uint32_t *const *,
                                                 const uint32_t *const *);

// Compute loose WMMA with packed FP16/BF16 C/D and validated modifiers.
// D may overlap inputs; preserve the unused gfx11 half. saturate controls
// finite FP16 overflow at each narrowing stage.
template <bool Bf16, int Lanes, bool Rdna3>
void wmma_packed_x86_64_v3(uint32_t modifiers, bool saturate, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  wmma<Bf16, true, Lanes, Rdna3, true>(modifiers, d, a, b, c, saturate);
}

template void wmma_packed_x86_64_v3<false, 32, false>(uint32_t, bool, uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *);
template void wmma_packed_x86_64_v3<false, 32, true>(uint32_t, bool, uint32_t *const *,
                                                     const uint32_t *const *,
                                                     const uint32_t *const *,
                                                     const uint32_t *const *);
template void wmma_packed_x86_64_v3<false, 64, false>(uint32_t, bool, uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *);
template void wmma_packed_x86_64_v3<false, 64, true>(uint32_t, bool, uint32_t *const *,
                                                     const uint32_t *const *,
                                                     const uint32_t *const *,
                                                     const uint32_t *const *);
template void wmma_packed_x86_64_v3<true, 32, false>(uint32_t, bool, uint32_t *const *,
                                                     const uint32_t *const *,
                                                     const uint32_t *const *,
                                                     const uint32_t *const *);
template void wmma_packed_x86_64_v3<true, 32, true>(uint32_t, bool, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template void wmma_packed_x86_64_v3<true, 64, false>(uint32_t, bool, uint32_t *const *,
                                                     const uint32_t *const *,
                                                     const uint32_t *const *,
                                                     const uint32_t *const *);
template void wmma_packed_x86_64_v3<true, 64, true>(uint32_t, bool, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);

} // namespace rocjitsu::arith
