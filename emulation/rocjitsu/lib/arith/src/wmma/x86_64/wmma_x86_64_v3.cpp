// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 loose dense FP16/BF16 WMMA kernels for gfx11/gfx12 and both wave sizes.
// Reuses decoded factors across eight-column FMA accumulation, applying operand
// modifiers before final stores.

#include "common/float_formats.h"
#include "common/simd.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Compute loose FP16/BF16 A*B+C in the selected gfx layout and wave size.
// Modifiers must be validated; Modified=false requires zero instruction_flags.
template <bool Bf16, bool Modified, int Lanes, bool Rdna3>
void wmma(uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
          const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t a_sign = ((instruction_flags & RJ_ARITH_WMMA_NEG_LO_A) ? 0x8000U : 0) |
                          ((instruction_flags & RJ_ARITH_WMMA_NEG_HI_A) ? 0x80000000U : 0);
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
          __m256i signs =
              _mm256_set1_epi32(((instruction_flags & RJ_ARITH_WMMA_NEG_LO_B) ? 0x8000 : 0) |
                                ((instruction_flags & RJ_ARITH_WMMA_NEG_HI_B) ? INT32_MIN : 0));
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
    int source_reg = reg;
    // Read eight FP32 C values at a column offset.
    auto read_acc = [&](int offset) {
      auto words =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[source_reg] + lane + offset));

      return _mm256_castsi256_ps(words);
    };
    __m256 low = read_acc(0), high = read_acc(8);
    if constexpr (Modified) {
      __m256 keep = _mm256_castsi256_ps(
          _mm256_set1_epi32((instruction_flags & RJ_ARITH_WMMA_ABS_C) ? INT32_MAX : -1));
      __m256 sign = _mm256_castsi256_ps(
          _mm256_set1_epi32((instruction_flags & RJ_ARITH_WMMA_NEG_C) ? INT32_MIN : 0));
      low = _mm256_xor_ps(_mm256_and_ps(low, keep), sign);
      high = _mm256_xor_ps(_mm256_and_ps(high, keep), sign);
    }
    for (int step = 0; step < 16; ++step) {
      int k = step;
      uint32_t word =
          a[Rdna3 ? k / 2 : (k % KPerLane) / 2][row + 16 * (Rdna3 ? group : k / KPerLane)];
      if constexpr (Modified) {
        word ^= a_sign;
      }
      uint16_t bits = uint16_t(word >> (16 * (k % 2)));
      __m256 left = _mm256_set1_ps(Bf16 ? rocjitsu::arith::bf16_to_float(bits) : _cvtsh_ss(bits));
      low = _mm256_fmadd_ps(left, _mm256_load_ps(right[group][k]), low);
      high = _mm256_fmadd_ps(left, _mm256_load_ps(right[group][k] + 8), high);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[reg] + lane), _mm256_castps_si256(low));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[reg] + lane + 8),
                        _mm256_castps_si256(high));
  }

  // Stage all outputs before any stores, since D may overlap A, B or C.
  constexpr int DestinationRegs = 256 / Lanes;
  for (int reg = 0; reg < DestinationRegs; ++reg) {
    for (int lane = 0; lane < Lanes; lane += 8) {
      int source_reg = reg;
      auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[source_reg] + lane));

      _mm256_storeu_si256(reinterpret_cast<__m256i *>(d[reg] + lane), value);
    }
  }
}

} // namespace

namespace rocjitsu::arith {

// Compute loose FP16/BF16 A*B+C with FP32 output and validated instruction flags.
// D may overlap inputs; Lanes and Rdna3 select the physical register layout.
template <bool Bf16, int Lanes, bool Rdna3>
void wmma_f32_x86_64_v3(uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags) {
    wmma<Bf16, true, Lanes, Rdna3>(instruction_flags, d, a, b, c);
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

} // namespace rocjitsu::arith
