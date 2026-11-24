// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 loose FP16/BF16 WMMA. Four rows share each decoded B vector;
// packed outputs narrow after each gfx11 DOT2 or gfx12 DOT4.

#include "common/compiler.h"
#include "common/simd.h"
#include "common/x86_64/half_x86_64_v3.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/layout.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

template <bool Bf16> __m256 widen(__m256i words, int shift) {
  if constexpr (Bf16) {
    return _mm256_castsi256_ps(shift ? _mm256_and_si256(words, _mm256_set1_epi32(0xffff0000U))
                                     : _mm256_slli_epi32(words, 16));
  } else {
    return half_input(words, shift, 0);
  }
}

template <bool Bf16, bool Saturate> __m256i narrow(__m256 x) {
  if constexpr (Bf16) {
    return _mm256_srli_epi32(_mm256_castps_si256(x), 16);
  } else {
    return half_narrow(x, Saturate);
  }
}

template <bool Bf16, bool Saturate> __m256 round_stage(__m256 x) {
  if constexpr (Bf16) {
    return _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0xffff0000U)));
  } else if constexpr (Saturate) {
    return widen<false>(narrow<false, true>(x), 0);
  } else {
    return _mm256_cvtph_ps(_mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
}

// Decode a K pair from each packed word in one pass. Keep both tiles in
// accumulation order, including the gfx12 Wave32 packed-output permutation.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
void decode(float (&tile)[16][16], const uint32_t *const *operand, __m256i signs) {
  for (int step = 0; step < 16; step += 2) {
    int k = Packed && !Rdna3 && Lanes == 32 ? wmma_gfx12_w32_k(step) : step;
    auto src = wmma_half_input<Lanes, Rdna3>(k);
    for (int col = 0; col < 16; col += 8) {
      auto words =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(operand[src.reg] + src.lane + col));
      words = _mm256_xor_si256(words, signs);
      _mm256_store_ps(tile[step] + col, widen<Bf16>(words, 0));
      _mm256_store_ps(tile[step + 1] + col, widen<Bf16>(words, 16));
    }
  }
}

template <bool Bf16, int Lanes, bool Rdna3, bool Packed, bool Saturate>
void compute(uint32_t mods, uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
             const uint32_t *const *c) {
  constexpr int Rows = 4;
  constexpr int Registers = 256 / Lanes;
  alignas(32) float left[16][16], right[16][16];
  auto signs = [&](uint32_t lo, uint32_t hi) {
    return _mm256_set1_epi32((mods & lo ? 0x8000 : 0) | (mods & hi ? INT32_MIN : 0));
  };
  decode<Bf16, Lanes, Rdna3, Packed>(left, a,
                                     signs(RJ_ARITH_WMMA_NEG_LO_A, RJ_ARITH_WMMA_NEG_HI_A));
  decode<Bf16, Lanes, Rdna3, Packed>(right, b,
                                     signs(RJ_ARITH_WMMA_NEG_LO_B, RJ_ARITH_WMMA_NEG_HI_B));
  auto keep = _mm256_castsi256_ps(_mm256_set1_epi32(mods & RJ_ARITH_WMMA_ABS_C ? INT32_MAX : -1));
  auto sign = _mm256_castsi256_ps(_mm256_set1_epi32(mods & RJ_ARITH_WMMA_NEG_C ? INT32_MIN : 0));
  bool high_half = mods & RJ_ARITH_WMMA_HIGH_C_D;
  alignas(32) uint32_t result[Registers][Lanes];
  for (int base = 0; base < 16; base += Rows) {
    __m256 low[Rows], high[Rows];
    for (int r = 0; r < Rows; ++r) {
      auto out = wmma_half_output<Lanes, Rdna3>(base + r);
      auto read = [&](int col) {
        int src = Packed && !Rdna3 ? out.reg / 2 : out.reg;
        auto bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[src] + out.lane + col));
        auto value = _mm256_castsi256_ps(bits);
        if constexpr (Packed) {
          value = widen<Bf16>(bits, 16 * (Rdna3 ? int(high_half) : out.reg % 2));
        }
        return _mm256_xor_ps(_mm256_and_ps(value, keep), sign);
      };
      low[r] = read(0);
      high[r] = read(8);
    }
    RJ_ARITH_UNROLL(4)
    for (int step = 0; step < 16; ++step) {
      auto b_low = _mm256_load_ps(right[step]);
      auto b_high = _mm256_load_ps(right[step] + 8);
      RJ_ARITH_UNROLL(4)
      for (int r = 0; r < Rows; ++r) {
        auto x = _mm256_set1_ps(left[step][base + r]);
        low[r] = _mm256_fmadd_ps(x, b_low, low[r]);
        high[r] = _mm256_fmadd_ps(x, b_high, high[r]);
        if constexpr (Packed) {
          if (step < 15 && (step + 1) % (Rdna3 ? 2 : 4) == 0) {
            low[r] = round_stage<Bf16, Saturate>(low[r]);
            high[r] = round_stage<Bf16, Saturate>(high[r]);
          }
        }
      }
    }
    for (int r = 0; r < Rows; ++r) {
      auto out = wmma_half_output<Lanes, Rdna3>(base + r);
      auto store = [&](int col, __m256 x) {
        auto bits = _mm256_castps_si256(x);
        if constexpr (Packed) {
          bits = narrow<Bf16, Saturate>(x);
        }
        _mm256_store_si256(reinterpret_cast<__m256i *>(result[out.reg] + out.lane + col), bits);
      };
      store(0, low[r]);
      store(8, high[r]);
    }
  }
  // All input reads precede the first D store. Write repeated D registers in order.
  constexpr int DestinationRegs = Registers / (Packed && !Rdna3 ? 2 : 1);
  for (int reg = 0; reg < DestinationRegs; ++reg) {
    for (int lane = 0; lane < Lanes; lane += 8) {
      int src = Packed && !Rdna3 ? reg * 2 : reg;
      auto value = _mm256_load_si256(reinterpret_cast<const __m256i *>(result[src] + lane));
      if constexpr (Packed && Rdna3) {
        auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d[reg] + lane));
        value = high_half ? _mm256_blend_epi16(old, _mm256_slli_epi32(value, 16), 0xaa)
                          : _mm256_blend_epi16(old, value, 0x55);
      } else if constexpr (Packed) {
        auto upper = _mm256_load_si256(reinterpret_cast<const __m256i *>(result[src + 1] + lane));
        value = _mm256_or_si256(value, _mm256_slli_epi32(upper, 16));
      }
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(d[reg] + lane), value);
    }
  }
}
} // namespace

// Compute loose FP16/BF16 A*B+C with FP32 output and validated instruction flags.
// D may overlap inputs; Lanes and Rdna3 select the physical register layout.
template <bool Bf16, int Lanes, bool Rdna3>
void wmma_f32_x86_64_v3(uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c) {
  compute<Bf16, Lanes, Rdna3, false, false>(instruction_flags, d, a, b, c);
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

// Compute loose WMMA with packed FP16/BF16 C/D and validated instruction flags.
// D may overlap inputs; preserve the unused gfx11 half. saturate controls
// finite FP16 overflow at each narrowing stage.
template <bool Bf16, int Lanes, bool Rdna3>
void wmma_packed_x86_64_v3(uint32_t instruction_flags, bool saturate, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  if constexpr (!Bf16) {
    if (saturate) {
      compute<Bf16, Lanes, Rdna3, true, true>(instruction_flags, d, a, b, c);
      return;
    }
  }
  compute<Bf16, Lanes, Rdna3, true, false>(instruction_flags, d, a, b, c);
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
