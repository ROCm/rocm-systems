// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 loose FP16/BF16 WMMA. Four rows share each decoded B vector;
// packed outputs narrow only at the final store.

#include "common/compiler.h"
#include "common/simd.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/layout.h"
#include "wmma/x86_64/wmma_loose_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Decode a K pair from each packed word in one pass. Keep both tiles in
// accumulation order, including the gfx12 Wave32 packed-output permutation.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
[[gnu::always_inline]] inline void decode(float (&tile)[16][16], const uint32_t *const *operand,
                                          __m256i signs) {
  for (int step = 0; step < 16; step += 2) {
    const int k = Packed && !Rdna3 && Lanes == 32 ? wmma_gfx12_w32_k(step) : step;
    const WmmaRegisterSection src = wmma_half_input<Lanes, Rdna3>(k);
    const uint32_t *ptr = operand[src.reg] + src.lane;
    for (int col = 0; col < 16; col += 8) {
      const __m256i words =
          _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr + col)), signs);
      __m256 even, odd;
      wmma_widen_half_pair_v3<Bf16>(words, even, odd);
      _mm256_store_ps(tile[step] + col, even);
      _mm256_store_ps(tile[step + 1] + col, odd);
    }
  }
}

// Return true when direct stores to d inside the 4-row tile loop are guaranteed
// equivalent to buffered ascending stores (all d[i] pairwise disjoint, and
// either exact in-place d[i] == c[i] or disjoint from all c[j]).
template <int Regs, int Lanes>
[[gnu::always_inline]] inline bool direct_store_safe(uint32_t *const *d, const uint32_t *const *c) {
  constexpr uintptr_t kBytes = uintptr_t(Lanes) * sizeof(uint32_t);
  const uintptr_t d0 = reinterpret_cast<uintptr_t>(d[0]);
  uintptr_t prev_d = d0;
  uintptr_t min_c = reinterpret_cast<uintptr_t>(c[0]);
  uintptr_t max_c = min_c;
  uintptr_t same_diff = d0 ^ min_c;
  for (int i = 1; i < Regs; ++i) {
    const uintptr_t di = reinterpret_cast<uintptr_t>(d[i]);
    if (di < prev_d || (di - prev_d) < kBytes) {
      return false;
    }
    prev_d = di;
    const uintptr_t ci = reinterpret_cast<uintptr_t>(c[i]);
    same_diff |= (di ^ ci);
    if (ci < min_c) {
      min_c = ci;
    }
    if (ci > max_c) {
      max_c = ci;
    }
  }
  if (same_diff == 0) {
    return true;
  }
  const bool c_before_d = (max_c < d0) && ((d0 - max_c) >= kBytes);
  const bool d_before_c = (prev_d < min_c) && ((min_c - prev_d) >= kBytes);
  return c_before_d || d_before_c;
}

template <bool Bf16, int Lanes, bool Rdna3, bool Packed, bool Saturate, bool DirectStore>
[[gnu::noinline]] void compute_impl(uint32_t mods, uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  constexpr int Rows = 4;
  constexpr int Registers = 256 / Lanes;
  constexpr int DestinationRegs = Registers / (Packed && !Rdna3 ? 2 : 1);
  alignas(32) float left[16][16], right[16][16];
  const __m256i a_signs = _mm256_set1_epi32((mods & RJ_ARITH_WMMA_NEG_LO_A ? 0x8000 : 0) |
                                            (mods & RJ_ARITH_WMMA_NEG_HI_A ? INT32_MIN : 0));
  const __m256i b_signs = _mm256_set1_epi32((mods & RJ_ARITH_WMMA_NEG_LO_B ? 0x8000 : 0) |
                                            (mods & RJ_ARITH_WMMA_NEG_HI_B ? INT32_MIN : 0));
  decode<Bf16, Lanes, Rdna3, Packed>(left, a, a_signs);
  decode<Bf16, Lanes, Rdna3, Packed>(right, b, b_signs);

  const __m256 keep =
      _mm256_castsi256_ps(_mm256_set1_epi32(mods & RJ_ARITH_WMMA_ABS_C ? INT32_MAX : -1));
  const __m256 sign =
      _mm256_castsi256_ps(_mm256_set1_epi32(mods & RJ_ARITH_WMMA_NEG_C ? INT32_MIN : 0));
  const bool high_half = (mods & RJ_ARITH_WMMA_HIGH_C_D) != 0;
  auto apply_c_mod = [&](__m256 v) { return _mm256_xor_ps(_mm256_and_ps(v, keep), sign); };

  alignas(32) uint32_t result[DirectStore ? 1 : DestinationRegs][DirectStore ? 1 : Lanes];

#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
  for (int base = 0; base < 16; base += Rows) {
    __m256 low[Rows], high[Rows];
    if constexpr (Packed && !Rdna3) {
      // gfx12 packed C holds adjacent rows (base+r, base+r+1) in one word.
      for (int r = 0; r < Rows; r += 2) {
        const WmmaRegisterSection out = wmma_half_output<Lanes, false>(base + r);
        const uint32_t *c_ptr = c[out.reg / 2] + out.lane;
        const __m256i lo_words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c_ptr));
        const __m256i hi_words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c_ptr + 8));
        __m256 e0, o0, e1, o1;
        wmma_widen_half_pair_v3<Bf16>(lo_words, e0, o0);
        wmma_widen_half_pair_v3<Bf16>(hi_words, e1, o1);
        low[r] = apply_c_mod(e0);
        low[r + 1] = apply_c_mod(o0);
        high[r] = apply_c_mod(e1);
        high[r + 1] = apply_c_mod(o1);
      }
    } else {
      for (int r = 0; r < Rows; ++r) {
        const WmmaRegisterSection out = wmma_half_output<Lanes, Rdna3>(base + r);
        const uint32_t *c_ptr = c[out.reg] + out.lane;
        const __m256i lo_bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c_ptr));
        const __m256i hi_bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c_ptr + 8));
        if constexpr (Packed) {
          low[r] = apply_c_mod(wmma_widen_half_v3<Bf16>(lo_bits, high_half));
          high[r] = apply_c_mod(wmma_widen_half_v3<Bf16>(hi_bits, high_half));
        } else {
          low[r] = apply_c_mod(_mm256_castsi256_ps(lo_bits));
          high[r] = apply_c_mod(_mm256_castsi256_ps(hi_bits));
        }
      }
    }

    RJ_ARITH_UNROLL(4)
    for (int step = 0; step < 16; ++step) {
      const __m256 b_low = _mm256_load_ps(right[step]);
      const __m256 b_high = _mm256_load_ps(right[step] + 8);
      RJ_ARITH_UNROLL(4)
      for (int r = 0; r < Rows; ++r) {
        const __m256 x = _mm256_broadcast_ss(&left[step][base + r]);
        low[r] = _mm256_fmadd_ps(x, b_low, low[r]);
        high[r] = _mm256_fmadd_ps(x, b_high, high[r]);
      }
    }

    if constexpr (Packed && !Rdna3) {
      for (int r = 0; r < Rows; r += 2) {
        const WmmaRegisterSection out = wmma_half_output<Lanes, false>(base + r);
        uint32_t *dst =
            DirectStore ? (d[out.reg / 2] + out.lane) : (result[out.reg / 2] + out.lane);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst),
                            wmma_pack_half_pair_v3<Bf16, Saturate>(low[r], low[r + 1]));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + 8),
                            wmma_pack_half_pair_v3<Bf16, Saturate>(high[r], high[r + 1]));
      }
    } else if constexpr (Packed && Rdna3) {
      auto encode = [&](__m256 x) {
        if constexpr (Bf16) {
          const __m256i bits = _mm256_castps_si256(x);
          return high_half ? bits : _mm256_srli_epi32(bits, 16);
        } else {
          const __m256i narrowed = _mm256_cvtepu16_epi32(wmma_narrow_f16_v3<Saturate>(x));
          return high_half ? _mm256_slli_epi32(narrowed, 16) : narrowed;
        }
      };
      for (int r = 0; r < Rows; ++r) {
        const WmmaRegisterSection out = wmma_half_output<Lanes, true>(base + r);
        if constexpr (DirectStore) {
          uint32_t *dst = d[out.reg] + out.lane;
          const __m256i old0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst));
          const __m256i old1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst + 8));
          const __m256i v0 = encode(low[r]);
          const __m256i v1 = encode(high[r]);
          _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst),
                              high_half ? _mm256_blend_epi16(old0, v0, 0xaa)
                                        : _mm256_blend_epi16(old0, v0, 0x55));
          _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + 8),
                              high_half ? _mm256_blend_epi16(old1, v1, 0xaa)
                                        : _mm256_blend_epi16(old1, v1, 0x55));
        } else {
          uint32_t *dst = result[out.reg] + out.lane;
          _mm256_store_si256(reinterpret_cast<__m256i *>(dst), encode(low[r]));
          _mm256_store_si256(reinterpret_cast<__m256i *>(dst + 8), encode(high[r]));
        }
      }
    } else {
      for (int r = 0; r < Rows; ++r) {
        const WmmaRegisterSection out = wmma_half_output<Lanes, Rdna3>(base + r);
        uint32_t *dst = DirectStore ? (d[out.reg] + out.lane) : (result[out.reg] + out.lane);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), _mm256_castps_si256(low[r]));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + 8), _mm256_castps_si256(high[r]));
      }
    }
  }

  if constexpr (!DirectStore) {
    // All input reads precede the first D store. Write repeated D registers in order.
    for (int reg = 0; reg < DestinationRegs; ++reg) {
      for (int lane = 0; lane < Lanes; lane += 8) {
        __m256i value = _mm256_load_si256(reinterpret_cast<const __m256i *>(result[reg] + lane));
        if constexpr (Packed && Rdna3) {
          const __m256i old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d[reg] + lane));
          value = high_half ? _mm256_blend_epi16(old, value, 0xaa)
                            : _mm256_blend_epi16(old, value, 0x55);
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(d[reg] + lane), value);
      }
    }
  }
}

template <bool Bf16, int Lanes, bool Rdna3, bool Packed, bool Saturate>
[[gnu::always_inline]] inline void compute(uint32_t mods, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *c) {
  constexpr int DestinationRegs = (256 / Lanes) / (Packed && !Rdna3 ? 2 : 1);
  if (direct_store_safe<DestinationRegs, Lanes>(d, c)) {
    compute_impl<Bf16, Lanes, Rdna3, Packed, Saturate, true>(mods, d, a, b, c);
  } else {
    compute_impl<Bf16, Lanes, Rdna3, Packed, Saturate, false>(mods, d, a, b, c);
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
// finite FP16 overflow at the final store.
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
