// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Stage gfx1250 accumulators and pack adjacent output rows in VGPR lane order.
#pragma once
#if defined(__AVX512F__)
#include "common/x86_64/half_x86_64_v4.h"
#else
#include "common/x86_64/half_x86_64_v3.h"
#endif
#include "wmma/gfx1250_half.h"
#include <cstring>
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
#if defined(__AVX512F__)
using Gfx1250PackedWords [[gnu::vector_size(64)]] = uint32_t;
using Gfx1250PackedFloat = __m512;
#else
using Gfx1250PackedWords [[gnu::vector_size(32)]] = uint32_t;
using Gfx1250PackedFloat = __m256;
#endif
template <Wmma1250Form Form>
inline void gfx1250_load_accumulators(const uint32_t *const *c, uint64_t controls,
                                      float tile[16][16]) {
  using Words = Gfx1250PackedWords;
  constexpr int width = sizeof(Words) / sizeof(uint32_t);
  constexpr bool packed = Form == Wmma1250Form::F16 || Form == Wmma1250Form::Bf16;
  if constexpr (packed) {
    if (controls & RJ_ARITH_WMMA_INLINE_C) {
      const auto value = gfx1250_accumulator(c[0][0], controls);
      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
          tile[row][col] = value;
        }
      }
      return;
    }
  }
  const uint32_t keep = (controls & RJ_ARITH_WMMA_ABS_C) ? 0x7fffffffu : 0xffffffffu;
  const uint32_t negate = (controls & RJ_ARITH_WMMA_NEG_C) ? 0x80000000u : 0;
  for (int row = 0; row < 16; ++row) {
    const int reg = (row % 8) / (packed ? 2 : 1);
    const int group = row / 8;
    for (int col = 0; col < 16; col += width) {
      Words bits;
      std::memcpy(&bits, c[reg] + 16 * group + col, sizeof(bits));
      if constexpr (packed) {
        const int shift = (row % 2) * 16;
        Words h = (bits >> shift) & 0xffffu;
        if constexpr (Form == Wmma1250Form::F16) {
#ifdef __AVX512F__
          auto result = half_input(bit_cast<__m512i>(bits), shift, 0);
#else
          auto result = half_input(bit_cast<__m256i>(bits), shift, 0);
#endif
          bits = bit_cast<Words>(result);
          // The C encoding retains the signaling bit until accumulation.
          bits &= ~(bit_cast<Words>((h & 0x7e00u) == 0x7c00u) & 0x00400000u);
        } else {
          bits = h << 16;
        }
      }
      bits = (bits & keep) ^ negate;
      std::memcpy(tile[row] + col, &bits, sizeof(bits));
    }
  }
}

template <Wmma1250Form Form, bool Sparse = false>
inline Gfx1250PackedWords gfx1250_narrow_result(Gfx1250PackedFloat values, bool compatibility,
                                                bool saturate) {
  using Words = Gfx1250PackedWords;
  Words bits = bit_cast<Words>(values);
  Words upper = bits >> 16;
  if constexpr (Form == Wmma1250Form::F16) {
    Words value = bit_cast<Words>(half_narrow(values, false));
    // F16C quiets signaling inputs. The WMMA narrowing rule retains their
    // top payload bits and keeps at least one fraction bit set.
    Words signaling =
        bit_cast<Words>(((bits & 0x7fffffffu) > 0x7f800000u) & ((bits & 0x00400000u) == 0u));
    value &= ~(signaling & 0x0200u);
    value |= signaling & bit_cast<Words>((value & 0x03ffu) == 0u) & 1u;
    if (saturate) {
      value = ((value & 0x7fffu) == 0x7c00u) ? value - 1u : value;
    }
    return value;
  } else {
    if (compatibility) {
      return upper;
    }
    Words rounded = (bits + 0x7fffu + (upper & 1u)) >> 16;
    Words value;
    if constexpr (Sparse) {
      // Sparse BF16 retains a marker if narrowing discards any NaN payload bit.
      Words special = upper | (bit_cast<Words>((bits & 0xffffu) != 0u) & 1u);
      value = ((bits & 0x7f800000u) == 0x7f800000u) ? special : rounded;
    } else {
      value = ((bits & 0x7fffffffu) > 0x7f800000u) ? (upper | 0x40u) : rounded;
    }
    if (saturate) {
      value = ((value & 0x7fffu) == 0x7f80u) ? value - 1u : value;
    }
    return value;
  }
}

template <Wmma1250Form Form, bool Sparse = false>
inline void gfx1250_store_results(uint64_t flags, uint32_t *const *d, const float result[16][16],
                                  bool saturate_f16 = false) {
  if constexpr (Form == Wmma1250Form::F32F16 || Form == Wmma1250Form::F32Bf16) {
    gfx1250_write_result<Form>(flags, d, result, saturate_f16);
  } else {
    using Words = Gfx1250PackedWords;
    constexpr int width = sizeof(Words) / sizeof(uint32_t);
    const bool compatibility =
        !Sparse && (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
    const bool saturate = (Form == Wmma1250Form::F16 && saturate_f16) ||
                          (!compatibility && (flags & RJ_ARITH_FP16_OVFL));
    for (int reg = 0; reg < 4; ++reg) {
      for (int group = 0; group < 2; ++group) {
        int row = 2 * reg + 8 * group;
        for (int col = 0; col < 16; col += width) {
          Gfx1250PackedFloat lo, hi;
          std::memcpy(&lo, result[row] + col, sizeof(lo));
          std::memcpy(&hi, result[row + 1] + col, sizeof(hi));
          auto words = gfx1250_narrow_result<Form, Sparse>(lo, compatibility, saturate) |
                       (gfx1250_narrow_result<Form, Sparse>(hi, compatibility, saturate) << 16);
          std::memcpy(d[reg] + 16 * group + col, &words, sizeof(words));
        }
      }
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
