// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// V4 entry points for gfx1201 sparse floating-point WMMA.
#include "common/x86_64/fp8.h"
#include "wmma/swmmac_float.h"
#include "wmma/x86_64/swmmac_loose_x86_64.h"

namespace rocjitsu::arith {
template <bool Bf16, int Wave, bool Packed>
void swmmac16_x86_64_v4(uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *index, bool saturate) {
  sparse_x86::compute_float<16, Bf16, Bf16, Wave, Packed>(mode, d, a, b, index, saturate);
}

template Swmmac16Backend swmmac16_x86_64_v4<false, 32, false>;
template Swmmac16Backend swmmac16_x86_64_v4<false, 32, true>;
template Swmmac16Backend swmmac16_x86_64_v4<false, 64, false>;
template Swmmac16Backend swmmac16_x86_64_v4<false, 64, true>;
template Swmmac16Backend swmmac16_x86_64_v4<true, 32, false>;
template Swmmac16Backend swmmac16_x86_64_v4<true, 32, true>;
template Swmmac16Backend swmmac16_x86_64_v4<true, 64, false>;
template Swmmac16Backend swmmac16_x86_64_v4<true, 64, true>;

template <bool BfA, bool BfB, int Wave>
void swmmac8_x86_64_v4(uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
                       const uint32_t *const *b, const uint32_t *const *index) {
  using namespace sparse_x86;
  alignas(64) float left[16][16];
  alignas(64) uint32_t selected[16][16];
  F32 right[32];
  const int key = SwmmacIndex<8, 32, Wave>::shift(mode);
  for (int ck = 0; ck < 16; ++ck) {
    const SparseField field = swmmac_a_field<8, Wave>(0, ck);
    const int slot = swmmac_index_slot<8, Wave>(ck);
    __m512 v = widen_fp8<BfA>(bit_cast<__m512i>(load(a[field.reg] + field.lane) >> field.shift));
    std::memcpy(left[ck], &v, sizeof(v));
    U32 k = 4 * (ck / 2) + ((load(index[0] + field.lane) >> (key + 2 * slot)) & 3u);
    std::memcpy(selected[ck], &k, sizeof(k));
  }
  for (int k = 0; k < 32; ++k) {
    const SparseField field = swmmac_b_field<8, Wave>(0, k);
    right[k] = widen_fp8<BfB>(bit_cast<__m512i>(load(b[field.reg] + field.lane) >> field.shift));
  }

  U32 result[16];
  constexpr int Rows = 4;
  for (int tile = 0; tile < 16; tile += Rows) {
    F32 acc[Rows];
    RJ_ARITH_UNROLL(8)
    for (int r = 0; r < Rows; ++r) {
      const int row = tile + r;
      acc[r] = bit_cast<F32>(load(d[row % 8] + 16 * (row / 8)));
    }
    for (int ck = 0; ck < 16; ++ck) {
      RJ_ARITH_UNROLL(8)
      for (int r = 0; r < Rows; ++r) {
        const F32 a = _mm512_set1_ps(left[ck][tile + r]);
        acc[r] = _mm512_fmadd_ps(a, right[selected[ck][tile + r]], acc[r]);
      }
    }
    for (int r = 0; r < Rows; ++r) {
      result[tile + r] = bit_cast<U32>(acc[r]);
    }
  }
  for (int reg = 0; reg < 8; ++reg) {
    std::memcpy(d[reg], &result[reg], sizeof(U32));
    std::memcpy(d[reg] + 16, &result[reg + 8], sizeof(U32));
  }
}

template Swmmac8Backend swmmac8_x86_64_v4<false, false, 32>;
template Swmmac8Backend swmmac8_x86_64_v4<false, false, 64>;
template Swmmac8Backend swmmac8_x86_64_v4<false, true, 32>;
template Swmmac8Backend swmmac8_x86_64_v4<false, true, 64>;
template Swmmac8Backend swmmac8_x86_64_v4<true, false, 32>;
template Swmmac8Backend swmmac8_x86_64_v4<true, false, 64>;
template Swmmac8Backend swmmac8_x86_64_v4<true, true, 32>;
template Swmmac8Backend swmmac8_x86_64_v4<true, true, 64>;

} // namespace rocjitsu::arith
