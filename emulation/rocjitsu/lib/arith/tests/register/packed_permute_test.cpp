// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed element selectors, word-boundary crossing, masks and overlapping VGPRs.
#include "register/helpers.h"
#include "rj_arith/detail/rj_arith_register.h"

namespace {
// A bit-at-a-time reference avoids reproducing the implementation's word-pair
// extraction. Deterministic random selectors cover all sixteen table elements.
template <int Bits, int Lanes, typename Call> void check_packed(Call call) {
  constexpr int Words = Bits / 2, LowWords = Bits == 4 ? 1 : 2;
  rocjitsu::arith::test::check_register<Lanes, false, 12>(
      [call](auto f, auto e, auto m, auto &r, int d, int) {
        const uint32_t *a[] = {r.v[0], r.v[1]};
        const uint32_t *b[] = {r.v[2], r.v[3]};
        const uint32_t *c[] = {r.v[4], r.v[5]};
        return call(f, e, m, r.v + d, a, b, c);
      },
      [](auto &out, const auto &in, auto e, auto, int d, int) {
        for (int lane = 0; lane < Lanes; ++lane) {
          if (!((e >> lane) & 1)) {
            continue;
          }
          for (int word = 0; word < Words; ++word) {
            out.v[d + word][lane] = 0;
          }
          for (int element = 0; element < 16; ++element) {
            int index = int((in.v[4 + element / 8][lane] >> ((element % 8) * 4)) & 15);
            for (int bit = 0; bit < Bits; ++bit) {
              int source = index * Bits + bit, word = source / 32;
              int reg = word < LowWords ? 2 + word : word - LowWords;
              uint32_t value = (in.v[reg][lane] >> (source % 32)) & 1;
              int output = element * Bits + bit;
              out.v[d + output / 32][lane] |= value << (output % 32);
            }
          }
        }
      });
}
} // namespace

TEST(RegisterPackedPermuteB4W32, Arithmetic) {
  check_packed<4, 32>(rj_arith_v_perm_pk16_b4_u4_w32);
}

TEST(RegisterPackedPermuteB4W64, Arithmetic) {
  check_packed<4, 64>(rj_arith_v_perm_pk16_b4_u4_w64);
}

TEST(RegisterPackedPermuteB6W32, Arithmetic) {
  check_packed<6, 32>(rj_arith_v_perm_pk16_b6_u4_w32);
}

TEST(RegisterPackedPermuteB6W64, Arithmetic) {
  check_packed<6, 64>(rj_arith_v_perm_pk16_b6_u4_w64);
}

TEST(RegisterPackedPermuteB8W32, Arithmetic) {
  check_packed<8, 32>(rj_arith_v_perm_pk16_b8_u4_w32);
}

TEST(RegisterPackedPermuteB8W64, Arithmetic) {
  check_packed<8, 64>(rj_arith_v_perm_pk16_b8_u4_w64);
}
