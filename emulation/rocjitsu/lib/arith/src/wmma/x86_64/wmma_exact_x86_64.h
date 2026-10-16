// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared bit-exact SIMD FP16/BF16 WMMA implementation for gfx11 and gfx12.
// Maps registers and modifiers into staged dot products, then packs results
// and defers stores for aliasing. The arithmetic model comes from
// dot/x86_64/exact_dot_x86_64.h; vector width is selected by the wrappers.

#pragma once

#include "dot/x86_64/exact_dot_x86_64.h"
#include "rj_arith/rj_arith.h"
#include "wmma/layout.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Compute empirical bit-exact FP16/BF16 WMMA using V-wide columns.
// Requires validated mode; D may alias inputs. Packed gfx11 preserves the
// unselected halfword; saturate controls finite FP16 overflow.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed, class V>
void wmma_exact(uint32_t mode, bool saturate, uint32_t *const *d, const uint32_t *const *a,
                const uint32_t *const *b, const uint32_t *const *c) {
  constexpr int Columns = sizeof(V) / sizeof(uint32_t);
  constexpr int Registers = 256 / Lanes, KPerLane = 256 / Lanes;
  constexpr int DestinationRegisters = Packed && !Rdna3 ? Registers / 2 : Registers;
  // gfx11 rounds DOT2 stages; gfx12 rounds DOT4 stages, including packed narrowing.
  constexpr int Step = Rdna3 ? 2 : 4;
  alignas(64) uint32_t result[DestinationRegisters][Lanes] = {};
  const int high = mode & RJ_ARITH_WMMA_HIGH_C_D ? 16 : 0;
  for (int row = 0; row < 16; ++row) {
    int group = row % (Lanes / 16);
    int reg = Rdna3 ? row / (Lanes / 16) : row % Registers;
    int first_lane = Rdna3 ? 16 * group : 16 * (row / 8) + (Lanes == 64 ? 32 * ((row / 4) % 2) : 0);
    int dest_reg = Packed && !Rdna3 ? reg / 2 : reg;
    int shift = Rdna3 ? high : 16 * (reg % 2);
    for (int col = 0; col < 16; col += Columns) {
      auto acc = exact_dot::load<V>(c[dest_reg] + first_lane + col);
      if constexpr (Packed) {
        acc = exact_dot::bit_and(exact_dot::shift_right(acc, exact_dot::constant<V>(shift)),
                                 exact_dot::constant<V>(65535));
      }
      if (mode & RJ_ARITH_WMMA_ABS_C) {
        acc = exact_dot::bit_and(acc, exact_dot::constant<V>(Packed ? 0x7fff : 0x7fffffff));
      }
      if (mode & RJ_ARITH_WMMA_NEG_C) {
        acc = exact_dot::bit_xor(acc, exact_dot::constant<V>(Packed ? 0x8000 : int(0x80000000U)));
      }
      for (int step = 0; step < 16; step += Step) {
        V av[Step], bv[Step];
        for (int j = 0; j < Step; ++j) {
          int logical = step + j;
          // gfx12 Wave32 visits K blocks 0,8,4,12; other layouts use 0,4,8,12.
          int k = !Rdna3 && Lanes == 32 ? rocjitsu::arith::wmma_gfx12_w32_k(logical) : logical;
          int input_reg = Rdna3 ? k / 2 : (k % KPerLane) / 2;
          int input_lane = 16 * (Rdna3 ? group : k / KPerLane);
          uint32_t left = (a[input_reg][row + input_lane] >> (16 * (k % 2))) & 65535;
          if (mode & (1U << (3 * (k % 2)))) {
            left ^= 0x8000;
          }
          av[j] = exact_dot::constant<V>(int(left));
          auto right = exact_dot::load<V>(b[input_reg] + input_lane + col);
          right = exact_dot::bit_and(
              exact_dot::shift_right(right, exact_dot::constant<V>(16 * (k % 2))),
              exact_dot::constant<V>(65535));
          if (mode & (2U << (3 * (k % 2)))) {
            right = exact_dot::bit_xor(right, exact_dot::constant<V>(0x8000));
          }
          bv[j] = right;
        }
        if constexpr (Packed) {
          acc = exact_dot::widen<Bf16>(acc);
        }
        acc = exact_dot::dot<Bf16, Rdna3, Packed>(av, bv, acc, saturate);
      }
      if constexpr (Packed) {
        acc = exact_dot::shift_left(acc, exact_dot::constant<V>(shift));
        // gfx11 retains the other half of D; gfx12 combines two newly computed halves.
        auto old = Rdna3 ? exact_dot::load<V>(d[dest_reg] + first_lane + col)
                         : exact_dot::load<V>(result[dest_reg] + first_lane + col);
        acc = exact_dot::bit_or(
            acc, exact_dot::andnot(exact_dot::constant<V>(int(65535U << shift)), old));
      }
      exact_dot::store(result[dest_reg] + first_lane + col, acc);
    }
  }
  // All source and preserved-half reads precede writes, including partial aliasing.
  for (int reg = 0; reg < DestinationRegisters; ++reg) {
    for (int lane = 0; lane < Lanes; lane += Columns) {
      exact_dot::store(d[reg] + lane, exact_dot::load<V>(result[reg] + lane));
    }
  }
}

} // namespace rocjitsu::arith
