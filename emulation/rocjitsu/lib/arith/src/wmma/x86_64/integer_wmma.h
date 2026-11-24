// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense integer WMMA factor layout and staged accumulation for packed SIMD dots.
#pragma once

#include "common/compiler.h"
#include "wmma/layout.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Evaluate a tile with validated flags. Each Dot handles the signedness and
// native product width of its packed operands. D may overlap any source VGPR.
template <int Bits, int K, int Lanes, bool Rdna3, int RowsPerBlock, typename Dot>
void integer_wmma_tile(const Dot &dot, uint32_t *const *d, const uint32_t *const *a,
                       const uint32_t *const *b, const uint32_t *const *c) {
  using V = typename Dot::V;
  constexpr int Groups = Rdna3 ? Lanes / 16 : 1;
  constexpr int Rows = 16 / Groups;
  constexpr int Block = RowsPerBlock < Rows ? RowsPerBlock : Rows;
  constexpr int Chunks = 16 / Dot::width;
  constexpr int Stages = !Dot::staged_clamp ? 1 : Rdna3 ? K / (Bits == 8 ? 2 : 4) : 2;
  constexpr int Dots = K / (Stages * Dot::products);
  static_assert(Rows % Block == 0 && K % (Stages * Dot::products) == 0);
  alignas(64) uint32_t result[256 / Lanes][Lanes];
  for (int group = 0; group < Groups; ++group) {
    // gfx11 interleaves rows from independent sixteen-lane factor groups.
    // gfx12 Wave64 storage has already been mapped into Wave32 register sections.
    alignas(64) uint32_t left[Stages][Dots][16];
    V right[Stages][Dots][Chunks], bias[Stages][Chunks];
    for (int stage = 0; stage < Stages; ++stage) {
      for (auto &value : bias[stage]) {
        value = V{};
      }
      RJ_ARITH_UNROLL(8)
      for (int q = 0; q < Dots; ++q) {
        int step = q * Dot::products;
        int k = Rdna3               ? stage * (K / Stages) + step
                : Dot::staged_clamp ? wmma_gfx12_integer_stage_k(stage, step)
                                    : step;
        int local = Rdna3 ? k : wmma_gfx12_packed_local(k);
        int reg = local * Bits / 32;
        int lane = 16 * (Rdna3 ? group : (k / 8) % 2);
        for (int chunk = 0; chunk < Chunks; ++chunk) {
          int col = chunk * Dot::width;
          V av = dot.unpack_a(dot.load(a[reg] + lane + col), local);
          V bv = dot.unpack_b(dot.load(b[reg] + lane + col), local);
          dot.store(left[stage][q] + col, av);
          right[stage][q][chunk] = bv;
          if constexpr (Dot::biased) {
            bias[stage][chunk] = dot.sum_b(bias[stage][chunk], bv);
          }
        }
      }
      if constexpr (Dot::biased) {
        for (auto &value : bias[stage]) {
          value = dot.bias_correction(value);
        }
      }
    }
    for (int first = 0; first < Rows; first += Block) {
      typename Dot::Accumulator acc[Block][Chunks];
      uint32_t *out[Block];
      for (int i = 0; i < Block; ++i) {
        int row = (first + i) * Groups + group;
        int reg = Rdna3 ? first + i : row % 8;
        int lane = 16 * (Rdna3 ? group : row / 8);
        out[i] = result[reg] + lane;
        for (int chunk = 0; chunk < Chunks; ++chunk) {
          acc[i][chunk] = dot.accumulator(dot.load(c[reg] + lane + chunk * Dot::width));
        }
      }
      for (int stage = 0; stage < Stages; ++stage) {
        V sums[Block][Chunks];
        for (int i = 0; i < Block; ++i) {
          for (int chunk = 0; chunk < Chunks; ++chunk) {
            sums[i][chunk] = dot.start_stage(acc[i][chunk], bias[stage][chunk]);
          }
        }
        RJ_ARITH_UNROLL(8)
        for (int q = 0; q < Dots; ++q) {
          for (int i = 0; i < Block; ++i) {
            V av = dot.broadcast(left[stage][q][(first + i) * Groups + group]);
            for (int chunk = 0; chunk < Chunks; ++chunk) {
              sums[i][chunk] = dot.product(sums[i][chunk], av, right[stage][q][chunk]);
            }
          }
        }
        for (int i = 0; i < Block; ++i) {
          for (int chunk = 0; chunk < Chunks; ++chunk) {
            dot.finish_stage(acc[i][chunk], sums[i][chunk]);
          }
        }
      }
      for (int i = 0; i < Block; ++i) {
        for (int chunk = 0; chunk < Chunks; ++chunk) {
          dot.store(out[i] + chunk * Dot::width, dot.result(acc[i][chunk]));
        }
      }
    }
  }
  // Complete source reads before ascending VGPR writes, including repeated D pointers.
  for (int reg = 0; reg < 256 / Lanes; ++reg) {
    for (int lane = 0; lane < Lanes; lane += Dot::width) {
      dot.store(d[reg] + lane, dot.load(result[reg] + lane));
    }
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
