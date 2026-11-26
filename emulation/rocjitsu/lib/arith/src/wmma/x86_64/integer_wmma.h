// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense integer WMMA factor layout and staged accumulation for packed SIMD dots.
#pragma once

#include "common/compiler.h"
#include "wmma/layout.h"

#include <cstring>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

namespace detail {

// True when every register of one operand ends at or below the first register
// of the other, so no D store can reach a C register. The scalar early exits
// read each pointer once and never reload a pointer list as a vector.
template <int Registers, int Lanes>
[[gnu::always_inline]] inline bool integer_registers_apart(uint32_t *const *d,
                                                           const uint32_t *const *c) {
  constexpr uintptr_t Span = Lanes * sizeof(uint32_t);
  const bool d_above = reinterpret_cast<uintptr_t>(d[0]) > reinterpret_cast<uintptr_t>(c[0]);
  const uint32_t *const *above = d_above ? d : c;
  const uint32_t *const *below = d_above ? c : d;
  const uintptr_t pivot = reinterpret_cast<uintptr_t>(above[0]);
  for (int reg = 0; reg < Registers; ++reg) {
    if (reinterpret_cast<uintptr_t>(above[reg]) < pivot ||
        reinterpret_cast<uintptr_t>(below[reg]) + Span > pivot) {
      return false;
    }
  }
  return true;
}

// Return the accumulator registers to read once D may overlap C. D tied to C
// register by register, ascending and disjoint, lets each row replace only the
// lanes it has already read. Any other layout reads from a copy of C.
template <int Registers, int Lanes>
[[gnu::noinline, gnu::cold]] const uint32_t *const *
integer_accumulator_source(uint32_t *const *d, const uint32_t *const *c,
                           uint32_t (&copy)[Registers][Lanes], const uint32_t *(&rows)[Registers]) {
  constexpr uintptr_t Span = Lanes * sizeof(uint32_t);
  bool tied = true;
  for (int reg = 0; reg < Registers; ++reg) {
    tied &= d[reg] == c[reg] && (reg == 0 || reinterpret_cast<uintptr_t>(d[reg]) >=
                                                 reinterpret_cast<uintptr_t>(d[reg - 1]) + Span);
  }
  if (tied) {
    return c;
  }
  for (int reg = 0; reg < Registers; ++reg) {
    std::memcpy(copy[reg], c[reg], sizeof(copy[reg]));
    rows[reg] = copy[reg];
  }
  return rows;
}

} // namespace detail

// Evaluate a tile with validated flags. Each Dot handles the signedness and
// native product width of its packed operands. D may overlap any source VGPR:
// A and B are unpacked before the first store and D is written in ascending
// VGPR and lane order. Dot::early_store writes each block of rows as it
// finishes, snapshotting C first when D could reach it; otherwise full-width
// rows stay in vector values until every source has been read.
template <int Bits, int K, int Lanes, bool Rdna3, int RowsPerBlock, typename Dot>
void integer_wmma_tile(const Dot &dot, uint32_t *const *d, const uint32_t *const *a,
                       const uint32_t *const *b, const uint32_t *const *c) {
  using V = typename Dot::V;
  constexpr bool Early = Dot::early_store;
  constexpr int Registers = 256 / Lanes;
  constexpr int Groups = Lanes / 16;
  constexpr int Block = RowsPerBlock < 16 ? RowsPerBlock : 16;
  constexpr int Chunks = 16 / Dot::width;
  constexpr int Stages = !Dot::staged_clamp ? 1 : Rdna3 ? K / (Bits == 8 ? 2 : 4) : 2;
  constexpr int Dots = K / (Stages * Dot::products);
  static_assert(16 % Block == 0 && K % (Stages * Dot::products) == 0);
  static_assert(Early || Chunks == 1);

  // Copy C only when an early D store could change an accumulator still to be read.
  alignas(64) uint32_t c_copy[Early ? Registers : 1][Lanes];
  const uint32_t *c_rows[Registers];
  const uint32_t *const *acc_in = c;
  if constexpr (Early) {
    if (!detail::integer_registers_apart<Registers, Lanes>(d, c)) [[unlikely]] {
      acc_in = detail::integer_accumulator_source<Registers, Lanes>(d, c, c_copy, c_rows);
    }
  }

  // gfx11 replicates A and B across every 16-lane group, so group 0 supplies the
  // entire 16x16 tile. gfx12 Wave64 storage has already been mapped into Wave32.
  alignas(64) uint32_t left[Stages][Dots][16];
  V right[Stages][Dots][Chunks], bias[Stages][Chunks];
  for (int stage = 0; stage < Stages; ++stage) {
    if constexpr (Dot::biased) {
      for (auto &value : bias[stage]) {
        value = V{};
      }
    }
    RJ_ARITH_UNROLL(8)
    for (int q = 0; q < Dots; ++q) {
      int step = q * Dot::products;
      int k = Rdna3               ? stage * (K / Stages) + step
              : Dot::staged_clamp ? wmma_gfx12_integer_stage_k(stage, step)
                                  : step;
      int local = Rdna3 ? k : wmma_gfx12_packed_local(k);
      int reg = local * Bits / 32;
      int lane = Rdna3 ? 0 : 16 * ((k / 8) % 2);
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

  // Early stores visit rows in destination order, so each block ends with its
  // ascending stores and repeated or overlapping D registers keep the last
  // register written. Otherwise rows keep their matrix order until the end.
  V late[Early ? 1 : 16];
  for (int first = 0; first < 16; first += Block) {
    typename Dot::Accumulator acc[Block][Chunks];
    int rows[Block];
    for (int i = 0; i < Block; ++i) {
      int slot = first + i;
      int row = Rdna3 || !Early ? slot : slot / 2 + 8 * (slot % 2);
      int reg = Rdna3 ? row / Groups : row % 8;
      int lane = 16 * (Rdna3 ? row % Groups : row / 8);
      rows[i] = row;
      for (int chunk = 0; chunk < Chunks; ++chunk) {
        acc[i][chunk] = dot.accumulator(dot.load(acc_in[reg] + lane + chunk * Dot::width));
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
          V av = dot.broadcast(left[stage][q][rows[i]]);
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
      if constexpr (Early) {
        int reg = Rdna3 ? rows[i] / Groups : rows[i] % 8;
        int lane = 16 * (Rdna3 ? rows[i] % Groups : rows[i] / 8);
        for (int chunk = 0; chunk < Chunks; ++chunk) {
          dot.store(d[reg] + lane + chunk * Dot::width, dot.result(acc[i][chunk]));
        }
      } else {
        late[rows[i]] = dot.result(acc[i][0]);
      }
    }
  }
  if constexpr (!Early) {
    for (int reg = 0; reg < Registers; ++reg) {
      for (int group = 0; group < Groups; ++group) {
        int row = Rdna3 ? reg * Groups + group : reg + 8 * group;
        dot.store(d[reg] + 16 * group, late[row]);
      }
    }
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
