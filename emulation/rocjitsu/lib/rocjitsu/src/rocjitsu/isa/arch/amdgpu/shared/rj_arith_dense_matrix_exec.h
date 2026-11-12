// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA dense and sparse matrix bridge: retain observed source regions through the arithmetic call,
// then publish whole-register results. Accumulator-bank resolution stays in the handler.
#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dot_packed16.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_exec.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <array>
#include <optional>

namespace rocjitsu::amdgpu::arith {

// Execute dense MFMA/WMMA on the issuer after resolving physical register bases.
// Error state belongs to the current instruction and must not be shared with helpers. Integer and
// floating entry points share this pointer ABI; inline values are passed as raw operand bits.
// Unlike RDNA matrix sources, CDNA tuples are not redirected to v0 here.
template <uint32_t FactorRegisters, uint32_t ResultRegisters, uint32_t Lanes,
          uint32_t AccumulatorRegisters = ResultRegisters, bool PackedAccumulator = false,
          bool WideAccumulator = false, typename Function = WmmaFunction>
inline void exec_dense_fp_matrix(Wavefront &wave, Function function, uint64_t flags,
                                 uint64_t controls, uint32_t destination, uint32_t source_a,
                                 uint32_t source_b, uint32_t source_c,
                                 std::optional<uint64_t> constant_c,
                                 uint64_t a_lanes = ~uint64_t{0} >> (64 - Lanes),
                                 uint64_t b_lanes = ~uint64_t{0} >> (64 - Lanes),
                                 uint32_t a_registers = FactorRegisters,
                                 uint32_t b_registers = FactorRegisters) {
  static_assert(FactorRegisters <= 16 && ResultRegisters <= 32 && AccumulatorRegisters <= 32);
  static_assert(Lanes == 32 || Lanes == 64);
  if (wave.instruction_execution_failed()) {
    return;
  }
  if (wave.wf_size() != Lanes) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  if (a_registers > FactorRegisters || b_registers > FactorRegisters) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  RegisterAccess registers(wave);
  constexpr uint64_t all_lanes = ~uint64_t{0} >> (64 - Lanes);
  auto a_region = registers.read_vgpr_region(source_a, a_registers, a_lanes);
  auto b_region = registers.read_vgpr_region(source_b, b_registers, b_lanes);
  std::array<const uint32_t *, FactorRegisters> a{}, b{};
  std::array<const uint32_t *, AccumulatorRegisters> c{};
  if (!read_matrix_pointers(a_region, a.data()) || !read_matrix_pointers(b_region, b.data())) {
    return;
  }
  std::array<uint32_t, 64> broadcast{}, broadcast_high{};
  std::optional<RegisterAccess::VgprReadRegion> c_region;
  if (constant_c) {
    if constexpr (PackedAccumulator) {
      // INLINE_C distinguishes a broadcast scalar from a VGPR tuple.
      controls |= RJ_ARITH_WMMA_INLINE_C;
    }
    if constexpr (WideAccumulator) {
      static_assert(AccumulatorRegisters % 2 == 0 && !PackedAccumulator);
      const uint64_t bits = *constant_c;
      broadcast.fill(uint32_t(bits));
      broadcast_high.fill(uint32_t(bits >> 32));
      for (uint32_t reg = 0; reg < AccumulatorRegisters; ++reg) {
        c[reg] = reg % 2 ? broadcast_high.data() : broadcast.data();
      }
    } else {
      broadcast.fill(*constant_c);
      c.fill(broadcast.data());
    }
  } else {
    c_region.emplace(registers.read_vgpr_region(source_c, AccumulatorRegisters, all_lanes));
    if (!read_matrix_pointers(*c_region, c.data())) {
      return;
    }
  }
  // The unobserved path borrows pointers directly; the library stages aliased inputs.
  auto status = registers.try_bulk_write_vgprs<ResultRegisters>(
      destination, ResultRegisters, [&](uint32_t *const *d) {
        return function(flags, controls, d, a.data(), b.data(), c.data());
      });
  if (status) {
    if (*status != RJ_ARITH_SUCCESS) {
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    }
    return;
  }
  commit_matrix_result<ResultRegisters>(
      wave, registers, destination, ResultRegisters, 0xf, [&](uint32_t *const *d) {
        return function(flags, controls, d, a.data(), b.data(), c.data());
      });
}

// Execute a tied-accumulator CDNA sparse matrix with independently sized operands.
// Retain observed reads until the library has consumed all potentially overlapping inputs.
template <uint32_t A, uint32_t B, uint32_t D, uint32_t Index, uint32_t Lanes, uint32_t C = D>
inline void exec_sparse_matrix(Wavefront &wave, WmmaFunction function, uint64_t flags,
                               uint64_t controls, uint32_t destination, uint32_t source_a,
                               uint32_t source_b, uint32_t source_index) {
  static_assert(Lanes == 32 || Lanes == 64);
  if (wave.instruction_execution_failed()) {
    return;
  }
  if (wave.wf_size() != Lanes) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  // RegisterAccess validates physical destinations, including the separate CDNA
  // accumulator bank beyond the ordinary VGPR allocation.
  RegisterAccess registers(wave);
  constexpr uint64_t all_lanes = ~uint64_t{0} >> (64 - Lanes);
  auto ar = registers.read_vgpr_region(source_a, A, all_lanes);
  auto br = registers.read_vgpr_region(source_b, B, all_lanes);
  auto ir = registers.read_vgpr_region(source_index, Index, all_lanes);
  auto cr = registers.read_vgpr_region(destination, C, all_lanes);
  std::array<const uint32_t *, A> a{};
  std::array<const uint32_t *, B> b{};
  std::array<const uint32_t *, Index> index{};
  std::array<const uint32_t *, C> c{};
  if (!read_matrix_pointers(ar, a.data()) || !read_matrix_pointers(br, b.data()) ||
      !read_matrix_pointers(ir, index.data()) || !read_matrix_pointers(cr, c.data())) {
    return;
  }
  if constexpr (C != D) {
    // Mixed BF16F32 reads eight accumulator registers but publishes only four.
    // Private staging keeps the extra input pointers out of the write observation range.
    commit_matrix_result<D>(wave, registers, destination, D, 0xf, [&](uint32_t *const *d) {
      uint32_t storage[C][Lanes];
      uint32_t *tied[C];
      for (uint32_t reg = 0; reg < C; ++reg) {
        tied[reg] = storage[reg];
        for (uint32_t lane = 0; lane < Lanes; ++lane) {
          storage[reg][lane] = c[reg][lane];
        }
      }
      int result = function(flags, controls, tied, a.data(), b.data(), index.data());
      if (result == RJ_ARITH_SUCCESS) {
        for (uint32_t reg = 0; reg < D; ++reg) {
          for (uint32_t lane = 0; lane < Lanes; ++lane) {
            d[reg][lane] = storage[reg][lane];
          }
        }
      }
      return result;
    });
    return;
  }
  auto status = registers.try_bulk_write_vgprs<D>(destination, D, [&](uint32_t *const *d) {
    return function(flags, controls, d, a.data(), b.data(), index.data());
  });
  if (status) {
    if (*status != RJ_ARITH_SUCCESS) {
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    }
    return;
  }
  commit_matrix_result<D>(wave, registers, destination, D, 0xf, [&](uint32_t *const *d) {
    // The observed-write path needs a private copy of the tied accumulator.
    for (uint32_t reg = 0; reg < D; ++reg) {
      for (uint32_t lane = 0; lane < Lanes; ++lane) {
        d[reg][lane] = c[reg][lane];
      }
    }
    return function(flags, controls, d, a.data(), b.data(), index.data());
  });
}

} // namespace rocjitsu::amdgpu::arith
