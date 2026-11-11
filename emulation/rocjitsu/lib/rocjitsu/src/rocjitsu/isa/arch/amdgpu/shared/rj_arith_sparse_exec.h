// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_exec.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>

namespace rocjitsu::amdgpu::arith {

using SwmmacFunction = int (*)(uint64_t, uint64_t, uint32_t *const *, const uint32_t *const *,
                               const uint32_t *const *, const uint32_t *const *);

/// @brief Execute sparse WMMA, observing tied accumulator reads before any writes.
/// @details The instruction has already resolved physical register indices.
/// a_lanes is 32 for Wave64 K=32 IU4 and the full wave width otherwise; it
/// describes the consumed lanes of both compressed A and its index register.
inline void exec_swmmac(Wavefront &wave, SwmmacFunction function, uint64_t flags,
                        uint64_t instruction_flags, uint32_t destination, uint32_t source_a,
                        uint32_t source_b, uint32_t source_index, uint32_t a_registers,
                        uint32_t b_registers, uint32_t result_registers, uint32_t a_lanes) {
  if (wave.instruction_execution_failed())
    return;
  // Reject the tied destination before reading it or any other operand.
  if (!wave.vgpr_alloc().contains(destination, result_registers))
    return;
  RegisterAccess registers(wave);
  assert(wave.wf_size() == 32 || wave.wf_size() == 64);
  // Keep the architectural bound visible to the compiler even with assertions disabled.
  const uint32_t lanes = wave.wf_size() == 64 ? 64 : 32;
  assert(a_lanes == 32 || a_lanes == lanes);
  assert(a_registers > 0 && a_registers <= 4);
  assert(b_registers > 0 && b_registers <= 8);
  assert(result_registers > 0 && result_registers <= 8);
  const uint64_t all_lanes = lanes == 64 ? ~uint64_t{0} : 0xffffffffu;
  const uint64_t compressed_lanes = a_lanes == 64 ? ~uint64_t{0} : 0xffffffffu;
  auto a_region = registers.read_vgpr_region(matrix_source_base(wave, source_a, a_registers),
                                             a_registers, compressed_lanes);
  auto b_region = registers.read_vgpr_region(matrix_source_base(wave, source_b, b_registers),
                                             b_registers, all_lanes);
  auto index_region =
      registers.read_vgpr_region(matrix_source_base(wave, source_index, 1), 1, compressed_lanes);
  auto accumulator = registers.read_vgpr_region(
      matrix_source_base(wave, destination, result_registers), result_registers, all_lanes);
  std::array<const uint32_t *, 4> a{};
  std::array<const uint32_t *, 8> b{};
  std::array<const uint32_t *, 8> c{};
  const uint32_t *index[1];
  if (!read_matrix_pointers(a_region, a.data()) || !read_matrix_pointers(b_region, b.data()) ||
      !read_matrix_pointers(index_region, index) || !read_matrix_pointers(accumulator, c.data()))
    return;
  // The tied accumulator is already observed. rj-arith owns its alias-safe update.
  {
    auto status =
        registers.try_bulk_write_vgprs<8>(destination, result_registers, [&](uint32_t *const *d) {
          return function(flags, instruction_flags, d, a.data(), b.data(), index);
        });
    if (status) {
      if (*status != RJ_ARITH_SUCCESS)
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      return;
    }
  }
  commit_matrix_result(wave, registers, destination, result_registers, 0xf,
                       [&](uint32_t *const *d) {
                         for (uint32_t reg = 0; reg < result_registers; ++reg)
                           std::copy_n(c[reg], lanes, d[reg]);
                         return function(flags, instruction_flags, d, a.data(), b.data(), index);
                       });
}

} // namespace rocjitsu::amdgpu::arith
