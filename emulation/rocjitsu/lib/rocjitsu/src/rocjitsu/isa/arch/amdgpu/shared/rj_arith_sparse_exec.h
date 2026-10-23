// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_SPARSE_EXEC_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_SPARSE_EXEC_H_

#include "rj_arith/rj_arith.h"
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
inline void execute_swmmac(Wavefront &wave, SwmmacFunction function, uint64_t flags,
                           uint64_t instruction_flags, uint32_t destination, uint32_t source_a,
                           uint32_t source_b, uint32_t source_index, uint32_t a_registers,
                           uint32_t b_registers, uint32_t result_registers, uint32_t a_lanes) {
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
  auto a_region = registers.read_vgpr_region(source_a, a_registers, compressed_lanes);
  auto b_region = registers.read_vgpr_region(source_b, b_registers, all_lanes);
  auto index_region = registers.read_vgpr_region(source_index, 1, compressed_lanes);
  auto accumulator = registers.read_vgpr_region(destination, result_registers, all_lanes);
  std::array<const uint32_t *, 4> a{};
  std::array<const uint32_t *, 8> b{};
  const uint32_t *index[] = {index_region.lanes(0).data()};
  for (uint32_t reg = 0; reg < a_registers; ++reg)
    a[reg] = a_region.lanes(reg).data();
  for (uint32_t reg = 0; reg < b_registers; ++reg)
    b[reg] = b_region.lanes(reg).data();
  // The tied accumulator is already observed. rj-arith owns its alias-safe update.
  if ((wave.vgpr_write_mask() & all_lanes) == all_lanes) {
    auto status = registers.try_bulk_write_vgprs<8>(
        destination, result_registers, all_lanes, [&](uint32_t *const *d, uint64_t) {
          return function(flags, instruction_flags, d, a.data(), b.data(), index);
        });
    if (status) {
      if (*status != RJ_ARITH_SUCCESS)
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      return;
    }
  }
  std::array<uint32_t *, 8> d{};
  std::array<std::array<uint32_t, 64>, 8> results{};
  for (uint32_t reg = 0; reg < results.size() && reg < result_registers; ++reg) {
    std::copy_n(accumulator.lanes(reg).data(), lanes, results[reg].data());
    d[reg] = results[reg].data();
  }
  const int status = function(flags, instruction_flags, d.data(), a.data(), b.data(), index);
  if (status != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  auto output = registers.write_vgpr_region(destination, result_registers, all_lanes);
  for (uint32_t reg = 0; reg < result_registers; ++reg)
    for (uint32_t lane = 0; lane < lanes; ++lane)
      output.set_lane(reg, lane, results[reg][lane]);
}

} // namespace rocjitsu::amdgpu::arith

#endif // ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_SPARSE_EXEC_H_
