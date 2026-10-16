// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_EXEC_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_EXEC_H_

#include "rj_arith/rj_arith.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu::arith {

using WmmaFunction = int (*)(uint64_t, uint64_t, uint32_t *const *, const uint32_t *const *,
                             const uint32_t *const *, const uint32_t *const *);

/// @brief Execute dense WMMA with observed source reads and transactional stores.
/// @details Physical register indices are resolved by the decoded instruction.
/// Packed RDNA3 destinations use write_bytes to preserve the unselected half
/// without reporting that preserved storage as an architectural source read.
/// factor_lanes limits A/B read observations; zero selects the whole wave.
inline void execute_wmma(Wavefront &wave, WmmaFunction function, uint64_t flags,
                         uint64_t instruction_flags, uint32_t destination, uint32_t source_a,
                         uint32_t source_b, uint32_t source_c, uint32_t factor_registers,
                         uint32_t result_registers, std::optional<uint32_t> constant_c,
                         uint8_t write_bytes = 0xf, uint32_t factor_lanes = 0) {
  RegisterAccess registers(wave);
  const uint32_t lanes = wave.wf_size();
  assert(lanes == 32 || lanes == 64);
  assert(factor_registers > 0 && factor_registers <= 8);
  assert(result_registers > 0 && result_registers <= 8);
  assert(factor_lanes == 0 || factor_lanes == 32 || factor_lanes == lanes);
  const uint64_t all_lanes = lanes == 64 ? ~uint64_t{0} : 0xffffffffu;
  const uint64_t input_lanes = factor_lanes == 32 ? 0xffffffffu : all_lanes;
  auto a_region = registers.read_vgpr_region(source_a, factor_registers, input_lanes);
  auto b_region = registers.read_vgpr_region(source_b, factor_registers, input_lanes);
  std::array<const uint32_t *, 8> a{}, b{}, c{};
  std::array<uint32_t, 64> broadcast{};
  for (uint32_t reg = 0; reg < factor_registers; ++reg) {
    a[reg] = a_region.lanes(reg).data();
    b[reg] = b_region.lanes(reg).data();
  }
  // Keep the region alive through the call: invalid source views may own their
  // zero backing storage. A constant accumulator is not a register read.
  std::optional<RegisterAccess::VgprReadRegion> c_region;
  if (constant_c) {
    broadcast.fill(*constant_c);
    c.fill(broadcast.data());
  } else {
    c_region.emplace(registers.read_vgpr_region(source_c, result_registers, all_lanes));
    for (uint32_t reg = 0; reg < result_registers; ++reg)
      c[reg] = c_region->lanes(reg).data();
  }
  // WMMA has no execution mask. Partial/observed stores retain staged commits.
  if (write_bytes == 0xf && (wave.vgpr_write_mask() & all_lanes) == all_lanes) {
    auto status = registers.try_bulk_write_vgprs<8>(
        destination, result_registers, all_lanes, [&](uint32_t *const *d, uint64_t) {
          return function(flags, instruction_flags, d, a.data(), b.data(), c.data());
        });
    if (status) {
      if (*status != RJ_ARITH_SUCCESS)
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      return;
    }
  }
  std::array<uint32_t *, 8> d{};
  std::array<std::array<uint32_t, 64>, 8> results{};
  for (uint32_t reg = 0; reg < result_registers; ++reg)
    d[reg] = results[reg].data();
  const int status = function(flags, instruction_flags, d.data(), a.data(), b.data(), c.data());
  if (status != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  // Acquiring a write region notifies plugins. Do this only after rj-arith accepts
  // the invocation, and after all source values have been consumed.
  auto output = registers.write_vgpr_region(destination, result_registers, all_lanes, write_bytes);
  for (uint32_t reg = 0; reg < result_registers; ++reg)
    for (uint32_t lane = 0; lane < lanes; ++lane)
      output.set_lane(reg, lane, results[reg][lane]);
}

} // namespace rocjitsu::amdgpu::arith

#endif // ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_EXEC_H_
