// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

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

/// @brief Borrow full-byte register spans, retaining the region through arithmetic execution.
/// Failed access leaves the instruction error on the wavefront and stops collection.
[[nodiscard]] inline bool read_matrix_pointers(const RegisterAccess::VgprReadRegion &region,
                                               const uint32_t **pointers) {
  for (uint32_t reg = 0; reg < region.reg_count(); ++reg) {
    auto lanes = region.lanes(reg);
    if (lanes.failed())
      return false;
    pointers[reg] = lanes.value().data();
  }
  return true;
}

/// @brief Evaluate matrix output in scratch storage and notify/store only after success.
/// The callback may initialize tied accumulators before invoking the arithmetic entry point.
template <uint32_t MaxRegisters = 8, typename Invoke>
inline void commit_matrix_result(Wavefront &wave, RegisterAccess &registers, uint32_t destination,
                                 uint32_t result_registers, uint8_t write_bytes, Invoke &&invoke) {
  if (wave.instruction_execution_failed())
    return;
  const uint32_t lanes = wave.wf_size();
  const uint64_t all_lanes = lanes == 64 ? ~uint64_t{0} : 0xffffffffu;
  std::array<uint32_t *, MaxRegisters> d{};
  std::array<std::array<uint32_t, 64>, MaxRegisters> results{};
  for (uint32_t reg = 0; reg < result_registers; ++reg)
    d[reg] = results[reg].data();
  if (invoke(d.data()) != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  // Acquiring a write region notifies plugins, so it must follow successful evaluation.
  auto output = registers.write_vgpr_region(destination, result_registers, all_lanes, write_bytes);
  for (uint32_t reg = 0; reg < result_registers; ++reg)
    for (uint32_t lane = 0; lane < lanes; ++lane)
      output.set_lane(reg, lane, results[reg][lane]);
}

// RDNA3/RDNA4 redirect a source tuple outside the architectural allocation to v0
// (ISA section 3.3.2.2). The backing storage slot may be larger than that allocation.
inline uint32_t matrix_source_base(const Wavefront &wave, uint32_t base, uint32_t count) {
  return wave.vgpr_alloc().contains(base, count) ? base : wave.vgpr_alloc().base;
}

/// @brief Execute dense WMMA with observed source reads and transactional stores.
/// @details Physical register indices are resolved by the decoded instruction.
/// Packed RDNA3 destinations use write_bytes to preserve the unselected half
/// without reporting that preserved storage as an architectural source read.
/// factor_lanes limits A/B read observations; zero selects the whole wave.
inline void exec_wmma(Wavefront &wave, WmmaFunction function, uint64_t flags,
                      uint64_t instruction_flags, uint32_t destination, uint32_t source_a,
                      uint32_t source_b, uint32_t source_c, uint32_t factor_registers,
                      uint32_t result_registers, std::optional<uint32_t> constant_c,
                      uint8_t write_bytes = 0xf, uint32_t factor_lanes = 0) {
  RegisterAccess registers(wave);
  if (wave.instruction_execution_failed())
    return;
  // An invalid destination suppresses the instruction, including operand observations.
  if (!wave.vgpr_alloc().contains(destination, result_registers))
    return;
  const uint32_t lanes = wave.wf_size();
  assert(lanes == 32 || lanes == 64);
  assert(factor_registers > 0 && factor_registers <= 8);
  assert(result_registers > 0 && result_registers <= 8);
  assert(factor_lanes == 0 || factor_lanes == 32 || factor_lanes == lanes);
  const uint64_t all_lanes = lanes == 64 ? ~uint64_t{0} : 0xffffffffu;
  const uint64_t input_lanes = factor_lanes == 32 ? 0xffffffffu : all_lanes;
  auto a_region = registers.read_vgpr_region(matrix_source_base(wave, source_a, factor_registers),
                                             factor_registers, input_lanes);
  auto b_region = registers.read_vgpr_region(matrix_source_base(wave, source_b, factor_registers),
                                             factor_registers, input_lanes);
  std::array<const uint32_t *, 8> a{}, b{}, c{};
  std::array<uint32_t, 64> broadcast{};
  if (!read_matrix_pointers(a_region, a.data()) || !read_matrix_pointers(b_region, b.data()))
    return;
  // Keep the region alive through the call: invalid source views may own their
  // zero backing storage. A constant accumulator is not a register read.
  std::optional<RegisterAccess::VgprReadRegion> c_region;
  if (constant_c) {
    broadcast.fill(*constant_c);
    c.fill(broadcast.data());
  } else {
    c_region.emplace(registers.read_vgpr_region(
        matrix_source_base(wave, source_c, result_registers), result_registers, all_lanes));
    if (!read_matrix_pointers(*c_region, c.data()))
      return;
  }
  // WMMA has no execution mask. Partial/observed stores retain staged commits.
  if (write_bytes == 0xf) {
    auto status =
        registers.try_bulk_write_vgprs<8>(destination, result_registers, [&](uint32_t *const *d) {
          return function(flags, instruction_flags, d, a.data(), b.data(), c.data());
        });
    if (status) {
      if (*status != RJ_ARITH_SUCCESS)
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      return;
    }
  }
  commit_matrix_result(wave, registers, destination, result_registers, write_bytes,
                       [&](uint32_t *const *d) {
                         return function(flags, instruction_flags, d, a.data(), b.data(), c.data());
                       });
}

} // namespace rocjitsu::amdgpu::arith
