// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_device_primitives.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_report_emission.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <limits>

namespace rocjitsu::consan::detail {
namespace {

[[nodiscard]] bool append_global_atomic_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  const TargetProfile *target = target_profile(arch);
  return target != nullptr && detail::append_global_atomic_completion(words, *target);
}

} // namespace

[[nodiscard]] bool append_atomic_fetch_add_one_u32(std::vector<uint32_t> &words,
                                                   uint64_t counter_address, uint16_t result_vgpr,
                                                   uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const TargetProfile *target = target_profile(arch);
  return target != nullptr &&
         detail::append_atomic_counter_increment(words,
                                                 {
                                                     .counter_address = counter_address,
                                                     .result_vgpr = result_vgpr,
                                                     .address_vgpr = scratch_vgpr,
                                                 },
                                                 *target);
}

bool append_publication_ticket(std::vector<uint32_t> &words, uint64_t counter_address,
                               uint16_t result_vgpr, uint16_t address_vgpr, rj_code_arch_t arch) {
  if (result_vgpr + 1u >= 256u || address_vgpr + 1u >= 256u ||
      (result_vgpr <= address_vgpr + 1u && address_vgpr <= result_vgpr + 1u))
    return false;
  InstructionSequence sequence(words);
  sequence
      .append(instrumentation::build_v_mov_b32_literal(
                  address_vgpr, static_cast<uint32_t>(counter_address), arch),
              instrumentation::build_v_mov_b32_literal(
                  address_vgpr + 1u, static_cast<uint32_t>(counter_address >> 32u), arch),
              instrumentation::build_v_mov_b32_literal(result_vgpr, 1u, arch),
              instrumentation::build_v_mov_b32_literal(result_vgpr + 1u, 0u, arch),
              instrumentation::build_flat_atomic_add_u64(address_vgpr, result_vgpr, result_vgpr,
                                                         true, kAmdGpuScopeDevice, arch))
      .require(append_global_atomic_wait(words, arch))
      .append(instrumentation::build_v_add_u64_literal(result_vgpr, 1u, arch));
  return sequence.finish();
}

[[nodiscard]] bool append_atomic_load_u32(std::vector<uint32_t> &words, uint16_t address_vgpr,
                                          uint16_t result_vgpr, rj_code_arch_t arch) {
  const auto atomic_load = instrumentation::build_flat_atomic_add_u32(
      address_vgpr, result_vgpr, result_vgpr, /*return_old_value=*/true, kAmdGpuScopeDevice, arch);
  const uint32_t zero = build_v_mov_b32_e32(result_vgpr, scalar_positive_inline_u32(0), arch);
  InstructionSequence sequence(words);
  sequence.append(zero, atomic_load).require(append_global_atomic_wait(words, arch));
  return sequence.finish();
}

bool append_select_first_lane_in_exec_mask(std::vector<uint32_t> &words, uint16_t lane_rank_vgpr,
                                           uint16_t active_exec_sgpr, uint16_t saved_exec_sgpr,
                                           rj_code_arch_t arch) {
  InstructionSequence sequence(words);
  sequence.append(
      instrumentation::build_v_mbcnt_lo_u32_b32(lane_rank_vgpr, active_exec_sgpr,
                                                scalar_positive_inline_u32(0), arch),
      instrumentation::build_v_mbcnt_hi_u32_b32(lane_rank_vgpr,
                                                static_cast<uint16_t>(active_exec_sgpr + 1u),
                                                vector_source_vgpr(lane_rank_vgpr), arch),
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), lane_rank_vgpr, arch),
      instrumentation::build_valu_vcc_to_salu_dependency_wait(arch),
      instrumentation::build_s_and_saveexec_b64(saved_exec_sgpr, kAmdGpuVccLo, arch));
  return sequence.finish();
}

} // namespace rocjitsu::consan::detail
