// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_internal.h"

#include <optional>
#include <vector>

namespace rocjitsu::consan::detail {

/// Immutable policy and ABI input to the shared runtime workgroup gate.
struct RuntimeWorkgroupGatePlan {
  uint16_t exec_save_sgpr = 0;
  uint32_t sample_stride = 1;
  uint32_t sample_offset = 0;
  std::optional<uint16_t> dispatch_id_sgpr;
  uint64_t literal_dispatch_id = 0;
};

/// Position-independent predicate prefix for a whole-text probe fragment.
/// `bypass_branch_word` is an unresolved unconditional branch owned by the
/// text transaction; its target is the base operation after the complete
/// instrumentation nest.
struct RuntimeWorkgroupGatePrefix {
  std::vector<uint32_t> words;
  uint32_t bypass_branch_word = 0;
};

[[nodiscard]] bool
has_probe_entry_runtime_workgroup_gate(const WorkgroupSources &workgroup_sources);

[[nodiscard]] std::optional<RuntimeWorkgroupGatePlan>
plan_runtime_workgroup_gate(const RuntimeWorkgroupGatePlan &plan,
                            const WorkgroupSources &workgroup_sources);

[[nodiscard]] std::optional<RuntimeWorkgroupGatePrefix>
build_runtime_workgroup_gate_prefix(const RuntimeWorkgroupGatePlan &plan,
                                    const WorkgroupSources &workgroup_sources, rj_code_arch_t arch);

[[nodiscard]] bool append_runtime_workgroup_residue_compare(std::vector<uint32_t> &words,
                                                            uint16_t residue_sgpr,
                                                            uint16_t temporary_sgpr,
                                                            uint32_t selected_residue,
                                                            rj_code_arch_t arch);

[[nodiscard]] bool append_runtime_workgroup_mix(std::vector<uint32_t> &words,
                                                const WorkgroupSource &source, uint16_t quotient,
                                                uint16_t residue, rj_code_arch_t arch);

} // namespace rocjitsu::consan::detail
