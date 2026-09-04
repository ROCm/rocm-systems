// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_internal.h"

#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::consan_moi_impl {

/// Immutable policy and ABI input to the shared runtime workgroup gate.
struct MoiRuntimeWorkgroupGatePlan {
  enum class Flavor : uint8_t { RecordReplay, Sampled };

  uint16_t exec_save_sgpr = 0;
  uint32_t sample_stride = 1;
  uint32_t sample_offset = 0;
  Flavor flavor = Flavor::Sampled;
  std::optional<ConSanMoiWorkgroupSource> cached_selection;
  std::optional<uint16_t> dispatch_id_sgpr;
  uint64_t literal_dispatch_id = 0;
  ConSanDirectCallForm direct_call_form = ConSanDirectCallForm::SCallB64;
};

/// Exact normalized facts supplied by a mode owner before shared gate
/// eligibility and cache selection are resolved.
struct MoiRuntimeWorkgroupGateInputs {
  uint16_t exec_save_sgpr = 0;
  uint32_t sample_stride = 1;
  uint32_t sample_offset = 0;
  MoiRuntimeWorkgroupGatePlan::Flavor flavor = MoiRuntimeWorkgroupGatePlan::Flavor::Sampled;
  std::optional<uint16_t> dispatch_id_sgpr;
  uint64_t literal_dispatch_id = 0;
  ConSanDirectCallForm direct_call_form = ConSanDirectCallForm::SCallB64;
};

/// Position-independent predicate prefix for a whole-text probe fragment.
/// `bypass_branch_word` is an unresolved unconditional branch owned by the
/// text transaction; its target is the base operation after the complete
/// instrumentation nest.
struct MoiRuntimeWorkgroupGatePrefix {
  std::vector<uint32_t> words;
  uint32_t bypass_branch_word = 0;
};

[[nodiscard]] std::optional<ConSanMoiWorkgroupSource> moi_runtime_workgroup_selection_source();

[[nodiscard]] bool
moi_has_probe_entry_runtime_workgroup_gate(const ConSanMoiWorkgroupSources &workgroup_sources);

[[nodiscard]] std::optional<MoiRuntimeWorkgroupGatePlan>
plan_moi_runtime_workgroup_gate(const MoiRuntimeWorkgroupGateInputs &inputs,
                                const ConSanMoiWorkgroupSources &workgroup_sources);

[[nodiscard]] std::optional<MoiRuntimeWorkgroupGatePrefix>
build_moi_runtime_workgroup_gate_prefix(const MoiRuntimeWorkgroupGatePlan &plan,
                                        const ConSanMoiWorkgroupSources &workgroup_sources,
                                        rj_code_arch_t arch);

[[nodiscard]] uint64_t
moi_runtime_workgroup_gate_reserved_words(uint32_t guest_byte_count,
                                          bool has_cluster_workgroup_id = false);

[[nodiscard]] bool append_moi_runtime_workgroup_residue_compare(std::vector<uint32_t> &words,
                                                                uint16_t residue_sgpr,
                                                                uint16_t temporary_sgpr,
                                                                uint32_t selected_residue,
                                                                rj_code_arch_t arch);

[[nodiscard]] bool append_moi_runtime_workgroup_mix(std::vector<uint32_t> &words,
                                                    const ConSanMoiWorkgroupSource &source,
                                                    uint16_t quotient, uint16_t residue,
                                                    rj_code_arch_t arch);

} // namespace rocjitsu::consan_moi_impl
