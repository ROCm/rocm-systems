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

[[nodiscard]] std::optional<ConSanMoiWorkgroupSource> moi_runtime_workgroup_selection_source();

[[nodiscard]] bool
moi_has_probe_entry_runtime_workgroup_gate(const ConSanMoiWorkgroupSources &workgroup_sources);

[[nodiscard]] std::optional<MoiRuntimeWorkgroupGatePlan> plan_moi_runtime_workgroup_gate(
    const ConSanRequest &request, const BoundRuntimeResources &resources,
    const ConSanMoiOperatingPoint &point, const ConSanMoiWorkgroupSources &workgroup_sources,
    MoiRuntimeWorkgroupGatePlan::Flavor flavor, rj_code_arch_t arch);

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

[[nodiscard]] std::optional<std::vector<uint32_t>> build_moi_runtime_workgroup_gate_island_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const MoiRuntimeWorkgroupGatePlan &plan, const ConSanMoiWorkgroupSources &workgroup_sources,
    uint64_t island_text_offset, uint64_t cave_text_offset, uint64_t return_text_offset,
    uint64_t island_word_count, rj_code_arch_t arch);

[[nodiscard]] std::optional<std::vector<uint32_t>> build_moi_runtime_workgroup_gate_call_words(
    std::span<const uint8_t> guest_bytes, const MoiRuntimeWorkgroupGatePlan &plan,
    const ConSanMoiWorkgroupSources &workgroup_sources, uint64_t gate_text_offset,
    uint64_t return_text_offset, uint64_t reserved_word_count, rj_code_arch_t arch);

} // namespace rocjitsu::consan_moi_impl
