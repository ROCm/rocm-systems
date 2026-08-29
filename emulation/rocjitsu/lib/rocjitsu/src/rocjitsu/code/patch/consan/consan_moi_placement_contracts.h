// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_placement_contracts.h
/// @brief Explicit contracts exported by MOI resource placement.

#pragma once

#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"

#include <unordered_map>

namespace rocjitsu {
class AmdGpuCodeObject;
class CodeObjectPatcher;
} // namespace rocjitsu

namespace rocjitsu::consan_moi_impl {

inline constexpr uint32_t kMoiLocalIndirectIslandWords = 8u;
inline constexpr uint16_t kMoiDispatchStateSgprCount = 2u;

[[nodiscard]] constexpr uint32_t moi_dense_entry_island_words(bool derive_key_at_entry,
                                                              bool key_encodes_scc = false) {
  // Relocatable dense routers derive the dispatcher key at entry. The SCC
  // snapshot can share the key scalar under pressure, adding one carrier word.
  return derive_key_at_entry ? 6u + (key_encodes_scc ? 1u : 0u) + (7u - 1u) : 7u;
}

using MoiDescriptorVgprRequirements = std::unordered_map<uint64_t, uint16_t>;
using MoiDescriptorSgprRequirements = std::unordered_map<uint64_t, uint16_t>;
using MoiDescriptorPrivateRequirements = std::unordered_map<uint64_t, uint32_t>;
using MoiDescriptorLdsRequirements = std::unordered_map<uint64_t, uint32_t>;
using MoiSpillManagers = std::unordered_map<uint64_t, SpillManager>;

[[nodiscard]] uint32_t moi_descriptor_user_sgpr_count(const KD &descriptor);
[[nodiscard]] uint16_t moi_descriptor_system_sgpr_count(const KD &descriptor);
[[nodiscard]] bool moi_descriptor_has_kernarg_preload(const KD &descriptor);
[[nodiscard]] ConSanMoiDispatchIdPreloadPlan
moi_descriptor_dispatch_id_preload_plan(const KD &descriptor, rj_code_arch_t arch);

[[nodiscard]] std::optional<uint8_t> moi_descriptor_workitem_id_dimensions(const KD &descriptor);

[[nodiscard]] std::optional<ConSanMoiWorkgroupSources> moi_descriptor_workgroup_sources(
    std::span<const uint8_t> image, uint64_t descriptor_file_offset, rj_code_arch_t arch,
    std::vector<std::string> &errors, bool uses_cluster_workgroup_id = false,
    std::optional<uint16_t> cdna_full_payload_user_sgpr_count = std::nullopt);

[[nodiscard]] constexpr uint16_t moi_ordinary_sgpr_limit(rj_code_arch_t arch) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile ? profile->ordinary_sgpr_limit : kMaxSgprs;
}

[[nodiscard]] bool
apply_moi_transient_sgpr_assignment(const ConSanRequest &request, ConSanMoiOperatingPoint &point,
                                    const ConSanMoiOperatingPoint &allocation,
                                    std::span<const uint64_t> owner_descriptor_offsets);

[[nodiscard]] bool
apply_moi_persistent_vgpr_assignment(ConSanMoiOperatingPoint &point,
                                     const ConSanMoiOperatingPoint &allocation,
                                     std::span<const uint64_t> owner_descriptor_offsets);

void append_nop_padding_to_alignment(std::vector<uint8_t> &bytes, uint64_t alignment,
                                     rj_code_arch_t arch);

[[nodiscard]] bool write_word_bytes(std::vector<uint8_t> &bytes, uint64_t offset, uint32_t word);

void note_dynamic_stack_private_requirement(ConSanPatchInfo &info, const VgprSpillSequence *spill);

[[nodiscard]] bool apply_spill_descriptor_requirements(
    CodeObjectPatcher &patcher, const AmdGpuCodeObject &code_object, std::span<const uint8_t> image,
    const ConSanTransformArtifacts &result, const MoiDescriptorPrivateRequirements &requirements,
    std::vector<std::string> &errors);

} // namespace rocjitsu::consan_moi_impl
