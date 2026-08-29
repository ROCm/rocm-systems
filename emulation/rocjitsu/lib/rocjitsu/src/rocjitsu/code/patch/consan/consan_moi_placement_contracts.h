// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_placement_contracts.h
/// @brief Explicit contracts exported by MOI resource placement.

#pragma once

#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"

#include <memory>
#include <unordered_map>

namespace rocjitsu {
class AmdGpuCodeObject;
class CodeObjectPatcher;
} // namespace rocjitsu

namespace rocjitsu::consan_moi_impl {

/// Opaque decoded-CFG and liveness workspace owned by resource placement.
/// Engine components may retain and pass this workspace, but cannot inspect
/// placement's caches or owner analysis directly.
struct MoiResourcePlanningState;

struct MoiResourcePlanningStateDeleter {
  void operator()(MoiResourcePlanningState *state) const;
};

using MoiResourcePlanningStatePtr =
    std::unique_ptr<MoiResourcePlanningState, MoiResourcePlanningStateDeleter>;

inline constexpr uint32_t kMoiLocalIndirectIslandWords = 8u;
inline constexpr uint16_t kMoiDispatchStateSgprCount = 2u;

[[nodiscard]] constexpr bool moi_supports_dynamic_stack_spill(rj_code_arch_t arch,
                                                              ConSanMoiEngine engine) {
  if (engine == ConSanMoiEngine::InlineShadow)
    return true;
  return consan_is_capability_arch(arch) &&
         (engine == ConSanMoiEngine::RecordReplay || engine == ConSanMoiEngine::Sampled);
}

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

/// Owner-complete scratch allocation consumed by shared MOI lowering.
struct ResolvedMoiScratchPlan {
  uint16_t base = 0;
  uint16_t count = 0;
  uint16_t required_vgpr_count = 0;
  uint32_t original_private_segment_size = 0;
  std::vector<uint64_t> owner_descriptor_file_offsets;
  ConSanRegisterAllocationSource source = ConSanRegisterAllocationSource::Unsupported;
};

/// Entry-relay state retained after a caller reserves the appended host body.
struct MoiDenseEntryHost {
  uint64_t host_offset = 0;
  std::vector<uint32_t> displaced_words;
  uint64_t body_offset = 0;
};

[[nodiscard]] MoiResourcePlanningStatePtr make_moi_resource_planning_state(
    const MoiResourceProblem &problem, const ConSanMoiOperatingPoint &allocation,
    std::span<const ConSanCandidateResourcePlan> prior_site_plans = {});

[[nodiscard]] MoiResourcePlanningStatePtr make_moi_resource_planning_state(
    std::span<const uint8_t> input, rj_code_arch_t arch, const ProgramInventory &program_inventory,
    const ConSanObservationPlan &observation_plan,
    std::span<const ConSanCandidateResourcePlan> site_plans, const ConSanRequest &request,
    const BoundRuntimeResources &resources, const ConSanMoiOperatingPoint &point);

[[nodiscard]] ConSanCandidateResourcePlan
plan_moi_resource_site(MoiResourcePlanningState &state, const ConSanRequest &semantic_request,
                       const ConSanDebugOverrides &debug, const ConSanMoiOperatingPoint &point,
                       ConSanResourceSiteKind site_kind, size_t candidate_index,
                       uint64_t text_offset, std::optional<uint64_t> kernel_descriptor_file_offset,
                       uint16_t scratch_count, const ConSanMoiCandidate *access_candidate = nullptr,
                       const ConSanAtomicLoweringForm *atomic_form = nullptr,
                       bool require_spill = false);

[[nodiscard]] const ConSanCandidateResourcePlan *
resource_plan_for_site(std::span<const ConSanCandidateResourcePlan> plans,
                       ConSanResourceSiteKind site_kind, uint64_t text_offset);

[[nodiscard]] std::optional<ResolvedMoiScratchPlan>
resolve_moi_scratch_plan(const ConSanCandidateResourcePlan &plan,
                         const ConSanMoiOperatingPoint &site_point,
                         const ConSanMoiOperatingPoint &allocation, uint16_t expected_count);

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

[[nodiscard]] std::optional<ConSanMoiWorkgroupSources>
record_replay_persistent_workgroup_sources(ConSanMoiEngine engine,
                                           const ConSanMoiOperatingPoint &point);

[[nodiscard]] bool grow_moi_kernel_descriptor_vgprs(CodeObjectPatcher &patcher,
                                                    std::span<const uint8_t> image,
                                                    const ConSanKernelInfo &kernel,
                                                    uint32_t required_count, rj_code_arch_t arch,
                                                    std::vector<std::string> &errors);

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

[[nodiscard]] bool apply_record_replay_entry_workgroup_assignment(
    const ConSanRequest &request, ConSanMoiOperatingPoint &point,
    const ConSanMoiOperatingPoint &allocation, std::span<const uint64_t> owner_descriptor_offsets);

void note_descriptor_requirements(MoiDescriptorVgprRequirements &requirements,
                                  const ResolvedMoiScratchPlan &plan);

void note_moi_sgpr_requirements(MoiDescriptorSgprRequirements &requirements,
                                const ResolvedMoiScratchPlan &plan, const ConSanRequest &request,
                                const BoundRuntimeResources &resources,
                                const ConSanMoiOperatingPoint &point, rj_code_arch_t arch);

void note_spill_descriptor_requirements(MoiDescriptorPrivateRequirements &requirements,
                                        const ResolvedMoiScratchPlan &plan,
                                        const VgprSpillSequence &spill);

void append_nop_padding_to_alignment(std::vector<uint8_t> &bytes, uint64_t alignment,
                                     rj_code_arch_t arch);

[[nodiscard]] bool write_word_bytes(std::vector<uint8_t> &bytes, uint64_t offset, uint32_t word);

void note_dynamic_stack_private_requirement(ConSanPatchInfo &info, const VgprSpillSequence *spill);

[[nodiscard]] bool apply_spill_descriptor_requirements(
    CodeObjectPatcher &patcher, const AmdGpuCodeObject &code_object, std::span<const uint8_t> image,
    const ConSanTransformArtifacts &result, const MoiDescriptorPrivateRequirements &requirements,
    std::vector<std::string> &errors);

} // namespace rocjitsu::consan_moi_impl
