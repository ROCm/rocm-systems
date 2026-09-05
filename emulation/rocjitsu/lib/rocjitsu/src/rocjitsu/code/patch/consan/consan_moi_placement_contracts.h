// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_placement_contracts.h
/// @brief Explicit contracts exported by MOI resource placement.

#pragma once

#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_descriptor_growth.h"
#include "rocjitsu/code/patch/consan/consan_moi_candidate_projection.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/trampoline_builder.h"

#include <functional>
#include <map>
#include <memory>
#include <unordered_map>

namespace rocjitsu {
class AmdGpuCodeObject;
class CodeObjectPatcher;
class Decoder;
} // namespace rocjitsu

namespace rocjitsu::consan_moi_impl {

/// Opaque decoded-CFG and liveness workspace owned by resource placement.
/// Engine components may retain and pass this workspace, but cannot inspect
/// placement's caches or owner analysis directly.
struct MoiResourcePlanningState;
struct MoiCfgForwardDistanceIndex;

struct MoiResourcePlanningStateDeleter {
  void operator()(MoiResourcePlanningState *state) const;
};

using MoiResourcePlanningStatePtr =
    std::unique_ptr<MoiResourcePlanningState, MoiResourcePlanningStateDeleter>;

struct MoiCfgForwardDistanceIndexDeleter {
  void operator()(MoiCfgForwardDistanceIndex *index) const;
};

using MoiCfgForwardDistanceIndexPtr =
    std::unique_ptr<MoiCfgForwardDistanceIndex, MoiCfgForwardDistanceIndexDeleter>;

inline constexpr uint16_t kMoiDispatchStateSgprCount = 2u;

using MoiDescriptorVgprRequirements = ConSanDescriptorRegisterRequirements;
using MoiDescriptorSgprRequirements = ConSanDescriptorRegisterRequirements;
using MoiDescriptorPrivateRequirements = ConSanDescriptorMemoryRequirements;
using MoiDescriptorLdsRequirements = ConSanDescriptorMemoryRequirements;
using MoiSpillManagers = std::unordered_map<uint64_t, SpillManager>;

/// Exact owner-local assignments published by register allocation.  The
/// converting constructor lets callers project an accepted operating point at
/// an assignment-consuming boundary without exposing its unrelated policy.
struct MoiOwnerAssignments {
  std::span<const ConSanMoiPersistentVgprAssignment> persistent_vgprs;
  std::span<const ConSanMoiTransientSgprAssignment> transient_sgprs;

  MoiOwnerAssignments(const ConSanMoiOperatingPoint &point)
      : persistent_vgprs(point.owner_persistent_vgprs),
        transient_sgprs(point.owner_transient_sgprs) {}
};

/// Owner-complete scratch allocation consumed by shared MOI lowering.
struct ResolvedMoiScratchPlan {
  uint16_t base = 0;
  uint16_t count = 0;
  uint16_t required_vgpr_count = 0;
  uint32_t original_private_segment_size = 0;
  std::vector<uint64_t> owner_descriptor_file_offsets;
  ConSanRegisterAllocationSource source = ConSanRegisterAllocationSource::Unsupported;
};

/// Read-only scalar-preservation state projected from one accepted operating
/// point. Mode owners select their scalar ABI and spill behavior from this
/// value without inspecting unrelated vector, dispatch, or private-state
/// placement decisions.
struct MoiScalarPreservationState {
  std::optional<uint16_t> exec_save_sgpr;
  ConSanMoiScalarSpillLayout spill_layout = ConSanMoiScalarSpillLayout::None;
  std::optional<ConSanMoiScalarSpillSetup> scalar_spill_setup;
  bool has_branch_only_spill = false;

  [[nodiscard]] bool has_scalar_spill() const {
    return spill_layout != ConSanMoiScalarSpillLayout::None;
  }
  [[nodiscard]] bool has_inline_spill() const {
    return spill_layout == ConSanMoiScalarSpillLayout::Inline;
  }
  [[nodiscard]] bool has_compact_spill() const {
    return spill_layout == ConSanMoiScalarSpillLayout::Compact;
  }

  bool operator==(const MoiScalarPreservationState &) const = default;
};

[[nodiscard]] inline MoiScalarPreservationState
moi_scalar_preservation_state(const ConSanMoiOperatingPoint &point) {
  return {
      .exec_save_sgpr = point.moi_exec_save_sgpr,
      .spill_layout = point.automatic_moi_scalar_spill_layout,
      .scalar_spill_setup = point.moi_scalar_spill_setup,
      .has_branch_only_spill = point.moi_branch_only_spill.has_value(),
  };
}

/// A scalar register interval that a relocated host must preserve while it
/// establishes generated probe state. This is a placement constraint, not an
/// engine emission policy.
using MoiSgprRange = consan_detail::ScalarOwnerSgprRange;

/// Persistent VGPRs that cannot be borrowed by an entry relay before the
/// probe's ordinary spill transaction has run.
struct MoiPersistentVgprStateView {
  ConSanMoiOwnerEpochVgprSources owner_epoch;
  std::optional<uint16_t> workgroup_key;
  std::optional<uint16_t> dispatch_id;
  ConSanMoiPersistentWorkgroupRegisters exact_workgroup;

  template <typename Visitor>
  void for_each_range(Visitor &&visit, bool include_dispatch = true) const {
    visit(owner_epoch.owner, 1u);
    visit(owner_epoch.epoch, 1u);
    visit(workgroup_key, 1u);
    if (include_dispatch)
      visit(dispatch_id, 2u);
    for (std::optional<uint16_t> reg : exact_workgroup.values())
      visit(reg, 1u);
  }

  bool operator==(const MoiPersistentVgprStateView &) const = default;
};

[[nodiscard]] inline MoiPersistentVgprStateView
moi_persistent_vgpr_state_view(const ConSanMoiOperatingPoint &point) {
  return {
      .owner_epoch = moi_owner_epoch_vgpr_sources(point.moi_owner_epoch_vgprs),
      .workgroup_key = point.moi_workgroup_key_vgpr,
      .dispatch_id = point.moi_dispatch_identity.vgpr(),
      .exact_workgroup = point.moi_exact_workgroup_vgprs,
  };
}

[[nodiscard]] inline MoiPersistentVgprStateView
moi_persistent_vgpr_state_view(const ConSanMoiPersistentVgprAssignment &assignment) {
  return {
      .owner_epoch = {.owner = assignment.owner_epoch_vgprs.owner,
                      .epoch = assignment.owner_epoch_vgprs.epoch},
      .workgroup_key = assignment.workgroup_key_vgpr,
      .dispatch_id = assignment.dispatch_id_vgpr,
      .exact_workgroup = assignment.exact_workgroup_vgprs,
  };
}

struct MoiCfgDistance {
  uint32_t block_edges = 0;
  uint64_t text_distance = 0;

  auto operator<=>(const MoiCfgDistance &) const = default;
};

[[nodiscard]] std::vector<const ConSanMoiCandidate *>
order_moi_replay_access_candidates(std::span<const ConSanMoiCandidate> admitted,
                                   const ConSanObservationPlan &observation_plan,
                                   bool track_atomics);

[[nodiscard]] std::span<const ConSanPreappliedReservedRange>
moi_resource_reserved_ranges(const MoiResourcePlanningState &state);

/// Reserve a placement-owned text range unless an existing reservation
/// already covers its anchor. Returns true when a new reservation is added.
[[nodiscard]] bool moi_resource_reserve_range_if_uncovered(MoiResourcePlanningState &state,
                                                           ConSanPreappliedReservedRange range);

[[nodiscard]] Decoder *moi_resource_decoder(MoiResourcePlanningState &state);

[[nodiscard]] bool moi_resource_offsets_share_block(const MoiResourcePlanningState &state,
                                                    uint64_t first_offset, uint64_t second_offset);

[[nodiscard]] bool moi_resource_is_valid(const MoiResourcePlanningState &state);

[[nodiscard]] rj_code_arch_t moi_resource_arch(const MoiResourcePlanningState &state);

[[nodiscard]] bool moi_resource_has_owner_context(const MoiResourcePlanningState &state,
                                                  uint64_t descriptor_offset);

[[nodiscard]] MoiCfgForwardDistanceIndexPtr
make_moi_cfg_forward_distance_index(const MoiResourcePlanningState &state,
                                    uint64_t descriptor_offset, uint64_t target_text_offset);

[[nodiscard]] std::optional<MoiCfgDistance>
moi_cfg_forward_distance(const MoiCfgForwardDistanceIndex &index, uint64_t from_text_offset);

[[nodiscard]] const ConSanCandidateResourcePlan *
resource_plan_for_candidate(std::span<const ConSanCandidateResourcePlan> plans,
                            const ConSanMoiCandidate &candidate);

[[nodiscard]] bool
moi_transient_sgpr_assignment_is_branch_only(MoiOwnerAssignments assignments,
                                             std::span<const uint64_t> owner_descriptor_offsets);

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
                       uint64_t text_offset, std::optional<ConSanProgramSiteId> owner_site,
                       uint16_t scratch_count, const ConSanMoiCandidate *access_candidate = nullptr,
                       const ConSanAtomicLoweringForm *atomic_form = nullptr,
                       bool require_spill = false);

void append_moi_resource_plans(MoiResourcePlanningState &state, const ConSanRequest &request,
                               const BoundRuntimeResources &resources,
                               const ConSanDebugOverrides &debug,
                               const ConSanMoiOperatingPoint &point,
                               std::span<const ConSanMoiCandidate> candidates,
                               std::vector<ConSanCandidateResourcePlan> &plans);

[[nodiscard]] bool
moi_resource_owner_sgpr_window_admitted(const MoiResourcePlanningState &state,
                                        std::span<const ConSanProgramContainerId> owners,
                                        uint16_t base, uint16_t count);

[[nodiscard]] bool configure_automatic_moi_exec_save_sgprs(
    ConSanMoiOperatingPoint &options, const MoiResourceProblem &problem,
    std::span<const ConSanCandidateResourcePlan> site_plans, std::vector<std::string> &warnings,
    const MoiResourcePlanningState &planning_state);

[[nodiscard]] bool automatic_moi_scalar_spill_needs_dynamic_stack_planning(
    const ConSanMoiOperatingPoint &point, const ProgramInventory &inventory,
    std::span<const ConSanCandidateResourcePlan> site_plans);

[[nodiscard]] bool configure_automatic_moi_owner_sgpr(
    ConSanMoiOperatingPoint &point, const MoiResourceProblem &problem,
    std::span<const ConSanCandidateResourcePlan> site_plans, std::vector<std::string> &diagnostics,
    const MoiResourcePlanningState &state);

[[nodiscard]] bool configure_inline_moi_owner_sgpr(const ConSanRequest &request,
                                                   ConSanMoiOperatingPoint &point,
                                                   std::vector<std::string> &diagnostics);

[[nodiscard]] ConSanMoiOperatingPointUpdate configure_automatic_moi_dispatch_id_sgprs(
    const ConSanMoiOperatingPoint &base, const MoiResourceProblem &problem,
    std::span<const ConSanCandidateResourcePlan> site_plans, const MoiResourcePlanningState &state);

[[nodiscard]] ConSanMoiOperatingPointAttempt
plan_moi_dispatch_id_fallback(const ConSanRequest &request, const BoundRuntimeResources &resources,
                              const ConSanMoiOperatingPoint &base,
                              const MoiResourceProblem &problem,
                              std::span<const ConSanCandidateResourcePlan> site_plans);

[[nodiscard]] std::optional<std::string>
validate_moi_scalar_state(const ConSanRequest &request, const BoundRuntimeResources &resources,
                          const ConSanMoiOperatingPoint &point,
                          const MoiObjectModeSemantics &mode_semantics, rj_code_arch_t arch);

[[nodiscard]] std::optional<std::string>
validate_moi_dispatch_id_vgprs(const ConSanMoiOperatingPoint &point);

[[nodiscard]] ConSanMoiPersistentPlacementUpdate configure_automatic_moi_persistent_vgprs(
    const ConSanMoiOperatingPoint &base, const MoiResourceProblem &problem,
    const ConSanDebugOverrides &debug, std::span<const ConSanCandidateResourcePlan> resource_plans,
    const MoiResourcePlanningState &planning_state);

[[nodiscard]] const ConSanCandidateResourcePlan *
resource_plan_for_site(std::span<const ConSanCandidateResourcePlan> plans,
                       ConSanResourceSiteKind site_kind, uint64_t text_offset);

[[nodiscard]] std::optional<ResolvedMoiScratchPlan>
resolve_moi_scratch_plan(const ConSanCandidateResourcePlan &plan, const ProgramInventory &inventory,
                         const ConSanMoiOperatingPoint &site_point, MoiOwnerAssignments assignments,
                         uint16_t expected_count);

[[nodiscard]] std::optional<ResolvedMoiScratchPlan>
resolve_moi_scratch(std::span<const ConSanCandidateResourcePlan> plans,
                    const ProgramInventory &inventory, const ConSanMoiCandidate &candidate,
                    const ConSanMoiOperatingPoint &operating_point, uint16_t expected_count);

[[nodiscard]] const ConSanMoiPersistentVgprAssignment *
moi_persistent_vgpr_assignment(MoiOwnerAssignments assignments, uint64_t descriptor_offset);

[[nodiscard]] const ConSanMoiTransientSgprAssignment *
moi_transient_sgpr_assignment(MoiOwnerAssignments assignments, uint64_t descriptor_offset);

void note_moi_lds_requirements(MoiDescriptorLdsRequirements &requirements,
                               const ResolvedMoiScratchPlan &plan,
                               const ConSanMoiWorkgroupShadowLayout &layout);

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

[[nodiscard]] std::optional<ConSanMoiWorkgroupSources> moi_exact_entry_workgroup_sources(
    const ConSanMoiOperatingPoint &point,
    const ConSanMoiPersistentWorkgroupPrivateOffsets *private_offsets = nullptr);

[[nodiscard]] constexpr uint16_t moi_ordinary_sgpr_limit(rj_code_arch_t arch) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile ? profile->ordinary_sgpr_limit : kMaxSgprs;
}

[[nodiscard]] constexpr bool
persistent_sgpr_range_overlaps_reserved_ordinary_range(uint16_t base, uint16_t width,
                                                       rj_code_arch_t arch) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile && consan_profile_reserved_sgpr_range_overlaps(*profile, base, width);
}

[[nodiscard]] bool
apply_moi_transient_sgpr_assignment(const ConSanRequest &request, ConSanMoiOperatingPoint &point,
                                    MoiOwnerAssignments assignments,
                                    std::span<const uint64_t> owner_descriptor_offsets);

[[nodiscard]] bool
apply_moi_persistent_vgpr_assignment(ConSanMoiOperatingPoint &point,
                                     MoiOwnerAssignments assignments,
                                     std::span<const uint64_t> owner_descriptor_offsets);

void note_descriptor_requirements(MoiDescriptorVgprRequirements &requirements,
                                  const ResolvedMoiScratchPlan &plan);

void note_moi_sgpr_requirements(MoiDescriptorSgprRequirements &requirements,
                                const ResolvedMoiScratchPlan &plan, const ConSanRequest &request,
                                const BoundRuntimeResources &resources,
                                const ConSanMoiOperatingPoint &point,
                                const MoiObjectModeSemantics &mode_semantics, rj_code_arch_t arch);

void note_spill_descriptor_requirements(MoiDescriptorPrivateRequirements &requirements,
                                        const ResolvedMoiScratchPlan &plan,
                                        const VgprSpillSequence &spill);

void append_nop_padding_to_alignment(std::vector<uint8_t> &bytes, uint64_t alignment,
                                     rj_code_arch_t arch);

[[nodiscard]] bool write_word_bytes(std::vector<uint8_t> &bytes, uint64_t offset, uint32_t word);

void note_dynamic_stack_private_requirement(ConSanPatchAbiEffects &effects,
                                            const VgprSpillSequence *spill);

} // namespace rocjitsu::consan_moi_impl
