// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_register_allocation.h
/// @brief Explicit contracts exported by ConSan resource placement.

#pragma once

#include "rocjitsu/code/patch/consan/consan_candidate_projection.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_descriptor_growth.h"
#include "rocjitsu/code/patch/consan/consan_internal.h"
#include "rocjitsu/code/patch/consan/consan_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_relocation.h"
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

namespace rocjitsu::consan {} // namespace rocjitsu::consan

namespace rocjitsu::consan::detail {

/// Opaque decoded-CFG and liveness workspace owned by resource placement.
/// Mode components may retain and pass this workspace, but cannot inspect
/// placement's caches or owner analysis directly.
struct ResourcePlanningState;
struct CfgForwardDistanceIndex;

struct ResourcePlanningStateDeleter {
  void operator()(ResourcePlanningState *state) const;
};

using ResourcePlanningStatePtr =
    std::unique_ptr<ResourcePlanningState, ResourcePlanningStateDeleter>;

struct CfgForwardDistanceIndexDeleter {
  void operator()(CfgForwardDistanceIndex *index) const;
};

using CfgForwardDistanceIndexPtr =
    std::unique_ptr<CfgForwardDistanceIndex, CfgForwardDistanceIndexDeleter>;

inline constexpr uint16_t kDispatchStateSgprCount = 2u;

using DescriptorVgprRequirements = DescriptorRegisterRequirements;
using DescriptorSgprRequirements = DescriptorRegisterRequirements;
using DescriptorPrivateRequirements = DescriptorMemoryRequirements;
using DescriptorLdsRequirements = DescriptorMemoryRequirements;
using SpillManagers = std::unordered_map<uint64_t, SpillManager>;

/// Exact owner-local assignments published by register allocation.  The
/// converting constructor lets callers project an accepted operating point at
/// an assignment-consuming boundary without exposing its unrelated policy.
struct OwnerAssignments {
  std::span<const PersistentVgprAssignment> persistent_vgprs;
  std::span<const TransientSgprAssignment> transient_sgprs;

  OwnerAssignments(const OperatingPoint &point)
      : persistent_vgprs(point.owner_persistent_vgprs),
        transient_sgprs(point.owner_transient_sgprs) {}
};

/// Owner-complete scratch allocation consumed by shared ConSan lowering.
struct ResolvedScratchPlan {
  uint16_t base = 0;
  uint16_t count = 0;
  uint16_t required_vgpr_count = 0;
  uint32_t original_private_segment_size = 0;
  std::vector<uint64_t> owner_descriptor_file_offsets;
  RegisterAllocationSource source = RegisterAllocationSource::Unsupported;
};

/// Read-only scalar-preservation state projected from one accepted operating
/// point. ConSan selects its scalar ABI and spill behavior from this
/// value without inspecting unrelated vector, dispatch, or private-state
/// placement decisions.
struct ScalarPreservationState {
  std::optional<uint16_t> exec_save_sgpr;
  ScalarSpillLayout spill_layout = ScalarSpillLayout::None;
  std::optional<ScalarSpillSetup> scalar_spill_setup;
  bool has_branch_only_spill = false;

  [[nodiscard]] bool has_scalar_spill() const { return spill_layout != ScalarSpillLayout::None; }

  [[nodiscard]] bool has_compact_spill() const {
    return spill_layout == ScalarSpillLayout::Compact;
  }

  bool operator==(const ScalarPreservationState &) const = default;
};

[[nodiscard]] inline ScalarPreservationState
scalar_preservation_state(const OperatingPoint &point) {
  return {
      .exec_save_sgpr = point.exec_save_sgpr,
      .spill_layout = point.automatic_scalar_spill_layout,
      .scalar_spill_setup = point.scalar_spill_setup,
      .has_branch_only_spill = point.branch_only_spill.has_value(),
  };
}

/// A scalar register interval that a relocated host must preserve while it
/// establishes generated probe state. This is a placement constraint, not an
/// mode emission policy.
using SgprRange = detail::ScalarOwnerSgprRange;

/// Persistent VGPRs that cannot be borrowed by an entry relay before the
/// probe's ordinary spill transaction has run.
struct PersistentVgprStateView {
  OwnerEpochVgprSources owner_epoch;
  PersistentWorkgroupRegisters exact_workgroup;

  template <typename Visitor> void for_each_range(Visitor &&visit) const {
    visit(owner_epoch.owner, 1u);
    visit(owner_epoch.epoch, 1u);
    for (std::optional<uint16_t> reg : exact_workgroup.values())
      visit(reg, 1u);
  }

  bool operator==(const PersistentVgprStateView &) const = default;
};

[[nodiscard]] inline PersistentVgprStateView
persistent_vgpr_state_view(const OperatingPoint &point) {
  return {
      .owner_epoch = owner_epoch_vgpr_sources(point.owner_epoch_vgprs),
      .exact_workgroup = point.exact_workgroup_vgprs,
  };
}

[[nodiscard]] inline PersistentVgprStateView
persistent_vgpr_state_view(const PersistentVgprAssignment &assignment) {
  return {
      .owner_epoch = {.owner = assignment.owner_epoch_vgprs.owner,
                      .epoch = assignment.owner_epoch_vgprs.epoch},
      .exact_workgroup = assignment.exact_workgroup_vgprs,
  };
}

struct CfgDistance {
  uint32_t block_edges = 0;
  uint64_t text_distance = 0;

  auto operator<=>(const CfgDistance &) const = default;
};

[[nodiscard]] std::vector<const Candidate *>
order_replay_access_candidates(std::span<const Candidate> admitted,
                               const ObservationPlan &observation_plan, bool track_atomics);

/// Reserve a placement-owned text range unless an existing reservation
/// already covers its anchor. Returns true when a new reservation is added.
[[nodiscard]] bool resource_reserve_range_if_uncovered(ResourcePlanningState &state,
                                                       PreappliedReservedRange range);

[[nodiscard]] std::optional<uint64_t>
resource_scalar_clause_at_offset(const ResourcePlanningState &state, uint64_t text_offset);

[[nodiscard]] bool resource_offsets_share_block(const ResourcePlanningState &state,
                                                uint64_t first_offset, uint64_t second_offset);

[[nodiscard]] bool resource_is_valid(const ResourcePlanningState &state);

[[nodiscard]] rj_code_arch_t resource_arch(const ResourcePlanningState &state);

[[nodiscard]] bool resource_has_owner_context(const ResourcePlanningState &state,
                                              uint64_t descriptor_offset);

[[nodiscard]] CfgForwardDistanceIndexPtr
make_cfg_forward_distance_index(const ResourcePlanningState &state, uint64_t descriptor_offset,
                                uint64_t target_text_offset);

[[nodiscard]] std::optional<CfgDistance> cfg_forward_distance(const CfgForwardDistanceIndex &index,
                                                              uint64_t from_text_offset);

[[nodiscard]] const CandidateResourcePlan *
resource_plan_for_candidate(std::span<const CandidateResourcePlan> plans,
                            const Candidate &candidate);

[[nodiscard]] ResourcePlanningStatePtr
make_resource_planning_state(const ResourceProblem &problem, const OperatingPoint &allocation,
                             std::span<const CandidateResourcePlan> prior_site_plans = {});

[[nodiscard]] ResourcePlanningStatePtr make_resource_planning_state(
    std::span<const uint8_t> input, rj_code_arch_t arch, const ProgramInventory &program_inventory,
    const ObservationPlan &observation_plan, std::span<const CandidateResourcePlan> site_plans,
    const Request &request, const BoundRuntimeResources &resources, const OperatingPoint &point);

[[nodiscard]] CandidateResourcePlan
plan_resource_site(ResourcePlanningState &state, const Request &semantic_request,
                   const DebugOverrides &debug, const OperatingPoint &point,
                   ResourceSiteKind site_kind, size_t candidate_index, uint64_t text_offset,
                   std::optional<ProgramSiteId> owner_site, uint16_t scratch_count,
                   const Candidate *access_candidate = nullptr,
                   const AtomicLoweringForm *atomic_form = nullptr, bool require_spill = false);

void append_resource_plans(ResourcePlanningState &state, const Request &request,
                           const DebugOverrides &debug, const OperatingPoint &point,
                           std::span<const Candidate> candidates,
                           std::vector<CandidateResourcePlan> &plans);

[[nodiscard]] bool resource_owner_sgpr_window_admitted(const ResourcePlanningState &state,
                                                       std::span<const ProgramContainerId> owners,
                                                       uint16_t base, uint16_t count,
                                                       uint32_t required_sgpr_count_floor);

[[nodiscard]] uint32_t required_sgpr_count_floor(const Request &request,
                                                 const BoundRuntimeResources &resources,
                                                 const OperatingPoint &point, rj_code_arch_t arch);

[[nodiscard]] bool
configure_automatic_exec_save_sgprs(OperatingPoint &options, const ResourceProblem &problem,
                                    std::span<const CandidateResourcePlan> site_plans,
                                    std::vector<std::string> &warnings,
                                    const ResourcePlanningState &planning_state);

[[nodiscard]] bool automatic_scalar_spill_needs_dynamic_stack_planning(
    const OperatingPoint &point, const ProgramInventory &inventory,
    std::span<const CandidateResourcePlan> site_plans);

[[nodiscard]] bool configure_automatic_owner_sgpr(OperatingPoint &point,
                                                  const ResourceProblem &problem,
                                                  std::span<const CandidateResourcePlan> site_plans,
                                                  std::vector<std::string> &diagnostics,
                                                  const ResourcePlanningState &state);

[[nodiscard]] OperatingPointUpdate
configure_automatic_dispatch_id_sgprs(const OperatingPoint &base, const ResourceProblem &problem,
                                      std::span<const CandidateResourcePlan> site_plans,
                                      const ResourcePlanningState &state);

[[nodiscard]] OperatingPointAttempt
plan_dispatch_id_fallback(const Request &request, const BoundRuntimeResources &resources,
                          const OperatingPoint &base, const ResourceProblem &problem,
                          std::span<const CandidateResourcePlan> site_plans);

[[nodiscard]] std::optional<std::string>
validate_scalar_state(const Request &request, const BoundRuntimeResources &resources,
                      const OperatingPoint &point, rj_code_arch_t arch);

[[nodiscard]] PersistentPlacementUpdate
configure_automatic_persistent_vgprs(const OperatingPoint &base, const ResourceProblem &problem,
                                     const DebugOverrides &debug,
                                     std::span<const CandidateResourcePlan> resource_plans,
                                     const ResourcePlanningState &planning_state);

[[nodiscard]] const CandidateResourcePlan *
resource_plan_for_site(std::span<const CandidateResourcePlan> plans, ResourceSiteKind site_kind,
                       uint64_t text_offset);

[[nodiscard]] std::optional<ResolvedScratchPlan>
resolve_scratch_plan(const CandidateResourcePlan &plan, const ProgramInventory &inventory,
                     const OperatingPoint &site_point, OwnerAssignments assignments,
                     uint16_t expected_count);

[[nodiscard]] std::optional<ResolvedScratchPlan>
resolve_scratch(std::span<const CandidateResourcePlan> plans, const ProgramInventory &inventory,
                const Candidate &candidate, const OperatingPoint &operating_point,
                uint16_t expected_count);

[[nodiscard]] const PersistentVgprAssignment *
persistent_vgpr_assignment(OwnerAssignments assignments, uint64_t descriptor_offset);

[[nodiscard]] const TransientSgprAssignment *transient_sgpr_assignment(OwnerAssignments assignments,
                                                                       uint64_t descriptor_offset);

[[nodiscard]] uint32_t descriptor_user_sgpr_count(const KD &descriptor);
[[nodiscard]] uint16_t descriptor_system_sgpr_count(const KD &descriptor);
[[nodiscard]] DispatchIdPreloadPlan descriptor_dispatch_id_preload_plan(const KD &descriptor,
                                                                        rj_code_arch_t arch);

[[nodiscard]] std::optional<uint8_t> descriptor_workitem_id_dimensions(const KD &descriptor);

[[nodiscard]] std::optional<WorkgroupSources> descriptor_workgroup_sources(
    std::span<const uint8_t> image, uint64_t descriptor_file_offset, rj_code_arch_t arch,
    std::vector<std::string> &errors, bool uses_cluster_workgroup_id = false,
    std::optional<uint16_t> cdna_full_payload_user_sgpr_count = std::nullopt);

[[nodiscard]] std::optional<WorkgroupSources>
exact_entry_workgroup_sources(const OperatingPoint &point,
                              const PersistentWorkgroupPrivateOffsets *private_offsets = nullptr);

[[nodiscard]] inline uint16_t ordinary_sgpr_limit(rj_code_arch_t arch) {
  const TargetProfile *profile = target_profile(arch);
  return profile ? profile->ordinary_sgpr_limit : kMaxSgprs;
}

[[nodiscard]] inline bool
persistent_sgpr_range_overlaps_reserved_ordinary_range(uint16_t base, uint16_t width,
                                                       rj_code_arch_t arch) {
  const TargetProfile *profile = target_profile(arch);
  return profile && profile_reserved_sgpr_range_overlaps(*profile, base, width);
}

[[nodiscard]] bool
apply_transient_sgpr_assignment(const Request &request, OperatingPoint &point,
                                OwnerAssignments assignments,
                                std::span<const uint64_t> owner_descriptor_offsets);

[[nodiscard]] bool
apply_persistent_vgpr_assignment(OperatingPoint &point, OwnerAssignments assignments,
                                 std::span<const uint64_t> owner_descriptor_offsets);

void note_descriptor_requirements(DescriptorVgprRequirements &requirements,
                                  const ResolvedScratchPlan &plan);

void note_sgpr_requirements(DescriptorSgprRequirements &requirements,
                            const ResolvedScratchPlan &plan, const Request &request,
                            const BoundRuntimeResources &resources, const OperatingPoint &point,
                            rj_code_arch_t arch);

void note_spill_descriptor_requirements(DescriptorPrivateRequirements &requirements,
                                        const ResolvedScratchPlan &plan,
                                        const VgprSpillSequence &spill);

void note_dynamic_stack_private_requirement(PatchAbiEffects &effects,
                                            const VgprSpillSequence *spill);

} // namespace rocjitsu::consan::detail
