// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_mode_planning.h
/// @brief Target-neutral per-object demand selected by one MOI engine.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"

#include <cstddef>
#include <string>
#include <vector>

namespace rocjitsu::consan_moi_impl {

/// Shared inventory facts presented to every MOI engine. Target inspection is
/// normalized before this boundary; mode owners never decode an ISA family.
struct MoiObjectFacts {
  bool has_access_candidate = false;
  bool has_admitted_atomic = false;
  bool has_admitted_fence = false;
  bool has_admitted_barrier = false;
  size_t admitted_barrier_count = 0;
  bool has_stranded_admitted_barrier = false;
  bool target_supports_dense_barrier_router = false;
  bool has_explicit_persistent_state = false;
  bool has_report_buffer = false;
};

/// Effective mode demand consumed by common resource solving. This deliberately
/// contains only the request/operating-point fields that an engine may refine
/// after seeing one object's semantic inventory.
struct MoiObjectModePlan {
  ConSanMoiOwnerSource owner_source = ConSanMoiOwnerSource::Automatic;
  bool track_atomics = false;
  bool track_barriers = false;
  std::optional<bool> initialize_owner_epoch;
  bool dense_barrier_router = false;
  bool inline_access_present = false;
  bool inline_atomic_without_access = false;
  bool atomic_or_fence_relevant = false;
  bool reserve_dynamic_stack_prologue_entry = false;
  bool prologue_requires_consumer = false;
  std::vector<std::string> warnings;
  std::vector<std::string> errors;
};

/// Mode-neutral summary of operational sites that survived semantic admission
/// and resource planning. Mode owners decide which persistent state those
/// consumers require; common placement only solves the resulting demand.
struct MoiPersistentStateFacts {
  size_t access_count = 0;
  size_t atomic_count = 0;
  size_t barrier_count = 0;
  size_t fence_count = 0;
  bool has_operational_dynamic_stack_owner = false;
  bool has_operational_dynamic_lds_owner = false;
};

struct MoiPersistentStateDemand {
  bool needs_workgroup_key = false;
  bool needs_entry_workgroup_tuple = false;
  bool needs_persistent_state = false;
  bool synchronization_requires_persistent_owner = false;
  bool needs_persistent_dispatch_capture = false;
  bool prefer_compact_barriers = false;
  bool private_workgroup_tuple_supported = false;
  bool private_dispatch_incompatible_with_dynamic_stack = false;
  bool prefer_private_epoch_for_dynamic_lds = false;
  bool scalar_state_supported = false;
  bool private_state_supported = false;
  bool scalar_state_required_for_private_or_overflow = false;
  bool prefer_private_epoch_for_descriptor_growth = false;
};

/// Target-neutral input to the mode-owned dynamic-stack spill policy.
struct MoiDynamicStackSpillFacts {
  bool target_has_backend = false;
};

/// Mode-owned contract for a site whose ordinary scratch allocation spills
/// through a runtime-sized private frame. Target inspection is reduced to the
/// common capability profile before modes publish this policy.
struct MoiDynamicStackSpillPolicy {
  bool backend_supported = false;
  /// Record/Replay and Sampled use one frame recipe shared by all site owners.
  /// Inline can defer a mixed-owner rejection to its per-owner emission path.
  bool requires_every_owner_dynamic = true;

  bool operator==(const MoiDynamicStackSpillPolicy &) const = default;
};

/// Mode-owned permission to retry an otherwise unplaceable site by spilling a
/// scratch window that overlaps short-lived guest operands. Common placement
/// executes the retry; the selected mode owns whether its emission ordering
/// can recover the operands and which guest result must remain disjoint.
struct MoiOperandOverlapSpillPolicy {
  bool supported = false;
  std::optional<uint16_t> protected_vgpr;
  uint8_t protected_vgpr_count = 0;
};

struct MoiOperandOverlapSpillContext {
  const ConSanRequest &request;
  const ConSanMoiOperatingPoint &point;
  const ConSanMoiCandidate *access_candidate = nullptr;
  ConSanResourceSiteKind site_kind = ConSanResourceSiteKind::Access;
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  bool guest_replay_requires_disjoint_address_scratch = false;
};

/// Shared Record/Replay + Sampled entry-identity lifetime rule.
[[nodiscard]] MoiPersistentStateDemand make_exact_workgroup_capture_demand(
    const ConSanRequest &request, const BoundRuntimeResources &resources,
    const ConSanMoiOperatingPoint &point, const MoiPersistentStateFacts &facts);

[[nodiscard]] MoiObjectModePlan
make_moi_object_mode_plan(const ConSanRequest &request, const ConSanMoiOperatingPoint &point,
                          ConSanMoiOwnerSource automatic_owner_source);

/// The only cross-engine selection point for per-object demand. Adding an MOI
/// engine requires one engine-owned planner and one entry in this dispatcher;
/// common resource solving consumes the same narrow product unchanged.
[[nodiscard]] MoiObjectModePlan plan_moi_object_mode(const ConSanRequest &request,
                                                     const ConSanMoiOperatingPoint &point,
                                                     const MoiObjectFacts &facts,
                                                     const ConSanObservationPlan &observation_plan);

/// Run the selected engine's lowering sequence. Shared placement has already
/// accepted an operating point; the engine owns which access and sync
/// consumers run, their order, and any mode-local post-placement cleanup.
void apply_moi_mode_patches(std::span<const uint8_t> bytes, MoiOptions &options,
                            rj_code_arch_t arch, MoiResourcePlanningState &resource_state,
                            std::span<const ConSanMoiCandidate> candidates,
                            const MoiObjectFacts &facts, ConSanTransformArtifacts &result);

[[nodiscard]] MoiPersistentStateDemand plan_moi_persistent_state_demand(
    const ConSanRequest &request, const BoundRuntimeResources &resources,
    const ConSanMoiOperatingPoint &point, const MoiPersistentStateFacts &facts);

[[nodiscard]] MoiDynamicStackSpillPolicy plan_moi_dynamic_stack_spill(ConSanMoiEngine engine,
                                                                      rj_code_arch_t arch);

[[nodiscard]] MoiOperandOverlapSpillPolicy
plan_moi_operand_overlap_spill(const MoiOperandOverlapSpillContext &context);

struct MoiModeOperations {
  MoiObjectModePlan (*plan)(const ConSanRequest &, const ConSanMoiOperatingPoint &,
                            const MoiObjectFacts &, const ConSanObservationPlan &);
  void (*apply)(std::span<const uint8_t>, MoiOptions &, rj_code_arch_t, MoiResourcePlanningState &,
                std::span<const ConSanMoiCandidate>, const MoiObjectFacts &,
                ConSanTransformArtifacts &);
  uint16_t (*access_scratch_vgpr_count)(const ConSanRequest &, const BoundRuntimeResources &,
                                        const ConSanMoiOperatingPoint &, const ConSanMoiCandidate &,
                                        rj_code_arch_t);
  MoiPersistentStateDemand (*persistent_state_demand)(const ConSanRequest &,
                                                      const BoundRuntimeResources &,
                                                      const ConSanMoiOperatingPoint &,
                                                      const MoiPersistentStateFacts &);
  MoiDynamicStackSpillPolicy (*dynamic_stack_spill)(const MoiDynamicStackSpillFacts &);
  MoiOperandOverlapSpillPolicy (*operand_overlap_spill)(const MoiOperandOverlapSpillContext &);
};

extern const MoiModeOperations kRecordReplayModeOperations;
extern const MoiModeOperations kSampledModeOperations;
extern const MoiModeOperations kInlineShadowModeOperations;

[[nodiscard]] const MoiModeOperations &moi_mode_operations(ConSanMoiEngine engine);

} // namespace rocjitsu::consan_moi_impl
