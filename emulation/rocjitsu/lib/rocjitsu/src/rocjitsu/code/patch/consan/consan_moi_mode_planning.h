// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_mode_planning.h
/// @brief Target-neutral per-object demand selected by one MOI engine.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::consan_moi_impl {

/// Shared inventory facts presented to every MOI engine. Target inspection is
/// normalized before this boundary; mode owners never decode an ISA family.
struct MoiObjectFacts {
  bool has_access_candidate = false;
  bool has_admitted_atomic = false;
  size_t admitted_atomic_count = 0;
  bool has_admitted_fence = false;
  bool has_admitted_barrier = false;
  size_t admitted_barrier_count = 0;
  bool has_stranded_admitted_barrier = false;
  bool target_supports_dense_barrier_router = false;
  bool has_explicit_persistent_state = false;
  bool has_report_buffer = false;
};

/// Mode semantics consumed by the one shared owner/epoch prologue builder.
/// The builder owns descriptor inspection, resource preservation, placement,
/// and emission; modes own only the meaning that changes those mechanics.
struct MoiPrologueModePolicy {
  bool skip_unobserved_barrier_only_initialization = false;
  bool backup_compact_spill_for_runtime_sampling = false;
  bool one_based_owner_ids = false;
  bool persistent_state_requires_in_place_entry = false;
};

/// Effective mode demand consumed by common resource solving. This deliberately
/// contains only the request/operating-point fields that an engine may refine
/// after seeing one object's semantic inventory.
struct MoiObjectModePlan {
  ConSanMoiOwnerSource owner_source = ConSanMoiOwnerSource::Automatic;
  bool track_atomics = false;
  bool track_barriers = false;
  bool initialize_owner_epoch = false;
  MoiObjectModeSemantics semantics;
  MoiPrologueModePolicy prologue;
  bool inline_atomic_without_access = false;
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

enum class MoiCdnaPersistentOverflowStrategy : uint8_t {
  Unsupported,
  ExactWorkgroupState,
  OwnerSnapshot,
  ResidentWavePrivateState,
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
  MoiCdnaPersistentOverflowStrategy cdna_overflow_strategy =
      MoiCdnaPersistentOverflowStrategy::Unsupported;
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
  const MoiAccessResourceFacts &resource_facts;
  const ConSanMoiCandidate *access_candidate = nullptr;
  ConSanResourceSiteKind site_kind = ConSanResourceSiteKind::Access;
};

/// Mode-owned request for a smaller spill-backed transaction after ordinary
/// access placement. Common placement retries with the returned scratch size.
struct MoiAccessSpillFallbackContext {
  const ConSanRequest &request;
  const MoiAccessResourceFacts &resource_facts;
  const ConSanMoiCandidate &candidate;
  bool no_ordinary_window = false;
  bool spill_required = false;
  bool initial_spill_overlaps_guest = false;
};

/// Normalized facts used by a mode to declare its dispatch-identity demand.
/// Target-family inspection and site traversal remain common solver work.
struct MoiDispatchIdentityFacts {
  bool access_reports_need_explicit_identity = true;
  bool has_access_or_atomic_consumer = false;
};

/// Mode-owned dispatch-identity policy consumed by common placement. The
/// solver owns register search and retry; the mode owns whether identity is
/// semantically required and which lossless fallback it permits.
struct MoiDispatchIdentityPlan {
  bool needs_dispatch_id = false;
  bool permits_private_entry_capture = false;
  std::optional<ConSanMoiFallbackKind> fallback_kind;
  bool fallback_replans_dispatch_only = false;
  std::string_view fallback_diagnostic;
};

/// Complete scalar preservation ABI selected by one mode. The common builder
/// handles spill-backed indirect state; modes supply their fixed layout or
/// publication-derived special-state registers.
struct MoiScalarAbiPlan {
  std::optional<consan_detail::MoiSpecialStateSgprs> special_state;
  std::optional<ConSanMoiIndirectJumpSgprs> indirect_jump;
  bool access_router_uses_dense_abi = false;
};

/// Mode-neutral inputs from the transform pipeline to evidence planning.
/// Modes translate these caller and target bounds into their own typed
/// capacity policies beside their report implementation.
struct MoiEvidencePlanningContext {
  const ProgramInventory &program_inventory;
  const ConSanEvidenceIntentPlan &evidence_intents;
  uint64_t requested_report_buffer_size = 0;
  std::optional<uint64_t> maximum_access_probe_count;
  std::optional<uint32_t> maximum_workgroup_lds_bytes;
  bool dynamic_access_records = false;
};

/// Mode-owned representation selected when no code-object-wide transient
/// scalar window is legal. Common placement owns the register search; a mode
/// declares which spill ABI it can emit and any exact constraints of that ABI.
struct MoiTransientScalarPlacementTraits {
  ConSanMoiScalarSpillLayout spill_layout = ConSanMoiScalarSpillLayout::None;
  bool requires_capability_target = false;
  bool branch_only_spill_preserves_indirect_state = false;
  /// Zero retains the ordinary transient ABI width. Sampled's compact private
  /// frame has a fixed eight-scalar representation.
  uint16_t compact_spill_scalar_count = 0u;
};

/// Semantic synchronization operations selected by one mode for common
/// resource solving. A new mode may compose existing evidence operations
/// without adding mode branches to the solver.
struct MoiOperationalEvidenceKinds {
  ConSanProbeIntentKind barrier = ConSanProbeIntentKind::Count;
  ConSanProbeIntentKind atomic = ConSanProbeIntentKind::Count;
  bool fence = false;
};

/// Exact target-neutral facts used by a mode to size one barrier probe. The
/// common resource solver projects these facts but does not interpret them.
struct MoiBarrierScratchFacts {
  bool has_report_buffer = false;
  bool inline_access_present = false;
  bool automatic_private_epoch = false;
  bool persistent_scalar_state_complete = false;
};

[[nodiscard]] inline MoiBarrierScratchFacts
project_moi_barrier_scratch_facts(const BoundRuntimeResources &resources,
                                  const ConSanMoiOperatingPoint &point,
                                  const MoiObjectModeSemantics &semantics) {
  return {resources.moi_report_buffer_address.has_value(), semantics.inline_access_present,
          point.automatic_moi_private_epoch, point.moi_persistent_sgprs.complete()};
}

/// Narrow target facts that affect a mode's transient scalar ABI and dense
/// routing. Modes see the semantic facility, not an architecture identity or
/// complete capability profile; adding a target therefore does not require
/// editing a mode.
struct MoiScalarTargetFacts {
  ConSanDirectCallForm direct_call_form = ConSanDirectCallForm::SCallB64;
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
[[nodiscard]] MoiObjectModePlan
plan_moi_object_mode(const ConSanRequest &request, const BoundRuntimeResources &resources,
                     const TransformPolicy &policy, const ConSanMoiOperatingPoint &point,
                     const MoiObjectFacts &facts, const ConSanObservationPlan &observation_plan);

/// Run the selected engine's lowering sequence. Shared placement has already
/// accepted an operating point; the engine owns which access and sync
/// consumers run, their order, and any mode-local post-placement cleanup.
void apply_moi_mode_patches(std::span<const uint8_t> bytes, const ConSanOptions &options,
                            ConSanMoiOperatingPoint &operating_point, rj_code_arch_t arch,
                            MoiResourcePlanningState &resource_state,
                            std::span<const ConSanMoiCandidate> candidates,
                            const MoiObjectFacts &facts, const MoiObjectModeSemantics &semantics,
                            ConSanTransformArtifacts &result);

[[nodiscard]] MoiPersistentStateDemand plan_moi_persistent_state_demand(
    const ConSanRequest &request, const BoundRuntimeResources &resources,
    const ConSanMoiOperatingPoint &point, const MoiPersistentStateFacts &facts);

[[nodiscard]] MoiDynamicStackSpillPolicy plan_moi_dynamic_stack_spill(ConSanMoiEngine engine,
                                                                      rj_code_arch_t arch);

[[nodiscard]] MoiOperandOverlapSpillPolicy
plan_moi_operand_overlap_spill(const MoiOperandOverlapSpillContext &context);

[[nodiscard]] MoiDispatchIdentityPlan
plan_moi_dispatch_identity(const ConSanRequest &request, const MoiDispatchIdentityFacts &facts);

[[nodiscard]] MoiScalarAbiPlan
make_moi_scalar_abi_plan(const MoiScalarRoutingState &routing_state,
                         std::optional<consan_detail::MoiSpecialStateSgprs> special_state,
                         uint16_t fixed_indirect_pc_offset, bool access_router_uses_dense_abi);

[[nodiscard]] MoiScalarAbiPlan plan_moi_scalar_abi(const ConSanRequest &request,
                                                   const ConSanMoiOperatingPoint &point);

/// Compose the common Record/Replay + Sampled dense-router representation
/// from a mode-owned scalar ABI and normalized target facts.
[[nodiscard]] std::optional<MoiDenseRouterPlan>
make_recording_moi_dense_router_plan(const MoiScalarAbiPlan &scalar_abi,
                                     const MoiScalarRoutingState &routing_state,
                                     const MoiScalarTargetFacts &target);

/// Resolve the selected mode's dense-router mechanics at the narrow mode
/// registry boundary. Consumers never branch on the engine themselves.
[[nodiscard]] std::optional<MoiDenseRouterPlan>
plan_moi_dense_router(const ConSanRequest &request, const ConSanMoiOperatingPoint &point,
                      rj_code_arch_t arch);

[[nodiscard]] MoiDenseAccessRouteTraits moi_dense_access_route_traits(ConSanMoiEngine engine);

[[nodiscard]] ConSanEvidenceRequirements
plan_moi_evidence_requirements(ConSanMoiEngine engine, const MoiEvidencePlanningContext &context);

[[nodiscard]] ConSanEvidenceRequirements
plan_record_replay_evidence_requirements(const MoiEvidencePlanningContext &context);
[[nodiscard]] ConSanEvidenceRequirements
plan_sampled_evidence_requirements(const MoiEvidencePlanningContext &context);
[[nodiscard]] ConSanEvidenceRequirements
plan_inline_shadow_evidence_requirements(const MoiEvidencePlanningContext &context);

struct MoiModeOperations {
  MoiObjectModePlan (*plan)(const ConSanRequest &, const BoundRuntimeResources &,
                            const TransformPolicy &, const ConSanMoiOperatingPoint &,
                            const MoiObjectFacts &, const ConSanObservationPlan &);
  void (*apply)(std::span<const uint8_t>, const ConSanOptions &, ConSanMoiOperatingPoint &,
                rj_code_arch_t, MoiResourcePlanningState &, std::span<const ConSanMoiCandidate>,
                const MoiObjectFacts &, const MoiObjectModeSemantics &, ConSanTransformArtifacts &);
  uint16_t (*access_scratch_vgpr_count)(const ConSanRequest &, const BoundRuntimeResources &,
                                        const MoiAccessResourceFacts &, const ConSanMoiCandidate &);
  MoiOperationalEvidenceKinds operational_evidence;
  uint16_t (*barrier_scratch_vgpr_count)(const MoiBarrierScratchFacts &);
  std::optional<uint16_t> dynamic_stack_frame_save_sgpr_offset;
  uint16_t (*exec_save_sgpr_count)(const MoiExecSaveRequirement &, const MoiScalarTargetFacts &);
  MoiPrologueModePolicy prologue;
  MoiPersistentStateDemand (*persistent_state_demand)(const ConSanRequest &,
                                                      const BoundRuntimeResources &,
                                                      const ConSanMoiOperatingPoint &,
                                                      const MoiPersistentStateFacts &);
  MoiTransientScalarPlacementTraits transient_scalar_placement;
  bool dynamic_stack_spill_without_target_backend;
  bool dynamic_stack_spill_requires_every_owner_dynamic;
  MoiOperandOverlapSpillPolicy (*operand_overlap_spill)(const MoiOperandOverlapSpillContext &);
  std::optional<uint16_t> (*access_spill_fallback)(const MoiAccessSpillFallbackContext &);
  MoiDispatchIdentityPlan (*dispatch_identity)(const ConSanRequest &,
                                               const MoiDispatchIdentityFacts &);
  MoiScalarAbiPlan (*scalar_abi)(const MoiScalarRoutingState &);
  MoiDenseAccessRouteTraits dense_access_route;
  std::optional<MoiDenseRouterPlan> (*dense_router)(const MoiScalarAbiPlan &,
                                                    const MoiScalarRoutingState &,
                                                    const MoiScalarTargetFacts &);
  ConSanEvidenceRequirements (*plan_evidence)(const MoiEvidencePlanningContext &);
  bool (*plan_report_layout)(const ConSanMoiAutoReportInventory &, ConSanMoiAutoReportPlan &,
                             uint64_t &cursor);
  std::optional<ConSanMoiAutoReportInventory> (*reconstruct_report_inventory)(
      const ConSanMoiReportBufferLayout &);
};

template <typename ModeKey> struct MoiModeRegistrationFor {
  ModeKey mode;
  const MoiModeOperations *operations = nullptr;
};

using MoiModeRegistration = MoiModeRegistrationFor<ConSanMoiEngine>;

template <typename ModeKey>
[[nodiscard]] const MoiModeOperations *
find_moi_mode_operations(std::span<const MoiModeRegistrationFor<ModeKey>> registrations,
                         ModeKey mode) {
  const auto registration =
      std::ranges::find(registrations, mode, &MoiModeRegistrationFor<ModeKey>::mode);
  return registration == registrations.end() ? nullptr : registration->operations;
}

extern const MoiModeOperations kRecordReplayModeOperations;
extern const MoiModeOperations kSampledModeOperations;
extern const MoiModeOperations kInlineShadowModeOperations;

[[nodiscard]] const MoiModeOperations &moi_mode_operations(ConSanMoiEngine engine);

} // namespace rocjitsu::consan_moi_impl
