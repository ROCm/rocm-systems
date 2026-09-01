// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_placement_contracts.h
/// @brief Explicit contracts exported by MOI resource placement.

#pragma once

#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"
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
class MoiLocalNopIslandAllocator;

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

inline constexpr uint32_t kMoiLocalIndirectIslandWords = 8u;
inline constexpr uint16_t kMoiDispatchStateSgprCount = 2u;
inline constexpr uint32_t kMoiRecordReplayIndirectIslandWords = 7u;
inline constexpr uint32_t kMoiInlineShadowIndirectIslandWords = 8u;
// A branch-only access layout retains a two-word relay pair after every
// reserved synchronization island. The island remains available to the later
// sync pass while the dedicated pair forms a capacity-accounted relay spine.
inline constexpr uint32_t kMoiRecordReplayBarrierRelayWords = 2u;
inline constexpr uint32_t kMoiRecordReplayBarrierRelaySlotWords =
    kMoiRecordReplayIndirectIslandWords + kMoiRecordReplayBarrierRelayWords;
inline constexpr uint32_t kMoiDenseBarrierEntryKeyPrologueWords = 6u;
inline constexpr uint32_t kMoiRecordReplayBorrowedEntryIslandWords = 20u;

[[nodiscard]] constexpr uint32_t moi_dense_entry_island_words(bool derive_key_at_entry,
                                                              bool key_encodes_scc = false) {
  // Relocatable dense routers derive the dispatcher key at entry. The SCC
  // snapshot can share the key scalar under pressure, adding one carrier word.
  return derive_key_at_entry ? 6u + (key_encodes_scc ? 1u : 0u) + (7u - 1u) : 7u;
}

using MoiDescriptorVgprRequirements = ConSanDescriptorRegisterRequirements;
using MoiDescriptorSgprRequirements = ConSanDescriptorRegisterRequirements;
using MoiDescriptorPrivateRequirements = ConSanDescriptorMemoryRequirements;
using MoiDescriptorLdsRequirements = ConSanDescriptorMemoryRequirements;
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

/// Target-neutral placement and resource result shared by every planned MOI
/// access patch. Engine evidence semantics are deliberately absent.
struct MoiPlannedAccessPatch {
  const ConSanMoiCandidate *candidate = nullptr;
  DbiPatchPlacement placement;
  std::optional<uint64_t> entry_island_offset;
  bool entry_island_at_anchor = false;
  bool dense_call_anchor = false;
  std::optional<uint64_t> dense_dispatcher_offset;
  std::optional<BranchOnlyRelayRoute> branch_only_route;
  std::vector<uint32_t> displaced_tail_words;
  ResolvedMoiScratchPlan resources;
  std::optional<VgprSpillSequence> spill;
  std::optional<SgprSpillSequence> scalar_spill;
  std::optional<uint32_t> private_epoch_offset;
  std::optional<consan_detail::MoiWorkitemOwnerDerivationPlan> owner_derivation;
  uint32_t persistent_private_state_end = 0;
  uint32_t required_private_bytes = 0;
  uint32_t guest_instruction_word_count = 0;
};

/// Common replay-body placement state used by Record/Replay and Sampled.
struct MoiPlannedReplayAccessPatch : MoiPlannedAccessPatch {
  uint64_t entry_island_word_count = 0;
  std::optional<uint16_t> incoming_vgpr_bank_mode;
  ConSanMoiPersistentWorkgroupPrivateOffsets private_workgroup_offsets;
  uint32_t probe_guest_instruction_offset = 0;
  bool spill_overlaps_guest_operands = false;
  bool wrap_embedded_guest_vgpr_bank = false;
  bool branch_only_scalar_spill = false;
};

enum class MoiDenseAccessRouteAbi {
  RecordReplay,
  Sampled,
  InlineShadow,
};

struct MoiInlineDenseRouterScalarAbi {
  ConSanMoiIndirectJumpSgprs indirect_jump;
  uint16_t dispatch_key_sgpr = 0;
  std::optional<uint16_t> call_return_sgpr;
};

[[nodiscard]] std::optional<MoiInlineDenseRouterScalarAbi>
moi_inline_dense_router_scalar_abi(const ConSanRequest &request,
                                   const ConSanMoiOperatingPoint &point, rj_code_arch_t arch);

struct MoiDenseAccessRoutePlan {
  consan_detail::MoiDenseCandidatePartition partition;
  std::unordered_map<const ConSanMoiCandidate *, uint64_t> entry_islands;
  std::unordered_map<const ConSanMoiCandidate *, uint64_t> dispatchers;
  std::map<uint64_t, MoiDenseEntryHost> entry_hosts;
};

struct MoiAccessEntryIslandPlan {
  std::optional<uint64_t> offset;
  uint64_t word_count = 0u;
  uint32_t original_size = 0u;
  std::vector<uint32_t> displaced_tail_words;
  bool at_anchor = false;
  bool uses_preferred_island = false;
};

/// A scalar register interval that a relocated host must preserve while it
/// bootstraps a generated router. This is a placement constraint, not an
/// engine emission policy.
struct MoiSgprRange {
  uint16_t base = 0;
  uint16_t width = 0;
};

enum class MoiRelocatableHostScalarState : uint8_t {
  AccessRouter,
  BarrierRouter,
};

/// A host discovered in existing text. Appended-body placement is absent:
/// callers still own the transaction that reserves and populates generated
/// bytes.
struct MoiDenseRelayHost {
  uint64_t host_offset = 0;
  std::vector<uint32_t> displaced_words;
};

/// Engine-neutral inputs to owner-complete dense-host discovery. Placement
/// owns the CFG/liveness proof; engine components supply only their occupied
/// ranges and the scalar ABI that the relocated host must preserve.
struct MoiDenseRelayHostRequest {
  uint64_t owner_begin = 0;
  uint64_t owner_end = 0;
  size_t host_word_count = 0;
  std::span<const uint64_t> anchors;
  std::span<const ConSanPreappliedReservedRange> preapplied_reserved_ranges;
  std::span<const uint64_t> owner_descriptor_file_offsets;
  std::span<const MoiSgprRange> bootstrap_ranges;
  std::span<const std::pair<uint64_t, uint64_t>> claimed_host_ranges;
};

/// Persistent VGPRs that cannot be borrowed by an entry relay before the
/// probe's ordinary spill transaction has run.
struct MoiPersistentVgprStateView {
  ConSanMoiOwnerEpochRegisterState moi_owner_epoch_vgprs;
  std::optional<uint16_t> workgroup_key;
  std::optional<uint16_t> dispatch_id;
  ConSanMoiPersistentWorkgroupRegisters record_replay_workgroup;
};

struct MoiCfgDistance {
  uint32_t block_edges = 0;
  uint64_t text_distance = 0;

  auto operator<=>(const MoiCfgDistance &) const = default;
};

struct MoiDecodedInstructionRange {
  uint64_t text_offset = 0;
  uint32_t size = 0;
  bool is_scalar_clause = false;
};

[[nodiscard]] constexpr uint32_t
moi_record_replay_entry_island_words(bool spill_backed_scalar_assignment) {
  return kMoiRecordReplayIndirectIslandWords + (spill_backed_scalar_assignment ? 1u : 0u);
}

[[nodiscard]] std::vector<const ConSanMoiCandidate *>
order_moi_replay_access_candidates(std::span<const ConSanMoiCandidate> admitted,
                                   const ConSanObservationPlan &observation_plan,
                                   bool track_atomics);

[[nodiscard]] uint64_t
moi_reserved_access_sync_island_count(const ProgramInventory &program_inventory,
                                      const ConSanObservationPlan &observation_plan,
                                      const ConSanRequest &request, const TransformPolicy &policy,
                                      const ConSanMoiOperatingPoint &point);

[[nodiscard]] std::span<const ConSanPreappliedReservedRange>
moi_resource_reserved_ranges(const MoiResourcePlanningState &state);

/// Reserve a placement-owned text range unless an existing reservation
/// already covers its anchor. Returns true when a new reservation is added.
[[nodiscard]] bool moi_resource_reserve_range_if_uncovered(MoiResourcePlanningState &state,
                                                           ConSanPreappliedReservedRange range);

[[nodiscard]] Decoder *moi_resource_decoder(MoiResourcePlanningState &state);

[[nodiscard]] bool moi_resource_entry_window_is_single_entry(const MoiResourcePlanningState &state,
                                                             uint64_t text_offset,
                                                             uint64_t byte_count);

[[nodiscard]] bool moi_resource_offsets_share_block(const MoiResourcePlanningState &state,
                                                    uint64_t first_offset, uint64_t second_offset);

/// True only when placement can associate the instruction with at least one
/// owner context and SCC is dead before it in every such owner.
[[nodiscard]] bool moi_resource_scc_is_dead_before(const MoiResourcePlanningState &state,
                                                   uint64_t text_offset);

[[nodiscard]] bool moi_resource_is_valid(const MoiResourcePlanningState &state);

[[nodiscard]] rj_code_arch_t moi_resource_arch(const MoiResourcePlanningState &state);

[[nodiscard]] bool moi_resource_has_owner_context(const MoiResourcePlanningState &state,
                                                  uint64_t descriptor_offset);

[[nodiscard]] bool moi_resource_owner_executes_offset(const MoiResourcePlanningState &state,
                                                      uint64_t descriptor_offset,
                                                      uint64_t text_offset);

[[nodiscard]] MoiCfgForwardDistanceIndexPtr
make_moi_cfg_forward_distance_index(const MoiResourcePlanningState &state,
                                    uint64_t descriptor_offset, uint64_t target_text_offset);

[[nodiscard]] std::optional<MoiCfgDistance>
moi_cfg_forward_distance(const MoiCfgForwardDistanceIndex &index, uint64_t from_text_offset);

[[nodiscard]] std::optional<MoiDecodedInstructionRange>
moi_resource_preceding_instruction(const MoiResourcePlanningState &state, uint64_t text_offset);

[[nodiscard]] bool moi_resource_owner_anchors_admit_sgpr_ranges(
    MoiResourcePlanningState &state, std::span<const uint64_t> owners,
    std::span<const uint64_t> anchors, std::span<const MoiSgprRange> ranges);

[[nodiscard]] bool moi_resource_owner_anchors_admit_call_clobber_ranges(
    MoiResourcePlanningState &state, std::span<const uint64_t> owners,
    std::span<const uint64_t> anchors, std::span<const MoiSgprRange> ranges);

[[nodiscard]] bool moi_resource_owner_ranges_conflict_with_physical_vcc(
    MoiResourcePlanningState &state, std::span<const uint64_t> owners,
    std::span<const MoiSgprRange> ranges, rj_code_arch_t arch);

[[nodiscard]] const ConSanCandidateResourcePlan *
resource_plan_for_candidate(std::span<const ConSanCandidateResourcePlan> plans,
                            const ConSanMoiCandidate &candidate);

[[nodiscard]] bool
moi_transient_sgpr_assignment_is_branch_only(const ConSanMoiOperatingPoint &allocation,
                                             std::span<const uint64_t> owner_descriptor_offsets);

[[nodiscard]] std::vector<std::pair<uint64_t, uint64_t>>
collect_moi_synchronization_ranges(const ProgramInventory &program_inventory);

[[nodiscard]] bool
moi_candidate_uses_branch_only_scalar_spill(std::span<const ConSanCandidateResourcePlan> plans,
                                            const ConSanMoiOperatingPoint &operating_point,
                                            const ConSanMoiCandidate &candidate);

[[nodiscard]] bool plan_moi_access_direct_reservoirs(
    const ProgramInventory &program_inventory, const TransformPolicy &policy, rj_code_arch_t arch,
    const MoiResourcePlanningState &resource_state, std::span<const uint8_t> original_text,
    std::span<const std::pair<uint64_t, uint64_t>> synchronization_ranges,
    std::span<const std::pair<uint64_t, uint64_t>> candidate_ranges,
    const std::map<uint64_t, MoiDenseEntryHost> &dense_entry_hosts, uint64_t route_frontier_source,
    size_t target_relay_count, DbiPatchPlacementPlanner &placement_planner,
    BranchOnlyRelayRouter &branch_router, BranchOnlyDirectRelayReservoirSet &reservoirs,
    std::string *error_out);

[[nodiscard]] MoiDenseAccessRoutePlan plan_moi_dense_access_routes(
    const ProgramInventory &program_inventory,
    std::span<const ConSanCandidateResourcePlan> site_plans, const ConSanRequest &request,
    const ConSanMoiOperatingPoint &point, rj_code_arch_t arch,
    MoiResourcePlanningState &resource_state, std::span<const uint8_t> original_text,
    std::span<const ConSanMoiCandidate *const> candidates, size_t max_candidates_per_group,
    uint64_t access_island_begin, uint64_t access_slot_words, bool use_indirect_appended,
    MoiDenseAccessRouteAbi abi, MoiLocalNopIslandAllocator *local_entry_islands,
    BranchOnlyRelayRouter *relay_ownership = nullptr);

[[nodiscard]] std::optional<MoiAccessEntryIslandPlan> plan_moi_access_entry_island(
    std::span<const uint8_t> original_text, const ConSanMoiCandidate &candidate,
    uint32_t required_island_words, uint64_t preferred_offset, uint64_t preferred_island_words,
    bool use_preferred_without_direct_branch, MoiLocalNopIslandAllocator *local_islands,
    MoiResourcePlanningState &resource_state, rj_code_arch_t arch,
    std::vector<std::string> &errors);

[[nodiscard]] std::optional<MoiAccessEntryIslandPlan> plan_moi_access_entry_island_with_policy(
    std::span<const uint8_t> original_text, const ConSanMoiCandidate &candidate,
    uint32_t required_island_words, uint64_t preferred_offset, uint64_t preferred_island_words,
    bool use_preferred_without_direct_branch, MoiLocalNopIslandAllocator *local_islands,
    MoiResourcePlanningState &resource_state, Decoder *decoder, rj_code_arch_t arch,
    std::vector<std::string> &errors,
    const std::function<bool(uint64_t, uint64_t)> &relocation_allowed);

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

void append_moi_resource_plans(MoiResourcePlanningState &state, const ConSanRequest &request,
                               const BoundRuntimeResources &resources,
                               const ConSanDebugOverrides &debug,
                               const ConSanMoiOperatingPoint &point,
                               std::span<const ConSanMoiCandidate> candidates,
                               std::vector<ConSanCandidateResourcePlan> &plans);

[[nodiscard]] bool moi_resource_owner_sgpr_window_admitted(const MoiResourcePlanningState &state,
                                                           std::span<const uint64_t> owners,
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
plan_moi_dispatch_id_fallback(const ConSanOptions &input, const ConSanMoiOperatingPoint &base,
                              const MoiResourceProblem &problem,
                              std::span<const ConSanCandidateResourcePlan> site_plans);

[[nodiscard]] std::optional<ConSanMoiScalarValidationFailure>
validate_moi_dispatch_id_sgprs(const ConSanRequest &request, const BoundRuntimeResources &resources,
                               const ConSanMoiOperatingPoint &point, rj_code_arch_t arch);

[[nodiscard]] std::optional<ConSanMoiScalarValidationFailure>
validate_moi_dispatch_id_vgprs(const ConSanMoiOperatingPoint &point);

[[nodiscard]] std::optional<ConSanMoiScalarValidationFailure>
validate_moi_ordinary_scalar_state(const ConSanRequest &request,
                                   const BoundRuntimeResources &resources,
                                   const ConSanMoiOperatingPoint &point, rj_code_arch_t arch);

[[nodiscard]] ConSanMoiPersistentPlacementUpdate configure_automatic_moi_persistent_vgprs(
    const ConSanMoiOperatingPoint &base, const MoiResourceProblem &problem,
    const ConSanDebugOverrides &debug, std::span<const ConSanCandidateResourcePlan> resource_plans,
    const MoiResourcePlanningState &planning_state);

[[nodiscard]] const ConSanCandidateResourcePlan *
resource_plan_for_site(std::span<const ConSanCandidateResourcePlan> plans,
                       ConSanResourceSiteKind site_kind, uint64_t text_offset);

[[nodiscard]] std::optional<ResolvedMoiScratchPlan>
resolve_moi_scratch_plan(const ConSanCandidateResourcePlan &plan,
                         const ConSanMoiOperatingPoint &site_point,
                         const ConSanMoiOperatingPoint &allocation, uint16_t expected_count);

[[nodiscard]] std::optional<ResolvedMoiScratchPlan>
resolve_moi_scratch(std::span<const ConSanCandidateResourcePlan> plans,
                    const ConSanMoiCandidate &candidate,
                    const ConSanMoiOperatingPoint &operating_point, uint16_t expected_count);

[[nodiscard]] std::optional<uint16_t> find_moi_borrowed_entry_backup_vgpr_for_point(
    MoiResourcePlanningState &state, std::span<const uint64_t> owners, uint64_t text_offset,
    uint16_t allocated_vgpr_count, const ConSanMoiOperatingPoint &point);

[[nodiscard]] std::optional<uint16_t> find_moi_borrowed_entry_backup_vgpr_for_state(
    MoiResourcePlanningState &state, std::span<const uint64_t> owners, uint64_t text_offset,
    uint16_t allocated_vgpr_count, const MoiPersistentVgprStateView &persistent_state);

[[nodiscard]] const ConSanMoiPersistentVgprAssignment *
moi_persistent_vgpr_assignment(const ConSanMoiOperatingPoint &allocation,
                               uint64_t descriptor_offset);

[[nodiscard]] const ConSanMoiTransientSgprAssignment *
moi_transient_sgpr_assignment(const ConSanMoiOperatingPoint &allocation,
                              uint64_t descriptor_offset);

[[nodiscard]] std::vector<MoiSgprRange>
moi_relocatable_host_scalar_ranges(const ConSanRequest &request,
                                   const ConSanMoiOperatingPoint &point,
                                   MoiRelocatableHostScalarState scalar_state, rj_code_arch_t arch);

[[nodiscard]] std::optional<MoiDenseRelayHost> find_moi_dense_relay_host_for_owners(
    MoiResourcePlanningState &resource_state, std::span<const uint8_t> text,
    const MoiDenseRelayHostRequest &request,
    const std::function<bool(uint64_t, uint64_t)> &overlaps_reserved);

[[nodiscard]] std::optional<uint16_t>
moi_sampled_access_return_scc_sgpr(const ConSanRequest &request,
                                   const ConSanMoiOperatingPoint &point);

void note_moi_access_private_requirements(MoiDescriptorPrivateRequirements &requirements,
                                          const MoiPlannedAccessPatch &patch);

void note_moi_access_patch_info(ConSanPatchPlacementEffects &effects,
                                const MoiPlannedAccessPatch &patch);

void note_moi_lds_requirements(MoiDescriptorLdsRequirements &requirements,
                               const ResolvedMoiScratchPlan &plan,
                               const ConSanMoiWorkgroupShadowLayout &layout);

void note_moi_replay_access_patch_info(ConSanPatchPlacementEffects &effects,
                                       const MoiPlannedReplayAccessPatch &patch);

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
                                    const ConSanMoiOperatingPoint &allocation,
                                    std::span<const uint64_t> owner_descriptor_offsets);

[[nodiscard]] bool
apply_moi_persistent_vgpr_assignment(ConSanMoiOperatingPoint &point,
                                     const ConSanMoiOperatingPoint &allocation,
                                     std::span<const uint64_t> owner_descriptor_offsets);

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

void note_dynamic_stack_private_requirement(ConSanPatchAbiEffects &effects,
                                            const VgprSpillSequence *spill);

} // namespace rocjitsu::consan_moi_impl
