// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "hsa/hsa_api_trace_minimal.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <atomic>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace rocjitsu::consan_hook {

enum HookLogLevel : int {
  kLogDisabled = 0,
  kLogInfo = 1,
  kLogVerbose = 2,
  kLogDebug = 3,
};

enum class CheckTrapMode : uint8_t {
  All,
  Lds,
  Flat,
};

enum class ScReportMode : uint8_t {
  Auto,
  Trap,
};

enum class HookPolicy : uint8_t {
  Default,
  Strict,
};

// This is an implementation safety ceiling, not a user-facing selection
// policy. It avoids the unbounded vector reservations that UINT32_MAX would
// trigger in planners while exceeding the supported-site count of current
// production code objects by orders of magnitude.
constexpr uint32_t kConSanAllSupportedPatchBudget = 65536;

struct HookConfig {
  bool enabled = false;
  HookPolicy policy = HookPolicy::Default;
  std::optional<rocjitsu::ConSanFlavor> flavor;
  rocjitsu::ConSanMoiEngine moi_engine = rocjitsu::ConSanMoiEngine::RecordReplay;
  rocjitsu::ConSanMoiOwnerSource moi_owner_source = rocjitsu::ConSanMoiOwnerSource::WorkitemId;
  rocjitsu::ConSanFlatProvenanceMode flat_provenance_mode =
      rocjitsu::ConSanFlatProvenanceMode::Likely;
  bool fail_closed = false;
  bool require_patch = false;
  bool probe_nop = false;
  bool probe_trampoline_nop = false;
  bool probe_endpgm = false;
  bool probe_lds_endpgm = false;
  CheckTrapMode check_trap_mode = CheckTrapMode::All;
  bool probe_lds_check_trap = false;
  bool probe_flat_check_trap = false;
  bool probe_flat_trap = false;
  bool abort_unmatched_barrier_wait = false;
  bool fault_drop_barrier = false;
  bool fault_allow_destructive_incomplete_barrier_drop = false;
  bool fault_move_barrier = false;
  bool fault_allow_completing_conditional_barrier_move = false;
  bool fault_allow_destructive_divergent_barrier_move = false;
  bool fault_mutate_barrier_id_scope = false;
  bool fault_mutate_barrier_participants = false;
  rocjitsu::ConSanBarrierMoveDirection fault_barrier_move_direction =
      rocjitsu::ConSanBarrierMoveDirection::LegacyMarker;
  std::string fault_barrier_destination_identity;
  std::string fault_barrier_sequence_identity;
  std::string fault_barrier_companion_site_identity;
  std::string fault_barrier_companion_sequence_identity;
  std::optional<int32_t> fault_barrier_target_id;
  std::optional<uint32_t> fault_barrier_target_participant_count;
  std::optional<uint64_t> fault_barrier_target_participant_mask;
  bool fault_atomic_wrong_address = false;
  bool fault_atomic_weaken_order = false;
  rocjitsu::ConSanAtomicOrderEdge fault_atomic_order_edge = rocjitsu::ConSanAtomicOrderEdge::Any;
  bool fault_atomic_weaken_scope = false;
  bool fault_ordinary_weaken_order = false;
  bool fault_ordinary_weaken_scope = false;
  uint32_t fault_atomic_address_delta = 0;
  bool fault_ordinary_wrong_address = false;
  uint32_t fault_ordinary_address_delta = 0;
  bool fault_dry_run = false;
  bool fault_require_exactly_one = false;
  std::optional<uint32_t> fault_load_occurrence;
  rocjitsu::ConSanPerturbationKind sc_perturb_kind = rocjitsu::ConSanPerturbationKind::None;
  rocjitsu::ConSanPerturbationEdge sc_perturb_edge = rocjitsu::ConSanPerturbationEdge::Release;
  std::string sc_perturb_identity;
  uint32_t sc_perturb_index = 0;
  uint32_t sc_perturb_max = 1;
  uint32_t sc_perturb_sleep = 1;
  uint32_t sc_perturb_required_count = 0;
  bool moi_init_owner_epoch = false;
  bool moi_track_barriers = false;
  bool moi_track_atomics = false;
  bool moi_dynamic_access_records = false;
  bool moi_sampled_check = false;
  bool moi_partition_mask_debug = false;
  bool test_force_vgpr_spill = false;
  bool test_force_private_epoch = false;
  bool test_seed_inline_exact_odd = false;
  std::string test_kernel_name_filter;
  bool moi_require_records = false;
  bool moi_require_diagnostics = false;
  bool moi_forbid_diagnostics = false;
  bool moi_require_replay_conflict = false;
  bool moi_forbid_overflow = false;
  uint32_t fault_barrier_index = 0;
  uint32_t fault_atomic_index = 0;
  uint32_t fault_ordinary_index = 0;
  std::string fault_site_identity;
  rocjitsu::ConSanDelayMode delay_mode = rocjitsu::ConSanDelayMode::Nop;
  uint16_t delay_var_ssrc = 106;
  std::optional<uint16_t> scratch_vgpr;
  std::optional<uint16_t> moi_exec_save_sgpr;
  std::optional<uint16_t> moi_owner_sgpr;
  std::optional<uint16_t> moi_owner_vgpr;
  std::optional<uint16_t> moi_epoch_vgpr;
  std::optional<uint64_t> report_buffer_address;
  ScReportMode sc_report_mode = ScReportMode::Auto;
  std::optional<uint64_t> moi_report_buffer_address;
  uint64_t moi_report_buffer_size = 0;
  uint64_t moi_auto_report_buffer_size = 0;
  bool moi_auto_report_buffer_size_explicit = false;
  uint32_t delay_nops = 0;
  uint32_t max_patches = kConSanAllSupportedPatchBudget;
  bool max_patches_explicit = false;
  uint32_t moi_sample_stride = 1;
  uint32_t moi_sample_offset = 0;
  uint32_t moi_runtime_sample_stride = 1;
  bool moi_runtime_sample_stride_explicit = false;
  uint32_t moi_runtime_sample_offset = 0;
  uint32_t report_marker = 1;
  int log_level = kLogDisabled;
  std::string dump_dir;
};

constexpr std::string_view kMoiStandardProfile = "standard-v1";
constexpr uint32_t kMoiSampledStandardRuntimeStride = 16384u;

[[nodiscard]] inline const char *preflight_action_name(rocjitsu::ConSanPreflightAction action) {
  switch (action) {
  case rocjitsu::ConSanPreflightAction::NotRun:
    return "not-run";
  case rocjitsu::ConSanPreflightAction::Candidate:
    return "candidate";
  case rocjitsu::ConSanPreflightAction::Skip:
    return "skip";
  case rocjitsu::ConSanPreflightAction::Reject:
    return "reject";
  }
  return "unknown";
}

[[nodiscard]] inline const char *patch_kind_name(rocjitsu::ConSanPatchKind kind) {
  switch (kind) {
  case rocjitsu::ConSanPatchKind::InlineNopRewrite:
    return "inline-nop-rewrite";
  case rocjitsu::ConSanPatchKind::InlineEndpgmRewrite:
    return "inline-endpgm-rewrite";
  case rocjitsu::ConSanPatchKind::InlineLdsEndpgmRewrite:
    return "inline-lds-endpgm-rewrite";
  case rocjitsu::ConSanPatchKind::InlineLdsLoadCheckTrap:
    return "inline-lds-load-check-trap";
  case rocjitsu::ConSanPatchKind::InlineLdsStoreCheckTrap:
    return "inline-lds-store-check-trap";
  case rocjitsu::ConSanPatchKind::LocalCaveLdsLoadCheckTrap:
    return "local-cave-lds-load-check-trap";
  case rocjitsu::ConSanPatchKind::LocalCaveLdsStoreCheckTrap:
    return "local-cave-lds-store-check-trap";
  case rocjitsu::ConSanPatchKind::InlineFlatLoadCheckTrap:
    return "inline-flat-load-check-trap";
  case rocjitsu::ConSanPatchKind::InlineFlatStoreCheckTrap:
    return "inline-flat-store-check-trap";
  case rocjitsu::ConSanPatchKind::LocalCaveFlatLoadCheckTrap:
    return "local-cave-flat-load-check-trap";
  case rocjitsu::ConSanPatchKind::LocalCaveFlatStoreCheckTrap:
    return "local-cave-flat-store-check-trap";
  case rocjitsu::ConSanPatchKind::InlineFlatTrapRewrite:
    return "inline-flat-trap-rewrite";
  case rocjitsu::ConSanPatchKind::InlineBarrierNopRewrite:
    return "inline-barrier-nop-rewrite";
  case rocjitsu::ConSanPatchKind::InlineBarrierIdScopeRewrite:
    return "inline-barrier-id-scope-rewrite";
  case rocjitsu::ConSanPatchKind::InlineBarrierParticipantCountRewrite:
    return "inline-barrier-participant-count-rewrite";
  case rocjitsu::ConSanPatchKind::InlineBarrierMoveSourceRewrite:
    return "inline-barrier-move-source-rewrite";
  case rocjitsu::ConSanPatchKind::InlineBarrierMoveTargetRewrite:
    return "inline-barrier-move-target-rewrite";
  case rocjitsu::ConSanPatchKind::InlineAtomicAddressRewrite:
    return "inline-atomic-address-rewrite";
  case rocjitsu::ConSanPatchKind::InlineAtomicOrderRewrite:
    return "inline-atomic-order-rewrite";
  case rocjitsu::ConSanPatchKind::InlineAtomicScopeRewrite:
    return "inline-atomic-scope-rewrite";
  case rocjitsu::ConSanPatchKind::InlineOrdinaryOrderRewrite:
    return "inline-ordinary-order-rewrite";
  case rocjitsu::ConSanPatchKind::InlineOrdinaryAddressRewrite:
    return "inline-ordinary-address-rewrite";
  case rocjitsu::ConSanPatchKind::InlineOrdinaryScopeRewrite:
    return "inline-ordinary-scope-rewrite";
  case rocjitsu::ConSanPatchKind::InlineMoiAccessRecordStore:
    return "inline-moi-access-record-store";
  case rocjitsu::ConSanPatchKind::TrampolineMoiAccessRecordStore:
    return "trampoline-moi-access-record-store";
  case rocjitsu::ConSanPatchKind::InlineMoiExactShadowStore:
    return "inline-moi-exact-shadow-store";
  case rocjitsu::ConSanPatchKind::TrampolineMoiExactShadowStore:
    return "trampoline-moi-exact-shadow-store";
  case rocjitsu::ConSanPatchKind::InlineMoiSampledWatchpointStore:
    return "inline-moi-sampled-watchpoint-store";
  case rocjitsu::ConSanPatchKind::TrampolineMoiSampledWatchpointStore:
    return "trampoline-moi-sampled-watchpoint-store";
  case rocjitsu::ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue:
    return "kernel-entry-moi-owner-epoch-prologue";
  case rocjitsu::ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue:
    return "kernel-entry-moi-private-epoch-prologue";
  case rocjitsu::ConSanPatchKind::TrampolineMoiBarrierRecord:
    return "trampoline-moi-barrier-record";
  case rocjitsu::ConSanPatchKind::TrampolineMoiInlineEpochBarrier:
    return "trampoline-moi-inline-epoch-barrier";
  case rocjitsu::ConSanPatchKind::TrampolineMoiInlineAtomicOrdering:
    return "trampoline-moi-inline-atomic-ordering";
  case rocjitsu::ConSanPatchKind::TrampolineMoiAtomicRecord:
    return "trampoline-moi-atomic-record";
  case rocjitsu::ConSanPatchKind::TrampolineMoiSampledSyncMetadata:
    return "trampoline-moi-sampled-sync-metadata";
  case rocjitsu::ConSanPatchKind::TrampolineMoiFenceRecord:
    return "trampoline-moi-fence-record";
  case rocjitsu::ConSanPatchKind::InlineMalformedBarrierAbort:
    return "inline-malformed-barrier-abort";
  case rocjitsu::ConSanPatchKind::TrampolineScPerturbation:
    return "trampoline-sc-perturbation";
  case rocjitsu::ConSanPatchKind::TrampolineScIndirectBranchIsland:
    return "trampoline-sc-indirect-branch-island";
  case rocjitsu::ConSanPatchKind::TrampolineScDenseCallDispatcher:
    return "trampoline-sc-dense-call-dispatcher";
  case rocjitsu::ConSanPatchKind::TrampolineScDenseEntryHost:
    return "trampoline-sc-dense-entry-host";
  case rocjitsu::ConSanPatchKind::TrampolineScBranchRelayDonor:
    return "trampoline-sc-branch-relay-donor";
  case rocjitsu::ConSanPatchKind::TrampolineScRelayReservoir:
    return "trampoline-sc-relay-reservoir";
  case rocjitsu::ConSanPatchKind::TrampolineMoiIndirectBranchIsland:
    return "trampoline-moi-indirect-branch-island";
  case rocjitsu::ConSanPatchKind::TrampolineNop:
    return "trampoline-nop";
  }
  return "unknown";
}

[[nodiscard]] inline const char *
moi_candidate_source_name(rocjitsu::ConSanMoiCandidateSource source) {
  return rocjitsu::consan_moi_candidate_source_name(source);
}

[[nodiscard]] inline const char *
moi_resource_site_kind_name(rocjitsu::ConSanResourceSiteKind site_kind) {
  return rocjitsu::consan_resource_site_kind_name(site_kind);
}

[[nodiscard]] inline const char *fault_site_kind_name(rocjitsu::ConSanFaultSiteKind kind) {
  switch (kind) {
  case rocjitsu::ConSanFaultSiteKind::Barrier:
    return "barrier";
  case rocjitsu::ConSanFaultSiteKind::Atomic:
    return "atomic";
  case rocjitsu::ConSanFaultSiteKind::OrdinaryMemory:
    return "ordinary-memory";
  }
  return "unknown";
}

[[nodiscard]] inline const char *
ordinary_memory_support_reason_name(rocjitsu::ConSanOrdinaryMemorySupportReason reason) {
  switch (reason) {
  case rocjitsu::ConSanOrdinaryMemorySupportReason::NotApplicable:
    return "not-applicable";
  case rocjitsu::ConSanOrdinaryMemorySupportReason::Supported:
    return "supported";
  case rocjitsu::ConSanOrdinaryMemorySupportReason::SupportedSynchronizationOnly:
    return "supported-synchronization-only";
  case rocjitsu::ConSanOrdinaryMemorySupportReason::UnsupportedArchitecture:
    return "unsupported-architecture";
  case rocjitsu::ConSanOrdinaryMemorySupportReason::UnsupportedEncodingSize:
    return "unsupported-encoding-size";
  case rocjitsu::ConSanOrdinaryMemorySupportReason::MalformedEncoding:
    return "malformed-encoding";
  case rocjitsu::ConSanOrdinaryMemorySupportReason::MissingAddressVgpr:
    return "missing-address-vgpr";
  case rocjitsu::ConSanOrdinaryMemorySupportReason::MissingDestinationVgpr:
    return "missing-destination-vgpr";
  case rocjitsu::ConSanOrdinaryMemorySupportReason::MissingValueVgpr:
    return "missing-value-vgpr";
  }
  return "unknown";
}

[[nodiscard]] inline const char *fault_mutation_kind_name(rocjitsu::ConSanFaultMutationKind kind) {
  switch (kind) {
  case rocjitsu::ConSanFaultMutationKind::DropBarrier:
    return "drop-barrier";
  case rocjitsu::ConSanFaultMutationKind::MoveBarrierPair:
    return "move-barrier-pair";
  case rocjitsu::ConSanFaultMutationKind::BarrierIdScope:
    return "barrier-id-scope";
  case rocjitsu::ConSanFaultMutationKind::BarrierParticipantCount:
    return "barrier-participant-count";
  case rocjitsu::ConSanFaultMutationKind::AtomicWrongAddress:
    return "atomic-wrong-address";
  case rocjitsu::ConSanFaultMutationKind::AtomicWeakenOrder:
    return "atomic-weaken-order";
  case rocjitsu::ConSanFaultMutationKind::AtomicWeakenScope:
    return "atomic-weaken-scope";
  case rocjitsu::ConSanFaultMutationKind::OrdinaryWeakenOrder:
    return "ordinary-weaken-order";
  case rocjitsu::ConSanFaultMutationKind::OrdinaryWrongAddress:
    return "ordinary-wrong-address";
  case rocjitsu::ConSanFaultMutationKind::OrdinaryWeakenScope:
    return "ordinary-weaken-scope";
  }
  return "unknown";
}

[[nodiscard]] inline const char *perturbation_kind_name(rocjitsu::ConSanPerturbationKind kind) {
  switch (kind) {
  case rocjitsu::ConSanPerturbationKind::None:
    return "none";
  case rocjitsu::ConSanPerturbationKind::Barrier:
    return "barrier";
  case rocjitsu::ConSanPerturbationKind::Atomic:
    return "atomic";
  }
  return "unknown";
}

[[nodiscard]] inline const char *perturbation_edge_name(rocjitsu::ConSanPerturbationEdge edge) {
  switch (edge) {
  case rocjitsu::ConSanPerturbationEdge::Release:
    return "release";
  case rocjitsu::ConSanPerturbationEdge::Acquire:
    return "acquire";
  }
  return "unknown";
}

[[nodiscard]] inline const char *
barrier_move_direction_name(rocjitsu::ConSanBarrierMoveDirection direction) {
  switch (direction) {
  case rocjitsu::ConSanBarrierMoveDirection::LegacyMarker:
    return "legacy-marker";
  case rocjitsu::ConSanBarrierMoveDirection::Earlier:
    return "earlier";
  case rocjitsu::ConSanBarrierMoveDirection::Later:
    return "later";
  }
  return "unknown";
}

[[nodiscard]] inline const char *
barrier_move_cfg_contract_name(rocjitsu::ConSanBarrierMoveCfgContract contract) {
  switch (contract) {
  case rocjitsu::ConSanBarrierMoveCfgContract::SameBlock:
    return "same-block";
  case rocjitsu::ConSanBarrierMoveCfgContract::CompletingStructuredDiamond:
    return "completing-structured-diamond";
  case rocjitsu::ConSanBarrierMoveCfgContract::DestructiveStructuredExecDiamond:
    return "destructive-structured-exec-diamond";
  }
  return "unknown";
}

[[nodiscard]] inline const char *sync_event_kind_name(rocjitsu::ConSanSyncEventKind kind) {
  switch (kind) {
  case rocjitsu::ConSanSyncEventKind::Barrier:
    return "barrier";
  case rocjitsu::ConSanSyncEventKind::Fence:
    return "fence";
  case rocjitsu::ConSanSyncEventKind::Atomic:
    return "atomic";
  case rocjitsu::ConSanSyncEventKind::OrdinaryMemory:
    return "ordinary-memory";
  }
  return "unknown";
}

[[nodiscard]] inline const char *sync_sequence_kind_name(rocjitsu::ConSanSyncSequenceKind kind) {
  switch (kind) {
  case rocjitsu::ConSanSyncSequenceKind::Barrier:
    return "barrier";
  case rocjitsu::ConSanSyncSequenceKind::Fence:
    return "fence";
  case rocjitsu::ConSanSyncSequenceKind::Atomic:
    return "atomic";
  case rocjitsu::ConSanSyncSequenceKind::OrdinaryMemory:
    return "ordinary-memory";
  }
  return "unknown";
}

[[nodiscard]] inline const char *
barrier_operand_source_name(rocjitsu::ConSanBarrierSite::OperandSource source) {
  return rocjitsu::consan_barrier_operand_source_name(source);
}

[[nodiscard]] inline const char *barrier_scope_name(rocjitsu::ConSanBarrierSite::Scope scope) {
  return rocjitsu::consan_barrier_scope_name(scope);
}

[[nodiscard]] inline const char *sync_operation_name(rocjitsu::ConSanSyncOperation operation) {
  switch (operation) {
  case rocjitsu::ConSanSyncOperation::Unknown:
    return "unknown";
  case rocjitsu::ConSanSyncOperation::BarrierSignal:
    return "barrier-signal";
  case rocjitsu::ConSanSyncOperation::BarrierWait:
    return "barrier-wait";
  case rocjitsu::ConSanSyncOperation::BarrierFull:
    return "barrier-full";
  case rocjitsu::ConSanSyncOperation::BarrierInit:
    return "barrier-init";
  case rocjitsu::ConSanSyncOperation::BarrierJoin:
    return "barrier-join";
  case rocjitsu::ConSanSyncOperation::BarrierLeave:
    return "barrier-leave";
  case rocjitsu::ConSanSyncOperation::BarrierWakeup:
    return "barrier-wakeup";
  case rocjitsu::ConSanSyncOperation::BarrierStateQuery:
    return "barrier-state-query";
  case rocjitsu::ConSanSyncOperation::Fence:
    return "fence";
  case rocjitsu::ConSanSyncOperation::AtomicRmw:
    return "atomic-rmw";
  case rocjitsu::ConSanSyncOperation::AtomicCompareExchange:
    return "atomic-compare-exchange";
  case rocjitsu::ConSanSyncOperation::OrdinaryLoad:
    return "ordinary-load";
  case rocjitsu::ConSanSyncOperation::OrdinaryStore:
    return "ordinary-store";
  }
  return "unknown";
}

[[nodiscard]] inline const char *
sync_address_source_name(rocjitsu::ConSanSyncAddressSource source) {
  switch (source) {
  case rocjitsu::ConSanSyncAddressSource::NotApplicable:
    return "not-applicable";
  case rocjitsu::ConSanSyncAddressSource::Unknown:
    return "unknown";
  case rocjitsu::ConSanSyncAddressSource::LdsVector:
    return "lds-vector";
  case rocjitsu::ConSanSyncAddressSource::FlatVector:
    return "flat-vector";
  case rocjitsu::ConSanSyncAddressSource::GlobalScalarVector:
    return "global-scalar-vector";
  case rocjitsu::ConSanSyncAddressSource::BufferResource:
    return "buffer-resource";
  case rocjitsu::ConSanSyncAddressSource::ScratchVector:
    return "scratch-vector";
  }
  return "unknown";
}

[[nodiscard]] inline const char *sync_memory_role_name(rocjitsu::ConSanSyncMemoryRole role) {
  switch (role) {
  case rocjitsu::ConSanSyncMemoryRole::Unknown:
    return "unknown";
  case rocjitsu::ConSanSyncMemoryRole::None:
    return "none";
  case rocjitsu::ConSanSyncMemoryRole::Acquire:
    return "acquire";
  case rocjitsu::ConSanSyncMemoryRole::Release:
    return "release";
  case rocjitsu::ConSanSyncMemoryRole::AcquireRelease:
    return "acquire-release";
  case rocjitsu::ConSanSyncMemoryRole::SequentiallyConsistent:
    return "sequentially-consistent";
  }
  return "unknown";
}

[[nodiscard]] inline const char *sync_rmw_outcome_name(rocjitsu::ConSanSyncRmwOutcome outcome) {
  switch (outcome) {
  case rocjitsu::ConSanSyncRmwOutcome::NotApplicable:
    return "not-applicable";
  case rocjitsu::ConSanSyncRmwOutcome::Unknown:
    return "unknown";
  case rocjitsu::ConSanSyncRmwOutcome::NoReturn:
    return "no-return";
  case rocjitsu::ConSanSyncRmwOutcome::ReturnsOldValue:
    return "returns-old-value";
  case rocjitsu::ConSanSyncRmwOutcome::CompareExchange:
    return "compare-exchange";
  }
  return "unknown";
}

[[nodiscard]] inline const char *
sync_confidence_name(rocjitsu::ConSanSemanticConfidence confidence) {
  switch (confidence) {
  case rocjitsu::ConSanSemanticConfidence::Exact:
    return "exact";
  case rocjitsu::ConSanSemanticConfidence::Conservative:
    return "conservative";
  case rocjitsu::ConSanSemanticConfidence::Ambiguous:
    return "ambiguous";
  case rocjitsu::ConSanSemanticConfidence::Unsupported:
    return "unsupported";
  }
  return "unknown";
}

[[nodiscard]] inline const char *owner_proof_name(rocjitsu::ConSanOwnerProofKind proof) {
  switch (proof) {
  case rocjitsu::ConSanOwnerProofKind::KernelLocal:
    return "kernel-local";
  case rocjitsu::ConSanOwnerProofKind::DirectCall:
    return "direct-call";
  case rocjitsu::ConSanOwnerProofKind::RecoveredIndirectCall:
    return "recovered-indirect-call";
  }
  return "unknown";
}

struct OwnerLogFields {
  std::string names = "-";
  std::string proofs = "-";
};

[[nodiscard]] inline OwnerLogFields
owner_log_fields(std::span<const rocjitsu::ConSanExecutionOwner> owners,
                 std::span<const rocjitsu::ConSanKernelInfo> kernels) {
  OwnerLogFields fields;
  if (owners.empty())
    return fields;
  fields.names.clear();
  fields.proofs.clear();
  for (const rocjitsu::ConSanExecutionOwner &owner : owners) {
    const auto kernel = std::ranges::find(kernels, owner.descriptor_file_offset,
                                          &rocjitsu::ConSanKernelInfo::descriptor_file_offset);
    if (kernel == kernels.end())
      return {};
    if (!fields.names.empty()) {
      fields.names += ',';
      fields.proofs += ',';
    }
    fields.names += kernel->name;
    fields.proofs += owner_proof_name(owner.proof);
  }
  return fields;
}

[[nodiscard]] inline const char *
moi_resource_source_name(rocjitsu::ConSanRegisterAllocationSource source) {
  return rocjitsu::consan_register_allocation_source_name(source);
}

[[nodiscard]] inline const char *delay_mode_name(rocjitsu::ConSanDelayMode mode) {
  return rocjitsu::consan_delay_mode_name(mode);
}

[[nodiscard]] inline const char *owner_source_name(rocjitsu::ConSanMoiOwnerSource source) {
  switch (source) {
  case rocjitsu::ConSanMoiOwnerSource::WorkitemId:
    return "workitem_id";
  case rocjitsu::ConSanMoiOwnerSource::HwId:
    return "hw_id";
  }
  return "unknown";
}

[[nodiscard]] inline const char *
flat_provenance_mode_name(rocjitsu::ConSanFlatProvenanceMode mode) {
  switch (mode) {
  case rocjitsu::ConSanFlatProvenanceMode::Likely:
    return "likely";
  case rocjitsu::ConSanFlatProvenanceMode::Strict:
    return "strict";
  }
  return "unknown";
}

[[nodiscard]] inline const char *flavor_name(rocjitsu::ConSanFlavor flavor) {
  return rocjitsu::consan_flavor_name(flavor);
}

[[nodiscard]] inline const char *check_trap_mode_name(CheckTrapMode mode) {
  switch (mode) {
  case CheckTrapMode::All:
    return "all";
  case CheckTrapMode::Lds:
    return "lds";
  case CheckTrapMode::Flat:
    return "flat";
  }
  return "unknown";
}

[[nodiscard]] inline const char *sc_report_mode_name(ScReportMode mode) {
  return mode == ScReportMode::Auto ? "auto" : "trap";
}

[[nodiscard]] inline const char *hook_policy_name(HookPolicy policy) {
  return policy == HookPolicy::Default ? "default" : "strict";
}

[[nodiscard]] inline const char *lds_access_kind_name(rocjitsu::ConSanLdsAccessKind kind) {
  return rocjitsu::consan_lds_access_kind_name(kind);
}

[[nodiscard]] inline const char *
flat_address_space_hint_name(rocjitsu::ConSanFlatAddressSpaceHint hint) {
  return rocjitsu::consan_flat_address_space_hint_name(hint);
}

struct AutoMoiReportSummary {
  uint64_t buffer_count = 0;
  uint64_t required_report_bytes = 0;
  uint64_t allocated_report_bytes = 0;
  uint64_t current_live_report_bytes = 0;
  uint64_t current_live_report_bytes_after_cleanup = 0;
  uint64_t peak_live_report_bytes = 0;
  uint64_t allocation_failure_count = 0;
  uint64_t capacity_failure_count = 0;
  uint64_t cleanup_failure_count = 0;
  uint64_t visible_access_record_count = 0;
  uint64_t visible_barrier_record_count = 0;
  uint64_t visible_atomic_record_count = 0;
  uint64_t visible_fence_record_count = 0;
  uint64_t visible_diagnostic_record_count = 0;
  uint64_t visible_inline_publication_count = 0;
  uint64_t visible_exact_shadow_entry_count = 0;
  uint64_t exact_incomplete_snapshot_count = 0;
  uint64_t exact_changed_snapshot_count = 0;
  uint64_t exact_malformed_snapshot_count = 0;
  uint64_t visible_inline_atomic_release_count = 0;
  uint64_t release_incomplete_snapshot_count = 0;
  uint64_t release_changed_snapshot_count = 0;
  uint64_t release_overflow_snapshot_count = 0;
  uint64_t release_source_incomplete_snapshot_count = 0;
  uint64_t release_malformed_snapshot_count = 0;
  uint64_t visible_inline_acquired_token_count = 0;
  uint64_t token_incomplete_snapshot_count = 0;
  uint64_t token_changed_snapshot_count = 0;
  uint64_t token_malformed_snapshot_count = 0;
  uint64_t inline_undercoverage_count = 0;
  uint64_t inline_overflow_count = 0;
  uint64_t inline_unsupported_count = 0;
  uint64_t inline_malformed_count = 0;
  uint64_t visible_sampled_watchpoint_count = 0;
  uint64_t visible_sampled_sync_metadata_count = 0;
  uint64_t dropped_access_record_count = 0;
  uint64_t dropped_barrier_record_count = 0;
  uint64_t dropped_atomic_record_count = 0;
  uint64_t dropped_fence_record_count = 0;
  uint64_t dropped_diagnostic_record_count = 0;
  uint64_t record_replay_bank_saturation_count = 0;
  uint64_t replay_conflict_count = 0;
  uint64_t replay_diagnostic_count = 0;
  uint64_t replay_dropped_access_count = 0;
  uint64_t replay_dropped_barrier_count = 0;
  uint64_t replay_unsupported_access_count = 0;
  uint64_t replay_unsupported_atomic_count = 0;
  uint64_t replay_unsupported_fence_count = 0;
  uint64_t replay_metadata_full_count = 0;
  uint64_t replay_diagnostic_capacity_exhausted_count = 0;
  uint64_t sampled_conflict_count = 0;
  uint64_t sampled_immediate_conflict_count = 0;
  uint64_t sampled_claimed_window_count = 0;
  uint64_t sampled_dropped_window_count = 0;
  uint64_t sampled_saturated_window_count = 0;
  uint64_t sampled_stale_snapshot_count = 0;
  uint64_t sampled_incomplete_snapshot_count = 0;
  uint64_t sampled_changed_snapshot_count = 0;
  uint64_t sampled_malformed_snapshot_count = 0;
  uint64_t sampled_unsupported_sync_count = 0;
  uint64_t sampled_malformed_sync_count = 0;

  [[nodiscard]] uint64_t sampled_unusable_snapshot_count() const {
    return sampled_stale_snapshot_count + sampled_incomplete_snapshot_count +
           sampled_changed_snapshot_count + sampled_malformed_snapshot_count;
  }

  [[nodiscard]] uint64_t exact_unusable_snapshot_count() const {
    return exact_incomplete_snapshot_count + exact_changed_snapshot_count +
           exact_malformed_snapshot_count;
  }

  [[nodiscard]] uint64_t release_unusable_snapshot_count() const {
    return release_incomplete_snapshot_count + release_changed_snapshot_count +
           release_overflow_snapshot_count + release_source_incomplete_snapshot_count +
           release_malformed_snapshot_count;
  }

  [[nodiscard]] uint64_t token_unusable_snapshot_count() const {
    return token_incomplete_snapshot_count + token_changed_snapshot_count +
           token_malformed_snapshot_count;
  }
};

[[nodiscard]] constexpr uint64_t
record_replay_bank_saturation_count(const ConSanMoiReportHeader &header, ConSanMoiEngine engine) {
  return engine == ConSanMoiEngine::RecordReplay &&
                 (header.flags & kConSanMoiReportFlagRecordReplayBankSaturated) != 0u
             ? 1u
             : 0u;
}

void reject_auto_moi_report_plan(uint64_t reader, uint64_t required_size, uint64_t configured_cap,
                                 std::string_view reason);
[[nodiscard]] bool
allocate_auto_moi_report_buffer(CoreApiTable *core, hsa_agent_t agent, uint64_t reader,
                                uint64_t required_size, uint64_t requested_size,
                                uint64_t configured_cap, const ConSanMoiReportBufferLayout &layout,
                                ConSanMoiEngine engine, bool track_barriers, bool track_atomics,
                                bool test_seed_inline_exact_odd, uint64_t *address,
                                uint64_t *registered_size, uint64_t *registered_generation);
void register_auto_moi_report_compact_tokens(uint64_t reader, const ConSanResult &result);
[[nodiscard]] AutoMoiReportSummary summarize_and_clear_auto_moi_report_buffers(CoreApiTable *core);

using ConSanTransformOverride = ConSanResult (*)(std::span<const uint8_t>, const ConSanOptions &);

extern std::atomic<int> g_log_level;
extern std::atomic<uint64_t> g_dump_sequence;
extern std::atomic<ConSanTransformOverride> g_test_consan_transform_override;

[[nodiscard]] std::optional<HookConfig> parse_config();
[[nodiscard]] bool refresh_report_config_from_env(HookConfig *config);

void log_message(int required_level, const char *format, ...);

ConSanResult run_consan_transform(std::span<const uint8_t> bytes, const ConSanOptions &options);

} // namespace rocjitsu::consan_hook
