// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "hsa/hsa_api_trace_minimal.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_pipeline.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_analyzer.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_snapshot.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_trust.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

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
// Bound an interposed loader wait so a stalled owner cannot deadlock the
// process. Campaigns whose individual code-object load can legitimately hold
// the reservation longer must raise this alongside their workload deadline.
constexpr uint32_t kConSanDefaultFaultReservationTimeoutMs = 30000;

/// Fully parsed hook configuration.
///
/// The six public base subobjects are the Slice-2 contracts constructed by the
/// environment parser. The remaining members are hook-local presentation,
/// allocation-mode, and parsing-provenance state; they are deliberately not
/// forwarded to legacy lowering. Inheritance preserves the existing parser's
/// concise field spelling while allowing production callers to pass each
/// immutable contract independently.
struct HookConfig : rocjitsu::ConSanRequest,
                    rocjitsu::TransformPolicy,
                    rocjitsu::RuntimePolicy,
                    rocjitsu::ConSanDebugOverrides,
                    rocjitsu::MutationRequest,
                    rocjitsu::BoundRuntimeResources {
  HookPolicy policy = HookPolicy::Default;
  CheckTrapMode check_trap_mode = CheckTrapMode::All;
  ScReportMode sc_report_mode = ScReportMode::Auto;
  bool moi_auto_report_buffer_size_explicit = false;
  bool max_patches_explicit = false;
  bool moi_runtime_sample_stride_explicit = false;
  int log_level = kLogDisabled;
  std::string dump_dir;

  HookConfig() {
    max_patches = kConSanAllSupportedPatchBudget;
    max_patches_is_expert_limit = false;
    fault_reservation_timeout_ms = kConSanDefaultFaultReservationTimeoutMs;
  }
};

constexpr std::string_view kMoiStandardProfile = "standard-v1";

inline constexpr auto kFaultSiteKinds = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanFaultSiteKind::Barrier, "barrier"),
    consan_enum(ConSanFaultSiteKind::Atomic, "atomic"),
    consan_enum(ConSanFaultSiteKind::LdsAccess, "lds-access"),
    consan_enum(ConSanFaultSiteKind::OrdinaryMemory, "ordinary-memory"));

[[nodiscard]] inline const char *fault_site_kind_name(rocjitsu::ConSanFaultSiteKind kind) {
  return kFaultSiteKinds.name(kind).data();
}

inline constexpr auto kOrdinaryMemorySupportReasons = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanOrdinaryMemorySupportReason::NotApplicable, "not-applicable"),
    consan_enum(ConSanOrdinaryMemorySupportReason::Supported, "supported"),
    consan_enum(ConSanOrdinaryMemorySupportReason::SupportedSynchronizationOnly,
                "supported-synchronization-only"),
    consan_enum(ConSanOrdinaryMemorySupportReason::UnsupportedArchitecture,
                "unsupported-architecture"),
    consan_enum(ConSanOrdinaryMemorySupportReason::UnsupportedEncodingSize,
                "unsupported-encoding-size"),
    consan_enum(ConSanOrdinaryMemorySupportReason::MalformedEncoding, "malformed-encoding"),
    consan_enum(ConSanOrdinaryMemorySupportReason::MissingAddressVgpr, "missing-address-vgpr"),
    consan_enum(ConSanOrdinaryMemorySupportReason::MissingDestinationVgpr,
                "missing-destination-vgpr"),
    consan_enum(ConSanOrdinaryMemorySupportReason::MissingValueVgpr, "missing-value-vgpr"));

[[nodiscard]] inline const char *
ordinary_memory_support_reason_name(rocjitsu::ConSanOrdinaryMemorySupportReason reason) {
  return kOrdinaryMemorySupportReasons.name(reason).data();
}

inline constexpr auto kFaultMutationKinds = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanFaultMutationKind::DropBarrier, "drop-barrier"),
    consan_enum(ConSanFaultMutationKind::MoveBarrierPair, "move-barrier-pair"),
    consan_enum(ConSanFaultMutationKind::BarrierIdScope, "barrier-id-scope"),
    consan_enum(ConSanFaultMutationKind::BarrierParticipantCount, "barrier-participant-count"),
    consan_enum(ConSanFaultMutationKind::AtomicWrongAddress, "atomic-wrong-address"),
    consan_enum(ConSanFaultMutationKind::AtomicWeakenOrder, "atomic-weaken-order"),
    consan_enum(ConSanFaultMutationKind::AtomicWeakenScope, "atomic-weaken-scope"),
    consan_enum(ConSanFaultMutationKind::LdsWrongAddress, "lds-wrong-address"),
    consan_enum(ConSanFaultMutationKind::OrdinaryWeakenOrder, "ordinary-weaken-order"),
    consan_enum(ConSanFaultMutationKind::OrdinaryWrongAddress, "ordinary-wrong-address"),
    consan_enum(ConSanFaultMutationKind::OrdinaryWeakenScope, "ordinary-weaken-scope"));

[[nodiscard]] inline const char *fault_mutation_kind_name(rocjitsu::ConSanFaultMutationKind kind) {
  return kFaultMutationKinds.name(kind).data();
}

inline constexpr auto kBarrierMoveDirections = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanBarrierMoveDirection::LegacyMarker, "legacy-marker"),
    consan_enum(ConSanBarrierMoveDirection::Earlier, "earlier"),
    consan_enum(ConSanBarrierMoveDirection::Later, "later"));

[[nodiscard]] inline const char *
barrier_move_direction_name(rocjitsu::ConSanBarrierMoveDirection direction) {
  return kBarrierMoveDirections.name(direction).data();
}

inline constexpr auto kBarrierMoveCfgContracts = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanBarrierMoveCfgContract::SameBlock, "same-block"),
    consan_enum(ConSanBarrierMoveCfgContract::CompletingStructuredDiamond,
                "completing-structured-diamond"),
    consan_enum(ConSanBarrierMoveCfgContract::DestructiveStructuredExecDiamond,
                "destructive-structured-exec-diamond"));

[[nodiscard]] inline const char *
barrier_move_cfg_contract_name(rocjitsu::ConSanBarrierMoveCfgContract contract) {
  return kBarrierMoveCfgContracts.name(contract).data();
}

inline constexpr auto kSyncSequenceKinds = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanSyncKind::Barrier, "barrier"),
    consan_enum(ConSanSyncKind::Fence, "fence"), consan_enum(ConSanSyncKind::Atomic, "atomic"),
    consan_enum(ConSanSyncKind::OrdinaryMemory, "ordinary-memory"));

[[nodiscard]] inline const char *sync_sequence_kind_name(rocjitsu::ConSanSyncKind kind) {
  return kSyncSequenceKinds.name(kind).data();
}

inline constexpr auto kSyncOperations = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanSyncOperation::Unknown, "unknown"),
    consan_enum(ConSanSyncOperation::BarrierSignal, "barrier-signal"),
    consan_enum(ConSanSyncOperation::BarrierWait, "barrier-wait"),
    consan_enum(ConSanSyncOperation::BarrierFull, "barrier-full"),
    consan_enum(ConSanSyncOperation::BarrierInit, "barrier-init"),
    consan_enum(ConSanSyncOperation::BarrierJoin, "barrier-join"),
    consan_enum(ConSanSyncOperation::BarrierLeave, "barrier-leave"),
    consan_enum(ConSanSyncOperation::BarrierWakeup, "barrier-wakeup"),
    consan_enum(ConSanSyncOperation::BarrierStateQuery, "barrier-state-query"),
    consan_enum(ConSanSyncOperation::Fence, "fence"),
    consan_enum(ConSanSyncOperation::AtomicRmw, "atomic-rmw"),
    consan_enum(ConSanSyncOperation::AtomicCompareExchange, "atomic-compare-exchange"),
    consan_enum(ConSanSyncOperation::OrdinaryLoad, "ordinary-load"),
    consan_enum(ConSanSyncOperation::OrdinaryStore, "ordinary-store"));

[[nodiscard]] inline const char *sync_operation_name(rocjitsu::ConSanSyncOperation operation) {
  return kSyncOperations.name(operation).data();
}

inline constexpr auto kSyncAddressSources = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanSyncAddressSource::NotApplicable, "not-applicable"),
    consan_enum(ConSanSyncAddressSource::Unknown, "unknown"),
    consan_enum(ConSanSyncAddressSource::LdsVector, "lds-vector"),
    consan_enum(ConSanSyncAddressSource::FlatVector, "flat-vector"),
    consan_enum(ConSanSyncAddressSource::GlobalScalarVector, "global-scalar-vector"),
    consan_enum(ConSanSyncAddressSource::BufferResource, "buffer-resource"),
    consan_enum(ConSanSyncAddressSource::ScratchVector, "scratch-vector"));

[[nodiscard]] inline const char *
sync_address_source_name(rocjitsu::ConSanSyncAddressSource source) {
  return kSyncAddressSources.name(source).data();
}

inline constexpr auto kSyncMemoryRoles = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanSyncMemoryRole::Unknown, "unknown"),
    consan_enum(ConSanSyncMemoryRole::None, "none"),
    consan_enum(ConSanSyncMemoryRole::Acquire, "acquire"),
    consan_enum(ConSanSyncMemoryRole::Release, "release"),
    consan_enum(ConSanSyncMemoryRole::AcquireRelease, "acquire-release"),
    consan_enum(ConSanSyncMemoryRole::SequentiallyConsistent, "sequentially-consistent"));

[[nodiscard]] inline const char *sync_memory_role_name(rocjitsu::ConSanSyncMemoryRole role) {
  return kSyncMemoryRoles.name(role).data();
}

inline constexpr auto kSyncRmwOutcomes = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanSyncRmwOutcome::NotApplicable, "not-applicable"),
    consan_enum(ConSanSyncRmwOutcome::Unknown, "unknown"),
    consan_enum(ConSanSyncRmwOutcome::NoReturn, "no-return"),
    consan_enum(ConSanSyncRmwOutcome::ReturnsOldValue, "returns-old-value"),
    consan_enum(ConSanSyncRmwOutcome::CompareExchange, "compare-exchange"));

[[nodiscard]] inline const char *sync_rmw_outcome_name(rocjitsu::ConSanSyncRmwOutcome outcome) {
  return kSyncRmwOutcomes.name(outcome).data();
}

inline constexpr auto kSyncConfidences =
    make_consan_enum_vocabulary("unknown", consan_enum(ConSanSemanticConfidence::Exact, "exact"),
                                consan_enum(ConSanSemanticConfidence::Conservative, "conservative"),
                                consan_enum(ConSanSemanticConfidence::Ambiguous, "ambiguous"),
                                consan_enum(ConSanSemanticConfidence::Unsupported, "unsupported"));

[[nodiscard]] inline const char *
sync_confidence_name(rocjitsu::ConSanSemanticConfidence confidence) {
  return kSyncConfidences.name(confidence).data();
}

inline constexpr auto kOwnerProofs = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanOwnerProofKind::KernelLocal, "kernel-local"),
    consan_enum(ConSanOwnerProofKind::DirectCall, "direct-call"),
    consan_enum(ConSanOwnerProofKind::RecoveredIndirectCall, "recovered-indirect-call"));

[[nodiscard]] inline const char *owner_proof_name(rocjitsu::ConSanOwnerProofKind proof) {
  return kOwnerProofs.name(proof).data();
}

struct OwnerLogFields {
  std::string names = "-";
  std::string proofs = "-";
};

[[nodiscard]] inline OwnerLogFields
owner_log_fields(std::span<const rocjitsu::ConSanExecutionOwner> owners,
                 std::span<const rocjitsu::ConSanProgramContainer> kernels) {
  OwnerLogFields fields;
  if (owners.empty())
    return fields;
  fields.names.clear();
  fields.proofs.clear();
  for (const rocjitsu::ConSanExecutionOwner &owner : owners) {
    if (!owner.kernel.valid() || owner.kernel.ordinal >= kernels.size() ||
        kernels[owner.kernel.ordinal].id != owner.kernel)
      return {};
    const rocjitsu::ConSanProgramContainer &kernel = kernels[owner.kernel.ordinal];
    if (!fields.names.empty()) {
      fields.names += ',';
      fields.proofs += ',';
    }
    fields.names += kernel.name;
    fields.proofs += owner_proof_name(owner.proof);
  }
  return fields;
}

inline constexpr auto kPatchedImageGrowthLimitKinds = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanPatchedImageGrowthLimitKind::AbsoluteBytes, "absolute-bytes"),
    consan_enum(ConSanPatchedImageGrowthLimitKind::InputPercent, "input-percent"));

[[nodiscard]] inline const char *
patched_image_growth_limit_kind_name(rocjitsu::ConSanPatchedImageGrowthLimitKind kind) {
  return kPatchedImageGrowthLimitKinds.name(kind).data();
}

[[nodiscard]] inline uint64_t
patched_image_growth_limit_value(const rocjitsu::ConSanPatchedImageGrowthLimit &limit) {
  return limit.kind == rocjitsu::ConSanPatchedImageGrowthLimitKind::InputPercent
             ? limit.input_percent
             : limit.absolute_bytes;
}

inline constexpr auto kOwnerSources = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanMoiOwnerSource::Automatic, "automatic"),
    consan_enum(ConSanMoiOwnerSource::WorkitemId, "workitem_id"),
    consan_enum(ConSanMoiOwnerSource::HwId, "hw_id"));

[[nodiscard]] inline const char *owner_source_name(rocjitsu::ConSanMoiOwnerSource source) {
  return kOwnerSources.name(source).data();
}

inline constexpr auto kFlatProvenanceModes =
    make_consan_enum_vocabulary("unknown", consan_enum(ConSanFlatProvenanceMode::Likely, "likely"),
                                consan_enum(ConSanFlatProvenanceMode::Strict, "strict"));

[[nodiscard]] inline const char *
flat_provenance_mode_name(rocjitsu::ConSanFlatProvenanceMode mode) {
  return kFlatProvenanceModes.name(mode).data();
}

inline constexpr auto kCheckTrapModes = make_consan_enum_vocabulary(
    "unknown", consan_enum(CheckTrapMode::All, "all"), consan_enum(CheckTrapMode::Lds, "lds"),
    consan_enum(CheckTrapMode::Flat, "flat"));

[[nodiscard]] inline const char *check_trap_mode_name(CheckTrapMode mode) {
  return kCheckTrapModes.name(mode).data();
}

[[nodiscard]] inline const char *sc_report_mode_name(ScReportMode mode) {
  return mode == ScReportMode::Auto ? "auto" : "trap";
}

[[nodiscard]] inline const char *hook_policy_name(HookPolicy policy) {
  return policy == HookPolicy::Default ? "default" : "strict";
}
void reject_auto_moi_report_plan(uint64_t reader, uint64_t required_size, uint64_t configured_cap,
                                 std::string_view reason);
[[nodiscard]] bool allocate_auto_moi_report_buffer(
    CoreApiTable *core, hsa_agent_t agent, uint64_t reader, uint64_t required_size,
    uint64_t requested_size, uint64_t configured_cap, const ConSanMoiReportBufferLayout &layout,
    bool track_barriers, bool track_atomics, bool test_seed_inline_exact_odd, uint64_t *address,
    uint64_t *registered_size, uint64_t *registered_generation);
void register_auto_moi_report_metadata(uint64_t reader, uint64_t generation,
                                       std::string_view input_fingerprint,
                                       const ConSanRuntimeStaticMapping &static_mapping);
void bind_auto_moi_report_buffer_to_executable(uint64_t reader, uint64_t generation,
                                               hsa_executable_t executable);
void discard_auto_moi_report_buffer(CoreApiTable *core, uint64_t reader, uint64_t generation);
void retire_auto_moi_report_buffers(CoreApiTable *core, hsa_executable_t executable);
[[nodiscard]] AutoMoiReportSummary summarize_and_clear_auto_moi_report_buffers(CoreApiTable *core);

/// Fully typed transform seam observed by HSA-hook unit tests. A test double
/// receives the same immutable contracts and returns the same production
/// result as the real transformer; raw mechanism fixtures never cross this
/// hook boundary.
using ConSanTransformOverride = TransformResult (*)(std::span<const uint8_t>, const ConSanRequest &,
                                                    const TransformPolicy &, const RuntimePolicy &,
                                                    const ConSanDebugOverrides &,
                                                    const MutationRequest &,
                                                    const RuntimeCapabilities &,
                                                    const BoundRuntimeResources &);
using LogSinkOverride = void (*)(const char *, size_t);

extern std::atomic<int> g_log_level;
extern std::atomic<uint64_t> g_dump_sequence;
extern std::atomic<ConSanTransformOverride> g_test_consan_transform_override;

[[nodiscard]] std::optional<HookConfig> parse_config();
[[nodiscard]] bool refresh_report_config_from_env(HookConfig *config);

[[gnu::format(printf, 2, 3)]] void log_message(int required_level, const char *format, ...);

} // namespace rocjitsu::consan_hook
