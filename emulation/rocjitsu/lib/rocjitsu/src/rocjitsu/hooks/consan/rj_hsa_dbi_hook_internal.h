// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "hsa/hsa_api_trace_minimal.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_pipeline.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_conflict_analysis.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_decoder.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_snapshot.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_trust.h"

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

namespace rocjitsu::consan::hook {

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

enum class SuperColliderReportMode : uint8_t {
  Auto,
  Trap,
};

enum class HookPolicy : uint8_t {
  Default,
  Strict,
};

enum class EpochCheckpointStatus : uint32_t {
  Complete = 0,
  Inactive = 1,
  ModeDoesNotUseReports = 2,
  ReportSnapshotFailed = 3,
};

struct ReportCheckpointResult {
  EpochCheckpointStatus status = EpochCheckpointStatus::Complete;
  uint64_t report_count = 0;
};

// This is an implementation safety ceiling, not a user-facing selection
// policy. It avoids the unbounded vector reservations that UINT32_MAX would
// trigger in planners while exceeding the supported-site count of current
// production code objects by orders of magnitude.
constexpr uint32_t kAllSupportedPatchBudget = 65536;
// Bound an interposed loader wait so a stalled owner cannot deadlock the
// process. Campaigns whose individual code-object load can legitimately hold
// the reservation longer must raise this alongside their workload deadline.
constexpr uint32_t kDefaultFaultReservationTimeoutMs = 30000;

/// Fully parsed hook configuration.
///
/// The six public base subobjects are the input contracts constructed by the
/// environment parser. The remaining members are hook-local presentation,
/// allocation-mode, and parsing-provenance state; they are deliberately not
/// forwarded to lowering. Inheritance preserves the parser's
/// concise field spelling while allowing production callers to pass each
/// immutable contract independently.
struct HookConfig : Request,
                    TransformPolicy,
                    RuntimePolicy,
                    DebugOverrides,
                    MutationRequest,
                    BoundRuntimeResources {
  HookPolicy policy = HookPolicy::Default;
  CheckTrapMode check_trap_mode = CheckTrapMode::All;
  SuperColliderReportMode supercollider_report_mode = SuperColliderReportMode::Auto;
  bool auto_report_buffer_size_explicit = false;
  bool max_patches_explicit = false;
  bool runtime_sample_stride_explicit = false;
  bool sampling_controls_explicit = false;
  const char *preset = "default";
  bool allow_uniform_lds_stores = false;
  uint32_t conflict_limit = 8;
  uint32_t total_conflict_limit = 64;
  enum class EpochAnalysisKind : uint8_t { Every, Nth, Periodic, Manual };
  struct EpochAnalysisPolicy {
    EpochAnalysisKind kind = EpochAnalysisKind::Every;
    uint64_t value = 1;
    uint64_t offset = 1;

    [[nodiscard]] bool selects(uint64_t epoch, bool manual_window_open) const {
      switch (kind) {
      case EpochAnalysisKind::Every:
        return true;
      case EpochAnalysisKind::Nth:
        return epoch == value;
      case EpochAnalysisKind::Periodic:
        return epoch >= offset && (epoch - offset) % value == 0;
      case EpochAnalysisKind::Manual:
        return manual_window_open;
      }
      return false;
    }
  };
  EpochAnalysisPolicy epoch_analysis;
  int log_level = kLogDisabled;
  std::string dump_dir;

  HookConfig() {
    max_patches = kAllSupportedPatchBudget;
    max_patches_is_expert_limit = false;
    fault_reservation_timeout_ms = kDefaultFaultReservationTimeoutMs;
  }
};

constexpr std::string_view kStandardProfile = "standard-v1";

inline constexpr auto kFaultSiteKinds = make_enum_vocabulary(
    "unknown", enum_entry(FaultSiteKind::Barrier, "barrier"),
    enum_entry(FaultSiteKind::Atomic, "atomic"), enum_entry(FaultSiteKind::LdsAccess, "lds-access"),
    enum_entry(FaultSiteKind::OrdinaryMemory, "ordinary-memory"));

[[nodiscard]] inline const char *fault_site_kind_name(FaultSiteKind kind) {
  return kFaultSiteKinds.name(kind).data();
}

inline constexpr auto kOrdinaryMemorySupportReasons = make_enum_vocabulary(
    "unknown", enum_entry(OrdinaryMemorySupportReason::NotApplicable, "not-applicable"),
    enum_entry(OrdinaryMemorySupportReason::Supported, "supported"),
    enum_entry(OrdinaryMemorySupportReason::SupportedSynchronizationOnly,
               "supported-synchronization-only"),
    enum_entry(OrdinaryMemorySupportReason::UnsupportedArchitecture, "unsupported-architecture"),
    enum_entry(OrdinaryMemorySupportReason::UnsupportedEncodingSize, "unsupported-encoding-size"),
    enum_entry(OrdinaryMemorySupportReason::MalformedEncoding, "malformed-encoding"),
    enum_entry(OrdinaryMemorySupportReason::MissingAddressVgpr, "missing-address-vgpr"),
    enum_entry(OrdinaryMemorySupportReason::MissingDestinationVgpr, "missing-destination-vgpr"),
    enum_entry(OrdinaryMemorySupportReason::MissingValueVgpr, "missing-value-vgpr"));

[[nodiscard]] inline const char *
ordinary_memory_support_reason_name(OrdinaryMemorySupportReason reason) {
  return kOrdinaryMemorySupportReasons.name(reason).data();
}

inline constexpr auto kFaultMutationKinds = make_enum_vocabulary(
    "unknown", enum_entry(FaultMutationKind::DropBarrier, "drop-barrier"),
    enum_entry(FaultMutationKind::MoveBarrierPair, "move-barrier-pair"),
    enum_entry(FaultMutationKind::BarrierIdScope, "barrier-id-scope"),
    enum_entry(FaultMutationKind::BarrierParticipantCount, "barrier-participant-count"),
    enum_entry(FaultMutationKind::AtomicWrongAddress, "atomic-wrong-address"),
    enum_entry(FaultMutationKind::AtomicWeakenOrder, "atomic-weaken-order"),
    enum_entry(FaultMutationKind::AtomicWeakenScope, "atomic-weaken-scope"),
    enum_entry(FaultMutationKind::LdsWrongAddress, "lds-wrong-address"),
    enum_entry(FaultMutationKind::OrdinaryWeakenOrder, "ordinary-weaken-order"),
    enum_entry(FaultMutationKind::OrdinaryWrongAddress, "ordinary-wrong-address"),
    enum_entry(FaultMutationKind::OrdinaryWeakenScope, "ordinary-weaken-scope"));

[[nodiscard]] inline const char *fault_mutation_kind_name(FaultMutationKind kind) {
  return kFaultMutationKinds.name(kind).data();
}

inline constexpr auto kBarrierMoveDirections =
    make_enum_vocabulary("unknown", enum_entry(BarrierMoveDirection::LegacyMarker, "legacy-marker"),
                         enum_entry(BarrierMoveDirection::Earlier, "earlier"),
                         enum_entry(BarrierMoveDirection::Later, "later"));

[[nodiscard]] inline const char *barrier_move_direction_name(BarrierMoveDirection direction) {
  return kBarrierMoveDirections.name(direction).data();
}

inline constexpr auto kBarrierMoveCfgContracts =
    make_enum_vocabulary("unknown", enum_entry(BarrierMoveCfgContract::SameBlock, "same-block"),
                         enum_entry(BarrierMoveCfgContract::CompletingStructuredDiamond,
                                    "completing-structured-diamond"),
                         enum_entry(BarrierMoveCfgContract::DestructiveStructuredExecDiamond,
                                    "destructive-structured-exec-diamond"));

[[nodiscard]] inline const char *barrier_move_cfg_contract_name(BarrierMoveCfgContract contract) {
  return kBarrierMoveCfgContracts.name(contract).data();
}

inline constexpr auto kSyncSequenceKinds = make_enum_vocabulary(
    "unknown", enum_entry(SyncKind::Barrier, "barrier"), enum_entry(SyncKind::Fence, "fence"),
    enum_entry(SyncKind::Atomic, "atomic"),
    enum_entry(SyncKind::OrdinaryMemory, "ordinary-memory"));

[[nodiscard]] inline const char *sync_sequence_kind_name(SyncKind kind) {
  return kSyncSequenceKinds.name(kind).data();
}

inline constexpr auto kSyncOperations = make_enum_vocabulary(
    "unknown", enum_entry(SyncOperation::Unknown, "unknown"),
    enum_entry(SyncOperation::BarrierSignal, "barrier-signal"),
    enum_entry(SyncOperation::BarrierWait, "barrier-wait"),
    enum_entry(SyncOperation::BarrierFull, "barrier-full"),
    enum_entry(SyncOperation::BarrierInit, "barrier-init"),
    enum_entry(SyncOperation::BarrierJoin, "barrier-join"),
    enum_entry(SyncOperation::BarrierLeave, "barrier-leave"),
    enum_entry(SyncOperation::BarrierWakeup, "barrier-wakeup"),
    enum_entry(SyncOperation::BarrierStateQuery, "barrier-state-query"),
    enum_entry(SyncOperation::Fence, "fence"), enum_entry(SyncOperation::AtomicRmw, "atomic-rmw"),
    enum_entry(SyncOperation::AtomicCompareExchange, "atomic-compare-exchange"),
    enum_entry(SyncOperation::OrdinaryLoad, "ordinary-load"),
    enum_entry(SyncOperation::OrdinaryStore, "ordinary-store"));

[[nodiscard]] inline const char *sync_operation_name(SyncOperation operation) {
  return kSyncOperations.name(operation).data();
}

inline constexpr auto kSyncAddressSources =
    make_enum_vocabulary("unknown", enum_entry(SyncAddressSource::NotApplicable, "not-applicable"),
                         enum_entry(SyncAddressSource::Unknown, "unknown"),
                         enum_entry(SyncAddressSource::LdsVector, "lds-vector"),
                         enum_entry(SyncAddressSource::FlatVector, "flat-vector"),
                         enum_entry(SyncAddressSource::GlobalScalarVector, "global-scalar-vector"),
                         enum_entry(SyncAddressSource::BufferResource, "buffer-resource"),
                         enum_entry(SyncAddressSource::ScratchVector, "scratch-vector"));

[[nodiscard]] inline const char *sync_address_source_name(SyncAddressSource source) {
  return kSyncAddressSources.name(source).data();
}

inline constexpr auto kSyncMemoryRoles = make_enum_vocabulary(
    "unknown", enum_entry(SyncMemoryRole::Unknown, "unknown"),
    enum_entry(SyncMemoryRole::None, "none"), enum_entry(SyncMemoryRole::Acquire, "acquire"),
    enum_entry(SyncMemoryRole::Release, "release"),
    enum_entry(SyncMemoryRole::AcquireRelease, "acquire-release"),
    enum_entry(SyncMemoryRole::SequentiallyConsistent, "sequentially-consistent"));

[[nodiscard]] inline const char *sync_memory_role_name(SyncMemoryRole role) {
  return kSyncMemoryRoles.name(role).data();
}

inline constexpr auto kSyncRmwOutcomes =
    make_enum_vocabulary("unknown", enum_entry(SyncRmwOutcome::NotApplicable, "not-applicable"),
                         enum_entry(SyncRmwOutcome::Unknown, "unknown"),
                         enum_entry(SyncRmwOutcome::NoReturn, "no-return"),
                         enum_entry(SyncRmwOutcome::ReturnsOldValue, "returns-old-value"),
                         enum_entry(SyncRmwOutcome::CompareExchange, "compare-exchange"));

[[nodiscard]] inline const char *sync_rmw_outcome_name(SyncRmwOutcome outcome) {
  return kSyncRmwOutcomes.name(outcome).data();
}

inline constexpr auto kSyncConfidences =
    make_enum_vocabulary("unknown", enum_entry(SemanticConfidence::Exact, "exact"),
                         enum_entry(SemanticConfidence::Conservative, "conservative"),
                         enum_entry(SemanticConfidence::Ambiguous, "ambiguous"),
                         enum_entry(SemanticConfidence::Unsupported, "unsupported"));

[[nodiscard]] inline const char *sync_confidence_name(SemanticConfidence confidence) {
  return kSyncConfidences.name(confidence).data();
}

inline constexpr auto kOwnerProofs = make_enum_vocabulary(
    "unknown", enum_entry(OwnerProofKind::KernelLocal, "kernel-local"),
    enum_entry(OwnerProofKind::DirectCall, "direct-call"),
    enum_entry(OwnerProofKind::RecoveredIndirectCall, "recovered-indirect-call"));

[[nodiscard]] inline const char *owner_proof_name(OwnerProofKind proof) {
  return kOwnerProofs.name(proof).data();
}

struct OwnerLogFields {
  std::string names = "-";
  std::string proofs = "-";
};

[[nodiscard]] inline OwnerLogFields owner_log_fields(std::span<const ExecutionOwner> owners,
                                                     std::span<const ProgramContainer> kernels) {
  OwnerLogFields fields;
  if (owners.empty())
    return fields;
  fields.names.clear();
  fields.proofs.clear();
  for (const ExecutionOwner &owner : owners) {
    if (!owner.kernel.valid() || owner.kernel.ordinal >= kernels.size() ||
        kernels[owner.kernel.ordinal].id != owner.kernel)
      return {};
    const ProgramContainer &kernel = kernels[owner.kernel.ordinal];
    if (!fields.names.empty()) {
      fields.names += ',';
      fields.proofs += ',';
    }
    fields.names += kernel.name;
    fields.proofs += owner_proof_name(owner.proof);
  }
  return fields;
}

inline constexpr auto kPatchedImageGrowthLimitKinds = make_enum_vocabulary(
    "unknown", enum_entry(PatchedImageGrowthLimitKind::AbsoluteBytes, "absolute-bytes"),
    enum_entry(PatchedImageGrowthLimitKind::InputPercent, "input-percent"));

[[nodiscard]] inline const char *
patched_image_growth_limit_kind_name(PatchedImageGrowthLimitKind kind) {
  return kPatchedImageGrowthLimitKinds.name(kind).data();
}

[[nodiscard]] inline uint64_t
patched_image_growth_limit_value(const PatchedImageGrowthLimit &limit) {
  return limit.kind == PatchedImageGrowthLimitKind::InputPercent ? limit.input_percent
                                                                 : limit.absolute_bytes;
}

inline constexpr auto kOwnerSources = make_enum_vocabulary(
    "unknown", enum_entry(OwnerSource::Automatic, "automatic"),
    enum_entry(OwnerSource::WorkitemId, "workitem_id"), enum_entry(OwnerSource::HwId, "hw_id"));

[[nodiscard]] inline const char *owner_source_name(OwnerSource source) {
  return kOwnerSources.name(source).data();
}

inline constexpr auto kFlatProvenanceModes =
    make_enum_vocabulary("unknown", enum_entry(FlatProvenanceMode::Likely, "likely"),
                         enum_entry(FlatProvenanceMode::Strict, "strict"));

[[nodiscard]] inline const char *flat_provenance_mode_name(FlatProvenanceMode mode) {
  return kFlatProvenanceModes.name(mode).data();
}

inline constexpr auto kCheckTrapModes = make_enum_vocabulary(
    "unknown", enum_entry(CheckTrapMode::All, "all"), enum_entry(CheckTrapMode::Lds, "lds"),
    enum_entry(CheckTrapMode::Flat, "flat"));

[[nodiscard]] inline const char *check_trap_mode_name(CheckTrapMode mode) {
  return kCheckTrapModes.name(mode).data();
}

[[nodiscard]] inline const char *supercollider_report_mode_name(SuperColliderReportMode mode) {
  return mode == SuperColliderReportMode::Auto ? "auto" : "trap";
}

[[nodiscard]] inline const char *hook_policy_name(HookPolicy policy) {
  return policy == HookPolicy::Default ? "default" : "strict";
}
void reject_report_plan(uint64_t reader, uint64_t required_size, uint64_t configured_cap,
                        std::string_view reason);
[[nodiscard]] bool advance_report_generation_for_test(uint64_t generation);

[[nodiscard]] bool allocate_report_buffer(CoreApiTable *core, hsa_agent_t agent, uint64_t reader,
                                          uint64_t required_size, uint64_t requested_size,
                                          uint64_t configured_cap, const ReportBufferLayout &layout,
                                          bool track_barriers, bool track_atomics,
                                          uint64_t *address, uint64_t *registered_size,
                                          uint64_t *registered_generation);
void register_report_metadata(uint64_t reader, uint64_t generation,
                              std::string_view input_fingerprint,
                              const StaticAccessMappings &static_mapping);
void bind_report_buffer_to_executable(uint64_t reader, uint64_t generation,
                                      hsa_executable_t executable);
void discard_report_buffer(CoreApiTable *core, uint64_t reader, uint64_t generation);
void retire_report_buffers(CoreApiTable *core, hsa_executable_t executable);
void configure_epoch_analysis(HookConfig::EpochAnalysisPolicy policy, uint32_t conflict_limit = 8,
                              uint32_t total_conflict_limit = 64,
                              bool allow_uniform_lds_stores = false);
[[nodiscard]] bool begin_epoch_analysis_window();
[[nodiscard]] bool end_epoch_analysis_window();
/// Analyze and recycle every live automatic ConSan report after the caller has
/// established device-wide quiescence. The operation is transactional: no
/// report is reset unless every live report has a complete host snapshot.
[[nodiscard]] ReportCheckpointResult
checkpoint_report_buffers_after_device_synchronize(CoreApiTable *core);
[[nodiscard]] ReportCheckpointResult checkpoint_report_buffers_automatically(CoreApiTable *core);
[[nodiscard]] ReportSummary summarize_and_clear_report_buffers(CoreApiTable *core);

/// Fully typed transform seam observed by HSA-hook unit tests. A test double
/// receives the same immutable contracts and returns the same production
/// result as the real transformer; raw mechanism fixtures never cross this
/// hook boundary.
using TransformOverride = TransformResult (*)(std::span<const uint8_t>, const Request &,
                                              const TransformPolicy &, const RuntimePolicy &,
                                              const DebugOverrides &, const MutationRequest &,
                                              const RuntimeCapabilities &,
                                              const BoundRuntimeResources &);
using LogSinkOverride = void (*)(const char *, size_t);

extern std::atomic<int> g_log_level;
extern std::atomic<uint64_t> g_dump_sequence;
extern std::atomic<TransformOverride> g_test_transform_override;

[[nodiscard]] std::optional<HookConfig> parse_config();
void report_config_rejection();
[[nodiscard]] bool refresh_report_config_from_env(HookConfig *config);

[[gnu::format(printf, 2, 3)]] void log_message(int required_level, const char *format, ...);

} // namespace rocjitsu::consan::hook
