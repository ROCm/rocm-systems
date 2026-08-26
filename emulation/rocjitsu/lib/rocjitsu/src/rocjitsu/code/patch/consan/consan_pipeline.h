// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_pipeline.h
/// @brief Typed orchestration and static-result boundary for ConSan.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

/// Names the ordered contracts crossed by one production ConSan transform.
///
/// The order is the dependency order visible to callers, not a claim that the
/// compatibility lowerer has already been physically split into one function
/// per value. In particular, `LegacyConSanLowering` temporarily produces the
/// inventory and observation artifacts that the pipeline publishes at their
/// logical boundaries. Every pipeline result records every stage exactly once;
/// a stage that does not apply or awaits runtime binding still has an explicit
/// status. `Count` is an iteration sentinel and never denotes work.
enum class ConSanPipelineStage : uint8_t {
  Configuration,
  TargetAndRuntimeCapabilities,
  ProgramInventory,
  ObservationPlan,
  EvidenceRequirements,
  RuntimeBinding,
  LegacyLowering,
  FinalValidation,
  Complete,
  Count,
};

/// Complete dependency-ordered set of ConSan transformation stages.
inline constexpr std::array<ConSanPipelineStage, 9> kConSanPipelineStages = {
    ConSanPipelineStage::Configuration,
    ConSanPipelineStage::TargetAndRuntimeCapabilities,
    ConSanPipelineStage::ProgramInventory,
    ConSanPipelineStage::ObservationPlan,
    ConSanPipelineStage::EvidenceRequirements,
    ConSanPipelineStage::RuntimeBinding,
    ConSanPipelineStage::LegacyLowering,
    ConSanPipelineStage::FinalValidation,
    ConSanPipelineStage::Complete,
};

/// Return the stable diagnostic spelling of a pipeline stage. Sentinels and
/// out-of-range values deliberately cannot acquire a plausible stage name.
[[nodiscard]] constexpr std::string_view consan_pipeline_stage_name(ConSanPipelineStage stage) {
  switch (stage) {
  case ConSanPipelineStage::Configuration:
    return "configuration";
  case ConSanPipelineStage::TargetAndRuntimeCapabilities:
    return "target-and-runtime-capabilities";
  case ConSanPipelineStage::ProgramInventory:
    return "program-inventory";
  case ConSanPipelineStage::ObservationPlan:
    return "observation-plan";
  case ConSanPipelineStage::EvidenceRequirements:
    return "evidence-requirements";
  case ConSanPipelineStage::RuntimeBinding:
    return "runtime-binding";
  case ConSanPipelineStage::LegacyLowering:
    return "legacy-lowering";
  case ConSanPipelineStage::FinalValidation:
    return "final-validation";
  case ConSanPipelineStage::Complete:
    return "complete";
  case ConSanPipelineStage::Count:
    break;
  }
  return "invalid-pipeline-stage";
}

static_assert(kConSanPipelineStages.size() == static_cast<size_t>(ConSanPipelineStage::Count));

/// Describes how far one typed pipeline contract progressed.
///
/// `Completed` means the stage produced and validated its promised value.
/// `Deferred` is a successful static result that intentionally awaits a later
/// runtime allocation or dispatch binding; it is neither an error nor silent
/// success. `NotApplicable` means the selected mode or result needs no value
/// from that stage. `Unsupported` represents a known input outside the current
/// contract, while `Invalid` means an input or produced value violated a
/// required invariant. `Count` is never a recorded status.
enum class ConSanPipelineStageStatus : uint8_t {
  Completed,
  Deferred,
  NotApplicable,
  Unsupported,
  Invalid,
  Count,
};

/// Complete iterable set of pipeline-stage statuses.
inline constexpr std::array<ConSanPipelineStageStatus, 5> kConSanPipelineStageStatuses = {
    ConSanPipelineStageStatus::Completed,     ConSanPipelineStageStatus::Deferred,
    ConSanPipelineStageStatus::NotApplicable, ConSanPipelineStageStatus::Unsupported,
    ConSanPipelineStageStatus::Invalid,
};

/// Return the stable diagnostic spelling of a stage status.
[[nodiscard]] constexpr std::string_view
consan_pipeline_stage_status_name(ConSanPipelineStageStatus status) {
  switch (status) {
  case ConSanPipelineStageStatus::Completed:
    return "completed";
  case ConSanPipelineStageStatus::Deferred:
    return "deferred";
  case ConSanPipelineStageStatus::NotApplicable:
    return "not-applicable";
  case ConSanPipelineStageStatus::Unsupported:
    return "unsupported";
  case ConSanPipelineStageStatus::Invalid:
    return "invalid";
  case ConSanPipelineStageStatus::Count:
    break;
  }
  return "invalid-pipeline-stage-status";
}

static_assert(kConSanPipelineStageStatuses.size() ==
              static_cast<size_t>(ConSanPipelineStageStatus::Count));

/// Immutable status record for one pipeline contract and one pristine image.
///
/// Repeating the full collision-aware code-object identity on every record
/// makes accidental stage composition across two images directly detectable.
/// `contract_issue` is populated only when typed configuration or capability
/// validation produced a `ConSanContractIssue`; other stage failures use the
/// transform's structured issue list.
struct ConSanPipelineStageRecord {
  /// Contract whose result this record describes.
  ConSanPipelineStage stage = ConSanPipelineStage::Count;
  /// Completion, deferral, exclusion, or failure state of the contract.
  ConSanPipelineStageStatus status = ConSanPipelineStageStatus::Invalid;
  /// Pristine input identity that every stage in the transform must share.
  ConSanCodeObjectId code_object;
  /// Narrow configuration/capability failure, or `None` for other outcomes.
  ConSanContractIssue contract_issue = ConSanContractIssue::None;

  /// Return whether the record uses valid enum values and a coherent typed
  /// contract issue. Pipeline order and cross-record identity are validated by
  /// `TransformResult::well_formed()`.
  [[nodiscard]] bool well_formed() const;

  bool operator==(const ConSanPipelineStageRecord &) const = default;
};

/// Typed category for a static transform failure retained by `TransformResult`.
///
/// Configuration and runtime-capability failures carry a
/// `ConSanContractIssue`. Inventory, policy, evidence, legacy lowering, and
/// final validation failures instead retain their owning pipeline stage and a
/// diagnostic detail. This enum is intentionally about code-object rewriting;
/// runtime race verdicts and device-evidence loss belong to `RunVerdict`, not
/// here. `Count` is an invalid iteration sentinel.
enum class ConSanTransformIssueKind : uint8_t {
  Contract,
  Inventory,
  ObservationPlan,
  EvidenceRequirements,
  RuntimeBinding,
  LegacyLowering,
  FinalValidation,
  Count,
};

/// Complete iterable set of static transform issue categories.
inline constexpr std::array<ConSanTransformIssueKind, 7> kConSanTransformIssueKinds = {
    ConSanTransformIssueKind::Contract,        ConSanTransformIssueKind::Inventory,
    ConSanTransformIssueKind::ObservationPlan, ConSanTransformIssueKind::EvidenceRequirements,
    ConSanTransformIssueKind::RuntimeBinding,  ConSanTransformIssueKind::LegacyLowering,
    ConSanTransformIssueKind::FinalValidation,
};

/// Return the stable diagnostic spelling of a transform issue category.
[[nodiscard]] constexpr std::string_view
consan_transform_issue_kind_name(ConSanTransformIssueKind kind) {
  switch (kind) {
  case ConSanTransformIssueKind::Contract:
    return "contract";
  case ConSanTransformIssueKind::Inventory:
    return "inventory";
  case ConSanTransformIssueKind::ObservationPlan:
    return "observation-plan";
  case ConSanTransformIssueKind::EvidenceRequirements:
    return "evidence-requirements";
  case ConSanTransformIssueKind::RuntimeBinding:
    return "runtime-binding";
  case ConSanTransformIssueKind::LegacyLowering:
    return "legacy-lowering";
  case ConSanTransformIssueKind::FinalValidation:
    return "final-validation";
  case ConSanTransformIssueKind::Count:
    break;
  }
  return "invalid-transform-issue-kind";
}

static_assert(kConSanTransformIssueKinds.size() ==
              static_cast<size_t>(ConSanTransformIssueKind::Count));

/// One machine-readable static-transform failure with optional prose context.
///
/// `stage` identifies the contract that owns the failure. A `Contract` issue
/// must carry a non-`None` `contract_issue`; every other kind must leave that
/// field `None` and provide nonempty detail. This prevents callers from parsing
/// a message to identify configuration failures while still preserving exact
/// diagnostics emitted by the compatibility lowerer.
struct ConSanTransformIssue {
  /// Stable category consumed by control flow and tests.
  ConSanTransformIssueKind kind = ConSanTransformIssueKind::Count;
  /// Pipeline contract that owns this failure.
  ConSanPipelineStage stage = ConSanPipelineStage::Count;
  /// Typed request/capability issue for `Contract`, otherwise `None`.
  ConSanContractIssue contract_issue = ConSanContractIssue::None;
  /// Human-readable context that is never used to recover the category.
  std::string detail;

  /// Return whether category, stage, typed payload, and diagnostic context
  /// obey the issue contract.
  [[nodiscard]] bool well_formed() const;

  bool operator==(const ConSanTransformIssue &) const = default;
};

/// Static output of one typed ConSan transformation attempt.
///
/// This type separates caller-facing stage state, immutable semantic artifacts,
/// address-free evidence requirements, validated replacement bytes, and typed
/// failures from the prototype's large mutable `ConSanResult`. The private
/// compatibility value retains only mechanism telemetry and fields not yet
/// migrated. It is exposed read-only while those fields acquire narrower
/// production owners; caller-visible outcome, installation, diagnostics, and
/// replacement storage must not be reconstructed from it. A result never
/// contains runtime conflict evidence and makes no race-free claim. Public
/// fields allow precise construction and invariant testing during the
/// migration, while production creates values only through `transform_consan`
/// or `transform_consan_with_mutation`.
class TransformResult {
public:
  TransformResult() = default;

  /// Collision-aware identity of the pristine input image.
  ConSanCodeObjectId code_object;
  /// Every pipeline contract exactly once in dependency order.
  std::vector<ConSanPipelineStageRecord> stages;
  /// Valid immutable original-program facts when inventory succeeded.
  ProgramInventory program_inventory;
  /// Valid target-neutral policy output when policy was applicable.
  ConSanObservationPlan observation_plan;
  /// Static semantic/lowering coverage corresponding to `observation_plan`.
  ConSanCoverageLedger coverage_ledger;
  /// Address-free engine-specific report/marker contract when applicable.
  std::optional<ConSanEvidenceRequirements> evidence_requirements;
  /// Validated replacement image only for `ModifiedValid`.
  std::vector<uint8_t> replacement_bytes;
  /// Final static transform classification.
  ConSanTransformOutcome outcome = ConSanTransformOutcome::Invalid;
  /// True only after structural and semantic validation of replacement bytes.
  bool final_validation_passed = false;
  /// First typed configuration failure, or `None` after valid configuration.
  ConSanContractIssue configuration_issue = ConSanContractIssue::None;
  /// Machine-readable failures owned by static pipeline stages.
  std::vector<ConSanTransformIssue> issues;
  /// Non-fatal diagnostics retained from analysis and lowering.
  std::vector<std::string> warnings;

  /// Return the record for one stage, or null for an invalid stage or malformed
  /// result that omitted it.
  [[nodiscard]] const ConSanPipelineStageRecord *stage(ConSanPipelineStage value) const;

  /// Verify ordering, pristine identity, artifact relationships, status, and
  /// replacement-image invariants without consulting mutable global state.
  [[nodiscard]] bool well_formed() const;

  /// Derive loader policy solely from the split static result.
  [[nodiscard]] ConSanInstallAction install_action(bool fail_closed) const;

  /// Read lowering telemetry whose typed production owner has not yet been
  /// extracted. Runtime callers may use this for patch geometry, mutation
  /// telemetry, and native-resource details only. Installation policy,
  /// replacement bytes, diagnostics, inventory, observation policy, and
  /// coverage are owned by this `TransformResult` and must be read here.
  [[nodiscard]] const ConSanResult &legacy_mechanism() const { return legacy_compatibility_; }

  /// Demote an otherwise installable transform after a runtime-owned resource
  /// operation fails. This keeps the typed outcome, stage records, replacement
  /// storage, and temporary mechanism telemetry coherent without allowing the
  /// runtime adapter to mutate either representation field by field.
  void discard_replacement(std::string warning);

private:
  friend class LegacyConSanLowering;
  friend TransformResult transform_consan(std::span<const uint8_t>, const ConSanRequest &,
                                          const TransformPolicy &, const RuntimePolicy &,
                                          const ConSanDebugOverrides &, const RuntimeCapabilities &,
                                          const BoundRuntimeResources &);
  friend TransformResult
  transform_consan_with_mutation(std::span<const uint8_t>, const ConSanRequest &,
                                 const TransformPolicy &, const RuntimePolicy &,
                                 const ConSanDebugOverrides &, const MutationRequest &,
                                 const RuntimeCapabilities &, const BoundRuntimeResources &);

  /// Prototype mechanism result with migrated artifacts moved out.
  ConSanResult legacy_compatibility_;
};

/// Sole compatibility component allowed to invoke the prototype patcher.
///
/// It converts immutable typed inputs into one fresh `ConSanOptions`, invokes
/// the existing probe/resource/placement implementation, and returns its
/// mutable result only to the new pipeline. It owns no semantic policy and no
/// caller-visible installation decision. Keeping this boundary named makes
/// every remaining legacy responsibility searchable and independently
/// removable as lowerer components migrate.
class LegacyConSanLowering {
public:
  /// Run the pristine MOI inventory pass used before the HSA adapter allocates
  /// and binds an automatic report buffer.
  ///
  /// `preserve_extended_barrier_pairs` carries one fact computed from the live
  /// mutation request before that request is deliberately disabled for the
  /// pristine pass. It exists only because the prototype represents this
  /// two-pass composition as a mutable option; the future inventory component
  /// will receive the later mutation requirements directly. No caller may use
  /// this entry for an installable transform.
  [[nodiscard]] static TransformResult run_pristine_moi_inventory(
      std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
      const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
      const ConSanDebugOverrides &debug, const MutationRequest &disabled_mutation,
      const RuntimeCapabilities &capabilities, const BoundRuntimeResources &unbound_resources,
      bool preserve_extended_barrier_pairs);

  /// Bind runtime resources and re-run only MOI planning and lowering from a
  /// pristine inventory returned by `run_pristine_moi_inventory`. The HSA
  /// coordinator never sees or reconstructs the prototype retry value.
  [[nodiscard]] static TransformResult retry_pristine_moi_inventory(
      std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
      const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
      const ConSanDebugOverrides &debug, const MutationRequest &mutation,
      const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
      TransformResult inventory);

  /// Publish a mechanism result already produced by the inventory-retry or
  /// hook-test seam through the same typed pipeline contract as an ordinary
  /// lowering. This is the only compatibility entry for results not produced
  /// by `run()` and disappears with the legacy retry boundary.
  [[nodiscard]] static TransformResult
  publish(std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
          const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
          const ConSanDebugOverrides &debug, const MutationRequest &mutation,
          const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
          ConSanResult legacy_result);

private:
  friend TransformResult
  transform_consan_with_mutation(std::span<const uint8_t>, const ConSanRequest &,
                                 const TransformPolicy &, const RuntimePolicy &,
                                 const ConSanDebugOverrides &, const MutationRequest &,
                                 const RuntimeCapabilities &, const BoundRuntimeResources &);

  [[nodiscard]] static TransformResult
  publish_optional(std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
                   const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
                   const ConSanDebugOverrides &debug, const MutationRequest &mutation,
                   const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
                   std::optional<ConSanResult> legacy_result);
};

/// Run the ordinary observation pipeline. Fault mutation and timing
/// perturbation are deliberately absent from this entry point.
[[nodiscard]] TransformResult
transform_consan(std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
                 const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
                 const ConSanDebugOverrides &debug, const RuntimeCapabilities &capabilities,
                 const BoundRuntimeResources &resources);

/// Run validation-only mutation/perturbation composition through a distinct
/// entry point. The legacy adapter currently owns the internal staged-image
/// mechanics; the returned result still obeys the same static pipeline
/// contract as an ordinary transform.
[[nodiscard]] TransformResult transform_consan_with_mutation(
    std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const ConSanDebugOverrides &debug, const MutationRequest &mutation,
    const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources);

} // namespace rocjitsu
