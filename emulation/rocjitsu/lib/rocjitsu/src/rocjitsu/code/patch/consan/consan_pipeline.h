// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_pipeline.h
/// @brief Typed orchestration and static-result boundary for ConSan.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_report_contract.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace rocjitsu {

/// Names the ordered contracts crossed by one production ConSan transform.
///
/// The order is the dependency order visible to callers, not a claim that the
/// internal native lowerer has already been physically split into one function
/// per value. The lowerer produces the inventory and
/// observation artifacts that the pipeline publishes at their logical
/// boundaries. Every pipeline result records every stage exactly once;
/// a stage that does not apply or awaits runtime binding still has an explicit
/// status. `Count` is an iteration sentinel and never denotes work.
enum class ConSanPipelineStage : uint8_t {
  Configuration,
  TargetAndRuntimeCapabilities,
  ProgramInventory,
  ObservationPlan,
  EvidenceRequirements,
  RuntimeBinding,
  ResourceSolvingAndLowering,
  FinalValidation,
  ResultPublication,
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
    ConSanPipelineStage::ResourceSolvingAndLowering,
    ConSanPipelineStage::FinalValidation,
    ConSanPipelineStage::ResultPublication,
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
  case ConSanPipelineStage::ResourceSolvingAndLowering:
    return "resource-solving-and-lowering";
  case ConSanPipelineStage::FinalValidation:
    return "final-validation";
  case ConSanPipelineStage::ResultPublication:
    return "result-publication";
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
/// success. `Blocked` means the stage did not execute because an earlier
/// dependency failed. `NotApplicable` means the selected mode or result needs
/// no value from that stage. `Unsupported` represents a known input outside
/// the current contract, while `Invalid` means an input or produced value
/// violated a required invariant. `Count` is never a recorded status.
enum class ConSanPipelineStageStatus : uint8_t {
  Completed,
  Deferred,
  Blocked,
  NotApplicable,
  Unsupported,
  Invalid,
  Count,
};

/// Complete iterable set of pipeline-stage statuses.
inline constexpr std::array<ConSanPipelineStageStatus, 6> kConSanPipelineStageStatuses = {
    ConSanPipelineStageStatus::Completed,   ConSanPipelineStageStatus::Deferred,
    ConSanPipelineStageStatus::Blocked,     ConSanPipelineStageStatus::NotApplicable,
    ConSanPipelineStageStatus::Unsupported, ConSanPipelineStageStatus::Invalid,
};

/// Return the stable diagnostic spelling of a stage status.
[[nodiscard]] constexpr std::string_view
consan_pipeline_stage_status_name(ConSanPipelineStageStatus status) {
  switch (status) {
  case ConSanPipelineStageStatus::Completed:
    return "completed";
  case ConSanPipelineStageStatus::Deferred:
    return "deferred";
  case ConSanPipelineStageStatus::Blocked:
    return "blocked";
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

/// Status payload for one position in `TransformResult::stages`.
///
/// The fixed array position owns the stage identity, and `TransformResult`
/// owns the pristine code-object identity once for the complete transform.
/// This value therefore carries only facts that can differ by contract.
/// `contract_issue` is populated only when typed configuration, capability,
/// or binding validation produced a `ConSanContractIssue`; other failures use
/// the transform's structured issue list.
struct ConSanPipelineStageState {
  /// Completion, deferral, exclusion, or failure state of the contract.
  ConSanPipelineStageStatus status = ConSanPipelineStageStatus::Invalid;
  /// Number of times this stage actually executed across all transaction
  /// passes. A blocked stage is therefore distinguishable from an executed
  /// stage whose typed result is unsupported or invalid.
  uint32_t execution_count = 0;
  /// Narrow configuration/capability failure, or `None` for other outcomes.
  ConSanContractIssue contract_issue = ConSanContractIssue::None;

  /// Return whether this payload is coherent for the stage selected by its
  /// fixed array position.
  [[nodiscard]] bool well_formed(ConSanPipelineStage stage) const;

  bool operator==(const ConSanPipelineStageState &) const = default;
};

/// Runtime dispatch facts required by the validated replacement of one kernel.
///
/// A lowerer can increase the fixed private or group segment used by a kernel,
/// and a spill in a dynamically sized private frame can require an additional
/// per-dispatch addend that cannot be represented by the static descriptor
/// alone. `has_instrumented_probe` is a semantic attribution fact: it says at
/// least one target-neutral probe intent owned by this kernel reached the
/// `Instrumented` lowering outcome. The runtime uses that fact only to
/// distinguish dispatches of instrumented kernels from unrelated dispatches;
/// it must not rediscover it from emitted patch kinds.
struct ConSanKernelDispatchRequirement {
  /// Original kernel symbol name used to bind this requirement after loading.
  std::string kernel_name;
  /// Absolute minimum private-segment bytes required by the replacement.
  uint32_t required_private_bytes = 0;
  /// Extra private bytes added to the runtime-selected dynamic frame size.
  uint32_t dynamic_private_addend = 0;
  /// Absolute minimum group-segment bytes required by the replacement.
  uint32_t required_group_bytes = 0;
  /// Whether a successfully lowered probe can execute through this kernel.
  bool has_instrumented_probe = false;

  /// Return whether this kernel requires any dispatch-packet segment update.
  [[nodiscard]] bool has_segment_requirement() const {
    return required_private_bytes != 0u || dynamic_private_addend != 0u ||
           required_group_bytes != 0u;
  }

  /// Verify the symbol identity, useful payload, and dynamic-frame bound.
  [[nodiscard]] bool well_formed() const {
    return !kernel_name.empty() && (has_segment_requirement() || has_instrumented_probe) &&
           dynamic_private_addend <= required_private_bytes;
  }

  bool operator==(const ConSanKernelDispatchRequirement &) const = default;
};

/// Complete per-kernel runtime-dispatch contract for one replacement image.
///
/// Entries are ordered by kernel name and names are unique, so repeated
/// lowering records for a shared or aliased kernel have already been reduced
/// to maximum segment requirements and one semantic instrumentation bit. An
/// empty value is valid when a replacement needs neither packet adjustment nor
/// dispatch attribution. This value owns no executable, symbol, or kernel
/// object handles; the HSA adapter adds those runtime-lifetime identities only
/// after the replacement has loaded successfully.
struct ConSanDispatchRequirements {
  /// Deterministic, name-unique requirements for affected kernels.
  std::vector<ConSanKernelDispatchRequirement> kernels;

  /// Return whether any dynamic frame requires dispatch-packet interception.
  [[nodiscard]] bool requires_packet_interception() const {
    return std::ranges::any_of(kernels, [](const ConSanKernelDispatchRequirement &requirement) {
      return requirement.dynamic_private_addend != 0u;
    });
  }

  /// Verify every entry and the deterministic unique-name ordering.
  [[nodiscard]] bool well_formed() const {
    for (size_t index = 0; index < kernels.size(); ++index) {
      if (!kernels[index].well_formed() ||
          (index != 0u && kernels[index - 1u].kernel_name >= kernels[index].kernel_name)) {
        return false;
      }
    }
    return true;
  }

  bool operator==(const ConSanDispatchRequirements &) const = default;
};

class TransformResult;
struct ConSanTransformDiagnosticReport;
class ConSanDeferredBinding;
class ConSanTransformTransaction;

[[nodiscard]] ConSanTransformDiagnosticReport
consan_transform_diagnostic_report(const TransformResult &result);

/// Optional injected transform executor used by test/runtime adapters while
/// the library retains automatic preparation and resume ownership.
using ConSanTransformExecutor = TransformResult (*)(std::span<const uint8_t>, const ConSanRequest &,
                                                    const TransformPolicy &, const RuntimePolicy &,
                                                    const ConSanDebugOverrides &,
                                                    const MutationRequest &,
                                                    const RuntimeCapabilities &,
                                                    const BoundRuntimeResources &);

/// Static output of one typed ConSan transformation attempt.
///
/// This type separates caller-facing stage state, immutable semantic artifacts,
/// address-free evidence requirements, validated replacement bytes, and typed
/// failures from the prototype's large mutable `ConSanTransformArtifacts`. A result never
/// contains runtime conflict evidence and makes no race-free claim. Public
/// fields allow precise construction and invariant testing during the
/// migration, while production creates values only through `transform_consan`
/// or `transform_consan_with_mutation`.
class TransformResult {
public:
  TransformResult() {
    outcome = ConSanTransformOutcome::Invalid;
    for (ConSanPipelineStageState &state : stages)
      state.status = ConSanPipelineStageStatus::NotApplicable;
  }

  /// Collision-aware identity of the pristine input image.
  ConSanCodeObjectId code_object;
  /// Every pipeline contract exactly once, indexed by `ConSanPipelineStage`.
  std::array<ConSanPipelineStageState, kConSanPipelineStages.size()> stages;
  /// Canonical, address-free translation of flavor policy into the evidence
  /// roles that report sizing must retain. This is published independently of
  /// the ABI-bearing requirement below so later stages never have to recover
  /// policy meaning by inspecting report capacities or emitted patches.
  std::optional<ConSanEvidenceIntentPlan> evidence_intent_plan;
  /// Address-free engine-specific report/marker contract when applicable.
  std::optional<ConSanEvidenceRequirements> evidence_requirements;
  /// Runtime dispatch contract derived once from validated lowering and typed
  /// semantic coverage, then bound to executable symbols by the HSA adapter.
  ConSanDispatchRequirements dispatch_requirements;
  /// Immutable ownership of original code-object/container/access facts.
  ProgramInventory program_inventory;
  /// Target-neutral policy result assembled once for the selected engine.
  ConSanObservationPlan observation_plan;
  /// Joined semantic-policy and authoritative lowering outcomes.
  ConSanCoverageLedger coverage_ledger;
  /// Validated runtime-facing projection of committed lowerings.
  ConSanRuntimeStaticMapping runtime_static_mapping;
  /// Validation-only mutation result and stable applied identity.
  ConSanMutationOutcome mutation;
  /// Independently validated replacement image, or empty when not installable.
  std::vector<uint8_t> replacement;
  /// Final static classification of the transformation attempt.
  ConSanTransformOutcome outcome = ConSanTransformOutcome::Invalid;
  /// Non-fatal diagnostics from analysis, lowering, and binding.
  std::vector<std::string> warnings;
  /// Fatal static-transform diagnostics.
  std::vector<std::string> errors;

  /// Return the state for one stage, or null when `value` is not a real stage.
  [[nodiscard]] const ConSanPipelineStageState *stage(ConSanPipelineStage value) const;

  /// Verify fixed-stage status, artifact relationships, result identity, and
  /// replacement-image invariants without consulting mutable global state.
  [[nodiscard]] bool well_formed() const;

  /// Derive loader policy solely from the split static result.
  [[nodiscard]] ConSanInstallAction install_action(bool fail_closed) const;

  /// Demote an otherwise installable transform after a runtime-owned resource
  /// operation fails. This keeps the outcome, stage records, replacement
  /// storage, and private proof inventory coherent without
  /// allowing the runtime adapter to mutate either representation field by
  /// field.
  void discard_replacement(std::string warning);

private:
  friend struct TransformResultTestAccess;
  friend class ConSanDeferredBinding;
  friend class ConSanTransformTransaction;
  friend ConSanTransformDiagnosticReport
  consan_transform_diagnostic_report(const TransformResult &);
  friend TransformResult transform_consan(std::span<const uint8_t>, const ConSanRequest &,
                                          const TransformPolicy &, const RuntimePolicy &,
                                          const ConSanDebugOverrides &, const RuntimeCapabilities &,
                                          const BoundRuntimeResources &);
  friend TransformResult
  transform_consan_with_mutation(std::span<const uint8_t>, const ConSanRequest &,
                                 const TransformPolicy &, const RuntimePolicy &,
                                 const ConSanDebugOverrides &, const MutationRequest &,
                                 const RuntimeCapabilities &, const BoundRuntimeResources &);
  friend TransformResult resume_consan_automatic_transform(std::span<const uint8_t>,
                                                           const BoundRuntimeResources &,
                                                           ConSanDeferredBinding);
  friend TransformResult cancel_consan_automatic_transform(ConSanDeferredBinding, std::string);

  /// Test-only ingress for a synthetic lowerer product. Production entry
  /// points construct the same transaction directly.
  [[nodiscard]] static TransformResult
  execute_test_transaction(std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
                           const TransformPolicy &transform_policy,
                           const RuntimePolicy &runtime_policy, const ConSanDebugOverrides &debug,
                           const MutationRequest &mutation, const RuntimeCapabilities &capabilities,
                           const BoundRuntimeResources &resources,
                           ConSanTransformArtifacts lowering_artifacts);

  /// Move a lowerer aggregate into the reviewed public products and private
  /// diagnostic storage. Only the transaction publication boundary calls it.
  void publish_lowering_artifacts(ConSanTransformArtifacts lowering);

  /// Reconstitute a lowerer aggregate for the transaction-owned MOI resume
  /// strategy. The caller consumes this result object completely.
  [[nodiscard]] ConSanTransformArtifacts take_lowering_artifacts();

  struct PrivateLoweringArtifacts {
    std::vector<ConSanFaultSite> fault_sites;
    std::vector<ConSanBarrierMoveDestination> barrier_move_destinations;
    std::vector<ConSanFaultMutationPlan> fault_plans;
    std::vector<ConSanCandidateResourcePlan> resource_plans;
    ConSanMoiOperatingPoint moi_operating_point;
    std::vector<ConSanCommittedLowering> staged_moi_sync_lowerings;
    std::vector<ConSanPatchInfo> patches;
  } private_lowering_;
};

/// Immutable pre-binding product for one automatic ConSan transform.
///
/// The value owns the exact input identity, final and inventory mutation
/// provenance, assembled observation/evidence products, executed stage
/// records, and any private inventory retained by the library's selected
/// resume strategy. A runtime may inspect the address-free contract to size
/// and allocate resources, but it cannot select or access retry artifacts.
class ConSanDeferredBinding {
public:
  ConSanDeferredBinding(const ConSanDeferredBinding &) = delete;
  ConSanDeferredBinding &operator=(const ConSanDeferredBinding &) = delete;
  ConSanDeferredBinding(ConSanDeferredBinding &&) noexcept = default;
  ConSanDeferredBinding &operator=(ConSanDeferredBinding &&) noexcept = default;

  [[nodiscard]] const ConSanCodeObjectId &code_object() const {
    return inventory_result_.code_object;
  }
  [[nodiscard]] const MutationRequest &requested_mutation() const { return requested_mutation_; }
  [[nodiscard]] const MutationRequest &inventory_mutation() const { return inventory_mutation_; }
  [[nodiscard]] const ProgramInventory &program_inventory() const {
    return inventory_result_.program_inventory;
  }
  [[nodiscard]] const ConSanObservationPlan &observation_plan() const {
    return inventory_result_.observation_plan;
  }
  [[nodiscard]] const std::optional<ConSanEvidenceIntentPlan> &evidence_intent_plan() const {
    return inventory_result_.evidence_intent_plan;
  }
  [[nodiscard]] const std::optional<ConSanEvidenceRequirements> &evidence_requirements() const {
    return inventory_result_.evidence_requirements;
  }
  [[nodiscard]] const std::array<ConSanPipelineStageState, kConSanPipelineStages.size()> &
  stages() const {
    return inventory_result_.stages;
  }

  /// Verify identity, mutation provenance, address-free evidence, executed
  /// stage state, and the library-owned resume strategy.
  [[nodiscard]] bool well_formed() const;

private:
  enum class ResumeStrategy : uint8_t {
    RelowerFromInput,
    RetryMoiInventory,
    InvokeExecutor,
  };

  friend std::variant<TransformResult, ConSanDeferredBinding>
  prepare_consan_automatic_transform(std::span<const uint8_t>, const ConSanRequest &,
                                     const TransformPolicy &, const RuntimePolicy &,
                                     const ConSanDebugOverrides &, const MutationRequest &,
                                     const RuntimeCapabilities &, ConSanTransformExecutor);
  friend TransformResult resume_consan_automatic_transform(std::span<const uint8_t>,
                                                           const BoundRuntimeResources &,
                                                           ConSanDeferredBinding);
  friend TransformResult cancel_consan_automatic_transform(ConSanDeferredBinding, std::string);

  ConSanDeferredBinding(ConSanRequest request, TransformPolicy transform_policy,
                        RuntimePolicy runtime_policy, ConSanDebugOverrides debug,
                        MutationRequest requested_mutation, MutationRequest inventory_mutation,
                        RuntimeCapabilities capabilities, ResumeStrategy strategy,
                        ConSanTransformExecutor executor, TransformResult inventory_result)
      : request_(std::move(request)), transform_policy_(std::move(transform_policy)),
        runtime_policy_(std::move(runtime_policy)), debug_(std::move(debug)),
        requested_mutation_(std::move(requested_mutation)),
        inventory_mutation_(std::move(inventory_mutation)), capabilities_(std::move(capabilities)),
        strategy_(strategy), executor_(executor), inventory_result_(std::move(inventory_result)) {}

  ConSanRequest request_;
  TransformPolicy transform_policy_;
  RuntimePolicy runtime_policy_;
  ConSanDebugOverrides debug_;
  MutationRequest requested_mutation_;
  MutationRequest inventory_mutation_;
  RuntimeCapabilities capabilities_;
  ResumeStrategy strategy_ = ResumeStrategy::RelowerFromInput;
  ConSanTransformExecutor executor_ = nullptr;
  TransformResult inventory_result_;
};

/// Result of beginning an automatic-binding transform. A completed/failed
/// result needs no allocation; a deferred value publishes the exact
/// address-free contract that `resume_consan_automatic_transform` consumes.
using ConSanAutomaticTransformPreparation = std::variant<TransformResult, ConSanDeferredBinding>;

/// Execute a transform through evidence planning, returning a typed deferred
/// binding value exactly when runtime-owned allocation is required.
[[nodiscard]] ConSanAutomaticTransformPreparation prepare_consan_automatic_transform(
    std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const ConSanDebugOverrides &debug, const MutationRequest &mutation,
    const RuntimeCapabilities &capabilities, ConSanTransformExecutor executor = nullptr);

/// Bind runtime-owned resources and execute the resume strategy selected by
/// the library. The caller cannot substitute inventory or retry mechanics.
[[nodiscard]] TransformResult
resume_consan_automatic_transform(std::span<const uint8_t> code_object_bytes,
                                  const BoundRuntimeResources &resources,
                                  ConSanDeferredBinding deferred);

/// End a deferred transaction after runtime allocation fails, publishing the
/// same coherent non-installable result shape as other binding failures.
[[nodiscard]] TransformResult cancel_consan_automatic_transform(ConSanDeferredBinding deferred,
                                                                std::string warning);

/// Run the ordinary observation pipeline. Fault mutation and timing
/// perturbation are deliberately absent from this entry point.
[[nodiscard]] TransformResult
transform_consan(std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
                 const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
                 const ConSanDebugOverrides &debug, const RuntimeCapabilities &capabilities,
                 const BoundRuntimeResources &resources);

/// Run validation-only mutation/perturbation composition through a distinct
/// entry point. The internal native lowerer owns staged-image mechanics; the
/// returned result obeys the same static pipeline contract as an ordinary
/// transform.
[[nodiscard]] TransformResult transform_consan_with_mutation(
    std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const ConSanDebugOverrides &debug, const MutationRequest &mutation,
    const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources);

} // namespace rocjitsu
