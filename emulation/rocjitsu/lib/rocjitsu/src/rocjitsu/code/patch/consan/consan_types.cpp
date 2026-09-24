// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"

#include <string>
#include <string_view>

namespace rocjitsu::consan {

namespace {

constexpr auto kModeNames = make_enum_vocabulary("unknown", enum_entry(Mode::None, "none"),
                                                 enum_entry(Mode::SuperCollider, "supercollider"),
                                                 enum_entry(Mode::Default, "default"));

constexpr auto kTransformOutcomes =
    make_enum_vocabulary("unknown", enum_entry(TransformOutcome::Unchanged, "unchanged"),
                         enum_entry(TransformOutcome::ModifiedValid, "modified-valid"),
                         enum_entry(TransformOutcome::Unsupported, "unsupported"),
                         enum_entry(TransformOutcome::Invalid, "invalid"));

constexpr auto kResourceSiteKinds = make_enum_vocabulary(
    "unknown", enum_entry(ResourceSiteKind::Access, "access"),
    enum_entry(ResourceSiteKind::Barrier, "barrier"),
    enum_entry(ResourceSiteKind::Atomic, "atomic"), enum_entry(ResourceSiteKind::Fence, "fence"));

constexpr auto kRegisterAllocationSources = make_enum_vocabulary(
    "unknown", enum_entry(RegisterAllocationSource::Unsupported, "unsupported"),
    enum_entry(RegisterAllocationSource::Explicit, "explicit"),
    enum_entry(RegisterAllocationSource::LivenessDead, "dead"),
    enum_entry(RegisterAllocationSource::DescriptorGrowth, "descriptor-growth"),
    enum_entry(RegisterAllocationSource::SpillRequired, "spill"));

constexpr auto kDelayModes =
    make_enum_vocabulary("unknown", enum_entry(SuperColliderDelayMode::Nop, "nop"),
                         enum_entry(SuperColliderDelayMode::Sleep, "sleep"),
                         enum_entry(SuperColliderDelayMode::SleepVar, "sleep_var"),
                         enum_entry(SuperColliderDelayMode::SleepWave, "sleep_wave"));

constexpr auto kBarrierOperandSources = make_enum_vocabulary(
    "unknown", enum_entry(BarrierSite::OperandSource::Unknown, "unknown"),
    enum_entry(BarrierSite::OperandSource::Immediate, "immediate"),
    enum_entry(BarrierSite::OperandSource::DynamicM0, "dynamic-m0"),
    enum_entry(BarrierSite::OperandSource::StaticM0Literal32, "static-m0-literal32"),
    enum_entry(BarrierSite::OperandSource::Literal32, "literal32"),
    enum_entry(BarrierSite::OperandSource::Literal64, "literal64"));

constexpr auto kBarrierScopes =
    make_enum_vocabulary("unknown", enum_entry(BarrierSite::Scope::Unknown, "unknown"),
                         enum_entry(BarrierSite::Scope::Workgroup, "workgroup"),
                         enum_entry(BarrierSite::Scope::Cluster, "cluster"));

constexpr auto kRegisterPlanReasons = make_enum_vocabulary(
    "unknown", enum_entry(RegisterPlanReason::None, "none"),
    enum_entry(RegisterPlanReason::InvalidRequest, "invalid_request"),
    enum_entry(RegisterPlanReason::ExplicitMisaligned, "explicit_misaligned"),
    enum_entry(RegisterPlanReason::ExplicitOutOfRange, "explicit_out_of_range"),
    enum_entry(RegisterPlanReason::ExplicitLive, "explicit_live"),
    enum_entry(RegisterPlanReason::ForbiddenOverlap, "forbidden_overlap"),
    enum_entry(RegisterPlanReason::MissingInstruction, "missing_instruction"),
    enum_entry(RegisterPlanReason::MissingOwner, "missing_owner"),
    enum_entry(RegisterPlanReason::AmbiguousOwners, "ambiguous_owners"),
    enum_entry(RegisterPlanReason::InvalidDescriptor, "invalid_descriptor"),
    enum_entry(RegisterPlanReason::NoLegalWindow, "no_legal_window"),
    enum_entry(RegisterPlanReason::DynamicStack, "dynamic_stack"));

constexpr auto kResourcePlanAlternativeKinds =
    make_enum_vocabulary("unknown",
                         enum_entry(ResourcePlanAlternativeKind::GuestOperandOverlapSpill,
                                    "guest_operand_overlap_spill"),
                         enum_entry(ResourcePlanAlternativeKind::SpillBackedOperandRecovery,
                                    "spill_backed_operand_recovery"),
                         enum_entry(ResourcePlanAlternativeKind::EmptyAccumulatorDescriptorGrowth,
                                    "empty_accumulator_descriptor_growth"));

constexpr auto kResourcePlanAlternativeOutcomes = make_enum_vocabulary(
    "unknown", enum_entry(ResourcePlanAlternativeOutcome::Selected, "selected"),
    enum_entry(ResourcePlanAlternativeOutcome::Rejected, "rejected"),
    enum_entry(ResourcePlanAlternativeOutcome::Superseded, "superseded"),
    enum_entry(ResourcePlanAlternativeOutcome::Contributed, "contributed"),
    enum_entry(ResourcePlanAlternativeOutcome::Vetoed, "vetoed"));

constexpr auto kBarrierLifecycleIssues = make_enum_vocabulary(
    "invalid barrier-lifecycle issue", enum_entry(BarrierLifecycleIssue::None, ""),
    enum_entry(BarrierLifecycleIssue::InitMissingStaticIdOrScope,
               "lifecycle init has no proven static ID and scope"),
    enum_entry(BarrierLifecycleIssue::NonContiguousRun,
               "lifecycle run crosses a block, container, or instruction gap"),
    enum_entry(BarrierLifecycleIssue::MemberIdOrScopeMismatch,
               "lifecycle members do not have one matching static ID and scope"),
    enum_entry(BarrierLifecycleIssue::MissingJoin,
               "lifecycle leave has no preceding matching static join association"),
    enum_entry(BarrierLifecycleIssue::MissingCompletingBarrier,
               "lifecycle run has no contiguous same-block completing barrier pair"),
    enum_entry(BarrierLifecycleIssue::MissingLeave,
               "lifecycle run has no contiguous same-block leave operation"),
    enum_entry(BarrierLifecycleIssue::InvalidLeaveEncoding,
               "lifecycle leave is not the fixed-zero RDNA4 encoding"));

} // namespace

std::string barrier_move_destination_issue_message(BarrierMoveDestinationIssue issue,
                                                   std::string_view detail) {
  using Issue = BarrierMoveDestinationIssue;
  switch (issue) {
  case Issue::None:
    return {};
  case Issue::InsideScalarClause:
    return "inside-s-clause";
  case Issue::BarrierSourceOrLifecycle:
    return "barrier-source-or-lifecycle";
  case Issue::FenceOperation:
    return "fence-operation";
  case Issue::NotMemoryOperation:
    return "not-memory-operation";
  case Issue::NotRelocatable:
    return "not-relocatable:" + std::string(detail);
  case Issue::Count:
    break;
  }
  return "invalid-barrier-move-destination-issue";
}

std::string_view barrier_lifecycle_issue_message(BarrierLifecycleIssue issue) {
  return kBarrierLifecycleIssues.name(issue);
}

const char *mode_name(Mode mode) { return kModeNames.name(mode).data(); }

const char *transform_outcome_name(TransformOutcome outcome) {
  return kTransformOutcomes.name(outcome).data();
}

const char *resource_site_kind_name(ResourceSiteKind kind) {
  return kResourceSiteKinds.name(kind).data();
}

const char *register_allocation_source_name(RegisterAllocationSource source) {
  return kRegisterAllocationSources.name(source).data();
}

const char *delay_mode_name(SuperColliderDelayMode mode) { return kDelayModes.name(mode).data(); }

const char *barrier_operand_source_name(BarrierSite::OperandSource source) {
  return kBarrierOperandSources.name(source).data();
}

const char *barrier_scope_name(BarrierSite::Scope scope) {
  return kBarrierScopes.name(scope).data();
}

const char *register_plan_reason_name(RegisterPlanReason reason) {
  return kRegisterPlanReasons.name(reason).data();
}

ResourcePlanAlternativeOutcome
resource_plan_alternative_outcome(const CandidateResourcePlan &plan,
                                  const ResourcePlanAlternative &alternative) {
  if (alternative.outcome == ResourcePlanAlternativeOutcome::Selected &&
      plan.source == RegisterAllocationSource::Unsupported)
    return ResourcePlanAlternativeOutcome::Vetoed;
  return alternative.outcome;
}

const char *resource_plan_alternative_kind_name(ResourcePlanAlternativeKind kind) {
  return kResourcePlanAlternativeKinds.name(kind).data();
}

const char *resource_plan_alternative_outcome_name(ResourcePlanAlternativeOutcome outcome) {
  return kResourcePlanAlternativeOutcomes.name(outcome).data();
}

} // namespace rocjitsu::consan
