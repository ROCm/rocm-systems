// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"

#include <cctype>
#include <string>
#include <string_view>

namespace rocjitsu {

namespace {

[[nodiscard]] bool ascii_iequals(std::string_view lhs, std::string_view rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const auto l = static_cast<unsigned char>(lhs[i]);
    const auto r = static_cast<unsigned char>(rhs[i]);
    if (std::tolower(l) != std::tolower(r))
      return false;
  }
  return true;
}

constexpr auto kConSanFlavors =
    make_consan_enum_vocabulary("unknown", consan_enum(ConSanFlavor::None, "none"),
                                consan_enum(ConSanFlavor::SuperCollider, "supercollider"),
                                consan_enum(ConSanFlavor::Moi, "moi"));

constexpr auto kConSanMoiEngines = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanMoiEngine::RecordReplay, "record_replay"),
    consan_enum(ConSanMoiEngine::InlineShadow, "inline_shadow"),
    consan_enum(ConSanMoiEngine::Sampled, "sampled"));

constexpr auto kConSanTransformOutcomes = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanTransformOutcome::Unchanged, "unchanged"),
    consan_enum(ConSanTransformOutcome::ModifiedValid, "modified-valid"),
    consan_enum(ConSanTransformOutcome::Unsupported, "unsupported"),
    consan_enum(ConSanTransformOutcome::Invalid, "invalid"));

constexpr auto kConSanResourceSiteKinds =
    make_consan_enum_vocabulary("unknown", consan_enum(ConSanResourceSiteKind::Access, "access"),
                                consan_enum(ConSanResourceSiteKind::Barrier, "barrier"),
                                consan_enum(ConSanResourceSiteKind::Atomic, "atomic"),
                                consan_enum(ConSanResourceSiteKind::Fence, "fence"));

constexpr auto kConSanRegisterAllocationSources = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanRegisterAllocationSource::Unsupported, "unsupported"),
    consan_enum(ConSanRegisterAllocationSource::Explicit, "explicit"),
    consan_enum(ConSanRegisterAllocationSource::LivenessDead, "dead"),
    consan_enum(ConSanRegisterAllocationSource::DescriptorGrowth, "descriptor-growth"),
    consan_enum(ConSanRegisterAllocationSource::SpillRequired, "spill"));

constexpr auto kConSanDelayModes =
    make_consan_enum_vocabulary("unknown", consan_enum(ConSanDelayMode::Nop, "nop"),
                                consan_enum(ConSanDelayMode::Sleep, "sleep"),
                                consan_enum(ConSanDelayMode::SleepVar, "sleep_var"));

constexpr auto kConSanBarrierOperandSources = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanBarrierSite::OperandSource::Unknown, "unknown"),
    consan_enum(ConSanBarrierSite::OperandSource::Immediate, "immediate"),
    consan_enum(ConSanBarrierSite::OperandSource::DynamicM0, "dynamic-m0"),
    consan_enum(ConSanBarrierSite::OperandSource::StaticM0Literal32, "static-m0-literal32"),
    consan_enum(ConSanBarrierSite::OperandSource::Literal32, "literal32"),
    consan_enum(ConSanBarrierSite::OperandSource::Literal64, "literal64"));

constexpr auto kConSanBarrierScopes = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanBarrierSite::Scope::Unknown, "unknown"),
    consan_enum(ConSanBarrierSite::Scope::Workgroup, "workgroup"),
    consan_enum(ConSanBarrierSite::Scope::Cluster, "cluster"));

constexpr auto kConSanRegisterPlanReasons = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanRegisterPlanReason::None, "none"),
    consan_enum(ConSanRegisterPlanReason::InvalidRequest, "invalid_request"),
    consan_enum(ConSanRegisterPlanReason::ExplicitMisaligned, "explicit_misaligned"),
    consan_enum(ConSanRegisterPlanReason::ExplicitOutOfRange, "explicit_out_of_range"),
    consan_enum(ConSanRegisterPlanReason::ExplicitLive, "explicit_live"),
    consan_enum(ConSanRegisterPlanReason::ForbiddenOverlap, "forbidden_overlap"),
    consan_enum(ConSanRegisterPlanReason::MissingInstruction, "missing_instruction"),
    consan_enum(ConSanRegisterPlanReason::MissingOwner, "missing_owner"),
    consan_enum(ConSanRegisterPlanReason::AmbiguousOwners, "ambiguous_owners"),
    consan_enum(ConSanRegisterPlanReason::InvalidDescriptor, "invalid_descriptor"),
    consan_enum(ConSanRegisterPlanReason::NoLegalWindow, "no_legal_window"),
    consan_enum(ConSanRegisterPlanReason::DynamicStack, "dynamic_stack"));

constexpr auto kConSanResourcePlanAlternativeKinds = make_consan_enum_vocabulary(
    "unknown",
    consan_enum(ConSanResourcePlanAlternativeKind::GuestOperandOverlapSpill,
                "guest_operand_overlap_spill"),
    consan_enum(ConSanResourcePlanAlternativeKind::SpillBackedOperandRecovery,
                "spill_backed_operand_recovery"),
    consan_enum(ConSanResourcePlanAlternativeKind::EmptyAccumulatorDescriptorGrowth,
                "empty_accumulator_descriptor_growth"));

constexpr auto kConSanResourcePlanAlternativeOutcomes = make_consan_enum_vocabulary(
    "unknown", consan_enum(ConSanResourcePlanAlternativeOutcome::Selected, "selected"),
    consan_enum(ConSanResourcePlanAlternativeOutcome::Rejected, "rejected"),
    consan_enum(ConSanResourcePlanAlternativeOutcome::Superseded, "superseded"),
    consan_enum(ConSanResourcePlanAlternativeOutcome::Contributed, "contributed"),
    consan_enum(ConSanResourcePlanAlternativeOutcome::Vetoed, "vetoed"));

constexpr auto kConSanBarrierLifecycleIssues = make_consan_enum_vocabulary(
    "invalid barrier-lifecycle issue", consan_enum(ConSanBarrierLifecycleIssue::None, ""),
    consan_enum(ConSanBarrierLifecycleIssue::InitMissingStaticIdOrScope,
                "lifecycle init has no proven static ID and scope"),
    consan_enum(ConSanBarrierLifecycleIssue::NonContiguousRun,
                "lifecycle run crosses a block, container, or instruction gap"),
    consan_enum(ConSanBarrierLifecycleIssue::MemberIdOrScopeMismatch,
                "lifecycle members do not have one matching static ID and scope"),
    consan_enum(ConSanBarrierLifecycleIssue::MissingJoin,
                "lifecycle leave has no preceding matching static join association"),
    consan_enum(ConSanBarrierLifecycleIssue::MissingCompletingBarrier,
                "lifecycle run has no contiguous same-block completing barrier pair"),
    consan_enum(ConSanBarrierLifecycleIssue::MissingLeave,
                "lifecycle run has no contiguous same-block leave operation"),
    consan_enum(ConSanBarrierLifecycleIssue::InvalidLeaveEncoding,
                "lifecycle leave is not the fixed-zero RDNA4 encoding"));

} // namespace

std::string consan_barrier_move_destination_issue_message(ConSanBarrierMoveDestinationIssue issue,
                                                          std::string_view detail) {
  using Issue = ConSanBarrierMoveDestinationIssue;
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

std::string_view consan_barrier_lifecycle_issue_message(ConSanBarrierLifecycleIssue issue) {
  return kConSanBarrierLifecycleIssues.name(issue);
}

const char *consan_flavor_name(ConSanFlavor flavor) { return kConSanFlavors.name(flavor).data(); }

const char *consan_moi_engine_name(ConSanMoiEngine engine) {
  return kConSanMoiEngines.name(engine).data();
}

const char *consan_transform_outcome_name(ConSanTransformOutcome outcome) {
  return kConSanTransformOutcomes.name(outcome).data();
}

const char *consan_resource_site_kind_name(ConSanResourceSiteKind kind) {
  return kConSanResourceSiteKinds.name(kind).data();
}

const char *consan_register_allocation_source_name(ConSanRegisterAllocationSource source) {
  return kConSanRegisterAllocationSources.name(source).data();
}

const char *consan_delay_mode_name(ConSanDelayMode mode) {
  return kConSanDelayModes.name(mode).data();
}

const char *consan_barrier_operand_source_name(ConSanBarrierSite::OperandSource source) {
  return kConSanBarrierOperandSources.name(source).data();
}

const char *consan_barrier_scope_name(ConSanBarrierSite::Scope scope) {
  return kConSanBarrierScopes.name(scope).data();
}

const char *consan_register_plan_reason_name(ConSanRegisterPlanReason reason) {
  return kConSanRegisterPlanReasons.name(reason).data();
}

ConSanResourcePlanAlternativeOutcome
consan_resource_plan_alternative_outcome(const ConSanCandidateResourcePlan &plan,
                                         const ConSanResourcePlanAlternative &alternative) {
  if (alternative.outcome == ConSanResourcePlanAlternativeOutcome::Selected &&
      plan.source == ConSanRegisterAllocationSource::Unsupported)
    return ConSanResourcePlanAlternativeOutcome::Vetoed;
  return alternative.outcome;
}

const char *consan_resource_plan_alternative_kind_name(ConSanResourcePlanAlternativeKind kind) {
  return kConSanResourcePlanAlternativeKinds.name(kind).data();
}

const char *
consan_resource_plan_alternative_outcome_name(ConSanResourcePlanAlternativeOutcome outcome) {
  return kConSanResourcePlanAlternativeOutcomes.name(outcome).data();
}

std::optional<ConSanFlavor> parse_consan_flavor(std::string_view value) {
  for (ConSanFlavor flavor : kConSanFlavors)
    if (flavor != ConSanFlavor::None && ascii_iequals(value, kConSanFlavors.name(flavor)))
      return flavor;
  return std::nullopt;
}

std::optional<ConSanMoiEngine> parse_consan_moi_engine(std::string_view value) {
  for (ConSanMoiEngine engine : kConSanMoiEngines)
    if (ascii_iequals(value, kConSanMoiEngines.name(engine)))
      return engine;
  if (ascii_iequals(value, "record-replay") || ascii_iequals(value, "context"))
    return ConSanMoiEngine::RecordReplay;
  if (ascii_iequals(value, "inline-shadow"))
    return ConSanMoiEngine::InlineShadow;
  if (ascii_iequals(value, "sampled_watchpoint") || ascii_iequals(value, "sampled-watchpoint"))
    return ConSanMoiEngine::Sampled;
  return std::nullopt;
}

} // namespace rocjitsu
