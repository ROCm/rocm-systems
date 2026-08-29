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
  using Issue = ConSanBarrierLifecycleIssue;
  switch (issue) {
  case Issue::None:
    return {};
  case Issue::InitMissingStaticIdOrScope:
    return "lifecycle init has no proven static ID and scope";
  case Issue::NonContiguousRun:
    return "lifecycle run crosses a block, container, or instruction gap";
  case Issue::MemberIdOrScopeMismatch:
    return "lifecycle members do not have one matching static ID and scope";
  case Issue::MissingJoin:
    return "lifecycle leave has no preceding matching static join association";
  case Issue::MissingCompletingBarrier:
    return "lifecycle run has no contiguous same-block completing barrier pair";
  case Issue::MissingLeave:
    return "lifecycle run has no contiguous same-block leave operation";
  case Issue::InvalidLeaveEncoding:
    return "lifecycle leave is not the fixed-zero GFX12 encoding";
  case Issue::Count:
    break;
  }
  return "invalid barrier-lifecycle issue";
}

bool ConSanFaultMutationPlan::well_formed() const {
  if (!source_code_object.valid() || primary_identity.empty())
    return false;
  const auto nonempty = [](const std::optional<std::string> &value) {
    return !value || !value->empty();
  };
  if (!nonempty(companion_identity) || !nonempty(logical_sequence_identity) ||
      !nonempty(selection_sequence_identity) || !nonempty(companion_selection_sequence_identity) ||
      !nonempty(destination_identity))
    return false;
  for (const std::string &identity : ordered_member_identities) {
    if (identity.empty())
      return false;
  }
  const bool no_move = !destination_identity &&
                       barrier_move_direction == ConSanBarrierMoveDirection::LegacyMarker &&
                       barrier_move_cfg_contract == ConSanBarrierMoveCfgContract::SameBlock &&
                       !structured_guard_block_index && !structured_destination_block_index &&
                       !structured_source_block_index && !structured_guard_offset &&
                       !structured_destination_offset && !structured_source_offset;
  const bool no_barrier_retarget = !original_barrier_id && !target_barrier_id &&
                                   original_barrier_scope == ConSanBarrierSite::Scope::Unknown &&
                                   target_barrier_scope == ConSanBarrierSite::Scope::Unknown &&
                                   !original_participant_count && !target_participant_count;
  const bool no_address = !address_delta_bytes && !original_address_vgpr && !target_address_vgpr;
  const bool sequence_present = logical_sequence_identity && !ordered_member_identities.empty();
  const bool valid_delta = address_delta_bytes && *address_delta_bytes != 0u &&
                           *address_delta_bytes % sizeof(uint32_t) == 0u &&
                           *address_delta_bytes <= 0x7fffffu;
  const auto valid_barrier_scope = [](ConSanBarrierSite::Scope scope) {
    return scope == ConSanBarrierSite::Scope::Workgroup ||
           scope == ConSanBarrierSite::Scope::Cluster;
  };
  const auto valid_atomic_edge = [](const std::optional<ConSanAtomicOrderEdge> &edge) {
    if (!edge)
      return false;
    switch (*edge) {
    case ConSanAtomicOrderEdge::Any:
    case ConSanAtomicOrderEdge::Release:
    case ConSanAtomicOrderEdge::Acquire:
      return true;
    }
    return false;
  };
  switch (kind) {
  case ConSanFaultMutationKind::DropBarrier:
    return no_move && no_barrier_retarget && no_address && !atomic_order_edge &&
           ((!companion_identity && !logical_sequence_identity && !selection_sequence_identity &&
             !companion_selection_sequence_identity && ordered_member_identities.empty()) ||
            (companion_identity && sequence_present && selection_sequence_identity &&
             (!companion_selection_sequence_identity ||
              *companion_selection_sequence_identity != *selection_sequence_identity))) &&
           (!destructive_incomplete_barrier_drop || !companion_identity);
  case ConSanFaultMutationKind::MoveBarrierPair: {
    if (!companion_identity || !sequence_present || !selection_sequence_identity ||
        companion_selection_sequence_identity || no_barrier_retarget == false || !no_address ||
        atomic_order_edge || destructive_incomplete_barrier_drop)
      return false;
    if (barrier_move_direction == ConSanBarrierMoveDirection::LegacyMarker)
      return !destination_identity &&
             barrier_move_cfg_contract == ConSanBarrierMoveCfgContract::SameBlock &&
             !structured_guard_block_index && !structured_destination_block_index &&
             !structured_source_block_index && !structured_guard_offset &&
             !structured_destination_offset && !structured_source_offset;
    if (!destination_identity || (barrier_move_direction != ConSanBarrierMoveDirection::Earlier &&
                                  barrier_move_direction != ConSanBarrierMoveDirection::Later))
      return false;
    const bool has_structured_coordinates =
        structured_guard_block_index && structured_destination_block_index &&
        structured_source_block_index && structured_guard_offset && structured_destination_offset &&
        structured_source_offset;
    switch (barrier_move_cfg_contract) {
    case ConSanBarrierMoveCfgContract::SameBlock:
      return !structured_guard_block_index && !structured_destination_block_index &&
             !structured_source_block_index && !structured_guard_offset &&
             !structured_destination_offset && !structured_source_offset;
    case ConSanBarrierMoveCfgContract::CompletingStructuredDiamond:
    case ConSanBarrierMoveCfgContract::DestructiveStructuredExecDiamond:
      return has_structured_coordinates;
    }
    return false;
  }
  case ConSanFaultMutationKind::BarrierIdScope:
    return no_move && companion_identity && sequence_present && selection_sequence_identity &&
           !companion_selection_sequence_identity && original_barrier_id && target_barrier_id &&
           *original_barrier_id != *target_barrier_id &&
           valid_barrier_scope(original_barrier_scope) &&
           valid_barrier_scope(target_barrier_scope) && !original_participant_count &&
           !target_participant_count && no_address && !atomic_order_edge &&
           !destructive_incomplete_barrier_drop;
  case ConSanFaultMutationKind::BarrierParticipantCount:
    return no_move && !companion_identity && sequence_present && selection_sequence_identity &&
           !companion_selection_sequence_identity && original_barrier_id && target_barrier_id &&
           *original_barrier_id == *target_barrier_id &&
           valid_barrier_scope(original_barrier_scope) &&
           original_barrier_scope == target_barrier_scope && original_participant_count &&
           target_participant_count && *original_participant_count != 0u &&
           *target_participant_count != 0u && *target_participant_count <= 63u &&
           *original_participant_count != *target_participant_count && no_address &&
           !atomic_order_edge && !destructive_incomplete_barrier_drop;
  case ConSanFaultMutationKind::AtomicWrongAddress:
    return no_move && no_barrier_retarget && !companion_identity && !logical_sequence_identity &&
           !selection_sequence_identity && !companion_selection_sequence_identity &&
           ordered_member_identities.empty() && valid_delta && !original_address_vgpr &&
           !target_address_vgpr && !atomic_order_edge && !destructive_incomplete_barrier_drop;
  case ConSanFaultMutationKind::AtomicWeakenOrder:
    return no_move && no_barrier_retarget && companion_identity && sequence_present &&
           selection_sequence_identity && !companion_selection_sequence_identity && no_address &&
           valid_atomic_edge(atomic_order_edge) && !destructive_incomplete_barrier_drop;
  case ConSanFaultMutationKind::AtomicWeakenScope:
    return no_move && no_barrier_retarget && !companion_identity && !logical_sequence_identity &&
           !selection_sequence_identity && !companion_selection_sequence_identity &&
           ordered_member_identities.empty() && no_address && !atomic_order_edge &&
           !destructive_incomplete_barrier_drop;
  case ConSanFaultMutationKind::LdsWrongAddress:
    return no_move && no_barrier_retarget && !companion_identity && !logical_sequence_identity &&
           !selection_sequence_identity && !companion_selection_sequence_identity &&
           ordered_member_identities.empty() && !address_delta_bytes && original_address_vgpr &&
           target_address_vgpr && *original_address_vgpr != *target_address_vgpr &&
           !atomic_order_edge && !destructive_incomplete_barrier_drop;
  case ConSanFaultMutationKind::OrdinaryWrongAddress:
    return no_move && no_barrier_retarget && !companion_identity && sequence_present &&
           selection_sequence_identity && !companion_selection_sequence_identity && valid_delta &&
           !original_address_vgpr && !target_address_vgpr && !atomic_order_edge &&
           !destructive_incomplete_barrier_drop;
  case ConSanFaultMutationKind::OrdinaryWeakenOrder:
    return no_move && no_barrier_retarget && companion_identity && sequence_present &&
           selection_sequence_identity && !companion_selection_sequence_identity && no_address &&
           !atomic_order_edge && !destructive_incomplete_barrier_drop;
  case ConSanFaultMutationKind::OrdinaryWeakenScope:
    return no_move && no_barrier_retarget && !companion_identity && sequence_present &&
           selection_sequence_identity && !companion_selection_sequence_identity && no_address &&
           !atomic_order_edge && !destructive_incomplete_barrier_drop;
  }
  return false;
}

const char *consan_flavor_name(ConSanFlavor flavor) {
  switch (flavor) {
  case ConSanFlavor::None:
    return "none";
  case ConSanFlavor::SuperCollider:
    return "supercollider";
  case ConSanFlavor::Moi:
    return "moi";
  }
  return "unknown";
}

const char *consan_moi_engine_name(ConSanMoiEngine engine) {
  switch (engine) {
  case ConSanMoiEngine::RecordReplay:
    return "record_replay";
  case ConSanMoiEngine::InlineShadow:
    return "inline_shadow";
  case ConSanMoiEngine::Sampled:
    return "sampled";
  }
  return "unknown";
}

const char *consan_transform_outcome_name(ConSanTransformOutcome outcome) {
  switch (outcome) {
  case ConSanTransformOutcome::Unchanged:
    return "unchanged";
  case ConSanTransformOutcome::ModifiedValid:
    return "modified-valid";
  case ConSanTransformOutcome::Unsupported:
    return "unsupported";
  case ConSanTransformOutcome::Invalid:
    return "invalid";
  }
  return "unknown";
}

const char *consan_resource_site_kind_name(ConSanResourceSiteKind kind) {
  switch (kind) {
  case ConSanResourceSiteKind::Access:
    return "access";
  case ConSanResourceSiteKind::Barrier:
    return "barrier";
  case ConSanResourceSiteKind::Atomic:
    return "atomic";
  case ConSanResourceSiteKind::Fence:
    return "fence";
  }
  return "unknown";
}

const char *consan_register_allocation_source_name(ConSanRegisterAllocationSource source) {
  switch (source) {
  case ConSanRegisterAllocationSource::Unsupported:
    return "unsupported";
  case ConSanRegisterAllocationSource::Explicit:
    return "explicit";
  case ConSanRegisterAllocationSource::LivenessDead:
    return "dead";
  case ConSanRegisterAllocationSource::DescriptorGrowth:
    return "descriptor-growth";
  case ConSanRegisterAllocationSource::SpillRequired:
    return "spill";
  }
  return "unknown";
}

const char *consan_delay_mode_name(ConSanDelayMode mode) {
  switch (mode) {
  case ConSanDelayMode::Nop:
    return "nop";
  case ConSanDelayMode::Sleep:
    return "sleep";
  case ConSanDelayMode::SleepVar:
    return "sleep_var";
  }
  return "unknown";
}

const char *consan_barrier_operand_source_name(ConSanBarrierSite::OperandSource source) {
  switch (source) {
  case ConSanBarrierSite::OperandSource::Unknown:
    return "unknown";
  case ConSanBarrierSite::OperandSource::Immediate:
    return "immediate";
  case ConSanBarrierSite::OperandSource::DynamicM0:
    return "dynamic-m0";
  case ConSanBarrierSite::OperandSource::StaticM0Literal32:
    return "static-m0-literal32";
  case ConSanBarrierSite::OperandSource::Literal32:
    return "literal32";
  case ConSanBarrierSite::OperandSource::Literal64:
    return "literal64";
  }
  return "unknown";
}

const char *consan_barrier_scope_name(ConSanBarrierSite::Scope scope) {
  switch (scope) {
  case ConSanBarrierSite::Scope::Unknown:
    return "unknown";
  case ConSanBarrierSite::Scope::Workgroup:
    return "workgroup";
  case ConSanBarrierSite::Scope::Cluster:
    return "cluster";
  }
  return "unknown";
}

const char *consan_register_plan_reason_name(ConSanRegisterPlanReason reason) {
  switch (reason) {
  case ConSanRegisterPlanReason::None:
    return "none";
  case ConSanRegisterPlanReason::InvalidRequest:
    return "invalid_request";
  case ConSanRegisterPlanReason::ExplicitMisaligned:
    return "explicit_misaligned";
  case ConSanRegisterPlanReason::ExplicitOutOfRange:
    return "explicit_out_of_range";
  case ConSanRegisterPlanReason::ExplicitLive:
    return "explicit_live";
  case ConSanRegisterPlanReason::ForbiddenOverlap:
    return "forbidden_overlap";
  case ConSanRegisterPlanReason::MissingInstruction:
    return "missing_instruction";
  case ConSanRegisterPlanReason::MissingOwner:
    return "missing_owner";
  case ConSanRegisterPlanReason::AmbiguousOwners:
    return "ambiguous_owners";
  case ConSanRegisterPlanReason::InvalidDescriptor:
    return "invalid_descriptor";
  case ConSanRegisterPlanReason::NoLegalWindow:
    return "no_legal_window";
  case ConSanRegisterPlanReason::DynamicStack:
    return "dynamic_stack";
  }
  return "unknown";
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
  switch (kind) {
  case ConSanResourcePlanAlternativeKind::GuestOperandOverlapSpill:
    return "guest_operand_overlap_spill";
  case ConSanResourcePlanAlternativeKind::SpillBackedOperandRecovery:
    return "spill_backed_operand_recovery";
  case ConSanResourcePlanAlternativeKind::EmptyAccumulatorDescriptorGrowth:
    return "empty_accumulator_descriptor_growth";
  }
  return "unknown";
}

const char *
consan_resource_plan_alternative_outcome_name(ConSanResourcePlanAlternativeOutcome outcome) {
  switch (outcome) {
  case ConSanResourcePlanAlternativeOutcome::Selected:
    return "selected";
  case ConSanResourcePlanAlternativeOutcome::Rejected:
    return "rejected";
  case ConSanResourcePlanAlternativeOutcome::Superseded:
    return "superseded";
  case ConSanResourcePlanAlternativeOutcome::Contributed:
    return "contributed";
  case ConSanResourcePlanAlternativeOutcome::Vetoed:
    return "vetoed";
  }
  return "unknown";
}

std::optional<ConSanFlavor> parse_consan_flavor(std::string_view value) {
  if (ascii_iequals(value, "supercollider"))
    return ConSanFlavor::SuperCollider;
  if (ascii_iequals(value, "moi"))
    return ConSanFlavor::Moi;
  return std::nullopt;
}

std::optional<ConSanMoiEngine> parse_consan_moi_engine(std::string_view value) {
  if (ascii_iequals(value, "record_replay") || ascii_iequals(value, "record-replay") ||
      ascii_iequals(value, "context"))
    return ConSanMoiEngine::RecordReplay;
  if (ascii_iequals(value, "inline_shadow") || ascii_iequals(value, "inline-shadow"))
    return ConSanMoiEngine::InlineShadow;
  if (ascii_iequals(value, "sampled_watchpoint") || ascii_iequals(value, "sampled-watchpoint") ||
      ascii_iequals(value, "sampled"))
    return ConSanMoiEngine::Sampled;
  return std::nullopt;
}

} // namespace rocjitsu
