// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/supercollider/consan_supercollider_perturbation_policy.h"

#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"

namespace rocjitsu::consan {
namespace {

[[nodiscard]] bool sync_role_has_edge(SyncMemoryRole role, SuperColliderPerturbationEdge edge) {
  if (edge == SuperColliderPerturbationEdge::Release) {
    return role == SyncMemoryRole::Release || role == SyncMemoryRole::AcquireRelease ||
           role == SyncMemoryRole::SequentiallyConsistent;
  }
  return role == SyncMemoryRole::Acquire || role == SyncMemoryRole::AcquireRelease ||
         role == SyncMemoryRole::SequentiallyConsistent;
}

[[nodiscard]] bool
sequence_atomic_has_agent_or_system_scope(const SynchronizationInventoryView &events,
                                          const SyncSequence &sequence) {
  const SyncEvent *atomic = nullptr;
  for (const SyncEventId identity : sequence.member_event_ids) {
    const SyncEvent *event = events.find_event(identity);
    if (event == nullptr || event->kind != SyncKind::Atomic)
      continue;
    if (atomic != nullptr)
      return false;
    atomic = event;
  }
  // Read this from the exact atomic member rather than trusting aggregate
  // sequence metadata inherited while cache/fence members are associated.
  return atomic != nullptr && atomic->scope && memory_scope_is_agent_or_system(*atomic->scope);
}

} // namespace

SuperColliderPerturbationRejectionReason supercollider_perturbation_rejection_reason(
    const SynchronizationInventoryView &events, const SyncSequence &sequence,
    SuperColliderPerturbationKind kind, SuperColliderPerturbationEdge edge) {
  using Reason = SuperColliderPerturbationRejectionReason;
  if (!sequence.basic_block_index)
    return Reason::MissingExactBasicBlock;
  if (sequence.in_cyclic_cfg_component)
    return Reason::CyclicCfgComponent;
  if (sequence.inside_scalar_clause)
    return Reason::InsideScalarClause;
  if (is_rocclr_runtime_kernel_name(events.container_name(sequence)))
    return Reason::RuntimeHelper;
  if (!sequence_has_exact_members(events, sequence))
    return Reason::NonExactSequenceMembers;
  if (!sync_confidence_meets(sequence.confidence, SemanticConfidence::Conservative))
    return Reason::AmbiguousOrUnsupportedSequence;

  if (kind == SuperColliderPerturbationKind::Barrier) {
    if (sequence.kind != SyncKind::Barrier || sequence.operation != SyncOperation::BarrierFull ||
        sequence.member_event_ids.size() != 2u)
      return Reason::NotQualifiedFullBarrier;
    if (sequence.barrier_operand_source != BarrierSite::OperandSource::Immediate ||
        !sequence.barrier_id || sequence.barrier_scope == BarrierSite::Scope::Unknown)
      return Reason::DynamicOrUnknownBarrierParticipants;
    return Reason::None;
  }

  if (kind == SuperColliderPerturbationKind::Atomic) {
    if (sequence.kind != SyncKind::Atomic)
      return Reason::NotAtomicSequence;
    if (!sync_confidence_meets(sequence.memory_role_confidence, SemanticConfidence::Conservative) ||
        !sync_role_has_edge(sequence.memory_role, edge))
      return Reason::UnknownOrInapplicableMemoryRole;
    if (!sequence_atomic_has_agent_or_system_scope(events, sequence))
      return Reason::UnsupportedAtomicScope;
    if (sequence.address_source == SyncAddressSource::Unknown)
      return Reason::AmbiguousAddressProvenance;
    return Reason::None;
  }
  return Reason::NoPerturbationKind;
}

} // namespace rocjitsu::consan
