// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_perturbation_policy.h"

#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"

namespace rocjitsu {
namespace {

[[nodiscard]] bool sync_role_has_edge(ConSanSyncMemoryRole role, ConSanPerturbationEdge edge) {
  if (edge == ConSanPerturbationEdge::Release) {
    return role == ConSanSyncMemoryRole::Release || role == ConSanSyncMemoryRole::AcquireRelease ||
           role == ConSanSyncMemoryRole::SequentiallyConsistent;
  }
  return role == ConSanSyncMemoryRole::Acquire || role == ConSanSyncMemoryRole::AcquireRelease ||
         role == ConSanSyncMemoryRole::SequentiallyConsistent;
}

[[nodiscard]] bool sequence_atomic_has_device_or_system_scope(const SyncEventSemanticIndex &events,
                                                              const ConSanSyncSequence &sequence) {
  const ConSanSyncEvent *atomic = nullptr;
  for (const SemanticSiteId &identity : sequence.member_semantic_ids) {
    const ConSanSyncEvent *event = find_sequence_member_event(events, identity);
    if (event == nullptr || event->kind != ConSanSyncEventKind::Atomic)
      continue;
    if (atomic != nullptr)
      return false;
    atomic = event;
  }
  // RDNA4 encodes device and system scope as 2 and 3 respectively. Read this
  // from the exact atomic member rather than trusting aggregate sequence
  // metadata inherited while cache/fence members are associated.
  return atomic != nullptr && atomic->raw_scope &&
         (*atomic->raw_scope == 2u || *atomic->raw_scope == 3u);
}

} // namespace

ConSanPerturbationRejectionReason
perturbation_rejection_reason(const SyncEventSemanticIndex &events,
                              const ConSanSyncSequence &sequence, ConSanPerturbationKind kind,
                              ConSanPerturbationEdge edge) {
  using Reason = ConSanPerturbationRejectionReason;
  if (!sequence.basic_block_index)
    return Reason::MissingExactBasicBlock;
  if (sequence.in_cyclic_cfg_component)
    return Reason::CyclicCfgComponent;
  if (sequence.inside_scalar_clause)
    return Reason::InsideScalarClause;
  if (is_rocclr_runtime_kernel_name(sequence.container_name))
    return Reason::RuntimeHelper;
  if (!sequence_has_exact_members(events, sequence))
    return Reason::NonExactSequenceMembers;
  if (!consan_sync_confidence_meets(sequence.confidence, ConSanSemanticConfidence::Conservative))
    return Reason::AmbiguousOrUnsupportedSequence;

  if (kind == ConSanPerturbationKind::Barrier) {
    if (sequence.kind != ConSanSyncSequenceKind::Barrier ||
        sequence.operation != ConSanSyncOperation::BarrierFull ||
        sequence.member_event_identities.size() != 2u)
      return Reason::NotQualifiedFullBarrier;
    if (sequence.barrier_operand_source != ConSanBarrierSite::OperandSource::Immediate ||
        !sequence.barrier_id || sequence.barrier_scope == ConSanBarrierSite::Scope::Unknown)
      return Reason::DynamicOrUnknownBarrierParticipants;
    return Reason::None;
  }

  if (kind == ConSanPerturbationKind::Atomic) {
    if (sequence.kind != ConSanSyncSequenceKind::Atomic)
      return Reason::NotAtomicSequence;
    if (!consan_sync_confidence_meets(sequence.memory_role_confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        !sync_role_has_edge(sequence.memory_role, edge))
      return Reason::UnknownOrInapplicableMemoryRole;
    if (!sequence_atomic_has_device_or_system_scope(events, sequence))
      return Reason::UnsupportedAtomicScope;
    if (sequence.address_source == ConSanSyncAddressSource::Unknown)
      return Reason::AmbiguousAddressProvenance;
    return Reason::None;
  }
  return Reason::NoPerturbationKind;
}

std::string_view
consan_perturbation_rejection_reason_name(ConSanPerturbationRejectionReason reason) {
  using Reason = ConSanPerturbationRejectionReason;
  switch (reason) {
  case Reason::None:
    return "";
  case Reason::MissingExactBasicBlock:
    return "missing-exact-basic-block";
  case Reason::CyclicCfgComponent:
    return "cyclic-cfg-component";
  case Reason::InsideScalarClause:
    return "inside-s-clause";
  case Reason::RuntimeHelper:
    return "runtime-helper";
  case Reason::NonExactSequenceMembers:
    return "non-exact-sequence-members";
  case Reason::AmbiguousOrUnsupportedSequence:
    return "ambiguous-or-unsupported-sequence";
  case Reason::NotQualifiedFullBarrier:
    return "not-qualified-full-barrier";
  case Reason::DynamicOrUnknownBarrierParticipants:
    return "dynamic-or-unknown-barrier-participants";
  case Reason::NotAtomicSequence:
    return "not-atomic-sequence";
  case Reason::UnknownOrInapplicableMemoryRole:
    return "unknown-or-inapplicable-memory-role";
  case Reason::UnsupportedAtomicScope:
    return "unsupported-atomic-scope";
  case Reason::AmbiguousAddressProvenance:
    return "ambiguous-address-provenance";
  case Reason::NoPerturbationKind:
    return "no-perturbation-kind";
  case Reason::MissingAnchorEvent:
    return "missing-anchor-event";
  case Reason::Count:
    return "invalid-perturbation-rejection-reason";
  }
  return "invalid-perturbation-rejection-reason";
}

} // namespace rocjitsu
