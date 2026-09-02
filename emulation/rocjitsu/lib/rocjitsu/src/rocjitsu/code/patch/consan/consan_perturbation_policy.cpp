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

[[nodiscard]] bool sequence_atomic_has_agent_or_system_scope(const SyncEventSemanticIndex &events,
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
  // Read this from the exact atomic member rather than trusting aggregate
  // sequence metadata inherited while cache/fence members are associated.
  return atomic != nullptr && atomic->scope &&
         consan_memory_scope_is_agent_or_system(*atomic->scope);
}

constexpr auto kConSanPerturbationRejectionReasons = make_consan_enum_vocabulary(
    "invalid-perturbation-rejection-reason",
    consan_enum(ConSanPerturbationRejectionReason::None, ""),
    consan_enum(ConSanPerturbationRejectionReason::MissingExactBasicBlock,
                "missing-exact-basic-block"),
    consan_enum(ConSanPerturbationRejectionReason::CyclicCfgComponent, "cyclic-cfg-component"),
    consan_enum(ConSanPerturbationRejectionReason::InsideScalarClause, "inside-s-clause"),
    consan_enum(ConSanPerturbationRejectionReason::RuntimeHelper, "runtime-helper"),
    consan_enum(ConSanPerturbationRejectionReason::NonExactSequenceMembers,
                "non-exact-sequence-members"),
    consan_enum(ConSanPerturbationRejectionReason::AmbiguousOrUnsupportedSequence,
                "ambiguous-or-unsupported-sequence"),
    consan_enum(ConSanPerturbationRejectionReason::NotQualifiedFullBarrier,
                "not-qualified-full-barrier"),
    consan_enum(ConSanPerturbationRejectionReason::DynamicOrUnknownBarrierParticipants,
                "dynamic-or-unknown-barrier-participants"),
    consan_enum(ConSanPerturbationRejectionReason::NotAtomicSequence, "not-atomic-sequence"),
    consan_enum(ConSanPerturbationRejectionReason::UnknownOrInapplicableMemoryRole,
                "unknown-or-inapplicable-memory-role"),
    consan_enum(ConSanPerturbationRejectionReason::UnsupportedAtomicScope,
                "unsupported-atomic-scope"),
    consan_enum(ConSanPerturbationRejectionReason::AmbiguousAddressProvenance,
                "ambiguous-address-provenance"),
    consan_enum(ConSanPerturbationRejectionReason::NoPerturbationKind, "no-perturbation-kind"),
    consan_enum(ConSanPerturbationRejectionReason::MissingAnchorEvent, "missing-anchor-event"));

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
    if (!sequence_atomic_has_agent_or_system_scope(events, sequence))
      return Reason::UnsupportedAtomicScope;
    if (sequence.address_source == ConSanSyncAddressSource::Unknown)
      return Reason::AmbiguousAddressProvenance;
    return Reason::None;
  }
  return Reason::NoPerturbationKind;
}

std::string_view
consan_perturbation_rejection_reason_name(ConSanPerturbationRejectionReason reason) {
  return kConSanPerturbationRejectionReasons.name(reason);
}

} // namespace rocjitsu
