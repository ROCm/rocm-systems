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

std::string perturbation_rejection_reason(const SyncEventSemanticIndex &events,
                                          const ConSanSyncSequence &sequence,
                                          ConSanPerturbationKind kind,
                                          ConSanPerturbationEdge edge) {
  if (!sequence.basic_block_index)
    return "missing-exact-basic-block";
  if (sequence.in_cyclic_cfg_component)
    return "cyclic-cfg-component";
  if (sequence.inside_scalar_clause)
    return "inside-s-clause";
  if (is_rocclr_runtime_kernel_name(sequence.container_name))
    return "runtime-helper";
  if (!sequence_has_exact_members(events, sequence))
    return "non-exact-sequence-members";
  if (!consan_sync_confidence_meets(sequence.confidence, ConSanSemanticConfidence::Conservative))
    return "ambiguous-or-unsupported-sequence";

  if (kind == ConSanPerturbationKind::Barrier) {
    if (sequence.kind != ConSanSyncSequenceKind::Barrier ||
        sequence.operation != ConSanSyncOperation::BarrierFull ||
        sequence.member_event_identities.size() != 2u)
      return "not-qualified-full-barrier";
    if (sequence.barrier_operand_source != ConSanBarrierSite::OperandSource::Immediate ||
        !sequence.barrier_id || sequence.barrier_scope == ConSanBarrierSite::Scope::Unknown)
      return "dynamic-or-unknown-barrier-participants";
    return {};
  }

  if (kind == ConSanPerturbationKind::Atomic) {
    if (sequence.kind != ConSanSyncSequenceKind::Atomic)
      return "not-atomic-sequence";
    if (!consan_sync_confidence_meets(sequence.memory_role_confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        !sync_role_has_edge(sequence.memory_role, edge))
      return "unknown-or-inapplicable-memory-role";
    if (!sequence_atomic_has_device_or_system_scope(events, sequence))
      return "unsupported-atomic-scope";
    if (sequence.address_source == ConSanSyncAddressSource::Unknown)
      return "ambiguous-address-provenance";
    return {};
  }
  return "no-perturbation-kind";
}

} // namespace rocjitsu
