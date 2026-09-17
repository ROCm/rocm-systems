// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"

#include <algorithm>
#include <map>
#include <ranges>
#include <tuple>

namespace rocjitsu::consan {
namespace {

[[nodiscard]] bool event_semantics_equal(const SynchronizationInventoryView &inventory,
                                         const SyncEvent &lhs, const SyncEvent &rhs) {
  const ProgramSite *lhs_source = inventory.source(lhs);
  const ProgramSite *rhs_source = inventory.source(rhs);
  if (lhs_source == nullptr || rhs_source == nullptr || !lhs_source->same_payload(*rhs_source))
    return false;
  return std::tie(lhs.kind, lhs.operation, lhs.address_source, lhs.memory_role, lhs.rmw_outcome,
                  lhs.confidence, lhs.memory_role_confidence, lhs.scope) ==
         std::tie(rhs.kind, rhs.operation, rhs.address_source, rhs.memory_role, rhs.rmw_outcome,
                  rhs.confidence, rhs.memory_role_confidence, rhs.scope);
}

[[nodiscard]] std::vector<uint64_t> sequence_offsets(const SynchronizationInventoryView &inventory,
                                                     const SyncSequence &sequence) {
  std::vector<uint64_t> offsets;
  offsets.reserve(sequence.member_event_ids.size());
  for (const SyncEventId member : sequence.member_event_ids) {
    if (const SyncEvent *event = inventory.find_event(member))
      offsets.push_back(event->text_offset());
  }
  return offsets;
}

[[nodiscard]] bool sequence_semantics_equal(const SynchronizationInventoryView &inventory,
                                            const SyncSequence &lhs, const SyncSequence &rhs) {
  return sequence_offsets(inventory, lhs) == sequence_offsets(inventory, rhs) &&
         std::tie(lhs.kind, lhs.operation, lhs.address_source, lhs.memory_role, lhs.rmw_outcome,
                  lhs.confidence, lhs.memory_role_confidence, lhs.begin_text_offset,
                  lhs.end_text_offset, lhs.basic_block_index, lhs.in_cyclic_cfg_component,
                  lhs.inside_scalar_clause, lhs.scope, lhs.barrier_id, lhs.barrier_operand_source,
                  lhs.barrier_scope, lhs.participant_count, lhs.participant_mask) ==
             std::tie(rhs.kind, rhs.operation, rhs.address_source, rhs.memory_role, rhs.rmw_outcome,
                      rhs.confidence, rhs.memory_role_confidence, rhs.begin_text_offset,
                      rhs.end_text_offset, rhs.basic_block_index, rhs.in_cyclic_cfg_component,
                      rhs.inside_scalar_clause, rhs.scope, rhs.barrier_id,
                      rhs.barrier_operand_source, rhs.barrier_scope, rhs.participant_count,
                      rhs.participant_mask);
}

[[nodiscard]] CapabilityForm barrier_form(const BarrierSite &barrier) {
  return barrier.scope == BarrierSite::Scope::Cluster ? CapabilityForm::ClusterBarrier
                                                      : CapabilityForm::WorkgroupBarrier;
}

[[nodiscard]] const SyncEvent *completion_event(const SyncSequence &sequence,
                                                const SynchronizationInventoryView &inventory) {
  const SyncEvent *result = nullptr;
  for (const SyncEventId member : sequence.member_event_ids) {
    const SyncEvent *event = inventory.find_event(member);
    if (event == nullptr)
      continue;
    const ProgramSite *source = inventory.source(*event);
    if (source == nullptr)
      continue;
    if (event->text_offset() <= sequence.end_text_offset &&
        source->size() == sequence.end_text_offset - event->text_offset()) {
      if (result != nullptr && result->semantic_id.physical != event->semantic_id.physical)
        return nullptr;
      result = event;
    }
  }
  return result;
}

[[nodiscard]] std::vector<SemanticSiteId>
covered_event_ids(const SynchronizationInventoryView &inventory, const SyncSequence &sequence) {
  std::vector<SemanticSiteId> ids;
  ids.reserve(sequence.member_event_ids.size());
  for (const SyncEventId member : sequence.member_event_ids) {
    if (const SyncEvent *event = inventory.find_event(member))
      ids.push_back(event->semantic_id);
  }
  return ids;
}

} // namespace

BarrierPolicyResult plan_barrier_observation(const ProgramInventory &inventory,
                                             const BarrierPolicyRequest &request) {
  BarrierPolicyResult result;
  result.plan.mode = request.mode;
  const ModeProbeVocabulary *vocabulary = mode_probe_vocabulary(request.mode);
  if (vocabulary == nullptr || inventory.empty())
    return result;

  const SynchronizationInventoryView synchronization = inventory.sync();
  std::map<uint64_t, std::vector<const SyncEvent *>> events_by_offset;
  for (const SyncEvent &event : synchronization.sync_events) {
    if (event.kind == SyncKind::Barrier)
      events_by_offset[event.semantic_id.physical.original_text_offset].push_back(&event);
  }

  std::map<uint64_t, std::vector<const SyncSequence *>> sequences_by_member_offset;
  for (const SyncSequence &sequence : synchronization.sync_sequences) {
    if (sequence.kind != SyncKind::Barrier)
      continue;
    for (const SyncEventId member : sequence.member_event_ids) {
      const SyncEvent *member_event = synchronization.find_event(member);
      if (member_event == nullptr)
        continue;
      auto &sequences = sequences_by_member_offset[member_event->text_offset()];
      if (std::ranges::none_of(sequences, [&](const auto *known) {
            return sequence_semantics_equal(synchronization, *known, sequence);
          })) {
        sequences.push_back(&sequence);
      }
    }
  }

  std::map<std::vector<uint64_t>, ProbeIntentId> intent_by_sequence;
  std::map<std::string, std::pair<uint64_t, uint32_t>> previous_full_by_container;
  for (const auto &[offset, aliases] : events_by_offset) {
    const SyncEvent &event = *aliases.front();
    const BarrierSite *barrier = synchronization.source_as<BarrierSite>(event);
    const std::vector<std::string> names =
        synchronization.source_container_names(event.semantic_id.physical);
    SiteDecisionKind decision_kind = SiteDecisionKind::NotApplicable;
    BarrierPolicyReason reason = BarrierPolicyReason::TrackingDisabled;
    std::optional<ProbeIntentId> intent_id;

    const bool conflicting_alias = std::ranges::any_of(aliases, [&](const auto *alias) {
      return !event_semantics_equal(synchronization, event, *alias);
    });
    const bool filtered = !request.container_filter.empty() &&
                          std::ranges::none_of(names, [&](const std::string &name) {
                            return name.find(request.container_filter) != std::string::npos;
                          });
    const std::vector<uint64_t> owner_descriptors =
        synchronization.execution_owner_descriptors(aliases);
    const bool allowlist_filtered = !site_matches_kernel_allowlist(
        inventory, owner_descriptors, names, request.kernel_name_allowlist);
    const bool only_runtime_kernels = std::ranges::all_of(aliases, [&](const auto *alias) {
      return synchronization.in_kernel(*alias) &&
             is_rocclr_runtime_kernel_name(synchronization.container_name(*alias));
    });
    bool redundant = false;
    if (event.operation == SyncOperation::BarrierFull) {
      redundant = std::ranges::any_of(names, [&](const std::string &name) {
        const auto previous = previous_full_by_container.find(name);
        return previous != previous_full_by_container.end() &&
               previous->second.first + previous->second.second == offset;
      });
      for (const std::string &name : names)
        previous_full_by_container[name] = {offset, barrier == nullptr ? 0u : barrier->size};
    } else {
      for (const std::string &name : names)
        previous_full_by_container.erase(name);
    }

    const CapabilityDisposition capability = capability_disposition(
        inventory.target(), request.mode,
        barrier == nullptr ? CapabilityForm::WorkgroupBarrier : barrier_form(*barrier));
    const auto member_sequences = sequences_by_member_offset.find(offset);
    std::vector<const SyncSequence *> usable_sequences;
    if (member_sequences != sequences_by_member_offset.end()) {
      for (const SyncSequence *sequence : member_sequences->second) {
        if (request.mode != Mode::Default ||
            qualifies_barrier_sequence(*sequence,
                                       !synchronization.execution_owners(*sequence).empty())) {
          usable_sequences.push_back(sequence);
        }
      }
    }

    const SyncSequence *logical_sequence = nullptr;
    if (!usable_sequences.empty()) {
      logical_sequence = usable_sequences.front();
      if (std::ranges::any_of(usable_sequences, [&](const auto *candidate) {
            return !sequence_semantics_equal(synchronization, *logical_sequence, *candidate);
          })) {
        reason = BarrierPolicyReason::AmbiguousSequenceMembership;
        decision_kind = SiteDecisionKind::Unsupported;
        logical_sequence = nullptr;
      }
    }

    if (conflicting_alias) {
      decision_kind = SiteDecisionKind::Unsupported;
      reason = BarrierPolicyReason::ConflictingPhysicalAliases;
      result.errors.push_back(reason);
    } else if (!request.tracking_enabled) {
      reason = BarrierPolicyReason::TrackingDisabled;
    } else if (request.mode == Mode::SuperCollider ||
               capability == CapabilityDisposition::MutationOnly) {
      reason = BarrierPolicyReason::ModeMutationOnly;
    } else if (filtered || allowlist_filtered) {
      reason = BarrierPolicyReason::ContainerFilterExcluded;
    } else if (only_runtime_kernels) {
      reason = BarrierPolicyReason::RuntimeKernelExcluded;
    } else if (capability != CapabilityDisposition::Supported) {
      reason = BarrierPolicyReason::TargetCapabilityUnavailable;
    } else if (redundant) {
      reason = BarrierPolicyReason::RedundantAdjacentFullBarrier;
    } else if (barrier == nullptr) {
      decision_kind = SiteDecisionKind::Unsupported;
      reason = BarrierPolicyReason::InvalidBarrierEncoding;
    } else if (request.mode != Mode::Default && barrier->size != sizeof(uint32_t)) {
      decision_kind = SiteDecisionKind::Unsupported;
      reason = BarrierPolicyReason::InvalidBarrierEncoding;
    } else if (reason == BarrierPolicyReason::AmbiguousSequenceMembership) {
      // The typed ambiguity selected above is already the final answer.
    } else if (request.mode == Mode::Default && logical_sequence == nullptr) {
      decision_kind = SiteDecisionKind::Unsupported;
      reason = BarrierPolicyReason::UnqualifiedSyncSequence;
    } else {
      const bool sequence_intent = request.mode == Mode::Default;
      const SyncEvent *placement = &event;
      std::vector<SemanticSiteId> covered{event.semantic_id};
      std::vector<uint64_t> sequence_key;
      if (sequence_intent) {
        placement = completion_event(*logical_sequence, synchronization);
        if (placement == nullptr) {
          decision_kind = SiteDecisionKind::Unsupported;
          reason = BarrierPolicyReason::MissingCompletingEvent;
        } else {
          covered = covered_event_ids(synchronization, *logical_sequence);
          sequence_key = sequence_offsets(synchronization, *logical_sequence);
        }
      }
      if (placement != nullptr && !covered.empty() &&
          reason != BarrierPolicyReason::MissingCompletingEvent) {
        const auto existing =
            sequence_intent ? intent_by_sequence.find(sequence_key) : intent_by_sequence.end();
        if (existing != intent_by_sequence.end()) {
          intent_id = existing->second;
        } else {
          intent_id = ProbeIntentId{static_cast<uint32_t>(result.plan.probe_intents.size())};
          result.plan.probe_intents.push_back({
              .id = *intent_id,
              .mode = request.mode,
              .source_site = placement->source_site,
              .physical_site = placement->semantic_id.physical,
              .covered_semantic_sites = std::move(covered),
              .kind = vocabulary->barrier,
              .position = ProbePosition::After,
              .synchronization_association = std::nullopt,
              .dynamic_result = DynamicResultRequirement::None,
              .atomic_lowering_form = std::nullopt,
          });
          if (sequence_intent)
            intent_by_sequence.emplace(std::move(sequence_key), *intent_id);
        }
        decision_kind = SiteDecisionKind::Admitted;
        reason = BarrierPolicyReason::None;
      }
    }

    BarrierSiteDecision decision{
        .semantic_site = event.semantic_id,
        .kind = decision_kind,
        .reason = reason,
    };
    result.plan.barrier_site_decisions.push_back(std::move(decision));
  }
  return result;
}

} // namespace rocjitsu::consan
