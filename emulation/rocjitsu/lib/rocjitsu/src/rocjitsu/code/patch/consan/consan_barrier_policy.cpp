// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"

#include <algorithm>
#include <map>
#include <ranges>
#include <tuple>

namespace rocjitsu {
namespace {

[[nodiscard]] bool valid_engine(ConSanCapabilityEngine engine) {
  return static_cast<uint8_t>(engine) < static_cast<uint8_t>(ConSanCapabilityEngine::Count);
}

[[nodiscard]] bool runtime_kernel(std::string_view name) {
  return name.starts_with("__amd_rocclr_");
}

[[nodiscard]] bool event_semantics_equal(const ConSanSyncEvent &lhs, const ConSanSyncEvent &rhs) {
  return std::tie(lhs.kind, lhs.operation, lhs.address_source, lhs.memory_role, lhs.rmw_outcome,
                  lhs.confidence, lhs.memory_role_confidence, lhs.file_offset, lhs.size,
                  lhs.width_bits, lhs.mnemonic, lhs.static_byte_offset, lhs.raw_scope,
                  lhs.barrier_id, lhs.barrier_operand_source, lhs.barrier_raw_operand_selector,
                  lhs.barrier_literal_width_bits, lhs.barrier_literal_value, lhs.barrier_raw_simm16,
                  lhs.barrier_scope, lhs.participant_count, lhs.participant_mask) ==
         std::tie(rhs.kind, rhs.operation, rhs.address_source, rhs.memory_role, rhs.rmw_outcome,
                  rhs.confidence, rhs.memory_role_confidence, rhs.file_offset, rhs.size,
                  rhs.width_bits, rhs.mnemonic, rhs.static_byte_offset, rhs.raw_scope,
                  rhs.barrier_id, rhs.barrier_operand_source, rhs.barrier_raw_operand_selector,
                  rhs.barrier_literal_width_bits, rhs.barrier_literal_value, rhs.barrier_raw_simm16,
                  rhs.barrier_scope, rhs.participant_count, rhs.participant_mask);
}

[[nodiscard]] std::vector<uint64_t> sequence_offsets(const ConSanSyncSequence &sequence) {
  std::vector<uint64_t> offsets;
  offsets.reserve(sequence.member_semantic_ids.size());
  for (const SemanticSiteId &member : sequence.member_semantic_ids)
    offsets.push_back(member.physical.original_text_offset);
  return offsets;
}

[[nodiscard]] bool sequence_semantics_equal(const ConSanSyncSequence &lhs,
                                            const ConSanSyncSequence &rhs) {
  return sequence_offsets(lhs) == sequence_offsets(rhs) &&
         std::tie(lhs.kind, lhs.operation, lhs.address_source, lhs.memory_role, lhs.rmw_outcome,
                  lhs.confidence, lhs.memory_role_confidence, lhs.begin_text_offset,
                  lhs.end_text_offset, lhs.basic_block_index, lhs.in_cyclic_cfg_component,
                  lhs.inside_scalar_clause, lhs.width_bits, lhs.static_byte_offset, lhs.raw_scope,
                  lhs.barrier_id, lhs.barrier_operand_source, lhs.barrier_raw_operand_selector,
                  lhs.barrier_literal_width_bits, lhs.barrier_literal_value, lhs.barrier_raw_simm16,
                  lhs.barrier_scope, lhs.participant_count, lhs.participant_mask) ==
             std::tie(rhs.kind, rhs.operation, rhs.address_source, rhs.memory_role, rhs.rmw_outcome,
                      rhs.confidence, rhs.memory_role_confidence, rhs.begin_text_offset,
                      rhs.end_text_offset, rhs.basic_block_index, rhs.in_cyclic_cfg_component,
                      rhs.inside_scalar_clause, rhs.width_bits, rhs.static_byte_offset,
                      rhs.raw_scope, rhs.barrier_id, rhs.barrier_operand_source,
                      rhs.barrier_raw_operand_selector, rhs.barrier_literal_width_bits,
                      rhs.barrier_literal_value, rhs.barrier_raw_simm16, rhs.barrier_scope,
                      rhs.participant_count, rhs.participant_mask);
}

[[nodiscard]] std::vector<std::string>
container_names(std::span<const ConSanSyncEvent *const> aliases) {
  std::vector<std::string> names;
  names.reserve(aliases.size());
  for (const ConSanSyncEvent *event : aliases) {
    const std::span<const std::string> sources = event->source_containers.empty()
                                                     ? std::span(&event->container_name, 1u)
                                                     : std::span(event->source_containers);
    for (const std::string &source : sources) {
      if (std::ranges::find(names, source) == names.end())
        names.push_back(source);
    }
  }
  std::ranges::sort(names);
  return names;
}

[[nodiscard]] ConSanCapabilityForm barrier_form(const ConSanSyncEvent &event) {
  return event.barrier_scope == ConSanBarrierSite::Scope::Cluster
             ? ConSanCapabilityForm::ClusterBarrier
             : ConSanCapabilityForm::WorkgroupBarrier;
}

[[nodiscard]] ConSanProbeIntentKind barrier_intent_kind(ConSanCapabilityEngine engine) {
  switch (engine) {
  case ConSanCapabilityEngine::RecordReplay:
    return ConSanProbeIntentKind::BarrierRecord;
  case ConSanCapabilityEngine::Sampled:
    return ConSanProbeIntentKind::SampledBarrierEpoch;
  case ConSanCapabilityEngine::InlineShadow:
    return ConSanProbeIntentKind::ExactBarrierEpoch;
  case ConSanCapabilityEngine::SuperCollider:
  case ConSanCapabilityEngine::Count:
    break;
  }
  return ConSanProbeIntentKind::Count;
}

[[nodiscard]] const ConSanSyncEvent *
completion_event(const ConSanSyncSequence &sequence,
                 const std::map<uint64_t, std::vector<const ConSanSyncEvent *>> &events_by_offset) {
  const ConSanSyncEvent *result = nullptr;
  for (const SemanticSiteId &member : sequence.member_semantic_ids) {
    const auto found = events_by_offset.find(member.physical.original_text_offset);
    if (found == events_by_offset.end() || found->second.empty())
      continue;
    const ConSanSyncEvent *event = found->second.front();
    if (event->text_offset <= sequence.end_text_offset &&
        event->size == sequence.end_text_offset - event->text_offset) {
      if (result != nullptr && result->semantic_id.physical != event->semantic_id.physical)
        return nullptr;
      result = event;
    }
  }
  return result;
}

[[nodiscard]] std::vector<SemanticSiteId> covered_event_ids(
    const ConSanSyncSequence &sequence,
    const std::map<uint64_t, std::vector<const ConSanSyncEvent *>> &events_by_offset) {
  std::vector<SemanticSiteId> ids;
  ids.reserve(sequence.member_semantic_ids.size());
  for (const SemanticSiteId &member : sequence.member_semantic_ids) {
    const auto found = events_by_offset.find(member.physical.original_text_offset);
    if (found != events_by_offset.end() && !found->second.empty())
      ids.push_back(found->second.front()->semantic_id);
  }
  return ids;
}

} // namespace

ConSanBarrierPolicyResult
plan_consan_barrier_observation(const ProgramInventory &inventory,
                                const ConSanBarrierPolicyRequest &request) {
  ConSanBarrierPolicyResult result;
  result.plan.engine = request.engine;
  if (!valid_engine(request.engine) || inventory.empty())
    return result;

  const SynchronizationInventoryView synchronization = inventory.sync();
  std::map<uint64_t, std::vector<const ConSanSyncEvent *>> events_by_offset;
  for (const ConSanSyncEvent &event : synchronization.sync_events) {
    if (event.kind == ConSanSyncEventKind::Barrier)
      events_by_offset[event.semantic_id.physical.original_text_offset].push_back(&event);
  }

  std::map<uint64_t, std::vector<const ConSanSyncSequence *>> sequences_by_member_offset;
  for (const ConSanSyncSequence &sequence : synchronization.sync_sequences) {
    if (sequence.kind != ConSanSyncSequenceKind::Barrier)
      continue;
    for (const SemanticSiteId &member : sequence.member_semantic_ids) {
      auto &sequences = sequences_by_member_offset[member.physical.original_text_offset];
      if (std::ranges::none_of(sequences, [&](const auto *known) {
            return sequence_semantics_equal(*known, sequence);
          })) {
        sequences.push_back(&sequence);
      }
    }
  }

  std::map<std::vector<uint64_t>, ConSanProbeIntentId> intent_by_sequence;
  std::map<std::string, std::pair<uint64_t, uint32_t>> previous_full_by_container;
  for (const auto &[offset, aliases] : events_by_offset) {
    const ConSanSyncEvent &event = *aliases.front();
    const std::vector<std::string> names = container_names(aliases);
    ConSanSiteDecisionKind decision_kind = ConSanSiteDecisionKind::NotApplicable;
    ConSanBarrierPolicyReason reason = ConSanBarrierPolicyReason::TrackingDisabled;
    std::optional<ConSanProbeIntentId> intent_id;

    const bool conflicting_alias = std::ranges::any_of(
        aliases, [&](const auto *alias) { return !event_semantics_equal(event, *alias); });
    const bool filtered = !request.container_filter.empty() &&
                          std::ranges::none_of(names, [&](const std::string &name) {
                            return name.find(request.container_filter) != std::string::npos;
                          });
    const bool only_runtime_kernels = std::ranges::all_of(aliases, [](const auto *alias) {
      return alias->in_kernel && runtime_kernel(alias->container_name);
    });
    bool redundant = false;
    if (event.operation == ConSanSyncOperation::BarrierFull) {
      redundant = std::ranges::any_of(names, [&](const std::string &name) {
        const auto previous = previous_full_by_container.find(name);
        return previous != previous_full_by_container.end() &&
               previous->second.first + previous->second.second == offset;
      });
      for (const std::string &name : names)
        previous_full_by_container[name] = {offset, event.size};
    } else {
      for (const std::string &name : names)
        previous_full_by_container.erase(name);
    }

    const ConSanCapabilityDisposition capability =
        consan_capability_disposition(inventory.target(), request.engine, barrier_form(event));
    const auto member_sequences = sequences_by_member_offset.find(offset);
    std::vector<const ConSanSyncSequence *> usable_sequences;
    if (member_sequences != sequences_by_member_offset.end()) {
      for (const ConSanSyncSequence *sequence : member_sequences->second) {
        if (request.engine != ConSanCapabilityEngine::Sampled ||
            consan_moi_sampled_qualifies_barrier_sequence(*sequence)) {
          usable_sequences.push_back(sequence);
        }
      }
    }

    const ConSanSyncSequence *logical_sequence = nullptr;
    if (!usable_sequences.empty()) {
      logical_sequence = usable_sequences.front();
      if (std::ranges::any_of(usable_sequences, [&](const auto *candidate) {
            return !sequence_semantics_equal(*logical_sequence, *candidate);
          })) {
        reason = ConSanBarrierPolicyReason::AmbiguousSequenceMembership;
        decision_kind = ConSanSiteDecisionKind::Unsupported;
        logical_sequence = nullptr;
      }
    }

    if (conflicting_alias) {
      decision_kind = ConSanSiteDecisionKind::Unsupported;
      reason = ConSanBarrierPolicyReason::ConflictingPhysicalAliases;
      result.errors.push_back(reason);
    } else if (!request.tracking_enabled) {
      reason = ConSanBarrierPolicyReason::TrackingDisabled;
    } else if (request.engine == ConSanCapabilityEngine::SuperCollider ||
               capability == ConSanCapabilityDisposition::MutationOnly) {
      reason = ConSanBarrierPolicyReason::EngineMutationOnly;
    } else if (filtered) {
      reason = ConSanBarrierPolicyReason::ContainerFilterExcluded;
    } else if (only_runtime_kernels) {
      reason = ConSanBarrierPolicyReason::RuntimeKernelExcluded;
    } else if (capability != ConSanCapabilityDisposition::Supported) {
      reason = ConSanBarrierPolicyReason::TargetCapabilityUnavailable;
    } else if (redundant) {
      reason = ConSanBarrierPolicyReason::RedundantAdjacentFullBarrier;
    } else if (request.engine != ConSanCapabilityEngine::Sampled &&
               event.size != sizeof(uint32_t)) {
      decision_kind = ConSanSiteDecisionKind::Unsupported;
      reason = ConSanBarrierPolicyReason::InvalidBarrierEncoding;
    } else if (reason == ConSanBarrierPolicyReason::AmbiguousSequenceMembership) {
      // The typed ambiguity selected above is already the final answer.
    } else if (request.engine == ConSanCapabilityEngine::Sampled && logical_sequence == nullptr) {
      decision_kind = ConSanSiteDecisionKind::Unsupported;
      reason = ConSanBarrierPolicyReason::UnqualifiedSyncSequence;
    } else if (request.engine == ConSanCapabilityEngine::InlineShadow &&
               event.operation == ConSanSyncOperation::BarrierSignal &&
               (logical_sequence == nullptr || logical_sequence->member_semantic_ids.size() < 2u)) {
      reason = ConSanBarrierPolicyReason::UnqualifiedSyncSequence;
    } else {
      const bool sequence_intent =
          request.engine == ConSanCapabilityEngine::Sampled ||
          (request.engine == ConSanCapabilityEngine::InlineShadow && logical_sequence != nullptr &&
           logical_sequence->member_semantic_ids.size() > 1u);
      const ConSanSyncEvent *placement = &event;
      std::vector<SemanticSiteId> covered{event.semantic_id};
      std::vector<uint64_t> sequence_key;
      if (sequence_intent) {
        placement = completion_event(*logical_sequence, events_by_offset);
        if (placement == nullptr) {
          decision_kind = ConSanSiteDecisionKind::Unsupported;
          reason = ConSanBarrierPolicyReason::MissingCompletingEvent;
        } else {
          covered = covered_event_ids(*logical_sequence, events_by_offset);
          sequence_key = sequence_offsets(*logical_sequence);
        }
      }
      if (placement != nullptr && !covered.empty() &&
          reason != ConSanBarrierPolicyReason::MissingCompletingEvent) {
        const auto existing =
            sequence_intent ? intent_by_sequence.find(sequence_key) : intent_by_sequence.end();
        if (existing != intent_by_sequence.end()) {
          intent_id = existing->second;
        } else {
          intent_id = ConSanProbeIntentId{static_cast<uint32_t>(result.plan.probe_intents.size())};
          result.plan.probe_intents.push_back({
              .id = *intent_id,
              .engine = request.engine,
              .physical_site = placement->semantic_id.physical,
              .covered_semantic_sites = std::move(covered),
              .kind = barrier_intent_kind(request.engine),
              .position = ConSanProbePosition::After,
              .synchronization_association = std::nullopt,
              .dynamic_result = ConSanDynamicResultRequirement::None,
          });
          if (sequence_intent)
            intent_by_sequence.emplace(std::move(sequence_key), *intent_id);
        }
        decision_kind = ConSanSiteDecisionKind::Admitted;
        reason = ConSanBarrierPolicyReason::None;
      }
    }

    ConSanBarrierSiteDecision decision{
        .engine = request.engine,
        .semantic_site = event.semantic_id,
        .kind = decision_kind,
        .reason = reason,
        .intent_ids = {},
        .source_containers = names,
    };
    if (intent_id)
      decision.intent_ids.push_back(*intent_id);
    result.plan.barrier_site_decisions.push_back(std::move(decision));
  }
  return result;
}

} // namespace rocjitsu
