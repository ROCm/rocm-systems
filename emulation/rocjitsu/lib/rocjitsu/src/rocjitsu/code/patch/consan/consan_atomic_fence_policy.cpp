// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"

#include <algorithm>
#include <map>
#include <ranges>
#include <tuple>

namespace rocjitsu {
namespace {

/// Unique logical sequence claiming one synchronization-event identity.
///
/// The pointer borrows immutable inventory for the duration of a pure policy
/// call. `ambiguous` is distinct from a missing pointer so policy can preserve
/// whether no sequence or multiple incompatible sequences caused rejection.
[[nodiscard]] bool owner_semantics_equal(std::span<const ConSanExecutionOwner> lhs,
                                         std::span<const ConSanExecutionOwner> rhs) {
  return consan_execution_owners_equal(lhs, rhs);
}

[[nodiscard]] bool event_policy_semantics_equal(const SynchronizationInventoryView &inventory,
                                                const ConSanSyncEvent &lhs,
                                                const ConSanSyncEvent &rhs) {
  const ConSanProgramSite *lhs_source = inventory.source(lhs);
  const ConSanProgramSite *rhs_source = inventory.source(rhs);
  return lhs_source != nullptr && rhs_source != nullptr && lhs_source->same_payload(*rhs_source) &&
         lhs.semantic_id.physical == rhs.semantic_id.physical &&
         std::tie(lhs.kind, lhs.operation, lhs.address_source, lhs.memory_role, lhs.rmw_outcome,
                  lhs.confidence, lhs.memory_role_confidence, lhs.scope) ==
             std::tie(rhs.kind, rhs.operation, rhs.address_source, rhs.memory_role, rhs.rmw_outcome,
                      rhs.confidence, rhs.memory_role_confidence, rhs.scope) &&
         owner_semantics_equal(inventory.execution_owners(lhs), inventory.execution_owners(rhs));
}

[[nodiscard]] bool fence_policy_semantics_equal(const ConSanMoiFenceCandidate &lhs,
                                                const ConSanMoiFenceCandidate &rhs) {
  return lhs.fence_event == rhs.fence_event && lhs.sequence == rhs.sequence &&
         lhs.communication_event == rhs.communication_event && lhs.memory_role == rhs.memory_role &&
         lhs.association == rhs.association;
}

[[nodiscard]] bool filter_matches(std::span<const std::string> names, std::string_view filter) {
  return filter.empty() || std::ranges::any_of(names, [&](const std::string &name) {
           return name.find(filter) != std::string::npos;
         });
}

[[nodiscard]] SemanticSiteId event_semantic_id(const ProgramInventory &inventory,
                                               const ConSanSyncEvent &event) {
  if (event.semantic_id.valid())
    return event.semantic_id;
  return {
      .physical = {.code_object = inventory.code_object_id(),
                   .original_text_offset = event.text_offset()},
      .domain = ConSanSemanticSiteDomain::SynchronizationEvent,
  };
}

[[nodiscard]] const ConSanAtomicSite *find_atomic_site(const ProgramInventory &inventory,
                                                       const ConSanSyncEvent &event) {
  return inventory.program_site<ConSanAtomicSite>(event.source_site);
}

[[nodiscard]] const ConSanOrdinaryMemorySite *find_ordinary_site(const ProgramInventory &inventory,
                                                                 const ConSanSyncEvent &event) {
  return inventory.program_site<ConSanOrdinaryMemorySite>(event.source_site);
}

[[nodiscard]] std::optional<ConSanCapabilityForm>
atomic_capability_form(const SynchronizationInventoryView &inventory,
                       const ConSanSyncEvent &event) {
  if (event.kind == ConSanSyncKind::OrdinaryMemory)
    return ConSanCapabilityForm::AddressedOrdinaryFence;
  if (event.kind != ConSanSyncKind::Atomic)
    return std::nullopt;
  const ConSanProgramSite *source = inventory.source(event);
  if (event.address_source == ConSanSyncAddressSource::LdsVector ||
      (source != nullptr && source->mnemonic_view().starts_with("ds_"))) {
    return ConSanCapabilityForm::OrderedLdsAtomic;
  }
  if (event.address_source == ConSanSyncAddressSource::FlatVector)
    return ConSanCapabilityForm::OrderedFlatAtomic;
  if (event.address_source == ConSanSyncAddressSource::GlobalScalarVector)
    return ConSanCapabilityForm::OrderedVglobalAtomic;
  return std::nullopt;
}

[[nodiscard]] ConSanDynamicResultRequirement dynamic_requirement(const ConSanSyncEvent &event) {
  switch (event.rmw_outcome) {
  case ConSanSyncRmwOutcome::NotApplicable:
  case ConSanSyncRmwOutcome::NoReturn:
    return ConSanDynamicResultRequirement::None;
  case ConSanSyncRmwOutcome::ReturnsOldValue:
    return ConSanDynamicResultRequirement::ReturnedOldValue;
  case ConSanSyncRmwOutcome::CompareExchange:
    return ConSanDynamicResultRequirement::CompareExchangeSuccess;
  case ConSanSyncRmwOutcome::Unknown:
    return ConSanDynamicResultRequirement::Count;
  }
  return ConSanDynamicResultRequirement::Count;
}

[[nodiscard]] ConSanAtomicPolicyReason
atomic_classifier_reason(ConSanAtomicClassifierReason reason) {
  switch (reason) {
  case ConSanAtomicClassifierReason::None:
    return ConSanAtomicPolicyReason::None;
  case ConSanAtomicClassifierReason::UnsupportedAddressSource:
    return ConSanAtomicPolicyReason::UnsupportedAddressSource;
  case ConSanAtomicClassifierReason::InvalidAccessWidth:
    return ConSanAtomicPolicyReason::InvalidAccessWidth;
  case ConSanAtomicClassifierReason::UnsupportedEncoding:
  case ConSanAtomicClassifierReason::NonzeroImmediateOffset:
    return ConSanAtomicPolicyReason::UnsupportedEncoding;
  case ConSanAtomicClassifierReason::MissingOperands:
    return ConSanAtomicPolicyReason::MissingOperands;
  case ConSanAtomicClassifierReason::UnsupportedInputWidth:
  case ConSanAtomicClassifierReason::ResultAddressAlias:
    return ConSanAtomicPolicyReason::MissingOperands;
  case ConSanAtomicClassifierReason::UnsupportedOffset:
    return ConSanAtomicPolicyReason::UnsupportedEncoding;
  case ConSanAtomicClassifierReason::CompareExchangeOutcomeUnavailable:
    return ConSanAtomicPolicyReason::CompareExchangeOutcomeUnavailable;
  case ConSanAtomicClassifierReason::MissingOrderingMetadata:
    return ConSanAtomicPolicyReason::MissingScope;
  case ConSanAtomicClassifierReason::UnsupportedScope:
    return ConSanAtomicPolicyReason::UnsupportedScope;
  case ConSanAtomicClassifierReason::TargetUnavailable:
  case ConSanAtomicClassifierReason::Count:
    return ConSanAtomicPolicyReason::TargetCapabilityUnavailable;
  }
  return ConSanAtomicPolicyReason::TargetCapabilityUnavailable;
}

struct AtomicEncodingDecision {
  ConSanAtomicPolicyReason reason = ConSanAtomicPolicyReason::TargetCapabilityUnavailable;
  std::optional<ConSanAtomicLoweringForm> form;
};

[[nodiscard]] AtomicEncodingDecision classify_atomic_encoding(const ProgramInventory &inventory,
                                                              const ConSanSyncEvent &event,
                                                              const ConSanSyncSequence &sequence,
                                                              ConSanCapabilityEngine engine) {
  const bool ordinary = event.kind == ConSanSyncKind::OrdinaryMemory;
  const ConSanAtomicSite *native = ordinary ? nullptr : find_atomic_site(inventory, event);
  const ConSanOrdinaryMemorySite *ordinary_site =
      ordinary ? find_ordinary_site(inventory, event) : nullptr;
  if ((ordinary && ordinary_site == nullptr) || (!ordinary && native == nullptr))
    return {.reason = ConSanAtomicPolicyReason::MissingOperands, .form = std::nullopt};
  ConSanAtomicSite site = ordinary ? consan_atomic_communication_site(*ordinary_site) : *native;

  // Synchronization analysis owns normalized semantic scope. It may preserve
  // an encoded value or derive a stronger fact from address-space provenance
  // and sequence association; policy always classifies that single contract.
  site.scope = sequence.scope;

  const ConSanAtomicLoweringClassification classification =
      classify_consan_atomic_lowering(site, inventory.arch(), !ordinary);
  const bool record_buffer_ordinary =
      ordinary && engine == ConSanCapabilityEngine::RecordReplay && classification.form &&
      classification.form->kind == ConSanAtomicLoweringFormKind::BufferResourceVectorOffset;
  const ConSanAtomicClassifierReason reason =
      engine == ConSanCapabilityEngine::InlineShadow || (ordinary && !record_buffer_ordinary)
          ? classification.exact_ordering_reason
          : classification.causal_ordering_reason;
  const ConSanAtomicPolicyReason policy_reason = atomic_classifier_reason(reason);
  return {
      .reason = policy_reason,
      .form = policy_reason == ConSanAtomicPolicyReason::None ? classification.form : std::nullopt,
  };
}

[[nodiscard]] bool sequence_is_atomic_contract(const ConSanSyncEvent &event,
                                               const ConSanSyncSequence &sequence) {
  if (event.kind == ConSanSyncKind::Atomic)
    return sequence.kind == ConSanSyncKind::Atomic;
  return event.kind == ConSanSyncKind::OrdinaryMemory &&
         sequence.kind == ConSanSyncKind::OrdinaryMemory;
}

[[nodiscard]] ConSanAtomicPolicyReason
classify_atomic_semantics(const ConSanSyncEvent &event, const ConSanSyncSequence *sequence,
                          bool ambiguous_membership) {
  if (ambiguous_membership)
    return ConSanAtomicPolicyReason::AmbiguousSequenceMembership;
  if (sequence == nullptr || !sequence_is_atomic_contract(event, *sequence) ||
      !consan_sync_confidence_meets(sequence->confidence, ConSanSemanticConfidence::Conservative) ||
      !consan_sync_confidence_meets(sequence->memory_role_confidence,
                                    ConSanSemanticConfidence::Conservative)) {
    return ConSanAtomicPolicyReason::UnqualifiedSyncSequence;
  }
  switch (sequence->memory_role) {
  case ConSanSyncMemoryRole::Release:
  case ConSanSyncMemoryRole::Acquire:
  case ConSanSyncMemoryRole::AcquireRelease:
    break;
  case ConSanSyncMemoryRole::Unknown:
  case ConSanSyncMemoryRole::None:
  case ConSanSyncMemoryRole::SequentiallyConsistent:
    return ConSanAtomicPolicyReason::UnsupportedMemoryRole;
  }
  const std::optional<ConSanMemoryScope> semantic_scope = sequence->scope;
  if (!semantic_scope)
    return ConSanAtomicPolicyReason::MissingScope;
  if (!consan_memory_scope_is_supported(*semantic_scope) ||
      *semantic_scope == ConSanMemoryScope::Wavefront)
    return ConSanAtomicPolicyReason::UnsupportedScope;
  if (event.kind == ConSanSyncKind::Atomic &&
      dynamic_requirement(event) == ConSanDynamicResultRequirement::Count) {
    return ConSanAtomicPolicyReason::UnsupportedDynamicOutcome;
  }
  return ConSanAtomicPolicyReason::None;
}

[[nodiscard]] ConSanProbeIntentId
add_intent(ConSanObservationPlan &plan, const PhysicalSiteId &physical_site,
           std::vector<SemanticSiteId> covered_sites, ConSanProbeIntentKind kind,
           ConSanProbePosition position, const ConSanSynchronizationAssociationId &association,
           ConSanDynamicResultRequirement dynamic_result) {
  const ConSanProbeIntentId id{static_cast<uint32_t>(plan.probe_intents.size())};
  plan.probe_intents.push_back({
      .id = id,
      .engine = plan.engine,
      .physical_site = physical_site,
      .covered_semantic_sites = std::move(covered_sites),
      .kind = kind,
      .position = position,
      .synchronization_association = association,
      .dynamic_result = dynamic_result,
  });
  return id;
}

void add_covered_site(ConSanObservationPlan &plan, ConSanProbeIntentId id,
                      const SemanticSiteId &site) {
  if (id.value >= plan.probe_intents.size())
    return;
  std::vector<SemanticSiteId> &covered = plan.probe_intents[id.value].covered_semantic_sites;
  if (std::ranges::find(covered, site) == covered.end())
    covered.push_back(site);
}

[[nodiscard]] bool has_qualified_fence_for(const SynchronizationInventoryView &inventory,
                                           ConSanSyncEventId communication) {
  return std::ranges::any_of(
      inventory.moi_fence_candidates, [&](const ConSanMoiFenceCandidate &fence) {
        return fence.eligible() && fence.communication_event == communication;
      });
}

} // namespace

ConSanAtomicFencePolicyResult
plan_consan_atomic_fence_observation(const ProgramInventory &inventory,
                                     const ConSanAtomicFencePolicyRequest &request) {
  ConSanAtomicFencePolicyResult result;
  result.plan.engine = request.engine;
  const ConSanEngineProbeVocabulary *vocabulary = consan_engine_probe_vocabulary(request.engine);
  if (vocabulary == nullptr || inventory.empty())
    return result;

  const SynchronizationInventoryView synchronization = inventory.sync();
  std::map<std::pair<ConSanSyncKind, uint64_t>, std::vector<const ConSanSyncEvent *>>
      aliases_by_site;
  for (const ConSanSyncEvent &event : synchronization.sync_events) {
    if (event.kind == ConSanSyncKind::Atomic || event.kind == ConSanSyncKind::OrdinaryMemory) {
      aliases_by_site[{event.kind, event.text_offset()}].push_back(&event);
    }
  }

  for (const auto &[site_key, aliases] : aliases_by_site) {
    (void)site_key;
    const ConSanSyncEvent &event = *aliases.front();
    const ConSanSyncEventId event_id = synchronization.event_id(event);
    const ConSanSyncSequenceMembership membership = synchronization.sequence_membership(event_id);
    const ConSanSyncSequence *sequence = synchronization.find_unique_sequence_containing(event_id);
    const bool ambiguous_membership = membership.ambiguous;
    // Ordinary memory belongs to access policy unless synchronization
    // analysis associated an acquire/release sequence around it.
    if (event.kind == ConSanSyncKind::OrdinaryMemory &&
        (sequence == nullptr || sequence->kind != ConSanSyncKind::OrdinaryMemory ||
         sequence->memory_role == ConSanSyncMemoryRole::Unknown ||
         sequence->memory_role == ConSanSyncMemoryRole::None)) {
      continue;
    }

    const SemanticSiteId semantic_id = event_semantic_id(inventory, event);
    const std::vector<std::string> names =
        synchronization.source_container_names(event.semantic_id.physical);
    const bool conflicting_alias = std::ranges::any_of(aliases, [&](const ConSanSyncEvent *alias) {
      return !event_policy_semantics_equal(synchronization, event, *alias);
    });
    const std::optional<ConSanCapabilityForm> form = atomic_capability_form(synchronization, event);
    const ConSanCapabilityDisposition capability =
        form ? consan_capability_disposition(inventory.target(), request.engine, *form)
             : ConSanCapabilityDisposition::OutOfContract;
    const std::vector<uint64_t> owner_descriptors =
        synchronization.execution_owner_descriptors(aliases);
    ConSanSiteDecisionKind kind = ConSanSiteDecisionKind::NotApplicable;
    ConSanAtomicPolicyReason reason = ConSanAtomicPolicyReason::TrackingDisabled;
    std::optional<ConSanAtomicLoweringForm> lowering_form;

    if (!request.tracking_enabled) {
      reason = ConSanAtomicPolicyReason::TrackingDisabled;
    } else if (request.engine == ConSanCapabilityEngine::SuperCollider) {
      reason = ConSanAtomicPolicyReason::EngineMutationOnly;
    } else if (!filter_matches(names, request.container_filter) ||
               !consan_site_matches_kernel_allowlist(inventory, owner_descriptors, names,
                                                     request.kernel_name_allowlist)) {
      reason = ConSanAtomicPolicyReason::ContainerFilterExcluded;
    } else if (synchronization.execution_owners(event).empty()) {
      reason = ConSanAtomicPolicyReason::MissingExecutionOwner;
    } else if (request.engine == ConSanCapabilityEngine::Sampled &&
               !request.sampled_access_window_available) {
      reason = ConSanAtomicPolicyReason::MissingSampledAccessWindow;
    } else if (!form || (capability != ConSanCapabilityDisposition::Supported &&
                         capability != ConSanCapabilityDisposition::AssociatedOnly)) {
      reason = ConSanAtomicPolicyReason::TargetCapabilityUnavailable;
    } else if (conflicting_alias) {
      kind = ConSanSiteDecisionKind::Unsupported;
      reason = ConSanAtomicPolicyReason::ConflictingPhysicalAliases;
      result.atomic_errors.push_back(reason);
    } else {
      reason = classify_atomic_semantics(event, sequence, ambiguous_membership);
      if (reason == ConSanAtomicPolicyReason::None) {
        AtomicEncodingDecision encoding =
            classify_atomic_encoding(inventory, event, *sequence, request.engine);
        reason = encoding.reason;
        lowering_form = std::move(encoding.form);
      }
      const bool semantic_not_applicable =
          reason == ConSanAtomicPolicyReason::UnqualifiedSyncSequence ||
          (reason == ConSanAtomicPolicyReason::UnsupportedMemoryRole && sequence &&
           (sequence->memory_role == ConSanSyncMemoryRole::Unknown ||
            sequence->memory_role == ConSanSyncMemoryRole::None)) ||
          (reason == ConSanAtomicPolicyReason::UnsupportedScope && sequence &&
           sequence->scope == ConSanMemoryScope::Wavefront);
      kind = reason == ConSanAtomicPolicyReason::None ? ConSanSiteDecisionKind::Admitted
             : semantic_not_applicable                ? ConSanSiteDecisionKind::NotApplicable
                                                      : ConSanSiteDecisionKind::Unsupported;
    }

    ConSanDynamicResultRequirement required_dynamic_result =
        event.kind == ConSanSyncKind::Atomic ? dynamic_requirement(event)
                                             : ConSanDynamicResultRequirement::None;
    if (required_dynamic_result == ConSanDynamicResultRequirement::Count)
      required_dynamic_result = ConSanDynamicResultRequirement::None;
    ConSanAtomicSiteDecision decision{
        .engine = request.engine,
        .semantic_site = semantic_id,
        .kind = kind,
        .capability = capability,
        .reason = reason,
        .association = std::nullopt,
        .dynamic_result = required_dynamic_result,
        .lowering_form = std::move(lowering_form),
        .intent_ids = {},
    };
    if (sequence != nullptr)
      decision.association = ConSanSynchronizationAssociationId{sequence->identity};

    if (decision.kind == ConSanSiteDecisionKind::Admitted && decision.association) {
      const ConSanProbeIntentId capture =
          add_intent(result.plan, semantic_id.physical, {semantic_id},
                     ConSanProbeIntentKind::AtomicAddressCapture, ConSanProbePosition::Before,
                     *decision.association, ConSanDynamicResultRequirement::None);
      decision.intent_ids.push_back(capture);
      const bool independent_fence_owns_ordinary =
          vocabulary->fence != ConSanProbeIntentKind::Count &&
          event.kind == ConSanSyncKind::OrdinaryMemory &&
          has_qualified_fence_for(synchronization, synchronization.event_id(event));
      if (!independent_fence_owns_ordinary) {
        decision.intent_ids.push_back(add_intent(result.plan, semantic_id.physical, {semantic_id},
                                                 vocabulary->atomic, ConSanProbePosition::After,
                                                 *decision.association, decision.dynamic_result));
      }
    }
    result.plan.atomic_site_decisions.push_back(std::move(decision));
  }

  std::map<uint64_t, std::vector<const ConSanMoiFenceCandidate *>> fence_aliases_by_site;
  for (const ConSanMoiFenceCandidate &fence : synchronization.moi_fence_candidates) {
    if (const ConSanSyncEvent *event = synchronization.find_event(fence.fence_event))
      fence_aliases_by_site[event->text_offset()].push_back(&fence);
  }

  for (const auto &fence_aliases : fence_aliases_by_site) {
    const auto &aliases = fence_aliases.second;
    const ConSanMoiFenceCandidate &fence = *aliases.front();
    const ConSanSyncEvent *fence_event = synchronization.find_event(fence.fence_event);
    std::vector<const ConSanSyncEvent *> fence_events;
    fence_events.reserve(aliases.size());
    for (const ConSanMoiFenceCandidate *alias : aliases) {
      if (const ConSanSyncEvent *event = synchronization.find_event(alias->fence_event)) {
        fence_events.push_back(event);
      }
    }
    const SemanticSiteId fence_id =
        fence_event == nullptr ? SemanticSiteId{} : fence_event->semantic_id;
    const std::vector<std::string> names =
        fence_event == nullptr
            ? std::vector<std::string>{}
            : synchronization.source_container_names(fence_event->semantic_id.physical);
    const ConSanCapabilityDisposition capability = consan_capability_disposition(
        inventory.target(), request.engine, ConSanCapabilityForm::AddressedOrdinaryFence);
    const bool conflicting_alias =
        std::ranges::any_of(aliases, [&](const ConSanMoiFenceCandidate *alias) {
          return !fence_policy_semantics_equal(fence, *alias);
        });
    ConSanFenceSiteDecision decision{
        .engine = request.engine,
        .semantic_site = fence_id,
        .kind = ConSanSiteDecisionKind::NotApplicable,
        .capability = capability,
        .reason = ConSanFencePolicyReason::TrackingDisabled,
        .inventory_association = fence.association,
        .association = std::nullopt,
        .communication_lowering_form = std::nullopt,
        .intent_ids = {},
    };
    if (const ConSanSyncSequence *sequence = synchronization.find_sequence(fence.sequence))
      decision.association = ConSanSynchronizationAssociationId{sequence->identity};
    const std::vector<uint64_t> owner_descriptors =
        synchronization.execution_owner_descriptors(fence_events);

    if (!request.tracking_enabled) {
      decision.reason = ConSanFencePolicyReason::TrackingDisabled;
    } else if (request.engine == ConSanCapabilityEngine::SuperCollider) {
      decision.reason = ConSanFencePolicyReason::EngineMutationOnly;
    } else if (!filter_matches(names, request.container_filter) ||
               !consan_site_matches_kernel_allowlist(inventory, owner_descriptors,
                                                     names,
                                                     request.kernel_name_allowlist)) {
      decision.reason = ConSanFencePolicyReason::ContainerFilterExcluded;
    } else if (conflicting_alias) {
      decision.kind = ConSanSiteDecisionKind::Unsupported;
      decision.reason = ConSanFencePolicyReason::ConflictingPhysicalAliases;
      result.fence_errors.push_back(decision.reason);
    } else if (!fence.eligible()) {
      decision.reason = ConSanFencePolicyReason::AssociationUnavailable;
    } else if (capability != ConSanCapabilityDisposition::Supported &&
               capability != ConSanCapabilityDisposition::AssociatedOnly) {
      decision.reason = ConSanFencePolicyReason::TargetCapabilityUnavailable;
    } else {
      const ConSanSyncEvent *communication =
          fence.communication_event ? synchronization.find_event(*fence.communication_event)
                                    : nullptr;
      const auto atomic_decision =
          communication == nullptr
              ? result.plan.atomic_site_decisions.end()
              : std::ranges::find_if(
                    result.plan.atomic_site_decisions,
                    [&](const ConSanAtomicSiteDecision &candidate) {
                      return candidate.semantic_site.physical ==
                                 event_semantic_id(inventory, *communication).physical &&
                             candidate.kind == ConSanSiteDecisionKind::Admitted &&
                             candidate.association == decision.association;
                    });
      if (fence_event != nullptr && communication != nullptr &&
          (synchronization.execution_owners(*fence_event).empty() ||
           synchronization.execution_owners(*communication).empty())) {
        decision.reason = ConSanFencePolicyReason::MissingExecutionOwner;
      } else if (fence_event == nullptr || communication == nullptr ||
                 atomic_decision == result.plan.atomic_site_decisions.end()) {
        decision.kind = ConSanSiteDecisionKind::Unsupported;
        decision.reason = ConSanFencePolicyReason::MissingCommunicationEvent;
      } else {
        decision.kind = ConSanSiteDecisionKind::Admitted;
        decision.reason = ConSanFencePolicyReason::None;
        decision.communication_lowering_form = atomic_decision->lowering_form;
        if (vocabulary->fence != ConSanProbeIntentKind::Count) {
          if (!atomic_decision->intent_ids.empty()) {
            const ConSanProbeIntentId capture = atomic_decision->intent_ids.front();
            add_covered_site(result.plan, capture, fence_id);
            decision.intent_ids.push_back(capture);
          }
          const ConSanProbeIntentId record =
              add_intent(result.plan, fence_id.physical, {atomic_decision->semantic_site, fence_id},
                         vocabulary->fence, ConSanProbePosition::After, *decision.association,
                         ConSanDynamicResultRequirement::None);
          decision.intent_ids.push_back(record);
          atomic_decision->intent_ids.push_back(record);
        } else {
          decision.intent_ids = atomic_decision->intent_ids;
          for (ConSanProbeIntentId id : decision.intent_ids)
            add_covered_site(result.plan, id, fence_id);
        }
      }
    }
    result.plan.fence_site_decisions.push_back(std::move(decision));
  }

  // An engine with independent fence evidence normally delegates a qualified
  // ordinary sequence's after evidence to that fence. If a corrupt inventory
  // claimed a qualified candidate but no usable fence decision survived, fall
  // back to the engine's direct atomic evidence rather than publishing a
  // before-only admitted contract.
  if (vocabulary->fence != ConSanProbeIntentKind::Count) {
    for (ConSanAtomicSiteDecision &decision : result.plan.atomic_site_decisions) {
      if (decision.kind != ConSanSiteDecisionKind::Admitted || !decision.association ||
          decision.intent_ids.size() != 1u)
        continue;
      decision.intent_ids.push_back(add_intent(result.plan, decision.semantic_site.physical,
                                               {decision.semantic_site}, vocabulary->atomic,
                                               ConSanProbePosition::After, *decision.association,
                                               decision.dynamic_result));
    }
  }

  return result;
}

} // namespace rocjitsu
