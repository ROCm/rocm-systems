// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"

#include <algorithm>
#include <map>
#include <ranges>
#include <tuple>

namespace rocjitsu::consan {
namespace {

/// Unique logical sequence claiming one synchronization-event identity.
///
/// The pointer borrows immutable inventory for the duration of a pure policy
/// call. `ambiguous` is distinct from a missing pointer so policy can preserve
/// whether no sequence or multiple incompatible sequences caused rejection.
[[nodiscard]] bool owner_semantics_equal(std::span<const ExecutionOwner> lhs,
                                         std::span<const ExecutionOwner> rhs) {
  return execution_owners_equal(lhs, rhs);
}

[[nodiscard]] bool event_policy_semantics_equal(const SynchronizationInventoryView &inventory,
                                                const SyncEvent &lhs, const SyncEvent &rhs) {
  const ProgramSite *lhs_source = inventory.source(lhs);
  const ProgramSite *rhs_source = inventory.source(rhs);
  return lhs_source != nullptr && rhs_source != nullptr && lhs_source->same_payload(*rhs_source) &&
         lhs.semantic_id.physical == rhs.semantic_id.physical &&
         std::tie(lhs.kind, lhs.operation, lhs.address_source, lhs.memory_role, lhs.rmw_outcome,
                  lhs.confidence, lhs.memory_role_confidence, lhs.scope) ==
             std::tie(rhs.kind, rhs.operation, rhs.address_source, rhs.memory_role, rhs.rmw_outcome,
                      rhs.confidence, rhs.memory_role_confidence, rhs.scope) &&
         owner_semantics_equal(inventory.execution_owners(lhs), inventory.execution_owners(rhs));
}

[[nodiscard]] bool fence_policy_semantics_equal(const FenceCandidate &lhs,
                                                const FenceCandidate &rhs) {
  return lhs.fence_event == rhs.fence_event && lhs.sequence == rhs.sequence &&
         lhs.communication_event == rhs.communication_event && lhs.memory_role == rhs.memory_role &&
         lhs.association == rhs.association;
}

[[nodiscard]] bool filter_matches(std::span<const std::string> names, std::string_view filter) {
  return filter.empty() || std::ranges::any_of(names, [&](const std::string &name) {
           return name.find(filter) != std::string::npos;
         });
}

[[nodiscard]] bool directional_access_window_available(std::span<const ExecutionOwner> owners,
                                                       SyncMemoryRole role,
                                                       const AtomicFencePolicyRequest &request) {
  return std::ranges::any_of(owners, [&](const ExecutionOwner &owner) {
    const auto availability = std::ranges::find(request.directional_access_windows, owner.kernel,
                                                &DirectionalAccessAvailability::owner);
    if (availability == request.directional_access_windows.end())
      return false;
    switch (role) {
    case SyncMemoryRole::Release:
      return availability->write;
    case SyncMemoryRole::Acquire:
      return availability->read;
    case SyncMemoryRole::AcquireRelease:
      return availability->read || availability->write;
    case SyncMemoryRole::Unknown:
    case SyncMemoryRole::None:
    case SyncMemoryRole::SequentiallyConsistent:
      return false;
    }
    return false;
  });
}

[[nodiscard]] SemanticSiteId event_semantic_id(const ProgramInventory &inventory,
                                               const SyncEvent &event) {
  if (event.semantic_id.valid())
    return event.semantic_id;
  return {
      .physical = {.code_object = inventory.code_object_id(),
                   .original_text_offset = event.text_offset()},
      .domain = SemanticSiteDomain::SynchronizationEvent,
  };
}

[[nodiscard]] const AtomicSite *find_atomic_site(const ProgramInventory &inventory,
                                                 const SyncEvent &event) {
  return inventory.program_site<AtomicSite>(event.source_site);
}

[[nodiscard]] const OrdinaryMemorySite *find_ordinary_site(const ProgramInventory &inventory,
                                                           const SyncEvent &event) {
  return inventory.program_site<OrdinaryMemorySite>(event.source_site);
}

[[nodiscard]] std::optional<CapabilityForm>
atomic_capability_form(const SynchronizationInventoryView &inventory, const SyncEvent &event) {
  if (event.kind == SyncKind::OrdinaryMemory)
    return CapabilityForm::AddressedOrdinaryFence;
  if (event.kind != SyncKind::Atomic)
    return std::nullopt;
  const ProgramSite *source = inventory.source(event);
  if (event.address_source == SyncAddressSource::LdsVector ||
      (source != nullptr && source->mnemonic_view().starts_with("ds_"))) {
    return CapabilityForm::OrderedLdsAtomic;
  }
  if (event.address_source == SyncAddressSource::FlatVector)
    return CapabilityForm::OrderedFlatAtomic;
  if (event.address_source == SyncAddressSource::GlobalScalarVector)
    return CapabilityForm::OrderedVglobalAtomic;
  return std::nullopt;
}

[[nodiscard]] DynamicResultRequirement dynamic_requirement(const SyncEvent &event) {
  switch (event.rmw_outcome) {
  case SyncRmwOutcome::NotApplicable:
  case SyncRmwOutcome::NoReturn:
    return DynamicResultRequirement::None;
  case SyncRmwOutcome::ReturnsOldValue:
    return DynamicResultRequirement::ReturnedOldValue;
  case SyncRmwOutcome::CompareExchange:
    return DynamicResultRequirement::CompareExchangeSuccess;
  case SyncRmwOutcome::Unknown:
    return DynamicResultRequirement::Count;
  }
  return DynamicResultRequirement::Count;
}

[[nodiscard]] AtomicPolicyReason atomic_classifier_reason(AtomicClassifierReason reason) {
  switch (reason) {
  case AtomicClassifierReason::None:
    return AtomicPolicyReason::None;
  case AtomicClassifierReason::UnsupportedAddressSource:
    return AtomicPolicyReason::UnsupportedAddressSource;
  case AtomicClassifierReason::InvalidAccessWidth:
    return AtomicPolicyReason::InvalidAccessWidth;
  case AtomicClassifierReason::UnsupportedEncoding:
  case AtomicClassifierReason::NonzeroImmediateOffset:
    return AtomicPolicyReason::UnsupportedEncoding;
  case AtomicClassifierReason::MissingOperands:
    return AtomicPolicyReason::MissingOperands;
  case AtomicClassifierReason::UnsupportedInputWidth:
  case AtomicClassifierReason::ResultAddressAlias:
    return AtomicPolicyReason::MissingOperands;
  case AtomicClassifierReason::UnsupportedOffset:
    return AtomicPolicyReason::UnsupportedEncoding;
  case AtomicClassifierReason::CompareExchangeOutcomeUnavailable:
    return AtomicPolicyReason::CompareExchangeOutcomeUnavailable;
  case AtomicClassifierReason::MissingOrderingMetadata:
    return AtomicPolicyReason::MissingScope;
  case AtomicClassifierReason::UnsupportedScope:
    return AtomicPolicyReason::UnsupportedScope;
  case AtomicClassifierReason::TargetUnavailable:
  case AtomicClassifierReason::Count:
    return AtomicPolicyReason::TargetCapabilityUnavailable;
  }
  return AtomicPolicyReason::TargetCapabilityUnavailable;
}

struct AtomicEncodingDecision {
  AtomicPolicyReason reason = AtomicPolicyReason::TargetCapabilityUnavailable;
  std::optional<AtomicLoweringForm> form;
};

[[nodiscard]] AtomicEncodingDecision classify_atomic_encoding(const ProgramInventory &inventory,
                                                              const SyncEvent &event,
                                                              const SyncSequence &sequence,
                                                              Mode mode) {
  const bool ordinary = event.kind == SyncKind::OrdinaryMemory;
  const AtomicSite *native = ordinary ? nullptr : find_atomic_site(inventory, event);
  const OrdinaryMemorySite *ordinary_site =
      ordinary ? find_ordinary_site(inventory, event) : nullptr;
  if ((ordinary && ordinary_site == nullptr) || (!ordinary && native == nullptr))
    return {.reason = AtomicPolicyReason::MissingOperands, .form = std::nullopt};
  AtomicSite site = ordinary ? atomic_communication_site(*ordinary_site) : *native;

  // Synchronization analysis owns normalized semantic scope. It may preserve
  // an encoded value or derive a stronger fact from address-space provenance
  // and sequence association; policy always classifies that single contract.
  site.scope = sequence.scope;

  const AtomicLoweringClassification classification =
      classify_atomic_lowering(site, inventory.arch(), !ordinary);
  const bool causal_buffer_ordinary =
      ordinary && mode == Mode::Default && classification.form &&
      classification.form->kind == AtomicLoweringFormKind::BufferResourceVectorOffset;
  const AtomicClassifierReason reason = ordinary && !causal_buffer_ordinary
                                            ? classification.exact_ordering_reason
                                            : classification.causal_ordering_reason;
  const AtomicPolicyReason policy_reason = atomic_classifier_reason(reason);
  return {
      .reason = policy_reason,
      .form = policy_reason == AtomicPolicyReason::None ? classification.form : std::nullopt,
  };
}

[[nodiscard]] bool sequence_is_atomic_contract(const SyncEvent &event,
                                               const SyncSequence &sequence) {
  if (event.kind == SyncKind::Atomic)
    return sequence.kind == SyncKind::Atomic;
  return event.kind == SyncKind::OrdinaryMemory && sequence.kind == SyncKind::OrdinaryMemory;
}

[[nodiscard]] AtomicPolicyReason classify_atomic_semantics(const SyncEvent &event,
                                                           const SyncSequence *sequence,
                                                           bool ambiguous_membership) {
  if (ambiguous_membership)
    return AtomicPolicyReason::AmbiguousSequenceMembership;
  if (sequence == nullptr || !sequence_is_atomic_contract(event, *sequence) ||
      !sync_confidence_meets(sequence->confidence, SemanticConfidence::Conservative) ||
      !sync_confidence_meets(sequence->memory_role_confidence, SemanticConfidence::Conservative)) {
    return AtomicPolicyReason::UnqualifiedSyncSequence;
  }
  switch (sequence->memory_role) {
  case SyncMemoryRole::Release:
  case SyncMemoryRole::Acquire:
  case SyncMemoryRole::AcquireRelease:
    break;
  case SyncMemoryRole::Unknown:
  case SyncMemoryRole::None:
  case SyncMemoryRole::SequentiallyConsistent:
    return AtomicPolicyReason::UnsupportedMemoryRole;
  }
  const std::optional<MemoryScope> semantic_scope = sequence->scope;
  if (!semantic_scope)
    return AtomicPolicyReason::MissingScope;
  if (!memory_scope_is_supported(*semantic_scope) || *semantic_scope == MemoryScope::Wavefront)
    return AtomicPolicyReason::UnsupportedScope;
  if (event.kind == SyncKind::Atomic &&
      dynamic_requirement(event) == DynamicResultRequirement::Count) {
    return AtomicPolicyReason::UnsupportedDynamicOutcome;
  }
  return AtomicPolicyReason::None;
}

ProbeIntentId add_intent(ObservationPlan &plan, ProgramSiteId source_site,
                         const PhysicalSiteId &physical_site,
                         std::vector<SemanticSiteId> covered_sites, ProbeIntentKind kind,
                         ProbePosition position, const SynchronizationAssociationId &association,
                         DynamicResultRequirement dynamic_result,
                         std::optional<AtomicLoweringForm> lowering_form = std::nullopt) {
  const ProbeIntentId id{static_cast<uint32_t>(plan.probe_intents.size())};
  plan.probe_intents.push_back({
      .id = id,
      .mode = plan.mode,
      .source_site = source_site,
      .physical_site = physical_site,
      .covered_semantic_sites = std::move(covered_sites),
      .kind = kind,
      .position = position,
      .synchronization_association = association,
      .dynamic_result = dynamic_result,
      .atomic_lowering_form = std::move(lowering_form),
  });
  return id;
}

void add_covered_site(ObservationPlan &plan, ProbeIntentId id, const SemanticSiteId &site) {
  if (id.value >= plan.probe_intents.size())
    return;
  std::vector<SemanticSiteId> &covered = plan.probe_intents[id.value].covered_semantic_sites;
  if (std::ranges::find(covered, site) == covered.end())
    covered.push_back(site);
}

[[nodiscard]] bool intent_covers(const ProbeIntent &intent, const SemanticSiteId &site) {
  return std::ranges::find(intent.covered_semantic_sites, site) !=
         intent.covered_semantic_sites.end();
}

[[nodiscard]] std::optional<ProbeIntentId>
find_intent(const ObservationPlan &plan, const SemanticSiteId &site,
            const SynchronizationAssociationId &association, ProbeIntentKind kind) {
  const auto found = std::ranges::find_if(plan.probe_intents, [&](const ProbeIntent &intent) {
    return intent.kind == kind && intent.synchronization_association == association &&
           intent_covers(intent, site);
  });
  return found == plan.probe_intents.end() ? std::nullopt : std::optional{found->id};
}

} // namespace

AtomicFencePolicyResult plan_atomic_fence_observation(const ProgramInventory &inventory,
                                                      const AtomicFencePolicyRequest &request) {
  AtomicFencePolicyResult result;
  result.plan.mode = request.mode;
  const ModeProbeVocabulary *vocabulary = mode_probe_vocabulary(request.mode);
  if (vocabulary == nullptr || inventory.empty())
    return result;

  const SynchronizationInventoryView synchronization = inventory.sync();
  std::map<std::pair<SyncKind, uint64_t>, std::vector<const SyncEvent *>> aliases_by_site;
  for (const SyncEvent &event : synchronization.sync_events) {
    if (event.kind == SyncKind::Atomic || event.kind == SyncKind::OrdinaryMemory) {
      aliases_by_site[{event.kind, event.text_offset()}].push_back(&event);
    }
  }

  for (const auto &[site_key, aliases] : aliases_by_site) {
    (void)site_key;
    const SyncEvent &event = *aliases.front();
    const SyncEventId event_id = synchronization.event_id(event);
    const SyncSequenceMembership membership = synchronization.sequence_membership(event_id);
    const SyncSequence *sequence = synchronization.find_unique_sequence_containing(event_id);
    const bool ambiguous_membership = membership.ambiguous;
    // Ordinary memory belongs to access policy unless synchronization
    // analysis associated an acquire/release sequence around it.
    if (event.kind == SyncKind::OrdinaryMemory &&
        (sequence == nullptr || sequence->kind != SyncKind::OrdinaryMemory ||
         sequence->memory_role == SyncMemoryRole::Unknown ||
         sequence->memory_role == SyncMemoryRole::None)) {
      continue;
    }

    const SemanticSiteId semantic_id = event_semantic_id(inventory, event);
    const std::vector<std::string> names =
        synchronization.source_container_names(event.semantic_id.physical);
    const bool conflicting_alias = std::ranges::any_of(aliases, [&](const SyncEvent *alias) {
      return !event_policy_semantics_equal(synchronization, event, *alias);
    });
    const std::optional<CapabilityForm> form = atomic_capability_form(synchronization, event);
    const CapabilityDisposition capability =
        form ? capability_disposition(inventory.target(), request.mode, *form)
             : CapabilityDisposition::OutOfContract;
    const std::vector<uint64_t> owner_descriptors =
        synchronization.execution_owner_descriptors(aliases);
    SiteDecisionKind kind = SiteDecisionKind::NotApplicable;
    AtomicPolicyReason reason = AtomicPolicyReason::TrackingDisabled;
    std::optional<AtomicLoweringForm> lowering_form;

    if (!request.tracking_enabled) {
      reason = AtomicPolicyReason::TrackingDisabled;
    } else if (request.mode == Mode::SuperCollider) {
      reason = AtomicPolicyReason::ModeMutationOnly;
    } else if (!filter_matches(names, request.container_filter) ||
               !site_matches_kernel_allowlist(inventory, owner_descriptors, names,
                                              request.kernel_name_allowlist)) {
      reason = AtomicPolicyReason::ContainerFilterExcluded;
    } else if (synchronization.execution_owners(event).empty()) {
      reason = AtomicPolicyReason::MissingExecutionOwner;
    } else if (vocabulary->ordering_requires_directional_access_window && sequence != nullptr &&
               !directional_access_window_available(synchronization.execution_owners(event),
                                                    sequence->memory_role, request)) {
      reason = AtomicPolicyReason::MissingDirectionalAccessWindow;
    } else if (!form || (capability != CapabilityDisposition::Supported &&
                         capability != CapabilityDisposition::AssociatedOnly)) {
      reason = AtomicPolicyReason::TargetCapabilityUnavailable;
    } else if (conflicting_alias) {
      kind = SiteDecisionKind::Unsupported;
      reason = AtomicPolicyReason::ConflictingPhysicalAliases;
      result.atomic_errors.push_back(reason);
    } else {
      reason = classify_atomic_semantics(event, sequence, ambiguous_membership);
      if (reason == AtomicPolicyReason::None) {
        AtomicEncodingDecision encoding =
            classify_atomic_encoding(inventory, event, *sequence, request.mode);
        reason = encoding.reason;
        lowering_form = std::move(encoding.form);
      }
      const bool semantic_not_applicable =
          reason == AtomicPolicyReason::UnqualifiedSyncSequence ||
          (reason == AtomicPolicyReason::UnsupportedMemoryRole && sequence &&
           (sequence->memory_role == SyncMemoryRole::Unknown ||
            sequence->memory_role == SyncMemoryRole::None)) ||
          (reason == AtomicPolicyReason::UnsupportedScope && sequence &&
           sequence->scope == MemoryScope::Wavefront);
      kind = reason == AtomicPolicyReason::None ? SiteDecisionKind::Admitted
             : semantic_not_applicable          ? SiteDecisionKind::NotApplicable
                                                : SiteDecisionKind::Unsupported;
    }

    DynamicResultRequirement required_dynamic_result = event.kind == SyncKind::Atomic
                                                           ? dynamic_requirement(event)
                                                           : DynamicResultRequirement::None;
    if (required_dynamic_result == DynamicResultRequirement::Count)
      required_dynamic_result = DynamicResultRequirement::None;
    const std::optional<SynchronizationAssociationId> association =
        sequence == nullptr ? std::nullopt
                            : std::optional{SynchronizationAssociationId{sequence->identity}};
    AtomicSiteDecision decision{
        .semantic_site = semantic_id,
        .kind = kind,
        .capability = capability,
        .reason = reason,
    };

    if (decision.kind == SiteDecisionKind::Admitted && association) {
      add_intent(result.plan, event.source_site, semantic_id.physical, {semantic_id},
                 ProbeIntentKind::AtomicAddressCapture, ProbePosition::Before, *association,
                 DynamicResultRequirement::None, lowering_form);

      add_intent(result.plan, event.source_site, semantic_id.physical, {semantic_id},
                 vocabulary->atomic, ProbePosition::After, *association, required_dynamic_result);
    }
    result.plan.atomic_site_decisions.push_back(std::move(decision));
  }

  std::map<uint64_t, std::vector<const FenceCandidate *>> fence_aliases_by_site;
  for (const FenceCandidate &fence : synchronization.fence_candidates) {
    if (const SyncEvent *event = synchronization.find_event(fence.fence_event))
      fence_aliases_by_site[event->text_offset()].push_back(&fence);
  }

  for (const auto &fence_aliases : fence_aliases_by_site) {
    const auto &aliases = fence_aliases.second;
    const FenceCandidate &fence = *aliases.front();
    const SyncEvent *fence_event = synchronization.find_event(fence.fence_event);
    std::vector<const SyncEvent *> fence_events;
    fence_events.reserve(aliases.size());
    for (const FenceCandidate *alias : aliases) {
      if (const SyncEvent *event = synchronization.find_event(alias->fence_event)) {
        fence_events.push_back(event);
      }
    }
    const SemanticSiteId fence_id =
        fence_event == nullptr ? SemanticSiteId{} : fence_event->semantic_id;
    const std::vector<std::string> names =
        fence_event == nullptr
            ? std::vector<std::string>{}
            : synchronization.source_container_names(fence_event->semantic_id.physical);
    const CapabilityDisposition capability = capability_disposition(
        inventory.target(), request.mode, CapabilityForm::AddressedOrdinaryFence);
    const bool conflicting_alias = std::ranges::any_of(aliases, [&](const FenceCandidate *alias) {
      return !fence_policy_semantics_equal(fence, *alias);
    });
    FenceSiteDecision decision{
        .semantic_site = fence_id,
        .kind = SiteDecisionKind::NotApplicable,
        .capability = capability,
        .reason = FencePolicyReason::TrackingDisabled,
        .inventory_association = fence.association,
    };
    const SyncSequence *fence_sequence = synchronization.find_sequence(fence.sequence);
    const std::optional<SynchronizationAssociationId> association =
        fence_sequence == nullptr
            ? std::nullopt
            : std::optional{SynchronizationAssociationId{fence_sequence->identity}};
    const std::vector<uint64_t> owner_descriptors =
        synchronization.execution_owner_descriptors(fence_events);

    if (!request.tracking_enabled) {
      decision.reason = FencePolicyReason::TrackingDisabled;
    } else if (request.mode == Mode::SuperCollider) {
      decision.reason = FencePolicyReason::ModeMutationOnly;
    } else if (!filter_matches(names, request.container_filter) ||
               !site_matches_kernel_allowlist(inventory, owner_descriptors, names,
                                              request.kernel_name_allowlist)) {
      decision.reason = FencePolicyReason::ContainerFilterExcluded;
    } else if (conflicting_alias) {
      decision.kind = SiteDecisionKind::Unsupported;
      decision.reason = FencePolicyReason::ConflictingPhysicalAliases;
      result.fence_errors.push_back(decision.reason);
    } else if (!fence.eligible()) {
      decision.reason = FencePolicyReason::AssociationUnavailable;
    } else if (capability != CapabilityDisposition::Supported &&
               capability != CapabilityDisposition::AssociatedOnly) {
      decision.reason = FencePolicyReason::TargetCapabilityUnavailable;
    } else {
      const SyncEvent *communication = fence.communication_event
                                           ? synchronization.find_event(*fence.communication_event)
                                           : nullptr;
      const auto communication_decision =
          communication == nullptr
              ? result.plan.atomic_site_decisions.end()
              : std::ranges::find_if(result.plan.atomic_site_decisions,
                                     [&](const AtomicSiteDecision &candidate) {
                                       return candidate.semantic_site.physical ==
                                              event_semantic_id(inventory, *communication).physical;
                                     });
      const bool communication_admitted =
          communication_decision != result.plan.atomic_site_decisions.end() &&
          communication_decision->kind == SiteDecisionKind::Admitted && association &&
          find_intent(result.plan, communication_decision->semantic_site, *association,
                      ProbeIntentKind::AtomicAddressCapture);
      if (fence_event != nullptr && communication != nullptr &&
          (synchronization.execution_owners(*fence_event).empty() ||
           synchronization.execution_owners(*communication).empty())) {
        decision.reason = FencePolicyReason::MissingExecutionOwner;
      } else if (communication_decision != result.plan.atomic_site_decisions.end() &&
                 communication_decision->kind == SiteDecisionKind::NotApplicable &&
                 communication_decision->reason ==
                     AtomicPolicyReason::MissingDirectionalAccessWindow) {
        decision.reason = FencePolicyReason::CommunicationNotApplicable;
      } else if (fence_event == nullptr || communication == nullptr || !association ||
                 !communication_admitted) {
        decision.kind = SiteDecisionKind::Unsupported;
        decision.reason = FencePolicyReason::MissingCommunicationEvent;
      } else {
        decision.kind = SiteDecisionKind::Admitted;
        decision.reason = FencePolicyReason::None;

        std::vector<ProbeIntentId> associated_intents;
        for (const ProbeIntent &intent : result.plan.probe_intents) {
          if (intent.synchronization_association == association &&
              intent_covers(intent, communication_decision->semantic_site)) {
            associated_intents.push_back(intent.id);
          }
        }
        for (ProbeIntentId id : associated_intents)
          add_covered_site(result.plan, id, fence_id);
      }
    }
    result.plan.fence_site_decisions.push_back(std::move(decision));
  }

  if (request.publication_modifications_enabled && request.tracking_enabled &&
      request.mode == Mode::Default && inventory.arch() == ROCJITSU_CODE_ARCH_RDNA4) {
    std::vector<PhysicalSiteId> covered;
    for (const ProgramSite &source : inventory.program_sites()) {
      const auto *atomic = source.get_if<AtomicSite>();
      const auto *ordinary = source.get_if<OrdinaryMemorySite>();
      if (!atomic && (!ordinary || ordinary->operation != OrdinaryMemoryOperation::Store))
        continue;
      // LDS has a distinct address space from the global publication objects.
      if (source.mnemonic_view().starts_with("ds_"))
        continue;
      const auto names = inventory.source_container_names(source.physical_id);
      const auto owners = inventory.execution_owner_descriptors(source);
      if (owners.empty() || !filter_matches(names, request.container_filter) ||
          !site_matches_kernel_allowlist(inventory, owners, names, request.kernel_name_allowlist) ||
          std::ranges::find(covered, source.physical_id) != covered.end() ||
          std::ranges::any_of(result.plan.probe_intents, [&](const ProbeIntent &intent) {
            return intent.physical_site == source.physical_id &&
                   intent.kind == ProbeIntentKind::AtomicAddressCapture;
          }))
        continue;
      const auto site = atomic ? *atomic : atomic_communication_site(*ordinary);
      const auto classification =
          classify_atomic_lowering(site, inventory.arch(), atomic != nullptr);
      if (!classification.address_available())
        continue; // Completeness is deliberately not asserted by this capture path.
      covered.push_back(source.physical_id);
      const SemanticSiteId semantic{.physical = source.physical_id,
                                    .domain = SemanticSiteDomain::Access};
      for (const auto kind :
           {ProbeIntentKind::PublicationAddressCapture, ProbeIntentKind::PublicationModification}) {
        const bool capture = kind == ProbeIntentKind::PublicationAddressCapture;
        result.plan.probe_intents.push_back({
            .id = ProbeIntentId{static_cast<uint32_t>(result.plan.probe_intents.size())},
            .mode = request.mode,
            .source_site = source.id,
            .physical_site = source.physical_id,
            .covered_semantic_sites = {semantic},
            .kind = kind,
            .position = capture ? ProbePosition::Before : ProbePosition::After,
            .synchronization_association = std::nullopt,
            .dynamic_result = DynamicResultRequirement::None,
            .atomic_lowering_form = capture ? classification.form : std::nullopt,
        });
      }
    }
  }

  return result;
}

} // namespace rocjitsu::consan
