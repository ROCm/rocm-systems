// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"

#include <algorithm>
#include <map>
#include <ranges>
#include <tuple>
#include <unordered_map>

namespace rocjitsu {
namespace {

/// Unique logical sequence claiming one synchronization-event identity.
///
/// The pointer borrows immutable inventory for the duration of a pure policy
/// call. `ambiguous` is distinct from a missing pointer so policy can preserve
/// whether no sequence or multiple incompatible sequences caused rejection.
struct SequenceMembership {
  const ConSanSyncSequence *sequence = nullptr;
  bool ambiguous = false;
};

/// Read-only join from stable event identity to its sequence-membership fact.
using SequenceMembershipIndex = std::unordered_map<std::string_view, SequenceMembership>;

[[nodiscard]] bool valid_engine(ConSanCapabilityEngine engine) {
  return static_cast<uint8_t>(engine) < static_cast<uint8_t>(ConSanCapabilityEngine::Count);
}

[[nodiscard]] SequenceMembershipIndex
build_sequence_membership_index(std::span<const ConSanSyncSequence> sequences) {
  SequenceMembershipIndex result;
  size_t member_count = 0;
  for (const ConSanSyncSequence &sequence : sequences)
    member_count += sequence.member_event_identities.size();
  result.reserve(member_count);
  for (const ConSanSyncSequence &sequence : sequences) {
    for (const std::string &identity : sequence.member_event_identities) {
      const auto [entry, inserted] =
          result.try_emplace(identity, SequenceMembership{&sequence, false});
      if (!inserted) {
        entry->second.sequence = nullptr;
        entry->second.ambiguous = true;
      }
    }
  }
  return result;
}

[[nodiscard]] std::unordered_map<std::string_view, const ConSanSyncEvent *>
build_event_index(std::span<const ConSanSyncEvent> events) {
  std::unordered_map<std::string_view, const ConSanSyncEvent *> result;
  result.reserve(events.size());
  for (const ConSanSyncEvent &event : events) {
    const auto [entry, inserted] = result.emplace(event.identity, &event);
    if (!inserted)
      entry->second = nullptr;
  }
  return result;
}

[[nodiscard]] bool owner_semantics_equal(std::span<const ConSanExecutionOwner> lhs,
                                         std::span<const ConSanExecutionOwner> rhs) {
  return std::ranges::equal(lhs, rhs, [](const auto &left, const auto &right) {
    return left.descriptor_file_offset == right.descriptor_file_offset && left.proof == right.proof;
  });
}

[[nodiscard]] bool event_policy_semantics_equal(const ConSanSyncEvent &lhs,
                                                const ConSanSyncEvent &rhs) {
  return std::tie(lhs.kind, lhs.operation, lhs.address_source, lhs.memory_role, lhs.rmw_outcome,
                  lhs.confidence, lhs.memory_role_confidence, lhs.text_offset, lhs.file_offset,
                  lhs.size, lhs.width_bits, lhs.cache_operation,
                  lhs.ordinary_acquire_mutation_supported, lhs.mnemonic, lhs.static_byte_offset,
                  lhs.raw_scope) ==
             std::tie(rhs.kind, rhs.operation, rhs.address_source, rhs.memory_role, rhs.rmw_outcome,
                      rhs.confidence, rhs.memory_role_confidence, rhs.text_offset, rhs.file_offset,
                      rhs.size, rhs.width_bits, rhs.cache_operation,
                      rhs.ordinary_acquire_mutation_supported, rhs.mnemonic, rhs.static_byte_offset,
                      rhs.raw_scope) &&
         owner_semantics_equal(lhs.execution_owners, rhs.execution_owners);
}

[[nodiscard]] bool fence_policy_semantics_equal(const ConSanMoiFenceCandidate &lhs,
                                                const ConSanMoiFenceCandidate &rhs) {
  return lhs.fence_event == rhs.fence_event && lhs.sequence_identity == rhs.sequence_identity &&
         lhs.communication_event == rhs.communication_event && lhs.memory_role == rhs.memory_role &&
         lhs.association == rhs.association;
}

[[nodiscard]] std::vector<std::string>
source_container_names(std::span<const ConSanSyncEvent *const> aliases) {
  std::vector<std::string> result;
  result.reserve(aliases.size());
  for (const ConSanSyncEvent *event : aliases) {
    const std::span<const std::string> sources = event->source_containers.empty()
                                                     ? std::span(&event->container_name, 1u)
                                                     : std::span(event->source_containers);
    for (const std::string &source : sources) {
      if (std::ranges::find(result, source) == result.end())
        result.push_back(source);
    }
  }
  std::ranges::sort(result);
  return result;
}

[[nodiscard]] bool filter_matches(std::span<const ConSanSyncEvent *const> aliases,
                                  std::string_view filter) {
  return filter.empty() || std::ranges::any_of(aliases, [&](const ConSanSyncEvent *event) {
           if (event->container_name.find(filter) != std::string::npos)
             return true;
           return std::ranges::any_of(event->source_containers, [&](const std::string &source) {
             return source.find(filter) != std::string::npos;
           });
         });
}

[[nodiscard]] SemanticSiteId event_semantic_id(const ProgramInventory &inventory,
                                               const ConSanSyncEvent &event) {
  if (event.semantic_id.valid())
    return event.semantic_id;
  return {
      .physical = {.code_object = inventory.code_object_id(),
                   .original_text_offset = event.text_offset},
      .domain = ConSanSemanticSiteDomain::SynchronizationEvent,
  };
}

template <typename Container>
[[nodiscard]] const ConSanAtomicSite *atomic_site_in_container(const Container &container,
                                                               uint64_t text_offset) {
  const auto site =
      std::ranges::find(container.atomic_sites, text_offset, &ConSanAtomicSite::text_offset);
  if (site == container.atomic_sites.end() ||
      std::ranges::count(container.atomic_sites, text_offset, &ConSanAtomicSite::text_offset) !=
          1) {
    return nullptr;
  }
  return &*site;
}

[[nodiscard]] const ConSanAtomicSite *find_atomic_site(const ProgramInventory &inventory,
                                                       const ConSanSyncEvent &event) {
  if (event.in_kernel) {
    const ConSanKernelInfo *container = inventory.find_kernel_by_name(event.container_name);
    if (container != nullptr)
      return atomic_site_in_container(*container, event.text_offset);
  } else {
    const ConSanFunctionInfo *container = inventory.find_function_by_name(event.container_name);
    if (container != nullptr)
      return atomic_site_in_container(*container, event.text_offset);
  }
  return nullptr;
}

template <typename Container>
[[nodiscard]] const ConSanOrdinaryMemorySite *ordinary_site_in_container(const Container &container,
                                                                         uint64_t text_offset) {
  const auto site = std::ranges::find(container.ordinary_memory_sites, text_offset,
                                      &ConSanOrdinaryMemorySite::text_offset);
  if (site == container.ordinary_memory_sites.end() ||
      std::ranges::count(container.ordinary_memory_sites, text_offset,
                         &ConSanOrdinaryMemorySite::text_offset) != 1) {
    return nullptr;
  }
  return &*site;
}

[[nodiscard]] const ConSanOrdinaryMemorySite *find_ordinary_site(const ProgramInventory &inventory,
                                                                 const ConSanSyncEvent &event) {
  if (event.in_kernel) {
    const ConSanKernelInfo *container = inventory.find_kernel_by_name(event.container_name);
    if (container != nullptr)
      return ordinary_site_in_container(*container, event.text_offset);
  } else {
    const ConSanFunctionInfo *container = inventory.find_function_by_name(event.container_name);
    if (container != nullptr)
      return ordinary_site_in_container(*container, event.text_offset);
  }
  return nullptr;
}

[[nodiscard]] ConSanAtomicSite normalize_ordinary_site(const ConSanOrdinaryMemorySite &site) {
  ConSanAtomicSite result;
  if (site.flat_address_space_hint == ConSanFlatAddressSpaceHint::Group)
    result.address_space_hint = ConSanAtomicAddressSpaceHint::FlatGroup;
  result.text_offset = site.text_offset;
  result.file_offset = site.file_offset;
  result.size = site.size;
  result.width_bits = site.width_bits;
  result.dst_vgpr = site.destination_vgpr;
  result.addr_vgpr = site.address_vgpr;
  result.data_vgpr = site.operation == ConSanOrdinaryMemoryOperation::Load ? site.destination_vgpr
                                                                           : site.value_vgpr;
  result.saddr_sgpr = site.address_sgpr;
  result.raw_saddr = site.raw_saddr;
  result.raw_scale_offset = site.raw_scale_offset;
  result.raw_vaddr = site.raw_vaddr;
  result.raw_vsrc = site.raw_vsrc;
  result.raw_vdst = site.raw_vdst;
  result.raw_vdata = site.operation == ConSanOrdinaryMemoryOperation::Load
                         ? std::optional<uint32_t>(site.destination_vgpr)
                         : std::optional<uint32_t>(site.value_vgpr);
  result.raw_rsrc = site.raw_rsrc;
  result.raw_soffset = site.raw_soffset;
  result.raw_offen = site.raw_offen;
  result.raw_idxen = site.raw_idxen;
  result.raw_ioffset = site.raw_ioffset;
  result.raw_scope = site.raw_scope;
  result.raw_th = site.raw_th;
  result.returns_old_value = false;
  result.mnemonic = site.mnemonic;
  return result;
}

[[nodiscard]] std::optional<ConSanCapabilityForm>
atomic_capability_form(const ConSanSyncEvent &event) {
  if (event.kind == ConSanSyncEventKind::OrdinaryMemory)
    return ConSanCapabilityForm::AddressedOrdinaryFence;
  if (event.kind != ConSanSyncEventKind::Atomic)
    return std::nullopt;
  if (event.address_source == ConSanSyncAddressSource::LdsVector ||
      event.mnemonic.starts_with("ds_")) {
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
  const bool ordinary = event.kind == ConSanSyncEventKind::OrdinaryMemory;
  const ConSanAtomicSite *native = ordinary ? nullptr : find_atomic_site(inventory, event);
  const ConSanOrdinaryMemorySite *ordinary_site =
      ordinary ? find_ordinary_site(inventory, event) : nullptr;
  if ((ordinary && ordinary_site == nullptr) || (!ordinary && native == nullptr))
    return {.reason = ConSanAtomicPolicyReason::MissingOperands, .form = std::nullopt};
  ConSanAtomicSite site = ordinary ? normalize_ordinary_site(*ordinary_site) : *native;

  // Synchronization analysis owns normalized semantic scope. It may preserve
  // an encoded value or derive a stronger fact from address-space provenance
  // and sequence association; policy always classifies that single contract.
  site.raw_scope = sequence.raw_scope;

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
  if (event.kind == ConSanSyncEventKind::Atomic)
    return sequence.kind == ConSanSyncSequenceKind::Atomic;
  return event.kind == ConSanSyncEventKind::OrdinaryMemory &&
         sequence.kind == ConSanSyncSequenceKind::OrdinaryMemory;
}

[[nodiscard]] ConSanAtomicPolicyReason
classify_atomic_semantics(const ConSanSyncEvent &event, const SequenceMembership &membership) {
  if (membership.ambiguous)
    return ConSanAtomicPolicyReason::AmbiguousSequenceMembership;
  const ConSanSyncSequence *sequence = membership.sequence;
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
  const std::optional<uint32_t> semantic_scope = sequence->raw_scope;
  if (!semantic_scope)
    return ConSanAtomicPolicyReason::MissingScope;
  if (*semantic_scope < 1u || *semantic_scope > 3u)
    return ConSanAtomicPolicyReason::UnsupportedScope;
  if (sequence->width_bits == 0u || sequence->width_bits % 8u != 0u)
    return ConSanAtomicPolicyReason::InvalidAccessWidth;
  if (event.kind == ConSanSyncEventKind::Atomic &&
      dynamic_requirement(event) == ConSanDynamicResultRequirement::Count) {
    return ConSanAtomicPolicyReason::UnsupportedDynamicOutcome;
  }
  return ConSanAtomicPolicyReason::None;
}

[[nodiscard]] ConSanProbeIntentKind atomic_evidence_kind(ConSanCapabilityEngine engine) {
  switch (engine) {
  case ConSanCapabilityEngine::RecordReplay:
    return ConSanProbeIntentKind::AtomicRecord;
  case ConSanCapabilityEngine::Sampled:
    return ConSanProbeIntentKind::SampledAtomicOrdering;
  case ConSanCapabilityEngine::InlineShadow:
    return ConSanProbeIntentKind::ExactAtomicOrdering;
  case ConSanCapabilityEngine::SuperCollider:
  case ConSanCapabilityEngine::Count:
    break;
  }
  return ConSanProbeIntentKind::Count;
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
                                           const SemanticSiteId &communication_identity) {
  return std::ranges::any_of(
      inventory.moi_fence_candidates, [&](const ConSanMoiFenceCandidate &fence) {
        return fence.eligible() && fence.communication_event == communication_identity;
      });
}

} // namespace

ConSanAtomicFencePolicyResult
plan_consan_atomic_fence_observation(const ProgramInventory &inventory,
                                     const ConSanAtomicFencePolicyRequest &request) {
  ConSanAtomicFencePolicyResult result;
  result.plan.engine = request.engine;
  if (!valid_engine(request.engine) || inventory.empty())
    return result;

  const SynchronizationInventoryView synchronization = inventory.sync();
  const SequenceMembershipIndex memberships =
      build_sequence_membership_index(synchronization.sync_sequences);
  const auto events_by_identity = build_event_index(synchronization.sync_events);

  std::map<std::pair<ConSanSyncEventKind, uint64_t>, std::vector<const ConSanSyncEvent *>>
      aliases_by_site;
  for (const ConSanSyncEvent &event : synchronization.sync_events) {
    if (event.kind == ConSanSyncEventKind::Atomic ||
        event.kind == ConSanSyncEventKind::OrdinaryMemory) {
      aliases_by_site[{event.kind, event.text_offset}].push_back(&event);
    }
  }

  for (const auto &[site_key, aliases] : aliases_by_site) {
    (void)site_key;
    const ConSanSyncEvent &event = *aliases.front();
    const auto membership_entry = memberships.find(event.identity);
    const SequenceMembership membership =
        membership_entry == memberships.end() ? SequenceMembership{} : membership_entry->second;
    // Ordinary memory belongs to access policy unless synchronization
    // analysis associated an acquire/release sequence around it.
    if (event.kind == ConSanSyncEventKind::OrdinaryMemory &&
        (membership.sequence == nullptr ||
         membership.sequence->kind != ConSanSyncSequenceKind::OrdinaryMemory ||
         membership.sequence->memory_role == ConSanSyncMemoryRole::Unknown ||
         membership.sequence->memory_role == ConSanSyncMemoryRole::None)) {
      continue;
    }

    const SemanticSiteId semantic_id = event_semantic_id(inventory, event);
    const std::vector<std::string> names = source_container_names(aliases);
    const bool conflicting_alias = std::ranges::any_of(aliases, [&](const ConSanSyncEvent *alias) {
      return !event_policy_semantics_equal(event, *alias);
    });
    const std::optional<ConSanCapabilityForm> form = atomic_capability_form(event);
    const ConSanCapabilityDisposition capability =
        form ? consan_capability_disposition(inventory.target(), request.engine, *form)
             : ConSanCapabilityDisposition::OutOfContract;
    std::vector<uint64_t> owner_descriptors;
    for (const ConSanSyncEvent *alias : aliases) {
      for (const ConSanExecutionOwner &owner : alias->execution_owners) {
        if (std::ranges::find(owner_descriptors, owner.descriptor_file_offset) ==
            owner_descriptors.end()) {
          owner_descriptors.push_back(owner.descriptor_file_offset);
        }
      }
    }
    ConSanSiteDecisionKind kind = ConSanSiteDecisionKind::NotApplicable;
    ConSanAtomicPolicyReason reason = ConSanAtomicPolicyReason::TrackingDisabled;
    std::optional<ConSanAtomicLoweringForm> lowering_form;

    if (!request.tracking_enabled) {
      reason = ConSanAtomicPolicyReason::TrackingDisabled;
    } else if (request.engine == ConSanCapabilityEngine::SuperCollider) {
      reason = ConSanAtomicPolicyReason::EngineMutationOnly;
    } else if (!filter_matches(aliases, request.container_filter) ||
               !consan_site_matches_kernel_allowlist(inventory, owner_descriptors, names,
                                                     request.kernel_name_allowlist)) {
      reason = ConSanAtomicPolicyReason::ContainerFilterExcluded;
    } else if (event.execution_owners.empty()) {
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
      reason = classify_atomic_semantics(event, membership);
      if (reason == ConSanAtomicPolicyReason::None) {
        AtomicEncodingDecision encoding =
            classify_atomic_encoding(inventory, event, *membership.sequence, request.engine);
        reason = encoding.reason;
        lowering_form = std::move(encoding.form);
      }
      const bool semantic_not_applicable =
          reason == ConSanAtomicPolicyReason::UnqualifiedSyncSequence ||
          (reason == ConSanAtomicPolicyReason::UnsupportedMemoryRole && membership.sequence &&
           (membership.sequence->memory_role == ConSanSyncMemoryRole::Unknown ||
            membership.sequence->memory_role == ConSanSyncMemoryRole::None)) ||
          (reason == ConSanAtomicPolicyReason::UnsupportedScope && membership.sequence &&
           membership.sequence->raw_scope == 0u);
      kind = reason == ConSanAtomicPolicyReason::None ? ConSanSiteDecisionKind::Admitted
             : semantic_not_applicable                ? ConSanSiteDecisionKind::NotApplicable
                                                      : ConSanSiteDecisionKind::Unsupported;
    }

    ConSanDynamicResultRequirement required_dynamic_result =
        event.kind == ConSanSyncEventKind::Atomic ? dynamic_requirement(event)
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
        .source_containers = names,
    };
    if (membership.sequence != nullptr)
      decision.association = ConSanSynchronizationAssociationId{membership.sequence->identity};

    if (decision.kind == ConSanSiteDecisionKind::Admitted && decision.association) {
      const ConSanProbeIntentId capture =
          add_intent(result.plan, semantic_id.physical, {semantic_id},
                     ConSanProbeIntentKind::AtomicAddressCapture, ConSanProbePosition::Before,
                     *decision.association, ConSanDynamicResultRequirement::None);
      decision.intent_ids.push_back(capture);
      const bool record_replay_fence_owns_ordinary =
          request.engine == ConSanCapabilityEngine::RecordReplay &&
          event.kind == ConSanSyncEventKind::OrdinaryMemory &&
          has_qualified_fence_for(synchronization, event.semantic_id);
      if (!record_replay_fence_owns_ordinary) {
        decision.intent_ids.push_back(add_intent(
            result.plan, semantic_id.physical, {semantic_id}, atomic_evidence_kind(request.engine),
            ConSanProbePosition::After, *decision.association, decision.dynamic_result));
      }
    }
    result.plan.atomic_site_decisions.push_back(std::move(decision));
  }

  std::map<uint64_t, std::vector<const ConSanMoiFenceCandidate *>> fence_aliases_by_site;
  for (const ConSanMoiFenceCandidate &fence : synchronization.moi_fence_candidates)
    fence_aliases_by_site[fence.fence_event.physical.original_text_offset].push_back(&fence);

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
    const SemanticSiteId fence_id = fence.fence_event;
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
        .source_containers = source_container_names(fence_events),
    };
    if (!fence.sequence_identity.empty())
      decision.association = ConSanSynchronizationAssociationId{fence.sequence_identity};
    std::vector<uint64_t> owner_descriptors;
    for (const ConSanSyncEvent *event : fence_events) {
      for (const ConSanExecutionOwner &owner : event->execution_owners) {
        if (std::ranges::find(owner_descriptors, owner.descriptor_file_offset) ==
            owner_descriptors.end()) {
          owner_descriptors.push_back(owner.descriptor_file_offset);
        }
      }
    }

    if (!request.tracking_enabled) {
      decision.reason = ConSanFencePolicyReason::TrackingDisabled;
    } else if (request.engine == ConSanCapabilityEngine::SuperCollider) {
      decision.reason = ConSanFencePolicyReason::EngineMutationOnly;
    } else if (!filter_matches(fence_events, request.container_filter) ||
               !consan_site_matches_kernel_allowlist(inventory, owner_descriptors,
                                                     decision.source_containers,
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
          (fence_event->execution_owners.empty() || communication->execution_owners.empty())) {
        decision.reason = ConSanFencePolicyReason::MissingExecutionOwner;
      } else if (fence_event == nullptr || communication == nullptr ||
                 atomic_decision == result.plan.atomic_site_decisions.end()) {
        decision.kind = ConSanSiteDecisionKind::Unsupported;
        decision.reason = ConSanFencePolicyReason::MissingCommunicationEvent;
      } else {
        decision.kind = ConSanSiteDecisionKind::Admitted;
        decision.reason = ConSanFencePolicyReason::None;
        decision.communication_lowering_form = atomic_decision->lowering_form;
        if (request.engine == ConSanCapabilityEngine::RecordReplay) {
          if (!atomic_decision->intent_ids.empty()) {
            const ConSanProbeIntentId capture = atomic_decision->intent_ids.front();
            add_covered_site(result.plan, capture, fence_id);
            decision.intent_ids.push_back(capture);
          }
          const ConSanProbeIntentId record =
              add_intent(result.plan, fence_id.physical, {atomic_decision->semantic_site, fence_id},
                         ConSanProbeIntentKind::FenceRecord, ConSanProbePosition::After,
                         *decision.association, ConSanDynamicResultRequirement::None);
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

  // A qualified ordinary Record/Replay sequence normally delegates its after
  // evidence to a fence. If a corrupt inventory claimed a qualified candidate
  // but no usable fence decision survived, fall back to the direct atomic
  // record rather than publishing a before-only admitted contract.
  if (request.engine == ConSanCapabilityEngine::RecordReplay) {
    for (ConSanAtomicSiteDecision &decision : result.plan.atomic_site_decisions) {
      if (decision.kind != ConSanSiteDecisionKind::Admitted || !decision.association ||
          decision.intent_ids.size() != 1u)
        continue;
      decision.intent_ids.push_back(
          add_intent(result.plan, decision.semantic_site.physical, {decision.semantic_site},
                     ConSanProbeIntentKind::AtomicRecord, ConSanProbePosition::After,
                     *decision.association, decision.dynamic_result));
    }
  }

  return result;
}

} // namespace rocjitsu
