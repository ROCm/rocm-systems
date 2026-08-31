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

[[nodiscard]] bool valid_decision_kind(ConSanSiteDecisionKind kind) {
  return static_cast<uint8_t>(kind) < static_cast<uint8_t>(ConSanSiteDecisionKind::Count);
}

[[nodiscard]] bool valid_access_reason(ConSanAccessPolicyReason reason) {
  return static_cast<uint8_t>(reason) < static_cast<uint8_t>(ConSanAccessPolicyReason::Count);
}

[[nodiscard]] bool valid_barrier_reason(ConSanBarrierPolicyReason reason) {
  return static_cast<uint8_t>(reason) < static_cast<uint8_t>(ConSanBarrierPolicyReason::Count);
}

[[nodiscard]] bool valid_atomic_reason(ConSanAtomicPolicyReason reason) {
  return static_cast<uint8_t>(reason) < static_cast<uint8_t>(ConSanAtomicPolicyReason::Count);
}

[[nodiscard]] bool valid_fence_reason(ConSanFencePolicyReason reason) {
  return static_cast<uint8_t>(reason) < static_cast<uint8_t>(ConSanFencePolicyReason::Count);
}

[[nodiscard]] bool valid_fence_association(ConSanFenceAssociation association) {
  return static_cast<uint8_t>(association) < static_cast<uint8_t>(ConSanFenceAssociation::Count);
}

[[nodiscard]] bool valid_capability_disposition(ConSanCapabilityDisposition disposition) {
  switch (disposition) {
  case ConSanCapabilityDisposition::OutOfContract:
  case ConSanCapabilityDisposition::NotApplicable:
  case ConSanCapabilityDisposition::Supported:
  case ConSanCapabilityDisposition::MutationOnly:
  case ConSanCapabilityDisposition::AccessOnly:
  case ConSanCapabilityDisposition::AssociatedOnly:
    return true;
  }
  return false;
}

[[nodiscard]] bool valid_intent_kind(ConSanProbeIntentKind kind) {
  return static_cast<uint8_t>(kind) < static_cast<uint8_t>(ConSanProbeIntentKind::Count);
}

[[nodiscard]] bool valid_position(ConSanProbePosition position) {
  return static_cast<uint8_t>(position) < static_cast<uint8_t>(ConSanProbePosition::Count);
}

[[nodiscard]] bool valid_dynamic_result(ConSanDynamicResultRequirement requirement) {
  return static_cast<uint8_t>(requirement) <
         static_cast<uint8_t>(ConSanDynamicResultRequirement::Count);
}

[[nodiscard]] bool atomic_or_fence_intent(ConSanProbeIntentKind kind) {
  switch (kind) {
  case ConSanProbeIntentKind::AtomicAddressCapture:
  case ConSanProbeIntentKind::AtomicRecord:
  case ConSanProbeIntentKind::SampledAtomicOrdering:
  case ConSanProbeIntentKind::ExactAtomicOrdering:
  case ConSanProbeIntentKind::FenceRecord:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] bool valid_lowering_outcome(ConSanLoweringOutcomeKind outcome) {
  return static_cast<uint8_t>(outcome) < static_cast<uint8_t>(ConSanLoweringOutcomeKind::Count);
}

[[nodiscard]] bool valid_resource_rejection_reason(std::optional<ConSanRegisterPlanReason> reason) {
  return !reason || (*reason > ConSanRegisterPlanReason::None &&
                     *reason <= ConSanRegisterPlanReason::DynamicStack);
}

[[nodiscard]] bool contains_physical_site(std::span<const PhysicalSiteId> sites,
                                          const PhysicalSiteId &site) {
  return std::ranges::find(sites, site) != sites.end();
}

[[nodiscard]] bool contains_substring(std::span<const ConSanAccessInventorySite *const> aliases,
                                      std::string_view filter) {
  return filter.empty() || std::ranges::any_of(aliases, [&](const auto *access) {
           return access->container.name.find(filter) != std::string::npos;
         });
}

[[nodiscard]] bool access_alias_semantics_equal(const ConSanAccessInventorySite &lhs,
                                                const ConSanAccessInventorySite &rhs) {
  return std::tie(lhs.container.kind, lhs.container.entry_text_offset, lhs.origin, lhs.kind,
                  lhs.address_space, lhs.provenance, lhs.confidence, lhs.lowering, lhs.file_offset,
                  lhs.instruction_size, lhs.decoded_width_bits, lhs.mnemonic,
                  lhs.flat_address_space_hint, lhs.operands, lhs.ranges, lhs.exclusions) ==
         std::tie(rhs.container.kind, rhs.container.entry_text_offset, rhs.origin, rhs.kind,
                  rhs.address_space, rhs.provenance, rhs.confidence, rhs.lowering, rhs.file_offset,
                  rhs.instruction_size, rhs.decoded_width_bits, rhs.mnemonic,
                  rhs.flat_address_space_hint, rhs.operands, rhs.ranges, rhs.exclusions);
}

[[nodiscard]] ConSanAccessPolicyReason access_classifier_reason(ConSanAccessClassifierReason reason,
                                                                ConSanAccessOrigin origin) {
  switch (reason) {
  case ConSanAccessClassifierReason::None:
    return ConSanAccessPolicyReason::None;
  case ConSanAccessClassifierReason::NonAccessInstruction:
    return ConSanAccessPolicyReason::NonAccessInstruction;
  case ConSanAccessClassifierReason::InvalidInstructionSize:
    return ConSanAccessPolicyReason::InvalidInstructionSize;
  case ConSanAccessClassifierReason::InvalidAccessWidth:
    return ConSanAccessPolicyReason::InvalidAccessWidth;
  case ConSanAccessClassifierReason::MissingAddressOperand:
    return ConSanAccessPolicyReason::MissingAddressOperand;
  case ConSanAccessClassifierReason::RangeEncodingUnavailable:
    return ConSanAccessPolicyReason::RangeEncodingUnavailable;
  case ConSanAccessClassifierReason::InstructionOutOfBounds:
    return ConSanAccessPolicyReason::InstructionOutOfBounds;
  case ConSanAccessClassifierReason::UnsupportedMnemonic:
  case ConSanAccessClassifierReason::MissingResultOperand:
  case ConSanAccessClassifierReason::MissingDataOperand:
  case ConSanAccessClassifierReason::OperandRegisterRange:
    return ConSanAccessPolicyReason::UnsupportedMnemonic;
  case ConSanAccessClassifierReason::UnsupportedEncoding:
    return origin == ConSanAccessOrigin::Flat ? ConSanAccessPolicyReason::UnsupportedFlatEncoding
                                              : ConSanAccessPolicyReason::UnsupportedMnemonic;
  case ConSanAccessClassifierReason::NonzeroImmediateOffset:
    return ConSanAccessPolicyReason::NonzeroFlatOffset;
  case ConSanAccessClassifierReason::ReservedAddressRegister:
    return ConSanAccessPolicyReason::ReservedFlatAddressRegister;
  case ConSanAccessClassifierReason::TargetUnavailable:
    return ConSanAccessPolicyReason::TargetCapabilityUnavailable;
  case ConSanAccessClassifierReason::Count:
    break;
  }
  return ConSanAccessPolicyReason::TargetCapabilityUnavailable;
}

[[nodiscard]] ConSanAccessPolicyReason
classified_operation_reason(const ConSanAccessInventorySite &access,
                            ConSanAccessLoweringOperation operation) {
  return access_classifier_reason(access.lowering.operation(operation).reason, access.origin);
}

[[nodiscard]] ConSanProbeIntentKind intent_kind(ConSanCapabilityEngine engine) {
  switch (engine) {
  case ConSanCapabilityEngine::SuperCollider:
    return ConSanProbeIntentKind::RedundantAccessObservation;
  case ConSanCapabilityEngine::RecordReplay:
    return ConSanProbeIntentKind::AccessRecord;
  case ConSanCapabilityEngine::Sampled:
    return ConSanProbeIntentKind::SampledAccess;
  case ConSanCapabilityEngine::InlineShadow:
    return ConSanProbeIntentKind::ExactShadowAccess;
  case ConSanCapabilityEngine::Count:
    break;
  }
  return ConSanProbeIntentKind::Count;
}

[[nodiscard]] SemanticSiteId fallback_access_id(const ConSanAccessInventorySite &access) {
  return {
      .physical = access.physical_id,
      .domain = ConSanSemanticSiteDomain::Access,
      .member_ordinal = 0,
      .range_ordinal = 0,
  };
}

[[nodiscard]] std::vector<SemanticSiteId> semantic_ids(const ConSanAccessInventorySite &access) {
  std::vector<SemanticSiteId> result;
  result.reserve(std::max<size_t>(access.ranges.size(), 1u));
  for (const ConSanAccessRange &range : access.ranges)
    result.push_back(range.id.valid() ? range.id : fallback_access_id(access));
  if (result.empty())
    result.push_back(fallback_access_id(access));
  return result;
}

[[nodiscard]] std::vector<std::string>
container_names(std::span<const ConSanAccessInventorySite *const> aliases) {
  std::vector<std::string> result;
  result.reserve(aliases.size());
  for (const ConSanAccessInventorySite *access : aliases) {
    if (std::ranges::find(result, access->container.name) == result.end())
      result.push_back(access->container.name);
  }
  std::ranges::sort(result);
  return result;
}

} // namespace

const ConSanProbeIntent *ConSanObservationPlan::intent(ConSanProbeIntentId id) const {
  if (!id.valid() || id.value >= probe_intents.size())
    return nullptr;
  const ConSanProbeIntent &candidate = probe_intents[id.value];
  return candidate.id == id ? &candidate : nullptr;
}

bool ConSanObservationPlan::valid() const {
  if (!valid_engine(engine))
    return false;
  for (size_t index = 0; index < probe_intents.size(); ++index) {
    const ConSanProbeIntent &probe = probe_intents[index];
    if (probe.id.value != index || probe.engine != engine || !probe.physical_site.valid() ||
        probe.covered_semantic_sites.empty() || !valid_intent_kind(probe.kind) ||
        !valid_position(probe.position) || !valid_dynamic_result(probe.dynamic_result) ||
        std::ranges::any_of(probe.covered_semantic_sites,
                            [](const SemanticSiteId &site) { return !site.valid(); })) {
      return false;
    }
    const bool synchronization_intent = atomic_or_fence_intent(probe.kind);
    if (synchronization_intent != probe.synchronization_association.has_value() ||
        (probe.synchronization_association && !probe.synchronization_association->valid())) {
      return false;
    }
    if (probe.kind == ConSanProbeIntentKind::AtomicAddressCapture) {
      if (probe.position != ConSanProbePosition::Before ||
          probe.dynamic_result != ConSanDynamicResultRequirement::None) {
        return false;
      }
    } else if (synchronization_intent && probe.position != ConSanProbePosition::After) {
      return false;
    } else if (!synchronization_intent &&
               probe.dynamic_result != ConSanDynamicResultRequirement::None) {
      return false;
    }
  }
  for (const ConSanSiteDecision &decision : site_decisions) {
    if (decision.engine != engine || !decision.semantic_site.valid() ||
        !valid_decision_kind(decision.kind) || !valid_access_reason(decision.reason) ||
        decision.source_containers.empty()) {
      return false;
    }
    const bool admitted = decision.kind == ConSanSiteDecisionKind::Admitted;
    if (admitted != (decision.reason == ConSanAccessPolicyReason::None) ||
        admitted != !decision.intent_ids.empty()) {
      return false;
    }
    for (ConSanProbeIntentId id : decision.intent_ids) {
      const ConSanProbeIntent *probe = intent(id);
      if (probe == nullptr ||
          std::ranges::find(probe->covered_semantic_sites, decision.semantic_site) ==
              probe->covered_semantic_sites.end()) {
        return false;
      }
    }
  }
  for (const ConSanBarrierSiteDecision &decision : barrier_site_decisions) {
    if (decision.engine != engine || !decision.semantic_site.valid() ||
        decision.semantic_site.domain != ConSanSemanticSiteDomain::SynchronizationEvent ||
        !valid_decision_kind(decision.kind) || !valid_barrier_reason(decision.reason) ||
        decision.source_containers.empty()) {
      return false;
    }
    const bool admitted = decision.kind == ConSanSiteDecisionKind::Admitted;
    if (admitted != (decision.reason == ConSanBarrierPolicyReason::None) ||
        admitted != !decision.intent_ids.empty()) {
      return false;
    }
    for (ConSanProbeIntentId id : decision.intent_ids) {
      const ConSanProbeIntent *probe = intent(id);
      if (probe == nullptr ||
          std::ranges::find(probe->covered_semantic_sites, decision.semantic_site) ==
              probe->covered_semantic_sites.end()) {
        return false;
      }
    }
  }
  for (const ConSanAtomicSiteDecision &decision : atomic_site_decisions) {
    if (decision.engine != engine || !decision.semantic_site.valid() ||
        decision.semantic_site.domain != ConSanSemanticSiteDomain::SynchronizationEvent ||
        !valid_decision_kind(decision.kind) || !valid_capability_disposition(decision.capability) ||
        !valid_atomic_reason(decision.reason) || !valid_dynamic_result(decision.dynamic_result) ||
        decision.source_containers.empty() ||
        (decision.association && !decision.association->valid())) {
      return false;
    }
    const bool admitted = decision.kind == ConSanSiteDecisionKind::Admitted;
    if (admitted != (decision.reason == ConSanAtomicPolicyReason::None) ||
        admitted != !decision.intent_ids.empty() ||
        admitted != decision.lowering_form.has_value() ||
        (decision.lowering_form && !decision.lowering_form->is_well_formed()) ||
        (admitted && (!decision.association ||
                      (decision.capability != ConSanCapabilityDisposition::Supported &&
                       decision.capability != ConSanCapabilityDisposition::AssociatedOnly)))) {
      return false;
    }
    for (ConSanProbeIntentId id : decision.intent_ids) {
      const ConSanProbeIntent *probe = intent(id);
      if (probe == nullptr || probe->synchronization_association != decision.association ||
          std::ranges::find(probe->covered_semantic_sites, decision.semantic_site) ==
              probe->covered_semantic_sites.end()) {
        return false;
      }
    }
  }
  for (const ConSanFenceSiteDecision &decision : fence_site_decisions) {
    if (decision.engine != engine || !decision.semantic_site.valid() ||
        decision.semantic_site.domain != ConSanSemanticSiteDomain::SynchronizationEvent ||
        !valid_decision_kind(decision.kind) || !valid_capability_disposition(decision.capability) ||
        !valid_fence_reason(decision.reason) ||
        !valid_fence_association(decision.inventory_association) ||
        decision.source_containers.empty() ||
        (decision.association && !decision.association->valid())) {
      return false;
    }
    const bool admitted = decision.kind == ConSanSiteDecisionKind::Admitted;
    if (admitted != (decision.reason == ConSanFencePolicyReason::None) ||
        admitted != !decision.intent_ids.empty() ||
        admitted != decision.communication_lowering_form.has_value() ||
        (decision.communication_lowering_form &&
         !decision.communication_lowering_form->is_well_formed()) ||
        (admitted && (decision.inventory_association != ConSanFenceAssociation::Qualified ||
                      !decision.association ||
                      (decision.capability != ConSanCapabilityDisposition::Supported &&
                       decision.capability != ConSanCapabilityDisposition::AssociatedOnly)))) {
      return false;
    }
    for (ConSanProbeIntentId id : decision.intent_ids) {
      const ConSanProbeIntent *probe = intent(id);
      if (probe == nullptr || probe->synchronization_association != decision.association ||
          std::ranges::find(probe->covered_semantic_sites, decision.semantic_site) ==
              probe->covered_semantic_sites.end()) {
        return false;
      }
    }
  }
  return true;
}

bool ConSanObservationPlan::append(const ConSanObservationPlan &fragment) {
  if (!valid() || !fragment.valid() || engine != fragment.engine)
    return false;
  if (probe_intents.size() > ConSanProbeIntentId::invalid_value - fragment.probe_intents.size())
    return false;

  ConSanObservationPlan combined = *this;
  const uint32_t intent_base = static_cast<uint32_t>(combined.probe_intents.size());
  for (ConSanProbeIntent probe : fragment.probe_intents) {
    probe.id.value += intent_base;
    combined.probe_intents.push_back(std::move(probe));
  }
  for (ConSanSiteDecision decision : fragment.site_decisions) {
    for (ConSanProbeIntentId &id : decision.intent_ids)
      id.value += intent_base;
    combined.site_decisions.push_back(std::move(decision));
  }
  for (ConSanBarrierSiteDecision decision : fragment.barrier_site_decisions) {
    for (ConSanProbeIntentId &id : decision.intent_ids)
      id.value += intent_base;
    combined.barrier_site_decisions.push_back(std::move(decision));
  }
  for (ConSanAtomicSiteDecision decision : fragment.atomic_site_decisions) {
    for (ConSanProbeIntentId &id : decision.intent_ids)
      id.value += intent_base;
    combined.atomic_site_decisions.push_back(std::move(decision));
  }
  for (ConSanFenceSiteDecision decision : fragment.fence_site_decisions) {
    for (ConSanProbeIntentId &id : decision.intent_ids)
      id.value += intent_base;
    combined.fence_site_decisions.push_back(std::move(decision));
  }
  if (!combined.valid())
    return false;
  *this = std::move(combined);
  return true;
}

ConSanCoverageLedger::ConSanCoverageLedger(const ConSanObservationPlan &plan)
    : site_decisions_(plan.site_decisions), barrier_site_decisions_(plan.barrier_site_decisions),
      atomic_site_decisions_(plan.atomic_site_decisions),
      fence_site_decisions_(plan.fence_site_decisions) {
  intent_entries_.reserve(plan.probe_intents.size());
  for (const ConSanProbeIntent &intent : plan.probe_intents)
    intent_entries_.push_back({
        .intent = intent,
        .lowering = ConSanLoweringOutcomeKind::Pending,
        .resource_rejection_reason = std::nullopt,
        .detail = {},
    });
}

const ConSanIntentCoverageEntry *ConSanCoverageLedger::intent_entry(ConSanProbeIntentId id) const {
  if (!id.valid() || id.value >= intent_entries_.size())
    return nullptr;
  const ConSanIntentCoverageEntry &entry = intent_entries_[id.value];
  return entry.intent.id == id ? &entry : nullptr;
}

template <typename ResolveIntent>
bool runtime_static_mapping_matches_commit(const ConSanCommittedLowering &commit,
                                           ResolveIntent resolve_intent);

template <typename ResolveIntent>
bool committed_lowering_is_valid(const ConSanCommittedLowering &commit,
                                 ResolveIntent resolve_intent);

std::optional<ConSanCommittedLowering> make_consan_committed_lowering(
    const ConSanObservationPlan &plan, std::span<const ConSanProbeIntentId> intent_ids,
    std::span<const ConSanCommittedLoweringLocation> locations, ConSanLoweringOutcomeKind outcome,
    std::string detail, ConSanRuntimeStaticMapping runtime_mapping,
    std::optional<ConSanRegisterPlanReason> resource_rejection_reason) {
  ConSanCommittedLowering commit{
      .intent_ids = {},
      .original_physical_sites = {},
      .original_semantic_sites = {},
      .locations = std::vector(locations.begin(), locations.end()),
      .runtime_mapping = std::move(runtime_mapping),
      .outcome = outcome,
      .resource_rejection_reason = resource_rejection_reason,
      .detail = std::move(detail),
  };
  commit.intent_ids.reserve(intent_ids.size());
  for (ConSanProbeIntentId id : intent_ids) {
    const ConSanProbeIntent *intent = plan.intent(id);
    if (intent == nullptr)
      return std::nullopt;
    commit.intent_ids.push_back(id);
    if (std::ranges::find(commit.original_physical_sites, intent->physical_site) ==
        commit.original_physical_sites.end()) {
      commit.original_physical_sites.push_back(intent->physical_site);
    }
    for (const SemanticSiteId &site : intent->covered_semantic_sites) {
      if (std::ranges::find(commit.original_semantic_sites, site) ==
          commit.original_semantic_sites.end()) {
        commit.original_semantic_sites.push_back(site);
      }
    }
  }
  if (!committed_lowering_is_valid(commit, [&](ConSanProbeIntentId id) { return plan.intent(id); }))
    return std::nullopt;
  return commit;
}

template <typename ResolveIntent>
bool runtime_static_mapping_matches_commit(const ConSanCommittedLowering &commit,
                                           ResolveIntent resolve_intent) {
  if (commit.outcome != ConSanLoweringOutcomeKind::Instrumented)
    return commit.runtime_mapping.empty();

  std::vector<ConSanProbeIntentId> mapped_intents;
  const auto valid_attribution = [&](const ConSanStaticAccessAttribution &access,
                                     ConSanProbeIntentKind expected_kind) {
    if (access.intent_ids.empty() || !access.original_site.valid() ||
        std::ranges::find(commit.original_physical_sites, access.original_site) ==
            commit.original_physical_sites.end()) {
      return false;
    }
    if (access.owner_provenance_complete &&
        access.execution_owner_descriptor_file_offsets.empty()) {
      return false;
    }
    for (uint64_t owner : access.execution_owner_descriptor_file_offsets) {
      if (std::ranges::count(access.execution_owner_descriptor_file_offsets, owner) != 1)
        return false;
    }

    std::vector<SemanticSiteId> expected_semantic_sites;
    for (ConSanProbeIntentId id : access.intent_ids) {
      if (std::ranges::find(commit.intent_ids, id) == commit.intent_ids.end() ||
          std::ranges::find(mapped_intents, id) != mapped_intents.end()) {
        return false;
      }
      const ConSanProbeIntent *intent = resolve_intent(id);
      if (intent == nullptr || intent->kind != expected_kind ||
          intent->physical_site != access.original_site) {
        return false;
      }
      mapped_intents.push_back(id);
      for (const SemanticSiteId &site : intent->covered_semantic_sites) {
        if (std::ranges::find(expected_semantic_sites, site) == expected_semantic_sites.end())
          expected_semantic_sites.push_back(site);
      }
    }
    if (access.original_semantic_sites.size() != expected_semantic_sites.size())
      return false;
    return std::ranges::all_of(expected_semantic_sites, [&](const SemanticSiteId &site) {
      return std::ranges::count(access.original_semantic_sites, site) == 1;
    });
  };

  for (const ConSanRecordReplayStaticAccessMapping &mapping :
       commit.runtime_mapping.record_replay_accesses) {
    if (!valid_attribution(mapping.access, ConSanProbeIntentKind::AccessRecord))
      return false;
  }
  for (const ConSanSampledStaticAccessMapping &mapping : commit.runtime_mapping.sampled_accesses) {
    if (!valid_attribution(mapping.access, ConSanProbeIntentKind::SampledAccess) ||
        mapping.range_count == 0u || mapping.bank_count == 0u) {
      return false;
    }
  }
  for (const ConSanInlineCompactStaticAccessMapping &mapping :
       commit.runtime_mapping.inline_compact_accesses) {
    if (!valid_attribution(mapping.access, ConSanProbeIntentKind::ExactShadowAccess) ||
        mapping.token == 0u || !mapping.access.owner_provenance_complete ||
        mapping.access.execution_owner_descriptor_file_offsets.size() != 1u) {
      return false;
    }
  }
  for (ConSanProbeIntentId id : commit.intent_ids) {
    const ConSanProbeIntent *intent = resolve_intent(id);
    if (intent == nullptr)
      return false;
    const size_t mapping_count = std::ranges::count(mapped_intents, id);
    if ((intent->kind == ConSanProbeIntentKind::AccessRecord ||
         intent->kind == ConSanProbeIntentKind::SampledAccess) &&
        mapping_count != 1u) {
      return false;
    }
  }
  return true;
}

template <typename ResolveIntent>
bool committed_lowering_is_valid(const ConSanCommittedLowering &commit,
                                 ResolveIntent resolve_intent) {
  if (commit.intent_ids.empty() || commit.outcome == ConSanLoweringOutcomeKind::Pending ||
      !valid_lowering_outcome(commit.outcome) ||
      !valid_resource_rejection_reason(commit.resource_rejection_reason) ||
      (commit.resource_rejection_reason &&
       commit.outcome != ConSanLoweringOutcomeKind::ResourceRejected)) {
    return false;
  }
  const bool instrumented = commit.outcome == ConSanLoweringOutcomeKind::Instrumented;
  if (instrumented != !commit.locations.empty() ||
      !runtime_static_mapping_matches_commit(commit, resolve_intent)) {
    return false;
  }
  if (instrumented) {
    for (const ConSanCommittedLoweringLocation &location : commit.locations) {
      if (!location.original_site.valid() || location.emitted_size == 0u ||
          std::ranges::find(commit.original_physical_sites, location.original_site) ==
              commit.original_physical_sites.end()) {
        return false;
      }
    }
    for (const PhysicalSiteId &site : commit.original_physical_sites) {
      if (std::ranges::none_of(commit.locations, [&](const auto &location) {
            return location.original_site == site;
          })) {
        return false;
      }
    }
  }
  for (ConSanProbeIntentId id : commit.intent_ids) {
    const ConSanProbeIntent *intent = resolve_intent(id);
    if (intent == nullptr || std::ranges::count(commit.intent_ids, id) != 1 ||
        std::ranges::find(commit.original_physical_sites, intent->physical_site) ==
            commit.original_physical_sites.end() ||
        std::ranges::any_of(intent->covered_semantic_sites, [&](const SemanticSiteId &site) {
          return std::ranges::find(commit.original_semantic_sites, site) ==
                 commit.original_semantic_sites.end();
        })) {
      return false;
    }
  }
  return true;
}

bool ConSanCoverageLedger::publish_lowering_commit(ConSanCommittedLowering commit) {
  if (!committed_lowering_is_valid(commit, [&](ConSanProbeIntentId id) {
        const ConSanIntentCoverageEntry *entry = intent_entry(id);
        return entry == nullptr ? nullptr : &entry->intent;
      })) {
    return false;
  }
  for (ConSanProbeIntentId id : commit.intent_ids) {
    const ConSanIntentCoverageEntry *entry = intent_entry(id);
    if (entry == nullptr || entry->lowering != ConSanLoweringOutcomeKind::Pending)
      return false;
  }
  for (ConSanProbeIntentId id : commit.intent_ids) {
    ConSanIntentCoverageEntry &entry = intent_entries_[id.value];
    entry.lowering = commit.outcome;
    entry.resource_rejection_reason = commit.resource_rejection_reason;
    entry.detail = commit.detail;
  }
  lowering_commits_.push_back(std::move(commit));
  return true;
}

bool ConSanCoverageLedger::publish_lowering_commits(std::vector<ConSanCommittedLowering> commits) {
  ConSanCoverageLedger next = *this;
  for (ConSanCommittedLowering &commit : commits) {
    if (!next.publish_lowering_commit(std::move(commit)))
      return false;
  }
  *this = std::move(next);
  return true;
}

bool ConSanCoverageLedger::publish_coalescing_instrumented_commits(
    std::vector<ConSanCommittedLowering> commits) {
  ConSanCoverageLedger next = *this;
  const auto append_unique = [](auto &destination, const auto &source) {
    for (const auto &value : source) {
      if (std::ranges::find(destination, value) == destination.end())
        destination.push_back(value);
    }
  };
  for (ConSanCommittedLowering &incoming : commits) {
    if (incoming.outcome != ConSanLoweringOutcomeKind::Instrumented ||
        !committed_lowering_is_valid(incoming, [&](ConSanProbeIntentId id) {
          const ConSanIntentCoverageEntry *entry = next.intent_entry(id);
          return entry == nullptr ? nullptr : &entry->intent;
        })) {
      return false;
    }
    for (size_t index = 0; index < next.lowering_commits_.size();) {
      ConSanCommittedLowering &accepted = next.lowering_commits_[index];
      const bool overlaps = std::ranges::any_of(accepted.intent_ids, [&](const auto &id) {
        return std::ranges::find(incoming.intent_ids, id) != incoming.intent_ids.end();
      });
      if (!overlaps) {
        ++index;
        continue;
      }
      if (accepted.outcome != ConSanLoweringOutcomeKind::Instrumented)
        return false;
      append_unique(incoming.intent_ids, accepted.intent_ids);
      append_unique(incoming.original_physical_sites, accepted.original_physical_sites);
      append_unique(incoming.original_semantic_sites, accepted.original_semantic_sites);
      append_unique(incoming.locations, accepted.locations);
      incoming.runtime_mapping.append(std::move(accepted.runtime_mapping));
      if (incoming.detail.empty())
        incoming.detail = std::move(accepted.detail);
      for (ConSanProbeIntentId id : accepted.intent_ids) {
        ConSanIntentCoverageEntry &entry = next.intent_entries_[id.value];
        entry.lowering = ConSanLoweringOutcomeKind::Pending;
        entry.resource_rejection_reason.reset();
        entry.detail.clear();
      }
      next.lowering_commits_.erase(next.lowering_commits_.begin() +
                                   static_cast<std::ptrdiff_t>(index));
    }
    if (!next.publish_lowering_commit(std::move(incoming)))
      return false;
  }
  *this = std::move(next);
  return true;
}

ConSanRuntimeStaticMapping ConSanCoverageLedger::runtime_static_mapping() const {
  ConSanRuntimeStaticMapping mapping;
  for (const ConSanCommittedLowering &commit : lowering_commits_)
    mapping.append(commit.runtime_mapping);
  return mapping;
}

bool ConSanCoverageLedger::matches_plan(const ConSanObservationPlan &plan) const {
  if (site_decisions_ != plan.site_decisions ||
      barrier_site_decisions_ != plan.barrier_site_decisions ||
      atomic_site_decisions_ != plan.atomic_site_decisions ||
      fence_site_decisions_ != plan.fence_site_decisions ||
      intent_entries_.size() != plan.probe_intents.size()) {
    return false;
  }
  for (size_t index = 0; index < intent_entries_.size(); ++index) {
    if (intent_entries_[index].intent != plan.probe_intents[index])
      return false;
  }
  return true;
}

void ConSanCoverageLedger::discard_instrumented_lowerings() {
  std::erase_if(lowering_commits_, [](const ConSanCommittedLowering &commit) {
    return commit.outcome == ConSanLoweringOutcomeKind::Instrumented;
  });
  for (ConSanIntentCoverageEntry &entry : intent_entries_) {
    entry.lowering = ConSanLoweringOutcomeKind::Pending;
    entry.resource_rejection_reason.reset();
    entry.detail.clear();
  }
  for (const ConSanCommittedLowering &commit : lowering_commits_) {
    for (ConSanProbeIntentId id : commit.intent_ids) {
      ConSanIntentCoverageEntry &entry = intent_entries_[id.value];
      entry.lowering = commit.outcome;
      entry.resource_rejection_reason = commit.resource_rejection_reason;
      entry.detail = commit.detail;
    }
  }
}

bool ConSanCoverageLedger::all_intents_instrumented() const {
  return std::ranges::all_of(intent_entries_, [](const ConSanIntentCoverageEntry &entry) {
    return entry.lowering == ConSanLoweringOutcomeKind::Instrumented;
  });
}

ConSanAccessPolicyResult plan_consan_access_observation(const ProgramInventory &inventory,
                                                        const ConSanAccessPolicyRequest &request) {
  ConSanAccessPolicyResult result;
  result.plan.engine = request.engine;
  if (!valid_engine(request.engine) || inventory.empty())
    return result;

  std::map<uint64_t, std::vector<const ConSanAccessInventorySite *>> aliases_by_offset;
  for (const ConSanAccessInventorySite &access : inventory.access_sites())
    aliases_by_offset[access.physical_id.original_text_offset].push_back(&access);

  for (const auto &[offset, aliases] : aliases_by_offset) {
    (void)offset;
    const ConSanAccessInventorySite &access = *aliases.front();
    const std::vector<SemanticSiteId> ids = semantic_ids(access);
    const std::vector<std::string> names = container_names(aliases);
    ConSanSiteDecisionKind decision_kind = ConSanSiteDecisionKind::NotApplicable;
    ConSanAccessPolicyReason reason = ConSanAccessPolicyReason::AccessFamilyDisabled;

    const bool conflicting_alias = std::ranges::any_of(
        aliases, [&](const auto *alias) { return !access_alias_semantics_equal(access, *alias); });
    const bool flat = access.origin == ConSanAccessOrigin::Flat;
    const ConSanCapabilityForm form =
        flat ? ConSanCapabilityForm::GroupFlatAccess : ConSanCapabilityForm::NativeLdsAccess;
    const bool enabled = flat ? request.group_flat_enabled : request.native_lds_enabled;
    std::vector<uint64_t> owner_descriptors;
    for (const ConSanAccessInventorySite *alias : aliases) {
      for (uint64_t owner : alias->execution_owner_descriptor_file_offsets) {
        if (std::ranges::find(owner_descriptors, owner) == owner_descriptors.end())
          owner_descriptors.push_back(owner);
      }
    }

    if (!contains_substring(aliases, request.container_filter) ||
        !consan_site_matches_kernel_allowlist(inventory, owner_descriptors, names,
                                              request.kernel_name_allowlist)) {
      reason = ConSanAccessPolicyReason::ContainerFilterExcluded;
    } else if (contains_physical_site(request.reserved_for_synchronization, access.physical_id)) {
      reason = ConSanAccessPolicyReason::ReservedForSynchronizationPolicy;
    } else if (!enabled) {
      reason = ConSanAccessPolicyReason::AccessFamilyDisabled;
    } else if (!consan_arch_supports_capability_form(inventory.arch(), form) ||
               consan_capability_disposition(inventory.target(), request.engine, form) !=
                   ConSanCapabilityDisposition::Supported) {
      reason = ConSanAccessPolicyReason::TargetCapabilityUnavailable;
    } else if (access.kind != ConSanLdsAccessKind::Read &&
               access.kind != ConSanLdsAccessKind::Write &&
               !(request.engine != ConSanCapabilityEngine::SuperCollider && !flat &&
                 access.kind == ConSanLdsAccessKind::Atomic &&
                 consan_arch_supports_capability_form(
                     inventory.arch(), ConSanCapabilityForm::RelaxedLdsAtomicAccess) &&
                 access.lowering.operation(ConSanAccessLoweringOperation::ReplayGuestAccess)
                     .available())) {
      reason = ConSanAccessPolicyReason::OperationKindExcluded;
    } else if (flat && access.address_space == ConSanAccessAddressSpace::NonGroup) {
      reason = ConSanAccessPolicyReason::NonGroupAddressSpace;
    } else if (flat && access.address_space == ConSanAccessAddressSpace::Unresolved) {
      reason = ConSanAccessPolicyReason::FlatProvenancePolicyExcluded;
    } else if (flat && access.confidence != ConSanSemanticConfidence::Exact &&
               request.flat_provenance_mode == ConSanFlatProvenanceMode::Strict) {
      reason = ConSanAccessPolicyReason::FlatProvenancePolicyExcluded;
    } else if (conflicting_alias) {
      decision_kind = ConSanSiteDecisionKind::Unsupported;
      reason = ConSanAccessPolicyReason::ConflictingPhysicalAliases;
      result.errors.push_back(reason);
    } else {
      const ConSanAccessLoweringOperation operation =
          request.engine == ConSanCapabilityEngine::SuperCollider
              ? ConSanAccessLoweringOperation::CompareObservedValue
              : ConSanAccessLoweringOperation::ReplayGuestAccess;
      reason = classified_operation_reason(access, operation);
      decision_kind = reason == ConSanAccessPolicyReason::None
                          ? ConSanSiteDecisionKind::Admitted
                          : ConSanSiteDecisionKind::Unsupported;
    }

    std::optional<ConSanProbeIntentId> intent_id;
    if (decision_kind == ConSanSiteDecisionKind::Admitted) {
      intent_id = ConSanProbeIntentId{static_cast<uint32_t>(result.plan.probe_intents.size())};
      result.plan.probe_intents.push_back({
          .id = *intent_id,
          .engine = request.engine,
          .physical_site = access.physical_id,
          .covered_semantic_sites = ids,
          .kind = intent_kind(request.engine),
          .position = ConSanProbePosition::Before,
          .synchronization_association = std::nullopt,
          .dynamic_result = ConSanDynamicResultRequirement::None,
      });
    }
    for (const SemanticSiteId &id : ids) {
      ConSanSiteDecision decision{
          .engine = request.engine,
          .semantic_site = id,
          .kind = decision_kind,
          .reason = reason,
          .intent_ids = {},
          .source_containers = names,
      };
      if (intent_id)
        decision.intent_ids.push_back(*intent_id);
      result.plan.site_decisions.push_back(std::move(decision));
    }
  }
  return result;
}

} // namespace rocjitsu
