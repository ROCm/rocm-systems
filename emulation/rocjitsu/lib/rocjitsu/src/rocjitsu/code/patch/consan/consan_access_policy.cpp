// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"

#include <algorithm>
#include <map>
#include <ranges>
#include <tuple>

namespace rocjitsu {
namespace {

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

using PhysicalSitesByOffset = std::unordered_multimap<uint64_t, const PhysicalSiteId *>;

[[nodiscard]] PhysicalSitesByOffset
index_physical_sites_by_offset(std::span<const PhysicalSiteId> sites) {
  PhysicalSitesByOffset index;
  index.reserve(sites.size());
  for (const PhysicalSiteId &site : sites)
    index.emplace(site.original_text_offset, &site);
  return index;
}

[[nodiscard]] bool contains_physical_site(const PhysicalSitesByOffset &sites,
                                          const PhysicalSiteId &site) {
  const auto [begin, end] = sites.equal_range(site.original_text_offset);
  return std::ranges::any_of(std::ranges::subrange(begin, end),
                             [&](const auto &candidate) { return *candidate.second == site; });
}

[[nodiscard]] bool contains_substring(const ProgramInventory &inventory,
                                      std::span<const ConSanProgramSite *const> aliases,
                                      std::string_view filter) {
  return filter.empty() || std::ranges::any_of(aliases, [&](const auto *access) {
           const ConSanProgramContainer *container = inventory.container(access->container);
           return container != nullptr && container->name.find(filter) != std::string::npos;
         });
}

[[nodiscard]] std::vector<std::string>
source_container_names(const ProgramInventory &inventory,
                       std::span<const ConSanProgramSite *const> aliases) {
  std::vector<std::string> names;
  names.reserve(aliases.size());
  for (const ConSanProgramSite *alias : aliases) {
    const ConSanProgramContainer *container = inventory.container(alias->container);
    if (container != nullptr)
      names.push_back(container->name);
  }
  std::ranges::sort(names);
  names.erase(std::ranges::unique(names).begin(), names.end());
  return names;
}

[[nodiscard]] bool access_alias_semantics_equal(const ProgramInventory &inventory,
                                                const ConSanProgramSite &lhs,
                                                const ConSanProgramSite &rhs) {
  const auto container_entry = [&](const ConSanProgramSite &site) {
    const ConSanProgramContainer *container = inventory.container(site.container);
    return container == nullptr ? std::optional<uint64_t>{}
                                : std::optional{container->entry_text_offset};
  };
  const auto container_kind = [&](const ConSanProgramSite &site) {
    const ConSanProgramContainer *container = inventory.container(site.container);
    return container == nullptr ? ConSanProgramContainerKind::Count : container->kind;
  };
  return std::tuple(container_kind(lhs), container_entry(lhs), lhs.origin, lhs.kind,
                    lhs.address_space, lhs.provenance, lhs.confidence, lhs.lowering,
                    lhs.decoded_file_offset(), lhs.size(), lhs.decoded_width_bits,
                    lhs.mnemonic_view(), lhs.flat_address_space_hint, lhs.operands, lhs.ranges,
                    lhs.exclusions) ==
         std::tuple(container_kind(rhs), container_entry(rhs), rhs.origin, rhs.kind,
                    rhs.address_space, rhs.provenance, rhs.confidence, rhs.lowering,
                    rhs.decoded_file_offset(), rhs.size(), rhs.decoded_width_bits,
                    rhs.mnemonic_view(), rhs.flat_address_space_hint, rhs.operands, rhs.ranges,
                    rhs.exclusions);
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
classified_operation_reason(const ConSanProgramSite &access,
                            ConSanAccessLoweringOperation operation) {
  return access_classifier_reason(access.lowering.operation(operation).reason, access.origin);
}

[[nodiscard]] SemanticSiteId fallback_access_id(const ConSanProgramSite &access) {
  return {
      .physical = access.physical_id,
      .domain = ConSanSemanticSiteDomain::Access,
      .member_ordinal = 0,
      .range_ordinal = 0,
  };
}

[[nodiscard]] std::vector<SemanticSiteId> semantic_ids(const ConSanProgramSite &access) {
  std::vector<SemanticSiteId> result;
  result.reserve(std::max<size_t>(access.ranges.size(), 1u));
  for (const ConSanAccessRange &range : access.ranges)
    result.push_back(range.id.valid() ? range.id : fallback_access_id(access));
  if (result.empty())
    result.push_back(fallback_access_id(access));
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
  if (consan_engine_probe_vocabulary(engine) == nullptr)
    return false;
  for (size_t index = 0; index < probe_intents.size(); ++index) {
    const ConSanProbeIntent &probe = probe_intents[index];
    if (probe.id.value != index || probe.engine != engine || !probe.source_site.valid() ||
        !probe.physical_site.valid() || probe.covered_semantic_sites.empty() ||
        !valid_intent_kind(probe.kind) || !valid_position(probe.position) ||
        !valid_dynamic_result(probe.dynamic_result) ||
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
          probe.dynamic_result != ConSanDynamicResultRequirement::None ||
          !probe.atomic_lowering_form || !probe.atomic_lowering_form->is_well_formed()) {
        return false;
      }
    } else if (synchronization_intent && probe.position != ConSanProbePosition::After) {
      return false;
    } else if (!synchronization_intent &&
               probe.dynamic_result != ConSanDynamicResultRequirement::None) {
      return false;
    } else if (probe.atomic_lowering_form) {
      return false;
    }
  }
  for (const ConSanSiteDecision &decision : site_decisions) {
    if (!decision.semantic_site.valid() || !valid_decision_kind(decision.kind) ||
        !valid_access_reason(decision.reason)) {
      return false;
    }
    const bool admitted = decision.kind == ConSanSiteDecisionKind::Admitted;
    if (admitted != (decision.reason == ConSanAccessPolicyReason::None))
      return false;
  }
  for (const ConSanBarrierSiteDecision &decision : barrier_site_decisions) {
    if (!decision.semantic_site.valid() ||
        decision.semantic_site.domain != ConSanSemanticSiteDomain::SynchronizationEvent ||
        !valid_decision_kind(decision.kind) || !valid_barrier_reason(decision.reason)) {
      return false;
    }
    const bool admitted = decision.kind == ConSanSiteDecisionKind::Admitted;
    if (admitted != (decision.reason == ConSanBarrierPolicyReason::None))
      return false;
  }
  for (const ConSanAtomicSiteDecision &decision : atomic_site_decisions) {
    if (!decision.semantic_site.valid() ||
        decision.semantic_site.domain != ConSanSemanticSiteDomain::SynchronizationEvent ||
        !valid_decision_kind(decision.kind) || !valid_capability_disposition(decision.capability) ||
        !valid_atomic_reason(decision.reason)) {
      return false;
    }
    const bool admitted = decision.kind == ConSanSiteDecisionKind::Admitted;
    if (admitted != (decision.reason == ConSanAtomicPolicyReason::None) ||
        (admitted && decision.capability != ConSanCapabilityDisposition::Supported &&
         decision.capability != ConSanCapabilityDisposition::AssociatedOnly)) {
      return false;
    }
  }
  for (const ConSanFenceSiteDecision &decision : fence_site_decisions) {
    if (!decision.semantic_site.valid() ||
        decision.semantic_site.domain != ConSanSemanticSiteDomain::SynchronizationEvent ||
        !valid_decision_kind(decision.kind) || !valid_capability_disposition(decision.capability) ||
        !valid_fence_reason(decision.reason) ||
        !valid_fence_association(decision.inventory_association)) {
      return false;
    }
    const bool admitted = decision.kind == ConSanSiteDecisionKind::Admitted;
    if (admitted != (decision.reason == ConSanFencePolicyReason::None) ||
        (admitted && (decision.inventory_association != ConSanFenceAssociation::Qualified ||
                      (decision.capability != ConSanCapabilityDisposition::Supported &&
                       decision.capability != ConSanCapabilityDisposition::AssociatedOnly)))) {
      return false;
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
  combined.site_decisions.insert(combined.site_decisions.end(), fragment.site_decisions.begin(),
                                 fragment.site_decisions.end());
  combined.barrier_site_decisions.insert(combined.barrier_site_decisions.end(),
                                         fragment.barrier_site_decisions.begin(),
                                         fragment.barrier_site_decisions.end());
  combined.atomic_site_decisions.insert(combined.atomic_site_decisions.end(),
                                        fragment.atomic_site_decisions.begin(),
                                        fragment.atomic_site_decisions.end());
  combined.fence_site_decisions.insert(combined.fence_site_decisions.end(),
                                       fragment.fence_site_decisions.begin(),
                                       fragment.fence_site_decisions.end());
  if (!combined.valid())
    return false;
  *this = std::move(combined);
  return true;
}

ConSanCoverageLedger::ConSanCoverageLedger(ConSanObservationPlan plan)
    : observation_plan_(std::move(plan)) {
  intent_entries_.reserve(observation_plan_.probe_intents.size());
  for (const ConSanProbeIntent &intent : observation_plan_.probe_intents) {
    intent_entries_.push_back({
        .intent_id = intent.id,
        .lowering = ConSanLoweringOutcomeKind::Pending,
        .resource_rejection_reason = std::nullopt,
        .detail = {},
    });
    const size_t kind_index = static_cast<size_t>(intent.kind);
    if (kind_index < intent_ids_by_kind_and_text_offset_.size()) {
      intent_ids_by_kind_and_text_offset_[kind_index][intent.physical_site.original_text_offset]
          .push_back(intent.id);
    }
    for (const SemanticSiteId &semantic_site : intent.covered_semantic_sites) {
      auto &ids = intent_ids_by_semantic_text_offset_[semantic_site.physical.original_text_offset];
      if (ids.empty() || ids.back() != intent.id)
        ids.push_back(intent.id);
    }
  }
}

const ConSanIntentCoverageEntry *ConSanCoverageLedger::intent_entry(ConSanProbeIntentId id) const {
  if (!id.valid() || id.value >= intent_entries_.size())
    return nullptr;
  const ConSanIntentCoverageEntry &entry = intent_entries_[id.value];
  return entry.intent_id == id ? &entry : nullptr;
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
  const auto owns_physical_site = [&](const PhysicalSiteId &site) {
    return std::ranges::any_of(commit.intent_ids, [&](ConSanProbeIntentId id) {
      const ConSanProbeIntent *intent = resolve_intent(id);
      return intent != nullptr && intent->physical_site == site;
    });
  };
  const auto valid_attribution = [&](const ConSanStaticAccessAttribution &access,
                                     ConSanProbeIntentKind expected_kind) {
    if (access.intent_ids.empty() || !access.original_site.valid() ||
        !owns_physical_site(access.original_site)) {
      return false;
    }
    if (access.owner_provenance_complete && access.execution_owner_kernel_ids.empty()) {
      return false;
    }
    for (ConSanProgramContainerId owner : access.execution_owner_kernel_ids) {
      if (!owner.valid() || std::ranges::count(access.execution_owner_kernel_ids, owner) != 1)
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

  if (const auto *mappings = commit.runtime_mapping.record_replay()) {
    for (const ConSanRecordReplayStaticAccessMapping &mapping : *mappings) {
      if (!valid_attribution(mapping.access, ConSanProbeIntentKind::AccessRecord))
        return false;
    }
  }
  if (const auto *mappings = commit.runtime_mapping.sampled()) {
    for (const ConSanSampledStaticAccessMapping &mapping : *mappings) {
      if (!valid_attribution(mapping.access, ConSanProbeIntentKind::SampledAccess) ||
          mapping.range_count == 0u || mapping.bank_count == 0u) {
        return false;
      }
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
          std::ranges::none_of(commit.intent_ids, [&](ConSanProbeIntentId id) {
            const ConSanProbeIntent *intent = resolve_intent(id);
            return intent != nullptr && intent->physical_site == location.original_site;
          })) {
        return false;
      }
    }
  }
  for (ConSanProbeIntentId id : commit.intent_ids) {
    const ConSanProbeIntent *intent = resolve_intent(id);
    if (intent == nullptr || std::ranges::count(commit.intent_ids, id) != 1 ||
        (instrumented && std::ranges::none_of(commit.locations, [&](const auto &location) {
           return location.original_site == intent->physical_site;
         }))) {
      return false;
    }
  }
  return true;
}

bool ConSanCoverageLedger::publish_lowering_commit(ConSanCommittedLowering commit) {
  if (!committed_lowering_is_valid(
          commit, [&](ConSanProbeIntentId id) { return observation_plan_.intent(id); })) {
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

bool ConSanCoverageLedger::publish_lowering_rejection(
    std::span<const ConSanProbeIntentId> intent_ids, ConSanLoweringOutcomeKind outcome,
    std::string detail, std::optional<ConSanRegisterPlanReason> resource_rejection_reason) {
  auto commit = make_consan_committed_lowering(
      observation_plan_, intent_ids, std::span<const ConSanCommittedLoweringLocation>{}, outcome,
      std::move(detail), ConSanRuntimeStaticMapping{}, resource_rejection_reason);
  return commit && publish_lowering_commit(std::move(*commit));
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
        !committed_lowering_is_valid(
            incoming, [&](ConSanProbeIntentId id) { return next.observation_plan_.intent(id); })) {
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
      append_unique(incoming.locations, accepted.locations);
      if (!incoming.runtime_mapping.append(std::move(accepted.runtime_mapping)))
        return false;
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

bool ConSanCoverageLedger::publish_replacing_instrumented_commits(
    std::vector<ConSanCommittedLowering> commits) {
  ConSanCoverageLedger next = *this;
  std::vector<ConSanProbeIntentId> replacement_intents;
  for (const ConSanCommittedLowering &commit : commits) {
    for (ConSanProbeIntentId id : commit.intent_ids) {
      if (std::ranges::find(replacement_intents, id) == replacement_intents.end())
        replacement_intents.push_back(id);
    }
  }

  for (size_t index = 0; index < next.lowering_commits_.size();) {
    const ConSanCommittedLowering &accepted = next.lowering_commits_[index];
    const bool overlaps = std::ranges::any_of(accepted.intent_ids, [&](const auto &id) {
      return std::ranges::find(replacement_intents, id) != replacement_intents.end();
    });
    if (!overlaps) {
      ++index;
      continue;
    }
    if (accepted.outcome != ConSanLoweringOutcomeKind::Instrumented ||
        std::ranges::any_of(accepted.intent_ids, [&](const auto &id) {
          return std::ranges::find(replacement_intents, id) == replacement_intents.end();
        })) {
      return false;
    }
    for (ConSanProbeIntentId id : accepted.intent_ids) {
      ConSanIntentCoverageEntry &entry = next.intent_entries_[id.value];
      entry.lowering = ConSanLoweringOutcomeKind::Pending;
      entry.resource_rejection_reason.reset();
      entry.detail.clear();
    }
    next.lowering_commits_.erase(next.lowering_commits_.begin() +
                                 static_cast<std::ptrdiff_t>(index));
  }
  if (!next.publish_coalescing_instrumented_commits(std::move(commits)))
    return false;
  *this = std::move(next);
  return true;
}

ConSanRuntimeStaticMapping ConSanCoverageLedger::runtime_static_mapping() const {
  ConSanRuntimeStaticMapping mapping;
  for (const ConSanCommittedLowering &commit : lowering_commits_) {
    if (!mapping.append(commit.runtime_mapping))
      return {};
  }
  return mapping;
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

ConSanAccessPolicyResult plan_consan_access_observation(const ProgramInventory &inventory,
                                                        const ConSanAccessPolicyRequest &request) {
  ConSanAccessPolicyResult result;
  result.plan.engine = request.engine;
  const ConSanEngineProbeVocabulary *vocabulary = consan_engine_probe_vocabulary(request.engine);
  if (vocabulary == nullptr || inventory.empty())
    return result;

  std::map<uint64_t, std::vector<const ConSanProgramSite *>> aliases_by_offset;
  for (const ConSanProgramSite &access : inventory.access_sites())
    aliases_by_offset[access.physical_id.original_text_offset].push_back(&access);
  const PhysicalSitesByOffset synchronization_reservations =
      index_physical_sites_by_offset(request.reserved_for_synchronization);

  for (const auto &[offset, aliases] : aliases_by_offset) {
    (void)offset;
    const ConSanProgramSite &access = *aliases.front();
    const std::vector<SemanticSiteId> ids = semantic_ids(access);
    // The aliases were already grouped by physical offset above. Derive their
    // presentation names from that group instead of rescanning the complete
    // program-site arena once per access.
    const std::vector<std::string> names = source_container_names(inventory, aliases);
    ConSanSiteDecisionKind decision_kind = ConSanSiteDecisionKind::NotApplicable;
    ConSanAccessPolicyReason reason = ConSanAccessPolicyReason::AccessFamilyDisabled;

    const bool conflicting_alias = std::ranges::any_of(aliases, [&](const auto *alias) {
      return !access_alias_semantics_equal(inventory, access, *alias);
    });
    const bool flat = access.origin == ConSanAccessOrigin::Flat;
    const ConSanCapabilityForm form =
        flat ? ConSanCapabilityForm::GroupFlatAccess : ConSanCapabilityForm::NativeLdsAccess;
    const bool enabled = flat ? request.group_flat_enabled : request.native_lds_enabled;
    const std::vector<uint64_t> owner_descriptors = inventory.execution_owner_descriptors(aliases);

    if (!contains_substring(inventory, aliases, request.container_filter) ||
        !consan_site_matches_kernel_allowlist(inventory, owner_descriptors, names,
                                              request.kernel_name_allowlist)) {
      reason = ConSanAccessPolicyReason::ContainerFilterExcluded;
    } else if (contains_physical_site(synchronization_reservations, access.physical_id)) {
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

    if (decision_kind == ConSanSiteDecisionKind::Admitted) {
      const ConSanProbeIntentId intent_id{static_cast<uint32_t>(result.plan.probe_intents.size())};
      result.plan.probe_intents.push_back({
          .id = intent_id,
          .engine = request.engine,
          .source_site = access.id,
          .physical_site = access.physical_id,
          .covered_semantic_sites = ids,
          .kind = vocabulary->access,
          .position = ConSanProbePosition::Before,
          .synchronization_association = std::nullopt,
          .dynamic_result = ConSanDynamicResultRequirement::None,
          .atomic_lowering_form = std::nullopt,
      });
    }
    for (const SemanticSiteId &id : ids) {
      ConSanSiteDecision decision{
          .semantic_site = id,
          .kind = decision_kind,
          .reason = reason,
      };
      result.plan.site_decisions.push_back(std::move(decision));
    }
  }
  return result;
}

} // namespace rocjitsu
