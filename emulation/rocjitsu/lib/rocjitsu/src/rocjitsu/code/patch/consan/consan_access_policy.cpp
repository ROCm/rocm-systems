// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"

#include <algorithm>
#include <map>
#include <ranges>
#include <tuple>

namespace rocjitsu::consan {
namespace {

[[nodiscard]] bool valid_decision_kind(SiteDecisionKind kind) {
  return static_cast<uint8_t>(kind) < static_cast<uint8_t>(SiteDecisionKind::Count);
}

[[nodiscard]] bool valid_access_reason(AccessPolicyReason reason) {
  return static_cast<uint8_t>(reason) < static_cast<uint8_t>(AccessPolicyReason::Count);
}

[[nodiscard]] bool valid_barrier_reason(BarrierPolicyReason reason) {
  return static_cast<uint8_t>(reason) < static_cast<uint8_t>(BarrierPolicyReason::Count);
}

[[nodiscard]] bool valid_atomic_reason(AtomicPolicyReason reason) {
  return static_cast<uint8_t>(reason) < static_cast<uint8_t>(AtomicPolicyReason::Count);
}

[[nodiscard]] bool valid_fence_reason(FencePolicyReason reason) {
  return static_cast<uint8_t>(reason) < static_cast<uint8_t>(FencePolicyReason::Count);
}

[[nodiscard]] bool valid_fence_association(FenceAssociation association) {
  return static_cast<uint8_t>(association) < static_cast<uint8_t>(FenceAssociation::Count);
}

[[nodiscard]] bool valid_capability_disposition(CapabilityDisposition disposition) {
  switch (disposition) {
  case CapabilityDisposition::OutOfContract:
  case CapabilityDisposition::NotApplicable:
  case CapabilityDisposition::Supported:
  case CapabilityDisposition::MutationOnly:
  case CapabilityDisposition::AccessOnly:
  case CapabilityDisposition::AssociatedOnly:
    return true;
  }
  return false;
}

[[nodiscard]] bool valid_intent_kind(ProbeIntentKind kind) {
  return static_cast<uint8_t>(kind) < static_cast<uint8_t>(ProbeIntentKind::Count);
}

[[nodiscard]] bool valid_position(ProbePosition position) {
  return static_cast<uint8_t>(position) < static_cast<uint8_t>(ProbePosition::Count);
}

[[nodiscard]] bool valid_dynamic_result(DynamicResultRequirement requirement) {
  return static_cast<uint8_t>(requirement) < static_cast<uint8_t>(DynamicResultRequirement::Count);
}

[[nodiscard]] bool atomic_or_fence_intent(ProbeIntentKind kind) {
  switch (kind) {
  case ProbeIntentKind::AtomicAddressCapture:
  case ProbeIntentKind::AtomicOrdering:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] bool valid_lowering_outcome(LoweringOutcomeKind outcome) {
  return static_cast<uint8_t>(outcome) < static_cast<uint8_t>(LoweringOutcomeKind::Count);
}

[[nodiscard]] bool valid_resource_rejection_reason(std::optional<RegisterPlanReason> reason) {
  return !reason ||
         (*reason > RegisterPlanReason::None && *reason <= RegisterPlanReason::DynamicStack);
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
                                      std::span<const ProgramSite *const> aliases,
                                      std::string_view filter) {
  return filter.empty() || std::ranges::any_of(aliases, [&](const auto *access) {
           const ProgramContainer *container = inventory.container(access->container);
           return container != nullptr && container->name.find(filter) != std::string::npos;
         });
}

[[nodiscard]] std::vector<std::string>
source_container_names(const ProgramInventory &inventory,
                       std::span<const ProgramSite *const> aliases) {
  std::vector<std::string> names;
  names.reserve(aliases.size());
  for (const ProgramSite *alias : aliases) {
    const ProgramContainer *container = inventory.container(alias->container);
    if (container != nullptr)
      names.push_back(container->name);
  }
  std::ranges::sort(names);
  names.erase(std::ranges::unique(names).begin(), names.end());
  return names;
}

[[nodiscard]] bool access_alias_semantics_equal(const ProgramInventory &inventory,
                                                const ProgramSite &lhs, const ProgramSite &rhs) {
  const auto container_entry = [&](const ProgramSite &site) {
    const ProgramContainer *container = inventory.container(site.container);
    return container == nullptr ? std::optional<uint64_t>{}
                                : std::optional{container->entry_text_offset};
  };
  const auto container_kind = [&](const ProgramSite &site) {
    const ProgramContainer *container = inventory.container(site.container);
    return container == nullptr ? ProgramContainerKind::Count : container->kind;
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

[[nodiscard]] AccessPolicyReason access_classifier_reason(AccessClassifierReason reason,
                                                          AccessOrigin origin) {
  switch (reason) {
  case AccessClassifierReason::None:
    return AccessPolicyReason::None;
  case AccessClassifierReason::NonAccessInstruction:
    return AccessPolicyReason::NonAccessInstruction;
  case AccessClassifierReason::InvalidInstructionSize:
    return AccessPolicyReason::InvalidInstructionSize;
  case AccessClassifierReason::InvalidAccessWidth:
    return AccessPolicyReason::InvalidAccessWidth;
  case AccessClassifierReason::MissingAddressOperand:
    return AccessPolicyReason::MissingAddressOperand;
  case AccessClassifierReason::RangeEncodingUnavailable:
    return AccessPolicyReason::RangeEncodingUnavailable;
  case AccessClassifierReason::InstructionOutOfBounds:
    return AccessPolicyReason::InstructionOutOfBounds;
  case AccessClassifierReason::UnsupportedMnemonic:
  case AccessClassifierReason::MissingResultOperand:
  case AccessClassifierReason::MissingDataOperand:
  case AccessClassifierReason::OperandRegisterRange:
    return AccessPolicyReason::UnsupportedMnemonic;
  case AccessClassifierReason::UnsupportedEncoding:
    return origin == AccessOrigin::Flat ? AccessPolicyReason::UnsupportedFlatEncoding
                                        : AccessPolicyReason::UnsupportedMnemonic;
  case AccessClassifierReason::NonzeroImmediateOffset:
    return AccessPolicyReason::NonzeroFlatOffset;
  case AccessClassifierReason::ReservedAddressRegister:
    return AccessPolicyReason::ReservedFlatAddressRegister;
  case AccessClassifierReason::TargetUnavailable:
    return AccessPolicyReason::TargetCapabilityUnavailable;
  case AccessClassifierReason::Count:
    break;
  }
  return AccessPolicyReason::TargetCapabilityUnavailable;
}

[[nodiscard]] AccessPolicyReason classified_operation_reason(const ProgramSite &access,
                                                             AccessLoweringOperation operation) {
  return access_classifier_reason(access.lowering.operation(operation).reason, access.origin);
}

[[nodiscard]] SemanticSiteId fallback_access_id(const ProgramSite &access) {
  return {
      .physical = access.physical_id,
      .domain = SemanticSiteDomain::Access,
      .member_ordinal = 0,
      .range_ordinal = 0,
  };
}

[[nodiscard]] std::vector<SemanticSiteId> semantic_ids(const ProgramSite &access) {
  std::vector<SemanticSiteId> result;
  result.reserve(std::max<size_t>(access.ranges.size(), 1u));
  for (const AccessRange &range : access.ranges)
    result.push_back(range.id.valid() ? range.id : fallback_access_id(access));
  if (result.empty())
    result.push_back(fallback_access_id(access));
  return result;
}

} // namespace

const ProbeIntent *ObservationPlan::intent(ProbeIntentId id) const {
  if (!id.valid() || id.value >= probe_intents.size())
    return nullptr;
  const ProbeIntent &candidate = probe_intents[id.value];
  return candidate.id == id ? &candidate : nullptr;
}

bool ObservationPlan::valid() const {
  if (mode_probe_vocabulary(mode) == nullptr)
    return false;
  for (size_t index = 0; index < probe_intents.size(); ++index) {
    const ProbeIntent &probe = probe_intents[index];
    if (probe.id.value != index || probe.mode != mode || !probe.source_site.valid() ||
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
    if (probe.kind == ProbeIntentKind::AtomicAddressCapture) {
      if (probe.position != ProbePosition::Before ||
          probe.dynamic_result != DynamicResultRequirement::None || !probe.atomic_lowering_form ||
          !probe.atomic_lowering_form->is_well_formed()) {
        return false;
      }
    } else if (synchronization_intent && probe.position != ProbePosition::After) {
      return false;
    } else if (!synchronization_intent && probe.dynamic_result != DynamicResultRequirement::None) {
      return false;
    } else if (probe.atomic_lowering_form) {
      return false;
    }
  }
  for (const SiteDecision &decision : site_decisions) {
    if (!decision.semantic_site.valid() || !valid_decision_kind(decision.kind) ||
        !valid_access_reason(decision.reason)) {
      return false;
    }
    const bool admitted = decision.kind == SiteDecisionKind::Admitted;
    if (admitted != (decision.reason == AccessPolicyReason::None))
      return false;
  }
  for (const BarrierSiteDecision &decision : barrier_site_decisions) {
    if (!decision.semantic_site.valid() ||
        decision.semantic_site.domain != SemanticSiteDomain::SynchronizationEvent ||
        !valid_decision_kind(decision.kind) || !valid_barrier_reason(decision.reason)) {
      return false;
    }
    const bool admitted = decision.kind == SiteDecisionKind::Admitted;
    if (admitted != (decision.reason == BarrierPolicyReason::None))
      return false;
  }
  for (const AtomicSiteDecision &decision : atomic_site_decisions) {
    if (!decision.semantic_site.valid() ||
        decision.semantic_site.domain != SemanticSiteDomain::SynchronizationEvent ||
        !valid_decision_kind(decision.kind) || !valid_capability_disposition(decision.capability) ||
        !valid_atomic_reason(decision.reason)) {
      return false;
    }
    const bool admitted = decision.kind == SiteDecisionKind::Admitted;
    if (admitted != (decision.reason == AtomicPolicyReason::None) ||
        (admitted && decision.capability != CapabilityDisposition::Supported &&
         decision.capability != CapabilityDisposition::AssociatedOnly)) {
      return false;
    }
  }
  for (const FenceSiteDecision &decision : fence_site_decisions) {
    if (!decision.semantic_site.valid() ||
        decision.semantic_site.domain != SemanticSiteDomain::SynchronizationEvent ||
        !valid_decision_kind(decision.kind) || !valid_capability_disposition(decision.capability) ||
        !valid_fence_reason(decision.reason) ||
        !valid_fence_association(decision.inventory_association)) {
      return false;
    }
    const bool admitted = decision.kind == SiteDecisionKind::Admitted;
    if (admitted != (decision.reason == FencePolicyReason::None) ||
        (admitted && (decision.inventory_association != FenceAssociation::Qualified ||
                      (decision.capability != CapabilityDisposition::Supported &&
                       decision.capability != CapabilityDisposition::AssociatedOnly)))) {
      return false;
    }
  }
  return true;
}

bool ObservationPlan::append(const ObservationPlan &fragment) {
  if (!valid() || !fragment.valid() || mode != fragment.mode)
    return false;
  if (probe_intents.size() > ProbeIntentId::invalid_value - fragment.probe_intents.size())
    return false;

  ObservationPlan combined = *this;
  const uint32_t intent_base = static_cast<uint32_t>(combined.probe_intents.size());
  for (ProbeIntent probe : fragment.probe_intents) {
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

CoverageLedger::CoverageLedger(ObservationPlan plan) : observation_plan_(std::move(plan)) {
  intent_entries_.reserve(observation_plan_.probe_intents.size());
  for (const ProbeIntent &intent : observation_plan_.probe_intents) {
    intent_entries_.push_back({
        .intent_id = intent.id,
        .lowering = LoweringOutcomeKind::Pending,
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

const IntentCoverageEntry *CoverageLedger::intent_entry(ProbeIntentId id) const {
  if (!id.valid() || id.value >= intent_entries_.size())
    return nullptr;
  const IntentCoverageEntry &entry = intent_entries_[id.value];
  return entry.intent_id == id ? &entry : nullptr;
}

template <typename ResolveIntent>
bool runtime_static_mapping_matches_commit(const CommittedLowering &commit,
                                           ResolveIntent resolve_intent);

template <typename ResolveIntent>
bool committed_lowering_is_valid(const CommittedLowering &commit, ResolveIntent resolve_intent);

std::optional<CommittedLowering>
make_committed_lowering(const ObservationPlan &plan, std::span<const ProbeIntentId> intent_ids,
                        std::span<const CommittedLoweringLocation> locations,
                        LoweringOutcomeKind outcome, std::string detail,
                        RuntimeStaticMapping runtime_mapping,
                        std::optional<RegisterPlanReason> resource_rejection_reason) {
  CommittedLowering commit{
      .intent_ids = {},
      .locations = std::vector(locations.begin(), locations.end()),
      .runtime_mapping = std::move(runtime_mapping),
      .outcome = outcome,
      .resource_rejection_reason = resource_rejection_reason,
      .detail = std::move(detail),
  };
  commit.intent_ids.reserve(intent_ids.size());
  for (ProbeIntentId id : intent_ids) {
    const ProbeIntent *intent = plan.intent(id);
    if (intent == nullptr)
      return std::nullopt;
    commit.intent_ids.push_back(id);
  }
  if (!committed_lowering_is_valid(commit, [&](ProbeIntentId id) { return plan.intent(id); }))
    return std::nullopt;
  return commit;
}

template <typename ResolveIntent>
bool runtime_static_mapping_matches_commit(const CommittedLowering &commit,
                                           ResolveIntent resolve_intent) {
  if (commit.outcome != LoweringOutcomeKind::Instrumented)
    return commit.runtime_mapping.empty();

  std::vector<ProbeIntentId> mapped_intents;
  const auto owns_physical_site = [&](const PhysicalSiteId &site) {
    return std::ranges::any_of(commit.intent_ids, [&](ProbeIntentId id) {
      const ProbeIntent *intent = resolve_intent(id);
      return intent != nullptr && intent->physical_site == site;
    });
  };
  const auto valid_attribution = [&](const StaticAccessAttribution &access,
                                     ProbeIntentKind expected_kind) {
    if (access.intent_ids.empty() || !access.original_site.valid() ||
        !owns_physical_site(access.original_site)) {
      return false;
    }
    if (access.owner_provenance_complete && access.execution_owner_kernel_ids.empty()) {
      return false;
    }
    for (ProgramContainerId owner : access.execution_owner_kernel_ids) {
      if (!owner.valid() || std::ranges::count(access.execution_owner_kernel_ids, owner) != 1)
        return false;
    }

    std::vector<SemanticSiteId> expected_semantic_sites;
    for (ProbeIntentId id : access.intent_ids) {
      if (std::ranges::find(commit.intent_ids, id) == commit.intent_ids.end() ||
          std::ranges::find(mapped_intents, id) != mapped_intents.end()) {
        return false;
      }
      const ProbeIntent *intent = resolve_intent(id);
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

  if (const auto *mappings = commit.runtime_mapping.accesses()) {
    for (const StaticAccessMapping &mapping : *mappings) {
      if (!valid_attribution(mapping.access, ProbeIntentKind::Access) ||
          mapping.range_count == 0u || mapping.bank_count == 0u) {
        return false;
      }
    }
  }
  for (ProbeIntentId id : commit.intent_ids) {
    const ProbeIntent *intent = resolve_intent(id);
    if (intent == nullptr)
      return false;
    const size_t mapping_count = std::ranges::count(mapped_intents, id);
    if (intent->kind == ProbeIntentKind::Access && mapping_count != 1u) {
      return false;
    }
  }
  return true;
}

template <typename ResolveIntent>
bool committed_lowering_is_valid(const CommittedLowering &commit, ResolveIntent resolve_intent) {
  if (commit.intent_ids.empty() || commit.outcome == LoweringOutcomeKind::Pending ||
      !valid_lowering_outcome(commit.outcome) ||
      !valid_resource_rejection_reason(commit.resource_rejection_reason) ||
      (commit.resource_rejection_reason &&
       commit.outcome != LoweringOutcomeKind::ResourceRejected)) {
    return false;
  }
  const bool instrumented = commit.outcome == LoweringOutcomeKind::Instrumented;
  if (instrumented != !commit.locations.empty() ||
      !runtime_static_mapping_matches_commit(commit, resolve_intent)) {
    return false;
  }
  if (instrumented) {
    for (const CommittedLoweringLocation &location : commit.locations) {
      if (!location.original_site.valid() || location.emitted_size == 0u ||
          std::ranges::none_of(commit.intent_ids, [&](ProbeIntentId id) {
            const ProbeIntent *intent = resolve_intent(id);
            return intent != nullptr && intent->physical_site == location.original_site;
          })) {
        return false;
      }
    }
  }
  for (ProbeIntentId id : commit.intent_ids) {
    const ProbeIntent *intent = resolve_intent(id);
    if (intent == nullptr || std::ranges::count(commit.intent_ids, id) != 1 ||
        (instrumented && std::ranges::none_of(commit.locations, [&](const auto &location) {
           return location.original_site == intent->physical_site;
         }))) {
      return false;
    }
  }
  return true;
}

bool CoverageLedger::publish_lowering_commit(CommittedLowering commit) {
  if (!committed_lowering_is_valid(
          commit, [&](ProbeIntentId id) { return observation_plan_.intent(id); })) {
    return false;
  }
  for (ProbeIntentId id : commit.intent_ids) {
    const IntentCoverageEntry *entry = intent_entry(id);
    if (entry == nullptr || entry->lowering != LoweringOutcomeKind::Pending)
      return false;
  }
  for (ProbeIntentId id : commit.intent_ids) {
    IntentCoverageEntry &entry = intent_entries_[id.value];
    entry.lowering = commit.outcome;
    entry.resource_rejection_reason = commit.resource_rejection_reason;
    entry.detail = commit.detail;
  }
  lowering_commits_.push_back(std::move(commit));
  return true;
}

bool CoverageLedger::publish_lowering_rejection(
    std::span<const ProbeIntentId> intent_ids, LoweringOutcomeKind outcome, std::string detail,
    std::optional<RegisterPlanReason> resource_rejection_reason) {
  auto commit = make_committed_lowering(
      observation_plan_, intent_ids, std::span<const CommittedLoweringLocation>{}, outcome,
      std::move(detail), RuntimeStaticMapping{}, resource_rejection_reason);
  return commit && publish_lowering_commit(std::move(*commit));
}

bool CoverageLedger::publish_lowering_commits(std::vector<CommittedLowering> commits) {
  CoverageLedger next = *this;
  for (CommittedLowering &commit : commits) {
    if (!next.publish_lowering_commit(std::move(commit)))
      return false;
  }
  *this = std::move(next);
  return true;
}

bool CoverageLedger::publish_coalescing_instrumented_commits(
    std::vector<CommittedLowering> commits) {
  CoverageLedger next = *this;
  const auto append_unique = [](auto &destination, const auto &source) {
    for (const auto &value : source) {
      if (std::ranges::find(destination, value) == destination.end())
        destination.push_back(value);
    }
  };
  for (CommittedLowering &incoming : commits) {
    if (incoming.outcome != LoweringOutcomeKind::Instrumented ||
        !committed_lowering_is_valid(
            incoming, [&](ProbeIntentId id) { return next.observation_plan_.intent(id); })) {
      return false;
    }
    for (size_t index = 0; index < next.lowering_commits_.size();) {
      CommittedLowering &accepted = next.lowering_commits_[index];
      const bool overlaps = std::ranges::any_of(accepted.intent_ids, [&](const auto &id) {
        return std::ranges::find(incoming.intent_ids, id) != incoming.intent_ids.end();
      });
      if (!overlaps) {
        ++index;
        continue;
      }
      if (accepted.outcome != LoweringOutcomeKind::Instrumented)
        return false;
      append_unique(incoming.intent_ids, accepted.intent_ids);
      append_unique(incoming.locations, accepted.locations);
      if (!incoming.runtime_mapping.append(std::move(accepted.runtime_mapping)))
        return false;
      if (incoming.detail.empty())
        incoming.detail = std::move(accepted.detail);
      for (ProbeIntentId id : accepted.intent_ids) {
        IntentCoverageEntry &entry = next.intent_entries_[id.value];
        entry.lowering = LoweringOutcomeKind::Pending;
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

bool CoverageLedger::publish_replacing_instrumented_commits(
    std::vector<CommittedLowering> commits) {
  CoverageLedger next = *this;
  std::vector<ProbeIntentId> replacement_intents;
  for (const CommittedLowering &commit : commits) {
    for (ProbeIntentId id : commit.intent_ids) {
      if (std::ranges::find(replacement_intents, id) == replacement_intents.end())
        replacement_intents.push_back(id);
    }
  }

  for (size_t index = 0; index < next.lowering_commits_.size();) {
    const CommittedLowering &accepted = next.lowering_commits_[index];
    const bool overlaps = std::ranges::any_of(accepted.intent_ids, [&](const auto &id) {
      return std::ranges::find(replacement_intents, id) != replacement_intents.end();
    });
    if (!overlaps) {
      ++index;
      continue;
    }
    if (accepted.outcome != LoweringOutcomeKind::Instrumented ||
        std::ranges::any_of(accepted.intent_ids, [&](const auto &id) {
          return std::ranges::find(replacement_intents, id) == replacement_intents.end();
        })) {
      return false;
    }
    for (ProbeIntentId id : accepted.intent_ids) {
      IntentCoverageEntry &entry = next.intent_entries_[id.value];
      entry.lowering = LoweringOutcomeKind::Pending;
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

RuntimeStaticMapping CoverageLedger::runtime_static_mapping() const {
  RuntimeStaticMapping mapping;
  for (const CommittedLowering &commit : lowering_commits_) {
    if (!mapping.append(commit.runtime_mapping))
      return {};
  }
  return mapping;
}

void CoverageLedger::discard_instrumented_lowerings() {
  std::erase_if(lowering_commits_, [](const CommittedLowering &commit) {
    return commit.outcome == LoweringOutcomeKind::Instrumented;
  });
  for (IntentCoverageEntry &entry : intent_entries_) {
    entry.lowering = LoweringOutcomeKind::Pending;
    entry.resource_rejection_reason.reset();
    entry.detail.clear();
  }
  for (const CommittedLowering &commit : lowering_commits_) {
    for (ProbeIntentId id : commit.intent_ids) {
      IntentCoverageEntry &entry = intent_entries_[id.value];
      entry.lowering = commit.outcome;
      entry.resource_rejection_reason = commit.resource_rejection_reason;
      entry.detail = commit.detail;
    }
  }
}

AccessPolicyResult plan_access_observation(const ProgramInventory &inventory,
                                           const AccessPolicyRequest &request) {
  AccessPolicyResult result;
  result.plan.mode = request.mode;
  const ModeProbeVocabulary *vocabulary = mode_probe_vocabulary(request.mode);
  if (vocabulary == nullptr || inventory.empty())
    return result;

  std::map<uint64_t, std::vector<const ProgramSite *>> aliases_by_offset;
  for (const ProgramSite &access : inventory.access_sites())
    aliases_by_offset[access.physical_id.original_text_offset].push_back(&access);
  const PhysicalSitesByOffset synchronization_reservations =
      index_physical_sites_by_offset(request.reserved_for_synchronization);

  for (const auto &[offset, aliases] : aliases_by_offset) {
    (void)offset;
    const ProgramSite &access = *aliases.front();
    const std::vector<SemanticSiteId> ids = semantic_ids(access);
    // The aliases were already grouped by physical offset above. Derive their
    // presentation names from that group instead of rescanning the complete
    // program-site arena once per access.
    const std::vector<std::string> names = source_container_names(inventory, aliases);
    SiteDecisionKind decision_kind = SiteDecisionKind::NotApplicable;
    AccessPolicyReason reason = AccessPolicyReason::AccessFamilyDisabled;

    const bool conflicting_alias = std::ranges::any_of(aliases, [&](const auto *alias) {
      return !access_alias_semantics_equal(inventory, access, *alias);
    });
    const bool flat = access.origin == AccessOrigin::Flat;
    const CapabilityForm form =
        flat ? CapabilityForm::GroupFlatAccess : CapabilityForm::NativeLdsAccess;
    const bool enabled = flat ? request.group_flat_enabled : request.native_lds_enabled;
    const std::vector<uint64_t> owner_descriptors = inventory.execution_owner_descriptors(aliases);

    if (!contains_substring(inventory, aliases, request.container_filter) ||
        !site_matches_kernel_allowlist(inventory, owner_descriptors, names,
                                       request.kernel_name_allowlist)) {
      reason = AccessPolicyReason::ContainerFilterExcluded;
    } else if (contains_physical_site(synchronization_reservations, access.physical_id)) {
      reason = AccessPolicyReason::ReservedForSynchronizationPolicy;
    } else if (!enabled) {
      reason = AccessPolicyReason::AccessFamilyDisabled;
    } else if (!arch_supports_capability_form(inventory.arch(), form) ||
               capability_disposition(inventory.target(), request.mode, form) !=
                   CapabilityDisposition::Supported) {
      reason = AccessPolicyReason::TargetCapabilityUnavailable;
    } else if (access.kind != LdsAccessKind::Read && access.kind != LdsAccessKind::Write &&
               !(request.mode != Mode::SuperCollider && !flat &&
                 access.kind == LdsAccessKind::Atomic &&
                 arch_supports_capability_form(inventory.arch(),
                                               CapabilityForm::RelaxedLdsAtomicAccess) &&
                 access.lowering.operation(AccessLoweringOperation::ReplayGuestAccess)
                     .available())) {
      reason = AccessPolicyReason::OperationKindExcluded;
    } else if (flat && access.address_space == AccessAddressSpace::NonGroup) {
      reason = AccessPolicyReason::NonGroupAddressSpace;
    } else if (flat && access.address_space == AccessAddressSpace::Unresolved) {
      reason = AccessPolicyReason::FlatProvenancePolicyExcluded;
    } else if (flat && access.confidence != SemanticConfidence::Exact &&
               request.flat_provenance_mode == FlatProvenanceMode::Strict) {
      reason = AccessPolicyReason::FlatProvenancePolicyExcluded;
    } else if (conflicting_alias) {
      decision_kind = SiteDecisionKind::Unsupported;
      reason = AccessPolicyReason::ConflictingPhysicalAliases;
      result.errors.push_back(reason);
    } else {
      const AccessLoweringOperation operation = request.mode == Mode::SuperCollider
                                                    ? AccessLoweringOperation::CompareObservedValue
                                                    : AccessLoweringOperation::ReplayGuestAccess;
      reason = classified_operation_reason(access, operation);
      decision_kind = reason == AccessPolicyReason::None ? SiteDecisionKind::Admitted
                                                         : SiteDecisionKind::Unsupported;
    }

    if (decision_kind == SiteDecisionKind::Admitted) {
      const ProbeIntentId intent_id{static_cast<uint32_t>(result.plan.probe_intents.size())};
      result.plan.probe_intents.push_back({
          .id = intent_id,
          .mode = request.mode,
          .source_site = access.id,
          .physical_site = access.physical_id,
          .covered_semantic_sites = ids,
          .kind = vocabulary->access,
          .position = ProbePosition::Before,
          .synchronization_association = std::nullopt,
          .dynamic_result = DynamicResultRequirement::None,
          .atomic_lowering_form = std::nullopt,
      });
    }
    for (const SemanticSiteId &id : ids) {
      SiteDecision decision{
          .semantic_site = id,
          .kind = decision_kind,
          .reason = reason,
      };
      result.plan.site_decisions.push_back(std::move(decision));
    }
  }
  return result;
}

} // namespace rocjitsu::consan
