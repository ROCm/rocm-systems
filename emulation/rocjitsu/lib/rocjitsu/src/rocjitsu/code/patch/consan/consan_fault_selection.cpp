// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_fault_selection.h"

#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"

#include <algorithm>
#include <iterator>
#include <ranges>
#include <span>
#include <utility>

namespace rocjitsu::consan {
namespace {

[[nodiscard]] const ProgramContainer *requested_kernel_owner(const ProgramInventory &inventory,
                                                             std::string_view filter) {
  if (filter.empty())
    return nullptr;
  if (const ProgramContainer *exact = inventory.find_kernel_by_name(filter))
    return exact;
  const ProgramContainer *match = nullptr;
  for (const ProgramContainer &kernel : inventory.kernels()) {
    if (kernel.name.find(filter) == std::string::npos)
      continue;
    if (match != nullptr)
      return nullptr;
    match = &kernel;
  }
  return match;
}

[[nodiscard]] std::pair<uint64_t, uint64_t>
exact_barrier_drop_pair_range(const FaultSelectionView &inventory,
                              const ExactBarrierDropPair &pair) {
  const ProgramSite *primary = inventory.source(*pair.primary);
  const ProgramSite *companion =
      pair.companion == nullptr ? primary : inventory.source(*pair.companion);
  if (primary == nullptr || companion == nullptr)
    return {};
  const uint64_t begin = std::min(primary->text_offset(), companion->text_offset());
  const uint64_t end = std::max(primary->text_offset() + primary->size(),
                                companion->text_offset() + companion->size());
  return {begin, end};
}

constexpr auto kExactBarrierDropPairIssues = make_enum_vocabulary(
    "invalid exact barrier-drop pair issue", enum_entry(ExactBarrierDropPairIssue::None, ""),
    enum_entry(ExactBarrierDropPairIssue::MissingExactIdentity,
               "an exact site identity and logical sequence identity are both required"),
    enum_entry(ExactBarrierDropPairIssue::SequenceNotFound,
               "the exact logical sequence identity was not found"),
    enum_entry(ExactBarrierDropPairIssue::SequenceNotQualified,
               "the exact sequence is not an owned complete conservative two-member barrier"),
    enum_entry(ExactBarrierDropPairIssue::PrimaryNotMember,
               "the exact site is not a member of the requested logical barrier"),
    enum_entry(ExactBarrierDropPairIssue::MemberSiteMissing,
               "a logical barrier member has no exact owned patch site"),
    enum_entry(ExactBarrierDropPairIssue::MemberSiteAmbiguous,
               "a logical barrier member maps to multiple patch sites"),
    enum_entry(ExactBarrierDropPairIssue::InvalidPairGeometry,
               "the exact logical barrier does not have two distinct one-word patch sites"));

} // namespace

bool fault_admits_cross_block_barrier_move(const BarrierMoveDestination &destination,
                                           const MutationRequest &mutation) {
  if (mutation.fault_barrier_move_direction != BarrierMoveDirection::Earlier ||
      !destination.structured_guard_block_index || !destination.structured_source_block_index ||
      !destination.structured_guard_offset || !destination.structured_source_offset) {
    return false;
  }
  switch (destination.cfg_contract) {
  case BarrierMoveCfgContract::SameBlock:
    return false;
  case BarrierMoveCfgContract::CompletingStructuredDiamond:
    return mutation.fault_allow_completing_conditional_barrier_move;
  case BarrierMoveCfgContract::DestructiveStructuredExecDiamond:
    return mutation.fault_allow_destructive_divergent_barrier_move;
  }
  return false;
}

const FaultSite *find_fault_site_by_identity(const FaultSelectionView &inventory,
                                             std::string_view identity, FaultSiteKind kind) {
  const auto site = std::ranges::find_if(inventory.fault_sites, [&](const FaultSite &candidate) {
    return candidate.kind == kind && candidate.identity == identity;
  });
  return site == inventory.fault_sites.end() ? nullptr : &*site;
}

bool execution_owners_include_requested_kernel(std::span<const ExecutionOwner> owners,
                                               const FaultSelectionView &inventory,
                                               std::string_view kernel_name_filter) {
  if (kernel_name_filter.empty())
    return true;
  const ProgramContainer *kernel =
      requested_kernel_owner(inventory.program_inventory, kernel_name_filter);
  return kernel != nullptr &&
         std::ranges::find(owners, kernel->id, &ExecutionOwner::kernel) != owners.end();
}

const FaultSite *select_fault_site_for_plan(const FaultSelectionView &inventory,
                                            const FaultSelection &selection, FaultSiteKind kind) {
  if (!selection.primary_site_identity.empty()) {
    const FaultSite *site =
        find_fault_site_by_identity(inventory, selection.primary_site_identity, kind);
    return site != nullptr &&
                   execution_owners_include_requested_kernel(
                       inventory.execution_owners(*site), inventory, selection.kernel_name_filter)
               ? site
               : nullptr;
  }
  uint32_t index = 0;
  for (const FaultSite &site : inventory.fault_sites) {
    if (site.kind != kind ||
        !execution_owners_include_requested_kernel(inventory.execution_owners(site), inventory,
                                                   selection.kernel_name_filter))
      continue;
    if (index++ == selection.ordinal)
      return &site;
  }
  return nullptr;
}

std::optional<OrdinaryAcquireMutationTarget>
select_ordinary_acquire_mutation_target(const FaultSelectionView &inventory,
                                        const FaultSelection &selection) {
  const SynchronizationInventoryView sync = inventory.program_inventory.sync();
  const auto qualify = [&](const FaultSite &site) -> std::optional<OrdinaryAcquireMutationTarget> {
    const ProgramSite *source = inventory.source(site);
    const OrdinaryMemorySite *memory =
        source == nullptr ? nullptr : source->get_if<OrdinaryMemorySite>();
    if (site.kind != FaultSiteKind::OrdinaryMemory || memory == nullptr ||
        memory->support_reason != OrdinaryMemorySupportReason::Supported ||
        memory->mnemonic != "global_load_b32" ||
        !execution_owners_include_requested_kernel(inventory.execution_owners(site), inventory,
                                                   selection.kernel_name_filter))
      return std::nullopt;
    const SyncEvent *load = sync.find_event(site.source_site);
    const SyncSequence *sequence = sync.find_unique_sequence_containing(site.source_site);
    if (load == nullptr || sequence == nullptr)
      return std::nullopt;
    const std::vector<ExecutionOwner> sequence_owners = sync.execution_owners(*sequence);
    if (load->operation != SyncOperation::OrdinaryLoad || !load->scope ||
        !memory_scope_is_agent_or_system(*load->scope) ||
        sequence->kind != SyncKind::OrdinaryMemory ||
        sequence->operation != SyncOperation::OrdinaryLoad ||
        sequence->memory_role != SyncMemoryRole::Acquire ||
        !sync_confidence_meets(sequence->memory_role_confidence,
                               SemanticConfidence::Conservative) ||
        !sequence->basic_block_index || sequence->member_event_ids.size() != 2u ||
        !sequence_has_exact_members(sync, *sequence) ||
        sync.find_event(sequence->member_event_ids.front()) != load ||
        !nonempty_execution_owners_equal(sequence_owners, inventory.execution_owners(site)))
      return std::nullopt;
    const SyncEvent *cache = sync.find_event(sequence->member_event_ids.back());
    const FenceSite *cache_source = cache == nullptr ? nullptr : sync.source_as<FenceSite>(*cache);
    if (cache_source == nullptr || cache->kind != SyncKind::Fence ||
        cache_source->cache_operation != CacheOperation::Acquire ||
        !cache_source->ordinary_acquire_mutation_supported || !sync.same_container(*cache, *load) ||
        !nonempty_execution_owners_equal(sync.execution_owners(*cache),
                                         sync.execution_owners(*load)) ||
        cache_source->size == 0u || cache_source->size % sizeof(uint32_t) != 0u)
      return std::nullopt;
    return OrdinaryAcquireMutationTarget{&site, load, cache, sequence};
  };

  if (!selection.primary_site_identity.empty()) {
    const FaultSite *site = find_fault_site_by_identity(inventory, selection.primary_site_identity,
                                                        FaultSiteKind::OrdinaryMemory);
    return site == nullptr ? std::nullopt : qualify(*site);
  }
  uint32_t index = 0;
  for (const FaultSite &site : inventory.fault_sites) {
    auto target = qualify(site);
    if (target && index++ == selection.ordinal)
      return target;
  }
  return std::nullopt;
}

ExactBarrierDropPairResolution resolve_exact_barrier_drop_pair(const FaultSelectionView &inventory,
                                                               const FaultSelection &selection,
                                                               bool allow_full_singleton) {
  using Issue = ExactBarrierDropPairIssue;
  const SynchronizationInventoryView sync = inventory.program_inventory.sync();
  if (selection.primary_site_identity.empty() || selection.primary_sequence_identity.empty()) {
    return {.pair = std::nullopt, .issue = Issue::MissingExactIdentity};
  }
  const auto sequence = std::ranges::find(sync.sync_sequences, selection.primary_sequence_identity,
                                          &SyncSequence::identity);
  if (sequence == sync.sync_sequences.end()) {
    return {.pair = std::nullopt, .issue = Issue::SequenceNotFound};
  }
  const std::vector<ExecutionOwner> sequence_owners = sync.execution_owners(*sequence);
  if (sequence->kind != SyncKind::Barrier || sequence->operation != SyncOperation::BarrierFull ||
      !sync_confidence_meets(sequence->confidence, SemanticConfidence::Conservative) ||
      (sequence->member_event_ids.size() != 2u &&
       !(allow_full_singleton && sequence->member_event_ids.size() == 1u)) ||
      !sequence_has_exact_members(sync, *sequence) ||
      !execution_owners_include_requested_kernel(sequence_owners, inventory,
                                                 selection.kernel_name_filter)) {
    return {.pair = std::nullopt, .issue = Issue::SequenceNotQualified};
  }

  const FaultSite *primary = find_fault_site_by_identity(inventory, selection.primary_site_identity,
                                                         FaultSiteKind::Barrier);
  const SyncEvent *primary_event =
      primary == nullptr ? nullptr : sync.find_event(primary->source_site);
  SyncEventId primary_member_id;
  if (primary_event != nullptr)
    primary_member_id = sync.event_id(*primary_event);
  if (primary == nullptr || primary_event == nullptr ||
      std::ranges::find(sequence->member_event_ids, primary_member_id) ==
          sequence->member_event_ids.end() ||
      !execution_owners_include_requested_kernel(inventory.execution_owners(*primary), inventory,
                                                 selection.kernel_name_filter)) {
    return {.pair = std::nullopt, .issue = Issue::PrimaryNotMember};
  }

  const FaultSite *companion = nullptr;
  const ProgramContainer *sequence_container = sync.container(*sequence);
  if (sequence_container == nullptr)
    return {.pair = std::nullopt, .issue = Issue::SequenceNotQualified};
  for (SyncEventId member_id : sequence->member_event_ids) {
    const SyncEvent *member = sync.find_event(member_id);
    if (member == nullptr)
      return {.pair = std::nullopt, .issue = Issue::MemberSiteMissing};
    const auto matching_site =
        std::ranges::find_if(inventory.fault_sites, [&](const FaultSite &candidate) {
          const ProgramSite *candidate_source = inventory.source(candidate);
          return candidate.kind == FaultSiteKind::Barrier &&
                 sync.find_event(candidate.source_site) == member && candidate_source != nullptr &&
                 candidate_source->container == sequence_container->id &&
                 execution_owners_include_requested_kernel(inventory.execution_owners(candidate),
                                                           inventory, selection.kernel_name_filter);
        });
    if (matching_site == inventory.fault_sites.end()) {
      return {.pair = std::nullopt, .issue = Issue::MemberSiteMissing};
    }
    const auto duplicate = std::ranges::find_if(
        std::next(matching_site), inventory.fault_sites.end(), [&](const FaultSite &candidate) {
          return candidate.kind == FaultSiteKind::Barrier &&
                 sync.find_event(candidate.source_site) == member;
        });
    if (duplicate != inventory.fault_sites.end()) {
      return {.pair = std::nullopt, .issue = Issue::MemberSiteAmbiguous};
    }
    if (matching_site->identity != primary->identity)
      companion = &*matching_site;
  }
  const ProgramSite *primary_source = inventory.source(*primary);
  const ProgramSite *companion_source =
      companion == nullptr ? primary_source : inventory.source(*companion);
  if (primary_source == nullptr || companion_source == nullptr ||
      primary_source->size() != sizeof(uint32_t) || companion_source->size() != sizeof(uint32_t) ||
      (companion != nullptr &&
       primary_source->decoded_file_offset() == companion_source->decoded_file_offset())) {
    return {.pair = std::nullopt, .issue = Issue::InvalidPairGeometry};
  }
  return {.pair = ExactBarrierDropPair{&*sequence, primary, companion}, .issue = Issue::None};
}

ExactBarrierDropGroupResolution
resolve_exact_barrier_drop_group(const FaultSelectionView &inventory,
                                 const FaultSelection &selection) {
  using Issue = ExactBarrierDropGroupIssue;
  if (selection.companion_site_identity.empty() || selection.companion_sequence_identity.empty()) {
    return {.group = std::nullopt,
            .issue = Issue::MissingCompanionIdentity,
            .member_issue = ExactBarrierDropPairIssue::None};
  }
  FaultSelection first_selection = selection;
  first_selection.companion_site_identity = {};
  first_selection.companion_sequence_identity = {};
  const auto first = resolve_exact_barrier_drop_pair(inventory, first_selection, true);
  if (!first.pair)
    return {.group = std::nullopt, .issue = Issue::FirstPairRejected, .member_issue = first.issue};
  FaultSelection second_selection = selection;
  second_selection.primary_site_identity = selection.companion_site_identity;
  second_selection.primary_sequence_identity = selection.companion_sequence_identity;
  second_selection.companion_site_identity = {};
  second_selection.companion_sequence_identity = {};
  const auto second = resolve_exact_barrier_drop_pair(inventory, second_selection, true);
  if (!second.pair)
    return {
        .group = std::nullopt, .issue = Issue::SecondPairRejected, .member_issue = second.issue};
  const auto first_range = exact_barrier_drop_pair_range(inventory, *first.pair);
  const auto second_range = exact_barrier_drop_pair_range(inventory, *second.pair);
  const SynchronizationInventoryView sync = inventory.program_inventory.sync();
  const std::vector<ExecutionOwner> first_owners = sync.execution_owners(*first.pair->sequence);
  const std::vector<ExecutionOwner> second_owners = sync.execution_owners(*second.pair->sequence);
  if (first.pair->sequence->identity == second.pair->sequence->identity ||
      first_range.second > second_range.first) {
    return {.group = std::nullopt,
            .issue = Issue::PairsOverlapOrUnordered,
            .member_issue = ExactBarrierDropPairIssue::None};
  }
  if (!sync.same_container(*first.pair->sequence, *second.pair->sequence) ||
      !nonempty_execution_owners_equal(first_owners, second_owners)) {
    return {.group = std::nullopt,
            .issue = Issue::PairsHaveDifferentOwners,
            .member_issue = ExactBarrierDropPairIssue::None};
  }
  return {.group = ExactBarrierDropGroup{*first.pair, *second.pair},
          .issue = Issue::None,
          .member_issue = ExactBarrierDropPairIssue::None};
}

std::string_view exact_barrier_drop_pair_issue_message(ExactBarrierDropPairIssue issue) {
  return kExactBarrierDropPairIssues.name(issue);
}

std::string exact_barrier_drop_group_issue_message(ExactBarrierDropGroupIssue issue,
                                                   ExactBarrierDropPairIssue member_issue) {
  using Issue = ExactBarrierDropGroupIssue;
  switch (issue) {
  case Issue::None:
    return {};
  case Issue::MissingCompanionIdentity:
    return "a grouped drop requires an exact companion site and sequence identity";
  case Issue::FirstPairRejected:
    return "group member zero is invalid: " +
           std::string(exact_barrier_drop_pair_issue_message(member_issue));
  case Issue::SecondPairRejected:
    return "group member one is invalid: " +
           std::string(exact_barrier_drop_pair_issue_message(member_issue));
  case Issue::PairsOverlapOrUnordered:
    return "the two exact pairs are duplicate, overlapping, or not strictly ordered";
  case Issue::PairsHaveDifferentOwners:
    return "the two exact pairs do not have identical container and execution owners";
  case Issue::Count:
    break;
  }
  return "invalid exact barrier-drop group issue";
}

std::string exact_barrier_drop_group_identity(const ExactBarrierDropGroup &group) {
  return "barrier-group[" + group.first.sequence->identity + "][" +
         group.second.sequence->identity + "]";
}

} // namespace rocjitsu::consan
