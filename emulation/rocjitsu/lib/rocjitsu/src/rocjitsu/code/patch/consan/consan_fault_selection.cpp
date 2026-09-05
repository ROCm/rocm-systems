// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_fault_selection.h"

#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"

#include <algorithm>
#include <iterator>
#include <ranges>
#include <span>
#include <utility>

namespace rocjitsu {
namespace {

[[nodiscard]] const ConSanProgramContainer *
requested_kernel_owner(const ProgramInventory &inventory, std::string_view filter) {
  if (filter.empty())
    return nullptr;
  if (const ConSanProgramContainer *exact = inventory.find_kernel_by_name(filter))
    return exact;
  const ConSanProgramContainer *match = nullptr;
  for (const ConSanProgramContainer &kernel : inventory.kernels()) {
    if (kernel.name.find(filter) == std::string::npos)
      continue;
    if (match != nullptr)
      return nullptr;
    match = &kernel;
  }
  return match;
}

[[nodiscard]] std::pair<uint64_t, uint64_t>
exact_barrier_drop_pair_range(const ConSanFaultSelectionView &inventory,
                              const ExactBarrierDropPair &pair) {
  const ConSanProgramSite *primary = inventory.source(*pair.primary);
  const ConSanProgramSite *companion = inventory.source(*pair.companion);
  if (primary == nullptr || companion == nullptr)
    return {};
  const uint64_t begin = std::min(primary->text_offset(), companion->text_offset());
  const uint64_t end = std::max(primary->text_offset() + primary->size(),
                                companion->text_offset() + companion->size());
  return {begin, end};
}

constexpr auto kExactBarrierDropPairIssues = make_consan_enum_vocabulary(
    "invalid exact barrier-drop pair issue", consan_enum(ExactBarrierDropPairIssue::None, ""),
    consan_enum(ExactBarrierDropPairIssue::MissingExactIdentity,
                "an exact site identity and logical sequence identity are both required"),
    consan_enum(ExactBarrierDropPairIssue::SequenceNotFound,
                "the exact logical sequence identity was not found"),
    consan_enum(ExactBarrierDropPairIssue::SequenceNotQualified,
                "the exact sequence is not an owned complete conservative two-member barrier"),
    consan_enum(ExactBarrierDropPairIssue::PrimaryNotMember,
                "the exact site is not a member of the requested logical barrier"),
    consan_enum(ExactBarrierDropPairIssue::MemberSiteMissing,
                "a logical barrier member has no exact owned patch site"),
    consan_enum(ExactBarrierDropPairIssue::MemberSiteAmbiguous,
                "a logical barrier member maps to multiple patch sites"),
    consan_enum(ExactBarrierDropPairIssue::InvalidPairGeometry,
                "the exact logical barrier does not have two distinct one-word patch sites"));

} // namespace

bool consan_fault_admits_cross_block_barrier_move(const ConSanBarrierMoveDestination &destination,
                                                  const MutationRequest &mutation) {
  if (mutation.fault_barrier_move_direction != ConSanBarrierMoveDirection::Earlier ||
      !destination.structured_guard_block_index || !destination.structured_source_block_index ||
      !destination.structured_guard_offset || !destination.structured_source_offset) {
    return false;
  }
  switch (destination.cfg_contract) {
  case ConSanBarrierMoveCfgContract::SameBlock:
    return false;
  case ConSanBarrierMoveCfgContract::CompletingStructuredDiamond:
    return mutation.fault_allow_completing_conditional_barrier_move;
  case ConSanBarrierMoveCfgContract::DestructiveStructuredExecDiamond:
    return mutation.fault_allow_destructive_divergent_barrier_move;
  }
  return false;
}

const ConSanFaultSite *find_fault_site_by_identity(const ConSanFaultSelectionView &inventory,
                                                   std::string_view identity,
                                                   ConSanFaultSiteKind kind) {
  const auto site =
      std::ranges::find_if(inventory.fault_sites, [&](const ConSanFaultSite &candidate) {
        return candidate.kind == kind && candidate.identity == identity;
      });
  return site == inventory.fault_sites.end() ? nullptr : &*site;
}

bool consan_execution_owners_include_requested_kernel(std::span<const ConSanExecutionOwner> owners,
                                                      const ConSanFaultSelectionView &inventory,
                                                      std::string_view kernel_name_filter) {
  if (kernel_name_filter.empty())
    return true;
  const ConSanProgramContainer *kernel =
      requested_kernel_owner(inventory.program_inventory, kernel_name_filter);
  return kernel != nullptr &&
         std::ranges::find(owners, kernel->id, &ConSanExecutionOwner::kernel) != owners.end();
}

const ConSanFaultSite *select_fault_site_for_plan(const ConSanFaultSelectionView &inventory,
                                                  const ConSanFaultSelection &selection,
                                                  ConSanFaultSiteKind kind) {
  if (!selection.primary_site_identity.empty()) {
    const ConSanFaultSite *site =
        find_fault_site_by_identity(inventory, selection.primary_site_identity, kind);
    return site != nullptr &&
                   consan_execution_owners_include_requested_kernel(
                       inventory.execution_owners(*site), inventory, selection.kernel_name_filter)
               ? site
               : nullptr;
  }
  uint32_t index = 0;
  for (const ConSanFaultSite &site : inventory.fault_sites) {
    if (site.kind != kind ||
        !consan_execution_owners_include_requested_kernel(inventory.execution_owners(site),
                                                          inventory, selection.kernel_name_filter))
      continue;
    if (index++ == selection.ordinal)
      return &site;
  }
  return nullptr;
}

std::optional<OrdinaryAcquireMutationTarget>
select_ordinary_acquire_mutation_target(const ConSanFaultSelectionView &inventory,
                                        const ConSanFaultSelection &selection) {
  const SynchronizationInventoryView sync = inventory.program_inventory.sync();
  const auto qualify =
      [&](const ConSanFaultSite &site) -> std::optional<OrdinaryAcquireMutationTarget> {
    const ConSanProgramSite *source = inventory.source(site);
    const ConSanOrdinaryMemorySite *memory =
        source == nullptr ? nullptr : source->get_if<ConSanOrdinaryMemorySite>();
    if (site.kind != ConSanFaultSiteKind::OrdinaryMemory || memory == nullptr ||
        memory->support_reason != ConSanOrdinaryMemorySupportReason::Supported ||
        memory->mnemonic != "global_load_b32" ||
        !consan_execution_owners_include_requested_kernel(inventory.execution_owners(site),
                                                          inventory, selection.kernel_name_filter))
      return std::nullopt;
    const ConSanSyncEvent *load = sync.find_event(site.source_site);
    const ConSanSyncSequence *sequence = sync.find_unique_sequence_containing(site.source_site);
    if (load == nullptr || sequence == nullptr)
      return std::nullopt;
    const std::vector<ConSanExecutionOwner> sequence_owners = sync.execution_owners(*sequence);
    if (load->operation != ConSanSyncOperation::OrdinaryLoad || !load->scope ||
        !consan_memory_scope_is_agent_or_system(*load->scope) ||
        sequence->kind != ConSanSyncKind::OrdinaryMemory ||
        sequence->operation != ConSanSyncOperation::OrdinaryLoad ||
        sequence->memory_role != ConSanSyncMemoryRole::Acquire ||
        !consan_sync_confidence_meets(sequence->memory_role_confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        !sequence->basic_block_index || sequence->member_event_ids.size() != 2u ||
        !sequence_has_exact_members(sync, *sequence) ||
        sync.find_event(sequence->member_event_ids.front()) != load ||
        !consan_nonempty_execution_owners_equal(sequence_owners, inventory.execution_owners(site)))
      return std::nullopt;
    const ConSanSyncEvent *cache = sync.find_event(sequence->member_event_ids.back());
    const ConSanFenceSite *cache_source =
        cache == nullptr ? nullptr : sync.source_as<ConSanFenceSite>(*cache);
    if (cache_source == nullptr || cache->kind != ConSanSyncKind::Fence ||
        cache_source->cache_operation != ConSanCacheOperation::Acquire ||
        !cache_source->ordinary_acquire_mutation_supported || !sync.same_container(*cache, *load) ||
        !consan_nonempty_execution_owners_equal(sync.execution_owners(*cache),
                                                sync.execution_owners(*load)) ||
        cache_source->size == 0u || cache_source->size % sizeof(uint32_t) != 0u)
      return std::nullopt;
    return OrdinaryAcquireMutationTarget{&site, load, cache, sequence};
  };

  if (!selection.primary_site_identity.empty()) {
    const ConSanFaultSite *site = find_fault_site_by_identity(
        inventory, selection.primary_site_identity, ConSanFaultSiteKind::OrdinaryMemory);
    return site == nullptr ? std::nullopt : qualify(*site);
  }
  uint32_t index = 0;
  for (const ConSanFaultSite &site : inventory.fault_sites) {
    auto target = qualify(site);
    if (target && index++ == selection.ordinal)
      return target;
  }
  return std::nullopt;
}

ExactBarrierDropPairResolution
resolve_exact_barrier_drop_pair(const ConSanFaultSelectionView &inventory,
                                const ConSanFaultSelection &selection) {
  using Issue = ExactBarrierDropPairIssue;
  const SynchronizationInventoryView sync = inventory.program_inventory.sync();
  if (selection.primary_site_identity.empty() || selection.primary_sequence_identity.empty()) {
    return {.pair = std::nullopt, .issue = Issue::MissingExactIdentity};
  }
  const auto sequence = std::ranges::find(sync.sync_sequences, selection.primary_sequence_identity,
                                          &ConSanSyncSequence::identity);
  if (sequence == sync.sync_sequences.end()) {
    return {.pair = std::nullopt, .issue = Issue::SequenceNotFound};
  }
  const std::vector<ConSanExecutionOwner> sequence_owners = sync.execution_owners(*sequence);
  if (sequence->kind != ConSanSyncKind::Barrier ||
      sequence->operation != ConSanSyncOperation::BarrierFull ||
      !consan_sync_confidence_meets(sequence->confidence, ConSanSemanticConfidence::Conservative) ||
      sequence->member_event_ids.size() != 2u || !sequence_has_exact_members(sync, *sequence) ||
      !consan_execution_owners_include_requested_kernel(sequence_owners, inventory,
                                                        selection.kernel_name_filter)) {
    return {.pair = std::nullopt, .issue = Issue::SequenceNotQualified};
  }

  const ConSanFaultSite *primary = find_fault_site_by_identity(
      inventory, selection.primary_site_identity, ConSanFaultSiteKind::Barrier);
  const ConSanSyncEvent *primary_event =
      primary == nullptr ? nullptr : sync.find_event(primary->source_site);
  ConSanSyncEventId primary_member_id;
  if (primary_event != nullptr)
    primary_member_id = sync.event_id(*primary_event);
  if (primary == nullptr || primary_event == nullptr ||
      std::ranges::find(sequence->member_event_ids, primary_member_id) ==
          sequence->member_event_ids.end() ||
      !consan_execution_owners_include_requested_kernel(inventory.execution_owners(*primary),
                                                        inventory, selection.kernel_name_filter)) {
    return {.pair = std::nullopt, .issue = Issue::PrimaryNotMember};
  }

  const ConSanFaultSite *companion = nullptr;
  const ConSanProgramContainerRef *sequence_container = sync.container(*sequence);
  if (sequence_container == nullptr)
    return {.pair = std::nullopt, .issue = Issue::SequenceNotQualified};
  for (ConSanSyncEventId member_id : sequence->member_event_ids) {
    const ConSanSyncEvent *member = sync.find_event(member_id);
    if (member == nullptr)
      return {.pair = std::nullopt, .issue = Issue::MemberSiteMissing};
    const auto matching_site =
        std::ranges::find_if(inventory.fault_sites, [&](const ConSanFaultSite &candidate) {
          const ConSanProgramSite *candidate_source = inventory.source(candidate);
          return candidate.kind == ConSanFaultSiteKind::Barrier &&
                 sync.find_event(candidate.source_site) == member && candidate_source != nullptr &&
                 candidate_source->container == *sequence_container &&
                 consan_execution_owners_include_requested_kernel(
                     inventory.execution_owners(candidate), inventory,
                     selection.kernel_name_filter);
        });
    if (matching_site == inventory.fault_sites.end()) {
      return {.pair = std::nullopt, .issue = Issue::MemberSiteMissing};
    }
    const auto duplicate =
        std::ranges::find_if(std::next(matching_site), inventory.fault_sites.end(),
                             [&](const ConSanFaultSite &candidate) {
                               return candidate.kind == ConSanFaultSiteKind::Barrier &&
                                      sync.find_event(candidate.source_site) == member;
                             });
    if (duplicate != inventory.fault_sites.end()) {
      return {.pair = std::nullopt, .issue = Issue::MemberSiteAmbiguous};
    }
    if (matching_site->identity != primary->identity)
      companion = &*matching_site;
  }
  const ConSanProgramSite *primary_source = inventory.source(*primary);
  const ConSanProgramSite *companion_source =
      companion == nullptr ? nullptr : inventory.source(*companion);
  if (primary_source == nullptr || companion_source == nullptr ||
      primary_source->size() != sizeof(uint32_t) || companion_source->size() != sizeof(uint32_t) ||
      primary_source->decoded_file_offset() == companion_source->decoded_file_offset()) {
    return {.pair = std::nullopt, .issue = Issue::InvalidPairGeometry};
  }
  return {.pair = ExactBarrierDropPair{&*sequence, primary, companion}, .issue = Issue::None};
}

ExactBarrierDropGroupResolution
resolve_exact_barrier_drop_group(const ConSanFaultSelectionView &inventory,
                                 const ConSanFaultSelection &selection) {
  using Issue = ExactBarrierDropGroupIssue;
  if (selection.companion_site_identity.empty() || selection.companion_sequence_identity.empty()) {
    return {.group = std::nullopt,
            .issue = Issue::MissingCompanionIdentity,
            .member_issue = ExactBarrierDropPairIssue::None};
  }
  ConSanFaultSelection first_selection = selection;
  first_selection.companion_site_identity = {};
  first_selection.companion_sequence_identity = {};
  const auto first = resolve_exact_barrier_drop_pair(inventory, first_selection);
  if (!first.pair)
    return {.group = std::nullopt, .issue = Issue::FirstPairRejected, .member_issue = first.issue};
  ConSanFaultSelection second_selection = selection;
  second_selection.primary_site_identity = selection.companion_site_identity;
  second_selection.primary_sequence_identity = selection.companion_sequence_identity;
  second_selection.companion_site_identity = {};
  second_selection.companion_sequence_identity = {};
  const auto second = resolve_exact_barrier_drop_pair(inventory, second_selection);
  if (!second.pair)
    return {
        .group = std::nullopt, .issue = Issue::SecondPairRejected, .member_issue = second.issue};
  const auto first_range = exact_barrier_drop_pair_range(inventory, *first.pair);
  const auto second_range = exact_barrier_drop_pair_range(inventory, *second.pair);
  const SynchronizationInventoryView sync = inventory.program_inventory.sync();
  const std::vector<ConSanExecutionOwner> first_owners =
      sync.execution_owners(*first.pair->sequence);
  const std::vector<ConSanExecutionOwner> second_owners =
      sync.execution_owners(*second.pair->sequence);
  if (first.pair->sequence->identity == second.pair->sequence->identity ||
      first_range.second > second_range.first) {
    return {.group = std::nullopt,
            .issue = Issue::PairsOverlapOrUnordered,
            .member_issue = ExactBarrierDropPairIssue::None};
  }
  if (!sync.same_container(*first.pair->sequence, *second.pair->sequence) ||
      !consan_nonempty_execution_owners_equal(first_owners, second_owners)) {
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

} // namespace rocjitsu
