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

[[nodiscard]] bool same_execution_owners(std::span<const ConSanExecutionOwner> lhs,
                                         std::span<const ConSanExecutionOwner> rhs) {
  return !lhs.empty() && lhs.size() == rhs.size() &&
         std::ranges::equal(lhs, rhs, [](const auto &left, const auto &right) {
           return left.descriptor_file_offset == right.descriptor_file_offset &&
                  left.proof == right.proof;
         });
}

[[nodiscard]] const ConSanKernelInfo *requested_kernel_owner(const ProgramInventory &inventory,
                                                             std::string_view filter) {
  if (filter.empty())
    return nullptr;
  if (const ConSanKernelInfo *exact = inventory.find_kernel_by_name(filter))
    return exact;
  const ConSanKernelInfo *match = nullptr;
  for (const ConSanKernelInfo &kernel : inventory.kernels()) {
    if (kernel.name.find(filter) == std::string::npos)
      continue;
    if (match != nullptr)
      return nullptr;
    match = &kernel;
  }
  return match;
}

[[nodiscard]] std::pair<uint64_t, uint64_t>
exact_barrier_drop_pair_range(const ExactBarrierDropPair &pair) {
  const uint64_t begin = std::min(pair.primary->text_offset, pair.companion->text_offset);
  const uint64_t end = std::max(pair.primary->text_offset + pair.primary->size,
                                pair.companion->text_offset + pair.companion->size);
  return {begin, end};
}

} // namespace

bool consan_fault_admits_cross_block_barrier_move(const ConSanBarrierMoveDestination &destination,
                                                  const ConSanOptions &options) {
  if (options.fault_barrier_move_direction != ConSanBarrierMoveDirection::Earlier ||
      !destination.structured_guard_block_index || !destination.structured_source_block_index ||
      !destination.structured_guard_offset || !destination.structured_source_offset) {
    return false;
  }
  switch (destination.cfg_contract) {
  case ConSanBarrierMoveCfgContract::SameBlock:
    return false;
  case ConSanBarrierMoveCfgContract::CompletingStructuredDiamond:
    return options.fault_allow_completing_conditional_barrier_move;
  case ConSanBarrierMoveCfgContract::DestructiveStructuredExecDiamond:
    return options.fault_allow_destructive_divergent_barrier_move;
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
  const ConSanKernelInfo *kernel =
      requested_kernel_owner(inventory.program_inventory, kernel_name_filter);
  return kernel != nullptr &&
         std::ranges::find(owners, kernel->descriptor_file_offset,
                           &ConSanExecutionOwner::descriptor_file_offset) != owners.end();
}

const ConSanFaultSite *select_fault_site_for_plan(const ConSanFaultSelectionView &inventory,
                                                  const ConSanFaultSelection &selection,
                                                  ConSanFaultSiteKind kind) {
  if (!selection.primary_site_identity.empty()) {
    const ConSanFaultSite *site =
        find_fault_site_by_identity(inventory, selection.primary_site_identity, kind);
    return site != nullptr && consan_execution_owners_include_requested_kernel(
                                  site->execution_owners, inventory, selection.kernel_name_filter)
               ? site
               : nullptr;
  }
  uint32_t index = 0;
  for (const ConSanFaultSite &site : inventory.fault_sites) {
    if (site.kind != kind || !consan_execution_owners_include_requested_kernel(
                                 site.execution_owners, inventory, selection.kernel_name_filter))
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
    if (site.kind != ConSanFaultSiteKind::OrdinaryMemory ||
        site.ordinary_memory_support_reason != ConSanOrdinaryMemorySupportReason::Supported ||
        site.mnemonic != "global_load_b32" || !site.sync_event_identity ||
        !site.sync_sequence_identity ||
        !consan_execution_owners_include_requested_kernel(site.execution_owners, inventory,
                                                          selection.kernel_name_filter))
      return std::nullopt;
    const ConSanSyncEvent *load = sync.find_event(*site.sync_event_identity);
    const auto sequence_it = std::ranges::find(sync.sync_sequences, *site.sync_sequence_identity,
                                               &ConSanSyncSequence::identity);
    if (load == nullptr || sequence_it == sync.sync_sequences.end())
      return std::nullopt;
    const ConSanSyncSequence *sequence = &*sequence_it;
    if (load->operation != ConSanSyncOperation::OrdinaryLoad || !load->raw_scope ||
        (*load->raw_scope != 2u && *load->raw_scope != 3u) ||
        sequence->kind != ConSanSyncSequenceKind::OrdinaryMemory ||
        sequence->operation != ConSanSyncOperation::OrdinaryLoad ||
        sequence->memory_role != ConSanSyncMemoryRole::Acquire ||
        !consan_sync_confidence_meets(sequence->memory_role_confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        !sequence->basic_block_index || sequence->member_event_identities.size() != 2u ||
        !sequence_has_exact_members(sync, *sequence) ||
        sequence->member_event_identities.front() != load->identity ||
        !same_execution_owners(sequence->execution_owners, site.execution_owners))
      return std::nullopt;
    const ConSanSyncEvent *cache = sync.find_event(sequence->member_event_identities.back());
    if (cache == nullptr || cache->kind != ConSanSyncEventKind::Fence ||
        cache->cache_operation != ConSanCacheOperation::Acquire ||
        !cache->ordinary_acquire_mutation_supported ||
        cache->container_name != load->container_name || cache->in_kernel != load->in_kernel ||
        !same_execution_owners(cache->execution_owners, load->execution_owners) ||
        cache->size == 0u || cache->size % sizeof(uint32_t) != 0u)
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
  if (sequence->kind != ConSanSyncSequenceKind::Barrier ||
      sequence->operation != ConSanSyncOperation::BarrierFull ||
      !consan_sync_confidence_meets(sequence->confidence, ConSanSemanticConfidence::Conservative) ||
      sequence->member_event_identities.size() != 2u ||
      !sequence_has_exact_members(sync, *sequence) ||
      !consan_execution_owners_include_requested_kernel(sequence->execution_owners, inventory,
                                                        selection.kernel_name_filter)) {
    return {.pair = std::nullopt, .issue = Issue::SequenceNotQualified};
  }

  const ConSanFaultSite *primary = find_fault_site_by_identity(
      inventory, selection.primary_site_identity, ConSanFaultSiteKind::Barrier);
  if (primary == nullptr || !primary->sync_event_identity || !primary->sync_sequence_identity ||
      *primary->sync_sequence_identity != sequence->identity ||
      std::ranges::find(sequence->member_event_identities, *primary->sync_event_identity) ==
          sequence->member_event_identities.end() ||
      !consan_execution_owners_include_requested_kernel(primary->execution_owners, inventory,
                                                        selection.kernel_name_filter)) {
    return {.pair = std::nullopt, .issue = Issue::PrimaryNotMember};
  }

  const ConSanFaultSite *companion = nullptr;
  for (const std::string &member_identity : sequence->member_event_identities) {
    const auto matching_site =
        std::ranges::find_if(inventory.fault_sites, [&](const ConSanFaultSite &candidate) {
          return candidate.kind == ConSanFaultSiteKind::Barrier &&
                 candidate.sync_event_identity == member_identity &&
                 candidate.sync_sequence_identity == sequence->identity &&
                 candidate.container_name == sequence->container_name &&
                 candidate.in_kernel == sequence->in_kernel &&
                 consan_execution_owners_include_requested_kernel(
                     candidate.execution_owners, inventory, selection.kernel_name_filter);
        });
    if (matching_site == inventory.fault_sites.end()) {
      return {.pair = std::nullopt, .issue = Issue::MemberSiteMissing};
    }
    const auto duplicate =
        std::ranges::find_if(std::next(matching_site), inventory.fault_sites.end(),
                             [&](const ConSanFaultSite &candidate) {
                               return candidate.kind == ConSanFaultSiteKind::Barrier &&
                                      candidate.sync_event_identity == member_identity;
                             });
    if (duplicate != inventory.fault_sites.end()) {
      return {.pair = std::nullopt, .issue = Issue::MemberSiteAmbiguous};
    }
    if (matching_site->identity != primary->identity)
      companion = &*matching_site;
  }
  if (companion == nullptr || primary->size != sizeof(uint32_t) ||
      companion->size != sizeof(uint32_t) || primary->file_offset == companion->file_offset) {
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
  const auto first_range = exact_barrier_drop_pair_range(*first.pair);
  const auto second_range = exact_barrier_drop_pair_range(*second.pair);
  if (first.pair->sequence->identity == second.pair->sequence->identity ||
      first_range.second > second_range.first) {
    return {.group = std::nullopt,
            .issue = Issue::PairsOverlapOrUnordered,
            .member_issue = ExactBarrierDropPairIssue::None};
  }
  if (first.pair->sequence->container_name != second.pair->sequence->container_name ||
      first.pair->sequence->in_kernel != second.pair->sequence->in_kernel ||
      !same_execution_owners(first.pair->sequence->execution_owners,
                             second.pair->sequence->execution_owners)) {
    return {.group = std::nullopt,
            .issue = Issue::PairsHaveDifferentOwners,
            .member_issue = ExactBarrierDropPairIssue::None};
  }
  return {.group = ExactBarrierDropGroup{*first.pair, *second.pair},
          .issue = Issue::None,
          .member_issue = ExactBarrierDropPairIssue::None};
}

std::string_view exact_barrier_drop_pair_issue_message(ExactBarrierDropPairIssue issue) {
  using Issue = ExactBarrierDropPairIssue;
  switch (issue) {
  case Issue::None:
    return {};
  case Issue::MissingExactIdentity:
    return "an exact site identity and logical sequence identity are both required";
  case Issue::SequenceNotFound:
    return "the exact logical sequence identity was not found";
  case Issue::SequenceNotQualified:
    return "the exact sequence is not an owned complete conservative two-member barrier";
  case Issue::PrimaryNotMember:
    return "the exact site is not a member of the requested logical barrier";
  case Issue::MemberSiteMissing:
    return "a logical barrier member has no exact owned patch site";
  case Issue::MemberSiteAmbiguous:
    return "a logical barrier member maps to multiple patch sites";
  case Issue::InvalidPairGeometry:
    return "the exact logical barrier does not have two distinct one-word patch sites";
  case Issue::Count:
    break;
  }
  return "invalid exact barrier-drop pair issue";
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
