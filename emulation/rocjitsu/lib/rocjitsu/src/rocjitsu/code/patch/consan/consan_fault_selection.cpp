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

[[nodiscard]] const ConSanKernelInfo *requested_kernel_owner(const ConSanTransformArtifacts &result,
                                                             std::string_view filter) {
  if (filter.empty())
    return nullptr;
  if (const ConSanKernelInfo *exact = result.program_inventory.find_kernel_by_name(filter))
    return exact;
  const ConSanKernelInfo *match = nullptr;
  for (const ConSanKernelInfo &kernel : result.program_inventory.kernels()) {
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

const ConSanFaultSite *find_fault_site_by_identity(const ConSanTransformArtifacts &result,
                                                   std::string_view identity,
                                                   ConSanFaultSiteKind kind) {
  const auto site = std::ranges::find_if(result.fault_sites, [&](const ConSanFaultSite &candidate) {
    return candidate.kind == kind && candidate.identity == identity;
  });
  return site == result.fault_sites.end() ? nullptr : &*site;
}

bool consan_execution_owners_include_requested_kernel(std::span<const ConSanExecutionOwner> owners,
                                                      const ConSanTransformArtifacts &result,
                                                      std::string_view kernel_name_filter) {
  if (kernel_name_filter.empty())
    return true;
  const ConSanKernelInfo *kernel = requested_kernel_owner(result, kernel_name_filter);
  return kernel != nullptr &&
         std::ranges::find(owners, kernel->descriptor_file_offset,
                           &ConSanExecutionOwner::descriptor_file_offset) != owners.end();
}

const ConSanFaultSite *select_fault_site_for_plan(const ConSanTransformArtifacts &result,
                                                  const ConSanFaultSelection &selection,
                                                  ConSanFaultSiteKind kind) {
  if (!selection.primary_site_identity.empty()) {
    const ConSanFaultSite *site =
        find_fault_site_by_identity(result, selection.primary_site_identity, kind);
    return site != nullptr && consan_execution_owners_include_requested_kernel(
                                  site->execution_owners, result, selection.kernel_name_filter)
               ? site
               : nullptr;
  }
  uint32_t index = 0;
  for (const ConSanFaultSite &site : result.fault_sites) {
    if (site.kind != kind || !consan_execution_owners_include_requested_kernel(
                                 site.execution_owners, result, selection.kernel_name_filter))
      continue;
    if (index++ == selection.ordinal)
      return &site;
  }
  return nullptr;
}

std::optional<OrdinaryAcquireMutationTarget>
select_ordinary_acquire_mutation_target(const ConSanTransformArtifacts &result,
                                        const ConSanFaultSelection &selection) {
  const SynchronizationInventoryView sync = result.program_inventory.sync();
  const auto qualify =
      [&](const ConSanFaultSite &site) -> std::optional<OrdinaryAcquireMutationTarget> {
    if (site.kind != ConSanFaultSiteKind::OrdinaryMemory ||
        site.ordinary_memory_support_reason != ConSanOrdinaryMemorySupportReason::Supported ||
        site.mnemonic != "global_load_b32" || !site.sync_event_identity ||
        !site.sync_sequence_identity ||
        !consan_execution_owners_include_requested_kernel(site.execution_owners, result,
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
        !sequence_has_exact_members(result, *sequence) ||
        sequence->member_event_identities.front() != load->identity ||
        !same_execution_owners(sequence->execution_owners, site.execution_owners))
      return std::nullopt;
    const ConSanSyncEvent *cache = sync.find_event(sequence->member_event_identities.back());
    if (cache == nullptr || cache->kind != ConSanSyncEventKind::Fence ||
        cache->mnemonic != "global_inv" || cache->container_name != load->container_name ||
        cache->in_kernel != load->in_kernel ||
        !same_execution_owners(cache->execution_owners, load->execution_owners) ||
        cache->size == 0u || cache->size % sizeof(uint32_t) != 0u)
      return std::nullopt;
    return OrdinaryAcquireMutationTarget{&site, load, cache, sequence};
  };

  if (!selection.primary_site_identity.empty()) {
    const ConSanFaultSite *site = find_fault_site_by_identity(
        result, selection.primary_site_identity, ConSanFaultSiteKind::OrdinaryMemory);
    return site == nullptr ? std::nullopt : qualify(*site);
  }
  uint32_t index = 0;
  for (const ConSanFaultSite &site : result.fault_sites) {
    auto target = qualify(site);
    if (target && index++ == selection.ordinal)
      return target;
  }
  return std::nullopt;
}

std::optional<ExactBarrierDropPair>
resolve_exact_barrier_drop_pair(const ConSanTransformArtifacts &result,
                                const ConSanFaultSelection &selection, std::string *reason) {
  const SynchronizationInventoryView sync = result.program_inventory.sync();
  if (selection.primary_site_identity.empty() || selection.primary_sequence_identity.empty()) {
    *reason = "an exact site identity and logical sequence identity are both required";
    return std::nullopt;
  }
  const auto sequence = std::ranges::find(sync.sync_sequences, selection.primary_sequence_identity,
                                          &ConSanSyncSequence::identity);
  if (sequence == sync.sync_sequences.end()) {
    *reason = "the exact logical sequence identity was not found";
    return std::nullopt;
  }
  if (sequence->kind != ConSanSyncSequenceKind::Barrier ||
      sequence->operation != ConSanSyncOperation::BarrierFull ||
      !consan_sync_confidence_meets(sequence->confidence, ConSanSemanticConfidence::Conservative) ||
      sequence->member_event_identities.size() != 2u ||
      !sequence_has_exact_members(result, *sequence) ||
      !consan_execution_owners_include_requested_kernel(sequence->execution_owners, result,
                                                        selection.kernel_name_filter)) {
    *reason = "the exact sequence is not an owned complete conservative two-member barrier";
    return std::nullopt;
  }

  const ConSanFaultSite *primary = find_fault_site_by_identity(
      result, selection.primary_site_identity, ConSanFaultSiteKind::Barrier);
  if (primary == nullptr || !primary->sync_event_identity || !primary->sync_sequence_identity ||
      *primary->sync_sequence_identity != sequence->identity ||
      std::ranges::find(sequence->member_event_identities, *primary->sync_event_identity) ==
          sequence->member_event_identities.end() ||
      !consan_execution_owners_include_requested_kernel(primary->execution_owners, result,
                                                        selection.kernel_name_filter)) {
    *reason = "the exact site is not a member of the requested logical barrier";
    return std::nullopt;
  }

  const ConSanFaultSite *companion = nullptr;
  for (const std::string &member_identity : sequence->member_event_identities) {
    const auto matching_site =
        std::ranges::find_if(result.fault_sites, [&](const ConSanFaultSite &candidate) {
          return candidate.kind == ConSanFaultSiteKind::Barrier &&
                 candidate.sync_event_identity == member_identity &&
                 candidate.sync_sequence_identity == sequence->identity &&
                 candidate.container_name == sequence->container_name &&
                 candidate.in_kernel == sequence->in_kernel &&
                 consan_execution_owners_include_requested_kernel(
                     candidate.execution_owners, result, selection.kernel_name_filter);
        });
    if (matching_site == result.fault_sites.end()) {
      *reason = "a logical barrier member has no exact owned patch site";
      return std::nullopt;
    }
    const auto duplicate = std::ranges::find_if(
        std::next(matching_site), result.fault_sites.end(), [&](const ConSanFaultSite &candidate) {
          return candidate.kind == ConSanFaultSiteKind::Barrier &&
                 candidate.sync_event_identity == member_identity;
        });
    if (duplicate != result.fault_sites.end()) {
      *reason = "a logical barrier member maps to multiple patch sites";
      return std::nullopt;
    }
    if (matching_site->identity != primary->identity)
      companion = &*matching_site;
  }
  if (companion == nullptr || primary->size != sizeof(uint32_t) ||
      companion->size != sizeof(uint32_t) || primary->file_offset == companion->file_offset) {
    *reason = "the exact logical barrier does not have two distinct one-word patch sites";
    return std::nullopt;
  }
  return ExactBarrierDropPair{&*sequence, primary, companion};
}

std::optional<ExactBarrierDropGroup>
resolve_exact_barrier_drop_group(const ConSanTransformArtifacts &result,
                                 const ConSanFaultSelection &selection, std::string *reason) {
  if (selection.companion_site_identity.empty() || selection.companion_sequence_identity.empty()) {
    *reason = "a grouped drop requires an exact companion site and sequence identity";
    return std::nullopt;
  }
  std::string member_reason;
  ConSanFaultSelection first_selection = selection;
  first_selection.companion_site_identity = {};
  first_selection.companion_sequence_identity = {};
  const auto first = resolve_exact_barrier_drop_pair(result, first_selection, &member_reason);
  if (!first) {
    *reason = "group member zero is invalid: " + member_reason;
    return std::nullopt;
  }
  ConSanFaultSelection second_selection = selection;
  second_selection.primary_site_identity = selection.companion_site_identity;
  second_selection.primary_sequence_identity = selection.companion_sequence_identity;
  second_selection.companion_site_identity = {};
  second_selection.companion_sequence_identity = {};
  const auto second = resolve_exact_barrier_drop_pair(result, second_selection, &member_reason);
  if (!second) {
    *reason = "group member one is invalid: " + member_reason;
    return std::nullopt;
  }
  const auto first_range = exact_barrier_drop_pair_range(*first);
  const auto second_range = exact_barrier_drop_pair_range(*second);
  if (first->sequence->identity == second->sequence->identity ||
      first_range.second > second_range.first) {
    *reason = "the two exact pairs are duplicate, overlapping, or not strictly ordered";
    return std::nullopt;
  }
  if (first->sequence->container_name != second->sequence->container_name ||
      first->sequence->in_kernel != second->sequence->in_kernel ||
      !same_execution_owners(first->sequence->execution_owners,
                             second->sequence->execution_owners)) {
    *reason = "the two exact pairs do not have identical container and execution owners";
    return std::nullopt;
  }
  return ExactBarrierDropGroup{*first, *second};
}

std::string exact_barrier_drop_group_identity(const ExactBarrierDropGroup &group) {
  return "barrier-group[" + group.first.sequence->identity + "][" +
         group.second.sequence->identity + "]";
}

} // namespace rocjitsu
