// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <ranges>
#include <tuple>
#include <unordered_set>

namespace rocjitsu {

namespace {

void report(std::string *error_out, std::string error) {
  if (error_out != nullptr)
    *error_out = std::move(error);
}

} // namespace

bool is_consan_branch_relay_reservoir_instruction(const Instruction &instruction, uint64_t offset,
                                                  std::span<const uint8_t> text,
                                                  rj_code_arch_t arch) {
  const int size = instruction.size();
  const std::string_view mnemonic = instruction.mnemonic();
  if (size <= 0 || size % static_cast<int>(sizeof(uint32_t)) != 0 ||
      instruction.raw_encoding() == nullptr || offset % sizeof(uint32_t) != 0u ||
      offset > text.size() || static_cast<uint64_t>(size) > text.size() - offset ||
      mnemonic.starts_with("ds_") || mnemonic == "s_clause" || mnemonic == "s_delay_alu") {
    return false;
  }
  if (size == static_cast<int>(sizeof(uint32_t)) ||
      size == 2 * static_cast<int>(sizeof(uint32_t))) {
    return is_relocatable_anchor(instruction, offset, text, arch);
  }
  constexpr uint64_t kControlFlowFlags =
      BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR;
  return size == 3 * static_cast<int>(sizeof(uint32_t)) &&
         (mnemonic.starts_with("flat_load") || mnemonic.starts_with("flat_store")) &&
         !(instruction.flags() & kControlFlowFlags) && !instruction.branch_offset_bytes();
}

bool BranchOnlyRelayRouter::offer(uint64_t offset, BranchOnlyRelayProvenance provenance) {
  if (offset % sizeof(uint32_t) != 0u)
    return false;
  return relays_.emplace(offset, provenance).second;
}

void BranchOnlyRelayRouter::retire_range(uint64_t offset, uint64_t size) {
  if (size == 0u)
    return;
  const uint64_t end = size > std::numeric_limits<uint64_t>::max() - offset
                           ? std::numeric_limits<uint64_t>::max()
                           : offset + size;
  auto relay = relays_.lower_bound(offset);
  while (relay != relays_.end() && relay->first < end)
    relay = relays_.erase(relay);
}

std::optional<BranchOnlyRelayRoute>
BranchOnlyRelayRouter::plan_pair(DbiPatchPlacementPlanner &tentative_planner, uint64_t entry_source,
                                 uint64_t entry_target, uint64_t return_source,
                                 uint64_t return_target, std::string *error_out,
                                 BranchOnlyRelayPlanFailure *failure_out) const {
  const std::array requests = {
      BranchOnlyRelayPairRequest{entry_source, entry_target, return_source, return_target},
  };
  BranchOnlyRelayBatchPlan batch = plan_pairs(tentative_planner, requests, error_out);
  if (failure_out != nullptr)
    *failure_out = batch.failure;
  if (!batch.complete() || batch.routes.size() != 1u)
    return std::nullopt;
  return std::move(batch.routes.front());
}

BranchOnlyRelayBatchPlan
BranchOnlyRelayRouter::plan_pairs(DbiPatchPlacementPlanner &tentative_planner,
                                  std::span<const BranchOnlyRelayPairRequest> requests,
                                  std::string *error_out) const {
  BranchOnlyRelayBatchPlan batch;
  batch.routes.resize(requests.size());
  if (requests.empty())
    return batch;

  std::vector<uint64_t> relay_offsets;
  relay_offsets.reserve(relays_.size());
  for (const auto &[offset, provenance] : relays_) {
    (void)provenance;
    relay_offsets.push_back(offset);
  }

  std::vector<uint64_t> entry_sources;
  std::vector<uint64_t> entry_targets;
  entry_sources.reserve(requests.size());
  entry_targets.reserve(requests.size());
  for (const BranchOnlyRelayPairRequest &request : requests) {
    entry_sources.push_back(request.entry_source);
    entry_targets.push_back(request.entry_target);
  }
  const auto entry_plan =
      plan_forward_sopp_branch_relays(entry_sources, relay_offsets, entry_targets, error_out);
  if (!entry_plan) {
    batch.failure = BranchOnlyRelayPlanFailure::EntryRoute;
    batch.rejected_pair_indices.resize(requests.size());
    std::iota(batch.rejected_pair_indices.begin(), batch.rejected_pair_indices.end(), 0u);
    return batch;
  }

  const auto route_reaches_exact_target = [](uint64_t source, std::span<const uint64_t> relays,
                                             uint64_t target, bool forward) {
    uint64_t cursor = source;
    for (uint64_t hop : relays) {
      if ((forward ? hop <= cursor : hop >= cursor) || !compute_sopp_branch_simm16(cursor, hop)) {
        return false;
      }
      cursor = hop;
    }
    return (forward ? target > cursor : target < cursor) &&
           compute_sopp_branch_simm16(cursor, target).has_value();
  };

  std::unordered_set<uint64_t> entry_relays;
  std::vector<bool> entry_assigned(requests.size(), false);
  std::unordered_map<uint64_t, size_t> entry_owner_by_target;
  entry_owner_by_target.reserve(requests.size());
  for (size_t index = 0; index < requests.size(); ++index)
    entry_owner_by_target.emplace(requests[index].entry_target, index);
  for (const SoppBranchRelayRoute &route : entry_plan->routes) {
    const auto owner = entry_owner_by_target.find(route.island_offset);
    if (owner == entry_owner_by_target.end())
      continue;
    const size_t request_index = owner->second;
    if (!route_reaches_exact_target(requests[request_index].entry_source, route.relay_offsets,
                                    route.island_offset, /*forward=*/true)) {
      continue;
    }
    batch.routes[request_index].entry_relay_offsets = route.relay_offsets;
    entry_assigned[request_index] = true;
    entry_relays.insert(route.relay_offsets.begin(), route.relay_offsets.end());
  }
  std::unordered_set<size_t> rejected_entry_indices;
  for (size_t index = 0; index < entry_assigned.size(); ++index) {
    if (!entry_assigned[index])
      rejected_entry_indices.insert(index);
  }
  std::erase_if(relay_offsets, [&](uint64_t offset) { return entry_relays.contains(offset); });

  std::vector<uint64_t> return_sources;
  std::vector<uint64_t> return_targets;
  return_sources.reserve(requests.size());
  return_targets.reserve(requests.size());
  for (const BranchOnlyRelayPairRequest &request : requests) {
    return_sources.push_back(request.return_source);
    return_targets.push_back(request.return_target);
  }
  const auto return_plan =
      plan_backward_sopp_branch_relays(return_sources, relay_offsets, return_targets, error_out);
  if (!return_plan) {
    batch.failure = BranchOnlyRelayPlanFailure::ReturnRoute;
    batch.rejected_pair_indices.resize(requests.size());
    std::iota(batch.rejected_pair_indices.begin(), batch.rejected_pair_indices.end(), 0u);
    return batch;
  }

  std::vector<bool> return_assigned(requests.size(), false);
  std::unordered_map<uint64_t, size_t> return_owner_by_target;
  return_owner_by_target.reserve(requests.size());
  for (size_t index = 0; index < requests.size(); ++index)
    return_owner_by_target.emplace(requests[index].return_target, index);
  for (const SoppBranchRelayRoute &route : return_plan->routes) {
    const auto owner = return_owner_by_target.find(route.island_offset);
    if (owner == return_owner_by_target.end())
      continue;
    const size_t request_index = owner->second;
    if (!route_reaches_exact_target(requests[request_index].return_source, route.relay_offsets,
                                    route.island_offset, /*forward=*/false)) {
      continue;
    }
    batch.routes[request_index].return_relay_offsets = route.relay_offsets;
    return_assigned[request_index] = true;
  }
  std::unordered_set<size_t> rejected_return_indices;
  for (size_t index = 0; index < return_assigned.size(); ++index) {
    if (!return_assigned[index])
      rejected_return_indices.insert(index);
  }
  std::unordered_set<size_t> rejected(rejected_entry_indices.begin(), rejected_entry_indices.end());
  rejected.insert(rejected_return_indices.begin(), rejected_return_indices.end());
  batch.rejected_pair_indices.assign(rejected.begin(), rejected.end());
  std::ranges::sort(batch.rejected_pair_indices);
  if (!batch.rejected_pair_indices.empty()) {
    batch.failure = !rejected_entry_indices.empty() ? BranchOnlyRelayPlanFailure::EntryRoute
                                                    : BranchOnlyRelayPlanFailure::ReturnRoute;
    if (error_out == nullptr || error_out->empty()) {
      report(error_out, batch.failure == BranchOnlyRelayPlanFailure::EntryRoute
                            ? "branch-only router could not reach every appended entry"
                            : "branch-only router could not reach every original continuation");
    }
  }

  for (BranchOnlyRelayRoute &route : batch.routes) {
    std::vector<uint64_t> claimed_offsets = route.entry_relay_offsets;
    claimed_offsets.insert(claimed_offsets.end(), route.return_relay_offsets.begin(),
                           route.return_relay_offsets.end());
    route.claims.reserve(claimed_offsets.size());
    for (uint64_t offset : claimed_offsets) {
      const auto relay = relays_.find(offset);
      if (relay == relays_.end()) {
        batch.failure = BranchOnlyRelayPlanFailure::Reservation;
        report(error_out, "branch-only router selected an unknown relay");
        return batch;
      }
      route.claims.push_back({offset, relay->second});
    }
  }

  if (!batch.rejected_pair_indices.empty())
    return batch;

  DbiPatchPlacementPlanner reserved_planner = tentative_planner;
  for (const BranchOnlyRelayRoute &route : batch.routes) {
    for (const BranchOnlyRelayClaim &claim : route.claims) {
      if (claim.provenance == BranchOnlyRelayProvenance::PristineNop &&
          !reserved_planner.reserve_existing_range(claim.offset, sizeof(uint32_t), error_out)) {
        batch.failure = BranchOnlyRelayPlanFailure::Reservation;
        return batch;
      }
    }
  }
  tentative_planner = std::move(reserved_planner);
  return batch;
}

bool BranchOnlyRelayRouter::commit(const BranchOnlyRelayRoute &route, std::string *error_out) {
  const std::array routes = {route};
  return commit(routes, error_out);
}

bool BranchOnlyRelayRouter::commit(std::span<const BranchOnlyRelayRoute> routes,
                                   std::string *error_out) {
  std::unordered_set<uint64_t> claimed_offsets;
  for (const BranchOnlyRelayRoute &route : routes) {
    for (const BranchOnlyRelayClaim &claim : route.claims) {
      const auto relay = relays_.find(claim.offset);
      if (relay == relays_.end() || relay->second != claim.provenance ||
          !claimed_offsets.insert(claim.offset).second) {
        report(error_out, "branch-only router claim changed before commit");
        return false;
      }
    }
  }
  for (uint64_t offset : claimed_offsets)
    relays_.erase(offset);
  return true;
}

bool BranchOnlyDirectRelayReservoirSet::mark_claims_used(
    std::span<const BranchOnlyRelayClaim> claims, std::string *error_out) {
  std::vector<size_t> reservoir_indices;
  for (const BranchOnlyRelayClaim &claim : claims) {
    if (claim.provenance != BranchOnlyRelayProvenance::OwnedReservoir)
      continue;
    const auto reservoir = reservoir_by_relay.find(claim.offset);
    if (reservoir == reservoir_by_relay.end() || reservoir->second >= reservoirs.size()) {
      report(error_out, "branch-only router lost a claimed direct reservoir");
      return false;
    }
    reservoir_indices.push_back(reservoir->second);
  }
  for (size_t index : reservoir_indices)
    reservoirs[index].used = true;
  return true;
}

bool BranchOnlyRelayRouter::plan_direct_reservoirs(
    std::span<BasicBlock *const> blocks, std::span<const uint8_t> pristine_text,
    std::span<const std::pair<uint64_t, uint64_t>> protected_ranges, rj_code_arch_t arch,
    uint64_t route_midpoint, size_t target_relay_count, DbiPatchPlacementPlanner &placement_planner,
    BranchOnlyDirectRelayReservoirSet &reservoirs, std::string *error_out) {
  if (target_relay_count == 0u)
    return true;
  if (pristine_text.empty()) {
    report(error_out, "branch-only router cannot discover reservoirs without pristine text");
    return false;
  }

  std::vector<std::pair<uint64_t, uint64_t>> merged_ranges(protected_ranges.begin(),
                                                           protected_ranges.end());
  std::erase_if(merged_ranges, [](const auto &range) { return range.first >= range.second; });
  std::ranges::sort(merged_ranges);
  size_t merged_count = 0u;
  for (const auto &range : merged_ranges) {
    if (merged_count != 0u && merged_ranges[merged_count - 1u].second >= range.first) {
      merged_ranges[merged_count - 1u].second =
          std::max(merged_ranges[merged_count - 1u].second, range.second);
    } else {
      merged_ranges[merged_count++] = range;
    }
  }
  merged_ranges.resize(merged_count);
  const auto overlaps_protected = [&](uint64_t begin, uint64_t end) {
    const auto after = std::ranges::lower_bound(merged_ranges, end, {},
                                                [](const auto &range) { return range.first; });
    return after != merged_ranges.begin() && std::prev(after)->second > begin;
  };

  struct Candidate {
    uint64_t offset = 0;
    std::vector<uint32_t> words;
  };
  std::vector<Candidate> candidates;
  constexpr size_t kMinimumReservoirWords = 16u;
  constexpr size_t kMaximumReservoirWords = 64u;
  for (BasicBlock *block : blocks) {
    if (block == nullptr)
      continue;
    std::vector<const Instruction *> run;
    const auto flush_run = [&]() {
      size_t run_end = run.size();
      while (run_end != 0u) {
        size_t run_begin = run_end;
        size_t word_count = 0u;
        while (run_begin != 0u) {
          const size_t instruction_words =
              static_cast<size_t>(run[run_begin - 1u]->size()) / sizeof(uint32_t);
          if (word_count + instruction_words > kMaximumReservoirWords)
            break;
          word_count += instruction_words;
          --run_begin;
        }
        if (word_count >= kMinimumReservoirWords) {
          Candidate candidate;
          candidate.offset = run[run_begin]->src_loc();
          candidate.words.resize(word_count);
          std::memcpy(candidate.words.data(), pristine_text.data() + candidate.offset,
                      word_count * sizeof(uint32_t));
          candidates.push_back(std::move(candidate));
        }
        if (run_begin == run_end)
          break;
        run_end = run_begin;
      }
      run.clear();
    };

    std::optional<uint64_t> expected_offset;
    uint32_t clause_remaining = 0u;
    for (const Instruction &instruction : block->instructions()) {
      const uint64_t begin = instruction.src_loc();
      const uint64_t end =
          instruction.size() > 0 ? begin + static_cast<uint64_t>(instruction.size()) : begin;
      const bool clause_blocked = clause_remaining != 0u || instruction.mnemonic() == "s_clause";
      const bool admissible =
          !clause_blocked &&
          is_consan_branch_relay_reservoir_instruction(instruction, begin, pristine_text, arch) &&
          !overlaps_protected(begin, end) && (!expected_offset || begin == *expected_offset);
      if (!admissible)
        flush_run();
      if (instruction.mnemonic() == "s_clause" && instruction.raw_encoding() != nullptr)
        clause_remaining = (instruction.raw_encoding()[0] & 0x3fu) + 1u;
      else if (clause_remaining != 0u)
        --clause_remaining;
      if (!admissible) {
        expected_offset.reset();
        continue;
      }
      run.push_back(&instruction);
      expected_offset = end;
    }
    flush_run();
  }

  std::ranges::sort(candidates, [&](const Candidate &lhs, const Candidate &rhs) {
    const uint64_t lhs_distance =
        lhs.offset < route_midpoint ? route_midpoint - lhs.offset : lhs.offset - route_midpoint;
    const uint64_t rhs_distance =
        rhs.offset < route_midpoint ? route_midpoint - rhs.offset : rhs.offset - route_midpoint;
    return std::tie(lhs_distance, lhs.offset) < std::tie(rhs_distance, rhs.offset);
  });

  size_t offered_relay_count = 0u;
  for (const Candidate &candidate : candidates) {
    const uint64_t first_relay = candidate.offset + sizeof(uint32_t);
    const uint64_t candidate_end = candidate.offset + candidate.words.size() * sizeof(uint32_t);
    const auto first_existing = relays_.lower_bound(first_relay);
    if (first_existing != relays_.end() && first_existing->first < candidate_end)
      continue;

    DbiPatchPlacementPlanner tentative_planner = placement_planner;
    DbiPatchPlacementRequest request;
    request.anchor_offset = candidate.offset;
    request.original_size = static_cast<uint32_t>(candidate.words.size() * sizeof(uint32_t));
    request.body_size = request.original_size;
    request.inline_capacity = 0u;
    request.allow_appended_cave = true;
    const auto placement = tentative_planner.plan(request);
    if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave)
      continue;

    BranchOnlyDirectRelayReservoir reservoir{
        .anchor_offset = candidate.offset,
        .original_words = candidate.words,
        .placement = *placement,
    };
    const size_t reservoir_index = reservoirs.reservoirs.size();
    for (uint64_t word = 1u; word < candidate.words.size(); ++word) {
      const uint64_t relay = candidate.offset + word * sizeof(uint32_t);
      if (!offer(relay, BranchOnlyRelayProvenance::OwnedReservoir)) {
        report(error_out, "branch-only router could not atomically adopt a direct reservoir");
        return false;
      }
      reservoirs.reservoir_by_relay.emplace(relay, reservoir_index);
      ++offered_relay_count;
    }
    reservoirs.reservoirs.push_back(std::move(reservoir));
    placement_planner = std::move(tentative_planner);
    if (offered_relay_count >= target_relay_count)
      break;
  }
  return true;
}

bool BranchOnlyRelayRouter::emit_and_record(std::span<uint8_t> text,
                                            const BranchOnlyRelayRoute &route,
                                            uint64_t entry_target, uint64_t return_target,
                                            rj_code_arch_t arch,
                                            std::vector<ConSanPatchInfo> &patches,
                                            std::string *error_out) {
  const auto emit_route = [&](std::span<const uint64_t> relays, uint64_t target) {
    for (size_t index = 0; index < relays.size(); ++index) {
      const uint64_t source = relays[index];
      const uint64_t hop_target = index + 1u < relays.size() ? relays[index + 1u] : target;
      const auto delta = compute_sopp_branch_simm16(source, hop_target);
      if (!delta || source > text.size() || sizeof(uint32_t) > text.size() - source)
        return false;
      const uint32_t branch = build_s_branch(*delta, arch);
      std::memcpy(text.data() + source, &branch, sizeof(branch));
    }
    return true;
  };
  if (!emit_route(route.entry_relay_offsets, entry_target) ||
      !emit_route(route.return_relay_offsets, return_target)) {
    report(error_out, "branch-only router could not emit a planned relay route");
    return false;
  }

  for (const BranchOnlyRelayClaim &claim : route.claims) {
    if (claim.provenance == BranchOnlyRelayProvenance::OwnedAnchor ||
        claim.provenance == BranchOnlyRelayProvenance::OwnedReservoir)
      continue;
    ConSanPatchInfo relay_info;
    relay_info.kind = claim.provenance == BranchOnlyRelayProvenance::PristineNop
                          ? ConSanPatchKind::TrampolineNopBranchRelay
                          : ConSanPatchKind::TrampolineBranchRelayReservoir;
    relay_info.anchor_offset = claim.offset;
    relay_info.trampoline_offset = claim.offset;
    relay_info.original_size =
        claim.provenance == BranchOnlyRelayProvenance::PristineNop ? sizeof(uint32_t) : 0u;
    relay_info.trampoline_size = sizeof(uint32_t);
    patches.push_back(std::move(relay_info));
  }
  return true;
}

bool BranchOnlyRelayRouter::emit_direct_reservoir(std::vector<uint8_t> &text,
                                                  const BranchOnlyDirectRelayReservoir &reservoir,
                                                  rj_code_arch_t arch,
                                                  std::vector<ConSanPatchInfo> &patches,
                                                  std::string *error_out) {
  const uint64_t original_size = reservoir.original_words.size() * sizeof(uint32_t);
  if (reservoir.original_words.size() < 2u ||
      reservoir.placement.anchor_offset != reservoir.anchor_offset ||
      reservoir.placement.original_size != original_size ||
      reservoir.placement.body_size != original_size ||
      reservoir.placement.return_branch_offset != reservoir.placement.body_offset + original_size ||
      reservoir.placement.return_target != reservoir.anchor_offset + original_size) {
    report(error_out, "branch-only router found invalid direct-reservoir geometry");
    return false;
  }
  if (reservoir.placement.return_branch_offset >
      std::numeric_limits<uint64_t>::max() - sizeof(uint32_t)) {
    report(error_out, "branch-only router direct reservoir exceeds host address space");
    return false;
  }
  const uint64_t emitted_end = reservoir.placement.return_branch_offset + sizeof(uint32_t);
  if (emitted_end > std::numeric_limits<size_t>::max()) {
    report(error_out, "branch-only router direct reservoir exceeds host address space");
    return false;
  }
  if (text.size() < emitted_end)
    text.resize(static_cast<size_t>(emitted_end));
  if (!reservoir.used) {
    const uint32_t nop = build_s_nop(0, arch);
    for (uint64_t offset = reservoir.placement.body_offset; offset < emitted_end;
         offset += sizeof(uint32_t))
      std::memcpy(text.data() + offset, &nop, sizeof(nop));
    return true;
  }

  const auto entry_delta =
      compute_sopp_branch_simm16(reservoir.anchor_offset, reservoir.placement.body_offset);
  const auto return_delta = compute_sopp_branch_simm16(reservoir.placement.return_branch_offset,
                                                       reservoir.placement.return_target);
  if (!entry_delta || !return_delta || reservoir.anchor_offset > text.size() ||
      original_size > text.size() - reservoir.anchor_offset) {
    report(error_out, "branch-only router direct reservoir exceeds branch reach");
    return false;
  }
  const uint32_t nop = build_s_nop(0, arch);
  const uint32_t entry = build_s_branch(*entry_delta, arch);
  std::memcpy(text.data() + reservoir.anchor_offset, &entry, sizeof(entry));
  for (uint64_t offset = reservoir.anchor_offset + sizeof(uint32_t);
       offset < reservoir.anchor_offset + original_size; offset += sizeof(uint32_t))
    std::memcpy(text.data() + offset, &nop, sizeof(nop));

  const uint8_t *original_begin =
      reinterpret_cast<const uint8_t *>(reservoir.original_words.data());
  std::memcpy(text.data() + reservoir.placement.body_offset, original_begin, original_size);
  const uint32_t return_branch = build_s_branch(*return_delta, arch);
  std::memcpy(text.data() + reservoir.placement.return_branch_offset, &return_branch,
              sizeof(return_branch));

  ConSanPatchInfo info;
  info.kind = ConSanPatchKind::TrampolineBranchRelayReservoir;
  info.anchor_offset = reservoir.anchor_offset;
  info.trampoline_offset = reservoir.placement.body_offset;
  info.original_size = static_cast<uint32_t>(original_size);
  info.trampoline_size = static_cast<uint32_t>(original_size + sizeof(uint32_t));
  patches.push_back(std::move(info));
  return true;
}

} // namespace rocjitsu
