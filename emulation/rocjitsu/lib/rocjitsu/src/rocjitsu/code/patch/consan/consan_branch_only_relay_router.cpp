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
#include <set>
#include <tuple>
#include <unordered_set>

namespace rocjitsu {

namespace {

void report(std::string *error_out, std::string error) {
  if (error_out != nullptr)
    *error_out = std::move(error);
}

struct FixedRelayDemand {
  size_t pair_index = 0u;
  bool entry = false;
  uint64_t source = 0u;
  uint64_t target = 0u;
};

/// Exact disjoint-path solver for fixed SOPP source/target pairs.
///
/// The relay graph is a one-dimensional DAG: every hop moves monotonically
/// toward its target. The search assigns the most constrained remaining
/// demand first, prunes states whose independent shortest paths need more
/// relays than remain, and backtracks across both entry and return demands.
/// Route enumeration stops as soon as the target is directly reachable;
/// adding another relay at that point is strictly dominated because removing
/// it preserves the route while freeing capacity.
class ExactFixedRelayBatchSolver {
public:
  ExactFixedRelayBatchSolver(std::span<const FixedRelayDemand> demands,
                             std::span<const uint64_t> relays)
      : demands_(demands), relays_(relays), available_(relays.size(), true),
        assigned_(demands.size(), false), routes_(demands.size()) {}

  [[nodiscard]] bool solve(std::vector<std::vector<uint64_t>> &routes_out) {
    if (!solve_remaining(0u))
      return false;
    routes_out.resize(routes_.size());
    for (size_t demand = 0u; demand < routes_.size(); ++demand) {
      routes_out[demand].reserve(routes_[demand].size());
      for (size_t relay : routes_[demand])
        routes_out[demand].push_back(relays_[relay]);
    }
    return true;
  }

private:
  struct DemandSummary {
    size_t minimum_relay_count = 0u;
    size_t first_hop_options = 0u;
    size_t corridor_relays = 0u;
  };

  [[nodiscard]] bool is_forward(const FixedRelayDemand &demand) const {
    return demand.target > demand.source;
  }

  [[nodiscard]] bool is_between(const FixedRelayDemand &demand, uint64_t cursor,
                                uint64_t relay) const {
    return is_forward(demand) ? cursor < relay && relay < demand.target
                              : demand.target < relay && relay < cursor;
  }

  [[nodiscard]] bool can_hop(uint64_t source, uint64_t target) const {
    return compute_sopp_branch_simm16(source, target).has_value();
  }

  [[nodiscard]] std::optional<size_t> minimum_relay_count(const FixedRelayDemand &demand,
                                                          uint64_t cursor) const {
    size_t count = 0u;
    while (!can_hop(cursor, demand.target)) {
      std::optional<size_t> best;
      if (is_forward(demand)) {
        for (size_t relay = relays_.size(); relay-- != 0u;) {
          if (available_[relay] && is_between(demand, cursor, relays_[relay]) &&
              can_hop(cursor, relays_[relay])) {
            best = relay;
            break;
          }
        }
      } else {
        for (size_t relay = 0u; relay < relays_.size(); ++relay) {
          if (available_[relay] && is_between(demand, cursor, relays_[relay]) &&
              can_hop(cursor, relays_[relay])) {
            best = relay;
            break;
          }
        }
      }
      if (!best)
        return std::nullopt;
      cursor = relays_[*best];
      ++count;
    }
    return count;
  }

  [[nodiscard]] std::optional<DemandSummary> summarize(const FixedRelayDemand &demand) const {
    const auto minimum = minimum_relay_count(demand, demand.source);
    if (!minimum)
      return std::nullopt;
    DemandSummary summary{.minimum_relay_count = *minimum};
    for (size_t relay = 0u; relay < relays_.size(); ++relay) {
      if (!available_[relay] || !is_between(demand, demand.source, relays_[relay]))
        continue;
      ++summary.corridor_relays;
      if (can_hop(demand.source, relays_[relay]))
        ++summary.first_hop_options;
    }
    return summary;
  }

  template <typename Callback>
  [[nodiscard]] bool enumerate_routes(size_t demand_index, uint64_t cursor,
                                      std::vector<size_t> &route, Callback &callback) {
    const FixedRelayDemand &demand = demands_[demand_index];
    if (can_hop(cursor, demand.target))
      return callback(route);

    const auto try_relay = [&](size_t relay) {
      if (!available_[relay] || !is_between(demand, cursor, relays_[relay]) ||
          !can_hop(cursor, relays_[relay]) || !minimum_relay_count(demand, relays_[relay])) {
        return false;
      }
      route.push_back(relay);
      const bool accepted = enumerate_routes(demand_index, relays_[relay], route, callback);
      route.pop_back();
      return accepted;
    };

    // Prefer maximum progress to retain stable, short routes. Completeness
    // comes from exploring every alternative when a later demand is stranded.
    if (is_forward(demand)) {
      for (size_t relay = relays_.size(); relay-- != 0u;) {
        if (try_relay(relay))
          return true;
      }
    } else {
      for (size_t relay = 0u; relay < relays_.size(); ++relay) {
        if (try_relay(relay))
          return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool solve_remaining(size_t assigned_count) {
    if (assigned_count == demands_.size())
      return true;

    size_t available_count = 0u;
    for (bool available : available_)
      available_count += available ? 1u : 0u;

    std::optional<size_t> selected;
    DemandSummary selected_summary;
    size_t required_relays = 0u;
    for (size_t demand = 0u; demand < demands_.size(); ++demand) {
      if (assigned_[demand])
        continue;
      const auto summary = summarize(demands_[demand]);
      if (!summary)
        return false;
      required_relays += summary->minimum_relay_count;
      if (required_relays > available_count)
        return false;
      const auto score = [](const DemandSummary &value) {
        return std::tuple{value.minimum_relay_count == 0u ? 0u : value.first_hop_options + 1u,
                          value.corridor_relays,
                          std::numeric_limits<size_t>::max() - value.minimum_relay_count};
      };
      if (!selected || score(*summary) < score(selected_summary)) {
        selected = demand;
        selected_summary = *summary;
      }
    }
    if (!selected)
      return false;

    assigned_[*selected] = true;
    std::vector<size_t> candidate_route;
    const auto accept_route = [&](const std::vector<size_t> &route) {
      for (size_t relay : route)
        available_[relay] = false;
      routes_[*selected] = route;
      const bool solved = solve_remaining(assigned_count + 1u);
      if (!solved) {
        routes_[*selected].clear();
        for (size_t relay : route)
          available_[relay] = true;
      }
      return solved;
    };
    const bool solved =
        enumerate_routes(*selected, demands_[*selected].source, candidate_route, accept_route);
    if (!solved)
      assigned_[*selected] = false;
    return solved;
  }

  std::span<const FixedRelayDemand> demands_;
  std::span<const uint64_t> relays_;
  std::vector<bool> available_;
  std::vector<bool> assigned_;
  std::vector<std::vector<size_t>> routes_;
};

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

  std::map<uint64_t, size_t> endpoint_owner;
  std::set<uint64_t> entry_coordinates;
  std::set<uint64_t> return_coordinates;
  for (size_t request_index = 0u; request_index < requests.size(); ++request_index) {
    const BranchOnlyRelayPairRequest &request = requests[request_index];
    const std::array entry_endpoints = {request.entry_source, request.entry_target};
    const std::array return_endpoints = {request.return_source, request.return_target};
    if (std::ranges::any_of(entry_endpoints,
                            [](uint64_t offset) { return offset % sizeof(uint32_t) != 0u; }) ||
        !entry_coordinates.insert(request.entry_source).second ||
        !entry_coordinates.insert(request.entry_target).second) {
      batch.failure = BranchOnlyRelayPlanFailure::EntryRoute;
      batch.rejected_pair_indices.resize(requests.size());
      std::iota(batch.rejected_pair_indices.begin(), batch.rejected_pair_indices.end(), 0u);
      report(error_out, "branch-only router requires distinct dword-aligned entry coordinates");
      return batch;
    }
    if (std::ranges::any_of(return_endpoints,
                            [](uint64_t offset) { return offset % sizeof(uint32_t) != 0u; }) ||
        !return_coordinates.insert(request.return_source).second ||
        !return_coordinates.insert(request.return_target).second) {
      batch.failure = BranchOnlyRelayPlanFailure::ReturnRoute;
      batch.rejected_pair_indices.resize(requests.size());
      std::iota(batch.rejected_pair_indices.begin(), batch.rejected_pair_indices.end(), 0u);
      report(error_out, "branch-only router requires distinct dword-aligned return coordinates");
      return batch;
    }
    for (uint64_t offset : entry_endpoints)
      endpoint_owner.try_emplace(offset, request_index);
    for (uint64_t offset : return_endpoints)
      endpoint_owner.try_emplace(offset, request_index);
    if (request.entry_target <= request.entry_source ||
        request.return_target >= request.return_source) {
      batch.failure = request.entry_target <= request.entry_source
                          ? BranchOnlyRelayPlanFailure::EntryRoute
                          : BranchOnlyRelayPlanFailure::ReturnRoute;
      batch.rejected_pair_indices.resize(requests.size());
      std::iota(batch.rejected_pair_indices.begin(), batch.rejected_pair_indices.end(), 0u);
      report(error_out, "branch-only router requires monotonic entry and return routes");
      return batch;
    }
  }
  for (uint64_t relay : relay_offsets) {
    const auto endpoint = endpoint_owner.find(relay);
    if (endpoint == endpoint_owner.end())
      continue;
    batch.failure = BranchOnlyRelayPlanFailure::Reservation;
    batch.rejected_pair_indices.push_back(endpoint->second);
    report(error_out, "branch-only router relay aliases a fixed pair coordinate");
    return batch;
  }

  std::vector<FixedRelayDemand> demands;
  demands.reserve(2u * requests.size());
  for (size_t request_index = 0u; request_index < requests.size(); ++request_index) {
    demands.push_back({
        .pair_index = request_index,
        .entry = true,
        .source = requests[request_index].entry_source,
        .target = requests[request_index].entry_target,
    });
  }
  // Preserve the established deterministic return preference as a tie-break,
  // while the solver remains free to pick a more constrained demand first.
  for (size_t request_index = requests.size(); request_index-- != 0u;) {
    demands.push_back({
        .pair_index = request_index,
        .entry = false,
        .source = requests[request_index].return_source,
        .target = requests[request_index].return_target,
    });
  }

  std::vector<std::vector<uint64_t>> solved_routes;
  ExactFixedRelayBatchSolver exact_solver(demands, relay_offsets);
  if (exact_solver.solve(solved_routes)) {
    for (size_t demand_index = 0u; demand_index < demands.size(); ++demand_index) {
      const FixedRelayDemand &demand = demands[demand_index];
      std::vector<uint64_t> &route = demand.entry
                                         ? batch.routes[demand.pair_index].entry_relay_offsets
                                         : batch.routes[demand.pair_index].return_relay_offsets;
      route = std::move(solved_routes[demand_index]);
    }
  } else {
    // A failed full assignment still returns pair-atomic partial routes. These
    // claims let the convergence loop materialize useful optimistic storage,
    // but a rejected pair never consumes capacity or reports phantom claims.
    std::set<uint64_t> unused_relays(relay_offsets.begin(), relay_offsets.end());
    bool rejected_entry = false;
    for (size_t request_index = 0u; request_index < requests.size(); ++request_index) {
      const BranchOnlyRelayPairRequest &request = requests[request_index];
      const std::array pair_demands = {
          FixedRelayDemand{request_index, true, request.entry_source, request.entry_target},
          FixedRelayDemand{request_index, false, request.return_source, request.return_target},
      };
      const std::vector<uint64_t> available_relays(unused_relays.begin(), unused_relays.end());
      std::vector<std::vector<uint64_t>> pair_routes;
      ExactFixedRelayBatchSolver pair_solver(pair_demands, available_relays);
      if (!pair_solver.solve(pair_routes)) {
        const std::array entry_demand = {pair_demands.front()};
        std::vector<std::vector<uint64_t>> ignored;
        ExactFixedRelayBatchSolver entry_solver(entry_demand, available_relays);
        rejected_entry |= !entry_solver.solve(ignored);
        batch.rejected_pair_indices.push_back(request_index);
        continue;
      }
      batch.routes[request_index].entry_relay_offsets = std::move(pair_routes[0]);
      batch.routes[request_index].return_relay_offsets = std::move(pair_routes[1]);
      for (uint64_t relay : batch.routes[request_index].entry_relay_offsets)
        unused_relays.erase(relay);
      for (uint64_t relay : batch.routes[request_index].return_relay_offsets)
        unused_relays.erase(relay);
    }
    batch.failure = rejected_entry ? BranchOnlyRelayPlanFailure::EntryRoute
                                   : BranchOnlyRelayPlanFailure::ReturnRoute;
    report(error_out, batch.failure == BranchOnlyRelayPlanFailure::EntryRoute
                          ? "branch-only router could not reach every appended entry"
                          : "branch-only router could not reach every original continuation");
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

  const DbiPatchPlacementPlanner original_planner = placement_planner;
  const size_t original_reservoir_count = reservoirs.reservoirs.size();
  std::vector<uint64_t> call_adopted_relays;
  const auto rollback_call = [&]() {
    for (uint64_t relay : call_adopted_relays) {
      relays_.erase(relay);
      reservoirs.reservoir_by_relay.erase(relay);
    }
    reservoirs.reservoirs.resize(original_reservoir_count);
    placement_planner = original_planner;
  };

  size_t offered_relay_count = 0u;
  for (const Candidate &candidate : candidates) {
    const uint64_t candidate_end = candidate.offset + candidate.words.size() * sizeof(uint32_t);
    const auto first_existing = relays_.lower_bound(candidate.offset);
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
    std::vector<uint64_t> adopted_relays;
    adopted_relays.reserve(candidate.words.size() - 1u);
    for (uint64_t word = 1u; word < candidate.words.size(); ++word) {
      const uint64_t relay = candidate.offset + word * sizeof(uint32_t);
      const bool offered = offer(relay, BranchOnlyRelayProvenance::OwnedReservoir);
      const bool indexed =
          offered && reservoirs.reservoir_by_relay.emplace(relay, reservoir_index).second;
      if (!offered || !indexed) {
        if (offered)
          relays_.erase(relay);
        for (uint64_t adopted : adopted_relays) {
          relays_.erase(adopted);
          reservoirs.reservoir_by_relay.erase(adopted);
        }
        rollback_call();
        report(error_out, "branch-only router could not atomically adopt a direct reservoir");
        return false;
      }
      adopted_relays.push_back(relay);
    }
    offered_relay_count += adopted_relays.size();
    reservoirs.reservoirs.push_back(std::move(reservoir));
    placement_planner = std::move(tentative_planner);
    call_adopted_relays.insert(call_adopted_relays.end(), adopted_relays.begin(),
                               adopted_relays.end());
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
  if (emitted_end > text.max_size()) {
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
