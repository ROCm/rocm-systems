// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
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

[[nodiscard]] size_t exact_batch_search_work(const BranchOnlyRelaySearchLimits &limits,
                                             size_t demand_count) {
  if (limits.batch_search_work_per_demand == 0u)
    return limits.batch_base_search_work;
  if (demand_count > (std::numeric_limits<size_t>::max() - limits.batch_base_search_work) /
                         limits.batch_search_work_per_demand) {
    return std::numeric_limits<size_t>::max();
  }
  return limits.batch_base_search_work + demand_count * limits.batch_search_work_per_demand;
}

[[nodiscard]] size_t multiply_saturated(size_t lhs, size_t rhs) {
  if (lhs != 0u && rhs > std::numeric_limits<size_t>::max() / lhs)
    return std::numeric_limits<size_t>::max();
  return lhs * rhs;
}

[[nodiscard]] size_t saturated_sum(size_t lhs, size_t rhs) {
  return rhs > std::numeric_limits<size_t>::max() - lhs ? std::numeric_limits<size_t>::max()
                                                        : lhs + rhs;
}

[[nodiscard]] size_t exact_batch_scan_work(const BranchOnlyRelaySearchLimits &limits,
                                           size_t demand_count, size_t relay_count) {
  const size_t input_units = multiply_saturated(demand_count, relay_count);
  const size_t headroom = multiply_saturated(input_units, limits.batch_scan_work_per_demand_relay);
  return saturated_sum(limits.batch_base_scan_work, headroom);
}

[[nodiscard]] size_t exact_pair_scan_work(const BranchOnlyRelaySearchLimits &limits,
                                          size_t relay_count) {
  return saturated_sum(limits.pair_base_scan_work,
                       multiply_saturated(relay_count, limits.pair_scan_work_per_relay));
}

void add_saturated(size_t &total, size_t value) { total = saturated_sum(total, value); }

class BoundedWorkMeter {
public:
  explicit BoundedWorkMeter(size_t limit) : limit_(std::max<size_t>(limit, 1u)) {}

  [[nodiscard]] bool consume(size_t amount = 1u) {
    if (amount > limit_ - consumed_) {
      consumed_ = limit_;
      exhausted_ = true;
      return false;
    }
    consumed_ += amount;
    return true;
  }

  [[nodiscard]] size_t consumed() const { return consumed_; }
  [[nodiscard]] bool exhausted() const { return exhausted_; }

private:
  size_t limit_ = 1u;
  size_t consumed_ = 0u;
  bool exhausted_ = false;
};

/// Exact disjoint-path solver for fixed SOPP source/target pairs.
///
/// The relay graph is a one-dimensional DAG: every hop moves monotonically
/// toward its target. The search assigns the most constrained remaining
/// demand first, prunes states whose independent shortest paths need more
/// relays than remain, and backtracks across both entry and return demands.
/// Route enumeration stops as soon as the target is directly reachable;
/// adding another relay at that point is strictly dominated because removing
/// it preserves the route while freeing capacity. Relay offsets must be
/// sorted and unique; the only caller derives them from the router's ordered
/// map. Separate deterministic budgets count search states and explored relay
/// alternatives, plus every relay inspection performed by polynomial
/// summaries.
class ExactFixedRelayBatchSolver {
public:
  enum class Result : uint8_t {
    Solved,
    Infeasible,
    WorkBudgetExhausted,
  };

  ExactFixedRelayBatchSolver(std::span<const FixedRelayDemand> demands,
                             std::span<const uint64_t> relays, BoundedWorkMeter &search_work,
                             BoundedWorkMeter &scan_work)
      : demands_(demands), relays_(relays), available_(relays.size(), true),
        assigned_(demands.size(), false), routes_(demands.size()), search_work_(search_work),
        scan_work_(scan_work) {
    assert(std::ranges::is_sorted(relays_));
    assert(std::ranges::adjacent_find(relays_) == relays_.end());
  }

  [[nodiscard]] Result solve(std::vector<std::vector<uint64_t>> &routes_out) {
    if (!solve_remaining(0u))
      return work_budget_exhausted() ? Result::WorkBudgetExhausted : Result::Infeasible;
    routes_out.resize(routes_.size());
    for (size_t demand = 0u; demand < routes_.size(); ++demand) {
      routes_out[demand].reserve(routes_[demand].size());
      for (size_t relay : routes_[demand])
        routes_out[demand].push_back(relays_[relay]);
    }
    return Result::Solved;
  }

private:
  struct DemandSummary {
    size_t minimum_relay_count = 0u;
    size_t first_hop_options = 0u;
    size_t corridor_relays = 0u;
  };

  [[nodiscard]] bool consume_search_work() { return search_work_.consume(); }

  [[nodiscard]] bool consume_scan_work() { return scan_work_.consume(); }

  [[nodiscard]] bool work_budget_exhausted() const {
    return search_work_.exhausted() || scan_work_.exhausted();
  }

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
                                                          uint64_t cursor) {
    size_t count = 0u;
    while (!can_hop(cursor, demand.target)) {
      std::optional<size_t> best;
      if (is_forward(demand)) {
        for (size_t relay = relays_.size(); relay-- != 0u;) {
          if (!consume_scan_work())
            return std::nullopt;
          if (available_[relay] && is_between(demand, cursor, relays_[relay]) &&
              can_hop(cursor, relays_[relay])) {
            best = relay;
            break;
          }
        }
      } else {
        for (size_t relay = 0u; relay < relays_.size(); ++relay) {
          if (!consume_scan_work())
            return std::nullopt;
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

  [[nodiscard]] std::optional<DemandSummary> summarize(const FixedRelayDemand &demand) {
    const auto minimum = minimum_relay_count(demand, demand.source);
    if (!minimum)
      return std::nullopt;
    DemandSummary summary{.minimum_relay_count = *minimum};
    for (size_t relay = 0u; relay < relays_.size(); ++relay) {
      if (!consume_scan_work())
        return std::nullopt;
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
          !can_hop(cursor, relays_[relay])) {
        return false;
      }
      if (!minimum_relay_count(demand, relays_[relay])) {
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
        if (work_budget_exhausted() || !consume_search_work())
          return false;
        if (try_relay(relay))
          return true;
      }
    } else {
      for (size_t relay = 0u; relay < relays_.size(); ++relay) {
        if (work_budget_exhausted() || !consume_search_work())
          return false;
        if (try_relay(relay))
          return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool solve_remaining(size_t assigned_count) {
    if (!consume_search_work())
      return false;
    if (assigned_count == demands_.size())
      return true;

    size_t available_count = 0u;
    for (bool available : available_) {
      if (!consume_scan_work())
        return false;
      available_count += available ? 1u : 0u;
    }

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
  BoundedWorkMeter &search_work_;
  BoundedWorkMeter &scan_work_;
};

struct GreedyFixedRelayRoute {
  enum class Status : uint8_t {
    Solved,
    Infeasible,
    WorkBudgetExhausted,
  };

  Status status = Status::Infeasible;
  std::vector<uint64_t> offsets;
};

/// Exact for one validated monotonic demand. On failure the relay set is
/// restored, so callers may use it transactionally without copying the whole
/// inventory. Each ordered-set query is charged by its logarithmic comparison
/// depth.
[[nodiscard]] GreedyFixedRelayRoute plan_greedy_fixed_relay_route(const FixedRelayDemand &demand,
                                                                  std::set<uint64_t> &unused_relays,
                                                                  BoundedWorkMeter &work) {
  assert(demand.source % sizeof(uint32_t) == 0u);
  assert(demand.target % sizeof(uint32_t) == 0u);
  assert(demand.source != demand.target);
  const bool forward = demand.target > demand.source;
  uint64_t cursor = demand.source;
  std::vector<uint64_t> route;
  while (!compute_sopp_branch_simm16(cursor, demand.target)) {
    const size_t query_work = std::max<size_t>(std::bit_width(unused_relays.size()), 1u);
    if (!work.consume(query_work)) {
      unused_relays.insert(route.begin(), route.end());
      return {
          .status = GreedyFixedRelayRoute::Status::WorkBudgetExhausted,
          .offsets = {},
      };
    }
    std::set<uint64_t>::iterator relay = unused_relays.end();
    if (forward) {
      const uint64_t limit =
          cursor > std::numeric_limits<uint64_t>::max() - kSoppBranchMaximumForwardReachBytes
              ? std::numeric_limits<uint64_t>::max()
              : cursor + kSoppBranchMaximumForwardReachBytes;
      const auto reachable_end =
          unused_relays.upper_bound(std::min(limit, demand.target - sizeof(uint32_t)));
      if (reachable_end != unused_relays.begin())
        relay = std::prev(reachable_end);
      if (relay == unused_relays.end() || *relay <= cursor || *relay >= demand.target ||
          !compute_sopp_branch_simm16(cursor, *relay)) {
        relay = unused_relays.end();
      }
    } else {
      const uint64_t limit = cursor > kSoppBranchMaximumBackwardReachBytes
                                 ? cursor - kSoppBranchMaximumBackwardReachBytes
                                 : 0u;
      relay = unused_relays.lower_bound(std::max(limit, demand.target + sizeof(uint32_t)));
      if (relay == unused_relays.end() || *relay >= cursor || *relay <= demand.target ||
          !compute_sopp_branch_simm16(cursor, *relay)) {
        relay = unused_relays.end();
      }
    }
    if (relay == unused_relays.end()) {
      unused_relays.insert(route.begin(), route.end());
      return {
          .status = GreedyFixedRelayRoute::Status::Infeasible,
          .offsets = {},
      };
    }
    cursor = *relay;
    route.push_back(cursor);
    unused_relays.erase(relay);
  }
  return {
      .status = GreedyFixedRelayRoute::Status::Solved,
      .offsets = std::move(route),
  };
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

std::optional<BranchOnlyRelayRoute> BranchOnlyRelayRouter::plan_pair(
    DbiPatchPlacementPlanner &tentative_planner, uint64_t entry_source, uint64_t entry_target,
    uint64_t return_source, uint64_t return_target, std::string *error_out,
    BranchOnlyRelayPlanOutcome *outcome_out, const BranchOnlyRelaySearchLimits &limits) const {
  const std::array requests = {
      BranchOnlyRelayPairRequest{entry_source, entry_target, return_source, return_target},
  };
  BranchOnlyRelayBatchPlan batch = plan_pairs(tentative_planner, requests, error_out, limits);
  if (outcome_out != nullptr)
    *outcome_out = batch;
  if (!batch.complete() || batch.routes.size() != 1u)
    return std::nullopt;
  return std::move(batch.routes.front());
}

BranchOnlyRelayBatchPlan
BranchOnlyRelayRouter::plan_pairs(DbiPatchPlacementPlanner &tentative_planner,
                                  std::span<const BranchOnlyRelayPairRequest> requests,
                                  std::string *error_out,
                                  const BranchOnlyRelaySearchLimits &limits) const {
  BranchOnlyRelayBatchPlan batch;
  batch.routes.resize(requests.size());
  batch.rejection_reasons.resize(requests.size());
  batch.pair_strategies.resize(requests.size(), BranchOnlyRelayPlanStrategy::ExactBatch);
  if (requests.empty())
    return batch;

  std::vector<uint64_t> relay_offsets;
  relay_offsets.reserve(relays_.size());
  for (const auto &[offset, provenance] : relays_) {
    (void)provenance;
    relay_offsets.push_back(offset);
  }

  std::vector<bool> invalid_entry(requests.size(), false);
  std::vector<bool> invalid_return(requests.size(), false);
  for (size_t request_index = 0u; request_index < requests.size(); ++request_index) {
    const BranchOnlyRelayPairRequest &request = requests[request_index];
    const std::array entry_endpoints = {request.entry_source, request.entry_target};
    const std::array return_endpoints = {request.return_source, request.return_target};
    invalid_entry[request_index] = request.entry_target <= request.entry_source ||
                                   std::ranges::any_of(entry_endpoints, [](uint64_t offset) {
                                     return offset % sizeof(uint32_t) != 0u;
                                   });
    invalid_return[request_index] = request.return_target >= request.return_source ||
                                    std::ranges::any_of(return_endpoints, [](uint64_t offset) {
                                      return offset % sizeof(uint32_t) != 0u;
                                    });
  }

  // Intrinsically invalid requests are already omitted from the batch.
  // Coordinate owners are removed as soon as a live collision rejects them,
  // so their stale coordinates cannot reject later requests either.
  std::map<uint64_t, size_t> entry_coordinate_owner;
  std::map<uint64_t, size_t> return_coordinate_owner;
  const auto unregister_coordinates = [&](size_t request_index) {
    const BranchOnlyRelayPairRequest &request = requests[request_index];
    for (uint64_t offset : {request.entry_source, request.entry_target}) {
      const auto owner = entry_coordinate_owner.find(offset);
      if (owner != entry_coordinate_owner.end() && owner->second == request_index)
        entry_coordinate_owner.erase(owner);
    }
    for (uint64_t offset : {request.return_source, request.return_target}) {
      const auto owner = return_coordinate_owner.find(offset);
      if (owner != return_coordinate_owner.end() && owner->second == request_index)
        return_coordinate_owner.erase(owner);
    }
  };
  for (size_t request_index = 0u; request_index < requests.size(); ++request_index) {
    if (invalid_entry[request_index] || invalid_return[request_index])
      continue;
    const BranchOnlyRelayPairRequest &request = requests[request_index];
    const std::array entry_endpoints = {request.entry_source, request.entry_target};
    const std::array return_endpoints = {request.return_source, request.return_target};
    const auto register_coordinates = [&](std::span<const uint64_t> coordinates,
                                          std::map<uint64_t, size_t> &owners,
                                          std::vector<bool> &invalid) {
      for (uint64_t offset : coordinates) {
        const auto owner = owners.find(offset);
        if (owner == owners.end()) {
          owners.emplace(offset, request_index);
          continue;
        }
        const size_t conflicting_request = owner->second;
        invalid[request_index] = true;
        invalid[conflicting_request] = true;
        unregister_coordinates(request_index);
        unregister_coordinates(conflicting_request);
        return false;
      }
      return true;
    };
    if (!register_coordinates(entry_endpoints, entry_coordinate_owner, invalid_entry))
      continue;
    (void)register_coordinates(return_endpoints, return_coordinate_owner, invalid_return);
  }

  std::vector<bool> valid_request(requests.size(), true);
  for (size_t request_index = 0u; request_index < requests.size(); ++request_index) {
    if (!invalid_entry[request_index] && !invalid_return[request_index])
      continue;
    valid_request[request_index] = false;
    batch.rejection_reasons[request_index] =
        invalid_entry[request_index] ? BranchOnlyRelayPairRejection::InvalidEntryCoordinates
                                     : BranchOnlyRelayPairRejection::InvalidReturnCoordinates;
    batch.rejected_pair_indices.push_back(request_index);
  }

  std::set<uint64_t> pair_coordinates;
  for (size_t request_index = 0u; request_index < requests.size(); ++request_index) {
    if (!valid_request[request_index])
      continue;
    const BranchOnlyRelayPairRequest &request = requests[request_index];
    for (uint64_t offset :
         {request.entry_source, request.entry_target, request.return_source, request.return_target})
      pair_coordinates.insert(offset);
  }

  // Offered storage that aliases a branch source or destination is not relay
  // capacity. A successful commit retires it so later plans cannot overwrite a
  // branch endpoint; a rejected transaction leaves router state untouched.
  std::erase_if(relay_offsets, [&](uint64_t relay) { return pair_coordinates.contains(relay); });

  std::vector<FixedRelayDemand> demands;
  demands.reserve(2u * requests.size());
  for (size_t request_index = 0u; request_index < requests.size(); ++request_index) {
    if (!valid_request[request_index])
      continue;
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
    if (!valid_request[request_index])
      continue;
    demands.push_back({
        .pair_index = request_index,
        .entry = false,
        .source = requests[request_index].return_source,
        .target = requests[request_index].return_target,
    });
  }

  ExactFixedRelayBatchSolver::Result exact_result = ExactFixedRelayBatchSolver::Result::Infeasible;
  std::vector<std::vector<uint64_t>> solved_routes;
  if (!demands.empty()) {
    BoundedWorkMeter search_work(exact_batch_search_work(limits, demands.size()));
    BoundedWorkMeter scan_work(exact_batch_scan_work(limits, demands.size(), relay_offsets.size()));
    if (!scan_work.consume(relay_offsets.size())) {
      exact_result = ExactFixedRelayBatchSolver::Result::WorkBudgetExhausted;
    } else {
      ExactFixedRelayBatchSolver exact_solver(demands, relay_offsets, search_work, scan_work);
      exact_result = exact_solver.solve(solved_routes);
    }
    add_saturated(batch.search_work_consumed, search_work.consumed());
    add_saturated(batch.scan_work_consumed, scan_work.consumed());
  }
  if (exact_result == ExactFixedRelayBatchSolver::Result::Solved) {
    for (size_t demand_index = 0u; demand_index < demands.size(); ++demand_index) {
      const FixedRelayDemand &demand = demands[demand_index];
      std::vector<uint64_t> &route = demand.entry
                                         ? batch.routes[demand.pair_index].entry_relay_offsets
                                         : batch.routes[demand.pair_index].return_relay_offsets;
      route = std::move(solved_routes[demand_index]);
    }
  } else if (!demands.empty()) {
    batch.strategy = BranchOnlyRelayPlanStrategy::ExactPairFallback;
    batch.work_budget_exhausted =
        exact_result == ExactFixedRelayBatchSolver::Result::WorkBudgetExhausted;

    // A failed or bounded full assignment still returns pair-atomic partial
    // routes. Each pair gets a bounded exact solve before the final greedy tier
    // so entry routing cannot strand an otherwise feasible return route.
    BoundedWorkMeter fallback_setup_work(limits.batch_fallback_setup_work);
    const bool fallback_inventory_available = fallback_setup_work.consume(relay_offsets.size());
    add_saturated(batch.scan_work_consumed, fallback_setup_work.consumed());
    std::set<uint64_t> unused_relays;
    if (fallback_inventory_available)
      unused_relays.insert(relay_offsets.begin(), relay_offsets.end());
    for (size_t request_index = 0u; request_index < requests.size(); ++request_index) {
      if (!valid_request[request_index])
        continue;
      batch.pair_strategies[request_index] = BranchOnlyRelayPlanStrategy::ExactPairFallback;
      if (!fallback_inventory_available) {
        batch.work_budget_exhausted = true;
        batch.rejected_pair_indices.push_back(request_index);
        batch.rejection_reasons[request_index] = BranchOnlyRelayPairRejection::WorkBudget;
        continue;
      }
      const BranchOnlyRelayPairRequest &request = requests[request_index];
      const std::array pair_demands = {
          FixedRelayDemand{request_index, true, request.entry_source, request.entry_target},
          FixedRelayDemand{request_index, false, request.return_source, request.return_target},
      };

      BoundedWorkMeter pair_search_work(limits.pair_search_work);
      BoundedWorkMeter pair_scan_work(exact_pair_scan_work(limits, unused_relays.size()));
      std::vector<std::vector<uint64_t>> pair_routes;
      ExactFixedRelayBatchSolver::Result pair_result =
          ExactFixedRelayBatchSolver::Result::WorkBudgetExhausted;
      if (pair_scan_work.consume(unused_relays.size())) {
        const std::vector<uint64_t> available_relays(unused_relays.begin(), unused_relays.end());
        ExactFixedRelayBatchSolver pair_solver(pair_demands, available_relays, pair_search_work,
                                               pair_scan_work);
        pair_result = pair_solver.solve(pair_routes);
      }
      add_saturated(batch.search_work_consumed, pair_search_work.consumed());
      add_saturated(batch.scan_work_consumed, pair_scan_work.consumed());
      if (pair_result == ExactFixedRelayBatchSolver::Result::Solved) {
        batch.routes[request_index].entry_relay_offsets = std::move(pair_routes[0]);
        batch.routes[request_index].return_relay_offsets = std::move(pair_routes[1]);
        for (uint64_t relay : batch.routes[request_index].entry_relay_offsets)
          unused_relays.erase(relay);
        for (uint64_t relay : batch.routes[request_index].return_relay_offsets)
          unused_relays.erase(relay);
        continue;
      }

      if (pair_result == ExactFixedRelayBatchSolver::Result::WorkBudgetExhausted) {
        batch.strategy = BranchOnlyRelayPlanStrategy::GreedyPairFallback;
        batch.pair_strategies[request_index] = BranchOnlyRelayPlanStrategy::GreedyPairFallback;
        batch.work_budget_exhausted = true;
        BoundedWorkMeter greedy_work(limits.pair_greedy_work);
        GreedyFixedRelayRoute entry_route =
            plan_greedy_fixed_relay_route(pair_demands[0], unused_relays, greedy_work);
        GreedyFixedRelayRoute return_route;
        if (entry_route.status == GreedyFixedRelayRoute::Status::Solved) {
          return_route = plan_greedy_fixed_relay_route(pair_demands[1], unused_relays, greedy_work);
        }
        add_saturated(batch.scan_work_consumed, greedy_work.consumed());
        if (entry_route.status == GreedyFixedRelayRoute::Status::Solved &&
            return_route.status == GreedyFixedRelayRoute::Status::Solved) {
          batch.routes[request_index].entry_relay_offsets = std::move(entry_route.offsets);
          batch.routes[request_index].return_relay_offsets = std::move(return_route.offsets);
          continue;
        }
        unused_relays.insert(entry_route.offsets.begin(), entry_route.offsets.end());
        if (entry_route.status == GreedyFixedRelayRoute::Status::WorkBudgetExhausted ||
            return_route.status == GreedyFixedRelayRoute::Status::WorkBudgetExhausted) {
          batch.rejected_pair_indices.push_back(request_index);
          batch.rejection_reasons[request_index] = BranchOnlyRelayPairRejection::WorkBudget;
          continue;
        }
      }

      batch.rejected_pair_indices.push_back(request_index);

      // A single monotonic demand is feasible exactly when farthest-progress
      // greedy routing succeeds. Probe each half independently so shared relay
      // contention is not misreported as an unreachable return corridor. The
      // probes restore successful routes and share one bounded meter.
      BoundedWorkMeter classification_work(limits.pair_greedy_work);
      const auto probe = [&](const FixedRelayDemand &demand) {
        GreedyFixedRelayRoute result =
            plan_greedy_fixed_relay_route(demand, unused_relays, classification_work);
        unused_relays.insert(result.offsets.begin(), result.offsets.end());
        return result.status;
      };
      const GreedyFixedRelayRoute::Status entry_status = probe(pair_demands[0]);
      const GreedyFixedRelayRoute::Status return_status =
          entry_status == GreedyFixedRelayRoute::Status::WorkBudgetExhausted
              ? GreedyFixedRelayRoute::Status::WorkBudgetExhausted
              : probe(pair_demands[1]);
      add_saturated(batch.scan_work_consumed, classification_work.consumed());
      if (entry_status == GreedyFixedRelayRoute::Status::WorkBudgetExhausted ||
          return_status == GreedyFixedRelayRoute::Status::WorkBudgetExhausted) {
        batch.work_budget_exhausted = true;
        batch.rejection_reasons[request_index] = BranchOnlyRelayPairRejection::WorkBudget;
        continue;
      }
      batch.rejection_reasons[request_index] =
          entry_status != GreedyFixedRelayRoute::Status::Solved
              ? BranchOnlyRelayPairRejection::EntryUnreachable
          : return_status != GreedyFixedRelayRoute::Status::Solved
              ? BranchOnlyRelayPairRejection::ReturnUnreachable
              : BranchOnlyRelayPairRejection::RelayContention;
    }
  }

  if (!batch.rejected_pair_indices.empty()) {
    std::ranges::sort(batch.rejected_pair_indices);
    batch.rejected_pair_indices.erase(std::ranges::unique(batch.rejected_pair_indices).begin(),
                                      batch.rejected_pair_indices.end());

    std::array<size_t, kBranchOnlyRelayPairRejectionCount> rejection_counts{};
    for (BranchOnlyRelayPairRejection reason : batch.rejection_reasons) {
      const size_t index = static_cast<size_t>(reason);
      assert(index < rejection_counts.size());
      ++rejection_counts[index];
    }
    const size_t invalid_entry_count = rejection_counts[static_cast<size_t>(
        BranchOnlyRelayPairRejection::InvalidEntryCoordinates)];
    const size_t invalid_return_count = rejection_counts[static_cast<size_t>(
        BranchOnlyRelayPairRejection::InvalidReturnCoordinates)];
    const size_t entry_unreachable_count =
        rejection_counts[static_cast<size_t>(BranchOnlyRelayPairRejection::EntryUnreachable)];
    const size_t return_unreachable_count =
        rejection_counts[static_cast<size_t>(BranchOnlyRelayPairRejection::ReturnUnreachable)];
    const size_t relay_contention_count =
        rejection_counts[static_cast<size_t>(BranchOnlyRelayPairRejection::RelayContention)];
    const size_t work_budget_count =
        rejection_counts[static_cast<size_t>(BranchOnlyRelayPairRejection::WorkBudget)];

    // Malformed coordinates outrank resource/search outcomes in this coarse
    // batch summary. The indexed rejection vector remains the source of truth
    // when a batch contains multiple causes.
    if (invalid_entry_count != 0u) {
      batch.failure = BranchOnlyRelayPlanFailure::EntryRoute;
    } else if (invalid_return_count != 0u) {
      batch.failure = BranchOnlyRelayPlanFailure::ReturnRoute;
    } else if (work_budget_count != 0u) {
      batch.failure = BranchOnlyRelayPlanFailure::WorkBudget;
    } else if (entry_unreachable_count != 0u) {
      batch.failure = BranchOnlyRelayPlanFailure::EntryRoute;
    } else if (return_unreachable_count != 0u) {
      batch.failure = BranchOnlyRelayPlanFailure::ReturnRoute;
    } else {
      batch.failure = BranchOnlyRelayPlanFailure::RelayContention;
    }

    const auto pair_count_phrase = [](size_t count, std::string_view description) {
      return std::to_string(count) + " " + std::string(description) +
             (count == 1u ? " pair" : " pairs");
    };
    std::vector<std::string> reasons;
    if (invalid_entry_count != 0u)
      reasons.push_back(pair_count_phrase(
          invalid_entry_count,
          "invalid entry-coordinate (coordinates must be distinct, dword-aligned, and monotonic)"));
    if (invalid_return_count != 0u)
      reasons.push_back(pair_count_phrase(invalid_return_count,
                                          "invalid return-coordinate (coordinates must be "
                                          "distinct, dword-aligned, and monotonic)"));
    if (entry_unreachable_count != 0u)
      reasons.push_back(
          pair_count_phrase(entry_unreachable_count, "unreachable appended entry route"));
    if (return_unreachable_count != 0u)
      reasons.push_back(
          pair_count_phrase(return_unreachable_count, "unreachable original continuation route"));
    if (relay_contention_count != 0u)
      reasons.push_back(pair_count_phrase(relay_contention_count, "relay-contended"));
    if (work_budget_count != 0u)
      reasons.push_back(pair_count_phrase(work_budget_count, "work-budget-limited"));
    if (batch.work_budget_exhausted)
      reasons.emplace_back("batch work was bounded; plan may be suboptimal");

    std::string diagnostic = "branch-only router rejected " +
                             std::to_string(batch.rejected_pair_indices.size()) +
                             (batch.rejected_pair_indices.size() == 1u ? " pair: " : " pairs: ");
    for (size_t reason = 0u; reason < reasons.size(); ++reason) {
      if (reason != 0u)
        diagnostic += "; ";
      diagnostic += reasons[reason];
    }
    report(error_out, std::move(diagnostic));
  }

  for (size_t request_index = 0u; request_index < requests.size(); ++request_index) {
    if (batch.rejection_reasons[request_index] != BranchOnlyRelayPairRejection::None)
      continue;
    const BranchOnlyRelayPairRequest &request = requests[request_index];
    for (uint64_t offset : {request.entry_source, request.entry_target, request.return_source,
                            request.return_target}) {
      const auto relay = relays_.find(offset);
      if (relay == relays_.end())
        continue;
      std::vector<BranchOnlyRelayClaim> &retired = batch.routes[request_index].retired_relay_claims;
      if (std::ranges::find(retired, offset, &BranchOnlyRelayClaim::offset) == retired.end())
        retired.push_back({offset, relay->second});
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
  std::unordered_set<uint64_t> retired_offsets;
  for (const BranchOnlyRelayRoute &route : routes) {
    for (const BranchOnlyRelayClaim &retired : route.retired_relay_claims) {
      const auto relay = relays_.find(retired.offset);
      if (relay == relays_.end() || relay->second != retired.provenance ||
          claimed_offsets.contains(retired.offset)) {
        report(error_out, "branch-only router endpoint retirement changed before commit");
        return false;
      }
      retired_offsets.insert(retired.offset);
    }
  }
  for (uint64_t offset : claimed_offsets)
    relays_.erase(offset);
  for (uint64_t offset : retired_offsets)
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
