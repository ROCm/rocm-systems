// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
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

[[nodiscard]] bool fixed_relay_demand_is_forward(const FixedRelayDemand &demand) {
  return demand.target > demand.source;
}

[[nodiscard]] bool fixed_relay_is_between(const FixedRelayDemand &demand, uint64_t cursor,
                                          uint64_t relay) {
  return fixed_relay_demand_is_forward(demand) ? cursor < relay && relay < demand.target
                                               : demand.target < relay && relay < cursor;
}

[[nodiscard]] bool fixed_relay_can_hop(uint64_t source, uint64_t target) {
  return compute_sopp_branch_simm16(source, target).has_value();
}

[[nodiscard]] uint64_t direct_reservoir_appended_bytes(uint64_t displaced_bytes) {
  // Saturate so the caller's representation and address-space guards observe
  // an oversized footprint instead of wrapped geometry.
  return displaced_bytes > std::numeric_limits<uint64_t>::max() - sizeof(uint32_t)
             ? std::numeric_limits<uint64_t>::max()
             : displaced_bytes + sizeof(uint32_t);
}

[[nodiscard]] size_t exact_batch_search_work(const BranchOnlyRelaySearchLimits &limits,
                                             size_t demand_count) {
  return saturated_add(limits.batch.feasibility_search.base,
                       saturated_multiply(demand_count, limits.batch.feasibility_search.per_input));
}

[[nodiscard]] size_t exact_batch_scan_work(const BranchOnlyRelaySearchLimits &limits,
                                           size_t demand_count, size_t relay_count) {
  const size_t input_units = saturated_multiply(demand_count, relay_count);
  const size_t headroom = saturated_multiply(input_units, limits.batch.feasibility_scan.per_input);
  return saturated_add(limits.batch.feasibility_scan.base, headroom);
}

[[nodiscard]] size_t relay_qualification_work_limit(const BranchOnlyRelaySearchLimits &limits,
                                                    size_t relay_count,
                                                    size_t occupied_range_count) {
  const size_t relay_headroom = saturated_multiply(relay_count, limits.qualification.per_relay);
  const size_t range_query_levels = std::max<size_t>(std::bit_width(occupied_range_count), 1u);
  const size_t relay_range_levels = saturated_multiply(relay_count, range_query_levels);
  const size_t range_headroom =
      saturated_multiply(relay_range_levels, limits.qualification.per_relay_range_level);
  return saturated_add(limits.qualification.base, saturated_add(relay_headroom, range_headroom));
}

[[nodiscard]] size_t relay_qualification_lookup_work(size_t endpoint_count) {
  return saturated_add(1u, std::max<size_t>(std::bit_width(endpoint_count), 1u));
}

[[nodiscard]] size_t exact_pair_search_work(const BranchOnlyRelaySearchLimits &limits,
                                            size_t demand_count) {
  return saturated_add(limits.pair.exact_search.base,
                       saturated_multiply(demand_count, limits.pair.exact_search.per_input));
}

[[nodiscard]] size_t exact_pair_scan_work(const BranchOnlyRelaySearchLimits &limits,
                                          size_t demand_count, size_t relay_count) {
  return saturated_add(limits.pair.exact_scan.base,
                       saturated_multiply(saturated_multiply(demand_count, relay_count),
                                          limits.pair.exact_scan.per_input));
}

struct FarthestReachableRelayResult {
  std::optional<size_t> relay;
  bool exhausted = false;
};

template <typename Admissible>
[[nodiscard]] FarthestReachableRelayResult
farthest_reachable_relay(const FixedRelayDemand &demand, uint64_t cursor,
                         std::span<const uint64_t> relays, Admissible admissible,
                         BoundedPlanningWorkMeter &scan_work) {
  const auto inspect = [&](size_t relay) -> FarthestReachableRelayResult {
    if (!scan_work.consume())
      return {.relay = std::nullopt, .exhausted = true};
    return admissible(relay) && fixed_relay_is_between(demand, cursor, relays[relay]) &&
                   fixed_relay_can_hop(cursor, relays[relay])
               ? FarthestReachableRelayResult{.relay = relay}
               : FarthestReachableRelayResult{};
  };
  if (fixed_relay_demand_is_forward(demand)) {
    for (size_t relay = relays.size(); relay-- != 0u;) {
      const auto selected = inspect(relay);
      if (selected.relay || selected.exhausted)
        return selected;
    }
  } else {
    for (size_t relay = 0u; relay < relays.size(); ++relay) {
      const auto selected = inspect(relay);
      if (selected.relay || selected.exhausted)
        return selected;
    }
  }
  return {};
}

/// Exact disjoint-path solver for fixed SOPP source/target pairs.
///
/// The relay graph is a one-dimensional DAG: every hop moves monotonically
/// toward its target. The search assigns the most constrained remaining
/// demand first, prunes states whose independent shortest paths need more
/// relays than remain, and backtracks across both entry and return demands.
///
/// Feasibility retains the established maximum-progress enumeration order.
/// Capped optimization retains the shortest encountered route below the
/// supplied feasibility bound; exact-pair fallback uses that bound to preserve
/// capacity for later pairs.
///
/// Enumeration stops as soon as the target is directly reachable because an
/// additional relay cannot improve owner count and only consumes capacity.
/// `feasibility` stops at the first solution without scoring route length.
///
/// Relay offsets must be sorted and unique. Each solver instance is single-use.
/// Separate deterministic budgets count search states and alternatives, plus
/// every relay inspection performed by polynomial summaries.
class ExactFixedRelayBatchSolver {
public:
  enum class Termination : uint8_t {
    Solved,
    Infeasible,
    WorkBudgetExhausted,
  };

  struct SolveResult {
    Termination termination = Termination::Infeasible;
    /// Optimization may retain an improved route even when proving
    /// optimality reaches its independent work limit.
    bool solution_available = false;
  };

  ExactFixedRelayBatchSolver(std::span<const FixedRelayDemand> demands,
                             std::span<const uint64_t> relays,
                             BoundedPlanningWorkMeter &search_work,
                             BoundedPlanningWorkMeter &scan_work,
                             std::optional<size_t> incumbent_relay_count = std::nullopt)
      : demands_(demands), relays_(relays), best_relay_count_(incumbent_relay_count),
        maximum_relay_count_(incumbent_relay_count), available_(relays.size(), true),
        assigned_(demands.size(), false), routes_(demands.size()), search_work_(search_work),
        scan_work_(scan_work) {
    assert(std::ranges::is_sorted(relays_));
    assert(std::ranges::adjacent_find(relays_) == relays_.end());
  }

  [[nodiscard]] SolveResult solve(std::vector<std::vector<uint64_t>> &routes_out) {
    routes_out.clear();
    (void)solve_remaining(0u);
    if (found_better_solution_) {
      routes_out.resize(best_routes_.size());
      for (size_t demand = 0u; demand < best_routes_.size(); ++demand) {
        routes_out[demand].reserve(best_routes_[demand].size());
        for (size_t relay : best_routes_[demand])
          routes_out[demand].push_back(relays_[relay]);
      }
    }
    if (is_feasibility() && found_better_solution_)
      return {Termination::Solved, true};
    if (work_budget_exhausted())
      return {Termination::WorkBudgetExhausted, found_better_solution_};
    return {found_better_solution_ ? Termination::Solved : Termination::Infeasible,
            found_better_solution_};
  }

private:
  [[nodiscard]] bool is_feasibility() const { return !maximum_relay_count_; }

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

  [[nodiscard]] std::optional<size_t> minimum_relay_count(const FixedRelayDemand &demand,
                                                          uint64_t cursor) {
    size_t count = 0u;
    while (!fixed_relay_can_hop(cursor, demand.target)) {
      const auto best = farthest_reachable_relay(
          demand, cursor, relays_, [&](size_t relay) { return available_[relay]; }, scan_work_);
      if (best.exhausted || !best.relay)
        return std::nullopt;
      cursor = relays_[*best.relay];
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
      if (!available_[relay] || !fixed_relay_is_between(demand, demand.source, relays_[relay]))
        continue;
      ++summary.corridor_relays;
      if (fixed_relay_can_hop(demand.source, relays_[relay]))
        ++summary.first_hop_options;
    }
    return summary;
  }

  template <typename Callback>
  [[nodiscard]] bool enumerate_routes(size_t demand_index, uint64_t cursor,
                                      std::vector<size_t> &route, Callback &callback) {
    const FixedRelayDemand &demand = demands_[demand_index];
    if (fixed_relay_can_hop(cursor, demand.target))
      return callback(route);

    const auto try_relay = [&](size_t relay) {
      if (!minimum_relay_count(demand, relays_[relay]))
        return false;
      route.push_back(relay);
      const bool stop = enumerate_routes(demand_index, relays_[relay], route, callback);
      route.pop_back();
      return stop;
    };

    const auto consider = [&](size_t relay) {
      return available_[relay] && fixed_relay_is_between(demand, cursor, relays_[relay]) &&
             fixed_relay_can_hop(cursor, relays_[relay]) && try_relay(relay);
    };
    if (fixed_relay_demand_is_forward(demand)) {
      for (size_t relay = relays_.size(); relay-- != 0u;) {
        if (work_budget_exhausted() || !consume_search_work())
          return false;
        if (consider(relay))
          return true;
      }
    } else {
      for (size_t relay = 0u; relay < relays_.size(); ++relay) {
        if (work_budget_exhausted() || !consume_search_work())
          return false;
        if (consider(relay))
          return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool record_solution() {
    if (is_feasibility()) {
      best_routes_ = routes_;
      found_better_solution_ = true;
      return true;
    }
    assert(best_relay_count_ && maximum_relay_count_);
    const size_t relay_count = std::accumulate(
        routes_.begin(), routes_.end(), size_t{0u},
        [](size_t count, const auto &route) { return saturated_add(count, route.size()); });
    if (relay_count > *maximum_relay_count_ || relay_count >= *best_relay_count_)
      return false;
    best_relay_count_ = relay_count;
    best_routes_ = routes_;
    found_better_solution_ = true;
    return relay_count == 0u;
  }

  [[nodiscard]] bool solve_remaining(size_t assigned_count) {
    if (!consume_search_work())
      return false;
    if (assigned_count == demands_.size())
      return record_solution();

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
      required_relays = saturated_add(required_relays, summary->minimum_relay_count);
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
    if (maximum_relay_count_) {
      const size_t assigned_relay_count = std::accumulate(
          routes_.begin(), routes_.end(), size_t{0u},
          [](size_t count, const auto &route) { return saturated_add(count, route.size()); });
      if (saturated_add(assigned_relay_count, required_relays) > *maximum_relay_count_)
        return false;
    }

    assigned_[*selected] = true;
    std::vector<size_t> candidate_route;
    const auto accept_route = [&](const std::vector<size_t> &route) {
      for (size_t relay : route)
        available_[relay] = false;
      routes_[*selected] = route;
      const bool stop = solve_remaining(assigned_count + 1u);
      routes_[*selected].clear();
      for (size_t relay : route)
        available_[relay] = true;
      return stop;
    };
    const bool stop =
        enumerate_routes(*selected, demands_[*selected].source, candidate_route, accept_route);
    assigned_[*selected] = false;
    return stop;
  }

  std::span<const FixedRelayDemand> demands_;
  std::span<const uint64_t> relays_;
  std::optional<size_t> best_relay_count_;
  std::optional<size_t> maximum_relay_count_;
  bool found_better_solution_ = false;
  std::vector<bool> available_;
  std::vector<bool> assigned_;
  std::vector<std::vector<size_t>> routes_;
  std::vector<std::vector<size_t>> best_routes_;
  BoundedPlanningWorkMeter &search_work_;
  BoundedPlanningWorkMeter &scan_work_;
};

struct ExactRelaySolveResult {
  ExactFixedRelayBatchSolver::Termination termination =
      ExactFixedRelayBatchSolver::Termination::Infeasible;
  bool refinement_exhausted = false;
  std::vector<uint64_t> feasibility_relay_offsets;
};

[[nodiscard]] ExactRelaySolveResult
solve_exact_routes(std::span<const FixedRelayDemand> demands, std::span<const uint64_t> relays,
                   BoundedPlanningWorkMeter &search_work, BoundedPlanningWorkMeter &scan_work,
                   bool minimize_relay_count, bool constrain_optimized_relays_to_baseline,
                   std::vector<std::vector<uint64_t>> &routes_out) {
  ExactRelaySolveResult result;
  routes_out.clear();
  ExactFixedRelayBatchSolver feasibility_solver(demands, relays, search_work, scan_work);
  const ExactFixedRelayBatchSolver::SolveResult feasibility = feasibility_solver.solve(routes_out);
  result.termination = feasibility.termination;
  if (feasibility.termination != ExactFixedRelayBatchSolver::Termination::Solved)
    return result;
  assert(feasibility.solution_available);

  const size_t baseline_relay_count = std::accumulate(
      routes_out.begin(), routes_out.end(), size_t{0u},
      [](size_t count, const auto &route) { return saturated_add(count, route.size()); });
  if (constrain_optimized_relays_to_baseline) {
    result.feasibility_relay_offsets.reserve(baseline_relay_count);
    for (const auto &route : routes_out) {
      result.feasibility_relay_offsets.insert(result.feasibility_relay_offsets.end(), route.begin(),
                                              route.end());
    }
    std::ranges::sort(result.feasibility_relay_offsets);
    result.feasibility_relay_offsets.erase(
        std::ranges::unique(result.feasibility_relay_offsets).begin(),
        result.feasibility_relay_offsets.end());
    assert(result.feasibility_relay_offsets.size() == baseline_relay_count);
  }

  if (!minimize_relay_count)
    return result;

  std::span<const uint64_t> optimization_relays = relays;
  if (constrain_optimized_relays_to_baseline)
    optimization_relays = result.feasibility_relay_offsets;

  std::vector<std::vector<uint64_t>> optimized_routes;
  ExactFixedRelayBatchSolver optimization_solver(demands, optimization_relays, search_work,
                                                 scan_work, baseline_relay_count);
  const ExactFixedRelayBatchSolver::SolveResult optimization =
      optimization_solver.solve(optimized_routes);
  result.refinement_exhausted =
      optimization.termination == ExactFixedRelayBatchSolver::Termination::WorkBudgetExhausted;
  if (optimization.solution_available)
    routes_out = std::move(optimized_routes);
  return result;
}

using FixedRelayInventory = std::set<uint64_t>;

struct GreedyFixedRelayRoute {
  enum class Status : uint8_t {
    Solved,
    Infeasible,
    WorkBudgetExhausted,
    InvariantFailure,
  };

  Status status = Status::Infeasible;
  std::vector<uint64_t> offsets;
  /// Exact inventory nodes removed by a successful route. This preserves
  /// owner affinity for commit or transactional rollback without consulting
  /// a second source of truth.
  FixedRelayInventory claimed_relays;
};

/// Exact for one validated monotonic demand. On failure the relay set is
/// restored, so callers may use it transactionally without copying the whole
/// inventory. Each selected relay precharges its ordered-set query, removal,
/// and possible rollback insertion by logarithmic comparison depth.
[[nodiscard]] GreedyFixedRelayRoute
plan_greedy_fixed_relay_route(const FixedRelayDemand &demand, FixedRelayInventory &unused_relays,
                              BoundedPlanningWorkMeter &work) {
  assert(demand.source % sizeof(uint32_t) == 0u);
  assert(demand.target % sizeof(uint32_t) == 0u);
  assert(demand.source != demand.target);
  const bool forward = fixed_relay_demand_is_forward(demand);
  uint64_t cursor = demand.source;
  std::vector<uint64_t> route;
  FixedRelayInventory claimed_relays;
  const auto rollback = [&](GreedyFixedRelayRoute::Status status) {
    unused_relays.merge(claimed_relays);
    if (!claimed_relays.empty()) {
      return GreedyFixedRelayRoute{
          .status = GreedyFixedRelayRoute::Status::InvariantFailure,
          .offsets = {},
          .claimed_relays = std::move(claimed_relays),
      };
    }
    return GreedyFixedRelayRoute{
        .status = status,
        .offsets = {},
        .claimed_relays = {},
    };
  };
  while (!fixed_relay_can_hop(cursor, demand.target)) {
    const size_t query_work = std::max<size_t>(std::bit_width(unused_relays.size()), 1u);
    if (!work.consume(query_work))
      return rollback(GreedyFixedRelayRoute::Status::WorkBudgetExhausted);
    FixedRelayInventory::iterator relay = unused_relays.end();
    if (forward) {
      const uint64_t limit =
          cursor > std::numeric_limits<uint64_t>::max() - kSoppBranchMaximumForwardReachBytes
              ? std::numeric_limits<uint64_t>::max()
              : cursor + kSoppBranchMaximumForwardReachBytes;
      const auto reachable_end =
          unused_relays.upper_bound(std::min(limit, demand.target - sizeof(uint32_t)));
      if (reachable_end != unused_relays.begin())
        relay = std::prev(reachable_end);
      if (relay == unused_relays.end() || !fixed_relay_is_between(demand, cursor, *relay) ||
          !fixed_relay_can_hop(cursor, *relay)) {
        relay = unused_relays.end();
      }
    } else {
      const uint64_t limit = cursor > kSoppBranchMaximumBackwardReachBytes
                                 ? cursor - kSoppBranchMaximumBackwardReachBytes
                                 : 0u;
      relay = unused_relays.lower_bound(std::max(limit, demand.target + sizeof(uint32_t)));
      if (relay == unused_relays.end() || !fixed_relay_is_between(demand, cursor, *relay) ||
          !fixed_relay_can_hop(cursor, *relay)) {
        relay = unused_relays.end();
      }
    }
    if (relay == unused_relays.end())
      return rollback(GreedyFixedRelayRoute::Status::Infeasible);
    if (!work.consume(saturated_multiply(2u, query_work)))
      return rollback(GreedyFixedRelayRoute::Status::WorkBudgetExhausted);
    auto claimed = unused_relays.extract(relay);
    assert(!claimed.empty());
    if (claimed.empty())
      return rollback(GreedyFixedRelayRoute::Status::Infeasible);
    cursor = claimed.value();
    route.push_back(cursor);
    claimed_relays.insert(std::move(claimed));
  }
  return {
      .status = GreedyFixedRelayRoute::Status::Solved,
      .offsets = std::move(route),
      .claimed_relays = std::move(claimed_relays),
  };
}

/// Borrowed prefix produced by the ordered relay-qualification pass. Offsets
/// are sorted and unique; `complete` distinguishes the full inventory from a
/// sound prefix retained after bounded qualification.
struct QualifiedRelayInventoryView {
  std::span<const uint64_t> offsets;
  bool complete = false;
};

/// Recovers pair-atomic routes after the full-batch exact solve fails. This
/// helper owns the fallback inventory, owner affinities materialized by
/// earlier pairs, and the policy that preserves a nonfinal pair's complete
/// feasibility baseline for later requests.
[[nodiscard]] std::optional<std::string> plan_exact_pair_fallbacks(
    std::span<const BranchOnlyRelayPairRequest> requests, const std::vector<bool> &valid_request,
    QualifiedRelayInventoryView qualified_relays, const BranchOnlyRelaySearchLimits &limits,
    BranchOnlyRelayBatchPlan &batch) {
  batch.strategy = BranchOnlyRelayPlanStrategy::ExactPairFallback;
  const bool input_shape_valid = valid_request.size() == requests.size();
  assert(input_shape_valid);
  if (!input_shape_valid) {
    batch.routing_invariant_failed = true;
    batch.failure = BranchOnlyRelayPlanFailure::Reservation;
    return "branch-only router received inconsistent fallback inventory";
  }

  const std::span<const uint64_t> relay_offsets = qualified_relays.offsets;

  BoundedPlanningWorkMeter fallback_setup_work(limits.batch.fallback_setup);
  const size_t fallback_setup_cost = saturated_multiply(
      relay_offsets.size(), std::max<size_t>(std::bit_width(relay_offsets.size()), 1u));
  const bool fallback_inventory_available = fallback_setup_work.consume(fallback_setup_cost);
  accumulate_saturated(batch.fallback_setup_work_consumed, fallback_setup_work.consumed());
  FixedRelayInventory unused_relays;
  if (fallback_inventory_available)
    unused_relays.insert(relay_offsets.begin(), relay_offsets.end());

  const auto restore_unused_relays = [&](GreedyFixedRelayRoute &route) {
    unused_relays.merge(route.claimed_relays);
    if (!route.claimed_relays.empty()) {
      batch.routing_invariant_failed = true;
      return false;
    }
    return true;
  };

  std::vector<bool> valid_request_at_or_after(requests.size() + 1u, false);
  for (size_t request_index = requests.size(); request_index-- != 0u;) {
    valid_request_at_or_after[request_index] =
        valid_request[request_index] || valid_request_at_or_after[request_index + 1u];
  }

  for (size_t request_index = 0u; request_index < requests.size(); ++request_index) {
    if (!valid_request[request_index])
      continue;
    batch.pair_strategies[request_index] = BranchOnlyRelayPlanStrategy::ExactPairFallback;
    if (!fallback_inventory_available) {
      batch.routing_work_exhausted = true;
      batch.rejected_pair_indices.push_back(request_index);
      batch.rejection_reasons[request_index] = BranchOnlyRelayPairRejection::WorkBudget;
      continue;
    }

    const BranchOnlyRelayPairRequest &request = requests[request_index];
    std::vector<FixedRelayDemand> pair_demands;
    pair_demands.reserve(request.entry_preplaced ? 1u : 2u);
    if (!request.entry_preplaced) {
      pair_demands.push_back(
          FixedRelayDemand{request_index, true, request.entry_source, request.entry_target});
    }
    pair_demands.push_back(
        FixedRelayDemand{request_index, false, request.return_source, request.return_target});
    const bool has_later_valid_request = valid_request_at_or_after[request_index + 1u];

    BoundedPlanningWorkMeter pair_search_work(exact_pair_search_work(limits, pair_demands.size()));
    BoundedPlanningWorkMeter pair_scan_work(
        exact_pair_scan_work(limits, pair_demands.size(), unused_relays.size()));
    std::vector<std::vector<uint64_t>> pair_routes;
    ExactFixedRelayBatchSolver::Termination pair_termination =
        ExactFixedRelayBatchSolver::Termination::WorkBudgetExhausted;
    bool pair_refinement_exhausted = false;
    if (pair_scan_work.consume(unused_relays.size())) {
      std::vector<uint64_t> available_relays(unused_relays.begin(), unused_relays.end());
      const ExactRelaySolveResult solve = solve_exact_routes(
          pair_demands, available_relays, pair_search_work, pair_scan_work,
          /*minimize_relay_count=*/true,
          /*constrain_optimized_relays_to_baseline=*/has_later_valid_request, pair_routes);
      pair_termination = solve.termination;
      pair_refinement_exhausted = solve.refinement_exhausted;
      if (pair_termination == ExactFixedRelayBatchSolver::Termination::Solved) {
        const size_t reserved_relay_count =
            has_later_valid_request
                ? solve.feasibility_relay_offsets.size()
                : std::accumulate(pair_routes.begin(), pair_routes.end(), size_t{0u},
                                  [](size_t count, const auto &route) {
                                    return saturated_add(count, route.size());
                                  });
        const size_t removal_work = saturated_multiply(
            reserved_relay_count, std::max<size_t>(std::bit_width(unused_relays.size()), 1u));
        if (!pair_scan_work.consume(removal_work)) {
          pair_termination = ExactFixedRelayBatchSolver::Termination::WorkBudgetExhausted;
          pair_routes.clear();
        } else if (has_later_valid_request) {
          // Keep the exact feasibility baseline unavailable until the batch
          // ends. The optimized route is a subset, so later pairs see the
          // same inventory as the feasibility-only plan.
          for (uint64_t relay : solve.feasibility_relay_offsets)
            unused_relays.erase(relay);
        }
      }
    }
    if (pair_termination == ExactFixedRelayBatchSolver::Termination::Solved)
      batch.routing_work_exhausted = batch.routing_work_exhausted || pair_refinement_exhausted;
    accumulate_saturated(batch.search_work_consumed, pair_search_work.consumed());
    accumulate_saturated(batch.feasibility_scan_work_consumed, pair_scan_work.consumed());

    if (pair_termination == ExactFixedRelayBatchSolver::Termination::Solved) {
      assert(pair_routes.size() == pair_demands.size());
      if (pair_routes.size() != pair_demands.size()) {
        batch.routing_invariant_failed = true;
        batch.failure = BranchOnlyRelayPlanFailure::Reservation;
        return "branch-only router received inconsistent exact fallback routes";
      }
      for (size_t demand_index = 0u; demand_index < pair_demands.size(); ++demand_index) {
        std::vector<uint64_t> &route = pair_demands[demand_index].entry
                                           ? batch.routes[request_index].entry_relay_offsets
                                           : batch.routes[request_index].return_relay_offsets;
        route = std::move(pair_routes[demand_index]);
      }
      if (!has_later_valid_request) {
        for (uint64_t relay : batch.routes[request_index].entry_relay_offsets)
          unused_relays.erase(relay);
        for (uint64_t relay : batch.routes[request_index].return_relay_offsets)
          unused_relays.erase(relay);
      }
      continue;
    }

    if (pair_termination == ExactFixedRelayBatchSolver::Termination::WorkBudgetExhausted) {
      batch.strategy = BranchOnlyRelayPlanStrategy::GreedyPairFallback;
      batch.pair_strategies[request_index] = BranchOnlyRelayPlanStrategy::GreedyPairFallback;
      batch.routing_work_exhausted |=
          pair_termination == ExactFixedRelayBatchSolver::Termination::WorkBudgetExhausted;
      BoundedPlanningWorkMeter greedy_work(limits.pair.greedy);
      GreedyFixedRelayRoute entry_route;
      if (request.entry_preplaced) {
        entry_route.status = GreedyFixedRelayRoute::Status::Solved;
      } else {
        entry_route =
            plan_greedy_fixed_relay_route(pair_demands.front(), unused_relays, greedy_work);
      }
      GreedyFixedRelayRoute return_route;
      if (entry_route.status == GreedyFixedRelayRoute::Status::Solved) {
        return_route =
            plan_greedy_fixed_relay_route(pair_demands.back(), unused_relays, greedy_work);
      }
      accumulate_saturated(batch.feasibility_scan_work_consumed, greedy_work.consumed());
      if (entry_route.status == GreedyFixedRelayRoute::Status::InvariantFailure ||
          return_route.status == GreedyFixedRelayRoute::Status::InvariantFailure) {
        batch.routing_invariant_failed = true;
        batch.failure = BranchOnlyRelayPlanFailure::Reservation;
        return "branch-only router could not restore its greedy relay inventory";
      }
      if (entry_route.status == GreedyFixedRelayRoute::Status::Solved &&
          return_route.status == GreedyFixedRelayRoute::Status::Solved) {
        batch.routes[request_index].entry_relay_offsets = std::move(entry_route.offsets);
        batch.routes[request_index].return_relay_offsets = std::move(return_route.offsets);
        continue;
      }
      if (!restore_unused_relays(entry_route)) {
        batch.failure = BranchOnlyRelayPlanFailure::Reservation;
        return "branch-only router could not restore its greedy relay inventory";
      }
      if (entry_route.status == GreedyFixedRelayRoute::Status::WorkBudgetExhausted ||
          return_route.status == GreedyFixedRelayRoute::Status::WorkBudgetExhausted) {
        batch.routing_work_exhausted = true;
        batch.rejected_pair_indices.push_back(request_index);
        batch.rejection_reasons[request_index] = BranchOnlyRelayPairRejection::WorkBudget;
        continue;
      }
    }

    batch.rejected_pair_indices.push_back(request_index);
    if (!qualified_relays.complete) {
      batch.rejection_reasons[request_index] = BranchOnlyRelayPairRejection::WorkBudget;
      continue;
    }

    // A single monotonic demand is feasible exactly when farthest-progress
    // greedy routing succeeds. Probe each half independently so shared relay
    // contention is not misreported as an unreachable return corridor. The
    // probes restore successful routes and share one bounded meter.
    BoundedPlanningWorkMeter classification_work(limits.pair.greedy);
    const auto probe = [&](const FixedRelayDemand &demand) {
      GreedyFixedRelayRoute result =
          plan_greedy_fixed_relay_route(demand, unused_relays, classification_work);
      if (!restore_unused_relays(result))
        result.status = GreedyFixedRelayRoute::Status::InvariantFailure;
      return result.status;
    };
    const GreedyFixedRelayRoute::Status entry_status = request.entry_preplaced
                                                           ? GreedyFixedRelayRoute::Status::Solved
                                                           : probe(pair_demands.front());
    const GreedyFixedRelayRoute::Status return_status =
        entry_status == GreedyFixedRelayRoute::Status::WorkBudgetExhausted ||
                entry_status == GreedyFixedRelayRoute::Status::InvariantFailure
            ? entry_status
            : probe(pair_demands.back());
    accumulate_saturated(batch.feasibility_scan_work_consumed, classification_work.consumed());
    if (entry_status == GreedyFixedRelayRoute::Status::InvariantFailure ||
        return_status == GreedyFixedRelayRoute::Status::InvariantFailure) {
      batch.failure = BranchOnlyRelayPlanFailure::Reservation;
      return "branch-only router could not restore its relay-classification inventory";
    }
    if (entry_status == GreedyFixedRelayRoute::Status::WorkBudgetExhausted ||
        return_status == GreedyFixedRelayRoute::Status::WorkBudgetExhausted) {
      batch.routing_work_exhausted = true;
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

  return std::nullopt;
}

} // namespace

size_t branch_only_relay_conservative_work_limit(const BranchOnlyRelaySearchLimits &limits,
                                                 size_t pair_count, size_t relay_count,
                                                 size_t occupied_range_count) {
  const auto normalized = [](size_t value) { return std::max<size_t>(value, 1u); };
  const size_t demand_count = saturated_multiply(pair_count, 2u);
  size_t total =
      normalized(relay_qualification_work_limit(limits, relay_count, occupied_range_count));
  accumulate_saturated(total, normalized(exact_batch_search_work(limits, demand_count)));
  accumulate_saturated(total, normalized(exact_batch_scan_work(limits, demand_count, relay_count)));
  accumulate_saturated(total, normalized(limits.batch.fallback_setup));

  size_t per_pair = normalized(exact_pair_search_work(limits, 2u));
  accumulate_saturated(per_pair, normalized(exact_pair_scan_work(limits, 2u, relay_count)));
  // A failed greedy attempt may be followed by an independently bounded
  // rejection-classification pass.
  accumulate_saturated(per_pair, saturated_multiply(normalized(limits.pair.greedy), 2u));
  accumulate_saturated(total, saturated_multiply(pair_count, per_pair));
  return total;
}

BranchOnlyRelaySearchLimits branch_only_relay_greedy_pair_limits(size_t relay_count) {
  BranchOnlyRelaySearchLimits limits;
  limits.batch.feasibility_search = {1u, 0u};
  limits.batch.feasibility_scan = {1u, 0u};
  limits.pair.exact_search = {1u, 0u};
  limits.pair.exact_scan = {1u, 0u};
  const size_t relay_index_levels = std::max<size_t>(std::bit_width(relay_count), 1u);
  limits.batch.fallback_setup = saturated_multiply(relay_count, relay_index_levels);
  // A pair can claim the complete inventory across its entry and return
  // routes. Each claim precharges an ordered-map query, removal, and possible
  // rollback insertion at the current index depth, so a relay-count-only
  // allowance can reject a long sparse corridor before completing it.
  limits.pair.greedy = std::max<size_t>(
      saturated_multiply(3u, saturated_multiply(relay_count, relay_index_levels)), 1u);
  return limits;
}

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
  return relays_.emplace(offset, RelayOffer{provenance}).second;
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
    *outcome_out = batch.plan_outcome();
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
  const size_t conservative_work_limit = branch_only_relay_conservative_work_limit(
      limits, requests.size(), relays_.size(), tentative_planner.occupied_ranges().size());
  const auto finish = [&]() -> BranchOnlyRelayBatchPlan {
    const bool work_bound_respected = batch.total_work_consumed() <= conservative_work_limit;
    batch.routing_invariant_failed |= !work_bound_respected;
    assert(work_bound_respected && "branch-only router exceeded its configured work bound");
    return std::move(batch);
  };
  if (requests.empty())
    return finish();

  std::vector<uint64_t> relay_offsets;
  relay_offsets.reserve(relays_.size());

  std::vector<bool> invalid_entry(requests.size(), false);
  std::vector<bool> invalid_return(requests.size(), false);
  for (size_t request_index = 0u; request_index < requests.size(); ++request_index) {
    const BranchOnlyRelayPairRequest &request = requests[request_index];
    const std::array entry_endpoints = {request.entry_source, request.entry_target};
    const std::array return_endpoints = {request.return_source, request.return_target};
    invalid_entry[request_index] =
        !request.entry_preplaced && (request.entry_target <= request.entry_source ||
                                     std::ranges::any_of(entry_endpoints, [](uint64_t offset) {
                                       return offset % sizeof(uint32_t) != 0u;
                                     }));
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
    if (!request.entry_preplaced) {
      for (uint64_t offset : {request.entry_source, request.entry_target}) {
        const auto owner = entry_coordinate_owner.find(offset);
        if (owner != entry_coordinate_owner.end() && owner->second == request_index)
          entry_coordinate_owner.erase(owner);
      }
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
    if (!request.entry_preplaced &&
        !register_coordinates(entry_endpoints, entry_coordinate_owner, invalid_entry))
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
    if (!request.entry_preplaced) {
      pair_coordinates.insert(request.entry_source);
      pair_coordinates.insert(request.entry_target);
    }
    for (uint64_t offset : {request.return_source, request.return_target})
      pair_coordinates.insert(offset);
  }

  std::vector<FixedRelayDemand> demands;
  demands.reserve(2u * requests.size());
  for (size_t request_index = 0u; request_index < requests.size(); ++request_index) {
    if (!valid_request[request_index] || requests[request_index].entry_preplaced)
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

  // Offered storage that aliases a branch source or destination is not relay
  // capacity. Likewise, a pristine word that is already reserved or outside
  // the original image cannot become a claim. Offered offsets are unique
  // dwords, so independent read-only qualification proves that any selected
  // subset can be reserved together without the quadratic planner-copy pass.
  // A bounded pass retains its proven prefix; omitted suffix entries can only
  // reduce routing capacity, never invalidate a selected route.
  const size_t occupied_range_count = tentative_planner.occupied_ranges().size();
  const size_t occupancy_query_work = std::max<size_t>(std::bit_width(occupied_range_count), 1u);
  BoundedPlanningWorkMeter qualification_work(
      relay_qualification_work_limit(limits, relays_.size(), occupied_range_count));
  bool relay_inventory_complete = true;
  if (!demands.empty()) {
    const size_t lookup_work = relay_qualification_lookup_work(pair_coordinates.size());
    for (const auto &[offset, relay] : relays_) {
      if (!qualification_work.consume(lookup_work)) {
        relay_inventory_complete = false;
        break;
      }
      if (pair_coordinates.contains(offset))
        continue;
      if (relay.provenance == BranchOnlyRelayProvenance::PristineNop) {
        if (!qualification_work.consume(occupancy_query_work)) {
          relay_inventory_complete = false;
          break;
        }
        if (!tentative_planner.can_reserve_existing_range(offset, sizeof(uint32_t))) {
          accumulate_saturated(batch.pristine_relay_occupancy_rejection_count, 1u);
          continue;
        }
      }
      relay_offsets.push_back(offset);
    }
  }
  const QualifiedRelayInventoryView qualified_relays{
      .offsets = relay_offsets,
      .complete = relay_inventory_complete,
  };
  accumulate_saturated(batch.relay_qualification_work_consumed, qualification_work.consumed());
  batch.relay_qualification_exhausted |= !qualified_relays.complete;

  ExactFixedRelayBatchSolver::Termination exact_termination =
      ExactFixedRelayBatchSolver::Termination::Infeasible;
  std::vector<std::vector<uint64_t>> solved_routes;
  if (!demands.empty()) {
    BoundedPlanningWorkMeter search_work(exact_batch_search_work(limits, demands.size()));
    BoundedPlanningWorkMeter scan_work(
        exact_batch_scan_work(limits, demands.size(), qualified_relays.offsets.size()));
    if (!scan_work.consume(qualified_relays.offsets.size())) {
      exact_termination = ExactFixedRelayBatchSolver::Termination::WorkBudgetExhausted;
    } else {
      const ExactRelaySolveResult solve =
          solve_exact_routes(demands, qualified_relays.offsets, search_work, scan_work,
                             /*minimize_relay_count=*/false,
                             /*constrain_optimized_relays_to_baseline=*/false, solved_routes);
      exact_termination = solve.termination;
    }
    accumulate_saturated(batch.search_work_consumed, search_work.consumed());
    accumulate_saturated(batch.feasibility_scan_work_consumed, scan_work.consumed());
  }
  if (exact_termination == ExactFixedRelayBatchSolver::Termination::Solved) {
    for (size_t demand_index = 0u; demand_index < demands.size(); ++demand_index) {
      const FixedRelayDemand &demand = demands[demand_index];
      std::vector<uint64_t> &route = demand.entry
                                         ? batch.routes[demand.pair_index].entry_relay_offsets
                                         : batch.routes[demand.pair_index].return_relay_offsets;
      route = std::move(solved_routes[demand_index]);
    }
  } else if (!demands.empty()) {
    batch.routing_work_exhausted |=
        exact_termination == ExactFixedRelayBatchSolver::Termination::WorkBudgetExhausted;
    if (std::optional<std::string> fallback_error =
            plan_exact_pair_fallbacks(requests, valid_request, qualified_relays, limits, batch)) {
      report(error_out, std::move(*fallback_error));
      return finish();
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
      if (index >= rejection_counts.size())
        continue;
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
    if (batch.work_budget_exhausted())
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
        retired.push_back({offset, relay->second.provenance});
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
        return finish();
      }
      route.claims.push_back({offset, relay->second.provenance});
    }
  }

  if (!batch.rejected_pair_indices.empty())
    return finish();

  DbiPatchPlacementPlanner reserved_planner = tentative_planner;
  for (const BranchOnlyRelayRoute &route : batch.routes) {
    for (const BranchOnlyRelayClaim &claim : route.claims) {
      if (claim.provenance == BranchOnlyRelayProvenance::PristineNop &&
          !reserved_planner.reserve_existing_range(claim.offset, sizeof(uint32_t), error_out)) {
        batch.failure = BranchOnlyRelayPlanFailure::Reservation;
        return finish();
      }
    }
  }
  tentative_planner = std::move(reserved_planner);
  return finish();
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
      const RelayOffer expected{claim.provenance};
      if (relay == relays_.end() || relay->second != expected ||
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
      const RelayOffer expected{retired.provenance};
      if (relay == relays_.end() || relay->second != expected ||
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

bool BranchOnlyDirectRelayReservoirSet::mark_relays_used(std::span<const uint64_t> relays,
                                                         std::string *error_out) {
  std::vector<size_t> pending;
  const auto append_relays = [&](std::span<const uint64_t> source_relays) {
    for (uint64_t relay : source_relays) {
      const auto reservoir = reservoir_by_relay.find(relay);
      if (reservoir == reservoir_by_relay.end() || reservoir->second >= reservoirs.size())
        return false;
      pending.push_back(reservoir->second);
    }
    return true;
  };
  if (!append_relays(relays)) {
    report(error_out, "branch-only router lost a claimed direct reservoir");
    return false;
  }

  // A routed reservoir may itself consume tails from later-text reservoirs.
  // Materialization therefore follows the ownership graph transitively. The
  // graph is acyclic by construction (each reservoir is planned only from the
  // previously committed frontier), but the visited set also fails safely if
  // malformed external state repeats an owner.
  std::vector<bool> selected(reservoirs.size(), false);
  for (size_t cursor = 0u; cursor < pending.size(); ++cursor) {
    const size_t index = pending[cursor];
    if (selected[index])
      continue;
    selected[index] = true;
    const BranchOnlyDirectRelayReservoir &reservoir = reservoirs[index];
    if (reservoir.route) {
      std::vector<uint64_t> dependencies;
      for (const BranchOnlyRelayClaim &claim : reservoir.route->claims)
        if (reservoir_by_relay.contains(claim.offset))
          dependencies.push_back(claim.offset);
      if (!append_relays(dependencies)) {
        report(error_out, "branch-only router lost a routed-reservoir dependency");
        return false;
      }
    }
  }
  for (size_t index = 0u; index < selected.size(); ++index)
    reservoirs[index].used = reservoirs[index].used || selected[index];
  return true;
}

bool BranchOnlyDirectRelayReservoirSet::mark_claims_used(
    std::span<const BranchOnlyRelayClaim> claims, std::string *error_out) {
  std::vector<uint64_t> relays;
  for (const BranchOnlyRelayClaim &claim : claims) {
    if (reservoir_by_relay.contains(claim.offset))
      relays.push_back(claim.offset);
  }
  return mark_relays_used(relays, error_out);
}

bool BranchOnlyRelayRouter::plan_direct_reservoirs(
    std::span<BasicBlock *const> blocks, std::span<const uint8_t> pristine_text,
    std::span<const std::pair<uint64_t, uint64_t>> protected_ranges, rj_code_arch_t arch,
    uint64_t route_frontier_source, size_t target_relay_count,
    DbiPatchPlacementPlanner &placement_planner, BranchOnlyDirectRelayReservoirSet &reservoirs,
    std::string *error_out, const BranchOnlyDirectReservoirWorkLimits &work_limits,
    PlanningWorkMeasurement *work_measurement) {
  if (target_relay_count == 0u)
    return true;
  if (pristine_text.empty()) {
    report(error_out, "branch-only router cannot discover reservoirs without pristine text");
    return false;
  }
  if (work_limits.minimum_words < 2u || work_limits.minimum_words > work_limits.maximum_words) {
    report(error_out, "branch-only router received invalid direct-reservoir word bounds");
    return false;
  }

  const size_t text_words =
      saturated_add(pristine_text.size(), sizeof(uint32_t) - 1u) / sizeof(uint32_t);
  const size_t raw_input_count = saturated_add(
      saturated_add(saturated_add(text_words, protected_ranges.size()), blocks.size()),
      relays_.size());
  const size_t complexity_units =
      saturated_multiply(raw_input_count, std::max<size_t>(std::bit_width(raw_input_count), 1u));
  MeteredPlanningWork discovery_work(
      work_limits.discovery.for_inputs(complexity_units),
      work_measurement == nullptr ? nullptr : &work_measurement->work_count,
      work_measurement == nullptr ? nullptr : &work_measurement->exhaustion_count);
  const auto charge_discovery = [&](size_t amount = 1u) { return discovery_work.consume(amount); };
  const auto report_exhaustion = [&]() {
    report(error_out, "branch-only direct-reservoir discovery exhausted its work allowance");
  };

  std::vector<std::pair<uint64_t, uint64_t>> merged_ranges;
  merged_ranges.reserve(protected_ranges.size());
  for (const auto &range : protected_ranges) {
    if (!charge_discovery()) {
      report_exhaustion();
      return false;
    }
    merged_ranges.push_back(range);
  }
  if (!charge_discovery(merged_ranges.size())) {
    report_exhaustion();
    return false;
  }
  std::erase_if(merged_ranges, [](const auto &range) { return range.first >= range.second; });
  if (!charge_discovery(saturated_multiply(
          merged_ranges.size(), std::max<size_t>(std::bit_width(merged_ranges.size()), 1u)))) {
    report_exhaustion();
    return false;
  }
  std::ranges::sort(merged_ranges);
  size_t merged_count = 0u;
  for (const auto &range : merged_ranges) {
    if (!charge_discovery()) {
      report_exhaustion();
      return false;
    }
    if (merged_count != 0u && merged_ranges[merged_count - 1u].second >= range.first) {
      merged_ranges[merged_count - 1u].second =
          std::max(merged_ranges[merged_count - 1u].second, range.second);
    } else {
      merged_ranges[merged_count++] = range;
    }
  }
  merged_ranges.resize(merged_count);
  const auto overlaps_protected = [&](uint64_t begin, uint64_t end) -> std::optional<bool> {
    if (!charge_discovery(std::max<size_t>(std::bit_width(merged_ranges.size()), 1u)))
      return std::nullopt;
    const auto after = std::ranges::lower_bound(merged_ranges, end, {},
                                                [](const auto &range) { return range.first; });
    return after != merged_ranges.begin() && std::prev(after)->second > begin;
  };

  struct Candidate {
    uint64_t offset = 0;
    std::vector<uint32_t> words;
  };
  std::vector<Candidate> candidates;
  bool discovery_exhausted = false;
  for (BasicBlock *block : blocks) {
    if (!charge_discovery()) {
      discovery_exhausted = true;
      break;
    }
    if (block == nullptr)
      continue;
    std::vector<const Instruction *> run;
    const auto flush_run = [&]() -> bool {
      size_t run_end = run.size();
      while (run_end != 0u) {
        size_t run_begin = run_end;
        size_t word_count = 0u;
        while (run_begin != 0u) {
          if (!charge_discovery())
            return false;
          const size_t instruction_words =
              static_cast<size_t>(run[run_begin - 1u]->size()) / sizeof(uint32_t);
          if (word_count + instruction_words > work_limits.maximum_words)
            break;
          word_count += instruction_words;
          --run_begin;
        }
        if (word_count >= work_limits.minimum_words) {
          if (!charge_discovery(word_count))
            return false;
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
      return true;
    };

    std::optional<uint64_t> expected_offset;
    uint32_t clause_remaining = 0u;
    for (const Instruction &instruction : block->instructions()) {
      if (!charge_discovery()) {
        discovery_exhausted = true;
        break;
      }
      const uint64_t begin = instruction.src_loc();
      const uint64_t end =
          instruction.size() > 0 ? begin + static_cast<uint64_t>(instruction.size()) : begin;
      const bool clause_blocked = clause_remaining != 0u || instruction.mnemonic() == "s_clause";
      const std::optional<bool> protected_overlap = overlaps_protected(begin, end);
      if (!protected_overlap) {
        discovery_exhausted = true;
        break;
      }
      const bool admissible =
          !clause_blocked &&
          is_consan_branch_relay_reservoir_instruction(instruction, begin, pristine_text, arch) &&
          !*protected_overlap && (!expected_offset || begin == *expected_offset);
      if (!admissible && !flush_run()) {
        discovery_exhausted = true;
        break;
      }
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
    if (discovery_exhausted || !flush_run()) {
      discovery_exhausted = true;
      break;
    }
  }
  if (discovery_exhausted) {
    report_exhaustion();
    return false;
  }

  if (!charge_discovery(saturated_multiply(
          candidates.size(), std::max<size_t>(std::bit_width(candidates.size()), 1u)))) {
    report_exhaustion();
    return false;
  }
  // Appended bodies lie after pristine text. Keep candidates ordered so each
  // iteration can jump to the earliest sequence within one SOPP hop of the
  // current frontier instead of materializing every intervening run.
  std::ranges::sort(candidates, {}, &Candidate::offset);

  // Relocating a reservoir through prior relays consumes router capacity.
  // Keep the router, placement state, and owner index in one transaction so a
  // late work-limit or ownership failure cannot leak a partially extended
  // frontier to the caller.
  BranchOnlyRelayRouter planned_router = *this;
  DbiPatchPlacementPlanner planned_planner = placement_planner;
  BranchOnlyDirectRelayReservoirSet planned_reservoirs = reservoirs;

  // Only generated banks and already materialized reservoirs may relocate a
  // speculative donor. Pristine NOPs and selected anchor tails remain reserved
  // for the eventual device route, so they must not pull the recursive donor
  // frontier away from appended storage.
  uint64_t recursive_frontier = planned_planner.appended_end();
  for (const auto &[offset, offer] : planned_router.relays_) {
    if (!charge_discovery()) {
      report_exhaustion();
      return false;
    }
    if (offer.provenance == BranchOnlyRelayProvenance::GeneratedBank ||
        offer.provenance == BranchOnlyRelayProvenance::OwnedReservoir) {
      recursive_frontier = offset;
      break;
    }
  }

  std::vector<bool> adopted_candidates(candidates.size(), false);
  bool stage_active = false;
  uint64_t stage_search_offset = 0u;
  uint64_t stage_hard_lower_bound = 0u;
  uint64_t stage_upper_bound = 0u;
  size_t stage_offered_relay_count = 0u;
  bool stage_uses_hard_lower_bound = false;
  for (;;) {
    if (!stage_active) {
      stage_upper_bound = recursive_frontier;
      stage_hard_lower_bound = stage_upper_bound > kSoppBranchMaximumForwardReachBytes
                                   ? stage_upper_bound - kSoppBranchMaximumForwardReachBytes
                                   : 0u;
      const uint64_t half_reach = kSoppBranchMaximumForwardReachBytes / 2u;
      stage_search_offset = stage_upper_bound > half_reach ? stage_upper_bound - half_reach : 0u;
      stage_offered_relay_count = 0u;
      stage_uses_hard_lower_bound = stage_search_offset == stage_hard_lower_bound;
      stage_active = true;
    }
    auto candidate_it =
        std::ranges::lower_bound(candidates, stage_search_offset, {}, &Candidate::offset);
    bool adopted_candidate = false;
    for (; candidate_it != candidates.end(); ++candidate_it) {
      const size_t candidate_index = static_cast<size_t>(candidate_it - candidates.begin());
      if (adopted_candidates[candidate_index])
        continue;
      const Candidate &candidate = *candidate_it;
      if (!charge_discovery(saturated_add(candidate.words.size(), 1u))) {
        report_exhaustion();
        return false;
      }
      const size_t candidate_bytes = saturated_multiply(candidate.words.size(), sizeof(uint32_t));
      if (candidate_bytes > std::numeric_limits<uint32_t>::max() ||
          candidate_bytes > std::numeric_limits<uint64_t>::max() - candidate.offset) {
        continue;
      }
      const uint64_t candidate_end = candidate.offset + candidate_bytes;
      if (candidate.offset >= stage_upper_bound)
        break;
      const auto first_existing = planned_router.relays_.lower_bound(candidate.offset);
      if (first_existing != planned_router.relays_.end() && first_existing->first < candidate_end)
        continue;

      DbiPatchPlacementRequest request;
      request.anchor_offset = candidate.offset;
      request.original_size = static_cast<uint32_t>(candidate_bytes);
      request.body_size = request.original_size;
      request.inline_capacity = 0u;
      request.allow_appended_cave = true;
      std::string placement_error;
      // The whole function already owns a private router/planner transaction.
      // A failed direct request does not mutate that planner, while a
      // successful direct request is unconditionally adopted below. Avoid
      // cloning either ordered inventory for every candidate.
      std::optional<DbiPatchPlacement> placement = planned_planner.plan(request, &placement_error);
      std::optional<DbiPatchPlacementPlanner> routed_planner;
      std::optional<BranchOnlyRelayRoute> route;
      if (!placement) {
        routed_planner.emplace(planned_planner);
        placement_error.clear();
        if (candidate_bytes > std::numeric_limits<uint64_t>::max() - sizeof(uint32_t))
          continue;
        placement = routed_planner->plan_indirect_appended(
            candidate.offset, static_cast<uint32_t>(candidate_bytes),
            candidate_bytes + sizeof(uint32_t), &placement_error);
        if (!placement)
          continue;
        placement->body_size = candidate_bytes;
        placement->return_branch_offset = placement->body_offset + candidate_bytes;

        BranchOnlyRelayPlanOutcome route_outcome;
        std::string route_error;
        route = planned_router.plan_pair(
            *routed_planner, candidate.offset, placement->body_offset,
            placement->return_branch_offset, candidate_end, &route_error, &route_outcome,
            branch_only_relay_greedy_pair_limits(planned_router.available_count()));
        if (!route)
          continue;
        // A speculative reservoir must not consume pristine NOPs or selected
        // access-anchor tails that the eventual device patch may need. Only
        // the generated bank and already-materialized reservoir dependencies
        // are dedicated to extending this frontier.
        const bool owns_every_dependency =
            std::ranges::all_of(route->claims, [](const BranchOnlyRelayClaim &claim) {
              return claim.provenance == BranchOnlyRelayProvenance::GeneratedBank ||
                     claim.provenance == BranchOnlyRelayProvenance::OwnedReservoir;
            });
        if (!owns_every_dependency || !planned_router.commit(*route, &route_error))
          continue;
      } else if (placement->kind != DbiPatchPlacementKind::AppendedCave) {
        report(error_out, "branch-only router direct reservoir received an invalid placement");
        return false;
      }

      BranchOnlyDirectRelayReservoir reservoir{
          .anchor_offset = candidate.offset,
          .original_words = candidate.words,
          .placement = *placement,
          .route = route,
      };
      const size_t reservoir_index = planned_reservoirs.reservoirs.size();
      std::vector<uint64_t> adopted_relays;
      adopted_relays.reserve(candidate.words.size() - 1u);
      for (uint64_t word = 1u; word < candidate.words.size(); ++word) {
        const uint64_t relay = candidate.offset + word * sizeof(uint32_t);
        const bool offered = planned_router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir);
        const bool indexed =
            offered && planned_reservoirs.reservoir_by_relay.emplace(relay, reservoir_index).second;
        if (!offered || !indexed) {
          report(error_out, "branch-only router could not atomically adopt a direct reservoir");
          return false;
        }
        adopted_relays.push_back(relay);
      }
      stage_offered_relay_count += adopted_relays.size();
      planned_reservoirs.reservoirs.push_back(std::move(reservoir));
      if (routed_planner)
        planned_planner = std::move(*routed_planner);
      recursive_frontier = std::min(recursive_frontier, candidate.offset + sizeof(uint32_t));
      adopted_candidates[candidate_index] = true;
      stage_search_offset = candidate_end;
      adopted_candidate = true;
      break;
    }

    if (!adopted_candidate && !stage_uses_hard_lower_bound) {
      stage_search_offset = stage_hard_lower_bound;
      stage_uses_hard_lower_bound = true;
      continue;
    }
    if (!adopted_candidate)
      break;
    if (stage_offered_relay_count < target_relay_count)
      continue;

    size_t source_reachable_relay_count = 0u;
    for (auto relay = planned_router.relays_.upper_bound(route_frontier_source);
         relay != planned_router.relays_.end() && source_reachable_relay_count < 2u; ++relay) {
      if (!compute_sopp_branch_simm16(route_frontier_source, relay->first))
        break;
      ++source_reachable_relay_count;
    }
    if (source_reachable_relay_count >= 2u)
      break;
    stage_active = false;
  }
  *this = std::move(planned_router);
  placement_planner = std::move(planned_planner);
  reservoirs = std::move(planned_reservoirs);
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
  const uint64_t original_size =
      saturated_multiply(reservoir.original_words.size(), sizeof(uint32_t));
  const uint64_t appended_bytes = direct_reservoir_appended_bytes(original_size);
  if (reservoir.original_words.size() < 2u ||
      reservoir.placement.anchor_offset != reservoir.anchor_offset ||
      reservoir.placement.original_size != original_size ||
      reservoir.placement.body_size != original_size ||
      reservoir.placement.return_branch_offset != reservoir.placement.body_offset + original_size ||
      reservoir.placement.return_target != reservoir.anchor_offset + original_size) {
    report(error_out, "branch-only router found invalid direct-reservoir geometry");
    return false;
  }
  if (appended_bytes > std::numeric_limits<uint32_t>::max()) {
    report(error_out, "branch-only router direct reservoir appended footprint is too large");
    return false;
  }
  if (reservoir.placement.body_offset > std::numeric_limits<uint64_t>::max() - appended_bytes) {
    report(error_out, "branch-only router direct reservoir exceeds host address space");
    return false;
  }
  const uint64_t emitted_end = reservoir.placement.body_offset + appended_bytes;
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

  const uint64_t entry_target = reservoir.route && !reservoir.route->entry_relay_offsets.empty()
                                    ? reservoir.route->entry_relay_offsets.front()
                                    : reservoir.placement.body_offset;
  const uint64_t return_target = reservoir.route && !reservoir.route->return_relay_offsets.empty()
                                     ? reservoir.route->return_relay_offsets.front()
                                     : reservoir.placement.return_target;
  const auto entry_delta = compute_sopp_branch_simm16(reservoir.anchor_offset, entry_target);
  const auto return_delta =
      compute_sopp_branch_simm16(reservoir.placement.return_branch_offset, return_target);
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
  info.trampoline_size = static_cast<uint32_t>(appended_bytes);
  if (reservoir.route)
    info.branch_only_route = consan_branch_only_continuation(*reservoir.route);
  patches.push_back(std::move(info));
  if (reservoir.route &&
      !emit_and_record(text, *reservoir.route, reservoir.placement.body_offset,
                       reservoir.placement.return_target, arch, patches, error_out)) {
    return false;
  }
  return true;
}

} // namespace rocjitsu
