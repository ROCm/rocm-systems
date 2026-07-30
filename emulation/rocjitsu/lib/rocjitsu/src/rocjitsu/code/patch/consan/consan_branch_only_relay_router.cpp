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
#include <numeric>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <variant>

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

[[nodiscard]] size_t multiply_saturated(size_t lhs, size_t rhs) {
  if (lhs != 0u && rhs > std::numeric_limits<size_t>::max() / lhs)
    return std::numeric_limits<size_t>::max();
  return lhs * rhs;
}

[[nodiscard]] size_t saturated_sum(size_t lhs, size_t rhs) {
  return rhs > std::numeric_limits<size_t>::max() - lhs ? std::numeric_limits<size_t>::max()
                                                        : lhs + rhs;
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
  return saturated_sum(limits.batch_base_search_work,
                       multiply_saturated(demand_count, limits.batch_search_work_per_demand));
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

[[nodiscard]] size_t relay_qualification_work_limit(const BranchOnlyRelaySearchLimits &limits,
                                                    size_t relay_count,
                                                    size_t occupied_range_count) {
  const size_t relay_headroom =
      multiply_saturated(relay_count, limits.batch_relay_qualification_work_per_relay);
  const size_t range_query_levels = std::max<size_t>(std::bit_width(occupied_range_count), 1u);
  const size_t relay_range_levels = multiply_saturated(relay_count, range_query_levels);
  const size_t range_headroom = multiply_saturated(
      relay_range_levels, limits.batch_relay_qualification_work_per_relay_range_level);
  return saturated_sum(limits.batch_relay_qualification_work,
                       saturated_sum(relay_headroom, range_headroom));
}

[[nodiscard]] size_t relay_qualification_lookup_work(size_t endpoint_count) {
  return saturated_sum(1u, std::max<size_t>(std::bit_width(endpoint_count), 1u));
}

[[nodiscard]] size_t route_optimization_search_work(size_t base, size_t per_demand,
                                                    size_t demand_count) {
  return saturated_sum(base, multiply_saturated(per_demand, demand_count));
}

[[nodiscard]] size_t route_optimization_scan_work(size_t base, size_t per_demand_relay,
                                                  size_t demand_count, size_t relay_count) {
  return saturated_sum(
      base, multiply_saturated(multiply_saturated(demand_count, relay_count), per_demand_relay));
}

void add_saturated(size_t &total, size_t value) { total = saturated_sum(total, value); }

inline constexpr size_t kNoNewOwnerGroup = std::numeric_limits<size_t>::max();

class BoundedWorkMeter {
public:
  explicit BoundedWorkMeter(size_t limit) : limit_(std::max<size_t>(limit, 1u)) {}

  [[nodiscard]] bool consume(size_t amount = 1u) {
    if (exhausted_)
      return false;
    if (amount > limit_ - consumed_) {
      exhausted_ = true;
      return false;
    }
    consumed_ += amount;
    return true;
  }

  [[nodiscard]] size_t consumed() const { return consumed_; }
  [[nodiscard]] size_t remaining() const { return exhausted_ ? 0u : limit_ - consumed_; }
  [[nodiscard]] bool exhausted() const { return exhausted_; }

private:
  size_t limit_ = 1u;
  size_t consumed_ = 0u;
  bool exhausted_ = false;
};

struct FarthestReachableRelayResult {
  std::optional<size_t> relay;
  bool exhausted = false;
};

template <typename Admissible>
[[nodiscard]] FarthestReachableRelayResult
farthest_reachable_relay(const FixedRelayDemand &demand, uint64_t cursor,
                         std::span<const uint64_t> relays, Admissible admissible,
                         BoundedWorkMeter &scan_work) {
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

struct RelayOwnerGrouping {
  std::vector<size_t> group_by_relay;
  size_t group_count = 0u;
};

[[nodiscard]] std::optional<RelayOwnerGrouping>
group_relay_owners(std::span<const std::optional<BranchOnlyRelayOwnerIdentity>> owner_affinities,
                   std::span<const BranchOnlyRelayOwnerMaterialization> owner_materializations,
                   const std::set<BranchOnlyRelayOwnerIdentity> &materialized_owner_affinities,
                   BoundedWorkMeter &scan_work) {
  assert(owner_affinities.size() == owner_materializations.size());
  if (owner_affinities.size() != owner_materializations.size())
    return std::nullopt;
  std::vector<std::pair<BranchOnlyRelayOwnerIdentity, size_t>> tagged_relays;
  tagged_relays.reserve(owner_affinities.size());
  for (size_t relay = 0u; relay < owner_affinities.size(); ++relay) {
    if (!scan_work.consume())
      return std::nullopt;
    const std::optional<BranchOnlyRelayOwnerIdentity> owner = owner_affinities[relay];
    if (!owner || owner_materializations[relay] != BranchOnlyRelayOwnerMaterialization::Deferred) {
      continue;
    }
    if (materialized_owner_affinities.empty()) {
      tagged_relays.emplace_back(*owner, relay);
      continue;
    }
    if (!scan_work.consume(
            std::max<size_t>(std::bit_width(materialized_owner_affinities.size()), 1u))) {
      return std::nullopt;
    }
    if (!materialized_owner_affinities.contains(*owner))
      tagged_relays.emplace_back(*owner, relay);
  }
  if (tagged_relays.empty())
    return RelayOwnerGrouping{};
  const size_t sorting_work = multiply_saturated(
      tagged_relays.size(), std::max<size_t>(std::bit_width(tagged_relays.size()), 1u));
  if (!scan_work.consume(sorting_work))
    return std::nullopt;
  std::ranges::sort(tagged_relays);
  RelayOwnerGrouping result{
      .group_by_relay = std::vector<size_t>(owner_affinities.size(), kNoNewOwnerGroup),
  };
  std::optional<BranchOnlyRelayOwnerIdentity> previous_owner;
  for (const auto &[owner, relay] : tagged_relays) {
    if (!previous_owner || *previous_owner != owner) {
      previous_owner = owner;
      ++result.group_count;
    }
    result.group_by_relay[relay] = result.group_count - 1u;
  }
  return result;
}

/// Exact disjoint-path solver for fixed SOPP source/target pairs.
///
/// The relay graph is a one-dimensional DAG: every hop moves monotonically
/// toward its target. The search assigns the most constrained remaining
/// demand first, prunes states whose independent shortest paths need more
/// relays than remain, and backtracks across both entry and return demands.
///
/// Feasibility is intentionally owner-oblivious and retains the established
/// maximum-progress enumeration order. In optimization mode, route enumeration
/// prefers zero-marginal-cost owner groups and then maximum progress within
/// each cost tier, while branch-and-bound minimizes the number of distinct
/// non-materialized owner groups across every demand. Owner activations from
/// earlier assigned demands remain live while later demands are solved, so the
/// active group count is monotonic along a branch and is an admissible
/// incumbent bound. Feasibility does not score route length. Capped
/// optimization retains the shortest encountered route among equal-owner
/// improvements; exact-pair fallback uses that cap to preserve capacity for
/// later pairs.
///
/// Enumeration stops as soon as the target is directly reachable because an
/// additional relay cannot improve owner count and only consumes capacity.
/// `feasibility` stops at the first solution without scoring owners or route
/// length. `bounded_optimization` accepts only a mode produced by
/// `BoundedOptimizationMode::for_owner_groups`: a nonempty owner grouping
/// carries its incumbent owner count and optional relay cap, while an empty
/// grouping is representable only as relay-count minimization with a zero-owner
/// incumbent and an engaged relay cap. Optimization returns `NoImprovement`
/// when its incumbent stands.
///
/// Relay offsets must be sorted and unique. Each solver instance is single-use.
/// Separate deterministic budgets count search states and alternatives, plus
/// every relay inspection performed by polynomial summaries.
class ExactFixedRelayBatchSolver {
  struct BoundedOwnerOptimizationMode {
    std::span<const size_t> new_owner_group_by_relay;
    size_t new_owner_group_count = 0u;
    size_t incumbent_new_owner_count = 0u;
    std::optional<size_t> maximum_relay_count;
  };

  struct RelayCountOptimizationMode {
    size_t incumbent_relay_count = 0u;
  };

public:
  enum class Termination : uint8_t {
    Solved,
    Infeasible,
    NoImprovement,
    WorkBudgetExhausted,
  };

  struct SolveResult {
    Termination termination = Termination::Infeasible;
    /// Optimization may retain an improved route even when proving
    /// optimality reaches its independent work limit.
    bool solution_available = false;
    bool invariant_failed = false;
  };

  class BoundedOptimizationMode {
  public:
    [[nodiscard]] static std::optional<BoundedOptimizationMode>
    for_owner_groups(std::span<const size_t> new_owner_group_by_relay, size_t new_owner_group_count,
                     size_t incumbent_new_owner_count, std::optional<size_t> maximum_relay_count) {
      if (new_owner_group_count == 0u) {
        if (!new_owner_group_by_relay.empty() || incumbent_new_owner_count != 0u ||
            !maximum_relay_count) {
          return std::nullopt;
        }
        return BoundedOptimizationMode(
            RelayCountOptimizationMode{.incumbent_relay_count = *maximum_relay_count});
      }
      if (new_owner_group_by_relay.empty() || incumbent_new_owner_count > new_owner_group_count) {
        return std::nullopt;
      }
      return BoundedOptimizationMode(BoundedOwnerOptimizationMode{
          .new_owner_group_by_relay = new_owner_group_by_relay,
          .new_owner_group_count = new_owner_group_count,
          .incumbent_new_owner_count = incumbent_new_owner_count,
          .maximum_relay_count = maximum_relay_count,
      });
    }

  private:
    friend class ExactFixedRelayBatchSolver;

    explicit BoundedOptimizationMode(BoundedOwnerOptimizationMode mode) : mode_(mode) {}
    explicit BoundedOptimizationMode(RelayCountOptimizationMode mode) : mode_(mode) {}

    std::variant<BoundedOwnerOptimizationMode, RelayCountOptimizationMode> mode_;
  };

  [[nodiscard]] static ExactFixedRelayBatchSolver
  feasibility(std::span<const FixedRelayDemand> demands, std::span<const uint64_t> relays,
              BoundedWorkMeter &search_work, BoundedWorkMeter &scan_work) {
    return ExactFixedRelayBatchSolver(FeasibilityMode{}, demands, relays, search_work, scan_work);
  }

  [[nodiscard]] static ExactFixedRelayBatchSolver
  bounded_optimization(std::span<const FixedRelayDemand> demands, std::span<const uint64_t> relays,
                       const BoundedOptimizationMode &mode, BoundedWorkMeter &search_work,
                       BoundedWorkMeter &scan_work) {
    if (const auto *owner_mode = std::get_if<BoundedOwnerOptimizationMode>(&mode.mode_)) {
      return ExactFixedRelayBatchSolver(*owner_mode, demands, relays, search_work, scan_work);
    }
    return ExactFixedRelayBatchSolver(std::get<RelayCountOptimizationMode>(mode.mode_), demands,
                                      relays, search_work, scan_work);
  }

  [[nodiscard]] SolveResult solve(std::vector<std::vector<uint64_t>> &routes_out) {
    // Every no-solution result owns an empty output, including invalid reuse;
    // leaving routes from an earlier successful call would make the result
    // object and its output disagree.
    routes_out.clear();
    assert(!solve_started_);
    if (solve_started_)
      return {no_solution_termination(), false, true};
    solve_started_ = true;
    if (!inputs_valid_)
      return {no_solution_termination(), false, true};
    (void)solve_remaining(0u);
    if (invariant_failed_)
      return {no_solution_termination(), false, true};
    if (found_better_solution_) {
      routes_out.resize(best_routes_.size());
      for (size_t demand = 0u; demand < best_routes_.size(); ++demand) {
        routes_out[demand].reserve(best_routes_[demand].size());
        for (size_t relay : best_routes_[demand])
          routes_out[demand].push_back(relays_[relay]);
      }
    }
    if (is_feasibility() && found_better_solution_)
      return {Termination::Solved, true, false};
    if (work_budget_exhausted())
      return {Termination::WorkBudgetExhausted, found_better_solution_, false};
    return {found_better_solution_ ? Termination::Solved : no_solution_termination(),
            found_better_solution_, false};
  }

private:
  struct FeasibilityMode {};

  enum class Mode : uint8_t {
    Feasibility,
    OwnerOptimization,
    RelayCountOptimization,
  };

  ExactFixedRelayBatchSolver(FeasibilityMode, std::span<const FixedRelayDemand> demands,
                             std::span<const uint64_t> relays, BoundedWorkMeter &search_work,
                             BoundedWorkMeter &scan_work)
      : demands_(demands), relays_(relays), mode_(Mode::Feasibility),
        available_(relays.size(), true), assigned_(demands.size(), false), routes_(demands.size()),
        search_work_(search_work), scan_work_(scan_work) {
    validate_inputs();
  }

  ExactFixedRelayBatchSolver(BoundedOwnerOptimizationMode mode,
                             std::span<const FixedRelayDemand> demands,
                             std::span<const uint64_t> relays, BoundedWorkMeter &search_work,
                             BoundedWorkMeter &scan_work)
      : demands_(demands), relays_(relays), best_new_owner_count_(mode.incumbent_new_owner_count),
        best_relay_count_(mode.maximum_relay_count), maximum_relay_count_(mode.maximum_relay_count),
        mode_(Mode::OwnerOptimization), available_(relays.size(), true),
        assigned_(demands.size(), false), routes_(demands.size()), search_work_(search_work),
        scan_work_(scan_work), new_owner_group_by_relay_(mode.new_owner_group_by_relay),
        active_new_owner_group_counts_(mode.new_owner_group_count, 0u) {
    validate_inputs();
  }

  ExactFixedRelayBatchSolver(RelayCountOptimizationMode mode,
                             std::span<const FixedRelayDemand> demands,
                             std::span<const uint64_t> relays, BoundedWorkMeter &search_work,
                             BoundedWorkMeter &scan_work)
      : demands_(demands), relays_(relays), best_new_owner_count_(0u),
        best_relay_count_(mode.incumbent_relay_count),
        maximum_relay_count_(mode.incumbent_relay_count), mode_(Mode::RelayCountOptimization),
        available_(relays.size(), true), assigned_(demands.size(), false), routes_(demands.size()),
        search_work_(search_work), scan_work_(scan_work) {
    validate_inputs();
  }

  void validate_inputs() {
    const bool relays_sorted = std::ranges::is_sorted(relays_);
    const bool relays_unique = std::ranges::adjacent_find(relays_) == relays_.end();
    const bool owner_shape_valid =
        new_owner_group_by_relay_.empty() || new_owner_group_by_relay_.size() == relays_.size();
    const bool owner_groups_valid =
        std::ranges::all_of(new_owner_group_by_relay_, [this](size_t group) {
          return group == kNoNewOwnerGroup || group < active_new_owner_group_counts_.size();
        });
    const bool mode_shape_valid = [this] {
      switch (mode_) {
      case Mode::Feasibility:
        return new_owner_group_by_relay_.empty() && active_new_owner_group_counts_.empty() &&
               !best_new_owner_count_ && !maximum_relay_count_;
      case Mode::OwnerOptimization:
        return new_owner_group_by_relay_.size() == relays_.size() &&
               !active_new_owner_group_counts_.empty() && best_new_owner_count_.has_value();
      case Mode::RelayCountOptimization:
        return new_owner_group_by_relay_.empty() && active_new_owner_group_counts_.empty() &&
               best_new_owner_count_ == 0u && maximum_relay_count_.has_value();
      }
      return false;
    }();
    inputs_valid_ = relays_sorted && relays_unique && owner_shape_valid && owner_groups_valid &&
                    mode_shape_valid;
    assert(relays_sorted);
    assert(relays_unique);
    assert(owner_shape_valid);
    assert(owner_groups_valid);
    assert(mode_shape_valid);
  }

  [[nodiscard]] bool is_feasibility() const { return mode_ == Mode::Feasibility; }

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

  [[nodiscard]] Termination no_solution_termination() const {
    return is_feasibility() ? Termination::Infeasible : Termination::NoImprovement;
  }

  [[nodiscard]] size_t owner_group(size_t relay) const {
    if (relay >= new_owner_group_by_relay_.size())
      return kNoNewOwnerGroup;
    const size_t group = new_owner_group_by_relay_[relay];
    return group < active_new_owner_group_counts_.size() ? group : kNoNewOwnerGroup;
  }

  [[nodiscard]] bool owner_is_active(size_t relay) const {
    const size_t group = owner_group(relay);
    return group != kNoNewOwnerGroup && active_new_owner_group_counts_[group] != 0u;
  }

  [[nodiscard]] bool relay_has_zero_marginal_cost(size_t relay) const {
    return owner_group(relay) == kNoNewOwnerGroup || owner_is_active(relay);
  }

  [[nodiscard]] bool can_activate_any_new_owner() const {
    if (!best_new_owner_count_)
      return true;
    const size_t activated_count = active_new_owner_group_count_ + 1u;
    return activated_count < *best_new_owner_count_ ||
           (maximum_relay_count_.has_value() && activated_count == *best_new_owner_count_);
  }

  [[nodiscard]] bool can_activate_owner(size_t relay) const {
    if (relay_has_zero_marginal_cost(relay))
      return true;
    return can_activate_any_new_owner();
  }

  void activate_owner(size_t relay) {
    const size_t group = owner_group(relay);
    if (group == kNoNewOwnerGroup)
      return;
    if (active_new_owner_group_counts_[group]++ == 0u)
      ++active_new_owner_group_count_;
  }

  void deactivate_owner(size_t relay) {
    const size_t group = owner_group(relay);
    if (group == kNoNewOwnerGroup)
      return;
    assert(active_new_owner_group_counts_[group] != 0u);
    if (active_new_owner_group_counts_[group] == 0u) {
      invariant_failed_ = true;
      return;
    }
    if (--active_new_owner_group_counts_[group] == 0u)
      --active_new_owner_group_count_;
  }

  [[nodiscard]] std::optional<size_t> minimum_relay_count(const FixedRelayDemand &demand,
                                                          uint64_t cursor) {
    size_t count = 0u;
    while (!fixed_relay_can_hop(cursor, demand.target)) {
      const auto best = farthest_reachable_relay(
          demand, cursor, relays_,
          [&](size_t relay) { return available_[relay] && can_activate_owner(relay); }, scan_work_);
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
      if (!available_[relay] || !can_activate_owner(relay) ||
          !fixed_relay_is_between(demand, demand.source, relays_[relay]))
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
      if (!can_activate_owner(relay)) {
        return false;
      }
      if (!minimum_relay_count(demand, relays_[relay]))
        return false;
      route.push_back(relay);
      activate_owner(relay);
      const bool stop = enumerate_routes(demand_index, relays_[relay], route, callback);
      deactivate_owner(relay);
      route.pop_back();
      return stop || invariant_failed_;
    };

    // Zero-marginal-cost relays include pristine/generated capacity,
    // previously materialized owners, and owners active in this route. Prefer
    // maximum progress within that complete tier before considering a new
    // owner, while exploring every alternative when proving an improvement.
    // Search work charges every relay inspected by the first traversal,
    // restoring the original per-frame bound even for unavailable or
    // unreachable entries. A mixed inventory may take one bounded second
    // traversal for deferred new-owner entries; it reuses the first-pass
    // charge. Physical inspections are therefore at most twice the charged
    // traversal units per frame, preserving the documented search window.
    bool deferred_new_owner_candidate = false;
    const auto try_cost_tier = [&](bool require_zero_marginal_cost, bool charge_relay_inspections) {
      const auto consider = [&](size_t relay) {
        if (!available_[relay] || !fixed_relay_is_between(demand, cursor, relays_[relay]) ||
            !fixed_relay_can_hop(cursor, relays_[relay])) {
          return false;
        }
        const bool zero_marginal_cost = relay_has_zero_marginal_cost(relay);
        if (zero_marginal_cost != require_zero_marginal_cost) {
          deferred_new_owner_candidate |= require_zero_marginal_cost && !zero_marginal_cost;
          return false;
        }
        return try_relay(relay);
      };
      if (fixed_relay_demand_is_forward(demand)) {
        for (size_t relay = relays_.size(); relay-- != 0u;) {
          if (invariant_failed_ || work_budget_exhausted() ||
              (charge_relay_inspections && !consume_search_work())) {
            return false;
          }
          if (consider(relay))
            return true;
        }
      } else {
        for (size_t relay = 0u; relay < relays_.size(); ++relay) {
          if (invariant_failed_ || work_budget_exhausted() ||
              (charge_relay_inspections && !consume_search_work())) {
            return false;
          }
          if (consider(relay))
            return true;
        }
      }
      return false;
    };
    if (try_cost_tier(/*require_zero_marginal_cost=*/true,
                      /*charge_relay_inspections=*/true)) {
      return true;
    }
    if (deferred_new_owner_candidate && can_activate_any_new_owner() && !work_budget_exhausted() &&
        !invariant_failed_ &&
        try_cost_tier(/*require_zero_marginal_cost=*/false,
                      /*charge_relay_inspections=*/false)) {
      return true;
    }
    return false;
  }

#ifndef NDEBUG
  [[nodiscard]] bool active_owner_count_matches_routes() const {
    std::vector<bool> active_groups(active_new_owner_group_counts_.size(), false);
    for (const std::vector<size_t> &route : routes_) {
      for (size_t relay : route) {
        const size_t group = owner_group(relay);
        if (group != kNoNewOwnerGroup)
          active_groups[group] = true;
      }
    }
    return static_cast<size_t>(std::ranges::count(active_groups, true)) ==
           active_new_owner_group_count_;
  }
#endif

  [[nodiscard]] bool record_solution() {
#ifndef NDEBUG
    assert(active_owner_count_matches_routes());
#endif
    const size_t owner_count = active_new_owner_group_count_;
    if (!best_new_owner_count_) {
      assert(is_feasibility());
      if (!is_feasibility()) {
        invariant_failed_ = true;
        return true;
      }
      best_new_owner_count_ = owner_count;
      best_routes_ = routes_;
      found_better_solution_ = true;
      return true;
    }
    if (owner_count > *best_new_owner_count_)
      return false;
    if (owner_count == *best_new_owner_count_ && !maximum_relay_count_)
      return false;

    std::optional<size_t> relay_count;
    if (maximum_relay_count_) {
      assert(best_relay_count_);
      if (!best_relay_count_) {
        invariant_failed_ = true;
        return true;
      }
      relay_count = std::accumulate(
          routes_.begin(), routes_.end(), size_t{0u},
          [](size_t count, const auto &route) { return saturated_sum(count, route.size()); });
      if (*relay_count > *maximum_relay_count_ ||
          (owner_count == *best_new_owner_count_ && *relay_count >= *best_relay_count_)) {
        return false;
      }
    }
    best_new_owner_count_ = owner_count;
    best_relay_count_ = relay_count;
    best_routes_ = routes_;
    found_better_solution_ = true;
    return is_feasibility() || (owner_count == 0u && !maximum_relay_count_);
  }

  [[nodiscard]] bool solve_remaining(size_t assigned_count) {
    if (invariant_failed_)
      return true;
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
      required_relays = saturated_sum(required_relays, summary->minimum_relay_count);
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
    // Relay count is only a tie-break after owner count is known. Prune the
    // partial route against the caller's feasibility cap, not the incumbent
    // relay count: a longer branch can still improve the primary owner score.
    if (maximum_relay_count_) {
      const size_t assigned_relay_count = std::accumulate(
          routes_.begin(), routes_.end(), size_t{0u},
          [](size_t count, const auto &route) { return saturated_sum(count, route.size()); });
      if (saturated_sum(assigned_relay_count, required_relays) > *maximum_relay_count_)
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
  std::optional<size_t> best_new_owner_count_;
  std::optional<size_t> best_relay_count_;
  std::optional<size_t> maximum_relay_count_;
  Mode mode_;
  bool solve_started_ = false;
  bool inputs_valid_ = false;
  bool invariant_failed_ = false;
  bool found_better_solution_ = false;
  std::vector<bool> available_;
  std::vector<bool> assigned_;
  std::vector<std::vector<size_t>> routes_;
  std::vector<std::vector<size_t>> best_routes_;
  BoundedWorkMeter &search_work_;
  BoundedWorkMeter &scan_work_;
  std::span<const size_t> new_owner_group_by_relay_;
  std::vector<size_t> active_new_owner_group_counts_;
  size_t active_new_owner_group_count_ = 0u;
};

struct ExactOwnerAffinitySolveResult {
  ExactFixedRelayBatchSolver::Termination termination =
      ExactFixedRelayBatchSolver::Termination::Infeasible;
  bool optimization_exhausted = false;
  bool routing_invariant_failed = false;
  bool optimization_invariant_failed = false;
  size_t optimization_search_work = 0u;
  size_t optimization_scan_work = 0u;
  std::vector<uint64_t> feasibility_relay_offsets;
};

/// Returns a binary lower bound on newly activated owner groups.
///
/// Farthest-progress greedy is exact for one monotonic demand over the sorted,
/// unique zero-cost relays, so a demand it cannot route needs a new owner.
/// Otherwise, summing each demand's minimum relay count while allowing every
/// demand to reuse the same relays is a relaxation of the disjoint assignment.
/// If even that sum exceeds the zero-cost inventory, at least one new owner is
/// necessary. The relaxation proves only zero versus one; baselines above one
/// still require the bounded branch-and-bound pass.
[[nodiscard]] std::optional<size_t> provable_new_owner_lower_bound(
    std::span<const FixedRelayDemand> demands, std::span<const uint64_t> relays,
    std::span<const size_t> new_owner_group_by_relay, BoundedWorkMeter &scan_work) {
  if (new_owner_group_by_relay.empty())
    return 0u;
  const bool owner_shape_valid = new_owner_group_by_relay.size() == relays.size();
  const bool relays_sorted = std::ranges::is_sorted(relays);
  const bool relays_unique = std::ranges::adjacent_find(relays) == relays.end();
  assert(owner_shape_valid);
  assert(relays_sorted);
  assert(relays_unique);
  if (!owner_shape_valid || !relays_sorted || !relays_unique)
    return std::nullopt;
  size_t zero_cost_relay_count = 0u;
  for (size_t group : new_owner_group_by_relay) {
    if (!scan_work.consume())
      return std::nullopt;
    zero_cost_relay_count += group == kNoNewOwnerGroup ? 1u : 0u;
  }

  size_t independently_required_relays = 0u;
  for (const FixedRelayDemand &demand : demands) {
    uint64_t cursor = demand.source;
    while (!fixed_relay_can_hop(cursor, demand.target)) {
      const auto best = farthest_reachable_relay(
          demand, cursor, relays,
          [&](size_t relay) { return new_owner_group_by_relay[relay] == kNoNewOwnerGroup; },
          scan_work);
      if (best.exhausted)
        return std::nullopt;
      if (!best.relay)
        return 1u;
      cursor = relays[*best.relay];
      independently_required_relays = saturated_sum(independently_required_relays, 1u);
      if (independently_required_relays > zero_cost_relay_count)
        return 1u;
    }
  }
  return 0u;
}

[[nodiscard]] ExactOwnerAffinitySolveResult solve_exact_minimum_owner_affinity(
    std::span<const FixedRelayDemand> demands, std::span<const uint64_t> relays,
    std::span<const std::optional<BranchOnlyRelayOwnerIdentity>> owner_affinities,
    std::span<const BranchOnlyRelayOwnerMaterialization> owner_materializations,
    const std::set<BranchOnlyRelayOwnerIdentity> &materialized_owner_affinities,
    BoundedWorkMeter &search_work, BoundedWorkMeter &scan_work, size_t optimization_search_limit,
    size_t optimization_scan_limit, bool constrain_optimized_relay_count,
    bool constrain_optimized_relays_to_baseline, bool has_deferred_owner_affinity,
    std::vector<std::vector<uint64_t>> &routes_out) {
  ExactOwnerAffinitySolveResult result;
  routes_out.clear();
  assert(owner_affinities.size() == relays.size());
  assert(owner_materializations.size() == relays.size());
  if (owner_affinities.size() != relays.size() || owner_materializations.size() != relays.size()) {
    result.routing_invariant_failed = true;
    return result;
  }
  ExactFixedRelayBatchSolver feasibility_solver =
      ExactFixedRelayBatchSolver::feasibility(demands, relays, search_work, scan_work);
  const ExactFixedRelayBatchSolver::SolveResult feasibility = feasibility_solver.solve(routes_out);
  result.termination = feasibility.termination;
  result.routing_invariant_failed = feasibility.invariant_failed;
  if (feasibility.invariant_failed) {
    result.termination = ExactFixedRelayBatchSolver::Termination::Infeasible;
    routes_out.clear();
    return result;
  }
  if (feasibility.termination != ExactFixedRelayBatchSolver::Termination::Solved)
    return result;
  assert(feasibility.solution_available);
  if (!feasibility.solution_available) {
    result.routing_invariant_failed = true;
    result.termination = ExactFixedRelayBatchSolver::Termination::Infeasible;
    routes_out.clear();
    return result;
  }

  const size_t baseline_relay_count = std::accumulate(
      routes_out.begin(), routes_out.end(), size_t{0u},
      [](size_t count, const auto &route) { return saturated_sum(count, route.size()); });
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
    if (result.feasibility_relay_offsets.size() != baseline_relay_count) {
      // Capacity-one relays cannot appear in more than one route. Reject the
      // inconsistent exact assignment and let the caller recover through an
      // independent routing tier; it is not safe to retain or commit.
      result.routing_invariant_failed = true;
      result.termination = ExactFixedRelayBatchSolver::Termination::Infeasible;
      result.feasibility_relay_offsets.clear();
      routes_out.clear();
      return result;
    }
  }

  if (!has_deferred_owner_affinity && !constrain_optimized_relay_count)
    return result;

  BoundedWorkMeter optimization_search_work(optimization_search_limit);
  BoundedWorkMeter optimization_scan_work(optimization_scan_limit);
  const auto record_optimization_work = [&] {
    result.optimization_search_work = optimization_search_work.consumed();
    result.optimization_scan_work = optimization_scan_work.consumed();
  };

  std::vector<uint64_t> constrained_relays;
  std::vector<std::optional<BranchOnlyRelayOwnerIdentity>> constrained_owner_affinities;
  std::vector<BranchOnlyRelayOwnerMaterialization> constrained_owner_materializations;
  std::span<const uint64_t> optimization_relays = relays;
  std::span<const std::optional<BranchOnlyRelayOwnerIdentity>> optimization_owner_affinities =
      owner_affinities;
  std::span<const BranchOnlyRelayOwnerMaterialization> optimization_owner_materializations =
      owner_materializations;
  if (constrain_optimized_relays_to_baseline) {
    constrained_relays.reserve(result.feasibility_relay_offsets.size());
    constrained_owner_affinities.reserve(result.feasibility_relay_offsets.size());
    constrained_owner_materializations.reserve(result.feasibility_relay_offsets.size());
    const size_t relay_lookup_work = std::max<size_t>(std::bit_width(relays.size()), 1u);
    for (uint64_t offset : result.feasibility_relay_offsets) {
      if (!optimization_scan_work.consume(relay_lookup_work)) {
        result.optimization_exhausted = true;
        record_optimization_work();
        return result;
      }
      const auto relay = std::ranges::lower_bound(relays, offset);
      assert(relay != relays.end() && *relay == offset);
      if (relay == relays.end() || *relay != offset) {
        result.optimization_invariant_failed = true;
        record_optimization_work();
        return result;
      }
      const size_t relay_index = static_cast<size_t>(std::distance(relays.begin(), relay));
      constrained_relays.push_back(offset);
      constrained_owner_affinities.push_back(owner_affinities[relay_index]);
      constrained_owner_materializations.push_back(owner_materializations[relay_index]);
    }
    optimization_relays = constrained_relays;
    optimization_owner_affinities = constrained_owner_affinities;
    optimization_owner_materializations = constrained_owner_materializations;
  }

  const std::optional<RelayOwnerGrouping> owner_groups =
      group_relay_owners(optimization_owner_affinities, optimization_owner_materializations,
                         materialized_owner_affinities, optimization_scan_work);
  if (!owner_groups) {
    result.optimization_exhausted = true;
    record_optimization_work();
    return result;
  }

  size_t baseline_owner_count = 0u;
  if (owner_groups->group_count != 0u) {
    std::vector<bool> baseline_owner_selected(owner_groups->group_count, false);
    const size_t relay_lookup_work =
        std::max<size_t>(std::bit_width(optimization_relays.size()), 1u);
    for (const auto &route : routes_out) {
      for (uint64_t offset : route) {
        if (!optimization_scan_work.consume(relay_lookup_work)) {
          result.optimization_exhausted = true;
          record_optimization_work();
          return result;
        }
        const auto relay = std::ranges::lower_bound(optimization_relays, offset);
        assert(relay != optimization_relays.end() && *relay == offset);
        if (relay == optimization_relays.end() || *relay != offset) {
          result.optimization_invariant_failed = true;
          record_optimization_work();
          return result;
        }
        const size_t relay_index =
            static_cast<size_t>(std::distance(optimization_relays.begin(), relay));
        const size_t group = owner_groups->group_by_relay[relay_index];
        if (group != kNoNewOwnerGroup && !baseline_owner_selected[group]) {
          baseline_owner_selected[group] = true;
          ++baseline_owner_count;
        }
      }
    }
  }
  const bool optimize_zero_owner_relay_count =
      constrain_optimized_relay_count && baseline_owner_count == 0u;
  if (!optimize_zero_owner_relay_count &&
      (baseline_owner_count == 0u || owner_groups->group_count == 0u)) {
    record_optimization_work();
    return result;
  }

  std::vector<std::vector<uint64_t>> optimized_routes;
  ExactFixedRelayBatchSolver::SolveResult optimization = {
      ExactFixedRelayBatchSolver::Termination::NoImprovement,
      false,
      false,
  };
  // Exact-pair fallback always minimizes relay count after owner count, so a
  // binary owner lower bound cannot prove its incumbent optimal.
  bool should_optimize = constrain_optimized_relay_count;
  if (!should_optimize) {
    // The accelerator is optional: reserve at least half of the remaining scan
    // allowance for the minimizer itself. If the accelerator cannot finish in
    // its share, run the bounded minimizer instead of consuming its window.
    const size_t lower_bound_scan_limit = optimization_scan_work.remaining() / 2u;
    std::optional<size_t> lower_bound;
    if (lower_bound_scan_limit != 0u) {
      BoundedWorkMeter lower_bound_scan_work(lower_bound_scan_limit);
      lower_bound = provable_new_owner_lower_bound(
          demands, optimization_relays, owner_groups->group_by_relay, lower_bound_scan_work);
      const bool accounted = optimization_scan_work.consume(lower_bound_scan_work.consumed());
      assert(accounted);
      if (!accounted) {
        record_optimization_work();
        result.optimization_invariant_failed = true;
        return result;
      }
    }
    should_optimize = !lower_bound || baseline_owner_count > *lower_bound;
  }
  if (should_optimize) {
    const std::optional<ExactFixedRelayBatchSolver::BoundedOptimizationMode> optimization_mode =
        ExactFixedRelayBatchSolver::BoundedOptimizationMode::for_owner_groups(
            owner_groups->group_by_relay, owner_groups->group_count, baseline_owner_count,
            constrain_optimized_relay_count ? std::optional<size_t>(baseline_relay_count)
                                            : std::nullopt);
    assert(optimization_mode);
    if (!optimization_mode) {
      record_optimization_work();
      result.optimization_invariant_failed = true;
      return result;
    }
    ExactFixedRelayBatchSolver optimization_solver =
        ExactFixedRelayBatchSolver::bounded_optimization(
            demands, optimization_relays, *optimization_mode, optimization_search_work,
            optimization_scan_work);
    optimization = optimization_solver.solve(optimized_routes);
  }
  record_optimization_work();
  result.optimization_exhausted =
      optimization.termination == ExactFixedRelayBatchSolver::Termination::WorkBudgetExhausted;
  result.optimization_invariant_failed =
      result.optimization_invariant_failed || optimization.invariant_failed;
  if (optimization.solution_available)
    routes_out = std::move(optimized_routes);
  return result;
}

struct FixedRelayInventoryEntry {
  std::optional<BranchOnlyRelayOwnerIdentity> owner_affinity;
  BranchOnlyRelayOwnerMaterialization owner_materialization =
      BranchOnlyRelayOwnerMaterialization::Paid;
};

using FixedRelayInventory = std::map<uint64_t, FixedRelayInventoryEntry>;

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
                              BoundedWorkMeter &work) {
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
      if (relay == unused_relays.end() || !fixed_relay_is_between(demand, cursor, relay->first) ||
          !fixed_relay_can_hop(cursor, relay->first)) {
        relay = unused_relays.end();
      }
    } else {
      const uint64_t limit = cursor > kSoppBranchMaximumBackwardReachBytes
                                 ? cursor - kSoppBranchMaximumBackwardReachBytes
                                 : 0u;
      relay = unused_relays.lower_bound(std::max(limit, demand.target + sizeof(uint32_t)));
      if (relay == unused_relays.end() || !fixed_relay_is_between(demand, cursor, relay->first) ||
          !fixed_relay_can_hop(cursor, relay->first)) {
        relay = unused_relays.end();
      }
    }
    if (relay == unused_relays.end())
      return rollback(GreedyFixedRelayRoute::Status::Infeasible);
    if (!work.consume(multiply_saturated(2u, query_work)))
      return rollback(GreedyFixedRelayRoute::Status::WorkBudgetExhausted);
    auto claimed = unused_relays.extract(relay);
    assert(!claimed.empty());
    if (claimed.empty())
      return rollback(GreedyFixedRelayRoute::Status::Infeasible);
    cursor = claimed.key();
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
/// are sorted and unique, affinities are parallel, and `complete` distinguishes
/// the full inventory from a sound prefix retained after bounded qualification.
struct QualifiedRelayInventoryView {
  std::span<const uint64_t> offsets;
  std::span<const std::optional<BranchOnlyRelayOwnerIdentity>> owner_affinities;
  std::span<const BranchOnlyRelayOwnerMaterialization> owner_materializations;
  bool complete = false;

  [[nodiscard]] bool shape_valid() const {
    return offsets.size() == owner_affinities.size() &&
           offsets.size() == owner_materializations.size();
  }
};

/// Recovers pair-atomic routes after the full-batch exact solve fails. This
/// helper owns the fallback inventory, owner affinities materialized by
/// earlier pairs, and the policy that preserves a nonfinal pair's complete
/// feasibility baseline for later requests.
[[nodiscard]] std::optional<std::string> plan_exact_pair_fallbacks(
    std::span<const BranchOnlyRelayPairRequest> requests, const std::vector<bool> &valid_request,
    QualifiedRelayInventoryView qualified_relays, bool has_deferred_owner_affinity,
    const BranchOnlyRelaySearchLimits &limits, BranchOnlyRelayBatchPlan &batch) {
  batch.strategy = BranchOnlyRelayPlanStrategy::ExactPairFallback;
  const bool input_shape_valid =
      valid_request.size() == requests.size() && qualified_relays.shape_valid();
  assert(input_shape_valid);
  if (!input_shape_valid) {
    batch.routing_invariant_failed = true;
    batch.failure = BranchOnlyRelayPlanFailure::Reservation;
    return "branch-only router received inconsistent fallback inventory";
  }

  const std::span<const uint64_t> relay_offsets = qualified_relays.offsets;
  const std::span<const std::optional<BranchOnlyRelayOwnerIdentity>> relay_owner_affinities =
      qualified_relays.owner_affinities;
  const std::span<const BranchOnlyRelayOwnerMaterialization> relay_owner_materializations =
      qualified_relays.owner_materializations;

  BoundedWorkMeter fallback_setup_work(limits.batch_fallback_setup_work);
  const size_t fallback_setup_cost = multiply_saturated(
      relay_offsets.size(), std::max<size_t>(std::bit_width(relay_offsets.size()), 1u));
  const bool fallback_inventory_available = fallback_setup_work.consume(fallback_setup_cost);
  add_saturated(batch.scan_work_consumed, fallback_setup_work.consumed());
  FixedRelayInventory unused_relays;
  if (fallback_inventory_available) {
    for (size_t relay = 0u; relay < relay_offsets.size(); ++relay) {
      unused_relays.emplace(relay_offsets[relay],
                            FixedRelayInventoryEntry{relay_owner_affinities[relay],
                                                     relay_owner_materializations[relay]});
    }
  }

  std::set<BranchOnlyRelayOwnerIdentity> materialized_owner_affinities;
  const auto restore_unused_relays = [&](GreedyFixedRelayRoute &route) {
    unused_relays.merge(route.claimed_relays);
    if (!route.claimed_relays.empty()) {
      batch.routing_invariant_failed = true;
      return false;
    }
    return true;
  };
  const auto materialize_inventory_owners = [&](const FixedRelayInventory &inventory) {
    for (const auto &[offset, entry] : inventory) {
      (void)offset;
      if (entry.owner_affinity &&
          entry.owner_materialization == BranchOnlyRelayOwnerMaterialization::Deferred) {
        materialized_owner_affinities.insert(*entry.owner_affinity);
      }
    }
  };
  // Exact routes draw only from `unused_relays`, which is constructed from
  // this qualified inventory. Resolving against the same domain detects a
  // broken fallback-inventory invariant instead of accepting a broader router
  // offer that was ineligible for this plan.
  const auto materialize_route_owners = [&](const BranchOnlyRelayRoute &route) {
    for (const std::vector<uint64_t> *offsets :
         {&route.entry_relay_offsets, &route.return_relay_offsets}) {
      for (uint64_t offset : *offsets) {
        const auto offered = std::ranges::lower_bound(relay_offsets, offset);
        assert(offered != relay_offsets.end() && *offered == offset);
        if (offered == relay_offsets.end() || *offered != offset) {
          batch.routing_invariant_failed = true;
          return false;
        }
        const size_t relay = static_cast<size_t>(offered - relay_offsets.begin());
        if (relay_owner_affinities[relay] &&
            relay_owner_materializations[relay] == BranchOnlyRelayOwnerMaterialization::Deferred) {
          materialized_owner_affinities.insert(*relay_owner_affinities[relay]);
        }
      }
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
    const bool has_later_valid_request = valid_request_at_or_after[request_index + 1u];

    BoundedWorkMeter pair_search_work(limits.pair_search_work);
    BoundedWorkMeter pair_scan_work(exact_pair_scan_work(limits, unused_relays.size()));
    std::vector<std::vector<uint64_t>> pair_routes;
    ExactFixedRelayBatchSolver::Termination pair_termination =
        ExactFixedRelayBatchSolver::Termination::WorkBudgetExhausted;
    bool pair_routing_invariant_failed = false;
    bool pair_optimization_exhausted = false;
    bool pair_optimization_invariant_failed = false;
    if (pair_scan_work.consume(unused_relays.size())) {
      std::vector<uint64_t> available_relays;
      std::vector<std::optional<BranchOnlyRelayOwnerIdentity>> available_owner_affinities;
      std::vector<BranchOnlyRelayOwnerMaterialization> available_owner_materializations;
      available_relays.reserve(unused_relays.size());
      available_owner_affinities.reserve(unused_relays.size());
      available_owner_materializations.reserve(unused_relays.size());
      for (const auto &[relay, entry] : unused_relays) {
        available_relays.push_back(relay);
        available_owner_affinities.push_back(entry.owner_affinity);
        available_owner_materializations.push_back(entry.owner_materialization);
      }
      const ExactOwnerAffinitySolveResult solve = solve_exact_minimum_owner_affinity(
          pair_demands, available_relays, available_owner_affinities,
          available_owner_materializations, materialized_owner_affinities, pair_search_work,
          pair_scan_work,
          route_optimization_search_work(limits.pair_route_optimization_search_work,
                                         limits.pair_route_optimization_search_work_per_demand,
                                         pair_demands.size()),
          route_optimization_scan_work(limits.pair_route_optimization_scan_work,
                                       limits.pair_route_optimization_scan_work_per_demand_relay,
                                       pair_demands.size(), available_relays.size()),
          /*constrain_optimized_relay_count=*/true,
          /*constrain_optimized_relays_to_baseline=*/has_later_valid_request,
          has_deferred_owner_affinity, pair_routes);
      pair_termination = solve.termination;
      pair_routing_invariant_failed = solve.routing_invariant_failed;
      pair_optimization_exhausted = solve.optimization_exhausted;
      pair_optimization_invariant_failed = solve.optimization_invariant_failed;
      batch.routing_invariant_failed =
          batch.routing_invariant_failed || solve.routing_invariant_failed;
      add_saturated(batch.route_optimization_search_work_consumed, solve.optimization_search_work);
      add_saturated(batch.route_optimization_scan_work_consumed, solve.optimization_scan_work);
      if (pair_termination == ExactFixedRelayBatchSolver::Termination::Solved) {
        const size_t reserved_relay_count = has_later_valid_request
                                                ? solve.feasibility_relay_offsets.size()
                                                : pair_routes[0].size() + pair_routes[1].size();
        const size_t removal_work = multiply_saturated(
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
    if (pair_termination == ExactFixedRelayBatchSolver::Termination::Solved) {
      batch.route_optimization_exhausted =
          batch.route_optimization_exhausted || pair_optimization_exhausted;
      batch.route_optimization_invariant_failed =
          batch.route_optimization_invariant_failed || pair_optimization_invariant_failed;
    }
    add_saturated(batch.search_work_consumed, pair_search_work.consumed());
    add_saturated(batch.scan_work_consumed, pair_scan_work.consumed());

    if (pair_termination == ExactFixedRelayBatchSolver::Termination::Solved) {
      batch.routes[request_index].entry_relay_offsets = std::move(pair_routes[0]);
      batch.routes[request_index].return_relay_offsets = std::move(pair_routes[1]);
      if (!has_later_valid_request) {
        for (uint64_t relay : batch.routes[request_index].entry_relay_offsets)
          unused_relays.erase(relay);
        for (uint64_t relay : batch.routes[request_index].return_relay_offsets)
          unused_relays.erase(relay);
      }
      if (!materialize_route_owners(batch.routes[request_index])) {
        batch.failure = BranchOnlyRelayPlanFailure::Reservation;
        return "branch-only router lost relay ownership during exact fallback";
      }
      continue;
    }

    if (pair_termination == ExactFixedRelayBatchSolver::Termination::WorkBudgetExhausted ||
        pair_routing_invariant_failed) {
      batch.strategy = BranchOnlyRelayPlanStrategy::GreedyPairFallback;
      batch.pair_strategies[request_index] = BranchOnlyRelayPlanStrategy::GreedyPairFallback;
      batch.work_budget_exhausted |=
          pair_termination == ExactFixedRelayBatchSolver::Termination::WorkBudgetExhausted;
      BoundedWorkMeter greedy_work(limits.pair_greedy_work);
      GreedyFixedRelayRoute entry_route =
          plan_greedy_fixed_relay_route(pair_demands[0], unused_relays, greedy_work);
      GreedyFixedRelayRoute return_route;
      if (entry_route.status == GreedyFixedRelayRoute::Status::Solved) {
        return_route = plan_greedy_fixed_relay_route(pair_demands[1], unused_relays, greedy_work);
      }
      add_saturated(batch.scan_work_consumed, greedy_work.consumed());
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
        materialize_inventory_owners(entry_route.claimed_relays);
        materialize_inventory_owners(return_route.claimed_relays);
        continue;
      }
      if (!restore_unused_relays(entry_route)) {
        batch.failure = BranchOnlyRelayPlanFailure::Reservation;
        return "branch-only router could not restore its greedy relay inventory";
      }
      if (entry_route.status == GreedyFixedRelayRoute::Status::WorkBudgetExhausted ||
          return_route.status == GreedyFixedRelayRoute::Status::WorkBudgetExhausted) {
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
    BoundedWorkMeter classification_work(limits.pair_greedy_work);
    const auto probe = [&](const FixedRelayDemand &demand) {
      GreedyFixedRelayRoute result =
          plan_greedy_fixed_relay_route(demand, unused_relays, classification_work);
      if (!restore_unused_relays(result))
        result.status = GreedyFixedRelayRoute::Status::InvariantFailure;
      return result.status;
    };
    const GreedyFixedRelayRoute::Status entry_status = probe(pair_demands[0]);
    const GreedyFixedRelayRoute::Status return_status =
        entry_status == GreedyFixedRelayRoute::Status::WorkBudgetExhausted ||
                entry_status == GreedyFixedRelayRoute::Status::InvariantFailure
            ? entry_status
            : probe(pair_demands[1]);
    add_saturated(batch.scan_work_consumed, classification_work.consumed());
    if (entry_status == GreedyFixedRelayRoute::Status::InvariantFailure ||
        return_status == GreedyFixedRelayRoute::Status::InvariantFailure) {
      batch.failure = BranchOnlyRelayPlanFailure::Reservation;
      return "branch-only router could not restore its relay-classification inventory";
    }
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

  return std::nullopt;
}

} // namespace

void record_branch_only_relay_plan(ConSanBranchOnlyRoutingTelemetry &telemetry,
                                   const BranchOnlyRelayPlanOutcome &outcome,
                                   std::span<const BranchOnlyRelayPlanStrategy> pair_strategies) {
  add_saturated(telemetry.pair_attempt_count, pair_strategies.size());
  add_saturated(telemetry.plan_call_count, 1u);
  if (outcome.work_budget_exhausted)
    add_saturated(telemetry.work_budget_exhaustion_count, 1u);
  if (outcome.routing_invariant_failed)
    add_saturated(telemetry.routing_invariant_failure_count, 1u);
  if (outcome.route_optimization_exhausted)
    add_saturated(telemetry.route_optimization_exhaustion_count, 1u);
  if (outcome.route_optimization_invariant_failed)
    add_saturated(telemetry.route_optimization_invariant_failure_count, 1u);
  add_saturated(telemetry.search_work_count, outcome.search_work_consumed);
  add_saturated(telemetry.scan_work_count, outcome.scan_work_consumed);
  add_saturated(telemetry.route_optimization_search_work_count,
                outcome.route_optimization_search_work_consumed);
  add_saturated(telemetry.route_optimization_scan_work_count,
                outcome.route_optimization_scan_work_consumed);
  for (BranchOnlyRelayPlanStrategy strategy : pair_strategies) {
    if (strategy == BranchOnlyRelayPlanStrategy::ExactPairFallback ||
        strategy == BranchOnlyRelayPlanStrategy::GreedyPairFallback) {
      add_saturated(telemetry.exact_pair_fallback_attempt_count, 1u);
    }
    if (strategy == BranchOnlyRelayPlanStrategy::GreedyPairFallback)
      add_saturated(telemetry.greedy_pair_fallback_attempt_count, 1u);
  }
}

void record_branch_only_relay_failure(ConSanBranchOnlyRoutingTelemetry &telemetry,
                                      BranchOnlyRelayPlanFailure failure) {
  switch (failure) {
  case BranchOnlyRelayPlanFailure::None:
    break;
  case BranchOnlyRelayPlanFailure::EntryRoute:
    add_saturated(telemetry.entry_route_failure_count, 1u);
    break;
  case BranchOnlyRelayPlanFailure::ReturnRoute:
    add_saturated(telemetry.return_route_failure_count, 1u);
    break;
  case BranchOnlyRelayPlanFailure::RelayContention:
    add_saturated(telemetry.relay_contention_failure_count, 1u);
    break;
  case BranchOnlyRelayPlanFailure::WorkBudget:
    add_saturated(telemetry.work_budget_failure_count, 1u);
    break;
  case BranchOnlyRelayPlanFailure::Reservation:
    add_saturated(telemetry.reservation_failure_count, 1u);
    break;
  }
}

void record_branch_only_relay_rejection(ConSanBranchOnlyRoutingTelemetry &telemetry,
                                        BranchOnlyRelayPairRejection rejection) {
  switch (rejection) {
  case BranchOnlyRelayPairRejection::None:
    break;
  case BranchOnlyRelayPairRejection::InvalidEntryCoordinates:
  case BranchOnlyRelayPairRejection::EntryUnreachable:
    add_saturated(telemetry.entry_route_failure_count, 1u);
    break;
  case BranchOnlyRelayPairRejection::InvalidReturnCoordinates:
  case BranchOnlyRelayPairRejection::ReturnUnreachable:
    add_saturated(telemetry.return_route_failure_count, 1u);
    break;
  case BranchOnlyRelayPairRejection::RelayContention:
    add_saturated(telemetry.relay_contention_failure_count, 1u);
    break;
  case BranchOnlyRelayPairRejection::WorkBudget:
    add_saturated(telemetry.work_budget_failure_count, 1u);
    break;
  case BranchOnlyRelayPairRejection::Count:
    assert(false && "rejection count sentinel is not a rejection reason");
    break;
  }
}

static_assert(
    sizeof(ConSanBranchOnlyRoutingTelemetry) == 17u * sizeof(size_t),
    "add new routing counters to branch_only_relay_telemetry_delta and its exhaustive unit test");

ConSanBranchOnlyRoutingTelemetry
branch_only_relay_telemetry_delta(const ConSanBranchOnlyRoutingTelemetry &after,
                                  const ConSanBranchOnlyRoutingTelemetry &before) {
  const auto delta = [](size_t after_value, size_t before_value) {
    assert(after_value >= before_value && "routing telemetry must accumulate monotonically");
    return after_value >= before_value ? after_value - before_value : 0u;
  };
  return {
      .pair_attempt_count = delta(after.pair_attempt_count, before.pair_attempt_count),
      .plan_call_count = delta(after.plan_call_count, before.plan_call_count),
      .entry_route_failure_count =
          delta(after.entry_route_failure_count, before.entry_route_failure_count),
      .return_route_failure_count =
          delta(after.return_route_failure_count, before.return_route_failure_count),
      .relay_contention_failure_count =
          delta(after.relay_contention_failure_count, before.relay_contention_failure_count),
      .work_budget_failure_count =
          delta(after.work_budget_failure_count, before.work_budget_failure_count),
      .work_budget_exhaustion_count =
          delta(after.work_budget_exhaustion_count, before.work_budget_exhaustion_count),
      .routing_invariant_failure_count =
          delta(after.routing_invariant_failure_count, before.routing_invariant_failure_count),
      .route_optimization_exhaustion_count = delta(after.route_optimization_exhaustion_count,
                                                   before.route_optimization_exhaustion_count),
      .route_optimization_invariant_failure_count =
          delta(after.route_optimization_invariant_failure_count,
                before.route_optimization_invariant_failure_count),
      .reservation_failure_count =
          delta(after.reservation_failure_count, before.reservation_failure_count),
      .exact_pair_fallback_attempt_count =
          delta(after.exact_pair_fallback_attempt_count, before.exact_pair_fallback_attempt_count),
      .greedy_pair_fallback_attempt_count = delta(after.greedy_pair_fallback_attempt_count,
                                                  before.greedy_pair_fallback_attempt_count),
      .search_work_count = delta(after.search_work_count, before.search_work_count),
      .scan_work_count = delta(after.scan_work_count, before.scan_work_count),
      .route_optimization_search_work_count = delta(after.route_optimization_search_work_count,
                                                    before.route_optimization_search_work_count),
      .route_optimization_scan_work_count = delta(after.route_optimization_scan_work_count,
                                                  before.route_optimization_scan_work_count),
  };
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

bool BranchOnlyRelayRouter::offer_with_owner_materialization(
    uint64_t offset, BranchOnlyRelayProvenance provenance,
    std::optional<BranchOnlyRelayOwnerIdentity> owner_affinity,
    BranchOnlyRelayOwnerMaterialization owner_materialization) {
  const bool owner_state_valid =
      owner_affinity || owner_materialization == BranchOnlyRelayOwnerMaterialization::Paid;
  assert(owner_state_valid);
  if (offset % sizeof(uint32_t) != 0u || !owner_state_valid)
    return false;
  const bool inserted =
      relays_.emplace(offset, RelayOffer{provenance, owner_affinity, owner_materialization}).second;
  has_deferred_owner_affinity_ |=
      inserted && owner_affinity &&
      owner_materialization == BranchOnlyRelayOwnerMaterialization::Deferred;
  return inserted;
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
  if (requests.empty())
    return batch;

  std::vector<uint64_t> relay_offsets;
  std::vector<std::optional<BranchOnlyRelayOwnerIdentity>> relay_owner_affinities;
  std::vector<BranchOnlyRelayOwnerMaterialization> relay_owner_materializations;
  relay_offsets.reserve(relays_.size());
  relay_owner_affinities.reserve(relays_.size());
  relay_owner_materializations.reserve(relays_.size());

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

  // Offered storage that aliases a branch source or destination is not relay
  // capacity. Likewise, a pristine word that is already reserved or outside
  // the original image cannot become a claim. Offered offsets are unique
  // dwords, so independent read-only qualification proves that any selected
  // subset can be reserved together without the quadratic planner-copy pass.
  // A bounded pass retains its proven prefix; omitted suffix entries can only
  // reduce routing capacity, never invalidate a selected route.
  const size_t occupied_range_count = tentative_planner.occupied_ranges().size();
  const size_t occupancy_query_work = std::max<size_t>(std::bit_width(occupied_range_count), 1u);
  BoundedWorkMeter qualification_work(
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
        if (!tentative_planner.can_reserve_existing_range(offset, sizeof(uint32_t)))
          continue;
      }
      relay_offsets.push_back(offset);
      relay_owner_affinities.push_back(relay.owner_affinity);
      relay_owner_materializations.push_back(relay.owner_materialization);
    }
  }
  const QualifiedRelayInventoryView qualified_relays{
      .offsets = relay_offsets,
      .owner_affinities = relay_owner_affinities,
      .owner_materializations = relay_owner_materializations,
      .complete = relay_inventory_complete,
  };
  assert(qualified_relays.shape_valid());
  add_saturated(batch.scan_work_consumed, qualification_work.consumed());
  batch.work_budget_exhausted |= !qualified_relays.complete;

  ExactFixedRelayBatchSolver::Termination exact_termination =
      ExactFixedRelayBatchSolver::Termination::Infeasible;
  std::vector<std::vector<uint64_t>> solved_routes;
  // Paid identities are encoded in the parallel offer state. The empty carry
  // set here is distinct: a simultaneous batch has no deferred owner selected
  // by an earlier fallback pair in this call.
  const std::set<BranchOnlyRelayOwnerIdentity> no_materialized_owner_affinities;
  if (!demands.empty()) {
    BoundedWorkMeter search_work(exact_batch_search_work(limits, demands.size()));
    BoundedWorkMeter scan_work(
        exact_batch_scan_work(limits, demands.size(), qualified_relays.offsets.size()));
    if (!scan_work.consume(qualified_relays.offsets.size())) {
      exact_termination = ExactFixedRelayBatchSolver::Termination::WorkBudgetExhausted;
    } else {
      const ExactOwnerAffinitySolveResult solve = solve_exact_minimum_owner_affinity(
          demands, qualified_relays.offsets, qualified_relays.owner_affinities,
          qualified_relays.owner_materializations, no_materialized_owner_affinities, search_work,
          scan_work,
          route_optimization_search_work(limits.batch_route_optimization_search_work,
                                         limits.batch_route_optimization_search_work_per_demand,
                                         demands.size()),
          route_optimization_scan_work(limits.batch_route_optimization_scan_work,
                                       limits.batch_route_optimization_scan_work_per_demand_relay,
                                       demands.size(), qualified_relays.offsets.size()),
          /*constrain_optimized_relay_count=*/false,
          /*constrain_optimized_relays_to_baseline=*/false, has_deferred_owner_affinity_,
          solved_routes);
      exact_termination = solve.termination;
      batch.route_optimization_exhausted =
          batch.route_optimization_exhausted || solve.optimization_exhausted;
      batch.routing_invariant_failed =
          batch.routing_invariant_failed || solve.routing_invariant_failed;
      batch.route_optimization_invariant_failed =
          batch.route_optimization_invariant_failed || solve.optimization_invariant_failed;
      add_saturated(batch.route_optimization_search_work_consumed, solve.optimization_search_work);
      add_saturated(batch.route_optimization_scan_work_consumed, solve.optimization_scan_work);
    }
    add_saturated(batch.search_work_consumed, search_work.consumed());
    add_saturated(batch.scan_work_consumed, scan_work.consumed());
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
    batch.work_budget_exhausted |=
        exact_termination == ExactFixedRelayBatchSolver::Termination::WorkBudgetExhausted;
    if (std::optional<std::string> fallback_error =
            plan_exact_pair_fallbacks(requests, valid_request, qualified_relays,
                                      has_deferred_owner_affinity_, limits, batch)) {
      report(error_out, std::move(*fallback_error));
      return batch;
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
        retired.push_back({offset, relay->second.provenance, relay->second.owner_affinity,
                           relay->second.owner_materialization});
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
      route.claims.push_back({offset, relay->second.provenance, relay->second.owner_affinity,
                              relay->second.owner_materialization});
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
      const RelayOffer expected{
          claim.provenance,
          claim.owner_affinity,
          claim.owner_materialization,
      };
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
      const RelayOffer expected{
          retired.provenance,
          retired.owner_affinity,
          retired.owner_materialization,
      };
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

bool BranchOnlyDirectRelayReservoirSet::mark_claims_used(
    std::span<const BranchOnlyRelayClaim> claims, std::string *error_out) {
  std::vector<size_t> reservoir_indices;
  for (const BranchOnlyRelayClaim &claim : claims) {
    if (claim.provenance != BranchOnlyRelayProvenance::OwnedReservoir || !claim.owner_affinity ||
        claim.owner_affinity->kind() != BranchOnlyRelayOwnerKind::DirectReservoir) {
      continue;
    }
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

ConSanBranchOnlyReservoirTelemetry BranchOnlyDirectRelayReservoirSet::telemetry() const {
  ConSanBranchOnlyReservoirTelemetry result;
  for (const BranchOnlyDirectRelayReservoir &reservoir : reservoirs) {
    // Direct placements describe displaced body bytes separately from their
    // immediately following return word.
    const size_t displaced_bytes =
        multiply_saturated(reservoir.original_words.size(), sizeof(uint32_t));
    accumulate_branch_only_reservoir_telemetry(result, displaced_bytes, sizeof(uint32_t),
                                               reservoir.used);
  }
  return result;
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
    const size_t candidate_bytes = multiply_saturated(candidate.words.size(), sizeof(uint32_t));
    if (candidate_bytes > std::numeric_limits<uint32_t>::max() ||
        candidate_bytes > std::numeric_limits<uint64_t>::max() - candidate.offset) {
      continue;
    }
    const uint64_t candidate_end = candidate.offset + candidate_bytes;
    const auto first_existing = relays_.lower_bound(candidate.offset);
    if (first_existing != relays_.end() && first_existing->first < candidate_end)
      continue;

    DbiPatchPlacementPlanner tentative_planner = placement_planner;
    DbiPatchPlacementRequest request;
    request.anchor_offset = candidate.offset;
    request.original_size = static_cast<uint32_t>(candidate_bytes);
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
      const bool offered = offer_materialized_owner(
          relay, BranchOnlyRelayProvenance::OwnedReservoir,
          BranchOnlyRelayOwnerIdentity::direct_reservoir(candidate.offset));
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
  const uint64_t original_size =
      multiply_saturated(reservoir.original_words.size(), sizeof(uint32_t));
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
  info.trampoline_size = static_cast<uint32_t>(appended_bytes);
  patches.push_back(std::move(info));
  return true;
}

} // namespace rocjitsu
