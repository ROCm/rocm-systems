// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/planning_work.h"
#include "rocjitsu/code/patch/trampoline_builder.h"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu {

class BasicBlock;
class Instruction;

/// Periodic banks are a bounded-density routing aid, not an alternate copy of
/// the transformed text. Keep their payload to at most one eighth of the body
/// interval they serve. This bounds the bank-induced text-growth multiplier
/// while leaving ample SOPP reach between corresponding words in adjacent
/// banks.
[[nodiscard]] constexpr bool
branch_only_periodic_relay_bank_has_bounded_density(size_t bank_word_count,
                                                    uint64_t body_interval_bytes) {
  constexpr uint64_t kMaximumBankDensityDenominator = 8u;
  return body_interval_bytes != 0u &&
         bank_word_count <=
             body_interval_bytes / (kMaximumBankDensityDenominator * sizeof(uint32_t));
}

[[nodiscard]] bool is_consan_branch_relay_reservoir_instruction(const Instruction &instruction,
                                                                uint64_t offset,
                                                                std::span<const uint8_t> text,
                                                                rj_code_arch_t arch);

enum class BranchOnlyRelayProvenance : uint8_t {
  PristineNop,
  GeneratedBank,
  OwnedAnchor,
  OwnedReservoir,
};

struct BranchOnlyRelayClaim {
  uint64_t offset = 0;
  BranchOnlyRelayProvenance provenance = BranchOnlyRelayProvenance::PristineNop;
};

struct BranchOnlyRelayRoute {
  std::vector<uint64_t> entry_relay_offsets;
  std::vector<uint64_t> return_relay_offsets;
  /// Offered words that overlap committed branch endpoints. They are not
  /// emitted as relays, but commit retires them from future plans.
  std::vector<BranchOnlyRelayClaim> retired_relay_claims;
  std::vector<BranchOnlyRelayClaim> claims;
};

/// Project planning-private relay ownership into the exact route retained by
/// an emitted patch. Claims and retired candidates do not cross the lowering
/// boundary.
[[nodiscard]] inline ConSanBranchOnlyContinuation
consan_branch_only_continuation(const BranchOnlyRelayRoute &route) {
  return {ConSanBranchOnlyRelayEntry{route.entry_relay_offsets}, route.return_relay_offsets};
}

struct BranchOnlyRelayPairRequest {
  uint64_t entry_source = 0;
  uint64_t entry_target = 0;
  uint64_t return_source = 0;
  uint64_t return_target = 0;
  /// The caller has already materialized a non-SOPP entry at the anchor. The
  /// router must preserve an empty entry route and spend capacity only on the
  /// return demand. Entry coordinates are ignored in this form.
  bool entry_preplaced = false;
};

enum class BranchOnlyRelayPlanFailure : uint8_t {
  None,
  EntryRoute,
  ReturnRoute,
  RelayContention,
  WorkBudget,
  Reservation,
};

enum class BranchOnlyRelayPairRejection : uint8_t {
  None,
  InvalidEntryCoordinates,
  InvalidReturnCoordinates,
  EntryUnreachable,
  ReturnUnreachable,
  RelayContention,
  WorkBudget,
  Count,
};

inline constexpr size_t kBranchOnlyRelayPairRejectionCount =
    static_cast<size_t>(BranchOnlyRelayPairRejection::Count);

enum class BranchOnlyRelayPlanStrategy : uint8_t {
  ExactBatch,
  ExactPairFallback,
  GreedyPairFallback,
};

struct BranchOnlyRelayQualificationWorkLimits {
  /// Base allowance for endpoint lookup and placement-occupancy inspections
  /// while qualifying the offered relay inventory. This phase is separate
  /// from routing because every later tier consumes the same filtered set.
  /// Per-input headroom covers one relay traversal and the logarithmic
  /// occupied-range query for each relay; only pristine relays consume the
  /// latter. The per-relay default covers the traversal plus the maximum
  /// endpoint-lookup depth for a size_t-sized coordinate set.
  size_t base = 2'000'000u;
  size_t per_relay = 1u + std::numeric_limits<size_t>::digits;
  size_t per_relay_range_level = 1u;
};

struct BranchOnlyRelayBatchWorkLimits {
  /// Search states and feasible alternatives allowed for a complete batch.
  PlanningWorkLimit feasibility_search{100'000u, 4'096u};
  /// Polynomial relay inspections allowed for a complete batch. Per-input
  /// headroom keeps a larger request/relay inventory from shrinking the
  /// useful exact-search window solely because its summaries cost more.
  PlanningWorkLimit feasibility_scan{2'000'000u, 256u};
  /// One-time construction of the ordered fallback inventory. Selected-route
  /// removal and rollback work is charged by the per-pair meters.
  size_t fallback_setup = 100'000u;
};

struct BranchOnlyRelayPairWorkLimits {
  /// Exact routing and its best-known route refinement share one bounded
  /// transaction. Greedy routing and classification are independently
  /// metered, so total fallback work remains linear in the number of pairs.
  /// Each recursive exact-search frame charges its complete inventory
  /// traversal.
  /// Zero base/tier limits are normalized to one unit; zero per-input
  /// headroom is allowed.
  ///
  /// Fixed per-pair allowance. Keeping this unscaled preserves a total bound
  /// linear in the number of pairs.
  PlanningWorkLimit exact_search{10'240u, 512u};
  PlanningWorkLimit exact_scan{125'000u, 96u};
  /// Fixed allowance for one greedy routing or independent-classification
  /// phase. Each pair can consume at most one allowance for each phase.
  size_t greedy = 8'192u;
};

struct BranchOnlyRelaySearchLimits {
  BranchOnlyRelayQualificationWorkLimits qualification;
  BranchOnlyRelayBatchWorkLimits batch;
  BranchOnlyRelayPairWorkLimits pair;
};

/// Configures a bounded plan to skip exponential exact routing and use the
/// pair-atomic greedy tier after one ordered-inventory construction. This is
/// appropriate for large monotonic corridors where feasibility, rather than
/// route minimization, is the caller's contract.
[[nodiscard]] BranchOnlyRelaySearchLimits branch_only_relay_greedy_pair_limits(size_t relay_count);

static_assert(sizeof(BranchOnlyRelayQualificationWorkLimits) == 3u * sizeof(size_t));
static_assert(sizeof(BranchOnlyRelayBatchWorkLimits) == 5u * sizeof(size_t));
static_assert(sizeof(BranchOnlyRelayPairWorkLimits) == 5u * sizeof(size_t));
static_assert(sizeof(BranchOnlyRelaySearchLimits) == 13u * sizeof(size_t),
              "update the phase-scoped relay search-limit tests");

/// Returns a saturating conservative bound for one plan call. It includes
/// qualification, exact-batch routing, fallback setup, and every per-pair
/// exact, minimization, greedy, and classification allowance.
/// Some tiers are mutually exclusive, so the actual charged work cannot
/// exceed this discoverable configuration bound.
[[nodiscard]] size_t
branch_only_relay_conservative_work_limit(const BranchOnlyRelaySearchLimits &limits,
                                          size_t pair_count, size_t relay_count,
                                          size_t occupied_range_count);

struct BranchOnlyRelayPlanFlags {
  /// True when the ordered relay-qualification pass retained only a sound
  /// prefix after reaching its own allowance.
  bool relay_qualification_exhausted = false;
  /// True when exact routing, fallback setup, greedy routing, or rejection
  /// classification reached a routing allowance.
  bool routing_work_exhausted = false;
  /// True when defensive validation rejected inconsistent exact-routing
  /// inputs or output. A later exact-pair or greedy tier may still recover the
  /// plan. This is not a budget event.
  bool routing_invariant_failed = false;
  /// True when qualification or a routing tier reached its configured limit,
  /// including a successful exact-pair plan that retained its best-known
  /// route when refinement exhausted the remainder of the shared allowance.
  [[nodiscard]] bool work_budget_exhausted() const {
    return relay_qualification_exhausted || routing_work_exhausted;
  }
};

static_assert(sizeof(BranchOnlyRelayPlanFlags) == 3u * sizeof(bool));

struct BranchOnlyRelayPlanOutcome : BranchOnlyRelayPlanFlags {
  BranchOnlyRelayPlanFailure failure = BranchOnlyRelayPlanFailure::None;
  BranchOnlyRelayPlanStrategy strategy = BranchOnlyRelayPlanStrategy::ExactBatch;
  /// Search work includes exact feasibility and best-known route refinement.
  /// Scan work is derived from the independently bounded phases below.
  size_t search_work_consumed = 0u;
  /// These phase counters saturating-sum to the compatibility scan total.
  size_t relay_qualification_work_consumed = 0u;
  size_t fallback_setup_work_consumed = 0u;
  size_t feasibility_scan_work_consumed = 0u;
  /// Pristine offers rejected by placement occupancy before routing.
  size_t pristine_relay_occupancy_rejection_count = 0u;
  [[nodiscard]] size_t scan_work_consumed() const {
    return saturated_add(
        saturated_add(relay_qualification_work_consumed, fallback_setup_work_consumed),
        feasibility_scan_work_consumed);
  }

  [[nodiscard]] size_t total_work_consumed() const {
    return saturated_add(search_work_consumed, scan_work_consumed());
  }
};

struct BranchOnlyRelayBatchPlan : BranchOnlyRelayPlanOutcome {
  std::vector<BranchOnlyRelayRoute> routes;
  std::vector<BranchOnlyRelayPairRejection> rejection_reasons;
  /// Per-request strategy. The inherited strategy is the most degraded tier
  /// reached anywhere in the batch.
  std::vector<BranchOnlyRelayPlanStrategy> pair_strategies;
  std::vector<size_t> rejected_pair_indices;

  [[nodiscard]] bool complete() const {
    return failure == BranchOnlyRelayPlanFailure::None && rejected_pair_indices.empty();
  }

  [[nodiscard]] const BranchOnlyRelayPlanOutcome &plan_outcome() const { return *this; }
};

struct BranchOnlyDirectRelayReservoir {
  uint64_t anchor_offset = 0;
  std::vector<uint32_t> original_words;
  DbiPatchPlacement placement;
  /// Present when the relocated sequence reaches its appended body through
  /// already-owned branch-only relays instead of one direct SOPP hop.
  std::optional<BranchOnlyRelayRoute> route;
  bool used = false;
};

struct BranchOnlyDirectRelayReservoirSet {
  std::vector<BranchOnlyDirectRelayReservoir> reservoirs;
  std::unordered_map<uint64_t, size_t> reservoir_by_relay;

  [[nodiscard]] bool mark_relays_used(std::span<const uint64_t> relays,
                                      std::string *error_out = nullptr);
  [[nodiscard]] bool mark_claims_used(std::span<const BranchOnlyRelayClaim> claims,
                                      std::string *error_out = nullptr);
};

struct BranchOnlyDirectReservoirWorkLimits {
  /// The input is a conservative complexity unit derived from pristine text
  /// words and protected ranges. It includes logarithmic headroom for the two
  /// ordered inventories built by discovery. Exhaustion fails the complete
  /// discovery call and rolls back every adopted reservoir: a partial
  /// candidate inventory is not a sound substitute for a whole-input result.
  /// The margin also guards future algorithmic growth; programmatic callers
  /// may override it through ConSanOptions.
  PlanningWorkLimit discovery = kDefaultDirectReservoirPlanningWorkLimit;
  /// Bounds each relocated straight-line donor. The defaults preserve the
  /// compact access-engine policy; callers with a denser relay demand may
  /// admit a wider donor without changing reservoir mechanics.
  size_t minimum_words = 16u;
  size_t maximum_words = 64u;
};

/// Owns capacity-one branch relay hosts from discovery through emission.
///
/// Original NOPs require a placement reservation. Generated banks, anchor
/// tails, and reservoirs are offered only after their storage or donor patch
/// is selected, so their ranges are already owned.
///
/// This router preserves fixed source/target pairs. The generic SOPP relay
/// planners are for interchangeable island destinations instead.
class BranchOnlyRelayRouter {
public:
  /// Offers one capacity-one relay. Storage ownership is established before
  /// it is offered; the router tracks only relay capacity and provenance.
  [[nodiscard]] bool offer(uint64_t offset, BranchOnlyRelayProvenance provenance);
  void retire_range(uint64_t offset, uint64_t size);

  [[nodiscard]] std::optional<BranchOnlyRelayRoute>
  plan_pair(DbiPatchPlacementPlanner &tentative_planner, uint64_t entry_source,
            uint64_t entry_target, uint64_t return_source, uint64_t return_target,
            std::string *error_out = nullptr, BranchOnlyRelayPlanOutcome *outcome_out = nullptr,
            const BranchOnlyRelaySearchLimits &limits = {}) const;

  /// Plans fixed request pairs and transactionally reserves pristine relay
  /// words in @p tentative_planner. Exact backtracking has a deterministic work
  /// budget; exhaustion falls back to bounded exact pair routing, then to
  /// greedy pair-atomic routing. `strategy` and `work_budget_exhausted`
  /// report degradation even when fallback completes the batch. Relay
  /// qualification retains its sound prefix when bounded, so later tiers may
  /// still complete from the proven subset. A failed plan leaves the planner
  /// unchanged.
  ///
  /// Exact routing first preserves a complete route. Exact-pair fallback then
  /// uses the remainder of that same bounded transaction to search only routes
  /// shorter than its feasibility baseline.
  /// For every nonfinal pair, optimization is restricted to that baseline's
  /// relay set and the complete set remains unavailable to later pairs until
  /// batch planning ends. This preserves the deterministic feasibility path
  /// while shorter subset routes reduce the claims retained after the batch.
  /// The final pair may optimize over the remaining inventory. The tier
  /// minimizes relay count independently for each pair. The final greedy
  /// fallback preserves pair feasibility rather than any minimization
  /// guarantee. `routing_work_exhausted` may accompany a successful retained
  /// route when refinement reaches the shared bound. `routing_invariant_failed`
  /// reports an exact tier rejected for inconsistent routing inputs or output.
  /// If no complete disjoint assignment exists, returned partial routes are
  /// pair-atomic. `rejection_reasons` is indexed like the requests, and
  /// `rejected_pair_indices` identifies every omitted pair; callers may inspect
  /// their claims for convergence but must not commit them.
  /// A successful plan updates the planner but does not consume router capacity;
  /// the caller must either commit every returned route or discard its planner
  /// copy together with the plan.
  [[nodiscard]] BranchOnlyRelayBatchPlan
  plan_pairs(DbiPatchPlacementPlanner &tentative_planner,
             std::span<const BranchOnlyRelayPairRequest> requests, std::string *error_out = nullptr,
             const BranchOnlyRelaySearchLimits &limits = {}) const;

  [[nodiscard]] bool commit(const BranchOnlyRelayRoute &route, std::string *error_out = nullptr);
  [[nodiscard]] bool commit(std::span<const BranchOnlyRelayRoute> routes,
                            std::string *error_out = nullptr);

  /// Relocates safe ordinary instruction runs into appended storage and offers
  /// their vacated tail words as materialized branch-only relays. A run that
  /// cannot reach appended storage directly may use already planned relay
  /// capacity, recursively extending the frontier toward
  /// @p route_frontier_source. The router, placement planner, and reservoir
  /// inventory are committed atomically.
  [[nodiscard]] bool plan_direct_reservoirs(
      std::span<BasicBlock *const> blocks, std::span<const uint8_t> pristine_text,
      std::span<const std::pair<uint64_t, uint64_t>> protected_ranges, rj_code_arch_t arch,
      uint64_t route_frontier_source, size_t target_relay_count,
      DbiPatchPlacementPlanner &placement_planner, BranchOnlyDirectRelayReservoirSet &reservoirs,
      std::string *error_out = nullptr, const BranchOnlyDirectReservoirWorkLimits &work_limits = {},
      PlanningWorkMeasurement *work_measurement = nullptr);

  [[nodiscard]] static bool
  emit_and_record(std::span<uint8_t> text, const BranchOnlyRelayRoute &route, uint64_t entry_target,
                  uint64_t return_target, rj_code_arch_t arch,
                  std::vector<ConSanPatchInfo> &patches, std::string *error_out = nullptr);

  [[nodiscard]] static bool emit_direct_reservoir(std::vector<uint8_t> &text,
                                                  const BranchOnlyDirectRelayReservoir &reservoir,
                                                  rj_code_arch_t arch,
                                                  std::vector<ConSanPatchInfo> &patches,
                                                  std::string *error_out = nullptr);

  [[nodiscard]] size_t available_count() const { return relays_.size(); }

private:
  struct RelayOffer {
    BranchOnlyRelayProvenance provenance = BranchOnlyRelayProvenance::PristineNop;

    bool operator==(const RelayOffer &) const = default;
  };

  std::map<uint64_t, RelayOffer> relays_;
};

} // namespace rocjitsu
