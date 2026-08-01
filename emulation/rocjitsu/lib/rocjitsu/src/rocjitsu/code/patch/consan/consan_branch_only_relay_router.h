// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
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

enum class BranchOnlyRelayOwnerKind : uint8_t {
  LdsReservoir,
  DirectReservoir,
};

enum class BranchOnlyRelayOwnerMaterialization : uint8_t {
  Paid,
  Deferred,
};

/// Identifies one relay-storage owner. The producer kind is part of the
/// identity, so equal producer-local keys from different domains cannot share
/// planning or commit state. Whether using the owner adds a materialization
/// cost belongs to each router offer rather than to this identity: the same
/// owner may be deferred in one planning view and already paid in another.
struct BranchOnlyRelayOwnerIdentity {
  BranchOnlyRelayOwnerIdentity() = delete;

  [[nodiscard]] static constexpr BranchOnlyRelayOwnerIdentity lds_reservoir(uint64_t producer_key) {
    return BranchOnlyRelayOwnerIdentity(BranchOnlyRelayOwnerKind::LdsReservoir, producer_key);
  }

  [[nodiscard]] static constexpr BranchOnlyRelayOwnerIdentity
  direct_reservoir(uint64_t producer_key) {
    return BranchOnlyRelayOwnerIdentity(BranchOnlyRelayOwnerKind::DirectReservoir, producer_key);
  }

  [[nodiscard]] constexpr BranchOnlyRelayOwnerKind kind() const { return kind_; }
  [[nodiscard]] constexpr uint64_t producer_key() const { return producer_key_; }

  auto operator<=>(const BranchOnlyRelayOwnerIdentity &) const = default;

private:
  constexpr BranchOnlyRelayOwnerIdentity(BranchOnlyRelayOwnerKind kind, uint64_t producer_key)
      : kind_(kind), producer_key_(producer_key) {}

  BranchOnlyRelayOwnerKind kind_;
  uint64_t producer_key_;
};

struct BranchOnlyRelayClaim {
  uint64_t offset = 0;
  BranchOnlyRelayProvenance provenance = BranchOnlyRelayProvenance::PristineNop;
  /// Optional owner shared by every relay from the same producer. Deferred
  /// owners participate in materialization-cost planning; pre-materialized
  /// owners remain visible for claim and commit identity. Neither changes
  /// relay emission.
  std::optional<BranchOnlyRelayOwnerIdentity> owner_affinity;
  BranchOnlyRelayOwnerMaterialization owner_materialization =
      BranchOnlyRelayOwnerMaterialization::Paid;
};

struct BranchOnlyRelayRoute {
  std::vector<uint64_t> entry_relay_offsets;
  std::vector<uint64_t> return_relay_offsets;
  /// Offered words that overlap committed branch endpoints. They are not
  /// emitted as relays, but commit retires them from future plans.
  std::vector<BranchOnlyRelayClaim> retired_relay_claims;
  std::vector<BranchOnlyRelayClaim> claims;
};

struct BranchOnlyRelayPairRequest {
  uint64_t entry_source = 0;
  uint64_t entry_target = 0;
  uint64_t return_source = 0;
  uint64_t return_target = 0;
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

struct BranchOnlyRelaySearchLimits {
  /// Search states and feasible alternatives allowed for a complete batch.
  size_t batch_base_search_work = 100'000u;
  size_t batch_search_work_per_demand = 4'096u;
  /// Polynomial relay inspections allowed for a complete batch. Per-input
  /// headroom keeps a larger request/relay inventory from shrinking the
  /// useful exact-search window solely because its summaries cost more.
  size_t batch_base_scan_work = 2'000'000u;
  size_t batch_scan_work_per_demand_relay = 256u;
  /// A separate bounded pass may improve the first exact route
  /// lexicographically: newly activated owner groups first, then relay count
  /// for capped exact-pair fallback. Exhausting this allowance retains the
  /// exact route instead of degrading the routing tier. Per-input headroom
  /// keeps the minimization window useful as the number of demands and relay
  /// slots grows.
  size_t batch_route_optimization_search_work = 25'000u;
  size_t batch_route_optimization_search_work_per_demand = 1'024u;
  size_t batch_route_optimization_scan_work = 500'000u;
  size_t batch_route_optimization_scan_work_per_demand_relay = 64u;
  /// Base allowance for endpoint lookup and placement-occupancy inspections
  /// while qualifying the offered relay inventory. This phase is separate
  /// from routing because every later tier consumes the same filtered set.
  /// Per-input headroom covers one relay traversal and the logarithmic
  /// occupied-range query for each relay; only pristine relays consume the
  /// latter. The per-relay default covers the traversal plus the maximum
  /// endpoint-lookup depth for a size_t-sized coordinate set.
  size_t batch_relay_qualification_work = 2'000'000u;
  size_t batch_relay_qualification_work_per_relay = 1u + std::numeric_limits<size_t>::digits;
  size_t batch_relay_qualification_work_per_relay_range_level = 1u;
  /// One-time construction of the ordered fallback inventory. Selected-route
  /// removal and rollback work is charged by the per-pair meters. The default
  /// preserves the original useful inventory window.
  size_t batch_fallback_setup_work = 100'000u;
  /// Per-pair fallback allowances. Setup, exact routing, greedy routing, and
  /// classification are all metered, so total fallback work remains linear in
  /// the number of pairs. Each recursive frame charges its first complete
  /// inventory traversal; a bounded second owner-cost traversal deliberately
  /// reuses that charge to preserve the established one-traversal limits and
  /// telemetry scale. Physical inspections are therefore at most twice the
  /// charged traversal units. Zero base/tier limits are normalized to one
  /// unit; zero per-input headroom is allowed.
  size_t pair_search_work = 8'192u;
  size_t pair_base_scan_work = 100'000u;
  size_t pair_scan_work_per_relay = 64u;
  size_t pair_route_optimization_search_work = 2'048u;
  size_t pair_route_optimization_search_work_per_demand = 512u;
  size_t pair_route_optimization_scan_work = 25'000u;
  size_t pair_route_optimization_scan_work_per_demand_relay = 16u;
  /// Allowance for one greedy routing or independent-classification phase.
  size_t pair_greedy_work = 8'192u;
};

static_assert(sizeof(BranchOnlyRelaySearchLimits) == 20u * sizeof(size_t),
              "update all-field relay search-limit tests");

struct BranchOnlyRelayPlanFlags {
  /// True when a routing search, scan, setup, or greedy tier reached its
  /// limit, including successful plans recovered by a later fallback tier.
  /// The independent route-minimization allowance is reported separately.
  bool work_budget_exhausted = false;
  /// True when defensive validation rejected inconsistent exact-routing
  /// inputs or output. A later exact-pair or greedy tier may still recover the
  /// plan. This is not a budget event.
  bool routing_invariant_failed = false;
  /// True when at least one independent lexicographic route-minimization pass
  /// retained its best known feasible route after reaching its own bound. A
  /// later pair may still degrade for an independent routing-limit reason.
  bool route_optimization_exhausted = false;
  /// True when defensive validation found inconsistent route-minimization
  /// inputs and retained the feasibility route. This is not a budget event.
  bool route_optimization_invariant_failed = false;
};

static_assert(sizeof(BranchOnlyRelayPlanFlags) == 4u * sizeof(bool),
              "map new plan flags in record_branch_only_relay_plan");

struct BranchOnlyRelayPlanOutcome : BranchOnlyRelayPlanFlags {
  BranchOnlyRelayPlanFailure failure = BranchOnlyRelayPlanFailure::None;
  BranchOnlyRelayPlanStrategy strategy = BranchOnlyRelayPlanStrategy::ExactBatch;
  /// All compatibility search work is feasibility-search work. Compatibility
  /// scan work is derived from the independently bounded phases below. Both
  /// exclude the independent route-minimization pass.
  size_t search_work_consumed = 0u;
  /// Route-minimization work is reported independently even when a later
  /// routing reservation discards the improved route.
  size_t route_optimization_search_work_consumed = 0u;
  size_t route_optimization_scan_work_consumed = 0u;
  /// These phase counters saturating-sum to the compatibility scan total.
  size_t relay_qualification_work_consumed = 0u;
  size_t fallback_setup_work_consumed = 0u;
  size_t feasibility_scan_work_consumed = 0u;

  [[nodiscard]] size_t scan_work_consumed() const {
    const auto saturated_add = [](size_t lhs, size_t rhs) {
      return rhs > std::numeric_limits<size_t>::max() - lhs ? std::numeric_limits<size_t>::max()
                                                            : lhs + rhs;
    };
    return saturated_add(
        saturated_add(relay_qualification_work_consumed, fallback_setup_work_consumed),
        feasibility_scan_work_consumed);
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

/// Records one router invocation using pair-counted units shared by FLAT, LDS,
/// and MOI producers. A greedy pair also counts as an exact-pair fallback
/// attempt because it reaches that tier first.
void record_branch_only_relay_plan(ConSanBranchOnlyRoutingTelemetry &telemetry,
                                   const BranchOnlyRelayPlanOutcome &outcome,
                                   std::span<const BranchOnlyRelayPlanStrategy> pair_strategies);

/// Records the aggregate failure for a single-pair call. Batch callers instead
/// record per-pair causes with `record_branch_only_relay_rejection` and use
/// this helper only for Reservation, which has no per-pair representation.
void record_branch_only_relay_failure(ConSanBranchOnlyRoutingTelemetry &telemetry,
                                      BranchOnlyRelayPlanFailure failure);

void record_branch_only_relay_rejection(ConSanBranchOnlyRoutingTelemetry &telemetry,
                                        BranchOnlyRelayPairRejection rejection);

/// Returns the component-wise activity accumulated after a prior snapshot.
/// Every telemetry field is kept here so speculative producers share the same
/// delta definition.
[[nodiscard]] ConSanBranchOnlyRoutingTelemetry
branch_only_relay_telemetry_delta(const ConSanBranchOnlyRoutingTelemetry &after,
                                  const ConSanBranchOnlyRoutingTelemetry &before);

struct BranchOnlyDirectRelayReservoir {
  uint64_t anchor_offset = 0;
  std::vector<uint32_t> original_words;
  DbiPatchPlacement placement;
  bool used = false;
};

struct BranchOnlyDirectRelayReservoirSet {
  std::vector<BranchOnlyDirectRelayReservoir> reservoirs;
  std::unordered_map<uint64_t, size_t> reservoir_by_relay;

  [[nodiscard]] bool mark_claims_used(std::span<const BranchOnlyRelayClaim> claims,
                                      std::string *error_out = nullptr);
  [[nodiscard]] ConSanBranchOnlyReservoirTelemetry telemetry() const;
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
  /// Offers one capacity-one relay. Relays with the same non-null owner
  /// identity form one owner group within this router instance; identities
  /// from independent routers have no shared namespace. The domain is
  /// structural, so different producer kinds cannot collide on a numeric
  /// value. Storage with no producer identity uses the ownerless overload.
  ///
  /// The three overloads distinguish ownerless storage, a deferred owner, and
  /// a paid owner without an optional identity or cost-state sentinel at call
  /// sites. For lazily selected LDS reservoirs, a newly used deferred group
  /// represents one materialization and convergence replay, so group count is
  /// the primary objective even though reservoir byte sizes vary. Preplanned
  /// direct reservoirs use `offer_materialized_owner` because their appended
  /// storage is already committed before routing. The objective is intentionally
  /// unweighted; retained appended bytes are reported separately rather than
  /// used as a secondary score.
  [[nodiscard]] bool offer(uint64_t offset, BranchOnlyRelayProvenance provenance) {
    return offer_with_owner_materialization(offset, provenance, std::nullopt,
                                            BranchOnlyRelayOwnerMaterialization::Paid);
  }
  [[nodiscard]] bool offer(uint64_t offset, BranchOnlyRelayProvenance provenance,
                           BranchOnlyRelayOwnerIdentity deferred_owner) {
    return offer_with_owner_materialization(offset, provenance, deferred_owner,
                                            BranchOnlyRelayOwnerMaterialization::Deferred);
  }
  [[nodiscard]] bool offer_materialized_owner(uint64_t offset, BranchOnlyRelayProvenance provenance,
                                              BranchOnlyRelayOwnerIdentity paid_owner) {
    return offer_with_owner_materialization(offset, provenance, paid_owner,
                                            BranchOnlyRelayOwnerMaterialization::Paid);
  }
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
  /// Exact routing first preserves a complete route, then minimizes the number
  /// of newly used owner groups in an independent bounded pass; group count
  /// outranks relay count. A complete batch deliberately does not use route
  /// length as a secondary objective because proving that wider optimum would
  /// enlarge the bounded search. The result remains capacity-safe but may
  /// retain more relay claims than the feasibility baseline. Exact-pair
  /// fallback searches only routes no longer than its feasibility baseline.
  /// For every nonfinal pair, optimization is restricted to that baseline's
  /// relay set and the complete set remains unavailable to later pairs until
  /// batch planning ends. This preserves the deterministic feasibility path
  /// while shorter subset routes reduce the claims retained after the batch.
  /// The final pair may optimize over the remaining inventory. The tier
  /// minimizes relay count at the zero-owner lower bound and among equal-owner
  /// improvements. It minimizes each pair against owners selected by earlier
  /// pairs, so it is greedy across the batch rather than globally minimal.
  /// Owner carry is intentionally scoped to one batch; independent plan calls
  /// have no implicit shared ownership state. Offers may retain a paid owner
  /// identity at zero cost; the exact-pair tier also treats deferred identities
  /// selected by earlier fallback pairs in that same call as paid. The final
  /// greedy fallback is owner-oblivious and preserves pair feasibility rather
  /// than any minimization guarantee. `route_optimization_exhausted` reports a
  /// retained feasible route whose independent minimization pass reached its
  /// bound. `routing_invariant_failed` reports an exact tier rejected for
  /// inconsistent routing inputs or output, while
  /// `route_optimization_invariant_failed` reports inconsistent minimizer
  /// inputs whose validated feasibility route was retained.
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

  [[nodiscard]] bool plan_direct_reservoirs(
      std::span<BasicBlock *const> blocks, std::span<const uint8_t> pristine_text,
      std::span<const std::pair<uint64_t, uint64_t>> protected_ranges, rj_code_arch_t arch,
      uint64_t route_midpoint, size_t target_relay_count,
      DbiPatchPlacementPlanner &placement_planner, BranchOnlyDirectRelayReservoirSet &reservoirs,
      std::string *error_out = nullptr);

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
    std::optional<BranchOnlyRelayOwnerIdentity> owner_affinity;
    BranchOnlyRelayOwnerMaterialization owner_materialization =
        BranchOnlyRelayOwnerMaterialization::Paid;

    bool operator==(const RelayOffer &) const = default;
  };

  [[nodiscard]] bool
  offer_with_owner_materialization(uint64_t offset, BranchOnlyRelayProvenance provenance,
                                   std::optional<BranchOnlyRelayOwnerIdentity> owner_affinity,
                                   BranchOnlyRelayOwnerMaterialization owner_materialization);

  std::map<uint64_t, RelayOffer> relays_;
  // Monotonic: false enables the common no-deferred-owner fast path. Keeping
  // true after the final deferred relay retires only permits a redundant
  // bounded scan.
  bool has_deferred_owner_affinity_ = false;
};

} // namespace rocjitsu
