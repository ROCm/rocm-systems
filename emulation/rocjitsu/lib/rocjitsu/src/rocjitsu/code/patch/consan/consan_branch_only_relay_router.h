// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/trampoline_builder.h"

#include <cstdint>
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

struct BranchOnlyRelayClaim {
  uint64_t offset = 0;
  BranchOnlyRelayProvenance provenance = BranchOnlyRelayProvenance::PristineNop;
};

struct BranchOnlyRelayRoute {
  std::vector<uint64_t> entry_relay_offsets;
  std::vector<uint64_t> return_relay_offsets;
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
  Reservation,
};

struct BranchOnlyRelayBatchPlan {
  std::vector<BranchOnlyRelayRoute> routes;
  std::vector<size_t> rejected_pair_indices;
  BranchOnlyRelayPlanFailure failure = BranchOnlyRelayPlanFailure::None;

  [[nodiscard]] bool complete() const {
    return failure == BranchOnlyRelayPlanFailure::None && rejected_pair_indices.empty();
  }
};

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
  [[nodiscard]] bool offer(uint64_t offset, BranchOnlyRelayProvenance provenance);
  void retire_range(uint64_t offset, uint64_t size);

  [[nodiscard]] std::optional<BranchOnlyRelayRoute>
  plan_pair(DbiPatchPlacementPlanner &tentative_planner, uint64_t entry_source,
            uint64_t entry_target, uint64_t return_source, uint64_t return_target,
            std::string *error_out = nullptr,
            BranchOnlyRelayPlanFailure *failure_out = nullptr) const;

  /// Plans exact request pairs and transactionally reserves pristine relay
  /// words in @p tentative_planner. A failed plan leaves the planner unchanged.
  /// If no complete disjoint assignment exists, returned partial routes are
  /// pair-atomic and `rejected_pair_indices` identifies every omitted pair;
  /// callers may inspect their claims for convergence but must not commit them.
  /// A successful plan updates the planner but does not consume router capacity;
  /// the caller must either commit every returned route or discard its planner
  /// copy together with the plan.
  [[nodiscard]] BranchOnlyRelayBatchPlan
  plan_pairs(DbiPatchPlacementPlanner &tentative_planner,
             std::span<const BranchOnlyRelayPairRequest> requests,
             std::string *error_out = nullptr) const;

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
  std::map<uint64_t, BranchOnlyRelayProvenance> relays_;
};

} // namespace rocjitsu
