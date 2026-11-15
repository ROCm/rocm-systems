// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file contract.h
/// @brief The verification object for one region pair.
///
/// @details The relation tested by P0 is
///   C(s_old, s_new) AND R_in(s_old, s_new)
///     => both reach the logical exit AND R_out(final_old, final_new) AND frame preserved
/// with
///   - R_in: equality on `entry_equal`, at the same physical location on both sides
///     (identity mapping at a common cutpoint). Locations outside it get independent
///     per-side canaries, which turn a missed dependency into a visible mismatch.
///   - C: `constraints`, each carrying its own source.
///   - R_out: equality on `observable`, the union of both continuations' live-ins.
///   - frame: fixed control state that P0 instructions must not change.
/// Every set records where it came from, and a hand-written contract is never upgraded
/// from ASSUMED.
#pragma once

#include "json.h"
#include "location.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rocjitsu::isa_diff {

enum class ContractStatus { Established, Assumed, Unresolved };
[[nodiscard]] const char *contract_status_name(ContractStatus s);
[[nodiscard]] std::optional<ContractStatus> parse_contract_status(std::string_view s);

/// One entry constraint on a single location's value.
struct Constraint {
  unsigned loc = 0;
  uint32_t known_zero = 0; ///< Bits that must be 0.
  uint32_t known_one = 0;  ///< Bits that must be 1.
  std::optional<uint32_t> umin, umax;
  std::optional<unsigned> equal_to; ///< Must equal another location's value.
  /// One of: "common_prefix_analysis", "relation_summary", "reachability_witness",
  /// "user_assumption".
  std::string source = "user_assumption";
};

struct Assumption {
  std::string what;
  std::string source;
};

struct RegionContract {
  std::string target = "gfx950";
  std::string wave = "wave64";
  std::string arithmetic = "default";
  std::string mapping = "identity_at_common_cutpoint";
  LocSet entry_equal;
  LocSet canary;
  LocSet observable;
  LocSet declared_writes_old; ///< Effect model; execution may change nothing else.
  LocSet declared_writes_new;
  LocSet wide_reads; ///< Input-generation feature: low halves of 64-bit operands.
  std::vector<Constraint> constraints;
  std::string exit_mapping = "single fall-through exit after the last region instruction";
  /// Hand-written register correspondence (handoff 3.1), old location -> new location,
  /// at the region entry and at its exit: after a register-allocation change the two can
  /// differ (an input in s[4:5] vs s[8:9], the result in s4 on both sides). Empty means
  /// identity. entry_equal/canary/observable use old naming; the new side receives inputs
  /// at entry_map(l) and is compared at exit_map(l). Never upgraded beyond ASSUMED.
  std::vector<std::pair<unsigned, unsigned>> entry_map, exit_map;
  [[nodiscard]] bool mapped() const { return !entry_map.empty() || !exit_map.empty(); }
  /// Permutations (forward old->new, inverse new->old) completed from the maps.
  [[nodiscard]] std::array<unsigned, kNumLocs> entry_forward() const;
  [[nodiscard]] std::array<unsigned, kNumLocs> entry_inverse() const;
  [[nodiscard]] std::array<unsigned, kNumLocs> exit_forward() const;
  [[nodiscard]] std::array<unsigned, kNumLocs> exit_inverse() const;
  /// Memory operations outstanding at the cutpoint (handoff 6.1): "NONE"; "DISJOINT"
  /// (outstanding, but their destinations are not read or written by the region);
  /// "CONFLICT" (an outstanding write targets a location the region reads or writes);
  /// "UNKNOWN" (an outstanding op has unknown effects). CONFLICT and UNKNOWN leave the
  /// context unresolved: the isolated run starts with every input ready, real hardware may not.
  std::string pending_effects = "NONE";
  /// Composition check (handoff 5.2): "HOLDS" when must-equality propagated from the
  /// kernel entry through common code and upstream regions' R_out covers entry_equal,
  /// "FAILS" otherwise (then the contract is UNRESOLVED), "NOT_CHECKED" for synthetic or
  /// hand-built contracts.
  std::string composition = "NOT_CHECKED";
  ContractStatus status = ContractStatus::Unresolved;
  std::vector<std::string> depends_on;
  std::vector<Assumption> sources;

  [[nodiscard]] Json to_json() const;
  static std::optional<RegionContract> from_json(const Json &j, std::string &error);
};

/// @brief Parse "s4=s8,s5=s9" into a register map; nullopt (and @p error) when invalid
/// (unknown names, a location mapped twice, or two locations mapped to one).
[[nodiscard]] std::optional<std::vector<std::pair<unsigned, unsigned>>>
parse_register_map(std::string_view text, std::string &error);
[[nodiscard]] std::string format_register_map(const std::vector<std::pair<unsigned, unsigned>> &m);
/// @brief Complete a partial injective map to a permutation (forward old->new): a target
/// that is not itself mapped takes the start of its chain (s4=s8,s8=s9 gives s9 -> s4).
[[nodiscard]] std::array<unsigned, kNumLocs>
complete_register_map(const std::vector<std::pair<unsigned, unsigned>> &m);
[[nodiscard]] std::array<unsigned, kNumLocs>
invert_permutation(const std::array<unsigned, kNumLocs> &forward);

/// Fixed frame fields checked for preservation (names match MachineState::Frame).
[[nodiscard]] const std::vector<std::string> &frame_field_names();

} // namespace rocjitsu::isa_diff
