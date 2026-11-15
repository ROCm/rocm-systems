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
  ContractStatus status = ContractStatus::Unresolved;
  std::vector<std::string> depends_on;
  std::vector<Assumption> sources;

  [[nodiscard]] Json to_json() const;
  static std::optional<RegionContract> from_json(const Json &j, std::string &error);
};

/// Fixed frame fields checked for preservation (names match MachineState::Frame).
[[nodiscard]] const std::vector<std::string> &frame_field_names();

} // namespace rocjitsu::isa_diff
