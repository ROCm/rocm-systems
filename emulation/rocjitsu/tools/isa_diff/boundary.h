// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file boundary.h
/// @brief Mandatory boundary, NaN and corner-case sweep (always run before random cases).
///
/// @details Every `test` run executes this deterministic sweep in full before any random
/// case; there is no option to skip it. A size cap exists only as a safety limit, and
/// reaching it is reported as partial coverage. The sweep is built from named value
/// catalogs:
///   - 32-bit: integer limits (0, 1, -1, INT_MIN/MAX and neighbours), byte/half
///     sign-extension edges, small shift counts, bit patterns, single bits and their
///     inverses, and f32/f16/bf16 special encodings (+-0, +-Inf, quiet/signalling NaN,
///     NaN payloads, denormals, min/max normal, packed halves);
///   - 64-bit pairs: int64 limits, half-crossing values (2^32-1, 2^32), f64 specials,
///     and EXEC lane patterns (lane 0, lane 63, low/high 32 lanes, alternating).
/// The sweep runs five phases:
///   1. uniform: every entry register at the same catalog value;
///   2. single: each register the region reads through the 32-bit catalog plus its role values
///      (shift amounts, S_BFE fields, sign-extension edges, bit indexes, values
///      relative to constants it is combined with), the other registers held at zeros,
///      ones or a seeded random base, with both SCC values when SCC is an input;
///   3. pair64: each 64-bit operand through the 64-bit catalog;
///   4. pairs: each two-register source pair through carry, borrow, overflow, equal,
///      complementary and NaN pairs and a core cross product;
///   5. constraint edges: umin/umax and their neighbours, known-bit extremes.
/// Base states are projected onto the contract's constraints. A sweep value that
/// violates a constraint is rejected and counted, never silently changed.
#pragma once

#include "input_policy.h"
#include "roles.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu::isa_diff {

struct NamedValue {
  uint32_t value = 0;
  std::string name; ///< "family.detail", e.g. "f32.qnan".
};

struct NamedValue64 {
  uint64_t value = 0;
  std::string name;
};

struct PairValue {
  uint32_t a = 0, b = 0;
  std::string name;
};

struct PairValue64 {
  uint64_t a = 0, b = 0;
  std::string name;
};

/// Generic 32-bit catalog (deduplicated by value; first name wins).
[[nodiscard]] const std::vector<NamedValue> &catalog32();
/// Generic 64-bit catalog.
[[nodiscard]] const std::vector<NamedValue64> &catalog64();
/// Extra values for the given role bits.
[[nodiscard]] std::vector<NamedValue> role_catalog(uint32_t roles);
/// Values around a constant the register is combined with.
[[nodiscard]] std::vector<NamedValue> const_relative_values(uint32_t constant);
/// Two-source pairs (carry/borrow/overflow/equal/complement/NaN + core cross product).
[[nodiscard]] const std::vector<PairValue> &pair_catalog();
[[nodiscard]] const std::vector<PairValue64> &pair_catalog64();

/// One sweep case: a base state plus overrides.
struct SweepCase {
  uint8_t base = 0; ///< 0 zeros, 1 ones, 2 seeded random.
  int8_t scc = -1;  ///< -1 keeps the base's SCC.
  std::vector<std::pair<unsigned, uint32_t>> overrides;
  std::string label;  ///< e.g. "single s3=shift.32 base=random scc=1".
  std::string family; ///< Catalog family, e.g. "f32", "shift", "pair.carry".
};

struct BoundarySweep {
  std::array<std::array<uint32_t, kNumLocs>, 3> bases{};
  std::vector<SweepCase> cases;
  size_t generated = 0;  ///< Before de-duplication and the cap.
  size_t duplicates = 0; ///< Identical input states dropped.
  bool truncated = false;

  [[nodiscard]] Sample materialize(const SweepCase &c) const;
};

struct SweepOptions {
  uint64_t seed = 1;
  size_t max_cases = 250000; ///< Safety cap; exceeding it is reported, never silent.
};

[[nodiscard]] BoundarySweep build_boundary_sweep(const RegionRoles &roles,
                                                 const RegionContract &contract,
                                                 const SweepOptions &options);

/// @brief Deterministically move a state onto the constraints (masks, clamping, equalities).
void clamp_to_constraints(std::array<uint32_t, kNumLocs> &values, const RegionContract &contract);

/// @brief The family part of a catalog name ("f32.qnan" -> "f32").
[[nodiscard]] std::string value_family(const std::string &name);

} // namespace rocjitsu::isa_diff
