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
/// Lane-parallel VGPR inputs of one sweep case (VALU support), applied over the default
/// lane fill. Lane i of a chunk k takes element k*64+i of the source list.
struct VOverride {
  enum Kind : uint8_t {
    Chunk32, ///< VGPR a: the 32-bit catalog.
    Chunk64, ///< VGPRs a, a+1: the 64-bit catalog.
    Cross,   ///< VGPRs a and b: the 32-bit catalog cross product (64-bit when wide).
    Values,  ///< VGPR a: value list `list` of the sweep (literal-relative values).
  } kind = Chunk32;
  uint16_t a = 0, b = 0;
  bool wide = false;
  uint32_t chunk = 0;
  uint32_t list = 0;
};

struct SweepCase {
  uint8_t base = 0; ///< 0 zeros, 1 ones, 2 seeded random.
  int8_t scc = -1;  ///< -1 keeps the base's SCC.
  std::vector<std::pair<unsigned, uint32_t>> overrides;
  std::vector<VOverride> voverrides;
  int8_t exec = -1;   ///< -1 base EXEC, 0 all lanes, 1 alternate lanes.
  uint32_t fill = 0;  ///< Index of the default VGPR lane fill.
  std::string label;  ///< e.g. "single s3=shift.32 base=random scc=1".
  std::string family; ///< Catalog family, e.g. "f32", "shift", "pair.carry".
};

struct BoundarySweep {
  std::array<std::array<uint32_t, kNumLocs>, 3> bases{};
  unsigned vgpr_count = 0;
  unsigned lanes = 64; ///< Lanes of the wave: vector phases spread values over these.
  VSet vwide;
  uint64_t seed = 0;
  bool exec_entry = false;
  std::vector<uint32_t> vcat;                     ///< 32-bit catalog values for lanes.
  std::vector<std::vector<uint32_t>> value_lists; ///< VOverride::Values sources.
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
