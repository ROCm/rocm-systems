// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file input_policy.h
/// @brief Pluggable input generation and constraint checking.
///
/// @details This is the seam for later learned input policies. A policy proposes values
/// for the contract's entry_equal locations and may observe feedback. It can never edit
/// the programs, the mapping, the observable set or the constraints; those belong to the
/// contract. The mandatory boundary sweep (boundary.h) always runs first. The random phase
/// then uses BaselinePolicy, which is deterministic in (seed, case index) and draws:
///   - catalog values (integer limits, f32/f16/bf16 specials, bit patterns, ...);
///   - role values for the register's role (shift amounts, S_BFE fields, ...);
///   - 64-bit catalog values across the two locations of pair operands;
///   - correlated operands: equal or complementary copies of another input;
///   - uniform random values;
///   - SCC alternating with the case parity, plus random flips.
#pragma once

#include "contract.h"
#include "location.h"
#include "roles.h"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace rocjitsu::isa_diff {

struct Sample {
  std::array<uint32_t, kNumLocs> value{}; ///< Only entry_equal locations are meaningful.
  std::vector<std::string> events;        ///< Boundary events used (logged for later learning).
  /// VGPR lanes (VALU support): vgpr_count * kWaveLanes values, register-major. Only
  /// vgpr entry_equal registers are meaningful.
  std::vector<uint32_t> vvalue;
};

/// @brief Deterministic per-lane VGPR values for one case (VALU support). Each lane draws
/// independently from the 32-bit catalog (NaN/Inf/denormal encodings included), the
/// 64-bit catalog for 64-bit sources, single bits, copies of another register's lane,
/// and uniform random values; some registers get one value on every lane. One case
/// therefore exercises 64 different inputs per VGPR.
void fill_vgpr_lanes(Sample &s, unsigned vgpr_count, const VSet &wide, uint64_t seed,
                     uint64_t index);

struct RegionFeatures {
  std::vector<std::string> old_text; ///< Disassembly of the old side.
  std::vector<std::string> new_text;
  RegionRoles roles;
  /// Role-specific values per location (shift amounts, S_BFE fields, sign-extension
  /// edges, values around combined constants), precomputed once per region.
  std::array<std::vector<uint32_t>, kNumLocs> role_values;
};

/// @brief Decode the region once and derive its features (roles, role values).
[[nodiscard]] RegionFeatures make_region_features(
    const std::vector<RegionInst> &old_insts, const std::vector<RegionInst> &new_insts,
    const std::array<unsigned, kNumLocs> *new_to_old = nullptr, const Target &target = {});

/// Raw feedback for one case (handoff 8: InputPolicy.observe(raw_feedback)).
struct Feedback {
  uint64_t case_index = 0;
  std::string phase; ///< "boundary" or "random".
  bool accepted = false;
  std::string rejection; ///< Violated constraint when not accepted.
  std::string exit_old, exit_new;
  bool mismatch = false;
  /// Bounded classification: "ok", a mismatch kind, "excluded" or "rejected".
  std::string category;
  LocSet locs; ///< Differing observable locations.
  VSet vlocs;  ///< Differing observable VGPRs.
  /// Bounded novelty class: outcome plus, per written location, a coarse value class
  /// (zero/one/all-ones/sign/other) and old==new. Raw register values never form a class.
  uint64_t bucket = 0;
  bool novel = false; ///< First case of this region in its bucket.
  uint64_t propose_ns = 0, load_ns = 0, old_ns = 0, new_ns = 0, compare_ns = 0;
  uint64_t insts_old = 0, insts_new = 0;
  std::vector<std::string> events; ///< Input events of the sample.
};

/// Bounded history passed to propose() (handoff 8: propose(features, contract, history)).
struct History {
  static constexpr size_t kWindow = 32;
  uint64_t cases = 0, accepted = 0, valid = 0, mismatches = 0, novel = 0;
  std::map<std::string, uint64_t> categories;
  std::vector<Feedback> recent; ///< The last kWindow cases, oldest first.
  void record(const Feedback &f);
};

class InputPolicy {
public:
  virtual ~InputPolicy() = default;
  [[nodiscard]] virtual std::string name() const = 0;
  /// Propose values for the contract's entry_equal locations. A policy cannot change the
  /// programs, the mapping, the observable set or the constraints; every proposal is
  /// validated by ConstraintChecker before it runs.
  virtual Sample propose(const RegionFeatures &features, const RegionContract &contract,
                         const History &history, uint64_t case_index) = 0;
  virtual void observe(const Feedback &) {}
};

class BaselinePolicy final : public InputPolicy {
public:
  explicit BaselinePolicy(uint64_t seed) : seed_(seed) {}
  [[nodiscard]] std::string name() const override { return "baseline-v1"; }
  Sample propose(const RegionFeatures &features, const RegionContract &contract,
                 const History &history, uint64_t case_index) override;

private:
  uint64_t seed_;
};

/// Handoff 8: ConstraintChecker.validate(sample, contract) -> accepted/rejected.
class ConstraintChecker {
public:
  /// Returns the first violated constraint, or nullopt when the sample is accepted.
  [[nodiscard]] static std::optional<std::string> validate(const Sample &sample,
                                                           const RegionContract &contract);
};

/// @brief Validate a sample against the contract's constraints; returns the first violation.
[[nodiscard]] std::optional<std::string> validate_sample(const Sample &sample,
                                                         const RegionContract &contract);

/// @brief Nudge a sample toward its constraints (masks, ranges, equalities) before validation.
void project_onto_constraints(Sample &sample, const RegionContract &contract, uint64_t entropy);

/// @brief Deterministic per-side canary value for a location dead at entry.
[[nodiscard]] uint32_t canary_value(uint64_t seed, uint64_t case_index, unsigned loc,
                                    bool new_side);

/// @brief Initial memory content (memory model): the 32-bit word at byte address
/// 4 * @p word for memory seed @p seed. Half of the words are 32-bit catalog values
/// (NaN/Inf/denormal encodings, integer limits), the rest uniform random. Memory is an
/// input of the region: both sides start from the same content.
[[nodiscard]] uint32_t initial_memory_word(uint64_t seed, uint64_t word);
[[nodiscard]] inline uint8_t initial_memory_byte(uint64_t seed, uint64_t addr) {
  return static_cast<uint8_t>(initial_memory_word(seed, addr >> 2) >> ((addr & 3) * 8));
}

/// @brief SplitMix64 step (deterministic, portable).
[[nodiscard]] uint64_t splitmix64(uint64_t x);

} // namespace rocjitsu::isa_diff
