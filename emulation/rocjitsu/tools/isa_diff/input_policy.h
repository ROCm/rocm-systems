// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file input_policy.h
/// @brief Pluggable input generation and constraint checking.
///
/// @details This is the seam for later learned input policies. A policy proposes values
/// for the contract's entry_equal locations and may observe feedback. It can never edit
/// the programs, the mapping, the observable set or the constraints; those belong to the
/// contract. P0 ships only BaselinePolicy, which is deterministic in (seed, case index)
/// and draws:
///   - boundary values: 0, 1, all-ones, signed and unsigned limits, single bits and
///     their inverses, alternating patterns, shift amounts around 31/32/63;
///   - 64-bit boundaries across the two locations of pair operands;
///   - correlated operands: equal or complementary copies of another input;
///   - uniform random values;
///   - SCC alternating with the case parity, plus random flips.
#pragma once

#include "contract.h"
#include "location.h"

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
};

struct RegionFeatures {
  std::vector<std::string> old_mnemonics;
  std::vector<std::string> new_mnemonics;
};

struct Feedback {
  uint64_t case_index = 0;
  bool accepted = false;
  bool mismatch = false;
  std::string category; ///< Bounded classification (mismatch kind or rejection reason).
};

class InputPolicy {
public:
  virtual ~InputPolicy() = default;
  [[nodiscard]] virtual std::string name() const = 0;
  virtual Sample propose(const RegionFeatures &features, const RegionContract &contract,
                         uint64_t case_index) = 0;
  virtual void observe(const Feedback &) {}
};

class BaselinePolicy final : public InputPolicy {
public:
  explicit BaselinePolicy(uint64_t seed) : seed_(seed) {}
  [[nodiscard]] std::string name() const override { return "baseline-v1"; }
  Sample propose(const RegionFeatures &features, const RegionContract &contract,
                 uint64_t case_index) override;

private:
  uint64_t seed_;
};

/// @brief Validate a sample against the contract's constraints; returns the first violation.
[[nodiscard]] std::optional<std::string> validate_sample(const Sample &sample,
                                                         const RegionContract &contract);

/// @brief Nudge a sample toward its constraints (masks, ranges, equalities) before validation.
void project_onto_constraints(Sample &sample, const RegionContract &contract, uint64_t entropy);

/// @brief Deterministic per-side canary value for a location dead at entry.
[[nodiscard]] uint32_t canary_value(uint64_t seed, uint64_t case_index, unsigned loc,
                                    bool new_side);

/// @brief SplitMix64 step (deterministic, portable).
[[nodiscard]] uint64_t splitmix64(uint64_t x);

} // namespace rocjitsu::isa_diff
