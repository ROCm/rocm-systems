// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>

namespace rocjitsu::consan_hook {

/// Tracks byte growth charged by a caller-provided lifetime registry.
///
/// This value type is deliberately not internally synchronized. A registry
/// must plan, retain its corresponding resource, and commit while holding one
/// lock so admission and ownership form a single transaction.
class ProcessGrowthBudget {
public:
  enum class ChargeOutcome : uint8_t {
    WithinLimit,
    LimitExceeded,
    AccountingOverflow,
  };

  struct ChargePlan {
    ChargeOutcome outcome = ChargeOutcome::AccountingOverflow;
    uint64_t live_bytes = 0;
    uint64_t growth_bytes = 0;
    std::optional<uint64_t> required_bytes;
    std::optional<uint64_t> limit_bytes;

    [[nodiscard]] explicit operator bool() const { return outcome == ChargeOutcome::WithinLimit; }
  };

  struct Summary {
    uint64_t live_bytes = 0;
    uint64_t peak_bytes = 0;
  };

  [[nodiscard]] ChargePlan plan_charge(uint64_t growth_bytes,
                                       std::optional<uint64_t> limit_bytes) const {
    const std::optional<uint64_t> required_bytes =
        growth_bytes <= std::numeric_limits<uint64_t>::max() - live_bytes_
            ? std::optional<uint64_t>(live_bytes_ + growth_bytes)
            : std::nullopt;
    ChargeOutcome outcome = ChargeOutcome::WithinLimit;
    if (!required_bytes) {
      outcome = ChargeOutcome::AccountingOverflow;
    } else if (limit_bytes && *required_bytes > *limit_bytes) {
      outcome = ChargeOutcome::LimitExceeded;
    }
    return {
        .outcome = outcome,
        .live_bytes = live_bytes_,
        .growth_bytes = growth_bytes,
        .required_bytes = required_bytes,
        .limit_bytes = limit_bytes,
    };
  }

  void commit_charge(const ChargePlan &plan) {
    assert(plan.outcome == ChargeOutcome::WithinLimit && plan.required_bytes &&
           plan.live_bytes == live_bytes_ && "committed growth plan must still be current");
    live_bytes_ = *plan.required_bytes;
    peak_bytes_ = std::max(peak_bytes_, live_bytes_);
  }

  /// Returns false and resets to a permissive zero state on invariant failure.
  [[nodiscard]] bool refund(uint64_t growth_bytes) {
    if (growth_bytes > live_bytes_) {
      live_bytes_ = 0;
      return false;
    }
    live_bytes_ -= growth_bytes;
    return true;
  }

  [[nodiscard]] Summary summary() const {
    return {
        .live_bytes = live_bytes_,
        .peak_bytes = peak_bytes_,
    };
  }

  void clear() {
    live_bytes_ = 0;
    peak_bytes_ = 0;
  }

private:
  uint64_t live_bytes_ = 0;
  uint64_t peak_bytes_ = 0;
};

} // namespace rocjitsu::consan_hook
