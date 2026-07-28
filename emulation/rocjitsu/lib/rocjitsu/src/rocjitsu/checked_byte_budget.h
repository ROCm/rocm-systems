// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_CHECKED_BYTE_BUDGET_H_
#define ROCJITSU_CHECKED_BYTE_BUDGET_H_

#include "util/bit.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace rocjitsu::byte_accounting {

/// Add `count * element_bytes` to an existing byte charge when representable.
[[nodiscard]] inline constexpr std::optional<uint64_t>
checked_allocation_charge(uint64_t accumulated, uint64_t count, uint64_t element_bytes) {
  const auto bytes = util::checked_mul(count, element_bytes);
  if (!bytes)
    return std::nullopt;
  return util::checked_add(accumulated, *bytes);
}

/// Charge only capacity beyond a caller's requested element count.
[[nodiscard]] inline constexpr std::optional<uint64_t>
excess_capacity_charge(uint64_t capacity, uint64_t requested, uint64_t element_bytes) {
  if (capacity < requested)
    return std::nullopt;
  return checked_allocation_charge(0, capacity - requested, element_bytes);
}

/// Multiply byte counts, saturating at UINT64_MAX instead of overflowing.
[[nodiscard]] inline constexpr uint64_t saturating_multiply(uint64_t lhs, uint64_t rhs) {
  return util::checked_mul(lhs, rhs).value_or(std::numeric_limits<uint64_t>::max());
}

enum class ChargeOutcome : uint8_t {
  WithinLimit,
  LimitExceeded,
  AccountingOverflow,
};

struct ChargeResult {
  ChargeOutcome outcome = ChargeOutcome::AccountingOverflow;
  uint64_t previous_used_bytes = 0;
  std::optional<uint64_t> required_bytes;
  uint64_t limit_bytes = 0;

  [[nodiscard]] constexpr explicit operator bool() const {
    return outcome == ChargeOutcome::WithinLimit;
  }
};

[[nodiscard]] inline constexpr ChargeResult
plan_charge(uint64_t used_bytes, uint64_t count, uint64_t element_bytes, uint64_t limit_bytes) {
  const auto required_bytes = checked_allocation_charge(used_bytes, count, element_bytes);
  const ChargeOutcome outcome = !required_bytes                 ? ChargeOutcome::AccountingOverflow
                                : *required_bytes > limit_bytes ? ChargeOutcome::LimitExceeded
                                                                : ChargeOutcome::WithinLimit;
  return {
      .outcome = outcome,
      .previous_used_bytes = used_bytes,
      .required_bytes = required_bytes,
      .limit_bytes = limit_bytes,
  };
}

/// A fixed-limit byte ledger whose rejected charges never mutate used bytes.
class CheckedByteBudget {
public:
  explicit constexpr CheckedByteBudget(uint64_t limit_bytes) : limit_bytes_(limit_bytes) {}

  [[nodiscard]] constexpr ChargeResult charge(uint64_t bytes) {
    return charge_allocation(1, bytes);
  }

  [[nodiscard]] constexpr ChargeResult charge_allocation(uint64_t count, uint64_t element_bytes) {
    const ChargeResult result = plan_charge(used_bytes_, count, element_bytes, limit_bytes_);
    if (result)
      used_bytes_ = *result.required_bytes;
    return result;
  }

  [[nodiscard]] constexpr uint64_t used_bytes() const { return used_bytes_; }

private:
  uint64_t limit_bytes_ = 0;
  uint64_t used_bytes_ = 0;
};

static_assert(checked_allocation_charge(0, 2, std::numeric_limits<uint64_t>::max() / 2) ==
              std::numeric_limits<uint64_t>::max() - 1);
static_assert(checked_allocation_charge(1, 1, std::numeric_limits<uint64_t>::max() - 1) ==
              std::numeric_limits<uint64_t>::max());
static_assert(checked_allocation_charge(std::numeric_limits<uint64_t>::max(), 0, 8) ==
              std::numeric_limits<uint64_t>::max());
static_assert(excess_capacity_charge(0, 0, 8) == 0);

} // namespace rocjitsu::byte_accounting

#endif // ROCJITSU_CHECKED_BYTE_BUDGET_H_
