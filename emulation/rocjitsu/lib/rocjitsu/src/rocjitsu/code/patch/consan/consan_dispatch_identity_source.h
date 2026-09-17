// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <compare>
#include <cstdint>
#include <optional>

namespace rocjitsu::consan {

/// One already-authorized source for an inseparable 64-bit dispatch identity.
/// Register alternatives name the low word of a consecutive pair.
struct DispatchIdentity {
  std::optional<uint16_t> sgpr;
  std::optional<uint32_t> private_offset;
  std::optional<uint64_t> literal;

  [[nodiscard]] bool is_well_formed() const {
    return static_cast<uint8_t>(sgpr.has_value()) +
               static_cast<uint8_t>(private_offset.has_value()) +
               static_cast<uint8_t>(literal.has_value()) ==
           1u;
  }

  auto operator<=>(const DispatchIdentity &) const = default;
};

} // namespace rocjitsu::consan
