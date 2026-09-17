// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>

namespace rocjitsu::consan {

/// The default race detector or the explicitly selected SuperCollider detector.
/// None disables instrumentation; an unspecified request remains a separate optional state.
enum class Mode : uint8_t { None, Default, SuperCollider };

[[nodiscard]] constexpr std::optional<Mode> enabled_mode(Mode mode) {
  return mode == Mode::Default || mode == Mode::SuperCollider ? std::optional(mode) : std::nullopt;
}

} // namespace rocjitsu::consan
