// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::multiplexer_fuzz {

/// @brief Maximum multiplexer input length, including the selector byte.
inline constexpr std::size_t kMaxInputSize = (1u << 20) + 1;

/// @brief Stable selector-byte identities for the fuzz harnesses reached by rj_fuzz.
enum class FuzzTarget : uint8_t {
  Decode = 0,
  Execution = 1,
  CodeObject = 2,
  ConcurrentDispatch = 3,
};

/// @brief One registered multiplexer fuzz destination.
struct TargetDescriptor {
  FuzzTarget id;
  std::string_view name;
};

/// @brief Observable result of one multiplexer dispatch.
struct DispatchResult {
  FuzzTarget target;
  uint64_t work_units;
};

/// @brief Registered targets in stable selector order.
[[nodiscard]] std::span<const TargetDescriptor> targets();
/// @brief Select a target from an exact selector ID, or no target for an unassigned ID.
[[nodiscard]] std::optional<FuzzTarget> selected_target(uint8_t selector);
/// @brief Dispatch one input, excluding its selector from the destination payload.
[[nodiscard]] std::optional<DispatchResult> run_input(std::span<const uint8_t> input);
/// @brief Structured multiplexer seeds, with at least one for every registered target.
[[nodiscard]] std::vector<std::vector<uint8_t>> seed_corpus();

} // namespace rocjitsu::multiplexer_fuzz
