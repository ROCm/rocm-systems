// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file scaling_thread_counts.h
/// @brief Argument parsing for the scaling benchmark's thread-count list.
///
/// Split out of scaling_test.cpp so it can be unit-tested: the benchmark itself
/// needs a GPU-sized simulation and a device-kernel build, and its main() is
/// compiled away entirely without HAS_DEVICE_KERNELS.

#include <charconv>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>
#include <vector>

namespace rocjitsu::test {

/// @brief Parse the benchmark's thread-count arguments.
///
/// @details Each argument must be a whole decimal number in [1, @p max_threads]
/// -- no sign, no trailing text, no value that would narrow into a nonsense
/// thread count. An empty argument list means "sweep 1..max_threads", which is
/// the benchmark's default; a malformed one is an error rather than a silent
/// fallback to that sweep, because a typo would otherwise look like a
/// deliberate full run.
///
/// @param args Arguments after argv[0].
/// @param max_threads Largest accepted thread count; also the end of the sweep.
/// @returns The requested counts, or nullopt if any argument is rejected.
inline std::optional<std::vector<uint32_t>> parse_thread_counts(std::span<const char *const> args,
                                                                uint32_t max_threads) {
  std::vector<uint32_t> counts;
  for (const char *arg : args) {
    const std::string_view text(arg);
    uint32_t value = 0;
    const auto *end = text.data() + text.size();
    const auto [stop, ec] = std::from_chars(text.data(), end, value);
    // from_chars rejects a leading '+'/'-' for an unsigned type, so this also
    // covers the negatives that strtoul would have wrapped into huge counts.
    if (ec != std::errc{} || stop != end)
      return std::nullopt;
    if (value < 1 || value > max_threads)
      return std::nullopt;
    counts.push_back(value);
  }

  if (counts.empty())
    for (uint32_t t = 1; t <= max_threads; ++t)
      counts.push_back(t);
  return counts;
}

} // namespace rocjitsu::test
