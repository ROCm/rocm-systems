// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu::code_object_fuzz {

/// @brief Maximum byte length accepted by the loader fuzz entry point.
inline constexpr std::size_t kMaxInputSize = 1u << 20;

/// @brief Observable parsing stages reached while processing fuzz inputs.
/// @details A standalone device ELF is inspected both directly and through
/// Executable, so counters describe exercised loader paths rather than unique
/// objects in the input.
struct RunStats {
  uint64_t inputs = 0;
  uint64_t code_objects_valid = 0;
  uint64_t executables_valid = 0;
  uint64_t bundled_code_objects = 0;
  uint64_t sections = 0;
  uint64_t text_sections = 0;
  uint64_t descriptors = 0;
  uint64_t load_segments = 0;
};

/// @brief Accumulate per-stage counters into a campaign total.
RunStats &operator+=(RunStats &left, const RunStats &right);

/// @brief Whether a reported load segment lies entirely within @p image.
[[nodiscard]] bool load_segment_in_bounds(std::span<const uint8_t> image,
                                          std::uintptr_t data_address, std::size_t size);

/// @brief Exercise code-object and executable loading for one byte sequence.
[[nodiscard]] RunStats run_input(std::span<const uint8_t> bytes);
/// @brief Return the deterministic structured corpus used by the generator.
[[nodiscard]] std::vector<std::vector<uint8_t>> seed_corpus();
/// @brief Generate one bounded input deterministically from @p seed.
[[nodiscard]] std::vector<uint8_t> generate_input(uint64_t seed);

} // namespace rocjitsu::code_object_fuzz
