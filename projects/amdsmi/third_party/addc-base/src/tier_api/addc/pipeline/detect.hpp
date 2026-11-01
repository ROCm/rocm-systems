// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace addc::pipeline
{

/// Detect the platform name from a CPER section descriptor revision major byte.
///
/// The revision `major` encodes program_rev (lo nibble) and gen_rev (hi
/// nibble).
///   major=0x31 -> program_rev=1 (EPYC), gen_rev=3 -> "venice"
///   major=0x21 -> program_rev=1 (EPYC), gen_rev=2 -> "turin"
///   major=0x12 -> program_rev=2 (GPU),  gen_rev=1 -> "mi300"
///
/// Returns the platform name, or std::nullopt if unrecognized.
[[nodiscard]] std::optional<std::string> detect_platform(
    uint32_t revision_major);

/// Return a human-readable display name for a platform string.
/// Falls back to "AMD EPYC" for unknown platforms.
[[nodiscard]] std::string_view platform_display_name(
    std::string_view platform) noexcept;

} // namespace addc::pipeline
