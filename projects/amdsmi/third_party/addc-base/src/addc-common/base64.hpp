// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace addc::detail
{

/// Encode bytes using canonical RFC 4648 base64 with '=' padding.
[[nodiscard]] std::string base64_encode(std::span<const uint8_t> data);

/// Decode canonical RFC 4648 base64.
///
/// The input length must be a multiple of four, padding may appear only in the
/// final quartet, and unused padding bits must be zero. Invalid input returns
/// std::nullopt. Allocation failures are intentionally allowed to propagate to
/// the supported API boundary.
[[nodiscard]] std::optional<std::vector<uint8_t>> base64_decode(
    std::string_view text);

} // namespace addc::detail
