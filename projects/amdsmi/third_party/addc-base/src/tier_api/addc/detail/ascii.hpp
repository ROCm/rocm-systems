// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace addc::detail
{

/// Read a bounded, NUL-terminated firmware ASCII field. Firmware may leave
/// poison or other binary fill patterns in fields marked valid, so truncate at
/// the first byte outside printable ASCII rather than creating invalid JSON.
[[nodiscard]] inline std::string read_ascii_field(
    std::span<const uint8_t> bytes, std::size_t offset,
    std::size_t max_length)
{
    std::string result;
    result.reserve(max_length);
    for (std::size_t index = 0U;
         index < max_length && offset + index < bytes.size(); ++index)
    {
        const uint8_t byte = bytes[offset + index];
        if (byte == 0U || byte < 0x20U || byte > 0x7eU)
        {
            break;
        }
        result.push_back(static_cast<char>(byte));
    }
    return result;
}

} // namespace addc::detail
