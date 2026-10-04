// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace addc::cper
{

[[nodiscard]] inline uint16_t read_u16_le(std::span<const uint8_t> data,
                                          std::size_t offset) noexcept
{
    if (offset + 2u > data.size())
        return 0u;
    return static_cast<uint16_t>(data[offset]) |
           (static_cast<uint16_t>(data[offset + 1u]) << 8u);
}

[[nodiscard]] inline uint32_t read_u32_le(std::span<const uint8_t> data,
                                          std::size_t offset) noexcept
{
    if (offset + 4u > data.size())
        return 0u;
    return static_cast<uint32_t>(data[offset]) |
           (static_cast<uint32_t>(data[offset + 1u]) << 8u) |
           (static_cast<uint32_t>(data[offset + 2u]) << 16u) |
           (static_cast<uint32_t>(data[offset + 3u]) << 24u);
}

[[nodiscard]] inline uint64_t read_u64_le(std::span<const uint8_t> data,
                                          std::size_t offset) noexcept
{
    if (offset + 8u > data.size())
        return 0u;
    return static_cast<uint64_t>(data[offset]) |
           (static_cast<uint64_t>(data[offset + 1u]) << 8u) |
           (static_cast<uint64_t>(data[offset + 2u]) << 16u) |
           (static_cast<uint64_t>(data[offset + 3u]) << 24u) |
           (static_cast<uint64_t>(data[offset + 4u]) << 32u) |
           (static_cast<uint64_t>(data[offset + 5u]) << 40u) |
           (static_cast<uint64_t>(data[offset + 6u]) << 48u) |
           (static_cast<uint64_t>(data[offset + 7u]) << 56u);
}

} // namespace addc::cper
