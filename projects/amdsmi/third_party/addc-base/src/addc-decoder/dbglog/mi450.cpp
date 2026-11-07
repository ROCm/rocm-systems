// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/decoder/dbglog/mi450.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace addc::decoder
{

namespace
{

[[nodiscard]] uint64_t readU64Le(const uint8_t* data,
                                 std::size_t offset) noexcept
{
    uint64_t value = 0U;
    for (unsigned i = 0U; i < 8U; ++i)
    {
        value |= static_cast<uint64_t>(data[offset + i]) << (i * 8U);
    }
    return value;
}

[[nodiscard]] addc::pipeline::McaBankDump parseMca(const uint8_t* data)
{
    return addc::pipeline::McaBankDump{
        .ctl = readU64Le(data, 0x00U),
        .status = readU64Le(data, 0x08U),
        .addr = readU64Le(data, 0x10U),
        .misc0 = readU64Le(data, 0x18U),
        .config = readU64Le(data, 0x20U),
        .ipid = readU64Le(data, 0x28U),
        .synd = readU64Le(data, 0x30U),
        .destat = readU64Le(data, 0x38U),
        .deaddr = readU64Le(data, 0x40U),
        .misc1 = readU64Le(data, 0x48U),
        .synd1 = readU64Le(data, 0x50U),
        .synd2 = readU64Le(data, 0x58U),
        .ctl_mask = readU64Le(data, 0x60U),
        .transsynd = readU64Le(data, 0x68U),
        .transaddr = readU64Le(data, 0x70U),
        .transstat = readU64Le(data, 0x78U),
    };
}

} // namespace

std::optional<addc::pipeline::McaBankDump> parse_mi450_register_array(
    std::span<const uint8_t> bytes)
{
    if (bytes.size() < 128U)
    {
        return std::nullopt;
    }
    return parseMca(bytes.data());
}

std::vector<DecodedDbgLog> decode_mi450_dbglog(
    const std::vector<addc::pipeline::DbgLogGroup>& groups)
{
    auto decoded = decode_epyc_dbglog(groups);
    for (std::size_t index = 0U; index < decoded.size(); ++index)
    {
        const auto& group = groups[index];
        if (group.block_id != 0x20U || group.instances.empty() ||
            group.block_size < 128U)
        {
            continue;
        }
        const auto& bytes = group.instances.front().data;
        if (bytes.size() != 128U)
        {
            continue;
        }
        decoded[index].mca_bank = parse_mi450_register_array(bytes);
    }
    return decoded;
}

} // namespace addc::decoder
