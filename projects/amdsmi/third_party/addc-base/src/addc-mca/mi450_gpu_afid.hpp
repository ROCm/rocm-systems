// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/mca/decoder.hpp"
#include "addc/mca/registers.hpp"

#include <cstdint>
#include <optional>
#include <string_view>

namespace addc::mca::detail
{

inline std::optional<uint32_t> mi450_bank_base_afid(uint16_t hw_id,
                                                    uint16_t mca_type) noexcept
{
    switch (hw_id)
    {
        case 0x2e:
            switch (mca_type)
            {
                case 0x1:
                    return 40257u;
                case 0x2:
                    return 40001u;
                default:
                    return std::nullopt;
            }
        case 0x01:
            switch (mca_type)
            {
                case 0x1:
                    return 41025u;
                case 0x2:
                    return 46081u;
                default:
                    return std::nullopt;
            }
        case 0xff:
            switch (mca_type)
            {
                case 0x2:
                    return 41793u;
                default:
                    return std::nullopt;
            }
        case 0x77:
            return mca_type == 0x5 ? std::optional<uint32_t>{40513u}
                                   : std::nullopt;
        case 0x96:
            return mca_type == 0x0 ? std::optional<uint32_t>{40769u}
                                   : std::nullopt;
        case 0xfd:
            return mca_type == 0x0 ? std::optional<uint32_t>{41281u}
                                   : std::nullopt;
        case 0x12:
            return mca_type == 0x0 ? std::optional<uint32_t>{41537u}
                                   : std::nullopt;
        case 0x1fd:
            return mca_type == 0x0 ? std::optional<uint32_t>{42049u}
                                   : std::nullopt;
        case 0x18:
            return mca_type == 0x0 ? std::optional<uint32_t>{42305u}
                                   : std::nullopt;
        case 0x117:
            return mca_type == 0x0 ? std::optional<uint32_t>{42561u}
                                   : std::nullopt;
        case 0x46:
            return mca_type == 0x1 ? std::optional<uint32_t>{42817u}
                                   : std::nullopt;
        case 0x259:
            return mca_type == 0x0 ? std::optional<uint32_t>{43073u}
                                   : std::nullopt;
        case 0x50:
            return mca_type == 0x0 ? std::optional<uint32_t>{43329u}
                                   : std::nullopt;
        case 0x5c:
            return mca_type == 0x0 ? std::optional<uint32_t>{46273u}
                                   : std::nullopt;
        case 0x6c:
            return mca_type == 0x0 ? std::optional<uint32_t>{43585u}
                                   : std::nullopt;
        case 0x80:
            return mca_type == 0x0 ? std::optional<uint32_t>{43842u}
                                   : std::nullopt;
        case 0xdd:
            return mca_type == 0x0 ? std::optional<uint32_t>{46465u}
                                   : std::nullopt;
        case 0x70:
            return mca_type == 0x0 ? std::optional<uint32_t>{44097u}
                                   : std::nullopt;
        case 0x73:
            return mca_type == 0x0 ? std::optional<uint32_t>{44353u}
                                   : std::nullopt;
        case 0x27:
            return mca_type == 0x0 ? std::optional<uint32_t>{44613u}
                                   : std::nullopt;
        case 0x1e1:
            return mca_type == 0x0 ? std::optional<uint32_t>{44865u}
                                   : std::nullopt;
        case 0x40:
            return mca_type == 0x0 ? std::optional<uint32_t>{45121u}
                                   : std::nullopt;
        case 0x346:
            return mca_type == 0x0 ? std::optional<uint32_t>{45377u}
                                   : std::nullopt;
        case 0x164:
            return mca_type == 0x0 ? std::optional<uint32_t>{45633u}
                                   : std::nullopt;
        case 0x160:
            return mca_type == 0x0 ? std::optional<uint32_t>{45889u}
                                   : std::nullopt;
        case 0x16c:
            return mca_type == 0x0 ? std::optional<uint32_t>{46657u}
                                   : std::nullopt;
        default:
            return std::nullopt;
    }
}

inline uint32_t mi450_gpu_decode_afid(McaRegisters regs) noexcept
{
    const auto base =
        mi450_bank_base_afid(regs.ipid.hardware_id, regs.ipid.mca_type);
    if (!base)
        return kAfidSentinel;

    const uint8_t ext = regs.status.error_code_ext;
    if (ext > 63)
        return kAfidSentinel;

    const auto& s = regs.status;

    uint32_t sev_offset;
    if (s.poison)
        sev_offset = 64u;
    else if (s.pcc)
        sev_offset = 128u;
    else if (!s.pcc && s.uc && s.tcc)
        sev_offset = 128u;
    else if (!s.pcc && s.uc && !s.tcc)
        sev_offset = 64u;
    else if (!s.pcc && !s.uc && !s.tcc && s.deferred)
        sev_offset = 64u;
    else if (!s.pcc && !s.uc && !s.tcc && !s.deferred)
        sev_offset = 0u;
    else
        return kAfidSentinel;

    return *base + sev_offset + ext;
}

} // namespace addc::mca::detail
