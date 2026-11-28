// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/boot/decoder.hpp"
#include "addc/common.hpp"

namespace addc::boot
{

int boot_error_type_to_afid(uint8_t error_type_id, uint8_t format_version,
                            uint8_t msg_0) noexcept
{
    if (msg_0 == 0xBAU)
    {
        return kAfidSentinel;
    }

    if (msg_0 != 0xA4U)
    {
        return 5;
    }

    if (format_version == 0U)
    {
        switch (error_type_id)
        {
            case 0x01:
                return 4;
            case 0x04:
                return 1;
            case 0x05:
                return 9;
            case 0x06:
                return 10;
            case 0x07:
                return 7;
            case 0x08:
                return 8;
            case 0x09:
                return 3;
            case 0x0A:
                return 2;
            case 0x0B:
                return 12;
            default:
                return 6;
        }
    }

    switch (error_type_id)
    {
        case 0x01:
            return 4;
        case 0x04:
            return 1;
        case 0x05:
            return 9;
        case 0x06:
            return 10;
        case 0x07:
            return 7;
        case 0x08:
            return 8;
        case 0x09:
            return 3;
        case 0x0A:
            return 2;
        case 0x0C:
            return 12;
        case 0x0D:
            return 11;
        default:
            return 6;
    }
}

} // namespace addc::boot
