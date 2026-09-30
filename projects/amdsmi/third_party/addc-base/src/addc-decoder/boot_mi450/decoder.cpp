// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/boot_mi450/decoder.hpp"

#include "addc/common.hpp"

namespace addc::boot_mi450
{

int decode_afid(uint64_t value) noexcept
{
    const auto code = static_cast<uint8_t>(value);
    const auto type = static_cast<int>((value >> 8U) & 0x1FU);

    switch (code)
    {
        case 0xA1U:
            return 46849 + type;
        case 0xB0U:
            return 47105 + type;
        case 0xBAU:
            return 47361 + type;
        case 0xBFU:
            return 47617 + type;
        default:
            return addc::kAfidSentinel;
    }
}

} // namespace addc::boot_mi450
