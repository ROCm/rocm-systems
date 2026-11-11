// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/common.hpp"

#include <cstdint>

namespace addc::boot
{

struct OamBootMsg
{
    uint8_t msg_0{};
    uint8_t msg_1{};
    uint8_t msg_2{};
    uint8_t msg_3{};
    uint8_t msg_4{};
    uint8_t msg_5{};
    uint8_t msg_6{};
    uint8_t msg_7{};
    uint8_t error_type_id{};
    uint8_t format_version{};

    constexpr explicit OamBootMsg(uint64_t raw) noexcept :
        msg_0{static_cast<uint8_t>((raw >> 0u) & 0xFFu)},
        msg_1{static_cast<uint8_t>((raw >> 8u) & 0xFFu)},
        msg_2{static_cast<uint8_t>((raw >> 16u) & 0xFFu)},
        msg_3{static_cast<uint8_t>((raw >> 24u) & 0xFFu)},
        msg_4{static_cast<uint8_t>((raw >> 32u) & 0xFFu)},
        msg_5{static_cast<uint8_t>((raw >> 40u) & 0xFFu)},
        msg_6{static_cast<uint8_t>((raw >> 48u) & 0xFFu)},
        msg_7{static_cast<uint8_t>((raw >> 56u) & 0xFFu)},
        error_type_id{static_cast<uint8_t>(msg_1 & 0x1Fu)},
        format_version{static_cast<uint8_t>((msg_1 >> 5u) & 0x7u)}
    {}
};

/// Map a boot message's error_type_id (byte1[4:0]) and format_version
/// to an AFID number. Returns kUnclassifiedAfid for unrecognised types.
[[nodiscard]] int boot_error_type_to_afid(
    uint8_t error_type_id, uint8_t format_version, uint8_t msg_0) noexcept;

} // namespace addc::boot
