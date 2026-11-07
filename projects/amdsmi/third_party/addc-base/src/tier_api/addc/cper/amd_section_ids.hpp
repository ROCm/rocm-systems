// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"

#include <cstdint>

namespace addc::cper
{

/// AMD section descriptor revision, major byte: bits 3:0 are the program
/// (1 EPYC, 2 Instinct, ...) and bits 7:4 are the RAS generation.
[[nodiscard]] constexpr uint8_t program_gen(uint8_t revision_major) noexcept
{
    return revision_major & 0xFU;
}

[[nodiscard]] constexpr uint8_t ras_gen(uint8_t revision_major) noexcept
{
    return (revision_major >> 4U) & 0xFU;
}

/// AMD Crashdump section. EPYC, MI300 and MI450 use this GUID; the descriptor
/// revision selects the format.
inline constexpr Guid kAmdCrashdumpGuid{
    0x32ac0c78U,
    0x2623U,
    0x48f6U,
    {0xb0U, 0xd0U, 0x73U, 0x65U, 0x72U, 0x5fU, 0xd6U, 0xaeU}};

/// AMD GPU Non-Standard (runtime) error section.
inline constexpr Guid kAmdGpuRuntimeGuid{
    0x32ac0c78U,
    0x2623U,
    0x48f6U,
    {0x81U, 0xa2U, 0xacU, 0x69U, 0x17U, 0x80U, 0x55U, 0x1dU}};

} // namespace addc::cper
