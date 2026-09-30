// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

namespace addc
{

enum class ParseMode : uint8_t
{
    Tolerant,
    Strict,
};

struct DecodeOptions
{
    ParseMode parse_mode = ParseMode::Tolerant;
};

} // namespace addc
