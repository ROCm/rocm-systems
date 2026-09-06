// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

inline constexpr std::uint32_t cg_block_size = 256;
inline constexpr std::uint32_t cg_rows       = 65536;

struct CgArgs
{
    const std::uint32_t* row_offsets;
    const std::uint32_t* column_indices;
    const float*         values;
    const float*         p;
    float*               q;
    float*               x;
    float*               r;
    float*               partials;
    std::uint32_t        rows;
    std::uint32_t        rounds;
};

// Every kernel library exports this one symbol, so the driver resolves the same
// name against each handle instead of tracking a symbol name per library.
using CgLaunch = void (*)(const CgArgs&);
