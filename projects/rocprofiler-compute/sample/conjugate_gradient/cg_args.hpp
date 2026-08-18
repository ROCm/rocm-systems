// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

inline constexpr std::uint32_t cg_block_size = 256;

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
