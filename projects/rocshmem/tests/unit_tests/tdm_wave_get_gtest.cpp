/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#include "tdm_wave_get_gtest.hpp"

using namespace rocshmem;

TEST_P(TdmWaveGetFixture, get_bytes) {
    auto [bytes_per_wave, num_waves] = GetParam();
    run_get(bytes_per_wave, num_waves);
}

static constexpr size_t TDM_TILE_BYTES = TEST_TDM_WAVE_GET_TILE_BYTES;
static constexpr size_t GET_THRESHOLD  = TEST_TDM_WAVE_GET_THRESHOLD;

INSTANTIATE_TEST_SUITE_P(
    TdmWaveGet,
    TdmWaveGetFixture,
    ::testing::Combine(
        ::testing::Values(
            // Immediately below the get threshold: must still fall back
            static_cast<size_t>(GET_THRESHOLD - 1),
            // Exactly at the get threshold (sub-tile, since GET_THRESHOLD <
            // TDM_TILE_BYTES): must take the TDM path with a single
            // partial tile
            static_cast<size_t>(GET_THRESHOLD),
            // At least one tile per wave even when evenly split several ways
            static_cast<size_t>(TDM_TILE_BYTES),
            // Several tiles per wave, each with distinct data: exercises
            // the copy_pipelined_deep loop without the tile-reordering
            // blind spot a constant fill pattern would have
            static_cast<size_t>(TDM_TILE_BYTES * 4),
            // Non-power-of-two: partial remainder handled by memcpy_wave
            static_cast<size_t>(TDM_TILE_BYTES * 3 + 1337)
        ),
        // 1 wave (degenerate: no splitting), a power-of-two split, and an
        // uneven split (reg.bytes doesn't divide evenly by 3).
        ::testing::Values(1, 4, 3)
    )
);
