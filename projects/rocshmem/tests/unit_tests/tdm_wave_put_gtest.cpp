/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#include "tdm_wave_put_gtest.hpp"

using namespace rocshmem;

TEST_P(TdmWavePutFixture, put_bytes) {
    auto [bytes_per_wave, num_waves] = GetParam();
    run_put(bytes_per_wave, num_waves);
}

static constexpr size_t TDM_TILE_BYTES = TEST_TDM_TILE_BYTES;

INSTANTIATE_TEST_SUITE_P(
    TdmWavePut,
    TdmWavePutFixture,
    ::testing::Combine(
        ::testing::Values(
            // Sub-tile regardless of split: exercises memcpy_wave fallback
            static_cast<size_t>(1024),
            // At least one tile per wave even when evenly split several ways
            static_cast<size_t>(TDM_TILE_BYTES),
            // Several tiles per wave: exercises the double-buffered loop
            static_cast<size_t>(TDM_TILE_BYTES * 4),
            // Non-power-of-two: partial remainder handled by memcpy_wave
            static_cast<size_t>(TDM_TILE_BYTES * 3 + 1337)
        ),
        // 1 wave (degenerate: no splitting), a power-of-two split, and an
        // uneven split (reg.bytes doesn't divide evenly by 3).
        ::testing::Values(1, 4, 3)
    )
);
