/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#include "tdm_raw_get_gtest.hpp"

using namespace rocshmem;

TEST_P(TdmRawGetFixture, get_bytes) {
    run_get(GetParam());
}

static constexpr size_t TDM_TILE_BYTES = TEST_TDM_GET_TILE_BYTES;
static constexpr size_t GET_THRESHOLD  = TEST_TDM_GET_THRESHOLD;

INSTANTIATE_TEST_SUITE_P(
    TdmRawGet,
    TdmRawGetFixture,
    ::testing::Values(
        // Immediately below the get threshold: must still fall back
        static_cast<size_t>(GET_THRESHOLD - 1),
        // Exactly at the get threshold (sub-tile, since GET_THRESHOLD <
        // TDM_TILE_BYTES): must take the TDM path with a single partial tile
        static_cast<size_t>(GET_THRESHOLD),
        // Exactly one TDM tile
        static_cast<size_t>(TDM_TILE_BYTES),
        // Multiple tiles: exercises the double-buffered loop
        static_cast<size_t>(TDM_TILE_BYTES * 4),
        // Non-power-of-two: partial remainder handled by memcpy_wg
        static_cast<size_t>(TDM_TILE_BYTES * 3 + 1337)
    )
);
