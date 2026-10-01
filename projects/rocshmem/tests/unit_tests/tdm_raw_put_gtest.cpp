/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#include "tdm_raw_put_gtest.hpp"

using namespace rocshmem;

TEST_P(TdmRawPutFixture, put_bytes) {
    run_put(GetParam());
}

static constexpr size_t TDM_TILE_BYTES = TEST_TDM_TILE_BYTES;

INSTANTIATE_TEST_SUITE_P(
    TdmRawPut,
    TdmRawPutFixture,
    ::testing::Values(
        // Sub-tile: exercises memcpy_wg fallback path
        static_cast<size_t>(1024),
        // Exactly one TDM tile
        static_cast<size_t>(TDM_TILE_BYTES),
        // Multiple tiles: exercises the double-buffered loop
        static_cast<size_t>(TDM_TILE_BYTES * 4),
        // Non-power-of-two: partial remainder handled by memcpy_wg
        static_cast<size_t>(TDM_TILE_BYTES * 3 + 1337)
    )
);
