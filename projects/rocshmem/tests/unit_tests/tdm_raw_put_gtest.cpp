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
static constexpr size_t PUT_THRESHOLD  = TEST_TDM_PUT_THRESHOLD;

INSTANTIATE_TEST_SUITE_P(
    TdmRawPut,
    TdmRawPutFixture,
    ::testing::Values(
        // Sub-tile: exercises memcpy_wg fallback path
        static_cast<size_t>(1024),
        // Immediately below the put threshold: must still fall back
        static_cast<size_t>(PUT_THRESHOLD - 1),
        // Exactly at the put threshold (and exactly one TDM tile): must
        // take the TDM path
        static_cast<size_t>(PUT_THRESHOLD),
        // Multiple tiles: exercises the double-buffered loop
        static_cast<size_t>(TDM_TILE_BYTES * 4),
        // Non-power-of-two: partial remainder handled by memcpy_wg
        static_cast<size_t>(TDM_TILE_BYTES * 3 + 1337)
    )
);
