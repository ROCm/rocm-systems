/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#include "tdm_struct_put_gtest.hpp"

using namespace rocshmem;

TEST_P(TdmStructPutFixture, put_records) {
    run_put(GetParam());
}

// Default TDM tile size (4096 bytes) expressed in whole TdmTestRecords (32 bytes each).
static constexpr size_t TDM_TILE_RECORDS = 4096 / sizeof(TdmTestRecord);  // 128

INSTANTIATE_TEST_SUITE_P(
    TdmStructPut,
    TdmStructPutFixture,
    ::testing::Values(
        // Sub-tile: below TDM_TILE_RECORDS, exercises memcpy_wg fallback
        static_cast<size_t>(64),
        // Exactly one tile
        static_cast<size_t>(TDM_TILE_RECORDS),
        // Four tiles
        static_cast<size_t>(4 * TDM_TILE_RECORDS),
        // Non-aligned: partial remainder
        static_cast<size_t>(3 * TDM_TILE_RECORDS + 17)
    )
);
