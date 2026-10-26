/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Host Rect
 * @{
 * @ingroup HRRTest
 * GPU-free unit tests for the host footprint of a pitched copy
 * (hrr/hrr_host_rect.h). Capture sizes the host snapshot with it and replay
 * sizes and lays out the buffer it reads the snapshot back into, so the
 * numbers pinned here are the ones both sides must agree on.
 */

#include "hrr_test_common.hh"
#include "hrr/hrr_host_rect.h"

#include <cstdint>

/**
 * Test Description
 * ----------------
 *   - A 64x32x3 copy out of a host buffer padded on every axis (rows 96 bytes
 *     apart, slices 40 rows apart) starting at x=8, y=2, z=1: the first copied
 *     byte is at z*slice + y*row + x = 4040, and the footprint runs through the
 *     last row of the last slice, 4040 + 2*3840 + 31*96 + 64 = 14760 bytes.
 */
HRR_TEST_CASE(Unit_HRR_HostRect_PitchedPadded3D) {
  const HrrHostRect r = hrr_host_rect(/*pitch=*/96, /*pitch_height=*/40, /*x=*/8, /*y=*/2,
                                      /*z=*/1, /*width=*/64, /*height=*/32, /*depth=*/3);
  REQUIRE(r.ok);
  CHECK(r.row == 96);
  CHECK(r.slice == 3840);
  CHECK(r.first == 4040);
  CHECK(r.extent == 14760);
  CHECK(r.width == 64);
  CHECK(r.height == 32);
  CHECK(r.depth == 3);
  CHECK_FALSE(hrr_host_rect_dense(&r));
  CHECK(hrr_host_rect_extent(96, 40, 8, 2, 1, 64, 32, 3) == 14760);
}

/**
 * Test Description
 * ----------------
 *   - A zero pitch means rows `width` apart, a zero pitch height means slices
 *     `height` rows apart, and a pitch below the width is raised to it, as the
 *     runtime does. Such a rect is dense: its rows are one run.
 */
HRR_TEST_CASE(Unit_HRR_HostRect_RuntimeDefaults) {
  const HrrHostRect r = hrr_host_rect(0, 0, 0, 0, 0, 16, 4, 2);
  REQUIRE(r.ok);
  CHECK(r.row == 16);
  CHECK(r.slice == 64);
  CHECK(r.first == 0);
  CHECK(r.extent == 128);
  CHECK(hrr_host_rect_dense(&r));

  const HrrHostRect narrow = hrr_host_rect(/*pitch=*/8, 0, 0, 0, 0, 16, 4, 1);
  REQUIRE(narrow.ok);
  CHECK(narrow.row == 16);
  CHECK(narrow.extent == 64);
}

/**
 * Test Description
 * ----------------
 *   - A footprint that does not fit in size_t is refused, not wrapped, whether
 *     the rows themselves overflow or only the offset of the first one does,
 *     and hrr_host_rect_extent() reports SIZE_MAX, which no buffer covers. An
 *     empty copy touches nothing and is ok with a zero extent.
 */
HRR_TEST_CASE(Unit_HRR_HostRect_Overflow) {
  CHECK_FALSE(hrr_host_rect(SIZE_MAX / 2, 0, 0, 0, 0, 1, 3, 1).ok);
  CHECK(hrr_host_rect_extent(SIZE_MAX / 2, 0, 0, 0, 0, 1, 3, 1) == SIZE_MAX);

  CHECK_FALSE(hrr_host_rect(4096, 0, 0, 0, SIZE_MAX / 4096, 1, 2, 1).ok);
  CHECK(hrr_host_rect_extent(4096, 0, 0, 0, SIZE_MAX / 4096, 1, 2, 1) == SIZE_MAX);

  const HrrHostRect empty = hrr_host_rect(SIZE_MAX, SIZE_MAX, 1, 1, 1, 0, 4, 4);
  CHECK(empty.ok);
  CHECK(empty.extent == 0);
}

/** @} */
