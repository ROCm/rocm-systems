/*
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "gtest/gtest.h"
#include "../../../runtime/hsa-runtime/core/util/poll_backoff.h"

TEST(rocrtstPollBackoff, StartsAtFloor) {
  ASSERT_EQ(rocr::core::kAsyncEventsPollNapFloorUs,
            rocr::core::NextAsyncEventsPollNapUs(0));
}

TEST(rocrtstPollBackoff, DoublesUntilCeiling) {
  auto value = rocr::core::kAsyncEventsPollNapFloorUs;
  value = rocr::core::NextAsyncEventsPollNapUs(value);
  ASSERT_EQ(40u, value);
  value = rocr::core::NextAsyncEventsPollNapUs(value);
  ASSERT_EQ(80u, value);
}

TEST(rocrtstPollBackoff, CapsAtCeiling) {
  auto value = rocr::core::kAsyncEventsPollNapCeilingUs;
  ASSERT_EQ(rocr::core::kAsyncEventsPollNapCeilingUs,
            rocr::core::NextAsyncEventsPollNapUs(value));
}
