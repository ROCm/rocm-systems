/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the RCCL_MEMALLOC_FABRIC decision of ncclMemAlloc (src/allocator.cc,
// rcclMemAllocWantFabric): auto mode upgrades to FABRIC handles only on gfx1250 with FABRIC
// support and NCCL_MNNVL_ENABLE != 0; every other device / setting keeps the stock POSIX_FD buffer.

#include <gtest/gtest.h>

#include "rccl_memalloc_fabric.h"

namespace {
constexpr int64_t kAuto = -1, kOff = 0, kForce = 1;
constexpr int64_t kMnnvlOff = 0, kMnnvlOn = 1, kMnnvlAuto = 2;
}  // namespace

TEST(MemAllocFabricMicrotest, AutoUpgradesOnlyGfx1250WithFabricAndMnnvl) {
  EXPECT_TRUE(rcclMemAllocWantFabric(kAuto, kMnnvlAuto, /*fabric=*/true, /*gfx1250=*/true));
  EXPECT_TRUE(rcclMemAllocWantFabric(kAuto, kMnnvlOn, true, true));
}

TEST(MemAllocFabricMicrotest, AutoKeepsStockBufferElsewhere) {
  // non-gfx1250 (gfx942 / gfx950 with FABRIC-capable drivers): unchanged
  EXPECT_FALSE(rcclMemAllocWantFabric(kAuto, kMnnvlAuto, true, false));
  // gfx1250 with MNNVL disabled (single-tray / non-MNNVL comms must keep registering POSIX buffers)
  EXPECT_FALSE(rcclMemAllocWantFabric(kAuto, kMnnvlOff, true, true));
  // no FABRIC handle support
  EXPECT_FALSE(rcclMemAllocWantFabric(kAuto, kMnnvlAuto, false, true));
}

TEST(MemAllocFabricMicrotest, ExplicitModes) {
  for (bool gfx1250 : {false, true}) {
    for (int64_t mnnvl : {kMnnvlOff, kMnnvlAuto}) {
      EXPECT_TRUE(rcclMemAllocWantFabric(kForce, mnnvl, true, gfx1250));
      EXPECT_FALSE(rcclMemAllocWantFabric(kForce, mnnvl, false, gfx1250));  // cannot force without support
      EXPECT_FALSE(rcclMemAllocWantFabric(kOff, mnnvl, true, gfx1250));
    }
  }
}
