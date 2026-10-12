/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#if defined(ENABLE_ROCSHMEM_GIN)

#include <infiniband/verbs.h>

#include "gin_rocshmem_gda_inspect.h"

namespace {

TEST(GinMtuNegotiation, IbvMtuToIntValidValues) {
  EXPECT_EQ(ginTestIbvMtuToInt(IBV_MTU_256), 256);
  EXPECT_EQ(ginTestIbvMtuToInt(IBV_MTU_512), 512);
  EXPECT_EQ(ginTestIbvMtuToInt(IBV_MTU_1024), 1024);
  EXPECT_EQ(ginTestIbvMtuToInt(IBV_MTU_2048), 2048);
  EXPECT_EQ(ginTestIbvMtuToInt(IBV_MTU_4096), 4096);
}

TEST(GinMtuNegotiation, IbvMtuToIntInvalidValue) {
  EXPECT_EQ(ginTestIbvMtuToInt(0), 0);
  EXPECT_EQ(ginTestIbvMtuToInt(6), 0);
}

TEST(GinMtuNegotiation, NegotiatePicksSmallerRemote) {
  int out = -1;
  ASSERT_EQ(ginTestNegotiatePathMtu(IBV_MTU_4096, IBV_MTU_1024, &out), 0);
  EXPECT_EQ(out, 1024);
}

TEST(GinMtuNegotiation, NegotiatePicksSmallerLocal) {
  int out = -1;
  ASSERT_EQ(ginTestNegotiatePathMtu(IBV_MTU_1024, IBV_MTU_4096, &out), 0);
  EXPECT_EQ(out, 1024);
}

TEST(GinMtuNegotiation, NegotiateEqualMtus) {
  int out = -1;
  ASSERT_EQ(ginTestNegotiatePathMtu(IBV_MTU_4096, IBV_MTU_4096, &out), 0);
  EXPECT_EQ(out, 4096);
}

TEST(GinMtuNegotiation, NegotiateRejectsInvalidRemoteMtu) {
  int out = -1;
  EXPECT_EQ(ginTestNegotiatePathMtu(IBV_MTU_4096, 0, &out), -1);
  EXPECT_EQ(out, -1);  // untouched on failure
}

}  // namespace

#else  // !ENABLE_ROCSHMEM_GIN

TEST(GinMtuNegotiation, SkippedWithoutGinBuild) {
  GTEST_SKIP() << "GIN MTU negotiation tests require ENABLE_ROCSHMEM_GIN";
}

#endif  // ENABLE_ROCSHMEM_GIN
