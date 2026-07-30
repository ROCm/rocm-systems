// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cassert>

#if !defined(RJ_ASSERTIONS_ENABLED)
#error "RJ_ENABLE_ASSERTIONS requires RJ_ASSERTIONS_ENABLED"
#endif

#if defined(NDEBUG)
#error "RJ_ENABLE_ASSERTIONS requires NDEBUG to be undefined"
#endif

TEST(BuildConfigurationTest, StandardAssertAbortsInOptimizedBuilds) {
  EXPECT_DEATH({ assert(false && "assertions are live"); }, "assertions are live");
}
