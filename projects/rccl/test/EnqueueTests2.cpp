/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// Tests for enqueue.cc that don't depend on GPUs
// 
#include <gtest/gtest.h>

#include ENQUEUE_CC_PATH

namespace RcclUnitTesting
{
namespace
{
    TEST(AppleSauce, Banana)
    {
        EXPECT_TRUE(false);
    }
}
}
