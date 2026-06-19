/*************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "comm.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{
    TEST(MiscTests, Sorter)
    {
        ncclTaskCollSorter* me_ptr = new ncclTaskCollSorter;
        me_ptr->head                 = nullptr;

        ASSERT_EQ(ncclTaskCollSorterEmpty(me_ptr), true);
        delete me_ptr;
    }
}
