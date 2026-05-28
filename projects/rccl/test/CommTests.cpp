/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "gtest/gtest.h"
#include "comm.h"



namespace RcclUnitTesting
{
  TEST(CommTests, Sorter)
  {
	// Configuration
	ncclTaskCollSorter* me_ptr = new ncclTaskCollSorter;
	me_ptr->head = nullptr;

	ASSERT_EQ(ncclTaskCollSorterEmpty(me_ptr), true);
	delete me_ptr;
  }
}




