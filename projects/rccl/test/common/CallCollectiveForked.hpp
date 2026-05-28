/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef CALLCOLLECTIVEFORKED_H
#define CALLCOLLECTIVEFORKED_H

#include <vector>

namespace RcclUnitTesting
{
    void callCollectiveForked(int nranks, int collID, const std::vector<int>& sendBuff, std::vector<int>& recvBuff, const std::vector<int>& expected, bool use_managed_mem = false);
}

#endif
