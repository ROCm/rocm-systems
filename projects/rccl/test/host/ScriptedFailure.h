/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_SCRIPTED_FAILURE_H_
#define RCCL_TEST_HOST_SCRIPTED_FAILURE_H_

#include "nccl.h"

// "Fail call N of this seam": the fake asks at(++calls); onCall is 1-based, 0 fails
// every call. A plain aggregate with no counter of its own, so tests can assign
// (`= {ncclSystemError, 2}`) and reset (`= {}`) whole scripts, and one script can be
// shared across seams whose counters the test asserts on anyway.
struct ScriptedFailure {
    ncclResult_t result = ncclSuccess;
    int onCall = 0;

    // Status the seam should return for call number `callNumber` (1-based).
    ncclResult_t at(int callNumber) const
    {
        if (result == ncclSuccess) return ncclSuccess;
        if (onCall == 0 || onCall == callNumber) return result;
        return ncclSuccess;
    }
};

#endif  // RCCL_TEST_HOST_SCRIPTED_FAILURE_H_
