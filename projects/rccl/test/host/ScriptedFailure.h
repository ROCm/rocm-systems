/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_SCRIPTED_FAILURE_H_
#define RCCL_TEST_HOST_SCRIPTED_FAILURE_H_

#include "nccl.h"

// ScriptedFailure -- "fail call N of this entry point"
//
// A test that needs the *second* connect() to fail otherwise grows a bespoke
// pair of file-scope knobs per seam (`g_fooCalls` + `g_failFooCall`) and a
// hand-written comparison inside every fake. ScriptedFailure is that pattern
// once: the fake asks what to return for this call number, and the test says
// which call fails with what.
//
// `onCall` is 1-based; 0 means "every call", so a test that only cares that the
// seam fails does not have to know how many times production calls it.
//
// Usage -- in the fake, next to the counter the test asserts on:
//
//   int connectCalls = 0;
//   ScriptedFailure failConnect;        // default: every call succeeds
//
//   ncclResult_t Connect(...) {
//       ncclResult_t ret = failConnect.at(++connectCalls);
//       if (ret != ncclSuccess) return ret;
//       ...succeed...
//   }
//
// and in the test:
//
//   fake_.failConnect = {ncclSystemError, 2};   // the second connect fails
//
// Deliberately a plain aggregate with no counter of its own: tests assign whole
// scripts with `= {ncclSystemError, 2}` and reset them with `= {}`, and the call
// counter usually has to be visible to the test's own assertions anyway. Keeping
// the count outside also leaves the fake free to share one script across several
// entry points, or to count calls the script does not care about.
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
