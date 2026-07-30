/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file TimeoutTests.cpp
 * @brief Unit coverage for the ncclTimeout result code (AICOMNET-193).
 *
 * The NCCL v2.30.4 sync brings ncclTimeout (=8) plus device-side producers in
 * the LSA and GIN barriers. These host-side unit tests pin the result-code
 * contract that the device producers (and any other caller) rely on:
 *   - enum identity / ordering  (nccl.h.in)
 *   - ncclGetErrorString()      (init.cc)
 *   - ncclCommSetAsyncError()   range guard accepts ncclTimeout, rejects 9/NULL
 *   - ncclCommGetAsyncError()   round-trips the stored ncclTimeout back out
 */

#include <cstring>

#include <gtest/gtest.h>

#include "comm.h" // ncclComm, NCCL_MAGIC, ncclCommSetAsyncError (internal)
#include "nccl.h" // public ncclResult_t / ncclGetErrorString

namespace RcclUnitTesting
{
// Build a minimally-valid comm so CommCheck (startMagic/endMagic) passes.
static ncclComm_t MakeMagicComm()
{
    ncclComm_t comm = new ncclComm();
    comm->startMagic = NCCL_MAGIC;
    comm->endMagic   = NCCL_MAGIC;
    return comm;
}

static void FreeComm(ncclComm_t comm) { delete comm; }

// ncclTimeout must sit at 8, just after ncclInProgress, and the result-count
// sentinel must be 9 so the new code is inside the valid range. This matches
// the upstream NCCL enum value exactly.
TEST(TimeoutTests, EnumValueAndOrdering)
{
    EXPECT_EQ(static_cast<int>(ncclInProgress), 7);
    EXPECT_EQ(static_cast<int>(ncclTimeout),    8);
    EXPECT_EQ(static_cast<int>(ncclNumResults), 9);

    EXPECT_LT(static_cast<int>(ncclInProgress), static_cast<int>(ncclTimeout));
    EXPECT_LT(static_cast<int>(ncclTimeout),    static_cast<int>(ncclNumResults));
}

// ncclGetErrorString(ncclTimeout) returns the upstream-matching "timeout".
TEST(TimeoutTests, ErrorStringIsTimeout)
{
    EXPECT_STREQ(ncclGetErrorString(ncclTimeout), "timeout");
}

// Adding ncclTimeout must not disturb the existing strings, and out-of-range
// codes must still fall through to "unknown result code".
TEST(TimeoutTests, ErrorStringRegression)
{
    EXPECT_STREQ(ncclGetErrorString(ncclSuccess),       "no error");
    EXPECT_STREQ(ncclGetErrorString(ncclInProgress),    "NCCL operation in progress");
    EXPECT_STREQ(ncclGetErrorString(ncclRemoteError),
                 "remote process exited or there was a network error");
    // ncclNumResults (9) is not a real code -> must remain "unknown".
    EXPECT_STREQ(ncclGetErrorString(ncclNumResults),    "unknown result code");
    EXPECT_STREQ(ncclGetErrorString(static_cast<ncclResult_t>(42)), "unknown result code");
}

// The range guard in ncclCommSetAsyncError (nextState >= ncclNumResults) must
// accept ncclTimeout=8. This is the path a device-side timeout producer takes
// to surface ncclTimeout through the async-error mechanism.
TEST(TimeoutTests, SetAsyncErrorAcceptsTimeout)
{
    ncclComm_t comm = MakeMagicComm();

    EXPECT_EQ(ncclCommSetAsyncError(comm, ncclTimeout), ncclSuccess);
    EXPECT_EQ(comm->asyncResult, ncclTimeout);

    FreeComm(comm);
}

// ncclNumResults (9) and anything above, plus negatives, stay rejected so the
// valid range is exactly [0, 9).
TEST(TimeoutTests, SetAsyncErrorRejectsOutOfRange)
{
    ncclComm_t comm = MakeMagicComm();

    EXPECT_EQ(ncclCommSetAsyncError(comm, ncclNumResults), ncclInvalidArgument);
    EXPECT_EQ(ncclCommSetAsyncError(comm, static_cast<ncclResult_t>(9)),  ncclInvalidArgument);
    EXPECT_EQ(ncclCommSetAsyncError(comm, static_cast<ncclResult_t>(-1)), ncclInvalidArgument);
    // A rejected set must not have mutated the stored state.
    EXPECT_EQ(comm->asyncResult, ncclSuccess);

    FreeComm(comm);
}

// The same guard rejects a NULL comm. ncclTimeout is in range, so this proves
// the comm==NULL arm fires independently of the range check.
TEST(TimeoutTests, SetAsyncErrorRejectsNullComm)
{
    EXPECT_EQ(ncclCommSetAsyncError(nullptr, ncclTimeout), ncclInvalidArgument);
}

// End-to-end pipeline: set ncclTimeout, read it back via the public getter,
// and confirm the getter-returned code stringifies to "timeout".
TEST(TimeoutTests, GetAsyncErrorRoundTripsTimeout)
{
    ncclComm_t comm = MakeMagicComm();

    ASSERT_EQ(ncclCommSetAsyncError(comm, ncclTimeout), ncclSuccess);

    ncclResult_t observed = ncclSuccess;
    EXPECT_EQ(ncclCommGetAsyncError(comm, &observed), ncclSuccess);
    EXPECT_EQ(observed, ncclTimeout);
    EXPECT_STREQ(ncclGetErrorString(observed), "timeout");

    FreeComm(comm);
}

// ncclTimeout is a clean, clearable async state: setting then resetting to
// ncclSuccess leaves the getter reporting success again.
TEST(TimeoutTests, AsyncErrorIsClearable)
{
    ncclComm_t comm = MakeMagicComm();

    ASSERT_EQ(ncclCommSetAsyncError(comm, ncclTimeout), ncclSuccess);
    ASSERT_EQ(ncclCommSetAsyncError(comm, ncclSuccess), ncclSuccess);

    ncclResult_t observed = ncclTimeout;
    EXPECT_EQ(ncclCommGetAsyncError(comm, &observed), ncclSuccess);
    EXPECT_EQ(observed, ncclSuccess);

    FreeComm(comm);
}
} // namespace RcclUnitTesting
