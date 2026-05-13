/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Functional tests for the ncclCommGrow / ncclCommGetUniqueId APIs that were
// added in NCCL 2.29.2. The grow API allows callers to dynamically add ranks
// to an existing communicator and is typically paired with ncclCommShrink for
// elastic workloads.
//
// This file exercises the synchronous, single-process error paths of
// ncclCommGrow and the basic behavior of ncclCommGetUniqueId on a one-rank
// communicator. Multi-rank functional growth (with two or more participating
// ranks) requires multi-process orchestration; those scenarios are intended
// to live in a separate MPI-based test file.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "StandaloneUtils.hpp"

namespace RcclUnitTesting
{

static bool hasGpuAvailable() {
    int numDevices = 0;
    hipError_t err = hipGetDeviceCount(&numDevices);
    return (err == hipSuccess && numDevices >= 1);
}

#define SKIP_IF_NO_GPU()                                                  \
    do {                                                                  \
        if (!hasGpuAvailable()) {                                         \
            GTEST_SKIP() << "This test requires at least 1 GPU device."; \
            return;                                                       \
        }                                                                 \
    } while (0)

static ncclResult_t initSingleRankComm(ncclComm_t* comm) {
    ncclUniqueId id;
    ncclResult_t res = ncclGetUniqueId(&id);
    if (res != ncclSuccess) return res;
    return ncclCommInitRank(comm, 1, id, 0);
}

// ---------------------------------------------------------------------------
// ncclCommGrow: argument validation
// ---------------------------------------------------------------------------

// newcomm pointer is required by the API; passing NULL must be rejected
// without dereferencing.
static void testGrowRejectsNullOutComm() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));

    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));

    ncclResult_t r = ncclCommGrow(comm, 2, &id, -1, /*newcomm=*/nullptr, nullptr);
    EXPECT_EQ(r, ncclInvalidArgument)
        << "Expected ncclInvalidArgument for NULL newcomm, got "
        << ncclGetErrorString(r);

    NCCLCHECK(ncclCommDestroy(comm));
}

// nRanks <= 0 is a documented error; matches the WARN("total ranks must be
// positive") path in init.cc.
static void testGrowRejectsNonPositiveNRanks() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));

    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));
    ncclComm_t newcomm = nullptr;

    EXPECT_EQ(ncclCommGrow(comm, 0,  &id, -1, &newcomm, nullptr),
              ncclInvalidArgument);
    EXPECT_EQ(ncclCommGrow(comm, -1, &id, -1, &newcomm, nullptr),
              ncclInvalidArgument);
    EXPECT_EQ(newcomm, nullptr)
        << "newcomm must not be populated on argument error";

    NCCLCHECK(ncclCommDestroy(comm));
}

// nRanks must be strictly greater than comm->nRanks: passing the same size
// or smaller is rejected.
static void testGrowRejectsShrinkingSize() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));

    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));
    ncclComm_t newcomm = nullptr;

    // existing comm has nRanks = 1
    EXPECT_EQ(ncclCommGrow(comm, 1, &id, -1, &newcomm, nullptr),
              ncclInvalidArgument);

    NCCLCHECK(ncclCommDestroy(comm));
}

// Existing ranks must pass rank == -1. Any other value is rejected.
static void testGrowRejectsExistingRankWithExplicitRank() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));

    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));
    ncclComm_t newcomm = nullptr;

    // existing-rank path: comm != NULL but rank != -1
    EXPECT_EQ(ncclCommGrow(comm, 2, &id, 0,  &newcomm, nullptr),
              ncclInvalidArgument);
    EXPECT_EQ(ncclCommGrow(comm, 2, &id, 5,  &newcomm, nullptr),
              ncclInvalidArgument);

    NCCLCHECK(ncclCommDestroy(comm));
}

// New ranks (comm == NULL) must pass a non-negative rank.
static void testGrowRejectsNewRankWithNegativeRank() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));
    ncclComm_t newcomm = nullptr;

    EXPECT_EQ(ncclCommGrow(/*comm=*/nullptr, 2, &id, -1, &newcomm, nullptr),
              ncclInvalidArgument);
}

// New ranks must pass a non-NULL uniqueId pointer.
static void testGrowRejectsNewRankWithNullUniqueId() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t newcomm = nullptr;
    EXPECT_EQ(ncclCommGrow(/*comm=*/nullptr, 2, /*uniqueId=*/nullptr, 1,
                           &newcomm, nullptr),
              ncclInvalidArgument);
}

// New ranks must pass rank < nRanks.
static void testGrowRejectsNewRankOutOfRange() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));
    ncclComm_t newcomm = nullptr;

    // nRanks = 2 -> rank must be < 2
    EXPECT_EQ(ncclCommGrow(/*comm=*/nullptr, 2, &id, 2,  &newcomm, nullptr),
              ncclInvalidArgument);
    EXPECT_EQ(ncclCommGrow(/*comm=*/nullptr, 2, &id, 99, &newcomm, nullptr),
              ncclInvalidArgument);
}

// ---------------------------------------------------------------------------
// ncclCommGetUniqueId
// ---------------------------------------------------------------------------

// On a healthy single-rank comm, ncclCommGetUniqueId must succeed and produce
// a non-zero uniqueId payload.
static void testGetUniqueIdSucceedsOnHealthyComm() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));

    ncclUniqueId id;
    std::memset(&id, 0, sizeof(id));
    NCCLCHECK(ncclCommGetUniqueId(comm, &id));

    // Confirm the payload was actually written (at least one non-zero byte).
    bool anyNonZero = false;
    for (size_t i = 0; i < sizeof(id.internal); ++i) {
        if (((unsigned char*)id.internal)[i] != 0) {
            anyNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(anyNonZero) << "ncclCommGetUniqueId returned an all-zero id";

    NCCLCHECK(ncclCommDestroy(comm));
}

// NULL uniqueId pointer is rejected without dereferencing.
static void testGetUniqueIdRejectsNullOut() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));

    ncclResult_t r = ncclCommGetUniqueId(comm, nullptr);
    EXPECT_EQ(r, ncclInvalidArgument)
        << "Expected ncclInvalidArgument for NULL uniqueId, got "
        << ncclGetErrorString(r);

    NCCLCHECK(ncclCommDestroy(comm));
}

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

TEST(CommGrow, ProcessIsolatedArgValidation)
{
    RUN_ISOLATED_TESTS(
        ProcessIsolatedTestRunner::TestConfig("Grow_RejectsNullOutComm",
            testGrowRejectsNullOutComm),
        ProcessIsolatedTestRunner::TestConfig("Grow_RejectsNonPositiveNRanks",
            testGrowRejectsNonPositiveNRanks),
        ProcessIsolatedTestRunner::TestConfig("Grow_RejectsShrinkingSize",
            testGrowRejectsShrinkingSize),
        ProcessIsolatedTestRunner::TestConfig("Grow_RejectsExistingRankWithExplicitRank",
            testGrowRejectsExistingRankWithExplicitRank),
        ProcessIsolatedTestRunner::TestConfig("Grow_RejectsNewRankWithNegativeRank",
            testGrowRejectsNewRankWithNegativeRank),
        ProcessIsolatedTestRunner::TestConfig("Grow_RejectsNewRankWithNullUniqueId",
            testGrowRejectsNewRankWithNullUniqueId),
        ProcessIsolatedTestRunner::TestConfig("Grow_RejectsNewRankOutOfRange",
            testGrowRejectsNewRankOutOfRange),
        ProcessIsolatedTestRunner::TestConfig("GetUniqueId_SucceedsOnHealthyComm",
            testGetUniqueIdSucceedsOnHealthyComm),
        ProcessIsolatedTestRunner::TestConfig("GetUniqueId_RejectsNullOut",
            testGetUniqueIdRejectsNullOut)
    );
}

} // namespace RcclUnitTesting
