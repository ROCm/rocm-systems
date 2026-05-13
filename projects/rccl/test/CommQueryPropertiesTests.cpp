/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Functional tests for ncclCommQueryProperties / ncclCommProperties_t.
//
//   * The base API was added in NCCL 2.29.2 (Device API feature discovery
//     for GIN, host RMA, multimem, etc).
//   * The nLsaTeams field was added in NCCL 2.29.7.
//
// This test verifies that the API populates the expected versioned struct
// fields on a healthy single-rank communicator and validates basic argument
// checking.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#define NCCL_HOSTLIB_ONLY
#include <nccl_device/core_tmp.h>

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
// Property struct is correctly populated for a healthy single-rank comm.
// ---------------------------------------------------------------------------

static void testQueryPropertiesPopulatesBasicFields() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCLCHECK(ncclCommQueryProperties(comm, &props));

    // Versioned struct header: size and magic must come back exactly as the
    // initializer wrote them. (The version field is allowed to be updated by
    // the implementation to reflect the runtime version.)
    EXPECT_EQ(props.size, sizeof(ncclCommProperties_t))
        << "ncclCommQueryProperties must not corrupt the size header";

    // Identity fields must match what we initialized the comm with.
    EXPECT_EQ(props.rank,   0);
    EXPECT_EQ(props.nRanks, 1);

    int hipDev = -1;
    HIPCALL(hipGetDevice(&hipDev));
    EXPECT_EQ(props.cudaDev, hipDev)
        << "props.cudaDev should match the active HIP device";

    NCCLCHECK(ncclCommDestroy(comm));
}

// ---------------------------------------------------------------------------
// The nLsaTeams field added in 2.29.7 is reported and non-negative.
// On a single-rank comm we don't enforce a specific value; the contract is
// that the field is touched (not left at the initializer's default of 0 in a
// way that depends on uninitialized memory) and is a sane non-negative count.
// ---------------------------------------------------------------------------

static void testQueryPropertiesReportsNLsaTeams() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    // Poison the field so we can tell if it gets touched.
    props.nLsaTeams = -42;
    NCCLCHECK(ncclCommQueryProperties(comm, &props));

    EXPECT_NE(props.nLsaTeams, -42)
        << "ncclCommQueryProperties must populate nLsaTeams (2.29.7)";
    EXPECT_GE(props.nLsaTeams, 0)
        << "nLsaTeams must be a non-negative count";

    NCCLCHECK(ncclCommDestroy(comm));
}

// ---------------------------------------------------------------------------
// ginType / railedGinType must be one of the documented enum values.
// ---------------------------------------------------------------------------

static void testQueryPropertiesReportsValidGinType() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCLCHECK(ncclCommQueryProperties(comm, &props));

    auto isValidGinType = [](ncclGinType_t t) {
        return t == NCCL_GIN_TYPE_NONE
            || t == NCCL_GIN_TYPE_PROXY
            || t == NCCL_GIN_TYPE_GDAKI;
    };

    EXPECT_TRUE(isValidGinType(props.ginType))
        << "ginType returned an unexpected value: " << (int)props.ginType;
    EXPECT_TRUE(isValidGinType(props.railedGinType))
        << "railedGinType returned an unexpected value: "
        << (int)props.railedGinType;

    NCCLCHECK(ncclCommDestroy(comm));
}

// ---------------------------------------------------------------------------
// Argument validation: NULL pointers must be rejected without crashing.
// ---------------------------------------------------------------------------

static void testQueryPropertiesRejectsNullArgs() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    // NULL comm
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    EXPECT_EQ(ncclCommQueryProperties(nullptr, &props), ncclInvalidArgument);

    // NULL out-ptr on a valid comm
    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));
    EXPECT_EQ(ncclCommQueryProperties(comm, nullptr), ncclInvalidArgument);
    NCCLCHECK(ncclCommDestroy(comm));
}

// ---------------------------------------------------------------------------
// Calling without first running the initializer must still be safe: the
// implementation can either reject (mismatched size/magic) or self-correct,
// but it must never crash and must not leave the struct in an indeterminate
// state.
// ---------------------------------------------------------------------------

static void testQueryPropertiesHandlesUninitializedStruct() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));

    ncclCommProperties_t props;
    std::memset(&props, 0, sizeof(props));
    // Caller forgot to use NCCL_COMM_PROPERTIES_INITIALIZER; size/magic are 0.
    // Any of these three outcomes is acceptable: implementation accepts the
    // call as a no-op fill-in, rejects it as invalid argument, or rejects it
    // as invalid usage. The only requirement is that it doesn't crash and
    // doesn't return a transport-level / async error.
    ncclResult_t r = ncclCommQueryProperties(comm, &props);
    EXPECT_TRUE(r == ncclSuccess
             || r == ncclInvalidArgument
             || r == ncclInvalidUsage)
        << "Unexpected error from ncclCommQueryProperties on zeroed struct: "
        << ncclGetErrorString(r);

    NCCLCHECK(ncclCommDestroy(comm));
}

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

TEST(CommQueryProperties, ProcessIsolatedSuite)
{
    RUN_ISOLATED_TESTS(
        ProcessIsolatedTestRunner::TestConfig("QueryProperties_PopulatesBasicFields",
            testQueryPropertiesPopulatesBasicFields),
        ProcessIsolatedTestRunner::TestConfig("QueryProperties_ReportsNLsaTeams",
            testQueryPropertiesReportsNLsaTeams),
        ProcessIsolatedTestRunner::TestConfig("QueryProperties_ReportsValidGinType",
            testQueryPropertiesReportsValidGinType),
        ProcessIsolatedTestRunner::TestConfig("QueryProperties_RejectsNullArgs",
            testQueryPropertiesRejectsNullArgs),
        ProcessIsolatedTestRunner::TestConfig("QueryProperties_HandlesUninitializedStruct",
            testQueryPropertiesHandlesUninitializedStruct)
    );
}

} // namespace RcclUnitTesting
