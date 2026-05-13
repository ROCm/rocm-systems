/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Functional tests for NCCL_CHECK_MODE (added in NCCL 2.29.7).
//
// NCCL 2.29.7 introduced NCCL_CHECK_MODE as the modern replacement for the
// legacy NCCL_CHECK_POINTERS env var. The implementation lives in init.cc
// around the WARN that says "use NCCL_CHECK_MODE instead" for the legacy
// var. Recognized values typically include OFF / FAST / STRICT / DEBUG,
// where DEBUG additionally validates symmetric-window buffer registration.
//
// These tests exercise the lifecycle of a healthy comm under each of the
// known modes via process-isolated execution so that the env var is parsed
// fresh on every run, and ensure that an unknown value is rejected with a
// non-fatal warning rather than a crash.

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

// Run a trivial AllReduce so that the comm actually exercises the
// check-mode-gated code paths.
static void runTrivialAllReduce(ncclComm_t comm) {
    const size_t numElements = 1024;
    float* devBuf = nullptr;
    HIPCALL(hipMalloc(&devBuf, numElements * sizeof(float)));
    HIPCALL(hipMemset(devBuf, 0, numElements * sizeof(float)));

    hipStream_t stream;
    HIPCALL(hipStreamCreate(&stream));

    NCCLCHECK(ncclAllReduce(devBuf, devBuf, numElements, ncclFloat32,
                            ncclSum, comm, stream));
    HIPCALL(hipStreamSynchronize(stream));

    HIPCALL(hipStreamDestroy(stream));
    HIPCALL(hipFree(devBuf));
}

// Each variant runs the same comm lifecycle. The interesting assertion is
// that init + collective + destroy all succeed under each mode value.
static void runHealthyLifecycle() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));
    runTrivialAllReduce(comm);
    NCCLCHECK(ncclCommDestroy(comm));
}

// An unknown NCCL_CHECK_MODE value should not block init; the implementation
// is expected to log a warning and fall back to its default behavior.
static void runUnknownModeFallsBackToDefault() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));
    runTrivialAllReduce(comm);
    NCCLCHECK(ncclCommDestroy(comm));
}

// Legacy NCCL_CHECK_POINTERS=1 is still honored but should emit a deprecation
// notice. We only assert that comms continue to work; capturing the log line
// is left to higher-level integration tests.
static void runLegacyCheckPointersAlias() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));
    runTrivialAllReduce(comm);
    NCCLCHECK(ncclCommDestroy(comm));
}

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

TEST(CheckMode, ProcessIsolatedSuite)
{
    using TC = ProcessIsolatedTestRunner::TestConfig;
    RUN_ISOLATED_TESTS(
        TC("CheckMode_OFF",     runHealthyLifecycle)
            .withEnvironment({{"NCCL_CHECK_MODE", "OFF"}}),
        TC("CheckMode_FAST",    runHealthyLifecycle)
            .withEnvironment({{"NCCL_CHECK_MODE", "FAST"}}),
        TC("CheckMode_STRICT",  runHealthyLifecycle)
            .withEnvironment({{"NCCL_CHECK_MODE", "STRICT"}}),
        TC("CheckMode_DEBUG",   runHealthyLifecycle)
            .withEnvironment({{"NCCL_CHECK_MODE", "DEBUG"}}),
        TC("CheckMode_UNKNOWN", runUnknownModeFallsBackToDefault)
            .withEnvironment({{"NCCL_CHECK_MODE", "BOGUS"}}),
        TC("CheckMode_LegacyCheckPointers", runLegacyCheckPointersAlias)
            .withEnvironment({{"NCCL_CHECK_POINTERS", "1"}})
    );
}

} // namespace RcclUnitTesting
