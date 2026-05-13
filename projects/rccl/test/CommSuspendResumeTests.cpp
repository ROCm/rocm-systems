/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Functional tests for the Dynamic Memory Offload APIs added in NCCL 2.29.7:
//   - ncclCommSuspend(comm, flags)   // NCCL_SUSPEND_MEM
//   - ncclCommResume(comm)
//   - ncclCommMemStats(comm, stat, *value)
//
// These exercise the round-trip lifecycle on a single-rank communicator on a
// single GPU.
//
// Note on linkage:
//   These three symbols are declared in <rccl/rccl.h> but at the time this
//   test was written:
//     * ncclCommSuspend / ncclCommResume have a public declaration in
//       nccl.h but no implementation in librccl; and
//     * ncclCommMemStats is implemented but lacks the NCCL_API() visibility
//       wrapper so the symbol is hidden in the shared object.
//   Resolving them via dlsym() makes this test compile and link in all of
//   the above states, and skip cleanly at runtime when the implementation
//   is unavailable. Once the RCCL sync exposes these symbols, the tests
//   will exercise them automatically.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "StandaloneUtils.hpp"

namespace RcclUnitTesting
{

using SuspendFn  = ncclResult_t (*)(ncclComm_t, int);
using ResumeFn   = ncclResult_t (*)(ncclComm_t);
using MemStatsFn = ncclResult_t (*)(ncclComm_t, ncclCommMemStat_t, uint64_t*);

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

#define SKIP_IF_SYMBOL_MISSING(sym_var, name)                                  \
    do {                                                                       \
        if ((sym_var) == nullptr) {                                            \
            GTEST_SKIP() << "Skipping: '" << (name)                            \
                         << "' is not exported by librccl. "                   \
                         << "This release of RCCL does not yet implement the " \
                         << "API; the test is preserved for once it lands.";  \
            return;                                                            \
        }                                                                      \
    } while (0)

static SuspendFn  resolveSuspend()  { return (SuspendFn)  dlsym(RTLD_DEFAULT, "ncclCommSuspend"); }
static ResumeFn   resolveResume()   { return (ResumeFn)   dlsym(RTLD_DEFAULT, "ncclCommResume"); }
static MemStatsFn resolveMemStats() { return (MemStatsFn) dlsym(RTLD_DEFAULT, "ncclCommMemStats"); }

static ncclResult_t initSingleRankComm(ncclComm_t* comm) {
    ncclUniqueId id;
    ncclResult_t res = ncclGetUniqueId(&id);
    if (res != ncclSuccess) return res;
    return ncclCommInitRank(comm, 1, id, 0);
}

// Run a trivial AllReduce on a single-rank comm so that any "lazy" comm
// resources get allocated before suspend/resume is exercised.
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

// ---------------------------------------------------------------------------
// Round-trip: suspend then resume must allow a follow-up collective.
// ---------------------------------------------------------------------------

static void testSuspendResumeRoundTrip() {
    SKIP_IF_NO_GPU();
    SuspendFn ncclCommSuspend_p = resolveSuspend();
    ResumeFn  ncclCommResume_p  = resolveResume();
    SKIP_IF_SYMBOL_MISSING(ncclCommSuspend_p, "ncclCommSuspend");
    SKIP_IF_SYMBOL_MISSING(ncclCommResume_p,  "ncclCommResume");

    HIPCALL(hipSetDevice(0));
    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));

    runTrivialAllReduce(comm);

    NCCLCHECK(ncclCommSuspend_p(comm, NCCL_SUSPEND_MEM));
    NCCLCHECK(ncclCommResume_p(comm));

    // After resume, the comm must continue to function.
    runTrivialAllReduce(comm);

    NCCLCHECK(ncclCommDestroy(comm));
}

// ---------------------------------------------------------------------------
// Memory statistics expose the suspend state and a non-zero total footprint.
// ---------------------------------------------------------------------------

static void testMemStatsReflectSuspendState() {
    SKIP_IF_NO_GPU();
    SuspendFn  ncclCommSuspend_p  = resolveSuspend();
    ResumeFn   ncclCommResume_p   = resolveResume();
    MemStatsFn ncclCommMemStats_p = resolveMemStats();
    SKIP_IF_SYMBOL_MISSING(ncclCommSuspend_p,  "ncclCommSuspend");
    SKIP_IF_SYMBOL_MISSING(ncclCommResume_p,   "ncclCommResume");
    SKIP_IF_SYMBOL_MISSING(ncclCommMemStats_p, "ncclCommMemStats");

    HIPCALL(hipSetDevice(0));
    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));
    runTrivialAllReduce(comm);

    uint64_t totalBefore = 0;
    uint64_t suspendedBefore = 0;
    NCCLCHECK(ncclCommMemStats_p(comm, ncclStatGpuMemTotal,     &totalBefore));
    NCCLCHECK(ncclCommMemStats_p(comm, ncclStatGpuMemSuspended, &suspendedBefore));

    EXPECT_GT(totalBefore, 0u)
        << "Expected non-zero total GPU memory tracked by NCCL after a collective";
    EXPECT_EQ(suspendedBefore, 0u)
        << "Suspended flag should be 0 before ncclCommSuspend is called";

    NCCLCHECK(ncclCommSuspend_p(comm, NCCL_SUSPEND_MEM));

    uint64_t suspendedAfter = 0;
    NCCLCHECK(ncclCommMemStats_p(comm, ncclStatGpuMemSuspended, &suspendedAfter));
    EXPECT_EQ(suspendedAfter, 1u)
        << "Suspended flag should be 1 after ncclCommSuspend(NCCL_SUSPEND_MEM)";

    NCCLCHECK(ncclCommResume_p(comm));

    uint64_t suspendedAfterResume = 0;
    NCCLCHECK(ncclCommMemStats_p(comm, ncclStatGpuMemSuspended, &suspendedAfterResume));
    EXPECT_EQ(suspendedAfterResume, 0u)
        << "Suspended flag should be back to 0 after ncclCommResume";

    NCCLCHECK(ncclCommDestroy(comm));
}

// ---------------------------------------------------------------------------
// Suspend with an unknown flag bit must be rejected.
// ---------------------------------------------------------------------------

static void testSuspendRejectsUnknownFlags() {
    SKIP_IF_NO_GPU();
    SuspendFn  ncclCommSuspend_p  = resolveSuspend();
    MemStatsFn ncclCommMemStats_p = resolveMemStats();
    SKIP_IF_SYMBOL_MISSING(ncclCommSuspend_p,  "ncclCommSuspend");
    SKIP_IF_SYMBOL_MISSING(ncclCommMemStats_p, "ncclCommMemStats");

    HIPCALL(hipSetDevice(0));
    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));

    // 0 is not a valid suspend request - nothing to do; we accept either
    // ncclSuccess (no-op) or ncclInvalidArgument, but the state flag must
    // remain 0 afterwards.
    ncclResult_t r = ncclCommSuspend_p(comm, 0);
    EXPECT_TRUE(r == ncclSuccess || r == ncclInvalidArgument)
        << "Unexpected error from ncclCommSuspend(0): "
        << ncclGetErrorString(r);

    uint64_t suspended = 0;
    NCCLCHECK(ncclCommMemStats_p(comm, ncclStatGpuMemSuspended, &suspended));
    EXPECT_EQ(suspended, 0u);

    // Unknown high-bit flag must be rejected.
    r = ncclCommSuspend_p(comm, 0x80000000);
    EXPECT_EQ(r, ncclInvalidArgument)
        << "Expected ncclInvalidArgument for unknown suspend flag, got "
        << ncclGetErrorString(r);

    NCCLCHECK(ncclCommDestroy(comm));
}

// ---------------------------------------------------------------------------
// Calling resume on a comm that was not suspended must not corrupt state.
// ---------------------------------------------------------------------------

static void testResumeWithoutSuspendIsSafe() {
    SKIP_IF_NO_GPU();
    ResumeFn ncclCommResume_p = resolveResume();
    SKIP_IF_SYMBOL_MISSING(ncclCommResume_p, "ncclCommResume");

    HIPCALL(hipSetDevice(0));
    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));
    runTrivialAllReduce(comm);

    ncclResult_t r = ncclCommResume_p(comm);
    EXPECT_TRUE(r == ncclSuccess || r == ncclInvalidArgument)
        << "ncclCommResume on a non-suspended comm must not crash; got "
        << ncclGetErrorString(r);

    // Comm must still be usable regardless of which behavior the implementation
    // chose for the no-op case.
    runTrivialAllReduce(comm);

    NCCLCHECK(ncclCommDestroy(comm));
}

// ---------------------------------------------------------------------------
// Argument validation: NULL comm must be rejected.
// ---------------------------------------------------------------------------

static void testSuspendResumeRejectNullComm() {
    SKIP_IF_NO_GPU();
    SuspendFn  ncclCommSuspend_p  = resolveSuspend();
    ResumeFn   ncclCommResume_p   = resolveResume();
    MemStatsFn ncclCommMemStats_p = resolveMemStats();

    if (ncclCommSuspend_p) {
        EXPECT_EQ(ncclCommSuspend_p(nullptr, NCCL_SUSPEND_MEM),
                  ncclInvalidArgument);
    }
    if (ncclCommResume_p) {
        EXPECT_EQ(ncclCommResume_p(nullptr), ncclInvalidArgument);
    }
    if (ncclCommMemStats_p) {
        uint64_t v = 0;
        EXPECT_EQ(ncclCommMemStats_p(nullptr, ncclStatGpuMemTotal, &v),
                  ncclInvalidArgument);
    }
    if (!ncclCommSuspend_p && !ncclCommResume_p && !ncclCommMemStats_p) {
        GTEST_SKIP() << "None of ncclCommSuspend/Resume/MemStats are exported "
                        "by librccl in this build.";
    }
}

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

TEST(CommSuspendResume, ProcessIsolatedSuite)
{
    RUN_ISOLATED_TESTS(
        ProcessIsolatedTestRunner::TestConfig("SuspendResume_RoundTrip",
            testSuspendResumeRoundTrip),
        ProcessIsolatedTestRunner::TestConfig("MemStats_ReflectSuspendState",
            testMemStatsReflectSuspendState),
        ProcessIsolatedTestRunner::TestConfig("Suspend_RejectsUnknownFlags",
            testSuspendRejectsUnknownFlags),
        ProcessIsolatedTestRunner::TestConfig("Resume_WithoutSuspend_IsSafe",
            testResumeWithoutSuspendIsSafe),
        ProcessIsolatedTestRunner::TestConfig("Suspend_Resume_RejectNullComm",
            testSuspendResumeRejectNullComm)
    );
}

} // namespace RcclUnitTesting
