/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Functional tests for the one-sided host APIs added in NCCL 2.29.2.
//
// The public API surface is:
//
//   ncclResult_t ncclPutSignal(const void* localbuff, size_t count,
//                              ncclDataType_t datatype, int peer,
//                              ncclWindow_t peerWin, size_t peerWinOffset,
//                              int sigIdx, int ctx, unsigned int flags,
//                              ncclComm_t comm, hipStream_t stream);
//
//   ncclResult_t ncclSignal(int peer, int sigIdx, int ctx,
//                           unsigned int flags, ncclComm_t comm,
//                           hipStream_t stream);
//
//   ncclResult_t ncclWaitSignal(int nDesc,
//                               ncclWaitSignalDesc_t* signalDescs,
//                               ncclComm_t comm, hipStream_t stream);
//
//   typedef struct { int opCnt; int peer; int sigIdx; int ctx; }
//       ncclWaitSignalDesc_t;
//
// What this test covers:
//
//   1) Symbol presence: each entry point links and is callable as a
//      regular ABI function (no dlsym() needed; these are exported
//      from librccl.so).
//   2) Type shape: ncclWaitSignalDesc_t has the documented public
//      layout - opCnt, peer, sigIdx, ctx - and zero-initializes cleanly.
//   3) Argument validation: each API rejects a null comm with a
//      non-success error code rather than crashing.
//   4) End-to-end best-effort drive: on a single-rank comm with a
//      registered window, issue Signal followed by WaitSignal and
//      check the stream completes without error. The single-rank
//      loopback is best-effort: implementations are free to gate this
//      path on multi-rank topologies, so we accept ncclSuccess as well
//      as the well-defined "not supported here" error codes.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// 1) Symbol presence: the entry points are part of the public ABI.
// Taking the address forces the loader to resolve them at link time,
// and the runtime check confirms they end up at a non-null address.
static void testSymbolsPresent()
{
    using PutSignalFn = ncclResult_t (*)(const void*, size_t, ncclDataType_t,
                                         int, ncclWindow_t, size_t,
                                         int, int, unsigned int,
                                         ncclComm_t, hipStream_t);
    using SignalFn    = ncclResult_t (*)(int, int, int, unsigned int,
                                         ncclComm_t, hipStream_t);
    using WaitFn      = ncclResult_t (*)(int, ncclWaitSignalDesc_t*,
                                         ncclComm_t, hipStream_t);

    PutSignalFn fpPut  = &ncclPutSignal;
    SignalFn    fpSig  = &ncclSignal;
    WaitFn      fpWait = &ncclWaitSignal;

    EXPECT_NE(fpPut,  nullptr) << "ncclPutSignal is not linked.";
    EXPECT_NE(fpSig,  nullptr) << "ncclSignal is not linked.";
    EXPECT_NE(fpWait, nullptr) << "ncclWaitSignal is not linked.";
}

// 2) Public type shape of ncclWaitSignalDesc_t.
static void testWaitSignalDescShape()
{
    ncclWaitSignalDesc_t d{};
    EXPECT_EQ(d.opCnt,  0);
    EXPECT_EQ(d.peer,   0);
    EXPECT_EQ(d.sigIdx, 0);
    EXPECT_EQ(d.ctx,    0);

    d.opCnt  = 1;
    d.peer   = 2;
    d.sigIdx = 3;
    d.ctx    = 4;
    EXPECT_EQ(d.opCnt,  1);
    EXPECT_EQ(d.peer,   2);
    EXPECT_EQ(d.sigIdx, 3);
    EXPECT_EQ(d.ctx,    4);

    EXPECT_GE(sizeof(ncclWaitSignalDesc_t), 4u * sizeof(int))
        << "ncclWaitSignalDesc_t is smaller than the four documented int "
           "fields; struct layout may have drifted.";
}

// Helper: an outcome is "rejected" if it's a non-success result we
// expect from a defensive argcheck path on bad input.
static bool isRejected(ncclResult_t r) {
    return r != ncclSuccess;
}

// 3) Argument validation: passing a null communicator should be
// rejected by every entry point and must not crash the process.
static void testNullCommRejected()
{
    SKIP_IF_NO_GPU();

    hipStream_t s = nullptr;

    // Allocate a small device buffer just to give the put a plausible
    // source pointer. The call should still fail on the null comm
    // before any data is read.
    void* devBuf = nullptr;
    ASSERT_EQ(hipMalloc(&devBuf, 64), hipSuccess);

    ncclResult_t rPut = ncclPutSignal(
        devBuf, /*count=*/16, ncclFloat32,
        /*peer=*/0, /*peerWin=*/nullptr, /*peerWinOffset=*/0,
        /*sigIdx=*/0, /*ctx=*/0, /*flags=*/0,
        /*comm=*/nullptr, s);
    EXPECT_TRUE(isRejected(rPut))
        << "ncclPutSignal accepted a null comm; got "
        << ncclGetErrorString(rPut);

    ncclResult_t rSig = ncclSignal(
        /*peer=*/0, /*sigIdx=*/0, /*ctx=*/0, /*flags=*/0,
        /*comm=*/nullptr, s);
    EXPECT_TRUE(isRejected(rSig))
        << "ncclSignal accepted a null comm; got "
        << ncclGetErrorString(rSig);

    ncclWaitSignalDesc_t desc{};
    desc.opCnt = 1;
    desc.peer  = 0;
    ncclResult_t rWait = ncclWaitSignal(
        /*nDesc=*/1, &desc,
        /*comm=*/nullptr, s);
    EXPECT_TRUE(isRejected(rWait))
        << "ncclWaitSignal accepted a null comm; got "
        << ncclGetErrorString(rWait);

    (void)hipFree(devBuf);
}

// 4) End-to-end best-effort: on a single-rank comm with a registered
// window, issue Signal followed by WaitSignal and let the stream drain.
// Implementations are free to require multi-rank topologies, so we
// accept ncclSuccess as well as the standard "not supported here"
// rejection codes - the goal here is to detect a regression that
// turns these APIs into crashes or undefined-error returns rather
// than to assert that single-rank loopback is supported.
static void testSelfSignalAndWaitBestEffort()
{
    SKIP_IF_NO_GPU();

    ncclComm_t comm = nullptr;
    ASSERT_EQ(ncclCommInitAll(&comm, /*nRanks=*/1, /*devList=*/nullptr),
              ncclSuccess);

    (void)hipSetDevice(0);

    constexpr size_t kBytes = 4096;
    void* devBuf = nullptr;
    ASSERT_EQ(ncclMemAlloc(&devBuf, kBytes), ncclSuccess);

    ncclWindow_t win = nullptr;
    ncclResult_t rReg = ncclCommWindowRegister(comm, devBuf, kBytes, &win,
                                               /*winFlags=*/0);

    // If symmetric memory isn't supported on this build/topology, the
    // window registration itself can fail - that's fine for this test;
    // we still verified arg-validation and symbol presence elsewhere.
    if (rReg != ncclSuccess || win == nullptr) {
        (void)ncclMemFree(devBuf);
        (void)ncclCommDestroy(comm);
        GTEST_SKIP() << "ncclCommWindowRegister did not register a window "
                        "on this topology (rReg="
                     << ncclGetErrorString(rReg)
                     << "). Symbol checks and arg-validation already "
                        "exercised the API surface; skipping the "
                        "end-to-end drive.";
        return;
    }

    hipStream_t stream{};
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    // Self-signal: rank 0 signals itself.
    ncclResult_t rSig = ncclSignal(/*peer=*/0, /*sigIdx=*/0, /*ctx=*/0,
                                   /*flags=*/0, comm, stream);

    // Self-wait: pair the wait that consumes the same (peer, sigIdx, ctx).
    ncclWaitSignalDesc_t desc{};
    desc.opCnt  = 1;
    desc.peer   = 0;
    desc.sigIdx = 0;
    desc.ctx    = 0;
    ncclResult_t rWait = ncclWaitSignal(/*nDesc=*/1, &desc, comm, stream);

    // Best-effort acceptance: success means single-rank loopback works;
    // ncclInvalidUsage / ncclInvalidArgument / ncclSystemError means
    // the implementation gated this path, which is acceptable here.
    auto acceptable = [](ncclResult_t r) {
        return r == ncclSuccess        ||
               r == ncclInvalidUsage   ||
               r == ncclInvalidArgument||
               r == ncclSystemError;
    };
    EXPECT_TRUE(acceptable(rSig))
        << "ncclSignal returned an unexpected error: "
        << ncclGetErrorString(rSig);
    EXPECT_TRUE(acceptable(rWait))
        << "ncclWaitSignal returned an unexpected error: "
        << ncclGetErrorString(rWait);

    // Drain the stream if both calls were accepted by NCCL.
    if (rSig == ncclSuccess && rWait == ncclSuccess) {
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess)
            << "Stream did not drain after Signal/WaitSignal.";
    } else {
        // Even if the API rejected the request, the stream itself
        // should still be usable.
        (void)hipStreamSynchronize(stream);
    }

    (void)ncclCommWindowDeregister(comm, win);
    (void)hipStreamDestroy(stream);
    (void)ncclMemFree(devBuf);
    (void)ncclCommDestroy(comm);
}

TEST(OneSidedHostApi, ProcessIsolatedSuite)
{
    RUN_ISOLATED_TESTS(
        ProcessIsolatedTestRunner::TestConfig("SymbolsPresent",
            testSymbolsPresent),
        ProcessIsolatedTestRunner::TestConfig("WaitSignalDescShape",
            testWaitSignalDescShape),
        ProcessIsolatedTestRunner::TestConfig("NullCommRejected",
            testNullCommRejected),
        ProcessIsolatedTestRunner::TestConfig("SelfSignalAndWaitBestEffort",
            testSelfSignalAndWaitBestEffort)
    );
}

} // namespace RcclUnitTesting
