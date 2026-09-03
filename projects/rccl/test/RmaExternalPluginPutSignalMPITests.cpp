/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file RmaExternalPluginPutSignalMPITests.cpp
 * @brief Fire-and-forget invocation check: the one-sided RMA host APIs must
 *        reach an externally loaded GIN/RMA plugin's put/signal primitives.
 *
 * This is intentionally NOT a data-path test. The stub plugin in
 * test/plugin/net_reload_plugin.cpp (ncclRmaPlugin_v13) implements iput /
 * iputSignal / iget as inert no-ops that move no data and raise no signal --
 * a loopback memcpy/atomic cannot service a real cross-rank put/signal because
 * the proxy hands the plugin the *peer's* exchanged handle. Real data-path
 * correctness is validated separately with a real plugin on real IB.
 *
 * What this test proves: calling ncclPutSignal / ncclSignal on the sender
 * actually routes through the external plugin's primitives. The stub records
 * each iput/iputSignal invocation by appending a line to the file named in
 * RCCL_RMA_RELOAD_COUNTER_FILE; after the sender synchronizes its stream we
 * assert that file grew. There is deliberately:
 *   - NO ncclWaitSignal (the receiver never consumes the signal, so a no-op
 *     stub cannot hang the test), and
 *   - NO data / signal validation.
 *
 * Requirements to actually exercise the plugin (supplied by the test-runner
 * entry, see tools/scripts/test_runner/configs/mi300x_mellanox_ib.json):
 *   - NCCL_GIN_ENABLE=1, NCCL_GIN_PLUGIN=STATIC_PLUGIN, and
 *     NCCL_RMA_PLUGIN=STATIC_PLUGIN so the in-binary ncclRmaPlugin_v13
 *     is discovered/loaded/assigned (GIN and RMA have separate plugin
 *     loaders since the v14 GIN/RMA split).
 *   - RCCL_RMA_RELOAD_COUNTER_FILE=<per-node path> so invocations are recorded.
 *   - A cross-LSA-team peer (e.g. 2 nodes x 1 GPU) so the RMA proxy path -- and
 *     therefore the plugin's iput/iputSignal -- is taken rather than the local
 *     copy-engine path. If no external plugin/counter file is configured the
 *     test skips.
 *
 * Run (example):
 *   mpirun -np 2 ./rccl-UnitTestsMPI \
 *     --gtest_filter=RmaExternalPluginPutSignalTest.*
 */

#if defined(MPI_TESTS_ENABLED) && defined(RCCL_ENABLE_HOST_API_TESTS)

#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "HostApiHelpers.hpp"
#include "TestChecks.hpp"

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLHostApiHelpers;

namespace RcclUnitTesting
{

namespace
{
constexpr size_t kTransferSize = 256;             // bytes per PUT (data value is irrelevant)
constexpr size_t kWinSize      = 2 * kTransferSize; // send region + recv region
constexpr size_t kSendOffset   = 0;               // sender carves src from here
constexpr size_t kRecvOffset   = kTransferSize;   // peer's (nominal) dest region
constexpr int    kSigIdx       = 0;
constexpr int    kCtx          = 0;
constexpr unsigned int kFlags  = 0;

inline int winMode()
{
    static int cumem_ = -1;
    if (cumem_ < 0) {
        const char* e = getenv("NCCL_CUMEM_ENABLE");
        cumem_ = e ? (atoi(e) != 0) : true;
    }
    return cumem_ ? NCCL_WIN_COLL_SYMMETRIC : NCCL_WIN_DEFAULT;
}

long countLines(const char* path)
{
    FILE* f = fopen(path, "r");
    if (f == nullptr) return 0;
    long n = 0;
    int c;
    while ((c = fgetc(f)) != EOF)
        if (c == '\n') ++n;
    fclose(f);
    return n;
}

void resetCounterFile(const char* path)
{
    FILE* f = fopen(path, "w");
    if (f != nullptr) fclose(f);
}
} // namespace

// ============================================================================
// Test fixture
// ============================================================================

class RmaExternalPluginPutSignalTest : public MPITestBase
{
protected:
    void SetUp() override
    {
        MPITestBase::SetUp();
        ASSERT_EQ(ncclSuccess, createTestCommunicator());
    }

    int rank() const
    {
        int r = -1;
        ncclCommUserRank(
            const_cast<RmaExternalPluginPutSignalTest*>(this)->getActiveCommunicator(), &r);
        return r;
    }
};

// ============================================================================
// IssuesPutSignalToStub
// ============================================================================

/**
 * @test RmaExternalPluginPutSignalTest.IssuesPutSignalToStub
 * @brief Rank 0 issues ncclPutSignal + ncclSignal to rank 1 (no wait); assert
 *        the external stub plugin's put/signal primitives were invoked.
 */
TEST_F(RmaExternalPluginPutSignalTest, IssuesPutSignalToStub)
{
    const char* counterPath = getenv("RCCL_RMA_RELOAD_COUNTER_FILE");
    if (counterPath == nullptr)
    {
        GTEST_SKIP() << "RCCL_RMA_RELOAD_COUNTER_FILE not set: external RMA stub "
                        "plugin not configured (set NCCL_GIN_PLUGIN=STATIC_PLUGIN, "
                        "NCCL_RMA_PLUGIN=STATIC_PLUGIN, and this env var to run).";
    }

    if (!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    if (myRank == 0)
        resetCounterFile(counterPath);
    MPI_Barrier(MPI_COMM_WORLD);

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, winMode());
    ASSERT_MPI_NE(win, nullptr);
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    ncclResult_t putRes = ncclSuccess;
    ncclResult_t sigRes = ncclSuccess;
    if (myRank == 0)
    {
        void* srcBuf = static_cast<uint8_t*>(winBuf) + kSendOffset;
        putRes = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/kRecvOffset,
            kSigIdx, kCtx, kFlags, comm, stream);
        sigRes = ncclSignal(/*peer=*/1, kSigIdx, kCtx, kFlags, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, putRes);
    ASSERT_MPI_EQ(ncclSuccess, sigRes);
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    bool pluginInvoked = true; // ranks other than 0 pass trivially
    if (myRank == 0)
    {
        long invocations = countLines(counterPath);
        pluginInvoked = (invocations >= 1);
        TEST_INFO("rank 0: external RMA plugin put/signal primitives invoked %ld time(s) "
                  "(expected >= 1). If 0, ensure the peer is cross-LSA-team so the proxy "
                  "path -- not the local copy engine -- services the put/signal.",
                  invocations);
    }
    ASSERT_MPI_TRUE(pluginInvoked);
}

} // namespace RcclUnitTesting

#endif // MPI_TESTS_ENABLED && RCCL_ENABLE_HOST_API_TESTS
