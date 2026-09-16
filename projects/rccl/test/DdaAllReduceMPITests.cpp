/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaAllReduceMPITests.cpp
 * @brief MPI end-to-end tests for DDA AllReduce: the fabric LL tiers and the launch contract
 *
 * Mirrors test/DdaAllReduceTests.cpp but uses MPI + ncclCommInitRank instead of
 * ncclCommInitAll. Correctness is checked against the analytic sum; the COLL log
 * line confirms which LL tier actually ran (fallback paths also sum correctly).
 * One rank per process is also what lets the DDA IPC tier initialise at all, which
 * is why the launch-contract cases for AICOMRCCL-2184 live here.
 *
 * Run (example):
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaMPI_AllReduce.*
 */

#ifdef MPI_TESTS_ENABLED

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "comm.h" // internal: struct ncclComm::doneEvent, see CapturedForkJoinStaysInsideGraph
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace
{
constexpr size_t kOneShotCount = 65536; // 256 KiB f32; below default 1 MiB one-shot threshold

// Default thresholds put two-shot above 1 MiB; 262176 floats is 1 MiB + 128 B and
// satisfies the two-shot shard alignment rules at typical MPI rank counts (see
// DdaFabricEligibilityTests.cpp AllReduceLL_TwoShotClaimsPastOneShotThreshold).
constexpr size_t kTwoShotBaseCount = 262176;

constexpr char kOneShotLogNeedle[] = "taking DDA fabric LL one-shot path";
constexpr char kTwoShotLogNeedle[] = "taking DDA fabric LL two-shot path";

constexpr size_t kContractCount = 1024 * 1024;
constexpr int    kContractIters = 4;

// The addon backends launch on the user's stream themselves and carry the launch contract in their
// wrapper; every other backend reaches doLaunches and gets it there.
bool launchesOnUserStream(int algo)
{
    switch(algo)
    {
    case RCCL_CE_2SHOT:
    case RCCL_DDA_FABRIC_LL:
    case RCCL_DDA_FABRIC_LL128:
    case RCCL_DDA_FABRIC_VMM:
    case RCCL_DDA_IPC:
    case RCCL_HIERARCHICAL_ALLGATHER:
    case RCCL_HIERARCHICAL_REDUCESCATTER:
    case RCCL_GIN_SDMA: return true;
    default: return false;
    }
}

bool isGfx1250Device()
{
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
        return false;
    return std::string(props.gcnArchName).find("gfx1250") != std::string::npos;
}

void fillRankScalar(void* buf, size_t nElem, int rank)
{
    std::vector<float> host(nElem, static_cast<float>(rank + 1));
    HIP_CHECK(hipMemcpy(buf, host.data(), nElem * sizeof(float), hipMemcpyHostToDevice));
}

bool ddaLLTwoShotShapeOk(size_t count, int nRanks)
{
    const size_t bytes = count * sizeof(float);
    if(bytes % static_cast<size_t>(nRanks) != 0)
        return false;
    return (bytes / static_cast<size_t>(nRanks)) % 16 == 0;
}

// Pick a count past the default one-shot threshold whose total bytes satisfy the
// two-shot per-rank shard alignment for the active communicator width.
size_t twoShotCountForRanks(int nRanks)
{
    size_t count = kTwoShotBaseCount;
    if(!ddaLLTwoShotShapeOk(count, nRanks))
    {
        count = (count + 3) & ~size_t(3);
        for(int i = 0; i < 1024; ++i, count += 4)
        {
            if(ddaLLTwoShotShapeOk(count, nRanks))
                return count;
        }
        return 0;
    }
    return count;
}

bool logContainsNeedle(const MPIHelpers::TestLogAssertionContext& logCtx, const char* needle)
{
    const std::string merged = logCtx.readNcclDebugLog() + logCtx.readPerRankStderrLog();
    return merged.find(needle) != std::string::npos;
}
} // namespace

/**
 * @class DdaAllReduceMPITest
 * @brief Shared fixture: COLL log capture and gfx1250 gate for DDA fabric LL tiers.
 */
class DdaAllReduceMPITest : public MPITestBase
{
protected:
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugSubsysGuard_;
    std::unique_ptr<MPIHelpers::TestLogAssertionContext> logCtx_;

    void SetUp() override
    {
        MPITestBase::SetUp();
        debugGuard_       = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG", "INFO");
        debugSubsysGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG_SUBSYS", "NET,INIT,COLL,TUNING");
        logCtx_           = std::make_unique<MPIHelpers::TestLogAssertionContext>(
            MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));
    }

    void TearDown() override
    {
        MPITestBase::TearDown();
        logCtx_.reset();
        debugSubsysGuard_.reset();
        debugGuard_.reset();
    }

    void runAllReduce(size_t count, const char* logNeedle, const char* testId)
    {
        if(!validateTestPrerequisites(kMinProcessesForMPI))
            GTEST_SKIP() << "Need at least 2 MPI ranks";

        //if(!isGfx1250Device())
        //    GTEST_SKIP() << "DDA fabric LL requires gfx1250";

        ASSERT_EQ(ncclSuccess, createTestCommunicator());

        int rank{}, nRanks{};
        ncclCommUserRank(getActiveCommunicator(), &rank);
        ncclCommCount(getActiveCommunicator(), &nRanks);

        const size_t bytes = count * sizeof(float);

        void* sendBuf = nullptr;
        ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);

        void* recvBuf = nullptr;
        ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        fillRankScalar(sendBuf, count, rank);
        ASSERT_EQ(hipSuccess, hipMemset(recvBuf, 0, bytes));

        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(sendBuf, recvBuf, count, ncclFloat32, ncclSum,
                                getActiveCommunicator(), getActiveStream()));
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        const float expectedSum = static_cast<float>(nRanks * (nRanks + 1) / 2);
        ASSERT_TRUE(verifyBufferData<float>(recvBuf, count,
                                            [expectedSum](size_t) { return expectedSum; }))
            << "Rank " << rank << ": DDA LL AllReduce verification failed";

        const bool tookExpectedPath = logContainsNeedle(*logCtx_, logNeedle);
        EXPECT_TRUE(tookExpectedPath)
            << "Rank " << rank << ": " << testId
            << " did not log the expected DDA LL tier (needle: \"" << logNeedle << "\")";

        if(getTestMpiRank() == 0 && tookExpectedPath)
            TEST_INFO("%s: DDA LL path confirmed via COLL log", testId);
    }

    // Rank 0 alone names the backend these operands reach. Every backend has to honour the launch
    // contract, so this reports which one ran rather than requiring a particular one. The query
    // itself runs outside capture and cannot see graph mode, so a captured collective declares it.
    void reportSelectedTier(const char* testId, const void* sendBuf, void* recvBuf, size_t count,
                            int graphCapturing = 0)
    {
        if(getTestMpiRank() != 0)
            return;

        int                algo = -1, protocol = -1, maxChannels = -1;
        const ncclResult_t res
            = rcclGetCollImplInfo(getActiveCommunicator(), ncclFuncAllReduce, count, ncclFloat32,
                                  ncclSum, sendBuf, recvBuf, graphCapturing, &algo, &protocol,
                                  &maxChannels);
        if(res != ncclSuccess)
        {
            TEST_WARN("%s: rcclGetCollImplInfo failed: %s", testId, ncclGetErrorString(res));
            return;
        }

        const char* algoName = nullptr;
        rcclGetAlgoName(algo, &algoName);
        if(launchesOnUserStream(algo))
            TEST_INFO("%s: ran on %s", testId, algoName ? algoName : "?");
        else
            TEST_WARN("%s: ran on %s, which takes the launch contract from doLaunches instead of the"
                      " addon wrapper",
                      testId,
                      algoName ? algoName : "?");
    }
};

class DdaMPI_AllReduce : public DdaAllReduceMPITest
{};

TEST_F(DdaMPI_AllReduce, LLOneShotMultiRank)
{
    runAllReduce(kOneShotCount, kOneShotLogNeedle, "DdaMPI_AllReduce/LLOneShotMultiRank");
}

TEST_F(DdaMPI_AllReduce, LLTwoShotMultiRank)
{
    if(!validateTestPrerequisites(kMinProcessesForMPI))
        GTEST_SKIP() << "Need at least 2 MPI ranks";

    int nRanks = MPIEnvironment::world_size;
    const size_t count = twoShotCountForRanks(nRanks);
    if(count == 0)
        GTEST_SKIP() << "Could not find a two-shot-aligned element count for nRanks="
                     << nRanks;

    runAllReduce(count, kTwoShotLogNeedle, "DdaMPI_AllReduce/LLTwoShotMultiRank");
}

// AICOMRCCL-2184: RCCL selects the collective's device itself, and the device the caller left
// current is not part of the API contract.
TEST_F(DdaMPI_AllReduce, ForeignCurrentDeviceUngrouped)
{
    if(!validateTestPrerequisites(kMinProcessesForMPI))
        GTEST_SKIP() << "Need at least 2 MPI ranks";

    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    if(deviceCount < 2)
        GTEST_SKIP() << "Need at least 2 GPUs visible to each rank; found " << deviceCount;

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    int ownDevice = -1;
    HIP_CHECK(hipGetDevice(&ownDevice));

    const size_t bytes = kContractCount * sizeof(float);

    void* sendBuf = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
    DeviceBufferAutoGuard sendGuard(sendBuf);

    void* recvBuf = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
    DeviceBufferAutoGuard recvGuard(recvBuf);

    fillRankScalar(sendBuf, kContractCount, rank);
    ASSERT_EQ(hipSuccess, hipMemset(recvBuf, 0, bytes));

    const int foreignDevice = (ownDevice + 1) % deviceCount;
    HIP_CHECK(hipSetDevice(foreignDevice));

    const ncclResult_t res = ncclAllReduce(sendBuf, recvBuf, kContractCount, ncclFloat32, ncclSum,
                                           getActiveCommunicator(), getActiveStream());
    EXPECT_EQ(ncclSuccess, res) << "Rank " << rank << " launched with device " << foreignDevice
                                << " current: " << ncclGetErrorString(res);

    int deviceAfterCall = -1;
    HIP_CHECK(hipGetDevice(&deviceAfterCall));
    EXPECT_EQ(foreignDevice, deviceAfterCall)
        << "Rank " << rank << ": AllReduce left device " << deviceAfterCall << " current";

    HIP_CHECK(hipSetDevice(ownDevice));
    ASSERT_EQ(ncclSuccess, res);
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    const float expectedSum = static_cast<float>(nRanks * (nRanks + 1) / 2);
    ASSERT_TRUE(verifyBufferData<float>(recvBuf, kContractCount,
                                        [expectedSum](size_t) { return expectedSum; }))
        << "Rank " << rank << ": AllReduce issued with a foreign current device wrote wrong data";

    reportSelectedTier("DdaMPI_AllReduce/ForeignCurrentDeviceUngrouped", sendBuf, recvBuf,
                       kContractCount);
}

// AICOMRCCL-2184: consecutive collectives on one communicator are ordered against each other even
// when they run on different streams.
TEST_F(DdaMPI_AllReduce, StreamAlternationUngrouped)
{
    if(!validateTestPrerequisites(kMinProcessesForMPI))
        GTEST_SKIP() << "Need at least 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    hipStream_t altStream = nullptr;
    ASSERT_EQ(hipSuccess, hipStreamCreate(&altStream));
    HipStreamAutoGuard altStreamGuard(altStream);

    const size_t bytes = kContractCount * sizeof(float);

    void* bufA = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&bufA, bytes));
    DeviceBufferAutoGuard guardA(bufA);

    void* bufB = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&bufB, bytes));
    DeviceBufferAutoGuard guardB(bufB);

    fillRankScalar(bufA, kContractCount, rank);
    ASSERT_EQ(hipSuccess, hipMemset(bufB, 0, bytes));

    // Each iteration reduces the previous one's output, so the final value depends on the order the
    // iterations ran in. Reducing the same operand every time would pass under any interleaving.
    // The addon backends decline a grouped call, so wrapping the odd iterations in a group puts them
    // on the native path and makes each handoff between the two paths carry the ordering edge.
    for(int iter = 0; iter < kContractIters; ++iter)
    {
        const bool  even   = (iter % 2 == 0);
        hipStream_t stream = even ? getActiveStream() : altStream;
        if(!even)
            ASSERT_EQ(ncclSuccess, ncclGroupStart());
        ncclResult_t res = ncclAllReduce(even ? bufA : bufB, even ? bufB : bufA, kContractCount,
                                         ncclFloat32, ncclSum, getActiveCommunicator(), stream);
        if(!even)
        {
            const ncclResult_t endRes = ncclGroupEnd();
            if(res == ncclSuccess)
                res = endRes;
        }
        ASSERT_EQ(ncclSuccess, res) << "Rank " << rank << ": iteration " << iter;
    }

    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(altStream));

    // Every rank contributes rank+1, so the first reduction yields nRanks*(nRanks+1)/2 and each
    // later one multiplies that by nRanks.
    float expectedSum = static_cast<float>(nRanks * (nRanks + 1) / 2);
    for(int iter = 1; iter < kContractIters; ++iter)
        expectedSum *= static_cast<float>(nRanks);

    void* finalBuf = (kContractIters % 2 == 0) ? bufA : bufB;
    ASSERT_TRUE(verifyBufferData<float>(finalBuf, kContractCount,
                                        [expectedSum](size_t) { return expectedSum; }))
        << "Rank " << rank << ": AllReduce across alternating streams wrote wrong data";

    reportSelectedTier("DdaMPI_AllReduce/StreamAlternationUngrouped", bufA, bufB, kContractCount);
}

// AICOMRCCL-2184: work forked off a captured addon collective belongs to the captured graph.
//
// rcclAddonLaunchBegin offers comm->doneEvent for the collective's last kernel to carry as its
// stop event, but never while the stream is capturing: there it offers nothing and the epilogue
// records separately. This case covers that carve-out, whose failure mode is silent. A fused stop
// event is not bound under capture, so a stream joined into the capture through comm->doneEvent
// never enters the capture at all: its work runs eagerly outside the graph while every call,
// hipStreamEndCapture included, returns success. Work escaping a captured region is a worse
// outcome than a lost ordering edge.
//
// Two simpler formulations were measured on gfx942 and neither of them discriminates, so do not
// reduce this case to one of them:
//   - A cross-stream hipStreamWaitEvent after hipGraphLaunch. RCCL does not order that and no
//     record can: a hipEventRecord issued during capture leaves the event's host-visible state
//     untouched, so a wait outside the graph binds to the last real record, returns success and
//     orders nothing. develop, which carries no fused stop event anywhere, fails such a case
//     identically.
//   - A fork on an event this test records itself during the capture. That event is bound either
//     way, giving the same node and edge counts, so it cannot observe whether the collective's
//     own event was bound. Only comm->doneEvent can, which is why the communicator is opened up
//     below.
//
// The assertion is on the graph's shape rather than on the data, because escaped work still
// computes the right answer whenever it happens to run late enough.
TEST_F(DdaMPI_AllReduce, CapturedForkJoinStaysInsideGraph)
{
    if(!validateTestPrerequisites(kMinProcessesForMPI))
        GTEST_SKIP() << "Need at least 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    hipStream_t sideStream = nullptr;
    ASSERT_EQ(hipSuccess, hipStreamCreate(&sideStream));
    HipStreamAutoGuard sideStreamGuard(sideStream);

    hipEvent_t joinEvent = nullptr;
    ASSERT_EQ(hipSuccess, hipEventCreateWithFlags(&joinEvent, hipEventDisableTiming));
    HipEventAutoGuard joinEventGuard(joinEvent);

    const size_t bytes = kContractCount * sizeof(float);

    void* bufA = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&bufA, bytes));
    DeviceBufferAutoGuard guardA(bufA);

    void* bufB = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&bufB, bytes));
    DeviceBufferAutoGuard guardB(bufB);

    void* bufC = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&bufC, bytes));
    DeviceBufferAutoGuard guardC(bufC);

    fillRankScalar(bufA, kContractCount, rank);
    ASSERT_EQ(hipSuccess, hipMemset(bufB, 0, bytes));
    ASSERT_EQ(hipSuccess, hipMemset(bufC, 0, bytes));

    // How many nodes the collective itself contributes depends on the tier that serves it, so the
    // baseline is measured rather than assumed: one capture of the collective on its own. Capture
    // turns a hipEventRecord into a dependency marker instead of a node, so this count is the same
    // whether the epilogue records or the stop event is fused.
    ASSERT_EQ(hipSuccess, hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal));
    const ncclResult_t baseRes = ncclAllReduce(bufA, bufB, kContractCount, ncclFloat32, ncclSum,
                                               getActiveCommunicator(), getActiveStream());

    // End every capture before judging its result: returning from an assertion while the fixture
    // stream is still capturing would break every later test in the process.
    hipGraph_t       baseGraph = nullptr;
    const hipError_t baseEnd   = hipStreamEndCapture(getActiveStream(), &baseGraph);
    ASSERT_EQ(ncclSuccess, baseRes)
        << "Rank " << rank << ": capturing the baseline AllReduce failed: "
        << ncclGetErrorString(baseRes);
    ASSERT_EQ(hipSuccess, baseEnd);
    ASSERT_NE(nullptr, baseGraph);
    auto baseGraphGuard = makeScopeGuard([&]() { (void)hipGraphDestroy(baseGraph); });

    size_t baseNodes = 0, baseEdges = 0;
    ASSERT_EQ(hipSuccess, hipGraphGetNodes(baseGraph, nullptr, &baseNodes));
    ASSERT_EQ(hipSuccess, hipGraphGetEdges(baseGraph, nullptr, nullptr, &baseEdges));
    ASSERT_GT(baseNodes, 0u) << "Rank " << rank << ": the captured AllReduce produced no graph node";

    // comm->doneEvent is the event the addon bracket records, and reaching for it is the whole
    // point: it is the only handle on whether that record was bound to this capture.
    hipEvent_t doneEvent = getActiveCommunicator()->doneEvent;
    ASSERT_NE(nullptr, doneEvent);

    ASSERT_EQ(hipSuccess, hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal));
    const ncclResult_t forkRes = ncclAllReduce(bufA, bufB, kContractCount, ncclFloat32, ncclSum,
                                               getActiveCommunicator(), getActiveStream());
    const hipError_t forkWaitRes = hipStreamWaitEvent(sideStream, doneEvent, 0);
    const hipError_t sideWorkRes
        = hipMemcpyAsync(bufC, bufB, bytes, hipMemcpyDeviceToDevice, sideStream);
    const hipError_t joinRecordRes = hipEventRecord(joinEvent, sideStream);
    const hipError_t joinWaitRes   = hipStreamWaitEvent(getActiveStream(), joinEvent, 0);

    hipGraph_t       graph      = nullptr;
    const hipError_t endCapture = hipStreamEndCapture(getActiveStream(), &graph);

    ASSERT_EQ(ncclSuccess, forkRes)
        << "Rank " << rank << ": capturing the AllReduce failed: " << ncclGetErrorString(forkRes);
    ASSERT_EQ(hipSuccess, forkWaitRes);
    ASSERT_EQ(hipSuccess, sideWorkRes);
    ASSERT_EQ(hipSuccess, joinRecordRes);
    ASSERT_EQ(hipSuccess, joinWaitRes);
    ASSERT_EQ(hipSuccess, endCapture);
    ASSERT_NE(nullptr, graph);
    auto graphGuard = makeScopeGuard([&]() { (void)hipGraphDestroy(graph); });

    size_t nodes = 0, edges = 0;
    ASSERT_EQ(hipSuccess, hipGraphGetNodes(graph, nullptr, &nodes));
    ASSERT_EQ(hipSuccess, hipGraphGetEdges(graph, nullptr, nullptr, &edges));

    // The forked copy is one node depending on the collective, so both counts rise by exactly one.
    // Unchanged counts mean the copy never joined the capture and ran outside the graph.
    EXPECT_EQ(baseNodes + 1, nodes)
        << "Rank " << rank << ": the copy forked off the captured AllReduce is not in the graph ("
        << baseNodes << " nodes for the collective alone, " << nodes << " with the fork)";
    EXPECT_EQ(baseEdges + 1, edges)
        << "Rank " << rank << ": the fork added no edge to the graph (" << baseEdges
        << " edges for the collective alone, " << edges << " with the fork)";

    // Replaying is a second, weaker check on the same edge: a copy ordered after the collective
    // sees its result, while one that escaped copied whatever bufB held at capture time.
    hipGraphExec_t graphExec = nullptr;
    ASSERT_EQ(hipSuccess, hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    auto graphExecGuard = makeScopeGuard([&]() { (void)hipGraphExecDestroy(graphExec); });

    ASSERT_EQ(hipSuccess, hipGraphLaunch(graphExec, getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    const float expectedSum = static_cast<float>(nRanks * (nRanks + 1) / 2);
    EXPECT_TRUE(verifyBufferData<float>(bufC, kContractCount,
                                        [expectedSum](size_t) { return expectedSum; }))
        << "Rank " << rank << ": the copy forked off the captured AllReduce read stale data";

    reportSelectedTier("DdaMPI_AllReduce/CapturedForkJoinStaysInsideGraph", bufA, bufB,
                       kContractCount, /*graphCapturing=*/1);
}

// AICOMRCCL-2184: a captured addon collective replays correctly, and the application can order a
// collective on another stream after the replay with its own event.
//
// Nothing else covers an addon collective inside a graph, which is why this stays. It asserts only
// what holds: the ordering edge is the test's own hipEventRecord after hipGraphLaunch, issued
// outside any capture, and not comm->doneEvent. RCCL cannot supply that edge from inside the
// capture, for the reason spelled out on CapturedForkJoinStaysInsideGraph above.
TEST_F(DdaMPI_AllReduce, CapturedReplayOrderedByAppEvent)
{
    if(!validateTestPrerequisites(kMinProcessesForMPI))
        GTEST_SKIP() << "Need at least 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    hipStream_t foreignStream = nullptr;
    ASSERT_EQ(hipSuccess, hipStreamCreate(&foreignStream));
    HipStreamAutoGuard foreignStreamGuard(foreignStream);

    hipEvent_t replayEvent = nullptr;
    ASSERT_EQ(hipSuccess, hipEventCreateWithFlags(&replayEvent, hipEventDisableTiming));
    HipEventAutoGuard replayEventGuard(replayEvent);

    const size_t bytes = kContractCount * sizeof(float);

    void* bufA = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&bufA, bytes));
    DeviceBufferAutoGuard guardA(bufA);

    void* bufB = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&bufB, bytes));
    DeviceBufferAutoGuard guardB(bufB);

    fillRankScalar(bufA, kContractCount, rank);
    ASSERT_EQ(hipSuccess, hipMemset(bufB, 0, bytes));

    ASSERT_EQ(hipSuccess, hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal));
    const ncclResult_t captureRes = ncclAllReduce(bufA, bufB, kContractCount, ncclFloat32, ncclSum,
                                                  getActiveCommunicator(), getActiveStream());

    hipGraph_t       graph         = nullptr;
    const hipError_t endCaptureRes = hipStreamEndCapture(getActiveStream(), &graph);
    ASSERT_EQ(ncclSuccess, captureRes)
        << "Rank " << rank << ": capturing the AllReduce failed: " << ncclGetErrorString(captureRes);
    ASSERT_EQ(hipSuccess, endCaptureRes);
    ASSERT_NE(nullptr, graph);
    auto graphGuard = makeScopeGuard([&]() { (void)hipGraphDestroy(graph); });

    hipGraphExec_t graphExec = nullptr;
    ASSERT_EQ(hipSuccess, hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    auto graphExecGuard = makeScopeGuard([&]() { (void)hipGraphExecDestroy(graphExec); });

    ASSERT_EQ(hipSuccess, hipGraphLaunch(graphExec, getActiveStream()));
    ASSERT_EQ(hipSuccess, hipEventRecord(replayEvent, getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamWaitEvent(foreignStream, replayEvent, 0));
    ASSERT_EQ(ncclSuccess,
              ncclAllReduce(bufB, bufA, kContractCount, ncclFloat32, ncclSum,
                            getActiveCommunicator(), foreignStream));

    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(foreignStream));

    const float replaySum   = static_cast<float>(nRanks * (nRanks + 1) / 2);
    const float expectedSum = replaySum * static_cast<float>(nRanks);
    ASSERT_TRUE(verifyBufferData<float>(bufA, kContractCount,
                                        [expectedSum](size_t) { return expectedSum; }))
        << "Rank " << rank << ": the AllReduce after the graph replay read unordered data";

    reportSelectedTier("DdaMPI_AllReduce/CapturedReplayOrderedByAppEvent (captured)", bufA, bufB,
                       kContractCount, /*graphCapturing=*/1);
    reportSelectedTier("DdaMPI_AllReduce/CapturedReplayOrderedByAppEvent (foreign stream)", bufB,
                       bufA, kContractCount);
}

#endif // MPI_TESTS_ENABLED
