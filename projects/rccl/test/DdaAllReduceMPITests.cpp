/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaAllReduceMPITests.cpp
 * @brief MPI end-to-end tests for DDA AllReduce: the fabric LL and LL128 tiers and the
 *        launch contract
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

#include "DdaFabricMPITestFixture.hpp"
#include "DdaMPITestHelpers.hpp"
#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "ce_coll.h" // internal: NCCL_CE_REDUCE_THREADS and ncclCeImplemented, see DdaMPI_AllReduceRegistered
#include "comm.h" // internal: struct ncclComm::doneEvent, see CapturedForkJoinStaysInsideGraph
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;
using namespace DdaMPITestHelpers;

namespace
{
constexpr char kOneShotLogNeedle[] = "taking DDA fabric LL one-shot path";
constexpr char kTwoShotLogNeedle[] = "taking DDA fabric LL two-shot path";

constexpr char kLL128OneShotLogNeedle[] = "taking DDA fabric LL128 one-shot path";
constexpr char kLL128TwoShotLogNeedle[] = "taking DDA fabric LL128 two-shot path";

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

void fillRankScalar(void* buf, size_t nElem, int rank, float scale = 1.0f)
{
    std::vector<float> host(nElem, static_cast<float>(rank + 1) * scale);
    HIP_CHECK(hipMemcpy(buf, host.data(), nElem * sizeof(float), hipMemcpyHostToDevice));
}

// Rank i%n holds +big, the next -big, the rest 1.0, so the sum depends on the fold order.
std::vector<uint8_t> makeOrderSensitiveInput(ncclDataType_t type, size_t count, int rank, int nRanks)
{
    const float big = (type == ncclBfloat16) ? 1024.0f : 33554432.0f;
    std::vector<uint8_t> host(count * (type == ncclBfloat16 ? sizeof(uint16_t) : sizeof(float)));
    for(size_t i = 0; i < count; ++i)
    {
        const int owner = static_cast<int>(i % static_cast<size_t>(nRanks));
        float     v     = 1.0f;
        if(rank == owner)
        {
            v = big;
        }
        else if(rank == (owner + 1) % nRanks)
        {
            v = -big;
        }
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        if(type == ncclBfloat16)
        {
            // Every value here is a power of two, so dropping the low half is the exact bf16 value.
            const uint16_t hi = static_cast<uint16_t>(bits >> 16);
            std::memcpy(host.data() + i * sizeof(hi), &hi, sizeof(hi));
        }
        else
        {
            std::memcpy(host.data() + i * sizeof(bits), &bits, sizeof(bits));
        }
    }
    return host;
}

// Kernel, copy and memset nodes of a graph. The captures below enqueue no work of their own, so
// these are the collectives' nodes.
std::vector<hipGraphNode_t> graphWorkNodes(hipGraph_t graph)
{
    size_t count = 0;
    EXPECT_EQ(hipSuccess, hipGraphGetNodes(graph, nullptr, &count));
    std::vector<hipGraphNode_t> all(count), work;
    EXPECT_EQ(hipSuccess, hipGraphGetNodes(graph, all.data(), &count));
    for(hipGraphNode_t node : all)
    {
        hipGraphNodeType type{};
        EXPECT_EQ(hipSuccess, hipGraphNodeGetType(node, &type));
        if(type == hipGraphNodeTypeKernel || type == hipGraphNodeTypeMemcpy
           || type == hipGraphNodeTypeMemset)
            work.push_back(node);
    }
    return work;
}

bool graphPathExists(hipGraphNode_t from, hipGraphNode_t to)
{
    std::vector<hipGraphNode_t> pending{from};
    std::set<hipGraphNode_t>    seen{from};
    while(!pending.empty())
    {
        hipGraphNode_t node = pending.back();
        pending.pop_back();
        size_t count = 0;
        EXPECT_EQ(hipSuccess, hipGraphNodeGetDependentNodes(node, nullptr, &count));
        std::vector<hipGraphNode_t> next(count);
        EXPECT_EQ(hipSuccess, hipGraphNodeGetDependentNodes(node, next.data(), &count));
        for(hipGraphNode_t n : next)
        {
            if(n == to)
                return true;
            if(seen.insert(n).second)
                pending.push_back(n);
        }
    }
    return false;
}

size_t countEventNodes(hipGraph_t graph, hipGraphNodeType type, hipEvent_t event)
{
    size_t count = 0;
    EXPECT_EQ(hipSuccess, hipGraphGetNodes(graph, nullptr, &count));
    std::vector<hipGraphNode_t> all(count);
    EXPECT_EQ(hipSuccess, hipGraphGetNodes(graph, all.data(), &count));
    size_t matches = 0;
    for(hipGraphNode_t node : all)
    {
        hipGraphNodeType nodeType{};
        EXPECT_EQ(hipSuccess, hipGraphNodeGetType(node, &nodeType));
        if(nodeType != type)
            continue;
        hipEvent_t nodeEvent = nullptr;
        EXPECT_EQ(hipSuccess, type == hipGraphNodeTypeWaitEvent
                                  ? hipGraphEventWaitNodeGetEvent(node, &nodeEvent)
                                  : hipGraphEventRecordNodeGetEvent(node, &nodeEvent));
        if(nodeEvent == event)
            ++matches;
    }
    return matches;
}
} // namespace

/**
 * @class DdaAllReduceMPITest
 * @brief Shared fixture: COLL log capture and gfx1250 gate for DDA fabric LL tiers.
 */
class DdaAllReduceMPITest : public DdaFabricMPITest
{
protected:
    void runAllReduce(size_t count, const char* logNeedle, const char* testId)
    {
        if(!validateTestPrerequisites(kMinProcessesForMPI))
            GTEST_SKIP() << "Need at least 2 MPI ranks";

        DDA_FABRIC_SKIP_OR_FAIL(archSkipReason("DDA fabric LL requires gfx1250"));

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

        DDA_FABRIC_SKIP_OR_FAIL(fabricInitSkipReason());

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

        // With fabric DDA up, RCCL must offer an LL tier for this message; a non-fabric pick is a selection regression.
        int                algo = -1, protocol = -1, maxChannels = -1;
        const ncclResult_t query
            = rcclGetCollImplInfo(getActiveCommunicator(), ncclFuncAllReduce, count, ncclFloat32,
                                  ncclSum, sendBuf, recvBuf, /*graphCapturing=*/0, &algo, &protocol,
                                  &maxChannels);
        EXPECT_EQ(ncclSuccess, query) << "Rank " << rank << ": rcclGetCollImplInfo failed";
        const char* algoName = nullptr;
        rcclGetAlgoName(algo, &algoName);
        EXPECT_TRUE(algo == RCCL_DDA_FABRIC_LL || algo == RCCL_DDA_FABRIC_LL128)
            << "Rank " << rank << ": " << testId << " selected " << (algoName ? algoName : "?")
            << " (algo " << algo << "), not DDA fabric LL/LL128";

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

        const bool tookExpectedPath = logContainsNeedle(logNeedle);
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

    // Every rank must get the same bits back; logNeedle pins the DDA tier, nullptr allows any.
    void runBitIdenticalAcrossRanks(const std::vector<ncclDataType_t>& types, const std::vector<size_t>& sizes,
                                    const char* logNeedle, const char* testId)
    {
        if(!validateTestPrerequisites(kMinProcessesForMPI))
        {
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        }

        // a + b == b + a exactly, so only a third operand can expose a rank-dependent fold order.
        const std::string rankSkip
            = mpiCoordinatedSkipReason(MPIEnvironment::world_size < 3, "Need at least 3 MPI ranks");
        if(!rankSkip.empty())
        {
            GTEST_SKIP() << rankSkip;
        }

        if(logNeedle != nullptr)
        {
            DDA_FABRIC_SKIP_OR_FAIL(archSkipReason("DDA fabric LL requires gfx1250"));
        }

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

        if(logNeedle != nullptr)
        {
            DDA_FABRIC_SKIP_OR_FAIL(fabricInitSkipReason());
        }

        int rank{}, nRanks{};
        ncclCommUserRank(getActiveCommunicator(), &rank);
        ncclCommCount(getActiveCommunicator(), &nRanks);

        for(const ncclDataType_t type : types)
        {
            for(const size_t bytes : sizes)
            {
                const size_t eltSize = (type == ncclBfloat16) ? sizeof(uint16_t) : sizeof(float);
                const size_t count   = bytes / eltSize;
                SCOPED_TRACE(std::string(testId) + " type=" + std::to_string(static_cast<int>(type))
                             + " bytes=" + std::to_string(bytes));

                void* sendBuf = nullptr;
                ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
                DeviceBufferAutoGuard sendGuard(sendBuf);
                void* recvBuf = nullptr;
                ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
                DeviceBufferAutoGuard recvGuard(recvBuf);

                const std::vector<uint8_t> input = makeOrderSensitiveInput(type, count, rank, nRanks);
                ASSERT_MPI_EQ(hipSuccess, hipMemcpy(sendBuf, input.data(), bytes, hipMemcpyHostToDevice));
                // A per-rank fill means an output the kernel never wrote cannot compare equal across ranks.
                ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, rank + 1, bytes));

                if(rank == 0)
                {
                    int                algo = -1, protocol = -1, maxChannels = -1;
                    const char*        algoName = nullptr;
                    const ncclResult_t query
                        = rcclGetCollImplInfo(getActiveCommunicator(), ncclFuncAllReduce, count, type, ncclSum,
                                              sendBuf, recvBuf, /*graphCapturing=*/0, &algo, &protocol,
                                              &maxChannels);
                    if(query == ncclSuccess)
                    {
                        rcclGetAlgoName(algo, &algoName);
                    }
                    TEST_INFO("%s: type %d, %zu bytes ran on %s", testId, static_cast<int>(type), bytes,
                              algoName ? algoName : "?");
                }

                ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(sendBuf, recvBuf, count, type, ncclSum,
                                                         getActiveCommunicator(), getActiveStream()));
                ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

                std::vector<uint8_t> mine(bytes);
                ASSERT_MPI_EQ(hipSuccess, hipMemcpy(mine.data(), recvBuf, bytes, hipMemcpyDeviceToHost));
                std::vector<uint8_t> ref = mine;
                ASSERT_EQ(MPI_SUCCESS, MPI_Bcast(ref.data(), static_cast<int>(bytes), MPI_BYTE, 0, MPI_COMM_WORLD));

                int64_t mismatches = 0;
                size_t  firstBad   = count;
                for(size_t i = 0; i < count; ++i)
                {
                    if(std::memcmp(mine.data() + i * eltSize, ref.data() + i * eltSize, eltSize) != 0)
                    {
                        if(mismatches == 0)
                        {
                            firstBad = i;
                        }
                        ++mismatches;
                    }
                }
                int64_t totalMismatches = mismatches;
                MPI_Allreduce(MPI_IN_PLACE, &totalMismatches, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

                EXPECT_EQ(0, mismatches) << "Rank " << rank << ": " << mismatches << " of " << count
                                         << " elements differ from rank 0, first at index " << firstBad;
                EXPECT_EQ(0, totalMismatches) << "Rank " << rank << ": AllReduce output is not bit-identical"
                                              << " across ranks (" << totalMismatches << " element mismatches)";
            }
        }

        if(logNeedle != nullptr)
        {
            EXPECT_TRUE(logContainsNeedle(logNeedle))
                << "Rank " << rank << ": " << testId << " did not log the expected DDA tier (needle: \""
                << logNeedle << "\")";
        }
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

// 1 MiB of f32 is the largest LL one-shot message, which from 5 ranks runs as the spread two-shot on its grid.
TEST_F(DdaMPI_AllReduce, LLSpreadTwoShotMultiRank)
{
    const char* needle = MPIEnvironment::world_size < 5 ? kOneShotLogNeedle : "DDA fabric AllReduce LL spread two-shot";
    runAllReduce(kDefaultLLOneShotThresholdBytes / sizeof(float), needle, "DdaMPI_AllReduce/LLSpreadTwoShotMultiRank");
}

// The LL128 tiers sit behind the LL ones in ncclAllReduceDdaFabricLL, each gated
// on its own threshold. Under the defaults the LL tiers claim everything up to
// 16 MiB while an LL128 slot tops out near the same size, which leaves LL128 a
// few hundred bytes of window at 4 ranks and none at 8. Rather than chase that,
// the two below zero the thresholds of the tiers ahead so the size under test
// lands unambiguously on the tier being exercised.
//
// IMPORTANT: RCCL_PARAM caches on first read, per process, so whichever test in
// the binary reads a threshold first fixes it for every later one -- including
// the LL tests above. Each of these has to run in its own process:
//
//   mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter=DdaMPI_AllReduce.LL128OneShot*
//   mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter=DdaMPI_AllReduce.LL128TwoShot*
//   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaMPI_AllReduce.LL128StagedWire*
//
// Run in a process that already touched those thresholds, the guards are inert
// and the COLL-log assertion reports whichever tier actually claimed the message.

// 256 KiB of f32 is inside the 4 MiB LL128 one-shot threshold and inside a slot
// at any supported rank count, so with the LL tiers switched off it is the LL128
// one-shot tier that claims it.
TEST_F(DdaMPI_AllReduce, LL128OneShotMultiRank)
{
    MPIHelpers::MpiEnvGuard llOneShotGuard("RCCL_DDA_LL_ONESHOT_THRESHOLD", "0");
    MPIHelpers::MpiEnvGuard llTwoShotGuard("RCCL_DDA_LL_TWOSHOT_THRESHOLD", "0");

    runAllReduce(kOneShotCount, kLL128OneShotLogNeedle, "DdaMPI_AllReduce/LL128OneShotMultiRank");
}

// Zeroing the LL128 one-shot threshold as well leaves the two-shot tier as the
// only claimant. The count reuses the LL two-shot shape rules -- the message has
// to divide into per-rank shards and each shard has to be a whole number of
// 16-byte chunks -- which the LL128 two-shot tier applies identically.
TEST_F(DdaMPI_AllReduce, LL128TwoShotMultiRank)
{
    if(!validateTestPrerequisites(kMinProcessesForMPI))
        GTEST_SKIP() << "Need at least 2 MPI ranks";

    MPIHelpers::MpiEnvGuard llOneShotGuard("RCCL_DDA_LL_ONESHOT_THRESHOLD", "0");
    MPIHelpers::MpiEnvGuard llTwoShotGuard("RCCL_DDA_LL_TWOSHOT_THRESHOLD", "0");
    MPIHelpers::MpiEnvGuard ll128OneShotGuard("RCCL_DDA_LL128_ONESHOT_THRESHOLD", "0");

    int          nRanks = MPIEnvironment::world_size;
    const size_t count  = twoShotCountForRanks(nRanks);
    if(count == 0)
        GTEST_SKIP() << "Could not find a two-shot-aligned element count for nRanks=" << nRanks;

    runAllReduce(count, kLL128TwoShotLogNeedle, "DdaMPI_AllReduce/LL128TwoShotMultiRank");
}

// RCCL_DDA_LL128_ASYNC_WIRE=1: staged writer for 2 MiB+ shards on two ranks, peer-split from three; own process.
// Integer sums are exact in any fold order, so bit for bit the cooperative writer's; the COLL log names each writer.
TEST_F(DdaMPI_AllReduce, LL128StagedWireTwoShotMultiRank)
{
    if(!validateTestPrerequisites(kMinProcessesForMPI))
    {
        GTEST_SKIP() << "Need at least 2 MPI ranks";
    }

    MPIHelpers::MpiEnvGuard llOneShotGuard("RCCL_DDA_LL_ONESHOT_THRESHOLD", "0");
    MPIHelpers::MpiEnvGuard llTwoShotGuard("RCCL_DDA_LL_TWOSHOT_THRESHOLD", "0");
    MPIHelpers::MpiEnvGuard ll128OneShotGuard("RCCL_DDA_LL128_ONESHOT_THRESHOLD", "0");
    MPIHelpers::MpiEnvGuard asyncWireGuard("RCCL_DDA_LL128_ASYNC_WIRE", "1");

    DDA_FABRIC_SKIP_OR_FAIL(archSkipReason("DDA fabric LL128 requires gfx1250"));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    DDA_FABRIC_SKIP_OR_FAIL(fabricInitSkipReason());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    // A 1 MiB shard keeps the cooperative writer; a 2 MiB shard (16 MiB at 8 ranks) takes the staged one.
    constexpr size_t kStagedMinShardBytes = size_t(2) * 1024 * 1024;
    for(const size_t shardBytes : {kStagedMinShardBytes / 2, kStagedMinShardBytes})
    {
        const size_t count = shardBytes / sizeof(float) * static_cast<size_t>(nRanks);
        const size_t bytes = count * sizeof(float);
        SCOPED_TRACE("shardBytes=" + std::to_string(shardBytes));

        // Index-dependent integers below 251: every partial sum stays exact in f32.
        auto value = [](size_t i, int r) { return static_cast<float>((i * 7 + static_cast<size_t>(r) * 13) % 251); };
        std::vector<float> host(count);
        for(size_t i = 0; i < count; ++i)
        {
            host[i] = value(i, rank);
        }

        void* sendBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);
        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);
        ASSERT_MPI_EQ(hipSuccess, hipMemcpy(sendBuf, host.data(), bytes, hipMemcpyHostToDevice));
        ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0xFF, bytes));

        ASSERT_MPI_EQ(ncclSuccess, ncclAllReduce(sendBuf, recvBuf, count, ncclFloat32, ncclSum,
                                                 getActiveCommunicator(), getActiveStream()));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        ASSERT_MPI_EQ(hipSuccess, hipMemcpy(host.data(), recvBuf, bytes, hipMemcpyDeviceToHost));

        int64_t bad      = 0;
        size_t  firstBad = count;
        for(size_t i = 0; i < count; ++i)
        {
            float want = 0.0f;
            for(int r = 0; r < nRanks; ++r)
            {
                want += value(i, r);
            }
            if(std::memcmp(&host[i], &want, sizeof(want)) != 0)
            {
                if(bad == 0)
                {
                    firstBad = i;
                }
                ++bad;
            }
        }
        EXPECT_EQ(0, bad) << "Rank " << rank << ": " << bad << " of " << count << " elements wrong, first at "
                          << firstBad;
    }

    EXPECT_TRUE(logContainsNeedle(kLL128TwoShotLogNeedle)) << "Rank " << rank << ": no LL128 two-shot launch";
    if(nRanks > 2)
    {
        EXPECT_GE(countLogNeedle("(peer-split"), 2u) << "Rank " << rank << ": a message missed the peer-split";
    }
    else
    {
        EXPECT_TRUE(logContainsNeedle("wire=0 lds=0")) << "Rank " << rank << ": 1 MiB shards left the coop writer";
        EXPECT_TRUE(logContainsNeedle("wire=1 lds=")) << "Rank " << rank << ": 2 MiB shards missed the staged writer";
    }
}

// Whatever backend the default selection picks for 256 KiB to 4 MiB, every rank must get the same bits back.
TEST_F(DdaMPI_AllReduce, BitIdenticalAcrossRanksDefaultSelection)
{
    runBitIdenticalAcrossRanks({ncclFloat32, ncclBfloat16}, {256 * 1024, 1024 * 1024, 4 * 1024 * 1024},
                               /*logNeedle=*/nullptr, "DdaMPI_AllReduce/BitIdenticalAcrossRanksDefaultSelection");
}

// 256 KiB is under the default LL one-shot threshold, so the LL one-shot kernel produces every byte compared.
TEST_F(DdaMPI_AllReduce, LLOneShotBitIdenticalAcrossRanks)
{
    runBitIdenticalAcrossRanks({ncclFloat32, ncclBfloat16}, {256 * 1024}, kOneShotLogNeedle,
                               "DdaMPI_AllReduce/LLOneShotBitIdenticalAcrossRanks");
}

// Fresh comms, staggered arrival, two LL calls on a non-blocking stream right after init: a late init zero fill hangs.
TEST_F(DdaMPI_AllReduce, LLFreshCommNonBlockingStream)
{
    if(!validateTestPrerequisites(kMinProcessesForMPI))
        GTEST_SKIP() << "Need at least 2 MPI ranks";

    DDA_FABRIC_SKIP_OR_FAIL(archSkipReason("DDA fabric LL requires gfx1250"));

    constexpr int    kIters        = 32;
    constexpr int    kMaxStaggerMs = 100;
    constexpr size_t kCount        = 12288; // 48 KiB of f32: one LL one-shot launch that finishes fast
    const size_t     bytes         = kCount * sizeof(float);

    const int rank   = MPIEnvironment::world_rank;
    const int nRanks = MPIEnvironment::world_size;

    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HipStreamAutoGuard streamGuard(stream);

    void *sendA = nullptr, *sendB = nullptr, *recvA = nullptr, *recvB = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendA, bytes));
    DeviceBufferAutoGuard sendAGuard(sendA);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendB, bytes));
    DeviceBufferAutoGuard sendBGuard(sendB);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvA, bytes));
    DeviceBufferAutoGuard recvAGuard(recvA);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvB, bytes));
    DeviceBufferAutoGuard recvBGuard(recvB);

    std::vector<float> host(kCount);
    const float        rankSum = static_cast<float>(nRanks * (nRanks + 1) / 2);

    std::mt19937                       rng(1234u + static_cast<unsigned>(rank));
    std::uniform_int_distribution<int> staggerMs(0, kMaxStaggerMs);

    for(int iter = 0; iter < kIters; ++iter)
    {
        SCOPED_TRACE("iteration " + std::to_string(iter));

        // Per-iteration values, so stale packets from an earlier comm can't pass the check.
        const float scaleA = static_cast<float>(2 * iter + 1);
        const float scaleB = static_cast<float>(2 * iter + 2);
        fillRankScalar(sendA, kCount, rank, scaleA);
        fillRankScalar(sendB, kCount, rank, scaleB);
        const float sumA = rankSum * scaleA;
        const float sumB = rankSum * scaleB;

        ncclUniqueId id{};
        if(rank == 0)
            ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
        ASSERT_EQ(MPI_SUCCESS, MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

        // Stream-ordered, so it cannot drain the null stream the way a blocking call after init would.
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(recvA, 0, bytes, stream));
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(recvB, 0, bytes, stream));

        std::this_thread::sleep_for(std::chrono::milliseconds(staggerMs(rng)));

        ncclComm_t comm = nullptr;
        ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comm, nRanks, id, rank));
        auto commGuard = makeScopeGuard([&]() { (void)ncclCommDestroy(comm); });

        // Nothing between init and these launches may synchronize the device or the ranks.
        const ncclResult_t launchA = ncclAllReduce(sendA, recvA, kCount, ncclFloat32, ncclSum, comm, stream);
        const ncclResult_t launchB = ncclAllReduce(sendB, recvB, kCount, ncclFloat32, ncclSum, comm, stream);
        hipError_t         drained = DdaFabricMPI::waitStreamBounded(stream, std::chrono::seconds(30));
        if(launchA != ncclSuccess || launchB != ncclSuccess)
        {
            ADD_FAILURE() << "Rank " << rank << ": ncclAllReduce failed: " << ncclGetErrorString(launchA) << ", "
                          << ncclGetErrorString(launchB);
            if(drained == hipSuccess)
                drained = hipErrorLaunchFailure;
        }
        DdaFabricMPI::abortJobUnlessAllRanksDrained(drained, "LL AllReduce pair on a fresh communicator");

        if(iter == 0)
            DDA_FABRIC_SKIP_OR_FAIL(fabricInitSkipReason(comm));

        int                algo = -1, protocol = -1, maxChannels = -1;
        const ncclResult_t query = rcclGetCollImplInfo(comm, ncclFuncAllReduce, kCount, ncclFloat32, ncclSum, sendA,
                                                       recvA, /*graphCapturing=*/0, &algo, &protocol, &maxChannels);
        EXPECT_EQ(ncclSuccess, query) << "Rank " << rank << ": rcclGetCollImplInfo failed";
        EXPECT_EQ(RCCL_DDA_FABRIC_LL, algo) << "Rank " << rank << ": this comm did not select DDA fabric LL";

        int64_t bad = 0;
        ASSERT_MPI_EQ(hipSuccess, hipMemcpy(host.data(), recvA, bytes, hipMemcpyDeviceToHost));
        bad += std::count_if(host.begin(), host.end(), [sumA](float v) { return v != sumA; });
        ASSERT_MPI_EQ(hipSuccess, hipMemcpy(host.data(), recvB, bytes, hipMemcpyDeviceToHost));
        const int64_t badB = std::count_if(host.begin(), host.end(), [sumB](float v) { return v != sumB; });
        bad += badB;
        EXPECT_EQ(0, bad) << "Rank " << rank << ": " << bad << " wrong elements (" << badB
                          << " in the second AllReduce, first sample " << host[0] << ", expected " << sumB << ")";

        int64_t totalBad = bad + ((query == ncclSuccess && algo == RCCL_DDA_FABRIC_LL) ? 0 : 1);
        MPI_Allreduce(MPI_IN_PLACE, &totalBad, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
        if(totalBad != 0)
            break;
    }

    EXPECT_TRUE(logContainsNeedle(kOneShotLogNeedle))
        << "Rank " << rank << ": the AllReduce pairs did not take the DDA fabric LL one-shot path";
}

// Sets the same thresholds as LL128OneShotMultiRank, so it needs its own process (see the note above).
TEST_F(DdaMPI_AllReduce, LL128OneShotBitIdenticalAcrossRanks)
{
    MPIHelpers::MpiEnvGuard llOneShotGuard("RCCL_DDA_LL_ONESHOT_THRESHOLD", "0");
    MPIHelpers::MpiEnvGuard llTwoShotGuard("RCCL_DDA_LL_TWOSHOT_THRESHOLD", "0");

    runBitIdenticalAcrossRanks({ncclFloat32, ncclBfloat16}, {256 * 1024}, kLL128OneShotLogNeedle,
                               "DdaMPI_AllReduce/LL128OneShotBitIdenticalAcrossRanks");
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
// The failure mode is silent: work joined into the capture through comm->doneEvent can run eagerly
// outside the graph while every call, hipStreamEndCapture included, returns success. That is why
// the assertion is on the graph's shape and not on the data, which escaped work still computes
// correctly whenever it happens to run late enough. Reaching into struct ncclComm for doneEvent is
// deliberate: it is the only handle on whether the collective's own record was bound to this
// capture.
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

    hipEvent_t doneEvent = getActiveCommunicator()->doneEvent;
    ASSERT_NE(nullptr, doneEvent);

    ASSERT_EQ(hipSuccess, hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal));
    const ncclResult_t forkRes = ncclAllReduce(bufA, bufB, kContractCount, ncclFloat32, ncclSum,
                                               getActiveCommunicator(), getActiveStream());
    const hipError_t forkWaitRes = hipStreamWaitEvent(sideStream, doneEvent, 0);
    const hipError_t sideWorkRes
        = hipMemcpyAsync(bufC, bufB, bytes, hipMemcpyDeviceToDevice, sideStream);
    // HIP refuses to end a capture with an unjoined forked stream, so the copy rejoins here.
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
// The ordering edge is the test's own hipEventRecord after hipGraphLaunch, issued outside any
// capture, because RCCL cannot supply one from inside a capture.
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

// AICOMRCCL-2184: two addon collectives of one communicator captured on two streams of one graph
// are ordered in the graph, as native collectives are through sharedRes->deviceStream. Without that
// edge a replay may run both at once on the communicator's single DDA scratch buffer and barrier.
//
// The side stream forks before either collective, so only RCCL can order them.
TEST_F(DdaMPI_AllReduce, CapturedOnForkedStreamsSerializedInGraph)
{
    if(!validateTestPrerequisites(kMinProcessesForMPI))
        GTEST_SKIP() << "Need at least 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());
    if(getActiveCommunicator()->config.graphStreamOrdering == 0)
        GTEST_SKIP() << "graphStreamOrdering=0 orders captures through serialEvent; covered by "
                        "CapturedOnGraphOriginJoinsSerialEvent";

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    hipStream_t sideStream = nullptr;
    ASSERT_EQ(hipSuccess, hipStreamCreate(&sideStream));
    HipStreamAutoGuard sideStreamGuard(sideStream);

    hipEvent_t forkEvent = nullptr, joinEvent = nullptr;
    ASSERT_EQ(hipSuccess, hipEventCreateWithFlags(&forkEvent, hipEventDisableTiming));
    HipEventAutoGuard forkEventGuard(forkEvent);
    ASSERT_EQ(hipSuccess, hipEventCreateWithFlags(&joinEvent, hipEventDisableTiming));
    HipEventAutoGuard joinEventGuard(joinEvent);

    const size_t bytes = kContractCount * sizeof(float);

    void *bufA = nullptr, *bufB = nullptr, *bufC = nullptr, *bufD = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&bufA, bytes));
    DeviceBufferAutoGuard guardA(bufA);
    ASSERT_EQ(hipSuccess, hipMalloc(&bufB, bytes));
    DeviceBufferAutoGuard guardB(bufB);
    ASSERT_EQ(hipSuccess, hipMalloc(&bufC, bytes));
    DeviceBufferAutoGuard guardC(bufC);
    ASSERT_EQ(hipSuccess, hipMalloc(&bufD, bytes));
    DeviceBufferAutoGuard guardD(bufD);

    fillRankScalar(bufA, kContractCount, rank);
    fillRankScalar(bufC, kContractCount, rank);
    ASSERT_EQ(hipSuccess, hipMemset(bufB, 0, bytes));
    ASSERT_EQ(hipSuccess, hipMemset(bufD, 0, bytes));

    ASSERT_EQ(hipSuccess, hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal));
    const hipError_t   forkRecordRes = hipEventRecord(forkEvent, getActiveStream());
    const hipError_t   forkWaitRes   = hipStreamWaitEvent(sideStream, forkEvent, 0);
    const ncclResult_t originRes     = ncclAllReduce(bufA, bufB, kContractCount, ncclFloat32,
                                                     ncclSum, getActiveCommunicator(), getActiveStream());
    const ncclResult_t sideRes       = ncclAllReduce(bufC, bufD, kContractCount, ncclFloat32,
                                                     ncclSum, getActiveCommunicator(), sideStream);
    // HIP refuses to end a capture with an unjoined forked stream.
    const hipError_t joinRecordRes = hipEventRecord(joinEvent, sideStream);
    const hipError_t joinWaitRes   = hipStreamWaitEvent(getActiveStream(), joinEvent, 0);

    // End the capture before judging any result; see CapturedForkJoinStaysInsideGraph.
    hipGraph_t       graph      = nullptr;
    const hipError_t endCapture = hipStreamEndCapture(getActiveStream(), &graph);
    ASSERT_EQ(hipSuccess, forkRecordRes);
    ASSERT_EQ(hipSuccess, forkWaitRes);
    ASSERT_EQ(ncclSuccess, originRes) << "Rank " << rank << ": " << ncclGetErrorString(originRes);
    ASSERT_EQ(ncclSuccess, sideRes) << "Rank " << rank << ": " << ncclGetErrorString(sideRes);
    ASSERT_EQ(hipSuccess, joinRecordRes);
    ASSERT_EQ(hipSuccess, joinWaitRes);
    ASSERT_EQ(hipSuccess, endCapture);
    ASSERT_NE(nullptr, graph);
    auto graphGuard = makeScopeGuard([&]() { (void)hipGraphDestroy(graph); });

    const std::vector<hipGraphNode_t> work = graphWorkNodes(graph);
    ASSERT_GE(work.size(), 2u) << "Rank " << rank << ": expected work nodes from both AllReduces";
    for(size_t i = 0; i < work.size(); ++i)
        for(size_t j = i + 1; j < work.size(); ++j)
            EXPECT_TRUE(graphPathExists(work[i], work[j]) || graphPathExists(work[j], work[i]))
                << "Rank " << rank << ": graph work nodes " << i << " and " << j
                << " are unordered, so the two captured AllReduces may run concurrently";

    hipGraphExec_t graphExec = nullptr;
    ASSERT_EQ(hipSuccess, hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    auto graphExecGuard = makeScopeGuard([&]() { (void)hipGraphExecDestroy(graphExec); });
    ASSERT_EQ(hipSuccess, hipGraphLaunch(graphExec, getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    const float expectedSum = static_cast<float>(nRanks * (nRanks + 1) / 2);
    EXPECT_TRUE(verifyBufferData<float>(bufB, kContractCount,
                                        [expectedSum](size_t) { return expectedSum; }))
        << "Rank " << rank << ": the AllReduce captured on the origin stream wrote wrong data";
    EXPECT_TRUE(verifyBufferData<float>(bufD, kContractCount,
                                        [expectedSum](size_t) { return expectedSum; }))
        << "Rank " << rank << ": the AllReduce captured on the side stream wrote wrong data";

    reportSelectedTier("DdaMPI_AllReduce/CapturedOnForkedStreamsSerializedInGraph", bufA, bufB,
                       kContractCount, /*graphCapturing=*/1);
}

// AICOMRCCL-2184: with graphStreamOrdering=0, native captures are ordered across graph launches by
// an external wait on sharedRes->deviceStream.serialEvent before the kernel and a record of it
// after, and a captured addon collective carries the same pair. Run with
// NCCL_GRAPH_STREAM_ORDERING=0.
TEST_F(DdaMPI_AllReduce, CapturedOnGraphOriginJoinsSerialEvent)
{
    if(!validateTestPrerequisites(kMinProcessesForMPI))
        GTEST_SKIP() << "Need at least 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();
    if(comm->config.graphStreamOrdering != 0)
        GTEST_SKIP() << "Needs NCCL_GRAPH_STREAM_ORDERING=0";

    int rank{}, nRanks{};
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t bytes = kContractCount * sizeof(float);

    void *bufA = nullptr, *bufB = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&bufA, bytes));
    DeviceBufferAutoGuard guardA(bufA);
    ASSERT_EQ(hipSuccess, hipMalloc(&bufB, bytes));
    DeviceBufferAutoGuard guardB(bufB);

    fillRankScalar(bufA, kContractCount, rank);
    ASSERT_EQ(hipSuccess, hipMemset(bufB, 0, bytes));

    ASSERT_EQ(hipSuccess, hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal));
    const ncclResult_t captureRes
        = ncclAllReduce(bufA, bufB, kContractCount, ncclFloat32, ncclSum, comm, getActiveStream());
    hipGraph_t       graph      = nullptr;
    const hipError_t endCapture = hipStreamEndCapture(getActiveStream(), &graph);
    ASSERT_EQ(ncclSuccess, captureRes) << "Rank " << rank << ": " << ncclGetErrorString(captureRes);
    ASSERT_EQ(hipSuccess, endCapture);
    ASSERT_NE(nullptr, graph);
    auto graphGuard = makeScopeGuard([&]() { (void)hipGraphDestroy(graph); });

    hipEvent_t serialEvent = comm->sharedRes->deviceStream.serialEvent;
    EXPECT_EQ(1u, countEventNodes(graph, hipGraphNodeTypeWaitEvent, serialEvent))
        << "Rank " << rank << ": the captured AllReduce does not wait on serialEvent";
    EXPECT_EQ(1u, countEventNodes(graph, hipGraphNodeTypeEventRecord, serialEvent))
        << "Rank " << rank << ": the captured AllReduce does not record serialEvent";

    hipGraphExec_t graphExec = nullptr;
    ASSERT_EQ(hipSuccess, hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    auto graphExecGuard = makeScopeGuard([&]() { (void)hipGraphExecDestroy(graphExec); });
    for(int replay = 0; replay < 2; ++replay)
        ASSERT_EQ(hipSuccess, hipGraphLaunch(graphExec, getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    const float expectedSum = static_cast<float>(nRanks * (nRanks + 1) / 2);
    EXPECT_TRUE(verifyBufferData<float>(bufB, kContractCount,
                                        [expectedSum](size_t) { return expectedSum; }))
        << "Rank " << rank << ": the replayed AllReduce wrote wrong data";

    reportSelectedTier("DdaMPI_AllReduce/CapturedOnGraphOriginJoinsSerialEvent", bufA, bufB,
                       kContractCount, /*graphCapturing=*/1);
}


namespace
{
constexpr size_t kMiB = 1024 * 1024;
constexpr char   kDdaArTakenNeedle[] = "AllReduce: taking DDA fabric";
constexpr char   kRerouteOffReason[] = "the arch table keeps this re-route off (DPX or RCCL_IGNORE_ARCH_TABLE=1)";

// Grid and block of the last DDA fabric AllReduce kernel launch logged in `log`; false when there is none.
bool lastDdaArLaunchGeometry(const std::string& log, unsigned* grid, unsigned* block)
{
    const std::string kGrid = " grid=";
    for(size_t at = log.rfind(kGrid); at != std::string::npos;
        at        = at == 0 ? std::string::npos : log.rfind(kGrid, at - 1))
    {
        const size_t      bol  = log.rfind('\n', at);
        const size_t      from = bol == std::string::npos ? 0 : bol + 1;
        const size_t      eol  = log.find('\n', at);
        const std::string line = log.substr(from, eol == std::string::npos ? std::string::npos : eol - from);
        if(line.find("DDA fabric AllReduce") != std::string::npos)
        {
            return std::sscanf(line.c_str() + (at - from), " grid=%u block=%u", grid, block) == 2;
        }
    }
    return false;
}

// Rank r holds (r + 1) x (i % 509 + 1): exact in float, distinct per rank, and a misplaced shard shows.
float registeredInput(int rank, size_t i)
{
    return static_cast<float>(rank + 1) * static_cast<float>(i % 509 + 1);
}
} // namespace

/**
 * @class DdaMPI_AllReduceRegistered
 * @brief gfx1250 registered sum AllReduce past symMinR2: symk withdraws and DDA runs within the replaced grid.
 *
 * With the gfx1250 SPX table, capture replaces Ring from symMinR2 (8 MiB) up to max(32 MiB, nRanks x 2 MiB), and eager
 * replaces CE-registered for 512 KiB to 1 MiB shards, so it needs 8+ ranks at the default thresholds.
 */
class DdaMPI_AllReduceRegistered : public DdaAllReduceMPITest
{
protected:
    // Both windows registered: the decision is DDA with a cap, the launch stays within it, and every sum is exact.
    void runRegistered(size_t bytes, bool capture, const char* testId)
    {
        SCOPED_TRACE(std::string(testId) + " bytes=" + std::to_string(bytes));
        ncclComm_t comm = getActiveCommunicator();
        int        rank{}, nRanks{};
        ncclCommUserRank(comm, &rank);
        ncclCommCount(comm, &nRanks);
        const size_t count = bytes / sizeof(float);

        void* sendBuf = nullptr;
        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&sendBuf, bytes));
        auto sendFree = makeScopeGuard([&]() { (void)ncclMemFree(sendBuf); });
        ASSERT_MPI_EQ(ncclSuccess, ncclMemAlloc(&recvBuf, bytes));
        auto recvFree = makeScopeGuard([&]() { (void)ncclMemFree(recvBuf); });
        ncclWindow_t sendWin = nullptr;
        ncclWindow_t recvWin = nullptr;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, sendBuf, bytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
        auto sendDereg = makeScopeGuard([&]() { (void)ncclCommWindowDeregister(comm, sendWin); });
        ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowRegister(comm, recvBuf, bytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC));
        auto recvDereg = makeScopeGuard([&]() { (void)ncclCommWindowDeregister(comm, recvWin); });

        rcclCollDecision decision{};
        ASSERT_MPI_EQ(ncclSuccess, rcclSelectAllReduce(comm, sendBuf, recvBuf, count, ncclFloat32, ncclSum,
                                                       /*stream=*/nullptr, /*query=*/true, capture, &decision));
        const char* algoName = nullptr;
        rcclGetAlgoName(decision.algo, &algoName);
        if(capture)
        {
            EXPECT_TRUE(decision.algo == RCCL_DDA_FABRIC_LL || decision.algo == RCCL_DDA_FABRIC_LL128
                        || decision.algo == RCCL_DDA_FABRIC_VMM)
                << "Rank " << rank << ": captured call selected " << (algoName ? algoName : "?") << ", not DDA";
            EXPECT_EQ(0, static_cast<int>(decision.ddaCap.threads) % comm->WarpSize) << "Ring's block is whole warps";
        }
        else
        {
            EXPECT_EQ(RCCL_DDA_FABRIC_LL128, decision.algo)
                << "Rank " << rank << ": eager call selected " << (algoName ? algoName : "?");
            EXPECT_EQ(static_cast<uint32_t>(NCCL_CE_REDUCE_THREADS), decision.ddaCap.threads) << "CE's reduce block";
        }
        EXPECT_GE(decision.ddaCap.blocks, 1u) << "Rank " << rank << ": no cap, so DDA did not replace a backend";
        EXPECT_GE(decision.ddaCap.threads, 1u);
        EXPECT_LE(static_cast<uint32_t>(decision.nMaxChannels), decision.ddaCap.blocks);

        std::vector<float> host(count);
        for(size_t i = 0; i < count; ++i)
        {
            host[i] = registeredInput(rank, i);
        }
        ASSERT_MPI_EQ(hipSuccess, hipMemcpy(sendBuf, host.data(), bytes, hipMemcpyHostToDevice));

        hipStream_t    stream    = getActiveStream();
        const size_t   launches  = countLogNeedle(kDdaArTakenNeedle);
        hipGraphExec_t graphExec = nullptr;
        auto           execGuard = makeScopeGuard([&]() {
            if(graphExec != nullptr)
            {
                (void)hipGraphExecDestroy(graphExec);
            }
        });
        if(capture)
        {
            ASSERT_MPI_EQ(hipSuccess, hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
            const ncclResult_t captured = ncclAllReduce(sendBuf, recvBuf, count, ncclFloat32, ncclSum, comm, stream);
            hipGraph_t         graph    = nullptr;
            // End the capture before judging the call, or the fixture stream stays capturing.
            const hipError_t ended      = hipStreamEndCapture(stream, &graph);
            auto             graphGuard = makeScopeGuard([&]() { (void)hipGraphDestroy(graph); });
            ASSERT_MPI_EQ(ncclSuccess, captured);
            ASSERT_MPI_EQ(hipSuccess, ended);
            ASSERT_MPI_EQ(hipSuccess, hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        }

        // Two runs: a replayed graph and a repeated eager call must each write every sum again.
        const float rankSum = static_cast<float>(nRanks * (nRanks + 1) / 2);
        for(int run = 0; run < 2; ++run)
        {
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0xff, bytes));
            if(capture)
            {
                ASSERT_MPI_EQ(hipSuccess, hipGraphLaunch(graphExec, stream));
            }
            else
            {
                ASSERT_MPI_EQ(ncclSuccess,
                              ncclAllReduce(sendBuf, recvBuf, count, ncclFloat32, ncclSum, comm, stream));
            }
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
            EXPECT_TRUE(verifyBufferData<float>(recvBuf, count, [rankSum](size_t i) {
                return rankSum * static_cast<float>(i % 509 + 1);
            })) << "Rank " << rank << ": run " << run << " of the registered AllReduce wrote a wrong sum";
        }

        // Replays log nothing, so the last DDA launch logged is the captured one or the second eager call.
        EXPECT_GT(countLogNeedle(kDdaArTakenNeedle), launches) << "Rank " << rank << ": no DDA launch was logged";
        unsigned grid = 0, block = 0;
        ASSERT_TRUE(lastDdaArLaunchGeometry(logCtx_->readNcclDebugLog(), &grid, &block)
                    || lastDdaArLaunchGeometry(logCtx_->readPerRankStderrLog(), &grid, &block))
            << "Rank " << rank << ": the DDA launch logged no grid";
        EXPECT_LE(grid, decision.ddaCap.blocks) << "Rank " << rank << ": the launch grid exceeds the replaced one";
        EXPECT_LE(block, decision.ddaCap.threads) << "Rank " << rank << ": the launch block exceeds the replaced one";
        if(rank == 0)
        {
            TEST_INFO("%s: %zu bytes ran on %s, cap %ux%u, launched %ux%u", testId, bytes, algoName ? algoName : "?",
                      decision.ddaCap.blocks, decision.ddaCap.threads, grid, block);
        }
    }

    // Shared gates: skips the test, together on every rank, when a prerequisite is missing.
    void setUpRegistered(int minRanks, const char* rankReason)
    {
        if(!validateTestPrerequisites(kMinProcessesForMPI))
        {
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        }
        if(MPIEnvironment::world_size < minRanks)
        {
            GTEST_SKIP() << rankReason;
        }
        DDA_FABRIC_SKIP_OR_FAIL(archSkipReason("the registered AllReduce re-route is gfx1250 only"));
        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        DDA_FABRIC_SKIP_OR_FAIL(fabricInitSkipReason());
        const std::string symSkip = mpiCoordinatedSkipReason(getActiveCommunicator()->symmetricSupport == 0,
                                                             "registered windows need symmetric memory support");
        if(!symSkip.empty())
        {
            GTEST_SKIP() << symSkip;
        }
    }
};

// Graph capture: symMinR2 and the capture bound, where DDA replaces Ring within Ring's grid.
TEST_F(DdaMPI_AllReduceRegistered, CapturedSumsWithinRingGrid)
{
    setUpRegistered(kMinProcessesForMPI, "Need at least 2 MPI ranks");
    if(IsSkipped() || HasFatalFailure())
    {
        return;
    }
    const size_t      captureMax = rcclDdaArRegCaptureMaxBytes(getActiveCommunicator());
    const std::string offSkip    = mpiCoordinatedSkipReason(captureMax < 8 * kMiB, kRerouteOffReason);
    if(!offSkip.empty())
    {
        GTEST_SKIP() << offSkip;
    }
    for(const size_t bytes : {8 * kMiB, captureMax})
    {
        runRegistered(bytes, /*capture=*/true, "DdaMPI_AllReduceRegistered/CapturedSumsWithinRingGrid");
        if(HasFatalFailure())
        {
            return;
        }
    }
}

// Eager: 1 MiB shards, where the DDA LL128 two-shot replaces CE-registered within CE's reduce grid.
TEST_F(DdaMPI_AllReduceRegistered, EagerSumsWithinCeGrid)
{
    setUpRegistered(8, "Need at least 8 MPI ranks: 1 MiB shards reach the 8 MiB symMinR2 from 8 ranks");
    if(IsSkipped() || HasFatalFailure())
    {
        return;
    }
    ncclComm_t        comm   = getActiveCommunicator();
    const std::string ceSkip = mpiCoordinatedSkipReason(
        !ncclCeImplemented(ncclFuncAllReduce, ncclDevSum, ncclFloat32) || comm->nNodes > 1,
        "CE-registered, the backend the eager re-route replaces, needs CE driver support and one node");
    if(!ceSkip.empty())
    {
        GTEST_SKIP() << ceSkip;
    }
    int nRanks{};
    ncclCommCount(comm, &nRanks);
    const std::string offSkip = mpiCoordinatedSkipReason(
        !rcclDdaArRegEagerShardFits(comm, static_cast<size_t>(nRanks) * kMiB), kRerouteOffReason);
    if(!offSkip.empty())
    {
        GTEST_SKIP() << offSkip;
    }
    runRegistered(static_cast<size_t>(nRanks) * kMiB, /*capture=*/false,
                  "DdaMPI_AllReduceRegistered/EagerSumsWithinCeGrid");
}

#endif // MPI_TESTS_ENABLED
