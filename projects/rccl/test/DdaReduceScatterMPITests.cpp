/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaReduceScatterMPITests.cpp
 * @brief MPI end-to-end tests for the DDA fabric ReduceScatter LL and LL128 tiers
 *
 * Each input element depends on the sending rank, the destination shard, the element index and the
 * call, so a wrong peer, shard, packet or line index, or a stale scratch bank, changes the sum. The
 * two single-tier tests enqueue every call back to back with no host sync, so consecutive calls
 * alternate the two scratch banks and the third reuses the first; the two interleaved tests drain
 * the stream and barrier at each LL/LL128 switch. The COLL log line of every launch confirms the
 * tier, since the fallback paths also sum correctly.
 *
 * nRanks 4 and 8 reach the launchers' unrolled NRANKS_CT kernels; any other count reaches the
 * runtime one. Run (example):
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaMPI_ReduceScatter.*
 */

#ifdef MPI_TESTS_ENABLED

#include "DdaFabricMPITestFixture.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll128.h"
#include "comm.h"
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;

namespace
{
constexpr char kLLNeedle[]    = "DDA fabric ReduceScatter LL:";
constexpr char kLL128Needle[] = "DDA fabric ReduceScatter LL128:";

constexpr int      kCallsPerShape = 3;
constexpr size_t   kLLShapes      = 3;      // llShapes() under the default gfx1250 thresholds at up to 16 ranks
constexpr size_t   kLL128Shapes   = 4;      // ll128Shapes() under the default gfx1250 thresholds at up to 87 ranks
constexpr size_t   kGuardElems    = 64;     // past recvcount; a store beyond the shard lands here
constexpr size_t   kAlignElems    = 64;     // 256 B between per-call buffers in one arena
constexpr uint32_t kPoisonBits    = ~0u;    // NaN; an unwritten output never compares equal
constexpr uint32_t kValueModulus  = 2039;   // keeps every sum exact in f32 up to kDdaMaxNranks ranks
constexpr auto     kDrainTimeout  = std::chrono::seconds(120);

static_assert(static_cast<uint64_t>(dda::common::kDdaMaxNranks) * kValueModulus < (1u << 24),
              "ReduceScatter sums must stay exact in f32");

enum class Tier
{
    kLL,
    kLL128
};

struct Step
{
    Tier   tier;
    size_t recvcount;
};

const char* tierName(Tier tier)
{
    return tier == Tier::kLL ? "LL" : "LL128";
}

int expectedAlgo(Tier tier)
{
    return tier == Tier::kLL ? RCCL_DDA_FABRIC_LL : RCCL_DDA_FABRIC_LL128;
}

// Element `elem` of the chunk that rank `sender` sends to rank `shard` in call `call`.
float sendValue(int sender, int shard, size_t elem, int call)
{
    const uint64_t key = static_cast<uint64_t>(sender) * 977u + static_cast<uint64_t>(shard) * 131u
                         + static_cast<uint64_t>(elem) * 7u + static_cast<uint64_t>(call) * 409u;
    return static_cast<float>(1 + key % kValueModulus);
}

uint32_t floatBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

size_t alignUp(size_t elems)
{
    return (elems + kAlignElems - 1) / kAlignElems * kAlignElems;
}

// f32 shards that are whole 16 B LL packet pairs, from one pair to just under the slot / LL ceiling.
std::vector<size_t> llShapes(const ncclComm* comm, int nRanks)
{
    const size_t capBytes = std::min(dda::common::kDdaLLRsMaxBytes,
                                     rcclDdaLLThreshold(comm, ncclFuncReduceScatter) / nRanks);
    const size_t maxCount = capBytes / 16 * 4;
    std::vector<size_t> shapes;
    // 4: one 16 B pair. 2052: 1026 packets, five 256-thread blocks with a partial last one.
    for(size_t count : {size_t{4}, size_t{2052}})
    {
        if(count <= maxCount)
        {
            shapes.push_back(count);
        }
    }
    // Largest shard below the cap: more packets than the block cap times 256, so the grid strides.
    if(maxCount > 2052 + 4)
    {
        shapes.push_back(maxCount - 4);
    }
    return shapes;
}

// LL claims every 16 B-multiple shard first, so under default tables LL128 only serves an odd number of 8 B words.
std::vector<size_t> ll128Shapes(const ncclComm* comm, int nRanks)
{
    const size_t capBytes = std::min(dda::common::kDdaLL128RsMaxBytes,
                                     rcclDdaLL128Threshold(comm, ncclFuncReduceScatter) / nRanks);
    const size_t maxWords = capBytes / 8;
    if(maxWords == 0)
    {
        return {};
    }
    std::vector<size_t> words;
    // 1: a lone partial line. 77: five lines plus two words. 3007: 201 lines, four blocks of 64 line groups.
    for(size_t w : {size_t{1}, size_t{77}, size_t{3007}})
    {
        if(w <= maxWords)
        {
            words.push_back(w);
        }
    }
    size_t largest = (maxWords % 2 == 0) ? maxWords - 1 : maxWords;
    while(largest > 3007 && largest % dda::common::kDdaLL128DataElems == 0)
    {
        largest -= 2;
    }
    if(largest > 3007)
    {
        words.push_back(largest);
    }
    std::vector<size_t> shapes;
    for(size_t w : words)
    {
        shapes.push_back(w * 2);
    }
    return shapes;
}

std::vector<Step> repeatEachShape(Tier tier, const std::vector<size_t>& shapes)
{
    std::vector<Step> steps;
    for(size_t count : shapes)
    {
        for(int call = 0; call < kCallsPerShape; ++call)
        {
            steps.push_back({tier, count});
        }
    }
    return steps;
}

// Both tiers share the epoch cells and the scratch with different line layouts; alternate them, twice over.
std::vector<Step> interleavedSteps(const ncclComm* comm, int nRanks)
{
    const std::vector<size_t> ll    = llShapes(comm, nRanks);
    const std::vector<size_t> ll128 = ll128Shapes(comm, nRanks);
    std::vector<Step>         steps;
    for(int round = 0; round < 2; ++round)
    {
        for(size_t i = 0; i < std::max(ll.size(), ll128.size()); ++i)
        {
            if(i < ll.size())
            {
                steps.push_back({Tier::kLL, ll[i]});
            }
            if(i < ll128.size())
            {
                steps.push_back({Tier::kLL128, ll128[i]});
            }
        }
    }
    return steps;
}
} // namespace

class DdaMPI_ReduceScatter : public DdaFabricMPITest
{
protected:
    using StepBuilder = std::function<std::vector<Step>(const ncclComm*, int)>;

    // Must be the last statement of the test body: the coordinated skips and asserts return from here.
    void runSteps(const char* testId, bool needsLL128, const StepBuilder& buildSteps, size_t expectedSteps,
                  const char* maxBlocks = nullptr, bool syncOnTierSwitch = false)
    {
        if(!validateTestPrerequisites(kMinProcessesForMPI))
        {
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        }

        DDA_FABRIC_SKIP_OR_FAIL(archSkipReason("DDA fabric ReduceScatter LL/LL128 requires gfx1250"));

        // Fabric init reads RCCL_DDA_FABRIC_MAXBLOCKS with getenv on every comm, so the override is per test.
        std::unique_ptr<MPIHelpers::MpiEnvGuard> maxBlocksGuard;
        if(maxBlocks != nullptr)
        {
            maxBlocksGuard = std::make_unique<MPIHelpers::MpiEnvGuard>("RCCL_DDA_FABRIC_MAXBLOCKS", maxBlocks);
        }

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

        DDA_FABRIC_SKIP_OR_FAIL(fabricInitSkipReason());

        ncclComm_t comm = getActiveCommunicator();
        // These switch every DDA LL tier off for the process, so there is no LL path to assert.
        const bool ddaOff = rcclNcclAlgoEnvIsSet() || rcclParamDdaEnable() == 0 || rcclParamDdaLL() == 0
                            || comm->config.launchOrderImplicit == 1
                            || rcclDdaLLThreshold(comm, ncclFuncReduceScatter) == 0
                            || (needsLL128 && rcclDdaLL128Threshold(comm, ncclFuncReduceScatter) == 0);
        const std::string ddaOffSkip = mpiCoordinatedSkipReason(
            ddaOff,
            "DDA ReduceScatter LL tier off (NCCL_ALGO, RCCL_DDA_ENABLE/LL/LL128, thresholds, implicit launch order)");
        if(!ddaOffSkip.empty())
        {
            GTEST_SKIP() << ddaOffSkip;
        }

        if(maxBlocks != nullptr)
        {
            ASSERT_MPI_EQ(std::atoi(maxBlocks), comm->ddaFabricMaxBlocks);
        }

        int rank = -1, nRanks = 0;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(comm, &rank));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(comm, &nRanks));

        const std::vector<Step> steps = buildSteps(comm, nRanks);
        // A lowered threshold silently drops the grid-striding shapes, so the full shape set is required.
        if(steps.size() != expectedSteps)
        {
            ADD_FAILURE() << "Rank " << rank << ": " << testId << " built " << steps.size() << " calls, expected "
                          << expectedSteps << "; DDA ReduceScatter thresholds differ from the gfx1250 defaults";
        }
        ASSERT_MPI_EQ(expectedSteps, steps.size());

        std::vector<size_t> sendOffset(steps.size()), recvOffset(steps.size());
        size_t              sendElems = 0, recvElems = 0;
        for(size_t s = 0; s < steps.size(); ++s)
        {
            sendOffset[s] = sendElems;
            recvOffset[s] = recvElems;
            sendElems += alignUp(steps[s].recvcount * nRanks);
            recvElems += alignUp(steps[s].recvcount + kGuardElems);
        }

        void*            sendArena = nullptr;
        const hipError_t sendAlloc = hipMalloc(&sendArena, sendElems * sizeof(float));
        DeviceBufferAutoGuard sendGuard(sendArena);
        void*            recvArena = nullptr;
        const hipError_t recvAlloc = hipMalloc(&recvArena, recvElems * sizeof(float));
        DeviceBufferAutoGuard recvGuard(recvArena);
        ASSERT_MPI_EQ(hipSuccess, sendAlloc);
        ASSERT_MPI_EQ(hipSuccess, recvAlloc);

        float* sendBase = static_cast<float*>(sendArena);
        float* recvBase = static_cast<float*>(recvArena);

        // Every call is checked against the tier it must take; a silent fallback would still sum correctly.
        bool               selectionOk = true;
        std::ostringstream selectionLog;
        for(size_t s = 0; s < steps.size(); ++s)
        {
            int                algo = -1, protocol = -1, maxChannels = -1;
            const ncclResult_t query = rcclGetCollImplInfo(
                comm, ncclFuncReduceScatter, steps[s].recvcount, ncclFloat32, ncclSum, sendBase + sendOffset[s],
                recvBase + recvOffset[s], /*graphCapturing=*/0, &algo, &protocol, &maxChannels);
            if(query != ncclSuccess || algo != expectedAlgo(steps[s].tier))
            {
                const char* algoName = nullptr;
                rcclGetAlgoName(algo, &algoName);
                selectionOk = false;
                selectionLog << " call " << s << " (recvcount " << steps[s].recvcount << ") selected "
                             << (algoName ? algoName : "?") << " instead of DDA fabric "
                             << tierName(steps[s].tier) << ";";
            }
        }
        if(!selectionOk)
        {
            ADD_FAILURE() << "Rank " << rank << ": " << testId << ":" << selectionLog.str();
        }
        ASSERT_MPI_TRUE(selectionOk);

        std::vector<float> hostSend(sendElems, 0.0f);
        for(size_t s = 0; s < steps.size(); ++s)
        {
            const size_t count = steps[s].recvcount;
            for(int shard = 0; shard < nRanks; ++shard)
            {
                for(size_t i = 0; i < count; ++i)
                {
                    hostSend[sendOffset[s] + shard * count + i] = sendValue(rank, shard, i, static_cast<int>(s));
                }
            }
        }
        const hipError_t sendCopy
            = hipMemcpy(sendArena, hostSend.data(), sendElems * sizeof(float), hipMemcpyHostToDevice);
        const hipError_t recvPoison = hipMemset(recvArena, 0xFF, recvElems * sizeof(float));
        ASSERT_MPI_EQ(hipSuccess, sendCopy);
        ASSERT_MPI_EQ(hipSuccess, recvPoison);

        // No host sync between calls: a fast rank may start call k+1 while a peer still drains call k.
        // After a failed launch this rank stops launching but keeps looping, so all ranks reach the same collectives.
        ncclResult_t launchRes  = ncclSuccess;
        size_t       launchFail = steps.size();
        for(size_t s = 0; s < steps.size(); ++s)
        {
            if(syncOnTierSwitch && s > 0 && steps[s].tier != steps[s - 1].tier)
            {
                // LL and LL128 banks alias in the scratch, so a rank must not switch protocol while a peer still polls.
                DdaFabricMPI::abortJobUnlessAllRanksDrained(
                    DdaFabricMPI::waitStreamBounded(getActiveStream(), kDrainTimeout), testId);
                MPI_Barrier(MPI_COMM_WORLD);
            }
            if(launchRes != ncclSuccess)
            {
                continue;
            }
            launchRes = ncclReduceScatter(sendBase + sendOffset[s], recvBase + recvOffset[s], steps[s].recvcount,
                                          ncclFloat32, ncclSum, comm, getActiveStream());
            if(launchRes != ncclSuccess)
            {
                launchFail = s;
            }
        }

        DdaFabricMPI::abortJobUnlessAllRanksDrained(DdaFabricMPI::waitStreamBounded(getActiveStream(), kDrainTimeout),
                                                    testId);
        if(launchRes != ncclSuccess)
        {
            ADD_FAILURE() << "Rank " << rank << ": ncclReduceScatter call " << launchFail
                          << " failed: " << ncclGetErrorString(launchRes);
        }
        ASSERT_MPI_EQ(ncclSuccess, launchRes);

        std::vector<float> hostRecv(recvElems);
        ASSERT_MPI_EQ(hipSuccess,
                      hipMemcpy(hostRecv.data(), recvArena, recvElems * sizeof(float), hipMemcpyDeviceToHost));

        size_t llCalls = 0, ll128Calls = 0;
        for(size_t s = 0; s < steps.size(); ++s)
        {
            const Step&  step  = steps[s];
            const float* out   = hostRecv.data() + recvOffset[s];
            size_t       bad   = 0;
            size_t       first = step.recvcount;
            float        want  = 0.0f;
            for(size_t i = 0; i < step.recvcount; ++i)
            {
                float expected = 0.0f;
                for(int sender = 0; sender < nRanks; ++sender)
                {
                    expected += sendValue(sender, rank, i, static_cast<int>(s));
                }
                if(floatBits(out[i]) != floatBits(expected))
                {
                    if(bad++ == 0)
                    {
                        first = i;
                        want  = expected;
                    }
                }
            }
            EXPECT_EQ(0u, bad) << "Rank " << rank << ": " << testId << " call " << s << " ("
                               << tierName(step.tier) << ", recvcount " << step.recvcount << "): " << bad
                               << " wrong elements, first at " << first << ": expected " << want << ", got "
                               << (first < step.recvcount ? out[first] : 0.0f);

            size_t clobbered = 0;
            for(size_t g = 0; g < kGuardElems; ++g)
            {
                if(floatBits(out[step.recvcount + g]) != kPoisonBits)
                {
                    ++clobbered;
                }
            }
            EXPECT_EQ(0u, clobbered) << "Rank " << rank << ": " << testId << " call " << s << " ("
                                     << tierName(step.tier) << ", recvcount " << step.recvcount << ") wrote "
                                     << clobbered << " elements past the shard";

            if(step.tier == Tier::kLL)
            {
                ++llCalls;
            }
            else
            {
                ++ll128Calls;
            }
        }

        // One launcher line per call: a call that fell back to another backend leaves the count short.
        EXPECT_EQ(llCalls, countLogNeedle(kLLNeedle))
            << "Rank " << rank << ": " << testId << ": DDA fabric ReduceScatter LL launches in the COLL log";
        EXPECT_EQ(ll128Calls, countLogNeedle(kLL128Needle))
            << "Rank " << rank << ": " << testId << ": DDA fabric ReduceScatter LL128 launches in the COLL log";

        if(rank == 0)
        {
            TEST_INFO("%s: %zu LL + %zu LL128 calls at nRanks=%d (NRANKS_CT %d)", testId, llCalls, ll128Calls,
                      nRanks, (nRanks == 4 || nRanks == 8) ? nRanks : 0);
        }
    }
};

TEST_F(DdaMPI_ReduceScatter, LLTierBackToBackShapes)
{
    runSteps("DdaMPI_ReduceScatter/LLTierBackToBackShapes", /*needsLL128=*/false,
             [](const ncclComm* comm, int nRanks) { return repeatEachShape(Tier::kLL, llShapes(comm, nRanks)); },
             kLLShapes * kCallsPerShape);
}

// Shards of an odd number of 8 B words: one partial line, several lines with a partial last one, many blocks.
TEST_F(DdaMPI_ReduceScatter, LL128TierBackToBackShapes)
{
    runSteps("DdaMPI_ReduceScatter/LL128TierBackToBackShapes", /*needsLL128=*/true,
             [](const ncclComm* comm, int nRanks) {
                 return repeatEachShape(Tier::kLL128, ll128Shapes(comm, nRanks));
             },
             kLL128Shapes * kCallsPerShape);
}

// Same-tier calls stay back to back; each tier switch drains and barriers because LL and LL128 banks overlap.
TEST_F(DdaMPI_ReduceScatter, LLAndLL128Interleaved)
{
    runSteps("DdaMPI_ReduceScatter/LLAndLL128Interleaved", /*needsLL128=*/true, interleavedSteps,
             2 * (kLLShapes + kLL128Shapes), /*maxBlocks=*/nullptr, /*syncOnTierSwitch=*/true);
}

// BUG: unsynced LL<->LL128 switches hang (job aborts) or sum stale slots at 2 blocks, np 3 and 8. Enable when fixed.
TEST_F(DdaMPI_ReduceScatter, DISABLED_LLAndLL128InterleavedUnsynchronised)
{
    runSteps("DdaMPI_ReduceScatter/LLAndLL128InterleavedUnsynchronised", /*needsLL128=*/true, interleavedSteps,
             2 * (kLLShapes + kLL128Shapes), /*maxBlocks=*/"2", /*syncOnTierSwitch=*/false);
}

// The default block cap fits every LL128 shard in one pass; two blocks make both kernels grid-stride.
TEST_F(DdaMPI_ReduceScatter, GridStrideWithTwoBlocks)
{
    runSteps("DdaMPI_ReduceScatter/GridStrideWithTwoBlocks", /*needsLL128=*/true, interleavedSteps,
             2 * (kLLShapes + kLL128Shapes), /*maxBlocks=*/"2", /*syncOnTierSwitch=*/true);
}

#endif // MPI_TESTS_ENABLED
