/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaFabricSimpleMPITests.cpp
 * @brief MPI end-to-end tests for DDA fabric SIMPLE collectives
 *
 * The selected payloads bypass LL, while the fixture explicitly disables
 * LL128 before communicator creation, so every case exercises a multi-block
 * SIMPLE kernel. Each test checks both output data and the COLL log so a
 * correct fallback cannot hide a DDA regression.
 *
 * AllReduce, AllGather and ReduceScatter size their payload from the comm's
 * tiers: above the LL tier, up to the VMM tier. The gfx1250 tables can leave
 * that window empty (SPX keeps the AllGather VMM tier off, and below 8 ranks
 * its AllReduce LL tier reaches past the VMM tier), so these cases skip unless
 * the binary is launched with RCCL_DDA_LL=0 and RCCL_DDA_THRESHOLD covering
 * the payload.
 *
 * AllToAll is the exception: AllToAllOutOfPlace, RepeatedAllToAll, the AllToAll
 * step of AlternatingCollectivesOnSameCommunicator and the AllToAll barrier
 * stress run its LL tier. The AllToAll SIMPLE kernel tests need the DDA SIMPLE
 * tier to cover their payload: the gfx1250 DPX and SPX tables enable it up to
 * 256 MiB, and on a table that leaves it off they skip unless
 * RCCL_DDA_THRESHOLD covers it. Their kernel is chosen by RCCL_DDA_A2A_PUSH,
 * RCCL_DDA_A2A_PUSH_PEER and RCCL_DDA_A2A_TDM, which are cached per process, so
 * the peer-major vector push, all-peer TDM push and pull tests also skip unless
 * the binary is launched with RCCL_DDA_A2A_TDM=0, RCCL_DDA_A2A_PUSH_PEER=0 or
 * RCCL_DDA_A2A_PUSH=0.
 */

#ifdef MPI_TESTS_ENABLED

#include "DdaMPITestHelpers.hpp"
#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

RCCL_PARAM_DECLARE(DdaA2APush);
RCCL_PARAM_DECLARE(DdaA2APushPeer);
RCCL_PARAM_DECLARE(DdaA2ATdm);

namespace
{
// Preferred payloads; simpleBytes() moves them into the comm's SIMPLE window.
constexpr size_t kAllReduceCount     = 8 * 1024 * 1024 + 1;
constexpr size_t kAllGatherBytes     = 8 * 1024 * 1024;  // total
constexpr size_t kReduceScatterCount = 32 * 1024;       // 1 MiB input at 8 ranks
constexpr int    kAllToAllAlternatingRounds = 10;
constexpr int    kRepeatedIterations = 10;

// The fabric AllReduce SIMPLE launcher runs its flat kernel up to this size.
constexpr size_t kAllReduceFlatMaxBytes = 256 * 1024;

constexpr size_t kAllReduceBarrierStressCount = kAllReduceCount;

// A ReduceScatter shard 16 B above 512 KiB, small enough to expose publication
// races. AllToAll payloads are capped at 4 MiB in total.
constexpr size_t kBarrierStressCount      = 128 * 1024 + 4;
constexpr int    kBarrierStressIterations = 100;
constexpr size_t kAllToAllMaxTotalBytes   = 4 * 1024 * 1024;

constexpr char kAllReduceNeedle[] =
    "DDA fabric AllReduce: launching tree (two-shot) kernel";
constexpr char kAllGatherNeedle[] =
    "DDA fabric AllGather: launching kernel";
constexpr char kReduceScatterNeedle[] =
    "DDA fabric ReduceScatter: launching kernel";
constexpr char kAllToAllPullNeedle[] =
    "DDA fabric AllToAll: launching kernel";
constexpr char kAllToAllPushNeedle[] =
    "DDA fabric AllToAll: launching push kernel";
constexpr char kAllToAllPeerPushNeedle[] =
    "DDA fabric AllToAll: launching peer-major push kernel";
constexpr char kAllToAllTdmPushNeedle[] =
    "DDA fabric AllToAll: launching TDM push kernel";
constexpr char kAllToAllPeerTdmPushNeedle[] =
    "DDA fabric AllToAll: launching peer-major TDM push kernel";
constexpr char kAllToAllLLNeedle[] =
    "DDA fabric AllToAll LL:";
} // namespace

class DdaFabricSimpleMPITest : public MPITestBase
{
protected:
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugSubsysGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             ll128Guard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             maxBlocksGuard_;
    std::unique_ptr<MPIHelpers::TestLogAssertionContext> logCtx_;
    int                                                   rank_{};
    int                                                   nRanks_{};

    virtual const char* maxBlocksOverride() const
    {
        return nullptr;
    }

    void SetUp() override
    {
        MPITestBase::SetUp();
        debugGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG", "INFO");
        debugSubsysGuard_ =
            std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG_SUBSYS", "INIT,COLL");
        ll128Guard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("RCCL_DDA_LL128", "0");
        if(const char* value = maxBlocksOverride())
            maxBlocksGuard_ =
                std::make_unique<MPIHelpers::MpiEnvGuard>("RCCL_DDA_FABRIC_MAXBLOCKS", value);
        logCtx_ = std::make_unique<MPIHelpers::TestLogAssertionContext>(
            MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));

        if(!validateTestPrerequisites(kMinProcessesForMPI))
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        if(!DdaMPITestHelpers::isGfx1250Device())
            GTEST_SKIP() << "DDA fabric SIMPLE requires gfx1250";
        if(rcclParamDdaLL128() != 0)
            GTEST_SKIP() << "RCCL_DDA_LL128 was cached as enabled before this fixture; "
                            "launch with RCCL_DDA_LL128=0 to exercise SIMPLE";

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank_));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks_));

        // Fabric init can fail on some ranks (VMM unavailable, OOM) while succeeding
        // on others. Coordinate the skip across all ranks to avoid hangs in collectives.
        bool localSkip = (getActiveCommunicator()->ddaFabricBarrierState == nullptr);
        std::string skipReason =
            mpiCoordinatedSkipReason(localSkip, "DDA fabric path did not initialize");
        if(!skipReason.empty())
            GTEST_SKIP() << skipReason;
    }

    void TearDown() override
    {
        MPITestBase::TearDown();
        logCtx_.reset();
        maxBlocksGuard_.reset();
        ll128Guard_.reset();
        debugSubsysGuard_.reset();
        debugGuard_.reset();
    }

    void expectLog(const char* needle)
    {
        const std::string merged =
            logCtx_->readNcclDebugLog() + logCtx_->readPerRankStderrLog();
        EXPECT_NE(merged.find(needle), std::string::npos)
            << "Rank " << rank_ << " did not log expected DDA fabric marker: " << needle;
    }

    size_t allToAllCount(size_t desired) const
    {
        const size_t maxCount =
            kAllToAllMaxTotalBytes / (static_cast<size_t>(nRanks_) * sizeof(float));
        const size_t alignedMaxCount = maxCount - maxCount % (16 / sizeof(float));
        return desired < alignedMaxCount ? desired : alignedMaxCount;
    }

    // The largest count under kAllToAllMaxTotalBytes: above the AllToAll LL tier on SPX (128 KiB) and DPX (2 MiB), so
    // the SIMPLE kernels run.
    size_t simpleAllToAllCount() const
    {
        return allToAllCount(static_cast<size_t>(-1));
    }

    // The largest count in the AllToAll LL tier, or 0 when the tier is off.
    size_t llAllToAllCount()
    {
        const size_t count = rcclDdaLLThreshold(getActiveCommunicator(), ncclFuncAlltoAll)
                             / (static_cast<size_t>(nRanks_) * sizeof(float));
        return allToAllCount(count - count % (16 / sizeof(float)));
    }

    // Total bytes for a func call on its SIMPLE kernel: desired rounded up to align, raised above the LL tier and
    // minBytes and lowered to the VMM tier. 0 when that window is empty, with why set. Every rank sees the same
    // tiers, so they all skip together.
    size_t simpleBytes(const char* name, ncclFunc_t func, size_t desired, size_t minBytes, size_t align,
                       std::string& why)
    {
        const size_t ll     = rcclDdaLLThreshold(getActiveCommunicator(), func);
        const size_t vmm    = rcclDdaVmmThreshold(getActiveCommunicator(), func);
        const size_t lowest = (std::max(ll, minBytes) / align + 1) * align;
        const size_t highest = vmm - vmm % align;
        if(lowest > highest)
        {
            why = std::string(name) + " has no DDA SIMPLE window on this comm (LL to " + std::to_string(ll)
                  + " bytes, VMM to " + std::to_string(vmm)
                  + "); launch with RCCL_DDA_LL=0 and RCCL_DDA_THRESHOLD >= " + std::to_string(lowest);
            return 0;
        }
        return std::clamp((desired + align - 1) / align * align, lowest, highest);
    }

    // AllReduce SIMPLE bytes, above the flat kernel so the tree kernel runs.
    size_t allReduceBytes(size_t desiredCount, std::string& why)
    {
        return simpleBytes("AllReduce", ncclFuncAllReduce, desiredCount * sizeof(float), kAllReduceFlatMaxBytes,
                           static_cast<size_t>(nRanks_) * 16, why);
    }

    void runAllReduce(bool inPlace, int iterations = 1)
    {
        std::string why;
        const size_t bytes = allReduceBytes(kAllReduceCount, why);
        if(bytes == 0)
            GTEST_SKIP() << why;
        const size_t count = bytes / sizeof(float);
        void* sendBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);

        void* recvBuf = sendBuf;
        void* separateRecvBuf = nullptr;
        if(!inPlace)
        {
            ASSERT_MPI_EQ(hipSuccess, hipMalloc(&separateRecvBuf, bytes));
            recvBuf = separateRecvBuf;
        }
        DeviceBufferAutoGuard recvGuard(separateRecvBuf);

        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
            sendBuf,
            count,
            [this](size_t) { return static_cast<float>(rank_ + 1); }));
        if(!inPlace)
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, bytes));

        for(int i = 0; i < iterations; ++i)
        {
            ASSERT_MPI_EQ(ncclSuccess,
                          ncclAllReduce(sendBuf,
                                        recvBuf,
                                        count,
                                        ncclFloat32,
                                        ncclSum,
                                        getActiveCommunicator(),
                                        getActiveStream()));
        }

        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        const float expected = static_cast<float>(nRanks_ * (nRanks_ + 1) / 2);
        ASSERT_MPI_TRUE(verifyBufferData<float>(
            recvBuf, count, [expected](size_t) { return expected; }));
        expectLog(kAllReduceNeedle);
    }

    void runAllGather(bool inPlace)
    {
        // Each rank's block keeps 16-byte alignment.
        std::string why;
        const size_t totalBytes = simpleBytes("AllGather", ncclFuncAllGather, kAllGatherBytes, 0,
                                              static_cast<size_t>(nRanks_) * 16, why);
        if(totalBytes == 0)
            GTEST_SKIP() << why;
        const size_t totalCount = totalBytes / sizeof(float);
        const size_t count = totalCount / static_cast<size_t>(nRanks_);
        const size_t sendBytes = count * sizeof(float);

        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, totalBytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);
        ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, totalBytes));

        void* sendBuf = static_cast<float*>(recvBuf)
                      + static_cast<size_t>(rank_) * count;
        void* separateSendBuf = nullptr;
        if(!inPlace)
        {
            ASSERT_MPI_EQ(hipSuccess, hipMalloc(&separateSendBuf, sendBytes));
            sendBuf = separateSendBuf;
        }
        DeviceBufferAutoGuard sendGuard(separateSendBuf);

        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
            sendBuf,
            count,
            [this](size_t i) { return static_cast<float>(rank_ * 100000 + i); }));

        ASSERT_MPI_EQ(ncclSuccess,
                      ncclAllGather(sendBuf,
                                    recvBuf,
                                    count,
                                    ncclFloat32,
                                    getActiveCommunicator(),
                                    getActiveStream()));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        ASSERT_MPI_TRUE(verifyBufferData<float>(
            recvBuf, totalCount, [count](size_t i) {
                const size_t src = i / count;
                const size_t idx = i % count;
                return static_cast<float>(src * 100000 + idx);
            }));
        expectLog(kAllGatherNeedle);
    }

    void runReduceScatter(bool inPlace, int iterations = 1)
    {
        std::string why;
        const size_t sendBytes = simpleBytes("ReduceScatter", ncclFuncReduceScatter,
                                             kReduceScatterCount * static_cast<size_t>(nRanks_) * sizeof(float), 0,
                                             static_cast<size_t>(nRanks_) * 16, why);
        if(sendBytes == 0)
            GTEST_SKIP() << why;
        const size_t totalCount = sendBytes / sizeof(float);
        const size_t recvCount = totalCount / static_cast<size_t>(nRanks_);
        const size_t recvBytes = recvCount * sizeof(float);

        void* sendBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, sendBytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);

        void* recvBuf = static_cast<float*>(sendBuf)
                      + static_cast<size_t>(rank_) * recvCount;
        void* separateRecvBuf = nullptr;
        if(!inPlace)
        {
            ASSERT_MPI_EQ(hipSuccess, hipMalloc(&separateRecvBuf, recvBytes));
            recvBuf = separateRecvBuf;
        }
        DeviceBufferAutoGuard recvGuard(separateRecvBuf);

        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
            sendBuf, totalCount, [this](size_t i) {
                return static_cast<float>(rank_ + 1 + static_cast<int>(i % 17));
            }));
        if(!inPlace)
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, recvBytes));

        for(int i = 0; i < iterations; ++i)
        {
            ASSERT_MPI_EQ(ncclSuccess,
                          ncclReduceScatter(sendBuf,
                                            recvBuf,
                                            recvCount,
                                            ncclFloat32,
                                            ncclSum,
                                            getActiveCommunicator(),
                                            getActiveStream()));
        }

        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        const float rankSum = static_cast<float>(nRanks_ * (nRanks_ + 1) / 2);
        ASSERT_MPI_TRUE(verifyBufferData<float>(
            recvBuf, recvCount, [this, rankSum, recvCount](size_t i) {
                const size_t globalIdx =
                    static_cast<size_t>(rank_) * recvCount + i;
                return rankSum
                     + static_cast<float>(nRanks_ * static_cast<int>(globalIdx % 17));
            }));
        expectLog(kReduceScatterNeedle);
    }

    // Why this process cannot run the AllToAll kernel a test needs, or empty if it can: the
    // AllToAll DDA SIMPLE tier must cover simpleAllToAllCount(), and the cached AllToAll CVARs
    // must match (-1 accepts any value).
    std::string allToAllSkipReason(int64_t push, int64_t pushPeer, int64_t tdm)
    {
        const size_t bytes =
            simpleAllToAllCount() * static_cast<size_t>(nRanks_) * sizeof(float);
        if(rcclDdaVmmThreshold(getActiveCommunicator(), ncclFuncAlltoAll) < bytes)
            return "the AllToAll DDA SIMPLE tier does not cover the payload; launch with "
                   "RCCL_DDA_THRESHOLD >= "
                   + std::to_string(bytes);
        auto unmet = [](const char* name, int64_t want, int64_t have) {
            return want >= 0 && want != have
                       ? std::string(" RCCL_DDA_A2A_") + name + "=" + std::to_string(want)
                       : std::string();
        };
        const std::string env = unmet("PUSH", push, rcclParamDdaA2APush())
                                + unmet("PUSH_PEER", pushPeer, rcclParamDdaA2APushPeer())
                                + unmet("TDM", tdm, rcclParamDdaA2ATdm());
        return env.empty() ? env : "AllToAll CVARs are cached per process; launch with" + env;
    }

    // Encoding must stay within float32 exact integer range (2^24 = 16.7M).
    // With 144 ranks: rank*10000 + dest*100 + idx%97 gives max ~1.44M.
    hipError_t fillAllToAllSend(void* buf, size_t countPerPeer)
    {
        return initializeBufferWithPattern<float>(
            buf, countPerPeer * static_cast<size_t>(nRanks_), [this, countPerPeer](size_t i) {
                const size_t dest = i / countPerPeer;
                const size_t idx = i % countPerPeer;
                return static_cast<float>(
                    rank_ * 10000 + static_cast<int>(dest) * 100 + idx % 97);
            });
    }

    bool checkAllToAllRecv(void* buf, size_t countPerPeer)
    {
        return verifyBufferData<float>(
            buf, countPerPeer * static_cast<size_t>(nRanks_), [this, countPerPeer](size_t i) {
                const size_t src = i / countPerPeer;
                const size_t idx = i % countPerPeer;
                return static_cast<float>(
                    static_cast<int>(src) * 10000 + rank_ * 100 + idx % 97);
            });
    }

    // In place runs once: a second in-place AllToAll would transpose the data back.
    void runAllToAll(int iterations, bool inPlace, size_t desiredCount, const char* needle)
    {
        const size_t countPerPeer = allToAllCount(desiredCount);
        ASSERT_GT(countPerPeer, 0u);
        const size_t totalCount = countPerPeer * static_cast<size_t>(nRanks_);
        const size_t bytes = totalCount * sizeof(float);

        void* sendBuf = nullptr;
        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);
        if(inPlace)
        {
            recvBuf = sendBuf;
        }
        else
        {
            ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
        }
        DeviceBufferAutoGuard recvGuard(inPlace ? nullptr : recvBuf);

        ASSERT_MPI_EQ(hipSuccess, fillAllToAllSend(sendBuf, countPerPeer));
        if(!inPlace)
        {
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, bytes));
        }

        for(int i = 0; i < iterations; ++i)
        {
            ASSERT_MPI_EQ(ncclSuccess,
                          ncclAllToAll(sendBuf,
                                       recvBuf,
                                       countPerPeer,
                                       ncclFloat32,
                                       getActiveCommunicator(),
                                       getActiveStream()));
        }

        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        ASSERT_MPI_TRUE(checkAllToAllRecv(recvBuf, countPerPeer));
        expectLog(needle);
    }

    // AllToAll on its LL tier; skips when the tier is off.
    void runLLAllToAll(int iterations)
    {
        const size_t count = llAllToAllCount();
        if(count == 0)
            GTEST_SKIP() << "the AllToAll DDA LL tier is off";
        runAllToAll(iterations, /*inPlace=*/false, count, kAllToAllLLNeedle);
    }

    // Enqueues an LL AllToAll and a SIMPLE AllToAll back to back for `rounds` rounds without syncing, so an LL kernel
    // can start on one rank while another is still in the SIMPLE kernel; both share the DDA scratch.
    void runAllToAllAlternatingWithLL(int rounds, const char* simpleNeedle)
    {
        const size_t counts[2] = {llAllToAllCount(), simpleAllToAllCount()};
        ASSERT_GT(counts[1], counts[0]);
        void* sendBufs[2] = {};
        void* recvBufs[2] = {};
        std::vector<std::unique_ptr<DeviceBufferAutoGuard>> guards;
        for(int b = 0; b < 2; ++b)
        {
            const size_t bytes = counts[b] * static_cast<size_t>(nRanks_) * sizeof(float);
            ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBufs[b], bytes));
            guards.push_back(std::make_unique<DeviceBufferAutoGuard>(sendBufs[b]));
            ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBufs[b], bytes));
            guards.push_back(std::make_unique<DeviceBufferAutoGuard>(recvBufs[b]));
            ASSERT_MPI_EQ(hipSuccess, fillAllToAllSend(sendBufs[b], counts[b]));
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBufs[b], 0, bytes));
        }

        for(int r = 0; r < rounds; ++r)
        {
            for(int b = 0; b < 2; ++b)
            {
                ASSERT_MPI_EQ(ncclSuccess,
                              ncclAllToAll(sendBufs[b],
                                           recvBufs[b],
                                           counts[b],
                                           ncclFloat32,
                                           getActiveCommunicator(),
                                           getActiveStream()));
            }
        }

        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        for(int b = 0; b < 2; ++b)
        {
            ASSERT_MPI_TRUE(checkAllToAllRecv(recvBufs[b], counts[b]));
        }
        expectLog(kAllToAllLLNeedle);
        expectLog(simpleNeedle);
    }
};

class DdaFabricSimple : public DdaFabricSimpleMPITest
{};

class DdaFabricBlockCap : public DdaFabricSimpleMPITest
{
protected:
    const char* maxBlocksOverride() const override
    {
        return getTestMpiRank() == 0 ? "32" : "64";
    }
};

TEST_F(DdaFabricBlockCap, UsesCliqueWideMinimum)
{
    // Rank 0 advertises 32 while every other rank advertises 64. The clique-wide
    // minimum should be 32 on all ranks after the bootstrap exchange.
    EXPECT_EQ(getActiveCommunicator()->ddaFabricMaxBlocks, 32)
        << "Rank " << rank_ << " did not converge to clique-wide minimum";
}

TEST_F(DdaFabricSimple, AllReduceOutOfPlace)
{
    runAllReduce(false);
}

TEST_F(DdaFabricSimple, AllReduceInPlace)
{
    runAllReduce(true);
}

TEST_F(DdaFabricSimple, AllGatherOutOfPlace)
{
    runAllGather(false);
}

TEST_F(DdaFabricSimple, AllGatherInPlace)
{
    runAllGather(true);
}

TEST_F(DdaFabricSimple, ReduceScatterOutOfPlace)
{
    runReduceScatter(false);
}

TEST_F(DdaFabricSimple, ReduceScatterInPlace)
{
    runReduceScatter(true);
}

TEST_F(DdaFabricSimple, AllToAllOutOfPlace)
{
    runLLAllToAll(1);
}

TEST_F(DdaFabricSimple, RepeatedAllReduce)
{
    runAllReduce(false, kRepeatedIterations);
}

TEST_F(DdaFabricSimple, RepeatedReduceScatter)
{
    runReduceScatter(false, kRepeatedIterations);
}

TEST_F(DdaFabricSimple, RepeatedAllToAll)
{
    runLLAllToAll(kRepeatedIterations);
}

// In place, every push setting takes the all-peer vector push.
TEST_F(DdaFabricSimple, AllToAllInPlace)
{
    if(const std::string why = allToAllSkipReason(1, -1, -1); !why.empty())
        GTEST_SKIP() << why;
    runAllToAll(1, /*inPlace=*/true, simpleAllToAllCount(), kAllToAllPushNeedle);
}

// The push kernels sync all ranks at exit, so LL / LL128 cannot reuse the scratch while a peer still drains it.
TEST_F(DdaFabricSimple, AllToAllPeerTdmPushAlternatingWithLL)
{
    if(const std::string why = allToAllSkipReason(1, 1, 1); !why.empty())
        GTEST_SKIP() << why;
    if(llAllToAllCount() == 0)
        GTEST_SKIP() << "the AllToAll DDA LL tier is off";
    runAllToAllAlternatingWithLL(kAllToAllAlternatingRounds, kAllToAllPeerTdmPushNeedle);
}

TEST_F(DdaFabricSimple, AllToAllTdmPushUnalignedChunks)
{
    if(const std::string why = allToAllSkipReason(1, -1, 1); !why.empty())
        GTEST_SKIP() << why;
    // 16 B short of the largest count per peer, so a peer's chunk is not a multiple of 256 B.
    runAllToAll(kRepeatedIterations,
                /*inPlace=*/false,
                simpleAllToAllCount() - 4,
                rcclParamDdaA2APushPeer() ? kAllToAllPeerTdmPushNeedle : kAllToAllTdmPushNeedle);
}

TEST_F(DdaFabricSimple, AllToAllPeerVectorPush)
{
    if(const std::string why = allToAllSkipReason(1, 1, 0); !why.empty())
        GTEST_SKIP() << why;
    runAllToAll(kRepeatedIterations, /*inPlace=*/false, simpleAllToAllCount(), kAllToAllPeerPushNeedle);
}

TEST_F(DdaFabricSimple, AllToAllAllPeerTdmPush)
{
    if(const std::string why = allToAllSkipReason(1, 0, 1); !why.empty())
        GTEST_SKIP() << why;
    runAllToAll(kRepeatedIterations, /*inPlace=*/false, simpleAllToAllCount(), kAllToAllTdmPushNeedle);
}

TEST_F(DdaFabricSimple, AllToAllPull)
{
    if(const std::string why = allToAllSkipReason(0, -1, -1); !why.empty())
        GTEST_SKIP() << why;
    runAllToAll(kRepeatedIterations, /*inPlace=*/false, simpleAllToAllCount(), kAllToAllPullNeedle);
}

TEST_F(DdaFabricSimple, AlternatingCollectivesOnSameCommunicator)
{
    runAllReduce(false);
    if(HasFatalFailure() || IsSkipped())
        return;
    runAllGather(false);
    if(HasFatalFailure() || IsSkipped())
        return;
    runReduceScatter(false);
    if(HasFatalFailure() || IsSkipped())
        return;
    runLLAllToAll(1);
}

// ---------------------------------------------------------------------------
// Barrier stress tests repeatedly exercise the SIMPLE synchronization path.
// AllReduce uses a payload above its LL range while the fixture disables LL128;
// ReduceScatter can use smaller 512 KiB shards; AllToAll runs its LL tier.
// ---------------------------------------------------------------------------

class DdaFabricBarrierStress : public DdaFabricSimpleMPITest
{
protected:
    void runBarrierStressAllReduce()
    {
        std::string why;
        const size_t bytes = allReduceBytes(kAllReduceBarrierStressCount, why);
        if(bytes == 0)
            GTEST_SKIP() << why;
        const size_t count = bytes / sizeof(float);

        void* sendBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);

        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        long long totalMismatches = 0;
        int reportsEmitted = 0;

        for(int iter = 0; iter < kBarrierStressIterations; ++iter)
        {
            // Fresh pattern each iteration to detect stale reads
            const float pattern = static_cast<float>(iter * 1000 + rank_ + 1);
            ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
                sendBuf, count, [pattern](size_t) { return pattern; }));
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, bytes));

            ASSERT_MPI_EQ(ncclSuccess,
                          ncclAllReduce(sendBuf,
                                        recvBuf,
                                        count,
                                        ncclFloat32,
                                        ncclSum,
                                        getActiveCommunicator(),
                                        getActiveStream()));
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

            // Expected: sum of all ranks' patterns for this iteration
            float expected = 0.0f;
            for(int r = 0; r < nRanks_; ++r)
            {
                expected += static_cast<float>(iter * 1000 + r + 1);
            }

            // Count mismatches without failing immediately
            std::vector<float> hostBuf(count);
            ASSERT_MPI_EQ(hipSuccess,
                          hipMemcpy(hostBuf.data(), recvBuf, bytes, hipMemcpyDeviceToHost));

            int iterMismatches = 0;
            for(size_t i = 0; i < count; ++i)
            {
                if(hostBuf[i] != expected)
                {
                    ++iterMismatches;
                }
            }

            if(iterMismatches > 0)
            {
                totalMismatches += iterMismatches;
                if(reportsEmitted < 5)
                {
                    // Log first few failures
                    fprintf(stderr,
                            "Rank %d: Iteration %d had %d mismatches (expected %.0f)\n",
                            rank_, iter, iterMismatches, expected);
                    ++reportsEmitted;
                }
            }
        }

        // Aggregate across ranks
        long long globalMismatches = 0;
        MPI_Reduce(
            &totalMismatches, &globalMismatches, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank_ == 0)
        {
            EXPECT_EQ(globalMismatches, 0)
                << "Barrier stress test detected " << globalMismatches
                << " data corruption errors across " << kBarrierStressIterations
                << " iterations. This indicates a barrier synchronization bug.";
        }

        expectLog(kAllReduceNeedle);
    }

    void runBarrierStressReduceScatter()
    {
        const size_t recvCount = kBarrierStressCount;
        const size_t totalCount = recvCount * static_cast<size_t>(nRanks_);
        const size_t sendBytes = totalCount * sizeof(float);
        const size_t recvBytes = recvCount * sizeof(float);
        // The odd shard is the point of this test, so skip rather than resize it.
        std::string why;
        if(simpleBytes("ReduceScatter", ncclFuncReduceScatter, sendBytes, 0, 16, why) != sendBytes)
            GTEST_SKIP() << (why.empty() ? "the odd ReduceScatter shard is outside the DDA SIMPLE window" : why);

        void* sendBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, sendBytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);

        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, recvBytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        long long totalMismatches = 0;

        for(int iter = 0; iter < kBarrierStressIterations; ++iter)
        {
            ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
                sendBuf, totalCount, [this, iter](size_t i) {
                    return static_cast<float>(iter * 1000 + rank_ + 1 + static_cast<int>(i % 17));
                }));
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, recvBytes));

            ASSERT_MPI_EQ(ncclSuccess,
                          ncclReduceScatter(sendBuf,
                                            recvBuf,
                                            recvCount,
                                            ncclFloat32,
                                            ncclSum,
                                            getActiveCommunicator(),
                                            getActiveStream()));
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

            // Verify this rank's slice
            std::vector<float> hostBuf(recvCount);
            ASSERT_MPI_EQ(hipSuccess,
                          hipMemcpy(hostBuf.data(), recvBuf, recvBytes, hipMemcpyDeviceToHost));

            const float rankSum = static_cast<float>(nRanks_ * (nRanks_ + 1) / 2);
            int iterMismatches = 0;
            for(size_t i = 0; i < recvCount; ++i)
            {
                const size_t globalIdx = static_cast<size_t>(rank_) * recvCount + i;
                const float expected =
                    static_cast<float>(iter * 1000) * nRanks_ + rankSum
                    + static_cast<float>(nRanks_ * static_cast<int>(globalIdx % 17));
                if(hostBuf[i] != expected)
                {
                    ++iterMismatches;
                }
            }

            totalMismatches += iterMismatches;
        }

        long long globalMismatches = 0;
        MPI_Reduce(
            &totalMismatches, &globalMismatches, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank_ == 0)
        {
            EXPECT_EQ(globalMismatches, 0)
                << "ReduceScatter barrier stress test detected " << globalMismatches
                << " data corruption errors.";
        }

        expectLog(kReduceScatterNeedle);
    }

    void runBarrierStressAllToAll()
    {
        const size_t countPerPeer = llAllToAllCount();
        if(countPerPeer == 0)
            GTEST_SKIP() << "the AllToAll DDA LL tier is off";
        const size_t totalCount = countPerPeer * static_cast<size_t>(nRanks_);
        const size_t bytes = totalCount * sizeof(float);

        void* sendBuf = nullptr;
        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        long long totalMismatches = 0;

        for(int iter = 0; iter < kBarrierStressIterations; ++iter)
        {
            // Encoding must stay within float32 exact integer range (2^24 = 16.7M).
            // With 100 iters and 144 ranks: iter*100000 + rank*1000 + dest*7 + idx%7 gives max ~10M.
            ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
                sendBuf, totalCount, [this, iter, countPerPeer](size_t i) {
                    const size_t dest = i / countPerPeer;
                    const size_t idx = i % countPerPeer;
                    return static_cast<float>(
                        iter * 100000 + rank_ * 1000 + static_cast<int>(dest) * 7 + idx % 7);
                }));
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, bytes));

            ASSERT_MPI_EQ(ncclSuccess,
                          ncclAllToAll(sendBuf,
                                       recvBuf,
                                       countPerPeer,
                                       ncclFloat32,
                                       getActiveCommunicator(),
                                       getActiveStream()));
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

            std::vector<float> hostBuf(totalCount);
            ASSERT_MPI_EQ(hipSuccess,
                          hipMemcpy(hostBuf.data(), recvBuf, bytes, hipMemcpyDeviceToHost));

            int iterMismatches = 0;
            for(size_t i = 0; i < totalCount; ++i)
            {
                const size_t src = i / countPerPeer;
                const size_t idx = i % countPerPeer;
                const float expected = static_cast<float>(
                    iter * 100000 + static_cast<int>(src) * 1000 + rank_ * 7 + idx % 7);
                if(hostBuf[i] != expected)
                {
                    ++iterMismatches;
                }
            }

            totalMismatches += iterMismatches;
        }

        long long globalMismatches = 0;
        MPI_Reduce(
            &totalMismatches, &globalMismatches, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank_ == 0)
        {
            EXPECT_EQ(globalMismatches, 0)
                << "AllToAll barrier stress test detected " << globalMismatches
                << " data corruption errors.";
        }

        expectLog(kAllToAllLLNeedle);
    }
};

TEST_F(DdaFabricBarrierStress, AllReduceSmallBuffer)
{
    runBarrierStressAllReduce();
}

TEST_F(DdaFabricBarrierStress, ReduceScatterSmallBuffer)
{
    runBarrierStressReduceScatter();
}

TEST_F(DdaFabricBarrierStress, AllToAllSmallBuffer)
{
    runBarrierStressAllToAll();
}

#endif // MPI_TESTS_ENABLED
