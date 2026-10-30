/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Regression tests for NCCL symmetric kernel data corruption fixes (AICOMRCCL-1119).
//
// Upstream: NCCL 2.28.7-1 (commit ae7aed194d) fixed two classes of corruption:
//
//   1. AllGather LL and ReduceScatter LL alignment checks used per-chunk nElts
//      instead of the total nAllElts when computing the 8-byte lowBits mask.
//      When aggregated symmetric ops caused nElts to be small enough that its
//      low bits were zero despite the buffer not being 8-byte aligned, the
//      kernel took an optimized 8-byte path on misaligned data.
//
//   2. The work-range splitting logic in primitives.cuh used a single dw for
//      both workLo and workHi bounds, and fracLo didn't check whether the
//      previous channel's workHi matched the current work item, producing
//      incorrect element ranges for fused (grouped) operations.
//
// These tests exercise the fixed code paths with inputs that would have
// triggered corruption before the fix.

#include "DdaFabricMPITestFixture.hpp" // DdaFabricMPI::waitStreamBounded, abortJobUnlessAllRanksDrained
#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "ResourceGuards.hpp"
#include "SymmetricBufferHelpers.hpp"
#include "TestChecks.hpp"
#include "VerifiableFp8.hpp"
#include "comm.h"  // ncclComm::symkState, ncclSymkMaxBlocks
#include "nccl_device.h"
#include "rccl/rccl.h"
#include "rccl_common.h"  // rcclSymKGetInfo, rcclAddonAlgos_t

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace RCCLTestHelpers;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class SymmetricKernelCorruptionTest : public MPITestBase
{
protected:
    using SymBuf = RCCLTestHelpers::SymBuf;

    std::unique_ptr<MPIHelpers::MpiEnvGuard> cuMemGuard_;

    void SetUp() override
    {
        MPITestBase::SetUp();
        cuMemGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>(
            "NCCL_CUMEM_ENABLE", "1");
    }

    void TearDown() override
    {
        cuMemGuard_.reset();
        MPITestBase::TearDown();
    }

    ncclResult_t allocSymBuf(size_t bytes, SymBuf& sb)
    {
        return RCCLTestHelpers::ncclSymBufAlloc(
            getActiveCommunicator(), bytes, sb);
    }

    bool tryAllocSymBuf(size_t bytes, SymBuf& sb)
    {
        return allocSymBuf(bytes, sb) == ncclSuccess;
    }

    // Registers every buffer even after a failure, so all ranks issue the same collective registrations, then votes.
    std::string allocSymBufsSkipReason(std::initializer_list<std::pair<size_t, SymBuf*>> bufs)
    {
        bool ok = true;
        for(const auto& [bytes, sb] : bufs)
        {
            ok = tryAllocSymBuf(bytes, *sb) && ok;
        }
        return mpiCoordinatedSkipReason(!ok, "Symmetric memory not available (VMM/cuMem unsupported)");
    }
};

// ===========================================================================
// Test group 1: Sub-8-byte alignment — nElts vs nAllElts fix
//
// Exercises symmetric AllGather and ReduceScatter with message sizes that are
// NOT multiples of 8 bytes.  Before the fix, the LL kernels used per-chunk
// nElts to compute the lowBits alignment mask; when nElts happened to be
// 8-byte aligned but the buffer wasn't, the kernel took the optimized 8-byte
// path on misaligned data, causing corruption.
// ===========================================================================

// AllGather with float counts that produce non-8-byte-aligned total sizes.
// Each count is chosen so count*sizeof(float) is NOT a multiple of 8.
// (float = 4 bytes, so odd counts give 4-byte granularity.)
TEST_F(SymmetricKernelCorruptionTest, AllGather_Sub8ByteAlignment)
{
    if(!validateTestPrerequisites(2))
        GTEST_SKIP() << "Need >= 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    // Odd element counts → total bytes not divisible by 8.
    const std::vector<size_t> counts = {1, 3, 5, 7, 9, 13, 17, 31};

    for(size_t count : counts)
    {
        size_t sendBytes = count * sizeof(float);
        size_t recvBytes = count * static_cast<size_t>(nRanks) * sizeof(float);

        SymBuf sendSym, recvSym;
        const std::string noSym = allocSymBufsSkipReason({{sendBytes, &sendSym}, {recvBytes, &recvSym}});
        if(!noSym.empty())
        {
            GTEST_SKIP() << noSym;
        }

        ASSERT_EQ(hipSuccess,
                  initializeBufferWithPattern<float>(
                      sendSym.ptr, count,
                      [rank](size_t) { return static_cast<float>(rank + 1); }));

        ASSERT_EQ(hipSuccess, zeroInitializeBuffer<float>(recvSym.ptr, count * nRanks));

        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(sendSym.ptr, recvSym.ptr, count, ncclFloat,
                                getActiveCommunicator(), getActiveStream()));
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        size_t errIdx{};
        float  expVal{}, actVal{};
        ASSERT_TRUE(verifyBufferData<float>(
            recvSym.ptr, count * nRanks,
            [count](size_t i) {
                return static_cast<float>(i / count + 1);
            },
            0, 1e-5, &errIdx, &expVal, &actVal))
            << "AllGather corruption at count=" << count
            << " index=" << errIdx
            << " expected=" << expVal << " got=" << actVal;

        sendSym.release();
        recvSym.release();
    }
}

// ReduceScatter with float counts that produce non-8-byte-aligned total sizes.
TEST_F(SymmetricKernelCorruptionTest, ReduceScatter_Sub8ByteAlignment)
{
    if(!validateTestPrerequisites(2))
        GTEST_SKIP() << "Need >= 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const std::vector<size_t> counts = {1, 3, 5, 7, 9, 13, 17, 31};

    for(size_t recvCount : counts)
    {
        size_t sendCount = recvCount * static_cast<size_t>(nRanks);
        size_t sendBytes = sendCount * sizeof(float);
        size_t recvBytes = recvCount * sizeof(float);

        SymBuf sendSym, recvSym;
        const std::string noSym = allocSymBufsSkipReason({{sendBytes, &sendSym}, {recvBytes, &recvSym}});
        if(!noSym.empty())
        {
            GTEST_SKIP() << noSym;
        }

        // Each rank fills its send buffer with float(rank + 1) at every position.
        ASSERT_EQ(hipSuccess,
                  initializeBufferWithPattern<float>(
                      sendSym.ptr, sendCount,
                      [rank](size_t) { return static_cast<float>(rank + 1); }));

        ASSERT_EQ(hipSuccess, zeroInitializeBuffer<float>(recvSym.ptr, recvCount));

        ASSERT_EQ(ncclSuccess,
                  ncclReduceScatter(sendSym.ptr, recvSym.ptr, recvCount, ncclFloat,
                                    ncclSum, getActiveCommunicator(), getActiveStream()));
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        // After sum reduce-scatter, each element should be sum(rank+1) for all ranks
        // = nRanks*(nRanks+1)/2.
        float expectedSum = static_cast<float>(nRanks * (nRanks + 1)) / 2.0f;

        size_t errIdx{};
        float  expVal{}, actVal{};
        ASSERT_TRUE(verifyBufferData<float>(
            recvSym.ptr, recvCount,
            [expectedSum](size_t) { return expectedSum; },
            0, 1e-5, &errIdx, &expVal, &actVal))
            << "ReduceScatter corruption at recvCount=" << recvCount
            << " index=" << errIdx
            << " expected=" << expVal << " got=" << actVal;

        sendSym.release();
        recvSym.release();
    }
}

// ===========================================================================
// Test group 2: Aggregated/grouped symmetric operations — dwLo/dwHi fix
//
// Issues multiple symmetric collectives within a single ncclGroupStart/End
// block with varying message sizes so the work-range splitting logic assigns
// different workLo/workHi items.  Before the fix, using a single dw for
// both bounds, and not checking workHi in fracLo, caused incorrect element
// ranges and data corruption.
// ===========================================================================

// Multiple AllGather operations of varying sizes in a single group.
TEST_F(SymmetricKernelCorruptionTest, GroupedAllGather_VaryingSizes)
{
    if(!validateTestPrerequisites(2))
        GTEST_SKIP() << "Need >= 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    // Varying sizes so work items have different nElts, exercising the
    // dwLo/dwHi split and fracLo boundary logic.
    const std::vector<size_t> counts = {64, 1024, 256, 4096};
    const size_t              numOps = counts.size();

    std::vector<std::unique_ptr<SymBuf>> sendBufs(numOps);
    std::vector<std::unique_ptr<SymBuf>> recvBufs(numOps);

    for(size_t i = 0; i < numOps; ++i)
    {
        size_t sendBytes = counts[i] * sizeof(float);
        size_t recvBytes = counts[i] * static_cast<size_t>(nRanks) * sizeof(float);

        sendBufs[i] = std::make_unique<SymBuf>();
        recvBufs[i] = std::make_unique<SymBuf>();

        const std::string noSym
            = allocSymBufsSkipReason({{sendBytes, sendBufs[i].get()}, {recvBytes, recvBufs[i].get()}});
        if(!noSym.empty())
        {
            GTEST_SKIP() << noSym;
        }

        ASSERT_EQ(hipSuccess,
                  initializeBufferWithPattern<float>(
                      sendBufs[i]->ptr, counts[i],
                      [rank, i](size_t idx) {
                          return static_cast<float>((rank + 1) * 100 + i * 10 + (idx % 10));
                      }));

        ASSERT_EQ(hipSuccess,
                  zeroInitializeBuffer<float>(recvBufs[i]->ptr, counts[i] * nRanks));
    }

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for(size_t i = 0; i < numOps; ++i)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(sendBufs[i]->ptr, recvBufs[i]->ptr, counts[i],
                                ncclFloat, getActiveCommunicator(), getActiveStream()));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    for(size_t i = 0; i < numOps; ++i)
    {
        size_t count = counts[i];
        size_t errIdx{};
        float  expVal{}, actVal{};
        ASSERT_TRUE(verifyBufferData<float>(
            recvBufs[i]->ptr, count * nRanks,
            [count, i](size_t idx) {
                int    srcRank = static_cast<int>(idx / count);
                size_t srcIdx  = idx % count;
                return static_cast<float>((srcRank + 1) * 100 + i * 10 + (srcIdx % 10));
            },
            0, 1e-5, &errIdx, &expVal, &actVal))
            << "GroupedAllGather op=" << i << " count=" << count
            << " index=" << errIdx
            << " expected=" << expVal << " got=" << actVal;
    }
}

// Multiple ReduceScatter operations of varying sizes in a single group.
TEST_F(SymmetricKernelCorruptionTest, GroupedReduceScatter_VaryingSizes)
{
    if(!validateTestPrerequisites(2))
        GTEST_SKIP() << "Need >= 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const std::vector<size_t> recvCounts = {64, 1024, 256, 4096};
    const size_t              numOps     = recvCounts.size();

    std::vector<std::unique_ptr<SymBuf>> sendBufs(numOps);
    std::vector<std::unique_ptr<SymBuf>> recvBufs(numOps);

    for(size_t i = 0; i < numOps; ++i)
    {
        size_t sendCount = recvCounts[i] * static_cast<size_t>(nRanks);
        size_t sendBytes = sendCount * sizeof(float);
        size_t recvBytes = recvCounts[i] * sizeof(float);

        sendBufs[i] = std::make_unique<SymBuf>();
        recvBufs[i] = std::make_unique<SymBuf>();

        const std::string noSym
            = allocSymBufsSkipReason({{sendBytes, sendBufs[i].get()}, {recvBytes, recvBufs[i].get()}});
        if(!noSym.empty())
        {
            GTEST_SKIP() << noSym;
        }

        ASSERT_EQ(hipSuccess,
                  initializeBufferWithPattern<float>(
                      sendBufs[i]->ptr, sendCount,
                      [rank](size_t) { return static_cast<float>(rank + 1); }));

        ASSERT_EQ(hipSuccess,
                  zeroInitializeBuffer<float>(recvBufs[i]->ptr, recvCounts[i]));
    }

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for(size_t i = 0; i < numOps; ++i)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclReduceScatter(sendBufs[i]->ptr, recvBufs[i]->ptr, recvCounts[i],
                                    ncclFloat, ncclSum,
                                    getActiveCommunicator(), getActiveStream()));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    float expectedSum = static_cast<float>(nRanks * (nRanks + 1)) / 2.0f;

    for(size_t i = 0; i < numOps; ++i)
    {
        size_t errIdx{};
        float  expVal{}, actVal{};
        ASSERT_TRUE(verifyBufferData<float>(
            recvBufs[i]->ptr, recvCounts[i],
            [expectedSum](size_t) { return expectedSum; },
            0, 1e-5, &errIdx, &expVal, &actVal))
            << "GroupedReduceScatter op=" << i << " recvCount=" << recvCounts[i]
            << " index=" << errIdx
            << " expected=" << expVal << " got=" << actVal;
    }
}

// Mixed AllGather + ReduceScatter in a single group — exercises cross-collective
// work item boundaries in the fused work-range loop.
TEST_F(SymmetricKernelCorruptionTest, GroupedMixed_AllGatherAndReduceScatter)
{
    if(!validateTestPrerequisites(2))
        GTEST_SKIP() << "Need >= 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t agCount = 512;
    const size_t rsRecvCount = 1024;
    const size_t rsSendCount = rsRecvCount * static_cast<size_t>(nRanks);

    SymBuf agSend, agRecv, rsSend, rsRecv;
    const std::string noSym = allocSymBufsSkipReason({{agCount * sizeof(float), &agSend},
                                                      {agCount * nRanks * sizeof(float), &agRecv},
                                                      {rsSendCount * sizeof(float), &rsSend},
                                                      {rsRecvCount * sizeof(float), &rsRecv}});
    if(!noSym.empty())
    {
        GTEST_SKIP() << noSym;
    }

    ASSERT_EQ(hipSuccess,
              initializeBufferWithPattern<float>(
                  agSend.ptr, agCount,
                  [rank](size_t) { return static_cast<float>(rank + 1); }));
    ASSERT_EQ(hipSuccess,
              zeroInitializeBuffer<float>(agRecv.ptr, agCount * nRanks));

    ASSERT_EQ(hipSuccess,
              initializeBufferWithPattern<float>(
                  rsSend.ptr, rsSendCount,
                  [rank](size_t) { return static_cast<float>(rank + 1); }));
    ASSERT_EQ(hipSuccess,
              zeroInitializeBuffer<float>(rsRecv.ptr, rsRecvCount));

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_EQ(ncclSuccess,
              ncclAllGather(agSend.ptr, agRecv.ptr, agCount, ncclFloat,
                            getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(ncclSuccess,
              ncclReduceScatter(rsSend.ptr, rsRecv.ptr, rsRecvCount, ncclFloat,
                                ncclSum, getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    // Verify AllGather
    {
        size_t errIdx{};
        float  expVal{}, actVal{};
        ASSERT_TRUE(verifyBufferData<float>(
            agRecv.ptr, agCount * nRanks,
            [agCount](size_t i) {
                return static_cast<float>(i / agCount + 1);
            },
            0, 1e-5, &errIdx, &expVal, &actVal))
            << "Mixed group AllGather corruption at index=" << errIdx
            << " expected=" << expVal << " got=" << actVal;
    }

    // Verify ReduceScatter
    {
        float expectedSum = static_cast<float>(nRanks * (nRanks + 1)) / 2.0f;
        size_t errIdx{};
        float  expVal{}, actVal{};
        ASSERT_TRUE(verifyBufferData<float>(
            rsRecv.ptr, rsRecvCount,
            [expectedSum](size_t) { return expectedSum; },
            0, 1e-5, &errIdx, &expVal, &actVal))
            << "Mixed group ReduceScatter corruption at index=" << errIdx
            << " expected=" << expVal << " got=" << actVal;
    }
}

// Grouped operations with sub-8-byte message sizes — combines both bug classes.
TEST_F(SymmetricKernelCorruptionTest, GroupedOps_Sub8ByteAlignment)
{
    if(!validateTestPrerequisites(2))
        GTEST_SKIP() << "Need >= 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    // Odd counts so total bytes are not multiples of 8, inside a group.
    const size_t agCount     = 3;
    const size_t rsRecvCount = 5;
    const size_t rsSendCount = rsRecvCount * static_cast<size_t>(nRanks);

    SymBuf agSend, agRecv, rsSend, rsRecv;
    const std::string noSym = allocSymBufsSkipReason({{agCount * sizeof(float), &agSend},
                                                      {agCount * nRanks * sizeof(float), &agRecv},
                                                      {rsSendCount * sizeof(float), &rsSend},
                                                      {rsRecvCount * sizeof(float), &rsRecv}});
    if(!noSym.empty())
    {
        GTEST_SKIP() << noSym;
    }

    ASSERT_EQ(hipSuccess,
              initializeBufferWithPattern<float>(
                  agSend.ptr, agCount,
                  [rank](size_t) { return static_cast<float>(rank + 1); }));
    ASSERT_EQ(hipSuccess,
              zeroInitializeBuffer<float>(agRecv.ptr, agCount * nRanks));

    ASSERT_EQ(hipSuccess,
              initializeBufferWithPattern<float>(
                  rsSend.ptr, rsSendCount,
                  [rank](size_t) { return static_cast<float>(rank + 1); }));
    ASSERT_EQ(hipSuccess,
              zeroInitializeBuffer<float>(rsRecv.ptr, rsRecvCount));

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_EQ(ncclSuccess,
              ncclAllGather(agSend.ptr, agRecv.ptr, agCount, ncclFloat,
                            getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(ncclSuccess,
              ncclReduceScatter(rsSend.ptr, rsRecv.ptr, rsRecvCount, ncclFloat,
                                ncclSum, getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    {
        size_t errIdx{};
        float  expVal{}, actVal{};
        ASSERT_TRUE(verifyBufferData<float>(
            agRecv.ptr, agCount * nRanks,
            [agCount](size_t i) {
                return static_cast<float>(i / agCount + 1);
            },
            0, 1e-5, &errIdx, &expVal, &actVal))
            << "GroupedSub8 AllGather corruption at index=" << errIdx
            << " expected=" << expVal << " got=" << actVal;
    }

    {
        float expectedSum = static_cast<float>(nRanks * (nRanks + 1)) / 2.0f;
        size_t errIdx{};
        float  expVal{}, actVal{};
        ASSERT_TRUE(verifyBufferData<float>(
            rsRecv.ptr, rsRecvCount,
            [expectedSum](size_t) { return expectedSum; },
            0, 1e-5, &errIdx, &expVal, &actVal))
            << "GroupedSub8 ReduceScatter corruption at index=" << errIdx
            << " expected=" << expVal << " got=" << actVal;
    }
}

// ===========================================================================
// Test group 3: LL all-to-all session data placement
//
// Forces ReduceScatter_LL, the only caller of ncclLLA2ASession::send, and
// sends values unique per (rank, index) so a lane or slot permutation in the
// send path lands a wrong value in a checked output element. Also forces
// AllReduce_AGxLL_R, whose last epoch narrows the bcast slot stride.
// ===========================================================================

// A slot shift that leaves a slot unwritten hangs recvReduce instead of failing; only the runner timeout shows it.
TEST_F(SymmetricKernelCorruptionTest, ReduceScatterLL_PositionDependentData)
{
    // ReduceScatter_LL needs a single LSA team: at most 8 ranks on one node.
    constexpr int kMaxRanks = 8;
    if(!validateTestPrerequisites(2, kMaxRanks, false, 1, 1))
    {
        GTEST_SKIP() << "Need 2 to " << kMaxRanks << " MPI ranks on a single node";
    }

    // Read at communicator init, so the guard must outlive createTestCommunicator.
    MPIHelpers::MpiEnvGuard symKernelGuard("NCCL_SYM_KERNEL", "ReduceScatter_LL");
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{};
    int nRanks{};
    ASSERT_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank));
    ASSERT_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks));

    // Even counts take the aligned 8-byte path, odd ones the bounded path; 512 floats is one LL iteration per block.
    const std::vector<size_t> counts   = {1, 2, 7, 512, 1536, 1537, 1538, 4096, 4097};
    const size_t              maxCount = *std::max_element(counts.begin(), counts.end());

    // value(s, i) = i * 8 + s is injective; the largest sum over 8 ranks is about 2^21, so float addition is exact.
    auto value = [](int src, size_t idx) {
        return static_cast<float>(idx * kMaxRanks + static_cast<size_t>(src));
    };

    // Window registration is collective, so allocate once and agree on the outcome before any rank can skip.
    SymBuf            sendSym;
    SymBuf            recvSym;
    const std::string noSym
        = allocSymBufsSkipReason({{maxCount * nRanks * sizeof(float), &sendSym}, {maxCount * sizeof(float), &recvSym}});
    if(!noSym.empty())
    {
        GTEST_SKIP() << noSym;
    }

    // Without symmetricSupport the forced kernel is ineligible; skip with that reason instead of on ncclInvalidUsage.
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommQueryProperties(getActiveCommunicator(), &props));
    int symmetric = props.deviceApiSupport ? 1 : 0;
    ASSERT_EQ(MPI_SUCCESS, MPI_Allreduce(MPI_IN_PLACE, &symmetric, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if(!symmetric)
    {
        GTEST_SKIP() << "Symmetric kernels unavailable (symmetricSupport off, e.g. cuMem/VMM unavailable)";
    }

    for(size_t recvCount : counts)
    {
        size_t sendCount = recvCount * static_cast<size_t>(nRanks);

        ASSERT_MPI_EQ(hipSuccess,
                      initializeBufferWithPattern<float>(
                          sendSym.ptr, sendCount,
                          [rank, &value](size_t i) { return value(rank, i); }));
        ASSERT_MPI_EQ(hipSuccess, zeroInitializeBuffer<float>(recvSym.ptr, recvCount));

        // Only the first count may skip on ncclInvalidUsage; agree collectively so diverging verdicts cannot hang.
        ncclResult_t res = ncclReduceScatter(sendSym.ptr, recvSym.ptr, recvCount, ncclFloat, ncclSum,
                                             getActiveCommunicator(), getActiveStream());
        // Drain before the vote so a skip never leaves a kernel reading a window that ~SymBuf deregisters and frees.
        hipError_t syncErr = (res == ncclSuccess) ? hipStreamSynchronize(getActiveStream()) : hipSuccess;
        const std::string ineligible
            = mpiCoordinatedSkipReason(res == ncclInvalidUsage && recvCount == counts.front(),
                                       "ReduceScatter_LL symmetric kernel not eligible on this topology");
        if(!ineligible.empty())
        {
            GTEST_SKIP() << ineligible;
        }
        ASSERT_MPI_EQ(ncclSuccess, res);
        ASSERT_MPI_EQ(hipSuccess, syncErr);

        size_t base     = static_cast<size_t>(rank) * recvCount;
        auto   expected = [nRanks, base, &value](size_t j) {
            float sum = 0.0f;
            for(int s = 0; s < nRanks; ++s)
            {
                sum += value(s, base + j);
            }
            return sum;
        };

        size_t errIdx{};
        float  expVal{}, actVal{};
        bool   ok = verifyBufferData<float>(recvSym.ptr, recvCount, expected, 0, 1e-5, &errIdx, &expVal, &actVal);
        if(!ok)
        {
            ADD_FAILURE() << "ReduceScatter_LL mismatch at recvCount=" << recvCount << " index=" << errIdx
                          << " expected=" << expVal << " got=" << actVal;
        }
        // Collective so a rank-local mismatch stops every rank instead of leaving peers in the next LL kernel.
        ASSERT_MPI_TRUE(ok);
    }
}

// One CTA of 256 threads moves 256 8-byte packs per LL epoch, so 1100 floats end in a 38-pack wave and 2050 in a
// 1-pack wave, where the slot stride shrinks; odd counts and a one-float input skew take the unaligned arm.
TEST_F(SymmetricKernelCorruptionTest, AllReduceLL_PinnedTailWave)
{
    // AllReduce_AGxLL_R needs a single LSA team: at most 8 ranks on one node.
    constexpr int kMaxRanks = 8;
    if(!validateTestPrerequisites(2, kMaxRanks, false, 1, 1))
    {
        GTEST_SKIP() << "Need 2 to " << kMaxRanks << " MPI ranks on a single node";
    }

    // NCCL_PARAM caches NCCL_SYM_CTAS at its first read, so only the launch environment pins it reliably.
    constexpr int     kCtas   = 1;
    const char*       ctasEnv = std::getenv("NCCL_SYM_CTAS");
    const std::string noCtas  = mpiCoordinatedSkipReason(ctasEnv == nullptr || std::atoi(ctasEnv) != kCtas,
                                                        "Needs NCCL_SYM_CTAS=1 in the launch environment");
    if(!noCtas.empty())
    {
        GTEST_SKIP() << noCtas;
    }

    // Read at communicator init, so the guard must outlive createTestCommunicator.
    MPIHelpers::MpiEnvGuard symKernelGuard("NCCL_SYM_KERNEL", "AllReduce_AGxLL_R");
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{};
    int nRanks{};
    ASSERT_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank));
    ASSERT_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks));

    struct Case
    {
        size_t count;
        size_t skew; // input offset in floats
    };
    const std::vector<Case> cases = {{1100, 0}, {1101, 0}, {2050, 0}, {2051, 0}, {1100, 1}, {2050, 1}};
    constexpr size_t        kMaxCount = 2051;
    constexpr size_t        kMaxSkew  = 1;
    // The output sits 4 floats in, keeping it 8-byte aligned with a guard before it.
    constexpr size_t kOutOff  = 4;
    constexpr size_t kGuard   = 16;
    constexpr float  kSendPad = -3.0f;
    constexpr float  kRecvPad = -7.0f;

    // value(s, i) = i * 8 + s is injective; the largest sum over 8 ranks is about 2^17, so float addition is exact.
    auto value = [](int src, size_t idx) {
        return static_cast<float>(idx * kMaxRanks + static_cast<size_t>(src));
    };

    const size_t sendWinElts = kMaxSkew + kMaxCount + kGuard;
    const size_t recvWinElts = kOutOff + kMaxCount + kGuard;

    // Window registration is collective, so allocate once and agree on the outcome before any rank can skip.
    SymBuf            sendSym;
    SymBuf            recvSym;
    const std::string noSym
        = allocSymBufsSkipReason({{sendWinElts * sizeof(float), &sendSym}, {recvWinElts * sizeof(float), &recvSym}});
    if(!noSym.empty())
    {
        GTEST_SKIP() << noSym;
    }

    // Without symmetricSupport the forced kernel is ineligible; skip with that reason instead of on ncclInvalidUsage.
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommQueryProperties(getActiveCommunicator(), &props));
    int symmetric = props.deviceApiSupport ? 1 : 0;
    ASSERT_EQ(MPI_SUCCESS, MPI_Allreduce(MPI_IN_PLACE, &symmetric, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if(!symmetric)
    {
        GTEST_SKIP() << "Symmetric kernels unavailable (symmetricSupport off, e.g. cuMem/VMM unavailable)";
    }

    bool first = true;
    for(const Case& c : cases)
    {
        float* send = static_cast<float*>(sendSym.ptr) + c.skew;
        float* recv = static_cast<float*>(recvSym.ptr) + kOutOff;

        ASSERT_MPI_EQ(hipSuccess,
                      initializeBufferWithPattern<float>(sendSym.ptr, sendWinElts, [&](size_t j) {
                          return j >= c.skew && j < c.skew + c.count ? value(rank, j - c.skew) : kSendPad;
                      }));
        ASSERT_MPI_EQ(hipSuccess,
                      initializeBufferWithPattern<float>(recvSym.ptr, recvWinElts, [](size_t) { return kRecvPad; }));

        // Only the first call may skip on ncclInvalidUsage; agree collectively so diverging verdicts cannot hang.
        ncclResult_t res = ncclAllReduce(send, recv, c.count, ncclFloat, ncclSum, getActiveCommunicator(),
                                         getActiveStream());
        // A slot no peer writes leaves recvReduce spinning; abort the job rather than wait for the runner timeout.
        if(res == ncclSuccess)
        {
            const std::string what = "AllReduce_AGxLL_R count=" + std::to_string(c.count)
                                     + " skew=" + std::to_string(c.skew);
            DdaFabricMPI::abortJobUnlessAllRanksDrained(
                DdaFabricMPI::waitStreamBounded(getActiveStream(), std::chrono::seconds(60)), what.c_str());
        }
        const std::string ineligible
            = mpiCoordinatedSkipReason(res == ncclInvalidUsage && first,
                                       "AllReduce_AGxLL_R symmetric kernel not eligible on this topology");
        if(!ineligible.empty())
        {
            GTEST_SKIP() << ineligible;
        }
        first = false;
        ASSERT_MPI_EQ(ncclSuccess, res);

        // A fallback to another backend or CTA count would not reach the tail waves above.
        int algo{-1}, proto{-1}, nChannels{};
        ASSERT_MPI_EQ(ncclSuccess,
                      rcclGetCollImplInfo(getActiveCommunicator(), ncclFuncAllReduce, c.count, ncclFloat, ncclSum,
                                          send, recv, /*graphCapturing=*/0, &algo, &proto, &nChannels));
        ASSERT_MPI_EQ(static_cast<int>(RCCL_SYMMETRIC), algo);
        ASSERT_MPI_EQ(NCCL_PROTO_LL, proto);
        ASSERT_MPI_EQ(kCtas, nChannels);

        const size_t count    = c.count;
        auto         expected = [nRanks, count, &value](size_t j) {
            if(j < kOutOff || j >= kOutOff + count)
            {
                return kRecvPad;
            }
            float sum = 0.0f;
            for(int s = 0; s < nRanks; ++s)
            {
                sum += value(s, j - kOutOff);
            }
            return sum;
        };

        size_t errIdx{};
        float  expVal{}, actVal{};
        bool   ok = verifyBufferData<float>(recvSym.ptr, recvWinElts, expected, 0, 0, &errIdx, &expVal, &actVal);
        if(!ok)
        {
            const bool inOutput = errIdx >= kOutOff && errIdx < kOutOff + count;
            ADD_FAILURE() << "AllReduce_AGxLL_R mismatch at count=" << count << " skew=" << c.skew << ", "
                          << (inOutput ? "output index " : "guard at window index ")
                          << (inOutput ? errIdx - kOutOff : errIdx) << " expected=" << expVal << " got=" << actVal;
        }
        // Collective so a rank-local mismatch stops every rank instead of leaving peers in the next LL kernel.
        ASSERT_MPI_TRUE(ok);
    }
}

// ===========================================================================
// Test group 4: gfx950 LD tuning, chunk floor, pack tier and AllGather kernel selection. The
// data tests use position-dependent data, since a constant fill hides a mispartitioned range.
// ===========================================================================

namespace
{

// Prime, so the period does not divide the 1024-float gfx950 chunk and a chunk-aligned
// misplacement still changes the values.
constexpr size_t kLdPatternPeriod = 1021;

// Value a rank writes at a global index, small enough to be exact in float.
inline float ldPatternValue(int rank, size_t globalIdx)
{
    return static_cast<float>(rank + 1) + static_cast<float>(globalIdx % kLdPatternPeriod);
}

// Sum of ldPatternValue over all ranks at one global index.
inline float ldPatternSum(int nRanks, size_t globalIdx)
{
    return static_cast<float>(nRanks * (nRanks + 1)) / 2.0f
           + static_cast<float>(nRanks) * static_cast<float>(globalIdx % kLdPatternPeriod);
}

} // namespace

// gfx950 gates the deep path on a floor instead of trimming to a whole wave.
// The large counts are not multiples of the chunk size, so the last wave is partial, while the
// small ones stay on the LL kernels, where a stride disagreeing with the launch width would alias
// one rank's slots onto another's.
TEST_F(SymmetricKernelCorruptionTest, CountSweep_PartitioningAndSlots)
{
    if(!validateTestPrerequisites(2))
        GTEST_SKIP() << "Need >= 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    // The first three land on the LL kernels: ReduceScatter runs wide at all of its LL sizes, and
    // AllReduce runs narrow below 64 KB of message bytes and wide at 16K floats. The rest have an
    // odd chunk count (count / 1024), which nRanks * nBlocks cannot divide for any block count the
    // cost model picks, so the trim gfx950 drops would always have removed a partial wave.
    // AllGather reuses the AllReduce input and reaches its store kernel at the larger counts.
    const std::vector<size_t> counts = {2 * 1024,
                                        8 * 1024,
                                        16 * 1024,
                                        129 * 1024 + 1,
                                        257 * 1024 + 7,
                                        513 * 1024 + 129,
                                        1025 * 1024 + 1023};

    // Allocate once at the largest count. Registering a window per iteration
    // exhausts the symmetric pool well before the buffer bytes matter.
    const size_t maxCount = counts.back();

    SymBuf rsSend, rsRecv, arSend, arRecv, agRecv;
    const std::string noSym = allocSymBufsSkipReason({{maxCount * nRanks * sizeof(float), &rsSend},
                                                      {maxCount * sizeof(float), &rsRecv},
                                                      {maxCount * sizeof(float), &arSend},
                                                      {maxCount * sizeof(float), &arRecv},
                                                      {maxCount * nRanks * sizeof(float), &agRecv}});
    if(!noSym.empty())
    {
        GTEST_SKIP() << noSym;
    }

    // The pattern is a function of the global index alone, so one fill serves every count.
    ASSERT_EQ(hipSuccess,
              initializeBufferWithPattern<float>(
                  rsSend.ptr, maxCount * nRanks,
                  [rank](size_t i) { return ldPatternValue(rank, i); }));
    ASSERT_EQ(hipSuccess,
              initializeBufferWithPattern<float>(
                  arSend.ptr, maxCount,
                  [rank](size_t i) { return ldPatternValue(rank, i); }));

    for(size_t count : counts)
    {
        // --- ReduceScatter: count is the per-rank output size ---
        {
            ASSERT_EQ(hipSuccess, zeroInitializeBuffer<float>(rsRecv.ptr, count));

            ASSERT_EQ(ncclSuccess,
                      ncclReduceScatter(rsSend.ptr, rsRecv.ptr, count, ncclFloat,
                                        ncclSum, getActiveCommunicator(), getActiveStream()));
            ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

            // Output element j on this rank comes from global index rank*count + j.
            size_t errIdx{};
            float  expVal{}, actVal{};
            ASSERT_TRUE(verifyBufferData<float>(
                rsRecv.ptr, count,
                [nRanks, rank, count](size_t j) {
                    return ldPatternSum(nRanks, static_cast<size_t>(rank) * count + j);
                },
                0, 1e-3, &errIdx, &expVal, &actVal))
                << "ReduceScatter mismatch at count=" << count
                << " index=" << errIdx
                << " expected=" << expVal << " got=" << actVal;
        }

        // --- AllReduce: count is the full message size ---
        {
            ASSERT_EQ(hipSuccess, zeroInitializeBuffer<float>(arRecv.ptr, count));

            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(arSend.ptr, arRecv.ptr, count, ncclFloat,
                                    ncclSum, getActiveCommunicator(), getActiveStream()));
            ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

            size_t errIdx{};
            float  expVal{}, actVal{};
            ASSERT_TRUE(verifyBufferData<float>(
                arRecv.ptr, count,
                [nRanks](size_t i) { return ldPatternSum(nRanks, i); },
                0, 1e-3, &errIdx, &expVal, &actVal))
                << "AllReduce mismatch at count=" << count
                << " index=" << errIdx
                << " expected=" << expVal << " got=" << actVal;
        }

        // --- AllGather: count is the per-rank input size ---
        {
            ASSERT_EQ(hipSuccess, zeroInitializeBuffer<float>(agRecv.ptr, count * nRanks));

            ASSERT_EQ(ncclSuccess,
                      ncclAllGather(arSend.ptr, agRecv.ptr, count, ncclFloat,
                                    getActiveCommunicator(), getActiveStream()));
            ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

            // Output element i comes from rank i / count, which wrote it at its own index i % count.
            size_t errIdx{};
            float  expVal{}, actVal{};
            ASSERT_TRUE(verifyBufferData<float>(
                agRecv.ptr, count * nRanks,
                [count](size_t i) { return ldPatternValue(static_cast<int>(i / count), i % count); },
                0, 1e-3, &errIdx, &expVal, &actVal))
                << "AllGather mismatch at count=" << count
                << " index=" << errIdx
                << " expected=" << expVal << " got=" << actVal;
        }
    }
}

// Skewing the input by one float makes the relative offset non 16-byte aligned,
// which skips the 16-byte tier and reaches the 4-byte tier gfx950 widens.
TEST_F(SymmetricKernelCorruptionTest, MisalignedBuffers_FourBytePackTier)
{
    if(!validateTestPrerequisites(2))
        GTEST_SKIP() << "Need >= 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const std::vector<size_t> counts = {128 * 1024, 512 * 1024 + 4, 1024 * 1024 + 8};

    // One float of headroom so the input can start 4 bytes into its window.
    constexpr size_t kSkewElts = 1;

    const size_t maxCount = counts.back();

    SymBuf rsSend, rsRecv, arSend, arRecv, agRecv;
    const std::string noSym = allocSymBufsSkipReason({{(maxCount * nRanks + kSkewElts) * sizeof(float), &rsSend},
                                                      {maxCount * sizeof(float), &rsRecv},
                                                      {(maxCount + kSkewElts) * sizeof(float), &arSend},
                                                      {maxCount * sizeof(float), &arRecv},
                                                      {maxCount * nRanks * sizeof(float), &agRecv}});
    if(!noSym.empty())
    {
        GTEST_SKIP() << noSym;
    }

    float* rsSendSkewed = static_cast<float*>(rsSend.ptr) + kSkewElts;
    float* arSendSkewed = static_cast<float*>(arSend.ptr) + kSkewElts;

    ASSERT_EQ(hipSuccess,
              initializeBufferWithPattern<float>(
                  rsSendSkewed, maxCount * nRanks,
                  [rank](size_t i) { return ldPatternValue(rank, i); }));
    ASSERT_EQ(hipSuccess,
              initializeBufferWithPattern<float>(
                  arSendSkewed, maxCount,
                  [rank](size_t i) { return ldPatternValue(rank, i); }));

    for(size_t count : counts)
    {
        ASSERT_EQ(0u, count % 4) << "count must keep the per-rank stride 16-byte aligned";

        // --- ReduceScatter ---
        {
            ASSERT_EQ(hipSuccess, zeroInitializeBuffer<float>(rsRecv.ptr, count));

            ASSERT_EQ(ncclSuccess,
                      ncclReduceScatter(rsSendSkewed, rsRecv.ptr, count, ncclFloat,
                                        ncclSum, getActiveCommunicator(), getActiveStream()));
            ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

            size_t errIdx{};
            float  expVal{}, actVal{};
            ASSERT_TRUE(verifyBufferData<float>(
                rsRecv.ptr, count,
                [nRanks, rank, count](size_t j) {
                    return ldPatternSum(nRanks, static_cast<size_t>(rank) * count + j);
                },
                0, 1e-3, &errIdx, &expVal, &actVal))
                << "ReduceScatter misaligned mismatch at count=" << count
                << " index=" << errIdx
                << " expected=" << expVal << " got=" << actVal;
        }

        // --- AllReduce ---
        {
            ASSERT_EQ(hipSuccess, zeroInitializeBuffer<float>(arRecv.ptr, count));

            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(arSendSkewed, arRecv.ptr, count, ncclFloat,
                                    ncclSum, getActiveCommunicator(), getActiveStream()));
            ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

            size_t errIdx{};
            float  expVal{}, actVal{};
            ASSERT_TRUE(verifyBufferData<float>(
                arRecv.ptr, count,
                [nRanks](size_t i) { return ldPatternSum(nRanks, i); },
                0, 1e-3, &errIdx, &expVal, &actVal))
                << "AllReduce misaligned mismatch at count=" << count
                << " index=" << errIdx
                << " expected=" << expVal << " got=" << actVal;
        }

        // --- AllGather ---
        {
            ASSERT_EQ(hipSuccess, zeroInitializeBuffer<float>(agRecv.ptr, count * nRanks));

            ASSERT_EQ(ncclSuccess,
                      ncclAllGather(arSendSkewed, agRecv.ptr, count, ncclFloat,
                                    getActiveCommunicator(), getActiveStream()));
            ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

            size_t errIdx{};
            float  expVal{}, actVal{};
            ASSERT_TRUE(verifyBufferData<float>(
                agRecv.ptr, count * nRanks,
                [count](size_t i) { return ldPatternValue(static_cast<int>(i / count), i % count); },
                0, 1e-3, &errIdx, &expVal, &actVal))
                << "AllGather misaligned mismatch at count=" << count
                << " index=" << errIdx
                << " expected=" << expVal << " got=" << actVal;
        }
    }
}

namespace
{

// fp16 input: a position term with a prime period plus a distinct term per rank. The largest sum over
// 8 ranks is 8 * 198 + 16 * 28 = 2032, so every partial sum is an integer fp16 holds exactly.
inline float subWordHalfValue(int rank, size_t idx)
{
    return static_cast<float>(idx % 199 + 16 * static_cast<size_t>(rank));
}

} // namespace

// A 1- or 2-byte input skew sends fp16 and fp8 through the per-element head and tail alone; a 4-byte
// skew admits them to the 4-byte tier, which the two larger sizes enter at up to 8 ranks and 2 CTAs.
TEST_F(SymmetricKernelCorruptionTest, AllReduceSubWord_SkewedHeadAndFourByteTier)
{
    constexpr int kMaxRanks = 8;
    if(!validateTestPrerequisites(2, kMaxRanks, false, 1, 1))
    {
        GTEST_SKIP() << "Need 2 to " << kMaxRanks << " MPI ranks on a single node";
    }

    // NCCL_PARAM caches NCCL_SYM_CTAS at its first read, so only the launch environment pins it reliably.
    constexpr int     kCtas   = 2;
    const char*       ctasEnv = std::getenv("NCCL_SYM_CTAS");
    const std::string noCtas  = mpiCoordinatedSkipReason(ctasEnv == nullptr || std::atoi(ctasEnv) != kCtas,
                                                        "Needs NCCL_SYM_CTAS=2 in the launch environment");
    if(!noCtas.empty())
    {
        GTEST_SKIP() << noCtas;
    }

    // Read at communicator init, so the guard must outlive createTestCommunicator.
    MPIHelpers::MpiEnvGuard symKernelGuard("NCCL_SYM_KERNEL", "AllReduce_RSxLD_AGxST");
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{};
    int nRanks{};
    ASSERT_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank));
    ASSERT_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks));

    // skew is in bytes. The two larger sizes clear the 4-byte tier minimum (gfx942: 16 x 4 KiB at 8 ranks x 2 CTAs).
    struct Case
    {
        ncclDataType_t type;
        size_t         count;
        size_t         skew;
    };
    std::vector<Case> cases;
    for(size_t count : {1, 1021, 49155, 524291})
    {
        for(size_t skew : {2, 4})
        {
            cases.push_back({ncclFloat16, count, skew});
        }
    }
    for(size_t count : {1, 2045, 98311, 1048583})
    {
        for(size_t skew : {1, 2, 3, 4})
        {
            cases.push_back({ncclFloat8e4m3, count, skew});
        }
    }

    // The output sits 16 bytes in, leaving a guard before it without changing the offsets mod 16.
    constexpr size_t  kMaxSkew  = 4;
    constexpr size_t  kOutOff   = 16;
    constexpr size_t  kGuard    = 64;
    constexpr uint8_t kSendFill = 0x5A;
    constexpr uint8_t kRecvFill = 0xA5;
    size_t            maxBytes  = 0;
    for(const Case& c : cases)
    {
        maxBytes = std::max(maxBytes, c.count * (c.type == ncclFloat16 ? 2 : 1));
    }
    const size_t sendWinBytes = kMaxSkew + maxBytes + kGuard;
    const size_t recvWinBytes = kOutOff + maxBytes + kGuard;

    // Window registration is collective, so allocate once and agree on the outcome before any rank can skip.
    SymBuf            sendSym;
    SymBuf            recvSym;
    const std::string noSym = allocSymBufsSkipReason({{sendWinBytes, &sendSym}, {recvWinBytes, &recvSym}});
    if(!noSym.empty())
    {
        GTEST_SKIP() << noSym;
    }

    // Without symmetricSupport the forced kernel is ineligible; skip with that reason instead of on ncclInvalidUsage.
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommQueryProperties(getActiveCommunicator(), &props));
    int symmetric = props.deviceApiSupport ? 1 : 0;
    ASSERT_EQ(MPI_SUCCESS, MPI_Allreduce(MPI_IN_PLACE, &symmetric, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if(!symmetric)
    {
        GTEST_SKIP() << "Symmetric kernels unavailable (symmetricSupport off, e.g. cuMem/VMM unavailable)";
    }

    void* fp8Expected = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&fp8Expected, maxBytes));
    auto fp8ExpectedGuard = RCCLTestGuards::makeDeviceBufferAutoGuard(fp8Expected);

    std::vector<uint8_t> host(recvWinBytes);
    bool                 first = true;
    for(const Case& c : cases)
    {
        const bool   isHalf = c.type == ncclFloat16;
        const size_t bytes  = c.count * (isHalf ? 2 : 1);
        uint8_t*     send   = static_cast<uint8_t*>(sendSym.ptr) + c.skew;
        uint8_t*     recv   = static_cast<uint8_t*>(recvSym.ptr) + kOutOff;
        // Varies the fp8 data per case so a stale output cannot match.
        const uint64_t seed = c.count * 8 + c.skew;

        std::vector<uint8_t> expected(bytes);
        ASSERT_MPI_EQ(hipSuccess, hipMemset(sendSym.ptr, kSendFill, sendWinBytes));
        ASSERT_MPI_EQ(hipSuccess, hipMemset(recvSym.ptr, kRecvFill, recvWinBytes));
        // The fills run on the null stream; finish them before the fp8 input is generated on the active stream.
        ASSERT_MPI_EQ(hipSuccess, hipDeviceSynchronize());
        if(isHalf)
        {
            std::vector<__half> in(c.count);
            std::vector<__half> out(c.count);
            for(size_t i = 0; i < c.count; ++i)
            {
                in[i]     = __float2half(subWordHalfValue(rank, i));
                float sum = 0.0f;
                for(int s = 0; s < nRanks; ++s)
                {
                    sum += subWordHalfValue(s, i);
                }
                out[i] = __float2half(sum);
            }
            std::memcpy(expected.data(), out.data(), bytes);
            ASSERT_MPI_EQ(hipSuccess, hipMemcpy(send, in.data(), bytes, hipMemcpyHostToDevice));
        }
        else
        {
            ASSERT_MPI_EQ(hipSuccess,
                          RcclUnitTesting::VerifiableFp8PrepareInput(send, c.count, c.type, ncclSum, nRanks, rank,
                                                                     seed, 0, getActiveStream()));
            ASSERT_MPI_EQ(hipSuccess,
                          RcclUnitTesting::VerifiableFp8PrepareExpected(fp8Expected, c.count, c.type, ncclSum, nRanks,
                                                                        seed, 0, getActiveStream()));
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
            ASSERT_MPI_EQ(hipSuccess, hipMemcpy(expected.data(), fp8Expected, bytes, hipMemcpyDeviceToHost));
        }

        // Only the first call may skip on ncclInvalidUsage; agree collectively so diverging verdicts cannot hang.
        ncclResult_t res = ncclAllReduce(send, recv, c.count, c.type, ncclSum, getActiveCommunicator(),
                                         getActiveStream());
        // Drain before the vote so a skip never leaves a kernel writing a window that ~SymBuf deregisters and frees.
        hipError_t syncErr = (res == ncclSuccess) ? hipStreamSynchronize(getActiveStream()) : hipSuccess;
        const std::string ineligible
            = mpiCoordinatedSkipReason(res == ncclInvalidUsage && first,
                                       "AllReduce_RSxLD_AGxST symmetric kernel not eligible on this topology");
        if(!ineligible.empty())
        {
            GTEST_SKIP() << ineligible;
        }
        first = false;
        ASSERT_MPI_EQ(ncclSuccess, res);
        ASSERT_MPI_EQ(hipSuccess, syncErr);

        // A fallback to another backend or CTA count would leave the tier sizes above meaningless.
        int algo{-1}, proto{-1}, nChannels{};
        ASSERT_MPI_EQ(ncclSuccess,
                      rcclGetCollImplInfo(getActiveCommunicator(), ncclFuncAllReduce, c.count, c.type, ncclSum, send,
                                          recv, /*graphCapturing=*/0, &algo, &proto, &nChannels));
        ASSERT_MPI_EQ(static_cast<int>(RCCL_SYMMETRIC), algo);
        ASSERT_MPI_EQ(NCCL_PROTO_SIMPLE, proto);
        ASSERT_MPI_EQ(kCtas, nChannels);

        ASSERT_MPI_EQ(hipSuccess, hipMemcpy(host.data(), recvSym.ptr, recvWinBytes, hipMemcpyDeviceToHost));
        // Compares bytes: every sum is exact, so the encodings must match bit for bit.
        size_t badByte = recvWinBytes;
        for(size_t b = 0; b < recvWinBytes && badByte == recvWinBytes; ++b)
        {
            const bool inOutput = b >= kOutOff && b < kOutOff + bytes;
            if(host[b] != (inOutput ? expected[b - kOutOff] : kRecvFill))
            {
                badByte = b;
            }
        }
        const bool ok = badByte == recvWinBytes;
        if(!ok)
        {
            const bool inOutput = badByte >= kOutOff && badByte < kOutOff + bytes;
            ADD_FAILURE() << "AllReduce_RSxLD_AGxST mismatch, " << (isHalf ? "fp16" : "fp8e4m3")
                          << " count=" << c.count << " skew=" << c.skew << " bytes: "
                          << (inOutput ? "output element " : "guard byte at window offset ")
                          << (inOutput ? (badByte - kOutOff) / (isHalf ? 2 : 1) : badByte) << " got=0x" << std::hex
                          << static_cast<int>(host[badByte]) << std::dec;
        }
        // Collective so a rank-local mismatch stops every rank instead of leaving peers in the next kernel.
        ASSERT_MPI_TRUE(ok);
    }
}

// On gfx950 AllGather moves from LL to the store kernel at 1 MB of bus bytes, and the store kernel
// launches at most 24 blocks. The data tests above pass on either kernel, so this asks the symmetric
// tuner through the reporter rccl-tests uses.
TEST_F(SymmetricKernelCorruptionTest, AllGather_StoreKernelFrom1MB)
{
    if(!validateTestPrerequisites(2))
        GTEST_SKIP() << "Need >= 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int             dev{}, nRanks{};
    hipDeviceProp_t prop{};
    ASSERT_EQ(hipSuccess, hipGetDevice(&dev));
    ASSERT_EQ(hipSuccess, hipGetDeviceProperties(&prop, dev));
    if(std::strncmp(prop.gcnArchName, "gfx950", 6) != 0)
        GTEST_SKIP() << "The AllGather store crossover is tuned for gfx950 only";

    ncclCommCount(getActiveCommunicator(), &nRanks);
    if(nRanks != 8)
        GTEST_SKIP() << "The AllGather store crossover is fitted on 8 ranks";

    // The threshold counts bus bytes, which is nRanks times AllGather's per-rank count.
    constexpr size_t kStoreFromBusBytes = 1 << 20;
    const size_t     switchCount = kStoreFromBusBytes / (static_cast<size_t>(nRanks) * sizeof(float));

    SymBuf agSend, agRecv;
    const std::string noSym = allocSymBufsSkipReason({{switchCount * sizeof(float), &agSend},
                                                      {switchCount * nRanks * sizeof(float), &agRecv}});
    if(!noSym.empty())
    {
        GTEST_SKIP() << noSym;
    }

    // Run once on the registered windows so the reporter sees the state a real run leaves.
    ASSERT_EQ(ncclSuccess,
              ncclAllGather(agSend.ptr, agRecv.ptr, switchCount, ncclFloat,
                            getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    int algo{-1}, proto{-1}, nChannels{};
    ASSERT_EQ(ncclSuccess,
              rcclSymKGetInfo(getActiveCommunicator(), ncclFuncAllGather, switchCount - 1, ncclFloat,
                              ncclSum, &algo, &proto, &nChannels));
    EXPECT_EQ(static_cast<int>(RCCL_SYMMETRIC), algo) << "AllGather just below 1 MB left the symmetric kernels";
    EXPECT_EQ(NCCL_PROTO_LL, proto) << "AllGather just below 1 MB should stay on LL";

    ASSERT_EQ(ncclSuccess,
              rcclSymKGetInfo(getActiveCommunicator(), ncclFuncAllGather, switchCount, ncclFloat,
                              ncclSum, &algo, &proto, &nChannels));
    EXPECT_EQ(static_cast<int>(RCCL_SYMMETRIC), algo) << "AllGather at 1 MB left the symmetric kernels";
    EXPECT_EQ(NCCL_PROTO_SIMPLE, proto) << "AllGather at 1 MB should take the store kernel";
    EXPECT_LE(nChannels, 24) << "gfx950 caps the AllGather store kernel at 24 blocks";
}

// ===========================================================================
// Test group 5: AllGather store kernel data placement
//
// Forces AllGather_ST so every size takes it, including those LL would get, and
// checks position-dependent data across its tiers: the 16-byte deep loop, the
// 4-byte tier a 4-byte input skew reaches, the per-byte head and tail a 1-byte
// skew or odd size reaches, and in-place.
// ===========================================================================

namespace
{

// Prime, so the period does not divide any power-of-two chunk or tile and a
// misplaced chunk still changes the bytes it lands on.
constexpr size_t kAgStPatternPeriod = 251;

// Byte a rank writes at an index of its contribution. The rank term is distinct
// for up to 8 ranks, so a swapped slot changes the value too.
inline uint8_t agStValue(int rank, size_t idx)
{
    return static_cast<uint8_t>((idx + 37 * static_cast<size_t>(rank)) % kAgStPatternPeriod);
}

} // namespace

TEST_F(SymmetricKernelCorruptionTest, AllGatherST_PositionDependentData)
{
    // The store kernel writes straight into every peer's window: at most 8 ranks on one node.
    constexpr int kMaxRanks = 8;
    if(!validateTestPrerequisites(2, kMaxRanks, false, 1, 1))
    {
        GTEST_SKIP() << "Need 2 to " << kMaxRanks << " MPI ranks on a single node";
    }

    // Read at communicator init, so the guard must outlive createTestCommunicator.
    MPIHelpers::MpiEnvGuard symKernelGuard("NCCL_SYM_KERNEL", "AllGather_ST");
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{};
    int nRanks{};
    ASSERT_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank));
    ASSERT_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks));

    // Per-rank byte counts: below the 16-byte tier, around tile edges, and deep enough for many waves.
    const std::vector<size_t> counts   = {1, 15, 16, 17, 255, 4096, 4097, 65539, (1 << 20) + 5, 4 << 20};
    const size_t              maxCount = *std::max_element(counts.begin(), counts.end());

    // Input skew in bytes from its window base, and whether the input is the rank's own output slot.
    struct Placement
    {
        size_t      skew;
        bool        inPlace;
        const char* name;
    };
    const std::vector<Placement> placements = {
        {0, false, "aligned"}, {4, false, "4-byte skew"}, {1, false, "1-byte skew"}, {0, true, "in-place"}};
    constexpr size_t kMaxSkew = 4;

    // Window registration is collective, so allocate once and agree on the outcome before any rank can skip.
    SymBuf            sendSym;
    SymBuf            recvSym;
    const std::string noSym
        = allocSymBufsSkipReason({{maxCount + kMaxSkew, &sendSym}, {maxCount * nRanks, &recvSym}});
    if(!noSym.empty())
    {
        GTEST_SKIP() << noSym;
    }

    // Without symmetricSupport the forced kernel is ineligible; skip with that reason instead of on ncclInvalidUsage.
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommQueryProperties(getActiveCommunicator(), &props));
    int symmetric = props.deviceApiSupport ? 1 : 0;
    ASSERT_EQ(MPI_SUCCESS, MPI_Allreduce(MPI_IN_PLACE, &symmetric, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if(!symmetric)
    {
        GTEST_SKIP() << "Symmetric kernels unavailable (symmetricSupport off, e.g. cuMem/VMM unavailable)";
    }

    bool first = true;
    for(size_t count : counts)
    {
        for(const Placement& p : placements)
        {
            uint8_t* recv = static_cast<uint8_t*>(recvSym.ptr);
            uint8_t* send = p.inPlace ? recv + static_cast<size_t>(rank) * count
                                      : static_cast<uint8_t*>(sendSym.ptr) + p.skew;

            ASSERT_MPI_EQ(hipSuccess, zeroInitializeBuffer<uint8_t>(recv, count * nRanks));
            ASSERT_MPI_EQ(hipSuccess,
                          initializeBufferWithPattern<uint8_t>(
                              send, count, [rank](size_t i) { return agStValue(rank, i); }));

            // Only the first call may skip on ncclInvalidUsage; agree collectively so diverging verdicts cannot hang.
            ncclResult_t res = ncclAllGather(send, recv, count, ncclUint8, getActiveCommunicator(), getActiveStream());
            // Drain before the vote so a skip never leaves a kernel writing a window that ~SymBuf deregisters and frees.
            hipError_t syncErr = (res == ncclSuccess) ? hipStreamSynchronize(getActiveStream()) : hipSuccess;
            const std::string ineligible
                = mpiCoordinatedSkipReason(res == ncclInvalidUsage && first,
                                           "AllGather_ST symmetric kernel not eligible on this topology");
            if(!ineligible.empty())
            {
                GTEST_SKIP() << ineligible;
            }
            first = false;
            ASSERT_MPI_EQ(ncclSuccess, res);
            ASSERT_MPI_EQ(hipSuccess, syncErr);

            size_t  errIdx{};
            uint8_t expVal{}, actVal{};
            bool    ok = verifyBufferData<uint8_t>(
                recv, count * nRanks,
                [count](size_t j) { return agStValue(static_cast<int>(j / count), j % count); },
                0, 0, &errIdx, &expVal, &actVal);
            if(!ok)
            {
                ADD_FAILURE() << "AllGather_ST mismatch, " << p.name << ", count=" << count << " index=" << errIdx
                              << " (rank " << errIdx / count << ", offset " << errIdx % count << ")"
                              << " expected=" << static_cast<int>(expVal) << " got=" << static_cast<int>(actVal);
            }
            // Collective so a rank-local mismatch stops every rank instead of leaving peers in the next kernel.
            ASSERT_MPI_TRUE(ok);
        }
    }
}

// ===========================================================================
// Test group 6: LL A2A epoch wraparound
//
// Every LL symmetric kernel keeps a per-block epoch in the header line of the
// shared LL A2A buffer. endEpoch clears the slot half when the epoch reaches
// -2u or -1u and skips from -1u to 2, which takes ~4 billion collectives to
// reach, so the test seeds the header directly.
// ===========================================================================

// Generated by NCCL_PARAM(SymCTAs, ...) in src/tuning/sym_model.cc.
int64_t ncclParamSymCTAs();

namespace
{

// Sets the header word of every block; ncclLLA2ASession starts at header + 2.
__global__ void seedLLA2AHeadersKernel(struct ncclDevComm devComm, struct ncclLLA2AHandle handle, int nBlocks,
                                       uint32_t header)
{
    uint4* lines = static_cast<uint4*>(ncclGetResourceBufferLocalPointer(devComm, handle.bufHandle));
    for(int b = threadIdx.x; b < nBlocks; b += blockDim.x)
    {
        lines[static_cast<size_t>(b) * (1 + 2 * handle.nSlots)].x = header;
    }
}

// Copies every block's header word and counts nonzero lines in both slot halves of block 0.
__global__ void inspectLLA2AKernel(struct ncclDevComm devComm, struct ncclLLA2AHandle handle, int nBlocks,
                                   uint32_t* headers, unsigned int* block0Nonzero)
{
    const uint4* lines = static_cast<const uint4*>(ncclGetResourceBufferLocalPointer(devComm, handle.bufHandle));
    for(int b = threadIdx.x; b < nBlocks; b += blockDim.x)
    {
        headers[b] = lines[static_cast<size_t>(b) * (1 + 2 * handle.nSlots)].x;
    }
    unsigned int nonzero = 0;
    for(uint32_t i = threadIdx.x; i < 2 * handle.nSlots; i += blockDim.x)
    {
        const uint4 line = lines[1 + i];
        nonzero += (line.x | line.y | line.z | line.w) != 0;
    }
    atomicAdd(block0Nonzero, nonzero);
}

} // namespace

// Seeds the header to -4u, so the next two ReduceScatter_LL calls run at epochs -2u and -1u,
// each clearing one slot half, and leave the header at 0 (epoch 2).
TEST_F(SymmetricKernelCorruptionTest, LLA2A_EpochWraparound)
{
    constexpr int kMaxRanks = 8;
    if(!validateTestPrerequisites(2, kMaxRanks, false, 1, 1))
    {
        GTEST_SKIP() << "Need 2 to " << kMaxRanks << " MPI ranks on a single node";
    }

    // One block keeps each call to one LL epoch on block 0. NCCL_SYM_CTAS is cached on first read.
    constexpr int           kCtas = 1;
    MPIHelpers::MpiEnvGuard ctasGuard("NCCL_SYM_CTAS", std::to_string(kCtas).c_str());
    const std::string       staleCtas
        = mpiCoordinatedSkipReason(ncclParamSymCTAs() != kCtas,
                                   "NCCL_SYM_CTAS already cached this process; run this test in its own mpirun");
    if(!staleCtas.empty())
    {
        GTEST_SKIP() << staleCtas;
    }

    MPIHelpers::MpiEnvGuard symKernelGuard("NCCL_SYM_KERNEL", "ReduceScatter_LL");
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{};
    int nRanks{};
    ASSERT_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank));
    ASSERT_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks));

    // 256 packs per rank: one LL epoch per call, and on gfx942 (256 threads) every slot of the half is written.
    constexpr size_t kRecvCount = 512;

    SymBuf            sendSym;
    SymBuf            recvSym;
    const std::string noSym = allocSymBufsSkipReason(
        {{kRecvCount * nRanks * sizeof(float), &sendSym}, {kRecvCount * sizeof(float), &recvSym}});
    if(!noSym.empty())
    {
        GTEST_SKIP() << noSym;
    }

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommQueryProperties(getActiveCommunicator(), &props));
    int symmetric = props.deviceApiSupport ? 1 : 0;
    ASSERT_EQ(MPI_SUCCESS, MPI_Allreduce(MPI_IN_PLACE, &symmetric, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if(!symmetric)
    {
        GTEST_SKIP() << "Symmetric kernels unavailable (symmetricSupport off, e.g. cuMem/VMM unavailable)";
    }

    const ncclSymkDevComm& kcomm = getActiveCommunicator()->symkState.kcomm;
    uint32_t*     headersDev = nullptr;
    unsigned int* nonzeroDev = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&headersDev, ncclSymkMaxBlocks * sizeof(uint32_t)));
    RCCLTestGuards::DeviceBufferAutoGuard headersGuard(headersDev);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&nonzeroDev, sizeof(unsigned int)));
    RCCLTestGuards::DeviceBufferAutoGuard nonzeroGuard(nonzeroDev);

    // Header of each block and nonzero line count of block 0, read with the stream idle.
    std::vector<uint32_t> hostHeaders(ncclSymkMaxBlocks);
    unsigned int          hostNonzero = 0;
    auto inspect = [&]() {
        if(hipMemset(nonzeroDev, 0, sizeof(unsigned int)) != hipSuccess)
            return false;
        hipLaunchKernelGGL(inspectLLA2AKernel, dim3(1), dim3(256), 0, getActiveStream(), kcomm.devComm,
                           kcomm.lsaLLA2A, ncclSymkMaxBlocks, headersDev, nonzeroDev);
        return hipStreamSynchronize(getActiveStream()) == hipSuccess
               && hipMemcpy(hostHeaders.data(), headersDev, ncclSymkMaxBlocks * sizeof(uint32_t),
                            hipMemcpyDeviceToHost) == hipSuccess
               && hipMemcpy(&hostNonzero, nonzeroDev, sizeof(unsigned int), hipMemcpyDeviceToHost) == hipSuccess;
    };

    // Distinct per call, so a slot left over from an earlier call changes the sum.
    auto value = [](int src, size_t idx, int call) {
        return static_cast<float>((idx * kMaxRanks + static_cast<size_t>(src)) * 4 + static_cast<size_t>(call));
    };

    constexpr uint32_t kSeedHeader = -4u;
    // Calls: 0 allocates the buffer at normal epochs, 1 and 2 run at -2u and -1u, 3 runs after the wrap.
    for(int call = 0; call < 4; ++call)
    {
        if(call == 1)
        {
            ASSERT_MPI_TRUE(kcomm.lsaLLA2A.nSlots > 0);
            // A fresh header starts at epoch 2, so one call leaves it at 1.
            ASSERT_MPI_TRUE(inspect());
            ASSERT_MPI_EQ(1u, hostHeaders[0]);

            hipLaunchKernelGGL(seedLLA2AHeadersKernel, dim3(1), dim3(64), 0, getActiveStream(), kcomm.devComm,
                               kcomm.lsaLLA2A, ncclSymkMaxBlocks, kSeedHeader);
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        }

        const size_t sendCount = kRecvCount * static_cast<size_t>(nRanks);
        ASSERT_MPI_EQ(hipSuccess,
                      initializeBufferWithPattern<float>(
                          sendSym.ptr, sendCount, [rank, call, &value](size_t i) { return value(rank, i, call); }));
        ASSERT_MPI_EQ(hipSuccess,
                      initializeBufferWithPattern<float>(recvSym.ptr, kRecvCount, [](size_t) { return -1.0f; }));

        ncclResult_t res = ncclReduceScatter(sendSym.ptr, recvSym.ptr, kRecvCount, ncclFloat, ncclSum,
                                             getActiveCommunicator(), getActiveStream());
        hipError_t syncErr = (res == ncclSuccess) ? hipStreamSynchronize(getActiveStream()) : hipSuccess;
        const std::string ineligible
            = mpiCoordinatedSkipReason(res == ncclInvalidUsage && call == 0,
                                       "ReduceScatter_LL symmetric kernel not eligible on this topology");
        if(!ineligible.empty())
        {
            GTEST_SKIP() << ineligible;
        }
        ASSERT_MPI_EQ(ncclSuccess, res);
        ASSERT_MPI_EQ(hipSuccess, syncErr);

        if(call == 0)
        {
            int algo{-1}, proto{-1}, nChannels{};
            ASSERT_MPI_EQ(ncclSuccess,
                          rcclGetCollImplInfo(getActiveCommunicator(), ncclFuncReduceScatter, kRecvCount, ncclFloat,
                                              ncclSum, sendSym.ptr, recvSym.ptr, /*graphCapturing=*/0, &algo, &proto,
                                              &nChannels));
            const bool onPath
                = algo == static_cast<int>(RCCL_SYMMETRIC) && proto == NCCL_PROTO_LL && nChannels == kCtas;
            if(!onPath)
            {
                ADD_FAILURE() << "expected SYM/LL on " << kCtas << " block, got algo=" << algo << " proto=" << proto
                              << " blocks=" << nChannels;
            }
            ASSERT_MPI_TRUE(onPath);
            ASSERT_MPI_TRUE(getActiveCommunicator()->symkState.initialized);
        }

        const size_t base     = static_cast<size_t>(rank) * kRecvCount;
        auto         expected = [nRanks, base, call, &value](size_t j) {
            float sum = 0.0f;
            for(int s = 0; s < nRanks; ++s)
            {
                sum += value(s, base + j, call);
            }
            return sum;
        };
        size_t errIdx{};
        float  expVal{}, actVal{};
        bool ok = verifyBufferData<float>(recvSym.ptr, kRecvCount, expected, 0, 0, &errIdx, &expVal, &actVal);
        if(!ok)
        {
            ADD_FAILURE() << "ReduceScatter_LL mismatch at call " << call << " index=" << errIdx
                          << " expected=" << expVal << " got=" << actVal;
        }
        ASSERT_MPI_TRUE(ok);

        if(call == 2)
        {
            // Every peer write to this rank's slots was received before its kernel returned.
            ASSERT_MPI_TRUE(inspect());
            if(hostHeaders[0] != 0u || hostNonzero != 0u)
            {
                ADD_FAILURE() << "after the wrap block 0 header=" << hostHeaders[0] << " (want 0), nonzero slot lines="
                              << hostNonzero << " (want 0)";
            }
            ASSERT_MPI_TRUE(hostHeaders[0] == 0u && hostNonzero == 0u);
            // Only block 0 ran, so every other block keeps the seed.
            bool othersSeeded = true;
            for(int b = 1; b < ncclSymkMaxBlocks; ++b)
            {
                othersSeeded = othersSeeded && hostHeaders[b] == kSeedHeader;
            }
            ASSERT_MPI_TRUE(othersSeeded);
        }
    }

    // The call after the wrap ran at epoch 2.
    ASSERT_MPI_TRUE(inspect());
    ASSERT_MPI_EQ(1u, hostHeaders[0]);
}

// ===========================================================================
// Test group 6: AllReduce deep loop peer rotation
//
// allreduceDeep reduces and stores peers in batches of UnrollPeers (2 or 4 on
// gfx942, 8 on gfx950) plus one partial batch. At 8 ranks the partial store
// batch is empty, so CI runs this at 3, 5 and 7 ranks.
// ===========================================================================

// Generated by NCCL_PARAM(SymCTAs, ...) in src/tuning/sym_model.cc.
int64_t ncclParamSymCTAs();

namespace
{

// Rank weight 8^r keeps every peer multiset distinct up to 8 ranks; sums stay below 2^24, exact in float.
inline float arDeepValue(int rank, size_t idx)
{
    return static_cast<float>((size_t{1} << (3 * rank)) * (1 + idx % 5));
}

inline float arDeepSum(int nRanks, size_t idx)
{
    float sum = 0.0f;
    for(int r = 0; r < nRanks; ++r)
    {
        sum += arDeepValue(r, idx);
    }
    return sum;
}

} // namespace

// With 2 blocks this count runs several deep-loop iterations per warp in the 16-byte tier and,
// on gfx942, in the 4-byte tier on what the trim leaves; a 4-byte input skew runs only the 4-byte tier.
TEST_F(SymmetricKernelCorruptionTest, AllReduceDeep_PeerRotationTail)
{
    constexpr int kMaxRanks = 8;
    if(!validateTestPrerequisites(2, kMaxRanks, false, 1, 1))
    {
        GTEST_SKIP() << "Need 2 to " << kMaxRanks << " MPI ranks on a single node";
    }

    // NCCL_SYM_CTAS is cached on first read, so CI also sets it at launch.
    constexpr int           kCtas = 2;
    MPIHelpers::MpiEnvGuard ctasGuard("NCCL_SYM_CTAS", std::to_string(kCtas).c_str());
    const std::string       staleCtas
        = mpiCoordinatedSkipReason(ncclParamSymCTAs() != kCtas,
                                   "NCCL_SYM_CTAS already cached this process; run this test in its own mpirun");
    if(!staleCtas.empty())
    {
        GTEST_SKIP() << staleCtas;
    }

    // Read at communicator init, so the guard must outlive createTestCommunicator.
    MPIHelpers::MpiEnvGuard symKernelGuard("NCCL_SYM_KERNEL", "AllReduce_RSxLD_AGxST");
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{};
    int nRanks{};
    ASSERT_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank));
    ASSERT_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks));

    // 65 x 16 KiB or 261 x 4 KiB chunks plus a tail; odd chunk counts, so the gfx942 trim to nRanks * 2 always cuts.
    constexpr size_t kCount    = 267267;
    constexpr size_t kSkewElts = 1;

    SymBuf            sendSym;
    SymBuf            recvSym;
    const std::string noSym = allocSymBufsSkipReason(
        {{(kCount + kSkewElts) * sizeof(float), &sendSym}, {kCount * sizeof(float), &recvSym}});
    if(!noSym.empty())
    {
        GTEST_SKIP() << noSym;
    }

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommQueryProperties(getActiveCommunicator(), &props));
    int symmetric = props.deviceApiSupport ? 1 : 0;
    ASSERT_EQ(MPI_SUCCESS, MPI_Allreduce(MPI_IN_PLACE, &symmetric, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if(!symmetric)
    {
        GTEST_SKIP() << "Symmetric kernels unavailable (symmetricSupport off, e.g. cuMem/VMM unavailable)";
    }

    bool first = true;
    for(size_t skew : {size_t{0}, kSkewElts})
    {
        float* send = static_cast<float*>(sendSym.ptr) + skew;
        float* recv = static_cast<float*>(recvSym.ptr);

        ASSERT_MPI_EQ(hipSuccess,
                      initializeBufferWithPattern<float>(
                          send, kCount, [rank](size_t i) { return arDeepValue(rank, i); }));
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(recv, kCount, [](size_t) { return -1.0f; }));

        // Only the first call may skip on ncclInvalidUsage; agree collectively so diverging verdicts cannot hang.
        ncclResult_t res = ncclAllReduce(send, recv, kCount, ncclFloat, ncclSum, getActiveCommunicator(),
                                         getActiveStream());
        hipError_t syncErr = (res == ncclSuccess) ? hipStreamSynchronize(getActiveStream()) : hipSuccess;
        const std::string ineligible
            = mpiCoordinatedSkipReason(res == ncclInvalidUsage && first,
                                       "AllReduce_RSxLD_AGxST symmetric kernel not eligible on this topology");
        if(!ineligible.empty())
        {
            GTEST_SKIP() << ineligible;
        }
        first = false;
        ASSERT_MPI_EQ(ncclSuccess, res);
        ASSERT_MPI_EQ(hipSuccess, syncErr);

        // The forced kernel is the only symmetric Simple AllReduce left enabled.
        int algo{-1}, proto{-1}, nChannels{};
        ASSERT_MPI_EQ(ncclSuccess,
                      rcclGetCollImplInfo(getActiveCommunicator(), ncclFuncAllReduce, kCount, ncclFloat, ncclSum, send,
                                          recv, /*graphCapturing=*/0, &algo, &proto, &nChannels));
        const bool onPath = algo == static_cast<int>(RCCL_SYMMETRIC) && proto == NCCL_PROTO_SIMPLE
                            && nChannels == kCtas;
        if(!onPath)
        {
            ADD_FAILURE() << "skew=" << skew << ": expected SYM/Simple on " << kCtas << " blocks, got algo=" << algo
                          << " proto=" << proto << " blocks=" << nChannels;
        }
        ASSERT_MPI_TRUE(onPath);

        size_t errIdx{};
        float  expVal{}, actVal{};
        bool   ok = verifyBufferData<float>(
            recv, kCount, [nRanks](size_t i) { return arDeepSum(nRanks, i); }, 0, 0, &errIdx, &expVal, &actVal);
        if(!ok)
        {
            ADD_FAILURE() << "AllReduce_RSxLD_AGxST mismatch at nRanks=" << nRanks << " skew=" << skew
                          << " index=" << errIdx << " expected=" << expVal << " got=" << actVal;
        }
        ASSERT_MPI_TRUE(ok);
    }
}

#endif // MPI_TESTS_ENABLED
