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

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "SymmetricBufferHelpers.hpp"
#include "TestChecks.hpp"
#include "nccl_device.h"
#include "rccl/rccl.h"
#include "rccl_common.h"  // rcclSymKGetInfo, rcclAddonAlgos_t

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
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
                int srcRank = static_cast<int>(idx / count);
                return static_cast<float>((srcRank + 1) * 100 + i * 10 + (idx % 10));
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
// send path lands a wrong value in a checked output element.
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
        GTEST_SKIP() << "Symmetric kernels unavailable (symmetricSupport off, e.g. cuMem needs Linux >= 6.8)";
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

// On gfx950 AllGather moves from LL to the store kernel at 4 MB of bus bytes. The data tests above
// pass on either kernel, so this asks the symmetric tuner through the reporter rccl-tests uses.
TEST_F(SymmetricKernelCorruptionTest, AllGather_StoreKernelFrom4MB)
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
    constexpr size_t kStoreFromBusBytes = 4 << 20;
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
    EXPECT_EQ(static_cast<int>(RCCL_SYMMETRIC), algo) << "AllGather just below 4 MB left the symmetric kernels";
    EXPECT_EQ(NCCL_PROTO_LL, proto) << "AllGather just below 4 MB should stay on LL";

    ASSERT_EQ(ncclSuccess,
              rcclSymKGetInfo(getActiveCommunicator(), ncclFuncAllGather, switchCount, ncclFloat,
                              ncclSum, &algo, &proto, &nChannels));
    EXPECT_EQ(static_cast<int>(RCCL_SYMMETRIC), algo) << "AllGather at 4 MB left the symmetric kernels";
    EXPECT_EQ(NCCL_PROTO_SIMPLE, proto) << "AllGather at 4 MB should take the store kernel";
}

#endif // MPI_TESTS_ENABLED
