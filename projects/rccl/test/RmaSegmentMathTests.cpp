/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "transport/net_ib_multiseg/seg_reg_mseg.h"
#include "transport/net_ib_multiseg/seg_rma_mseg.h"
#include "transport/net_ib_multiseg/seg_rma_round_mseg.h"
#include "transport/net_ib_multiseg/seg_sym_mseg.h"
#include "transport/net_ib_multiseg/seg_wr_mseg.h"
#include "transport/rccl_mseg_rma.h"

namespace {

void FillUniformOffsets(size_t* off, int nSeg, size_t segBytes)
{
    for (int i = 0; i <= nSeg; ++i) off[i] = static_cast<size_t>(i) * segBytes;
}

rcclIbSegTable Table(const size_t* off, int nSeg, uint64_t base = 0)
{
    return {base, static_cast<uint32_t>(nSeg), reinterpret_cast<const uint64_t*>(off)};
}

// Slices the production cursor takes for `size` bytes; -1 if the range runs off a table.
int CursorSlices(const size_t* localOff, int nLocal, const size_t* remoteOff, int nRemote, uint64_t lOff,
                 uint64_t rOff, uint64_t size, std::vector<rcclIbSegSlice>* out = nullptr)
{
    const rcclIbSegTable local = Table(localOff, nLocal);
    const rcclIbSegTable remote = Table(remoteOff, nRemote);
    rcclIbSegCursor cur;
    rcclIbSegCursorInit(&cur, &local, &remote, lOff, rOff, size);
    rcclIbSegSlice slice;
    int n = 0;
    while (rcclIbSegCursorNext(&cur, &local, &remote, &slice)) {
        if (out != nullptr) out->push_back(slice);
        n++;
    }
    return cur.remaining == 0 ? n : -1;
}

int CountLayout(const size_t* localOff, int nLocal, const size_t* remoteOff, int nRemote)
{
    const size_t size = localOff[nLocal] < remoteOff[nRemote] ? localOff[nLocal] : remoteOff[nRemote];
    return CursorSlices(localOff, nLocal, remoteOff, nRemote, 0, 0, size);
}

} // namespace

TEST(RmaSegmentMathTest, SplitsAtVerbsLengthLimitWithoutLargeAllocation)
{
    ASSERT_GT(SIZE_MAX, static_cast<size_t>(UINT32_MAX));
    const size_t off[] = {0, static_cast<size_t>(UINT32_MAX) + 1};
    std::vector<rcclIbSegSlice> slices;
    ASSERT_EQ(CursorSlices(off, 1, off, 1, 0, 0, off[1], &slices), 2);
    EXPECT_EQ(slices[0].len, static_cast<uint64_t>(UINT32_MAX));
    EXPECT_EQ(slices[1].len, uint64_t{1});
    EXPECT_EQ(slices[1].localOff, static_cast<uint64_t>(UINT32_MAX));
}

// The builder fills at most maxWr WRs; a longer chain is rejected rather than truncated.
TEST(RmaSegmentMathTest, FixedDataWrBudgetRejectsUnrepresentableChain)
{
    ASSERT_GT(SIZE_MAX, static_cast<size_t>(UINT32_MAX));
    const size_t exact = static_cast<size_t>(UINT32_MAX) * RCCL_RMA_MAX_DATA_WRS;
    const size_t off[] = {0, exact + static_cast<size_t>(UINT32_MAX)};
    const rcclIbSegTable table = Table(off, 1);
    const uint32_t keys[] = {1};
    std::vector<ibv_send_wr> wrs(RCCL_RMA_MAX_DATA_WRS);
    std::vector<ibv_sge> sges(RCCL_RMA_MAX_DATA_WRS);
    rcclIbSegWrIn in = {};
    in.local = &table;
    in.lkeys = keys;
    in.remote = &table;
    in.rkeys = keys;
    in.opcode = IBV_WR_RDMA_WRITE;
    in.maxWr = RCCL_RMA_MAX_DATA_WRS;
    for (auto [len, expect] : {std::pair<size_t, int>{0, 0}, {1, 1}, {exact, RCCL_RMA_MAX_DATA_WRS}, {exact + 1, -1}}) {
        in.len = len;
        EXPECT_EQ(rcclIbSegAppendWrs(&in, wrs.data(), sges.data()), expect) << "len " << len;
    }
}

// Aligned 4/8/16-segment 8 GiB windows on matching peers. 16x8 GiB needs 48 WRs.
TEST(RmaSegmentMathTest, HomogeneousSameKindNodesFitDataWrBudget)
{
    ASSERT_GT(SIZE_MAX, static_cast<size_t>(UINT32_MAX));
    const size_t eightGiB = size_t{8} << 30;
    const size_t twoMiB = size_t{2} << 20;
    const int twoPerSegmentBudget = 2 * 16;
    size_t segOff[16 + 1];

    for (int nSeg : {4, 8, 16})
    {
        FillUniformOffsets(segOff, nSeg, eightGiB);
        const int wrs = CountLayout(segOff, nSeg, segOff, nSeg);
        EXPECT_EQ(wrs, nSeg * 3) << "aligned 8 GiB same-kind, nSeg=" << nSeg;
        EXPECT_LE(wrs, RCCL_RMA_MAX_DATA_WRS);
        if (nSeg <= 8)
        {
            EXPECT_LE(wrs, twoPerSegmentBudget) << "4/8 GPU same-kind 8 GiB windows fit two WRs per segment";
        }
        else
        {
            EXPECT_GT(wrs, twoPerSegmentBudget);
            EXPECT_EQ(wrs, 48);
        }

        FillUniformOffsets(segOff, nSeg, twoMiB);
        EXPECT_EQ(CountLayout(segOff, nSeg, segOff, nSeg), nSeg);
    }
}

TEST(RmaSegmentMathTest, OnlyFinalWrIsSignaled)
{
    constexpr int kWrs = 7;
    for (int i = 0; i < kWrs; ++i)
        EXPECT_EQ(rcclRmaWrIsSignaled(i, kWrs), i == kWrs - 1);
    EXPECT_FALSE(rcclRmaWrIsSignaled(0, 0));
}

TEST(RmaSegmentMathTest, SignalAtomicMustBeAlignedWithinSegment)
{
    EXPECT_TRUE(rcclRmaSignalOffsetValid(/*signalOff=*/8, /*segmentEnd=*/16));
    EXPECT_FALSE(rcclRmaSignalOffsetValid(/*signalOff=*/4, /*segmentEnd=*/16));
    EXPECT_FALSE(rcclRmaSignalOffsetValid(/*signalOff=*/8, /*segmentEnd=*/12));
}

namespace {

struct FakeWr
{
    int id;
    FakeWr* next;
};

int Posted(FakeWr* wr, int nWr, FakeWr* badWr)
{
    return rcclRmaPostedWrCount(wr, nWr, badWr, offsetof(FakeWr, next));
}

} // namespace

// wrap_ibv_post_send fails even after a prefix of the chain is accepted. Only
// the last WR is signaled, so posted==0 frees the request and posted>0 keeps it.
TEST(RmaSegmentMathTest, PrefixPostCountsWrsBeforeBadWr)
{
    FakeWr wr[3];
    wr[0] = {0, &wr[1]};
    wr[1] = {1, &wr[2]};
    wr[2] = {2, nullptr};

    EXPECT_EQ(Posted(&wr[0], 3, nullptr), 3);
    EXPECT_EQ(Posted(&wr[0], 3, &wr[0]), 0);
    EXPECT_EQ(Posted(&wr[0], 3, &wr[1]), 1);
    EXPECT_EQ(Posted(&wr[0], 3, &wr[2]), 2);
    EXPECT_EQ(Posted(&wr[0], 2, nullptr), 2);
    EXPECT_EQ(Posted(&wr[0], 0, &wr[0]), 0);
    EXPECT_EQ(Posted(nullptr, 3, &wr[0]), 0);
}

TEST(RmaSegmentMathTest, PrefixPostKeepsRequestOnlyWhenSomethingPosted)
{
    int keep = 0, failed = 0;
    EXPECT_EQ(rcclRmaCompletePostedRequest(ncclSystemError, /*posted=*/1, /*nWr=*/3, &keep, &failed), ncclSuccess);
    EXPECT_TRUE(keep);
    EXPECT_TRUE(failed);
    EXPECT_EQ(rcclRmaCompletePostedRequest(ncclSystemError, /*posted=*/0, /*nWr=*/3, &keep, &failed), ncclSystemError);
    EXPECT_FALSE(keep);
    EXPECT_FALSE(failed);
    EXPECT_EQ(rcclRmaCompletePostedRequest(ncclSuccess, /*posted=*/0, /*nWr=*/0, &keep, &failed), ncclSuccess);
    EXPECT_TRUE(keep);
    EXPECT_FALSE(failed);
    EXPECT_EQ(rcclRmaCompletePostedRequest(ncclSuccess, /*posted=*/2, /*nWr=*/2, &keep, &failed), ncclSuccess);
    EXPECT_TRUE(keep);
    EXPECT_FALSE(failed);
    // A batched chain counts as queued before its doorbell fails; the request must still fail.
    EXPECT_EQ(rcclRmaCompletePostedRequest(ncclSystemError, /*posted=*/3, /*nWr=*/3, &keep, &failed), ncclSuccess);
    EXPECT_TRUE(keep);
    EXPECT_TRUE(failed);
    EXPECT_TRUE(rcclRmaPrefixPostLostSignaledTail(/*posted=*/1, /*nWr=*/3));
    EXPECT_FALSE(rcclRmaPrefixPostLostSignaledTail(/*posted=*/3, /*nWr=*/3));
    EXPECT_FALSE(rcclRmaPrefixPostLostSignaledTail(/*posted=*/0, /*nWr=*/3));
    EXPECT_FALSE(rcclRmaPrefixPostLostSignaledTail(/*posted=*/0, /*nWr=*/0));
}

TEST(RmaSegmentMathTest, SegCountSupportedIsTheOnlyCap)
{
    EXPECT_FALSE(rcclIbSegCountSupported(0));
    EXPECT_FALSE(rcclIbSegCountSupported(-1));
    EXPECT_TRUE(rcclIbSegCountSupported(1));
    EXPECT_TRUE(rcclIbSegCountSupported(17));
    EXPECT_TRUE(rcclIbSegCountSupported(RCCL_IB_REG_SEG_DEFAULT));
    EXPECT_FALSE(rcclIbSegCountSupported(RCCL_IB_REG_SEG_DEFAULT + 1));
}

// RMA bounds ranks x segments: the full count at small scale, fewer segments as ranks grow, one always.
TEST(RmaSegmentMathTest, SegTableFitsBoundsRanksTimesSegments)
{
    EXPECT_TRUE(rcclRmaSegTableFits(1024, RCCL_IB_REG_SEG_DEFAULT));
    EXPECT_FALSE(rcclRmaSegTableFits(1025, RCCL_IB_REG_SEG_DEFAULT));
    EXPECT_TRUE(rcclRmaSegTableFits(2048, 512));
    EXPECT_FALSE(rcclRmaSegTableFits(2049, 512));
    EXPECT_TRUE(rcclRmaSegTableFits(80000, 13));
    EXPECT_FALSE(rcclRmaSegTableFits(80000, 14));
    EXPECT_TRUE(rcclRmaSegTableFits(80000, 1));
    EXPECT_TRUE(rcclRmaSegTableFits(INT_MAX, 1));
    EXPECT_FALSE(rcclRmaSegTableFits(2, RCCL_IB_REG_SEG_DEFAULT + 1));
    EXPECT_FALSE(rcclRmaSegTableFits(0, 1));
    EXPECT_FALSE(rcclRmaSegTableFits(2, 0));
}

TEST(RmaSegmentMathTest, PeerSegOffIndexesPerRankTable)
{
    size_t local[] = {0, 100};
    // Two ranks with three segments each: the stride is nSegments + 1, not the count cap.
    size_t table[] = {0, 4096, 8192, 12288, 0, 1024, 2048, 3072};
    EXPECT_EQ(rcclRmaPeerSegOff(nullptr, local, 0, 3), local);
    EXPECT_EQ(rcclRmaPeerSegOff(table, local, -1, 3), local);
    EXPECT_EQ(rcclRmaPeerSegOff(table, local, 0, 0), local);
    EXPECT_EQ(rcclRmaPeerSegOff(table, local, 0, 3)[1], size_t{4096});
    EXPECT_EQ(rcclRmaPeerSegOff(table, local, 1, 3), table + 4);
    EXPECT_EQ(rcclRmaPeerSegOff(table, local, 1, 3)[3], size_t{3072});
}

// One cursor walks the whole transfer, splitting wherever either side's segment ends.
TEST(RmaSegmentMathTest, CursorSplitsAtBothSidesBoundaries)
{
    const size_t local[] = {0, 100, 250};
    const size_t remote[] = {0, 60, 250};
    const uint64_t expectBytes[] = {60, 40, 150};
    const int expectLocal[] = {0, 0, 1};
    const int expectRemote[] = {0, 1, 1};
    std::vector<rcclIbSegSlice> slices;
    ASSERT_EQ(CursorSlices(local, 2, remote, 2, 0, 0, 250, &slices), 3);
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(slices[i].len, expectBytes[i]) << "slice " << i;
        EXPECT_EQ(slices[i].localSeg, expectLocal[i]) << "slice " << i;
        EXPECT_EQ(slices[i].remoteSeg, expectRemote[i]) << "slice " << i;
    }
}

TEST(RmaSegmentMathTest, AppendWrsChainsOneWritePerSlice)
{
    const uint64_t localOff[] = {0, 100, 250};
    const uint64_t remoteOff[] = {0, 60, 250};
    const struct rcclIbSegTable local = {0x1000, 2, localOff};
    const struct rcclIbSegTable remote = {0x8000, 2, remoteOff};
    const uint32_t lkeys[] = {11, 22};
    const uint32_t rkeys[] = {33, 44};
    struct ibv_send_wr wrs[4];
    struct ibv_sge sges[4];
    struct rcclIbSegWrIn in = {};
    in.local = &local;
    in.lkeys = lkeys;
    in.remote = &remote;
    in.rkeys = rkeys;
    in.len = 250;
    in.opcode = IBV_WR_RDMA_WRITE;
    in.maxWr = 4;
    const int n = rcclIbSegAppendWrs(&in, wrs, sges);
    ASSERT_EQ(n, 3);
    EXPECT_EQ(sges[0].length, 60u);
    EXPECT_EQ(sges[0].lkey, 11u);
    EXPECT_EQ(sges[0].addr, static_cast<uintptr_t>(0x1000));
    EXPECT_EQ(wrs[0].wr.rdma.rkey, 33u);
    EXPECT_EQ(wrs[0].wr.rdma.remote_addr, static_cast<uint64_t>(0x8000));
    EXPECT_EQ(sges[1].length, 40u);
    EXPECT_EQ(wrs[1].wr.rdma.rkey, 44u);
    EXPECT_EQ(sges[1].addr, static_cast<uintptr_t>(0x1000 + 60));
    EXPECT_EQ(sges[2].lkey, 22u);
    EXPECT_EQ(sges[2].length, 150u);
    EXPECT_EQ(wrs[0].next, &wrs[1]);
    EXPECT_EQ(wrs[1].next, &wrs[2]);
    EXPECT_EQ(wrs[2].next, nullptr);
    EXPECT_EQ(wrs[0].opcode, IBV_WR_RDMA_WRITE);
    in.maxWr = 2;
    EXPECT_EQ(rcclIbSegAppendWrs(&in, wrs, sges), -1);

    // With covered set, the same chain stops at maxWr and reports the bytes it moves.
    uint64_t covered = 0;
    in.covered = &covered;
    ASSERT_EQ(rcclIbSegAppendWrs(&in, wrs, sges), 2);
    EXPECT_EQ(covered, 100u);
    EXPECT_EQ(wrs[1].next, nullptr);
    in.maxWr = 4;
    ASSERT_EQ(rcclIbSegAppendWrs(&in, wrs, sges), 3);
    EXPECT_EQ(covered, 250u);
}

TEST(RmaSegmentMathTest, CursorStartsMidSegmentAndCapsAtSgeLength)
{
    ASSERT_GT(SIZE_MAX, static_cast<size_t>(UINT32_MAX));
    const size_t big = static_cast<size_t>(UINT32_MAX) + 4096;
    const size_t local[] = {0, big};
    const size_t remote[] = {0, 512, big};
    std::vector<rcclIbSegSlice> slices;
    ASSERT_EQ(CursorSlices(local, 1, remote, 2, 100, 100, 1000, &slices), 2);
    EXPECT_EQ(slices[0].len, uint64_t{412});
    EXPECT_EQ(slices[0].remoteSeg, 0);
    slices.clear();
    ASSERT_EQ(CursorSlices(local, 1, remote, 2, 512, 512, big - 512, &slices), 2);
    EXPECT_EQ(slices[0].len, static_cast<uint64_t>(UINT32_MAX));
    EXPECT_EQ(slices[0].remoteSeg, 1);
}

// A zero slice would spin the builder forever; the cursor stops and leaves the rest unconsumed instead.
TEST(RmaSegmentMathTest, CursorStopsPastEitherTable)
{
    const size_t local[] = {0, 100};
    const size_t remote[] = {0, 200};
    EXPECT_EQ(CursorSlices(local, 1, remote, 1, 100, 0, 50), -1);
    EXPECT_EQ(CursorSlices(local, 1, remote, 1, 0, 200, 50), -1);
    std::vector<rcclIbSegSlice> slices;
    EXPECT_EQ(CursorSlices(local, 1, remote, 1, 0, 0, 150, &slices), -1);
    ASSERT_EQ(slices.size(), 1u);
    EXPECT_EQ(slices[0].len, uint64_t{100});
}

// Asymmetric windows: three local segments against one remote segment is three slices.
TEST(RmaSegmentMathTest, CursorLocalHasMoreSegmentsThanRemote)
{
    const size_t local[] = {0, 50, 120, 300};
    const size_t remote[] = {0, 300};
    const uint64_t expectBytes[] = {50, 70, 180};
    std::vector<rcclIbSegSlice> slices;
    ASSERT_EQ(CursorSlices(local, 3, remote, 1, 0, 0, 300, &slices), 3);
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(slices[i].len, expectBytes[i]) << "slice " << i;
        EXPECT_EQ(slices[i].localSeg, i) << "slice " << i;
        EXPECT_EQ(slices[i].remoteSeg, 0) << "slice " << i;
    }
}

// With multi-segment chains the proxies stop at RCCL_RMA_MAX_INFLIGHT_REQUESTS per peer; otherwise their own limit.
TEST(RmaSegmentMathTest, InflightCapAppliesOnlyToMultiSegmentChains)
{
    EXPECT_EQ(rcclRmaInflightRequestsCap(false, RCCL_RMA_MAX_REQUESTS), static_cast<uint32_t>(RCCL_RMA_MAX_REQUESTS));
    EXPECT_EQ(rcclRmaInflightRequestsCap(true, RCCL_RMA_MAX_REQUESTS),
              static_cast<uint32_t>(RCCL_RMA_MAX_INFLIGHT_REQUESTS));
    constexpr uint32_t kSmallerLimit = RCCL_RMA_MAX_INFLIGHT_REQUESTS / 2;
    EXPECT_EQ(rcclRmaInflightRequestsCap(true, kSmallerLimit), kSmallerLimit) << "a smaller proxy limit stays";
}

struct SegCountPrefix
{
    uint32_t tag;
    int nSegments;
};

TEST(RmaSegmentMathTest, SegmentCountNullAndUntaggedAreZero)
{
    EXPECT_EQ(rcclRmaMrSegmentCount(nullptr), 0);
    SegCountPrefix wrong{0x11111111u, 4};
    EXPECT_EQ(rcclRmaMrSegmentCount(&wrong), 0);
    uintptr_t pointerShaped = static_cast<uintptr_t>(0x7f0000001000ull);
    EXPECT_EQ(rcclRmaMrSegmentCount(&pointerShaped), 0);
}

TEST(RmaSegmentMathTest, SegmentCountReadsTaggedHandle)
{
    SegCountPrefix one{rcclRmaSegMrTag(), 1};
    SegCountPrefix four{rcclRmaSegMrTag(), 4};
    SegCountPrefix cap{rcclRmaSegMrTag(), RCCL_IB_REG_SEG_DEFAULT};
    EXPECT_EQ(rcclRmaMrSegmentCount(&one), 1);
    EXPECT_EQ(rcclRmaMrSegmentCount(&four), 4);
    EXPECT_EQ(rcclRmaMrSegmentCount(&cap), RCCL_IB_REG_SEG_DEFAULT);
    EXPECT_FALSE(rcclRmaMrSegmentCount(&one) > 1);
    EXPECT_TRUE(rcclRmaMrSegmentCount(&four) > 1);
    EXPECT_TRUE(rcclRmaMrSegmentCount(&cap) > 1);
}

TEST(RmaSegmentMathTest, SegmentCountKeepsZeroAndNegative)
{
    SegCountPrefix zero{rcclRmaSegMrTag(), 0};
    SegCountPrefix negative{rcclRmaSegMrTag(), -1};
    EXPECT_EQ(rcclRmaMrSegmentCount(&zero), 0);
    EXPECT_EQ(rcclRmaMrSegmentCount(&negative), -1);
    EXPECT_FALSE(rcclRmaMrSegmentCount(&zero) > 1);
    EXPECT_FALSE(rcclRmaMrSegmentCount(&negative) > 1);
}

namespace {

struct RegTrace
{
    int regIndex[8];
    int deregValue[8];
    int nReg = 0;
    int nDereg = 0;
    int failAt = -1;
    bool storeOnFail = false;
};

ncclResult_t TraceReg(void* ctx, int index, void** mr)
{
    RegTrace* trace = static_cast<RegTrace*>(ctx);
    if (index == trace->failAt)
    {
        if (trace->storeOnFail) *mr = reinterpret_cast<void*>(static_cast<uintptr_t>(index + 1));
        return ncclInvalidUsage;
    }
    trace->regIndex[trace->nReg++] = index;
    *mr = reinterpret_cast<void*>(static_cast<uintptr_t>(index + 1));
    return ncclSuccess;
}

ncclResult_t TraceDereg(void* ctx, int, void* mr)
{
    RegTrace* trace = static_cast<RegTrace*>(ctx);
    trace->deregValue[trace->nDereg++] = static_cast<int>(reinterpret_cast<uintptr_t>(mr));
    return ncclSuccess;
}

ncclResult_t DeregFirstError(void* ctx, int, void* mr)
{
    int* calls = static_cast<int*>(ctx);
    ++(*calls);
    if (mr == reinterpret_cast<void*>(static_cast<uintptr_t>(4))) return ncclSystemError;
    return ncclInvalidUsage;
}

} // namespace

TEST(RmaSegmentMathTest, RegisterFailureReleasesPartialSlots)
{
    RegTrace trace;
    trace.failAt = 2;
    trace.storeOnFail = true;
    void* mrs[4] = {reinterpret_cast<void*>(1), reinterpret_cast<void*>(1), reinterpret_cast<void*>(1),
                    reinterpret_cast<void*>(1)};
    rcclIbSegRegOps ops{TraceReg, TraceDereg};

    EXPECT_EQ(rcclIbSegRegister(&ops, &trace, mrs, 4), ncclInvalidUsage);
    EXPECT_EQ(trace.nReg, 2);
    EXPECT_EQ(trace.regIndex[0], 0);
    EXPECT_EQ(trace.regIndex[1], 1);
    EXPECT_EQ(trace.nDereg, 3);
    EXPECT_EQ(trace.deregValue[0], 1);
    EXPECT_EQ(trace.deregValue[1], 2);
    EXPECT_EQ(trace.deregValue[2], 3);
    for (int i = 0; i < 3; ++i) EXPECT_EQ(mrs[i], nullptr);
    EXPECT_EQ(mrs[3], reinterpret_cast<void*>(1));
}

TEST(RmaSegmentMathTest, DeregisterSkipsNullAndKeepsFirstError)
{
    void* mrs[3] = {reinterpret_cast<void*>(static_cast<uintptr_t>(4)), nullptr,
                    reinterpret_cast<void*>(static_cast<uintptr_t>(6))};
    int calls = 0;
    rcclIbSegRegOps ops{nullptr, DeregFirstError};

    EXPECT_EQ(rcclIbSegDeregister(&ops, &calls, mrs, 3), ncclSystemError);
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(mrs[0], nullptr);
    EXPECT_EQ(mrs[2], nullptr);
    EXPECT_EQ(rcclIbSegDeregister(&ops, &calls, mrs, 0), ncclSuccess);
}

namespace {

// Two-rank allGather: rank 0 contributes src, rank 1 the same bytes unless a test edits them after the copy.
struct SymGather {
    int calls = 0;
    std::vector<size_t> sizes;
    std::function<void(int, void*, size_t)> peer;
};

ncclResult_t SymAllGather(void* ctx, void* src, void* dst, size_t bytes)
{
    auto* g = static_cast<SymGather*>(ctx);
    g->sizes.push_back(bytes);
    std::memcpy(dst, src, bytes);
    std::memcpy(static_cast<char*>(dst) + bytes, src, bytes);
    if (g->peer) g->peer(g->calls, static_cast<char*>(dst) + bytes, bytes);
    g->calls++;
    return ncclSuccess;
}

struct SymLocal {
    static constexpr int kSeg = 2;
    static constexpr int kKeys = 2;
    size_t off[kSeg + 1] = {0, 0x1000, 0x1800};
    uintptr_t vas[kSeg] = {0x10000, 0x20000};
    uint64_t keys[kSeg * kKeys] = {0xa0, 0xa1, 0xb0, 0xb1};
    rcclIbSegSymRecord records[2] = {};
    rcclIbSegSymTable t = {};
    ~SymLocal() { rcclIbSegSymTableFree(&t); }
    ncclResult_t Exchange(SymGather* g, ncclResult_t status = ncclSuccess)
    {
        return rcclIbSegSymExchange(SymAllGather, g, 2, records, status, kSeg, kKeys, off, vas, keys, this, "test",
                                    &t);
    }
};

} // namespace

TEST(RmaSegmentMathTest, SymExchangeBuildsEveryRanksTable)
{
    SymLocal l;
    SymGather g;
    g.peer = [](int call, void* peer, size_t) {
        if (call == 3) static_cast<uintptr_t*>(peer)[1] = 0x90000;
        if (call == 4) static_cast<uint64_t*>(peer)[3] = 0xc1;
    };

    ASSERT_EQ(l.Exchange(&g), ncclSuccess);
    ASSERT_EQ(g.sizes.size(), 5u);
    EXPECT_EQ(g.sizes[2], 3 * sizeof(size_t));
    EXPECT_EQ(g.sizes[4], SymLocal::kSeg * SymLocal::kKeys * sizeof(uint64_t));
    EXPECT_EQ(l.t.rankOff[3 + 2], size_t{0x1800});
    EXPECT_EQ(rcclIbSegSymVa(&l.t, SymLocal::kSeg, 1, 0), uintptr_t{0x10000});
    EXPECT_EQ(rcclIbSegSymVa(&l.t, SymLocal::kSeg, 1, 1), uintptr_t{0x90000});
    EXPECT_EQ(rcclIbSegSymKey(&l.t, SymLocal::kSeg, SymLocal::kKeys, 0, 1, 1), uint64_t{0xb1});
    EXPECT_EQ(rcclIbSegSymKey(&l.t, SymLocal::kSeg, SymLocal::kKeys, 1, 1, 1), uint64_t{0xc1});
}

TEST(RmaSegmentMathTest, SymExchangeFailsEveryRankWithTheRemoteError)
{
    SymLocal l;
    SymGather g;
    g.peer = [](int call, void* peer, size_t) {
        if (call == 0) static_cast<rcclIbSegSymRecord*>(peer)->status = ncclSystemError;
    };

    EXPECT_EQ(l.Exchange(&g), ncclSystemError);
    EXPECT_EQ(g.calls, 1) << "no table gather after a failed status";
    EXPECT_EQ(l.t.rankOff, nullptr);
}

TEST(RmaSegmentMathTest, SymExchangeCarriesALocalFailureThroughTheFirstGather)
{
    SymLocal l;
    SymGather g;

    EXPECT_EQ(l.Exchange(&g, ncclInvalidUsage), ncclInvalidUsage);
    EXPECT_EQ(g.calls, 1);
}

TEST(RmaSegmentMathTest, SymExchangeRejectsADifferentSegmentCount)
{
    SymLocal l;
    SymGather g;
    g.peer = [](int call, void* peer, size_t) {
        if (call == 0) static_cast<rcclIbSegSymRecord*>(peer)->nSegments = 3;
    };

    EXPECT_EQ(l.Exchange(&g), ncclInvalidUsage);
    EXPECT_EQ(g.calls, 1);
}

TEST(RmaSegmentMathTest, SymExchangeRejectsUnorderedPeerOffsets)
{
    // A decreasing entry, and a table that does not start at 0.
    for (auto [index, value] : {std::pair<int, size_t>{2, 0x800}, {0, 0x40}})
    {
        SymLocal l;
        SymGather g;
        g.peer = [index = index, value = value](int call, void* peer, size_t) {
            if (call == 2) static_cast<size_t*>(peer)[index] = value;
        };

        EXPECT_EQ(l.Exchange(&g), ncclInvalidUsage) << "offset " << index;
        EXPECT_EQ(g.calls, 3) << "VA and key gathers do not run";
    }
}

namespace {

// Fake plugin half of a round: every WR moves one byte, and posts are recorded.
struct RoundLog
{
    std::vector<int> posts;
    std::vector<std::pair<void*, bool>> retired;
    int failPostAt = -1;
    bool outstanding = true;
};
RoundLog* g_roundLog = nullptr;

ncclResult_t FakeRoundBuild(const rcclRmaRoundOp* op, ibv_send_wr* wr, ibv_sge* sge, int maxWr, int* nWr,
                            uint64_t* covered)
{
    int n = op->remaining < (uint64_t)maxWr ? (int)op->remaining : maxWr;
    for (int i = 0; i < n; ++i) {
        wr[i] = {};
        sge[i] = {};
        wr[i].sg_list = &sge[i];
        wr[i].next = i + 1 < n ? &wr[i + 1] : nullptr;
    }
    *nWr = n;
    *covered = n;
    return ncclSuccess;
}

bool FakeRoundOutstanding(const rcclRmaRoundOp*) { return g_roundLog->outstanding; }

ncclResult_t FakeRoundPost(rcclRmaRoundOp*, ibv_send_wr*, int nWr, int* posted)
{
    const bool fail = (int)g_roundLog->posts.size() == g_roundLog->failPostAt;
    g_roundLog->posts.push_back(nWr);
    *posted = fail ? 0 : nWr;
    return fail ? ncclSystemError : ncclSuccess;
}

void FakeRoundRetire(rcclRmaRoundOp* op, bool ok, int, int) { g_roundLog->retired.push_back({op->req, ok}); }

bool FakeRoundFailed(const rcclRmaRoundOp*) { return false; }

const rcclRmaRoundHooks kFakeRoundHooks = {FakeRoundBuild, FakeRoundOutstanding, FakeRoundPost, FakeRoundRetire,
                                           FakeRoundFailed};

class RmaRoundTest : public ::testing::Test
{
protected:
    void SetUp() override { g_roundLog = &log; }
    void TearDown() override
    {
        while (q.head != nullptr) rcclRmaRoundPop(&q);
        g_roundLog = nullptr;
    }
    rcclRmaRoundOp Op(uint64_t bytes, bool signal = false)
    {
        rcclRmaRoundOp op = rcclRmaRoundDataOp(nullptr, nullptr, IBV_WR_RDMA_WRITE, nullptr, 0, 0, nullptr, 1, 0, bytes);
        if (signal) rcclRmaRoundSetSignal(&op, 0x1000, 7, 1, 0x2000, 8, 9);
        return op;
    }
    // Queued ops are released with free(), so they come from calloc.
    static rcclRmaRoundOp* Node(const rcclRmaRoundOp& op)
    {
        auto* node = static_cast<rcclRmaRoundOp*>(calloc(1, sizeof(rcclRmaRoundOp)));
        rcclRmaRoundFillNode(node, &op);
        return node;
    }
    void Queue(void* req, uint64_t bytes, bool signal = false)
    {
        rcclRmaRoundOp* node = Node(Op(bytes, signal));
        node->req = req;
        rcclRmaRoundPush(&q, node);
    }
    RoundLog log;
    rcclRmaRoundQueue q{};
    int a = 0, b = 0, c = 0;
};

} // namespace

// The head posts one round at a time; its last round carries the signal and retires it, and the next op follows.
TEST_F(RmaRoundTest, PumpPostsOneRoundAtATimeAndRetiresOnTheLast)
{
    Queue(&a, 150, /*signal=*/true);
    Queue(&b, 10);
    rcclRmaRoundPump(&kFakeRoundHooks, &q);
    EXPECT_EQ(log.posts, std::vector<int>({64}));
    rcclRmaRoundPump(&kFakeRoundHooks, &q);
    EXPECT_EQ(log.posts.size(), 1u) << "the round in flight holds the queue";
    EXPECT_FALSE(rcclRmaRoundTailDone(&q, &b));
    for (int r = 0; r < 2; ++r) {
        ASSERT_TRUE(rcclRmaRoundTailDone(&q, &a));
        rcclRmaRoundPump(&kFakeRoundHooks, &q);
    }
    EXPECT_EQ(log.posts, std::vector<int>({64, 64, 23, 10}));
    ASSERT_EQ(log.retired.size(), 2u);
    EXPECT_EQ(log.retired[0], std::make_pair((void*)&a, true));
    EXPECT_EQ(log.retired[1], std::make_pair((void*)&b, true));
    EXPECT_EQ(q.head, nullptr);
    EXPECT_EQ(q.tail, nullptr);
}

// A plugin that tracks completion itself can let the next round post as soon as the previous one is done.
TEST_F(RmaRoundTest, PumpPostsTheNextRoundOnceThePluginSaysTheLastOneIsDone)
{
    Queue(&a, 130);
    rcclRmaRoundPump(&kFakeRoundHooks, &q);
    log.outstanding = false;
    rcclRmaRoundPump(&kFakeRoundHooks, &q);
    EXPECT_EQ(log.posts, std::vector<int>({64, 64}));
}

TEST_F(RmaRoundTest, FailedRoundRetiresItsOpAndTheNextOpStillPosts)
{
    log.failPostAt = 0;
    Queue(&a, 100);
    Queue(&b, 10);
    rcclRmaRoundPump(&kFakeRoundHooks, &q);
    EXPECT_EQ(log.posts, std::vector<int>({64, 10}));
    ASSERT_EQ(log.retired.size(), 2u);
    EXPECT_EQ(log.retired[0], std::make_pair((void*)&a, false));
    EXPECT_EQ(log.retired[1], std::make_pair((void*)&b, true));
    EXPECT_EQ(q.head, nullptr);
}

TEST_F(RmaRoundTest, FirstIsDirectOnlyForOneRoundWithNothingQueued)
{
    ibv_send_wr wr[RCCL_RMA_MAX_SIGNAL_WRS];
    ibv_sge sge[RCCL_RMA_MAX_SIGNAL_WRS];
    int nWr = -1;
    bool direct = false;
    rcclRmaRoundOp op = Op(10, /*signal=*/true);
    ASSERT_EQ(rcclRmaRoundFirst(&kFakeRoundHooks, &q, &op, wr, sge, &nWr, &direct), ncclSuccess);
    EXPECT_TRUE(direct);
    EXPECT_EQ(nWr, 11);
    EXPECT_EQ(wr[10].opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
    EXPECT_EQ(wr[10].sg_list, &sge[10]);
    EXPECT_EQ(wr[9].next, &wr[10]);
    EXPECT_EQ(op.remaining, 10u) << "the first round is built on a copy";
    op = Op(100);
    ASSERT_EQ(rcclRmaRoundFirst(&kFakeRoundHooks, &q, &op, wr, sge, &nWr, &direct), ncclSuccess);
    EXPECT_FALSE(direct);
    EXPECT_EQ(nWr, 64);
    Queue(&a, 10);
    op = Op(10);
    ASSERT_EQ(rcclRmaRoundFirst(&kFakeRoundHooks, &q, &op, wr, sge, &nWr, &direct), ncclSuccess);
    EXPECT_FALSE(direct) << "behind a queued op";
    op = Op(0);
    ASSERT_EQ(rcclRmaRoundFirst(&kFakeRoundHooks, &q, &op, wr, sge, &nWr, &direct), ncclSuccess);
    EXPECT_TRUE(direct) << "an empty op posts nothing";
    EXPECT_EQ(nWr, 0);
}

TEST_F(RmaRoundTest, FillNodeResetsTheLinkAndTheRoundState)
{
    rcclRmaRoundOp op = Op(10);
    op.next = &op;
    op.roundInFlight = true;
    rcclRmaRoundOp node;
    rcclRmaRoundFillNode(&node, &op);
    EXPECT_EQ(node.next, nullptr);
    EXPECT_FALSE(node.roundInFlight);
    EXPECT_EQ(node.remaining, 10u);
    EXPECT_FALSE(rcclRmaRoundQueueBusy(nullptr));
}

TEST_F(RmaRoundTest, QueuesFreeReleasesQueuedOps)
{
    rcclRmaRoundQueues qs{};
    qs.send = static_cast<rcclRmaRoundQueue*>(calloc(2, sizeof(rcclRmaRoundQueue)));
    rcclRmaRoundQueue* peer = rcclRmaRoundQueueOf(&qs, 1);
    EXPECT_TRUE(rcclRmaRoundPush(peer, Node(Op(10))));
    EXPECT_FALSE(rcclRmaRoundPush(peer, Node(Op(10))));
    EXPECT_TRUE(rcclRmaRoundQueueBusy(peer));
    EXPECT_FALSE(rcclRmaRoundQueueBusy(rcclRmaRoundQueueOf(&qs, 0)));
    rcclRmaRoundQueuesFree(&qs, 2);
    EXPECT_EQ(qs.send, nullptr);
    rcclRmaRoundQueuesFree(&qs, 2);
    rcclRmaRoundQueues none{};
    EXPECT_EQ(rcclRmaRoundQueueOf(&none, 1), nullptr);
}

TEST(RmaSegmentMathTest, SegIndexOfUsesHalfOpenOffsets)
{
    const size_t off[] = {0, 4096, 8192};
    EXPECT_EQ(rcclRmaSegIndexOf(off, 2, 0), 0);
    EXPECT_EQ(rcclRmaSegIndexOf(off, 2, 4095), 0);
    EXPECT_EQ(rcclRmaSegIndexOf(off, 2, 4096), 1);
    EXPECT_EQ(rcclRmaSegIndexOf(off, 2, 8192), 1);
    EXPECT_EQ(rcclRmaSegIndexOf(off, 2, 1 << 20), 1);
    EXPECT_EQ(rcclRmaSegIndexOf(off, 0, 0), -1);
    EXPECT_EQ(rcclRmaSegIndexOf(nullptr, 2, 0), -1);
}
