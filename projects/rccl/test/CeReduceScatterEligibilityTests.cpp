/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/CeReduceScatterTestHelpers.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

#include "ce_coll.h"
#include "collectives.h"
#include "gtest/gtest.h"
#include "nccl.h"
#include "rccl_common.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace RcclUnitTesting
{

// Production divides comm->ceColl.ceArStagingBytes by nRanks. The compile-time
// default is NCCL_CE_AR_STAGING_BYTES; RCCL_CE_AR_STAGING_BYTES replaces it.
// 100MiB is not that default and does not divide evenly by the awkward rank
// counts, so slot rounding is actually exercised.
constexpr size_t kReducedCeRsStagingBytes = 100ull * 1024 * 1024;

inline size_t ceRsPerRankCapacity(size_t stagingBytes, int nRanks)
{
    return stagingBytes / static_cast<size_t>(nRanks);
}

class CeReduceScatterEligibilityTest : public ::testing::Test
{
protected:
    CeReduceScatterMockComm mockComm_;
};

TEST_F(CeReduceScatterEligibilityTest, FuncToStringReturnsReduceScatter)
{
    EXPECT_STREQ(ncclFuncToString(ncclFuncReduceScatter), "ReduceScatter");
}

TEST_F(CeReduceScatterEligibilityTest, CeImplementedReturnsFalseForUnsupportedCollectives)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    EXPECT_FALSE(ncclCeImplemented(ncclFuncBroadcast, ncclDevSum, ncclFloat32));
    EXPECT_FALSE(ncclCeImplemented(ncclFuncReduce, ncclDevSum, ncclFloat32));
}

TEST_F(CeReduceScatterEligibilityTest, CeImplementedReturnsTrueForReduceScatterOnSupportedDriver)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range "
                        "(need ROCm >= 7.12 or 7.0.2.x backport [70051831, 70060000))";

    EXPECT_TRUE(ncclCeImplemented(ncclFuncReduceScatter, ncclDevSum, ncclFloat32));
}

TEST_F(CeReduceScatterEligibilityTest, CeAvailable_EligibleWithSymmetricSingleNode)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    EXPECT_TRUE(ncclCeAvailable(mockComm_.get(),
                                ncclFuncReduceScatter,
                                ncclDevSum,
                                ncclFloat32,
                                ncclSymSendRegRecvReg, nullptr, nullptr));
    EXPECT_TRUE(ncclCeAvailable(mockComm_.get(),
                                ncclFuncReduceScatter,
                                ncclDevSum,
                                ncclFloat32,
                                ncclSymSendNonregRecvReg, nullptr, nullptr));
}

TEST_F(CeReduceScatterEligibilityTest, CeAvailable_MultiNodeRejected)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    mockComm_.comm.nNodes = 2;
    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncReduceScatter,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendRegRecvReg, nullptr, nullptr));
}

TEST_F(CeReduceScatterEligibilityTest, CeAvailable_NoSymmetricSupportRejected)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    mockComm_.comm.symmetricSupport = false;
    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncReduceScatter,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendRegRecvReg, nullptr, nullptr));
}

TEST_F(CeReduceScatterEligibilityTest, CeAvailable_UnsupportedWindowRegistrationRejected)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncReduceScatter,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendNonregRecvNonreg, nullptr, nullptr));
    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncReduceScatter,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendRegRecvNonreg, nullptr, nullptr));
}

TEST_F(CeReduceScatterEligibilityTest, ChunkLayout_SmallMessageSingleChunk)
{
    constexpr int    nRanks    = 4;
    constexpr size_t recvcount = 1024;

    // ncclCeReduceScatter() only ever sees recvcounts the eligibility gate accepted:
    // nonzero, and a total message (recvcount * nRanks) no larger than the staging buffer.
    // Unlike AllReduce, recvcount is already the per-rank shard, so it does not have to
    // divide nRanks.
    ASSERT_GT(recvcount, 0u);
    ASSERT_LE(recvcount * sizeof(float) * static_cast<size_t>(nRanks),
              kCeRsMaxMsgBytesDefault);

    const size_t shardElems = recvcount;
    const size_t shardBytes = shardElems * sizeof(float);

    // A shard this small fits one slot at the default capacity and at a reduced
    // RCCL_CE_AR_STAGING_BYTES, so ncclCeReduceScatter() sends it as a single
    // chunk and never enters the pipelined path.
    EXPECT_EQ(shardElems, 1024u);
    for(size_t stagingBytes : {static_cast<size_t>(NCCL_CE_AR_STAGING_BYTES), kReducedCeRsStagingBytes})
    {
        const size_t slotChunkBytes =
            ncclCeAllReduceSlotChunkBytes(ceRsPerRankCapacity(stagingBytes, nRanks));
        EXPECT_LE(shardBytes, slotChunkBytes) << "stagingBytes=" << stagingBytes;
    }
}

// The host scatter addresses staging slots in bytes (rank * slotChunkBytes) while
// the reduce kernel addresses them in elements (rank * slotChunkElems). If those
// two strides disagree by even one byte, every rank but rank 0 reduces shifted
// data. CE ReduceScatter reuses the AllReduce staging layout, so the same
// alignment constraint applies. NCCL_CE_AR_STAGING_BYTES / nRanks only divides
// evenly for power-of-2 rank counts, so those were the only ones that used to work.
TEST_F(CeReduceScatterEligibilityTest, ChunkLayout_SlotStridesAgreeForAnyRankCount)
{
    const std::vector<int>    rankCounts   = {2, 3, 4, 5, 6, 7, 8, 12, 16, 24};
    const std::vector<size_t> elementSizes = {1, 2, 4, 8};

    for(size_t stagingBytes : {static_cast<size_t>(NCCL_CE_AR_STAGING_BYTES), kReducedCeRsStagingBytes})
    {
        for(int nRanks : rankCounts)
        {
            const size_t perRank        = ceRsPerRankCapacity(stagingBytes, nRanks);
            const size_t slotChunkBytes = ncclCeAllReduceSlotChunkBytes(perRank);
            SCOPED_TRACE("stagingBytes=" + std::to_string(stagingBytes) +
                         " nRanks=" + std::to_string(nRanks));

            // Rank boundaries stay aligned for the kernel's 16B vector loads, and the
            // slots stay inside the per-rank capacity ncclCeEnsureAllReduceStaging()
            // divides ceArStagingBytes into.
            EXPECT_EQ(slotChunkBytes % 16, 0u);
            EXPECT_LE(slotChunkBytes, perRank);

            for(size_t eltSize : elementSizes)
            {
                // A slot holds a whole number of elements, so the byte view and the
                // element view describe the same stride.
                EXPECT_EQ((slotChunkBytes / eltSize) * eltSize, slotChunkBytes)
                    << "eltSize=" << eltSize;
            }
        }
    }
}

TEST_F(CeReduceScatterEligibilityTest, ChunkLayout_LargeMessagePipelined)
{
    // A shard only spills past one slot when stagingBytes / nRanks is not
    // 16B-aligned, i.e. for a non-power-of-2 rank count at the per-rank cap.
    // For ReduceScatter the shard is recvcount itself. Both the compile-time
    // default and a reduced RCCL_CE_AR_STAGING_BYTES take that path at nRanks=6.
    constexpr int nRanks = 6;
    for(size_t stagingBytes : {static_cast<size_t>(NCCL_CE_AR_STAGING_BYTES), kReducedCeRsStagingBytes})
    {
        SCOPED_TRACE("stagingBytes=" + std::to_string(stagingBytes));
        const size_t perRank    = ceRsPerRankCapacity(stagingBytes, nRanks);
        const size_t shardElems = perRank / sizeof(float);
        const size_t recvcount  = shardElems;

        // The gate checks the full send buffer (recvcount * nRanks). The default
        // capacity matches the 2-shot window; a reduced capacity is smaller.
        if(stagingBytes == NCCL_CE_AR_STAGING_BYTES)
        {
            ASSERT_LE(recvcount * sizeof(float) * static_cast<size_t>(nRanks),
                      kCeRsMaxMsgBytesDefault);
        }

        const size_t shardBytes     = shardElems * sizeof(float);
        const size_t slotChunkBytes = ncclCeAllReduceSlotChunkBytes(perRank);
        ASSERT_GT(shardBytes, slotChunkBytes);

        // Same bookkeeping ncclCeReduceScatter() does once it has picked a chunk size.
        const size_t chunkBytes     = ncclCeAllReduceChooseChunkBytes(shardBytes, slotChunkBytes);
        const size_t baseChunkElems = chunkBytes / sizeof(float);
        const size_t tailChunkElems = shardElems % baseChunkElems;
        const size_t chunksPerShard = shardElems / baseChunkElems + (tailChunkElems != 0 ? 1 : 0);
        const size_t lastChunkElems = tailChunkElems != 0 ? tailChunkElems : baseChunkElems;

        ASSERT_GT(chunksPerShard, 1u);
        EXPECT_EQ(chunkBytes % 16, 0u);
        EXPECT_EQ(baseChunkElems * sizeof(float), chunkBytes);
        EXPECT_LE(chunkBytes, slotChunkBytes);

        // Chunks must cover the shard exactly: the host reads chunk ch at
        // ch * chunkBytes, so a chunk size that is not a whole number of elements
        // walks the last chunk past the end of the shard.
        EXPECT_EQ((chunksPerShard - 1) * baseChunkElems + lastChunkElems, shardElems);
        EXPECT_EQ((chunksPerShard - 1) * chunkBytes + lastChunkElems * sizeof(float), shardBytes);
    }
}

// ---------------------------------------------------------------------------
// Staging-offset arithmetic (ce_coll.h). Offsets are hand-computed, not re-derived, so an
// off-by-one shows up as a wrong number instead of two wrong numbers agreeing. Nothing here calls
// ncclCeReduceScatter(), so a call site passing a wrong index is out of reach.
// The capacity passed in is ceArStagingBytes / nRanks, the value
// ncclCeEnsureAllReduceStaging() and ncclCeReduceScatter() both divide by, so the
// bounds are that allocated extent rather than ncclCeAllReduceMaxChunkBytes().
// ---------------------------------------------------------------------------

TEST_F(CeReduceScatterEligibilityTest, DstSlotOffset_HandComputedForEveryRankAndSlot)
{
    // Region numbering written out by hand, so no right-hand side re-derives slot * nRanks + sender,
    // and scaled by slotChunkBytes so a staging retune cannot fail this for an unrelated reason.
    // The static_assert turns a NCCL_CE_NUM_SLOTS bump into a build error, not unchecked slots.
    static_assert(NCCL_CE_NUM_SLOTS == 2, "the slot tables below enumerate every slot by hand");
    constexpr int nRanks = 4;
    for(size_t stagingBytes : {static_cast<size_t>(NCCL_CE_AR_STAGING_BYTES), kReducedCeRsStagingBytes})
    {
        SCOPED_TRACE("stagingBytes=" + std::to_string(stagingBytes));
        const size_t perRank        = ceRsPerRankCapacity(stagingBytes, nRanks);
        const size_t slotChunkBytes = ncclCeAllReduceSlotChunkBytes(perRank);

        // nRanks=4 divides both capacities exactly, so the 16B round-down is a no-op and consecutive
        // regions really are one slotChunkBytes apart with no rounding gap.
        ASSERT_EQ(slotChunkBytes, perRank);

        // Slot 0: the four senders' regions, back to back from the top of the buffer.
        EXPECT_EQ(ncclCeReduceScatterDstSlotOffsetBytes(0, 0, nRanks, slotChunkBytes), 0 * slotChunkBytes);
        EXPECT_EQ(ncclCeReduceScatterDstSlotOffsetBytes(0, 1, nRanks, slotChunkBytes), 1 * slotChunkBytes);
        EXPECT_EQ(ncclCeReduceScatterDstSlotOffsetBytes(0, 2, nRanks, slotChunkBytes), 2 * slotChunkBytes);
        EXPECT_EQ(ncclCeReduceScatterDstSlotOffsetBytes(0, 3, nRanks, slotChunkBytes), 3 * slotChunkBytes);

        // Slot 1 starts one whole slot stride (nRanks * slotChunkBytes) later.
        EXPECT_EQ(ncclCeReduceScatterDstSlotOffsetBytes(1, 0, nRanks, slotChunkBytes), 4 * slotChunkBytes);
        EXPECT_EQ(ncclCeReduceScatterDstSlotOffsetBytes(1, 1, nRanks, slotChunkBytes), 5 * slotChunkBytes);
        EXPECT_EQ(ncclCeReduceScatterDstSlotOffsetBytes(1, 2, nRanks, slotChunkBytes), 6 * slotChunkBytes);
        EXPECT_EQ(ncclCeReduceScatterDstSlotOffsetBytes(1, 3, nRanks, slotChunkBytes), 7 * slotChunkBytes);

        // The last region ends exactly at nRanks * perRank * slots, the extent
        // ncclCeEnsureAllReduceStaging() allocates before its 16-byte align-up.
        EXPECT_EQ(ncclCeReduceScatterDstSlotOffsetBytes(
                      NCCL_CE_NUM_SLOTS - 1, nRanks - 1, nRanks, slotChunkBytes) +
                      slotChunkBytes,
                  static_cast<size_t>(NCCL_CE_NUM_SLOTS) * nRanks * perRank);
    }
}

TEST_F(CeReduceScatterEligibilityTest, DstSlotOffset_RegionsNeverOverlapAndStayInBounds)
{
    // Every (slot, sender) pair must own a private slotChunkBytes region; sharing one would have
    // the reduce kernel read one rank's data twice and another's never. Sorted and checked as a
    // real inequality over the computed offsets, not a restatement of the formula.
    for(size_t stagingBytes : {static_cast<size_t>(NCCL_CE_AR_STAGING_BYTES), kReducedCeRsStagingBytes})
    {
        for(int nRanks : {2, 3, 4, 5, 6, 7, 8, 12, 16})
        {
        SCOPED_TRACE("stagingBytes=" + std::to_string(stagingBytes) + " nRanks=" + std::to_string(nRanks));
        const size_t maxChunkBytes  = ceRsPerRankCapacity(stagingBytes, nRanks);
        const size_t slotChunkBytes = ncclCeAllReduceSlotChunkBytes(maxChunkBytes);
        const size_t bufferBytes    = static_cast<size_t>(NCCL_CE_NUM_SLOTS) * nRanks * maxChunkBytes;

        std::vector<size_t> starts;
        for(int slot = 0; slot < NCCL_CE_NUM_SLOTS; ++slot)
        {
            for(int sender = 0; sender < nRanks; ++sender)
            {
                starts.push_back(
                    ncclCeReduceScatterDstSlotOffsetBytes(slot, sender, nRanks, slotChunkBytes));
            }
        }

        ASSERT_EQ(starts.size(), static_cast<size_t>(NCCL_CE_NUM_SLOTS) * nRanks);
        std::sort(starts.begin(), starts.end());
        for(size_t i = 1; i < starts.size(); ++i)
        {
            EXPECT_GE(starts[i], starts[i - 1] + slotChunkBytes)
                << "regions " << (i - 1) << " and " << i << " overlap";
        }
        EXPECT_LE(starts.back() + slotChunkBytes, bufferBytes);
        }
    }
}

TEST_F(CeReduceScatterEligibilityTest, SrcOffset_HandComputedWholeShardAndPipelined)
{
    // Four 4096B shards (1024 floats, 4 ranks) walked four chunks at a time. Rank 2 / chunk 2 lands
    // at 10240 either way round, so the asymmetric cases (rank 1 / chunk 3, rank 3 / chunk 1) are
    // the ones that separate the rank term from the chunk term.
    constexpr int    nRanks     = 4;
    constexpr size_t shardBytes = 4096;
    constexpr size_t chunkBytes = 1024;

    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(0, shardBytes, 0, chunkBytes), 0u);
    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(0, shardBytes, 3, chunkBytes), 3072u);
    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(1, shardBytes, 0, chunkBytes), 4096u);
    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(1, shardBytes, 3, chunkBytes), 7168u);
    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(2, shardBytes, 0, chunkBytes), 8192u);
    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(2, shardBytes, 2, chunkBytes), 10240u);
    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(3, shardBytes, 0, chunkBytes), 12288u);
    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(3, shardBytes, 1, chunkBytes), 13312u);
    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(3, shardBytes, 3, chunkBytes), 15360u);
    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(3, shardBytes, 3, chunkBytes) + chunkBytes,
              static_cast<size_t>(nRanks) * shardBytes);

    // The other chunk size ncclCeReduceScatter() can pick: a shard that fits one slot travels
    // whole, so chunkBytes is the shard and the chunk index stays 0. The rank offsets must be
    // unchanged; any defect keyed on the two sizes being equal is invisible at 1024 vs 4096.
    constexpr size_t wholeChunkBytes = shardBytes;

    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(0, shardBytes, 0, wholeChunkBytes), 0u);
    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(1, shardBytes, 0, wholeChunkBytes), 4096u);
    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(2, shardBytes, 0, wholeChunkBytes), 8192u);
    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(3, shardBytes, 0, wholeChunkBytes), 12288u);
    EXPECT_EQ(ncclCeReduceScatterSrcOffsetBytes(nRanks - 1, shardBytes, 0, wholeChunkBytes) + wholeChunkBytes,
              static_cast<size_t>(nRanks) * shardBytes);
}

TEST_F(CeReduceScatterEligibilityTest, SrcOffset_TilesTheSendBufferExactly)
{
    // The (dstRank, chunk) copies must tile [0, nRanks * shardBytes) exactly: a gap leaves stale data
    // in the peer's slot, an overlap drops the bytes it skipped. 2500 elements over a 700-element
    // chunk gives every shard a short tail. Sizes are synthetic; the property holds at any.
    constexpr int    nRanks        = 3;
    constexpr size_t eltSize       = sizeof(float);
    constexpr size_t shardElems    = 2500;
    constexpr size_t shardBytes    = shardElems * eltSize;
    constexpr size_t chunkBytes    = 700 * eltSize;

    // Mirrors ncclCeReduceScatter()'s own chunk bookkeeping.
    const size_t baseChunkElems = chunkBytes / eltSize;
    const size_t tailChunkElems = shardElems % baseChunkElems;
    const size_t chunksPerShard = shardElems / baseChunkElems + (tailChunkElems != 0 ? 1 : 0);
    ASSERT_EQ(baseChunkElems, 700u);
    ASSERT_EQ(tailChunkElems, 400u); // 2500 = 3*700 + 400
    ASSERT_EQ(chunksPerShard, 4u);

    std::vector<std::pair<size_t, size_t>> spans; // [begin, end)
    for(int dstRank = 0; dstRank < nRanks; ++dstRank)
    {
        for(int chunk = 0; chunk < static_cast<int>(chunksPerShard); ++chunk)
        {
            const bool   isTail = (chunk == static_cast<int>(chunksPerShard) - 1) && (tailChunkElems > 0);
            const size_t bytes  = isTail ? tailChunkElems * eltSize : chunkBytes;
            const size_t begin  = ncclCeReduceScatterSrcOffsetBytes(dstRank, shardBytes, chunk, chunkBytes);
            spans.emplace_back(begin, begin + bytes);
        }
    }

    std::sort(spans.begin(), spans.end());
    ASSERT_EQ(spans.front().first, 0u);
    for(size_t i = 1; i < spans.size(); ++i)
    {
        EXPECT_EQ(spans[i].first, spans[i - 1].second)
            << "span " << i << " does not start where span " << (i - 1) << " ended";
    }
    EXPECT_EQ(spans.back().second, static_cast<size_t>(nRanks) * shardBytes);
}

TEST_F(CeReduceScatterEligibilityTest, SignalIndex_HandComputedBoundedAndUnique)
{
    // signalBuffer is [NCCL_CE_NUM_SLOTS][nRanks], read as an array index locally and as a byte
    // offset into signalWin for peers. Both views now come from this helper, so comparing them
    // would compare a value with itself; the indices and the two properties below are what is left.
    EXPECT_EQ(ncclCeReduceScatterSignalIndex(0, 0, 4), 0u);
    EXPECT_EQ(ncclCeReduceScatterSignalIndex(0, 3, 4), 3u);
    EXPECT_EQ(ncclCeReduceScatterSignalIndex(1, 0, 4), 4u);
    EXPECT_EQ(ncclCeReduceScatterSignalIndex(1, 3, 4), 7u);

    // Every (slot, rank) pair gets its own flag, and no index runs past the NUM_SLOTS * nRanks
    // words ncclCeInit() allocates. Swept over several rank counts because at nRanks 4 alone a
    // formula that hardcoded 4, or XORed the two terms, is still bounded and collision-free.
    for(int nRanks : {2, 3, 4, 5, 6, 7, 8, 12, 16})
    {
        SCOPED_TRACE("nRanks=" + std::to_string(nRanks));
        std::vector<size_t> indices;
        for(int slot = 0; slot < NCCL_CE_NUM_SLOTS; ++slot)
        {
            for(int rank = 0; rank < nRanks; ++rank)
            {
                const size_t index = ncclCeReduceScatterSignalIndex(slot, rank, nRanks);
                EXPECT_LT(index, static_cast<size_t>(NCCL_CE_NUM_SLOTS) * nRanks);
                indices.push_back(index);
            }
        }
        std::sort(indices.begin(), indices.end());
        EXPECT_EQ(std::adjacent_find(indices.begin(), indices.end()), indices.end())
            << "two (slot, rank) pairs share one doorbell";
    }
}

TEST_F(CeReduceScatterEligibilityTest, DstSlotOffset_ChunkNeverSpillsIntoTheNextSendersRegion)
{
    // Where the region formula and the chunk-size formula meet: a copy runs currentChunkBytes from
    // its region start and must end no later than the next region begins, or one sender overwrites
    // the next one's contribution. Each end is compared against the following start, the last
    // against the staging extent. Driven at the awkward rank counts where slotChunkBytes is a
    // truncated capacity, which split across the whole-shard and pipelined branches; which branch
    // falls out of the 16B rounding, so the flags pin that both were really taken.
    bool sawWholeShard = false;
    bool sawPipelined  = false;

    for(size_t stagingBytes : {static_cast<size_t>(NCCL_CE_AR_STAGING_BYTES), kReducedCeRsStagingBytes})
    {
        for(int nRanks : {3, 5, 6, 7, 12})
        {
        SCOPED_TRACE("stagingBytes=" + std::to_string(stagingBytes) + " nRanks=" + std::to_string(nRanks));
        const size_t eltSize        = sizeof(float);
        const size_t maxChunkBytes  = ceRsPerRankCapacity(stagingBytes, nRanks);
        const size_t slotChunkBytes = ncclCeAllReduceSlotChunkBytes(maxChunkBytes);
        const size_t shardElems     = kCeRsMaxMsgBytesDefault / (eltSize * static_cast<size_t>(nRanks));
        const size_t shardBytes     = shardElems * eltSize;
        const bool   wholeShard     = (shardBytes <= slotChunkBytes);
        const size_t chunkBytes =
            wholeShard ? shardBytes : ncclCeAllReduceChooseChunkBytes(shardBytes, slotChunkBytes);

        // ncclCeReduceScatter()'s bookkeeping: every copy is chunkBytes except the shard's last,
        // which carries any remainder. That remainder is always short, so the max can only pick
        // chunkBytes; it names both copy sizes rather than choosing between them.
        const size_t baseChunkElems = chunkBytes / eltSize;
        ASSERT_GT(baseChunkElems, 0u); // guards the % baseChunkElems division below
        const size_t tailChunkElems = shardElems % baseChunkElems;
        const size_t tailChunkBytes = (tailChunkElems != 0) ? tailChunkElems * eltSize : chunkBytes;
        const size_t maxCopyBytes   = std::max(chunkBytes, tailChunkBytes);

        sawWholeShard = sawWholeShard || wholeShard;
        sawPipelined  = sawPipelined || (!wholeShard && tailChunkElems != 0);

        const size_t bufferBytes = static_cast<size_t>(NCCL_CE_NUM_SLOTS) * nRanks * maxChunkBytes;
        for(int slot = 0; slot < NCCL_CE_NUM_SLOTS; ++slot)
        {
            for(int sender = 0; sender < nRanks; ++sender)
            {
                const bool   isLast     = (slot == NCCL_CE_NUM_SLOTS - 1) && (sender == nRanks - 1);
                const int    nextSlot   = (sender + 1 == nRanks) ? slot + 1 : slot;
                const int    nextSender = (sender + 1 == nRanks) ? 0 : sender + 1;
                const size_t start      = ncclCeReduceScatterDstSlotOffsetBytes(slot, sender, nRanks, slotChunkBytes);
                const size_t limit      =
                    isLast ? bufferBytes
                           : ncclCeReduceScatterDstSlotOffsetBytes(nextSlot, nextSender, nRanks, slotChunkBytes);
                EXPECT_LE(start + maxCopyBytes, limit)
                    << "slot=" << slot << " sender=" << sender << " overruns the next region";
            }
        }
        }
    }

    EXPECT_TRUE(sawWholeShard) << "no rank count reached the whole-shard branch";
    EXPECT_TRUE(sawPipelined) << "no rank count reached the pipelined short-tail branch";
}

TEST_F(CeReduceScatterEligibilityTest, MaxStagingBytesPerRank)
{
    // ncclCeEnsureAllReduceStaging() divides ceArStagingBytes by nRanks, then
    // rounds the slot down to 16 bytes. Integer division must not exceed the
    // capacity, at the default and at a reduced RCCL_CE_AR_STAGING_BYTES.
    for(size_t stagingBytes : {static_cast<size_t>(NCCL_CE_AR_STAGING_BYTES), kReducedCeRsStagingBytes})
    {
        for(int nRanks : {2, 3, 4, 5, 6, 7, 8, 12, 16, 24})
        {
            SCOPED_TRACE("stagingBytes=" + std::to_string(stagingBytes) +
                         " nRanks=" + std::to_string(nRanks));
            const size_t perRank = ceRsPerRankCapacity(stagingBytes, nRanks);
            EXPECT_LE(perRank * static_cast<size_t>(nRanks), stagingBytes);
            EXPECT_LE(ncclCeAllReduceSlotChunkBytes(perRank), perRank);
        }
    }
}

TEST(RcclCeReduceScatterEligibility, RcclUseCeReduceScatter_Isolated)
{
    struct UseCeRsCase
    {
        std::string                                  name;
        int                                          nRanks;
        int                                          nNodes;
        bool                                         symmetricSupport;
        int                                          ctaPolicy;
        size_t                                       recvcount;
        ncclRedOp_t                                  op;
        ncclDataType_t                               datatype;
        bool                                         expected;
        std::unordered_map<std::string, std::string> extraEnv;
    };

    const std::unordered_map<std::string, std::string> baseEnv = {
        {"RCCL_CE_REDUCESCATTER", "1"},
    };

    const std::vector<UseCeRsCase> cases = {
        {"DisabledByDefault_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 1024, ncclSum, ncclFloat32, false, {}},
        {"EligibleFloat32Sum_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 1024, ncclSum, ncclFloat32, true, baseEnv},
        {"MultiNodeRejected_Isolated", 4, 2, true, NCCL_CTA_POLICY_ZERO, 1024, ncclSum, ncclFloat32, false, baseEnv},
        {"NoSymmetricSupportRejected_Isolated", 4, 1, false, NCCL_CTA_POLICY_ZERO, 1024, ncclSum, ncclFloat32, false, baseEnv},
        {"WrongCtaPolicyRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_DEFAULT, 1024, ncclSum, ncclFloat32, false, baseEnv},
        // recvcount is already the per-rank shard, so it does not have to divide nRanks.
        {"RecvcountNotDivisibleByRanksStillEligible_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 4097, ncclSum, ncclFloat32, true, baseEnv},
        // Non-power-of-2 rank counts are eligible too, and are the ones whose
        // staging layout the chunk-layout tests above cover.
        {"EligibleSixRanks_Isolated", 6, 1, true, NCCL_CTA_POLICY_ZERO, 683, ncclSum, ncclFloat32, true, baseEnv},
        // 1001 floats is 4004 bytes. Chunking rounds the pipeline chunk to 16 bytes
        // and the kernel scalar-reduces the tail, so the selector must still accept it.
        {"ShardBytesNotMultipleOf16StillEligible_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 1001, ncclSum, ncclFloat32, true, baseEnv},
        {"ZeroCountRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 0, ncclSum, ncclFloat32, false, baseEnv},
        {"UnsupportedOpRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 1024, ncclAvg, ncclFloat32, false, baseEnv},
        {"Float8Rejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 1024, ncclSum, ncclFloat8e4m3, false, baseEnv},
        // msgBytes is recvcount * sizeof(datatype) * nRanks, not recvcount alone.
        {"MessageTooLargeRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO,
         (kCeRsMaxMsgBytesDefault / (sizeof(float) * 4)) + 1, ncclSum, ncclFloat32, false, baseEnv},
        {"MessageAtCapAccepted_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO,
         kCeRsMaxMsgBytesDefault / (sizeof(float) * 4), ncclSum, ncclFloat32, true, baseEnv},
    };

    for(const auto& tc : cases)
    {
        auto env = tc.extraEnv;
        ProcessIsolatedTestRunner::registerTest(
            ProcessIsolatedTestRunner::TestConfig(
                tc.name,
                [tc]()
                {
                    CeReduceScatterMockComm mock;
                    mock.comm.nRanks           = tc.nRanks;
                    mock.comm.nNodes           = tc.nNodes;
                    mock.comm.symmetricSupport = tc.symmetricSupport;
                    mock.comm.config.CTAPolicy = tc.ctaPolicy;

                    const bool result =
                        rcclUseCeReduceScatter(mock.get(), tc.recvcount, tc.datatype, tc.op);
                    EXPECT_EQ(result, tc.expected) << tc.name;
                })
                .withEnvironment(env)
                .withTimeout(std::chrono::seconds(30))
                .withNumGpus(0));
    }

    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;
    EXPECT_TRUE(ProcessIsolatedTestRunner::executeAllTests(options));
}

} // namespace RcclUnitTesting
