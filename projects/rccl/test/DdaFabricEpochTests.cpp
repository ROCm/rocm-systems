/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-side tests for the shared LL epoch counter (comm->ddaLLEpochDev) used by
// the DDA fabric collectives (AllGather / AllReduce / ReduceScatter / AllToAll,
// LL and LL128). The counter is sized by ddaLLEpochCount() and restamped every
// launch; each block derives its LL flag from its cell (bank = flag & 1) and the
// collectives rely on every cell advancing in lock-step so peer-swapped block
// pairs derive the same flag+bank. Source of truth for the modelled kernels:
//   src/include/algorithms/all_gather/all_gather_dda_fabric_ll.h       (2-D flag+write-back)
//   src/include/algorithms/alltoall/alltoall_dda_fabric_ll.h           (same 2-D shape)
//   src/include/algorithms/all_reduce/all_reduce_dda_ll.h              (1-D; note: the AR-LL
//       kernel lives here, there is NO all_reduce_dda_fabric_ll.h)
//   src/include/algorithms/reduce_scatter/reduce_scatter_dda_fabric_ll.h (1-D)
//   src/include/dda_init_detail.h                                      (ddaLLEpochCount)
//
// Faithfulness notes for the host replay below:
//  * A launch is modelled as read-all-then-write-all. That is a valid
//    serialization despite the absence of any grid-wide sync because block b
//    reads only cell b and writes only cells e == b (mod total); those residue
//    classes are disjoint, so no block writes a cell another block reads.
//  * Cross-launch lock-step (bank alternation) holds only because DDA launches on
//    a comm are stream-serialized, so launch N fully restamps before launch N+1.
//
// Scope note: these tests validate the HOST MODEL of the epoch arithmetic (using
// the real dda_init_detail.h constants) and the producer/consumer scratch
// contract. The negative controls fail if the *model* loses a property, proving
// the positive assertions are not vacuous; they do NOT read the device kernels,
// so they cannot detect a device-side regression (e.g. the write-back loop in
// all_gather_dda_fabric_ll.h being changed from `e < epochLen` to `e < total`).
// This file is in rccl-UnitTestsFixtures (Release+Debug): it uses only
// inline/constexpr helpers and the DDA_FABRIC_BUFFER_SIZE macro, no librccl
// symbols, so the producer-coverage assertion runs in Release CI too.

#include "common/DdaFabricFootprints.hpp" // shared footprint mirrors
#include "dda_init_detail.h" // ddaLLEpochCount, kDdaFabricLLArMaxBlocks, kDdaLLAgMaxBlocksPerPeer, DDA_FABRIC_MAXBLOCKS, DDA_FABRIC_BUFFER_SIZE
#include "fabric_gpu_barrier.h" // meta::comms::kDdaMaxNranks

// Real launcher constants, to pin every DdaFabricFootprints.hpp mirror at compile
// time. These test sources compile as host TUs with __HIP_PLATFORM_AMD__ defined
// and link hip::device, so including the device-code-bearing kernel headers is
// fine (nothing instantiates the kernels); device/TestOp128.cpp is the precedent.
// The static_asserts below build in the Release-capable rccl-UnitTestsFixtures
// target -- a production stride/cap change fails this build. LLPacket16 comes from
// CollCommon.h (pulled via dda_init_detail.h).
#include "algorithms/CollCommon_ll128.h"                            // meta::comms::{LLLine128,kDdaLL128DataElems}
#include "algorithms/all_reduce/all_reduce_dda_ll.h"                // kDdaLLArMaxBytes, kDdaLLArSlotStridePkts
#include "algorithms/all_gather/all_gather_dda_fabric_ll.h"         // kDdaLLAgMaxPerRankBytes
#include "algorithms/alltoall/alltoall_dda_fabric_ll.h"            // kDdaLLA2AMaxPerChunkBytes
#include "algorithms/reduce_scatter/reduce_scatter_dda_fabric_ll.h" // kDdaLLRsMaxBytes
#include "algorithms/all_gather/all_gather_dda_fabric_ll128.h"      // kDdaLL128AgMaxPerRankBytes, kDdaLL128AgSlotStrideLines
#include "algorithms/alltoall/alltoall_dda_fabric_ll128.h"          // kDdaLL128A2AMaxPerChunkBytes
#include "algorithms/reduce_scatter/reduce_scatter_dda_fabric_ll128.h" // kDdaLL128RsMaxBytes

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace RcclUnitTesting
{

// Compile-time tie between every footprint mirror and the real launcher
// constants. All independently-declared per-tier caps are asserted (not just one
// per family), plus the two derived slot strides -- a correct cap with a
// redefined stride would still yield a wrong footprint.
static_assert(kLLPacketBytes == sizeof(meta::comms::LLPacket16), "LLPacket16 size mirror drift");
static_assert(kLL128LineBytes == sizeof(meta::comms::LLLine128), "LLLine128 size mirror drift");
static_assert(kLL128DataElems == static_cast<size_t>(meta::comms::kDdaLL128DataElems), "LL128 data-elems mirror drift");
static_assert(kLLTierMaxBytes == meta::comms::kDdaLLArMaxBytes, "LL AR cap mirror drift");
static_assert(kLLTierMaxBytes == meta::comms::kDdaLLAgMaxPerRankBytes, "LL AG cap mirror drift");
static_assert(kLLTierMaxBytes == meta::comms::kDdaLLA2AMaxPerChunkBytes, "LL A2A cap mirror drift");
static_assert(kLLTierMaxBytes == meta::comms::kDdaLLRsMaxBytes, "LL RS cap mirror drift");
static_assert(kLL128TierMaxBytes == meta::comms::kDdaLL128AgMaxPerRankBytes, "LL128 AG cap mirror drift");
static_assert(kLL128TierMaxBytes == meta::comms::kDdaLL128A2AMaxPerChunkBytes, "LL128 A2A cap mirror drift");
static_assert(kLL128TierMaxBytes == meta::comms::kDdaLL128RsMaxBytes, "LL128 RS cap mirror drift");
static_assert(kLLTierMaxBytes / 8 == meta::comms::kDdaLLArSlotStridePkts, "LL slot-stride mirror drift");
static_assert(ddaLL128LinesForBytes(kLL128TierMaxBytes) == meta::comms::kDdaLL128AgSlotStrideLines,
              "LL128 slot-stride mirror drift");

namespace
{

using nccl_dda_detail::ddaLLEpochCount;
using nccl_dda_detail::kDdaFabricLLArMaxBlocks;
using nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer;

// Sentinel meaning "restamp the whole array" (the production epochLen).
constexpr int kFullWriteBack = -1;

// Number of shared epoch cells, from the real sizing function.
size_t epochCells(int nRanks, int arMaxBlocks = DDA_FABRIC_MAXBLOCKS)
{
    return ddaLLEpochCount(nRanks, arMaxBlocks);
}

// Per-block LL flag from its epoch cell:  f = cell + 1; if (f == 0) f = 2.
uint32_t deriveFlag(uint32_t cell)
{
    uint32_t f = cell + 1u;
    if (f == 0u)
    {
        f = 2u; // skip 0 sentinel; keep bank parity
    }
    return f;
}

unsigned bankOf(uint32_t flag)
{
    return flag & 1u;
}

// Replay one launch's epoch update: `total` launched blocks, restamping cells
// [0, writeBackLen) with block b owning cells b, b+total, b+2*total, ... exactly
// as the kernels do: `for (e = flatBlockId; e < epochLen; e += total) epoch[e]=flag`.
// writeBackLen == kFullWriteBack models the production full-length restamp.
// The ASSERT_ bounds guards are preconditions (call sites are constructed to
// satisfy them); wrap calls in ASSERT_NO_FATAL_FAILURE so a guard trip stops the
// caller instead of cascading against an unmodified array.
void applyLaunch(std::vector<uint32_t>& epoch, int total, int writeBackLen = kFullWriteBack)
{
    const int cells    = static_cast<int>(epoch.size());
    const int epochLen = (writeBackLen == kFullWriteBack) ? cells : writeBackLen;
    ASSERT_GT(total, 0);
    ASSERT_GT(epochLen, 0);     // a zero write-back would make allCellsUniform vacuous
    ASSERT_LE(total, cells);    // read index flatBlockId < total stays in bounds
    ASSERT_LE(epochLen, cells); // write index stays in bounds
    std::vector<uint32_t> flag(total);
    for (int b = 0; b < total; ++b)
    {
        flag[b] = deriveFlag(epoch[b]);
    }
    for (int b = 0; b < total; ++b)
    {
        for (int e = b; e < epochLen; e += total)
        {
            epoch[e] = flag[b];
        }
    }
}

// AllGather / AllToAll: 2-D grid nRanks x bpp, total = nRanks*bpp.
void applyAllGather(std::vector<uint32_t>& epoch, int nRanks, int bpp, int writeBackLen = kFullWriteBack)
{
    applyLaunch(epoch, nRanks * bpp, writeBackLen);
}

// AllToAll shares AllGather's 2-D grid shape; alias for readability at call sites.
void applyAllToAll(std::vector<uint32_t>& epoch, int nRanks, int bpp)
{
    applyLaunch(epoch, nRanks * bpp);
}

// AllReduce: 1-D grid capped at kDdaFabricLLArMaxBlocks (the AR-only cap,
// dda_all_reduce_fabric_ll.cu).
void applyAllReduce(std::vector<uint32_t>& epoch, int blocks)
{
    applyLaunch(epoch, blocks);
}

// ReduceScatter: 1-D grid. Its launcher uses nBlocksMax = comm->ddaFabricMaxBlocks
// (up to DDA_FABRIC_MAXBLOCKS = 256), then clamps to ddaLLEpochLen -- NOT the
// AR-only kDdaFabricLLArMaxBlocks. `blocks` need not divide the cell count.
void applyReduceScatter(std::vector<uint32_t>& epoch, int blocks)
{
    applyLaunch(epoch, blocks);
}

// AllGather/AllToAll peer-swap pairing violations for a launch with `bpp`: for
// each (R,P,c) with P!=R the reader cell (P*bpp+c) and the paired writer cell
// (R*bpp+c) must derive the same flag. Returns the mismatch count (-1 on OOB).
long agPairingMismatches(const std::vector<uint32_t>& epoch, int nRanks, int bpp)
{
    if (nRanks * bpp > static_cast<int>(epoch.size()))
    {
        ADD_FAILURE() << "pairing check indexes past the epoch array: nRanks*bpp=" << nRanks * bpp
                      << " cells=" << epoch.size();
        return -1;
    }
    long mismatches = 0;
    for (int R = 0; R < nRanks; ++R)
    {
        for (int P = 0; P < nRanks; ++P)
        {
            if (P == R)
            {
                continue; // self column is a local copy, no cross-rank flag
            }
            for (int c = 0; c < bpp; ++c)
            {
                if (deriveFlag(epoch[P * bpp + c]) != deriveFlag(epoch[R * bpp + c]))
                {
                    ++mismatches;
                }
            }
        }
    }
    return mismatches;
}

bool allCellsUniform(const std::vector<uint32_t>& epoch)
{
    for (size_t i = 1; i < epoch.size(); ++i)
    {
        if (epoch[i] != epoch[0])
        {
            return false;
        }
    }
    return true;
}

constexpr int kSmallBpp = 1;
constexpr int kLargeBpp = kDdaLLAgMaxBlocksPerPeer; // widest per-peer fan-out

} // namespace

// ===========================================================================
// Static contracts (no rank fixture)
// ===========================================================================

// The array must be sized to cover the widest collective across the full rank
// range (up to kDdaMaxNranks) and every RCCL_DDA_FABRIC_MAXBLOCKS value [1,256].
// Uses nRanks past the nRanks*8 vs 256 crossover (>32) so both arms of the max
// are exercised.
TEST(DdaFabricEpochStaticTest, EpochCount_CoversWidestCollective)
{
    for (int arMaxBlocks : {1, kDdaFabricLLArMaxBlocks, DDA_FABRIC_MAXBLOCKS})
    {
        for (int nRanks : {2, 8, 32, 33, meta::comms::kDdaMaxNranks})
        {
            const long cells = static_cast<long>(ddaLLEpochCount(nRanks, arMaxBlocks));
            ASSERT_EQ(cells, std::max(nRanks * kDdaLLAgMaxBlocksPerPeer, arMaxBlocks))
                << "nRanks=" << nRanks << " arMaxBlocks=" << arMaxBlocks;
            // Must index the widest AllGather/AllToAll grid (nRanks*8), the AR-LL
            // grid (capped at kDdaFabricLLArMaxBlocks), and the RS-LL / LL128 grids
            // which launch up to comm->ddaFabricMaxBlocks (== arMaxBlocks here).
            ASSERT_GE(cells, static_cast<long>(nRanks) * kDdaLLAgMaxBlocksPerPeer)
                << "nRanks=" << nRanks << " arMaxBlocks=" << arMaxBlocks;
            ASSERT_GE(cells, std::min(arMaxBlocks, kDdaFabricLLArMaxBlocks))
                << "nRanks=" << nRanks << " arMaxBlocks=" << arMaxBlocks;
            ASSERT_GE(cells, arMaxBlocks) // RS-LL / AR-LL128 / RS-LL128 launch up to ddaFabricMaxBlocks
                << "nRanks=" << nRanks << " arMaxBlocks=" << arMaxBlocks;
        }
    }
}

// The AllGather-LL launcher has no runtime clamp; it relies on
// ddaLLAgBlocksPerPeer capping per-peer blocks at kDdaLLAgMaxBlocksPerPeer. (The
// LL128 AG/A2A launchers instead clamp blocksPerPeer against epochLen at runtime,
// since RCCL_DDA_LL128_AG_MAXBPP / _A2A_MAXBPP can raise their cap.) That compile-time cap must
// keep the AG-LL grid within the epoch array for every rank count and MAXBLOCKS.
TEST(DdaFabricEpochStaticTest, AllGatherGridAlwaysFitsEpochArray)
{
    for (int arMaxBlocks : {1, kDdaFabricLLArMaxBlocks, DDA_FABRIC_MAXBLOCKS})
    {
        for (int nRanks = 2; nRanks <= meta::comms::kDdaMaxNranks; ++nRanks)
        {
            const long widestGrid = static_cast<long>(nRanks) * kDdaLLAgMaxBlocksPerPeer;
            ASSERT_LE(widestGrid, static_cast<long>(ddaLLEpochCount(nRanks, arMaxBlocks)))
                << "nRanks=" << nRanks << " arMaxBlocks=" << arMaxBlocks;
        }
    }
}

// Producer side of the scratch contract: the shipped fabric scratch buffer must
// cover every fixed-footprint tier at the maximum rank count. This is the
// consumer-side gate's counterpart (the eligibility predicates, exercised in
// DdaFabricScratchTests.cpp) and runs in Release CI. If DDA_FABRIC_BUFFER_SIZE is
// lowered (or a tier's fixed footprint raised) below this, LL/LL128 fall back at
// runtime with no error -- this assertion catches that.
TEST(DdaFabricEpochStaticTest, ScratchBufferCoversFixedTiersAtMaxRanks)
{
    const int n = meta::comms::kDdaMaxNranks;
    EXPECT_GE(DDA_FABRIC_BUFFER_SIZE, ddaLLFixedFootprint(n)) << "scratch < LL footprint at " << n << " ranks";
    EXPECT_GE(DDA_FABRIC_BUFFER_SIZE, ddaLL128FixedFootprint(n))
        << "scratch < LL128 AG/A2A/RS footprint at " << n << " ranks";
    // (AR-LL128 is message-dependent and intentionally NOT covered to its 1 GiB
    //  cap at high rank counts; see DdaFabricScratchTests EffectiveCapShrinksWithRanks.)
}

// Flag derivation must skip the 0 sentinel and preserve bank parity, including
// at the uint32 wraparound the kernel comment guards against.
TEST(DdaFabricEpochStaticTest, DeriveFlag_SkipsZeroKeepsBankParity)
{
    for (uint32_t x : {0u, 1u, 2u, 3u, 100u, UINT32_MAX - 1u, UINT32_MAX})
    {
        EXPECT_NE(deriveFlag(x), 0u) << "x=" << x;
        EXPECT_EQ(bankOf(deriveFlag(x)), (x + 1u) & 1u) << "x=" << x; // parity survives wrap
    }
}

// ===========================================================================
// Rank-parameterized behavior
// ===========================================================================

class DdaFabricEpochTest : public ::testing::TestWithParam<int>
{
};

// Name the instances 2Ranks / 4Ranks / 8Ranks / 72Ranks instead of /0../3.
INSTANTIATE_TEST_SUITE_P(Ranks, DdaFabricEpochTest, ::testing::Values(2, 4, 8, meta::comms::kDdaMaxNranks),
                         [](const ::testing::TestParamInfo<int>& info)
                         {
                             return std::to_string(info.param) + "Ranks";
                         });

// A small AllGather then a large one: array stays uniform, the wider AllGather's
// peer-swap pairing is consistent, and consecutive launches alternate banks.
TEST_P(DdaFabricEpochTest, AllGatherSmallThenLarge_StaysConsistent)
{
    const int nRanks = GetParam();
    std::vector<uint32_t> epoch(epochCells(nRanks), 0u);

    const uint32_t f1 = deriveFlag(epoch[0]);
    ASSERT_NO_FATAL_FAILURE(applyAllGather(epoch, nRanks, kSmallBpp));
    ASSERT_TRUE(allCellsUniform(epoch));
    ASSERT_EQ(agPairingMismatches(epoch, nRanks, kLargeBpp), 0);

    const uint32_t f2 = deriveFlag(epoch[0]);
    ASSERT_NE(bankOf(f2), bankOf(f1)) << "consecutive launches must alternate banks";
    ASSERT_NO_FATAL_FAILURE(applyAllGather(epoch, nRanks, kLargeBpp));
    ASSERT_TRUE(allCellsUniform(epoch));
    ASSERT_EQ(agPairingMismatches(epoch, nRanks, kLargeBpp), 0);
}

// AR then AG: after an AllReduce, the shared array is still pairing-consistent for
// a following AllGather (no AllGather is applied here -- the check is the pairing
// a subsequent AG would derive).
TEST_P(DdaFabricEpochTest, AllReduce_LeavesArrayAllGatherPairable)
{
    const int nRanks = GetParam();
    std::vector<uint32_t> epoch(epochCells(nRanks), 0u);
    ASSERT_NO_FATAL_FAILURE(applyAllReduce(epoch, kDdaFabricLLArMaxBlocks)); // widest AR grid
    ASSERT_TRUE(allCellsUniform(epoch));
    ASSERT_EQ(agPairingMismatches(epoch, nRanks, kLargeBpp), 0);
}

// RS then AG. ReduceScatter's real cap is DDA_FABRIC_MAXBLOCKS (clamped to the
// cell count), not the AR-only cap; also exercise a block count that does not
// divide the cell count (the write-back stride must still tile every cell).
TEST_P(DdaFabricEpochTest, ReduceScatter_LeavesArrayAllGatherPairable)
{
    const int nRanks = GetParam();
    const int cells  = static_cast<int>(epochCells(nRanks));

    std::vector<uint32_t> epoch(cells, 0u);
    ASSERT_NO_FATAL_FAILURE(applyReduceScatter(epoch, std::min(DDA_FABRIC_MAXBLOCKS, cells))); // widest real RS grid
    ASSERT_TRUE(allCellsUniform(epoch));
    ASSERT_EQ(agPairingMismatches(epoch, nRanks, kLargeBpp), 0);

    const int nonDividing = 100; // 100 divides neither 256 nor 576; <= 256 cap and <= cells
    ASSERT_LE(nonDividing, cells);
    ASSERT_NO_FATAL_FAILURE(applyReduceScatter(epoch, nonDividing));
    ASSERT_TRUE(allCellsUniform(epoch)) << "non-dividing block count must still restamp every cell";
    ASSERT_EQ(agPairingMismatches(epoch, nRanks, kLargeBpp), 0);
}

TEST_P(DdaFabricEpochTest, AllToAll_LeavesArrayAllGatherPairable)
{
    const int nRanks = GetParam();
    std::vector<uint32_t> epoch(epochCells(nRanks), 0u);
    ASSERT_NO_FATAL_FAILURE(applyAllToAll(epoch, nRanks, kLargeBpp)); // A2A is peer-swapped like AG
    ASSERT_TRUE(allCellsUniform(epoch));
    ASSERT_EQ(agPairingMismatches(epoch, nRanks, kLargeBpp), 0);
}

// Interleaved mixed-size launches keep the array uniform and every AllGather /
// AllToAll pairing consistent each step. ASSERT_ stops at the first divergence.
TEST_P(DdaFabricEpochTest, RepeatedMixedSizes_StaysConsistent)
{
    const int nRanks = GetParam();
    const int agBpp[] = {kSmallBpp, 2, 4, kLargeBpp, 3};
    const int arBlk[] = {1, 4, kDdaFabricLLArMaxBlocks, 7};
    std::vector<uint32_t> epoch(epochCells(nRanks), 0u);

    // Seed for the first bank-alternation check: bank 0 vs the first launch's
    // flag deriveFlag(0)=1 (bank 1), so the first iteration's ASSERT_NE holds.
    uint32_t prevFlag = 0u;
    for (int iter = 0; iter < 4; ++iter)
    {
        for (int bpp : agBpp)
        {
            ASSERT_EQ(agPairingMismatches(epoch, nRanks, bpp), 0) << "iter=" << iter << " bpp=" << bpp;
            const uint32_t f = deriveFlag(epoch[0]);
            ASSERT_NE(bankOf(f), bankOf(prevFlag)) << "banks must alternate, iter=" << iter << " bpp=" << bpp;
            prevFlag = f;
            ASSERT_NO_FATAL_FAILURE(applyAllGather(epoch, nRanks, bpp));
            ASSERT_TRUE(allCellsUniform(epoch)) << "after AG iter=" << iter << " bpp=" << bpp;
        }
        for (int blk : arBlk)
        {
            const uint32_t f = deriveFlag(epoch[0]);
            ASSERT_NE(bankOf(f), bankOf(prevFlag)) << "banks must alternate, iter=" << iter << " blk=" << blk;
            prevFlag = f;
            ASSERT_NO_FATAL_FAILURE(applyAllReduce(epoch, blk));
            ASSERT_TRUE(allCellsUniform(epoch)) << "after AR iter=" << iter << " blk=" << blk;
        }
    }
}

// ===========================================================================
// Negative controls -- these MUST fail if the modelled behavior loses the
// property; they prove the positive tests are not vacuous. They validate the
// HOST MODEL, not the device kernels.
// ===========================================================================

// If the full-length restamp is weakened to own-cell-only (modelling the
// `e < epochLen` -> `e < total` regression in the write-back loop), a small AG
// followed by a wide AG leaves stale high cells and the pairing breaks. The
// full-length restamp on the same sequence stays consistent.
TEST_P(DdaFabricEpochTest, NegativeControl_ShortWriteBackBreaksPairing)
{
    const int nRanks = GetParam();

    std::vector<uint32_t> full(epochCells(nRanks), 0u);
    ASSERT_NO_FATAL_FAILURE(applyAllGather(full, nRanks, kSmallBpp, kFullWriteBack));
    ASSERT_EQ(agPairingMismatches(full, nRanks, kLargeBpp), 0) << "full restamp must stay consistent";

    std::vector<uint32_t> shortWb(epochCells(nRanks), 0u);
    ASSERT_NO_FATAL_FAILURE(applyAllGather(shortWb, nRanks, kSmallBpp, /*writeBackLen=*/nRanks * kSmallBpp));
    EXPECT_GT(agPairingMismatches(shortWb, nRanks, kLargeBpp), 0)
        << "own-cell-only restamp must break pairing -> proves the full-length restamp is load-bearing";
}

// The kernels early-return for blocks with peer >= nRanks (grid.x > nRanks)
// BEFORE the write-back, so those residue classes are never restamped and the
// array goes non-uniform. Launch sites use grid.x == nRanks; this pins that
// assumption.
TEST_P(DdaFabricEpochTest, NegativeControl_PeerBeyondNRanksBreaksUniformity)
{
    const int nRanks = GetParam();
    const int bpp    = 2;

    std::vector<uint32_t> ok(epochCells(nRanks), 0u);
    ASSERT_NO_FATAL_FAILURE(applyAllGather(ok, nRanks, bpp));
    ASSERT_TRUE(allCellsUniform(ok)) << "grid.x == nRanks must stay uniform";

    // Model grid.x == nRanks+1: all blocks read, but peer==nRanks blocks return
    // before restamping their residue class.
    std::vector<uint32_t> bad(epochCells(nRanks), 0u);
    const int gridX = nRanks + 1;
    const int total = gridX * bpp;
    ASSERT_LE(total, static_cast<int>(bad.size()));
    std::vector<uint32_t> flag(total);
    for (int b = 0; b < total; ++b)
    {
        flag[b] = deriveFlag(bad[b]);
    }
    for (int b = 0; b < total; ++b)
    {
        if (b / bpp < nRanks) { // peer = flatBlockId / bpp; peer >= nRanks returns early
            for (int e = b; e < static_cast<int>(bad.size()); e += total)
            {
                bad[e] = flag[b];
            }
        }
    }
    EXPECT_FALSE(allCellsUniform(bad)) << "grid.x > nRanks leaves residue classes unstamped -> non-uniform";
}

} // namespace RcclUnitTesting
