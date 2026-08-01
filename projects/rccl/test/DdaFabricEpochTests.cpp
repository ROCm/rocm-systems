/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-side consistency tests for the shared LL epoch counter used by the DDA
// fabric collectives (AllGather / AllReduce / ReduceScatter / AllToAll).
//
// comm->ddaLLEpochDev is a single device array of comm->ddaLLEpochLen uint32
// cells, shared across those collectives and sized by ddaLLEpochCount() to cover
// the widest block grid any of them launches. On each launch tid 0 of block b
// reads its epoch cell and derives this launch's LL flag:
//     f = cell + 1;  if (f == 0) f = 2;   // skip the 0 sentinel
//     bank = f & 1;                        // double-buffer selector
// and at the end re-stamps the array over its full length:
//     for (e = flatBlockId; e < ddaLLEpochLen; e += total) epoch[e] = flag;
//
// The collectives rely on every cell of the shared array advancing in lock-step
// so peer-swapped block pairs derive the same flag and bank. For AllGather the
// reader (rank R, block peer=P, chunk c) uses epoch[P*bpp+c] while the matching
// writer (rank P, block peer=R, chunk c) uses epoch[R*bpp+c]; since every rank
// evolves the array identically (SPMD), the poll matches only when
// epoch[P*bpp+c] == epoch[R*bpp+c] for all R,P,c. AllReduce/ReduceScatter are
// 1-D and same-index paired (reader block b pairs with writer block b).
//
// These tests replay the epoch arithmetic on the host, using the real constants
// from dda_init_detail.h, and verify the array stays uniform and the AllGather
// pairing stays consistent across single and mixed-size collective sequences.
// They need no GPU. They mirror:
//   src/include/algorithms/all_gather/all_gather_dda_fabric_ll.h
//   src/include/algorithms/all_reduce/all_reduce_dda_ll.h
//   src/dda_all_gather_fabric_ll.cu, src/dda_all_reduce_fabric_ll.cu

#include "dda_init_detail.h" // nccl_dda_detail::{ddaLLEpochCount,kDdaFabricLLArMaxBlocks,kDdaLLAgMaxBlocksPerPeer}, DDA_FABRIC_MAXBLOCKS

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace RcclUnitTesting
{
namespace
{

using nccl_dda_detail::ddaLLEpochCount;
using nccl_dda_detail::kDdaFabricLLArMaxBlocks;
using nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer;

// Number of shared epoch cells, from the real sizing function.
int allocCells(int nRanks) { return static_cast<int>(ddaLLEpochCount(nRanks, DDA_FABRIC_MAXBLOCKS)); }

// Per-block LL flag from its epoch cell:  f = cell + 1; if (f == 0) f = 2.
uint32_t deriveFlag(uint32_t cell)
{
    uint32_t f = cell + 1u;
    if (f == 0u) f = 2u;
    return f;
}

// Replay one launch's epoch update: `total` blocks, full-length write-back over
// [0, ddaLLEpochLen), block b owning cells b, b+total, b+2*total, ... exactly as
// the kernels do:  for (e = flatBlockId; e < epochLen; e += total) epoch[e]=flag.
void applyLaunch(std::vector<uint32_t>& epoch, int total)
{
    const int epochLen = static_cast<int>(epoch.size()); // full ddaLLEpochLen
    std::vector<uint32_t> flag(total);
    for (int b = 0; b < total; ++b)
        flag[b] = deriveFlag(epoch[b]);
    for (int b = 0; b < total; ++b)
        for (int e = b; e < epochLen; e += total)
            epoch[e] = flag[b];
}

// AllGather launch: grid = nRanks x blocksPerPeer, total = nRanks*bpp.
void applyAllGather(std::vector<uint32_t>& epoch, int nRanks, int bpp) { applyLaunch(epoch, nRanks * bpp); }

// AllReduce launch: 1-D grid of arBlocks (<= kDdaFabricLLArMaxBlocks) blocks.
void applyAllReduce(std::vector<uint32_t>& epoch, int arBlocks) { applyLaunch(epoch, arBlocks); }

// AllGather peer-swap pairing violations an AG launch with `bpp` would see when
// reading `epoch`: for each (R,P,c) with P!=R the reader cell (P*bpp+c) and the
// paired writer cell (R*bpp+c) must derive the same flag.
long agPairingMismatches(const std::vector<uint32_t>& epoch, int nRanks, int bpp)
{
    long mismatches = 0;
    for (int R = 0; R < nRanks; ++R)
        for (int P = 0; P < nRanks; ++P)
        {
            if (P == R) continue; // self column is a local copy, no cross-rank flag
            for (int c = 0; c < bpp; ++c)
                if (deriveFlag(epoch[P * bpp + c]) != deriveFlag(epoch[R * bpp + c])) ++mismatches;
        }
    return mismatches;
}

// True when every shared epoch cell holds the same value.
bool allCellsUniform(const std::vector<uint32_t>& epoch)
{
    for (size_t i = 1; i < epoch.size(); ++i)
        if (epoch[i] != epoch[0]) return false;
    return true;
}

constexpr int kSmallBpp = 1;                        // tiny message: one block per peer
constexpr int kLargeBpp = kDdaLLAgMaxBlocksPerPeer; // 8: widest per-peer fan-out
constexpr int kArBlocks = 16;                       // representative AR grid (<= 24)

} // namespace

// ---------------------------------------------------------------------------
// The shared array is sized to cover the widest LL collective.
// ---------------------------------------------------------------------------
TEST(DdaFabricEpochTest, EpochCount_CoversWidestCollective)
{
    for (int nRanks : {2, 4, 8})
    {
        const int cells = static_cast<int>(ddaLLEpochCount(nRanks, DDA_FABRIC_MAXBLOCKS));
        EXPECT_EQ(cells, std::max(nRanks * kDdaLLAgMaxBlocksPerPeer, DDA_FABRIC_MAXBLOCKS)) << "nRanks=" << nRanks;
        // Must index the widest AllGather grid (nRanks*8) and the AR cap (24).
        EXPECT_GE(cells, nRanks * kLargeBpp) << "nRanks=" << nRanks;
        EXPECT_GE(cells, kDdaFabricLLArMaxBlocks) << "nRanks=" << nRanks;
    }
}

// ---------------------------------------------------------------------------
// A small AllGather followed by a large one: the shared array stays uniform and
// the wider AllGather's peer-swap pairing stays consistent.
// ---------------------------------------------------------------------------
TEST(DdaFabricEpochTest, SharedEpoch_AllGatherSmallThenLarge_StaysConsistent)
{
    for (int nRanks : {4, 8})
    {
        std::vector<uint32_t> epoch(allocCells(nRanks), 0u); // cudaMemset(epochDev, 0)
        applyAllGather(epoch, nRanks, kSmallBpp);
        EXPECT_TRUE(allCellsUniform(epoch)) << "after small AG, nRanks=" << nRanks;
        EXPECT_EQ(agPairingMismatches(epoch, nRanks, kLargeBpp), 0)
            << "large AG must find every peer-swap pair consistent (nRanks=" << nRanks << ")";
    }
}

// ---------------------------------------------------------------------------
// A tiny AllReduce followed by a wide AllGather leaves the shared array
// consistent for the AllGather.
// ---------------------------------------------------------------------------
TEST(DdaFabricEpochTest, SharedEpoch_AllReduceThenAllGather_StaysConsistent)
{
    for (int nRanks : {4, 8})
    {
        std::vector<uint32_t> epoch(allocCells(nRanks), 0u);
        applyAllReduce(epoch, kArBlocks);
        EXPECT_TRUE(allCellsUniform(epoch)) << "after AR, nRanks=" << nRanks;
        EXPECT_EQ(agPairingMismatches(epoch, nRanks, kLargeBpp), 0)
            << "AG after AR must find every peer-swap pair consistent (nRanks=" << nRanks << ")";
    }
}

// ---------------------------------------------------------------------------
// A long interleaved sequence of mixed-size AllGather / AllReduce launches keeps
// the shared array uniform and each AllGather pairing-consistent every iteration.
// ---------------------------------------------------------------------------
TEST(DdaFabricEpochTest, SharedEpoch_RepeatedMixedSizes_StaysConsistent)
{
    const int agBpp[] = {1, 2, 4, 8, 3};
    const int arBlk[] = {1, 4, 16, 24, 7};
    for (int nRanks : {4, 8})
    {
        std::vector<uint32_t> epoch(allocCells(nRanks), 0u);
        for (int iter = 0; iter < 50; ++iter)
        {
            for (int bpp : agBpp)
            {
                EXPECT_EQ(agPairingMismatches(epoch, nRanks, bpp), 0)
                    << "iter=" << iter << " nRanks=" << nRanks << " bpp=" << bpp;
                applyAllGather(epoch, nRanks, bpp);
                EXPECT_TRUE(allCellsUniform(epoch)) << "after AG iter=" << iter << " bpp=" << bpp;
            }
            for (int blk : arBlk)
            {
                applyAllReduce(epoch, blk);
                EXPECT_TRUE(allCellsUniform(epoch)) << "after AR iter=" << iter << " blk=" << blk;
            }
        }
    }
}

} // namespace RcclUnitTesting
