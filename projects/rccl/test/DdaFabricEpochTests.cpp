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
// pairs derive the same flag+bank. Source of truth:
//   src/include/algorithms/all_gather/all_gather_dda_fabric_ll.h  (flag+write-back)
//   src/include/algorithms/alltoall/alltoall_dda_fabric_ll.h      (same 2-D shape)
//   src/include/algorithms/reduce_scatter/reduce_scatter_dda_fabric_ll.h (1-D)
//   src/include/dda_init_detail.h                                 (ddaLLEpochCount)
//
// Faithfulness notes for the host replay below:
//  * A launch is modelled as read-all-then-write-all. That is a valid
//    serialization despite the absence of any grid-wide sync because block b
//    reads only cell b and writes only cells e == b (mod total); those residue
//    classes are disjoint, so no block writes a cell another block reads.
//  * Cross-launch lock-step (bank alternation) holds only because DDA launches on
//    a comm are stream-serialized, so launch N fully restamps before launch N+1.
//
// The tests import the real dda_init_detail.h constants/functions so they track
// the source, and include a negative control that fails if the full-length
// restamp (all_gather_dda_fabric_ll.h: `for (e=flatBlockId; e<epochLen; e+=total)`)
// is weakened to own-cell-only. No GPU is needed.

#include "dda_init_detail.h" // ddaLLEpochCount, kDdaFabricLLArMaxBlocks, kDdaLLAgMaxBlocksPerPeer, DDA_FABRIC_MAXBLOCKS
#include "fabric_gpu_barrier.h" // meta::comms::kDdaMaxNranks

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace RcclUnitTesting {

namespace {

using nccl_dda_detail::ddaLLEpochCount;
using nccl_dda_detail::kDdaFabricLLArMaxBlocks;
using nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer;

// Sentinel meaning "restamp the whole array" (the production epochLen).
constexpr int kFullWriteBack = -1;

// Number of shared epoch cells, from the real sizing function.
size_t epochCells(int nRanks, int arMaxBlocks = DDA_FABRIC_MAXBLOCKS) {
    return ddaLLEpochCount(nRanks, arMaxBlocks);
}

// Per-block LL flag from its epoch cell:  f = cell + 1; if (f == 0) f = 2.
uint32_t deriveFlag(uint32_t cell) {
    uint32_t f = cell + 1u;
    if (f == 0u) {
        f = 2u; // skip 0 sentinel; keep bank parity
    }
    return f;
}

unsigned bankOf(uint32_t flag) {
    return flag & 1u;
}

// Replay one launch's epoch update: `total` launched blocks, restamping cells
// [0, writeBackLen) with block b owning cells b, b+total, b+2*total, ... exactly
// as the kernels do: `for (e = flatBlockId; e < epochLen; e += total) epoch[e]=flag`.
// writeBackLen == kFullWriteBack models the production full-length restamp.
void applyLaunch(std::vector<uint32_t>& epoch, int total, int writeBackLen = kFullWriteBack) {
    const int cells    = static_cast<int>(epoch.size());
    const int epochLen = (writeBackLen == kFullWriteBack) ? cells : writeBackLen;
    ASSERT_GT(total, 0);
    ASSERT_LE(total, cells);    // read index flatBlockId < total stays in bounds
    ASSERT_LE(epochLen, cells); // write index stays in bounds
    std::vector<uint32_t> flag(total);
    for (int b = 0; b < total; ++b) {
        flag[b] = deriveFlag(epoch[b]);
    }
    for (int b = 0; b < total; ++b) {
        for (int e = b; e < epochLen; e += total) {
            epoch[e] = flag[b];
        }
    }
}

// AllGather / AllToAll: 2-D grid nRanks x bpp, total = nRanks*bpp.
void applyAllGather(std::vector<uint32_t>& epoch, int nRanks, int bpp, int writeBackLen = kFullWriteBack) {
    applyLaunch(epoch, nRanks * bpp, writeBackLen);
}

void applyAllToAll(std::vector<uint32_t>& epoch, int nRanks, int bpp) {
    applyLaunch(epoch, nRanks * bpp);
}

// AllReduce / ReduceScatter: 1-D grid of `blocks` (<= kDdaFabricLLArMaxBlocks or
// clamped to ddaLLEpochLen at the launch site).
void applyAllReduce(std::vector<uint32_t>& epoch, int blocks) {
    applyLaunch(epoch, blocks);
}

void applyReduceScatter(std::vector<uint32_t>& epoch, int blocks) {
    applyLaunch(epoch, blocks);
}

// AllGather/AllToAll peer-swap pairing violations for a launch with `bpp`: for
// each (R,P,c) with P!=R the reader cell (P*bpp+c) and the paired writer cell
// (R*bpp+c) must derive the same flag. Returns the mismatch count (-1 on OOB).
long agPairingMismatches(const std::vector<uint32_t>& epoch, int nRanks, int bpp) {
    if (nRanks * bpp > static_cast<int>(epoch.size())) {
        ADD_FAILURE() << "pairing check indexes past the epoch array: nRanks*bpp=" << nRanks * bpp
                      << " cells=" << epoch.size();
        return -1;
    }
    long mismatches = 0;
    for (int R = 0; R < nRanks; ++R) {
        for (int P = 0; P < nRanks; ++P) {
            if (P == R) {
                continue; // self column is a local copy, no cross-rank flag
            }
            for (int c = 0; c < bpp; ++c) {
                if (deriveFlag(epoch[P * bpp + c]) != deriveFlag(epoch[R * bpp + c])) {
                    ++mismatches;
                }
            }
        }
    }
    return mismatches;
}

bool allCellsUniform(const std::vector<uint32_t>& epoch) {
    for (size_t i = 1; i < epoch.size(); ++i) {
        if (epoch[i] != epoch[0]) {
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
TEST(DdaFabricEpochStatic, EpochCount_CoversWidestCollective) {
    for (int arMaxBlocks : {1, kDdaFabricLLArMaxBlocks, DDA_FABRIC_MAXBLOCKS}) {
        for (int nRanks : {2, 8, 32, 33, meta::comms::kDdaMaxNranks}) {
            const long cells = static_cast<long>(ddaLLEpochCount(nRanks, arMaxBlocks));
            EXPECT_EQ(cells, std::max(nRanks * kDdaLLAgMaxBlocksPerPeer, arMaxBlocks))
                << "nRanks=" << nRanks << " arMaxBlocks=" << arMaxBlocks;
            // Must index the widest AllGather/AllToAll grid (nRanks*8) and the
            // actual AR grid, which is capped at min(arMaxBlocks, kDdaFabricLLArMaxBlocks).
            EXPECT_GE(cells, static_cast<long>(nRanks) * kDdaLLAgMaxBlocksPerPeer);
            EXPECT_GE(cells, std::min(arMaxBlocks, kDdaFabricLLArMaxBlocks));
        }
    }
}

// The AllGather launcher has no runtime clamp; it relies on ddaLLAgBlocksPerPeer
// capping per-peer blocks at kDdaLLAgMaxBlocksPerPeer. That cap must keep the
// grid within the epoch array for every rank count and MAXBLOCKS setting.
TEST(DdaFabricEpochStatic, AllGatherGridAlwaysFitsEpochArray) {
    for (int arMaxBlocks : {1, kDdaFabricLLArMaxBlocks, DDA_FABRIC_MAXBLOCKS}) {
        for (int nRanks = 2; nRanks <= meta::comms::kDdaMaxNranks; ++nRanks) {
            const long widestGrid = static_cast<long>(nRanks) * kDdaLLAgMaxBlocksPerPeer;
            EXPECT_LE(widestGrid, static_cast<long>(ddaLLEpochCount(nRanks, arMaxBlocks)))
                << "nRanks=" << nRanks << " arMaxBlocks=" << arMaxBlocks;
        }
    }
}

// Flag derivation must skip the 0 sentinel and preserve bank parity, including
// at the uint32 wraparound the kernel comment guards against.
TEST(DdaFabricEpochStatic, DeriveFlag_SkipsZeroKeepsBankParity) {
    for (uint32_t x : {0u, 1u, 2u, 3u, 100u, UINT32_MAX - 1u, UINT32_MAX}) {
        EXPECT_NE(deriveFlag(x), 0u) << "x=" << x;
        EXPECT_EQ(bankOf(deriveFlag(x)), (x + 1u) & 1u) << "x=" << x; // parity survives wrap
    }
}

// ===========================================================================
// Rank-parameterized behavior
// ===========================================================================

class DdaFabricEpoch : public ::testing::TestWithParam<int> {
};

// Name the instances 2Ranks / 4Ranks / 8Ranks / 72Ranks instead of /0../3.
INSTANTIATE_TEST_SUITE_P(Ranks, DdaFabricEpoch, ::testing::Values(2, 4, 8, meta::comms::kDdaMaxNranks),
                         [](const ::testing::TestParamInfo<int>& info) {
                             return std::to_string(info.param) + "Ranks";
                         });

// A small AllGather then a large one: array stays uniform, the wider AllGather's
// peer-swap pairing is consistent, and consecutive launches alternate banks.
TEST_P(DdaFabricEpoch, AllGatherSmallThenLarge_StaysConsistent) {
    const int nRanks = GetParam();
    std::vector<uint32_t> epoch(epochCells(nRanks), 0u);

    const uint32_t f1 = deriveFlag(epoch[0]);
    applyAllGather(epoch, nRanks, kSmallBpp);
    ASSERT_TRUE(allCellsUniform(epoch));
    ASSERT_EQ(agPairingMismatches(epoch, nRanks, kLargeBpp), 0);

    const uint32_t f2 = deriveFlag(epoch[0]);
    ASSERT_NE(bankOf(f2), bankOf(f1)) << "consecutive launches must alternate banks";
    applyAllGather(epoch, nRanks, kLargeBpp);
    ASSERT_TRUE(allCellsUniform(epoch));
    ASSERT_EQ(agPairingMismatches(epoch, nRanks, kLargeBpp), 0);
}

TEST_P(DdaFabricEpoch, AllReduceThenAllGather_StaysConsistent) {
    const int nRanks = GetParam();
    std::vector<uint32_t> epoch(epochCells(nRanks), 0u);
    applyAllReduce(epoch, kDdaFabricLLArMaxBlocks); // widest AR grid
    ASSERT_TRUE(allCellsUniform(epoch));
    ASSERT_EQ(agPairingMismatches(epoch, nRanks, kLargeBpp), 0);
}

TEST_P(DdaFabricEpoch, ReduceScatterThenAllGather_StaysConsistent) {
    const int nRanks = GetParam();
    std::vector<uint32_t> epoch(epochCells(nRanks), 0u);
    applyReduceScatter(epoch, std::min<int>(kDdaFabricLLArMaxBlocks, static_cast<int>(epoch.size())));
    ASSERT_TRUE(allCellsUniform(epoch));
    ASSERT_EQ(agPairingMismatches(epoch, nRanks, kLargeBpp), 0);
}

TEST_P(DdaFabricEpoch, AllToAllThenAllGather_StaysConsistent) {
    const int nRanks = GetParam();
    std::vector<uint32_t> epoch(epochCells(nRanks), 0u);
    applyAllToAll(epoch, nRanks, kLargeBpp); // A2A is peer-swapped like AG
    ASSERT_TRUE(allCellsUniform(epoch));
    ASSERT_EQ(agPairingMismatches(epoch, nRanks, kLargeBpp), 0);
}

// Interleaved mixed-size launches keep the array uniform and every AllGather /
// AllToAll pairing consistent each step. ASSERT_ stops at the first divergence.
TEST_P(DdaFabricEpoch, RepeatedMixedSizes_StaysConsistent) {
    const int nRanks = GetParam();
    const int agBpp[] = {1, 2, 4, kLargeBpp, 3};
    const int arBlk[] = {1, 4, kDdaFabricLLArMaxBlocks, 7};
    std::vector<uint32_t> epoch(epochCells(nRanks), 0u);

    uint32_t prevFlag = 0u;
    for (int iter = 0; iter < 4; ++iter) {
        for (int bpp : agBpp) {
            ASSERT_EQ(agPairingMismatches(epoch, nRanks, bpp), 0) << "iter=" << iter << " bpp=" << bpp;
            const uint32_t f = deriveFlag(epoch[0]);
            ASSERT_NE(bankOf(f), bankOf(prevFlag)) << "banks must alternate, iter=" << iter << " bpp=" << bpp;
            prevFlag = f;
            applyAllGather(epoch, nRanks, bpp);
            ASSERT_TRUE(allCellsUniform(epoch)) << "after AG iter=" << iter << " bpp=" << bpp;
        }
        for (int blk : arBlk) {
            const uint32_t f = deriveFlag(epoch[0]);
            ASSERT_NE(bankOf(f), bankOf(prevFlag)) << "banks must alternate, iter=" << iter << " blk=" << blk;
            prevFlag = f;
            applyAllReduce(epoch, blk);
            ASSERT_TRUE(allCellsUniform(epoch)) << "after AR iter=" << iter << " blk=" << blk;
        }
    }
}

// ===========================================================================
// Negative controls -- these MUST fail if the harness (or the modelled kernel
// behavior) loses the property; they prove the positive tests have teeth.
// ===========================================================================

// If the full-length restamp is weakened to own-cell-only (the exact regression
// `e < epochLen` -> `e < total` in all_gather_dda_fabric_ll.h:133), a small AG
// followed by a wide AG leaves stale high cells and the pairing breaks. The
// full-length restamp on the same sequence stays consistent.
TEST_P(DdaFabricEpoch, NegativeControl_ShortWriteBackBreaksPairing) {
    const int nRanks = GetParam();

    std::vector<uint32_t> full(epochCells(nRanks), 0u);
    applyAllGather(full, nRanks, kSmallBpp, kFullWriteBack);
    ASSERT_EQ(agPairingMismatches(full, nRanks, kLargeBpp), 0) << "full restamp must stay consistent";

    std::vector<uint32_t> shortWb(epochCells(nRanks), 0u);
    applyAllGather(shortWb, nRanks, kSmallBpp, /*writeBackLen=*/nRanks * kSmallBpp); // own-cell-only
    EXPECT_GT(agPairingMismatches(shortWb, nRanks, kLargeBpp), 0)
        << "own-cell-only restamp must break pairing -> proves the full-length restamp is load-bearing";
}

// The kernels early-return for blocks with peer >= nRanks (grid.x > nRanks)
// BEFORE the write-back, so those residue classes are never restamped and the
// array goes non-uniform. Launch sites use grid.x == nRanks; this pins that
// assumption.
TEST_P(DdaFabricEpoch, NegativeControl_PeerBeyondNRanksBreaksUniformity) {
    const int nRanks = GetParam();
    const int bpp    = 2;

    std::vector<uint32_t> ok(epochCells(nRanks), 0u);
    applyAllGather(ok, nRanks, bpp);
    ASSERT_TRUE(allCellsUniform(ok)) << "grid.x == nRanks must stay uniform";

    // Model grid.x == nRanks+1: all blocks read, but peer==nRanks blocks return
    // before restamping their residue class.
    std::vector<uint32_t> bad(epochCells(nRanks), 0u);
    const int gridX = nRanks + 1;
    const int total = gridX * bpp;
    ASSERT_LE(total, static_cast<int>(bad.size()));
    std::vector<uint32_t> flag(total);
    for (int b = 0; b < total; ++b) {
        flag[b] = deriveFlag(bad[b]);
    }
    for (int b = 0; b < total; ++b) {
        if (b / bpp < nRanks) { // peer = flatBlockId / bpp; peer >= nRanks returns early
            for (int e = b; e < static_cast<int>(bad.size()); e += total) {
                bad[e] = flag[b];
            }
        }
    }
    EXPECT_FALSE(allCellsUniform(bad)) << "grid.x > nRanks leaves residue classes unstamped -> non-uniform";
}

} // namespace RcclUnitTesting
