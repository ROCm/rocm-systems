/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Size bands for the gfx950 symmetric reduce kernel block widths.
//
// ncclSymkGfx950BlockThreads() is the width decision lifted out of
// ncclSymkPickKernel(). It is a pure function of the collective, the kernel
// class, the rank count and the message size, so these tests need no
// communicator and no GPU.
//
// The expected widths below are written as literals on purpose. Asserting
// against the same constants the implementation uses would pass even if a
// constant were changed, which is exactly the regression these tests exist to
// catch.
//
// Two of these tests are not about the bands at all. AllGatherAlwaysUpstreamWidth
// and UntunedCollectivesKeepUpstreamWidth pin down the blast radius: the
// tuning must never reach a collective other than AllReduce and ReduceScatter.

#include "gtest/gtest.h"
#include "nccl.h"

#include "collectives.h"
#include "sym_kernels.h"

#include <cstddef>

namespace RcclUnitTesting
{

namespace
{

// Widths the bands select between, spelled out independently of the source.
constexpr int kNarrow  = 256;  // also upstream's ncclSymkMaxThreads
constexpr int kWide    = 512;
constexpr int kWidest  = 1024;

constexpr int kRanks = 8; // every threshold was fitted on 8 ranks

constexpr size_t KiB = 1024;
constexpr size_t MiB = 1024 * KiB;
constexpr size_t GiB = 1024 * MiB;

// ReduceScatter bands are expressed in bus bytes, which is nRanks * nBytes.
// Tests state the bus figure they mean and convert here so the intent is visible.
constexpr size_t busToBytes(size_t busBytes)
{
    return busBytes / kRanks;
}

int llWidth(ncclFunc_t coll, size_t nBytes)
{
    return ncclSymkGfx950BlockThreads(coll, /*isLL=*/true, kRanks, nBytes);
}

int ldWidth(ncclFunc_t coll, size_t nBytes)
{
    return ncclSymkGfx950BlockThreads(coll, /*isLL=*/false, kRanks, nBytes);
}

} // namespace

// ===========================================================================
// AllReduce LL: narrows below 64 KB so the epoch barrier does not widen
// without removing an epoch. The cost model hands AllReduce to LD from 128 KB
// up, so 64 KB is the only LL size above the threshold.
// ===========================================================================

TEST(SymkBlockWidthTest, AllReduceLLNarrowsBelow64K)
{
    EXPECT_EQ(llWidth(ncclFuncAllReduce, 4), kNarrow);
    EXPECT_EQ(llWidth(ncclFuncAllReduce, 2 * KiB), kNarrow);
    EXPECT_EQ(llWidth(ncclFuncAllReduce, 32 * KiB), kNarrow);

    // One element short of the threshold still narrows.
    EXPECT_EQ(llWidth(ncclFuncAllReduce, 64 * KiB - sizeof(float)), kNarrow);

    // The threshold itself is the first wide size.
    EXPECT_EQ(llWidth(ncclFuncAllReduce, 64 * KiB), kWide);
}

// ===========================================================================
// ReduceScatter LL: never narrows. Narrowing it was measured and did not help,
// so it stays at the full width across every size.
// ===========================================================================

TEST(SymkBlockWidthTest, ReduceScatterLLAlwaysWide)
{
    EXPECT_EQ(llWidth(ncclFuncReduceScatter, 128), kWide);
    EXPECT_EQ(llWidth(ncclFuncReduceScatter, 8 * KiB), kWide);

    // Crossing AllReduce's LL threshold must not affect ReduceScatter.
    EXPECT_EQ(llWidth(ncclFuncReduceScatter, 64 * KiB - sizeof(float)), kWide);
    EXPECT_EQ(llWidth(ncclFuncReduceScatter, 64 * KiB), kWide);
    EXPECT_EQ(llWidth(ncclFuncReduceScatter, 256 * KiB), kWide);
}

// ===========================================================================
// AllReduce LD: narrow below 512 KB, wide to 2 MB, narrow again to 1 GB, wide
// above. Thresholds are message bytes.
// ===========================================================================

TEST(SymkBlockWidthTest, AllReduceLDNarrowBelowTailSaturation)
{
    EXPECT_EQ(ldWidth(ncclFuncAllReduce, 128 * KiB), kNarrow);
    EXPECT_EQ(ldWidth(ncclFuncAllReduce, 256 * KiB), kNarrow);
    EXPECT_EQ(ldWidth(ncclFuncAllReduce, 512 * KiB - sizeof(float)), kNarrow);
}

TEST(SymkBlockWidthTest, AllReduceLDWideAcrossTailSaturatedBand)
{
    EXPECT_EQ(ldWidth(ncclFuncAllReduce, 512 * KiB), kWide);
    EXPECT_EQ(ldWidth(ncclFuncAllReduce, 1 * MiB), kWide);
    EXPECT_EQ(ldWidth(ncclFuncAllReduce, 2 * MiB - sizeof(float)), kWide);
}

TEST(SymkBlockWidthTest, AllReduceLDNarrowAcrossDeepTiers)
{
    EXPECT_EQ(ldWidth(ncclFuncAllReduce, 2 * MiB), kNarrow);
    EXPECT_EQ(ldWidth(ncclFuncAllReduce, 512 * MiB), kNarrow);
    EXPECT_EQ(ldWidth(ncclFuncAllReduce, 1 * GiB - sizeof(float)), kNarrow);
}

TEST(SymkBlockWidthTest, AllReduceLDWideAboveOccupancyBound)
{
    EXPECT_EQ(ldWidth(ncclFuncAllReduce, 1 * GiB), kWide);
    EXPECT_EQ(ldWidth(ncclFuncAllReduce, 4 * GiB), kWide);
}

// ===========================================================================
// ReduceScatter LD: wide below 1 MB bus, widest to 16 MB bus, narrow above.
// Thresholds are bus bytes, so they match the rccl-tests size column.
// ===========================================================================

TEST(SymkBlockWidthTest, ReduceScatterLDBands)
{
    // Below the 1 MB bus floor.
    EXPECT_EQ(ldWidth(ncclFuncReduceScatter, busToBytes(512 * KiB)), kWide);
    EXPECT_EQ(ldWidth(ncclFuncReduceScatter, busToBytes(1 * MiB) - 1), kWide);

    // 1 MB to 16 MB bus takes the widest block.
    EXPECT_EQ(ldWidth(ncclFuncReduceScatter, busToBytes(1 * MiB)), kWidest);
    EXPECT_EQ(ldWidth(ncclFuncReduceScatter, busToBytes(8 * MiB)), kWidest);
    EXPECT_EQ(ldWidth(ncclFuncReduceScatter, busToBytes(16 * MiB) - 1), kWidest);

    // 16 MB bus and up narrows to keep iterations per globally strided warp high.
    EXPECT_EQ(ldWidth(ncclFuncReduceScatter, busToBytes(16 * MiB)), kNarrow);
    EXPECT_EQ(ldWidth(ncclFuncReduceScatter, busToBytes(4 * GiB)), kNarrow);
}

// ===========================================================================
// Blast radius. These two matter more than the bands above: they are the only
// automated guard that the tuning stays confined to the two reduce kernels.
// ===========================================================================

TEST(SymkBlockWidthTest, AllGatherAlwaysUpstreamWidth)
{
    const size_t sizes[] = {4, 2 * KiB, 64 * KiB, 512 * KiB, 8 * MiB, 1 * GiB};

    for(size_t nBytes : sizes)
    {
        EXPECT_EQ(llWidth(ncclFuncAllGather, nBytes), kNarrow)
            << "AllGather LL width changed at " << nBytes << " bytes";
        EXPECT_EQ(ldWidth(ncclFuncAllGather, nBytes), kNarrow)
            << "AllGather LD width changed at " << nBytes << " bytes";
    }
}

TEST(SymkBlockWidthTest, UntunedCollectivesKeepUpstreamWidth)
{
    const ncclFunc_t colls[] = {ncclFuncBroadcast, ncclFuncReduce, ncclFuncAlltoAll};
    const size_t     sizes[] = {4, 64 * KiB, 8 * MiB, 1 * GiB};

    for(ncclFunc_t coll : colls)
    {
        for(size_t nBytes : sizes)
        {
            EXPECT_EQ(llWidth(coll, nBytes), kNarrow)
                << "coll " << ncclFuncToString(coll) << " LL width changed at " << nBytes;
            EXPECT_EQ(ldWidth(coll, nBytes), kNarrow)
                << "coll " << ncclFuncToString(coll) << " LD width changed at " << nBytes;
        }
    }
}

// ===========================================================================
// Every band must land on a width the hardware can actually launch, and the
// launch must divide evenly into both wavefront sizes so nWarps * WarpSize
// reproduces it exactly.
// ===========================================================================

TEST(SymkBlockWidthTest, AllWidthsAreLaunchable)
{
    const ncclFunc_t colls[] = {ncclFuncAllReduce, ncclFuncReduceScatter, ncclFuncAllGather};
    const size_t     sizes[] = {4,
                                2 * KiB,
                                64 * KiB,
                                256 * KiB,
                                512 * KiB,
                                1 * MiB,
                                2 * MiB,
                                16 * MiB,
                                512 * MiB,
                                1 * GiB,
                                4 * GiB};

    for(ncclFunc_t coll : colls)
    {
        for(size_t nBytes : sizes)
        {
            for(bool isLL : {false, true})
            {
                int nThreads = ncclSymkGfx950BlockThreads(coll, isLL, kRanks, nBytes);

                EXPECT_GE(nThreads, kNarrow);
                EXPECT_LE(nThreads, kWidest) << "exceeds the 1024 thread workgroup limit";
                EXPECT_EQ(nThreads % 64, 0) << "not a multiple of a 64-wide wavefront";
            }
        }
    }
}

} // namespace RcclUnitTesting
