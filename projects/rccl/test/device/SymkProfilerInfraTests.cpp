/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Device-side unit tests for device-kernel changes introduced by the
// NCCL v2.31.2-1 sync:
//
//  1. loadSsize() -- new PTX-backed volatile ssize_t load added to
//     common_kernel.h, sibling of loadInt().
//
//  2. ncclSymk tuning constants -- ncclSymkMinWarpsPerBlock,
//     ncclSymkBytePerPack, ncclSymkGetBytesPerChunk() and the six
//     derived *BytePerChunk constants extracted from per-function
//     local constexpr variables into sym_kernels.h.
//
//  3. ncclSymkDevWorkArgs profiler layout -- new profilerEnabled field,
//     calcArgsSize(profiler=true/false), getProfilerCounters() and
//     getWorkRange() offset arithmetic.
//
//  4. ncclSymKernelStr[] -- must have exactly ncclSymkKernelId_Count
//     entries, covering every kernel in the enum.
//
// All tests here use only header-only symbols (constexpr, inline, template)
// from sym_kernels.h and common_kernel.h; they belong in rccl-UnitTestsFixtures.
// Tests that reference the extern librccl symbols (ncclSymkKernelCount,
// ncclSymkKernelList, ncclSymkKernelListProfile) live in
// SymkProfilerKernelListTests.cpp (rccl-UnitTests).
//
// Test 1 launches a hipLaunchKernelGGL kernel.

#include "DeviceTestBase.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "common_kernel.h"   // loadSsize (and loadInt for reference)
#include "sym_kernels.h"     // ncclSymkDevWorkArgs, tuning constants

namespace RcclUnitTesting
{

// ---------------------------------------------------------------------------
// 1. loadSsize() device function
// ---------------------------------------------------------------------------
//
// loadSsize() does a PTX ld.volatile.global.s64 on a ssize_t*.  We verify
// that a device kernel using loadSsize() returns the same value that was
// written from the host, and that negative values survive the signed transfer.

__global__ void kernelLoadSsize(const ssize_t* __restrict__ src, ssize_t* __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = loadSsize(const_cast<ssize_t*>(&src[i]));
}

class LoadSsizeTests : public DeviceTestBase
{
protected:
    void RunTest(const std::vector<ssize_t>& h_in)
    {
        const int N = static_cast<int>(h_in.size());
        DeviceBuffer<ssize_t> d_in(N), d_out(N);
        d_in.copyFrom(h_in);
        d_out.zero();

        hipLaunchKernelGGL(kernelLoadSsize, gridFor(N), kDefaultBlockSize, 0, nullptr,
                           d_in.ptr, d_out.ptr, N);
        syncAndCheck();

        auto h_out = d_out.copyTo();
        for (int i = 0; i < N; i++)
            EXPECT_EQ(h_in[i], h_out[i]) << "mismatch at index " << i;
    }
};

// Positive values pass through unchanged.
TEST_F(LoadSsizeTests, PositiveValues)
{
    const int N = 512;
    std::vector<ssize_t> h_in(N);
    for (int i = 0; i < N; i++) h_in[i] = static_cast<ssize_t>(i) * 1024 + 1;
    RunTest(h_in);
}

// Negative values must survive the signed 64-bit PTX load.
TEST_F(LoadSsizeTests, NegativeValues)
{
    const int N = 256;
    std::vector<ssize_t> h_in(N);
    for (int i = 0; i < N; i++) h_in[i] = -static_cast<ssize_t>(i + 1) * 4096;
    RunTest(h_in);
}

// Zero and extremes (SSIZE_MIN / SSIZE_MAX).
TEST_F(LoadSsizeTests, ExtremesAndZero)
{
    std::vector<ssize_t> h_in = {
        ssize_t(0),
        ssize_t(1),
        ssize_t(-1),
        std::numeric_limits<ssize_t>::max(),
        std::numeric_limits<ssize_t>::min()
    };
    RunTest(h_in);
}

// ---------------------------------------------------------------------------
// 2. Tuning constant correctness
// ---------------------------------------------------------------------------
//
// ncclSymkGetBytesPerChunk() is __host__ __device__, so host-side evaluation
// uses WARP_SIZE=32 (the host fallback defined in device.h).  The derived
// constants are all built from the same formula, so we pin the formula-to-
// constant relationship rather than hard-coding numeric results.

class SymkTuningConstantTests : public ::testing::Test {};

TEST_F(SymkTuningConstantTests, MinWarpsPerBlockIsPositive)
{
    EXPECT_GT(ncclSymkMinWarpsPerBlock, 0);
}

TEST_F(SymkTuningConstantTests, BytePerPackIsPowerOfTwo)
{
    int bpp = ncclSymkBytePerPack;
    EXPECT_GT(bpp, 0);
    EXPECT_EQ(bpp & (bpp - 1), 0) << "ncclSymkBytePerPack must be a power of two";
}

TEST_F(SymkTuningConstantTests, GetBytesPerChunkMatchesDerivedConstants)
{
    EXPECT_EQ(ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, ncclSymkUnrollPacks),
              ncclSymkBytePerChunk);
    EXPECT_EQ(ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, ncclSymkDeepUnrollPacks),
              ncclSymkDeepBytePerChunk);
    EXPECT_EQ(ncclSymkGetBytesPerChunk(1, ncclSymkDeepUnrollPacks),
              ncclSymkMultimemDeepBytePerChunk);
    EXPECT_EQ(ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, ncclSymkAlign256BDeepUnrollPacks),
              ncclSymkAlign256BDeepBytePerChunk);
}

TEST_F(SymkTuningConstantTests, DeepIsStrictlyLargerThanSm)
{
    // TMA path uses a deeper unroll; its chunk must be strictly larger.
    EXPECT_GT(ncclSymkDeepBytePerChunk, ncclSymkBytePerChunk);
}

TEST_F(SymkTuningConstantTests, Align256BDeepIsLargestChunk)
{
    EXPECT_GE(ncclSymkAlign256BDeepBytePerChunk, ncclSymkDeepBytePerChunk);
}

TEST_F(SymkTuningConstantTests, MultimemDeepUsesSingleWarp)
{
    // ncclSymkMultimemDeepBytePerChunk is exactly 1/ncclSymkMinWarpsPerBlock
    // of ncclSymkDeepBytePerChunk because the multimem bcast loop runs a
    // single warp at a time.
    EXPECT_EQ(ncclSymkMultimemDeepBytePerChunk * ncclSymkMinWarpsPerBlock,
              ncclSymkDeepBytePerChunk);
}

TEST_F(SymkTuningConstantTests, ChunksAreAlignedToBytePerPack)
{
    EXPECT_EQ(ncclSymkBytePerChunk              % ncclSymkBytePerPack, 0);
    EXPECT_EQ(ncclSymkDeepBytePerChunk          % ncclSymkBytePerPack, 0);
    EXPECT_EQ(ncclSymkMultimemDeepBytePerChunk  % ncclSymkBytePerPack, 0);
    EXPECT_EQ(ncclSymkAlign256BDeepBytePerChunk % ncclSymkBytePerPack, 0);
}

// ---------------------------------------------------------------------------
// 3. ncclSymkDevWorkArgs profiler layout
// ---------------------------------------------------------------------------
//
// The sync added:
//   - int profilerEnabled  (new field)
//   - calcArgsSize(nChannels, nWorks, profiler)  extended signature
//   - getProfilerCounters() returns pointer immediately after the fixed header
//   - getWorkRange()  skips the counter array when profilerEnabled is set
//
// We verify the layout without touching device memory.

class SymkDevWorkArgsLayoutTests : public ::testing::Test
{
protected:
    // alignUp(sizeof(ncclSymkDevWorkArgs), 16)
    static constexpr size_t kHeaderSize =
        (sizeof(ncclSymkDevWorkArgs) + 15u) & ~15u;

    static constexpr size_t kU64Size   = sizeof(uint64_t);
    static constexpr size_t kRangeSize = sizeof(ncclSymkChannelWorkRange);
    static constexpr size_t kWorkSize  = sizeof(ncclSymkDevWork);
};

TEST_F(SymkDevWorkArgsLayoutTests, CalcArgsSizeNoProfilerMatchesManual)
{
    const int nCh = 4, nWk = 8;
    size_t expected = kHeaderSize
                    + ((nCh * kRangeSize + 15u) & ~15u)
                    + nWk * kWorkSize;
    EXPECT_EQ(ncclSymkDevWorkArgs::calcArgsSize(nCh, nWk, /*profiler=*/false), expected);
}

TEST_F(SymkDevWorkArgsLayoutTests, CalcArgsSizeWithProfilerMatchesManual)
{
    const int nCh = 4, nWk = 8;
    size_t expected = kHeaderSize
                    + ((nCh * kU64Size   + 15u) & ~15u)
                    + ((nCh * kRangeSize + 15u) & ~15u)
                    + nWk * kWorkSize;
    EXPECT_EQ(ncclSymkDevWorkArgs::calcArgsSize(nCh, nWk, /*profiler=*/true), expected);
}

TEST_F(SymkDevWorkArgsLayoutTests, ProfilerSizeIsLargerThanBase)
{
    const int nCh = 4, nWk = 8;
    EXPECT_GT(ncclSymkDevWorkArgs::calcArgsSize(nCh, nWk, /*profiler=*/true),
              ncclSymkDevWorkArgs::calcArgsSize(nCh, nWk, /*profiler=*/false));
}

TEST_F(SymkDevWorkArgsLayoutTests, GetProfilerCountersPointsImmediatelyAfterHeader)
{
    const int nCh = 4, nWk = 0;
    size_t total = ncclSymkDevWorkArgs::calcArgsSize(nCh, nWk, /*profiler=*/true);
    std::vector<uint8_t> buf(total, 0);

    auto* args = reinterpret_cast<ncclSymkDevWorkArgs*>(buf.data());
    args->profilerEnabled = 1;
    args->nMaxChannels    = nCh;

    ptrdiff_t countersOff =
        reinterpret_cast<uint8_t*>(args->getProfilerCounters()) - buf.data();
    EXPECT_EQ(static_cast<size_t>(countersOff), kHeaderSize)
        << "getProfilerCounters() must point to the byte immediately after the aligned header";
}

TEST_F(SymkDevWorkArgsLayoutTests, GetWorkRangeSkipsCounterArrayWhenProfilerEnabled)
{
    const int nCh = 4, nWk = 0;
    size_t total = ncclSymkDevWorkArgs::calcArgsSize(nCh, nWk, /*profiler=*/true);
    std::vector<uint8_t> buf(total, 0);

    auto* args = reinterpret_cast<ncclSymkDevWorkArgs*>(buf.data());
    args->profilerEnabled = 1;
    args->nMaxChannels    = nCh;

    ptrdiff_t rangeOff =
        reinterpret_cast<uint8_t*>(args->getWorkRange()) - buf.data();
    size_t expectedOff = kHeaderSize + ((nCh * kU64Size + 15u) & ~15u);
    EXPECT_EQ(static_cast<size_t>(rangeOff), expectedOff)
        << "getWorkRange() with profilerEnabled must skip the profiler counter array";
}

TEST_F(SymkDevWorkArgsLayoutTests, GetWorkRangeNoProfilerPointsImmediatelyAfterHeader)
{
    const int nCh = 2, nWk = 0;
    size_t total = ncclSymkDevWorkArgs::calcArgsSize(nCh, nWk, /*profiler=*/false);
    std::vector<uint8_t> buf(total, 0);

    auto* args = reinterpret_cast<ncclSymkDevWorkArgs*>(buf.data());
    args->profilerEnabled = 0;
    args->nMaxChannels    = nCh;

    ptrdiff_t rangeOff =
        reinterpret_cast<uint8_t*>(args->getWorkRange()) - buf.data();
    EXPECT_EQ(static_cast<size_t>(rangeOff), kHeaderSize)
        << "getWorkRange() without profiler must point immediately after the header";
}

TEST_F(SymkDevWorkArgsLayoutTests, CounterArrayIs16ByteAligned)
{
    // The header size must itself be 16-byte aligned (by construction).
    EXPECT_EQ(kHeaderSize % 16, 0u);
    // The rounded counter region also produces a 16-byte-aligned boundary.
    constexpr size_t countersRounded = (4 * sizeof(uint64_t) + 15u) & ~15u;
    EXPECT_EQ((kHeaderSize + countersRounded) % 16, 0u);
}

// ---------------------------------------------------------------------------
// 4. ncclSymKernelStr[] -- coverage of every kernel in the enum
// ---------------------------------------------------------------------------
//
// The new ncclSymKernelStr[] array must have exactly ncclSymkKernelId_Count
// entries and every entry must be a non-empty C string.

class SymkKernelStrTests : public ::testing::Test {};

TEST_F(SymkKernelStrTests, ArrayLengthMatchesEnum)
{
    constexpr size_t strCount = sizeof(ncclSymKernelStr) / sizeof(ncclSymKernelStr[0]);
    EXPECT_EQ(static_cast<int>(strCount), ncclSymkKernelId_Count)
        << "ncclSymKernelStr[] entry count must equal ncclSymkKernelId_Count; "
           "add/remove a string whenever you add/remove a kernel id";
}

TEST_F(SymkKernelStrTests, AllEntriesAreNonEmpty)
{
    constexpr size_t strCount = sizeof(ncclSymKernelStr) / sizeof(ncclSymKernelStr[0]);
    for (size_t i = 0; i < strCount; i++) {
        ASSERT_NE(ncclSymKernelStr[i], nullptr)    << "null name at index " << i;
        EXPECT_GT(strlen(ncclSymKernelStr[i]), 0u) << "empty name at index " << i;
    }
}

TEST_F(SymkKernelStrTests, KnownNamesArePresent)
{
    const char* required[] = {
        "AllReduce_AGxLL_R",
        "AllReduce_RSxLD_AGxST",
        "AllGather_ST",
        "AllGather_LL",
        "ReduceScatter_LD",
        "ReduceScatter_LL",
    };
    constexpr size_t strCount = sizeof(ncclSymKernelStr) / sizeof(ncclSymKernelStr[0]);
    for (const char* name : required) {
        bool found = false;
        for (size_t i = 0; i < strCount && !found; i++)
            if (ncclSymKernelStr[i] && strcmp(ncclSymKernelStr[i], name) == 0)
                found = true;
        EXPECT_TRUE(found) << "kernel name '" << name << "' missing from ncclSymKernelStr[]";
    }
}

} // namespace RcclUnitTesting
