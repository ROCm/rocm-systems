/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/device/CollCommon_ll128.h"
#include "gtest/gtest.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>
#include <vector>

namespace RcclUnitTesting
{

namespace
{
using dda::common::fp8e4m3;
using dda::common::fp8e5m2;

constexpr uint32_t kFp8Pairs = 256 * 256;

struct Fp8AddMismatches
{
    uint32_t word32;
    uint32_t word64;
    uint32_t firstBadPair;
};

// RCCL's core fp8 add, which the ring/tree reduce kernels use. The pair's
// second element is zero.
template <typename Tag>
__device__ uint8_t coreFp8Add(uint8_t a, uint8_t b)
{
    if constexpr (std::is_same<Tag, fp8e4m3>::value)
        return static_cast<uint8_t>(hadd2(a, b) & 0xff);
    else
        return static_cast<uint8_t>(hadd2_b(a, b) & 0xff);
}

// One thread per (a, b) byte pair, placed in each lane of the 4-byte word that
// vecElementAdd adds and of the 8-byte word ddaLL128AddWord adds. The other
// lanes carry unrelated pairs, so a lane that leaks into another shows up too.
template <typename Tag>
__global__ void fp8AddMatchesCoreKernel(Fp8AddMismatches* out)
{
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= kFp8Pairs)
        return;

    for (int lane = 0; lane < 4; ++lane)
    {
        uint32_t wa = 0, wb = 0;
        uint8_t  expected[4];
        for (int k = 0; k < 4; ++k)
        {
            const uint32_t q  = (k == lane) ? p : ((p * 2654435761u + k * 40503u) & 0xffff);
            const uint8_t  qa = q & 0xff, qb = q >> 8;
            wa |= static_cast<uint32_t>(qa) << (8 * k);
            wb |= static_cast<uint32_t>(qb) << (8 * k);
            expected[k] = coreFp8Add<Tag>(qa, qb);
        }
        const uint32_t sum = dda::common::vecElementAdd<Tag>(wa, wb);
        for (int k = 0; k < 4; ++k)
        {
            if (((sum >> (8 * k)) & 0xff) != expected[k])
            {
                atomicAdd(&out->word32, 1u);
                atomicCAS(&out->firstBadPair, 0xffffffffu, p);
            }
        }
    }

    for (int lane = 0; lane < 8; ++lane)
    {
        uint64_t wa = 0, wb = 0;
        uint8_t  expected[8];
        for (int k = 0; k < 8; ++k)
        {
            const uint32_t q  = (k == lane) ? p : ((p * 2246822519u + k * 3266489917u) & 0xffff);
            const uint8_t  qa = q & 0xff, qb = q >> 8;
            wa |= static_cast<uint64_t>(qa) << (8 * k);
            wb |= static_cast<uint64_t>(qb) << (8 * k);
            expected[k] = coreFp8Add<Tag>(qa, qb);
        }
        const uint64_t sum = dda::common::ddaLL128AddWord<Tag>(wa, wb);
        for (int k = 0; k < 8; ++k)
        {
            if (((sum >> (8 * k)) & 0xff) != expected[k])
            {
                atomicAdd(&out->word64, 1u);
                atomicCAS(&out->firstBadPair, 0xffffffffu, p);
            }
        }
    }
}

// Adds a[i] + b[i] four lanes at a time through vecElementAdd, converting to
// and from the device's own fp8 encoding (FNUZ or OCP) in device code.
template <typename Fp8, typename Tag>
__global__ void fp8AddFloatsKernel(const float* a, const float* b, float* out)
{
    uint32_t wa = 0, wb = 0;
    for (int k = 0; k < 4; ++k)
    {
        const Fp8 fa(a[k]), fb(b[k]);
        uint8_t   ba, bb;
        __builtin_memcpy(&ba, &fa, 1);
        __builtin_memcpy(&bb, &fb, 1);
        wa |= static_cast<uint32_t>(ba) << (8 * k);
        wb |= static_cast<uint32_t>(bb) << (8 * k);
    }
    const uint32_t sum = dda::common::vecElementAdd<Tag>(wa, wb);
    for (int k = 0; k < 4; ++k)
    {
        const uint8_t bits = (sum >> (8 * k)) & 0xff;
        Fp8           v;
        __builtin_memcpy(&v, &bits, 1);
        out[k] = static_cast<float>(v);
    }
}

bool haveGpu()
{
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

template <typename Tag>
void expectFp8AddMatchesCore()
{
    Fp8AddMismatches* d = nullptr;
    ASSERT_EQ(hipMalloc(&d, sizeof(*d)), hipSuccess);
    const Fp8AddMismatches init{0, 0, 0xffffffffu};
    ASSERT_EQ(hipMemcpy(d, &init, sizeof(init), hipMemcpyHostToDevice), hipSuccess);
    hipLaunchKernelGGL(fp8AddMatchesCoreKernel<Tag>, dim3(kFp8Pairs / 256), dim3(256), 0, 0, d);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    Fp8AddMismatches h{};
    ASSERT_EQ(hipMemcpy(&h, d, sizeof(h), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(hipFree(d), hipSuccess);

    EXPECT_EQ(h.word32, 0u) << "first mismatching (a, b) pair 0x" << std::hex << h.firstBadPair;
    EXPECT_EQ(h.word64, 0u) << "first mismatching (a, b) pair 0x" << std::hex << h.firstBadPair;
}

// Sums every fp8 format represents exactly, so the result does not depend on
// rounding; one per lane, so a lane swap shows up as a wrong value.
template <typename Fp8, typename Tag>
void expectFp8AddExactSums()
{
    const std::vector<float> a        = {1.0f, 0.5f, -1.5f, 8.0f};
    const std::vector<float> b        = {2.0f, 0.25f, 0.5f, 8.0f};
    const std::vector<float> expected = {3.0f, 0.75f, -1.0f, 16.0f};

    float* d = nullptr;
    ASSERT_EQ(hipMalloc(&d, 12 * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemcpy(d, a.data(), 4 * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d + 4, b.data(), 4 * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    hipLaunchKernelGGL((fp8AddFloatsKernel<Fp8, Tag>), dim3(1), dim3(1), 0, 0, d, d + 4, d + 8);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    std::vector<float> got(4);
    ASSERT_EQ(hipMemcpy(got.data(), d + 8, 4 * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(hipFree(d), hipSuccess);

    for (int k = 0; k < 4; ++k)
        EXPECT_EQ(got[k], expected[k]) << a[k] << " + " << b[k] << " in lane " << k;
}
} // namespace

// The DDA reduce kernels add fp8 through vecElementAdd / ddaLL128AddWord, which
// must round exactly as the core reduce kernels do, or DDA and non-DDA
// collectives would disagree on the same inputs.
TEST(DdaCollCommon, Fp8E4M3AddMatchesCoreForAllPairs)
{
    if (!haveGpu())
        GTEST_SKIP() << "No GPU";
    expectFp8AddMatchesCore<fp8e4m3>();
}

TEST(DdaCollCommon, Fp8E5M2AddMatchesCoreForAllPairs)
{
    if (!haveGpu())
        GTEST_SKIP() << "No GPU";
    expectFp8AddMatchesCore<fp8e5m2>();
}

TEST(DdaCollCommon, Fp8E4M3AddExactSums)
{
    if (!haveGpu())
        GTEST_SKIP() << "No GPU";
    expectFp8AddExactSums<rccl_float8, fp8e4m3>();
}

TEST(DdaCollCommon, Fp8E5M2AddExactSums)
{
    if (!haveGpu())
        GTEST_SKIP() << "No GPU";
    expectFp8AddExactSums<rccl_bfloat8, fp8e5m2>();
}

TEST(DdaCollCommon, divRoundUp)
{
    EXPECT_EQ(dda::common::divRoundUp(0, 8), 1u);
    EXPECT_EQ(dda::common::divRoundUp(1, 8), 1u);
    EXPECT_EQ(dda::common::divRoundUp(7, 8), 1u);
    EXPECT_EQ(dda::common::divRoundUp(8, 8), 1u);
    EXPECT_EQ(dda::common::divRoundUp(9, 8), 2u);
}

TEST(DdaCollCommon, calcBlockCount)
{
    EXPECT_EQ(dda::common::calcBlockCount(0, 512, 24), 1u);
    EXPECT_EQ(dda::common::calcBlockCount(512, 512, 24), 1u);
    EXPECT_EQ(dda::common::calcBlockCount(513, 512, 24), 2u);
    EXPECT_EQ(dda::common::calcBlockCount(12288, 512, 24), 24u);
    EXPECT_EQ(dda::common::calcBlockCount(20000, 512, 24), 24u);
}

TEST(DdaCollCommon, getGridAndBlockDims_SmallCount)
{
    constexpr int   kTypeSize  = static_cast<int>(sizeof(float));
    constexpr size_t kMaxBlocks = 24;
    constexpr size_t kCount     = 4;

    auto gridBlock =
        dda::common::getGridAndBlockDims(kCount, kTypeSize, kMaxBlocks);

    EXPECT_EQ(gridBlock.first.x, 1u);
    EXPECT_EQ(gridBlock.second.x, 64u);
}

TEST(DdaCollCommon, getGridAndBlockDims_LargeCount)
{
    constexpr int    kTypeSize  = static_cast<int>(sizeof(float));
    constexpr size_t kMaxBlocks = 24;
    constexpr size_t kCount     = 4096;

    auto gridBlock =
        dda::common::getGridAndBlockDims(kCount, kTypeSize, kMaxBlocks);

    EXPECT_EQ(gridBlock.first.x, 2u);
    EXPECT_EQ(gridBlock.second.x, 512u);
}

TEST(DdaCollCommon, getGridAndBlockDims_CapsBlocksAtMax)
{
    constexpr int    kTypeSize  = static_cast<int>(sizeof(float));
    constexpr size_t kMaxBlocks = 24;
    constexpr size_t kCount     = 100000;

    auto gridBlock =
        dda::common::getGridAndBlockDims(kCount, kTypeSize, kMaxBlocks);

    EXPECT_EQ(gridBlock.first.x, kMaxBlocks);
    EXPECT_EQ(gridBlock.second.x, 512u);
}

TEST(DdaCollCommon, getGridAndBlockDims_HalfPrecision)
{
    constexpr int    kTypeSize  = static_cast<int>(sizeof(uint16_t));
    constexpr size_t kMaxBlocks = 24;
    constexpr size_t kCount     = 1024;

    auto gridBlock =
        dda::common::getGridAndBlockDims(kCount, kTypeSize, kMaxBlocks);

    EXPECT_GE(gridBlock.first.x, 1u);
    EXPECT_LE(gridBlock.first.x, kMaxBlocks);
    EXPECT_GE(gridBlock.second.x, 64u);
    EXPECT_LE(gridBlock.second.x, 512u);
}

} // namespace RcclUnitTesting
