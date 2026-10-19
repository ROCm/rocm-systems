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
#include <string>
#include <type_traits>
#include <vector>

namespace RcclUnitTesting
{

namespace
{
using dda::common::fp8e4m3;
using dda::common::fp8e5m2;

constexpr uint32_t kFp8Pairs = 256 * 256;

// Kernels below take only the tag: rccl_float8 / rccl_bfloat8 name different
// types in the host and gfx942 device passes, so a kernel templated on them
// would register under a name the gfx942 code object lacks.
template <typename Tag>
using Fp8Of = typename std::conditional<std::is_same<Tag, fp8e4m3>::value, rccl_float8, rccl_bfloat8>::type;

struct Fp8AddMismatches
{
    uint32_t llPayload;
    uint32_t ll128Word;
    uint32_t vec16B;
    uint32_t firstBadPair;
};

// RCCL's core fp8 pair add (hadd2 / hadd2_b), which the reduce kernels use below
// eight lanes. It converts two lanes per instruction, so it shares no
// conversion with the pk8 add DDA uses. The pair's second element is zero.
template <typename Tag>
__device__ uint8_t coreFp8Add(uint8_t a, uint8_t b)
{
    if constexpr (std::is_same<Tag, fp8e4m3>::value)
        return static_cast<uint8_t>(hadd2(a, b) & 0xff);
    else
        return static_cast<uint8_t>(hadd2_b(a, b) & 0xff);
}

// Places pair p in each of the Lanes byte lanes in turn, with unrelated pairs in
// the other lanes so a lane that leaks into another shows up too, and counts
// lanes where `add` disagrees with coreFp8Add.
template <typename Tag, int Lanes, typename Add>
__device__ void checkFp8Lanes(uint32_t p, uint32_t salt, Add add, uint32_t* mismatches, uint32_t* firstBadPair)
{
    for (int lane = 0; lane < Lanes; ++lane)
    {
        uint8_t a[Lanes], b[Lanes], expected[Lanes], sum[Lanes];
        for (int k = 0; k < Lanes; ++k)
        {
            const uint32_t q = (k == lane) ? p : ((p * 2654435761u + k * salt) & 0xffff);
            a[k]             = q & 0xff;
            b[k]             = q >> 8;
            expected[k]      = coreFp8Add<Tag>(a[k], b[k]);
        }
        add(a, b, sum);
        for (int k = 0; k < Lanes; ++k)
        {
            if (sum[k] != expected[k])
            {
                atomicAdd(mismatches, 1u);
                atomicCAS(firstBadPair, 0xffffffffu, p);
            }
        }
    }
}

// One thread per (a, b) byte pair, run through each width the DDA kernels add
// fp8 at: an LL packet's two payload dwords, an LL128 word, and a 16B vector.
template <typename Tag>
__global__ void fp8AddMatchesCoreKernel(Fp8AddMismatches* out)
{
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= kFp8Pairs)
        return;

    checkFp8Lanes<Tag, 8>(
        p, 40503u,
        [](const uint8_t* a, const uint8_t* b, uint8_t* sum) {
            uint32_t acc[2], d[2];
            __builtin_memcpy(acc, a, 8);
            __builtin_memcpy(d, b, 8);
            dda::common::ddaLLAddPayload<Tag>(acc[0], acc[1], d[0], d[1]);
            __builtin_memcpy(sum, acc, 8);
        },
        &out->llPayload, &out->firstBadPair);

    checkFp8Lanes<Tag, 8>(
        p, 3266489917u,
        [](const uint8_t* a, const uint8_t* b, uint8_t* sum) {
            uint64_t wa, wb;
            __builtin_memcpy(&wa, a, 8);
            __builtin_memcpy(&wb, b, 8);
            const uint64_t s = dda::common::ddaLL128AddWord<Tag>(wa, wb);
            __builtin_memcpy(sum, &s, 8);
        },
        &out->ll128Word, &out->firstBadPair);

    checkFp8Lanes<Tag, 16>(
        p, 2246822519u,
        [](const uint8_t* a, const uint8_t* b, uint8_t* sum) {
            uint4 wa, wb;
            __builtin_memcpy(&wa, a, 16);
            __builtin_memcpy(&wb, b, 16);
            const uint4 s = dda::common::vecElementAdd<Tag>(wa, wb);
            __builtin_memcpy(sum, &s, 16);
        },
        &out->vec16B, &out->firstBadPair);
}

constexpr int kFp8ExactLanes = 8;

// Adds a[i] + b[i] across one 8-byte word, converting to and from the device's
// fp8 encoding in device code.
template <typename Tag>
__global__ void fp8AddFloatsKernel(const float* a, const float* b, float* out)
{
    using Fp8   = Fp8Of<Tag>;
    uint64_t wa = 0, wb = 0;
    for (int k = 0; k < kFp8ExactLanes; ++k)
    {
        const Fp8 fa(a[k]), fb(b[k]);
        uint8_t   ba, bb;
        __builtin_memcpy(&ba, &fa, 1);
        __builtin_memcpy(&bb, &fb, 1);
        wa |= static_cast<uint64_t>(ba) << (8 * k);
        wb |= static_cast<uint64_t>(bb) << (8 * k);
    }
    const uint64_t sum = dda::common::vecElementAddFp8x8<Tag>(wa, wb);
    for (int k = 0; k < kFp8ExactLanes; ++k)
    {
        const uint8_t bits = (sum >> (8 * k)) & 0xff;
        Fp8           v;
        __builtin_memcpy(&v, &bits, 1);
        out[k] = static_cast<float>(v);
    }
}

// DDA admits fp8 only on gfx1250; elsewhere the fp8 add traps.
const char* fp8SkipReason()
{
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0)
        return "No GPU";
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) != hipSuccess
        || std::string(props.gcnArchName).find("gfx1250") == std::string::npos)
        return "DDA fp8 requires gfx1250";
    return nullptr;
}

template <typename Tag>
void expectFp8AddMatchesCore()
{
    Fp8AddMismatches* d = nullptr;
    ASSERT_EQ(hipMalloc(&d, sizeof(*d)), hipSuccess);
    const Fp8AddMismatches init{0, 0, 0, 0xffffffffu};
    ASSERT_EQ(hipMemcpy(d, &init, sizeof(init), hipMemcpyHostToDevice), hipSuccess);
    hipLaunchKernelGGL(fp8AddMatchesCoreKernel<Tag>, dim3(kFp8Pairs / 256), dim3(256), 0, 0, d);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    Fp8AddMismatches h{};
    ASSERT_EQ(hipMemcpy(&h, d, sizeof(h), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(hipFree(d), hipSuccess);

    EXPECT_EQ(h.llPayload, 0u) << "first mismatching (a, b) pair 0x" << std::hex << h.firstBadPair;
    EXPECT_EQ(h.ll128Word, 0u) << "first mismatching (a, b) pair 0x" << std::hex << h.firstBadPair;
    EXPECT_EQ(h.vec16B, 0u) << "first mismatching (a, b) pair 0x" << std::hex << h.firstBadPair;
}

// Sums every fp8 format represents exactly, so the result does not depend on
// rounding; one per lane, so a lane swap shows up as a wrong value.
template <typename Tag>
void expectFp8AddExactSums()
{
    const std::vector<float> a        = {1.0f, 0.5f, -1.5f, 8.0f, 0.25f, -2.0f, 3.0f, 0.0f};
    const std::vector<float> b        = {2.0f, 0.25f, 0.5f, 8.0f, 0.25f, -2.0f, -3.0f, 0.0f};
    const std::vector<float> expected = {3.0f, 0.75f, -1.0f, 16.0f, 0.5f, -4.0f, 0.0f, 0.0f};
    constexpr size_t         n        = kFp8ExactLanes;

    float* d = nullptr;
    ASSERT_EQ(hipMalloc(&d, 3 * n * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemcpy(d, a.data(), n * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d + n, b.data(), n * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    hipLaunchKernelGGL(fp8AddFloatsKernel<Tag>, dim3(1), dim3(1), 0, 0, d, d + n, d + 2 * n);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    std::vector<float> got(n);
    ASSERT_EQ(hipMemcpy(got.data(), d + 2 * n, n * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(hipFree(d), hipSuccess);

    for (size_t k = 0; k < n; ++k)
        EXPECT_EQ(got[k], expected[k]) << a[k] << " + " << b[k] << " in lane " << k;
}
} // namespace

// The DDA reduce kernels add fp8 through ddaLLAddPayload, ddaLL128AddWord and
// vecElementAdd, which must round exactly as the core reduce kernels do, or DDA
// and non-DDA collectives would disagree on the same inputs.
TEST(DdaCollCommon, Fp8E4M3AddMatchesCoreForAllPairs)
{
    if (const char* reason = fp8SkipReason())
        GTEST_SKIP() << reason;
    expectFp8AddMatchesCore<fp8e4m3>();
}

TEST(DdaCollCommon, Fp8E5M2AddMatchesCoreForAllPairs)
{
    if (const char* reason = fp8SkipReason())
        GTEST_SKIP() << reason;
    expectFp8AddMatchesCore<fp8e5m2>();
}

TEST(DdaCollCommon, Fp8E4M3AddExactSums)
{
    if (const char* reason = fp8SkipReason())
        GTEST_SKIP() << reason;
    expectFp8AddExactSums<fp8e4m3>();
}

TEST(DdaCollCommon, Fp8E5M2AddExactSums)
{
    if (const char* reason = fp8SkipReason())
        GTEST_SKIP() << reason;
    expectFp8AddExactSums<fp8e5m2>();
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
