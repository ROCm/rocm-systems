/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MULTI_SEGMENT_TEST_SUPPORT_HPP
#define MULTI_SEGMENT_TEST_SUPPORT_HPP

#ifdef MPI_TESTS_ENABLED

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "MPIHelpers.hpp"
#include "MultiSegmentVmmHelpers.hpp"
#include "SegPattern.hpp"

// Shared by the NET and RMA multi-segment MPI suites.
namespace RCCLTestHelpers
{

// A suite's prerequisite policy: every rank skips together, and with `requireEnv`=1 (CI) the skip also fails.
struct MsegPrereq
{
    const char* requireEnv;

    bool Required() const { return MPIHelpers::getEnvParam<int>(requireEnv, 0) != 0; }

    // Collective: true on every rank when `missing` holds on any rank.
    bool Skip(bool missing, const char* what) const
    {
        const bool any = MPIHelpers::anyRankTrue(missing);
        if (any) FailIfRequired(what);
        return any;
    }

    void FailIfRequired(const char* what) const
    {
        if (Required()) ADD_FAILURE() << what << " with " << requireEnv << "=1";
    }
};

// VMM windows on the current GPU, freed by Release(). Release only after every MR over them is deregistered.
class OwnedVmmPool
{
public:
    ~OwnedVmmPool() { Release(); }

    MultiSegmentVmmBuffer* Alloc(const std::vector<VmmSegSpec>& specs, size_t alignment = 0)
    {
        return Keep([&](int dev, MultiSegmentVmmBuffer* b) { return AllocSegmentedVmm(dev, specs, b, alignment); });
    }
    MultiSegmentVmmBuffer* AllocLayout(const std::vector<size_t>& segBytes)
    {
        return Keep([&](int dev, MultiSegmentVmmBuffer* b) { return AllocSegmentedVmm(dev, segBytes, b); });
    }
    MultiSegmentVmmBuffer* AllocUniform(int nSegments, size_t segBytes)
    {
        return Keep([&](int dev, MultiSegmentVmmBuffer* b) { return AllocMultiSegmentVmm(dev, nSegments, segBytes, b); });
    }
    MultiSegmentVmmBuffer* AllocDeepEpElastic(size_t gpuBytes, size_t cpuBytes)
    {
        return Keep([&](int dev, MultiSegmentVmmBuffer* b) { return AllocDeepEpElasticVmm(dev, gpuBytes, cpuBytes, b); });
    }

    void Release()
    {
        for (auto& b : bufs_) FreeMultiSegmentVmm(*b);
        bufs_.clear();
    }

private:
    template <typename AllocFn>
    MultiSegmentVmmBuffer* Keep(AllocFn alloc)
    {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) return nullptr;
        auto b = std::make_unique<MultiSegmentVmmBuffer>();
        if (!alloc(dev, b.get())) return nullptr;
        bufs_.push_back(std::move(b));
        return bufs_.back().get();
    }

    std::vector<std::unique_ptr<MultiSegmentVmmBuffer>> bufs_;
};

// One iteration of a two-sided boundary stress over nSeg segments of seg bytes: the source crosses boundary
// 1 + i % (nSeg - 1) and the destination 1 + (i * dstMul + dstAdd) % (nSeg - 1), each by the same edge widths.
struct BoundaryStressCase
{
    size_t srcOff, dstOff, size;
};
inline BoundaryStressCase BoundaryStress(int i, int nSeg, size_t seg, int dstMul, int dstAdd, int rightShift)
{
    static constexpr size_t kEdgeWidths[] = {1, 63, 4095, 65535, 131071};
    constexpr size_t kNumEdges = sizeof(kEdgeWidths) / sizeof(kEdgeWidths[0]);
    const size_t left = kEdgeWidths[static_cast<size_t>(i) % kNumEdges];
    const size_t right = kEdgeWidths[static_cast<size_t>(i + rightShift) % kNumEdges];
    const size_t srcBoundary = 1 + static_cast<size_t>(i % (nSeg - 1));
    const size_t dstBoundary = 1 + static_cast<size_t>((i * dstMul + dstAdd) % (nSeg - 1));
    return {srcBoundary * seg - left, dstBoundary * seg - left, left + right};
}

// Index of the segment of b holding byte off; the last segment for an offset past the end.
inline int SegmentIndexOf(const MultiSegmentVmmBuffer& b, size_t off)
{
    size_t end = 0;
    for (int s = 0; s < b.nSegments; s++)
    {
        end += b.segSizes.empty() ? b.segSize : b.segSizes[s];
        if (off < end) return s;
    }
    return b.nSegments - 1;
}

// Whether a transfer of size bytes from src[srcOff] to dst[dstOff] reaches a segment past the first on either side.
inline bool ReachesLaterSegment(const MultiSegmentVmmBuffer& src, size_t srcOff, const MultiSegmentVmmBuffer& dst,
                                size_t dstOff, size_t size)
{
    return size > 0 && (SegmentIndexOf(src, srcOff + size - 1) > 0 || SegmentIndexOf(dst, dstOff + size - 1) > 0);
}

// Writes SegPatternByte(seed, i) into device or host-mapped buf[0, size).
inline void FillSegPattern(void* buf, size_t size, int seed)
{
    std::vector<uint8_t> tmp(size);
    for (size_t i = 0; i < size; ++i) tmp[i] = SegPatternByte(seed, i);
    ASSERT_EQ(hipSuccess, hipMemcpy(buf, tmp.data(), size, hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
}

// Checks buf[0, size) holds the pattern from `seed`; on a mismatch reports the first offset and the mismatch count.
inline ::testing::AssertionResult VerifySegPattern(const void* buf, size_t size, int seed)
{
    std::vector<uint8_t> got(size);
    if (hipMemcpy(got.data(), buf, size, hipMemcpyDeviceToHost) != hipSuccess || hipDeviceSynchronize() != hipSuccess)
        return ::testing::AssertionFailure() << "hipMemcpy D2H failed";
    size_t nBad = 0, first = 0;
    for (size_t i = 0; i < size; ++i)
    {
        if (got[i] == SegPatternByte(seed, i)) continue;
        if (nBad++ == 0) first = i;
    }
    if (nBad == 0) return ::testing::AssertionSuccess();
    char detail[96];
    std::snprintf(detail, sizeof(detail), "want 0x%02x got 0x%02x", SegPatternByte(seed, first), got[first]);
    return ::testing::AssertionFailure() << nBad << " of " << size << " bytes differ (seed " << seed
                                         << "), first at offset " << first << ": " << detail;
}

// Checks buf[0, total) is `sentinel` outside [off, off + size).
inline ::testing::AssertionResult VerifySentinelAround(const void* buf, size_t total, size_t off, size_t size,
                                                       uint8_t sentinel)
{
    std::vector<uint8_t> got(total);
    if (hipMemcpy(got.data(), buf, total, hipMemcpyDeviceToHost) != hipSuccess || hipDeviceSynchronize() != hipSuccess)
        return ::testing::AssertionFailure() << "hipMemcpy D2H failed";
    for (size_t i = 0; i < total; ++i)
    {
        if (i >= off && i < off + size) continue;
        if (got[i] != sentinel)
            return ::testing::AssertionFailure() << "byte " << i << " outside [" << off << ", " << off + size
                                                 << ") was overwritten";
    }
    return ::testing::AssertionSuccess();
}

} // namespace RCCLTestHelpers

#endif // MPI_TESTS_ENABLED

#endif // MULTI_SEGMENT_TEST_SUPPORT_HPP
