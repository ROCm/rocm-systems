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
#include <memory>
#include <vector>

#include "MPIHelpers.hpp"
#include "MultiSegmentVmmHelpers.hpp"

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

} // namespace RCCLTestHelpers

#endif // MPI_TESTS_ENABLED

#endif // MULTI_SEGMENT_TEST_SUPPORT_HPP
