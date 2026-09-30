/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * GPU multi-segment and DeepEP [GPU][CPU] windows for collective tests that
 * do not link the RMA plugin.
 */

#ifndef RCCL_TEST_COMMON_MULTISEGMENT_VMM_HELPERS_HPP
#define RCCL_TEST_COMMON_MULTISEGMENT_VMM_HELPERS_HPP

#ifdef MPI_TESTS_ENABLED

#include "HybridVmmHelpers.hpp"

#include <hip/hip_runtime.h>
#include <utility>
#include <vector>

namespace RCCLMultiSegmentTests
{

struct VmmWindow
{
    void*                                        ptr       = nullptr;
    hipDeviceptr_t                               base      = 0;
    size_t                                       totalSize = 0;
    int                                          nSegments = 0;
    std::vector<hipMemGenericAllocationHandle_t> handles;
    std::vector<size_t>                          segSizes;
};

inline void FreeVmmWindow(VmmWindow& b)
{
    if (b.base != 0)
    {
        hipDeviceptr_t off = b.base;
        for (size_t i = 0; i < b.handles.size(); ++i)
        {
            const size_t bytes = i < b.segSizes.size() ? b.segSizes[i] : 0;
            if (bytes != 0)
                hipMemUnmap(off, bytes);
            if (b.handles[i] != 0)
                hipMemRelease(b.handles[i]);
            off += bytes;
        }
        hipMemAddressFree(b.base, b.totalSize);
    }
    b = VmmWindow{};
}

inline bool AllocUniformGpu(int dev, int nSegments, size_t segBytes, VmmWindow* out)
{
    constexpr size_t kAlign = 2u * 1024 * 1024;
    if (out == nullptr || nSegments < 2 || segBytes == 0 || segBytes % kAlign != 0)
        return false;
    *out = VmmWindow{};

    hipMemAllocationProp prop = {};
    prop.type                            = hipMemAllocationTypePinned;
    prop.location.type                   = hipMemLocationTypeDevice;
    prop.location.id                     = dev;
    prop.requestedHandleType             = hipMemHandleTypePosixFileDescriptor;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    size_t granularity = 0;
    if (hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum) != hipSuccess ||
        granularity == 0 || segBytes % granularity != 0)
        return false;

    const size_t total = segBytes * static_cast<size_t>(nSegments);
    hipDeviceptr_t base = 0;
    if (hipMemAddressReserve(&base, total, kAlign, 0, 0) != hipSuccess)
        return false;

    VmmWindow window;
    window.base      = base;
    window.totalSize = total;
    window.nSegments = nSegments;
    window.handles.reserve(nSegments);
    window.segSizes.assign(nSegments, segBytes);

    for (int i = 0; i < nSegments; ++i)
    {
        hipMemGenericAllocationHandle_t handle = 0;
        if (hipMemCreate(&handle, segBytes, &prop, 0) != hipSuccess ||
            hipMemMap(base + static_cast<hipDeviceptr_t>(i * segBytes), segBytes, 0, handle, 0) != hipSuccess)
        {
            if (handle != 0)
                hipMemRelease(handle);
            FreeVmmWindow(window);
            return false;
        }
        window.handles.push_back(handle);
    }

    hipMemAccessDesc access = {};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id   = dev;
    access.flags         = hipMemAccessFlagsProtReadWrite;
    if (hipMemSetAccess(base, total, &access, 1) != hipSuccess)
    {
        FreeVmmWindow(window);
        return false;
    }
    window.ptr = reinterpret_cast<void*>(base);
    *out       = std::move(window);
    return true;
}

inline bool AllocElastic(int dev, size_t gpuBytes, size_t cpuBytes, VmmWindow* out)
{
    if (out == nullptr)
        return false;
    RCCLHybridVmmTests::DeepEpElasticRange range;
    if (!RCCLHybridVmmTests::AllocDeepEpElasticRange(dev, gpuBytes, cpuBytes, &range))
        return false;
    *out           = VmmWindow{};
    out->ptr       = reinterpret_cast<void*>(range.base);
    out->base      = range.base;
    out->totalSize = range.totalSize;
    out->nSegments = static_cast<int>(range.handles.size());
    out->handles   = std::move(range.handles);
    out->segSizes  = std::move(range.segSizes);
    return true;
}

} // namespace RCCLMultiSegmentTests

#endif // MPI_TESTS_ENABLED
#endif
