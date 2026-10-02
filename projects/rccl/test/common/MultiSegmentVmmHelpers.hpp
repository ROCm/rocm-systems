/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * GPU multi-segment windows for collective tests that do not link the
 * RMA plugin.
 ************************************************************************/

#ifndef RCCL_TEST_COMMON_MULTISEGMENT_VMM_HELPERS_HPP
#define RCCL_TEST_COMMON_MULTISEGMENT_VMM_HELPERS_HPP

#ifdef MPI_TESTS_ENABLED

#include <hip/hip_runtime.h>
#include <cstdint>
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
        auto* off = static_cast<char*>(b.base);
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
    // Reject a size the caller did not align. The RMA helper rounds up instead.
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
        hipDeviceptr_t segVa = reinterpret_cast<hipDeviceptr_t>(
            reinterpret_cast<uintptr_t>(base) + static_cast<uintptr_t>(i) * segBytes);
        if (hipMemCreate(&handle, segBytes, &prop, 0) != hipSuccess ||
            hipMemMap(segVa, segBytes, 0, handle, 0) != hipSuccess)
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

} // namespace RCCLMultiSegmentTests

#endif // MPI_TESTS_ENABLED
#endif
