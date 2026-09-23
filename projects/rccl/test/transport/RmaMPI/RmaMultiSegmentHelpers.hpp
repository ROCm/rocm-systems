/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RMA_MULTI_SEGMENT_HELPERS_HPP
#define RMA_MULTI_SEGMENT_HELPERS_HPP

#ifdef MPI_TESTS_ENABLED
#ifdef RCCL_HAS_RMA_IB_PROXY

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace RCCLRmaTests
{

// Contiguous VA range backed by N distinct VMM allocations mapped back-to-back.
struct MultiSegmentVmmBuffer
{
    void*                                        ptr        = nullptr;
    size_t                                       totalSize  = 0;
    size_t                                       segSize    = 0;
    int                                          nSegments  = 0;
    hipDeviceptr_t                               base       = 0;
    std::vector<hipMemGenericAllocationHandle_t> handles;
    std::vector<size_t>                          segSizes;

    bool valid() const { return ptr != nullptr && nSegments > 0; }
};

// Unmap, release, and free a reserved VA. `segSizes` wins when non-empty;
// otherwise each mapped handle is `uniformSegSize` bytes.
inline void ReleaseMappedVmm(hipDeviceptr_t base, size_t totalSize,
                             const std::vector<hipMemGenericAllocationHandle_t>& handles,
                             const std::vector<size_t>& segSizes, size_t uniformSegSize)
{
    size_t off = 0;
    for (size_t i = 0; i < handles.size(); ++i)
    {
        const size_t len = segSizes.empty() ? uniformSegSize : segSizes[i];
        if (base != 0 && len != 0)
        {
            hipDeviceptr_t segVa = reinterpret_cast<hipDeviceptr_t>(
                reinterpret_cast<uintptr_t>(base) + off);
            (void)hipMemUnmap(segVa, len);
        }
        if (handles[i] != 0) (void)hipMemRelease(handles[i]);
        off += len;
    }
    if (base != 0 && totalSize != 0) (void)hipMemAddressFree(base, totalSize);
}

// Map `nSegments` granularity-rounded segments contiguously on `dev`. Returns
// false (cleaned up) on any HIP failure so the caller can GTEST_SKIP.
inline bool AllocMultiSegmentVmm(int dev, int nSegments, size_t segBytes,
                                 MultiSegmentVmmBuffer* out)
{
    if (out == nullptr || nSegments <= 0) return false;
    *out = MultiSegmentVmmBuffer{};

    hipMemAllocationProp prop            = {};
    prop.type                            = hipMemAllocationTypePinned;
    prop.location.type                   = hipMemLocationTypeDevice;
    prop.location.id                     = dev;
    prop.requestedHandleType             = hipMemHandleTypePosixFileDescriptor;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    size_t granularity = 0;
    if (hipMemGetAllocationGranularity(&granularity, &prop,
                                       hipMemAllocationGranularityMinimum) != hipSuccess
        || granularity == 0)
    {
        return false;
    }

    const size_t segSize   = ((segBytes + granularity - 1) / granularity) * granularity;
    const size_t totalSize = segSize * static_cast<size_t>(nSegments);

    hipDeviceptr_t base = 0;
    if (hipMemAddressReserve(&base, totalSize, granularity, 0, 0) != hipSuccess)
    {
        return false;
    }

    std::vector<hipMemGenericAllocationHandle_t> handles;
    handles.reserve(nSegments);

    auto cleanup = [&]() {
        ReleaseMappedVmm(base, totalSize, handles, /*segSizes=*/{}, segSize);
    };

    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type    = hipMemLocationTypeDevice;
    accessDesc.location.id      = dev;
    accessDesc.flags            = hipMemAccessFlagsProtReadWrite;

    for (int s = 0; s < nSegments; ++s)
    {
        hipMemGenericAllocationHandle_t h = 0;
        if (hipMemCreate(&h, segSize, &prop, 0) != hipSuccess)
        {
            cleanup();
            return false;
        }
        handles.push_back(h);

        hipDeviceptr_t segVa = reinterpret_cast<hipDeviceptr_t>(
            reinterpret_cast<uintptr_t>(base) + static_cast<uintptr_t>(s) * segSize);
        if (hipMemMap(segVa, segSize, 0, h, 0) != hipSuccess)
        {
            cleanup();
            return false;
        }
    }

    // One access grant over the whole contiguous range.
    if (hipMemSetAccess(base, totalSize, &accessDesc, 1) != hipSuccess)
    {
        cleanup();
        return false;
    }

    out->ptr       = reinterpret_cast<void*>(base);
    out->base      = base;
    out->totalSize = totalSize;
    out->segSize   = segSize;
    out->nSegments = nSegments;
    out->handles   = std::move(handles);
    out->segSizes.assign(nSegments, segSize);
    return true;
}

// AllocDeepEpElasticVmm lives in HybridVmmHelpers.hpp so UBR and RMA share one
// HIP sequence. Include that header after this one; it fills MultiSegmentVmmBuffer.

// Release in HIP-required order: unmap each segment, release each handle, free VA.
inline void FreeMultiSegmentVmm(MultiSegmentVmmBuffer& b)
{
    if (b.ptr == nullptr && b.base == 0) return;
    ReleaseMappedVmm(b.base, b.totalSize, b.handles, b.segSizes, b.segSize);
    b = MultiSegmentVmmBuffer{};
}

} // namespace RCCLRmaTests

#endif // RCCL_HAS_RMA_IB_PROXY
#endif // MPI_TESTS_ENABLED

#endif // RMA_MULTI_SEGMENT_HELPERS_HPP
