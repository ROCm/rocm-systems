/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MULTI_SEGMENT_VMM_HELPERS_HPP
#define MULTI_SEGMENT_VMM_HELPERS_HPP

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "rocmwrap.h"

// hipMemLocationTypeHost is undeclared without the CMake host probe; outside the gate the shim is harmless (HOST == 2).
#if !NCCL_CUMEM_HOST_GATE
#ifndef hipMemLocationTypeHost
#define hipMemLocationTypeHost (static_cast<hipMemLocationType>(2))
#endif
#endif

// Shared by hybrid, RMA and CE tests. Not gated on a transport backend.
namespace RCCLTestHelpers
{

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

// Pinned, fd-exportable VMM allocation properties at one location.
inline hipMemAllocationProp MakeVmmProp(hipMemLocationType locType, int locId, bool gpuDirectRdma)
{
    hipMemAllocationProp prop            = {};
    prop.type                            = hipMemAllocationTypePinned;
    prop.location.type                   = locType;
    prop.location.id                     = locId;
    prop.requestedHandleType             = hipMemHandleTypePosixFileDescriptor;
    prop.allocFlags.gpuDirectRDMACapable = gpuDirectRdma ? 1 : 0;
    return prop;
}

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

// Map one granularity-rounded segment per `segBytes` entry contiguously on `dev`.
// `segSize` is the common size, or 0 when the sizes differ. Returns false
// (cleaned up) on any HIP failure so the caller can GTEST_SKIP.
inline bool AllocSegmentedVmm(int dev, const std::vector<size_t>& segBytes,
                              MultiSegmentVmmBuffer* out)
{
    if (out == nullptr || segBytes.empty()) return false;
    *out = MultiSegmentVmmBuffer{};
    const int nSegments = static_cast<int>(segBytes.size());

    hipMemAllocationProp prop = MakeVmmProp(hipMemLocationTypeDevice, dev, /*gpuDirectRdma=*/true);

    size_t granularity = 0;
    if (hipMemGetAllocationGranularity(&granularity, &prop,
                                       hipMemAllocationGranularityMinimum) != hipSuccess
        || granularity == 0)
    {
        return false;
    }

    std::vector<size_t> segSizes;
    segSizes.reserve(nSegments);
    size_t totalSize = 0;
    for (size_t bytes : segBytes)
    {
        segSizes.push_back(((bytes + granularity - 1) / granularity) * granularity);
        totalSize += segSizes.back();
    }
    size_t segSize = segSizes[0];
    for (size_t len : segSizes)
        if (len != segSize) segSize = 0;

    hipDeviceptr_t base = 0;
    if (hipMemAddressReserve(&base, totalSize, granularity, 0, 0) != hipSuccess)
    {
        return false;
    }

    std::vector<hipMemGenericAllocationHandle_t> handles;
    handles.reserve(nSegments);

    auto cleanup = [&]() {
        ReleaseMappedVmm(base, totalSize, handles, segSizes, /*uniformSegSize=*/0);
    };

    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type    = hipMemLocationTypeDevice;
    accessDesc.location.id      = dev;
    accessDesc.flags            = hipMemAccessFlagsProtReadWrite;

    size_t off = 0;
    for (int s = 0; s < nSegments; ++s)
    {
        hipMemGenericAllocationHandle_t h = 0;
        if (hipMemCreate(&h, segSizes[s], &prop, 0) != hipSuccess)
        {
            cleanup();
            return false;
        }
        handles.push_back(h);

        hipDeviceptr_t segVa = reinterpret_cast<hipDeviceptr_t>(
            reinterpret_cast<uintptr_t>(base) + off);
        if (hipMemMap(segVa, segSizes[s], 0, h, 0) != hipSuccess)
        {
            cleanup();
            return false;
        }
        off += segSizes[s];
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
    out->segSizes  = std::move(segSizes);
    return true;
}

// Map `nSegments` granularity-rounded segments of `segBytes` each contiguously on `dev`.
inline bool AllocMultiSegmentVmm(int dev, int nSegments, size_t segBytes,
                                 MultiSegmentVmmBuffer* out)
{
    if (nSegments <= 0) return false;
    return AllocSegmentedVmm(dev, std::vector<size_t>(nSegments, segBytes), out);
}

// Release in HIP-required order: unmap each segment, release each handle, free VA.
inline void FreeMultiSegmentVmm(MultiSegmentVmmBuffer& b)
{
    if (b.ptr == nullptr && b.base == 0) return;
    ReleaseMappedVmm(b.base, b.totalSize, b.handles, b.segSizes, b.segSize);
    b = MultiSegmentVmmBuffer{};
}

} // namespace RCCLTestHelpers

#endif // MULTI_SEGMENT_VMM_HELPERS_HPP
