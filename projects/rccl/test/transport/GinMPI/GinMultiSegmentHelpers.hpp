/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#ifdef MPI_TESTS_ENABLED
#ifdef RCCL_HAS_GIN_IB_PROXY

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace RCCLGinTests
{

// Contiguous VA range backed by N distinct VMM allocations mapped back-to-back
// (the AIRUNTIME-2351 shape). See docs/dev/gin-multi-segment-dmabuf.md.
struct MultiSegmentVmmBuffer
{
    void*                                        ptr        = nullptr;
    size_t                                       totalSize  = 0;
    size_t                                       segSize    = 0;
    int                                          nSegments  = 0;
    hipDeviceptr_t                               base       = 0;
    std::vector<hipMemGenericAllocationHandle_t> handles;

    bool valid() const { return ptr != nullptr && nSegments > 0; }
};

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
        for (size_t i = 0; i < handles.size(); ++i)
        {
            (void)hipMemUnmap(reinterpret_cast<hipDeviceptr_t>(
                                  reinterpret_cast<uintptr_t>(base) + i * segSize),
                              segSize);
            (void)hipMemRelease(handles[i]);
        }
        (void)hipMemAddressFree(base, totalSize);
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
    return true;
}

// Release in HIP-required order: unmap each segment, release each handle, free VA.
inline void FreeMultiSegmentVmm(MultiSegmentVmmBuffer& b)
{
    if (b.ptr == nullptr) return;
    for (size_t i = 0; i < b.handles.size(); ++i)
    {
        hipDeviceptr_t segVa = reinterpret_cast<hipDeviceptr_t>(
            reinterpret_cast<uintptr_t>(b.base) + i * b.segSize);
        (void)hipMemUnmap(segVa, b.segSize);
        (void)hipMemRelease(b.handles[i]);
    }
    (void)hipMemAddressFree(b.base, b.totalSize);
    b = MultiSegmentVmmBuffer{};
}

// RAII wrapper so a test body can early-return / ASSERT without leaking VMM.
class MultiSegmentVmmGuard
{
public:
    MultiSegmentVmmGuard() = default;
    explicit MultiSegmentVmmGuard(MultiSegmentVmmBuffer buf) : buf_(std::move(buf)) {}
    ~MultiSegmentVmmGuard() { FreeMultiSegmentVmm(buf_); }

    MultiSegmentVmmGuard(const MultiSegmentVmmGuard&)            = delete;
    MultiSegmentVmmGuard& operator=(const MultiSegmentVmmGuard&) = delete;

    MultiSegmentVmmBuffer&       get()       { return buf_; }
    const MultiSegmentVmmBuffer& get() const { return buf_; }

private:
    MultiSegmentVmmBuffer buf_;
};

} // namespace RCCLGinTests

#endif // RCCL_HAS_GIN_IB_PROXY
#endif // MPI_TESTS_ENABLED
