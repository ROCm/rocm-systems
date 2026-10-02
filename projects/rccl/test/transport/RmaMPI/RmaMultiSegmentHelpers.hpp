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

#include "MultiSegmentVmmHelpers.hpp"

namespace RCCLRmaTests
{

using RCCLTestHelpers::AllocMultiSegmentVmm;
using RCCLTestHelpers::FreeMultiSegmentVmm;
using RCCLTestHelpers::MultiSegmentVmmBuffer;
using RCCLTestHelpers::ReleaseMappedVmm;

// Reproduce DeepEP's non-hybrid ElasticSymmetricMemory layout:
// one 2 MiB-aligned GPU segment followed by one independently-sized CPU segment
// in a single reserved VA range. DeepEP registers the whole range as one NCCL
// symmetric window and uses RMA IGet from the CPU segment into the GPU segment.
inline bool AllocDeepEpElasticVmm(int dev, size_t gpuBytes, size_t cpuBytes,
                                  MultiSegmentVmmBuffer* out)
{
    constexpr size_t kDeepEpAlignment = 2u * 1024 * 1024;
    if (out == nullptr || gpuBytes == 0 || cpuBytes == 0 ||
        gpuBytes % kDeepEpAlignment != 0 || cpuBytes % kDeepEpAlignment != 0)
        return false;
    *out = MultiSegmentVmmBuffer{};

    hipMemAllocationProp gpuProp            = {};
    gpuProp.type                            = hipMemAllocationTypePinned;
    gpuProp.location.type                   = hipMemLocationTypeDevice;
    gpuProp.location.id                     = dev;
    gpuProp.requestedHandleType             = hipMemHandleTypePosixFileDescriptor;
    gpuProp.allocFlags.gpuDirectRDMACapable = 1;

    hipMemAllocationProp cpuProp = {};
    cpuProp.type                    = hipMemAllocationTypePinned;
    cpuProp.location.type           = hipMemLocationTypeHost;
    cpuProp.location.id             = 0;
    cpuProp.requestedHandleType     = hipMemHandleTypePosixFileDescriptor;

    size_t gpuGran = 0, cpuGran = 0;
    if (hipMemGetAllocationGranularity(&gpuGran, &gpuProp, hipMemAllocationGranularityMinimum) != hipSuccess ||
        hipMemGetAllocationGranularity(&cpuGran, &cpuProp, hipMemAllocationGranularityMinimum) != hipSuccess ||
        gpuGran == 0 || cpuGran == 0 ||
        kDeepEpAlignment % gpuGran != 0 || kDeepEpAlignment % cpuGran != 0)
        return false;

    const size_t totalSize = gpuBytes + cpuBytes;
    hipDeviceptr_t base = 0;
    if (hipMemAddressReserve(&base, totalSize, kDeepEpAlignment, 0, 0) != hipSuccess)
        return false;

    std::vector<hipMemGenericAllocationHandle_t> handles;
    std::vector<size_t> segSizes = {gpuBytes, cpuBytes};
    auto cleanup = [&]() {
        ReleaseMappedVmm(base, totalSize, handles, segSizes, /*uniformSegSize=*/0);
    };

    for (int s = 0; s < 2; ++s) {
        hipMemGenericAllocationHandle_t h = 0;
        hipMemAllocationProp* prop = s == 0 ? &gpuProp : &cpuProp;
        if (hipMemCreate(&h, segSizes[s], prop, 0) != hipSuccess) {
            cleanup();
            return false;
        }
        handles.push_back(h);
        const size_t off = s == 0 ? 0 : gpuBytes;
        hipDeviceptr_t segVa = reinterpret_cast<hipDeviceptr_t>(
            reinterpret_cast<uintptr_t>(base) + off);
        if (hipMemMap(segVa, segSizes[s], 0, h, 0) != hipSuccess) {
            cleanup();
            return false;
        }
    }

    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type    = hipMemLocationTypeDevice;
    accessDesc.location.id      = dev;
    accessDesc.flags            = hipMemAccessFlagsProtReadWrite;
    if (hipMemSetAccess(base, gpuBytes, &accessDesc, 1) != hipSuccess ||
        hipMemSetAccess(reinterpret_cast<hipDeviceptr_t>(
                            reinterpret_cast<uintptr_t>(base) + gpuBytes),
                        cpuBytes, &accessDesc, 1) != hipSuccess) {
        cleanup();
        return false;
    }

    out->ptr       = reinterpret_cast<void*>(base);
    out->base      = base;
    out->totalSize = totalSize;
    out->segSize   = 0; // non-uniform by design
    out->nSegments = 2;
    out->handles   = std::move(handles);
    out->segSizes  = std::move(segSizes);
    return true;
}

} // namespace RCCLRmaTests

#endif // RCCL_HAS_RMA_IB_PROXY
#endif // MPI_TESTS_ENABLED

#endif // RMA_MULTI_SEGMENT_HELPERS_HPP
