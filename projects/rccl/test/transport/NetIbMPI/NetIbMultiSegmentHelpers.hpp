/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_MULTI_SEGMENT_HELPERS_HPP
#define NET_IB_MULTI_SEGMENT_HELPERS_HPP

#ifdef MPI_TESTS_ENABLED

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <unistd.h>
#include <vector>

#include "nccl.h"
#include "rccl_ib_multiseg.h"
#include "rocmwrap.h"

namespace RCCLNetIbTests {

// Contiguous VA range backed by N distinct VMM allocations mapped back-to-back.
// Mirrors the GIN test helper but is not gated on the GIN backend.
struct MultiSegmentVmmBuffer {
    void*                                        ptr       = nullptr;
    size_t                                       totalSize = 0;
    size_t                                       segSize   = 0;
    int                                          nSegments = 0;
    hipDeviceptr_t                               base      = 0;
    std::vector<hipMemGenericAllocationHandle_t> handles;
    bool valid() const { return ptr != nullptr && nSegments > 0; }
};

inline bool AllocMultiSegmentVmm(int dev, int nSegments, size_t segBytes, MultiSegmentVmmBuffer* out) {
    if (out == nullptr || nSegments <= 0) return false;
    *out = MultiSegmentVmmBuffer{};

    hipMemAllocationProp prop            = {};
    prop.type                            = hipMemAllocationTypePinned;
    prop.location.type                   = hipMemLocationTypeDevice;
    prop.location.id                     = dev;
    prop.requestedHandleType             = hipMemHandleTypePosixFileDescriptor;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    size_t granularity = 0;
    if (hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum) != hipSuccess
        || granularity == 0)
        return false;

    const size_t segSize   = ((segBytes + granularity - 1) / granularity) * granularity;
    const size_t totalSize = segSize * static_cast<size_t>(nSegments);

    hipDeviceptr_t base = 0;
    if (hipMemAddressReserve(&base, totalSize, granularity, 0, 0) != hipSuccess) return false;

    std::vector<hipMemGenericAllocationHandle_t> handles;
    auto cleanup = [&]() {
        for (size_t i = 0; i < handles.size(); ++i) {
            (void)hipMemUnmap(reinterpret_cast<hipDeviceptr_t>(reinterpret_cast<uintptr_t>(base) + i * segSize), segSize);
            (void)hipMemRelease(handles[i]);
        }
        (void)hipMemAddressFree(base, totalSize);
    };

    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type    = hipMemLocationTypeDevice;
    accessDesc.location.id      = dev;
    accessDesc.flags            = hipMemAccessFlagsProtReadWrite;

    for (int s = 0; s < nSegments; ++s) {
        hipMemGenericAllocationHandle_t h = 0;
        if (hipMemCreate(&h, segSize, &prop, 0) != hipSuccess) { cleanup(); return false; }
        handles.push_back(h);
        hipDeviceptr_t segVa = reinterpret_cast<hipDeviceptr_t>(reinterpret_cast<uintptr_t>(base) + static_cast<uintptr_t>(s) * segSize);
        if (hipMemMap(segVa, segSize, 0, h, 0) != hipSuccess) { cleanup(); return false; }
    }
    if (hipMemSetAccess(base, totalSize, &accessDesc, 1) != hipSuccess) { cleanup(); return false; }

    out->ptr = reinterpret_cast<void*>(base);
    out->base = base; out->totalSize = totalSize; out->segSize = segSize;
    out->nSegments = nSegments; out->handles = std::move(handles);
    return true;
}

inline void FreeMultiSegmentVmm(MultiSegmentVmmBuffer& b) {
    if (b.ptr == nullptr) return;
    for (size_t i = 0; i < b.handles.size(); ++i) {
        hipDeviceptr_t segVa = reinterpret_cast<hipDeviceptr_t>(reinterpret_cast<uintptr_t>(b.base) + i * b.segSize);
        (void)hipMemUnmap(segVa, b.segSize);
        (void)hipMemRelease(b.handles[i]);
    }
    (void)hipMemAddressFree(b.base, b.totalSize);
    b = MultiSegmentVmmBuffer{};
}

// Export one dma-buf fd per segment and register the buffer as a multi-segment
// MR via the matching NET plugin helper (classic ncclIb vs CAST IbCast). The
// comm object is plugin-specific, so NCCL_NET=IB-CAST must not call the classic
// helper. Returns the registration result; on ncclSuccess *mhandle holds the
// composite handle. Returns ncclInvalidUsage (without touching *mhandle) if the
// dma-buf export API is unavailable at build time (older HIP), so callers can SKIP.
inline ncclResult_t RegisterMultiSegmentMr(void* comm, const MultiSegmentVmmBuffer& b, bool isCast, void** mhandle) {
#if NCCL_CUMEM_DMABUF_EXPORT_GATE
    std::vector<void*>    segAddrs(b.nSegments);
    std::vector<size_t>   segLens(b.nSegments);
    std::vector<uint64_t> segOffsets(b.nSegments, 0ULL);
    std::vector<int>      segFds(b.nSegments, -1);

    ncclResult_t ret = ncclSuccess;
    for (int s = 0; s < b.nSegments; s++) {
        uintptr_t segVa = reinterpret_cast<uintptr_t>(b.base) + static_cast<uintptr_t>(s) * b.segSize;
        int fd = -1;
        if (hipMemGetHandleForAddressRange((void*)&fd, (hipDeviceptr_t)segVa, b.segSize,
                                           hipMemRangeHandleTypeDmaBufFd, 0) != hipSuccess) {
            ret = ncclInvalidUsage; goto cleanup;
        }
        segAddrs[s] = reinterpret_cast<void*>(segVa);
        segLens[s]  = b.segSize;
        segFds[s]   = fd;
    }
    ret = isCast
        ? IbCastRegMrDmaBufMultiSeg(comm, b.nSegments, segAddrs.data(), segLens.data(),
                                    segOffsets.data(), segFds.data(), NCCL_PTR_CUDA, mhandle)
        : ncclIbRegMrDmaBufMultiSeg(comm, b.nSegments, segAddrs.data(), segLens.data(),
                                    segOffsets.data(), segFds.data(), NCCL_PTR_CUDA, mhandle);
cleanup:
    for (int s = 0; s < b.nSegments; s++) if (segFds[s] != -1) (void)close(segFds[s]);
    return ret;
#else
    (void)comm; (void)b; (void)mhandle;
    return ncclInvalidUsage; // dma-buf export API unavailable at build time
#endif
}

} // namespace RCCLNetIbTests

#endif // MPI_TESTS_ENABLED

#endif // NET_IB_MULTI_SEGMENT_HELPERS_HPP
