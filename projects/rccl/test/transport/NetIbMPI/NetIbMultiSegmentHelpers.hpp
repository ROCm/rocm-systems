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

#include "MultiSegmentVmmHelpers.hpp"
#include "nccl.h"
#include "rccl_ib_multiseg.h"
#include "rocmwrap.h"

namespace RCCLNetIbTests {

using RCCLTestHelpers::AllocMultiSegmentVmm;
using RCCLTestHelpers::FreeMultiSegmentVmm;
using RCCLTestHelpers::MultiSegmentVmmBuffer;

// Export one dma-buf fd per segment and register the buffer as a multi-segment
// MR via the classic NET/IB plugin entry point. Returns the registration result;
// on ncclSuccess *mhandle holds the composite handle. Returns ncclInvalidUsage
// (without touching *mhandle) if the dma-buf export API is unavailable at build
// time (older HIP). Callers must not treat that as a successful cap reject.
inline ncclResult_t RegisterMultiSegmentMr(void* comm, const MultiSegmentVmmBuffer& b, void** mhandle) {
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
    ret = ncclIbRegMrDmaBufMultiSeg(comm, b.nSegments, segAddrs.data(), segLens.data(),
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
