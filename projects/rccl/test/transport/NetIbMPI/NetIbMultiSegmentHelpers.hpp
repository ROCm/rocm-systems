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

#include "MultiSegmentVmmHelpers.hpp"
#include "nccl.h"
#include "plugin/nccl_net.h"
#include "rocmwrap.h"

namespace RCCLNetIbTests {

using RCCLTestHelpers::AllocMultiSegmentVmm;
using RCCLTestHelpers::AllocSegmentedVmm;
using RCCLTestHelpers::FreeMultiSegmentVmm;
using RCCLTestHelpers::MultiSegmentVmmBuffer;

// Register the buffer the way the NET proxy does: regMrDmaBuf with fd -1 and the
// export flags in offset. The plugin walks the range and counts the segments.
// Returns ncclInvalidUsage (without touching *mhandle) if the dma-buf export API
// is unavailable at build time (older HIP). Callers must not treat that as a
// successful cap reject.
inline ncclResult_t RegisterMultiSegmentMr(ncclNet_t* net, void* comm, const MultiSegmentVmmBuffer& b,
                                           void** mhandle) {
#if NCCL_CUMEM_DMABUF_EXPORT_GATE
    constexpr uint64_t kExportFlags = 0; // no forced PCIe mapping
    return net->regMrDmaBuf(comm, b.ptr, b.totalSize, NCCL_PTR_CUDA, kExportFlags, -1, mhandle);
#else
    (void)net; (void)comm; (void)b; (void)mhandle;
    return ncclInvalidUsage; // dma-buf export API unavailable at build time
#endif
}

} // namespace RCCLNetIbTests

#endif // MPI_TESTS_ENABLED

#endif // NET_IB_MULTI_SEGMENT_HELPERS_HPP
