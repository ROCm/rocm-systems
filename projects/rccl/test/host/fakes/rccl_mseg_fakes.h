/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Multi-segment entry points of src/rccl_dev_runtime.cc, for binaries that do not link it.

#ifndef RCCL_TEST_HOST_RCCL_MSEG_FAKES_H_
#define RCCL_TEST_HOST_RCCL_MSEG_FAKES_H_

#include <cstddef>
#include <cstdint>
#include <functional>

// rcclRmaProxyMaxInflightRequests: defaults to no multi-segment cap.
extern std::function<uint32_t(bool /*multiSeg*/, uint32_t /*maxRequests*/)> g_rmaMaxInflightRequests;
// rcclHostSegmentFdLookup: defaults to no published owner fd (-1).
extern std::function<int(uintptr_t /*base*/, size_t /*len*/, uint64_t* /*offset*/)> g_hostSegmentFdLookup;
// rcclMultiSegOn: defaults to multi-segment off.
extern std::function<bool(bool /*pluginRegistersPerSegment*/)> g_multiSegOn;

void ResetRcclMsegFakes();

#endif // RCCL_TEST_HOST_RCCL_MSEG_FAKES_H_
