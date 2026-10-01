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

// AllocDeepEpElasticVmm lives in HybridVmmHelpers.hpp so UBR and RMA share one
// HIP sequence. Include that header after this one; it fills MultiSegmentVmmBuffer.

} // namespace RCCLRmaTests

#endif // RCCL_HAS_RMA_IB_PROXY
#endif // MPI_TESTS_ENABLED

#endif // RMA_MULTI_SEGMENT_HELPERS_HPP
