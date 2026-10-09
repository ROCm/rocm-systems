/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Floor for src/device/ce_reduce.cc's host launcher, which ce_coll.cc declares itself.

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "fail_loud.h"
#include "nccl.h"

ncclResult_t ncclCeLaunchPersistentReduce(const void*, void*, int, size_t, size_t, size_t, size_t, uint32_t*, size_t,
                                          uint32_t*, ncclDataType_t, ncclRedOp_t, hipStream_t, int) {
  FailLoudUnfaked("ce_reduce_fakes", "ncclCeLaunchPersistentReduce");
}
