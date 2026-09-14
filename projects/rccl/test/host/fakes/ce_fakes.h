/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Copy-engine availability gates defined by src/ce_coll.cc. All default to
// "unavailable" so the CE arms stay off unless a test asks for them.

#ifndef RCCL_TEST_HOST_CE_FAKES_H_
#define RCCL_TEST_HOST_CE_FAKES_H_

#include <functional>

#include <hip/hip_runtime_api.h>

#include "nccl.h"

struct ncclComm;
struct ncclCeBatchOpsParams;
struct ncclCeCollArgs;

extern bool g_ceImplemented;  // UNDRIVEN
extern bool g_ceAvailable;  // UNDRIVEN
extern bool g_ceScratchAvailable;  // UNDRIVEN
extern bool g_hierCeAvailable;  // UNDRIVEN

// ce_coll.h's batch-ops API. rma_ce.cc builds a params block and submits it, so
// a test asserting what it submitted drives these. Defaults succeed and record
// nothing; ncclCeLaunchBatchOps is the observation point.
extern std::function<ncclResult_t(struct ncclCeBatchOpsParams* /*params*/, int /*capacity*/)>
    g_ceInitBatchOpsParams;
extern std::function<ncclResult_t(struct ncclComm* /*comm*/,
                                  struct ncclCeBatchOpsParams* /*params*/,
                                  hipStream_t /*stream*/,
                                  struct ncclCeCollArgs* /*profilerArgs*/)>
    g_ceLaunchBatchOps;

void ResetCeFakes();

#endif  // RCCL_TEST_HOST_CE_FAKES_H_
