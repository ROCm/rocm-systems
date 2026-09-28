/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Copy-engine availability gates defined by src/ce_coll.cc. All default to
// "unavailable" so the CE arms stay off unless a test asks for them.

#ifndef RCCL_TEST_HOST_CE_FAKES_H_
#define RCCL_TEST_HOST_CE_FAKES_H_

#include <cstddef>
#include <functional>

#include <hip/hip_runtime_api.h>

#include "nccl.h"
#include "sym_kernels.h"

struct ncclComm;
struct ncclDevrWindow;
struct ncclCeBatchOpsParams;
struct ncclCeCollArgs;

extern std::function<ncclResult_t(struct ncclComm*)> g_ncclCeInit;

extern bool g_ceImplemented;
extern bool g_ceAvailableValue;
extern bool g_ceScratchAvailableValue;
extern bool g_hierCeAvailable;
extern std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t,
                          struct ncclDevrWindow*, struct ncclDevrWindow*)>
    g_ceAvailable;
extern std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t)>
    g_ceScratchAvailable;
extern std::function<int(ncclDataType_t, size_t)> g_ceLocalReduceBlocks;

// ce_coll.h's batch-ops API. Init allocates the op arrays as production does.
extern std::function<ncclResult_t(struct ncclCeBatchOpsParams* /*params*/, int /*capacity*/)>
    g_ceInitBatchOpsParams;
extern std::function<ncclResult_t(struct ncclComm* /*comm*/,
                                  struct ncclCeBatchOpsParams* /*params*/,
                                  hipStream_t /*stream*/,
                                  struct ncclCeCollArgs* /*profilerArgs*/)>
    g_ceLaunchBatchOps;

void ResetCeFakes();

#endif  // RCCL_TEST_HOST_CE_FAKES_H_
