/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the GIN-SDMA collective entry points owned by src/algorithms/gin/sdma/*.cu.

#ifndef RCCL_TEST_HOST_GIN_SDMA_FAKES_H_
#define RCCL_TEST_HOST_GIN_SDMA_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclComm;

#if defined(ENABLE_ROCSHMEM_GIN)
extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t)>
    g_allGatherGinSdmaEligible;
extern std::function<ncclResult_t(const void*, void*, size_t, ncclDataType_t, ncclComm*, hipStream_t)>
    g_allGatherGinSdma;
extern std::function<int(int, size_t)> g_allGatherGinSdmaCtaCount;

extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t)>
    g_allToAllGinSdmaEligible;
extern std::function<ncclResult_t(const void*, void*, size_t, ncclDataType_t, ncclComm*, hipStream_t)>
    g_allToAllGinSdma;

extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t)>
    g_allReduceGinSdmaEligible;
extern std::function<ncclResult_t(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm*, hipStream_t)>
    g_allReduceGinSdma;
#endif /* ENABLE_ROCSHMEM_GIN */

void ResetGinSdmaFakes();

#endif  // RCCL_TEST_HOST_GIN_SDMA_FAKES_H_
