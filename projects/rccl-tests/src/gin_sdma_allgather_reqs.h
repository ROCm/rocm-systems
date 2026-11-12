/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Device-communicator requirements of the GIN Anvil-SDMA hybrid AllGather and
// Gather (-D 3). Both kernels index the barrier, LSA barrier and signal pools by
// blockIdx.x over the same grid, so they request one pool from one place.
#ifndef GIN_SDMA_ALLGATHER_REQS_H_
#define GIN_SDMA_ALLGATHER_REQS_H_

#include "common.h"
#include "gin_sdma_allgather_policy.h"

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
static inline testResult_t ginSdmaAllGatherDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs,
                                                               ncclComm_t comm, const char* kernelName) {
  if (!reqs || !comm) return testInternalError;

  ncclCommProperties_t commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
  if (ncclCommQueryProperties(comm, &commProperties) != ncclSuccess) {
    return testNcclError;
  }

  if (commProperties.nRanks != ncclTeamLsa(comm).nRanks) {
    fprintf(stderr, "%s requires CUDA P2P connectivity across all ranks. "
                    "Not all ranks of this communicator have P2P connectivity.\n", kernelName);
    return testInvalidUsage;
  }

  switch(deviceImpl) {
    case 3: {
      if (commProperties.ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInvalidUsage;
      }
#ifdef NCCL_GIN_BACKEND_MASK_ALL
      TESTCHECK(testCheckGinBackendCompiled(commProperties.ginType));
#endif
      // Cover both the -V/deviceCtaCount launch and the size-adaptive CTA count the
      // kernels self-select (allGatherCtas, clamped to allGatherPoolCtas()).
      {
        const int poolCtas = gin_sdma_allgather::allGatherPoolCtas(deviceCtaCount);
        reqs->barrierCount = poolCtas;
        reqs->lsaBarrierCount = poolCtas;
        reqs->ginSignalCount = poolCtas;
      }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,7)
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
      reqs->ginForceEnable = true;
#endif
      return testSuccess;
    }
    default:
      return testNotImplemented;
  }
}
#elif defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
static inline bool ginSdmaAllGatherDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));

  switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: {
      const int poolCtas = gin_sdma_allgather::allGatherPoolCtas(deviceCtaCount);
      reqs->barrierCount = poolCtas;
      reqs->lsaBarrierCount = poolCtas;
      reqs->ginSignalCount = poolCtas;
      return true;
    }
#endif
    default:
      return false;
  }
}
#endif

#endif  // GIN_SDMA_ALLGATHER_REQS_H_
