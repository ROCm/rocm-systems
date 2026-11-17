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

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
// Host-side launch settings of the two kernels. Coll provides:
//   kThresholdEnv, kDefaultThreshold, hostThresholdDefault(): the per-collective
//     threshold env and its compiled default (kDefaultThreshold in the device pass);
//   kCtasEnv, kCtasAlias (nullptr for none): the grid pin;
//   kSdmaCtas, lsaCtas(chunkBytes): the size-adaptive grid of each tier.
// Env values are read once per process.

// gcnArchName of the current device, read once per process; nullptr if unavailable.
static inline const char* ginSdmaGcnArchName() {
  static char name[256] = "";
  static const bool ok = [] {
    int dev = 0;
    hipDeviceProp_t prop;
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
    snprintf(name, sizeof(name), "%s", prop.gcnArchName);
    return true;
  }();
  return ok ? name : nullptr;
}

template <typename Coll>
static inline size_t ginSdmaResolveSdmaThreshold() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return Coll::kDefaultThreshold;
#else
  static const size_t cached =
      gin_sdma_allgather::resolveSdmaThresholdFromEnv(Coll::kThresholdEnv, Coll::hostThresholdDefault());
  return cached;
#endif
}

template <typename Coll>
static inline size_t ginSdmaResolveCtasEnv() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return gin_sdma_allgather::kAllGatherCtasUnset;
#else
  static const size_t cached = gin_sdma_allgather::parseCtasEnv(Coll::kCtasEnv, Coll::kCtasAlias);
  return cached;
#endif
}

// The size-adaptive grid, or the env pin, clamped to the pool the requirements above request.
template <typename Coll>
static inline int ginSdmaResolveLaunchCtas(size_t chunkBytes, size_t sdmaThreshold) {
  return gin_sdma_allgather::allGatherCtas(chunkBytes, sdmaThreshold, ginSdmaResolveCtasEnv<Coll>(),
                                           gin_sdma_allgather::allGatherPoolCtas(deviceCtaCount),
                                           Coll::kSdmaCtas, Coll::lsaCtas(chunkBytes));
}

// Mirrors testLaunchDeviceKernel() plus the host-resolved sdmaThreshold. `count` is what
// the kernel takes: elements for AllGather, bytes for Gather.
template <typename F>
static inline testResult_t ginSdmaLaunchDeviceKernel(F kernel, void* sendbuff, size_t sendoffset,
                                                     void* recvbuff, size_t recvoffset, size_t count,
                                                     int root, ncclComm_t comm, cudaStream_t stream,
                                                     size_t sdmaThreshold, int gridCtas) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  if (gridCtas < 1) gridCtas = 1;
  kernel<<<gridCtas, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, sdmaThreshold, *devComm);
  return testSuccess;
}
#endif

#endif  // GIN_SDMA_ALLGATHER_REQS_H_
