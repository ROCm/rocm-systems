/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cstdlib>
#include "cuda_runtime.h"
#include "common.h"
#include "gin_sdma_allgather_policy.h"
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#endif

void GatherGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *sendcount = (count/nranks) & ~(16/eltSize - 1);
  *recvcount = (*sendcount)*nranks;
  *sendInplaceOffset = *sendcount;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t GatherInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? ((char*)args->recvbuffs[i])+rank*args->sendBytes : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, rank*sendcount, type, ncclSum, rep, 1, 0));
    CUDACHECK(cudaMemcpy(args->expected[i], args->recvbuffs[i], args->expectedBytes, cudaMemcpyDefault));
    if (rank == root) {
      TESTCHECK(InitData(args->expected[i], nranks*sendcount, 0, type, ncclSum, rep, 1, 0));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

void GatherGetBw(size_t count, size_t typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
testResult_t GatherGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclComm_t comm) {
  if (!reqs || !comm) return testInternalError;

  ncclCommProperties_t commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
  if (ncclCommQueryProperties(comm, &commProperties) != ncclSuccess) {
    return testNcclError;
  }

  if (commProperties.nRanks != ncclTeamLsa(comm).nRanks) {
    fprintf(stderr, "GinHybridGatherKernel requires P2P connectivity across all ranks. "
                    "Not all ranks of this communicator have P2P connectivity.\n");
    return testInvalidUsage;
  }

  switch(deviceImpl) {
    case 3: { // GinHybridGatherKernel: LSA direct (small) + one GIN put to root (large)
      if (commProperties.ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInvalidUsage;
      }
#ifdef NCCL_GIN_BACKEND_MASK_ALL
      TESTCHECK(testCheckGinBackendCompiled(commProperties.ginType));
#endif
      // Same barrier/signal pool as GinHybridAllGatherKernel. The launched grid is
      // size-adaptive and never exceeds this pool; kernels index it by blockIdx.x.
      {
        const int barCtas = gin_sdma_allgather::allGatherPoolCtas(deviceCtaCount);
        reqs->barrierCount = barCtas;
        reqs->lsaBarrierCount = barCtas;
        reqs->ginSignalCount = barCtas;
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
bool GatherGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));

  switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: {
      const int barCtas = gin_sdma_allgather::allGatherPoolCtas(deviceCtaCount);
      reqs->barrierCount = barCtas;
      reqs->lsaBarrierCount = barCtas;
      reqs->ginSignalCount = barCtas;
      return true;
    }
#endif
    default:
      return false;
  }
}
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
// Gather LSA<->SDMA crossover. GATHER overrides the shared AllGather threshold;
// otherwise the global NCCL_GIN_ANVIL_SDMA_THRESHOLD, then the AllGather default.
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
static size_t gatherSdmaThresholdOnce() {
  unsigned long long perColl = 0;
  unsigned long long global = 0;
  const bool perCollSet = gin_sdma_allgather::parseEnvU64("NCCL_GIN_ANVIL_SDMA_THRESHOLD_GATHER", &perColl);
  const bool globalSet = gin_sdma_allgather::parseEnvU64("NCCL_GIN_ANVIL_SDMA_THRESHOLD", &global);
  return gin_sdma_allgather::pickSdmaThreshold(perCollSet, perColl, globalSet, global,
                                               gin_sdma_allgather::kAllGatherSdmaThresholdDefault);
}

static size_t gatherCtasEnvOnce() {
  const char* e = getenv("NCCL_GIN_ANVIL_SDMA_GATHER_CTAS");
  if (!e || !e[0] || *e == '-') return gin_sdma_allgather::kAllGatherCtasUnset;
  char* end = nullptr;
  unsigned long long v = strtoull(e, &end, 10);
  if (end == e || *end != '\0') return gin_sdma_allgather::kAllGatherCtasUnset;
  if ((size_t)v == gin_sdma_allgather::kAllGatherCtasUnset) return gin_sdma_allgather::kAllGatherCtasUnset;
  return (size_t)v;
}
#endif

static inline size_t GatherResolveSdmaThreshold() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return gin_sdma_allgather::kAllGatherSdmaThresholdDefault;
#else
  static const size_t cached = gatherSdmaThresholdOnce();
  return cached;
#endif
}

// One CTA issues the root put. The LSA tier reuses the AllGather 16-CTA store grid.
// NCCL_GIN_ANVIL_SDMA_GATHER_CTAS pins both tiers, clamped to the barrier pool.
static constexpr int kGatherCtasSdma = 1;

static inline size_t GatherResolveCtasEnv() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return gin_sdma_allgather::kAllGatherCtasUnset;
#else
  static const size_t cached = gatherCtasEnvOnce();
  return cached;
#endif
}

static inline int GatherResolveLaunchCtas(size_t chunkBytes, size_t sdmaThreshold, int poolCtas) {
  if (poolCtas < 1) poolCtas = 1;
  const size_t envCtas = GatherResolveCtasEnv();
  if (envCtas != gin_sdma_allgather::kAllGatherCtasUnset && envCtas > 0)
    return (envCtas > (size_t)poolCtas) ? poolCtas : (int)envCtas;
  const int adaptive = gin_sdma_allgather::chunkUsesLsaTier(chunkBytes, sdmaThreshold)
                           ? gin_sdma_allgather::kAllGatherCtasLsa
                           : kGatherCtasSdma;
  return (adaptive > poolCtas) ? poolCtas : adaptive;
}

// Each rank stores its chunk into root's recv slice. `root` is a world rank;
// ncclGetPeerPointer maps it into the LSA window. Non-root recv buffers stay untouched.
template <typename T>
__device__ void GatherLsaDirect(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int rank, int root, int tid, int nthreads) {
  T* src = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  const size_t dstOff = gin_sdma_allgather::allGatherRecvSliceOffset(rank, count);
  T* dst = (T*)ncclGetPeerPointer(recvwin, recvoffset, root) + dstOff;
  for (size_t i = tid; i < count; i += nthreads) {
    dst[i] = src[i];
  }
}

// Single-node hybrid Gather body (-D 3), the AllGather body with one destination:
//   chunkBytes <= sdmaThreshold: direct LSA store into root (all CTAs).
//   chunkBytes >  sdmaThreshold: one GIN put to root, issued by a single thread.
//     The AllGather peer stripe still picks that thread: root is owned by CTA
//     (root % gridDim.x), thread (root / gridDim.x). Every rank issues that one
//     put, so root's signal[root % gridDim.x] advances by nRanks and only that
//     CTA on root waits. The exit barrier holds every rank until that wait
//     completes, which is what makes back-to-back launches safe.
template <typename T>
__device__ void ginGatherBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, size_t sdmaThreshold, const struct ncclDevComm& devComm) {
  const size_t chunkBytes = count * sizeof(T);

  if (gin_sdma_allgather::chunkUsesLsaTier(chunkBytes, sdmaThreshold)) {
    ncclTeam lsa = ncclTeamLsa(devComm);
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int nthreads = blockDim.x * gridDim.x;

    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
    GatherLsaDirect<T>(sendwin, sendoffset, recvwin, recvoffset, count, devComm.rank, root, tid, nthreads);
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    return;
  }

  const int ginContext = 0;
  const unsigned int signalIndex = blockIdx.x;
  ncclGin gin { devComm, ginContext };
  const uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);

  const int peerStride = gridDim.x;
  for (int r = blockIdx.x + (int)threadIdx.x * peerStride; r < devComm.nRanks;
       r += peerStride * (int)blockDim.x) {
    if (r != root) continue;
    gin.put(ncclTeamWorld(devComm), root,
        recvwin, recvoffset + (size_t)devComm.rank * chunkBytes,
        sendwin, sendoffset,
        chunkBytes, ncclGin_SignalInc{signalIndex});
  }
  __syncthreads();

  if (devComm.rank == root && blockIdx.x == (root % peerStride))
    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());

  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

template <typename T>
__global__ void GinHybridGatherKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, size_t sdmaThreshold, struct ncclDevComm devComm) {
  ginGatherBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, root, sdmaThreshold, devComm);
}

template <typename F>
static testResult_t GatherLaunchDeviceKernel(F kernel, void* sendbuff, size_t sendoffset,
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

testResult_t GatherRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  if (deviceImpl == 0) {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    size_t rankOffset = count * wordSize(type);
    if (count == 0) return testSuccess;

    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
    NCCLCHECK(ncclGather(sptr, rptr, count, type, root, comm, stream));
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,7,0)
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclSend(sptr, count, type, root, comm, stream));
    if (rank == root) {
      for (int r=0; r<nRanks; r++) {
        NCCLCHECK(ncclRecv(rptr + r * rankOffset, count, type, r, comm, stream));
      }
    }
    NCCLCHECK(ncclGroupEnd());
#else
    printf("NCCL 2.7 or later is needed for gather. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#endif
  } else {
    switch(deviceImpl) {
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
      case 3: {
        if (count == 0) return testSuccess;
        const size_t sdmaThreshold = GatherResolveSdmaThreshold();
        const size_t chunkBytes = gin_sdma_allgather::chunkBytes(count, (size_t)wordSize(type));
        const int gridCtas = GatherResolveLaunchCtas(
            chunkBytes, sdmaThreshold, gin_sdma_allgather::allGatherPoolCtas(deviceCtaCount));
        TESTCHECK(GatherLaunchDeviceKernel(SPECIALIZE_KERNEL(GinHybridGatherKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, root, comm, stream, sdmaThreshold, gridCtas));
        return testSuccess;
      }
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

struct testColl gatherTest = {
  "Gather",
  GatherGetCollByteCount,
  GatherInitData,
  GatherGetBw,
  GatherRunColl,
  NULL,
  NULL
};

void GatherGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  GatherGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t GatherRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &gatherTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;
  int begin_root, end_root;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if (root != -1) {
    begin_root = end_root = root;
  } else {
    begin_root = 0;
    end_root = args->nProcs*args->nThreads*args->nGpus-1;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=begin_root; j<=end_root; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", j));
    }
  }
  return testSuccess;
}

NCCL_WEAK struct testEngine ncclTestEngine = {
  /* .getBuffSize = */ GatherGetBuffSize,
  /* .runTest = */ GatherRunTest,
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
  /* .initCommConfig = */ nullptr,
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0) || (defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0))
  /* .getDevCommRequirements = */ GatherGetDevCommRequirements,
#endif
};
