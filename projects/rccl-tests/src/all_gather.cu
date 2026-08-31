
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
#include "rccl_vector_types.h"
#include "gin_sdma_devtime.h" // shared device-side (wall_clock64) timing scaffold
#endif

void AllGatherGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  size_t base = gin_sdma_allgather::chunkBaseCount(count, eltSize, nranks);
  *sendcount = base;
  *recvcount = base*nranks;
  *sendInplaceOffset = base;
  *recvInplaceOffset = 0;
  *paramcount = base;
}

testResult_t AllGatherInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? ((char*)args->recvbuffs[i])+rank*args->sendBytes : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep + rank, 1, 0));
    for (int j=0; j<nranks; j++) {
      TESTCHECK(InitData((char*)args->expected[i] + args->sendBytes*j, sendcount, 0, type, ncclSum, 33*rep + j, 1, 0));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

testResult_t  AllGatherGetAlgoProtoChannels(ncclComm_t comm, size_t count, ncclDataType_t type, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetAlgoInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetAlgoInfo(comm, ncclFunc_t::ncclFuncAllGather , count, type , 0, 0, 1, algo, proto, nchannels));
  return testSuccess;
}

testResult_t  AllGatherGetSymkInfo(ncclComm_t comm, size_t count, ncclDataType_t type, ncclRedOp_t op, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetSymkInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetSymkInfo(comm, ncclFunc_t::ncclFuncAllGather , count, type , op, algo, proto, nchannels));
  return testSuccess;
}

testResult_t  AllGatherGetCollImplInfo(ncclComm_t comm, size_t count, ncclDataType_t type, ncclRedOp_t op,
    const void* sendbuff, void* recvbuff, int graphCapturing, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetCollImplInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetCollImplInfo(comm, ncclFuncAllGather, count, type, op, sendbuff, recvbuff, graphCapturing, algo, proto, nchannels));
  return testSuccess;
}

void AllGatherGetBw(size_t count, size_t typesize, double sec, double* algBw, double* busBw, int nranks) {
  gin_sdma_allgather::bandwidthGBps(count, typesize, sec, nranks, algBw, busBw);
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
testResult_t AllGatherGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclComm_t comm) {
  if (!reqs || !comm) return testInternalError;

  ncclCommProperties_t commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
  if (ncclCommQueryProperties(comm, &commProperties) != ncclSuccess) {
    return testNcclError;
  }

  if (commProperties.nRanks != ncclTeamLsa(comm).nRanks) {
    fprintf(stderr, "GinHybridAllGatherKernel requires CUDA P2P connectivity across all ranks. "
                    "Not all ranks of this communicator have P2P connectivity.\n");
    return testInvalidUsage;
  }

  switch(deviceImpl) {
    case 3: { // GinHybridAllGatherKernel: LSA direct (small) + direct GIN puts (large)
      if (commProperties.ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInvalidUsage;
      }
      // Cover both the -V/deviceCtaCount launch and the size-adaptive CTA count the
      // kernel self-selects (allGatherCtas, clamped to allGatherPoolCtas()), decoupled
      // from -V -- the kernel indexes barrier/lsaBarrier/signal by blockIdx.x.
      {
        const int agBarCtas = gin_sdma_allgather::allGatherPoolCtas(deviceCtaCount);
        reqs->barrierCount = agBarCtas;
        reqs->lsaBarrierCount = agBarCtas;
        reqs->ginSignalCount = agBarCtas;
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
bool AllGatherGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));

  switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: { // GinHybridAllGatherKernel: LSA direct (small) + direct GIN puts (large)
      // Size to cover the size-adaptive CTA count (allGatherCtas) as well as -V.
      const int agBarCtas = gin_sdma_allgather::allGatherPoolCtas(deviceCtaCount);
      reqs->barrierCount = agBarCtas;
      reqs->lsaBarrierCount = agBarCtas;
      reqs->ginSignalCount = agBarCtas;
      return true;
    }
#endif
    default:
      return false;
  }
}
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
// Host-resolve the AllGather LSA<->SDMA crossover (bytes/rank). Parsed once per
// process; see gin_sdma_allgather::resolveSdmaThresholdFromEnv().
static inline size_t AllGatherResolveSdmaThreshold() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return gin_sdma_allgather::kAllGatherSdmaThresholdDefault;
#else
  static const size_t cached = gin_sdma_allgather::resolveSdmaThresholdFromEnv();
  return cached;
#endif
}

// Parse NCCL_GIN_ANVIL_SDMA_ALLGATHER_CTAS once per process (deprecated alias
// NCCL_GIN_ANVIL_AG_CTAS still honored in the policy helper).
static inline size_t AllGatherResolveCtasEnv() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return gin_sdma_allgather::kAllGatherCtasUnset;
#else
  static const size_t cached = gin_sdma_allgather::parseAllGatherCtasEnv();
  return cached;
#endif
}

static inline int AllGatherResolveLaunchCtas(size_t chunkBytes, size_t sdmaThreshold, int poolCtas) {
  return gin_sdma_allgather::allGatherCtas(
      chunkBytes, sdmaThreshold, AllGatherResolveCtasEnv(), poolCtas);
}

template <typename T>
__device__ void AllGatherLsaDirect(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int rank, int lsaRanks, int tid, int nthreads) {
  T* src = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  const size_t dstOff = gin_sdma_allgather::allGatherRecvSliceOffset(rank, count);
  for (size_t i = tid; i < count; i += nthreads) {
    T value = src[i];
    for (int lp = 0; lp < lsaRanks; lp++) {
      T* dst = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp) + dstOff;
      dst[i] = value;
    }
  }
}

// Single-node hybrid AllGather body (-D 3), factored out so both the production
// kernel and the device-timing kernel share one implementation:
//   chunkBytes <= sdmaThreshold: direct LSA (all CTAs).
//   chunkBytes >  sdmaThreshold: direct all-peers GIN puts (proven MI355X path).
// Each invocation re-derives its entry barrier (LSA barrier for the small tier,
// GIN world barrier + signal read for the large tier), so back-to-back calls are
// self-synchronizing -- the timed kernel loops this body with no extra bookkeeping.
template <typename T>
__device__ void ginAllGatherBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, size_t sdmaThreshold, const struct ncclDevComm& devComm) {
  const size_t chunkBytes = count * sizeof(T);

  if (gin_sdma_allgather::chunkUsesLsaTier(chunkBytes, sdmaThreshold)) {
    ncclTeam lsa = ncclTeamLsa(devComm);
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int nthreads = blockDim.x * gridDim.x;

    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
    AllGatherLsaDirect<T>(sendwin, sendoffset, recvwin, recvoffset, count, devComm.rank, lsa.nRanks, tid, nthreads);
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    return;
  }

  const int ginContext = 0;
  const unsigned int signalIndex = blockIdx.x;
  ncclGin gin { devComm, ginContext };
  const uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;

  for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
        recvwin, recvoffset + (size_t)devComm.rank * chunkBytes,
        sendwin, sendoffset,
        chunkBytes, ncclGin_SignalInc{signalIndex});
  }

  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());
}

// Single-node hybrid AllGather (-D 3): one AllGather via the shared body.
template <typename T>
__global__ void GinHybridAllGatherKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, size_t sdmaThreshold, struct ncclDevComm devComm) {
  (void)root;
  ginAllGatherBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, sdmaThreshold, devComm);
}

// Device-timing kernel (shared gin_devtime methodology): run skip+loop back-to-back
// AllGather bodies under ONE persistent launch, bracketing only the timed region
// with wall_clock64() per CTA. Every body re-derives its entry barrier (a full
// inter-iteration sync), so looping is correct with no extra bookkeeping.
template <typename T>
__global__ void GinHybridAllGatherTimedKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, size_t sdmaThreshold, struct ncclDevComm devComm, int loop, int skip, long long* start_time, long long* end_time) {
  (void)root;
  for (int i = 0; i < skip + loop; i++) {
    if (i == skip) {
      __syncthreads();
      if (threadIdx.x == 0) start_time[blockIdx.x] = wall_clock64();
    }
    ginAllGatherBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, sdmaThreshold, devComm);
  }
  __syncthreads();
  if (threadIdx.x == 0) end_time[blockIdx.x] = wall_clock64();
}

// AllGather-specific launch: mirrors testLaunchDeviceKernel() but threads the
// host-resolved per-collective sdmaThreshold into the kernel (the shared launcher
// has a fixed signature used by the other collectives, so it is left untouched).
template <typename F>
static testResult_t AllGatherLaunchDeviceKernel(F kernel, void* sendbuff, size_t sendoffset,
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
#endif

testResult_t AllGatherRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
    NCCLCHECK(ncclAllGather(sptr, rptr, count, type, comm, stream));
  } else {
    switch(deviceImpl) {
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
      case 3: {
        // Size-adaptive CTA count (decoupled from -V), keyed off the same LSA<->SDMA
        // tier predicate the kernel evaluates: ~16 CTAs for the direct-store LSA
        // tier, few (4) for the GIN-put/SDMA tier where only nRanks threads issue
        // the puts and extra CTAs are pure barrier overhead. NCCL_GIN_ANVIL_SDMA_ALLGATHER_CTAS
        // pins a fixed count (diagnostic).
        const size_t agSdmaThreshold = AllGatherResolveSdmaThreshold();
        const size_t agChunkBytes = gin_sdma_allgather::chunkBytes(count, (size_t)wordSize(type));
        const int agGridCtas = AllGatherResolveLaunchCtas(
            agChunkBytes, agSdmaThreshold, gin_sdma_allgather::allGatherPoolCtas(deviceCtaCount));
        TESTCHECK(AllGatherLaunchDeviceKernel(SPECIALIZE_KERNEL(GinHybridAllGatherKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, root, comm, stream, agSdmaThreshold, agGridCtas));
        return testSuccess;
      }
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

// Device-side (in-kernel wall_clock64) timing for the GIN-SDMA AllGather (-D 3),
// via the shared gin_devtime scaffold. Opt-in via --device_timing: 1=augment
// (print an extra #[ag-devtime] line next to the graph numbers), 2=device-time-only
// (report the in-kernel latency as the out-of-place metric; in-place keeps normal
// timing). loop/skip come from --devtime_loop/--devtime_skip (default 10/10); size-
// tier overrides via --devtime_loop_mid/_large and --devtime_skip_mid/_large. Self-
// selects the same size-adaptive CTA count as the perf path (decoupled from -V),
// threading the host-resolved per-collective sdmaThreshold so the timed tier matches
// the launched cfg.
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
testResult_t AllGatherDeviceTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec) {
  if (!deviceTimingMode) return testSuccess;

  // Only the GIN hybrid impl (-D 3) provisions the GIN signals/barriers the timed
  // body relies on. Other device impls would fault or hang on missing GIN state.
  if (deviceImpl != 3) return testSuccess;

  const size_t count = args->nbytes / wordSize(type);   // per-rank input-chunk count
  if (count == 0 || devtimeLoop < 1) return testSuccess;

  const size_t chunkBytes = gin_sdma_allgather::chunkBytes(count, (size_t)wordSize(type));
  int loop = devtimeLoop;
  int skip = devtimeSkip < 0 ? 0 : devtimeSkip;
  gin_devtime::resolveLoopSkip(chunkBytes, loop, skip);

  auto kernel = SPECIALIZE_KERNEL(GinHybridAllGatherTimedKernel, type, op);
  if (kernel == nullptr) return testSuccess;

  const size_t sdmaThreshold = AllGatherResolveSdmaThreshold();
  const int gridCtas = AllGatherResolveLaunchCtas(
      chunkBytes, sdmaThreshold, gin_sdma_allgather::allGatherPoolCtas(deviceCtaCount));
  double devUs = 0.0;
  TESTCHECK(gin_devtime::measure(args, gridCtas, loop,
      [&](int i, long long* d_start, long long* d_end) {
        ncclDevComm* devComm = args->devComms + i;
        ncclWindow_t sendwin = (ncclWindow_t)(in_place ? args->recvRegHandles[i] : args->sendRegHandles[i]);
        ncclWindow_t recvwin = (ncclWindow_t)args->recvRegHandles[i];
        size_t sendoff = in_place ? args->sendInplaceOffset * (size_t)devComm->rank : 0;
        size_t recvoff = in_place ? args->recvInplaceOffset * (size_t)devComm->rank : 0;
        kernel<<<gridCtas, 512, 0, args->streams[i]>>>(sendwin, sendoff, recvwin, recvoff, count, root, sdmaThreshold, *devComm,
                 loop, skip, d_start, d_end);
      },
      &devUs));

  if (outDeltaSec != nullptr) {
    *outDeltaSec = (devUs > 0.0) ? devUs * 1.0e-6 : -1.0;
    return testSuccess;
  }

  if (args->proc == 0 && args->thread == 0 && devUs > 0.0) {
    int nRanksGlobal = args->nProcs * args->nThreads * args->nGpus;
    const size_t totalBytes = gin_sdma_allgather::chunkBytes(count, (size_t)wordSize(type)) * (size_t)nRanksGlobal;
    double sec = devUs * 1.0e-6;
    double algBw = 0.0, busBw = 0.0;
    gin_sdma_allgather::bandwidthGBps(count, (size_t)wordSize(type), sec, nRanksGlobal, &algBw, &busBw);
    const char* tierName = gin_sdma_allgather::chunkUsesLsaTier(chunkBytes, sdmaThreshold) ? "LSA" : "SDMA";
    snprintf(args->devtimeAugmentLine, sizeof(args->devtimeAugmentLine),
             "#[ag-devtime] size %12zu B  tier %-4s  ctas %2d  loop %2d skip %2d  devtime %10.2f us  algbw %8.2f GB/s  busbw %8.2f GB/s\n",
             totalBytes, tierName, gridCtas, loop, skip, devUs, algBw, busBw);
  }
  return testSuccess;
}
#else
testResult_t AllGatherDeviceTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec) {
  return testSuccess;  // device API path not available in this build
}
#endif

struct testColl allGatherTest = {
  "AllGather",
  AllGatherGetCollByteCount,
  AllGatherInitData,
  AllGatherGetBw,
  AllGatherRunColl,
  AllGatherGetAlgoProtoChannels,
  AllGatherGetSymkInfo,
  AllGatherGetCollImplInfo,
  AllGatherDeviceTime
};

void AllGatherGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllGatherGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AllGatherRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &allGatherTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i=0; i<type_count; i++) {
    TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
  }
  return testSuccess;
}

NCCL_WEAK struct testEngine ncclTestEngine = {
  /* .getBuffSize = */ AllGatherGetBuffSize,
  /* .runTest = */ AllGatherRunTest,
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
  /* .initCommConfig = */ nullptr,
#endif
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  /* .getDevCommRequirements = */ AllGatherGetDevCommRequirements,
#endif
};
