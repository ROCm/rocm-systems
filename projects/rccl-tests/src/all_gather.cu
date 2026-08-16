
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
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"
#include "rccl_vector_types.h"
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
testResult_t AllGatherGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclCommProperties_t* commProperties) {
  if (!reqs || !commProperties) return testInternalError;

  switch(deviceImpl) {
    case 3: { // GinHybridAllGatherKernel: LSA direct (small) + direct GIN puts (large)
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      reqs->barrierCount = deviceCtaCount;
      reqs->lsaBarrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
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
    case 3: // GinHybridAllGatherKernel: LSA direct (small) + direct GIN puts (large)
      reqs->barrierCount = deviceCtaCount;
      reqs->lsaBarrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
      return true;
#endif
    default:
      return false;
  }
}
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
// Host-resolve the AllGather LSA<->SDMA crossover (bytes/rank). Per-collective
// NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER overrides the global
// NCCL_GIN_ANVIL_SDMA_THRESHOLD, which overrides the compiled default. Parsed as
// a 64-bit size_t (an explicit 0 is honored -> forces the all-SDMA tier; large
// values do not wrap, unlike the backend's int-typed inline-put threshold). The
// resolved value is passed to the kernel, so the AllGather tier is decoupled from
// the backend gin.put inline-vs-copy-engine threshold (ctx->sdmaThreshold).
static size_t AllGatherResolveSdmaThreshold() {
  auto parseEnv = [](const char* name, bool* isSet, unsigned long long* val) {
    const char* e = getenv(name);
    if (e && e[0]) {
      char* end = nullptr;
      unsigned long long v = strtoull(e, &end, 10);
      if (end != e) { *isSet = true; *val = v; return; }
    }
    *isSet = false; *val = 0;
  };
  bool perCollSet = false, globalSet = false;
  unsigned long long perCollVal = 0, globalVal = 0;
  parseEnv("NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER", &perCollSet, &perCollVal);
  parseEnv("NCCL_GIN_ANVIL_SDMA_THRESHOLD", &globalSet, &globalVal);
  return gin_sdma_allgather::pickSdmaThreshold(
      perCollSet, perCollVal, globalSet, globalVal,
      gin_sdma_allgather::kAllGatherSdmaThresholdDefault);
}

template <typename T>
__device__ void AllGatherLsaDirect(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int rank, int nRanks, int tid, int nthreads) {
  T* src = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  const size_t dstOff = (size_t)rank * count;
  for (size_t i = tid; i < count; i += nthreads) {
    T value = src[i];
    for (int lp = 0; lp < nRanks; lp++) {
      T* dst = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp) + dstOff;
      dst[i] = value;
    }
  }
}

// Single-node hybrid AllGather (-D 3):
//   chunkBytes <= sdmaThreshold: direct LSA (all CTAs).
//   chunkBytes >  sdmaThreshold: direct all-peers GIN puts (proven MI355X path).
template <typename T>
__global__ void GinHybridAllGatherKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, size_t sdmaThreshold, struct ncclDevComm devComm) {
  const size_t chunkBytes = count * sizeof(T);

  if (gin_sdma_allgather::chunkUsesLsaTier(chunkBytes, sdmaThreshold)) {
    ncclTeam lsa = ncclTeamLsa(devComm);
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int nthreads = blockDim.x * gridDim.x;

    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
    AllGatherLsaDirect<T>(sendwin, sendoffset, recvwin, recvoffset, count, devComm.rank, devComm.nRanks, tid, nthreads);
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    return;
  }

  const int ginContext = 0;
  const unsigned int signalIndex = 0;
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

// AllGather-specific launch: mirrors testLaunchDeviceKernel() but threads the
// host-resolved per-collective sdmaThreshold into the kernel (the shared launcher
// has a fixed signature used by the other collectives, so it is left untouched).
template <typename F>
static testResult_t AllGatherLaunchDeviceKernel(F kernel, void* sendbuff, size_t sendoffset,
                                                void* recvbuff, size_t recvoffset, size_t count,
                                                int root, ncclComm_t comm, cudaStream_t stream,
                                                size_t sdmaThreshold) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  kernel<<<deviceCtaCount, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, sdmaThreshold, *devComm);
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
      case 3:
        TESTCHECK(AllGatherLaunchDeviceKernel(SPECIALIZE_KERNEL(GinHybridAllGatherKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, root, comm, stream, AllGatherResolveSdmaThreshold()));
        return testSuccess;
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

struct testColl allGatherTest = {
  "AllGather",
  AllGatherGetCollByteCount,
  AllGatherInitData,
  AllGatherGetBw,
  AllGatherRunColl,
  AllGatherGetAlgoProtoChannels,
  AllGatherGetSymkInfo,
  AllGatherGetCollImplInfo
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
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  /* .getDevCommRequirements = */ AllGatherGetDevCommRequirements,
#endif
};
