/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#include "gin_sdma_allgather_policy.h"
#include "gin_sdma_allgather_reqs.h"
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

// -D 3 is GinHybridGatherKernel: LSA direct (small) + one GIN put to root (large),
// on the same pool as GinHybridAllGatherKernel.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
testResult_t GatherGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclComm_t comm) {
  return ginSdmaAllGatherDevCommRequirements(deviceImpl, reqs, comm, "GinHybridGatherKernel");
}
#elif defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
bool GatherGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  return ginSdmaAllGatherDevCommRequirements(deviceImpl, reqs);
}
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
// Gather LSA<->SDMA crossover in bytes per rank: NCCL_GIN_ANVIL_SDMA_THRESHOLD_GATHER >
// NCCL_GIN_ANVIL_SDMA_THRESHOLD > per-arch default. Fallback for archs not in the table below.
static constexpr size_t kGatherSdmaThresholdDefault = 1u << 20;

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
static constexpr gin_sdma_allgather::ArchSdmaThreshold kGatherSdmaThresholdByArch[] = {
  {"gfx942", 128u << 20},
  {"gfx1250", 512u << 10},
};

// gcnArchName of the current device, read once per process; nullptr if unavailable.
static const char* gatherGcnArchName() {
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

static size_t gatherArchSdmaThreshold() {
  return gin_sdma_allgather::archSdmaThreshold(
      gatherGcnArchName(), kGatherSdmaThresholdByArch,
      sizeof(kGatherSdmaThresholdByArch) / sizeof(kGatherSdmaThresholdByArch[0]),
      kGatherSdmaThresholdDefault);
}

static size_t gatherSdmaThresholdOnce() {
  return gin_sdma_allgather::resolveSdmaThresholdFromEnv(gin_sdma_allgather::kGatherSdmaThresholdEnv,
                                                         gatherArchSdmaThreshold());
}
#endif

static inline size_t GatherResolveSdmaThreshold() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return kGatherSdmaThresholdDefault;
#else
  static const size_t cached = gatherSdmaThresholdOnce();
  return cached;
#endif
}

// One CTA issues the root put. The LSA tier uses the AllGather 16-CTA store grid, except
// on archs with a measured ladder below. NCCL_GIN_ANVIL_SDMA_GATHER_CTAS pins both tiers,
// clamped to the barrier pool.
static constexpr int kGatherCtasSdma = 1;

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
// Every CTA runs its own LSA barrier, so small chunks are barrier-bound and fewer CTAs win.
// Measured per rank on 8x MI300X (gfx942): 1 CTA up to 2 KiB, 2 up to 8 KiB, 4 up to
// 32 KiB, 8 up to 512 KiB, 16 above. On 4x MI455X (gfx1250): 1 up to 8 KiB, 2 up to
// 32 KiB, 8 above.
static constexpr gin_sdma_allgather::LsaCtaStep kGatherLsaCtasGfx942[] = {
  {2u << 10, 1},
  {8u << 10, 2},
  {32u << 10, 4},
  {512u << 10, 8},
};
static constexpr gin_sdma_allgather::LsaCtaStep kGatherLsaCtasGfx1250[] = {
  {8u << 10, 1},
  {32u << 10, 2},
};
static constexpr gin_sdma_allgather::ArchLsaCtaLadder kGatherLsaCtasByArch[] = {
  {"gfx942", kGatherLsaCtasGfx942, sizeof(kGatherLsaCtasGfx942) / sizeof(kGatherLsaCtasGfx942[0]), 16},
  {"gfx1250", kGatherLsaCtasGfx1250, sizeof(kGatherLsaCtasGfx1250) / sizeof(kGatherLsaCtasGfx1250[0]), 8},
};
#endif

static inline int GatherLsaCtas(size_t chunkBytes) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return gin_sdma_allgather::kAllGatherCtasLsa;
#else
  static const gin_sdma_allgather::ArchLsaCtaLadder* ladder = gin_sdma_allgather::archLsaCtaLadder(
      gatherGcnArchName(), kGatherLsaCtasByArch,
      sizeof(kGatherLsaCtasByArch) / sizeof(kGatherLsaCtasByArch[0]));
  if (!ladder) return gin_sdma_allgather::kAllGatherCtasLsa;
  return gin_sdma_allgather::lsaCtasFromLadder(chunkBytes, ladder->steps, ladder->nSteps,
                                               ladder->above);
#endif
}

static inline size_t GatherResolveCtasEnv() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return gin_sdma_allgather::kAllGatherCtasUnset;
#else
  static const size_t cached = gin_sdma_allgather::parseCtasEnv("NCCL_GIN_ANVIL_SDMA_GATHER_CTAS");
  return cached;
#endif
}

static inline int GatherResolveLaunchCtas(size_t chunkBytes, size_t sdmaThreshold, int poolCtas) {
  return gin_sdma_allgather::allGatherCtas(chunkBytes, sdmaThreshold, GatherResolveCtasEnv(),
                                           poolCtas, kGatherCtasSdma, GatherLsaCtas(chunkBytes));
}

// Each rank stores its chunk into root's recv slice. `root` is a world rank;
// ncclGetPeerPointer maps it into the LSA window. Non-root recv buffers stay untouched.
// Gather only moves bytes, so the copy is 16 bytes per thread whatever the element
// type; a chunk or window offset that is not 16-byte aligned falls back to a byte copy.
__device__ void GatherLsaDirect(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t chunkBytes, int rank, int root, int tid, int nthreads) {
  const char* src = (const char*)ncclGetLocalPointer(sendwin, sendoffset);
  const size_t dstOff = gin_sdma_allgather::allGatherRecvSliceOffset(rank, chunkBytes);
  char* dst = (char*)ncclGetPeerPointer(recvwin, recvoffset, root) + dstOff;
  if (src == dst) return;
  if ((((uintptr_t)src | (uintptr_t)dst | chunkBytes) & 15) == 0) {
    // src and dst are distinct buffers once src == dst has returned, so __restrict__
    // holds. Issuing all kUnroll loads before their stores keeps several remote stores
    // in flight per thread (about 2x LSA-tier bandwidth on gfx1250). The tail stays in
    // the batched loop: a plain one-load remainder loop is as slow as no unrolling.
    // On gfx942 the copy is xGMI-bound and the unrolled loop is about 5% slower.
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    constexpr int kUnroll = 8;
#else
    constexpr int kUnroll = 1;
#endif
    const uint4* __restrict__ s = (const uint4*)src;
    uint4* __restrict__ d = (uint4*)dst;
    const size_t n = chunkBytes / 16;
    const size_t stride = (size_t)nthreads;
    for (size_t i = tid; i < n; i += kUnroll * stride) {
      uint4 v[kUnroll];
#pragma unroll
      for (int k = 0; k < kUnroll; k++) {
        if (i + k * stride < n) v[k] = s[i + k * stride];
      }
#pragma unroll
      for (int k = 0; k < kUnroll; k++) {
        if (i + k * stride < n) d[i + k * stride] = v[k];
      }
    }
  } else {
    for (size_t i = tid; i < chunkBytes; i += nthreads) dst[i] = src[i];
  }
}

// Single-node hybrid Gather body (-D 3), the AllGather body with one destination:
//   chunkBytes <= sdmaThreshold: direct LSA store into root (all CTAs).
//   chunkBytes >  sdmaThreshold: one GIN put to root from the thread the AllGather
//     peer stripe assigns to root. Only CTA (root % gridDim.x) on root waits, for
//     one signal per put; the exit barrier keeps back-to-back launches safe.
__device__ void ginGatherBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t chunkBytes, int root, size_t sdmaThreshold, const struct ncclDevComm& devComm) {
  if (gin_sdma_allgather::chunkUsesLsaTier(chunkBytes, sdmaThreshold)) {
    ncclTeam lsa = ncclTeamLsa(devComm);
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int nthreads = blockDim.x * gridDim.x;

    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
    GatherLsaDirect(sendwin, sendoffset, recvwin, recvoffset, chunkBytes, devComm.rank, root, tid, nthreads);
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    return;
  }

  const int ginContext = 0;
  const unsigned int signalIndex = blockIdx.x;
  ncclGin gin { devComm, ginContext };
  const uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);

  const bool isRoot = devComm.rank == root;
  const bool rootInPlace = gin_sdma_allgather::gatherRootSkipsSelfPut(
      isRoot, sendwin == recvwin, sendoffset, recvoffset, root, chunkBytes);
  const int peerStride = gridDim.x;
  for (int r = blockIdx.x + (int)threadIdx.x * peerStride; r < devComm.nRanks;
       r += peerStride * (int)blockDim.x) {
    if (r != root || rootInPlace) continue;
    gin.put(ncclTeamWorld(devComm), root,
        recvwin, recvoffset + (size_t)devComm.rank * chunkBytes,
        sendwin, sendoffset,
        chunkBytes, TEST_GIN_WEAK_SIGNAL_INC(signalIndex));
  }
  __syncthreads();

  if (isRoot && blockIdx.x == (root % peerStride)) {
    gin.waitSignal(ncclCoopCta(), signalIndex,
                   signalValue + gin_sdma_allgather::gatherRootExpectedSignals(devComm.nRanks, rootInPlace));
  }
  gin.flush(ncclCoopCta());

  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

__global__ void GinHybridGatherKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t chunkBytes, int root, size_t sdmaThreshold, struct ncclDevComm devComm) {
  ginGatherBody(sendwin, sendoffset, recvwin, recvoffset, chunkBytes, root, sdmaThreshold, devComm);
}

static testResult_t GatherLaunchDeviceKernel(void* sendbuff, size_t sendoffset,
                                             void* recvbuff, size_t recvoffset, size_t chunkBytes,
                                             int root, ncclComm_t comm, cudaStream_t stream,
                                             size_t sdmaThreshold, int gridCtas) {
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  if (gridCtas < 1) gridCtas = 1;
  GinHybridGatherKernel<<<gridCtas, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, chunkBytes, root, sdmaThreshold, *devComm);
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
#if NCCL_VERSION_CODE < NCCL_VERSION(2,29,0)
        {
          // The 2.28.x requirements hook gets no comm to reject this layout; the LSA
          // tier addresses root by world rank, which is only its LSA rank on one node.
          const ncclDevComm* devComm = (const ncclDevComm*)comm;
          if (devComm->lsaSize != devComm->nRanks) {
            fprintf(stderr, "GinHybridGatherKernel requires CUDA P2P connectivity across all ranks. "
                            "Not all ranks of this communicator have P2P connectivity.\n");
            return testInvalidUsage;
          }
        }
#endif
        if (count == 0) return testSuccess;
        const size_t sdmaThreshold = GatherResolveSdmaThreshold();
        const size_t chunkBytes = gin_sdma_allgather::chunkBytes(count, (size_t)wordSize(type));
        const int gridCtas = GatherResolveLaunchCtas(
            chunkBytes, sdmaThreshold, gin_sdma_allgather::allGatherPoolCtas(deviceCtaCount));
        TESTCHECK(GatherLaunchDeviceKernel(sendbuff, sendoffset, recvbuff, recvoffset, chunkBytes, root, comm, stream, sdmaThreshold, gridCtas));
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
