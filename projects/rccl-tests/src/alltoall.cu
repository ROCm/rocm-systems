/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#include "rccl_vector_types.h"
#endif
#if defined(ENABLE_DEVICE_API) && !defined(ENABLE_ROCSHMEM_GIN)
#error "alltoall_perf GIN device tests require -DENABLE_ROCSHMEM=ON (sets ENABLE_ROCSHMEM_GIN; NCCL_GIN_ANVIL_ENABLE stays 0 without it)"
#endif
#if defined(ENABLE_DEVICE_API) && defined(ENABLE_ROCSHMEM_GIN)
#include "nccl_device/gin/gin_device_common.h"
#if NCCL_GIN_ANVIL_ENABLE != 1
#error "NCCL_GIN_ANVIL_ENABLE is 0: alltoall_perf device code will mis-dispatch GIN calls"
#endif
#endif
#ifdef ENABLE_ROCSHMEM
#include <rocshmem/rocshmem.hpp>
#ifdef MPI_SUPPORT
#include <mpi.h>
#endif
// Initialize rocshmem before ncclCommInit so that rocshmem_malloc etc. work.
// Called via test_pre_init_callback from common.cu's main(), after MPI_Init.
static bool rocshmemTestPreInitialized = false;

// rocSHMEM lifecycle for the test binary, keyed on NCCL_GIN_TYPE when set.
//   type 4 (GIN_ROCSHMEM): test init before ncclCommInit; librccl reuses + finalizes
//   type 5 (GIN_ANVIL):    no rocshmem in the test binary (Anvil SDMA via librccl)
//   other GIN / RCCL paths: librccl owns rocSHMEM; standalone IPC tests init+finalize
struct RocshmemTestPolicy {
  bool init;
  bool finalize;
};

static constexpr long kNcclGinTypeRocshmem = 4;
static constexpr long kNcclGinTypeAnvil = 5;

static long parseGinTypeEnv() {
  const char* ginTypeEnv = getenv("NCCL_GIN_TYPE");
  if (!ginTypeEnv) return -1;
  char* end = nullptr;
  long ginType = strtol(ginTypeEnv, &end, 0);
  if (end == ginTypeEnv) return -1;
  return ginType;
}

static RocshmemTestPolicy rocshmemTestPolicy() {
  RocshmemTestPolicy none = {false, false};
  long ginType = parseGinTypeEnv();
  if (ginType == kNcclGinTypeRocshmem) {
    return {true, false};
  }
  if (ginType == kNcclGinTypeAnvil) {
    return none;
  }

    // GIN proxy and other RCCL GIN tests: librccl owns rocSHMEM init/finalize.
  const char* ginEnable = getenv("NCCL_GIN_ENABLE");
  if (ginEnable && ginEnable[0] == '1') return none;

  // RCCL_ROCSHMEM_ENABLE set (0 or 1): avoid double-init / orphan heap at exit.
  const char* rcclRocshmem = getenv("RCCL_ROCSHMEM_ENABLE");
  if (rcclRocshmem != nullptr) return none;

  // Standalone rocSHMEM IPC reference path (no NCCL GIN).
  return {true, true};
}

static void rocshmemPreInit(int rank, int nranks) {
  if (!rocshmemTestPolicy().init) return;

  // Set the correct GPU before rocshmem_init so that device symbols
  // (ROCSHMEM_CTX_DEFAULT etc.) are initialized on the right device.
  int nGpus = 0;
  hipGetDeviceCount(&nGpus);
  if (nGpus > 0) hipSetDevice(rank % nGpus);

  rocshmem::rocshmem_uniqueid_t uid;
  if (rank == 0) rocshmem::rocshmem_get_uniqueid(&uid);
#ifdef MPI_SUPPORT
  MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
  rocshmem::rocshmem_init_attr_t attr;
  rocshmem::rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
  rocshmem::rocshmem_init_attr(rocshmem::ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
  rocshmemTestPreInitialized = true;
}

static void rocshmemTestFinalize() {
  if (!rocshmemTestPreInitialized) return;
  rocshmemTestPreInitialized = false;
  if (!rocshmemTestPolicy().finalize) return;
  rocshmem::rocshmem_finalize();
}

// Register the callback at static init time (before main)
static struct RocshmemCallbackRegistrar {
  RocshmemCallbackRegistrar() {
    test_pre_init_callback = rocshmemPreInit;
    test_post_finalize_callback = rocshmemTestFinalize;
  }
} _rocshmemReg;
#endif

void AlltoAllGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *paramcount = (count/nranks) & -(16/eltSize);
  *sendcount = nranks*(*paramcount);
  *recvcount = *sendcount;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
}

testResult_t AlltoAllInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep + rank, 1, 0));
    for (int j=0; j<nranks; j++) {
      size_t partcount = sendcount/nranks;
      TESTCHECK(InitData((char*)args->expected[i] + j*partcount*wordSize(type), partcount, rank*partcount, type, ncclSum, 33*rep + j, 1, 0));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  // We don't support in-place alltoall
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void AlltoAllGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
// Single-node: LSA barriers only in devComm, but connect GIN_ANVIL transport (GPU-initiated
// SDMA queues, needsProxyProgress=0). RCCL connects transport without GIN window registration
// or devComm GIN barrier/signal allocation when barrier/signal counts are zero.
static inline void AlltoAllSetSingleNodeTransportDevCommReqs(ncclDevCommRequirements* reqs) {
  reqs->lsaBarrierCount = deviceCtaCount;
  reqs->ginContextCount = 1;
  reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
}

static inline void AlltoAllSetGinHybridDevCommReqs(ncclDevCommRequirements* reqs) {
  reqs->ginContextCount = 1;
  reqs->lsaBarrierCount = deviceCtaCount;
  reqs->barrierCount = deviceCtaCount;
  reqs->railGinBarrierCount = deviceCtaCount;
  reqs->ginSignalCount = deviceCtaCount;
  reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
}

// set devComm reqs for alltoall device kernels
testResult_t AlltoAllGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclCommProperties_t* commProperties) {
  if (!reqs || !commProperties) return testInternalError;

  switch(deviceImpl) {
    case 1: // NvlAlltoAllKernel
    case 2: // NvlAlltoAllKernelOptimized
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
    case 3: // GinAlltoAllKernel — testLaunchDeviceKernel uses deviceCtaCount CTAs
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      if (commProperties->nLsaTeams == 1) {
        AlltoAllSetSingleNodeTransportDevCommReqs(reqs);
      } else {
        AlltoAllSetGinHybridDevCommReqs(reqs);
      }
      return testSuccess;
    case 4: // HybridAlltoAllKernel (LSA+GIN)
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      reqs->barrierCount = deviceCtaCount;
      reqs->railGinBarrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
      return testSuccess;
    case 5: // GinAdaptiveAlltoAllKernel (LSA-only intra-node + hybrid inter-node)
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      if (commProperties->nLsaTeams == 1) {
        AlltoAllSetSingleNodeTransportDevCommReqs(reqs);
      } else {
        AlltoAllSetGinHybridDevCommReqs(reqs);
      }
      return testSuccess;
    default:
      return testNotImplemented;
  }
}
#elif defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
// set devComm reqs for alltoall device kernels
bool AlltoAllGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));

  switch(deviceImpl) {
    case 1: // NvlAlltoAllKernel
    case 2: // NvlAlltoAllKernelOptimized
      reqs->lsaBarrierCount = deviceCtaCount;
      return true;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: // GinAlltoAllKernel
      reqs->ginContextCount = 1;
      reqs->lsaBarrierCount = deviceCtaCount;
      reqs->barrierCount = deviceCtaCount;
      reqs->railGinBarrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
      return true;
    case 4: // HybridAlltoAllKernel (LSA+GIN)
      reqs->barrierCount = deviceCtaCount;
      reqs->railGinBarrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
      return true;
    case 5: // GinAdaptiveAlltoAllKernel (LSA-only intra-node + hybrid inter-node)
      reqs->ginContextCount = 1;
      reqs->lsaBarrierCount = deviceCtaCount;
      reqs->barrierCount = deviceCtaCount;
      reqs->railGinBarrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
      return true;
#endif
    default:
      return false;
  }
}
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
// shared scalar AlltoAll implementation used by both kernels
template <typename T>
__device__ void AlltoAllScalarImpl(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int rank, int nRanks, int tid, int nthreads) {
  T* sendPtr = (T*)ncclGetLsaPointer(sendwin, sendoffset, rank);

  for (size_t offset = tid; offset < count; offset += nthreads) {
    for (int peer = 0; peer < nRanks; peer++) {
      T value = sendPtr[peer * count + offset];
      T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer);
      recvPtr[rank * count + offset] = value;
    }
  }
}

// Device implementation #1 - simple NVL kernel
template <typename T>
__global__ void NvlAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  int rank = devComm.rank, nRanks = devComm.nRanks;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int nthreads = blockDim.x * gridDim.x;

  AlltoAllScalarImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

// Message slice size above which CTAs stripe peers (not offsets) for xGMI link parallelism
// (AlltoAllNvlCopySelect). Same constant bounds the NVL-only fast path in AlltoAllGinAdaptiveLocalCopy:
// small slices use AlltoAllNvlCopyOptimized; large + multi-CTA uses NvlCopySelect; else LsaCopy.
constexpr size_t kAlltoAllPeerParallelByteThreshold = 128 * 1024;
// Rank-slice bytes below which GinAdaptiveAlltoAllKernel uses one CTA (latency); larger uses
// deviceCtaCount CTAs (see alltoall_perf -V).
static constexpr size_t kAlltoAllSmallMessageRankSliceBytes = 128 * 1024;

// Hybrid GIN path: pipeline remote SDMA puts with local LSA copy.
constexpr size_t kAlltoAllGinPipelineChunkBytes = 8 << 20;

// Vectorized copy of one rank-slice to one peer (used by peer-parallel alltoall).
template <typename T>
__device__ __forceinline__ void AlltoAllCopyOnePeerVectorized(T* sendPeer, T* recvPeer, size_t count,
                                                              int tid, int nthreads) {
  using TN = typename VectorTypeMapping<T>::Type;
  constexpr int VECTOR_FACTOR = sizeof(TN) / sizeof(T);
  constexpr int UNROLL_FACTOR = 128 / sizeof(TN);

  bool canVectorize = (sizeof(TN) > sizeof(T)) &&
                      (reinterpret_cast<uintptr_t>(sendPeer) % sizeof(TN) == 0) &&
                      (reinterpret_cast<uintptr_t>(recvPeer) % sizeof(TN) == 0) &&
                      ((count * sizeof(T)) % sizeof(TN) == 0);
  if (!canVectorize) {
    for (size_t offset = tid; offset < count; offset += nthreads)
      recvPeer[offset] = sendPeer[offset];
    return;
  }

  size_t vector_count = count / VECTOR_FACTOR;
  int elements_per_iteration = nthreads * UNROLL_FACTOR;
  size_t aligned_vector_count = (vector_count / elements_per_iteration) * elements_per_iteration;

  TN* sendVec = (TN*)sendPeer;
  TN* recvVec = (TN*)recvPeer;
  for (size_t base_offset = tid; base_offset < aligned_vector_count; base_offset += elements_per_iteration) {
    TN values[UNROLL_FACTOR];
    #pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
      size_t offset = base_offset + i * nthreads;
      values[i] = sendVec[offset];
    }
    #pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
      size_t offset = base_offset + i * nthreads;
      recvVec[offset] = values[i];
    }
  }
  for (size_t base_offset = aligned_vector_count + tid; base_offset < vector_count; base_offset += nthreads)
    recvVec[base_offset] = sendVec[base_offset];
  size_t scalar_start = vector_count * VECTOR_FACTOR;
  for (size_t offset = scalar_start + tid; offset < count; offset += nthreads)
    recvPeer[offset] = sendPeer[offset];
}

// CTA-peer striping: each block copies full slices to a subset of peers (large messages).
template <typename T>
__device__ __forceinline__ void AlltoAllNvlCopyPeerParallel(ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin, size_t recvoffset, size_t count, int rank, int nRanks, int blockId,
    int nBlocks, int tid, int nthreads) {
  (void)tid;
  (void)nthreads;
  T* sendLocal = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  for (int peer = blockId; peer < nRanks; peer += nBlocks) {
    T* sendPeer = sendLocal + peer * count;
    T* recvPeer = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + rank * count;
    int localTid = threadIdx.x;
    int localNthreads = blockDim.x;
    AlltoAllCopyOnePeerVectorized<T>(sendPeer, recvPeer, count, localTid, localNthreads);
  }
}

// Vectorized alltoall copy for full NVL peer sets (single-node / all-local paths).
template <typename T>
__device__ __forceinline__ void AlltoAllNvlCopyOptimized(ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin, size_t recvoffset, size_t count, int rank, int nRanks, int tid, int nthreads) {
  using TN = typename VectorTypeMapping<T>::Type;
  constexpr int VECTOR_FACTOR = sizeof(TN) / sizeof(T);
  constexpr int UNROLL_FACTOR = 128/sizeof(TN);
  constexpr int PEER_UNROLL = 4;

  T* sendPtr = (T*)ncclGetLsaPointer(sendwin, sendoffset, rank);

  bool canVectorize = (sizeof(TN) > sizeof(T)) &&
                      (reinterpret_cast<uintptr_t>(sendPtr) % sizeof(TN) == 0) &&
                      ((count * sizeof(T)) % sizeof(TN) == 0);

  if (!canVectorize) {
    AlltoAllScalarImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);
    return;
  }

  size_t vector_count = count / VECTOR_FACTOR;
  int elements_per_iteration = nthreads * UNROLL_FACTOR;
  size_t aligned_vector_count = (vector_count / elements_per_iteration) * elements_per_iteration;

  for (size_t base_offset = tid; base_offset < aligned_vector_count; base_offset += elements_per_iteration) {
    for (int peerBase = 0; peerBase < nRanks; peerBase += PEER_UNROLL) {
      int peersInGroup = min(PEER_UNROLL, nRanks - peerBase);

      #pragma unroll
      for (int p = 0; p < peersInGroup; p++) {
        int peer = peerBase + p;
        TN* sendVecPtr = (TN*)(sendPtr + peer * count);
        TN* recvVecPtr = (TN*)((T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + rank * count);
        TN values[UNROLL_FACTOR];

        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; i++) {
          size_t offset = base_offset + i * nthreads;
          values[i] = sendVecPtr[offset];
        }
        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; i++) {
          size_t offset = base_offset + i * nthreads;
          recvVecPtr[offset] = values[i];
        }
      }
    }
  }

  for (size_t base_offset = aligned_vector_count + tid; base_offset < vector_count; base_offset += nthreads) {
    for (int peer = 0; peer < nRanks; peer++) {
      TN* sendVecPtr = (TN*)(sendPtr + peer * count);
      TN* recvVecPtr = (TN*)((T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + rank * count);
      recvVecPtr[base_offset] = sendVecPtr[base_offset];
    }
  }

  size_t scalar_start = vector_count * VECTOR_FACTOR;
  for (size_t offset = scalar_start + tid; offset < count; offset += nthreads) {
    for (int peer = 0; peer < nRanks; peer++) {
      T value = sendPtr[peer * count + offset];
      T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer);
      recvPtr[rank * count + offset] = value;
    }
  }
}

template <typename T>
__device__ __forceinline__ void AlltoAllNvlCopySelect(ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin, size_t recvoffset, size_t count, int rank, int nRanks, int blockId,
    int nBlocks, int tid, int nthreads) {
  if (count * sizeof(T) >= kAlltoAllPeerParallelByteThreshold && nBlocks > 1 && nRanks > 1) {
    AlltoAllNvlCopyPeerParallel<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks,
                                   blockId, nBlocks, tid, nthreads);
  } else {
    AlltoAllNvlCopyOptimized<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks,
                                tid, nthreads);
  }
}

// Device implementation #2 - optimized NVL kernel using vectorization and unrolling
template <typename T>
__global__ void NvlAlltoAllKernelOptimized(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  int rank = devComm.rank, nRanks = devComm.nRanks;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int nthreads = blockDim.x * gridDim.x;

  AlltoAllNvlCopySelect<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks,
                           blockIdx.x, gridDim.x, tid, nthreads);

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
// Scalar LSA alltoall copy fallback.
template <typename T>
__device__ void AlltoAllLsaCopyScalar(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin,
                                      size_t recvoffset, size_t count, int recvRank, int startLsa,
                                      int lsaSize, int tid, int nthreads) {
  T* sendLocal = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  for (size_t offset = tid; offset < count; offset += nthreads) {
    for (int lp = 0; lp < lsaSize; lp++) {
      int wr = startLsa + lp;
      T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp);
      recvPtr[recvRank * count + offset] = sendLocal[wr * count + offset];
    }
  }
}

// LSA alltoall copy shared by NVL and GIN single-node fast paths.
template <typename T>
__device__ __forceinline__ void AlltoAllLsaCopy(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin,
                                size_t recvoffset, size_t count, int recvRank, int startLsa,
                                int lsaSize, int tid, int nthreads) {
  using TN = typename VectorTypeMapping<T>::Type;
  constexpr int VECTOR_FACTOR = sizeof(TN) / sizeof(T);
  constexpr int UNROLL_FACTOR = 128/sizeof(TN);
  constexpr int PEER_UNROLL = 4;

  T* sendLocal = (T*)ncclGetLocalPointer(sendwin, sendoffset);

  bool canVectorize = (sizeof(TN) > sizeof(T)) &&
                      (reinterpret_cast<uintptr_t>(sendLocal) % sizeof(TN) == 0) &&
                      ((count * sizeof(T)) % sizeof(TN) == 0);

  if (!canVectorize) {
    AlltoAllLsaCopyScalar<T>(sendwin, sendoffset, recvwin, recvoffset, count, recvRank, startLsa,
                             lsaSize, tid, nthreads);
    return;
  }

  size_t vector_count = count / VECTOR_FACTOR;
  int elements_per_iteration = nthreads * UNROLL_FACTOR;
  size_t aligned_vector_count = (vector_count / elements_per_iteration) * elements_per_iteration;

  for (size_t base_offset = tid; base_offset < aligned_vector_count; base_offset += elements_per_iteration) {
    for (int peerBase = 0; peerBase < lsaSize; peerBase += PEER_UNROLL) {
      int peersInGroup = min(PEER_UNROLL, lsaSize - peerBase);

      #pragma unroll
      for (int p = 0; p < peersInGroup; p++) {
        int lp = peerBase + p;
        int wr = startLsa + lp;
        TN* sendVecPtr = (TN*)(sendLocal + wr * count);
        TN* recvVecPtr = (TN*)((T*)ncclGetLsaPointer(recvwin, recvoffset, lp) + recvRank * count);
        TN values[UNROLL_FACTOR];

        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; i++) {
          size_t offset = base_offset + i * nthreads;
          values[i] = sendVecPtr[offset];
        }
        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; i++) {
          size_t offset = base_offset + i * nthreads;
          recvVecPtr[offset] = values[i];
        }
      }
    }
  }

  for (size_t base_offset = aligned_vector_count + tid; base_offset < vector_count; base_offset += nthreads) {
    for (int lp = 0; lp < lsaSize; lp++) {
      int wr = startLsa + lp;
      TN* sendVecPtr = (TN*)(sendLocal + wr * count);
      TN* recvVecPtr = (TN*)((T*)ncclGetLsaPointer(recvwin, recvoffset, lp) + recvRank * count);
      recvVecPtr[base_offset] = sendVecPtr[base_offset];
    }
  }

  size_t scalar_start = vector_count * VECTOR_FACTOR;
  for (size_t offset = scalar_start + tid; offset < count; offset += nthreads) {
    for (int lp = 0; lp < lsaSize; lp++) {
      int wr = startLsa + lp;
      T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp);
      recvPtr[recvRank * count + offset] = sendLocal[wr * count + offset];
    }
  }
}

// Gin-anvil single-node: small slices use NVL vectorized copy; large + multi-CTA uses
// AlltoAllNvlCopySelect (peer striping like -D 2); single-CTA large slices use LsaCopy.
template <typename T>
__device__ __forceinline__ void AlltoAllGinAdaptiveLocalCopy(ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin, size_t recvoffset, size_t count, int recvRank, int startLsa,
    int lsaSize, int rank, int nRanks, int blockId, int nBlocks, int tid, int nthreads) {
  if (count * sizeof(T) < kAlltoAllPeerParallelByteThreshold) {
    AlltoAllNvlCopyOptimized<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks,
                                tid, nthreads);
  } else if (nBlocks > 1) {
    AlltoAllNvlCopySelect<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks,
                             blockId, nBlocks, tid, nthreads);
  } else {
    AlltoAllLsaCopy<T>(sendwin, sendoffset, recvwin, recvoffset, count, recvRank, startLsa,
                       lsaSize, tid, nthreads);
  }
}

// Element-range variant for pipelined hybrid alltoall chunks.
template <typename T>
__device__ __forceinline__ void AlltoAllLsaCopyRange(ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin, size_t recvoffset, size_t count, size_t elemOff, size_t elemLen,
    int recvRank, int startLsa, int lsaSize, int tid, int nthreads) {
  T* sendLocal = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  for (int lp = 0; lp < lsaSize; lp++) {
    int wr = startLsa + lp;
    T* sendPeer = sendLocal + wr * count + elemOff;
    T* recvPeer = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp) + recvRank * count + elemOff;
    AlltoAllCopyOnePeerVectorized<T>(sendPeer, recvPeer, elemLen, tid, nthreads);
  }
}

// Issue remote GIN puts for one byte slice of each rank's alltoall chunk.
template <typename T>
__device__ __forceinline__ void AlltoAllGinPutRemoteSlice(ncclGin& gin, ncclTeam world, int startLsa,
    int lsaSize, int myRank, ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin,
    size_t recvoffset, size_t rankSliceBytes, size_t sliceOff, size_t sliceBytes, int tid,
    int nthreads, unsigned int signalIndex) {
  for (int r = tid; r < startLsa; r += nthreads) {
    gin.put(world, r,
        recvwin, recvoffset + (size_t)myRank * rankSliceBytes + sliceOff,
        sendwin, sendoffset + (size_t)r * rankSliceBytes + sliceOff,
        sliceBytes, ncclGin_SignalInc{signalIndex});
  }
  for (int r = startLsa + lsaSize + tid; r < world.nRanks; r += nthreads) {
    gin.put(world, r,
        recvwin, recvoffset + (size_t)myRank * rankSliceBytes + sliceOff,
        sendwin, sendoffset + (size_t)r * rankSliceBytes + sliceOff,
        sliceBytes, ncclGin_SignalInc{signalIndex});
  }
}

// Pipelined hybrid: for large messages, chunk remote SDMA puts and overlap with local LSA copy.
template <typename T>
__device__ __forceinline__ void AlltoAllHybridGinPipelined(ncclGin& gin, ncclTeam world, int startLsa,
    int lsaSize, int myRank, ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin,
    size_t recvoffset, size_t count, int tid, int nthreads, unsigned int signalIndex) {
  const size_t rankSliceBytes = count * sizeof(T);
  if (rankSliceBytes <= kAlltoAllGinPipelineChunkBytes) {
    AlltoAllGinPutRemoteSlice<T>(gin, world, startLsa, lsaSize, myRank, sendwin, sendoffset,
                                 recvwin, recvoffset, rankSliceBytes, 0, rankSliceBytes, tid,
                                 nthreads, signalIndex);
    AlltoAllLsaCopy<T>(sendwin, sendoffset, recvwin, recvoffset, count, myRank, startLsa, lsaSize,
                       tid, nthreads);
    return;
  }

  size_t elemPerChunk = kAlltoAllGinPipelineChunkBytes / sizeof(T);
  if (elemPerChunk == 0) elemPerChunk = 1;
  for (size_t elemOff = 0; elemOff < count; elemOff += elemPerChunk) {
    size_t elemLen = min(elemPerChunk, count - elemOff);
    size_t sliceBytes = elemLen * sizeof(T);
    size_t sliceOff = elemOff * sizeof(T);
    AlltoAllGinPutRemoteSlice<T>(gin, world, startLsa, lsaSize, myRank, sendwin, sendoffset,
                                 recvwin, recvoffset, rankSliceBytes, sliceOff, sliceBytes, tid,
                                 nthreads, signalIndex);
    AlltoAllLsaCopyRange<T>(sendwin, sendoffset, recvwin, recvoffset, count, elemOff, elemLen,
                            myRank, startLsa, lsaSize, tid, nthreads);
  }
}

template <typename T>
__global__ void GinAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  constexpr int ginContext = 0;

  ncclTeam world = ncclTeamWorld(devComm);
  ncclTeam lsa = ncclTeamLsa(devComm);
  const int startLsa = world.rank - lsa.rank;
  const int numRemotePeers = world.nRanks - lsa.nRanks;
  const bool allLocal = (numRemotePeers == 0);
  const size_t size = count * sizeof(T);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  if (allLocal) {
    // Single-node: LSA barriers + LSA copies only (no GIN signal/wait/flush).
    ncclLsaBarrierSession<ncclCoopCta> lsaBar {
      ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x};
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

    AlltoAllGinAdaptiveLocalCopy<T>(sendwin, sendoffset, recvwin, recvoffset, count,
                                    world.rank, startLsa, lsa.nRanks, world.rank, world.nRanks,
                                    blockIdx.x, gridDim.x, tid, nthreads);

    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    return;
  }

  // Multi-node: LSA inbox + GIN-rail barrier, remote GIN puts, local LSA copy.
  ncclGin gin { devComm, ginContext };
  unsigned int signalIndex = 0;
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar {
    ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  int numChunks = (int)((size + kAlltoAllGinPipelineChunkBytes - 1) / kAlltoAllGinPipelineChunkBytes);
  int totalSignals = numRemotePeers * numChunks;
  AlltoAllHybridGinPipelined<T>(gin, world, startLsa, lsa.nRanks, world.rank, sendwin, sendoffset,
                                recvwin, recvoffset, count, tid, nthreads, signalIndex);

  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + totalSignals);
  gin.flush(ncclCoopCta());

  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

template <typename T>
__global__ void HybridAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  int ginContext = blockIdx.x;
  unsigned int signalIndex = 0;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  ncclTeam world = ncclTeamWorld(devComm);
  ncclTeam lsa = ncclTeamLsa(devComm);
  const int startLsa = world.rank - lsa.rank;
  const int lsaSize  = lsa.nRanks;

  const size_t size = count * sizeof(T);
  int numRemotePeers = world.nRanks - lsa.nRanks;
  int numChunks = (int)((size + kAlltoAllGinPipelineChunkBytes - 1) / kAlltoAllGinPipelineChunkBytes);
  int totalSignals = numRemotePeers * numChunks;
  AlltoAllHybridGinPipelined<T>(gin, world, startLsa, lsaSize, world.rank, sendwin, sendoffset,
                                recvwin, recvoffset, count, tid, nthreads, signalIndex);

  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + totalSignals);
  gin.flush(ncclCoopCta());

  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

// Device implementation #5 - adaptive GIN alltoall:
//   single-node (all LSA peers): LSA barriers + LSA copy only (no GIN session), multi-CTA
//   multi-node: hybrid GIN puts for remote peers + LSA copy for local peers (like -D 4)
template <typename T>
__global__ void GinAdaptiveAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclTeam world = ncclTeamWorld(devComm);
  ncclTeam lsa = ncclTeamLsa(devComm);
  const int startLsa = world.rank - lsa.rank;
  const int lsaSize = lsa.nRanks;
  const int numRemotePeers = world.nRanks - lsa.nRanks;
  const bool allLocal = (numRemotePeers == 0);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  if (allLocal) {
    ncclLsaBarrierSession<ncclCoopCta> lsaBar {
      ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x};
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

    AlltoAllGinAdaptiveLocalCopy<T>(sendwin, sendoffset, recvwin, recvoffset, count,
                                    world.rank, startLsa, lsa.nRanks, world.rank, world.nRanks,
                                    blockIdx.x, gridDim.x, tid, nthreads);

    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    return;
  }

  int ginContext = (devComm.ginContextCount > 0) ? (blockIdx.x % devComm.ginContextCount) : 0;
  unsigned int signalIndex = 0;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  const size_t size = count * sizeof(T);
  int numChunks = (int)((size + kAlltoAllGinPipelineChunkBytes - 1) / kAlltoAllGinPipelineChunkBytes);
  int totalSignals = numRemotePeers * numChunks;
  AlltoAllHybridGinPipelined<T>(gin, world, startLsa, lsaSize, world.rank, sendwin, sendoffset,
                                recvwin, recvoffset, count, tid, nthreads, signalIndex);

  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + totalSignals);
  gin.flush(ncclCoopCta());

  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

#endif
#endif

testResult_t AlltoAllRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
    if (test_ncclVersion >= NCCL_VERSION(2,28,0)) {
      NCCLCHECK(ncclAlltoAll(sptr, rptr, count, type, comm, stream));
      return testSuccess;
    }
    if (test_ncclVersion >= NCCL_VERSION(2,19,0)) {
      NCCLCHECK(ncclAllToAll(sptr, rptr, count, type, comm, stream));
      return testSuccess;
    }
    printf("RCCL 2.19 or later is needed for alltoall API path. This test was compiled with %d.%d, but is running with RCCL %d.\n",
           NCCL_MAJOR, NCCL_MINOR, test_ncclVersion);
    return testNcclError;
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,7,0)
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t rankOffset = count * wordSize(type);
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<nRanks; r++) {
      NCCLCHECK(ncclSend(sptr+r*rankOffset, count, type, r, comm, stream));
      NCCLCHECK(ncclRecv(rptr+r*rankOffset, count, type, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
#else
    printf("NCCL 2.7 or later is needed for alltoall. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#endif
  } else {
    switch(deviceImpl) {
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
      case 1:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(NvlAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
      case 2:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(NvlAlltoAllKernelOptimized, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
      case 3: {
        auto ginKernel = SPECIALIZE_KERNEL(GinAlltoAllKernel, type, op);
        if (ginKernel == nullptr) return testNotImplemented;
        ncclDevComm* devComm = (ncclDevComm*)comm;
        ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
        ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
        constexpr int kGinCTAs = 1;
        constexpr int kGinThreads = 512;
        ginKernel<<<kGinCTAs, kGinThreads, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm);
        return testSuccess;
      }
      case 4:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(HybridAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
      case 5: {
        auto ginAdaptiveKernel = SPECIALIZE_KERNEL(GinAdaptiveAlltoAllKernel, type, op);
        if (ginAdaptiveKernel == nullptr) return testNotImplemented;
        ncclDevComm* devComm = (ncclDevComm*)comm;
        ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
        ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
        size_t rankSliceBytes = count * wordSize(type);
        // Latency-bound slices: one CTA avoids 16x LSA barrier tax; large slices keep peer striping.
        int ctas = (rankSliceBytes < kAlltoAllSmallMessageRankSliceBytes) ? 1 : deviceCtaCount;
        constexpr int kGinAdaptiveThreads = 512;
        ginAdaptiveKernel<<<ctas, kGinAdaptiveThreads, 0, stream>>>(
            sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm);
        return testSuccess;
      }
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

struct testColl alltoAllTest = {
  "AlltoAll",
  AlltoAllGetCollByteCount,
  AlltoAllInitData,
  AlltoAllGetBw,
  AlltoAllRunColl,
  NULL
};

void AlltoAllGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAllGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AlltoAllRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &alltoAllTest;
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

struct testEngine ncclTestEngine = {
  .getBuffSize = AlltoAllGetBuffSize,
  .runTest = AlltoAllRunTest,
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  .getDevCommRequirements = AlltoAllGetDevCommRequirements
#endif
};
