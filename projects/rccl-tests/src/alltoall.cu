/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#include "gin_sdma_devtime.h"  // shared device-side (wall_clock64) timing scaffold
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#include "rccl_vector_types.h"
#endif

#if defined(NCCL_OS_LINUX)
#pragma weak ncclAlltoAll
#endif

void AlltoAllGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *paramcount = (count/nranks) & ~(16/eltSize - 1);
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

void AlltoAllGetBw(size_t count, size_t typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
// set devComm reqs for alltoall device kernels
testResult_t AlltoAllGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclComm_t comm) {
  if (!reqs || !comm) return testInternalError;

  ncclCommProperties_t commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
  if (ncclCommQueryProperties(comm, &commProperties) != ncclSuccess) {
    return testNcclError;
  }

  switch(deviceImpl) {
    case 1: // NvlAlltoAllKernel
    case 2: // NvlAlltoAllKernelOptimized
      if (commProperties.nRanks != ncclTeamLsa(comm).nRanks) {
        fprintf(stderr, "DeviceImplementation 1 and 2 requires CUDA P2P "
                        "connectivity across all ranks. Not all ranks of this "
                        "communicator have P2P connectivity.\n");
        return testInvalidUsage;
      }
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
    #if defined(NCCL_OS_LINUX)
    case 3: // GinAlltoAllKernel: all CTAs participate
      if (commProperties.ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInvalidUsage;
      }
      reqs->barrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
      reqs->ginForceEnable = true;
#endif
      return testSuccess;
    case 4: // HybridAlltoAllKernel: CTA 0 = GIN, CTAs 1..N = LSA
      if (deviceCtaCount < 2) {
        fprintf(stderr, "HybridAlltoAllKernel requires at least 2 CTAs.\n");
        return testInvalidUsage;
      }
      if (commProperties.ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInvalidUsage;
      }
      // barrierCount provisions per-CTA hybrid world barriers (LSA+rail+world) used
      // by HybridAlltoAllTimedKernel to join CTA 0 (GIN) with LSA CTAs each iter.
      // The production body still uses barrier index 0 on CTA 0 and lsaBarrier on
      // CTAs 1..N; ginSignalCount stays 1 because only CTA 0 issues GIN puts.
      reqs->barrierCount = deviceCtaCount;
      reqs->lsaBarrierCount = deviceCtaCount - 1;
      reqs->ginSignalCount = 1;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
      reqs->ginForceEnable = true;
#endif
      return testSuccess;
    #endif
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
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7) && defined(NCCL_OS_LINUX)
    case 3: // GinAlltoAllKernel: all CTAs participate
      reqs->barrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
      return true;
    case 4: // HybridAlltoAllKernel: CTA 0 = GIN, CTAs 1..N = LSA
      if (deviceCtaCount < 2) return false;
      reqs->barrierCount = deviceCtaCount;
      reqs->lsaBarrierCount = deviceCtaCount - 1;
      reqs->ginSignalCount = 1;
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
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  int rank = devComm.rank, nRanks = devComm.nRanks;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int nthreads = blockDim.x * gridDim.x;

  AlltoAllScalarImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

// Device implementation #2 - optimized NVL kernel using vectorization and unrolling
template <typename T>
__global__ void NvlAlltoAllKernelOptimized(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  using TN = typename VectorTypeMapping<T>::Type;
  constexpr int VECTOR_FACTOR = sizeof(TN) / sizeof(T);
  constexpr int UNROLL_FACTOR = 128/sizeof(TN);
  constexpr int PEER_UNROLL = 2;

  int rank = devComm.rank, nRanks = devComm.nRanks;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int nthreads = blockDim.x * gridDim.x;

  T* sendPtr = (T*)ncclGetLsaPointer(sendwin, sendoffset, rank);

  // alignment check: can we use vectorized operations?
  bool canVectorize = (sizeof(TN) > sizeof(T)) &&  // Only if vectorization helps
                      (reinterpret_cast<uintptr_t>(sendPtr) % sizeof(TN) == 0) &&  // Base aligned
                      ((count * sizeof(T)) % sizeof(TN) == 0);  // Stride compatible

  if (canVectorize) {
    size_t vector_count = count / VECTOR_FACTOR;
    int elements_per_iteration = nthreads * UNROLL_FACTOR;

    // process aligned vectorized elements without bounds checks
    size_t aligned_vector_count = (vector_count / elements_per_iteration) * elements_per_iteration;
    for (size_t base_offset = tid; base_offset < aligned_vector_count; base_offset += elements_per_iteration) {
      // unroll a limited number of peers at a time
      for (int peerBase = 0; peerBase < nRanks; peerBase += PEER_UNROLL) {
        int peersInGroup = min(PEER_UNROLL, nRanks - peerBase);

        #pragma unroll
        for (int p = 0; p < peersInGroup; p++) {
          int peer = peerBase + p;
          TN* sendVecPtr = (TN*)(sendPtr + peer * count);
          TN* recvVecPtr = (TN*)((T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + rank * count);
          TN values[UNROLL_FACTOR];

          // split load/store into separate loops for better overlap and ILP
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

    // handle remaining vectorized elements that didn't fit in aligned chunks
    for (size_t base_offset = aligned_vector_count + tid; base_offset < vector_count; base_offset += nthreads) {
      for (int peer = 0; peer < nRanks; peer++) {
        TN* sendVecPtr = (TN*)(sendPtr + peer * count);
        TN* recvVecPtr = (TN*)((T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + rank * count);
        recvVecPtr[base_offset] = sendVecPtr[base_offset];
      }
    }

    // handle any remaining elements not divisible by vectorization factor
    size_t scalar_start = vector_count * VECTOR_FACTOR;
    for (size_t offset = scalar_start + tid; offset < count; offset += nthreads) {
      for (int peer = 0; peer < nRanks; peer++) {
        T value = sendPtr[peer * count + offset];
        T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer);
        recvPtr[rank * count + offset] = value;
      }
    }
  } else {
    // simple scalar fallback for unaligned data (identical to simple kernel)
    AlltoAllScalarImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);
  }

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7) && defined(NCCL_OS_LINUX)
// Collective body, factored out of the __global__ entry so it can be invoked
// either once (production kernel) or in a persistent skip+loop (device-timing
// kernel). It re-derives all per-call sync state at entry (GIN re-reads the
// accumulated signal), so calling it back-to-back in a loop yields a sequence of
// complete, correct alltoalls with no external bookkeeping.
template <typename T>
__device__ void ginAlltoAllBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  int ginContext = 0;
  unsigned int signalIndex = blockIdx.x;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  /* send to all peers via GIN; the Anvil-SDMA backend segments large puts
   * internally (<=128 MiB per SDMA copy) so a single gin.put() is safe at any
   * size and needs no application-side chunking. */
  const size_t size = count * sizeof(T);
  for (int r=tid; r<devComm.nRanks; r+=nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,
        sendwin, sendoffset + r * size,
        size, ncclGin_SignalInc{signalIndex});
  }

  int receivingCta = (devComm.rank % nthreads) / blockDim.x;
  if (blockIdx.x == receivingCta)
    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());

  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

// Production entry: one collective call. Thin wrapper over the body so the
// device-timed kernel and the production path share identical code.
template <typename T>
__global__ void GinAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ginAlltoAllBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm);
}

// Hybrid LSA+GIN alltoall: CTA 0 handles remote peers via GIN,
// CTAs 1..N handle intra-node peers via LSA.
// Production body: CTA 0 uses hybrid world barrier index 0; LSA CTAs use
// lsaBarrier (blockIdx.x - 1). DevComm barrierCount == deviceCtaCount also
// provisions per-CTA hybrid world barriers for HybridAlltoAllTimedKernel joins.
template <typename T>
__device__ void hybridAlltoAllBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclTeam world = ncclTeamWorld(devComm);
  ncclTeam lsa = ncclTeamLsa(devComm);
  const int startLsa = world.rank - lsa.rank;
  const int lsaSize  = lsa.nRanks;
  const size_t size = count * sizeof(T);
  int numRemotePeers = world.nRanks - lsa.nRanks;

  if (blockIdx.x == 0) {
    /* CTA 0: remote peers via GIN */
    int ginContext = 0;
    unsigned int signalIndex = 0;
    ncclGin gin { devComm, ginContext };
    uint64_t signalValue = gin.readSignal(signalIndex);

    ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, 0 };
    bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    for (int r = tid; r < startLsa; r += nthreads) {
      gin.put(world, r,
          recvwin, recvoffset + world.rank * size,
          sendwin, sendoffset + r * size,
          size, ncclGin_SignalInc{signalIndex});
    }
    for (int r = startLsa + lsaSize + tid; r < world.nRanks; r += nthreads) {
      gin.put(world, r,
          recvwin, recvoffset + world.rank * size,
          sendwin, sendoffset + r * size,
          size, ncclGin_SignalInc{signalIndex});
    }

    if (numRemotePeers > 0)
      gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + numRemotePeers);
    gin.flush(ncclCoopCta());
    bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
  } else {
    /* CTAs 1..N: local peers via LSA */
    ncclLsaBarrierSession<ncclCoopCta> lsaBar {
      ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x - 1
    };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_acquire);

    int tid = threadIdx.x + (blockIdx.x - 1) * blockDim.x;
    int nthreads = blockDim.x * (gridDim.x - 1);
    T* sendLocal = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    for (size_t offset = tid; offset < count; offset += nthreads) {
      for (int lp = 0; lp < lsa.nRanks; lp++) {
        int wr = startLsa + lp;
        T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp);
        recvPtr[world.rank * count + offset] = sendLocal[wr * count + offset];
      }
    }
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
  }
}

// Production entry: one collective call. Thin wrapper over the body so the
// device-timed kernel and the production path share identical code.
template <typename T>
__global__ void HybridAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  hybridAlltoAllBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm);
}

// Join all CTAs (GIN CTA 0 + LSA CTAs 1..N) in the timed-kernel loop. The
// production body uses split sync domains; ncclBarrierSession(world) spans LSA,
// rail, and world GIN barriers so every CTA waits together (see NCCL hybrid
// alltoall example). Requires barrierCount == gridDim.x in devComm setup.
__device__ inline void hybridTimedGridJoin(struct ncclDevComm devComm) {
  int ginContext = 0;
  ncclGin gin { devComm, ginContext };
  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, (uint32_t)blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);
}

// Device-side timing kernels (rocSHMEM AllToAll methodology, AICOMRCCL-1459).
// A single persistent launch runs (skip + loop) back-to-back collective bodies
// and brackets only the timed region with the GPU fixed-frequency wall clock
// (wall_clock64()), so the reported span excludes host launch, teardown, and
// stream/graph overhead -- it is the pure device-function execution time.
//
// skip warmup iterations run first (steady-state caches/queues, discarded).
// At i == skip every CTA records its start stamp; after the final iteration
// every CTA records its end stamp. The host reduces min(start)/max(end) across
// CTAs (the grid's true busy window) and MAX across ranks (the slowest rank
// closes the collective), then divides by loop for per-iteration latency.
//
// Because the body re-derives all per-call sync state at entry (GIN re-reads the
// accumulated signal; LSA rebuilds its barrier session), looping is correct with
// no extra signal bookkeeping.
template <typename T>
__global__ void GinAlltoAllTimedKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int loop, int skip, long long* start_time, long long* end_time) {
  for (int i = 0; i < skip + loop; i++) {
    if (i == skip) {
      __syncthreads();
      if (threadIdx.x == 0) start_time[blockIdx.x] = wall_clock64();
    }
    ginAlltoAllBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm);
  }
  __syncthreads();
  if (threadIdx.x == 0) end_time[blockIdx.x] = wall_clock64();
}

template <typename T>
__global__ void HybridAlltoAllTimedKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int loop, int skip, long long* start_time, long long* end_time) {
  for (int i = 0; i < skip + loop; i++) {
    if (i == skip) {
      hybridTimedGridJoin(devComm);
      __syncthreads();
      if (threadIdx.x == 0) start_time[blockIdx.x] = wall_clock64();
    }
    hybridAlltoAllBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm);
  }
  hybridTimedGridJoin(devComm);
  __syncthreads();
  if (threadIdx.x == 0) end_time[blockIdx.x] = wall_clock64();
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
    // fall-through to send/recv implementation if ncclAlltoAll is not available
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
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7) && defined(NCCL_OS_LINUX)
      case 3:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(GinAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
      case 4:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(HybridAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

// Device-side (in-kernel wall_clock64) timing for AllToAll (GIN and Hybrid
// tiers). Opt-in via --device_timing: launches the persistent timed kernel once
// for the current size, brackets only the (skip+loop) steady-state collectives
// with the GPU wall clock, reduces the grid busy window (min start .. max end
// over CTAs) and the slowest rank (MPI MAX), and reports the per-iteration
// device latency. loop/skip come from --devtime_loop/--devtime_skip (default
// 10/10); size-tier overrides via --devtime_loop_mid/_large and
// --devtime_skip_mid/_large. The timed kernel launches with the same
// <<<deviceCtaCount, 512>>> grid the production path uses, and selects the
// GIN (deviceImpl 3) or Hybrid (deviceImpl 4) body to match RunColl.
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
testResult_t AlltoAllDeviceTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec) {
  if (!deviceTimingMode) return testSuccess;

  // Only the GIN (deviceImpl 3) and Hybrid (deviceImpl 4) tiers provision the GIN
  // signals/barriers the timed bodies rely on. The Nvl LSA impls (-D1/-D2) only
  // provision the LSA barrier, so running the GIN timed kernel there would fault
  // on an out-of-range signal index or wait forever -- skip them.
  if (deviceImpl != 3 && deviceImpl != 4) return testSuccess;

  const size_t count = args->nbytes / wordSize(type);
  if (count == 0 || devtimeLoop < 1) return testSuccess;
  const size_t perPeerBytes = count * wordSize(type);

  // By default use the exact skip/loop counts at every size so the device-timing
  // window matches the host tests' -w/-n. Optional mid/large tier overrides cap
  // iteration counts for very large chunks (bounded runtime; per-call copy dominates).
  // Clamp skip to >= 0: the timed kernel only stamps start_time[] at i == skip, so
  // a negative skip would leave start_time[] as uninitialized cudaMalloc memory and
  // measure() would reduce mx - mn over garbage (reported as the mode-2 metric).
  int loop = devtimeLoop;
  int skip = devtimeSkip < 0 ? 0 : devtimeSkip;
  if (devtimeLoopLarge > 0 && perPeerBytes >= (size_t)64 * 1024 * 1024) {
    loop = devtimeLoopLarge;
    if (devtimeSkipLarge >= 0) skip = devtimeSkipLarge;
    else skip = (skip < 1) ? skip : 1;
  } else if (devtimeLoopMid > 0 && perPeerBytes >= (size_t)8 * 1024 * 1024) {
    loop = devtimeLoopMid;
    if (devtimeSkipMid >= 0) skip = devtimeSkipMid;
    else skip = (skip < 2) ? skip : 2;
  }

  int gridCtas = deviceCtaCount;
  if (gridCtas < 1) gridCtas = 1;

  // Match the kernel RunColl would launch for the selected -D device impl.
  const bool hybrid = (deviceImpl == 4);
  auto kernel = hybrid ? SPECIALIZE_KERNEL(HybridAlltoAllTimedKernel, type, op)
                       : SPECIALIZE_KERNEL(GinAlltoAllTimedKernel, type, op);
  const char* tierName = hybrid ? "HYB" : "GIN";
  if (kernel == nullptr) return testSuccess;

  // Shared scaffold: allocates per-CTA start/end stamps, launches the timed kernel,
  // reduces min(start)..max(end) over CTAs and MPI-MAX across ranks -> per-iter us.
  double devUs = 0.0;
  TESTCHECK(gin_devtime::measure(args, gridCtas, loop,
      [&](int i, long long* d_start, long long* d_end) {
        ncclDevComm* devComm = args->devComms + i;
        ncclWindow_t sendwin = (ncclWindow_t)args->sendRegHandles[i];
        ncclWindow_t recvwin = (ncclWindow_t)args->recvRegHandles[i];
        kernel<<<gridCtas, 512, 0, args->streams[i]>>>(
            sendwin, 0, recvwin, 0, count, root, *devComm, loop, skip, d_start, d_end);
      },
      &devUs));

  int nRanksGlobal = args->nProcs * args->nThreads * args->nGpus;

  // Mode 2 (device-time-only): hand the per-iteration latency (seconds) back to
  // BenchTime, which reports it as THE time/busbw metric on the normal result
  // line. Stay silent here so that line is not split.
  if (outDeltaSec != nullptr) {
    // Negative sentinel = "no valid measurement" (the timed grid produced no
    // positive busy window). The caller (BenchTime mode 2) then WARNs and skips the
    // row instead of reporting a bogus 0.00 us / inf GB/s.
    *outDeltaSec = (devUs > 0.0) ? devUs * 1.0e-6 : -1.0;
    return testSuccess;
  }

  // Mode 1 (augment): buffer the extra device-only line; TimeTest flushes it after
  // writeBenchmarkLineTerminator so the row is not split between OOP/IP columns.
  if (args->proc == 0 && args->thread == 0 && devUs > 0.0) {
    double sec = devUs * 1.0e-6;
    double algBw = (double)(perPeerBytes * (size_t)nRanksGlobal) / 1.0e9 / sec;
    double busBw = algBw * ((double)(nRanksGlobal - 1) / (double)nRanksGlobal);
    snprintf(args->devtimeAugmentLine, sizeof(args->devtimeAugmentLine),
             "#[a2a-devtime] size %12zu B  tier %-3s  ctas %2d  loop %2d skip %2d  devtime %10.2f us  algbw %8.2f GB/s  busbw %8.2f GB/s\n",
             perPeerBytes * (size_t)nRanksGlobal, tierName, gridCtas, loop, skip, devUs, algBw, busBw);
  }
  return testSuccess;
}
#else
testResult_t AlltoAllDeviceTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec) {
  return testSuccess;  // device API path not available in this build
}
#endif

struct testColl alltoAllTest = {
  "AlltoAll",
  AlltoAllGetCollByteCount,
  AlltoAllInitData,
  AlltoAllGetBw,
  AlltoAllRunColl,
  NULL,
  NULL,
  NULL,
  AlltoAllDeviceTime
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

NCCL_WEAK struct testEngine ncclTestEngine = {
  /* .getBuffSize = */ AlltoAllGetBuffSize,
  /* .runTest = */ AlltoAllRunTest,
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
  /* .initCommConfig = */ nullptr,
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0) || (defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0))
  /* .getDevCommRequirements = */ AlltoAllGetDevCommRequirements
#endif
};
