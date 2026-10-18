/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#include "rccl_vector_types.h"
#if HAVE_FP8
#include "rccl_float8.h"
#endif
#include "gin_sdma_reduce.h"  // device reduction ops bit-matching verifiable.cu
#include "gin_sdma_devtime.h" // shared device-side (wall_clock64) timing scaffold
#endif

// GIN-SDMA Reduce (-D 3) is a "reduce-scatter to the root": rank r read-reduces
// the slice [r*base] directly from EVERY peer's sendbuff via ncclGetLsaPointer,
// then stores its reduced slice into the ROOT's recvbuff at [r*base]. The root's
// recvbuff ends up holding the full reduced array while every non-root recvbuff
// is left untouched (out-of-place: still zero; in-place: still its input), which
// is what ReduceInitData expects. Work and read egress are spread across all N
// ranks; only the final store targets the root. Entry + exit LSA barrier only --
// no scratch, signals, or GIN puts. In-place safe: rank r is the sole reader and
// writer of the root's [r*base] slice and loads all N sources before storing.
//
// SDMA Reduce (-D 4, prototype) moves the data with SDMA and reduces with CUs.
// The N-1 non-root ranks each own one of N-1 equal partitions. Every rank
// gin.put()s its slice of each partition into the owner's recvbuff, one slot per
// sender (slot = sender < owner ? sender : sender - 1); the owner then folds its
// own sendbuff slice and the N-1 received slots in ascending source-rank order
// and stores the result into the root's recvbuff. Non-root recvbuffs double as
// the receive scratch, so -D 4 is out-of-place only (-O 1). Large partitions go
// out as several puts (chunks), each followed by a per-chunk GIN signal; every
// reducing CTA folds its share of chunk k as soon as chunk k's signal has
// counted all N-1 senders, so the reduce overlaps the rest of the copy. With
// at least 2N CTAs the owner stages its reduced chunks in the root's slot and
// SDMAs them to the root; with fewer, it stores them to the root from CUs.
// Each peer's partition is put by a single thread: the anvil SDMA queue is per
// peer, and concurrent producers on different XCDs can ring its doorbell out of
// order, leaving committed packets unprocessed (the put hangs in quiet).
// Because the issuing CTA differs from the reducing CTA, a CTA's only guarantee
// that the previous Reduce finished with its slots is the kernel boundary, so
// -D 4 has no persistent device-timed variant.

void ReduceGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  size_t aligned = count;
  // -D 3 splits the array into nranks slices and -D 4 into nranks-1 partitions,
  // all of whole 128-bit packs with no scalar tail, so round the count down to a
  // multiple of nParts*16 bytes. The host path keeps the exact requested count.
  const int nParts = (deviceImpl == 4) ? nranks - 1 : nranks;
  if (deviceImpl != 0 && nParts > 0) {
    size_t base = (count / (size_t)nParts) & -(16 / eltSize);
    aligned = base * (size_t)nParts;
  }
  *sendcount = aligned;
  *recvcount = aligned;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = aligned;
}

testResult_t ReduceInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, op, rep, nranks, rank));
    CUDACHECK(cudaMemcpy(args->expected[i], args->recvbuffs[i], args->expectedBytes, cudaMemcpyDefault));
    if (rank == root) TESTCHECK(InitDataReduce(args->expected[i], recvcount, 0, type, op, rep, nranks));
    else if (deviceImpl == 4 && !in_place && nranks > 1) {
      // -D 4 leaves each sender's slice of this rank's partition in its slot.
      const size_t part = recvcount / (size_t)(nranks - 1);
      const int owner = (rank - root - 1 + nranks) % nranks;
      // With issuer CTAs, the root's slot holds the reduced partition instead.
      const bool staged = deviceCtaCount >= 2 * nranks;
      for (int sender = 0; sender < nranks; sender++) {
        if (sender == rank) continue;
        const int slot = sender < rank ? sender : sender - 1;
        void* slotPtr = (char*)args->expected[i] + (size_t)slot * part * wordSize(type);
        if (staged && sender == root) TESTCHECK(InitDataReduce(slotPtr, part, (size_t)owner * part, type, op, rep, nranks));
        else TESTCHECK(InitData(slotPtr, part, (size_t)owner * part, type, op, rep, nranks, sender));
      }
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

testResult_t  ReduceGetAlgoProtoChannels(ncclComm_t comm, size_t count, ncclDataType_t type, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetAlgoInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetAlgoInfo(comm, ncclFuncReduce , count, type , 0, 0, 1, algo, proto, nchannels));
  return testSuccess;
}


void ReduceGetBw(size_t count, size_t typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;
  *algBw = baseBw;
  *busBw = baseBw;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
testResult_t ReduceGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclComm_t comm) {
  if (!reqs || !comm) return testInternalError;

  ncclCommProperties_t commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
  if (ncclCommQueryProperties(comm, &commProperties) != ncclSuccess) {
    return testNcclError;
  }

  if (commProperties.nRanks != ncclTeamLsa(comm).nRanks) {
    fprintf(stderr, "GinReduceKernel requires CUDA P2P connectivity across all ranks. "
                    "Not all ranks of this communicator have P2P connectivity.\n");
    return testInvalidUsage;
  }

  switch(deviceImpl) {
    case 3:   // GinReduceKernel: reduce-scatter-to-root LSA read-reduce (no scratch)
    case 4: { // GinSdmaReduceKernel: SDMA puts to partition owners + CU reduce
      if (deviceImpl == 4 && commProperties.nRanks < 2) {
        fprintf(stderr, "Reduce -D 4 needs at least 2 ranks.\n");
        return testInvalidUsage;
      }
      if (commProperties.ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInvalidUsage;
      }
#ifdef NCCL_GIN_BACKEND_MASK_ALL
      TESTCHECK(testCheckGinBackendCompiled(commProperties.ginType));
#endif
      reqs->barrierCount = deviceCtaCount;
      reqs->lsaBarrierCount = deviceCtaCount;
      // -D 4 alternates between two sets of per-chunk signals.
      reqs->ginSignalCount = (deviceImpl == 4 ? 2 : 1) * deviceCtaCount;
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
bool ReduceGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));

  switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3:   // GinReduceKernel: reduce-scatter-to-root LSA read-reduce (no scratch)
    case 4: { // GinSdmaReduceKernel: SDMA puts to partition owners + CU reduce
      reqs->barrierCount = deviceCtaCount;
      reqs->lsaBarrierCount = deviceCtaCount;
      reqs->ginSignalCount = (deviceImpl == 4 ? 2 : 1) * deviceCtaCount;
      return true;
    }
#endif
    default:
      return false;
  }
}
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
// `count` is the FULL element count (== base*nRanks, aligned by
// ReduceGetCollByteCount). Small/mid messages use a register-light grid-stride
// loop with 2-way peer ILP; large messages use a warp-strided pack-unrolled loop
// (U outstanding coalesced loads) with 2-way peer ILP and a source-0 prefetch of
// the next tile. Both schedules fold sources in ascending rank order.
template <typename T>
__device__ __forceinline__ void ginReduceBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp) {
  const int nRanks = devComm.nRanks;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
  const size_t base = (nRanks > 0) ? (count / (size_t)nRanks) : count;  // this rank's slice length

  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
  lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  // dstP is the root's full recvbuff (self for the root rank), so stores are
  // indexed by the global pack offset myBaseP+pk.
  constexpr int VEC = (sizeof(T) <= 16) ? (int)(16 / sizeof(T)) : 1;
  struct alignas(16) Pack { T e[VEC]; };
  Pack* dstP = (Pack*)ncclGetLsaPointer(recvwin, recvoffset, root);
  const size_t nPacks = base / (size_t)VEC;                           // packs in this rank's slice
  const size_t myBaseP = ((size_t)devComm.rank * base) / (size_t)VEC; // pack index of my slice

  constexpr size_t UNROLL_MIN_BYTES = (size_t)48 << 20;  // total message size
  const size_t totalBytes = base * (size_t)nRanks * sizeof(T);

  if (totalBytes < UNROLL_MIN_BYTES) {
    for (size_t pk = (size_t)tid; pk < nPacks; pk += (size_t)nthreads) {
      Pack v0 = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, 0))[myBaseP + pk];
      T acc[VEC];
      #pragma unroll
      for (int e = 0; e < VEC; e++) acc[e] = gin_sdma_reduce::preOp(redOp, v0.e[e], nRanks);
      int s = 1;
      for (; s + 1 < nRanks; s += 2) {
        Pack a = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s))[myBaseP + pk];
        Pack b = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s + 1))[myBaseP + pk];
        #pragma unroll
        for (int e = 0; e < VEC; e++)
          acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, a.e[e], nRanks));
        #pragma unroll
        for (int e = 0; e < VEC; e++)
          acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, b.e[e], nRanks));
      }
      for (; s < nRanks; s++) {
        Pack vs = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s))[myBaseP + pk];
        #pragma unroll
        for (int e = 0; e < VEC; e++)
          acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, vs.e[e], nRanks));
      }
      Pack o;
      #pragma unroll
      for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], nRanks);
      dstP[myBaseP + pk] = o;
    }
  } else {
    constexpr int U = (VEC <= 8) ? 4 : 2;
    constexpr int WARP = 64;  // CDNA wavefront
    const int lane = tid & (WARP - 1);
    const size_t warpId = (size_t)(tid / WARP);
    const size_t nWarps = (size_t)(nthreads / WARP);
    const size_t tile = (size_t)U * WARP;
    const size_t gridStride = nWarps * tile;

    const Pack* src0Base = (const Pack*)ncclGetLsaPointer(sendwin, sendoffset, 0) + myBaseP;
    size_t wbase = warpId * tile;
    Pack seed[U];
    if (wbase + tile <= nPacks) {
      const Pack* sp = src0Base + wbase + (size_t)lane;
      #pragma unroll
      for (int u = 0; u < U; u++) seed[u] = sp[(size_t)u * WARP];
    }
    for (; wbase < nPacks; wbase += gridStride) {
      if (wbase + tile <= nPacks) {
        const size_t p0 = wbase + (size_t)lane;  // this lane's first pack (slice-local)
        T acc[U][VEC];
        #pragma unroll  // source 0 comes from the prefetched seed
        for (int u = 0; u < U; u++)
          #pragma unroll
          for (int e = 0; e < VEC; e++) acc[u][e] = gin_sdma_reduce::preOp(redOp, seed[u].e[e], nRanks);
        int s = 1;
        for (; s + 1 < nRanks; s += 2) {
          const Pack* sa = (const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s)     + myBaseP + p0;
          const Pack* sb = (const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s + 1) + myBaseP + p0;
          Pack ta[U], tb[U];
          #pragma unroll
          for (int u = 0; u < U; u++) ta[u] = sa[(size_t)u * WARP];
          #pragma unroll
          for (int u = 0; u < U; u++) tb[u] = sb[(size_t)u * WARP];
          #pragma unroll
          for (int u = 0; u < U; u++)
            #pragma unroll
            for (int e = 0; e < VEC; e++)
              acc[u][e] = gin_sdma_reduce::combine(redOp, acc[u][e], gin_sdma_reduce::preOp(redOp, ta[u].e[e], nRanks));
          #pragma unroll
          for (int u = 0; u < U; u++)
            #pragma unroll
            for (int e = 0; e < VEC; e++)
              acc[u][e] = gin_sdma_reduce::combine(redOp, acc[u][e], gin_sdma_reduce::preOp(redOp, tb[u].e[e], nRanks));
        }
        for (; s < nRanks; s++) {
          const Pack* sp = (const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s) + myBaseP + p0;
          Pack t[U];
          #pragma unroll
          for (int u = 0; u < U; u++) t[u] = sp[(size_t)u * WARP];
          #pragma unroll
          for (int u = 0; u < U; u++)
            #pragma unroll
            for (int e = 0; e < VEC; e++)
              acc[u][e] = gin_sdma_reduce::combine(redOp, acc[u][e], gin_sdma_reduce::preOp(redOp, t[u].e[e], nRanks));
        }
        const size_t nb = wbase + gridStride;
        if (nb + tile <= nPacks) {
          const Pack* spn = src0Base + nb + (size_t)lane;
          #pragma unroll
          for (int u = 0; u < U; u++) seed[u] = spn[(size_t)u * WARP];
        }
        #pragma unroll
        for (int u = 0; u < U; u++) {
          Pack o;
          #pragma unroll
          for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[u][e], nRanks);
          dstP[myBaseP + p0 + (size_t)u * WARP] = o;
        }
      } else {
        // Partial warp tile: each lane covers packs wbase+u*WARP+lane.
        #pragma unroll
        for (int u = 0; u < U; u++) {
          const size_t pk = wbase + (size_t)u * WARP + (size_t)lane;
          if (pk >= nPacks) continue;
          Pack v = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, 0))[myBaseP + pk];
          T acc[VEC];
          #pragma unroll
          for (int e = 0; e < VEC; e++) acc[e] = gin_sdma_reduce::preOp(redOp, v.e[e], nRanks);
          for (int s = 1; s < nRanks; s++) {
            Pack vs = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s))[myBaseP + pk];
            #pragma unroll
            for (int e = 0; e < VEC; e++)
              acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, vs.e[e], nRanks));
          }
          Pack o;
          #pragma unroll
          for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], nRanks);
          dstP[myBaseP + pk] = o;
        }
      }
    }
  }

  lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template <typename K>
__global__ void GinReduceKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp) {
  ginReduceBody<typename gin_sdma_reduce::Elem<K>::type>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm, redOp);
}

// Runs skip+loop back-to-back Reduce bodies under one launch, bracketing only the
// timed region with wall_clock64() per CTA. Each body runs its own entry/exit LSA
// barrier, so iterations stay ordered without extra bookkeeping.
template <typename K>
__global__ void GinReduceTimedKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp, int loop, int skip, long long* start_time, long long* end_time) {
  for (int i = 0; i < skip + loop; i++) {
    if (i == skip) {
      __syncthreads();
      if (threadIdx.x == 0) start_time[blockIdx.x] = wall_clock64();
    }
    ginReduceBody<typename gin_sdma_reduce::Elem<K>::type>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm, redOp);
  }
  __syncthreads();
  if (threadIdx.x == 0) end_time[blockIdx.x] = wall_clock64();
}

// Folds U packs per thread from NR sources in ascending source-rank order and
// stores them, issuing all U*NR loads before the first combine.
template <typename T, int NR, int U, typename Pack>
__device__ __forceinline__ void sdmaFoldPacks(const Pack* const* src, Pack* dst, size_t first, size_t end, size_t stride, int redOp) {
  constexpr int VEC = (int)(sizeof(Pack) / sizeof(T));
  for (size_t base = first; base < end; base += stride * U) {
    Pack v[U][NR];
    #pragma unroll
    for (int u = 0; u < U; u++) {
      const size_t pk = base + (size_t)u * stride;
      if (pk < end) {
        #pragma unroll
        for (int r = 0; r < NR; r++) v[u][r] = src[r][pk];
      }
    }
    #pragma unroll
    for (int u = 0; u < U; u++) {
      const size_t pk = base + (size_t)u * stride;
      if (pk >= end) continue;
      T acc[VEC];
      #pragma unroll
      for (int r = 0; r < NR; r++) {
        #pragma unroll
        for (int e = 0; e < VEC; e++) {
          const T x = gin_sdma_reduce::preOp(redOp, v[u][r].e[e], NR);
          acc[e] = (r == 0) ? x : gin_sdma_reduce::combine(redOp, acc[e], x);
        }
      }
      Pack o;
      #pragma unroll
      for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], NR);
      dst[pk] = o;
    }
  }
}

// Per-chunk count of reducing CTAs that finished, in two sets alternated by
// launch parity like the chunk signals. Device globals are per GPU, and each
// GPU runs at most one -D 4 Reduce at a time.
__device__ unsigned int gSdmaReduceChunkDone[2][128];

// `count` is the FULL element count (== part*(nRanks-1), aligned by
// ReduceGetCollByteCount).
template <typename T>
__device__ __forceinline__ void ginSdmaReduceBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp, int dmaChunks) {
  const int nRanks = devComm.nRanks;
  const int me = devComm.rank;
  const int nOwners = nRanks - 1;
  const size_t partBytes = (count / (size_t)nOwners) * sizeof(T);
  constexpr int VEC = (sizeof(T) <= 16) ? (int)(16 / sizeof(T)) : 1;
  struct alignas(16) Pack { T e[VEC]; };
  const size_t partPacks = partBytes / sizeof(Pack);
  const int G = (int)gridDim.x;

  ncclGin gin { devComm, 0 };
  ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  // Every launch syncs each barrier exactly twice, so the epoch gives every CTA
  // on every rank the same launch parity. Launches alternate between two sets
  // of G signals: this one counts chunks in set `sigBase` and clears the other
  // set, which the previous launch has finished waiting on and which the next
  // launch's senders cannot touch until this kernel has ended.
  const unsigned int par = (lsaBar.epoch >> 1) & 1u;
  const unsigned int sigBase = par * (unsigned int)G;
  lsaBar.sync(ncclCoopCta(), cuda::memory_order_acquire);
  if (threadIdx.x == 0) gin.resetSignal((unsigned int)G - sigBase + blockIdx.x);
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < 128; j += G * blockDim.x) gSdmaReduceChunkDone[par ^ 1u][j] = 0;

  // With at least 2N CTAs, CTAs [0, N) only issue puts, so no chunk waits on a
  // CTA that is still busy draining queues. With fewer, CTA b issues for peers
  // q = b (mod G) and then reduces alongside the rest. Passing the barrier with
  // q's matching CTA means q has finished the previous Reduce, so its slots are
  // free.
  const int nIssue = (G >= 2 * nRanks) ? nRanks : 0;
  const int nReducers = G - nIssue;
  // With issuer CTAs, owners stage each reduced chunk in the root's slot (just
  // consumed by that reduce) and SDMA it to the root. Storing to the root from
  // CUs instead slows some of the concurrent owner-to-owner SDMA copies, and
  // each chunk waits for its slowest sender.
  if (nIssue && (int)blockIdx.x == root && me != root) {
    // CTA `root` has no owner-to-owner peer, so it feeds the owner-to-root
    // queue; it is that queue's only producer.
    if (threadIdx.x == 0) {
      const int owner = (me - root - 1 + nRanks) % nRanks;
      const int rootSlot = root < me ? root : root - 1;
      for (int k = 0; k < dmaChunks; k++) {
        const size_t lo = (size_t)k * partPacks / dmaChunks * sizeof(Pack);
        const size_t hi = (size_t)(k + 1) * partPacks / dmaChunks * sizeof(Pack);
        if (hi == lo) continue;
        while (__hip_atomic_load(&gSdmaReduceChunkDone[par][k], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < (unsigned int)nReducers) {}
        gin.put(ncclTeamWorld(devComm), root,
                recvwin, recvoffset + (size_t)owner * partBytes + lo,
                recvwin, recvoffset + (size_t)rootSlot * partBytes + lo,
                hi - lo);
      }
    }
  } else if (nIssue == 0 || (int)blockIdx.x < nIssue) {
    const int nMine = (nRanks - (int)blockIdx.x + G - 1) / G;
    if ((int)threadIdx.x < nMine) {
      const int q = blockIdx.x + threadIdx.x * G;
      if (q != root && q != me) {
        const int owner = (q - root - 1 + nRanks) % nRanks;
        const int slot = me < q ? me : me - 1;
        // The signal quiets the queue first, so put k+1 is only queued once
        // put k has landed and chunk k has been released to the owner.
        for (int k = 0; k < dmaChunks; k++) {
          const size_t lo = (size_t)k * partPacks / dmaChunks * sizeof(Pack);
          const size_t hi = (size_t)(k + 1) * partPacks / dmaChunks * sizeof(Pack);
          if (hi == lo) continue;
          gin.put(ncclTeamWorld(devComm), q,
                  recvwin, recvoffset + (size_t)slot * partBytes + lo,
                  sendwin, sendoffset + (size_t)owner * partBytes + lo,
                  hi - lo);
          gin.signal(ncclTeamWorld(devComm), q, ncclGin_SignalInc{sigBase + (unsigned int)k});
        }
      }
    }
  }

  if (me != root && (int)blockIdx.x >= nIssue) {
    const int owner = (me - root - 1 + nRanks) % nRanks;
    const char* slots = (const char*)ncclGetLocalPointer(recvwin, recvoffset);
    const Pack* own = (const Pack*)((const char*)ncclGetLocalPointer(sendwin, sendoffset) + (size_t)owner * partBytes);
    // Each pack of the root's slot is read and then overwritten by the same
    // thread, so staging the result there is safe.
    Pack* dst = nIssue ? (Pack*)(slots + (size_t)(root < me ? root : root - 1) * partBytes)
                       : (Pack*)((char*)ncclGetLsaPointer(recvwin, recvoffset, root) + (size_t)owner * partBytes);
    const size_t first = (size_t)((int)blockIdx.x - nIssue) * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)nReducers * blockDim.x;
    constexpr int NR8 = 8;
    const Pack* src8[NR8];
    if (nRanks == NR8) {
      #pragma unroll
      for (int r = 0; r < NR8; r++)
        src8[r] = (r == me) ? own : (const Pack*)(slots + (size_t)(r < me ? r : r - 1) * partBytes);
    }
    // Every reducing CTA takes a share of every chunk, so the chunk that lands
    // last is reduced by all of them rather than by a few.
    for (int k = 0; k < dmaChunks; k++) {
      const size_t p0 = (size_t)k * partPacks / dmaChunks;
      const size_t p1 = (size_t)(k + 1) * partPacks / dmaChunks;
      if (p1 == p0) continue;
      gin.waitSignal(ncclCoopCta(), sigBase + (unsigned int)k, (uint64_t)nOwners);
      // The slots were written by a remote SDMA engine, not by a CU, so every
      // thread needs system-scope acquire before loading them.
      cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
      if (nRanks == NR8) {
        sdmaFoldPacks<T, NR8, 2>(src8, dst, p0 + first, p1, stride, redOp);
      } else {
        for (size_t pk = p0 + first; pk < p1; pk += stride) {
          T acc[VEC];
          for (int r = 0; r < nRanks; r++) {
            const Pack v = (r == me) ? own[pk] : ((const Pack*)(slots + (size_t)(r < me ? r : r - 1) * partBytes))[pk];
            #pragma unroll
            for (int e = 0; e < VEC; e++) {
              const T x = gin_sdma_reduce::preOp(redOp, v.e[e], nRanks);
              acc[e] = (r == 0) ? x : gin_sdma_reduce::combine(redOp, acc[e], x);
            }
          }
          Pack o;
          #pragma unroll
          for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], nRanks);
          dst[pk] = o;
        }
      }
      if (nIssue) {
        // The SDMA engine reads the staged chunk from memory, not through L2.
        cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
        __syncthreads();
        if (threadIdx.x == 0) __hip_atomic_fetch_add(&gSdmaReduceChunkDone[par][k], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }
  // Flush drains this rank's SDMA queues, so the owner-to-root puts have landed
  // before the exit barrier releases the root.
  gin.flush(ncclCoopCta());
  lsaBar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);
}

template <typename K>
__global__ void GinSdmaReduceKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp, int dmaChunks) {
  ginSdmaReduceBody<typename gin_sdma_reduce::Elem<K>::type>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm, redOp, dmaChunks);
}

// SDMA puts per partition for -D 4. More chunks overlap the copies with the
// reduce and the owner-to-root copy, but each put waits for the previous one to
// drain, so the default grows with sqrt(partBytes): about sqrt(partBytes/512KiB),
// capped at 32. Partitions are only split when there are dedicated issuer CTAs
// (ctas >= 2*nRanks). NCCL_GIN_ANVIL_SDMA_REDUCE_CHUNKS overrides the default.
// Every rank must use the same value.
static int sdmaReduceDmaChunks(size_t partBytes, int ctas, int nRanks) {
  static const long envChunks = [] {
    const char* e = getenv("NCCL_GIN_ANVIL_SDMA_REDUCE_CHUNKS");
    if (e == nullptr || e[0] == '\0' || e[0] == '-') return 0L;
    char* end = nullptr;
    long v = strtol(e, &end, 10);
    return (end != e && *end == '\0' && v >= 1) ? v : 0L;
  }();
  long k = envChunks;
  if (k == 0 && ctas >= 2 * nRanks) {
    const size_t units = partBytes >> 19;
    while (k < 32 && (size_t)(k + 1) * (size_t)(k + 1) <= units) k++;
  }
  return (int)std::max(1L, std::min(k, (long)ctas));
}

// Like testLaunchDeviceKernel() but forwards the reduction op (and any extra
// kernel arguments), which the shared launcher's fixed signature does not carry.
template <typename F, typename... Extra>
static testResult_t ReduceLaunchDeviceKernel(F kernel, void* sendbuff, size_t sendoffset,
                                             void* recvbuff, size_t recvoffset, size_t count,
                                             ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream,
                                             Extra... extra) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  kernel<<<deviceCtaCount, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, (int)op, extra...);
  return testSuccess;
}
#endif

testResult_t ReduceRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
    NCCLCHECK(ncclReduce(sptr, rptr, count, type, op, root, comm, stream));
  } else {
    switch(deviceImpl) {
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
      case 3: {
        if (count == 0) return testSuccess;
        TESTCHECK(ReduceLaunchDeviceKernel(SPECIALIZE_REDUCE_KERNEL(GinReduceKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, op, root, comm, stream));
        return testSuccess;
      }
      case 4: {
        if (sendbuff == recvbuff && sendoffset == recvoffset) {
          fprintf(stderr, "Reduce -D 4 reuses recvbuff as receive scratch and is out-of-place only; run with -O 1.\n");
          return testInvalidUsage;
        }
        if (count == 0) return testSuccess;
        const int nRanks = ((ncclDevComm*)comm)->nRanks;
        const size_t partBytes = count / (size_t)(nRanks - 1) * wordSize(type);
        TESTCHECK(ReduceLaunchDeviceKernel(SPECIALIZE_REDUCE_KERNEL(GinSdmaReduceKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, op, root, comm, stream,
                                           sdmaReduceDmaChunks(partBytes, deviceCtaCount, nRanks)));
        return testSuccess;
      }
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

// Device-side (in-kernel wall_clock64) timing for the GIN Reduce (-D 3), via
// the shared gin_devtime scaffold. Opt-in via --device_timing: 1=augment (print an
// extra #[reduce-devtime] line next to the graph numbers), 2=device-time-only
// (report the in-kernel latency as the out-of-place metric; in-place keeps normal
// timing). loop/skip come from --devtime_loop/--devtime_skip and their size-tier
// overrides, keyed off the per-rank slice size.
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
testResult_t ReduceDeviceTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec) {
  if (!deviceTimingMode) return testSuccess;
  if (deviceImpl != 3) return testSuccess;

  const size_t count = args->nbytes / wordSize(type);   // full (aligned) element count
  if (count == 0 || devtimeLoop < 1) return testSuccess;

  auto kernel = SPECIALIZE_REDUCE_KERNEL(GinReduceTimedKernel, type, op);
  if (kernel == nullptr) return testSuccess;

  const int nRanksGlobal = args->nProcs * args->nThreads * args->nGpus;
  const size_t totalBytes = count * (size_t)wordSize(type);
  int loop = 0;
  int skip = 0;
  gin_devtime::resolveLoopSkip(totalBytes / (size_t)nRanksGlobal, loop, skip);

  const int gridCtas = deviceCtaCount;
  double devUs = 0.0;
  TESTCHECK(gin_devtime::measure(args, gridCtas, loop,
      [&](int i, long long* d_start, long long* d_end) {
        ncclDevComm* devComm = args->devComms + i;
        ncclWindow_t sendwin = (ncclWindow_t)(in_place ? args->recvRegHandles[i] : args->sendRegHandles[i]);
        ncclWindow_t recvwin = (ncclWindow_t)args->recvRegHandles[i];
        kernel<<<gridCtas, 512, 0, args->streams[i]>>>(sendwin, 0, recvwin, 0, count, root, *devComm,
                 (int)op, loop, skip, d_start, d_end);
      },
      &devUs));

  if (outDeltaSec != nullptr) {
    *outDeltaSec = (devUs > 0.0) ? devUs * 1.0e-6 : -1.0;
    return testSuccess;
  }

  if (args->proc == 0 && args->thread == 0 && devUs > 0.0) {
    double algBw = 0.0, busBw = 0.0;
    ReduceGetBw(count, (size_t)wordSize(type), devUs * 1.0e-6, &algBw, &busBw, nRanksGlobal);
    snprintf(args->devtimeAugmentLine, sizeof(args->devtimeAugmentLine),
             "#[reduce-devtime] size %12zu B  ctas %2d  loop %2d skip %2d  devtime %10.2f us  algbw %8.2f GB/s  busbw %8.2f GB/s\n",
             totalBytes, gridCtas, loop, skip, devUs, algBw, busBw);
  }
  return testSuccess;
}
#else
testResult_t ReduceDeviceTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec) {
  return testSuccess;  // device API path not available in this build
}
#endif

struct testColl reduceTest = {
  "Reduce",
  ReduceGetCollByteCount,
  ReduceInitData,
  ReduceGetBw,
  ReduceRunColl,
  ReduceGetAlgoProtoChannels,
  NULL,
  NULL,
  ReduceDeviceTime
};

void ReduceGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  ReduceGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t ReduceRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &reduceTest;
  ncclDataType_t *run_types;
  ncclRedOp_t *run_ops;
  const char **run_typenames, **run_opnames;
  int type_count, op_count;
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

  if ((int)op != -1) {
    op_count = 1;
    run_ops = &op;
    run_opnames = &opName;
  } else {
    op_count = test_opnum;
    run_ops = test_ops;
    run_opnames = test_opnames;
  }

  if (root != -1) {
    begin_root = end_root = root;
  } else {
    begin_root = 0;
    end_root = args->nProcs*args->nThreads*args->nGpus-1;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=0; j<op_count; j++) {
      // The -D 3 kernel has no PreMulSum ("mulsum") variant;
      // SPECIALIZE_REDUCE_KERNEL returns nullptr for it, which would abort the
      // sweep on testNotImplemented.
      if (deviceImpl != 0 && strcmp(run_opnames[j], "mulsum") == 0)
        continue;
#if defined(RCCL_FLOAT8)
if((run_types[i] == ncclFloat8e4m3 || run_types[i] == ncclFloat8e5m2) && (run_ops[j] == ncclProd || strcmp(run_opnames[j],"mulsum") == 0))
    continue;
#endif
      for (int k=begin_root; k<=end_root; k++) {
        TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], k));
      }
    }
  }
  return testSuccess;
}

NCCL_WEAK struct testEngine ncclTestEngine = {
  /* .getBuffSize = */ ReduceGetBuffSize,
  /* .runTest = */ ReduceRunTest,
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
  /* .initCommConfig = */ nullptr,
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0) || (defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0))
  /* .getDevCommRequirements = */ ReduceGetDevCommRequirements,
#endif
};
