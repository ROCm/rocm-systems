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

void ReduceGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  size_t aligned = count;
  // -D 3 splits the array into nranks slices of whole 128-bit packs with no
  // scalar tail, so round the count down to a multiple of nranks*16 bytes. The
  // host path keeps the exact requested count.
  if (deviceImpl != 0 && nranks > 0) {
    size_t base = (count / (size_t)nranks) & -(16 / eltSize);
    aligned = base * (size_t)nranks;
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

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
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
    case 3: // tiled LSA read-reduce
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
    default:
      if (is_main_proc && is_main_thread)
        fprintf(stderr, "Reduce has no -D %d; it only has -D 3.\n", deviceImpl);
      return testNotImplemented;
  }
}
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
// Without ENABLE_DEVICE_API no device kernel is built.
testResult_t ReduceGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclComm_t comm) {
  return testNotImplemented;
}
#elif defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
bool ReduceGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));

  switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: { // GinReduceKernel: reduce-scatter-to-root LSA read-reduce (no scratch)
      reqs->barrierCount = deviceCtaCount;
      reqs->lsaBarrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
      return true;
    }
#endif
    default:
      return false;
  }
}
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
// Waits until this wave's stores have reached L2, so a system-scope release by
// one thread after a CTA barrier also covers the other waves' stores.
// __syncthreads() alone does not wait for them on every target.
__device__ __forceinline__ void waitWaveStores() {
#if defined(__GFX12__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#elif defined(__GFX10__) || defined(__GFX11__)
  asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
#else
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
}

// An LSA barrier sync whose memory ordering runs once per CTA. The barrier's
// own release fence runs on every wave (a system-scope L2 writeback each) and
// its acquire on every poll (an L2 invalidate each), which with many CTAs costs
// more than the data movement.
__device__ __forceinline__ void leanBarSync(ncclLsaBarrierSession<ncclCoopCta>& bar, bool release, bool acquire) {
  if (release) {
    waitWaveStores();
    __syncthreads();
    if (threadIdx.x == 0) cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
  if (acquire) {
    if (threadIdx.x == 0) cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
    __syncthreads();
  }
}

// `count` is the FULL element count (== base*nRanks, aligned by
// ReduceGetCollByteCount). A warp-strided pack-unrolled loop (U outstanding
// coalesced loads per source) with 2-way peer ILP and a source-0 prefetch of the
// next tile, folding sources in ascending rank order. Remote loads are latency
// bound, so even small messages need U loads in flight per thread.
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

  constexpr int U = (VEC <= 8) ? 4 : 2;
  constexpr int WARP = 64;  // lanes per tile row (two waves on wave32 targets)
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

  leanBarSync(lsaBar, /*release=*/true, /*acquire=*/false);
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

// Like testLaunchDeviceKernel() but forwards the reduction op, which the shared
// launcher's fixed signature does not carry.
template <typename F>
static testResult_t ReduceLaunchDeviceKernel(F kernel, int grid, void* sendbuff, size_t sendoffset,
                                             void* recvbuff, size_t recvoffset, size_t count,
                                             ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  kernel<<<grid, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, (int)op);
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
        // Every rank's sendbuff and the root's recvbuff are accessed through
        // LSA pointers. The 2.29 requirements hook rejects this earlier; the
        // 2.28 one has no communicator to check.
        const ncclDevComm* devComm = (const ncclDevComm*)comm;
        if (devComm->lsaSize != devComm->nRanks) {
          fprintf(stderr, "Reduce -D 3 needs every rank in one LSA team (single node); "
                          "this communicator has %d ranks and an LSA team of %d.\n",
                  devComm->nRanks, devComm->lsaSize);
          return testInvalidUsage;
        }
        if (count == 0) return testSuccess;
        TESTCHECK(ReduceLaunchDeviceKernel(SPECIALIZE_REDUCE_KERNEL(GinReduceKernel, type, op), deviceCtaCount, sendbuff, sendoffset, recvbuff, recvoffset, count, op, root, comm, stream));
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
      // The device kernels have no PreMulSum ("mulsum") variant, so an -o all
      // sweep skips it; asked for explicitly, it must fail rather than exit 0
      // with no results.
      if (deviceImpl != 0 && strcmp(run_opnames[j], "mulsum") == 0) {
        if ((int)op != -1) {
          fprintf(stderr, "Reduce -D %d has no mulsum (PreMulSum) variant.\n", deviceImpl);
          return testNotImplemented;
        }
        continue;
      }
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
