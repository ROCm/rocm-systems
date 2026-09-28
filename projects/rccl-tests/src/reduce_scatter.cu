/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cstdlib>
#include "cuda_runtime.h"
#include "common.h"
#include "gin_sdma_reducescatter_policy.h"
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"
#include "rccl_vector_types.h"
#if HAVE_FP8
#include "rccl_float8.h"
#endif
#include "gin_sdma_reduce.h"  // device Apply<op,T>, mirrors verifiable.cu exactly
#include "gin_sdma_devtime.h" // shared device-side (wall_clock64) timing scaffold

// ReduceScatter (-D 3): the first reduction collective. Single tier -- balanced
// LSA read-reduce for all sizes. Every rank reads its owned output slice
// [rank*count] directly from EVERY peer's sendbuff via ncclGetLsaPointer, folds
// the N contributions with gin_sdma_reduce (ascending source-rank order, matching
// the verifier bit-for-bit) and writes its local recvbuff. This is the same
// direct-parallel-pull algorithm RCCL's symmetric ReduceScatter LD kernel uses.
// Balanced egress, no scratch/signals -- entry LSA barrier only. Reads are
// 128-bit packed in a grid-stride loop (see GinReduceScatterKernel) to keep the
// xGMI read pipe full.
//
// An earlier size-hybrid design added a large-tier "put-partials + SM reduce"
// path that staged into the GIN resource window and SM-reduced it. That was slow
// for two reasons: (1) the extra staging round-trip, and (2) the reduce read all
// N partials from a SINGLE local buffer, whereas the direct LSA pull spreads the
// reduce reads across N peers' memories / xGMI links in parallel. (Note: under a
// HIP_VMM_UNCACHED_MEMORY build -- active here -- both the resource window and
// ncclMemAlloc'd send/recv buffers are UNCACHED, so caching is not the
// differentiator; read parallelism + load scheduling are.) The launch still
// carries the (unused) sdmaThreshold/scratch args for ABI stability; scratch is
// no longer registered. PreMulSum/mulsum is deferred (SPECIALIZE_REDUCE_KERNEL
// returns nullptr -> testNotImplemented); fp8 prod is excluded there and by the
// ReduceScatterRunTest skip.
static ncclDevResourceHandle g_rsScratchHandle = 0;  // unused (no scratch); passed to kernel as 0

// Caching wrapper. The parser lives in gin_sdma_reducescatter_policy.h so the
// host unit test can lock "8foo" / leading '-' without compiling this TU.
static inline size_t ReduceScatterParseCtasEnv(const char* name) {
  return gin_sdma_reducescatter::parseReduceScatterCtasEnv(name);
}

// Op-aware dispatch for the reduction collective (ReduceScatter). Unlike the
// shared SPECIALIZE_KERNEL (which forces op==ncclSum), this selects kernel<T>
// across the full element-type set for any built-in reduction op
// (sum/prod/max/min/avg); the op itself is passed to the kernel at launch (runtime
// switch), so the template varies only on T. PreMulSum ("mulsum", a created op
// handle >= ncclNumOps) is deferred -> nullptr -> testNotImplemented; fp8 prod is
// excluded too (matches the ReduceScatterRunTest skip). Self-contained here since
// the target common.h defines only SPECIALIZE_KERNEL.
#ifndef SPECIALIZE_REDUCE_KERNEL
#if HAVE_BF16
#define RS_BF16_CASE(kernel, type) (type) == ncclBfloat16 ? kernel<hip_bfloat16> :
#else
#define RS_BF16_CASE(kernel, type)
#endif
#if HAVE_FP8
#define RS_FP8_CASE(kernel, type) (type) == ncclFloat8e4m3 ? kernel<rccl_float8> : (type) == ncclFloat8e5m2 ? kernel<rccl_bfloat8> :
#else
#define RS_FP8_CASE(kernel, type)
#endif
#define SPECIALIZE_REDUCE_KERNEL(kernel, type, op) \
  ( (int)(op) >= (int)ncclNumOps ? nullptr : \
    (((op) == ncclProd && ((type) == ncclFloat8e4m3 || (type) == ncclFloat8e5m2)) ? nullptr : \
     (type) == ncclInt8 ? kernel<int8_t> : \
     (type) == ncclUint8 ? kernel<uint8_t> : \
     (type) == ncclInt32 ? kernel<int32_t> : \
     (type) == ncclUint32 ? kernel<uint32_t> : \
     (type) == ncclInt64 ? kernel<int64_t> : \
     (type) == ncclUint64 ? kernel<uint64_t> : \
     (type) == ncclFloat16 ? kernel<half> : \
     (type) == ncclFloat32 ? kernel<float> : \
     (type) == ncclFloat64 ? kernel<double> : \
     RS_BF16_CASE(kernel, type) \
     RS_FP8_CASE(kernel, type) \
     nullptr) \
  )
#endif
#endif

void ReduceScatterGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  size_t base = gin_sdma_reducescatter::sliceBaseCount(count, eltSize, nranks);
  *sendcount = base*nranks;
  *recvcount = base;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = base;
  *paramcount = base;
}

testResult_t ReduceScatterInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
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
    TESTCHECK(InitDataReduce(args->expected[i], recvcount, rank*recvcount, type, op, rep, nranks));
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

testResult_t  ReduceScatterGetAlgoProtoChannels(ncclComm_t comm, size_t count, ncclDataType_t type, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetAlgoInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetAlgoInfo(comm, ncclFunc_t::ncclFuncReduceScatter , count, type , 0, 0, 1, algo, proto, nchannels));
  return testSuccess;
}

testResult_t  ReduceScatterGetSymkInfo(ncclComm_t comm, size_t count, ncclDataType_t type, ncclRedOp_t op, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetSymkInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetSymkInfo(comm, ncclFunc_t::ncclFuncReduceScatter , count, type , op, algo, proto, nchannels));
  return testSuccess;
}

testResult_t  ReduceScatterGetCollImplInfo(ncclComm_t comm, size_t count, ncclDataType_t type, ncclRedOp_t op,
    const void* sendbuff, void* recvbuff, int graphCapturing, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetCollImplInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetCollImplInfo(comm, ncclFunc_t::ncclFuncReduceScatter, count, type, op, sendbuff, recvbuff, graphCapturing, algo, proto, nchannels));
  return testSuccess;
}

void ReduceScatterGetBw(size_t count, size_t typesize, double sec, double* algBw, double* busBw, int nranks) {
  gin_sdma_reducescatter::bandwidthGBps(count, (int)typesize, sec, nranks, algBw, busBw);
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
testResult_t ReduceScatterGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclComm_t comm) {
  if (!reqs || !comm) return testInternalError;

  ncclCommProperties_t commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
  if (ncclCommQueryProperties(comm, &commProperties) != ncclSuccess) {
    return testNcclError;
  }

  // The read-reduce indexes peers with ncclGetLsaPointer(sendwin, sendoffset, s)
  // for s over [0, devComm.nRanks), i.e. it feeds a WORLD rank where an LSA-team
  // index is expected (ncclGetLsaPointer resolves lsaFlatBase + peer*stride4G).
  // That is only the same number when the world team is exactly the LSA team, so
  // require it here rather than reading a wrong or out-of-aperture peer. Matches
  // the AllGather gate in all_gather.cu.
  if (commProperties.nRanks != ncclTeamLsa(comm).nRanks) {
    fprintf(stderr, "GinReduceScatterKernel requires CUDA P2P connectivity across all ranks. "
                    "Not all ranks of this communicator have P2P connectivity.\n");
    return testInvalidUsage;
  }

  switch(deviceImpl) {
    case 3: { // GinReduceScatterKernel: single-tier LSA read-reduce (no scratch)
      if (commProperties.ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInvalidUsage;
      }
      // Cover both the -V/deviceCtaCount launch and the size-adaptive CTA count the
      // kernel self-selects (reduceScatterCtas, up to reduceScatterMaxCtas()),
      // decoupled from -V -- the read-reduce indexes devComm.lsaBarrier by blockIdx.x.
      // reduceScatterPoolCtas is also the ceiling reduceScatterGridCtas() clamps the
      // launched grid to, so "grid <= pool" is decided in one place.
      const int rsBarCtas = gin_sdma_reducescatter::reduceScatterPoolCtas(deviceCtaCount);
      gin_sdma_reducescatter::DevReqs dr = gin_sdma_reducescatter::reduceScatterDevReqs(rsBarCtas);
      reqs->barrierCount = dr.barrierCount;
      reqs->lsaBarrierCount = dr.lsaBarrierCount;
      reqs->ginSignalCount = dr.ginSignalCount;
      // No resource/scratch window: the LSA read-reduce reads peers' sendbuffs
      // directly, so nothing needs to be staged.
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
bool ReduceScatterGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));
  switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: { // single-tier LSA read-reduce: barriers only, no scratch
      const int rsBarCtas = gin_sdma_reducescatter::reduceScatterPoolCtas(deviceCtaCount);
      gin_sdma_reducescatter::DevReqs dr = gin_sdma_reducescatter::reduceScatterDevReqs(rsBarCtas);
      reqs->barrierCount = dr.barrierCount;
      reqs->lsaBarrierCount = dr.lsaBarrierCount;
      reqs->ginSignalCount = dr.ginSignalCount;
      return true;
    }
#endif
    default:
      return false;
  }
}
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
// Single-node ReduceScatter (-D 3). count is the per-rank output-slice element
// count; the send buffer holds nRanks such slices ([p*count]).
// Single tier: balanced LSA read-reduce. Each rank reads its owned output slice
// [rank*count] directly from EVERY peer's sendbuff via ncclGetLsaPointer, folds
// the N contributions in ascending source-rank order (matching verifiable.cu
// bit-for-bit via gin_sdma_reduce), and writes its local recvbuff. Entry LSA
// barrier only (own-writes-local pull, no exit barrier) -- no scratch, signals,
// or GIN puts.
//
// Reads are 128-bit packed (Pack = 16 bytes = VEC elements). One load schedule
// covers all sizes: a register-light grid-stride loop (one pack/thread) with
// FULL N-way peer ILP + source-0 prefetch. count is always a multiple of
// 16/sizeof(T) (see ReduceScatterGetCollByteCount) so packs tile exactly.
//
// NOTE on the accumulator: low-precision types (half/bf16/fp8) MUST narrow back
// to T on every pairwise step (gin_sdma_reduce::combine) to bit-match the
// verifier, so acc[] stays in T rather than a wider float -- the reduction ALU is
// not the bottleneck; the load schedule is.
//
// This replaces an earlier size-hybrid design whose large tier staged partials
// via GIN put into the resource window and SM-reduced them; the direct LSA pull
// avoids the staging round-trip and spreads reduce reads across N peers' links
// (see the file-top note). The sdmaThreshold/scratch launch args are retained for
// ABI compatibility but unused.
template <typename T>
__device__ __forceinline__ void ginReduceScatterBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, struct ncclDevComm devComm, int redOp) {
  const int nRanks = devComm.nRanks;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;

  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
  lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  // 128-bit packed read-reduce over the owned slice. In-place safe: each thread
  // reads its own input pack (source s == rank) into registers before writing the
  // same recv pack, and each thread owns a disjoint set of pack indices.
  constexpr int VEC = (sizeof(T) <= 16) ? (int)(16 / sizeof(T)) : 1;
  struct alignas(16) Pack { T e[VEC]; };
  Pack* dstP = (Pack*)ncclGetLocalPointer(recvwin, recvoffset);
  const size_t nPacks = count / (size_t)VEC;
  const size_t myBaseP = ((size_t)devComm.rank * count) / (size_t)VEC;  // pack idx of my slice

  // High-occupancy grid-stride with FULL N-way PEER ILP + src0 prefetch.
  // One pack per thread (register-light -> max wave occupancy). This path is
  // xGMI-read-LATENCY bound (the host ring fans out to ~128 WarpSpeed channels
  // and hides latency across many small pairwise steps; our flat all-peer pull
  // runs on 32-48 CTAs, so more CTAs only add xGMI incast -- confirmed by the
  // pinned-CTA sweep -- and the remaining lever is per-thread load ILP + cross-
  // iteration pipelining), so we attack it two ways:
  //   (1) up to EIGHT independent 128-bit peer loads issued before any fold, so
  //       all N=8 source ranks' xGMI read latencies overlap. Eight Pack temps
  //       (128 B) still keeps enough waves resident to matter.
  //   (2) software-prefetch source 0 of the NEXT grid-stride pack while this pack
  //       reduces peers 1..N-1 and writes its output, hiding source 0's read
  //       latency across iterations.
  // Loads may land out of order, but the fold still runs in ascending source-rank
  // order (s = 0,1,...,N-1), so the result is bit-for-bit identical to the verifier.
  const Pack* src0Base = (const Pack*)ncclGetLsaPointer(sendwin, sendoffset, 0) + myBaseP;
  size_t pk = (size_t)tid;
  Pack seed0;
  if (pk < nPacks) seed0 = src0Base[pk];  // prime source-0 for this thread's first pack
  for (; pk < nPacks; pk += (size_t)nthreads) {
    T acc[VEC];
    int s;
    if (nRanks >= 8) {
      // ---- 8-wide peer batch: 8 outstanding 128-bit loads before any fold ----
      // t[0] is the prefetched source-0 seed (no load here); t[1..7] issue fresh,
      // so 7 fresh loads overlap the already-in-flight seed and next-seed prefetch.
      Pack t[8];
      t[0] = seed0;
      #pragma unroll
      for (int j = 1; j < 8; j++)
        t[j] = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, j))[myBaseP + pk];
      #pragma unroll
      for (int e = 0; e < VEC; e++) acc[e] = gin_sdma_reduce::preOp(redOp, t[0].e[e], nRanks);
      #pragma unroll
      for (int j = 1; j < 8; j++)
        #pragma unroll
        for (int e = 0; e < VEC; e++)
          acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, t[j].e[e], nRanks));
      s = 8;
    } else {
      #pragma unroll
      for (int e = 0; e < VEC; e++) acc[e] = gin_sdma_reduce::preOp(redOp, seed0.e[e], nRanks);
      s = 1;
    }
    for (; s + 3 < nRanks; s += 4) {  // four peer loads in flight before folding
      Pack a = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s))[myBaseP + pk];
      Pack b = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s + 1))[myBaseP + pk];
      Pack c = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s + 2))[myBaseP + pk];
      Pack d = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s + 3))[myBaseP + pk];
      #pragma unroll
      for (int e = 0; e < VEC; e++)
        acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, a.e[e], nRanks));
      #pragma unroll
      for (int e = 0; e < VEC; e++)
        acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, b.e[e], nRanks));
      #pragma unroll
      for (int e = 0; e < VEC; e++)
        acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, c.e[e], nRanks));
      #pragma unroll
      for (int e = 0; e < VEC; e++)
        acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, d.e[e], nRanks));
    }
    for (; s + 1 < nRanks; s += 2) {  // two-peer remainder
      Pack a = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s))[myBaseP + pk];
      Pack b = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s + 1))[myBaseP + pk];
      #pragma unroll
      for (int e = 0; e < VEC; e++)
        acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, a.e[e], nRanks));
      #pragma unroll
      for (int e = 0; e < VEC; e++)
        acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, b.e[e], nRanks));
    }
    for (; s < nRanks; s++) {  // odd peer tail
      Pack vs = ((const Pack*)ncclGetLsaPointer(sendwin, sendoffset, s))[myBaseP + pk];
      #pragma unroll
      for (int e = 0; e < VEC; e++)
        acc[e] = gin_sdma_reduce::combine(redOp, acc[e], gin_sdma_reduce::preOp(redOp, vs.e[e], nRanks));
    }
    // Prefetch source 0 for the next grid-stride pack; the load overlaps the
    // output store below and the back-edge into the next iteration's peer loads.
    const size_t nb = pk + (size_t)nthreads;
    if (nb < nPacks) seed0 = src0Base[nb];
    Pack o;
    #pragma unroll
    for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], nRanks);
    dstP[pk] = o;
  }

  // NO exit barrier: this is an own-writes-local pull (each rank writes ONLY its
  // local recvbuff and reads peers' read-only sendbuffs), so there is no cross-
  // rank write to publish and no memset race to fence. The entry barrier already
  // guarantees every peer's sendbuff is filled before any read; a rank that
  // finishes early cannot corrupt what a slow peer still reads (rank R writes only
  // slice R, which no peer reads; in-place, sendbuff IS recvbuff), and the next
  // collective's entry barrier resynchronizes before sendbuff is re-read. In the
  // looped timed kernel each iteration's entry barrier is itself a full
  // inter-iteration sync, so entry-only stays lockstep.
}

// -D 3 kernel: one ReduceScatter. sdmaThreshold/scratch args retained for ABI.
template <typename T>
__global__ void GinReduceScatterKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride, int redOp, ncclDevResourceHandle scratchHandle) {
  (void)sdmaThresholdOverride; (void)scratchHandle; (void)root;
  ginReduceScatterBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, devComm, redOp);
}

// Device-timing kernel (shared gin_devtime methodology): run skip+loop back-to-back
// ReduceScatter bodies under ONE persistent launch, bracketing only the timed region
// with wall_clock64() per CTA. Every body re-derives its entry LSA barrier, which
// is itself a full inter-iteration sync, so looping is correct with no extra
// bookkeeping (pure LSA -> no GIN cadence concern).
template <typename T>
__global__ void GinReduceScatterTimedKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp, int loop, int skip, long long* start_time, long long* end_time) {
  (void)root;
  for (int i = 0; i < skip + loop; i++) {
    if (i == skip) {
      __syncthreads();
      if (threadIdx.x == 0) start_time[blockIdx.x] = wall_clock64();
    }
    ginReduceScatterBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, devComm, redOp);
  }
  __syncthreads();
  if (threadIdx.x == 0) end_time[blockIdx.x] = wall_clock64();
}

// ReduceScatter -D 3 launch with an EXPLICIT grid size (gridCtas) instead of the
// global deviceCtaCount, so the kernel self-selects a size-adaptive CTA count
// (gin_sdma_reducescatter::reduceScatterCtas) decoupled from -V, like the
// broadcast/reduce rings. gridCtas must be <= the barrier/lsaBarrier count
// registered in ReduceScatterGetDevCommRequirements (sized to
// max(deviceCtaCount, tuned)); the kernel indexes devComm.lsaBarrier by blockIdx.x,
// so over-launching would corrupt/hang. sdmaThreshold/scratch args are forwarded
// (inert; retained for ABI). Self-contained here (the target common.h has no
// testLaunchDeviceKernelThresholdScratchGrid).
template <typename F>
static testResult_t ReduceScatterLaunchDeviceKernelGrid(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, ncclDevResourceHandle scratchHandle, int gridCtas) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  if (gridCtas < 1) gridCtas = 1;
  kernel<<<gridCtas, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, sdmaThresholdOverride, (int)op, scratchHandle);
  return testSuccess;
}
#endif

testResult_t ReduceScatterRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
    NCCLCHECK(ncclReduceScatter(sptr, rptr, count, type, op, comm, stream));
  } else {
    switch(deviceImpl) {
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
      case 3: {
        if (count == 0) return testSuccess;
        // Single-tier LSA read-reduce (grid-stride, 8-way peer ILP + src0 prefetch),
        // launched at a SIZE-ADAPTIVE CTA count decoupled from -V (mirrors the
        // broadcast/reduce rings). It is occupancy-bound in the grid-stride mid-band
        // (~8-48 MiB) and peaks at ~48 CTAs (33 MiB ~100% of host, 16 MiB ->98%);
        // small and >=48 MiB sizes peak at 32 (more CTAs add xGMI incast, e.g. 4 MiB
        // 194->168 at 48 CTAs). The bare -V default (16) badly under-launches the
        // mid-band; self-selecting repairs that for callers that don't pass -V.
        // sdmaThreshold/scratch args stay inert (ABI stability).
        const ncclDevComm* rsDc = (const ncclDevComm*)comm;
        const int rsNRanks = (rsDc != nullptr) ? rsDc->nRanks : 1;
        const size_t rsTotalBytes = count * (size_t)wordSize(type) * (size_t)rsNRanks;
        static const size_t rsCtasEnv = ReduceScatterParseCtasEnv("NCCL_GIN_ANVIL_RS_CTAS");
        const int rsGridCtas = gin_sdma_reducescatter::reduceScatterGridCtas(
            rsTotalBytes, rsCtasEnv,
            gin_sdma_reducescatter::reduceScatterPoolCtas(deviceCtaCount));
        TESTCHECK(ReduceScatterLaunchDeviceKernelGrid(SPECIALIZE_REDUCE_KERNEL(GinReduceScatterKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, op, root, comm, stream, gin_sdma_reducescatter::kThresholdUnset, g_rsScratchHandle, rsGridCtas));
        return testSuccess;
      }
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

// Device-side (in-kernel wall_clock64) timing for the GIN-SDMA ReduceScatter (-D 3),
// via the shared gin_devtime scaffold. Opt-in via --device_timing: 1=augment
// (print an extra #[rs-devtime] line next to the graph numbers), 2=device-time-only
// (report the in-kernel latency as the out-of-place metric; in-place keeps normal
// timing). loop/skip come from --devtime_loop/--devtime_skip (default 10/10); size-
// tier overrides via --devtime_loop_mid/_large and --devtime_skip_mid/_large.
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
testResult_t ReduceScatterDeviceTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec) {
  if (!deviceTimingMode) return testSuccess;

  // Only the GIN hybrid impl (-D 3) provisions the GIN signals/barriers the timed
  // body relies on. Other device impls would fault or hang on missing GIN state.
  if (deviceImpl != 3) return testSuccess;

  const size_t count = args->nbytes / wordSize(type);   // per-rank output-slice count
  if (count == 0 || devtimeLoop < 1) return testSuccess;

  const int nRanksGlobalCta = args->nProcs * args->nThreads * args->nGpus;
  const size_t totalBytesCta = count * wordSize(type) * (size_t)nRanksGlobalCta;
  int loop = devtimeLoop;
  int skip = devtimeSkip < 0 ? 0 : devtimeSkip;
  if (devtimeLoopLarge > 0 && totalBytesCta >= (size_t)64 * 1024 * 1024) {
    loop = devtimeLoopLarge;
    if (devtimeSkipLarge >= 0) skip = devtimeSkipLarge;
    else skip = (skip < 1) ? skip : 1;
  } else if (devtimeLoopMid > 0 && totalBytesCta >= (size_t)8 * 1024 * 1024) {
    loop = devtimeLoopMid;
    if (devtimeSkipMid >= 0) skip = devtimeSkipMid;
    else skip = (skip < 2) ? skip : 2;
  }

  auto kernel = SPECIALIZE_REDUCE_KERNEL(GinReduceScatterTimedKernel, type, op);
  if (kernel == nullptr) return testSuccess;

  // Match the perf path: self-select the size-adaptive CTA count (decoupled from
  // -V) so the device-timed number reflects the launched configuration.
  static const size_t rsCtasEnv = ReduceScatterParseCtasEnv("NCCL_GIN_ANVIL_RS_CTAS");
  const int gridCtas = gin_sdma_reducescatter::reduceScatterGridCtas(
      totalBytesCta, rsCtasEnv,
      gin_sdma_reducescatter::reduceScatterPoolCtas(deviceCtaCount));
  double devUs = 0.0;
  TESTCHECK(gin_devtime::measure(args, gridCtas, loop,
      [&](int i, long long* d_start, long long* d_end) {
        ncclDevComm* devComm = args->devComms + i;
        ncclWindow_t sendwin = (ncclWindow_t)(in_place ? args->recvRegHandles[i] : args->sendRegHandles[i]);
        ncclWindow_t recvwin = (ncclWindow_t)args->recvRegHandles[i];
        size_t sendoff = in_place ? args->sendInplaceOffset * (size_t)devComm->rank : 0;
        size_t recvoff = in_place ? args->recvInplaceOffset * (size_t)devComm->rank : 0;
        kernel<<<gridCtas, 512, 0, args->streams[i]>>>(sendwin, sendoff, recvwin, recvoff, count, root, *devComm,
                 (int)op, loop, skip, d_start, d_end);
      },
      &devUs));

  if (outDeltaSec != nullptr) {
    *outDeltaSec = (devUs > 0.0) ? devUs * 1.0e-6 : -1.0;
    return testSuccess;
  }

  if (args->proc == 0 && args->thread == 0 && devUs > 0.0) {
    int nRanksGlobal = args->nProcs * args->nThreads * args->nGpus;
    const size_t totalBytes = count * wordSize(type) * (size_t)nRanksGlobal;
    double sec = devUs * 1.0e-6;
    double algBw = (double)totalBytes / 1.0e9 / sec;
    double busBw = algBw * ((double)(nRanksGlobal - 1) / (double)nRanksGlobal);
    snprintf(args->devtimeAugmentLine, sizeof(args->devtimeAugmentLine),
             "#[rs-devtime] size %12zu B  ctas %2d  loop %2d skip %2d  devtime %10.2f us  algbw %8.2f GB/s  busbw %8.2f GB/s\n",
             totalBytes, gridCtas, loop, skip, devUs, algBw, busBw);
  }
  return testSuccess;
}
#else
testResult_t ReduceScatterDeviceTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec) {
  return testSuccess;  // device API path not available in this build
}
#endif

struct testColl reduceScatterTest = {
  "ReduceScatter",
  ReduceScatterGetCollByteCount,
  ReduceScatterInitData,
  ReduceScatterGetBw,
  ReduceScatterRunColl,
  ReduceScatterGetAlgoProtoChannels,
  ReduceScatterGetSymkInfo,
  ReduceScatterGetCollImplInfo,
  ReduceScatterDeviceTime
};

void ReduceScatterGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  ReduceScatterGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t ReduceScatterRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &reduceScatterTest;
  ncclDataType_t *run_types;
  ncclRedOp_t *run_ops;
  const char **run_typenames, **run_opnames;
  int type_count, op_count;

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
    run_ops = &op;
    run_opnames = &opName;
    op_count = 1;
  } else {
    op_count = test_opnum;
    run_ops = test_ops;
    run_opnames = test_opnames;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=0; j<op_count; j++) {
#if defined(RCCL_FLOAT8)
      // fp8 avg is supported; fp8 prod/mulsum remain out of scope for fp8. This
      // exclusion is unconditional -- it predates the device path and applies to
      // the host path (deviceImpl == 0) too.
      if ((run_types[i] == ncclFloat8e4m3 || run_types[i] == ncclFloat8e5m2) &&
          (run_ops[j] == ncclProd || strcmp(run_opnames[j], "mulsum") == 0))
        continue;
#endif
      // Ring ReduceScatter uses non-direct primitives (Direct=0). Floating-point avg
      // is implemented via PreMulSum and is validated on AllReduce's direct path;
      // integer avg (SumPostDiv) is fine. Skip floating avg on the host path until
      // that parity lands. The GIN device path (-D 3) implements avg itself.
      if (deviceImpl == 0 && run_ops[j] == ncclAvg) {
        switch ((int)run_types[i]) {
        case ncclFloat16:
        case ncclFloat32:
        case ncclFloat64:
#if defined(RCCL_BFLOAT16)
        case ncclBfloat16:
#endif
#if defined(RCCL_FLOAT8)
        case ncclFloat8e4m3:
        case ncclFloat8e5m2:
#endif
          continue;
        default:
          break;
        }
      }
      // Additionally, the GIN device path has no PreMulSum ("mulsum") kernel for
      // ANY type (deferred): SPECIALIZE_REDUCE_KERNEL returns nullptr, which would
      // abort the op x type matrix on testNotImplemented. Skip it so the device
      // sweep only exercises implemented combos; the host path keeps mulsum.
      if (deviceImpl != 0 && strcmp(run_opnames[j], "mulsum") == 0) continue;
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], -1));
    }
  }
  return testSuccess;
}

NCCL_WEAK struct testEngine ncclTestEngine = {
  /* .getBuffSize = */ ReduceScatterGetBuffSize,
  /* .runTest = */ ReduceScatterRunTest,
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
  /* .initCommConfig = */ nullptr,
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0) || (defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0))
  /* .getDevCommRequirements = */ ReduceScatterGetDevCommRequirements,
#endif
};
