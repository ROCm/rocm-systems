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
// 128-bit packed and pack-unrolled (see GinReduceScatterKernel) to keep the xGMI
// read pipe full.
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

// Thin getenv() wrapper: parse a decimal CTA-count env var (NCCL_GIN_ANVIL_RS_CTAS)
// into a size_t, returning the policy "unset" sentinel when absent/empty/unparseable
// so gin_sdma_reducescatter::reduceScatterCtas() falls back to its size-adaptive
// ladder. Self-contained here (the target common.h has no testParseSdmaThresholdEnv).
static inline size_t ReduceScatterParseCtasEnv(const char* name) {
  const char* e = getenv(name);
  if (e == nullptr || e[0] == '\0') return gin_sdma_reducescatter::kThresholdUnset;
  char* end = nullptr;
  unsigned long long v = strtoull(e, &end, 10);
  if (end == e) return gin_sdma_reducescatter::kThresholdUnset;
  return (size_t)v;
}

// Mid/large SCHEDULE crossover (total message bytes) for the -D 3 read-reduce.
// DEFAULT 0 -> the warp-unroll large tier is DISABLED and the mid grid-stride path
// (8-way peer ILP + source-0 prefetch) runs for all sizes. Measured on 8x MI355X,
// the improved mid tier beats the old warp-unroll large tier at EVERY size >= 64 MiB
// (64 MiB 246->370, 128 MiB 300->389, 2 GiB 383->409 GB/s busBw) and matches it at
// 32 MiB, so the large tier -- tuned for the earlier 4-way mid loop -- is obsolete.
// NCCL_GIN_ANVIL_RS_UNROLL_MIN (MiB, e.g. "48") re-enables the large tier at that
// crossover for diagnostics/regression; absent/unparseable -> the 0 default.
static inline size_t ReduceScatterUnrollMinBytes() {
  const char* e = getenv("NCCL_GIN_ANVIL_RS_UNROLL_MIN");
  if (e == nullptr || e[0] == '\0') return 0;
  char* end = nullptr;
  unsigned long long mib = strtoull(e, &end, 10);
  if (end == e) return 0;
  return (size_t)mib << 20;
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

  switch(deviceImpl) {
    case 3: { // GinReduceScatterKernel: single-tier LSA read-reduce (no scratch)
      if (commProperties.ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInvalidUsage;
      }
      // Cover both the -V/deviceCtaCount launch and the size-adaptive CTA count the
      // kernel self-selects (reduceScatterCtas, up to reduceScatterMaxCtas()),
      // decoupled from -V -- the read-reduce indexes devComm.lsaBarrier by blockIdx.x.
      const int rsBarCtas = (deviceCtaCount > gin_sdma_reducescatter::reduceScatterMaxCtas())
                              ? deviceCtaCount : gin_sdma_reducescatter::reduceScatterMaxCtas();
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
      const int rsBarCtas = (deviceCtaCount > gin_sdma_reducescatter::reduceScatterMaxCtas())
                              ? deviceCtaCount : gin_sdma_reducescatter::reduceScatterMaxCtas();
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
// Reads are 128-bit packed (Pack = 16 bytes = VEC elements). The load SCHEDULE is
// chosen by total message size (both schedules fold identically, bit-for-bit):
//   * < ~48 MiB: a register-light grid-stride loop (one pack/thread for maximum
//     wave occupancy) that consumes peers FOUR at a time -- four independent peer
//     loads issued before any reduce, so four source ranks overlap their xGMI read
//     latency (deeper peer-ILP than the old 2-way; the read-latency-bound mid-band
//     has its CTA count already maxed, so per-thread load ILP is the lever). This
//     breaks past the serial-per-peer plateau without the register/occupancy cost
//     of the large tier's warp-strided pack-unroll;
//   * >= ~48 MiB: a WARP-STRIDED loop that unrolls over BOTH packs and peers -- a
//     warp owns a tile of U*WARP packs and issues U independent 128-bit loads per
//     source rank at stride WARP (lane varies fastest, so every load stays fully
//     coalesced), AND consumes source ranks two at a time, so 2*U loads are in
//     flight before any reduce. This mirrors RCCL's symmetric ReduceScatter LD
//     kernel (UnrollPacks=4 x UnrollPeers=2 = 8 outstanding loads): pack-ILP fills
//     the pipe within a source, peer-ILP overlaps consecutive sources' xGMI read
//     latency. ~parity with the host symmetric path at >=64 MiB.
// count is always a multiple of 16/sizeof(T) (see ReduceScatterGetCollByteCount) so
// packs tile exactly with no scalar element tail; a short per-lane tail loop covers
// a partial final warp-tile in the unrolled path.
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
__device__ __forceinline__ void ginReduceScatterBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, struct ncclDevComm devComm, int redOp, size_t unrollMinBytes) {
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

  // Adaptive load schedule. Both branches are the identical direct LSA read-reduce
  // (same ascending source-rank fold, bit-for-bit); they differ ONLY in how loads
  // are scheduled:
  //   * grid-stride path (DEFAULT for ALL sizes): a register-light grid-stride loop
  //     (one 128-bit pack per thread) with FULL N-way peer ILP + source-0 prefetch.
  //     Maximizes wave occupancy AND outstanding loads; measured fastest across the
  //     whole 8x MI355X sweep.
  //   * warp-unroll path (LEGACY, off by default): a warp-strided loop unrolled over
  //     packs AND peers (2*U outstanding loads, mirroring RCCL's symmetric LD
  //     UnrollPacks=4 x UnrollPeers=2). This was the large-size tier when the grid-
  //     stride loop only did 4-way ILP; the 8-way + prefetch loop now BEATS it at
  //     every size >= 64 MiB, so it is disabled unless NCCL_GIN_ANVIL_RS_UNROLL_MIN
  //     re-enables it at a chosen MiB crossover (diagnostics/regression only).
  const size_t RS_UNROLL_MIN = unrollMinBytes;
  const size_t totalBytes = count * (size_t)nRanks * sizeof(T);

  if (RS_UNROLL_MIN == 0 || totalBytes < RS_UNROLL_MIN) {
    // ---- small/mid: high-occupancy grid-stride, FULL N-way PEER ILP + src0 prefetch ----
    // One pack per thread (register-light -> max wave occupancy). This band is
    // xGMI-read-LATENCY bound (the host ring fans out to ~128 WarpSpeed channels
    // and hides latency across many small pairwise steps; our flat all-peer pull
    // runs on 32-48 CTAs, so more CTAs only add xGMI incast -- confirmed by the
    // pinned-CTA sweep -- and the remaining lever is per-thread load ILP + cross-
    // iteration pipelining), so we attack it two ways:
    //   (1) up to EIGHT independent 128-bit peer loads issued before any fold, so
    //       all N=8 source ranks' xGMI read latencies overlap (was 4-way; doubling
    //       the outstanding-load count is the mid-band's dominant lever). Eight
    //       Pack temps (128 B) still keeps enough waves resident to matter.
    //   (2) software-prefetch source 0 of the NEXT grid-stride pack while this pack
    //       reduces peers 1..N-1 and writes its output, hiding source 0's read
    //       latency across iterations (mirrors the large tier's next-iter seed).
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
  } else {
    // ---- large: warp-strided pack-unrolled (U outstanding coalesced loads) ----
    // U=4 matches RCCL's symmetric LD UnrollPacks and is the measured sweet spot
    // (U=8 adds a few % at >=512 MiB but costs occupancy); fp8 (VEC16) drops to 2
    // to bound the acc[] footprint. Within one load a warp's lanes read WARP
    // consecutive packs; the U loads stride by WARP so lane varies fastest.
    constexpr int U = (VEC <= 8) ? 4 : 2;
    constexpr int WARP = 64;  // CDNA wavefront
    const int lane = tid & (WARP - 1);
    const size_t warpId = (size_t)(tid / WARP);
    const size_t nWarps = (size_t)(nthreads / WARP);
    const size_t tile = (size_t)U * WARP;
    const size_t gridStride = nWarps * tile;

    // Software-pipelined source-0 seed: the source-0 tile for the NEXT full
    // iteration is loaded while the current iteration reduces peers 1..N-1 and
    // writes its output, hiding source 0's xGMI read latency across grid-stride
    // iterations (mirrors the next-iteration prefetch in RCCL's symmetric LD
    // reduceDeep). A warp's full tiles are contiguous with a fixed stride, so at
    // most one trailing partial tile follows the last full one -- the prefetch
    // guard (nb + tile <= nPacks) simply skips priming when the next tile is that
    // partial remainder, which the tail branch handles without a seed.
    const Pack* src0Base = (const Pack*)ncclGetLsaPointer(sendwin, sendoffset, 0) + myBaseP;
    size_t wbase = warpId * tile;
    Pack seed[U];
    if (wbase + tile <= nPacks) {  // prime the pipeline for this warp's first full tile
      const Pack* sp = src0Base + wbase + (size_t)lane;
      #pragma unroll
      for (int u = 0; u < U; u++) seed[u] = sp[(size_t)u * WARP];
    }
    for (; wbase < nPacks; wbase += gridStride) {
      if (wbase + tile <= nPacks) {
        // ---- fully coalesced tile: 2*U outstanding 128-bit loads ----
        // Combine pack-unroll (U packs/source, stride WARP) with PEER-unroll (two
        // source ranks issued before either reduces): 2*U loads are in flight,
        // mirroring RCCL's symmetric LD kernel (UnrollPacks=4 x UnrollPeers=2 = 8).
        // Pack-ILP alone left the per-source dependency chain exposed at large
        // sizes; adding peer-ILP overlaps consecutive ranks' xGMI read latency and
        // is what closes the gap to the host path. Ascending source-rank fold
        // (s, then s+1) is preserved, so the reduction stays bit-for-bit identical.
        const size_t p0 = wbase + (size_t)lane;  // this lane's first pack
        T acc[U][VEC];
        #pragma unroll  // source s == 0: consume the prefetched seed (ascending fold)
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
        for (; s < nRanks; s++) {  // odd peer tail
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
        // Prefetch source 0 for the next full tile; the loads overlap the output
        // write below (and the back-edge into the next iteration's peer loads).
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
          dstP[p0 + (size_t)u * WARP] = o;
        }
      } else {
        // ---- tail: partial warp-tile; each lane covers packs wbase+u*WARP+lane ----
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
          dstP[pk] = o;
        }
      }
    }
  }

  // NO exit barrier: this is an own-writes-local pull (each rank writes ONLY its
  // local recvbuff and reads peers' read-only sendbuffs), so there is no cross-
  // rank write to publish and no memset race to fence -- the same reasoning that
  // makes the AllGather LSA pull tier entry-only. The entry barrier already
  // guarantees every peer's sendbuff is filled before any read; a rank that
  // finishes early cannot corrupt what a slow peer still reads (sendbuff is never
  // written by the kernel), and the next collective's entry barrier resynchronizes
  // before sendbuff is re-read. In the looped timed kernel each iteration's entry
  // barrier is itself a full inter-iteration sync, so entry-only stays lockstep.
}

// -D 3 kernel: one ReduceScatter. sdmaThreshold/scratch args retained for ABI.
template <typename T>
__global__ void GinReduceScatterKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride, int redOp, ncclDevResourceHandle scratchHandle, size_t unrollMinBytes) {
  (void)sdmaThresholdOverride; (void)scratchHandle; (void)root;
  ginReduceScatterBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, devComm, redOp, unrollMinBytes);
}

// Device-timing kernel (shared gin_devtime methodology): run skip+loop back-to-back
// ReduceScatter bodies under ONE persistent launch, bracketing only the timed region
// with wall_clock64() per CTA. Every body re-derives its entry LSA barrier, which
// is itself a full inter-iteration sync, so looping is correct with no extra
// bookkeeping (pure LSA -> no GIN cadence concern).
template <typename T>
__global__ void GinReduceScatterTimedKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp, int loop, int skip, long long* start_time, long long* end_time, size_t unrollMinBytes) {
  (void)root;
  for (int i = 0; i < skip + loop; i++) {
    if (i == skip) {
      __syncthreads();
      if (threadIdx.x == 0) start_time[blockIdx.x] = wall_clock64();
    }
    ginReduceScatterBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, devComm, redOp, unrollMinBytes);
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
static testResult_t ReduceScatterLaunchDeviceKernelGrid(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, ncclDevResourceHandle scratchHandle, int gridCtas, size_t unrollMinBytes) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  if (gridCtas < 1) gridCtas = 1;
  kernel<<<gridCtas, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, sdmaThresholdOverride, (int)op, scratchHandle, unrollMinBytes);
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
        // Single-tier LSA read-reduce (grid-stride, 8-way peer ILP + src0 prefetch;
        // the warp-unroll large tier is off by default -- see ReduceScatterUnrollMin-
        // Bytes), launched at a SIZE-ADAPTIVE CTA count decoupled from -V (mirrors the
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
        const int rsGridCtas = gin_sdma_reducescatter::reduceScatterCtas(rsTotalBytes, rsCtasEnv);
        static const size_t rsUnrollMin = ReduceScatterUnrollMinBytes();
        TESTCHECK(ReduceScatterLaunchDeviceKernelGrid(SPECIALIZE_REDUCE_KERNEL(GinReduceScatterKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, op, root, comm, stream, gin_sdma_reducescatter::kThresholdUnset, g_rsScratchHandle, rsGridCtas, rsUnrollMin));
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
  const int gridCtas = gin_sdma_reducescatter::reduceScatterCtas(totalBytesCta, rsCtasEnv);
  static const size_t rsUnrollMin = ReduceScatterUnrollMinBytes();
  double devUs = 0.0;
  TESTCHECK(gin_devtime::measure(args, gridCtas, loop,
      [&](int i, long long* d_start, long long* d_end) {
        ncclDevComm* devComm = args->devComms + i;
        ncclWindow_t sendwin = (ncclWindow_t)(in_place ? args->recvRegHandles[i] : args->sendRegHandles[i]);
        ncclWindow_t recvwin = (ncclWindow_t)args->recvRegHandles[i];
        size_t sendoff = in_place ? args->sendInplaceOffset * (size_t)devComm->rank : 0;
        size_t recvoff = in_place ? args->recvInplaceOffset * (size_t)devComm->rank : 0;
        kernel<<<gridCtas, 512, 0, args->streams[i]>>>(sendwin, sendoff, recvwin, recvoff, count, root, *devComm,
                 (int)op, loop, skip, d_start, d_end, rsUnrollMin);
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
      // The GIN device path (deviceImpl != 0) has no PreMulSum ("mulsum") kernel
      // for any type (deferred), and no prod kernel for fp8; SPECIALIZE_REDUCE_KERNEL
      // returns nullptr for those, which would abort the op x type matrix on
      // testNotImplemented. Skip them here so the device sweep only exercises the
      // implemented combos. The host path (deviceImpl == 0) supports them all via
      // ncclReduceScatter, so it keeps full coverage.
      if (deviceImpl != 0) {
        if (strcmp(run_opnames[j], "mulsum") == 0) continue;
#if defined(RCCL_FLOAT8)
        if ((run_types[i] == ncclFloat8e4m3 || run_types[i] == ncclFloat8e5m2) && run_ops[j] == ncclProd)
          continue;
#endif
      }
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
