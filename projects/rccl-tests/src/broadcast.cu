/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#include "gin_sdma_broadcast_policy.h"  // pure host/device Broadcast tier policy (gin_sdma::)
#include "gin_sdma_devtime.h"  // shared device-side (wall_clock64) timing scaffold
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"
// signalPeer/remoteSignalAddr: standalone IPC/LSA peer-signal (atomic add, no SDMA
// queue) used by the point-to-point ring so hops signal only their successor
// instead of rendezvousing all CTAs at a per-step LSA barrier.
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma.h"
// v4u / v4u_gptr / RCCL_SYSTEM_SYNCSCOPE / RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS for
// nontemporal system-scope b128 peer stores (the store path LL/host use over xGMI).
#include "nccl_device/rccl_ptr.h"
#include "rccl_vector_types.h"
#endif

// LL (low-latency, packed data+flag) small-message Broadcast path. Broadcast is
// one-to-all, so it maps directly onto the LL A2A session's bcast() primitive:
// only the root writes the payload (into every peer's epoch-tagged scratch), and
// every rank polls it out of its OWN scratch into its local recvbuff. The
// epoch-tag recv replaces the EXIT barrier, and each rank writes only its own
// recvbuff, so it is immune to the initData recvbuff-memset race. Unlike
// AllGather LL it keeps a single ENTRY barrier for correctness (broadcast lacks
// mutual-recv backpressure; see BroadcastLLImpl for the deadlock rationale). The
// device types come from nccl_device.h, which common.h includes on the device
// API (>=2.28) or any >=2.29 build; gate on that.
#if (defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)) || NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
#define BC_HAVE_LL 1
// The compile ceiling (gin_sdma::kBroadcastLLMaxBytes, 64 KiB) and default
// LL<->LSA cutover (gin_sdma::kBroadcastLLDefaultMaxBytes, 2 KiB) live in
// gin_sdma_broadcast_policy.h so the kernel branch, the requirements sizing and
// the host unit tests all share one source. The actual cutover is runtime-gated
// via env NCCL_GIN_ANVIL_BCAST_LL_MAX_BYTES (0 = disable). LL wins for tiny
// broadcasts by dropping one of the two LSA barriers: on 8x MI355X (50 iters x 3
// reps, 2026-07-27) it cut small-message latency a robust 13-16% at <=1 KiB and
// ~8% at 2 KiB, crossing over (~+3%) at 4 KiB, hence the 2 KiB default.
// LL scratch handle: bufHandle assigned during ncclDevCommCreate; nSlots set by
// ncclLLA2ACreateRequirement when the LL path is enabled. nSlots==0 => LL not
// configured, so the kernel uses the direct-LSA fan-out. The cutover is tunable
// via NCCL_GIN_ANVIL_BCAST_LL_MAX_BYTES=<bytes> (0 = disable, unset = default
// BROADCAST_LL_DEFAULT_MAX_BYTES); the value is clamped to BROADCAST_LL_MAX_BYTES
// and the pre-sized slot count makes it the effective cutoff.
static size_t g_bcastLLMaxBytes = 0;  // resolved from env at requirements time
static ncclLLA2AHandle g_bcastLLHandle = {};
static ncclDevResourceRequirements g_bcastLLReq = {};
#endif

// ---------------------------------------------------------------------------
// Self-contained GIN-SDMA launch / threshold / put helpers.
//
// The upstream (stage2j) design carried these in common.h so every collective
// shared them. On this branch the per-collective .cu files stay self-contained
// (see all_gather.cu, which keeps its own threshold resolver and launcher), so
// Broadcast defines exactly the helpers it needs here rather than widening
// common.h. The pure size math (gin_sdma::parseSize/resolveThreshold and the
// put-segmentation kGin* constants) is reused from gin_sdma_broadcast_policy.h
// and the backend put-policy header, so there is no duplicated policy logic.
// ---------------------------------------------------------------------------

// Sentinel meaning "no per-collective override; use the device/backend value
// (rsCtx->sdmaThreshold, populated from NCCL_GIN_ANVIL_SDMA_THRESHOLD)".
#ifndef TEST_SDMA_THRESHOLD_UNSET
#define TEST_SDMA_THRESHOLD_UNSET (gin_sdma::kThresholdUnset)
#endif

// Parse a per-collective LSA<->GIN threshold env var (bytes; optional K/M/G
// suffix). Returns TEST_SDMA_THRESHOLD_UNSET when unset/empty/unparseable so the
// kernel falls back to the shared NCCL_GIN_ANVIL_SDMA_THRESHOLD value. Thin
// getenv() wrapper over the pure, unit-tested gin_sdma::parseSize().
static inline size_t testParseSdmaThresholdEnv(const char* name) {
  return gin_sdma::parseSize(getenv(name));
}

// Resolve a collective's LSA<->GIN threshold with the fallback chain:
//   1. the collective-specific env var (collVar), if set;
//   2. the shared NCCL_GIN_ANVIL_SDMA_THRESHOLD, if explicitly set;
//   3. the collective's data-driven default (collDefault).
static inline size_t testResolveSdmaThreshold(const char* collVar, size_t collDefault) {
  return gin_sdma::resolveThreshold(gin_sdma::parseSize(getenv(collVar)),
                                    gin_sdma::parseSize(getenv("NCCL_GIN_ANVIL_SDMA_THRESHOLD")),
                                    collDefault);
}

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
// NOTE: large-put segmentation is NOT done here anymore. The Anvil-SDMA backend
// ncclGinApi_Put (gin_anvil_sdma.h) internally splits any put into <=128 MiB
// (gin_sdma::kGinPutSegBytes) in-order copies, carrying the remote action on the
// final segment, so a single gin.put() of an arbitrarily large message is both
// overflow-safe (30-bit SDMA copy-count) and hang-safe (MI355X 256 MiB stall) for
// ALL callers. The test therefore issues plain gin.put() calls and exercises the
// same path a real application would; see gin_anvil_sdma_put_policy.h.

// Like common.h's testLaunchDeviceKernel but threads a per-collective LSA<->GIN
// threshold override as the last kernel argument (the flat/SAG GIN kernels).
template <typename F>
testResult_t testLaunchDeviceKernelThreshold(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  kernel<<<deviceCtaCount, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, sdmaThresholdOverride);
  return testSuccess;
}

// Like ...Threshold, but the caller picks the grid (CTA count) instead of the
// global deviceCtaCount (-V). Used by the SM ring kernels, whose pipelined
// throughput is CTA-bound: they launch their own power-of-2 count (~128 to
// saturate all xGMI links) decoupled from -V. The requested grid must be <= the
// lsaBarrier count registered in BroadcastGetDevCommRequirements (sized to
// max(deviceCtaCount, ring CTAs)); the kernels index devComm.lsaBarrier by
// blockIdx.x, so over-launching would corrupt/hang.
template <typename F>
testResult_t testLaunchDeviceKernelThresholdCtas(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, int gridCtas) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  if (gridCtas < 1) gridCtas = 1;
  kernel<<<gridCtas, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, sdmaThresholdOverride);
  return testSuccess;
}

// Variant that also forwards an LL (low-latency, packed data+flag) handle as the
// last kernel argument. Used by the Broadcast GIN kernel for its tiny-message LL
// fast path; the handle is a small POD ({bufHandle, nSlots}) assigned during
// ncclDevCommCreate.
template <typename F>
testResult_t testLaunchDeviceKernelThresholdLL(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, ncclLLA2AHandle llHandle, int gridCtas) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  if (gridCtas < 1) gridCtas = 1;
  kernel<<<gridCtas, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, sdmaThresholdOverride, llHandle);
  return testSuccess;
}

// Like common.h's testLaunchDeviceKernel but the caller picks the grid (CTA count).
template <typename F>
testResult_t testLaunchDeviceKernelCtas(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int gridCtas) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  if (gridCtas < 1) gridCtas = 1;
  kernel<<<gridCtas, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm);
  return testSuccess;
}
#endif  // ENABLE_DEVICE_API && >= 2.28.0

// Set RCCL_TESTS_ALLGATHERV_ENABLE=1 to run grouped-broadcast (AllGatherV fusion) mode
static int broadcast_grouped = [] {
  const char* s = getenv("RCCL_TESTS_ALLGATHERV_ENABLE");
  return s != nullptr && s[0] == '1';
}();

void BroadcastGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  if (broadcast_grouped) {
    if (count < (size_t)nranks) {
      *sendcount = *recvcount = *paramcount = *sendInplaceOffset = *recvInplaceOffset = 0;
      return;
    }
    size_t chunkElems = (count / nranks) & -(16 / eltSize);
    if (chunkElems == 0) {
      *sendcount = *recvcount = *paramcount = *sendInplaceOffset = *recvInplaceOffset = 0;
      return;
    }
    *sendcount  = chunkElems;
    *recvcount  = chunkElems * nranks;
    *paramcount = chunkElems;
    *sendInplaceOffset = chunkElems;
    *recvInplaceOffset = 0;
  } else {
    *sendcount = count;
    *recvcount = count;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = *sendcount;
  }
}

testResult_t BroadcastInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  if (broadcast_grouped) {
    size_t chunkElems = args->sendBytes / wordSize(type);
    int nranks = args->nProcs * args->nThreads * args->nGpus;
    for (int i = 0; i < args->nGpus; i++) {
      CUDACHECK(cudaSetDevice(args->gpus[i]));
      int rank = (args->proc * args->nThreads + args->thread) * args->nGpus + i;
      CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
      void* data = in_place ? ((char*)args->recvbuffs[i]) + rank * args->sendBytes : args->sendbuffs[i];
      TESTCHECK(InitData(data, chunkElems, 0, type, ncclSum, rank + 1, 1, 0));
      for (int k = 0; k < nranks; k++) {
        char* slot = (char*)args->expected[i] + (size_t)k * chunkElems * wordSize(type);
        TESTCHECK(InitData(slot, chunkElems, 0, type, ncclSum, k + 1, 1, 0));
      }
      CUDACHECK(cudaDeviceSynchronize());
    }
    return testSuccess;
  }

  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    if (rank == root) TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, rep, 1, 0));
    TESTCHECK(InitData(args->expected[i], recvcount, 0, type, ncclSum, rep, 1, 0));
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

testResult_t  BroadcastGetAlgoProtoChannels(ncclComm_t comm, size_t count, ncclDataType_t type, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetAlgoInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetAlgoInfo(comm, ncclFuncBroadcast , count, type , 0, 0, 1, algo, proto, nchannels));
  return testSuccess;
}

void BroadcastGetBw(size_t count, size_t typesize, double sec, double* algBw, double* busBw, int nranks) {
  if (broadcast_grouped) {
    double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;
    *algBw = baseBw;
    double factor = (double)(nranks - 1) / (double)nranks;
    *busBw = baseBw * factor;
  } else {
    double baseBw = (double)(count * typesize) / 1.0E9 / sec;
    *algBw = baseBw;
    double factor = 1;
    *busBw = baseBw * factor;
  }
}

// CTA (stripe) count for the GIN ring broadcast kernels. The ring picks its own
// count (its throughput is CTA-bound -- it needs enough CTAs to saturate all xGMI
// links; measured on 8x MI355X @2 GiB: 32 CTAs = 238 GB/s, 64 = 345, 128 = 350;
// SAG does not scale this way). This count is DECOUPLED from -V/deviceCtaCount:
// -V defaults to 16 (tuned for every OTHER collective, several of which regress
// with more CTAs -- e.g. Reduce), so clamping the ring to -V would have starved
// it (16 CTAs ~= 140 GB/s). Instead the ring always self-selects 128 (or the
// NCCL_GIN_ANVIL_BCAST_RING_CTAS override), so the promoted 350 GB/s holds under
// the bare default AND under the gate/board's -V 32. Only power-of-2 counts avoid
// a wave-quantization cliff (non-pow2 like 96 collapse to ~140 GB/s), so the
// result is rounded DOWN to a power of 2 in [1,128]. Read identically here and in
// BroadcastGetDevCommRequirements, which allocates max(deviceCtaCount, this) LSA
// barriers so the launched grid never exceeds the allocated lsaBarrier count
// (kernels index devComm.lsaBarrier by blockIdx.x, so over-launching would
// corrupt/hang).
static inline int bcastRingCtas() {
  int pref = 128;
  const char* e = getenv("NCCL_GIN_ANVIL_BCAST_RING_CTAS");
  if (e && e[0]) {
    long p = strtol(e, nullptr, 0);
    if (p >= 1) pref = (int)p;
  }
  int v = pref;                           // ring's own count, independent of -V
  if (v < 1) v = 1;
  if (v > 128) v = 128;                   // hard cap (barrier resource / tested range)
  int p2 = 1;
  while ((p2 << 1) <= v) p2 <<= 1;        // largest power of 2 <= v
  return p2;
}

// Parse NCCL_GIN_ANVIL_BCAST_CTAS (diagnostic pin) into a size_t, returning the
// "unset" sentinel when absent/empty/unparseable so bcastHybridCtas()/bcastSagCtas()
// fall back to the size-adaptive ladder.
static inline size_t BroadcastParseCtasEnv() {
  const char* e = getenv("NCCL_GIN_ANVIL_BCAST_CTAS");
  if (e == nullptr || e[0] == '\0') return gin_sdma::kBroadcastCtasUnset;
  char* end = nullptr;
  unsigned long long v = strtoull(e, &end, 10);
  if (end == e) return gin_sdma::kBroadcastCtasUnset;
  return (size_t)v;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
testResult_t BroadcastGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclCommProperties_t* commProperties) {
  if (!reqs || !commProperties) return testInternalError;

  switch(deviceImpl) {
    case 3: { // GinHybridBroadcastKernel: LSA direct (small) + root GIN puts (large)
      if (commProperties->ginType == NCCL_GIN_TYPE_NONE) {
        fprintf(stderr, "This test requires GIN support, but GIN support is not enabled for this communicator.\n");
        return testInternalError;
      }
      // Barriers must cover BOTH the size-adaptive hybrid/SAG grids (pool =
      // max(-V, ladder peak)) and the ring's own (larger) CTA count.
      {
        int barCtas = gin_sdma::bcastPoolCtas(deviceCtaCount, bcastRingCtas());
        reqs->barrierCount = barCtas;
        reqs->lsaBarrierCount = barCtas;
      }
      // >=2 so the scatter+allgather large tier (§4.8) can use two independent
      // signal indices (scatter=0, gather=1); the flat/LSA paths use only 0.
      reqs->ginSignalCount = gin_sdma::bcastSignalCount(deviceCtaCount);
      // LL scratch for the tiny-message fast path (single CTA => nBlocks=1),
      // on by default up to BROADCAST_LL_DEFAULT_MAX_BYTES, tunable via
      // NCCL_GIN_ANVIL_BCAST_LL_MAX_BYTES (bytes; 0 = disable). Broadcast carries
      // a single message (only the root sends), so a receiver needs just cap/8
      // u64 slots -- unlike AllGather/AllToAll which need nRanks*cap/8.
      {
        g_bcastLLMaxBytes = gin_sdma::resolveLLCap(
            testParseSdmaThresholdEnv("NCCL_GIN_ANVIL_BCAST_LL_MAX_BYTES"),
            gin_sdma::kBroadcastLLDefaultMaxBytes, gin_sdma::kBroadcastLLMaxBytes);
        if (g_bcastLLMaxBytes > 0) {
          int llMaxElts = (int)(g_bcastLLMaxBytes / 8);
          int nSlots = ncclLLA2ACalcSlots(llMaxElts, /*maxEltSize=*/8);
          ncclLLA2ACreateRequirement(/*nBlocks=*/1, nSlots, &g_bcastLLHandle, &g_bcastLLReq);
          g_bcastLLReq.next = reqs->resourceRequirementsList;
          reqs->resourceRequirementsList = &g_bcastLLReq;
        }
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
bool BroadcastGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));

  switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
    case 3: { // GinHybridBroadcastKernel: LSA direct (small) + root GIN puts (large)
      // Cover hybrid/SAG size-adaptive grids and the ring's own larger CTA count.
      int barCtas = gin_sdma::bcastPoolCtas(deviceCtaCount, bcastRingCtas());
      reqs->barrierCount = barCtas;
      reqs->lsaBarrierCount = barCtas;
      // >=2 for the scatter+allgather large tier's two signal indices (§4.8).
      reqs->ginSignalCount = gin_sdma::bcastSignalCount(deviceCtaCount);
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
__device__ size_t BroadcastGetSdmaThreshold(struct ncclDevComm const& devComm) {
  using nccl::utility::loadConst;
  if (devComm.ginConnectionCount == 0 || devComm.ginHandles[0] == nullptr) {
    return NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT;
  }
  ncclGinAnvilSdmaGPUContext* rsCtx =
      (ncclGinAnvilSdmaGPUContext*)devComm.ginHandles[0];
  if (rsCtx == nullptr ||
      loadConst(&rsCtx->layoutMagic) != NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC) {
    return NCCL_GIN_ANVIL_SDMA_THRESHOLD_DEFAULT;
  }
  return loadConst(&rsCtx->sdmaThreshold);
}

// Local send->recv copy on the root (out-of-place). Uses 16-byte vector stores
// when both buffers and the byte count are 16-byte aligned, else scalar.
template <typename T>
__device__ void BroadcastLocalCopy(T* dst, const T* src, size_t count, int tid, int nthreads) {
  const size_t bytes = count * sizeof(T);
  const uintptr_t da = (uintptr_t)dst;
  const uintptr_t sa = (uintptr_t)src;
  if ((bytes % sizeof(uint4)) == 0 && (da % sizeof(uint4)) == 0 && (sa % sizeof(uint4)) == 0) {
    uint4* d4 = (uint4*)dst;
    const uint4* s4 = (const uint4*)src;
    const size_t n4 = bytes / sizeof(uint4);
    for (size_t i = tid; i < n4; i += nthreads) d4[i] = s4[i];
  } else {
    for (size_t i = tid; i < count; i += nthreads) dst[i] = src[i];
  }
}

// Streaming peer copy: like BroadcastLocalCopy but the DESTINATION stores use
// nontemporal system-scope 128-bit writes (__builtin_amdgcn_global_store_b128 with
// RCCL_SYSTEM_SYNCSCOPE) instead of cached uint4 stores. This is the store path the
// LL A2A primitive and the host ring use to push xGMI writes at full rate without
// polluting cache; the source read stays a normal (cached) load since it is local.
// Falls back to BroadcastLocalCopy when the b128 builtin or 16B alignment is absent.
template <typename T>
__device__ void BroadcastPeerCopyStream(T* dst, const T* src, size_t count, int tid, int nthreads) {
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  const size_t bytes = count * sizeof(T);
  const uintptr_t da = (uintptr_t)dst;
  const uintptr_t sa = (uintptr_t)src;
  if ((bytes % sizeof(uint4)) == 0 && (da % sizeof(uint4)) == 0 && (sa % sizeof(uint4)) == 0) {
    const uint4* s4 = (const uint4*)src;
    uint4* d4 = (uint4*)dst;
    const size_t n4 = bytes / sizeof(uint4);
    for (size_t i = tid; i < n4; i += nthreads) {
      union { uint4 u; v4u v; } cv;
      cv.u = s4[i];
      __builtin_amdgcn_global_store_b128((v4u_gptr)(d4 + i), cv.v, RCCL_SYSTEM_SYNCSCOPE);
    }
    return;
  }
#endif
  BroadcastLocalCopy<T>(dst, src, count, tid, nthreads);
}

// Root reads one send slice and writes the same recv offset on every LSA peer
// (including itself, which also covers the out-of-place self copy).
template <typename T>
__device__ void BroadcastLsaDirect(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int nRanks, int tid, int nthreads) {
  T* src = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  for (size_t i = tid; i < count; i += nthreads) {
    T value = src[i];
    for (int lp = 0; lp < nRanks; lp++) {
      T* dst = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp);
      dst[i] = value;
    }
  }
}

// LL (low-latency) Broadcast for tiny messages, single CTA. Only the root has
// the payload, so only the root bcast()s its message (as 8-byte units) into the
// epoch-tagged LL scratch of every peer; every rank (root included) then polls
// the message out of its OWN scratch into its local recvbuff. The recv's epoch
// tag replaces the exit barrier (a non-root knows the payload arrived when the
// tag matches), and cross-rank traffic stays in the LL scratch (each rank writes
// only its own recvbuff), so this is immune to the initData recvbuff-memset race.
//
// Why an ENTRY barrier is still required (unlike AllGather LL, which needs none).
// The LL session double-buffers by epoch parity ((epoch&1)*nSlots), so it only
// tolerates a <2-epoch skew between writer and reader. In AllGather every rank
// both sends and receives, so the mutual recv self-throttles all ranks to within
// one epoch. Broadcast has no such backpressure: only the root writes, so across
// the perf loop's back-to-back kernel launches a fast root can run >=2 epochs
// ahead of a lagging non-root and overwrite a slot region that rank is still
// polling with a newer tag -> the reader's tag never matches -> deadlock
// (reproduced on MI355 as a hang). The entry LSA barrier bounds the skew to <1
// epoch, making the double-buffer safe; the exit barrier stays removed.
__device__ void BroadcastLLImpl(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset,
                                size_t chunkU64, int root, struct ncclDevComm const& devComm, ncclTeam lsa,
                                ncclLLA2AHandle llHandle) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  uint64_t* dst = (uint64_t*)ncclGetLocalPointer(recvwin, recvoffset);

  // Entry barrier: bounds root-vs-laggard epoch skew (see note above).
  ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, /*block=*/0 };
  lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  ncclLLA2ASession<ncclCoopCta> ll { ncclCoopCta(), devComm, lsa, llHandle, /*block=*/0,
                                     /*maxElts=*/(int)chunkU64 };

  // Root broadcasts its message into every peer's scratch slot [0..chunkU64).
  // NB: ll.bcast() already fans out to all N peers per call with an 8-way
  // unrolled, register-reusing store loop, so it is more efficient than an
  // explicit per-(peer,slot) send flatten across the LL range (measured: the
  // flatten helps only <=512 B but regresses 1-2 KiB by 20-40%). Unlike Scatter
  // (which had a genuinely serial explicit-send loop), keep bcast() here.
  if (devComm.rank == root) {
    const uint64_t* src = (const uint64_t*)ncclGetLocalPointer(sendwin, sendoffset);
    for (size_t j = tid; j < chunkU64; j += nthreads) {
      ll.bcast((int)j, src[j]);
    }
  }
  // Every rank (root included) gathers the message out of its own scratch.
  for (size_t j = tid; j < chunkU64; j += nthreads) {
    dst[j] = ll.recv<uint64_t>((int)j);
  }
  ll.endEpoch(ncclCoopCta());
}

// Single-node hybrid Broadcast (-D 3): flat / star fan-out from root.
//   msgBytes <= LL cap (opt-in):  LL packed data+flag, single CTA (tiny, no barrier).
//   msgBytes <= sdmaThreshold:    root writes every peer's recvbuff via LSA.
//   msgBytes >  sdmaThreshold:    root issues one GIN put per non-self peer
//                                 (Anvil picks IPC or SDMA by size).
// Non-root ranks only participate in the entry/exit barriers (LSA path).
//
// sdmaThresholdOverride lets the Broadcast-specific env var
// NCCL_GIN_ANVIL_SDMA_THRESHOLD_BROADCAST tune this LSA<->GIN cutover
// independently; TEST_SDMA_THRESHOLD_UNSET falls back to the shared backend
// value (rsCtx->sdmaThreshold from NCCL_GIN_ANVIL_SDMA_THRESHOLD). The LL tier is
// on by default up to BROADCAST_LL_DEFAULT_MAX_BYTES (disable with
// NCCL_GIN_ANVIL_BCAST_LL_MAX_BYTES=0, which leaves llHandle.nSlots==0).
template <typename T>
__global__ void GinHybridBroadcastKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t sdmaThresholdOverride, ncclLLA2AHandle llHandle) {
  const size_t msgBytes = count * sizeof(T);
  const size_t sdmaThreshold = (sdmaThresholdOverride != TEST_SDMA_THRESHOLD_UNSET)
                                   ? sdmaThresholdOverride
                                   : BroadcastGetSdmaThreshold(devComm);
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;

  if (msgBytes <= sdmaThreshold) {
    ncclTeam lsa = ncclTeamLsa(devComm);

    // Tiny messages: LL packed data+flag path (single CTA), barrier-free and
    // memset-race-immune. Used when LL is configured (nSlots>0), the message is
    // 8-byte aligned, and it fits the pre-sized slot count.
    if (gin_sdma::bcastLLEligible(msgBytes, llHandle.nSlots, gin_sdma::kBroadcastLLMaxBytes)) {
      if (blockIdx.x != 0) return;  // single CTA
      const size_t chunkU64 = msgBytes / 8;
      BroadcastLLImpl(sendwin, sendoffset, recvwin, recvoffset, chunkU64, root, devComm, lsa, llHandle);
      return;
    }

    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
    if (devComm.rank == root) {
      BroadcastLsaDirect<T>(sendwin, sendoffset, recvwin, recvoffset, count, lsa.nRanks, tid, nthreads);
    }
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
    return;
  }

  const int ginContext = 0;
  const unsigned int signalIndex = 0;
  ncclGin gin { devComm, ginContext };
  // ncclGin_SignalInc is a *remote* action: each put increments the receiving
  // peer's signal (confirmed vs AllGather/AlltoAll, where a rank receives N puts
  // and waits base+N). So the root -- which receives no puts -- must not wait on
  // its own signal; instead every non-root receives exactly one put and waits
  // base+1. flush() alone does not guarantee remote data has settled, so this
  // receiver-side waitSignal is what makes the payload visible on non-roots.
  const uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  // Entry barrier: every rank's recv buffer quiescent before root starts writing.
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  if (devComm.rank == root) {
    // Out-of-place self copy (no-op when in-place: local src == local dst).
    T* lsrc = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* ldst = (T*)ncclGetLocalPointer(recvwin, recvoffset);
    if (lsrc != ldst) {
      BroadcastLocalCopy<T>(ldst, lsrc, count, tid, nthreads);
    }

    // Flat fan-out: one put per non-self peer; skip self. The backend segments
    // large messages internally (<=128 MiB copies, signal on the final one).
    for (int r = tid; r < devComm.nRanks; r += nthreads) {
      if (r == root) continue;
      gin.put(ncclTeamWorld(devComm), r,
          recvwin, recvoffset,
          sendwin, sendoffset,
          msgBytes, ncclGin_SignalInc{signalIndex});
    }

    // flush(): all source buffers safe to reuse once the puts are pushed.
    gin.flush(ncclCoopCta());
  } else {
    // Each non-root receives exactly one put from root; wait for it to settle.
    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + 1);
  }
}

// Large-message Broadcast via scatter + in-place allgather (van de Geijn). See
// gin-anvil-sdma-broadcast-design-plan.md §4.8.
//
// The flat fan-out (GinHybridBroadcastKernel large path) is latency-optimal
// (1 hop) but its bandwidth is capped by *root egress*: the root alone pushes
// N-1 full copies of the message, so busBw = B_root_egress / N (measured
// ~60 GB/s @128 MiB on 8x MI355X). This kernel breaks that ceiling by making
// every rank forward data, the same way the AllGather GIN path reaches
// ~388 GB/s:
//
//   Phase 1 (scatter): the root sends chunk r (= the r-th M/N slice) to rank r,
//     so only M total bytes leave the root instead of (N-1)*M. Each non-root
//     receives exactly one scatter put (signal 0) and waits base0+1; the root
//     copies its own slice locally (its slot is never written by the allgather).
//   Phase 2 (allgather): every rank puts its own slice into every peer's
//     matching slot, so egress is distributed across all N ranks. Each rank
//     receives exactly N-1 puts (signal 1) and waits base1+(N-1).
//
// Two distinct signal indices (scatter=0, gather=1) keep the per-phase completion
// counts from interleaving, so NO inter-phase barrier is needed: a non-root only
// begins forwarding after its waitSignal(0) (a CTA-wide op that also makes the
// scatter data visible), and the root reads its own slice from the stable
// sendwin (not the just-written recvwin), so its local copy needs no ordering
// vs its allgather puts. Only the entry barrier is kept (recvbuff quiescent
// before the root writes -- initData memset race, as in the flat path).
//
// Host-gated to large messages (NCCL_GIN_ANVIL_BCAST_SCATTER_AG_MIN_BYTES,
// default 2 MiB) with count >= nRanks, where the egress ceiling dominates the
// extra round + scatter setup. Requires ginSignalCount >= 2 (set in
// BroadcastGetDevCommRequirements case 3).
template <typename T>
__global__ void GinScatterAllgatherBroadcastKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  const int N = devComm.nRanks;
  const int rank = devComm.rank;
  // Even split with the remainder folded into the last rank's slice (shared with
  // the host unit tests via gin_sdma::sagChunk). Host guarantees count >= N.
  const size_t baseCount = count / (size_t)N;
  const gin_sdma::Chunk myChunk = gin_sdma::sagChunk(count, N, rank);
  const size_t myCount = myChunk.count;
  const size_t myByteOff = myChunk.eltOffset * sizeof(T);
  const size_t myBytes = myCount * sizeof(T);

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;

  ncclGin gin { devComm, /*context=*/0 };
  const unsigned int sigScatter = 0;
  const unsigned int sigGather = 1;
  const uint64_t baseScatter = gin.readSignal(sigScatter);
  const uint64_t baseGather = gin.readSignal(sigGather);

  // Entry barrier: every rank's recvbuff quiescent before the root scatters.
  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  // --- Phase 1: scatter chunk r -> rank r (root only) ---
  if (rank == root) {
    // The root's own slot is never written by the allgather (peers only forward
    // their own slice), so fill it locally from the source. Independent of the
    // allgather puts below (which read the root slice from sendwin), so no
    // intra-CTA ordering is required. No-op when in-place (local src == dst).
    T* lsrc = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* ldst = (T*)ncclGetLocalPointer(recvwin, recvoffset);
    if (lsrc != ldst) {
      const size_t myElt = (size_t)rank * baseCount;
      BroadcastLocalCopy<T>(ldst + myElt, lsrc + myElt, myCount, tid, nthreads);
    }
    for (int r = tid; r < N; r += nthreads) {
      if (r == root) continue;
      const gin_sdma::Chunk rChunk = gin_sdma::sagChunk(count, N, r);
      const size_t rOff = rChunk.eltOffset * sizeof(T);
      // Backend segments large copies internally (<=128 MiB, 30-bit-safe).
      gin.put(ncclTeamWorld(devComm), r,
          recvwin, recvoffset + rOff,
          sendwin, sendoffset + rOff,
          rChunk.count * sizeof(T), ncclGin_SignalInc{sigScatter});
    }
    // No intermediate flush: the scatter and allgather sources are both the
    // read-only sendwin, so there is no source-reuse hazard, and completion is
    // signal-based. Skipping it lets the scatter and allgather puts pipeline;
    // the single flush at the end drains all dirty queues.
  } else {
    // Wait (CTA-wide) for my scatter slice before I forward it in the allgather.
    gin.waitSignal(ncclCoopCta(), sigScatter, baseScatter + 1);
  }

  // --- Phase 2: in-place allgather of the N slices ---
  // My slice source: the root reads its stable sendwin (avoids a local-copy vs
  // put ordering hazard); non-roots read the scatter result in their recvwin
  // (made visible by the waitSignal above).
  ncclWindow_t myWin = (rank == root) ? sendwin : recvwin;
  const size_t myWinOff = (rank == root) ? (sendoffset + myByteOff) : (recvoffset + myByteOff);
  for (int r = tid; r < N; r += nthreads) {
    if (r == rank) continue;
    // Backend segments large copies internally (<=128 MiB, 30-bit-safe).
    gin.put(ncclTeamWorld(devComm), r,
        recvwin, recvoffset + myByteOff,
        myWin, myWinOff,
        myBytes, ncclGin_SignalInc{sigGather});
  }
  // Every rank (root included) receives exactly N-1 allgather puts.
  gin.waitSignal(ncclCoopCta(), sigGather, baseGather + (uint64_t)(N - 1));
  gin.flush(ncclCoopCta());
}

// Large-message Broadcast via a device-side PIPELINED RING (the van de Geijn ring
// shape RCCL's host path uses). Motivation: the SAG tier above still carries a
// serial, root-only scatter phase -- ~M bytes leaving one GPU on one SDMA channel
// -- that stops being hidden as M grows, so SAG plateaus (~229 GB/s @>=256 MiB)
// while the host ring keeps climbing (~322 GB/s @2 GiB). The ring removes that
// hotspot entirely:
//
//   The message is split into `nChunks` chunks streamed around the ring
//   root -> root+1 -> ... -> root+(N-1). Every rank forwards each chunk it
//   receives to its single successor, so at steady state all N-1 ring links
//   carry DIFFERENT chunks concurrently, each driven by a different rank's SDMA
//   engine (unlike SAG's scatter, where one root engine drives all N-1 puts).
//   Deep pipelining across chunks hides the (N-1)-hop fill latency.
//
// Ring position is root-relative: pos = (rank - root + N) % N. pos 0 = root
// (source; sends only, receives none), pos N-1 = tail (receives only, forwards
// none), interior ranks receive then forward. A single signal index counts
// arrivals: the per-source SDMA queue is in-order, so chunk c has landed once my
// signal reaches base + c + 1; an interior rank waits that value before
// forwarding chunk c. The wait graph is the linear chain pos0->..->pos(N-1) and
// the root never waits, so it cannot deadlock.
//
// A single CTA drives the pipeline: the puts are SDMA-offloaded (tid 0 enqueues
// the descriptor; the DMA engine performs the copy), so one CTA saturates the
// link and the per-chunk ordering stays trivially sequential on tid 0's queue.
// Only the entry barrier is kept (recvbuff quiescent before the predecessor
// writes -- the initData memset race, as in the flat/SAG paths).
//
// nChunks is host-resolved (gin_sdma::bcastRingChunks; env
// NCCL_GIN_ANVIL_BCAST_RING_CHUNKS) and passed via the threshold launcher.
template <typename T>
__global__ void GinRingBroadcastKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t nChunksArg) {
  if (blockIdx.x != 0) return;                 // single CTA drives the ring
  const int N = devComm.nRanks;
  const int rank = devComm.rank;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const int pos = (rank - root + N) % N;
  const bool isRoot = (pos == 0);
  const bool isTail = (pos == N - 1);
  const int nextRank = (rank + 1) % N;         // ring successor

  // Effective pipeline depth: never more chunks than elements (keeps every chunk
  // non-empty); uniform across ranks since count/nChunksArg are identical.
  int C = (int)((size_t)nChunksArg < count ? (size_t)nChunksArg : count);
  if (C < 1) C = 1;

  ncclGin gin { devComm, /*context=*/0 };
  const unsigned int sig = 0;
  const uint64_t base = gin.readSignal(sig);

  // Entry barrier: every rank's recvbuff quiescent before its predecessor writes.
  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, /*block=*/0 };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  // Root fills its own recvbuff from the source (OOP only; no-op in-place). It is
  // never written by a peer, and its puts read the stable sendwin, so this local
  // copy is independent of the forwarding loop below.
  if (isRoot) {
    T* lsrc = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* ldst = (T*)ncclGetLocalPointer(recvwin, recvoffset);
    if (lsrc != ldst) BroadcastLocalCopy<T>(ldst, lsrc, count, tid, nthreads);
  }

  // Pipeline: receive chunk c (interior/tail) then forward it (root/interior).
  for (int c = 0; c < C; c++) {
    const gin_sdma::Chunk ck = gin_sdma::sagChunk(count, C, c);
    const size_t offBytes = ck.eltOffset * sizeof(T);
    const size_t cBytes   = ck.count * sizeof(T);

    if (!isRoot) {
      // Chunk c has landed once my signal reaches base + c + 1 (in-order queue).
      gin.waitSignal(ncclCoopCta(), sig, base + (uint64_t)(c + 1));
    }
    if (!isTail && tid == 0) {
      // Root forwards from its stable sendwin; interior ranks forward the chunk
      // they just received into their recvwin.
      ncclWindow_t srcWin = isRoot ? sendwin : recvwin;
      const size_t srcOff = (isRoot ? sendoffset : recvoffset) + offBytes;
      gin.put(ncclTeamWorld(devComm), nextRank,
          recvwin, recvoffset + offBytes,
          srcWin, srcOff,
          cBytes, ncclGin_SignalInc{sig});
    }
  }

  gin.flush(ncclCoopCta());
}

// ---------------------------------------------------------------------------
// SM-driven pipelined ring broadcast (attack (a): replace the single-queue SDMA
// put with direct xGMI peer stores by the whole CTA). Where GinRingBroadcastKernel
// forwards each chunk with one SDMA descriptor issued by tid==0 (a latency-bound,
// completion-signal-gated hop that plateaus ~13 GB/s regardless of N), this kernel
// forwards with all `blockDim.x` threads issuing vectorized 16-byte stores straight
// into the successor's recvbuff via ncclGetLsaPointer -- the same SM-copy mechanism
// the LSA fan-out (BroadcastLsaDirect) uses, but distributed around the ring so
// every GPU drives only its one successor link and all N-1 links run in parallel.
//
// Parallelism / correctness model:
//  * Each CTA owns a contiguous stripe [sBase,sEnd) of the buffer and runs an
//    INDEPENDENT ring pipeline on it, so multiple CTAs (deviceCtaCount, one LSA
//    barrier line each) stripe the message for aggregate xGMI bandwidth. Block i
//    only ever touches block i's stripe on every rank, so a per-block-index LSA
//    barrier is a sufficient team sync (no grid-wide sync needed).
//  * The stripe is split into C chunks. At pipeline step s the rank at ring
//    position p forwards chunk c = s-p to its successor (0<=c<C, p<N-1). A rank
//    reads chunk c from its OWN recvbuff (written by its predecessor at step s-1)
//    and writes it into the successor's recvbuff; the per-step release barrier
//    publishes that write before the successor reads it next step. The root reads
//    the source (sendwin OOP / recvwin in-place) and, out-of-place, also fills its
//    own recvbuff chunk so it ends with the full result. The tail (p==N-1) never
//    forwards -- it only receives. Total steps = C + N - 2, identical on every
//    rank (count/gridDim/C/N match), so all ranks issue the same barrier sequence.
template <typename T>
__global__ void GinRingSmBroadcastKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t nChunksArg) {
  const int N = devComm.nRanks;
  const int rank = devComm.rank;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const int pos = (rank - root + N) % N;
  const bool isRoot = (pos == 0);
  const int nextRank = (rank + 1) % N;          // ring successor (global rank == LSA index, single node)

  ncclTeam lsa = ncclTeamLsa(devComm);

  // This CTA's contiguous stripe of the buffer (balanced split over gridDim.x).
  const size_t sBase  = (count * (size_t)blockIdx.x) / gridDim.x;
  const size_t sEnd   = (count * (size_t)(blockIdx.x + 1)) / gridDim.x;
  const size_t sCount = sEnd - sBase;

  // Pipeline depth for this stripe: never more chunks than elements.
  int C = (int)((size_t)nChunksArg < sCount ? (size_t)nChunksArg : sCount);
  if (C < 1) C = 1;

  // Peer/local base pointers (element typed) at the collective offset.
  T* myRecv  = (T*)ncclGetLocalPointer(recvwin, recvoffset);
  T* mySend  = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  T* peerRecv = (T*)ncclGetLsaPointer(recvwin, recvoffset, nextRank);
  const bool inPlace = (mySend == myRecv);
  // Root forwards from its stable source; interior ranks forward what they received.
  T* fwdSrc = isRoot ? (inPlace ? myRecv : mySend) : myRecv;

  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };

  // Entry barrier: every rank's recvbuff quiescent (post-memset, rank present)
  // before any predecessor writes into it.
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  int totalSteps = C + (N - 1) - 1;             // C + N - 2
  if (totalSteps < 1) totalSteps = 1;

  for (int s = 0; s < totalSteps; s++) {
    const int c = s - pos;
    const bool active = (pos < N - 1) && (c >= 0) && (c < C);
    if (active) {
      const size_t cStart = sBase + (sCount * (size_t)c) / C;
      const size_t cEnd   = sBase + (sCount * (size_t)(c + 1)) / C;
      const size_t cCount = cEnd - cStart;
      // Push chunk c into the successor's recvbuff with the whole CTA.
      BroadcastLocalCopy<T>(peerRecv + cStart, fwdSrc + cStart, cCount, tid, nthreads);
      // Root out-of-place: also materialize its own recvbuff chunk.
      if (isRoot && !inPlace) {
        BroadcastLocalCopy<T>(myRecv + cStart, mySend + cStart, cCount, tid, nthreads);
      }
    }
    // Publish this step's peer writes before any successor reads them next step.
    bar.sync(ncclCoopCta(), cuda::memory_order_release);
  }
}

// ---------------------------------------------------------------------------
// MULTI-RING SM broadcast (attack (a), fixed): the single-orientation SM ring
// above tops out at ~one xGMI link's rate (~58 GB/s @8 GPUs) because every CTA
// forwards to the SAME successor (rank+1), so all CTAs on a GPU pile onto ONE
// outgoing link. The host ring hits ~320 GB/s by running many channels over
// DIFFERENT ring permutations so each GPU drives all its links at once (verified:
// RCCL builds nChannels=28 rings with rotated orderings). This kernel does the
// same: it uses every stride s coprime to N (gcd(s,N)=1) as an independent
// Hamiltonian ring (order root, root+s, root+2s, ...), and assigns CTA b the
// stride strides[b % nStrides]. Successor = (rank+s)%N differs per stride, so
// across CTAs each GPU forwards on nStrides distinct links in parallel. Each CTA
// still owns a contiguous buffer stripe and runs the same pipelined SM-store ring
// (per-block LSA barrier line); stripes partition the buffer, so every element is
// broadcast exactly once via whichever ring its CTA uses. N=8 => strides {1,3,5,7}
// (4 links); N=4 => {1,3} (2 links).
template <typename T>
__global__ void GinRingSmMultiBroadcastKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t nChunksArg) {
  const int N = devComm.nRanks;
  const int rank = devComm.rank;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  // Strides coprime to N form single Hamiltonian rings; each is one link direction.
  int strides[64];
  int nStrides = 0;
  for (int s = 1; s < N && nStrides < 64; s++) {
    int a = s, b = N;
    while (b) { int t = a % b; a = b; b = t; }   // gcd(s,N)
    if (a == 1) strides[nStrides++] = s;
  }
  if (nStrides == 0) { strides[0] = 1; nStrides = 1; }   // N==1 guard

  const int s = strides[blockIdx.x % nStrides];
  int sinv = 1;                                          // modular inverse of s mod N
  for (int k = 1; k < N; k++) { if ((s * k) % N == 1) { sinv = k; break; } }

  const int pos = (int)(((long)(((rank - root) % N + N) % N) * sinv) % N);
  const bool isRoot = (pos == 0);
  const int nextRank = (rank + s) % N;                   // this ring's successor

  const size_t sBase  = (count * (size_t)blockIdx.x) / gridDim.x;
  const size_t sEnd   = (count * (size_t)(blockIdx.x + 1)) / gridDim.x;
  const size_t sCount = sEnd - sBase;

  int C = (int)((size_t)nChunksArg < sCount ? (size_t)nChunksArg : sCount);
  if (C < 1) C = 1;

  T* myRecv   = (T*)ncclGetLocalPointer(recvwin, recvoffset);
  T* mySend   = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  T* peerRecv = (T*)ncclGetLsaPointer(recvwin, recvoffset, nextRank);
  const bool inPlace = (mySend == myRecv);
  T* fwdSrc = isRoot ? (inPlace ? myRecv : mySend) : myRecv;

  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  int totalSteps = C + (N - 1) - 1;
  if (totalSteps < 1) totalSteps = 1;

  for (int step = 0; step < totalSteps; step++) {
    const int c = step - pos;
    const bool active = (pos < N - 1) && (c >= 0) && (c < C);
    if (active) {
      const size_t cStart = sBase + (sCount * (size_t)c) / C;
      const size_t cEnd   = sBase + (sCount * (size_t)(c + 1)) / C;
      const size_t cCount = cEnd - cStart;
      BroadcastLocalCopy<T>(peerRecv + cStart, fwdSrc + cStart, cCount, tid, nthreads);
      if (isRoot && !inPlace) {
        BroadcastLocalCopy<T>(myRecv + cStart, mySend + cStart, cCount, tid, nthreads);
      }
    }
    bar.sync(ncclCoopCta(), cuda::memory_order_release);
  }
}

// ---------------------------------------------------------------------------
// FULL edge-disjoint multi-ring SM broadcast (attack (a), maximal): use ALL of
// a GPU's outgoing xGMI links by decomposing the complete symmetric digraph K_N*
// into N-1 arc-disjoint directed Hamiltonian cycles (Tillson: exists for N!=4,6).
// The coprime-stride set (GinRingSmMultiBroadcastKernel) only reaches 4 of 8
// GPUs' 7 links (offsets 1,3,5,7 -- even offsets don't form single cycles); this
// path uses a host-computed decomposition covering every arc, so each GPU drives
// all N-1 links at once (mirrors how RCCL's 28 rotated-ring channels saturate the
// fabric). The N-1 cycles are found by verified backtracking on the host and
// uploaded to constant memory; CTA b runs ring b%nRings on buffer stripe b.
#define BCAST_RING_MAXR 16
#define BCAST_RING_MAXN 16
__constant__ int c_bcastNRings;
__constant__ int c_bcastRingSucc[BCAST_RING_MAXR * BCAST_RING_MAXN];  // [ring*N + rank] -> successor rank
__constant__ int c_bcastRingPos[BCAST_RING_MAXR * BCAST_RING_MAXN];   // [ring*N + rank] -> position in cycle
__constant__ int c_bcastStream;  // 1 => peer writes use nontemporal system-scope b128 stores

// Set by BroadcastRunColl once the edge-disjoint ring decomposition for the current
// rank count is built + uploaded to constant memory; read by BroadcastDeviceTime to
// gate whether in-kernel timing can run the ring body (needs the tables present).
static int g_bcastBuiltN = -1;
static int g_bcastBuiltNRings = 0;

// Body extracted so the device-timing kernel below can run it skip+loop times
// under one persistent launch. Each call re-creates its LSA barrier session and
// re-syncs (like the AllGather timed body), so looping is self-contained.
template <typename T>
__device__ void ginRingSmTableBroadcastBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm& devComm, size_t nChunksArg) {
  const int N = devComm.nRanks;
  const int rank = devComm.rank;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const int nRings = c_bcastNRings;
  const int ring = (int)(blockIdx.x % (unsigned)nRings);
  const int nextRank = c_bcastRingSucc[ring * N + rank];
  const int posCur   = c_bcastRingPos[ring * N + rank];
  const int posRoot  = c_bcastRingPos[ring * N + root];
  const int pos = (posCur - posRoot + N) % N;   // hops from root along this ring
  const bool isRoot = (pos == 0);

  const size_t sBase  = (count * (size_t)blockIdx.x) / gridDim.x;
  const size_t sEnd   = (count * (size_t)(blockIdx.x + 1)) / gridDim.x;
  const size_t sCount = sEnd - sBase;

  int C = (int)((size_t)nChunksArg < sCount ? (size_t)nChunksArg : sCount);
  if (C < 1) C = 1;

  T* myRecv   = (T*)ncclGetLocalPointer(recvwin, recvoffset);
  T* mySend   = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  T* peerRecv = (T*)ncclGetLsaPointer(recvwin, recvoffset, nextRank);
  const bool inPlace = (mySend == myRecv);
  T* fwdSrc = isRoot ? (inPlace ? myRecv : mySend) : myRecv;

  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  int totalSteps = C + (N - 1) - 1;
  if (totalSteps < 1) totalSteps = 1;

  for (int step = 0; step < totalSteps; step++) {
    const int c = step - pos;
    const bool active = (pos < N - 1) && (c >= 0) && (c < C);
    if (active) {
      const size_t cStart = sBase + (sCount * (size_t)c) / C;
      const size_t cEnd   = sBase + (sCount * (size_t)(c + 1)) / C;
      const size_t cCount = cEnd - cStart;
      // Streaming (nontemporal system-scope b128) peer stores vs cached uint4 is an
      // A/B knob (c_bcastStream); the local root self-copy stays cached.
      if (c_bcastStream) {
        BroadcastPeerCopyStream<T>(peerRecv + cStart, fwdSrc + cStart, cCount, tid, nthreads);
      } else {
        BroadcastLocalCopy<T>(peerRecv + cStart, fwdSrc + cStart, cCount, tid, nthreads);
      }
      if (isRoot && !inPlace) {
        BroadcastLocalCopy<T>(myRecv + cStart, mySend + cStart, cCount, tid, nthreads);
      }
    }
    bar.sync(ncclCoopCta(), cuda::memory_order_release);
  }
}

template <typename T>
__global__ void GinRingSmTableBroadcastKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t nChunksArg) {
  ginRingSmTableBroadcastBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm, nChunksArg);
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
// Device-timing kernel (shared gin_devtime methodology): run skip+loop back-to-back
// ring bodies under ONE persistent launch, bracketing only the timed region with
// wall_clock64() per CTA. min(start)..max(end) over CTAs is the grid busy window.
template <typename T>
__global__ void GinRingSmTableBroadcastTimedKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t nChunksArg, int loop, int skip, long long* start_time, long long* end_time) {
  for (int i = 0; i < skip + loop; i++) {
    if (i == skip) {
      __syncthreads();
      if (threadIdx.x == 0) start_time[blockIdx.x] = wall_clock64();
    }
    ginRingSmTableBroadcastBody<T>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm, nChunksArg);
  }
  __syncthreads();
  if (threadIdx.x == 0) end_time[blockIdx.x] = wall_clock64();
}
#endif

// ---------------------------------------------------------------------------
// POINT-TO-POINT edge-disjoint ring broadcast: same 7-ring decomposition as the
// table kernel, but the per-step grid-wide LSA barrier is replaced by lightweight
// per-hop successor signaling. Each hop SM-copies chunk c into the successor's
// recvbuff, then bumps ONLY the successor's per-CTA GIN signal (signalPeer: an
// IPC/LSA atomic-add, no SDMA queue); the successor waits on its OWN signal before
// forwarding. So the N-1 rings pipeline fully independently -- a straggler on one
// ring no longer stalls the others at a shared barrier -- which is where the host
// generic ring (per-channel signaling, no grid-wide sync) beat the barrier kernel.
// Per-CTA signal `sig=blockIdx.x` (ginSignalCount=deviceCtaCount) has exactly ONE
// writer (the predecessor's CTA b) so the atomic count is race-free; base is re-read
// per launch so it accumulates across perf iters. Only the ENTRY LSA barrier is kept
// (recvbuff quiescent before the predecessor writes -- the initData memset race).
// Wait graph is the linear chain pos0->..->pos(N-1) per ring; root never waits, tail
// never forwards, so no deadlock.
template <typename T>
__global__ void GinRingSmP2PBroadcastKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, size_t nChunksArg) {
  const int N = devComm.nRanks;
  const int rank = devComm.rank;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const int nRings = c_bcastNRings;
  const int ring = (int)(blockIdx.x % (unsigned)nRings);
  const int nextRank = c_bcastRingSucc[ring * N + rank];
  const int posCur   = c_bcastRingPos[ring * N + rank];
  const int posRoot  = c_bcastRingPos[ring * N + root];
  const int pos = (posCur - posRoot + N) % N;
  const bool isRoot = (pos == 0);
  const bool isTail = (pos == N - 1);

  const size_t sBase  = (count * (size_t)blockIdx.x) / gridDim.x;
  const size_t sEnd   = (count * (size_t)(blockIdx.x + 1)) / gridDim.x;
  const size_t sCount = sEnd - sBase;

  int C = (int)((size_t)nChunksArg < sCount ? (size_t)nChunksArg : sCount);
  if (C < 1) C = 1;

  T* myRecv   = (T*)ncclGetLocalPointer(recvwin, recvoffset);
  T* mySend   = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  T* peerRecv = (T*)ncclGetLsaPointer(recvwin, recvoffset, nextRank);
  const bool inPlace = (mySend == myRecv);
  T* fwdSrc = isRoot ? (inPlace ? myRecv : mySend) : myRecv;

  ncclGin gin { devComm, /*context=*/0 };
  const unsigned int sig = (unsigned int)blockIdx.x;
  const uint64_t sigBase = gin.readSignal(sig);
  ncclGinAnvilSdmaGPUContext* rsCtx = (ncclGinAnvilSdmaGPUContext*)devComm.ginHandles[0];

  // Entry barrier only: recvbuff quiescent before the predecessor writes.
  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  // Root out-of-place: materialize its own recvbuff stripe (never peer-written).
  if (isRoot && !inPlace) {
    BroadcastLocalCopy<T>(myRecv + sBase, mySend + sBase, sCount, tid, nthreads);
  }

  for (int c = 0; c < C; c++) {
    const size_t cStart = sBase + (sCount * (size_t)c) / C;
    const size_t cEnd   = sBase + (sCount * (size_t)(c + 1)) / C;
    const size_t cCount = cEnd - cStart;

    // Interior/tail: chunk c has landed once my per-CTA signal reaches base+c+1.
    if (!isRoot) {
      gin.waitSignal(ncclCoopCta(), sig, sigBase + (uint64_t)(c + 1));
    }
    // Forward chunk c into the successor's recvbuff, then signal ONLY it.
    if (!isTail) {
      BroadcastLocalCopy<T>(peerRecv + cStart, fwdSrc + cStart, cCount, tid, nthreads);
      __syncthreads();               // all CTA stores issued
      __threadfence_system();        // ...and visible on the successor GPU
      if (tid == 0) {
        nccl::gin::anvil::detail::signalPeer(rsCtx, nextRank, sig, (uint64_t)1);
      }
    }
  }
}

// Host-side decomposition of K_N* into (N-1) arc-disjoint directed Hamiltonian
// cycles by verified backtracking (Tillson guarantees existence for N!=4,6; the
// search simply fails and we fall back to the coprime-stride kernel otherwise).
// Fills succ[ring*N+rank] and pos[ring*N+rank]; returns nRings (N-1) or 0.
struct BcastRingDecomp {
  int N;
  bool used[BCAST_RING_MAXN][BCAST_RING_MAXN];
  int order[BCAST_RING_MAXR][BCAST_RING_MAXN];
  long budget;
  bool extend(int r, int* path, bool* vis, int len) {
    if (--budget < 0) return false;
    const int cur = path[len - 1];
    if (len == N) {
      if (used[cur][0]) return false;                     // must close to start
      for (int i = 0; i < N; i++) used[path[i]][path[(i + 1) % N]] = true;
      for (int i = 0; i < N; i++) order[r][i] = path[i];
      if (solve(r + 1)) return true;
      for (int i = 0; i < N; i++) used[path[i]][path[(i + 1) % N]] = false;
      return false;
    }
    for (int nx = 1; nx < N; nx++) {                      // start vertex 0 fixed
      if (!vis[nx] && !used[cur][nx]) {
        vis[nx] = true; path[len] = nx;
        if (extend(r, path, vis, len + 1)) return true;
        vis[nx] = false;
      }
    }
    return false;
  }
  bool solve(int r) {
    if (r == N - 1) return true;                          // N-1 arc-disjoint cycles placed
    int path[BCAST_RING_MAXN]; bool vis[BCAST_RING_MAXN];
    for (int i = 0; i < N; i++) vis[i] = false;
    path[0] = 0; vis[0] = true;
    return extend(r, path, vis, 1);
  }
};

static int buildBcastRingDecomp(int N, int* succ, int* pos) {
  if (N < 2 || N > BCAST_RING_MAXN || (N - 1) > BCAST_RING_MAXR) return 0;
  static BcastRingDecomp d;                               // large; keep off-stack
  d.N = N;
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) d.used[i][j] = false;
  d.budget = 20000000L;                                   // ample for N<=8; else fall back
  if (!d.solve(0)) return 0;
  for (int r = 0; r < N - 1; r++)
    for (int k = 0; k < N; k++) {
      const int rk = d.order[r][k];
      succ[r * N + rk] = d.order[r][(k + 1) % N];
      pos[r * N + rk]  = k;
    }
  return N - 1;
}
#endif
#endif

testResult_t BroadcastRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  if (broadcast_grouped) {
    // nranks broadcasts with distinct roots inside one group — fuses into AllGatherV ring kernel
    if (deviceImpl != 0) return testNotImplemented;
    int nranks;
    NCCLCHECK(ncclCommCount(comm, &nranks));
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
    NCCLCHECK(ncclGroupStart());
    for (int k = 0; k < nranks; k++) {
      NCCLCHECK(ncclBroadcast(sptr, rptr + (size_t)k * count * wordSize(type),
                              count, type, /*root=*/k, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    return testSuccess;
  }

  if (deviceImpl == 0) {
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));

    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
#if NCCL_MAJOR >= 2 && NCCL_MINOR >= 2
    NCCLCHECK(ncclBroadcast(sptr, rptr, count, type, root, comm, stream));
#else
    if (rank == root) {
      NCCLCHECK(ncclBcast(sptr, count, type, root, comm, stream));
    } else {
      NCCLCHECK(ncclBcast(rptr, count, type, root, comm, stream));
    }
#endif
  } else {
    switch(deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
      case 3: {
        // Broadcast-specific LSA<->GIN threshold. Default = 256 KiB (full
        // message): on 8x MI355X (NCCL_GIN_TYPE=5) LSA wins <=256K and GIN wins
        // >=512K (measured 2026-07-24). Override with
        // NCCL_GIN_ANVIL_SDMA_THRESHOLD_BROADCAST, or the shared
        // NCCL_GIN_ANVIL_SDMA_THRESHOLD.
        static const size_t bcastThr = testResolveSdmaThreshold("NCCL_GIN_ANVIL_SDMA_THRESHOLD_BROADCAST", gin_sdma::kBroadcastSdmaThresholdDefault);

        // Large-message tier (§4.8): scatter + in-place allgather. Distributes
        // egress across all ranks to beat the flat fan-out's root-egress ceiling
        // (busBw = B_root_egress / N). Gated to large messages via
        // NCCL_GIN_ANVIL_BCAST_SCATTER_AG_MIN_BYTES (bytes, optional K/M/G
        // suffix; default 2 MiB; 0 = disable, keeping the proven flat path).
        // Crossover measured on 8x MI355X (-V 32, in-place, 2026-07-27): SAG is a
        // slight win at 2M (35.9 vs flat 33.4), decisive >=4M (4M 62 vs 42, 128M
        // 224 vs 60), and loses <2M (1M 19.4 vs 23.7). So default 2 MiB.
        static const size_t bcastSagMin = []() {
          size_t v = testParseSdmaThresholdEnv("NCCL_GIN_ANVIL_BCAST_SCATTER_AG_MIN_BYTES");
          return (v == TEST_SDMA_THRESHOLD_UNSET) ? gin_sdma::kBroadcastScatterAgMinDefault : v;
        }();
        // Pipelined-ring large tier (§ ring kernel): distributes forwarding across
        // all ranks with no serial root-scatter phase, to beat the SAG plateau at
        // very large messages. Opt-in via NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES
        // (0 = disabled, default) until a warm A/B vs SAG sets a default; the
        // pipeline depth is NCCL_GIN_ANVIL_BCAST_RING_CHUNKS (else size-derived).
        static const size_t bcastRingMin = []() {
          size_t v = testParseSdmaThresholdEnv("NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES");
          return (v == TEST_SDMA_THRESHOLD_UNSET) ? gin_sdma::kBroadcastRingMinDefault : v;
        }();
        static const size_t bcastRingChunksEnv =
            testParseSdmaThresholdEnv("NCCL_GIN_ANVIL_BCAST_RING_CHUNKS");
        // Ring forwarding backend: SM peer stores (default, attack (a)) vs the
        // original single-queue SDMA put. NCCL_GIN_ANVIL_BCAST_RING_SM=0 selects
        // the SDMA ring for A/B; anything else (unset/1) uses the SM-store ring.
        static const int bcastRingSm = []() {
          const char* e = getenv("NCCL_GIN_ANVIL_BCAST_RING_SM");
          return (e && e[0] == '0') ? 0 : 1;
        }();
        // Multi-ring (rotated stride orderings so each GPU drives all its xGMI
        // links) vs single-orientation SM ring. Default multi; set
        // NCCL_GIN_ANVIL_BCAST_RING_MULTI=0 for the single-orientation A/B.
        static const int bcastRingMulti = []() {
          const char* e = getenv("NCCL_GIN_ANVIL_BCAST_RING_MULTI");
          return (e && e[0] == '0') ? 0 : 1;
        }();
        // Point-to-point per-hop signaling vs the per-step grid-wide LSA barrier
        // table kernel. A/B (8x MI355X, 2026-08-04, @2 GiB) found the barrier table
        // kernel FASTER (232 vs P2P 191 GB/s): the barrier bundles the cross-GPU
        // release fence with the sync, whereas P2P pays a __threadfence_system per
        // chunk/hop and loses lock-step pipelining. So P2P is OFF by default (opt in
        // with NCCL_GIN_ANVIL_BCAST_RING_P2P=1 to reproduce the A/B).
        static const int bcastRingP2P = []() {
          const char* e = getenv("NCCL_GIN_ANVIL_BCAST_RING_P2P");
          return (e && e[0] == '1') ? 1 : 0;
        }();
        // In the -D 3 path `comm` is the ncclDevComm handle (testLaunchDeviceKernel*
        // casts it), NOT a real ncclComm_t -- so read nRanks from the devComm
        // struct rather than ncclCommCount (which would fault on a corrupted comm).
        const int sagRanks = (int)((struct ncclDevComm*)comm)->nRanks;
        const size_t msgBytes = count * wordSize(type);
        static const size_t bcastCtasEnv = BroadcastParseCtasEnv();
        const int hybridPool = gin_sdma::bcastHybridPoolCtas(deviceCtaCount);
        if (gin_sdma::bcastUseRing(msgBytes, count, sagRanks, bcastRingMin)) {
          const size_t nChunks = (size_t)gin_sdma::bcastRingChunks(msgBytes, bcastRingCtas(), bcastRingChunksEnv);
          // Full edge-disjoint decomposition (all N-1 links) built + uploaded
          // once per N; falls back to the coprime-stride kernel if unavailable.
          // File-scope so BroadcastDeviceTime can see whether the tables exist.
          int& builtN = g_bcastBuiltN;
          int& builtNRings = g_bcastBuiltNRings;
          if (bcastRingSm && bcastRingMulti && builtN != sagRanks) {
            static int h_succ[BCAST_RING_MAXR * BCAST_RING_MAXN];
            static int h_pos[BCAST_RING_MAXR * BCAST_RING_MAXN];
            builtNRings = buildBcastRingDecomp(sagRanks, h_succ, h_pos);
            if (builtNRings > 0) {
              // Streaming (nontemporal system-scope b128) peer stores default ON;
              // NCCL_GIN_ANVIL_BCAST_RING_STREAM=0 reverts to cached uint4 stores.
              int streamStores = []() {
                const char* e = getenv("NCCL_GIN_ANVIL_BCAST_RING_STREAM");
                return (e && e[0] == '0') ? 0 : 1;
              }();
              CUDACHECK(cudaMemcpyToSymbol(c_bcastNRings, &builtNRings, sizeof(int)));
              CUDACHECK(cudaMemcpyToSymbol(c_bcastRingSucc, h_succ, sizeof(int) * sagRanks * builtNRings));
              CUDACHECK(cudaMemcpyToSymbol(c_bcastRingPos, h_pos, sizeof(int) * sagRanks * builtNRings));
              CUDACHECK(cudaMemcpyToSymbol(c_bcastStream, &streamStores, sizeof(int)));
            }
            builtN = sagRanks;
          }
          // The SM ring kernels launch their own power-of-2 CTA count (bcastRingCtas,
          // default 128) instead of -V/deviceCtaCount: the pipeline is CTA-bound and
          // needs ~128 CTAs to saturate all xGMI links (238 -> 350 GB/s @2 GiB). The
          // P2P variant keeps deviceCtaCount (its per-CTA GIN signals are sized to it).
          const int ringCtas = bcastRingCtas();
          if (bcastRingSm && bcastRingMulti && builtNRings > 0 && bcastRingP2P) {
            TESTCHECK(testLaunchDeviceKernelThreshold(SPECIALIZE_KERNEL(GinRingSmP2PBroadcastKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, nChunks));
          } else if (bcastRingSm && bcastRingMulti && builtNRings > 0) {
            TESTCHECK(testLaunchDeviceKernelThresholdCtas(SPECIALIZE_KERNEL(GinRingSmTableBroadcastKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, nChunks, ringCtas));
          } else if (bcastRingSm && bcastRingMulti) {
            TESTCHECK(testLaunchDeviceKernelThresholdCtas(SPECIALIZE_KERNEL(GinRingSmMultiBroadcastKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, nChunks, ringCtas));
          } else if (bcastRingSm) {
            TESTCHECK(testLaunchDeviceKernelThresholdCtas(SPECIALIZE_KERNEL(GinRingSmBroadcastKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, nChunks, ringCtas));
          } else {
            TESTCHECK(testLaunchDeviceKernelThreshold(SPECIALIZE_KERNEL(GinRingBroadcastKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, nChunks));
          }
          return testSuccess;
        }
        if (gin_sdma::bcastUseScatterAllgather(msgBytes, count, sagRanks, bcastSagMin)) {
          const int sagGrid = gin_sdma::bcastSagCtas(msgBytes, sagRanks, bcastThr, bcastCtasEnv, hybridPool);
          TESTCHECK(testLaunchDeviceKernelCtas(SPECIALIZE_KERNEL(GinScatterAllgatherBroadcastKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, sagGrid));
          return testSuccess;
        }
        {
          const int llSlots = g_bcastLLHandle.nSlots;
          const int hybridGrid = gin_sdma::bcastHybridCtas(msgBytes, bcastThr, llSlots, g_bcastLLMaxBytes, bcastCtasEnv, hybridPool);
#if defined(BC_HAVE_LL)
          TESTCHECK(testLaunchDeviceKernelThresholdLL(SPECIALIZE_KERNEL(GinHybridBroadcastKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, bcastThr, g_bcastLLHandle, hybridGrid));
#else
          TESTCHECK(testLaunchDeviceKernelThresholdCtas(SPECIALIZE_KERNEL(GinHybridBroadcastKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, bcastThr, hybridGrid));
#endif
        }
        return testSuccess;
      }
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

// Device-side (in-kernel wall_clock64) timing for the GIN-SDMA Broadcast (-D 3),
// via the shared gin_devtime scaffold. Opt-in via --device_timing: 1=augment
// (print an extra #[bcast-devtime] line next to the graph numbers), 2=device-time-only
// (report the in-kernel latency as the out-of-place metric; in-place keeps normal
// timing). loop/skip come from --devtime_loop/--devtime_skip (default 10/10); size-
// tier overrides via --devtime_loop_mid/_large and --devtime_skip_mid/_large. Times
// the default large tier -- the SM edge-disjoint ring -- so it only runs when the
// ring is the active tier (message >= cutover, decomposition built during warmup);
// otherwise it leaves the host-timed metric untouched.
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
testResult_t BroadcastDeviceTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec) {
  if (!deviceTimingMode) return testSuccess;

  // Only the GIN hybrid impl (-D 3) provisions the GIN signals/barriers the timed
  // body relies on. Other device impls would fault or hang on missing GIN state.
  if (deviceImpl != 3) return testSuccess;

  const size_t count = args->nbytes / wordSize(type);   // broadcast message count
  if (count == 0 || devtimeLoop < 1) return testSuccess;
  const size_t msgBytes = count * wordSize(type);

  int loop = devtimeLoop;
  int skip = devtimeSkip < 0 ? 0 : devtimeSkip;
  if (devtimeLoopLarge > 0 && msgBytes >= (size_t)64 * 1024 * 1024) {
    loop = devtimeLoopLarge;
    if (devtimeSkipLarge >= 0) skip = devtimeSkipLarge;
    else skip = (skip < 1) ? skip : 1;
  } else if (devtimeLoopMid > 0 && msgBytes >= (size_t)8 * 1024 * 1024) {
    loop = devtimeLoopMid;
    if (devtimeSkipMid >= 0) skip = devtimeSkipMid;
    else skip = (skip < 2) ? skip : 2;
  }

  static const size_t bcastRingMin = []() {
    size_t v = testParseSdmaThresholdEnv("NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES");
    return (v == TEST_SDMA_THRESHOLD_UNSET) ? gin_sdma::kBroadcastRingMinDefault : v;
  }();
  const int nRanks = (int)(args->devComms[0].nRanks);
  // Only device-time the ring when it is the active tier and its tables are built.
  if (!gin_sdma::bcastUseRing(msgBytes, count, nRanks, bcastRingMin)) return testSuccess;
  if (g_bcastBuiltNRings <= 0) return testSuccess;

  auto kernel = SPECIALIZE_KERNEL(GinRingSmTableBroadcastTimedKernel, type, op);
  if (kernel == nullptr) return testSuccess;

  static const size_t chunksEnv = testParseSdmaThresholdEnv("NCCL_GIN_ANVIL_BCAST_RING_CHUNKS");
  const int gridCtas = bcastRingCtas();
  const size_t nChunks = (size_t)gin_sdma::bcastRingChunks(msgBytes, gridCtas, chunksEnv);
  double devUs = 0.0;
  TESTCHECK(gin_devtime::measure(args, gridCtas, loop,
      [&](int i, long long* d_start, long long* d_end) {
        ncclDevComm* devComm = args->devComms + i;
        ncclWindow_t sendwin = (ncclWindow_t)(in_place ? args->recvRegHandles[i] : args->sendRegHandles[i]);
        ncclWindow_t recvwin = (ncclWindow_t)args->recvRegHandles[i];
        kernel<<<gridCtas, 512, 0, args->streams[i]>>>(sendwin, 0, recvwin, 0, count, root, *devComm,
                 nChunks, loop, skip, d_start, d_end);
      },
      &devUs));

  if (outDeltaSec != nullptr) {
    *outDeltaSec = (devUs > 0.0) ? devUs * 1.0e-6 : -1.0;
    return testSuccess;
  }

  if (args->proc == 0 && args->thread == 0 && devUs > 0.0) {
    double sec = devUs * 1.0e-6;
    double algBw = (double)msgBytes / 1.0e9 / sec;   // broadcast busbw factor = 1
    snprintf(args->devtimeAugmentLine, sizeof(args->devtimeAugmentLine),
             "#[bcast-devtime] size %12zu B  ctas %2d  loop %2d skip %2d  devtime %10.2f us  algbw %8.2f GB/s  busbw %8.2f GB/s\n",
             msgBytes, gridCtas, loop, skip, devUs, algBw, algBw);
  }
  return testSuccess;
}
#else
testResult_t BroadcastDeviceTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec) {
  return testSuccess;  // device API path not available in this build
}
#endif

struct testColl broadcastTest = {
  "Broadcast",
  BroadcastGetCollByteCount,
  BroadcastInitData,
  BroadcastGetBw,
  BroadcastRunColl,
  BroadcastGetAlgoProtoChannels,
  NULL,
  BroadcastDeviceTime
};

void BroadcastGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  BroadcastGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t BroadcastRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &broadcastTest;
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

  if (broadcast_grouped) {
    for (int i = 0; i < type_count; i++)
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
    return testSuccess;
  }

  int begin_root, end_root;
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
  .getBuffSize = BroadcastGetBuffSize,
  .runTest = BroadcastRunTest,
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  .getDevCommRequirements = BroadcastGetDevCommRequirements
#endif
};
