/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <atomic>
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
#include "gin_sdma_broadcast_policy.h"  // gin_sdma::parseSize / resolveLLCap
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
// Small -D 3 messages take the LL tier instead (see GinReduceLLKernel).
// NCCL_GIN_ANVIL_REDUCE_LL_MAX_BYTES=<bytes> sets the cutover (0 = disable),
// clamped to kReduceLLMaxBytes. Every rank must use the same value. The LL
// buffer takes about 4*nRanks*cap bytes per rank (two epoch parities of 16-byte
// lines carrying 8 data bytes each), so by default the cap is sized to keep it
// within kReduceLLDefaultBufferBytes.
static constexpr size_t kReduceLLMaxBytes = 1 << 20;
static constexpr size_t kReduceLLDefaultBufferBytes = 16 << 20;
static constexpr int kReduceLLMaxCtas = 256;
static constexpr int kReduceLLUnitsPerCta = 512;  // 8-byte units per CTA, one per thread
static_assert(kReduceLLMaxBytes / 8 <= (size_t)kReduceLLMaxCtas * kReduceLLUnitsPerCta,
              "an LL block must never hold more than kReduceLLUnitsPerCta units");

// SDMA-tier credit flags, per rank (uint32): ready[nRanks] posted by each peer
// at kernel start, done[nRanks] posted by each owner to the root, then the
// launch sequence of each CTA (local only). Then, 16-byte aligned, the tail
// scratch: the root receives each owner's up to nRanks-2 tail packs there.
static constexpr int kReduceFlagMaxCtas = 128;
__host__ __device__ inline size_t sdmaReduceTailOffset(int nRanks) {
  return ((size_t)(2 * nRanks + kReduceFlagMaxCtas) * sizeof(uint32_t) + 15) & ~(size_t)15;
}

// The resolved LL cap, whether the SDMA tier is on, and the handles
// ncclDevCommCreate fills in. With -p 1 every host thread builds its own
// requirements and devComms, so each thread
// keeps its own copy. Otherwise the main thread builds them for every thread
// before starting the others, so all threads must read that one copy: a
// thread_local alone leaves threads 1..N-1 with a zero cap and zero handles,
// and their ranks then launch a different kernel than thread 0's and hang.
struct ReduceDevState {
  size_t llMaxBytes;
  bool sdma;
  ncclLLA2AHandle llHandle;
  ncclDevResourceHandle_t flagHandle;
};
extern int parallel_init;
extern int nThreads;
extern int nGpus;
static ReduceDevState g_reduceSharedState = {};
thread_local ReduceDevState g_reduceThreadState = {};
static ReduceDevState& reduceState() {
  return parallel_init ? g_reduceThreadState : g_reduceSharedState;
}
// Each thread that builds requirements links its own nodes into its own list.
thread_local ncclDevResourceRequirements g_reduceLLReq = {};
thread_local ncclDevResourceRequirements g_reduceFlagReq = {};

static int reduceLLCtas(size_t bytes) {
  const size_t units = bytes / 8;
  const size_t ctas = (units + kReduceLLUnitsPerCta - 1) / kReduceLLUnitsPerCta;
  return (int)std::max<size_t>(1, std::min<size_t>(ctas, kReduceLLMaxCtas));
}

// Whether a -D 3 message of `bytes` goes to the LL tier. Every element type is
// at most 8 bytes, so 16-byte-aligned counts always fill whole LL units.
static bool reduceUseLL(size_t bytes, int nRanks) {
  const ReduceDevState& st = reduceState();
  if (bytes == 0 || bytes > st.llMaxBytes) return false;
  const int ctas = reduceLLCtas(bytes);
  const size_t unitsPerCta = (bytes / 8 + ctas - 1) / ctas;
  return (size_t)nRanks * unitsPerCta + 1 <= st.llHandle.nSlots;
}

// Out-of-place -D 3 messages of at least this many bytes take the SDMA tier
// (GinSdmaReduceKernel) when it is on. NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCE,
// then NCCL_GIN_ANVIL_SDMA_THRESHOLD, override the default (0 = every
// out-of-place message). Every rank must use the same value. The tiled kernel
// speeds up with more CTAs and the SDMA tier mostly does not, so the default
// grows with -V: on MI450 with 4 ranks (float sum, power-of-two sizes) the SDMA
// tier is faster from 32 MiB at 16 CTAs, 64 MiB at 32, 1 GiB at 64 and 2 GiB at
// 128, and each default sits between that size and the one below it. With
// fewer than 2*nRanks CTAs the owners store to the root from CUs, which is far
// slower (about 50 ms at 1 GiB with 4 CTAs), so the default leaves the tier off.
static size_t reduceSdmaThreshold(int ctas, int nRanks) {
  static const size_t envThreshold = gin_sdma::resolveThreshold(
      gin_sdma::parseSize(getenv("NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCE")),
      gin_sdma::parseSize(getenv("NCCL_GIN_ANVIL_SDMA_THRESHOLD")), gin_sdma::kThresholdUnset);
  if (envThreshold != gin_sdma::kThresholdUnset) return envThreshold;
  if (ctas < 2 * nRanks) return gin_sdma::kThresholdUnset;
  if (ctas <= 16) return (size_t)24 << 20;
  if (ctas <= 32) return (size_t)48 << 20;
  if (ctas <= 64) return (size_t)768 << 20;
  return (size_t)1536 << 20;
}

// Whether a -D 3 message of `bytes` goes to the SDMA tier. The SDMA tier keeps
// receive slots in the non-root recvbuffs, so in-place messages never do.
static bool reduceUseSdma(size_t bytes, int nRanks, bool inPlace) {
  return reduceState().sdma && !inPlace && bytes >= reduceSdmaThreshold(deviceCtaCount, nRanks);
}
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
// Which kernel launched, for tests that must tell the tiers apart: the data is
// correct whichever tier runs, so #wrong alone cannot show that a tier was
// skipped. Opt-in via RCCL_TESTS_REDUCE_TIER_LOG, once per tier, like
// broadcast.cu's bcastReportTier. It goes to stderr: the in-place pass can take
// another tier than the out-of-place pass, and on stdout its first launch would
// land in the middle of the result row.
enum class ReduceLaunchTier { LL, Tiled, SdmaCu, SdmaStaged };

static void reduceReportTier(ReduceLaunchTier tier) {
  static const char* const kTierNames[] = { "ll", "tiled", "sdma-cu", "sdma-staged" };
  static const bool enabled = getenv("RCCL_TESTS_REDUCE_TIER_LOG") != nullptr;
  if (!enabled || !is_main_proc) return;
  static std::atomic<unsigned> reported{0};
  const unsigned bit = 1u << (unsigned)tier;
  if (reported.fetch_or(bit, std::memory_order_relaxed) & bit) return;
  fprintf(stderr, "#[reduce-tier] %s\n", kTierNames[(unsigned)tier]);
}
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
// Messages up to NCCL_GIN_ANVIL_REDUCE_LL_MAX_BYTES instead go to the root as
// LL packets in one hop with no barrier (GinReduceLLKernel).
//
// Out-of-place messages from the SDMA threshold up (reduceSdmaThreshold) take
// the SDMA tier, which moves the data with SDMA and reduces with CUs (GIN Anvil
// SDMA backend only). The N-1 non-root ranks each own one of N-1 equal
// partitions. Every rank gin.put()s its slice of each partition into the
// owner's recvbuff, one slot per sender (slot = sender < owner ? sender :
// sender - 1); the owner then folds its own sendbuff slice and the N-1 received
// slots in ascending source-rank order and stores the result into the root's
// recvbuff. Non-root recvbuffs double as the receive scratch, so in-place
// messages stay on the tiled kernel. Large partitions go out as several puts
// (chunks), each followed by a per-chunk GIN signal; every
// reducing CTA folds its share of chunk k as soon as chunk k's signal has
// counted all N-1 senders, so the reduce overlaps the rest of the copy. With
// at least 2N CTAs the owner stages its reduced chunks in the root's slot and
// SDMAs them to the root; with fewer, it stores them to the root from CUs.
// Each peer's partition is put by a single thread: the anvil SDMA queue is per
// peer, and concurrent producers on different XCDs can ring its doorbell out of
// order, leaving committed packets unprocessed (the put hangs in quiet).
// There are no barriers: each rank posts a per-launch ready flag to every peer
// and writes into a peer only once that peer's flag has arrived, and each
// owner posts a done flag to the root, which waits for all of them.
// Because the issuing CTA differs from the reducing CTA, a CTA's only guarantee
// that the previous Reduce finished with its slots is the kernel boundary, so
// the SDMA tier has no persistent device-timed variant.

// SDMA-tier layout, shared by GinSdmaReduceKernel and ReduceInitData. A message
// of `packs` 16-byte packs has nRanks-1 partitions of sdmaReducePartPacks packs
// each; the root reduces the remaining tail packs. Non-root rank `rank` owns
// partition sdmaReduceOwnedPart(rank, root, nRanks), and keeps sender s's slice
// of it in recvbuff slot sdmaReduceSlot(s, rank). Chunk k of a partition of
// `packs` packs starts at pack sdmaReduceChunkStart(packs, k, chunks).
__host__ __device__ inline size_t sdmaReducePartPacks(size_t packs, int nRanks) {
  return packs / (size_t)(nRanks - 1);
}
__host__ __device__ inline int sdmaReduceOwnedPart(int rank, int root, int nRanks) {
  return (rank - root - 1 + nRanks) % nRanks;
}
__host__ __device__ inline int sdmaReduceSlot(int sender, int owner) {
  return sender < owner ? sender : sender - 1;
}
__host__ __device__ inline size_t sdmaReduceChunkStart(size_t packs, int k, int chunks) {
  return (size_t)k * packs / chunks;
}

void ReduceGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  size_t aligned = count;
  // -D 3 splits the array into nranks slices of whole 128-bit packs with no
  // scalar tail, so round the count down to a multiple of nranks*16 bytes. The
  // SDMA tier's nranks-1 partitions take whole packs too and leave the rest to
  // the root (sdmaReducePartPacks). The host path keeps the exact requested
  // count.
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
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
    else if (deviceImpl == 3 && nranks > 1 && reduceUseSdma(args->expectedBytes, nranks, in_place)) {
      // The SDMA tier leaves each sender's slice of this rank's partition in its slot.
      const size_t pack = 16 / wordSize(type);
      const size_t part = sdmaReducePartPacks(recvcount / pack, nranks) * pack;
      const int owner = sdmaReduceOwnedPart(rank, root, nranks);
      // With issuer CTAs, the root's slot holds the reduced partition instead.
      const bool staged = deviceCtaCount >= 2 * nranks;
      for (int sender = 0; sender < nranks; sender++) {
        if (sender == rank) continue;
        const int slot = sdmaReduceSlot(sender, rank);
        void* slotPtr = (char*)args->expected[i] + (size_t)slot * part * wordSize(type);
        if (staged && sender == root) TESTCHECK(InitDataReduce(slotPtr, part, (size_t)owner * part, type, op, rep, nranks));
        else TESTCHECK(InitData(slotPtr, part, (size_t)owner * part, type, op, rep, nranks, sender));
      }
    }
#endif
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

  ReduceDevState& st = reduceState();
  st = {};
  switch(deviceImpl) {
    case 3: { // LL tier, tiled LSA read-reduce, and the SDMA tier
      reqs->lsaBarrierCount = deviceCtaCount;
      st.llMaxBytes = gin_sdma::resolveLLCap(
          gin_sdma::parseSize(getenv("NCCL_GIN_ANVIL_REDUCE_LL_MAX_BYTES")),
          kReduceLLDefaultBufferBytes / (4 * (size_t)commProperties.nRanks), kReduceLLMaxBytes);
      if (is_main_proc && is_main_thread)
        printf("# Reduce -D 3: LL tier for messages up to %zu bytes (NCCL_GIN_ANVIL_REDUCE_LL_MAX_BYTES)\n",
               st.llMaxBytes);
      if (st.llMaxBytes > 0) {
        // Each block receives every rank's share of its units, plus one ack slot.
        const size_t unitsPerCta = std::min<size_t>(st.llMaxBytes / 8, kReduceLLUnitsPerCta);
        const int nSlots = commProperties.nRanks * (int)unitsPerCta + 1;
        ncclLLA2ACreateRequirement(reduceLLCtas(st.llMaxBytes), nSlots, &st.llHandle, &g_reduceLLReq);
        g_reduceLLReq.next = reqs->resourceRequirementsList;
        reqs->resourceRequirementsList = &g_reduceLLReq;
      }

      // The SDMA tier needs the GIN Anvil SDMA backend: the owners tell the
      // root their staged chunks arrived after a flush, which the GIN API only
      // promises frees the source; Anvil's flush also waits for the copies to
      // land. That backend takes a GPU memory fault when one process drives two
      // GPUs. Without the tier, every message stays on the LSA tiers.
#ifdef NCCL_GIN_BACKEND_MASK_ALL
      const bool anvilBuilt = (NCCL_GIN_BACKEND_MASK_ALL >> (unsigned)NCCL_GIN_TYPE_ANVIL_SDMA) & 1u;
#else
      const bool anvilBuilt = true;
#endif
      const bool sdmaUsable = commProperties.nRanks >= 2 && anvilBuilt &&
                              commProperties.ginType == NCCL_GIN_TYPE_ANVIL_SDMA &&
                              nThreads * nGpus == 1;
      const size_t sdmaThreshold = reduceSdmaThreshold(deviceCtaCount, commProperties.nRanks);
      st.sdma = sdmaUsable && sdmaThreshold != gin_sdma::kThresholdUnset;
      if (is_main_proc && is_main_thread) {
        if (st.sdma)
          printf("# Reduce -D 3: SDMA tier for out-of-place messages from %zu bytes (NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCE)\n",
                 sdmaThreshold);
        else if (sdmaUsable)
          printf("# Reduce -D 3: SDMA tier off by default below %d CTAs (NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCE)\n",
                 2 * commProperties.nRanks);
        else
          printf("# Reduce -D 3: no SDMA tier (needs NCCL_GIN_ENABLE=1 NCCL_GIN_TYPE=%d and one GPU per process)\n",
                 (int)NCCL_GIN_TYPE_ANVIL_SDMA);
      }
      if (!st.sdma) return testSuccess;
      reqs->barrierCount = deviceCtaCount;
      // Two sets of per-chunk signals alternated by launch parity.
      reqs->ginSignalCount = 2 * deviceCtaCount;
      memset(&g_reduceFlagReq, 0, sizeof(g_reduceFlagReq));
      g_reduceFlagReq.bufferSize = sdmaReduceTailOffset(commProperties.nRanks) +
                                   (size_t)commProperties.nRanks * (commProperties.nRanks - 1) * 16;
      g_reduceFlagReq.bufferAlign = 16;
      g_reduceFlagReq.outBufferHandle = &st.flagHandle;
      g_reduceFlagReq.next = reqs->resourceRequirementsList;
      reqs->resourceRequirementsList = &g_reduceFlagReq;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,7)
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
      reqs->ginForceEnable = true;
#endif
      return testSuccess;
    }
    default:
      if (is_main_proc && is_main_thread)
        fprintf(stderr, "Reduce has no -D %d; -D 3 picks the LL, LSA or SDMA tier by message size.\n", deviceImpl);
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

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
// LL tier of -D 3: one hop, no barrier. CTA b owns a contiguous range of n
// 8-byte units. Every rank (the root included) stores its units into the root's
// LL slots [rank*n, rank*n+n) of block b, tagged with the block's epoch; the
// root polls them, folds the N contributions per unit in ascending rank order,
// and stores the result to its recvbuff. Only the root receives data, so
// nothing would stop a sender from running two epochs ahead and overwriting
// slots of the same parity the root has not read yet. After it has read every
// slot, the root therefore sends each sender an ack in the last slot, and a
// sender does not finish its epoch until the ack arrives. In-place safe: the
// thread that stores result unit j is the one that sent source unit j.
template <typename T>
__device__ __forceinline__ void ginReduceLLBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm const& devComm, int redOp, ncclLLA2AHandle llHandle) {
  static_assert(sizeof(T) <= 8 && 8 % sizeof(T) == 0, "LL units hold whole elements");
  constexpr int EPU = (int)(8 / sizeof(T));
  struct alignas(8) Unit { T e[EPU]; };
  const int nRanks = devComm.nRanks;
  const int rank = devComm.rank;
  const size_t nUnits = count * sizeof(T) / 8;
  const size_t u0 = nUnits * blockIdx.x / gridDim.x;
  const int n = (int)(nUnits * (blockIdx.x + 1) / gridDim.x - u0);
  const int ackSlot = (int)llHandle.nSlots - 1;

  ncclLLA2ASession<ncclCoopCta> ll { ncclCoopCta(), devComm, ncclTeamLsa(devComm), llHandle, blockIdx.x, (int)llHandle.nSlots };
  const uint64_t* src = (const uint64_t*)ncclGetLocalPointer(sendwin, sendoffset) + u0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) ll.send(root, rank * n + j, src[j]);

  if (rank == root) {
    uint64_t* dst = (uint64_t*)ncclGetLocalPointer(recvwin, recvoffset) + u0;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      T acc[EPU];
      for (int r0 = 0; r0 < nRanks; r0 += 8) {
        uint64_t v[8];
        const int c = min(8, nRanks - r0);
        ll.template recvUnrolled</*MinEltCount=*/1, /*MaxEltCount=*/8>(r0 * n + j, c, n, v);
        for (int r = 0; r < c; r++) {
          Unit x;
          __builtin_memcpy(&x, &v[r], 8);
          #pragma unroll
          for (int e = 0; e < EPU; e++) {
            const T y = gin_sdma_reduce::preOp(redOp, x.e[e], nRanks);
            acc[e] = (r0 + r == 0) ? y : gin_sdma_reduce::combine(redOp, acc[e], y);
          }
        }
      }
      Unit o;
      #pragma unroll
      for (int e = 0; e < EPU; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], nRanks);
      uint64_t out;
      __builtin_memcpy(&out, &o, 8);
      dst[j] = out;
    }
    __syncthreads();  // every slot of this epoch has been read
    for (int p = threadIdx.x; p < nRanks; p += blockDim.x)
      if (p != root) ll.send(p, ackSlot, (uint64_t)1);
  } else if (threadIdx.x == 0) {
    (void)ll.template recv<uint64_t>(ackSlot);
  }
  ll.endEpoch(ncclCoopCta());
}

template <typename K>
__global__ void GinReduceLLKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp, ncclLLA2AHandle llHandle) {
  ginReduceLLBody<typename gin_sdma_reduce::Elem<K>::type>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm, redOp, llHandle);
}

template <typename K>
__global__ void GinReduceLLTimedKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp, ncclLLA2AHandle llHandle, int loop, int skip, long long* start_time, long long* end_time) {
  for (int i = 0; i < skip + loop; i++) {
    if (i == skip) {
      __syncthreads();
      if (threadIdx.x == 0) start_time[blockIdx.x] = wall_clock64();
    }
    ginReduceLLBody<typename gin_sdma_reduce::Elem<K>::type>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm, redOp, llHandle);
  }
  __syncthreads();
  if (threadIdx.x == 0) end_time[blockIdx.x] = wall_clock64();
}
#endif

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


#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
// Per-chunk count of reducing CTAs that finished, and the count of CTAs that
// finished storing to the root, in two sets alternated by launch parity like
// the chunk signals. Device globals are per GPU, and each GPU runs at most one
// SDMA-tier Reduce at a time.
__device__ unsigned int gSdmaReduceChunkDone[2][kReduceFlagMaxCtas];
__device__ unsigned int gSdmaReduceRootDone[2];

__device__ __forceinline__ void postFlag(uint32_t* flag, uint32_t seq) {
  __hip_atomic_store(flag, seq, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void waitFlag(const uint32_t* flag, uint32_t seq) {
  while ((int32_t)(__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) - seq) < 0) {}
}

// `count` is the FULL element count, whole packs (ReduceGetCollByteCount).
template <typename T>
__device__ __forceinline__ void ginSdmaReduceBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp, int dmaChunks, ncclDevResourceHandle_t flagHandle) {
  const int nRanks = devComm.nRanks;
  const int me = devComm.rank;
  const int nOwners = nRanks - 1;
  constexpr int VEC = (sizeof(T) <= 16) ? (int)(16 / sizeof(T)) : 1;
  struct alignas(16) Pack { T e[VEC]; };
  const size_t partPacks = sdmaReducePartPacks(count / VEC, nRanks);
  const size_t partBytes = partPacks * sizeof(Pack);
  const size_t tail0 = (size_t)nOwners * partPacks;
  const size_t tailPacks = count / VEC - tail0;
  const int G = (int)gridDim.x;

  ncclGin gin { devComm, 0 };
  // Credits instead of barriers. Each rank posts `seq` to ready[me] on every
  // peer when it starts, meaning its previous Reduce and whatever its stream
  // ran since are done; a rank writes into peer q's memory only after ready[q]
  // reaches seq. Senders reuse their sendbuff once their own flush returns, so
  // only the root waits at the end, for done[] from every owner.
  uint32_t* flags = (uint32_t*)ncclGetResourceBufferLocalPointer(devComm, flagHandle);
  uint32_t* const ready = flags;
  uint32_t* const done = flags + nRanks;
  uint32_t* const ctaSeq = flags + 2 * nRanks;
  auto peerFlags = [&](int peer) {
    return (uint32_t*)ncclGetResourceBufferPeerPointer(devComm, flagHandle, ncclTeamLsa(devComm), peer);
  };
  const Pack* const sendPacks = (const Pack*)ncclGetLocalPointer(sendwin, sendoffset);
  // Every CTA of every rank counts the same launches, so seq agrees across
  // them. Launches alternate between two sets of G signals: this one counts
  // chunks in set `sigBase` and clears the other set, which the previous launch
  // has finished waiting on and which the next launch's senders cannot touch
  // before this rank posts its next ready.
  const uint32_t seq = ctaSeq[blockIdx.x] + 1;
  const unsigned int par = seq & 1u;
  const unsigned int sigBase = par * (unsigned int)G;
  // An owner's last act: store its tail packs into the root's tail scratch,
  // then tell the root its partition and tail are in place. Called by a thread
  // that has waited for the root's ready.
  auto postDone = [&]() {
    if (tailPacks) {
      Pack* scratch = (Pack*)((char*)peerFlags(root) + sdmaReduceTailOffset(nRanks)) + (size_t)me * nOwners;
      for (size_t t = 0; t < tailPacks; t++) scratch[t] = sendPacks[tail0 + t];
      cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
    }
    postFlag(peerFlags(root) + nRanks + me, seq);
  };
  if (threadIdx.x == 0) gin.resetSignal((unsigned int)G - sigBase + blockIdx.x);
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < kReduceFlagMaxCtas; j += G * blockDim.x) gSdmaReduceChunkDone[par ^ 1u][j] = 0;
  if (blockIdx.x == 0 && threadIdx.x == 0) gSdmaReduceRootDone[par ^ 1u] = 0;
  if (blockIdx.x == 0)
    for (int p = threadIdx.x; p < nRanks; p += blockDim.x)
      if (p != me) postFlag(peerFlags(p) + me, seq);

  // With at least 2N CTAs, CTAs [0, N) only issue puts, so no chunk waits on a
  // CTA that is still busy draining queues. With fewer, CTA b issues for peers
  // q = b (mod G) and then reduces alongside the rest. Each issuing thread
  // waits for its own peer's ready before its first put.
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
      const int owner = sdmaReduceOwnedPart(me, root, nRanks);
      const int rootSlot = sdmaReduceSlot(root, me);
      waitFlag(ready + root, seq);
      for (int k = 0; k < dmaChunks; k++) {
        const size_t lo = sdmaReduceChunkStart(partPacks, k, dmaChunks) * sizeof(Pack);
        const size_t hi = sdmaReduceChunkStart(partPacks, k + 1, dmaChunks) * sizeof(Pack);
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
        const int owner = sdmaReduceOwnedPart(q, root, nRanks);
        const int slot = sdmaReduceSlot(me, q);
        waitFlag(ready + q, seq);
        // The signal quiets the queue first, so put k+1 is only queued once
        // put k has landed and chunk k has been released to the owner.
        for (int k = 0; k < dmaChunks; k++) {
          const size_t lo = sdmaReduceChunkStart(partPacks, k, dmaChunks) * sizeof(Pack);
          const size_t hi = sdmaReduceChunkStart(partPacks, k + 1, dmaChunks) * sizeof(Pack);
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
    const int owner = sdmaReduceOwnedPart(me, root, nRanks);
    const char* slots = (const char*)ncclGetLocalPointer(recvwin, recvoffset);
    const Pack* own = (const Pack*)((const char*)ncclGetLocalPointer(sendwin, sendoffset) + (size_t)owner * partBytes);
    // Each pack of the root's slot is read and then overwritten by the same
    // thread, so staging the result there is safe.
    Pack* dst = nIssue ? (Pack*)(slots + (size_t)sdmaReduceSlot(root, me) * partBytes)
                       : (Pack*)((char*)ncclGetLsaPointer(recvwin, recvoffset, root) + (size_t)owner * partBytes);
    if (!nIssue) {
      if (threadIdx.x == 0) waitFlag(ready + root, seq);
      __syncthreads();
    }
    const size_t first = (size_t)((int)blockIdx.x - nIssue) * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)nReducers * blockDim.x;
    constexpr int NR8 = 8;
    const Pack* src8[NR8];
    if (nRanks <= NR8) {
      for (int r = 0; r < nRanks; r++)
        src8[r] = (r == me) ? own : (const Pack*)(slots + (size_t)sdmaReduceSlot(r, me) * partBytes);
    }
    // Every reducing CTA takes a share of every chunk, so the chunk that lands
    // last is reduced by all of them rather than by a few.
    for (int k = 0; k < dmaChunks; k++) {
      const size_t p0 = sdmaReduceChunkStart(partPacks, k, dmaChunks);
      const size_t p1 = sdmaReduceChunkStart(partPacks, k + 1, dmaChunks);
      if (p1 == p0) continue;
      gin.waitSignal(ncclCoopCta(), sigBase + (unsigned int)k, (uint64_t)nOwners);
      // The slots were written by a remote SDMA engine, not by a CU, so they
      // need a system-scope acquire before any thread loads them. Every wave
      // that runs the fence invalidates the whole L2, so only thread 0 runs it
      // and the barrier extends it to the rest of the CTA.
      if (threadIdx.x == 0) cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
      __syncthreads();
      if (nRanks >= 2 && nRanks <= NR8) {
        switch (nRanks) {
          case 2: sdmaFoldPacks<T, 2, 4>(src8, dst, p0 + first, p1, stride, redOp); break;
          case 3: sdmaFoldPacks<T, 3, 4>(src8, dst, p0 + first, p1, stride, redOp); break;
          case 4: sdmaFoldPacks<T, 4, 4>(src8, dst, p0 + first, p1, stride, redOp); break;
          case 5: sdmaFoldPacks<T, 5, 2>(src8, dst, p0 + first, p1, stride, redOp); break;
          case 6: sdmaFoldPacks<T, 6, 2>(src8, dst, p0 + first, p1, stride, redOp); break;
          case 7: sdmaFoldPacks<T, 7, 2>(src8, dst, p0 + first, p1, stride, redOp); break;
          default: sdmaFoldPacks<T, NR8, 2>(src8, dst, p0 + first, p1, stride, redOp); break;
        }
      } else {
        for (size_t pk = p0 + first; pk < p1; pk += stride) {
          T acc[VEC];
          for (int r = 0; r < nRanks; r++) {
            const Pack v = (r == me) ? own[pk] : ((const Pack*)(slots + (size_t)sdmaReduceSlot(r, me) * partBytes))[pk];
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
        // The SDMA engine reads the staged chunk from memory, not through L2,
        // so the CTA's stores need a system-scope release (an L2 writeback).
        // As above, one wave runs it after the barrier.
        waitWaveStores();
        __syncthreads();
        if (threadIdx.x == 0) {
          cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
          __hip_atomic_fetch_add(&gSdmaReduceChunkDone[par][k], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    }
    if (!nIssue) {
      // CU stores to the root: the last CTA to finish tells the root.
      waitWaveStores();
      __syncthreads();
      if (threadIdx.x == 0) {
        cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
        if (__hip_atomic_fetch_add(&gSdmaReduceRootDone[par], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == (unsigned int)nReducers - 1)
          postDone();
      }
    }
  }
  // Flush drains this rank's SDMA queues, so this rank's puts have landed and
  // its sendbuff is free once it returns. With issuer CTAs, each owner-to-owner
  // chunk ends in a strong signal, which has already settled every put to that
  // peer, so only CTA `root` flushes, for its signal-free owner-to-root queue.
  // It must be the only flusher: Anvil keeps one dirty bit per queue, and a
  // concurrent flush can clear the bit of a put it did not drain.
  // Written as a select: the || form of the same test makes amdclang unswitch
  // the fold loops differently, which costs about 4% at 1 GiB on MI450.
  const bool flusher = nIssue ? ((int)blockIdx.x == root && me != root) : true;
  if (flusher) gin.flush(ncclCoopCta());
  // CTA `root` fed the owner-to-root queue and drained it with that flush, so
  // the staged chunks are in the root's recvbuff.
  if (nIssue && (int)blockIdx.x == root && me != root && threadIdx.x == 0)
    postDone();
  if (me == root && blockIdx.x == 0) {
    for (int p = threadIdx.x; p < nRanks; p += blockDim.x)
      if (p != root) waitFlag(done + p, seq);
    __syncthreads();
    if (threadIdx.x == 0) cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
    if (tailPacks) {
      // Fold the tail in ascending rank order, like the partitions.
      __syncthreads();
      const Pack* scratch = (const Pack*)((const char*)flags + sdmaReduceTailOffset(nRanks));
      Pack* out = (Pack*)ncclGetLocalPointer(recvwin, recvoffset);
      for (size_t t = threadIdx.x; t < tailPacks; t += blockDim.x) {
        T acc[VEC];
        for (int r = 0; r < nRanks; r++) {
          const Pack v = (r == root) ? sendPacks[tail0 + t] : scratch[(size_t)r * nOwners + t];
          #pragma unroll
          for (int e = 0; e < VEC; e++) {
            const T x = gin_sdma_reduce::preOp(redOp, v.e[e], nRanks);
            acc[e] = (r == 0) ? x : gin_sdma_reduce::combine(redOp, acc[e], x);
          }
        }
        Pack o;
        #pragma unroll
        for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp(redOp, acc[e], nRanks);
        out[tail0 + t] = o;
      }
    }
  }
  if (threadIdx.x == 0) ctaSeq[blockIdx.x] = seq;
}

template <typename K>
__global__ void GinSdmaReduceKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm, int redOp, int dmaChunks, ncclDevResourceHandle_t flagHandle) {
  ginSdmaReduceBody<typename gin_sdma_reduce::Elem<K>::type>(sendwin, sendoffset, recvwin, recvoffset, count, root, devComm, redOp, dmaChunks, flagHandle);
}

// SDMA puts per partition for the SDMA tier. More chunks overlap the copies with the
// reduce and the owner-to-root copy, but each chunk adds a fixed 30-60 us (put
// issue, queue drain, signal wait), so the default grows with sqrt(partBytes):
// about sqrt(partBytes/2MiB), capped at 16. Partitions are only split when
// there are dedicated issuer CTAs (ctas >= 2*nRanks).
// NCCL_GIN_ANVIL_SDMA_REDUCE_CHUNKS overrides the default.
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
    const size_t units = partBytes >> 21;
    while (k < 16 && (size_t)(k + 1) * (size_t)(k + 1) <= units) k++;
  }
  return (int)std::max(1L, std::min(k, (long)ctas));
}
#endif

// Like testLaunchDeviceKernel() but forwards the reduction op (and any extra
// kernel arguments), which the shared launcher's fixed signature does not carry.
template <typename F, typename... Extra>
static testResult_t ReduceLaunchDeviceKernel(F kernel, int grid, void* sendbuff, size_t sendoffset,
                                             void* recvbuff, size_t recvoffset, size_t count,
                                             ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream,
                                             Extra... extra) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  kernel<<<grid, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, (int)op, extra...);
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
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
        const int nRanks = devComm->nRanks;
        const size_t bytes = count * wordSize(type);
        const bool inPlace = sendbuff == recvbuff && sendoffset == recvoffset;
        if (reduceUseSdma(bytes, nRanks, inPlace)) {
          const size_t partBytes = sdmaReducePartPacks(bytes / 16, nRanks) * 16;
          reduceReportTier(deviceCtaCount >= 2 * nRanks ? ReduceLaunchTier::SdmaStaged : ReduceLaunchTier::SdmaCu);
          TESTCHECK(ReduceLaunchDeviceKernel(SPECIALIZE_REDUCE_KERNEL(GinSdmaReduceKernel, type, op), deviceCtaCount, sendbuff, sendoffset, recvbuff, recvoffset, count, op, root, comm, stream,
                                             sdmaReduceDmaChunks(partBytes, deviceCtaCount, nRanks), reduceState().flagHandle));
          return testSuccess;
        }
        if (reduceUseLL(bytes, nRanks)) {
          reduceReportTier(ReduceLaunchTier::LL);
          TESTCHECK(ReduceLaunchDeviceKernel(SPECIALIZE_REDUCE_KERNEL(GinReduceLLKernel, type, op), reduceLLCtas(bytes), sendbuff, sendoffset, recvbuff, recvoffset, count, op, root, comm, stream,
                                             reduceState().llHandle));
          return testSuccess;
        }
#endif
        reduceReportTier(ReduceLaunchTier::Tiled);
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
// overrides, keyed off the per-rank slice size. The SDMA tier has no timed
// kernel, so sizes that take it are not device-timed.
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

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
  if (reduceUseSdma(totalBytes, nRanksGlobal, in_place)) return testSuccess;
  const bool useLL = reduceUseLL(totalBytes, nRanksGlobal);
  auto llKernel = SPECIALIZE_REDUCE_KERNEL(GinReduceLLTimedKernel, type, op);
  const int gridCtas = useLL ? reduceLLCtas(totalBytes) : deviceCtaCount;
#else
  const int gridCtas = deviceCtaCount;
#endif
  double devUs = 0.0;
  TESTCHECK(gin_devtime::measure(args, gridCtas, loop,
      [&](int i, long long* d_start, long long* d_end) {
        ncclDevComm* devComm = args->devComms + i;
        ncclWindow_t sendwin = (ncclWindow_t)(in_place ? args->recvRegHandles[i] : args->sendRegHandles[i]);
        ncclWindow_t recvwin = (ncclWindow_t)args->recvRegHandles[i];
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
        if (useLL) {
          llKernel<<<gridCtas, 512, 0, args->streams[i]>>>(sendwin, 0, recvwin, 0, count, root, *devComm,
                   (int)op, reduceState().llHandle, loop, skip, d_start, d_end);
          return;
        }
#endif
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
