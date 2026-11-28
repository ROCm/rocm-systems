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
#include "nccl_device/rccl_ptr.h"
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

// -D 3 is GinHybridGatherKernel: LL (tiny), LSA direct (small) and one GIN put to root
// (large), on GinHybridAllGatherKernel's pools raised to the Gather ladder's peak
// (GatherPoolCtas) plus two LL resources: the LL tier's scratch (one block) and one
// done flag per rank for each LSA-tier CTA.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0) || (defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7))
// The LSA-tier CTA ladder of this device's architecture, looked up once per process.
static const gin_sdma_allgather::ArchLsaCtaLadder* GatherLsaCtaLadder() {
  static const gin_sdma_allgather::ArchLsaCtaLadder* ladder =
      gin_sdma_allgather::gatherLsaCtaLadder(ginSdmaGcnArchName());
  return ladder;
}

static int GatherPoolCtas() {
  return gin_sdma_allgather::gatherPoolCtas(deviceCtaCount, GatherLsaCtaLadder());
}

static void GatherRaisePools(ncclDevCommRequirements* reqs, bool gin) {
  const int pool = GatherPoolCtas();
  if (reqs->lsaBarrierCount < pool) reqs->lsaBarrierCount = pool;
  if (!gin) return;
  if (reqs->barrierCount < pool) reqs->barrierCount = pool;
  if (reqs->ginSignalCount < pool) reqs->ginSignalCount = pool;
}
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
// thread_local for -t N, like Broadcast's LL requirement: each host thread links its own
// nodes into its own requirements list.
thread_local size_t g_gatherLLMaxBytes = 0;
thread_local ncclLLA2AHandle g_gatherLLHandle = {};
thread_local ncclDevResourceRequirements g_gatherLLReq = {};
thread_local ncclLLA2AHandle g_gatherFlagHandle = {};
thread_local ncclDevResourceRequirements g_gatherFlagReq = {};
// False when the communicator has no GIN, e.g. ANVIL SDMA on a scale-up domain that
// spans hosts: the LL and LSA tiers then cover every size.
thread_local bool g_gatherGin = true;
thread_local size_t g_gatherGroupMinBytes = gin_sdma_allgather::kGatherGroupNever;

testResult_t GatherGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclComm_t comm) {
  testResult_t res = ginSdmaAllGatherDevCommRequirements(deviceImpl, reqs, comm, "GinHybridGatherKernel", &g_gatherGin);
  if (res != testSuccess) return res;
  GatherRaisePools(reqs, g_gatherGin);
  int nRanks = 0;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  g_gatherLLHandle = {};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  g_gatherLLMaxBytes = 0;
#else
  g_gatherLLMaxBytes = gin_sdma_allgather::resolveGatherLLMaxBytesFromEnv(
      gin_sdma_allgather::gatherArchLLMaxBytesForRanks(ginSdmaGcnArchName(), nRanks));
  g_gatherGroupMinBytes = gin_sdma_allgather::gatherArchGroupMinBytes(ginSdmaGcnArchName());
#endif
  if (g_gatherLLMaxBytes > 0) {
    NCCLCHECK(ncclLLA2ACreateRequirement(/*nBlocks=*/1, (int)gin_sdma_allgather::gatherLLSlots(nRanks, g_gatherLLMaxBytes),
                                         &g_gatherLLHandle, &g_gatherLLReq));
    g_gatherLLReq.next = reqs->resourceRequirementsList;
    reqs->resourceRequirementsList = &g_gatherLLReq;
  }
  NCCLCHECK(ncclLLA2ACreateRequirement(GatherPoolCtas(), nRanks, &g_gatherFlagHandle, &g_gatherFlagReq));
  g_gatherFlagReq.next = reqs->resourceRequirementsList;
  reqs->resourceRequirementsList = &g_gatherFlagReq;
  return testSuccess;
}
#elif defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
bool GatherGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!ginSdmaAllGatherDevCommRequirements(deviceImpl, reqs)) return false;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
  GatherRaisePools(reqs, /*gin=*/true);
#endif
  return true;
}
#endif

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
// -D 3 launch settings for ginSdmaResolve*() (gin_sdma_allgather_reqs.h). The crossover is
// NCCL_GIN_ANVIL_SDMA_THRESHOLD_GATHER > NCCL_GIN_ANVIL_SDMA_THRESHOLD > the per-arch default.
// One CTA issues the root put; the LSA tier uses the per-arch ladder, or the AllGather
// 16-CTA grid without one. NCCL_GIN_ANVIL_SDMA_GATHER_CTAS pins both tiers.
struct GatherSdmaColl {
  static constexpr const char* kThresholdEnv = gin_sdma_allgather::kGatherSdmaThresholdEnv;
  static constexpr size_t kDefaultThreshold = gin_sdma_allgather::kGatherSdmaThresholdDefault;
  static size_t hostThresholdDefault() {
    return gin_sdma_allgather::gatherArchSdmaThreshold(ginSdmaGcnArchName());
  }
  static constexpr const char* kCtasEnv = gin_sdma_allgather::kGatherCtasEnv;
  static constexpr const char* kCtasAlias = nullptr;
  static constexpr int kSdmaCtas = 1;
  static int lsaCtas(size_t chunkBytes) {
    return gin_sdma_allgather::gatherLsaCtas(GatherLsaCtaLadder(), chunkBytes);
  }
  static int poolCtas() { return GatherPoolCtas(); }
};

static size_t GatherSdmaThreshold() {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
  if (!g_gatherGin) return gin_sdma_allgather::kGatherSdmaNever;
#endif
  return ginSdmaResolveSdmaThreshold<GatherSdmaColl>();
}

// Per-launch -D 3 options resolved on the host. A zero nSlots handle turns its feature off.
struct GatherD3Opts {
  ncclLLA2AHandle ll;     // LL tier scratch
  size_t llMaxBytes;      // LL tier cap per rank
  ncclLLA2AHandle flags;  // ready and done flags; without them the LL and LSA tiers use LSA barriers
  int exitBarrier;        // LSA barriers instead of flags, plus the SDMA tier's exit barrier
  int streamStores;       // system-scope 16-byte stores to the root
  int lightFence;         // publish streamed stores by waiting for them, not with a release fence
  int groupRanks;         // LSA tier: root flags senders ready this many ranks at a time; 0 = all at once
  int groupWindow;        // LSA tier: groups storing into the root at a time (NCCL_GIN_ANVIL_SDMA_GATHER_GROUP_WINDOW)
};

// 16-byte store into the root's recv buffer.
__device__ __forceinline__ void GatherStore16(uint4* dst, uint4 v, int streamStores) {
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  if (streamStores) {
    union { uint4 u; v4u v; } cv;
    cv.u = v;
    __builtin_amdgcn_global_store_b128((v4u_gptr)dst, cv.v, RCCL_SYSTEM_SYNCSCOPE);
    return;
  }
#endif
  *dst = v;
}

// Grid-stride 16-byte copy of n elements. Issuing all kUnroll loads before their stores
// keeps several remote stores in flight per thread. The tail stays in the batched loop:
// a plain one-load remainder loop is as slow as no unrolling.
template <int kUnroll>
__device__ __forceinline__ void GatherCopy16(const uint4* __restrict__ s, uint4* __restrict__ d, size_t n,
                                             size_t tid, size_t stride, int streamStores) {
  for (size_t i = tid; i < n; i += kUnroll * stride) {
    uint4 v[kUnroll];
#pragma unroll
    for (int k = 0; k < kUnroll; k++) {
      if (i + k * stride < n) v[k] = s[i + k * stride];
    }
#pragma unroll
    for (int k = 0; k < kUnroll; k++) {
      if (i + k * stride < n) GatherStore16(&d[i + k * stride], v[k], streamStores);
    }
  }
}

// Each rank stores its chunk into root's recv slice. `root` is a world rank;
// ncclGetPeerPointer maps it into the LSA window. Non-root recv buffers stay untouched.
// Gather only moves bytes, so the copy is 16 bytes per thread whatever the element
// type; a chunk or window offset that is not 16-byte aligned falls back to a byte copy.
// Returns false if it used the byte copy, whose plain stores need the full release
// fence before a done flag.
__device__ bool GatherLsaDirect(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t chunkBytes, int rank, int root, int tid, int nthreads, int streamStores) {
  const char* src = (const char*)ncclGetLocalPointer(sendwin, sendoffset);
  const size_t dstOff = gin_sdma_allgather::allGatherRecvSliceOffset(rank, chunkBytes);
  char* dst = (char*)ncclGetPeerPointer(recvwin, recvoffset, root) + dstOff;
  if (src == dst) return true;
  if ((((uintptr_t)src | (uintptr_t)dst | chunkBytes) & 15) == 0) {
    // A slice that starts mid cache line (chunkBytes not a multiple of 128) makes every
    // wavefront's stores straddle lines, which costs 1.6x on 4 MI455X nodes. Peel up to
    // 112 bytes so the grid-stride copy writes whole lines; src stays 16-byte aligned.
    const size_t head = min(chunkBytes, (size_t)(-(uintptr_t)dst & 127));
    if ((size_t)tid < head / 16) GatherStore16(&((uint4*)dst)[tid], ((const uint4*)src)[tid], streamStores);
    // src and dst are distinct buffers once src == dst has returned, so __restrict__ holds.
    const uint4* s = (const uint4*)(src + head);
    uint4* d = (uint4*)(dst + head);
    const size_t n = (chunkBytes - head) / 16;
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    // On 4x MI455X with the 96-CTA grid, 8-deep batches are fastest up to 4 MiB per
    // rank and 4-deep ones above (about 2% either way).
    if (chunkBytes <= (4u << 20)) GatherCopy16<8>(s, d, n, tid, nthreads, streamStores);
    else GatherCopy16<4>(s, d, n, tid, nthreads, streamStores);
#else
    // On gfx942 and gfx950 the copy is xGMI-bound and unrolling is 5 to 8% slower.
    GatherCopy16<1>(s, d, n, tid, nthreads, streamStores);
#endif
    return true;
  }
  for (size_t i = tid; i < chunkBytes; i += nthreads) dst[i] = src[i];
  return false;
}

// Ready flags on the done-flag session: a sender may write into the root (its recv
// buffer or LL scratch) once the root has entered this launch, so the root's CTA flags
// its same-index CTA on each sender as ready instead of all ranks meeting in an LSA
// barrier. The flag also keeps a sender within one epoch of the root.
__device__ void GatherRootReady(ncclLLA2ASession<ncclCoopCta>& done, const struct ncclDevComm& devComm,
                                ncclTeam lsa, int root, int first, int last) {
  for (int p = first + threadIdx.x; p < last; p += blockDim.x) {
    if (p != root) done.send(lsa.rank + (p - devComm.rank), root, (uint64_t)1);
  }
}

__device__ void GatherSenderWaitReady(ncclLLA2ASession<ncclCoopCta>& done, int root) {
  if (threadIdx.x == 0) (void)done.recv<uint64_t>(root);
  __syncthreads();
}

// LL tier, one CTA: each non-root sends its chunk as 8-byte LL elements (data and epoch
// tag in one 16-byte store) into the root's scratch, and the root copies them into its
// recv buffer, so no rank writes another rank's recv buffer. Only non-roots write, so,
// as in Broadcast's LL tier, a writer must not run two epochs ahead of the root and
// overwrite scratch the root has not read yet. The root's ready flag ensures that,
// or an entry barrier without done flags.
__device__ void GatherLL(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset,
                         size_t chunkBytes, int root, const struct ncclDevComm& devComm, ncclLLA2AHandle llHandle,
                         ncclLLA2AHandle flags) {
  ncclTeam lsa = ncclTeamLsa(devComm);
  const int rank = devComm.rank;
  const int nRanks = devComm.nRanks;
  const size_t chunkU64 = chunkBytes / 8;

  ncclLLA2ASession<ncclCoopCta> ll { ncclCoopCta(), devComm, lsa, llHandle, /*block=*/0,
                                     /*maxElts=*/(int)((size_t)nRanks * chunkU64) };
  if (flags.nSlots == 0) {
    ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, /*block=*/0 };
    lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
  } else {
    ncclLLA2ASession<ncclCoopCta> ready { ncclCoopCta(), devComm, lsa, flags, /*block=*/0, nRanks };
    if (rank == root) GatherRootReady(ready, devComm, lsa, root, 0, nRanks);
    else GatherSenderWaitReady(ready, root);
    ready.endEpoch(ncclCoopCta());
  }

  const uint64_t* src = (const uint64_t*)ncclGetLocalPointer(sendwin, sendoffset);
  if (rank != root) {
    const int rootLsa = lsa.rank + (root - rank);
    const size_t base = (size_t)rank * chunkU64;
    for (size_t j = threadIdx.x; j < chunkU64; j += blockDim.x) ll.send(rootLsa, (int)(base + j), src[j]);
  } else {
    uint64_t* dst = (uint64_t*)ncclGetLocalPointer(recvwin, recvoffset);
    uint64_t* own = dst + (size_t)rank * chunkU64;
    if (own != src) {
      for (size_t j = threadIdx.x; j < chunkU64; j += blockDim.x) own[j] = src[j];
    }
    const size_t peerElts = (size_t)(nRanks - 1) * chunkU64;
    for (size_t e = threadIdx.x; e < peerElts; e += blockDim.x) {
      const size_t p = e / chunkU64;
      const size_t slot = (p < (size_t)rank ? e : e + chunkU64);
      dst[slot] = ll.recv<uint64_t>((int)slot);
    }
  }
  ll.endEpoch(ncclCoopCta());
}

// Makes every store this CTA issued to the root visible before a flag that follows.
// The device-scope release fence writes back L2 in every CTA, which makes extra CTAs
// expensive. With `light`, each wave only waits for its own memory operations, the
// cheap post-send fence RCCL's Simple protocol uses (prims_simple.h). That is enough
// only when every store was a system-scope streaming store, which writes through to
// the root; GatherLsaDirect reports whether that held.
__device__ __forceinline__ void GatherPublishStores(bool light) {
  __syncthreads();
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  if (light) {
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    asm volatile("s_wait_loadcnt 0x0\n\ts_wait_storecnt 0x0");
#else
    asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
#endif
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
  } else
#endif
    cuda::atomic_thread_fence(cuda::memory_order_release);
  __syncthreads();
}

__device__ __forceinline__ bool GatherLightFence(const GatherD3Opts& opts) {
  return RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS && opts.lightFence && opts.streamStores;
}

// LSA-tier done flags: the sender answers the root's ready flag (above) with a done flag
// once its stores are visible.
__device__ void GatherRootWaitDone(ncclLLA2ASession<ncclCoopCta>& done, int root, int first, int last) {
  for (int p = first + threadIdx.x; p < last; p += blockDim.x) {
    if (p != root) (void)done.recv<uint64_t>(p);
  }
}

__device__ void GatherSenderDone(ncclLLA2ASession<ncclCoopCta>& done, const struct ncclDevComm& devComm,
                                 ncclTeam lsa, int root, bool light) {
  GatherPublishStores(light);
  if (threadIdx.x == 0) done.send(lsa.rank + (root - devComm.rank), devComm.rank, (uint64_t)1);
}

// Hybrid Gather body (-D 3) for a world that one LSA team covers (one node, or a
// scale-up domain spanning nodes), the AllGather body with one destination:
//   chunkBytes <= sdmaThreshold: direct LSA store into root (all CTAs).
//   chunkBytes >  sdmaThreshold: one GIN put to root from the thread the AllGather
//     peer stripe assigns to root. Only CTA (root % gridDim.x) on root waits, for
//     one signal per put.
// Only the root needs to know the data landed, and the next launch cannot start writing
// before the root has finished this one, so without opts.exitBarrier neither tier ends
// with a barrier. LSA-tier CTAs instead wait for the root's ready flag, store, and send
// the root's same-index CTA a done flag after publishing their stores.
__device__ void ginGatherBody(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t chunkBytes, int root, size_t sdmaThreshold, const struct ncclDevComm& devComm, const GatherD3Opts& opts) {
  if (gin_sdma_allgather::chunkUsesLsaTier(chunkBytes, sdmaThreshold)) {
    ncclTeam lsa = ncclTeamLsa(devComm);
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int nthreads = blockDim.x * gridDim.x;

    if (opts.exitBarrier || opts.flags.nSlots == 0) {
      ncclLsaBarrierSession<ncclCoopCta> lsaBar { ncclCoopCta(), devComm, lsa, devComm.lsaBarrier, blockIdx.x };
      lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
      GatherLsaDirect(sendwin, sendoffset, recvwin, recvoffset, chunkBytes, devComm.rank, root, tid, nthreads,
                      opts.streamStores);
      lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
      return;
    }
    ncclLLA2ASession<ncclCoopCta> done { ncclCoopCta(), devComm, lsa, opts.flags, blockIdx.x, devComm.nRanks };
    if (devComm.rank != root) {
      GatherSenderWaitReady(done, root);
      const bool streamed = GatherLsaDirect(sendwin, sendoffset, recvwin, recvoffset, chunkBytes, devComm.rank,
                                            root, tid, nthreads, opts.streamStores);
      GatherSenderDone(done, devComm, lsa, root, GatherLightFence(opts) && streamed);
    } else {
      // Ranks [g * group, (g + 1) * group) form group g; at most `window` groups store
      // into the root at a time.
      const int nRanks = devComm.nRanks;
      const int group = opts.groupRanks > 0 && opts.groupRanks < nRanks ? opts.groupRanks : nRanks;
      const int nGroups = (nRanks + group - 1) / group;
      const int window = opts.groupWindow > 1 ? opts.groupWindow : 1;
      for (int g = 0; g < window && g < nGroups; g++) {
        GatherRootReady(done, devComm, lsa, root, g * group, min(nRanks, (g + 1) * group));
      }
      GatherLsaDirect(sendwin, sendoffset, recvwin, recvoffset, chunkBytes, root, root, tid, nthreads,
                      opts.streamStores);
      for (int g = 0; g < nGroups; g++) {
        GatherRootWaitDone(done, root, g * group, min(nRanks, (g + 1) * group));
        if (g + window < nGroups) {
          __syncthreads();
          GatherRootReady(done, devComm, lsa, root, (g + window) * group, min(nRanks, (g + window + 1) * group));
        }
      }
    }
    done.endEpoch(ncclCoopCta());
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

  if (opts.exitBarrier) bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

__global__ void GinHybridGatherKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t chunkBytes, int root, size_t sdmaThreshold, struct ncclDevComm devComm, GatherD3Opts opts) {
  if (gin_sdma_allgather::gatherUsesLLTier(chunkBytes, opts.llMaxBytes, opts.ll.nSlots, devComm.nRanks)) {
    if (blockIdx.x == 0) {
      GatherLL(sendwin, sendoffset, recvwin, recvoffset, chunkBytes, root, devComm, opts.ll,
               opts.exitBarrier ? ncclLLA2AHandle{} : opts.flags);
    }
    return;
  }
  ginGatherBody(sendwin, sendoffset, recvwin, recvoffset, chunkBytes, root, sdmaThreshold, devComm, opts);
}

// Host-resolved -D 3 options; the env knobs are read once per process.
static GatherD3Opts GatherResolveOpts() {
  static const int exitBarrier = [] {
    const char* e = getenv("NCCL_GIN_ANVIL_SDMA_GATHER_EXIT_BARRIER");
    return (e && e[0] == '1') ? 1 : 0;
  }();
  static const int streamStores = [] {
    const char* e = getenv("NCCL_GIN_ANVIL_SDMA_GATHER_STREAM");
    return (e && e[0] == '0') ? 0 : 1;
  }();
  static const int lightFence = [] {
    const char* e = getenv("NCCL_GIN_ANVIL_SDMA_GATHER_LIGHT_FENCE");
    return (e && e[0] == '0') ? 0 : 1;
  }();
  GatherD3Opts opts = {};
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
  opts.ll = g_gatherLLHandle;
  opts.llMaxBytes = g_gatherLLMaxBytes;
  opts.flags = g_gatherFlagHandle;
#endif
  opts.exitBarrier = exitBarrier;
  opts.streamStores = streamStores;
  opts.lightFence = lightFence;
  static const int groupWindow = [] {
    const char* e = getenv("NCCL_GIN_ANVIL_SDMA_GATHER_GROUP_WINDOW");
    return e ? atoi(e) : 1;
  }();
  opts.groupWindow = groupWindow;
  return opts;
}

// LSA-tier ranks per group for one launch (gin_sdma_allgather::gatherGroupRanks).
static int GatherGroupRanks(size_t chunkBytes, int nRanks) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
  static const int envRanks = [] {
    const char* e = getenv(gin_sdma_allgather::kGatherGroupRanksEnv);
    return e ? atoi(e) : -1;
  }();
  return gin_sdma_allgather::gatherGroupRanks(chunkBytes, nRanks, g_gatherGroupMinBytes, envRanks);
#else
  return 0;
#endif
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
        const ncclDevComm* devComm = (const ncclDevComm*)comm;
        GatherD3Opts opts = GatherResolveOpts();
        const size_t sdmaThreshold = GatherSdmaThreshold();
        const size_t chunkBytes = gin_sdma_allgather::chunkBytes(count, (size_t)wordSize(type));
        opts.groupRanks = GatherGroupRanks(chunkBytes, devComm->nRanks);
        const int gridCtas =
            gin_sdma_allgather::gatherUsesLLTier(chunkBytes, opts.llMaxBytes, opts.ll.nSlots, devComm->nRanks)
                ? 1
                : ginSdmaResolveLaunchCtas<GatherSdmaColl>(chunkBytes, sdmaThreshold);
        GinHybridGatherKernel<<<gridCtas, 512, 0, stream>>>((ncclWindow_t)sendbuff, sendoffset, (ncclWindow_t)recvbuff,
                                                           recvoffset, chunkBytes, root, sdmaThreshold, *devComm, opts);
        return testSuccess;
      }
#endif
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

// Tier report for -D 3: LL into the root's scratch for tiny chunks, then direct stores
// into root over LSA up to the crossover and one GIN put above it. `count` is the
// per-rank element count.
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
testResult_t GatherGetDeviceImplInfo(int /*deviceImpl*/, size_t count, ncclDataType_t type,
    int nRanks, int /*lsaSize*/, char* algoOut, char* protoOut) {
  snprintf(algoOut, 16, "%s", "T-DIRECT");
  const GatherD3Opts opts = GatherResolveOpts();
  const size_t sdmaThreshold = GatherSdmaThreshold();
  const size_t chunkBytes = gin_sdma_allgather::chunkBytes(count, (size_t)wordSize(type));
  const char* tier = gin_sdma_allgather::gatherUsesLLTier(chunkBytes, opts.llMaxBytes, opts.ll.nSlots, nRanks) ? "LL"
                     : gin_sdma_allgather::chunkUsesLsaTier(chunkBytes, sdmaThreshold)                       ? "LSA"
                                                                                                             : "GIN";
  snprintf(protoOut, 16, "%s", tier);
  return testSuccess;
}
#else
testResult_t GatherGetDeviceImplInfo(int, size_t, ncclDataType_t, int, int, char* algoOut, char* protoOut) {
  snprintf(algoOut, 16, "%s", "T-DIRECT");
  snprintf(protoOut, 16, "%s", "GIN");
  return testSuccess;
}
#endif

struct testColl gatherTest = {
  "Gather",
  GatherGetCollByteCount,
  GatherInitData,
  GatherGetBw,
  GatherRunColl,
  NULL,
  NULL,
  NULL,
  NULL,
  GatherGetDeviceImplInfo
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
