/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * Modification Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "sym_kernels.h"
#include "archinfo.h"
#include "alloc.h"
#include "comm.h"
#include "checks.h"
#include "device.h"
#include "nccl_device/core_tmp.h"
#include "transport.h"
#include "tuning.h"
#if defined(__HIP_PLATFORM_AMD__)
#include "tdm/tdmCopy.h" // TDM_TOOLCHAIN_AVAILABLE, for ncclSymkTmaAvailable()
#endif
#include <atomic>
#include <cmath>
#include <cfloat>

constexpr ncclSymkKernelMask kernelMask_STMC =
  1ull << ncclSymkKernelId_AllGather_LLMC | 1ull << ncclSymkKernelId_AllGather_STMC |
  1ull << ncclSymkKernelId_AllGather_TmaSTMC | 1ull << ncclSymkKernelId_AllReduce_AGxLLMC_R |
  1ull << ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC | 1ull << ncclSymkKernelId_ReduceScatter_LDMC |
  1ull << ncclSymkKernelId_AllGather_RailRing_LsaSTMC;

constexpr ncclSymkKernelMask kernelMask_LDMC = 1ull << ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                               1ull << ncclSymkKernelId_ReduceScatter_LDMC |
                                               1ull << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC;

constexpr ncclSymkKernelMask kernelMask_GenkLL128 =
  1ull << ncclSymkKernelId_AllReduce_Ring_LL128 | 1ull << ncclSymkKernelId_AllReduce_Tree_LL128 |
  1ull << ncclSymkKernelId_AllGather_Ring_LL128 | 1ull << ncclSymkKernelId_ReduceScatter_Ring_LL128 |
  1ull << ncclSymkKernelId_Broadcast_Ring_LL128 | 1ull << ncclSymkKernelId_Reduce_Ring_LL128;
constexpr ncclSymkKernelMask kernelMask_GenkLL128Reduce =
  1ull << ncclSymkKernelId_AllReduce_Ring_LL128 | 1ull << ncclSymkKernelId_AllReduce_Tree_LL128 |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_LL128 | 1ull << ncclSymkKernelId_Reduce_Ring_LL128;

constexpr ncclSymkKernelMask kernelMask_LL =
  1ull << ncclSymkKernelId_AllReduce_AGxLL_R | 1ull << ncclSymkKernelId_AllReduce_AGxLLMC_R |
  1ull << ncclSymkKernelId_AllGather_LL | 1ull << ncclSymkKernelId_AllGather_LLMC |
  1ull << ncclSymkKernelId_ReduceScatter_LL | 1ull << ncclSymkKernelId_AllReduce_Ring_LL |
  1ull << ncclSymkKernelId_AllReduce_Tree_LL | 1ull << ncclSymkKernelId_AllGather_Ring_LL |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_LL | 1ull << ncclSymkKernelId_Broadcast_Ring_LL |
  1ull << ncclSymkKernelId_Reduce_Ring_LL | kernelMask_GenkLL128 | 1ull << ncclSymkKernelId_AlltoAll_LL;

constexpr ncclSymkKernelMask kernelMask_Bcast = 1ull << ncclSymkKernelId_Broadcast_Ring_Simple |
                                                1ull << ncclSymkKernelId_Broadcast_Ring_LL |
                                                1ull << ncclSymkKernelId_Broadcast_Ring_LL128;
constexpr ncclSymkKernelMask kernelMask_Reduce = 1ull << ncclSymkKernelId_Reduce_Ring_Simple |
                                                 1ull << ncclSymkKernelId_Reduce_Ring_LL |
                                                 1ull << ncclSymkKernelId_Reduce_Ring_LL128;

constexpr ncclSymkKernelMask kernelMask_A2A = 1ull << ncclSymkKernelId_AlltoAll_LL;

constexpr ncclSymkKernelMask kernelMask_AG =
  1ull << ncclSymkKernelId_AllGather_LL | 1ull << ncclSymkKernelId_AllGather_LLMC |
  1ull << ncclSymkKernelId_AllGather_ST | 1ull << ncclSymkKernelId_AllGather_STMC |
  1ull << ncclSymkKernelId_AllGather_TmaST | 1ull << ncclSymkKernelId_AllGather_TmaSTMC |
  1ull << ncclSymkKernelId_AllGather_RailRing_LsaSTMC | 1ull << ncclSymkKernelId_AllGather_Ring_Simple |
  1ull << ncclSymkKernelId_AllGather_Ring_LL | 1ull << ncclSymkKernelId_AllGather_Ring_LL128 |
  1ull << ncclSymkKernelId_AllGather_RailRing_LsaST | 1ull << ncclSymkKernelId_AllGather_HierLsaST |
  1ull << ncclSymkKernelId_AllGather_HierLsaTmaST;

constexpr ncclSymkKernelMask kernelMask_AR =
  1ull << ncclSymkKernelId_AllReduce_AGxLLMC_R | 1ull << ncclSymkKernelId_AllReduce_AGxLL_R |
  1ull << ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC | 1ull << ncclSymkKernelId_AllReduce_RSxLD_AGxST |
  1ull << ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST | 1ull << ncclSymkKernelId_AllReduce_Ring_Simple |
  1ull << ncclSymkKernelId_AllReduce_Tree_Simple | 1ull << ncclSymkKernelId_AllReduce_Ring_LL |
  1ull << ncclSymkKernelId_AllReduce_Tree_LL | 1ull << ncclSymkKernelId_AllReduce_Ring_LL128 |
  1ull << ncclSymkKernelId_AllReduce_Tree_LL128;

constexpr ncclSymkKernelMask kernelMask_RS =
  1ull << ncclSymkKernelId_ReduceScatter_LD | 1ull << ncclSymkKernelId_ReduceScatter_LDMC |
  1ull << ncclSymkKernelId_ReduceScatter_TmaLD | 1ull << ncclSymkKernelId_ReduceScatter_LL |
  1ull << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD | 1ull << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_Simple | 1ull << ncclSymkKernelId_ReduceScatter_Ring_LL |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_LL128;

constexpr ncclSymkKernelMask kernelMask_LSA =
  1ull << ncclSymkKernelId_AllReduce_AGxLL_R | 1ull << ncclSymkKernelId_AllReduce_AGxLLMC_R |
  1ull << ncclSymkKernelId_AllReduce_RSxLD_AGxST | 1ull << ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
  1ull << ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST | 1ull << ncclSymkKernelId_AllReduce_Ring_Simple |
  1ull << ncclSymkKernelId_AllReduce_Tree_Simple | 1ull << ncclSymkKernelId_AllGather_LL |
  1ull << ncclSymkKernelId_AllGather_LLMC | 1ull << ncclSymkKernelId_AllGather_ST |
  1ull << ncclSymkKernelId_AllGather_STMC | 1ull << ncclSymkKernelId_AllGather_TmaST |
  1ull << ncclSymkKernelId_AllGather_TmaSTMC | 1ull << ncclSymkKernelId_AllGather_Ring_Simple |
  1ull << ncclSymkKernelId_ReduceScatter_LL | 1ull << ncclSymkKernelId_ReduceScatter_LD |
  1ull << ncclSymkKernelId_ReduceScatter_LDMC | 1ull << ncclSymkKernelId_ReduceScatter_TmaLD |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_Simple | 1ull << ncclSymkKernelId_Broadcast_Ring_Simple |
  1ull << ncclSymkKernelId_Reduce_Ring_Simple | 1ull << ncclSymkKernelId_AllReduce_Ring_LL |
  1ull << ncclSymkKernelId_AllReduce_Tree_LL | 1ull << ncclSymkKernelId_AllGather_Ring_LL |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_LL | 1ull << ncclSymkKernelId_Broadcast_Ring_LL |
  1ull << ncclSymkKernelId_Reduce_Ring_LL | kernelMask_GenkLL128 | 1ull << ncclSymkKernelId_AlltoAll_LL;

constexpr ncclSymkKernelMask kernelMask_GinOnly = 1ull << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD |
                                                  1ull << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC |
                                                  1ull << ncclSymkKernelId_AllGather_RailRing_LsaST |
                                                  1ull << ncclSymkKernelId_AllGather_RailRing_LsaSTMC;

constexpr ncclSymkKernelMask kernelMask_Genk =
  1ull << ncclSymkKernelId_AllReduce_Ring_Simple | 1ull << ncclSymkKernelId_AllReduce_Tree_Simple |
  1ull << ncclSymkKernelId_AllGather_Ring_Simple | 1ull << ncclSymkKernelId_ReduceScatter_Ring_Simple |
  1ull << ncclSymkKernelId_Broadcast_Ring_Simple | 1ull << ncclSymkKernelId_Reduce_Ring_Simple |
  1ull << ncclSymkKernelId_AllReduce_Ring_LL | 1ull << ncclSymkKernelId_AllReduce_Tree_LL |
  1ull << ncclSymkKernelId_AllGather_Ring_LL | 1ull << ncclSymkKernelId_ReduceScatter_Ring_LL |
  1ull << ncclSymkKernelId_Broadcast_Ring_LL | 1ull << ncclSymkKernelId_Reduce_Ring_LL | kernelMask_GenkLL128;
constexpr ncclSymkKernelMask kernelMask_GenkReduce =
  1ull << ncclSymkKernelId_AllReduce_Ring_Simple | 1ull << ncclSymkKernelId_AllReduce_Tree_Simple |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_Simple | 1ull << ncclSymkKernelId_Reduce_Ring_Simple |
  1ull << ncclSymkKernelId_AllReduce_Ring_LL | 1ull << ncclSymkKernelId_AllReduce_Tree_LL |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_LL | 1ull << ncclSymkKernelId_Reduce_Ring_LL | kernelMask_GenkLL128Reduce;
constexpr ncclSymkKernelMask kernelMask_GinCapable = kernelMask_GinOnly | kernelMask_Genk;

constexpr ncclSymkKernelMask kernelMask_Tma =
  1ull << ncclSymkKernelId_AllGather_TmaST | 1ull << ncclSymkKernelId_AllGather_TmaSTMC |
  1ull << ncclSymkKernelId_AllGather_HierLsaTmaST | 1ull << ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST |
  1ull << ncclSymkKernelId_ReduceScatter_TmaLD;

constexpr ncclSymkKernelMask kernelMask_DynamicSmem = (kernelMask_GinOnly & kernelMask_RS) | kernelMask_Tma;

// Needs the LSA team to factor into equal-size packages
constexpr ncclSymkKernelMask kernelMask_Hier =
  1ull << ncclSymkKernelId_AllGather_HierLsaST | 1ull << ncclSymkKernelId_AllGather_HierLsaTmaST;

constexpr ncclSymkKernelMask kernelMask_Symk = (kernelMask_AG | kernelMask_AR | kernelMask_RS) & ~kernelMask_Genk;

ncclSymkKernelMask ncclSymkLLKernelMask() {
  return kernelMask_LL;
}
ncclSymkKernelMask ncclSymkDynamicSmemKernelMask() {
  return kernelMask_DynamicSmem;
}
ncclSymkKernelMask ncclSymkTmaKernelMask() {
  return kernelMask_Tma;
}

ncclSymkKernelMask ncclSymkGinKernelMask() {
  return kernelMask_GinOnly | kernelMask_A2A;
}

ncclSymkKernelMask ncclSymkLsaKernelMask() {
  return kernelMask_LSA;
}

ncclSymkKernelMask ncclGenkKernelMask() {
  return kernelMask_Genk;
}

ncclSymkKernelMask ncclSymkAGKernelMask() {
  return kernelMask_AG;
}

ncclSymkKernelMask ncclSymkARKernelMask() {
  return kernelMask_AR;
}

ncclSymkKernelMask ncclSymkRSKernelMask() {
  return kernelMask_RS;
}

ncclSymkKernelMask ncclSymkNonGenkKernelMask() {
  return kernelMask_Symk;
}

int ncclSymkKernelMaxWarps(ncclSymkKernelId kernelId, int red, ncclDataType_t ty) {
  int kernelIndex = ncclSymkGetKernelIndex(kernelId, red, ty);
  int maxThreads = kernelIndex < 0 ? 0 : ncclSymkKernelMaxThreads[kernelIndex];
  if (maxThreads <= 0) maxThreads = ncclSymkMaxThreads;
  return std::max(1, maxThreads / WARP_SIZE);
}

// Return a device-specific grid limit at which every CTA of this exact kernel
// can be resident. Use the lower occupancy of the normal and instrumented
// variants because either may be selected after tuning.
static ncclResult_t ncclSymkKernelMaxResidentCtas(ncclSymkKernelId kernelId, int red, ncclDataType_t ty,
                                                  int* maxResidentCtas) {
  *maxResidentCtas = 1;
#if !defined(NCCL_OS_WINDOWS)
  int kernelIndex = ncclSymkGetKernelIndex(kernelId, red, ty);
  if (kernelIndex < 0 || ncclSymkKernelList[kernelIndex] == nullptr) return ncclSuccess;

  int threads = ncclSymkKernelMaxThreads[kernelIndex];
  if (threads <= 0) threads = ncclSymkMaxThreads;

  int blocksPerSmLimit = 0;
  void* variants[2] = {ncclSymkKernelList[kernelIndex], ncclSymkKernelListProfile[kernelIndex]};
  for (void* fn : variants) {
    if (fn == nullptr) continue;
    int blocksPerSm;
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, fn, threads, /*dynamicSMemSize=*/0));
    if (blocksPerSmLimit == 0 || blocksPerSm < blocksPerSmLimit) {
      blocksPerSmLimit = blocksPerSm;
    }
  }

  int cudaDev;
  int multiprocessorCount;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUDACHECK(cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, cudaDev));
  *maxResidentCtas = std::max(1, std::min(ncclSymkMaxBlocks, blocksPerSmLimit * multiprocessorCount));
#endif
  return ncclSuccess;
}

// Host picker: true when nBytes is large enough for this kernel's TMA deep loop at nBlocks.
bool ncclSymkTmaDeepEligible(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, int nBlocks) {
  int bytePerChunk = 0;
  int chunkMod = 0; // imodFast32 divisor on deep-loop chunk count (see src/device/symmetric/*.cuh)
  switch (k) {
  case ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST:
  case ncclSymkKernelId_ReduceScatter_TmaLD:
    bytePerChunk = ncclSymkDeepMaxBytePerChunk;
    chunkMod = comm->nRanks * nBlocks;
    break;
  case ncclSymkKernelId_AllGather_TmaST:
    // Picker bar uses the 16 B-aligned deep tier; the 256 B TMA tier may still be skipped.
    bytePerChunk = ncclSymkBytePerChunk;
    chunkMod = nBlocks;
    break;
  case ncclSymkKernelId_AllGather_TmaSTMC:
    bytePerChunk = ncclSymkMultimemDeepBytePerChunk;
    chunkMod = 1;
    break;
  case ncclSymkKernelId_AllGather_HierLsaTmaST:
    // Each chunk goes through its own single-block bcast<Deep=true>(), so the first one reaches the 16 B deep
    // tier whatever the block count; the 256 B tier may still be skipped. Deep unrolls that tier 4x.
    bytePerChunk = ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, 4 * ncclSymkUnrollPacks);
    chunkMod = 1;
    break;
  default:
    WARN("Unexpected kernel id %d in ncclSymkTmaDeepEligible", (int)k);
    return false;
  }
  if (bytePerChunk == 0 || chunkMod == 0) return false;
  return nBytes >= (size_t)bytePerChunk * (size_t)chunkMod;
}

static ncclSymkKernelMask kernelMask_coll(ncclFunc_t coll) {
  switch (coll) {
  case ncclFuncBroadcast:
    return kernelMask_Bcast;
  case ncclFuncReduce:
    return kernelMask_Reduce;
  case ncclFuncAlltoAll:
    return kernelMask_A2A;
  case ncclFuncAllGather:
    return kernelMask_AG;
  case ncclFuncAllReduce:
    return kernelMask_AR;
  case ncclFuncReduceScatter:
    return kernelMask_RS;
  default:
    return 0;
  }
}

NCCL_PARAM(SymGinKernelsEnable, "SYM_GIN_KERNELS_ENABLE", 1)
NCCL_PARAM(SymkLLScratchSize, "SYMK_LL_SCRATCH_SIZE", 256 << 10)
NCCL_PARAM(SymRsGinChunkSize, "SYM_RS_GIN_CHUNK_SIZE", -1)

// [RCCL] Anvil SDMA reaches a GIN peer only through its slot in this rank's LSA window, and every
// rail peer of the GIN kernels is outside the LSA team.
static bool ncclSymkGinBackendUnsupported(struct ncclComm* comm) {
  ncclGinType_t ginType = NCCL_GIN_TYPE_NONE;
  return ncclGetRailedGinType(comm, &ginType) == ncclSuccess && ginType == NCCL_GIN_TYPE_ANVIL_SDMA;
}
// [RCCL] These kernels stage tiles through a DMA engine: TMA on NVIDIA, the Tensor
// Data Mover on gfx1250. On by default as in NCCL 2.31; ncclSymkTmaAvailable() still
// confines them to gfx1250 builds whose SDK ships the TDM descriptor header.
// 0 off, 1 offer to the tuner, 2 force (skips the cost model and the deep-loop size bar; for A/B
// measurement). NCCL_SYM_KERNEL forces one named kernel, 2 forces whichever the collective has.
NCCL_PARAM(SymTmaEnable, "SYM_TMA_ENABLE", 1)
#ifdef SYM_GENK
NCCL_PARAM(SymGenkEnable, "SYM_GENK_ENABLE", 0)
#else
NCCL_PARAM(SymGenkEnableRequested, "SYM_GENK_ENABLE", 0)
int64_t ncclParamSymGenkEnable() {
  // [RCCL] Atomic: concurrent comm inits may reach this at the same time.
  static std::atomic<bool> warned{false};
  if (!warned.exchange(true, std::memory_order_relaxed)) {
    if (ncclParamSymGenkEnableRequested() != 0) {
      ATTN("NCCL_SYM_GENK_ENABLE is set but NCCL was built without GENK kernels (SYM_GENK=0)");
    }
  }
  return 0;
}
#endif

static bool ncclGenkGinAvailable(struct ncclComm* comm) {
#if !defined(NCCL_OS_WINDOWS)
  if (comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL) return false;

  struct ncclGinState const& ginState = comm->ginState;
  for (int i = 0; i < ginState.numActiveBackends; i++) {
    if (ginState.backends[i].ginType == NCCL_GIN_TYPE_PROXY) return true;
  }
#else
  (void)comm;
#endif
  return false;
}

extern int64_t ncclParamSymCTAs();

bool ncclSymkTmaAvailable(struct ncclComm* comm) {
  if (!ncclParamSymTmaEnable()) return false;
#if defined(__HIP_PLATFORM_AMD__)
  return TDM_TOOLCHAIN_AVAILABLE && comm->archName && IsArchMatch(comm->archName, "gfx1250");
#else
  // TMA requires up to (8KB data + 8B mbarrier + alignment) x 16 warps SMEM.
  // SMEM is partitioned across the 16 warps such that each warp gets ncclTmaShmemScratchWarpSize() bytes.
  if (comm->maxSharedMemOptin < ncclTmaShmemScratchWarpSize() * 16) {
    return false;
  }
  return comm->minCompCap >= 100;
#endif
}

bool ncclSymkTmaForced(struct ncclComm* comm) {
  // Availability still gates it: an arch that emitted no Tma kernels has none to force to.
  return ncclParamSymTmaEnable() >= 2 && ncclSymkTmaAvailable(comm);
}

bool ncclSymkAlltoAllLLForced(struct ncclComm* comm) {
  constexpr int tuningId = NCCL_TUNING_SYM_KERNEL_ID_OFFSET + ncclSymkKernelId_AlltoAll_LL;
  int tuningFunc = ncclTuningFuncIndex(ncclFuncAlltoAll);
  return comm->tuningContext.forced[tuningFunc] && comm->tuningContext.enabled[tuningId][tuningFunc];
}

// [RCCL] No HIP build has the AlltoAll_LL kernel (see ncclSymkImplemented), so forcing it must not provision its
// scratch or GIN resources. The force itself still stands, so enqueue rejects the AlltoAll it cannot run.
static bool ncclSymkAlltoAllLLNeedsResources(struct ncclComm* comm) {
#if defined(__HIP_PLATFORM_AMD__)
  (void)comm;
  return false;
#else
  return ncclSymkAlltoAllLLForced(comm);
#endif
}

static size_t ncclSymkAlltoAllLLScratchBudget() {
  int64_t param = ncclParamSymkLLScratchSize();
  if (param <= (int64_t)sizeof(ncclSymkAlltoAllLLScratchState)) return 0;
  size_t budget = std::min<size_t>((size_t)param, UINT32_MAX);
  return budget & ~(ncclSymkAlltoAllLLScratchAlignment - 1);
}

static size_t ncclSymkAlltoAllLLProvisionBytes(struct ncclComm* comm) {
  if (comm->symkState.initialized) return comm->symkState.alltoAllLLProvisionBytes;
  if (!ncclSymkAlltoAllLLNeedsResources(comm)) return 0;
  size_t scratchBudget = ncclSymkAlltoAllLLScratchBudget();
  if (scratchBudget <= sizeof(ncclSymkAlltoAllLLScratchState)) return 0;
  if (comm->nRanks <= 1) return ncclSymkAlltoAllLLMaxPayloadBytes;

  // Derive a topology-independent provision from four all-peer plane capacities.
  // Initialization compacts the actual GIN send planes after this limit is known.
  size_t planeBytes = ncclSymkAlltoAllLLPlaneCapacityForBudget(scratchBudget);
  size_t linesPerPeer = planeBytes / ncclSymkAlltoAllLLWireBytesPerLine / (comm->nRanks - 1);
  size_t scratchLimit = linesPerPeer * ncclSymkAlltoAllLLPayloadBytesPerLine;
  return std::min(ncclSymkAlltoAllLLMaxPayloadBytes, scratchLimit);
}

static constexpr size_t ncclSymkRsGinDefaultChunkBytes = 128 << 10;
static constexpr size_t ncclSymkRsGinMinChunkBytes = 128;
static constexpr size_t ncclSymkRsGinMaxChunkBytes = size_t(1) << 30;

size_t ncclSymkRsGinChunkBytes() {
  int64_t param = ncclParamSymRsGinChunkSize();
  size_t chunkBytes = param > 0 ? (size_t)param : ncclSymkRsGinDefaultChunkBytes;
  chunkBytes = std::max(ncclSymkRsGinMinChunkBytes, std::min(chunkBytes, ncclSymkRsGinMaxChunkBytes));
  return pow2Down(chunkBytes);
}

static uint32_t ncclSymkRsGinAccumBytesPerBlock() {
  return (uint32_t)alignUp(2 * ncclSymkRsGinChunkBytes(), 128);
}

static void getRequirements_gin(struct ncclComm* comm, int* out_nBlocks, size_t* out_bufSize) {
  *out_nBlocks = 0;
  *out_bufSize = 0;
  for (int ldmc = 0; ldmc <= 1; ldmc++) {
    double lsaBw = ncclTuningGetLsaBw(comm);
    double ginBw = ncclTuningGetGinBw(comm);
    double ginLat = ncclTuningGetGinLat(comm);
    double smLat = ncclTuningGetSmLatReduceScatterRailA2A(comm, ldmc);
    double smMul, lsaMul, ginMul;
    ncclTuningGetBusMulReduceScatterRailA2A(comm, ldmc, &smMul, &lsaMul, &ginMul);
    // GIN could be throttled by LSA work
    double ginBwRenorm = std::min(lsaBw / lsaMul, ginBw / ginMul) * ginMul;
    size_t bufSize = ginBwRenorm * (ginLat + smLat);
    int nBlocks = ncclTuningCalcSatBlocksReduceScatterRailA2A(comm, ldmc);
    if (comm->rank == 0) {
      double minLsaGinEffBw = std::min(lsaBw / lsaMul, ginBw / ginMul);
      INFO(NCCL_TUNING, "ReduceScatter_RailA2A_Lsa%s : satblocks=%d bufsize=%d effbw=%g", ldmc ? "LDMC" : "LD", nBlocks,
           (int)bufSize, minLsaGinEffBw * smMul);
    }
    *out_nBlocks = std::max(*out_nBlocks, nBlocks);
    *out_bufSize = std::max(*out_bufSize, bufSize);
  }
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// The block width tuning is fitted to gfx950 and must not reach other architectures. It sizes the
// shared LL slots and blockDim, which peers must agree on, so every rank has to share this arch.
bool ncclSymkIsGfx950(struct ncclComm* comm) {
  return comm->minCompCap == comm->maxCompCap && comm->archName != nullptr &&
         IsArchMatch(comm->archName, "gfx950");
}

// The partition-mode AllGather tuning is fitted to gfx1250 and must not reach other architectures.
// Peers must pick the same kernel, so every rank has to share this arch.
bool ncclSymkIsGfx1250(struct ncclComm* comm) {
  return comm->minCompCap == comm->maxCompCap && comm->archName != nullptr &&
         IsArchMatch(comm->archName, "gfx1250");
}
#endif

// The hierarchical AllGather is tuned and validated only on gfx1250 in partition mode.
static bool ncclSymkAgHierAvailable(struct ncclComm* comm) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return ncclSymkIsGfx1250(comm) && comm->hasMloPart;
#else
  (void)comm;
  return false;
#endif
}

static bool sameCard(struct ncclPeerInfo const& a, struct ncclPeerInfo const& b) {
  return a.hostHash == b.hostHash && (a.busId & ~0xfLL) == (b.busId & ~0xfLL);
}

// How many consecutive ranks share a physical device.
static int computeHierScaleInSize(struct ncclComm* comm) {
  struct ncclPeerInfo* peerInfo = comm->peerInfo;
  if (peerInfo == nullptr || comm->nRanks < 2) return 0;

  int size = 1;
  while (size < comm->nRanks && sameCard(peerInfo[size], peerInfo[0])) size++;
  if (size < 2 || size == comm->nRanks || comm->nRanks % size != 0) return 0;

  for (int r = 1; r < comm->nRanks; r++) {
    if (sameCard(peerInfo[r], peerInfo[r - 1]) != (r % size != 0)) return 0;
  }
  INFO(NCCL_INIT, "Symmetric hierarchical scale-in size = %d (nRanks %d)", size, comm->nRanks);
  return size;
}

// Packages the hierarchical AllGather spans, or 0 where it can't run: it needs gfx1250 in partition
// mode and an LSA team of at least two equal packages, so there is a cross-package hop to save.
static int ncclSymkAgHierGroups(struct ncclComm* comm) {
  int scaleIn = comm->symkState.kcomm.hierScaleInSize;
  int lsaRanks = ncclTeamLsa(comm).nRanks;
  if (!ncclSymkAgHierAvailable(comm) || scaleIn <= 0 || lsaRanks % scaleIn != 0 || lsaRanks / scaleIn < 2) return 0;
  return lsaRanks / scaleIn;
}

int ncclSymkAgHierLaunchBlocks(struct ncclComm* comm, int nBlocks) {
  int nGroups = ncclSymkAgHierGroups(comm);
  if (nGroups == 0) return nBlocks;
  int nUsed = ncclSymkAgHierG0Blocks(nBlocks, nGroups) + (nGroups - 1) * ncclSymkAgHierGroupBlocks(nBlocks, nGroups);
  return std::min(nBlocks, nUsed);
}

ncclResult_t ncclSymkInitOnce(struct ncclComm* comm) {
  // ncclTeamLsa() below calls this internally but drops the error code so we do it here.
  NCCLCHECK(ncclDevrInitOnce(comm));

  struct ncclSymkState* symk = &comm->symkState;
  if (!symk->initialized) {
    bool alltoAllLLNeedsResources = ncclSymkAlltoAllLLNeedsResources(comm);
    if (alltoAllLLNeedsResources) {
      NCCLCHECK(ncclSymkKernelMaxResidentCtas(ncclSymkKernelId_AlltoAll_LL, /*red=*/0, ncclUint8,
                                              &symk->alltoAllLLMaxResidentCtas));
    }

    symk->kcomm.hierScaleInSize = computeHierScaleInSize(comm);
    int agHierGroups = ncclSymkAgHierGroups(comm);
    if (ncclSymkAgHierAvailable(comm) && symk->kcomm.hierScaleInSize == 0) {
      INFO(NCCL_INIT,
           "Symmetric hierarchical AllGather off: ranks do not form two or more consecutive, equal-size groups "
           "per physical device (nRanks %d)", comm->nRanks);
    }
    struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    int lsaRanks = ncclTeamLsa(comm).nRanks;

    // Disable LSA multicast for cross-clique since NVLS isn't available across cliques.
    symk->hasLsaMultimem = ncclNvlsSymmetricMultimemEnabled(comm) && lsaRanks > 2 && !comm->p2pCrossClique;
    reqs.lsaMultimem = symk->hasLsaMultimem;
    reqs.lsaBarrierCount = agHierGroups > 0 ? std::max(ncclSymkMaxBlocks, ncclSymkAgHierMaxBlocks) : ncclSymkMaxBlocks;
    reqs.ginStrongSignalsRequired = false;
    reqs.ginVaSignalsRequired = false;

    // Sized for the widest LL launch any collective will use, since one shared buffer is allocated
    // here before the first collective is known. Doubling the width costs 4 MiB on an 8-rank comm.
    int llThreads = ncclSymkMaxThreads;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (ncclSymkIsGfx950(comm)) llThreads = ncclSymkGfx950LLThreads;
#endif

    // Preserve the existing LSA LL resource used by AllGather and
    // ReduceScatter. AlltoAll_LL uses the separate scratch resource below.
    struct ncclDevResourceRequirements lla2aReq;
    ncclLLA2ACreateRequirement(ncclSymkMaxBlocks,
                               ncclLLA2ACalcSlots(lsaRanks * llThreads, ncclSymkLLMaxEltSize),
                               &symk->kcomm.lsaLLA2A, &lla2aReq);
    lla2aReq.next = reqs.resourceRequirementsList;
    reqs.resourceRequirementsList = &lla2aReq;

    struct ncclDevResourceRequirements hierPushFlagsReq = {};
    if (agHierGroups > 0) {
      hierPushFlagsReq.bufferSize = size_t(ncclSymkAgHierMaxBlocks) * agHierGroups * sizeof(uint64_t);
      hierPushFlagsReq.bufferAlign = 128;
      hierPushFlagsReq.outBufferHandle = &symk->kcomm.hierPushFlags;
      hierPushFlagsReq.next = reqs.resourceRequirementsList;
      reqs.resourceRequirementsList = &hierPushFlagsReq;
    }

    size_t alltoAllLLScratchBudget = alltoAllLLNeedsResources ? ncclSymkAlltoAllLLScratchBudget() : 0;
    symk->alltoAllLLProvisionBytes = (uint32_t)ncclSymkAlltoAllLLProvisionBytes(comm);
    size_t alltoAllLLReceivePlaneBytes =
      ncclSymkAlltoAllLLRequiredReceivePlaneBytes(comm->nRanks, symk->alltoAllLLProvisionBytes);
    size_t alltoAllLLSendPlaneBytes =
      ncclSymkAlltoAllLLRequiredSendPlaneBytes(comm->nRanks, lsaRanks, symk->alltoAllLLProvisionBytes);
    symk->kcomm.alltoAllLLScratch.receivePlaneBytes = (uint32_t)alltoAllLLReceivePlaneBytes;
    symk->kcomm.alltoAllLLScratch.sendPlaneBytes = (uint32_t)alltoAllLLSendPlaneBytes;

    if (comm->rank == 0 && alltoAllLLNeedsResources &&
        symk->alltoAllLLProvisionBytes < ncclSymkAlltoAllLLMaxPayloadBytes) {
      INFO(NCCL_INIT,
           "AlltoAll_LL: NCCL_SYMK_LL_SCRATCH_SIZE=%zu supports %u bytes per peer at %d ranks; "
           "increase it to at least %zu bytes to support the %zu-byte kernel maximum",
           alltoAllLLScratchBudget, symk->alltoAllLLProvisionBytes, comm->nRanks,
           ncclSymkAlltoAllLLRequiredScratchBudgetBytes(comm->nRanks, ncclSymkAlltoAllLLMaxPayloadBytes),
           ncclSymkAlltoAllLLMaxPayloadBytes);
    }

    struct ncclDevResourceRequirements alltoAllLLScratchReq = {};
    if (symk->alltoAllLLProvisionBytes != 0) {
      alltoAllLLScratchReq.bufferSize =
        ncclSymkAlltoAllLLRequiredScratchBytes(comm->nRanks, lsaRanks, symk->alltoAllLLProvisionBytes);
      alltoAllLLScratchReq.bufferAlign = ncclSymkAlltoAllLLScratchAlignment;
      alltoAllLLScratchReq.outBufferHandle = &symk->kcomm.alltoAllLLScratch.bufHandle;
      alltoAllLLScratchReq.next = reqs.resourceRequirementsList;
      reqs.resourceRequirementsList = &alltoAllLLScratchReq;
    }

    struct ncclDevResourceRequirements ginInboxRailReq = {};
    struct ncclDevResourceRequirements ginOutboxReq = {};
    struct ncclDevResourceRequirements rsGinAccumReq = {};
    struct ncclDevResourceRequirements railSignalReq = {};
    const bool needGin = lsaRanks < comm->nRanks;
    symk->ginBackendUnsupported = needGin && ncclSymkGinBackendUnsupported(comm);
    if (symk->ginBackendUnsupported && ncclParamSymGinKernelsEnable()) {
      INFO(NCCL_INIT, "Symmetric GIN kernels disabled: the Anvil SDMA GIN backend cannot reach rail peers outside "
                      "the LSA team (lsaSize=%d, nRanks=%d)",
           lsaRanks, comm->nRanks);
    }
    if (ncclParamSymGinKernelsEnable() && needGin && !symk->ginBackendUnsupported) {
      ncclSymkKernelMask enabledGinKernels = kernelMask_GinOnly | kernelMask_A2A;
      bool needAlltoAllGin = (enabledGinKernels & kernelMask_A2A) != 0 &&
                             comm->globalGinSupport == NCCL_GIN_CONNECTION_FULL && symk->alltoAllLLProvisionBytes != 0;
      bool needRailGin = (enabledGinKernels & ~kernelMask_A2A) != 0;

      // AlltoAll LL reuses this devComm's FULL GIN contexts and the unified
      // signal-free scratch resource created above. Rail inbox state remains
      // separate because it carries credits and signals that AlltoAll does not use.

      if (needRailGin) {
        int maxBlocks;
        size_t bufSize;
        getRequirements_gin(comm, &maxBlocks, &bufSize);

        maxBlocks = std::max(maxBlocks, comm->config.minCTAs);
        maxBlocks = std::min(maxBlocks, comm->config.maxCTAs);
        if (ncclParamSymCTAs() >= 1) maxBlocks = ncclParamSymCTAs();
        maxBlocks = std::min(maxBlocks, ncclSymkMaxBlocks);
        symk->maxGinInboxBlocks = maxBlocks;
        symk->kcomm.rsGinAccumBytesPerBlock = ncclSymkRsGinAccumBytesPerBlock();

        rsGinAccumReq.bufferSize = (size_t)maxBlocks * symk->kcomm.rsGinAccumBytesPerBlock;
        rsGinAccumReq.bufferAlign = 128;
        rsGinAccumReq.outBufferHandle = &symk->kcomm.rsGinAccumBuf;
        rsGinAccumReq.next = reqs.resourceRequirementsList;
        reqs.resourceRequirementsList = &rsGinAccumReq;

        ncclGinInboxA2ACreateRequirement(ncclTeamRail(comm), maxBlocks, log2Up(bufSize), &symk->kcomm.ginInboxRail,
                                         &ginInboxRailReq);
        ginInboxRailReq.next = reqs.resourceRequirementsList;
        reqs.resourceRequirementsList = &ginInboxRailReq;

        ncclGinOutboxCreateRequirement(maxBlocks, log2Up(bufSize), &symk->kcomm.ginOutbox, &ginOutboxReq);
        ginOutboxReq.next = reqs.resourceRequirementsList;
        reqs.resourceRequirementsList = &ginOutboxReq;

        uint32_t railSignalCount = ncclTeamRail(comm).nRanks * ncclSymkMaxBlocks;
        railSignalReq.bufferSize = 0;
        railSignalReq.bufferAlign = 0;
        railSignalReq.outBufferHandle = nullptr;
        railSignalReq.ginSignalCount = railSignalCount;
        railSignalReq.outGinSignalStart = &symk->kcomm.ginSyncHandle.railSignals;
        railSignalReq.next = reqs.resourceRequirementsList;
        reqs.resourceRequirementsList = &railSignalReq;

        reqs.barrierCount = ncclSymkMaxBlocks;
      }

      // AlltoAll_LL only enters its world barrier while clearing receive
      // scratch at LL-flag rollover, so one hybrid barrier slot is sufficient.
      if (needAlltoAllGin) reqs.barrierCount = std::max(reqs.barrierCount, 1);

      if (needAlltoAllGin || needRailGin) {
        reqs.ginConnectionType = needAlltoAllGin ? NCCL_GIN_CONNECTION_FULL : NCCL_GIN_CONNECTION_RAIL;
        reqs.ginStrongSignalsRequired = needAlltoAllGin || needRailGin;
        reqs.ginVaSignalsRequired = needRailGin;
      }
    }

    NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &symk->kcomm.devComm, /*isInternal=*/true,
                                         /*deviceCodeVersion=*/NCCL_VERSION_CODE));
    symk->kcomm.workStarted = comm->profiler.symWorkStarted;
    symk->kcomm.workCompleted = comm->profiler.symWorkCompleted;
    symk->kcomm.workPhases = comm->profiler.symWorkPhases;
    symk->initialized = true;
  }
  return ncclSuccess;
}

ncclResult_t ncclSymkFinalize(struct ncclComm* comm) {
  struct ncclSymkState* symk = &comm->symkState;
  if (symk->genkInitialized) {
    NCCLCHECK(ncclDevCommDestroy(comm, &symk->genkComm.devComm));
    symk->genkInitialized = false;
  }
  if (symk->initialized) {
    NCCLCHECK(ncclDevCommDestroy(comm, &symk->kcomm.devComm));
  }
  return ncclSuccess;
}

static bool ncclGenkReduceImplemented(int /*ncclDevRedOp_t*/ red) {
  return red == ncclDevSum || red == ncclDevProd || red == ncclDevMinMax;
}

static bool ncclSymkOtherReduceImplemented(int /*ncclDevRedOp_t*/ red, ncclDataType_t ty, ncclFunc_t coll) {
  bool isFloat;
  switch (ty) {
  case ncclFloat64:
  case ncclFloat32:
  case ncclFloat16:
  case ncclBfloat16:
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    isFloat = true;
    break;
  default:
    isFloat = false;
    break;
  }
  if (coll == ncclFuncReduceScatter) {
    return (red == ncclDevSum || red == ncclDevSumPostDiv) && isFloat && ty != ncclFloat64;
  }
  return (red == ncclDevSum) && isFloat && ty != ncclFloat64;
}

static bool ncclSymkImplemented(ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
  switch (coll) {
  case ncclFuncAlltoAll:
#if defined(NCCL_OS_WINDOWS) || defined(__HIP_PLATFORM_AMD__)
    // generate.py excludes this GIN-capable kernel from the Windows table.
    // [RCCL] and from every HIP build (AlltoAll_LL is inline PTX with no ROCm port).
    return false;
#else
    return true;
#endif
  case ncclFuncBroadcast:
  case ncclFuncAllGather:
    return true;
  case ncclFuncReduce:
  case ncclFuncAllReduce:
  case ncclFuncReduceScatter:
    return ncclGenkReduceImplemented(red) || ncclSymkOtherReduceImplemented(red, ty, coll);
  default:
    return false;
  }
}

ncclSymkKernelMask ncclSymkMask(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                                size_t nElts, bool symAligned16B) {
  ncclSymkKernelMask kmask = kernelMask_coll(coll);

  bool const isReduction = coll == ncclFuncReduce || coll == ncclFuncAllReduce || coll == ncclFuncReduceScatter;
  if (isReduction) {
    if (!ncclGenkReduceImplemented(red)) kmask &= ~kernelMask_GenkReduce;
    if (!ncclSymkOtherReduceImplemented(red, ty, coll)) kmask &= kernelMask_GenkReduce;
  }

  bool hasSTMC = comm->symkState.hasLsaMultimem;
  bool hasLDMC = false;
  if (comm->symkState.hasLsaMultimem) {
    switch (ty) {
    case ncclInt32:
    case ncclUint32:
    case ncclInt64:
    case ncclUint64:
    case ncclFloat16:
    case ncclBfloat16:
      hasLDMC = red == ncclDevSum || red == ncclDevMinMax || red == ncclDevSumPostDiv;
      break;
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
      hasLDMC = red == ncclDevSum || red == ncclDevMinMax || red == ncclDevSumPostDiv;
      hasLDMC &= comm->compCap >= 100;
      break;
    case ncclFloat:
    case ncclDouble:
      hasLDMC = red == ncclDevSum || red == ncclDevSumPostDiv;
      break;
    default:
      break;
    }
  }
  if (!hasSTMC) kmask &= ~kernelMask_STMC;
  if (!hasLDMC) kmask &= ~kernelMask_LDMC;

  size_t rawBytes = nElts * ncclTypeSize(ty);
  size_t nBytes = coll == ncclFuncAlltoAll ? rawBytes : alignUp(rawBytes, NCCL_SYM_KERNEL_CELL_SIZE);
  size_t nBusBytes =
    (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter || coll == ncclFuncAlltoAll ? comm->nRanks : 1) *
    nBytes;
  // LL kernels use 32-bit ints to track element counts and indices.
  if (nBusBytes >= (size_t(2) << 30)) kmask &= ~kernelMask_LL;
  // Any kernel might use 32-bit int to track unrolled loop chunks (which are going
  // to be at least 32 bytes per chunk)
  if (nBusBytes >= 32 * (size_t(2) << 30)) kmask = 0;

  if (!ncclSymkTmaAvailable(comm)) kmask &= ~kernelMask_Tma;
  if (!symAligned16B) kmask &= ~kernelMask_Tma;
  // Force leaves the tuner nothing else to pick. Conditional so that a collective with no surviving
  // Tma kernel degrades to normal selection instead of ending up with an empty mask.
  if (ncclSymkTmaForced(comm) && (kmask & kernelMask_Tma) != 0) kmask &= kernelMask_Tma;

  // Specialized kernels still require direct NVLink; Genk can also use LSA over CUDA P2P or GIN.
  if (!comm->isAllDirectNvlink) kmask &= kernelMask_Genk;
  if (ncclParamSymGenkEnable() == 0 || comm->minCompCap < 90) kmask &= ~kernelMask_Genk;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  // [RCCL] Genk (general/flow) kernels are not built for AMD, and minCompCap < 90 does not exclude
  // gfx942/gfx950 (94/95). Remove them explicitly rather than relying on the absent SYM_GENK macro.
  kmask &= ~kernelMask_Genk;
#endif

  bool const hasGin = ncclParamSymGinKernelsEnable() != 0 && !comm->symkState.ginBackendUnsupported;
  bool const hasMultipleLsaTeams = ncclTeamLsa(comm).nRanks < comm->nRanks;
  if (hasMultipleLsaTeams) {
    kmask &= kernelMask_GinCapable | kernelMask_A2A;
    if (!hasGin) kmask &= ~(kernelMask_GinOnly | kernelMask_A2A);
    if (comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL) kmask &= ~kernelMask_A2A;
  } else {
    kmask &= ~kernelMask_GinOnly;
  }
  if (comm->nNodes > 1 && (!hasGin || !ncclGenkGinAvailable(comm))) kmask &= ~kernelMask_Genk;
  // Remove AlltoAll from kmask when it is unsupported.
  if (!comm->symkState.initialized || comm->p2pCrossClique || nBytes == 0 ||
      nBytes > comm->symkState.alltoAllLLProvisionBytes) {
    kmask &= ~kernelMask_A2A;
  }

  if (ncclSymkAgHierGroups(comm) == 0) kmask &= ~kernelMask_Hier;
  return kmask;
}

bool ncclSymkAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                       size_t nElts) {
  if (!comm->symmetricSupport) return false;
  if (!ncclSymkImplemented(coll, red, ty)) return false;

  return (ncclSymkMask(comm, coll, red, ty, nElts) != 0);
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// Thresholds bounding the block width of the gfx950 symmetric kernels, fitted on 8 ranks only.
// ReduceScatter's and AllGather's are bus bytes since their counts are per rank. AllReduce's are message bytes.
static constexpr size_t ncclSymkRsWideBlockMinBusBytes = 1 << 20;
static constexpr size_t ncclSymkRsNarrowBlockBusBytes = 16 << 20;
static constexpr size_t ncclSymkArTailSaturatedBytes = 512 << 10;
static constexpr size_t ncclSymkArDeepTierBytes = 2 << 20;
static constexpr size_t ncclSymkArOccupancyBoundBytes = 1 << 30;
// Below this message size AllReduce's LL packs fit few enough epochs that a wider block only adds
// threads to the epoch barrier without removing an epoch.
static constexpr size_t ncclSymkArLLWideBytes = 64 << 10;
static constexpr size_t ncclSymkAgLLWideBusBytes = 512 << 10;
static constexpr size_t ncclSymkAgWideBlockBusBytes = 64 << 20;
// Where AllGather's store kernel overtakes LL, which the shared cost model places past 8 MB.
static constexpr size_t ncclSymkAgStoreMinBusBytes = 1 << 20;
// Block widths those thresholds select between. 1024 is the widest workgroup gfx950 will launch.
static constexpr int ncclSymkGfx950NarrowThreads = 256;
static constexpr int ncclSymkGfx950WideThreads = 512;
static constexpr int ncclSymkGfx950WidestThreads = 1024;

bool ncclSymkGfx950AllGatherPrefersStore(int nRanks, size_t nBytes) {
  return size_t(nRanks) * nBytes >= ncclSymkAgStoreMinBusBytes;
}

// Where AllGather's store kernel overtakes LL on gfx1250 in partition mode, and where the hierarchical
// kernel overtakes the store kernel.
static constexpr size_t ncclSymkGfx1250AgLLMaxBusBytes = 1 << 20;
static constexpr size_t ncclSymkGfx1250AgSTMaxBusBytes = 16 << 20;

bool ncclSymkGfx1250AllGatherPrefersStore(int nRanks, size_t nBytes) {
  return size_t(nRanks) * nBytes >= ncclSymkGfx1250AgLLMaxBusBytes;
}

bool ncclSymkGfx1250AllGatherPrefersHier(int nRanks, size_t nBytes) {
  return size_t(nRanks) * nBytes >= ncclSymkGfx1250AgSTMaxBusBytes;
}

int ncclSymkGfx950BlockThreads(ncclFunc_t coll, bool isLL, int nRanks, size_t nBytes) {
  if (coll == ncclFuncAllGather) {
    // A wider LL block halves the epoch count, which pays from 512 KB. The store kernel stays narrow
    // below 64 MB, where a wider block leaves each warp too few iterations.
    size_t busBytes = size_t(nRanks) * nBytes;
    if (isLL) return busBytes >= ncclSymkAgLLWideBusBytes ? ncclSymkGfx950LLThreads : ncclSymkGfx950NarrowThreads;
    return busBytes >= ncclSymkAgWideBlockBusBytes ? ncclSymkGfx950WideThreads : ncclSymkGfx950NarrowThreads;
  }
  if (coll != ncclFuncReduceScatter && coll != ncclFuncAllReduce) return ncclSymkMaxThreads;

  if (isLL) {
    // AllReduce narrows below the threshold, where a wider block only adds threads to the epoch
    // barrier. ReduceScatter always stays at the full width.
    bool narrowLL = coll == ncclFuncAllReduce && nBytes < ncclSymkArLLWideBytes;
    return narrowLL ? ncclSymkGfx950NarrowThreads : ncclSymkGfx950LLThreads;
  }

  if (coll == ncclFuncReduceScatter) {
    // Small sizes are latency bound on per-peer loads and want every thread. Large ones are
    // bandwidth bound, where a narrower block keeps iterations per globally strided warp high.
    size_t busBytes = size_t(nRanks) * nBytes;
    if (busBytes >= ncclSymkRsNarrowBlockBusBytes) return ncclSymkGfx950NarrowThreads;
    if (busBytes >= ncclSymkRsWideBlockMinBusBytes) return ncclSymkGfx950WidestThreads;
    return ncclSymkGfx950WideThreads;
  }

  // AllReduce folds rank into its thread index, so across the deep tiers a wider block halves
  // iterations per warp rather than covering more GPU. Outside them the wider block wins.
  bool narrowBlock = nBytes < ncclSymkArTailSaturatedBytes ||
                     (ncclSymkArDeepTierBytes <= nBytes && nBytes < ncclSymkArOccupancyBoundBytes);
  return narrowBlock ? ncclSymkGfx950NarrowThreads : ncclSymkGfx950WideThreads;
}
#endif

const char* ncclSymkKernelIdToString(int kernelId) {
  if (kernelId < 0 || kernelId >= ncclSymkKernelId_Count) {
    return "Unknown";
  }
  return ncclSymKernelStr[kernelId];
}

int ncclSymkMaxChunkElts(struct ncclComm* comm, ncclSymkKernelId kernelId, int /*ncclDevRedOp_t*/ red,
                         ncclDataType_t ty) {
  bool isReduce = 1 & ((kernelMask_Reduce | kernelMask_AR | kernelMask_RS) >> (int)kernelId);
  int eltSize = ncclTypeSize(ty);
  int accMult = !isReduce ? 1 : eltSize < 4 ? 2 : 1;
  bool isGenk = 1 & (kernelMask_Genk >> (int)kernelId);
  int kernelIndex = isGenk ? ncclGenkGetKernelIndex(kernelId, red, ty) : ncclSymkGetKernelIndex(kernelId, red, ty);
  if (kernelIndex < 0) return 0;
  int maxDynamicSmem = isGenk ? ncclGenkKernelMaxDynamicSmem[kernelIndex] : ncclSymkKernelMaxDynamicSmem[kernelIndex];
  return maxDynamicSmem / (eltSize * accMult);
}

/* this function fills in the devWork except nextWorkOffset */
ncclResult_t ncclSymkMakeDevWork(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclSymkDevWork* outDevWork) {
  outDevWork->rootRank = task->root;
  outDevWork->redOpArg = task->opDev.scalarArg;
  outDevWork->nElts = task->count;
  outDevWork->inputWin = task->sendWin ? task->sendWin->vidmem : nullptr;
  outDevWork->inputOff =
    task->sendWin ? (uint8_t*)task->sendbuff - (uint8_t*)task->sendWin->userPtr : (size_t)task->sendbuff;
  outDevWork->outputWin = task->recvWin ? task->recvWin->vidmem : nullptr;
  outDevWork->outputOff =
    task->recvWin ? (uint8_t*)task->recvbuff - (uint8_t*)task->recvWin->userPtr : (size_t)task->recvbuff;
  outDevWork->sChannelId = 0xffff;
  outDevWork->nChannels = 0;
  return ncclSuccess;
}

ncclResult_t ncclGetSymRegType(struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin,
                               ncclSymRegType_t* winRegType) {
  bool isSendSymmReg = false;
  bool isRecvSymmReg = false;
  if (sendWin && (sendWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) isSendSymmReg = true;
  if (recvWin && (recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) isRecvSymmReg = true;
  // determine the registration type
  if (!isSendSymmReg && !isRecvSymmReg) {
    *winRegType = ncclSymSendNonregRecvNonreg;
  } else if (isSendSymmReg && !isRecvSymmReg) {
    *winRegType = ncclSymSendRegRecvNonreg;
  } else if (!isSendSymmReg && isRecvSymmReg) {
    *winRegType = ncclSymSendNonregRecvReg;
  } else if (isSendSymmReg && isRecvSymmReg) {
    *winRegType = ncclSymSendRegRecvReg;
  }
  return ncclSuccess;
}

bool rcclSymkKernelIdIsLL(int kernelId) {
  if (kernelId < 0 || kernelId >= (int)ncclSymkKernelId_Count) return false;
  return (kernelMask_LL >> kernelId) & 1;
}

bool rcclSymkKernelIdIsTma(int kernelId) {
  if (kernelId < 0 || kernelId >= (int)ncclSymkKernelId_Count) return false;
  return (kernelMask_Tma >> kernelId) & 1;
}

#ifndef GENERATE_SYM_KERNELS
// [RCCL] When symmetric kernels aren't generated by generate.py we still
// need the symbols referenced by symmetric_sched.cc, enqueue.cc and the
// GIN requirement path so the link succeeds. These stubs make every
// kernel path return "no kernel available" -- the scheduler then falls
// back to the regular non-symmetric kernels.
void* ncclSymkGetKernelPtr(ncclSymkKernelId kernelId, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
  return nullptr;
}

extern int const ncclSymkKernelCount = 0;
void* ncclSymkKernelList[ncclSymkKernelId_Count] = {nullptr};
void* ncclSymkKernelListProfile[ncclSymkKernelId_Count] = {nullptr};
int ncclSymkKernelRequirements[ncclSymkKernelId_Count] = {0};
int ncclSymkKernelMaxDynamicSmem[ncclSymkKernelId_Count] = {0};
int ncclSymkKernelMaxThreads[ncclSymkKernelId_Count] = {0};

int ncclSymkGetKernelIndex(ncclSymkKernelId /*id*/, int /*red*/, ncclDataType_t /*ty*/) {
  return 0;
}
#endif

#ifndef SYM_GENK
// [RCCL] The Genk (general/flow) kernels are not built on AMD (device/general/generate.py does not run), so
// supply the empty kernel set. enqueue.cc skips a zero-count set and the scheduler treats index -1 as "no
// kernel".
extern int const ncclGenkKernelCount = 0;
void* ncclGenkKernelList[1] = {nullptr};
void* ncclGenkKernelListProfile[1] = {nullptr};
int ncclGenkKernelRequirements[1] = {0};
int ncclGenkKernelMaxDynamicSmem[1] = {0};

int ncclGenkGetKernelIndex(ncclSymkKernelId /*id*/, int /*red*/, ncclDataType_t /*ty*/) {
  return -1;
}
#endif
