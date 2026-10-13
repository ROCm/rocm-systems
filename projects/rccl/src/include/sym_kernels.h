/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_SYM_KERNELS_H_
#define NCCL_SYM_KERNELS_H_
#include "nccl.h"
#include "nccl_device.h"
#include "nccl_common.h"
#include "device.h"
#if !defined(NCCL_OS_WINDOWS)
#include "../device/symmetric/gin_scratch.h"
#else
#include "nccl_device/gin_win_stub.h"
#endif

////////////////////////////////////////////////////////////////////////////////
// ncclSymk[Foo]: Kernels built on the device API
// cuda::memory_order is provided on HIP by hip_compat.h, pulled in transitively via nccl_device.h above.
#define NCCL_MEM_ORDER_RELAXED cuda::memory_order_relaxed
#define NCCL_MEM_ORDER_RELEASE cuda::memory_order_release

#define NCCL_SYM_KERNEL_CELL_SIZE 1024 // no less than 16 bytes minimal cell size

constexpr int ncclSymkMaxBlocks = 64;
constexpr int ncclSymkMaxThreads = 256;
constexpr int ncclSymkLLMaxEltSize = 8;

// [RCCL] Launch width for the GIN and Tma symmetric kernels. The rest default to ncclSymkMaxThreads.
// This was tuned for TDM, so if you need to change it, please add a constexpr function that 
// selects an appropriate value for the architecture and method of transfer (TDM or non-TDM).
constexpr int ncclSymkWarpsPerBlock = 16;

// Widest LL block the gfx950 symmetric kernels launch, worth 5 to 10% at the sizes where it removes an
// epoch, since an LL epoch carries one element per thread. This bounds the shared slot buffer the
// host allocates; the kernels take their slot stride from blockDim.
constexpr int ncclSymkGfx950LLThreads = 512;

constexpr __host__ __device__ int ncclSymkLLMaxSlots(int eltSize = ncclSymkLLMaxEltSize) {
  return ncclSymkMaxThreads * ncclSymkLLMaxEltSize / eltSize;
}

enum ncclSymkKernelId {
  ncclSymkKernelId_AllReduce_AGxLL_R,
  ncclSymkKernelId_AllReduce_AGxLLMC_R,
  ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST,
  ncclSymkKernelId_AllReduce_RSxLD_AGxST,
  ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC,

  ncclSymkKernelId_AllGather_LL,
  ncclSymkKernelId_AllGather_LLMC,
  ncclSymkKernelId_AllGather_TmaST,
  ncclSymkKernelId_AllGather_ST,
  ncclSymkKernelId_AllGather_TmaSTMC,
  ncclSymkKernelId_AllGather_STMC,
  ncclSymkKernelId_AllGather_RailRing_LsaSTMC,

  ncclSymkKernelId_ReduceScatter_LL,
  ncclSymkKernelId_ReduceScatter_TmaLD,
  ncclSymkKernelId_ReduceScatter_LD,
  ncclSymkKernelId_ReduceScatter_LDMC,
  ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD,
  ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC,

  // [RCCL] NaN-flag kernels (NCCL_PROTO=NaN), appended so the upstream ids keep their values.
  ncclSymkKernelId_AllGather_NaN,
  ncclSymkKernelId_AllGather_TmaNaN,
  ncclSymkKernelId_AllReduce_AGxNaN_R,
  ncclSymkKernelId_AllReduce_AGxTmaNaN_R,
  ncclSymkKernelId_ReduceScatter_NaN,
  ncclSymkKernelId_ReduceScatter_TmaNaN,

  ncclSymkKernelId_Count
};

constexpr char const* ncclSymKernelStr[] = {
  // Must align with enum ncclSymkKernelId definition in src/include/sym_kernels.h
  "AllReduce_AGxLL_R",
  "AllReduce_AGxLLMC_R",
  "AllReduce_RSxTmaLD_AGxTmaST",
  "AllReduce_RSxLD_AGxST",
  "AllReduce_RSxLDMC_AGxSTMC",
  "AllGather_LL",
  "AllGather_LLMC",
  "AllGather_TmaST",
  "AllGather_ST",
  "AllGather_TmaSTMC",
  "AllGather_STMC",
  "AllGather_RailRing_LsaSTMC",
  "ReduceScatter_LL",
  "ReduceScatter_TmaLD",
  "ReduceScatter_LD",
  "ReduceScatter_LDMC",
  "ReduceScatter_RailA2A_LsaLD",
  "ReduceScatter_RailA2A_LsaLDMC",
  "AllGather_NaN",
  "AllGather_TmaNaN",
  "AllReduce_AGxNaN_R",
  "AllReduce_AGxTmaNaN_R",
  "ReduceScatter_NaN",
  "ReduceScatter_TmaNaN"
};
static_assert(sizeof(ncclSymKernelStr) / sizeof(ncclSymKernelStr[0]) == ncclSymkKernelId_Count,
              "ncclSymKernelStr needs one name per ncclSymkKernelId");

// [RCCL] Scratch of the NaN-flag kernels (src/device/symmetric/nan.cuh). A header line per block
// carries the block's bank across launches, then each block owns a region of banks that rotate by
// epoch. A bank holds a slot per source rank of ncclSymkNanEpochUnits 16 B units.
constexpr int ncclSymkNanMaxRanks = 8;
constexpr int ncclSymkNanBanks = 3;
constexpr int ncclSymkNanThreads = 256;
// Units each thread of the register path moves per source per epoch.
constexpr int ncclSymkNanUnroll = 2;
constexpr int ncclSymkNanEpochUnits = ncclSymkNanThreads * ncclSymkNanUnroll;
// Above ncclSymkMaxBlocks: the large calls keep scaling with the grid, as the NaN DDA kernels do.
constexpr int ncclSymkNanMaxBlocks = 256;
constexpr size_t ncclSymkNanHeaderBytes = ncclSymkNanMaxBlocks * 16;
constexpr size_t ncclSymkNanBlockBytes = size_t(ncclSymkNanBanks) * ncclSymkNanMaxRanks * ncclSymkNanEpochUnits * 16;
constexpr size_t ncclSymkNanScratchBytes = ncclSymkNanHeaderBytes + ncclSymkNanMaxBlocks * ncclSymkNanBlockBytes;
static_assert(ncclSymkNanHeaderBytes % 128 == 0, "the regions start on a line");
// AllReduce calls of more bytes take the two-shot (a reduce-scatter epoch, then an all-gather
// epoch) instead of pushing the whole buffer to every rank.
constexpr size_t ncclSymkNanTwoShotMinBytes = size_t(128) << 10;
// A team of 4 streams a block part of more than one epoch through the register path, except that
// the TDM kernels keep the TDM path for parts of up to this many epochs.
constexpr int ncclSymkNanTdmMaxEpochs4 = 4;

struct ncclSymkDevComm {
  struct ncclDevComm devComm;
  struct ncclLLA2AHandle lsaLLA2A;
  struct ncclGinOutboxHandle ginOutbox;
  struct ncclGinInboxA2AHandle ginInboxRail;
  struct ncclGinSyncHandle ginSyncHandle;
  ncclDevResourceHandle rsGinAccumBuf;
  uint32_t rsGinAccumBytesPerBlock;
  // Allocated only under NCCL_PROTO=NaN; no NaN kernel is selected otherwise.
  ncclDevResourceHandle nanBuf;
  // Profiler counters (host-pinned), indexed by channel id and per-channel slot.
  // workPhases holds per-phase timestamps published behind a fence.
  struct ncclDevProfiler* workStarted;
  struct ncclDevProfiler* workCompleted;
  struct ncclDevProfilerPhases* workPhases;
};

struct ncclSymkState {
  bool initialized;
  bool hasLsaMultimem;
  bool hasNanScratch;
  int maxGinInboxBlocks;
  struct ncclSymkDevComm kcomm;
};

struct ncclSymkChannelWorkRange {
  uint16_t workHi; // inclusive index of my ending work
  uint16_t fracHi; // 16-bit fraction in (0.0, 1.0] indicating where my part ends
};

// 16 bytes aligned
struct alignas(16) ncclSymkDevWork {
  uint64_t redOpArg; // must be collectively uniform
  size_t nElts;
  struct ncclWindow_vidmem *inputWin, *outputWin;
  size_t inputOff, outputOff; // these = origUserOffset + cbdPartOffset
  int rootRank;
  uint64_t sChannelId:16, nChannels:16, padding:32;
};

// Device-side profiling requested for a launch; KernelPhase implies KernelCh. Only the
// symmetric kernels stamp phases, and they read these bits at runtime.
enum ncclDevProfilerMode : uint8_t {
  ncclDevProfilerModeNone = 0,
  ncclDevProfilerModeKernelCh = 1 << 0,
  ncclDevProfilerModeKernelPhase = 1 << 1,
};

struct alignas(16) ncclSymkDevWorkArgs {
  struct ncclSymkDevComm kcomm;
  int nMaxChannels;
  int maxDynamicSmem;
  // int, not uint8_t: masking a narrower field in the symmetric-kernel prologue misaligns their hot loops.
  int profilerMode; // ncclDevProfilerMode bits; nonzero means profilerWorkCounters[nMaxChannels] follows
  // Variable-length trailing data layout:
  //   if profilerMode: uint64_t profilerWorkCounters[nMaxChannels] (aligned to 16)
  //   ncclSymkChannelWorkRange[nChannels] (aligned to 16)
  //   ncclSymkDevWork[nWorks]
  // aux functions
  __host__ static constexpr size_t calcArgsSize(int nChannels, int nWorks, bool profiler = false) {
    return alignUp(sizeof(struct ncclSymkDevWorkArgs), 16) +
           (profiler ? alignUp(nChannels * sizeof(uint64_t), 16) : size_t(0)) +
           alignUp(nChannels * sizeof(struct ncclSymkChannelWorkRange), 16) + nWorks * sizeof(struct ncclSymkDevWork);
  }
  __host__ __device__ uint64_t* getProfilerCounters() const {
    return (uint64_t*)((uint8_t*)this + alignUp(sizeof(struct ncclSymkDevWorkArgs), 16));
  }
  __host__ __device__ struct ncclSymkChannelWorkRange* getWorkRange() const {
    size_t off = alignUp(sizeof(struct ncclSymkDevWorkArgs), 16);
    if (profilerMode) off += alignUp(nMaxChannels * sizeof(uint64_t), 16);
    return (struct ncclSymkChannelWorkRange*)((uint8_t*)this + off);
  }
  __host__ __device__ struct ncclSymkDevWork* getWorks(int nChannels) const {
    return (struct ncclSymkDevWork*)((uint8_t*)this->getWorkRange() +
                                     alignUp(nChannels * sizeof(struct ncclSymkChannelWorkRange), 16));
  }
};

union ncclSymkDevWorkArgs4K {
  struct ncclSymkDevWorkArgs args;
  char buf4K[4096];
};

typedef enum {
  ncclSymSendNonregRecvNonreg = 0,
  ncclSymSendNonregRecvReg = 1,
  ncclSymSendRegRecvNonreg = 2,
  ncclSymSendRegRecvReg = 3,
  ncclNumSymRegTypes = 4
} ncclSymRegType_t;

// We assume ncclComm contains a field: `ncclSymkState symkState`
ncclResult_t ncclSymkInitOnce(struct ncclComm* comm);
ncclResult_t ncclSymkFinalize(struct ncclComm* comm);

bool ncclSymkAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                       size_t nElts);
uint32_t ncclSymkMask(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                      size_t nElts, bool symAligned16B = true);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
bool ncclSymkIsGfx950(struct ncclComm* comm);
#endif

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// Block width the gfx950 symmetric kernels launch at, split out of the symmetric tuning model so the
// size bands can be unit tested. Returns ncclSymkMaxThreads for any collective without width bands.
int ncclSymkGfx950BlockThreads(ncclFunc_t coll, bool isLL, int nRanks, size_t nBytes);
// Whether AllGather should take the store kernel over LL on gfx950, where the shared cost model
// keeps LL well past the point the store kernel overtakes it.
bool ncclSymkGfx950AllGatherPrefersStore(int nRanks, size_t nBytes);
#endif

ncclResult_t ncclSymkMakeDevWork(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclSymkDevWork* outDevWork);
bool ncclSymkTmaAvailable(struct ncclComm* comm);
// NCCL_SYM_TMA_ENABLE=2: take the DMA-staged kernel regardless of predicted time or message size.
bool ncclSymkTmaForced(struct ncclComm* comm);
bool ncclSymkTmaDeepEligible(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, int nBlocks);
// [RCCL] Whether forced NaN hands this call to the Simple Tma kernels (RCCL_SYM_NAN_SIMPLE_TMA).
bool rcclSymkNanYieldsToTma(struct ncclComm* comm, ncclFunc_t coll, ncclDataType_t ty, size_t nElts);

// Generated by src/device/symmetric/generate.py
extern int const ncclSymkKernelCount;
extern void* ncclSymkKernelList[/*ncclSymkKernelCount*/];
// Instrumented variants, indexed identically to ncclSymkKernelList. Selected
// (instead of ncclSymkKernelList) only when kernel-channel profiling is active.
extern void* ncclSymkKernelListProfile[/*ncclSymkKernelCount*/];
extern int ncclSymkKernelRequirements[/*ncclSymkKernelCount*/];
extern int ncclSymkKernelMaxDynamicSmem[/*ncclSymkKernelCount*/]; // initialized by ncclInitKernelsForDevice()
int ncclSymkGetKernelIndex(ncclSymkKernelId kernelId, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty);
const char* ncclSymkKernelIdToString(int kernelId);
ncclResult_t ncclGetSymRegType(struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin,
                               ncclSymRegType_t* winRegType);
bool rcclSymkKernelIdIsLL(int kernelId);

int ncclSymkLLKernelMask();
int ncclSymkDynamicSmemKernelMask();
int ncclSymkTmaKernelMask();
int ncclSymkGinKernelMask();
int ncclSymkLsaKernelMask();
int ncclSymkAGKernelMask();
int ncclSymkARKernelMask();
int ncclSymkRSKernelMask();
int ncclSymkNanKernelMask();
size_t ncclSymkRsGinChunkBytes();

constexpr int ncclSymkAllGather_RailRing_ChunkSize = 1 << 20;

constexpr int ncclSymkMinWarpsPerBlock = 4;
constexpr int ncclSymkBytePerPack = 16;
constexpr __host__ __device__ int ncclSymkGetBytesPerChunk(int nWarps, int unrollPacks) {
  return nWarps * unrollPacks * WARP_SIZE * ncclSymkBytePerPack;
}

// SM kernel unroll packs
constexpr int ncclSymkUnrollPacks = 4;
constexpr int ncclSymkBytePerChunk = ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, ncclSymkUnrollPacks);

// TMA kernel unroll packs, by element size.
//
// [RCCL] The reduce deep loops carry acc0[UnrollPacks] of T plus acc1[UnrollPacks] of the
// accumulator, and AccPack is BytePack<ncclSymkBytePerPack * sizeof(Acc) / sizeof(T)> with
// Acc always float -- so a narrower T makes each accumulator pack WIDER: 32 B/pack at f32,
// 48 B at 2-byte types, 80 B at 1-byte. A single pack count therefore cannot suit all of
// them; at 16 packs everything below f32 spills the register file. Scaling the count the
// other way holds the per-lane accumulator footprint roughly flat (~512 B), so each type
// gets the widest tile it can actually hold.
constexpr __host__ __device__ int ncclSymkDeepUnrollPacks(int eltSize) {
  return eltSize >= 4 ? 16 : eltSize == 2 ? 8 : 4;
}
constexpr __host__ __device__ int ncclSymkDeepBytePerChunk(int eltSize) {
  return ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, ncclSymkDeepUnrollPacks(eltSize));
}
// [RCCL] ncclSymkTmaDeepEligible() has no datatype, so it bars on the widest chunk any
// element size produces. That is the safe direction: a narrower type's chunk is smaller,
// so it needs fewer bytes to fill one, and the picker only ever asks for more than
// necessary. The deep loops below still size themselves from their own sizeof(T).
constexpr int ncclSymkDeepMaxUnrollPacks = ncclSymkDeepUnrollPacks(4);
constexpr int ncclSymkDeepMaxBytePerChunk = ncclSymkDeepBytePerChunk(4);

// Multimem bcast deep loop (single warp). A pure relay with no accumulators, so it takes
// the widest tile regardless of type.
constexpr int ncclSymkMultimemDeepBytePerChunk = ncclSymkGetBytesPerChunk(1, ncclSymkDeepMaxUnrollPacks);

// Spread concurrent MC operations across different addresses to avoid contention.
constexpr int ncclSymkMcPerRankOffsetBytes = 32 * 1024 * 1024;
static_assert(ncclSymkMcPerRankOffsetBytes % ncclSymkMultimemDeepBytePerChunk == 0);

// Deep loop when input/output are 256 B-aligned. AllGather stages a tile it never reduces,
// so it carries no accumulator and can take double the packs without register pressure.
constexpr int ncclSymkAlign256BDeepUnrollPacks = 32;
constexpr int ncclSymkAlign256BDeepBytePerChunk =
  ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, ncclSymkAlign256BDeepUnrollPacks);

// [RCCL] Core asserts for TDM-powered kernels.  Do not remove the static asserts.
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
constexpr int ncclSymkTileSmemBudget = 320 << 10; // gfx1250 has 320KiB of LDS
static_assert(ncclSymkWarpsPerBlock * ncclTmaShmemScratchWarpSize() <= ncclSymkTileSmemBudget,
              "async-tile staging windows do not fit gfx1250's per-block LDS budget");
static_assert(ncclSymkWarpsPerBlock == 16, "TDM symmetric kernels expect a 16-warp launch");
#endif

#endif
