// Modification Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_
#define NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_

#include "sym_kernels.h"
#include "bitops.h"
#include "collectives.h"
#include "../op128.h"
#include "../reduce_kernel.h"
#if !defined(NCCL_OS_WINDOWS)
#include "gin_scratch.h"
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13010
#include <cuda/std/ranges>
#endif

////////////////////////////////////////////////////////////////////////////////
// Async tile staging: NVIDIA TMA and AMD gfx1250 TDM
//
// The deep loops in all_gather/all_reduce/reduce_scatter can move a tile through
// the warp's shared-memory scratch with a DMA engine instead of per-lane vector
// loads. NVIDIA drives that with TMA, gfx1250 with the Tensor Data Mover. The two
// engines disagree on who issues a transfer and on how completion is observed:
//
//                 NVIDIA (sm_100+)               AMD gfx1250
//   issue         one thread; the descriptor     the whole warp; the descriptor
//                 is built per-thread            lives in SGPRs, so the
//                                                arguments must be wave-uniform
//   completion    a shared-memory mbarrier       the per-wave TENSORcnt, so only
//                 carrying a byte count, which   the warp that issued a transfer
//                 any thread may wait on         can drain it, and no shared
//                                                object is involved
//
// The ncclSymkTile* wrappers below absorb that difference. Every one of them is
// called by the WHOLE warp with wave-uniform arguments and narrows to a single
// thread internally where the hardware wants that, so the loops keep one shape.
// The `tma` names below (tmaSmemStruct, EnableTma, ...) now denote the async-tile
// path on either vendor; they match the ncclSymkKernelId_*Tma* kernel names.
////////////////////////////////////////////////////////////////////////////////
#if defined(__HIP_PLATFORM_AMD__)
#include "tdm/tdmCopy.h"
// TDM_SUPPORTED is a device-pass constant: gfx1250, a compiler carrying the
// builtin, and an SDK shipping the descriptor header. Every other arch, and the
// host pass, keep the vector path.
#define NCCL_SYMK_TILE_TMA 0
#define NCCL_SYMK_TILE_TDM TDM_SUPPORTED
#else
#define NCCL_SYMK_TILE_TMA (__CUDA_ARCH__ >= 1000)
#define NCCL_SYMK_TILE_TDM 0
#endif
#define NCCL_SYMK_ASYNC_TILE (NCCL_SYMK_TILE_TMA || NCCL_SYMK_TILE_TDM)

#if NCCL_SYMK_ASYNC_TILE
// The tile wrappers are keyed on a CachePolicy and their signatures are common to both
// engines, so the encoding has to be visible even on the TMA path, which has no use for
// it. tdmCopy.h already pulls this in on the TDM side; the header guard absorbs that.
#include "tdm/cachePolicy.h"

// Only TDM acts on these: they become the immediate cpol operand on its tensor
// instructions. TMA has no equivalent knob and discards the argument.
//
// A tile whose other end is a peer has to be visible off this GPU, so it is streamed
// at system scope and kept out of the local caches. Matches kRcclTdmPolicy on the
// SIMPLE TDM path.
static constexpr CachePolicy kNcclSymkTilePeerPolicy = createCachePolicy(TemporalHint::NT, MemScope::SYS);
// A tile that never leaves this GPU only has to reach device coherence, so it can
// stop at L2 rather than paying for the system-scope round trip.
static constexpr CachePolicy kNcclSymkTileLocalPolicy = createCachePolicy(TemporalHint::NT, MemScope::DEV);

#if NCCL_SYMK_TILE_TMA
#include <cuda/barrier>
#include <cuda/ptx>

namespace ptx = cuda::ptx;

using ncclSymkTileBar = cuda::barrier<cuda::thread_scope_block>;
#else
// TDM completion is a per-wave counter, so gfx1250 needs no shared-memory
// barrier. The empty type keeps tmaSmemStruct's layout common to both paths.
struct ncclSymkTileBar {};
#endif

template <typename Pack, int UnrollPacks, int UnrollPeers = 1>
struct tmaSmemStruct {
  alignas(16) Pack buff[UnrollPeers][UnrollPacks * WARP_SIZE];
  ncclSymkTileBar bar;
};

// Arm the staging barrier. `arrivers` is how many lanes will arrive on it, which
// differs by loop: the all-gather deep loop stages from lane 0 alone, the reduce
// loops stage across the whole warp. TDM has no barrier to arm.
static __device__ __forceinline__ void ncclSymkTileBarInit(ncclSymkTileBar* bar, int arrivers, int lane) {
#if NCCL_SYMK_TILE_TMA
  if (lane == 0) init(bar, arrivers);
#else
  (void)bar, (void)arrivers, (void)lane;
#endif
}

// Start a global -> shared transfer of `bytes` into the warp's tile, leaving it in
// flight. `pending` accumulates the bytes the next barrier arrival has to account
// for; TDM counts its own transfers, so it is left alone there. `cp` says how far the
// transfer has to be made visible: pass kNcclSymkTileLocalPolicy only when `global` is
// this GPU's own memory, kNcclSymkTilePeerPolicy otherwise.
template <CachePolicy cp>
static __device__ __forceinline__ void ncclSymkTileLoad(void* smem, void const* global, size_t bytes,
                                                        ncclSymkTileBar& bar, size_t& pending, int lane) {
#if NCCL_SYMK_TILE_TMA
  if (lane == 0) {
    cuda::device::memcpy_async_tx((char*)smem, (char const*)global, cuda::aligned_size_t<16>(bytes), bar);
    pending += bytes;
  }
#else
  (void)bar, (void)pending, (void)lane;
  tdm::asyncLoadToLDS<SyncPolicy::Async, cp>((uint8_t const*)global, (uint8_t*)smem, bytes);
#endif
}

// Wait for the loads this warp started to land in shared memory. `Arrivers` must
// match what ncclSymkTileBarInit() armed the barrier with.
template <int Arrivers>
static __device__ __forceinline__ void ncclSymkTileLoadWait(ncclSymkTileBar& bar, size_t& pending, int lane) {
#if NCCL_SYMK_TILE_TMA
  if (Arrivers != 1 || lane == 0) {
    ncclSymkTileBar::arrival_token token = cuda::device::barrier_arrive_tx(bar, 1, pending);
    bar.wait(std::move(token));
  }
  pending = 0;
#else
  (void)bar, (void)pending, (void)lane;
  tdm::tdmWait();
#endif
}

// Start a shared -> global transfer of `bytes` out of the warp's tile, leaving it
// in flight. Several of these may be issued back to back (one per peer) before a
// single ncclSymkTileStoreWait(); they all read the tile, so they cannot conflict.
// `cp` carries the same meaning as on ncclSymkTileLoad(), here about `global` as the
// destination.
template <CachePolicy cp>
static __device__ __forceinline__ void ncclSymkTileStore(void* global, void const* smem, size_t bytes, int lane) {
#if NCCL_SYMK_TILE_TMA
  if (lane == 0) ptx::cp_async_bulk(ptx::space_global, ptx::space_shared, global, smem, bytes);
#else
  (void)lane;
  tdm::asyncStoreFromLDS<SyncPolicy::Async, cp>((uint8_t const*)smem, (uint8_t*)global, bytes);
#endif
}

// Drain the stores this warp started, so its tile can be refilled.
static __device__ __forceinline__ void ncclSymkTileStoreWait(int lane) {
#if NCCL_SYMK_TILE_TMA
  if (lane == 0) {
    ptx::cp_async_bulk_commit_group();
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
#else
  (void)lane;
  tdm::tdmWait();
#endif
}

// Publish the lanes' ordinary shared-memory writes to the DMA engine that is about
// to read them out of the tile.
static __device__ __forceinline__ void ncclSymkTileFenceSmem() {
#if NCCL_SYMK_TILE_TMA
  ptx::fence_proxy_async(ptx::space_shared);
#else
  __threadfence_block();
#endif
  __syncwarp();
}
#endif // NCCL_SYMK_ASYNC_TILE

// HIP has no __isShared() (used only as a __builtin_assume hint); map it to the AMDGCN builtin.
#if defined(__HIP_PLATFORM_AMD__) && !defined(NCCL_SYMK_HAVE_ISSHARED)
#define NCCL_SYMK_HAVE_ISSHARED 1
static __device__ __forceinline__ bool __isShared(const void* p) {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
  return __builtin_amdgcn_is_shared((void*)p);
#else
  // __builtin_amdgcn_is_shared is a device-only intrinsic; host pass only needs a stub for the __builtin_assume hint.
  (void)p;
  return true;
#endif
}
#endif

#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif

template <bool val>
struct BoolTag {
  static constexpr bool value = val;
};

// A cheap approximation of std::decay
template <typename T>
struct ncclDecayType {
  using Type = T;
};
template <typename T>
struct ncclDecayType<T&> {
  using Type = T;
};
template <typename T>
struct ncclDecayType<T&&> {
  using Type = T;
};
template <typename T>
struct ncclDecayType<T const> {
  using Type = T;
};
template <typename T>
struct ncclDecayType<T volatile> {
  using Type = T;
};

template <typename T>
using ncclDecayType_t = typename ncclDecayType<T>::Type;

// flattenIx(pos0, dim0, pos1, dim1, pos2, dim2, ...)
// Given a position vector `pos` in a rectangular index space with lengths in the `dim`
// vector, flatten that down to a linear index. The fastest moving dimension is given first.
__device__ __forceinline__ int flattenIx() {
  return 0;
}

template <typename Int0, typename Int1, typename... Ints>
static __device__ Int0 flattenIx(Int0 pos, Int1 size, Ints... more) {
  return pos + size * flattenIx(more...);
}

template <typename T>
static __device__ void partitionElts(unsigned nParts, unsigned part, size_t* nElts, ncclSymPtr<T>* inPtr,
                                     ncclSymPtr<T>* outPtr) {
  constexpr int eltPerB16 = 16 / sizeof(T);
  size_t nB16 = (*nElts + eltPerB16 - 1) / eltPerB16;
  size_t beginB16 = part * (nB16 / nParts) + min(part, uint32_t(nB16 % nParts));
  *inPtr += beginB16 * eltPerB16;
  *outPtr += beginB16 * eltPerB16;
  if (part < nParts - 1) {
    nB16 = nB16 / nParts + (part < nB16 % nParts ? 1 : 0);
    *nElts = nB16 * eltPerB16;
  } else {
    *nElts = *nElts - beginB16 * eltPerB16;
  }
}

namespace {
struct ncclSymkArgsHandler {
  ncclDevComm const& comm;
  ncclLLA2AHandle const& lsaLLA2A;
  ncclGinOutboxHandle const& ginOutbox;
  ncclGinInboxA2AHandle const& ginInboxRail;
  ncclGinSyncHandle const& ginSyncHandle;
  ncclDevResourceHandle rsGinAccumBuf;
  uint32_t rsGinAccumBytesPerBlock;
  struct ncclSymkChannelWorkRange* channelWorkRange;
  struct ncclSymkDevWork* devWork;
  uint32_t nRanks_rcp32;

  __device__ ncclSymkArgsHandler(ncclSymkDevWorkArgs const* args)
    : comm(args->kcomm.devComm), lsaLLA2A(args->kcomm.lsaLLA2A), ginOutbox(args->kcomm.ginOutbox),
      ginInboxRail(args->kcomm.ginInboxRail), ginSyncHandle(args->kcomm.ginSyncHandle),
      rsGinAccumBuf(args->kcomm.rsGinAccumBuf), rsGinAccumBytesPerBlock(args->kcomm.rsGinAccumBytesPerBlock) {
    channelWorkRange = args->getWorkRange();

    devWork = args->getWorks(args->nMaxChannels);
    nRanks_rcp32 = comm.nRanks_rcp32;
  }

  template <typename T>
  __device__ void getWorkRange(int block, uint16_t& workLo, size_t& indexLo, uint16_t& workHi, size_t& indexHi) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    uint32_t fracLo, fracHi;

    // Where the work begins
    workLo = (block == 0) ? 0 : channelWorkRange[block - 1].workHi; // start where predecessor ends
    fracLo = (block == 0) ? 0 : channelWorkRange[block - 1].fracHi + 1;
    // If the predecessor ended on the work boundary, then we step to the beginning of the next work.
    // This ensures we never have empty parts.
    if (fracLo == 0x10000) {
      workLo++;
      fracLo = 0;
    }
    struct ncclSymkDevWork const& dwLo = devWork[workLo];
    indexLo = ((fracLo * divUp(dwLo.nElts, EltPerCell)) >> 16) * EltPerCell;

    // Where the work ends
    workHi = channelWorkRange[block].workHi;
    fracHi = channelWorkRange[block].fracHi + 1;
    struct ncclSymkDevWork const& dwHi = devWork[workHi];
    indexHi = min(((fracHi * divUp(dwHi.nElts, EltPerCell)) >> 16) * EltPerCell, dwHi.nElts);
  }

  template <typename T>
  __device__ void getWorkRangeFused(int blockIdx, int w, int& block, int& nBlocks, size_t& indexLo, size_t& indexHi) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    struct ncclSymkDevWork const& dw = devWork[w];
    uint32_t fracLo, fracHi;
    int lastBlock;

    block = blockIdx - dw.sChannelId;
    nBlocks = dw.nChannels;
    lastBlock = dw.sChannelId + dw.nChannels - 1;

    // Where the work begins
    fracLo = (dw.sChannelId > 0 && channelWorkRange[dw.sChannelId - 1].workHi == w) ?
               ((channelWorkRange[dw.sChannelId - 1].fracHi + 1) & 0xFFFF) :
               0;
    indexLo = ((fracLo * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell;
    fracHi = (channelWorkRange[lastBlock].workHi == w) ? channelWorkRange[lastBlock].fracHi + 1 : 0x10000;
    indexHi = min(((fracHi * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell, dw.nElts);
  }

  template <typename T, typename Fn>
  __device__ void forEachWork(Fn const& fn) {
    uint16_t workLo, workHi;
    size_t indexLo, indexHi;

    getWorkRange<T>(blockIdx.x, workLo, indexLo, workHi, indexHi);

    NVCC_PRAGMA_UNROLL_DISABLED
    for (int w = workLo; w <= workHi; w++) {
      struct ncclSymkDevWork const& dw = devWork[w];
      size_t const& nAllElts = dw.nElts;
      size_t currentIndexLo, currentIndexHi;
      int block, nBlocks;
      if (blockIdx.x >= dw.sChannelId && blockIdx.x < dw.sChannelId + dw.nChannels) {
        getWorkRangeFused<T>(blockIdx.x, w, block, nBlocks, currentIndexLo, currentIndexHi);
      } else {
        currentIndexLo = (w > workLo) ? 0 : indexLo;
        currentIndexHi = (w < workHi) ? nAllElts : indexHi;
        block = 0;
        nBlocks = 1;
      }

      fn(block, nBlocks, currentIndexHi - currentIndexLo, nAllElts,
         ncclSymPtr<T>(dw.inputWin, dw.inputOff) + currentIndexLo,
         ncclSymPtr<T>(dw.outputWin, dw.outputOff) + currentIndexLo);

      currentIndexLo = 0;
    }
  }

  template <typename T, typename Fn>
  __device__ void singleWork(Fn const& fn) {
    uint16_t w;
    size_t indexLo, indexHi;

    getWorkRange<T>(blockIdx.x, w, indexLo, w, indexHi);

    struct ncclSymkDevWork const& dw = devWork[w];

    fn(indexHi - indexLo, dw.nElts, ncclSymPtr<T>(dw.inputWin, dw.inputOff) + indexLo,
       ncclSymPtr<T>(dw.outputWin, dw.outputOff) + indexLo);
  }

  template <typename T, typename Fn>
  __device__ void forEachWorkNoFusion(Fn const& fn) {
    uint16_t workLo, workHi;
    size_t indexLo, indexHi;

    getWorkRange<T>(blockIdx.x, workLo, indexLo, workHi, indexHi);

    NVCC_PRAGMA_UNROLL_DISABLED
    for (int w = workLo; w <= workHi; w++) {
      struct ncclSymkDevWork const& dw = devWork[w];
      size_t const& nAllElts = dw.nElts;
      size_t currentIndexLo, currentIndexHi;
      currentIndexLo = (w > workLo) ? 0 : indexLo;
      currentIndexHi = (w < workHi) ? nAllElts : indexHi;

      fn(currentIndexHi - currentIndexLo, nAllElts, ncclSymPtr<T>(dw.inputWin, dw.inputOff) + currentIndexLo,
         ncclSymPtr<T>(dw.outputWin, dw.outputOff) + currentIndexLo);
    }
  }
};
} // namespace

template <template <typename> typename Red, typename T, bool nvls>
struct ncclSymkAccumType {
  using Type = T;
};

// Only Red's whose opArg is invariant w.r.t. the datatype can have a different
// accumulator type. At the moment this excludes integer min/max, sumpostdiv,
// and premulsum.
template <>
struct ncclSymkAccumType<FuncSum, __half, false> {
  using Type = float;
};
template <>
struct ncclSymkAccumType<FuncSumPostDiv, __half, false> {
  using Type = float;
};
#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
struct ncclSymkAccumType<FuncSum, __nv_bfloat16, false> {
  using Type = float;
};
template <>
struct ncclSymkAccumType<FuncSumPostDiv, __nv_bfloat16, false> {
  using Type = float;
};
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
template <>
struct ncclSymkAccumType<FuncSum, __nv_fp8_e4m3, false> {
  using Type = float;
};
template <>
struct ncclSymkAccumType<FuncSum, __nv_fp8_e5m2, false> {
  using Type = float;
};
template <>
struct ncclSymkAccumType<FuncSumPostDiv, __nv_fp8_e4m3, false> {
  using Type = __half;
};
template <>
struct ncclSymkAccumType<FuncSumPostDiv, __nv_fp8_e5m2, false> {
  using Type = __half;
};
#endif
#if defined(__HIP_PLATFORM_AMD__)
// [RCCL] The upstream bf16/fp8 specializations above are gated behind
// __CUDA_{BF16,FP8}_TYPES_EXIST__ and keyed on CUDA type names; re-key them on the
// RCCL device types. bf16 and fp8 both reduce in a float accumulator: the fp8
// software type used on non-fp8 arches (e.g. gfx908) has no __half conversion, so
// fp8 accumulates in float (which casts cleanly on every arch).
template <>
struct ncclSymkAccumType<FuncSum, hip_bfloat16, false> {
  using Type = float;
};
template <>
struct ncclSymkAccumType<FuncSum, rccl_float8, false> {
  using Type = float;
};
template <>
struct ncclSymkAccumType<FuncSum, rccl_bfloat8, false> {
  using Type = float;
};
template <>
struct ncclSymkAccumType<FuncSumPostDiv, hip_bfloat16, false> {
  using Type = float;
};
template <>
struct ncclSymkAccumType<FuncSumPostDiv, rccl_float8, false> {
  using Type = float;
};
template <>
struct ncclSymkAccumType<FuncSumPostDiv, rccl_bfloat8, false> {
  using Type = float;
};
#endif

// Accumulator type held in smem for GIN algos.
template <template <typename> typename Red, typename T>
struct ncclSymkGinAccumType {
  using Type = T;
};

template <>
struct ncclSymkGinAccumType<FuncSum, __half> {
  using Type = float;
};
template <>
struct ncclSymkGinAccumType<FuncSumPostDiv, __half> {
  using Type = float;
};
#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
struct ncclSymkGinAccumType<FuncSum, __nv_bfloat16> {
  using Type = float;
};
template <>
struct ncclSymkGinAccumType<FuncSumPostDiv, __nv_bfloat16> {
  using Type = float;
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
// fp8 types accumulate in fp16. Multimem algo sends fp8 on wire because it's
// impossible to get fp16 accumulator from switch. Non-multimem sends fp16 to
// give users a higher precision alternative.
template <>
struct ncclSymkGinAccumType<FuncSum, __nv_fp8_e4m3> {
  using Type = __half;
};
template <>
struct ncclSymkGinAccumType<FuncSum, __nv_fp8_e5m2> {
  using Type = __half;
};
template <>
struct ncclSymkGinAccumType<FuncSumPostDiv, __nv_fp8_e4m3> {
  using Type = __half;
};
template <>
struct ncclSymkGinAccumType<FuncSumPostDiv, __nv_fp8_e5m2> {
  using Type = __half;
};
#endif

#if defined(__HIP_PLATFORM_AMD__)
// [RCCL] avg reduces in float on ROCm: raw bf16/fp8 have no FuncSumPostDiv, but FuncSumPostDiv<float> does.
template <>
struct ncclSymkGinAccumType<FuncSumPostDiv, hip_bfloat16> {
  using Type = float;
};
template <>
struct ncclSymkGinAccumType<FuncSumPostDiv, rccl_float8> {
  using Type = float;
};
template <>
struct ncclSymkGinAccumType<FuncSumPostDiv, rccl_bfloat8> {
  using Type = float;
};
#endif

#if NCCL_SYMK_ASYNC_TILE
// Round-trip one tile global -> shared -> global. Called by the whole warp with
// wave-uniform arguments. `source` is this rank's own input; `dest` is a multimem
// address that fans the tile out to every peer.
static __device__ __forceinline__ void tmaLoadStoreMc(char* dest, char* smem, char const* source, size_t size,
                                                      ncclSymkTileBar& bar, int lane) {
  size_t pending = 0;
  ncclSymkTileLoad<kNcclSymkTileLocalPolicy>(smem, source, size, bar, pending, lane);
  ncclSymkTileLoadWait</*Arrivers=*/1>(bar, pending, lane);
  ncclSymkTileStore<kNcclSymkTilePeerPolicy>(dest, smem, size, lane);
  ncclSymkTileStoreWait(lane);
}
#endif

// TODO: move this into data_ops.cuh
template <typename T, bool EnableTma = false>
static __device__ void bcastMultimem(ncclSymkArgsHandler& handler, int tn, int t, ncclSymPtr<T> input,
                                     ncclSymPtr<T> output, size_t nElts) {
  size_t nBytes = nElts * sizeof(T);
  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input.localPtr());
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output.multimemPtr(handler.comm.lsaMultimem));
  uint32_t alignment = uint32_t(inputUptr - outputUptr);
  uint32_t nPreBytes =
#if NCCL_SYMK_ASYNC_TILE
    (EnableTma && alignment % 256 == 0) ? (256 - input.offset) % 256 :
#endif
                                          (16 - input.offset) % 16;

  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t nSufBytes;

#if NCCL_SYMK_ASYNC_TILE
  int lane = t % WARP_SIZE;
  int lw = threadIdx.x / WARP_SIZE;
  extern __shared__ char smemScratch[];
#endif

  if (alignment % 16 == 0) {
    constexpr int BytePerPack = ncclSymkBytePerPack, UnrollPacks = ncclSymkDeepUnrollPacks;
    constexpr int BytePerChunk = ncclSymkMultimemDeepBytePerChunk;
    uintptr_t cursor = nPreBytes;
    uint32_t nChunks = (nBytes - cursor) / BytePerChunk;
    uintptr_t cursorAfter = cursor + uintptr_t(nChunks) * BytePerChunk;

#if NCCL_SYMK_ASYNC_TILE
    // Initialize share memory pointer and barrier
    constexpr size_t tileSize = UnrollPacks * WARP_SIZE * BytePerPack;
    using tmaSmemStruct_t = tmaSmemStruct<BytePack<BytePerPack>, UnrollPacks>;
    constexpr int smemSizePerWarp = ncclTmaShmemScratchWarpSize();
    tmaSmemStruct_t* tmaSmem = reinterpret_cast<tmaSmemStruct_t*>(smemScratch + lw * smemSizePerWarp);
    if NCCL_IF_CONSTEXPR (EnableTma) {
      ncclSymkTileBarInit(&tmaSmem->bar, /*arrivers=*/1, lane);
    }
#endif

    nSufBytes = nBytes - cursorAfter;
    cursor += (t / WARP_SIZE) * UnrollPacks * WARP_SIZE * BytePerPack;
    cursor += (t % WARP_SIZE) * BytePerPack;
    int nIters = nChunks - t / WARP_SIZE;
    NVCC_PRAGMA_UNROLL_DISABLED
    while (0 < nIters) {
#if NCCL_SYMK_ASYNC_TILE
      if NCCL_IF_CONSTEXPR (EnableTma) {
        // The vector path gives every lane its own slot in the tile; a DMA
        // descriptor is wave-uniform, so back that slot out to the tile base.
        uintptr_t tileCursor = cursor - uintptr_t(lane) * BytePerPack;
        tmaLoadStoreMc((char*)(outputUptr + tileCursor), (char*)tmaSmem->buff[0], (char const*)(inputUptr + tileCursor),
                       tileSize, tmaSmem->bar, lane);
      } else
#endif
      {
        BytePack<BytePerPack> tmp[UnrollPacks];
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < UnrollPacks; u++) {
          tmp[u] = *reinterpret_cast<BytePack<BytePerPack>*>(inputUptr + cursor + u * WARP_SIZE * BytePerPack);
        }
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < UnrollPacks; u++) {
          multimem_st_global(outputUptr + cursor + u * WARP_SIZE * BytePerPack, tmp[u]);
        }
      }
      cursor += tn * UnrollPacks * BytePerPack;
      nIters -= tn / WARP_SIZE;
    }
  } else {
    nPreBytes = 0;
    nSufBytes = nBytes;
  }

  // Get the prefix+suffix element one at a time.
  NVCC_PRAGMA_UNROLL(4)
  for (uintptr_t i = t * sizeof(T); i < nPreBytes + nSufBytes; i += tn * sizeof(T)) {
    uintptr_t cursor = i < nPreBytes ? i : nBytes - nSufBytes + (i - nPreBytes);
    BytePack<sizeof(T)> val = *reinterpret_cast<BytePack<sizeof(T)>*>(inputUptr + cursor);
    multimem_st_global(outputUptr + cursor, val);
  }
}

extern __shared__ ulong2 ncclSymkSmem[];

static __device__ void ncclSymkSmemPartition_help(int bumper) {}
template <typename T, typename... More>
static __device__ void ncclSymkSmemPartition_help(int bumper, T** ptr, int size, More... more) {
  T* ans = reinterpret_cast<T*>(ncclSymkSmem + bumper);
  __builtin_assume(__isShared(ans)); // Let compiler know this is shared memory (reinterpret_cast obscured as much).
  __builtin_assume_aligned(ans, sizeof(ulong2));
  *ptr = ans;
  bumper += (size * sizeof(T) + sizeof(ulong2) - 1) / sizeof(ulong2);
  ncclSymkSmemPartition_help(bumper, more...);
}

template <typename... Arg>
static __device__ void ncclSymkSmemPartition(Arg... args) {
  ncclSymkSmemPartition_help(/*bumper=*/0, args...);
}

////////////////////////////////////////////////////////////////////////////////
// Extensions to nccl_device.h needed to help compiler make good SASS:
////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct ncclLsaPointerGetter {
  void* base;
  uint32_t stride4G;
  __device__ ncclLsaPointerGetter(ncclSymPtr<T> ptr) {
    base = (char*)nccl::utility::loadConst(&ptr.window->lsaFlatBase);
    base = (char*)base + ptr.offset;
    stride4G = nccl::utility::loadConst(&ptr.window->stride4G);
  }
  __device__ T* operator()(int lsaPeer) const {
    return (T*)nccl::utility::add4G(base, lsaPeer * stride4G);
  }
};
#endif // NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_
