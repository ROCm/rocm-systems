/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Async-to-LDS path for the LL128 user source and destination buffers. The FIFO legs
// are not eligible: they interleave flag words into the payload. Both legs share one
// per-warp staging window, so each write to it drains first.

static constexpr int TdmAlign = 128;
// Mirrors loadUser128/storeUser128: system scope only when the buffer is registered.
static constexpr CachePolicy TdmLoadDevPolicy = createCachePolicy(TemporalHint::NT, MemScope::DEV);
static constexpr CachePolicy TdmLoadSysPolicy = createCachePolicy(TemporalHint::NT, MemScope::SYS);
static constexpr CachePolicy TdmStoreDevPolicy = createCachePolicy(TemporalHint::RT, MemScope::DEV);
static constexpr CachePolicy TdmStoreSysPolicy = createCachePolicy(TemporalHint::RT, MemScope::SYS);

// Set by GenericOp: the async load only helps when there is a receive to spin on.
bool tdmLoadAllowed = false;
// A load issued by tdmLoadBegin() and not yet consumed by tdmLoadFinish().
bool tdmLoadPending = false;
int tdmLoadEltN = 0;

__device__ __forceinline__ uint8_t* tdmWindow() const {
  uint8_t* p = reinterpret_cast<uint8_t*>(ncclScratchForWarp(warpInBlock));
  return p + ((-reinterpret_cast<uintptr_t>(p)) & (TdmAlign - 1));
}

// Issues the source read into LDS; true means loadRegsBegin must not run its own path.
template <int WordPerThread>
__device__ __forceinline__ bool tdmLoadBegin(T const* src, int eltN) {
  constexpr int WireBytes = WordPerThread * WARP_SIZE * sizeof(uint64_t);
  constexpr int DataBytes = WireBytes - WireBytes / NCCL_LL128_LINEELEMS;
  static_assert(ncclShmemScratchWarpSize() >= DataBytes + TdmAlign - 1,
                "LL128 TDM needs one aligned data slice of per-warp scratch");
  if (!tdmLoadAllowed) return false;
  if (reinterpret_cast<uintptr_t>(src) & (TdmAlign - 1)) return false;

  asyncWait<0>();  // previous slice's store may still be reading the window
  uint8_t* shm = tdmWindow();
  size_t bytes = static_cast<size_t>(eltN) * sizeof(T);
  if (userBypass()) {
    asyncLoadToLDS<SyncPolicy::Async, TdmLoadSysPolicy, true>(reinterpret_cast<const uint8_t*>(src), shm, bytes);
  } else {
    asyncLoadToLDS<SyncPolicy::Async, TdmLoadDevPolicy, true>(reinterpret_cast<const uint8_t*>(src), shm, bytes);
  }
  tdmLoadPending = true;
  tdmLoadEltN = eltN;
  return true;
}

// Waits on the outstanding load and reads LDS into the pre-shuffled register layout.
template <int WordPerThread>
__device__ __forceinline__ bool tdmLoadFinish(uint64_t (&regs)[WordPerThread]) {
  if (!tdmLoadPending) return false;
  tdmLoadPending = false;
  constexpr int EltPer16B = 16 / sizeof(T);
  constexpr int LineElems = NCCL_LL128_LINEELEMS;
  constexpr int LineSkip = 2 * WARP_SIZE / LineElems;
  uint64_t* shm8 = shmemCvtPtr(reinterpret_cast<uint64_t*>(tdmWindow()));
  asyncWait<0>();
#pragma unroll
  for (int g = 0; g < WordPerThread / 2; g++) {
    int ix = g * WARP_SIZE - LineSkip * (g / 2) + wid - (g % 2) * (wid / (LineElems / 2));
    if ((!flagThread || g % 2 == 0) && ix * EltPer16B < tdmLoadEltN)
      loadShmem128(shm8 + 2 * ix, regs[2 * g + 0], regs[2 * g + 1]);
  }
  return true;
}

// Stages the destination slice in LDS and pushes it out as one async store.
template <int WordPerThread>
__device__ __forceinline__ bool tdmStoreRegs(T* dst, uint64_t (&regs)[WordPerThread], int eltN) {
  constexpr int LineElems = NCCL_LL128_LINEELEMS;
  constexpr int LineSkip = 2 * WARP_SIZE / LineElems;
  asyncWait<0>();  // drain the previous slice before overwriting the window
  uint64_t* shm8 = shmemCvtPtr(reinterpret_cast<uint64_t*>(tdmWindow()));
#pragma unroll
  for (int g = 0; g < WordPerThread / 2; g++) {
    int ix = g * WARP_SIZE - LineSkip * (g / 2) + wid - (g % 2) * (wid / (LineElems / 2));
    if (!flagThread || g % 2 == 0) storeShmem128(shm8 + 2 * ix, regs[2 * g + 0], regs[2 * g + 1]);
  }
  __syncwarp();
  const uint8_t* src8 = tdmWindow();
  size_t bytes = static_cast<size_t>(eltN) * sizeof(T);
  if (userBypass()) {
    asyncStoreFromLDS<SyncPolicy::Async, TdmStoreSysPolicy, false>(src8, reinterpret_cast<uint8_t*>(dst), bytes);
  } else {
    asyncStoreFromLDS<SyncPolicy::Async, TdmStoreDevPolicy, false>(src8, reinterpret_cast<uint8_t*>(dst), bytes);
  }
  return true;
}

// The last store must land before the barrier that publishes completion.
__device__ __forceinline__ void tdmDrain() { asyncWait<0>(); }
