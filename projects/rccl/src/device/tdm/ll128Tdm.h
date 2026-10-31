/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tensor-data-mover path for the LL128 user destination buffer: each warp's slice is
// staged in LDS and pushed out as one bulk transfer instead of per-lane vector stores.

// The receive and send FIFO legs are not eligible either: both interleave
// protocol flag words into the payload.

// Every entry point is gated on tdmEnable, captured from shmem in the Primitives
// constructor, so the off case costs one register test per slice and no shmem read.

static constexpr int TdmAlign = 128;
// storeUser128 takes system scope only when the buffer is registered.
static constexpr CachePolicy TdmStoreDevPolicy = createCachePolicy(TemporalHint::RT, MemScope::DEV);
static constexpr CachePolicy TdmStoreSysPolicy = createCachePolicy(TemporalHint::RT, MemScope::SYS);

__device__ __forceinline__ int tdmSlotIx(int g) const {
  return ll128SlotIx<NCCL_LL128_LINEELEMS, WARP_SIZE>(g, laneId());
}

__device__ __forceinline__ uint8_t* tdmWindow() const {
  uint8_t* p = reinterpret_cast<uint8_t*>(ncclScratchForWarp(warpInBlock));
  return p + ((-reinterpret_cast<uintptr_t>(p)) & (TdmAlign - 1));
}

// Stages the destination slice in LDS and pushes it out as one bulk store.
template <int WordPerThread>
__device__ __forceinline__ bool tdmStoreRegs(T* dst, uint64_t (&regs)[WordPerThread], int eltN) {
  constexpr int WireBytes = WordPerThread * WARP_SIZE * sizeof(uint64_t);
  constexpr int DataBytes = WireBytes - WireBytes / NCCL_LL128_LINEELEMS;
  static_assert(ncclShmemScratchWarpSize() >= DataBytes + TdmAlign - 1,
                "LL128 TDM needs one aligned data slice of per-warp scratch");
  if (!tdmEnable) return false;
  tdm::tdmWait();  // drain the previous slice before overwriting the window
  uint64_t* shm8 = shmemCvtPtr(reinterpret_cast<uint64_t*>(tdmWindow()));
#pragma unroll
  for (int g = 0; g < WordPerThread / 2; g++) {
    int ix = tdmSlotIx(g);
    if (!isFlagLane() || g % 2 == 0) storeShmem128(shm8 + 2 * ix, regs[2 * g + 0], regs[2 * g + 1]);
  }
  // The mover reads the window by descriptor, so nothing orders the lane stores above
  // against it; __syncwarp() is empty on AMD (common.h:19).
  __threadfence_block();
  const uint8_t* src8 = tdmWindow();
  size_t bytes = static_cast<size_t>(eltN) * sizeof(T);
  if (userBypass()) {
    tdm::asyncStoreFromLDS<SyncPolicy::Async, TdmStoreSysPolicy, false>(src8, reinterpret_cast<uint8_t*>(dst), bytes);
  } else {
    tdm::asyncStoreFromLDS<SyncPolicy::Async, TdmStoreDevPolicy, false>(src8, reinterpret_cast<uint8_t*>(dst), bytes);
  }
  return true;
}

// The last store must land before the barrier that publishes completion.
__device__ __forceinline__ void tdmDrain() {
  if (tdmEnable) tdm::tdmWait();
}
