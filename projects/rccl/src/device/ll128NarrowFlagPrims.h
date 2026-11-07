/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Narrow-flag (RCCL_LL128_FLAG_BYTES < 8) members of the LL128 Primitives, included inside the class body in
// prims_ll128.h in place of the 8-byte loadRegsBegin / loadRegsFinish / storeRegs. The flag-lane layout is in
// ll128NarrowFlag.h.

using NarrowLayout =
  Ll128NarrowLayout<RCCL_LL128_FLAG_BYTES, NCCL_LL128_LINEELEMS, NCCL_LL128_SHMEM_ELEMS_PER_THREAD, WARP_SIZE>;

// Bit g set: register pair g starts on the source's last word, which loadRegsBegin loaded into the pair's high half.
int loadShift = 0;

__device__ __forceinline__ void storeUser64(uint64_t* ptr, uint64_t v) {
  if (userBypass()) {
    __atomic_store_n(ptr, v, __ATOMIC_RELAXED);
  } else {
    *((u64_gptr)ptr) = v;
  }
}

template <int WordPerThread>
__device__ __forceinline__ void loadRegsBegin(uint64_t (&regs)[WordPerThread], T const* src, int eltN) {
  constexpr int EltPer8B = 8 / sizeof(T);
  constexpr int Chunks = WordPerThread / 2;
  static_assert(Chunks == NarrowLayout::ThreadChunks, "slice registers must match the layout");
  int ix[Chunks];
#pragma unroll
  for (int g = 0; g < Chunks; g++) ix[g] = NarrowLayout::userWordIx(g, wid);
  if (NarrowLayout::OddStarts) loadShift = 0;
  if (reinterpret_cast<uintptr_t>(src) % 16 == 0) {
    // Clamped as in the 8-byte path, to the last 16-byte block holding data, so a pair starting on an odd word stays
    // inside it. If that pair starts on the last word, the word lands in its high half and loadRegsFinish moves it.
    const int lastIx = 2 * ((eltN - 1) / (2 * EltPer8B));
#pragma unroll
    for (int g = 0; g < Chunks; g++) {
      const int ixc = min(ix[g], lastIx);
      if (NarrowLayout::OddStarts) loadShift |= (ix[g] == lastIx + 1) << g;
      loadUser128((uint64_t*)(src + ixc * EltPer8B), regs[2 * g + 0], regs[2 * g + 1]);
    }
  } else {
    // Not aligned. Stage the smallest 16 byte aligned region subsuming the buffer into shmem.
    int misalignment = reinterpret_cast<uintptr_t>(src) % 16;
    uint64_t* src8 = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(src) & -uintptr_t(16));
    uint64_t* shm8 = shmemCvtPtr((uint64_t*)ncclScratchForWarp(warpInBlock));
#pragma unroll
    for (int g = 0; g < Chunks; g++)
      if ((g * WARP_SIZE + wid) * 16 < misalignment + eltN * sizeof(T))
        loadUser128(src8 + 2 * (g * WARP_SIZE + wid), regs[2 * g + 0], regs[2 * g + 1]);
#pragma unroll
    for (int g = 0; g < Chunks; g++) storeShmem128(shm8 + 2 * (g * WARP_SIZE + wid), regs[2 * g + 0], regs[2 * g + 1]);
    __syncwarp();
    T* shm = (T*)shm8 + misalignment / sizeof(T);
#pragma unroll
    for (int g = 0; g < Chunks; g++) {
      if (NarrowLayout::storedWords(g, wid) != 0 && ix[g] * EltPer8B < eltN)
        loadShmemMisaligned128(shm + ix[g] * EltPer8B, regs[2 * g + 0], regs[2 * g + 1]);
    }
  }
}

template <int WordPerThread>
__device__ __forceinline__ void loadRegsFinish(uint64_t (&regs)[WordPerThread]) {
  if (!NarrowLayout::OddStarts) return;
#pragma unroll
  for (int g = 0; g < WordPerThread / 2; g++) {
    if ((loadShift >> g) & 1) regs[2 * g] = regs[2 * g + 1];
  }
}

template <int WordPerThread>
__device__ __forceinline__ void storeRegs(T* dst, uint64_t (&regs)[WordPerThread], int eltN) {
  constexpr int EltPer8B = 8 / sizeof(T);
  // Write to dst if 16-byte aligned, shmem otherwise. Without odd starts every stored pair is aligned and whole.
  int misalignment = reinterpret_cast<uintptr_t>(dst) % 16;
  uint64_t* shm8 = shmemCvtPtr((uint64_t*)ncclScratchForWarp(warpInBlock));
#pragma unroll
  for (int g = 0; g < WordPerThread / 2; g++) {
    const int ix = NarrowLayout::userWordIx(g, wid);
    const int m = NarrowLayout::storedWords(g, wid);
    if (m == 3 && misalignment == 0 && (ix + 2) * EltPer8B <= eltN) {
      storeUser128((uint64_t*)(dst + ix * EltPer8B), regs[2 * g + 0], regs[2 * g + 1]);
    } else if (!NarrowLayout::OddStarts) {
      if (m == 3) storeShmem128(shm8 + ix, regs[2 * g + 0], regs[2 * g + 1]);
    } else {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        if ((m >> k) & 1) {
          if (misalignment == 0 && (ix + k + 1) * EltPer8B <= eltN) {
            storeUser64((uint64_t*)(dst + (ix + k) * EltPer8B), regs[2 * g + k]);
          } else {
            shm8[ix + k] = regs[2 * g + k];
          }
        }
      }
    }
  }
  __syncwarp();
  // Write rest from shmem to dst.
  T* shm = (T*)ncclScratchForWarp(warpInBlock);
  int skip = misalignment == 0 ? eltN & -(NarrowLayout::OddStarts ? EltPer8B : 2 * EltPer8B) : 0;
  for (int i = skip + wid; i < eltN; i += WARP_SIZE) dst[i] = shm[i];
}

// Send register chunk u / 2 of v, packing a flag lane's user words around the narrow flag.
template <int N>
__device__ __forceinline__ void storeWire(uint64_t* ptr, uint64_t const (&v)[N], int u, uint64_t flag) {
  uint64_t p0, p1;
  NarrowLayout::packFlagChunk(v, u / 2, NarrowLayout::isOddFlagLine(wid), flag, p0, p1);
  store128Fifo(ptr, flagThread ? p0 : v[u], flagThread ? p1 : v[u + 1]);
}

template <int N>
__device__ __forceinline__ void unpackWire(uint64_t (&x)[N]) {
  NarrowLayout::unpack(x, flagThread, NarrowLayout::isOddFlagLine(wid),
                       [](uint32_t v) { return __shfl_xor(v, NarrowLayout::LineLanes, WARP_SIZE); });
}

// connFifo size of a clean step: the whole step, so the net proxy forwards the lines stampCleanStep wrote.
__device__ __forceinline__ int cleanStepBytes() const {
  return (stepSize / WireWordPerSlice) * WireWordPerSlice * sizeof(uint64_t);
}

// On a clean step, stamp the current flag on every line of the step this op did not write (slices from slicesSent).
__device__ __forceinline__ void stampCleanStep(int slicesSent) {
  const int slicesPerStep = stepSize / WireWordPerSlice;
  const int nwarps = nthreads / WARP_SIZE;
#pragma unroll
  for (int i = 0; i < MaxSend; i++) {
    if (i >= fan.nsend()) break;
    if (!RCCL_LL128_CLEAN_STEP(sendStep[i])) continue;
    uint64_t* ptr = sendPtr(i) + 2 * wid;
    const uint64_t flag = sendFlag(i);
    for (int s = slicesSent + warp; s < slicesPerStep; s += nwarps) {
#pragma unroll
      for (int u = 0; u < NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u += 2) {
        if (flagThread) store128Fifo(ptr + s * WireWordPerSlice + u * WARP_SIZE, 0, flag << RCCL_LL128_FLAG_SHIFT);
      }
    }
  }
}
