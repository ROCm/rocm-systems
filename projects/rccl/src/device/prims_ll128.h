/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#define NCCL_LL128_FLAGTHREAD (NCCL_LL128_LINEELEMS - 1)

// 1: data is only trusted from a load issued after its flags were seen valid,
//    so every row is reloaded once after it validates (upstream NCCL behavior).
// 0: trust the data that came back in the same load as the valid flag. This
//    relies on a line's data never becoming visible later than its flag to the
//    single warp-wide load that reads both.
#ifndef RCCL_LL128_CONFIRM_RELOAD
#define RCCL_LL128_CONFIRM_RELOAD 0
#endif

#ifndef RCCL_LL128_SKIP_CLEAN
#define RCCL_LL128_SKIP_CLEAN 0
#endif

#ifndef RCCL_USE_WBINVL1_VOL
#if defined(__GFX8__) || defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__)
#define RCCL_USE_WBINVL1_VOL 1
#else
#define RCCL_USE_WBINVL1_VOL 0
#endif
#endif

// 128-bit load for FIFO and non-registered user buffers. On gfx1250, sibling
// P2P into a cacheable comm FIFO is not coherent under a nontemporal load; use
// system-scope b128 there. See RCCL_LL_FIFO_SYS_SCOPE in rccl_ptr.h (default
// hipMalloc / cuMem hung; uncached did not). For registered user buffers, use
// load128 which bypasses the cache.
inline __device__ void load128NT(const uint64_t* ptr, uint64_t& v0, uint64_t& v1) {
#if RCCL_LL_FIFO_SYS_SCOPE
  union {
    v4u v;
    uint64_t u64[2];
  } u;
  u.v = __builtin_amdgcn_global_load_b128((v4u_gptr)ptr, RCCL_SYSTEM_SYNCSCOPE);
  v0 = u.u64[0];
  v1 = u.u64[1];
#else
  v0 = __builtin_nontemporal_load((u64_gptr)ptr);
  v1 = __builtin_nontemporal_load((u64_gptr)ptr + 1);
#endif
}

// Plain (cacheable) 128-bit store. Used for non-registered user buffers, and off
// gfx1250 it is also where store128Fifo sends the LL128 comm FIFO. Width is one
// b128 so data and flag stay in a single transaction.
inline __device__ void store128Plain(uint64_t* ptr, uint64_t v0, uint64_t v1) {
  union {
    v4u v;
    uint64_t u64[2];
  } u;
  u.u64[0] = v0;
  u.u64[1] = v1;
  *((v4u_gptr)ptr) = u.v;
}

// LL128 comm-FIFO store. Same 128-bit width as store128Plain (data+flag one
// transaction). On gfx1250 the store is system-scope: a plain store to a
// cacheable FIFO can retire in the writer's cache and never reach a sibling
// partition's poll. See RCCL_LL_FIFO_SYS_SCOPE in rccl_ptr.h.
inline __device__ void store128Fifo(uint64_t* ptr, uint64_t v0, uint64_t v1) {
#if RCCL_LL_FIFO_SYS_SCOPE
  union {
    v4u v;
    uint64_t u64[2];
  } u;
  u.u64[0] = v0;
  u.u64[1] = v1;
  __builtin_amdgcn_global_store_b128((v4u_gptr)ptr, u.v, RCCL_SYSTEM_SYNCSCOPE);
#else
  store128Plain(ptr, v0, v1);
#endif
}

template <typename T, typename RedOp, typename Fan, int Direct, int P2p, bool isNetOffload, int Metadata, int Pipeline,
          int useAcc, int UserRegMode>
class Primitives<T, RedOp, Fan, Direct, ProtoLL128, P2p, isNetOffload, Metadata, Pipeline, useAcc, UserRegMode>
  : public PrimitivesWithoutDirect<
      Primitives<T, RedOp, Fan, Direct, ProtoLL128, P2p, isNetOffload, Metadata, Pipeline, useAcc, UserRegMode>> {
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input = 0, Output = 1, Acc = 2;
  ;
  RedOp redOp;
  const int tid;
  const int nthreads;
  const int stepSize;
  const int warp;
  const int warpInBlock; // warp index in thread block
  const int group;
  // Primitives groups start on a warp boundary, so tid % WARP_SIZE is the hardware lane.
  static __device__ __forceinline__ int laneId() { return (int)__lane_id(); }
  static __device__ __forceinline__ bool isFlagLane() {
    return (laneId() % (NCCL_LL128_LINEELEMS / 2)) == (NCCL_LL128_LINEELEMS / 2 - 1);
  }
  const int threadsPerBlock;
  Fan fan;
  T* userBufs[3];
  bool userRegUsed = false; // runtime: user buffers registered (cacheable) -> need cache-bypass
  // No two NULL-initialized members may be adjacent. MemCpyOpt merges adjacent null stores
  // into a memset on this scratch-resident object, and the LL128 loops then reload these
  // pointers from scratch.
  struct ncclConnInfo* recvConn = NULL;
  uint64_t recvConnHead;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t sendConnTail;
  struct ncclConnInfo* sendConn = NULL;
  uint64_t sendConnHead;
  volatile struct ncclConnFifo* sendConnFifo = NULL;
  uint64_t sendConnHeadCache; // Cache last seen value
  volatile uint64_t* sendConnTailPtr = NULL;
  // Index the per-peer arrays only with compile-time indices. A run-time index moves the whole object to scratch,
  // and every scratch load in the slice loop costs a vmcnt(0) that also waits for the previous slice's FIFO stores.
  uint64_t recvStep[MaxRecv];
  volatile uint64_t* sendConnHeadPtr = NULL;

  uint64_t sendStep[MaxSend];
  uint64_t* recvBuff[MaxRecv];
  uint64_t* sendBuff[MaxSend];

  inline __device__ int recvOffset(int i) {
    return (recvStep[i] % NCCL_STEPS) * stepSize;
  }
  inline __device__ int sendOffset(int i) {
    return (sendStep[i] % NCCL_STEPS) * stepSize;
  }
  inline __device__ uint64_t* recvPtr(int i) {
    return recvBuff[i] + recvOffset(i);
  }
  inline __device__ uint64_t* sendPtr(int i) {
    return sendBuff[i] + sendOffset(i);
  }
  static constexpr int FlagBytes = RCCL_LL128_FLAG_BYTES;
  static constexpr int FlagShift = RCCL_LL128_FLAG_SHIFT;
  static inline __device__ uint64_t flagOf(uint64_t step1) {
    return RCCL_LL128_FLAG_OF(step1);
  }
  // word is the flag lane's second 64-bit word of a 16-byte chunk.
  static inline __device__ bool flagMatches(uint64_t word, uint64_t flag) {
    return (word >> FlagShift) == flag;
  }
  inline __device__ uint64_t recvFlag(int i) {
    return flagOf(recvStep[i] + 1);
  }
  inline __device__ uint64_t sendFlag(int i) {
    return flagOf(sendStep[i] + 1);
  }

  uint64_t* barriers;
  uint64_t barrier_next = 0;
  bool skip_fence = false;
  int loadShift = 0;

  inline __device__ void barrier() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (nthreads != WARP_SIZE)
#if defined(__gfx942__) || defined(__gfx950__) || (defined(__gfx1250__) || defined(__gfx1250_strict__))
      barrier_generic(__threadfence_block(), nthreads, barrier_next, barriers);
#else
      barrier_generic(__threadfence(), nthreads, barrier_next, barriers);
#endif
#else
    barrier_sync(15 - group, nthreads);
#endif
  }

  int abort = 0;

  __device__ inline int checkAbort(int& abortCache, const int abortValue, int& spins) {
    if (abortCache == 0 && ++spins == NCCL_SPINS_BEFORE_CHECK_ABORT) {
      int abort = __atomic_load_n((ncclShmem.comm.abortFlag), __ATOMIC_SEQ_CST);
      spins = 0;
      if (abort) {
        __atomic_store_n(&ncclShmem.aborted, abort, __ATOMIC_SEQ_CST);
        abortCache |= abortValue;
      }
    }
    return abortCache;
  }

  inline __device__ void waitSend(int nbytes) {
    if (sendConnHeadPtr) {
      int spins = 0;
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
        __builtin_amdgcn_s_sleep(1);
        sendConnHeadCache = __atomic_load_n(sendConnHeadPtr, __ATOMIC_RELAXED);
        if (checkAbort(abort, 1, spins)) break;
      }
      if (sendConnFifo) {
        uint64_t step = sendStep[0];
#pragma unroll
        for (int i = 1; i < MaxSend; i++) {
          if (laneId() == i) step = sendStep[i];
        }
        // A clean step stamps every line of the step; the proxy must forward all of them to the receiver.
        sendConnFifo[step % NCCL_STEPS].size =
          !RCCL_LL128_SKIP_CLEAN && RCCL_LL128_CLEAN_STEP(step)
            ? (stepSize / WireWordPerSlice) * WireWordPerSlice * sizeof(uint64_t)
            : nbytes;
      }
      sendConnHead += 1;
    }
  }

  inline __device__ void postRecv() {
    if (recvConnHeadPtr) STORE(recvConnHeadPtr, recvConnHead += 1);
  }
  inline __device__ void postSend() {
    if (sendConnTailPtr) {
      if (skip_fence) {
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
#if (defined(__gfx1250__) || defined(__gfx1250_strict__))
        // To be revisited for correctness and performance on gfx1250
        asm volatile("s_wait_loadcnt 0x0\n\ts_wait_storecnt 0x0");
#else
        asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
#endif
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
      } else {
        __threadfence_system();
      }
      STORE((unsigned long long*)sendConnTailPtr, sendConnTail += 1);
    }
  }

  // User-buffer 128-bit access mode selection.
  // Cache-bypassing (system-scope) loads/stores are only required when this op
  // accesses a registered, potentially-cached user buffer directly. That needs
  // both Direct != 0 (compile-time: this instantiation supports direct access)
  // and registration actually being in use.
  //
  // Whether registration is in use can be resolved at compile time via the
  // UserRegMode template parameter, which lets the caller instantiate clean
  // single-path kernels:
  //   UserRegMode==2 -> never registered: always plain/non-temporal, so the
  //                     system-scope load128/store128 code is never emitted and
  //                     the kernel keeps February-level occupancy.
  //   UserRegMode==1 -> always registered: always system-scope cache-bypass.
  //   UserRegMode==0 -> unknown at compile time: fall back to the per-op
  //                     userRegUsed member (dual path, legacy behavior).
  // When Direct == 0 the bypass folds away entirely regardless of UserRegMode.
  __device__ __forceinline__ bool userBypass() const {
    if (!Direct) {
      return false;
    }
    if (UserRegMode == 1) {
      return true;
    }
    if (UserRegMode == 2) {
      return false;
    }
    return userRegUsed;
  }
  __device__ __forceinline__ void loadUser128(const uint64_t* ptr, uint64_t& v0, uint64_t& v1) {
    if (userBypass()) {
      load128(ptr, v0, v1);
    } else {
      load128NT(ptr, v0, v1);
    }
  }
  __device__ __forceinline__ void storeUser128(uint64_t* ptr, uint64_t v0, uint64_t v1) {
    // Non-registered stores use the plain cacheable path (February behavior) to
    // recover store throughput; only the registered path keeps system-scope.
    if (userBypass()) {
      store128(ptr, v0, v1);
    } else {
      store128Plain(ptr, v0, v1);
    }
  }

  // LL128 slice geometry. Each thread owns Chunks 16-byte wire chunks per slice, one per row of WARP_SIZE lanes, and
  // every LineLanes consecutive lanes of a row form one line whose last lane is the flag lane.
  static constexpr int LineLanes = NCCL_LL128_LINEELEMS / 2;
  static constexpr int ThreadChunks = NCCL_LL128_SHMEM_ELEMS_PER_THREAD / 2;
  static constexpr int FlagLanesPerRow = WARP_SIZE / LineLanes;
  static constexpr int DataLanesPerRow = WARP_SIZE - FlagLanesPerRow;
  // Narrow flags: a flag lane's wire chunk carries WireData data bytes, then the flag. Its user words form a byte
  // stream of FlagStreamBytes that fills its chunks in order (packWire / unpackWire), so 8-byte types stay whole.
  static constexpr int WireData = 16 - FlagBytes;
  static constexpr int FlagStreamBytes = ThreadChunks * WireData;
  static constexpr int StreamDw = FlagStreamBytes / 4;
  // Whole user words per flag lane. When the stream ends in half a word (1-byte flag with ThreadChunks % 8 == 4),
  // the flag lanes of lines 2p and 2p+1 both hold the word they split, and each sends one half of it.
  static constexpr int FlagWords = FlagStreamBytes / 8;
  static constexpr bool SplitWord = RCCL_LL128_NARROW_FLAG && FlagStreamBytes % 8 != 0;
  // An even FlagWords keeps every flag-lane pair 16-byte aligned, interleaved across the row's flag lanes. Otherwise
  // a flag lane's words are contiguous, and some of its pairs start on an odd word.
  static constexpr bool OddStarts = RCCL_LL128_NARROW_FLAG && (SplitWord || FlagWords % 2 != 0);
  static_assert(!RCCL_LL128_NARROW_FLAG || FlagStreamBytes % 4 == 0, "flag-lane stream must be whole dwords");
  static_assert(!SplitWord || (FlagLanesPerRow % 2 == 0 && FlagWords + 1 <= 2 * ThreadChunks),
                "a split word needs pairs of flag lanes and a register for it");

  // Slice-relative user word index of register pair g: regs[2g], regs[2g+1] hold user words ix, ix+1.
  //
  // 8-byte flags: a flag lane's pair 2k+1 is filled from pair 2k in loadRegsFinish, so it loads only even pairs.
  // Narrow flags: data lanes hold whole chunks; flag lanes follow them. Words a lane holds but does not store
  // (storedWords) are duplicates.
  template <int Chunks>
  __device__ __forceinline__ int userWordIx(int g) {
    if (RCCL_LL128_NARROW_FLAG) {
      static_assert(!RCCL_LL128_NARROW_FLAG || Chunks == ThreadChunks, "slice registers must match the geometry");
      constexpr int Base = 2 * Chunks * DataLanesPerRow;
      const int line = laneId() / LineLanes;
      if (!isFlagLane()) return 2 * (g * DataLanesPerRow + laneId() - line);
      if (!OddStarts) return Base + 2 * (min(g, FlagWords / 2 - 1) * FlagLanesPerRow + line);
      if (!SplitWord) return Base + FlagWords * line + 2 * g;
      return Base + (2 * FlagWords + 1) * (line / 2) + FlagWords * (line % 2) + 2 * g;
    }
    // gfx1250 doubles LINEELEMS (128-byte non-tearing line), so the stride and flag-subgroup width double too.
    constexpr int LineElems = NCCL_LL128_LINEELEMS;
    constexpr int LineSkip = 2 * WARP_SIZE / LineElems;
    return 2 * (g * WARP_SIZE - LineSkip * (g / 2) + laneId() - (g % 2) * (laneId() / (LineElems / 2)));
  }
  __device__ __forceinline__ bool isOddFlagLine() {
    return SplitWord && (laneId() / LineLanes) % 2 == 1;
  }
  // Bit k set: this lane stores word k of pair g. Of a split word, only the odd line stores it.
  template <int Chunks>
  __device__ __forceinline__ int storedWords(int g) {
    if (!isFlagLane()) return 3;
    if (!RCCL_LL128_NARROW_FLAG) return g % 2 == 0 ? 3 : 0;
    const int words = FlagWords + (isOddFlagLine() ? 1 : 0);
    return (2 * g < words ? 1 : 0) | (2 * g + 1 < words ? 2 : 0);
  }

  template <int N>
  static __device__ __forceinline__ uint32_t dwordOf(uint64_t const (&x)[N], int k) {
    return k >= 2 * N ? 0u : k % 2 ? uint32_t(x[k / 2] >> 32) : uint32_t(x[k / 2]);
  }
  // Bytes [sh, sh + 4) of the 8 bytes hi:lo.
  static __device__ __forceinline__ uint32_t funnel(uint32_t lo, uint32_t hi, int sh) {
    return sh == 0 ? lo : __builtin_amdgcn_alignbyte(hi, lo, sh);
  }

  // Narrow flags: wire chunk c of a flag lane is bytes [c * WireData, (c + 1) * WireData) of its stream, then the
  // flag. The stream is the lane's user dwords in order. With a split word X (the even line's u[FlagWords], the odd
  // line's u[0]), the even line's stream ends with lo(X); the odd line skips lo(X) and sends the high dword of its
  // last word in its place, at position 0.
  template <int N>
  __device__ __forceinline__ void packWire(uint64_t const (&u)[N], int c, uint64_t flag, uint64_t& w0, uint64_t& w1) {
    if (!RCCL_LL128_NARROW_FLAG) {
      w0 = u[2 * c];
      w1 = isFlagLane() ? flag : u[2 * c + 1];
      return;
    }
    const bool odd = isOddFlagLine();
    auto s = [&](int i) -> uint32_t {
      if (i >= StreamDw) return 0u;
      if (SplitWord && i == 0) return odd ? dwordOf(u, 2 * FlagWords + 1) : dwordOf(u, 0);
      return dwordOf(u, i);
    };
    uint32_t p[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int b = c * WireData + 4 * j;
      p[j] = funnel(s(b / 4), s(b / 4 + 1), b % 4);
    }
    constexpr int DataBits = 32 - 8 * FlagBytes;
    p[3] = DataBits == 0 ? uint32_t(flag) : (p[3] & ((1u << DataBits) - 1)) | (uint32_t(flag) << DataBits);
    w0 = isFlagLane() ? (p[0] | uint64_t(p[1]) << 32) : u[2 * c];
    w1 = isFlagLane() ? (p[2] | uint64_t(p[3]) << 32) : u[2 * c + 1];
  }
  template <int N>
  __device__ __forceinline__ void unpackWire(uint64_t (&x)[N]) {
    if (!RCCL_LL128_NARROW_FLAG) return;
    static_assert(!RCCL_LL128_NARROW_FLAG || N == 2 * ThreadChunks, "slice registers must match the geometry");
    const bool fl = isFlagLane();
    uint32_t d[2 * N];
#pragma unroll
    for (int k = 0; k < 2 * N; k++) d[k] = dwordOf(x, k);
    uint32_t s[StreamDw];
#pragma unroll
    for (int i = 0; i < StreamDw; i++) {
      const int c0 = 4 * i / WireData, o = 4 * i % WireData;
      const int k = 4 * c0 + o / 4;
      uint32_t v = funnel(d[k], k + 1 < 2 * N ? d[k + 1] : 0u, o % 4);
      if (o + 4 > WireData) {
        const int n = WireData - o;
        v = (v & ((1u << (8 * n)) - 1)) | (d[4 * (c0 + 1)] << (8 * n));
      }
      s[i] = v;
    }
#pragma unroll
    for (int i = 0; i < StreamDw; i++) d[i] = fl ? s[i] : d[i];
    if (SplitWord) {
      // Swap halves of the split word with the partner line's flag lane.
      const bool odd = isOddFlagLine();
      const uint32_t other = __shfl_xor(odd ? s[1] : s[StreamDw - 1], LineLanes);
      if (fl) {
        d[0] = odd ? other : s[0];
        d[2 * FlagWords + 1] = odd ? s[0] : other;
      }
    }
#pragma unroll
    for (int k = 0; k < N; k++) x[k] = d[2 * k] | uint64_t(d[2 * k + 1]) << 32;
  }

  __device__ __forceinline__ void storeUser64(uint64_t* ptr, uint64_t v) {
    *((u64_gptr)ptr) = v;
  }

  template <int WordPerThread>
  __device__ __forceinline__ void loadRegsBegin(uint64_t (&regs)[WordPerThread], T const* src, int eltN) {
    constexpr int EltPer8B = 8 / sizeof(T);
    constexpr int Chunks = WordPerThread / 2;
    int ix[Chunks];
#pragma unroll
    for (int g = 0; g < Chunks; g++) ix[g] = userWordIx<Chunks>(g);
    if (reinterpret_cast<uintptr_t>(src) % 16 == 0) {
      /* We are aligned to 16 bytes, so load directly to registers no shmem.
       * Flag threads load half as much data which gets shuffled to the even
       * registers during Finish. The point of splitting into two phases is to
       * defer that shuffle, which incurs a dependency stall, until after other
       * memops are launched by the caller.
       */
      // Load unconditionally from an index clamped to the last chunk holding data. A guarded load leaves a phi that
      // needs the loaded value at the join, which forces a vmcnt wait right after the load, and on gfx9 that wait
      // also covers this warp's earlier FIFO stores. Words loaded only to satisfy the clamp are never stored:
      // storeRegs and the receiver stop at eltN.
      // The clamp keeps a pair starting on an odd word inside the last 16-byte block; if that pair starts on the last
      // word, the word lands in the pair's high half and loadRegsFinish moves it down.
      const int lastIx = 2 * ((eltN - 1) / (2 * EltPer8B));
      if (OddStarts) loadShift = 0;
#pragma unroll
      for (int g = 0; g < Chunks; g++) {
        const int ixc = min(ix[g], lastIx);
        if (OddStarts) loadShift |= (ix[g] == lastIx + 1) << g;
        loadUser128((uint64_t*)(src + ixc * EltPer8B), regs[2 * g + 0], regs[2 * g + 1]);
      }
    } else {
      if (OddStarts) loadShift = 0;
      // Not aligned. Stage the smallest 16 byte aligned region subsuming the
      // buffer into shmem.
      int misalignment = reinterpret_cast<uintptr_t>(src) % 16;
      uint64_t* src8 = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(src) & -uintptr_t(16));
      uint64_t* shm8 = shmemCvtPtr((uint64_t*)ncclScratchForWarp(warpInBlock));
#pragma unroll
      for (int g = 0; g < Chunks; g++)
        if ((g * WARP_SIZE + laneId()) * 16 < misalignment + eltN * sizeof(T))
          loadUser128(src8 + 2 * (g * WARP_SIZE + laneId()), regs[2 * g + 0], regs[2 * g + 1]);
#pragma unroll
      for (int g = 0; g < Chunks; g++)
        storeShmem128(shm8 + 2 * (g * WARP_SIZE + laneId()), regs[2 * g + 0], regs[2 * g + 1]);

      __syncwarp();

      // Now load from shmem stage to regs. Preserve the same pre-shuffled layout
      // as the aligned case since Finish() will be applied regardless.
      T* shm = (T*)shm8 + misalignment / sizeof(T);
#pragma unroll
      for (int g = 0; g < Chunks; g++) {
        if (storedWords<Chunks>(g) != 0 && ix[g] * EltPer8B < eltN)
          loadShmemMisaligned128(shm + ix[g] * EltPer8B, regs[2 * g + 0], regs[2 * g + 1]);
      }
    }
  }

  template <int WordPerThread>
  __device__ __forceinline__ void loadRegsFinish(uint64_t (&regs)[WordPerThread]) {
    if (OddStarts) {
#pragma unroll
      for (int g = 0; g < WordPerThread / 2; g++) {
        if ((loadShift >> g) & 1) regs[2 * g] = regs[2 * g + 1];
      }
    }
    if (RCCL_LL128_NARROW_FLAG) return;
    // Move data out of flag registers into the vacant registers.
#pragma unroll
    for (int g = 1; g < WordPerThread / 2; g += 2) {
      if (isFlagLane()) regs[2 * g] = regs[2 * g - 1];
    }
  }

  template <int WordPerThread>
  __device__ __forceinline__ void storeRegs(T* dst, uint64_t (&regs)[WordPerThread], int eltN) {
    constexpr int EltPer8B = 8 / sizeof(T);
    constexpr int Chunks = WordPerThread / 2;
    // Reverse Finish() register permuatation.
#pragma unroll
    for (int g = 1; g < Chunks; g += 2) {
      if (!RCCL_LL128_NARROW_FLAG && isFlagLane()) regs[2 * g - 1] = regs[2 * g];
    }

    // Write to dst if 16-byte aligned, shmem otherwise.
    int misalignment = reinterpret_cast<uintptr_t>(dst) % 16;
    uint64_t* shm8 = shmemCvtPtr((uint64_t*)ncclScratchForWarp(warpInBlock));
    if (!OddStarts) {
      // Every stored pair is 16-byte aligned and whole.
#pragma unroll
      for (int g = 0; g < Chunks; g++) {
        const int ix = userWordIx<Chunks>(g);
        if (storedWords<Chunks>(g) == 3) {
          if (misalignment == 0 && (ix + 2) * EltPer8B <= eltN) {
            storeUser128((uint64_t*)(dst + ix * EltPer8B), regs[2 * g + 0], regs[2 * g + 1]);
          } else {
            storeShmem128(shm8 + ix, regs[2 * g + 0], regs[2 * g + 1]);
          }
        }
      }
      __syncwarp();
      T* shm = (T*)ncclScratchForWarp(warpInBlock);
      int skip = misalignment == 0 ? eltN & -(2 * EltPer8B) : 0;
      for (int i = skip + laneId(); i < eltN; i += WARP_SIZE) dst[i] = shm[i];
      return;
    }
#pragma unroll
    for (int g = 0; g < Chunks; g++) {
      const int ix = userWordIx<Chunks>(g);
      const int m = storedWords<Chunks>(g);
      if (m == 3 && misalignment == 0 && (ix + 2) * EltPer8B <= eltN) {
        storeUser128((uint64_t*)(dst + ix * EltPer8B), regs[2 * g + 0], regs[2 * g + 1]);
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
    // Write rest from shmem to dst. No need to coalesce stores to 16-bytes,
    // the hardware keeps up fine.
    T* shm = (T*)ncclScratchForWarp(warpInBlock);
    int skip = misalignment == 0 ? eltN & -EltPer8B : 0;
    for (int i = skip + laneId(); i < eltN; i += WARP_SIZE) dst[i] = shm[i];
  }

#define WARP_MASK 0xffffffff

  // Spin until every flag of one peer's slice is valid, leaving the slice in vr.
  // Each spin reloads only the window of rows, lowest to highest, whose flags
  // were not all valid at the previous vote; see RCCL_LL128_CONFIRM_RELOAD.
  template <int ELEMS_PER_THREAD>
  __device__ __forceinline__ void recvWaitLoad(uint64_t* ptr, uint64_t flag, uint64_t (&vr)[ELEMS_PER_THREAD]) {
    constexpr int Rows = ELEMS_PER_THREAD / 2;
    constexpr bool Confirm = RCCL_LL128_CONFIRM_RELOAD;
    static_assert(Rows <= 32, "reload tracking holds one bit per row");
    constexpr uint32_t AllRows = Rows == 32 ? ~0u : (1u << Rows) - 1;
    constexpr uint64_t WarpMask = WARP_SIZE == 64 ? ~0ull : (1ull << WARP_SIZE) - 1;
    constexpr int LineLanes = NCCL_LL128_LINEELEMS / 2;
    constexpr uint64_t FlagLanes = (WarpMask / ((1ull << LineLanes) - 1)) << (LineLanes - 1);

    // Per-row flag votes; lanes not holding a flag always vote yes.
    uint64_t ok[Rows];
    // Returns true (warp-uniform) once every flag in the slice is valid.
    auto vote = [&]() -> bool {
      uint64_t allOk = WarpMask;
#pragma unroll
      for (int r = 0; r < Rows; r++) {
        ok[r] = __ballot(flagMatches(vr[2 * r + 1], flag)) | (WarpMask & ~FlagLanes);
        allOk &= ok[r];
      }
      return allOk == WarpMask;
    };
    auto loadAll = [&]() {
#pragma unroll
      for (int u = 0; u < ELEMS_PER_THREAD; u += 2) load128NT(ptr + u * WARP_SIZE, vr[u], vr[u + 1]);
    };
    // Rows (warp-uniform) with every flag valid at the last vote.
    auto validRows = [&]() -> uint32_t {
      uint32_t rows = 0;
#pragma unroll
      for (int r = 0; r < Rows; r++) {
        if (ok[r] == WarpMask) rows |= 1u << r;
      }
      return rows;
    };
    // Reload the window spanning the lowest to the highest row not yet done.
    auto loadRows = [&](uint32_t skip) {
      uint32_t pending = AllRows & ~skip;
      if (pending == 0) return;
      int lo = __builtin_ctz(pending);
      int hi = 32 - __builtin_clz(pending);
#pragma unroll
      for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
        if (u / 2 >= lo && u / 2 < hi) load128NT(ptr + u * WARP_SIZE, vr[u], vr[u + 1]);
      }
    };

    loadAll();
    if (vote()) {
      if (Confirm) loadAll();
      return;
    }
    uint32_t valid = validRows();
    uint32_t done = Confirm ? 0 : valid;
    bool needReload;
    int spins = 0;
    do {
      loadRows(done);
      if (Confirm) done |= valid;
      needReload = !vote();
      valid = needReload ? validRows() : AllRows;
      if (!Confirm) done = valid;
      needReload &= (0 == checkAbort(abort, 1, spins));
    } while (__any(needReload));
    if (Confirm) loadRows(done);
  }

  // recvPtr0/recvFlag0/sendPtr0/sendFlag0 are peer 0's FIFO and flag for this step, computed once per op so the
  // slice loop does not reload them from scratch. A scratch load there costs a vmcnt(0), which on gfx9 also waits
  // for the previous slice's remote stores to complete.
  template <int ELEMS_PER_THREAD, int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void recvReduceSendCopy(uint64_t (&v)[ELEMS_PER_THREAD], int ll128Offset, bool postOp,
                                                     uint64_t* recvPtr0, uint64_t recvFlag0, uint64_t* sendPtr0,
                                                     uint64_t sendFlag0) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    uint64_t vr[ELEMS_PER_THREAD];

    __syncwarp();
    /************************ Wait first recv ********************/
    if (RECV) {
      recvWaitLoad(recvPtr0 + ll128Offset, recvFlag0, vr);
      unpackWire(vr);
    }

    /************* Finish register load **************/
    if (SRC) {
      // By deferring register shuffle here we've overlapped spinning on first
      // peer's data with memory loads of src data.
      loadRegsFinish(v);
      if (SrcBuf == Input) {
#pragma unroll
        for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
          v[u] = applyPreOp(redOp, v[u]);
          if (RCCL_LL128_NARROW_FLAG || !isFlagLane()) v[u + 1] = applyPreOp(redOp, v[u + 1]);
        }
      }
    }

    /************************ Recv rest *********************/
    if (RECV) {
      { // Consume data from first recv
#pragma unroll
        for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
          v[u] = SRC ? applyReduce(redOp, vr[u], v[u]) : vr[u];
          v[u + 1] = SRC ? applyReduce(redOp, vr[u + 1], v[u + 1]) : vr[u + 1];
        }
      }

      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      for (int i = 1; i < MaxRecv && i < fan.nrecv(); i++) {
        recvWaitLoad(recvPtr(i) + ll128Offset, recvFlag(i), vr);
        unpackWire(vr);

#pragma unroll
        for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
          v[u] = applyReduce(redOp, vr[u], v[u]);
          v[u + 1] = applyReduce(redOp, vr[u + 1], v[u + 1]);
        }
      }
    }
    /********************** End Recv ************************/

    if (postOp) {
#pragma unroll
      for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
        v[u] = applyPostOp(redOp, v[u]);
        v[u + 1] = applyPostOp(redOp, v[u + 1]);
      }
    }

#if RCCL_USE_WBINVL1_VOL
    if (tid == 0) __builtin_amdgcn_buffer_wbinvl1();
#endif
    /************************ Send **************************/
    if (SEND) {
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      for (int i = 1; i < MaxSend && i < fan.nsend(); i++) {
        uint64_t flag = sendFlag(i);
        uint64_t* ptr = sendPtr(i) + ll128Offset;
#pragma unroll
        for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
          uint64_t w0, w1;
          packWire(v, u / 2, flag, w0, w1);
          store128Fifo(ptr + u * WARP_SIZE, w0, w1);
        }
      }
      uint64_t* ptr = sendPtr0 + ll128Offset;
#pragma unroll
      for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
        uint64_t w0, w1;
        packWire(v, u / 2, sendFlag0, w0, w1);
        store128Fifo(ptr + u * WARP_SIZE, w0, w1);
      }
    }
    /********************** End Send ************************/
  }

  static constexpr int WireWordPerSlice = WARP_SIZE * NCCL_LL128_SHMEM_ELEMS_PER_THREAD;
  static constexpr int DataEltPerSlice =
    (WireWordPerSlice / NCCL_LL128_LINEELEMS) * (NCCL_LL128_LINESIZE - FlagBytes) / sizeof(T);

  template <int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void GenericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    constexpr int DST = DstBuf != -1 ? 1 : 0;
    T const* srcPtr = SrcBuf == -1 ? nullptr : userBufs[SrcBuf] + srcIx;
    T* dstPtr = DstBuf == -1 ? nullptr : userBufs[DstBuf] + dstIx;
    T* accPtr = (DstBuf == -1 || !useAcc) ? nullptr : userBufs[Acc] + dstIx;
    int wireOffset = WireWordPerSlice * warp + 2 * laneId();
    const int nwarps = nthreads / WARP_SIZE;
    nelem = nelem < 0 ? 0 : nelem;
    const int slicesSent = divUp(nelem, DataEltPerSlice);

    if (SEND) waitSend(slicesSent * WireWordPerSlice * sizeof(uint64_t));
    barrier();

    sqtt_marker_enter("PRIM_LL128_DATA_PROCESS");
    nelem -= DataEltPerSlice * warp;
    srcPtr += DataEltPerSlice * warp;
    dstPtr += DataEltPerSlice * warp;
    if (accPtr != nullptr) accPtr += DataEltPerSlice * warp;
    uint64_t* const recvPtr0 = RECV ? recvPtr(0) : nullptr;
    const uint64_t recvFlag0 = RECV ? recvFlag(0) : 0;
    uint64_t* const sendPtr0 = SEND ? sendPtr(0) : nullptr;
    const uint64_t sendFlag0 = SEND ? sendFlag(0) : 0;
    while (nelem > 0) {
      const int eltInSlice = min(nelem, DataEltPerSlice);
      uint64_t regs[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
      if (SRC) loadRegsBegin(regs, srcPtr, eltInSlice);
      recvReduceSendCopy<NCCL_LL128_SHMEM_ELEMS_PER_THREAD, RECV, SEND, SrcBuf, DstBuf>(
        regs, wireOffset, postOp, recvPtr0, recvFlag0, sendPtr0, sendFlag0);
      if (DST) {
        if (accPtr != nullptr) {
          uint64_t accRegs[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
          loadRegsBegin(accRegs, accPtr, eltInSlice);
          loadRegsFinish(accRegs);
          accPtr += DataEltPerSlice * nwarps;
#pragma unroll
          for (int u = 0; u < NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u++) {
            regs[u] = applyReduce(redOp, accRegs[u], regs[u]);
          }
        }
        storeRegs(dstPtr, regs, eltInSlice);
      }

      wireOffset += WireWordPerSlice * nwarps;
      srcPtr += DataEltPerSlice * nwarps;
      dstPtr += DataEltPerSlice * nwarps;
      nelem -= DataEltPerSlice * nwarps;
    }

    if (SEND && RCCL_LL128_NARROW_FLAG && !RCCL_LL128_SKIP_CLEAN) {
      const int slicesPerStep = stepSize / WireWordPerSlice;
#pragma unroll
      for (int i = 0; i < MaxSend; i++) {
        if (i >= fan.nsend()) break;
        if (!RCCL_LL128_CLEAN_STEP(sendStep[i])) continue;
        uint64_t* ptr = (i == 0 ? sendPtr0 : sendPtr(i)) + 2 * laneId();
        const uint64_t flag = i == 0 ? sendFlag0 : sendFlag(i);
        for (int s = slicesSent + warp; s < slicesPerStep; s += nwarps) {
#pragma unroll
          for (int u = 0; u < NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u += 2) {
            if (isFlagLane()) store128Fifo(ptr + s * WireWordPerSlice + u * WARP_SIZE, 0, flag << FlagShift);
          }
        }
      }
    }

    barrier();

    sqtt_marker_exit("PRIM_LL128_DATA_PROCESS");

    if (SEND)
      for (int i = 0; i < MaxSend; i++) sendStep[i] += 1;
    if (SEND) postSend();
    if (RECV)
      for (int i = 0; i < MaxRecv; i++) recvStep[i] += 1;
    if (RECV) postRecv();
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
    recvStep[i] = conn->step;
    if (laneId() == i) recvConn = conn;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= nthreads - WARP_SIZE && laneId() < fan.nrecv()) {
      recvConnHeadPtr = recvConn->head;
      recvConnHead = recvConn->step;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
    sendStep[i] = conn->step;
    if (laneId() == i) sendConn = conn;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < fan.nsend()) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;
      sendConnHead = sendConn->step;
      sendConnFifo = sendConn->connFifo;
    }
    if (tid >= nthreads - WARP_SIZE && laneId() < fan.nsend()) {
      if (sendConn->connFifo) {
        sendConnTailPtr = sendConn->tail;
        sendConnTail = sendConn->step;
      }
    }
  }

public:
  __device__ Primitives(const int tid, const int nthreads, int const* recvPeers, int const* sendPeers,
                        void const* inputBuf, void* outputBuf, uint64_t redOpArg, uint8_t group = 0,
                        uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0, struct ncclDevWorkColl* e = nullptr,
                        bool ipcReg = false, bool netReg = false, int stepSize_ = 0)
    : redOp(redOpArg), tid(tid), nthreads(nthreads), /*compiler warnings*/
      stepSize(ncclShmem.comm.buffSizes[NCCL_PROTO_LL128] / NCCL_STEPS / sizeof(uint64_t)), warp(tid / WARP_SIZE),
      warpInBlock(threadIdx.x / WARP_SIZE),
      group(group),
      threadsPerBlock(blockDim.x) {
#ifdef ENABLE_WARP_SPEED
    auto* channel = ncclShmem.warpComm ? &ncclShmem.warpChannel[warpInBlock] : &ncclShmem.channel;
#else
    auto* channel = &ncclShmem.channel;
#endif
    barriers = &ncclShmem.groups[group].barrier;
    int nrecv = 0, nsend = 0;
#pragma unroll
    for (int i = 0; i < MaxRecv; i++) {
      if (nrecv == i && recvPeers[i] >= 0) {
        loadRecvConn(&channel->peers[recvPeers[i]]->recv[connIndexRecv], i);
        nrecv++;
      }
    }
#pragma unroll
    for (int i = 0; i < MaxSend; i++) {
      if (nsend == i && sendPeers[i] >= 0) {
        loadSendConn(&channel->peers[sendPeers[i]]->send[connIndexSend], i);
        nsend++;
      }
    }
    this->fan = Fan(nrecv, nsend);
    // Coverity reports recvConn and sendConn being possibly NULL at this point but that won't actually
    // happen given the two "while" loops just above.
    // coverity[var_deref_model:FALSE]
    loadRecvSync();
    // coverity[var_deref_model:FALSE]
    loadSendSync();
    userRegUsed = (e != nullptr) && (e->regUsed || e->netRegUsed);
    setDataPtrs(inputBuf, outputBuf, e != nullptr ? e->acc : nullptr);
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
    skip_fence = !ncclShmem.comm.cheapPostSendFenceOff;
#else
    // The cheap post-peer fence is only safe with global DWORDX4 builtins
    // (system-scope cache-bypassing stores); otherwise always use the full fence.
    skip_fence = false;
#endif
  }

  __forceinline__ __device__ Primitives(int tid, int nthreads, int const* recvPeers, int const* sendPeers,
                                        void const* inputBuf, void* outputBuf, uint64_t redOpArg, uint8_t group,
                                        uint8_t connIndexRecv, uint8_t connIndexSend, struct ncclDevWorkColl* collWork,
                                        struct ncclDevWorkP2p* p2pWork, int stepSize_ = 0, int mode = primsModeDefault)
    : Primitives(tid, nthreads, recvPeers, sendPeers, inputBuf, outputBuf, redOpArg, group, connIndexRecv,
                 connIndexSend, collWork) {}

  __device__ ~Primitives() {
    // Save steps for the next operation
    if (tid >= nthreads - WARP_SIZE && laneId() < fan.nrecv()) recvConn->step = recvConnHead;
    if (tid < fan.nsend()) sendConn->step = sendConnHead;
    // Ensure all steps written back
    barrier();
  }

  __device__ void setDataPtrs(void const* inputBuf, void* outputBuf, void const* acc = nullptr) {
    userBufs[Input] = (T*)inputBuf;
    userBufs[Output] = (T*)outputBuf;
    userBufs[Acc] = (T*)acc;
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    userBufs[Input] += delta;
    userBufs[Output] += delta;
  }

  __device__ void send(intptr_t inpIx, int eltN) {
    GenericOp<0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ void sendFromOutput(intptr_t outIx, int eltN) {
    GenericOp<0, 1, Output, -1>(outIx, -1, eltN, false);
  }
  __device__ void recv(intptr_t outIx, int eltN, bool postOp = false) {
    GenericOp<1, 0, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ void recvReduceSend(intptr_t inpIx, int eltN) {
    GenericOp<1, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    GenericOp<1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    GenericOp<0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void recvCopySend(intptr_t outIx, int eltN, bool postOp = false) {
    GenericOp<1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    GenericOp<1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void recvSend(int eltN) {
    return GenericOp<1, 1, -1, -1>(-1, -1, eltN, false);
  }
};
