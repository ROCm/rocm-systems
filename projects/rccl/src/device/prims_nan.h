/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// NaN-flag protocol. Coordination mirrors LL128 -- same NCCL_STEPS credit FIFO,
// same 128-bit per-lane FIFO access -- but the payload is its own ready flag, so
// there is no flag lane and no register shuffle to move data out of it. A slice
// has arrived once none of its elements read back as NaN.
//
// The receiver polls its own FIFO with nontemporal loads, so a stale cached line
// cannot satisfy a poll. FIFO stores, including the sentinel restore, are plain,
// as LL128's are on gfx9.
//
// Two consequences follow from dropping the flag.
//
// Tearing stops mattering. LL128 needs the data and its flag to land in one
// indivisible 128-bit transaction; here a half-landed store just leaves the rest
// of the region at the sentinel, so the receiver keeps spinning. Correctness does
// not rest on store atomicity at any width.
//
// The receiver has to restore the sentinel. LL128's flag is step+1 and increases
// monotonically, so a slot holding last lap's flag can never be mistaken for this
// lap's. "Not NaN" carries no generation, so stale payload from the previous lap
// of the 8-slot FIFO would read as ready immediately. The receiver therefore
// writes the sentinel back over every slice it consumes, released ahead of the
// credit that lets the sender reuse the slot.
//
// Real data can still hold an all-ones dword, but only as a NaN: the sender
// clears bit 31 of every such dword before it goes on the wire (see
// escapeSentinel), which leaves a NaN that cannot be mistaken for an empty slot.

#include "rccl_ptr.h"

// Element-sized integer per floating-point type, for the sub-pack tail of a
// slice. Deliberately left undefined for every other type: generate.py only
// emits NaN-protocol kernels for these four, and a missing specialization should
// be a compile error rather than a silent wrong answer.
template <typename U>
struct ncclNanBits;
template <>
struct ncclNanBits<float> {
  using Bits = uint32_t;
};
template <>
struct ncclNanBits<double> {
  using Bits = uint64_t;
};
template <>
struct ncclNanBits<half> {
  using Bits = uint16_t;
};
template <>
struct ncclNanBits<hip_bfloat16> {
  using Bits = uint16_t;
};

// 128-bit FIFO load and store, one per lane.
//
// The receiver's poll must be a nontemporal load (or be followed by an acquire
// fence): a plain load can be served from a stale L1 line and spin on old data.
// The store is a plain one, as LL128's FIFO store is on gfx9.
inline __device__ v4u loadNanPack(v4u_gptr ptr) {
  return __builtin_nontemporal_load(ptr);
}

template <typename P>
inline __device__ v4u nanUserLoad(P ptr) {
  return __builtin_nontemporal_load(ptr);
}

inline __device__ void storeNanPack(v4u_gptr ptr, v4u v) {
  *ptr = v;
}

// 16-byte user-buffer vector that promises only the element's alignment, so a
// load or store through it is still a single dwordx4 on gfx9 (which accepts
// any alignment there) without asserting an alignment the buffer may lack.
typedef v4u ncclNanVecA2 __attribute__((aligned(2)));
typedef v4u ncclNanVecA4 __attribute__((aligned(4)));
typedef v4u ncclNanVecA8 __attribute__((aligned(8)));
template <int Align>
struct ncclNanUserVec;
template <>
struct ncclNanUserVec<2> {
  typedef __attribute__((address_space(1))) ncclNanVecA2* Ptr;
};
template <>
struct ncclNanUserVec<4> {
  typedef __attribute__((address_space(1))) ncclNanVecA4* Ptr;
};
template <>
struct ncclNanUserVec<8> {
  typedef __attribute__((address_space(1))) ncclNanVecA8* Ptr;
};

template <typename T, typename RedOp, typename Fan, int Direct, int P2p, bool isNetOffload, int Metadata, int Pipeline,
          int useAcc, int UserRegMode>
class Primitives<T, RedOp, Fan, Direct, ProtoNaN, P2p, isNetOffload, Metadata, Pipeline, useAcc, UserRegMode>
  : public PrimitivesWithoutDirect<
      Primitives<T, RedOp, Fan, Direct, ProtoNaN, P2p, isNetOffload, Metadata, Pipeline, useAcc, UserRegMode>> {
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  // Local FIFO pointer arrays need at least one slot: a send-only or recv-only
  // fan has a zero bound, which HIP device code rejects.
  static constexpr int RecvSlots = MaxRecv > 0 ? MaxRecv : 1, SendSlots = MaxSend > 0 ? MaxSend : 1;
  static constexpr int Input = 0, Output = 1, Acc = 2;
  // The unit of everything below is a 16-byte pack: one dwordx4 on the wire, in
  // the user buffer and in registers. LL128 is held to 64-bit words because its
  // flag is one; with no flag, nothing here is narrower than a pack. Pack p of a
  // lane sits at p * WARP_SIZE + lane within its warp's slice, on the wire and in
  // the user buffer alike, so a row of packs is 1 KiB of contiguous memory.
  static constexpr int EltPerPack = 16 / sizeof(T);
  static constexpr int PackPerThread = NCCL_NAN_ELEMS_PER_THREAD * sizeof(uint64_t) / 16;
  static constexpr int PackPerSlice = WARP_SIZE * PackPerThread;
  static constexpr int DataEltPerSlice = PackPerSlice * EltPerPack;

  RedOp redOp;
  const int tid;
  const int nthreads;
  const int wid;
  const int stepSize;
  const int warp;
  const int warpInBlock;
  const int group;
  const int threadsPerBlock;
  Fan fan;
  T* userBufs[3];
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;

  struct ncclConnInfo* sendConn = NULL;
  volatile struct ncclConnFifo* sendConnFifo = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache;

  uint64_t recvStep[MaxRecv];
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

  uint64_t* barriers;
  uint64_t barrier_next = 0;

  inline __device__ void barrier() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (nthreads != WARP_SIZE)
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1250__)
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
      int abort = __builtin_amdgcn_readfirstlane(
        __scoped_atomic_load_n((__attribute__((address_space(1))) uint32_t*)ncclShmem.comm.abortFlag,
                               __ATOMIC_SEQ_CST, __MEMORY_SCOPE_SYSTEM));
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
      // The credit also certifies that the peer has restored the sentinel in the
      // slot we are about to write. Without that the payload would be racing the
      // re-clear and the receiver could read our data back as empty.
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
        __builtin_amdgcn_s_sleep(1);
        sendConnHeadCache = __scoped_atomic_load_n((u64_gptr)sendConnHeadPtr, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
        if (checkAbort(abort, 1, spins)) break;
      }
      if (sendConnFifo) {
        __scoped_atomic_store_n((__attribute__((address_space(1))) int64_t*)&sendConnFifo[sendStep[wid] % NCCL_STEPS].size,
                                (int64_t)nbytes, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
      }
      sendConnHead += 1;
    }
  }

  inline __device__ void postRecv() {
    if (recvConnHeadPtr) {
      // The FIFO accesses are system-scope, so ordering the sentinel restore
      // ahead of the credit needs only a compiler barrier here.
      __atomic_signal_fence(__ATOMIC_SEQ_CST);
      STORE(recvConnHeadPtr, recvConnHead += 1);
    }
  }
  inline __device__ void postSend() {
    if (sendConnTailPtr) {
      __atomic_signal_fence(__ATOMIC_SEQ_CST);
      STORE((unsigned long long*)sendConnTailPtr, sendConnTail += 1);
    }
  }

  // True if any of this lane's packs has not been written yet, wholly or in
  // part. Stores land at least dword-atomically, so a half-landed pack leaves
  // whole dwords at the all-ones sentinel.
  //
  // For f16, bf16 and f32 every element sits inside one dword, and an all-ones
  // dword is NaN under all three. For f64 only the high dword decides: it is
  // all-ones exactly when the double is a NaN with the sign set, whereas a
  // finite double can carry an all-ones low dword. escapeSentinel() keeps
  // either kind of all-ones dword off the wire.
  //
  // "Some dword is all-ones" is "the unsigned max is all-ones", so the whole
  // test folds into a max3 tree and one compare instead of a compare per dword.
  template <int NR>
  __device__ __forceinline__ static bool anyNan(const v4u (&x)[NR]) {
    uint32_t m = 0;
#pragma unroll
    for (int p = 0; p < NR; p++) {
      if constexpr (sizeof(T) <= 4) {
        m = max(m, max(max(x[p][0], x[p][1]), max(x[p][2], x[p][3])));
      } else {
        m = max(m, max(x[p][1], x[p][3]));
      }
    }
    return m == 0xFFFFFFFFu;
  }

  // Clears bit 31 of every dword anyNan() would read as the sentinel. That bit
  // is a sign bit in each case (of the f32, of the upper f16/bf16, or of the
  // f64), so the value stays a NaN. It has to run after the reduce, not just on
  // input: a sum can carry NaN payloads into an all-ones dword, e.g. two f16
  // dwords that each hold one all-ones half.
  template <int NR>
  __device__ __forceinline__ static void escapeSentinel(v4u (&x)[PackPerThread]) {
    constexpr int First = sizeof(T) <= 4 ? 0 : 1, Stride = sizeof(T) <= 4 ? 1 : 2;
#pragma unroll
    for (int p = 0; p < NR; p++) {
#pragma unroll
      for (int d = First; d < 4; d += Stride) x[p][d] = x[p][d] == 0xFFFFFFFFu ? 0x7FFFFFFFu : x[p][d];
    }
  }

  // First element of the user buffer that pack p of this lane carries.
  __device__ __forceinline__ static int eltOf(int p, int lane) {
    return (p * WARP_SIZE + lane) * EltPerPack;
  }

  using UserVecPtr = typename ncclNanUserVec<alignof(T)>::Ptr;
  using EltBits = typename ncclNanBits<T>::Bits;
  typedef __attribute__((address_space(1))) EltBits* EltBitsPtr;

  // The FIFO and the user buffer are both handled in rows: row p is the p-th
  // pack of every lane, 1 KiB of contiguous memory. A slice is NR rows, with NR
  // a compile-time constant picked per slice by GenericOp(), so every loop
  // below is straight-line code with the full exec mask and no per-row test.
  // Within the last row the sender zero-fills past eltN; the whole row goes out
  // on the wire and the receiver polls and restores exactly the rows sent.
  static constexpr int EltPerRow = WARP_SIZE * EltPerPack;

  // A full slice is the steady state and takes one uniform branch to straight
  // dwordx4 loads with no per-lane predication. Everything else is the cold
  // tail of an op.
  //
  // In the tail a pack that straddles eltN cannot be read at its own offset:
  // those 16 bytes run past the slice and possibly into an unmapped page. It
  // reads the 16 bytes that end exactly at eltN instead, and alignTail() later
  // shifts them down into place, which also zero-fills the slots past eltN. That
  // keeps it one dwordx4 like every other pack. Elements past eltN end up zero
  // rather than whatever the buffer holds, so the padding that goes out on the
  // wire never depends on memory outside the slice.
  //
  // Only a slice shorter than one pack has no such window; it falls back to
  // element-wide loads.
  template <int NR>
  __device__ __forceinline__ void loadRegs(v4u (&regs)[PackPerThread], T const* src, int eltN, int lane) {
    if (eltN == NR * EltPerRow) {
#pragma unroll
      for (int p = 0; p < NR; p++) regs[p] = nanUserLoad((UserVecPtr)(src + eltOf(p, lane)));
    } else if (eltN >= EltPerPack) {
#pragma unroll
      for (int p = 0; p < NR; p++) {
        int e0 = eltOf(p, lane);
        v4u x = {0, 0, 0, 0};
        if (e0 < eltN) x = nanUserLoad((UserVecPtr)(src + min(e0, eltN - EltPerPack)));
        regs[p] = x;
      }
    } else {
#pragma unroll
      for (int p = 0; p < NR; p++) {
        int e0 = eltOf(p, lane);
        union {
          v4u v;
          EltBits b[EltPerPack];
        } x;
        x.v = v4u{0, 0, 0, 0};
#pragma unroll
        for (int e = 0; e < EltPerPack; e++) {
          if (e0 + e < eltN) x.b[e] = *((EltBitsPtr)(src + e0 + e));
        }
        regs[p] = x.v;
      }
    }
  }

  // Shift a straddling pack loaded by loadRegs() down into place. Kept apart
  // from the load so the caller can run it after the spin: touching the loaded
  // values any earlier makes the compiler wait for the user buffer before it
  // starts polling the FIFO. A no-op on every pack that is not the straddler.
  template <int NR>
  __device__ __forceinline__ static void alignTail(v4u (&regs)[PackPerThread], int eltN, int lane) {
    if (eltN == NR * EltPerRow || eltN < EltPerPack) return;
#pragma unroll
    for (int p = 0; p < NR; p++) {
      int d = max(0, eltOf(p, lane) - (eltN - EltPerPack)) * (int)sizeof(T);
      uint64_t lo = (uint64_t)regs[p][0] | ((uint64_t)regs[p][1] << 32);
      uint64_t hi = (uint64_t)regs[p][2] | ((uint64_t)regs[p][3] << 32);
      int b = 8 * (d & 7);
      uint64_t loS = b ? (lo >> b) | (hi << (64 - b)) : lo;
      uint64_t hiS = hi >> b;
      lo = d >= 16 ? 0 : d >= 8 ? hiS : loS;
      hi = d >= 8 ? 0 : hiS;
      regs[p] = v4u{(uint32_t)lo, (uint32_t)(lo >> 32), (uint32_t)hi, (uint32_t)(hi >> 32)};
    }
  }

  template <int NR>
  __device__ __forceinline__ void storeRegs(T* dst, v4u (&regs)[PackPerThread], int eltN, int lane) {
    if (eltN == NR * EltPerRow) {
#pragma unroll
      for (int p = 0; p < NR; p++) *((UserVecPtr)(dst + eltOf(p, lane))) = regs[p];
      return;
    }
#pragma unroll
    for (int p = 0; p < NR; p++) {
      int e0 = eltOf(p, lane);
      if (e0 + EltPerPack <= eltN) {
        *((UserVecPtr)(dst + e0)) = regs[p];
      } else if (e0 < eltN) {
        union {
          v4u v;
          EltBits b[EltPerPack];
        } x;
        x.v = regs[p];
#pragma unroll
        for (int e = 0; e < EltPerPack; e++) {
          if (e0 + e < eltN) *((EltBitsPtr)(dst + e0 + e)) = x.b[e];
        }
      }
    }
  }

  // Source registers are only touched once the first peer's data is in (or
  // straight away when there is nothing to receive), so the user-buffer load
  // overlaps the spin instead of being waited on ahead of it.
  template <int NR, int SrcBuf>
  __device__ __forceinline__ void finishSrc(v4u (&v)[PackPerThread], int eltN, int lane) {
    alignTail<NR>(v, eltN, lane);
    if (SrcBuf == Input) {
#pragma unroll
      for (int p = 0; p < NR; p++) v[p] = applyPreOp(redOp, v[p]);
    }
  }

  // Spin until every pack this lane owns has left the sentinel, then reduce and
  // forward. Unlike LL128 there is no reload after the spin: each lane validates
  // exactly the packs it keeps, so the values that ended the loop are the values
  // it goes on to use.
  template <int NR, int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void recvReduceSendCopy(v4u (&v)[PackPerThread], v4u_gptr const (&recvFifo)[RecvSlots],
                                                     v4u_gptr const (&sendFifo)[SendSlots], int nrecv, int nsend,
                                                     int lane, int& abortCache, int eltN, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;

    // Sparse readiness was tried here and removed. The sender drained with an
    // s_waitcnt before writing a marker pair so the receiver could poll one line
    // and then stream the rest open-loop, on the theory that the per-line
    // load -> test -> branch dependency was what capped throughput. It measured
    // 251.7 vs 251.8 GB/s at 1 GiB: the sender's drain costs exactly what the
    // receiver's open-loop reads gain. Worth knowing before anyone tries it
    // again -- the value dependency is not the ceiling.
    if (RECV) {
#pragma unroll
      for (int i = 0; i < MaxRecv; i++) {
        if (i >= nrecv) break;
        v4u_gptr ptr = recvFifo[i];
        v4u vr[NR];
        int spins = 0;
        bool pending;
        // The warp leaves the spin together; letting lanes exit independently
        // was measurably worse below ~32 MiB. An s_sleep backoff between
        // re-reads makes no difference at 1 GiB, so the loop stays tight.
        do {
#pragma unroll
          for (int p = 0; p < NR; p++) vr[p] = loadNanPack(ptr + p * WARP_SIZE);
          pending = anyNan<NR>(vr);
          if (checkAbort(abortCache, 1, spins)) break;
        } while (__any(pending));

        if (SRC && i == 0) finishSrc<NR, SrcBuf>(v, eltN, lane);
        // The first peer seeds v[] when there is no local source to reduce
        // against; every peer after that accumulates.
#pragma unroll
        for (int p = 0; p < NR; p++) v[p] = (SRC || i > 0) ? applyReduce(redOp, vr[p], v[p]) : vr[p];
      }
    }

    if (SRC && !RECV) finishSrc<NR, SrcBuf>(v, eltN, lane);

    if (postOp) {
#pragma unroll
      for (int p = 0; p < NR; p++) v[p] = applyPostOp(redOp, v[p]);
    }

    // Escaped in place, so the local DST copy matches what the peers receive.
    if (SEND) {
      escapeSentinel<NR>(v);
#pragma unroll
      for (int i = 1; i < MaxSend; i++) {
        if (i >= nsend) break;
#pragma unroll
        for (int p = 0; p < NR; p++) storeNanPack(sendFifo[i] + p * WARP_SIZE, v[p]);
      }
#pragma unroll
      for (int p = 0; p < NR; p++) storeNanPack(sendFifo[0] + p * WARP_SIZE, v[p]);
    }

    // Hand the consumed slots back to the sentinel only once everything that
    // depends on them is out. gfx9 counts stores in vmcnt alongside loads, so a
    // restore issued ahead of the reduce makes the wait on the user-buffer load
    // a wait for the restore too, and the forward store leaves late. The credit
    // published by postRecv() is what makes the restore safe at all: the peer
    // cannot touch this slot again until it sees that credit.
    if (RECV) {
      const v4u sentinel = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
#pragma unroll
      for (int i = 0; i < MaxRecv; i++) {
        if (i >= nrecv) break;
#pragma unroll
        for (int p = 0; p < NR; p++) storeNanPack(recvFifo[i] + p * WARP_SIZE, sentinel);
      }
    }
  }

  // One slice of eltN elements, carried in NR rows.
  template <int NR, int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void runSlice(T const* srcPtr, T* dstPtr, T const* accPtr,
                                           v4u_gptr const (&recvFifo)[RecvSlots], v4u_gptr const (&sendFifo)[SendSlots],
                                           int nrecv, int nsend, int lane, int& abortCache, int eltN,
                                           bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    constexpr int DST = DstBuf != -1 ? 1 : 0;
    v4u regs[PackPerThread];
    if (SRC) loadRegs<NR>(regs, srcPtr, eltN, lane);
    recvReduceSendCopy<NR, RECV, SEND, SrcBuf, DstBuf>(regs, recvFifo, sendFifo, nrecv, nsend, lane, abortCache, eltN,
                                                       postOp);
    if (DST) {
      if (accPtr != nullptr) {
        v4u accRegs[PackPerThread];
        loadRegs<NR>(accRegs, accPtr, eltN, lane);
        alignTail<NR>(accRegs, eltN, lane);
#pragma unroll
        for (int p = 0; p < NR; p++) regs[p] = applyReduce(redOp, accRegs[p], regs[p]);
      }
      storeRegs<NR>(dstPtr, regs, eltN, lane);
    }
  }

  template <int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void GenericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
    T const* srcPtr = SrcBuf == -1 ? nullptr : userBufs[SrcBuf] + srcIx;
    T* dstPtr = DstBuf == -1 ? nullptr : userBufs[DstBuf] + dstIx;
    T* accPtr = (DstBuf == -1 || !useAcc) ? nullptr : userBufs[Acc] + dstIx;
    const int lane = wid;
    const int nwarps = nthreads / WARP_SIZE;
    nelem = nelem < 0 ? 0 : nelem;
    const int nrecv = fan.nrecv(), nsend = fan.nsend();
    v4u_gptr recvFifo[RecvSlots];
    v4u_gptr sendFifo[SendSlots];
#pragma unroll
    for (int i = 0; i < MaxRecv; i++)
      recvFifo[i] = RECV && i < nrecv ? (v4u_gptr)recvPtr(i) + PackPerSlice * warp + lane : nullptr;
#pragma unroll
    for (int i = 0; i < MaxSend; i++)
      sendFifo[i] = SEND && i < nsend ? (v4u_gptr)sendPtr(i) + PackPerSlice * warp + lane : nullptr;
    int abortCache = __builtin_amdgcn_readfirstlane(abort);

    if (SEND) waitSend(divUp(nelem, DataEltPerSlice) * PackPerSlice * 16);
    barrier();

    sqtt_marker_enter("PRIM_NAN_DATA_PROCESS");
    nelem -= DataEltPerSlice * warp;
    srcPtr += DataEltPerSlice * warp;
    dstPtr += DataEltPerSlice * warp;
    if (accPtr != nullptr) accPtr += DataEltPerSlice * warp;
    // warp is wave-uniform, so this loop and the row dispatch are scalar
    // branches. A slice that fits in one row -- every slice of a small op --
    // moves 1 KiB instead of a full slice of padding.
    while (nelem > 0) {
      const int eltInSlice = min(nelem, DataEltPerSlice);
      if (PackPerThread > 1 && eltInSlice <= EltPerRow) {
        runSlice<1, RECV, SEND, SrcBuf, DstBuf>(srcPtr, dstPtr, accPtr, recvFifo, sendFifo, nrecv, nsend, lane,
                                                abortCache, eltInSlice, postOp);
      } else if (PackPerThread > 2 && eltInSlice <= 2 * EltPerRow) {
        runSlice<2, RECV, SEND, SrcBuf, DstBuf>(srcPtr, dstPtr, accPtr, recvFifo, sendFifo, nrecv, nsend, lane,
                                                abortCache, eltInSlice, postOp);
      } else {
        runSlice<PackPerThread, RECV, SEND, SrcBuf, DstBuf>(srcPtr, dstPtr, accPtr, recvFifo, sendFifo, nrecv, nsend,
                                                            lane, abortCache, eltInSlice, postOp);
      }
#pragma unroll
      for (int i = 0; i < MaxRecv; i++) recvFifo[i] += PackPerSlice * nwarps;
#pragma unroll
      for (int i = 0; i < MaxSend; i++) sendFifo[i] += PackPerSlice * nwarps;
      srcPtr += DataEltPerSlice * nwarps;
      dstPtr += DataEltPerSlice * nwarps;
      if (accPtr != nullptr) accPtr += DataEltPerSlice * nwarps;
      nelem -= DataEltPerSlice * nwarps;
    }

    abort = abortCache;
    barrier();

    sqtt_marker_exit("PRIM_NAN_DATA_PROCESS");

    if (SEND)
      for (int i = 0; i < MaxSend; i++) sendStep[i] += 1;
    if (SEND) postSend();
    if (RECV)
      for (int i = 0; i < MaxRecv; i++) recvStep[i] += 1;
    if (RECV) postRecv();
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_NAN];
    recvStep[i] = conn->step;
    if (wid == i) recvConn = conn;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= nthreads - WARP_SIZE && wid < fan.nrecv()) {
      recvConnHeadPtr = recvConn->head;
      recvConnHead = recvConn->step;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_NAN];
    sendStep[i] = conn->step;
    if (wid == i) sendConn = conn;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < fan.nsend()) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;
      sendConnHead = sendConn->step;
      sendConnFifo = sendConn->connFifo;
    }
    if (tid >= nthreads - WARP_SIZE && wid < fan.nsend()) {
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
    : redOp(redOpArg), tid(tid), nthreads(nthreads), wid(tid % WARP_SIZE),
      stepSize(ncclShmem.comm.buffSizes[NCCL_PROTO_NAN] / NCCL_STEPS / sizeof(uint64_t)), warp(__builtin_amdgcn_readfirstlane(tid / WARP_SIZE)),
      warpInBlock(threadIdx.x / WARP_SIZE), group(group), threadsPerBlock(blockDim.x) {
#ifdef ENABLE_WARP_SPEED
    auto* channel = ncclShmem.warpComm ? &ncclShmem.warpChannel[warpInBlock] : &ncclShmem.channel;
#else
    auto* channel = &ncclShmem.channel;
#endif
    barriers = &ncclShmem.groups[group].barrier;
    int nrecv = 0, nsend = 0;
    while (nrecv < MaxRecv && recvPeers[nrecv] >= 0) {
      loadRecvConn(&channel->peers[recvPeers[nrecv]]->recv[connIndexRecv], nrecv);
      nrecv++;
    }
    while (nsend < MaxSend && sendPeers[nsend] >= 0) {
      loadSendConn(&channel->peers[sendPeers[nsend]]->send[connIndexSend], nsend);
      nsend++;
    }
    this->fan = Fan(nrecv, nsend);
    // coverity[var_deref_model:FALSE]
    loadRecvSync();
    // coverity[var_deref_model:FALSE]
    loadSendSync();
    setDataPtrs(inputBuf, outputBuf, e != nullptr ? e->acc : nullptr);
  }

  __forceinline__ __device__ Primitives(int tid, int nthreads, int const* recvPeers, int const* sendPeers,
                                        void const* inputBuf, void* outputBuf, uint64_t redOpArg, uint8_t group,
                                        uint8_t connIndexRecv, uint8_t connIndexSend, struct ncclDevWorkColl* collWork,
                                        struct ncclDevWorkP2p* p2pWork, int stepSize_ = 0, int mode = primsModeDefault)
    : Primitives(tid, nthreads, recvPeers, sendPeers, inputBuf, outputBuf, redOpArg, group, connIndexRecv,
                 connIndexSend, collWork) {}

  __device__ ~Primitives() {
    // Save steps for the next operation
    if (tid >= nthreads - WARP_SIZE && wid < fan.nrecv()) recvConn->step = recvConnHead;
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
