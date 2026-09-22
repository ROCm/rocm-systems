/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// NaN-flag protocol. Coordination mirrors LL128 -- same NCCL_STEPS credit FIFO,
// same non-temporal / system-scope 128-bit FIFO access -- but the payload is its
// own ready flag, so there is no flag lane and no register shuffle to move data
// out of it. A slice has arrived once none of its elements read back as NaN.
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
// Limitation: a genuine NaN in user data is indistinguishable from an empty slot
// and hangs the receiver. The protocol is opt-in for that reason (see the gate in
// updateCollCostTable), and NCCL_NAN_PROTO_CHECK_INPUT traps on NaN input in
// debug builds.

#include "rccl_ptr.h"

// Exponent/mantissa masks per floating-point type, used to test an element for
// NaN straight out of the wire word without any type punning. Deliberately left
// undefined for every other type: generate.py only emits NaN-protocol kernels for
// these four, and a missing specialization should be a compile error rather than
// a silent wrong answer.
template <typename U>
struct ncclNanBits;
template <>
struct ncclNanBits<float> {
  using Bits = uint32_t;
  static constexpr Bits Exp = 0x7F800000u;
  static constexpr Bits Mant = 0x007FFFFFu;
};
template <>
struct ncclNanBits<double> {
  using Bits = uint64_t;
  static constexpr Bits Exp = 0x7FF0000000000000ull;
  static constexpr Bits Mant = 0x000FFFFFFFFFFFFFull;
};
template <>
struct ncclNanBits<half> {
  using Bits = uint16_t;
  static constexpr Bits Exp = 0x7C00u;
  static constexpr Bits Mant = 0x03FFu;
};
template <>
struct ncclNanBits<hip_bfloat16> {
  using Bits = uint16_t;
  static constexpr Bits Exp = 0x7F80u;
  static constexpr Bits Mant = 0x007Fu;
};

// Plain 64-bit FIFO load. Matches load128NT's scope choice (see
// RCCL_LL_FIFO_SYS_SCOPE in rccl_ptr.h): on gfx1250 a sibling partition's write
// to a cacheable FIFO is not observable under a nontemporal load.
inline __device__ uint64_t loadNanWord(const uint64_t* ptr) {
#if RCCL_LL_FIFO_SYS_SCOPE
  return __scoped_atomic_load_n((u64_gptr)ptr, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
#else
  return __builtin_nontemporal_load((u64_gptr)ptr);
#endif
}

// Plain 64-bit FIFO store, scope-matched to loadNanWord. Used both for payload
// and for restoring the sentinel.
inline __device__ void storeNanWord(uint64_t* ptr, uint64_t v) {
#if RCCL_LL_FIFO_SYS_SCOPE
  __scoped_atomic_store_n((u64_gptr)ptr, v, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
#else
  __builtin_nontemporal_store(v, (u64_gptr)ptr);
#endif
}

template <typename T, typename RedOp, typename Fan, int Direct, int P2p, bool isNetOffload, int Metadata, int Pipeline,
          int useAcc, int UserRegMode>
class Primitives<T, RedOp, Fan, Direct, ProtoNaN, P2p, isNetOffload, Metadata, Pipeline, useAcc, UserRegMode>
  : public PrimitivesWithoutDirect<
      Primitives<T, RedOp, Fan, Direct, ProtoNaN, P2p, isNetOffload, Metadata, Pipeline, useAcc, UserRegMode>> {
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input = 0, Output = 1, Acc = 2;
  // Elements of T carried by one 64-bit wire word. sizeof(T) <= 8 for every type
  // this protocol supports, so this is at least 1.
  static constexpr int EltPerWord = sizeof(uint64_t) / sizeof(T);
  static constexpr int WordPerThread = NCCL_NAN_ELEMS_PER_THREAD;

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
      // The credit also certifies that the peer has restored the sentinel in the
      // slot we are about to write. Without that the payload would be racing the
      // re-clear and the receiver could read our data back as empty.
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
        __builtin_amdgcn_s_sleep(1);
        sendConnHeadCache = __atomic_load_n(sendConnHeadPtr, __ATOMIC_RELAXED);
        if (checkAbort(abort, 1, spins)) break;
      }
      if (sendConnFifo) {
        sendConnFifo[sendStep[wid] % NCCL_STEPS].size = nbytes;
      }
      sendConnHead += 1;
    }
  }

  inline __device__ void postRecv() {
    if (recvConnHeadPtr) {
      // Release the sentinel writes before the credit that acts on them, the same
      // way postSend releases payload before the tail.
      __threadfence_system();
      STORE(recvConnHeadPtr, recvConnHead += 1);
    }
  }
  inline __device__ void postSend() {
    if (sendConnTailPtr) {
      __threadfence_system();
      STORE((unsigned long long*)sendConnTailPtr, sendConnTail += 1);
    }
  }

  // True if any element packed into this wire word is NaN, i.e. the word has not
  // been written yet (wholly or in part) this step.
  __device__ __forceinline__ static bool anyNan(uint64_t word) {
    using Bits = typename ncclNanBits<T>::Bits;
    bool nan = false;
#pragma unroll
    for (int i = 0; i < EltPerWord; i++) {
      Bits bits = (Bits)(word >> (i * 8 * sizeof(T)));
      nan |= ((bits & ncclNanBits<T>::Exp) == ncclNanBits<T>::Exp) && ((bits & ncclNanBits<T>::Mant) != 0);
    }
    return nan;
  }

  // Gather the EltPerWord slice elements starting at eltBase into one wire word.
  // Slots past eltN are zero-filled rather than read from src: reading past the
  // caller's buffer could pull in a NaN, which would strand the receiver on a word
  // it can never see complete. Zero is finite in every supported dtype.
  __device__ __forceinline__ uint64_t packWord(T const* src, int eltBase, int eltN) {
    union {
      uint64_t word;
      T elt[EltPerWord];
    } u;
    if (eltBase + EltPerWord <= eltN && (reinterpret_cast<uintptr_t>(src + eltBase) % sizeof(uint64_t)) == 0) {
      return *reinterpret_cast<uint64_t const*>(src + eltBase);
    }
    u.word = 0;
#pragma unroll
    for (int i = 0; i < EltPerWord; i++) {
      if (eltBase + i < eltN) u.elt[i] = src[eltBase + i];
    }
    return u.word;
  }

  // Scatter one wire word back out to dst, dropping the zero-filled tail.
  __device__ __forceinline__ void unpackWord(T* dst, uint64_t word, int eltBase, int eltN) {
    if (eltBase + EltPerWord <= eltN && (reinterpret_cast<uintptr_t>(dst + eltBase) % sizeof(uint64_t)) == 0) {
      *reinterpret_cast<uint64_t*>(dst + eltBase) = word;
      return;
    }
    union {
      uint64_t w;
      T elt[EltPerWord];
    } u;
    u.w = word;
#pragma unroll
    for (int i = 0; i < EltPerWord; i++) {
      if (eltBase + i < eltN) dst[eltBase + i] = u.elt[i];
    }
  }

  // Wire words needed to carry eltN elements of this slice.
  __device__ __forceinline__ static int wordsForElts(int eltN) {
    return divUp(eltN, EltPerWord);
  }

  __device__ __forceinline__ void loadRegs(uint64_t (&regs)[WordPerThread], T const* src, int eltN) {
    int nWords = wordsForElts(eltN);
#pragma unroll
    for (int u = 0; u < WordPerThread; u++) {
      int w = u * WARP_SIZE + wid;
      if (w < nWords) regs[u] = packWord(src, w * EltPerWord, eltN);
    }
  }

  __device__ __forceinline__ void storeRegs(T* dst, uint64_t (&regs)[WordPerThread], int eltN) {
    int nWords = wordsForElts(eltN);
#pragma unroll
    for (int u = 0; u < WordPerThread; u++) {
      int w = u * WARP_SIZE + wid;
      if (w < nWords) unpackWord(dst, regs[u], w * EltPerWord, eltN);
    }
  }

  // Spin until every live word this lane owns has left the sentinel, then reduce
  // and forward. Unlike LL128 there is no reload after the spin: each lane
  // validates exactly the words it keeps, so the values that ended the loop are
  // the values it goes on to use.
  template <int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void recvReduceSendCopy(uint64_t (&v)[WordPerThread], int wireOffset, int eltN,
                                                     bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    uint64_t vr[WordPerThread];
    int nWords = wordsForElts(eltN);

    __syncwarp();

    if (RECV) {
      for (int i = 0; i < MaxRecv && i < fan.nrecv(); i++) {
        uint64_t* ptr = recvPtr(i) + wireOffset;
        bool needReload;
        int spins = 0;
        do {
          needReload = false;
#pragma unroll
          for (int u = 0; u < WordPerThread; u++) {
            int w = u * WARP_SIZE + wid;
            if (w < nWords) {
              vr[u] = loadNanWord(ptr + u * WARP_SIZE);
              needReload |= anyNan(vr[u]);
            }
          }
          needReload &= (0 == checkAbort(abort, 1, spins));
        } while (__any(needReload));

        // Hand the slot back to the sentinel while the lines are still hot. The
        // credit published by postRecv() is what makes this safe to do here: the
        // peer cannot touch this slot again until it sees that credit.
#pragma unroll
        for (int u = 0; u < WordPerThread; u++) {
          int w = u * WARP_SIZE + wid;
          if (w < nWords) storeNanWord(ptr + u * WARP_SIZE, NCCL_NAN_SENTINEL64);
        }

#pragma unroll
        for (int u = 0; u < WordPerThread; u++) {
          int w = u * WARP_SIZE + wid;
          // The first peer seeds v[] when there is no local source to reduce
          // against; every peer after that accumulates.
          if (w < nWords) v[u] = (SRC || i > 0) ? applyReduce(redOp, vr[u], v[u]) : vr[u];
        }
      }
    }

    if (postOp) {
#pragma unroll
      for (int u = 0; u < WordPerThread; u++) {
        int w = u * WARP_SIZE + wid;
        if (w < nWords) v[u] = applyPostOp(redOp, v[u]);
      }
    }

    if (SEND) {
      for (int i = 1; i < MaxSend && i < fan.nsend(); i++) {
        uint64_t* ptr = sendPtr(i) + wireOffset;
#pragma unroll
        for (int u = 0; u < WordPerThread; u++) {
          int w = u * WARP_SIZE + wid;
          if (w < nWords) storeNanWord(ptr + u * WARP_SIZE, v[u]);
        }
      }
      uint64_t* ptr = sendPtr(0) + wireOffset;
#pragma unroll
      for (int u = 0; u < WordPerThread; u++) {
        int w = u * WARP_SIZE + wid;
        if (w < nWords) storeNanWord(ptr + u * WARP_SIZE, v[u]);
      }
    }
  }

  static constexpr int WireWordPerSlice = WARP_SIZE * WordPerThread;
  static constexpr int DataEltPerSlice = WireWordPerSlice * EltPerWord;

  template <int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void GenericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    constexpr int DST = DstBuf != -1 ? 1 : 0;
    T const* srcPtr = SrcBuf == -1 ? nullptr : userBufs[SrcBuf] + srcIx;
    T* dstPtr = DstBuf == -1 ? nullptr : userBufs[DstBuf] + dstIx;
    T* accPtr = (DstBuf == -1 || !useAcc) ? nullptr : userBufs[Acc] + dstIx;
    int wireOffset = WireWordPerSlice * warp + wid;
    const int nwarps = nthreads / WARP_SIZE;
    nelem = nelem < 0 ? 0 : nelem;

    if (SEND) waitSend(divUp(nelem, DataEltPerSlice) * WireWordPerSlice * sizeof(uint64_t));
    barrier();

    sqtt_marker_enter("PRIM_NAN_DATA_PROCESS");
    nelem -= DataEltPerSlice * warp;
    srcPtr += DataEltPerSlice * warp;
    dstPtr += DataEltPerSlice * warp;
    if (accPtr != nullptr) accPtr += DataEltPerSlice * warp;
    while (nelem > 0) {
      const int eltInSlice = min(nelem, DataEltPerSlice);
      uint64_t regs[WordPerThread];
      if (SRC) {
        loadRegs(regs, srcPtr, eltInSlice);
        if (SrcBuf == Input) {
          int nWords = wordsForElts(eltInSlice);
#pragma unroll
          for (int u = 0; u < WordPerThread; u++) {
            if (u * WARP_SIZE + wid < nWords) regs[u] = applyPreOp(redOp, regs[u]);
          }
        }
      }
      recvReduceSendCopy<RECV, SEND, SrcBuf, DstBuf>(regs, wireOffset, eltInSlice, postOp);
      if (DST) {
        if (accPtr != nullptr) {
          uint64_t accRegs[WordPerThread];
          loadRegs(accRegs, accPtr, eltInSlice);
          accPtr += DataEltPerSlice * nwarps;
          int nWords = wordsForElts(eltInSlice);
#pragma unroll
          for (int u = 0; u < WordPerThread; u++) {
            if (u * WARP_SIZE + wid < nWords) regs[u] = applyReduce(redOp, accRegs[u], regs[u]);
          }
        }
        storeRegs(dstPtr, regs, eltInSlice);
      }

      wireOffset += WireWordPerSlice * nwarps;
      srcPtr += DataEltPerSlice * nwarps;
      dstPtr += DataEltPerSlice * nwarps;
      nelem -= DataEltPerSlice * nwarps;
    }

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
      stepSize(ncclShmem.comm.buffSizes[NCCL_PROTO_NAN] / NCCL_STEPS / sizeof(uint64_t)), warp(tid / WARP_SIZE),
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
