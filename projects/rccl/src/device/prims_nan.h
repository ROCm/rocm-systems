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
// Scopes split by direction: a store lands in the peer's buffer (or has to beat
// the peer's next write, in the case of the sentinel restore) and goes out at
// system scope, while a load only reads this GPU's own FIFO and stays at agent
// scope.
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

// 128-bit FIFO load, one per lane. The receive FIFO is this GPU's own memory,
// so agent scope is enough to observe what the peer pushed into it.
inline __device__ void loadNanLine(const uint64_t* ptr, uint64_t& v0, uint64_t& v1) {
  union {
    v4u v;
    uint64_t u64[2];
  } u;
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  u.v = __builtin_amdgcn_global_load_b128((v4u_gptr)ptr, RCCL_AGENT_SYNCSCOPE);
#else
  u.u64[0] = __builtin_nontemporal_load((u64_gptr)ptr);
  u.u64[1] = __builtin_nontemporal_load((u64_gptr)ptr + 1);
#endif
  v0 = u.u64[0];
  v1 = u.u64[1];
}

// 128-bit FIFO store, one per lane. Payload lands in the peer's buffer, and the
// sentinel restore has to beat the peer's next write to the same slot, so both
// go out at system scope.
inline __device__ void storeNanLine(uint64_t* ptr, uint64_t v0, uint64_t v1) {
  union {
    v4u v;
    uint64_t u64[2];
  } u;
  u.u64[0] = v0;
  u.u64[1] = v1;
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  __builtin_amdgcn_global_store_b128((v4u_gptr)ptr, u.v, RCCL_SYSTEM_SYNCSCOPE);
#else
  *((v4u_gptr)ptr) = u.v;
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
      // Compiler barrier only: keep sentinel stores from floating past the credit.
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

  // True if this wire word has not been written yet, wholly or in part.
  __device__ __forceinline__ static bool anyNan(uint64_t word) {
    if constexpr (sizeof(T) <= 4) {
      // Dword granularity is the right unit for these dtypes. Stores land
      // dword-atomically, so a half-landed 16-byte line always leaves whole
      // dwords at the sentinel, and every element fits inside one dword. An
      // all-ones dword is NaN read as f16, bf16 or f32 alike, and NaN input is
      // forbidden, so this never fires on real data. Two compares replace the
      // per-element exponent/mantissa decode.
      return (uint32_t)word == 0xFFFFFFFFu || (uint32_t)(word >> 32) == 0xFFFFFFFFu;
    } else {
      // f64 spans both dwords and a finite double can legitimately carry an
      // all-ones low half, so it keeps the exponent/mantissa test. A torn store
      // leaves the high dword at the sentinel, which reads as NaN.
      using Bits = typename ncclNanBits<T>::Bits;
      Bits bits = (Bits)word;
      return ((bits & ncclNanBits<T>::Exp) == ncclNanBits<T>::Exp) && ((bits & ncclNanBits<T>::Mant) != 0);
    }
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

  // Wire word inside the warp's slice that register u maps to. A lane owns two
  // consecutive words per register pair so the pair is one 16-byte access.
  __device__ __forceinline__ int wordOf(int u) const {
    return (u & ~1) * WARP_SIZE + 2 * wid + (u & 1);
  }

  // The pair holding register u is driven whole whenever its first word carries
  // data. The second word may sit past eltN; it rides the same 16-byte access,
  // and both sides hold zero there so the receiver's NaN test still terminates.
  __device__ __forceinline__ bool pairLive(int u, int nWords) const {
    return wordOf(u & ~1) < nWords;
  }

  __device__ __forceinline__ void loadRegs(uint64_t (&regs)[WordPerThread], T const* src, int eltN) {
    int nWords = wordsForElts(eltN);
#pragma unroll
    for (int u = 0; u < WordPerThread; u++) {
      int w = wordOf(u);
      regs[u] = w < nWords ? packWord(src, w * EltPerWord, eltN) : 0;
    }
  }

  __device__ __forceinline__ void storeRegs(T* dst, uint64_t (&regs)[WordPerThread], int eltN) {
    int nWords = wordsForElts(eltN);
#pragma unroll
    for (int u = 0; u < WordPerThread; u++) {
      int w = wordOf(u);
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
          for (int u = 0; u < WordPerThread; u += 2) {
            if (pairLive(u, nWords)) {
              loadNanLine(ptr + u * WARP_SIZE, vr[u], vr[u + 1]);
              needReload |= anyNan(vr[u]) || anyNan(vr[u + 1]);
            }
          }
          needReload &= (0 == checkAbort(abort, 1, spins));
        } while (__any(needReload));

        // Hand the slot back to the sentinel while the lines are still hot --
        // measurably better than deferring it past the forward store. The credit
        // published by postRecv() is what makes this safe here: the peer cannot
        // touch this slot again until it sees that credit.
#pragma unroll
        for (int u = 0; u < WordPerThread; u += 2) {
          if (pairLive(u, nWords)) storeNanLine(ptr + u * WARP_SIZE, NCCL_NAN_SENTINEL64, NCCL_NAN_SENTINEL64);
        }

#pragma unroll
        for (int u = 0; u < WordPerThread; u++) {
          // The first peer seeds v[] when there is no local source to reduce
          // against; every peer after that accumulates.
          if (pairLive(u, nWords)) v[u] = (SRC || i > 0) ? applyReduce(redOp, vr[u], v[u]) : vr[u];
        }
      }
    }

    if (postOp) {
#pragma unroll
      for (int u = 0; u < WordPerThread; u++) {
        if (pairLive(u, nWords)) v[u] = applyPostOp(redOp, v[u]);
      }
    }

    if (SEND) {
      for (int i = 1; i < MaxSend && i < fan.nsend(); i++) {
        uint64_t* ptr = sendPtr(i) + wireOffset;
#pragma unroll
        for (int u = 0; u < WordPerThread; u += 2) {
          if (pairLive(u, nWords)) storeNanLine(ptr + u * WARP_SIZE, v[u], v[u + 1]);
        }
      }
      uint64_t* ptr = sendPtr(0) + wireOffset;
#pragma unroll
      for (int u = 0; u < WordPerThread; u += 2) {
        if (pairLive(u, nWords)) storeNanLine(ptr + u * WARP_SIZE, v[u], v[u + 1]);
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
    int wireOffset = WireWordPerSlice * warp + 2 * wid;
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
            if (pairLive(u, nWords)) regs[u] = applyPreOp(redOp, regs[u]);
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
            if (wordOf(u) < nWords) regs[u] = applyReduce(redOp, accRegs[u], regs[u]);
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
