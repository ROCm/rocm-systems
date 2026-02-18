/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#if defined(ENABLE_NPKIT)
#include "npkit/npkit.h"
#endif

#define NCCL_LL128_FLAGTHREAD (NCCL_LL128_LINEELEMS-1)

#ifndef RCCL_USE_WBINVL1_VOL
#if defined(__GFX8__) || defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__)
#define RCCL_USE_WBINVL1_VOL 1
#else
#define RCCL_USE_WBINVL1_VOL 0
#endif
#endif

template<typename T, typename RedOp, typename Fan, int Direct, int P2p, bool isNetOffload, int Metadata, int Pipeline, int useAcc>
class Primitives<T, RedOp, Fan, Direct, ProtoLL128, P2p, isNetOffload, Metadata, Pipeline, useAcc>:
  public PrimitivesWithoutDirect<Primitives<T, RedOp, Fan, Direct, ProtoLL128, P2p, isNetOffload, Metadata, Pipeline, useAcc>> {

  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input=0, Output=1, Acc=2;;
  RedOp redOp;
  const int tid;
  const int nthreads;
  const int wid;
  const int stepSize;
  const int warp;
  const int warpInBlock; // warp index in thread block
  const bool flagThread;
  const int group;
  const int threadsPerBlock;
  Fan fan;
  T *userBufs[3];
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;

  struct ncclConnInfo* sendConn = NULL;
  volatile struct ncclConnFifo* sendConnFifo = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[MaxRecv];
  uint64_t sendStep[MaxSend];
  uint64_t* recvBuff[MaxRecv];
  uint64_t* sendBuff[MaxSend];

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ uint64_t* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
  inline __device__ uint64_t* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
  inline __device__ uint64_t recvFlag(int i) { return recvStep[i]+1; }
  inline __device__ uint64_t sendFlag(int i) { return sendStep[i]+1; }

  uint64_t* barriers;
  uint64_t barrier_next = 0;

#if defined(ENABLE_NPKIT)
public:
  int npKitCtxIdx = 0;
  uint64_t npKitDataProcessEntryTime = 0;
  uint64_t npKitDataProcessExitTime = 0;
  uint64_t npKitDataProcessTotalTime = 0;
private:
#endif

#if defined(ENABLE_NPKIT) && (defined(ENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_ENTRY) && defined(ENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_EXIT) || defined(ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME))
  uint64_t npKitWaitRecvDataProcessSize = 0;
  uint64_t npKitWaitRecvEntryTime = 0;
  uint64_t npKitWaitRecvExitTime = 0;
  uint64_t npKitWaitRecvTotalTime = 0;
#endif

  inline __device__ void barrier() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (nthreads != WARP_SIZE)
    #if defined(__gfx942__) || defined(__gfx950__)
      barrier_generic(__threadfence_block(), nthreads, barrier_next, barriers);
    #else
      barrier_generic(__threadfence(), nthreads, barrier_next, barriers);
    #endif
#else
   barrier_sync(15-group, nthreads);
#endif
  }

  int abort = 0;

  __device__ inline int checkAbort(int &abortCache, const int abortValue, int &spins) {
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
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_PRIM_LL128_WAIT_SEND_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_PRIM_LL128_WAIT_SEND_ENTRY, nbytes, 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    if (sendConnHeadPtr) {
      int spins = 0;
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
        __builtin_amdgcn_s_sleep(1);
        sendConnHeadCache = __atomic_load_n(sendConnHeadPtr, __ATOMIC_RELAXED);
        if (checkAbort(abort, 1, spins)) break;
      }
      if (sendConnFifo) {
        sendConnFifo[sendStep[wid]%NCCL_STEPS].size = nbytes;
      }
      sendConnHead += 1;
    }
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_PRIM_LL128_WAIT_SEND_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_PRIM_LL128_WAIT_SEND_EXIT, nbytes, 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }

  inline __device__ void postRecv() {
    if (recvConnHeadPtr) STORE(recvConnHeadPtr, recvConnHead += 1);
  }
  inline __device__ void postSend() {
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
    __atomic_signal_fence(__ATOMIC_SEQ_CST);

    if (sendConnTailPtr) {
#if __CUDA_ARCH__ >= 900
      __threadfence_system();
#endif
      STORE((unsigned long long *)sendConnTailPtr, sendConnTail += 1);
    }
  }

  template<int WordPerThread>
  __device__ __forceinline__ void loadRegsBegin(uint64_t(&regs)[WordPerThread], T const *src, int eltN) {
    constexpr int EltPer16B = 16/sizeof(T);
    int ix[WordPerThread/2];
    #pragma unroll
    for(int g=0; g < WordPerThread/2; g++) {
      ix[g] = g*WARP_SIZE - 16*(g/2) + wid - (g%2)*(wid/4);
    }
    if(reinterpret_cast<uintptr_t>(src)%16 == 0) {
      /* We are aligned to 16 bytes, so load directly to registers no shmem.
       * Flag threads load half as much data which gets shuffled to the even
       * registers during Finish. The point of splitting into two phases is to
       * defer that shuffle, which incurs a dependency stall, until after other
       * memops are launched by the caller.
       */
      #pragma unroll
      for(int g=0; g < WordPerThread/2; g++) {
        if(!flagThread || g%2==0) {
          if(ix[g]*EltPer16B < eltN)
            load128((uint64_t*)(src + ix[g]*EltPer16B), regs[2*g+0], regs[2*g+1]);
        }
      }
    }
    else {
      // Not aligned. Stage the smallest 16 byte aligned region subsuming the
      // buffer into shmem.
      int misalignment = reinterpret_cast<uintptr_t>(src) % 16;
      uint64_t *src8 = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(src) & -uintptr_t(16));
      uint64_t *shm8 = shmemCvtPtr((uint64_t*)ncclScratchForWarp(warpInBlock));
      #pragma unroll
      for(int g=0; g < WordPerThread/2; g++)
        if((g*WARP_SIZE + wid)*16 < misalignment + eltN*sizeof(T))
          load128(src8 + 2*(g*WARP_SIZE + wid), regs[2*g+0], regs[2*g+1]);
      #pragma unroll
      for(int g=0; g < WordPerThread/2; g++)
        storeShmem128(shm8 + 2*(g*WARP_SIZE + wid), regs[2*g+0], regs[2*g+1]);

      __syncwarp();

      // Now load from shmem stage to regs. Preserve the same pre-shuffled layout
      // as the aligned case since Finish() will be applied regardless.
      T *shm = (T*)shm8 + misalignment/sizeof(T);
      #pragma unroll
      for(int g=0; g < WordPerThread/2; g++) {
        // int ix = g*WARP_SIZE - 16*(g/2) + wid - (g%2)*(wid/4);
        if(!flagThread || g%2==0) {
          if(ix[g]*EltPer16B < eltN)
            loadShmemMisaligned128(shm + ix[g]*EltPer16B, regs[2*g+0], regs[2*g+1]);
        }
      }
    }
  }

  template<int WordPerThread>
  __device__ __forceinline__ void loadRegsFinish(uint64_t(&regs)[WordPerThread]) {
    // Move data out of flag registers into the vacant registers.
    #pragma unroll
    for (int g=1; g < WordPerThread/2; g+=2) {
      if (flagThread) regs[2*g] = regs[2*g-1];
    }
  }

  template<int WordPerThread>
  __device__ __forceinline__ void storeRegs(T *dst, uint64_t(&regs)[WordPerThread], int eltN) {
    constexpr int EltPer16B = 16/sizeof(T);
    // Reverse Finish() register permuatation.
    #pragma unroll
    for (int g=1; g < WordPerThread/2; g+=2) {
      if (flagThread) regs[2*g-1] = regs[2*g];
    }

    // Write to dst if 4-byte aligned, shmem otherwise.
    int misalignment = reinterpret_cast<uintptr_t>(dst)%16;
    uint64_t *shm8 = shmemCvtPtr((uint64_t*)ncclScratchForWarp(warpInBlock));
    #pragma unroll
    for(int g=0; g < WordPerThread/2; g++) {
      int ix = g*WARP_SIZE - 16*(g/2) + wid - (g%2)*(wid/4);
      if (!flagThread || g%2==0) {
        if(misalignment == 0 && (ix+1)*EltPer16B <= eltN)
          store128((uint64_t*)(dst + ix*EltPer16B), regs[2*g+0], regs[2*g+1]);
        else
          storeShmem128(shm8+2*ix, regs[2*g+0], regs[2*g+1]);
      }
    }
    __syncwarp();
    // Write rest from shmem to dst. No need to coalesce stores to 16-bytes,
    // the hardware keeps up fine.
    T *shm = (T*)ncclScratchForWarp(warpInBlock);
    int skip = misalignment == 0 ? eltN & -EltPer16B : 0;
    for(int i=skip+wid; i < eltN; i += WARP_SIZE)
      dst[i] = shm[i];
  }

  #define WARP_MASK 0xffffffff

  template <int ELEMS_PER_THREAD, int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void recvReduceSendCopy(uint64_t(&v)[ELEMS_PER_THREAD], int ll128Offset, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    uint64_t vr[ELEMS_PER_THREAD];

    __syncwarp();
    /************************ Wait first recv ********************/
    if (RECV) {
      uint64_t* ptr = recvPtr(0)+ll128Offset;
      uint64_t flag = recvFlag(0);
      bool needReload;
      int spins = 0;
      do {
        needReload = false;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          load128(ptr+u*WARP_SIZE, vr[u], vr[u+1]);
          needReload |= flagThread && (vr[u+1] != flag);
        }
        needReload &= (0 == checkAbort(abort, 1, spins));
      } while (__any(needReload));
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2)
        load128(ptr+u*WARP_SIZE, vr[u], vr[u+1]);
    }

    /************* Finish register load **************/
    if (SRC) {
      // By deferring register shuffle here we've overlapped spinning on first
      // peer's data with memory loads of src data.
      loadRegsFinish(v);
      if (SrcBuf == Input) {
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u] = applyPreOp(redOp, v[u]);
          if (!flagThread)
            v[u+1] = applyPreOp(redOp, v[u+1]);
        }
      }
    }

    /************************ Recv rest *********************/
    if (RECV) {
      { // Consume data from first recv
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u]   = SRC ? applyReduce(redOp, vr[u], v[u]) : vr[u];
          v[u+1] = SRC ? applyReduce(redOp, vr[u+1], v[u+1]) : vr[u+1];
        }
      }

      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      for (int i=1; i<MaxRecv && i<fan.nrecv(); i++) {
        uint64_t flag = recvFlag(i);
        uint64_t* ptr = recvPtr(i)+ll128Offset;
        bool needReload;
        int spins = 0;
        do {
          needReload = false;
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            load128(ptr+u*WARP_SIZE, vr[u], vr[u+1]);
            needReload |= flagThread && (vr[u+1] != flag);
          }
          needReload &= (0 == checkAbort(abort, 1, spins));
        } while (__any(needReload));

        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2)
          load128(ptr+u*WARP_SIZE, vr[u], vr[u+1]);

        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u]   = applyReduce(redOp, vr[u], v[u]);
          v[u+1] = applyReduce(redOp, vr[u+1], v[u+1]);
        }
      }
    }
    /********************** End Recv ************************/

    if (postOp) {
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        v[u]   = applyPostOp(redOp, v[u]);
        v[u+1] = applyPostOp(redOp, v[u+1]);
      }
    }

#if RCCL_USE_WBINVL1_VOL
    if (tid == 0) __builtin_amdgcn_buffer_wbinvl1();
#endif
    /************************ Send **************************/
    if (SEND) {
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      for (int i=1; i<MaxSend && i<fan.nsend(); i++) {
        uint64_t flag = sendFlag(i);
        uint64_t* ptr = sendPtr(i)+ll128Offset;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
        }
      }
      uint64_t flag = sendFlag(0);
      uint64_t* ptr = sendPtr(0)+ll128Offset;
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
      }
    }
    /********************** End Send ************************/
  }

  static constexpr int WireWordPerSlice = WARP_SIZE*NCCL_LL128_SHMEM_ELEMS_PER_THREAD;
  static constexpr int DataEltPerSlice = (WireWordPerSlice - WireWordPerSlice/NCCL_LL128_LINEELEMS)*(sizeof(uint64_t)/sizeof(T));

  template <int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void GenericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    constexpr int DST = DstBuf != -1 ? 1 : 0;
    T const *srcPtr = SrcBuf == -1 ? nullptr : userBufs[SrcBuf] + srcIx;
    T       *dstPtr = DstBuf == -1 ? nullptr : userBufs[DstBuf] + dstIx;
    T       *accPtr = (DstBuf == -1 || !useAcc) ? nullptr : userBufs[Acc] + dstIx;
    int wireOffset = WireWordPerSlice*warp + 2*wid;
    const int nwarps = nthreads/WARP_SIZE;
    nelem = nelem < 0 ? 0 : nelem;

    if (SEND) waitSend(divUp(nelem, DataEltPerSlice)*WireWordPerSlice*sizeof(uint64_t));
    barrier();

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_ENTRY) && defined(ENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_EXIT)
    if (tid == 0) {
      npKitWaitRecvTotalTime = 0;
      npKitWaitRecvDataProcessSize = nelem*sizeof(T);
      NpKit::CollectGpuEvent(NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_ENTRY,
          npKitWaitRecvDataProcessSize, 0, NPKIT_GET_GPU_TIMESTAMP(), ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME)
    if (tid == 0) {
      npKitWaitRecvTotalTime = 0;
      npKitDataProcessEntryTime = NPKIT_GET_GPU_TIMESTAMP();
    }
#endif

    nelem -= DataEltPerSlice*warp;
    srcPtr += DataEltPerSlice*warp;
    dstPtr += DataEltPerSlice*warp;
    if (accPtr != nullptr) accPtr += DataEltPerSlice*warp;
    while (nelem > 0) {
      const int eltInSlice = min(nelem, DataEltPerSlice);
      uint64_t regs[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
      if (SRC) loadRegsBegin(regs, srcPtr, eltInSlice);
      recvReduceSendCopy<NCCL_LL128_SHMEM_ELEMS_PER_THREAD, RECV, SEND, SrcBuf, DstBuf>(regs, wireOffset, postOp);
      if (DST) {
        if (accPtr != nullptr) {
          uint64_t accRegs[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
          loadRegsBegin(accRegs, accPtr, eltInSlice);
          loadRegsFinish(accRegs);
          accPtr += DataEltPerSlice*nwarps;
          #pragma unroll
          for (int u=0; u<NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u++) {
            regs[u] = applyReduce(redOp, accRegs[u], regs[u]);
          }
        }
        storeRegs(dstPtr, regs, eltInSlice);
      }

      wireOffset += WireWordPerSlice*nwarps;
      srcPtr += DataEltPerSlice*nwarps;
      dstPtr += DataEltPerSlice*nwarps;
      nelem -= DataEltPerSlice*nwarps;
    }

    barrier();

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME)
    if (tid == 0) {
      npKitDataProcessExitTime = NPKIT_GET_GPU_TIMESTAMP();
      npKitDataProcessTotalTime += npKitDataProcessExitTime - npKitDataProcessEntryTime - npKitWaitRecvTotalTime;
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_ENTRY) && defined(ENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_EXIT,
          npKitWaitRecvDataProcessSize, npKitWaitRecvTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    if (SEND) for (int i=0; i < MaxSend; i++) sendStep[i] += 1;
    if (SEND) postSend();
    if (RECV) for (int i=0; i < MaxRecv; i++) recvStep[i] += 1;
    if (RECV) postRecv();
  }

  template <int REDUCE, int COPY, int MULTISRCS, int MULTIDSTS>
  __device__ __forceinline__ void mscclGenericOp(T** srcs, int nsrcs, T** dsts, int ndsts, int nelem) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_GENERIC_OP_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_MSCCL_GENERIC_OP_ENTRY, nelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    T const *srcPtr = srcs[0];
    T       *dstPtr = dsts[0];
    const int nwarps = nthreads/WARP_SIZE;
    nelem = nelem < 0 ? 0 : nelem;

    nelem -= DataEltPerSlice*warp;
    srcPtr += DataEltPerSlice*warp;
    dstPtr += DataEltPerSlice*warp;
    if (MULTISRCS){
      for (int i = 1; i < nsrcs; i++){
        srcs[i] += DataEltPerSlice*warp;
      }
    }
    if (MULTIDSTS){
      for (int i = 1; i < ndsts; i++){
        dsts[i] += DataEltPerSlice*warp;
      }
    }
    while (nelem > 0) {
      const int eltInSlice = min(nelem, DataEltPerSlice);
      uint64_t regs[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
      loadRegsBegin(regs, srcPtr, eltInSlice);
      loadRegsFinish(regs);
      if (REDUCE){
        uint64_t regsD[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
        loadRegsBegin(regsD, dstPtr, eltInSlice);
        loadRegsFinish(regsD);
        #pragma unroll
        for (int u=0; u<NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u+=2) {
          regsD[u] = applyReduce(redOp, regs[u], regsD[u]);
          if (!flagThread)
            regsD[u+1] = applyReduce(redOp, regs[u+1], regsD[u+1]);
        }
        if (MULTISRCS){
          for (int i = 1; i < nsrcs; i++){
            loadRegsBegin(regs, srcs[i], eltInSlice);
            loadRegsFinish(regs);
            for (int u=0; u<NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u+=2) {
              regsD[u] = applyReduce(redOp, regs[u], regsD[u]);
              if (!flagThread)
                regsD[u+1] = applyReduce(redOp, regs[u+1], regsD[u+1]);
            }
          }
        }
        storeRegs(dstPtr, regsD, eltInSlice);
      }
      if (COPY){
        storeRegs(dstPtr, regs, eltInSlice);
        if (MULTIDSTS){
          for (int i = 1; i < nsrcs; i++){
            loadRegsBegin(regs, srcs[i], eltInSlice);
            loadRegsFinish(regs);
            storeRegs(dsts[i], regs, eltInSlice);
          }
        }
      }

      srcPtr += DataEltPerSlice*nwarps;
      dstPtr += DataEltPerSlice*nwarps;
      if (MULTISRCS){
        for (int i = 1; i < nsrcs; i++){
          srcs[i] += DataEltPerSlice*nwarps;
        }
      }
      if (MULTIDSTS){
        for (int i = 1; i < ndsts; i++){
          dsts[i] += DataEltPerSlice*nwarps;
        }
      }
      nelem -= DataEltPerSlice*nwarps;
    }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_GENERIC_OP_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_MSCCL_GENERIC_OP_EXIT, nelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    barrier();
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
    recvStep[i] = conn->step;
    if (wid == i) recvConn = conn;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < fan.nrecv()) {
      recvConnHeadPtr = recvConn->head;
      recvConnHead = recvConn->step;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
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
    if (tid >= nthreads-WARP_SIZE && wid<fan.nsend()) {
      if (sendConn->connFifo) {
        sendConnTailPtr = sendConn->tail;
        sendConnTail = sendConn->step;
      }
    }
  }

public:
  __device__ Primitives(
      const int tid, const int nthreads, int const *recvPeers, int const *sendPeers,
      void const *inputBuf, void *outputBuf, uint64_t redOpArg, uint8_t group=0,
      uint8_t connIndexRecv=0, uint8_t connIndexSend=0, struct ncclDevWorkColl* e = nullptr,
      bool ipcReg = false, bool netReg = false, int stepSize_ = 0
    ):
    redOp(redOpArg),
    tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE),                                /*compiler warnings*/
    stepSize(ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS/sizeof(uint64_t)),
    warp(tid/WARP_SIZE), warpInBlock(threadIdx.x/WARP_SIZE), flagThread((tid%4)==3), group(group), threadsPerBlock(blockDim.x){
#ifdef ENABLE_WARP_SPEED
    auto *channel = isMsccl(Metadata) ? &ncclShmem.channel : &ncclShmem.warpChannel[warpInBlock];
#else
    auto *channel = &ncclShmem.channel;
#endif
    barriers = &ncclShmem.groups[group].barrier;
    int nrecv=0, nsend=0;
    while (nrecv < MaxRecv && recvPeers[nrecv] >= 0) {
      loadRecvConn(&channel->peers[recvPeers[nrecv]]->recv[connIndexRecv], nrecv);
      nrecv++;
    }
    while (nsend < MaxSend && sendPeers[nsend] >= 0) {
      loadSendConn(&channel->peers[sendPeers[nsend]]->send[connIndexSend], nsend);
      nsend++;
    }
    this->fan = Fan(nrecv, nsend);
    // Coverity reports recvConn and sendConn being possibly NULL at this point but that won't actually
    // happen given the two "while" loops just above.
    // coverity[var_deref_model:FALSE]
    loadRecvSync();
    // coverity[var_deref_model:FALSE]
    loadSendSync();
    setDataPtrs(inputBuf, outputBuf, e != nullptr ? e->acc : nullptr);
  }

  __device__ ~Primitives() {
    // Save steps for the next operation
    if (tid >= nthreads-WARP_SIZE && wid < fan.nrecv())
      recvConn->step = recvConnHead;
    if (tid < fan.nsend())
      sendConn->step = sendConnHead;
    // Ensure all steps written back
    barrier();
  }

  __device__ void setDataPtrs(void const *inputBuf, void *outputBuf, void const *acc = nullptr) {
    userBufs[Input] = (T*)inputBuf;
    userBufs[Output] = (T*)outputBuf;
    userBufs[Acc] = (T*)acc;
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    userBufs[Input] += delta;
    userBufs[Output] += delta;
  }

  __device__ void send(intptr_t inpIx, int eltN) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    GenericOp<0, 1, Input, -1>(inpIx, -1, eltN, false);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void sendFromOutput(intptr_t outIx, int eltN) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_FROM_OUTPUT_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_FROM_OUTPUT_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    GenericOp<0, 1, Output, -1>(outIx, -1, eltN, false);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_FROM_OUTPUT_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_FROM_OUTPUT_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    GenericOp<1, 0, -1, Output>(-1, outIx, eltN, postOp);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void recvReduceSend(intptr_t inpIx, int eltN) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_REDUCE_SEND_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_REDUCE_SEND_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    GenericOp<1, 1, Input, -1>(inpIx, -1, eltN, false);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_REDUCE_SEND_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_REDUCE_SEND_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_REDUCE_COPY_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    GenericOp<1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_REDUCE_COPY_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_COPY_SEND_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_COPY_SEND_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    GenericOp<0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_COPY_SEND_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_COPY_SEND_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_COPY_SEND_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_COPY_SEND_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    GenericOp<1, 1, -1, Output>(-1, outIx, eltN, postOp);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_COPY_SEND_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_COPY_SEND_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
  #if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_SEND_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_REDUCE_COPY_SEND_ENTRY, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    GenericOp<1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_SEND_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_RECV_REDUCE_COPY_SEND_EXIT, eltN*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
  __device__ void recvSend(int eltN) {
    return GenericOp<1, 1, -1, -1>(-1, -1, eltN, false);
  }

  // MSCCL primitives
  __device__ void sendWithBarrier(intptr_t inpIx, int eltN) {
    send(inpIx, eltN);
  }
  __device__ void localCopy(T* srcs, T* dsts, int eltN) {
    return mscclGenericOp<0,1,0,0>(&srcs, 1, &dsts, 1, eltN);
  }
};
