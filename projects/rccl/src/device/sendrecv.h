/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"
#if defined(ENABLE_NPKIT)
#include "npkit/npkit.h"
#endif

template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  static_assert(sizeof(T)==1, "SendRecv only works on single byte types T.");

  template<typename Proto>
  __device__ void runSend(int tid, int tn, int group, struct ncclDevWorkP2p* work) {
    size_t bytes = work->sendBytes;
    bool useLargeChunk = (work->sendIpcReg && ncclShmem.comm.isAllNvlink) || work->sendNetReg;
    int chunkSize = useLargeChunk ? NCCL_MAX_NET_SIZE : u32fp8Decode(work->sendChunkSize_u32fp8);
    int stepSize = useLargeChunk ? NCCL_MAX_NET_SIZE : ncclShmem.comm.p2pChunkSize;

#if defined(ENABLE_NPKIT)
    bool isNpKitThread = (tid == 0);
    int npKitCtxIdx = blockIdx.x * NCCL_MAX_DEV_WORK_P2P_ELEMENTS + group;
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
    if (isNpKitThread) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, NPKIT_GET_CPU_TIMESTAMP_FROM_BLOCK,
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
    if (isNpKitThread) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    Primitives<T, RedOp, FanAsymmetric<0, 1>, 0, Proto, 1>
      prims(tid, tn, nullptr, &work->sendRank, work->sendAddr, nullptr,
            /*redOpArg(ignored)=*/0, group, work->sendConnIndex, work->sendConnIndex, nullptr, work, stepSize);

#if defined(ENABLE_NPKIT)
      if (isNpKitThread) {
        prims.npKitCtxIdx = npKitCtxIdx;
      }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_RECV_SEND_ENTRY)
      if (isNpKitThread) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_RECV_SEND_ENTRY, bytes*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

    size_t cursor = 0;
    do {
      int n = min(size_t(chunkSize), bytes-cursor);
      prims.directSend(cursor, cursor, n);
      cursor += n;
    } while (cursor < bytes);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_RECV_SEND_EXIT)
      if (isNpKitThread) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_RECV_SEND_EXIT, bytes*sizeof(T), prims.npKitDataProcessTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif
  }

  template<typename Proto>
  __device__ void runRecv(int tid, int tn, int group, struct ncclDevWorkP2p* work) {
    size_t bytes = work->recvBytes;
    bool useLargeChunk = (work->recvIpcReg && ncclShmem.comm.isAllNvlink) || work->recvNetReg;
    int chunkSize = useLargeChunk ? NCCL_MAX_NET_SIZE : u32fp8Decode(work->recvChunkSize_u32fp8);
    int stepSize = useLargeChunk ? NCCL_MAX_NET_SIZE : ncclShmem.comm.p2pChunkSize;

#if defined(ENABLE_NPKIT)
    bool isNpKitThread = (tid == 0);
    int npKitCtxIdx = blockIdx.x * NCCL_MAX_DEV_WORK_P2P_ELEMENTS + group;
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
    if (isNpKitThread) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, NPKIT_GET_CPU_TIMESTAMP_FROM_BLOCK,
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
    if (isNpKitThread) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    Primitives<T, RedOp, FanAsymmetric<1, 0>, 0, Proto, 1>
      prims(tid, tn, &work->recvRank, nullptr, nullptr, work->recvAddr,
            /*redOpArg(ignored)=*/0, group, work->recvConnIndex, work->recvConnIndex, nullptr, work, stepSize);

#if defined(ENABLE_NPKIT)
      if (isNpKitThread) {
        prims.npKitCtxIdx = npKitCtxIdx;
      }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_RECV_RECV_ENTRY)
      if (isNpKitThread) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_RECV_RECV_ENTRY, bytes*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

    size_t cursor = 0;
    do {
      int n = min(size_t(chunkSize), bytes-cursor);
      prims.directRecv(cursor, n);
      cursor += n;
    } while (cursor < bytes);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_SEND_RECV_RECV_EXIT)
      if (isNpKitThread) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_SEND_RECV_RECV_EXIT, bytes*sizeof(T), prims.npKitDataProcessTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif
  }

#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx942__) && !defined(__gfx950__)
  __device__  void run() {
#else
  __device__  __attribute__((noinline)) void run() {
#endif
    const int tid = threadIdx.x;
    const int tn = blockDim.x;
    const int wid = tid/WARP_SIZE;
    const int nWarps = tn/WARP_SIZE;
    const int lane = tid%WARP_SIZE;

    struct Shared {
      uint32_t workSendMask; // bitmasks of which work indices have send/recv
      uint32_t workRecvMask;
    };
    Shared* shared = (Shared*)ncclScratchForWarp(0);

    struct ncclDevWorkP2p* works = (ncclDevWorkP2p*)ncclShmem.workStorage;
    int nWorks = ncclShmem.nWorks;

    if (wid == 0) {
      // Modify the memory range of each work[] to reflect this channel's
      // partition of the work. Since integer divides are very heavy it's
      // best to do them all in one warp.
      int workIx = lane%16;
      int isSend = lane < 16 ? 0 : 1;
      bool hasWork = false;
      if (workIx < nWorks) {
        struct ncclDevWorkP2p* work = &works[workIx];
        size_t bytes = isSend ? work->sendBytes : work->recvBytes;
        int nParts = isSend ? work->nSendChannels : work->nRecvChannels;
        int part = ncclP2pChannelToPart(work->nP2pChannels, work->channelBase, ncclShmem.channelId, ncclShmem.comm.p2pnChannelsPerPeer, ncclShmem.comm.nNodes);
        hasWork = (part < nParts);
        if (nParts != 0) {
          size_t partBeg, partEnd;
          ncclP2pPartBounds(nParts, part, bytes, &partBeg, &partEnd);
          (isSend ? work->sendAddr : work->recvAddr) = (char*)(isSend ? work->sendAddr : work->recvAddr) + partBeg;
          (isSend ? work->sendBytes : work->recvBytes) = partEnd - partBeg;
        }
      }
      // Coverity reports a possible thread divergence due to not all threads participating in the collective.
      // However, the code ensures that the participation is on a per-warp basis.
      // coverity[device_thread_diverged:FALSE]
      uint32_t mask = __ballot(hasWork);
      if (lane == 0) {
        shared->workSendMask = mask>>16;
        shared->workRecvMask = mask & 0xffff;
      }
    }

    // The fastest way to compute a warp uniform division x/y in [0,32) is to
    // use each lane to guess a solution and count the ones that don't exceed
    // the numerator:
    //   __popc(__ballot_sync(~0u, y*(lane+1) <= x))
    // That takes 1/3 the time of standard division and about 3/4 the time of
    // approximate floating point division:
    //   __float2int_rd(__fdividef(float(x),float(y))).

    // nWarpPerWork = nWarps/nWorks
    int nWarpPerWork = __popcll(__ballot(nWorks*(lane+1) <= nWarps));
    int nRecvWarpPerWork = nWarpPerWork/2;
    int nSendWarpPerWork = nWarpPerWork - nRecvWarpPerWork;
    // This might reduce nWarpPerWork which is probably desirable. It is better
    // to have a balanced number of reading and writing threads even if that
    // leaves warps unused.
    nWarpPerWork = nSendWarpPerWork + nRecvWarpPerWork;
    // The work index this warp belongs to: workIx = wid/nWarpPerWork
    int workIx = __popcll(__ballot((lane+1)*nWarpPerWork <= wid));

    __syncthreads(); // Wait for works[] and shared->* to be updated by warp=0

    uint32_t workSendMask = shared->workSendMask;
    uint32_t workRecvMask = shared->workRecvMask;

    __syncthreads(); // release scratch space used by shared->*
    if (nWorks <= workIx) return;

    // Thread range for whole work (send & recv combined)
    int subtid = tid - workIx*nWarpPerWork*WARP_SIZE;
    int subtn = nWarpPerWork*WARP_SIZE;

    // A send primtive of sufficient size requires 2 cuda barrier ids.
    constexpr int nSendWarpsForExtraGroup = NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE/WARP_SIZE;
    // Count up all group ids used below this workIx:
    int group, extra;
    // Each recv gets one group id:
    group = __popcll(workRecvMask & ((1<<workIx)-1));
    // Sends accompanying recvs get one and maybe an extra:
    extra = (nSendWarpPerWork >= nSendWarpsForExtraGroup) ? 1 : 0;
    group += __popcll((workSendMask & workRecvMask) & ((1<<workIx)-1))*(1+extra);
    // Sends without recvs use more warps so compute extra accordingly:
    extra = (nWarpPerWork >= nSendWarpsForExtraGroup) ? 1 : 0;
    group += __popcll((workSendMask & ~workRecvMask) & ((1<<workIx)-1))*(1+extra);

    struct ncclDevWorkP2p* work = &works[workIx];
    bool hasSend = 1 & (workSendMask>>workIx);
    bool hasRecv = 1 & (workRecvMask>>workIx);
    bool isCopy = work->sendRank == ncclShmem.comm.rank;
    bool isSend = !hasRecv || (hasSend && subtid < nSendWarpPerWork*WARP_SIZE);

    if (!isCopy && hasSend && hasRecv) {
      // Translate thread ids to reflect just this send or recv as opposed to whole work.
      if (isSend) {
        subtn = nSendWarpPerWork*WARP_SIZE;
      } else {
        subtid -= nSendWarpPerWork*WARP_SIZE;
        subtn = nRecvWarpPerWork*WARP_SIZE;
        group += 1 + (nSendWarpPerWork >= nSendWarpsForExtraGroup ? 1 : 0);
      }
    }

    if (isCopy) {
#if defined(__gfx942__) || defined(__gfx950__)
      reduceCopy<COLL_UNROLL*2, 0, RedOp, T, 0,1,1, 0,1,1, /*PreOpSrcs=*/0>
        (subtid, subtn, 0, nullptr, false, 1, &work->sendAddr, 1, &work->recvAddr, (ssize_t)work->sendBytes);
#else
      reduceCopy<COLL_UNROLL, 0, RedOp, T, 0,1,1, 0,1,1, /*PreOpSrcs=*/0>
        (subtid, subtn, 0, nullptr, false, 1, &work->sendAddr, 1, &work->recvAddr, (ssize_t)work->sendBytes);
#endif
    } else if (isSend) {
      if (work->sendProtoLL) {
        runSend<ProtoLL>(subtid, subtn, group, work);
      } else {
#if defined(__gfx90a__)
        runSend<ProtoSimple<1,1,0,8>>(subtid, subtn, group, work);
#elif defined(__gfx908__) || defined(__gfx942__) || defined(__gfx950__)
        runSend<ProtoSimple<1,1,0,4>>(subtid, subtn, group, work);
#else
        runSend<ProtoSimple<1,1>>(subtid, subtn, group, work);
#endif
      }
    } else {
      if (work->recvProtoLL) {
        runRecv<ProtoLL>(subtid, subtn, group, work);
      } else {
#if defined(__gfx90a__)
        runRecv<ProtoSimple<1,1,0,8>>(subtid, subtn, group, work);
#elif defined(__gfx908__) || defined(__gfx942__) || defined(__gfx950__)
        runRecv<ProtoSimple<1,1,0,4>>(subtid, subtn, group, work);
#else
        runRecv<ProtoSimple<1,1>>(subtid, subtn, group, work);
#endif
      }
    }
  }
};
