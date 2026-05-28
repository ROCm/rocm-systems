/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto, bool isNetOffload = false>
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx942__) && !defined(__gfx950__)
  __device__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
#else
  __device__ __attribute__((noinline)) void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
#endif
#if defined(ENABLE_NPKIT)
    const int bid = ncclShmem.channelId - work->channelLo;
    int npKitCtxIdx = bid; // unused variable - compiler warning
#endif
    ncclRing *ring = &ncclShmem.channel.ring;
    const int *ringRanks = ring->userRanks;
    const int nranks = ncclShmem.comm.nRanks;
    ssize_t count, partOffset, partCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &partOffset, &partCount, &chunkCount);
    ssize_t offset;
    ssize_t dataOffset;
    int nelem;
    int rankDest;


#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, NPKIT_GET_CPU_TIMESTAMP_FROM_BLOCK,
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_ENTRY, count*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    int workNthreads;
    T *inputBuf = (T*)work->sendbuff;
    T *outputBuf = (T*)work->recvbuff;

    // If isNetOffload == true, we only use 1 warp to drive Ring algo/network communication
    // and the rest of warps proceed to copy src data into dst buffer in parallel when AG
    // is not in-place.
    if (isNetOffload) {
      workNthreads = WARP_SIZE;
      chunkCount = NCCL_MAX_NET_SIZE;
    } else {
      workNthreads = nthreads;
    }

    if (tid < workNthreads) {
      // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
      // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0, isNetOffload> prims
        (tid, workNthreads, &ring->prev, &ring->next, inputBuf, outputBuf, work->redOpArg, 0, work->connIndex, work->connIndex, work, NULL, isNetOffload ? NCCL_MAX_NET_SIZE : 0);

#if defined(ENABLE_NPKIT)
      if (tid == 0) {
        prims.npKitCtxIdx = npKitCtxIdx;
      }
#endif
      for (size_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkCount) {
        /////////////// begin AllGather steps ///////////////
        nelem = min(chunkCount, partCount - elemOffset);
        dataOffset = partOffset + elemOffset;

        // step 0: push data to next GPU
        rankDest = ringRanks[0];
        offset = dataOffset + rankDest * count;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_ENTRY)
        if (tid == 0) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_SEND_ENTRY, nelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
              ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
          prims.npKitDataProcessTotalTime = 0;
        }
#endif

        if ((inputBuf + dataOffset == outputBuf + offset) || isNetOffload) { // In place or onePPN
          prims.directSend(dataOffset, offset, nelem);
        } else {
          prims.directCopySend(dataOffset, offset, nelem);
        }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_EXIT)
        if (tid == 0) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_SEND_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
              ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_ENTRY)
        if (tid == 0 && nranks > 2) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_ENTRY, nelem*(nranks-2)*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
              ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
          prims.npKitDataProcessTotalTime = 0;
        }
#endif

        // k-2 steps: copy to next GPU
        for (int j = 1; j < nranks - 1; ++j) {
          rankDest = ringRanks[nranks - j];
          offset = dataOffset + rankDest * count;
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_EXIT)
        if (tid == 0 && nranks > 2) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_EXIT, nelem*(nranks-2)*sizeof(T), prims.npKitDataProcessTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
              ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        }
#endif

        // Make final copy from buffer to dest.
        rankDest = ringRanks[1];
        offset = dataOffset + rankDest * count;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_ENTRY)
        if (tid == 0) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_ENTRY, nelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
              ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
          prims.npKitDataProcessTotalTime = 0;
        }
#endif
        // Final wait/copy.
        prims.directRecv(offset, nelem);
  
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_EXIT)
        if (tid == 0) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
              ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        }
#endif



      }
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_EXIT, count*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif
    } else if (inputBuf != outputBuf + ringRanks[0] * count) {
      inputBuf = inputBuf + partOffset;
      outputBuf = outputBuf + partOffset + ringRanks[0] * count;
      reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>
        (tid - workNthreads, nthreads - workNthreads, work->redOpArg, &work->redOpArg, false, 1, (void**)&inputBuf, 1, (void**)&outputBuf, partCount);
    }
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
    // we have to wait for all warps before we can proceed to the next work;
    // otherwise, we can have contention if next work will use the outputBuf
    // in this work. We use bar 14 to avoid conflicts with prims barrier and
    // __syncthread().
    if (isNetOffload) barrier_sync(14, nThreads);
#endif
  }
}

#if defined(__gfx942__) || defined(__gfx950__) // Use a single slice per simple primitive for a single node on some GFX9 devices.
#define rcclAllGatherRunRingSimpleProtoImpl(tid, nthreads, work) \
  if(work->rcclUseOneSlice){ \
    runRing<T, RedOp, ProtoSimple<ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS_SINGLE_NODE, ALLGATHER_SLICESTEPS_SINGLE_NODE>, false>(tid, nthreads, work); \
  } else{ \
    runRing<T, RedOp, ProtoSimple<ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>, false>(tid, nthreads, work); \
  }
#else
#define rcclAllGatherRunRingSimpleProtoImpl(tid, nthreads, work) \
  runRing<T, RedOp, ProtoSimple<ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>, false>(tid, nthreads, work);
#endif

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    bool isNetOffload = false;
#else
    bool isNetOffload = work->isOneRPN && work->netRegUsed;
#endif
    if (isNetOffload)
      runRing<T, RedOp, ProtoSimple<1, 1>, true>(tid, nthreads, work);
    else{
      rcclAllGatherRunRingSimpleProtoImpl(tid, nthreads, work);
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<1, 1>;
    const int nranks = ncclShmem.comm.nRanks;
    const int rank = ncclShmem.comm.rank;
    size_t count, channelOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &channelOffset, &channelCount, &chunkCount);

    static constexpr int nworkers = NCCL_PAT_NWORKERS;
    struct ncclPatShmem* shmem = (struct ncclPatShmem*)ncclScratchForWarp(0);
    uint64_t pollCount = 0;
    (void)pollCount; // unused variable - compiler warning
    __syncthreads(); // Don't start using shared mem until everyone arrives
    for (int i=tid; i<NCCL_SHMEM_PAT_STEPS; i+=nthreads) shmem->patSteps[i].flags = 0;
    if (tid == 0) shmem->localAccSize = 0;
    if (tid == nworkers) shmem->parallelFactor = 0;
    __syncthreads();

    if (tid == nworkers) { // Algo computation thread
      PatAGAlgorithm<T> patAlgo(chunkCount*sizeof(T), NCCL_STEPS, NCCL_PAT_NWORKERS/WARP_SIZE, channelOffset, channelOffset + channelCount, count, chunkCount, rank, nranks);
      int parallelFactor = shmem->parallelFactor = patAlgo.getParallelFactor();
      (void)parallelFactor;// unused variable - compiler warning
      int step = 0;
      while (1) {
        struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
        int* poll = &ps->flags;
        while (__hip_atomic_load(poll, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) != 0) {
          pollCount++ ;// Wait for workers to be done with step 'step-NCCL_SHMEM_PAT_STEPS'
        }
        patAlgo.getNextOp(ps);
        int last = ps->last;
        step++;
        if (last == 2) break;
      }
    } else if (tid < nworkers) { // Worker threads
      T *inputBuf = (T*)work->sendbuff;
      T *outputBuf = (T*)work->recvbuff;
      int parallelFactor = 0;
      volatile int* pfPtr = &shmem->parallelFactor;
      while (parallelFactor == 0) parallelFactor = *pfPtr;

      int groupSize = nworkers/(WARP_SIZE*parallelFactor) * WARP_SIZE;
      int group = tid / groupSize;
      int nGroups = nworkers / groupSize;
      int tidInGroup = tid - group*groupSize;
      // We don't use recvPeers/sendPeers so let's pass shmem structs instead
      Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims
        (tidInGroup, groupSize, (int*)shmem->recvDims, (int*)shmem->sendDims, inputBuf, outputBuf, work->redOpArg, group, 0, 0, nullptr, nullptr, 0, primsModePatAg);

      int step = group;
      while(1) {
        struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
        int* poll = &ps->flags;
        while (__hip_atomic_load(poll, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) == 0){
          pollCount++; // Wait for compute thread
        }
        int last = ps->last;
        prims.patCopy(ps, shmem);
        if (tidInGroup == 0) __hip_atomic_store(poll, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP); // Return element to compute thread
        if (last) break;
        step += nGroups;
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  template<bool BcastSendNotRecv>
  struct Scatterer {
    struct ncclDevWorkColl* work;
    ssize_t chunkSize;
    ssize_t railGridOffset;

    template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void operator()(
        int tid, int tn, int slice, int maxSliceSize,
        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes, uint32_t sendDirectFlag, uint32_t recvDirectFlag
      ) {
      static_assert(SlicePerChunk==1, "require: SlicePerChunk==1");
      static_assert(MaxDsts<=1 || MaxSrcs<=1, "require: MaxDsts<=1 || MaxSrcs<=1");

      struct ncclNvls* nvls = &ncclShmem.channel.nvls;
      int nNodes = ncclShmem.comm.nNodes;
      int nRails = nvls->nHeads;
      int part = ncclShmem.channelId - work->channelLo;
      char* inbuf = (char*)work->sendbuff;
      char* outbuf = (char*)work->recvbuff;
      ssize_t countPerRank = work->collnet.count;
      bool inPlace = (inbuf == outbuf + ncclShmem.comm.rank * countPerRank);
      ssize_t railAllBeg = min(railGridOffset + part * chunkSize, nNodes * countPerRank);
      ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes * countPerRank);
      int railAllSize = railAllEnd - railAllBeg;
      int rail = 0;
      int src = 0;

      if (BcastSendNotRecv) {
        rail = nvls->headRank;
      } else {
        if (work->regUsed) return;
        rail = 0;
      }
      if (tid < nDsts) dstSizes[tid] = railAllSize;
      do {
        int node = railAllBeg / countPerRank;
        int railAllOffset = 0;
        while (railAllOffset < railAllSize) {
          ssize_t railOneBeg = node * countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          ssize_t railOneOffset = (railAllBeg + railAllOffset) - railOneBeg;
          int delta = min(railAllEnd, railOneEnd) - (railAllBeg + railAllOffset);
          int rank = ncclShmem.comm.collNetDenseToUserRank[node * nRails + rail];
          ssize_t userOneBeg = rank * countPerRank + railOneOffset;
          int outIsDst = (inPlace && rank == ncclShmem.comm.rank) || BcastSendNotRecv || work->regUsed ? 0 : 1;
          if (nSrcs != 0 && outIsDst + nDsts != 0) {
            reduceCopy<ncclCollUnroll(), USE_ACC, RedOp, T,
              /*MultimemSrcs,MinSrcs,MaxSrcs=*/MultimemSrcs, 1, 1,
              /*MultimemDsts=*/MultimemDsts, 0 + MultimemDsts + MinDsts, 1 + MaxDsts,
              /*PreOpSrcs=*/0>
              (tid, tn, 0, nullptr, false,
                /*nSrcs=*/1, [=]__device__(int s/*==0*/) -> void* {
              return (char*)srcPtrs[src] + railAllOffset;
            },
                /*nDsts=*/outIsDst + nDsts, [=]__device__(int d) -> void* {
              return d < outIsDst ? outbuf + userOneBeg
                : work->regUsed ? (char*)dstPtrs[d - outIsDst] + userOneBeg
                : (char*)dstPtrs[d - outIsDst] + railAllOffset;
            }, delta);
          }
          railAllOffset += delta;
          node += 1;
        }
        rail += 1;
        src += 1;
      } while (!BcastSendNotRecv && src < nRails);
    }
  };

  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    int nelem;

    const int nThreadsNetSend = work->oneNode ? 0 : (work->netRegUsed ? WARP_SIZE :  6 * WARP_SIZE);
    const int nThreadsGather = work->regUsed ? roundUp(nvls->nHeads << 2, WARP_SIZE) : 8 * WARP_SIZE;
    const int nThreadsBcast = NCCL_MAX_NTHREADS - nThreadsNetSend - nThreadsGather;

    const int tidEndGather = nThreadsGather;
    const int tidEndNetSend = tidEndGather + nThreadsNetSend;
    const int tidEndBcast = tidEndNetSend + nThreadsBcast;

    if (work->oneNode) {
      const ssize_t rank = ncclShmem.comm.rank;
      size_t count, gridOffset, channelCount, offset, chunkCount;
      ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T), &count, &gridOffset, &channelCount, &chunkCount);
      if (!work->regUsed) {
        if (tid < tidEndGather) {
          // Gather
          using Proto = ProtoSimple<1, 1, USE_ACC, COLL_UNROLL>;
          Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
            prims(tid, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff,
              work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            offset = gridOffset + elemOffset;
            nelem = min(chunkCount, channelCount - elemOffset);
            prims.gather(offset, nvls->nHeads * count, nelem, count, -1, 0);
          }
          // coverity[overrun-call] => Coverity think prims.index can be greater than 1
        } else if (tid < tidEndBcast) {
          // Bcast through NVLS
          using Proto = ProtoSimple<1, 1, USE_ACC, COLL_UNROLL, 0, 1>;
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
            prims(tid - tidEndGather, nThreadsBcast, NULL, &nvls->down, work->sendbuff, NULL,
              work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0);
          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            offset = gridOffset + elemOffset;
            nelem = min(chunkCount, channelCount - elemOffset);
            prims.send(offset, nelem);
          }
          // coverity[overrun-call] => Coverity think prims.index can be greater than 1
        }
      } else {
        if (tid < tidEndGather) {
          using Proto = ProtoSimple<1, 1, USE_ACC, COLL_UNROLL>;
          Primitives<T, RedOp, FanSymmetric<NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
            prims(tid, nThreadsGather, nvls->up, nvls->up, NULL, NULL,
              work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);

          /* used as sync */
          prims.scatter(0, 0, 0, 0, -1, 0);

          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            prims.gather(0, 0, 0, 0, -1, 0);
          }
        } else if (tid < tidEndBcast) {
          using Proto = ProtoSimple<1, 1, USE_ACC, COLL_UNROLL, 0, 1>;
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(tid - tidEndGather, nThreadsBcast, &nvls->down, &nvls->down, work->sendbuff, NULL,
              work->redOpArg, 1 * Proto::MaxGroupWidth, 0, 0, work);
          /* used as sync */
          prims.recv(0, 0);

          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            ssize_t inpOffset = gridOffset + elemOffset;
            ssize_t outOffset = inpOffset + rank * count;
            nelem = min(chunkCount, channelCount - elemOffset);
            prims.directSend(inpOffset, outOffset, nelem);
          }
        }
      }
    } else {
      // NVLS + IB SHARP
      int nNodes = ncclShmem.comm.nNodes;
      int part = ncclShmem.channelId - work->channelLo;
      ssize_t countPerRank = work->collnet.count;
      const int nChannels = work->channelHi - work->channelLo + 1;
      ssize_t chunkCount = work->collnet.chunkCount;
      if (tid < tidEndGather) {
        using Proto = ProtoSimple<1, 1, USE_ACC, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/1, Proto, 0>
          prims(tid, nThreadsGather, nvls->up, nullptr, nullptr, work->recvbuff,
            /*redOpArg=*/0, 1 * Proto::MaxGroupWidth, 1, 1, work);
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkCount) {
          Scatterer</*BcastSendNotRecv=*/false> scat;
          scat.work = work;
          scat.chunkSize = chunkCount;
          scat.railGridOffset = railGridOffset;
          prims.template process</*Recv=*/1, /*Send=*/0>(scat);
        }
      } else {
        if (work->netRegUsed) {
          using ProtoSend = ProtoSimple<1, 1, USE_ACC, COLL_UNROLL>;
          using ProtoBcast = ProtoSimple<1, 1, USE_ACC, COLL_UNROLL, 0, 1>;
          int maxSteps = (int)divUp(nNodes * countPerRank, nChannels * chunkCount);
          int curSteps = -1;
          int postThread = tid - tidEndGather == 0 ? 1 : 0;
          // for UB, we need to control the send speed to avoid net congestion.
          // first unroll 2 steps, then unroll the rest steps when the data is received.
          if (postThread) {
            curSteps = min(2, maxSteps);
            Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/1, ProtoSend, 0>::sendPeerNotify(nvls->out, 1, curSteps);
          }
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, ProtoBcast, 0>
            prims(tid - tidEndGather, nThreadsNetSend + nThreadsBcast, &nvls->out, &nvls->down, nullptr, nullptr,
              /*redOpArg=*/0, 2 * ProtoBcast::MaxGroupWidth, 0, 0, work);
          for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkCount) {
            Scatterer</*BcastSendNotRecv=*/true> scat;
            scat.work = work;
            scat.chunkSize = chunkCount;
            scat.railGridOffset = railGridOffset;
            prims.template process</*Recv=*/1, /*Send=*/1>(scat);
            if (postThread && curSteps < maxSteps) {
              curSteps++;
              Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/1, ProtoSend, 0>::sendPeerNotify(nvls->out, 1, 1);
            }
          }
        } else {
          if (tid < tidEndNetSend) {
            using Proto = ProtoSimple<1, 1, USE_ACC, COLL_UNROLL>;
            Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
              prims(tid - tidEndGather, nThreadsNetSend, nullptr, &nvls->out, work->sendbuff, nullptr,
                /*redOpArg=*/0, 0 * Proto::MaxGroupWidth, 1, 1);
            for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkCount) {
              ssize_t railAllBeg = railGridOffset + part * chunkCount;
              ssize_t railAllEnd = min(railAllBeg + chunkCount, nNodes * countPerRank);
              ssize_t railOneBeg = ncclShmem.comm.node * countPerRank;
              ssize_t railOneEnd = railOneBeg + countPerRank;
              ssize_t beg = max(railAllBeg, railOneBeg);
              ssize_t end = min(railAllEnd, railOneEnd);
              prims.send(beg - railOneBeg, max(ssize_t(0), end - beg));
            }
          } else {
            using Proto = ProtoSimple<1, 1, USE_ACC, COLL_UNROLL, 0, 1>;
            Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/0, Proto, 0>
              prims(tid - tidEndNetSend, nThreadsBcast, &nvls->out, &nvls->down, nullptr, nullptr,
                /*redOpArg=*/0, 2 * Proto::MaxGroupWidth, 0, 0);
            for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkCount) {
              Scatterer</*BcastSendNotRecv=*/true> scat;
              scat.work = work;
              scat.chunkSize = chunkCount;
              scat.railGridOffset = railGridOffset;
              prims.template process</*Recv=*/1, /*Send=*/1>(scat);
            }
          }
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  template<bool BcastSendNotRecv>
  struct Scatterer {
    struct ncclDevWorkColl* work;
    ssize_t chunkSize;
    ssize_t railGridOffset;

    template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void operator()(
        int tid, int tn, int slice, int maxSliceSize,
        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes, uint32_t sendDirectFlag, uint32_t recvDirectFlag
      ) {
      static_assert(SlicePerChunk==1, "require: SlicePerChunk==1");
      static_assert(MaxDsts<=1 || MaxSrcs<=1, "require: MaxDsts<=1 || MaxSrcs<=1");

      struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
      int nNodes = ncclShmem.comm.nNodes;
      int nRails = direct->nHeads;
      int part = ncclShmem.channelId - work->channelLo;
      char* inbuf = (char*)work->sendbuff;
      char* outbuf = (char*)work->recvbuff;
      ssize_t countPerRank = work->collnet.count*sizeof(T);
      bool inPlace = (inbuf == outbuf + ncclShmem.comm.rank*countPerRank);

      ssize_t railAllBeg = min(railGridOffset + part*chunkSize, nNodes*countPerRank);
      ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes*countPerRank);
      int railAllSize = railAllEnd - railAllBeg;
      if (tid < nDsts) dstSizes[tid] = railAllSize;

      int src = 0;
      int rail;
      if (BcastSendNotRecv) {
        rail = direct->headRank;
      } else {
        rail = direct->headRank+1;
        if (rail == nRails) rail = 0;
      }
      do {
        int node = railAllBeg/countPerRank;
        int railAllOffset = 0;
        while (railAllOffset < railAllSize) {
          ssize_t railOneBeg = node*countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          ssize_t railOneOffset = (railAllBeg+railAllOffset) - railOneBeg;
          int delta = min(railAllEnd, railOneEnd) - (railAllBeg+railAllOffset);
          int rank = ncclShmem.comm.collNetDenseToUserRank[node*nRails + rail];
          ssize_t userOneBeg = rank*countPerRank + railOneOffset;
          int outIsDst = (inPlace && rank == ncclShmem.comm.rank) ? 0 : 1;
          if (nSrcs != 0 && outIsDst+nDsts != 0) {
            reduceCopy<ncclCollUnroll(), USE_ACC, RedOp, T,
                    /*MultimemSrcs,MinSrcs,MaxSrcs=*/0,1,1,
                    /*MultimemDsts=*/0, 0+MinDsts, 1+MaxDsts,
                    /*PreOpSrcs=*/0>
            (tid, tn, 0, nullptr, false,
             /*nSrcs=*/1, [=]__device__(int s/*==0*/) -> void* {
               return work->regUsed && (recvDirectFlag & NCCL_P2P_READ) ? (char*)srcPtrs[src] + userOneBeg : (char*)srcPtrs[src] + railAllOffset;
             },
             /*nDsts=*/outIsDst+nDsts, [=]__device__(int d) -> void* {
               return d < outIsDst ? outbuf + userOneBeg
                                   : work->regUsed && (sendDirectFlag & NCCL_P2P_WRITE) ? (char*)dstPtrs[d-outIsDst] + userOneBeg
                                   : (char*)dstPtrs[d-outIsDst] + railAllOffset;
             },
             delta);
          }
          railAllOffset += delta;
          node += 1;
        }
        src += 1;
        rail += 1;
        if (rail == nRails) rail = 0;
      } while (!BcastSendNotRecv && src < nRails-1);
    }
  };

  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    const int part = ncclShmem.channelId - work->channelLo;
    const int nChannels = work->channelHi - work->channelLo + 1;
    struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
    int const &nNodes = ncclShmem.comm.nNodes;
    ssize_t countPerRank = work->collnet.count;
    size_t chunkSize = work->collnet.chunkCount;
    const int hasDn = (direct->down[0] >= 0) ? 1 : 0;
    bool isMultiRail = (direct->nHeads > 1);
    int nWarps1 = 1;
    int nWarps2 = (isMultiRail ? 2 : 1);
    int nWarps3 = (isMultiRail ? 2 : 0);
    float denom = float(work->nWarps)/float(nWarps1+nWarps2+nWarps3);
    nWarps3 = int(denom*nWarps3);
    nWarps2 = int(denom*nWarps2);
    nWarps1 = work->nWarps - (nWarps2+nWarps3);

    using Proto = ProtoSimple<1, 1>;

    int tn = nWarps1*WARP_SIZE;
    if (tid < tn) {
      if (work->netRegUsed) {
        if (tid == 0) {
          // If this rank has local peers (i.e, hasDn == true), we cannot offload all data to network.
          // In this case, steps should be computed based on chunkSize and so on; otherwise, we just
          // bump the step by 1 to kick off collnet progress.
          int steps = hasDn ? (int)divUp(nNodes * countPerRank, nChannels * chunkSize) : 1;
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>::sendPeerNotify(direct->out, 1, steps);
        }
        __syncwarp();
      } else {
        // Phase 1: send to network
        Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
          prims(tid, tn, nullptr, &direct->out, work->sendbuff, nullptr,
            /*redOpArg=*/0, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkSize) {
          ssize_t railAllBeg = railGridOffset + part * chunkSize;
          ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes * countPerRank);
          ssize_t railOneBeg = ncclShmem.comm.node * countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          ssize_t beg = max(railAllBeg, railOneBeg);
          ssize_t end = min(railAllEnd, railOneEnd);
          prims.send(beg - railOneBeg, max(ssize_t(0), end - beg));
        }
      }
      return;
    }
    tid -= tn;

    tn = nWarps2*WARP_SIZE;
    if (tid < tn) {
      if (work->netRegUsed && !hasDn) {
        if (tid == 0) {
          Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>::recvPeerNotify(direct->out, 0, 1);
        }
        __syncwarp();
      } else {
        // Phase 2: Recv network -> deposit output + send to bcast
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/1, Proto, 0>
          prims(tid, tn, &direct->out, direct->heads + 1, nullptr, work->recvbuff,
            /*redOpArg=*/0, 1 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkSize) {
          Scatterer</*BcastSendNotRecv=*/true> scat;
          scat.work = work;
          scat.chunkSize = chunkSize;
          scat.railGridOffset = railGridOffset;
          prims.template process</*Recv=*/1, /*Send=*/1>(scat, work->direct, 0);
        }
      }
      return;
    }
    tid -= tn;

    tn = nWarps3*WARP_SIZE;
    if (tid < tn) {
      // Phase 3: Recv bcast -> deposit output
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 0>, /*Direct=*/1, Proto, 0>
        prims(tid, tn, direct->heads+1, nullptr, nullptr, work->recvbuff,
              /*redOpArg=*/0, 2*Proto::MaxGroupWidth, 0, 0, work);
      for (ssize_t railGridOffset=0; railGridOffset < nNodes*countPerRank; railGridOffset += nChannels*chunkSize) {
        Scatterer</*BcastSendNotRecv=*/false> scat;
        scat.work = work;
        scat.chunkSize = chunkSize;
        scat.railGridOffset = railGridOffset;
        prims.template process</*Recv=*/1, /*Send=*/0>(scat, 0, work->direct);
      }
      return;
    }
  }
};