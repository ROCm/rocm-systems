/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx942__) && !defined(__gfx950__)
  __device__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
#else
  __device__ __attribute__((noinline)) void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
#endif
    const int bid = ncclShmem.channelId - work->channelLo;
    const int nranks = ncclShmem.comm.nRanks;
    size_t count, partOffset, partCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &partOffset, &partCount, &chunkCount);

    const ncclRing *ring = &ncclShmem.channel.ring;
    const int num_bi_rings = work->pivotA2ANumBiRings;
    const int num_uni_rings = num_bi_rings * 2;
    const int num_chunks = (work->channelHi - work->channelLo + 1) / 2;
    const int chunk_id = (bid % num_bi_rings) + (bid / num_uni_rings * num_bi_rings);
    const int elem_size = min(256, count & (~(count) + 1));
    const ssize_t num_elems = count / elem_size;
    const int num_padding_chunks = num_elems % num_chunks;
    const ssize_t chunk_offset = elem_size * (num_elems / num_chunks * chunk_id + (chunk_id < num_padding_chunks ? chunk_id : num_padding_chunks));
    const ssize_t chunk_size = elem_size * (num_elems / num_chunks + (chunk_id < num_padding_chunks ? 1 : 0));
    const int pivot_direction = (bid % num_uni_rings) / num_bi_rings;
    const ssize_t prims_size = chunkCount;

    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, /*redOpArg(ignored)=*/0);

    for (int num_hops = 0; num_hops <= nranks / 2; num_hops++) {
      const int src_rank = ring->userRanks[(nranks - num_hops) % nranks];
      const int dst_rank = ring->userRanks[num_hops];
      const ssize_t send_offset =
          dst_rank * count + chunk_offset +
          (src_rank == dst_rank ? pivot_direction * chunk_size / 2 : 0);
      const ssize_t recv_offset =
          src_rank * count + chunk_offset +
          (src_rank == dst_rank ? pivot_direction * chunk_size / 2 : 0);
      const ssize_t send_recv_size =
          src_rank == dst_rank ?
          (pivot_direction == 0 ? chunk_size / 2 : chunk_size - chunk_size / 2) : chunk_size;

      if (num_hops == 0 && work->sendbuff != work->recvbuff) {
        const T* sendbuff = (const T*)work->sendbuff + send_offset;
        T* recvbuff = (T *)work->recvbuff + recv_offset;
        reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nthreads, 0, nullptr, false, 1, (void **)&sendbuff, 1, (void **)&recvbuff, send_recv_size);
      } else {
        for (ssize_t prims_offset = 0; prims_offset < send_recv_size; prims_offset += prims_size) {
          const int prims_nelem = min(prims_size, send_recv_size - prims_offset);

          // step 0: send
          prims.send(send_offset + prims_offset, prims_nelem);

          // num_hops - 1 steps: recv and copy to next gpu
          for (int i = 0; i < num_hops - 1; i++) {
            prims.recvSend(prims_nelem);
          }

          // final step: recv
          prims.directRecv(recv_offset + prims_offset, recv_offset + prims_offset, prims_nelem);
        }
      }
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllToAllPivot, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nThreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLTOALL_PIVOT_CHUNKSTEPS/ALLTOALL_PIVOT_SLICESTEPS, ALLTOALL_PIVOT_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nThreads, work);
  }
};
