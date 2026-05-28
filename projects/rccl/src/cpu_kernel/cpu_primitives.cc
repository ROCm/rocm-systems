/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#include "cpu_primitives.h"

#include "cpu_host_coll.h"
#include "cpu_reduce.h"
#include "collectives.h"

#include <algorithm>
#include <cstring>

namespace {

constexpr int kInput = 0;
constexpr int kOutput = 1;

uint64_t rcclCpuBarrierFetchAdd(uint64_t* barrier, int nWarps) {
  (void)nWarps;
  return __atomic_fetch_add(barrier, 1, __ATOMIC_RELEASE);
}

}  // namespace

rcclCpuPrimitives::rcclCpuPrimitives(
    struct rcclCpuBlockContext* ctx_, struct rcclCpuBlockBarrier* bar_,
    struct rcclCpuFuncDesc const& desc_, int tid_, int tn_,
    int const* recvPeers, int const* sendPeers,
    struct ncclDevWorkColl* work_, int groupId_)
    : ctx(ctx_), bar(bar_), desc(desc_), work(work_), tid(tid_), nthreads(tn_), groupId(groupId_) {
  std::memset(&group, 0, sizeof(group));
  group.userInput = work->sendbuff;
  group.userOutput = work->recvbuff;
  connIndexRecv = work->connIndex;
  connIndexSend = work->connIndex;

  recvPeer = recvPeers ? recvPeers[0] : -1;
  sendPeer = sendPeers ? sendPeers[0] : -1;

  if (desc.proto == NCCL_PROTO_SIMPLE) {
    slicePerChunk = ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS;
    stepPerSlice = ALLREDUCE_SLICESTEPS;
    if (ctx->comm.nNodes == 1) {
      slicePerChunk = ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS_SINGLE_NODE;
      stepPerSlice = ALLREDUCE_SLICESTEPS_SINGLE_NODE;
    }
    stepSize = ctx->comm.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS / std::max(1, ncclTypeSize(desc.datatype));
  } else {
    slicePerChunk = 1;
    stepPerSlice = 1;
    stepSize = ctx->comm.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
  }

  flags = 0;
  index = 0;
  int nrecv = recvPeer >= 0 ? 1 : 0;
  int nsend = sendPeer >= 0 ? 1 : 0;
  if (tid < nrecv) { flags |= RoleWaitRecv; index = tid; }
  else if (tid < nrecv + nsend) { flags |= RoleWaitSend; index = tid - nrecv; }
  else if (tid >= nthreads - nsend) { flags |= RolePostSend; index = tid - (nthreads - nsend); }
  else if (tid >= nthreads - nrecv - nsend) { flags |= RolePostRecv; index = tid - (nthreads - nrecv - nsend); }

  if (flags & (RoleWaitRecv | RolePostRecv) && recvPeer >= 0) {
    group.recvConns[0] = &ctx->channel.peers[recvPeer]->recv[connIndexRecv];
    step = group.recvConns[0]->step;
  }
  if (flags & (RoleWaitSend | RolePostSend) && sendPeer >= 0) {
    group.sendConns[0] = &ctx->channel.peers[sendPeer]->send[connIndexSend];
    if (!(flags & (RoleWaitRecv | RolePostRecv))) step = group.sendConns[0]->step;
  }
  group.barrier = 0;
}

void rcclCpuPrimitives::barrier() {
  if (nthreads == ctx->warpSize) {
    rcclCpuFenceBlock();
    return;
  }
  rcclCpuBlockBarrierWait(bar, tid, nthreads);
}

void rcclCpuPrimitives::subBarrier() {
  barrier();
}

uint64_t rcclCpuPrimitives::loadStepValue(uint64_t* ptr) {
  return rcclCpuLoadRelaxedU64(ptr);
}

void rcclCpuPrimitives::waitPeer(int recv, int send, int nbytes) {
  (void)nbytes;
  if (flags & RoleWaitRecv && recv) {
    struct ncclConnInfo* conn = group.recvConns[index];
    connStepPtr = conn->head;
    connStepCache = loadStepValue(connStepPtr);
    connStepSize = conn->stepSize / std::max(1, ncclTypeSize(desc.datatype));
    connEltsFifo = conn->buffs[NCCL_PROTO_SIMPLE];
    int spins = 0;
    while (connStepCache + (send ? NCCL_STEPS : 0) < step + stepPerSlice) {
      connStepCache = loadStepValue(connStepPtr);
      if (ctx->comm.abortFlag && rcclCpuLoadSeqCstU32(const_cast<uint32_t*>(ctx->comm.abortFlag))) {
        ctx->aborted = 1;
        break;
      }
      if (++spins > 10000000) break;
      sched_yield();
    }
    group.srcs[index] = static_cast<char*>(connEltsFifo) + (step % NCCL_STEPS) * connStepSize * ncclTypeSize(desc.datatype);
  }
  if (flags & RoleWaitSend && send) {
    struct ncclConnInfo* conn = group.sendConns[index];
    connStepPtr = conn->tail;
    connStepCache = loadStepValue(connStepPtr);
    connStepSize = conn->stepSize / std::max(1, ncclTypeSize(desc.datatype));
    connEltsFifo = conn->buffs[NCCL_PROTO_SIMPLE];
    nextHdpReg = conn->next_hdp_reg;
    int spins = 0;
    while (connStepCache + NCCL_STEPS < step + stepPerSlice) {
      connStepCache = loadStepValue(connStepPtr);
      if (ctx->comm.abortFlag && rcclCpuLoadSeqCstU32(const_cast<uint32_t*>(ctx->comm.abortFlag))) {
        ctx->aborted = 1;
        break;
      }
      if (++spins > 10000000) break;
      sched_yield();
    }
    group.dsts[index] = static_cast<char*>(connEltsFifo) + (step % NCCL_STEPS) * connStepSize * ncclTypeSize(desc.datatype);
  }
}

void rcclCpuPrimitives::postPeer(int recv, int send, bool dataStored) {
  if (recv && (flags & RolePostRecv) && group.recvConns[index]) {
    struct ncclConnInfo* conn = group.recvConns[index];
    conn->step = step;
    rcclCpuStoreRelaxedU64(conn->head, step);
  }
  if (send && (flags & RolePostSend) && group.sendConns[index]) {
    if (dataStored) rcclCpuFenceSystem();
    if (nextHdpReg) rcclCpuStoreRelaxedU32(nextHdpReg, 1u);
    struct ncclConnInfo* conn = group.sendConns[index];
    conn->step = step;
    rcclCpuStoreRelaxedU64(conn->tail, step);
  }
}

void rcclCpuPrimitives::genericOp(int recv, int send, int srcBuf, int dstBuf,
                                  intptr_t inpIx, intptr_t outIx, int eltN, bool postOp) {
  int eltSize = ncclTypeSize(desc.datatype);
  int sliceSize = stepSize * stepPerSlice;
  sliceSize = std::max(static_cast<int>(rcclCpuDivUp(eltN, 16) * 16), sliceSize / 32);
  int offset = 0;
  int slice = 0;

  while (slice < slicePerChunk && offset < eltN) {
    int nelem = std::min(sliceSize, eltN - offset);
    if (tid < nthreads) {
      waitPeer(recv, send, nelem * eltSize);
      subBarrier();

      void const* srcs[2] = {nullptr, nullptr};
      void* dsts[2] = {nullptr, nullptr};
      int nSrcs = 0, nDsts = 0;

      if (srcBuf == kInput) {
        srcs[nSrcs++] = static_cast<char*>(group.userInput) + (inpIx + offset) * eltSize;
      } else if (srcBuf == kOutput && recv) {
        srcs[nSrcs++] = group.srcs[index];
      } else if (recv && group.srcs[index]) {
        srcs[nSrcs++] = group.srcs[index];
      }

      if (dstBuf == kOutput) {
        dsts[nDsts++] = static_cast<char*>(group.userOutput) + (outIx + offset) * eltSize;
      } else if (send && group.dsts[index]) {
        dsts[nDsts++] = group.dsts[index];
      }

      if (nSrcs > 0 || nDsts > 0) {
        rcclCpuReduceCopy(tid, nthreads, desc.datatype, desc.devRedOp,
                          srcs, nSrcs, dsts, nDsts, static_cast<size_t>(nelem), work->redOpArg, postOp);
      }
      barrier();
      postPeer(recv, send, nSrcs > 0 || nDsts > 0);
    } else {
      waitPeer(recv, send, 0);
      barrier();
      postPeer(recv, send, false);
    }
    offset += sliceSize;
    slice++;
    step += stepPerSlice;
  }
}

void rcclCpuPrimitives::directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
  genericOp(0, 1, kInput, -1, inpIx, outIx, eltN, false);
}

void rcclCpuPrimitives::directRecv(intptr_t outIx, int eltN, bool postOp) {
  genericOp(1, 0, -1, kOutput, outIx, outIx, eltN, postOp);
}

void rcclCpuPrimitives::directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp) {
  genericOp(1, 1, kInput, -1, inpIx, outIx, eltN, postOp);
}

void rcclCpuPrimitives::directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp) {
  genericOp(1, 1, kInput, kOutput, inpIx, outIx, eltN, postOp);
}

void rcclCpuPrimitives::directRecvCopyDirectSend(intptr_t outIx, int eltN, bool postOp) {
  genericOp(1, 1, -1, kOutput, -1, outIx, eltN, postOp);
}

void rcclCpuPrimitives::directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp) {
  genericOp(0, 1, kInput, kOutput, inpIx, outIx, eltN, postOp);
}
