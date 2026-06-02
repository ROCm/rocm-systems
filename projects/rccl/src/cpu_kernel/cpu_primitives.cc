/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#include "cpu_primitives.h"

#include "cpu_host_coll.h"
#include "cpu_reduce.h"
#include "cpu_mem.h"
#include "checks.h"
#include "collectives.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace {

constexpr int kInput = 0;
constexpr int kOutput = 1;

uint64_t rcclCpuBarrierFetchAdd(uint64_t* barrier, int nWarps) {
  (void)nWarps;
  return __atomic_fetch_add(barrier, 1, __ATOMIC_RELEASE);
}

}  // namespace

static bool rcclCpuCheckAbort(struct rcclCpuBlockContext* ctx) {
  if (ctx->hostAbortFlag && rcclCpuLoadSeqCstU32(const_cast<uint32_t*>(ctx->hostAbortFlag))) {
    ctx->aborted = 1;
    return true;
  }
  return false;
}

static struct ncclConnInfo* rcclCpuGetRecvConn(struct rcclCpuBlockContext* ctx, int peer, uint8_t connIndex) {
  if (ctx->comm == nullptr || ctx->channel == nullptr) return nullptr;
  if (peer < 0 || peer >= ctx->comm->nRanks) return nullptr;
  if (connIndex >= NCCL_MAX_CONNS) return nullptr;
  if (ctx->channel->peers == nullptr || ctx->channel->peers[peer] == nullptr) return nullptr;
  return &ctx->channel->peers[peer]->recv[connIndex];
}

static struct ncclConnInfo* rcclCpuGetSendConn(struct rcclCpuBlockContext* ctx, int peer, uint8_t connIndex) {
  if (ctx->comm == nullptr || ctx->channel == nullptr) return nullptr;
  if (peer < 0 || peer >= ctx->comm->nRanks) return nullptr;
  if (connIndex >= NCCL_MAX_CONNS) return nullptr;
  if (ctx->channel->peers == nullptr || ctx->channel->peers[peer] == nullptr) return nullptr;
  return &ctx->channel->peers[peer]->send[connIndex];
}

static ncclResult_t rcclCpuWaitSend(int cudaDev, struct ncclConnInfo* conn, uint64_t step, int stepPerSlice) {
  if (conn == nullptr || conn->tail == nullptr) return ncclInvalidArgument;
  int spins = 0;
  uint64_t tailVal = 0;
  do {
    NCCLCHECK(rcclCpuLoadDevU64(cudaDev, conn->tail, &tailVal));
    if (tailVal + NCCL_STEPS >= step + stepPerSlice) break;
    if (++spins > 100000) return ncclInternalError;
    sched_yield();
  } while (true);
  return ncclSuccess;
}

static ncclResult_t rcclCpuWaitRecv(int cudaDev, struct ncclConnInfo* conn, uint64_t step, int stepPerSlice) {
  if (conn == nullptr || conn->head == nullptr) return ncclInvalidArgument;
  int spins = 0;
  uint64_t headVal = 0;
  do {
    NCCLCHECK(rcclCpuLoadDevU64(cudaDev, conn->head, &headVal));
    if (headVal + 0 >= step + stepPerSlice) break;
    if (++spins > 100000) return ncclInternalError;
    sched_yield();
  } while (true);
  return ncclSuccess;
}

static void* rcclCpuConnFifoPtr(struct ncclConnInfo* conn, uint64_t step) {
  if (conn == nullptr || conn->buffs[NCCL_PROTO_SIMPLE] == nullptr) return nullptr;
  return static_cast<char*>(conn->buffs[NCCL_PROTO_SIMPLE]) + (step % NCCL_STEPS) * conn->stepSize;
}

static ncclResult_t rcclCpuPostSend(int cudaDev, struct ncclConnInfo* conn, uint64_t step) {
  if (conn == nullptr) return ncclInvalidArgument;
  rcclCpuFenceSystem();
  if (conn->next_hdp_reg) NCCLCHECK(rcclCpuStoreDevU32(cudaDev, conn->next_hdp_reg, 1u));
  conn->step = step;
  if (conn->tail) NCCLCHECK(rcclCpuStoreDevU64(cudaDev, conn->tail, step));
  return ncclSuccess;
}

static ncclResult_t rcclCpuPostRecv(int cudaDev, struct ncclConnInfo* conn, uint64_t step) {
  if (conn == nullptr) return ncclInvalidArgument;
  conn->step = step;
  if (conn->head) NCCLCHECK(rcclCpuStoreDevU64(cudaDev, conn->head, step));
  return ncclSuccess;
}

#define RCCL_CPU_VOID_CHECK(call)          \
  do {                                     \
    ncclResult_t _rc = (call);             \
    if (_rc != ncclSuccess) return;        \
  } while (0)

rcclCpuPrimitives::rcclCpuPrimitives(
    struct rcclCpuBlockContext* ctx_, struct rcclCpuBlockBarrier* bar_,
    struct rcclCpuFuncDesc const& desc_, int tid_, int tn_,
    int const* recvPeers, int const* sendPeers,
    struct ncclDevWorkColl* work_, int groupId_)
    : ctx(ctx_), bar(bar_), desc(desc_), work(work_), tid(tid_), nthreads(tn_), groupId(groupId_) {
  if (ctx == nullptr || work == nullptr || ctx->comm == nullptr) return;
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
    if (ctx->comm->nNodes == 1) {
      slicePerChunk = ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS_SINGLE_NODE;
      stepPerSlice = ALLREDUCE_SLICESTEPS_SINGLE_NODE;
    }
    stepSize = ctx->comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS / std::max(1, ncclTypeSize(desc.datatype));
  } else {
    slicePerChunk = 1;
    stepPerSlice = 1;
    stepSize = ctx->comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
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
    group.recvConns[0] = rcclCpuGetRecvConn(ctx, recvPeer, connIndexRecv);
    if (group.recvConns[0]) step = group.recvConns[0]->step;
  }
  if (flags & (RoleWaitSend | RolePostSend) && sendPeer >= 0) {
    group.sendConns[0] = rcclCpuGetSendConn(ctx, sendPeer, connIndexSend);
    if (group.sendConns[0] && group.recvConns[0] == nullptr) step = group.sendConns[0]->step;
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
    if (conn == nullptr) return;
    connStepPtr = conn->head;
    RCCL_CPU_VOID_CHECK(rcclCpuLoadDevU64(ctx->cudaDev, connStepPtr, &connStepCache));
    connStepSize = conn->stepSize / std::max(1, ncclTypeSize(desc.datatype));
    connEltsFifo = conn->buffs[NCCL_PROTO_SIMPLE];
    int spins = 0;
    while (connStepCache + (send ? NCCL_STEPS : 0) < step + stepPerSlice) {
      RCCL_CPU_VOID_CHECK(rcclCpuLoadDevU64(ctx->cudaDev, connStepPtr, &connStepCache));
      if (rcclCpuCheckAbort(ctx)) break;
      if (++spins > 100000) {
        ctx->aborted = 1;
        break;
      }
      sched_yield();
    }
    group.srcs[index] = static_cast<char*>(connEltsFifo) + (step % NCCL_STEPS) * connStepSize * ncclTypeSize(desc.datatype);
  }
  if (flags & RoleWaitSend && send) {
    struct ncclConnInfo* conn = group.sendConns[index];
    if (conn == nullptr) return;
    connStepPtr = conn->tail;
    RCCL_CPU_VOID_CHECK(rcclCpuLoadDevU64(ctx->cudaDev, connStepPtr, &connStepCache));
    connStepSize = conn->stepSize / std::max(1, ncclTypeSize(desc.datatype));
    connEltsFifo = conn->buffs[NCCL_PROTO_SIMPLE];
    nextHdpReg = conn->next_hdp_reg;
    int spins = 0;
    while (connStepCache + NCCL_STEPS < step + stepPerSlice) {
      RCCL_CPU_VOID_CHECK(rcclCpuLoadDevU64(ctx->cudaDev, connStepPtr, &connStepCache));
      if (rcclCpuCheckAbort(ctx)) break;
      if (++spins > 100000) {
        ctx->aborted = 1;
        break;
      }
      sched_yield();
    }
    group.dsts[index] = static_cast<char*>(connEltsFifo) + (step % NCCL_STEPS) * connStepSize * ncclTypeSize(desc.datatype);
  }
}

void rcclCpuPrimitives::postPeer(int recv, int send, bool dataStored) {
  if (recv && (flags & RolePostRecv) && group.recvConns[index]) {
    struct ncclConnInfo* conn = group.recvConns[index];
    conn->step = step;
    RCCL_CPU_VOID_CHECK(rcclCpuStoreDevU64(ctx->cudaDev, conn->head, step));
  }
  if (send && (flags & RolePostSend) && group.sendConns[index]) {
    if (dataStored) rcclCpuFenceSystem();
    if (nextHdpReg) RCCL_CPU_VOID_CHECK(rcclCpuStoreDevU32(ctx->cudaDev, nextHdpReg, 1u));
    struct ncclConnInfo* conn = group.sendConns[index];
    conn->step = step;
    RCCL_CPU_VOID_CHECK(rcclCpuStoreDevU64(ctx->cudaDev, conn->tail, step));
  }
}

void rcclCpuPrimitives::genericOp(int recv, int send, int srcBuf, int dstBuf,
                                  intptr_t inpIx, intptr_t outIx, int eltN, bool postOp) {
  (void)srcBuf;
  (void)dstBuf;
  (void)postOp;
  if (eltN <= 0) return;

  int eltSize = ncclTypeSize(desc.datatype);
  int bytes = eltN * eltSize;
  struct ncclConnInfo* recvConn = recv ? rcclCpuGetRecvConn(ctx, recvPeer, connIndexRecv) : nullptr;
  struct ncclConnInfo* sendConn = send ? rcclCpuGetSendConn(ctx, sendPeer, connIndexSend) : nullptr;

  if (recv) {
    if (recvConn == nullptr) return;
    if (rcclCpuCheckAbort(ctx)) return;
    RCCL_CPU_VOID_CHECK(rcclCpuWaitRecv(ctx->cudaDev, recvConn, step, stepPerSlice));
    void* fifo = rcclCpuConnFifoPtr(recvConn, step);
    void* userOut = static_cast<char*>(group.userOutput) + outIx * eltSize;
    if (fifo && userOut) RCCL_CPU_VOID_CHECK(rcclCpuCopyBytes(ctx->cudaDev, userOut, fifo, bytes));
    step += stepPerSlice;
    RCCL_CPU_VOID_CHECK(rcclCpuPostRecv(ctx->cudaDev, recvConn, step));
  }

  if (send) {
    if (sendConn == nullptr) return;
    if (rcclCpuCheckAbort(ctx)) return;
    RCCL_CPU_VOID_CHECK(rcclCpuWaitSend(ctx->cudaDev, sendConn, step, stepPerSlice));
    void* fifo = rcclCpuConnFifoPtr(sendConn, step);
    void const* userIn = static_cast<char*>(group.userInput) + inpIx * eltSize;
    if (fifo && userIn) RCCL_CPU_VOID_CHECK(rcclCpuCopyBytes(ctx->cudaDev, fifo, userIn, bytes));
    step += stepPerSlice;
    RCCL_CPU_VOID_CHECK(rcclCpuPostSend(ctx->cudaDev, sendConn, step));
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
