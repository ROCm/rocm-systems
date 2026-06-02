/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host collective executors mirroring device RunWorkColl / ring / tree paths.
 ************************************************************************/

#include "cpu_coll_exec.h"

#include "cpu_host_coll.h"
#include "cpu_memory_model.h"
#include "cpu_primitives.h"
#include "collectives.h"
#include "debug.h"

#include <algorithm>
#include <cstring>

namespace {

using rcclCpuPrimitives = ::rcclCpuPrimitives;

ncclResult_t runRingAllReduce(
    rcclCpuPrimitives& prims, int tid, int tn, struct ncclDevWorkColl* work, int proto) {
  ncclRing* ring = &prims.ctx->channel->ring;
  int nranks = prims.ctx->comm->nRanks;
  int eltSize = ncclTypeSize(prims.desc.datatype);
  ssize_t size, gridOffset, channelCount, chunkCount;
  rcclCpuCollCbdPart(work, prims.ctx->channelId, proto, eltSize, &size, &gridOffset, &channelCount, &chunkCount);
  ssize_t loopCount = nranks * chunkCount;

  for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
    ssize_t remCount = channelCount - elemOffset;
    ssize_t curChunk = chunkCount;
    if (remCount < loopCount) curChunk = rcclCpuAlignUp(rcclCpuDivUp(remCount, nranks), 16 / eltSize);

    auto modRanks = [nranks](int r) { return r >= nranks ? r - nranks : r; };
    int ringIx = ring->index;

    int chunk = modRanks(ringIx + nranks - 1);
    ssize_t offset = gridOffset + elemOffset + chunk * curChunk;
    int nelem = static_cast<int>(std::min(curChunk, remCount - chunk * curChunk));
    prims.directSend(offset, offset, nelem);

    for (int j = 2; j < nranks; j++) {
      chunk = modRanks(ringIx + nranks - j);
      offset = gridOffset + elemOffset + chunk * curChunk;
      nelem = static_cast<int>(std::min(curChunk, remCount - chunk * curChunk));
      prims.directRecvReduceDirectSend(offset, offset, nelem);
    }

    chunk = ringIx;
    offset = gridOffset + elemOffset + chunk * curChunk;
    nelem = static_cast<int>(std::min(curChunk, remCount - chunk * curChunk));
    prims.directRecvReduceCopyDirectSend(offset, offset, nelem, true);

    for (int j = 1; j < nranks - 1; j++) {
      chunk = modRanks(ringIx + nranks - j);
      offset = gridOffset + elemOffset + chunk * curChunk;
      nelem = static_cast<int>(std::min(curChunk, remCount - chunk * curChunk));
      prims.directRecvCopyDirectSend(offset, nelem);
    }

    chunk = modRanks(ringIx + 1);
    offset = gridOffset + elemOffset + chunk * curChunk;
    nelem = static_cast<int>(std::min(curChunk, remCount - chunk * curChunk));
    prims.directRecv(offset, nelem);
  }
  return ncclSuccess;
}

ncclResult_t runRingBroadcast(
    rcclCpuPrimitives& prims, int tid, int tn, struct ncclDevWorkColl* work, int proto) {
  ncclRing* ring = &prims.ctx->channel->ring;
  int rank = ring->userRanks[0];
  int nextRank = ring->userRanks[1];
  int root = work->root;
  int eltSize = ncclTypeSize(prims.desc.datatype);
  ssize_t size, gridOffset, channelCount, chunkCount;
  rcclCpuCollCbdPart(work, prims.ctx->channelId, proto, eltSize, &size, &gridOffset, &channelCount, &chunkCount);

  for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
    ssize_t offset = gridOffset + elemOffset;
    int nelem = static_cast<int>(std::min(chunkCount, channelCount - elemOffset));
    if (rank == root) {
      if (work->sendbuff == work->recvbuff) prims.directSend(offset, offset, nelem);
      else prims.directCopySend(offset, offset, nelem);
    } else if (nextRank == root) {
      prims.directRecv(offset, nelem);
    } else {
      prims.directRecvCopyDirectSend(offset, nelem);
    }
  }
  return ncclSuccess;
}

ncclResult_t runRingReduce(
    rcclCpuPrimitives& prims, int tid, int tn, struct ncclDevWorkColl* work, int proto) {
  ncclRing* ring = &prims.ctx->channel->ring;
  int rank = ring->userRanks[0];
  int root = work->root;
  int eltSize = ncclTypeSize(prims.desc.datatype);
  ssize_t size, gridOffset, channelCount, chunkCount;
  rcclCpuCollCbdPart(work, prims.ctx->channelId, proto, eltSize, &size, &gridOffset, &channelCount, &chunkCount);

  for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
    ssize_t offset = gridOffset + elemOffset;
    int nelem = static_cast<int>(std::min(chunkCount, channelCount - elemOffset));
    if (rank == root) prims.directRecv(offset, nelem, true);
    else prims.directRecvReduceDirectSend(offset, offset, nelem);
  }
  return ncclSuccess;
}

ncclResult_t runRingAllGather(
    rcclCpuPrimitives& prims, int tid, int tn, struct ncclDevWorkColl* work, int proto) {
  ncclRing* ring = &prims.ctx->channel->ring;
  int nranks = prims.ctx->comm->nRanks;
  int rank = prims.ctx->comm->rank;
  int eltSize = ncclTypeSize(prims.desc.datatype);
  ssize_t size, gridOffset, channelCount, chunkCount;
  rcclCpuCollCbdPart(work, prims.ctx->channelId, proto, eltSize, &size, &gridOffset, &channelCount, &chunkCount);
  ssize_t loopCount = nranks * chunkCount;

  for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
    ssize_t rem = channelCount - elemOffset;
    ssize_t curChunk = rem < loopCount ? rcclCpuAlignUp(rcclCpuDivUp(rem, nranks), std::max(1, 16 / eltSize)) : chunkCount;
    ssize_t offset = gridOffset + elemOffset + rank * curChunk;
    int nelem = static_cast<int>(std::min(curChunk, rem - rank * curChunk));
    if (nelem > 0) prims.directSend(offset, offset, nelem);
    for (int j = 1; j < nranks; j++) {
      prims.directRecvCopyDirectSend(offset, nelem);
    }
  }
  return ncclSuccess;
}

ncclResult_t runRingReduceScatter(
    rcclCpuPrimitives& prims, int tid, int tn, struct ncclDevWorkColl* work, int proto) {
  return runRingAllReduce(prims, tid, tn, work, proto);
}

ncclResult_t runTreeAllReduce(
    rcclCpuPrimitives& prims, int tid, int tn, struct ncclDevWorkColl* work, int proto) {
  // Tree up/down approximated via ring path on CPU when tree channels are configured.
  (void)proto;
  return runRingAllReduce(prims, tid, tn, work, NCCL_PROTO_SIMPLE);
}

ncclResult_t dispatchRing(
    struct rcclCpuBlockContext* ctx, struct rcclCpuBlockBarrier* bar,
    int tid, int tn, struct ncclDevWorkColl* work, struct rcclCpuFuncDesc const& desc) {
  int prev = ctx->channel->ring.prev;
  int next = ctx->channel->ring.next;
  int peers[2] = {prev, next};
  struct ncclDevWorkColl workLocal = *work;
  int cpuTn = 1;
  rcclCpuPrimitives prims(ctx, bar, desc, tid, cpuTn, &peers[0], &peers[1], &workLocal);

  switch (desc.coll) {
  case ncclFuncAllReduce:
    return runRingAllReduce(prims, tid, tn, work, desc.proto);
  case ncclFuncBroadcast:
    return runRingBroadcast(prims, tid, tn, work, desc.proto);
  case ncclFuncReduce:
    return runRingReduce(prims, tid, tn, work, desc.proto);
  case ncclFuncAllGather:
    return runRingAllGather(prims, tid, tn, work, desc.proto);
  case ncclFuncReduceScatter:
    return runRingReduceScatter(prims, tid, tn, work, desc.proto);
  default:
    return runRingAllReduce(prims, tid, tn, work, desc.proto);
  }
}

ncclResult_t dispatchTree(
    struct rcclCpuBlockContext* ctx, struct rcclCpuBlockBarrier* bar,
    int tid, int tn, struct ncclDevWorkColl* work, struct rcclCpuFuncDesc const& desc) {
  int up = ctx->channel->tree.up;
  int down0 = ctx->channel->tree.down[0];
  int recvPeers[3] = {up, down0, -1};
  int sendPeers[3] = {down0, up, -1};
  int cpuTn = 1;
  rcclCpuPrimitives prims(ctx, bar, desc, tid, cpuTn, recvPeers, sendPeers, work);
  return runTreeAllReduce(prims, tid, tn, work, desc.proto);
}

}  // namespace

ncclResult_t rcclCpuExecuteCollWork(
    struct rcclCpuBlockContext* ctx,
    struct rcclCpuBlockBarrier* bar,
    int tid, int tn,
    struct ncclDevWorkColl* work,
    struct rcclCpuFuncDesc const& desc) {
  if (ctx->hostAbortFlag && rcclCpuLoadSeqCstU32(const_cast<uint32_t*>(ctx->hostAbortFlag))) {
    ctx->aborted = 1;
    return ncclSuccess;
  }

  if (desc.proto != NCCL_PROTO_SIMPLE) {
    // LL / LL128 use the same collective patterns with different step sizes; fall back to SIMPLE
    // transport on CPU until dedicated LL fifo paths are required.
    struct rcclCpuFuncDesc simpleDesc = desc;
    simpleDesc.proto = NCCL_PROTO_SIMPLE;
    return rcclCpuExecuteCollWork(ctx, bar, tid, tn, work, simpleDesc);
  }

  switch (desc.algo) {
  case NCCL_ALGO_RING:
  case NCCL_ALGO_PAT:
    return dispatchRing(ctx, bar, tid, tn, work, desc);
  case NCCL_ALGO_TREE:
    return dispatchTree(ctx, bar, tid, tn, work, desc);
  case NCCL_ALGO_COLLNET_DIRECT:
  case NCCL_ALGO_COLLNET_CHAIN:
  case NCCL_ALGO_NVLS:
  case NCCL_ALGO_NVLS_TREE:
    // Network/collnet legs are driven by proxy; execute local channel work via ring primitives.
    return dispatchRing(ctx, bar, tid, tn, work, desc);
  default:
    WARN("rcclCpuExecuteCollWork: unsupported algo %d, using RING", desc.algo);
    return dispatchRing(ctx, bar, tid, tn, work, desc);
  }
}

ncclResult_t rcclCpuExecuteP2pWork(
    struct rcclCpuBlockContext* ctx,
    struct rcclCpuBlockBarrier* bar,
    int tid, int tn,
    struct ncclDevWorkP2p* work,
    struct rcclCpuFuncDesc const& desc) {
  struct ncclDevWorkColl coll{};
  coll.sendbuff = work->sendAddr;
  coll.recvbuff = work->recvAddr;
  coll.connIndex = work->sendConnIndex;
  coll.redOpArg = 0;
  coll.nWarps = static_cast<uint32_t>((tn + ctx->warpSize - 1) / ctx->warpSize);

  if (work->sendRank >= 0) {
    int sendPeer = work->sendRank;
    int cpuTn = 1;
    rcclCpuPrimitives prims(ctx, bar, desc, tid, cpuTn, nullptr, &sendPeer, &coll);
    size_t bytes = work->sendBytes;
    int chunk = ctx->comm->p2pChunkSize;
  size_t cursor = 0;
    while (cursor < bytes) {
      int n = static_cast<int>(std::min(static_cast<size_t>(chunk), bytes - cursor));
      prims.directSend(static_cast<intptr_t>(cursor), static_cast<intptr_t>(cursor), n);
      cursor += n;
    }
  }
  if (work->recvRank >= 0) {
    int recvPeer = work->recvRank;
    coll.connIndex = work->recvConnIndex;
    int cpuTn = 1;
    rcclCpuPrimitives prims(ctx, bar, desc, tid, cpuTn, &recvPeer, nullptr, &coll);
    size_t bytes = work->recvBytes;
    int chunk = ctx->comm->p2pChunkSize;
    size_t cursor = 0;
    while (cursor < bytes) {
      int n = static_cast<int>(std::min(static_cast<size_t>(chunk), bytes - cursor));
      prims.directRecv(static_cast<intptr_t>(cursor), n);
      cursor += n;
    }
  }
  return ncclSuccess;
}
