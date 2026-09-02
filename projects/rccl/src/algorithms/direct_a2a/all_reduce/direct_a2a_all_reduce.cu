/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/direct_a2a/all_reduce/direct_a2a_all_reduce.h"

#include "algorithms/dda/device/CollCommon.h"
#include "alloc.h"
#include "archinfo.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "param.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

RCCL_PARAM(DirectA2aEnable, "DIRECT_A2A_ENABLE", 0);
RCCL_PARAM(DirectA2aOneShotThreshold, "DIRECT_A2A_ONESHOT_THRESHOLD",
           RCCL_DIRECT_A2A_DEFAULT_ONESHOT_THRESHOLD_BYTES);
RCCL_PARAM(DirectA2aMaxBytes, "DIRECT_A2A_MAX_BYTES", RCCL_DIRECT_A2A_MAX_BYTES);

namespace {

constexpr int kDirectA2aThreads = 256;
constexpr int kDirectA2aMaxBlocks = 256;
constexpr size_t kDirectA2aMaxBytesCeiling = 1ULL << 25; // env cap 32 MiB
static_assert(RCCL_DIRECT_A2A_DEFAULT_ONESHOT_THRESHOLD_BYTES <= RCCL_DIRECT_A2A_MAX_BYTES,
              "DIRECT_A2A one-shot threshold must not exceed its maximum message size");
static_assert(RCCL_DIRECT_A2A_TWO_RANK_MAX_BYTES <= RCCL_DIRECT_A2A_MAX_BYTES,
              "DIRECT_A2A two-rank maximum must not exceed its global maximum message size");
static_assert(RCCL_DIRECT_A2A_MAX_BYTES <= kDirectA2aMaxBytesCeiling,
              "DIRECT_A2A default maximum must not exceed the env ceiling");

size_t directA2aMaxBytes(int nRanks) {
  if (nRanks == 2) return RCCL_DIRECT_A2A_TWO_RANK_MAX_BYTES;
  const int64_t configured = rcclParamDirectA2aMaxBytes();
  if (configured <= 0) return 0;
  return std::min<size_t>((size_t)configured, kDirectA2aMaxBytesCeiling);
}

size_t directA2aOneShotThreshold(int nRanks) {
  // With two ranks, ReduceScatter + AllGather moves the same total bytes as
  // one-shot but adds a second P2P phase, so two-shot is never selected.
  if (nRanks == 2) return RCCL_DIRECT_A2A_TWO_RANK_MAX_BYTES;

  const int64_t configured = rcclParamDirectA2aOneShotThreshold();
  if (configured <= 0) return 0;
  return std::min<size_t>((size_t)configured, directA2aMaxBytes(nRanks));
}

template <typename T>
__global__ void directA2aSumKernel(const T* __restrict__ scratch, T* __restrict__ recvbuff, size_t count, int nRanks) {
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < count;
       idx += (size_t)gridDim.x * blockDim.x) {
    T sum = scratch[idx];
    for (int rank = 1; rank < nRanks; ++rank) {
      sum = sum + scratch[(size_t)rank * count + idx];
    }
    recvbuff[idx] = sum;
  }
}

template <typename T>
ncclResult_t launchDirectA2aSum(void* scratch, void* recvbuff, size_t count, int nRanks, cudaStream_t stream) {
  int blocks = (int)std::min<size_t>((count + kDirectA2aThreads - 1) / kDirectA2aThreads, kDirectA2aMaxBlocks);
  directA2aSumKernel<T><<<blocks, kDirectA2aThreads, 0, stream>>>(
    static_cast<const T*>(scratch), static_cast<T*>(recvbuff), count, nRanks);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

ncclResult_t launchDirectA2aSumByType(void* scratch, void* recvbuff, size_t count, int nRanks,
                                      ncclDataType_t datatype, cudaStream_t stream) {
  switch (datatype) {
  case ncclFloat16:
    return launchDirectA2aSum<half>(scratch, recvbuff, count, nRanks, stream);
  case ncclBfloat16:
    return launchDirectA2aSum<bf16>(scratch, recvbuff, count, nRanks, stream);
  case ncclFloat32:
    return launchDirectA2aSum<float>(scratch, recvbuff, count, nRanks, stream);
  default:
    return ncclInvalidArgument;
  }
}

size_t directA2aChunkCount(size_t count, int nRanks, int chunk) {
  return count / (size_t)nRanks + ((size_t)chunk < count % (size_t)nRanks ? 1 : 0);
}

size_t directA2aChunkOffset(size_t count, int nRanks, int chunk) {
  const size_t base = count / (size_t)nRanks;
  return (size_t)chunk * base + std::min<size_t>((size_t)chunk, count % (size_t)nRanks);
}

ncclResult_t directA2aOneShot(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                              ncclComm* comm, cudaStream_t stream) {
  const size_t bytes = count * ncclTypeSize(datatype);
  ncclResult_t ret = ncclSuccess;

  void* selfSlot = static_cast<char*>(comm->directA2aScratch) + (size_t)comm->rank * bytes;
  CUDACHECK(cudaMemcpyAsync(selfSlot, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));

  rccl::Recorder::instance().skip(true);
  ret = ncclGroupStart();
  if (ret == ncclSuccess) {
    for (int peer = 0; peer < comm->nRanks && ret == ncclSuccess; ++peer) {
      if (peer == comm->rank) continue;
      ret = ncclSend(sendbuff, count, datatype, peer, comm, stream);
      if (ret == ncclSuccess) {
        void* peerSlot = static_cast<char*>(comm->directA2aScratch) + (size_t)peer * bytes;
        ret = ncclRecv(peerSlot, count, datatype, peer, comm, stream);
      }
    }
    ncclResult_t groupEndResult = ncclGroupEnd();
    if (ret == ncclSuccess) ret = groupEndResult;
  }
  rccl::Recorder::instance().skip(false);
  NCCLCHECK(ret);

  INFO(NCCL_COLL, "AllReduce: taking standalone DIRECT_A2A one-shot path: rank=%d/%d count=%zu datatype=%d bytes=%zu",
       comm->rank, comm->nRanks, count, (int)datatype, bytes);
  return launchDirectA2aSumByType(comm->directA2aScratch, recvbuff, count, comm->nRanks, datatype, stream);
}

ncclResult_t directA2aTwoShot(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                              ncclComm* comm, cudaStream_t stream) {
  const size_t typeSize = ncclTypeSize(datatype);
  const size_t bytes = count * typeSize;
  const size_t ownedCount = directA2aChunkCount(count, comm->nRanks, comm->rank);
  const size_t ownedOffset = directA2aChunkOffset(count, comm->nRanks, comm->rank);
  const size_t ownedBytes = ownedCount * typeSize;
  char* scratch = static_cast<char*>(comm->directA2aScratch);
  void* reducedShard = scratch + (size_t)comm->nRanks * ownedBytes;
  ncclResult_t ret = ncclSuccess;

  // ReduceScatter: every rank sends each peer the shard owned by that peer,
  // while receiving all contributions for its own shard into rank-indexed slots.
  void* selfSlot = scratch + (size_t)comm->rank * ownedBytes;
  const char* sendBytes = static_cast<const char*>(sendbuff);
  CUDACHECK(cudaMemcpyAsync(selfSlot, sendBytes + ownedOffset * typeSize, ownedBytes,
                            cudaMemcpyDeviceToDevice, stream));

  rccl::Recorder::instance().skip(true);
  ret = ncclGroupStart();
  if (ret == ncclSuccess) {
    for (int peer = 0; peer < comm->nRanks && ret == ncclSuccess; ++peer) {
      if (peer == comm->rank) continue;
      const size_t peerCount = directA2aChunkCount(count, comm->nRanks, peer);
      const size_t peerOffset = directA2aChunkOffset(count, comm->nRanks, peer);
      ret = ncclSend(sendBytes + peerOffset * typeSize, peerCount, datatype, peer, comm, stream);
      if (ret == ncclSuccess) {
        ret = ncclRecv(scratch + (size_t)peer * ownedBytes, ownedCount, datatype, peer, comm, stream);
      }
    }
    ncclResult_t groupEndResult = ncclGroupEnd();
    if (ret == ncclSuccess) ret = groupEndResult;
  }

  if (ret == ncclSuccess) {
    ret = launchDirectA2aSumByType(scratch, reducedShard, ownedCount, comm->nRanks, datatype, stream);
  }

  // AllGather: publish the reduced local shard and place each peer's shard
  // directly into its final offset. Self is copied locally to avoid self P2P.
  if (ret == ncclSuccess) ret = ncclGroupStart();
  if (ret == ncclSuccess) {
    for (int peer = 0; peer < comm->nRanks && ret == ncclSuccess; ++peer) {
      if (peer == comm->rank) continue;
      const size_t peerCount = directA2aChunkCount(count, comm->nRanks, peer);
      const size_t peerOffset = directA2aChunkOffset(count, comm->nRanks, peer);
      ret = ncclSend(reducedShard, ownedCount, datatype, peer, comm, stream);
      if (ret == ncclSuccess) {
        ret = ncclRecv(static_cast<char*>(recvbuff) + peerOffset * typeSize, peerCount, datatype, peer, comm, stream);
      }
    }
    ncclResult_t groupEndResult = ncclGroupEnd();
    if (ret == ncclSuccess) ret = groupEndResult;
  }
  rccl::Recorder::instance().skip(false);
  NCCLCHECK(ret);

  CUDACHECK(cudaMemcpyAsync(static_cast<char*>(recvbuff) + ownedOffset * typeSize, reducedShard, ownedBytes,
                            cudaMemcpyDeviceToDevice, stream));
  INFO(NCCL_COLL, "AllReduce: taking standalone DIRECT_A2A two-shot path: rank=%d/%d count=%zu datatype=%d bytes=%zu",
       comm->rank, comm->nRanks, count, (int)datatype, bytes);
  return ncclSuccess;
}

} // namespace

ncclResult_t rcclDirectA2aAllReduceCommInit(ncclComm* comm) {
  if (comm == nullptr || !rcclParamDirectA2aEnable()) return ncclSuccess;
  if (!IsArchMatch(comm->archName, "gfx1151") || comm->nRanks < RCCL_DIRECT_A2A_MIN_RANKS ||
      comm->nRanks > RCCL_DIRECT_A2A_MAX_RANKS || comm->nNodes != comm->nRanks || comm->minLocalRanks != 1 ||
      comm->maxLocalRanks != 1) {
    return ncclSuccess;
  }

  const size_t oneShotScratchBytes = (size_t)comm->nRanks * directA2aOneShotThreshold(comm->nRanks);
  const size_t maxTwoShotChunkBytes =
    (directA2aMaxBytes(comm->nRanks) + (size_t)comm->nRanks - 1) / (size_t)comm->nRanks + sizeof(double);
  const size_t twoShotScratchBytes = ((size_t)comm->nRanks + 1) * maxTwoShotChunkBytes;
  const size_t scratchBytes = std::max(oneShotScratchBytes, twoShotScratchBytes);
  ncclResult_t res = ncclCudaMalloc(&comm->directA2aScratch, scratchBytes, comm->memManager);
  if (res != ncclSuccess) {
    comm->directA2aScratch = nullptr;
    comm->directA2aScratchBytes = 0;
    INFO(NCCL_INIT, "DIRECT_A2A standalone scratch allocation failed; falling back to standard AllReduce");
    return ncclSuccess;
  }
  comm->directA2aScratchBytes = scratchBytes;
  INFO(NCCL_INIT, "DIRECT_A2A standalone path initialized: rank=%d/%d scratchBytes=%zu", comm->rank, comm->nRanks,
       scratchBytes);
  return ncclSuccess;
}

ncclResult_t rcclDirectA2aAllReduceCommFini(ncclComm* comm) {
  if (comm != nullptr && comm->directA2aScratch != nullptr) {
    NCCLCHECK(ncclCudaFree(comm->directA2aScratch, comm->memManager));
    comm->directA2aScratch = nullptr;
    comm->directA2aScratchBytes = 0;
  }
  return ncclSuccess;
}

bool rcclDirectA2aAllReduceEligible(ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op) {
  if (comm == nullptr || comm->bootstrap == nullptr || comm->directA2aScratch == nullptr) return false;
  // The standalone path launches its local reduction immediately after the
  // internal P2P group completes, which requires synchronous group completion.
  if (!comm->config.blocking) return false;
  if (!IsArchMatch(comm->archName, "gfx1151")) return false;
  if (comm->nRanks < RCCL_DIRECT_A2A_MIN_RANKS || comm->nRanks > RCCL_DIRECT_A2A_MAX_RANKS ||
      comm->nNodes != comm->nRanks || comm->minLocalRanks != 1 || comm->maxLocalRanks != 1) {
    return false;
  }
  if (count == 0 || op != ncclSum) return false;
  if (datatype != ncclFloat16 && datatype != ncclBfloat16 && datatype != ncclFloat32) return false;

  const size_t typeSize = ncclTypeSize(datatype);
  if (count > directA2aMaxBytes(comm->nRanks) / typeSize) return false;
  const size_t bytes = count * typeSize;
  if (bytes <= directA2aOneShotThreshold(comm->nRanks)) {
    return (size_t)comm->nRanks * bytes <= comm->directA2aScratchBytes;
  }

  const size_t maxChunkCount = (count + (size_t)comm->nRanks - 1) / (size_t)comm->nRanks;
  const size_t maxChunkBytes = maxChunkCount * typeSize;
  return ((size_t)comm->nRanks + 1) * maxChunkBytes <= comm->directA2aScratchBytes;
}

ncclResult_t rcclDirectA2aAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  if (!rcclDirectA2aAllReduceEligible(comm, count, datatype, op)) return ncclInvalidUsage;

  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes <= directA2aOneShotThreshold(comm->nRanks)) {
    return directA2aOneShot(sendbuff, recvbuff, count, datatype, comm, stream);
  }
  return directA2aTwoShot(sendbuff, recvbuff, count, datatype, comm, stream);
}
