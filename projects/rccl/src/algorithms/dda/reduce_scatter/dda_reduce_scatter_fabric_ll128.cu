/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA fabric reduce-scatter.
 * Reduce-scatter analogue of the LL128 one-shot launcher in
 * dda_all_reduce_fabric_ll.cu. Messages up to the peer-parallel cap take the
 * ddaReduceScatterFabricLL128PeerPar kernel (reduce_scatter_dda_fabric_ll128_peerpar.h),
 * larger ones the batched ddaReduceScatterFabricLL128 kernel (reduce_scatter_dda_fabric_ll128.h).
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"

#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll128.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll128_peerpar.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "param.h"
#include "rccl_common.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Batched kernel block size, overriding the table: a multiple of the warp up to kDdaLL128RsMaxThreads, else ignored.
RCCL_PARAM(DdaLL128RSThreads, "DDA_LL128_RS_THREADS", 0);
// Largest message the peer-parallel kernel takes, overriding the arch table. 0 keeps the batched kernel at every size.
RCCL_PARAM(DdaLL128RSPeerMaxBytes, "DDA_LL128_RS_PEER_MAX_BYTES", kDdaThresholdUnset);

namespace {

using dda::common::ddaBankSize;
using dda::common::ddaLL128RsPeerWarps;
using dda::common::ddaLL128RsSlotWords;
using dda::common::ddaLL128Slices;
using dda::common::ddaLL128ValidThreads;
using dda::common::kDdaLL128RsMaxThreads;
using dda::common::kDdaLL128RsPeerMaxThreads;
using dda::common::kDdaLL128RsPeerMaxWarps;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireWordsPerSlice;

// A valid DDA_LL128_RS_THREADS, else the arch table's block size if the launch bound allows it, else the bound.
static inline unsigned ddaLL128RSThreads(const ncclComm* comm) {
  const auto fits = [](int64_t v) { return ddaLL128ValidThreads(v) && v <= (int64_t)kDdaLL128RsMaxThreads; };
  const int64_t user = rcclParamDdaLL128RSThreads();
  if (fits(user)) {
    return (unsigned)user;
  }
  const int tuned = rcclDdaFabricRsLL128Threads(comm);
  return fits(tuned) ? (unsigned)tuned : kDdaLL128RsMaxThreads;
}

// Peer-parallel while the whole message is within DDA_LL128_RS_PEER_MAX_BYTES, or the arch table's cap when unset.
static inline bool ddaLL128RsPeerParallel(const ncclComm* comm, size_t shardBytes) {
  const size_t bytes = shardBytes * (size_t)comm->nRanks;
  const int64_t user = rcclParamDdaLL128RSPeerMaxBytes();
  if (user != kDdaThresholdUnset) {
    return user > 0 && bytes <= (size_t)user;
  }
  return bytes <= rcclDdaFabricRsLL128PeerMaxBytes(comm);
}

// Launch shape of whichever kernel the message takes.
struct DdaLL128RsGeom {
  dim3 grid;
  dim3 block;
  bool peerPar;
};

// A warp (batched) or a warp group (peer-parallel) carries one slice, so blocks = ceil(slices / slicesPerBlock),
// capped by the grid budget. More slices than the grid holds just means more passes through the slice loop.
static inline DdaLL128RsGeom ddaReduceScatterFabricLL128Geom(ncclComm* comm, size_t shardBytes) {
  const size_t slices = ddaLL128Slices(shardBytes);
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  const bool peerPar = ddaLL128RsPeerParallel(comm, shardBytes);
  unsigned threads = 0;
  size_t slicesPerBlock = 0;
  if (peerPar) {
    // A group is a warp per rank. A block starts as one group and doubles its groups up to kDdaLL128RsPeerMaxThreads
    // only while one group per block would pass the block cap.
    const unsigned groupThreads =
      (unsigned)ddaLL128RsPeerWarps(comm->nRanks, kDdaLL128RsPeerMaxWarps) * (unsigned)kDdaLL128Warp;
    threads = groupThreads;
    while (threads * 2 <= (unsigned)kDdaLL128RsPeerMaxThreads) {
      const size_t groups = threads / groupThreads;
      if ((slices + groups - 1) / groups <= (size_t)nBlocksMax) {
        break;
      }
      threads *= 2;
    }
    slicesPerBlock = threads / groupThreads;
  } else {
    threads = ddaLL128RSThreads(comm);
    slicesPerBlock = threads / (unsigned)kDdaLL128Warp;
  }
  unsigned blocks = (unsigned)std::min<size_t>((slices + slicesPerBlock - 1) / slicesPerBlock, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  // flatBlockId (blockIdx.x) must stay within the device epoch array.
  if ((int)blocks > comm->ddaLLEpochLen) {
    blocks = (unsigned)comm->ddaLLEpochLen;
    if (blocks == 0) {
      blocks = 1;
    }
  }
  return {dim3(blocks), dim3(threads), peerPar};
}

template <typename T>
static ncclResult_t ncclReduceScatterDdaFabricLL128Typed(const void* sendbuff, void* recvbuff, size_t recvcount,
                                                         ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t shardBytes = recvcount * sizeof(T);
  const size_t slices = ddaLL128Slices(shardBytes);
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);

  const DdaLL128RsGeom geom = ddaReduceScatterFabricLL128Geom(comm, shardBytes);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Same epoch counter every other LL/LL128 tier uses: they share a scratch
  // layout, so one monotonic flag is what keeps either from accepting a line
  // the other left.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  // Alignment differs between ranks, so it only picks this rank's user-buffer access, never the wire.
  const bool aligned = dda::common::ddaUserBuffers16ByteAligned(sendbuff, recvbuff);
  INFO(NCCL_COLL,
       "DDA fabric ReduceScatter LL128: nRanks=%d shardBytes=%zu slices=%zu grid=%u block=%u "
       "(%s, bankSize=%zu, aligned=%d)",
       nRanks, shardBytes, slices, geom.grid.x, geom.block.x, geom.peerPar ? "peer-parallel" : "warp-per-slice",
       bankSize, (int)aligned);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  // The two kernels take the same arguments.
  const auto launch = [&](auto batched, auto peerPar) {
    hipExtLaunchKernelGGL(geom.peerPar ? peerPar : batched, geom.grid, geom.block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          shardBytes, comm->rank, nRanks, epochDev, epochLen, slices, bankSize, params);
  };
  // NRANKS_CT 4/8: unrolled peer loops; 0: runtime fallback.
  const auto launchRanks = [&](auto n) {
    constexpr int kN = decltype(n)::value;
    dda::common::ddaPickAligned(aligned, [&](auto a) {
      constexpr bool kAligned = decltype(a)::value;
      launch(dda::common::ddaReduceScatterFabricLL128<T, kN, kAligned>,
             dda::common::ddaReduceScatterFabricLL128PeerPar<T, kN, kAligned>);
    });
  };
  switch (nRanks) {
  case 4:
    launchRanks(std::integral_constant<int, 4>{});
    break;
  case 8:
    launchRanks(std::integral_constant<int, 8>{});
    break;
  default:
    launchRanks(std::integral_constant<int, 0>{});
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclReduceScatterDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                             ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->ddaLLEpochDev == nullptr || comm->ddaLLEpochLen < 1) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (recvcount == 0) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return false;
  }

  const size_t shardBytes = recvcount * ncclTypeSize(datatype);

  // The LL128 line format packs 16B-aligned data with no chunk straddling a
  // line, so a partial 16B chunk has nowhere to go. Buffer alignment is not
  // checked: it differs between ranks, and every rank has to pick the same tier.
  if (shardBytes % 16 != 0) {
    return false;
  }

  // Wrap compares the full message (nRanks shards) against the arch table; keep
  // that same total here so direct Eligible() callers agree with the selector.
  if (shardBytes * (size_t)comm->nRanks > rcclDdaLL128Threshold(comm, ncclFuncReduceScatter)) {
    return false;
  }

  // The shard has to fit the slices one rank's slot holds.
  const size_t slotWords = ddaLL128RsSlotWords(ddaBankSize(comm->ddaScratchBytes), comm->nRanks);
  if (ddaLL128Slices(shardBytes) * (size_t)kDdaLL128WireWordsPerSlice > slotWords) {
    return false;
  }

  return true;
}

uint32_t ncclReduceScatterDdaFabricLL128Blocks(ncclComm* comm, size_t recvcount, ncclDataType_t datatype) {
  const dim3 grid = ddaReduceScatterFabricLL128Geom(comm, recvcount * ncclTypeSize(datatype)).grid;
  return grid.x * grid.y;
}

ncclResult_t ncclReduceScatterDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t recvcount,
                                             ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
                                             cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclReduceScatterDdaFabricLL128Typed<float>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat16:
    return ncclReduceScatterDdaFabricLL128Typed<half>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclBfloat16:
    return ncclReduceScatterDdaFabricLL128Typed<bf16>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat8e4m3:
    return ncclReduceScatterDdaFabricLL128Typed<dda::common::fp8e4m3>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat8e5m2:
    return ncclReduceScatterDdaFabricLL128Typed<dda::common::fp8e5m2>(sendbuff, recvbuff, recvcount, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
