/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_reduce/dda_all_reduce.h"

#include "algorithms/dda/all_reduce/all_reduce_dda_fabric_hier.h"
#include "algorithms/dda/dda_init_detail.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "rccl_common.h"

#include <hip/hip_ext.h>

#include <algorithm>
#include <cstddef>

namespace {

using dda::common::HierFusedLayout;
using nccl_dda_detail::DdaFabricBarrierState;

size_t hierElemSize(ncclDataType_t datatype) {
  switch (datatype) {
  case ncclFloat32:
    return sizeof(float);
  case ncclFloat16:
  case ncclBfloat16:
    return 2;
  default:
    return 0;
  }
}

// On 2 gfx1250 DPX hosts, 64 MiB rounds beat both 16 MiB and 128 MiB rounds and one round as large as scratch allows.
constexpr size_t kHierRoundBytes = size_t{64} << 20;

// Elements per round: at most kHierRoundBytes, two banks fit in scratch, and a slice stays a whole number of 16-byte
// vectors.
size_t hierRoundCount(const ncclComm* comm, size_t count, size_t elemSize, int ranksPerHost, int nHosts) {
  const size_t granule = static_cast<size_t>(ranksPerHost) * (16 / elemSize);
  const size_t scratchElems = comm->ddaScratchBytes / elemSize;
  const size_t fit = scratchElems / 2 * ranksPerHost / (2 * ranksPerHost + nHosts);
  return std::min(count, std::min(fit, kHierRoundBytes / elemSize) / granule * granule);
}

template <typename T>
ncclResult_t launchHier(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm, cudaStream_t stream,
                        const HierFusedLayout& lay) {
  const size_t roundCount = hierRoundCount(comm, count, sizeof(T), lay.ranksPerHost, lay.nHosts);
  const size_t slice = roundCount / lay.ranksPerHost;
  const auto geom = dda::common::getGridAndBlockDims(slice, sizeof(T), (size_t)rcclDdaFabricVmmBlocks(comm));
  auto* barrierState = static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  INFO(NCCL_COLL, "DDA fabric hierarchical AllReduce: count %zu round %zu hosts %d ranksPerHost %d grid %u block %u",
       count, roundCount, lay.nHosts, lay.ranksPerHost, geom.first.x, geom.second.x);
  auto launch = [&](auto kernel) {
    hipExtLaunchKernelGGL(kernel, geom.first, geom.second, 0, stream, /*startEvent=*/nullptr,
                          rcclTakeAddonStopEvent(comm), /*flags=*/0, reinterpret_cast<T* const*>(comm->ddaPeerPtrsDev),
                          static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, roundCount, lay,
                          barrierState->barrierHost);
  };
  if (lay.ranksPerHost == 8 && lay.nHosts == 1) {
    launch(dda::common::ddaAllReduceHierFabric<T, 8, 1>);
  } else if (lay.ranksPerHost == 8 && lay.nHosts == 2) {
    launch(dda::common::ddaAllReduceHierFabric<T, 8, 2>);
  } else if (lay.ranksPerHost == 8 && lay.nHosts == 4) {
    launch(dda::common::ddaAllReduceHierFabric<T, 8, 4>);
  } else if (lay.ranksPerHost == 8 && lay.nHosts == 8) {
    launch(dda::common::ddaAllReduceHierFabric<T, 8, 8>);
  } else {
    launch(dda::common::ddaAllReduceHierFabric<T, 0, 0>);
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

} // namespace

bool ncclAllReduceDdaFabricHierEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                        ncclDataType_t datatype, ncclRedOp_t op, int nHosts, int ranksPerHost) {
  const size_t elemSize = hierElemSize(datatype);
  if (comm == nullptr || op != ncclSum || elemSize == 0 || count == 0) return false;
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaFabricBarrierState == nullptr || comm->ddaScratch == nullptr ||
      comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (nHosts < 1 || nHosts > dda::common::kHierFusedMaxHosts || ranksPerHost < 2 ||
      ranksPerHost > dda::common::kHierFusedMaxRanksPerHost) {
    return false;
  }
  if (!dda::common::ddaUserBuffers16ByteAligned(sendbuff, recvbuff)) return false;
  const size_t granule = static_cast<size_t>(ranksPerHost) * (16 / elemSize);
  return count % granule == 0 && hierRoundCount(comm, count, elemSize, ranksPerHost, nHosts) >= granule;
}

ncclResult_t ncclAllReduceDdaFabricHier(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                        ncclComm* comm, cudaStream_t stream, int nHosts, int ranksPerHost,
                                        int hostIndex, int localRank, const int* intraRanks, const int* interRanks) {
  HierFusedLayout lay{};
  lay.nHosts = nHosts;
  lay.ranksPerHost = ranksPerHost;
  lay.hostIndex = hostIndex;
  lay.localRank = localRank;
  std::copy(intraRanks, intraRanks + ranksPerHost, lay.intra);
  std::copy(interRanks, interRanks + nHosts, lay.inter);
  switch (datatype) {
  case ncclFloat32:
    return launchHier<float>(sendbuff, recvbuff, count, comm, stream, lay);
  case ncclFloat16:
    return launchHier<half>(sendbuff, recvbuff, count, comm, stream, lay);
  case ncclBfloat16:
    return launchHier<bf16>(sendbuff, recvbuff, count, comm, stream, lay);
  default:
    return ncclInvalidArgument;
  }
}
