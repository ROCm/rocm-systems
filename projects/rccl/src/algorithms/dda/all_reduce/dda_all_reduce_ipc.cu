/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_reduce/dda_all_reduce.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/all_reduce/all_reduce_dda.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/ipc/ipc_gpu_barrier.h"
#include "algorithms/dda/dda_init_detail.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <utility>

namespace {

using nccl_dda_detail::DdaIpcBarrierState;
using nccl_dda_detail::ddaMaxNBlocksForScratch;
using nccl_dda_detail::kDdaNranks;

/** Flat below this size; tree above (see ddaAllReduceFlatIpc / ddaAllReduceTreeIpc). */
constexpr size_t kDdaFlatTreeThresholdBytes = 1ULL << 18;

// Single source of the launch geometry: grid/block for `count` elements of
// `typeSize` bytes, capped by the scratch-derived block count.
static inline std::pair<dim3, dim3> ddaAllReduceIpcGeom(size_t count, int typeSize) {
  return dda::common::getGridAndBlockDims(count, typeSize, ddaMaxNBlocksForScratch());
}

template <typename T, int NRANKS>
static ncclResult_t ncclAllReduceDdaIpcLaunch(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                              cudaStream_t stream) {
  // Defends the dispatcher's invariant in ncclAllReduceDdaIpcTyped(): NRANKS is
  // either 0 (use comm->nRanks at runtime) or exactly comm->nRanks (the
  // compile-time kDdaNranks specialisation). A dispatch bug that picked the
  // wrong instantiation for the live rank count would otherwise silently read
  // peer slots the comm was never told about -- ipc_init.cu pre-zeroes the
  // unused tail of the peer table rather than leaving it as garbage, so this
  // would produce a wrong numeric result on hardware, not a crash.
  if (NRANKS != 0 && NRANKS != comm->nRanks) {
    WARN("DDA IPC allreduce: dispatch bug, instantiated for %d ranks but comm has %d", NRANKS, comm->nRanks);
    return ncclInternalError;
  }
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaIpcBarrierState == nullptr) {
    return ncclInvalidUsage;
  }
  if (count * sizeof(T) > comm->ddaScratchBytes) {
    WARN("DDA IPC allreduce: element count %zu needs %zu bytes; comm scratch is %zu bytes", count, count * sizeof(T),
         comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  // NRANKS == 0 selects the runtime kernel; the clique size then comes from the
  // comm. Use nRanks (never the NRANKS template parameter) for any arithmetic
  // here, or the runtime instantiation divides by zero.
  const int nRanks = (NRANKS > 0) ? NRANKS : comm->nRanks;
  if (nRanks <= 0) {
    WARN("DDA IPC allreduce: invalid nRanks %d", nRanks);
    return ncclInvalidUsage;
  }

  const size_t sizeBytes = count * sizeof(T);
  const bool wantTree = sizeBytes > kDdaFlatTreeThresholdBytes;
  // ncclAllReduceDdaIpcEligible() already gates every reachable caller on
  // ncclAllReduceDdaIpcTreeEligible(), but Launch is templated on a
  // compile-time NRANKS and reads a runtime nRanks/count independently of
  // that gate, so it re-derives the same answer here rather than trusting
  // the caller -- see the function for why both call it instead of each
  // keeping their own copy of the condition.
  const bool treeOk = wantTree && ncclAllReduceDdaIpcTreeEligible(count, nRanks, sizeof(T));

  if (wantTree && !treeOk) {
    INFO(NCCL_ALL, "DDA IPC: size %zu B > 256KB but count %zu with %d ranks is not tree-eligible; using flat kernel",
         sizeBytes, count, nRanks);
  }

  auto gridBlock = ddaAllReduceIpcGeom(count, sizeof(T));
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  dda::common::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  if (treeOk) {
    CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    dda::common::ddaAllReduceTreeIpc<T, NRANKS, false><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks,
      barrierHost, nullptr);
  } else {
    dda::common::ddaAllReduceFlatIpc<T, NRANKS, false><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks,
      barrierHost, nullptr);
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

// Dispatch to the template instantiation for the active participant count.
// ncclAllReduceDdaIpcEligible() guarantees comm->nRanks is in [2, kDdaNranks]
// (exactly kDdaNranks when RCCL_DDA_NRANKS_RELAX is off) before we get here.
//
// Only the default kDdaNranks clique gets a compile-time specialisation, keeping
// that path bit- and perf-identical to baseline. Every other supported count uses
// the NRANKS_CT == 0 runtime kernel, which the CollCommon reduceScatter/allGather
// helpers unroll 8-wide. Specialising all seven counts instead would compile
// 2 kernels x 7 counts x 3 types for each of the DEFAULT_GPUS targets, for a path
// that is default-off and confined to gfx942/gfx950.
template <typename T>
static ncclResult_t ncclAllReduceDdaIpcTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                             cudaStream_t stream) {
  if (!ncclDdaIpcNranksSupported(comm->nRanks)) {
    WARN("DDA IPC allreduce: unsupported nRanks %d", comm->nRanks);
    return ncclInvalidUsage;
  }
  if (comm->nRanks == kDdaNranks) {
    return ncclAllReduceDdaIpcLaunch<T, kDdaNranks>(sendbuff, recvbuff, count, comm, stream);
  }
  return ncclAllReduceDdaIpcLaunch<T, 0>(sendbuff, recvbuff, count, comm, stream);
}

} // namespace

// Consolidates what used to be two independent copies of this condition (one
// here, one in the eligibility gate below) that could silently drift apart;
// see the declaration in dda_all_reduce.h for the full rationale.
bool ncclAllReduceDdaIpcTreeEligible(size_t count, int nRanks, size_t typeSize) {
  if (nRanks <= 0) {
    return false;
  }
  if (count % static_cast<size_t>(nRanks) != 0) {
    return false;
  }
  // Two-shot/tree path: each rank reduces count/nRanks elements, so that
  // per-rank slice must also be 16-byte aligned for the tree kernel's
  // vectorized (uint4) loads.
  return ((count / static_cast<size_t>(nRanks)) * typeSize) % 16 == 0;
}

bool ncclAllReduceDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr) {
    return false;
  }
  // IPC path: requires its own handler + barrier state, a single node, and a
  // supported participant count (kDdaNranks by default; any 2..kDdaNranks when
  // RCCL_DDA_NRANKS_RELAX=1). Only kDdaNranks gets a compile-time specialisation;
  // every other count uses the NRANKS == 0 runtime kernel.
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (!ncclDdaIpcNranksSupported(comm->nRanks)) {
    return false;
  }
  // Checks shared by both DDA all-reduce backends.
  if (comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }
  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes > comm->ddaScratchBytes) {
    return false;
  }
  if (bytes % 16) {
    // 16-byte alignment: the DDA kernels do 16-byte vectorized loads.
    return false;
  }
  if (bytes > kDdaFlatTreeThresholdBytes && !ncclAllReduceDdaIpcTreeEligible(count, comm->nRanks, ncclTypeSize(datatype))) {
    return false;
  }
  return true;
}

uint32_t ncclAllReduceDdaIpcBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  (void)comm;
  const auto grid = ddaAllReduceIpcGeom(count, ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllReduceDdaIpc(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclAllReduceDdaIpcTyped<float>(sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllReduceDdaIpcTyped<half>(sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllReduceDdaIpcTyped<bf16>(sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
