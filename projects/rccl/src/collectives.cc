/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "config/collconfig.h"
#include "enqueue.h"
#include "graph/topo.h"
#include "nccl.h"
#include "api_trace.h"
#include "nvtx_payload_schemas.h"
#include "device/hierarchical_shuffle.h"
#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"
#include "algorithms/dda/all_gather/dda_all_gather.h"
#include "algorithms/dda/alltoall/dda_alltoall.h"
#include "sym_kernels.h"
#include "dev_runtime.h"
#include "ce_coll.h"
#include "alltoallv_meta.h"
#include "strongstream.h"

#ifdef ENABLE_ROCSHMEM
#include <rocshmem/rocshmem.hpp>
#endif

using namespace rccl;

const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {
  case ncclFuncAllGather:
    return "AllGather";
  case ncclFuncAllReduce:
    return "AllReduce";
  case ncclFuncAlltoAll:
    return "AlltoAll";
  case ncclFuncAlltoAllv:
    return "AlltoAllv";
  case ncclFuncBroadcast:
    return "Broadcast";
  case ncclFuncGather:
    return "Gather";
  case ncclFuncRecv:
    return "Recv";
  case ncclFuncReduce:
    return "Reduce";
  case ncclFuncReduceScatter:
    return "ReduceScatter";
  case ncclFuncScatter:
    return "Scatter";
  case ncclFuncSendRecv:
    return "SendRecv";
  case ncclFuncSend:
    return "Send";
  case ncclFuncPutSignal:
    return "PutSignal";
  case ncclFuncSignal:
    return "Signal";
  case ncclFuncWaitSignal:
    return "WaitSignal";
  default:
    return "Invalid";
  }
}

const char* ncclDevRedOpToString(ncclDevRedOp_t op) {
  switch (op) {
  case ncclDevSum:
    return "Sum";
  case ncclDevProd:
    return "Prod";
  case ncclDevMinMax:
    return "MinMax";
  case ncclDevPreMulSum:
    return "PreMulSum";
  case ncclDevSumPostDiv:
    return "SumPostDiv";
  default:
    return "Unknown";
  }
}

const char* ncclDatatypeToString(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
    return "ncclInt8";
  case ncclInt32:
    return "ncclInt32";
  case ncclUint32:
    return "ncclUint32";
  case ncclInt64:
    return "ncclInt64";
  case ncclUint64:
    return "ncclUint64";
  case ncclFloat16:
    return "ncclFloat16";
  case ncclFloat32:
    return "ncclFloat32";
  case ncclFloat64:
    return "ncclFloat64";
  case ncclBfloat16:
    return "ncclBfloat16";
  case ncclFloat8e4m3:
    return "ncclFloat8e4m3";
  case ncclFloat8e5m2:
    return "ncclFloat8e5m2";
  default:
    return "Unknown";
  }
}

const char* ncclAlgoToString(int algo) {
  switch (algo) {
  case NCCL_ALGO_TREE:
    return "TREE";
  case NCCL_ALGO_RING:
    return "RING";
  case NCCL_ALGO_COLLNET_DIRECT:
    return "COLLNET_DIRECT";
  case NCCL_ALGO_COLLNET_CHAIN:
    return "COLLNET_CHAIN";
  case NCCL_ALGO_NVLS:
    return "NVLS";
  case NCCL_ALGO_NVLS_TREE:
    return "NVLS_TREE";
  case NCCL_ALGO_PAT:
    return "PAT";
  default:
    return "Unknown";
  }
}

const char* ncclProtoToString(int proto) {
  switch (proto) {
  case NCCL_PROTO_LL:
    return "LL";
  case NCCL_PROTO_LL128:
    return "LL128";
  case NCCL_PROTO_SIMPLE:
    return "SIMPLE";
  default:
    return "Unknown";
  }
}

static inline ncclResult_t ncclAllGatherConfigImpl(const void* sendbuff, void* recvbuff, size_t sendcount,
                                                   ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream,
                                                   const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncAllGather, "AllGather",
                          sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
                          ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
         ncclComm_t comm, cudaStream_t stream);

// Direct AllGather: posts Send/Recv to every peer (including self) for every
// rank using the same iteration order on all ranks. Mirrors the AlltoAll
// scheduling path so that, when RCCL_P2P_BATCH_ENABLE=1, all ranks emit the
// same sequence of (sendRank, recvRank) rounds and therefore the same fused
// ncclDevWorkBatch composition.
//
// We deliberately keep this minimal:
//   - No (rank + r) % nRanks rotation: all ranks visit peers in order 0..N-1.
//   - No in-place self-peer skip: send/recv to self is always posted; the
//     device kernel handles isCopy = (sendRank == self) as a local memcpy
//     (see device/sendrecv.h).
//   - Posting is delegated to taskAppend() via a single ncclEnqueueCheck
//     call. taskAppend() then loops once and calls p2pTaskAppend directly,
//     the same way ncclAlltoAll does, avoiding the per-peer ncclSend/ncclRecv
//     overhead (ArgsCheck, Recorder, profiler events, group start/end
//     internal) that previously made cross-rank batch composition fragile at
//     scale.
static ncclResult_t rcclDirectAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                        ncclComm_t comm, cudaStream_t stream) {
  struct ncclInfo info = {
    ncclFuncAllGather,    "AllGather",          sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream,
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS, nullptr
  };
  info.useDirect = true;
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllGatherConfig, const void* sendbuff, void* recvbuff, size_t sendcount,
         ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclAllGatherConfig(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                 ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) {
  // Just pass the size of one message and not the total bytes sent/received.
  NVTX3_FUNC_WITH_PARAMS(AllGatherConfig, NcclNvtxParamsAllGather,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcount * ncclTypeSize(datatype)));
  return ncclAllGatherConfigImpl(sendbuff, recvbuff, sendcount, datatype, comm, stream, config);
}

static inline ncclResult_t ncclAlltoAllConfigImpl(const void* sendbuff, void* recvbuff, size_t count,
                                                  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream,
                                                  const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncAlltoAll, "AlltoAll",
                          sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
                          ALLTOALL_CHUNKSTEPS, ALLTOALL_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

RCCL_PARAM_DECLARE(ForceCeAllReduce);

// rcclDdaEnabled() is now in rccl_wrap.cc (declared in rccl_common.h)

// Decides whether ncclAllReduce_impl takes the DDA path for this call. Kept as a small named helper
// (rather than an inline expression) so the dispatch decision is reachable from host unit tests; the
// logic is identical to the guard at the AllReduce call site below.
//
// `ceAllReduceAllowed` is the caller's single source of truth for "will CE AllReduce actually service
// this call" -- computed once at the call site from rcclUseCeAllReduce() plus whatever additional
// gating the CE AllReduce implementation requires (graph latch, ncclGroupDepth, force/symReg
// eligibility, etc). This helper does not re-derive CE eligibility itself: threading the same boolean
// through both the early CE return and this guard keeps the two decisions from silently drifting apart
// as CE AllReduce's own eligibility rules evolve.
bool rcclAllReduceShouldTakeDdaPath(const ncclComm* comm, size_t count, ncclDataType_t datatype, bool symEligible,
                                    bool ceAllReduceAllowed) {
  const size_t msgBytes = count * ncclTypeSize(datatype);
  // gfx1250 DDA fabric AR is bounded by rcclDdaEnabled (RCCL_DDA_THRESHOLD) and the per-tier
  // thresholds, so it may claim the full range regardless of CE eligibility -- ddaFabricArch1250
  // forces this branch unconditionally. On other arches, yield to DDA whenever CE won't actually
  // service this call; yielding on message size alone left comms without CE's prerequisites (e.g.
  // gfx950 with symmetricSupport off) with no DDA and no CE, falling back to the generic ring/tree
  // kernel across the whole 4 MiB+ range that DDA still wins.
  const bool ddaFabricArch1250 = IsArchMatch(comm->archName, "gfx1250");
  return !symEligible && (ddaFabricArch1250 || !ceAllReduceAllowed) && rcclDdaEnabled(comm, msgBytes, 8388608);
}

// Check if symmteric kernels is requested for this collective
bool isSymmetricKernelRequested(ncclComm* comm, ncclFunc_t coll, int symkOp, ncclDataType_t datatype, size_t nElts,
                                const void* sendbuff, void* recvbuff) {
  if (comm == nullptr || !comm->symmetricSupport) return false;
  if (ncclSymkInitOnce(comm) != ncclSuccess) return false;
  if (!ncclSymkAvailable(comm, coll, symkOp, datatype, nElts)) return false;

  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;
  ncclDevrFindWindow(comm, sendbuff, &sendWin);
  ncclDevrFindWindow(comm, recvbuff, &recvWin);
  return sendWin != nullptr && recvWin != nullptr && (sendWin->winFlags & NCCL_WIN_COLL_SYMMETRIC) &&
         (recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC);
}

static inline int hierarchicalShuffleNumBlocks(size_t totalBytes) {
  if (totalBytes <= (size_t)64 * 1024) return 8;
  if (totalBytes <= (size_t)16 * 1024 * 1024) return 16;
  return 32;
}

static ncclResult_t ncclHierarchicalAllGather_Impl(const void* sendbuff, void* recvbuff, size_t sendcount,
                                                   ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  if (sendcount == 0) return ncclSuccess;
  ncclComm* intraComm = comm->hierarchicalIntraComm;
  ncclComm* interComm = comm->hierarchicalInterComm;
  int localRanks = intraComm->nRanks; // Ranks per node
  int nNodes = interComm->nRanks; // Number of nodes
  size_t typeSize = ncclTypeSize(datatype);

  void* tempBuffer = comm->hierarchicalTempBuffer;
  const void* interSendBuff = sendbuff;
  size_t rankOffset = sendcount * typeSize;
  if (sendbuff == ((char*)recvbuff) + comm->rank * rankOffset) {
    CUDACHECK(hipMemcpyAsync(tempBuffer, sendbuff, rankOffset, hipMemcpyDeviceToDevice, stream));
    interSendBuff = tempBuffer;
  }

  // Step 1: Inter-node AllGather
  size_t interMsgSize = sendcount * nNodes * typeSize;
  if (nNodes <= 16 && rcclUseAllGatherDirect(interComm, interMsgSize)) {
    NCCLCHECK(rcclDirectAllGather(interSendBuff, recvbuff, sendcount, datatype, interComm, stream));
  } else {
    struct ncclInfo infoInterAG = {ncclFuncAllGather,
                                   "HierarchicalAllGather-Inter",
                                   interSendBuff,
                                   recvbuff,
                                   sendcount,
                                   datatype,
                                   ncclSum,
                                   0,
                                   interComm,
                                   stream,
                                   ALLGATHER_CHUNKSTEPS,
                                   ALLGATHER_SLICESTEPS,
                                   nullptr};
    NCCLCHECK(ncclEnqueueCheck(&infoInterAG));
  }

  // Step 2: Intra-node AllGather
  size_t intraSendCount = sendcount * nNodes;
  size_t intraMsgSize = intraSendCount * typeSize * localRanks;
  if (rcclUseAllGatherDirect(intraComm, intraMsgSize)) {
    // Use direct allgather
    NCCLCHECK(rcclDirectAllGather(recvbuff, tempBuffer, intraSendCount, datatype, intraComm, stream));
  } else {
    struct ncclInfo infoIntraAG = {ncclFuncAllGather,
                                   "HierarchicalAllGather-Intra",
                                   recvbuff,
                                   tempBuffer,
                                   intraSendCount,
                                   datatype,
                                   ncclSum,
                                   0,
                                   intraComm,
                                   stream,
                                   ALLGATHER_CHUNKSTEPS,
                                   intraComm->rcclUseOneSlice ? ALLGATHER_SLICESTEPS_SINGLE_NODE : ALLGATHER_SLICESTEPS,
                                   nullptr};
    NCCLCHECK(ncclEnqueueCheck(&infoIntraAG));
  }

  // Step 3: Shuffle tempBuffer (local-rank-major) -> recvbuff (node-major).
  size_t totalAGBytes = (size_t)nNodes * localRanks * rankOffset;
  int numBlocks = hierarchicalShuffleNumBlocks(totalAGBytes);
  hierarchicalShuffle<<<numBlocks, HIERARCHICAL_SHUFFLE_THREADS, 0, stream>>>((const char*)tempBuffer, (char*)recvbuff,
                                                                              rankOffset, nNodes, localRanks);
  CUDACHECK(hipGetLastError());

  return ncclSuccess;
}

ncclResult_t ncclAllGather_impl(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AllGather, NcclNvtxParamsAllGather,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcount * ncclTypeSize(datatype), datatype));
    // RCCL update slice steps for AllGather if single node
  const bool isGfx950 = IsArchMatch(comm->archName, "gfx950");

  int chunkSteps = (isGfx950 && comm->rcclUseOneSlice) ? 1 : ALLGATHER_CHUNKSTEPS;
  int sliceSteps = comm->rcclUseOneSlice ? (isGfx950 ? 1 : ALLGATHER_SLICESTEPS_SINGLE_NODE) : ALLGATHER_SLICESTEPS;
  struct ncclInfo info = {ncclFuncAllGather, "AllGather", sendbuff, recvbuff, sendcount,
                          datatype,          ncclSum,     0,        comm,     stream, /* Args */
                          chunkSteps,        sliceSteps,  nullptr};
  int nRanks, rank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  size_t msgSize = sendcount * ncclTypeSize(datatype) * nRanks;

  NCCLCHECK(Recorder::instance().record(rrAllGather, info));

  // Select the implementation once, in one place (rccl_wrap.cc). The same function
  // backs rcclGetCollImplInfo so rccl-tests attributes numbers to the backend that
  // actually ran. Symmetric-registered buffers are extracted downstream, so DDA is
  // gated on !symEligible inside the decision, exactly as before.
  struct rcclCollDecision decision;
  NCCLCHECK(rcclSelectAllGather(comm, sendbuff, recvbuff, sendcount, datatype, /*query=*/false,
                                /*graphCapturingHint=*/false, &decision));

  // Canonical selection line for addon backends (CE / DDA / Direct / Hier /
  // symmetric). Native kernels report via the enqueue.cc channel{Lo..Hi} tuning
  // line instead; this names the addon RCCL runs so rcclGetCollImplInfo can be
  // checked against it.
  if (comm->rank == 0 && decision.algo >= NCCL_NUM_ALGORITHMS) {
    const char* an = nullptr;
    rcclGetAlgoName(decision.algo, &an);
    INFO(NCCL_COLL, "AllGather impl selected: algo %s", an ? an : "?");
  }

  switch (decision.algo) {
  case RCCL_DDA_FABRIC_LL:
    INFO(NCCL_COLL,
         "AllGather: taking DDA fabric LL path: nRanks=%d nNodes=%d sendcount=%zu datatype=%d totalBytes=%zu",
         comm->nRanks, comm->nNodes, sendcount, (int)datatype, msgSize);
    NCCLCHECK(ncclAllGatherDdaFabricLL(sendbuff, recvbuff, sendcount, datatype, comm, stream));
    return ncclSuccess;
  case RCCL_DDA_FABRIC_LL128:
    INFO(NCCL_COLL,
         "AllGather: taking DDA fabric LL128 path: nRanks=%d nNodes=%d sendcount=%zu datatype=%d totalBytes=%zu",
         comm->nRanks, comm->nNodes, sendcount, (int)datatype, msgSize);
    NCCLCHECK(ncclAllGatherDdaFabricLL128(sendbuff, recvbuff, sendcount, datatype, comm, stream));
    return ncclSuccess;
  case RCCL_DDA_FABRIC_VMM:
    INFO(NCCL_COLL, "AllGather: taking DDA fabric (VMM) path: nRanks=%d nNodes=%d sendcount=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, sendcount, (int)datatype, sendcount * ncclTypeSize(datatype));
    NCCLCHECK(ncclAllGatherDdaFabric(sendbuff, recvbuff, sendcount, datatype, comm, stream));
    return ncclSuccess;
  case RCCL_DDA_IPC:
    NCCLCHECK(ncclAllGatherDdaIpc(sendbuff, recvbuff, sendcount, datatype, comm, stream));
    return ncclSuccess;
  case RCCL_HIERARCHICAL_ALLGATHER:
    return ncclHierarchicalAllGather_Impl(sendbuff, recvbuff, sendcount, datatype, comm, stream);
  case RCCL_DIRECT_ALLGATHER:
    INFO(NCCL_TUNING,
         "RCCL DIRECT ALLGATHER count = %zu, msgSize = %zu, comm = %p, stream = %p, rank = %d, sendbuff = %p, recvbuff "
         "= %p",
         sendcount, msgSize, comm, stream, rank, sendbuff, recvbuff);
    if (sendcount == 0) return ncclSuccess;
    // Mark the info so taskAppend posts this as A2A-style per-peer Send/Recv
    // P2P tasks (no peer rotation, no in-place self skip).
    info.useDirect = true;
    return ncclEnqueueCheck(&info);
  case RCCL_CE_REGISTERED:
    // CE dispatch happens in taskAppend(); just enqueue.
    return ncclEnqueueCheck(&info);
  default:
    // RCCL_AG_RING / native kernel algorithms go through the standard enqueue path.
    return ncclEnqueueCheck(&info);
  }
}

RCCL_PARAM(AlltoAllPivotEnable, "ALL_TO_ALL_PIVOT_ENABLE", 0);

NCCL_API(ncclResult_t, ncclAlltoAll, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAlltoAll_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                               ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AlltoAll, NcclNvtxParamsAlltoAll,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), datatype));

  NCCLCHECK(Recorder::instance().record(rrAllToAll, sendbuff, recvbuff, count, datatype, comm, stream));

  size_t rankOffset = count * ncclTypeSize(datatype);
  size_t rankAlign = rankOffset & ((~rankOffset) + 1);

  struct ncclInfo info;
  if (comm->topo->pivotA2AEnabled && comm->nChannels >= comm->topo->pivotA2ANumBiRings * 2 &&
      rankOffset >= 744 * 1024 && rankAlign != 4 && rcclParamAlltoAllPivotEnable()) {
    info = {ncclFuncAlltoAllPivot,
            "AlltoAllPivot",
            sendbuff,
            recvbuff,
            count,
            datatype,
            ncclSum,
            0,
            comm,
            stream, /* Args */
            ALLTOALL_PIVOT_CHUNKSTEPS,
            ALLTOALL_PIVOT_SLICESTEPS,
            nullptr};
  } else {
#ifdef ENABLE_ROCSHMEM
    size_t msgSize = count * ncclTypeSize(datatype) * comm->nRanks;
    if (rcclUseAlltoAllGda(comm) && msgSize <= comm->rocshmemThreshold) {
      struct ncclInfo info = {ncclFuncAlltoAllGda,
                              "AlltoAllGda",
                              sendbuff,
                              recvbuff,
                              count,
                              datatype,
                              ncclSum,
                              0,
                              comm,
                              stream,
                              ALLTOALL_PIVOT_CHUNKSTEPS,
                              ALLTOALL_PIVOT_SLICESTEPS,
                              nullptr};

      return ncclEnqueueCheck(&info);
    }
#endif // ENABLE_ROCSHMEM
    // alltoall does not need symEligible check as symmetric kernel is not supported for alltoall
    if (rcclDdaEnabled(comm, comm->nRanks * count * ncclTypeSize(datatype), kDdaAlltoAllGfx942ThresholdBytes,
                       kDdaAlltoAllGfx950ThresholdBytes, kDdaAlltoAllGfx1250ThresholdBytes)) {
      if (IsArchMatch(comm->archName, "gfx1250")) {
        const size_t a2aBytes = comm->nRanks * count * ncclTypeSize(datatype);
        const int64_t llThresh = rcclParamDdaLLThreshold();
        const int64_t ll128Thresh = rcclParamDdaLL128Threshold();
        // Small-chunk fast lane: LL protocol (no GPU barrier).
        if (rcclParamDdaLL() && llThresh > 0 && a2aBytes <= (size_t)llThresh &&
            ncclAllToAllDdaFabricLLEligible(comm, sendbuff, recvbuff, count, datatype)) {
          INFO(NCCL_COLL, "AllToAll: taking DDA fabric LL path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
               comm->nRanks, comm->nNodes, count, (int)datatype, a2aBytes);
          NCCLCHECK(ncclAllToAllDdaFabricLL(sendbuff, recvbuff, count, datatype, comm, stream));
          return ncclSuccess;
        }
        // Mid-chunk fast lane: LL128 protocol (128B lines, no GPU barrier).
        if (rcclParamDdaLL128() && ll128Thresh > 0 && a2aBytes <= (size_t)ll128Thresh &&
            ncclAllToAllDdaFabricLL128Eligible(comm, sendbuff, recvbuff, count, datatype)) {
          INFO(NCCL_COLL, "AllToAll: taking DDA fabric LL128 path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
               comm->nRanks, comm->nNodes, count, (int)datatype, a2aBytes);
          NCCLCHECK(ncclAllToAllDdaFabricLL128(sendbuff, recvbuff, count, datatype, comm, stream));
          return ncclSuccess;
        }
        if (ncclAllToAllDdaFabricEligible(comm, sendbuff, recvbuff, count, datatype)) {
          INFO(NCCL_COLL, "AllToAll: taking DDA fabric (VMM) path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
               comm->nRanks, comm->nNodes, count, (int)datatype, count * ncclTypeSize(datatype));
          NCCLCHECK(ncclAllToAllDdaFabric(sendbuff, recvbuff, count, datatype, comm, stream));
          return ncclSuccess;
        }
      } else if (ncclAllToAllDdaIpcEligible(comm, sendbuff, recvbuff, count, datatype)) {
        NCCLCHECK(ncclAllToAllDdaIpc(sendbuff, recvbuff, count, datatype, comm, stream));
        return ncclSuccess;
      }
    }

    info = {
      ncclFuncAlltoAll,    "AlltoAll",         sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
      ALLTOALL_CHUNKSTEPS, ALLTOALL_SLICESTEPS
    };
  }
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAlltoAllConfig, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclAlltoAllConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) {
  NVTX3_FUNC_WITH_PARAMS(AlltoAllConfig, NcclNvtxParamsAlltoAll,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype)));
  return ncclAlltoAllConfigImpl(sendbuff, recvbuff, count, datatype, comm, stream, config);
}

static inline ncclResult_t ncclAllReduceConfigImpl(const void* sendbuff, void* recvbuff, size_t count,
                                                   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                                   cudaStream_t stream, const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncAllReduce, "AllReduce",
                          sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
                          ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAlltoAllv, const void* sendbuff, const size_t sendcounts[], const size_t sdispls[],
         void* recvbuff, const size_t recvcounts[], const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm,
         hipStream_t stream);
ncclResult_t ncclAlltoAllv_impl(const void* sendbuff, const size_t sendcounts[], const size_t sdispls[], void* recvbuff,
                                const size_t recvcounts[], const size_t rdispls[], ncclDataType_t datatype,
                                ncclComm_t comm, hipStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AlltoAllv, NcclNvtxParamsAlltoAllv,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcounts[comm->rank] * ncclTypeSize(datatype),
                                       recvcounts[comm->rank] * ncclTypeSize(datatype), datatype));

  NCCLCHECK(Recorder::instance().record(rrAllToAllv, sendbuff, recvbuff, 0, datatype, comm, stream, -1, sendcounts,
                                        sdispls, recvcounts, rdispls));

  int nRanks, rank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));

  std::vector<size_t> sdispls1(nRanks);
  std::vector<size_t> rdispls1(nRanks);
  std::vector<size_t> sendcounts1(nRanks);
  std::vector<size_t> recvcounts1(nRanks);

  std::vector<size_t> sizes(4 * nRanks); // [sendSizes, sendDispls, recvSizes, recvDispls] (bytes).
  std::vector<size_t> gatheredSizes(4 * nRanks * nRanks);

  for (int i = 0; i < nRanks; i++) {
    sdispls1[i] = sdispls[i] * ncclTypeSize(datatype);
    rdispls1[i] = rdispls[i] * ncclTypeSize(datatype);
    sendcounts1[i] = sendcounts[i] * ncclTypeSize(datatype);
    recvcounts1[i] = recvcounts[i] * ncclTypeSize(datatype);
    sizes[i] = sendcounts1[i];
    sizes[nRanks + i] = sdispls1[i];
    sizes[2 * nRanks + i] = recvcounts1[i];
    sizes[3 * nRanks + i] = rdispls1[i];
  }
#ifdef ENABLE_ROCSHMEM

  size_t count = sdispls1[nRanks - 1] + sendcounts1[nRanks - 1];

  if (comm->enableRocshmem && comm->nNodes > 1 && (comm->nRanks / comm->nNodes == 8)) {
    INFO(
      NCCL_INIT,
      "GDA alltoallv is supported for up to 128MB message size; Use ROCSHMEM_HEAP_SIZE=3GB for GDA support till 512MB");
    count = count / ncclTypeSize(datatype);

    // use CU for copy-in/copy-out for small <= 128KB sizes
    // TODO: the threshold could be different for different number of nodes
    if ((count * ncclTypeSize(datatype)) > 131072) {
      void* dest = (char*)comm->sourceRshmem + comm->symId * comm->bufThreshold;
      CUDACHECK(hipMemcpyAsync(dest, sendbuff, count * ncclTypeSize(datatype), hipMemcpyDeviceToDevice, stream));
    }
    struct ncclInfo info = {ncclFuncAlltoAllvGda,
                            "AlltoAllvGda",
                            sendbuff,
                            recvbuff,
                            count,
                            datatype,
                            ncclSum,
                            0,
                            comm,
                            stream,
                            ALLTOALL_PIVOT_CHUNKSTEPS,
                            ALLTOALL_PIVOT_SLICESTEPS,
                            nullptr};
#ifdef ENABLE_ROCSHMEM
    info.sizes = sizes.data();
#endif

    ncclResult_t ret = ncclEnqueueCheck(&info);

    if (ret == ncclSuccess && ((count * ncclTypeSize(datatype)) > 131072)) {
      void* src = (char*)comm->destRshmem + comm->symId * comm->bufThreshold;
      CUDACHECK(hipMemcpyAsync(recvbuff, src, count * ncclTypeSize(datatype), hipMemcpyDeviceToDevice, stream));
      comm->symId = (comm->symId + 1) % comm->numSymBuf;
    }
    return ret;
  }
#endif
  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;
  NCCLCHECK(ncclDevrFindWindow(comm, sendbuff, &sendWin));
  NCCLCHECK(ncclDevrFindWindow(comm, recvbuff, &recvWin));
  ncclSymRegType_t winRegType;
  NCCLCHECK(ncclGetSymRegType(sendWin, recvWin, &winRegType));
  bool hasSysmemSegment = ncclDevrWindowHasSysmemSegment(sendWin) || ncclDevrWindowHasSysmemSegment(recvWin);
  struct ncclCudaGraph ceGraph;
  NCCLCHECK(ncclCudaGetCapturingGraph(&ceGraph, stream, comm->config.graphUsageMode));
  bool ceCapturing = ncclCudaGraphValid(ceGraph);

  // CE AlltoAllv is single-node only (ncclCeAlltoAllvEligible requires nNodes==1).
  // Multi-node jobs (e.g. 18x4) use the send/recv fallback below for cross-node traffic.
  if (ncclCeAlltoAllvEligible(comm, datatype, winRegType, hasSysmemSegment, ceCapturing)) {
    const size_t nLocal = 4 * (size_t)nRanks;
    const size_t nGather = nLocal * (size_t)nRanks;

    CUDACHECK(cudaMemcpyAsync(comm->localSizes, sizes.data(), nLocal * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    NCCLCHECK(ncclGroupStart());
    for (int r = 0; r < nRanks; r++) {
      void* recvPtr = (void*)((char*)comm->gatheredSizes + (size_t)r * nLocal * sizeof(size_t));
      NCCLCHECK(ncclSend(comm->localSizes, nLocal, ncclUint64, r, comm, stream));
      NCCLCHECK(ncclRecv(recvPtr, nLocal, ncclUint64, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaMemcpyAsync(gatheredSizes.data(), comm->gatheredSizes, nGather * sizeof(size_t),
                              cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    struct ncclInfo info = {
      ncclFuncAlltoAllv,   "AlltoAllv",         sendbuff, recvbuff, 0, datatype, ncclSum, 0, comm, stream,
      ALLTOALL_CHUNKSTEPS, ALLTOALL_SLICESTEPS, nullptr
    };
    info.sizes = gatheredSizes.data();

    return ncclEnqueueCheck(&info);
  } else {
    Recorder::instance().skip(true);
    NCCLCHECK(ncclGroupStart());
    for (int r = 0; r < nRanks; r++) {
      NCCLCHECK(ncclSend(((char*)sendbuff) + sdispls[r] * ncclTypeSize(datatype), sendcounts[r], datatype, r, comm,
                         stream));
      NCCLCHECK(ncclRecv(((char*)recvbuff) + rdispls[r] * ncclTypeSize(datatype), recvcounts[r], datatype, r, comm,
                         stream));
    }
    NCCLCHECK(ncclGroupEnd());
    Recorder::instance().skip(false);
    return ncclSuccess;
  }
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op, datatype));

  // RCCL update slice steps for AllReduce if single node
  const bool isGfx950 = IsArchMatch(comm->archName, "gfx950");
  int chunkSteps = (isGfx950 && comm->rcclUseOneSlice) ? 1 : ALLREDUCE_CHUNKSTEPS;
  int sliceSteps = comm->rcclUseOneSlice ? (isGfx950 ? 1 : ALLREDUCE_SLICESTEPS_SINGLE_NODE) : ALLREDUCE_SLICESTEPS;

  struct ncclInfo info = {ncclFuncAllReduce, "AllReduce", sendbuff, recvbuff, count,
                          datatype,          op,          0,        comm,     stream, /* Args */
                          chunkSteps,        sliceSteps,  nullptr};

  NCCLCHECK(Recorder::instance().record(rrAllReduce, info));

  // Select the implementation once, in one place (rccl_wrap.cc). The returned
  // decision drives dispatch here (CE 2-shot / DDA return early) and is carried
  // into taskAppend() via info so the CE-vs-kernel choice and graph-capture state
  // are never recomputed. rcclSelectAllReduce() also performs the CE graph-latch
  // tick, matching the previous inline behavior. The same function backs the
  // reporting path (rcclGetCollImplInfo) so rccl-tests attributes numbers to the
  // backend that actually ran.
  struct rcclCollDecision decision;
  NCCLCHECK(rcclSelectAllReduce(comm, sendbuff, recvbuff, count, datatype, op, stream, /*query=*/false,
                                /*graphCapturingHint=*/false, &decision));
  info.decision = decision;
  info.decisionValid = true;

  // Canonical selection line for addon backends (CE / DDA / symmetric). Native
  // kernels report via the enqueue.cc channel{Lo..Hi} tuning line instead; this
  // names the addon RCCL runs so rcclGetCollImplInfo can be checked against it.
  if (comm->rank == 0 && decision.algo >= NCCL_NUM_ALGORITHMS) {
    const char* an = nullptr;
    rcclGetAlgoName(decision.algo, &an);
    INFO(NCCL_COLL, "AllReduce impl selected: algo %s", an ? an : "?");
  }

  switch (decision.algo) {
  case RCCL_CE_2SHOT:
    if (count == 0) return ncclSuccess;
    INFO(NCCL_COLL, "CE 2-shot AllReduce: count=%zu datatype=%d op=%d rank=%d/%d", count, (int)datatype, (int)op,
         comm->rank, comm->nRanks);
    return ncclCeAllReduce(comm, sendbuff, recvbuff, count, datatype, op, stream);
  case RCCL_DDA_FABRIC_LL:
    INFO(NCCL_COLL, "AllReduce: taking DDA fabric LL path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, count * ncclTypeSize(datatype));
    NCCLCHECK(ncclAllReduceDdaFabricLL(sendbuff, recvbuff, count, datatype, op, comm, stream));
    return ncclSuccess;
  case RCCL_DDA_FABRIC_LL128:
    INFO(NCCL_COLL, "AllReduce: taking DDA fabric LL128 path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, count * ncclTypeSize(datatype));
    NCCLCHECK(ncclAllReduceDdaFabricLL128(sendbuff, recvbuff, count, datatype, op, comm, stream));
    return ncclSuccess;
  case RCCL_DDA_FABRIC_VMM:
    INFO(NCCL_COLL, "AllReduce: taking DDA fabric (VMM) path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, count * ncclTypeSize(datatype));
    NCCLCHECK(ncclAllReduceDdaFabric(sendbuff, recvbuff, count, datatype, op, comm, stream));
    return ncclSuccess;
  case RCCL_DDA_IPC:
    NCCLCHECK(ncclAllReduceDdaIpc(sendbuff, recvbuff, count, datatype, op, comm, stream));
    return ncclSuccess;
  default:
    // RCCL_SYMMETRIC / RCCL_CE_REGISTERED / native kernel algorithms all go
    // through the standard enqueue path; taskAppend() honors info->decision.
    return ncclEnqueueCheck(&info);
  }
}

ncclResult_t ncclAllReduceWithBias_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                        ncclRedOp_t op, ncclComm* comm, cudaStream_t stream, const void* acc) {
  NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op, datatype));

  if (acc == nullptr) {
    WARN("ncclAllReduceWithBias : acc cannot be nullptr");
    return ncclInvalidArgument;
  }

  // RCCL update slice steps for AllReduceBias if single node
  // similar to changes made to AllReduce earlier
  const bool isGfx950 = IsArchMatch(comm->archName, "gfx950");
  int chunkSteps = (isGfx950 && comm->rcclUseOneSlice) ? 1 : ALLREDUCE_CHUNKSTEPS;
  int sliceSteps = comm->rcclUseOneSlice ? (isGfx950 ? 1 : ALLREDUCE_SLICESTEPS_SINGLE_NODE) : ALLREDUCE_SLICESTEPS;

  struct ncclInfo info = {ncclFuncAllReduce, "AllReduce", sendbuff, recvbuff, count,
                          datatype,          op,          0,        comm,     stream, /* Args */
                          chunkSteps,        sliceSteps,  acc};

  NCCLCHECK(Recorder::instance().record(rrAllReduceWithBias, info));

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceConfig, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclAllReduceConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) {
  NVTX3_FUNC_WITH_PARAMS(AllReduceConfig, NcclNvtxParamsAllReduce,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op));
  return ncclAllReduceConfigImpl(sendbuff, recvbuff, count, datatype, op, comm, stream, config);
}

static inline ncclResult_t ncclBroadcastConfigImpl(const void* sendbuff, void* recvbuff, size_t count,
                                                   ncclDataType_t datatype, int root, ncclComm_t comm,
                                                   cudaStream_t stream, const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncBroadcast, "Broadcast",
                          sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
                          BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                                ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Broadcast, NcclNvtxParamsBroadcast,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, datatype));

  struct ncclInfo info = {ncclFuncBroadcast,
                          "Broadcast",
                          sendbuff,
                          recvbuff,
                          count,
                          datatype,
                          ncclSum,
                          root,
                          comm,
                          stream, /* Args */
                          BROADCAST_CHUNKSTEPS,
                          BROADCAST_SLICESTEPS,
                          nullptr};

  NCCLCHECK(Recorder::instance().record(rrBroadcast, info));

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclBroadcastConfig, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclBroadcastConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                                 ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) {
  NVTX3_FUNC_WITH_PARAMS(BroadcastConfig, NcclNvtxParamsBroadcast,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));
  return ncclBroadcastConfigImpl(sendbuff, recvbuff, count, datatype, root, comm, stream, config);
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
         cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                       cudaStream_t stream) {
  NCCLCHECK(Recorder::instance().record(rrBcast, buff, buff, count, datatype, comm, stream, root));
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

static inline ncclResult_t ncclGatherConfigImpl(const void* sendbuff, void* recvbuff, size_t count,
                                                ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream,
                                                const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncGather, "Gather",
                          sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
                          GATHER_CHUNKSTEPS, GATHER_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         int root, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclGather_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                             ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Gather, NcclNvtxParamsGather,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  NCCLCHECK(Recorder::instance().record(rrGather, sendbuff, recvbuff, count, datatype, comm, stream, root));

  struct ncclInfo info = {ncclFuncGather,    "Gather",         sendbuff, recvbuff, count,
                          datatype,          ncclSum,          root,     comm,     stream, /* Args */
                          GATHER_CHUNKSTEPS, GATHER_SLICESTEPS};
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclGatherConfig, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclGatherConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                              ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) {
  NVTX3_FUNC_WITH_PARAMS(GatherConfig, NcclNvtxParamsGather,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));
  return ncclGatherConfigImpl(sendbuff, recvbuff, count, datatype, root, comm, stream, config);
}

static inline ncclResult_t ncclReduceConfigImpl(const void* sendbuff, void* recvbuff, size_t count,
                                                ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm,
                                                cudaStream_t stream, const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncReduce, "Reduce",
                          sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
                          REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                             ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Reduce, NcclNvtxParamsReduce,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, op, datatype));

  struct ncclInfo info = {
    ncclFuncReduce,    "Reduce",          sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS, nullptr
  };

  NCCLCHECK(Recorder::instance().record(rrReduce, info));

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduceConfig, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclReduceConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                              ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream,
                              const ncclCollConfig_t* config) {
  NVTX3_FUNC_WITH_PARAMS(ReduceConfig, NcclNvtxParamsReduce,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, op));
  return ncclReduceConfigImpl(sendbuff, recvbuff, count, datatype, op, root, comm, stream, config);
}

static inline ncclResult_t ncclReduceScatterConfigImpl(const void* sendbuff, void* recvbuff, size_t recvcount,
                                                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                                       cudaStream_t stream, const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncReduceScatter, "ReduceScatter",
                          sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
                          REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

// Direct ReduceScatter: two-phase pipeline
//   Phase 1: stage every peer's slice into comm->tempBuff via a single
//            useDirect=true ncclEnqueueCheck. taskAppend fans this out into
//            per-peer Send/Recv P2P tasks.
//   Phase 2: dispatch the local reduce kernel via a second
//            ncclEnqueueCheck under enableDirectReduceScatter=1. The kernel
//            reads all N peer slices from tempBuff and writes the reduced
//            result to recvbuff.
static ncclResult_t rcclDirectReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                                            ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                            cudaStream_t stream, int chunkSteps, int sliceSteps) {
  if (recvcount == 0) return ncclSuccess;
  void* tempbuff = comm->tempBuff;

  ncclResult_t ret = ncclSuccess;

  struct ncclInfo infoP2P = {ncclFuncReduceScatter,
                             "ReduceScatter",
                             sendbuff,
                             tempbuff,
                             recvcount,
                             datatype,
                             op,
                             0,
                             comm,
                             stream,
                             chunkSteps,
                             sliceSteps,
                             nullptr};
  infoP2P.useDirect = true;

  struct ncclInfo info = {ncclFuncReduceScatter,
                          "ReduceScatter",
                          sendbuff,
                          recvbuff,
                          recvcount,
                          datatype,
                          op,
                          0,
                          comm,
                          stream,
                          chunkSteps,
                          sliceSteps,
                          nullptr};

  comm->enableDirectReduceScatter = 1;
  NCCLCHECKGOTO(ncclEnqueueCheck(&infoP2P), ret, cleanup);
  NCCLCHECKGOTO(ncclEnqueueCheck(&info), ret, cleanup);

cleanup:
  comm->enableDirectReduceScatter = 0;
  return ret;
}

static ncclResult_t ncclHierarchicalReduceScatter_Impl(const void* sendbuff, void* recvbuff, size_t recvcount,
                                                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                                       cudaStream_t stream) {
  if (recvcount == 0) return ncclSuccess;
  ncclComm* intraComm = comm->hierarchicalIntraComm;
  ncclComm* interComm = comm->hierarchicalInterComm;
  int localRanks = intraComm->nRanks;
  int nNodes = interComm->nRanks;
  size_t typeSize = ncclTypeSize(datatype);
  size_t rankOffset = recvcount * typeSize;

  void* tempBuf = comm->hierarchicalTempBuffer;
  size_t intraOutCount = recvcount * nNodes; // per-rank intra-RS output
  void* intraOut = (char*)tempBuf + (size_t)intraComm->rank * intraOutCount * typeSize;

  // Step 1: Pre-shuffle sendbuff (node-major) -> tempBuf (local-rank-major).
  size_t totalBytes = (size_t)nNodes * localRanks * rankOffset;
  int numBlocks = hierarchicalShuffleNumBlocks(totalBytes);
  hierarchicalShuffle<<<numBlocks, HIERARCHICAL_SHUFFLE_THREADS, 0, stream>>>((const char*)sendbuff, (char*)tempBuf,
                                                                              rankOffset, localRanks, nNodes);
  CUDACHECK(hipGetLastError());

  // Step 2: Intra-node ReduceScatter, in-place inside tempBuf.
  struct ncclInfo infoIntraRS = {ncclFuncReduceScatter,
                                 "HierarchicalReduceScatter-Intra",
                                 tempBuf,
                                 intraOut,
                                 intraOutCount,
                                 datatype,
                                 op,
                                 0,
                                 intraComm,
                                 stream,
                                 REDUCESCATTER_CHUNKSTEPS,
                                 intraComm->rcclUseOneSlice ? REDUCESCATTER_SLICESTEPS_SINGLE_NODE :
                                                              REDUCESCATTER_SLICESTEPS,
                                 nullptr};
  NCCLCHECK(ncclEnqueueCheck(&infoIntraRS));

  // Step 3: Inter-node ReduceScatter  intraOut -> recvbuff.
  size_t interMsgSize = recvcount * typeSize * (size_t)nNodes;
  if (rcclUseReduceScatterDirect(interComm, interMsgSize)) {
    NCCLCHECK(rcclDirectReduceScatter(intraOut, recvbuff, recvcount, datatype, op, interComm, stream,
                                      REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS));
  } else {
    struct ncclInfo infoInterRS = {ncclFuncReduceScatter,
                                   "HierarchicalReduceScatter-Inter",
                                   intraOut,
                                   recvbuff,
                                   recvcount,
                                   datatype,
                                   op,
                                   0,
                                   interComm,
                                   stream,
                                   REDUCESCATTER_CHUNKSTEPS,
                                   REDUCESCATTER_SLICESTEPS,
                                   nullptr};
    NCCLCHECK(ncclEnqueueCheck(&infoInterRS));
  }

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
         ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter_impl(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, NcclNvtxParamsReduceScatter,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, recvcount * ncclTypeSize(datatype), op, datatype));
  // RCCL update slice steps for ReduceScatter if single node
  const bool isGfx950 = IsArchMatch(comm->archName, "gfx950");

  int chunkSteps = (isGfx950 && comm->rcclUseOneSlice) ? 1 : REDUCESCATTER_CHUNKSTEPS;
  int sliceSteps =
    comm->rcclUseOneSlice ? (isGfx950 ? 1 : REDUCESCATTER_SLICESTEPS_SINGLE_NODE) : REDUCESCATTER_SLICESTEPS;

  struct ncclInfo info = {ncclFuncReduceScatter,
                          "ReduceScatter",
                          sendbuff,
                          recvbuff,
                          recvcount,
                          datatype,
                          op,
                          0,
                          comm,
                          stream, /* Args */
                          chunkSteps,
                          sliceSteps,
                          nullptr};

  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t msgSize = recvcount * ncclTypeSize(datatype) * nRanks;

  NCCLCHECK(Recorder::instance().record(rrReduceScatter, info));

  // Reset value forcing direct reduce scatter algorithm
  comm->enableDirectReduceScatter = 0;

  // Select once in rccl_wrap.cc; the same function backs rcclGetCollImplInfo.
  struct rcclCollDecision decision;
  NCCLCHECK(rcclSelectReduceScatter(comm, sendbuff, recvbuff, recvcount, datatype, op, /*query=*/false, &decision));

  // Canonical line for addon backends; native kernels report via the enqueue tuning line.
  if (comm->rank == 0 && decision.algo >= NCCL_NUM_ALGORITHMS) {
    const char* an = nullptr;
    rcclGetAlgoName(decision.algo, &an);
    INFO(NCCL_COLL, "ReduceScatter impl selected: algo %s", an ? an : "?");
  }

  switch (decision.algo) {
  case RCCL_DDA_FABRIC_LL:
    INFO(NCCL_COLL, "ReduceScatter: taking DDA fabric LL path: nRanks=%d nNodes=%d recvcount=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, recvcount, (int)datatype, recvcount * ncclTypeSize(datatype));
    NCCLCHECK(ncclReduceScatterDdaFabricLL(sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
    return ncclSuccess;
  case RCCL_DDA_FABRIC_LL128:
    INFO(NCCL_COLL,
         "ReduceScatter: taking DDA fabric LL128 path: nRanks=%d nNodes=%d recvcount=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, recvcount, (int)datatype, recvcount * ncclTypeSize(datatype));
    NCCLCHECK(ncclReduceScatterDdaFabricLL128(sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
    return ncclSuccess;
  case RCCL_DDA_FABRIC_VMM:
    INFO(NCCL_COLL,
         "ReduceScatter: taking DDA fabric (VMM) path: nRanks=%d nNodes=%d recvcount=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, recvcount, (int)datatype, recvcount * ncclTypeSize(datatype));
    NCCLCHECK(ncclReduceScatterDdaFabric(sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
    return ncclSuccess;
  case RCCL_DDA_IPC:
    NCCLCHECK(ncclReduceScatterDdaIpc(sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
    return ncclSuccess;
  case RCCL_HIERARCHICAL_REDUCESCATTER:
    return ncclHierarchicalReduceScatter_Impl(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
  case RCCL_DIRECT_REDUCESCATTER:
    INFO(NCCL_TUNING,
         "RCCL DIRECT REDUCE-SCATTER recvcount=%zu msgSize=%zu rank=%d nRanks=%d nNodes=%d comm=%p stream=%p "
         "sendbuff=%p recvbuff=%p",
         recvcount, msgSize, comm->rank, nRanks, comm->nNodes, comm, stream, sendbuff, recvbuff);
    return rcclDirectReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream, chunkSteps, sliceSteps);
  default: // RCCL_SYMMETRIC / native go through the standard enqueue path.
    return ncclEnqueueCheck(&info);
  }
}

NCCL_API(ncclResult_t, ncclReduceScatterConfig, const void* sendbuff, void* recvbuff, size_t recvcount,
         ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclReduceScatterConfig(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                                     ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream,
                                     const ncclCollConfig_t* config) {
  NVTX3_FUNC_WITH_PARAMS(ReduceScatterConfig, NcclNvtxParamsReduceScatter,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, recvcount * ncclTypeSize(datatype), op));
  return ncclReduceScatterConfigImpl(sendbuff, recvbuff, recvcount, datatype, op, comm, stream, config);
}

static inline ncclResult_t ncclScatterConfigImpl(const void* sendbuff, void* recvbuff, size_t count,
                                                 ncclDataType_t datatype, int root, ncclComm_t comm,
                                                 cudaStream_t stream, const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncScatter, "Scatter",
                          sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
                          SCATTER_CHUNKSTEPS, SCATTER_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         int root, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclScatter_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                              ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Scatter, NcclNvtxParamsScatter,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, datatype));

  NCCLCHECK(Recorder::instance().record(rrScatter, sendbuff, recvbuff, count, datatype, comm, stream, root));

  struct ncclInfo info = {ncclFuncScatter,    "Scatter",         sendbuff, recvbuff, count,
                          datatype,           ncclSum,           root,     comm,     stream, /* Args */
                          SCATTER_CHUNKSTEPS, SCATTER_SLICESTEPS};
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclScatterConfig, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclScatterConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                               ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) {
  NVTX3_FUNC_WITH_PARAMS(ScatterConfig, NcclNvtxParamsScatter,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));
  return ncclScatterConfigImpl(sendbuff, recvbuff, count, datatype, root, comm, stream, config);
}

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
         cudaStream_t stream);
ncclResult_t ncclSend_impl(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                           cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Send, NcclNvtxParamsSendRecv,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer, datatype));

  struct ncclInfo info = {ncclFuncSend,
                          "Send",
                          NULL,
                          (void*)sendbuff,
                          count,
                          datatype,
                          ncclSum,
                          peer,
                          comm,
                          stream, /* Args */
                          1,
                          1,
                          nullptr};

  NCCLCHECK(Recorder::instance().record(rrSend, info));

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
         cudaStream_t stream);
ncclResult_t ncclRecv_impl(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                           cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Recv, NcclNvtxParamsSendRecv,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer, datatype));

  struct ncclInfo info = {ncclFuncRecv,
                          "Recv",
                          NULL,
                          recvbuff,
                          count,
                          datatype,
                          ncclSum,
                          peer,
                          comm,
                          stream, /* Args */
                          1,
                          1,
                          nullptr};

  NCCLCHECK(Recorder::instance().record(rrRecv, info));

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclPutSignal, const void* localbuff, size_t count, ncclDataType_t datatype, int peer,
         ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm,
         cudaStream_t stream);
ncclResult_t ncclPutSignal_impl(const void* localbuff, size_t count, ncclDataType_t datatype, int peer,
                                ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags,
                                ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(PutSignal, NcclNvtxParamsPut,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer, ctx));

  struct ncclInfo info = {ncclFuncPutSignal,
                          "PutSignal",
                          localbuff,
                          NULL,
                          count,
                          datatype,
                          ncclSum,
                          peer,
                          comm,
                          stream, /* Args */
                          1,
                          1,
                          nullptr, /* chunkSteps, sliceSteps, acc */
                          nullptr,
                          false, /* useDirect */
                          peerWinOffset,
                          peerWin,
                          sigIdx,
                          ctx,
                          flags, /* peerWinOffset, peerWin, sigIdx, ctx, flags */
                          0,
                          NULL}; /* nDesc, signalDescs */
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclSignal, int peer, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm,
         cudaStream_t stream);
ncclResult_t ncclSignal_impl(int peer, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Signal, NcclNvtxParamsSignal, NVTX3_PAYLOAD(comm ? comm->commHash : 0, peer, ctx));

  struct ncclInfo info = {ncclFuncSignal,
                          "Signal",
                          NULL,
                          NULL,
                          0,
                          ncclInt8,
                          ncclSum,
                          peer,
                          comm,
                          stream, /* Args */
                          1,
                          1,
                          nullptr, /* chunkSteps, sliceSteps, acc */
                          nullptr,
                          false, /* useDirect */
                          0,
                          NULL,
                          sigIdx,
                          ctx,
                          flags, /* peerWinOffset, peerWin, sigIdx, ctx, flags */
                          0,
                          NULL}; /* nDesc, signalDescs */
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclWaitSignal, int nDesc, ncclWaitSignalDesc_t* signalDescs, ncclComm_t comm,
         cudaStream_t stream);
ncclResult_t ncclWaitSignal_impl(int nDesc, ncclWaitSignalDesc_t* signalDescs, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(WaitSignal, NcclNvtxParamsWaitSignal, NVTX3_PAYLOAD(comm ? comm->commHash : 0, nDesc, 0));

  struct ncclInfo info = {ncclFuncWaitSignal,
                          "WaitSignal",
                          NULL,
                          NULL,
                          0,
                          ncclInt32,
                          ncclSum,
                          0,
                          comm,
                          stream, /* Args */
                          1,
                          1,
                          nullptr, /* chunkSteps, sliceSteps, acc */
                          nullptr,
                          false, /* useDirect */
                          0,
                          NULL,
                          0,
                          0,
                          0, /* peerWinOffset, peerWin, sigIdx, ctx, flags */
                          nDesc,
                          signalDescs}; /* nDesc, signalDescs */
  return ncclEnqueueCheck(&info);
}
