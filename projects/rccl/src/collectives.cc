/*************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "enqueue.h"
#include "ce_coll.h"  // ncclCeAllGather, NCCL_CE_AG_BUFF_BYTES

// Set RCCL_TWO_SHOT_AG_FORCE_IPC=1 to bypass CE AllGather and always use the IPC fallback.
RCCL_PARAM(TwoShotAgForceIpc, "TWO_SHOT_AG_FORCE_IPC", 0);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include "graph/topo.h"
#include "nccl.h"
#include "api_trace.h"
#include "nvtx_payload_schemas.h"
#include "device/hierarchical_ag_shuffle.h"
#ifdef ENABLE_ROCSHMEM
#include <rocshmem/rocshmem.hpp>
#endif

using namespace rccl;

const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {
  case ncclFuncAllGather: return "AllGather";
  case ncclFuncAllReduce: return "AllReduce";
  case ncclFuncAlltoAll: return "AlltoAll";
  case ncclFuncBroadcast: return "Broadcast";
  case ncclFuncGather: return "Gather";
  case ncclFuncRecv: return "Recv";
  case ncclFuncReduce: return "Reduce";
  case ncclFuncReduceScatter: return "ReduceScatter";
  case ncclFuncScatter: return "Scatter";
  case ncclFuncSendRecv: return "SendRecv";
  case ncclFuncSend: return "Send";
  default: return "Invalid";
  }
}

const char* ncclDevRedOpToString(ncclDevRedOp_t op) {
  switch (op) {
  case ncclDevSum: return "Sum";
  case ncclDevProd: return "Prod";
  case ncclDevMinMax: return "MinMax";
  case ncclDevPreMulSum: return "PreMulSum";
  case ncclDevSumPostDiv: return "SumPostDiv";
  default: return "Unknown";
  }
}

const char* ncclDatatypeToString(ncclDataType_t type) {
  switch (type) {
  case ncclInt8: return "ncclInt8";
  case ncclInt32: return "ncclInt32";
  case ncclUint32: return "ncclUint32";
  case ncclInt64: return "ncclInt64";
  case ncclUint64: return "ncclUint64";
  case ncclFloat16: return "ncclFloat16";
  case ncclFloat32: return "ncclFloat32";
  case ncclFloat64: return "ncclFloat64";
  case ncclBfloat16: return "ncclBfloat16";
  case ncclFloat8e4m3: return "ncclFloat8e4m3";
  case ncclFloat8e5m2: return "ncclFloat8e5m2";
  default: return "Unknown";
  }
}

const char* ncclAlgoToString(int algo) {
  switch (algo) {
  case NCCL_ALGO_TREE: return "TREE";
  case NCCL_ALGO_RING: return "RING";
  case NCCL_ALGO_COLLNET_DIRECT: return "COLLNET_DIRECT";
  case NCCL_ALGO_COLLNET_CHAIN: return "COLLNET_CHAIN";
  case NCCL_ALGO_NVLS: return "NVLS";
  case NCCL_ALGO_NVLS_TREE: return "NVLS_TREE";
  case NCCL_ALGO_PAT: return "PAT";
  default: return "Unknown";
  }
}

const char* ncclProtoToString(int proto) {
  switch (proto) {
  case NCCL_PROTO_LL: return "LL";
  case NCCL_PROTO_LL128: return "LL128";
  case NCCL_PROTO_SIMPLE: return "SIMPLE";
  default: return "Unknown";
  }
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

static ncclResult_t rcclDirectAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int in_place, ncclComm_t comm, cudaStream_t stream) {
  int nRanks = comm->nRanks;
  int rank = comm->rank;
  size_t rankOffset = sendcount * ncclTypeSize(datatype);

  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < nRanks; r++) {
    int peer = (rank + r) % nRanks;
    if (peer == rank && in_place) continue;
    NCCLCHECK(ncclSend(((char*)sendbuff), sendcount, datatype, peer, comm, stream));
    NCCLCHECK(ncclRecv(((char*)recvbuff) + peer * rankOffset, sendcount, datatype, peer, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());

  return ncclSuccess;
}

RCCL_PARAM(HierarchicalAllGather, "HIERARCHICAL_ALLGATHER", 0);

static bool rcclUseHierarchicalAllGather(struct ncclComm* comm, size_t msgSize) {
  if (comm->nNodes < 8) return false;
  if (rcclParamHierarchicalAllGather() != 1) return false;
  if (!comm->hierarchicalCommsInitialized) return false;

  size_t threshold = 0;
  if (comm->nNodes >= 16) {
    threshold = HIERARCHICAL_AG_TEMP_BUFFER_SIZE;
  } else if (comm->nNodes >= 8) {
    threshold = HIERARCHICAL_AG_TEMP_BUFFER_SIZE / 2;
  }

  return threshold > 0 && msgSize <= threshold;
}

static ncclResult_t ncclHierarchicalAllGather_Impl(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  if (sendcount == 0) return ncclSuccess;
  ncclComm* intraComm = comm->hierarchicalIntraComm;
  ncclComm* interComm = comm->hierarchicalInterComm;
  int localRanks = intraComm->nRanks; // Ranks per node
  int nNodes = interComm->nRanks; // Number of nodes
  size_t typeSize = ncclTypeSize(datatype);

  void* tempBuffer = comm->hierarchicalAGTempBuffer;
  const void* interSendBuff = sendbuff;
  size_t rankOffset = sendcount * typeSize;
  if (sendbuff == ((char*)recvbuff) + comm->rank * rankOffset) {
    CUDACHECK(hipMemcpyAsync(tempBuffer, sendbuff, rankOffset, hipMemcpyDeviceToDevice, stream));
    interSendBuff = tempBuffer;
  }

  // Step 1: Inter-node AllGather
  size_t interMsgSize = sendcount * nNodes * typeSize;
  if (rcclUseAllGatherDirect(interComm, interMsgSize)) {
    // Use direct allgather
    NCCLCHECK(rcclDirectAllGather(interSendBuff, recvbuff, sendcount, datatype, 0, interComm, stream));
  } else {
    struct ncclInfo infoInterAG = { ncclFuncAllGather, "HierarchicalAllGather-Inter",
      interSendBuff, recvbuff, sendcount, datatype, ncclSum, 0, interComm, stream,
      ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS, nullptr };
    NCCLCHECK(ncclEnqueueCheck(&infoInterAG));
  }

  // Step 2: Intra-node AllGather
  size_t intraSendCount = sendcount * nNodes;
  size_t intraMsgSize = intraSendCount * typeSize * localRanks;
  if (rcclUseAllGatherDirect(intraComm, intraMsgSize)) {
    // Use direct allgather
    NCCLCHECK(rcclDirectAllGather(recvbuff, tempBuffer, intraSendCount, datatype, 0, intraComm, stream));
  } else {
    struct ncclInfo infoIntraAG = { ncclFuncAllGather, "HierarchicalAllGather-Intra",
      recvbuff, tempBuffer, intraSendCount, datatype, ncclSum, 0, intraComm, stream,
      ALLGATHER_CHUNKSTEPS,
      intraComm->rcclUseOneSlice ? ALLGATHER_SLICESTEPS_SINGLE_NODE : ALLGATHER_SLICESTEPS, nullptr
    };
    NCCLCHECK(ncclEnqueueCheck(&infoIntraAG));
  }

  // Step 3: Shuffle tempBuffer to recvbuff
  // TODO: numBlocks is set to 16 based on testing up to 16 Nodes.
  // may need to adjust for larger configurations
  int numBlocks = 16;
  int threadsPerBlock = 1024;
  hierarchicalAGShuffle<<<numBlocks, threadsPerBlock, 0, stream>>>(
    (const char*)tempBuffer, (char*)recvbuff, rankOffset, nNodes, localRanks);
  CUDACHECK(hipGetLastError());

  return ncclSuccess;
}

ncclResult_t ncclAllGather_impl(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AllGather, NcclNvtxParamsAllGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcount * ncclTypeSize(datatype), datatype));
    // RCCL update slice steps for AllGather if single node
    const bool isGfx950 = IsArchMatch(comm->archName, "gfx950");
    int chunkSteps = (isGfx950 && comm->rcclUseOneSlice)? 1 : ALLGATHER_CHUNKSTEPS;
    int sliceSteps = comm->rcclUseOneSlice
      ? (isGfx950 ? 1 : ALLGATHER_SLICESTEPS_SINGLE_NODE)
      : ALLGATHER_SLICESTEPS;
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    chunkSteps, sliceSteps, nullptr };
  int nRanks, rank;
  int in_place = 0;
  const void* srcBuf;
  void* dstBuf;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  size_t msgSize = sendcount * ncclTypeSize(datatype) * nRanks;

  NCCLCHECK(Recorder::instance().record(rrAllGather, info));

  if (rcclUseHierarchicalAllGather(comm, msgSize)) {
    return ncclHierarchicalAllGather_Impl(sendbuff, recvbuff, sendcount, datatype, comm, stream);
  }

  if (rcclUseAllGatherDirect(comm, msgSize) && ncclGroupDepth == 0) {
     INFO(NCCL_INIT, "RCCL DIRECT ALLGATHER count = %zu, msgSize = %zu, comm = %p, stream = %p, rank = %d, sendbuff = %p, recvbuff = %p",
		     sendcount, msgSize, comm, stream, rank, sendbuff, recvbuff);
     // use direct allgather (only when not in a group; in-group use Ring so ncclGroupSimulateEnd gets estimatedTime)
     if (sendcount == 0) return ncclSuccess;
     size_t rankOffset = sendcount * ncclTypeSize(datatype);
     if (sendbuff == (((char*)recvbuff) + rank * rankOffset)) {
        srcBuf = ((char*)recvbuff) + rank * rankOffset;
        dstBuf = recvbuff;
        in_place = 1;
     } else {
        srcBuf = sendbuff;
        dstBuf = recvbuff;
     }

    NCCLCHECK(rcclDirectAllGather(srcBuf, dstBuf, sendcount, datatype, in_place, comm, stream));
    return ncclSuccess;
  } else {
     // use ring allgather
     return ncclEnqueueCheck(&info);
  }
}

RCCL_PARAM(AlltoAllPivotEnable, "ALL_TO_ALL_PIVOT_ENABLE", 0);

NCCL_API(ncclResult_t, ncclAlltoAll, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAlltoAll_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AlltoAll, NcclNvtxParamsAlltoAll,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), datatype));
  
  NCCLCHECK(Recorder::instance().record(rrAllToAll, sendbuff, recvbuff, count, datatype, comm, stream));

  size_t rankOffset = count * ncclTypeSize(datatype);
  size_t rankAlign = rankOffset & ((~rankOffset) + 1);

  struct ncclInfo info;
  if (comm->topo->pivotA2AEnabled && comm->nChannels >= comm->topo->pivotA2ANumBiRings * 2 &&
      rankOffset >= 744 * 1024 && rankAlign != 4 && rcclParamAlltoAllPivotEnable()) {
      info = { ncclFuncAlltoAllPivot, "AlltoAllPivot",
        sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
        ALLTOALL_PIVOT_CHUNKSTEPS, ALLTOALL_PIVOT_SLICESTEPS, nullptr };
  } else {
      #ifdef ENABLE_ROCSHMEM
      size_t msgSize = count * ncclTypeSize(datatype) * comm->nRanks;
      if (rcclUseAlltoAllGda(comm) && msgSize <= comm->rocshmemThreshold) {	
        struct ncclInfo info = { ncclFuncAlltoAllGda, "AlltoAllGda",
              sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream,
              ALLTOALL_PIVOT_CHUNKSTEPS, ALLTOALL_PIVOT_SLICESTEPS, nullptr };
            
        return ncclEnqueueCheck(&info);
      }
      #endif // ENABLE_ROCSHMEM
    info = { ncclFuncAlltoAll, "AlltoAll",
      sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
      ALLTOALL_CHUNKSTEPS, ALLTOALL_SLICESTEPS };
  }
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAlltoAllv, const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
ncclResult_t ncclAlltoAllv_impl(const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AlltoAllv, NcclNvtxParamsAlltoAllv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcounts[comm->rank] * ncclTypeSize(datatype),
      recvcounts[comm->rank] * ncclTypeSize(datatype), datatype));

  NCCLCHECK(Recorder::instance().record(rrAllToAllv, sendbuff, recvbuff, 0, datatype, comm, stream, -1, sendcounts, sdispls, recvcounts, rdispls));

  int nRanks, rank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));

  std::vector<size_t> sdispls1(nRanks);
  std::vector<size_t> rdispls1(nRanks);
  std::vector<size_t> sendcounts1(nRanks);
  std::vector<size_t> recvcounts1(nRanks);

  std::vector<size_t> sizes(4*nRanks);	//4 for sdispl, rdispl, scount, rcount
#ifdef ENABLE_ROCSHMEM
    for (int i = 0; i < nRanks; i++) {
       sdispls1[i] = sdispls[i] * ncclTypeSize(datatype);
       rdispls1[i] = rdispls[i] * ncclTypeSize(datatype);
       sendcounts1[i] = sendcounts[i] * ncclTypeSize(datatype);
       recvcounts1[i] = recvcounts[i] * ncclTypeSize(datatype);
    }

    size_t count = sdispls1[nRanks - 1] + sendcounts1[nRanks - 1];

    if (comm->enableRocshmem && comm->nNodes > 1 && (comm->nRanks/comm->nNodes == 8)) {
        INFO(NCCL_INIT, "GDA alltoallv is supported for up to 128MB message size; Use ROCSHMEM_HEAP_SIZE=3GB for GDA support till 512MB");  

        for (int i = 0; i < nRanks; i++) {
            sizes[i] = sendcounts1[i];
            sizes[nRanks + i] = sdispls1[i];
            sizes[2*nRanks + i] = recvcounts1[i];
            sizes[3*nRanks + i] = rdispls1[i];
        }
        count = count / ncclTypeSize(datatype);

	//use CU for copy-in/copy-out for small <= 128KB sizes
	//TODO: the threshold could be different for different number of nodes
	if ((count * ncclTypeSize(datatype)) > 131072) {
	    void *dest = (char*)comm->sourceRshmem + comm->symId * comm->bufThreshold;
            CUDACHECK(hipMemcpyAsync(dest, sendbuff, count * ncclTypeSize(datatype),
               hipMemcpyDeviceToDevice, stream));
        }
        struct ncclInfo info = { ncclFuncAlltoAllvGda, "AlltoAllvGda",
        sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream,
        ALLTOALL_PIVOT_CHUNKSTEPS, ALLTOALL_PIVOT_SLICESTEPS, nullptr };
#ifdef ENABLE_ROCSHMEM
        info.sizes = sizes.data();
#endif

        ncclResult_t ret = ncclEnqueueCheck(&info);

        if (ret == ncclSuccess && ((count * ncclTypeSize(datatype)) > 131072)) {
	    void *src = (char*)comm->destRshmem + comm->symId * comm->bufThreshold;
            CUDACHECK(hipMemcpyAsync(recvbuff, src, count * ncclTypeSize(datatype),
                    hipMemcpyDeviceToDevice, stream));
            comm->symId = (comm->symId + 1) % comm->numSymBuf;
        }
        return ret;
    }
#endif

  Recorder::instance().skip(true);
  NCCLCHECK(ncclGroupStart());
  for (int r=0; r<nRanks; r++) {
    NCCLCHECK(ncclSend(
        ((char*)sendbuff) + sdispls[r]*ncclTypeSize(datatype),
        sendcounts[r],
        datatype,
        r,
        comm,
        stream));
    NCCLCHECK(ncclRecv(
        ((char*)recvbuff) + rdispls[r]*ncclTypeSize(datatype),
        recvcounts[r],
        datatype,
        r,
        comm,
        stream));
  }
  NCCLCHECK(ncclGroupEnd());
  Recorder::instance().skip(false);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op, datatype));

  // RCCL update slice steps for AllReduce if single node
  const bool isGfx950 = IsArchMatch(comm->archName, "gfx950");
  int chunkSteps = (isGfx950 && comm->rcclUseOneSlice)? 1 : ALLREDUCE_CHUNKSTEPS;
  int sliceSteps = comm->rcclUseOneSlice
      ? (isGfx950 ? 1 : ALLREDUCE_SLICESTEPS_SINGLE_NODE)
      : ALLREDUCE_SLICESTEPS;

  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    chunkSteps, sliceSteps, nullptr };

  NCCLCHECK(Recorder::instance().record(rrAllReduce, info));

  // Two-shot AllReduce (opt-in via RCCL_TWO_SHOT_ALLREDUCE=1):
  // Shot 1 — reduce-scatter: CE/SDMA-style async copies into each peer's tempBuff staging
  // (registered at ncclCommInit via IPC or same-PID direct pointers), then a device reduce
  // kernel (direct ReduceScatter path) over tempBuff. 
  // Shot 2 — CE AllGather: symmetric-memory push of rsScratch → agBuff on all peers,
  // then a local batched pull from agBuff into recvbuff.
  if (ncclGroupDepth == 0 && comm->tempBuff && rcclUseTwoShotAllReduce(comm, count, datatype, op)) {
    int nRanks, rank;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    if (count == 0) return ncclSuccess;

    cudaStreamCaptureStatus capSt = cudaStreamCaptureStatusNone;
    unsigned long long captureIdIgnored = 0;
    CUDACHECK(hipStreamGetCaptureInfo_v2(stream, &capSt, &captureIdIgnored, nullptr, nullptr, nullptr));
    if (capSt == cudaStreamCaptureStatusActive) {
      return ncclEnqueueCheck(&info);
    }

    const size_t chunkElems = count / (size_t)nRanks;
    const size_t el = ncclTypeSize(datatype);
    const size_t chunkBytes = chunkElems * el;
    void* tempbuff = comm->tempBuff;
    void* rsScratch = (char*)tempbuff + (size_t)nRanks * chunkBytes;

    // CE AllGather requires a symmetric agWin and the message must fit in agBuff.
    // If it doesn't fit, fall through to the standard ring AllReduce.
    const bool ceAgFits = (comm->agWin != nullptr) &&
                          ((size_t)nRanks * chunkBytes <= NCCL_CE_AG_BUFF_BYTES);
    if (!ceAgFits) {
      return ncclEnqueueCheck(&info);
    }

    INFO(NCCL_INIT,
         "RCCL TWO-SHOT ALLREDUCE count=%zu chunkElems=%zu rank=%d nRanks=%d comm=%p stream=%p sendbuff=%p recvbuff=%p",
         count, chunkElems, rank, nRanks, comm, stream, sendbuff, recvbuff);

    const bool hipBatchedStaging =
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        (comm->peerTempImported != nullptr && comm->peerTwoShotSyncImported != nullptr);
#else
        false;
#endif
    INFO(NCCL_COLL, "TWO-SHOT STATE: rank=%d hipBatchedStaging=%d peerTempImported=%p peerTwoShotSyncImported=%p agWin=%p",
         rank, (int)hipBatchedStaging, comm->peerTempImported, comm->peerTwoShotSyncImported, comm->agWin);
    if (!hipBatchedStaging) {
      CUDACHECK(cudaMemcpyAsync((char*)tempbuff + rank * chunkBytes, (char*)sendbuff + rank * chunkBytes,
                                    chunkBytes, cudaMemcpyDefault, stream));
    }

    if (comm->peerTempImported) {
      if (comm->peerTwoShotSyncImported) {
        // Batched SDMA copies + stream-ordered device fence (no host barrier).
        const uint32_t seq = ++comm->twoShotBarrierSeq;
        void** batchDst = nullptr;
        void** batchSrc = nullptr;
        size_t* batchSz = nullptr;
        hipStreamBatchMemOpParams* memOps = nullptr;
        ncclResult_t ts = ncclSuccess;
        unsigned int nPeer = 0;
        unsigned int nMemOps = 0;
        unsigned int oi = 0;

        NCCLCHECKGOTO(ncclCalloc(&batchDst, nRanks), ts, twoshot_cleanup);
        NCCLCHECKGOTO(ncclCalloc(&batchSrc, nRanks), ts, twoshot_cleanup);
        NCCLCHECKGOTO(ncclCalloc(&batchSz, nRanks), ts, twoshot_cleanup);

        for (int d = 0; d < nRanks; d++) {
          if (d == rank) {
            batchDst[d] = (char*)tempbuff + rank * chunkBytes;
            batchSrc[d] = (char*)sendbuff + rank * chunkBytes;
          } else {
            batchDst[d] = (char*)comm->peerTempImported[d] + rank * chunkBytes;
            batchSrc[d] = (char*)sendbuff + d * chunkBytes;
          }
          batchSz[d] = chunkBytes;
        }
        CUDACHECKGOTO(hipMemcpyBatchAsync(batchDst, batchSrc, batchSz, (size_t)nRanks, nullptr, nullptr, 0,
                                          nullptr, stream), ts, twoshot_cleanup);

        nPeer = (unsigned int)(nRanks - 1);
        nMemOps = 2 * nPeer;
        NCCLCHECKGOTO(ncclCalloc(&memOps, nMemOps), ts, twoshot_cleanup);
        for (int dest = 0; dest < nRanks; dest++) {
          if (dest == rank) continue;
          memOps[oi] = {};
          memOps[oi].writeValue.operation = hipStreamMemOpWriteValue32;
          memOps[oi].writeValue.address =
              (hipDeviceptr_t)((uint32_t*)comm->peerTwoShotSyncImported[dest] + rank);
          memOps[oi].writeValue.value = seq;
          memOps[oi].writeValue.flags = 0;
          oi++;
        }
        for (int peer = 0; peer < nRanks; peer++) {
          if (peer == rank) continue;
          memOps[oi] = {};
          memOps[oi].waitValue.operation = hipStreamMemOpWaitValue32;
          memOps[oi].waitValue.address = (hipDeviceptr_t)((uint32_t*)comm->twoShotSyncBuff + peer);
          memOps[oi].waitValue.value = seq;
          memOps[oi].waitValue.flags = hipStreamWaitValueEq;
          oi++;
        }
        CUDACHECKGOTO(hipStreamBatchMemOp(stream, oi, memOps, 0), ts, twoshot_cleanup);

twoshot_cleanup:
        free(memOps);
        free(batchDst);
        free(batchSrc);
        free(batchSz);
        NCCLCHECK(ts);
      } 
    comm->enableDirectReduceScatter = 1;
    struct ncclInfo rsInfo = {ncclFuncReduceScatter, "ReduceScatter",
                              sendbuff, rsScratch, chunkElems, datatype, op, 0, comm, stream,
                              REDUCESCATTER_CHUNKSTEPS,
                              comm->rcclUseOneSlice ? REDUCESCATTER_SLICESTEPS_SINGLE_NODE : REDUCESCATTER_SLICESTEPS,
                              nullptr};
    NCCLCHECK(ncclEnqueueCheck(&rsInfo));
    comm->enableDirectReduceScatter = 0;

      // CE AllGather: push rsScratch to all peers' agBuff[rank*chunk] via symmetric window,
      // then pull agBuff[p*chunk] from all peers' agBuff[p*chunk] into recvbuff[p*chunk] locally.
      // ncclCeAllGather handles the pre- and post-barriers.
      WARN("TWO-SHOT AG CE: rank=%d msgBytes=%zu chunkBytes=%zu agWin=%p",
           rank, count * el, chunkBytes, comm->agWin);
      struct ncclCeCollArgs agArgs = {};
      agArgs.func     = ncclFuncAllGather;
      agArgs.nElts    = chunkElems;
      agArgs.eltSize  = el;
      agArgs.sendBuff = (uint8_t*)rsScratch;
      agArgs.recvBuff = (uint8_t*)comm->agBuff;
      agArgs.sendWin  = nullptr;
      agArgs.recvWin  = comm->agWin;
      NCCLCHECK(ncclCeAllGather(comm, &agArgs, stream));

      // Pull all nRanks shards from local agBuff → recvbuff.
      void** agPullDst = nullptr;
      void** agPullSrc = nullptr;
      size_t* agPullSz = nullptr;
      ncclResult_t agTs = ncclSuccess;
      NCCLCHECKGOTO(ncclCalloc(&agPullDst, nRanks), agTs, ag_cleanup_ce);
      NCCLCHECKGOTO(ncclCalloc(&agPullSrc, nRanks), agTs, ag_cleanup_ce);
      NCCLCHECKGOTO(ncclCalloc(&agPullSz,  nRanks), agTs, ag_cleanup_ce);
      for (int p = 0; p < nRanks; p++) {
        agPullDst[p] = (char*)recvbuff     + p * chunkBytes;
        agPullSrc[p] = (char*)comm->agBuff + p * chunkBytes;
        agPullSz[p]  = chunkBytes;
      }
      CUDACHECKGOTO(hipMemcpyBatchAsync(agPullDst, agPullSrc, agPullSz, (size_t)nRanks,
                                        nullptr, nullptr, 0, nullptr, stream),
                    agTs, ag_cleanup_ce);
ag_cleanup_ce:
      free(agPullDst); free(agPullSrc); free(agPullSz);
      NCCLCHECK(agTs);
      return ncclSuccess;
    }

    return ncclEnqueueCheck(&info);
  }

  return ncclEnqueueCheck(&info);
}

ncclResult_t ncclAllReduceWithBias_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream, const void* acc) {
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
  int sliceSteps = comm->rcclUseOneSlice
      ? (isGfx950 ? 1 : ALLREDUCE_SLICESTEPS_SINGLE_NODE)
      : ALLREDUCE_SLICESTEPS;

  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    chunkSteps, sliceSteps, acc };

  NCCLCHECK(Recorder::instance().record(rrAllReduceWithBias, info));

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Broadcast, NcclNvtxParamsBroadcast,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, datatype));

  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS, nullptr };

  NCCLCHECK(Recorder::instance().record(rrBroadcast, info));

  return ncclEnqueueCheck(&info);
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(Recorder::instance().record(rrBcast, buff, buff, count, datatype, comm, stream, root));
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclGather_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Gather, NcclNvtxParamsGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  NCCLCHECK(Recorder::instance().record(rrGather, sendbuff, recvbuff, count, datatype, comm, stream, root));

  struct ncclInfo info = { ncclFuncGather, "Gather",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    GATHER_CHUNKSTEPS, GATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Reduce, NcclNvtxParamsReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, op, datatype));

  struct ncclInfo info = { ncclFuncReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS, nullptr };

  NCCLCHECK(Recorder::instance().record(rrReduce, info));

  return ncclEnqueueCheck(&info);
}


NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter_impl(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, NcclNvtxParamsReduceScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, recvcount * ncclTypeSize(datatype), op, datatype));
    // RCCL update slice steps for ReduceScatter if single node
    const bool isGfx950 = IsArchMatch(comm->archName, "gfx950");
    int chunkSteps = (isGfx950 && comm->rcclUseOneSlice)? 1 : REDUCESCATTER_CHUNKSTEPS;
    int sliceSteps = comm->rcclUseOneSlice
      ? (isGfx950 ? 1 : REDUCESCATTER_SLICESTEPS_SINGLE_NODE)
      : REDUCESCATTER_SLICESTEPS;

  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    chunkSteps, sliceSteps, nullptr };

  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t msgSize = recvcount * ncclTypeSize(datatype) * nRanks;

  NCCLCHECK(Recorder::instance().record(rrReduceScatter, info));

  // Reset value forcing direct reduce scatter algorithm 
  comm->enableDirectReduceScatter = 0;

  if (rcclUseReduceScatterDirect(comm, msgSize)) {
    INFO(NCCL_INIT, "RCCL DIRECT REDUCE-SCATTER recvcount=%zu msgSize=%zu rank=%d nRanks=%d nNodes=%d comm=%p stream=%p sendbuff=%p recvbuff=%p",
      recvcount, msgSize, comm->rank, nRanks, comm->nNodes, comm, stream, sendbuff, recvbuff);

    // Temporary Buffer to store data from each rank
    void* tempbuff = comm->tempBuff;

    // Use Direct Reduce Scatter Algorithm
    comm->enableDirectReduceScatter = 1;
    
    if (recvcount == 0) return ncclSuccess;
    
    // Calculate offset into buffers
    size_t offset = recvcount * ncclTypeSize(datatype);
    
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nRanks; i++) {
      int peer = (comm->rank + i) % nRanks;
      NCCLCHECK(ncclSend((void*)((char*)sendbuff + peer * offset), recvcount, datatype, peer, comm, stream));
      NCCLCHECK(ncclRecv((void*)((char*)tempbuff + peer * offset), recvcount, datatype, peer, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
  }
  
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclScatter_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Scatter, NcclNvtxParamsScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, datatype));

  NCCLCHECK(Recorder::instance().record(rrScatter, sendbuff, recvbuff, count, datatype, comm, stream, root));

  struct ncclInfo info = { ncclFuncScatter, "Scatter",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    SCATTER_CHUNKSTEPS, SCATTER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend_impl(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Send, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer, datatype));

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1, nullptr };

  NCCLCHECK(Recorder::instance().record(rrSend, info));

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv_impl(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Recv, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer, datatype));

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1, nullptr };

  NCCLCHECK(Recorder::instance().record(rrRecv, info));

  return ncclEnqueueCheck(&info);
}
