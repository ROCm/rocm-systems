/*************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "enqueue.h"
#include "graph/topo.h"
#include "nccl.h"
#include "api_trace.h"
#include "nvtx_payload_schemas.h"
#include "msccl/msccl_lifecycle.h"

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

  if (!mscclIsCaller())
  {
    NCCLCHECK(Recorder::instance().record(rrAllGather, info));
  }

  if (mscclAvailable(comm) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      sendcount, datatype, 0, 0, ncclSum, mscclFuncAllGather, comm, stream);
  }

  if (rcclUseAllGatherDirect(comm, msgSize)) {
     INFO(NCCL_INIT, "RCCL DIRECT ALLGATHER count = %zu, msgSize = %zu, comm = %p, stream = %p, rank = %d, sendbuff = %p, recvbuff = %p",
		     sendcount, msgSize, comm, stream, rank, sendbuff, recvbuff);
     // use direct allgather
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

     NCCLCHECK(ncclGroupStart());

     for (int r = 0; r < nRanks; r++) {
         if (r == rank && in_place)
             continue;
         
         NCCLCHECK(ncclSend(((char*)srcBuf), sendcount, datatype, r, comm, stream));
         NCCLCHECK(ncclRecv(((char*)dstBuf) + r * rankOffset, sendcount, datatype, r, comm, stream));
     }
     NCCLCHECK(ncclGroupEnd());
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
  
  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrAllToAll, sendbuff, recvbuff, count, datatype, comm, stream));
  }

  if (mscclAvailable(comm) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, 0, 0, ncclSum, mscclFuncAllToAll, comm, stream);
  }

  size_t rankOffset = count * ncclTypeSize(datatype);
  size_t rankAlign = rankOffset & ((~rankOffset) + 1);
  size_t msgSize = count * ncclTypeSize(datatype) * comm->nRanks;

  struct ncclInfo info;
  if (comm->topo->pivotA2AEnabled && comm->nChannels >= comm->topo->pivotA2ANumBiRings * 2 &&
      rankOffset >= 744 * 1024 && rankAlign != 4 && rcclParamAlltoAllPivotEnable()) {
      info = { ncclFuncAlltoAllPivot, "AlltoAllPivot",
        sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
        ALLTOALL_PIVOT_CHUNKSTEPS, ALLTOALL_PIVOT_SLICESTEPS, nullptr };
  } else {
      #ifdef ENABLE_ROCSHMEM
      if (rcclUseAllToAllGda(comm) && msgSize <= comm->rocshmemThreshold) {	
        struct ncclInfo info = { ncclFuncAllToAllGda, "AllToAllGda",
              sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream,
              ALLTOALL_PIVOT_CHUNKSTEPS, ALLTOALL_PIVOT_SLICESTEPS, nullptr };
            
        return ncclEnqueueCheck(&info);
      }
      #endif ENABLE_ROCSHMEM
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

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrAllToAllv, sendbuff, recvbuff, 0, datatype, comm, stream, -1, sendcounts, sdispls, recvcounts, rdispls));
  }

  if (mscclAvailable(comm) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls,
      0, datatype, 0, 0, ncclSum, mscclFuncAllToAllv, comm, stream);
  }

  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  if (!mscclIsCaller()) Recorder::instance().skip(true);
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
  if (!mscclIsCaller()) Recorder::instance().skip(false);
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

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrAllReduce, info));
  }

  if (mscclAvailable(comm) && !mscclIsCaller()) {
    //MSCCL not supported for FP8 datatype
    if (datatype != ncclFloat8e4m3 && datatype != ncclFloat8e5m2) {
      // MSCCL threshold for Bfloat16 = 8MB
      if (datatype != ncclBfloat16 || (count * ncclTypeSize(datatype) <= 8388608)) {
        return mscclEnqueueCheck(
                      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
                      count, datatype, 0, 0, op, mscclFuncAllReduce, comm, stream);
      }
    }
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

  // RCCL update slice steps for AllReduce if single node
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, comm -> rcclUseOneSlice ? ALLREDUCE_SLICESTEPS_SINGLE_NODE : ALLREDUCE_SLICESTEPS, acc };

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

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrBroadcast, info));
  }

  if (mscclAvailable(comm) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, root, 0, ncclSum, mscclFuncBroadcast, comm, stream);
  }

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

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrGather, sendbuff, recvbuff, count, datatype, comm, stream, root));
  }

  if (mscclAvailable(comm) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, root, 0, ncclSum, mscclFuncGather, comm, stream);
  }

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

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrReduce, info));
  }

  if (mscclAvailable(comm) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, root, 0, op, mscclFuncReduce, comm, stream);
  }

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

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrReduceScatter, info));
  }

  if (mscclAvailable(comm) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      recvcount, datatype, 0, 0, op, mscclFuncReduceScatter, comm, stream);
  }
  
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
    
    // Copy Current ranks data to tempbuff
    // Enqueue the copy on the user stream so it is correctly ordered w.r.t. the subsequent
    // ncclSend/ncclRecv and the rest of the ReduceScatter work on the same stream.
    NCCLCHECK(ncclCudaMemcpyAsync((char*)tempbuff + comm->rank * offset, (char*)sendbuff + comm->rank * offset, offset, stream));

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nRanks; i++) {
      int peer = (comm->rank + i) % nRanks;
      if (peer == comm->rank) {
        continue;
      }
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

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrScatter, sendbuff, recvbuff, count, datatype, comm, stream, root));
  }

  if (mscclAvailable(comm) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, root, 0, ncclSum, mscclFuncScatter, comm, stream);
  }

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

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrSend, info));
  }

  if (mscclAvailable(comm) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, nullptr, nullptr, nullptr,
      count, datatype, 0, peer, ncclSum, mscclFuncSend, comm, stream);
  }

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

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrRecv, info));
  }

  if (mscclAvailable(comm) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      nullptr, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, 0, peer, ncclSum, mscclFuncRecv, comm, stream);
  }

  return ncclEnqueueCheck(&info);
}
