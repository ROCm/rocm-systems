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

using namespace rccl;

const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {
  case ncclFuncAllGather: return "AllGather";
  case ncclFuncAllReduce: return "AllReduce";
  case ncclFuncBroadcast: return "Broadcast";
  case ncclFuncRecv: return "Recv";
  case ncclFuncReduce: return "Reduce";
  case ncclFuncReduceScatter: return "ReduceScatter";
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

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, comm -> rcclUseOneSlice ? ALLGATHER_SLICESTEPS_SINGLE_NODE : ALLGATHER_SLICESTEPS, nullptr };

  int nRanks;
  int in_place = 0;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
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
     // use direct allgather
     if (sendcount == 0) return ncclSuccess;
     size_t rankOffset = sendcount * ncclTypeSize(datatype);
     if (((char*)sendbuff) == (((char*)recvbuff) + comm->rank * rankOffset)) {
        in_place = 1;
     } 

     NCCLCHECK(ncclGroupStart());
     for (int r = 0; r < nRanks; r++) {
         int peer = (comm->rank + r) % nRanks;
         if (in_place && (peer == comm->rank)) {
            continue;
         }
         NCCLCHECK(ncclSend(sendbuff, sendcount, datatype, peer, comm, stream));
         NCCLCHECK(ncclRecv(((char*)recvbuff) + peer * rankOffset, sendcount, datatype, peer, comm, stream));
     }
     NCCLCHECK(ncclGroupEnd());
     return ncclSuccess;
  } else {
     // use ring allgather
     return ncclEnqueueCheck(&info);
  }
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);


ncclResult_t ncclAllReduce_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op, datatype));

  // RCCL update slice steps for AllReduce if single node
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, comm -> rcclUseOneSlice ? ALLREDUCE_SLICESTEPS_SINGLE_NODE : ALLREDUCE_SLICESTEPS, nullptr };

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

RCCL_PARAM(AllToAllPivotEnable, "ALL_TO_ALL_PIVOT_ENABLE", 0);

NCCL_API(ncclResult_t, ncclAllToAll, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, hipStream_t stream);


ncclResult_t ncclAllToAll_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, hipStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AllToAll, NcclNvtxParamsAllToAll,
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
  // Determine Pivot A2A support now that we know number of channels
  if (comm->topo->pivotA2AEnabled && comm->nChannels >= comm->topo->pivotA2ANumBiRings * 2 &&
      rankOffset >= 744 * 1024 && rankAlign != 4 && rcclParamAllToAllPivotEnable()) {
    struct ncclInfo info = { ncclFuncAllToAllPivot, "AllToAllPivot",
      sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
      ALLTOALL_PIVOT_CHUNKSTEPS, ALLTOALL_PIVOT_SLICESTEPS, nullptr };
    return ncclEnqueueCheck(&info);
  } else {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    if (count == 0) return ncclSuccess;
    if (!mscclIsCaller()) Recorder::instance().skip(true);
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<nRanks; r++) {
      NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, count, datatype, r, comm, stream));
      NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, count, datatype, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    if (!mscclIsCaller()) Recorder::instance().skip(false);
    return ncclSuccess;
  }
}

NCCL_API(ncclResult_t, ncclAllToAllv, const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);


ncclResult_t ncclAllToAllv_impl(const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AllToAllv, NcclNvtxParamsAllToAllv,
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

NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);

ncclResult_t ncclGather_impl(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Gather, NcclNvtxParamsGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcount * ncclTypeSize(datatype), root, datatype));

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrGather, sendbuff, recvbuff, sendcount, datatype, comm, stream, root));
  }

  if (mscclAvailable(comm) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      sendcount, datatype, root, 0, ncclSum, mscclFuncGather, comm, stream);
  }

  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t rankOffset = sendcount * ncclTypeSize(datatype);
  if (sendcount == 0) return ncclSuccess;
  int rank;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  if (!mscclIsCaller()) Recorder::instance().skip(true);
  NCCLCHECK(ncclGroupStart());
  if (rank == root) {
    for (int r=0; r<nRanks; r++)
      NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, sendcount, datatype, r, comm, stream));
  }
  NCCLCHECK(ncclSend(sendbuff, sendcount, datatype, root, comm, stream));
  NCCLCHECK(ncclGroupEnd());
  if (!mscclIsCaller()) Recorder::instance().skip(false);
  return ncclSuccess;
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

  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    REDUCESCATTER_CHUNKSTEPS, comm -> rcclUseOneSlice ? REDUCESCATTER_SLICESTEPS_SINGLE_NODE : REDUCESCATTER_SLICESTEPS, nullptr };

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrReduceScatter, info));
  }

  if (mscclAvailable(comm) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      recvcount, datatype, 0, 0, op, mscclFuncReduceScatter, comm, stream);
  }

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream);


ncclResult_t ncclScatter_impl(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Scatter, NcclNvtxParamsScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, recvcount * ncclTypeSize(datatype), root, datatype));

  if (!mscclIsCaller()) // when msccl falls back to
  {
    NCCLCHECK(Recorder::instance().record(rrScatter, sendbuff, recvbuff, recvcount, datatype, comm, stream, root));
  }

  if (mscclAvailable(comm) && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      recvcount, datatype, root, 0, ncclSum, mscclFuncScatter, comm, stream);
  }

  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t rankOffset = recvcount * ncclTypeSize(datatype);
  if (recvcount == 0) return ncclSuccess;
  int rank;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  if (!mscclIsCaller()) Recorder::instance().skip(true);
  NCCLCHECK(ncclGroupStart());
  if (rank == root) {
    for (int r=0; r<nRanks; r++)
      NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, recvcount, datatype, r, comm, stream));
  }
  NCCLCHECK(ncclRecv(recvbuff, recvcount, datatype, root, comm, stream));
  NCCLCHECK(ncclGroupEnd());
  if (!mscclIsCaller()) Recorder::instance().skip(false);
  return ncclSuccess;
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
