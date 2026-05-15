/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include "enqueue.h"
#include "msccl/msccl_parser.h"
#include "msccl/msccl_setup.h"
#include "msccl/msccl_status.h"
#include "api_trace.h"
#include "nvtx_payload_schemas.h"
#include <cstdio>
#include <cstdlib>

using namespace rccl;

NCCL_API(ncclResult_t, mscclLoadAlgo, const char *mscclAlgoFilePath, mscclAlgoHandle_t *mscclAlgoHandle, int rank);
ncclResult_t mscclLoadAlgo_impl(const char *mscclAlgoFilePath, mscclAlgoHandle_t *mscclAlgoHandle, int rank) {
  // deprecated
  Recorder::instance().record("mscclLoadAlgo");
  WARN("mscclLoadAlgo is deprecated. Function call has no effect.");
  return ncclSuccess;
}

NCCL_API(ncclResult_t, mscclRunAlgo,
    const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
    void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
    size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
    mscclAlgoHandle_t mscclAlgoHandle, ncclComm_t comm, hipStream_t stream);
ncclResult_t mscclRunAlgo_impl(
    const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
    void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
    size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
    mscclAlgoHandle_t mscclAlgoHandle, ncclComm_t comm, hipStream_t stream) {
  // deprecated
  Recorder::instance().record("mscclRunAlgo");
  NVTX3_FUNC_WITH_PARAMS(MSCCL, NcclNvtxParamsMSCCL,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(dataType), op, dataType));
  WARN("mscclRunAlgo is deprecated. Function call has no effect.");
  return ncclSuccess;
}

NCCL_API(ncclResult_t, mscclUnloadAlgo, mscclAlgoHandle_t mscclAlgoHandle);
ncclResult_t mscclUnloadAlgo_impl(mscclAlgoHandle_t mscclAlgoHandle) {
  // deprecated
  Recorder::instance().record("mscclUnloadAlgo");
  WARN("mscclUnloadAlgo is deprecated. Function call has no effect.");
  return ncclSuccess;
}
