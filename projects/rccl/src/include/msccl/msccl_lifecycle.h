/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef MSCCL_LIFECYCLE_H_
#define MSCCL_LIFECYCLE_H_

#include "enqueue.h"

#include "msccl/msccl_struct.h"

bool mscclEnabled();
bool mscclForceEnabled();

void mscclSetIsCallerFlag();
void mscclClearIsCallerFlag();
bool mscclIsCaller();

/**
 * @brief mscclAvailable() is used to determine if msccl functionality is avaliable
 * @param comm is an optional rccl communicator, if provided uses the mscclStatus
 * from a global map<comm -> mscclStatus> to determine if msccl is available. If not available
 * in the map, this invocations inserts a new key value pair in the global map.
 * If comm == nullptr, on the first invocation it initializes a static thread local variable 
 * mscclStatus and uses the same object in subsequent calls from same thread if comm is null ptr
 */
bool mscclAvailable(const ncclComm_t comm = nullptr);

ncclResult_t mscclSchedulerInit(ncclComm_t comm, int* numChannelsRequired);

ncclResult_t mscclInit(ncclComm_t comm);

ncclResult_t mscclGroupStart();

ncclResult_t mscclEnqueueCheck(
    const void* sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void* recvbuff, const size_t recvcounts[], const size_t rdispls[],
    size_t count, ncclDataType_t datatype, int root, int peer, ncclRedOp_t op,
    mscclFunc_t mscclFunc, ncclComm_t comm, hipStream_t stream);

ncclResult_t mscclGroupEnd();

ncclResult_t mscclTeardown(const ncclComm_t comm);

size_t mscclKernMaxLocalSize();

#endif
