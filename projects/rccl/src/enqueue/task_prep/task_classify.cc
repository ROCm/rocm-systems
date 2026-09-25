/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "collective_execution_policy.h"
#include "enqueue.h"
#include "enqueue/task_classify.h"
#include "register.h"
#include "register_inline.h"

static ncclResult_t classifyFillSendRecvTuningInput(struct ncclComm* comm, struct ncclRawTaskSendRecv* raw,
                                                    ncclTuningInput_t* in) {
  struct ncclReg* regBuf;
  bool isValid;

  in->comm = comm;
  in->tuningMask = NCCL_TUNING_MASK_ALL;
  in->func = raw->func;
  in->datatype = raw->datatype;
  in->count = raw->count;
  in->countMax = raw->count;
  in->nWorks = 1;
  in->numPipeOps = 1;
  in->nBytes = raw->bytes;

  NCCLCHECK(ncclRegFind(comm, raw->buff, raw->bytes, &regBuf));
  NCCLCHECK(ncclRegLocalIsValid(regBuf, &isValid));
  in->regBuff = (regBuf && isValid) || (ncclCudaGraphValid(comm->planner.capturingGraph) && ncclParamGraphRegister());
  return ncclSuccess;
}

static ncclResult_t classifyEnqueueP2pTask(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* p2pTaskQueue,
  ncclFunc_t func, ncclFunc_t collAPI, void* buff, size_t count, ncclDataType_t datatype, int peer,
  cudaStream_t stream, bool inPlace) {
  struct ncclRawTask* raw = ncclMemoryPoolAlloc<struct ncclRawTask>(&comm->memPool_ncclRawTask, &comm->memPermanent);
  raw->kind = ncclTaskKindSendRecv;
  raw->sendRecv.func = func;
  raw->sendRecv.collAPI = collAPI;
  raw->sendRecv.buff = buff;
  raw->sendRecv.count = count;
  raw->sendRecv.datatype = datatype;
  raw->sendRecv.peer = peer;
  raw->sendRecv.bytes = count * ncclTypeSize(datatype);
  raw->sendRecv.stream = stream;
  raw->sendRecv.inPlace = inPlace;

  struct ncclTaskTuningInfo* pt = ncclMemoryStackAlloc<struct ncclTaskTuningInfo>(&comm->memScoped);
  pt->raw = raw;
  pt->tuningOut = NCCL_TUNING_RESULT_INIT;
  NCCLCHECK(classifyFillSendRecvTuningInput(comm, &raw->sendRecv, &pt->tuningIn));
  ncclIntruQueueEnqueue(p2pTaskQueue, pt);
  return ncclSuccess;
}

static size_t classifySaturatingMultiply(size_t lhs, size_t rhs) {
  return rhs != 0 && lhs > SIZE_MAX / rhs ? SIZE_MAX : lhs * rhs;
}

static size_t classifyAllToAllvBufferSpan(const size_t* counts, const size_t* displacements, int nRanks) {
  size_t span = 0;
  for (int rank = 0; rank < nRanks; rank++) {
    size_t end = counts[rank] > SIZE_MAX - displacements[rank] ? SIZE_MAX : displacements[rank] + counts[rank];
    span = std::max(span, end);
  }
  return span;
}

static ncclResult_t classifyCollToP2pTasks(
  struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo,
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* p2pTaskQueue) {
  struct ncclRawTaskColl* coll = &tInfo->raw->coll;
  ncclFunc_t collAPI = coll->func;
  size_t elemSize = ncclTypeSize(coll->datatype);
  cudaStream_t stream = coll->stream;
  size_t sendCount = coll->count;
  size_t recvCount = coll->count;
  const size_t* allToAllvSizes = nullptr;
  if (coll->func == ncclFuncAlltoAll) {
    sendCount = recvCount = classifySaturatingMultiply(coll->count, comm->nRanks);
  } else if (coll->func == ncclFuncAlltoAllv) {
    size_t localSizeCount = 4 * static_cast<size_t>(comm->nRanks);
    if (coll->sizes == nullptr) return ncclInvalidArgument;
    if (coll->sizesCount == localSizeCount) {
      allToAllvSizes = coll->sizes;
    } else if (coll->sizesCount == classifySaturatingMultiply(localSizeCount, comm->nRanks)) {
      allToAllvSizes = coll->sizes + static_cast<size_t>(comm->rank) * localSizeCount;
    } else {
      return ncclInvalidArgument;
    }
  } else if (coll->func == ncclFuncGather) {
    recvCount = classifySaturatingMultiply(coll->count, comm->nRanks);
  } else if (coll->func == ncclFuncScatter) {
    sendCount = classifySaturatingMultiply(coll->count, comm->nRanks);
  }
  size_t sendBytes = classifySaturatingMultiply(sendCount, elemSize);
  size_t recvBytes = classifySaturatingMultiply(recvCount, elemSize);
  if (allToAllvSizes != nullptr) {
    const size_t* sendSizes = allToAllvSizes;
    const size_t* sendDisplacements = sendSizes + comm->nRanks;
    const size_t* recvSizes = sendDisplacements + comm->nRanks;
    const size_t* recvDisplacements = recvSizes + comm->nRanks;
    sendBytes = classifyAllToAllvBufferSpan(sendSizes, sendDisplacements, comm->nRanks);
    recvBytes = classifyAllToAllvBufferSpan(recvSizes, recvDisplacements, comm->nRanks);
  }
  bool inPlace = rcclBuffersOverlap(coll->sendbuff, sendBytes, coll->recvbuff, recvBytes);

  if (coll->func == ncclFuncAlltoAll) {
    for (int r = 0; r < comm->nRanks; r++) {
      void* sendBuff = (void*)((char*)coll->sendbuff + r * coll->count * elemSize);
      void* recvBuff = (void*)((char*)coll->recvbuff + r * coll->count * elemSize);
      NCCLCHECK(classifyEnqueueP2pTask(comm, p2pTaskQueue, ncclFuncSend, collAPI, sendBuff, coll->count, coll->datatype,
                                       r, stream, inPlace));
      NCCLCHECK(classifyEnqueueP2pTask(comm, p2pTaskQueue, ncclFuncRecv, collAPI, recvBuff, coll->count, coll->datatype,
                                       r, stream, inPlace));
    }
  } else if (coll->func == ncclFuncAlltoAllv) {
    const size_t* sendSizes = allToAllvSizes;
    const size_t* sendDisplacements = sendSizes + comm->nRanks;
    const size_t* recvSizes = sendDisplacements + comm->nRanks;
    const size_t* recvDisplacements = recvSizes + comm->nRanks;
    for (int r = 0; r < comm->nRanks; r++) {
      void* sendBuff = (void*)((char*)coll->sendbuff + sendDisplacements[r]);
      void* recvBuff = (void*)((char*)coll->recvbuff + recvDisplacements[r]);
      NCCLCHECK(classifyEnqueueP2pTask(comm, p2pTaskQueue, ncclFuncSend, collAPI, sendBuff, sendSizes[r], ncclInt8,
                                       r, stream, inPlace));
      NCCLCHECK(classifyEnqueueP2pTask(comm, p2pTaskQueue, ncclFuncRecv, collAPI, recvBuff, recvSizes[r], ncclInt8,
                                       r, stream, inPlace));
    }
  } else if (coll->func == ncclFuncGather) {
    NCCLCHECK(classifyEnqueueP2pTask(comm, p2pTaskQueue, ncclFuncSend, collAPI, (void*)coll->sendbuff, coll->count,
                                     coll->datatype, coll->root, stream, inPlace));
    if (comm->rank == coll->root) {
      size_t offset = 0;
      for (int r = 0; r < comm->nRanks; r++) {
        void* buff = (void*)((char*)coll->recvbuff + offset);
        NCCLCHECK(classifyEnqueueP2pTask(comm, p2pTaskQueue, ncclFuncRecv, collAPI, buff, coll->count, coll->datatype,
                                         r, stream, inPlace));
        offset += coll->count * elemSize;
      }
    }
  } else if (coll->func == ncclFuncScatter) {
    if (comm->rank == coll->root) {
      size_t offset = 0;
      for (int r = 0; r < comm->nRanks; r++) {
        void* buff = (void*)((char*)coll->sendbuff + offset);
        NCCLCHECK(classifyEnqueueP2pTask(comm, p2pTaskQueue, ncclFuncSend, collAPI, buff, coll->count, coll->datatype,
                                         r, stream, inPlace));
        offset += coll->count * elemSize;
      }
    }
    NCCLCHECK(classifyEnqueueP2pTask(comm, p2pTaskQueue, ncclFuncRecv, collAPI, coll->recvbuff, coll->count,
                                     coll->datatype, coll->root, stream, inPlace));
  } else {
    return ncclInternalError;
  }

  ncclMemoryPoolFree(&comm->memPool_ncclRawTask, tInfo->raw);
  tInfo->raw = nullptr;
  return ncclSuccess;
}

static bool taskUsesSymKernel(struct ncclTaskTuningInfo* tInfo) {
  return tInfo->tuningOut.valid && tInfo->tuningOut.symKernelId >= 0 &&
         tInfo->tuningOut.symKernelId < ncclSymkKernelId_Count;
}

static bool taskUsesCe(struct ncclTaskTuningInfo* tInfo) {
  return tInfo->tuningOut.valid && (tInfo->tuningOut.ceMethodId == ncclCeMethodId_AllGather_UC ||
                                    tInfo->tuningOut.ceMethodId == ncclCeMethodId_AllGather_MC);
}

static bool taskUsesAllGatherV(struct ncclTaskTuningInfo* tInfo) {
  return tInfo->raw->kind == ncclTaskKindAllGatherV;
}

ncclResult_t ncclTaskClassification(struct ncclComm* comm, struct ncclTaskTuningInfoQueue* tiq,
                                    struct ncclClassifiedTaskQueues* ctq) {
  struct ncclTaskTuningInfo* tInfo;
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* outQueue;

  if (comm == nullptr || tiq == nullptr || ctq == nullptr) return ncclInvalidArgument;

  ncclIntruQueueConstruct(&ctq->symTaskQueue);
  ncclIntruQueueConstruct(&ctq->legacyTaskQueue);
  ncclIntruQueueConstruct(&ctq->allgathervTaskQueue);
  ncclIntruQueueConstruct(&ctq->p2pTaskQueue);
  ncclIntruQueueConstruct(&ctq->rmaTaskQueue);
  ncclIntruQueueConstruct(&ctq->ceTaskQueue);

  while (!ncclIntruQueueEmpty(&tiq->queue)) {
    tInfo = ncclIntruQueueDequeue(&tiq->queue);
    if (tInfo->raw == nullptr) return ncclInternalError;

    if (tInfo->raw->kind == ncclTaskKindColl &&
        (tInfo->raw->coll.func == ncclFuncAlltoAll || tInfo->raw->coll.func == ncclFuncAlltoAllv ||
         tInfo->raw->coll.func == ncclFuncScatter ||
         tInfo->raw->coll.func == ncclFuncGather)) {
      NCCLCHECK(classifyCollToP2pTasks(comm, tInfo, &ctq->p2pTaskQueue));
      continue;
    }

    if (tInfo->raw->kind == ncclTaskKindSendRecv) {
      outQueue = &ctq->p2pTaskQueue;
    } else if (tInfo->raw->kind == ncclTaskKindRma) {
      outQueue = &ctq->rmaTaskQueue;
    } else if (taskUsesCe(tInfo)) {
      outQueue = &ctq->ceTaskQueue;
    } else if (taskUsesSymKernel(tInfo)) {
      outQueue = &ctq->symTaskQueue;
    } else if (taskUsesAllGatherV(tInfo)) {
      outQueue = &ctq->allgathervTaskQueue;
    } else {
      outQueue = &ctq->legacyTaskQueue;
    }
    ncclIntruQueueEnqueue(outQueue, tInfo);
  }
  return ncclSuccess;
}
