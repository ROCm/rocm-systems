/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_M2N_RESHARD_CALL_SETUP_H_
#define NCCL_M2N_RESHARD_CALL_SETUP_H_

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"

#include "m2n_checks.h"
#include "m2n_checked_math.h"
#include "nccl_m2n.h"
#include "reshard_internal.h"
#include "reshard_types.h"

struct ReshardWorkStream {
  cudaStream_t stream;
  cudaEvent_t readyEvent;
  cudaEvent_t doneEvent;
};

/* Owns the shape, mesh-dimension, and placement storage referenced by
 * srcTensor and dstTensor. Public 1-D/2-D descriptors are canonicalized to
 * private 2-D layouts here. Keep the setup alive for the whole call. */
struct ReshardTensorSetup {
  ReshardTensorSetup() = default;
  ReshardTensorSetup(const ReshardTensorSetup&) = delete;
  ReshardTensorSetup& operator=(const ReshardTensorSetup&) = delete;
  ReshardTensorSetup(ReshardTensorSetup&&) = delete;
  ReshardTensorSetup& operator=(ReshardTensorSetup&&) = delete;

  size_t srcLocalShape[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
  size_t dstLocalShape[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
  int srcMeshDims[NCCL_RESHARD_MAX_MESH_DIMS] = {};
  int dstMeshDims[NCCL_RESHARD_MAX_MESH_DIMS] = {};
  int srcPlacements[NCCL_RESHARD_MAX_MESH_DIMS] = {};
  int dstPlacements[NCCL_RESHARD_MAX_MESH_DIMS] = {};
  ncclMesh_t srcMesh = NCCL_M2N_MESH_INITIALIZER;
  ncclMesh_t dstMesh = NCCL_M2N_MESH_INITIALIZER;
  ncclDistTensor_t srcTensor = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  ncclDistTensor_t dstTensor = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  int ndims = 0;
  size_t elementSize = 0;
};

static inline bool reshardAcquiredPoolSlot(const ReshardWorkStream* work) {
  return work->doneEvent != nullptr;
}

ncclResult_t reshardPrepareTensorSetup(const char* apiName, const ncclDistTensor_t* src, const ncclDistTensor_t* dst,
                                       ReshardTensorSetup* setup);

ncclResult_t reshardValidateActiveBuffers(const char* apiName, int worldRank, const ncclDistTensor_t* src,
                                          const ncclDistTensor_t* dst);

ncclResult_t reshardComputeLocalBytes(int logRank, const char* apiPrefix, const char* side, const void* buffer,
                                      const size_t* dims, int ndims, size_t elementSize, size_t* bytes);

ncclResult_t reshardComputeStagingGinCounts(int logRank, int numCtas, size_t maxPeers, int* signalCount,
                                            int* counterCount);

ncclResult_t reshardGetOrCreateDevComm(ncclComm_t comm, int numCtas, int ginSignalCount, int ginCounterCount,
                                       ReshardDevCommBarrierKind barrierKind, int ginContextCount, cudaStream_t stream,
                                       ncclDevComm* activeDevComm, ReshardDevCommUse* use);

/* General cache path for split comms whose logical grid size and DevComm
 * barrier count differ. ginConnectionType is part of the cache identity. */
ncclResult_t reshardGetOrCreateDevCommWithRequirements(ncclComm_t comm, int barrierCount, int ginSignalCount,
                                                       int ginCounterCount, ReshardDevCommBarrierKind barrierKind,
                                                       int ginContextCount, int ginConnectionType, cudaStream_t stream,
                                                       ncclDevComm* activeDevComm, ReshardDevCommUse* use,
                                                       bool retainPersistentState = false);

static inline ncclResult_t reshardRejectGraphCapture(const char* apiName, cudaStream_t stream) {
  cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
  NCCL_M2N_CUDACHECK(cudaStreamIsCapturing(stream, &captureStatus));
  if (captureStatus != cudaStreamCaptureStatusNone) {
    NCCL_M2N_FAIL(ncclInvalidUsage, -1, "%s does not support CUDA graph capture", apiName);
  }
  return ncclSuccess;
}

static inline ncclResult_t reshardMatchCommCudaDevice(ncclComm_t comm, int* currentCudaDev,
                                                      ncclCommProperties* commProps, ncclResult_t* propsResult) {
  NCCL_M2N_CUDACHECK(cudaGetDevice(currentCudaDev));
  *commProps = NCCL_COMM_PROPERTIES_INITIALIZER;
  *propsResult = ncclCommQueryProperties(comm, commProps);
  if (*propsResult == ncclSuccess && *currentCudaDev != commProps->cudaDev) {
    NCCL_M2N_CUDACHECK(cudaSetDevice(commProps->cudaDev));
  }
  return ncclSuccess;
}

static inline ncclResult_t reshardAcquireWorkStream(ncclComm_t comm, cudaStream_t callerStream, int currentCudaDev,
                                                    ncclResult_t propsResult, const ncclCommProperties* commProps,
                                                    ReshardWorkStream* work) {
  work->stream = callerStream;
  work->readyEvent = nullptr;
  work->doneEvent = nullptr;

  if (reshardUseInternalStreams()) {
    const int dev = (propsResult == ncclSuccess) ? commProps->cudaDev : currentCudaDev;
    NCCL_M2N_CHECK(streamPoolAcquire(comm, dev, &work->stream, &work->readyEvent, &work->doneEvent));
  }
  return ncclSuccess;
}

static inline ncclResult_t reshardStartWorkStream(cudaStream_t callerStream, const ReshardWorkStream* work) {
  if (reshardAcquiredPoolSlot(work)) {
    NCCL_M2N_CUDACHECK(cudaEventRecord(work->readyEvent, callerStream));
    NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(work->stream, work->readyEvent, 0));
  }
  return ncclSuccess;
}

static inline ncclResult_t reshardSetupWorkStream(ncclComm_t comm, cudaStream_t callerStream, int currentCudaDev,
                                                  ncclResult_t propsResult, const ncclCommProperties* commProps,
                                                  ReshardWorkStream* work) {
  NCCL_M2N_CHECK(reshardAcquireWorkStream(comm, callerStream, currentCudaDev, propsResult, commProps, work));
  return reshardStartWorkStream(callerStream, work);
}

static inline ncclResult_t reshardCompleteWorkStream(cudaStream_t callerStream, const ReshardWorkStream* work) {
  if (reshardAcquiredPoolSlot(work)) {
    NCCL_M2N_CUDACHECK(cudaEventRecord(work->doneEvent, work->stream));
    NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(callerStream, work->doneEvent, 0));
  }
  return ncclSuccess;
}

class ReshardWorkStreamCompletion {
public:
  ReshardWorkStreamCompletion(cudaStream_t callerStream, const ReshardWorkStream* work)
    : callerStream_(callerStream), work_(work) {}
  ReshardWorkStreamCompletion(const ReshardWorkStreamCompletion&) = delete;
  ReshardWorkStreamCompletion& operator=(const ReshardWorkStreamCompletion&) = delete;
  ~ReshardWorkStreamCompletion() {
    if (bActive_) {
      // Preserve the caller's primary error; success paths return complete() directly.
      (void)complete();
    }
  }

  ncclResult_t complete() {
    if (!bActive_) {
      return ncclSuccess;
    }
    bActive_ = false;
    return reshardCompleteWorkStream(callerStream_, work_);
  }

private:
  cudaStream_t callerStream_;
  const ReshardWorkStream* work_;
  bool bActive_ = true;
};

static inline ncclResult_t reshardDimsToBytes(int logRank, const char* apiName, int ndims, size_t elementSize,
                                              const size_t* srcDims, const size_t* dstDims,
                                              size_t srcDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS],
                                              size_t dstDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS]) {
  NCCL_M2N_CHECK_ARG(srcDims != nullptr && dstDims != nullptr, logRank,
                     "%s source and destination shape metadata must be present on every rank", apiName);
  for (int d = 0; d < NCCL_RESHARD_MAX_TENSOR_DIMS; d++) {
    srcDimsBytes[d] = 0;
    dstDimsBytes[d] = 0;
  }
  for (int d = 0; d < ndims; d++) {
    srcDimsBytes[d] = srcDims[d];
    dstDimsBytes[d] = dstDims[d];
  }
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(srcDimsBytes[ndims - 1], elementSize, &srcDimsBytes[ndims - 1]), logRank,
                     "%s source last dimension byte-size overflow: dim=%zu elementSize=%zu", apiName,
                     srcDims[ndims - 1], elementSize);
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(dstDimsBytes[ndims - 1], elementSize, &dstDimsBytes[ndims - 1]), logRank,
                     "%s destination last dimension byte-size overflow: dim=%zu elementSize=%zu", apiName,
                     dstDims[ndims - 1], elementSize);
  return ncclSuccess;
}

#endif // NCCL_M2N_RESHARD_CALL_SETUP_H_
