/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include "reshard_call_setup.h"

#include <limits>

static ncclResult_t validateDescriptorHeader(const char* apiName, const char* fieldName, size_t size,
                                             unsigned int version, size_t expectedSize) {
  NCCL_M2N_CHECK_ARG(size >= expectedSize, -1, "%s: %s descriptor size %zu is smaller than required size %zu", apiName,
                     fieldName, size, expectedSize);
  NCCL_M2N_CHECK_ARG(version == NCCL_M2N_API_VERSION, -1,
                     "%s: %s descriptor ABI version %u does not match library version %u", apiName, fieldName, version,
                     NCCL_M2N_API_VERSION);
  return ncclSuccess;
}

static ncclResult_t reshardCopyPublicTensor(const char* apiName, const char* fieldName, const ncclDistTensor_t* input,
                                            size_t* localShape, ncclMesh_t* mesh, int* meshDims,
                                            ncclDistTensor_t* tensor, int* placements) {
  NCCL_M2N_CHECK(validateDescriptorHeader(apiName, fieldName, input->size, input->version, sizeof(ncclDistTensor_t)));
  NCCL_M2N_CHECK_ARG(input->ndims >= 1 && input->ndims <= NCCL_RESHARD_MAX_TENSOR_DIMS, -1,
                     "%s: %s->ndims (%d) must be in [1, %d]", apiName, fieldName, input->ndims,
                     NCCL_RESHARD_MAX_TENSOR_DIMS);
  NCCL_M2N_CHECK_ARG(input->localShape != nullptr, -1, "%s: %s->localShape must be non-null", apiName, fieldName);
  NCCL_M2N_CHECK_ARG(input->mesh != nullptr, -1, "%s: %s->mesh must be non-null on every rank", apiName, fieldName);
  NCCL_M2N_CHECK(validateDescriptorHeader(apiName, "mesh", input->mesh->size, input->mesh->version,
                                          sizeof(ncclMesh_t)));
  NCCL_M2N_CHECK_ARG(input->mesh->ndims >= 1 && input->mesh->ndims <= NCCL_RESHARD_MAX_MESH_DIMS, -1,
                     "%s: %s->mesh->ndims (%d) must be in [1, %d]", apiName, fieldName, input->mesh->ndims,
                     NCCL_RESHARD_MAX_MESH_DIMS);
  NCCL_M2N_CHECK_ARG(input->mesh->dims != nullptr, -1, "%s: %s->mesh->dims must be non-null", apiName, fieldName);
  NCCL_M2N_CHECK_ARG(input->placements != nullptr, -1, "%s: %s->placements must be non-null", apiName, fieldName);

  *mesh = *input->mesh;
  mesh->ndims = NCCL_RESHARD_MAX_MESH_DIMS;
  mesh->dims = meshDims;
  for (int d = 0; d < NCCL_RESHARD_MAX_MESH_DIMS; d++) {
    meshDims[d] = (d < input->mesh->ndims) ? input->mesh->dims[d] : 1;
    placements[d] = (d < input->mesh->ndims) ? input->placements[d] : NCCL_RESHARD_REPLICATE;
  }

  *tensor = *input;
  for (int d = 0; d < input->ndims; d++) {
    localShape[d] = input->localShape[d];
  }
  tensor->localShape = localShape;
  tensor->mesh = mesh;
  tensor->placements = placements;
  return ncclSuccess;
}

static ncclResult_t reshardFixFullyReplicated(ncclMesh_t* mesh, int* placements) {
  if (placements[0] == NCCL_RESHARD_REPLICATE && placements[1] == NCCL_RESHARD_REPLICATE) {
    ReshardMeshInterval interval{};
    NCCL_M2N_CHECK(computeReshardMeshInterval(mesh, -1, &interval));
    mesh->dims[0] = interval.size;
    mesh->dims[1] = 1;
    placements[1] = NCCL_RESHARD_SHARD(0);
  }
  return ncclSuccess;
}

ncclResult_t reshardPrepareTensorSetup(const char* apiName, const ncclDistTensor_t* src, const ncclDistTensor_t* dst,
                                       ReshardTensorSetup* setup) {
  NCCL_M2N_CHECK_ARG(src != nullptr && dst != nullptr && setup != nullptr, -1,
                     "%s: src, dst, and setup must be non-null", apiName);
  NCCL_M2N_CHECK(reshardCopyPublicTensor(apiName, "src", src, setup->srcLocalShape, &setup->srcMesh, setup->srcMeshDims,
                                         &setup->srcTensor, setup->srcPlacements));
  NCCL_M2N_CHECK(reshardCopyPublicTensor(apiName, "dst", dst, setup->dstLocalShape, &setup->dstMesh, setup->dstMeshDims,
                                         &setup->dstTensor, setup->dstPlacements));
  NCCL_M2N_CHECK(validateReshardMeshDims(&setup->srcMesh, &setup->dstMesh));
  NCCL_M2N_CHECK_ARG(setup->srcTensor.ndims == setup->dstTensor.ndims, -1,
                     "%s: src->ndims (%d) and dst->ndims (%d) must match", apiName, setup->srcTensor.ndims,
                     setup->dstTensor.ndims);
  NCCL_M2N_CHECK_ARG(setup->srcTensor.dtype == setup->dstTensor.dtype, -1,
                     "%s: src->dtype (%d) and dst->dtype (%d) must match", apiName, (int)setup->srcTensor.dtype,
                     (int)setup->dstTensor.dtype);
  setup->ndims = setup->srcTensor.ndims;
  setup->elementSize = getNcclDtSize(setup->srcTensor.dtype);
  NCCL_M2N_CHECK_ARG(setup->elementSize != 0, -1, "%s: unsupported data type %d", apiName, (int)src->dtype);

  NCCL_M2N_CHECK(reshardFixFullyReplicated(&setup->srcMesh, setup->srcPlacements));
  NCCL_M2N_CHECK(reshardFixFullyReplicated(&setup->dstMesh, setup->dstPlacements));
  NCCL_M2N_CHECK(validateReshardPlacement(&setup->srcTensor, apiName, "src"));
  NCCL_M2N_CHECK(validateReshardPlacement(&setup->dstTensor, apiName, "dst"));
  return ncclSuccess;
}

ncclResult_t reshardValidateActiveBuffers(const char* apiName, int worldRank, const ncclDistTensor_t* src,
                                          const ncclDistTensor_t* dst) {
  NCCL_M2N_CHECK_ARG(src != nullptr && dst != nullptr && src->mesh != nullptr && dst->mesh != nullptr, worldRank,
                     "%s: tensor descriptors and meshes must be non-null", apiName);
  NCCL_M2N_CHECK_ARG(!reshardRankInMesh(src->mesh, worldRank) || src->dataPtr != nullptr, worldRank,
                     "%s: src->dataPtr must be non-null on active source rank %d", apiName, worldRank);
  NCCL_M2N_CHECK_ARG(!reshardRankInMesh(dst->mesh, worldRank) || dst->dataPtr != nullptr, worldRank,
                     "%s: dst->dataPtr must be non-null on active destination rank %d", apiName, worldRank);
  return ncclSuccess;
}

ncclResult_t reshardComputeLocalBytes(int logRank, const char* apiPrefix, const char* side, const void* buffer,
                                      const size_t* dims, int ndims, size_t elementSize, size_t* bytes) {
  *bytes = 0;
  if (buffer == nullptr) {
    return ncclSuccess;
  }
  size_t total = elementSize;
  for (int d = 0; d < ndims; d++) {
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(total, dims[d], &total), logRank,
                       "%s %s local byte size overflow at dim %d: current=%zu dim=%zu", apiPrefix, side, d, total,
                       dims[d]);
  }
  *bytes = total;
  return ncclSuccess;
}

ncclResult_t reshardComputeStagingGinCounts(int logRank, int numCtas, size_t maxPeers, int* signalCount,
                                            int* counterCount) {
  NCCL_M2N_CHECK_ARG(signalCount != nullptr && counterCount != nullptr, logRank,
                     "reshard: staging GIN count outputs must be non-null");
  NCCL_M2N_CHECK_ARG(numCtas > 0 && maxPeers > 0, logRank,
                     "reshard: staging GIN counts require positive numCtas and maxPeers (numCtas=%d, maxPeers=%zu)",
                     numCtas, maxPeers);

  size_t counters = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(static_cast<size_t>(numCtas), maxPeers, &counters) &&
                       counters <= static_cast<size_t>(std::numeric_limits<int>::max()),
                     logRank, "reshard: staging GIN counter count overflows NCCL int field (numCtas=%d, maxPeers=%zu)",
                     numCtas, maxPeers);
  size_t signals = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(counters, static_cast<size_t>(2), &signals) &&
                       signals <= static_cast<size_t>(std::numeric_limits<int>::max()),
                     logRank, "reshard: staging GIN signal count overflows NCCL int field (numCtas=%d, maxPeers=%zu)",
                     numCtas, maxPeers);

  *signalCount = static_cast<int>(signals);
  *counterCount = static_cast<int>(counters);
  return ncclSuccess;
}

ncclResult_t reshardGetOrCreateDevCommWithRequirements(ncclComm_t comm, int barrierCount, int ginSignalCount,
                                                       int ginCounterCount, ReshardDevCommBarrierKind barrierKind,
                                                       int ginContextCount, int ginConnectionType, cudaStream_t stream,
                                                       ncclDevComm* activeDevComm, ReshardDevCommUse* use,
                                                       bool retainPersistentState) {
  NCCL_M2N_CHECK_ARG(activeDevComm != nullptr && use != nullptr, -1,
                     "reshardGetOrCreateDevComm: output DevComm and use token must be non-null");
  cudaEvent_t completionEvent = nullptr;
  std::shared_ptr<ReshardDevCommUseState> useState;
  const int effectiveGinContextCount = (ginContextCount > 0) ? ginContextCount : reshardGetGinContextCount();
  const ReshardDevCommCacheKey key = {
    comm, barrierCount, ginSignalCount, ginCounterCount, effectiveGinContextCount, ginConnectionType, barrierKind
  };
  ncclDevComm* devComm = findCachedDevComm(key, &completionEvent, &useState);
  if (devComm != nullptr) {
    if (retainPersistentState && useState != nullptr) useState->bRetainPersistentState = true;
    *activeDevComm = *devComm;
    return reshardPrepareDevCommUse(completionEvent, useState, stream, use);
  }
  ReshardDevCommCacheReservation reservation;
  NCCL_M2N_CHECK(reserveDevCommCacheEntry(&reservation));
  ncclDevComm localDevComm = {};
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  if (barrierKind == RESHARD_DEVCOMM_BARRIER_WORLD) {
    reqs.worldGinBarrierCount = barrierCount;
  } else if (barrierKind == RESHARD_DEVCOMM_BARRIER_HYBRID) {
    reqs.barrierCount = barrierCount;
  }
  reqs.ginSignalCount = ginSignalCount;
  reqs.ginCounterCount = ginCounterCount;
  reqs.ginConnectionType = (decltype(reqs.ginConnectionType))ginConnectionType;
  reqs.ginContextCount = effectiveGinContextCount;

  {
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(ncclDevCommCreate(comm, &reqs, &localDevComm));
    NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  }
  ncclResult_t cacheResult = cacheDevComm(key, &localDevComm, retainPersistentState, &reservation);
  if (cacheResult != ncclSuccess) {
    {
      M2nApiUnlock apiUnlock;
      NCCL_M2N_CHECK_WARN(ncclDevCommDestroy(comm, &localDevComm));
    }
    return cacheResult;
  }
  devComm = findCachedDevComm(key, &completionEvent, &useState);
  NCCL_M2N_CHECK_ARG(devComm != nullptr, -1, "reshardGetOrCreateDevComm: newly cached DevComm was not found");
  *activeDevComm = *devComm;
  return reshardPrepareDevCommUse(completionEvent, useState, stream, use);
}

ncclResult_t reshardGetOrCreateDevComm(ncclComm_t comm, int numCtas, int ginSignalCount, int ginCounterCount,
                                       ReshardDevCommBarrierKind barrierKind, int ginContextCount, cudaStream_t stream,
                                       ncclDevComm* activeDevComm, ReshardDevCommUse* use) {
  return reshardGetOrCreateDevCommWithRequirements(comm, numCtas, ginSignalCount, ginCounterCount, barrierKind,
                                                   ginContextCount, NCCL_GIN_CONNECTION_FULL, stream, activeDevComm,
                                                   use);
}
