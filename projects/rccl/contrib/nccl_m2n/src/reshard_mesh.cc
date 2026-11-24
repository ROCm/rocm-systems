/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <cstring>
#include <algorithm>
#include <limits>
#include "m2n_checked_math.h"
#include "reshard_types.h"
#include "reshard_internal.h"
#include "m2n_checks.h"

/* Reject invalid mesh metadata before private call setup canonicalizes it to
 * two axes. Host-only — safe to call before CUDA setup and from unit tests. */
ncclResult_t computeReshardMeshInterval(const ncclMesh_t* mesh, int logRank, ReshardMeshInterval* interval) {
  NCCL_M2N_CHECK_ARG(mesh != nullptr && interval != nullptr, logRank,
                     "reshard: mesh and output interval must be non-null");
  NCCL_M2N_CHECK_ARG(mesh->ndims >= 1 && mesh->ndims <= NCCL_RESHARD_MAX_MESH_DIMS, logRank,
                     "reshard: mesh ndims=%d must be in [1, %d]", mesh->ndims, NCCL_RESHARD_MAX_MESH_DIMS);
  NCCL_M2N_CHECK_ARG(mesh->dims != nullptr, logRank, "reshard: mesh dims must be non-null");
  NCCL_M2N_CHECK_ARG(mesh->startRank >= 0, logRank, "reshard: mesh startRank=%d must be non-negative",
                     mesh->startRank);

  size_t meshSize = 1;
  for (int d = 0; d < mesh->ndims; d++) {
    NCCL_M2N_CHECK_ARG(mesh->dims[d] > 0, logRank, "reshard: mesh dims[%d]=%d must be positive", d,
                       mesh->dims[d]);
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(meshSize, static_cast<size_t>(mesh->dims[d]), &meshSize), logRank,
                       "reshard: mesh size overflows at dims[%d]=%d", d, mesh->dims[d]);
  }
  NCCL_M2N_CHECK_ARG(meshSize <= static_cast<size_t>(std::numeric_limits<int>::max()), logRank,
                     "reshard: mesh size %zu exceeds the supported rank range", meshSize);
  NCCL_M2N_CHECK_ARG(mesh->startRank <= std::numeric_limits<int>::max() - static_cast<int>(meshSize), logRank,
                     "reshard: mesh interval overflows rank range: startRank=%d size=%zu", mesh->startRank,
                     meshSize);

  interval->startRank = mesh->startRank;
  interval->size = static_cast<int>(meshSize);
  interval->endRank = mesh->startRank + interval->size;
  return ncclSuccess;
}

ncclResult_t validateReshardMeshDims(const ncclMesh_t* srcMesh, const ncclMesh_t* dstMesh) {
  ReshardMeshInterval srcInterval{};
  ReshardMeshInterval dstInterval{};
  NCCL_M2N_CHECK(computeReshardMeshInterval(srcMesh, -1, &srcInterval));
  NCCL_M2N_CHECK(computeReshardMeshInterval(dstMesh, -1, &dstInterval));
  return ncclSuccess;
}

ncclResult_t validateReshardMeshBounds(const ncclMesh_t* srcMesh, const ncclMesh_t* dstMesh, int commSize,
                                      int logRank) {
  NCCL_M2N_CHECK_ARG(commSize > 0, logRank, "reshard: communicator size=%d must be positive", commSize);
  ReshardMeshInterval srcInterval{};
  ReshardMeshInterval dstInterval{};
  NCCL_M2N_CHECK(computeReshardMeshInterval(srcMesh, logRank, &srcInterval));
  NCCL_M2N_CHECK(computeReshardMeshInterval(dstMesh, logRank, &dstInterval));
  NCCL_M2N_CHECK_ARG(srcInterval.endRank <= commSize, logRank,
                     "reshard: src mesh interval [%d, %d) exceeds communicator size %d", srcInterval.startRank,
                     srcInterval.endRank, commSize);
  NCCL_M2N_CHECK_ARG(dstInterval.endRank <= commSize, logRank,
                     "reshard: dst mesh interval [%d, %d) exceeds communicator size %d", dstInterval.startRank,
                     dstInterval.endRank, commSize);
  /* Preserve the one-rank self-copy used by local API-contract tests. Real
   * cross-group resharding requires a multi-rank communicator and disjoint
   * source/destination rank intervals. */
  NCCL_M2N_CHECK_ARG(commSize == 1 || srcInterval.endRank <= dstInterval.startRank ||
                       dstInterval.endRank <= srcInterval.startRank,
                     logRank,
                     "reshard: source and destination mesh intervals must be disjoint "
                     "(src=[%d,%d), dst=[%d,%d))",
                     srcInterval.startRank, srcInterval.endRank, dstInterval.startRank, dstInterval.endRank);
  return ncclSuccess;
}

ncclResult_t computeStridesChecked(const size_t dims[], int ndims, size_t strides[]) {
  NCCL_M2N_CHECK_ARG(dims != nullptr && strides != nullptr && ndims >= 1 && ndims <= NCCL_RESHARD_MAX_TENSOR_DIMS,
                     -1, "computeStridesChecked: dims/strides must be non-null and ndims=%d must be in [1, %d]", ndims,
                     NCCL_RESHARD_MAX_TENSOR_DIMS);
  strides[ndims - 1] = 1;
  for (int d = ndims - 2; d >= 0; d--) {
    if (!m2nCheckedMulSize(strides[d + 1], dims[d + 1], &strides[d])) {
      NCCL_M2N_FAIL(ncclInvalidArgument, -1,
                    "computeStridesChecked: stride overflow at dim %d (nextStride=%zu, nextDim=%zu)", d,
                    strides[d + 1], dims[d + 1]);
    }
  }
  return ncclSuccess;
}

void computeStrides(const size_t dims[], int ndims, size_t strides[]) {
  strides[ndims - 1] = 1;
  for (int d = ndims - 2; d >= 0; d--) strides[d] = strides[d + 1] * dims[d + 1];
}

ncclResult_t validateReshardPlacement(const ncclDistTensor_t* tensor, const char* apiName, const char* fieldName) {
  for (int meshDim = 0; meshDim < tensor->mesh->ndims; meshDim++) {
    const int placement = tensor->placements[meshDim];
    if (placement == NCCL_RESHARD_REPLICATE) {
      continue;
    }
    if (!isShardPlacement(placement)) {
      NCCL_M2N_FAIL(ncclInvalidArgument, -1, "%s: %s->placements[%d]=%d is invalid", apiName, fieldName, meshDim,
                    placement);
    }

    const int shardDim = getShardTensorDim(placement);
    if (shardDim < 0 || shardDim >= tensor->ndims) {
      NCCL_M2N_FAIL(ncclInvalidArgument, -1, "%s: %s->placements[%d]=SHARD(%d) is outside tensor rank %d", apiName,
                    fieldName, meshDim, shardDim, tensor->ndims);
    }
  }

  return ncclSuccess;
}

void computeMeshGroupInfo(const ncclDistTensor_t* tensor, int worldRank, ncclReshardMeshGroupInfo* info) {
  const ncclMesh_t* mesh = tensor->mesh;
  memset(info, 0, sizeof(*info));
  info->repMeshDim = -1;
  info->shardCount = 1;
  info->repCount = 1;
  for (int d = 0; d < NCCL_RESHARD_MAX_TENSOR_DIMS; d++) info->shardLayout.shardCounts[d] = 1;
  int localRank = worldRank - mesh->startRank;
  for (int axis = NCCL_RESHARD_MESH_NDIMS - 1; axis >= 0; axis--) {
    info->meshPos[axis] = localRank % mesh->dims[axis];
    localRank /= mesh->dims[axis];
  }
  for (int axis = 0; axis < NCCL_RESHARD_MESH_NDIMS; axis++) {
    const int size = mesh->dims[axis];
    const int dim = tensor->placements[axis];
    info->shardLayout.axisTensorDims[axis] = dim;
    info->shardLayout.axisSizes[axis] = isShardPlacement(dim) ? size : 1;
    if (isShardPlacement(dim)) {
      info->shardLayout.shardCounts[dim] *= size;
      info->shardCount *= size;
      info->shardIdx = info->shardIdx * size + info->meshPos[axis];
    } else {
      info->repMeshDim = axis;
      info->repCount *= size;
      info->repIdx = info->repIdx * size + info->meshPos[axis];
    }
  }
}

int getMeshRank(const ncclDistTensor_t* tensor, const ncclReshardMeshGroupInfo* info, int shardIdx, int repIdx) {
  (void)info;
  int rank = 0;
  int stride = 1;
  for (int axis = NCCL_RESHARD_MESH_NDIMS - 1; axis >= 0; axis--) {
    const int size = tensor->mesh->dims[axis];
    int& index = isShardPlacement(tensor->placements[axis]) ? shardIdx : repIdx;
    rank += (index % size) * stride;
    index /= size;
    stride *= size;
  }
  return tensor->mesh->startRank + rank;
}

ncclResult_t resolveReshardDims(int worldRank, const size_t srcDims[], const ncclReshardShardLayout_t& srcLayout,
                                 const size_t dstDims[], const ncclReshardShardLayout_t& dstLayout, int ndims,
                                 size_t resolvedSrc[], size_t resolvedDst[]) {
  NCCL_M2N_CHECK_ARG(srcDims != nullptr && dstDims != nullptr, worldRank,
                     "reshard: source and destination local shapes must be provided on every rank");
  for (int d = 0; d < ndims; d++) {
    size_t srcGlobal = 0, dstGlobal = 0;
    NCCL_M2N_CHECK_ARG(srcDims[d] > 0 && dstDims[d] > 0, worldRank,
                       "reshard: local extents must be positive at tensor dim %d", d);
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(srcDims[d], (size_t)srcLayout.shardCounts[d], &srcGlobal) &&
                         m2nCheckedMulSize(dstDims[d], (size_t)dstLayout.shardCounts[d], &dstGlobal), worldRank,
                       "reshard: global shape overflow at tensor dim %d", d);
    NCCL_M2N_CHECK_ARG(srcGlobal == dstGlobal, worldRank,
                       "reshard: src/dst global shape mismatch at dim %d (%zu != %zu)", d, srcGlobal, dstGlobal);
    resolvedSrc[d] = srcDims[d];
    resolvedDst[d] = dstDims[d];
  }
  return ncclSuccess;
}

static ncclResult_t computeGlobalRangeChecked(const size_t localDims[], int ndims,
                                              const ncclReshardShardLayout_t& layout, int shardIdx,
                                              size_t globalStart[], size_t globalEnd[]) {
  NCCL_M2N_CHECK_ARG(localDims != nullptr && globalStart != nullptr && globalEnd != nullptr &&
                       ndims >= 1 && ndims <= NCCL_RESHARD_MAX_TENSOR_DIMS && shardIdx >= 0, -1,
                     "computeGlobalRangeChecked: invalid dimensions, outputs, or shard index");
  size_t coordinates[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
  size_t strides[NCCL_RESHARD_MAX_TENSOR_DIMS];
  for (int d = 0; d < ndims; d++) strides[d] = 1;
  // Decode from the last mesh axis. For repeated Shard(d), earlier axes
  // select larger blocks and later axes subdivide them.
  for (int axis = NCCL_RESHARD_MESH_NDIMS - 1; axis >= 0; axis--) {
    const int size = layout.axisSizes[axis];
    const int dim = layout.axisTensorDims[axis];
    NCCL_M2N_CHECK_ARG(size > 0 && (dim == NCCL_RESHARD_REPLICATE || (dim >= 0 && dim < ndims)), -1,
                       "computeGlobalRangeChecked: invalid shard layout at mesh axis %d", axis);
    const int coordinate = shardIdx % size;
    shardIdx /= size;
    if (dim >= 0) {
      coordinates[dim] += (size_t)coordinate * strides[dim];
      strides[dim] *= (size_t)size;
    }
  }
  NCCL_M2N_CHECK_ARG(shardIdx == 0, -1, "computeGlobalRangeChecked: shard index outside mesh");
  for (int d = 0; d < ndims; d++) {
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(coordinates[d], localDims[d], &globalStart[d]) &&
                         m2nCheckedAddSize(globalStart[d], localDims[d], &globalEnd[d]), -1,
                       "computeGlobalRangeChecked: global range overflow at tensor dim %d", d);
  }
  return ncclSuccess;
}

bool computeOverlap(const size_t srcStart[], const size_t srcEnd[], const size_t dstStart[], const size_t dstEnd[],
                    int ndims, size_t overlapStart[], size_t overlapEnd[]) {
  for (int d = 0; d < ndims; d++) {
    overlapStart[d] = std::max(srcStart[d], dstStart[d]);
    overlapEnd[d] = std::min(srcEnd[d], dstEnd[d]);
    if (overlapStart[d] >= overlapEnd[d]) return false;
  }
  return true;
}

void computeTransferPlan(const size_t srcDims[], const size_t srcStrides[],
                         const ncclReshardShardLayout_t& srcLayout, int srcShardIdx,
                         const size_t dstDims[], const size_t dstStrides[],
                         const ncclReshardShardLayout_t& dstLayout, int dstShardIdx, int ndims,
                         size_t elementsPerChunk, ncclReshardTransferPlan* plan) {
  (void)computeTransferPlanChecked(srcDims, srcStrides, srcLayout, srcShardIdx, dstDims, dstStrides,
                                  dstLayout, dstShardIdx, ndims, elementsPerChunk, plan);
}

ncclResult_t computeTransferPlanChecked(const size_t srcDims[], const size_t srcStrides[], const ncclReshardShardLayout_t& srcLayout,
                                        int srcShardIdx, const size_t dstDims[], const size_t dstStrides[],
                                        const ncclReshardShardLayout_t& dstLayout, int dstShardIdx, int ndims, size_t elementsPerChunk,
                                        ncclReshardTransferPlan* plan) {
  (void)elementsPerChunk;
  NCCL_M2N_CHECK_ARG(plan != nullptr, -1, "computeTransferPlanChecked: output plan must be non-null");
  memset(plan, 0, sizeof(*plan));
  NCCL_M2N_CHECK_ARG(ndims >= 1 && ndims <= NCCL_RESHARD_MAX_TENSOR_DIMS, -1,
                     "computeTransferPlanChecked: ndims=%d must be in [1, %d]", ndims,
                     NCCL_RESHARD_MAX_TENSOR_DIMS);
  size_t srcGlobalStart[NCCL_RESHARD_MAX_TENSOR_DIMS], srcGlobalEnd[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t dstGlobalStart[NCCL_RESHARD_MAX_TENSOR_DIMS], dstGlobalEnd[NCCL_RESHARD_MAX_TENSOR_DIMS];
  if (computeGlobalRangeChecked(srcDims, ndims, srcLayout, srcShardIdx, srcGlobalStart, srcGlobalEnd) !=
        ncclSuccess ||
      computeGlobalRangeChecked(dstDims, ndims, dstLayout, dstShardIdx, dstGlobalStart, dstGlobalEnd) !=
        ncclSuccess) {
    return ncclInvalidArgument;
  }
  if (!computeOverlap(srcGlobalStart, srcGlobalEnd, dstGlobalStart, dstGlobalEnd, ndims, plan->overlapStart,
                      plan->overlapEnd)) {
    plan->totalInnerTransfers = 0;
    return ncclSuccess;
  }
  size_t overlapSize[NCCL_RESHARD_MAX_TENSOR_DIMS];
  for (int d = 0; d < ndims; d++) overlapSize[d] = plan->overlapEnd[d] - plan->overlapStart[d];
  int innerContigStart = ndims - 1;
  size_t innerSize = 1;
  for (int d = ndims - 1; d >= 0; d--) {
    if (overlapSize[d] == srcDims[d] && overlapSize[d] == dstDims[d]) {
      if (!m2nCheckedMulSize(innerSize, overlapSize[d], &innerSize)) {
        NCCL_M2N_FAIL(ncclInvalidArgument, -1,
                      "computeTransferPlanChecked: inner size overflow at dim %d (current=%zu, overlap=%zu)", d,
                      innerSize, overlapSize[d]);
      }
      innerContigStart = d;
    } else {
      if (!m2nCheckedMulSize(innerSize, overlapSize[d], &innerSize)) {
        NCCL_M2N_FAIL(ncclInvalidArgument, -1,
                      "computeTransferPlanChecked: inner size overflow at dim %d (current=%zu, overlap=%zu)", d,
                      innerSize, overlapSize[d]);
      }
      innerContigStart = d;
      break;
    }
  }
  plan->innerSize = innerSize;
  plan->numOuterLoops = innerContigStart;
  plan->totalInnerTransfers = 1;
  for (int d = 0; d < innerContigStart; d++) {
    plan->outerCounts[d] = overlapSize[d];
    plan->outerSrcStrides[d] = srcStrides[d];
    plan->outerDstStrides[d] = dstStrides[d];
    if (!m2nCheckedMulSize(plan->totalInnerTransfers, overlapSize[d], &plan->totalInnerTransfers)) {
      NCCL_M2N_FAIL(ncclInvalidArgument, -1,
                    "computeTransferPlanChecked: transfer count overflow at dim %d (current=%zu, overlap=%zu)", d,
                    plan->totalInnerTransfers, overlapSize[d]);
    }
  }
  plan->srcBaseOffset = 0;
  plan->dstBaseOffset = 0;
  for (int d = 0; d < ndims; d++) {
    size_t srcLocalStart = plan->overlapStart[d] - srcGlobalStart[d];
    size_t dstLocalStart = plan->overlapStart[d] - dstGlobalStart[d];
    size_t srcTerm = 0;
    size_t dstTerm = 0;
    if (!m2nCheckedMulSize(srcLocalStart, srcStrides[d], &srcTerm) ||
        !m2nCheckedAddSize(plan->srcBaseOffset, srcTerm, &plan->srcBaseOffset) ||
        !m2nCheckedMulSize(dstLocalStart, dstStrides[d], &dstTerm) ||
        !m2nCheckedAddSize(plan->dstBaseOffset, dstTerm, &plan->dstBaseOffset)) {
      NCCL_M2N_FAIL(ncclInvalidArgument, -1,
                    "computeTransferPlanChecked: base-offset overflow at dim %d (srcStart=%zu, dstStart=%zu)", d,
                    srcLocalStart, dstLocalStart);
    }
  }
  return ncclSuccess;
}
