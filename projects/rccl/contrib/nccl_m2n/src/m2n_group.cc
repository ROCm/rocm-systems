/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <algorithm>
#include <cstdio>
#include <new>
#include <utility>
#include <vector>

#include "m2n_checks.h"
#include "nccl_m2n.h"
#include "reshard_internal.h"

namespace {

struct M2nGroupEntry {
  size_t originalIndex = 0;
  ncclM2nHandle_t handle = nullptr;
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
  bool hasSrc = false;
  bool hasDst = false;
  bool hasSrcMesh = false;
  bool hasDstMesh = false;
  bool hasSrcMeshDims = false;
  bool hasDstMeshDims = false;
  bool hasSrcLocalShape = false;
  bool hasDstLocalShape = false;
  bool hasSrcPlacements = false;
  bool hasDstPlacements = false;
  ncclDistTensor_t src{};
  ncclDistTensor_t dst{};
  ncclMesh_t srcMesh{};
  ncclMesh_t dstMesh{};
  size_t srcLocalShape[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
  size_t dstLocalShape[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
  int srcMeshDims[NCCL_RESHARD_MAX_MESH_DIMS] = {};
  int dstMeshDims[NCCL_RESHARD_MAX_MESH_DIMS] = {};
  int srcPlacements[NCCL_RESHARD_MAX_MESH_DIMS] = {};
  int dstPlacements[NCCL_RESHARD_MAX_MESH_DIMS] = {};
};

struct M2nGroupState {
  bool active = false;
  size_t depth = 0;
  ncclResult_t error = ncclSuccess;
  char errorDetail[M2N_LAST_ERROR_BYTES] = {};
  std::vector<M2nGroupEntry> entries;
};

thread_local M2nGroupState gM2nGroupState;

void clearGroupState(M2nGroupState* state) {
  state->active = false;
  state->depth = 0;
  state->error = ncclSuccess;
  state->errorDetail[0] = '\0';
  state->entries.clear();
}

ncclResult_t failGroup(ncclResult_t result, const char* detail) {
  gM2nGroupState.error = result;
  (void)snprintf(gM2nGroupState.errorDetail, sizeof(gM2nGroupState.errorDetail), "%s", detail);
  m2nSetLastError(gM2nGroupState.errorDetail);
  RESHARD_WARN(-1, "%s", gM2nGroupState.errorDetail);
  return result;
}

ncclResult_t validateGroupDescriptorHeader(const char* field, const ncclDistTensor_t* tensor) {
  if (tensor->size < sizeof(ncclDistTensor_t)) {
    char detail[M2N_LAST_ERROR_BYTES];
    (void)snprintf(detail, sizeof(detail), "ncclM2n group %s tensor descriptor is too small", field);
    return failGroup(ncclInvalidArgument, detail);
  }
  if (tensor->version != NCCL_M2N_API_VERSION) {
    char detail[M2N_LAST_ERROR_BYTES];
    (void)snprintf(detail, sizeof(detail), "ncclM2n group %s tensor descriptor ABI version %u is unsupported", field,
                   tensor->version);
    return failGroup(ncclInvalidArgument, detail);
  }
  if (tensor->mesh != nullptr) {
    if (tensor->mesh->size < sizeof(ncclMesh_t)) {
      char detail[M2N_LAST_ERROR_BYTES];
      (void)snprintf(detail, sizeof(detail), "ncclM2n group %s mesh descriptor is too small", field);
      return failGroup(ncclInvalidArgument, detail);
    }
    if (tensor->mesh->version != NCCL_M2N_API_VERSION) {
      char detail[M2N_LAST_ERROR_BYTES];
      (void)snprintf(detail, sizeof(detail), "ncclM2n group %s mesh descriptor ABI version %u is unsupported", field,
                     tensor->mesh->version);
      return failGroup(ncclInvalidArgument, detail);
    }
  }
  return ncclSuccess;
}

cudaStream_t normalizeGroupStream(cudaStream_t stream) {
  return stream == nullptr || stream == cudaStreamLegacy ? nullptr : stream;
}

bool groupContextMatches(const M2nGroupEntry& first, const M2nGroupEntry& next) {
  return first.handle == next.handle && first.comm == next.comm &&
         normalizeGroupStream(first.stream) == normalizeGroupStream(next.stream);
}

ncclResult_t indexGroupError(ncclResult_t result, size_t originalIndex, size_t groupSize) {
  const char* detail = ncclM2nGetLastError();
  char currentDetail[M2N_LAST_ERROR_BYTES];
  (void)snprintf(currentDetail, sizeof(currentDetail), "%s", detail != nullptr ? detail : "");
  char indexedDetail[M2N_LAST_ERROR_BYTES];
  const int prefixLength = snprintf(indexedDetail, sizeof(indexedDetail),
                                    "ncclM2nGroupEnd: entry %zu of %zu failed: ", originalIndex, groupSize);
  size_t used = prefixLength > 0 ? static_cast<size_t>(prefixLength) : 0;
  used = std::min(used, sizeof(indexedDetail) - 1);
  const size_t remaining = sizeof(indexedDetail) - used;
  (void)snprintf(indexedDetail + used, remaining, "%.*s", static_cast<int>(remaining - 1), currentDetail);
  m2nSetLastError(indexedDetail);
  return result;
}

ncclResult_t partitionGroupEntries(std::vector<M2nGroupEntry>&& entries,
                                   std::vector<std::vector<M2nGroupEntry>>* buckets) {
  try {
    buckets->reserve(entries.size());
    for (M2nGroupEntry& entry : entries) {
      auto bucket = buckets->begin();
      // ponytail: linear lookup preserves first occurrence; add a context
      // index if large mixed groups become measurable.
      while (bucket != buckets->end() && !groupContextMatches(bucket->front(), entry)) {
        ++bucket;
      }
      if (bucket == buckets->end()) {
        buckets->emplace_back();
        bucket = buckets->end() - 1;
      }
      bucket->push_back(std::move(entry));
    }
  } catch (const std::bad_alloc&) {
    return failGroup(ncclSystemError, "ncclM2n group failed to allocate context bucket storage");
  }
  return ncclSuccess;
}

ncclResult_t executeGroupBucket(std::vector<M2nGroupEntry>& entries, size_t groupSize) {
  bool canFuse = entries.size() > 1 && entries.size() <= kM2nGroupMaxFusionEntries;
  for (const M2nGroupEntry& entry : entries) {
    canFuse = canFuse && entry.hasSrc && entry.hasDst;
  }
  if (canFuse) {
    std::vector<ncclDistTensor_t> srcs;
    std::vector<ncclDistTensor_t> dsts;
    std::vector<ncclMesh_t> srcMeshes;
    std::vector<ncclMesh_t> dstMeshes;
    std::vector<size_t> originalIndices;
    try {
      srcs.resize(entries.size());
      dsts.resize(entries.size());
      srcMeshes.resize(entries.size());
      dstMeshes.resize(entries.size());
      originalIndices.resize(entries.size());
    } catch (const std::bad_alloc&) {
      return failGroup(ncclSystemError, "ncclM2n group failed to allocate fusion descriptor storage");
    }
    for (size_t i = 0; i < entries.size(); i++) {
      srcs[i] = entries[i].src;
      dsts[i] = entries[i].dst;
      srcs[i].localShape = entries[i].hasSrcLocalShape ? entries[i].srcLocalShape : nullptr;
      dsts[i].localShape = entries[i].hasDstLocalShape ? entries[i].dstLocalShape : nullptr;
      srcMeshes[i] = entries[i].srcMesh;
      dstMeshes[i] = entries[i].dstMesh;
      srcMeshes[i].dims = entries[i].hasSrcMeshDims ? entries[i].srcMeshDims : nullptr;
      dstMeshes[i].dims = entries[i].hasDstMeshDims ? entries[i].dstMeshDims : nullptr;
      srcs[i].placements = entries[i].hasSrcPlacements ? entries[i].srcPlacements : nullptr;
      dsts[i].placements = entries[i].hasDstPlacements ? entries[i].dstPlacements : nullptr;
      srcs[i].mesh = entries[i].hasSrcMesh ? &srcMeshes[i] : nullptr;
      dsts[i].mesh = entries[i].hasDstMesh ? &dstMeshes[i] : nullptr;
      originalIndices[i] = entries[i].originalIndex;
    }

    bool handled = false;
    size_t failedOriginalIndex = entries.front().originalIndex;
    ncclResult_t result = reshardTryExecuteStagingGroup(entries.front().handle, entries.front().comm, srcs.data(),
                                                        dsts.data(), originalIndices.data(), entries.size(),
                                                        entries.front().stream, &handled, &failedOriginalIndex);
    if (result != ncclSuccess) {
      return indexGroupError(result, failedOriginalIndex, groupSize);
    }
    if (handled) {
      return ncclSuccess;
    }
  }

  for (M2nGroupEntry& entry : entries) {
    ncclDistTensor_t src = entry.src;
    ncclDistTensor_t dst = entry.dst;
    src.localShape = entry.hasSrcLocalShape ? entry.srcLocalShape : nullptr;
    dst.localShape = entry.hasDstLocalShape ? entry.dstLocalShape : nullptr;
    ncclMesh_t srcMesh = entry.srcMesh;
    ncclMesh_t dstMesh = entry.dstMesh;
    srcMesh.dims = entry.hasSrcMeshDims ? entry.srcMeshDims : nullptr;
    dstMesh.dims = entry.hasDstMeshDims ? entry.dstMeshDims : nullptr;
    src.placements = entry.hasSrcPlacements ? entry.srcPlacements : nullptr;
    dst.placements = entry.hasDstPlacements ? entry.dstPlacements : nullptr;
    src.mesh = entry.hasSrcMesh ? &srcMesh : nullptr;
    dst.mesh = entry.hasDstMesh ? &dstMesh : nullptr;
    const ncclDistTensor_t* srcPtr = entry.hasSrc ? &src : nullptr;
    const ncclDistTensor_t* dstPtr = entry.hasDst ? &dst : nullptr;
    ncclResult_t result = ncclReshard(entry.handle, entry.comm, srcPtr, dstPtr, entry.stream);
    if (result != ncclSuccess) {
      return indexGroupError(result, entry.originalIndex, groupSize);
    }
  }
  return ncclSuccess;
}

} // namespace

bool m2nGroupIsActive() {
  return gM2nGroupState.active;
}

ncclResult_t m2nGroupEnqueueReshard(ncclM2nHandle_t handle, ncclComm_t comm, const ncclDistTensor_t* src,
                                    const ncclDistTensor_t* dst, cudaStream_t stream) {
  m2nClearLastError();
  NCCL_M2N_CHECK_ARG(gM2nGroupState.active, -1, "m2nGroupEnqueueReshard: no group is active on this host thread");
  if (gM2nGroupState.error != ncclSuccess) {
    m2nSetLastError(gM2nGroupState.errorDetail);
    return gM2nGroupState.error;
  }

  M2nGroupEntry entry;
  entry.originalIndex = gM2nGroupState.entries.size();
  entry.handle = handle;
  entry.comm = comm;
  entry.stream = stream;
  entry.hasSrc = src != nullptr;
  entry.hasDst = dst != nullptr;

  if (entry.hasSrc) {
    NCCL_M2N_CHECK(validateGroupDescriptorHeader("source", src));
    entry.src = *src;
    entry.hasSrcLocalShape = src->localShape != nullptr;
    if (entry.hasSrcLocalShape && src->ndims >= 1 && src->ndims <= NCCL_RESHARD_MAX_TENSOR_DIMS) {
      for (int d = 0; d < src->ndims; d++) {
        entry.srcLocalShape[d] = src->localShape[d];
      }
    }
    entry.hasSrcMesh = src->mesh != nullptr;
    entry.hasSrcPlacements = src->placements != nullptr;
    if (entry.hasSrcMesh) {
      entry.srcMesh = *src->mesh;
      entry.hasSrcMeshDims = src->mesh->dims != nullptr;
      if (src->mesh->ndims >= 1 && src->mesh->ndims <= NCCL_RESHARD_MAX_MESH_DIMS) {
        for (int d = 0; d < src->mesh->ndims; d++) {
          if (entry.hasSrcMeshDims) {
            entry.srcMeshDims[d] = src->mesh->dims[d];
          }
          if (entry.hasSrcPlacements) {
            entry.srcPlacements[d] = src->placements[d];
          }
        }
      }
    }
  }
  if (entry.hasDst) {
    NCCL_M2N_CHECK(validateGroupDescriptorHeader("destination", dst));
    entry.dst = *dst;
    entry.hasDstLocalShape = dst->localShape != nullptr;
    if (entry.hasDstLocalShape && dst->ndims >= 1 && dst->ndims <= NCCL_RESHARD_MAX_TENSOR_DIMS) {
      for (int d = 0; d < dst->ndims; d++) {
        entry.dstLocalShape[d] = dst->localShape[d];
      }
    }
    entry.hasDstMesh = dst->mesh != nullptr;
    entry.hasDstPlacements = dst->placements != nullptr;
    if (entry.hasDstMesh) {
      entry.dstMesh = *dst->mesh;
      entry.hasDstMeshDims = dst->mesh->dims != nullptr;
      if (dst->mesh->ndims >= 1 && dst->mesh->ndims <= NCCL_RESHARD_MAX_MESH_DIMS) {
        for (int d = 0; d < dst->mesh->ndims; d++) {
          if (entry.hasDstMeshDims) {
            entry.dstMeshDims[d] = dst->mesh->dims[d];
          }
          if (entry.hasDstPlacements) {
            entry.dstPlacements[d] = dst->placements[d];
          }
        }
      }
    }
  }

  try {
    gM2nGroupState.entries.push_back(entry);
  } catch (const std::bad_alloc&) {
    return failGroup(ncclSystemError, "ncclM2n group failed to allocate descriptor storage");
  }
  return ncclSuccess;
}

extern "C" ncclResult_t ncclM2nGroupStart(void) {
  m2nClearLastError();
  if (gM2nGroupState.active) {
    gM2nGroupState.depth++;
    return ncclSuccess;
  }
  clearGroupState(&gM2nGroupState);
  gM2nGroupState.active = true;
  gM2nGroupState.depth = 1;
  return ncclSuccess;
}

extern "C" ncclResult_t ncclM2nGroupEnd(void) {
  m2nClearLastError();
  if (!gM2nGroupState.active) {
    NCCL_M2N_FAIL(ncclInvalidUsage, -1, "ncclM2nGroupEnd: no group is active on this host thread");
  }

  if (gM2nGroupState.depth > 1) {
    gM2nGroupState.depth--;
    return ncclSuccess;
  }

  if (gM2nGroupState.error != ncclSuccess) {
    const ncclResult_t result = gM2nGroupState.error;
    char detail[sizeof(gM2nGroupState.errorDetail)];
    (void)snprintf(detail, sizeof(detail), "%s", gM2nGroupState.errorDetail);
    clearGroupState(&gM2nGroupState);
    m2nSetLastError(detail);
    return result;
  }

  std::vector<M2nGroupEntry> entries;
  entries.swap(gM2nGroupState.entries);
  clearGroupState(&gM2nGroupState);

  const size_t groupSize = entries.size();
  std::vector<std::vector<M2nGroupEntry>> buckets;
#ifdef NCCL_M2N_TESTING
  reshardResetFusedSubmissionCountForTest();
#endif
  NCCL_M2N_CHECK(partitionGroupEntries(std::move(entries), &buckets));
  for (std::vector<M2nGroupEntry>& bucket : buckets) {
    NCCL_M2N_CHECK(executeGroupBucket(bucket, groupSize));
  }
  return ncclSuccess;
}

extern "C" ncclResult_t ncclM2nGroupAbort(void) {
  m2nClearLastError();
  clearGroupState(&gM2nGroupState);
  std::vector<M2nGroupEntry>().swap(gM2nGroupState.entries);
  return ncclSuccess;
}
