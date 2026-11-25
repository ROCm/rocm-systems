/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_M2N_CHECKS_H_
#define NCCL_M2N_CHECKS_H_

#include <sched.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"
#include "m2n_log.h"
#include "nccl_m2n.h"

void m2nClearLastError();
void m2nSetLastError(const char* message);

// Called after observing ncclInProgress from NCCL.
inline void m2nWarnNonblockingComm() {
  static std::atomic<bool> warned{false};
  if (warned.load(std::memory_order_relaxed) || reshardGetLogLevel() < RESHARD_LOG_WARN) return;
  if (warned.exchange(true, std::memory_order_relaxed)) return;
  RESHARD_WARN(-1, "NCCL non-blocking operations may add host submission overhead for M2N. "
                   "Consider a blocking communicator if your application permits it.");
}

// Keep this large enough for actionable details after ncclM2nGroupEnd adds its indexed prefix.
// Grouped errors may truncate the underlying detail to fit.
constexpr size_t M2N_LAST_ERROR_BYTES = 640;

static inline bool m2nSameMesh(const ncclMesh_t& a, const ncclMesh_t& b) {
  if (a.startRank != b.startRank || a.ndims != b.ndims || a.dims == nullptr || b.dims == nullptr) {
    return false;
  }
  for (int d = 0; d < a.ndims; d++) {
    if (a.dims[d] != b.dims[d]) {
      return false;
    }
  }
  return true;
}

static inline bool m2nSameTensorTopology(const ncclDistTensor_t& a, const ncclDistTensor_t& b) {
  if (a.mesh == nullptr || b.mesh == nullptr || a.placements == nullptr || b.placements == nullptr ||
      !m2nSameMesh(*a.mesh, *b.mesh)) {
    return false;
  }
  for (int d = 0; d < a.mesh->ndims; d++) {
    if (a.placements[d] != b.placements[d]) {
      return false;
    }
  }
  return true;
}

#define NCCL_M2N_CONCAT_INNER(a, b) a##b
#define NCCL_M2N_CONCAT(a, b) NCCL_M2N_CONCAT_INNER(a, b)
#define NCCL_M2N_UNIQUE(name) NCCL_M2N_CONCAT(name, __COUNTER__)

#define NCCL_M2N_SET_ERROR_IMPL(detailVar, ...)            \
  do {                                                     \
    char detailVar[M2N_LAST_ERROR_BYTES];                  \
    (void)snprintf(detailVar, sizeof(detailVar), __VA_ARGS__); \
    m2nSetLastError(detailVar);                            \
  } while (0)

#define NCCL_M2N_SET_ERROR(...) NCCL_M2N_SET_ERROR_IMPL(NCCL_M2N_UNIQUE(m2nErrorDetail_), __VA_ARGS__)

#define NCCL_M2N_FAIL(result, rank, ...)           \
  do {                                             \
    NCCL_M2N_SET_ERROR(__VA_ARGS__);               \
    RESHARD_WARN((rank), "%s", ncclM2nGetLastError()); \
    return result;                                 \
  } while (0)

/*
 * Library-safe macros: return an error code instead of calling exit().
 * Treat ncclInProgress as non-terminal; callers must establish readiness
 * before consuming asynchronous outputs. Use these in functions that return
 * ncclResult_t.
 */
#define NCCL_M2N_CHECK_IMPL(cmd, resultVar)                                                         \
  do {                                                                                              \
    ncclResult_t resultVar = (cmd);                                                                 \
    if ((resultVar) != ncclSuccess && (resultVar) != ncclInProgress) {                            \
      if (ncclM2nGetLastError()[0] == '\0') {                                                      \
        NCCL_M2N_SET_ERROR("NCCL operation %s failed: %s", #cmd, ncclGetErrorString(resultVar));  \
        RESHARD_WARN(-1, "%s", ncclM2nGetLastError());                                            \
      }                                                                                             \
      return resultVar;                                                                             \
    }                                                                                               \
  } while (0)

#define NCCL_M2N_CHECK(cmd) NCCL_M2N_CHECK_IMPL(cmd, NCCL_M2N_UNIQUE(m2nCheckResult_))

#define NCCL_M2N_CUDACHECK_IMPL(cmd, errorVar)                                                      \
  do {                                                                                             \
    cudaError_t errorVar = (cmd);                                                                  \
    if ((errorVar) != cudaSuccess) {                                                               \
      NCCL_M2N_SET_ERROR("CUDA operation %s failed: %s", #cmd, cudaGetErrorString(errorVar));     \
      RESHARD_WARN(-1, "%s", ncclM2nGetLastError());                                             \
      (void)cudaGetLastError();                                                                    \
      return ncclUnhandledCudaError;                                                               \
    }                                                                                              \
  } while (0)

#define NCCL_M2N_CUDACHECK(cmd) NCCL_M2N_CUDACHECK_IMPL(cmd, NCCL_M2N_UNIQUE(m2nCudaError_))

/* cudaMemcpy3DAsync rejects a source or destination pitch above cudaDevAttrMaxPitch (INT32_MAX on current GPUs;
 * HIP also rejects a pitch equal to it). A >= 2 GiB contiguous pack/unpack span or a >= 2 GiB outer stride
 * produces one. Such a copy is issued as one flat cudaMemcpyAsync when it is contiguous on both sides, else row by
 * row: rows are then >= 2 GiB apart, so there are few of them. */
inline size_t m2nMaxPitch() {
  /* Cached per thread and device: on HIP the attribute query rebuilds the whole device-properties struct, and
   * this runs for every copy in the PACK and PIPE submission loops. */
  static thread_local int cachedDev = -1;
  static thread_local size_t cachedPitch = 0;
  int dev = 0, attr = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0x7fffffff;
  }
  if (dev == cachedDev) return cachedPitch;
  if (cudaDeviceGetAttribute(&attr, cudaDevAttrMaxPitch, dev) != cudaSuccess || attr <= 0) {
    (void)cudaGetLastError();
    return 0x7fffffff;
  }
  cachedDev = dev;
  cachedPitch = (size_t)attr;
  return cachedPitch;
}

inline cudaError_t m2nMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) {
  const size_t maxPitch = m2nMaxPitch();
  const cudaPitchedPtr& s = p->srcPtr;
  const cudaPitchedPtr& d = p->dstPtr;
  if (s.pitch < maxPitch && d.pitch < maxPitch) return cudaMemcpy3DAsync(p, stream);
  const cudaExtent& e = p->extent;
  char* srcBase = (char*)s.ptr + (p->srcPos.z * s.ysize + p->srcPos.y) * s.pitch + p->srcPos.x;
  char* dstBase = (char*)d.ptr + (p->dstPos.z * d.ysize + p->dstPos.y) * d.pitch + p->dstPos.x;
  const bool flat = e.height * e.depth == 1 ||
                    (s.pitch == e.width && d.pitch == e.width &&
                     (e.depth == 1 || (s.ysize == e.height && d.ysize == e.height)));
  if (flat) return cudaMemcpyAsync(dstBase, srcBase, e.width * e.height * e.depth, p->kind, stream);
  for (size_t z = 0; z < e.depth; z++) {
    for (size_t y = 0; y < e.height; y++) {
      cudaError_t err = cudaMemcpyAsync(dstBase + (z * d.ysize + y) * d.pitch, srcBase + (z * s.ysize + y) * s.pitch,
                                        e.width, p->kind, stream);
      if (err != cudaSuccess) return err;
    }
  }
  return cudaSuccess;
}

#define NCCL_M2N_CHECK_ARG(cond, rank, ...) \
  do {                                          \
    if (!(cond)) {                              \
      NCCL_M2N_FAIL(ncclInvalidArgument, (rank), __VA_ARGS__); \
    }                                           \
  } while (0)

/*
 * Warn-and-continue variants: log the error code/message but do NOT
 * return.  Use in teardown paths where we must keep iterating to free
 * other resources.
 */
#define NCCL_M2N_CHECK_WARN_IMPL(cmd, resultVar)                                                     \
  do {                                                                                              \
    ncclResult_t resultVar = (cmd);                                                                 \
    if ((resultVar) != ncclSuccess) {                                                               \
      RESHARD_WARN(-1, "NCCL operation %s failed: %s (continuing)", #cmd, ncclGetErrorString(resultVar)); \
    }                                                                                               \
  } while (0)

#define NCCL_M2N_CHECK_WARN(cmd) NCCL_M2N_CHECK_WARN_IMPL(cmd, NCCL_M2N_UNIQUE(m2nCheckWarnResult_))

#define NCCL_M2N_CUDACHECK_WARN_IMPL(cmd, errorVar)                                                    \
  do {                                                                                                \
    cudaError_t errorVar = (cmd);                                                                     \
    if ((errorVar) != cudaSuccess) {                                                                  \
      RESHARD_WARN(-1, "CUDA operation %s failed: %s (continuing)", #cmd, cudaGetErrorString(errorVar)); \
    }                                                                                                 \
  } while (0)

#define NCCL_M2N_CUDACHECK_WARN(cmd) \
  NCCL_M2N_CUDACHECK_WARN_IMPL(cmd, NCCL_M2N_UNIQUE(m2nCudaWarnError_))

/* A non-blocking communicator can publish operation outputs after returning.
 * Callers unlock the M2N API gate before waiting. */
inline ncclResult_t m2nWaitCommReady(ncclComm_t comm) {
  ncclResult_t state = ncclSuccess;
  do {
    if (state == ncclInProgress) {
      m2nWarnNonblockingComm();
      sched_yield();
    }
    const ncclResult_t queryResult = ncclCommGetAsyncError(comm, &state);
    if (queryResult != ncclSuccess) return queryResult;
  } while (state == ncclInProgress);
  return state;
}

#endif /* NCCL_M2N_CHECKS_H_ */
