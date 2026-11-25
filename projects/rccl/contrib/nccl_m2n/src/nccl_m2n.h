/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * NCCL M2N — Public C API (reshard release)
 *
 * This header exposes the complete public API for the NCCL M2N library.
 *
 * Callers include this one header; internal implementation details are
 * in src/reshard_*.h (not installed).
 ************************************************************************/

#ifndef NCCL_M2N_H_
#define NCCL_M2N_H_

#include <limits.h>
#include <stddef.h>

#include "cuda_runtime.h"
#include "nccl.h"

// Library release version. These are the single source of truth for the
// version this library ships as. Distinct from NCCL_M2N_API_VERSION below,
// which guards the public struct ABI and moves only when that layout changes.
#define NCCL_M2N_MAJOR 0
#define NCCL_M2N_MINOR 2
#define NCCL_M2N_PATCH 0

// Packed version code: MAJOR*10000 + MINOR*100 + PATCH. Mirrors NCCL_VERSION_CODE.
#define NCCL_M2N_VERSION_CODE (NCCL_M2N_MAJOR * 10000 + NCCL_M2N_MINOR * 100 + NCCL_M2N_PATCH)

/** NCCL M2N public struct ABI version. Not the library release version. */
#define NCCL_M2N_API_VERSION 2u

/** Oldest NCCL release supported by this NCCL M2N API and implementation. */
#define NCCL_M2N_MIN_NCCL_VERSION_CODE NCCL_VERSION(2, 30, 5)

#if NCCL_VERSION_CODE < NCCL_M2N_MIN_NCCL_VERSION_CODE
#error "NCCL M2N requires NCCL 2.30.5 or newer"
#endif

#if defined(__GNUC__) || defined(__clang__)
#define NCCL_M2N_API __attribute__((visibility("default")))
#else
#define NCCL_M2N_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * Mesh Specification
 * ====================================================================*/

/** Maximum mesh rank accepted by this implementation. The public descriptors
 * use pointer-backed arrays so this implementation limit can grow without
 * another struct-layout change. */
#define NCCL_RESHARD_MAX_MESH_DIMS 2

/** Compatibility alias for code that used the former fixed array extent. */
#define NCCL_RESHARD_MESH_NDIMS NCCL_RESHARD_MAX_MESH_DIMS

/**
 * N-D mesh descriptor for one side of a reshard — pure topology, no
 * tensor placement.  Analogous to PyTorch DTensor's `DeviceMesh` or
 * JAX's `Mesh` (the per-tensor placement spec lives on the distributed
 * tensor, see ncclDistTensor_t::placements[]).  The mesh owns ranks
 * [startRank, startRank + product(dims[0..ndims))).
 *
 * dims is borrowed host memory. It must remain valid until the reshard call
 * returns. Grouped calls copy it while the entry is recorded, so it need not
 * remain valid until ncclM2nGroupEnd().
 */
typedef struct ncclMesh_v2 {
  /** Struct size; initialized by NCCL_M2N_MESH_INITIALIZER. */
  size_t size;
  /** NCCL M2N public struct ABI version. */
  unsigned int version;
  /** Number of active entries in dims (currently 1 or 2). */
  int ndims;
  /** Caller-owned positive mesh dimensions; host array of length ndims. */
  int* dims;
  /** First world rank in this side's contiguous rank interval.  Must be
   * non-negative, and the interval end must not exceed the communicator. */
  int startRank;
} ncclMesh_t;

/** Static initializer for a caller-owned mesh descriptor. */
#define NCCL_M2N_MESH_INITIALIZER \
  {                               \
    sizeof(ncclMesh_t),           \
    NCCL_M2N_API_VERSION,         \
    0,                            \
    NULL,                         \
    0,                            \
  }

/** Placement helper for replicated ncclDistTensor_t::placements[] entries. */
#define NCCL_RESHARD_REPLICATE (-1)

/** Placement helper for sharded ncclDistTensor_t::placements[] entries. */
#define NCCL_RESHARD_SHARD(td) (td)

/* ======================================================================
 * Distributed Tensor Descriptor
 * ====================================================================*/

/** Maximum tensor rank handled by ncclReshardWithWindow and ncclReshard. */
#define NCCL_RESHARD_MAX_TENSOR_DIMS 3

/**
 * Distributed tensor descriptor — the per-rank tile + the topology +
 * the placement of the global tensor over the mesh.  Modeled after
 * PyTorch DTensor's (local_tensor, DeviceMesh, placements) and JAX's
 * (jax.Array, NamedSharding(mesh, spec)) — the mesh is topology only;
 * placements[] describes how the global tensor maps onto mesh axes.
 */
typedef struct ncclDistTensor_v2 {
  /** Struct size; initialized by NCCL_M2N_DIST_TENSOR_INITIALIZER. */
  size_t size;
  /** NCCL M2N public struct ABI version. */
  unsigned int version;
  /** Local buffer for this rank.  Must be non-NULL when this rank belongs to
   * this side's mesh; may be NULL when it does not participate as this side. */
  void* dataPtr;
  /** Per-axis element count on this rank. This is borrowed host memory with
   * `ndims` entries. Inactive ranks must still provide the side's local shape
   * metadata so every rank derives identical transfer geometry. */
  size_t* localShape;
  /** Number of tensor dimensions (1, 2, or 3). */
  int ndims;
  /** Element data type.  Supported: ncclInt8, ncclUint8, ncclFloat8e4m3,
   * ncclFloat8e5m2, ncclFloat16, ncclBfloat16, ncclInt32, ncclUint32,
   * ncclFloat32, ncclInt64, ncclUint64, ncclFloat64. */
  ncclDataType_t dtype;
  /** Caller-owned mesh descriptor — topology only.  Required on every rank,
   * including ranks where dataPtr is NULL: the library uses both meshes
   * everywhere to compute who-talks-to-whom. */
  const ncclMesh_t* mesh;
  /** Per-mesh-axis tensor placements; required on every rank. This is
   * borrowed host memory with mesh->ndims entries and the same lifetime as
   * mesh->dims.
   * placements[i] is one of:
   *     NCCL_RESHARD_REPLICATE   Axis replicates the tensor slice.
   *     NCCL_RESHARD_SHARD(d)    Axis shards tensor dimension d.
   * Any combination of REPLICATE and SHARD is supported. SHARD dimensions
   * must satisfy 0 <= d < ndims. Mesh axes are applied in array order:
   * if both axes use SHARD(d), axis 1 subdivides the pieces from axis 0.
   * All ranks on one side must have the same localShape, with equally sized,
   * contiguous tiles and globalShape[d] = localShape[d] times the product
   * of mesh sizes whose placement is SHARD(d). */
  int* placements;
} ncclDistTensor_t;

/** Static initializer for a caller-owned distributed-tensor descriptor. */
#define NCCL_M2N_DIST_TENSOR_INITIALIZER \
  {                                      \
    sizeof(ncclDistTensor_t),            \
    NCCL_M2N_API_VERSION,                \
    NULL,                                \
    NULL,                                \
    0,                                   \
    (ncclDataType_t)0,                   \
    NULL,                                \
    NULL,                                \
  }

/* ======================================================================
 * Library Configuration
 * ====================================================================*/

/** Sentinel for config fields left at the library default. */
#define NCCL_M2N_CONFIG_UNDEF_INT INT_MIN

/** ABI guard value set by NCCL_M2N_CONFIG_INITIALIZER. */
#define NCCL_M2N_API_MAGIC 0x4d324e32u /* 'M2N2' */

/**
 * Modeled after ncclConfig_t.  Callers fill an ncclM2nConfig_t with
 * NCCL_M2N_CONFIG_INITIALIZER, optionally override fields, and pass
 * a pointer to ncclM2nInit() along with an output handle pointer.  The handle
 * stores a copy of the config for future API growth.  Process-global runtime
 * state is resolved on the first init call in an init/finalize epoch; runtime
 * env vars have highest precedence, so NCCL_RESHARD_NUM_CTAS can override
 * config.maxCta.  Passing NULL config is equivalent to passing an
 * all-default-initialized config.  Fields left at
 * NCCL_M2N_CONFIG_UNDEF_INT keep the library default.
 */
typedef struct ncclM2nConfig_v2 {
  /** Struct size; initialized by NCCL_M2N_CONFIG_INITIALIZER. */
  size_t size;
  /** ABI guard; initialized by NCCL_M2N_CONFIG_INITIALIZER. */
  unsigned int magic;
  /** NCCL M2N API version used by this config. */
  unsigned int version;
  /** Max number of CTAs used by reshard kernel. */
  int maxCta;
} ncclM2nConfig_t;

/** Static initializer for ncclM2nConfig_t. */
#define NCCL_M2N_CONFIG_INITIALIZER \
  {                                 \
    sizeof(ncclM2nConfig_t),        \
    NCCL_M2N_API_MAGIC,             \
    NCCL_M2N_API_VERSION,           \
    NCCL_M2N_CONFIG_UNDEF_INT,      \
  }

/* ======================================================================
 * Library Lifecycle
 * ====================================================================*/

/** Opaque handle returned by ncclM2nInit and passed to NCCL M2N calls. */
typedef struct ncclM2nHandle* ncclM2nHandle_t;

/**
 * Initialize NCCL M2N and return an explicit handle.
 *
 * The handle records the caller's configuration (or defaults when `config` is
 * NULL) and serves as the runtime context for subsequent NCCL M2N calls.
 * Implementation-owned runtime state, including environment-derived settings
 * and internal caches, is initialized once on the first successful init call in
 * an init/finalize epoch and released when the last handle is finalized.  Later
 * handles in the same epoch keep their own config copy but do not reconfigure
 * the shared runtime state.  Environment variables may override matching
 * configuration fields at runtime.
 *
 * @param[out] handle Pointer that receives the newly initialized handle.  Must
 *                    be non-NULL.  The output slot is set to NULL before any
 *                    work is performed and receives a valid handle on success.
 * @param[in] config  Optional configuration.  NULL means all defaults.
 *
 * @return ncclSuccess on success, ncclInvalidArgument for a NULL output pointer
 *         or malformed config, and ncclSystemError if the handle allocation
 *         fails.
 */
NCCL_M2N_API ncclResult_t ncclM2nInit(ncclM2nHandle_t* handle, const ncclM2nConfig_t* config);

/**
 * Finalize a handle returned by ncclM2nInit.
 *
 * Passing NULL finalizes the internal default handle used by reshard calls that
 * receive a NULL handle.  Repeated NULL finalization is allowed.  A non-NULL
 * handle must not be reused after this call returns; finalizing an unknown or
 * already-finalized non-NULL handle returns ncclInvalidArgument.  Internal
 * process-global caches and temporary PACK staging buffers are released when the
 * last active handle is finalized.  Caller-owned comms, windows, streams, and
 * buffers are not destroyed.  Reshard calls enqueue asynchronous CUDA work;
 * this function does not synchronize caller streams.  Before finalizing a
 * handle, callers must complete all M2N work submitted with that handle.  Before
 * finalizing the last explicit or default handle in an init/finalize epoch,
 * callers must complete all M2N work submitted in that epoch.  Communicators,
 * windows, streams, and buffers used by that work must remain valid until it
 * completes.  Callers must also not race finalization with a host reshard call
 * using the same handle or the internal default handle. Finalization while a
 * group is active on the calling host thread returns ncclInvalidArgument.
 *
 * @param[in] handle Handle returned by ncclM2nInit, or NULL for the internal
 *                   default handle.
 *
 * @return ncclSuccess on success, or ncclInvalidArgument for an unknown or
 *         already-finalized non-NULL handle.
 */
NCCL_M2N_API ncclResult_t ncclM2nFinalize(ncclM2nHandle_t handle);

/**
 * Return detail for the most recent NCCL M2N error on the calling host thread.
 *
 * The returned string is owned by the library and remains valid until the next
 * NCCL M2N API call on the same thread.  Returns an empty string when no detail
 * is available.  The numeric ncclResult_t remains the authoritative status.
 */
NCCL_M2N_API const char* ncclM2nGetLastError(void);

/* ======================================================================
 * Group Submission
 * ====================================================================*/

/**
 * Begin recording reshard calls on the calling host thread.
 *
 * Calls to ncclReshard or ncclReshardWithWindow made before the matching
 * ncclM2nGroupEnd are recorded instead of issued immediately.  One group may
 * span handles, communicators, and streams. Calls are bucketed by handle,
 * communicator, and normalized stream; the reshard entry point and window
 * argument do not create separate execution contexts.
 * `NULL` and `cudaStreamLegacy` identify the same execution context;
 * `cudaStreamPerThread` remains distinct.  Buckets are submitted sequentially
 * in first-occurrence order.  Entries outside documented fused paths retain
 * submission order; fused paths may partition independent entries by topology.
 * Nested groups are flattened into the outer group; only the outermost
 * ncclM2nGroupEnd issues the recorded calls.
 *
 * The library copies tensor descriptors and their meshes while recording.
 * Tensor storage remains caller-owned and must remain valid and unmodified
 * until ncclM2nGroupEnd has issued the group and the normal stream-ordered
 * completion contract has been satisfied.  Within one execution-context
 * bucket, PACK-fused calls may share read-only source storage, but every
 * destination range must remain non-overlapping with all ranges of other
 * entries.  Calls outside the PACK path retain submission order and do
 * not currently validate cross-entry aliases, so callers must avoid overlap
 * involving a destination.  Fused grouped calls must have no ordering
 * dependency.
 *
 * Each context bucket is a collective contract: every rank in its communicator
 * must record the same bucket entries in the same order, with identical
 * descriptor metadata.  Within each connected component of context buckets
 * whose communicator memberships overlap, callers must define one common total
 * order; each rank records its participating buckets in that order.  This
 * includes distinct contexts on the same communicator.  Consistent ordering
 * across ranks avoids cyclic collective-order deadlock; the library cannot
 * detect or prevent a mismatch.  Inactive ranks still provide the same meshes,
 * placements, local shapes, tensor ranks, and dtypes as active ranks.  A group
 * must start and end on the same host thread and must not span other M2N
 * lifecycle operations.  Callers must serialize concurrent ncclM2nGroupEnd
 * calls that target the same communicator; use separate communicators for
 * independently concurrent groups.
 *
 * @return ncclSuccess.
 */
NCCL_M2N_API ncclResult_t ncclM2nGroupStart(void);

/**
 * Close one group level and, at the outermost level, issue and clear the
 * reshard group recorded by ncclM2nGroupStart.
 *
 * Calls are partitioned by handle, communicator, and normalized stream.  The
 * reshard entry point and window argument do not create separate buckets.
 * Buckets are submitted sequentially on the calling host thread in
 * first-occurrence order; buckets on distinct streams may execute concurrently
 * on the device.  Within each bucket, compatible ncclReshard calls are
 * partitioned by normalized topology and packed into staging-bounded PACK
 * submissions.  Within one bucket, PACK-fused calls may share read-only
 * source storage, but every destination range must remain non-overlapping with
 * all ranges of other entries.  Calls outside the PACK staging path
 * retain submission order and do not currently validate cross-entry aliases,
 * so callers must avoid overlap involving a destination.  Calls must have no
 * ordering dependency when fused.  Storage must not overlap across buckets
 * because cross-bucket ranges are not checked.
 * Validation and execution errors that cannot be detected while recording are
 * returned here with the original group entry index.  Remaining entries in that
 * bucket and all later buckets are not issued.  An empty group succeeds.
 *
 * @return ncclSuccess when all recorded calls are issued, ncclInvalidUsage
 *         when no group is active, ncclInvalidArgument for incompatible calls,
 *         any deferred recorded group error (for example ncclSystemError), or
 *         the first error returned while issuing a recorded reshard call.
 */
NCCL_M2N_API ncclResult_t ncclM2nGroupEnd(void);

/* ======================================================================
 * Group Abort
 * ====================================================================*/

/**
 * Discard an active M2N group on the calling host thread.
 *
 * Clears all nested levels of a group begun by ncclM2nGroupStart without
 * issuing its recorded reshard calls.  The operation is idempotent when no
 * group is active.  It does not destroy M2N handles, abort an NCCL
 * communicator, or cancel CUDA/NCCL work that has already been submitted.
 *
 * @return ncclSuccess.
 */
NCCL_M2N_API ncclResult_t ncclM2nGroupAbort(void);

/* ======================================================================
 * Resharding Entry Points
 * ====================================================================*/

/**
 * Alternative reshard entry point for callers that provide an NCCL window.
 * The current implementation does not use the caller-provided window and
 * follows ncclReshard's transport selection, including
 * NCCL_RESHARD_COPY_ALGORITHM and its PACK default.  A future
 * implementation may use the window when beneficial; callers must not assume
 * that providing one guarantees zero-copy execution.
 *
 * Both descriptors are required on every rank — they each carry one
 * side's mesh, and the library reads both meshes everywhere to compute
 * which ranks own source data and which receive it.  A rank that does
 * not participate on a given side passes a fully-formed descriptor
 * with `dataPtr = NULL` on that side, while still providing shape
 * metadata for that side so all ranks validate the same plan.
 * For communicators larger than one rank, the source and destination mesh
 * rank intervals must be disjoint. A one-rank self-copy is retained for local
 * API-contract testing.
 *
 * @param[in] handle  NCCL M2N handle returned by ncclM2nInit, or NULL for the
 *                    internal default handle.
 * @param[in] comm    NCCL communicator containing all ranks (src + dst).
 * @param[in] window  User-provided NCCL window; may be NULL. The current
 *                    implementation may not use it for best performance.
 * @param[in] src     Source-side tensor descriptor (non-NULL on every rank).
 *                    `dataPtr` may be NULL on dest-only ranks.  `mesh`,
 *                    `placements`, `ndims`, and `dtype` are required and
 *                    `ndims` / `dtype` must match `dst->ndims` /
 *                    `dst->dtype`.
 * @param[in] dst     Destination-side tensor descriptor (non-NULL on every
 *                    rank).  `dataPtr` may be NULL on source-only ranks.
 * @param[in] stream  Explicit CUDA stream, or the default stream (NULL /
 *                    `cudaStreamLegacy` / `cudaStreamPerThread`).  Default-
 *                    stream callers run on a library-owned non-blocking
 *                    stream from a per-(comm, device) pool.  Readiness and
 *                    completion events preserve the caller stream's ordering
 *                    before and after the reshard operation.
 *
 * @return Same error contract as ncclReshard, including its fail-stop rule for
 *         errors reported after PACK host-RMA protocol entry.
 */
NCCL_M2N_API ncclResult_t ncclReshardWithWindow(ncclM2nHandle_t handle, ncclComm_t comm,
                                                ncclWindow_t window,
                                                const ncclDistTensor_t* src,
                                                const ncclDistTensor_t* dst,
                                                cudaStream_t stream);

/**
 * Copy/staging-based resharding (no caller-registered window needed).
 *
 * A communicator created with `ncclConfig_t.blocking = 0` is supported. The
 * library waits for it before accessing communicator metadata or resources and
 * returns any terminal asynchronous NCCL error.
 * For communicators larger than one rank, the source and destination mesh
 * rank intervals must be disjoint. A one-rank self-copy is retained for local
 * API-contract testing.
 *
 * @param[in] handle  NCCL M2N handle returned by ncclM2nInit, or NULL for the
 *                    internal default handle.
 * @param[in] comm    NCCL communicator containing all ranks (src + dst).
 * @param[in] src     Source-side tensor descriptor (non-NULL on every rank).
 * @param[in] dst     Destination-side tensor descriptor (non-NULL on every rank).
 * @param[in] stream  CUDA stream (explicit or default).
 *
 * @return ncclSuccess on success, ncclInvalidArgument if a precondition is
 *         violated, or another ncclResult_t reported by setup or transport
 *         operations. A pre-entry validation error enqueues no transfer work.
 *         If an error is reported after PACK host-RMA protocol work has
 *         begun, the participating communicator and M2N runtime epoch are
 *         fail-stop: all ranks must stop issuing M2N work on this communicator
 *         and coordinate communicator or process-group shutdown. Retrying the
 *         failed reshard or continuing with another tensor is unsupported;
 *         local resource quarantine is not distributed recovery.
 */
NCCL_M2N_API ncclResult_t ncclReshard(ncclM2nHandle_t handle, ncclComm_t comm,
                                      const ncclDistTensor_t* src, const ncclDistTensor_t* dst,
                                      cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* NCCL_M2N_H_ */
