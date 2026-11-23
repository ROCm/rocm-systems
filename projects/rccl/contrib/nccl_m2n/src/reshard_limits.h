/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_RESHARD_LIMITS_H_
#define NCCL_RESHARD_LIMITS_H_

#include <cstddef>

#include "nccl_m2n.h"

/*
 * Central definition of all compile-time constants used by the
 * nccl-reshard library.  Every translation unit includes this header
 * instead of defining its own copy.
 *
 * Integer constants are `inline constexpr` (header-only single
 * definition, typed, no preprocessor leakage); function-like helpers
 * are `constexpr` inline functions.  Names stay UPPER_CASE for ABI/
 * diff continuity with the previous `#define` era — clang-tidy
 * (`ConstantCase: aNy_CasE`) permits this.
 *
 * Public placement helpers and tensor-rank limit are owned by the
 * public header `nccl_m2n.h`; they are NOT redefined here.
 */

/* RING (hierarchical) algorithm array sizes. */
inline constexpr int MAX_SOURCES = 16;
inline constexpr int MAX_TARGETS = 64;
inline constexpr int MAX_LOCAL_FOLLOWERS = 128;
inline constexpr int MAX_WARP_GROUPS = 15;
inline constexpr int MAX_SRC_WARPS = 8;

/* DIRECT algorithm array sizes. */
inline constexpr int MAX_DIRECT_SOURCES = 32;
inline constexpr int MAX_DIRECT_TARGETS = 64;

/* Default chunking parameters. */
inline constexpr int DEFAULT_ELEMENTS_PER_CHUNK = 32;
inline constexpr size_t CHUNK_SIZE_BYTES = 256ULL * 1024ULL;

/* Default kernel-launch parameters.
 *
 * DEFAULT_KERNEL_MAX_NTHREADS must match the value baked into each
 * __launch_bounds__(DEFAULT_KERNEL_MAX_NTHREADS, 1) declaration on the
 * resharding kernels — keep them in sync (NCCL v2.30 register-pressure
 * fix from commit 420236f).  __launch_bounds__ accepts constexpr
 * integer constants as well as #define values. */
inline constexpr int DEFAULT_NUM_CTAS = 8;
inline constexpr int DEFAULT_KERNEL_MAX_NTHREADS = 512;
inline constexpr int DEFAULT_GIN_CONTEXT_COUNT = 4;
inline constexpr int DEFAULT_GPUS_PER_NODE = 8;

/* Assign contiguous CTA ranges as evenly as possible across the available
 * GIN contexts.  Callers guarantee 0 <= ctaIdx < numCtas. */
constexpr int reshardMapCtaToGinContext(int ctaIdx, int numCtas, int ginContextCount) {
  if (numCtas < 1) {
    return 0;
  }
  int numContexts = ginContextCount < numCtas ? ginContextCount : numCtas;
  if (numContexts < 1) {
    numContexts = 1;
  }
  return (int)(((size_t)ctaIdx * (size_t)numContexts) / (size_t)numCtas);
}

/* Cache capacities. */
inline constexpr int MAX_WINDOW_CACHE_ENTRIES = 128;
inline constexpr int MAX_DEVCOMM_CACHE_ENTRIES = 64;
inline constexpr int MAX_PACK_STAGING_ENTRIES = 16;

/* Hard upper bound on staging slots across all configured buckets. */
inline constexpr int MAX_SPLIT_CONCURRENCY = 64;

/* Staging (copy-based) algorithm sizes.
 *
 * NCCL_RESHARD_STAGING_CHANNEL_SIZE is the user-visible data capacity per
 * channel.  The staging allocator adds the per-channel control region on top
 * and then splits the data capacity into RDMA and LSA halves. */
inline constexpr int STAGING_MAX_CHANNELS = 64;
inline constexpr int STAGING_DEFAULT_NUM_CHANNELS = 4;
inline constexpr size_t STAGING_DEFAULT_CHANNEL_DATA_SIZE = 8ULL * 1024ULL * 1024ULL;
inline constexpr size_t STAGING_DEFAULT_CHUNK_SIZE = 1ULL * 1024ULL * 1024ULL;
/* Host-RMA PIPE uses a compact fixed pool and larger CE/RMA chunks. The
 * default eight-lane, 128 MiB-per-lane pool is repartitioned over active peer
 * lanes; control space is allocated separately. */
inline constexpr int STAGING_PIPE_HOST_RMA_DEFAULT_NUM_CHANNELS = 8;
inline constexpr size_t STAGING_PIPE_HOST_RMA_DEFAULT_CHANNEL_DATA_SIZE = 128ULL * 1024ULL * 1024ULL;
inline constexpr size_t STAGING_PIPE_HOST_RMA_DEFAULT_CHUNK_SIZE = 32ULL * 1024ULL * 1024ULL;

inline constexpr size_t STAGING_CTRL_ENTRY_SIZE = 128;
inline constexpr int STAGING_MAX_REMOTES = 32;
inline constexpr int STAGING_LOCAL_FC_BASE = STAGING_MAX_REMOTES;
inline constexpr int STAGING_CTRL_ENTRIES = STAGING_MAX_REMOTES + MAX_TARGETS;
inline constexpr size_t STAGING_CTRL_REGION_SIZE = (size_t)STAGING_CTRL_ENTRIES * STAGING_CTRL_ENTRY_SIZE;
inline constexpr int STAGING_DEFAULT_CONTROL_SLOTS = 1;
inline constexpr int STAGING_PIPE_CONTROL_SLOTS = 32;
inline constexpr int STAGING_PIPE_GIN_PEERS_PER_SLOT = 16;
inline constexpr int STAGING_PIPE_GIN_CHANNELS_PER_PEER = 8;
inline constexpr int STAGING_LSA_FANOUT_MAX_FOLLOWERS = MAX_TARGETS;

inline constexpr size_t CTRL_FIELD_RDMA_TAIL = 0;
inline constexpr size_t CTRL_FIELD_RDMA_HEAD = 8;
inline constexpr size_t CTRL_FIELD_LSA_TAIL = 16;
inline constexpr size_t CTRL_FIELD_LSA_HEAD = 24;
inline constexpr size_t CTRL_FIELD_CURSOR_TAIL = 32;
inline constexpr size_t CTRL_FIELD_CURSOR_HEAD = 40;
inline constexpr size_t CTRL_FIELD_PIPE_READY = 48;
inline constexpr size_t CTRL_FIELD_PIPE_READY_SHADOW = 56;

inline constexpr int MAX_STAGING_BUFFER_ENTRIES = 16;

/* Placement classifiers — operate on the int placement value stored
 * in ncclDistTensor_t::placements[i] (NCCL_RESHARD_REPLICATE or
 * NCCL_RESHARD_SHARD(d)).  constexpr inline so they evaluate at compile
 * time in template/array-index contexts. */
constexpr inline bool isShardPlacement(int p) {
  return p >= 0;
}
constexpr inline int getShardTensorDim(int p) {
  return p;
}

#endif /* NCCL_RESHARD_LIMITS_H_ */
