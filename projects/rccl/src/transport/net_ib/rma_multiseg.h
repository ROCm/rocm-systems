/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_NET_IB_RMA_MULTISEG_H_
#define NCCL_NET_IB_RMA_MULTISEG_H_

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

// Shared limits and dependency-free boundary helpers for classic and CAST
// NET/IB RMA. Keep these here so host tests exercise the exact production math.
#ifndef NCCL_RMA_MAX_SEGMENTS
#define NCCL_RMA_MAX_SEGMENTS 16
#endif

// A paired transfer can split at every local and remote physical boundary.
// Additional UINT32_MAX splits share this fixed budget and are rejected before
// posting when the chain would exceed it.
#define NCCL_RMA_MAX_DATA_WRS (2 * NCCL_RMA_MAX_SEGMENTS)
#define NCCL_RMA_MAX_SIGNAL_WRS (NCCL_RMA_MAX_DATA_WRS + 1)
#define NCCL_RMA_MAX_FLUSH_WRS NCCL_RMA_MAX_SEGMENTS

static inline size_t ncclRmaSegmentSliceBytes(size_t remaining, size_t localRemaining, size_t remoteRemaining) {
  size_t chunk = remaining;
  if (localRemaining < chunk) chunk = localRemaining;
  if (remoteRemaining < chunk) chunk = remoteRemaining;
  if ((size_t)UINT32_MAX < chunk) chunk = (size_t)UINT32_MAX;
  return chunk;
}

static inline int ncclRmaWrIsSignaled(int wrIndex, int nWrs) {
  return nWrs > 0 && wrIndex == nWrs - 1;
}

static inline int ncclRmaWrCreditsAvailable(int outstanding, int requested, int capacity) {
  return outstanding >= 0 && requested >= 0 && requested <= capacity && outstanding <= capacity - requested;
}

static inline int ncclRmaSignalOffsetValid(size_t signalOff, size_t segmentEnd) {
  return (signalOff & (sizeof(uint64_t) - 1)) == 0 && signalOff <= segmentEnd &&
         sizeof(uint64_t) <= segmentEnd - signalOff;
}

static inline int ncclRmaLayoutsMatch(int lhsSegments, const size_t* lhsOffsets, int rhsSegments,
                                      const size_t* rhsOffsets) {
  if (lhsSegments != rhsSegments || lhsSegments < 1 || lhsSegments > NCCL_RMA_MAX_SEGMENTS) return 0;
  for (int s = 0; s <= lhsSegments; s++)
    if (lhsOffsets[s] != rhsOffsets[s]) return 0;
  return 1;
}

#endif // NCCL_NET_IB_RMA_MULTISEG_H_
