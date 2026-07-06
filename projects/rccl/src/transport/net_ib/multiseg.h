/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_NET_IB_MULTISEG_H_
#define NCCL_NET_IB_MULTISEG_H_

// Pure, dependency-free segment-selection helpers for the classic NET/IB
// multi-segment DMA-BUF path (AIRUNTIME-2351 classic-path follow-up). Kept free
// of ibverbs / RCCL types so the boundary math can be unit-tested on the host
// without IB hardware (see test/.../NetIbMultiSegmentUnitTests.cpp).

#include <cstddef>
#include <cstdint>

// Cap on physical segments per registered buffer for the classic NET/IB path.
// Mirrors NCCL_GIN_MAX_SEGMENTS (net_ib/gin.cc): a ROCm/HIP dma-buf export only
// describes the first physical segment, so multi-segment cuMem/VMM buffers are
// registered as one MR per segment (AIRUNTIME-2351 classic-path follow-up).
// Defined here (rather than common.h) so the wire-protocol structs and the
// pure helpers below can both reference it.
#ifndef NCCL_IB_MAX_SEGMENTS
#define NCCL_IB_MAX_SEGMENTS 16
#endif

// Return the index of the segment that fully contains [addr, addr+len), or -1
// if addr is outside every segment or the range straddles a segment boundary
// (which the classic single-rkey wire protocol cannot express).
//
// segStart[s]/segLen[s] describe segment s (s in [0, nSegments)). A zero-length
// range is contained wherever addr lands. Overflow of addr+len is treated as
// not-contained.
static inline int ncclIbSegmentIndexForRange(int nSegments,
                                             const uintptr_t* segStart,
                                             const size_t* segLen,
                                             uintptr_t addr, size_t len) {
  for (int s = 0; s < nSegments; s++) {
    uintptr_t b = segStart[s];
    uintptr_t e = b + segLen[s];
    if (addr >= b && addr < e) {
      if (len == 0) return s;
      uintptr_t end = addr + len;
      if (end < addr) return -1;       // address overflow
      return (end <= e) ? s : -1;      // -1 => crosses a segment boundary
    }
  }
  return -1;
}

// True iff the segment layout is "uniform": every interior segment has the same
// size as segment 0, and the trailing segment is no larger. This keeps every
// segment boundary at a multiple of the segment size, so step-aligned transfers
// (stepSize <= segmentSize) never straddle a boundary. Registration declines
// non-uniform layouts and falls back to staging buffers.
static inline bool ncclIbSegmentsUniform(int nSegments, const size_t* segLen) {
  if (nSegments <= 1) return true;
  for (int s = 1; s < nSegments; s++) {
    bool last = (s == nSegments - 1);
    if ((!last && segLen[s] != segLen[0]) || (last && segLen[s] > segLen[0]))
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Option B (wire-protocol) segment-splitting.
//
// A single logical transfer maps offset o -> localBase+o on the sender and
// remoteBase+o on the receiver. The two sides may have *different* physical
// segment layouts (independent VMM allocations), so a contiguous logical range
// can straddle a boundary on either side. ncclIbSplitTransfer decomposes the
// range [off, off+len) into slices that each stay within one local AND one
// remote segment, resolving the per-slice local/remote VA. This is the classic
// analogue of GIN's ncclGinBuildSegmentedWrs and is kept dependency-free so the
// boundary math is unit-testable on the host.
//
// segOff[] is the cumulative logical start offset of each segment, with
// segOff[nSeg] == total byte count (so segment s spans [segOff[s], segOff[s+1])).
// segVA[s] is the virtual address of segment s's first byte.
// ---------------------------------------------------------------------------

struct ncclIbSegSlice {
  uint64_t localAddr;   // sender VA for this slice
  uint64_t remoteAddr;  // receiver VA for this slice
  uint32_t len;         // slice byte count
  int      localSeg;    // sender segment index
  int      remoteSeg;   // receiver segment index
};

// Return the segment index whose [segOff[s], segOff[s+1]) contains `off`, or -1.
static inline int ncclIbSegmentForOffset(int nSeg, const uint64_t* segOff, uint64_t off) {
  for (int s = 0; s < nSeg; s++) {
    if (off >= segOff[s] && off < segOff[s + 1]) return s;
  }
  return -1;
}

// Decompose [off, off+len) into segment-contained slices. Writes up to
// maxSlices entries into out[] and returns the count, or -1 if the range is
// out of bounds on either side or more than maxSlices slices are needed.
static inline int ncclIbSplitTransfer(
    int nLocal,  const uint64_t* localSegVA,  const uint64_t* localSegOff,
    int nRemote, const uint64_t* remoteSegVA, const uint64_t* remoteSegOff,
    uint64_t off, uint64_t len,
    struct ncclIbSegSlice* out, int maxSlices) {
  int n = 0;
  uint64_t end = off + len;
  if (end < off) return -1;                       // overflow
  if (len == 0) return 0;
  while (off < end) {
    int ls = ncclIbSegmentForOffset(nLocal, localSegOff, off);
    int rs = ncclIbSegmentForOffset(nRemote, remoteSegOff, off);
    if (ls < 0 || rs < 0) return -1;              // outside a registered segment
    uint64_t localSegEnd  = localSegOff[ls + 1];
    uint64_t remoteSegEnd = remoteSegOff[rs + 1];
    uint64_t sliceEnd = end;
    if (localSegEnd  < sliceEnd) sliceEnd = localSegEnd;
    if (remoteSegEnd < sliceEnd) sliceEnd = remoteSegEnd;
    if (n >= maxSlices) return -1;
    out[n].localAddr  = localSegVA[ls]  + (off - localSegOff[ls]);
    out[n].remoteAddr = remoteSegVA[rs] + (off - remoteSegOff[rs]);
    out[n].len        = (uint32_t)(sliceEnd - off);
    out[n].localSeg   = ls;
    out[n].remoteSeg  = rs;
    n++;
    off = sliceEnd;
  }
  return n;
}

#endif // NCCL_NET_IB_MULTISEG_H_
