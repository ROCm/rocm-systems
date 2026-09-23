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
#include <cstring>

// Cap on physical segments per registered buffer for the classic NET/IB path.
// Mirrors NCCL_RMA_MAX_SEGMENTS (net_ib/gin.cc): a ROCm/HIP dma-buf export only
// describes the first physical segment, so multi-segment cuMem/VMM buffers are
// registered as one MR per segment (AIRUNTIME-2351 classic-path follow-up).
// Defined here (rather than common.h) so the wire-protocol structs and the
// pure helpers below can both reference it.
#ifndef NCCL_IB_MAX_SEGMENTS
#define NCCL_IB_MAX_SEGMENTS 16
#endif

// Connect-time capability encoded in the reserved tail of the fixed-size
// devName field. This preserves the legacy metadata size and every legacy
// field offset, so old and new peers can complete the same wire exchange.
#define NCCL_IB_CAP_MULTISEG (1u << 0)
#define NCCL_IB_CONNECT_CAPS_MAGIC 0x4d534547u

struct ncclIbConnectCapsTrailer {
  uint32_t magic;
  uint32_t caps;
};

static inline void ncclIbSetConnectCaps(char* devName, size_t len, uint32_t caps) {
  if (len <= sizeof(struct ncclIbConnectCapsTrailer)) return;
  const size_t off = len - sizeof(struct ncclIbConnectCapsTrailer);
  const struct ncclIbConnectCapsTrailer trailer = {NCCL_IB_CONNECT_CAPS_MAGIC, caps};
  devName[off - 1] = '\0';
  std::memcpy(devName + off, &trailer, sizeof(trailer));
}

static inline uint32_t ncclIbGetConnectCaps(const char* devName, size_t len) {
  if (len < sizeof(struct ncclIbConnectCapsTrailer)) return 0;
  struct ncclIbConnectCapsTrailer trailer;
  std::memcpy(&trailer, devName + len - sizeof(trailer), sizeof(trailer));
  return trailer.magic == NCCL_IB_CONNECT_CAPS_MAGIC ? trailer.caps : 0;
}

// A multi-segment registration is only usable if the peer can parse the
// per-segment CTS side table. Single-segment buffers keep the legacy wire
// format, so they are accepted whatever the peer advertised.
static inline bool ncclIbDeclineMultiSegRegistration(uint32_t peerCaps, int nSegments) {
  return nSegments > 1 && (peerCaps & NCCL_IB_CAP_MULTISEG) == 0;
}

// Return the index of the segment that fully contains [addr, addr+len), or -1
// if addr is outside every segment or the range straddles a segment boundary
// (which the classic single-rkey wire protocol cannot express).
//
// segStart[s]/segLen[s] describe segment s (s in [0, nSegments)). A zero-length
// range is contained wherever addr lands. Overflow of addr+len is treated as
// not-contained.
static inline int ncclIbSegmentIndexForRange(int nSegments, const uintptr_t* segStart, const size_t* segLen,
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

// MR index for a 0-byte RDMA_WRITE at addr. Inclusive start; the exclusive end
// of the last segment still uses that last MR.
static inline int ncclIbSegmentIndexForZeroLength(int nSegments, const uintptr_t* segStart, const size_t* segLen,
                                                  uintptr_t addr) {
  int s = ncclIbSegmentIndexForRange(nSegments, segStart, segLen, addr, 0);
  if (s >= 0) return s;
  if (nSegments < 1 || segStart == NULL || segLen == NULL) return -1;
  uintptr_t end = segStart[nSegments - 1] + segLen[nSegments - 1];
  return addr == end ? nSegments - 1 : -1;
}

// True iff the segment layout is "uniform": every interior segment has the same
// size as segment 0, and the trailing segment is no larger. This keeps every
// segment boundary at a multiple of the segment size, so step-aligned transfers
// (stepSize <= segmentSize) never straddle a boundary. Registration declines
// non-uniform layouts and falls back to staging buffers.
static inline bool ncclIbSegmentsUniform(int nSegments, const size_t* segLen) {
  if (nSegments <= 1) return true;
  // Stride is the largest segment. Interior segments must equal it; the first
  // and last may be shorter when registration clips a physical allocation.
  size_t stride = 0;
  for (int s = 0; s < nSegments; s++) {
    if (segLen[s] > stride) stride = segLen[s];
  }
  if (stride == 0) return false;
  for (int s = 1; s < nSegments - 1; s++) {
    if (segLen[s] != stride) return false;
  }
  return segLen[0] <= stride && segLen[nSegments - 1] <= stride;
}

// Peer CTS side table is usable iff nSegments is in [1, maxSegments], remDevIdx
// is in range, segStart is strictly increasing, and every rkey is nonzero.
static inline bool ncclIbCtsRemoteLayoutValid(uint32_t nSegments, int remDevIdx, int maxDevs, int maxSegments,
                                              const uint64_t* segStart, const uint32_t* rkeys) {
  if (nSegments < 1 || nSegments > (uint32_t)maxSegments) return false;
  if (remDevIdx < 0 || remDevIdx >= maxDevs) return false;
  if (segStart == NULL || rkeys == NULL) return false;
  for (uint32_t s = 0; s < nSegments; s++) {
    if (s > 0 && segStart[s] <= segStart[s - 1]) return false;
    if (rkeys[s] == 0) return false;
  }
  return true;
}

// Write the indices of segments that overlap [addr, addr+len) into out[0..n).
// Returns the count, 0 if len==0 or nothing overlaps, or -1 on overflow /
// out-of-space. Used by classic IFlush to fence every GPUDirect segment the
// receive actually touched, not only data[last].
static inline int ncclIbSegmentsOverlappingRange(int nSegments, const uintptr_t* segStart, const size_t* segLen,
                                                 uintptr_t addr, size_t len, int* out, int maxOut) {
  if (nSegments < 1 || out == NULL || maxOut < 1) return -1;
  if (len == 0) return 0;
  uintptr_t end = addr + len;
  if (end < addr) return -1; // address overflow
  int n = 0;
  for (int s = 0; s < nSegments; s++) {
    uintptr_t b = segStart[s];
    uintptr_t e = b + segLen[s];
    if (e < b) continue;
    if (addr < e && b < end) {
      if (n >= maxOut) return -1;
      out[n++] = s;
    }
  }
  return n;
}

// ---------------------------------------------------------------------------
// Wire-protocol segment splitting.
//
// A single logical transfer maps offset o -> localBase+o on the sender and
// remoteBase+o on the receiver. The two sides may have *different* physical
// segment layouts (independent VMM allocations), so a contiguous logical range
// can straddle a boundary on either side. ncclIbSplitTransfer decomposes the
// range [off, off+len) into slices that each stay within one local AND one
// remote segment, resolving the per-slice local/remote VA. This is the classic
// analogue of RMA's ncclRmaBuildSegmentedWrs and is kept dependency-free so the
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
  int localSeg;         // sender segment index
  int remoteSeg;        // receiver segment index
};

// Return the segment index whose [segOff[s], segOff[s+1]) contains `off`, or -1.
static inline int ncclIbSegmentForOffset(int nSeg, const uint64_t* segOff, uint64_t off) {
  for (int s = 0; s < nSeg; s++) {
    if (off >= segOff[s] && off < segOff[s + 1]) return s;
  }
  return -1;
}

// Decompose local [localOff, localOff+len) and remote
// [remoteOff, remoteOff+len) into segment-contained slices. The two ranges may
// start at different offsets within their registrations. Writes up to
// maxSlices entries into out[] and returns the count, or -1 if either range is
// out of bounds or more than maxSlices slices are needed.
static inline int ncclIbSplitTransferAtOffsets(int nLocal, const uint64_t* localSegVA, const uint64_t* localSegOff,
                                               int nRemote, const uint64_t* remoteSegVA, const uint64_t* remoteSegOff,
                                               uint64_t localOff, uint64_t remoteOff, uint64_t len,
                                               struct ncclIbSegSlice* out, int maxSlices) {
  int n = 0;
  uint64_t localEnd = localOff + len;
  uint64_t remoteEnd = remoteOff + len;
  if (localEnd < localOff || remoteEnd < remoteOff) return -1; // overflow
  if (len == 0) return 0;
  uint64_t rem = len;
  while (rem > 0) {
    int ls = ncclIbSegmentForOffset(nLocal, localSegOff, localOff);
    int rs = ncclIbSegmentForOffset(nRemote, remoteSegOff, remoteOff);
    if (ls < 0 || rs < 0) return -1;              // outside a registered segment
    uint64_t localAvail = localSegOff[ls + 1] - localOff;
    uint64_t remoteAvail = remoteSegOff[rs + 1] - remoteOff;
    uint64_t sliceLen = rem;
    if (localAvail < sliceLen) sliceLen = localAvail;
    if (remoteAvail < sliceLen) sliceLen = remoteAvail;
    if (sliceLen == 0 || sliceLen > UINT32_MAX) return -1;
    if (n >= maxSlices) return -1;
    out[n].localAddr = localSegVA[ls] + (localOff - localSegOff[ls]);
    out[n].remoteAddr = remoteSegVA[rs] + (remoteOff - remoteSegOff[rs]);
    out[n].len = (uint32_t)sliceLen;
    out[n].localSeg = ls;
    out[n].remoteSeg = rs;
    n++;
    localOff += sliceLen;
    remoteOff += sliceLen;
    rem -= sliceLen;
  }
  return n;
}

// Common case where the local and remote request begin at the same logical
// offset within their registrations.
static inline int ncclIbSplitTransfer(int nLocal, const uint64_t* localSegVA, const uint64_t* localSegOff, int nRemote,
                                      const uint64_t* remoteSegVA, const uint64_t* remoteSegOff, uint64_t off,
                                      uint64_t len, struct ncclIbSegSlice* out, int maxSlices) {
  return ncclIbSplitTransferAtOffsets(nLocal, localSegVA, localSegOff, nRemote, remoteSegVA, remoteSegOff, off, off,
                                      len, out, maxSlices);
}

// Host VMM can report base 0; subtracting then wraps the remaining length.
static inline size_t ncclIbBytesRemainingInSegment(uintptr_t segPtr, uintptr_t segBase, size_t segSize) {
  return (segBase == 0) ? segSize : segSize - (segPtr - segBase);
}

#endif // NCCL_NET_IB_MULTISEG_H_
