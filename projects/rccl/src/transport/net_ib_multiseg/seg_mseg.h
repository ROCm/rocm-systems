/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

#ifndef RCCL_TRANSPORT_MSEG_MSEG_H_
#define RCCL_TRANSPORT_MSEG_MSEG_H_

#include <stddef.h>
#include <stdint.h>

#include "transport/rccl_mseg.h" // RCCL: the fixed per-registration segment cap and its count check.

// RCCL: segment table and cursor for multi-segment registrations. Upstream registers one MR for the whole range.

// ibv_sge::length is 32 bits, so a slice stops there.
static constexpr uint64_t RCCL_IB_SGE_MAX_BYTES = (uint64_t)UINT32_MAX;

// base is the VA of byte 0. off has nSeg + 1 entries: off[0] == 0, off[nSeg] == size. A window over segments
// [k, k + nSeg) keeps base and points off at &off[k]; offsets stay absolute.
struct rcclIbSegTable {
  uint64_t base;
  uint32_t nSeg;
  const uint64_t* off;
};

struct rcclIbSegSlice {
  int localSeg;
  int remoteSeg;
  uint64_t localOff;
  uint64_t remoteOff;
  uint64_t len;
};

// localSeg/remoteSeg advance only forward.
struct rcclIbSegCursor {
  uint64_t localOff;
  uint64_t remoteOff;
  uint64_t remaining;
  int localSeg;
  int remoteSeg;
};

// Last segment when off is past the end. Callers then see off >= off[seg + 1] and stop. -1 when nSeg < 1.
static inline int rcclIbSegIndexOf(const uint64_t* off, int nSeg, uint64_t byteOff) {
  if (nSeg < 1 || off == NULL) return -1;
  for (int s = 0; s < nSeg; s++) {
    if (byteOff < off[s + 1]) return s;
  }
  return nSeg - 1;
}

static inline void rcclIbSegCursorInit(struct rcclIbSegCursor* c, const struct rcclIbSegTable* local,
                                       const struct rcclIbSegTable* remote, uint64_t lOff, uint64_t rOff,
                                       uint64_t len) {
  c->localOff = lOff;
  c->remoteOff = rOff;
  c->remaining = len;
  c->localSeg = local == NULL ? -1 : rcclIbSegIndexOf(local->off, (int)local->nSeg, lOff);
  c->remoteSeg = remote == NULL ? -1 : rcclIbSegIndexOf(remote->off, (int)remote->nSeg, rOff);
}

// 1 and a filled slice, or 0 when nothing remains or the offset is past a table. A 0 return with remaining
// still set means the range ran off the end.
static inline int rcclIbSegCursorNext(struct rcclIbSegCursor* c, const struct rcclIbSegTable* local,
                                      const struct rcclIbSegTable* remote, struct rcclIbSegSlice* out) {
  if (c->remaining == 0 || local == NULL || remote == NULL) return 0;
  if (c->remoteSeg < 0 || c->remoteSeg >= (int)remote->nSeg || c->remoteOff >= remote->off[c->remoteSeg + 1]) return 0;
  if (c->localSeg < 0 || c->localSeg >= (int)local->nSeg || c->localOff >= local->off[c->localSeg + 1]) return 0;
  uint64_t chunk = c->remaining;
  uint64_t localRem = local->off[c->localSeg + 1] - c->localOff;
  uint64_t remoteRem = remote->off[c->remoteSeg + 1] - c->remoteOff;
  if (localRem < chunk) chunk = localRem;
  if (remoteRem < chunk) chunk = remoteRem;
  if (RCCL_IB_SGE_MAX_BYTES < chunk) chunk = RCCL_IB_SGE_MAX_BYTES;
  if (chunk == 0) return 0;
  out->localSeg = c->localSeg;
  out->remoteSeg = c->remoteSeg;
  out->localOff = c->localOff;
  out->remoteOff = c->remoteOff;
  out->len = chunk;
  c->localOff += chunk;
  c->remoteOff += chunk;
  c->remaining -= chunk;
  if (c->localOff >= local->off[c->localSeg + 1]) c->localSeg++;
  if (c->remoteOff >= remote->off[c->remoteSeg + 1]) c->remoteSeg++;
  return 1;
}

#endif
