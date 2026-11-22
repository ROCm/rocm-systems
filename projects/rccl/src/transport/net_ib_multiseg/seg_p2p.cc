/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "seg_p2p_mseg.h"

#include <atomic>
#include <string.h>

#include "alloc.h"
#include "debug.h"
#include "ibvwrap.h"

ncclResult_t rcclIbSegMrInit(struct rcclIbSegMr* m, int nSeg, void* const* segAddrs, const size_t* segLens) {
  m->nSegments = nSeg;
  NCCLCHECK(ncclCalloc(&m->segStart, (size_t)nSeg));
  NCCLCHECK(ncclCalloc(&m->segLen, (size_t)nSeg));
  NCCLCHECK(ncclCalloc(&m->segOff, (size_t)nSeg + 1));
  NCCLCHECK(ncclCalloc(&m->segMrs, (size_t)nSeg * RCCL_IB_MSEG_MAX_DEVS));
  for (int s = 0; s < nSeg; s++) {
    m->segStart[s] = (uintptr_t)segAddrs[s];
    m->segLen[s] = segLens[s];
    m->segOff[s] = m->segStart[s] - m->segStart[0];
  }
  m->segOff[nSeg] = m->segOff[nSeg - 1] + segLens[nSeg - 1];
  return ncclSuccess;
}

void rcclIbSegMrFreeArrays(struct rcclIbSegMr* m) {
  free(m->segStart);
  free(m->segLen);
  free(m->segOff);
  free(m->segMrs);
  m->segStart = NULL;
  m->segLen = NULL;
  m->segOff = NULL;
  m->segMrs = NULL;
}

ncclResult_t rcclIbSegMrPublishLayout(struct rcclIbSegMr* m, int ndevs, struct ibv_pd* const* pds) {
  static std::atomic<uint64_t> nextGen{0};
  const uint32_t nSeg = (uint32_t)m->nSegments;
  m->layoutBytes = rcclIbLayoutBlobBytes(nSeg, (uint32_t)ndevs);
  NCCLCHECK(ncclIbMalloc(&m->layoutBlob, m->layoutBytes));
  ((uint64_t*)m->layoutBlob)[0] = m->segStart[0];
  memcpy(rcclIbLayoutBlobOff(m->layoutBlob), m->segOff, sizeof(uint64_t) * ((size_t)nSeg + 1));
  for (int d = 0; d < ndevs; d++) {
    uint32_t* rkeys = rcclIbLayoutBlobRkeys(m->layoutBlob, nSeg, (uint32_t)d);
    for (uint32_t s = 0; s < nSeg; s++) rkeys[s] = rcclIbSegMrAt(m, (int)s, d)->rkey;
  }
  for (int d = 0; d < ndevs; d++) {
    NCCLCHECK(wrap_ibv_reg_mr(&m->layoutMrs[d], pds[d], m->layoutBlob, m->layoutBytes, IBV_ACCESS_REMOTE_READ));
  }
  m->layoutGen = ++nextGen;
  return ncclSuccess;
}

ncclResult_t rcclIbSegMrFreeLayout(struct rcclIbSegMr* m) {
  ncclResult_t ret = ncclSuccess;
  for (int i = 0; i < RCCL_IB_MSEG_MAX_DEVS; i++) {
    if (m->layoutMrs[i] == NULL) continue;
    ncclResult_t r = wrap_ibv_dereg_mr(m->layoutMrs[i]);
    if (ret == ncclSuccess) ret = r;
    m->layoutMrs[i] = NULL;
  }
  free(m->layoutBlob);
  m->layoutBlob = NULL;
  return ret;
}

struct ibv_mr* rcclIbSegMrForRange(const struct rcclIbSegMr* m, uintptr_t addr, size_t len, int dev) {
  int s = IbMsegSegmentIndexForRange(m->nSegments, m->segStart, m->segLen, addr, len);
  return (s < 0) ? NULL : rcclIbSegMrAt(m, s, dev);
}

ncclResult_t rcclIbSegLayoutFill(const struct rcclIbSegMr* m, int ndevs, uint64_t addr, uint64_t size,
                                 struct IbMsegSegLayout* side, uint32_t* shortRkeys, bool* multi) {
  side->nSegments = 0;
  side->byRef = 0;
  if (m->nSegments <= 1) return ncclSuccess;
  if (m->segStart == NULL || m->segOff == NULL || m->segMrs == NULL) return ncclInternalError;
  const uint64_t reqOff = addr - m->segStart[0];
  if (addr < m->segStart[0] || reqOff + size < reqOff || reqOff + size > m->segOff[m->nSegments]) {
    WARN("NET/IB: receive range is outside its %d-segment registration", m->nSegments);
    return ncclInternalError;
  }
  const int first = size ? rcclIbSegIndexOf(m->segOff, m->nSegments, reqOff)
                         : IbMsegSegmentForZeroLengthOffset(m->nSegments, m->segOff, reqOff);
  const int last = size ? rcclIbSegIndexOf(m->segOff, m->nSegments, reqOff + size - 1) : first;
  if (first < 0 || last < first) return ncclInternalError;
  const int count = last - first + 1;
  for (int s = first; s <= last; s++) {
    for (int j = 0; j < ndevs; j++) {
      if (rcclIbSegMrAt(m, s, j) == NULL) {
        WARN("NET/IB: receive is missing the MR for segment %d device %d", s, j);
        return ncclInternalError;
      }
    }
  }
  if (count == 1) {
    // RCCL: one segment holds the receive, so the short CTS carries that segment's rkeys.
    for (int j = 0; j < ndevs; j++) shortRkeys[j] = rcclIbSegMrAt(m, first, j)->rkey;
    return ncclSuccess;
  }
  *multi = true;
  if (count <= RCCL_IB_CTS_INLINE_SEGS) {
    side->nSegments = (uint32_t)count;
    for (int k = 0; k < count; k++) {
      side->segStart[k] = (uint64_t)m->segStart[first + k];
      for (int j = 0; j < ndevs; j++) rcclIbLayoutSetRkey(side, k, j, rcclIbSegMrAt(m, first + k, j)->rkey);
    }
    return ncclSuccess;
  }
  if (m->layoutBlob == NULL) {
    WARN("NET/IB: receive overlaps %d segments but its registration has no published layout", count);
    return ncclInternalError;
  }
  side->byRef = 1;
  side->ref.addr = (uint64_t)m->layoutBlob;
  side->ref.gen = m->layoutGen;
  side->ref.nSegments = (uint32_t)m->nSegments;
  side->ref.ndevs = (uint32_t)ndevs;
  for (int j = 0; j < ndevs; j++) side->ref.rkeys[j] = m->layoutMrs[j]->rkey;
  return ncclSuccess;
}

void rcclIbSegTableSingle(struct rcclIbSendSegTable* t, uint64_t base, uint64_t size, uint32_t rkey) {
  t->seg = {base, 1, t->offStore};
  t->multi = false;
  t->reqOff = 0;
  t->offStore[0] = 0;
  t->offStore[1] = size;
  t->rkeyStore[0] = rkey;
  t->rkeys = t->rkeyStore;
}

ncclResult_t rcclIbSegTableLocal(const struct rcclIbSegMr* m, uint64_t base, uint64_t size,
                                 struct rcclIbSendSegTable* t) {
  if (m == NULL || m->nSegments <= 1) {
    rcclIbSegTableSingle(t, base, size, 0);
    t->rkeys = NULL;
    return ncclSuccess;
  }
  if (m->segStart == NULL || m->segOff == NULL || m->segMrs == NULL || base < m->segStart[0]) return ncclInternalError;
  t->multi = true;
  t->rkeys = NULL;
  t->seg = {m->segStart[0], (uint32_t)m->nSegments, m->segOff};
  t->reqOff = base - m->segStart[0];
  return ncclSuccess;
}

ncclResult_t rcclIbSegTableInline(const volatile struct IbMsegSegLayout* side, uint64_t base, uint64_t size,
                                  int remDevIdx, struct rcclIbSendSegTable* t) {
  t->multi = true;
  t->seg.off = t->offStore;
  t->rkeys = t->rkeyStore;
  uint32_t remoteSegments = side->nSegments;
  uint64_t starts[RCCL_IB_CTS_INLINE_SEGS];
  const uint32_t nCopy = (remoteSegments >= 1 && remoteSegments <= RCCL_IB_CTS_INLINE_SEGS && remDevIdx >= 0 &&
                          remDevIdx < RCCL_IB_MSEG_MAX_DEVS)
                           ? remoteSegments
                           : 0;
  for (uint32_t s = 0; s < nCopy; s++) {
    starts[s] = side->segStart[s];
    t->rkeyStore[s] = rcclIbLayoutRkey(side, s, remDevIdx);
  }
  if (!IbMsegCtsRemoteLayoutValid(remoteSegments, remDevIdx, RCCL_IB_MSEG_MAX_DEVS, RCCL_IB_CTS_INLINE_SEGS,
                                  nCopy ? starts : NULL, nCopy ? t->rkeyStore : NULL)) {
    WARN("NET/IB: received invalid segment layout (nSegments=%u remDevIdx=%d)", remoteSegments, remDevIdx);
    return ncclInternalError;
  }
  t->seg.nSeg = remoteSegments;
  t->seg.base = starts[0];
  for (uint32_t s = 0; s < remoteSegments; s++) t->offStore[s] = starts[s] - starts[0];
  if (base < starts[0]) return ncclInternalError;
  t->reqOff = base - starts[0];
  uint64_t remoteReqEnd = t->reqOff + size;
  if (remoteReqEnd < t->reqOff) return ncclInternalError;
  while (t->seg.nSeg > 1 && t->offStore[t->seg.nSeg - 1] >= remoteReqEnd) t->seg.nSeg--;
  t->offStore[t->seg.nSeg] = remoteReqEnd;
  return ncclSuccess;
}

struct rcclIbLayoutCacheEntry* rcclIbLayoutCacheFind(struct rcclIbLayoutCache* cache, uint64_t addr, uint64_t gen) {
  for (int k = 0; k < RCCL_IB_LAYOUT_CACHE_ENTRIES; k++) {
    struct rcclIbLayoutCacheEntry* e = &cache->e[k];
    if (e->state != RCCL_IB_LAYOUT_EMPTY && e->addr == addr && e->gen == gen) return e;
  }
  return NULL;
}

// RCCL: post one signaled read of a referenced layout into entry e.
static ncclResult_t rcclIbLayoutRead(struct rcclIbLayoutCache* cache, const struct rcclIbLayoutReadQp* q,
                                     const volatile struct IbMsegSegLayout* side, struct rcclIbLayoutCacheEntry* e) {
  const uint32_t nSeg = side->ref.nSegments;
  const uint32_t ndevs = side->ref.ndevs;
  // RCCL: the count comes from the peer, so bound it before sizing the read.
  if (nSeg <= RCCL_IB_CTS_INLINE_SEGS || nSeg > RCCL_IB_REG_SEG_DEFAULT || ndevs < 1 || ndevs > RCCL_IB_MSEG_MAX_DEVS) {
    WARN("NET/IB: received invalid segment layout reference (nSegments=%u ndevs=%u)", nSeg, ndevs);
    return ncclInternalError;
  }
  const uint32_t rkey = (q->remDevIdx >= 0 && (uint32_t)q->remDevIdx < ndevs) ? side->ref.rkeys[q->remDevIdx] : 0;
  const size_t bytes = rcclIbLayoutBlobBytes(nSeg, ndevs);
  if (rkey == 0) {
    WARN("NET/IB: received invalid segment layout reference (remDevIdx=%d)", q->remDevIdx);
    return ncclInternalError;
  }
  // RCCL: forget the old key first, so a failed refill cannot leave a READY entry without a buffer.
  e->state = RCCL_IB_LAYOUT_EMPTY;
  e->addr = 0;
  e->gen = 0;
  if (e->cap < bytes || e->dev != q->devIndex) {
    if (e->mr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(e->mr));
    e->mr = NULL;
    free(e->buf);
    e->buf = NULL;
    e->cap = 0;
    NCCLCHECK(ncclIbMalloc(&e->buf, bytes));
    NCCLCHECK(wrap_ibv_reg_mr(&e->mr, q->pd, e->buf, bytes, IBV_ACCESS_LOCAL_WRITE));
    e->cap = bytes;
    e->dev = q->devIndex;
  }
  struct ibv_sge sge;
  sge.addr = (uint64_t)e->buf;
  sge.length = (uint32_t)bytes;
  sge.lkey = e->mr->lkey;
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)(e - cache->e);
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.wr.rdma.remote_addr = side->ref.addr;
  wr.wr.rdma.rkey = rkey;
  if (q->stamp != NULL) q->stamp(q->ctx, &wr);
  struct ibv_send_wr* bad = NULL;
  NCCLCHECK(wrap_ibv_post_send(q->qp, &wr, &bad));
  e->addr = side->ref.addr;
  e->gen = side->ref.gen;
  e->nSegments = nSeg;
  e->ndevs = ndevs;
  e->state = RCCL_IB_LAYOUT_READING;
  return ncclSuccess;
}

ncclResult_t rcclIbLayoutsReady(struct rcclIbLayoutCache* cache, const volatile struct IbMsegSegLayout* const* sides,
                                int n, rcclIbLayoutReadQpFn pickQp, void* arg, bool* ready) {
  *ready = true;
  const uint64_t stamp = ++cache->clock;
  struct rcclIbLayoutReadQp q;
  bool havePick = false;
  for (int r = 0; r < n; r++) {
    const volatile struct IbMsegSegLayout* side = sides[r];
    if (side == NULL || !side->byRef) continue;
    struct rcclIbLayoutCacheEntry* e = rcclIbLayoutCacheFind(cache, side->ref.addr, side->ref.gen);
    if (e != NULL) {
      if (e->state == RCCL_IB_LAYOUT_READY) e->lastUse = stamp;
      else *ready = false;
      continue;
    }
    *ready = false;
    // An empty entry, else the least recently used ready entry this slot does not use.
    for (int k = 0; k < RCCL_IB_LAYOUT_CACHE_ENTRIES; k++) {
      struct rcclIbLayoutCacheEntry* c = &cache->e[k];
      if (c->state == RCCL_IB_LAYOUT_EMPTY) {
        e = c;
        break;
      }
      if (c->state == RCCL_IB_LAYOUT_READY && c->lastUse < stamp && (e == NULL || c->lastUse < e->lastUse)) e = c;
    }
    if (e == NULL) continue; // every entry is in use; a later progress call retries
    if (!havePick) {
      memset(&q, 0, sizeof(q));
      NCCLCHECK(pickQp(arg, &q));
      havePick = true;
    }
    NCCLCHECK(rcclIbLayoutRead(cache, &q, side, e));
    e->lastUse = stamp;
  }
  return ncclSuccess;
}

ncclResult_t rcclIbLayoutReadDone(struct rcclIbLayoutCache* cache, uint64_t wrId) {
  if (wrId >= RCCL_IB_LAYOUT_CACHE_ENTRIES || cache->e[wrId].state != RCCL_IB_LAYOUT_READING) {
    WARN("NET/IB: unexpected segment layout read completion (wr_id=%lu)", wrId);
    return ncclInternalError;
  }
  struct rcclIbLayoutCacheEntry* e = &cache->e[wrId];
  if (!rcclIbLayoutBlobValid(e->buf, e->nSegments, e->ndevs)) {
    WARN("NET/IB: received invalid segment layout (nSegments=%u ndevs=%u)", e->nSegments, e->ndevs);
    e->state = RCCL_IB_LAYOUT_EMPTY;
    return ncclInternalError;
  }
  e->state = RCCL_IB_LAYOUT_READY;
  return ncclSuccess;
}

ncclResult_t rcclIbLayoutCacheFree(struct rcclIbLayoutCache* cache) {
  ncclResult_t ret = ncclSuccess;
  for (int k = 0; k < RCCL_IB_LAYOUT_CACHE_ENTRIES; k++) {
    struct rcclIbLayoutCacheEntry* e = &cache->e[k];
    if (e->mr != NULL) {
      ncclResult_t r = wrap_ibv_dereg_mr(e->mr);
      if (ret == ncclSuccess) ret = r;
    }
    free(e->buf);
    memset(e, 0, sizeof(*e));
  }
  return ret;
}

ncclResult_t rcclIbSegTableReferenced(struct rcclIbLayoutCache* cache, const volatile struct IbMsegSegLayout* side,
                                      uint64_t base, uint64_t size, int remDevIdx, struct rcclIbSendSegTable* t) {
  struct rcclIbLayoutCacheEntry* e = rcclIbLayoutCacheFind(cache, side->ref.addr, side->ref.gen);
  if (e == NULL || e->state != RCCL_IB_LAYOUT_READY || e->nSegments != side->ref.nSegments || remDevIdx < 0 ||
      (uint32_t)remDevIdx >= e->ndevs) {
    WARN("NET/IB: referenced segment layout is not ready (nSegments=%u remDevIdx=%d)", side->ref.nSegments, remDevIdx);
    return ncclInternalError;
  }
  t->multi = true;
  t->seg = {((const uint64_t*)e->buf)[0], e->nSegments, rcclIbLayoutBlobOff(e->buf)};
  t->rkeys = rcclIbLayoutBlobRkeys(e->buf, e->nSegments, (uint32_t)remDevIdx);
  if (base < t->seg.base) return ncclInternalError;
  t->reqOff = base - t->seg.base;
  if (t->reqOff + size < t->reqOff || t->reqOff + size > t->seg.off[t->seg.nSeg]) return ncclInternalError;
  e->lastUse = ++cache->clock;
  return ncclSuccess;
}
