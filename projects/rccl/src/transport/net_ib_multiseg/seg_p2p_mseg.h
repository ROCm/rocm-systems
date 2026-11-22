/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef RCCL_TRANSPORT_P2P_MULTISEG_MSEG_H_
#define RCCL_TRANSPORT_P2P_MULTISEG_MSEG_H_

#include <stddef.h>
#include <stdint.h>

#include "nccl.h"
#include "plugin/nccl_net.h"
#include "seg_mseg.h"

// RCCL: multi-segment P2P: the CTS layout protocol, the per-segment MRs a handle embeds, and the sender's
// referenced-layout cache. These take only the parts they need, not the comm structs.

struct ibv_mr;
struct ibv_pd;
struct ibv_qp;
struct ibv_send_wr;

// RCCL: IB-MSEG's NCCL_IB_MAX_DEVS_PER_NIC and NCCL_NET_IB_MAX_RECVS; common_mseg.h static_asserts they match.
#define RCCL_IB_MSEG_MAX_DEVS NCCL_NET_MAX_DEVS_PER_NIC
#define RCCL_IB_MSEG_MAX_RECVS 8

// ---- CTS protocol ----

// RCCL: segments a CTS carries inline; a receive overlapping more sends a reference to the published layout.
#define RCCL_IB_CTS_INLINE_SEGS 4

// RCCL: set in every CTS entry's nreqs when the write includes the layout prefix, so single-segment receives
// keep the short write. nreqs is 16 bits.
#define NCCL_IB_CTS_NREQS_LAYOUT (1u << 15)

static inline uint32_t IbMsegCtsNreqs(uint32_t nreqs) { return nreqs & ~NCCL_IB_CTS_NREQS_LAYOUT; }

static inline bool IbMsegCtsHasLayout(uint32_t nreqs) { return (nreqs & NCCL_IB_CTS_NREQS_LAYOUT) != 0; }

// RCCL: index of the segment that fully contains [addr, addr+len), or -1 when addr is outside every segment or the
// range crosses a boundary. A zero-length range is contained wherever addr lands; addr+len overflow is -1.
static inline int IbMsegSegmentIndexForRange(int nSegments, const uintptr_t* segStart, const size_t* segLen,
                                             uintptr_t addr, size_t len) {
  for (int s = 0; s < nSegments; s++) {
    uintptr_t b = segStart[s];
    uintptr_t e = b + segLen[s];
    if (addr >= b && addr < e) {
      if (len == 0) return s;
      uintptr_t end = addr + len;
      if (end < addr) return -1;
      return (end <= e) ? s : -1;
    }
  }
  return -1;
}

// RCCL: peer CTS layout is usable iff nSegments is in [1, maxSegments], remDevIdx is in range, segStart is
// strictly increasing, and every rkey is nonzero.
static inline bool IbMsegCtsRemoteLayoutValid(uint32_t nSegments, int remDevIdx, int maxDevs, int maxSegments,
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

// RCCL: published layout, {va0, off[nSeg + 1], rkeys[ndevs][nSeg]}; off is from va0, so off[0] is 0.
static inline size_t rcclIbLayoutBlobBytes(uint32_t nSeg, uint32_t ndevs) {
  return sizeof(uint64_t) * ((size_t)nSeg + 2) + sizeof(uint32_t) * (size_t)ndevs * nSeg;
}
static inline uint64_t* rcclIbLayoutBlobOff(void* blob) { return (uint64_t*)blob + 1; }
static inline uint32_t* rcclIbLayoutBlobRkeys(void* blob, uint32_t nSeg, uint32_t dev) {
  return (uint32_t*)((uint64_t*)blob + nSeg + 2) + (size_t)dev * nSeg;
}

// RCCL: a read layout is usable iff its offsets start at 0 and increase and every rkey is nonzero.
static inline bool rcclIbLayoutBlobValid(void* blob, uint32_t nSeg, uint32_t ndevs) {
  if (blob == NULL || nSeg < 1 || ndevs < 1) return false;
  const uint64_t* off = rcclIbLayoutBlobOff(blob);
  if (off[0] != 0) return false;
  for (uint32_t s = 0; s < nSeg; s++) {
    if (off[s + 1] <= off[s]) return false;
  }
  const uint32_t* rkeys = rcclIbLayoutBlobRkeys(blob, nSeg, 0);
  for (size_t k = 0; k < (size_t)ndevs * nSeg; k++) {
    if (rkeys[k] == 0) return false;
  }
  return true;
}

// Segment whose rkey a 0-byte RDMA_WRITE at `off` uses: the one holding off, or the last segment at its exclusive
// end. -1 otherwise.
static inline int IbMsegSegmentForZeroLengthOffset(int nSeg, const uint64_t* segOff, uint64_t off) {
  if (nSeg < 1 || off < segOff[0] || off > segOff[nSeg]) return -1;
  return rcclIbSegIndexOf(segOff, nSeg, off);
}

// RCCL: host VMM can report base 0; subtracting then wraps the remaining length.
static inline size_t IbMsegBytesRemainingInSegment(uintptr_t segPtr, uintptr_t segBase, size_t segSize) {
  return (segBase == 0) ? segSize : segSize - (segPtr - segBase);
}

// RCCL: where the sender reads a registration's whole layout (rcclIbLayoutBlob*) with one RDMA read.
struct rcclIbSegLayoutRef {
  uint64_t addr;
  uint64_t gen; // with addr, the sender's cache key; a new registration gets a new value
  uint32_t nSegments;
  uint32_t ndevs;
  uint32_t rkeys[RCCL_IB_MSEG_MAX_DEVS];
};

#define RCCL_IB_LAYOUT_RKEYS_HEAD 4
#define RCCL_IB_LAYOUT_RKEYS_LINE 14

// RCCL: layout written just before one slot's CTS entries; inline holds the overlapped segments from the first.
// Each 64-byte line ends with the CTS idx, so the sender sees every line land, not only the CTS row's.
struct alignas(64) IbMsegSegLayout {
  uint32_t nSegments; // inline records; 0 when byRef
  uint32_t byRef;
  uint64_t segStart[RCCL_IB_CTS_INLINE_SEGS];
  uint32_t rkeyHead[RCCL_IB_LAYOUT_RKEYS_HEAD]; // inline rkeys [seg][dev] in row-major order, via rcclIbLayoutRkey
  uint64_t idx0;
  union {
    uint32_t rkeyMid[RCCL_IB_LAYOUT_RKEYS_LINE];
    struct rcclIbSegLayoutRef ref;
  };
  uint64_t idx1;
  uint32_t rkeyTail[RCCL_IB_LAYOUT_RKEYS_LINE];
  uint64_t idx2;
};
// RCCL: 4 inline segments x 8 devices, 64-byte aligned. The peer RDMA-writes this many bytes per receive.
static_assert(sizeof(struct IbMsegSegLayout) == 192, "side slot size is the RDMA write length");
static_assert(offsetof(struct IbMsegSegLayout, idx0) == 56 && offsetof(struct IbMsegSegLayout, idx1) == 120 &&
                  offsetof(struct IbMsegSegLayout, idx2) == 184,
              "each layout line ends with its idx stamp");
static_assert(sizeof(struct rcclIbSegLayoutRef) == sizeof(uint32_t) * RCCL_IB_LAYOUT_RKEYS_LINE,
              "the referenced form fills the middle line");
static_assert(RCCL_IB_LAYOUT_RKEYS_HEAD + 2 * RCCL_IB_LAYOUT_RKEYS_LINE ==
                  RCCL_IB_CTS_INLINE_SEGS * RCCL_IB_MSEG_MAX_DEVS,
              "the three rkey runs hold every inline rkey");

// RCCL: inline rkey of segment s on device dev, wherever its line holds it.
template <typename L>
static inline auto* rcclIbLayoutRkeyPtr(L* l, int s, int dev) {
  int k = s * RCCL_IB_MSEG_MAX_DEVS + dev;
  if (k < RCCL_IB_LAYOUT_RKEYS_HEAD) return &l->rkeyHead[k];
  k -= RCCL_IB_LAYOUT_RKEYS_HEAD;
  return k < RCCL_IB_LAYOUT_RKEYS_LINE ? &l->rkeyMid[k] : &l->rkeyTail[k - RCCL_IB_LAYOUT_RKEYS_LINE];
}
static inline uint32_t rcclIbLayoutRkey(const volatile struct IbMsegSegLayout* l, int s, int dev) {
  return *rcclIbLayoutRkeyPtr(l, s, dev);
}
static inline void rcclIbLayoutSetRkey(struct IbMsegSegLayout* l, int s, int dev, uint32_t rkey) {
  *rcclIbLayoutRkeyPtr(l, s, dev) = rkey;
}
static inline void rcclIbLayoutStamp(struct IbMsegSegLayout* l, uint64_t idx) { l->idx0 = l->idx1 = l->idx2 = idx; }
static inline bool rcclIbLayoutArrived(const volatile struct IbMsegSegLayout* l, uint64_t idx) {
  return l->idx0 == idx && l->idx1 == idx && l->idx2 == idx;
}

// ---- Per-segment MRs ----

// RCCL: a registration's segments, embedded in IbMsegMrHandle. nSegments 1 (or 0) is one MR.
struct rcclIbSegMr {
  int nSegments;
  // RCCL: walked handles only (NULL for a real fd), sized from the count. segMrs is [s * RCCL_IB_MSEG_MAX_DEVS + dev].
  uintptr_t* segStart; // VA start of each segment
  size_t* segLen;      // bytes per segment
  uint64_t* segOff;    // nSegments + 1 offsets from segStart[0]
  void** segMrs;
  // RCCL: receive side, nSegments > RCCL_IB_CTS_INLINE_SEGS only: the remote-readable layout a CTS references.
  void* layoutBlob;
  size_t layoutBytes;
  uint64_t layoutGen;
  struct ibv_mr* layoutMrs[RCCL_IB_MSEG_MAX_DEVS];
};

static inline struct ibv_mr* rcclIbSegMrAt(const struct rcclIbSegMr* m, int s, int dev) {
  return (struct ibv_mr*)m->segMrs[(size_t)s * RCCL_IB_MSEG_MAX_DEVS + dev];
}

// RCCL: segment arrays and offsets from the walked segments; the MRs are registered by the caller.
ncclResult_t rcclIbSegMrInit(struct rcclIbSegMr* m, int nSeg, void* const* segAddrs, const size_t* segLens);
// RCCL: frees the segment arrays; the MRs and the layout are already released.
void rcclIbSegMrFreeArrays(struct rcclIbSegMr* m);
// RCCL: the remote-readable layout a CTS references, registered on each device's PD.
ncclResult_t rcclIbSegMrPublishLayout(struct rcclIbSegMr* m, int ndevs, struct ibv_pd* const* pds);
// RCCL: releases the published layout's MRs and memory; returns the first deregistration error.
ncclResult_t rcclIbSegMrFreeLayout(struct rcclIbSegMr* m);
// RCCL: the segment MR covering [addr, addr+len) on dev, or NULL when the range crosses a segment boundary.
struct ibv_mr* rcclIbSegMrForRange(const struct rcclIbSegMr* m, uintptr_t addr, size_t len, int dev);
// RCCL: fills one receive's layout prefix from its registration. A receive inside one segment writes that
// segment's rkeys to shortRkeys[0..ndevs) and leaves the prefix empty; *multi is set when it spans several.
ncclResult_t rcclIbSegLayoutFill(const struct rcclIbSegMr* m, int ndevs, uint64_t addr, uint64_t size,
                                 struct IbMsegSegLayout* side, uint32_t* shortRkeys, bool* multi);

// ---- Sender segment tables ----

// RCCL: one side's segment table for a request. seg.off is the handle's, a read layout's, or offStore; reqOff is the
// request's offset from seg.base.
struct rcclIbSendSegTable {
  struct rcclIbSegTable seg;
  bool multi;
  uint64_t reqOff;
  const uint32_t* rkeys; // remote only, seg.nSeg entries: a read layout's or rkeyStore
  uint64_t offStore[RCCL_IB_CTS_INLINE_SEGS + 1];
  uint32_t rkeyStore[RCCL_IB_CTS_INLINE_SEGS];
};

// RCCL: one segment covering [base, base+size), with rkey for the remote side.
void rcclIbSegTableSingle(struct rcclIbSendSegTable* t, uint64_t base, uint64_t size, uint32_t rkey);
// RCCL: the local table from the send handle (NULL or single-segment gives one segment).
ncclResult_t rcclIbSegTableLocal(const struct rcclIbSegMr* m, uint64_t base, uint64_t size,
                                 struct rcclIbSendSegTable* t);
// RCCL: the remote table from an inline layout; the starts and rkeys are read once, so the WRs use what was checked.
ncclResult_t rcclIbSegTableInline(const volatile struct IbMsegSegLayout* side, uint64_t base, uint64_t size,
                                  int remDevIdx, struct rcclIbSendSegTable* t);

// ---- Referenced-layout cache ----

// RCCL: referenced layouts read from the receiver. One slot needs at most RCCL_IB_MSEG_MAX_RECVS at a time.
#define RCCL_IB_LAYOUT_CACHE_ENTRIES (2 * RCCL_IB_MSEG_MAX_RECVS)
enum rcclIbLayoutState : uint8_t { RCCL_IB_LAYOUT_EMPTY, RCCL_IB_LAYOUT_READING, RCCL_IB_LAYOUT_READY };
struct rcclIbLayoutCacheEntry {
  uint64_t addr;
  uint64_t gen;
  uint32_t nSegments;
  uint32_t ndevs;
  enum rcclIbLayoutState state;
  int dev; // device whose PD holds mr
  uint64_t lastUse;
  size_t cap;
  void* buf;
  struct ibv_mr* mr;
};
struct rcclIbLayoutCache {
  struct rcclIbLayoutCacheEntry e[RCCL_IB_LAYOUT_CACHE_ENTRIES];
  uint64_t clock;
};

// RCCL: the QP a layout read posts on, chosen by the plugin. stamp (may be NULL) gets ctx and the WR before posting.
struct rcclIbLayoutReadQp {
  struct ibv_qp* qp;
  struct ibv_pd* pd; // PD of devIndex, which holds the read buffer's MR
  int devIndex;
  int remDevIdx;
  void* ctx;
  void (*stamp)(void* ctx, struct ibv_send_wr* wr);
};
typedef ncclResult_t (*rcclIbLayoutReadQpFn)(void* arg, struct rcclIbLayoutReadQp* out);

struct rcclIbLayoutCacheEntry* rcclIbLayoutCacheFind(struct rcclIbLayoutCache* cache, uint64_t addr, uint64_t gen);
// RCCL: *ready once every referenced layout in sides[0..n) is read; starts the reads still needed. NULL sides and
// inline ones are skipped. pickQp runs at most once, when the first read is posted.
ncclResult_t rcclIbLayoutsReady(struct rcclIbLayoutCache* cache, const volatile struct IbMsegSegLayout* const* sides,
                                int n, rcclIbLayoutReadQpFn pickQp, void* arg, bool* ready);
// RCCL: a layout read completed. wrId is its cache entry.
ncclResult_t rcclIbLayoutReadDone(struct rcclIbLayoutCache* cache, uint64_t wrId);
ncclResult_t rcclIbLayoutCacheFree(struct rcclIbLayoutCache* cache);
// RCCL: the remote table from a referenced layout the sender already read; isend waits for that read.
ncclResult_t rcclIbSegTableReferenced(struct rcclIbLayoutCache* cache, const volatile struct IbMsegSegLayout* side,
                                      uint64_t base, uint64_t size, int remDevIdx, struct rcclIbSendSegTable* t);

#endif // RCCL_TRANSPORT_P2P_MULTISEG_MSEG_H_
