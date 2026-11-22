/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// RCCL: classic NET/IB work-request fill and CTS layout. IbMsegMultiSend calls rcclIbFillSendWrs;
// the other hooks are IbMsegIsend, IbMsegPostFifo, IbMsegIrecv and IbMsegIflush.

#include "p2p_multiseg_mseg.h"
#include "p2p_mseg.h"
#include "common_mseg.h"
#include "seg_wr_mseg.h"

#include <limits>

ibv_mr* IbMsegMrForRange(const struct IbMsegMrHandle* h, uintptr_t addr, size_t len, int devIndex) {
  if (h == NULL) return NULL;
  if (h->seg.nSegments <= 1) return h->mrs[devIndex];
  return rcclIbSegMrForRange(&h->seg, addr, len, devIndex);
}

// RCCL: this CTS entry's layout prefix, or NULL when the write did not include one.
static inline const volatile struct IbMsegSegLayout* rcclIbCtsRemoteLayout(const struct IbMsegSendComm* comm, int slot,
                                                                           int r) {
  const volatile struct IbMsegSendFifo* ctsRow = rcclIbSendCtsRow(comm, slot);
  if (!comm->useMultiSeg || !IbMsegCtsHasLayout(ctsRow[r].nreqs)) return NULL;
  int nreqs = (int)IbMsegCtsNreqs(ctsRow[0].nreqs);
  if (nreqs < 1 || nreqs > NCCL_NET_IB_MAX_RECVS || r >= nreqs) return NULL;
  return rcclIbCtsLayout(ctsRow, nreqs, r);
}

// RCCL: true when this CTS entry carries more than one segment inline or by reference.
static inline bool IbMsegCtsRemoteMultiSeg(const struct IbMsegSendComm* comm, int slot, int r) {
  const volatile struct IbMsegSegLayout* l = rcclIbCtsRemoteLayout(comm, slot, r);
  return l != NULL && (l->byRef || l->nSegments > 1);
}

// RCCL: remote table from the inline or referenced layout, else one segment from the CTS slot.
static ncclResult_t rcclIbBuildRemoteSegmentLayout(struct IbMsegSendComm* comm, int slot, int r, int remDevIdx,
                                                   struct rcclIbSendSegTable* t) {
  volatile struct IbMsegSendFifo* slots = rcclIbSendCtsRow(comm, slot);
  if (!IbMsegCtsRemoteMultiSeg(comm, slot, r)) {
    rcclIbSegTableSingle(t, slots[r].addr, slots[r].size, slots[r].rkeys[remDevIdx]);
    return ncclSuccess;
  }
  const volatile struct IbMsegSegLayout* side = rcclIbCtsRemoteLayout(comm, slot, r);
  if (side->byRef) {
    return rcclIbSegTableReferenced(comm->layoutCache, side, slots[r].addr, slots[r].size, remDevIdx, t);
  }
  return rcclIbSegTableInline(side, slots[r].addr, slots[r].size, remDevIdx, t);
}
static inline struct ibv_send_wr* rcclIbSegSendNextWr(struct IbMsegSendComm* comm, int w, struct ibv_sge** sge) {
  struct ibv_send_wr* wr = comm->wrs + w;
  *sge = comm->sges + w;
  memset(wr, 0, sizeof(struct ibv_send_wr));
  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->send_flags = 0;
  return wr;
}

// RCCL: one data slice from off, inside one local and one remote segment; *moved is its length. A zero len writes
// the zero-length WR instead. *laterSeg is set when either side's slice is past its first segment.
static ncclResult_t rcclIbAppendSegmentedRequestWrs(struct IbMsegSendComm* comm, const struct IbMsegRequest* req,
                                                    const struct rcclIbSendSegTable* local,
                                                    const struct rcclIbSendSegTable* remote, uint32_t off, uint32_t len,
                                                    int devIndex, uint64_t wrId, int* w, uint32_t* moved,
                                                    bool* laterSeg) {
  *moved = 0;
  if (*w >= NCCL_NET_IB_MAX_RECVS) {
    WARN("NET/IB: multi-segment send exceeded WR pool");
    return ncclInternalError;
  }
  if (len == 0) {
    // The remote address can sit in a later physical segment, so use that segment's rkey.
    struct ibv_sge* sge;
    struct ibv_send_wr* wr = rcclIbSegSendNextWr(comm, *w, &sge);
    wr->wr_id = wrId;
    wr->wr.rdma.remote_addr = remote->seg.base + remote->reqOff + off;
    if (remote->multi) {
      int s = IbMsegSegmentForZeroLengthOffset((int)remote->seg.nSeg, remote->seg.off, remote->reqOff + off);
      if (s < 0) return ncclInternalError;
      wr->wr.rdma.rkey = remote->rkeys[s];
      if (s > 0) *laterSeg = true;
    } else {
      wr->wr.rdma.rkey = remote->rkeys[0];
    }
    sge->addr = local->seg.base + local->reqOff + off;
    sge->length = 0;
    sge->lkey = req->send.lkeys[devIndex];
    wr->sg_list = sge;
    wr->num_sge = 0;
    (*w)++;
    return ncclSuccess;
  }

  struct rcclIbSegCursor cur;
  rcclIbSegCursorInit(&cur, &local->seg, &remote->seg, local->reqOff + off, remote->reqOff + off, len);
  struct rcclIbSegSlice slice;
  if (!rcclIbSegCursorNext(&cur, &local->seg, &remote->seg, &slice)) {
    WARN("NET/IB: multi-segment send split failed (data=%p off=%u len=%u)", req->send.data, off, len);
    return ncclInternalError;
  }
  uint32_t lkey = req->send.lkeys[devIndex];
  if (local->multi) {
    struct ibv_mr* mr = rcclIbSegMrAt(&req->send.mh->seg, slice.localSeg, devIndex);
    if (mr == NULL) {
      WARN("NET/IB: multi-segment send split failed (data=%p off=%u len=%u)", req->send.data, off, len);
      return ncclInternalError;
    }
    lkey = mr->lkey;
  }
  rcclIbSegFillWr(&local->seg, lkey, &remote->seg, remote->rkeys[slice.remoteSeg], &slice, IBV_WR_RDMA_WRITE,
                  comm->wrs + *w, comm->sges + *w);
  comm->wrs[*w].wr_id = wrId;
  (*w)++;
  *moved = (uint32_t)slice.len;
  if (slice.localSeg > 0 || slice.remoteSeg > 0) *laterSeg = true;
  TRACE(NCCL_NET, "NET/IB: segmented send slice off %u len %lu local seg %d remote seg %d", off,
        (unsigned long)slice.len, slice.localSeg, slice.remoteSeg);
  return ncclSuccess;
}

// RCCL: the sizes record for a multi-recv or non-BY_INDEX receiver, at comm->wrs[w].
static void rcclIbSegSendSizesWr(struct IbMsegSendComm* comm, int slot, int nreqs, int devIndex,
                                 struct ibv_send_wr* wr) {
  wr->wr.rdma.remote_addr = comm->remCmplsRecords.addr + slot * sizeof(struct IbMsegRequestCompletionRecord);
  wr->sg_list = &(comm->devs[devIndex].sge);
  wr->sg_list[0].addr = (uint64_t)(comm->remCmplsRecords.elems[slot]);
  wr->sg_list[0].length = nreqs * sizeof(int);
  wr->wr.rdma.rkey = comm->remCmplsRecords.rkeys[devIndex];
  wr->num_sge = 1;
}

// RCCL: final WR carrying the WRITE_WITH_IMM: this QP's flush flag (after a plain sizes write), a dedicated
// sizes/imm WR, or the last data WR.
static ncclResult_t rcclIbAppendSegmentedTailWr(struct IbMsegSendComm* comm, int slot, int nreqs, bool needSizesWr,
                                                bool extraImmWr, bool flagFlush, int devIndex, int qpIndex,
                                                uint64_t wrId, uint32_t immData, int* w, struct ibv_send_wr** lastWr) {
  constexpr int wrPool = (int)(sizeof(comm->wrs) / sizeof(comm->wrs[0]));
  if (flagFlush) {
    if (*w + (needSizesWr ? 1 : 0) >= wrPool) {
      WARN("NET/IB: multi-segment send exceeded WR pool");
      return ncclInternalError;
    }
    if (needSizesWr) {
      struct ibv_send_wr* sizesWr = comm->wrs + *w;
      memset(sizesWr, 0, sizeof(struct ibv_send_wr));
      sizesWr->wr_id = wrId;
      sizesWr->opcode = IBV_WR_RDMA_WRITE;
      rcclIbSegSendSizesWr(comm, slot, nreqs, devIndex, sizesWr);
      (*w)++;
    }
    *lastWr = comm->wrs + *w;
    memset(*lastWr, 0, sizeof(struct ibv_send_wr));
    (*lastWr)->sg_list = &comm->flagFlushSge;
    (*lastWr)->num_sge = 1;
    (*lastWr)->wr.rdma.rkey = comm->flagFlushRkey;
    (*lastWr)->wr.rdma.remote_addr = comm->flagFlushRemAddr + qpIndex * NCCL_GDR_FLUSH_FLAG_STRIDE;
    (*w)++;
  } else if (extraImmWr) {
    if (*w >= wrPool - 1) {
      WARN("NET/IB: multi-segment send exceeded WR pool");
      return ncclInternalError;
    }
    *lastWr = comm->wrs + *w;
    memset(*lastWr, 0, sizeof(struct ibv_send_wr));
    if (needSizesWr) {
      rcclIbSegSendSizesWr(comm, slot, nreqs, devIndex, *lastWr);
    } else {
      (*lastWr)->num_sge = 0;
    }
    (*w)++;
  } else {
    *lastWr = comm->wrs + (*w - 1); // reuse the last data WR
  }
  (*lastWr)->wr_id = wrId;
  (*lastWr)->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  (*lastWr)->imm_data = htobe32(immData);
  (*lastWr)->send_flags = IBV_SEND_SIGNALED;
  return ncclSuccess;
}

// RCCL: one QP's next post. Each receive moves one slice from off[r] toward end[r]; the first post also writes the
// zero-length receives. Only the post that finishes every receive writes the tail; otherwise its last write signals.
ncclResult_t rcclIbFillSendWrs(struct IbMsegSendComm* comm, int slot, int devIndex, int remDevIdx, int qpIndex,
                               uint32_t* off, const uint32_t* end, bool first, uint64_t wrId, uint32_t immData,
                               bool needSizesWr, bool extraImmWr, bool flagFlush, struct ibv_send_wr** lastWr,
                               bool* last) {
  struct IbMsegRequest** reqs = comm->sendReqs[slot];
  const int nreqs = reqs[0]->nreqs;
  int w = 0;
  bool laterSeg = false;
  *last = true;
  for (int r = 0; r < nreqs; r++) {
    if (off[r] == end[r] && !first) continue;
    struct rcclIbSendSegTable local, remote;
    NCCLCHECK(rcclIbSegTableLocal(reqs[r]->send.mh ? &reqs[r]->send.mh->seg : NULL, (uintptr_t)reqs[r]->send.data,
                                  reqs[r]->send.size, &local));
    NCCLCHECK(rcclIbBuildRemoteSegmentLayout(comm, slot, r, remDevIdx, &remote));
    uint32_t moved = 0;
    NCCLCHECK(rcclIbAppendSegmentedRequestWrs(comm, reqs[r], &local, &remote, off[r], end[r] - off[r], devIndex, wrId,
                                              &w, &moved, &laterSeg));
    off[r] += moved;
    if (off[r] != end[r]) *last = false;
  }
  rcclIbSegStatsAdd(&comm->segStats, w, laterSeg);
  if (*last) {
    NCCLCHECK(rcclIbAppendSegmentedTailWr(comm, slot, nreqs, needSizesWr, extraImmWr, flagFlush, devIndex, qpIndex,
                                          wrId, immData, &w, lastWr));
  } else {
    *lastWr = comm->wrs + (w - 1);
    (*lastWr)->send_flags = IBV_SEND_SIGNALED;
  }
  for (int k = 0; k < w - 1; k++) comm->wrs[k].next = comm->wrs + k + 1;
  comm->wrs[w - 1].next = NULL;
  return ncclSuccess;
}

// RCCL: true for the signaled slice of the slot still posting, which lets that QP post its next slice.
bool rcclIbSendSliceDone(struct IbMsegSendComm* comm, int slot, int devIndex, uint32_t qpNum) {
  struct rcclIbSendCursor* c = comm->sendCursor;
  if (c == NULL || !c->active || c->slot != slot) return false;
  for (int i = 0; i < c->nqps; i++) {
    // RCCL: qp_num is unique only per HCA, so match the completion's device too.
    if (c->qpState[i] == RCCL_IB_SEND_QP_INFLIGHT && c->qp[i]->devIndex == devIndex &&
        c->qp[i]->qp->qp_num == qpNum) {
      c->qpState[i] = RCCL_IB_SEND_QP_IDLE;
      return true;
    }
  }
  return false;
}

// RCCL: segmented builder when any request's local or remote buffer spans several segments.
bool rcclIbMultiSendNeedsSegmented(struct IbMsegSendComm* comm, int slot, int nreqs) {
  for (int r = 0; r < nreqs; r++) {
    struct IbMsegMrHandle* mh = comm->sendReqs[slot][r]->send.mh;
    if ((mh && mh->seg.nSegments > 1) || IbMsegCtsRemoteMultiSeg(comm, slot, r)) return true;
  }
  return false;
}

// RCCL: one CTS WR. The layout prefix is the bytes immediately before the CTS row, so idx stays last.
void rcclIbPrepareCtsWrite(struct IbMsegRecvComm* comm, struct IbMsegRequest* req, int slot, struct ibv_send_wr* wr) {
  struct IbMsegSendFifo* row = rcclIbRecvCtsRow(comm, slot);
  int nreqs = req->nreqs;
  uint64_t stride = comm->ctsSlotStride != 0 ? comm->ctsSlotStride
                                             : (uint64_t)NCCL_NET_IB_MAX_RECVS * sizeof(struct IbMsegSendFifo);
  uint64_t remote = comm->remCtsFifo.addr + (uint64_t)slot * stride;
  uint64_t local = (uint64_t)row;
  uint32_t length = (uint32_t)((size_t)nreqs * sizeof(struct IbMsegSendFifo));
  bool longForm = comm->ctsSlots != NULL && nreqs > 0 && IbMsegCtsHasLayout(row[0].nreqs);
  if (longForm) {
    uint64_t prefix = (uint64_t)nreqs * sizeof(struct IbMsegSegLayout);
    remote -= prefix;
    local -= prefix;
    length += (uint32_t)prefix;
    wr->send_flags &= ~IBV_SEND_INLINE;
  }
  wr->wr.rdma.remote_addr = remote;
  if (wr->sg_list != NULL) {
    wr->sg_list[0].addr = local;
    wr->sg_list[0].length = length;
  }
}

// RCCL: a NULL handle or per-device MR would be dereferenced for the CTS rkey.
ncclResult_t rcclIbIrecvCheckHandle(struct IbMsegRecvComm* comm, struct IbMsegMrHandle* mh, int i) {
  if (mh == NULL) {
    WARN("NET/IB: irecv[%d] has a NULL mhandle", i);
    return ncclInternalError;
  }
  for (int j = 0; j < comm->base.vProps.ndevs; j++) {
    if (mh->mrs[j] == NULL) {
      WARN("NET/IB: irecv[%d] missing MR for device %d", i, j);
      return ncclInternalError;
    }
  }
  return ncclSuccess;
}

ncclResult_t rcclIbIrecvFillLayout(struct IbMsegRecvComm* comm, int slot, int i,
                                      struct IbMsegMrHandle* mhandleWrapper, bool* anyMultiSeg) {
  // RCCL: a legacy-CTS comm has no layout prefix. The CTS idx remains the arrival flag.
  if (!comm->useMultiSeg || comm->ctsSlots == NULL) return ncclSuccess;
  struct IbMsegSendFifo* localElem = rcclIbRecvCtsRow(comm, slot);
  int nreqs = (int)IbMsegCtsNreqs(localElem[i].nreqs);
  if (nreqs < 1 || i >= nreqs) return ncclInternalError;
  struct IbMsegSegLayout* sideElem = rcclIbCtsLayout(localElem, nreqs, i);
  return rcclIbSegLayoutFill(&mhandleWrapper->seg, comm->base.vProps.ndevs, localElem[i].addr, localElem[i].size,
                             sideElem, localElem[i].rkeys, anyMultiSeg);
}

static void rcclIbLayoutReadStamp(void* qp, struct ibv_send_wr* wr) {
  IbMsegWqeLatMonStampSend((struct IbMsegQp*)qp, wr);
}

struct rcclIbLayoutReadArg {
  struct IbMsegSendComm* comm;
  int slot;
};

// RCCL: layout reads post on the QP of the slot's first request.
static ncclResult_t rcclIbLayoutReadQp(void* arg, struct rcclIbLayoutReadQp* out) {
  struct IbMsegSendComm* comm = ((struct rcclIbLayoutReadArg*)arg)->comm;
  int slot = ((struct rcclIbLayoutReadArg*)arg)->slot;
  IbMsegQp* qp = NULL;
  int qpIndex = -1;
  NCCLCHECK(IbMsegCommBaseGetQpForRequest(&comm->base, comm->sendReqs[slot][0]->id, 0, &qp, &qpIndex));
  out->qp = qp->qp;
  out->pd = comm->devs[qp->devIndex].base.pd;
  out->devIndex = qp->devIndex;
  out->remDevIdx = qp->remDevIdx;
  out->ctx = qp;
  out->stamp = IbMsegWqeLatEnabled ? rcclIbLayoutReadStamp : NULL;
  return ncclSuccess;
}

// RCCL: true once every layout the slot references is read; starts the reads it still needs.
ncclResult_t rcclIbSendLayoutsReady(struct IbMsegSendComm* comm, int slot, bool* ready) {
  *ready = true;
  if (comm->layoutCache == NULL) return ncclSuccess; // a legacy-CTS comm carries no layouts
  int nreqs = (int)IbMsegCtsNreqs(rcclIbSendCtsRow(comm, slot)[0].nreqs);
  if (nreqs > NCCL_NET_IB_MAX_RECVS) nreqs = NCCL_NET_IB_MAX_RECVS;
  const volatile struct IbMsegSegLayout* sides[NCCL_NET_IB_MAX_RECVS];
  for (int r = 0; r < nreqs; r++) sides[r] = rcclIbCtsRemoteLayout(comm, slot, r);
  struct rcclIbLayoutReadArg arg = {comm, slot};
  return rcclIbLayoutsReady(comm->layoutCache, sides, nreqs, rcclIbLayoutReadQp, &arg, ready);
}

ncclResult_t rcclIbSendLayoutReadDone(struct IbMsegSendComm* comm, uint64_t wrId) {
  if (comm->layoutCache == NULL) {
    WARN("NET/IB: unexpected segment layout read completion on a legacy-CTS comm (wr_id=%lu)", wrId);
    return ncclInternalError;
  }
  return rcclIbLayoutReadDone(comm->layoutCache, wrId);
}

ncclResult_t rcclIbSendLayoutCacheFree(struct IbMsegSendComm* comm) {
  if (comm->layoutCache == NULL) return ncclSuccess;
  return rcclIbLayoutCacheFree(comm->layoutCache);
}

// RCCL: the layout bit rides in the 16-bit nreqs above any receive count.
static_assert(NCCL_IB_CTS_NREQS_LAYOUT <= std::numeric_limits<decltype(IbMsegSendFifo::nreqs)>::max() &&
                  NCCL_IB_CTS_NREQS_LAYOUT > NCCL_NET_IB_MAX_RECVS,
              "the CTS layout bit must fit nreqs and clear every receive count");

// RCCL: set the layout bit so the sender reads the layout prefix, and stamp each layout line with idx.
void rcclIbCtsFlagLayout(struct IbMsegSendFifo* localElem, int n, uint64_t idx) {
  for (int i = 0; i < n; i++) {
    localElem[i].nreqs |= NCCL_IB_CTS_NREQS_LAYOUT;
    rcclIbLayoutStamp(rcclIbCtsLayout(localElem, n, i), idx);
  }
}

bool rcclIbCtsLayoutsArrived(const volatile struct IbMsegSendFifo* ctsRow, int nreqs, uint64_t idx) {
  for (int r = 0; r < nreqs; r++) {
    if (!rcclIbLayoutArrived(rcclIbCtsLayout(ctsRow, nreqs, r), idx)) return false;
  }
  return true;
}

ncclResult_t rcclIbSendCommSegStats(void* sendComm, struct rcclIbSegStats* out) {
  if (sendComm == NULL || out == NULL) return ncclInvalidArgument;
  rcclIbSegStatsRead(&((struct IbMsegSendComm*)sendComm)->segStats, out);
  return ncclSuccess;
}
