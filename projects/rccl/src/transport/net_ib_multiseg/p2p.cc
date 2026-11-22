/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "p2p_mseg.h"
#include "common_mseg.h"
#include "compiler.h"
#include "p2p_resiliency_mseg.h"
#include "p2p_multiseg_mseg.h" // RCCL: multi-segment hooks

extern int64_t ncclParamIbArThreshold();
int64_t IbMsegArThreshold = 8192;

// By default, use IbMsegRequestMatchingScheme::BY_INDEX matching scheme.
extern int64_t ncclParamIbReceiverSideMatchingScheme();
extern int64_t rcclParamIbSplitDataThreshold();
extern int64_t rcclParamIbGdrFlushGpuMemNoRelaxedOrdering();

const char* IbMsegReqTypeStr[] = {"Unused", "Send", "Recv", "Flush", "IPut", "IGet", "Failed"};

ncclResult_t IbMsegGetRequest(struct IbMsegNetCommBase* base, struct IbMsegRequest** req) {
  for (int i = 0; i < NET_IB_MAX_REQUESTS; i++) {
    struct IbMsegRequest* r = base->reqs + i;
    if (r->type == NCCL_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->sock = NULL;
      memset(r->devBases, 0, sizeof(r->devBases));
      memset(r->events, 0, sizeof(r->events));
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/IB : unable to allocate requests");
  *req = NULL;
  return ncclInternalError;
}

ncclResult_t IbMsegFreeRequest(struct IbMsegRequest* r) {
  r->type = NCCL_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}

void IbMsegAddEvent(struct IbMsegRequest* req, int devIndex) {
  struct IbMsegNetCommDevBase* base = IbMsegGetNetCommDevBase(req->base, devIndex);
  req->events[devIndex]++;
  req->devBases[devIndex] = base;
}

#ifdef ENABLE_TRACE
static ncclResult_t IbMsegPrintWr(struct ibv_send_wr* wr, char* wrStr) {
  if (wr == NULL) {
    sprintf(wrStr, "wr=NULL");
    return ncclSuccess;
  }
  const char* opcodeStr = ibvWrOpcodeStr(wr->opcode);
  switch (wr->opcode) {
  case IBV_WR_RDMA_WRITE:
    sprintf(wrStr,
            "wr=%p, wr_id=%ld, opcode=%s, num_sge=%d, sge[0].length=%" PRIu32 ", sge[0].addr=0x%016" PRIx64
            ", rdma.remote_addr=0x%016" PRIx64 ", rdma.rkey=0x%x",
            wr, wr->wr_id, opcodeStr, wr->num_sge, (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->length : 0,
            (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->addr : 0, wr->wr.rdma.remote_addr, wr->wr.rdma.rkey);
    break;
  case IBV_WR_RDMA_WRITE_WITH_IMM:
    sprintf(wrStr,
            "wr=%p, wr_id=%ld, opcode=%s, num_sge=%d, sge[0].length=%" PRIu32 ", sge[0].addr=0x%016" PRIx64
            ", rdma.remote_addr=0x%016" PRIx64 ",  rdma.rkey=0x%x, imm_data=0x%x",
            wr, wr->wr_id, opcodeStr, wr->num_sge, (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->length : 0,
            (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->addr : 0, wr->wr.rdma.remote_addr, wr->wr.rdma.rkey,
            wr->imm_data);
    break;
  default:
    WARN("NET/IB: %s: No format specified for opcode=%d", __func__, wr->opcode);
    return ncclInternalError;
  }
  return ncclSuccess;
}
#endif // ENABLE_TRACE

// The alignment for IB writes that is required to make LL and LL128 protocols work
#define IB_WRITE_CHUNK_ALIGNMENT 128
// RCCL: the segmented builder in p2p_multiseg.cc must split chunks the same way.
static_assert(IB_WRITE_CHUNK_ALIGNMENT == rcclIbWriteChunkAlignment, "segmented chunk alignment differs");

// RCCL: a slice post holds one more event per request on its device until its signaled write completes.
static void rcclIbSendHoldSlice(struct IbMsegRequest** reqs, int nreqs, int devIndex, int delta) {
  for (int r = 0; r < nreqs; r++) reqs[r]->events[devIndex] += delta;
}

// RCCL: posts comm->wrs on qp. A failed post releases the slice hold, drops the events of the QPs that
// unposted(prefixPosted, out) lists as never completing, and marks the comm fatal.
template <typename UnpostedFn>
static ncclResult_t rcclIbPostSendWrs(struct IbMsegSendComm* comm, IbMsegQp* qp, struct IbMsegRequest** reqs,
                                      int nreqs, bool last, UnpostedFn unposted) {
  if (!last) rcclIbSendHoldSlice(reqs, nreqs, qp->devIndex, 1);
  if (IbMsegWqeLatEnabled) IbMsegWqeLatMonStampSend(qp, comm->wrs);
  struct ibv_send_wr* bad_wr = NULL;
  ncclResult_t ret = wrap_ibv_post_send(qp->qp, comm->wrs, &bad_wr);
  if (ret == ncclSuccess) return ncclSuccess;
  if (!last) rcclIbSendHoldSlice(reqs, nreqs, qp->devIndex, -1); // RCCL: the signaled slice write did not post.
  int posted = 0;
  for (struct ibv_send_wr* p = comm->wrs; p != NULL && p != bad_wr; p = p->next) posted++;
  IbMsegQp* lost[NCCL_IB_MAX_QPS];
  const int nLost = unposted(posted > 0, lost);
  for (int k = 0; k < nLost; k++) {
    for (int r = 0; r < nreqs; r++) {
      if (reqs[r]->events[lost[k]->devIndex] > 0) reqs[r]->events[lost[k]->devIndex]--;
    }
  }
  // Mark connection as fatal. DO NOT free requests here - the caller
  // (IbMsegIsend) owns the requests and will handle cleanup.
  IbMsegStatsFatalError(&comm->base.stats);
  return ret;
}

// Keep this hot loop on a 64-byte boundary.
__attribute__((aligned(64)))
ncclResult_t IbMsegMultiSend(struct IbMsegSendComm* comm, int slot) {
  struct IbMsegRequest** reqs = comm->sendReqs[slot];
  volatile struct IbMsegSendFifo* slots = rcclIbSendCtsRow(comm, slot);
  int nreqs = reqs[0]->nreqs; // RCCL: the count IbMsegIsend validated, not a re-read of the peer-written CTS row.
  if (nreqs < 1 || nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  // RCCL: rcclIbFillSendWrs builds every chain, one WR when each side is one segment.
  const bool multi = rcclIbMultiSendNeedsSegmented(comm, slot, nreqs);

  TRACE(NCCL_NET, "NET/IB: %s: Posting a send request (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d)", __func__, reqs[0],
        reqs[0]->base, reqs[0]->id, slot, nreqs);

  bool flagFlush = comm->flagFlush && slots[0].flagFlush;
  int nqps = 0;
  NCCLCHECK(IbMsegCommBaseGetNqpsPerRequest(&comm->base, &nqps));
#ifdef NCCL_ENABLE_NET_PROFILING
  if (nqps > MAX_QPS_PER_REQ) {
    WARN("NET/IB: QP count %d exceeds maximum profiler event handle capacity %d", nqps, MAX_QPS_PER_REQ);
    return ncclInternalError;
  }
#endif
  if (nqps > NCCL_IB_MAX_QPS) {
    WARN("NET/IB: QP count %d exceeds maximum QP capacity %d", nqps, NCCL_IB_MAX_QPS);
    return ncclInternalError;
  }
  // Validate QPs before posting sends.
  IbMsegQp* qps[NCCL_IB_MAX_QPS];
  int qpIndexes[NCCL_IB_MAX_QPS];
  int qpsPerDev[NCCL_IB_MAX_DEVS_PER_NIC] = {};
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(IbMsegCommBaseGetQpForRequest(&comm->base, reqs[0]->id, i, &qps[i], &qpIndexes[i]));
    qpsPerDev[comm->base.qps[qpIndexes[i]].devIndex]++;
  }
  uint64_t wr_id = 0ULL;
  for (int r = 0; r < nreqs; r++) {
    wr_id += (uint64_t)(slot & 0xff) << (r * 8);
#ifdef NCCL_ENABLE_NET_PROFILING
    reqs[r]->pInfo[0].nEventHandles = 0;
#endif
  }

  // For ID-based matching scheme, immData carries the request ID.
  // For index-based matching scheme, immData carries the send size:
  // - nreqs == 1
  //      It's the send size.
  // - nreqs > 1
  //      Send size is still sent but receiver ignores it since the sizes are
  //      written to directly to remote completion records array
  uint32_t immData = comm->base.recvMatchingScheme == BY_ID ? (uint32_t)reqs[0]->id : reqs[0]->send.size;
  // A split send's immediate rides the last slice, so a non-BY_INDEX receiver needs the size record.
  const bool needSizesWr =
    nreqs > 1 || (multi && comm->base.recvMatchingScheme != BY_INDEX && reqs[0]->send.size > 0);
  // When Adaptive Routing is enabled, send the bulk of the data first as an RDMA Write.
  const bool arExtra =
    !(comm->base.remOooRq && comm->base.localOooRq) && comm->ar && reqs[0]->send.size > IbMsegArThreshold;
  const bool extraImmWr = (nreqs > 1) || needSizesWr || arExtra;
  if (flagFlush) {
    // Payload (and sizes) go out as plain RDMA writes followed by the flag
    // write, which carries the immediate data. Each QP writes its own flag.
    comm->flagFlushSrc[slot] = reqs[0]->id + 1;
    comm->flagFlushSge.addr = (uint64_t)&comm->flagFlushSrc[slot];
  }

  uint32_t sendOffsets[NCCL_NET_IB_MAX_RECVS] = {0};
  // RCCL: a QP whose chunk crosses a segment boundary keeps its cursor; rcclIbSendProgress posts the rest.
  // Without a cursor, a single-segment send keeps each QP's offsets on the stack.
  if (multi && comm->sendCursor == NULL) NCCLCHECK(ncclCalloc(&comm->sendCursor, 1));
  struct rcclIbSendCursor* cursor = comm->sendCursor;
  uint32_t qpOff[NCCL_NET_IB_MAX_RECVS];
  uint32_t qpEnd[NCCL_NET_IB_MAX_RECVS];
  if (cursor != NULL) {
    cursor->slot = slot;
    cursor->active = 0;
    cursor->nqps = nqps;
  }
  bool slicesLeft = false;
  // For small messages (< splitDataThreshold) with multiple QPs, avoid splitting:
  // send the entire payload through one QP selected via round-robin, post zero-length WRs on the rest.
  int64_t splitDataThreshold = rcclParamIbSplitDataThreshold();
  uint8_t* weights = reqs[0]->send.weights;
  for (int i = 0; i < nqps; i++) {
    IbMsegQp* qp = qps[i];
    int qpIndex = qpIndexes[i];

    TRACE(NCCL_NET,
          "NET/IB: %s: Posting send (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld) on QP (qp_num=%u, "
          "devIndex=%d, qpIndex=%d)",
          __func__, reqs[0], reqs[0]->base, reqs[0]->id, slot, nreqs, wr_id, qp->qp->qp_num, qp->devIndex, qpIndex);

    int devIndex = qp->devIndex;
    int origDevIndex = comm->base.qps[qpIndex].devIndex;
    uint32_t chunkLen[NCCL_NET_IB_MAX_RECVS];
    uint32_t* off = cursor != NULL ? cursor->off[i] : qpOff;
    uint32_t* end = cursor != NULL ? cursor->end[i] : qpEnd;
    for (int r = 0; r < nreqs; r++) {
      // Compute per-QP chunk size: for small messages, only the active QP carries data.
      // i==0 always corresponds to the active QP because IbMsegCommBaseGetQpForRequest
      // rotates by req->id, so qpIndex=(id+i)%nqps; at i=0 that equals id%nqps.
      // Larger messages are chunked proportionally across QPs based on per-device weights
      // (derived from device speeds). Each device's share is (size * weight / 100), further
      // split among the actual QPs on that device. When all devices have the same speed,
      // weights are equal and it reduces to the equal-split formula.
      uint32_t chunkSize;
      if (reqs[r]->send.size < splitDataThreshold) {
        chunkSize = (i == 0) ? reqs[r]->send.size : 0;
      } else {
        chunkSize = DIVUP(DIVUP((uint64_t)reqs[r]->send.size * weights[origDevIndex], 100 * qpsPerDev[origDevIndex]),
                          IB_WRITE_CHUNK_ALIGNMENT) *
                    IB_WRITE_CHUNK_ALIGNMENT;
      }
      // Check the data left to send. If the send is too small, it might be
      // that on the current QP there is no data left to be sent.
      chunkLen[r] = std::min<uint32_t>(reqs[r]->send.size - sendOffsets[r], chunkSize);
      off[r] = sendOffsets[r];
      end[r] = sendOffsets[r] + chunkLen[r];
    }
    if (cursor != NULL) {
      cursor->qp[i] = qp;
      cursor->qpIndex[i] = qpIndex;
      cursor->qpState[i] = RCCL_IB_SEND_QP_DONE;
    }

    // Selective retransmission: skip posting if already delivered. Offsets still advance.
    const bool delivered = comm->base.resiliency && reqs[0]->send.sentData[qpIndex] == true;
    struct ibv_send_wr* lastWr = NULL;
    if (!delivered) {
      bool last = true;
      NCCLCHECK(rcclIbFillSendWrs(comm, slot, devIndex, qp->remDevIdx, qpIndex, off, end, true, wr_id, immData,
                                  needSizesWr, extraImmWr, flagFlush, &lastWr, &last));
#ifdef NCCL_ENABLE_NET_PROFILING
      // QP profiling loop
      for (int r = 0; r < nreqs; r++) {
        // Store the qpIndex for this request
        int nEventHandles = reqs[r]->pInfo[0].nEventHandles;
        reqs[r]->pInfo[0].qpIndex[nEventHandles] = qpIndex;
        // Store info for profiler
        int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
        reqs[r]->pInfo[0].data.type = ncclProfileQp;
        reqs[r]->pInfo[0].data.qp.device = devIndex;
        reqs[r]->pInfo[0].data.qp.wr_id = multi ? lastWr->wr_id : comm->wrs[r].wr_id;
        reqs[r]->pInfo[0].data.qp.opcode = multi ? lastWr->opcode : comm->wrs[r].opcode;
        reqs[r]->pInfo[0].data.qp.qpNum = qp->qp->qp_num;
        reqs[r]->pInfo[0].data.qp.length = multi ? chunkLen[r] : comm->sges[r].length;
        void* pHandle = reqs[r]->pInfo[0].pHandle;
        NCCLCHECK(IbMsegProfilerFunction(&reqs[r]->pInfo[0].qpEventHandles[nEventHandles], ncclProfilerNetEventStart,
                                       pHandle, pluginId, &reqs[r]->pInfo[0].data));
        reqs[r]->pInfo[0].nEventHandles++;
      }
#endif
#ifdef ENABLE_TRACE
      for (int r = 0; r < nreqs; r++) {
        TRACE(NCCL_NET,
              "NET/IB: %s: Posting send work request on QP (qpn=%u, devIndex=%d, qpIndex=%d) (slot=%d, req[r=%d]=%p)",
              __func__, qp->qp->qp_num, qp->devIndex, qpIndex, slot, r, reqs[r]);
      }
      int wrIdx = 0;
      char wrStr[1024];
      struct ibv_send_wr* currWr = comm->wrs;
      while (currWr != NULL) {
        NCCLCHECK(IbMsegPrintWr(currWr, wrStr));
        TRACE(NCCL_NET, "NET/IB: %s: slot=%d, wrIdx[%d], %s", __func__, slot, wrIdx, wrStr);
        wrIdx++;
        currWr = currWr->next;
      }
#endif // ENABLE_TRACE
      // RCCL: segmented sends drop events for QPs that will not complete; a posted prefix keeps its QP.
      NCCLCHECK(rcclIbPostSendWrs(comm, qp, reqs, nreqs, last, [&](bool prefixPosted, IbMsegQp** lost) {
        int n = 0;
        if (!multi) return n;
        for (int k = prefixPosted ? i + 1 : i; k < nqps; k++) {
          if (!(comm->base.resiliency && reqs[0]->send.sentData[qpIndexes[k]])) lost[n++] = qps[k];
        }
        return n;
      }));
      if (!last) {
        cursor->qpState[i] = RCCL_IB_SEND_QP_INFLIGHT;
        slicesLeft = true;
      }
    }

    // Update the send offset for the next QP by the chunk this QP carried.
    for (int r = 0; r < nreqs; r++) {
      sendOffsets[r] = std::min<uint32_t>(sendOffsets[r] + chunkLen[r], reqs[r]->send.size);
      if (!delivered) reqs[r]->send.sentData[qpIndex] = true;
      if (!multi && !delivered) {
        TRACE(NCCL_NET,
              "NET/IB: %s: Send request (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, reqIdx=%d, wr_id=%ld) posted %d "
              "bytes on QP index %d (devIndex=%d, qp_num=%u), total posted %d/%d bytes",
              __func__, reqs[r], reqs[0]->base, reqs[r]->id, slot, nreqs, r, comm->wrs[r].wr_id, chunkLen[r], qpIndex,
              devIndex, qp->qp->qp_num, sendOffsets[r], reqs[r]->send.size);
        // TODO: add ibv_dev_index to remDevs struct to enable full TRACE log with remote device info
        TRACE(NCCL_VERBS,
              "Posted send wr_id=%lu, wr_indx=%d, qp_num=%d, src_nic=%d, opcode=%d, send_flags=%d, imm_data=%d, "
              "remote_addr=%lx, rkey=%x, length=%d, lkey=%x",
              comm->wrs[r].wr_id, r, qp->qp->qp_num, comm->devs[qp->devIndex].base.ibDevN, comm->wrs[r].opcode,
              comm->wrs[r].send_flags, comm->wrs[r].imm_data, comm->wrs[r].wr.rdma.remote_addr,
              comm->wrs[r].wr.rdma.rkey, comm->wrs[r].sg_list ? comm->wrs[r].sg_list->length : 0,
              comm->wrs[r].sg_list ? comm->wrs[r].sg_list->lkey : 0);
      }
    }
  }
  if (slicesLeft) {
    cursor->active = 1;
    cursor->wrId = wr_id;
    cursor->immData = immData;
    cursor->needSizesWr = needSizesWr;
    cursor->extraImmWr = extraImmWr;
    cursor->flagFlush = flagFlush;
  }

  TRACE(NCCL_NET, "NET/IB: %s: Send request posted (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld)", __func__,
        reqs[0], reqs[0]->base, reqs[0]->id, slot, nreqs, wr_id);

  return ncclSuccess;
}

// RCCL: post the next slice on each QP whose last slice completed. The slot is done once every QP posted its tail;
// flagFlushSge still holds this slot's flag because no other slot posts in between.
static ncclResult_t rcclIbSendResume(struct IbMsegSendComm* comm) {
  struct rcclIbSendCursor* c = comm->sendCursor;
  struct IbMsegRequest** reqs = comm->sendReqs[c->slot];
  const int nreqs = reqs[0]->nreqs;
  bool done = true;
  for (int i = 0; i < c->nqps; i++) {
    if (c->qpState[i] == RCCL_IB_SEND_QP_IDLE) {
      IbMsegQp* qp = c->qp[i];
      bool last = true;
      struct ibv_send_wr* lastWr = NULL;
      NCCLCHECK(rcclIbFillSendWrs(comm, c->slot, qp->devIndex, qp->remDevIdx, c->qpIndex[i], c->off[i], c->end[i],
                                  false, c->wrId, c->immData, c->needSizesWr, c->extraImmWr, c->flagFlush, &lastWr,
                                  &last));
      // RCCL: this QP and the idle ones after it do not post their next slice this pass.
      NCCLCHECK(rcclIbPostSendWrs(comm, qp, reqs, nreqs, last, [&](bool prefixPosted, IbMsegQp** lost) {
        int n = 0;
        for (int k = prefixPosted ? i + 1 : i; k < c->nqps; k++) {
          if (c->qpState[k] == RCCL_IB_SEND_QP_IDLE) lost[n++] = c->qp[k];
        }
        return n;
      }));
      c->qpState[i] = last ? RCCL_IB_SEND_QP_DONE : RCCL_IB_SEND_QP_INFLIGHT;
    }
    if (c->qpState[i] != RCCL_IB_SEND_QP_DONE) done = false;
  }
  if (done) c->active = 0;
  return ncclSuccess;
}

ncclResult_t rcclIbSendProgress(struct IbMsegSendComm* comm) {
  struct rcclIbSendCursor* c = comm->sendCursor;
  if (c == NULL) return ncclSuccess;
  for (;;) {
    if (c->active) {
      NCCLCHECK(rcclIbSendResume(comm));
      if (c->active) return ncclSuccess;
    }
    if (c->queued == 0) return ncclSuccess;
    const int next = (int)((comm->base.fifoHead - c->queued) % NET_IB_MAX_REQUESTS);
    bool ready = true;
    NCCLCHECK(rcclIbSendLayoutsReady(comm, next, &ready));
    if (!ready) return ncclSuccess;
    c->queued--;
    NCCLCHECK(IbMsegMultiSend(comm, next));
  }
}

// Keep this hot loop on a 64-byte boundary.
__attribute__((aligned(64)))
ncclResult_t IbMsegIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle,
                         void** request) {
  struct IbMsegSendComm* comm = (struct IbMsegSendComm*)sendComm;
  ncclResult_t ret = ncclSuccess;
  if (comm->base.ready == 0) {
    WARN("NET/IB: IbMsegIsend() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  struct IbMsegMrHandle* mhandleWrapper = (struct IbMsegMrHandle*)mhandle;

  // Wait for the receiver to have posted the corresponding receive
  int nreqs = 0;
  volatile struct IbMsegSendFifo* slots;

  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  struct IbMsegRequest** reqs = comm->sendReqs[slot];
  slots = rcclIbSendCtsRow(comm, slot);
  uint64_t idx = comm->base.fifoHead + 1;
  struct IbMsegRequest* req = NULL;

  NCCLCHECKGOTO(IbMsegStatsCheckFatalCount(&comm->base.stats, __func__), ret, isendFail);

  if (slots[0].idx != idx) {
    *request = NULL;
    return ncclSuccess;
  }
  // Order the nreqs load after the idx load above; on weakly ordered CPUs it could otherwise return a stale value.
  std::atomic_thread_fence(std::memory_order_acquire);
  nreqs = IbMsegCtsNreqs(slots[0].nreqs); // RCCL: ignore the layout bit in the count.
  if (nreqs > NCCL_NET_IB_MAX_RECVS) {
    WARN("NET/IB : peer posted nreqs %d > %d", nreqs, NCCL_NET_IB_MAX_RECVS);
    ret = ncclInternalError;
    goto isendFail;
  }
  // Wait until all data has arrived
  for (int r = 1; r < nreqs; r++)
    while (slots[r].idx != idx);
  // RCCL: the layout lines sit outside the CTS rows' cache lines, so each carries its own idx; retry until all land.
  if (comm->useMultiSeg && IbMsegCtsHasLayout(slots[0].nreqs) && !rcclIbCtsLayoutsArrived(slots, nreqs, idx)) {
    *request = NULL;
    return ncclSuccess;
  }
  std::atomic_thread_fence(std::memory_order_seq_cst); // order the idx load against the layout and CTS fields
  for (int r = 0; r < nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag) continue;

    if (size > slots[r].size) size = slots[r].size;
    // Sanity checks
    if (slots[r].addr == 0 || slots[r].rkeys[0] == 0) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union ncclSocketAddress addr;
      ncclSocketGetAddr(&comm->base.sock, &addr);
      WARN("NET/IB : req %d/%d tag %x peer %s posted incorrect receive info: size %ld addr %lx rkeys[0]=%x", r, nreqs,
           tag, ncclSocketToString(&addr, line, sizeof(line)), slots[r].size, slots[r].addr, slots[r].rkeys[0]);
      return ncclInternalError;
    }

    int nqps = 0;
    NCCLCHECKGOTO(IbMsegCommBaseGetNqpsPerRequest(&comm->base, &nqps), ret, isendFail);
    if (nqps > NCCL_IB_MAX_QPS) {
      WARN("NET/IB: QP count %d exceeds maximum QP capacity %d", nqps, NCCL_IB_MAX_QPS);
      return ncclInternalError;
    }
    // Validate QPs before allocating the request.
    int qpDevIndexes[NCCL_IB_MAX_QPS];
    for (int i = 0; i < nqps; i++) {
      int qpIndex = -1;
      IbMsegQp* qp = NULL;
      NCCLCHECKGOTO(IbMsegCommBaseGetQpForRequest(&comm->base, comm->base.fifoHead, i, &qp, &qpIndex), ret, isendFail);
      qpDevIndexes[i] = qp->devIndex;
    }

    NCCLCHECKGOTO(IbMsegGetRequest(&comm->base, &req), ret, isendFail);
    req->id = comm->base.fifoHead;
    req->type = NCCL_NET_IB_REQ_SEND;
    req->sock = &comm->base.sock;
    req->base = &comm->base;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    if (comm->base.resiliency) {
      memset(req->send.sentData, 0, sizeof(req->send.sentData));
    }
    if (r == 0) {
      memcpy(req->send.weights, comm->base.weights, sizeof(req->send.weights));
    }
#ifdef NCCL_ENABLE_NET_PROFILING
    req->pInfo[0].pHandle = phandle;
#endif

    // Populate events
    for (int i = 0; i < nqps; i++) {
      IbMsegAddEvent(req, qpDevIndexes[i]);
    }

    req->send.mh = mhandleWrapper; // RCCL: segmented send reads per-segment lkeys from this handle.
    // Store all lkeys
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
    }

    // In case the sender will write the size of the send directly to the
    // receiver's memory, prepare the source buffer which will hold the sizes
    // and be sent to the receiver.
    comm->remCmplsRecords.elems[slot][r] = req->send.size;

    TRACE(NCCL_NET,
          "NET/IB: %s: Send request created (req=%p, comm=%p, id=%ld, slot=%d, reqIdx=%d, nreqs=%d, tag=%x, size=%ld, "
          "data=0x%016" PRIx64 ", mhandle=%p, size=%ld)",
          __func__, req, req->base, req->id, slot, r, nreqs, tag, size, (uint64_t)data, mhandle, size);

    *request = reqs[r] = req;

    comm->sendReqsCnt[slot]++;
    // If this is a multi-recv, send only when all requests have matched.
    if (comm->sendReqsCnt[slot] < nreqs) return ncclSuccess;

    TIME_START(0);
    // RCCL: behind a slot still posting slices, or until its referenced layouts are read, wait;
    // rcclIbSendProgress posts queued slots in fifo order.
    bool layoutsReady = true;
    struct rcclIbSendCursor* cursor = comm->sendCursor;
    const bool idle = cursor == NULL || (!cursor->active && cursor->queued == 0);
    if (idle) {
      NCCLCHECKGOTO(rcclIbSendLayoutsReady(comm, slot, &layoutsReady), ret, isendFail);
    }
    if (!idle || !layoutsReady) {
      cursor->queued++; // unread layouts imply useMultiSeg, which allocates the cursor
    } else {
      // RCCL: keep the post error in ret; isendFail returns it when it frees the request.
      NCCLCHECKGOTO(IbMsegMultiSend(comm, slot), ret, isendFail);
    }

    comm->base.fifoHead++;
    TIME_STOP(0);
    return ncclSuccess;
  }
  *request = NULL;
  return ncclSuccess;

isendFail:
  IbMsegStatsFatalError(&comm->base.stats);
  if (req) {
    // If events were added (IBV ops posted), we can't free immediately -
    // completions may still arrive. Mark as failed and return it so
    // caller can call Test to drain completions.
    if (IbMsegRequestHasEvents(req)) {
      req->type = NCCL_NET_IB_REQ_FAILED;
      *request = req;
      // Return success so caller proceeds to call Test() which will drain
      // completions. The fatal error is recorded in stats and will be
      // caught on the next operation.
      return ncclSuccess;
    } else {
      // RCCL: the segmented builder can drop every event; unlink the slot entry before freeing.
      for (int k = 0; k < nreqs; k++) {
        if (reqs[k] == req) {
          reqs[k] = NULL;
          comm->sendReqsCnt[slot]--;
        }
      }
      IbMsegFreeRequest(req);
      *request = NULL;
    }
  } else {
    *request = NULL;
  }
  return ret;
}

// Keep this hot loop on a 64-byte boundary.
__attribute__((aligned(64)))
ncclResult_t IbMsegPostFifo(struct IbMsegRecvComm* comm, struct IbMsegRequest* req, int slot) {
  IbMsegQp* ctsQp = NULL;
  NCCLCHECK(IbMsegRecvCommGetQpForCts(comm, req->id, &ctsQp));

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));

  // Lookup the correct rkey
  wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].rkey;

  // Populating the correct gather information based on the device and user
  // provided information
  struct IbMsegSendFifo* localElem = rcclIbRecvCtsRow(comm, slot);
  wr.sg_list = &(comm->devs[ctsQp->devIndex].sge);
  wr.sg_list[0].addr = (uint64_t)localElem;
  wr.sg_list[0].length = req->nreqs * sizeof(struct IbMsegSendFifo);
  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remCtsFifo.flags; // IBV_SEND_INLINE
  // The QP's max_inline_data only covers one CTS entry, so multi-receive
  // CTS writes must not be inlined.
  if (req->nreqs > 1) wr.send_flags &= ~IBV_SEND_INLINE;

  // We need to occasionally post a request with the IBV_SEND_SIGNALED flag, otherwise
  // the send queue will never empty.
  //
  // From https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/
  // "How to use Unsignaled Completion?" / "Gotchas and Pitfalls"
  // All posted Send Requested, Signaled and Unsignaled, are considered outstanding until
  // a Work Completion that they, or Send Requests that were posted after them, was polled
  // from the Completion Queue associated with the Send Queue. This means if one works with
  // a Queue Pair that was configured to work with Unsignaled Completions, he must make
  // sure that occasionally (before the Send Queue is full with outstanding Send Requests)
  // a Send Request that generate Work Completion will be posted.
  //
  // Not following this rule may lead to a case that the Send Queue is full with Send
  // Requests that won't generate Work Completion:
  //
  //  - The Send Queue is full, so no new Send Requests can be posted to it
  //  - The Send Queue can't be emptied, since no Work Completion can be generated anymore
  //    (the reason is that no Work Completion, that can generate Work Completion that
  //    polling it will empty the Send Queue, can be posted)
  //  - The status of all posted Send Request is considered unknown
  //
  // slot == devIndex - When writing to CTS FIFO slot N, and this QP lives on device index N, it should send signalled.
  // This works out that each CTS posting QP gets drained
  if (slot == ctsQp->devIndex || comm->base.resiliency) {
    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = slot;
  }

  // RCCL: a multi-segment slot extends this one WR over the layout prefix. idx stays last.
  rcclIbPrepareCtsWrite(comm, req, slot, &wr);
  struct ibv_send_wr* firstWr = &wr;

  TRACE(NCCL_NET,
        "NET/IB: %s: Posting a CTS (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld, opcode=%d, send_flags=%d, "
        "qp_num=%u)",
        __func__, req, req->base, req->id, slot, req->nreqs, wr.wr_id, wr.opcode, wr.send_flags, ctsQp->qp->qp_num);

  struct ibv_send_wr* bad_wr;
  ncclResult_t postFifoRet;
  NCCLCHECKGOTO(wrap_ibv_post_send(ctsQp->qp, firstWr, &bad_wr), postFifoRet, postFifoFail);
  TRACE(NCCL_VERBS,
        "Posted send wr_id=%lu, wr_indx=%d, qp_num=%d, src_nic=%d, dst_nic=%d, dlid=%u, opcode=%d, send_flags=%d, "
        "imm_data=%d, remote_addr=%lx, rkey=%x, length=%d, lkey=%x",
        wr.wr_id, 0, ctsQp->qp->qp_num, comm->devs[ctsQp->devIndex].base.ibDevN,
        comm->base.remDevs[ctsQp->remDevIdx].ibv_dev_index, comm->base.remDevs[ctsQp->remDevIdx].lid, wr.opcode,
        wr.send_flags, wr.imm_data, wr.wr.rdma.remote_addr, wr.wr.rdma.rkey, wr.sg_list ? wr.sg_list->length : 0,
        wr.sg_list ? wr.sg_list->lkey : 0);

  TRACE(NCCL_NET,
        "NET/IB: %s: CTS posted (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld, opcode=%d, send_flags=%d, "
        "qp_num=%u)",
        __func__, req, req->base, req->id, slot, req->nreqs, wr.wr_id, wr.opcode, wr.send_flags, ctsQp->qp->qp_num);

  return ncclSuccess;

postFifoFail:
  IbMsegStatsFatalError(&comm->base.stats);
  return postFifoRet;
}

// Keep this hot loop on a 64-byte boundary.
__attribute__((aligned(64)))
ncclResult_t IbMsegIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles,
                         void** request) {
  struct IbMsegRecvComm* comm = (struct IbMsegRecvComm*)recvComm;
  ncclResult_t ret = ncclSuccess;
  struct IbMsegRequest* req = NULL;
  int nqps = 0;
  int slot = 0;
  IbMsegQp* qps[NCCL_IB_MAX_QPS];
  int qpIndexes[NCCL_IB_MAX_QPS];
  struct IbMsegSendFifo* localElem = nullptr;
  bool anyMultiSeg = false;
  if (comm->base.ready == 0) {
    WARN("NET/IB: IbMsegIrecv() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  if (n > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;
  NCCLCHECKGOTO(IbMsegStatsCheckFatalCount(&comm->base.stats, __func__), ret, irecvFail);

  NCCLCHECKGOTO(IbMsegCommBaseGetNqpsPerRequest(&comm->base, &nqps), ret, irecvFail);
#ifdef NCCL_ENABLE_NET_PROFILING
  if (nqps > MAX_QPS_PER_REQ) {
    WARN("NET/IB: QP count %d exceeds maximum profiler event handle capacity %d", nqps, MAX_QPS_PER_REQ);
    return ncclInternalError;
  }
#endif
  if (nqps > NCCL_IB_MAX_QPS) {
    WARN("NET/IB: QP count %d exceeds maximum QP capacity %d", nqps, NCCL_IB_MAX_QPS);
    return ncclInternalError;
  }

  // Validate QPs before allocating/publishing the request.
  for (int i = 0; i < nqps; i++) {
    NCCLCHECKGOTO(IbMsegCommBaseGetQpForRequest(&comm->base, comm->base.fifoHead, i, &qps[i], &qpIndexes[i]), ret,
                  irecvFail);
  }

  NCCLCHECKGOTO(IbMsegGetRequest(&comm->base, &req), ret, irecvFail);
  slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  req->id = comm->base.fifoHead;
  req->type = NCCL_NET_IB_REQ_RECV;
  req->sock = &comm->base.sock;
  req->nreqs = n;
  if (comm->base.resiliency) {
    // When resiliency is enabled, a recv request can be served by any device.
    for (int devIndex = 0; devIndex < comm->base.vProps.ndevs; devIndex++) {
      req->devBases[devIndex] = IbMsegGetNetCommDevBase(&comm->base, devIndex);
    }
  }
  TRACE(NCCL_NET, "NET/IB: %s: Recv request created (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, tag[0]=%x)", __func__,
        req, req->base, req->id, slot, n, tags[0]);

#ifdef NCCL_ENABLE_NET_PROFILING
  for (int r = 0; r < n && phandles; r++) req->pInfo[r].nEventHandles = 0;
#endif

  // Store the request in a table for easy retrieval by ID.
  comm->recvReqs[req->id % NET_IB_MAX_REQUESTS] = req;
  req->recv.flagFlush = comm->flagFlush && comm->flagFlushHint;

  TIME_START(1);
  for (int i = 0; i < nqps; i++) {
    IbMsegQp* qp = qps[i];
    IbMsegAddEvent(req, qp->devIndex);
    if (comm->prepostReceiveWorkRequests) {
      continue;
    }
    // Post receive work request on the QP
    comm->ibRecvWorkRequest.wr_id = slot;
    NCCLCHECK(IbMsegPostRecvWorkRequest(qp->qp, &comm->ibRecvWorkRequest));
#ifdef NCCL_ENABLE_NET_PROFILING
    // Start a QP event for every request in the multirecv and every qp
    int qpIndex = qpIndexes[i];
    for (int r = 0; r < n; r++) {
      int nEventHandles = req->pInfo[r].nEventHandles;
      req->pInfo[r].qpIndex[nEventHandles] = qpIndex;
      // Store info for profiler
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      req->pInfo[r].data.type = ncclProfileQp;
      req->pInfo[r].data.qp.device = qp->devIndex;
      req->pInfo[r].data.qp.wr_id = comm->ibRecvWorkRequest.wr_id;
      req->pInfo[r].data.qp.qpNum = qp->qp->qp_num;
      NCCLCHECK(IbMsegProfilerFunction(&req->pInfo[r].qpEventHandles[nEventHandles], ncclProfilerNetEventStart,
                                     phandles[r], pluginId, &req->pInfo[r].data));
      req->pInfo[r].nEventHandles++;
    }
#endif
  }
  TIME_STOP(1);

  req->recv.aggSize = 0;
  req->recv.cmplsRecords = &comm->cmplsRecords[slot];
  memset(req->recv.cmplsRecords->sizes, 0, sizeof(int) * n);
  memset(req->recv.cmplsRecords->completions, 0, sizeof(req->recv.cmplsRecords->completions));
  localElem = rcclIbRecvCtsRow(comm, slot);
  for (int i = 0; i < n; i++) {
    localElem[i].addr = (uint64_t)data[i];
    struct IbMsegMrHandle* mhandleWrapper = (struct IbMsegMrHandle*)mhandles[i];
    // RCCL: reject a missing per-device MR before the CTS rkey load.
    NCCLCHECKGOTO(rcclIbIrecvCheckHandle(comm, mhandleWrapper, i), ret, irecvFail);
    // Send all applicable rkeys
    for (int j = 0; j < comm->base.vProps.ndevs; j++) {
      localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;
    }
    localElem[i].nreqs = n;
    localElem[i].size = sizes[i]; // Sanity/Debugging
    localElem[i].tag = tags[i];
    localElem[i].flagFlush = req->recv.flagFlush;
    // RCCL: layout prefix first. idx is stored after it so it is the last word of the write.
    NCCLCHECKGOTO(rcclIbIrecvFillLayout(comm, slot, i, mhandleWrapper, &anyMultiSeg), ret, irecvFail);
  }
  // RCCL: this write includes the layout prefix.
  if (anyMultiSeg) rcclIbCtsFlagLayout(localElem, n, comm->base.fifoHead + 1);
  for (int i = 0; i < n; i++) localElem[i].idx = comm->base.fifoHead + 1;

  // Post to FIFO to notify sender
  TIME_START(2);
  ncclResult_t irecvRet;
  NCCLCHECKGOTO(IbMsegPostFifo(comm, req, slot), irecvRet, irecvFail);
  comm->base.fifoHead++;
  TIME_STOP(2);

  *request = req;
  return ncclSuccess;

irecvFail:
  IbMsegStatsFatalError(&comm->base.stats);
  if (req) {
    // If events were added (IBV ops posted), we can't free immediately -
    // completions may still arrive. Mark as failed and return it so
    // caller can call Test to drain completions.
    if (IbMsegRequestHasEvents(req)) {
      req->type = NCCL_NET_IB_REQ_FAILED;
      *request = req;
      // Return success so caller proceeds to call Test() which will drain
      // completions. The fatal error is recorded in stats and will be
      // caught on the next operation.
      return ncclSuccess;
    } else {
      IbMsegFreeRequest(req);
      *request = NULL;
    }
  } else {
    *request = NULL;
  }
  return ret;
}

ncclResult_t IbMsegIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct IbMsegRecvComm* comm = (struct IbMsegRecvComm*)recvComm;
  ncclResult_t ret = ncclSuccess;
  struct IbMsegRequest* req = NULL;
  struct IbMsegMrHandle* mhandle = NULL;
  int last = -1;
  for (int i = 0; i < n; i++)
    if (sizes[i]) last = i;
  if (comm->flushEnabled == 0 || last == -1) return ncclSuccess;
  NCCLCHECK(IbMsegCreateFlushQp(comm));

  // Only flush once using the last non-zero receive
  mhandle = (struct IbMsegMrHandle*)mhandles[last];
  NCCLCHECKGOTO(IbMsegGetRequest(&comm->base, &req), ret, iflushFail);
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;

  // We don't know which devIndex the recv was on, so we flush on all devices
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    struct ibv_send_wr wr;
    // RCCL: fence the RO=1 bulk data through an RO=0 scratchpad - RDMA_WRITE a
    // dummy payload into it, then RDMA_READ it back. The WRITE needs REMOTE_WRITE
    // on the flush QP.
    bool useGpuFlushMem = rcclParamIbGdrFlushGpuMemNoRelaxedOrdering() &&
                          comm->devs[i].gpuFlush.gpuFlushGpuMem != nullptr && comm->devs[i].gpuFlush.gpuMr != nullptr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)(req - comm->base.reqs) + NCCL_IB_FLUSH_REQ_WR_ID_OFFSET;

    // Chain the unsignaled write to the signaled read: one post, RC send-queue order keeps the fence.
    struct ibv_send_wr writeWr;
    if (useGpuFlushMem) {
      memset(&writeWr, 0, sizeof(writeWr));
      writeWr.wr_id = wr.wr_id;
      writeWr.wr.rdma.remote_addr = (uint64_t)(comm->devs[i].gpuFlush.gpuFlushGpuMem);
      writeWr.wr.rdma.rkey = comm->devs[i].gpuFlush.gpuMr->rkey;
      writeWr.sg_list = &comm->devs[i].gpuFlush.sge;
      writeWr.num_sge = 1;
      writeWr.opcode = IBV_WR_RDMA_WRITE;
      writeWr.send_flags = IBV_SEND_INLINE;
      writeWr.next = &wr;

      wr.wr.rdma.remote_addr = (uint64_t)(comm->devs[i].gpuFlush.gpuFlushGpuMem);
      wr.wr.rdma.rkey = comm->devs[i].gpuFlush.gpuMr->rkey;
    } else {
      // RCCL: on a multi-segment handle the read's rkey is the MR of the segment holding that byte.
      struct ibv_mr* flushMr = IbMsegMrForRange(mhandle, (uintptr_t)data[last], /*len=*/1, i);
      if (flushMr == NULL) {
        WARN("NET/IB: flush missing MR for receive %d buffer %p", last, data[last]);
        ret = ncclInternalError;
        goto iflushFail;
      }
      wr.wr.rdma.remote_addr = (uint64_t)data[last];
      wr.wr.rdma.rkey = flushMr->rkey;
    }
    wr.sg_list = &comm->devs[i].gpuFlush.sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    TRACE(NCCL_NET, "NET/IB: %s: Posting a flush request (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base,
          wr.wr_id);
    TIME_START(4);
    struct ibv_send_wr* bad_wr;
    struct ibv_send_wr* postWr = useGpuFlushMem ? &writeWr : &wr;
    NCCLCHECKGOTO(wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, postWr, &bad_wr), ret, iflushFail);
    TIME_STOP(4);

    IbMsegAddEvent(req, i);

    TRACE(NCCL_NET, "NET/IB: %s: Flush request posted (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base,
          wr.wr_id);
  }

  *request = req;
  return ncclSuccess;

iflushFail:
  IbMsegStatsFatalError(&comm->base.stats);
  if (req) {
    // If events were added (IBV ops posted), we can't free immediately -
    // completions may still arrive. Mark as failed and return it so
    // caller can call Test to drain completions.
    if (IbMsegRequestHasEvents(req)) {
      req->type = NCCL_NET_IB_REQ_FAILED;
      *request = req;
      // Return success so caller proceeds to call Test() which will drain
      // completions. The fatal error is recorded in stats and will be
      // caught on the next operation.
      return ncclSuccess;
    } else {
      IbMsegFreeRequest(req);
      *request = NULL;
    }
  } else {
    *request = NULL;
  }
  return ret;
}

#define HCA_NAME(req, index) ((req)->devBases[(index)]->pd->context->device->name)

#ifdef NCCL_ENABLE_NET_PROFILING
static int getReqQpIndex(struct IbMsegRequest* req, int request, int qpNumber) {
  for (int i = 0; i < MAX_QPS_PER_REQ; i++) {
    int qpIndex = req->pInfo[request].qpIndex[i];
    if (req->base->qps[qpIndex].qp->qp_num == qpNumber) return i;
  }
  return 0;
}
#endif

static inline ncclResult_t IbMsegRequestRetrieveFromCompletion(struct IbMsegNetCommBase* base, ibv_wc* wc,
                                                               IbMsegRequest** req) {
  if (req == NULL) {
    WARN("NET/IB: Request output pointer is NULL");
    return ncclInternalError;
  }
  if (wc == NULL) {
    WARN("NET/IB: Work completion is NULL");
    return ncclInternalError;
  }

  // In case of a completion with error, there is no guarantee that all fields
  // of the completion are valid.
  if (wc->status != IBV_WC_SUCCESS) {
    WARN("NET/IB: Work completion status is %s(%d), expected %s(%d)", ibvWcStatusStr(wc->status), wc->status,
         ibvWcStatusStr(IBV_WC_SUCCESS), IBV_WC_SUCCESS);
    return ncclInternalError;
  }

  TRACE(NCCL_NET, "NET/IB: %s: Retrieving a %s request (wr_id=%ld, opcode=%s)", __func__,
        base->isSend ? "send" : "recv", wc->wr_id, ibvWcOpcodeStr(wc->opcode));

  if (!base->isSend && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM && base->recvMatchingScheme == BY_ID) {
    TRACE(NCCL_NET, "NET/IB: %s: Retrieving a receive request (wr_id=%ld, opcode=%s, imm_data=%d, byte_len=%d)",
          __func__, wc->wr_id, ibvWcOpcodeStr(wc->opcode), be32toh(wc->imm_data), wc->byte_len);
    struct IbMsegRecvComm* recvComm = (struct IbMsegRecvComm*)base;
    *req = recvComm->recvReqs[be32toh(wc->imm_data) % NET_IB_MAX_REQUESTS];
  } else if (!base->isSend && wc->opcode == IBV_WC_RDMA_READ) {
    // Flush request completion
    NCCLCHECK(IbMsegRequestRetrieveAsIndex(base->reqs, (wc->wr_id - NCCL_IB_FLUSH_REQ_WR_ID_OFFSET), req));
  } else if (!base->isSend) {
    struct IbMsegRecvComm* recvComm = (struct IbMsegRecvComm*)base;
    *req = recvComm->recvReqs[wc->wr_id];
  } else {
    struct IbMsegSendComm* sendComm = (struct IbMsegSendComm*)base;
    // On the sender side, the lower 8 bits of wr_id are used to retrieve the
    // request, since in multi-send case, multiple IDs are encoded in the same
    // wr_id.,
    *req = sendComm->sendReqs[wc->wr_id & 0xff][0];
  }
  TRACE(NCCL_NET,
        "NET/IB: %s: Retrieved a %s request (req=%p, comm=%p, id=%ld, type=%s, wc.wr_id=%ld, wc.opcode=%s, "
        "wc.imm_data=%d, wc.byte_len=%d, wc.qp_num=%u)",
        __func__, base->isSend ? "send" : "recv", *req, (*req)->base, (*req)->id, IbMsegReqTypeStr[(*req)->type],
        wc->wr_id, ibvWcOpcodeStr(wc->opcode), be32toh(wc->imm_data), wc->byte_len, wc->qp_num);
  return ncclSuccess;
}

static inline ncclResult_t IbMsegRequestIsComplete(struct IbMsegRequest* request, bool* complete) {
  *complete =
    (request->events[0] == 0 && request->events[1] == 0 && request->events[2] == 0 && request->events[3] == 0);
  if (!*complete && request->base->resiliency) {
    NCCLCHECK(IbMsegResiliencyRequestIsComplete(request, complete));
  }
  return ncclSuccess;
}

static inline ncclResult_t IbMsegRequestComplete(struct IbMsegRequest* r, int* done, int* sizes) {
  TRACE(NCCL_NET, "NET/IB: %s: %s request completed (req=%p, comm=%p, id=%ld, type=%s)", __func__,
        r->base->isSend ? "Send" : "Recv", r, r->base, r->id, IbMsegReqTypeStr[r->type]);
  *done = 1;
  if (r->type == NCCL_NET_IB_REQ_RECV && r->recv.flagFlush) {
    // net.cc reads this right after IbMsegTest() completes the request, so it
    // describes this request even if a later one already completed.
    // Same QPs as IbMsegCommBaseGetQpForRequest: nQps consecutive QPs from
    // (id * nQps) % nqps; nQps is 1 or nqps, so they never wrap.
    struct IbMsegRecvComm* recvComm = (struct IbMsegRecvComm*)r->base;
    int nQpsPerRequest = 0;
    NCCLCHECK(IbMsegCommBaseGetNqpsPerRequest(&recvComm->base, &nQpsPerRequest));
    uint64_t nQps = nQpsPerRequest;
    uint64_t first = (r->id * nQps) % recvComm->base.nqps;
    recvComm->flagFlushLastDesc = (nQps << NCCL_GDR_FLUSH_COUNT_SHIFT) | (first << NCCL_GDR_FLUSH_FIRST_SHIFT) |
                                  ((r->id + 1) & NCCL_GDR_FLUSH_SEQ_MASK);
  }
  if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
    TRACE(NCCL_NET, "NET/IB: %s: Recv request completed (req=%p, comm=%p, id=%ld, type=%s, nreqs=%d)", __func__, r,
          r->base, r->id, IbMsegReqTypeStr[r->type], r->nreqs);
    int* sizesToReport =
      (r->nreqs > 1 || r->recv.cmplsRecords->sizes[0] > 0) ? r->recv.cmplsRecords->sizes : &(r->recv.aggSize);
    for (int i = 0; i < r->nreqs; i++) {
      sizes[i] = sizesToReport[i];
#ifdef NCCL_ENABLE_NET_PROFILING
      for (int j = 0; j < r->pInfo[i].nEventHandles; j++) {
        NCCLCHECK(IbMsegProfilerFunction(&r->pInfo[i].qpEventHandles[j], ncclProfilerNetEventStop, NULL, 0, NULL));
      }
#endif
    }
  }
  if (r->type == NCCL_NET_IB_REQ_SEND) {
    TRACE(NCCL_NET, "NET/IB: %s: Send request completed (req=%p, comm=%p, id=%ld)", __func__, r, r->base, r->id);
    if (sizes) {
      sizes[0] = r->send.size;
#ifdef NCCL_ENABLE_NET_PROFILING
      for (int j = 0; j < r->pInfo[0].nEventHandles; j++) {
        NCCLCHECK(IbMsegProfilerFunction(&r->pInfo[0].qpEventHandles[j], ncclProfilerNetEventStop, NULL, 0, NULL));
      }
#endif
    }
    int slot = r->id % NET_IB_MAX_REQUESTS;
    struct IbMsegSendComm* sendComm = (struct IbMsegSendComm*)r->base;
    sendComm->sendReqsCnt[slot]--;
    if (sendComm->sendReqsCnt[slot] == 0) {
      // Only after completing the last send of a multi-recv, allow accepting
      // following send requests on the same slot.
      memset(&sendComm->sendReqs[slot], 0, sizeof(sendComm->sendReqs[slot]));
      // In case of event-based LB, detect any speed change and update the device speeds
      if (ncclParamIbEventBasedLb()) IbMsegCheckSpeedChanges(sendComm, r->base);
    }
  }
  // Stop all remaining Qp events for this event
  NCCLCHECK(IbMsegFreeRequest(r));
  return ncclSuccess;
}

static ncclResult_t IbMsegPostSpeedUpdateToRemote(struct IbMsegRecvComm* comm) {
  uint64_t counter = COMPILER_ATOMIC_LOAD(&IbMsegSpeedChangeCounter, std::memory_order_acquire);
  if (counter == comm->base.speedChangeCounter || comm->remSpeedBufAddr == 0 || comm->postedSpeedUpdate) {
    return ncclSuccess;
  }

  int nDevs = comm->base.vProps.ndevs;
  uint16_t newSpeedsGbps[NCCL_IB_MAX_DEVS_PER_NIC] = {};
  for (int i = 0; i < nDevs; i++)
    newSpeedsGbps[i] =
      (uint16_t)(COMPILER_ATOMIC_LOAD(&IbMsegDevs[comm->base.vProps.devs[i]].currSpeed, std::memory_order_relaxed) /
                 1000);

  comm->base.speedChangeCounter = counter;
  if (memcmp(newSpeedsGbps, comm->lastSentSpeeds, nDevs * sizeof(uint16_t)) == 0) return ncclSuccess;

  int devIdx = comm->base.qps[0].devIndex;
  int remDevIdx = comm->base.qps[0].remDevIdx;
  comm->speedUpdateBuf.counter = comm->lastSentCounter + 1;
  memcpy(comm->speedUpdateBuf.speedGbps, newSpeedsGbps, nDevs * sizeof(uint16_t));

  struct ibv_send_wr wr = {}, *bad_wr;
  struct ibv_sge sge = {};
  sge.addr = (uint64_t)&comm->speedUpdateBuf;
  sge.length = sizeof(IbMsegRemoteSpeedBuf);
  sge.lkey = comm->devs[devIdx].speedUpdateMr->lkey;
  wr.wr_id = NCCL_IB_SPEED_UPDATE_WR_ID;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.wr.rdma.remote_addr = comm->remSpeedBufAddr;
  wr.wr.rdma.rkey = comm->base.remDevs[remDevIdx].remSpeedBufRkey;
  comm->postedSpeedUpdate = true;
  NCCLCHECK(wrap_ibv_post_send(comm->base.qps[0].qp, &wr, &bad_wr));
  memcpy(comm->lastSentSpeeds, newSpeedsGbps, nDevs * sizeof(uint16_t));
  comm->lastSentCounter = comm->speedUpdateBuf.counter;
  return ncclSuccess;
}

// Log the details of a completion with error. The provided devIndex is the index
// of the IB device on which the completion was received.
static ncclResult_t IbMsegLogCompletionWithError(struct IbMsegNetCommBase* commBase, struct ibv_wc* wc, int devIndex) {
  struct IbMsegNetCommDevBase* devBase = IbMsegGetNetCommDevBase(commBase, devIndex);
  char localGidString[INET6_ADDRSTRLEN] = "";
  char remoteGidString[INET6_ADDRSTRLEN] = "";
  const char *localGidStr = NULL, *remoteGidStr = NULL;
  if (devBase->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
    localGidStr = ibvGetGidStr(&devBase->gidInfo.localGid, localGidString, sizeof(localGidString));
    remoteGidStr = ibvGetGidStr(&commBase->remDevs[devIndex].remoteGid, remoteGidString, sizeof(remoteGidString));
  }

  char sockStr[SOCKET_NAME_MAXLEN + 1];
  union ncclSocketAddress addr;
  ncclSocketGetAddr(&commBase->sock, &addr);
  ncclSocketToString(&addr, sockStr, sizeof(sockStr));
  char* hcaName = devBase->pd->context->device->name;
  WARN("NET/IB: Got completion from peer %s with status=%s(%d) opcode=%s(%d) vendor_err=%u qp_num=%u %s%s%s%s hca %s",
       sockStr, ibvWcStatusStr(wc->status), wc->status, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->vendor_err,
       wc->qp_num, localGidStr ? " localGid " : "", localGidString, remoteGidStr ? " remoteGids" : "", remoteGidString,
       hcaName);
  printIbWcStatusHint(wc->status);
  return ncclSuccess;
}

static inline ncclResult_t IbMsegCompletionEventProcess(struct IbMsegNetCommBase* commBase, struct ibv_wc* wc,
                                                        int devIndex) {
  if (wc->wr_id == NCCL_IB_SPEED_UPDATE_WR_ID) {
    struct IbMsegRecvComm* rComm = (struct IbMsegRecvComm*)commBase;
    rComm->postedSpeedUpdate = false;
    return ncclSuccess;
  }
  // RCCL: a send comm reads only referenced segment layouts.
  if (commBase->isSend && wc->opcode == IBV_WC_RDMA_READ)
    return rcclIbSendLayoutReadDone((struct IbMsegSendComm*)commBase, wc->wr_id);

  union ncclSocketAddress addr;
  ncclSocketGetAddr(&commBase->sock, &addr);

  struct IbMsegRequest* req = NULL;
  NCCLCHECK(IbMsegRequestRetrieveFromCompletion(commBase, wc, &req));
  if (req == NULL) {
    WARN("NET/IB: %s: %s comm could not retreive a request found for a successful completion (comm=%p, wc.wr_id=%ld, "
         "opcode=%d, qp_num=%u)",
         __func__, commBase->isSend ? "Send" : "Recv", commBase, wc->wr_id, wc->opcode, wc->qp_num);
    return ncclInternalError;
  }

#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
  TRACE(NCCL_NET,
        "Got completion from peer %s with status=%d opcode=%d len=%u wr_id=%lu r=%p type=%d events={%d,%d,%d,%d}, "
        "devIndex=%d",
        ncclSocketToString(&addr, line, sizeof(line)), wc->status, wc->opcode, wc->byte_len, wc->wr_id, req, req->type,
        req->events[0], req->events[1], req->events[2], req->events[3], devIndex);
#endif

  if (commBase->isSend) {
    if (req->type != NCCL_NET_IB_REQ_SEND) {
      WARN("NET/IB: %s: Sender expected a 'send' request but got '%s' (req=%p, comm=%p, id=%ld, wc.wr_id=%ld, "
           "wc.opcode=%s(%d), wc.qp_num=%u)",
           __func__, IbMsegReqTypeStr[req->type], req, commBase, req->id, wc->wr_id, ibvWcOpcodeStr(wc->opcode),
           wc->opcode, wc->qp_num);
      return ncclInternalError;
    }
    struct IbMsegSendComm* sendComm = (struct IbMsegSendComm*)commBase;
    struct IbMsegRequest* sendReq = NULL;
    int slot = req->id % NET_IB_MAX_REQUESTS;
    // RCCL: a slice completion frees its QP for the next slice; the QP's profiler event runs until the tail.
    [[maybe_unused]] const bool slice = rcclIbSendSliceDone(sendComm, slot, devIndex, wc->qp_num);
    for (int j = 0; j < req->nreqs; j++) {
      sendReq = sendComm->sendReqs[slot][j];
      if (!commBase->resiliency && (sendReq->events[devIndex] <= 0)) {
        WARN("NET/IB: sendReq(%p)->events={%d,%d,%d,%d}, devIndex=%d, reqIdx=%d <= 0", sendReq, sendReq->events[0],
             sendReq->events[1], sendReq->events[2], sendReq->events[3], devIndex, j);
        return ncclInternalError;
      }
      sendReq->events[devIndex]--;
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a send request (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)",
            __func__, sendReq, sendReq->base, sendReq->id, devIndex, wc->qp_num);
#ifdef NCCL_ENABLE_NET_PROFILING
      // Stop Qp event for sendReq
      int qpIndex = getReqQpIndex(sendReq, j, wc->qp_num);
      if (!slice) {
        NCCLCHECK(IbMsegProfilerFunction(&sendReq->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL,
                                         0, NULL));
      }
#endif
    }
  } else {
    if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      if (req->type == NCCL_NET_IB_REQ_UNUSED && commBase->resiliency) {
        INFO(NCCL_NET,
             "NET/IB: %s: Receiver got a completion for a data transfer but retrieved an 'unused' request (req=%p, "
             "comm=%p, id=%ld, wc.status=%s(%d), wc.wr_id=%ld, wc.imm_data=%d, wc.opcode=%s(%d), wc.qp_num=%u)",
             __func__, req, commBase, req->id, ibvWcStatusStr(wc->status), wc->status, wc->wr_id, be32toh(wc->imm_data),
             ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num);
        return ncclSuccess;
      }
      if (req->type != NCCL_NET_IB_REQ_RECV && !commBase->resiliency) {
        WARN("NET/IB: %s: Receiver expected a 'recv' request but got '%s' (req=%p, comm=%p, id=%ld, wc.wr_id=%ld, "
             "wc.status=%s(%d) wc.opcode=%s(%d), wc.qp_num=%u)",
             __func__, IbMsegReqTypeStr[req->type], req, req->base, req->id, wc->wr_id, ibvWcStatusStr(wc->status),
             wc->status, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num);
        return ncclInternalError;
      }
      if (req->nreqs == 1) {
        if (commBase->recvMatchingScheme == BY_INDEX) {
          req->recv.cmplsRecords->sizes[0] = be32toh(wc->imm_data);
        } else if (req->recv.cmplsRecords->sizes[0] == 0) {
          req->recv.aggSize += wc->byte_len;
        }
      }
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a recv request (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)",
            __func__, req, req->base, req->id, devIndex, wc->qp_num);
      struct IbMsegRecvComm* recvComm = (struct IbMsegRecvComm*)commBase;
      if (recvComm->prepostReceiveWorkRequests) {
        // Post another receive work request on the QP
        IbMsegQp* qp = NULL;
        int qpIndex = -1;
        NCCLCHECK(IbMsegCommBaseGetQpByQpNum(commBase, devIndex, wc->qp_num, &qp, &qpIndex));
        req->recv.cmplsRecords->completions[qpIndex] = 1;
        IbMsegPostRecvWorkRequest(qp->qp, &recvComm->ibRecvWorkRequest);
      }
      req->events[devIndex]--;
    } else if (wc->opcode == IBV_WC_RDMA_READ) {
      TRACE(NCCL_NET,
            "NET/IB: %s: Got completion for a flush request (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)",
            __func__, req, req->base, req->id, devIndex, wc->qp_num);
      req->events[devIndex]--;
    } else if (wc->opcode == IBV_WC_RDMA_WRITE) {
      // This is a CTS completion
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a CTS (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)",
            __func__, req, req->base, req->id, devIndex, wc->qp_num);
      if (req->type == NCCL_NET_IB_REQ_UNUSED) {
        INFO(NCCL_NET,
             "NET/IB: %s: Receiver got a completion for a CTS but retrieved an 'unused' request (req=%p, comm=%p, "
             "id=%ld, wc.wr_id=%ld, wc.opcode=%s(%d), wc.qp_num=%u, wc.imm_data=%d)",
             __func__, req, req->base, req->id, wc->wr_id, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num,
             be32toh(wc->imm_data));
        return ncclSuccess;
      }
      // On CTS completion, propagate any local speed changes to the remote sender
      if (ncclParamIbEventBasedLb() && ncclParamIbEventBasedLbRemote())
        NCCLCHECK(IbMsegPostSpeedUpdateToRemote((struct IbMsegRecvComm*)commBase));
    } else {
      WARN("NET/IB: %s: Unknown completion (req=%p, comm=%p, id=%ld, devIndex=%d, req->type=%s, wc.wr_id=%ld, "
           "wc.opcode=%s(%d), wc.qp_num=%u, wc.imm_data=%d)",
           __func__, req, commBase, req ? req->id : -1, devIndex, IbMsegReqTypeStr[req->type], wc->wr_id,
           ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num, be32toh(wc->imm_data));
      return ncclInternalError;
    }
#ifdef NCCL_ENABLE_NET_PROFILING
    // Stop Qp event for workFifo
    for (int j = 0; j < req->nreqs; j++) {
      int qpIndex = getReqQpIndex(req, j, wc->qp_num);
      NCCLCHECK(IbMsegProfilerFunction(&req->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL, 0, NULL));
    }
#endif
  }
  return ncclSuccess;
}

// Keep this hot loop on a 64-byte boundary.
__attribute__((aligned(64)))
ncclResult_t IbMsegTest(void* request, int* done, int* sizes) {
  struct IbMsegRequest* r = (struct IbMsegRequest*)request;
  *done = 0;
  ncclResult_t ret = ncclSuccess;
  int failDevIdx = -1;

  if (r->base->resiliency && r->base->resiliency->inProgress) {
    NCCLCHECK(IbMsegResiliencyProgress(r->base->resiliency));
  }

  int totalWrDone = 0;
  int wrDone = 0;
  struct ibv_wc wcs[4];
  do {
    NCCLCHECKGOTO(IbMsegStatsCheckFatalCount(&r->base->stats, __func__), ret, fail);
    bool complete = false;
    NCCLCHECKGOTO(IbMsegRequestIsComplete(r, &complete), ret, fail);
    if (complete) {
      NCCLCHECK(IbMsegRequestComplete(r, done, sizes));
      return ncclSuccess;
    }

    totalWrDone = 0;
    for (int i = 0; i < r->base->vProps.ndevs; i++) {
      // Reasons to skip polling this device:
      // 1. When resiliency is enabled events counters might reach negative values.
      // 2. On the sender side, a request might not use all devices (e.g., upon
      //    submission of the send request, a device was not available)
      if (!r->devBases[i] || (r->events[i] == 0 && !r->base->resiliency)) {
        continue;
      }
      TIME_START(3);
      ret = wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone);
      if (ret != ncclSuccess) {
        failDevIdx = i;
        goto fail;
      }
      if (wrDone == 0) {
        TIME_CANCEL(3);
      } else {
        TIME_STOP(3);
      }
      totalWrDone += wrDone;

      uint64_t tPollNs = 0;
      bool tPollNsValid = false;
      for (int w = 0; w < wrDone; w++) {
        struct ibv_wc* wc = wcs + w;
        if (wc->status != IBV_WC_SUCCESS) {
          // Speed update write failure is non-fatal; clear the pending flag so future updates can be sent.
          if (wc->wr_id == NCCL_IB_SPEED_UPDATE_WR_ID) {
            struct IbMsegRecvComm* rComm = (struct IbMsegRecvComm*)r->base;
            rComm->postedSpeedUpdate = false;
            continue;
          }
          if (r->base->resiliency == NULL) {
            WARN("NET/IB: %s: Got CQE with error (devIndex=%d, req=%p, comm=%p (%s), wr_id=%lu, qp_num=%d)", __func__,
                 i, r, r->base, r->base->isSend ? "send" : "recv", wc->wr_id, wc->qp_num);
            IbMsegLogCompletionWithError(r->base, wc, i);
            // If resiliency is not enabled, we cannot recover from any error.
            return ncclRemoteError;
          }
          NCCLCHECK(IbMsegResiliencyHandleCompletionError(r->base->resiliency, wc, i));
        } else {
          if (IbMsegWqeLatEnabled) IbMsegWqeLatHandleCompletion(r->base, i, wc, &tPollNs, &tPollNsValid);
          TRACE(NCCL_NET,
                "NET/IB: %s: Processing a completion event (devIndex=%d, comm=%p (%s), req=%p, wr_id=%lu, qp_num=%d)",
                __func__, i, r->base, r->base->isSend ? "send" : "recv", r, wc->wr_id, wc->qp_num);
          NCCLCHECK(IbMsegCompletionEventProcess(r->base, wc, i));
        }
      }

      // Scan for stalled (late/missing-CQE) WQEs on every poll, including
      // iterations where poll_cq returned zero, so a hung WQE is caught
      // even when no completions are flowing.
      if (IbMsegWqeLatEnabled) IbMsegWqeLatScanStalls(r->base, i);

      if (wrDone == 0) continue;
      // Once the IB fatal event is reported in the async thread, we want to propagate this error
      // to communicator and prevent further polling to reduce error pollution.
      ret = IbMsegStatsCheckFatalCount(&IbMsegDevs[r->devBases[i]->ibDevN].stats, __func__);
      if (ret != ncclSuccess) {
        failDevIdx = i;
        goto fail;
      }
    }
    // RCCL: a completion may have freed a QP of a send still posting slices.
    if (r->base->isSend) NCCLCHECKGOTO(rcclIbSendProgress((struct IbMsegSendComm*)r->base), ret, fail);
  } while (totalWrDone > 0);

  // If no (more) CQEs found on any device, return and come back later
  return ncclSuccess;
fail:
  // Mark connection and device as fatal
  IbMsegStatsFatalError(&r->base->stats);
  if (failDevIdx >= 0 && r->devBases[failDevIdx] != NULL) {
    IbMsegStatsFatalError(&IbMsegDevs[r->devBases[failDevIdx]->ibDevN].stats);
  }
  // Mark request as failed so subsequent Test calls drain completions
  r->type = NCCL_NET_IB_REQ_FAILED;
  *done = 1;
  return ret;
}
