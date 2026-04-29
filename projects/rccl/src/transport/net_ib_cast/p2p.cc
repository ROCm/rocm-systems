/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "p2p_cast.h"
#include "common_cast.h"
#include "p2p_resiliency_cast.h"

NCCL_PARAM(IbCastArThreshold, "IB_AR_THRESHOLD", -2);
int64_t IbCastArThreshold = 8192;

// By default, use ncclIbRequestMatchingScheme::BY_INDEX matching scheme.
NCCL_PARAM(IbCastReceiverSideMatchingScheme, "IB_RECEIVER_SIDE_MATCHING_SCHEME", -2);
RCCL_PARAM(IbCastGdrFlushGpuMemNoRelaxedOrdering, "GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING", 1);

const char* IbCastReqTypeStr[] = { "Unused", "Send", "Recv", "Flush", "IPut" };

ncclResult_t IbCastGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  for (int i=0; i<NET_IB_MAX_REQUESTS; i++) {
    struct ncclIbRequest* r = base->reqs+i;
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

ncclResult_t IbCastFreeRequest(struct ncclIbRequest* r) {
  r->type = NCCL_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}

void IbCastAddEvent(struct ncclIbRequest* req, int devIndex) {
  struct ncclIbNetCommDevBase* base = IbCastGetNetCommDevBase(req->base, devIndex);
  req->events[devIndex]++;
  req->devBases[devIndex] = base;
}
void IbCastAddEventCTS(struct ncclIbRequest* req, int devIndex) {
  struct ncclIbNetCommDevBase* base = IbCastGetNetCommDevBase(req->base, devIndex);
  req->events[devIndex]++;
  req->ctsEvents[devIndex]++;
  req->devBases[devIndex] = base;
}

#ifdef ENABLE_TRACE
static ncclResult_t IbCastPrintWr(struct ibv_send_wr* wr, char* wrStr) {
  if (wr == NULL) {
    sprintf(wrStr, "wr=NULL");
    return ncclSuccess;
  }
  const char* opcodeStr = ibvWrOpcodeStr(wr->opcode);
  switch (wr->opcode) {
    case IBV_WR_RDMA_WRITE:
      sprintf(wrStr, "wr=%p, wr_id=%ld, opcode=%s, num_sge=%d, sge[0].length=%" PRIu32 ", sge[0].addr=0x%016" PRIx64 ", rdma.remote_addr=0x%016" PRIx64 ", rdma.rkey=0x%x",
        wr,
        wr->wr_id,
        opcodeStr,
        wr->num_sge,
        (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->length : 0,
        (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->addr : 0,
        wr->wr.rdma.remote_addr,
        wr->wr.rdma.rkey);
      break;
    case IBV_WR_RDMA_WRITE_WITH_IMM:
      sprintf(wrStr, "wr=%p, wr_id=%ld, opcode=%s, num_sge=%d, sge[0].length=%" PRIu32 ", sge[0].addr=0x%016" PRIx64 ", rdma.remote_addr=0x%016" PRIx64 ",  rdma.rkey=0x%x, imm_data=0x%x",
        wr,
        wr->wr_id,
        opcodeStr,
        wr->num_sge,
        (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->length : 0,
        (wr->num_sge > 0 && wr->sg_list) ? wr->sg_list->addr : 0,
        wr->wr.rdma.remote_addr,
        wr->wr.rdma.rkey,
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

ncclResult_t IbCastMultiSend(struct ncclIbSendComm* comm, int slot, int nqps, int startQpIndex, bool wrrSched, bool useWriteOp) {
  struct ncclIbRequest** reqs = comm->sendReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->ctsFifo[slot];
  int nreqs = slots[0].nreqs;
  uint64_t nowNs = 0;
  if (nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  if (comm->useCtsOffload) {
    nreqs = 1;
  }

  TRACE(NCCL_NET, "NET/IB: %s: Posting a send request (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d)", __func__, reqs[0], reqs[0]->base, reqs[0]->id, slot, nreqs);

  uint64_t wr_id = 0ULL;
  for (int r=0; r<nreqs; r++) {
    struct ibv_send_wr* wr = comm->wrs+r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge* sge = comm->sges+r;
    sge->addr=(uintptr_t)reqs[r]->send.data;
    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr.rdma.remote_addr = comm->useCtsOffload ? 0xdeadbeef : slots[r].addr;
    wr->next = wr + 1;
    wr_id += (uint64_t)(slot & 0xff) << (r*8);
    wr->wr_id = wr_id;
#ifdef NCCL_ENABLE_NET_PROFILING
    reqs[r]->pInfo[0].nEventHandles = 0;
#endif

    // Every request is chunked equally across all QPs that are used to transfer
    // the request (in case of a single QP, the chunk is the size of the request).
    // The chunk size of each request determined solely by the send size and the
    // number of QPs used to transfer the request. If the send size is not big
    // enough, starting from some QP there might be no data left to send and the
    // length will be zeroed.
    sge->length = DIVUP(DIVUP(reqs[r]->send.size, nqps), IB_WRITE_CHUNK_ALIGNMENT) * IB_WRITE_CHUNK_ALIGNMENT;
    wr->sg_list = sge;
    wr->num_sge = 1;
  }

  // For ID-based matching scheme, immData carries the request ID.
  // For index-based matching scheme, immData carries the send size:
  // - nreqs == 1
  //      It's the send size.
  // - nreqs > 1
  //      Send size is still sent but receiver ignores it since the sizes are
  //      written to directly to remote completion records array
  struct ibv_send_wr* lastWr = comm->wrs+nreqs-1;
  if (!useWriteOp) {
    uint32_t immData = comm->base.recvMatchingScheme == BY_ID ? (uint32_t)(reqs[0]->id % UINT32_MAX) : reqs[0]->send.size;

    
    if (nreqs > 1 || (!(comm->base.remOooRq && comm->base.localOooRq) && comm->ar && reqs[0]->send.size > IbCastArThreshold)) {
      // When Adaptive Routing is enabled, send the bulk of the data first as an
      // RDMA Write.
      lastWr++;
      memset(lastWr, 0, sizeof(struct ibv_send_wr));
      if (nreqs > 1) {
        // Write remote sizes array
        lastWr->wr.rdma.remote_addr = comm->remCmplsRecords.addr + slot*sizeof(struct ncclIbRequestCompletionRecord);
        lastWr->num_sge = 1;
      }
    }
    lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    lastWr->imm_data = immData;
  }
  lastWr->wr_id = wr_id;
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  uint32_t sendOffsets[NCCL_NET_IB_MAX_RECVS] = {0};
  int qpIndex = -1;
  ncclIbQp* qp = NULL;
  const int align = 128;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(IbCastCommBaseGetQpForRequest(&comm->base, startQpIndex, wrrSched ? 0 : i, &qp, &qpIndex));

    TRACE(NCCL_NET, "NET/IB: %s: Posting send (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld) on QP (qp_num=%u, devIndex=%d, qpIndex=%d)", __func__, reqs[0], reqs[0]->base, reqs[0]->id, slot, nreqs, wr_id, qp->qp->qp_num, qp->devIndex, qpIndex);

    // Selective retransmission
    if (comm->base.resiliency && reqs[0]->send.sentData[qpIndex] == true) {
      for (int r=0; r<nreqs; r++) {
        comm->wrs[r].sg_list->addr += comm->wrs[r].sg_list->length;
        comm->wrs[r].wr.rdma.remote_addr += comm->wrs[r].sg_list->length;
      }
      INFO(NCCL_NET, "NET/IB: %s: Skipping retransmission on QP index %d (req=%p, slot=%d) as it was already delivered.", __func__, qpIndex, reqs[0], slot);
      continue;
    }
    
    int devIndex = qp->devIndex;
    for (int r=0; r<nreqs; r++) {
      // Track this event for completion
      //IbCastAddEvent(reqs[r], devIndex);

      // Select proper rkey (needed even for 0-size send)
      comm->wrs[r].wr.rdma.rkey = comm->useCtsOffload ? 0xbade : slots[r].rkeys[qp->remDevIdx];

      int chunkSize, length;
      if ((nqps > 1) && reqs[r]->desc.parms.enable && comm->base.qpTxSchedInit) {
        int weightedSendSize = (int) (((double) reqs[r]->send.size) *
                                      comm->base.qpTxSched[qpIndex].weight);
        chunkSize = (weightedSendSize / align) * align;
        if (i == (nqps-1))
          length = std::max((int)(reqs[r]->send.size-sendOffsets[r]), chunkSize);
        else
          length = chunkSize;
      } else {
        chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
        length = std::min((int)(reqs[r]->send.size-sendOffsets[r]), chunkSize);
      }

      // Check the data left to send. If the send is too small, it might be
      // that on the current QP there is no data left to be sent.
      if (length <= 0) {
        comm->wrs[r].sg_list = NULL;
        comm->wrs[r].num_sge = 0;
        length = 0;
      } else {
        // Select proper lkey
        comm->sges[r].lkey = reqs[r]->send.lkeys[devIndex];
        comm->sges[r].length = length;
        comm->wrs[r].sg_list = comm->sges+r;
        comm->wrs[r].num_sge = 1;
      }

      if (r == (nreqs - 1)) {
        struct ncclIbRemapWrId *remapWrId;
        NCCLCHECK(IbCastQpSchedGetRemap(&comm->base, wr_id, qpIndex, &remapWrId));
        lastWr->wr_id = (uint64_t) remapWrId;
        /* save tx data for measuring QP performance */
        remapWrId->parms = reqs[r]->desc.parms;
        remapWrId->tx.bytes = length;
        nowNs = 0;
        if (reqs[r]->desc.parms.enable && ((nqps > 1) || reqs[r]->desc.parms.doWrr)) {
          struct timespec txStartTime;
          if (!clock_gettime(CLOCK_MONOTONIC, &txStartTime))
            nowNs = TIMESPEC_TO_NSEC(&txStartTime);
          remapWrId->tx.startTimeNs = nowNs;
        }
      }
    }

    if (nreqs > 1) {
      // Populating the correct gather information based on the device and
      // slot used.
      // Note that the lkey is already correct from the initialization phase.
      lastWr->sg_list = &(comm->devs[devIndex].sge);
      lastWr->sg_list[0].addr = (uint64_t)(comm->remCmplsRecords.elems[slot]);
      lastWr->sg_list[0].length = nreqs*sizeof(int);
      // Populate the correct RKey based on the device used
      lastWr->wr.rdma.rkey = comm->remCmplsRecords.rkeys[devIndex];
    }

    struct ibv_send_wr* bad_wr;
#ifdef NCCL_ENABLE_NET_PROFILING
    // QP profiling loop
    for (int r=0; r<nreqs; r++) {
      // Store the qpIndex for this request
      int nEventHandles = reqs[r]->pInfo[0].nEventHandles;
      assert(nEventHandles < MAX_QPS_PER_REQ);
      reqs[r]->pInfo[0].qpIndex[nEventHandles] = qpIndex;
      // Store info for profiler
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      reqs[r]->pInfo[0].data.type = ncclProfileQp;
      reqs[r]->pInfo[0].data.qp.device = devIndex;
      reqs[r]->pInfo[0].data.qp.wr_id = comm->wrs[r].wr_id;
      reqs[r]->pInfo[0].data.qp.opcode = comm->wrs[r].opcode;
      reqs[r]->pInfo[0].data.qp.qpNum = qp->qp->qp_num;
      reqs[r]->pInfo[0].data.qp.length = comm->sges[r].length;
      void* pHandle = reqs[r]->pInfo[0].pHandle;
      NCCLCHECK(IbCastProfilerFunction(&reqs[r]->pInfo[0].qpEventHandles[nEventHandles], ncclProfilerNetEventStart, pHandle, pluginId, &reqs[r]->pInfo[0].data));
      reqs[r]->pInfo[0].nEventHandles++;
    }
#endif
#ifdef ENABLE_TRACE
    for (int r = 0; r < nreqs; r++) {
      TRACE(NCCL_NET, "NET/IB: %s: Posting send work request on QP (qpn=%u, devIndex=%d, qpIndex=%d) (slot=%d, req[r=%d]=%p)", __func__, qp->qp->qp_num, qp->devIndex, qpIndex, slot, r, reqs[r]);
    }
    int wrIdx = 0;
    char wrStr[1024];
    struct ibv_send_wr* currWr = comm->wrs;
    while (currWr != NULL) {
      NCCLCHECK(IbCastPrintWr(currWr, wrStr));
      TRACE(NCCL_NET, "NET/IB: %s: slot=%d, wrIdx[%d], %s", __func__, slot, wrIdx, wrStr);
      wrIdx++;
      currWr = currWr->next;
    }
#endif // ENABLE_TRACE
    NCCLCHECK(wrap_ibv_post_send(qp->qp, comm->wrs, &bad_wr));

    // Update the send offset and addresses for the next QP according to the
    // actual data size that was sent on the current QP, for every request
    for (int r=0; r<nreqs; r++) {
      int chunkSize;

      if (reqs[r]->desc.parms.enable && comm->base.qpTxSchedInit) {
        int weightedSendSize = (int) (((double) reqs[r]->send.size) *
                                      comm->base.qpTxSched[qpIndex].weight);
        chunkSize = (weightedSendSize / align) * align;
        if (i == (nqps-1))
          chunkSize = std::max((int)(reqs[r]->send.size-sendOffsets[r]), chunkSize);
      } else
        chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;

      sendOffsets[r] += chunkSize;
      comm->sges[r].addr += chunkSize;
      comm->wrs[r].wr.rdma.remote_addr += chunkSize;

      TRACE(NCCL_VERBS, "Posted send wr_id=%lu, wr_indx=%d, qp_num=%d, src_nic=%d, dst_nic=%d, dlid=%d, opcode=%d, send_flags=%d, imm_data=%d, remote_addr=%lx, rkey=%x, length=%d, lkey=%x",
        comm->wrs[r].wr_id, r, qp->qp->qp_num, comm->devs[qp->devIndex].base.ibDevN , comm->base.remDevs[qp->remDevIdx].ibv_dev_index, comm->base.remDevs[qp->remDevIdx].lid,
        comm->wrs[r].opcode, comm->wrs[r].send_flags, comm->wrs[r].imm_data, comm->wrs[r].wr.rdma.remote_addr,
        comm->wrs[r].wr.rdma.rkey,comm->wrs[r].sg_list ? comm->wrs[r].sg_list->length : 0, comm->wrs[r].sg_list ? comm->wrs[r].sg_list->lkey : 0);
    }
  }

  TRACE(NCCL_NET, "NET/IB: %s: Send request posted (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld)", __func__, reqs[0], reqs[0]->base, reqs[0]->id, slot, nreqs, wr_id);
  if (nowNs) {
    if (comm->base.nextQpTxSchedUpdateNs == 0)
      comm->base.nextQpTxSchedUpdateNs = nowNs + comm->base.schedParms.updateInterval;
    else if (nowNs >= comm->base.nextQpTxSchedUpdateNs) {
      IbCastQpSchedUpdateTx(&comm->base);
      comm->base.nextQpTxSchedUpdateNs = nowNs + comm->base.schedParms.updateInterval;
    }
    if (comm->base.schedParms.logEnable) {
      if (comm->base.nextSchedLogNs == 0)
        comm->base.nextSchedLogNs = nowNs + comm->base.schedParms.logInterval;
      else if (nowNs >= comm->base.nextSchedLogNs) {
        IbCastLogSched(comm);
        comm->base.nextSchedLogNs = nowNs + comm->base.schedParms.logInterval;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t IbCastIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  bool useWriteOp = (IbCastAinicRoce &&(*request == (void *)NCCL_NET_OPTIONAL_RECV_COMPLETION)) ? true : false;
  if (comm->base.ready == 0) {
    WARN("NET/IB: IbCastIsend() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  NCCLCHECK(IbCastStatsCheckFatalCount(&comm->base.stats,__func__));

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandle;
  // Wait for the receiver to have posted the corresponding receive
  int nreqs = 1;
  volatile struct ncclIbSendFifo* slots;

  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  struct ncclIbRequest** reqs = comm->sendReqs[slot];
  if (!comm->useCtsOffload) {
    slots = comm->ctsFifo[slot];
    uint64_t idx = comm->base.fifoHead+1;
    if (slots[0].idx != idx) { *request = NULL; return ncclSuccess; }
    nreqs = slots[0].nreqs;
    // Wait until all data has arrived
    for (int r=1; r<nreqs; r++) while(slots[r].idx != idx);
    std::atomic_thread_fence(std::memory_order_seq_cst); // order the nreqsPtr load against tag/rkey/addr loads below
  }


  for (int r=0; r<nreqs; r++) {
    if (!comm->useCtsOffload) {
      if (reqs[r] != NULL || slots[r].tag != tag) continue;
    }

    if (size > slots[r].size) size = slots[r].size;
    // Sanity checks
    if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkeys[0] == 0) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union ncclSocketAddress addr;
      ncclSocketGetAddr(&comm->base.sock, &addr);
      WARN("NET/IB : req %d/%d tag %x peer %s posted incorrect receive info: size %ld addr %lx rkeys[0]=%x",
        r, nreqs, tag, ncclSocketToString(&addr, line), slots[r].size, slots[r].addr, slots[r].rkeys[0]);
      return ncclInternalError;
    }

    struct ncclIbRequest* req;
    NCCLCHECK(IbCastGetRequest(&comm->base, &req));
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
#ifdef NCCL_ENABLE_NET_PROFILING
    req->pInfo[0].pHandle = phandle;
#endif

    bool wrrSched = false;
    int nqps = 0;
    int startQpIndex = 0;
    int j;
    for (j = 0; j < nreqs; j++) {
      if (reqs[j] != NULL) {
        break;
      }
    }
    if (j == nreqs) {
      IbCastUpdateSchedParmsTry(&comm->base, nreqs, size);
      req->desc.parms = comm->base.schedParms;
      nqps = IbCastQpSchedGetEffectiveTxNqps(req, &startQpIndex, &wrrSched);
      req->desc.nqps = nqps;
      req->desc.startQpIndex = startQpIndex;
      req->desc.wrrSched = wrrSched;
    } else {
      req->desc.parms = reqs[j]->desc.parms;
      nqps = req->desc.nqps = reqs[j]->desc.nqps;
      startQpIndex = req->desc.startQpIndex = reqs[j]->desc.startQpIndex;
      wrrSched = req->desc.wrrSched = reqs[j]->desc.wrrSched;
    }

    // Populate events
    int qpIndex = -1;
    ncclIbQp* qp = NULL;
    for (int i = 0; i < nqps; i++) {
      NCCLCHECK(IbCastCommBaseGetQpForRequest(&comm->base, startQpIndex, i, &qp, &qpIndex));
      IbCastAddEvent(req, qp->devIndex);
    }

    // Store all lkeys
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
    }

    // In case the sender will write the size of the send directly to the
    // receiver's memory, prepare the source buffer which will hold the sizes
    // and be sent to the receiver.
    comm->remCmplsRecords.elems[slot][r] = req->send.size;

    TRACE(NCCL_NET, "NET/IB: %s: Send request created (req=%p, comm=%p, id=%ld, slot=%d, reqIdx=%d, nreqs=%d, tag=%x, size=%ld, data=0x%016" PRIx64 ", mhandle=%p, size=%ld)", __func__, req, req->base, req->id, slot, r, nreqs, tag, size, (uint64_t)data, mhandle, size);

    *request = reqs[r] = req;

    comm->sendReqsCnt[slot]++;
    // If this is a multi-recv, send only when all requests have matched.
    if (comm->sendReqsCnt[slot] < nreqs) return ncclSuccess;

    TIME_START(0);
    NCCLCHECK(IbCastMultiSend(comm, slot, nqps, startQpIndex, wrrSched, useWriteOp));

    comm->base.fifoHead++;
    TIME_STOP(0);
    return ncclSuccess;
  }

  *request = NULL;
  return ncclSuccess;
}

ncclResult_t IbCastPostFifo(struct ncclIbRecvComm* comm, struct ncclIbRequest* req, int slot, int n) {
  ncclIbQp* ctsQp = NULL;;
  int rid = req ? req->id : comm->base.fifoHead;
  NCCLCHECK(IbCastRecvCommGetQpForCts(comm, rid, &ctsQp));

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr.rdma.remote_addr = comm->remCtsFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbSendFifo);

  // Lookup the correct rkey
  wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].rkey;

  // Populating the correct gather information based on the device and user
  // provided information
  struct ncclIbSendFifo* localElem = comm->remCtsFifo.elems[slot];
  wr.sg_list = &(comm->devs[ctsQp->devIndex].sge);
  wr.sg_list[0].addr = (uint64_t)localElem;
  wr.sg_list[0].length = comm->useCtsOffload ? MAX_INLINE_DATA_SIZE : n*sizeof(struct ncclIbSendFifo);
  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remCtsFifo.flags; // IBV_SEND_INLINE

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
  if ((IbCastAinicRoce && (slot == ctsQp->ctsQpSlot))
     || (!IbCastAinicRoce && (slot == ctsQp->devIndex || comm->base.resiliency))) {
    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = slot;
    IbCastAddEventCTS(req, ctsQp->devIndex);
  }

  TRACE(NCCL_NET, "NET/IB: %s: Posting a CTS (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld, opcode=%d, send_flags=%d, qp_num=%u)", __func__, req, req->base, rid, slot, req->nreqs, wr.wr_id, wr.opcode, wr.send_flags, ctsQp->qp->qp_num);

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(ctsQp->qp, &wr, &bad_wr));

  TRACE(NCCL_NET, "NET/IB: %s: CTS posted (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, wr_id=%ld, opcode=%d, send_flags=%d, qp_num=%u)", __func__, req, req->base, rid, slot, req->nreqs, wr.wr_id, wr.opcode, wr.send_flags, ctsQp->qp->qp_num);

  return ncclSuccess;
}

ncclResult_t IbCastIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  uint16_t rxReqIndex = 0;
  ncclResult_t res = ncclSuccess;
  bool netOptRecvCompletionEnabled = false;
  if (comm->base.ready == 0) {
    WARN("NET/IB: IbCastIrecv() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  if (n > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;
  NCCLCHECK(IbCastStatsCheckFatalCount(&comm->base.stats,__func__));
  if (IbCastAinicRoce) {
    if (*request == (void *)NCCL_NET_OPTIONAL_RECV_COMPLETION) {
      netOptRecvCompletionEnabled = true;
    }
  }

  struct ncclIbRequest* req = NULL;
  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  if (!netOptRecvCompletionEnabled) {
    NCCLCHECK(IbCastGetRequest(&comm->base, &req));
    rxReqIndex = (uint16_t) (req - comm->base.reqs);
    req->id = comm->base.fifoHead;
    req->type = NCCL_NET_IB_REQ_RECV;
    req->sock = &comm->base.sock;
    req->nreqs = n;
    if (comm->base.resiliency) {
      // When resiliency is enabled, a recv request can be served by any device.
      for (int devIndex = 0; devIndex < comm->base.vProps.ndevs; devIndex++) {
        req->devBases[devIndex] = IbCastGetNetCommDevBase(&comm->base, devIndex);
      }
    }
    TRACE(NCCL_NET, "NET/IB: %s: Recv request created (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d, tag[0]=%x)", __func__, req, req->base, req->id, slot, n, tags[0]);

  #ifdef NCCL_ENABLE_NET_PROFILING
    for (int r = 0; r < n && phandles; r++) req->pInfo[r].nEventHandles = 0;
  #endif

    // Store the request in a table for easy retrieval by ID.
    comm->recvReqs[req->id % NET_IB_MAX_REQUESTS] = req;

    TIME_START(1);
    const int nqps = IbCastCommBaseGetNqpsPerRequest(&comm->base);
    int qpIndex = -1;
    ncclIbQp* qp = NULL;
    for (int i = 0; i < nqps; i++) {
      NCCLCHECK(IbCastCommBaseGetQpForRequest(&comm->base, req->id, i, &qp, &qpIndex));
      IbCastAddEvent(req, qp->devIndex);
      if (comm->prepostReceiveWorkRequests) {
        continue;
      }
      // Post receive work request on the QP
      if (comm->base.rxPosts[qpIndex] < NET_IB_MAX_REQUESTS) {
        comm->ibRecvWorkRequest.wr_id = qpIndex;
        NCCLCHECK(IbCastPostRecvWorkRequest(qp->qp, &comm->ibRecvWorkRequest));
        comm->base.rxPosts[qpIndex]++;
      }
  #ifdef NCCL_ENABLE_NET_PROFILING
      // Start a QP event for every request in the multirecv and every qp
      for (int r = 0; r < n; r++) {
        int nEventHandles = req->pInfo[r].nEventHandles;
        assert(nEventHandles < MAX_QPS_PER_REQ);
        req->pInfo[r].qpIndex[nEventHandles] = qpIndex;
        // Store info for profiler
        int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
        req->pInfo[r].data.type = ncclProfileQp;
        req->pInfo[r].data.qp.device = qp->devIndex;
        req->pInfo[r].data.qp.wr_id = comm->ibRecvWorkRequest.wr_id;
        req->pInfo[r].data.qp.qpNum = qp->qp->qp_num;
        NCCLCHECK(IbCastProfilerFunction(&req->pInfo[r].qpEventHandles[nEventHandles], ncclProfilerNetEventStart, phandles[r], pluginId, &req->pInfo[r].data));
        req->pInfo[r].nEventHandles++;
      }
  #endif
    }
    TIME_STOP(1);

    req->recv.aggSize = 0;
    req->recv.cmplsRecords = &comm->cmplsRecords[slot];
    memset(req->recv.cmplsRecords->sizes, 0, sizeof(int)*n);
    memset(req->recv.cmplsRecords->completions, 0, sizeof(req->recv.cmplsRecords->completions));
  }

  struct ncclIbSendFifo* localElem = comm->remCtsFifo.elems[slot];
  for (int i=0; i<n; i++) {
    struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandles[i];
    if (comm->useCtsOffload) {
      struct ncclIbSendFifoCtsInline* localElemCtsInline = (struct ncclIbSendFifoCtsInline*)&localElem[i];
      localElemCtsInline[i].addr = (uint64_t)data[i];
      localElemCtsInline[i].rkeys[0] = mhandleWrapper->mrs[0]->rkey;
      localElemCtsInline[i].nreqs = n;
      localElemCtsInline[i].size = sizes[i]; // Sanity/Debugging
      localElemCtsInline[i].tag = tags[i];
    } else {
      localElem[i].addr = (uint64_t)data[i];
      // Send all applicable rkeys
      for (int j = 0; j < comm->base.vProps.ndevs; j++) {
        localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;
      }
      localElem[i].nreqs = n;
      localElem[i].size = sizes[i]; // Sanity/Debugging
      localElem[i].tag = tags[i];
      localElem[i].idx = comm->base.fifoHead+1;
      localElem[i].rxReqIndex = rxReqIndex;
    }
  }

  // Post to FIFO to notify sender
  TIME_START(2);
  NCCLCHECKGOTO(IbCastPostFifo(comm, req, slot, n), res, err);
  comm->base.fifoHead++;
  TIME_STOP(2);

  *request = req;
  return res;
err:
  if (req) {
      IbCastFreeRequest(req);
  }
  return res;
}

ncclResult_t IbCastIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  int last = -1;
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  if (comm->flushEnabled == 0 || last == -1) return ncclSuccess;

  // Only flush once using the last non-zero receive
  struct ncclIbRequest* req;
  NCCLCHECK(IbCastGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  struct ncclIbMrHandle* mhandle = (struct ncclIbMrHandle*) mhandles[last];

  // We don't know which devIndex the recv was on, so we flush on all devices
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (req - comm->base.reqs) + NCCL_IB_FLUSH_REQ_WR_ID_OFFSET;

    wr.wr.rdma.remote_addr = (uint64_t)data[last];
    wr.wr.rdma.rkey = mhandle->mrs[i]->rkey;
    wr.sg_list = &comm->devs[i].gpuFlush.sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    TRACE(NCCL_NET, "NET/IB: %s: Posting a flush request (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);
    TIME_START(4);
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, &wr, &bad_wr));
    TIME_STOP(4);

    IbCastAddEvent(req, i);

    TRACE(NCCL_NET, "NET/IB: %s: Flush request posted (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);
  }

  *request = req;
  return ncclSuccess;
}

#define HCA_NAME(req, index) ((req)->devBases[(index)]->pd->context->device->name)

#ifdef NCCL_ENABLE_NET_PROFILING
static int getReqQpIndex(struct ncclIbRequest* req, int request, int qpNumber) {
  for (int i = 0; i < MAX_QPS_PER_REQ; i++) {
    int qpIndex = req->pInfo[request].qpIndex[i];
    if (req->base->qps[qpIndex].qp->qp_num == qpNumber) return i;
  }
  return 0;
}
#endif

static inline ncclResult_t IbCastRequestRetrieveFromCompletion(struct ncclIbNetCommBase* base, ibv_wc* wc, ncclIbRequest** req) {
  assert(req != NULL);
  assert(wc != NULL);

  // In case of a completion with error, there is no guarantee that all fields
  // of the completion are valid.
  assert(wc->status == IBV_WC_SUCCESS);

  TRACE(NCCL_NET, "NET/IB: %s: Retrieving a %s request (wr_id=%ld, opcode=%s)", __func__, base->isSend ? "send" : "recv", wc->wr_id, ibvWcOpcodeStr(wc->opcode));

  if (!base->isSend && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM && base->recvMatchingScheme == BY_ID) {
    TRACE(NCCL_NET, "NET/IB: %s: Retrieving a receive request (wr_id=%ld, opcode=%s, imm_data=%d, byte_len=%d)", __func__, wc->wr_id, ibvWcOpcodeStr(wc->opcode), be32toh(wc->imm_data), wc->byte_len);
    struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)base;
    *req = recvComm->recvReqs[be32toh(wc->imm_data) % NET_IB_MAX_REQUESTS];
  } else if (!base->isSend && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM && base->recvMatchingScheme == BY_INDEX) {
    *req = &base->reqs[((wc->imm_data >> WR_IMM_RX_REQ_IDX_SHIFT) & WR_IMM_RX_REQ_IDX_MASK)];
  } else if (!base->isSend && wc->opcode == IBV_WC_RDMA_READ) { // Flush request completion
    NCCLCHECK(IbCastRequestRetrieveAsIndex(base->reqs, (wc->wr_id - NCCL_IB_FLUSH_REQ_WR_ID_OFFSET), req));
  } else if (!base->isSend) {
    struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)base;
    *req = recvComm->recvReqs[wc->wr_id];
  } else {
    struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)base;
    struct ncclIbRemapWrId *remapWrId = (struct ncclIbRemapWrId *) wc->wr_id;
    assert(remapWrId != NULL);
    assert(remapWrId->state == NCCL_NET_IB_REMAP_USED);
    // On the sender side, the lower 8 bits of wr_id are used to retrieve the
    // request, since in multi-send case, multiple IDs are encoded in the same
    // wr_id.,
    *req = sendComm->sendReqs[remapWrId->origWrId & 0xff][0];
    if (remapWrId->parms.enable) IbCastQpSchedUpdateTxStats(remapWrId, base);
    IbCastQpSchedFreeRemap(remapWrId);
  }
  TRACE(NCCL_NET, "NET/IB: %s: Retrieved a %s request (req=%p, comm=%p, id=%ld, type=%s, wc.wr_id=%ld, wc.opcode=%s, wc.imm_data=%d, wc.byte_len=%d, wc.qp_num=%u)", __func__, base->isSend ? "send" : "recv", *req, (*req)->base, (*req)->id, IbCastReqTypeStr[(*req)->type], wc->wr_id, ibvWcOpcodeStr(wc->opcode), be32toh(wc->imm_data), wc->byte_len, wc->qp_num);
  return ncclSuccess;
}

static inline bool IbCastRequestIsComplete(struct ncclIbRequest *request) {
  bool complete = (request->events[0] == 0 && request->events[1] == 0 && request->events[2] == 0 && request->events[3] == 0);
  if (!complete && request->base->resiliency) {
    NCCLCHECK(IbCastResiliencyRequestIsComplete(request, &complete));
  }
  return complete;
}

static inline ncclResult_t IbCastRequestComplete(struct ncclIbRequest* r, int* done, int* sizes) {
  TRACE(NCCL_NET, "NET/IB: %s: %s request completed (req=%p, comm=%p, id=%ld, type=%s)", __func__, r->base->isSend ? "Send" : "Recv", r, r->base, r->id, IbCastReqTypeStr[r->type]);
  *done = 1;
  if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
    TRACE(NCCL_NET, "NET/IB: %s: Recv request completed (req=%p, comm=%p, id=%ld, type=%s, nreqs=%d)", __func__, r, r->base, r->id, IbCastReqTypeStr[r->type], r->nreqs);
    int *sizesToReport = (r->nreqs > 1 || r->recv.cmplsRecords->sizes[0] > 0) ? r->recv.cmplsRecords->sizes : &(r->recv.aggSize);
    for (int i=0; i<r->nreqs; i++) {
      sizes[i] = sizesToReport[i];
#ifdef NCCL_ENABLE_NET_PROFILING
      for (int j = 0; j < r->pInfo[i].nEventHandles; j++) {
        NCCLCHECK(IbCastProfilerFunction(&r->pInfo[i].qpEventHandles[j], ncclProfilerNetEventStop, NULL, 0, NULL));
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
        NCCLCHECK(IbCastProfilerFunction(&r->pInfo[0].qpEventHandles[j], ncclProfilerNetEventStop, NULL, 0, NULL));
      }
  #endif
    }
    int slot = r->id % NET_IB_MAX_REQUESTS;
    struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)r->base;
    sendComm->sendReqsCnt[slot]--;
    if (sendComm->sendReqsCnt[slot] == 0) {
      // Only after completing the last send of a multi-recv, allow accepting
      // following send requests on the same slot.
      memset(&sendComm->sendReqs[slot], 0, sizeof(sendComm->sendReqs[slot]));
    }
  }
  // Stop all remaining Qp events for this event
  NCCLCHECK(IbCastFreeRequest(r));
  return ncclSuccess;
}

// Log the details of a completion with error. The provided devIndex is the index
// of the IB device on which the completion was received.
static ncclResult_t IbCastLogCompletionWithError(struct ncclIbNetCommBase* commBase, struct ibv_wc* wc, int devIndex) {
  struct ncclIbNetCommDevBase* devBase = IbCastGetNetCommDevBase(commBase, devIndex);
  char localGidString[INET6_ADDRSTRLEN] = "";
  char remoteGidString[INET6_ADDRSTRLEN] = "";
  const char* localGidStr = NULL, *remoteGidStr = NULL;
  if (devBase->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
    localGidStr = ibvGetGidStr(&devBase->gidInfo.localGid, localGidString, sizeof(localGidString));
    remoteGidStr = ibvGetGidStr(&commBase->remDevs[devIndex].remoteGid, remoteGidString, sizeof(remoteGidString));
  }

  char sockStr[SOCKET_NAME_MAXLEN+1];
  union ncclSocketAddress addr;
  ncclSocketGetAddr(&commBase->sock, &addr);
  ncclSocketToString(&addr, sockStr);
  char *hcaName = devBase->pd->context->device->name;
  WARN("NET/IB: Got completion from peer %s with status=%s(%d) opcode=%s(%d) vendor_err=%u %s%s%s%s hca %s",
      sockStr, ibvWcStatusStr(wc->status), wc->status,
      ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->vendor_err,
      localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
  return ncclSuccess;
}

static inline ncclResult_t IbCastCompletionEventProcess(struct ncclIbNetCommBase* commBase, struct ibv_wc* wc, int devIndex) {
  union ncclSocketAddress addr;
  ncclSocketGetAddr(&commBase->sock, &addr);

  struct ncclIbRequest* req = NULL;
  NCCLCHECK(IbCastRequestRetrieveFromCompletion(commBase, wc, &req));
  if (req == NULL) {
    WARN("NET/IB: %s: %s comm could not retreive a request found for a successful completion (comm=%p, wc.wr_id=%ld, opcode=%d, qp_num=%u)", __func__, commBase->isSend ? "Send" : "Recv", commBase, wc->wr_id, wc->opcode, wc->qp_num);
    return ncclInternalError;
  }

  #ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
  TRACE(NCCL_NET, "Got completion from peer %s with status=%d opcode=%d len=%u wr_id=%lu r=%p type=%d events={%d,%d,%d,%d}, devIndex=%d",
    ncclSocketToString(&addr, line), wc->status, wc->opcode,wc->byte_len, wc->wr_id, req, req->type, req->events[0], req->events[1], req->events[2], req->events[3], devIndex);
  #endif

  if (commBase->isSend) {
    if (req->type != NCCL_NET_IB_REQ_SEND) {
      WARN("NET/IB: %s: Sender expected a 'send' request but got '%s' (req=%p, comm=%p, id=%ld, wc.wr_id=%ld, wc.opcode=%s(%d), wc.qp_num=%u)", __func__, IbCastReqTypeStr[req->type], req, commBase, req->id, wc->wr_id, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num);
      return ncclInternalError;
    }
    struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)commBase;
    struct ncclIbRequest* sendReq = NULL;
    int slot = req->id % NET_IB_MAX_REQUESTS;
    for (int j = 0; j < req->nreqs; j++) {
      sendReq = sendComm->sendReqs[slot][j];
      if (!commBase->resiliency && (sendReq->events[devIndex] <= 0)) {
        WARN("NET/IB: sendReq(%p)->events={%d,%d,%d,%d}, devIndex=%d, reqIdx=%d <= 0", sendReq, sendReq->events[0], sendReq->events[1], sendReq->events[2], sendReq->events[3], devIndex, j);
        return ncclInternalError;
      }
      sendReq->events[devIndex]--;
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a send request (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)", __func__, sendReq, sendReq->base, sendReq->id, devIndex, wc->qp_num);
#ifdef NCCL_ENABLE_NET_PROFILING
      // Stop Qp event for sendReq
      int qpIndex = getReqQpIndex(sendReq, j, wc->qp_num);
      NCCLCHECK(IbCastProfilerFunction(&sendReq->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL, 0, NULL));
#endif
    }
  } else {
    if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      if (req->type == NCCL_NET_IB_REQ_UNUSED && commBase->resiliency) {
        INFO(NCCL_NET, "NET/IB: %s: Receiver got a completion for a data transfer but retrieved an 'unused' request (req=%p, comm=%p, id=%ld, wc.status=%s(%d), wc.wr_id=%ld, wc.imm_data=%d, wc.opcode=%s(%d), wc.qp_num=%u)", __func__, req, commBase, req->id, ibvWcStatusStr(wc->status), wc->status, wc->wr_id, be32toh(wc->imm_data), ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num);
        return ncclSuccess;
      }
      if (req->type != NCCL_NET_IB_REQ_RECV && !commBase->resiliency) {
        WARN("NET/IB: %s: Receiver expected a 'recv' request but got '%s' (req=%p, comm=%p, id=%ld, wc.wr_id=%ld, wc.status=%s(%d) wc.opcode=%s(%d), wc.qp_num=%u)", __func__, IbCastReqTypeStr[req->type], req, req->base, req->id, wc->wr_id, ibvWcStatusStr(wc->status), wc->status, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num);
        return ncclInternalError;
      }
      if (req->nreqs == 1) {
        if (commBase->recvMatchingScheme == BY_INDEX) {
          req->recv.cmplsRecords->sizes[0] = (wc->imm_data & WR_IMM_SIZE_MASK);
        } else if (req->recv.cmplsRecords->sizes[0] == 0) {
          req->recv.aggSize+= wc->byte_len;
        }
      }
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a recv request (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)", __func__, req, req->base, req->id, devIndex, wc->qp_num);
      struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)commBase;
      if (recvComm->prepostReceiveWorkRequests) {
        // Post another receive work request on the QP
        ncclIbQp* qp = NULL;
        int qpIndex = -1;
        NCCLCHECK(IbCastCommBaseGetQpByQpNum(commBase, devIndex, wc->qp_num, &qp, &qpIndex));
        req->recv.cmplsRecords->completions[qpIndex] = 1;
        IbCastPostRecvWorkRequest(qp->qp, &recvComm->ibRecvWorkRequest);
      }
      if (commBase->recvMatchingScheme == BY_INDEX) {
        if (wc->imm_data & WR_IMM_SPLIT_DATA_FLAG) {
          req->events[devIndex]--;
        } else { // Single QP send path
                // The receiver posted recvs on all QPs, but the sender used a single QP
                // (no split-data), so only one QP received the RDMA Write with IMM.
                // Recvs posted on other QPs won't complete for this request.
                // Reset events to only wait for signaled CTS completions.
          for (int d = 0; d < NCCL_IB_MAX_DEVS_PER_NIC; d++) {
            req->events[d] = req->ctsEvents[d];
          }
        }
      } else {
        req->events[devIndex]--;
      }
    } else if (wc->opcode == IBV_WC_RDMA_READ) {
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a flush request (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)", __func__, req, req->base, req->id, devIndex, wc->qp_num);
      req->events[devIndex]--;
    } else if (wc->opcode == IBV_WC_RDMA_WRITE) {
      req->events[devIndex]--;
      req->ctsEvents[devIndex]--;
      // This is a CTS completion
      TRACE(NCCL_NET, "NET/IB: %s: Got completion for a CTS (req=%p, comm=%p, id=%ld, devIndex=%d, qp_num=%u)", __func__, req, req->base, req->id, devIndex, wc->qp_num);
      if (req->type == NCCL_NET_IB_REQ_UNUSED) {
        INFO(NCCL_NET, "NET/IB: %s: Receiver got a completion for a CTS but retrieved an 'unused' request (req=%p, comm=%p, id=%ld, wc.wr_id=%ld, wc.opcode=%s(%d), wc.qp_num=%u, wc.imm_data=%d)", __func__, req, req->base, req->id, wc->wr_id, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num, be32toh(wc->imm_data));
        return ncclSuccess;
      }
    } else {
      WARN("NET/IB: %s: Unknown completion (req=%p, comm=%p, id=%ld, devIndex=%d, req->type=%s, wc.wr_id=%ld, wc.opcode=%s(%d), wc.qp_num=%u, wc.imm_data=%d)", __func__, req, commBase, req ? req->id : -1, devIndex, IbCastReqTypeStr[req->type], wc->wr_id, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->qp_num, be32toh(wc->imm_data));
      return ncclInternalError;
    }
#ifdef NCCL_ENABLE_NET_PROFILING
    // Stop Qp event for workFifo
    for (int j = 0; j < req->nreqs; j++) {
      int qpIndex = getReqQpIndex(req, j, wc->qp_num);
      NCCLCHECK(IbCastProfilerFunction(&req->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL, 0, NULL));
    }
#endif
  }
  return ncclSuccess;
}

#define NCCL_CQ_POLL_MAX_EVENT        16

ncclResult_t IbCastTest(void* request, int* done, int* sizes) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;

  if (r->base->resiliency && r->base->resiliency->inProgress) {
    NCCLCHECK(IbCastResiliencyProgress(r->base->resiliency));
  }

  int totalWrDone = 0;
  int wrDone = 0;
  struct ibv_wc wcs[NCCL_CQ_POLL_MAX_EVENT];
  int cqMaxPollEvent = 4;
  if (IbCastAinicRoce) {
    cqMaxPollEvent = NCCL_CQ_POLL_MAX_EVENT;
  }
  do {
    NCCLCHECK(IbCastStatsCheckFatalCount(&r->base->stats,__func__));
    if (IbCastRequestIsComplete(r)) {
      NCCLCHECK(IbCastRequestComplete(r, done, sizes));
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
      NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, cqMaxPollEvent, wcs, &wrDone));
      if (wrDone == 0) { TIME_CANCEL(3); } else { TIME_STOP(3); }
      if (wrDone == 0) continue;
      totalWrDone += wrDone;
      for (int w=0; w<wrDone; w++) {
        struct ibv_wc *wc = wcs+w;
        if (wc->status != IBV_WC_SUCCESS) {
          if (r->base->resiliency == NULL) {
            WARN("NET/IB: %s: Got CQE with error (devIndex=%d, req=%p, comm=%p (%s), wr_id=%lu, qp_num=%d)", __func__, i, r, r->base, r->base->isSend ? "send" : "recv", wc->wr_id, wc->qp_num);
            IbCastLogCompletionWithError(r->base, wc, i);
            // If resiliency is not enabled, we cannot recover from any error.
            return ncclRemoteError;
          }
          NCCLCHECK(IbCastResiliencyHandleCompletionError(r->base->resiliency, wc, i));
        } else {
          TRACE(NCCL_NET, "NET/IB: %s: Processing a completion event (devIndex=%d, comm=%p (%s), req=%p, wr_id=%lu, qp_num=%d)", __func__, i, r->base, r->base->isSend ? "send" : "recv", r, wc->wr_id, wc->qp_num);
          NCCLCHECK(IbCastCompletionEventProcess(r->base, wc, i));
        }
      }
      // Once the IB fatal event is reported in the async thread, we want to propagate this error
      // to communicator and prevent further polling to reduce error pollution.
      NCCLCHECK(IbCastStatsCheckFatalCount(&IbCastDevs[r->devBases[i]->ibDevN].stats,__func__));
    }
  } while (totalWrDone > 0);

  // If no (more) CQEs found on any device, return and come back later
  return ncclSuccess;
}
