/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// RCCL: classic NET/IB multi-segment send, CTS side table and flush. Included by
// p2p.cc only, after IB_WRITE_CHUNK_ALIGNMENT.
#ifndef NET_IB_P2P_MULTISEG_H_
#define NET_IB_P2P_MULTISEG_H_

// Select the MR covering [addr, addr+len) for device devIndex.
// - Single-segment handles return the legacy mrs[devIndex].
// - Multi-segment handles return the containing segment's MR, or NULL if the
//   range is not fully contained in one physical segment.
static inline ibv_mr* ncclIbMrForRange(const struct ncclIbMrHandle* h, uintptr_t addr, size_t len, int devIndex) {
  if (h == NULL) return NULL;
  if (h->nSegments <= 1) return h->mrs[devIndex];
  int s = ncclIbSegmentIndexForRange(h->nSegments, h->segStart, h->segLen, addr, len);
  return (s < 0) ? NULL : h->segMrs[s][devIndex];
}

static inline bool ncclIbCtsRemoteMultiSeg(const struct ncclIbSendComm* comm, int slot, int r) {
  const volatile struct ncclIbSendFifo* cts = &comm->ctsFifo[slot][r];
  const volatile struct ncclIbSegLayout* side = &comm->segLayoutFifo[slot][r];
  return (comm->peerCaps & NCCL_IB_CAP_MULTISEG) && ncclIbCtsHasSideTable(cts->nreqs) && side->idx == cts->idx &&
         side->nSegments > 1;
}

// Per-QP chunk size, as in ncclIbMultiSend: below splitDataThreshold only the
// active QP (i==0, see ncclIbCommBaseGetQpForRequest) carries data; above it
// each device's weighted share is split across that device's QPs.
static inline uint32_t ncclIbQpChunkSize(int size, int64_t splitDataThreshold, int i, uint8_t weight, int qpsPerDev) {
  if (size < splitDataThreshold) return (i == 0) ? size : 0;
  return DIVUP(DIVUP((uint64_t)size * weight, 100 * qpsPerDev), IB_WRITE_CHUNK_ALIGNMENT) * IB_WRITE_CHUNK_ALIGNMENT;
}

// Multi-segment (AIRUNTIME-2351 classic-path follow-up) variant of
// ncclIbMultiSend. It splits each request's per-QP chunk into one RDMA-write WR
// per local+remote physical segment slice, so a transfer that straddles a
// segment boundary uses the correct per-segment lkey/rkey. Invoked only when at
// least one request's local or remote buffer is multi-segment; the
// single-segment fast path in p2p.cc stays byte-for-byte unchanged.
static ncclResult_t ncclIbMultiSendSegmented(struct ncclIbSendComm* comm, int slot) {
  struct ncclIbRequest** reqs = comm->sendReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->ctsFifo[slot];
  volatile struct ncclIbSegLayout* side = comm->segLayoutFifo[slot];
  int nreqs = ncclIbCtsNreqs(slots[0].nreqs);
  if (nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  int nqps = 0;
  NCCLCHECK(ncclIbCommBaseGetNqpsPerRequest(&comm->base, &nqps));
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
  ncclIbQp* qps[NCCL_IB_MAX_QPS];
  int qpIndexes[NCCL_IB_MAX_QPS];
  int qpsPerDev[NCCL_IB_MAX_DEVS_PER_NIC] = {};
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, reqs[0]->id, i, &qps[i], &qpIndexes[i]));
    qpsPerDev[comm->base.qps[qpIndexes[i]].devIndex]++;
  }
  uint8_t* weights = reqs[0]->send.weights;
  ncclResult_t ret;

  // Cumulative wr_id across requests; placed on the final signaled WR so
  // completion accounting matches ncclIbMultiSend.
  uint64_t wr_id = 0ULL;
  for (int r = 0; r < nreqs; r++) wr_id += (uint64_t)(slot & 0xff) << (r * 8);

  uint32_t immData = comm->base.recvMatchingScheme == BY_ID ? (uint32_t)(reqs[0]->id % UINT32_MAX) : reqs[0]->send.size;
  // Non-BY_INDEX receivers sum byte_len for nreqs==1, but the imm rides only the last
  // slice here, so write the size record. Not for size 0: its 4-byte record is summed.
  bool needSizesWr =
    (nreqs > 1) || (comm->base.recvMatchingScheme != BY_INDEX && reqs[0]->send.size > 0);
  bool arExtra =
    (!(comm->base.remOooRq && comm->base.localOooRq) && comm->ar && reqs[0]->send.size > ncclIbArThreshold);
  bool extraImmWr = needSizesWr || arExtra;   // dedicated final WR carries the imm

  int64_t splitDataThreshold = rcclParamIbSplitDataThreshold();
  uint32_t sendOffsets[NCCL_NET_IB_MAX_RECVS] = {0};
#ifdef NCCL_ENABLE_NET_PROFILING
  for (int r = 0; r < nreqs; r++) reqs[r]->pInfo[0].nEventHandles = 0;
#endif

  int qpIndex = -1;
  ncclIbQp* qp = NULL;
  for (int i = 0; i < nqps; i++) {
    qp = qps[i];
    qpIndex = qpIndexes[i];
    int devIndex = qp->devIndex;
    int remDevIdx = qp->remDevIdx;
    int origDevIndex = comm->base.qps[qpIndex].devIndex;

    // Per-request chunk length on this QP (computed independently of the
    // resiliency skip so skipped QPs still advance the offsets consistently).
    uint32_t chunkLen[NCCL_NET_IB_MAX_RECVS];
    for (int r = 0; r < nreqs; r++) {
      uint32_t chunkSize =
        ncclIbQpChunkSize(reqs[r]->send.size, splitDataThreshold, i, weights[origDevIndex], qpsPerDev[origDevIndex]);
      chunkLen[r] = std::min<uint32_t>(reqs[r]->send.size - sendOffsets[r], chunkSize);
    }

    // Selective retransmission: skip already-delivered QPs, keeping offsets synced.
    if (comm->base.resiliency && reqs[0]->send.sentData[qpIndex] == true) {
      for (int r = 0; r < nreqs; r++)
        sendOffsets[r] = std::min<uint32_t>(sendOffsets[r] + chunkLen[r], reqs[r]->send.size);
      continue;
    }

    int w = 0;
    for (int r = 0; r < nreqs; r++) {
      uint64_t localBase = (uintptr_t)reqs[r]->send.data;
      uint64_t remoteBase = slots[r].addr;
      struct ncclIbMrHandle* mh = reqs[r]->send.mh;

      // Local segment offset/VA tables (contiguous VMM VAs => linear addresses).
      uint64_t lVA[NCCL_IB_MAX_SEGMENTS], lOff[NCCL_IB_MAX_SEGMENTS + 1];
      int nLocal;
      uint64_t localReqOff = 0;
      if (mh && mh->nSegments > 1) {
        nLocal = mh->nSegments;
        for (int s = 0; s < nLocal; s++) {
          lVA[s] = mh->segStart[s];
          lOff[s] = mh->segStart[s] - mh->segStart[0];
        }
        lOff[nLocal] = lOff[nLocal - 1] + mh->segLen[nLocal - 1];
        if (localBase < mh->segStart[0]) return ncclInternalError;
        localReqOff = localBase - mh->segStart[0];
      } else {
        nLocal = 1;
        lVA[0] = localBase;
        lOff[0] = 0;
        lOff[1] = reqs[r]->send.size;
      }
      // Remote segment tables from the side table when the peer published a
      // matching multi-seg layout; otherwise a single-segment view of the CTS slot.
      // Snapshot starts and rkeys once: the WR path must ship the same values
      // ncclIbCtsRemoteLayoutValid checked, not a later volatile re-read.
      uint64_t rVA[NCCL_IB_MAX_SEGMENTS], rOff[NCCL_IB_MAX_SEGMENTS + 1];
      uint32_t remoteRkeys[NCCL_IB_MAX_SEGMENTS];
      int nRemote;
      uint64_t remoteReqOff = 0;
      bool remoteMulti = ncclIbCtsRemoteMultiSeg(comm, slot, r);
      if (remoteMulti) {
        uint32_t remoteSegments = side[r].nSegments;
        uint64_t starts[NCCL_IB_MAX_SEGMENTS];
        const uint32_t nCopy =
          (remoteSegments >= 1 && remoteSegments <= NCCL_IB_MAX_SEGMENTS && remDevIdx >= 0 &&
           remDevIdx < NCCL_IB_MAX_DEVS_PER_NIC)
            ? remoteSegments
            : 0;
        for (uint32_t s = 0; s < nCopy; s++) {
          starts[s] = side[r].segStart[s];
          remoteRkeys[s] = side[r].segRkeys[s][remDevIdx];
        }
        if (!ncclIbCtsRemoteLayoutValid(remoteSegments, remDevIdx, NCCL_IB_MAX_DEVS_PER_NIC, NCCL_IB_MAX_SEGMENTS,
                                        nCopy ? starts : NULL, nCopy ? remoteRkeys : NULL)) {
          WARN("NET/IB: received invalid segment layout (nSegments=%u remDevIdx=%d)", remoteSegments, remDevIdx);
          return ncclInternalError;
        }
        nRemote = (int)remoteSegments;
        for (int s = 0; s < nRemote; s++) {
          rVA[s] = starts[s];
          rOff[s] = starts[s] - starts[0];
        }
        if (remoteBase < starts[0]) return ncclInternalError;
        remoteReqOff = remoteBase - starts[0];
        uint64_t remoteReqEnd = remoteReqOff + slots[r].size;
        if (remoteReqEnd < remoteReqOff) return ncclInternalError;
        while (nRemote > 1 && rOff[nRemote - 1] >= remoteReqEnd) nRemote--;
        rOff[nRemote] = remoteReqEnd;
      } else {
        nRemote = 1;
        rVA[0] = remoteBase;
        rOff[0] = 0;
        rOff[1] = slots[r].size;
      }

      if (chunkLen[r] == 0) {
        // Zero-length write preserves the chain/imm. The remote address can sit
        // in a later physical segment, so use that segment's rkey.
        struct ibv_send_wr* wr = comm->wrs + w;
        struct ibv_sge* sge = comm->sges + w;
        memset(wr, 0, sizeof(struct ibv_send_wr));
        wr->opcode = IBV_WR_RDMA_WRITE;
        wr->send_flags = 0;
        wr->wr_id = wr_id;
        wr->wr.rdma.remote_addr = remoteBase + sendOffsets[r];
        if (remoteMulti) {
          uint64_t off = remoteReqOff + sendOffsets[r];
          int s = ncclIbSegmentForZeroLengthOffset(nRemote, rOff, off);
          if (s < 0) return ncclInternalError;
          wr->wr.rdma.rkey = remoteRkeys[s];
        } else {
          wr->wr.rdma.rkey = slots[r].rkeys[remDevIdx];
        }
        sge->addr = localBase + sendOffsets[r];
        sge->length = 0;
        sge->lkey = reqs[r]->send.lkeys[devIndex];
        wr->sg_list = sge;
        wr->num_sge = 0;
        w++;
      } else {
        struct ncclIbSegSlice slices[2 * NCCL_IB_MAX_SEGMENTS];
        int ns =
          ncclIbSplitTransferAtOffsets(nLocal, lVA, lOff, nRemote, rVA, rOff, localReqOff + sendOffsets[r],
                                       remoteReqOff + sendOffsets[r], chunkLen[r], slices, 2 * NCCL_IB_MAX_SEGMENTS);
        if (ns <= 0) {
          WARN("NET/IB: multi-segment send split failed (data=%p off=%u len=%u)", reqs[r]->send.data, sendOffsets[r],
               chunkLen[r]);
          return ncclInternalError;
        }
        for (int k = 0; k < ns; k++) {
          if (w >= NCCL_IB_MAX_WRS_PER_SEND) {
            WARN("NET/IB: multi-segment send exceeded WR pool (%d)", NCCL_IB_MAX_WRS_PER_SEND);
            return ncclInternalError;
          }
          struct ibv_send_wr* wr = comm->wrs + w;
          struct ibv_sge* sge = comm->sges + w;
          memset(wr, 0, sizeof(struct ibv_send_wr));
          wr->opcode = IBV_WR_RDMA_WRITE;
          wr->send_flags = 0;
          wr->wr_id = wr_id;
          wr->wr.rdma.remote_addr = slices[k].remoteAddr;
          wr->wr.rdma.rkey = remoteMulti ? remoteRkeys[slices[k].remoteSeg] : slots[r].rkeys[remDevIdx];
          sge->addr = slices[k].localAddr;
          sge->length = slices[k].len;
          sge->lkey =
            (mh && mh->nSegments > 1) ? mh->segMrs[slices[k].localSeg][devIndex]->lkey : reqs[r]->send.lkeys[devIndex];
          wr->sg_list = sge;
          wr->num_sge = 1;
          w++;
        }
      }
    }

    // Final WR carries the WRITE_WITH_IMM (and, for multi-recv, the sizes array),
    // mirroring ncclIbMultiSend's lastWr handling.
    struct ibv_send_wr* lastWr;
    if (extraImmWr) {
      if (w >= NCCL_IB_MAX_WRS_PER_SEND + 1) {
        WARN("NET/IB: multi-segment send exceeded WR pool");
        return ncclInternalError;
      }
      lastWr = comm->wrs + w;
      memset(lastWr, 0, sizeof(struct ibv_send_wr));
      if (needSizesWr) {
        lastWr->wr.rdma.remote_addr = comm->remCmplsRecords.addr + slot * sizeof(struct ncclIbRequestCompletionRecord);
        lastWr->sg_list = &(comm->devs[devIndex].sge);
        lastWr->sg_list[0].addr = (uint64_t)(comm->remCmplsRecords.elems[slot]);
        lastWr->sg_list[0].length = nreqs * sizeof(int);
        lastWr->wr.rdma.rkey = comm->remCmplsRecords.rkeys[devIndex];
        lastWr->num_sge = 1;
      } else {
        lastWr->num_sge = 0;
      }
      w++;
    } else {
      lastWr = comm->wrs + (w - 1);   // reuse the last data WR
    }
    lastWr->wr_id = wr_id;
    lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    lastWr->imm_data = htobe32(immData);
    lastWr->send_flags = IBV_SEND_SIGNALED;

    for (int k = 0; k < w - 1; k++) comm->wrs[k].next = comm->wrs + k + 1;
    comm->wrs[w - 1].next = NULL;

#ifdef NCCL_ENABLE_NET_PROFILING
    for (int r = 0; r < nreqs; r++) {
      int nEventHandles = reqs[r]->pInfo[0].nEventHandles;
      reqs[r]->pInfo[0].qpIndex[nEventHandles] = qpIndex;
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      reqs[r]->pInfo[0].data.type = ncclProfileQp;
      reqs[r]->pInfo[0].data.qp.device = devIndex;
      reqs[r]->pInfo[0].data.qp.wr_id = lastWr->wr_id;
      reqs[r]->pInfo[0].data.qp.opcode = lastWr->opcode;
      reqs[r]->pInfo[0].data.qp.qpNum = qp->qp->qp_num;
      reqs[r]->pInfo[0].data.qp.length = chunkLen[r];
      void* pHandle = reqs[r]->pInfo[0].pHandle;
      NCCLCHECK(ncclProfilerFunction(&reqs[r]->pInfo[0].qpEventHandles[nEventHandles], ncclProfilerNetEventStart,
                                     pHandle, pluginId, &reqs[r]->pInfo[0].data));
      reqs[r]->pInfo[0].nEventHandles++;
    }
#endif

    struct ibv_send_wr* bad_wr;
    if (ncclIbWqeLatEnabled) ncclIbWqeLatMonStampSend(qp, comm->wrs);
    ret = wrap_ibv_post_send(qp->qp, comm->wrs, &bad_wr);
    if (ret != ncclSuccess) {
      ncclIbStatsFatalError(&comm->base.stats);
      return ret;
    }

    for (int r = 0; r < nreqs; r++) {
      sendOffsets[r] = std::min<uint32_t>(sendOffsets[r] + chunkLen[r], reqs[r]->send.size);
      reqs[r]->send.sentData[qpIndex] = true;
    }
  }
  return ncclSuccess;
}

// Segmented builder when any request's local or remote buffer spans several segments.
static inline bool rcclIbMultiSendNeedsSegmented(struct ncclIbSendComm* comm, int slot, int nreqs) {
  for (int r = 0; r < nreqs; r++) {
    struct ncclIbMrHandle* mh = comm->sendReqs[slot][r]->send.mh;
    if ((mh && mh->nSegments > 1) || ncclIbCtsRemoteMultiSeg(comm, slot, r)) return true;
  }
  return false;
}

// Side table and CTS are separate RDMA writes. Wait for the matching idx so
// a late side table is not treated as single-segment.
static inline void rcclIbIsendWaitSideTable(struct ncclIbSendComm* comm, int slot, int nreqs, uint64_t idx) {
  volatile struct ncclIbSendFifo* slots = comm->ctsFifo[slot];
  if (!(comm->peerCaps & NCCL_IB_CAP_MULTISEG) || !ncclIbCtsHasSideTable(slots[0].nreqs)) return;
  volatile struct ncclIbSegLayout* side = comm->segLayoutFifo[slot];
  for (int r = 0; r < nreqs; r++)
    while (side[r].idx != idx);
  std::atomic_thread_fence(std::memory_order_seq_cst);
}

// Returns the WR chain to post: the side-table write chained before the CTS WR,
// or the CTS WR alone.
static inline struct ibv_send_wr* rcclIbPostFifoSideWr(struct ncclIbRecvComm* comm, struct ncclIbRequest* req,
                                                       int slot, ncclIbQp* ctsQp, struct ibv_send_wr* ctsWr,
                                                       struct ibv_send_wr* wrSide, struct ibv_sge* sgeSide) {
  struct ncclIbSendFifo* localElem = comm->remCtsFifo.elems[slot];
  // irecv flags the slot only when the peer understands the side table and
  // some receive in it is multi-segment; the sender waits on idx only then.
  bool postSide = (comm->peerCaps & NCCL_IB_CAP_MULTISEG) && ncclIbCtsHasSideTable(localElem[0].nreqs);

  if (!postSide) return ctsWr;
  memset(wrSide, 0, sizeof(*wrSide));
  memset(sgeSide, 0, sizeof(*sgeSide));
  // Flushed unsignaled WRs still complete on error; resiliency maps wr_id to a slot.
  wrSide->wr_id = ctsWr->wr_id;
  wrSide->wr.rdma.remote_addr =
    comm->remSegLayout.addr + (uint64_t)slot * NCCL_NET_IB_MAX_RECVS * sizeof(struct ncclIbSegLayout);
  wrSide->wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].rkey;
  sgeSide->addr = (uint64_t)comm->remSegLayout.elems[slot];
  sgeSide->length = req->nreqs * sizeof(struct ncclIbSegLayout);
  sgeSide->lkey = comm->devs[ctsQp->devIndex].segLayoutFifoMr->lkey;
  wrSide->sg_list = sgeSide;
  wrSide->num_sge = 1;
  wrSide->opcode = IBV_WR_RDMA_WRITE;
  wrSide->send_flags = 0; // unsignaled, never inline
  wrSide->next = ctsWr;
  return wrSide;
}

// A NULL handle or missing per-device MR would publish a zero rkey in the CTS.
static inline ncclResult_t rcclIbIrecvCheckHandle(struct ncclIbRecvComm* comm, struct ncclIbMrHandle* mh, int i) {
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

static inline ncclResult_t rcclIbIrecvFillSideTable(struct ncclIbRecvComm* comm, int slot, int i,
                                                    struct ncclIbMrHandle* mhandleWrapper, bool* anyMultiSeg) {
  struct ncclIbSendFifo* localElem = comm->remCtsFifo.elems[slot];
  // Multi-segment layout goes on the side table, never in CTS. Every entry of
  // a flagged slot carries idx so the sender can wait for the whole slot.
  struct ncclIbSegLayout* sideElem = comm->remSegLayout.elems[slot];
  if (comm->peerCaps & NCCL_IB_CAP_MULTISEG) {
    if (mhandleWrapper->nSegments > 1) {
      *anyMultiSeg = true;
      sideElem[i].nSegments = mhandleWrapper->nSegments;
      for (int s = 0; s < mhandleWrapper->nSegments; s++) {
        sideElem[i].segStart[s] = (uint64_t)mhandleWrapper->segStart[s];
        for (int j = 0; j < comm->base.vProps.ndevs; j++) {
          if (mhandleWrapper->segMrs[s][j] == NULL) {
            WARN("NET/IB: irecv[%d] missing MR for segment %d device %d", i, s, j);
            return ncclInternalError;
          }
          sideElem[i].segRkeys[s][j] = mhandleWrapper->segMrs[s][j]->rkey;
        }
      }
    } else {
      sideElem[i].nSegments = 0;
    }
    sideElem[i].idx = localElem[i].idx; // last store; sender spins on idx
  } else {
    sideElem[i].nSegments = 0;
    sideElem[i].idx = 0;
  }
  return ncclSuccess;
}

static inline void rcclIbCtsFlagSideTable(struct ncclIbSendFifo* localElem, int n) {
  for (int i = 0; i < n; i++) localElem[i].nreqs |= NCCL_IB_CTS_NREQS_SIDE_TABLE;
}

static inline bool rcclIbIflushNeedsMultiSeg(int n, int* sizes, void** mhandles) {
  if (n > NCCL_NET_IB_MAX_RECVS) return true;
  for (int i = 0; i < n; i++)
    if (sizes[i] && mhandles[i] != NULL && ((struct ncclIbMrHandle*)mhandles[i])->nSegments > 1) return true;
  return false;
}

// ncclIbIflush with one RDMA read per receive per overlapped segment.
static ncclResult_t rcclIbIflushMultiSeg(void* recvComm, int n, void** data, int* sizes, void** mhandles,
                                         void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  ncclResult_t ret = ncclSuccess;
  ncclResult_t iflushRet = ncclSuccess;
  struct ncclIbRequest* req = NULL;
  struct ncclIbMrHandle* mhandle = NULL;
  int last = -1;
  bool anyMultiSeg = false;
  if (n > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;
  for (int i = 0; i < n; i++) {
    if (sizes[i] == 0) continue;
    last = i;
    if (mhandles[i] != NULL && ((struct ncclIbMrHandle*)mhandles[i])->nSegments > 1) anyMultiSeg = true;
  }
  if (comm->flushEnabled == 0 || last == -1) return ncclSuccess;
  NCCLCHECK(ncclIbCreateFlushQp(comm));

  // Only flush once using the last non-zero receive, unless a multi-segment
  // handle is involved (see below).
  mhandle = (struct ncclIbMrHandle*)mhandles[last];
  NCCLCHECKGOTO(ncclIbGetRequest(&comm->base, &req), ret, iflushFail);
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

    if (useGpuFlushMem) {
      struct ibv_send_wr writeWr;
      memset(&writeWr, 0, sizeof(writeWr));
      writeWr.wr_id = wr.wr_id;
      writeWr.wr.rdma.remote_addr = (uint64_t)(comm->devs[i].gpuFlush.gpuFlushGpuMem);
      writeWr.wr.rdma.rkey = comm->devs[i].gpuFlush.gpuMr->rkey;
      writeWr.sg_list = &comm->devs[i].gpuFlush.sge;
      writeWr.num_sge = 1;
      writeWr.opcode = IBV_WR_RDMA_WRITE;
      writeWr.send_flags = 0;
      struct ibv_send_wr* badWriteWr;
      NCCLCHECKGOTO(wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, &writeWr, &badWriteWr), ret, iflushFail);
    }
    // The scratchpad and single-segment cases post one read. A multi-segment
    // handle may span several MRs, so then every non-zero entry and every
    // segment it overlaps gets a read, posted as one chain.
    struct ibv_send_wr flushWrs[NCCL_NET_IB_MAX_RECVS * NCCL_IB_MAX_SEGMENTS];
    int nFlushWrs = 0;
    if (useGpuFlushMem) {
      memset(&flushWrs[0], 0, sizeof(flushWrs[0]));
      flushWrs[0].wr.rdma.remote_addr = (uint64_t)(comm->devs[i].gpuFlush.gpuFlushGpuMem);
      flushWrs[0].wr.rdma.rkey = comm->devs[i].gpuFlush.gpuMr->rkey;
      nFlushWrs = 1;
    } else if (!anyMultiSeg) {
      memset(&flushWrs[0], 0, sizeof(flushWrs[0]));
      flushWrs[0].wr.rdma.remote_addr = (uint64_t)data[last];
      flushWrs[0].wr.rdma.rkey = mhandle->mrs[i]->rkey;
      nFlushWrs = 1;
    } else {
      for (int r = 0; r < n; r++) {
        if (sizes[r] == 0) continue;
        struct ncclIbMrHandle* rh = (struct ncclIbMrHandle*)mhandles[r];
        if (rh != NULL && rh->nSegments > 1) {
          int flushSeg[NCCL_IB_MAX_SEGMENTS];
          int nFlushSeg =
            ncclIbSegmentsOverlappingRange(rh->nSegments, rh->segStart, rh->segLen, (uintptr_t)data[r],
                                           (size_t)sizes[r], flushSeg, NCCL_IB_MAX_SEGMENTS);
          if (nFlushSeg < 1) {
            WARN("NET/IB: flush buffer %p size %d does not overlap any registered segment", data[r], sizes[r]);
            ret = ncclInternalError;
            goto iflushFail;
          }
          for (int s = 0; s < nFlushSeg; s++) {
            int seg = flushSeg[s];
            uintptr_t segBase = rh->segStart[seg];
            uintptr_t rangeStart = (uintptr_t)data[r];
            struct ibv_mr* flushMr = rh->segMrs[seg][i];
            if (flushMr == NULL) {
              WARN("NET/IB: flush missing MR for receive %d segment %d device %d", r, seg, i);
              ret = ncclInternalError;
              goto iflushFail;
            }
            memset(&flushWrs[nFlushWrs], 0, sizeof(flushWrs[0]));
            flushWrs[nFlushWrs].wr.rdma.remote_addr = (uint64_t)(rangeStart > segBase ? rangeStart : segBase);
            flushWrs[nFlushWrs].wr.rdma.rkey = flushMr->rkey;
            nFlushWrs++;
          }
        } else {
          struct ibv_mr* flushMr = ncclIbMrForRange(rh, (uintptr_t)data[r], sizeof(int), i);
          if (flushMr == NULL) {
            WARN("NET/IB: flush missing MR for receive %d buffer %p", r, data[r]);
            ret = ncclInternalError;
            goto iflushFail;
          }
          memset(&flushWrs[nFlushWrs], 0, sizeof(flushWrs[0]));
          flushWrs[nFlushWrs].wr.rdma.remote_addr = (uint64_t)data[r];
          flushWrs[nFlushWrs].wr.rdma.rkey = flushMr->rkey;
          nFlushWrs++;
        }
      }
    }
    for (int w = 0; w < nFlushWrs; w++) {
      flushWrs[w].wr_id = wr.wr_id;
      flushWrs[w].sg_list = &comm->devs[i].gpuFlush.sge;
      flushWrs[w].num_sge = 1;
      flushWrs[w].opcode = IBV_WR_RDMA_READ;
      flushWrs[w].send_flags = (w == nFlushWrs - 1) ? IBV_SEND_SIGNALED : 0;
      flushWrs[w].next = (w + 1 < nFlushWrs) ? &flushWrs[w + 1] : NULL;
    }
    TRACE(NCCL_NET, "NET/IB: %s: Posting a %d-read flush request (req=%p, comm=%p, wr_id=%ld)", __func__, nFlushWrs,
          req, req->base, wr.wr_id);
    TIME_START(4);
    struct ibv_send_wr* bad_wr = NULL;
    iflushRet = wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, &flushWrs[0], &bad_wr);
    TIME_STOP(4);
    if (iflushRet != ncclSuccess) {
      // Prefix posts do not call ncclIbAddEvent. Count accepted WRs so we keep
      // the request when a signaled tail is lost, matching isendFail.
      int posted = 0;
      for (struct ibv_send_wr* w = &flushWrs[0]; w != NULL && w != bad_wr; w = w->next) posted++;
      if (posted > 0) {
        ncclIbAddEvent(req, i);
        req->type = NCCL_NET_IB_REQ_FAILED;
        *request = req;
        ncclIbStatsFatalError(&comm->base.stats);
        return ncclSuccess;
      }
      goto iflushFail;
    }
    ncclIbAddEvent(req, i);

    TRACE(NCCL_NET, "NET/IB: %s: Flush request posted (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base,
          wr.wr_id);
  }

  *request = req;
  return ncclSuccess;

iflushFail:
  ncclIbStatsFatalError(&comm->base.stats);
  if (req) {
    // If events were added (IBV ops posted), we can't free immediately -
    // completions may still arrive. Mark as failed and return it so
    // caller can call Test to drain completions.
    if (ncclIbRequestHasEvents(req)) {
      req->type = NCCL_NET_IB_REQ_FAILED;
      *request = req;
      // Return success so caller proceeds to call Test() which will drain
      // completions. The fatal error is recorded in stats and will be
      // caught on the next operation.
      return ncclSuccess;
    } else {
      ncclIbFreeRequest(req);
      *request = NULL;
    }
  } else {
    *request = NULL;
  }
  // NCCLCHECKGOTO stored the post error in iflushRet; ret is still success.
  return (iflushRet != ncclSuccess) ? iflushRet : ret;
}

#endif
