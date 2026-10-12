/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// RCCL: NET/IB-CAST multi-segment send, CTS side table and flush. Included by p2p.cc only.
#ifndef NET_IB_CAST_P2P_MULTISEG_CAST_H_
#define NET_IB_CAST_P2P_MULTISEG_CAST_H_

#include <algorithm>
#include <atomic>
#include "p2p_cast.h"
#include "common_cast.h"
#include "qp_sharing.h"
#include "net_telemetry.h"

RCCL_PARAM_DECLARE(IbCastGdrFlushGpuMemNoRelaxedOrdering);

static inline bool IbCastCtsRemoteMultiSeg(const struct ncclIbSendComm* comm, int slot, int r) {
  if ((comm->peerCaps & NCCL_IB_CAP_MULTISEG) == 0) return false;
  volatile void* slotBase = (volatile void*)comm->ctsFifo[slot];
  const volatile struct ncclIbSegLayout* side = &comm->segLayoutFifo[slot][r];
  return side->nSegments > 1 && side->idx == (uint64_t)ctsFifoIdx(slotBase, r);
}

static inline ncclResult_t IbCastMultiSendSegmented(struct ncclIbSendComm* comm, int slot, int nqps, int startQpIndex,
                                                    bool wrrSched, bool useWriteOp) {
  struct ncclIbRequest** reqs = comm->sendReqs[slot];
  volatile void* slots = (volatile void*)comm->ctsFifo[slot];
  volatile struct ncclIbSegLayout* side = comm->segLayoutFifo[slot];
  int nreqs = ctsFifoNreqs(slots, 0);
  uint64_t nowNs = 0;
  if (nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  TRACE(NCCL_NET, "NET/IB: %s: Posting a segmented send request (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d)", __func__,
        reqs[0], reqs[0]->base, reqs[0]->id, slot, nreqs);

  uint64_t wr_id = 0ULL;
  for (int r = 0; r < nreqs; r++) wr_id += (uint64_t)(slot & 0xff) << (r * 8);

  uint32_t immData = 0;
  if (!useWriteOp) {
    if (comm->base.recvMatchingScheme != BY_INDEX) {
      immData = (uint32_t)(reqs[0]->id % UINT32_MAX);
    } else {
      uint32_t rxReqIdx = (uint32_t)ctsFifoRxReqIndex(slots, 0);
      immData = ((rxReqIdx << WR_IMM_RX_REQ_IDX_BIT_POS) & WR_IMM_RX_REQ_IDX_MASK) | WR_IMM_SEGMENTED_FLAG;
      if (nqps > 1) immData |= WR_IMM_SPLIT_DATA_FLAG;
    }
  }
  // The final WRITE_WITH_IMM may carry only the last segment slice, so its
  // byte_len is not the logical request size. Publish completion sizes for
  // every segmented request, including a single receive.
  bool needSizesWr = true;
  bool extraImmWr = !useWriteOp && needSizesWr;

  uint32_t sendOffsets[NCCL_NET_IB_MAX_RECVS] = {0};
#ifdef NCCL_ENABLE_NET_PROFILING
  for (int r = 0; r < nreqs; r++) reqs[r]->pInfo[0].nEventHandles = 0;
#endif
  int qpIndex = -1;
  ncclIbQp* qp = NULL;
  const int align = 128;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(IbCastCommBaseGetQpForRequest(&comm->base, startQpIndex, i, &qp, &qpIndex));
    int devIndex = qp->devIndex;
    int remDevIdx = qp->remDevIdx;

    uint32_t chunkLen[NCCL_NET_IB_MAX_RECVS];
    for (int r = 0; r < nreqs; r++) {
      int chunkSize, length;
      if ((nqps > 1) && reqs[r]->desc.parms.enable && comm->base.qpTxSchedInit) {
        int weightedSendSize = (int)(((double)reqs[r]->send.size) * comm->base.qpTxSched[qpIndex].weight);
        chunkSize = (weightedSendSize / align) * align;
        if (i == (nqps - 1)) length = std::max((int)(reqs[r]->send.size - sendOffsets[r]), chunkSize);
        else length = chunkSize;
      } else {
        chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
        length = std::min((int)(reqs[r]->send.size - sendOffsets[r]), chunkSize);
      }
      if (length < 0) length = 0;
      chunkLen[r] = (uint32_t)length;
    }

    if (comm->base.resiliency && reqs[0]->send.sentData[qpIndex] == true) {
      for (int r = 0; r < nreqs; r++)
        sendOffsets[r] = std::min<uint32_t>(sendOffsets[r] + chunkLen[r], reqs[r]->send.size);
      continue;
    }

    int w = 0;
    for (int r = 0; r < nreqs; r++) {
      uint64_t localBase = (uintptr_t)reqs[r]->send.data;
      uint64_t remoteBase = ctsFifoAddr(slots, r);
      struct ncclIbMrHandle* mh = reqs[r]->send.mh;

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

      uint64_t rVA[NCCL_IB_MAX_SEGMENTS], rOff[NCCL_IB_MAX_SEGMENTS + 1];
      int nRemote;
      uint64_t remoteReqOff = 0;
      uint64_t starts[NCCL_IB_MAX_SEGMENTS];
      uint32_t rkeys[NCCL_IB_MAX_SEGMENTS];
      bool remoteMulti = IbCastCtsRemoteMultiSeg(comm, slot, r);
      if (remoteMulti) {
        uint32_t remoteSegments = side[r].nSegments;
        const uint32_t nCopy = (remoteSegments >= 1 && remoteSegments <= NCCL_IB_MAX_SEGMENTS && remDevIdx >= 0 &&
                                remDevIdx < NCCL_IB_MAX_DEVS_PER_NIC) ?
                                 remoteSegments :
                                 0;
        for (uint32_t s = 0; s < nCopy; s++) {
          starts[s] = side[r].segStart[s];
          rkeys[s] = side[r].segRkeys[s][remDevIdx];
        }
        if (!ncclIbCtsRemoteLayoutValid(remoteSegments, remDevIdx, NCCL_IB_MAX_DEVS_PER_NIC, NCCL_IB_MAX_SEGMENTS,
                                        nCopy ? starts : NULL, nCopy ? rkeys : NULL)) {
          WARN("NET/IB: CAST received invalid segment layout (nSegments=%u remDevIdx=%d)", remoteSegments, remDevIdx);
          return ncclInternalError;
        }
        nRemote = (int)remoteSegments;
        for (int s = 0; s < nRemote; s++) {
          rVA[s] = starts[s];
          rOff[s] = starts[s] - starts[0];
        }
        if (remoteBase < starts[0]) return ncclInternalError;
        remoteReqOff = remoteBase - starts[0];
        uint64_t remoteReqEnd = remoteReqOff + ctsFifoSize(slots, r);
        if (remoteReqEnd < remoteReqOff) return ncclInternalError;
        while (nRemote > 1 && rOff[nRemote - 1] >= remoteReqEnd) nRemote--;
        rOff[nRemote] = remoteReqEnd;
      } else {
        nRemote = 1;
        rVA[0] = remoteBase;
        rOff[0] = 0;
        rOff[1] = ctsFifoSize(slots, r);
      }

      if (chunkLen[r] == 0) {
        if (w >= NCCL_IB_MAX_WRS_PER_SEND) return ncclInternalError;
        struct ibv_send_wr* wr = comm->wrs + w;
        struct ibv_sge* sge = comm->sges + w;
        memset(wr, 0, sizeof(struct ibv_send_wr));
        wr->opcode = IBV_WR_RDMA_WRITE;
        wr->send_flags = 0;
        wr->wr_id = wr_id;
        wr->wr.rdma.remote_addr = remoteBase + sendOffsets[r];
        if (remoteMulti) {
          int s = ncclIbSegmentForZeroLengthOffset(nRemote, rOff, remoteReqOff + sendOffsets[r]);
          if (s < 0) return ncclInternalError;
          wr->wr.rdma.rkey = rkeys[s];
        } else {
          wr->wr.rdma.rkey = ctsFifoRkey(slots, r, remDevIdx);
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
          WARN("NET/IB: CAST multi-segment send split failed (data=%p off=%u len=%u)", reqs[r]->send.data,
               sendOffsets[r], chunkLen[r]);
          return ncclInternalError;
        }
        for (int k = 0; k < ns; k++) {
          if (w >= NCCL_IB_MAX_WRS_PER_SEND) {
            WARN("NET/IB: CAST multi-segment send exceeded WR pool (%d)", NCCL_IB_MAX_WRS_PER_SEND);
            return ncclInternalError;
          }
          struct ibv_send_wr* wr = comm->wrs + w;
          struct ibv_sge* sge = comm->sges + w;
          memset(wr, 0, sizeof(struct ibv_send_wr));
          wr->opcode = IBV_WR_RDMA_WRITE;
          wr->send_flags = 0;
          wr->wr_id = wr_id;
          wr->wr.rdma.remote_addr = slices[k].remoteAddr;
          wr->wr.rdma.rkey = remoteMulti ? rkeys[slices[k].remoteSeg] : ctsFifoRkey(slots, r, remDevIdx);
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

    struct ibv_send_wr* lastWr;
    if (extraImmWr) {
      if (w >= NCCL_IB_MAX_WRS_PER_SEND + 1) return ncclInternalError;
      lastWr = comm->wrs + w;
      memset(lastWr, 0, sizeof(struct ibv_send_wr));
      if (needSizesWr) {
        lastWr->wr.rdma.remote_addr = comm->remCmplsRecords.addr + slot * sizeof(struct ncclIbRequestCompletionRecord);
        lastWr->sg_list = &(comm->devs[devIndex].sge);
        lastWr->sg_list[0].addr = (uint64_t)(comm->remCmplsRecords.elems[slot]);
        lastWr->sg_list[0].length = nreqs * sizeof(int);
        lastWr->wr.rdma.rkey = comm->remCmplsRecords.rkeys[devIndex];
        lastWr->num_sge = 1;
      }
      w++;
    } else {
      lastWr = comm->wrs + (w - 1);
    }
    lastWr->wr_id = wr_id;
    lastWr->next = NULL;
    lastWr->send_flags = IBV_SEND_SIGNALED;
    if (!useWriteOp) {
      lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      lastWr->imm_data = htobe32(immData);
    }

    struct ncclIbRemapWrId* remapWrId = nullptr;
    if ((comm->base.recvMatchingScheme == BY_INDEX)) {
      NCCLCHECK(IbCastQpSchedGetRemap(&comm->base, wr_id, qpIndex, &remapWrId));
      lastWr->wr_id = (uint64_t)remapWrId;
      remapWrId->parms = reqs[nreqs - 1]->desc.parms;
      remapWrId->tx.bytes = chunkLen[nreqs - 1];
      nowNs = 0;
      if (reqs[nreqs - 1]->desc.parms.enable && ((nqps > 1) || reqs[nreqs - 1]->desc.parms.doWrr)) {
        struct timespec txStartTime;
        if (!clock_gettime(CLOCK_MONOTONIC, &txStartTime)) nowNs = TIMESPEC_TO_NSEC(&txStartTime);
        remapWrId->tx.startTimeNs = nowNs;
      }
    }

    for (int k = 0; k < w - 1; k++) comm->wrs[k].next = comm->wrs + k + 1;

#ifdef NCCL_ENABLE_NET_PROFILING
    for (int r = 0; r < nreqs; r++) {
      int nEventHandles = reqs[r]->pInfo[0].nEventHandles;
      assert(nEventHandles < MAX_QPS_PER_REQ);
      reqs[r]->pInfo[0].qpIndex[nEventHandles] = qpIndex;
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      reqs[r]->pInfo[0].data.type = ncclProfileQp;
      reqs[r]->pInfo[0].data.qp.device = devIndex;
      reqs[r]->pInfo[0].data.qp.wr_id = lastWr->wr_id;
      reqs[r]->pInfo[0].data.qp.opcode = lastWr->opcode;
      reqs[r]->pInfo[0].data.qp.qpNum = qp->qp->qp_num;
      reqs[r]->pInfo[0].data.qp.length = chunkLen[r];
      void* pHandle = reqs[r]->pInfo[0].pHandle;
      NCCLCHECK(IbCastProfilerFunction(&reqs[r]->pInfo[0].qpEventHandles[nEventHandles], ncclProfilerNetEventStart,
                                       pHandle, pluginId, &reqs[r]->pInfo[0].data));
      reqs[r]->pInfo[0].nEventHandles++;
    }
#endif
#ifdef ENABLE_FAULT_INJECTION
    {
      const uint32_t faultDelay = comm->base.faultQpDelayUs[qpIndex];
      if (faultDelay) usleep(faultDelay);
      if (comm->base.faultQpError[qpIndex]) {
        if (remapWrId) IbCastQpSchedFreeRemap(remapWrId);
        IbCastStatsFatalError(&comm->base.stats);
        return ncclSystemError;
      }
    }
#endif
    // Mirror IbCastMultiSend: segmented CQEs still stop telemetry with tel_post_ts.
    if (rcclTelemetryOn() && i == 0) {
      int64_t _tel_ns = rcclTelemetryPostTs(qp->telQpStats);
      for (int r = 0; r < nreqs; r++) reqs[r]->tel_post_ts = _tel_ns;
    }

    struct ibv_send_wr* bad_wr;
    ncclResult_t postRet = wrap_ibv_post_send(qp->qp, comm->wrs, &bad_wr);
    if (postRet != ncclSuccess) {
      if (remapWrId) IbCastQpSchedFreeRemap(remapWrId);
      return postRet;
    }

    rcclTelemetryQpSendPosted(qp->telQpStats, useWriteOp ? 0 : 1);

    for (int r = 0; r < nreqs; r++) {
      sendOffsets[r] = std::min<uint32_t>(sendOffsets[r] + chunkLen[r], reqs[r]->send.size);
      reqs[r]->send.sentData[qpIndex] = true;
    }
  }

  if (nowNs) {
    if (comm->base.nextQpTxSchedUpdateNs == 0)
      comm->base.nextQpTxSchedUpdateNs = nowNs + comm->base.schedParms.updateInterval;
    else if (nowNs >= comm->base.nextQpTxSchedUpdateNs) {
      IbCastQpSchedUpdateTx(&comm->base);
      comm->base.nextQpTxSchedUpdateNs = nowNs + comm->base.schedParms.updateInterval;
    }
    if (comm->base.schedParms.logEnable) {
      if (comm->base.nextSchedLogNs == 0) comm->base.nextSchedLogNs = nowNs + comm->base.schedParms.logInterval;
      else if (nowNs >= comm->base.nextSchedLogNs) {
        IbCastLogSched(comm);
        comm->base.nextSchedLogNs = nowNs + comm->base.schedParms.logInterval;
      }
    }
  }
  return ncclSuccess;
}

// Segmented builder when any request's local or remote buffer spans several segments.
static inline bool rcclIbCastMultiSendNeedsSegmented(struct ncclIbSendComm* comm, int slot, int nreqs) {
  if (comm->useCtsOffload) return false;
  for (int r = 0; r < nreqs; r++) {
    struct ncclIbMrHandle* mh = comm->sendReqs[slot][r]->send.mh;
    if ((mh && mh->nSegments > 1) || IbCastCtsRemoteMultiSeg(comm, slot, r)) return true;
  }
  return false;
}

static inline bool rcclIbCastSendIsSegmented(const struct ncclIbSendComm* comm, const struct ncclIbMrHandle* mh,
                                             int slot, int r) {
  return !comm->useCtsOffload && ((mh != NULL && mh->nSegments > 1) || IbCastCtsRemoteMultiSeg(comm, slot, r));
}

static inline bool IbCastHasOtherSegmentedSend(struct ncclIbSendComm* comm, int slot) {
  for (int s = 0; s < NET_IB_MAX_REQUESTS; s++) {
    if (s == slot) continue; // Allow every member of the same multi-recv group.
    for (int r = 0; r < NCCL_NET_IB_MAX_RECVS; r++) {
      struct ncclIbRequest* active = comm->sendReqs[s][r];
      if (active != NULL && active->send.segmented) return true;
    }
  }
  return false;
}

// CTS and the side table are separate RDMA writes. Bound the wait so a peer
// that advertised MULTISEG but never posts the side table (CTS-offload
// disagreement) cannot hang the sender forever.
static inline ncclResult_t IbCastWaitSideTable(volatile struct ncclIbSegLayout* side, int nreqs, uint64_t idx) {
  struct timespec start;
  const uint64_t kTimeoutNs = 2ull * NSEC_PER_SEC;
  if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
    WARN("NET/IB: clock_gettime failed waiting for multi-segment CTS side table");
    return ncclRemoteError;
  }
  for (;;) {
    bool ready = true;
    for (int r = 0; r < nreqs; r++) {
      if (side[r].idx != idx) {
        ready = false;
        break;
      }
    }
    if (ready) return ncclSuccess;
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
      WARN("NET/IB: clock_gettime failed waiting for multi-segment CTS side table");
      return ncclRemoteError;
    }
    if ((TIMESPEC_TO_NSEC(&now) - TIMESPEC_TO_NSEC(&start)) >= kTimeoutNs) {
      WARN("NET/IB: timed out waiting for multi-segment CTS side table idx %lu", (unsigned long)idx);
      return ncclRemoteError;
    }
  }
}

static inline ncclResult_t rcclIbCastIsendWaitSideTable(struct ncclIbSendComm* comm, int slot, int nreqs,
                                                        uint64_t idx) {
  if (!(comm->peerCaps & NCCL_IB_CAP_MULTISEG)) return ncclSuccess;
  NCCLCHECK(IbCastWaitSideTable(comm->segLayoutFifo[slot], nreqs, idx));
  std::atomic_thread_fence(std::memory_order_seq_cst);
  return ncclSuccess;
}

// Returns the WR chain to post: the side-table write chained before the CTS WR, or the CTS WR alone.
// The side table goes out whenever the peer understands it, including nSegments==1, so the sender's wait on idx ends.
static inline struct ibv_send_wr* rcclIbCastPostFifoSideWr(struct ncclIbRecvComm* comm, int slot, int n,
                                                           ncclIbQp* ctsQp, struct ibv_send_wr* ctsWr,
                                                           struct ibv_send_wr* wrSide, struct ibv_sge* sgeSide) {
  if (comm->useCtsOffload || !(comm->peerCaps & NCCL_IB_CAP_MULTISEG)) return ctsWr;
  memset(wrSide, 0, sizeof(*wrSide));
  memset(sgeSide, 0, sizeof(*sgeSide));
  wrSide->wr.rdma.remote_addr =
    comm->remSegLayout.addr + (uint64_t)slot * NCCL_NET_IB_MAX_RECVS * sizeof(struct ncclIbSegLayout);
  wrSide->wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].rkey;
  sgeSide->addr = (uint64_t)comm->remSegLayout.elems[slot];
  sgeSide->length = n * sizeof(struct ncclIbSegLayout);
  sgeSide->lkey = comm->devs[ctsQp->devIndex].segLayoutFifoMr->lkey;
  wrSide->sg_list = sgeSide;
  wrSide->num_sge = 1;
  wrSide->opcode = IBV_WR_RDMA_WRITE;
  wrSide->send_flags = 0; // unsignaled, never inline
  wrSide->next = ctsWr;
  return wrSide;
}

// Error cleanup cannot recycle a request while completions still reference it, so irecv checks first.
static inline ncclResult_t rcclIbCastIrecvCheckHandle(struct ncclIbRecvComm* comm, struct ncclIbMrHandle* mhandle,
                                                      void* data, size_t size, int r) {
  if (mhandle == NULL || mhandle->nSegments < 1 || mhandle->nSegments > NCCL_IB_MAX_SEGMENTS) {
    WARN("NET/IB: irecv[%d] has an invalid mhandle or segment count", r);
    return ncclInternalError;
  }
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    if (mhandle->mrs[i] == NULL) {
      WARN("NET/IB: irecv[%d] missing MR for device %d", r, i);
      return ncclInternalError;
    }
  }
  if (mhandle->nSegments > 1) {
    uintptr_t registeredEnd = mhandle->segStart[0];
    for (int s = 0; s < mhandle->nSegments; s++) {
      if (mhandle->segLen[s] == 0 || mhandle->segStart[s] != registeredEnd) {
        WARN("NET/IB: irecv[%d] has an invalid segment layout at segment %d", r, s);
        return ncclInternalError;
      }
      registeredEnd += mhandle->segLen[s];
      if (registeredEnd < mhandle->segStart[s]) return ncclInternalError;
      for (int i = 0; i < comm->base.vProps.ndevs; i++) {
        if (mhandle->segMrs[s][i] == NULL) {
          WARN("NET/IB: irecv[%d] missing MR for segment %d device %d", r, s, i);
          return ncclInternalError;
        }
      }
    }
    uintptr_t recvStart = (uintptr_t)data;
    uintptr_t recvEnd = recvStart + size;
    if (recvEnd < recvStart || recvStart < mhandle->segStart[0] || recvEnd > registeredEnd) {
      WARN("NET/IB: irecv[%d] range %p+%zu is outside its registered segments", r, data, size);
      return ncclInternalError;
    }
  }
  return ncclSuccess;
}

static inline void rcclIbCastIrecvFillSideTable(struct ncclIbRecvComm* comm, int slot, int i,
                                                struct ncclIbMrHandle* mhandleWrapper) {
  struct ncclIbSegLayout* sideElem = comm->remSegLayout.elems[slot];
  if ((comm->peerCaps & NCCL_IB_CAP_MULTISEG) && !comm->useCtsOffload) {
    if (mhandleWrapper->nSegments > 1) {
      sideElem[i].nSegments = mhandleWrapper->nSegments;
      for (int s = 0; s < mhandleWrapper->nSegments; s++) {
        sideElem[i].segStart[s] = (uint64_t)mhandleWrapper->segStart[s];
        for (int j = 0; j < comm->base.vProps.ndevs; j++) {
          sideElem[i].segRkeys[s][j] = mhandleWrapper->segMrs[s][j]->rkey;
        }
      }
    } else {
      sideElem[i].nSegments = 0;
    }
    sideElem[i].idx = (uint32_t)(comm->base.fifoHead + 1); // last store; sender waits on idx
  } else {
    sideElem[i].nSegments = 0;
    sideElem[i].idx = 0;
  }
}

// BY_INDEX only: a segmented send publishes the full logical size before the immediate,
// because wc->byte_len covers only its final slice.
static inline bool rcclIbCastImmSegmented(const struct ibv_wc* wc) {
  return (be32toh(wc->imm_data) & WR_IMM_SEGMENTED_FLAG) != 0;
}

static inline bool rcclIbCastIflushNeedsMultiSeg(int n, int* sizes, void** mhandles) {
  for (int i = 0; i < n; i++)
    if (sizes[i] && mhandles[i] != NULL && ((struct ncclIbMrHandle*)mhandles[i])->nSegments > 1) return true;
  return false;
}

// IbCastIflush with one RDMA read per receive per overlapped segment, reached only when
// rcclIbCastIflushNeedsMultiSeg found a multi-segment handle.
static inline ncclResult_t rcclIbCastIflushMultiSeg(void* recvComm, int n, void** data, int* sizes, void** mhandles,
                                                    void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  bool hasData = false;
  for (int i = 0; i < n; i++)
    if (sizes[i]) hasData = true;
  if (comm->flushEnabled == 0 || !hasData) return ncclSuccess;

  // Validate every receive before posting anything. A multi-receive may use a
  // different MR for each entry, and each entry can overlap multiple segments.
  for (int r = 0; r < n; r++) {
    if (sizes[r] == 0) continue;
    struct ncclIbMrHandle* mhandle = (struct ncclIbMrHandle*)mhandles[r];
    if (mhandle == NULL) {
      WARN("NET/IB: flush receive %d has a NULL mhandle", r);
      return ncclInternalError;
    }
    if (mhandle->nSegments > 1) {
      int flushSeg[NCCL_IB_MAX_SEGMENTS];
      int nFlushSeg =
        ncclIbSegmentsOverlappingRange(mhandle->nSegments, mhandle->segStart, mhandle->segLen, (uintptr_t)data[r],
                                       (size_t)sizes[r], flushSeg, NCCL_IB_MAX_SEGMENTS);
      if (nFlushSeg < 1) {
        WARN("NET/IB: flush buffer %p size %d does not overlap any registered segment", data[r], sizes[r]);
        return ncclInternalError;
      }
      for (int s = 0; s < nFlushSeg; s++) {
        for (int i = 0; i < comm->base.vProps.ndevs; i++) {
          if (mhandle->segMrs[flushSeg[s]][i] == NULL) {
            WARN("NET/IB: flush missing MR for receive %d segment %d device %d", r, flushSeg[s], i);
            return ncclInternalError;
          }
        }
      }
    } else {
      for (int i = 0; i < comm->base.vProps.ndevs; i++) {
        if (mhandle->mrs[i] == NULL) {
          WARN("NET/IB: flush missing MR for receive %d device %d", r, i);
          return ncclInternalError;
        }
      }
    }
  }

  struct ncclIbRequest* req = NULL;
  ncclResult_t ret = ncclSuccess;
  ncclResult_t iflushRet = ncclSuccess;
  NCCLCHECKGOTO(IbCastGetRequest(&comm->base, &req), ret, iflushFail);
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;

  INFO(NCCL_NET, "NET/IB: %s: flush commId=%u group=%d isPrimary=%d ndevs=%d", __func__,
       comm->base.qpSharing.netIbCommId, comm->base.qpSharing.groupIdx, comm->base.qpSharing.isPrimary,
       comm->base.vProps.ndevs);

  // We don't know which devIndex the recv was on, so we flush on all devices
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    uint64_t wrId = (uint64_t)(req - comm->base.reqs) + NCCL_IB_FLUSH_REQ_WR_ID_OFFSET;
    if (IbCastCommIsSharing(&comm->base)) {
      wrId = IbCastEncodeCommId(wrId, comm->base.qpSharing.netIbCommId);
    }
    bool useGpuFlushMem = rcclParamIbCastGdrFlushGpuMemNoRelaxedOrdering() &&
                          comm->devs[i].gpuFlush.gpuFlushGpuMem != nullptr && comm->devs[i].gpuFlush.gpuMr != nullptr;
    if (useGpuFlushMem) {
      struct ibv_send_wr writeWr, wr;
      memset(&writeWr, 0, sizeof(writeWr));
      writeWr.wr_id = wrId;
      writeWr.wr.rdma.remote_addr = (uint64_t)(comm->devs[i].gpuFlush.gpuFlushGpuMem);
      writeWr.wr.rdma.rkey = comm->devs[i].gpuFlush.gpuMr->rkey;
      writeWr.sg_list = &comm->devs[i].gpuFlush.sge;
      writeWr.num_sge = 1;
      writeWr.opcode = IBV_WR_RDMA_WRITE;
      writeWr.send_flags = 0;
      struct ibv_send_wr* badWriteWr;
      NCCLCHECKGOTO(wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, &writeWr, &badWriteWr), ret, iflushFail);
      memset(&wr, 0, sizeof(wr));
      wr.wr_id = wrId;
      wr.wr.rdma.remote_addr = (uint64_t)(comm->devs[i].gpuFlush.gpuFlushGpuMem);
      wr.wr.rdma.rkey = comm->devs[i].gpuFlush.gpuMr->rkey;
      wr.sg_list = &comm->devs[i].gpuFlush.sge;
      wr.num_sge = 1;
      wr.opcode = IBV_WR_RDMA_READ;
      wr.send_flags = IBV_SEND_SIGNALED;
      TRACE(NCCL_NET, "NET/IB: %s: Posting a flush request (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base,
            wr.wr_id);
      TIME_START(4);
      struct ibv_send_wr* bad_wr;
      NCCLCHECKGOTO(wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, &wr, &bad_wr), iflushRet, iflushFail);
      TIME_STOP(4);
    } else {
      struct ibv_send_wr flushWrs[NCCL_NET_IB_MAX_RECVS * NCCL_IB_MAX_SEGMENTS];
      int nFlushWrs = 0;
      for (int r = 0; r < n; r++) {
        if (sizes[r] == 0) continue;
        struct ncclIbMrHandle* mhandle = (struct ncclIbMrHandle*)mhandles[r];
        if (mhandle->nSegments > 1) {
          int flushSeg[NCCL_IB_MAX_SEGMENTS];
          int nFlushSeg =
            ncclIbSegmentsOverlappingRange(mhandle->nSegments, mhandle->segStart, mhandle->segLen, (uintptr_t)data[r],
                                           (size_t)sizes[r], flushSeg, NCCL_IB_MAX_SEGMENTS);
          for (int s = 0; s < nFlushSeg; s++) {
            int seg = flushSeg[s];
            uintptr_t segBase = mhandle->segStart[seg];
            uintptr_t rangeStart = (uintptr_t)data[r];
            memset(&flushWrs[nFlushWrs], 0, sizeof(flushWrs[0]));
            flushWrs[nFlushWrs].wr.rdma.remote_addr = (uint64_t)(rangeStart > segBase ? rangeStart : segBase);
            flushWrs[nFlushWrs].wr.rdma.rkey = mhandle->segMrs[seg][i]->rkey;
            nFlushWrs++;
          }
        } else {
          memset(&flushWrs[nFlushWrs], 0, sizeof(flushWrs[0]));
          flushWrs[nFlushWrs].wr.rdma.remote_addr = (uint64_t)data[r];
          flushWrs[nFlushWrs].wr.rdma.rkey = mhandle->mrs[i]->rkey;
          nFlushWrs++;
        }
      }
      for (int w = 0; w < nFlushWrs; w++) {
        flushWrs[w].wr_id = wrId;
        flushWrs[w].sg_list = &comm->devs[i].gpuFlush.sge;
        flushWrs[w].num_sge = 1;
        flushWrs[w].opcode = IBV_WR_RDMA_READ;
        flushWrs[w].send_flags = (w == nFlushWrs - 1) ? IBV_SEND_SIGNALED : 0;
        flushWrs[w].next = (w + 1 < nFlushWrs) ? &flushWrs[w + 1] : NULL;
      }
      TRACE(NCCL_NET, "NET/IB: %s: Posting a %d-read flush request (req=%p, comm=%p, wr_id=0x%lx)", __func__, nFlushWrs,
            req, req->base, wrId);
      TIME_START(4);
      struct ibv_send_wr* bad_wr = NULL;
      INFO(NCCL_NET, "NET/IB: %s: posting flush dev=%d qpn=%u wr_id=0x%lx", __func__, i,
           comm->devs[i].gpuFlush.qp.qp->qp_num, wrId);
      iflushRet = wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, &flushWrs[0], &bad_wr);
      TIME_STOP(4);
      if (iflushRet != ncclSuccess) {
        // Only the chain tail is signaled, so a partial post never completes on this device.
        goto iflushFail;
      }
    }

    IbCastAddEvent(req, i);

    TRACE(NCCL_NET, "NET/IB: %s: Flush request posted (req=%p, comm=%p, wr_id=0x%lx)", __func__, req, req->base, wrId);
  }

  *request = req;
  return ncclSuccess;

iflushFail:
  IbCastStatsFatalError(&comm->base.stats);
  if (req) {
    if (IbCastRequestHasEvents(req)) {
      req->type = NCCL_NET_IB_REQ_FAILED;
      *request = req;
      return ncclSuccess;
    }
    IbCastFreeRequest(req);
  }
  *request = NULL;
  return (iflushRet != ncclSuccess) ? iflushRet : ret;
}

#endif // NET_IB_CAST_P2P_MULTISEG_CAST_H_
