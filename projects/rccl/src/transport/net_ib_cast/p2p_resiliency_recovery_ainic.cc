/*************************************************************************
 * Copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/* AINIC-specific QP recovery operations.
 *
 * On AINIC, ibv_modify_qp(RESET) is broken — QPs must be destroyed and
 * recreated to obtain a fresh QPN. The new QPNs are exchanged with the
 * remote peer via the ACK payload before RTR/RTS (two-phase handshake).
 *
 * Entry points are invoked by p2p_resiliency_recovery.cc from the
 * IbCastPortRecoveryQpsRestore / IbCastPortRecoveryPostAck flows.
 */

#include "p2p_resiliency_recovery_internal.h"
#include "p2p_resiliency_recovery_ainic.h"
#include "connect_cast.h"

/* Phase A: destroy+recreate QPs, recording new QPNs for exchange. */
ncclResult_t IbCastPortRecoveryQpsRestoreAinic(struct ncclIbPortRecoveryContext* recoveryContext) {
  uint nqps = recoveryContext->resCtx->baseComm->nqps;
  recoveryContext->nLocalQpns = 0;

  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    ncclIbQp* localQp = &recoveryContext->resCtx->baseComm->qps[qpIndex];
    if (localQp->devIndex != recoveryContext->devIndex) {
      continue;
    }

    uint32_t oldQpn = localQp->qp->qp_num;
    INFO(NCCL_NET, "NET/IB: %s: Destroying QP %d on device %d (comm=%p, old_qp_num=%u)",
         __func__, qpIndex, recoveryContext->devIndex,
         recoveryContext->resCtx->baseComm, oldQpn);

    NCCLCHECK(wrap_ibv_destroy_qp(localQp->qp));
    localQp->qp = NULL;

    struct ncclIbQpCreateAttr createAttr;
    IbCastBuildDataQpCreateAttr(recoveryContext->resCtx->baseComm, recoveryContext->devIndex, &createAttr);
    createAttr.channelId = localQp->channelId;
    createAttr.isDataQp = localQp->isDataQp;
    // isCtsEnabled and ctsQpSlot are left at 0 (from memset in IbCastBuildDataQpCreateAttr).
    // CTS offload is disabled when resiliency features are enabled (init.cc).
    NCCLCHECK(IbCastQpCreate(localQp, &createAttr));

    INFO(NCCL_NET, "NET/IB: %s: Recreated QP %d on device %d (comm=%p, old_qp_num=%u, new_qp_num=%u)",
         __func__, qpIndex, recoveryContext->devIndex,
         recoveryContext->resCtx->baseComm, oldQpn, localQp->qp->qp_num);

    NCCLCHECK(IbCastQpInit(localQp));

    int idx = recoveryContext->nLocalQpns;
    recoveryContext->localQpns[idx].qpIndex = qpIndex;
    recoveryContext->localQpns[idx].newQpn = localQp->qp->qp_num;
    recoveryContext->nLocalQpns++;
  }

  // Also destroy+recreate flush QP on receiver side (it connects to itself)
  if (!recoveryContext->resCtx->baseComm->isSend) {
    ncclIbRecvComm* recvComm = (ncclIbRecvComm*)recoveryContext->resCtx->baseComm;
    if (recvComm->flushEnabled) {
      for (int i = 0; i < recvComm->base.vProps.ndevs; i++) {
        if (i != recoveryContext->devIndex) continue;
        struct ncclIbRecvCommDev* rCommDev = &recvComm->devs[i];
        ncclIbQp* flushQp = &rCommDev->gpuFlush.qp;
        INFO(NCCL_NET, "NET/IB: %s: Destroying Flush QP on device %d (comm=%p, old_qp_num=%u)",
             __func__, i, recoveryContext->resCtx->baseComm, flushQp->qp->qp_num);
        NCCLCHECK(wrap_ibv_destroy_qp(flushQp->qp));
        flushQp->qp = NULL;
        struct ncclIbQpCreateAttr flushCreateAttr;
        IbCastBuildDataQpCreateAttr(recoveryContext->resCtx->baseComm, recoveryContext->devIndex, &flushCreateAttr);
        flushCreateAttr.maxRecvWorkRequest = 0;
        flushCreateAttr.maxSendWorkRequest = NET_IB_MAX_REQUESTS;
        flushCreateAttr.channelId = flushQp->channelId;
        flushCreateAttr.isDataQp = flushQp->isDataQp;
        NCCLCHECK(IbCastQpCreate(flushQp, &flushCreateAttr));
        INFO(NCCL_NET, "NET/IB: %s: Recreated Flush QP on device %d (comm=%p, new_qp_num=%u)",
             __func__, i, recoveryContext->resCtx->baseComm, flushQp->qp->qp_num);
        NCCLCHECK(IbCastQpInit(flushQp));
      }
    }
  }

  INFO(NCCL_NET, "NET/IB: %s: Phase A complete: %d QPs destroyed+recreated on device %d (comm=%p)",
       __func__, recoveryContext->nLocalQpns, recoveryContext->devIndex,
       recoveryContext->resCtx->baseComm);
  return ncclSuccess;
}

/* Phase B: transition recreated QPs to RTR+RTS using remote QPNs from ACK.
 * No-op on non-AINIC paths. */
ncclResult_t IbCastPortRecoveryQpsToRtsAinic(struct ncclIbPortRecoveryContext* recoveryContext) {
  if (!IbCastAinicRoce) return ncclSuccess;
  uint nqps = recoveryContext->resCtx->baseComm->nqps;

  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    ncclIbQp* localQp = &recoveryContext->resCtx->baseComm->qps[qpIndex];
    if (localQp->devIndex != recoveryContext->devIndex) {
      continue;
    }

    uint32_t remoteQpn = 0;
    bool found = false;
    for (int r = 0; r < recoveryContext->nRemoteQpns; r++) {
      if (recoveryContext->remoteQpns[r].qpIndex == qpIndex) {
        remoteQpn = recoveryContext->remoteQpns[r].newQpn;
        found = true;
        break;
      }
    }
    if (!found) {
      WARN("NET/IB: %s: No remote QPN found for QP index %d on device %d (comm=%p)",
           __func__, qpIndex, recoveryContext->devIndex,
           recoveryContext->resCtx->baseComm);
      return ncclInternalError;
    }

    localQp->rtrAttr.remoteQpNum = remoteQpn;
    INFO(NCCL_NET, "NET/IB: %s: QP %d RTR with remoteQpn=%u (comm=%p, qp_num=%u)",
         __func__, qpIndex, remoteQpn, recoveryContext->resCtx->baseComm,
         localQp->qp->qp_num);

    if (localQp->eceSupported) {
      NCCLCHECK(wrap_ibv_set_ece(localQp->qp, &localQp->ece, &localQp->eceSupported));
    }
    NCCLCHECK(IbCastQpRtr(localQp));
    NCCLCHECK(IbCastQpRts(localQp));

    INFO(NCCL_NET, "NET/IB: %s: Restored QP %d on device %d (comm=%p, qp_num=%u)",
         __func__, qpIndex, recoveryContext->devIndex,
         recoveryContext->resCtx->baseComm, localQp->qp->qp_num);

    if (!recoveryContext->resCtx->baseComm->isSend) {
      struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)recoveryContext->resCtx->baseComm;
      NCCLCHECK(IbCastPostReceiveWorkRequestsOnQp(recvComm, localQp));
    }
  }

  // Restore Flush QP on receiver (self-loopback — remote QPN is own new QPN)
  if (!recoveryContext->resCtx->baseComm->isSend) {
    ncclIbRecvComm* recvComm = (ncclIbRecvComm*)recoveryContext->resCtx->baseComm;
    if (recvComm->flushEnabled) {
      for (int i = 0; i < recvComm->base.vProps.ndevs; i++) {
        if (i != recoveryContext->devIndex) continue;
        struct ncclIbRecvCommDev* rCommDev = &recvComm->devs[i];
        ncclIbQp* flushQp = &rCommDev->gpuFlush.qp;
        flushQp->rtrAttr.remoteQpNum = flushQp->qp->qp_num;
        NCCLCHECK(IbCastQpRtr(flushQp));
        NCCLCHECK(IbCastQpRts(flushQp));
        INFO(NCCL_NET, "NET/IB: %s: Restored Flush QP on device %d (comm=%p, qp_num=%u)",
             __func__, i, recoveryContext->resCtx->baseComm, flushQp->qp->qp_num);
      }
    }
  }

  return ncclSuccess;
}

/* Post a UD ACK carrying local QPN entries as payload. */
ncclResult_t IbCastPortRecoveryPostAckWithQpnsAinic(struct ncclIbPortRecoveryContext* recoveryContext, bool* success) {
  int devIdx = recoveryContext->devIndex;
  struct ncclIbResiliency* resCtx = recoveryContext->resCtx;

  if (recoveryContext->nLocalQpns > NCCL_IB_MAX_RECOVERY_QPN_ENTRIES) {
    WARN("NET/IB: %s: Too many QPN entries (%d) for recovery ACK payload (max %d)",
         __func__, recoveryContext->nLocalQpns, NCCL_IB_MAX_RECOVERY_QPN_ENTRIES);
    return ncclInternalError;
  }
  int payloadSize = recoveryContext->nLocalQpns * sizeof(struct RecoveryQpnEntry);
  memcpy(resCtx->portRecoveryQpnBuf, recoveryContext->localQpns, payloadSize);

  struct ibv_sge qpnSge;
  qpnSge.addr = (uintptr_t)resCtx->portRecoveryQpnBuf;
  qpnSge.length = payloadSize;
  qpnSge.lkey = resCtx->devs[devIdx].portRecoveryQpnMr->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = recoveryContext->aliveMsgNextId;
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.imm_data = NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.sg_list = &qpnSge;
  wr.num_sge = 1;
  wr.wr.ud.ah = resCtx->portRecoveryAh[devIdx];
  wr.wr.ud.remote_qpn = resCtx->portRecoveryRemoteQpn[devIdx];
  wr.wr.ud.remote_qkey = NCCL_IB_RECOVERY_QKEY;

  INFO(NCCL_NET, "NET/IB: %s: %s posting ACK with %d QPN entries for device %d (comm=%p)",
       __func__, resCtx->baseComm->isSend ? "Sender" : "Receiver",
       recoveryContext->nLocalQpns, devIdx, resCtx->baseComm);

  struct ibv_send_wr* bad_wr = NULL;
  struct ncclIbQp* recoveryQp = &resCtx->portRecoveryQps[devIdx];
  if (wrap_ibv_post_send(recoveryQp->qp, &wr, &bad_wr) != ncclSuccess) {
    INFO(NCCL_NET, "NET/IB: %s: Failed to post ACK with QPNs on device %d (comm=%p, qp_num=%u)",
         __func__, devIdx, resCtx->baseComm, recoveryQp->qp->qp_num);
    *success = false;
    return ncclSuccess;
  }
  recoveryContext->ackPosted = true;
  *success = true;
  return ncclSuccess;
}

/* Extract remote QPNs from a received ACK completion buffer (after GRH).
 * No-op on non-AINIC paths. */
void IbCastPortRecoveryExtractQpnsAinic(struct ncclIbPortRecoveryContext* recoveryContext, struct ibv_wc* wc) {
  if (!IbCastAinicRoce) return;
  struct ncclIbResiliencyDev* resDev = &recoveryContext->resCtx->devs[recoveryContext->devIndex];
  int payloadLen = (int)(wc->byte_len - NCCL_IB_UD_GRH_SIZE);
  if (payloadLen <= 0) {
    recoveryContext->nRemoteQpns = 0;
    return;
  }
  int nEntries = payloadLen / sizeof(struct RecoveryQpnEntry);
  if (nEntries > NCCL_IB_MAX_QPS) nEntries = NCCL_IB_MAX_QPS;

  uint8_t* payload = resDev->portRecoveryGrhBuf + NCCL_IB_UD_GRH_SIZE;
  memcpy(recoveryContext->remoteQpns, payload, nEntries * sizeof(struct RecoveryQpnEntry));
  recoveryContext->nRemoteQpns = nEntries;

  for (int i = 0; i < nEntries; i++) {
    INFO(NCCL_NET, "NET/IB: %s: Extracted remote QPN: qpIndex=%u newQpn=%u (comm=%p)",
         __func__, recoveryContext->remoteQpns[i].qpIndex,
         recoveryContext->remoteQpns[i].newQpn,
         recoveryContext->resCtx->baseComm);
  }
}
