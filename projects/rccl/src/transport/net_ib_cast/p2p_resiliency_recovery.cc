/*************************************************************************
 * Copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "p2p_resiliency_recovery_cast.h"
#include <list>

NCCL_PARAM(IbCastResiliencyPortRecovery, "IB_RESILIENCY_PORT_RECOVERY", 0);
NCCL_PARAM(IbCastResiliencyPortRecoveryStartDelay, "IB_RESILIENCY_PORT_RECOVERY_START_DELAY", 200); // In milliseconds
NCCL_PARAM(IbCastResiliencyPortRecoveryAliveMsgBatchInterval, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_INTERVAL", 500); // In milliseconds
NCCL_PARAM(IbCastResiliencyPortRecoveryAliveMsgBatchSize, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE", 5);
NCCL_PARAM(IbCastResiliencyPortRecoveryAliveMsgSequenceSize, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE", 5);
NCCL_PARAM(IbCastResiliencyPortRecoveryAliveMsgTimeout, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_TIMEOUT", 4000); // In milliseconds
NCCL_PARAM(IbCastResiliencyPortRecoveryAckTimeout, "IB_RESILIENCY_PORT_RECOVERY_ACK_TIMEOUT", 5000); // In milliseconds
NCCL_PARAM(IbCastResiliencyPortRecoveryAttemptsMax, "IB_RESILIENCY_PORT_RECOVERY_ATTEMPTS_MAX", 5);

extern int64_t ncclParamIbCastPkey();

// Used to convert milliseconds to nanoseconds
#define MSEC_TO_NSEC 1000000ULL

#define NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE_MIN 1

// Determines the size of the work queue on the recovery QPs
#define NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX 10

// The asynchronous thread should be be able to handle the CQ fast enough so
// no more than two batches of "alive" messsages should be pending in the CQ at
// any time.
#define NCCL_IB_RESILIENCY_PORT_RECOVERY_CQ_SIZE (NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX*2)

#define NCCL_IB_RECOVERY_QKEY 0x1234ABCD

static ncclResult_t IbCastPortRecoveryQpInitUd(struct ncclIbQp* qp, int pkeyIndex, int portNum) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = pkeyIndex;
  attr.port_num = portNum;
  attr.qkey = NCCL_IB_RECOVERY_QKEY;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &attr,
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY));
  return ncclSuccess;
}

static ncclResult_t IbCastPortRecoveryQpRtrUd(struct ncclIbQp* qp) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &attr, IBV_QP_STATE));
  return ncclSuccess;
}

static ncclResult_t IbCastPortRecoveryQpRtsUd(struct ncclIbQp* qp) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN));
  return ncclSuccess;
}

// Asynchronous thread to handle port recovery operations
std::thread IbCastPortRecoveryAsyncThread;

// Reference count of active resiliency contexts using port recovery
static std::atomic<int> IbCastPortRecoveryRefCount(0);

// Flag indicating the recovery thread is active - cleared by IbCastPortRecoveryThreadStop() to signal the async thread to exit
static std::atomic<bool> IbCastPortRecoveryThreadActive(false);

// Mutex protecting shared state between the async recovery thread and callers:
// - recoveryInbox: shared list where producers add new failed devices to be
//   processed by the async thread
// - recoveryCloseRequests: vector of close requests from CommClose callers
// - closeReq->completed: flag signaling close completion to waiting callers
//
// The async thread splices recoveryInbox into a thread-local recoveryQueue
// under the lock, then iterates the local queue without holding the lock.
// This avoids holding the mutex during potentially long recovery processing
// while preventing data races on the shared inbox.
//
// closeReq->completed must be set under the lock to prevent a lost-wakeup
// race on IbCastPortRecoveryCloseCond: the closing thread could evaluate the
// wait predicate (false), then the async thread sets completed and notifies,
// then the closing thread sleeps on the condvar and never wakes.
//
// Usage pattern:
// 1. IbCastPortRecoveryHandleFailure: acquires lock to push to recoveryInbox
// 2. IbCastPortRecoveryClose: acquires lock to add close request, then waits on IbCastPortRecoveryCloseCond
// 3. IbCastPortRecoveryAsyncThreadMain: acquires lock to splice inbox, swap close requests;
//    processes both outside the lock; re-acquires lock to set completed and notify
static std::mutex IbCastPortRecoveryMutex;
static std::condition_variable IbCastPortRecoveryCond;

struct ncclIbPortRecoveryCloseRequest {
  struct ncclIbResiliency* resCtx;
  bool completed;
};

// Shared vector of close requests - protected by IbCastPortRecoveryMutex
static std::vector<ncclIbPortRecoveryCloseRequest*> recoveryCloseRequests;
// Condition variable to signal completion of close requests
static std::condition_variable IbCastPortRecoveryCloseCond;

enum ncclIbPortRecoveryState {
  // Recovery protocol has not started yet. Before moving to the next state,
  // the "data CQs" and "data QPs" of the failed device should be drained.
  ncclIbPortRecoveryStateInit,
  // On the requestor's side: The requestor posts "alive" messages and
  // responder receives them. After posting a batch of "alive" messages, and
  // waiting for their local completion, the requestor transitions to the "peer
  // ack" state waiting for an "ack" from the responder. If after a while no
  // "ack" is received, the requestor replays this state and retransmits more
  // "alive" messages.
  // On the responder's side: The responder, waits until all expected "alive"
  // messages are received and then sends an "ack" to the requestor and waits
  // for the "ack" to complete locally. If the responder does not receive a
  // consecutive sequence of "alive" messages, it replays this state and waits
  // for more "alive" messages.
  ncclIbPortRecoveryStateAliveMessages,
  // On the requestor's side: The requestor is waiting to receive an "ack" from
  // the responder. After receiving the "ack" message from the responder, the
  // requestor posts the "final ack" message to the responder and waits for its
  // local completion. After this local completion, the requestor concludes this
  // phase and determines the port has recovered.
  // On the responder's side: The responder waits for the "final ack" message
  // from the requestor. After receiving the "final ack" message, the responder
  // concludes the recovery protocol and determines the port has recovered.
  // If a "final ack" is not received within some time window, the responder
  // goes back to the "alive messages" state.
  ncclIbPortRecoveryStateAck,
  // This state represents a state where the failed device has completed the
  // recovery protocol successfully, "data QPs" were restored and and sender
  // and receiver can now resume posting new work requests using on the device.
  ncclIbPortRecoveryStateSuccess,
  // This state represents a state where the failed device could not be
  // recovered.
  ncclIbPortRecoveryStateFailed,
};

struct ncclIbPortRecoveryContext {
  struct ncclIbResiliency* resCtx;
  // The index of the device that is recovered.
  int devIndex;
  enum ncclIbPortRecoveryState state;

  struct ibv_cq* recoveryCq;

  // The time when the recovery context was initialized
  uint64_t timeInit;

  int nFailedAttempts;

  // Timestamp of last issued message.
  uint64_t timeLastMsg;

  uint32_t aliveMsgNextId;

  // Indicates whether an "ack" message was recived from the peer.
  bool ackReceived;

  // Indicates whether an "ack" message was posted to the peer.
  bool ackPosted;
  // Indicates whether an "ack" message posted to the peer was completed locally.
  bool ackCompleted;

  union {
    struct {
      bool aliveMsgPosted;
      bool aliveMsgCompleted;
    } send;
    struct {
      uint32_t windowBase;
      uint32_t receivedBits;
    } recv;
  };
};

// Shared inbox for new recovery contexts - protected by IbCastPortRecoveryMutex.
// Producer threads add items here; the async thread drains it under lock.
static std::list<ncclIbPortRecoveryContext*> recoveryInbox;

static inline ncclResult_t IbCastPortRecoveryQpsToError(ncclIbPortRecoveryContext* recoveryContext) {
  uint nqps = recoveryContext->resCtx->baseComm->nqps;
  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    ncclIbQp* localQp = &recoveryContext->resCtx->baseComm->qps[qpIndex];
    if (localQp->devIndex != recoveryContext->devIndex) {
      // The QP is not on the failed device; skip
      continue;
    }
    INFO(NCCL_NET, "NET/IB: %s: Transitioning QP %u on device %d to ERROR state (comm=%p, devIndex=%d, qp_num=%u)", __func__, localQp->qp->qp_num, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryContext->devIndex, localQp->qp->qp_num);
    NCCLCHECK(IbCastQpError(localQp));
  }
  return ncclSuccess;
}

// Predefined work ID for receive work request for port recovery messages
#define NCCL_IB_PORT_RECOVERY_WR_ID (0xAAAA)

inline static ncclResult_t IbCastPortRecoveryPostRecvWorkRequest(struct ibv_qp* qp, struct ncclIbResiliencyDev* resDev) {
  struct ibv_sge sge;
  sge.addr = (uintptr_t)resDev->portRecoveryGrhBuf;
  sge.length = sizeof(resDev->portRecoveryGrhBuf);
  sge.lkey = resDev->portRecoveryGrhMr->lkey;
  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = NCCL_IB_PORT_RECOVERY_WR_ID;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  struct ibv_recv_wr* bad_wr;
  return wrap_ibv_post_recv(qp, &wr, &bad_wr);
}

// Helper function to drain CQEs on the provided CQ and fill a Receive WQE for
// every CQE drained on the provided QP.
static ncclResult_t IbCastPortRecoveryDrainCqAndPostReceiveWRs(struct ncclIbPortRecoveryContext* recoveryContext, bool* success) {
  struct ibv_cq* cq = recoveryContext->recoveryCq;
  struct ibv_qp* qp = recoveryContext->resCtx->portRecoveryQps[recoveryContext->devIndex].qp;
  int wrDone = 0;
  int numMsgs = NCCL_IB_RESILIENCY_PORT_RECOVERY_CQ_SIZE;
  struct ibv_wc wcs[NCCL_IB_RESILIENCY_PORT_RECOVERY_CQ_SIZE];

  ncclResult_t ret = ncclSuccess;
  do {
    ret = wrap_ibv_poll_cq(cq, numMsgs, wcs, &wrDone);
    if (ret != ncclSuccess) {
      INFO(NCCL_NET, "NET/IB: %s: Failed to poll recovery CQ %p for device %d (comm=%p)", __func__, cq, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      *success = false;
      return ncclSuccess;
    }
    INFO(NCCL_NET, "NET/IB: %s: Drained %d CQEs from recovery CQ %p for device %d (comm=%p)", __func__, wrDone, cq, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
    for (int i = 0; i < wrDone; i++) {
      ret = IbCastPortRecoveryPostRecvWorkRequest(qp, &recoveryContext->resCtx->devs[recoveryContext->devIndex]);
      if (ret != ncclSuccess) {
        INFO(NCCL_NET, "NET/IB: %s: Failed to post recv WR on recovery QP %p for device %d (comm=%p)", __func__, qp, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
        *success = false;
        return ncclSuccess;
      }
    }
  } while (wrDone > 0);
  *success = true;
  return ncclSuccess;
}

static inline ncclResult_t IbCastPortRecoveryContextInit(struct ncclIbResiliency* resCtx, int failedDevIndex, ncclIbPortRecoveryContext** outRecoveryCtx) {
  ncclResult_t res = ncclSuccess;
  if (!outRecoveryCtx) return ncclInternalError;
  ncclIbPortRecoveryContext* recoveryCtx = (ncclIbPortRecoveryContext*)malloc(sizeof(ncclIbPortRecoveryContext));
  if (!recoveryCtx) {
    WARN("NET/IB: %s: Failed to allocate failure queue node (comm=%p)", __func__, resCtx->baseComm);
    *outRecoveryCtx = NULL;
    return ncclInternalError;
  }

  *outRecoveryCtx = NULL;

  // Initialize all fields
  memset(recoveryCtx, 0, sizeof(ncclIbPortRecoveryContext));
  recoveryCtx->resCtx = resCtx;
  recoveryCtx->devIndex = failedDevIndex;
  recoveryCtx->state = ncclIbPortRecoveryStateInit;
  recoveryCtx->timeInit = clockNano();
  recoveryCtx->nFailedAttempts = 0;
  recoveryCtx->timeLastMsg = 0;
  recoveryCtx->aliveMsgNextId = 0;
  recoveryCtx->ackReceived = false;
  recoveryCtx->ackPosted = false;
  recoveryCtx->ackCompleted = false;

  for (int i = 0; i < resCtx->ndevs; i++) {
    if (i != failedDevIndex) continue;
    recoveryCtx->recoveryCq = resCtx->devs[i].portRecoveryCq;
    break;
  }

  // Transitioning all QPs on the failed device to ERROR state
  res = IbCastPortRecoveryQpsToError(recoveryCtx);
  if (res != ncclSuccess) {
    INFO(NCCL_NET, "NET/IB: %s: Failed to transition QPs to ERROR state for device %d (comm=%p)", __func__, failedDevIndex, resCtx->baseComm);
    free(recoveryCtx);
    return res;
  }

  if (resCtx->baseComm->isSend) {
    recoveryCtx->send.aliveMsgPosted = false;
    recoveryCtx->send.aliveMsgCompleted = false;
    // Required for the ACK from the receiver
    INFO(NCCL_NET, "NET/IB: %s: Sender posting initial (%d) recv WRs for port recovery for device %d (comm=%p)", __func__, 1, recoveryCtx->devIndex, recoveryCtx->resCtx->baseComm);
    res = IbCastPortRecoveryPostRecvWorkRequest(recoveryCtx->resCtx->portRecoveryQps[recoveryCtx->devIndex].qp, &recoveryCtx->resCtx->devs[recoveryCtx->devIndex]);
    if (res != ncclSuccess) {
      INFO(NCCL_NET, "NET/IB: %s: Sender failed to post recv WR on recovery QP for device %d (comm=%p)", __func__, recoveryCtx->devIndex, recoveryCtx->resCtx->baseComm);
      free(recoveryCtx);
      return res;
    }
  } else {
    recoveryCtx->recv.windowBase = 0;
    recoveryCtx->recv.receivedBits = 0;
    // Required for the alive messages and final ACK from the sender
    INFO(NCCL_NET, "NET/IB: %s: Receiver posting initial (%d) recv WRs for port recovery for device %d (comm=%p)", __func__, NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX, recoveryCtx->devIndex, recoveryCtx->resCtx->baseComm);
    for (int i = 0; i < NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX; i++) {
      res = IbCastPortRecoveryPostRecvWorkRequest(recoveryCtx->resCtx->portRecoveryQps[recoveryCtx->devIndex].qp, &recoveryCtx->resCtx->devs[recoveryCtx->devIndex]);
      if (res != ncclSuccess) {
        INFO(NCCL_NET, "NET/IB: %s: Receiver failed to post recv WR on recovery QP for device %d (comm=%p)", __func__, recoveryCtx->devIndex, recoveryCtx->resCtx->baseComm);
        free(recoveryCtx);
        return res;
      }
    }
  }

  *outRecoveryCtx = recoveryCtx;
  return ncclSuccess;
}

static inline ncclResult_t IbCastPortRecoveryContextDestroy(ncclIbPortRecoveryContext* recoveryContext) {
  assert(recoveryContext);
  INFO(NCCL_NET, "NET/IB: %s: Destroying port recovery context for device %d (%s comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm);
  free(recoveryContext);
  return ncclSuccess;
}

ncclResult_t IbCastPortRecoveryDevInit(struct ncclIbResiliency* resCtx, int devIndex, ncclIbDev* ibDev) {
  assert(resCtx->recoveryEnabled);
  struct ncclIbResiliencyDev* resDev = &resCtx->devs[devIndex];
  void* cqContext = (void*)&resCtx->baseComm->stats;
  NCCLCHECK(wrap_ibv_create_cq(&resDev->portRecoveryCq, ibDev->context, NCCL_IB_RESILIENCY_PORT_RECOVERY_CQ_SIZE, cqContext, NULL, 0));
  memset(resDev->portRecoveryGrhBuf, 0, sizeof(resDev->portRecoveryGrhBuf));
  NCCLCHECK(wrap_ibv_reg_mr(&resDev->portRecoveryGrhMr, ibDev->pd, resDev->portRecoveryGrhBuf, sizeof(resDev->portRecoveryGrhBuf), IBV_ACCESS_LOCAL_WRITE));
  INFO(NCCL_NET, "NET/IB: %s: Created port recovery CQ and GRH MR for resiliency context (%s comm=%p) on device %d", __func__, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm, devIndex);
  return ncclSuccess;
}

ncclResult_t IbCastPortRecoveryDevDestroy(struct ncclIbResiliency* resCtx, int devIndex) {
  struct ncclIbResiliencyDev* resDev = &resCtx->devs[devIndex];
  if (resDev->portRecoveryGrhMr) {
    wrap_ibv_dereg_mr(resDev->portRecoveryGrhMr);
    resDev->portRecoveryGrhMr = nullptr;
  }
  if (resDev->portRecoveryCq) {
    NCCLCHECK(wrap_ibv_destroy_cq(resDev->portRecoveryCq));
  }
  INFO(NCCL_NET, "NET/IB: %s: Destroyed port recovery CQ and GRH MR on device %d for resiliency context (comm=%p)", __func__, devIndex, resCtx->baseComm);
  return ncclSuccess;
}

ncclResult_t IbCastPortRecoverySenderQpsCreate(struct ncclIbResiliency* resCtx, struct ncclIbQpInfo* localPortRecoveryQpsInfo, int nQps) {
  assert(nQps > 0);
  assert(resCtx->recoveryEnabled);
  ncclIbSendComm* sendComm = (ncclIbSendComm*)resCtx->baseComm;
  void* qpContext = (void*)&sendComm->base.stats;
  struct ncclIbQpCreateAttr qpCreateAttrs;
  memset(&qpCreateAttrs, 0, sizeof(qpCreateAttrs));
  qpCreateAttrs.type = IBV_QPT_UD;
  qpCreateAttrs.maxRecvWorkRequest = 1;
  qpCreateAttrs.maxSendWorkRequest = NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX;
  for (int localQpIndex = 0; localQpIndex < nQps; localQpIndex++) {
    int localDevIndex = localQpIndex % sendComm->base.vProps.ndevs;
    ncclIbSendCommDev* sendCommDev = &sendComm->devs[localDevIndex];
    ncclIbDev* ibDev = &IbCastDevs[sendCommDev->base.ibDevN];
    ncclIbQp* localQp = &resCtx->portRecoveryQps[localQpIndex];
    qpCreateAttrs.cq = resCtx->devs[localDevIndex].portRecoveryCq;
    qpCreateAttrs.pd = sendCommDev->base.pd;
    qpCreateAttrs.qpContext = qpContext;
    NCCLCHECK(IbCastQpCreate(localQp, &qpCreateAttrs));
    ncclIbQpInfo* localQpInfo = &localPortRecoveryQpsInfo[localQpIndex];
    localQpInfo->qpn = localQp->qp->qp_num;
    localQpInfo->devIndex = localDevIndex;

    NCCLCHECK(IbCastPortRecoveryQpInitUd(localQp, ncclParamIbCastPkey(), ibDev->portNum));
  }
  return ncclSuccess;
}

ncclResult_t IbCastPortRecoverySenderQpsToRts(struct ncclIbResiliency* resCtx, struct ncclIbConnectionMetadata* remInfo, int nQps) {
  assert(nQps > 0);
  assert(resCtx->recoveryEnabled);
  ncclIbSendComm* sendComm = (ncclIbSendComm*)resCtx->baseComm;
  ncclIbQp* localQp = NULL;
  ncclIbQpInfo* remQpInfo = NULL;
  for (int localQpIndex = 0; localQpIndex < nQps; localQpIndex++) {
    int localDevIndex = localQpIndex % sendComm->base.vProps.ndevs;
    ncclIbSendCommDev* sendCommDev = &sendComm->devs[localDevIndex];
    ncclIbDev* ibDev = &IbCastDevs[sendCommDev->base.ibDevN];
    localQp = &resCtx->portRecoveryQps[localQpIndex];
    remQpInfo = &(remInfo->resiliencyInfo.portRecoveryQpsInfo[localQpIndex]);
    localQp->remDevIdx = remQpInfo->devIndex;
    ncclIbDevInfo* remDevInfo = &remInfo->devs[remQpInfo->devIndex];

    NCCLCHECK(IbCastPortRecoveryQpRtrUd(localQp));
    NCCLCHECK(IbCastPortRecoveryQpRtsUd(localQp));

    // Create Address Handle for UD sends to remote recovery QP
    struct ibv_ah_attr ahAttr;
    memset(&ahAttr, 0, sizeof(ahAttr));
    ahAttr.port_num = ibDev->portNum;
    ahAttr.sl = remInfo->sl;
    if (remDevInfo->link_layer == IBV_LINK_LAYER_ETHERNET) {
      ahAttr.is_global = 1;
      ahAttr.grh.dgid = remDevInfo->gid;
      ahAttr.grh.sgid_index = sendCommDev->base.gidInfo.localGidIndex;
      ahAttr.grh.hop_limit = 255;
      ahAttr.grh.traffic_class = remInfo->tc;
    } else {
      ahAttr.dlid = remDevInfo->lid;
    }
    resCtx->portRecoveryAh[localDevIndex] = sendCommDev->base.pd->context->ops.create_ah(sendCommDev->base.pd, &ahAttr);
    if (!resCtx->portRecoveryAh[localDevIndex]) {
      WARN("NET/IB: %s: Failed to create AH for recovery QP on device %d (comm=%p)", __func__, localDevIndex, resCtx->baseComm);
      return ncclSystemError;
    }
    resCtx->portRecoveryRemoteQpn[localDevIndex] = remQpInfo->qpn;

    INFO(NCCL_NET, "NET/IB: %s: To RTS done on recovery UD QP (index=%d, qp_num=%u, remote_qpn=%u, deviceIndex=%d, comm=%p)", __func__, localQpIndex, localQp->qp->qp_num, remQpInfo->qpn, localDevIndex, resCtx->baseComm);
  }
  return ncclSuccess;
}

ncclResult_t IbCastPortRecoveryReceiverQpsCreateToRts(struct ncclIbResiliency* resCtx, struct ncclIbConnectionMetadata* remInfo, struct ncclIbQpInfo* localPortRecoveryQpsInfo, int nQps) {
  ncclIbRecvComm* recvComm = (ncclIbRecvComm*)resCtx->baseComm;
  void* qpContext = (void*)&recvComm->base.stats;
  struct ncclIbQpCreateAttr qpCreateAttrs;
  memset(&qpCreateAttrs, 0, sizeof(qpCreateAttrs));
  qpCreateAttrs.type = IBV_QPT_UD;
  qpCreateAttrs.maxRecvWorkRequest = NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX;
  qpCreateAttrs.maxSendWorkRequest = 1;
  for (int localQpIndex = 0; localQpIndex < nQps; localQpIndex++) {
    int localDevIndex = localQpIndex % recvComm->base.vProps.ndevs;
    ncclIbRecvCommDev* recvCommDev = &recvComm->devs[localDevIndex];
    ncclIbDev* ibDev = &IbCastDevs[recvCommDev->base.ibDevN];
    ncclIbQp* localQp = &resCtx->portRecoveryQps[localQpIndex];
    qpCreateAttrs.cq = resCtx->devs[localDevIndex].portRecoveryCq;
    qpCreateAttrs.pd = recvCommDev->base.pd;
    qpCreateAttrs.qpContext = qpContext;
    NCCLCHECK(IbCastQpCreate(localQp, &qpCreateAttrs));
    localPortRecoveryQpsInfo[localQpIndex].qpn = localQp->qp->qp_num;
    localPortRecoveryQpsInfo[localQpIndex].devIndex = localDevIndex;

    NCCLCHECK(IbCastPortRecoveryQpInitUd(localQp, ncclParamIbCastPkey(), ibDev->portNum));

    ncclIbQpInfo* remQpInfo = &remInfo->resiliencyInfo.portRecoveryQpsInfo[localQpIndex];
    ncclIbDevInfo* remDevInfo = &remInfo->devs[remQpInfo->devIndex];

    NCCLCHECK(IbCastPortRecoveryQpRtrUd(localQp));
    NCCLCHECK(IbCastPortRecoveryQpRtsUd(localQp));

    // Create Address Handle for UD sends (ACK messages) to remote recovery QP
    struct ibv_ah_attr ahAttr;
    memset(&ahAttr, 0, sizeof(ahAttr));
    ahAttr.port_num = ibDev->portNum;
    ahAttr.sl = remInfo->sl;
    if (remDevInfo->link_layer == IBV_LINK_LAYER_ETHERNET) {
      ahAttr.is_global = 1;
      ahAttr.grh.dgid = remDevInfo->gid;
      ahAttr.grh.sgid_index = recvCommDev->base.gidInfo.localGidIndex;
      ahAttr.grh.hop_limit = 255;
      ahAttr.grh.traffic_class = remInfo->tc;
    } else {
      ahAttr.dlid = remDevInfo->lid;
    }
    resCtx->portRecoveryAh[localDevIndex] = recvCommDev->base.pd->context->ops.create_ah(recvCommDev->base.pd, &ahAttr);
    if (!resCtx->portRecoveryAh[localDevIndex]) {
      WARN("NET/IB: %s: Failed to create AH for recovery QP on device %d (comm=%p)", __func__, localDevIndex, resCtx->baseComm);
      return ncclSystemError;
    }
    resCtx->portRecoveryRemoteQpn[localDevIndex] = remQpInfo->qpn;
    INFO(NCCL_NET, "NET/IB: %s: To RTS done on recovery UD QP (index=%d, qp_num=%u, remote_qpn=%u, deviceIndex=%d, comm=%p)", __func__, localQpIndex, localQp->qp->qp_num, remQpInfo->qpn, localDevIndex, resCtx->baseComm);
  }
  return ncclSuccess;
}

ncclResult_t IbCastPortRecoveryQpsDestroy(struct ncclIbResiliency* resCtx, int nQps) {
  for (int qpIndex = 0; qpIndex < nQps; qpIndex++) {
    if (resCtx->portRecoveryAh[qpIndex]) {
      resCtx->portRecoveryAh[qpIndex]->context->ops.destroy_ah(resCtx->portRecoveryAh[qpIndex]);
      resCtx->portRecoveryAh[qpIndex] = nullptr;
    }
    struct ncclIbQp* recoveryQp = &resCtx->portRecoveryQps[qpIndex];
    if (!recoveryQp->qp) {
      continue;
    }
    INFO(NCCL_NET, "NET/IB: %s: Destroying port recovery UD QP (index=%d, qp_num=%u) for resiliency context (comm=%p)", __func__, qpIndex, recoveryQp->qp->qp_num, resCtx->baseComm);
    NCCLCHECK(wrap_ibv_destroy_qp(recoveryQp->qp));
  }
  return ncclSuccess;
}

static inline ncclResult_t IbCastPortRecoveryQpsRestore(ncclIbPortRecoveryContext* recoveryContext, bool* success) {
  ncclResult_t res = ncclSuccess;
  uint nqps = recoveryContext->resCtx->baseComm->nqps;
  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    ncclIbQp* localQp = &recoveryContext->resCtx->baseComm->qps[qpIndex];
    if (localQp->devIndex != recoveryContext->devIndex) {
      // The QP is not on the recovered device; skip
      continue;
    }

    INFO(NCCL_NET, "NET/IB: %s: Restoring QP %d on device %d (comm=%p, devIndex=%d, qp_num=%u)", __func__, qpIndex, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryContext->devIndex, localQp->qp->qp_num);
    res = IbCastQpReset(localQp);
    if (res != ncclSuccess) {
      INFO(NCCL_NET, "NET/IB: %s: Failed to reset QP index %d on device %d (comm=%p, devIndex=%d, qp_num=%u)", __func__, qpIndex, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryContext->devIndex, localQp->qp->qp_num);
      *success = false;
      return ncclSuccess;
    }
    res = IbCastQpInit(localQp);
    if (res != ncclSuccess) {
      INFO(NCCL_NET, "NET/IB: %s: Failed to init QP index %d on device %d (comm=%p, devIndex=%d, qp_num=%u)", __func__, qpIndex, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryContext->devIndex, localQp->qp->qp_num);
      *success = false;
      return ncclSuccess;
    }
    if (localQp->eceSupported) {
      res = wrap_ibv_set_ece(localQp->qp, &localQp->ece, &localQp->eceSupported);
      if (res != ncclSuccess) {
        INFO(NCCL_NET, "NET/IB: %s: Failed to set ECE for QP index %d on device %d (comm=%p, devIndex=%d, qp_num=%u)", __func__, qpIndex, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryContext->devIndex, localQp->qp->qp_num);
        *success = false;
        return ncclSuccess;
      }
    }
    res = IbCastQpRtr(localQp);
    if (res != ncclSuccess) {
      INFO(NCCL_NET, "NET/IB: %s: Failed to modify to RTR QP index %d on device %d (comm=%p, devIndex=%d, qp_num=%u)", __func__, qpIndex, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryContext->devIndex, localQp->qp->qp_num);
      *success = false;
      return ncclSuccess;
    }
    res = IbCastQpRts(localQp);
    if (res != ncclSuccess) {
      INFO(NCCL_NET, "NET/IB: %s: Failed to modify to RTS QP index %d on device %d (comm=%p, devIndex=%d, qp_num=%u)", __func__, qpIndex, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryContext->devIndex, localQp->qp->qp_num);
      *success = false;
      return ncclSuccess;
    }
    INFO(NCCL_NET, "NET/IB: %s: Restored QP %d on device %d (comm=%p, devIndex=%d, qp_num=%u)", __func__, qpIndex, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryContext->devIndex, localQp->qp->qp_num);

    if (!recoveryContext->resCtx->baseComm->isSend) {
      // Pre-post receive work requests on the restored QP if this is a receiver.
      INFO(NCCL_NET, "NET/IB: %s: Posting receive work requests on the restored QP (comm=%p, qp_num=%u)", __func__, recoveryContext->resCtx->baseComm, localQp->qp->qp_num);
      struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)recoveryContext->resCtx->baseComm;
      res = IbCastPostReceiveWorkRequestsOnQp(recvComm, localQp);
      if (res != ncclSuccess) {
        INFO(NCCL_NET, "NET/IB: %s: Failed to post recv WQEs on QP index %d on device %d (comm=%p, devIndex=%d, qp_num=%u)", __func__, qpIndex, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryContext->devIndex, localQp->qp->qp_num);
        *success = false;
        return ncclSuccess;
      }
    }
  }

  if (recoveryContext->resCtx->baseComm->isSend == false) {
    // Restore "Flush QP"
    ncclIbRecvComm* recvComm = (ncclIbRecvComm*)recoveryContext->resCtx->baseComm;
    if (recvComm->flushEnabled) {
      for (int i = 0; i < recvComm->base.vProps.ndevs; i++) {
        if (i != recoveryContext->devIndex) continue;
        struct ncclIbRecvCommDev* rCommDev = &recvComm->devs[i];
        ncclIbQp* flushQp = &rCommDev->gpuFlush.qp;
        TRACE(NCCL_NET, "NET/IB: %s: Restoring Flush QP on device %d (comm=%p)", __func__, i, recoveryContext->resCtx->baseComm);
        res = IbCastQpReset(flushQp);
        if (res != ncclSuccess) {
          INFO(NCCL_NET, "NET/IB: %s: Failed to reset Flush QP on device %d (comm=%p, devIndex=%d, qp_num=%u)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryContext->devIndex, flushQp->qp->qp_num);
          *success = false;
          return ncclSuccess;
        }
        res = IbCastQpInit(flushQp);
        if (res != ncclSuccess) {
          INFO(NCCL_NET, "NET/IB: %s: Failed to init Flush QP on device %d (comm=%p, devIndex=%d, qp_num=%u)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryContext->devIndex, flushQp->qp->qp_num);
          *success = false;
          return ncclSuccess;
        }
        res = IbCastQpRtr(flushQp);
        if (res != ncclSuccess) {
          INFO(NCCL_NET, "NET/IB: %s: Failed to modify to RTR Flush QP on device %d (comm=%p, devIndex=%d, qp_num=%u)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryContext->devIndex, flushQp->qp->qp_num);
          *success = false;
          return ncclSuccess;
        }
        res = IbCastQpRts(flushQp);
        if (res != ncclSuccess) {
          INFO(NCCL_NET, "NET/IB: %s: Failed to modify to RTS Flush QP on device %d (comm=%p, devIndex=%d, qp_num=%u)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryContext->devIndex, flushQp->qp->qp_num);
          *success = false;
          return ncclSuccess;
        }
        INFO(NCCL_NET, "NET/IB: %s: Restored Flush QP on device %d (comm=%p)", __func__, i, recoveryContext->resCtx->baseComm);
      }
    }
  }

  return ncclSuccess;
}

#define NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID (0x1234)

static inline ncclResult_t IbCastPortRecoveryHandleCompletionReceiver(struct ncclIbPortRecoveryContext* recoveryContext, struct ibv_wc completion, bool* success) {
  ncclResult_t res = ncclSuccess;
  INFO(NCCL_NET, "NET/IB: %s: Receiver received completion for device %d (comm=%p, wr_id=%lu, opcode=%s, imm_data=%u)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, completion.wr_id, ibvWcOpcodeStr(completion.opcode), completion.imm_data);
  if (recoveryContext->state == ncclIbPortRecoveryStateAliveMessages) {
    if (recoveryContext->ackPosted) {
      // Receiver waits for it's own local ACK completion
      if (completion.opcode == IBV_WC_RECV) {
        INFO(NCCL_NET, "NET/IB: %s: Receiver expected local completion of ACK message for device %d (comm=%p) but got completion with opcode: %s", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, ibvWcOpcodeStr(completion.opcode));
        res = IbCastPortRecoveryPostRecvWorkRequest(recoveryContext->resCtx->portRecoveryQps[recoveryContext->devIndex].qp, &recoveryContext->resCtx->devs[recoveryContext->devIndex]);
        if (res != ncclSuccess) {
          INFO(NCCL_NET, "NET/IB: %s: Receiver failed to post recv WR on data QP for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
          *success = false;
          return ncclSuccess;
        }
        // The sender might have transmitted additional alive messages before
        // receiving the ACK from the receiver, resulting in the receiver
        // getting another alive message while waiting for local ACK completion.
        *success = true;
        return ncclSuccess;
      }
      assert(completion.opcode == IBV_WC_SEND);
      recoveryContext->ackCompleted = true;
      INFO(NCCL_NET, "NET/IB: %s: Receiver's ACK message completed locally for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
    } else {
      assert(completion.opcode == IBV_WC_RECV);
      uint32_t id = completion.imm_data;
      int seqSize = ncclParamIbCastResiliencyPortRecoveryAliveMsgSequenceSize();
      uint32_t windowEnd = recoveryContext->recv.windowBase + seqSize;
      if (id >= recoveryContext->recv.windowBase && id < windowEnd) {
        recoveryContext->recv.receivedBits |= (1u << (id - recoveryContext->recv.windowBase));
        INFO(NCCL_NET, "NET/IB: %s: Receiver got alive message %u in window [%u,%u) for device %d (bits=0x%x, comm=%p)", __func__, id, recoveryContext->recv.windowBase, windowEnd, recoveryContext->devIndex, recoveryContext->recv.receivedBits, recoveryContext->resCtx->baseComm);
      } else if (id >= windowEnd) {
        uint32_t newBase = id - (id % seqSize);
        recoveryContext->recv.windowBase = newBase;
        recoveryContext->recv.receivedBits = (1u << (id - newBase));
        INFO(NCCL_NET, "NET/IB: %s: Receiver shifted window to [%u,%u) on alive message %u for device %d (comm=%p)", __func__, newBase, newBase + seqSize, id, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      } else {
        INFO(NCCL_NET, "NET/IB: %s: Receiver ignoring stale alive message %u (window base=%u) for device %d (comm=%p)", __func__, id, recoveryContext->recv.windowBase, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      }
      recoveryContext->timeLastMsg = clockNano();
      res = IbCastPortRecoveryPostRecvWorkRequest(recoveryContext->resCtx->portRecoveryQps[recoveryContext->devIndex].qp, &recoveryContext->resCtx->devs[recoveryContext->devIndex]);
      if (res != ncclSuccess) {
          INFO(NCCL_NET, "NET/IB: %s: Receiver failed to post recv WR on recovery QP for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
        *success = false;
        return ncclSuccess;
      }
    }
  }
  if (recoveryContext->state == ncclIbPortRecoveryStateAck) {
    assert(completion.opcode == IBV_WC_RECV);
    assert(completion.imm_data == NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID);
    INFO(NCCL_NET, "NET/IB: %s: Receiver received final ACK message for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
    recoveryContext->ackReceived = true;
  }
  *success = true;
  return ncclSuccess;
}

static inline ncclResult_t IbCastPortRecoveryHandleCompletionSender(struct ncclIbPortRecoveryContext* recoveryContext, struct ibv_wc completion, bool* success) {
  INFO(NCCL_NET, "NET/IB: %s: Sender received completion for device %d (comm=%p, wr_id=%lu, opcode=%s, imm_data=%u)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, completion.wr_id, ibvWcOpcodeStr(completion.opcode), completion.imm_data);
  if (recoveryContext->state == ncclIbPortRecoveryStateAliveMessages) {
    if (completion.opcode == IBV_WC_SEND) {
      assert(!recoveryContext->send.aliveMsgCompleted);
      INFO(NCCL_NET, "NET/IB: %s: Sender's alive messages batch completed locally for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      recoveryContext->send.aliveMsgCompleted = true;
    } else {
      INFO(NCCL_NET, "NET/IB: %s: Sender received unexpected message completion for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      *success = false;
      return ncclSuccess;
    }
  }
  if (recoveryContext->state == ncclIbPortRecoveryStateAck) {
    if (!recoveryContext->ackReceived) {
      // Sender waits for ACK from receiver
      assert(completion.opcode == IBV_WC_RECV);
      assert(completion.imm_data == NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID);
      INFO(NCCL_NET, "NET/IB: %s: Sender received an ACK message from the receiver for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      recoveryContext->ackReceived = true;
      *success = true;
      return ncclSuccess;
    } else {
      // Sender waits for it's own local final ACK completion
      assert(completion.opcode == IBV_WC_SEND);
      INFO(NCCL_NET, "NET/IB: %s: Sender's final ACK message completed locally for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      recoveryContext->ackCompleted = true;
      *success = true;
      return ncclSuccess;
    }
  }
  *success = true;
  return ncclSuccess;
}

static inline ncclResult_t IbCastPortRecoveryPollCq(struct ncclIbPortRecoveryContext* recoveryContext, bool* success) {
  int completions = 0;
  struct ibv_wc completion;

  ncclResult_t ret = wrap_ibv_poll_cq(recoveryContext->recoveryCq, 1, &completion, &completions);
  if (ret != ncclSuccess) {
    INFO(NCCL_NET, "NET/IB: %s: Failed to poll recovery CQ %p for device %d (comm=%p)", __func__, recoveryContext->recoveryCq, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
    *success = false;
    return ncclSuccess;
  }

  if (completions == 0) {
    *success = true;
    return ncclSuccess;
  }

  if (completion.status != IBV_WC_SUCCESS) {
    *success = false;
    INFO(NCCL_NET, "NET/IB: %s: Work completion error on recovery CQ %p: status=%s(%d)", __func__, recoveryContext->recoveryCq, ibvWcStatusStr(completion.status), completion.status);
    return ncclSuccess;
  }

  if (recoveryContext->resCtx->baseComm->isSend) {
    NCCLCHECK(IbCastPortRecoveryHandleCompletionSender(recoveryContext, completion, success));
  } else {
    NCCLCHECK(IbCastPortRecoveryHandleCompletionReceiver(recoveryContext, completion, success));
  }
  if (!*success) {
    INFO(NCCL_NET, "NET/IB: %s: Failed to handle %s completion for device %d (comm=%p, wc.opcode=%s(%d), wc.wr_id=%ld)", __func__, recoveryContext->resCtx->baseComm->isSend ? "sender" : "receiver", recoveryContext->devIndex, recoveryContext->resCtx->baseComm, ibvWcOpcodeStr(completion.opcode), completion.opcode, completion.wr_id);
  }
  return ncclSuccess;
}

enum ncclIbPortRecoveryStateProgressResult {
  ncclIbPortRecoveryStateProgressResultInProgress,
  ncclIbPortRecoveryStateProgressResultGoToNextState,
  ncclIbPortRecoveryStateProgressResultGoToPrevState,
  ncclIbPortRecoveryStateProgressResultFailed
};

static inline ncclResult_t IbCastPortRecoveryPostAliveMessages(struct ncclIbPortRecoveryContext* recoveryContext, bool* success) {
  struct ibv_send_wr *bad_wr = NULL;
  struct ibv_send_wr wr[NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX];
  int nMsgsToPost = ncclParamIbCastResiliencyPortRecoveryAliveMsgSequenceSize();
  if (nMsgsToPost > NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX) {
    WARN("NET/IB: %s: Requested alive message batch size %d exceeds maximum supported %d", __func__, nMsgsToPost, NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX);
    return ncclInternalError;
  }
  int devIdx = recoveryContext->devIndex;
  for (int i = 0; i < nMsgsToPost; i++) {
    memset(&wr[i], 0, sizeof(wr[i]));
    wr[i].opcode = IBV_WR_SEND_WITH_IMM;
    wr[i].send_flags = (i == nMsgsToPost - 1) ? IBV_SEND_SIGNALED : 0;
    wr[i].imm_data = recoveryContext->aliveMsgNextId;
    wr[i].sg_list = NULL;
    wr[i].num_sge = 0;
    wr[i].wr_id = recoveryContext->aliveMsgNextId;
    wr[i].wr.ud.ah = recoveryContext->resCtx->portRecoveryAh[devIdx];
    wr[i].wr.ud.remote_qpn = recoveryContext->resCtx->portRecoveryRemoteQpn[devIdx];
    wr[i].wr.ud.remote_qkey = NCCL_IB_RECOVERY_QKEY;
    if (i < nMsgsToPost - 1) {
      wr[i].next = &wr[i + 1];
    } else {
      wr[i].next = NULL;
    }
    INFO(NCCL_NET, "NET/IB: %s: Sender prepared alive message %u for device %d (comm=%p)", __func__, recoveryContext->aliveMsgNextId, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
    recoveryContext->aliveMsgNextId++;
  }

  // Post the send operation on the recovery QP.
  struct ncclIbQp* recoveryQp = &recoveryContext->resCtx->portRecoveryQps[recoveryContext->devIndex];
  if (wrap_ibv_post_send(recoveryQp->qp, &wr[0], &bad_wr) != ncclSuccess) {
    INFO(NCCL_NET, "NET/IB: %s: Sender failed to post alive messages batch on device %d (comm=%p, qp_num=%u)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryQp->qp->qp_num);
    *success = false;
    return ncclSuccess;
  }
  *success = true;

  recoveryContext->timeLastMsg = clockNano();
  recoveryContext->send.aliveMsgPosted = true;
  return ncclSuccess;
}

static inline ncclResult_t IbCastPortRecoveryPostAck(ncclIbPortRecoveryContext* recoveryContext, bool* success) {
  struct ibv_send_wr *bad_wr = NULL;
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  int devIdx = recoveryContext->devIndex;
  wr.wr_id = recoveryContext->aliveMsgNextId;
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.imm_data = NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.sg_list = NULL;
  wr.num_sge = 0;
  wr.wr.ud.ah = recoveryContext->resCtx->portRecoveryAh[devIdx];
  wr.wr.ud.remote_qpn = recoveryContext->resCtx->portRecoveryRemoteQpn[devIdx];
  wr.wr.ud.remote_qkey = NCCL_IB_RECOVERY_QKEY;

  INFO(NCCL_NET, "NET/IB: %s: %s posting ACK message for device %d (comm=%p)", __func__, recoveryContext->resCtx->baseComm->isSend ? "Sender" : "Receiver", recoveryContext->devIndex, recoveryContext->resCtx->baseComm);

  // Post the send operation on the recovery QP.
  struct ncclIbQp* recoveryQp = &recoveryContext->resCtx->portRecoveryQps[recoveryContext->devIndex];
  if (wrap_ibv_post_send(recoveryQp->qp, &wr, &bad_wr) != ncclSuccess) {
    INFO(NCCL_NET, "NET/IB: %s: Failed to post ack message on device %d (comm=%p, qp_num=%u)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryQp->qp->qp_num);
    *success = false;
    return ncclSuccess;
  }
  recoveryContext->ackPosted = true;
  *success = true;
  return ncclSuccess;
}

static inline ncclResult_t IbCastPortRecoveryProgressAliveMessagesSender(ncclIbPortRecoveryContext* recoveryContext, enum ncclIbPortRecoveryStateProgressResult* outResult) {
  // Sender can either post new alive messages or poll for completions of previously
  // posted alive messages.
  if (!recoveryContext->send.aliveMsgPosted) {
    // Check if sender should send a new batch of alive messages
    uint64_t now = clockNano();
    if (now - recoveryContext->timeLastMsg < ncclParamIbCastResiliencyPortRecoveryAliveMsgBatchInterval() * MSEC_TO_NSEC) {
      *outResult = ncclIbPortRecoveryStateProgressResultInProgress;
      return ncclSuccess;
    }
    // Post a new batch of alive messages
    INFO(NCCL_NET, "NET/IB: %s: Posting alive message batch for device %d (comm=%p, devIndex=%d)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm, recoveryContext->devIndex);
    bool success = false;
    NCCLCHECK(IbCastPortRecoveryPostAliveMessages(recoveryContext, &success));
    if (!success) {
      INFO(NCCL_NET, "NET/IB: %s: Failed to post alive messages batch for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      *outResult = ncclIbPortRecoveryStateProgressResultFailed;
      return ncclSuccess;
    }
    *outResult = ncclIbPortRecoveryStateProgressResultInProgress;
  } else {
    // Poll for completions of previously posted alive messages
    bool success = false;
    NCCLCHECK(IbCastPortRecoveryPollCq(recoveryContext, &success));
    if (!success) {
      INFO(NCCL_NET, "NET/IB: %s: Failed to poll CQ for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      *outResult = ncclIbPortRecoveryStateProgressResultFailed;
      return ncclSuccess;
    }
    if (recoveryContext->send.aliveMsgCompleted) {
      INFO(NCCL_NET, "NET/IB: %s: Sender marked that alive messages batch completed for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      *outResult = ncclIbPortRecoveryStateProgressResultGoToNextState;
    } else {
      *outResult = ncclIbPortRecoveryStateProgressResultInProgress;
    }
  }
  return ncclSuccess;
}

static inline ncclResult_t IbCastPortRecoveryProgressAliveMessagesReceiver(ncclIbPortRecoveryContext* recoveryContext, enum ncclIbPortRecoveryStateProgressResult* outResult) {
  bool success = false;
  ncclResult_t res = ncclSuccess;
  if (recoveryContext->ackPosted == false) {
    NCCLCHECK(IbCastPortRecoveryPollCq(recoveryContext, &success));
    if (!success) {
      INFO(NCCL_NET, "NET/IB: %s: Failed to poll CQ for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      *outResult = ncclIbPortRecoveryStateProgressResultFailed;
      return ncclSuccess;
    }
    uint32_t allSet = (1u << ncclParamIbCastResiliencyPortRecoveryAliveMsgSequenceSize()) - 1;
    if (recoveryContext->recv.receivedBits == allSet) {
      // Received enough in-order alive messages. Draining the CQ from any
      // remaining alive messages and proceeding to restore QPs and post ACK.
      // Draining is important to ensure that if the receiver goes back to the
      // previous state, it will not see stale alive messages.
      NCCLCHECK(IbCastPortRecoveryDrainCqAndPostReceiveWRs(recoveryContext, &success));
      if (!success) {
        INFO(NCCL_NET, "NET/IB: %s: Failed to drain CQ and post recv WRs for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
        *outResult = ncclIbPortRecoveryStateProgressResultFailed;
        return ncclSuccess;
      }
      INFO(NCCL_NET, "NET/IB: %s: Receiver received enough in-order alive messages for device %d (comm=%p). Restoring QPs and posting ACK message.", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      NCCLCHECK(IbCastPortRecoveryQpsRestore(recoveryContext, &success));
      if (!success) {
        INFO(NCCL_NET, "NET/IB: %s: Receiver failed to restore QPs on device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
        *outResult = ncclIbPortRecoveryStateProgressResultFailed;
        return ncclSuccess;
      }
      NCCLCHECK(IbCastPortRecoveryPostAck(recoveryContext, &success));
      if (!success) {
        INFO(NCCL_NET, "NET/IB: %s: Failed to post ACK message for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
        *outResult = ncclIbPortRecoveryStateProgressResultFailed;
        return ncclSuccess;
      }
      *outResult = ncclIbPortRecoveryStateProgressResultInProgress;
    } else {
      uint64_t now = clockNano();
      if (now - recoveryContext->timeLastMsg > ncclParamIbCastResiliencyPortRecoveryAliveMsgTimeout() * MSEC_TO_NSEC) {
        recoveryContext->nFailedAttempts++;
        INFO(NCCL_NET, "NET/IB: %s: Alive message sequence timeout for device %d (%s comm=%p, failedAttempts=%d)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm, recoveryContext->nFailedAttempts);
        if (recoveryContext->nFailedAttempts >= ncclParamIbCastResiliencyPortRecoveryAttemptsMax()) {
          INFO(NCCL_NET, "NET/IB: %s: Recovery for device %d failed due to max attempts reached (%s comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm);
          *outResult = ncclIbPortRecoveryStateProgressResultFailed;
          return ncclSuccess;
        }
        // "Reset" the timer till timeout
        recoveryContext->timeLastMsg = now;
      }
      *outResult = ncclIbPortRecoveryStateProgressResultInProgress;
    }
  } else {
    bool success = false;
    NCCLCHECK(IbCastPortRecoveryPollCq(recoveryContext, &success));
    if (!success) {
      INFO(NCCL_NET, "NET/IB: %s: Failed to poll CQ for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      *outResult = ncclIbPortRecoveryStateProgressResultFailed;
      return ncclSuccess;
    }
    if (recoveryContext->ackCompleted) {
      *outResult = ncclIbPortRecoveryStateProgressResultGoToNextState;
    } else {
      uint64_t now = clockNano();
      if (now - recoveryContext->timeLastMsg > ncclParamIbCastResiliencyPortRecoveryAckTimeout() * MSEC_TO_NSEC) {
        recoveryContext->nFailedAttempts++;
        INFO(NCCL_NET, "NET/IB: %s: ACK message timeout for device %d (%s comm=%p, failedAttempts=%d)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm, recoveryContext->nFailedAttempts);
        if (recoveryContext->nFailedAttempts >= ncclParamIbCastResiliencyPortRecoveryAttemptsMax()) {
          INFO(NCCL_NET, "NET/IB: %s: Recovery for device %d failed due to max attempts reached (%s comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm);
          *outResult = ncclIbPortRecoveryStateProgressResultFailed;
          return ncclSuccess;
        }
        // Go back to alive messages state
        recoveryContext->ackPosted = false;
        recoveryContext->ackCompleted = false;
        recoveryContext->timeLastMsg = now;
        *outResult = ncclIbPortRecoveryStateProgressResultGoToPrevState;
        INFO(NCCL_NET, "NET/IB: %s: Receiver posting (%d) recv WRs on device %d (comm=%p)", __func__, NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
        for (int i = 0; i < NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX; i++) {
          res = IbCastPortRecoveryPostRecvWorkRequest(recoveryContext->resCtx->portRecoveryQps[recoveryContext->devIndex].qp, &recoveryContext->resCtx->devs[recoveryContext->devIndex]);
          if (res != ncclSuccess) {
            INFO(NCCL_NET, "NET/IB: %s: Receiver failed to post recv WR on recovery QP for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
            *outResult = ncclIbPortRecoveryStateProgressResultFailed;
            return ncclSuccess;
          }
        }
      } else {
        *outResult = ncclIbPortRecoveryStateProgressResultInProgress;
      }
    }
  }
  return ncclSuccess;
}

static inline ncclResult_t IbCastPortRecoveryProgressAliveMessages(ncclIbPortRecoveryContext* recoveryContext) {
  enum ncclIbPortRecoveryStateProgressResult progressResult = ncclIbPortRecoveryStateProgressResultGoToNextState;
  if (recoveryContext->resCtx->baseComm->isSend) {
    NCCLCHECK(IbCastPortRecoveryProgressAliveMessagesSender(recoveryContext, &progressResult));
  } else {
    NCCLCHECK(IbCastPortRecoveryProgressAliveMessagesReceiver(recoveryContext, &progressResult));
  }
  switch (progressResult) {
    case ncclIbPortRecoveryStateProgressResultGoToPrevState:
      WARN("NET/IB: %s: Unexpected GoToPrevState result in AliveMessages state for device %d (%s comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm);
      return ncclInternalError;
    case ncclIbPortRecoveryStateProgressResultGoToNextState:
      INFO(NCCL_NET, "NET/IB: %s: Alive messages phase completed for device %d. Moving to next phase (%s comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm);
      recoveryContext->state = ncclIbPortRecoveryStateAck;
      break;
    case ncclIbPortRecoveryStateProgressResultFailed:
      INFO(NCCL_NET, "NET/IB: %s: Recovery for device %d failed (%s comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm);
      recoveryContext->state = ncclIbPortRecoveryStateFailed;
      break;
    case ncclIbPortRecoveryStateProgressResultInProgress:
      // Do nothing
      break;
  }
  return ncclSuccess;
}

static inline ncclResult_t IbCastPortRecoveryProgressAckSender(ncclIbPortRecoveryContext* recoveryContext, enum ncclIbPortRecoveryStateProgressResult* outResult) {
  bool success = false;
  if (!recoveryContext->ackReceived) {
    NCCLCHECK(IbCastPortRecoveryPollCq(recoveryContext, &success));
    if (!success) {
      INFO(NCCL_NET, "NET/IB: %s: Failed to poll CQ for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      *outResult = ncclIbPortRecoveryStateProgressResultFailed;
      return ncclSuccess;
    }
    if (!recoveryContext->ackReceived) {
      // Check if the timer timed out waiting for ack
      uint64_t now = clockNano();
      if (now - recoveryContext->timeLastMsg > ncclParamIbCastResiliencyPortRecoveryAckTimeout() * MSEC_TO_NSEC) {
        recoveryContext->nFailedAttempts++;
        INFO(NCCL_NET, "NET/IB: %s: Port recovery attempt #%d failed for devIndex=%d (comm=%p)", __func__, recoveryContext->nFailedAttempts, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
        if (recoveryContext->nFailedAttempts >= ncclParamIbCastResiliencyPortRecoveryAttemptsMax()) {
          INFO(NCCL_NET, "NET/IB: %s: Recovery for device %d failed due to max attempts reached (send comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
          *outResult = ncclIbPortRecoveryStateProgressResultFailed;
          return ncclSuccess;
        }
        *outResult = ncclIbPortRecoveryStateProgressResultGoToPrevState;
        // Reset internal state
        recoveryContext->send.aliveMsgPosted = false;
        recoveryContext->send.aliveMsgCompleted = false;
      } else {
        *outResult = ncclIbPortRecoveryStateProgressResultInProgress;
      }
      return ncclSuccess;
    }
  }

  if (recoveryContext->ackPosted == false) {
    NCCLCHECK(IbCastPortRecoveryQpsRestore(recoveryContext, &success));
    if (!success) {
      INFO(NCCL_NET, "NET/IB: %s: Sender failed to restore QPs on device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      *outResult = ncclIbPortRecoveryStateProgressResultFailed;
      return ncclSuccess;
    }
    NCCLCHECK(IbCastPortRecoveryPostAck(recoveryContext, &success));
    if (!success) {
      INFO(NCCL_NET, "NET/IB: %s: Failed to post ack for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      *outResult = ncclIbPortRecoveryStateProgressResultFailed;
      return ncclSuccess;
    }
    *outResult = ncclIbPortRecoveryStateProgressResultInProgress;
  }

  if (!recoveryContext->ackCompleted) {
    NCCLCHECK(IbCastPortRecoveryPollCq(recoveryContext, &success));
    if (!success) {
      INFO(NCCL_NET, "NET/IB: %s: Failed to poll CQ for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      *outResult = ncclIbPortRecoveryStateProgressResultFailed;
      return ncclSuccess;
    }
    if (recoveryContext->ackCompleted) {
      *outResult = ncclIbPortRecoveryStateProgressResultGoToNextState;
    } else {
      *outResult = ncclIbPortRecoveryStateProgressResultInProgress;
    }
  }
  return ncclSuccess;
}

static inline ncclResult_t IbCastPortRecoveryProgressAckReceiver(ncclIbPortRecoveryContext* recoveryContext, enum ncclIbPortRecoveryStateProgressResult* outResult) {
  bool success = false;
  NCCLCHECK(IbCastPortRecoveryPollCq(recoveryContext, &success));
  if (!success) {
    INFO(NCCL_NET, "NET/IB: %s: Failed to poll CQ for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
    *outResult = ncclIbPortRecoveryStateProgressResultFailed;
    return ncclSuccess;
  }
  if (!recoveryContext->ackReceived) {
    // Check if the timer timed out waiting for ack
    uint64_t now = clockNano();
    if (now - recoveryContext->timeLastMsg > ncclParamIbCastResiliencyPortRecoveryAckTimeout() * MSEC_TO_NSEC) {
      recoveryContext->nFailedAttempts++;
      INFO(NCCL_NET, "NET/IB: %s: Port recovery attempt #%d failed for devIndex=%d (comm=%p)", __func__, recoveryContext->nFailedAttempts, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
      if (recoveryContext->nFailedAttempts >= ncclParamIbCastResiliencyPortRecoveryAttemptsMax()) {
        *outResult = ncclIbPortRecoveryStateProgressResultFailed;
        return ncclSuccess;
      }
      *outResult = ncclIbPortRecoveryStateProgressResultGoToPrevState;
      // Reset internal state
      recoveryContext->ackPosted = false;
      recoveryContext->ackCompleted = false;
      recoveryContext->recv.receivedBits = 0;
      NCCLCHECK(IbCastPortRecoveryDrainCqAndPostReceiveWRs(recoveryContext, &success));
      if (!success) {
        INFO(NCCL_NET, "NET/IB: %s: Failed to drain CQ and post recv WRs for device %d (comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm);
        *outResult = ncclIbPortRecoveryStateProgressResultFailed;
        return ncclSuccess;
      }
    } else {
      *outResult = ncclIbPortRecoveryStateProgressResultInProgress;
    }
  } else {
    *outResult = ncclIbPortRecoveryStateProgressResultGoToNextState;
  }
  return ncclSuccess;
}

static inline ncclResult_t IbCastPortRecoveryProgressAck(ncclIbPortRecoveryContext* recoveryContext) {
  enum ncclIbPortRecoveryStateProgressResult progressResult = ncclIbPortRecoveryStateProgressResultFailed;
  if (recoveryContext->resCtx->baseComm->isSend) {
    NCCLCHECK(IbCastPortRecoveryProgressAckSender(recoveryContext, &progressResult));
  } else {
    NCCLCHECK(IbCastPortRecoveryProgressAckReceiver(recoveryContext, &progressResult));
  }
  switch (progressResult) {
    case ncclIbPortRecoveryStateProgressResultGoToPrevState:
      INFO(NCCL_NET, "NET/IB: %s: Restarting alive messages phase for device %d (%s comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm);
      recoveryContext->state = ncclIbPortRecoveryStateAliveMessages;
      break;
    case ncclIbPortRecoveryStateProgressResultGoToNextState:
      INFO(NCCL_NET, "NET/IB: %s: ACK phase completed for device %d (%s comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm);
      recoveryContext->state = ncclIbPortRecoveryStateSuccess;
      break;
    case ncclIbPortRecoveryStateProgressResultFailed:
      INFO(NCCL_NET, "NET/IB: %s: Recovery for device %d failed (%s comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm);
      recoveryContext->state = ncclIbPortRecoveryStateFailed;
      break;
    case ncclIbPortRecoveryStateProgressResultInProgress:
      // Do nothing
      break;
  }
  return ncclSuccess;
}

static inline ncclResult_t IbCastPortRecoveryContextProgress(ncclIbPortRecoveryContext* recoveryContext, bool* outDone) {
  assert(recoveryContext);
  assert(outDone);

  if (recoveryContext->state == ncclIbPortRecoveryStateInit) {
    uint64_t now = clockNano();
    if (now - recoveryContext->timeInit < ncclParamIbCastResiliencyPortRecoveryStartDelay() * MSEC_TO_NSEC) {
      *outDone = false;
      return ncclSuccess;
    }
    INFO(NCCL_NET, "NET/IB: %s: Starting port recovery for %s comm=%p devIndex=%d", __func__, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm, recoveryContext->devIndex);
    if (recoveryContext->resCtx->baseComm->isSend) {
      recoveryContext-> timeLastMsg = 0;
    }
    recoveryContext->state = ncclIbPortRecoveryStateAliveMessages;
  }

  if (recoveryContext->state == ncclIbPortRecoveryStateAliveMessages) {
    NCCLCHECK(IbCastPortRecoveryProgressAliveMessages(recoveryContext));
  }

  if (recoveryContext->state == ncclIbPortRecoveryStateAck) {
    NCCLCHECK(IbCastPortRecoveryProgressAck(recoveryContext));
  }

  if (recoveryContext->state == ncclIbPortRecoveryStateSuccess) {
    INFO(NCCL_NET, "NET/IB: %s: Port recovery succeeded for devIndex=%d (%s comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm);
    for (int i = 0; i < recoveryContext->resCtx->ndevs; i++) {
        if (i != recoveryContext->devIndex) continue;
        INFO(NCCL_NET, "NET/IB: %s: Marking device %d as recovered (%s comm=%p)", __func__, i, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm);
        recoveryContext->resCtx->devs[i].state.store(ncclIbResiliencyDevStateRecovered, std::memory_order_release);
        break;
      }
    *outDone = true;
  }

  if (recoveryContext->state == ncclIbPortRecoveryStateFailed) {
    INFO(NCCL_NET, "NET/IB: %s: Port recovery failed for %s comm=%p devIndex=%d", __func__, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm, recoveryContext->devIndex);
    for (int i = 0; i < recoveryContext->resCtx->ndevs; i++) {
      if (i != recoveryContext->devIndex) continue;
      INFO(NCCL_NET, "NET/IB: %s: Marking device %d as permanently failed (%s comm=%p)", __func__, i, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm);
      recoveryContext->resCtx->devs[i].state.store(ncclIbResiliencyDevStateRecoveryFailed, std::memory_order_release);
    }
    *outDone = true;
  }

  return ncclSuccess;
}

// Scan recoveryQueue and remove (destroy) any contexts belonging to the
// resiliency contexts referenced by the close requests.
static void IbCastPortRecoveryHandleCloseRequests(std::vector<ncclIbPortRecoveryCloseRequest*>& closeRequests,
                                                 std::list<ncclIbPortRecoveryContext*>& recoveryQueue) {
  for (auto it = closeRequests.begin(); it != closeRequests.end(); ++it) {
    ncclIbPortRecoveryCloseRequest* closeReq = *it;
    int removedForThisReq = 0;

    // Iterate through the recovery queue and remove items belonging to this resCtx
    for (auto qIt = recoveryQueue.begin(); qIt != recoveryQueue.end(); ) {
      ncclIbPortRecoveryContext* recoveryContext = *qIt;
      if (recoveryContext->resCtx == closeReq->resCtx) {
        INFO(NCCL_NET, "NET/IB: %s: Removing recovery context for device %d belonging to closing resCtx (%s comm=%p)",
             __func__, recoveryContext->devIndex,
             recoveryContext->resCtx->baseComm->isSend ? "send" : "recv",
             recoveryContext->resCtx->baseComm);
        IbCastPortRecoveryContextDestroy(recoveryContext);
        qIt = recoveryQueue.erase(qIt);
        removedForThisReq++;
      } else {
        ++qIt;
      }
    }

    INFO(NCCL_NET, "NET/IB: %s: Close request completed for resCtx=%p, removed %d items",
         __func__, closeReq->resCtx, removedForThisReq);
  }
}

// The asynchronous thread main function for port recovery processing.
// The thread waits for nodes to be added to the recovery queue and
// processes them to advance their recovery protocol state.
// The thread runs until the shutdown flag is set.
ncclResult_t IbCastPortRecoveryAsyncThreadMain() {
  // Thread-local recovery queue. Items are transferred from recoveryInbox
  // under the lock and then iterated here without holding the lock.
  std::list<ncclIbPortRecoveryContext*> recoveryQueue;

  while (true) {
    // Wait until queue is not empty, there are pending close requests, or
    // shutdown is requested (ThreadActive == false)
    std::vector<ncclIbPortRecoveryCloseRequest*> localCloseRequests;
    {
      std::unique_lock<std::mutex> lock(IbCastPortRecoveryMutex);
      IbCastPortRecoveryCond.wait(lock, [&] {
        return !IbCastPortRecoveryThreadActive.load() ||
               !recoveryInbox.empty() ||
               !recoveryQueue.empty() ||
               !recoveryCloseRequests.empty();
      });

      // Transfer new items from shared inbox to local queue (O(1) splice)
      recoveryQueue.splice(recoveryQueue.end(), recoveryInbox);

      // Move pending close requests to local vector
      localCloseRequests.swap(recoveryCloseRequests);
    }

    // Process close requests outside the lock (recoveryQueue is thread-local)
    if (!localCloseRequests.empty()) {
      IbCastPortRecoveryHandleCloseRequests(localCloseRequests, recoveryQueue);

      // Mark requests completed under lock to prevent lost-wakeup
      // on the close condition variable
      {
        std::lock_guard<std::mutex> lock(IbCastPortRecoveryMutex);
        for (auto* closeReq : localCloseRequests) {
          closeReq->completed = true;
        }
        IbCastPortRecoveryCloseCond.notify_all();
      }
    }

    // Check if the thread has been signaled to stop
    if (!IbCastPortRecoveryThreadActive.load()) break;

    // Iterate and advance recovery protocol on all nodes
    for (auto it = recoveryQueue.begin(); it != recoveryQueue.end(); ) {
      bool isDone = false;
      ncclIbPortRecoveryContext* recoveryContext = *it;
      NCCLCHECK(IbCastPortRecoveryContextProgress(recoveryContext, &isDone));
      if (isDone) {
        INFO(NCCL_NET, "NET/IB: %s: Port recovery context done for device %d (%s comm=%p)", __func__, recoveryContext->devIndex, recoveryContext->resCtx->baseComm->isSend ? "send" : "recv", recoveryContext->resCtx->baseComm);
        NCCLCHECK(IbCastPortRecoveryContextDestroy(recoveryContext));
        it = recoveryQueue.erase(it);
      } else {
        ++it;
      }
    }
  }

  // All close requests should have been processed before the thread stops
  // (CommClose must be called before Destroy for each resiliency context)
  assert(recoveryCloseRequests.empty());

  // All recovery queue items should have been removed by CommClose calls
  assert(recoveryInbox.empty());
  assert(recoveryQueue.empty());

  INFO(NCCL_NET, "NET/IB: %s: Port recovery async thread exiting", __func__);
  return ncclSuccess;
}

// -----------------------------
// Implementation of entry point functions
// -----------------------------

ncclResult_t IbCastPortRecoveryInit(struct ncclIbResiliency* resCtx) {
  resCtx->recoveryEnabled = IbCastPortRecoveryThreadActive.load();
  INFO(NCCL_NET, "NET/IB: %s: Port recovery %s (%s comm=%p)", __func__,
       resCtx->recoveryEnabled ? "initialized" : "disabled",
       resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm);
  return ncclSuccess;
}

ncclResult_t IbCastPortRecoveryClose(struct ncclIbResiliency* resCtx) {
  if (resCtx->recoveryEnabled == false) {
    return ncclSuccess;
  }
  if (!IbCastPortRecoveryThreadActive.load()) {
    INFO(NCCL_NET, "NET/IB: %s: Recovery thread already stopped, skipping close (%s comm=%p)", __func__, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm);
    return ncclSuccess;
  }
  INFO(NCCL_NET, "NET/IB: %s: Closing port recovery for resiliency context (%s comm=%p)", __func__, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm);

  // Create a close request to remove any potential queue items belonging to
  // this resCtx
  ncclIbPortRecoveryCloseRequest closeReq;
  closeReq.resCtx = resCtx;
  closeReq.completed = false;

  {
    std::lock_guard<std::mutex> lock(IbCastPortRecoveryMutex);
    recoveryCloseRequests.push_back(&closeReq);
  }

  // Wake up the async thread to process the close request
  IbCastPortRecoveryCond.notify_one();

  // Wait for the close request to be completed by the async thread
  {
    std::unique_lock<std::mutex> lock(IbCastPortRecoveryMutex);
    IbCastPortRecoveryCloseCond.wait(lock, [&] {
      return closeReq.completed;
    });
  }

  INFO(NCCL_NET, "NET/IB: %s: Port recovery closed (%s comm=%p)", __func__, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm);
  return ncclSuccess;
}

ncclResult_t IbCastPortRecoveryThreadStart() {
  if (ncclParamIbCastResiliencyPortRecovery() == 0) {
    return ncclSuccess;
  }

  IbCastPortRecoveryRefCount++;

  // Only start the thread once (first caller wins)
  bool expected = false;
  if (!IbCastPortRecoveryThreadActive.compare_exchange_strong(expected, true)) {
    return ncclSuccess;
  }

  INFO(NCCL_NET, "NET/IB: %s: Starting port recovery async thread", __func__);
  IbCastPortRecoveryAsyncThread = std::thread(IbCastPortRecoveryAsyncThreadMain);
  ncclSetThreadName(IbCastPortRecoveryAsyncThread.native_handle(), "NCCL IbResiliencyPortRecoveryAsync");

  return ncclSuccess;
}

ncclResult_t IbCastPortRecoveryThreadStop() {
  if (ncclParamIbCastResiliencyPortRecovery() == 0) {
    return ncclSuccess;
  }

  if (--IbCastPortRecoveryRefCount > 0) {
    return ncclSuccess;
  }

  INFO(NCCL_NET, "NET/IB: %s: Shutting down port recovery async thread", __func__);

  IbCastPortRecoveryThreadActive.store(false);
  IbCastPortRecoveryCond.notify_one();

  if (IbCastPortRecoveryAsyncThread.joinable()) {
    IbCastPortRecoveryAsyncThread.join();
  }

  INFO(NCCL_NET, "NET/IB: %s: Port recovery async thread stopped", __func__);
  return ncclSuccess;
}

ncclResult_t IbCastPortRecoveryHandleFailure(struct ncclIbResiliency* resCtx, int devIndex) {
  assert(resCtx != NULL);
  assert(resCtx->recoveryEnabled);
  ncclResult_t res = ncclSuccess;
  enum ncclIbResiliencyDevState devState = resCtx->devs[devIndex].state.load(std::memory_order_acquire);
  assert(devState == ncclIbResiliencyDevStateRecoveryInProgress);
  ncclIbPortRecoveryContext* recoveryCtx = NULL;

  res = IbCastPortRecoveryContextInit(resCtx, devIndex, &recoveryCtx);
  if (res != ncclSuccess) {
    INFO(NCCL_NET,"NET/IB: %s: Failed to initialize recovery context for device %d (comm=%p)", __func__, devIndex, resCtx->baseComm);
    return res;
  }

  // Hold lock while adding to inbox to ensure proper synchronization with
  // the recovery thread
  {
    std::lock_guard<std::mutex> lock(IbCastPortRecoveryMutex);
    recoveryInbox.push_back(recoveryCtx);
  }

  // Wake up the async recovery thread
  IbCastPortRecoveryCond.notify_one();

  INFO(NCCL_NET, "NET/IB: %s: Added device %d into the recovery queue (%s comm=%p, devIndex=%d, isSend=%d)", __func__, devIndex, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm, devIndex, resCtx->baseComm->isSend);
  return ncclSuccess;
}
