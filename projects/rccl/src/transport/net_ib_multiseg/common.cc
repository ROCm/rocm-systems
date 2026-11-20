/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common_mseg.h"
#include "p2p_resiliency_mseg.h"

char IbMsegIfName[MAX_IF_NAME_SIZE + 1];
union ncclSocketAddress IbMsegIfAddr;

int IbMsegNMergedDevs = -1;
int IbMsegNDevs = -1;
struct IbMsegMergedDev IbMsegMergedDevs[MAX_IB_VDEVS];
struct IbMsegDev IbMsegDevs[MAX_IB_DEVS];
int IbMsegRelaxedOrderingEnabled = 0;
uint64_t IbMsegSpeedChangeCounter = 0;

ncclProfilerCallback_t IbMsegProfilerFunction;

extern int64_t ncclParamIbSplitDataOnQps();
extern int64_t ncclParamIbPrepostReceiveWorkRequests();
extern int64_t ncclParamIbAsyncEvents();
extern int64_t ncclParamIbEventBasedLb();
extern int64_t ncclParamIbEventBasedLbRemote();

extern int ncclParamIbReceiverSideMatchingScheme();
extern int ncclParamIbOooRq();
extern int ncclParamIbResiliencyPortFailover();

ncclResult_t IbMsegStatsCheckFatalCount(struct IbMsegStats* stat, const char* funcName) {
  if (ncclParamIbAsyncEvents() && COMPILER_ATOMIC_LOAD(&stat->fatalErrorCount, std::memory_order_relaxed)) {
    ERROR("RCCL encountered a communication fatal error (detected in %s)\n", funcName);
    ERROR("RCCL cannot recover from this network failure and now exiting. Please check the network health.");
    return ncclSystemError;
  }
  return ncclSuccess;
}

struct IbMsegNetCommDevBase* IbMsegGetNetCommDevBase(IbMsegNetCommBase* base, int devIndex) {
  if (base->isSend) {
    struct IbMsegSendComm* sComm = (struct IbMsegSendComm*)base;
    return &sComm->devs[devIndex].base;
  } else {
    struct IbMsegRecvComm* rComm = (struct IbMsegRecvComm*)base;
    return &rComm->devs[devIndex].base;
  }
}

ncclResult_t IbMsegBaseCommInit(struct IbMsegNetCommBase* baseComm, bool isSend) {
  for (int i = 0; i < NCCL_IB_MAX_QPS; i++) {
    baseComm->qps[i].devIndex = -1;
    baseComm->qps[i].remDevIdx = -1;
    baseComm->activeQps[i] = &baseComm->qps[i];
    baseComm->qps[i].eceSupported = 0;
    baseComm->qps[i].ece = {0};
    memset(&baseComm->qps[i].initAttr, 0, sizeof(baseComm->qps[i].initAttr));
    memset(&baseComm->qps[i].rtrAttr, 0, sizeof(baseComm->qps[i].rtrAttr));
    memset(&baseComm->qps[i].rtsAttr, 0, sizeof(baseComm->qps[i].rtsAttr));
  }
  baseComm->nqps = -1;
  baseComm->splitDataOnQps = ncclParamIbSplitDataOnQps();
  baseComm->nDataQps = -1;
  baseComm->isSend = isSend;
  baseComm->ready = 0;
  baseComm->speedChangeCounter = 0;
  baseComm->totalSpeed = 0;
  memset(baseComm->weights, 0, sizeof(baseComm->weights));
  memset(baseComm->devSpeeds, 0, sizeof(baseComm->devSpeeds));

  NCCLCHECK(IbMsegResiliencyInit(baseComm, &baseComm->resiliency));
  baseComm->recvMatchingScheme =
    ncclParamIbReceiverSideMatchingScheme() == -2 ? BY_INDEX : ncclParamIbReceiverSideMatchingScheme();

  if (ncclParamIbOooRq() || (ncclParamIbResiliencyPortFailover() == 1)) {
    baseComm->recvMatchingScheme = BY_ID;
    if (ncclParamIbReceiverSideMatchingScheme() == BY_INDEX) {
      INFO(NCCL_NET, "NET/IB: %s: Overriding matching scheme to ID-based (%d)", __func__, BY_ID);
    }
  }

  return ncclSuccess;
}

ncclResult_t IbMsegRecvCommInit(struct IbMsegRecvComm* recvComm) {
  // Set before anything can fail: the accept fail path closes this fd.
  recvComm->flagFlushDmabufFd = -1;
  NCCLCHECK(IbMsegBaseCommInit(&recvComm->base, false));
  recvComm->ibRecvWorkRequest = {.wr_id = NCCL_IB_RECV_WR_ID_DUMMY, .next = NULL, .sg_list = NULL, .num_sge = 0};

  recvComm->prepostReceiveWorkRequests =
    (ncclParamIbPrepostReceiveWorkRequests() == -2) ? false : ncclParamIbPrepostReceiveWorkRequests();

  if (recvComm->base.resiliency) {
    if (ncclParamIbPrepostReceiveWorkRequests() == 0) {
      INFO(NCCL_NET, "NET/IB: %s: Overriding pre-posting to true (1).", __func__);
    }
    recvComm->prepostReceiveWorkRequests = true;
  }
  if (ncclParamIbOooRq()) {
    if (ncclParamIbPrepostReceiveWorkRequests() == 0) {
      INFO(NCCL_NET, "NET/IB: %s: OOO RQ is enabled, Overriding pre-posting to true (1).", __func__);
    }
    recvComm->prepostReceiveWorkRequests = true;
  }

  INFO(NCCL_NET, "NET/IB: %s: Receive work requests will be %s", __func__,
       recvComm->prepostReceiveWorkRequests ? "pre-posted" : "posted on-demand");
  return ncclSuccess;
}

ncclResult_t IbMsegSendCommInit(struct IbMsegSendComm* sendComm) {
  NCCLCHECK(IbMsegBaseCommInit(&sendComm->base, true));
  return ncclSuccess;
}

static ncclResult_t IbMsegEventGidChange(struct IbMsegDev* dev) {
  INFO(NCCL_NET, "NET/IB: %s: GID table changed on %s:%d", __func__, dev->devName, dev->portNum);
  if (dev->gidInfo.link_layer != IBV_LINK_LAYER_ETHERNET) {
    INFO(NCCL_NET, "NET/IB : %s:%d link is not Ethernet; ignoring GID change", dev->devName, dev->portNum);
    return ncclSuccess;
  }
  struct ibv_port_attr portAttr;
  ncclResult_t res = wrap_ibv_query_port(dev->context, dev->portNum, &portAttr);
  if (res != ncclSuccess) {
    WARN("NET/IB : %s:%d query_port failed during GID change (res=%d)", dev->devName, dev->portNum, (int)res);
    return res;
  }

  std::lock_guard<std::mutex> lock(dev->mutex);
  int oldIdx = dev->gidInfo.localGidIndex;
  union ibv_gid oldGid = dev->gidInfo.localGid;
  res = IbMsegGidInfoQuery(dev->context, dev->portNum, &portAttr, &dev->gidInfo);
  if (res != ncclSuccess) {
    WARN("NET/IB : %s:%d GID info query failed (%d) during GID change", dev->devName, dev->portNum, (int)res);
    return res;
  }
  dev->portAttr = portAttr; // publish only once everything succeeded
  char oldGidStr[INET6_ADDRSTRLEN] = "";
  char newGidStr[INET6_ADDRSTRLEN] = "";
  ibvGetGidStr(&oldGid, oldGidStr, sizeof(oldGidStr));
  ibvGetGidStr(&dev->gidInfo.localGid, newGidStr, sizeof(newGidStr));
  INFO(NCCL_NET, "NET/IB : %s:%d GID refreshed: idx %d -> %d, gid %s -> %s", dev->devName, dev->portNum, oldIdx,
       dev->gidInfo.localGidIndex, oldGidStr, newGidStr);
  return ncclSuccess;
}

static void IbMsegUpdateDeviceSpeed(struct IbMsegDev* dev) {
  uint64_t oldSpeed = COMPILER_ATOMIC_LOAD(&dev->currSpeed, std::memory_order_relaxed);
  uint64_t newSpeed = 0;
  if (wrap_ibv_query_port_speed(dev->context, dev->portNum, &newSpeed) != ncclSuccess) return;

  // In cases of port failover, the speed change event is ignored and will be considered as a port fail event
  if (newSpeed == 0) return;
  // ibv_query_port_speed returns speed in granularity of 100 Mbps
  newSpeed *= 100;

  if (newSpeed != oldSpeed) {
    INFO(NCCL_NET, "NET/IB : %s:%d speed change detected: %lu -> %lu Mbps", dev->devName, dev->portNum,
         (unsigned long)oldSpeed, (unsigned long)newSpeed);
    COMPILER_ATOMIC_STORE(&dev->currSpeed, newSpeed, std::memory_order_relaxed);
    COMPILER_ATOMIC_FETCH_ADD(&IbMsegSpeedChangeCounter, (uint64_t)1, std::memory_order_release);
  }
}

std::thread IbMsegAsyncThread;
void* IbMsegAsyncThreadMain(void* args) {
  struct IbMsegDev* dev = (struct IbMsegDev*)args;
  while (1) {
    struct ibv_async_event event;
    if (ncclSuccess != wrap_ibv_get_async_event(dev->context, &event)) {
      break;
    }
    char* str;
    struct ibv_cq* cq = event.element.cq;    // only valid if CQ error
    struct ibv_qp* qp = event.element.qp;    // only valid if QP error
    struct ibv_srq* srq = event.element.srq; // only valid if SRQ error
    if (ncclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) {
      break;
    }
    switch (event.event_type) {
    case IBV_EVENT_DEVICE_FATAL:
      // the above is device fatal error
      WARN("NET/IB : %s:%d async fatal event: %s", dev->devName, dev->portNum, str);
      IbMsegDevFatalError(dev);
      break;
    case IBV_EVENT_CQ_ERR:
      WARN("NET/IB : %s:%d async fatal event on CQ (%p) handle=%u cqe=%d: %s", dev->devName, dev->portNum, cq,
           (unsigned)cq->handle, cq->cqe, str);
      IbMsegCqFatalError(cq);
      break;
    case IBV_EVENT_QP_FATAL:
    case IBV_EVENT_QP_REQ_ERR:
    case IBV_EVENT_QP_ACCESS_ERR:
      {
        struct ibv_qp_attr qpAttr;
        memset(&qpAttr, 0, sizeof(qpAttr));
        struct ibv_qp_init_attr qpInitAttr;
        memset(&qpInitAttr, 0, sizeof(qpInitAttr));
        (void)wrap_ibv_query_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_RQ_PSN, &qpInitAttr);
        WARN("NET/IB : %s:%d async fatal event on QP (%p) qpn=%u handle=%u send_cq=%u recv_cq=%u state=%d sq_psn=%u "
             "rq_psn=%u: %s",
             dev->devName, dev->portNum, qp, qp->qp_num, (unsigned)qp->handle,
             qp->send_cq ? (unsigned)qp->send_cq->handle : 0u, qp->recv_cq ? (unsigned)qp->recv_cq->handle : 0u,
             (int)qpAttr.qp_state, (unsigned)qpAttr.sq_psn, (unsigned)qpAttr.rq_psn, str);
        IbMsegQpFatalError(qp);
        break;
      }
    case IBV_EVENT_SRQ_ERR:
      // SRQ are not used in NCCL
      WARN("NET/IB : %s:%d async fatal event on SRQ, unused for now (%p): %s", dev->devName, dev->portNum, srq, str);
      break;
    case IBV_EVENT_GID_CHANGE:
      if (IbMsegEventGidChange(dev) != ncclSuccess) {
        WARN("NET/IB : %s:%d marking device with fatal error after GID-change event handler failed", dev->devName,
             dev->portNum);
        IbMsegDevFatalError(dev);
      }
      break;
    case IBV_EVENT_DEVICE_SPEED_CHANGE:
      IbMsegUpdateDeviceSpeed(dev);
      break;
    case IBV_EVENT_PATH_MIG_ERR:
    case IBV_EVENT_PORT_ERR:
    case IBV_EVENT_PATH_MIG:
    case IBV_EVENT_PORT_ACTIVE:
    case IBV_EVENT_SQ_DRAINED:
    case IBV_EVENT_LID_CHANGE:
    case IBV_EVENT_PKEY_CHANGE:
    case IBV_EVENT_SM_CHANGE:
    case IBV_EVENT_QP_LAST_WQE_REACHED:
    case IBV_EVENT_CLIENT_REREGISTER:
    case IBV_EVENT_SRQ_LIMIT_REACHED:
      // the above are non-fatal
      WARN("NET/IB : %s:%d Got non-fatal async event: %s(%d)", dev->devName, dev->portNum, str, event.event_type);
      break;
    case IBV_EVENT_COMM_EST:
      break;
    default:
      WARN("NET/IB : %s:%d unknown event type (%d)", dev->devName, dev->portNum, event.event_type);
      break;
    }
    // acknowledgment needs to happen last to avoid user-after-free
    if (ncclSuccess != wrap_ibv_ack_async_event(&event)) {
      break;
    }
  }
  return NULL;
}

ncclNet_t ncclNetIbMseg = {
  "IB-MSEG",
  IbMsegInit,
  IbMsegDevices,
  IbMsegGetProperties,
  IbMsegNetListen,
  IbMsegConnect,
  IbMsegAccept,
  IbMsegRegMr,
  IbMsegRegMrDmaBuf,
  IbMsegDeregMr,
  IbMsegIsend,
  IbMsegIrecv,
  IbMsegIflush,
  IbMsegTest,
  IbMsegCloseSend,
  IbMsegCloseRecv,
  IbMsegCloseListen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */,
  IbMsegMakeVDevice,
  IbMsegFinalize,
  IbMsegSetNetAttr,
};

#ifdef ENABLE_FAULT_INJECTION
#include "net_ib_gid_inspect_mseg.h"

extern "C" ncclResult_t IbMsegGidChangeEvent(int ibDev) {
  if (ibDev < 0 || ibDev >= IbMsegNDevs) return ncclInvalidArgument;
  return IbMsegEventGidChange(&IbMsegDevs[ibDev]);
}
#endif /* ENABLE_FAULT_INJECTION */
