/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "capability_cast.h"
#include "connect_cast.h"
#include "p2p_resiliency_recovery_cast.h"

extern int64_t ncclParamIbCastResiliencyPortFailover();
extern int64_t ncclParamIbCastResiliencyPortRecovery();
extern int64_t ncclParamIbCastPkey();
extern int64_t ncclParamIbCastTimeout();
extern int64_t ncclParamIbCastRetryCnt();
extern int64_t ncclParamIbCastSl();
extern int64_t ncclParamIbCastTc();

// A working READ completes in microseconds. Some NICs never complete it.
#define NCCL_IB_CAP_RDMA_READ_PROBE_TIMEOUT_NS (100ULL * 1000 * 1000)

// UD has no capability bit, so build a UD QP the way port recovery does: same
// queue sizes and the same INIT (with QKEY), RTR and RTS transitions.
static void IbCastCapProbeUd(struct ncclIbDev* dev) {
  bool supported = false;
  struct ibv_pd* pd = NULL;
  struct ibv_cq* cq = NULL;
  struct ncclIbQp probeQp;
  memset(&probeQp, 0, sizeof(probeQp));

  if (wrap_ibv_alloc_pd(&pd, dev->context) != ncclSuccess || pd == NULL) goto cleanup;
  if (wrap_ibv_create_cq(&cq, dev->context, NCCL_IB_RESILIENCY_PORT_RECOVERY_CQ_SIZE, NULL, NULL, 0) != ncclSuccess ||
      cq == NULL)
    goto cleanup;
  {
    struct ncclIbQpCreateAttr createAttr;
    memset(&createAttr, 0, sizeof(createAttr));
    IbCastQpCreateAttrInitSharing(&createAttr);
    createAttr.type = IBV_QPT_UD;
    createAttr.cq = cq;
    createAttr.pd = pd;
    createAttr.maxSendWorkRequest = NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX;
    createAttr.maxRecvWorkRequest = NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX;
    createAttr.qpContext = &dev->stats;
    createAttr.ctsQpSlot = NCCL_CTS_QP_SLOT_INVALID;
    if (IbCastQpCreate(&probeQp, &createAttr) != ncclSuccess || probeQp.qp == NULL) goto cleanup;
  }
  if (IbCastPortRecoveryQpUdToRts(&probeQp, dev->portNum) == ncclSuccess) supported = true;

cleanup:
  if (probeQp.qp) (void)wrap_ibv_destroy_qp(probeQp.qp);
  if (cq) (void)wrap_ibv_destroy_cq(cq);
  if (pd) (void)wrap_ibv_dealloc_pd(pd);
  dev->udSupported = supported ? 1 : 0;
  INFO(NCCL_NET, "NET/IB-CAST: device %s capability probe: UD=%s", dev->devName, supported ? "yes" : "no");
}

// Devices can advertise max_qp_rd_atom and still fail every READ, so issue one:
// a loopback RC QP, set up like the GDR flush QP, reads host memory from itself.
// Host memory only, the probe must not start peer-to-peer DMA.
static void IbCastCapProbeRdmaRead(struct ncclIbDev* dev) {
  const uint64_t pattern = 0x5ca1ab1e0ddba115ULL;
  bool supported = false;
  struct ibv_pd* pd = NULL;
  struct ibv_cq* cq = NULL;
  struct ibv_mr* mr = NULL;
  struct ncclIbQp probeQp;
  memset(&probeQp, 0, sizeof(probeQp));
  // [0] is the READ source, [1] the destination.
  volatile uint64_t buf[2] = {pattern, 0};
  int gidIndex = 0;
  union ibv_gid gid;
  memset(&gid, 0, sizeof(gid));

  if (wrap_ibv_alloc_pd(&pd, dev->context) != ncclSuccess || pd == NULL) goto cleanup;
  if (wrap_ibv_create_cq(&cq, dev->context, 1, NULL, NULL, 0) != ncclSuccess || cq == NULL) goto cleanup;
  if (wrap_ibv_reg_mr(&mr, pd, (void*)buf, sizeof(buf), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ) !=
        ncclSuccess ||
      mr == NULL)
    goto cleanup;
  {
    struct ncclIbQpCreateAttr createAttr;
    memset(&createAttr, 0, sizeof(createAttr));
    IbCastQpCreateAttrInitSharing(&createAttr);
    createAttr.type = IBV_QPT_RC;
    createAttr.cq = cq;
    createAttr.pd = pd;
    createAttr.maxSendWorkRequest = 1;
    createAttr.maxRecvWorkRequest = 1;
    createAttr.qpContext = &dev->stats;
    createAttr.ctsQpSlot = NCCL_CTS_QP_SLOT_INVALID;
    if (IbCastQpCreate(&probeQp, &createAttr) != ncclSuccess || probeQp.qp == NULL) goto cleanup;
  }
  if (IbCastGetGidIndex(dev->context, dev->portNum, &dev->portAttr, &gidIndex) != ncclSuccess) goto cleanup;
  if (wrap_ibv_query_gid(dev->context, dev->portNum, gidIndex, &gid) != ncclSuccess) goto cleanup;
  {
    struct ncclIbQpInitAttr* initAttr = &probeQp.initAttr;
    initAttr->state = IBV_QPS_INIT;
    initAttr->pkeyIndex = ncclParamIbCastPkey();
    initAttr->portNum = dev->portNum;
    initAttr->qpAccessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;
    if (IbCastQpInit(&probeQp) != ncclSuccess) goto cleanup;

    struct ncclIbQpRtrAttr* rtrAttr = &probeQp.rtrAttr;
    rtrAttr->mtu = dev->portAttr.active_mtu;
    rtrAttr->linkLayer = dev->portAttr.link_layer;
    rtrAttr->tc = ncclParamIbCastTc() != -1 ? ncclParamIbCastTc() : 0;
    rtrAttr->sl = ncclParamIbCastSl() != -1 ? ncclParamIbCastSl() : 0;
    rtrAttr->remoteQpNum = probeQp.qp->qp_num;
    rtrAttr->remoteLid = dev->portAttr.lid;
    rtrAttr->remoteGid = gid;
    rtrAttr->localIbPort = dev->portNum;
    rtrAttr->localPortFlags = dev->portAttr.flags;
    rtrAttr->localGid = gid;
    rtrAttr->localGidIndex = gidIndex;
    if (IbCastQpRtr(&probeQp) != ncclSuccess) goto cleanup;

    struct ncclIbQpRtsAttr* rtsAttr = &probeQp.rtsAttr;
    rtsAttr->timeout = ncclParamIbCastTimeout();
    rtsAttr->retryCnt = ncclParamIbCastRetryCnt();
    if (IbCastQpRts(&probeQp) != ncclSuccess) goto cleanup;
  }
  {
    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uint64_t)&buf[1];
    sge.length = sizeof(buf[1]);
    sge.lkey = mr->lkey;

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = (uint64_t)&buf[0];
    wr.wr.rdma.rkey = mr->rkey;
    struct ibv_send_wr* badWr = NULL;
    if (wrap_ibv_post_send(probeQp.qp, &wr, &badWr) != ncclSuccess) goto cleanup;

    uint64_t start = clockNano();
    while (clockNano() - start < NCCL_IB_CAP_RDMA_READ_PROBE_TIMEOUT_NS) {
      struct ibv_wc wc;
      int done = 0;
      if (wrap_ibv_poll_cq(cq, 1, &wc, &done) != ncclSuccess) break;
      if (done == 0) continue;
      supported = (wc.status == IBV_WC_SUCCESS && buf[1] == pattern);
      if (!supported) {
        INFO(NCCL_NET, "NET/IB-CAST: device %s RDMA READ probe completed with status %s (%d) vendor_err %u",
             dev->devName, ibvWcStatusStr(wc.status), wc.status, wc.vendor_err);
      }
      break;
    }
  }

cleanup:
  // Destroying the QP stops a READ that never completed before the buffer goes away.
  if (probeQp.qp) (void)wrap_ibv_destroy_qp(probeQp.qp);
  if (mr) (void)wrap_ibv_dereg_mr(mr);
  if (cq) (void)wrap_ibv_destroy_cq(cq);
  if (pd) (void)wrap_ibv_dealloc_pd(pd);
  dev->rdmaReadSupported = supported ? 1 : 0;
  INFO(NCCL_NET, "NET/IB-CAST: device %s capability probe: RDMA_READ=%s", dev->devName, supported ? "yes" : "no");
}

ncclResult_t IbCastCapProbeDevices() {
  for (int d = 0; d < IbCastNDevs; d++) {
    struct ncclIbDev* dev = &IbCastDevs[d];
    std::lock_guard<std::mutex> lock(dev->mutex);
    if (ncclParamIbCastResiliencyPortRecovery() && dev->udSupported < 0) {
      IbCastCapProbeUd(dev);
      if (dev->udSupported == 0) {
        WARN("NET/IB-CAST: device %s cannot create a UD queue pair; port recovery disabled on it", dev->devName);
      }
    }
    if (ncclParamIbCastResiliencyPortFailover() && dev->rdmaReadSupported < 0) {
      IbCastCapProbeRdmaRead(dev);
      if (dev->rdmaReadSupported == 0) {
        WARN("NET/IB-CAST: device %s cannot complete an RDMA READ; port failover probing will fail on it",
             dev->devName);
      }
    }
  }
  return ncclSuccess;
}

bool IbCastCapUdSupported(const struct ncclIbDev* dev) {
  return dev->udSupported == 1;
}

bool IbCastCapRdmaReadSupported(const struct ncclIbDev* dev) {
  return dev->rdmaReadSupported == 1;
}

extern "C" ncclResult_t ncclIbCastGetDeviceCaps(int dev, struct ncclIbCastDeviceCaps* out) {
  if (out == NULL) return ncclInvalidArgument;
  if (dev < 0 || dev >= IbCastNMergedDevs) return ncclInvalidArgument;
  // A merged device has a feature only if every member has it.
  const ncclNetVDeviceProps_t* vProps = &IbCastMergedDevs[dev].vProps;
  out->udSupported = true;
  out->rdmaReadSupported = true;
  for (int i = 0; i < vProps->ndevs; i++) {
    int physDev = vProps->devs[i];
    if (physDev < 0 || physDev >= IbCastNDevs) return ncclInvalidArgument;
    out->udSupported = out->udSupported && IbCastCapUdSupported(&IbCastDevs[physDev]);
    out->rdmaReadSupported = out->rdmaReadSupported && IbCastCapRdmaReadSupported(&IbCastDevs[physDev]);
  }
  return ncclSuccess;
}
