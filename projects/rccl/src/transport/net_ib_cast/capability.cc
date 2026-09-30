/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "capability_cast.h"
#include "connect_cast.h"
#include "p2p_resiliency_recovery_cast.h"

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

ncclResult_t IbCastCapProbeDevices() {
  for (int d = 0; d < IbCastNDevs; d++) {
    struct ncclIbDev* dev = &IbCastDevs[d];
    std::lock_guard<std::mutex> lock(dev->mutex);
    if (dev->udSupported >= 0) continue;
    IbCastCapProbeUd(dev);
    if (dev->udSupported == 0) {
      WARN("NET/IB-CAST: device %s cannot create a UD queue pair; port recovery disabled on it", dev->devName);
    }
  }
  return ncclSuccess;
}

bool IbCastCapUdSupported(const struct ncclIbDev* dev) {
  return dev->udSupported == 1;
}

extern "C" ncclResult_t ncclIbCastGetDeviceCaps(int dev, struct ncclIbCastDeviceCaps* out) {
  if (out == NULL) return ncclInvalidArgument;
  if (dev < 0 || dev >= IbCastNMergedDevs) return ncclInvalidArgument;
  // Port recovery needs UD on every member of a merged device.
  const ncclNetVDeviceProps_t* vProps = &IbCastMergedDevs[dev].vProps;
  out->udSupported = true;
  for (int i = 0; i < vProps->ndevs; i++) {
    int physDev = vProps->devs[i];
    if (physDev < 0 || physDev >= IbCastNDevs) return ncclInvalidArgument;
    out->udSupported = out->udSupported && IbCastCapUdSupported(&IbCastDevs[physDev]);
  }
  return ncclSuccess;
}
