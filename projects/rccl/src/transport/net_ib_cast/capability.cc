/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "capability_cast.h"
#include "connect_cast.h"

// UD has no capability bit, so try to create a UD QP. A failure means "no".
static bool IbCastCapProbeUd(struct ncclIbDev* dev) {
  bool supported = false;
  struct ibv_pd* pd = NULL;
  struct ibv_cq* cq = NULL;
  struct ncclIbQp probeQp;
  memset(&probeQp, 0, sizeof(probeQp));

  if (wrap_ibv_alloc_pd(&pd, dev->context) != ncclSuccess || pd == NULL) goto cleanup;
  if (wrap_ibv_create_cq(&cq, dev->context, 1, NULL, NULL, 0) != ncclSuccess || cq == NULL) goto cleanup;
  {
    struct ncclIbQpCreateAttr createAttr;
    memset(&createAttr, 0, sizeof(createAttr));
    IbCastQpCreateAttrInitSharing(&createAttr);
    createAttr.type = IBV_QPT_UD;
    createAttr.cq = cq;
    createAttr.pd = pd;
    createAttr.maxSendWorkRequest = 1;
    createAttr.maxRecvWorkRequest = 1;
    createAttr.qpContext = &dev->stats;
    createAttr.ctsQpSlot = NCCL_CTS_QP_SLOT_INVALID;
    if (IbCastQpCreate(&probeQp, &createAttr) == ncclSuccess && probeQp.qp != NULL) supported = true;
  }

cleanup:
  if (probeQp.qp) (void)wrap_ibv_destroy_qp(probeQp.qp);
  if (cq) (void)wrap_ibv_destroy_cq(cq);
  if (pd) (void)wrap_ibv_dealloc_pd(pd);
  return supported;
}

// Returns true if this call did the probe.
static bool IbCastCapProbeLocked(struct ncclIbDev* dev) {
  if (dev->udSupported >= 0) return false;
  dev->udSupported = IbCastCapProbeUd(dev) ? 1 : 0;
  INFO(NCCL_NET, "NET/IB-CAST: device %s capability probe: UD=%s", dev->devName, dev->udSupported ? "yes" : "no");
  return true;
}

ncclResult_t IbCastCapProbeDevices() {
  for (int d = 0; d < IbCastNDevs; d++) {
    struct ncclIbDev* dev = &IbCastDevs[d];
    std::lock_guard<std::mutex> lock(dev->mutex);
    if (IbCastCapProbeLocked(dev) && dev->udSupported == 0) {
      WARN("NET/IB-CAST: device %s cannot create a UD queue pair; port recovery disabled on it", dev->devName);
    }
  }
  return ncclSuccess;
}

ncclResult_t IbCastCapHasUd(struct ncclIbDev* dev, bool* hasUd) {
  if (dev == NULL || hasUd == NULL) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lock(dev->mutex);
  IbCastCapProbeLocked(dev);
  *hasUd = (dev->udSupported == 1);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclIbCastGetDeviceCaps(int dev, struct ncclIbCastDeviceCaps* out) {
  if (out == NULL) return ncclInvalidArgument;
  if (dev < 0 || dev >= IbCastNMergedDevs) return ncclInvalidArgument;
  int physDev = IbCastMergedDevs[dev].vProps.devs[0];
  if (physDev < 0 || physDev >= IbCastNDevs) return ncclInvalidArgument;
  return IbCastCapHasUd(&IbCastDevs[physDev], &out->hasUd);
}
