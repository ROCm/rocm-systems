/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_FAULT_INJECTION

#include "common.h"
#include "p2p_resiliency.h"
#include "net_ib_gid_inspect.h"

static void ncclIbGidToState(const struct ncclIbGidInfo* info, int ibDev, struct ncclIbGidState* out) {
  out->linkLayer = info->link_layer;
  out->gidIndex = info->localGidIndex;
  memcpy(out->gid, info->localGid.raw, sizeof(out->gid));
  out->ibDev = ibDev;
}

static void ncclIbGidFromState(const struct ncclIbGidState* in, struct ncclIbGidInfo* info) {
  info->link_layer = in->linkLayer;
  info->localGidIndex = in->gidIndex;
  memcpy(info->localGid.raw, in->gid, sizeof(in->gid));
}

static struct ncclIbNetCommBase* ncclIbGidCommBase(void* comm) {
  static_assert(offsetof(struct ncclIbSendComm, base) == 0 && offsetof(struct ncclIbRecvComm, base) == 0,
                "base must be the first member of send and recv comms");
  return comm ? &((struct ncclIbSendComm*)comm)->base : NULL;
}

static struct ncclIbNetCommDevBase* ncclIbGidCommDevBase(void* comm, int devIndex) {
  struct ncclIbNetCommBase* base = ncclIbGidCommBase(comm);
  if (base == NULL || devIndex < 0 || devIndex >= base->vProps.ndevs) return NULL;
  return ncclIbGetNetCommDevBase(base, devIndex);
}

extern "C" ncclResult_t ncclIbGidGetDev(int ibDev, struct ncclIbGidState* out) {
  if (out == NULL || ibDev < 0 || ibDev >= ncclNIbDevs) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lock(ncclIbDevs[ibDev].mutex);
  ncclIbGidToState(&ncclIbDevs[ibDev].gidInfo, -1, out);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclIbGidSetDev(int ibDev, const struct ncclIbGidState* in) {
  if (in == NULL || ibDev < 0 || ibDev >= ncclNIbDevs) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lock(ncclIbDevs[ibDev].mutex);
  ncclIbGidFromState(in, &ncclIbDevs[ibDev].gidInfo);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclIbGidGetComm(void* comm, int devIndex, struct ncclIbGidState* out) {
  struct ncclIbNetCommDevBase* devBase = ncclIbGidCommDevBase(comm, devIndex);
  if (devBase == NULL || out == NULL) return ncclInvalidArgument;
  ncclIbGidToState(&devBase->gidInfo, devBase->ibDevN, out);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclIbGidSetComm(void* comm, int devIndex, const struct ncclIbGidState* in) {
  struct ncclIbNetCommDevBase* devBase = ncclIbGidCommDevBase(comm, devIndex);
  if (devBase == NULL || in == NULL) return ncclInvalidArgument;
  ncclIbGidFromState(in, &devBase->gidInfo);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclIbGidGetQpState(void* comm, struct ncclIbGidQpState* out) {
  struct ncclIbNetCommBase* base = ncclIbGidCommBase(comm);
  if (base == NULL || out == NULL) return ncclInvalidArgument;
  memset(out, 0, sizeof(*out));
  out->nqps = std::min(std::max(base->nqps, 0), NCCL_IB_MAX_QPS);
  for (int i = 0; i < out->nqps; i++) {
    struct ncclIbQp* qp = &base->qps[i];
    out->devIndex[i] = qp->devIndex;
    out->rtrGidIndex[i] = qp->rtrAttr.localGidIndex;
    if (qp->qp == NULL) continue;
    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr initAttr;
    memset(&attr, 0, sizeof(attr));
    memset(&initAttr, 0, sizeof(initAttr));
    if (wrap_ibv_query_qp(qp->qp, &attr, IBV_QP_AV, &initAttr) == ncclSuccess) {
      out->sgidIndex[i] = attr.ah_attr.grh.sgid_index;
      out->queryOk[i] = true;
    }
  }
  return ncclSuccess;
}

extern "C" ncclResult_t ncclIbGidGetDevState(void* comm, int devIndex, int* state) {
  struct ncclIbNetCommBase* base = ncclIbGidCommBase(comm);
  if (base == NULL || state == NULL || base->resiliency == NULL) return ncclInvalidArgument;
  if (devIndex < 0 || devIndex >= base->resiliency->ndevs) return ncclInvalidArgument;
  *state = (int)base->resiliency->devs[devIndex].state.load(std::memory_order_acquire);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclIbGidGetRecoveryGidIndex(void* comm, int devIndex, int* gidIndex) {
  struct ncclIbNetCommBase* base = ncclIbGidCommBase(comm);
  if (base == NULL || gidIndex == NULL || base->resiliency == NULL) return ncclInvalidArgument;
  for (int i = 0; i < base->resiliency->nPortRecoveryQps; i++) {
    struct ncclIbQp* qp = &base->resiliency->portRecoveryQps[i];
    if (qp->qp == NULL || qp->devIndex != devIndex) continue;
    *gidIndex = qp->rtrAttr.localGidIndex;
    return ncclSuccess;
  }
  return ncclInvalidArgument;
}

extern "C" ncclResult_t ncclIbGidGetProbingGidIndex(void* comm, int devIndex, int* gidIndex) {
  struct ncclIbNetCommBase* base = ncclIbGidCommBase(comm);
  if (base == NULL || gidIndex == NULL || base->resiliency == NULL) return ncclInvalidArgument;
  for (int i = 0; i < base->resiliency->nProbingQps; i++) {
    struct ncclIbQp* qp = &base->resiliency->probingQps[i];
    if (qp->qp == NULL || qp->devIndex != devIndex) continue;
    *gidIndex = qp->rtrAttr.localGidIndex;
    return ncclSuccess;
  }
  return ncclInvalidArgument;
}

extern "C" ncclResult_t ncclIbGidDriveQpToError(void* comm, int qpIdx) {
  struct ncclIbNetCommBase* base = ncclIbGidCommBase(comm);
  if (base == NULL || qpIdx < 0 || qpIdx >= base->nqps || base->qps[qpIdx].qp == NULL) return ncclInvalidArgument;
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_ERR;
  NCCLCHECK(wrap_ibv_modify_qp(base->qps[qpIdx].qp, &attr, IBV_QP_STATE));
  return ncclSuccess;
}
#endif /* ENABLE_FAULT_INJECTION */
