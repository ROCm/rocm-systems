/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_FAULT_INJECTION

#include "common_cast.h"
#include "p2p_resiliency_cast.h"
#include "net_ib_gid_inspect.h"

static void IbCastGidToState(const struct ncclIbGidInfo* info, int ibDev, struct ncclIbGidState* out) {
  out->linkLayer = info->link_layer;
  out->gidIndex = info->localGidIndex;
  memcpy(out->gid, info->localGid.raw, sizeof(out->gid));
  out->ibDev = ibDev;
}

static void IbCastGidFromState(const struct ncclIbGidState* in, struct ncclIbGidInfo* info) {
  info->link_layer = in->linkLayer;
  info->localGidIndex = in->gidIndex;
  memcpy(info->localGid.raw, in->gid, sizeof(in->gid));
}

static struct ncclIbNetCommBase* IbCastGidCommBase(void* comm) {
  static_assert(offsetof(struct ncclIbSendComm, base) == 0 && offsetof(struct ncclIbRecvComm, base) == 0,
                "base must be the first member of send and recv comms");
  return comm ? &((struct ncclIbSendComm*)comm)->base : NULL;
}

static struct ncclIbNetCommDevBase* IbCastGidCommDevBase(void* comm, int devIndex) {
  struct ncclIbNetCommBase* base = IbCastGidCommBase(comm);
  if (base == NULL || devIndex < 0 || devIndex >= base->vProps.ndevs) return NULL;
  return IbCastGetNetCommDevBase(base, devIndex);
}

extern "C" ncclResult_t ncclIbCastGidGetDev(int ibDev, struct ncclIbGidState* out) {
  if (out == NULL || ibDev < 0 || ibDev >= IbCastNDevs) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lock(IbCastDevs[ibDev].mutex);
  IbCastGidToState(&IbCastDevs[ibDev].gidInfo, -1, out);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclIbCastGidSetDev(int ibDev, const struct ncclIbGidState* in) {
  if (in == NULL || ibDev < 0 || ibDev >= IbCastNDevs) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lock(IbCastDevs[ibDev].mutex);
  IbCastGidFromState(in, &IbCastDevs[ibDev].gidInfo);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclIbCastGidGetComm(void* comm, int devIndex, struct ncclIbGidState* out) {
  struct ncclIbNetCommDevBase* devBase = IbCastGidCommDevBase(comm, devIndex);
  if (devBase == NULL || out == NULL) return ncclInvalidArgument;
  IbCastGidToState(&devBase->gidInfo, devBase->ibDevN, out);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclIbCastGidSetComm(void* comm, int devIndex, const struct ncclIbGidState* in) {
  struct ncclIbNetCommDevBase* devBase = IbCastGidCommDevBase(comm, devIndex);
  if (devBase == NULL || in == NULL) return ncclInvalidArgument;
  IbCastGidFromState(in, &devBase->gidInfo);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclIbCastGidGetQpState(void* comm, struct ncclIbGidQpState* out) {
  struct ncclIbNetCommBase* base = IbCastGidCommBase(comm);
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

extern "C" ncclResult_t ncclIbCastGidGetDevState(void* comm, int devIndex, int* state) {
  struct ncclIbNetCommBase* base = IbCastGidCommBase(comm);
  if (base == NULL || state == NULL || base->resiliency == NULL) return ncclInvalidArgument;
  if (devIndex < 0 || devIndex >= base->resiliency->ndevs) return ncclInvalidArgument;
  *state = (int)base->resiliency->devs[devIndex].state.load(std::memory_order_acquire);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclIbCastGidGetRecoveryGidIndex(void* comm, int devIndex, int* gidIndex) {
  struct ncclIbNetCommBase* base = IbCastGidCommBase(comm);
  if (base == NULL || gidIndex == NULL || base->resiliency == NULL) return ncclInvalidArgument;
  if (devIndex < 0 || devIndex >= NCCL_IB_MAX_DEVS_PER_NIC) return ncclInvalidArgument;
  if (base->resiliency->portRecoveryAh[devIndex] == NULL || !base->resiliency->portRecoveryAhAttr[devIndex].is_global)
    return ncclInvalidArgument;
  *gidIndex = base->resiliency->portRecoveryAhAttr[devIndex].grh.sgid_index;
  return ncclSuccess;
}

extern "C" ncclResult_t ncclIbCastGidDriveQpToError(void* comm, int qpIdx) {
  struct ncclIbNetCommBase* base = IbCastGidCommBase(comm);
  if (base == NULL || qpIdx < 0 || qpIdx >= base->nqps || base->qps[qpIdx].qp == NULL) return ncclInvalidArgument;
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_ERR;
  NCCLCHECK(wrap_ibv_modify_qp(base->qps[qpIdx].qp, &attr, IBV_QP_STATE));
  return ncclSuccess;
}
#endif /* ENABLE_FAULT_INJECTION */
