/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_FAULT_INJECTION

#include "common_mseg.h"
#include "p2p_resiliency_mseg.h"
#include "net_ib_gid_inspect_mseg.h"

static void IbMsegGidToState(const struct IbMsegGidInfo* info, int ibDev, struct IbMsegGidState* out) {
  out->linkLayer = info->link_layer;
  out->gidIndex = info->localGidIndex;
  memcpy(out->gid, info->localGid.raw, sizeof(out->gid));
  out->ibDev = ibDev;
}

static void IbMsegGidFromState(const struct IbMsegGidState* in, struct IbMsegGidInfo* info) {
  info->link_layer = in->linkLayer;
  info->localGidIndex = in->gidIndex;
  memcpy(info->localGid.raw, in->gid, sizeof(in->gid));
}

static struct IbMsegNetCommBase* IbMsegGidCommBase(void* comm) {
  static_assert(offsetof(struct IbMsegSendComm, base) == 0 && offsetof(struct IbMsegRecvComm, base) == 0,
                "base must be the first member of send and recv comms");
  return comm ? &((struct IbMsegSendComm*)comm)->base : NULL;
}

static struct IbMsegNetCommDevBase* IbMsegGidCommDevBase(void* comm, int devIndex) {
  struct IbMsegNetCommBase* base = IbMsegGidCommBase(comm);
  if (base == NULL || devIndex < 0 || devIndex >= base->vProps.ndevs) return NULL;
  return IbMsegGetNetCommDevBase(base, devIndex);
}

extern "C" ncclResult_t IbMsegGidGetDev(int ibDev, struct IbMsegGidState* out) {
  if (out == NULL || ibDev < 0 || ibDev >= IbMsegNDevs) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lock(IbMsegDevs[ibDev].mutex);
  IbMsegGidToState(&IbMsegDevs[ibDev].gidInfo, -1, out);
  return ncclSuccess;
}

extern "C" ncclResult_t IbMsegGidSetDev(int ibDev, const struct IbMsegGidState* in) {
  if (in == NULL || ibDev < 0 || ibDev >= IbMsegNDevs) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lock(IbMsegDevs[ibDev].mutex);
  IbMsegGidFromState(in, &IbMsegDevs[ibDev].gidInfo);
  return ncclSuccess;
}

extern "C" ncclResult_t IbMsegGidGetComm(void* comm, int devIndex, struct IbMsegGidState* out) {
  struct IbMsegNetCommDevBase* devBase = IbMsegGidCommDevBase(comm, devIndex);
  if (devBase == NULL || out == NULL) return ncclInvalidArgument;
  IbMsegGidToState(&devBase->gidInfo, devBase->ibDevN, out);
  return ncclSuccess;
}

extern "C" ncclResult_t IbMsegGidSetComm(void* comm, int devIndex, const struct IbMsegGidState* in) {
  struct IbMsegNetCommDevBase* devBase = IbMsegGidCommDevBase(comm, devIndex);
  if (devBase == NULL || in == NULL) return ncclInvalidArgument;
  IbMsegGidFromState(in, &devBase->gidInfo);
  return ncclSuccess;
}

extern "C" ncclResult_t IbMsegGidGetQpState(void* comm, struct IbMsegGidQpState* out) {
  struct IbMsegNetCommBase* base = IbMsegGidCommBase(comm);
  if (base == NULL || out == NULL) return ncclInvalidArgument;
  memset(out, 0, sizeof(*out));
  out->nqps = std::min(std::max(base->nqps, 0), NCCL_IB_MAX_QPS);
  for (int i = 0; i < out->nqps; i++) {
    struct IbMsegQp* qp = &base->qps[i];
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

extern "C" ncclResult_t IbMsegGidGetDevState(void* comm, int devIndex, int* state) {
  struct IbMsegNetCommBase* base = IbMsegGidCommBase(comm);
  if (base == NULL || state == NULL || base->resiliency == NULL) return ncclInvalidArgument;
  if (devIndex < 0 || devIndex >= base->resiliency->ndevs) return ncclInvalidArgument;
  *state = (int)base->resiliency->devs[devIndex].state.load(std::memory_order_acquire);
  return ncclSuccess;
}

extern "C" ncclResult_t IbMsegGidGetRecoveryGidIndex(void* comm, int devIndex, int* gidIndex) {
  struct IbMsegNetCommBase* base = IbMsegGidCommBase(comm);
  if (base == NULL || gidIndex == NULL || base->resiliency == NULL) return ncclInvalidArgument;
  for (int i = 0; i < base->resiliency->nPortRecoveryQps; i++) {
    struct IbMsegQp* qp = &base->resiliency->portRecoveryQps[i];
    if (qp->qp == NULL || qp->devIndex != devIndex) continue;
    *gidIndex = qp->rtrAttr.localGidIndex;
    return ncclSuccess;
  }
  return ncclInvalidArgument;
}

extern "C" ncclResult_t IbMsegGidGetProbingGidIndex(void* comm, int devIndex, int* gidIndex) {
  struct IbMsegNetCommBase* base = IbMsegGidCommBase(comm);
  if (base == NULL || gidIndex == NULL || base->resiliency == NULL) return ncclInvalidArgument;
  for (int i = 0; i < base->resiliency->nProbingQps; i++) {
    struct IbMsegQp* qp = &base->resiliency->probingQps[i];
    if (qp->qp == NULL || qp->devIndex != devIndex) continue;
    *gidIndex = qp->rtrAttr.localGidIndex;
    return ncclSuccess;
  }
  return ncclInvalidArgument;
}

extern "C" ncclResult_t IbMsegGidUcSupported(int ibDev, int* supported) {
  if (supported == NULL || ibDev < 0 || ibDev >= IbMsegNDevs) return ncclInvalidArgument;
  struct ibv_context* context = IbMsegDevs[ibDev].context;
  struct ibv_pd* pd = NULL;
  struct ibv_cq* cq = NULL;
  struct IbMsegQp qp{};
  *supported = 0;
  if (wrap_ibv_alloc_pd(&pd, context) == ncclSuccess && wrap_ibv_create_cq(&cq, context, 1, NULL, NULL, 0) == ncclSuccess) {
    struct IbMsegQpCreateAttr attr{};
    attr.type = IBV_QPT_UC;
    attr.cq = cq;
    attr.pd = pd;
    attr.maxRecvWorkRequest = 1;
    attr.maxSendWorkRequest = 1;
    if (IbMsegQpCreate(&qp, &attr) == ncclSuccess && qp.qp != NULL) *supported = 1;
  }
  if (qp.qp) (void)wrap_ibv_destroy_qp(qp.qp);
  if (cq) (void)wrap_ibv_destroy_cq(cq);
  if (pd) (void)wrap_ibv_dealloc_pd(pd);
  return ncclSuccess;
}

extern "C" ncclResult_t IbMsegGidDriveQpToError(void* comm, int qpIdx) {
  struct IbMsegNetCommBase* base = IbMsegGidCommBase(comm);
  if (base == NULL || qpIdx < 0 || qpIdx >= base->nqps || base->qps[qpIdx].qp == NULL) return ncclInvalidArgument;
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_ERR;
  NCCLCHECK(wrap_ibv_modify_qp(base->qps[qpIdx].qp, &attr, IBV_QP_STATE));
  return ncclSuccess;
}
#endif /* ENABLE_FAULT_INJECTION */
