/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "wqe_lat_mon.h"
#include "net_ib_wqe_lat_inspect.h"

ncclResult_t ncclIbWqeLatGetCommState(void* comm, struct ncclIbWqeLatCommState* out) {
  static_assert(offsetof(struct ncclIbSendComm, base) == 0 && offsetof(struct ncclIbRecvComm, base) == 0,
                "base must be the first member of send and recv comms");
  if (comm == NULL || out == NULL) return ncclInvalidArgument;
  memset(out, 0, sizeof(*out));
  struct ncclIbNetCommBase* base = &((struct ncclIbSendComm*)comm)->base;
  if (base->nqps < 0 || base->nqps > NCCL_IB_MAX_QPS) return ncclInternalError;

  out->isSend = base->isSend;
  out->nqps = base->nqps;
  NCCLCHECK(ncclIbCommBaseGetNqpsPerRequest(base, &out->nqpsPerRequest));
  for (int q = 0; q < base->nqps; q++) {
    const struct ncclIbQp* qp = &base->qps[q];
    struct ncclIbWqeLatQpState* s = &out->qps[q];
    s->qpNum = qp->qp ? qp->qp->qp_num : 0;
    s->devIndex = qp->devIndex;
    s->tracking = qp->latMon.tracking;
    s->inflight = qp->latMon.inflight;
    s->pendingBefore = qp->latMon.pendingBefore;
    s->lastStallWarnNs = qp->latMon.lastStallWarnNs;
    ncclIbWqeLatMonSnapshot(&qp->latMon, &s->stats);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbWqeLatTestStampSend(struct ncclIbWqeLatMon* m, struct ibv_send_wr* head) {
  if (m == NULL) return ncclInvalidArgument;
  struct ncclIbQp qp;
  memset(&qp, 0, sizeof(qp));
  qp.latMon = *m;
  ncclIbWqeLatMonStampSend(&qp, head);
  *m = qp.latMon;
  return ncclSuccess;
}

ncclResult_t ncclIbWqeLatTestHandleCompletion(struct ncclIbWqeLatMon* m, enum ibv_wc_opcode opcode, uint64_t tPollNs) {
  if (m == NULL) return ncclInvalidArgument;
  // A one-device, one-QP comm, enough for the QP lookup by number.
  struct ncclIbNetCommBase* base = NULL;
  NCCLCHECK(ncclIbMalloc((void**)&base, sizeof(*base)));
  struct ibv_qp ibvQp;
  memset(&ibvQp, 0, sizeof(ibvQp));
  ibvQp.qp_num = 1;
  base->vProps.ndevs = 1;
  base->nqps = 1;
  base->qps[0].qp = &ibvQp;
  base->qps[0].latMon = *m;

  struct ibv_wc wc;
  memset(&wc, 0, sizeof(wc));
  wc.opcode = opcode;
  wc.qp_num = ibvQp.qp_num;
  bool tPollNsValid = true;
  ncclIbWqeLatHandleCompletion(base, 0, &wc, &tPollNs, &tPollNsValid);
  *m = base->qps[0].latMon;
  free(base);
  return ncclSuccess;
}
