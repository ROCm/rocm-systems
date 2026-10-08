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
  struct ncclIbNetCommBase* base = &((struct ncclIbSendComm*)comm)->base;
  if (base->nqps < 0 || base->nqps > NCCL_IB_MAX_QPS) return ncclInternalError;

  memset(out, 0, sizeof(*out));
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
