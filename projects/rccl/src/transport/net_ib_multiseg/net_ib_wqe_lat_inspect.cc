/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common_mseg.h"
#include "wqe_lat_mon_mseg.h"
#include "net_ib_wqe_lat_inspect_mseg.h"

ncclResult_t IbMsegWqeLatGetCommState(void* comm, struct IbMsegWqeLatCommState* out) {
  static_assert(offsetof(struct IbMsegSendComm, base) == 0 && offsetof(struct IbMsegRecvComm, base) == 0,
                "base must be the first member of send and recv comms");
  if (comm == NULL || out == NULL) return ncclInvalidArgument;
  memset(out, 0, sizeof(*out));
  struct IbMsegNetCommBase* base = &((struct IbMsegSendComm*)comm)->base;
  if (base->nqps < 0 || base->nqps > NCCL_IB_MAX_QPS) return ncclInternalError;

  out->isSend = base->isSend;
  out->nqps = base->nqps;
  NCCLCHECK(IbMsegCommBaseGetNqpsPerRequest(base, &out->nqpsPerRequest));
  for (int q = 0; q < base->nqps; q++) {
    const struct IbMsegQp* qp = &base->qps[q];
    struct IbMsegWqeLatQpState* s = &out->qps[q];
    s->qpNum = qp->qp ? qp->qp->qp_num : 0;
    s->devIndex = qp->devIndex;
    s->tracking = qp->latMon.tracking;
    s->inflight = qp->latMon.inflight;
    s->pendingBefore = qp->latMon.pendingBefore;
    s->lastStallWarnNs = qp->latMon.lastStallWarnNs;
    IbMsegWqeLatMonSnapshot(&qp->latMon, &s->stats);
  }
  return ncclSuccess;
}

ncclResult_t IbMsegWqeLatTestStampSend(struct IbMsegWqeLatMon* m, struct ibv_send_wr* head) {
  if (m == NULL) return ncclInvalidArgument;
  struct IbMsegQp qp;
  memset(&qp, 0, sizeof(qp));
  qp.latMon = *m;
  IbMsegWqeLatMonStampSend(&qp, head);
  *m = qp.latMon;
  return ncclSuccess;
}

ncclResult_t IbMsegWqeLatTestHandleCompletion(struct IbMsegWqeLatMon* m, enum ibv_wc_opcode opcode, uint64_t tPollNs) {
  if (m == NULL) return ncclInvalidArgument;
  // A one-device, one-QP comm, enough for the QP lookup by number.
  struct IbMsegNetCommBase* base = NULL;
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
  IbMsegWqeLatHandleCompletion(base, 0, &wc, &tPollNs, &tPollNsValid);
  *m = base->qps[0].latMon;
  free(base);
  return ncclSuccess;
}

ncclResult_t IbMsegGdrFlushGetScratch(void* recvComm, int dev, int** hostMem, int** gpuMem) {
  if (recvComm == NULL || hostMem == NULL || gpuMem == NULL) return ncclInvalidArgument;
  struct IbMsegRecvComm* comm = (struct IbMsegRecvComm*)recvComm;
  if (dev < 0 || dev >= comm->base.vProps.ndevs) return ncclInvalidArgument;
  *hostMem = &comm->gpuFlushHostMem;
  *gpuMem = comm->devs[dev].gpuFlush.gpuFlushGpuMem;
  return ncclSuccess;
}
