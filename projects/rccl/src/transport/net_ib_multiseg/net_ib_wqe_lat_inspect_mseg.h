/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_IB_WQE_LAT_INSPECT_MSEG_H_
#define RCCL_NET_IB_WQE_LAT_INSPECT_MSEG_H_

#include <stdint.h>
#include "net_ib_limits.h"
#include "wqe_lat_mon_mseg.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Test-only introspection of the net_ib per-QP WQE post-to-poll latency
 * monitor (wqe_lat_mon.cc). Lets tests read each QP's latMon of a live
 * send/recv comm without depending on the net_ib comm layout.
 */

struct IbMsegWqeLatQpState {
  uint32_t qpNum;
  int devIndex;
  bool tracking;
  uint32_t inflight;
  uint32_t pendingBefore;
  uint64_t lastStallWarnNs; /* 0 until the QP has reported a stall */
  struct IbMsegWqeLatStats stats;
};

struct IbMsegWqeLatCommState {
  int isSend;
  int nqps;
  int nqpsPerRequest;
  struct IbMsegWqeLatQpState qps[NCCL_IB_MAX_QPS];
};

/* comm is a net_ib send or recv comm as returned by ncclNetIbMseg connect/accept. */
ncclResult_t IbMsegWqeLatGetCommState(void* comm, struct IbMsegWqeLatCommState* out);

/* Runs IbMsegWqeLatMonStampSend on m as if m were the latMon of a QP posting head. */
ncclResult_t IbMsegWqeLatTestStampSend(struct IbMsegWqeLatMon* m, struct ibv_send_wr* head);

/* Runs IbMsegWqeLatHandleCompletion on m as if m were the latMon of a QP whose CQ
 * returned a completion with this opcode at tPollNs. The QP has no device, so the
 * completion must not be slow: a slow report would read device state. */
ncclResult_t IbMsegWqeLatTestHandleCompletion(struct IbMsegWqeLatMon* m, enum ibv_wc_opcode opcode, uint64_t tPollNs);

/* Test-only. hostMem is the flush dummy payload; gpuMem is device dev's RO=0
 * scratchpad, or null when that device has none. */
ncclResult_t IbMsegGdrFlushGetScratch(void* recvComm, int dev, int** hostMem, int** gpuMem);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RCCL_NET_IB_WQE_LAT_INSPECT_MSEG_H_ */
