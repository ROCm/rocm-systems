/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_IB_WQE_LAT_INSPECT_H_
#define RCCL_NET_IB_WQE_LAT_INSPECT_H_

#include <stdint.h>
#include "net_ib_limits.h"
#include "net_ib/wqe_lat_mon.h"

/*
 * Test-only introspection of the net_ib per-QP WQE post-to-poll latency
 * monitor (wqe_lat_mon.cc). Lets tests read each QP's latMon of a live
 * send/recv comm without depending on the net_ib comm layout.
 */

struct ncclIbWqeLatQpState {
  uint32_t qpNum;
  int devIndex;
  bool tracking;
  uint32_t inflight;
  uint32_t pendingBefore;
  struct ncclIbWqeLatStats stats;
};

struct ncclIbWqeLatCommState {
  int isSend;
  int nqps;
  int nqpsPerRequest;
  struct ncclIbWqeLatQpState qps[NCCL_IB_MAX_QPS];
};

/* comm is a net_ib send or recv comm as returned by ncclNetIb connect/accept. */
ncclResult_t ncclIbWqeLatGetCommState(void* comm, struct ncclIbWqeLatCommState* out);

/* Runs ncclIbWqeLatMonStampSend on m as if m were the latMon of a QP posting head. */
ncclResult_t ncclIbWqeLatTestStampSend(struct ncclIbWqeLatMon* m, struct ibv_send_wr* head);

#endif
