/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_CAST_WQE_LAT_MON_CAST_H_
#define NET_IB_CAST_WQE_LAT_MON_CAST_H_

#include "nccl.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef NCCL_BUILD_RDMA_CORE
#include <infiniband/verbs.h>
#else
#include "ibvcore.h"
#endif

// Port of src/transport/net_ib/wqe_lat_mon.{h,cc} for net_ib_cast. Reads the
// same NCCL_IB_WQE_LATENCY_* env vars as net_ib so behavior is seamless
// across NCCL_NET=IB vs IB-CAST/ROCM-IB, but uses independent symbol names
// throughout since both transports link into the same binary.

extern bool ncclIbCastWqeLatEnabled;
extern uint64_t ncclIbCastWqeLatThresholdNs;
extern bool ncclIbCastWqeLatReportEnabled;

uint64_t ncclIbCastWqeLatMonNowNs(void);

#define NCCL_IB_CAST_WQE_LAT_NUM_PCTL 4
struct ncclIbCastP2Quantile {
  double p;
  double q[5];
  uint64_t n[5];
};

struct ncclIbCastWqeLatMon {
  uint64_t trackedPostNs;
  uint32_t pendingBefore;
  uint32_t inflight;
  bool tracking;

  uint64_t count;
  double meanNs;
  double m2Ns;
  uint64_t maxNs;
  uint64_t slowCount;
  uint64_t stallCount;

  struct ncclIbCastP2Quantile p2[NCCL_IB_CAST_WQE_LAT_NUM_PCTL];

  uint64_t lastWarnNs;
  uint64_t lastStallWarnNs;
};

struct ncclIbCastWqeLatStats {
  uint64_t count;
  uint64_t slowCount;
  double meanNs;
  double stddevNs;
  uint64_t maxNs;
  uint64_t p50Ns;
  uint64_t p90Ns;
  uint64_t p99Ns;
  uint64_t p999Ns;
};

struct ncclIbQp;
struct ncclIbNetCommBase;
struct ibv_wc;

void IbCastWqeLatMonInit(struct ncclIbCastWqeLatMon* m);

void IbCastWqeLatMonStampSend(struct ncclIbQp* qp, struct ibv_send_wr* head);

bool IbCastWqeLatMonOnComplete(struct ncclIbCastWqeLatMon* m, uint64_t tPollNs, uint64_t* outDeltaNs,
                               uint64_t* outPostNs, struct ncclIbCastWqeLatStats* outStats);

bool IbCastWqeLatMonCheckStall(struct ncclIbCastWqeLatMon* m, uint64_t nowNs, uint64_t* outAgeNs,
                               uint32_t* outInflight);

void IbCastWqeLatMonSnapshot(const struct ncclIbCastWqeLatMon* m, struct ncclIbCastWqeLatStats* out);

void IbCastWqeLatHandleCompletion(struct ncclIbNetCommBase* base, int devIndex, struct ibv_wc* wc, uint64_t* tPollNs,
                                  bool* tPollNsValid);

void IbCastWqeLatScanStalls(struct ncclIbNetCommBase* base, int devIndex);

void IbCastWqeLatReportQpSummary(struct ncclIbNetCommBase* base, int devIndex, struct ncclIbQp* qp);

#endif  // NET_IB_CAST_WQE_LAT_MON_CAST_H_
