/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_WQE_LAT_MON_MSEG_H_
#define NET_IB_WQE_LAT_MON_MSEG_H_

#include "nccl.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef NCCL_BUILD_RDMA_CORE
#include <infiniband/verbs.h>
#else
#include "ibvcore.h"
#endif

extern bool IbMsegWqeLatEnabled;
extern uint64_t IbMsegWqeLatThresholdNs;
extern bool IbMsegWqeLatReportEnabled;

uint64_t IbMsegWqeLatMonNowNs(void);

#define NCCL_IB_WQE_LAT_NUM_PCTL 4
struct IbMsegP2Quantile {
  double p;
  double q[5];
  uint64_t n[5];
};

struct IbMsegWqeLatMon {
  uint64_t trackedPostNs;
  uint32_t pendingBefore;
  uint32_t inflight;
  bool tracking;

  uint64_t count;
  double meanNs;
  double m2Ns;
  uint64_t maxNs;
  uint64_t slowCount;

  struct IbMsegP2Quantile p2[NCCL_IB_WQE_LAT_NUM_PCTL];

  uint64_t lastWarnNs;
  uint64_t lastStallWarnNs;
};

struct IbMsegWqeLatStats {
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

struct IbMsegQp;
struct IbMsegNetCommBase;
struct ibv_wc;

void IbMsegWqeLatMonInit(struct IbMsegWqeLatMon* m);

void IbMsegWqeLatMonStampSend(struct IbMsegQp* qp, struct ibv_send_wr* head);

bool IbMsegWqeLatMonOnComplete(struct IbMsegWqeLatMon* m, uint64_t tPollNs, uint64_t* outDeltaNs, uint64_t* outPostNs,
                               struct IbMsegWqeLatStats* outStats);

bool IbMsegWqeLatMonCheckStall(struct IbMsegWqeLatMon* m, uint64_t nowNs, uint64_t* outAgeNs, uint32_t* outInflight);

void IbMsegWqeLatMonSnapshot(const struct IbMsegWqeLatMon* m, struct IbMsegWqeLatStats* out);

void IbMsegWqeLatHandleCompletion(struct IbMsegNetCommBase* base, int devIndex, struct ibv_wc* wc, uint64_t* tPollNs,
                                  bool* tPollNsValid);

void IbMsegWqeLatScanStalls(struct IbMsegNetCommBase* base, int devIndex);

void IbMsegWqeLatReportQpSummary(struct IbMsegNetCommBase* base, int devIndex, struct IbMsegQp* qp);

#endif  // NET_IB_WQE_LAT_MON_MSEG_H_
