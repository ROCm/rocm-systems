/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "transport_net_ib_wqe_lat_fakes.h"

#include <cstring>

#include "fail_loud.h"
#include "signature-drift.h"

bool ncclIbWqeLatEnabled = false;
uint64_t ncclIbWqeLatThresholdNs = 0;
bool ncclIbWqeLatReportEnabled = false;

void ncclIbWqeLatMonInit(struct ncclIbWqeLatMon* monitor) {
  if (monitor) std::memset(monitor, 0, sizeof(*monitor));
}

static void DefaultWqeLatMonStampSend(struct ncclIbQp*, struct ibv_send_wr*) {
  FailLoudUnfaked("transport_net_ib_wqe_lat_fakes", "ncclIbWqeLatMonStampSend");
}
static void DefaultWqeLatHandleCompletion(struct ncclIbNetCommBase*, int, struct ibv_wc*, uint64_t*, bool*) {
  FailLoudUnfaked("transport_net_ib_wqe_lat_fakes", "ncclIbWqeLatHandleCompletion");
}
static void DefaultWqeLatScanStalls(struct ncclIbNetCommBase*, int) {
  FailLoudUnfaked("transport_net_ib_wqe_lat_fakes", "ncclIbWqeLatScanStalls");
}

std::function<void(struct ncclIbQp*, struct ibv_send_wr*)> g_ncclIbWqeLatMonStampSend =
    DefaultWqeLatMonStampSend;
std::function<void(struct ncclIbNetCommBase*, int, struct ibv_wc*, uint64_t*, bool*)>
    g_ncclIbWqeLatHandleCompletion = DefaultWqeLatHandleCompletion;
std::function<void(struct ncclIbNetCommBase*, int)> g_ncclIbWqeLatScanStalls =
    DefaultWqeLatScanStalls;

ASSERT_HOOK_MATCHES_PROD(g_ncclIbWqeLatMonStampSend, ncclIbWqeLatMonStampSend);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbWqeLatHandleCompletion, ncclIbWqeLatHandleCompletion);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbWqeLatScanStalls, ncclIbWqeLatScanStalls);
#undef ASSERT_HOOK_MATCHES_PROD

void ncclIbWqeLatMonStampSend(struct ncclIbQp* qp, struct ibv_send_wr* wr) {
  g_ncclIbWqeLatMonStampSend(qp, wr);
}
void ncclIbWqeLatHandleCompletion(struct ncclIbNetCommBase* base, int devIndex, struct ibv_wc* wc,
                                  uint64_t* pollNs, bool* pollNsValid) {
  g_ncclIbWqeLatHandleCompletion(base, devIndex, wc, pollNs, pollNsValid);
}
void ncclIbWqeLatScanStalls(struct ncclIbNetCommBase* base, int devIndex) {
  g_ncclIbWqeLatScanStalls(base, devIndex);
}
void ncclIbWqeLatReportQpSummary(struct ncclIbNetCommBase*, int, struct ncclIbQp*) {}

void ResetTransportNetIbWqeLatFakes() {
  ncclIbWqeLatEnabled = false;
  ncclIbWqeLatThresholdNs = 0;
  ncclIbWqeLatReportEnabled = false;
  g_ncclIbWqeLatMonStampSend = DefaultWqeLatMonStampSend;
  g_ncclIbWqeLatHandleCompletion = DefaultWqeLatHandleCompletion;
  g_ncclIbWqeLatScanStalls = DefaultWqeLatScanStalls;
}
