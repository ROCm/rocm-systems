/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "transport_net_ib_mseg_wqe_lat_fakes.h"

#include <cstring>

#include "fail_loud.h"
#include "signature-drift.h"

bool IbMsegWqeLatEnabled = false;
uint64_t IbMsegWqeLatThresholdNs = 0;
bool IbMsegWqeLatReportEnabled = false;

void IbMsegWqeLatMonInit(struct IbMsegWqeLatMon* monitor) {
  if (monitor) std::memset(monitor, 0, sizeof(*monitor));
}

static void DefaultWqeLatMonStampSend(struct IbMsegQp*, struct ibv_send_wr*) {
  FailLoudUnfaked("transport_net_ib_mseg_wqe_lat_fakes", "IbMsegWqeLatMonStampSend");
}
static void DefaultWqeLatHandleCompletion(struct IbMsegNetCommBase*, int, struct ibv_wc*, uint64_t*, bool*) {
  FailLoudUnfaked("transport_net_ib_mseg_wqe_lat_fakes", "IbMsegWqeLatHandleCompletion");
}
static void DefaultWqeLatScanStalls(struct IbMsegNetCommBase*, int) {
  FailLoudUnfaked("transport_net_ib_mseg_wqe_lat_fakes", "IbMsegWqeLatScanStalls");
}

std::function<void(struct IbMsegQp*, struct ibv_send_wr*)> g_ncclIbWqeLatMonStampSend =
    DefaultWqeLatMonStampSend;
std::function<void(struct IbMsegNetCommBase*, int, struct ibv_wc*, uint64_t*, bool*)>
    g_ncclIbWqeLatHandleCompletion = DefaultWqeLatHandleCompletion;
std::function<void(struct IbMsegNetCommBase*, int)> g_ncclIbWqeLatScanStalls =
    DefaultWqeLatScanStalls;

ASSERT_HOOK_MATCHES_PROD(g_ncclIbWqeLatMonStampSend, IbMsegWqeLatMonStampSend);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbWqeLatHandleCompletion, IbMsegWqeLatHandleCompletion);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbWqeLatScanStalls, IbMsegWqeLatScanStalls);
#undef ASSERT_HOOK_MATCHES_PROD

void IbMsegWqeLatMonStampSend(struct IbMsegQp* qp, struct ibv_send_wr* wr) {
  g_ncclIbWqeLatMonStampSend(qp, wr);
}
void IbMsegWqeLatHandleCompletion(struct IbMsegNetCommBase* base, int devIndex, struct ibv_wc* wc,
                                  uint64_t* pollNs, bool* pollNsValid) {
  g_ncclIbWqeLatHandleCompletion(base, devIndex, wc, pollNs, pollNsValid);
}
void IbMsegWqeLatScanStalls(struct IbMsegNetCommBase* base, int devIndex) {
  g_ncclIbWqeLatScanStalls(base, devIndex);
}
void IbMsegWqeLatReportQpSummary(struct IbMsegNetCommBase*, int, struct IbMsegQp*) {}

void ResetTransportNetIbWqeLatFakes() {
  IbMsegWqeLatEnabled = false;
  IbMsegWqeLatThresholdNs = 0;
  IbMsegWqeLatReportEnabled = false;
  g_ncclIbWqeLatMonStampSend = DefaultWqeLatMonStampSend;
  g_ncclIbWqeLatHandleCompletion = DefaultWqeLatHandleCompletion;
  g_ncclIbWqeLatScanStalls = DefaultWqeLatScanStalls;
}
