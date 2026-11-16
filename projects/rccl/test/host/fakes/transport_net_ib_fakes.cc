/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Symbols owned by src/transport/net_ib/connect.cc, for the microtest binaries that do not compile it.

#include "transport_net_ib_fakes.h"

#include <cstdint>

#include "fail_loud.h"
#include "fakes/nccl_fakes.h"
#include "fakes/signature-drift.h"
#include "net.h"
#include "transport/net_ib/common.h"

int64_t ncclParamIbQpsPerConn() { return g_loadParam("IB_QPS_PER_CONNECTION", 1); }

bool g_ibGdrFlushFlagLocalSupport = false;
bool ncclIbGdrFlushFlagLocalSupport() { return g_ibGdrFlushFlagLocalSupport; }

std::vector<std::pair<void*, int>> g_ibSetGdrFlushFlagCalls;
void ncclIbSetGdrFlushFlag(void* netContext, int enable) { g_ibSetGdrFlushFlagCalls.emplace_back(netContext, enable); }

static ncclResult_t DefaultIbGidInfoQuery(struct ibv_context*, uint8_t, struct ibv_port_attr*,
                                          struct ncclIbGidInfo*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_context*, uint8_t, struct ibv_port_attr*, struct ncclIbGidInfo*)>
    g_ncclIbGidInfoQuery = DefaultIbGidInfoQuery;
ASSERT_HOOK_MATCHES_PROD(g_ncclIbGidInfoQuery, ncclIbGidInfoQuery);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t ncclIbGidInfoQuery(struct ibv_context* context, uint8_t port, struct ibv_port_attr* attr,
                                struct ncclIbGidInfo* info) {
  return g_ncclIbGidInfoQuery(context, port, attr, info);
}

ncclResult_t ncclIbListen(void*, int, void*, void**) {
  FailLoudUnfaked("transport_net_ib_fakes", "ncclIbListen");
}
ncclResult_t ncclIbNetListen(void*, int, void*, void**) {
  FailLoudUnfaked("transport_net_ib_fakes", "ncclIbNetListen");
}
ncclResult_t ncclIbConnect(void*, int, void*, void**, ncclNetDeviceHandle_t**) {
  FailLoudUnfaked("transport_net_ib_fakes", "ncclIbConnect");
}
ncclResult_t ncclIbAccept(void*, void**, ncclNetDeviceHandle_t**) {
  FailLoudUnfaked("transport_net_ib_fakes", "ncclIbAccept");
}
ncclResult_t ncclIbCloseSend(void*) {
  FailLoudUnfaked("transport_net_ib_fakes", "ncclIbCloseSend");
}
ncclResult_t ncclIbCloseRecv(void*) {
  FailLoudUnfaked("transport_net_ib_fakes", "ncclIbCloseRecv");
}
ncclResult_t ncclIbCloseListen(void*) {
  FailLoudUnfaked("transport_net_ib_fakes", "ncclIbCloseListen");
}

void ResetTransportNetIbFakes() {
  g_ibGdrFlushFlagLocalSupport = false;
  g_ibSetGdrFlushFlagCalls.clear();
  g_ncclIbGidInfoQuery = DefaultIbGidInfoQuery;
}
