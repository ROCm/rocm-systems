/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Symbols owned by src/transport/net_ib_multiseg/connect.cc, for the microtest binaries that do not compile it.

#include "transport_net_ib_mseg_fakes.h"

#include <cstdint>

#include "fail_loud.h"
#include "fakes/nccl_fakes.h"
#include "fakes/signature-drift.h"
#include "net.h"
#include "transport/net_ib_multiseg/common_mseg.h"

// connect.cc defines this as an ncclParam<int32_t> object (USE_NCCL_PARAM in ib_write_bw.cc). The value
// comes from ncclParamEnvPluginGet, which nccl_fakes.cc routes through g_loadParam uncached. The
// parser is NCCL_PARAM_DEFAULT, not connect.cc's ncclParamBounded(1, 128), so ib_write_bw.cc's own qps < 1
// check stays reachable.
DEFINE_NCCL_PARAM(ncclParamIbQpsPerConn, int32_t, NCCL_IB_QPS_PER_CONNECTION, 1, NCCL_PARAM_FLAG_NONE,
                  NCCL_PARAM_DEFAULT, "fake");

bool g_ibGdrFlushFlagLocalSupport = false;
bool IbMsegGdrFlushFlagLocalSupport() { return g_ibGdrFlushFlagLocalSupport; }

std::vector<std::pair<void*, int>> g_ibSetGdrFlushFlagCalls;
void IbMsegSetGdrFlushFlag(void* netContext, int enable) { g_ibSetGdrFlushFlagCalls.emplace_back(netContext, enable); }

static ncclResult_t DefaultIbGidInfoQuery(struct ibv_context*, uint8_t, struct ibv_port_attr*,
                                          struct IbMsegGidInfo*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_context*, uint8_t, struct ibv_port_attr*, struct IbMsegGidInfo*)>
    g_ncclIbGidInfoQuery = DefaultIbGidInfoQuery;
ASSERT_HOOK_MATCHES_PROD(g_ncclIbGidInfoQuery, IbMsegGidInfoQuery);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t IbMsegGidInfoQuery(struct ibv_context* context, uint8_t port, struct ibv_port_attr* attr,
                                struct IbMsegGidInfo* info) {
  return g_ncclIbGidInfoQuery(context, port, attr, info);
}

ncclResult_t IbMsegListen(void*, int, void*, void**) {
  FailLoudUnfaked("transport_net_ib_fakes", "IbMsegListen");
}
ncclResult_t IbMsegNetListen(void*, int, void*, void**) {
  FailLoudUnfaked("transport_net_ib_fakes", "IbMsegNetListen");
}
ncclResult_t IbMsegConnect(void*, int, void*, void**, ncclNetDeviceHandle_t**) {
  FailLoudUnfaked("transport_net_ib_fakes", "IbMsegConnect");
}
ncclResult_t IbMsegAccept(void*, void**, ncclNetDeviceHandle_t**) {
  FailLoudUnfaked("transport_net_ib_fakes", "IbMsegAccept");
}
ncclResult_t IbMsegCloseSend(void*) {
  FailLoudUnfaked("transport_net_ib_fakes", "IbMsegCloseSend");
}
ncclResult_t IbMsegCloseRecv(void*) {
  FailLoudUnfaked("transport_net_ib_fakes", "IbMsegCloseRecv");
}
ncclResult_t IbMsegCloseListen(void*) {
  FailLoudUnfaked("transport_net_ib_fakes", "IbMsegCloseListen");
}

void ResetTransportNetIbFakes() {
  g_ibGdrFlushFlagLocalSupport = false;
  g_ibSetGdrFlushFlagCalls.clear();
  g_ncclIbGidInfoQuery = DefaultIbGidInfoQuery;
}
