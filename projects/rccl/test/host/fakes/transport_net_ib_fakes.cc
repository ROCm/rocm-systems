/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Symbols owned by src/transport/net_ib/ and net_ib_cast/, for the microtest binaries that do not compile them.

#include "transport_net_ib_fakes.h"

#include <cstdint>

#include "fakes/fail_loud.h"
#include "fakes/nccl_fakes.h"
#include "fakes/signature-drift.h"
#include "net.h"

int64_t ncclParamIbQpsPerConn() { return g_loadParam("IB_QPS_PER_CONNECTION", 1); }

// net.cc compares a plugin's address against these.
ncclNet_t ncclNetIb = {"IB"};
ncclNet_t netIbCast = {"IBCast"};

static ncclResult_t DefaultNetP2pPolicy(void*, int) { FailLoudUnfaked("transport_net_ib_fakes", "rcclNetP2pPolicy"); }
std::function<ncclResult_t(void*, int)> g_rcclNetP2pPolicy = DefaultNetP2pPolicy;
ASSERT_HOOK_MATCHES_PROD(g_rcclNetP2pPolicy, rcclNetP2pPolicy);
ncclResult_t rcclNetP2pPolicy(void* handle, int isP2p) { return g_rcclNetP2pPolicy(handle, isP2p); }

static ncclResult_t DefaultCastNetP2pPolicy(void*, int) {
  FailLoudUnfaked("transport_net_ib_fakes", "rcclCastNetP2pPolicy");
}
std::function<ncclResult_t(void*, int)> g_rcclCastNetP2pPolicy = DefaultCastNetP2pPolicy;
ASSERT_HOOK_MATCHES_PROD(g_rcclCastNetP2pPolicy, rcclCastNetP2pPolicy);
ncclResult_t rcclCastNetP2pPolicy(void* handle, int isP2p) { return g_rcclCastNetP2pPolicy(handle, isP2p); }

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

void ResetTransportNetIbFakes() {
  g_rcclNetP2pPolicy = DefaultNetP2pPolicy;
  g_rcclCastNetP2pPolicy = DefaultCastNetP2pPolicy;
  g_ncclIbGidInfoQuery = DefaultIbGidInfoQuery;
}
