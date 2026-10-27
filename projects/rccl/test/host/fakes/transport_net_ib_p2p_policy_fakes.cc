/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of transport_net_ib_p2p_policy_fakes.h. Targets that compile the real
// net_ib/connect.cc must not link this file.

#include "transport_net_ib_p2p_policy_fakes.h"

#include "fakes/fail_loud.h"
#include "fakes/signature-drift.h"
#include "net.h"

static ncclResult_t DefaultNetP2pPolicy(void*, int) {
  FailLoudUnfaked("transport_net_ib_p2p_policy_fakes", "rcclNetP2pPolicy");
}
std::function<ncclResult_t(void*, int)> g_rcclNetP2pPolicy = DefaultNetP2pPolicy;
ASSERT_HOOK_MATCHES_PROD(g_rcclNetP2pPolicy, rcclNetP2pPolicy);
#undef ASSERT_HOOK_MATCHES_PROD
ncclResult_t rcclNetP2pPolicy(void* handle, int isP2p) { return g_rcclNetP2pPolicy(handle, isP2p); }

void ResetTransportNetIbP2pPolicyFakes() { g_rcclNetP2pPolicy = DefaultNetP2pPolicy; }
