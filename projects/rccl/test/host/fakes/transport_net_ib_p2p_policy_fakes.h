/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seam in transport_net_ib_p2p_policy_fakes.cc: the P2P QP policy src/transport/net.cc applies to an
// IB connect handle (owned by net_ib/connect.cc). Separate from transport_net_ib_fakes so a target that
// compiles the real connect.cc can still link that file.

#ifndef RCCL_TEST_HOST_FAKES_TRANSPORT_NET_IB_P2P_POLICY_FAKES_H_
#define RCCL_TEST_HOST_FAKES_TRANSPORT_NET_IB_P2P_POLICY_FAKES_H_

#include <functional>

#include "nccl.h"

// Defaults to fail-loud.
extern std::function<ncclResult_t(void* /*handle*/, int /*isP2p*/)> g_rcclNetP2pPolicy;

void ResetTransportNetIbP2pPolicyFakes();

#endif  // RCCL_TEST_HOST_FAKES_TRANSPORT_NET_IB_P2P_POLICY_FAKES_H_
