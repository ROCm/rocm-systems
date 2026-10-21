/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams in transport_net_ib_fakes.cc for src/transport/net_ib/ and net_ib_cast/.

#ifndef RCCL_TEST_HOST_FAKES_TRANSPORT_NET_IB_FAKES_H_
#define RCCL_TEST_HOST_FAKES_TRANSPORT_NET_IB_FAKES_H_

#include <functional>

#include "nccl.h"
#include "transport/net_ib/common.h"

// The P2P QP policies a transport applies to an IB (or IB-cast) connect handle. Default to fail-loud.
extern std::function<ncclResult_t(void* /*handle*/, int /*isP2p*/)> g_rcclNetP2pPolicy;
extern std::function<ncclResult_t(void* /*handle*/, int /*isP2p*/)> g_rcclCastNetP2pPolicy;
extern std::function<ncclResult_t(struct ibv_context*, uint8_t, struct ibv_port_attr*, struct ncclIbGidInfo*)>
    g_ncclIbGidInfoQuery;

void ResetTransportNetIbFakes();

#endif  // RCCL_TEST_HOST_FAKES_TRANSPORT_NET_IB_FAKES_H_
