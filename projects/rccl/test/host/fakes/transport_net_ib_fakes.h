/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for symbols owned by src/transport/net_ib (see transport_net_ib_fakes.cc).

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_FAKES_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_FAKES_H_

#include <functional>
#include <utility>
#include <vector>

#include "nccl.h"
#include "transport/net_ib/common.h"

// ncclIbGdrFlushFlagLocalSupport() returns this.
extern bool g_ibGdrFlushFlagLocalSupport;
// Every ncclIbSetGdrFlushFlag(netContext, enable) call, in order.
extern std::vector<std::pair<void*, int>> g_ibSetGdrFlushFlagCalls;
// ncclIbGidInfoQuery. Fails with ncclSystemError by default.
extern std::function<ncclResult_t(struct ibv_context*, uint8_t, struct ibv_port_attr*, struct ncclIbGidInfo*)>
    g_ncclIbGidInfoQuery;

void ResetTransportNetIbFakes();

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_FAKES_H_
