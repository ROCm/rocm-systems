/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for src/transport/net_ib/connect.cc.

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_FAKES_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_FAKES_H_

#include <functional>

#include "transport/net_ib/common.h"

extern std::function<ncclResult_t(struct ibv_context*, uint8_t, struct ibv_port_attr*, struct ncclIbGidInfo*)>
    g_ncclIbGidInfoQuery;

void ResetTransportNetIbFakes();

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_FAKES_H_
