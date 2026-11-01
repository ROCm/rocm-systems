/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_RESILIENCY_FAKES_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_RESILIENCY_FAKES_H_

#include <functional>

#include "transport/net_ib/p2p_resiliency.h"

extern std::function<ncclResult_t(struct ncclIbNetCommBase*, struct ncclIbResiliency**)>
    g_ncclIbResiliencyInit;

void ResetTransportNetIbResiliencyFakes();

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_RESILIENCY_FAKES_H_
