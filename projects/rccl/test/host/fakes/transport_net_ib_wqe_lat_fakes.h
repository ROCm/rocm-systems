/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Generic host-only seams for src/transport/net_ib/wqe_lat_mon.cc.

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_WQE_LAT_FAKES_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_WQE_LAT_FAKES_H_

#include <functional>

#include "transport/net_ib/wqe_lat_mon.h"

extern std::function<void(struct ncclIbQp*, struct ibv_send_wr*)> g_ncclIbWqeLatMonStampSend;
extern std::function<void(struct ncclIbNetCommBase*, int, struct ibv_wc*, uint64_t*, bool*)>
    g_ncclIbWqeLatHandleCompletion;
extern std::function<void(struct ncclIbNetCommBase*, int)> g_ncclIbWqeLatScanStalls;

void ResetTransportNetIbWqeLatFakes();

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_WQE_LAT_FAKES_H_
