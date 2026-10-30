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
extern std::function<ncclResult_t(struct ncclIbRequest*, bool*)> g_ncclIbResiliencyRequestIsComplete;
extern std::function<ncclResult_t(struct ncclIbResiliency*)> g_ncclIbResiliencyProgress;
extern std::function<ncclResult_t(struct ncclIbResiliency*, struct ibv_wc*, int)>
    g_ncclIbResiliencyHandleCompletionError;
extern std::function<ncclResult_t(struct ncclIbResiliency**)> g_ncclIbResiliencyDestroy;
extern std::function<ncclResult_t(struct ncclIbResiliency*, uint, struct ncclIbDev*)>
    g_ncclIbResiliencyDevInit;
extern std::function<ncclResult_t(struct ncclIbResiliency*, uint)> g_ncclIbResiliencyDevDestroy;
extern std::function<ncclResult_t(struct ncclIbResiliency*, uint, uint32_t*)>
    g_ncclIbResiliencyDataRqSizeGet;
extern std::function<ncclResult_t(struct ncclIbResiliency*, uint, int*)>
    g_ncclIbResiliencyDataCqSizeGet;
extern std::function<ncclResult_t(struct ncclIbResiliency*, int, int)> g_ncclIbResiliencyDeviceNumSet;
extern std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbResiliencyInfo*)>
    g_ncclIbResiliencySenderCreateQps;
extern std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*)>
    g_ncclIbResiliencySenderQpsToRts;
extern std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*,
                                  struct ncclIbResiliencyInfo*)>
    g_ncclIbResiliencyReceiverQpsCreateToRts;
extern std::function<ncclResult_t(struct ncclIbResiliency*)> g_ncclIbResiliencyClose;
extern std::function<ncclResult_t(struct ncclIbResiliency*, uint32_t, uint64_t, uint)>
    g_ncclIbResiliencyRemoteCompletionRecordsSet;

void ResetTransportNetIbResiliencyFakes();

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_RESILIENCY_FAKES_H_
