/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_MSEG_RESILIENCY_FAKES_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_MSEG_RESILIENCY_FAKES_H_

#include <functional>

#include "transport/net_ib_multiseg/p2p_resiliency_mseg.h"

extern std::function<ncclResult_t(struct IbMsegNetCommBase*, struct IbMsegResiliency**)>
    g_ncclIbResiliencyInit;
extern std::function<ncclResult_t(struct IbMsegRequest*, bool*)> g_ncclIbResiliencyRequestIsComplete;
extern std::function<ncclResult_t(struct IbMsegResiliency*)> g_ncclIbResiliencyProgress;
extern std::function<ncclResult_t(struct IbMsegResiliency*, struct ibv_wc*, int)>
    g_ncclIbResiliencyHandleCompletionError;
extern std::function<ncclResult_t(struct IbMsegResiliency**)> g_ncclIbResiliencyDestroy;
extern std::function<ncclResult_t(struct IbMsegResiliency*, uint, struct IbMsegDev*)>
    g_ncclIbResiliencyDevInit;
extern std::function<ncclResult_t(struct IbMsegResiliency*, uint)> g_ncclIbResiliencyDevDestroy;
extern std::function<ncclResult_t(struct IbMsegResiliency*, uint, uint32_t*)>
    g_ncclIbResiliencyDataRqSizeGet;
extern std::function<ncclResult_t(struct IbMsegResiliency*, uint, int*)>
    g_ncclIbResiliencyDataCqSizeGet;
extern std::function<ncclResult_t(struct IbMsegResiliency*, int, int)> g_ncclIbResiliencyDeviceNumSet;
extern std::function<ncclResult_t(struct IbMsegResiliency*, struct IbMsegResiliencyInfo*)>
    g_ncclIbResiliencySenderCreateQps;
extern std::function<ncclResult_t(struct IbMsegResiliency*, struct IbMsegConnectionMetadata*)>
    g_ncclIbResiliencySenderQpsToRts;
extern std::function<ncclResult_t(struct IbMsegResiliency*, struct IbMsegConnectionMetadata*,
                                  struct IbMsegResiliencyInfo*)>
    g_ncclIbResiliencyReceiverQpsCreateToRts;
extern std::function<ncclResult_t(struct IbMsegResiliency*)> g_ncclIbResiliencyClose;
extern std::function<ncclResult_t(struct IbMsegResiliency*, uint32_t, uint64_t, uint)>
    g_ncclIbResiliencyRemoteCompletionRecordsSet;
extern std::function<ncclResult_t(struct IbMsegResiliency*, struct IbMsegQp*,
                                  struct IbMsegNetCommDevBase*, int, bool*)>
    g_ncclIbResiliencyQpReconfigure;
extern std::function<ncclResult_t(struct IbMsegResiliency*, int, bool*)>
    g_ncclIbResiliencyQpsReconfigure;

void ResetTransportNetIbResiliencyFakes();

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_MSEG_RESILIENCY_FAKES_H_
