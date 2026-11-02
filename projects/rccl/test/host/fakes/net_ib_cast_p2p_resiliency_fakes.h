/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for src/transport/net_ib_cast/p2p_resiliency.cc, plus its
// params through g_loadParam. Every hook fails loud by default: production
// only reaches these with a non-NULL resiliency context, which a test must
// set up on purpose.

#ifndef RCCL_TEST_HOST_NET_IB_CAST_P2P_RESILIENCY_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_P2P_RESILIENCY_FAKES_H_

#include <functional>

#include "p2p_resiliency_cast.h"

extern std::function<ncclResult_t(struct ncclIbRequest* /*request*/, bool* /*isComplete*/)>
    g_ibCastResiliencyRequestIsComplete;
extern std::function<ncclResult_t(struct ncclIbResiliency* /*resCtx*/, struct ibv_wc* /*wc*/, int /*devIndex*/)>
    g_ibCastResiliencyHandleCompletionError;
extern std::function<ncclResult_t(struct ncclIbResiliency* /*resCtx*/)>
    g_ibCastResiliencyProgress;
extern std::function<ncclResult_t(struct ncclIbNetCommBase* /*baseComm*/, struct ncclIbResiliency** /*resCtx*/)>
    g_ibCastResiliencyInit;
extern std::function<ncclResult_t(struct ncclIbResiliency** /*resCtx*/)>
    g_ibCastResiliencyDestroy;
extern std::function<ncclResult_t(struct ncclIbResiliency* /*resCtx*/, uint /*devIndex*/, ncclIbDev* /*ibDev*/)>
    g_ibCastResiliencyDevInit;
extern std::function<ncclResult_t(struct ncclIbResiliency* /*resCtx*/, uint /*devIndex*/)>
    g_ibCastResiliencyDevDestroy;
extern std::function<ncclResult_t(struct ncclIbResiliency* /*resCtx*/, uint /*devIndex*/, int* /*cqSize*/)>
    g_ibCastResiliencyDataCqSizeGet;
extern std::function<ncclResult_t(struct ncclIbResiliency* /*resCtx*/, uint /*devIndex*/, uint32_t* /*rqSize*/)>
    g_ibCastResiliencyDataRqSizeGet;
extern std::function<ncclResult_t(struct ncclIbResiliency* /*resCtx*/, int /*nLocalDevs*/, int /*nRemDevs*/)>
    g_ibCastResiliencyDeviceNumSet;
extern std::function<ncclResult_t(struct ncclIbResiliency* /*resCtx*/,
                                  struct ncclIbResiliencyInfo* /*localResiliencyInfo*/)>
    g_ibCastResiliencySenderCreateQps;
extern std::function<ncclResult_t(struct ncclIbResiliency* /*resCtx*/, struct ncclIbConnectionMetadata* /*remInfo*/)>
    g_ibCastResiliencySenderQpsToRts;
extern std::function<ncclResult_t(struct ncclIbResiliency* /*resCtx*/, struct ncclIbConnectionMetadata* /*remInfo*/,
                                  struct ncclIbResiliencyInfo* /*localResiliencyInfo*/)>
    g_ibCastResiliencyReceiverQpsCreateToRts;
extern std::function<ncclResult_t(struct ncclIbResiliency* /*resCtx*/, int /*devIndex*/, bool* /*success*/)>
    g_ibCastResiliencyQpsReconfigure;
extern std::function<ncclResult_t(struct ncclIbResiliency* /*resCtx*/, struct ncclIbQp* /*qp*/,
                                  struct ncclIbNetCommDevBase* /*devBase*/, int /*devIndex*/, bool* /*success*/)>
    g_ibCastResiliencyQpReconfigure;
extern std::function<ncclResult_t(struct ncclIbResiliency* /*resCtx*/)>
    g_ibCastResiliencyClose;
extern std::function<ncclResult_t(struct ncclIbResiliency* /*resCtx*/, uint32_t /*cmplsRecordsRkey*/,
                                  uint64_t /*cmplsRecordsAddr*/, uint /*devIndex*/)>
    g_ibCastResiliencyRemoteCompletionRecordsSet;

void ResetNetIbCastP2pResiliencyFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_P2P_RESILIENCY_FAKES_H_
