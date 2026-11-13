/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_RECOVERY_FAKES_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_RECOVERY_FAKES_H_

#include <functional>

#include "nccl.h"
#include "transport/net_ib/p2p_resiliency_recovery.h"

extern std::function<ncclResult_t()> g_ncclIbPortRecoveryThreadStart;
extern std::function<ncclResult_t()> g_ncclIbPortRecoveryThreadStop;
extern std::function<ncclResult_t(struct ncclIbResiliency*)> g_ncclIbPortRecoveryInit;
extern std::function<ncclResult_t(struct ncclIbResiliency*)> g_ncclIbPortRecoveryClose;
extern std::function<ncclResult_t(struct ncclIbResiliency*, int, struct ncclIbDev*)>
    g_ncclIbPortRecoveryDevInit;
extern std::function<ncclResult_t(struct ncclIbResiliency*, int)> g_ncclIbPortRecoveryDevDestroy;
extern std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbQpInfo*, int)>
    g_ncclIbPortRecoverySenderQpsCreate;
extern std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*, int)>
    g_ncclIbPortRecoverySenderQpsToRts;
extern std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*,
                                  struct ncclIbQpInfo*, int)>
    g_ncclIbPortRecoveryReceiverQpsCreateToRts;
extern std::function<ncclResult_t(struct ncclIbResiliency*, int)> g_ncclIbPortRecoveryQpsDestroy;
extern std::function<ncclResult_t(struct ncclIbResiliency*, int)> g_ncclIbPortRecoveryHandleFailure;
extern std::function<ncclResult_t(struct ncclIbResiliency*, int, bool*)>
    g_ncclIbPortRecoveryQpsReconfigure;

void ResetTransportNetIbRecoveryFakes();

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_RECOVERY_FAKES_H_
