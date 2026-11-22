/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_MSEG_RECOVERY_FAKES_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_MSEG_RECOVERY_FAKES_H_

#include <functional>

#include "nccl.h"
#include "transport/net_ib_multiseg/p2p_resiliency_recovery_mseg.h"

extern std::function<ncclResult_t()> g_ncclIbPortRecoveryThreadStart;
extern std::function<ncclResult_t()> g_ncclIbPortRecoveryThreadStop;
extern std::function<ncclResult_t(struct IbMsegResiliency*)> g_ncclIbPortRecoveryInit;
extern std::function<ncclResult_t(struct IbMsegResiliency*)> g_ncclIbPortRecoveryClose;
extern std::function<ncclResult_t(struct IbMsegResiliency*, int, struct IbMsegDev*)>
    g_ncclIbPortRecoveryDevInit;
extern std::function<ncclResult_t(struct IbMsegResiliency*, int)> g_ncclIbPortRecoveryDevDestroy;
extern std::function<ncclResult_t(struct IbMsegResiliency*, struct IbMsegQpInfo*, int)>
    g_ncclIbPortRecoverySenderQpsCreate;
extern std::function<ncclResult_t(struct IbMsegResiliency*, struct IbMsegConnectionMetadata*, int)>
    g_ncclIbPortRecoverySenderQpsToRts;
extern std::function<ncclResult_t(struct IbMsegResiliency*, struct IbMsegConnectionMetadata*,
                                  struct IbMsegQpInfo*, int)>
    g_ncclIbPortRecoveryReceiverQpsCreateToRts;
extern std::function<ncclResult_t(struct IbMsegResiliency*, int)> g_ncclIbPortRecoveryQpsDestroy;
extern std::function<ncclResult_t(struct IbMsegResiliency*, int)> g_ncclIbPortRecoveryHandleFailure;
extern std::function<ncclResult_t(struct IbMsegResiliency*, int, bool*)>
    g_ncclIbPortRecoveryQpsReconfigure;

void ResetTransportNetIbRecoveryFakes();

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_MSEG_RECOVERY_FAKES_H_
