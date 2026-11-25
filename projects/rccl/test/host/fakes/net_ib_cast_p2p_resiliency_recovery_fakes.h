/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the src/transport/net_ib_cast/p2p_resiliency_recovery.cc symbols, for
// binaries that do not compile that TU, plus its IB_RESILIENCY_PORT_RECOVERY param
// through g_loadParam. Every hook fails loud by default.

#ifndef RCCL_TEST_HOST_NET_IB_CAST_P2P_RESILIENCY_RECOVERY_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_P2P_RESILIENCY_RECOVERY_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclIbQp;

extern std::function<ncclResult_t()> g_ibCastPortRecoveryThreadStart;
extern std::function<ncclResult_t()> g_ibCastPortRecoveryThreadStop;
extern std::function<ncclResult_t(struct ncclIbQp* /*qp*/, int /*portNum*/)> g_ibCastPortRecoveryQpUdToRts;

void ResetNetIbCastP2pResiliencyRecoveryFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_P2P_RESILIENCY_RECOVERY_FAKES_H_
