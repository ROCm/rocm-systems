/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/transport/net_ib_cast/p2p_resiliency_recovery.cc
// fakes. See net_ib_cast_p2p_resiliency_recovery_fakes.h.

#include "net_ib_cast_p2p_resiliency_recovery_fakes.h"

#include "fail_loud.h"
#include "nccl_fakes.h"
#include "param_redirect.h"
#include "p2p_resiliency_recovery_cast.h"
#include "signature-drift.h"

// Same line as p2p_resiliency_recovery.cc; test/host/CMakeLists.txt pins both copies.
NCCL_PARAM(IbCastResiliencyPortRecovery, "IB_RESILIENCY_PORT_RECOVERY", 0);

static ncclResult_t DefaultIbCastPortRecoveryThreadStart() {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_recovery_fakes", "IbCastPortRecoveryThreadStart");
}
std::function<ncclResult_t()> g_ibCastPortRecoveryThreadStart = DefaultIbCastPortRecoveryThreadStart;

static ncclResult_t DefaultIbCastPortRecoveryThreadStop() {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_recovery_fakes", "IbCastPortRecoveryThreadStop");
}
std::function<ncclResult_t()> g_ibCastPortRecoveryThreadStop = DefaultIbCastPortRecoveryThreadStop;

static ncclResult_t DefaultIbCastPortRecoveryQpUdToRts(struct ncclIbQp*, int) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_recovery_fakes", "IbCastPortRecoveryQpUdToRts");
}
std::function<ncclResult_t(struct ncclIbQp*, int)> g_ibCastPortRecoveryQpUdToRts = DefaultIbCastPortRecoveryQpUdToRts;

ASSERT_HOOK_MATCHES_PROD(g_ibCastPortRecoveryThreadStart, IbCastPortRecoveryThreadStart);
ASSERT_HOOK_MATCHES_PROD(g_ibCastPortRecoveryThreadStop, IbCastPortRecoveryThreadStop);
ASSERT_HOOK_MATCHES_PROD(g_ibCastPortRecoveryQpUdToRts, IbCastPortRecoveryQpUdToRts);

ncclResult_t IbCastPortRecoveryThreadStart() { return g_ibCastPortRecoveryThreadStart(); }
ncclResult_t IbCastPortRecoveryThreadStop() { return g_ibCastPortRecoveryThreadStop(); }
ncclResult_t IbCastPortRecoveryQpUdToRts(struct ncclIbQp* qp, int portNum) {
  return g_ibCastPortRecoveryQpUdToRts(qp, portNum);
}

void ResetNetIbCastP2pResiliencyRecoveryFakes() {
  g_ibCastPortRecoveryThreadStart = DefaultIbCastPortRecoveryThreadStart;
  g_ibCastPortRecoveryThreadStop = DefaultIbCastPortRecoveryThreadStop;
  g_ibCastPortRecoveryQpUdToRts = DefaultIbCastPortRecoveryQpUdToRts;
}
