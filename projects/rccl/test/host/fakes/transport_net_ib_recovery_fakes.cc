/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "transport_net_ib_recovery_fakes.h"

#include "fail_loud.h"
#include "signature-drift.h"
#include "transport/net_ib/p2p_resiliency_recovery.h"

static ncclResult_t DefaultPortRecoveryThreadStart() {
  FailLoudUnfaked("transport_net_ib_recovery_fakes", "ncclIbPortRecoveryThreadStart");
}
static ncclResult_t DefaultPortRecoveryThreadStop() {
  FailLoudUnfaked("transport_net_ib_recovery_fakes", "ncclIbPortRecoveryThreadStop");
}

std::function<ncclResult_t()> g_ncclIbPortRecoveryThreadStart = DefaultPortRecoveryThreadStart;
std::function<ncclResult_t()> g_ncclIbPortRecoveryThreadStop = DefaultPortRecoveryThreadStop;

ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryThreadStart, ncclIbPortRecoveryThreadStart);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryThreadStop, ncclIbPortRecoveryThreadStop);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t ncclIbPortRecoveryThreadStart() { return g_ncclIbPortRecoveryThreadStart(); }
ncclResult_t ncclIbPortRecoveryThreadStop() { return g_ncclIbPortRecoveryThreadStop(); }

void ResetTransportNetIbRecoveryFakes() {
  g_ncclIbPortRecoveryThreadStart = DefaultPortRecoveryThreadStart;
  g_ncclIbPortRecoveryThreadStop = DefaultPortRecoveryThreadStop;
}
