/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gdr_peermem_fakes.h"

#include "fail_loud.h"
#include "gdr_peermem.h"
#include "signature-drift.h"

static int DefaultScanDefaultPeerMemClients() {
  FailLoudUnfaked("gdr_peermem_fakes", "ncclIbScanDefaultPeerMemClients");
}
std::function<int()> g_ncclIbScanDefaultPeerMemClients = DefaultScanDefaultPeerMemClients;

ASSERT_HOOK_MATCHES_PROD(g_ncclIbScanDefaultPeerMemClients, ncclIbScanDefaultPeerMemClients);
#undef ASSERT_HOOK_MATCHES_PROD

int ncclIbScanDefaultPeerMemClients() { return g_ncclIbScanDefaultPeerMemClients(); }

void ResetGdrPeermemFakes() {
  g_ncclIbScanDefaultPeerMemClients = DefaultScanDefaultPeerMemClients;
}
