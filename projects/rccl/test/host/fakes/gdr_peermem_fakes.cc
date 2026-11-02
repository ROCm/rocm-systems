/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/gdr_peermem.cc fakes. See gdr_peermem_fakes.h.

#include "gdr_peermem_fakes.h"

#include "fail_loud.h"
#include "gdr_peermem.h"
#include "signature-drift.h"

static int DefaultNcclIbScanDefaultPeerMemClients() {
  FailLoudUnfaked("gdr_peermem_fakes", "ncclIbScanDefaultPeerMemClients");
}
std::function<int()> g_ncclIbScanDefaultPeerMemClients = DefaultNcclIbScanDefaultPeerMemClients;

ASSERT_HOOK_MATCHES_PROD(g_ncclIbScanDefaultPeerMemClients, ncclIbScanDefaultPeerMemClients);

int ncclIbScanDefaultPeerMemClients(void) { return g_ncclIbScanDefaultPeerMemClients(); }

void ResetGdrPeermemFakes() { g_ncclIbScanDefaultPeerMemClients = DefaultNcclIbScanDefaultPeerMemClients; }
