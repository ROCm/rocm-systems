/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/gdr_probe.cc fakes. See gdr_probe_fakes.h.

#include "gdr_probe_fakes.h"

#include "fail_loud.h"
#include "gdr_peermem.h"
#include "signature-drift.h"

static int DefaultNcclIbProbeGdrSupport(struct ibv_context*, int) {
  FailLoudUnfaked("gdr_probe_fakes", "ncclIbProbeGdrSupport");
}
std::function<int(struct ibv_context*, int)> g_ncclIbProbeGdrSupport = DefaultNcclIbProbeGdrSupport;

ASSERT_HOOK_MATCHES_PROD(g_ncclIbProbeGdrSupport, ncclIbProbeGdrSupport);

int ncclIbProbeGdrSupport(struct ibv_context* context, int relaxedOrderingEnabled) {
  return g_ncclIbProbeGdrSupport(context, relaxedOrderingEnabled);
}

void ResetGdrProbeFakes() { g_ncclIbProbeGdrSupport = DefaultNcclIbProbeGdrSupport; }
