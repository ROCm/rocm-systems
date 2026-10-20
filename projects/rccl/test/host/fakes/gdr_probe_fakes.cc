/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gdr_probe_fakes.h"

#include "fail_loud.h"
#include "gdr_peermem.h"
#include "signature-drift.h"

static int DefaultIbProbeGdrSupport(struct ibv_context*, int) {
  FailLoudUnfaked("gdr_probe_fakes", "ncclIbProbeGdrSupport");
}
std::function<int(struct ibv_context*, int)> g_ncclIbProbeGdrSupport = DefaultIbProbeGdrSupport;

ASSERT_HOOK_MATCHES_PROD(g_ncclIbProbeGdrSupport, ncclIbProbeGdrSupport);
#undef ASSERT_HOOK_MATCHES_PROD

int ncclIbProbeGdrSupport(struct ibv_context* context, int relaxedOrdering) {
  return g_ncclIbProbeGdrSupport(context, relaxedOrdering);
}

void ResetGdrProbeFakes() { g_ncclIbProbeGdrSupport = DefaultIbProbeGdrSupport; }
