/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/transport/net_ib_cast/scheduler.cc fakes. See
// net_ib_cast_scheduler_fakes.h.

#include "net_ib_cast_scheduler_fakes.h"

#include "common_cast.h"
#include "fail_loud.h"
#include "signature-drift.h"

static bool DefaultRcclUseIbCastQpSched() { FailLoudUnfaked("net_ib_cast_scheduler_fakes", "rcclUseIbCastQpSched"); }
std::function<bool()> g_rcclUseIbCastQpSched = DefaultRcclUseIbCastQpSched;

static ncclResult_t DefaultIbCastQpSchedInitParms(struct ncclIbQpSchedParms*) {
  FailLoudUnfaked("net_ib_cast_scheduler_fakes", "IbCastQpSchedInitParms");
}
std::function<ncclResult_t(struct ncclIbQpSchedParms*)> g_ibCastQpSchedInitParms = DefaultIbCastQpSchedInitParms;

ASSERT_HOOK_MATCHES_PROD(g_rcclUseIbCastQpSched, rcclUseIbCastQpSched);
ASSERT_HOOK_MATCHES_PROD(g_ibCastQpSchedInitParms, IbCastQpSchedInitParms);

bool micro_rcclUseIbCastQpSched() { return g_rcclUseIbCastQpSched(); }
ncclResult_t micro_IbCastQpSchedInitParms(struct ncclIbQpSchedParms* parms) { return g_ibCastQpSchedInitParms(parms); }

void ResetNetIbCastSchedulerFakes() {
  g_rcclUseIbCastQpSched = DefaultRcclUseIbCastQpSched;
  g_ibCastQpSchedInitParms = DefaultIbCastQpSchedInitParms;
}
