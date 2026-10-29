/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud hooks for the src/transport/net_ib_cast/scheduler.cc entry points init.cc calls. The binary also compiles
// the real scheduler.cc, so net_ib_cast_scheduler_seam.h renames those calls to the micro_* trampolines below.

#ifndef RCCL_TEST_HOST_NET_IB_CAST_SCHEDULER_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_SCHEDULER_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclIbQpSchedParms;

extern std::function<bool()> g_rcclUseIbCastQpSched;
extern std::function<ncclResult_t(struct ncclIbQpSchedParms* /*parms*/)> g_ibCastQpSchedInitParms;

bool micro_rcclUseIbCastQpSched();
ncclResult_t micro_IbCastQpSchedInitParms(struct ncclIbQpSchedParms* parms);

void ResetNetIbCastSchedulerFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_SCHEDULER_FAKES_H_
