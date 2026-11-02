/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the src/transport/net_ib_cast/scheduler.cc symbols, for binaries that
// do not compile scheduler.cc. castGlobalQpSchedParms starts zeroed (scheduler
// off), as in production before IbCastQpSchedInitParms runs; both hooks fail loud
// by default.

#ifndef RCCL_TEST_HOST_NET_IB_CAST_SCHEDULER_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_SCHEDULER_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclIbQpSchedParms;

extern std::function<bool()> g_rcclUseIbCastQpSched;
extern std::function<ncclResult_t(struct ncclIbQpSchedParms* /*parms*/)> g_ibCastQpSchedInitParms;

void ResetNetIbCastSchedulerFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_SCHEDULER_FAKES_H_
