/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "transport_net_ib_resiliency_fakes.h"

#include "fail_loud.h"
#include "fakes/nccl_fakes.h"
#include "signature-drift.h"

int64_t ncclParamIbResiliencyPortFailover() {
  return g_loadParam("IB_RESILIENCY_PORT_FAILOVER", 0);
}

static ncclResult_t DefaultIbResiliencyInit(struct ncclIbNetCommBase*, struct ncclIbResiliency**) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencyInit");
}
std::function<ncclResult_t(struct ncclIbNetCommBase*, struct ncclIbResiliency**)> g_ncclIbResiliencyInit =
    DefaultIbResiliencyInit;

ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyInit, ncclIbResiliencyInit);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t ncclIbResiliencyInit(struct ncclIbNetCommBase* base, struct ncclIbResiliency** resiliency) {
  return g_ncclIbResiliencyInit(base, resiliency);
}

void ResetTransportNetIbResiliencyFakes() {
  g_ncclIbResiliencyInit = DefaultIbResiliencyInit;
}
