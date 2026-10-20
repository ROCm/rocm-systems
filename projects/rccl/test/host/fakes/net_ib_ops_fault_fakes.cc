/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "fakes/net_ib_ops_fault_fakes.h"

#include "fail_loud.h"
#include "net_ib_ops_fault.h"

static ncclResult_t DefaultFaultInstall(struct ibv_context*) {
  FailLoudUnfaked("net_ib_ops_fault_fakes", "ncclIbOpsFaultInstall");
}
static ncclResult_t DefaultFaultRemove(struct ibv_context*) {
  FailLoudUnfaked("net_ib_ops_fault_fakes", "ncclIbOpsFaultRemove");
}

std::function<ncclResult_t(struct ibv_context*)> g_ncclIbOpsFaultInstall = DefaultFaultInstall;
std::function<ncclResult_t(struct ibv_context*)> g_ncclIbOpsFaultRemove = DefaultFaultRemove;

void ResetNetIbOpsFaultFakes() {
  g_ncclIbOpsFaultInstall = DefaultFaultInstall;
  g_ncclIbOpsFaultRemove = DefaultFaultRemove;
}

ncclResult_t ncclIbOpsFaultInstall(struct ibv_context* ctx) { return g_ncclIbOpsFaultInstall(ctx); }
ncclResult_t ncclIbOpsFaultRemove(struct ibv_context* ctx) { return g_ncclIbOpsFaultRemove(ctx); }
