/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/transport/net_ib_cast/multiplane.cc fakes. See
// net_ib_cast_multiplane_fakes.h.

#include "net_ib_cast_multiplane_fakes.h"

#include "multiplane.h"
#include "signature-drift.h"

static ncclResult_t DefaultIbCastMultiplaneLoad() { return ncclInternalError; }
std::function<ncclResult_t()> g_ibCastMultiplaneLoad = DefaultIbCastMultiplaneLoad;

static ncclResult_t DefaultIbCastMultiplaneEnabled(bool*) { return ncclInternalError; }
std::function<ncclResult_t(bool*)> g_ibCastMultiplaneEnabled = DefaultIbCastMultiplaneEnabled;

static ncclResult_t DefaultIbCastMultiplaneGetPipGids(const union ibv_gid*, union ibv_gid*, int*) {
  return ncclInternalError;
}
std::function<ncclResult_t(const union ibv_gid*, union ibv_gid*, int*)> g_ibCastMultiplaneGetPipGids =
    DefaultIbCastMultiplaneGetPipGids;

ASSERT_HOOK_MATCHES_PROD(g_ibCastMultiplaneLoad, IbCastMultiplaneLoad);
ASSERT_HOOK_MATCHES_PROD(g_ibCastMultiplaneEnabled, IbCastMultiplaneEnabled);
ASSERT_HOOK_MATCHES_PROD(g_ibCastMultiplaneGetPipGids, IbCastMultiplaneGetPipGids);

ncclResult_t IbCastMultiplaneLoad(void) { return g_ibCastMultiplaneLoad(); }
ncclResult_t IbCastMultiplaneEnabled(bool* enabled) { return g_ibCastMultiplaneEnabled(enabled); }
ncclResult_t IbCastMultiplaneGetPipGids(const union ibv_gid* vipGid, union ibv_gid* pipGids, int* nPips) {
  return g_ibCastMultiplaneGetPipGids(vipGid, pipGids, nPips);
}

void ResetNetIbCastMultiplaneFakes() {
  g_ibCastMultiplaneLoad = DefaultIbCastMultiplaneLoad;
  g_ibCastMultiplaneEnabled = DefaultIbCastMultiplaneEnabled;
  g_ibCastMultiplaneGetPipGids = DefaultIbCastMultiplaneGetPipGids;
}
