/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for src/transport/net_ib_cast/multiplane.cc. Every default
// fails with ncclInternalError, so no un-hooked path sees a VIP-to-PIP mapping.

#ifndef RCCL_TEST_HOST_NET_IB_CAST_MULTIPLANE_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_MULTIPLANE_FAKES_H_

#include <functional>

#include "nccl.h"

union ibv_gid;

extern std::function<ncclResult_t()> g_ibCastMultiplaneLoad;
extern std::function<ncclResult_t(bool* /*enabled*/)> g_ibCastMultiplaneEnabled;
extern std::function<ncclResult_t(const union ibv_gid* /*vipGid*/, union ibv_gid* /*pipGids*/, int* /*nPips*/)>
    g_ibCastMultiplaneGetPipGids;

void ResetNetIbCastMultiplaneFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_MULTIPLANE_FAKES_H_
