/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Link seams for ras/diagnostics_checks_common.cc's net-device path, for every binary that compiles it:
// graph/topo.cc's ncclTopoGetNetPropertiesSnapshot and transport/net_ib/'s IB device table and plugin.

#ifndef RCCL_TEST_HOST_FAKES_RAS_DIAG_NET_FAKES_H_
#define RCCL_TEST_HOST_FAKES_RAS_DIAG_NET_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclComm;
struct ncclTopoNetPropertiesSnapshot;

// Fails loud by default.
extern std::function<ncclResult_t(struct ncclComm*, bool, struct ncclTopoNetPropertiesSnapshot**, int*)>
    g_topoGetNetPropertiesSnapshot;

// Restores the hook and clears the IB device table (ncclNIbDevs and every device's currSpeed).
void ResetRasDiagNetFakes();

#endif  // RCCL_TEST_HOST_FAKES_RAS_DIAG_NET_FAKES_H_
