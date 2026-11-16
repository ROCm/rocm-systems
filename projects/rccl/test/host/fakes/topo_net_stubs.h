/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams in topo_stubs.cc for the net-device lookups src/graph/paths.cc and search.cc
// provide. Kept out of topo_stubs.h: their enum types need graph.h, which must not
// reach init-test.cc ahead of its macro shims. ResetTopoStubs() restores them.

#ifndef RCCL_TEST_HOST_TOPO_NET_STUBS_H_
#define RCCL_TEST_HOST_TOPO_NET_STUBS_H_

#include <cstdint>
#include <functional>

#include "graph.h"
#include "nccl.h"

// transport/net.cc picks its NIC, GDR mode and flush rule through these. Default to fail-loud.
extern std::function<ncclResult_t(struct ncclComm*, int, struct ncclTopoGraph*, int, int, int64_t*, int*, int*)>
    g_ncclTopoGetNetDev;
extern std::function<ncclResult_t(struct ncclTopoSystem*, int, struct ncclTopoGraph*, int, int, int64_t*, int*)>
    g_ncclTopoGetIntraNetDev;
extern std::function<ncclResult_t(struct ncclTopoSystem*, int, int64_t, int, enum ncclTopoGdrMode*)>
    g_ncclTopoCheckGdr;
extern std::function<ncclResult_t(struct ncclComm*, int64_t, int, int, bool, enum ncclTopoFlushType*)>
    g_ncclTopoNeedFlush;

#endif  // RCCL_TEST_HOST_TOPO_NET_STUBS_H_
