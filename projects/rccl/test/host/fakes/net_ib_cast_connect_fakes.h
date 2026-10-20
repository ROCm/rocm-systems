/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the src/transport/net_ib_cast/connect.cc symbols, for binaries that
// do not compile connect.cc. Every hook fails loud by default except
// IbCastByOrderRequested, which answers false (BY_ID, the default scheme).

#ifndef RCCL_TEST_HOST_NET_IB_CAST_CONNECT_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_CONNECT_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclIbQp;
struct ncclIbQpCreateAttr;
struct ncclIbNetCommBase;
struct ncclIbRecvComm;

extern std::function<ncclResult_t(struct ncclIbQp*, struct ncclIbQpCreateAttr*)> g_ibCastQpCreate;
extern std::function<void(struct ncclIbNetCommBase*, int, struct ncclIbQpCreateAttr*)> g_ibCastBuildDataQpCreateAttr;
extern std::function<ncclResult_t(struct ncclIbQp*)> g_ibCastQpInit;
extern std::function<ncclResult_t(struct ncclIbQp*)> g_ibCastQpRtr;
extern std::function<ncclResult_t(struct ncclIbQp*)> g_ibCastQpRts;
extern std::function<ncclResult_t(struct ncclIbQp*)> g_ibCastQpReset;
extern std::function<ncclResult_t(struct ncclIbQp*)> g_ibCastQpError;
extern std::function<ncclResult_t(struct ncclIbRecvComm*, struct ncclIbQp*)> g_ibCastPostReceiveWorkRequestsOnQp;
extern std::function<bool()> g_ibCastByOrderRequested;

void ResetNetIbCastConnectFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_CONNECT_FAKES_H_
