/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the src/transport/net_ib_cast/connect.cc symbols, for binaries that
// do not compile connect.cc. Every hook fails loud by default except
// IbCastByOrderRequested, which answers false (BY_ID, the default scheme).
// IbCastReportMatchingScheme only counts its calls.

#ifndef RCCL_TEST_HOST_NET_IB_CAST_CONNECT_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_CONNECT_FAKES_H_

#include <cstdint>
#include <functional>

#include "nccl.h"
#include "nccl_device/net_device.h"

struct ibv_context;
struct ibv_port_attr;
struct ncclIbGidInfo;
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
extern std::function<ncclResult_t(void*, int, void*, void**)> g_ibCastListen;
extern std::function<ncclResult_t(void*, int, void*, void**, ncclNetDeviceHandle_t**)> g_ibCastConnect;
extern std::function<ncclResult_t(void*, int, void*, void**, ncclNetDeviceHandle_t**, int)> g_ibCastConnectImpl;
extern std::function<ncclResult_t(void*, void**, ncclNetDeviceHandle_t**)> g_ibCastAccept;
extern std::function<ncclResult_t(void*)> g_ibCastCloseSend;
extern std::function<ncclResult_t(void*)> g_ibCastCloseRecv;
extern std::function<ncclResult_t(void*)> g_ibCastCloseListen;
extern std::function<ncclResult_t(struct ibv_context*, uint8_t, struct ibv_port_attr*, int*)> g_ibCastGetGidIndex;
extern std::function<ncclResult_t(struct ibv_context*, uint8_t, struct ibv_port_attr*, struct ncclIbGidInfo*)>
    g_ibCastGidInfoQuery;
extern std::function<void(void*, int)> g_ibCastSetTrafficClass;
extern int g_ibCastReportMatchingSchemeCalls;
// The P2P QP policy net.cc applies to an IB-cast connect handle. Defaults to fail-loud.
extern std::function<ncclResult_t(void* /*handle*/, int /*isP2p*/)> g_rcclCastNetP2pPolicy;

void ResetNetIbCastConnectFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_CONNECT_FAKES_H_
