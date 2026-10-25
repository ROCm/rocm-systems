/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/transport/net_ib_cast/connect.cc fakes. See net_ib_cast_connect_fakes.h.

#include "net_ib_cast_connect_fakes.h"

#include "connect_cast.h"
#include "fail_loud.h"
#include "fakes/nccl_fakes.h"
#include "param_redirect.h"
#include "signature-drift.h"

// Same lines as connect.cc; test/host/CMakeLists.txt pins both copies.
NCCL_PARAM(IbCastTimeout, "IB_TIMEOUT", 20);
NCCL_PARAM(IbCastRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(IbCastPkey, "IB_PKEY", 0);
NCCL_PARAM(IbCastQpsPerConn, "IB_QPS_PER_CONNECTION", 2);
NCCL_PARAM(IbCastUseInline, "IB_USE_INLINE", 0);
NCCL_PARAM(IbCastGdrFlushDisable, "GDR_FLUSH_DISABLE", 1);
NCCL_PARAM(IbCastSl, "IB_SL", -1);
NCCL_PARAM(IbCastTc, "IB_TC", -1);
RCCL_PARAM(IbCastUDMAPolicy, "IB_UDMA_POLICY", 0);

static ncclResult_t DefaultIbCastQpCreate(struct ncclIbQp*, struct ncclIbQpCreateAttr*) {
  FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastQpCreate");
}
static void DefaultIbCastBuildDataQpCreateAttr(struct ncclIbNetCommBase*, int, struct ncclIbQpCreateAttr*) {
  FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastBuildDataQpCreateAttr");
}
static ncclResult_t DefaultIbCastQpInit(struct ncclIbQp*) {
  FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastQpInit");
}
static ncclResult_t DefaultIbCastQpRtr(struct ncclIbQp*) { FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastQpRtr"); }
static ncclResult_t DefaultIbCastQpRts(struct ncclIbQp*) { FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastQpRts"); }
static ncclResult_t DefaultIbCastQpReset(struct ncclIbQp*) {
  FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastQpReset");
}
static ncclResult_t DefaultIbCastQpError(struct ncclIbQp*) {
  FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastQpError");
}
static ncclResult_t DefaultIbCastPostReceiveWorkRequestsOnQp(struct ncclIbRecvComm*, struct ncclIbQp*) {
  FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastPostReceiveWorkRequestsOnQp");
}
static bool DefaultIbCastByOrderRequested() { return false; }
static ncclResult_t DefaultIbCastListen(void*, int, void*, void**) {
  FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastListen");
}
static ncclResult_t DefaultIbCastConnect(void*, int, void*, void**, ncclNetDeviceHandle_t**) {
  FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastConnect");
}
static ncclResult_t DefaultIbCastConnectImpl(void*, int, void*, void**, ncclNetDeviceHandle_t**, int) {
  FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastConnectImpl");
}
static ncclResult_t DefaultIbCastAccept(void*, void**, ncclNetDeviceHandle_t**) {
  FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastAccept");
}
static ncclResult_t DefaultIbCastCloseSend(void*) { FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastCloseSend"); }
static ncclResult_t DefaultIbCastCloseRecv(void*) { FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastCloseRecv"); }
static ncclResult_t DefaultIbCastCloseListen(void*) {
  FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastCloseListen");
}
static ncclResult_t DefaultIbCastGetGidIndex(struct ibv_context*, uint8_t, struct ibv_port_attr*, int*) {
  FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastGetGidIndex");
}
static ncclResult_t DefaultIbCastGidInfoQuery(struct ibv_context*, uint8_t, struct ibv_port_attr*,
                                              struct ncclIbGidInfo*) {
  FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastGidInfoQuery");
}
static void DefaultIbCastSetTrafficClass(void*, int) {
  FailLoudUnfaked("net_ib_cast_connect_fakes", "IbCastSetTrafficClass");
}

std::function<ncclResult_t(struct ncclIbQp*, struct ncclIbQpCreateAttr*)> g_ibCastQpCreate = DefaultIbCastQpCreate;
std::function<void(struct ncclIbNetCommBase*, int, struct ncclIbQpCreateAttr*)> g_ibCastBuildDataQpCreateAttr =
    DefaultIbCastBuildDataQpCreateAttr;
std::function<ncclResult_t(struct ncclIbQp*)> g_ibCastQpInit = DefaultIbCastQpInit;
std::function<ncclResult_t(struct ncclIbQp*)> g_ibCastQpRtr = DefaultIbCastQpRtr;
std::function<ncclResult_t(struct ncclIbQp*)> g_ibCastQpRts = DefaultIbCastQpRts;
std::function<ncclResult_t(struct ncclIbQp*)> g_ibCastQpReset = DefaultIbCastQpReset;
std::function<ncclResult_t(struct ncclIbQp*)> g_ibCastQpError = DefaultIbCastQpError;
std::function<ncclResult_t(struct ncclIbRecvComm*, struct ncclIbQp*)> g_ibCastPostReceiveWorkRequestsOnQp =
    DefaultIbCastPostReceiveWorkRequestsOnQp;
std::function<bool()> g_ibCastByOrderRequested = DefaultIbCastByOrderRequested;
std::function<ncclResult_t(void*, int, void*, void**)> g_ibCastListen = DefaultIbCastListen;
std::function<ncclResult_t(void*, int, void*, void**, ncclNetDeviceHandle_t**)> g_ibCastConnect = DefaultIbCastConnect;
std::function<ncclResult_t(void*, int, void*, void**, ncclNetDeviceHandle_t**, int)> g_ibCastConnectImpl =
    DefaultIbCastConnectImpl;
std::function<ncclResult_t(void*, void**, ncclNetDeviceHandle_t**)> g_ibCastAccept = DefaultIbCastAccept;
std::function<ncclResult_t(void*)> g_ibCastCloseSend = DefaultIbCastCloseSend;
std::function<ncclResult_t(void*)> g_ibCastCloseRecv = DefaultIbCastCloseRecv;
std::function<ncclResult_t(void*)> g_ibCastCloseListen = DefaultIbCastCloseListen;
std::function<ncclResult_t(struct ibv_context*, uint8_t, struct ibv_port_attr*, int*)> g_ibCastGetGidIndex =
    DefaultIbCastGetGidIndex;
std::function<ncclResult_t(struct ibv_context*, uint8_t, struct ibv_port_attr*, struct ncclIbGidInfo*)>
    g_ibCastGidInfoQuery = DefaultIbCastGidInfoQuery;
std::function<void(void*, int)> g_ibCastSetTrafficClass = DefaultIbCastSetTrafficClass;
int g_ibCastReportMatchingSchemeCalls = 0;

ASSERT_HOOK_MATCHES_PROD(g_ibCastQpCreate, IbCastQpCreate);
ASSERT_HOOK_MATCHES_PROD(g_ibCastBuildDataQpCreateAttr, IbCastBuildDataQpCreateAttr);
ASSERT_HOOK_MATCHES_PROD(g_ibCastQpInit, IbCastQpInit);
ASSERT_HOOK_MATCHES_PROD(g_ibCastQpRtr, IbCastQpRtr);
ASSERT_HOOK_MATCHES_PROD(g_ibCastQpRts, IbCastQpRts);
ASSERT_HOOK_MATCHES_PROD(g_ibCastQpReset, IbCastQpReset);
ASSERT_HOOK_MATCHES_PROD(g_ibCastQpError, IbCastQpError);
ASSERT_HOOK_MATCHES_PROD(g_ibCastPostReceiveWorkRequestsOnQp, IbCastPostReceiveWorkRequestsOnQp);
ASSERT_HOOK_MATCHES_PROD(g_ibCastByOrderRequested, IbCastByOrderRequested);
ASSERT_HOOK_MATCHES_PROD(g_ibCastListen, IbCastListen);
ASSERT_HOOK_MATCHES_PROD(g_ibCastConnect, IbCastConnect);
ASSERT_HOOK_MATCHES_PROD(g_ibCastConnectImpl, IbCastConnectImpl);
ASSERT_HOOK_MATCHES_PROD(g_ibCastAccept, IbCastAccept);
ASSERT_HOOK_MATCHES_PROD(g_ibCastCloseSend, IbCastCloseSend);
ASSERT_HOOK_MATCHES_PROD(g_ibCastCloseRecv, IbCastCloseRecv);
ASSERT_HOOK_MATCHES_PROD(g_ibCastCloseListen, IbCastCloseListen);
ASSERT_HOOK_MATCHES_PROD(g_ibCastGetGidIndex, IbCastGetGidIndex);
ASSERT_HOOK_MATCHES_PROD(g_ibCastGidInfoQuery, IbCastGidInfoQuery);
ASSERT_HOOK_MATCHES_PROD(g_ibCastSetTrafficClass, IbCastSetTrafficClass);

ncclResult_t IbCastQpCreate(struct ncclIbQp* qp, struct ncclIbQpCreateAttr* attr) { return g_ibCastQpCreate(qp, attr); }
void IbCastBuildDataQpCreateAttr(struct ncclIbNetCommBase* base, int devIndex, struct ncclIbQpCreateAttr* out) {
  g_ibCastBuildDataQpCreateAttr(base, devIndex, out);
}
ncclResult_t IbCastQpInit(struct ncclIbQp* qp) { return g_ibCastQpInit(qp); }
ncclResult_t IbCastQpRtr(struct ncclIbQp* qp) { return g_ibCastQpRtr(qp); }
ncclResult_t IbCastQpRts(struct ncclIbQp* qp) { return g_ibCastQpRts(qp); }
ncclResult_t IbCastQpReset(struct ncclIbQp* qp) { return g_ibCastQpReset(qp); }
ncclResult_t IbCastQpError(struct ncclIbQp* qp) { return g_ibCastQpError(qp); }
ncclResult_t IbCastPostReceiveWorkRequestsOnQp(struct ncclIbRecvComm* recvComm, struct ncclIbQp* dataQp) {
  return g_ibCastPostReceiveWorkRequestsOnQp(recvComm, dataQp);
}
bool IbCastByOrderRequested() { return g_ibCastByOrderRequested(); }
ncclResult_t IbCastListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  return g_ibCastListen(ctx, dev, opaqueHandle, listenComm);
}
ncclResult_t IbCastConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm,
                           ncclNetDeviceHandle_t** sendDevComm) {
  return g_ibCastConnect(ctx, dev, opaqueHandle, sendComm, sendDevComm);
}
ncclResult_t IbCastConnectImpl(void* ctx, int dev, void* opaqueHandle, void** sendComm,
                               ncclNetDeviceHandle_t** sendDevComm, int envTrafficClass) {
  return g_ibCastConnectImpl(ctx, dev, opaqueHandle, sendComm, sendDevComm, envTrafficClass);
}
ncclResult_t IbCastAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm) {
  return g_ibCastAccept(listenComm, recvComm, recvDevComm);
}
ncclResult_t IbCastCloseSend(void* sendComm) { return g_ibCastCloseSend(sendComm); }
ncclResult_t IbCastCloseRecv(void* recvComm) { return g_ibCastCloseRecv(recvComm); }
ncclResult_t IbCastCloseListen(void* listenComm) { return g_ibCastCloseListen(listenComm); }
ncclResult_t IbCastGetGidIndex(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                               int* gidIndex) {
  return g_ibCastGetGidIndex(context, portNum, portAttr, gidIndex);
}
ncclResult_t IbCastGidInfoQuery(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                                struct ncclIbGidInfo* gidInfo) {
  return g_ibCastGidInfoQuery(context, portNum, portAttr, gidInfo);
}
void IbCastSetTrafficClass(void* ctx, int trafficClass) { g_ibCastSetTrafficClass(ctx, trafficClass); }
void IbCastReportMatchingScheme() { g_ibCastReportMatchingSchemeCalls++; }

void ResetNetIbCastConnectFakes() {
  g_ibCastQpCreate = DefaultIbCastQpCreate;
  g_ibCastBuildDataQpCreateAttr = DefaultIbCastBuildDataQpCreateAttr;
  g_ibCastQpInit = DefaultIbCastQpInit;
  g_ibCastQpRtr = DefaultIbCastQpRtr;
  g_ibCastQpRts = DefaultIbCastQpRts;
  g_ibCastQpReset = DefaultIbCastQpReset;
  g_ibCastQpError = DefaultIbCastQpError;
  g_ibCastPostReceiveWorkRequestsOnQp = DefaultIbCastPostReceiveWorkRequestsOnQp;
  g_ibCastByOrderRequested = DefaultIbCastByOrderRequested;
  g_ibCastListen = DefaultIbCastListen;
  g_ibCastConnect = DefaultIbCastConnect;
  g_ibCastConnectImpl = DefaultIbCastConnectImpl;
  g_ibCastAccept = DefaultIbCastAccept;
  g_ibCastCloseSend = DefaultIbCastCloseSend;
  g_ibCastCloseRecv = DefaultIbCastCloseRecv;
  g_ibCastCloseListen = DefaultIbCastCloseListen;
  g_ibCastGetGidIndex = DefaultIbCastGetGidIndex;
  g_ibCastGidInfoQuery = DefaultIbCastGidInfoQuery;
  g_ibCastSetTrafficClass = DefaultIbCastSetTrafficClass;
  g_ibCastReportMatchingSchemeCalls = 0;
}
