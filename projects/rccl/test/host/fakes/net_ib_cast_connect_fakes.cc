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

ASSERT_HOOK_MATCHES_PROD(g_ibCastQpCreate, IbCastQpCreate);
ASSERT_HOOK_MATCHES_PROD(g_ibCastBuildDataQpCreateAttr, IbCastBuildDataQpCreateAttr);
ASSERT_HOOK_MATCHES_PROD(g_ibCastQpInit, IbCastQpInit);
ASSERT_HOOK_MATCHES_PROD(g_ibCastQpRtr, IbCastQpRtr);
ASSERT_HOOK_MATCHES_PROD(g_ibCastQpRts, IbCastQpRts);
ASSERT_HOOK_MATCHES_PROD(g_ibCastQpReset, IbCastQpReset);
ASSERT_HOOK_MATCHES_PROD(g_ibCastQpError, IbCastQpError);
ASSERT_HOOK_MATCHES_PROD(g_ibCastPostReceiveWorkRequestsOnQp, IbCastPostReceiveWorkRequestsOnQp);
ASSERT_HOOK_MATCHES_PROD(g_ibCastByOrderRequested, IbCastByOrderRequested);

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
}
