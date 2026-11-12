/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/transport/net_ib_cast/p2p_resiliency.cc fakes. See
// net_ib_cast_p2p_resiliency_fakes.h.

#include "net_ib_cast_p2p_resiliency_fakes.h"

#include "fail_loud.h"
#include "nccl_fakes.h"
#include "signature-drift.h"

int64_t ncclParamIbCastResiliencyPortFailover() { return g_loadParam("IB_RESILIENCY_PORT_FAILOVER", 0); }
int64_t ncclParamIbCastResiliencyPortFailoverMaxAttempts() {
  return g_loadParam("IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS", 1);
}
int64_t ncclParamIbCastResiliencyPortFailoverProbeDelay() {
  return g_loadParam("IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY", 10);
}

static ncclResult_t DefaultIbCastResiliencyRequestIsComplete(struct ncclIbRequest*, bool*) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyRequestIsComplete");
}
std::function<ncclResult_t(struct ncclIbRequest*, bool*)>
    g_ibCastResiliencyRequestIsComplete = DefaultIbCastResiliencyRequestIsComplete;

static ncclResult_t DefaultIbCastResiliencyHandleCompletionError(struct ncclIbResiliency*, struct ibv_wc*, int) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyHandleCompletionError");
}
std::function<ncclResult_t(struct ncclIbResiliency*, struct ibv_wc*, int)>
    g_ibCastResiliencyHandleCompletionError = DefaultIbCastResiliencyHandleCompletionError;

static ncclResult_t DefaultIbCastResiliencyProgress(struct ncclIbResiliency*) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyProgress");
}
std::function<ncclResult_t(struct ncclIbResiliency*)> g_ibCastResiliencyProgress = DefaultIbCastResiliencyProgress;

static ncclResult_t DefaultIbCastResiliencyInit(struct ncclIbNetCommBase*, struct ncclIbResiliency**) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyInit");
}
std::function<ncclResult_t(struct ncclIbNetCommBase*, struct ncclIbResiliency**)>
    g_ibCastResiliencyInit = DefaultIbCastResiliencyInit;

static ncclResult_t DefaultIbCastResiliencyDestroy(struct ncclIbResiliency**) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyDestroy");
}
std::function<ncclResult_t(struct ncclIbResiliency**)> g_ibCastResiliencyDestroy = DefaultIbCastResiliencyDestroy;

static ncclResult_t DefaultIbCastResiliencyDevInit(struct ncclIbResiliency*, uint, ncclIbDev*) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyDevInit");
}
std::function<ncclResult_t(struct ncclIbResiliency*, uint, ncclIbDev*)>
    g_ibCastResiliencyDevInit = DefaultIbCastResiliencyDevInit;

static ncclResult_t DefaultIbCastResiliencyDevDestroy(struct ncclIbResiliency*, uint) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyDevDestroy");
}
std::function<ncclResult_t(struct ncclIbResiliency*, uint)>
    g_ibCastResiliencyDevDestroy = DefaultIbCastResiliencyDevDestroy;

static ncclResult_t DefaultIbCastResiliencyDataCqSizeGet(struct ncclIbResiliency*, uint, int*) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyDataCqSizeGet");
}
std::function<ncclResult_t(struct ncclIbResiliency*, uint, int*)>
    g_ibCastResiliencyDataCqSizeGet = DefaultIbCastResiliencyDataCqSizeGet;

static ncclResult_t DefaultIbCastResiliencyDataRqSizeGet(struct ncclIbResiliency*, uint, uint32_t*) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyDataRqSizeGet");
}
std::function<ncclResult_t(struct ncclIbResiliency*, uint, uint32_t*)>
    g_ibCastResiliencyDataRqSizeGet = DefaultIbCastResiliencyDataRqSizeGet;

static ncclResult_t DefaultIbCastResiliencyDeviceNumSet(struct ncclIbResiliency*, int, int) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyDeviceNumSet");
}
std::function<ncclResult_t(struct ncclIbResiliency*, int, int)>
    g_ibCastResiliencyDeviceNumSet = DefaultIbCastResiliencyDeviceNumSet;

static ncclResult_t DefaultIbCastResiliencySenderCreateQps(struct ncclIbResiliency*, struct ncclIbResiliencyInfo*) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencySenderCreateQps");
}
std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbResiliencyInfo*)>
    g_ibCastResiliencySenderCreateQps = DefaultIbCastResiliencySenderCreateQps;

static ncclResult_t DefaultIbCastResiliencySenderQpsToRts(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencySenderQpsToRts");
}
std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*)>
    g_ibCastResiliencySenderQpsToRts = DefaultIbCastResiliencySenderQpsToRts;

static ncclResult_t DefaultIbCastResiliencyReceiverQpsCreateToRts(struct ncclIbResiliency*,
                                                                  struct ncclIbConnectionMetadata*,
                                                                  struct ncclIbResiliencyInfo*) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyReceiverQpsCreateToRts");
}
std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*, struct ncclIbResiliencyInfo*)>
    g_ibCastResiliencyReceiverQpsCreateToRts = DefaultIbCastResiliencyReceiverQpsCreateToRts;

static ncclResult_t DefaultIbCastResiliencyQpsReconfigure(struct ncclIbResiliency*, int, bool*) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyQpsReconfigure");
}
std::function<ncclResult_t(struct ncclIbResiliency*, int, bool*)>
    g_ibCastResiliencyQpsReconfigure = DefaultIbCastResiliencyQpsReconfigure;

static ncclResult_t DefaultIbCastResiliencyQpReconfigure(struct ncclIbResiliency*, struct ncclIbQp*,
                                                         struct ncclIbNetCommDevBase*, int, bool*) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyQpReconfigure");
}
std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbQp*, struct ncclIbNetCommDevBase*, int, bool*)>
    g_ibCastResiliencyQpReconfigure = DefaultIbCastResiliencyQpReconfigure;

static ncclResult_t DefaultIbCastResiliencyClose(struct ncclIbResiliency*) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyClose");
}
std::function<ncclResult_t(struct ncclIbResiliency*)> g_ibCastResiliencyClose = DefaultIbCastResiliencyClose;

static ncclResult_t DefaultIbCastResiliencyRemoteCompletionRecordsSet(struct ncclIbResiliency*, uint32_t, uint64_t,
                                                                      uint) {
  FailLoudUnfaked("net_ib_cast_p2p_resiliency_fakes", "IbCastResiliencyRemoteCompletionRecordsSet");
}
std::function<ncclResult_t(struct ncclIbResiliency*, uint32_t, uint64_t, uint)>
    g_ibCastResiliencyRemoteCompletionRecordsSet = DefaultIbCastResiliencyRemoteCompletionRecordsSet;

ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyRequestIsComplete, IbCastResiliencyRequestIsComplete);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyHandleCompletionError, IbCastResiliencyHandleCompletionError);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyProgress, IbCastResiliencyProgress);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyInit, IbCastResiliencyInit);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyDestroy, IbCastResiliencyDestroy);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyDevInit, IbCastResiliencyDevInit);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyDevDestroy, IbCastResiliencyDevDestroy);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyDataCqSizeGet, IbCastResiliencyDataCqSizeGet);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyDataRqSizeGet, IbCastResiliencyDataRqSizeGet);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyDeviceNumSet, IbCastResiliencyDeviceNumSet);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencySenderCreateQps, IbCastResiliencySenderCreateQps);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencySenderQpsToRts, IbCastResiliencySenderQpsToRts);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyReceiverQpsCreateToRts, IbCastResiliencyReceiverQpsCreateToRts);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyQpsReconfigure, IbCastResiliencyQpsReconfigure);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyQpReconfigure, IbCastResiliencyQpReconfigure);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyClose, IbCastResiliencyClose);
ASSERT_HOOK_MATCHES_PROD(g_ibCastResiliencyRemoteCompletionRecordsSet, IbCastResiliencyRemoteCompletionRecordsSet);

ncclResult_t IbCastResiliencyRequestIsComplete(struct ncclIbRequest* request, bool* isComplete) {
  return g_ibCastResiliencyRequestIsComplete(request, isComplete);
}
ncclResult_t IbCastResiliencyHandleCompletionError(struct ncclIbResiliency* resCtx, struct ibv_wc* wc, int devIndex) {
  return g_ibCastResiliencyHandleCompletionError(resCtx, wc, devIndex);
}
ncclResult_t IbCastResiliencyProgress(struct ncclIbResiliency* resCtx) {
  return g_ibCastResiliencyProgress(resCtx);
}
ncclResult_t IbCastResiliencyInit(struct ncclIbNetCommBase* baseComm, struct ncclIbResiliency** resCtx) {
  return g_ibCastResiliencyInit(baseComm, resCtx);
}
ncclResult_t IbCastResiliencyDestroy(struct ncclIbResiliency** resCtx) {
  return g_ibCastResiliencyDestroy(resCtx);
}
ncclResult_t IbCastResiliencyDevInit(struct ncclIbResiliency* resCtx, uint devIndex, ncclIbDev* ibDev) {
  return g_ibCastResiliencyDevInit(resCtx, devIndex, ibDev);
}
ncclResult_t IbCastResiliencyDevDestroy(struct ncclIbResiliency* resCtx, uint devIndex) {
  return g_ibCastResiliencyDevDestroy(resCtx, devIndex);
}
ncclResult_t IbCastResiliencyDataCqSizeGet(struct ncclIbResiliency* resCtx, uint devIndex, int* cqSize) {
  return g_ibCastResiliencyDataCqSizeGet(resCtx, devIndex, cqSize);
}
ncclResult_t IbCastResiliencyDataRqSizeGet(struct ncclIbResiliency* resCtx, uint devIndex, uint32_t* rqSize) {
  return g_ibCastResiliencyDataRqSizeGet(resCtx, devIndex, rqSize);
}
ncclResult_t IbCastResiliencyDeviceNumSet(struct ncclIbResiliency* resCtx, int nLocalDevs, int nRemDevs) {
  return g_ibCastResiliencyDeviceNumSet(resCtx, nLocalDevs, nRemDevs);
}
ncclResult_t IbCastResiliencySenderCreateQps(struct ncclIbResiliency* resCtx,
                                             struct ncclIbResiliencyInfo* localResiliencyInfo) {
  return g_ibCastResiliencySenderCreateQps(resCtx, localResiliencyInfo);
}
ncclResult_t IbCastResiliencySenderQpsToRts(struct ncclIbResiliency* resCtx, struct ncclIbConnectionMetadata* remInfo) {
  return g_ibCastResiliencySenderQpsToRts(resCtx, remInfo);
}
ncclResult_t IbCastResiliencyReceiverQpsCreateToRts(struct ncclIbResiliency* resCtx,
                                                    struct ncclIbConnectionMetadata* remInfo,
                                                    struct ncclIbResiliencyInfo* localResiliencyInfo) {
  return g_ibCastResiliencyReceiverQpsCreateToRts(resCtx, remInfo, localResiliencyInfo);
}
ncclResult_t IbCastResiliencyQpsReconfigure(struct ncclIbResiliency* resCtx, int devIndex, bool* success) {
  return g_ibCastResiliencyQpsReconfigure(resCtx, devIndex, success);
}
ncclResult_t IbCastResiliencyQpReconfigure(struct ncclIbResiliency* resCtx, struct ncclIbQp* qp,
                                           struct ncclIbNetCommDevBase* devBase, int devIndex, bool* success) {
  return g_ibCastResiliencyQpReconfigure(resCtx, qp, devBase, devIndex, success);
}
ncclResult_t IbCastResiliencyClose(struct ncclIbResiliency* resCtx) {
  return g_ibCastResiliencyClose(resCtx);
}
ncclResult_t IbCastResiliencyRemoteCompletionRecordsSet(struct ncclIbResiliency* resCtx, uint32_t cmplsRecordsRkey,
                                                        uint64_t cmplsRecordsAddr, uint devIndex) {
  return g_ibCastResiliencyRemoteCompletionRecordsSet(resCtx, cmplsRecordsRkey, cmplsRecordsAddr, devIndex);
}

void ResetNetIbCastP2pResiliencyFakes() {
  g_ibCastResiliencyRequestIsComplete = DefaultIbCastResiliencyRequestIsComplete;
  g_ibCastResiliencyHandleCompletionError = DefaultIbCastResiliencyHandleCompletionError;
  g_ibCastResiliencyProgress = DefaultIbCastResiliencyProgress;
  g_ibCastResiliencyInit = DefaultIbCastResiliencyInit;
  g_ibCastResiliencyDestroy = DefaultIbCastResiliencyDestroy;
  g_ibCastResiliencyDevInit = DefaultIbCastResiliencyDevInit;
  g_ibCastResiliencyDevDestroy = DefaultIbCastResiliencyDevDestroy;
  g_ibCastResiliencyDataCqSizeGet = DefaultIbCastResiliencyDataCqSizeGet;
  g_ibCastResiliencyDataRqSizeGet = DefaultIbCastResiliencyDataRqSizeGet;
  g_ibCastResiliencyDeviceNumSet = DefaultIbCastResiliencyDeviceNumSet;
  g_ibCastResiliencySenderCreateQps = DefaultIbCastResiliencySenderCreateQps;
  g_ibCastResiliencySenderQpsToRts = DefaultIbCastResiliencySenderQpsToRts;
  g_ibCastResiliencyReceiverQpsCreateToRts = DefaultIbCastResiliencyReceiverQpsCreateToRts;
  g_ibCastResiliencyQpsReconfigure = DefaultIbCastResiliencyQpsReconfigure;
  g_ibCastResiliencyQpReconfigure = DefaultIbCastResiliencyQpReconfigure;
  g_ibCastResiliencyClose = DefaultIbCastResiliencyClose;
  g_ibCastResiliencyRemoteCompletionRecordsSet = DefaultIbCastResiliencyRemoteCompletionRecordsSet;
}
