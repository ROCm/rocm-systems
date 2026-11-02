/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "transport_net_ib_resiliency_fakes.h"

#include "fail_loud.h"
#include "fakes/nccl_fakes.h"
#include "signature-drift.h"

int64_t ncclParamIbResiliencyPortFailover() {
  return g_loadParam("IB_RESILIENCY_PORT_FAILOVER", 0);
}

static ncclResult_t DefaultIbResiliencyInit(struct ncclIbNetCommBase*, struct ncclIbResiliency**) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencyInit");
}
std::function<ncclResult_t(struct ncclIbNetCommBase*, struct ncclIbResiliency**)> g_ncclIbResiliencyInit =
    DefaultIbResiliencyInit;

static ncclResult_t DefaultIbResiliencyRequestIsComplete(struct ncclIbRequest*, bool*) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencyRequestIsComplete");
}
static ncclResult_t DefaultIbResiliencyProgress(struct ncclIbResiliency*) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencyProgress");
}
static ncclResult_t DefaultIbResiliencyHandleCompletionError(struct ncclIbResiliency*, struct ibv_wc*, int) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencyHandleCompletionError");
}
static ncclResult_t DefaultIbResiliencyDestroy(struct ncclIbResiliency**) { return ncclSuccess; }
static ncclResult_t DefaultIbResiliencyDevInit(struct ncclIbResiliency*, uint, struct ncclIbDev*) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencyDevInit");
}
static ncclResult_t DefaultIbResiliencyDevDestroy(struct ncclIbResiliency*, uint) { return ncclSuccess; }
static ncclResult_t DefaultIbResiliencyDataRqSizeGet(struct ncclIbResiliency*, uint, uint32_t*) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencyDataRqSizeGet");
}
static ncclResult_t DefaultIbResiliencyDataCqSizeGet(struct ncclIbResiliency*, uint, int*) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencyDataCqSizeGet");
}
static ncclResult_t DefaultIbResiliencyDeviceNumSet(struct ncclIbResiliency*, int, int) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencyDeviceNumSet");
}
static ncclResult_t DefaultIbResiliencySenderCreateQps(struct ncclIbResiliency*,
                                                       struct ncclIbResiliencyInfo*) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencySenderCreateQps");
}
static ncclResult_t DefaultIbResiliencySenderQpsToRts(struct ncclIbResiliency*,
                                                       struct ncclIbConnectionMetadata*) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencySenderQpsToRts");
}
static ncclResult_t DefaultIbResiliencyReceiverQpsCreateToRts(struct ncclIbResiliency*,
                                                               struct ncclIbConnectionMetadata*,
                                                               struct ncclIbResiliencyInfo*) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencyReceiverQpsCreateToRts");
}
static ncclResult_t DefaultIbResiliencyClose(struct ncclIbResiliency*) { return ncclSuccess; }
static ncclResult_t DefaultIbResiliencyRemoteCompletionRecordsSet(struct ncclIbResiliency*, uint32_t, uint64_t,
                                                                  uint) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencyRemoteCompletionRecordsSet");
}
static ncclResult_t DefaultIbResiliencyQpReconfigure(struct ncclIbResiliency*, struct ncclIbQp*,
                                                      struct ncclIbNetCommDevBase*, int, bool*) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencyQpReconfigure");
}
static ncclResult_t DefaultIbResiliencyQpsReconfigure(struct ncclIbResiliency*, int, bool*) {
  FailLoudUnfaked("transport_net_ib_resiliency_fakes", "ncclIbResiliencyQpsReconfigure");
}

std::function<ncclResult_t(struct ncclIbRequest*, bool*)> g_ncclIbResiliencyRequestIsComplete =
    DefaultIbResiliencyRequestIsComplete;
std::function<ncclResult_t(struct ncclIbResiliency*)> g_ncclIbResiliencyProgress =
    DefaultIbResiliencyProgress;
std::function<ncclResult_t(struct ncclIbResiliency*, struct ibv_wc*, int)>
    g_ncclIbResiliencyHandleCompletionError = DefaultIbResiliencyHandleCompletionError;
std::function<ncclResult_t(struct ncclIbResiliency**)> g_ncclIbResiliencyDestroy =
    DefaultIbResiliencyDestroy;
std::function<ncclResult_t(struct ncclIbResiliency*, uint, struct ncclIbDev*)> g_ncclIbResiliencyDevInit =
    DefaultIbResiliencyDevInit;
std::function<ncclResult_t(struct ncclIbResiliency*, uint)> g_ncclIbResiliencyDevDestroy =
    DefaultIbResiliencyDevDestroy;
std::function<ncclResult_t(struct ncclIbResiliency*, uint, uint32_t*)> g_ncclIbResiliencyDataRqSizeGet =
    DefaultIbResiliencyDataRqSizeGet;
std::function<ncclResult_t(struct ncclIbResiliency*, uint, int*)> g_ncclIbResiliencyDataCqSizeGet =
    DefaultIbResiliencyDataCqSizeGet;
std::function<ncclResult_t(struct ncclIbResiliency*, int, int)> g_ncclIbResiliencyDeviceNumSet =
    DefaultIbResiliencyDeviceNumSet;
std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbResiliencyInfo*)>
    g_ncclIbResiliencySenderCreateQps = DefaultIbResiliencySenderCreateQps;
std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*)>
    g_ncclIbResiliencySenderQpsToRts = DefaultIbResiliencySenderQpsToRts;
std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*,
                           struct ncclIbResiliencyInfo*)>
    g_ncclIbResiliencyReceiverQpsCreateToRts = DefaultIbResiliencyReceiverQpsCreateToRts;
std::function<ncclResult_t(struct ncclIbResiliency*)> g_ncclIbResiliencyClose = DefaultIbResiliencyClose;
std::function<ncclResult_t(struct ncclIbResiliency*, uint32_t, uint64_t, uint)>
    g_ncclIbResiliencyRemoteCompletionRecordsSet = DefaultIbResiliencyRemoteCompletionRecordsSet;
std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbQp*, struct ncclIbNetCommDevBase*, int,
                           bool*)>
    g_ncclIbResiliencyQpReconfigure = DefaultIbResiliencyQpReconfigure;
std::function<ncclResult_t(struct ncclIbResiliency*, int, bool*)> g_ncclIbResiliencyQpsReconfigure =
    DefaultIbResiliencyQpsReconfigure;

ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyInit, ncclIbResiliencyInit);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyRequestIsComplete, ncclIbResiliencyRequestIsComplete);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyProgress, ncclIbResiliencyProgress);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyHandleCompletionError, ncclIbResiliencyHandleCompletionError);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyDestroy, ncclIbResiliencyDestroy);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyDevInit, ncclIbResiliencyDevInit);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyDevDestroy, ncclIbResiliencyDevDestroy);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyDataRqSizeGet, ncclIbResiliencyDataRqSizeGet);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyDataCqSizeGet, ncclIbResiliencyDataCqSizeGet);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyDeviceNumSet, ncclIbResiliencyDeviceNumSet);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencySenderCreateQps, ncclIbResiliencySenderCreateQps);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencySenderQpsToRts, ncclIbResiliencySenderQpsToRts);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyReceiverQpsCreateToRts, ncclIbResiliencyReceiverQpsCreateToRts);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyClose, ncclIbResiliencyClose);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyRemoteCompletionRecordsSet,
                         ncclIbResiliencyRemoteCompletionRecordsSet);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyQpReconfigure, ncclIbResiliencyQpReconfigure);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyQpsReconfigure, ncclIbResiliencyQpsReconfigure);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t ncclIbResiliencyInit(struct ncclIbNetCommBase* base, struct ncclIbResiliency** resiliency) {
  return g_ncclIbResiliencyInit(base, resiliency);
}

ncclResult_t ncclIbResiliencyRequestIsComplete(struct ncclIbRequest* request, bool* complete) {
  return g_ncclIbResiliencyRequestIsComplete(request, complete);
}
ncclResult_t ncclIbResiliencyProgress(struct ncclIbResiliency* resiliency) {
  return g_ncclIbResiliencyProgress(resiliency);
}
ncclResult_t ncclIbResiliencyHandleCompletionError(struct ncclIbResiliency* resiliency, struct ibv_wc* wc,
                                                    int devIndex) {
  return g_ncclIbResiliencyHandleCompletionError(resiliency, wc, devIndex);
}
ncclResult_t ncclIbResiliencyDestroy(struct ncclIbResiliency** resiliency) {
  return g_ncclIbResiliencyDestroy(resiliency);
}
ncclResult_t ncclIbResiliencyDevInit(struct ncclIbResiliency* resiliency, uint devIndex, struct ncclIbDev* dev) {
  return g_ncclIbResiliencyDevInit(resiliency, devIndex, dev);
}
ncclResult_t ncclIbResiliencyDevDestroy(struct ncclIbResiliency* resiliency, uint devIndex) {
  return g_ncclIbResiliencyDevDestroy(resiliency, devIndex);
}
ncclResult_t ncclIbResiliencyDataRqSizeGet(struct ncclIbResiliency* resiliency, uint devIndex, uint32_t* size) {
  return g_ncclIbResiliencyDataRqSizeGet(resiliency, devIndex, size);
}
ncclResult_t ncclIbResiliencyDataCqSizeGet(struct ncclIbResiliency* resiliency, uint devIndex, int* size) {
  return g_ncclIbResiliencyDataCqSizeGet(resiliency, devIndex, size);
}
ncclResult_t ncclIbResiliencyDeviceNumSet(struct ncclIbResiliency* resiliency, int local, int remote) {
  return g_ncclIbResiliencyDeviceNumSet(resiliency, local, remote);
}
ncclResult_t ncclIbResiliencySenderCreateQps(struct ncclIbResiliency* resiliency,
                                             struct ncclIbResiliencyInfo* info) {
  return g_ncclIbResiliencySenderCreateQps(resiliency, info);
}
ncclResult_t ncclIbResiliencySenderQpsToRts(struct ncclIbResiliency* resiliency,
                                            struct ncclIbConnectionMetadata* info) {
  return g_ncclIbResiliencySenderQpsToRts(resiliency, info);
}
ncclResult_t ncclIbResiliencyReceiverQpsCreateToRts(struct ncclIbResiliency* resiliency,
                                                     struct ncclIbConnectionMetadata* remote,
                                                     struct ncclIbResiliencyInfo* local) {
  return g_ncclIbResiliencyReceiverQpsCreateToRts(resiliency, remote, local);
}
ncclResult_t ncclIbResiliencyClose(struct ncclIbResiliency* resiliency) {
  return g_ncclIbResiliencyClose(resiliency);
}
ncclResult_t ncclIbResiliencyRemoteCompletionRecordsSet(struct ncclIbResiliency* resiliency, uint32_t rkey,
                                                         uint64_t address, uint devIndex) {
  return g_ncclIbResiliencyRemoteCompletionRecordsSet(resiliency, rkey, address, devIndex);
}
ncclResult_t ncclIbResiliencyQpReconfigure(struct ncclIbResiliency* resiliency, struct ncclIbQp* qp,
                                            struct ncclIbNetCommDevBase* devBase, int devIndex,
                                            bool* success) {
  return g_ncclIbResiliencyQpReconfigure(resiliency, qp, devBase, devIndex, success);
}
ncclResult_t ncclIbResiliencyQpsReconfigure(struct ncclIbResiliency* resiliency, int devIndex,
                                             bool* success) {
  return g_ncclIbResiliencyQpsReconfigure(resiliency, devIndex, success);
}

void ResetTransportNetIbResiliencyFakes() {
  g_ncclIbResiliencyInit = DefaultIbResiliencyInit;
  g_ncclIbResiliencyRequestIsComplete = DefaultIbResiliencyRequestIsComplete;
  g_ncclIbResiliencyProgress = DefaultIbResiliencyProgress;
  g_ncclIbResiliencyHandleCompletionError = DefaultIbResiliencyHandleCompletionError;
  g_ncclIbResiliencyDestroy = DefaultIbResiliencyDestroy;
  g_ncclIbResiliencyDevInit = DefaultIbResiliencyDevInit;
  g_ncclIbResiliencyDevDestroy = DefaultIbResiliencyDevDestroy;
  g_ncclIbResiliencyDataRqSizeGet = DefaultIbResiliencyDataRqSizeGet;
  g_ncclIbResiliencyDataCqSizeGet = DefaultIbResiliencyDataCqSizeGet;
  g_ncclIbResiliencyDeviceNumSet = DefaultIbResiliencyDeviceNumSet;
  g_ncclIbResiliencySenderCreateQps = DefaultIbResiliencySenderCreateQps;
  g_ncclIbResiliencySenderQpsToRts = DefaultIbResiliencySenderQpsToRts;
  g_ncclIbResiliencyReceiverQpsCreateToRts = DefaultIbResiliencyReceiverQpsCreateToRts;
  g_ncclIbResiliencyClose = DefaultIbResiliencyClose;
  g_ncclIbResiliencyRemoteCompletionRecordsSet = DefaultIbResiliencyRemoteCompletionRecordsSet;
  g_ncclIbResiliencyQpReconfigure = DefaultIbResiliencyQpReconfigure;
  g_ncclIbResiliencyQpsReconfigure = DefaultIbResiliencyQpsReconfigure;
}
