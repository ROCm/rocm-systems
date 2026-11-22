/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "transport_net_ib_mseg_resiliency_fakes.h"

#include "fail_loud.h"
#include "fakes/nccl_fakes.h"
#include "signature-drift.h"

int64_t ncclParamIbResiliencyPortFailover() {
  return g_loadParam("IB_RESILIENCY_PORT_FAILOVER", 0);
}

static ncclResult_t DefaultIbResiliencyInit(struct IbMsegNetCommBase*, struct IbMsegResiliency**) {
  FailLoudUnfaked("transport_net_ib_mseg_resiliency_fakes", "IbMsegResiliencyInit");
}
std::function<ncclResult_t(struct IbMsegNetCommBase*, struct IbMsegResiliency**)> g_ncclIbResiliencyInit =
    DefaultIbResiliencyInit;

static ncclResult_t DefaultIbResiliencyRequestIsComplete(struct IbMsegRequest*, bool*) {
  FailLoudUnfaked("transport_net_ib_mseg_resiliency_fakes", "IbMsegResiliencyRequestIsComplete");
}
static ncclResult_t DefaultIbResiliencyProgress(struct IbMsegResiliency*) {
  FailLoudUnfaked("transport_net_ib_mseg_resiliency_fakes", "IbMsegResiliencyProgress");
}
static ncclResult_t DefaultIbResiliencyHandleCompletionError(struct IbMsegResiliency*, struct ibv_wc*, int) {
  FailLoudUnfaked("transport_net_ib_mseg_resiliency_fakes", "IbMsegResiliencyHandleCompletionError");
}
static ncclResult_t DefaultIbResiliencyDestroy(struct IbMsegResiliency**) { return ncclSuccess; }
static ncclResult_t DefaultIbResiliencyDevInit(struct IbMsegResiliency*, uint, struct IbMsegDev*) {
  FailLoudUnfaked("transport_net_ib_mseg_resiliency_fakes", "IbMsegResiliencyDevInit");
}
static ncclResult_t DefaultIbResiliencyDevDestroy(struct IbMsegResiliency*, uint) { return ncclSuccess; }
static ncclResult_t DefaultIbResiliencyDataRqSizeGet(struct IbMsegResiliency*, uint, uint32_t*) {
  FailLoudUnfaked("transport_net_ib_mseg_resiliency_fakes", "IbMsegResiliencyDataRqSizeGet");
}
static ncclResult_t DefaultIbResiliencyDataCqSizeGet(struct IbMsegResiliency*, uint, int*) {
  FailLoudUnfaked("transport_net_ib_mseg_resiliency_fakes", "IbMsegResiliencyDataCqSizeGet");
}
static ncclResult_t DefaultIbResiliencyDeviceNumSet(struct IbMsegResiliency*, int, int) {
  FailLoudUnfaked("transport_net_ib_mseg_resiliency_fakes", "IbMsegResiliencyDeviceNumSet");
}
static ncclResult_t DefaultIbResiliencySenderCreateQps(struct IbMsegResiliency*,
                                                       struct IbMsegResiliencyInfo*) {
  FailLoudUnfaked("transport_net_ib_mseg_resiliency_fakes", "IbMsegResiliencySenderCreateQps");
}
static ncclResult_t DefaultIbResiliencySenderQpsToRts(struct IbMsegResiliency*,
                                                       struct IbMsegConnectionMetadata*) {
  FailLoudUnfaked("transport_net_ib_mseg_resiliency_fakes", "IbMsegResiliencySenderQpsToRts");
}
static ncclResult_t DefaultIbResiliencyReceiverQpsCreateToRts(struct IbMsegResiliency*,
                                                               struct IbMsegConnectionMetadata*,
                                                               struct IbMsegResiliencyInfo*) {
  FailLoudUnfaked("transport_net_ib_mseg_resiliency_fakes", "IbMsegResiliencyReceiverQpsCreateToRts");
}
static ncclResult_t DefaultIbResiliencyClose(struct IbMsegResiliency*) { return ncclSuccess; }
static ncclResult_t DefaultIbResiliencyRemoteCompletionRecordsSet(struct IbMsegResiliency*, uint32_t, uint64_t,
                                                                  uint) {
  FailLoudUnfaked("transport_net_ib_mseg_resiliency_fakes", "IbMsegResiliencyRemoteCompletionRecordsSet");
}
static ncclResult_t DefaultIbResiliencyQpReconfigure(struct IbMsegResiliency*, struct IbMsegQp*,
                                                      struct IbMsegNetCommDevBase*, int, bool*) {
  FailLoudUnfaked("transport_net_ib_mseg_resiliency_fakes", "IbMsegResiliencyQpReconfigure");
}
static ncclResult_t DefaultIbResiliencyQpsReconfigure(struct IbMsegResiliency*, int, bool*) {
  FailLoudUnfaked("transport_net_ib_mseg_resiliency_fakes", "IbMsegResiliencyQpsReconfigure");
}

std::function<ncclResult_t(struct IbMsegRequest*, bool*)> g_ncclIbResiliencyRequestIsComplete =
    DefaultIbResiliencyRequestIsComplete;
std::function<ncclResult_t(struct IbMsegResiliency*)> g_ncclIbResiliencyProgress =
    DefaultIbResiliencyProgress;
std::function<ncclResult_t(struct IbMsegResiliency*, struct ibv_wc*, int)>
    g_ncclIbResiliencyHandleCompletionError = DefaultIbResiliencyHandleCompletionError;
std::function<ncclResult_t(struct IbMsegResiliency**)> g_ncclIbResiliencyDestroy =
    DefaultIbResiliencyDestroy;
std::function<ncclResult_t(struct IbMsegResiliency*, uint, struct IbMsegDev*)> g_ncclIbResiliencyDevInit =
    DefaultIbResiliencyDevInit;
std::function<ncclResult_t(struct IbMsegResiliency*, uint)> g_ncclIbResiliencyDevDestroy =
    DefaultIbResiliencyDevDestroy;
std::function<ncclResult_t(struct IbMsegResiliency*, uint, uint32_t*)> g_ncclIbResiliencyDataRqSizeGet =
    DefaultIbResiliencyDataRqSizeGet;
std::function<ncclResult_t(struct IbMsegResiliency*, uint, int*)> g_ncclIbResiliencyDataCqSizeGet =
    DefaultIbResiliencyDataCqSizeGet;
std::function<ncclResult_t(struct IbMsegResiliency*, int, int)> g_ncclIbResiliencyDeviceNumSet =
    DefaultIbResiliencyDeviceNumSet;
std::function<ncclResult_t(struct IbMsegResiliency*, struct IbMsegResiliencyInfo*)>
    g_ncclIbResiliencySenderCreateQps = DefaultIbResiliencySenderCreateQps;
std::function<ncclResult_t(struct IbMsegResiliency*, struct IbMsegConnectionMetadata*)>
    g_ncclIbResiliencySenderQpsToRts = DefaultIbResiliencySenderQpsToRts;
std::function<ncclResult_t(struct IbMsegResiliency*, struct IbMsegConnectionMetadata*,
                           struct IbMsegResiliencyInfo*)>
    g_ncclIbResiliencyReceiverQpsCreateToRts = DefaultIbResiliencyReceiverQpsCreateToRts;
std::function<ncclResult_t(struct IbMsegResiliency*)> g_ncclIbResiliencyClose = DefaultIbResiliencyClose;
std::function<ncclResult_t(struct IbMsegResiliency*, uint32_t, uint64_t, uint)>
    g_ncclIbResiliencyRemoteCompletionRecordsSet = DefaultIbResiliencyRemoteCompletionRecordsSet;
std::function<ncclResult_t(struct IbMsegResiliency*, struct IbMsegQp*, struct IbMsegNetCommDevBase*, int,
                           bool*)>
    g_ncclIbResiliencyQpReconfigure = DefaultIbResiliencyQpReconfigure;
std::function<ncclResult_t(struct IbMsegResiliency*, int, bool*)> g_ncclIbResiliencyQpsReconfigure =
    DefaultIbResiliencyQpsReconfigure;

ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyInit, IbMsegResiliencyInit);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyRequestIsComplete, IbMsegResiliencyRequestIsComplete);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyProgress, IbMsegResiliencyProgress);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyHandleCompletionError, IbMsegResiliencyHandleCompletionError);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyDestroy, IbMsegResiliencyDestroy);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyDevInit, IbMsegResiliencyDevInit);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyDevDestroy, IbMsegResiliencyDevDestroy);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyDataRqSizeGet, IbMsegResiliencyDataRqSizeGet);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyDataCqSizeGet, IbMsegResiliencyDataCqSizeGet);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyDeviceNumSet, IbMsegResiliencyDeviceNumSet);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencySenderCreateQps, IbMsegResiliencySenderCreateQps);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencySenderQpsToRts, IbMsegResiliencySenderQpsToRts);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyReceiverQpsCreateToRts, IbMsegResiliencyReceiverQpsCreateToRts);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyClose, IbMsegResiliencyClose);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyRemoteCompletionRecordsSet,
                         IbMsegResiliencyRemoteCompletionRecordsSet);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyQpReconfigure, IbMsegResiliencyQpReconfigure);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbResiliencyQpsReconfigure, IbMsegResiliencyQpsReconfigure);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t IbMsegResiliencyInit(struct IbMsegNetCommBase* base, struct IbMsegResiliency** resiliency) {
  return g_ncclIbResiliencyInit(base, resiliency);
}

ncclResult_t IbMsegResiliencyRequestIsComplete(struct IbMsegRequest* request, bool* complete) {
  return g_ncclIbResiliencyRequestIsComplete(request, complete);
}
ncclResult_t IbMsegResiliencyProgress(struct IbMsegResiliency* resiliency) {
  return g_ncclIbResiliencyProgress(resiliency);
}
ncclResult_t IbMsegResiliencyHandleCompletionError(struct IbMsegResiliency* resiliency, struct ibv_wc* wc,
                                                    int devIndex) {
  return g_ncclIbResiliencyHandleCompletionError(resiliency, wc, devIndex);
}
ncclResult_t IbMsegResiliencyDestroy(struct IbMsegResiliency** resiliency) {
  return g_ncclIbResiliencyDestroy(resiliency);
}
ncclResult_t IbMsegResiliencyDevInit(struct IbMsegResiliency* resiliency, uint devIndex, struct IbMsegDev* dev) {
  return g_ncclIbResiliencyDevInit(resiliency, devIndex, dev);
}
ncclResult_t IbMsegResiliencyDevDestroy(struct IbMsegResiliency* resiliency, uint devIndex) {
  return g_ncclIbResiliencyDevDestroy(resiliency, devIndex);
}
ncclResult_t IbMsegResiliencyDataRqSizeGet(struct IbMsegResiliency* resiliency, uint devIndex, uint32_t* size) {
  return g_ncclIbResiliencyDataRqSizeGet(resiliency, devIndex, size);
}
ncclResult_t IbMsegResiliencyDataCqSizeGet(struct IbMsegResiliency* resiliency, uint devIndex, int* size) {
  return g_ncclIbResiliencyDataCqSizeGet(resiliency, devIndex, size);
}
ncclResult_t IbMsegResiliencyDeviceNumSet(struct IbMsegResiliency* resiliency, int local, int remote) {
  return g_ncclIbResiliencyDeviceNumSet(resiliency, local, remote);
}
ncclResult_t IbMsegResiliencySenderCreateQps(struct IbMsegResiliency* resiliency,
                                             struct IbMsegResiliencyInfo* info) {
  return g_ncclIbResiliencySenderCreateQps(resiliency, info);
}
ncclResult_t IbMsegResiliencySenderQpsToRts(struct IbMsegResiliency* resiliency,
                                            struct IbMsegConnectionMetadata* info) {
  return g_ncclIbResiliencySenderQpsToRts(resiliency, info);
}
ncclResult_t IbMsegResiliencyReceiverQpsCreateToRts(struct IbMsegResiliency* resiliency,
                                                     struct IbMsegConnectionMetadata* remote,
                                                     struct IbMsegResiliencyInfo* local) {
  return g_ncclIbResiliencyReceiverQpsCreateToRts(resiliency, remote, local);
}
ncclResult_t IbMsegResiliencyClose(struct IbMsegResiliency* resiliency) {
  return g_ncclIbResiliencyClose(resiliency);
}
ncclResult_t IbMsegResiliencyRemoteCompletionRecordsSet(struct IbMsegResiliency* resiliency, uint32_t rkey,
                                                         uint64_t address, uint devIndex) {
  return g_ncclIbResiliencyRemoteCompletionRecordsSet(resiliency, rkey, address, devIndex);
}
ncclResult_t IbMsegResiliencyQpReconfigure(struct IbMsegResiliency* resiliency, struct IbMsegQp* qp,
                                            struct IbMsegNetCommDevBase* devBase, int devIndex,
                                            bool* success) {
  return g_ncclIbResiliencyQpReconfigure(resiliency, qp, devBase, devIndex, success);
}
ncclResult_t IbMsegResiliencyQpsReconfigure(struct IbMsegResiliency* resiliency, int devIndex,
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
