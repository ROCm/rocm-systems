/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "transport_net_ib_mseg_recovery_fakes.h"

#include "fail_loud.h"
#include "signature-drift.h"
#include "transport/net_ib_multiseg/p2p_resiliency_recovery_mseg.h"

static ncclResult_t DefaultPortRecoveryThreadStart() {
  FailLoudUnfaked("transport_net_ib_mseg_recovery_fakes", "IbMsegPortRecoveryThreadStart");
}
static ncclResult_t DefaultPortRecoveryThreadStop() {
  FailLoudUnfaked("transport_net_ib_mseg_recovery_fakes", "IbMsegPortRecoveryThreadStop");
}
static ncclResult_t DefaultPortRecoveryInit(struct IbMsegResiliency*) {
  FailLoudUnfaked("transport_net_ib_mseg_recovery_fakes", "IbMsegPortRecoveryInit");
}
static ncclResult_t DefaultPortRecoveryClose(struct IbMsegResiliency*) {
  FailLoudUnfaked("transport_net_ib_mseg_recovery_fakes", "IbMsegPortRecoveryClose");
}
static ncclResult_t DefaultPortRecoveryDevInit(struct IbMsegResiliency*, int, struct IbMsegDev*) {
  FailLoudUnfaked("transport_net_ib_mseg_recovery_fakes", "IbMsegPortRecoveryDevInit");
}
static ncclResult_t DefaultPortRecoveryDevDestroy(struct IbMsegResiliency*, int) {
  FailLoudUnfaked("transport_net_ib_mseg_recovery_fakes", "IbMsegPortRecoveryDevDestroy");
}
static ncclResult_t DefaultPortRecoverySenderQpsCreate(struct IbMsegResiliency*, struct IbMsegQpInfo*, int) {
  FailLoudUnfaked("transport_net_ib_mseg_recovery_fakes", "IbMsegPortRecoverySenderQpsCreate");
}
static ncclResult_t DefaultPortRecoverySenderQpsToRts(struct IbMsegResiliency*,
                                                       struct IbMsegConnectionMetadata*, int) {
  FailLoudUnfaked("transport_net_ib_mseg_recovery_fakes", "IbMsegPortRecoverySenderQpsToRts");
}
static ncclResult_t DefaultPortRecoveryReceiverQpsCreateToRts(struct IbMsegResiliency*,
                                                               struct IbMsegConnectionMetadata*,
                                                               struct IbMsegQpInfo*, int) {
  FailLoudUnfaked("transport_net_ib_mseg_recovery_fakes", "IbMsegPortRecoveryReceiverQpsCreateToRts");
}
static ncclResult_t DefaultPortRecoveryQpsDestroy(struct IbMsegResiliency*, int) {
  FailLoudUnfaked("transport_net_ib_mseg_recovery_fakes", "IbMsegPortRecoveryQpsDestroy");
}
static ncclResult_t DefaultPortRecoveryHandleFailure(struct IbMsegResiliency*, int) {
  FailLoudUnfaked("transport_net_ib_mseg_recovery_fakes", "IbMsegPortRecoveryHandleFailure");
}
static ncclResult_t DefaultPortRecoveryQpsReconfigure(struct IbMsegResiliency*, int, bool*) {
  FailLoudUnfaked("transport_net_ib_mseg_recovery_fakes", "IbMsegPortRecoveryQpsReconfigure");
}

std::function<ncclResult_t()> g_ncclIbPortRecoveryThreadStart = DefaultPortRecoveryThreadStart;
std::function<ncclResult_t()> g_ncclIbPortRecoveryThreadStop = DefaultPortRecoveryThreadStop;
std::function<ncclResult_t(struct IbMsegResiliency*)> g_ncclIbPortRecoveryInit = DefaultPortRecoveryInit;
std::function<ncclResult_t(struct IbMsegResiliency*)> g_ncclIbPortRecoveryClose = DefaultPortRecoveryClose;
std::function<ncclResult_t(struct IbMsegResiliency*, int, struct IbMsegDev*)> g_ncclIbPortRecoveryDevInit =
    DefaultPortRecoveryDevInit;
std::function<ncclResult_t(struct IbMsegResiliency*, int)> g_ncclIbPortRecoveryDevDestroy =
    DefaultPortRecoveryDevDestroy;
std::function<ncclResult_t(struct IbMsegResiliency*, struct IbMsegQpInfo*, int)>
    g_ncclIbPortRecoverySenderQpsCreate = DefaultPortRecoverySenderQpsCreate;
std::function<ncclResult_t(struct IbMsegResiliency*, struct IbMsegConnectionMetadata*, int)>
    g_ncclIbPortRecoverySenderQpsToRts = DefaultPortRecoverySenderQpsToRts;
std::function<ncclResult_t(struct IbMsegResiliency*, struct IbMsegConnectionMetadata*,
                           struct IbMsegQpInfo*, int)>
    g_ncclIbPortRecoveryReceiverQpsCreateToRts = DefaultPortRecoveryReceiverQpsCreateToRts;
std::function<ncclResult_t(struct IbMsegResiliency*, int)> g_ncclIbPortRecoveryQpsDestroy =
    DefaultPortRecoveryQpsDestroy;
std::function<ncclResult_t(struct IbMsegResiliency*, int)> g_ncclIbPortRecoveryHandleFailure =
    DefaultPortRecoveryHandleFailure;
std::function<ncclResult_t(struct IbMsegResiliency*, int, bool*)> g_ncclIbPortRecoveryQpsReconfigure =
    DefaultPortRecoveryQpsReconfigure;

ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryThreadStart, IbMsegPortRecoveryThreadStart);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryThreadStop, IbMsegPortRecoveryThreadStop);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryInit, IbMsegPortRecoveryInit);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryClose, IbMsegPortRecoveryClose);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryDevInit, IbMsegPortRecoveryDevInit);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryDevDestroy, IbMsegPortRecoveryDevDestroy);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoverySenderQpsCreate, IbMsegPortRecoverySenderQpsCreate);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoverySenderQpsToRts, IbMsegPortRecoverySenderQpsToRts);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryReceiverQpsCreateToRts, IbMsegPortRecoveryReceiverQpsCreateToRts);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryQpsDestroy, IbMsegPortRecoveryQpsDestroy);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryHandleFailure, IbMsegPortRecoveryHandleFailure);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryQpsReconfigure, IbMsegPortRecoveryQpsReconfigure);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t IbMsegPortRecoveryThreadStart() { return g_ncclIbPortRecoveryThreadStart(); }
ncclResult_t IbMsegPortRecoveryThreadStop() { return g_ncclIbPortRecoveryThreadStop(); }
ncclResult_t IbMsegPortRecoveryInit(struct IbMsegResiliency* resiliency) {
  return g_ncclIbPortRecoveryInit(resiliency);
}
ncclResult_t IbMsegPortRecoveryClose(struct IbMsegResiliency* resiliency) {
  return g_ncclIbPortRecoveryClose(resiliency);
}
ncclResult_t IbMsegPortRecoveryDevInit(struct IbMsegResiliency* resiliency, int devIndex,
                                       struct IbMsegDev* dev) {
  return g_ncclIbPortRecoveryDevInit(resiliency, devIndex, dev);
}
ncclResult_t IbMsegPortRecoveryDevDestroy(struct IbMsegResiliency* resiliency, int devIndex) {
  return g_ncclIbPortRecoveryDevDestroy(resiliency, devIndex);
}
ncclResult_t IbMsegPortRecoverySenderQpsCreate(struct IbMsegResiliency* resiliency,
                                               struct IbMsegQpInfo* info, int count) {
  return g_ncclIbPortRecoverySenderQpsCreate(resiliency, info, count);
}
ncclResult_t IbMsegPortRecoverySenderQpsToRts(struct IbMsegResiliency* resiliency,
                                              struct IbMsegConnectionMetadata* info, int count) {
  return g_ncclIbPortRecoverySenderQpsToRts(resiliency, info, count);
}
ncclResult_t IbMsegPortRecoveryReceiverQpsCreateToRts(struct IbMsegResiliency* resiliency,
                                                       struct IbMsegConnectionMetadata* remote,
                                                       struct IbMsegQpInfo* local, int count) {
  return g_ncclIbPortRecoveryReceiverQpsCreateToRts(resiliency, remote, local, count);
}
ncclResult_t IbMsegPortRecoveryQpsDestroy(struct IbMsegResiliency* resiliency, int count) {
  return g_ncclIbPortRecoveryQpsDestroy(resiliency, count);
}
ncclResult_t IbMsegPortRecoveryHandleFailure(struct IbMsegResiliency* resiliency, int devIndex) {
  return g_ncclIbPortRecoveryHandleFailure(resiliency, devIndex);
}
ncclResult_t IbMsegPortRecoveryQpsReconfigure(struct IbMsegResiliency* resiliency, int devIndex,
                                              bool* success) {
  return g_ncclIbPortRecoveryQpsReconfigure(resiliency, devIndex, success);
}

void ResetTransportNetIbRecoveryFakes() {
  g_ncclIbPortRecoveryThreadStart = DefaultPortRecoveryThreadStart;
  g_ncclIbPortRecoveryThreadStop = DefaultPortRecoveryThreadStop;
  g_ncclIbPortRecoveryInit = DefaultPortRecoveryInit;
  g_ncclIbPortRecoveryClose = DefaultPortRecoveryClose;
  g_ncclIbPortRecoveryDevInit = DefaultPortRecoveryDevInit;
  g_ncclIbPortRecoveryDevDestroy = DefaultPortRecoveryDevDestroy;
  g_ncclIbPortRecoverySenderQpsCreate = DefaultPortRecoverySenderQpsCreate;
  g_ncclIbPortRecoverySenderQpsToRts = DefaultPortRecoverySenderQpsToRts;
  g_ncclIbPortRecoveryReceiverQpsCreateToRts = DefaultPortRecoveryReceiverQpsCreateToRts;
  g_ncclIbPortRecoveryQpsDestroy = DefaultPortRecoveryQpsDestroy;
  g_ncclIbPortRecoveryHandleFailure = DefaultPortRecoveryHandleFailure;
  g_ncclIbPortRecoveryQpsReconfigure = DefaultPortRecoveryQpsReconfigure;
}
