/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "transport_net_ib_recovery_fakes.h"

#include "fail_loud.h"
#include "signature-drift.h"
#include "transport/net_ib/p2p_resiliency_recovery.h"

static ncclResult_t DefaultPortRecoveryThreadStart() {
  FailLoudUnfaked("transport_net_ib_recovery_fakes", "ncclIbPortRecoveryThreadStart");
}
static ncclResult_t DefaultPortRecoveryThreadStop() {
  FailLoudUnfaked("transport_net_ib_recovery_fakes", "ncclIbPortRecoveryThreadStop");
}
static ncclResult_t DefaultPortRecoveryInit(struct ncclIbResiliency*) {
  FailLoudUnfaked("transport_net_ib_recovery_fakes", "ncclIbPortRecoveryInit");
}
static ncclResult_t DefaultPortRecoveryClose(struct ncclIbResiliency*) {
  FailLoudUnfaked("transport_net_ib_recovery_fakes", "ncclIbPortRecoveryClose");
}
static ncclResult_t DefaultPortRecoveryDevInit(struct ncclIbResiliency*, int, struct ncclIbDev*) {
  FailLoudUnfaked("transport_net_ib_recovery_fakes", "ncclIbPortRecoveryDevInit");
}
static ncclResult_t DefaultPortRecoveryDevDestroy(struct ncclIbResiliency*, int) {
  FailLoudUnfaked("transport_net_ib_recovery_fakes", "ncclIbPortRecoveryDevDestroy");
}
static ncclResult_t DefaultPortRecoverySenderQpsCreate(struct ncclIbResiliency*, struct ncclIbQpInfo*, int) {
  FailLoudUnfaked("transport_net_ib_recovery_fakes", "ncclIbPortRecoverySenderQpsCreate");
}
static ncclResult_t DefaultPortRecoverySenderQpsToRts(struct ncclIbResiliency*,
                                                       struct ncclIbConnectionMetadata*, int) {
  FailLoudUnfaked("transport_net_ib_recovery_fakes", "ncclIbPortRecoverySenderQpsToRts");
}
static ncclResult_t DefaultPortRecoveryReceiverQpsCreateToRts(struct ncclIbResiliency*,
                                                               struct ncclIbConnectionMetadata*,
                                                               struct ncclIbQpInfo*, int) {
  FailLoudUnfaked("transport_net_ib_recovery_fakes", "ncclIbPortRecoveryReceiverQpsCreateToRts");
}
static ncclResult_t DefaultPortRecoveryQpsDestroy(struct ncclIbResiliency*, int) {
  FailLoudUnfaked("transport_net_ib_recovery_fakes", "ncclIbPortRecoveryQpsDestroy");
}
static ncclResult_t DefaultPortRecoveryHandleFailure(struct ncclIbResiliency*, int) {
  FailLoudUnfaked("transport_net_ib_recovery_fakes", "ncclIbPortRecoveryHandleFailure");
}
static ncclResult_t DefaultPortRecoveryQpsReconfigure(struct ncclIbResiliency*, int, bool*) {
  FailLoudUnfaked("transport_net_ib_recovery_fakes", "ncclIbPortRecoveryQpsReconfigure");
}

std::function<ncclResult_t()> g_ncclIbPortRecoveryThreadStart = DefaultPortRecoveryThreadStart;
std::function<ncclResult_t()> g_ncclIbPortRecoveryThreadStop = DefaultPortRecoveryThreadStop;
std::function<ncclResult_t(struct ncclIbResiliency*)> g_ncclIbPortRecoveryInit = DefaultPortRecoveryInit;
std::function<ncclResult_t(struct ncclIbResiliency*)> g_ncclIbPortRecoveryClose = DefaultPortRecoveryClose;
std::function<ncclResult_t(struct ncclIbResiliency*, int, struct ncclIbDev*)> g_ncclIbPortRecoveryDevInit =
    DefaultPortRecoveryDevInit;
std::function<ncclResult_t(struct ncclIbResiliency*, int)> g_ncclIbPortRecoveryDevDestroy =
    DefaultPortRecoveryDevDestroy;
std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbQpInfo*, int)>
    g_ncclIbPortRecoverySenderQpsCreate = DefaultPortRecoverySenderQpsCreate;
std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*, int)>
    g_ncclIbPortRecoverySenderQpsToRts = DefaultPortRecoverySenderQpsToRts;
std::function<ncclResult_t(struct ncclIbResiliency*, struct ncclIbConnectionMetadata*,
                           struct ncclIbQpInfo*, int)>
    g_ncclIbPortRecoveryReceiverQpsCreateToRts = DefaultPortRecoveryReceiverQpsCreateToRts;
std::function<ncclResult_t(struct ncclIbResiliency*, int)> g_ncclIbPortRecoveryQpsDestroy =
    DefaultPortRecoveryQpsDestroy;
std::function<ncclResult_t(struct ncclIbResiliency*, int)> g_ncclIbPortRecoveryHandleFailure =
    DefaultPortRecoveryHandleFailure;
std::function<ncclResult_t(struct ncclIbResiliency*, int, bool*)> g_ncclIbPortRecoveryQpsReconfigure =
    DefaultPortRecoveryQpsReconfigure;

ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryThreadStart, ncclIbPortRecoveryThreadStart);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryThreadStop, ncclIbPortRecoveryThreadStop);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryInit, ncclIbPortRecoveryInit);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryClose, ncclIbPortRecoveryClose);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryDevInit, ncclIbPortRecoveryDevInit);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryDevDestroy, ncclIbPortRecoveryDevDestroy);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoverySenderQpsCreate, ncclIbPortRecoverySenderQpsCreate);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoverySenderQpsToRts, ncclIbPortRecoverySenderQpsToRts);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryReceiverQpsCreateToRts, ncclIbPortRecoveryReceiverQpsCreateToRts);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryQpsDestroy, ncclIbPortRecoveryQpsDestroy);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryHandleFailure, ncclIbPortRecoveryHandleFailure);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbPortRecoveryQpsReconfigure, ncclIbPortRecoveryQpsReconfigure);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t ncclIbPortRecoveryThreadStart() { return g_ncclIbPortRecoveryThreadStart(); }
ncclResult_t ncclIbPortRecoveryThreadStop() { return g_ncclIbPortRecoveryThreadStop(); }
ncclResult_t ncclIbPortRecoveryInit(struct ncclIbResiliency* resiliency) {
  return g_ncclIbPortRecoveryInit(resiliency);
}
ncclResult_t ncclIbPortRecoveryClose(struct ncclIbResiliency* resiliency) {
  return g_ncclIbPortRecoveryClose(resiliency);
}
ncclResult_t ncclIbPortRecoveryDevInit(struct ncclIbResiliency* resiliency, int devIndex,
                                       struct ncclIbDev* dev) {
  return g_ncclIbPortRecoveryDevInit(resiliency, devIndex, dev);
}
ncclResult_t ncclIbPortRecoveryDevDestroy(struct ncclIbResiliency* resiliency, int devIndex) {
  return g_ncclIbPortRecoveryDevDestroy(resiliency, devIndex);
}
ncclResult_t ncclIbPortRecoverySenderQpsCreate(struct ncclIbResiliency* resiliency,
                                               struct ncclIbQpInfo* info, int count) {
  return g_ncclIbPortRecoverySenderQpsCreate(resiliency, info, count);
}
ncclResult_t ncclIbPortRecoverySenderQpsToRts(struct ncclIbResiliency* resiliency,
                                              struct ncclIbConnectionMetadata* info, int count) {
  return g_ncclIbPortRecoverySenderQpsToRts(resiliency, info, count);
}
ncclResult_t ncclIbPortRecoveryReceiverQpsCreateToRts(struct ncclIbResiliency* resiliency,
                                                       struct ncclIbConnectionMetadata* remote,
                                                       struct ncclIbQpInfo* local, int count) {
  return g_ncclIbPortRecoveryReceiverQpsCreateToRts(resiliency, remote, local, count);
}
ncclResult_t ncclIbPortRecoveryQpsDestroy(struct ncclIbResiliency* resiliency, int count) {
  return g_ncclIbPortRecoveryQpsDestroy(resiliency, count);
}
ncclResult_t ncclIbPortRecoveryHandleFailure(struct ncclIbResiliency* resiliency, int devIndex) {
  return g_ncclIbPortRecoveryHandleFailure(resiliency, devIndex);
}
ncclResult_t ncclIbPortRecoveryQpsReconfigure(struct ncclIbResiliency* resiliency, int devIndex,
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
