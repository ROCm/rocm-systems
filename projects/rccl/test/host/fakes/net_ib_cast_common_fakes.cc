/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/transport/net_ib_cast/common.cc fakes. See
// net_ib_cast_common_fakes.h.

#include "net_ib_cast_common_fakes.h"

#include "fail_loud.h"
#include "nccl_fakes.h"
#include "signature-drift.h"

char IbCastIfName[MAX_IF_NAME_SIZE + 1];
union ncclSocketAddress IbCastIfAddr;
int IbCastNMergedDevs = -1;
int IbCastNDevs = -1;
struct ncclIbMergedDev IbCastMergedDevs[MAX_IB_VDEVS];
struct ncclIbDev IbCastDevs[MAX_IB_DEVS];
int IbCastRelaxedOrderingEnabled = 0;
ncclProfilerCallback_t IbCastProfilerFunction;
std::thread IbCastAsyncThread;

int64_t ncclParamIbCastSplitDataOnQps() { return g_loadParam("IB_SPLIT_DATA_ON_QPS", 0); }
int64_t ncclParamIbCastPrepostReceiveWorkRequests() { return g_loadParam("IB_PREPOST_RECEIVE_WORK_REQUESTS", -2); }
int64_t ncclParamIbCastAsyncEvents() { return g_loadParam("IB_RETURN_ASYNC_EVENTS", 1); }
int64_t rcclParamIbCastOptionalRecvCompletion() { return g_loadParam("RCCL_IB_OPTIONAL_RECV_COMPLETION", 0); }

static ncclResult_t DefaultIbCastStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName) {
  if (ncclParamIbCastAsyncEvents() && COMPILER_ATOMIC_LOAD(&stat->fatalErrorCount, std::memory_order_relaxed)) {
    WARN("RCCL encountered a communication fatal error (detected in %s)", funcName);
    return ncclSystemError;
  }
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclIbStats*, const char*)> g_ibCastStatsCheckFatalCount =
    DefaultIbCastStatsCheckFatalCount;

static struct ncclIbNetCommDevBase* DefaultIbCastGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  if (base->isSend) return &((struct ncclIbSendComm*)base)->devs[devIndex].base;
  return &((struct ncclIbRecvComm*)base)->devs[devIndex].base;
}
std::function<struct ncclIbNetCommDevBase*(ncclIbNetCommBase*, int)> g_ibCastGetNetCommDevBase =
    DefaultIbCastGetNetCommDevBase;

static ncclResult_t DefaultIbCastBaseCommInit(struct ncclIbNetCommBase*, bool) {
  FailLoudUnfaked("net_ib_cast_common_fakes", "IbCastBaseCommInit");
}
std::function<ncclResult_t(struct ncclIbNetCommBase*, bool)> g_ibCastBaseCommInit = DefaultIbCastBaseCommInit;

static void DefaultIbCastInitOptRecvCompletion(struct ncclIbNetCommBase*, bool) {
  FailLoudUnfaked("net_ib_cast_common_fakes", "IbCastInitOptRecvCompletion");
}
std::function<void(struct ncclIbNetCommBase*, bool)> g_ibCastInitOptRecvCompletion =
    DefaultIbCastInitOptRecvCompletion;

static ncclResult_t DefaultIbCastRecvCommInit(struct ncclIbRecvComm*) {
  FailLoudUnfaked("net_ib_cast_common_fakes", "IbCastRecvCommInit");
}
std::function<ncclResult_t(struct ncclIbRecvComm*)> g_ibCastRecvCommInit = DefaultIbCastRecvCommInit;

static ncclResult_t DefaultIbCastSendCommInit(struct ncclIbSendComm*) {
  FailLoudUnfaked("net_ib_cast_common_fakes", "IbCastSendCommInit");
}
std::function<ncclResult_t(struct ncclIbSendComm*)> g_ibCastSendCommInit = DefaultIbCastSendCommInit;

static void* DefaultIbCastAsyncThreadMain(void*) {
  FailLoudUnfaked("net_ib_cast_common_fakes", "IbCastAsyncThreadMain");
}
std::function<void*(void*)> g_ibCastAsyncThreadMain = DefaultIbCastAsyncThreadMain;

ASSERT_HOOK_MATCHES_PROD(g_ibCastStatsCheckFatalCount, IbCastStatsCheckFatalCount);
ASSERT_HOOK_MATCHES_PROD(g_ibCastGetNetCommDevBase, IbCastGetNetCommDevBase);
ASSERT_HOOK_MATCHES_PROD(g_ibCastBaseCommInit, IbCastBaseCommInit);
ASSERT_HOOK_MATCHES_PROD(g_ibCastInitOptRecvCompletion, IbCastInitOptRecvCompletion);
ASSERT_HOOK_MATCHES_PROD(g_ibCastRecvCommInit, IbCastRecvCommInit);
ASSERT_HOOK_MATCHES_PROD(g_ibCastSendCommInit, IbCastSendCommInit);
ASSERT_HOOK_MATCHES_PROD(g_ibCastAsyncThreadMain, IbCastAsyncThreadMain);

ncclResult_t IbCastStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName) {
  return g_ibCastStatsCheckFatalCount(stat, funcName);
}
struct ncclIbNetCommDevBase* IbCastGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  return g_ibCastGetNetCommDevBase(base, devIndex);
}
ncclResult_t IbCastBaseCommInit(struct ncclIbNetCommBase* baseComm, bool isSend) {
  return g_ibCastBaseCommInit(baseComm, isSend);
}
void IbCastInitOptRecvCompletion(struct ncclIbNetCommBase* baseComm, bool useCtsOffload) {
  g_ibCastInitOptRecvCompletion(baseComm, useCtsOffload);
}
ncclResult_t IbCastRecvCommInit(struct ncclIbRecvComm* recvComm) { return g_ibCastRecvCommInit(recvComm); }
ncclResult_t IbCastSendCommInit(struct ncclIbSendComm* sendComm) { return g_ibCastSendCommInit(sendComm); }
void* IbCastAsyncThreadMain(void* args) { return g_ibCastAsyncThreadMain(args); }

void ResetNetIbCastCommonFakes() {
  g_ibCastStatsCheckFatalCount = DefaultIbCastStatsCheckFatalCount;
  g_ibCastGetNetCommDevBase = DefaultIbCastGetNetCommDevBase;
  g_ibCastBaseCommInit = DefaultIbCastBaseCommInit;
  g_ibCastInitOptRecvCompletion = DefaultIbCastInitOptRecvCompletion;
  g_ibCastRecvCommInit = DefaultIbCastRecvCommInit;
  g_ibCastSendCommInit = DefaultIbCastSendCommInit;
  g_ibCastAsyncThreadMain = DefaultIbCastAsyncThreadMain;
}
