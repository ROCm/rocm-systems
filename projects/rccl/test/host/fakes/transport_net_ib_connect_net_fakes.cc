/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of transport_net_ib_connect_net_fakes.h. Targets that compile the real
// net_ib/connect.cc must not link this file.

#include "transport_net_ib_connect_net_fakes.h"

#include "fakes/fail_loud.h"
#include "fakes/signature-drift.h"
#include "net.h"

static ncclResult_t DefaultNetP2pPolicy(void*, int) {
  FailLoudUnfaked("transport_net_ib_connect_net_fakes", "rcclNetP2pPolicy");
}
std::function<ncclResult_t(void*, int)> g_rcclNetP2pPolicy = DefaultNetP2pPolicy;
ncclResult_t rcclNetP2pPolicy(void* handle, int isP2p) { return g_rcclNetP2pPolicy(handle, isP2p); }

std::vector<std::pair<void*, int>> g_ibSetListenGdrFlushFlagCalls;
void ncclIbSetListenGdrFlushFlag(void* listenComm, int enable) {
  g_ibSetListenGdrFlushFlagCalls.emplace_back(listenComm, enable);
}

static uint64_t* DefaultIbGetGpuFlushFlag(void*) { return nullptr; }
std::function<uint64_t*(void*)> g_ibGetGpuFlushFlag = DefaultIbGetGpuFlushFlag;
uint64_t* ncclIbGetGpuFlushFlag(void* recvComm) { return g_ibGetGpuFlushFlag(recvComm); }

static uint64_t DefaultIbGetGpuFlushFlagSeq(void*) { return 0; }
std::function<uint64_t(void*)> g_ibGetGpuFlushFlagSeq = DefaultIbGetGpuFlushFlagSeq;
uint64_t ncclIbGetGpuFlushFlagSeq(void* recvComm) { return g_ibGetGpuFlushFlagSeq(recvComm); }

std::vector<std::pair<void*, int>> g_ibSetGpuFlushFlagHintCalls;
void ncclIbSetGpuFlushFlagHint(void* recvComm, int need) { g_ibSetGpuFlushFlagHintCalls.emplace_back(recvComm, need); }

ASSERT_HOOK_MATCHES_PROD(g_rcclNetP2pPolicy, rcclNetP2pPolicy);
ASSERT_HOOK_MATCHES_PROD(g_ibGetGpuFlushFlag, ncclIbGetGpuFlushFlag);
ASSERT_HOOK_MATCHES_PROD(g_ibGetGpuFlushFlagSeq, ncclIbGetGpuFlushFlagSeq);
#undef ASSERT_HOOK_MATCHES_PROD

void ResetTransportNetIbConnectNetFakes() {
  g_rcclNetP2pPolicy = DefaultNetP2pPolicy;
  g_ibSetListenGdrFlushFlagCalls.clear();
  g_ibGetGpuFlushFlag = DefaultIbGetGpuFlushFlag;
  g_ibGetGpuFlushFlagSeq = DefaultIbGetGpuFlushFlagSeq;
  g_ibSetGpuFlushFlagHintCalls.clear();
}
