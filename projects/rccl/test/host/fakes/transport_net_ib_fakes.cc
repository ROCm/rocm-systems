/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Symbols owned by src/transport/net_ib/connect.cc, for the microtest binaries that do not compile it.

#include "transport_net_ib_fakes.h"

#include <cstdint>

#include "fakes/nccl_fakes.h"
#include "fakes/signature-drift.h"
#include "net.h"
#include "transport/net_ib/common.h"

int64_t ncclParamIbQpsPerConn() { return g_loadParam("IB_QPS_PER_CONNECTION", 1); }

// src/transport/net_ib/common.cc. Only its address matters: init.cc compares comm->ncclNet against it.
ncclNet_t ncclNetIb{};

bool g_ibGdrFlushFlagLocalSupport = false;
bool ncclIbGdrFlushFlagLocalSupport() { return g_ibGdrFlushFlagLocalSupport; }

std::vector<std::pair<void*, int>> g_ibSetGdrFlushFlagCalls;
void ncclIbSetGdrFlushFlag(void* netContext, int enable) { g_ibSetGdrFlushFlagCalls.emplace_back(netContext, enable); }

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

static ncclResult_t DefaultIbGidInfoQuery(struct ibv_context*, uint8_t, struct ibv_port_attr*,
                                          struct ncclIbGidInfo*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_context*, uint8_t, struct ibv_port_attr*, struct ncclIbGidInfo*)>
    g_ncclIbGidInfoQuery = DefaultIbGidInfoQuery;
ASSERT_HOOK_MATCHES_PROD(g_ibGetGpuFlushFlag, ncclIbGetGpuFlushFlag);
ASSERT_HOOK_MATCHES_PROD(g_ibGetGpuFlushFlagSeq, ncclIbGetGpuFlushFlagSeq);
ASSERT_HOOK_MATCHES_PROD(g_ncclIbGidInfoQuery, ncclIbGidInfoQuery);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t ncclIbGidInfoQuery(struct ibv_context* context, uint8_t port, struct ibv_port_attr* attr,
                                struct ncclIbGidInfo* info) {
  return g_ncclIbGidInfoQuery(context, port, attr, info);
}

void ResetTransportNetIbFakes() {
  g_ibGdrFlushFlagLocalSupport = false;
  g_ibSetGdrFlushFlagCalls.clear();
  g_ibSetListenGdrFlushFlagCalls.clear();
  g_ibGetGpuFlushFlag = DefaultIbGetGpuFlushFlag;
  g_ibGetGpuFlushFlagSeq = DefaultIbGetGpuFlushFlagSeq;
  g_ibSetGpuFlushFlagHintCalls.clear();
  g_ncclIbGidInfoQuery = DefaultIbGidInfoQuery;
}
