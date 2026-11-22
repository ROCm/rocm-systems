/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of net_ib_multiseg_fakes.h. Targets that compile the real net_ib_multiseg must not
// link this file.

#include "net_ib_multiseg_fakes.h"

#include "fakes/fail_loud.h"
#include "fakes/signature-drift.h"
#include "net.h"
#include "nccl_rma.h"

// init.cc and net.cc compare a plugin's address against it.
ncclNet_t ncclNetIbMseg = {"IB-MSEG"};
// init.cc compares the comm's RMA backend against it.
ncclRma_t IbMsegRmaIbProxy{};

static ncclResult_t DefaultIbMsegNetP2pPolicy(void*, int) {
  FailLoudUnfaked("net_ib_multiseg_fakes", "IbMsegNetP2pPolicy");
}
std::function<ncclResult_t(void*, int)> g_ibMsegNetP2pPolicy = DefaultIbMsegNetP2pPolicy;
ncclResult_t IbMsegNetP2pPolicy(void* handle, int isP2p) { return g_ibMsegNetP2pPolicy(handle, isP2p); }

bool g_ibMsegGdrFlushFlagLocalSupport = false;
bool IbMsegGdrFlushFlagLocalSupport() { return g_ibMsegGdrFlushFlagLocalSupport; }

std::vector<std::pair<void*, int>> g_ibMsegSetGdrFlushFlagCalls;
void IbMsegSetGdrFlushFlag(void* netContext, int enable) {
  g_ibMsegSetGdrFlushFlagCalls.emplace_back(netContext, enable);
}

std::vector<std::pair<void*, int>> g_ibMsegSetListenGdrFlushFlagCalls;
void IbMsegSetListenGdrFlushFlag(void* listenComm, int enable) {
  g_ibMsegSetListenGdrFlushFlagCalls.emplace_back(listenComm, enable);
}

static uint64_t* DefaultIbMsegGetGpuFlushFlag(void*) { return nullptr; }
std::function<uint64_t*(void*)> g_ibMsegGetGpuFlushFlag = DefaultIbMsegGetGpuFlushFlag;
uint64_t* IbMsegGetGpuFlushFlag(void* recvComm) { return g_ibMsegGetGpuFlushFlag(recvComm); }

static uint64_t DefaultIbMsegGetGpuFlushFlagSeq(void*) { return 0; }
std::function<uint64_t(void*)> g_ibMsegGetGpuFlushFlagSeq = DefaultIbMsegGetGpuFlushFlagSeq;
uint64_t IbMsegGetGpuFlushFlagSeq(void* recvComm) { return g_ibMsegGetGpuFlushFlagSeq(recvComm); }

std::vector<std::pair<void*, int>> g_ibMsegSetGpuFlushFlagHintCalls;
void IbMsegSetGpuFlushFlagHint(void* recvComm, int need) {
  g_ibMsegSetGpuFlushFlagHintCalls.emplace_back(recvComm, need);
}

ASSERT_HOOK_MATCHES_PROD(g_ibMsegNetP2pPolicy, IbMsegNetP2pPolicy);
ASSERT_HOOK_MATCHES_PROD(g_ibMsegGetGpuFlushFlag, IbMsegGetGpuFlushFlag);
ASSERT_HOOK_MATCHES_PROD(g_ibMsegGetGpuFlushFlagSeq, IbMsegGetGpuFlushFlagSeq);
#undef ASSERT_HOOK_MATCHES_PROD

void ResetNetIbMultisegFakes() {
  g_ibMsegNetP2pPolicy = DefaultIbMsegNetP2pPolicy;
  g_ibMsegGdrFlushFlagLocalSupport = false;
  g_ibMsegSetGdrFlushFlagCalls.clear();
  g_ibMsegSetListenGdrFlushFlagCalls.clear();
  g_ibMsegGetGpuFlushFlag = DefaultIbMsegGetGpuFlushFlag;
  g_ibMsegGetGpuFlushFlagSeq = DefaultIbMsegGetGpuFlushFlagSeq;
  g_ibMsegSetGpuFlushFlagHintCalls.clear();
}
