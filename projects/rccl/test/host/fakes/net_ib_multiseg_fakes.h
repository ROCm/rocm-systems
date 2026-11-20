/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the src/transport/net_ib_multiseg entry points that src/init.cc and src/transport/net.cc use: the
// plugin object they compare against, the P2P QP policy and the GDR flush flag. Targets that compile the real
// fork leave this file out.

#ifndef RCCL_TEST_HOST_FAKES_NET_IB_MULTISEG_FAKES_H_
#define RCCL_TEST_HOST_FAKES_NET_IB_MULTISEG_FAKES_H_

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "nccl.h"

// IbMsegNetP2pPolicy(handle, isP2p). Defaults to fail-loud.
extern std::function<ncclResult_t(void* /*handle*/, int /*isP2p*/)> g_ibMsegNetP2pPolicy;
// IbMsegGdrFlushFlagLocalSupport() returns this; false by default.
extern bool g_ibMsegGdrFlushFlagLocalSupport;
// Every IbMsegSetGdrFlushFlag(netContext, enable) call, in order.
extern std::vector<std::pair<void*, int>> g_ibMsegSetGdrFlushFlagCalls;
// Every IbMsegSetListenGdrFlushFlag(listenComm, enable) call, in order.
extern std::vector<std::pair<void*, int>> g_ibMsegSetListenGdrFlushFlagCalls;
// IbMsegGetGpuFlushFlag(recvComm): NULL (the connection keeps the RDMA read flush) by default.
extern std::function<uint64_t*(void*)> g_ibMsegGetGpuFlushFlag;
// IbMsegGetGpuFlushFlagSeq(recvComm): 0 by default.
extern std::function<uint64_t(void*)> g_ibMsegGetGpuFlushFlagSeq;
// Every IbMsegSetGpuFlushFlagHint(recvComm, need) call, in order.
extern std::vector<std::pair<void*, int>> g_ibMsegSetGpuFlushFlagHintCalls;

void ResetNetIbMultisegFakes();

#endif  // RCCL_TEST_HOST_FAKES_NET_IB_MULTISEG_FAKES_H_
