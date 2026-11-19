/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the src/transport/net_ib/connect.cc entry points that src/transport/net.cc calls: the P2P QP
// policy and the GDR flush-flag accessors. Targets that compile the real connect.cc leave this file out.

#ifndef RCCL_TEST_HOST_FAKES_TRANSPORT_NET_IB_CONNECT_NET_FAKES_H_
#define RCCL_TEST_HOST_FAKES_TRANSPORT_NET_IB_CONNECT_NET_FAKES_H_

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "nccl.h"

// rcclNetP2pPolicy(handle, isP2p). Defaults to fail-loud.
extern std::function<ncclResult_t(void* /*handle*/, int /*isP2p*/)> g_rcclNetP2pPolicy;
// Every ncclIbSetListenGdrFlushFlag(listenComm, enable) call, in order.
extern std::vector<std::pair<void*, int>> g_ibSetListenGdrFlushFlagCalls;
// ncclIbGetGpuFlushFlag(recvComm): NULL (the connection keeps the RDMA read flush) by default.
extern std::function<uint64_t*(void*)> g_ibGetGpuFlushFlag;
// ncclIbGetGpuFlushFlagSeq(recvComm): 0 by default.
extern std::function<uint64_t(void*)> g_ibGetGpuFlushFlagSeq;
// Every ncclIbSetGpuFlushFlagHint(recvComm, need) call, in order.
extern std::vector<std::pair<void*, int>> g_ibSetGpuFlushFlagHintCalls;

void ResetTransportNetIbConnectNetFakes();

#endif  // RCCL_TEST_HOST_FAKES_TRANSPORT_NET_IB_CONNECT_NET_FAKES_H_
