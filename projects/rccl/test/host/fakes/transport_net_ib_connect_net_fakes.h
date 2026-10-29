/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the src/transport/net_ib/connect.cc entry points that src/transport/net.cc calls: the P2P QP
// policy and the GDR flush-flag accessors. The binary also compiles the real connect.cc, so
// transport_net_ib_connect_net_seam.h renames net.cc's calls to the micro_* trampolines below.

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

ncclResult_t micro_rcclNetP2pPolicy(void* handle, int isP2p);
void micro_ncclIbSetListenGdrFlushFlag(void* listenComm, int enable);
uint64_t* micro_ncclIbGetGpuFlushFlag(void* recvComm);
uint64_t micro_ncclIbGetGpuFlushFlagSeq(void* recvComm);
void micro_ncclIbSetGpuFlushFlagHint(void* recvComm, int need);

void ResetTransportNetIbConnectNetFakes();

#endif  // RCCL_TEST_HOST_FAKES_TRANSPORT_NET_IB_CONNECT_NET_FAKES_H_
