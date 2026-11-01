/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only seams for connect.cc's QP state helpers and clockNano, shared by the
 * src/transport/net_ib/p2p_resiliency*.cc tests.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_QP_SEAMS_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_QP_SEAMS_H_

#include <cstdint>
#include <functional>

#include "transport/net_ib/connect.h"

// This binary also compiles the real connect.cc, so a unit that only calls these
// helpers has its calls renamed onto the trampolines below before it is included.
inline constexpr uint64_t kNetIbSeamStartNs = 1000;

inline ncclResult_t NetIbQpStepFails(ncclIbQp*) { return ncclSystemError; }
inline ncclResult_t NetIbQpCreateFails(ncclIbQp*, ncclIbQpCreateAttr*) { return ncclSystemError; }

inline uint64_t g_netIbNowNs = kNetIbSeamStartNs;
inline std::function<ncclResult_t(ncclIbQp*)> g_netIbQpReset = NetIbQpStepFails;
inline std::function<ncclResult_t(ncclIbQp*, ncclIbQpCreateAttr*)> g_netIbQpCreate = NetIbQpCreateFails;
inline std::function<ncclResult_t(ncclIbQp*)> g_netIbQpInit = NetIbQpStepFails;
inline std::function<ncclResult_t(ncclIbQp*)> g_netIbQpRtr = NetIbQpStepFails;
inline std::function<ncclResult_t(ncclIbQp*)> g_netIbQpRts = NetIbQpStepFails;

inline uint64_t NetIbTestClockNano() { return g_netIbNowNs; }
inline ncclResult_t NetIbTestQpReset(ncclIbQp* qp) { return g_netIbQpReset(qp); }
inline ncclResult_t NetIbTestQpCreate(ncclIbQp* qp, ncclIbQpCreateAttr* attr) { return g_netIbQpCreate(qp, attr); }
inline ncclResult_t NetIbTestQpInit(ncclIbQp* qp) { return g_netIbQpInit(qp); }
inline ncclResult_t NetIbTestQpRtr(ncclIbQp* qp) { return g_netIbQpRtr(qp); }
inline ncclResult_t NetIbTestQpRts(ncclIbQp* qp) { return g_netIbQpRts(qp); }

inline void ResetNetIbQpSeams() {
  g_netIbNowNs = kNetIbSeamStartNs;
  g_netIbQpReset = NetIbQpStepFails;
  g_netIbQpCreate = NetIbQpCreateFails;
  g_netIbQpInit = NetIbQpStepFails;
  g_netIbQpRtr = NetIbQpStepFails;
  g_netIbQpRts = NetIbQpStepFails;
}

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_QP_SEAMS_H_
