/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_SUPPORT_SEAMS_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_SUPPORT_SEAMS_H_

#include <functional>

#include "nccl.h"

// src/transport/net_ib/gdr.cc's capability probes, for the net_ib units that call them. gdr-test.cc compiles the
// real gdr.cc into the same binary, so a unit under test renames its calls onto these hooks before it is included:
//   #define ncclIbGdrSupport NetIbTestGdrSupport
//   #define ncclIbPeerMemSupport NetIbTestPeerMemSupport
//   #define ncclIbDmaBufSupport NetIbTestDmaBufSupport
// Each reports no support until a test installs an answer.
inline std::function<ncclResult_t()> g_netIbGdrSupport = [] { return ncclSystemError; };
inline std::function<ncclResult_t()> g_netIbPeerMemSupport = [] { return ncclSystemError; };
inline std::function<ncclResult_t(int)> g_netIbDmaBufSupport = [](int) { return ncclSystemError; };

inline ncclResult_t NetIbTestGdrSupport() { return g_netIbGdrSupport(); }
inline ncclResult_t NetIbTestPeerMemSupport() { return g_netIbPeerMemSupport(); }
inline ncclResult_t NetIbTestDmaBufSupport(int dev) { return g_netIbDmaBufSupport(dev); }

inline void ResetNetIbSupportSeams() {
  g_netIbGdrSupport = [] { return ncclSystemError; };
  g_netIbPeerMemSupport = [] { return ncclSystemError; };
  g_netIbDmaBufSupport = [](int) { return ncclSystemError; };
}

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_NET_IB_SUPPORT_SEAMS_H_
