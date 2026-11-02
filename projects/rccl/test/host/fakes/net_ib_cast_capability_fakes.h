/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seam for src/transport/net_ib_cast/capability.cc, for binaries that do not compile it.
// Fails loud by default.

#ifndef RCCL_TEST_HOST_NET_IB_CAST_CAPABILITY_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_CAPABILITY_FAKES_H_

#include <functional>

struct ncclIbDev;

extern std::function<bool(const struct ncclIbDev*)> g_ibCastCapUdSupported;

void ResetNetIbCastCapabilityFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_CAPABILITY_FAKES_H_
