/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for src/transport/net_socket/common.cc.

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_SOCKET_COMMON_FAKES_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_SOCKET_COMMON_FAKES_H_

#include <functional>

#include "transport/net_socket/common.h"

extern std::function<ncclResult_t(const char*)> g_ncclNetSocketInitDevices;
extern std::function<ncclResult_t(const char*, int, struct ncclSocket*, union ncclSocketAddress*, uint64_t*)>
    g_ncclNetSocketCreateListener;

void ResetTransportNetSocketCommonFakes();

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_SOCKET_COMMON_FAKES_H_
