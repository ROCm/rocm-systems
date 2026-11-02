/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for src/transport/net_socket.cc.

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_SOCKET_FAKES_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_SOCKET_FAKES_H_

#include <functional>

#include "net.h"

extern std::function<ncclResult_t(int, ncclNetProperties_t*)> g_ncclNetSocketGetProperties;

void ResetTransportNetSocketFakes();

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_SOCKET_FAKES_H_
