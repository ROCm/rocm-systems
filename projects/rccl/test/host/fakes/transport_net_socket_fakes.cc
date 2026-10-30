/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "transport_net_socket_fakes.h"

#include "fail_loud.h"

static ncclResult_t DefaultNetSocketGetProperties(int, ncclNetProperties_t*) {
  FailLoudUnfaked("transport_net_socket_fakes", "ncclNetSocket.getProperties");
}

std::function<ncclResult_t(int, ncclNetProperties_t*)> g_ncclNetSocketGetProperties =
    DefaultNetSocketGetProperties;

static ncclResult_t NetSocketGetProperties(int dev, ncclNetProperties_t* props) {
  return g_ncclNetSocketGetProperties(dev, props);
}

ncclNet_t ncclNetSocket = {
    "Socket",
    nullptr,
    nullptr,
    NetSocketGetProperties,
};

void ResetTransportNetSocketFakes() { g_ncclNetSocketGetProperties = DefaultNetSocketGetProperties; }
