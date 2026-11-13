/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "transport_net_socket_common_fakes.h"

#include "fail_loud.h"
#include "signature-drift.h"

int ncclNetIfs = -1;

static ncclResult_t DefaultNetSocketInitDevices(const char*) {
  FailLoudUnfaked("transport_net_socket_common_fakes", "ncclNetSocketInitDevices");
}

static ncclResult_t DefaultNetSocketCreateListener(const char*, int, struct ncclSocket*,
                                                    union ncclSocketAddress*, uint64_t*) {
  FailLoudUnfaked("transport_net_socket_common_fakes", "ncclNetSocketCreateListener");
}

std::function<ncclResult_t(const char*)> g_ncclNetSocketInitDevices = DefaultNetSocketInitDevices;
std::function<ncclResult_t(const char*, int, struct ncclSocket*, union ncclSocketAddress*, uint64_t*)>
    g_ncclNetSocketCreateListener = DefaultNetSocketCreateListener;

ASSERT_HOOK_MATCHES_PROD(g_ncclNetSocketInitDevices, ncclNetSocketInitDevices);
ASSERT_HOOK_MATCHES_PROD(g_ncclNetSocketCreateListener, ncclNetSocketCreateListener);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t ncclNetSocketInitDevices(const char* logPrefix) {
  return g_ncclNetSocketInitDevices(logPrefix);
}

ncclResult_t ncclNetSocketCreateListener(const char* logPrefix, int dev, struct ncclSocket* sock,
                                         union ncclSocketAddress* connectAddr, uint64_t* magic) {
  return g_ncclNetSocketCreateListener(logPrefix, dev, sock, connectAddr, magic);
}

void ResetTransportNetSocketCommonFakes() {
  ncclNetIfs = -1;
  g_ncclNetSocketInitDevices = DefaultNetSocketInitDevices;
  g_ncclNetSocketCreateListener = DefaultNetSocketCreateListener;
}
