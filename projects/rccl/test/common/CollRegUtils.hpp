/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <comm.h>
#include <transport.h>
#include "TestBed.hpp"

ncclResult_t registerCheckP2PConnection(struct ncclComm* comm, struct ncclConnector* conn, struct ncclTopoGraph* graph, int peer, bool* needReg);

// Global mock transport definition (outside any namespace)
static ncclResult_t mockCanConnect(int* result, struct ncclComm*, struct ncclTopoGraph*,
  struct ncclPeerInfo*, struct ncclPeerInfo*) {
*result = 1;
return ncclSuccess;
}

static ncclResult_t mockCanConnectFalse(int* result, struct ncclComm*, struct ncclTopoGraph*,
  struct ncclPeerInfo*, struct ncclPeerInfo*) {
*result = 0;  // Simulate failure to connect
return ncclSuccess;
}

static struct ncclTransport mockTransport = {
.canConnect = mockCanConnect,
};

struct ncclTransport* ncclTransports[] = {
&mockTransport,
nullptr
};

struct ScopedCanConnectOverride {
  struct ncclTransport* transport;
  ncclResult_t (*originalFn)(int*, struct ncclComm*, struct ncclTopoGraph*,
                             struct ncclPeerInfo*, struct ncclPeerInfo*);

  ScopedCanConnectOverride(struct ncclTransport* t,
                           ncclResult_t (*mockFn)(int*, struct ncclComm*, struct ncclTopoGraph*,
                                                  struct ncclPeerInfo*, struct ncclPeerInfo*))
      : transport(t), originalFn(t->canConnect) {
    t->canConnect = mockFn;
  }

  ~ScopedCanConnectOverride() {
    transport->canConnect = originalFn;
  }
};
