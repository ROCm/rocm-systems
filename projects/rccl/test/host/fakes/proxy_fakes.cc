/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See proxy_fakes.h.

#include "proxy_fakes.h"

#include "comm.h"
#include "proxy.h"

#include "fail_loud.h"

ncclResult_t g_proxySaveOpResult = ncclSuccess;
int g_proxySaveOpCalls = 0;
bool g_proxySaveOpJustInquire = false;
struct ncclComm* g_proxySaveOpLastComm = nullptr;
struct ncclProxyOp* g_proxySaveOpLastOp = nullptr;
int g_proxySaveOpLastChannelId = -1;
uint64_t g_proxySaveOpLastOpCount = 0;
bool g_proxySaveOpSawNonNullJustInquire = false;

// Records its arguments. Without this, a caller could hand the transport the
// wrong communicator or a different op and every proxy test would still pass.
ncclResult_t ncclProxySaveOp(struct ncclComm* comm, struct ncclProxyOp* op, bool* justInquire) {
  ++g_proxySaveOpCalls;
  g_proxySaveOpLastComm = comm;
  g_proxySaveOpLastOp = op;
  // Production always passes a real op and a real out-param; a null here is a
  // defect in the caller, not a case worth tolerating silently.
  if (op == nullptr) {
    FailLoud("proxy_fakes", "ncclProxySaveOp called with a NULL op");
  }
  if (justInquire == nullptr) {
    FailLoud("proxy_fakes", "ncclProxySaveOp called with a NULL justInquire");
  }
  g_proxySaveOpLastChannelId = op->channelId;
  g_proxySaveOpLastOpCount = op->opCount;
  g_proxySaveOpSawNonNullJustInquire = true;  // null already aborted above
  // Mirror production: overwrite the pointee before deciding (proxy.cc:631).
  *justInquire = g_proxySaveOpJustInquire;
  return g_proxySaveOpResult;
}

ncclResult_t g_proxyStartResult = ncclSuccess;
ncclResult_t ncclProxyStart(struct ncclComm*) { return g_proxyStartResult; }

void ResetProxyFakes() {
  g_proxySaveOpResult = ncclSuccess;
  g_proxySaveOpCalls = 0;
  g_proxySaveOpJustInquire = false;
  g_proxySaveOpLastComm = nullptr;
  g_proxySaveOpLastOp = nullptr;
  g_proxySaveOpLastChannelId = -1;
  g_proxySaveOpLastOpCount = 0;
  g_proxySaveOpSawNonNullJustInquire = false;
  g_proxyStartResult = ncclSuccess;
}
