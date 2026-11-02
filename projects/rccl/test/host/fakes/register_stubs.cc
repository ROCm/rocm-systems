/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for src/register/coll_reg.cc and src/register/sendrecv_reg.cc.
// A host-only microtest that reaches buffer registration is broken, not merely
// unexercised, so reaching one aborts. A test that needs to drive one replaces
// that individual entry with a real fake.

#include "comm.h"
#include "nccl.h"
#include "transport.h"

#include "fail_loud.h"
#include "register_stubs.h"
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_ncclRegisterCollBuffers, ncclRegisterCollBuffers);
ASSERT_HOOK_MATCHES_PROD(g_ncclRegisterCollNvlsBuffers, ncclRegisterCollNvlsBuffers);
ASSERT_HOOK_MATCHES_PROD(g_ncclRegisterP2pIpcBuffer, ncclRegisterP2pIpcBuffer);
ASSERT_HOOK_MATCHES_PROD(g_ncclRegisterP2pNetBuffer, ncclRegisterP2pNetBuffer);
#undef ASSERT_HOOK_MATCHES_PROD

// First call in ncclTasksRegAndEnqueue's task loop, so the rest of that loop is
// unreachable while it aborts. Hookable seam, defaulting to the fail-loud floor.
static ncclResult_t DefaultRegisterCollBuffers(struct ncclComm*, struct ncclTaskColl*, void**,
                                              void**, ncclCommCallbackQueue*, bool*) {
  FailLoudUnfaked("register_stubs", "ncclRegisterCollBuffers");
}

ncclRegisterCollBuffersFn g_ncclRegisterCollBuffers = DefaultRegisterCollBuffers;

static ncclResult_t DefaultRegisterCollNvlsBuffers(struct ncclComm*, struct ncclTaskColl*, void**,
                                                   void**, ncclCommCallbackQueue*, bool*) {
  FailLoudUnfaked("register_stubs", "ncclRegisterCollNvlsBuffers");
}

ncclRegisterCollBuffersFn g_ncclRegisterCollNvlsBuffers = DefaultRegisterCollNvlsBuffers;

static ncclResult_t DefaultRegisterP2pIpcBuffer(struct ncclComm*, void*, size_t, int, int*, void**,
                                                ncclCommCallbackQueue*) {
  FailLoudUnfaked("register_stubs", "ncclRegisterP2pIpcBuffer");
}

ncclRegisterP2pIpcBufferFn g_ncclRegisterP2pIpcBuffer = DefaultRegisterP2pIpcBuffer;

static ncclResult_t DefaultRegisterP2pNetBuffer(struct ncclComm*, void*, size_t,
                                                struct ncclConnector*, int*, void**,
                                                ncclCommCallbackQueue*) {
  FailLoudUnfaked("register_stubs", "ncclRegisterP2pNetBuffer");
}

ncclRegisterP2pNetBufferFn g_ncclRegisterP2pNetBuffer = DefaultRegisterP2pNetBuffer;

void ResetRegisterStubs() {
  g_ncclRegisterCollBuffers = DefaultRegisterCollBuffers;
  g_ncclRegisterCollNvlsBuffers = DefaultRegisterCollNvlsBuffers;
  g_ncclRegisterP2pIpcBuffer = DefaultRegisterP2pIpcBuffer;
  g_ncclRegisterP2pNetBuffer = DefaultRegisterP2pNetBuffer;
}

ncclResult_t ncclRegisterCollBuffers(struct ncclComm* comm, struct ncclTaskColl* task,
                                     void** regBufSend, void** regBufRecv,
                                     ncclCommCallbackQueue* cleanupQueue, bool* needConnect) {
  return g_ncclRegisterCollBuffers(comm, task, regBufSend, regBufRecv, cleanupQueue, needConnect);
}
ncclResult_t ncclRegisterCollNvlsBuffers(struct ncclComm* comm, struct ncclTaskColl* task,
                                         void** regBufSend, void** regBufRecv,
                                         ncclCommCallbackQueue* cleanupQueue, bool* needConnect) {
  return g_ncclRegisterCollNvlsBuffers(comm, task, regBufSend, regBufRecv, cleanupQueue,
                                       needConnect);
}
ncclResult_t ncclRegisterP2pIpcBuffer(struct ncclComm* comm, void* buff, size_t size, int peer,
                                      int* regFlag, void** regAddr,
                                      ncclCommCallbackQueue* cleanupQueue) {
  return g_ncclRegisterP2pIpcBuffer(comm, buff, size, peer, regFlag, regAddr, cleanupQueue);
}
ncclResult_t ncclRegisterP2pNetBuffer(struct ncclComm* comm, void* buff, size_t size,
                                      struct ncclConnector* conn, int* regFlag, void** handle,
                                      ncclCommCallbackQueue* cleanupQueue) {
  return g_ncclRegisterP2pNetBuffer(comm, buff, size, conn, regFlag, handle, cleanupQueue);
}
