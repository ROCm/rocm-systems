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

#include "fail_loud.h"
#include "register_stubs.h"

// First call in ncclTasksRegAndEnqueue's task loop, so the rest of that loop is
// unreachable while it aborts. Hookable seam, defaulting to the fail-loud floor.
static ncclResult_t DefaultRegisterCollBuffers(struct ncclComm*, struct ncclTaskColl*, void**,
                                              void**,
                                              struct ncclIntruQueue<struct ncclCommCallback,
                                                                    &ncclCommCallback::next>*,
                                              bool*) {
  FailLoudUnfaked("register_stubs", "ncclRegisterCollBuffers");
}

ncclRegisterCollBuffersFn g_ncclRegisterCollBuffers = DefaultRegisterCollBuffers;

void ResetRegisterStubs() { g_ncclRegisterCollBuffers = DefaultRegisterCollBuffers; }

ncclResult_t ncclRegisterCollBuffers(struct ncclComm* comm, struct ncclTaskColl* task,
                                     void** regBufSend, void** regBufRecv,
                                     struct ncclIntruQueue<struct ncclCommCallback,
                                                           &ncclCommCallback::next>* cleanupQueue,
                                     bool* needConnect) {
  return g_ncclRegisterCollBuffers(comm, task, regBufSend, regBufRecv, cleanupQueue, needConnect);
}
ncclResult_t ncclRegisterCollNvlsBuffers(struct ncclComm*, struct ncclTaskColl*, void**, void**,
                                         struct ncclIntruQueue<struct ncclCommCallback,
                                                               &ncclCommCallback::next>*,
                                         bool*) {
  FailLoudUnfaked("register_stubs", "ncclRegisterCollNvlsBuffers");
}
ncclResult_t ncclRegisterP2pIpcBuffer(struct ncclComm*, void*, size_t, int, int*, void**,
                                      struct ncclIntruQueue<struct ncclCommCallback,
                                                            &ncclCommCallback::next>*) {
  FailLoudUnfaked("register_stubs", "ncclRegisterP2pIpcBuffer");
}
ncclResult_t ncclRegisterP2pNetBuffer(struct ncclComm*, void*, size_t, struct ncclConnector*,
                                      int*, void**,
                                      struct ncclIntruQueue<struct ncclCommCallback,
                                                            &ncclCommCallback::next>*) {
  FailLoudUnfaked("register_stubs", "ncclRegisterP2pNetBuffer");
}
