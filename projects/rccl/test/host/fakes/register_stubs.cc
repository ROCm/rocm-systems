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

namespace {
[[noreturn]] void Unreached(const char* fn) { FailLoud("register_stubs", fn); }
}  // namespace

ncclResult_t ncclRegisterCollBuffers(struct ncclComm*, struct ncclTaskColl*, void**, void**,
                                     struct ncclIntruQueue<struct ncclCommCallback,
                                                           &ncclCommCallback::next>*,
                                     bool*) {
  Unreached("ncclRegisterCollBuffers");
}
ncclResult_t ncclRegisterCollNvlsBuffers(struct ncclComm*, struct ncclTaskColl*, void**, void**,
                                         struct ncclIntruQueue<struct ncclCommCallback,
                                                               &ncclCommCallback::next>*,
                                         bool*) {
  Unreached("ncclRegisterCollNvlsBuffers");
}
ncclResult_t ncclRegisterP2pIpcBuffer(struct ncclComm*, void*, size_t, int, int*, void**,
                                      struct ncclIntruQueue<struct ncclCommCallback,
                                                            &ncclCommCallback::next>*) {
  Unreached("ncclRegisterP2pIpcBuffer");
}
ncclResult_t ncclRegisterP2pNetBuffer(struct ncclComm*, void*, size_t, struct ncclConnector*,
                                      int*, void**,
                                      struct ncclIntruQueue<struct ncclCommCallback,
                                                            &ncclCommCallback::next>*) {
  Unreached("ncclRegisterP2pNetBuffer");
}
