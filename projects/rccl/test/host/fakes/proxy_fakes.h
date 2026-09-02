/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for src/proxy.cc, shared by the host-only microtest binaries.

#ifndef RCCL_TEST_HOST_PROXY_FAKES_H_
#define RCCL_TEST_HOST_PROXY_FAKES_H_

#include <cstdint>

#include "nccl.h"

struct ncclComm;
struct ncclProxyOp;

// ncclProxySaveOp (proxy.cc:627) uses a "justInquire" protocol: the caller asks
// whether the op matters BEFORE allocating for it, so the out-param -- not the
// return code -- is what decides whether the caller enqueues.
extern ncclResult_t g_proxySaveOpResult;
extern int g_proxySaveOpCalls;
// The justInquire out-param: true means "this op matters, allocate for it".
extern bool g_proxySaveOpJustInquire;
// Recorded arguments. A caller's queued copy is made from the CALLER's op, so
// asserting on the queue cannot show WHICH op the transport was actually asked
// about -- these can. The fake aborts on a null op or null justInquire.
extern struct ncclComm* g_proxySaveOpLastComm;
extern struct ncclProxyOp* g_proxySaveOpLastOp;
extern int g_proxySaveOpLastChannelId;
extern uint64_t g_proxySaveOpLastOpCount;
// What signals inquiry is a NON-NULL justInquire pointer, not its pointee: real
// ncclProxySaveOp overwrites the pointee with false as its first statement
// (proxy.cc:631), so the caller's seed is never read and recording it would let a
// test pin this fake instead of production.
extern bool g_proxySaveOpSawNonNullJustInquire;

extern ncclResult_t g_proxyStartResult;  // UNDRIVEN

void ResetProxyFakes();

#endif  // RCCL_TEST_HOST_PROXY_FAKES_H_
