/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/transport/net_ib_cast/p2p.cc fakes. See
// net_ib_cast_p2p_fakes.h.

#include "net_ib_cast_p2p_fakes.h"

#include "fail_loud.h"
#include "signature-drift.h"

static void DefaultIbCastAddEvent(struct ncclIbRequest*, int) {
  FailLoudUnfaked("net_ib_cast_p2p_fakes", "IbCastAddEvent");
}
std::function<void(struct ncclIbRequest*, int)> g_ibCastAddEvent = DefaultIbCastAddEvent;

static ncclResult_t DefaultIbCastGetRequest(struct ncclIbNetCommBase*, struct ncclIbRequest**) {
  FailLoudUnfaked("net_ib_cast_p2p_fakes", "IbCastGetRequest");
}
std::function<ncclResult_t(struct ncclIbNetCommBase*, struct ncclIbRequest**)>
    g_ibCastGetRequest = DefaultIbCastGetRequest;

static ncclResult_t DefaultIbCastFreeRequest(struct ncclIbRequest*) {
  FailLoudUnfaked("net_ib_cast_p2p_fakes", "IbCastFreeRequest");
}
std::function<ncclResult_t(struct ncclIbRequest*)> g_ibCastFreeRequest = DefaultIbCastFreeRequest;

static ncclResult_t DefaultIbCastIsend(void*, void*, size_t, int, void*, void*, void**) {
  FailLoudUnfaked("net_ib_cast_p2p_fakes", "IbCastIsend");
}
std::function<ncclResult_t(void*, void*, size_t, int, void*, void*, void**)> g_ibCastIsend = DefaultIbCastIsend;

static ncclResult_t DefaultIbCastIrecv(void*, int, void**, size_t*, int*, void**, void**, void**) {
  FailLoudUnfaked("net_ib_cast_p2p_fakes", "IbCastIrecv");
}
std::function<ncclResult_t(void*, int, void**, size_t*, int*, void**, void**, void**)>
    g_ibCastIrecv = DefaultIbCastIrecv;

static ncclResult_t DefaultIbCastIflush(void*, int, void**, int*, void**, void**) {
  FailLoudUnfaked("net_ib_cast_p2p_fakes", "IbCastIflush");
}
std::function<ncclResult_t(void*, int, void**, int*, void**, void**)> g_ibCastIflush = DefaultIbCastIflush;

static ncclResult_t DefaultIbCastTest(void*, int*, int*) {
  FailLoudUnfaked("net_ib_cast_p2p_fakes", "IbCastTest");
}
std::function<ncclResult_t(void*, int*, int*)> g_ibCastTest = DefaultIbCastTest;

ASSERT_HOOK_MATCHES_PROD(g_ibCastAddEvent, IbCastAddEvent);
ASSERT_HOOK_MATCHES_PROD(g_ibCastGetRequest, IbCastGetRequest);
ASSERT_HOOK_MATCHES_PROD(g_ibCastFreeRequest, IbCastFreeRequest);
ASSERT_HOOK_MATCHES_PROD(g_ibCastIsend, IbCastIsend);
ASSERT_HOOK_MATCHES_PROD(g_ibCastIrecv, IbCastIrecv);
ASSERT_HOOK_MATCHES_PROD(g_ibCastIflush, IbCastIflush);
ASSERT_HOOK_MATCHES_PROD(g_ibCastTest, IbCastTest);

void micro_IbCastAddEvent(struct ncclIbRequest* req, int devIndex) { g_ibCastAddEvent(req, devIndex); }
ncclResult_t micro_IbCastGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  return g_ibCastGetRequest(base, req);
}
ncclResult_t micro_IbCastFreeRequest(struct ncclIbRequest* r) { return g_ibCastFreeRequest(r); }
ncclResult_t micro_IbCastIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle,
                               void** request) {
  return g_ibCastIsend(sendComm, data, size, tag, mhandle, phandle, request);
}
ncclResult_t micro_IbCastIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles,
                               void** phandles, void** request) {
  return g_ibCastIrecv(recvComm, n, data, sizes, tags, mhandles, phandles, request);
}
ncclResult_t micro_IbCastIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  return g_ibCastIflush(recvComm, n, data, sizes, mhandles, request);
}
ncclResult_t micro_IbCastTest(void* request, int* done, int* sizes) { return g_ibCastTest(request, done, sizes); }

void ResetNetIbCastP2pFakes() {
  g_ibCastAddEvent = DefaultIbCastAddEvent;
  g_ibCastGetRequest = DefaultIbCastGetRequest;
  g_ibCastFreeRequest = DefaultIbCastFreeRequest;
  g_ibCastIsend = DefaultIbCastIsend;
  g_ibCastIrecv = DefaultIbCastIrecv;
  g_ibCastIflush = DefaultIbCastIflush;
  g_ibCastTest = DefaultIbCastTest;
}
