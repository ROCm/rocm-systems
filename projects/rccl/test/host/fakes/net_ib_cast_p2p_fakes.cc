/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/transport/net_ib_cast/p2p.cc fakes. See
// net_ib_cast_p2p_fakes.h.

#include "net_ib_cast_p2p_fakes.h"

#include "fail_loud.h"
#include "nccl_fakes.h"
#include "signature-drift.h"

const char* IbCastReqTypeStr[] = {"Unused", "Send", "Recv", "Flush", "IPut"};
int64_t IbCastArThreshold = 8192;

int64_t ncclParamIbCastArThreshold() { return g_loadParam("IB_AR_THRESHOLD", -2); }
int64_t ncclParamIbCastReceiverSideMatchingScheme() { return g_loadParam("IB_RECEIVER_SIDE_MATCHING_SCHEME", -2); }
int64_t rcclParamIbCastGdrFlushGpuMemNoRelaxedOrdering() {
  return g_loadParam("RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING", 1);
}

static ncclResult_t DefaultIbCastMultiSend(struct ncclIbSendComm*, int, int, int, bool, bool) {
  FailLoudUnfaked("net_ib_cast_p2p_fakes", "IbCastMultiSend");
}
std::function<ncclResult_t(struct ncclIbSendComm*, int, int, int, bool, bool)> g_ibCastMultiSend =
    DefaultIbCastMultiSend;

static ncclResult_t DefaultIbCastPostFifo(struct ncclIbRecvComm*, struct ncclIbRequest*, int, int) {
  FailLoudUnfaked("net_ib_cast_p2p_fakes", "IbCastPostFifo");
}
std::function<ncclResult_t(struct ncclIbRecvComm*, struct ncclIbRequest*, int, int)> g_ibCastPostFifo =
    DefaultIbCastPostFifo;

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

ASSERT_HOOK_MATCHES_PROD(g_ibCastMultiSend, IbCastMultiSend);
ASSERT_HOOK_MATCHES_PROD(g_ibCastPostFifo, IbCastPostFifo);
ASSERT_HOOK_MATCHES_PROD(g_ibCastAddEvent, IbCastAddEvent);
ASSERT_HOOK_MATCHES_PROD(g_ibCastGetRequest, IbCastGetRequest);
ASSERT_HOOK_MATCHES_PROD(g_ibCastFreeRequest, IbCastFreeRequest);
ASSERT_HOOK_MATCHES_PROD(g_ibCastIsend, IbCastIsend);
ASSERT_HOOK_MATCHES_PROD(g_ibCastIrecv, IbCastIrecv);
ASSERT_HOOK_MATCHES_PROD(g_ibCastIflush, IbCastIflush);
ASSERT_HOOK_MATCHES_PROD(g_ibCastTest, IbCastTest);

ncclResult_t IbCastMultiSend(struct ncclIbSendComm* comm, int slot, int nqps, int startQpIndex, bool wrrSched,
                             bool useWriteOp) {
  return g_ibCastMultiSend(comm, slot, nqps, startQpIndex, wrrSched, useWriteOp);
}
ncclResult_t IbCastPostFifo(struct ncclIbRecvComm* comm, struct ncclIbRequest* req, int slot, int n) {
  return g_ibCastPostFifo(comm, req, slot, n);
}
void IbCastAddEvent(struct ncclIbRequest* req, int devIndex) { g_ibCastAddEvent(req, devIndex); }
ncclResult_t IbCastGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  return g_ibCastGetRequest(base, req);
}
ncclResult_t IbCastFreeRequest(struct ncclIbRequest* r) { return g_ibCastFreeRequest(r); }
ncclResult_t IbCastIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle,
                         void** request) {
  return g_ibCastIsend(sendComm, data, size, tag, mhandle, phandle, request);
}
ncclResult_t IbCastIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles,
                         void** request) {
  return g_ibCastIrecv(recvComm, n, data, sizes, tags, mhandles, phandles, request);
}
ncclResult_t IbCastIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  return g_ibCastIflush(recvComm, n, data, sizes, mhandles, request);
}
ncclResult_t IbCastTest(void* request, int* done, int* sizes) { return g_ibCastTest(request, done, sizes); }

void ResetNetIbCastP2pFakes() {
  g_ibCastMultiSend = DefaultIbCastMultiSend;
  g_ibCastPostFifo = DefaultIbCastPostFifo;
  g_ibCastAddEvent = DefaultIbCastAddEvent;
  g_ibCastGetRequest = DefaultIbCastGetRequest;
  g_ibCastFreeRequest = DefaultIbCastFreeRequest;
  g_ibCastIsend = DefaultIbCastIsend;
  g_ibCastIrecv = DefaultIbCastIrecv;
  g_ibCastIflush = DefaultIbCastIflush;
  g_ibCastTest = DefaultIbCastTest;
}
