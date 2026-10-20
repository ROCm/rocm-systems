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

ASSERT_HOOK_MATCHES_PROD(g_ibCastMultiSend, IbCastMultiSend);
ASSERT_HOOK_MATCHES_PROD(g_ibCastPostFifo, IbCastPostFifo);
ASSERT_HOOK_MATCHES_PROD(g_ibCastAddEvent, IbCastAddEvent);

ncclResult_t IbCastMultiSend(struct ncclIbSendComm* comm, int slot, int nqps, int startQpIndex, bool wrrSched,
                             bool useWriteOp) {
  return g_ibCastMultiSend(comm, slot, nqps, startQpIndex, wrrSched, useWriteOp);
}
ncclResult_t IbCastPostFifo(struct ncclIbRecvComm* comm, struct ncclIbRequest* req, int slot, int n) {
  return g_ibCastPostFifo(comm, req, slot, n);
}
void IbCastAddEvent(struct ncclIbRequest* req, int devIndex) { g_ibCastAddEvent(req, devIndex); }

void ResetNetIbCastP2pFakes() {
  g_ibCastMultiSend = DefaultIbCastMultiSend;
  g_ibCastPostFifo = DefaultIbCastPostFifo;
  g_ibCastAddEvent = DefaultIbCastAddEvent;
}
