/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the symbols defined by src/transport/net_ib_cast/p2p.cc: its
// params through g_loadParam, IbCastReqTypeStr, IbCastArThreshold, and
// fail-loud hooks for its functions.

#ifndef RCCL_TEST_HOST_NET_IB_CAST_P2P_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_P2P_FAKES_H_

#include <functional>

#include "p2p_cast.h"

extern std::function<ncclResult_t(struct ncclIbSendComm* /*comm*/, int /*slot*/, int /*nqps*/, int /*startQpIndex*/,
                                  bool /*wrrSched*/, bool /*useWriteOp*/)>
    g_ibCastMultiSend;
extern std::function<ncclResult_t(struct ncclIbRecvComm* /*comm*/, struct ncclIbRequest* /*req*/, int /*slot*/,
                                  int /*n*/)>
    g_ibCastPostFifo;
extern std::function<void(struct ncclIbRequest* /*req*/, int /*devIndex*/)> g_ibCastAddEvent;

extern std::function<ncclResult_t(struct ncclIbNetCommBase* /*base*/, struct ncclIbRequest** /*req*/)>
    g_ibCastGetRequest;
extern std::function<ncclResult_t(struct ncclIbRequest* /*r*/)> g_ibCastFreeRequest;
extern std::function<ncclResult_t(void* /*sendComm*/, void* /*data*/, size_t /*size*/, int /*tag*/, void* /*mhandle*/,
                                  void* /*phandle*/, void** /*request*/)>
    g_ibCastIsend;
extern std::function<ncclResult_t(void* /*recvComm*/, int /*n*/, void** /*data*/, size_t* /*sizes*/, int* /*tags*/,
                                  void** /*mhandles*/, void** /*phandles*/, void** /*request*/)>
    g_ibCastIrecv;
extern std::function<ncclResult_t(void* /*recvComm*/, int /*n*/, void** /*data*/, int* /*sizes*/, void** /*mhandles*/,
                                  void** /*request*/)>
    g_ibCastIflush;
extern std::function<ncclResult_t(void* /*request*/, int* /*done*/, int* /*sizes*/)> g_ibCastTest;

void ResetNetIbCastP2pFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_P2P_FAKES_H_
