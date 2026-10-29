/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud hooks for the src/transport/net_ib_cast/p2p.cc entry points common.cc and gin.cc call. The binary also
// compiles the real p2p.cc, so net_ib_cast_p2p_seam.h renames those calls to the micro_* trampolines below.

#ifndef RCCL_TEST_HOST_NET_IB_CAST_P2P_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_P2P_FAKES_H_

#include <functional>

#include "p2p_cast.h"

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

void micro_IbCastAddEvent(struct ncclIbRequest* req, int devIndex);
ncclResult_t micro_IbCastGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req);
ncclResult_t micro_IbCastFreeRequest(struct ncclIbRequest* r);
ncclResult_t micro_IbCastIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle,
                               void** request);
ncclResult_t micro_IbCastIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles,
                               void** phandles, void** request);
ncclResult_t micro_IbCastIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
ncclResult_t micro_IbCastTest(void* request, int* done, int* sizes);

void ResetNetIbCastP2pFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_P2P_FAKES_H_
