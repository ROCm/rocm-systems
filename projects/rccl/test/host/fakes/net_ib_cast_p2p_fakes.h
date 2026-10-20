/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the symbols defined by src/transport/net_ib_cast/p2p.cc: its
// params through g_loadParam, IbCastReqTypeStr, and fail-loud hooks for the
// data-path entry points other net_ib_cast TUs call.

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

void ResetNetIbCastP2pFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_P2P_FAKES_H_
