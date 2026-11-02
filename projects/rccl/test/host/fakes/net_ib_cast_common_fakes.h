/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the symbols defined by src/transport/net_ib_cast/common.cc: the
// device tables and globals as plain definitions, its params through
// g_loadParam, and hooks for its functions. IbCastGetNetCommDevBase and
// IbCastStatsCheckFatalCount default to production's logic (both are pure);
// the comm initialisers and IbCastAsyncThreadMain fail loud until a test
// installs a hook.

#ifndef RCCL_TEST_HOST_NET_IB_CAST_COMMON_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_COMMON_FAKES_H_

#include <functional>

#include "common_cast.h"

extern std::function<ncclResult_t(struct ncclIbStats* /*stat*/, const char* /*funcName*/)>
    g_ibCastStatsCheckFatalCount;
extern std::function<struct ncclIbNetCommDevBase*(ncclIbNetCommBase* /*base*/, int /*devIndex*/)>
    g_ibCastGetNetCommDevBase;
extern std::function<ncclResult_t(struct ncclIbNetCommBase* /*baseComm*/, bool /*isSend*/)> g_ibCastBaseCommInit;
extern std::function<void(struct ncclIbNetCommBase* /*baseComm*/, bool /*useCtsOffload*/)>
    g_ibCastInitOptRecvCompletion;
extern std::function<ncclResult_t(struct ncclIbRecvComm* /*recvComm*/)> g_ibCastRecvCommInit;
extern std::function<ncclResult_t(struct ncclIbSendComm* /*sendComm*/)> g_ibCastSendCommInit;
extern std::function<void*(void* /*args*/)> g_ibCastAsyncThreadMain;

void ResetNetIbCastCommonFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_COMMON_FAKES_H_
