/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable host-only seams for src/misc/socket.cc. Defaults fail loudly;
// fixtures install the exact socket lifecycle behavior each test needs.

#ifndef RCCL_TEST_HOST_SOCKET_FAKES_H_
#define RCCL_TEST_HOST_SOCKET_FAKES_H_

#include <functional>

#include "socket.h"

extern std::function<ncclResult_t(struct ncclSocket*, const union ncclSocketAddress*, uint64_t,
                                  enum ncclSocketType, volatile uint32_t*, int, int)>
    g_ncclSocketInit;
extern std::function<ncclResult_t(struct ncclSocket*)> g_ncclSocketConnect;
extern std::function<ncclResult_t(struct ncclSocket*, struct ncclSocket*, bool)> g_ncclSocketAccept;
extern std::function<ncclResult_t(struct ncclSocket*, int*)> g_ncclSocketReady;
extern std::function<ncclResult_t(struct ncclSocket*, void*, int)> g_ncclSocketSend;
extern std::function<ncclResult_t(struct ncclSocket*, void*, int)> g_ncclSocketRecv;
extern std::function<ncclResult_t(int, struct ncclSocket*, void*, int, int*, bool*)> g_ncclSocketProgress;
extern std::function<void(struct ncclSocket*, struct ncclSocket*)> g_ncclSocketMove;

void ResetSocketFakes();

#endif  // RCCL_TEST_HOST_SOCKET_FAKES_H_
