/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for src/misc/socket.cc. Defaults model a socket that always succeeds; install per-test
// behaviour with ScopedHook and restore everything with ResetSocketFakes().

#ifndef RCCL_TEST_HOST_FAKES_SOCKET_FAKES_H_
#define RCCL_TEST_HOST_FAKES_SOCKET_FAKES_H_

#include <cstdint>
#include <functional>

#include "socket.h"

extern std::function<ncclResult_t(struct ncclSocket*, const union ncclSocketAddress*, uint64_t, enum ncclSocketType,
                                  volatile uint32_t*, int, int)>
    g_ncclSocketInit;
extern std::function<ncclResult_t(struct ncclSocket*)> g_ncclSocketListen;
extern std::function<ncclResult_t(struct ncclSocket*, union ncclSocketAddress*)> g_ncclSocketGetAddr;
extern std::function<ncclResult_t(struct ncclSocket*)> g_ncclSocketConnect;
extern std::function<ncclResult_t(struct ncclSocket*, struct ncclSocket*, bool)> g_ncclSocketAccept;
extern std::function<ncclResult_t(struct ncclSocket*, int*)> g_ncclSocketReady;
extern std::function<ncclResult_t(int, struct ncclSocket*, void*, int, int*, bool*)> g_ncclSocketProgress;
extern std::function<ncclResult_t(struct ncclSocket*, bool)> g_ncclSocketClose;
extern std::function<ncclResult_t(struct ncclSocket*, ncclSocketDescriptor*)> g_ncclSocketGetFd;
extern std::function<ncclResult_t(char*, union ncclSocketAddress*, int, int, int*)> g_ncclFindInterfaces;

// The defaults, for a test that wraps one rather than replacing it.
ncclResult_t DefaultNcclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic,
                                   enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag,
                                   int customRetry);
ncclResult_t DefaultNcclSocketReady(struct ncclSocket* sock, int* running);
ncclResult_t DefaultNcclSocketClose(struct ncclSocket* sock, bool wait);

void ResetSocketFakes();

#endif  // RCCL_TEST_HOST_FAKES_SOCKET_FAKES_H_
