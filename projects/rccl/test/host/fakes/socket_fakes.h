/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable host-only seams for src/misc/socket.cc, plus ncclSocketClose
// from src/os/{linux,windows}.cc. Defaults fail loudly; fixtures install the
// exact socket lifecycle behavior each test needs. ncclSocketClose stays out of
// os_fakes because binaries that define their own close also link os_fakes.

#ifndef RCCL_TEST_HOST_SOCKET_FAKES_H_
#define RCCL_TEST_HOST_SOCKET_FAKES_H_

#include <functional>

#include "socket.h"

extern std::function<ncclResult_t(struct ncclSocket*, const union ncclSocketAddress*, uint64_t,
                                  enum ncclSocketType, volatile uint32_t*, int, int)>
    g_ncclSocketInit;
extern std::function<ncclResult_t(struct ncclSocket*)> g_ncclSocketListen;
extern std::function<ncclResult_t(struct ncclSocket*, union ncclSocketAddress*)> g_ncclSocketGetAddr;
extern std::function<ncclResult_t(struct ncclSocket*)> g_ncclSocketConnect;
extern std::function<ncclResult_t(struct ncclSocket*, struct ncclSocket*, bool)> g_ncclSocketAccept;
extern std::function<ncclResult_t(struct ncclSocket*, int*)> g_ncclSocketReady;
extern std::function<ncclResult_t(struct ncclSocket*, void*, int)> g_ncclSocketSend;
extern std::function<ncclResult_t(struct ncclSocket*, void*, int)> g_ncclSocketRecv;
extern std::function<ncclResult_t(int, struct ncclSocket*, void*, int, int*, bool*)> g_ncclSocketProgress;
extern std::function<ncclResult_t(struct ncclSocket*, bool)> g_ncclSocketClose;
extern std::function<void(struct ncclSocket*, struct ncclSocket*)> g_ncclSocketMove;
extern std::function<ncclResult_t(struct ncclSocket*, ncclSocketDescriptor*)> g_ncclSocketGetFd;
extern std::function<ncclResult_t(char*, union ncclSocketAddress*, int, int, int*)> g_ncclFindInterfaces;
extern std::function<const char*(const union ncclSocketAddress*, char*, size_t, int)> g_ncclSocketToString;

// Production-faithful helpers for tests that explicitly need a working
// baseline. ResetSocketFakes still restores fail-loud hooks.
ncclResult_t DefaultNcclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic,
                                   enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag,
                                   int customRetry);
ncclResult_t DefaultNcclSocketReady(struct ncclSocket* sock, int* running);

void ResetSocketFakes();

#endif  // RCCL_TEST_HOST_SOCKET_FAKES_H_
