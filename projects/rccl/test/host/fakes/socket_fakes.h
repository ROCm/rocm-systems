/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for the socket layer defined by src/misc/socket.cc, for
// units that reach it but are not about it. Defaults: every acquisition,
// transfer and query fails with ncclSystemError; ncclSocketClose succeeds;
// ncclFindInterfaces finds no interface; ncclSocketToString writes an empty
// string; ncclSocketDefaultMagic is NCCL_SOCKET_MAGIC, its unset-env value.

#ifndef RCCL_TEST_HOST_SOCKET_FAKES_H_
#define RCCL_TEST_HOST_SOCKET_FAKES_H_

#include <cstddef>
#include <cstdint>
#include <functional>

#include "socket.h"

extern std::function<ncclResult_t(struct ncclSocket* /*sock*/, const union ncclSocketAddress* /*addr*/,
                                  uint64_t /*magic*/, enum ncclSocketType /*type*/,
                                  volatile uint32_t* /*abortFlag*/, int /*asyncFlag*/, int /*customRetry*/)>
    g_ncclSocketInit;
extern std::function<ncclResult_t(struct ncclSocket* /*sock*/)> g_ncclSocketListen;
extern std::function<ncclResult_t(struct ncclSocket* /*sock*/, union ncclSocketAddress* /*addr*/)>
    g_ncclSocketGetAddr;
extern std::function<ncclResult_t(struct ncclSocket* /*sock*/)> g_ncclSocketConnect;
extern std::function<ncclResult_t(struct ncclSocket* /*sock*/, int* /*running*/)> g_ncclSocketReady;
extern std::function<ncclResult_t(struct ncclSocket* /*sock*/, struct ncclSocket* /*ulistenSock*/, bool /*retry*/)>
    g_ncclSocketAccept;
extern std::function<ncclResult_t(int /*op*/, struct ncclSocket* /*sock*/, void* /*ptr*/, int /*size*/,
                                  int* /*offset*/, bool* /*closed*/)>
    g_ncclSocketProgress;
extern std::function<ncclResult_t(struct ncclSocket* /*sock*/, bool /*wait*/)> g_ncclSocketClose;
extern std::function<const char*(const union ncclSocketAddress* /*addr*/, char* /*buf*/, size_t /*bufLen*/,
                                 const int /*numericHostForm*/)>
    g_ncclSocketToString;
extern std::function<ncclResult_t(char* /*ifNames*/, union ncclSocketAddress* /*ifAddrs*/, int /*ifNameMaxSize*/,
                                  int /*maxIfs*/, int* /*nIfs*/)>
    g_ncclFindInterfaces;

void ResetSocketFakes();

#endif  // RCCL_TEST_HOST_SOCKET_FAKES_H_
