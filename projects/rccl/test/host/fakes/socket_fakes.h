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

// More src/misc/socket.cc entry points, plus ncclFindInterfaceMatchSubnet, which src/os/linux.cc
// defines next to ncclSocketClose. Same fail-loud defaults. ResetSocketFakes() does not touch these;
// ResetExtraSocketFakes() does.
extern std::function<ncclResult_t(ncclSocketDescriptor, struct ncclSocket*)> g_ncclSocketSetFd;
extern std::function<ncclResult_t(struct ncclSocket*, void*, int, bool*, bool)> g_ncclSocketTryRecv;
extern std::function<ncclResult_t(struct ncclSocket*, void*, int, struct ncclSocket*, void*, int)>
    g_ncclSocketSendRecv;
extern std::function<ncclResult_t(struct ncclSocketOp*, int)> g_ncclSocketMultiOp;
extern std::function<ncclResult_t(union ncclSocketAddress*, const char*)> g_ncclSocketGetAddrFromString;
extern std::function<ncclResult_t(char*, union ncclSocketAddress*, union ncclSocketAddress*, int, int*)>
    g_ncclFindInterfaceMatchSubnet;

void ResetExtraSocketFakes();

// An in-memory socket layer for units that run several ranks as threads. A listener is a port with a
// queue of pending connections; a connection is two endpoints whose inbound byte queues the other side
// appends to. Blocking calls honour the socket's abort flag and fail the test instead of hanging.
// InstallInMemorySockets() resets every hook above, empties the network and points the hooks at it.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <set>

namespace in_memory_sockets {
using Clock = std::chrono::steady_clock;
constexpr int kEndpointFd = 10000;  // endpoint i's socketDescriptor is kEndpointFd + i
constexpr int kListenFd = 20000;    // listener i's is kListenFd + i
struct FakeListener {
  uint16_t port = 0;
  uint64_t magic = 0;
  bool open = true;
  std::deque<int> pending;  // connected endpoints waiting for accept
};
extern std::mutex g_sockMu;  // guards the network; g_sockCv is notified on every change
extern std::condition_variable g_sockCv;
extern std::deque<FakeListener> g_listeners;
extern std::set<int> g_openFds;
extern bool g_sockShutdown;  // fails every blocked and later wait with ncclSystemError
extern std::chrono::milliseconds g_waitLimit;
// Mirrors socket.cc: ncclSocketAccept(retry=false) hands back a rejected peer as Accepting, not BadHandshake.
extern ncclSocketState g_rejectedState;
extern int g_readyPolls;  // ncclSocketReady says "not yet" this many times first
extern std::atomic<int> g_multiOps;
extern int g_ifCount;  // interfaces ncclFindInterfaces reports, all "fake0"
extern union ncclSocketAddress g_subnetRemote;  // what ncclFindInterfaceMatchSubnet was last asked to match
using SockFaultFn = std::function<ncclResult_t(const char* /*op*/, struct ncclSocket*)>;
void SetSockFault(SockFaultFn fn);  // runs first in each call; a non-success result is returned as is
uint16_t PortOf(const union ncclSocketAddress& addr);
ncclResult_t SocketClose(struct ncclSocket* sock, bool wait);  // what g_ncclSocketClose is set to
}  // namespace in_memory_sockets

void InstallInMemorySockets();

#endif  // RCCL_TEST_HOST_SOCKET_FAKES_H_
