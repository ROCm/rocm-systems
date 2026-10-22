/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "socket_fakes.h"

#include <cstring>

#include "fail_loud.h"
#include "signature-drift.h"

static ncclResult_t DefaultSocketInit(struct ncclSocket*, const union ncclSocketAddress*, uint64_t,
                                      enum ncclSocketType, volatile uint32_t*, int, int) {
  FailLoudUnfaked("socket_fakes", "ncclSocketInit");
}

static ncclResult_t DefaultSocketListen(struct ncclSocket*) {
  FailLoudUnfaked("socket_fakes", "ncclSocketListen");
}

static ncclResult_t DefaultSocketGetAddr(struct ncclSocket*, union ncclSocketAddress*) {
  FailLoudUnfaked("socket_fakes", "ncclSocketGetAddr");
}

static ncclResult_t DefaultSocketConnect(struct ncclSocket*) {
  FailLoudUnfaked("socket_fakes", "ncclSocketConnect");
}

static ncclResult_t DefaultSocketAccept(struct ncclSocket*, struct ncclSocket*, bool) {
  FailLoudUnfaked("socket_fakes", "ncclSocketAccept");
}

static ncclResult_t DefaultSocketReady(struct ncclSocket*, int*) {
  FailLoudUnfaked("socket_fakes", "ncclSocketReady");
}

static ncclResult_t DefaultSocketSend(struct ncclSocket*, void*, int) {
  FailLoudUnfaked("socket_fakes", "ncclSocketSend");
}

static ncclResult_t DefaultSocketRecv(struct ncclSocket*, void*, int) {
  FailLoudUnfaked("socket_fakes", "ncclSocketRecv");
}

static ncclResult_t DefaultSocketProgress(int, struct ncclSocket*, void*, int, int*, bool*) {
  FailLoudUnfaked("socket_fakes", "ncclSocketProgress");
}

static ncclResult_t DefaultSocketClose(struct ncclSocket*, bool) {
  FailLoudUnfaked("socket_fakes", "ncclSocketClose");
}

static void DefaultSocketMove(struct ncclSocket*, struct ncclSocket*) {
  FailLoudUnfaked("socket_fakes", "ncclSocketMove");
}

static ncclResult_t DefaultSocketGetFd(struct ncclSocket*, ncclSocketDescriptor*) {
  FailLoudUnfaked("socket_fakes", "ncclSocketGetFd");
}

static ncclResult_t DefaultFindInterfaces(char*, union ncclSocketAddress*, int, int, int*) {
  FailLoudUnfaked("socket_fakes", "ncclFindInterfaces");
}

static const char* DefaultSocketToString(const union ncclSocketAddress*, char*, size_t, int) {
  FailLoudUnfaked("socket_fakes", "ncclSocketToString");
}

ncclResult_t DefaultNcclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic,
                                   enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag,
                                   int customRetry) {
  std::memset(sock, 0, sizeof(*sock));
  if (addr) sock->addr = *addr;
  sock->magic = magic;
  sock->type = type;
  sock->abortFlag = abortFlag;
  sock->asyncFlag = asyncFlag;
  sock->customRetry = customRetry;
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  sock->state = ncclSocketStateInitialized;
  return ncclSuccess;
}

ncclResult_t DefaultNcclSocketReady(struct ncclSocket* sock, int* running) {
  if (sock->state == ncclSocketStateError || sock->state == ncclSocketStateClosed) return ncclRemoteError;
  if (sock->state == ncclSocketStateConnecting || sock->state == ncclSocketStateAccepting) {
    sock->state = ncclSocketStateReady;
  }
  *running = sock->state == ncclSocketStateReady;
  return ncclSuccess;
}

std::function<ncclResult_t(struct ncclSocket*, const union ncclSocketAddress*, uint64_t,
                           enum ncclSocketType, volatile uint32_t*, int, int)>
    g_ncclSocketInit = DefaultSocketInit;
std::function<ncclResult_t(struct ncclSocket*)> g_ncclSocketListen = DefaultSocketListen;
std::function<ncclResult_t(struct ncclSocket*, union ncclSocketAddress*)> g_ncclSocketGetAddr =
    DefaultSocketGetAddr;
std::function<ncclResult_t(struct ncclSocket*)> g_ncclSocketConnect = DefaultSocketConnect;
std::function<ncclResult_t(struct ncclSocket*, struct ncclSocket*, bool)> g_ncclSocketAccept =
    DefaultSocketAccept;
std::function<ncclResult_t(struct ncclSocket*, int*)> g_ncclSocketReady = DefaultSocketReady;
std::function<ncclResult_t(struct ncclSocket*, void*, int)> g_ncclSocketSend = DefaultSocketSend;
std::function<ncclResult_t(struct ncclSocket*, void*, int)> g_ncclSocketRecv = DefaultSocketRecv;
std::function<ncclResult_t(int, struct ncclSocket*, void*, int, int*, bool*)> g_ncclSocketProgress =
    DefaultSocketProgress;
std::function<ncclResult_t(struct ncclSocket*, bool)> g_ncclSocketClose = DefaultSocketClose;
std::function<void(struct ncclSocket*, struct ncclSocket*)> g_ncclSocketMove = DefaultSocketMove;
std::function<ncclResult_t(struct ncclSocket*, ncclSocketDescriptor*)> g_ncclSocketGetFd =
    DefaultSocketGetFd;
std::function<ncclResult_t(char*, union ncclSocketAddress*, int, int, int*)> g_ncclFindInterfaces =
    DefaultFindInterfaces;
std::function<const char*(const union ncclSocketAddress*, char*, size_t, int)> g_ncclSocketToString =
    DefaultSocketToString;

ASSERT_HOOK_MATCHES_PROD(g_ncclSocketInit, ncclSocketInit);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketListen, ncclSocketListen);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketGetAddr, ncclSocketGetAddr);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketConnect, ncclSocketConnect);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketAccept, ncclSocketAccept);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketReady, ncclSocketReady);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketSend, ncclSocketSend);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketRecv, ncclSocketRecv);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketProgress, ncclSocketProgress);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketClose, ncclSocketClose);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketMove, ncclSocketMove);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketGetFd, ncclSocketGetFd);
ASSERT_HOOK_MATCHES_PROD(g_ncclFindInterfaces, ncclFindInterfaces);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketToString, ncclSocketToString);
#undef ASSERT_HOOK_MATCHES_PROD

uint64_t ncclSocketDefaultMagic() { return NCCL_SOCKET_MAGIC; }

ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic,
                            enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag, int customRetry) {
  return g_ncclSocketInit(sock, addr, magic, type, abortFlag, asyncFlag, customRetry);
}

ncclResult_t ncclSocketListen(struct ncclSocket* sock) { return g_ncclSocketListen(sock); }

ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr) {
  return g_ncclSocketGetAddr(sock, addr);
}

ncclResult_t ncclSocketConnect(struct ncclSocket* sock) { return g_ncclSocketConnect(sock); }

ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* listenSock, bool retry) {
  return g_ncclSocketAccept(sock, listenSock, retry);
}

ncclResult_t ncclSocketReady(struct ncclSocket* sock, int* running) {
  return g_ncclSocketReady(sock, running);
}

ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size) {
  return g_ncclSocketSend(sock, ptr, size);
}

ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size) {
  return g_ncclSocketRecv(sock, ptr, size);
}

ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, bool* closed) {
  return g_ncclSocketProgress(op, sock, ptr, size, offset, closed);
}

ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool wait) { return g_ncclSocketClose(sock, wait); }

void ncclSocketMove(struct ncclSocket* dst, struct ncclSocket* src) { g_ncclSocketMove(dst, src); }

ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, ncclSocketDescriptor* fd) {
  return g_ncclSocketGetFd(sock, fd);
}

ncclResult_t ncclFindInterfaces(char* names, union ncclSocketAddress* addresses, int nameSize, int maxIfs,
                                int* count) {
  return g_ncclFindInterfaces(names, addresses, nameSize, maxIfs, count);
}

const char* ncclSocketToString(const union ncclSocketAddress* address, char* buffer, size_t size, int numeric) {
  return g_ncclSocketToString(address, buffer, size, numeric);
}

void ResetSocketFakes() {
  g_ncclSocketInit = DefaultSocketInit;
  g_ncclSocketListen = DefaultSocketListen;
  g_ncclSocketGetAddr = DefaultSocketGetAddr;
  g_ncclSocketConnect = DefaultSocketConnect;
  g_ncclSocketAccept = DefaultSocketAccept;
  g_ncclSocketReady = DefaultSocketReady;
  g_ncclSocketSend = DefaultSocketSend;
  g_ncclSocketRecv = DefaultSocketRecv;
  g_ncclSocketProgress = DefaultSocketProgress;
  g_ncclSocketClose = DefaultSocketClose;
  g_ncclSocketMove = DefaultSocketMove;
  g_ncclSocketGetFd = DefaultSocketGetFd;
  g_ncclFindInterfaces = DefaultFindInterfaces;
  g_ncclSocketToString = DefaultSocketToString;
}

static ncclResult_t DefaultSocketSetFd(ncclSocketDescriptor, struct ncclSocket*) {
  FailLoudUnfaked("socket_fakes", "ncclSocketSetFd");
}

static ncclResult_t DefaultSocketTryRecv(struct ncclSocket*, void*, int, bool*, bool) {
  FailLoudUnfaked("socket_fakes", "ncclSocketTryRecv");
}

static ncclResult_t DefaultSocketSendRecv(struct ncclSocket*, void*, int, struct ncclSocket*, void*, int) {
  FailLoudUnfaked("socket_fakes", "ncclSocketSendRecv");
}

static ncclResult_t DefaultSocketMultiOp(struct ncclSocketOp*, int) {
  FailLoudUnfaked("socket_fakes", "ncclSocketMultiOp");
}

static ncclResult_t DefaultSocketGetAddrFromString(union ncclSocketAddress*, const char*) {
  FailLoudUnfaked("socket_fakes", "ncclSocketGetAddrFromString");
}

static ncclResult_t DefaultFindInterfaceMatchSubnet(char*, union ncclSocketAddress*, union ncclSocketAddress*, int,
                                                    int*) {
  FailLoudUnfaked("socket_fakes", "ncclFindInterfaceMatchSubnet");
}

std::function<ncclResult_t(ncclSocketDescriptor, struct ncclSocket*)> g_ncclSocketSetFd = DefaultSocketSetFd;
std::function<ncclResult_t(struct ncclSocket*, void*, int, bool*, bool)> g_ncclSocketTryRecv = DefaultSocketTryRecv;
std::function<ncclResult_t(struct ncclSocket*, void*, int, struct ncclSocket*, void*, int)> g_ncclSocketSendRecv =
    DefaultSocketSendRecv;
std::function<ncclResult_t(struct ncclSocketOp*, int)> g_ncclSocketMultiOp = DefaultSocketMultiOp;
std::function<ncclResult_t(union ncclSocketAddress*, const char*)> g_ncclSocketGetAddrFromString =
    DefaultSocketGetAddrFromString;
std::function<ncclResult_t(char*, union ncclSocketAddress*, union ncclSocketAddress*, int, int*)>
    g_ncclFindInterfaceMatchSubnet = DefaultFindInterfaceMatchSubnet;

// ASSERT_HOOK_MATCHES_PROD is #undef'd above; this is the same check.
#define EXTRA_HOOK_MATCHES_PROD(hook, prod)                                                       \
  static_assert(::rccl_test_host::HookMatchesProd<decltype(hook), decltype(&prod)>(),             \
                "signature drift: " #hook " no longer matches " #prod)
EXTRA_HOOK_MATCHES_PROD(g_ncclSocketSetFd, ncclSocketSetFd);
EXTRA_HOOK_MATCHES_PROD(g_ncclSocketTryRecv, ncclSocketTryRecv);
EXTRA_HOOK_MATCHES_PROD(g_ncclSocketSendRecv, ncclSocketSendRecv);
EXTRA_HOOK_MATCHES_PROD(g_ncclSocketMultiOp, ncclSocketMultiOp);
EXTRA_HOOK_MATCHES_PROD(g_ncclSocketGetAddrFromString, ncclSocketGetAddrFromString);
EXTRA_HOOK_MATCHES_PROD(g_ncclFindInterfaceMatchSubnet, ncclFindInterfaceMatchSubnet);
#undef EXTRA_HOOK_MATCHES_PROD

ncclResult_t ncclSocketSetFd(ncclSocketDescriptor fd, struct ncclSocket* sock) { return g_ncclSocketSetFd(fd, sock); }

ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, bool* closed, bool blocking) {
  return g_ncclSocketTryRecv(sock, ptr, size, closed, blocking);
}

ncclResult_t ncclSocketSendRecv(struct ncclSocket* sendSock, void* sendPtr, int sendSize, struct ncclSocket* recvSock,
                                void* recvPtr, int recvSize) {
  return g_ncclSocketSendRecv(sendSock, sendPtr, sendSize, recvSock, recvPtr, recvSize);
}

ncclResult_t ncclSocketMultiOp(struct ncclSocketOp* ops, int numOps) { return g_ncclSocketMultiOp(ops, numOps); }

ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress* ua, const char* ipPortPair) {
  return g_ncclSocketGetAddrFromString(ua, ipPortPair);
}

ncclResult_t ncclFindInterfaceMatchSubnet(char* ifName, union ncclSocketAddress* localAddr,
                                          union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int* found) {
  return g_ncclFindInterfaceMatchSubnet(ifName, localAddr, remoteAddr, ifNameMaxSize, found);
}

void ResetExtraSocketFakes() {
  g_ncclSocketSetFd = DefaultSocketSetFd;
  g_ncclSocketTryRecv = DefaultSocketTryRecv;
  g_ncclSocketSendRecv = DefaultSocketSendRecv;
  g_ncclSocketMultiOp = DefaultSocketMultiOp;
  g_ncclSocketGetAddrFromString = DefaultSocketGetAddrFromString;
  g_ncclFindInterfaceMatchSubnet = DefaultFindInterfaceMatchSubnet;
}

// The in-memory socket layer; see socket_fakes.h.
#include <arpa/inet.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>

namespace in_memory_sockets {
std::mutex g_sockMu;
std::condition_variable g_sockCv;
std::deque<FakeListener> g_listeners;
std::set<int> g_openFds;
bool g_sockShutdown = false;
std::chrono::milliseconds g_waitLimit(5000);
ncclSocketState g_rejectedState = ncclSocketStateAccepting;
int g_readyPolls = 0;
std::atomic<int> g_multiOps{0};
int g_ifCount = 1;
union ncclSocketAddress g_subnetRemote;

namespace {
struct FakeEndpoint {
  std::deque<char> in;
  int peer = -1;
  bool open = true;
  uint64_t magic = 0;  // the connector's, checked by accept
};
std::deque<FakeEndpoint> g_endpoints;
std::map<uint16_t, int> g_ports;
uint16_t g_nextPort = 1000;
std::shared_ptr<SockFaultFn> g_sockFault;  // swapped atomically: a rank thread may be calling it

ncclResult_t SockFault(const char* op, struct ncclSocket* sock) {
  auto fault = std::atomic_load(&g_sockFault);
  return fault ? (*fault)(op, sock) : ncclSuccess;
}
FakeEndpoint& Endpoint(const struct ncclSocket* sock) { return g_endpoints[sock->socketDescriptor - kEndpointFd]; }

ncclResult_t WaitFor(std::unique_lock<std::mutex>& lock, volatile uint32_t* abortFlag,
                     const std::function<bool()>& ready) {
  auto deadline = Clock::now() + g_waitLimit;
  while (!ready()) {
    if (abortFlag && __atomic_load_n(abortFlag, __ATOMIC_ACQUIRE)) return ncclInternalError;
    if (g_sockShutdown) return ncclSystemError;
    if (Clock::now() > deadline) {
      ADD_FAILURE() << "fake socket wait timed out";
      return ncclSystemError;
    }
    g_sockCv.wait_for(lock, std::chrono::milliseconds(1));
  }
  return ncclSuccess;
}

// Takes the next pending connection, rejecting a wrong-magic peer the way socketFinalizeAccept does.
bool PopPending(struct ncclSocket* sock, FakeListener& l, bool retry) {
  while (!l.pending.empty()) {
    int e = l.pending.front();
    l.pending.pop_front();
    if (g_endpoints[e].magic != l.magic) {
      g_endpoints[e].open = false;
      g_sockCv.notify_all();
      if (retry) continue;
      sock->state = g_rejectedState;
      return true;
    }
    sock->socketDescriptor = kEndpointFd + e;
    sock->state = ncclSocketStateReady;
    g_openFds.insert(sock->socketDescriptor);
    return true;
  }
  return false;
}

ncclResult_t SocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic,
                        enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag, int customRetry) {
  if (sock == nullptr) return ncclSuccess;
  std::memset(sock, 0, sizeof(*sock));
  if (addr) sock->addr = *addr;
  sock->magic = magic;
  sock->type = type;
  sock->abortFlag = abortFlag;
  sock->asyncFlag = asyncFlag;
  sock->customRetry = customRetry;
  sock->state = ncclSocketStateInitialized;
  sock->socketDescriptor = sock->acceptSocketDescriptor = NCCL_INVALID_SOCKET;
  return SockFault("init", sock);
}

void SocketMove(struct ncclSocket* dst, struct ncclSocket* src) {
  *dst = *src;
  src->socketDescriptor = src->acceptSocketDescriptor = NCCL_INVALID_SOCKET;
  src->state = ncclSocketStateNone;
}

ncclResult_t SocketListen(struct ncclSocket* sock) {
  if (ncclResult_t res = SockFault("listen", sock)) return res;
  std::lock_guard<std::mutex> lock(g_sockMu);
  uint16_t port = PortOf(sock->addr);
  if (port == 0) port = g_nextPort++;
  if (g_ports.count(port)) return ncclSystemError;
  sock->addr.sin.sin_family = AF_INET;
  sock->addr.sin.sin_port = htons(port);
  g_listeners.emplace_back();
  FakeListener& l = g_listeners.back();
  l.port = port;
  l.magic = sock->magic;
  g_ports[port] = g_listeners.size() - 1;
  sock->socketDescriptor = kListenFd + g_listeners.size() - 1;
  sock->state = ncclSocketStateReady;
  g_openFds.insert(sock->socketDescriptor);
  return ncclSuccess;
}

ncclResult_t SocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr) {
  if (ncclResult_t res = SockFault("getaddr", sock)) return res;
  *addr = sock->addr;
  return ncclSuccess;
}

ncclResult_t SocketConnect(struct ncclSocket* sock) {
  if (ncclResult_t res = SockFault("connect", sock)) return res;
  std::lock_guard<std::mutex> lock(g_sockMu);
  auto it = g_ports.find(PortOf(sock->addr));
  if (it == g_ports.end() || !g_listeners[it->second].open) return ncclRemoteError;
  int a = g_endpoints.size();
  g_endpoints.emplace_back();
  g_endpoints.emplace_back();
  g_endpoints[a].peer = a + 1;
  g_endpoints[a + 1].peer = a;
  g_endpoints[a + 1].magic = sock->magic;
  g_listeners[it->second].pending.push_back(a + 1);
  sock->socketDescriptor = kEndpointFd + a;
  sock->state = sock->asyncFlag ? ncclSocketStateConnecting : ncclSocketStateReady;
  g_openFds.insert(sock->socketDescriptor);
  g_sockCv.notify_all();
  return ncclSuccess;
}

ncclResult_t SocketAccept(struct ncclSocket* sock, struct ncclSocket* listenSock, bool retry) {
  if (ncclResult_t res = SockFault("accept", sock)) return res;
  if (listenSock->state != ncclSocketStateReady) return ncclInternalError;
  if (sock->acceptSocketDescriptor == NCCL_INVALID_SOCKET) {  // clones the listener, as socket.cc does
    *sock = *listenSock;
    sock->acceptSocketDescriptor = listenSock->socketDescriptor;
    sock->socketDescriptor = NCCL_INVALID_SOCKET;
    sock->state = ncclSocketStateAccepting;
  }
  std::unique_lock<std::mutex> lock(g_sockMu);
  FakeListener& l = g_listeners[listenSock->socketDescriptor - kListenFd];
  if (sock->asyncFlag) {
    PopPending(sock, l, retry);
    return ncclSuccess;
  }
  ncclResult_t res = WaitFor(lock, sock->abortFlag, [&] { return !l.pending.empty() || !l.open; });
  if (res != ncclSuccess) return res;
  if (!PopPending(sock, l, retry)) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t SocketReady(struct ncclSocket* sock, int* running) {
  if (ncclResult_t res = SockFault("ready", sock)) return res;
  std::lock_guard<std::mutex> lock(g_sockMu);
  *running = 0;
  if (g_readyPolls > 0) {
    g_readyPolls--;
    return ncclSuccess;
  }
  if (sock->state == ncclSocketStateConnecting) sock->state = ncclSocketStateReady;
  if (sock->state == ncclSocketStateAccepting)
    PopPending(sock, g_listeners[sock->acceptSocketDescriptor - kListenFd], /*retry=*/true);
  *running = sock->state == ncclSocketStateReady;
  return ncclSuccess;
}

ncclResult_t SocketSend(struct ncclSocket* sock, void* ptr, int size) {
  if (ncclResult_t res = SockFault("send", sock)) return res;
  std::lock_guard<std::mutex> lock(g_sockMu);
  if (sock->state != ncclSocketStateReady) return ncclInternalError;
  FakeEndpoint& peer = g_endpoints[Endpoint(sock).peer];
  if (!peer.open) return ncclRemoteError;
  peer.in.insert(peer.in.end(), (char*)ptr, (char*)ptr + size);
  g_sockCv.notify_all();
  return ncclSuccess;
}

ncclResult_t SocketRecv(struct ncclSocket* sock, void* ptr, int size) {
  if (ncclResult_t res = SockFault("recv", sock)) return res;
  std::unique_lock<std::mutex> lock(g_sockMu);
  if (sock->state != ncclSocketStateReady) return ncclInternalError;
  FakeEndpoint& me = Endpoint(sock);
  ncclResult_t res = WaitFor(lock, sock->abortFlag,
                             [&] { return (int)me.in.size() >= size || !g_endpoints[me.peer].open; });
  if (res != ncclSuccess) return res;
  if ((int)me.in.size() < size) return ncclRemoteError;  // peer closed short
  std::copy(me.in.begin(), me.in.begin() + size, (char*)ptr);
  me.in.erase(me.in.begin(), me.in.begin() + size);
  return ncclSuccess;
}

ncclResult_t SocketSendRecv(struct ncclSocket* sendSock, void* sendPtr, int sendSize, struct ncclSocket* recvSock,
                            void* recvPtr, int recvSize) {
  if (ncclResult_t res = SocketSend(sendSock, sendPtr, sendSize)) return res;
  return SocketRecv(recvSock, recvPtr, recvSize);
}

// Sends never block here, so issuing them all before any receive cannot deadlock.
ncclResult_t SocketMultiOp(struct ncclSocketOp* ops, int numOps) {
  g_multiOps++;
  for (int pass = NCCL_SOCKET_SEND; pass <= NCCL_SOCKET_RECV; pass++) {
    for (int i = 0; i < numOps; i++) {
      if (ops[i].op != pass || ops[i].size == 0) continue;
      ncclResult_t res = pass == NCCL_SOCKET_SEND ? SocketSend(ops[i].sock, ops[i].ptr, ops[i].size)
                                                  : SocketRecv(ops[i].sock, ops[i].ptr, ops[i].size);
      if (res != ncclSuccess) return res;
    }
  }
  return ncclSuccess;
}

const char* SocketToString(const union ncclSocketAddress* addr, char* buf, size_t bufLen, const int) {
  std::snprintf(buf, bufLen, "fake:%u", PortOf(*addr));
  return buf;
}

ncclResult_t SocketGetAddrFromString(union ncclSocketAddress* ua, const char* ipPortPair) {
  const char* colon = std::strrchr(ipPortPair, ':');
  if (colon == nullptr) return ncclInvalidArgument;
  std::memset(ua, 0, sizeof(*ua));
  ua->sin.sin_family = AF_INET;
  ua->sin.sin_port = htons(std::atoi(colon + 1));
  return ncclSuccess;
}

ncclResult_t FindInterfaces(char* ifNames, union ncclSocketAddress* ifAddrs, int ifNameMaxSize, int, int* nIfs) {
  *nIfs = g_ifCount;
  if (g_ifCount > 0) {
    std::snprintf(ifNames, ifNameMaxSize, "fake0");
    std::memset(ifAddrs, 0, sizeof(*ifAddrs));
    ifAddrs->sin.sin_family = AF_INET;
  }
  return ncclSuccess;
}

ncclResult_t FindInterfaceMatchSubnet(char* ifName, union ncclSocketAddress* localAddr,
                                      union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int* found) {
  g_subnetRemote = *remoteAddr;
  return FindInterfaces(ifName, localAddr, ifNameMaxSize, 1, found);
}
}  // namespace

void SetSockFault(SockFaultFn fn) {
  std::atomic_store(&g_sockFault, fn ? std::make_shared<SockFaultFn>(std::move(fn)) : nullptr);
}

uint16_t PortOf(const union ncclSocketAddress& addr) { return ntohs(addr.sin.sin_port); }

ncclResult_t SocketClose(struct ncclSocket* sock, bool wait) {
  if (sock == nullptr) return ncclSuccess;
  if (ncclResult_t res = SockFault("close", sock)) return res;
  std::unique_lock<std::mutex> lock(g_sockMu);
  int fd = sock->socketDescriptor;
  if (sock->state > ncclSocketStateNone && fd >= kListenFd) {
    FakeListener& l = g_listeners[fd - kListenFd];
    l.open = false;
    g_ports.erase(l.port);
    for (int e : l.pending) g_endpoints[e].open = false;
  } else if (sock->state > ncclSocketStateNone && fd >= kEndpointFd) {
    FakeEndpoint& me = Endpoint(sock);
    if (wait) (void)WaitFor(lock, sock->abortFlag, [&] { return !g_endpoints[me.peer].open; });
    me.open = false;
  }
  g_openFds.erase(fd);
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  sock->state = ncclSocketStateClosed;
  g_sockCv.notify_all();
  return ncclSuccess;
}
}  // namespace in_memory_sockets

void InstallInMemorySockets() {
  using namespace in_memory_sockets;
  ResetSocketFakes();
  ResetExtraSocketFakes();
  {
    std::lock_guard<std::mutex> lock(g_sockMu);
    g_endpoints.clear();
    g_listeners.clear();
    g_ports.clear();
    g_openFds.clear();
    g_sockShutdown = false;
  }
  g_waitLimit = std::chrono::milliseconds(5000);
  g_rejectedState = ncclSocketStateAccepting;
  g_readyPolls = 0;
  g_multiOps = 0;
  g_ifCount = 1;
  g_subnetRemote = {};
  SetSockFault(nullptr);
  g_ncclSocketInit = SocketInit;
  g_ncclSocketListen = SocketListen;
  g_ncclSocketGetAddr = SocketGetAddr;
  g_ncclSocketConnect = SocketConnect;
  g_ncclSocketAccept = SocketAccept;
  g_ncclSocketReady = SocketReady;
  g_ncclSocketSend = SocketSend;
  g_ncclSocketRecv = SocketRecv;
  g_ncclSocketClose = SocketClose;
  g_ncclSocketMove = SocketMove;
  g_ncclFindInterfaces = FindInterfaces;
  g_ncclSocketToString = SocketToString;
  g_ncclSocketSendRecv = SocketSendRecv;
  g_ncclSocketMultiOp = SocketMultiOp;
  g_ncclSocketGetAddrFromString = SocketGetAddrFromString;
  g_ncclFindInterfaceMatchSubnet = FindInterfaceMatchSubnet;
}
