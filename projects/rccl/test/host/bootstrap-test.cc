/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/bootstrap.cc. Ranks run as threads over an in-memory socket layer and
// net plugin defined in this TU, so the root rendezvous, both rings and the collectives run for real.

#include <gtest/gtest.h>

#include <arpa/inet.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "nccl.h"
#include "comm.h"
#include "debug.h"
#include "net.h"
#include "proxy.h"
#include "ras.h"
#include "socket.h"
#include "utils.h"

#include "fakes/env_fakes.h"

// This binary links no shared nccl fakes.
uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;

namespace {
std::mutex g_logMu;
std::vector<std::string> g_warns;

bool Warned(const char* needle) {
  std::lock_guard<std::mutex> lock(g_logMu);
  for (const auto& w : g_warns)
    if (w.find(needle) != std::string::npos) return true;
  return false;
}
}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long, const char*, int, const char* fmt, ...) {
  if (level != NCCL_LOG_WARN) return;
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  std::lock_guard<std::mutex> lock(g_logMu);
  g_warns.push_back(buf);
}

// Params are read on every call so a test can flip one; unset names keep their production default.
std::map<std::string, int64_t> g_params;
std::function<int64_t(const char*, int64_t)> g_loadParam = [](const char* env, int64_t deftVal) {
  auto it = g_params.find(env);
  return it == g_params.end() ? deftVal : it->second;
};
int64_t g_rasEnable = 1;
int64_t ncclParamRasEnable() { return g_rasEnable; }

uint64_t hashCombine(uint64_t baseHash, uint64_t value) { return baseHash * 31 + value; }
uint64_t getHostHash() { return 0x4057; }
uint64_t getPidHash() { return 0x9100; }
void RegisterSignalHandlers() {}

// Real matching rules, minus the port suffix: an empty list matches everything.
int parseStringList(const char* string, struct netIf* ifList, int maxList) {
  std::stringstream ss(string);
  std::string item;
  int n = 0;
  while (n < maxList && std::getline(ss, item, ',')) {
    std::snprintf(ifList[n].prefix, sizeof(ifList[n].prefix), "%s", item.c_str());
    ifList[n++].port = -1;
  }
  return n;
}
bool matchIfList(const char* string, int, struct netIf* ifList, int listSize, bool matchExact, int*) {
  if (listSize == 0) return true;
  for (int i = 0; i < listSize; i++) {
    const char* p = ifList[i].prefix;
    if (matchExact ? std::strcmp(string, p) == 0 : std::strncmp(string, p, std::strlen(p)) == 0) return true;
  }
  return false;
}

bool g_encrypted = false;
ncclResult_t ncclGetCryptConnectionMode(bool* encrypted) {
  *encrypted = g_encrypted;
  return ncclSuccess;
}

namespace {
struct ProxyInitCall {
  std::vector<ncclSocketAddress> addrs;
  std::vector<uint64_t> uds;
  ncclSocketAddress own;  // the proxy listen socket's
};
std::mutex g_seamMu;
std::map<ncclComm*, ProxyInitCall> g_proxyInits;
ncclResult_t g_proxyInitResult = ncclSuccess;
int g_rasCommInits = 0;
ncclResult_t g_rasCommInitResult = ncclSuccess;
std::vector<rasRankInit> g_rasAdded;
ncclResult_t g_rasAddRanksResult = ncclSuccess;
int g_ifCount = 1;
union ncclSocketAddress g_subnetRemote;
}  // namespace

// On success the proxy owns the listen socket, as in production.
ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses,
                           uint64_t* peerUDS) {
  {
    std::lock_guard<std::mutex> lock(g_seamMu);
    g_proxyInits[comm] = {std::vector<ncclSocketAddress>(peerAddresses, peerAddresses + comm->nRanks),
                          std::vector<uint64_t>(peerUDS, peerUDS + comm->nRanks), sock->addr};
    if (g_proxyInitResult != ncclSuccess) return g_proxyInitResult;
  }
  (void)ncclSocketClose(sock);
  free(sock);
  return ncclSuccess;
}
ncclResult_t ncclRasCommInit(struct ncclComm*, struct rasRankInit*) {
  std::lock_guard<std::mutex> lock(g_seamMu);
  g_rasCommInits++;
  return g_rasCommInitResult;
}
// On success the RAS thread owns (and frees) the array.
ncclResult_t ncclRasAddRanks(struct rasRankInit* ranks, int nranks) {
  std::lock_guard<std::mutex> lock(g_seamMu);
  if (g_rasAddRanksResult != ncclSuccess) return g_rasAddRanksResult;
  g_rasAdded.assign(ranks, ranks + nranks);
  free(ranks);
  return ncclSuccess;
}

// In-memory socket layer: a listener is a port with a queue of pending connections, a connection is
// two endpoints whose inbound byte queues the other side appends to. Blocking calls honour the
// socket's abort flag and fail the test instead of hanging.
namespace {
using Clock = std::chrono::steady_clock;
constexpr int kEndpointFd = 10000;
constexpr int kListenFd = 20000;

struct FakeEndpoint {
  std::deque<char> in;
  int peer = -1;
  bool open = true;
  uint64_t magic = 0;  // the connector's, checked by accept
};
struct FakeListener {
  uint16_t port = 0;
  uint64_t magic = 0;
  bool open = true;
  bool root = false;
  std::deque<int> pending;
};

std::mutex g_sockMu;
std::condition_variable g_sockCv;
std::deque<FakeEndpoint> g_endpoints;
std::deque<FakeListener> g_listeners;
std::map<uint16_t, int> g_ports;
std::set<int> g_openFds;
uint16_t g_nextPort = 1000;
bool g_sockShutdown = false;
int g_liveRoots = 0;
std::chrono::milliseconds g_waitLimit(5000);
// Mirrors socket.cc: ncclSocketAccept(retry=false) hands back a rejected peer as Accepting, not BadHandshake.
ncclSocketState g_rejectedState = ncclSocketStateAccepting;
int g_readyPolls = 0;  // ncclSocketReady says "not yet" this many times first
std::atomic<int> g_multiOps{0};  // only the bidirectional socket ring uses ncclSocketMultiOp
using SockFaultFn = std::function<ncclResult_t(const char* /*op*/, struct ncclSocket*)>;
std::shared_ptr<SockFaultFn> g_sockFault;  // swapped atomically: a root thread may be calling it

void SetSockFault(SockFaultFn fn) {
  std::atomic_store(&g_sockFault, fn ? std::make_shared<SockFaultFn>(std::move(fn)) : nullptr);
}
ncclResult_t SockFault(const char* op, struct ncclSocket* sock) {
  auto fault = std::atomic_load(&g_sockFault);
  return fault ? (*fault)(op, sock) : ncclSuccess;
}
uint16_t PortOf(const union ncclSocketAddress& addr) { return ntohs(addr.sin.sin_port); }
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
}  // namespace

uint64_t ncclSocketDefaultMagic(void) { return 0x5eed; }

ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic,
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

void ncclSocketMove(struct ncclSocket* dst, struct ncclSocket* src) {
  *dst = *src;
  src->socketDescriptor = src->acceptSocketDescriptor = NCCL_INVALID_SOCKET;
  src->state = ncclSocketStateNone;
}

ncclResult_t ncclSocketListen(struct ncclSocket* sock) {
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
  l.root = sock->type == ncclSocketTypeBootstrap && sock->abortFlag == nullptr;  // only bootstrapCreateRoot's
  g_ports[port] = g_listeners.size() - 1;
  sock->socketDescriptor = kListenFd + g_listeners.size() - 1;
  sock->state = ncclSocketStateReady;
  g_openFds.insert(sock->socketDescriptor);
  return ncclSuccess;
}

ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr) {
  if (ncclResult_t res = SockFault("getaddr", sock)) return res;
  *addr = sock->addr;
  return ncclSuccess;
}

ncclResult_t ncclSocketConnect(struct ncclSocket* sock) {
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

ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* listenSock, bool retry) {
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

ncclResult_t ncclSocketReady(struct ncclSocket* sock, int* running) {
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

ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size) {
  if (ncclResult_t res = SockFault("send", sock)) return res;
  std::lock_guard<std::mutex> lock(g_sockMu);
  if (sock->state != ncclSocketStateReady) return ncclInternalError;
  FakeEndpoint& peer = g_endpoints[Endpoint(sock).peer];
  if (!peer.open) return ncclRemoteError;
  peer.in.insert(peer.in.end(), (char*)ptr, (char*)ptr + size);
  g_sockCv.notify_all();
  return ncclSuccess;
}

ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size) {
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

ncclResult_t ncclSocketSendRecv(struct ncclSocket* sendSock, void* sendPtr, int sendSize, struct ncclSocket* recvSock,
                                void* recvPtr, int recvSize) {
  if (ncclResult_t res = ncclSocketSend(sendSock, sendPtr, sendSize)) return res;
  return ncclSocketRecv(recvSock, recvPtr, recvSize);
}

// Sends never block here, so issuing them all before any receive cannot deadlock.
ncclResult_t ncclSocketMultiOp(struct ncclSocketOp* ops, int numOps) {
  g_multiOps++;
  for (int pass = NCCL_SOCKET_SEND; pass <= NCCL_SOCKET_RECV; pass++) {
    for (int i = 0; i < numOps; i++) {
      if (ops[i].op != pass || ops[i].size == 0) continue;
      ncclResult_t res = pass == NCCL_SOCKET_SEND ? ncclSocketSend(ops[i].sock, ops[i].ptr, ops[i].size)
                                                  : ncclSocketRecv(ops[i].sock, ops[i].ptr, ops[i].size);
      if (res != ncclSuccess) return res;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool wait) {
  if (sock == nullptr) return ncclSuccess;
  if (ncclResult_t res = SockFault("close", sock)) return res;
  std::unique_lock<std::mutex> lock(g_sockMu);
  int fd = sock->socketDescriptor;
  if (sock->state > ncclSocketStateNone && fd >= kListenFd) {
    FakeListener& l = g_listeners[fd - kListenFd];
    l.open = false;
    g_ports.erase(l.port);
    for (int e : l.pending) g_endpoints[e].open = false;
    if (l.root) g_liveRoots--;
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

const char* ncclSocketToString(const union ncclSocketAddress* addr, char* buf, size_t bufLen, const int) {
  std::snprintf(buf, bufLen, "fake:%u", PortOf(*addr));
  return buf;
}

ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress* ua, const char* ipPortPair) {
  const char* colon = std::strrchr(ipPortPair, ':');
  if (colon == nullptr) return ncclInvalidArgument;
  std::memset(ua, 0, sizeof(*ua));
  ua->sin.sin_family = AF_INET;
  ua->sin.sin_port = htons(std::atoi(colon + 1));
  return ncclSuccess;
}

ncclResult_t ncclFindInterfaces(char* ifNames, union ncclSocketAddress* ifAddrs, int ifNameMaxSize, int, int* nIfs) {
  *nIfs = g_ifCount;
  if (g_ifCount > 0) {
    std::snprintf(ifNames, ifNameMaxSize, "fake0");
    std::memset(ifAddrs, 0, sizeof(*ifAddrs));
    ifAddrs->sin.sin_family = AF_INET;
  }
  return ncclSuccess;
}
ncclResult_t ncclFindInterfaceMatchSubnet(char* ifName, union ncclSocketAddress* localAddr,
                                          union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int* found) {
  g_subnetRemote = *remoteAddr;
  return ncclFindInterfaces(ifName, localAddr, ifNameMaxSize, 1, found);
}

// The root's thread is the only one named, and its listener the only one closed by it.
void ncclSetThreadName(std::thread&, const char* fmt, ...) {
  std::lock_guard<std::mutex> lock(g_sockMu);
  if (std::strcmp(fmt, "NCCL BootstrapR") == 0) g_liveRoots++;
}

// In-memory net plugin. listen() writes a listener id into the handle; connect() queues a connection
// on it that accept() picks up. isend copies into the connection at once; irecv matches by tag.
namespace {
struct FakeNetConn {
  std::deque<std::pair<int, std::string>> msgs;
};
struct FakeNetListener {
  int dev = 0;
  std::deque<FakeNetConn*> pending;
};
struct FakeNetReq {
  bool send;
  FakeNetConn* conn;
  void* data;
  size_t size;
  int tag;
};
std::mutex g_netMu;
std::deque<FakeNetListener> g_netListens;
std::vector<std::unique_ptr<FakeNetConn>> g_netConns;
std::vector<std::string> g_netDevs = {"mlx5_0"};
std::atomic<int> g_netDeferConnect{0}, g_netDeferIsend{0}, g_netDeferIrecv{0}, g_netDeferTest{0};
std::atomic<int> g_netRegs{0}, g_netDeregs{0}, g_netCloses{0};
std::function<ncclResult_t(const char* /*op*/)> g_netFault;
std::set<int> g_netListenDevs;

ncclResult_t NetFault(const char* op) { return g_netFault ? g_netFault(op) : ncclSuccess; }
bool Defer(std::atomic<int>& n) {
  int v = n.load();
  while (v > 0)
    if (n.compare_exchange_weak(v, v - 1)) return true;
  return false;
}

ncclResult_t FakeNetDevices(int* ndev) {
  *ndev = g_netDevs.size();
  return NetFault("devices");
}
ncclResult_t FakeNetGetProperties(int dev, ncclNetProperties_t* props) {
  std::memset(props, 0, sizeof(*props));
  props->name = const_cast<char*>(g_netDevs[dev].c_str());
  props->port = 1;
  return NetFault("getProperties");
}
ncclResult_t FakeNetListen(void*, int dev, void* handle, void** listenComm) {
  if (ncclResult_t res = NetFault("listen")) return res;
  std::lock_guard<std::mutex> lock(g_netMu);
  g_netListens.emplace_back();
  g_netListens.back().dev = dev;
  g_netListenDevs.insert(dev);
  std::memset(handle, 0, NCCL_NET_HANDLE_MAXSIZE);
  uint64_t id = g_netListens.size();
  std::memcpy(handle, &id, sizeof(id));
  *listenComm = &g_netListens.back();
  return ncclSuccess;
}
ncclResult_t FakeNetConnect(void*, int, void* handle, void** sendComm, ncclNetDeviceHandle_t**) {
  if (ncclResult_t res = NetFault("connect")) return res;
  *sendComm = nullptr;
  if (Defer(g_netDeferConnect)) return ncclSuccess;
  uint64_t id;
  std::memcpy(&id, handle, sizeof(id));
  std::lock_guard<std::mutex> lock(g_netMu);
  if (id == 0 || id > g_netListens.size()) return ncclSystemError;
  g_netConns.push_back(std::make_unique<FakeNetConn>());
  g_netListens[id - 1].pending.push_back(g_netConns.back().get());
  *sendComm = g_netConns.back().get();
  return ncclSuccess;
}
ncclResult_t FakeNetAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t**) {
  if (ncclResult_t res = NetFault("accept")) return res;
  std::lock_guard<std::mutex> lock(g_netMu);
  auto* l = static_cast<FakeNetListener*>(listenComm);
  *recvComm = nullptr;
  if (!l->pending.empty()) {
    *recvComm = l->pending.front();
    l->pending.pop_front();
  }
  return ncclSuccess;
}
ncclResult_t FakeNetRegMr(void*, void*, size_t, int, void** mhandle) {
  if (ncclResult_t res = NetFault("regMr")) return res;
  g_netRegs++;
  *mhandle = &g_netRegs;
  return ncclSuccess;
}
ncclResult_t FakeNetDeregMr(void*, void*) {
  g_netDeregs++;
  return NetFault("deregMr");
}
ncclResult_t FakeNetIsend(void* sendComm, void* data, size_t size, int tag, void*, void*, void** request) {
  if (ncclResult_t res = NetFault("isend")) return res;
  *request = nullptr;
  if (Defer(g_netDeferIsend)) return ncclSuccess;
  std::lock_guard<std::mutex> lock(g_netMu);
  static_cast<FakeNetConn*>(sendComm)->msgs.emplace_back(tag, std::string((char*)data, size));
  *request = new FakeNetReq{true, nullptr, nullptr, 0, tag};
  return ncclSuccess;
}
ncclResult_t FakeNetIrecv(void* recvComm, int, void** data, size_t* sizes, int* tags, void**, void**, void** request) {
  if (ncclResult_t res = NetFault("irecv")) return res;
  *request = nullptr;
  if (Defer(g_netDeferIrecv)) return ncclSuccess;
  *request = new FakeNetReq{false, static_cast<FakeNetConn*>(recvComm), data[0], sizes[0], tags[0]};
  return ncclSuccess;
}
ncclResult_t FakeNetTest(void* request, int* done, int*) {
  if (ncclResult_t res = NetFault("test")) return res;
  auto* req = static_cast<FakeNetReq*>(request);
  *done = 0;
  if (Defer(g_netDeferTest)) return ncclSuccess;
  std::lock_guard<std::mutex> lock(g_netMu);
  if (!req->send) {
    auto& msgs = req->conn->msgs;
    auto it = std::find_if(msgs.begin(), msgs.end(), [&](const auto& m) { return m.first == req->tag; });
    if (it == msgs.end()) return ncclSuccess;
    std::memcpy(req->data, it->second.data(), std::min(req->size, it->second.size()));
    msgs.erase(it);
  }
  *done = 1;
  delete req;
  return ncclSuccess;
}
ncclResult_t FakeNetClose(void*) {
  g_netCloses++;
  return NetFault("close");
}

ncclNet_t MakeFakeNet() {
  ncclNet_t net{};
  net.name = "fake";
  net.devices = FakeNetDevices;
  net.getProperties = FakeNetGetProperties;
  net.listen = FakeNetListen;
  net.connect = FakeNetConnect;
  net.accept = FakeNetAccept;
  net.regMr = FakeNetRegMr;
  net.deregMr = FakeNetDeregMr;
  net.isend = FakeNetIsend;
  net.irecv = FakeNetIrecv;
  net.test = FakeNetTest;
  net.closeSend = FakeNetClose;
  net.closeRecv = FakeNetClose;
  net.closeListen = FakeNetClose;
  return net;
}
ncclNet_t g_fakeNet = MakeFakeNet();

void ResetSeams() {
  {
    std::lock_guard<std::mutex> lock(g_logMu);
    g_warns.clear();
  }
  g_params.clear();
  g_rasEnable = 1;
  g_encrypted = false;
  g_ifCount = 1;
  g_proxyInits.clear();
  g_proxyInitResult = ncclSuccess;
  g_rasCommInits = 0;
  g_rasCommInitResult = ncclSuccess;
  g_rasAdded.clear();
  g_rasAddRanksResult = ncclSuccess;
  {
    std::lock_guard<std::mutex> lock(g_sockMu);
    g_endpoints.clear();
    g_listeners.clear();
    g_ports.clear();
    g_openFds.clear();
    g_sockShutdown = false;
    g_liveRoots = 0;
  }
  g_waitLimit = std::chrono::milliseconds(5000);
  g_rejectedState = ncclSocketStateAccepting;
  g_readyPolls = 0;
  g_multiOps = 0;
  SetSockFault(nullptr);
  {
    std::lock_guard<std::mutex> lock(g_netMu);
    g_netListens.clear();
    g_netConns.clear();
    g_netListenDevs.clear();
  }
  g_netDevs = {"mlx5_0"};
  g_netDeferConnect = g_netDeferIsend = g_netDeferIrecv = g_netDeferTest = 0;
  g_netRegs = g_netDeregs = g_netCloses = 0;
  g_netFault = nullptr;
  ResetEnvFakes();
}

// Lets every detached root thread unwind before the next test reuses the fake network.
void ShutdownSockets() {
  std::unique_lock<std::mutex> lock(g_sockMu);
  g_sockShutdown = true;
  g_sockCv.notify_all();
  auto deadline = Clock::now() + g_waitLimit;
  while (g_liveRoots > 0 && Clock::now() < deadline) g_sockCv.wait_for(lock, std::chrono::milliseconds(1));
  EXPECT_EQ(0, g_liveRoots) << "a bootstrap root thread did not finish";
}
}  // namespace

#include "fakes/param_redirect.h"

#include BOOTSTRAP_CC_PATH

namespace {

struct Rank {
  std::unique_ptr<ncclComm> comm = std::make_unique<ncclComm>();
  uint32_t abort = 0;
  std::vector<int> topParentRanks;
};

class BootstrapMicrotest : public ::testing::Test {
 protected:
  std::vector<std::unique_ptr<Rank>> ranks_;
  std::vector<std::unique_ptr<Rank>> parents_;  // the comm a split or grow starts from
  ncclBootstrapHandle handle_{};
  std::chrono::milliseconds watchdog_ = g_waitLimit;
  uint32_t abort_ = 0;  // for sockets the test thread drives itself
  Rank self_;           // the comm sendToRoot reads its abort flag from
  std::thread::id main_ = std::this_thread::get_id();

  void SetUp() override {
    ResetSeams();
    bootstrapNetInitDone = 0;
    ASSERT_EQ(ncclSuccess, bootstrapNetInit());
  }
  void TearDown() override {
    for (auto& r : ranks_) parents_.push_back(std::move(r));
    for (auto& r : parents_) {
      if (r->comm->bootstrap == nullptr) continue;
      __atomic_store_n(&r->abort, 1u, __ATOMIC_RELEASE);  // a failed test may leave unmatched connections
      (void)bootstrapClose(r->comm->bootstrap);
    }
    ShutdownSockets();
  }
  // For a test that leaves a root waiting and then reuses the fake network.
  void RestartSockets() {
    ShutdownSockets();
    std::lock_guard<std::mutex> lock(g_sockMu);
    g_sockShutdown = false;
  }

  // A connected pair over a fresh listener; `a` is the connecting side.
  void Pair(struct ncclSocket* a, struct ncclSocket* b) {
    struct ncclSocket l;
    ASSERT_EQ(ncclSuccess, ncclSocketInit(&l, &bootstrapNetIfAddr, 7, ncclSocketTypeBootstrap, &abort_));
    ASSERT_EQ(ncclSuccess, ncclSocketListen(&l));
    ASSERT_EQ(ncclSuccess, ncclSocketInit(a, &l.addr, 7, ncclSocketTypeBootstrap));
    ASSERT_EQ(ncclSuccess, ncclSocketConnect(a));
    ASSERT_EQ(ncclSuccess, ncclSocketInit(b));
    ASSERT_EQ(ncclSuccess, ncclSocketAccept(b, &l));
    ASSERT_EQ(ncclSuccess, ncclSocketClose(&l));
  }
  // Plays `rank` against the root behind handle_: listens for the answer, then checks in.
  void CheckIn(int rank, int nranks, struct ncclSocket* listen, int nroots = 1, int iroot = 0, int offset = 0) {
    struct extInfo info = {};
    info.rank = rank;
    info.nranks = nranks;
    info.nroots = nroots;
    info.iroot = iroot;
    info.offset = offset;
    info.connectInfo.fwd.handle[0] = char(1 + rank);
    info.connectInfo.revHandle[0] = char(101 + rank);
    ASSERT_EQ(ncclSuccess, ncclSocketInit(listen, &bootstrapNetIfAddr, handle_.magic, ncclSocketTypeBootstrap, &abort_));
    ASSERT_EQ(ncclSuccess, ncclSocketListen(listen));
    info.listenRootAddress = listen->addr;
    self_.comm->abortFlag = &self_.abort;
    ASSERT_EQ(ncclSuccess, sendToRoot(&handle_, self_.comm.get(), &info));
  }
  struct ringConnectInfo Answer(struct ncclSocket* listen) {
    struct ringConnectInfo out = {};
    struct ncclSocket sock;
    EXPECT_EQ(ncclSuccess, bootstrapAccept(&sock, listen, &abort_));
    EXPECT_EQ(ncclSuccess, socketRecv(&sock, &out, sizeof(out)));
    (void)ncclSocketClose(&sock);
    (void)ncclSocketClose(listen);
    return out;
  }
  void WaitRoots() {
    std::unique_lock<std::mutex> lock(g_sockMu);
    auto deadline = Clock::now() + g_waitLimit;
    while (g_liveRoots > 0 && Clock::now() < deadline) g_sockCv.wait_for(lock, std::chrono::milliseconds(1));
    EXPECT_EQ(0, g_liveRoots);
  }
  size_t PendingOn(const struct ncclSocket& listen) {
    std::lock_guard<std::mutex> lock(g_sockMu);
    return g_listeners[listen.socketDescriptor - kListenFd].pending.size();
  }
  int OpenListeners() {
    std::lock_guard<std::mutex> lock(g_sockMu);
    return std::count_if(g_openFds.begin(), g_openFds.end(), [](int fd) { return fd >= kListenFd; });
  }

  ncclComm* Comm(int r) { return ranks_[r]->comm.get(); }
  void* State(int r) { return Comm(r)->bootstrap; }
  void MakeRanks(int n) {
    for (int r = 0; r < n; r++) {
      auto rank = std::make_unique<Rank>();
      rank->comm->rank = r;
      rank->comm->nRanks = n;
      rank->comm->cudaDev = r;
      rank->comm->nvmlDev = 10 + r;
      rank->comm->abortFlag = &rank->abort;
      rank->comm->ncclNet = &g_fakeNet;
      ranks_.push_back(std::move(rank));
    }
  }
  // One thread per rank; past the wait limit every abort flag is raised so busy loops unwind too.
  std::vector<ncclResult_t> OnRanks(const std::function<ncclResult_t(int)>& fn) {
    std::vector<ncclResult_t> res(ranks_.size(), ncclSuccess);
    std::atomic<int> running(ranks_.size());
    std::vector<std::thread> threads;
    for (int r = 0; r < (int)ranks_.size(); r++)
      threads.emplace_back([&, r] {
        res[r] = fn(r);
        running--;
      });
    auto deadline = Clock::now() + watchdog_;
    while (running > 0 && Clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (running > 0)
      for (auto& rank : ranks_) __atomic_store_n(&rank->abort, 1u, __ATOMIC_RELEASE);
    for (auto& t : threads) t.join();
    return res;
  }
  static bool AllOk(const std::vector<ncclResult_t>& res) {
    return std::all_of(res.begin(), res.end(), [](ncclResult_t r) { return r == ncclSuccess; });
  }
  ncclResult_t InitRanks(int n) {
    MakeRanks(n);
    ncclResult_t res = bootstrapGetUniqueId(&handle_, nullptr);
    if (res != ncclSuccess) return res;
    auto all = OnRanks([&](int r) { return bootstrapInit(1, &handle_, Comm(r), nullptr); });
    for (ncclResult_t r : all)
      if (r != ncclSuccess) return r;
    WaitRoots();  // so a fault hook installed next is never reached from the root thread
    return ncclSuccess;
  }
  std::vector<ncclResult_t> CloseRanks() {
    auto res = OnRanks([&](int r) { return bootstrapClose(State(r)); });
    for (int r = 0; r < (int)ranks_.size(); r++)
      if (res[r] == ncclSuccess) Comm(r)->bootstrap = nullptr;
    return res;
  }
  // Every rank contributes `size` bytes of (rank, slice) and must end with every rank's slice.
  // Works across several comms at once: each rank gathers within its own.
  void ExpectAllGather(int size) {
    std::vector<std::vector<char>> data(ranks_.size());
    for (int r = 0; r < (int)ranks_.size(); r++) {
      data[r].resize((size_t)Comm(r)->nRanks * size);
      std::fill_n(data[r].begin() + (size_t)Comm(r)->rank * size, size, char('A' + Comm(r)->rank));
    }
    EXPECT_TRUE(AllOk(OnRanks([&](int r) { return bootstrapAllGather(State(r), data[r].data(), size); })));
    for (int r = 0; r < (int)ranks_.size(); r++)
      for (int s = 0; s < Comm(r)->nRanks; s++)
        EXPECT_EQ(std::string(size, char('A' + s)), std::string(&data[r][(size_t)s * size], size))
            << "rank " << r << " slice " << s;
  }
  // Splits ranks_ (which become parents_) into one child comm per group; group members are parent ranks in
  // child-rank order.
  std::vector<ncclResult_t> Split(const std::vector<std::vector<int>>& groups, bool shareResources) {
    parents_ = std::move(ranks_);
    ranks_.clear();
    std::vector<std::vector<int>*> parentRanks;
    std::vector<ncclComm*> parentOf;
    for (auto& p : parents_) {
      p->comm->shareResources = shareResources;
      p->topParentRanks.resize(parents_.size());
      for (int i = 0; i < (int)parents_.size(); i++) p->topParentRanks[i] = 100 + i;
      p->comm->topParentRanks = p->topParentRanks.data();
    }
    std::vector<std::vector<int>> members = groups;
    for (auto& g : members) {
      for (int i = 0; i < (int)g.size(); i++) {
        auto child = std::make_unique<Rank>();
        child->comm->rank = i;
        child->comm->nRanks = g.size();
        child->comm->abortFlag = &child->abort;
        child->comm->ncclNet = &g_fakeNet;
        child->topParentRanks.resize(g.size());
        child->comm->topParentRanks = child->topParentRanks.data();
        parentRanks.push_back(&g);
        parentOf.push_back(parents_[g[i]]->comm.get());
        ranks_.push_back(std::move(child));
      }
    }
    return OnRanks([&](int r) {
      return bootstrapSplit(0xC0FFEE, Comm(r), parentOf[r], 0, 0, parentRanks[r]->data());
    });
  }
};

TEST_F(BootstrapMicrotest, RootMath_EveryRankPastTheOffsetBelongsToExactlyOneRoot) {
  for (int nRanks = 1; nRanks <= 12; nRanks++)
    for (int nRoots = 1; nRoots <= std::min(4, nRanks); nRoots++)
      for (int offset = 0; offset < std::min(3, nRanks - nRoots + 1); offset++) {
        SCOPED_TRACE(testing::Message() << nRanks << " ranks " << nRoots << " roots offset " << offset);
        int next = offset;
        for (int root = 0; root < nRoots; root++) {
          int first = firstRankFromRoot(root, nRanks, nRoots, offset);
          EXPECT_EQ(next, first);
          EXPECT_EQ(nRankFromRoot(root, nRanks, nRoots, offset), nRankFromRoot(root + nRoots, nRanks, nRoots, offset));
          for (int k = 0; k < nRankFromRoot(root, nRanks, nRoots, offset); k++, next++) {
            EXPECT_EQ(root, rootIdFromRank(next, nRanks, nRoots, offset));
            EXPECT_EQ(k, localIdFromRoot(next, root, nRanks, nRoots, offset));
            EXPECT_EQ(k == 0, isFirstFromRoot(next, root, nRanks, nRoots, offset));
          }
        }
        EXPECT_EQ(nRanks, next);
        if (offset > 0) EXPECT_EQ(-1, rootIdFromRank(offset - 1, nRanks, nRoots, offset));
      }
  EXPECT_EQ(-1, rootIdFromRank(3, 8, 0, 0));
  EXPECT_EQ(0, firstRankFromRoot(-1, 8, 2, 0));
  EXPECT_EQ(0, nRankFromRoot(-1, 8, 2, 0));
  EXPECT_EQ(5, localIdFromRoot(5, -1, 8, 2, 0));
}

TEST_F(BootstrapMicrotest, BidirEnabled_SocketKnobIsOnUnlessDisabledOrNetTakesOver) {
  EXPECT_FALSE(bootstrapBidirEnabled(2, 0));
  EXPECT_TRUE(bootstrapBidirEnabled(3, 0));
  g_params["BOOTSTRAP_BIDIR_ALLGATHER"] = 0;
  EXPECT_FALSE(bootstrapBidirEnabled(3, 0));
  g_params["BOOTSTRAP_BIDIR_ALLGATHER"] = 1;
  g_params["OOB_NET_ENABLE"] = 1;
  EXPECT_FALSE(bootstrapBidirEnabled(3, 0));
  EXPECT_FALSE(bootstrapBidirEnabled(3, 2));
  EXPECT_FALSE(bootstrapBidirEnabled(3, -1));
}

TEST_F(BootstrapMicrotest, BidirEnabled_NetKnobIsATristateGatedByTheThreshold) {
  EXPECT_FALSE(bootstrapBidirEnabled(200, 1));  // net OOB off by default
  g_params["OOB_NET_ENABLE"] = 1;
  EXPECT_FALSE(bootstrapBidirEnabled(200, 1));  // bidir net off by default
  g_params["BOOTSTRAP_BIDIR_NET"] = 1;
  EXPECT_TRUE(bootstrapBidirEnabled(3, 1));
  g_params["BOOTSTRAP_BIDIR_NET"] = -1;
  g_params["BOOTSTRAP_BIDIR_THRESHOLD"] = 8;
  EXPECT_FALSE(bootstrapBidirEnabled(7, 1));
  EXPECT_TRUE(bootstrapBidirEnabled(8, 1));
  g_params["BOOTSTRAP_BIDIR_THRESHOLD"] = 0;
  EXPECT_FALSE(bootstrapBidirEnabled(8, 1));
}

TEST_F(BootstrapMicrotest, NetEnabled_AutoModeUsesTheBidirThreshold) {
  EXPECT_FALSE(bootstrapNetEnabledEffective(1000));
  g_params["OOB_NET_ENABLE"] = 2;
  EXPECT_TRUE(bootstrapNetEnabledEffective(1));
  g_params["OOB_NET_ENABLE"] = -1;
  g_params["BOOTSTRAP_BIDIR_THRESHOLD"] = 4;
  EXPECT_FALSE(bootstrapNetEnabledEffective(3));
  EXPECT_TRUE(bootstrapNetEnabledEffective(4));
  g_params["BOOTSTRAP_BIDIR_THRESHOLD"] = 0;
  EXPECT_FALSE(bootstrapNetEnabledEffective(4));
}

TEST_F(BootstrapMicrotest, CheckAbort_ReadsTheFlagOnlyOnceEveryTenThousandCalls) {
  uint32_t flag = 1;
  int cntr = 1;
  EXPECT_EQ(ncclSuccess, checkAbort(&flag, &cntr));
  EXPECT_EQ(2, cntr);
  cntr = BOOTSTRAP_N_CHECK_ABORT - 1;
  EXPECT_EQ(ncclSuccess, checkAbort(&flag, &cntr));
  EXPECT_EQ(0, cntr);
  EXPECT_EQ(ncclInternalError, checkAbort(&flag, &cntr));
  EXPECT_EQ(0, cntr);
  EXPECT_EQ(ncclSuccess, checkAbort(nullptr, &cntr));
  flag = 0;
  cntr = 0;
  EXPECT_EQ(ncclSuccess, checkAbort(&flag, &cntr));
}

TEST_F(BootstrapMicrotest, SocketRecv_SizePrefixedMessageFitsOrIsRejectedAsTruncated) {
  struct ncclSocket a, b;
  Pair(&a, &b);
  char out[8] = {};
  ASSERT_EQ(ncclSuccess, socketSend(&a, (void*)"abc", 3));
  EXPECT_EQ(ncclSuccess, socketRecv(&b, out, sizeof(out)));
  EXPECT_STREQ("abc", out);
  ASSERT_EQ(ncclSuccess, socketSend(&a, nullptr, 0));  // header only
  EXPECT_EQ(ncclSuccess, socketRecv(&b, nullptr, 0));
  ASSERT_EQ(ncclSuccess, socketSend(&a, (void*)"too long", 8));
  EXPECT_EQ(ncclInternalError, socketRecv(&b, out, 4));
  EXPECT_TRUE(Warned("Message truncated : received 8 bytes instead of 4"));
  (void)ncclSocketClose(&a);
  (void)ncclSocketClose(&b);
}

TEST_F(BootstrapMicrotest, SocketSendRecv_ShortMessageReceivedTooLongRejected) {
  struct ncclSocket a, b;
  Pair(&a, &b);
  char fromA[4] = {}, fromB[4] = {};
  ASSERT_EQ(ncclSuccess, socketSend(&b, (void*)"xy", 2));  // b's half of the exchange, sent ahead
  EXPECT_EQ(ncclSuccess, socketSendRecv(&a, (void*)"pq", 2, &a, fromB, sizeof(fromB)));
  EXPECT_STREQ("xy", fromB);
  EXPECT_EQ(ncclSuccess, socketRecv(&b, fromA, sizeof(fromA)));
  EXPECT_STREQ("pq", fromA);
  ASSERT_EQ(ncclSuccess, socketSend(&b, (void*)"long", 4));
  EXPECT_EQ(ncclInternalError, socketSendRecv(&a, (void*)"pq", 2, &a, fromB, 2));
  EXPECT_TRUE(Warned("Message truncated : received 4 bytes instead of 2"));
  (void)ncclSocketClose(&a);
  (void)ncclSocketClose(&b);
}

TEST_F(BootstrapMicrotest, SocketDoubleSendRecv_EitherDirectionTooLongIsRejected) {
  struct ncclSocket a, b;
  Pair(&a, &b);
  char r1[4] = {}, r2[4] = {};
  // Both "peers" are b, so its two size words and both payloads are queued up front.
  int three = 3, one = 1;
  ASSERT_EQ(ncclSuccess, ncclSocketSend(&b, &three, sizeof(int)));
  ASSERT_EQ(ncclSuccess, ncclSocketSend(&b, &one, sizeof(int)));
  ASSERT_EQ(ncclSuccess, ncclSocketSend(&b, (void*)"abcd", 4));
  struct ncclSocketOp ops[4] = {{NCCL_SOCKET_SEND, &a, (void*)"s1", 2, 0},
                                {NCCL_SOCKET_RECV, &a, r1, 4, 0},
                                {NCCL_SOCKET_SEND, &a, (void*)"s2", 2, 0},
                                {NCCL_SOCKET_RECV, &a, r2, 4, 0}};
  EXPECT_EQ(ncclSuccess, socketDoubleSendRecv(ops));
  EXPECT_EQ(3, ops[1].size);
  EXPECT_EQ(1, ops[3].size);
  EXPECT_STREQ("abc", r1);
  EXPECT_STREQ("d", r2);
  int big = 9;
  ASSERT_EQ(ncclSuccess, ncclSocketSend(&b, &one, sizeof(int)));
  ASSERT_EQ(ncclSuccess, ncclSocketSend(&b, &big, sizeof(int)));
  ops[1].size = ops[3].size = 4;
  EXPECT_EQ(ncclInternalError, socketDoubleSendRecv(ops));
  EXPECT_TRUE(Warned("Message truncated : received 1,9 bytes instead of 4,4"));
  (void)ncclSocketClose(&a);
  (void)ncclSocketClose(&b);
}

TEST_F(BootstrapMicrotest, NetMultiOp_RejectsEachMalformedOpBeforePostingAny) {
  char buf[4];
  int comm = 0;
  EXPECT_EQ(ncclInvalidArgument, netMultiOp(&g_fakeNet, nullptr, 1, nullptr));
  EXPECT_TRUE(Warned("netMultiOp: invalid arguments"));
  struct ncclNetOp ok = {NCCL_NET_OP_SEND, &comm, buf, 4, nullptr, 0, nullptr, 0};
  EXPECT_EQ(ncclInvalidArgument, netMultiOp(&g_fakeNet, &ok, 0, nullptr));
  struct ncclNetOp bad[4] = {ok, ok, ok, ok};
  bad[0].op = 7;
  bad[1].comm = nullptr;
  bad[2].size = -1;
  bad[3].data = nullptr;
  const char* warns[4] = {"invalid op 7 at index 1", "invalid comm at index 1", "invalid size -1 at index 1",
                          "NULL data with size 4 at index 1"};
  for (int i = 0; i < 4; i++) {
    struct ncclNetOp ops[2] = {ok, bad[i]};
    EXPECT_EQ(ncclInvalidArgument, netMultiOp(&g_fakeNet, ops, 2, nullptr)) << warns[i];
    EXPECT_TRUE(Warned(warns[i])) << warns[i];
  }
  EXPECT_EQ(0u, g_netConns.size());
}

TEST_F(BootstrapMicrotest, NetMultiOp_PollsUntilDeferredPostsAndCompletionsLand) {
  void *listen, *send, *recv;
  char handle[NCCL_NET_HANDLE_MAXSIZE];
  ASSERT_EQ(ncclSuccess, g_fakeNet.listen(nullptr, 0, handle, &listen));
  ASSERT_EQ(ncclSuccess, g_fakeNet.connect(nullptr, 0, handle, &send, nullptr));
  ASSERT_EQ(ncclSuccess, g_fakeNet.accept(listen, &recv, nullptr));
  char out[3] = {};
  struct ncclNetOp ops[2] = {{NCCL_NET_OP_RECV, recv, out, 2, nullptr, 5, nullptr, 0},
                             {NCCL_NET_OP_SEND, send, (void*)"hi", 2, nullptr, 5, nullptr, 0}};
  g_netDeferIrecv = 2;
  g_netDeferIsend = 2;
  g_netDeferTest = 3;
  uint32_t flag = 0;
  EXPECT_EQ(ncclSuccess, netMultiOp(&g_fakeNet, ops, 2, &flag));
  EXPECT_STREQ("hi", out);
  EXPECT_EQ(1, ops[0].done);
  EXPECT_EQ(nullptr, ops[0].req);
  EXPECT_EQ(0, g_netDeferTest.load());
  struct ncclNetOp empty[2] = {{NCCL_NET_OP_RECV, recv, nullptr, 0, nullptr, 6, nullptr, 0},
                               {NCCL_NET_OP_SEND, send, nullptr, 0, nullptr, 6, nullptr, 0}};
  EXPECT_EQ(ncclSuccess, netMultiOp(&g_fakeNet, empty, 2, &flag));  // no payload needs no buffer
}

TEST_F(BootstrapMicrotest, NetMultiOp_AbortStopsTheWaitAndPluginErrorsPropagate) {
  void *listen, *send, *recv;
  char handle[NCCL_NET_HANDLE_MAXSIZE], out[2];
  ASSERT_EQ(ncclSuccess, g_fakeNet.listen(nullptr, 0, handle, &listen));
  ASSERT_EQ(ncclSuccess, g_fakeNet.connect(nullptr, 0, handle, &send, nullptr));
  ASSERT_EQ(ncclSuccess, g_fakeNet.accept(listen, &recv, nullptr));
  struct ncclNetOp recvOp = {NCCL_NET_OP_RECV, recv, out, 2, nullptr, 1, nullptr, 0};
  uint32_t flag = 1;  // nothing was sent, so only the abort ends this
  EXPECT_EQ(ncclInternalError, netMultiOp(&g_fakeNet, &recvOp, 1, &flag));
  g_netFault = [](const char* op) { return std::strcmp(op, "isend") == 0 ? ncclSystemError : ncclSuccess; };
  struct ncclNetOp sendOp = {NCCL_NET_OP_SEND, send, out, 2, nullptr, 1, nullptr, 0};
  EXPECT_EQ(ncclSystemError, netMultiOp(&g_fakeNet, &sendOp, 1, nullptr));
}

TEST_F(BootstrapMicrotest, NetIsendIrecv_ADoneOpIsNotPostedAgain) {
  int done = 1;
  void* req = nullptr;
  EXPECT_EQ(ncclSuccess, netIsend(&g_fakeNet, nullptr, nullptr, 0, nullptr, 0, &req, &done));
  EXPECT_EQ(ncclSuccess, netIrecv(&g_fakeNet, nullptr, nullptr, 0, nullptr, 0, &req, &done));
  EXPECT_EQ(nullptr, req);
}

// Loops one comm back to itself so the ring helpers can be driven on a single thread.
struct NetLoop {
  void *listen = nullptr, *send = nullptr, *recv = nullptr;
  NetLoop() {
    char handle[NCCL_NET_HANDLE_MAXSIZE];
    EXPECT_EQ(ncclSuccess, g_fakeNet.listen(nullptr, 0, handle, &listen));
    EXPECT_EQ(ncclSuccess, g_fakeNet.connect(nullptr, 0, handle, &send, nullptr));
    EXPECT_EQ(ncclSuccess, g_fakeNet.accept(listen, &recv, nullptr));
  }
};

TEST_F(BootstrapMicrotest, NetRingAllGather_RegistrationOrTransferFailureStillDeregisters) {
  NetLoop loop;
  char data[8] = "abcdefg";
  for (const char* op : {"regMr", "isend"}) {
    SCOPED_TRACE(op);
    std::atomic<int> calls(0);
    g_netFault = [&](const char* o) {
      return std::strcmp(o, op) == 0 && ++calls == 2 ? ncclSystemError : ncclSuccess;
    };
    g_netRegs = g_netDeregs = 0;
    EXPECT_EQ(ncclSystemError, netRingAllGather(&g_fakeNet, loop.send, loop.recv, 0, 4, data, 2, nullptr));
    EXPECT_EQ(g_netRegs.load(), g_netDeregs.load());
  }
}

TEST_F(BootstrapMicrotest, NetBiDirRingAllGather_EachFailurePointStillDeregistersWhatItRegistered) {
  NetLoop fwd, rev;
  char data[8] = "abcdefg";
  struct Fault {
    const char* op;
    int nth, regs;
  } faults[] = {{"regMr", 1, 0}, {"regMr", 2, 1}, {"regMr", 3, 2}, {"regMr", 4, 3}, {"isend", 1, 4}, {"isend", 3, 4}};
  for (const auto& f : faults) {
    SCOPED_TRACE(testing::Message() << f.op << " #" << f.nth);
    std::atomic<int> calls(0);
    g_netFault = [&](const char* o) {
      return std::strcmp(o, f.op) == 0 && ++calls == f.nth ? ncclSystemError : ncclSuccess;
    };
    g_netRegs = g_netDeregs = 0;
    EXPECT_EQ(ncclSystemError,
              netBiDirRingAllGather(&g_fakeNet, fwd.send, fwd.recv, rev.send, rev.recv, 0, 4, data, 2, nullptr));
    EXPECT_EQ(f.regs, g_netRegs.load());
    EXPECT_EQ(f.regs, g_netDeregs.load());
  }
}

TEST_F(BootstrapMicrotest, Accept_AbortRaisedBeforehandFailsWithoutAccepting) {
  struct ncclSocket l, sock;
  ASSERT_EQ(ncclSuccess, ncclSocketInit(&l, &bootstrapNetIfAddr, 7, ncclSocketTypeBootstrap, &abort_));
  ASSERT_EQ(ncclSuccess, ncclSocketListen(&l));
  uint32_t flag = 1;
  EXPECT_EQ(ncclInternalError, bootstrapAccept(&sock, &l, &flag));
  (void)ncclSocketClose(&l);
}

// The contract bootstrapAccept was written for: a rejected peer comes back as BadHandshake and is skipped.
TEST_F(BootstrapMicrotest, Accept_BadHandshakePeerIsClosedAndTheNextOneAccepted) {
  g_rejectedState = ncclSocketStateBadHandshake;
  struct ncclSocket l, stray, good, sock;
  ASSERT_EQ(ncclSuccess, ncclSocketInit(&l, &bootstrapNetIfAddr, 7, ncclSocketTypeBootstrap, &abort_));
  ASSERT_EQ(ncclSuccess, ncclSocketListen(&l));
  ASSERT_EQ(ncclSuccess, ncclSocketInit(&stray, &l.addr, 8, ncclSocketTypeBootstrap));
  ASSERT_EQ(ncclSuccess, ncclSocketConnect(&stray));
  ASSERT_EQ(ncclSuccess, ncclSocketInit(&good, &l.addr, 7, ncclSocketTypeBootstrap));
  ASSERT_EQ(ncclSuccess, ncclSocketConnect(&good));
  ASSERT_EQ(ncclSuccess, socketSend(&good, (void*)"ok", 2));

  EXPECT_EQ(ncclSuccess, bootstrapAccept(&sock, &l, &abort_));

  char out[3] = {};
  EXPECT_EQ(ncclSuccess, socketRecv(&sock, out, 2));
  EXPECT_STREQ("ok", out);
  for (auto* s : {&l, &stray, &good, &sock}) (void)ncclSocketClose(s);
}

// Bug, pinned as observed: bootstrapAccept only retries on BadHandshake, so it hands back the rejected peer's
// unconnected socket. A fix in bootstrapAccept flips the state and recv expectations below.
TEST_F(BootstrapMicrotest, Accept_RejectedPeerAsSocketCcReportsIt_IsReturnedUnconnected) {
  struct ncclSocket l, stray, good, sock;
  ASSERT_EQ(ncclSuccess, ncclSocketInit(&l, &bootstrapNetIfAddr, 7, ncclSocketTypeBootstrap, &abort_));
  ASSERT_EQ(ncclSuccess, ncclSocketListen(&l));
  ASSERT_EQ(ncclSuccess, ncclSocketInit(&stray, &l.addr, 8, ncclSocketTypeBootstrap));
  ASSERT_EQ(ncclSuccess, ncclSocketConnect(&stray));
  ASSERT_EQ(ncclSuccess, ncclSocketInit(&good, &l.addr, 7, ncclSocketTypeBootstrap));
  ASSERT_EQ(ncclSuccess, ncclSocketConnect(&good));

  EXPECT_EQ(ncclSuccess, bootstrapAccept(&sock, &l, &abort_));

  EXPECT_EQ(ncclSocketStateAccepting, sock.state);
  char out[2];
  EXPECT_NE(ncclSuccess, socketRecv(&sock, out, 2));
  for (auto* s : {&l, &stray, &good, &sock}) (void)ncclSocketClose(s);
}

TEST_F(BootstrapMicrotest, NetInit_CommIdPicksTheMatchingSubnetOrFails) {
  bootstrapNetInitDone = 0;
  SetMicroEnv("NCCL_COMM_ID", "not-an-address");
  EXPECT_EQ(ncclInvalidArgument, bootstrapNetInit());
  EXPECT_TRUE(Warned("Invalid NCCL_COMM_ID"));
  SetMicroEnv("NCCL_COMM_ID", "10.1.2.3:4242");
  g_ifCount = 0;
  EXPECT_EQ(ncclSystemError, bootstrapNetInit());
  EXPECT_TRUE(Warned("No usable listening interface found"));
  EXPECT_EQ(4242, PortOf(g_subnetRemote));
  g_ifCount = 1;
  EXPECT_EQ(ncclSuccess, bootstrapNetInit());
  EXPECT_EQ(1, bootstrapNetInitDone);
  EXPECT_STREQ("fake0", bootstrapNetIfName);
}

TEST_F(BootstrapMicrotest, NetInit_NoInterfaceFailsAndSuccessIsCachedForever) {
  bootstrapNetInitDone = 0;
  g_ifCount = 0;
  EXPECT_EQ(ncclInvalidUsage, bootstrapNetInit());
  EXPECT_TRUE(Warned("Bootstrap : no socket interface found"));
  EXPECT_EQ(0, bootstrapNetInitDone);
  g_ifCount = 1;
  ASSERT_EQ(ncclSuccess, bootstrapNetInit());
  g_ifCount = 0;
  EXPECT_EQ(ncclSuccess, bootstrapNetInit());
}

TEST_F(BootstrapMicrotest, GetUniqueId_CommIdFromEnvIsUsedAsIsWithoutARoot) {
  SetMicroEnv("NCCL_COMM_ID", "10.1.2.3:4242");
  ncclBootstrapHandle h;
  std::memset(&h, 0xff, sizeof(h));
  EXPECT_EQ(ncclSuccess, bootstrapGetUniqueId(&h, nullptr));
  EXPECT_EQ((uint64_t)NCCL_MAGIC, h.magic);
  EXPECT_EQ(4242, PortOf(h.addr));
  EXPECT_EQ(0, h.nRanks);
  EXPECT_EQ(0, OpenListeners());
}

TEST_F(BootstrapMicrotest, GetUniqueId_CommIdFromEnvRejectsGrowAndBadAddresses) {
  SetMicroEnv("NCCL_COMM_ID", "10.1.2.3:4242");
  ncclBootstrapHandle h;
  EXPECT_EQ(ncclInvalidUsage, bootstrapGetUniqueId(&h, self_.comm.get()));
  EXPECT_TRUE(Warned("should not be called when NCCL_COMM_ID is set"));
  SetMicroEnv("NCCL_COMM_ID", "nonsense");
  EXPECT_EQ(ncclInvalidArgument, bootstrapGetUniqueId(&h, nullptr));
  EXPECT_TRUE(Warned("Invalid NCCL_COMM_ID"));
}

TEST_F(BootstrapMicrotest, GetUniqueId_GrowDerivesTheNextChildMagicAndStartsARoot) {
  self_.comm->magic = 0x1234;
  self_.comm->childCount = 2;
  self_.comm->nRanks = 6;
  ncclBootstrapHandle h;
  ASSERT_EQ(ncclSuccess, bootstrapGetUniqueId(&h, self_.comm.get()));
  EXPECT_EQ(hashCombine(0x1234, 3), h.magic);
  EXPECT_EQ(6, h.nRanks);
  EXPECT_NE(0, PortOf(h.addr));
  EXPECT_EQ(1, OpenListeners());
}

TEST_F(BootstrapMicrotest, CreateRoot_EachSocketStepFailureIsReturnedAndNoThreadStarts) {
  for (const char* step : {"init", "listen", "getaddr"}) {
    SCOPED_TRACE(step);
    SetSockFault([step](const char* op, struct ncclSocket*) {
      return std::strcmp(op, step) == 0 ? ncclSystemError : ncclSuccess;
    });
    ncclBootstrapHandle h = {};
    EXPECT_EQ(ncclSystemError, bootstrapCreateRoot(&h, false));
    EXPECT_EQ(0, g_liveRoots);
  }
}

// Bug, pinned as observed: when ncclSocketGetAddr fails after a successful listen, the fail path frees the socket
// without closing it. A fix that closes it makes OpenListeners() 0.
TEST_F(BootstrapMicrotest, CreateRoot_GetAddrFails_LeaksTheListeningSocket) {
  SetSockFault([](const char* op, struct ncclSocket*) {
    return std::strcmp(op, "getaddr") == 0 ? ncclSystemError : ncclSuccess;
  });
  ncclBootstrapHandle h = {};
  ASSERT_EQ(ncclSystemError, bootstrapCreateRoot(&h, false));
  EXPECT_EQ(1, OpenListeners());
}

TEST_F(BootstrapMicrotest, Root_EveryArrivalOrderGivesEachRankItsNextAndItsPrevsReverseHandle) {
  std::vector<int> order = {0, 1, 2, 3};
  do {
    SCOPED_TRACE(testing::Message() << order[0] << order[1] << order[2] << order[3]);
    ASSERT_EQ(ncclSuccess, bootstrapGetUniqueId(&handle_, nullptr));
    struct ncclSocket listen[4];
    for (int r : order) CheckIn(r, 4, &listen[r]);
    for (int r = 0; r < 4; r++) {
      struct ringConnectInfo got = Answer(&listen[r]);
      EXPECT_EQ(1 + (r + 1) % 4, got.fwd.handle[0]) << "rank " << r;
      EXPECT_EQ(101 + (r + 3) % 4, got.revHandle[0]) << "rank " << r;
    }
    WaitRoots();
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST_F(BootstrapMicrotest, Root_InconsistentRankInfoMakesTheRootGiveUp) {
  struct Case {
    const char* warn;
    int rank, nranks, nroots;
  } cases[] = {{"mismatch in info from procs, nranks 3 vs 4", 1, 4, 1},
               {"localId 5 is out of range", 5, 3, 1},
               {"rank 0 of 3 ranks has already checked in", 0, 3, 1}};
  for (const auto& c : cases) {
    SCOPED_TRACE(c.warn);
    ASSERT_EQ(ncclSuccess, bootstrapGetUniqueId(&handle_, nullptr));
    struct ncclSocket first, second;
    CheckIn(0, 3, &first);
    CheckIn(c.rank, c.nranks, &second, c.nroots);
    WaitRoots();
    EXPECT_TRUE(Warned(c.warn));
    (void)ncclSocketClose(&first);
    (void)ncclSocketClose(&second);
  }
}

TEST_F(BootstrapMicrotest, Root_RankGoneBeforeItsAnswer_MakesTheRootGiveUp) {
  ASSERT_EQ(ncclSuccess, bootstrapGetUniqueId(&handle_, nullptr));
  struct ncclSocket listen[2];
  CheckIn(0, 2, &listen[0]);
  (void)ncclSocketClose(&listen[0]);
  CheckIn(1, 2, &listen[1]);  // completes the ring: the root now answers both, starting with rank 0
  WaitRoots();
  EXPECT_EQ(0u, PendingOn(listen[1]));  // gave up rather than answering rank 1 anyway
  (void)ncclSocketClose(&listen[1]);
}

// Bug, pinned as observed (see Accept_RejectedPeerAsSocketCcReportsIt_IsReturnedUnconnected): one stray
// connection makes the root exit. A fix lets the root skip it, so the final sendToRoot then succeeds.
TEST_F(BootstrapMicrotest, Root_StrayPeerWithTheWrongMagic_MakesTheRootExit) {
  ASSERT_EQ(ncclSuccess, bootstrapGetUniqueId(&handle_, nullptr));
  struct ncclSocket stray;
  ASSERT_EQ(ncclSuccess, ncclSocketInit(&stray, &handle_.addr, handle_.magic + 1, ncclSocketTypeBootstrap));
  ASSERT_EQ(ncclSuccess, ncclSocketConnect(&stray));
  WaitRoots();
  struct extInfo info = {};
  EXPECT_EQ(ncclRemoteError, sendToRoot(&handle_, self_.comm.get(), &info));
  (void)ncclSocketClose(&stray);
}

TEST_F(BootstrapMicrotest, Init_FourRanks_AllGathersEveryRanksAddresses) {
  ASSERT_EQ(ncclSuccess, InitRanks(4));

  ASSERT_EQ(4u, g_proxyInits.size());
  const auto& addrs = g_proxyInits[Comm(0)].addrs;
  const auto& uds = g_proxyInits[Comm(0)].uds;
  EXPECT_EQ(4u, std::set<uint64_t>(uds.begin(), uds.end()).size());
  auto* p2p = static_cast<struct bootstrapState*>(State(0))->peerP2pAddresses;
  for (int r = 0; r < 4; r++) {
    EXPECT_EQ(0, std::memcmp(addrs.data(), g_proxyInits[Comm(r)].addrs.data(), 4 * sizeof(ncclSocketAddress)));
    EXPECT_EQ(uds, g_proxyInits[Comm(r)].uds);
    EXPECT_EQ(0, std::memcmp(&addrs[r], &g_proxyInits[Comm(r)].own, sizeof(ncclSocketAddress))) << r;
    auto* state = static_cast<struct bootstrapState*>(State(r));
    EXPECT_EQ(0, std::memcmp(&p2p[r], &state->listen.peerSocket.addr, sizeof(ncclSocketAddress))) << r;
  }
  ASSERT_EQ(4u, g_rasAdded.size());
  EXPECT_EQ(13, g_rasAdded[3].nvmlDev);
  EXPECT_TRUE(AllOk(CloseRanks()));
}

TEST_F(BootstrapMicrotest, Init_WithNeitherHandlesNorParent_FailsBeforeTouchingTheNetwork) {
  MakeRanks(1);
  EXPECT_EQ(ncclSystemError, bootstrapInit(0, nullptr, Comm(0), nullptr));
  EXPECT_TRUE(Warned("bootstrapInit: handles and parent are NULL"));
  EXPECT_EQ(0, OpenListeners());
}

TEST_F(BootstrapMicrotest, Init_SingleRankEachSocketFailureIsReturned) {
  struct Fault {
    const char* op;
    int nth;
  } faults[] = {{"listen", 1}, {"listen", 2}, {"listen", 3}, {"listen", 4}, {"connect", 1}, {"send", 1},
                {"accept", 1}, {"recv", 2},   {"connect", 2}, {"accept", 2}, {"ready", 1},   {"send", 2}};
  for (const auto& f : faults) {
    SCOPED_TRACE(testing::Message() << f.op << " #" << f.nth);
    ranks_.clear();
    MakeRanks(1);
    ASSERT_EQ(ncclSuccess, bootstrapGetUniqueId(&handle_, nullptr));
    std::map<std::string, int> calls;
    // The root thread's own socket calls are left alone.
    SetSockFault([&](const char* op, struct ncclSocket*) {
      if (std::this_thread::get_id() != main_) return ncclSuccess;
      return ++calls[op] == f.nth && std::strcmp(op, f.op) == 0 ? ncclSystemError : ncclSuccess;
    });
    EXPECT_NE(ncclSuccess, bootstrapInit(1, &handle_, Comm(0), nullptr));
    SetSockFault(nullptr);
    __atomic_store_n(&ranks_[0]->abort, 1u, __ATOMIC_RELEASE);
    EXPECT_EQ(ncclSuccess, bootstrapClose(State(0)));
    Comm(0)->bootstrap = nullptr;
    RestartSockets();
  }
}

TEST_F(BootstrapMicrotest, Init_StaggeredRanksStillMeetAtTheRoot) {
  g_params["UID_STAGGER_THRESHOLD"] = 1;
  g_params["UID_STAGGER_RATE"] = 1000000000;  // 1000 msgs/us: the computed delay rounds to 0
  ASSERT_EQ(ncclSuccess, InitRanks(3));
  ExpectAllGather(2);
}

TEST_F(BootstrapMicrotest, Init_RasOffOrFailingIsSkippedWithoutFailingInit) {
  g_rasEnable = 0;
  ASSERT_EQ(ncclSuccess, InitRanks(2));
  EXPECT_EQ(0, g_rasCommInits);
  EXPECT_TRUE(g_rasAdded.empty());
  EXPECT_TRUE(AllOk(CloseRanks()));

  ranks_.clear();
  g_rasEnable = 1;
  g_rasCommInitResult = ncclSystemError;
  ASSERT_EQ(ncclSuccess, InitRanks(2));
  EXPECT_EQ(2, g_rasCommInits);
  EXPECT_TRUE(g_rasAdded.empty());
  EXPECT_TRUE(AllOk(CloseRanks()));

  ranks_.clear();
  g_rasCommInitResult = ncclSuccess;
  g_rasAddRanksResult = ncclSystemError;
  ASSERT_EQ(ncclSuccess, InitRanks(2));
  EXPECT_TRUE(g_rasAdded.empty());
}

// Bug, pinned as observed: every failure past the proxy listen (here ncclProxyInit itself) frees proxySocket
// without closing it, and nothing else closes it since the proxy thread never started. A fix makes this 0.
TEST_F(BootstrapMicrotest, Init_ProxyInitFails_LeaksTheProxyListenSocket) {
  g_proxyInitResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, InitRanks(2));
  EXPECT_TRUE(AllOk(CloseRanks()));
  WaitRoots();
  EXPECT_EQ(2, OpenListeners());
}

TEST_F(BootstrapMicrotest, Init_AbortWhileARankIsMissing_UnblocksTheOthers) {
  MakeRanks(3);
  ASSERT_EQ(ncclSuccess, bootstrapGetUniqueId(&handle_, nullptr));
  watchdog_ = std::chrono::milliseconds(50);
  auto res = OnRanks([&](int r) { return r == 2 ? ncclSuccess : bootstrapInit(1, &handle_, Comm(r), nullptr); });
  EXPECT_EQ(ncclInternalError, res[0]);
  EXPECT_EQ(ncclInternalError, res[1]);
}

TEST_F(BootstrapMicrotest, Init_SeveralRootsShareTheRendezvousAndStillCloseOneRing) {
  for (int nRoots : {2, 3})
    for (int n : {nRoots, 5, 7}) {
      SCOPED_TRACE(testing::Message() << nRoots << " roots " << n << " ranks");
      ranks_.clear();
      MakeRanks(n);
      std::vector<ncclUniqueId> ids(nRoots);
      for (auto& id : ids) ASSERT_EQ(ncclSuccess, bootstrapGetUniqueId((ncclBootstrapHandle*)&id, nullptr));
      ASSERT_TRUE(AllOk(OnRanks([&](int r) { return bootstrapInit(nRoots, ids.data(), Comm(r), nullptr); })));
      ExpectAllGather(5);
      EXPECT_TRUE(AllOk(CloseRanks()));
      WaitRoots();
    }
}

TEST_F(BootstrapMicrotest, AllGather_BidirAndUnidirRingsAgreeForEveryRankCount) {
  for (int bidir = 0; bidir <= 1; bidir++) {
    g_params["BOOTSTRAP_BIDIR_ALLGATHER"] = bidir;
    for (int n = 1; n <= 6; n++) {
      SCOPED_TRACE(testing::Message() << "bidir " << bidir << " nranks " << n);
      ranks_.clear();
      ASSERT_EQ(ncclSuccess, InitRanks(n));
      g_multiOps = 0;
      ExpectAllGather(3);
      EXPECT_EQ(bidir && n >= 3, g_multiOps > 0);
      EXPECT_TRUE(AllOk(CloseRanks()));
    }
  }
}

TEST_F(BootstrapMicrotest, InitOverNet_UnidirRingAllGathersAndCloseReleasesEveryComm) {
  g_params["OOB_NET_ENABLE"] = 1;
  for (int n = 1; n <= 4; n++) {
    SCOPED_TRACE(n);
    ranks_.clear();
    g_netCloses = 0;
    ASSERT_EQ(ncclSuccess, InitRanks(n));
    g_netRegs = g_netDeregs = 0;
    ExpectAllGather(4);
    EXPECT_EQ(2 * n, g_netRegs.load());  // the buffer, once per ring comm
    EXPECT_EQ(g_netRegs.load(), g_netDeregs.load());
    EXPECT_TRUE(AllOk(CloseRanks()));
    EXPECT_EQ(3 * n, g_netCloses.load());  // send, recv, listen
    WaitRoots();
  }
}

TEST_F(BootstrapMicrotest, InitOverNet_BidirRingUsesThePiggybackedReverseHandles) {
  g_params["OOB_NET_ENABLE"] = 1;
  g_params["BOOTSTRAP_BIDIR_NET"] = 1;
  for (int n = 3; n <= 6; n++) {
    SCOPED_TRACE(n);
    ranks_.clear();
    g_netCloses = 0;
    ASSERT_EQ(ncclSuccess, InitRanks(n));
    for (int r = 0; r < n; r++) {
      auto* state = static_cast<struct bootstrapState*>(State(r));
      EXPECT_NE(nullptr, state->ring.net.revSendComm);
      EXPECT_NE(nullptr, state->ring.net.revRecvComm);
    }
    g_netRegs = g_netDeregs = 0;
    ExpectAllGather(3);
    EXPECT_EQ(4 * n, g_netRegs.load());  // forward and reverse pairs
    EXPECT_EQ(g_netRegs.load(), g_netDeregs.load());
    EXPECT_TRUE(AllOk(CloseRanks()));
    EXPECT_EQ(6 * n, g_netCloses.load());
    WaitRoots();
  }
}

TEST_F(BootstrapMicrotest, InitOverNet_SlowConnectsAndPostsAreRetriedUntilTheyLand) {
  g_params["OOB_NET_ENABLE"] = 1;
  g_params["BOOTSTRAP_BIDIR_NET"] = 1;
  g_netDeferConnect = 40;
  g_netDeferIsend = 40;
  g_netDeferIrecv = 40;
  ASSERT_EQ(ncclSuccess, InitRanks(4));
  ExpectAllGather(2);
  EXPECT_EQ(0, g_netDeferConnect.load());
}

TEST_F(BootstrapMicrotest, InitOverNet_ListenFailuresAreReturned) {
  g_params["OOB_NET_ENABLE"] = 1;
  g_params["BOOTSTRAP_BIDIR_NET"] = 1;
  for (int nth : {1, 2}) {
    SCOPED_TRACE(nth);
    ranks_.clear();
    MakeRanks(3);
    std::atomic<int> listens(0);
    g_netFault = [&](const char* op) {
      return std::strcmp(op, "listen") == 0 && ++listens == nth ? ncclSystemError : ncclSuccess;
    };
    ASSERT_EQ(ncclSuccess, bootstrapGetUniqueId(&handle_, nullptr));
    EXPECT_EQ(ncclSystemError, bootstrapInit(1, &handle_, Comm(0), nullptr));
    g_netFault = nullptr;
    EXPECT_EQ(ncclSuccess, bootstrapClose(State(0)));
    Comm(0)->bootstrap = nullptr;
    RestartSockets();
  }
}

// The reverse handles cannot cross roots, so several roots fall back to the forward net ring.
TEST_F(BootstrapMicrotest, InitOverNet_SeveralRootsKeepTheUnidirectionalRing) {
  g_params["OOB_NET_ENABLE"] = 1;
  g_params["BOOTSTRAP_BIDIR_NET"] = 1;
  MakeRanks(5);
  ncclUniqueId ids[2];
  for (auto& id : ids) ASSERT_EQ(ncclSuccess, bootstrapGetUniqueId((ncclBootstrapHandle*)&id, nullptr));
  ASSERT_TRUE(AllOk(OnRanks([&](int r) { return bootstrapInit(2, ids, Comm(r), nullptr); })));
  for (int r = 0; r < 5; r++) EXPECT_EQ(nullptr, static_cast<struct bootstrapState*>(State(r))->ring.net.revSendComm);
  ExpectAllGather(2);
  EXPECT_TRUE(AllOk(CloseRanks()));
}

TEST_F(BootstrapMicrotest, BidirRingSetup_HandleExchangeFailureDeregistersAndIsReturned) {
  g_params["OOB_NET_ENABLE"] = 1;
  g_params["BOOTSTRAP_BIDIR_NET"] = 1;
  NetLoop loop;
  struct bootstrapState state = {};
  state.nranks = 3;
  state.net = &g_fakeNet;
  state.ring.net.sendComm = loop.send;
  state.ring.net.recvComm = loop.recv;
  state.abortFlag = &abort_;
  for (int nth : {1, 2}) {
    SCOPED_TRACE(nth);
    std::atomic<int> regs(0);
    g_netFault = [&](const char* o) {
      return std::strcmp(o, "regMr") == 0 && ++regs == nth ? ncclSystemError : ncclSuccess;
    };
    g_netRegs = g_netDeregs = 0;
    EXPECT_EQ(ncclSystemError, bootstrapBidirRingSetup(self_.comm.get(), &state, nullptr));
    EXPECT_EQ(g_netRegs.load(), g_netDeregs.load());
  }
  EXPECT_NE(nullptr, state.listen.net.revComm);  // the in-place listen, done once
}

TEST_F(BootstrapMicrotest, BidirRingSetup_PiggybackedHandleSkipsTheExchange) {
  g_params["OOB_NET_ENABLE"] = 1;
  g_params["BOOTSTRAP_BIDIR_NET"] = 1;
  struct bootstrapState state = {};
  state.nranks = 3;
  state.net = &g_fakeNet;
  state.abortFlag = &abort_;
  ASSERT_EQ(ncclSuccess, g_fakeNet.listen(nullptr, 0, state.listen.net.revHandle, &state.listen.net.revComm));
  char prevRevHandle[NCCL_NET_HANDLE_MAXSIZE];
  std::memcpy(prevRevHandle, state.listen.net.revHandle, sizeof(prevRevHandle));  // loops back to itself

  EXPECT_EQ(ncclSuccess, bootstrapBidirRingSetup(self_.comm.get(), &state, prevRevHandle));

  EXPECT_NE(nullptr, state.ring.net.revSendComm);
  EXPECT_NE(nullptr, state.ring.net.revRecvComm);
  EXPECT_EQ(0, g_netRegs.load());
  EXPECT_EQ(1u, g_netListens.size());
}

TEST_F(BootstrapMicrotest, BidirRingSetupStart_NeedsAPiggybackedHandleToStartConnecting) {
  g_params["OOB_NET_ENABLE"] = 1;
  g_params["BOOTSTRAP_BIDIR_NET"] = 1;
  struct bootstrapState state = {};
  state.nranks = 3;
  state.net = &g_fakeNet;
  char out[NCCL_NET_HANDLE_MAXSIZE], zeros[NCCL_NET_HANDLE_MAXSIZE] = {};
  bool started = true;
  EXPECT_EQ(ncclSuccess, bootstrapBidirRingSetupStart(self_.comm.get(), &state, nullptr, out, &started));
  EXPECT_FALSE(started);
  EXPECT_NE(nullptr, state.listen.net.revComm);
  EXPECT_EQ(ncclSuccess, bootstrapBidirRingSetupStart(self_.comm.get(), &state, zeros, out, &started));
  EXPECT_FALSE(started);
  EXPECT_EQ(1u, g_netListens.size());
  state.nranks = 2;
  EXPECT_EQ(ncclSuccess, bootstrapBidirRingSetupStart(self_.comm.get(), &state, zeros, out, &started));
  EXPECT_EQ(ncclSuccess, bootstrapBidirRingSetupFinish(self_.comm.get(), &state, zeros));
  EXPECT_EQ(ncclSuccess, bootstrapBidirRingSetup(self_.comm.get(), &state, zeros));
}

// devOOB is a function-local static, so each NCCL_OOB_NET_IFNAME case runs in a fresh process.
class BootstrapOobDeviceMicrotest : public BootstrapMicrotest {
 protected:
  void ExpectDevice(const char* ifname, int expectDev, ncclResult_t expectRes = ncclSuccess,
                    const char* warn = nullptr) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    EXPECT_EXIT(
        {
          SetMicroEnv("NCCL_OOB_NET_IFNAME", ifname);
          self_.comm->ncclNet = &g_fakeNet;
          int dev = -1;
          bool ok = netGetDevice(0, self_.comm.get(), &dev) == expectRes && (warn == nullptr || Warned(warn));
          if (ok && expectRes == ncclSuccess) {
            SetMicroEnv("NCCL_OOB_NET_IFNAME", "nothing-matches");  // the first answer is cached
            int again = -1;
            ok = dev == expectDev && netGetDevice(0, self_.comm.get(), &again) == ncclSuccess && again == expectDev;
          }
          std::exit(ok ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "");
  }
  void SetUp() override {
    BootstrapMicrotest::SetUp();
    g_netDevs = {"mlx5_10", "mlx5_1", "ib0"};
  }
};

TEST_F(BootstrapOobDeviceMicrotest, Unset_PicksDeviceZero) { ExpectDevice("", 0); }
TEST_F(BootstrapOobDeviceMicrotest, Prefix_PicksTheFirstDeviceItPrefixes) { ExpectDevice("mlx5_1", 0); }
TEST_F(BootstrapOobDeviceMicrotest, Exact_SkipsLongerNames) { ExpectDevice("=mlx5_1", 1); }
TEST_F(BootstrapOobDeviceMicrotest, Excluded_PicksTheFirstOtherDevice) { ExpectDevice("^mlx5_10", 1); }
TEST_F(BootstrapOobDeviceMicrotest, ExcludedExact_PicksTheFirstOtherDevice) { ExpectDevice("^=mlx5_10", 1); }
TEST_F(BootstrapOobDeviceMicrotest, NoMatch_IsAnInvalidArgument) {
  ExpectDevice("=mlx5", -1, ncclInvalidArgument, "no device found matching exactly mlx5");
}
TEST_F(BootstrapOobDeviceMicrotest, EverythingExcluded_IsAnInvalidArgument) {
  ExpectDevice("^mlx5_1,ib", -1, ncclInvalidArgument, "no device found after excluding mlx5_1,ib");
}
TEST_F(BootstrapOobDeviceMicrotest, DeviceQueryFailure_IsReturned) {
  g_netFault = [](const char* op) { return std::strcmp(op, "devices") == 0 ? ncclSystemError : ncclSuccess; };
  ExpectDevice("ib", -1, ncclSystemError);
}
TEST_F(BootstrapOobDeviceMicrotest, PropertiesFailureOnlyAffectsTheLog) {
  g_netFault = [](const char* op) { return std::strcmp(op, "getProperties") == 0 ? ncclSystemError : ncclSuccess; };
  ExpectDevice("", 0);
}

TEST_F(BootstrapMicrotest, Split_OneChildPerColorEachWithItsOwnRing) {
  ASSERT_EQ(ncclSuccess, InitRanks(5));
  ASSERT_TRUE(AllOk(Split({{4, 1, 3}, {0, 2}}, /*shareResources=*/false)));
  ExpectAllGather(3);
  EXPECT_EQ(5u, g_proxyInits.size() - 5);  // the parents' five, then one per child
  EXPECT_TRUE(AllOk(CloseRanks()));
}

TEST_F(BootstrapMicrotest, Split_SharedResourcesMapsTopParentRanksAndSkipsTheProxy) {
  ASSERT_EQ(ncclSuccess, InitRanks(4));
  size_t proxies = g_proxyInits.size();
  ASSERT_TRUE(AllOk(Split({{3, 1, 0, 2}}, /*shareResources=*/true)));
  ExpectAllGather(2);
  EXPECT_EQ(proxies, g_proxyInits.size());
  EXPECT_EQ(std::vector<int>({103, 101, 100, 102}), ranks_[0]->topParentRanks);
  EXPECT_TRUE(AllOk(CloseRanks()));
}

TEST_F(BootstrapMicrotest, Split_NetModeExchangesReverseHandlesOverTheForwardRing) {
  g_params["OOB_NET_ENABLE"] = 1;
  g_params["BOOTSTRAP_BIDIR_NET"] = 1;
  ASSERT_EQ(ncclSuccess, InitRanks(4));
  g_netDeferConnect = 20;
  ASSERT_TRUE(AllOk(Split({{1, 2, 3, 0}}, false)));
  for (int r = 0; r < 4; r++) EXPECT_NE(nullptr, static_cast<struct bootstrapState*>(State(r))->ring.net.revSendComm);
  ExpectAllGather(4);
  EXPECT_TRUE(AllOk(CloseRanks()));
}

TEST_F(BootstrapMicrotest, Split_RasInitFailureIsOnlyLogged) {
  ASSERT_EQ(ncclSuccess, InitRanks(2));
  g_rasCommInitResult = ncclSystemError;
  EXPECT_TRUE(AllOk(Split({{1, 0}}, false)));
  ExpectAllGather(1);
}

// Bug, pinned as observed: the proxySocket leak of Init_ProxyInitFails_LeaksTheProxyListenSocket, on the split
// path. A fix makes this 0.
TEST_F(BootstrapMicrotest, Split_ProxyInitFails_LeaksTheProxyListenSocket) {
  ASSERT_EQ(ncclSuccess, InitRanks(2));
  g_proxyInitResult = ncclSystemError;
  auto res = Split({{1, 0}}, false);
  EXPECT_EQ(ncclSystemError, res[0]);
  EXPECT_EQ(ncclSystemError, res[1]);
  EXPECT_TRUE(AllOk(CloseRanks()));
  for (auto& p : parents_) {
    EXPECT_EQ(ncclSuccess, bootstrapClose(p->comm->bootstrap));
    p->comm->bootstrap = nullptr;
  }
  WaitRoots();
  EXPECT_EQ(2, OpenListeners());
}

TEST_F(BootstrapMicrotest, Split_EachSetupFailureIsReturned) {
  struct Fault {
    bool net, plugin;
    const char* op;
    int nth;
  } faults[] = {{false, false, "listen", 1}, {false, false, "listen", 2},  {false, false, "listen", 3},
                {false, false, "connect", 1}, {false, false, "connect", 2}, {false, false, "accept", 1},
                {true, false, "listen", 2},  {true, true, "listen", 1},    {true, true, "connect", 1}};
  for (const auto& f : faults) {
    SCOPED_TRACE(testing::Message() << f.net << f.plugin << " " << f.op << " #" << f.nth);
    g_params["OOB_NET_ENABLE"] = f.net;
    for (auto& r : ranks_) parents_.push_back(std::move(r));
    ranks_.clear();
    ASSERT_EQ(ncclSuccess, InitRanks(1));
    std::map<std::string, int> calls;
    auto fail = [&](const char* op) { return ++calls[op] == f.nth && std::strcmp(op, f.op) == 0; };
    if (f.plugin)
      g_netFault = [&](const char* op) { return fail(op) ? ncclSystemError : ncclSuccess; };
    else
      SetSockFault([&](const char* op, struct ncclSocket*) { return fail(op) ? ncclSystemError : ncclSuccess; });
    EXPECT_EQ(ncclSystemError, Split({{0}}, false)[0]);
    SetSockFault(nullptr);
    g_netFault = nullptr;
  }
}

TEST_F(BootstrapMicrotest, BcastGrowHandle_CoordinatorReachesOnlyTheBoundaryRanks) {
  ncclBootstrapHandle h = {};
  EXPECT_EQ(ncclInvalidArgument, bcastGrowHandle(nullptr, self_.comm.get(), true));
  EXPECT_EQ(ncclInvalidArgument, bcastGrowHandle(&h, nullptr, false));
  EXPECT_TRUE(Warned("parent comm and handle must be provided"));
  self_.comm->nRanks = 1;
  EXPECT_EQ(ncclSuccess, bcastGrowHandle(&h, self_.comm.get(), false));  // would block if it received

  ASSERT_EQ(ncclSuccess, InitRanks(4));
  ncclBootstrapHandle got[4] = {};
  got[1].magic = 0xAB;
  got[1].nRanks = 4;
  EXPECT_TRUE(AllOk(OnRanks([&](int r) {
    if (r == 2) return ncclSuccess;
    return bcastGrowHandle(&got[r], Comm(r), /*isRoot=*/r == 1);
  })));
  EXPECT_EQ(0xABu, got[0].magic);
  EXPECT_EQ(0xABu, got[3].magic);
  EXPECT_EQ(4, got[3].nRanks);
  EXPECT_TRUE(AllOk(CloseRanks()));
}

TEST_F(BootstrapMicrotest, Grow_ExistingAndNewRanksCloseOneRing) {
  ASSERT_EQ(ncclSuccess, InitRanks(3));
  parents_ = std::move(ranks_);
  ranks_.clear();
  ncclComm* coordinator = parents_[1]->comm.get();
  ncclBootstrapHandle grow;
  ASSERT_EQ(ncclSuccess, bootstrapGetUniqueId(&grow, coordinator));
  ncclBootstrapHandle boundary[3] = {{}, grow, {}};
  std::swap(ranks_, parents_);  // OnRanks runs over ranks_
  ASSERT_TRUE(AllOk(OnRanks([&](int r) { return bcastGrowHandle(&boundary[r], Comm(r), /*isRoot=*/r == 1); })));
  std::swap(ranks_, parents_);
  for (auto& p : parents_) p->comm->childCount++;  // ncclCommGrow bumps it before init

  MakeRanks(5);
  auto res = OnRanks([&](int r) {
    Comm(r)->isGrow = true;
    if (r >= 3) return bootstrapInit(1, &grow, Comm(r), nullptr);
    bool boundaryRank = r == 0 || r == 2;
    return bootstrapInit(boundaryRank ? 1 : 0, boundaryRank ? &boundary[r] : nullptr, Comm(r), parents_[r]->comm.get());
  });
  ASSERT_TRUE(AllOk(res));
  EXPECT_EQ(Comm(0)->magic, Comm(4)->magic);
  EXPECT_EQ(Comm(1)->magic, grow.magic);
  ExpectAllGather(3);
  EXPECT_TRUE(AllOk(CloseRanks()));
}

// Bug, pinned as observed: under NCCL_OOB_NET_ENABLE=1 + NCCL_BOOTSTRAP_BIDIR_NET=1 a grow never finishes.
// Existing ranks take nextPeer from rank+1 over the parent bootstrap, so its revHandle is the next rank's reverse
// listener where the previous rank's is expected, and existing ranks that got no handle (nHandles == 0) skip the
// reverse ring altogether. A fix makes the bidir runs succeed like the bidir=0 ones.
TEST_F(BootstrapMicrotest, Grow_NetBidir_ReverseRingNeverConnectsSoInitHangsUntilAbort) {
  g_params["OOB_NET_ENABLE"] = 1;
  for (int bidir : {0, 1})
    for (int existing : {2, 3}) {
      SCOPED_TRACE(testing::Message() << "bidir " << bidir << " existing " << existing);
      g_params["BOOTSTRAP_BIDIR_NET"] = bidir;
      for (auto& r : ranks_) parents_.push_back(std::move(r));
      ranks_.clear();
      ASSERT_EQ(ncclSuccess, InitRanks(existing));
      std::vector<Rank*> parent;
      for (auto& r : ranks_) parent.push_back(r.get());
      for (auto& r : ranks_) parents_.push_back(std::move(r));
      ranks_.clear();
      ncclBootstrapHandle grow;
      ASSERT_EQ(ncclSuccess, bootstrapGetUniqueId(&grow, parent[0]->comm.get()));
      for (auto* p : parent) p->comm->childCount++;
      MakeRanks(existing + 2);
      watchdog_ = bidir ? std::chrono::milliseconds(300) : g_waitLimit;
      auto res = OnRanks([&](int r) {
        Comm(r)->isGrow = true;
        if (r >= existing) return bootstrapInit(1, &grow, Comm(r), nullptr);
        bool boundary = r == 0 || r == existing - 1;  // only these receive the handle
        return bootstrapInit(boundary ? 1 : 0, boundary ? &grow : nullptr, Comm(r), parent[r]->comm.get());
      });
      for (int r = 0; r < existing + 2; r++) EXPECT_EQ(bidir ? ncclInternalError : ncclSuccess, res[r]) << r;
      WaitRoots();
    }
}

// Plain sends complete without the receiver, so one thread can fix the order connections arrive in.
TEST_F(BootstrapMicrotest, SendRecv_OutOfOrderConnectionsAreParkedUntilAsked) {
  ASSERT_EQ(ncclSuccess, InitRanks(3));
  char got[3] = {};
  for (const char* tag : {"1", "2", "3"}) ASSERT_EQ(ncclSuccess, bootstrapSend(State(0), 1, tag[0], (void*)tag, 2));
  EXPECT_EQ(ncclSuccess, bootstrapRecv(State(1), 0, '3', got, 2));  // parks 1 and 2
  EXPECT_STREQ("3", got);
  EXPECT_EQ(ncclSuccess, bootstrapRecv(State(1), 0, '2', got, 2));  // from the middle of the parked list
  EXPECT_STREQ("2", got);
  EXPECT_EQ(ncclSuccess, bootstrapRecv(State(1), 0, '1', got, 2));
  EXPECT_STREQ("1", got);

  ASSERT_EQ(ncclSuccess, bootstrapSend(State(2), 1, 7, (void*)"2", 2));
  ASSERT_EQ(ncclSuccess, bootstrapSend(State(0), 1, 7, (void*)"0", 2));
  EXPECT_EQ(ncclSuccess, bootstrapRecv(State(1), 0, 7, got, 2));  // same tag, other peer first: parked
  EXPECT_STREQ("0", got);
  EXPECT_EQ(ncclSuccess, bootstrapRecv(State(1), 2, 7, got, 2));
  EXPECT_STREQ("2", got);

  ASSERT_EQ(ncclSuccess, bootstrapSend(State(0), 1, 8, (void*)"a", 2));
  ASSERT_EQ(ncclSuccess, bootstrapSend(State(2), 1, 9, (void*)"b", 2));
  EXPECT_EQ(ncclSuccess, bootstrapRecv(State(1), -1, 9, got, 2));  // any peer
  EXPECT_STREQ("b", got);
  EXPECT_EQ(ncclSuccess, bootstrapRecv(State(1), -1, 8, got, 2));  // any peer, from the parked list
  EXPECT_STREQ("a", got);
  EXPECT_TRUE(AllOk(CloseRanks()));
}

TEST_F(BootstrapMicrotest, SendRecv_TransferFailuresAreReturned) {
  ASSERT_EQ(ncclSuccess, InitRanks(1));
  ASSERT_EQ(ncclSuccess, bootstrapSend(State(0), 0, 2, (void*)"long", 4));
  char c[2];
  EXPECT_EQ(ncclInternalError, bootstrapRecv(State(0), 0, 2, c, 2));
  EXPECT_TRUE(Warned("Message truncated : received 4 bytes instead of 2"));
  for (int nth : {1, 3}) {  // the (rank, tag) ack, then the payload
    int sends = 0;
    SetSockFault([&](const char* op, struct ncclSocket*) {
      return std::strcmp(op, "send") == 0 && ++sends == nth ? ncclSystemError : ncclSuccess;
    });
    EXPECT_EQ(ncclSystemError, bootstrapSend(State(0), 0, 1, (void*)"x", 1)) << nth;
  }
}

// Bug, pinned as observed: the failing close frees the parked connections but leaves the list head dangling and
// the state alive, so a second bootstrapClose (e.g. an abort after a failed destroy) frees the same list again.
// A fix that clears the head flips the last EXPECT.
TEST_F(BootstrapMicrotest, Close_WithAParkedConnection_FailsAndLeavesTheFreedListLinked) {
  ASSERT_EQ(ncclSuccess, InitRanks(2));
  auto res = OnRanks([&](int r) -> ncclResult_t {
    if (r == 0) {
      NCCLCHECK(bootstrapSend(State(0), 1, 5, (void*)"x", 1));
      return bootstrapSend(State(0), 1, 6, (void*)"y", 1);
    }
    char c;
    return bootstrapRecv(State(1), 0, 6, &c, 1);
  });
  ASSERT_TRUE(AllOk(res));
  auto* state = static_cast<struct bootstrapState*>(State(1));
  ASSERT_NE(nullptr, state->unexpectedConnections);

  EXPECT_EQ(ncclInternalError, bootstrapClose(state));

  EXPECT_TRUE(Warned("Unexpected connections are not empty"));
  EXPECT_NE(nullptr, state->unexpectedConnections);  // already freed
  state->unexpectedConnections = nullptr;
}

TEST_F(BootstrapMicrotest, Close_WhileAbortingDropsParkedConnectionsQuietly) {
  ASSERT_EQ(ncclSuccess, InitRanks(2));
  ASSERT_TRUE(AllOk(OnRanks([&](int r) -> ncclResult_t {
    char c;
    if (r == 0) {
      NCCLCHECK(bootstrapSend(State(0), 1, 5, (void*)"x", 1));
      return bootstrapSend(State(0), 1, 6, (void*)"y", 1);
    }
    return bootstrapRecv(State(1), 0, 6, &c, 1);
  })));
  ranks_[1]->abort = 1;
  EXPECT_TRUE(AllOk(CloseRanks()));
  EXPECT_FALSE(Warned("Unexpected connections"));
}

TEST_F(BootstrapMicrotest, Barrier_EveryRankLeavesAfterAllArrive) {
  ASSERT_EQ(ncclSuccess, InitRanks(5));
  // The last rank arrives late; nobody may leave before it does.
  std::atomic<bool> lateArrived(false);
  auto late = [&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    lateArrived = true;
  };
  EXPECT_TRUE(AllOk(OnRanks([&](int r) {
    if (r == 4) late();
    ncclResult_t res = bootstrapBarrier(State(r), r, 5, 0x77);
    EXPECT_TRUE(lateArrived) << "rank " << r << " left early";
    return res;
  })));
  // Intra-node barrier over ranks 1 and 3 only, in local-rank order.
  int local[2] = {1, 3};
  lateArrived = false;
  EXPECT_TRUE(AllOk(OnRanks([&](int r) {
    if (r != 1 && r != 3) return ncclSuccess;
    if (r == 3) late();
    ncclResult_t res = bootstrapIntraNodeBarrier(State(r), local, r == 1 ? 0 : 1, 2, 0x78);
    EXPECT_TRUE(lateArrived) << "rank " << r << " left early";
    return res;
  })));
  EXPECT_EQ(ncclSuccess, bootstrapBarrier(State(0), 0, 1, 0x79));
  EXPECT_TRUE(AllOk(CloseRanks()));
}

TEST_F(BootstrapMicrotest, Broadcast_RootPayloadReachesEveryRank) {
  ASSERT_EQ(ncclSuccess, InitRanks(4));
  char buf[4][4] = {};
  std::strcpy(buf[2], "abc");
  std::atomic<int> connects(0);
  SetSockFault([&](const char* op, struct ncclSocket*) {
    if (std::strcmp(op, "connect") == 0) connects++;
    return ncclSuccess;
  });
  EXPECT_TRUE(AllOk(OnRanks([&](int r) { return bootstrapBroadcast(State(r), r, 4, 2, buf[r], 4); })));
  SetSockFault(nullptr);
  for (int r = 0; r < 4; r++) EXPECT_STREQ("abc", buf[r]) << r;
  EXPECT_EQ(3, connects.load());  // not to itself
  // Intra-node broadcast over ranks {0, 2, 3}, rooted at local rank 1 (= rank 2).
  int local[3] = {0, 2, 3};
  char nbuf[4][4] = {};
  std::strcpy(nbuf[2], "xyz");
  EXPECT_TRUE(AllOk(OnRanks([&](int r) {
    if (r == 1) return ncclSuccess;
    return bootstrapIntraNodeBroadcast(State(r), local, r == 0 ? 0 : r - 1, 3, 1, nbuf[r], 4);
  })));
  EXPECT_STREQ("xyz", nbuf[0]);
  EXPECT_STREQ("xyz", nbuf[3]);
  EXPECT_STREQ("", nbuf[1]);
  EXPECT_EQ(ncclSuccess, bootstrapBroadcast(State(0), 0, 1, 0, buf[0], 4));
  EXPECT_TRUE(AllOk(CloseRanks()));
}

TEST_F(BootstrapMicrotest, IntraNodeAllGather_GathersOverALocalSubsetInBothRingModes) {
  ASSERT_EQ(ncclSuccess, InitRanks(4));
  int local[3] = {3, 0, 2};  // local rank i is rank local[i]
  for (int bidir = 0; bidir <= 1; bidir++) {
    SCOPED_TRACE(bidir);
    g_params["BOOTSTRAP_BIDIR_ALLGATHER"] = bidir;
    char data[4][3] = {};
    for (int i = 0; i < 3; i++) data[local[i]][i] = char('a' + i);
    EXPECT_TRUE(AllOk(OnRanks([&](int r) {
      int i = std::find(local, local + 3, r) - local;
      return i == 3 ? ncclSuccess : bootstrapIntraNodeAllGather(State(r), local, i, 3, data[r], 1);
    })));
    for (int r : local) EXPECT_EQ("abc", std::string(data[r], 3)) << r;
  }
  char one = 'q';
  EXPECT_EQ(ncclSuccess, bootstrapIntraNodeAllGather(State(0), local, 0, 1, &one, 1));
  EXPECT_TRUE(AllOk(CloseRanks()));
}

TEST_F(BootstrapMicrotest, IntraNodeAllGather_EncryptedConnectsOnAHelperThread) {
  ASSERT_EQ(ncclSuccess, InitRanks(3));
  g_encrypted = true;
  int local[3] = {0, 1, 2};
  char data[3][3] = {};
  for (int r = 0; r < 3; r++) data[r][r] = char('a' + r);
  // Thread ids get reused, so mark the rank threads themselves.
  static thread_local bool onRankThread = false;
  std::atomic<int> connects(0), connectsOnRankThreads(0);
  SetSockFault([&](const char* op, struct ncclSocket*) {
    if (std::strcmp(op, "connect") == 0) {
      connects++;
      if (onRankThread) connectsOnRankThreads++;
    }
    return ncclSuccess;
  });
  EXPECT_TRUE(AllOk(OnRanks([&](int r) {
    onRankThread = true;
    return bootstrapIntraNodeAllGather(State(r), local, r, 3, data[r], 1);
  })));
  SetSockFault(nullptr);
  for (int r = 0; r < 3; r++) EXPECT_EQ("abc", std::string(data[r], 3)) << r;
  EXPECT_EQ(3, connects.load());
  EXPECT_EQ(0, connectsOnRankThreads.load());
  EXPECT_TRUE(AllOk(CloseRanks()));
}

TEST_F(BootstrapMicrotest, IntraNodeAllGather_ConnectFailureIsReturnedPlainOrEncrypted) {
  ASSERT_EQ(ncclSuccess, InitRanks(2));
  int local[2] = {0, 1};
  char data[2][2];
  SetSockFault([](const char* op, struct ncclSocket*) {
    return std::strcmp(op, "connect") == 0 ? ncclSystemError : ncclSuccess;
  });
  for (bool encrypted : {false, true}) {
    SCOPED_TRACE(encrypted);
    g_encrypted = encrypted;
    watchdog_ = std::chrono::milliseconds(50);  // encrypted, the accept side waits for a connect that never comes
    auto res = OnRanks([&](int r) { return bootstrapIntraNodeAllGather(State(r), local, r, 2, data[r], 1); });
    EXPECT_EQ(ncclSystemError, res[0]);
    EXPECT_EQ(ncclSystemError, res[1]);
    for (auto& rank : ranks_) rank->abort = 0;
  }
}

TEST_F(BootstrapMicrotest, Send_EncryptedSendsToOnePeerAndTagArriveInCallOrder) {
  ASSERT_EQ(ncclSuccess, InitRanks(2));
  g_encrypted = true;
  char got[3][2] = {};
  const char* sent[3] = {"1", "2", "3"};
  EXPECT_TRUE(AllOk(OnRanks([&](int r) -> ncclResult_t {
    for (int i = 0; i < 3; i++)
      NCCLCHECK(r == 0 ? bootstrapSend(State(0), 1, 9, (void*)sent[i], 2) : bootstrapRecv(State(1), 0, 9, got[i], 2));
    return r == 0 ? bootstrapSend(State(0), 1, 10, nullptr, 0) : bootstrapRecv(State(1), 0, 10, nullptr, 0);
  })));
  for (int i = 0; i < 3; i++) EXPECT_STREQ(sent[i], got[i]);
  EXPECT_TRUE(AllOk(CloseRanks()));
}

TEST_F(BootstrapMicrotest, Send_EncryptedOnceCloseBegan_IsRejected) {
  ASSERT_EQ(ncclSuccess, InitRanks(1));
  g_encrypted = true;
  ASSERT_EQ(ncclSuccess, bootstrapAsyncSendDrain(static_cast<struct bootstrapState*>(State(0))));
  EXPECT_EQ(ncclInternalError, bootstrapSend(State(0), 0, 1, (void*)"x", 1));
  EXPECT_TRUE(Warned("bootstrapSend: bootstrap state is closing"));
}

class BootstrapAsyncSendMicrotest : public BootstrapMicrotest {
 protected:
  std::mutex gateMu_;
  std::condition_variable gateCv_;
  bool open_ = false;
  std::atomic<int> connects_{0};
  struct bootstrapState* state_ = nullptr;

  // Rank 0 queues two encrypted sends to (1, 4); the first one's connect is held, then fails.
  void FailFirstOfTwoSends() {
    ASSERT_EQ(ncclSuccess, InitRanks(2));
    g_encrypted = true;
    state_ = static_cast<struct bootstrapState*>(State(0));
    SetSockFault([this](const char* op, struct ncclSocket*) {
      if (std::strcmp(op, "connect") != 0 || ++connects_ != 1) return ncclSuccess;
      std::unique_lock<std::mutex> lock(gateMu_);
      EXPECT_TRUE(gateCv_.wait_for(lock, g_waitLimit, [this] { return open_; }));
      return ncclRemoteError;
    });
    ASSERT_EQ(ncclSuccess, bootstrapSend(state_, 1, 4, (void*)"a", 1));
    ASSERT_EQ(ncclSuccess, bootstrapSend(state_, 1, 4, (void*)"b", 1));
    {
      std::lock_guard<std::mutex> lock(gateMu_);
      open_ = true;
    }
    gateCv_.notify_all();
    std::unique_lock<std::mutex> lock(state_->asyncSendLock);
    ASSERT_TRUE(state_->asyncSendCond.wait_for(lock, g_waitLimit,
                                               [this] { return ncclIntruQueueEmpty(&state_->asyncSendQueue); }));
  }
};

TEST_F(BootstrapAsyncSendMicrotest, FailureAbortsDropsLaterSendsAndSurfacesOnTheNextCalls) {
  FailFirstOfTwoSends();

  EXPECT_EQ(1, connects_.load());  // "b" was dropped, not sent after the failed "a"
  EXPECT_EQ(1u, __atomic_load_n(&ranks_[0]->abort, __ATOMIC_ACQUIRE));
  char c;
  EXPECT_EQ(ncclRemoteError, bootstrapRecv(state_, 1, 1, &c, 1));
  EXPECT_TRUE(Warned("bootstrapRecv: an earlier asynchronous bootstrap send failed"));
  EXPECT_EQ(ncclRemoteError, bootstrapSend(state_, 1, 1, &c, 1));
  EXPECT_TRUE(Warned("bootstrapSend: an earlier asynchronous send to peer 1 failed"));
  EXPECT_EQ(ncclSuccess, bootstrapClose(state_));  // already reported, and aborting
  Comm(0)->bootstrap = nullptr;
}

TEST_F(BootstrapAsyncSendMicrotest, FailureNotYetReportedIsReturnedByCloseOnce) {
  FailFirstOfTwoSends();

  EXPECT_EQ(ncclRemoteError, bootstrapClose(state_));
  EXPECT_TRUE(Warned("bootstrapClose: an asynchronous bootstrap send failed"));
  EXPECT_EQ(ncclSuccess, bootstrapClose(state_));
  Comm(0)->bootstrap = nullptr;
}
}  // namespace
