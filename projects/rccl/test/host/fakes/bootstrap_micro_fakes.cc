/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See bootstrap_micro_fakes.h.

#include "bootstrap_micro_fakes.h"

#include <arpa/inet.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "env_fakes.h"
#include "nccl_fakes.h"
#include "proxy.h"

using namespace bootstrap_micro;

int64_t ncclParamRasEnable() { return g_loadParam("RAS_ENABLE", 1); }

// src/misc/signals.cc, which has no shared owner: the real one would install process-wide handlers.
void RegisterSignalHandlers() {}

bool g_encrypted = false;
ncclResult_t ncclGetCryptConnectionMode(bool* encrypted) {
  *encrypted = g_encrypted;
  return ncclSuccess;
}

namespace bootstrap_micro {
std::mutex g_seamMu;
std::map<ncclComm*, ProxyInitCall> g_proxyInits;
ncclResult_t g_proxyInitResult = ncclSuccess;
int g_rasCommInits = 0;
ncclResult_t g_rasCommInitResult = ncclSuccess;
std::vector<rasRankInit> g_rasAdded;
ncclResult_t g_rasAddRanksResult = ncclSuccess;
int g_ifCount = 1;
union ncclSocketAddress g_subnetRemote;
}  // namespace bootstrap_micro

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

namespace bootstrap_micro {


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
ncclSocketState g_rejectedState = ncclSocketStateAccepting;
int g_readyPolls = 0;
std::atomic<int> g_multiOps{0};
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
}  // namespace bootstrap_micro

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

namespace bootstrap_micro {
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

void ShutdownSockets() {
  std::unique_lock<std::mutex> lock(g_sockMu);
  g_sockShutdown = true;
  g_sockCv.notify_all();
  auto deadline = Clock::now() + g_waitLimit;
  while (g_liveRoots > 0 && Clock::now() < deadline) g_sockCv.wait_for(lock, std::chrono::milliseconds(1));
  EXPECT_EQ(0, g_liveRoots) << "a bootstrap root thread did not finish";
}
}  // namespace bootstrap_micro
