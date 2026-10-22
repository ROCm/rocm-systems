/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See bootstrap_micro_fakes.h.

#include "bootstrap_micro_fakes.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

#include "env_fakes.h"
#include "nccl_fakes.h"
#include "proxy.h"

using namespace bootstrap_micro;
using namespace in_memory_sockets;

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
int g_liveRoots = 0;
}  // namespace bootstrap_micro

// The proxy owns both address arrays from the call on, and the listen socket on success, as in production.
ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses,
                           uint64_t* peerUDS) {
  {
    std::lock_guard<std::mutex> lock(g_seamMu);
    g_proxyInits[comm] = {std::vector<ncclSocketAddress>(peerAddresses, peerAddresses + comm->nRanks),
                          std::vector<uint64_t>(peerUDS, peerUDS + comm->nRanks), sock->addr};
    free(peerAddresses);
    free(peerUDS);
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
std::deque<FakeNetReq> g_netReqs;  // kept until ResetSeams: failure paths abandon pending requests
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
  g_netReqs.push_back({true, nullptr, nullptr, 0, tag});
  *request = &g_netReqs.back();
  return ncclSuccess;
}
ncclResult_t FakeNetIrecv(void* recvComm, int, void** data, size_t* sizes, int* tags, void**, void**, void** request) {
  if (ncclResult_t res = NetFault("irecv")) return res;
  *request = nullptr;
  if (Defer(g_netDeferIrecv)) return ncclSuccess;
  std::lock_guard<std::mutex> lock(g_netMu);
  g_netReqs.push_back({false, static_cast<FakeNetConn*>(recvComm), data[0], sizes[0], tags[0]});
  *request = &g_netReqs.back();
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
    size_t n = std::min(req->size, it->second.size());
    if (n > 0) std::memcpy(req->data, it->second.data(), n);  // a zero-size recv may pass a null buffer
    msgs.erase(it);
  }
  *done = 1;
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

// bootstrapCreateRoot's listener is the only bootstrap-type one with no abort flag; its root thread closes it.
static bool IsRootListener(const struct ncclSocket* sock) {
  return sock != nullptr && sock->type == ncclSocketTypeBootstrap && sock->abortFlag == nullptr &&
         sock->state > ncclSocketStateNone && sock->socketDescriptor >= kListenFd;
}

void ResetSeams() {
  g_encrypted = false;
  g_proxyInits.clear();
  g_proxyInitResult = ncclSuccess;
  g_rasCommInits = 0;
  g_rasCommInitResult = ncclSuccess;
  g_rasAdded.clear();
  g_rasAddRanksResult = ncclSuccess;
  InstallInMemorySockets();
  g_ncclSocketClose = [](struct ncclSocket* sock, bool wait) {
    bool root = IsRootListener(sock);
    ncclResult_t res = SocketClose(sock, wait);
    if (root && res == ncclSuccess) {
      std::lock_guard<std::mutex> lock(g_sockMu);
      g_liveRoots--;
      g_sockCv.notify_all();
    }
    return res;
  };
  {
    std::lock_guard<std::mutex> lock(g_sockMu);
    g_liveRoots = 0;
  }
  {
    std::lock_guard<std::mutex> lock(g_netMu);
    g_netListens.clear();
    g_netConns.clear();
    g_netReqs.clear();
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
