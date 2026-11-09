/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Externals src/bootstrap.cc reaches, for rccl-UnitTestsMicroBootstrap, which compiles it. Distinct from
// bootstrap_stubs.cc, which fakes bootstrap.cc itself. Ranks run as threads over the in-memory socket layer
// and net plugin here; misc/utils.cc is the real one, and logging and g_loadParam, ncclGetEnv and linux.cc
// come from nccl_fakes.cc, env_fakes.cc and os_fakes.cc.

#ifndef RCCL_TEST_HOST_BOOTSTRAP_MICRO_FAKES_H_
#define RCCL_TEST_HOST_BOOTSTRAP_MICRO_FAKES_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "comm.h"
#include "net.h"
#include "ras.h"
#include "socket.h"

extern bool g_encrypted;  // what ncclGetCryptConnectionMode reports

namespace bootstrap_micro {

struct ProxyInitCall {
  std::vector<ncclSocketAddress> addrs;
  std::vector<uint64_t> uds;
  ncclSocketAddress own;  // the proxy listen socket's
};
extern std::map<ncclComm*, ProxyInitCall> g_proxyInits;
extern ncclResult_t g_proxyInitResult;
extern int g_rasCommInits;
extern ncclResult_t g_rasCommInitResult;
extern std::vector<rasRankInit> g_rasAdded;
extern ncclResult_t g_rasAddRanksResult;
extern int g_ifCount;
extern union ncclSocketAddress g_subnetRemote;

// In-memory socket layer: a listener is a port with a queue of pending connections, a connection is
// two endpoints whose inbound byte queues the other side appends to. Blocking calls honour the
// socket's abort flag and fail the test instead of hanging.
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
extern std::mutex g_sockMu;
extern std::condition_variable g_sockCv;
extern std::deque<FakeListener> g_listeners;
extern std::set<int> g_openFds;
extern bool g_sockShutdown;
extern int g_liveRoots;
extern std::chrono::milliseconds g_waitLimit;
// Mirrors socket.cc: ncclSocketAccept(retry=false) hands back a rejected peer as Accepting, not BadHandshake.
extern ncclSocketState g_rejectedState;
extern int g_readyPolls;  // ncclSocketReady says "not yet" this many times first
extern std::atomic<int> g_multiOps;  // only the bidirectional socket ring uses ncclSocketMultiOp
using SockFaultFn = std::function<ncclResult_t(const char* /*op*/, struct ncclSocket*)>;
void SetSockFault(SockFaultFn fn);
uint16_t PortOf(const union ncclSocketAddress& addr);

// In-memory net plugin. listen() writes a listener id into the handle; connect() queues a connection
// on it that accept() picks up. isend copies into the connection at once; irecv matches by tag.
struct FakeNetConn {
  std::deque<std::pair<int, std::string>> msgs;
};
struct FakeNetListener {
  int dev = 0;
  std::deque<FakeNetConn*> pending;
};
extern std::deque<FakeNetListener> g_netListens;
extern std::vector<std::string> g_netDevs;
extern std::atomic<int> g_netDeferConnect, g_netDeferIsend, g_netDeferIrecv, g_netDeferTest;
extern std::atomic<int> g_netRegs, g_netDeregs, g_netCloses;
extern std::function<ncclResult_t(const char* /*op*/)> g_netFault;
extern ncclNet_t g_fakeNet;

void ResetSeams();
// Lets every detached root thread unwind before the next test reuses the fake network.
void ShutdownSockets();

}  // namespace bootstrap_micro

#endif  // RCCL_TEST_HOST_BOOTSTRAP_MICRO_FAKES_H_
