/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Externals src/bootstrap.cc reaches, for rccl-UnitTestsMicroBootstrap, which compiles it. Distinct from
// bootstrap_stubs.cc, which fakes bootstrap.cc itself. Ranks run as threads over socket_fakes.cc's in-memory
// socket layer and the net plugin here; misc/utils.cc is the real one, and logging and g_loadParam, ncclGetEnv
// and linux.cc come from nccl_fakes.cc, env_fakes.cc and os_fakes.cc.

#ifndef RCCL_TEST_HOST_BOOTSTRAP_MICRO_FAKES_H_
#define RCCL_TEST_HOST_BOOTSTRAP_MICRO_FAKES_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "comm.h"
#include "net.h"
#include "ras.h"
#include "socket.h"
#include "socket_fakes.h"

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

// bootstrapCreateRoot threads still running: ncclSetThreadName("NCCL BootstrapR") counts one up and the
// close of its listener counts it down. ResetSeams() installs that close over socket_fakes.cc's in-memory one.
extern int g_liveRoots;

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
