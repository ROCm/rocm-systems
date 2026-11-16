/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Externals src/proxy.cc reaches, for rccl-UnitTestsMicroProxy, which compiles it. Distinct from
// proxy_fakes.cc, which fakes proxy.cc itself; the two are never linked together. HIP comes from
// hip_fakes.cc, linux.cc from os_fakes.cc and ncclGetEnv from env_fakes.cc.

#ifndef RCCL_TEST_HOST_PROXY_MICRO_FAKES_H_
#define RCCL_TEST_HOST_PROXY_MICRO_FAKES_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <vector>

#include "comm.h"
#include "ipcsocket.h"
#include "socket.h"
#include "transport.h"

// nccl_fakes.cc's ncclDebugLog only prints, and it cannot link here (it defines proxy.cc's client
// calls). These tests assert on the level of lines the proxy threads log, so this one records them.
bool ProxyLogged(ncclDebugLogLevel level, const char* needle);
int ProxyLogCount(const char* needle);
void ResetProxyLog();

// Backs every NCCL_PARAM in proxy.cc (via param_redirect.h); the default returns the default.
extern std::function<int64_t(const char*, int64_t)> g_loadParam;

// Sockets: socketDescriptor is one end of a socketpair (or pipe) the test owns.
struct ProxySockFakes {
  ncclResult_t initResult, connectResult, acceptResult, readyResult;
  int sendFailAt, recvFailAt, getFdFailAt;  // 0-based call index, -1 = never
  int sends, recvs, getFds, inits, connects, closes;
  int progressChunk;  // max bytes one ncclSocketProgress call moves
  int progresses, progressFailAt;
  std::function<void()> onProgress;  // runs before each ncclSocketProgress, e.g. to answer a request
  std::function<ncclResult_t()> onTryRecv;  // a non-success result is returned instead of reading
  std::function<void(int)> onClose;
  std::deque<int> connectFds, acceptFds;
  const union ncclSocketAddress* lastInitAddr;
  uint64_t lastInitMagic;
  int lastInitType;
};
extern ProxySockFakes g_sock;
extern std::vector<int> g_closedFds;  // every descriptor the unit closed, so fail-path cleanup never double-closes
bool WaitReadable(int fd);

// IPC (UDS) sockets: recorded, never touch the filesystem.
struct ProxyIpcSent {
  std::vector<char> data;
  ncclIpcFd fd;
  int rank;
  uint64_t hash;
};
struct ProxyIpcFakes {
  ncclResult_t initResult, sendResult, recvResult, closeResult, getFdResult, sendFdResult;
  int inits, closes, fdCloses, fdCloseResult;
  int lastInitRank;
  uint64_t lastInitHash;
  std::vector<ProxyIpcSent> sent;
  std::vector<char> recvBytes;  // copied into the receiver's buffer
  ncclIpcFd recvFd;
  std::vector<ncclIpcFd> closedFds;
  std::function<void()> onRecv;
};
extern ProxyIpcFakes g_ipc;

// Shared memory: heap-backed.
struct ProxyShmFakes {
  ncclResult_t openResult, closeResult, unlinkResult;
  int opens, closes, unlinks, lastRefcount;
  size_t lastSize;
  std::string lastPath;
};
extern ProxyShmFakes g_shm;

extern int g_cuMemEnabled, g_cuMemHostEnabled;
extern int g_threadNames;
extern bool g_proxyDiagEnabled;
extern int g_profOpStarts, g_profCtrlStarts, g_profCtrlStops;
extern std::vector<int> g_profCtrlStates;
extern std::atomic<int> g_profSleeps;  // set before the idle proxy waits, under the pool mutex

// Transports: g_transports[0] has every hook, each recording and returning a scripted result;
// g_transports[1] has none.
struct ProxyTransportFakes {
  std::vector<std::pair<ncclProxyConnection*, int>> frees;  // (connection, send)
  ncclResult_t freeResult, setupResult, connectResult, sharedInitResult, registerResult, deregisterResult;
  int setupDoneAfter, connectDone, registerDone, deregisterDone;  // setup reports done after this many calls
  int setups, connectsCalled, sharedInits, registers, deregisters;
  int lastSharedInitChannels;
  std::function<void()> onSetup;
};
extern ProxyTransportFakes g_tp;
extern ncclTransport g_transports[NTRANSPORTS];
extern std::atomic<int> g_progressCalls;
extern std::function<ncclResult_t(ncclProxyArgs*)> g_progressFn;  // default: progress nothing, stay busy
ncclResult_t FakeProxyProgress(ncclProxyState*, ncclProxyArgs* args);

void ResetProxyMicroFakes();

#endif  // RCCL_TEST_HOST_PROXY_MICRO_FAKES_H_
