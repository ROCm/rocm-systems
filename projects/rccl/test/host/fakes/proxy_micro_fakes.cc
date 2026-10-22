/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See proxy_micro_fakes.h.

#include "proxy_micro_fakes.h"

#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "debug_fakes.h"
#include "profiler.h"
#include "shmutils.h"
#include "socket_fakes.h"

// nccl_fakes.h declares g_loadParam, but nccl_fakes.cc cannot link here: it defines proxy.cc's client calls.
static int64_t DefaultLoadParam(const char*, int64_t deftVal) { return deftVal; }
std::function<int64_t(const char*, int64_t)> g_loadParam = DefaultLoadParam;

// Sockets.
ProxySockFakes g_sock;
std::vector<int> g_closedFds;

bool WaitReadable(int fd) {
  struct pollfd p = {fd, POLLIN, 0};
  return poll(&p, 1, 2000) > 0;
}

static ncclResult_t ProxySocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic,
                                    enum ncclSocketType type, volatile uint32_t* abortFlag, int, int) {
  g_sock.inits++;
  g_sock.lastInitAddr = addr;
  g_sock.lastInitMagic = magic;
  g_sock.lastInitType = type;
  sock->socketDescriptor = -1;
  sock->abortFlag = abortFlag;
  return g_sock.initResult;
}
static ncclResult_t ProxySocketConnect(struct ncclSocket* sock) {
  g_sock.connects++;
  if (g_sock.connectResult != ncclSuccess) return g_sock.connectResult;
  sock->socketDescriptor = g_sock.connectFds.empty() ? -1 : g_sock.connectFds.front();
  if (!g_sock.connectFds.empty()) g_sock.connectFds.pop_front();
  return ncclSuccess;
}
static ncclResult_t ProxySocketAccept(struct ncclSocket* sock, struct ncclSocket* listenSock, bool retry) {
  char c;
  if (read(listenSock->socketDescriptor, &c, 1) != 1 || retry) return ncclInternalError;
  if (g_sock.acceptResult != ncclSuccess || g_sock.acceptFds.empty()) return ncclSystemError;
  sock->socketDescriptor = g_sock.acceptFds.front();
  g_sock.acceptFds.pop_front();
  return ncclSuccess;
}
static ncclResult_t ProxySocketReady(struct ncclSocket* sock, int* running) {
  *running = sock->socketDescriptor != -1;
  return g_sock.readyResult;
}
static ncclResult_t ProxySocketGetFd(struct ncclSocket* sock, ncclSocketDescriptor* fd) {
  if (g_sock.getFds++ == g_sock.getFdFailAt) return ncclInternalError;
  *fd = sock->socketDescriptor;
  return ncclSuccess;
}
static ncclResult_t ProxySocketSetFd(ncclSocketDescriptor fd, struct ncclSocket* sock) {
  sock->socketDescriptor = fd;
  return ncclSuccess;
}
static ncclResult_t ProxySocketClose(struct ncclSocket* sock, bool) {
  g_sock.closes++;
  g_closedFds.push_back(sock->socketDescriptor);
  if (g_sock.onClose) g_sock.onClose(sock->socketDescriptor);
  if (sock->socketDescriptor >= 0) close(sock->socketDescriptor);
  sock->socketDescriptor = -1;
  return ncclSuccess;
}
static ncclResult_t ProxySocketSend(struct ncclSocket* sock, void* ptr, int size) {
  if (g_sock.sends++ == g_sock.sendFailAt) return ncclRemoteError;
  for (int off = 0; off < size;) {
    ssize_t n = send(sock->socketDescriptor, (char*)ptr + off, size - off, MSG_NOSIGNAL);
    if (n <= 0) return ncclSystemError;
    off += n;
  }
  return ncclSuccess;
}
static ncclResult_t ProxySocketRecv(struct ncclSocket* sock, void* ptr, int size) {
  if (g_sock.recvs++ == g_sock.recvFailAt) return ncclRemoteError;
  for (int off = 0; off < size;) {
    if (!WaitReadable(sock->socketDescriptor)) return ncclSystemError;
    ssize_t n = read(sock->socketDescriptor, (char*)ptr + off, size - off);
    if (n <= 0) return ncclSystemError;
    off += n;
  }
  return ncclSuccess;
}
static ncclResult_t ProxySocketProgress(int, struct ncclSocket* sock, void* ptr, int size, int* offset, bool*) {
  if (g_sock.onProgress) g_sock.onProgress();
  if (g_sock.progresses++ == g_sock.progressFailAt) return ncclRemoteError;
  ssize_t n = recv(sock->socketDescriptor, (char*)ptr + *offset, std::min(size - *offset, g_sock.progressChunk),
                   MSG_DONTWAIT);
  if (n > 0) *offset += n;
  return ncclSuccess;
}
static ncclResult_t ProxySocketTryRecv(struct ncclSocket* sock, void* ptr, int size, bool* closed, bool) {
  *closed = false;
  ncclResult_t res = g_sock.onTryRecv ? g_sock.onTryRecv() : ncclSuccess;
  if (res != ncclSuccess) return res;
  ssize_t n = recv(sock->socketDescriptor, ptr, size, MSG_DONTWAIT);
  if (n == 0) *closed = true;
  if (n <= 0) return n == 0 ? ncclSuccess : ncclInProgress;
  return n == size ? ncclSuccess : ProxySocketRecv(sock, (char*)ptr + n, size - n);
}
static const char* ProxySocketToString(const union ncclSocketAddress*, char* buf, size_t len, const int) {
  std::snprintf(buf, len, "fake");
  return buf;
}

// socketDescriptor is one end of a socketpair (or pipe) the test owns, so close() really closes it.
static void InstallProxySockets() {
  ResetSocketFakes();
  ResetExtraSocketFakes();
  g_ncclSocketInit = ProxySocketInit;
  g_ncclSocketConnect = ProxySocketConnect;
  g_ncclSocketAccept = ProxySocketAccept;
  g_ncclSocketReady = ProxySocketReady;
  g_ncclSocketGetFd = ProxySocketGetFd;
  g_ncclSocketSetFd = ProxySocketSetFd;
  g_ncclSocketClose = ProxySocketClose;
  g_ncclSocketSend = ProxySocketSend;
  g_ncclSocketRecv = ProxySocketRecv;
  g_ncclSocketProgress = ProxySocketProgress;
  g_ncclSocketTryRecv = ProxySocketTryRecv;
  g_ncclSocketToString = ProxySocketToString;
}

// IPC (UDS) sockets.
ProxyIpcFakes g_ipc;

ncclResult_t ncclIpcSocketInit(struct ncclIpcSocket* handle, int rank, uint64_t hash, volatile uint32_t* abortFlag) {
  g_ipc.inits++;
  g_ipc.lastInitRank = rank;
  g_ipc.lastInitHash = hash;
  handle->fd = 1000 + rank;
  handle->abortFlag = abortFlag;
  return g_ipc.initResult;
}
ncclResult_t ncclIpcSocketClose(struct ncclIpcSocket*) {
  g_ipc.closes++;
  return g_ipc.closeResult;
}
ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket* handle, ncclIpcFd* fd) {
  *fd = handle->fd;
  return g_ipc.getFdResult;
}
ncclResult_t ncclIpcSocketSendFd(struct ncclIpcSocket*, ncclIpcFd fd, int rank, uint64_t hash) {
  g_ipc.sent.push_back({{}, fd, rank, hash});
  return g_ipc.sendFdResult;
}
ncclResult_t ncclIpcSocketSendMsg(ncclIpcSocket*, void* hdr, int hdrLen, ncclIpcFd sendFd, int rank, uint64_t hash) {
  g_ipc.sent.push_back({std::vector<char>((char*)hdr, (char*)hdr + hdrLen), sendFd, rank, hash});
  return g_ipc.sendResult;
}
ncclResult_t ncclIpcSocketRecvMsg(ncclIpcSocket*, void* hdr, int hdrLen, ncclIpcFd* recvFd) {
  if (g_ipc.onRecv) g_ipc.onRecv();
  size_t n = std::min<size_t>(hdrLen, g_ipc.recvBytes.size());
  if (n > 0) std::memcpy(hdr, g_ipc.recvBytes.data(), n);  // hdr and data() may be null when n is 0
  if (recvFd) *recvFd = g_ipc.recvFd;
  return g_ipc.recvResult;
}
int ncclIpcFdClose(ncclIpcFd fd) {
  g_ipc.fdCloses++;
  g_ipc.closedFds.push_back(fd);
  errno = EBADF;
  return g_ipc.fdCloseResult;
}

// Shared memory.
ProxyShmFakes g_shm;

ncclResult_t ncclShmOpen(char* shmPath, size_t shmPathSize, size_t shmSize, void** shmPtr, void**, int refcount,
                         ncclShmHandle_t* handle) {
  g_shm.opens++;
  g_shm.lastRefcount = refcount;
  if (g_shm.openResult != ncclSuccess) return g_shm.openResult;
  if (shmPath[0] == '\0') std::snprintf(shmPath, shmPathSize, "/dev/shm/nccl-AbC123");
  g_shm.lastPath = shmPath;
  *shmPtr = *handle = calloc(1, shmSize);
  return ncclSuccess;
}
ncclResult_t ncclShmClose(ncclShmHandle_t handle) {
  g_shm.closes++;
  free(handle);
  return g_shm.closeResult;
}
ncclResult_t ncclShmUnlink(ncclShmHandle_t) {
  g_shm.unlinks++;
  return g_shm.unlinkResult;
}

// cuMem, profiler.
int g_cuMemEnabled, g_cuMemHostEnabled;
bool g_proxyDiagEnabled;
int g_profOpStarts, g_profCtrlStarts, g_profCtrlStops;
std::vector<int> g_profCtrlStates;
std::atomic<int> g_profSleeps;

int ncclCuMemEnable() { return g_cuMemEnabled; }
int ncclCuMemHostEnable() { return g_cuMemHostEnabled; }

bool ncclProfilerProxyDiagEnabled(void) { return g_proxyDiagEnabled; }
ncclResult_t ncclProfilerStartProxyOpEvent(int, struct ncclProxyArgs*) {
  g_profOpStarts++;
  return ncclSuccess;
}
ncclResult_t ncclProfilerStartProxyCtrlEvent(void*, void** eHandle) {
  g_profCtrlStarts++;
  *eHandle = nullptr;
  return ncclSuccess;
}
ncclResult_t ncclProfilerStopProxyCtrlEvent(void*) {
  g_profCtrlStops++;
  return ncclSuccess;
}
ncclResult_t ncclProfilerRecordProxyCtrlEventState(void*, int, ncclProfilerEventState_t state) {
  if (state == ncclProfilerProxyCtrlSleep) g_profSleeps++;
  g_profCtrlStates.push_back(state);
  return ncclSuccess;
}

// Transports.
ProxyTransportFakes g_tp;
std::atomic<int> g_progressCalls;
std::function<ncclResult_t(ncclProxyArgs*)> g_progressFn;

namespace {
ncclResult_t FakeProxyFree(ncclProxyConnection* c, ncclProxyState*) {
  g_tp.frees.push_back({c, c->send});
  return g_tp.freeResult;
}
ncclResult_t FakeProxySetup(ncclProxyConnection*, ncclProxyState*, void* req, int reqSize, void* resp, int respSize,
                            int* done) {
  g_tp.setups++;
  if (g_tp.onSetup) g_tp.onSetup();
  if (respSize >= 4 && reqSize >= 4) std::memcpy(resp, req, 4);  // echo
  *done = g_tp.setups > g_tp.setupDoneAfter;
  return g_tp.setupResult;
}
ncclResult_t FakeProxyConnect(ncclProxyConnection*, ncclProxyState*, void*, int, void*, int, int* done) {
  g_tp.connectsCalled++;
  *done = g_tp.connectDone;
  return g_tp.connectResult;
}
ncclResult_t FakeProxySharedInit(ncclProxyConnection*, ncclProxyState*, int nChannels) {
  g_tp.sharedInits++;
  g_tp.lastSharedInitChannels = nChannels;
  return g_tp.sharedInitResult;
}
ncclResult_t FakeProxyRegister(ncclProxyConnection*, ncclProxyState*, void*, int, void*, int, int* done) {
  g_tp.registers++;
  *done = g_tp.registerDone;
  return g_tp.registerResult;
}
ncclResult_t FakeProxyDeregister(ncclProxyConnection*, ncclProxyState*, void*, int, int* done) {
  g_tp.deregisters++;
  *done = g_tp.deregisterDone;
  return g_tp.deregisterResult;
}
}  // namespace

ncclResult_t FakeProxyProgress(ncclProxyState*, ncclProxyArgs* args) {
  g_progressCalls++;
  return g_progressFn ? g_progressFn(args) : ncclSuccess;
}

ncclTransport g_transports[NTRANSPORTS] = {{"full"}, {"empty"}, {"t2"}, {"t3"}};
struct ncclTransport* ncclTransports[NTRANSPORTS] = {&g_transports[0], &g_transports[1], &g_transports[2],
                                                     &g_transports[3]};

void ResetProxyMicroFakes() {
  ResetDebugLog();
  g_loadParam = DefaultLoadParam;
  InstallProxySockets();
  g_sock = ProxySockFakes{};
  g_sock.sendFailAt = g_sock.recvFailAt = g_sock.getFdFailAt = g_sock.progressFailAt = -1;
  g_sock.progressChunk = 1 << 20;
  g_closedFds.clear();
  g_ipc = ProxyIpcFakes{};
  g_ipc.recvFd = NCCL_INVALID_IPC_FD;
  g_shm = ProxyShmFakes{};
  g_cuMemEnabled = g_cuMemHostEnabled = 0;
  g_ncclSetThreadNameCalls = 0;
  g_proxyDiagEnabled = false;
  g_profOpStarts = g_profCtrlStarts = g_profCtrlStops = 0;
  g_profCtrlStates.clear();
  g_profSleeps = 0;
  g_tp = ProxyTransportFakes{};
  g_tp.connectDone = g_tp.registerDone = g_tp.deregisterDone = 1;
  g_progressCalls = 0;
  g_progressFn = nullptr;
  ncclTransportComm full = {};
  full.proxyFree = FakeProxyFree;
  full.proxySetup = FakeProxySetup;
  full.proxyConnect = FakeProxyConnect;
  full.proxySharedInit = FakeProxySharedInit;
  full.proxyRegister = FakeProxyRegister;
  full.proxyDeregister = FakeProxyDeregister;
  full.proxyProgress = FakeProxyProgress;
  g_transports[0].send = g_transports[0].recv = full;
  g_transports[1].send = g_transports[1].recv = ncclTransportComm{};
}
