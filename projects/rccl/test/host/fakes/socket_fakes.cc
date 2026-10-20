/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "fakes/socket_fakes.h"

#include <cstdio>
#include <cstring>

#include "signature-drift.h"

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

static ncclResult_t DefaultNcclSocketListen(struct ncclSocket* sock) {
  sock->state = ncclSocketStateReady;
  return ncclSuccess;
}

static ncclResult_t DefaultNcclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr) {
  if (sock->state != ncclSocketStateReady) return ncclInternalError;
  *addr = sock->addr;
  return ncclSuccess;
}

static ncclResult_t DefaultNcclSocketConnect(struct ncclSocket* sock) {
  sock->state = ncclSocketStateConnecting;
  return ncclSuccess;
}

static ncclResult_t DefaultNcclSocketAccept(struct ncclSocket* sock, struct ncclSocket*, bool) {
  sock->state = ncclSocketStateAccepting;
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

static ncclResult_t DefaultNcclSocketProgress(int, struct ncclSocket*, void*, int size, int* offset, bool* closed) {
  *offset = size;
  if (closed) *closed = false;
  return ncclSuccess;
}

ncclResult_t DefaultNcclSocketClose(struct ncclSocket* sock, bool) {
  sock->state = ncclSocketStateClosed;
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  return ncclSuccess;
}

static ncclResult_t DefaultNcclSocketGetFd(struct ncclSocket* sock, ncclSocketDescriptor* fd) {
  *fd = sock->socketDescriptor;
  return ncclSuccess;
}

static ncclResult_t DefaultNcclFindInterfaces(char*, union ncclSocketAddress*, int, int, int*) {
  return ncclInternalError;
}

std::function<ncclResult_t(struct ncclSocket*, const union ncclSocketAddress*, uint64_t, enum ncclSocketType,
                           volatile uint32_t*, int, int)>
    g_ncclSocketInit = DefaultNcclSocketInit;
std::function<ncclResult_t(struct ncclSocket*)> g_ncclSocketListen = DefaultNcclSocketListen;
std::function<ncclResult_t(struct ncclSocket*, union ncclSocketAddress*)> g_ncclSocketGetAddr =
    DefaultNcclSocketGetAddr;
std::function<ncclResult_t(struct ncclSocket*)> g_ncclSocketConnect = DefaultNcclSocketConnect;
std::function<ncclResult_t(struct ncclSocket*, struct ncclSocket*, bool)> g_ncclSocketAccept =
    DefaultNcclSocketAccept;
std::function<ncclResult_t(struct ncclSocket*, int*)> g_ncclSocketReady = DefaultNcclSocketReady;
std::function<ncclResult_t(int, struct ncclSocket*, void*, int, int*, bool*)> g_ncclSocketProgress =
    DefaultNcclSocketProgress;
std::function<ncclResult_t(struct ncclSocket*, bool)> g_ncclSocketClose = DefaultNcclSocketClose;
std::function<ncclResult_t(struct ncclSocket*, ncclSocketDescriptor*)> g_ncclSocketGetFd = DefaultNcclSocketGetFd;
std::function<ncclResult_t(char*, union ncclSocketAddress*, int, int, int*)> g_ncclFindInterfaces =
    DefaultNcclFindInterfaces;

ASSERT_HOOK_MATCHES_PROD(g_ncclSocketProgress, ncclSocketProgress);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketAccept, ncclSocketAccept);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketClose, ncclSocketClose);
#undef ASSERT_HOOK_MATCHES_PROD

void ResetSocketFakes() {
  g_ncclSocketInit = DefaultNcclSocketInit;
  g_ncclSocketListen = DefaultNcclSocketListen;
  g_ncclSocketGetAddr = DefaultNcclSocketGetAddr;
  g_ncclSocketConnect = DefaultNcclSocketConnect;
  g_ncclSocketAccept = DefaultNcclSocketAccept;
  g_ncclSocketReady = DefaultNcclSocketReady;
  g_ncclSocketProgress = DefaultNcclSocketProgress;
  g_ncclSocketClose = DefaultNcclSocketClose;
  g_ncclSocketGetFd = DefaultNcclSocketGetFd;
  g_ncclFindInterfaces = DefaultNcclFindInterfaces;
}

uint64_t ncclSocketDefaultMagic() { return NCCL_SOCKET_MAGIC; }

ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic,
                            enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag, int customRetry) {
  return g_ncclSocketInit(sock, addr, magic, type, abortFlag, asyncFlag, customRetry);
}

// Same field handling as the real ncclSocketMove, minus the TLS rebind; the size check flags a new field.
static_assert(sizeof(struct ncclSocket) == 104, "ncclSocket changed; re-check this copy of ncclSocketMove");
void ncclSocketMove(struct ncclSocket* dst, struct ncclSocket* src) {
  std::memcpy(dst, src, sizeof(*dst));
  src->socketDescriptor = NCCL_INVALID_SOCKET;
  src->acceptSocketDescriptor = NCCL_INVALID_SOCKET;
  src->state = ncclSocketStateNone;
  src->crypto = nullptr;
}

ncclResult_t ncclSocketListen(struct ncclSocket* sock) { return g_ncclSocketListen(sock); }
ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr) {
  return g_ncclSocketGetAddr(sock, addr);
}
ncclResult_t ncclSocketConnect(struct ncclSocket* sock) { return g_ncclSocketConnect(sock); }
ncclResult_t ncclSocketReady(struct ncclSocket* sock, int* running) { return g_ncclSocketReady(sock, running); }
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* listenSock, bool retry) {
  return g_ncclSocketAccept(sock, listenSock, retry);
}
ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, bool* closed) {
  return g_ncclSocketProgress(op, sock, ptr, size, offset, closed);
}
ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool wait) { return g_ncclSocketClose(sock, wait); }
ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, ncclSocketDescriptor* fd) { return g_ncclSocketGetFd(sock, fd); }
const char* ncclSocketToString(const union ncclSocketAddress*, char* buf, size_t, const int) {
  buf[0] = '\0';
  return buf;
}
ncclResult_t ncclFindInterfaces(char* ifNames, union ncclSocketAddress* ifAddrs, int ifNameMaxSize, int maxIfs,
                                int* nIfs) {
  return g_ncclFindInterfaces(ifNames, ifAddrs, ifNameMaxSize, maxIfs, nIfs);
}
