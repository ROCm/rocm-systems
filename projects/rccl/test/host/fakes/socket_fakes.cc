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
std::function<void(struct ncclSocket*, struct ncclSocket*)> g_ncclSocketMove = DefaultSocketMove;
std::function<ncclResult_t(struct ncclSocket*, ncclSocketDescriptor*)> g_ncclSocketGetFd = DefaultSocketGetFd;
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

void ncclSocketMove(struct ncclSocket* dst, struct ncclSocket* src) { g_ncclSocketMove(dst, src); }

ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, ncclSocketDescriptor* fd) { return g_ncclSocketGetFd(sock, fd); }

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
  g_ncclSocketMove = DefaultSocketMove;
  g_ncclSocketGetFd = DefaultSocketGetFd;
  g_ncclFindInterfaces = DefaultFindInterfaces;
  g_ncclSocketToString = DefaultSocketToString;
}

ncclResult_t SucceedingSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic,
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

ncclResult_t SucceedingSocketReady(struct ncclSocket* sock, int* running) {
  if (sock->state == ncclSocketStateError || sock->state == ncclSocketStateClosed) return ncclRemoteError;
  if (sock->state == ncclSocketStateConnecting || sock->state == ncclSocketStateAccepting) {
    sock->state = ncclSocketStateReady;
  }
  *running = sock->state == ncclSocketStateReady;
  return ncclSuccess;
}

// Same field handling as the real ncclSocketMove, minus the TLS rebind; the size check flags a new field.
static_assert(sizeof(struct ncclSocket) == 104, "ncclSocket changed; re-check this copy of ncclSocketMove");
static void SucceedingSocketMove(struct ncclSocket* dst, struct ncclSocket* src) {
  std::memcpy(dst, src, sizeof(*dst));
  src->socketDescriptor = NCCL_INVALID_SOCKET;
  src->acceptSocketDescriptor = NCCL_INVALID_SOCKET;
  src->state = ncclSocketStateNone;
  src->crypto = nullptr;
}

void InstallSucceedingSocketFakes() {
  ResetSocketFakes();
  g_ncclSocketInit = SucceedingSocketInit;
  g_ncclSocketListen = [](struct ncclSocket* sock) {
    sock->state = ncclSocketStateReady;
    return ncclSuccess;
  };
  g_ncclSocketGetAddr = [](struct ncclSocket* sock, union ncclSocketAddress* addr) {
    if (sock->state != ncclSocketStateReady) return ncclInternalError;
    *addr = sock->addr;
    return ncclSuccess;
  };
  g_ncclSocketConnect = [](struct ncclSocket* sock) {
    sock->state = ncclSocketStateConnecting;
    return ncclSuccess;
  };
  g_ncclSocketAccept = [](struct ncclSocket* sock, struct ncclSocket*, bool) {
    sock->state = ncclSocketStateAccepting;
    return ncclSuccess;
  };
  g_ncclSocketReady = SucceedingSocketReady;
  g_ncclSocketProgress = [](int, struct ncclSocket*, void*, int size, int* offset, bool* closed) {
    *offset = size;
    if (closed) *closed = false;
    return ncclSuccess;
  };
  g_ncclSocketMove = SucceedingSocketMove;
  g_ncclSocketGetFd = [](struct ncclSocket* sock, ncclSocketDescriptor* fd) {
    *fd = sock->socketDescriptor;
    return ncclSuccess;
  };
  g_ncclSocketToString = [](const union ncclSocketAddress*, char* buffer, size_t, int) {
    buffer[0] = '\0';
    return static_cast<const char*>(buffer);
  };
}
