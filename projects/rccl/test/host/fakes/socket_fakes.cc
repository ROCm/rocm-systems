/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/socket.cc fakes. See socket_fakes.h.

#include "socket_fakes.h"

#include <cstdio>

#include "fail_loud.h"
#include "signature-drift.h"

static ncclResult_t DefaultNcclSocketInit(struct ncclSocket*, const union ncclSocketAddress*, uint64_t,
                                          enum ncclSocketType, volatile uint32_t*, int, int) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ncclSocket*, const union ncclSocketAddress*, uint64_t, enum ncclSocketType,
                           volatile uint32_t*, int, int)>
    g_ncclSocketInit = DefaultNcclSocketInit;

static ncclResult_t DefaultNcclSocketListen(struct ncclSocket*) { return ncclSystemError; }
std::function<ncclResult_t(struct ncclSocket*)> g_ncclSocketListen = DefaultNcclSocketListen;

static ncclResult_t DefaultNcclSocketGetAddr(struct ncclSocket*, union ncclSocketAddress*) { return ncclSystemError; }
std::function<ncclResult_t(struct ncclSocket*, union ncclSocketAddress*)> g_ncclSocketGetAddr =
    DefaultNcclSocketGetAddr;

static ncclResult_t DefaultNcclSocketConnect(struct ncclSocket*) { return ncclSystemError; }
std::function<ncclResult_t(struct ncclSocket*)> g_ncclSocketConnect = DefaultNcclSocketConnect;

static ncclResult_t DefaultNcclSocketReady(struct ncclSocket*, int*) { return ncclSystemError; }
std::function<ncclResult_t(struct ncclSocket*, int*)> g_ncclSocketReady = DefaultNcclSocketReady;

static ncclResult_t DefaultNcclSocketAccept(struct ncclSocket*, struct ncclSocket*, bool) { return ncclSystemError; }
std::function<ncclResult_t(struct ncclSocket*, struct ncclSocket*, bool)> g_ncclSocketAccept =
    DefaultNcclSocketAccept;

static ncclResult_t DefaultNcclSocketSend(struct ncclSocket*, void*, int) { return ncclSystemError; }
std::function<ncclResult_t(struct ncclSocket*, void*, int)> g_ncclSocketSend = DefaultNcclSocketSend;

static ncclResult_t DefaultNcclSocketRecv(struct ncclSocket*, void*, int) { return ncclSystemError; }
std::function<ncclResult_t(struct ncclSocket*, void*, int)> g_ncclSocketRecv = DefaultNcclSocketRecv;

static ncclResult_t DefaultNcclSocketProgress(int, struct ncclSocket*, void*, int, int*, bool*) {
  return ncclSystemError;
}
std::function<ncclResult_t(int, struct ncclSocket*, void*, int, int*, bool*)> g_ncclSocketProgress =
    DefaultNcclSocketProgress;

static ncclResult_t DefaultNcclSocketClose(struct ncclSocket*, bool) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclSocket*, bool)> g_ncclSocketClose = DefaultNcclSocketClose;

static void DefaultNcclSocketMove(struct ncclSocket*, struct ncclSocket*) {
  FailLoudUnfaked("socket_fakes", "ncclSocketMove");
}
std::function<void(struct ncclSocket*, struct ncclSocket*)> g_ncclSocketMove = DefaultNcclSocketMove;

static ncclResult_t DefaultNcclSocketGetFd(struct ncclSocket*, ncclSocketDescriptor*) { return ncclSystemError; }
std::function<ncclResult_t(struct ncclSocket*, ncclSocketDescriptor*)> g_ncclSocketGetFd = DefaultNcclSocketGetFd;

static const char* DefaultNcclSocketToString(const union ncclSocketAddress*, char* buf, size_t bufLen, const int) {
  if (buf && bufLen > 0) buf[0] = '\0';
  return buf;
}
std::function<const char*(const union ncclSocketAddress*, char*, size_t, const int)> g_ncclSocketToString =
    DefaultNcclSocketToString;

static ncclResult_t DefaultNcclFindInterfaces(char*, union ncclSocketAddress*, int, int, int* nIfs) {
  if (nIfs) *nIfs = 0;
  return ncclSuccess;
}
std::function<ncclResult_t(char*, union ncclSocketAddress*, int, int, int*)> g_ncclFindInterfaces =
    DefaultNcclFindInterfaces;

ASSERT_HOOK_MATCHES_PROD(g_ncclSocketInit, ncclSocketInit);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketListen, ncclSocketListen);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketGetAddr, ncclSocketGetAddr);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketConnect, ncclSocketConnect);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketReady, ncclSocketReady);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketAccept, ncclSocketAccept);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketSend, ncclSocketSend);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketRecv, ncclSocketRecv);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketProgress, ncclSocketProgress);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketClose, ncclSocketClose);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketMove, ncclSocketMove);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketGetFd, ncclSocketGetFd);
ASSERT_HOOK_MATCHES_PROD(g_ncclSocketToString, ncclSocketToString);
ASSERT_HOOK_MATCHES_PROD(g_ncclFindInterfaces, ncclFindInterfaces);

uint64_t ncclSocketDefaultMagic(void) { return NCCL_SOCKET_MAGIC; }

ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic,
                            enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag, int customRetry) {
  return g_ncclSocketInit(sock, addr, magic, type, abortFlag, asyncFlag, customRetry);
}
ncclResult_t ncclSocketListen(struct ncclSocket* sock) { return g_ncclSocketListen(sock); }
ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr) {
  return g_ncclSocketGetAddr(sock, addr);
}
ncclResult_t ncclSocketConnect(struct ncclSocket* sock) { return g_ncclSocketConnect(sock); }
ncclResult_t ncclSocketReady(struct ncclSocket* sock, int* running) { return g_ncclSocketReady(sock, running); }
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* ulistenSock, bool retry) {
  return g_ncclSocketAccept(sock, ulistenSock, retry);
}
ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size) { return g_ncclSocketSend(sock, ptr, size); }
ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size) { return g_ncclSocketRecv(sock, ptr, size); }
ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, bool* closed) {
  return g_ncclSocketProgress(op, sock, ptr, size, offset, closed);
}
ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool wait) { return g_ncclSocketClose(sock, wait); }
void ncclSocketMove(struct ncclSocket* dst, struct ncclSocket* src) { g_ncclSocketMove(dst, src); }
ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, ncclSocketDescriptor* fd) { return g_ncclSocketGetFd(sock, fd); }
const char* ncclSocketToString(const union ncclSocketAddress* addr, char* buf, size_t bufLen,
                               const int numericHostForm) {
  return g_ncclSocketToString(addr, buf, bufLen, numericHostForm);
}
ncclResult_t ncclFindInterfaces(char* ifNames, union ncclSocketAddress* ifAddrs, int ifNameMaxSize, int maxIfs,
                                int* nIfs) {
  return g_ncclFindInterfaces(ifNames, ifAddrs, ifNameMaxSize, maxIfs, nIfs);
}

void ResetSocketFakes() {
  g_ncclSocketInit = DefaultNcclSocketInit;
  g_ncclSocketListen = DefaultNcclSocketListen;
  g_ncclSocketGetAddr = DefaultNcclSocketGetAddr;
  g_ncclSocketConnect = DefaultNcclSocketConnect;
  g_ncclSocketReady = DefaultNcclSocketReady;
  g_ncclSocketAccept = DefaultNcclSocketAccept;
  g_ncclSocketSend = DefaultNcclSocketSend;
  g_ncclSocketRecv = DefaultNcclSocketRecv;
  g_ncclSocketProgress = DefaultNcclSocketProgress;
  g_ncclSocketClose = DefaultNcclSocketClose;
  g_ncclSocketMove = DefaultNcclSocketMove;
  g_ncclSocketGetFd = DefaultNcclSocketGetFd;
  g_ncclSocketToString = DefaultNcclSocketToString;
  g_ncclFindInterfaces = DefaultNcclFindInterfaces;
}
