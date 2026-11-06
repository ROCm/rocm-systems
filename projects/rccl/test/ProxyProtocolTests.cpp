/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ProxyProtocol: the proxy service thread (src/proxy.cc, ncclProxyService) serves any TCP
// peer that knows the communicator magic. These tests act as such a peer against the proxy
// of a one-rank communicator: a request the proxy must not act on has to close the
// connection instead of being served.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <functional>
#include <ostream>
#include <vector>

#include "comm.h"
#include "transport.h"
#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {
namespace {

// Wire layouts of the Init request and response, which src/proxy.cc keeps private.
struct InitReq { int transport, send, tpLocalRank, tpRank, sameProcess; };
struct InitResp { void* connection; int connId; char devShmPath[6]; };

// Init of a P2P receive connection, which allocates no transport resources.
constexpr InitReq kValidInit = {TRANSPORT_P2P, /*send*/ 0, /*tpLocalRank*/ 0, /*tpRank*/ 0, /*sameProcess*/ 1};
constexpr int kTimeoutMs = 10000;
void* const kOpId = reinterpret_cast<void*>(uintptr_t{0x5eed});

enum class Reply { Answered, Closed, Silent };
std::ostream& operator<<(std::ostream& os, Reply r) {
  return os << (r == Reply::Answered ? "answered" : r == Reply::Closed ? "closed" : "silent");
}

// Connect to the proxy and send the plaintext hello (magic, socket type) ncclSocketAccept expects.
int connectToProxy(ncclComm_t comm) {
  const ncclSocket* listenSock = comm->proxyState->listenSock;
  // A cast, not addr.sa: socket.h swaps in an opaque union of the same size for device or OS-less builds.
  const sockaddr* sa = reinterpret_cast<const sockaddr*>(&listenSock->addr);
  struct { uint64_t magic; int type; } __attribute__((packed)) hello = {listenSock->magic, listenSock->type};
  timeval tv = {kTimeoutMs / 1000, 0};
  int fd = socket(sa->sa_family, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0 ||
      connect(fd, sa, sa->sa_family == AF_INET ? sizeof(sockaddr_in) : sizeof(sockaddr_in6)) != 0 ||
      send(fd, &hello, sizeof(hello), MSG_NOSIGNAL) != (ssize_t)sizeof(hello)) {
    close(fd);
    return -1;
  }
  return fd;
}

// Frame one RPC as ncclProxyCallAsync does: type, connId, reqSize, respSize, request, opId.
bool sendOp(int fd, int type, int connId, const void* req, int reqSize, int respSize) {
  int head[] = {type, connId, reqSize, respSize};
  std::vector<char> msg((const char*)head, (const char*)(head + 4));
  msg.insert(msg.end(), (const char*)req, (const char*)req + reqSize);
  msg.insert(msg.end(), (const char*)&kOpId, (const char*)(&kOpId + 1));
  return send(fd, msg.data(), msg.size(), MSG_NOSIGNAL) == (ssize_t)msg.size();
}

// How the proxy reacted within timeoutMs: with a response, by closing the connection, or not at all.
Reply awaitReply(int fd, int timeoutMs) {
  pollfd pfd = {fd, POLLIN, 0};
  if (poll(&pfd, 1, timeoutMs) <= 0) return Reply::Silent;
  char byte;
  return recv(fd, &byte, 1, MSG_PEEK) > 0 ? Reply::Answered : Reply::Closed;
}

// Send a well-formed Init and return the connId the proxy assigned, or -1.
int initConnection(int fd) {
  ncclProxyRpcResponseHeader hdr;
  InitResp resp;
  if (!sendOp(fd, ncclProxyMsgInit, -1, &kValidInit, sizeof(kValidInit), sizeof(resp)) ||
      recv(fd, &hdr, sizeof(hdr), MSG_WAITALL) != (ssize_t)sizeof(hdr) || hdr.opId != kOpId ||
      hdr.res != ncclSuccess || hdr.respSize != (int)sizeof(resp) ||
      recv(fd, &resp, sizeof(resp), MSG_WAITALL) != (ssize_t)sizeof(resp))
    return -1;
  return resp.connId;
}

// Run probe against the proxy of a fresh one-rank communicator on GPU 0. Each test body is
// only this call, so the skip below ends the test.
void withOneRankComm(const std::function<void(ncclComm_t)>& probe) {
  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1) GTEST_SKIP() << "requires at least one GPU";
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  ncclUniqueId id;
  ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
  ncclComm_t comm = nullptr;
  ASSERT_EQ(ncclCommInitRank(&comm, 1, id, 0), ncclSuccess);
  probe(comm);
  EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

} // namespace

// The client above speaks the protocol, so a closed connection in the tests below is a rejection.
TEST(ProxyProtocol, AnswersWellFormedInit) {
  RUN_ISOLATED_TEST("ProxyProtocol.AnswersWellFormedInit", []() {
    withOneRankComm([](ncclComm_t comm) {
      int fd = connectToProxy(comm);
      ASSERT_GE(fd, 0) << "could not connect to the proxy service";
      EXPECT_GE(initConnection(fd), 0) << "proxy did not answer a well-formed Init";
      close(fd);
    });
  });
}

// Init sizes and fields come from the peer: a request larger than ncclProxyInitReq, or a
// tpRank outside the communicator, must be refused.
TEST(ProxyProtocol, RejectsMalformedInit) {
  RUN_ISOLATED_TEST("ProxyProtocol.RejectsMalformedInit", []() {
    withOneRankComm([](ncclComm_t comm) {
      struct { InitReq req; int pad; } oversized = {kValidInit, 0};
      InitReq badRank = kValidInit;
      badRank.tpRank = 1; // the communicator has a single rank
      const struct { const char* what; const void* req; int size; } probes[] = {
        {"a request larger than ncclProxyInitReq", &oversized, (int)sizeof(oversized)},
        {"tpRank out of range", &badRank, (int)sizeof(badRank)}};
      for (const auto& probe : probes) {
        int fd = connectToProxy(comm);
        ASSERT_GE(fd, 0) << "could not connect to the proxy service";
        ASSERT_TRUE(sendOp(fd, ncclProxyMsgInit, -1, probe.req, probe.size, sizeof(InitResp)));
        EXPECT_EQ(awaitReply(fd, kTimeoutMs), Reply::Closed) << "proxy served an Init with " << probe.what;
        close(fd);
      }
    });
  });
}

// A connection belongs to the peer that created it: an op naming another peer's connId must
// be refused, not served with its response written into the owner's socket.
TEST(ProxyProtocol, RejectsOpOnAnotherPeersConnection) {
  RUN_ISOLATED_TEST("ProxyProtocol.RejectsOpOnAnotherPeersConnection", []() {
    withOneRankComm([](ncclComm_t comm) {
      int owner = connectToProxy(comm);
      int intruder = connectToProxy(comm);
      ASSERT_GE(owner, 0) << "could not connect to the proxy service";
      ASSERT_GE(intruder, 0) << "could not connect to the proxy service";
      int connId = initConnection(owner);
      ASSERT_GE(connId, 0) << "proxy did not answer a well-formed Init";
      // A Deregister the P2P transport refuses for its size, so it is harmless if served.
      int junk = 0;
      ASSERT_TRUE(sendOp(intruder, ncclProxyMsgDeregister, connId, &junk, sizeof(junk), 0));
      EXPECT_EQ(awaitReply(intruder, kTimeoutMs), Reply::Closed) << "proxy served an op on another peer's connection";
      EXPECT_EQ(awaitReply(owner, 0), Reply::Silent) << "proxy wrote the intruder's response to the owner";
      close(owner);
      close(intruder);
    });
  });
}

} // namespace RcclUnitTesting
