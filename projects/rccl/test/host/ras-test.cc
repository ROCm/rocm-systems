/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the message and poll-state primitives in
// src/ras/ras.cc. This intentionally excludes connection handshakes, live
// sockets, communicator lifecycle, and the RAS thread; those paths have a much
// larger dependency surface and belong in later reviewable batches. Every
// selected non-inline helper has full line coverage.

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <functional>

#include "fakes/signature-drift.h"
#include "socket.h"

namespace {

ncclResult_t DefaultSocketProgress(int, struct ncclSocket*, void*, int size, int* offset, int* closed) {
  *offset = size;
  if (closed) *closed = 0;
  return ncclSuccess;
}

std::function<ncclResult_t(int, struct ncclSocket*, void*, int, int*, int*)> g_socketProgress =
    DefaultSocketProgress;

}  // namespace

ASSERT_HOOK_MATCHES_PROD(g_socketProgress, ncclSocketProgress);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* closed) {
  return g_socketProgress(op, sock, ptr, size, offset, closed);
}

const char* ncclSocketToString(const union ncclSocketAddress*, char* buf, const int) {
  buf[0] = '\0';
  return buf;
}

#include RAS_CC_PATH

namespace {

class RasMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_socketProgress = DefaultSocketProgress;
    std::free(rasPfds);
    rasPfds = nullptr;
    nRasPfds = 0;
  }

  void TearDown() override {
    std::free(rasPfds);
    rasPfds = nullptr;
    nRasPfds = 0;
    g_socketProgress = DefaultSocketProgress;
  }
};

struct OwnedMsg {
  rasMsg* ptr = nullptr;
  explicit OwnedMsg(size_t len) { EXPECT_EQ(ncclSuccess, rasMsgAlloc(&ptr, len)); }
  ~OwnedMsg() { rasMsgFree(ptr); }
  rasMsg* release() {
    rasMsg* out = ptr;
    ptr = nullptr;
    return out;
  }
};

}  // namespace

TEST_F(RasMicrotest, MessageLengthsCoverEveryFixedAndCollectiveType) {
  EXPECT_EQ(offsetof(rasMsg, connInit) + sizeof(rasMsg{}.connInit), rasMsgLength(RAS_MSG_CONNINIT));
  EXPECT_EQ(offsetof(rasMsg, connInitAck) + sizeof(rasMsg{}.connInitAck), rasMsgLength(RAS_MSG_CONNINITACK));
  EXPECT_EQ(offsetof(rasMsg, keepAlive) + sizeof(rasMsg{}.keepAlive), rasMsgLength(RAS_MSG_KEEPALIVE));
  EXPECT_EQ(offsetof(rasMsg, peersUpdate) + sizeof(rasMsg{}.peersUpdate), rasMsgLength(RAS_MSG_PEERSUPDATE));
  EXPECT_EQ(offsetof(rasMsg, collResp) + sizeof(rasMsg{}.collResp), rasMsgLength(RAS_MSG_COLLRESP));
  EXPECT_EQ(offsetof(rasMsg, collReq) + rasCollDataLength(RAS_BC_DEADPEER),
            rasMsgLength(RAS_MSG_COLLREQ, RAS_BC_DEADPEER));
  EXPECT_EQ(offsetof(rasCollRequest, conns) + sizeof(rasCollRequest{}.conns), rasCollDataLength(RAS_COLL_CONNS));
  EXPECT_EQ(offsetof(rasMsg, collReq) + rasCollDataLength(RAS_COLL_CONNS),
            rasMsgLength(RAS_MSG_COLLREQ, RAS_COLL_CONNS));
  EXPECT_EQ(offsetof(rasCollRequest, comms) + sizeof(rasCollRequest{}.comms), rasCollDataLength(RAS_COLL_COMMS));
  EXPECT_EQ(offsetof(rasMsg, collReq) + rasCollDataLength(RAS_COLL_COMMS),
            rasMsgLength(RAS_MSG_COLLREQ, RAS_COLL_COMMS));
  EXPECT_EQ(0u, rasCollDataLength(RAS_MSG_NONE));
  EXPECT_EQ(0u, rasCollDataLength(static_cast<rasCollectiveType>(-1)));
  EXPECT_EQ(0u, rasMsgLength(static_cast<rasMsgType>(0)));
  EXPECT_EQ(0u, rasMsgLength(static_cast<rasMsgType>(-1)));
}

TEST_F(RasMicrotest, MsgAllocReturnsZeroedPayloadAndFreeAcceptsNull) {
  rasMsg* msg = nullptr;
  ASSERT_EQ(ncclSuccess, rasMsgAlloc(&msg, rasMsgLength(RAS_MSG_KEEPALIVE)));
  ASSERT_NE(nullptr, msg);
  EXPECT_EQ(RAS_MSG_NONE, msg->type);
  rasMsgFree(msg);
  rasMsgFree(nullptr);
}

TEST_F(RasMicrotest, ConnEnqueueBackInitializesMetadataAndArmsReadySocket) {
  rasPfds = static_cast<pollfd*>(std::calloc(1, sizeof(*rasPfds)));
  nRasPfds = 1;
  rasSocket sock{};
  sock.status = RAS_SOCK_READY;
  sock.pfd = 0;
  rasConnection conn{};
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_KEEPALIVE));
  owned.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_KEEPALIVE), false);

  rasMsgMeta* meta = ncclIntruQueueHead(&conn.sendQ);
  ASSERT_NE(nullptr, meta);
  EXPECT_EQ(0, meta->offset);
  EXPECT_EQ((int)rasMsgLength(RAS_MSG_KEEPALIVE), meta->length);
  EXPECT_NE(0, rasPfds[0].events & POLLOUT);
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnEnqueueFrontPrecedesExistingMessage) {
  rasConnection conn{};
  OwnedMsg first(rasMsgLength(RAS_MSG_KEEPALIVE));
  OwnedMsg second(rasMsgLength(RAS_MSG_CONNINIT));
  first.ptr->type = RAS_MSG_KEEPALIVE;
  second.ptr->type = RAS_MSG_CONNINIT;
  rasMsg* firstRaw = first.release();
  rasMsg* secondRaw = second.release();
  rasConnEnqueueMsg(&conn, firstRaw, rasMsgLength(RAS_MSG_KEEPALIVE), false);
  rasConnEnqueueMsg(&conn, secondRaw, rasMsgLength(RAS_MSG_CONNINIT), true);
  EXPECT_EQ(secondRaw, &ncclIntruQueueHead(&conn.sendQ)->msg);
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnEnqueueHandshakeArmsOnlyConnInitMessage) {
  rasPfds = static_cast<pollfd*>(std::calloc(1, sizeof(*rasPfds)));
  nRasPfds = 1;
  rasSocket sock{};
  sock.status = RAS_SOCK_HANDSHAKE;
  sock.pfd = 0;
  rasConnection conn{};
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_CONNINIT));
  owned.ptr->type = RAS_MSG_CONNINIT;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_CONNINIT));
  EXPECT_NE(0, rasPfds[0].events & POLLOUT);
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);

  rasPfds[0].events = 0;
  OwnedMsg blocked(rasMsgLength(RAS_MSG_KEEPALIVE));
  blocked.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, blocked.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  EXPECT_EQ(0, rasPfds[0].events & POLLOUT);
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnSendEmptyQueueReportsAllSent) {
  rasConnection conn{};
  rasSocket sock{};
  conn.sock = &sock;
  int closed = -1;
  bool allSent = false;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(0, closed);
  EXPECT_TRUE(allSent);
}

TEST_F(RasMicrotest, ConnSendHandshakeBlocksNonInitMessageWithoutCallingSocket) {
  rasConnection conn{};
  rasSocket sock{};
  sock.status = RAS_SOCK_HANDSHAKE;
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_KEEPALIVE));
  owned.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  int calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void*, int, int*, int*) {
    ++calls;
    return ncclSuccess;
  };
  int closed;
  bool allSent = false;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(0, calls);
  EXPECT_TRUE(allSent);
  EXPECT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnSendCompleteMessageDequeuesIt) {
  rasConnection conn{};
  rasSocket sock{};
  sock.status = RAS_SOCK_CLOSED; // enqueue need not arm rasPfds; send itself accepts this state
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_KEEPALIVE));
  owned.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  int calls = 0;
  g_socketProgress = [&](int op, ncclSocket*, void*, int size, int* offset, int* closed) {
    EXPECT_EQ(NCCL_SOCKET_SEND, op);
    ++calls;
    *offset = size;
    *closed = 0;
    return ncclSuccess;
  };
  int closed;
  bool allSent = false;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(2, calls);
  EXPECT_TRUE(allSent);
  EXPECT_TRUE(ncclIntruQueueEmpty(&conn.sendQ));
}

TEST_F(RasMicrotest, ConnSendPartialLengthKeepsMessageQueued) {
  rasConnection conn{};
  rasSocket sock{};
  sock.status = RAS_SOCK_CLOSED;
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_KEEPALIVE));
  owned.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  g_socketProgress = [](int, ncclSocket*, void*, int, int* offset, int* closed) {
    *offset = 2;
    *closed = 0;
    return ncclSuccess;
  };
  int closed;
  bool allSent = true;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_FALSE(allSent);
  EXPECT_EQ(2, ncclIntruQueueHead(&conn.sendQ)->offset);
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnSendClosedSocketReturnsWithoutDequeuing) {
  rasConnection conn{};
  rasSocket sock{};
  sock.status = RAS_SOCK_CLOSED;
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_KEEPALIVE));
  owned.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  g_socketProgress = [](int, ncclSocket*, void*, int, int*, int* closed) {
    *closed = 1;
    return ncclSuccess;
  };
  int closed = 0;
  bool allSent = false;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(1, closed);
  EXPECT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnSendPartialBodyKeepsMessageQueued) {
  rasConnection conn{};
  rasSocket sock{};
  sock.status = RAS_SOCK_CLOSED;
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_KEEPALIVE));
  owned.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  int calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void*, int size, int* offset, int* closed) {
    ++calls;
    *offset = calls == 1 ? size : size - 1;
    *closed = 0;
    return ncclSuccess;
  };
  int closed;
  bool allSent = true;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(2, calls);
  EXPECT_FALSE(allSent);
  EXPECT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnSendSocketErrorPropagates) {
  rasConnection conn{};
  rasSocket sock{};
  sock.status = RAS_SOCK_CLOSED;
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_KEEPALIVE));
  owned.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  g_socketProgress = [](int, ncclSocket*, void*, int, int*, int*) { return ncclSystemError; };
  int closed;
  bool allSent;
  EXPECT_EQ(ncclSystemError, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, MsgRecvCompletesLengthThenBodyAndResetsSocketState) {
  rasSocket sock{};
  const int msgLen = rasMsgLength(RAS_MSG_KEEPALIVE);
  int calls = 0;
  g_socketProgress = [&](int op, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    EXPECT_EQ(NCCL_SOCKET_RECV, op);
    ++calls;
    if (calls == 1) {
      *static_cast<int*>(ptr) = msgLen;
    } else {
      auto* msg = reinterpret_cast<rasMsg*>(static_cast<char*>(ptr) + sizeof(int));
      msg->type = RAS_MSG_KEEPALIVE;
    }
    *offset = size;
    *closed = 0;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed;
  ASSERT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  ASSERT_NE(nullptr, msg);
  EXPECT_EQ(RAS_MSG_KEEPALIVE, msg->type);
  EXPECT_EQ(2, calls);
  EXPECT_EQ(0, sock.recvOffset);
  EXPECT_EQ(0, sock.recvLength);
  EXPECT_EQ(nullptr, sock.recvMsg);
  std::free(msg);
}

TEST_F(RasMicrotest, MsgRecvPartialLengthReturnsWithoutAllocatingBody) {
  rasSocket sock{};
  g_socketProgress = [](int, ncclSocket*, void*, int, int* offset, int* closed) {
    *offset = 2;
    *closed = 0;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed;
  EXPECT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(nullptr, msg);
  EXPECT_EQ(nullptr, sock.recvMsg);
  EXPECT_EQ(2, sock.recvOffset);
}

TEST_F(RasMicrotest, MsgRecvClosedDuringLengthReturnsImmediately) {
  rasSocket sock{};
  g_socketProgress = [](int, ncclSocket*, void*, int, int*, int* closed) {
    *closed = 1;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed = 0;
  EXPECT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(1, closed);
  EXPECT_EQ(nullptr, msg);
  EXPECT_EQ(nullptr, sock.recvMsg);
}

TEST_F(RasMicrotest, MsgRecvSocketErrorPropagates) {
  rasSocket sock{};
  g_socketProgress = [](int, ncclSocket*, void*, int, int*, int*) { return ncclSystemError; };
  rasMsg* msg = nullptr;
  int closed;
  EXPECT_EQ(ncclSystemError, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(nullptr, msg);
  EXPECT_EQ(nullptr, sock.recvMsg);
}

TEST_F(RasMicrotest, MsgRecvClosedDuringBodyPreservesAllocatedMessage) {
  rasSocket sock{};
  const int msgLen = rasMsgLength(RAS_MSG_KEEPALIVE);
  int calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    ++calls;
    if (calls == 1) {
      *static_cast<int*>(ptr) = msgLen;
      *offset = size;
      *closed = 0;
    } else {
      *closed = 1;
    }
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed = 0;
  EXPECT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(1, closed);
  EXPECT_EQ(nullptr, msg);
  ASSERT_NE(nullptr, sock.recvMsg);
  std::free(sock.recvMsg);
  sock.recvMsg = nullptr;
}

TEST_F(RasMicrotest, MsgRecvPartialBodyPreservesProgressForNextCall) {
  rasSocket sock{};
  const int msgLen = rasMsgLength(RAS_MSG_KEEPALIVE);
  int calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    ++calls;
    if (calls == 1) *static_cast<int*>(ptr) = msgLen;
    *offset = calls == 1 ? size : size - 1;
    *closed = 0;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed;
  EXPECT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(nullptr, msg);
  ASSERT_NE(nullptr, sock.recvMsg);
  EXPECT_EQ(msgLen + (int)sizeof(int) - 1, sock.recvOffset);
  std::free(sock.recvMsg);
  sock.recvMsg = nullptr;
}

TEST_F(RasMicrotest, MsgRecvResumesPartialBodyWithoutRereadingLength) {
  rasSocket sock{};
  sock.recvLength = rasMsgLength(RAS_MSG_KEEPALIVE);
  sock.recvOffset = sizeof(sock.recvLength) + 3;
  sock.recvMsg = static_cast<rasMsg*>(std::calloc(1, sock.recvLength));
  int calls = 0;
  g_socketProgress = [&](int op, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    EXPECT_EQ(NCCL_SOCKET_RECV, op);
    EXPECT_EQ(sock.recvLength + (int)sizeof(sock.recvLength), size);
    ++calls;
    auto* msg = reinterpret_cast<rasMsg*>(static_cast<char*>(ptr) + sizeof(sock.recvLength));
    msg->type = RAS_MSG_KEEPALIVE;
    *offset = size;
    *closed = 0;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed;
  ASSERT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  ASSERT_NE(nullptr, msg);
  EXPECT_EQ(RAS_MSG_KEEPALIVE, msg->type);
  EXPECT_EQ(1, calls);
  EXPECT_EQ(0, sock.recvOffset);
  EXPECT_EQ(0, sock.recvLength);
  EXPECT_EQ(nullptr, sock.recvMsg);
  std::free(msg);
}

TEST_F(RasMicrotest, GetNewPollEntryGrowsInChunksAndReusesVacancies) {
  int first = -1;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&first));
  EXPECT_EQ(0, first);
  EXPECT_EQ(RAS_INCREMENT, nRasPfds);
  EXPECT_EQ(NCCL_INVALID_SOCKET, rasPfds[first].fd);

  rasPfds[0].fd = 7;
  int second = -1;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&second));
  EXPECT_EQ(1, second);
  rasPfds[0].fd = NCCL_INVALID_SOCKET;
  int reused = -1;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&reused));
  EXPECT_EQ(0, reused);
}

TEST_F(RasMicrotest, GetNewPollEntryExpandsAgainWhenEverySlotIsOccupied) {
  int index;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&index));
  ASSERT_EQ(RAS_INCREMENT, nRasPfds);
  for (int i = 0; i < nRasPfds; ++i) rasPfds[i].fd = i + 10;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&index));
  EXPECT_EQ(RAS_INCREMENT, index);
  EXPECT_EQ(2 * RAS_INCREMENT, nRasPfds);
  EXPECT_EQ(NCCL_INVALID_SOCKET, rasPfds[index].fd);
  EXPECT_EQ(0, rasPfds[index].events);
  EXPECT_EQ(0, rasPfds[index].revents);
}
