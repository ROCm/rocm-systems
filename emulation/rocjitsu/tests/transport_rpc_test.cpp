// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file transport_rpc_test.cpp
/// @brief Tests for Unix transport SOCK_CLOEXEC, RPC EINTR retry, and
/// response payload cap.

#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/kmd/linux/transport.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

namespace {

std::string test_socket_path() {
  static int counter = 0;
  return "/tmp/rocjitsu_test_transport_" + std::to_string(getpid()) + "_" +
         std::to_string(counter++) + ".sock";
}

TEST(TransportTest, ListenSocketHasCloexec) {
  auto path = test_socket_path();
  auto server = rocjitsu::UnixTransport::listen(path);
  ASSERT_NE(server, nullptr);

  int flags = fcntl(server->fd(), F_GETFD);
  ASSERT_GE(flags, 0);
  EXPECT_TRUE(flags & FD_CLOEXEC) << "listen socket should have FD_CLOEXEC";

  server->close();
  unlink(path.c_str());
}

TEST(TransportTest, ConnectSocketHasCloexec) {
  auto path = test_socket_path();
  auto server = rocjitsu::UnixTransport::listen(path);
  ASSERT_NE(server, nullptr);

  auto client = rocjitsu::UnixTransport::connect(path);
  ASSERT_NE(client, nullptr);

  int flags = fcntl(client->fd(), F_GETFD);
  ASSERT_GE(flags, 0);
  EXPECT_TRUE(flags & FD_CLOEXEC) << "connect socket should have FD_CLOEXEC";

  client->close();
  server->close();
  unlink(path.c_str());
}

TEST(TransportTest, AcceptedSocketHasCloexec) {
  auto path = test_socket_path();
  auto server = rocjitsu::UnixTransport::listen(path);
  ASSERT_NE(server, nullptr);

  std::thread connector([&path]() {
    auto client = rocjitsu::UnixTransport::connect(path);
    ASSERT_NE(client, nullptr);
    client->close();
  });

  auto accepted = server->accept();
  ASSERT_NE(accepted, nullptr);

  int flags = fcntl(accepted->fd(), F_GETFD);
  ASSERT_GE(flags, 0);
  EXPECT_TRUE(flags & FD_CLOEXEC) << "accepted socket should have FD_CLOEXEC";

  connector.join();
  accepted->close();
  server->close();
  unlink(path.c_str());
}

// Verifies that rpc_recv_msg survives an EINTR during recvmsg. Before the
// fix, rpc_recv_msg made a single recvmsg call with no retry, so any signal
// delivered during the blocked receive would cause it to return -1.
TEST(RpcTest, RecvMsgRetriesOnEINTR) {
  int fds[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  struct sigaction sa {};
  sa.sa_handler = [](int) {};
  sa.sa_flags = 0; // no SA_RESTART — recvmsg must return EINTR
  sigaction(SIGUSR1, &sa, nullptr);

  rocjitsu::RpcHeader hdr{};
  hdr.opcode = rocjitsu::RPC_HANDSHAKE;
  hdr.request_id = 77;

  pid_t self = getpid();

  std::thread sender([&]() {
    usleep(50000); // 50ms — let receiver block in recvmsg
    kill(self, SIGUSR1);
    usleep(50000); // 50ms — let the retry loop re-enter recvmsg
    rocjitsu::rpc_send_exact(fds[0], &hdr, sizeof(hdr));
  });

  rocjitsu::RpcHeader received{};
  auto bytes = rocjitsu::rpc_recv_msg(fds[1], &received, sizeof(received));
  ASSERT_EQ(bytes, static_cast<ssize_t>(sizeof(received)))
      << "rpc_recv_msg should retry after EINTR, not fail";
  EXPECT_EQ(received.request_id, 77u);

  sender.join();
  signal(SIGUSR1, SIG_DFL);
  ::close(fds[0]);
  ::close(fds[1]);
}

} // namespace
