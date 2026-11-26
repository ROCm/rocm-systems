/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only tests for rasRequest() in common/RasClientHelpers.hpp against a fake loopback server. The real RAS
// server always answers, so the connect retry, receive timeout and rejected SET FORMAT paths only run here.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <utility>

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include "common/RasClientHelpers.hpp"

namespace RcclUnitTesting {
namespace {

constexpr int kServerWaitSec = 10;

int listenLoopback(int port) {
  const int s = ::socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) return -1;
  const int one = 1;
  ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || ::listen(s, 1) != 0) {
    ::close(s);
    return -1;
  }
  return s;
}

// Returns one '\n'-terminated line, or "" if the peer closes or the receive times out first.
std::string readLine(int conn) {
  std::string line;
  char c;
  while (::recv(conn, &c, 1, 0) == 1) {
    line += c;
    if (c == '\n') return line;
  }
  return {};
}

std::string readUntilClosed(int conn) {
  std::string data;
  char buf[256];
  ssize_t n;
  while ((n = ::recv(conn, buf, sizeof(buf), 0)) > 0) data.append(buf, n);
  return data;
}

void sendAll(int conn, const std::string& data) { ::send(conn, data.data(), data.size(), MSG_NOSIGNAL); }

// Serves one client on 127.0.0.1:port from a background thread: listens after listenDelay, accepts one connection,
// runs serve on it and closes it. Accept and receives give up after kServerWaitSec, so the destructor cannot hang.
class FakeRasServer {
 public:
  FakeRasServer(int port, std::function<void(int)> serve,
                std::chrono::milliseconds listenDelay = std::chrono::milliseconds(0))
      : thread_([port, serve = std::move(serve), listenDelay] {
          std::this_thread::sleep_for(listenDelay);
          const int listener = listenLoopback(port);
          if (listener < 0) return;
          pollfd pfd{listener, POLLIN, 0};
          if (::poll(&pfd, 1, kServerWaitSec * 1000) == 1) {
            const int conn = ::accept(listener, nullptr, nullptr);
            if (conn >= 0) {
              timeval tv{kServerWaitSec, 0};
              ::setsockopt(conn, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
              serve(conn);
              ::close(conn);
            }
          }
          ::close(listener);
        }) {}
  ~FakeRasServer() { thread_.join(); }

 private:
  std::thread thread_;
};

}  // namespace

TEST(RasClientHelpers, SetsFormatThenSendsCommandAndReturnsReply) {
  const int port = pickFreePort();
  ASSERT_GT(port, 0);
  std::string formatLine, commandLine, body;
  {
    FakeRasServer server(port, [&](int conn) {
      formatLine = readLine(conn);
      sendAll(conn, "OK\n");
      commandLine = readLine(conn);
      sendAll(conn, "{\"communicators\": []}\n");
    });
    body = rasRequest(port, "STATUS", "json");
  }
  EXPECT_EQ(formatLine, "SET FORMAT json\n");
  EXPECT_EQ(commandLine, "STATUS\n");
  EXPECT_EQ(body, "{\"communicators\": []}\n");
}

TEST(RasClientHelpers, RetriesConnectUntilServerListens) {
  const int port = pickFreePort();
  ASSERT_GT(port, 0);
  std::string body;
  {
    // Several 100 ms retry periods pass before the listener exists, so the first connects are refused.
    FakeRasServer server(
        port,
        [](int conn) {
          readLine(conn);
          sendAll(conn, "late reply\n");
        },
        std::chrono::milliseconds(500));
    body = rasRequest(port, "STATUS");
  }
  EXPECT_EQ(body, "late reply\n");
}

TEST(RasClientHelpers, RejectedFormatReturnsEmptyWithoutSendingCommand) {
  const int port = pickFreePort();
  ASSERT_GT(port, 0);
  std::string afterReject, body;
  {
    // Same reply and behavior as the RAS server: it reports the error and keeps the connection open.
    FakeRasServer server(port, [&](int conn) {
      readLine(conn);
      sendAll(conn, "ERROR: Invalid format yaml\n");
      afterReject = readUntilClosed(conn);
    });
    body = rasRequest(port, "STATUS", "yaml");
  }
  EXPECT_EQ(body, "");
  EXPECT_EQ(afterReject, "");
}

TEST(RasClientHelpers, ReceiveTimeoutIsTestFailure) {
  const int port = pickFreePort();
  ASSERT_GT(port, 0);
  std::string commandLine, body;
  {
    FakeRasServer server(port, [&](int conn) {
      commandLine = readLine(conn);
      readUntilClosed(conn);
    });
    EXPECT_NONFATAL_FAILURE(body = rasRequest(port, "DIAGNOSTICS", "", std::chrono::seconds(1)),
                            "recv failed before the server closed the connection");
  }
  EXPECT_EQ(commandLine, "DIAGNOSTICS\n");
  EXPECT_EQ(body, "");
}

}  // namespace RcclUnitTesting
