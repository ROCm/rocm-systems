/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {

// Picks a likely-free TCP port by binding to port 0 and reading the kernel
// assignment, then closes the probe socket. The RAS listener uses SO_REUSEADDR,
// so the brief window before the chosen port is reused is acceptable.
static int pickFreePort() {
  int s = ::socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) return 0;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(s);
    return 0;
  }
  socklen_t len = sizeof(addr);
  if (::getsockname(s, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
    ::close(s);
    return 0;
  }
  int port = ntohs(addr.sin_port);
  ::close(s);
  return port;
}

// Reads a single '\n'-terminated line from the socket. Byte-at-a-time is fine
// here: the handshake/ack replies are tiny and sent one per command.
static bool recvLine(int sock, std::string& line) {
  line.clear();
  char c;
  for (;;) {
    ssize_t n = ::recv(sock, &c, 1, 0);
    if (n <= 0) return false;
    line.push_back(c);
    if (c == '\n') return true;
  }
}

// Connects to the local RAS server on the given port, performs the client
// handshake, switches output format, runs one STATUS query, and returns the
// full response body (empty string on connection/protocol failure).
//
// This speaks the RAS wire protocol directly (the same exchange rcclras does:
// CLIENT PROTOCOL -> SET FORMAT -> STATUS), so the test has no dependency on
// the external rcclras client binary being staged next to the unit tests.
static std::string queryRas(const std::string& portStr, const std::string& format) {
  int sock = -1;
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  for (int attempt = 0; attempt < 50 && sock < 0; ++attempt) {
    addrinfo* results = nullptr;
    if (::getaddrinfo("localhost", portStr.c_str(), &hints, &results) == 0) {
      for (addrinfo* ai = results; ai != nullptr; ai = ai->ai_next) {
        int s = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (s < 0) continue;
        if (::connect(s, ai->ai_addr, ai->ai_addrlen) == 0) {
          sock = s;
          break;
        }
        ::close(s);
      }
      ::freeaddrinfo(results);
    }
    if (sock < 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (sock < 0) return {};

  // Handshake. The server echoes its own version and ignores ours, so any
  // "SERVER PROTOCOL " reply is acceptable.
  std::string line;
  const std::string hello = "CLIENT PROTOCOL 2\n";
  if (::send(sock, hello.data(), hello.size(), 0) != static_cast<ssize_t>(hello.size()) ||
      !recvLine(sock, line) ||
      line.rfind("SERVER PROTOCOL ", 0) != 0) {
    ::close(sock);
    return {};
  }

  // Select output format (text or json); the server replies "OK\n".
  const std::string setFmt = "SET FORMAT " + format + "\n";
  if (::send(sock, setFmt.data(), setFmt.size(), 0) != static_cast<ssize_t>(setFmt.size()) ||
      !recvLine(sock, line) || line != "OK\n") {
    ::close(sock);
    return {};
  }

  // Request status; the server streams the body and then closes the socket.
  const char* status = "STATUS\n";
  if (::send(sock, status, ::strlen(status), 0) != static_cast<ssize_t>(::strlen(status))) {
    ::close(sock);
    return {};
  }

  std::string body;
  char buf[4096];
  for (;;) {
    ssize_t n = ::recv(sock, buf, sizeof(buf), 0);
    if (n > 0) {
      body.append(buf, n);
    } else if (n == 0) {
      break;  // Clean EOF: the server finished the body and closed the socket.
    } else {
      // recv() error: don't pass off a partial body as a complete response.
      ::close(sock);
      return {};
    }
  }
  ::close(sock);
  return body;
}

// Verifies the RAS subsystem supports JSON output and that switching to JSON
// actually returns a structured document rather than the human-readable text
// banner.
TEST(RasJson, JsonFormatIsSupportedAndDistinctFromText) {
  // Pick a free port in the parent so the isolated child uses an address that
  // doesn't collide with stale/other RCCL processes on the host (the RAS
  // default port 28028 is shared and may already be in use).
  int port = pickFreePort();
  ASSERT_GT(port, 0) << "Could not allocate a free TCP port for RAS";
  std::string portStr = std::to_string(port);
  std::string rasAddr = "localhost:" + portStr;

  RUN_ISOLATED_TEST_WITH_ENV(
      "RasJson_JsonFormatIsSupportedAndDistinctFromText",
      [portStr]() {
        int devCount = 0;
        if (hipGetDeviceCount(&devCount) != hipSuccess || devCount < 1) {
          GTEST_SKIP() << "No HIP-visible GPU; skipping";
        }
        ASSERT_EQ(hipSetDevice(0), hipSuccess);

        ncclUniqueId id;
        ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
        ncclComm_t comm = nullptr;
        ASSERT_EQ(ncclCommInitRank(&comm, /*nranks=*/1, id, /*rank=*/0),
                  ncclSuccess);

        // Give the RAS listener a brief moment to come up after init.
        std::this_thread::sleep_for(std::chrono::milliseconds(250));

        std::string textBody;
        std::string jsonBody;
        for (int attempt = 0; attempt < 20; ++attempt) {
          textBody = queryRas(portStr, "text");
          jsonBody = queryRas(portStr, "json");
          if (!textBody.empty() && !jsonBody.empty()) break;
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        ASSERT_FALSE(textBody.empty()) << "RAS returned no text response";
        ASSERT_FALSE(jsonBody.empty())
            << "RAS did not respond to 'SET FORMAT json' -- JSON support missing";

        EXPECT_NE(textBody.find("RCCL version"), std::string::npos)
            << "Text mode should contain the RCCL version banner";

        EXPECT_EQ(jsonBody.front(), '{')
            << "JSON mode should return a JSON object, not text. Got:\n"
            << jsonBody.substr(0, 200);
        EXPECT_NE(jsonBody.find("\"communicators\""), std::string::npos)
            << "JSON output missing the 'communicators' field";
        EXPECT_EQ(jsonBody.find("RCCL version"), std::string::npos)
            << "JSON output must not contain the human-readable text banner";

        ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);
      },
      {{"NCCL_RAS_ENABLE", "1"}, {"NCCL_RAS_ADDR", rasAddr}});
}

}  // namespace RcclUnitTesting
