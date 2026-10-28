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
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {

// Picks a likely-free TCP port by binding to port 0 and reading the kernel
// assignment, then closes the probe socket. Caller must use SO_REUSEADDR (the
// RAS listener does) or accept a small race window before the chosen port is
// reused by something else on the host.
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

// Connects to the local RAS server and runs one query (STATUS by default) in the
// requested format ("text" or "json"). Returns the entire response body, or an empty
// string on connection/protocol failure. Uses 127.0.0.1 (IPv4 loopback) to
// match the address family used by pickFreePort() and NCCL_RAS_ADDR.
static std::string queryRas(const std::string& format, const std::string& portStrIn,
                            const std::string& command = "STATUS") {
  int sock = -1;
  addrinfo hints{};
  hints.ai_family   = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  const char* portStr = portStrIn.c_str();

  for (int attempt = 0; attempt < 20 && sock < 0; ++attempt) {
    addrinfo* results = nullptr;
    if (::getaddrinfo("127.0.0.1", portStr, &hints, &results) == 0) {
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
  // Bound each receive: a stuck server then fails the test with the reply so far.
  timeval recvTimeout{60, 0};
  ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &recvTimeout, sizeof(recvTimeout));

  if (format != "text") {
    std::string setFmt = "SET FORMAT " + format + "\n";
    ::send(sock, setFmt.c_str(), setFmt.size(), 0);
    char ack[16] = {};
    ssize_t n = ::recv(sock, ack, sizeof(ack) - 1, 0);
    if (n <= 0 || std::string(ack, n).find("OK") == std::string::npos) {
      ::close(sock);
      return {};
    }
  }

  const std::string request = command + "\n";
  ::send(sock, request.c_str(), request.size(), 0);

  std::string body;
  char buf[4096];
  ssize_t n;
  while ((n = ::recv(sock, buf, sizeof(buf), 0)) > 0) body.append(buf, n);
  ::close(sock);
  return body;
}

// Verifies the RAS subsystem supports JSON output and that switching to JSON
// actually returns a structured document rather than the human-readable text
// banner.
TEST(RasJson, JsonFormatIsSupportedAndDistinctFromText) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "RasJson_JsonFormatIsSupportedAndDistinctFromText",
      []() {
        int devCount = 0;
        if (hipGetDeviceCount(&devCount) != hipSuccess || devCount < 1) {
          GTEST_SKIP() << "No HIP-visible GPU; skipping";
        }
        ASSERT_EQ(hipSetDevice(0), hipSuccess);

        // Pick the RAS port and publish NCCL_RAS_ADDR *inside* this process,
        // which is the same process that runs ncclCommInitRank() (so the RAS
        // server binds here) and queryRas() (so the client connects here).
        // The process-isolation harness re-execs the test binary, which would
        // re-run any port selection placed in the TEST() body in a *second*
        // process; the server and client would then disagree on the port and
        // every connect() would get ECONNREFUSED.  Owning the port here keeps
        // them in lockstep.
        //
        // Use 127.0.0.1 explicitly (not "localhost") so the RAS server binds on
        // the same IPv4 address pickFreePort() probed: on hosts where
        // getaddrinfo("localhost", AF_UNSPEC) returns ::1 first, the bind would
        // otherwise target an IPv6 port that was never checked for being free.
        int port = pickFreePort();
        ASSERT_GT(port, 0) << "Could not allocate a free TCP port for RAS";
        std::string portStr = std::to_string(port);
        std::string rasAddr = "127.0.0.1:" + portStr;
        ASSERT_EQ(setenv("NCCL_RAS_ENABLE", "1", /*overwrite=*/1), 0);
        ASSERT_EQ(setenv("NCCL_RAS_ADDR", rasAddr.c_str(), /*overwrite=*/1), 0);

        ncclUniqueId id;
        ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
        ncclComm_t comm = nullptr;
        ASSERT_EQ(ncclCommInitRank(&comm, /*nranks=*/1, id, /*rank=*/0),
                  ncclSuccess);

        // Give the RAS listener a brief moment to come up after init.
        std::this_thread::sleep_for(std::chrono::milliseconds(250));

        std::string textBody = queryRas("text", portStr);
        std::string jsonBody = queryRas("json", portStr);

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
      {{"NCCL_RAS_ENABLE", "1"}});
}

// Up to two GPUs, so that each rank has a peer to classify where possible.
static int diagnosticsGpus() {
  int devCount = 0;
  if (hipGetDeviceCount(&devCount) != hipSuccess) return 0;
  return std::min(devCount, 2);
}

// Creates one communicator over the first nGpus GPUs of this process, with its
// RAS server on a free port (picked in the isolated child, for the reason given
// above), and returns the server's reply to the DIAGNOSTICS command.
static std::string queryRasDiagnostics(int nGpus) {
  int port = pickFreePort();
  std::string portStr = std::to_string(port);
  std::string rasAddr = "127.0.0.1:" + portStr;
  std::vector<ncclComm_t> comms(nGpus, nullptr);
  if (port <= 0 || setenv("NCCL_RAS_ADDR", rasAddr.c_str(), /*overwrite=*/1) != 0 ||
      ncclCommInitAll(comms.data(), nGpus, /*devlist=*/nullptr) != ncclSuccess) {
    ADD_FAILURE() << "Could not create a " << nGpus << "-rank communicator with a RAS server";
    return {};
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  std::string reply = queryRas("text", portStr, "DIAGNOSTICS");
  for (ncclComm_t comm : comms) EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
  return reply;
}

// NCCL 2.32 adds a Paths check to the RAS diagnostics. Each rank classifies how
// it reaches every peer in the communicator's topology, and the report lists the
// union of those path types, "self" for the rank itself (e.g. "self+XGMI").
TEST(RasDiagnostics, ReportsCommunicatorPathTypes) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "RasDiagnostics_ReportsCommunicatorPathTypes",
      []() {
        int nGpus = diagnosticsGpus();
        if (nGpus < 1) GTEST_SKIP() << "No HIP-visible GPU; skipping";
        std::string reply = queryRasDiagnostics(nGpus);
        size_t pos = reply.find(nGpus > 1 ? "Paths: self+" : "Paths: self across");
        ASSERT_NE(pos, std::string::npos) << "RAS diagnostics reported no path types:\n" << reply;
        std::string line = reply.substr(pos, reply.find('\n', pos) - pos);
        std::string scope = " across " + std::to_string(nGpus) + " ranks in comm 0x";
        EXPECT_NE(line.find(scope), std::string::npos) << line;
      },
      {{"NCCL_RAS_ENABLE", "1"}});
}

// Control: the same DIAGNOSTICS request completes without a failed summary and
// reports the NCCL environment check, which predates the Paths check.
TEST(RasDiagnostics, ReportsNcclEnvironment) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "RasDiagnostics_ReportsNcclEnvironment",
      []() {
        int nGpus = diagnosticsGpus();
        if (nGpus < 1) GTEST_SKIP() << "No HIP-visible GPU; skipping";
        std::string reply = queryRasDiagnostics(nGpus);
        EXPECT_NE(reply.find("NCCL environment: "), std::string::npos) << reply;
        EXPECT_NE(reply.find("RAS diagnostics completed"), std::string::npos) << reply;
        EXPECT_EQ(reply.find("ERROR: diagnostics summary failed"), std::string::npos) << reply;
      },
      {{"NCCL_RAS_ENABLE", "1"}});
}

}  // namespace RcclUnitTesting
