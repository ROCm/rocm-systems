/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

// Keep in sync with src/ras/ras_internal.h
#define NCCL_RAS_CLIENT_PORT 28028

namespace RcclUnitTesting {

// Connects to the local RAS server and runs one STATUS query in the requested
// format ("text" or "json"). Returns the entire response body, or an empty
// string on connection/protocol failure.
static std::string queryRas(const std::string& format) {
  int sock = -1;
  for (int attempt = 0; attempt < 20; ++attempt) {
    sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return {};
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(NCCL_RAS_CLIENT_PORT);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) break;
    ::close(sock);
    sock = -1;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (sock < 0) return {};

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

  const char* status = "STATUS\n";
  ::send(sock, status, ::strlen(status), 0);

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

        ncclUniqueId id;
        ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
        ncclComm_t comm = nullptr;
        ASSERT_EQ(ncclCommInitRank(&comm, /*nranks=*/1, id, /*rank=*/0),
                  ncclSuccess);

        // Give the RAS listener a brief moment to come up after init.
        std::this_thread::sleep_for(std::chrono::milliseconds(250));

        std::string textBody = queryRas("text");
        std::string jsonBody = queryRas("json");

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

}  // namespace RcclUnitTesting
