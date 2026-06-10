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
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

#ifndef RCCL_RAS_CLIENT_BIN
#define RCCL_RAS_CLIENT_BIN "rcclras"
#endif

namespace RcclUnitTesting {

namespace {

static bool isExecutable(const std::string& path) {
  return !path.empty() && ::access(path.c_str(), X_OK) == 0;
}

static std::string dirnameOf(const std::string& path) {
  const auto pos = path.find_last_of('/');
  return pos == std::string::npos ? std::string{} : path.substr(0, pos);
}

static std::string joinPath(const std::string& dir, const char* name) {
  if (dir.empty()) return name;
  return dir.back() == '/' ? dir + name : dir + "/" + name;
}

// Resolves rcclras at runtime. The compile-time RCCL_RAS_CLIENT_BIN path points
// at the cmake build tree, which is absent in TheRock CI test containers that
// only stage artifacts under build/bin (THEROCK_BIN_DIR).
static std::string findRcclrasPath() {
  if (const char* binDir = std::getenv("THEROCK_BIN_DIR")) {
    const std::string candidate = joinPath(binDir, "rcclras");
    if (isExecutable(candidate)) return candidate;
  }

  char exeBuf[4096];
  const ssize_t exeLen = ::readlink("/proc/self/exe", exeBuf, sizeof(exeBuf) - 1);
  if (exeLen > 0) {
    exeBuf[exeLen] = '\0';
    const std::string candidate = joinPath(dirnameOf(exeBuf), "rcclras");
    if (isExecutable(candidate)) return candidate;
  }

  if (isExecutable(RCCL_RAS_CLIENT_BIN)) return RCCL_RAS_CLIENT_BIN;

  return "rcclras";
}

}  // namespace

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

// Runs rcclras via fork+exec and captures stdout. Returns {stdout, exit_code}.
static std::pair<std::string, int> runRcclras(const std::string& format,
                                              const std::string& portStr) {
  static const std::string rcclrasPath = findRcclrasPath();

  int pipefd[2];
  if (pipe(pipefd) != 0) return {"", -1};

  const std::vector<std::string> argStrings = {
      rcclrasPath, "-f", format, "-p", portStr, "-h", "localhost"};
  std::vector<std::vector<char>> argBufs;
  std::vector<char*> argv;
  argBufs.reserve(argStrings.size());
  argv.reserve(argStrings.size() + 1);
  for (const auto& arg : argStrings) {
    argBufs.emplace_back(arg.begin(), arg.end());
    argBufs.back().push_back('\0');
    argv.push_back(argBufs.back().data());
  }
  argv.push_back(nullptr);

  const pid_t pid = fork();
  if (pid < 0) {
    ::close(pipefd[0]);
    ::close(pipefd[1]);
    return {"", -1};
  }
  if (pid == 0) {
    ::close(pipefd[0]);
    if (dup2(pipefd[1], STDOUT_FILENO) < 0) _exit(127);
    ::close(pipefd[1]);
    ::close(STDERR_FILENO);
    execv(rcclrasPath.c_str(), argv.data());
    _exit(127);
  }

  ::close(pipefd[1]);
  std::string output;
  char buf[4096];
  ssize_t n;
  while ((n = ::read(pipefd[0], buf, sizeof(buf))) > 0) output.append(buf, n);
  ::close(pipefd[0]);

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) return {output, -1};
  if (!WIFEXITED(status)) return {output, -1};
  return {output, WEXITSTATUS(status)};
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
        int textRc = -1;
        int jsonRc = -1;
        for (int attempt = 0; attempt < 20; ++attempt) {
          std::tie(textBody, textRc) = runRcclras("text", portStr);
          std::tie(jsonBody, jsonRc) = runRcclras("json", portStr);
          if (textRc == 0 && jsonRc == 0 && !textBody.empty() && !jsonBody.empty()) {
            break;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        const std::string rcclrasPath = findRcclrasPath();
        ASSERT_EQ(textRc, 0) << "rcclras -f text failed (binary: " << rcclrasPath << ")";
        ASSERT_EQ(jsonRc, 0) << "rcclras -f json failed (binary: " << rcclrasPath << ")";
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
