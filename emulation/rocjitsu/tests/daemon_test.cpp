// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file daemon_test.cpp
/// @brief Runs HIP kernel tests through the rocjitsu daemon via RPC.
///
/// Each test starts a daemon process, runs the HIP test binary with
/// LD_PRELOAD as a subprocess, verifies the result, and tears down the
/// daemon. Paths are injected via CMake compile definitions.

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

struct ProcessResult {
  std::string output;
  int exit_code = -1;
};

bool socket_exists(const std::string &path) {
  struct stat st{};
  return stat(path.c_str(), &st) == 0 && S_ISSOCK(st.st_mode);
}

class DaemonTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Each test gets its own unique subdirectory under XDG_RUNTIME_DIR
    // for the daemon socket, preventing conflicts when tests run in parallel.
    const char *xdg = std::getenv("XDG_RUNTIME_DIR");
    std::string base = xdg ? xdg : "/tmp";
    std::string tmpl = base + "/rocjitsu-test-XXXXXX";
    ASSERT_NE(mkdtemp(tmpl.data()), nullptr) << "mkdtemp failed: " << strerror(errno);
    tmp_dir_ = tmpl;
    sock_path_ = tmp_dir_ + "/rocjitsu/daemon.sock";

    daemon_pid_ = fork();
    ASSERT_GE(daemon_pid_, 0) << "fork failed: " << strerror(errno);

    if (daemon_pid_ == 0) {
      setenv("XDG_RUNTIME_DIR", tmp_dir_.c_str(), 1);
      execl(RJ_DAEMON_BIN, RJ_DAEMON_BIN, "--daemon", "--config", RJ_DAEMON_CONFIG, nullptr);
      _exit(127);
    }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
      if (socket_exists(sock_path_))
        return;
      int status = 0;
      if (waitpid(daemon_pid_, &status, WNOHANG) > 0) {
        daemon_pid_ = -1;
        FAIL() << "daemon exited before creating socket";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    FAIL() << "daemon socket not created after 30s";
  }

  void TearDown() override {
    if (daemon_pid_ > 0) {
      kill(daemon_pid_, SIGTERM);
      int status = 0;
      for (int i = 0; i < 20; ++i) {
        if (waitpid(daemon_pid_, &status, WNOHANG) > 0)
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      kill(daemon_pid_, SIGKILL);
      waitpid(daemon_pid_, &status, 0);
      daemon_pid_ = -1;
    }
    unlink(sock_path_.c_str());
    std::string sock_dir = tmp_dir_ + "/rocjitsu";
    rmdir(sock_dir.c_str());
    rmdir(tmp_dir_.c_str());
  }

  ProcessResult run_hip_test(const char *binary, const char *gtest_filter) {
    std::string cmd = "XDG_RUNTIME_DIR=";
    cmd += tmp_dir_;
    cmd += " LD_PRELOAD=";
    cmd += RJ_PRELOAD_LIB;
    cmd += " HSA_ENABLE_SDMA=1 ";
    cmd += binary;
    if (gtest_filter && gtest_filter[0]) {
      cmd += " --gtest_filter=";
      cmd += gtest_filter;
    }
    cmd += " 2>&1";

    ProcessResult result;
    std::array<char, 4096> buf;
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
      result.exit_code = -1;
      return result;
    }
    while (fgets(buf.data(), buf.size(), pipe) != nullptr)
      result.output += buf.data();
    int status = pclose(pipe);
    result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return result;
  }

  pid_t daemon_pid_ = -1;
  std::string tmp_dir_;
  std::string sock_path_;
};

// --- hip_vector_add_test ---

TEST_F(DaemonTest, HipVectorAdd) {
  auto r = run_hip_test(RJ_HIP_VECTOR_ADD_BIN, "HipVectorAddTest.CorrectResult");
  EXPECT_EQ(r.exit_code, 0) << r.output;
}

// --- hip_memcpy_test ---

TEST_F(DaemonTest, HipMemcpyRoundTripFloat) {
  auto r = run_hip_test(RJ_HIP_MEMCPY_BIN, "HipMemcpyTest.RoundTripFloat");
  EXPECT_EQ(r.exit_code, 0) << r.output;
}

TEST_F(DaemonTest, HipMemcpyRoundTripInt) {
  auto r = run_hip_test(RJ_HIP_MEMCPY_BIN, "HipMemcpyTest.RoundTripInt");
  EXPECT_EQ(r.exit_code, 0) << r.output;
}

TEST_F(DaemonTest, HipMemcpyDeviceToDevice) {
  auto r = run_hip_test(RJ_HIP_MEMCPY_BIN, "HipMemcpyTest.DeviceToDevice");
  EXPECT_EQ(r.exit_code, 0) << r.output;
}

// --- Multi-client tests ---

TEST_F(DaemonTest, TwoIndependentClients) {
  std::thread t1, t2;
  ProcessResult r1, r2;

  t1 = std::thread(
      [&] { r1 = run_hip_test(RJ_HIP_VECTOR_ADD_BIN, "HipVectorAddTest.CorrectResult"); });
  t2 = std::thread([&] { r2 = run_hip_test(RJ_HIP_MEMCPY_BIN, "HipMemcpyTest.RoundTripFloat"); });

  t1.join();
  t2.join();

  EXPECT_EQ(r1.exit_code, 0) << "Client 1 (vector_add):\n" << r1.output;
  EXPECT_EQ(r2.exit_code, 0) << "Client 2 (memcpy):\n" << r2.output;
}

} // namespace
