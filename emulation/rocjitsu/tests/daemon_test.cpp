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
  struct stat st {};
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
      kill(daemon_pid_, SIGKILL);
      int status = 0;
      waitpid(daemon_pid_, &status, 0);
      daemon_pid_ = -1;
    }
    std::filesystem::remove_all(tmp_dir_);
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

  ProcessResult run_rccl_rank(int rank, int world_size, const std::string &shared_dir,
                              const char *gtest_filter) {
    std::string cmd = "XDG_RUNTIME_DIR=";
    cmd += tmp_dir_;
    cmd += " ";
    cmd += RJ_DAEMON_BIN;
    cmd += " --attach --config ";
    cmd += RJ_DAEMON_CONFIG;
    cmd += " -- ";
    cmd += RJ_HIP_RCCL_BIN;
    cmd += " --rank=";
    cmd += std::to_string(rank);
    cmd += " --world-size=";
    cmd += std::to_string(world_size);
    cmd += " --shared-dir=";
    cmd += shared_dir;
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

  void run_collective(const char *filter, int world_size = 2) {
    std::string shared_tmpl = tmp_dir_ + "/coll-XXXXXX";
    ASSERT_NE(mkdtemp(shared_tmpl.data()), nullptr) << strerror(errno);
    std::string shared_dir = shared_tmpl;

    std::vector<std::thread> threads(world_size);
    std::vector<ProcessResult> results(world_size);

    for (int r = 0; r < world_size; ++r)
      threads[r] =
          std::thread([&, r] { results[r] = run_rccl_rank(r, world_size, shared_dir, filter); });
    for (auto &t : threads)
      t.join();

    for (int r = 0; r < world_size; ++r)
      EXPECT_EQ(results[r].exit_code, 0) << "Rank " << r << " failed:\n" << results[r].output;

    std::filesystem::remove_all(shared_dir);
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

// --- RCCL collective tests (2-GPU daemon) ---

class RcclDaemonTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *xdg = std::getenv("XDG_RUNTIME_DIR");
    std::string base = xdg ? xdg : "/tmp";
    std::string tmpl = base + "/rocjitsu-rccl-XXXXXX";
    ASSERT_NE(mkdtemp(tmpl.data()), nullptr) << "mkdtemp failed: " << strerror(errno);
    tmp_dir_ = tmpl;
    sock_path_ = tmp_dir_ + "/rocjitsu/daemon.sock";

    daemon_pid_ = fork();
    ASSERT_GE(daemon_pid_, 0) << "fork failed: " << strerror(errno);

    if (daemon_pid_ == 0) {
      setenv("XDG_RUNTIME_DIR", tmp_dir_.c_str(), 1);
      execl(RJ_DAEMON_BIN, RJ_DAEMON_BIN, "--daemon", "--config", RJ_DAEMON_CONFIG_2GPU, nullptr);
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
      kill(daemon_pid_, SIGKILL);
      int status = 0;
      waitpid(daemon_pid_, &status, 0);
      daemon_pid_ = -1;
    }
    std::filesystem::remove_all(tmp_dir_);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  ProcessResult run_rccl_rank(int rank, int world_size, const std::string &shared_dir,
                              const char *gtest_filter) {
    std::string cmd = "timeout 150 env XDG_RUNTIME_DIR=";
    cmd += tmp_dir_;
    cmd += " HIP_VISIBLE_DEVICES=";
    cmd += std::to_string(rank);
    cmd += " NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 HSA_NO_SCRATCH_RECLAIM=1"
           " NCCL_SOCKET_NTHREADS=1 NCCL_NSOCKS_PERTHREAD=1 ";
    cmd += RJ_DAEMON_BIN;
    cmd += " --attach --config ";
    cmd += RJ_DAEMON_CONFIG_2GPU;
    cmd += " -- ";
    cmd += RJ_HIP_RCCL_BIN;
    cmd += " --rank=";
    cmd += std::to_string(rank);
    cmd += " --world-size=";
    cmd += std::to_string(world_size);
    cmd += " --shared-dir=";
    cmd += shared_dir;
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

  void run_collective(const char *filter, int world_size = 2) {
    std::string shared_tmpl = tmp_dir_ + "/coll-XXXXXX";
    ASSERT_NE(mkdtemp(shared_tmpl.data()), nullptr) << strerror(errno);
    std::string shared_dir = shared_tmpl;

    std::vector<std::thread> threads(world_size);
    std::vector<ProcessResult> results(world_size);

    for (int r = 0; r < world_size; ++r)
      threads[r] =
          std::thread([&, r] { results[r] = run_rccl_rank(r, world_size, shared_dir, filter); });
    for (auto &t : threads)
      t.join();

    for (int r = 0; r < world_size; ++r)
      EXPECT_EQ(results[r].exit_code, 0) << "Rank " << r << " failed:\n" << results[r].output;

    std::filesystem::remove_all(shared_dir);
  }

  pid_t daemon_pid_ = -1;
  std::string tmp_dir_;
  std::string sock_path_;
};

TEST_F(RcclDaemonTest, AllReduce) { run_collective("RcclTest.AllReduce"); }
TEST_F(RcclDaemonTest, Broadcast) { run_collective("RcclTest.Broadcast"); }
TEST_F(RcclDaemonTest, AllGather) { run_collective("RcclTest.AllGather"); }
TEST_F(RcclDaemonTest, ReduceScatter) { run_collective("RcclTest.ReduceScatter"); }
TEST_F(RcclDaemonTest, SendRecv) { run_collective("RcclTest.SendRecv"); }

} // namespace
