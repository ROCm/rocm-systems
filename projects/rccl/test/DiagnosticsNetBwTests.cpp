/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "common/ProcessIsolatedTestRunner.hpp"

// NCCL 2.32 (2f12593d) adds an ib_write_bw network bandwidth check to NCCL_RUN_DIAGNOSTICS=1, run only on
// communicators that span hosts. Two ranks with distinct NCCL_HOSTID values span two hosts on one machine, and a
// fake ib_write_bw first on PATH logs every call. NCCL_NET=Socket keeps the run hardware-independent: each rank
// probes the tool with "ib_write_bw --help", then the check stops because Socket is not an IB plugin.
namespace RcclUnitTesting
{
  namespace
  {
    constexpr int         kNumRanks      = 2;
    constexpr const char* kToolDirEnvVar = "RCCL_TEST_NETBW_DIR";

    // One rank of the 2-host comm; rank 0 hands its ncclUniqueId to rank 1 through the tool directory.
    void RunRank(int rank)
    {
      int numDevices = 0;
      ASSERT_EQ(hipGetDeviceCount(&numDevices), hipSuccess);
      ASSERT_GT(numDevices, 0) << "No devices available.";
      ASSERT_EQ(hipSetDevice(rank % numDevices), hipSuccess);

      const char* dir = getenv(kToolDirEnvVar);
      ASSERT_NE(dir, nullptr);
      const std::string idPath = std::string(dir) + "/unique_id";
      ncclUniqueId      id;
      if (rank == 0) {
        ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
        std::ofstream(idPath + ".tmp", std::ios::binary).write(id.internal, sizeof(id.internal));
        ASSERT_EQ(rename((idPath + ".tmp").c_str(), idPath.c_str()), 0); // rank 1 never reads a partial id
      } else {
        std::ifstream in;
        for (int i = 0; i < 600 && !in.is_open(); i++) {
          usleep(100 * 1000);
          in.open(idPath, std::ios::binary);
        }
        ASSERT_TRUE(in.read(id.internal, sizeof(id.internal))) << "Rank 0 published no ncclUniqueId";
      }
      ncclComm_t comm;
      ASSERT_EQ(ncclCommInitRank(&comm, kNumRanks, id, rank), ncclSuccess);
      ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);
      ASSERT_EQ(system("timeout 5 stdbuf -oL ib_write_bw --harness"), 0); // the fake tool is reachable here
    }

    // A new directory for the fake tool, beside the test binary where possible: /tmp may be mounted noexec.
    std::string MakeToolDir()
    {
      std::error_code   ec;
      const std::string exeDir = std::filesystem::read_symlink("/proc/self/exe", ec).parent_path().string();
      for (const std::string& base : {exeDir, std::string("/tmp")}) {
        std::string dir = base + "/rccl-netbw-XXXXXX";
        if (!base.empty() && mkdtemp(dir.data()) != nullptr) return dir;
      }
      return "";
    }

    // Runs both ranks, each on its own host, with diagnostics on; returns the fake tool's calls, one per line.
    std::string RunTwoHostComm()
    {
      // The re-exec'd rank processes also get here; the macro below runs their rank and exits.
      const bool        parent = getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar) == nullptr;
      const std::string dir    = parent ? MakeToolDir() : "";
      if (parent && dir.empty()) {
        ADD_FAILURE() << "Cannot create a directory for the fake ib_write_bw";
        return "";
      }
      const std::string log = dir + "/calls.log";
      if (parent) {
        const std::string tool = dir + "/ib_write_bw";
        std::ofstream(tool) << "#!/bin/sh\necho \"$*\" >> '" << log << "'\n";
        chmod(tool.c_str(), 0755);
      }
      const char* path       = getenv("PATH");
      auto        rankConfig = [&](int rank) {
        return ProcessIsolatedTestRunner::TestConfig("DiagnosticsNetBwRank" + std::to_string(rank),
                                                     [rank]() { RunRank(rank); })
          .withEnvironment({{"NCCL_HOSTID", "rccl-netbw-host" + std::to_string(rank)},
                            {"NCCL_RUN_DIAGNOSTICS", "1"},
                            {"NCCL_MNNVL_ENABLE", "0"}, // a GPU fabric must not merge the two hosts
                            {"NCCL_NET", "Socket"},
                            {"NCCL_SOCKET_IFNAME", "lo"},
                            {"NCCL_MAX_NCHANNELS", "2"},
                            {kToolDirEnvVar, dir},
                            {"PATH", dir + ":" + (path ? path : "/usr/bin:/bin")}})
          .withTimeout(std::chrono::seconds(120));
      };
      ProcessIsolatedTestRunner::ExecutionOptions options;
      options.maxParallelJobs = kNumRanks; // the ranks only form a comm when they run at the same time
      RUN_ISOLATED_TESTS_WITH_OPTIONS(options, rankConfig(0), rankConfig(1));

      std::stringstream calls;
      calls << std::ifstream(log).rdbuf();
      std::error_code ec;
      std::filesystem::remove_all(dir, ec);
      return calls.str();
    }

    int CountCalls(const std::string& calls, const std::string& args)
    {
      std::istringstream lines(calls);
      int                count = 0;
      for (std::string line; std::getline(lines, line);) count += (line == args);
      return count;
    }
  }

  // Control: a 2-host comm initializes with diagnostics on and every rank reaches the fake tool, on any RCCL.
  TEST(DiagnosticsNetBw, ControlTwoHostInitReachesFakeTool)
  {
    const std::string calls = RunTwoHostComm();
    EXPECT_EQ(CountCalls(calls, "--harness"), kNumRanks) << calls;
  }

  // On a comm that spans hosts, the check probes ib_write_bw's capabilities on every rank.
  TEST(DiagnosticsNetBw, ProbesIbWriteBwOnEveryRank)
  {
    const std::string calls = RunTwoHostComm();
    EXPECT_EQ(CountCalls(calls, "--harness"), kNumRanks) << calls;
    EXPECT_EQ(CountCalls(calls, "--help"), kNumRanks)
      << "NCCL_RUN_DIAGNOSTICS=1 did not run ib_write_bw on a 2-host comm; tool calls:\n" << calls;
  }
}
