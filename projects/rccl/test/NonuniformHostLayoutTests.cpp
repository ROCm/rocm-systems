/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// NCCL 71ef2f4 ("init: fix incomplete rank reductions"): initTransportsRank's
// AllGather1 peer loop used to break as soon as it saw hosts with different
// rank counts, so every per-peer check after that rank was skipped. Workers
// fake their host with NCCL_HOSTID and put two ranks on one GPU of one host;
// init must refuse that layout whether or not the hosts are uniform.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {
namespace {

constexpr const char* kEnvRank = "RCCL_NUHL_RANK";
constexpr const char* kEnvNumRanks = "RCCL_NUHL_NRANKS";
constexpr const char* kEnvDev = "RCCL_NUHL_DEV";
constexpr const char* kEnvUid = "RCCL_NUHL_UID";
constexpr const char* kEnvLog = "RCCL_NUHL_LOG";
constexpr const char* kDupWarn = "Multiple Ranks are using the same GPU/Partition";

struct Rank {
  const char* host;
  int dev;
};

enum { kOk = 0, kFailed = 1, kSkip = 2 };

int runWorker(int rank, int numRanks, int dev, const ncclUniqueId& id, const char* log) {
  alarm(120); // the unchecked layout may hang instead of failing
  if (hipSetDevice(dev) != hipSuccess) return kFailed;
  ncclComm_t comm = nullptr;
  const ncclResult_t res = ncclCommInitRank(&comm, numRanks, id, rank);
  fflush(NULL);
  std::ifstream in(log);
  std::stringstream text;
  text << in.rdbuf();
  const bool warned = text.str().find(kDupWarn) != std::string::npos;
  if (res != ncclInvalidUsage || !warned) {
    fprintf(stderr, "[nuhl rank%d] init returned %s, duplicate-GPU warning %s\n", rank,
            ncclGetErrorString(res), warned ? "logged" : "missing");
    return kFailed;
  }
  return kOk;
}

int runCoordinator(const std::vector<Rank>& layout) {
  int numDevices = 0;
  if (hipGetDeviceCount(&numDevices) != hipSuccess) return kFailed;
  if (numDevices < 3) return kSkip;
  ncclUniqueId id;
  if (ncclGetUniqueId(&id) != ncclSuccess) return kFailed;
  std::string uid;
  char buf[3];
  for (size_t i = 0; i < sizeof(id); ++i) {
    snprintf(buf, sizeof(buf), "%02x", reinterpret_cast<unsigned char*>(&id)[i]);
    uid += buf;
  }
  char dir[] = "/tmp/rccl_nuhl_XXXXXX";
  if (mkdtemp(dir) == nullptr) return kFailed;

  const int numRanks = static_cast<int>(layout.size());
  for (int rank = 0; rank < numRanks; ++rank) {
    if (fork() != 0) continue;
    const std::string log = std::string(dir) + "/rank" + std::to_string(rank) + ".log";
    setenv("NCCL_HOSTID", layout[rank].host, 1);
    setenv("NCCL_DEBUG", "WARN", 1);
    setenv("NCCL_DEBUG_FILE", log.c_str(), 1);
    unsetenv("NCCL_MULTI_RANK_GPU_ENABLE");
    setenv(kEnvRank, std::to_string(rank).c_str(), 1);
    setenv(kEnvNumRanks, std::to_string(numRanks).c_str(), 1);
    setenv(kEnvDev, std::to_string(layout[rank].dev).c_str(), 1);
    setenv(kEnvUid, uid.c_str(), 1);
    setenv(kEnvLog, log.c_str(), 1);
    unsetenv("GTEST_TOTAL_SHARDS");
    unsetenv("GTEST_SHARD_INDEX");
    unsetenv("GTEST_SHARD_STATUS_FILE");
    char argv0[] = "rccl-UnitTests";
    char filter[] = "--gtest_filter=NonuniformHostLayoutWorker.Run";
    char* argv[] = {argv0, filter, nullptr};
    execv("/proc/self/exe", argv);
    _exit(127);
  }
  int rc = kOk;
  for (int i = 0; i < numRanks; ++i) {
    int status = 0;
    if (wait(&status) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != kOk) rc = kFailed;
  }
  return rc;
}

void runLayout(const std::vector<Rank>& layout) {
  const pid_t coord = fork();
  if (coord == 0) _exit(runCoordinator(layout));
  ASSERT_GT(coord, 0);
  int status = 0;
  ASSERT_EQ(waitpid(coord, &status, 0), coord);
  ASSERT_TRUE(WIFEXITED(status)) << "coordinator terminated abnormally";
  if (WEXITSTATUS(status) == kSkip) GTEST_SKIP() << "needs at least 3 GPUs";
  EXPECT_EQ(WEXITSTATUS(status), kOk) << "a rank did not refuse the duplicate GPU; see stderr";
}

} // namespace

TEST(NonuniformHostLayoutWorker, Run) {
  const char* rankEnv = getenv(kEnvRank);
  if (rankEnv == nullptr) GTEST_SKIP() << "worker entry point, driven by NonuniformHostLayout.*";
  ncclUniqueId id;
  const std::string uid = getenv(kEnvUid);
  ASSERT_EQ(uid.size(), sizeof(id) * 2);
  for (size_t i = 0; i < sizeof(id); ++i) {
    reinterpret_cast<unsigned char*>(&id)[i] =
      static_cast<unsigned char>(strtoul(uid.substr(i * 2, 2).c_str(), nullptr, 16));
  }
  ProcessIsolatedTestRunner::exitChild(
    runWorker(atoi(rankEnv), atoi(getenv(kEnvNumRanks)), atoi(getenv(kEnvDev)), id, getenv(kEnvLog)));
}

// Hosts of 2, 1 and 1 ranks: rank 3 comes after the layout turns nonuniform and
// shares rank 0's host and GPU.
TEST(NonuniformHostLayout, DuplicateGpuAfterNonuniformHost) {
  runLayout({{"nuhl-a", 0}, {"nuhl-a", 1}, {"nuhl-b", 2}, {"nuhl-a", 0}});
}

// Two hosts of 2 ranks: rank 3 shares rank 2's host and GPU.
TEST(NonuniformHostLayout, DuplicateGpuUniformHosts) {
  runLayout({{"nuhl-a", 0}, {"nuhl-a", 1}, {"nuhl-b", 2}, {"nuhl-b", 2}});
}

} // namespace RcclUnitTesting
