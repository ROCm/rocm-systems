/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Regression test for ROCM-27034: RCCL communicator setup under an asymmetric
// HIP_VISIBLE_DEVICES topology (each rank sees a different device subset).
//
// The bug: RCCL's clique peer-access check in init.cc queried
// hipDeviceCanAccessPeer() with a peer rank's own-process ordinal, which can be
// out of range in the local process under asymmetric visibility. That records a
// pending HIP error (hipErrorInvalidDevice) which ncclCommInitRank returns
// successfully, but which PyTorch's allocator then reports as "invalid device
// ordinal" on the first tensor allocation.
//
// This test spawns its own worker processes via fork()+execv(), each with a
// distinct HIP_VISIBLE_DEVICES set before HIP is initialized.
// The orchestrator generates the ncclUniqueId and hands it to the workers
// hex-encoded in the environment. Each worker binds its device, joins a
// 2-rank communicator, asserts that init left no pending HIP error, and runs a
// small AllReduce to confirm the communicator is functional.
//
// Note: the legacy IPC path requires the cuMem/VMM handle path to survive
// asymmetric visibility, so the workers run with NCCL_CUMEM_ENABLE=1.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace RcclUnitTesting {
namespace {

// Env var names shared between orchestrator and worker.
constexpr const char* kEnvRank = "RCCL_ASYM_RANK"; // worker rank (0 or 1)
constexpr const char* kEnvLocalDev =
  "RCCL_ASYM_LOCALDEV"; // ordinal to bind within visible set
constexpr const char* kEnvUid = "RCCL_ASYM_UID"; // hex-encoded ncclUniqueId

constexpr int kNumRanks = 2;

// Worker process exit codes; distinct values so the orchestrator failure log
// identifies which step failed. Values start at 2 to avoid colliding with the
// exit code 1 that gtest itself returns on an assertion failure.
enum WorkerStatus {
  kOk = 0,
  kSetDeviceFailed = 2,
  kInitRankFailed = 3,
  kHipSetupFailed = 4,
  kAllReduceFailed = 5,
  kSyncFailed = 6,
  kWrongResult = 7,
  kPendingHipError = 8,
};

std::string toHex(const ncclUniqueId& id) {
  const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&id);
  std::string out;
  out.reserve(sizeof(ncclUniqueId) * 2);
  char buf[3];
  for (size_t i = 0; i < sizeof(ncclUniqueId); ++i) {
    snprintf(buf, sizeof(buf), "%02x", bytes[i]);
    out += buf;
  }
  return out;
}

bool fromHex(const std::string& hex, ncclUniqueId& id) {
  if (hex.size() != sizeof(ncclUniqueId) * 2) return false;
  unsigned char* bytes = reinterpret_cast<unsigned char*>(&id);
  for (size_t i = 0; i < sizeof(ncclUniqueId); ++i) {
    unsigned int v = 0;
    if (sscanf(hex.c_str() + i * 2, "%02x", &v) != 1) return false;
    bytes[i] = static_cast<unsigned char>(v);
  }
  return true;
}

// Body executed inside a re-exec'd worker process. Returns kOk on success.
int runWorker(int rank, int localDev, const ncclUniqueId& id) {
  if (hipSetDevice(localDev) != hipSuccess) {
    fprintf(stderr, "[asym rank%d] hipSetDevice(%d) failed\n", rank, localDev);
    return kSetDeviceFailed;
  }

  // Clear any pre-existing HIP error so the check below reflects only what
  // happened during ncclCommInitRank.
  (void)hipGetLastError();

  ncclComm_t comm = nullptr;
  ncclResult_t nres = ncclCommInitRank(&comm, kNumRanks, id, rank);
  if (nres != ncclSuccess) {
    fprintf(stderr, "[asym rank%d] ncclCommInitRank failed: %s\n", rank,
            ncclGetErrorString(nres));
    return kInitRankFailed;
  }

  // Core of the ROCM-27034 regression: a successful ncclCommInitRank must not
  // leave a pending HIP error behind.
  hipError_t pendingHipError = hipGetLastError();
  int rc = kOk;
  if (pendingHipError != hipSuccess) {
    fprintf(stderr,
            "[asym rank%d] ncclCommInitRank left a pending HIP error: %s\n",
            rank, hipGetErrorString(pendingHipError));
    rc = kPendingHipError;
  }

  // Also verify the communicator actually works with a small AllReduce. Both
  // ranks must reach the collective, so a failure recorded above is returned
  // only after it - bailing out early here would deadlock the peer rank.
  hipStream_t stream = nullptr;
  float* sendbuf = nullptr;
  float* recvbuf = nullptr;
  if (hipStreamCreate(&stream) != hipSuccess ||
      hipMalloc(&sendbuf, sizeof(float)) != hipSuccess ||
      hipMalloc(&recvbuf, sizeof(float)) != hipSuccess) {
    fprintf(stderr, "[asym rank%d] HIP setup for AllReduce failed\n", rank);
    ncclCommAbort(comm);
    return rc ? rc : kHipSetupFailed;
  }

  const float value = static_cast<float>(rank + 1);
  hipMemcpy(sendbuf, &value, sizeof(float), hipMemcpyHostToDevice);

  nres = ncclAllReduce(sendbuf, recvbuf, 1, ncclFloat32, ncclSum, comm, stream);
  if (nres != ncclSuccess) {
    fprintf(stderr, "[asym rank%d] ncclAllReduce failed: %s\n", rank,
            ncclGetErrorString(nres));
    if (!rc) rc = kAllReduceFailed;
  }

  if (!rc && hipStreamSynchronize(stream) != hipSuccess) {
    fprintf(stderr, "[asym rank%d] hipStreamSynchronize failed\n", rank);
    rc = kSyncFailed;
  }

  if (!rc) {
    float result = 0.0f;
    hipMemcpy(&result, recvbuf, sizeof(float), hipMemcpyDeviceToHost);
    const float expected =
      static_cast<float>(kNumRanks) * (kNumRanks + 1) / 2.0f;
    if (result != expected) {
      fprintf(stderr,
              "[asym rank%d] wrong AllReduce result %.1f (expected %.1f)\n",
              rank, result, expected);
      rc = kWrongResult;
    }
  }

  hipFree(sendbuf);
  hipFree(recvbuf);
  hipStreamDestroy(stream);
  ncclCommDestroy(comm);
  return rc;
}

// Spawn one worker: fork, set the asymmetric environment, re-exec the worker.
pid_t spawnWorker(int rank, const char* visibleDevices, int localDev,
                  const std::string& uidHex) {
  pid_t pid = fork();
  if (pid != 0) return pid; // parent (or fork error, reported by caller)

  // Child: set the per-rank environment BEFORE HIP initializes, then re-exec
  // a fresh image so the restricted visibility takes effect cleanly.
  setenv("HIP_VISIBLE_DEVICES", visibleDevices, 1);
  setenv("CUDA_VISIBLE_DEVICES", visibleDevices, 1);
  setenv(kEnvRank, std::to_string(rank).c_str(), 1);
  setenv(kEnvLocalDev, std::to_string(localDev).c_str(), 1);
  setenv(kEnvUid, uidHex.c_str(), 1);
  setenv("NCCL_CUMEM_ENABLE", "1", 0);

  std::string filter = "--gtest_filter=AsymmetricVisibilityWorker.Run";
  char argv0[] = "rccl-UnitTests";
  char color[] = "--gtest_color=no";
  char* argv[] = {argv0, filter.data(), color, nullptr};
  execv("/proc/self/exe", argv);
  // Only reached if execv failed.
  fprintf(stderr, "[asym rank%d] execv failed: %s\n", rank, strerror(errno));
  _exit(127);
}

} // namespace

// Worker entry point: only active when re-exec'd by the orchestrator (i.e. when
// the rank env var is present). In a normal full-suite run it skips.
TEST(AsymmetricVisibilityWorker, Run) {
  const char* rankEnv = getenv(kEnvRank);
  if (rankEnv == nullptr)
    GTEST_SKIP() << "worker entry point, driven by "
                    "AsymmetricVisibility.CommInitRankAllReduce";

  const int rank = atoi(rankEnv);
  const char* devEnv = getenv(kEnvLocalDev);
  const char* uidEnv = getenv(kEnvUid);
  ASSERT_NE(devEnv, nullptr);
  ASSERT_NE(uidEnv, nullptr);

  ncclUniqueId id;
  ASSERT_TRUE(fromHex(uidEnv, id)) << "malformed ncclUniqueId in environment";

  ASSERT_EQ(runWorker(rank, atoi(devEnv), id), kOk)
    << "asymmetric-visibility worker rank " << rank << " failed";
}

// Orchestrator: reproduces the ROCM-27034 asymmetric topology with two workers,
// rank0 seeing devices {0,1} and binding ordinal 1, rank1 seeing device {2} and
// binding ordinal 0.
TEST(AsymmetricVisibility, CommInitRankAllReduce) {
  int numDevices = 0;
  ASSERT_EQ(hipGetDeviceCount(&numDevices), hipSuccess);
  if (numDevices < 3)
    GTEST_SKIP() << "requires at least 3 GPUs for an asymmetric topology";

  ncclUniqueId id;
  ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
  const std::string uidHex = toHex(id);

  // rank0: HIP_VISIBLE_DEVICES=0,1 -> bind ordinal 1; rank1: =2 -> bind ordinal 0.
  const pid_t child0 = spawnWorker(0, "0,1", 1, uidHex);
  ASSERT_GT(child0, 0) << "fork failed for rank 0";
  const pid_t child1 = spawnWorker(1, "2", 0, uidHex);
  ASSERT_GT(child1, 0) << "fork failed for rank 1";

  auto waitOk = [](pid_t pid, int rank) {
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status))
      << "rank " << rank << " terminated abnormally";
    ASSERT_EQ(WEXITSTATUS(status), 0)
      << "rank " << rank << " worker returned failure";
  };
  waitOk(child0, 0);
  waitOk(child1, 1);
}

} // namespace RcclUnitTesting
