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

#include <csignal>

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

// Coordinator process exit code signalling "not enough GPUs"; the parent
// translates it to a gtest skip. Kept outside the WorkerStatus range (2-8).
constexpr int kCoordSkip = 10;

// Worker process exit codes; distinct values so the orchestrator failure log
// identifies which step failed. Non-zero codes start at 2 to avoid colliding
// with the exit code 1 that gtest itself returns on an assertion failure.
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
  if (hipMemcpy(sendbuf, &value, sizeof(float), hipMemcpyHostToDevice) !=
        hipSuccess && !rc)
    rc = kHipSetupFailed;

  nres = ncclAllReduce(sendbuf, recvbuf, 1, ncclFloat32, ncclSum, comm, stream);
  if (nres != ncclSuccess) {
    fprintf(stderr, "[asym rank%d] ncclAllReduce failed: %s\n", rank,
            ncclGetErrorString(nres));
    if (!rc) rc = kAllReduceFailed;
  }

  if (hipStreamSynchronize(stream) != hipSuccess && !rc) {
    fprintf(stderr, "[asym rank%d] hipStreamSynchronize failed\n", rank);
    rc = kSyncFailed;
  }

  if (!rc) {
    float result = 0.0f;
    if (hipMemcpy(&result, recvbuf, sizeof(float), hipMemcpyDeviceToHost) !=
        hipSuccess) {
      fprintf(stderr, "[asym rank%d] result copy back to host failed\n", rank);
      rc = kHipSetupFailed;
    } else {
      const float expected =
        static_cast<float>(kNumRanks) * (kNumRanks + 1) / 2.0f;
      if (result != expected) {
        fprintf(stderr,
                "[asym rank%d] wrong AllReduce result %.1f (expected %.1f)\n",
                rank, result, expected);
        rc = kWrongResult;
      }
    }
  }

  (void)hipFree(sendbuf);
  (void)hipFree(recvbuf);
  (void)hipStreamDestroy(stream);
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

// Coordinator body. Runs in a process forked from the main gtest process so
// that the main process never initializes the HIP/NCCL runtime: TestBed forks a
// fresh child for every collective test, and HIP is not fork-safe once
// initialized in the parent, so touching it here would crash later TestBed
// children. Returns an exit code the parent maps to skip/pass/fail.
int runCoordinator() {
  int numDevices = 0;
  if (hipGetDeviceCount(&numDevices) != hipSuccess) return kHipSetupFailed;
  if (numDevices < 3) return kCoordSkip;

  ncclUniqueId id;
  if (ncclGetUniqueId(&id) != ncclSuccess) return kInitRankFailed;
  const std::string uidHex = toHex(id);

  // rank0: HIP_VISIBLE_DEVICES=0,1 -> bind ordinal 1; rank1: =2 -> bind ordinal 0.
  const pid_t child0 = spawnWorker(0, "0,1", 1, uidHex);
  const pid_t child1 = (child0 > 0) ? spawnWorker(1, "2", 0, uidHex) : -1;
  if (child0 <= 0 || child1 <= 0) {
    if (child0 > 0) { kill(child0, SIGKILL); waitpid(child0, nullptr, 0); }
    if (child1 > 0) { kill(child1, SIGKILL); waitpid(child1, nullptr, 0); }
    fprintf(stderr, "[asym] fork failed (child0=%d, child1=%d)\n", child0,
            child1);
    return kSetDeviceFailed;
  }

  int rc = kOk;
  auto reap = [&rc](pid_t pid, int rank) {
    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) {
      fprintf(stderr, "[asym] rank %d terminated abnormally\n", rank);
      if (!rc) rc = kInitRankFailed;
    } else if (WEXITSTATUS(status) != kOk) {
      fprintf(stderr, "[asym] rank %d worker returned %d\n", rank,
              WEXITSTATUS(status));
      if (!rc) rc = WEXITSTATUS(status);
    }
  };
  reap(child0, 0);
  reap(child1, 1);
  return rc;
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

  // Exit with the step-specific WorkerStatus so the orchestrator's WEXITSTATUS
  // pinpoints the failing step instead of gtest's generic code.
  const int rc = runWorker(rank, atoi(devEnv), id);
  if (rc != kOk) _exit(rc);
}

// Orchestrator: reproduces the ROCM-27034 asymmetric topology with two workers,
// rank0 seeing devices {0,1} and binding ordinal 1, rank1 seeing device {2} and
// binding ordinal 0. All HIP/NCCL work happens in a forked coordinator so the
// main gtest process stays HIP-clean for TestBed's per-test forks.
TEST(AsymmetricVisibility, CommInitRankAllReduce) {
  const char* cumemEnv = getenv("NCCL_CUMEM_ENABLE");
  if (cumemEnv != nullptr && atoi(cumemEnv) == 0)
    GTEST_SKIP() << "NCCL_CUMEM_ENABLE explicitly disabled; this test requires "
                    "the cuMem path";

  const pid_t coord = fork();
  if (coord == 0) _exit(runCoordinator());
  ASSERT_GT(coord, 0) << "fork of coordinator process failed";

  int status = 0;
  ASSERT_EQ(waitpid(coord, &status, 0), coord);
  ASSERT_TRUE(WIFEXITED(status)) << "coordinator terminated abnormally";
  const int code = WEXITSTATUS(status);
  if (code == kCoordSkip)
    GTEST_SKIP() << "requires at least 3 GPUs for an asymmetric topology";
  EXPECT_EQ(code, kOk)
    << "asymmetric-visibility workers reported failure (exit code " << code
    << "); see stderr for the failing rank";
}

} // namespace RcclUnitTesting
