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

// Minimum GPUs for the asymmetric topology: rank0 sees 2 devices, rank1 sees
// a third, disjoint device.
constexpr int kMinDevices = 3;

// Process exit codes shared between workers and coordinator. kCoordSkip is
// returned by the coordinator when there aren't enough GPUs; the parent maps
// it to a gtest skip.
enum WorkerStatus {
  kOk = 0,
  kWorkerFailed = 1,
  kCoordSkip = 2,
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
  if (hex.size() != sizeof(ncclUniqueId) * 2) { return false; }
  unsigned char* bytes = reinterpret_cast<unsigned char*>(&id);
  for (size_t i = 0; i < sizeof(ncclUniqueId); ++i) {
    unsigned int v = 0;
    const int parsed = sscanf(hex.c_str() + i * 2, "%02x", &v);
    if (parsed != 1) { return false; }
    bytes[i] = static_cast<unsigned char>(v);
  }
  return true;
}

// RAII holder for a worker's AllReduce resources, so every early return
// cleans up automatically instead of duplicating hipFree/hipStreamDestroy at
// each exit point. ncclCommDestroy only runs after a confirmed success: on
// any failure the process exits right away, and the coordinator kills a
// hung peer outright (see runCoordinator), so there's no peer left to
// coordinate a graceful comm teardown with.
struct WorkerResources {
  hipStream_t stream = nullptr;
  float* sendbuf = nullptr;
  float* recvbuf = nullptr;
  ncclComm_t comm = nullptr;
  bool success = false;

  WorkerResources() = default;
  WorkerResources(const WorkerResources&) = delete;
  WorkerResources& operator=(const WorkerResources&) = delete;

  ~WorkerResources() {
    if (sendbuf) { (void)hipFree(sendbuf); }
    if (recvbuf) { (void)hipFree(recvbuf); }
    if (stream) { (void)hipStreamDestroy(stream); }
    if (comm && success) { ncclCommDestroy(comm); }
  }
};

// File-local check macros. We don't use the shared HIPCHECK/NCCLCHECK from
// TestChecks.hpp because those return a raw ncclResult_t (whose values would
// collide with our WorkerStatus exit codes) and log without a rank prefix,
// which two concurrent workers interleaving on stderr need. On failure these
// log the failing call, then run onFail (a return with the caller's own code).
#define ASYM_HIPCHECK(cmd, tag, onFail)                                        \
  do {                                                                         \
    err = (cmd);                                                               \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "%s %s failed: %s\n", tag, #cmd,                         \
              hipGetErrorString(err));                                         \
      onFail;                                                                  \
    }                                                                          \
  } while (0)

#define ASYM_NCCLCHECK(cmd, tag, onFail)                                       \
  do {                                                                         \
    nres = (cmd);                                                              \
    if (nres != ncclSuccess) {                                                 \
      fprintf(stderr, "%s %s failed: %s\n", tag, #cmd,                         \
              ncclGetErrorString(nres));                                       \
      onFail;                                                                  \
    }                                                                          \
  } while (0)

// Body executed inside a re-exec'd worker process. Returns kOk on success.
int runWorker(int rank, int localDev, const ncclUniqueId& id) {
  char tag[32];
  snprintf(tag, sizeof(tag), "[asym rank%d]", rank);

  hipError_t err;
  ASYM_HIPCHECK(hipSetDevice(localDev), tag, return kWorkerFailed);

  // Clear any pre-existing HIP error so the check below reflects only what
  // happened during ncclCommInitRank.
  (void)hipGetLastError();

  WorkerResources res;
  ncclResult_t nres;
  ASYM_NCCLCHECK(ncclCommInitRank(&res.comm, kNumRanks, id, rank), tag,
                 return kWorkerFailed);

  // Core of the ROCM-27034 regression: a successful ncclCommInitRank must not
  // leave a pending HIP error behind.
  err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "%s ncclCommInitRank left a pending HIP error: %s\n", tag,
            hipGetErrorString(err));
    return kWorkerFailed;
  }

  // Also verify the communicator actually works with a small AllReduce.
  ASYM_HIPCHECK(hipStreamCreate(&res.stream), tag, return kWorkerFailed);
  ASYM_HIPCHECK(hipMalloc(&res.sendbuf, sizeof(float)), tag,
                return kWorkerFailed);
  ASYM_HIPCHECK(hipMalloc(&res.recvbuf, sizeof(float)), tag,
                return kWorkerFailed);

  const float value = static_cast<float>(rank + 1);
  ASYM_HIPCHECK(hipMemcpy(res.sendbuf, &value, sizeof(float),
                          hipMemcpyHostToDevice),
                tag, return kWorkerFailed);

  ASYM_NCCLCHECK(ncclAllReduce(res.sendbuf, res.recvbuf, 1, ncclFloat32,
                               ncclSum, res.comm, res.stream),
                 tag, return kWorkerFailed);

  ASYM_HIPCHECK(hipStreamSynchronize(res.stream), tag, return kWorkerFailed);

  float result = 0.0f;
  ASYM_HIPCHECK(hipMemcpy(&result, res.recvbuf, sizeof(float),
                          hipMemcpyDeviceToHost),
                tag, return kWorkerFailed);

  const float expected = static_cast<float>(kNumRanks) * (kNumRanks + 1) / 2.0f;
  if (result != expected) {
    fprintf(stderr, "%s wrong AllReduce result %.1f (expected %.1f)\n", tag,
            result, expected);
    return kWorkerFailed;
  }

  res.success = true;
  return kOk;
}

// Spawn one worker: fork, set the asymmetric environment, re-exec the worker.
pid_t spawnWorker(int rank, const char* visibleDevices, int localDev,
                  const std::string& uidHex) {
  pid_t pid = fork();
  if (pid != 0) {
    return pid; // parent (or fork error, reported by caller)
  }

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
  const char* tag = "[asym]";

  int numDevices = 0;
  hipError_t err;
  ASYM_HIPCHECK(hipGetDeviceCount(&numDevices), tag, return kWorkerFailed);
  if (numDevices < kMinDevices) { return kCoordSkip; }

  ncclUniqueId id;
  ncclResult_t nres;
  ASYM_NCCLCHECK(ncclGetUniqueId(&id), tag, return kWorkerFailed);
  const std::string uidHex = toHex(id);

  // rank0: HIP_VISIBLE_DEVICES=0,1 -> bind ordinal 1; rank1: =2 -> bind ordinal 0.
  const pid_t child0 = spawnWorker(0, "0,1", 1, uidHex);
  const pid_t child1 = (child0 > 0) ? spawnWorker(1, "2", 0, uidHex) : -1;
  if (child0 <= 0 || child1 <= 0) {
    if (child0 > 0) {
      kill(child0, SIGKILL);
      waitpid(child0, nullptr, 0);
    }
    if (child1 > 0) {
      kill(child1, SIGKILL);
      waitpid(child1, nullptr, 0);
    }
    fprintf(stderr, "%s fork failed (child0=%d, child1=%d)\n", tag, child0,
            child1);
    return kWorkerFailed;
  }

  // Reap whichever worker exits first. If it failed, kill its peer right
  // away instead of waiting on it - a rank already blocked in ncclAllReduce
  // would otherwise hang forever for a partner that already gave up.
  const pid_t children[kNumRanks] = {child0, child1};
  bool reaped[kNumRanks] = {false, false};
  int rc = kOk;
  for (int done_count = 0; done_count < kNumRanks; ++done_count) {
    int status = 0;
    const pid_t done = waitpid(-1, &status, 0);
    if (done < 0) {
      fprintf(stderr, "%s waitpid failed: %s\n", tag, strerror(errno));
      if (!rc) { rc = kWorkerFailed; }
      continue;
    }

    const int rank = (done == child0) ? 0 : 1;
    reaped[rank] = true;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != kOk) {
      fprintf(stderr, "%s rank %d failed or terminated abnormally\n", tag,
              rank);
      if (!rc) { rc = kWorkerFailed; }
      // Only signal the peer if it hasn't already been reaped
      const int peer = 1 - rank;
      if (!reaped[peer]) { kill(children[peer], SIGKILL); }
    }
  }
  return rc;
}

} // namespace

// Worker entry point: only active when re-exec'd by the orchestrator (i.e. when
// the rank env var is present). In a normal full-suite run it skips.
TEST(AsymmetricVisibilityWorker, Run) {
  const char* rankEnv = getenv(kEnvRank);
  if (rankEnv == nullptr) {
    GTEST_SKIP() << "worker entry point, driven by "
                    "AsymmetricVisibility.CommInitRankAllReduce";
  }

  const int rank = atoi(rankEnv);
  const char* devEnv = getenv(kEnvLocalDev);
  const char* uidEnv = getenv(kEnvUid);
  ASSERT_NE(devEnv, nullptr);
  ASSERT_NE(uidEnv, nullptr);

  ncclUniqueId id;
  ASSERT_TRUE(fromHex(uidEnv, id)) << "malformed ncclUniqueId in environment";

  // Exit with the WorkerStatus code so the orchestrator's WEXITSTATUS reflects
  // this worker's own result rather than gtest's generic exit code.
  const int rc = runWorker(rank, atoi(devEnv), id);
  if (rc != kOk) { _exit(rc); }
}

// Orchestrator: reproduces the ROCM-27034 asymmetric topology with two workers,
// rank0 seeing devices {0,1} and binding ordinal 1, rank1 seeing device {2} and
// binding ordinal 0. All HIP/NCCL work happens in a forked coordinator so the
// main gtest process stays HIP-clean for TestBed's per-test forks.
TEST(AsymmetricVisibility, CommInitRankAllReduce) {
  const char* cumemEnv = getenv("NCCL_CUMEM_ENABLE");
  const int cumemValue = cumemEnv != nullptr ? atoi(cumemEnv) : 1;
  if (cumemValue == 0) {
    GTEST_SKIP() << "NCCL_CUMEM_ENABLE explicitly disabled; this test requires "
                    "the cuMem path";
  }

  const pid_t coord = fork();
  if (coord == 0) { _exit(runCoordinator()); }
  ASSERT_GT(coord, 0) << "fork of coordinator process failed";

  int status = 0;
  ASSERT_EQ(waitpid(coord, &status, 0), coord);
  ASSERT_TRUE(WIFEXITED(status)) << "coordinator terminated abnormally";
  const int code = WEXITSTATUS(status);
  if (code == kCoordSkip) {
    GTEST_SKIP() << "requires at least " << kMinDevices
                 << " GPUs for an asymmetric topology";
  }
  EXPECT_EQ(code, kOk)
    << "asymmetric-visibility workers reported failure (exit code " << code
    << "); see stderr for the failing rank";
}

} // namespace RcclUnitTesting
