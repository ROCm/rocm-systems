/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// The DXG open and topology-snapshot reference counts, driven through the
// public entry points of the built librocdxg.
//
// hsaKmtOpenKFD()/hsaKmtCloseKFD() and hsaKmtAcquireSystemProperties()/
// hsaKmtReleaseSystemProperties() may each be called more than once in one
// process, because more than one component uses them: ROCr, and on WSL amdsmi
// and rocprofiler-sdk alongside it. Only the first open and the first acquire
// build anything, and only the last close and the last release tear it down.
// Neither close nor release says which reference it gives back, so a forked
// child - which inherits every component's references along with the counts -
// must give back the inherited ones before any it took itself.
//
// Every case runs in a child of a runner that never opens the thunk, so the
// process-global state one case leaves behind cannot reach the next. Needs
// /dev/dxg and libdxcore; without them the test reports itself skipped.

#include <hsakmt/hsakmt.h>

#include <dirent.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

extern "C" int amdgpu_device_get_fd(struct amdgpu_device* dev);

namespace {

constexpr int kSkip = 77;

std::atomic<int> failures{0};

void Failed(const char* expr, int line) {
  std::fprintf(stderr, "dxg_lifecycle_test:%d: FAILED: %s\n", line, expr);
  ++failures;
}

#define CHECK(expr) ((expr) ? (void)0 : Failed(#expr, __LINE__))

bool IsOpen() {
  HsaVersionInfo version = {};
  return hsaKmtGetVersion(&version) == HSAKMT_STATUS_SUCCESS;
}

bool SnapshotIsLive() {
  HsaNodeProperties props = {};
  return hsaKmtGetNodeProperties(0, &props) == HSAKMT_STATUS_SUCCESS;
}

bool Acquire() {
  HsaSystemProperties props = {};
  return hsaKmtAcquireSystemProperties(&props) == HSAKMT_STATUS_SUCCESS && props.NumNodes > 0;
}

int CountOpenFds() {
  DIR* dir = opendir("/proc/self/fd");
  if (dir == nullptr) return -1;
  int n = 0;
  while (readdir(dir) != nullptr) ++n;
  closedir(dir);
  return n;
}

// Bypasses pthread_atfork, as clone() or a direct system call does. The thunk
// must notice the fork from getpid() alone.
pid_t RawFork() { return static_cast<pid_t>(syscall(SYS_fork)); }

// Waits for a child that reports its failure count as its exit status.
int Reap(pid_t pid) {
  int status = 0;
  if (waitpid(pid, &status, 0) != pid) return 1;
  if (WIFSIGNALED(status)) {
    std::fprintf(stderr, "dxg_lifecycle_test: child killed by signal %d\n", WTERMSIG(status));
    return 1;
  }
  if (!WIFEXITED(status)) return 1;
  return WEXITSTATUS(status);
}

void UnmatchedCallsAreRejected() {
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED);
  CHECK(hsaKmtReleaseSystemProperties() == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED);

  CHECK(hsaKmtOpenKFD() == HSAKMT_STATUS_SUCCESS);
  CHECK(hsaKmtReleaseSystemProperties() == HSAKMT_STATUS_INVALID_PARAMETER);
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_SUCCESS);

  // The count did not wrap: one more close is still an unmatched one.
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED);
}

void NestedOpensShareOneConnection() {
  CHECK(hsaKmtOpenKFD() == HSAKMT_STATUS_SUCCESS);
  CHECK(hsaKmtOpenKFD() == HSAKMT_STATUS_KERNEL_ALREADY_OPENED);

  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_SUCCESS);
  CHECK(IsOpen());  // the first consumer's reference is still there

  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_SUCCESS);
  CHECK(!IsOpen());
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED);

  // The last close left nothing behind that a fresh first open trips over.
  CHECK(hsaKmtOpenKFD() == HSAKMT_STATUS_SUCCESS);
  CHECK(Acquire());
  CHECK(hsaKmtReleaseSystemProperties() == HSAKMT_STATUS_SUCCESS);
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_SUCCESS);
}

void NestedAcquiresShareOneSnapshot() {
  CHECK(hsaKmtOpenKFD() == HSAKMT_STATUS_SUCCESS);
  CHECK(Acquire());
  CHECK(Acquire());

  CHECK(hsaKmtReleaseSystemProperties() == HSAKMT_STATUS_SUCCESS);
  CHECK(SnapshotIsLive());  // still held by the other acquirer

  CHECK(hsaKmtReleaseSystemProperties() == HSAKMT_STATUS_SUCCESS);
  CHECK(!SnapshotIsLive());
  CHECK(hsaKmtReleaseSystemProperties() == HSAKMT_STATUS_INVALID_PARAMETER);

  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_SUCCESS);
}

// A snapshot cannot outlive the DXCore session whose devices it names.
void LastCloseDropsAnOutstandingSnapshot() {
  CHECK(hsaKmtOpenKFD() == HSAKMT_STATUS_SUCCESS);
  CHECK(Acquire());
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_SUCCESS);

  CHECK(hsaKmtOpenKFD() == HSAKMT_STATUS_SUCCESS);
  CHECK(!SnapshotIsLive());
  CHECK(hsaKmtReleaseSystemProperties() == HSAKMT_STATUS_INVALID_PARAMETER);
  CHECK(Acquire());
  CHECK(hsaKmtReleaseSystemProperties() == HSAKMT_STATUS_SUCCESS);
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_SUCCESS);
}

// Components opening and closing on their own threads, so the count keeps
// passing through zero while others are mid-use: no thread may see its own
// open or snapshot torn down by another thread's close or release.
//
// The threads start together. This case runs in a child of a process that has
// loaded the library, so their first opens all race through the thunk's fork
// recovery as well.
void ConcurrentConsumersKeepTheirReferences() {
  constexpr int kThreads = 4;
  constexpr int kIterations = 20;

  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&go] {
      while (!go.load()) std::this_thread::yield();
      for (int i = 0; i < kIterations; ++i) {
        const HSAKMT_STATUS open = hsaKmtOpenKFD();
        CHECK(open == HSAKMT_STATUS_SUCCESS || open == HSAKMT_STATUS_KERNEL_ALREADY_OPENED);
        CHECK(Acquire());
        CHECK(IsOpen());
        CHECK(SnapshotIsLive());
        CHECK(hsaKmtReleaseSystemProperties() == HSAKMT_STATUS_SUCCESS);
        CHECK(IsOpen());
        CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_SUCCESS);
      }
    });
  }
  go = true;
  for (auto& thread : threads) thread.join();

  CHECK(!IsOpen());
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED);
}

// The child's half of the two fork cases below: give back what the parent
// held, as ROCr's KfdDriver::ShutDown() does in a child, and touch nothing.
void GiveBackInheritedReferences() {
  CHECK(hsaKmtRuntimeDisable() == HSAKMT_STATUS_SUCCESS);
  CHECK(hsaKmtReleaseSystemProperties() == HSAKMT_STATUS_SUCCESS);
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_SUCCESS);

  // One each was inherited; a second is unmatched.
  CHECK(hsaKmtReleaseSystemProperties() == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED);
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED);
  CHECK(!IsOpen());
}

// The child opens and acquires for itself, then a component that opened in
// the parent closes. That close must not be matched to the child's open.
void StaleCloseLeavesTheChildsOwnReferences() {
  CHECK(hsaKmtOpenKFD() == HSAKMT_STATUS_SUCCESS);  // a fresh first open, not a nested one
  CHECK(Acquire());

  CHECK(hsaKmtReleaseSystemProperties() == HSAKMT_STATUS_SUCCESS);  // the inherited one
  CHECK(SnapshotIsLive());
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_SUCCESS);  // the inherited one
  CHECK(IsOpen());
  CHECK(SnapshotIsLive());

  CHECK(hsaKmtReleaseSystemProperties() == HSAKMT_STATUS_SUCCESS);
  CHECK(!SnapshotIsLive());
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_SUCCESS);
  CHECK(!IsOpen());
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED);
}

void ForkWithParentReferences(pid_t (*do_fork)(), void (*child)()) {
  CHECK(hsaKmtOpenKFD() == HSAKMT_STATUS_SUCCESS);
  CHECK(Acquire());

  const pid_t pid = do_fork();
  if (pid == 0) {
    child();
    _exit(failures);
  }
  CHECK(pid > 0);
  if (pid > 0) CHECK(Reap(pid) == 0);

  // Nothing the child did reached the parent's references.
  CHECK(IsOpen());
  CHECK(SnapshotIsLive());
  CHECK(hsaKmtReleaseSystemProperties() == HSAKMT_STATUS_SUCCESS);
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_SUCCESS);
  CHECK(!IsOpen());
}

pid_t LibcFork() { return fork(); }

void ForkChildGivesBackInheritedReferences() {
  ForkWithParentReferences(LibcFork, GiveBackInheritedReferences);
}

void RawForkChildGivesBackInheritedReferences() {
  ForkWithParentReferences(RawFork, GiveBackInheritedReferences);
}

void ForkChildStaleCloseLeavesItsOwnReferences() {
  ForkWithParentReferences(LibcFork, StaleCloseLeavesTheChildsOwnReferences);
}

void RawForkChildStaleCloseLeavesItsOwnReferences() {
  ForkWithParentReferences(RawFork, StaleCloseLeavesTheChildsOwnReferences);
}

// Run with a libdxcore.so that exports none of the D3DKMT entry points, so the
// open fails after it has opened /dev/dxg. Nothing may survive the failure.
int PartialOpenFailure() {
  if (access("/dev/dxg", R_OK | W_OK) != 0) return kSkip;

  const int fds_before = CountOpenFds();
  CHECK(hsaKmtOpenKFD() != HSAKMT_STATUS_SUCCESS);
  CHECK(CountOpenFds() == fds_before);
  CHECK(amdgpu_device_get_fd(nullptr) == -1);  // no closed descriptor cached
  CHECK(!IsOpen());
  CHECK(hsaKmtCloseKFD() == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED);

  // A retry starts from the same state rather than adopting the dead fd.
  CHECK(hsaKmtOpenKFD() != HSAKMT_STATUS_SUCCESS);
  CHECK(CountOpenFds() == fds_before);
  CHECK(amdgpu_device_get_fd(nullptr) == -1);

  return failures == 0 ? 0 : 1;
}

struct Case {
  const char* name;
  void (*run)();
};

const Case kCases[] = {
    {"UnmatchedCallsAreRejected", UnmatchedCallsAreRejected},
    {"NestedOpensShareOneConnection", NestedOpensShareOneConnection},
    {"NestedAcquiresShareOneSnapshot", NestedAcquiresShareOneSnapshot},
    {"LastCloseDropsAnOutstandingSnapshot", LastCloseDropsAnOutstandingSnapshot},
    {"ConcurrentConsumersKeepTheirReferences", ConcurrentConsumersKeepTheirReferences},
    {"ForkChildGivesBackInheritedReferences", ForkChildGivesBackInheritedReferences},
    {"RawForkChildGivesBackInheritedReferences", RawForkChildGivesBackInheritedReferences},
    {"ForkChildStaleCloseLeavesItsOwnReferences", ForkChildStaleCloseLeavesItsOwnReferences},
    {"RawForkChildStaleCloseLeavesItsOwnReferences", RawForkChildStaleCloseLeavesItsOwnReferences},
};

// Whether this machine can open the thunk and take a snapshot at all.
bool DxgAvailable() {
  const pid_t pid = fork();
  if (pid == 0) {
    const bool ok = hsaKmtOpenKFD() == HSAKMT_STATUS_SUCCESS && Acquire();
    _exit(ok ? 0 : 1);
  }
  return pid > 0 && Reap(pid) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "partial-open") == 0) return PartialOpenFailure();

  if (!DxgAvailable()) {
    std::printf("dxg_lifecycle_test: no DXG adapter, skipping\n");
    return kSkip;
  }

  int failed_cases = 0;
  for (const Case& c : kCases) {
    const pid_t pid = fork();
    if (pid == 0) {
      alarm(60);  // a deadlocked case fails by SIGALRM instead of hanging the run
      c.run();
      _exit(failures);
    }
    const int result = pid > 0 ? Reap(pid) : 1;
    std::printf("dxg_lifecycle_test: %s %s\n", result == 0 ? "PASS" : "FAIL", c.name);
    if (result != 0) ++failed_cases;
  }

  if (failed_cases != 0) {
    std::fprintf(stderr, "dxg_lifecycle_test: %d case(s) failed\n", failed_cases);
    return 1;
  }

  std::printf("dxg_lifecycle_test: all cases passed\n");
  return 0;
}
