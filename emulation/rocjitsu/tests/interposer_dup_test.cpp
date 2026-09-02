// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file interposer_dup_test.cpp
/// @brief LD_PRELOAD regression tests for KFD/DRM descriptor bookkeeping,
///        GEM/PRIME mappings, and synchronous DRM timeline state.
///
/// @details These run with librocjitsu.so preloaded (see the ENVIRONMENT set in
/// tests/CMakeLists.txt) so that open("/dev/kfd") is serviced by the simulated
/// KFD driver. AMDKFD_IOC_GET_VERSION is used purely as a routing probe: it
/// succeeds (returns 0 and fills the version) only when the fd is routed to a
/// KFD backend, and fails when the fd falls through to the real (non-KFD)
/// descriptor. The SimulatedDriver-level unit tests (KfdIoctlTest) cannot catch
/// these because they exercise the driver object directly, bypassing the fd
/// tracking that lives in the interposer.

#include "scoped_temp.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
// Use the checked-in UAPI because older system libdrm headers do not expose the
// GEM_VA timeline fields exercised by these tests.
#include "linux/uapi/kfd_ioctl.h"
#include <libdrm/amdgpu_drm.h>
#include <libdrm/drm.h>
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <spawn.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/wait.h>
#include <thread>
#include <time.h>
#include <unistd.h>

extern char **environ;

namespace {

// Issue AMDKFD_IOC_GET_VERSION on fd. Returns true if the fd routed to a KFD
// backend (ioctl succeeded and reported the expected major version).
bool kfd_version_ok(int fd) {
  kfd_ioctl_get_version_args args{};
  int rc = ioctl(fd, AMDKFD_IOC_GET_VERSION, &args);
  return rc == 0 && args.major_version == KFD_IOCTL_MAJOR_VERSION;
}

int open_kfd() { return open("/dev/kfd", O_RDWR | O_CLOEXEC); }

void noop_signal_handler(int) {}

} // namespace

namespace {
int open_drm_render();
int make_sized_memfd(size_t size);
} // namespace

// A plain dup() of the KFD fd must keep routing KFD ioctls to the driver, and
// closing the original primary fd must not tear the process down while the dup
// still holds a reference (dup keeps the backend alive).
TEST(InterposerDupTest, DupKeepsKfdRoutingAfterPrimaryClose) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  EXPECT_TRUE(kfd_version_ok(kfd));

  int dup_fd = dup(kfd);
  ASSERT_GE(dup_fd, 0);
  EXPECT_TRUE(kfd_version_ok(dup_fd));

  // Close the primary; the dup must still route to the live KFD backend.
  EXPECT_EQ(close(kfd), 0);
  EXPECT_TRUE(kfd_version_ok(dup_fd));

  EXPECT_EQ(close(dup_fd), 0);
}

TEST(InterposerDupTest, ProcMapsNamesRemoteKfdMarker) {
  // The marker only exists on the remote path: RemoteDriver::open() creates it,
  // and in local mode there is no RemoteDriver at all. $ROCJITSU_INVOCATION_DIR
  // is set for every rocjitsu-launched process regardless of backend, so gate on
  // the daemon socket actually being there instead -- otherwise this skips
  // nothing under a plain `rocjitsu --config ... --` run and fails for a reason
  // that is not a defect.
  const char *invocation_dir = getenv("ROCJITSU_INVOCATION_DIR");
  if (invocation_dir == nullptr || *invocation_dir == '\0')
    GTEST_SKIP() << "remote backend required";
  if (access((std::string(invocation_dir) + "/daemon.sock").c_str(), F_OK) != 0)
    GTEST_SKIP() << "remote backend required";

  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);

  std::ifstream maps("/proc/self/maps");
  ASSERT_TRUE(maps.is_open());
  std::string contents((std::istreambuf_iterator<char>(maps)), std::istreambuf_iterator<char>());
  EXPECT_NE(contents.find("/dev/kfd"), std::string::npos);
  EXPECT_EQ(contents.find("rocjitsu_remote_kfd"), std::string::npos);

  EXPECT_EQ(close(kfd), 0);
}

// fcntl(F_DUPFD_CLOEXEC) is the dup path libdrm uses; it must also preserve KFD
// routing on the duplicate.
TEST(InterposerDupTest, FcntlDupfdKeepsKfdRouting) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);

  int dup_fd = fcntl(kfd, F_DUPFD_CLOEXEC, 0);
  ASSERT_GE(dup_fd, 0);
  EXPECT_TRUE(kfd_version_ok(dup_fd));

  EXPECT_EQ(close(dup_fd), 0);
  // Original still routes.
  EXPECT_TRUE(kfd_version_ok(kfd));
  EXPECT_EQ(close(kfd), 0);
}

// dup2 that OVERWRITES the primary KFD fd number must invalidate the old primary
// identity: after dup2(other, kfd) the kfd number now names 'other', so KFD
// ioctls on it must NOT be routed to the (now-replaced) KFD backend, and closing
// it must behave like closing a normal fd.
TEST(InterposerDupTest, Dup2OverPrimaryInvalidatesKfdIdentity) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  EXPECT_TRUE(kfd_version_ok(kfd));

  // A plain pipe fd to overwrite the primary KFD fd number with.
  int pipefd[2];
  ASSERT_EQ(pipe(pipefd), 0);

  // Overwrite the KFD primary number with the read end of the pipe.
  ASSERT_EQ(dup2(pipefd[0], kfd), kfd);

  // The kfd number now refers to the pipe, not the KFD backend: a KFD ioctl must
  // no longer succeed against it.
  EXPECT_FALSE(kfd_version_ok(kfd));

  // Cleanup. Closing the overwritten number closes the pipe read-end dup.
  EXPECT_EQ(close(kfd), 0);
  EXPECT_EQ(close(pipefd[0]), 0);
  EXPECT_EQ(close(pipefd[1]), 0);
}

// dup2 of the KFD fd ONTO a fresh number must make the target route KFD ioctls,
// and the reference bookkeeping must let both fds be closed without prematurely
// destroying the backend.
TEST(InterposerDupTest, Dup2OntoFreshFdRoutesKfd) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);

  // Reserve a target fd number with a pipe end, then dup2 the KFD fd onto it.
  int pipefd[2];
  ASSERT_EQ(pipe(pipefd), 0);
  int target = pipefd[0];

  ASSERT_EQ(dup2(kfd, target), target);
  EXPECT_TRUE(kfd_version_ok(target));

  // Close the primary; the dup2 target keeps the backend alive.
  EXPECT_EQ(close(kfd), 0);
  EXPECT_TRUE(kfd_version_ok(target));

  EXPECT_EQ(close(target), 0);
  EXPECT_EQ(close(pipefd[1]), 0);
}

// dup3 of the KFD fd onto a fresh number must route KFD ioctls to the target,
// and dup3(fd, fd, flags) must fail with EINVAL without disturbing tracking (the
// interposer's reserve/reconcile path must roll back cleanly on that failure).
TEST(InterposerDupTest, Dup3RoutesKfdAndRejectsSameFd) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  EXPECT_TRUE(kfd_version_ok(kfd));

  // dup3(fd, fd, ...) is required to fail with EINVAL and leave fd untouched.
  errno = 0;
  EXPECT_EQ(dup3(kfd, kfd, O_CLOEXEC), -1);
  EXPECT_EQ(errno, EINVAL);
  // The primary must still route after the rejected dup3 (no tracking disturbed).
  EXPECT_TRUE(kfd_version_ok(kfd));

  // dup3 onto a fresh number routes KFD, and both fds close cleanly.
  int pipefd[2];
  ASSERT_EQ(pipe(pipefd), 0);
  int target = pipefd[0];
  ASSERT_EQ(dup3(kfd, target, O_CLOEXEC), target);
  EXPECT_TRUE(kfd_version_ok(target));

  EXPECT_EQ(close(kfd), 0);
  EXPECT_TRUE(kfd_version_ok(target)); // dup3 target keeps the backend alive.

  EXPECT_EQ(close(target), 0);
  EXPECT_EQ(close(pipefd[1]), 0);
}

TEST(InterposerDupTest, VforkChildCloseKeepsParentKfdRoutable) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));

  pid_t child = vfork();
  ASSERT_NE(child, -1);
  if (child == 0) {
    close(kfd);
    _exit(0);
  }

  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  EXPECT_TRUE(kfd_version_ok(kfd));
  EXPECT_EQ(close(kfd), 0);
}

// Overwriting the primary KFD fd number via dup2 while a dup keeps the backend
// alive must (a) leave the dup routing, and (b) let a fresh open("/dev/kfd")
// return a valid, routable KFD fd. This is the reopen-after-overwrite path that
// re-mints the primary fd number (remote: reissue_synthetic_kfd_fd under
// remote_mutex_; local: ensure_fd_created under process_mutex_) without
// disturbing the still-live backend the dup holds. Runs identically on the local
// and daemon (remote) harnesses.
TEST(InterposerDupTest, ReopenAfterPrimaryOverwriteKeepsBackend) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  EXPECT_TRUE(kfd_version_ok(kfd));

  // A dup keeps the backend alive across the primary's overwrite.
  int keeper = dup(kfd);
  ASSERT_GE(keeper, 0);
  EXPECT_TRUE(kfd_version_ok(keeper));

  // Overwrite the primary fd number with an unrelated pipe end. After dup2 the
  // kfd number aliases the pipe read-end, so pipefd[0] is redundant and must be
  // closed to avoid leaking a descriptor across the other tests in this process.
  int pipefd[2];
  ASSERT_EQ(pipe(pipefd), 0);
  ASSERT_EQ(dup2(pipefd[0], kfd), kfd);
  EXPECT_EQ(close(pipefd[0]), 0);
  // The overwritten number now names the pipe, not the KFD backend.
  EXPECT_FALSE(kfd_version_ok(kfd));
  // The dup still routes: the backend stayed alive.
  EXPECT_TRUE(kfd_version_ok(keeper));

  // A fresh open must return a valid, routable KFD fd (re-minted primary).
  int kfd2 = open_kfd();
  ASSERT_GE(kfd2, 0) << "reopen after primary overwrite returned " << kfd2;
  EXPECT_TRUE(kfd_version_ok(kfd2));

  EXPECT_EQ(close(kfd2), 0);
  EXPECT_EQ(close(keeper), 0);
  EXPECT_EQ(close(kfd), 0); // closes the pipe-read dup installed over the number
  EXPECT_EQ(close(pipefd[1]), 0);
}

// Serialized reopen-after-overwrite under contention. The interposer keeps a
// single primary KFD fd slot, so a distinct dup (keeper) holds the backend alive
// while the main thread repeatedly overwrites the primary fd number and reopens
// "/dev/kfd" — always exactly one primary at a time, matching how a real client
// uses /dev/kfd. A background thread churns dup/close on the keeper so backend
// retain/release runs concurrently with invalidation + reopen. With invalidation
// and open serialized on the same lock (remote_mutex_ / process_mutex_), every
// reopen must return a routable KFD fd — never -1 or a reused non-KFD descriptor
// (the ENOTTY case the review reproduced). This is the invariant asserted here,
// and it holds on both the local (process_mutex_) and daemon/remote
// (remote_mutex_) harnesses. Note: the keeper's own routability is deliberately
// not asserted after the loop — on the local backend a reopen rebinds the shared
// backing fd and untracks existing dups (clear_dups), which is expected
// local-only behavior unrelated to the reopen-routability invariant under test.
TEST(InterposerDupTest, SerializedReopenUnderContentionStaysRoutable) {
  int primary = open_kfd();
  ASSERT_GE(primary, 0);
  ASSERT_TRUE(kfd_version_ok(primary));
  int keeper = dup(primary); // distinct number; holds the backend across reopens
  ASSERT_GE(keeper, 0);
  ASSERT_TRUE(kfd_version_ok(keeper));

  constexpr int kIters = 500;
  std::atomic<bool> stop{false};

  // Background churn: dup the keeper and close it, exercising backend
  // retain/release concurrently with the reopen loop below. A short yield/sleep
  // between iterations keeps the contention window open without spinning a full
  // core (which would add CI flakiness/timeouts under parallel test runs).
  std::thread churn([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      int d = dup(keeper);
      if (d >= 0)
        close(d);
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  });

  // Use non-fatal checks (EXPECT_*/break) rather than ASSERT_* inside the loop:
  // the churn thread is joinable here, so a fatal assertion that returned early
  // would skip stop/join and terminate the process. Any setup failure breaks to
  // the shutdown path below.
  int bad = 0;
  bool setup_failed = false;
  for (int i = 0; i < kIters; ++i) {
    // Overwrite the current primary fd number with a pipe end, releasing the
    // primary's reference (keeper keeps the backend alive). This drives
    // invalidate_overwritten_kfd_fd() concurrently with the churn thread. After
    // dup2 the primary number aliases the pipe read-end, so close both the
    // now-redundant pipefd[0] and the aliased primary number, plus pipefd[1].
    int pipefd[2];
    if (pipe(pipefd) != 0) {
      setup_failed = true;
      break;
    }
    if (dup2(pipefd[0], primary) != primary) {
      setup_failed = true;
      close(pipefd[0]);
      close(pipefd[1]);
      break;
    }
    close(primary); // now the pipe-read dup installed over the number
    close(pipefd[0]);
    close(pipefd[1]);

    // Reopen: the slot was cleared, so this must re-mint a fresh, routable
    // primary (a distinct number from the still-open keeper).
    primary = open_kfd();
    if (primary < 0) {
      ++bad;
      primary = dup(keeper); // recover so the loop can continue overwriting
      if (primary < 0) {     // recovery also failed under fd pressure: stop.
        setup_failed = true;
        break;
      }
      continue;
    }
    if (!kfd_version_ok(primary))
      ++bad;
  }

  stop.store(true);
  churn.join();

  EXPECT_FALSE(setup_failed) << "pipe()/dup2() failed under resource pressure";
  EXPECT_EQ(bad, 0) << "a reopen returned -1 or a non-routable fd under contention";
  if (primary >= 0) {
    EXPECT_EQ(close(primary), 0);
  }
  EXPECT_EQ(close(keeper), 0);
}

TEST(InterposerDupTest, FcntlDupfdReplacesStaleDrmTracking) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int first = open_drm_render();
  if (first < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";
  int stale_fd = dup(first);
  ASSERT_GE(stale_fd, 0);
  int second = open_drm_render();
  ASSERT_GE(second, 0);

  drm_syncobj_create first_syncobj{};
  drm_syncobj_create second_syncobj{};
  ASSERT_EQ(ioctl(first, DRM_IOCTL_SYNCOBJ_CREATE, &first_syncobj), 0);
  ASSERT_EQ(ioctl(second, DRM_IOCTL_SYNCOBJ_CREATE, &second_syncobj), 0);

  // Deliberately bypass the interposed close() so its DRM tracking entry stays
  // stale; the fcntl duplicate below must replace that stale entry safely.
  ASSERT_EQ(syscall(SYS_close, stale_fd), 0);
  int reused = fcntl(second, F_DUPFD_CLOEXEC, stale_fd);
  ASSERT_EQ(reused, stale_fd);

  drm_syncobj_destroy destroy{};
  destroy.handle = second_syncobj.handle;
  EXPECT_EQ(ioctl(reused, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);
  destroy.handle = first_syncobj.handle;
  EXPECT_EQ(ioctl(first, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);

  EXPECT_EQ(close(reused), 0);
  EXPECT_EQ(close(second), 0);
  EXPECT_EQ(close(first), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerDupTest, NonDrmDuplicatesClearStaleDrmTracking) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  drm_syncobj_create create{};
  ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &create), 0);
  int ordinary = make_sized_memfd(0x1000);
  ASSERT_GE(ordinary, 0);
  int stale_plain = dup(drm);
  int stale_fcntl = dup(drm);
  ASSERT_GE(stale_plain, 0);
  ASSERT_GE(stale_fcntl, 0);

  // Bypass the interposed close so both DRM entries remain stale, then force
  // ordinary-file duplicates onto those exact numbers. Successful duplication
  // must replace-or-clear tracking rather than preserving the old namespace.
  ASSERT_EQ(syscall(SYS_close, stale_plain), 0);
  ASSERT_EQ(syscall(SYS_close, stale_fcntl), 0);
  int plain = dup(ordinary);
  ASSERT_EQ(plain, stale_plain);
  int fcntl_dup = fcntl(ordinary, F_DUPFD_CLOEXEC, stale_fcntl);
  ASSERT_EQ(fcntl_dup, stale_fcntl);

  drm_syncobj_destroy destroy{};
  destroy.handle = create.handle;
  EXPECT_EQ(ioctl(plain, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), -1);
  EXPECT_EQ(errno, ENOTTY);
  EXPECT_EQ(ioctl(fcntl_dup, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), -1);
  EXPECT_EQ(errno, ENOTTY);
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);

  EXPECT_EQ(close(fcntl_dup), 0);
  EXPECT_EQ(close(plain), 0);
  EXPECT_EQ(close(ordinary), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerDupTest, DrmCloseReuseRaceNeverMisclassifiesDuplicate) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));

  constexpr int kIterations = 200;
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    int drm = open_drm_render();
    ASSERT_GE(drm, 0);
    drm_syncobj_create create{};
    ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &create), 0);
    struct stat drm_stat {};
    ASSERT_EQ(syscall(SYS_fstat, drm, &drm_stat), 0);

    int replacement = make_sized_memfd(0x1000);
    ASSERT_GE(replacement, 0);
    struct stat replacement_stat {};
    ASSERT_EQ(syscall(SYS_fstat, replacement, &replacement_stat), 0);

    std::barrier start(3);
    std::atomic<int> duplicated{-1};
    std::atomic<int> close_result{-1};
    std::atomic<int> reuse_result{-1};
    std::thread duplicator([&] {
      start.arrive_and_wait();
      duplicated = dup(drm);
    });
    std::thread closer([&] {
      start.arrive_and_wait();
      close_result = close(drm);
      reuse_result = dup2(replacement, drm);
    });
    start.arrive_and_wait();
    duplicator.join();
    closer.join();
    ASSERT_EQ(close_result.load(), 0);
    ASSERT_EQ(reuse_result.load(), drm);

    if (duplicated.load() >= 0) {
      struct stat duplicate_stat {};
      ASSERT_EQ(syscall(SYS_fstat, duplicated.load(), &duplicate_stat), 0);
      drm_syncobj_destroy destroy{};
      destroy.handle = create.handle;
      if (duplicate_stat.st_ino == drm_stat.st_ino) {
        EXPECT_EQ(ioctl(duplicated.load(), DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);
      } else {
        EXPECT_EQ(duplicate_stat.st_ino, replacement_stat.st_ino);
        EXPECT_EQ(ioctl(duplicated.load(), DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), -1);
        EXPECT_EQ(errno, ENOTTY);
      }
      EXPECT_EQ(close(duplicated.load()), 0);
    }

    EXPECT_EQ(close(drm), 0);
    EXPECT_EQ(close(replacement), 0);
  }
  EXPECT_EQ(close(kfd), 0);
}

namespace {

// Open the synthetic DRM render node the interposer exposes for the simulated GPU
// (render minor 128 in the KMD test configs). Requires the KFD driver to be up, so
// callers open /dev/kfd first.
int open_drm_render() { return open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC); }

bool query_drm_device_info(int drm_fd, drm_amdgpu_info_device *device) {
  drm_amdgpu_info query{};
  query.return_pointer = reinterpret_cast<uint64_t>(device);
  query.return_size = sizeof(*device);
  query.query = AMDGPU_INFO_DEV_INFO;
  return ioctl(drm_fd, DRM_IOCTL_AMDGPU_INFO, &query) == 0;
}

// Create an mmap-able, sized stand-in for a dmabuf export fd. PRIME_FD_TO_HANDLE
// fstats the fd for the BO size and later MAP mmaps it, so the fd must be a real
// sized, mappable object; a memfd satisfies both without a KFD allocation.
int make_sized_memfd(size_t size) {
  int fd = memfd_create("rocjitsu_gem_test", MFD_CLOEXEC);
  if (fd < 0)
    return -1;
  if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

// Mint a stable GEM handle for a dmabuf fd via PRIME_FD_TO_HANDLE on the DRM fd.
bool prime_import(int drm_fd, int dmabuf_fd, uint32_t *handle) {
  drm_prime_handle prime{};
  prime.fd = dmabuf_fd;
  if (ioctl(drm_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime) != 0)
    return false;
  *handle = prime.handle;
  return true;
}

// DRM_AMDGPU_GEM_VA is a DRM_COMMAND-relative ioctl; build the request number the
// same way libdrm_amdgpu does. Wrapped in a function so the test reads cleanly.
unsigned long DRM_AMDGPU_GEM_VA_request() {
  return DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_VA, drm_amdgpu_gem_va);
}

int gem_close(int drm_fd, uint32_t handle) {
  drm_gem_close gc{};
  gc.handle = handle;
  return ioctl(drm_fd, DRM_IOCTL_GEM_CLOSE, &gc);
}

int64_t monotonic_deadline_after(std::chrono::nanoseconds delay) {
  timespec now{};
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    return 0;
  return static_cast<int64_t>(now.tv_sec) * 1'000'000'000LL + now.tv_nsec + delay.count();
}

} // namespace

TEST(InterposerDrmTest, DeviceInfoReportsActiveCuCount) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));

  int drm = open_drm_render();
  ASSERT_GE(drm, 0);

  drm_amdgpu_info_device device{};
  ASSERT_TRUE(query_drm_device_info(drm, &device));
  EXPECT_EQ(device.device_id, 30112u);
  EXPECT_EQ(device.cu_active_number, 256u);

  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerSyncobjTest, VmTimelineWaitObservesSynchronousMapAndUnmap) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  drm_syncobj_create create{};
  ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &create), 0);
  ASSERT_NE(create.handle, 0u);

  constexpr size_t kBoSize = 0x1000;
  constexpr uint64_t kVa = 0x1000000000ULL;
  int dmabuf = make_sized_memfd(kBoSize);
  ASSERT_GE(dmabuf, 0);
  uint32_t gem_handle = 0;
  ASSERT_TRUE(prime_import(drm, dmabuf, &gem_handle));

  uint32_t handles[] = {create.handle};
  uint64_t points[] = {1};
  drm_syncobj_timeline_wait wait{};
  wait.handles = reinterpret_cast<uintptr_t>(handles);
  wait.points = reinterpret_cast<uintptr_t>(points);
  wait.count_handles = 1;
  wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), -1);
  EXPECT_EQ(errno, EINVAL);
  wait.flags |= DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), -1);
  EXPECT_EQ(errno, ETIME);

  const unsigned long gem_va = DRM_AMDGPU_GEM_VA_request();
  drm_amdgpu_gem_va map{};
  map.handle = gem_handle;
  map.operation = AMDGPU_VA_OP_MAP;
  map.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  map.va_address = kVa;
  map.map_size = kBoSize;
  map.vm_timeline_point = 1;
  map.vm_timeline_syncobj_out = create.handle;
  uint32_t input_handle = create.handle;
  map.num_syncobj_handles = 1;
  map.input_fence_syncobj_handles = reinterpret_cast<uintptr_t>(&input_handle);
  EXPECT_EQ(ioctl(drm, gem_va, &map), -1);
  EXPECT_EQ(errno, EINVAL);
  map.num_syncobj_handles = 0;
  // The count is authoritative. A reusable caller buffer may leave this unused
  // pointer non-null when there are no input fences.
  ASSERT_EQ(ioctl(drm, gem_va, &map), 0);

  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), 0);

  drm_amdgpu_gem_va unmap{};
  unmap.handle = gem_handle;
  unmap.operation = AMDGPU_VA_OP_UNMAP;
  unmap.va_address = kVa;
  unmap.map_size = kBoSize;
  unmap.vm_timeline_point = 2;
  unmap.vm_timeline_syncobj_out = create.handle;
  ASSERT_EQ(ioctl(drm, gem_va, &unmap), 0);
  points[0] = unmap.vm_timeline_point;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), 0);

  drm_syncobj_destroy destroy{};
  destroy.handle = create.handle;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), -1);
  EXPECT_EQ(errno, ENOENT);

  EXPECT_EQ(gem_close(drm, gem_handle), 0);
  EXPECT_EQ(close(dmabuf), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerSyncobjTest, VmDelayUpdateSuppressesTimelinePublication) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  drm_syncobj_create create{};
  ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &create), 0);
  constexpr size_t kBoSize = 0x1000;
  constexpr uint64_t kVa = 0x1000000000ULL;
  int dmabuf = make_sized_memfd(kBoSize);
  ASSERT_GE(dmabuf, 0);
  uint32_t gem_handle = 0;
  ASSERT_TRUE(prime_import(drm, dmabuf, &gem_handle));

  drm_amdgpu_gem_va update{};
  update.handle = gem_handle;
  update.operation = AMDGPU_VA_OP_MAP;
  update.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_DELAY_UPDATE;
  update.va_address = kVa;
  update.map_size = kBoSize;
  update.vm_timeline_point = 1;
  update.vm_timeline_syncobj_out = create.handle;
  ASSERT_EQ(ioctl(drm, DRM_AMDGPU_GEM_VA_request(), &update), 0);

  uint32_t handles[] = {create.handle};
  uint64_t points[] = {1};
  drm_syncobj_timeline_wait wait{};
  wait.handles = reinterpret_cast<uintptr_t>(handles);
  wait.points = reinterpret_cast<uintptr_t>(points);
  wait.count_handles = 1;
  wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), -1);
  EXPECT_EQ(errno, ETIME);

  update.operation = AMDGPU_VA_OP_UNMAP;
  update.flags = 0;
  ASSERT_EQ(ioctl(drm, DRM_AMDGPU_GEM_VA_request(), &update), 0);
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), 0);

  drm_syncobj_destroy destroy{};
  destroy.handle = create.handle;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);
  EXPECT_EQ(gem_close(drm, gem_handle), 0);
  EXPECT_EQ(close(dmabuf), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerSyncobjTest, GemVaSignalsBinarySyncobjsAtPointZero) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  std::array<drm_syncobj_create, 3> syncobjs{};
  for (auto &syncobj : syncobjs)
    ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &syncobj), 0);

  constexpr size_t kBoSize = 0x1000;
  constexpr uint64_t kVa = 0x1000000000ULL;
  int dmabuf = make_sized_memfd(kBoSize);
  ASSERT_GE(dmabuf, 0);
  uint32_t gem_handle = 0;
  ASSERT_TRUE(prime_import(drm, dmabuf, &gem_handle));

  const unsigned long gem_va = DRM_AMDGPU_GEM_VA_request();
  drm_amdgpu_gem_va update{};
  update.handle = gem_handle;
  update.operation = AMDGPU_VA_OP_MAP;
  update.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  update.va_address = kVa;
  update.map_size = kBoSize;
  update.vm_timeline_syncobj_out = syncobjs[0].handle;
  update.vm_timeline_point = 0;
  ASSERT_EQ(ioctl(drm, gem_va, &update), 0);

  auto expect_binary_signaled = [&](uint32_t handle) {
    uint32_t handles[] = {handle};
    uint64_t points[] = {0};
    drm_syncobj_timeline_wait wait{};
    wait.handles = reinterpret_cast<uintptr_t>(handles);
    wait.points = reinterpret_cast<uintptr_t>(points);
    wait.count_handles = 1;
    wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;
    EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), 0);
  };
  expect_binary_signaled(syncobjs[0].handle);

  update.operation = AMDGPU_VA_OP_UNMAP;
  update.flags = 0;
  update.vm_timeline_syncobj_out = syncobjs[1].handle;
  ASSERT_EQ(ioctl(drm, gem_va, &update), 0);
  expect_binary_signaled(syncobjs[1].handle);

  // A zero output handle still means no fence. Point zero alone must not change
  // that contract, and the mapping remains available for the following CLEAR.
  update.operation = AMDGPU_VA_OP_MAP;
  update.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  update.vm_timeline_syncobj_out = 0;
  ASSERT_EQ(ioctl(drm, gem_va, &update), 0);

  update.handle = 0;
  update.operation = AMDGPU_VA_OP_CLEAR;
  update.flags = 0;
  update.vm_timeline_syncobj_out = syncobjs[2].handle;
  ASSERT_EQ(ioctl(drm, gem_va, &update), 0);
  expect_binary_signaled(syncobjs[2].handle);

  for (const auto &syncobj : syncobjs) {
    drm_syncobj_destroy destroy{};
    destroy.handle = syncobj.handle;
    EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);
  }
  EXPECT_EQ(gem_close(drm, gem_handle), 0);
  EXPECT_EQ(close(dmabuf), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerSyncobjTest, PointZeroOutputReplacesTimelinePayload) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  drm_syncobj_create create{};
  ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &create), 0);
  constexpr size_t kBoSize = 0x1000;
  constexpr uint64_t kVa = 0x1000000000ULL;
  int dmabuf = make_sized_memfd(kBoSize);
  ASSERT_GE(dmabuf, 0);
  uint32_t gem_handle = 0;
  ASSERT_TRUE(prime_import(drm, dmabuf, &gem_handle));

  drm_amdgpu_gem_va update{};
  update.handle = gem_handle;
  update.operation = AMDGPU_VA_OP_MAP;
  update.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  update.va_address = kVa;
  update.map_size = kBoSize;
  update.vm_timeline_syncobj_out = create.handle;
  update.vm_timeline_point = 2;
  ASSERT_EQ(ioctl(drm, DRM_AMDGPU_GEM_VA_request(), &update), 0);

  uint32_t handles[] = {create.handle};
  uint64_t points[] = {2};
  drm_syncobj_timeline_wait wait{};
  wait.handles = reinterpret_cast<uintptr_t>(handles);
  wait.points = reinterpret_cast<uintptr_t>(points);
  wait.count_handles = 1;
  wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;
  ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), 0);

  update.operation = AMDGPU_VA_OP_UNMAP;
  update.flags = 0;
  update.vm_timeline_point = 0;
  ASSERT_EQ(ioctl(drm, DRM_AMDGPU_GEM_VA_request(), &update), 0);

  points[0] = 0;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), 0);
  points[0] = 2;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), -1);
  EXPECT_EQ(errno, EINVAL);

  drm_syncobj_destroy destroy{};
  destroy.handle = create.handle;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);
  EXPECT_EQ(gem_close(drm, gem_handle), 0);
  EXPECT_EQ(close(dmabuf), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerSyncobjTest, DuplicateDrmFdsShareNamespaceAndLifetime) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";
  int drm_dup = dup(drm);
  ASSERT_GE(drm_dup, 0);

  drm_syncobj_create create{};
  ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &create), 0);
  ASSERT_NE(create.handle, 0u);
  EXPECT_EQ(close(drm), 0);

  uint32_t handles[] = {create.handle};
  uint64_t points[] = {1};
  drm_syncobj_timeline_wait wait{};
  wait.handles = reinterpret_cast<uintptr_t>(handles);
  wait.points = reinterpret_cast<uintptr_t>(points);
  wait.count_handles = 1;
  wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;
  EXPECT_EQ(ioctl(drm_dup, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), -1);
  EXPECT_EQ(errno, EINVAL);

  drm_syncobj_destroy destroy{};
  destroy.handle = create.handle;
  EXPECT_EQ(ioctl(drm_dup, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);
  EXPECT_EQ(close(drm_dup), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerSyncobjTest, IndependentDrmOpensHaveSeparateNamespaces) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int first_drm = open_drm_render();
  int second_drm = open_drm_render();
  if (first_drm < 0 || second_drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  drm_syncobj_create create{};
  ASSERT_EQ(ioctl(first_drm, DRM_IOCTL_SYNCOBJ_CREATE, &create), 0);
  drm_syncobj_destroy destroy{};
  destroy.handle = create.handle;
  EXPECT_EQ(ioctl(second_drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), -1);
  EXPECT_EQ(errno, EINVAL);
  EXPECT_EQ(ioctl(first_drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);

  EXPECT_EQ(close(second_drm), 0);
  EXPECT_EQ(close(first_drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerSyncobjTest, CreateSignaledSeedsOnlyTheInitialPoint) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  drm_syncobj_create create{};
  create.flags = DRM_SYNCOBJ_CREATE_SIGNALED;
  ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &create), 0);
  uint32_t handles[] = {create.handle};
  uint64_t points[] = {0};
  drm_syncobj_timeline_wait wait{};
  wait.handles = reinterpret_cast<uintptr_t>(handles);
  wait.points = reinterpret_cast<uintptr_t>(points);
  wait.count_handles = 1;
  wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;
  wait.first_signaled = 17;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), 0);
  EXPECT_EQ(wait.first_signaled, UINT32_MAX);

  points[0] = 1;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), -1);
  EXPECT_EQ(errno, EINVAL);
  wait.flags |= DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), -1);
  EXPECT_EQ(errno, ETIME);
  wait.flags = 1u << 31;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), -1);
  EXPECT_EQ(errno, EINVAL);

  drm_syncobj_create second{};
  second.flags = DRM_SYNCOBJ_CREATE_SIGNALED;
  ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &second), 0);
  uint32_t any_handles[] = {create.handle, second.handle};
  uint64_t any_points[] = {1, 0};
  wait.handles = reinterpret_cast<uintptr_t>(any_handles);
  wait.points = reinterpret_cast<uintptr_t>(any_points);
  wait.count_handles = 2;
  wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT;
  wait.first_signaled = UINT32_MAX;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), 0);
  EXPECT_EQ(wait.first_signaled, 1u);

  drm_syncobj_destroy destroy{};
  destroy.handle = create.handle;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);
  destroy.handle = second.handle;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerSyncobjTest, WaitValidationMatchesDrm) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  drm_syncobj_timeline_wait empty{};
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &empty), 0);
  empty.pad = UINT32_MAX;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &empty), 0);
  empty.pad = 0;
  empty.flags = 1u << 31;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &empty), -1);
  EXPECT_EQ(errno, EINVAL);

  drm_syncobj_create create{};
  ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &create), 0);
  uint32_t handles[] = {create.handle};
  uint64_t points[] = {1};
  drm_syncobj_timeline_wait wait{};
  wait.handles = reinterpret_cast<uintptr_t>(handles);
  wait.points = reinterpret_cast<uintptr_t>(points);
  wait.count_handles = 1;
  wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), -1);
  EXPECT_EQ(errno, ETIME);

  uint32_t mixed_handles[] = {create.handle, UINT32_MAX};
  uint64_t mixed_points[] = {1, 0};
  wait.handles = reinterpret_cast<uintptr_t>(mixed_handles);
  wait.points = reinterpret_cast<uintptr_t>(mixed_points);
  wait.count_handles = 2;
  wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT;
  wait.first_signaled = 17;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), -1);
  EXPECT_EQ(errno, ENOENT);
  EXPECT_EQ(wait.first_signaled, 17u);

  wait.count_handles = 1;
  wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;
  wait.handles = 1;
  wait.points = reinterpret_cast<uintptr_t>(points);
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), -1);
  EXPECT_EQ(errno, EFAULT);

  wait.handles = reinterpret_cast<uintptr_t>(handles);
  wait.points = 1;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), -1);
  EXPECT_EQ(errno, EFAULT);

  drm_syncobj_destroy destroy{};
  destroy.handle = create.handle;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerSyncobjTest, DestroyRejectsInvalidInput) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  drm_syncobj_destroy destroy{};
  destroy.handle = UINT32_MAX;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), -1);
  EXPECT_EQ(errno, EINVAL);

  drm_syncobj_create create{};
  ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &create), 0);
  destroy.handle = create.handle;
  destroy.pad = 1;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), -1);
  EXPECT_EQ(errno, EINVAL);
  destroy.pad = 0;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);

  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerSyncobjTest, TimelineWaitBlocksUntilSignalOrDeadline) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  drm_syncobj_create create{};
  ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &create), 0);
  constexpr size_t kBoSize = 0x1000;
  constexpr uint64_t kVa = 0x1000000000ULL;
  int dmabuf = make_sized_memfd(kBoSize);
  ASSERT_GE(dmabuf, 0);
  uint32_t gem_handle = 0;
  ASSERT_TRUE(prime_import(drm, dmabuf, &gem_handle));

  uint32_t handles[] = {create.handle};
  uint64_t points[] = {1};
  drm_syncobj_timeline_wait wait{};
  wait.handles = reinterpret_cast<uintptr_t>(handles);
  wait.points = reinterpret_cast<uintptr_t>(points);
  wait.count_handles = 1;
  wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT;
  wait.timeout_nsec = monotonic_deadline_after(std::chrono::seconds(1));

  std::barrier ready(2);
  std::atomic<int> wait_rc{-2};
  std::atomic<int> wait_errno{0};
  std::thread waiter([&] {
    ready.arrive_and_wait();
    wait_rc = ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait);
    wait_errno = errno;
  });
  ready.arrive_and_wait();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const unsigned long gem_va = DRM_AMDGPU_GEM_VA_request();
  drm_amdgpu_gem_va map{};
  map.handle = gem_handle;
  map.operation = AMDGPU_VA_OP_MAP;
  map.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  map.va_address = kVa;
  map.map_size = kBoSize;
  map.vm_timeline_point = 1;
  map.vm_timeline_syncobj_out = create.handle;
  const int map_rc = ioctl(drm, gem_va, &map);
  waiter.join();
  ASSERT_EQ(map_rc, 0);
  EXPECT_EQ(wait_rc.load(), 0) << "wait errno=" << wait_errno.load();

  points[0] = 2;
  wait.timeout_nsec = monotonic_deadline_after(std::chrono::milliseconds(100));
  const auto timeout_start = std::chrono::steady_clock::now();
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), -1);
  const auto timeout_elapsed = std::chrono::steady_clock::now() - timeout_start;
  EXPECT_EQ(errno, ETIME);
  EXPECT_GE(timeout_elapsed, std::chrono::milliseconds(50));

  drm_amdgpu_gem_va unmap{};
  unmap.handle = gem_handle;
  unmap.operation = AMDGPU_VA_OP_UNMAP;
  unmap.va_address = kVa;
  unmap.map_size = kBoSize;
  EXPECT_EQ(ioctl(drm, gem_va, &unmap), 0);
  drm_syncobj_destroy destroy{};
  destroy.handle = create.handle;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);
  EXPECT_EQ(gem_close(drm, gem_handle), 0);
  EXPECT_EQ(close(dmabuf), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerSyncobjTest, TimelineWaitReturnsEintrForSignal) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  drm_syncobj_create create{};
  ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &create), 0);
  uint32_t handles[] = {create.handle};
  uint64_t points[] = {1};
  drm_syncobj_timeline_wait wait{};
  wait.handles = reinterpret_cast<uintptr_t>(handles);
  wait.points = reinterpret_cast<uintptr_t>(points);
  wait.count_handles = 1;
  wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT;
  wait.timeout_nsec = monotonic_deadline_after(std::chrono::seconds(2));

  struct sigaction action {};
  struct sigaction old_action {};
  action.sa_handler = noop_signal_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  ASSERT_EQ(sigaction(SIGUSR1, &action, &old_action), 0);

  std::atomic<bool> entered{false};
  std::atomic<int> wait_rc{-2};
  std::atomic<int> wait_errno{0};
  const auto start = std::chrono::steady_clock::now();
  std::thread waiter([&] {
    entered.store(true, std::memory_order_release);
    wait_rc = ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait);
    wait_errno = errno;
  });
  while (!entered.load(std::memory_order_acquire))
    std::this_thread::yield();
  const auto signal_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (wait_rc.load(std::memory_order_acquire) == -2 &&
         std::chrono::steady_clock::now() < signal_deadline) {
    EXPECT_EQ(pthread_kill(waiter.native_handle(), SIGUSR1), 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  waiter.join();
  const auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_EQ(wait_rc.load(), -1);
  EXPECT_EQ(wait_errno.load(), EINTR);
  EXPECT_LT(elapsed, std::chrono::seconds(1));
  EXPECT_EQ(sigaction(SIGUSR1, &old_action, nullptr), 0);

  drm_syncobj_destroy destroy{};
  destroy.handle = create.handle;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

// rocJITsu local mode supports fork-then-exec only: its simulator state lives in
// this address space, and a driver call holds private driver locks for its whole
// duration -- sometimes across a blocking wait -- so no atfork prepare handler could
// drain them without risking hanging fork() itself. Same contract as CUDA, HSA/ROCr
// and ThreadSanitizer. Enforcement is an immutable owner PID checked at the top of
// every interposed entry point, before any inherited lock, container or pointer.
//
// These cover the contract from a MULTITHREADED parent, which is the case that
// previously deadlocked: the child must reach exec (or _exit) without ever touching
// inherited state.

// A forked child must NOT silently fall through to the host's real GPU. If this box
// has a real /dev/kfd, passing the open through to libc would move the child off the
// emulator and onto real hardware -- worse than any error. It must fail closed.
TEST(InterposerForkTest, ChildRefusesGpuEndpointsBeforeExec) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));

  const pid_t child = fork();
  ASSERT_GE(child, 0) << "fork failed: " << std::strerror(errno);
  if (child == 0) {
    alarm(10);
    // Must fail, and must not hand back a real-hardware descriptor.
    int child_kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if (child_kfd >= 0)
      _exit(10); // opened something -- emulator or host -- both wrong here.
    if (errno != ENODEV)
      _exit(11);
    // Ordinary filesystem work still passes through normally.
    int devnull = open("/dev/null", O_RDONLY | O_CLOEXEC);
    if (devnull < 0)
      _exit(12);
    close(devnull);
    _exit(0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status)) << "child did not exit cleanly";
  EXPECT_EQ(WEXITSTATUS(status), 0)
      << "10=GPU endpoint opened in child, 11=wrong errno, 12=ordinary open broken";
  EXPECT_EQ(close(kfd), 0);
}

// A textual path check is not enough: /dev//kfd, /dev/./kfd, a cwd-relative
// openat(dirfd, "kfd", ...) and freopen() all reach the same device node by a
// different spelling. The child-side classifier resolves target IDENTITY instead, so
// every spelling must be refused. A leak here silently moves the child onto real
// hardware on any box that has a physical KFD.
TEST(InterposerForkTest, ChildRefusesEveryGpuEndpointSpelling) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));

  const pid_t child = fork();
  ASSERT_GE(child, 0) << "fork failed: " << std::strerror(errno);
  if (child == 0) {
    alarm(10);
    auto refused = [](int fd) {
      if (fd >= 0) {
        close(fd);
        return false;
      }
      return errno == ENODEV;
    };
    if (!refused(open("/dev/kfd", O_RDWR | O_CLOEXEC)))
      _exit(20);
    if (!refused(open("/dev//kfd", O_RDWR | O_CLOEXEC)))
      _exit(21);
    if (!refused(open("/dev/./kfd", O_RDWR | O_CLOEXEC)))
      _exit(22);
    int devdir = open("/dev", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (devdir < 0)
      _exit(23);
    if (!refused(openat(devdir, "kfd", O_RDWR | O_CLOEXEC)))
      _exit(24);
    close(devdir);
    // A refused freopen must still CLOSE the caller's original stream. POSIX freopen
    // closes it first and does so even when opening the replacement fails, so
    // returning ENODEV while leaving it live would keep a descriptor alive past a
    // failed freopen -- and past the exec that follows. Use a throwaway stream rather
    // than stdin so the check is observable.
    FILE *victim = fopen("/dev/null", "r");
    if (!victim)
      _exit(25);
    const int victim_fd = fileno(victim);
    errno = 0;
    if (freopen("/dev/kfd", "r", victim) != nullptr)
      _exit(26);
    if (errno != ENODEV)
      _exit(27);
    if (fcntl(victim_fd, F_GETFD) >= 0)
      _exit(28); // original descriptor outlived a failed freopen.
    // Ordinary character devices must still work -- the classifier refuses GPU
    // nodes, not every device.
    int devnull = open("/dev/null", O_RDONLY | O_CLOEXEC);
    if (devnull < 0)
      _exit(29);
    close(devnull);
    _exit(0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0)
      << "20-24=a GPU endpoint spelling was not refused, 25=setup, 26/27=freopen "
      << "leaked, 28=refused freopen left the original stream open, 29=ordinary "
      << "device open broken";
  EXPECT_EQ(close(kfd), 0);
}

// Both tests above only prove anything on a box that HAS the device node they name:
// the child-side classifier resolved identity by stat()ing the target, so on a host
// with no /dev/kfd and no render nodes -- CI, and any pure-emulation deployment --
// it classified nothing and every spelling fell through to libc with ENOENT. The
// parent does not work that way: it synthesizes GPU endpoints by spelling, so it
// serves an emulated endpoint whether or not the host has a matching node, and the
// child's contract must not quietly depend on the box having hardware.
//
// Pin that with an endpoint this host does NOT have, which reaches the same branch
// on every box rather than only on bare ones.
TEST(InterposerForkTest, ChildRefusesAnEmulatedEndpointTheHostDoesNotHave) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));

  // First render minor with no node on this host. DRM allocates them from 128.
  std::string absent;
  for (unsigned minor = 128; minor < 256; ++minor) {
    std::string candidate = "/dev/dri/renderD" + std::to_string(minor);
    struct stat st {};
    if (stat(candidate.c_str(), &st) != 0) {
      absent = std::move(candidate);
      break;
    }
  }
  if (absent.empty())
    GTEST_SKIP() << "every render minor 128-255 exists on this host";

  const std::string node = absent.substr(absent.rfind('/') + 1);
  // The same node reached by every spelling ChildRefusesEveryGpuEndpointSpelling
  // uses, so the absent-node path is held to the same contract as the present one.
  // With no node to resolve, the classifier has to normalize these itself, and that
  // normalization is only reachable here.
  const std::string doubled = "/dev/dri//" + node;
  const std::string dotted = "/dev/dri/./" + node;
  const std::string parented = "/dev/dri/../dri/" + node;

  const pid_t child = fork();
  ASSERT_GE(child, 0) << "fork failed: " << std::strerror(errno);
  if (child == 0) {
    alarm(10);
    auto refused = [](int fd) {
      if (fd >= 0) {
        close(fd);
        return false;
      }
      return errno == ENODEV;
    };
    if (!refused(open(absent.c_str(), O_RDWR | O_CLOEXEC)))
      _exit(30);
    if (!refused(open(doubled.c_str(), O_RDWR | O_CLOEXEC)))
      _exit(31);
    if (!refused(open(dotted.c_str(), O_RDWR | O_CLOEXEC)))
      _exit(32);
    if (!refused(open(parented.c_str(), O_RDWR | O_CLOEXEC)))
      _exit(33);
    int dridir = open("/dev/dri", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dridir >= 0) {
      // Relative to a dirfd there is no absolute name to inspect at all, so this
      // is the spelling that proves the classifier resolves rather than matches.
      if (!refused(openat(dridir, node.c_str(), O_RDWR | O_CLOEXEC)))
        _exit(34);
      close(dridir);
    }
    // A path that merely does not exist must still report ENOENT: the fallback
    // matches the endpoints the parent synthesizes, not everything absent.
    if (open("/dev/dri/definitely-not-a-render-node", O_RDWR | O_CLOEXEC) >= 0)
      _exit(35);
    if (errno != ENOENT)
      _exit(36);
    _exit(0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status)) << "child did not exit cleanly";
  EXPECT_EQ(WEXITSTATUS(status), 0)
      << "30-34=an absent GPU endpoint spelling was not refused with ENODEV, "
      << "35/36=an ordinary missing path stopped reporting ENOENT";
  EXPECT_EQ(close(kfd), 0);
}

// The classifier's stated rule is that an unclassifiable target is treated AS a GPU
// endpoint -- a child getting ENODEV on some unrelated node costs far less than a
// child silently acquiring real hardware. Both classifiers have a give-up path, and
// a give-up that answered "not a GPU endpoint" would be a silent fail-OPEN.
//
// A path too long to resolve reaches exactly that: no node to stat, and no absolute
// spelling to compare, so neither classifier can decide. libc alone would answer
// ENAMETOOLONG; the contract says refuse instead.
TEST(InterposerForkTest, ChildFailsClosedOnAnUnclassifiablePath) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));

  // Well past PATH_MAX, so resolution cannot produce a comparable absolute path.
  const std::string overlong = "/" + std::string(PATH_MAX + 64, 'x');

  const pid_t child = fork();
  ASSERT_GE(child, 0) << "fork failed: " << std::strerror(errno);
  if (child == 0) {
    alarm(10);
    int fd = open(overlong.c_str(), O_RDWR | O_CLOEXEC);
    if (fd >= 0) {
      close(fd);
      _exit(40);
    }
    if (errno != ENODEV)
      _exit(41);
    // ...and a path that IS classifiable still reports its own error, so failing
    // closed has not swallowed ordinary errno reporting.
    if (open("/tmp/rocjitsu-no-such-file-here", O_RDONLY | O_CLOEXEC) >= 0)
      _exit(42);
    if (errno != ENOENT)
      _exit(43);
    _exit(0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status)) << "child did not exit cleanly";
  EXPECT_EQ(WEXITSTATUS(status), 0)
      << "40=an unclassifiable path was opened, 41=it was not refused with ENODEV "
      << "(fail-open), 42/43=a classifiable missing path stopped reporting ENOENT";
  EXPECT_EQ(close(kfd), 0);
}

// The descriptor an application receives must carry NO real-hardware authority.
// Under fork-then-exec the interposer passes a child's ioctl/dup/dup2/fcntl straight
// to libc, so if the app fd were a real /dev/kfd duplicate a child could drive real
// hardware through it -- and dup2() it to a fresh number, clearing FD_CLOEXEC, to
// carry that authority across the exec that was supposed to sanitize the process.
// Both drivers therefore hand out synthetic descriptors and keep any real device fd
// strictly private.
TEST(InterposerForkTest, AppDescriptorCarriesNoRealHardwareAuthority) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));

  auto is_real_gpu = [](int fd) {
    struct stat st {};
    if (::fstat(fd, &st) != 0 || !S_ISCHR(st.st_mode))
      return false;
    struct stat kfd_st {};
    if (::stat("/dev/kfd", &kfd_st) == 0 && st.st_rdev == kfd_st.st_rdev)
      return true;
    return major(st.st_rdev) == 226u;
  };

  EXPECT_FALSE(is_real_gpu(kfd))
      << "the application-facing KFD descriptor must be synthetic, not a real device";

  const pid_t child = fork();
  ASSERT_GE(child, 0) << "fork failed: " << std::strerror(errno);
  if (child == 0) {
    alarm(10);
    // Laundering the inherited descriptor must not produce hardware authority.
    int laundered = dup2(kfd, 200);
    if (laundered >= 0) {
      struct stat st {};
      if (::fstat(200, &st) == 0 && S_ISCHR(st.st_mode)) {
        struct stat kfd_st {};
        if ((::stat("/dev/kfd", &kfd_st) == 0 && st.st_rdev == kfd_st.st_rdev) ||
            major(st.st_rdev) == 226u)
          _exit(40); // real GPU authority survived into the child.
      }
    }
    _exit(0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0)
      << "40=a real GPU descriptor was inherited and laundered across dup2";
  EXPECT_EQ(close(kfd), 0);
}

// Making the APPLICATION-facing descriptor synthetic is not enough on its own:
// fork() copies the whole descriptor table, so a hardware-backed guest's PRIVATE
// real /dev/kfd is inherited too, and a child can find it by walking /proc/self/fd.
// Child pass-through paths therefore check descriptor IDENTITY, refusing operations
// on any real GPU fd rather than trusting the path used to obtain it.
//
// This is a cooperative, API-level contract, not a security boundary -- a child
// issuing raw syscalls is out of reach of interposition -- but it stops the
// realistic failure: inheriting hardware by accident and laundering it across exec.
TEST(InterposerForkTest, ChildRefusesOperationsOnInheritedRealGpuDescriptors) {
  // card0 is a real DRM device the interposer does not synthesize, so the parent
  // genuinely hands real GPU authority to the child -- the same shape as a hardware
  // guest's private KFD descriptor.
  int real_gpu = ::open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
  if (real_gpu < 0)
    GTEST_SKIP() << "no real DRM device available to inherit";

  const pid_t child = fork();
  ASSERT_GE(child, 0) << "fork failed: " << std::strerror(errno);
  if (child == 0) {
    alarm(10);
    auto refused = [](int rc) { return rc < 0 && errno == ENODEV; };
    if (!refused(ioctl(real_gpu, 0, 0)))
      _exit(50); // operated real hardware through an inherited descriptor.
    if (!refused(dup(real_gpu)))
      _exit(51);
    if (!refused(dup2(real_gpu, 210)))
      _exit(52); // dup2 also strips FD_CLOEXEC, so this is the exec-laundering route.
    if (!refused(fcntl(real_gpu, F_DUPFD_CLOEXEC, 0)))
      _exit(53);
    _exit(0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0)
      << "50=ioctl, 51=dup, 52=dup2, 53=F_DUPFD each allowed on an inherited real "
      << "GPU descriptor";
  EXPECT_EQ(close(real_gpu), 0);
}

// Duplication is not the only way to launder an inherited descriptor past exec:
// clearing FD_CLOEXEC on the descriptor already held does it with no dup at all.
// The child gate must cover both, while leaving harmless fcntl use working.
TEST(InterposerForkTest, ChildCannotLaunderInheritedGpuFdAcrossExec) {
  int real_gpu = ::open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
  if (real_gpu < 0)
    GTEST_SKIP() << "no real DRM device available to inherit";

  const pid_t child = fork();
  ASSERT_GE(child, 0) << "fork failed: " << std::strerror(errno);
  if (child == 0) {
    alarm(10);
    // Clearing FD_CLOEXEC is the no-duplication laundering route.
    if (!(fcntl(real_gpu, F_SETFD, 0) < 0 && errno == ENODEV))
      _exit(60);
    // Setting it is harmless and must still be permitted.
    if (fcntl(real_gpu, F_SETFD, FD_CLOEXEC) != 0)
      _exit(61);
    // Ordinary queries must keep working.
    if (fcntl(real_gpu, F_GETFD) < 0)
      _exit(62);
    _exit(0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0)
      << "60=F_SETFD cleared FD_CLOEXEC on a real GPU fd, 61=setting FD_CLOEXEC was "
      << "wrongly refused, 62=an ordinary fcntl query was wrongly refused";
  EXPECT_EQ(close(real_gpu), 0);
}

// dup3 is gated on the same path as dup2 but is worth its own case: it takes an
// explicit flags argument, so the laundering shape (flags without O_CLOEXEC) is
// expressed differently, and a future refactor could plausibly gate one and not the
// other. Both flag forms must be refused on a real GPU fd -- a child has no business
// operating real hardware whether or not the copy would survive exec -- while dup3
// on an ordinary descriptor must keep working.
TEST(InterposerForkTest, ChildRefusesDup3OnInheritedGpuFd) {
  int real_gpu = ::open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
  if (real_gpu < 0)
    GTEST_SKIP() << "no real DRM device available to inherit";
  int ordinary = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
  ASSERT_GE(ordinary, 0);

  const pid_t child = fork();
  ASSERT_GE(child, 0) << "fork failed: " << std::strerror(errno);
  if (child == 0) {
    alarm(10);
    // Without O_CLOEXEC the copy would survive exec -- the laundering route.
    if (!(dup3(real_gpu, 220, 0) < 0 && errno == ENODEV))
      _exit(90);
    // With O_CLOEXEC it would not survive exec, but the child still must not get a
    // usable real-hardware descriptor.
    if (!(dup3(real_gpu, 221, O_CLOEXEC) < 0 && errno == ENODEV))
      _exit(91);
    // Ordinary descriptors are unaffected.
    if (dup3(ordinary, 222, O_CLOEXEC) < 0)
      _exit(92);
    _exit(0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0)
      << "90=dup3 laundered a real GPU fd, 91=dup3 with O_CLOEXEC still handed one "
      << "over, 92=dup3 on an ordinary fd was wrongly refused";
  EXPECT_EQ(close(ordinary), 0);
  EXPECT_EQ(close(real_gpu), 0);
}

// mmap64 is a distinct exported symbol, and _FILE_OFFSET_BITS=64 redirects even a
// source-level mmap() call to it. Exporting only mmap would let those calls bypass
// the child refusal and all emulator routing, exactly as would have happened for
// fcntl64. Call it explicitly so the test cannot be silently re-bound.
extern "C" void *mmap64(void *, size_t, int, int, int, __off64_t);

TEST(InterposerForkTest, ChildRefusesMmap64OnInheritedGpuFd) {
  int real_gpu = ::open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
  if (real_gpu < 0)
    GTEST_SKIP() << "no real DRM device available to inherit";

  const pid_t child = fork();
  ASSERT_GE(child, 0) << "fork failed: " << std::strerror(errno);
  if (child == 0) {
    alarm(10);
    void *p = mmap64(nullptr, 4096, PROT_READ, MAP_SHARED, real_gpu, 0);
    if (!(p == MAP_FAILED && errno == ENODEV))
      _exit(70);
    _exit(0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0) << "70=mmap64 bypassed the child GPU-fd refusal";
  EXPECT_EQ(close(real_gpu), 0);
}

// The child's fopen must keep libc's own mode parser. An earlier version re-derived
// open flags and used fdopen, which silently lost the 0666 creation mode for "w"/"a"
// (files came out mode 0000) and the "x" exclusive modifier (truncating instead of
// failing EEXIST).
TEST(InterposerForkTest, ChildFopenPreservesLibcModeSemantics) {
  rocjitsu::test::ScopedTempDirectory tmp("rj_fopen_");
  const std::string target = tmp.path() + "/created";

  const pid_t child = fork();
  ASSERT_GE(child, 0) << "fork failed: " << std::strerror(errno);
  if (child == 0) {
    alarm(10);
    // Pin the umask so the expected mode is exact. Without this the assertion has to
    // weaken to "some bit is set", which would accept genuinely wrong modes -- and
    // would wrongly fail under an ambient umask of 0777, where 0000 is correct.
    ::umask(0);

    FILE *w = fopen(target.c_str(), "w");
    if (!w)
      _exit(80);
    fclose(w);
    struct stat st {};
    if (::stat(target.c_str(), &st) != 0)
      _exit(81);
    if ((st.st_mode & 07777) != 0666)
      _exit(82); // must be exactly 0666 & ~umask(0); mode 0000 was the old bug.

    // Seed content so "wx" failing is distinguishable from "wx" truncating and then
    // failing -- a null return alone does not prove the file was left alone.
    FILE *seed = fopen(target.c_str(), "w");
    if (!seed)
      _exit(83);
    fputs("SENTINEL", seed);
    fclose(seed);

    errno = 0;
    FILE *excl = fopen(target.c_str(), "wx");
    const int excl_errno = errno;
    if (excl != nullptr) {
      fclose(excl);
      _exit(84); // "x" must refuse an existing file.
    }
    if (excl_errno != EEXIST)
      _exit(85);
    char buf[32] = {};
    FILE *rd = fopen(target.c_str(), "r");
    if (!rd)
      _exit(86);
    const char *got = fgets(buf, sizeof(buf), rd);
    fclose(rd);
    if (!got || std::strcmp(buf, "SENTINEL") != 0)
      _exit(87); // truncated despite failing: the destructive-then-fail shape.
    _exit(0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0)
      << "80/81=fopen(\"w\") failed, 82=creation mode was not 0666 under umask(0), "
      << "83=seeding failed, 84=\"wx\" opened an existing file, 85=wrong errno, "
      << "86/87=\"wx\" truncated the file before failing";
}

// Names lie; device identity does not. A symlink can point at the real KFD under any
// name, a chain can hide it another level down, and an unrelated node can be *named*
// like KFD without being one. The child classifier resolves the target and compares
// st_rdev against the host KFD recorded at init, so all three must come out right --
// including the reverse case, which a substring check would wrongly refuse.
TEST(InterposerForkTest, ChildClassifiesGpuEndpointsByIdentityNotName) {
  struct stat kfd_st {};
  if (::stat("/dev/kfd", &kfd_st) != 0 || !S_ISCHR(kfd_st.st_mode))
    GTEST_SKIP() << "host has no real /dev/kfd to alias";

  rocjitsu::test::ScopedTempDirectory tmp("rj_alias_");
  const std::string aliased = tmp.path() + "/gpu";          // -> /dev/kfd
  const std::string chained = tmp.path() + "/chained";      // -> gpu -> /dev/kfd
  const std::string lookalike = tmp.path() + "/kfd_shaped"; // -> /dev/null
  ASSERT_EQ(symlink("/dev/kfd", aliased.c_str()), 0);
  ASSERT_EQ(symlink("gpu", chained.c_str()), 0);
  ASSERT_EQ(symlink("/dev/null", lookalike.c_str()), 0);

  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));

  const pid_t child = fork();
  ASSERT_GE(child, 0) << "fork failed: " << std::strerror(errno);
  if (child == 0) {
    alarm(10);
    auto refused = [](const std::string &p) {
      int fd = open(p.c_str(), O_RDWR | O_CLOEXEC);
      if (fd >= 0) {
        close(fd);
        return false;
      }
      return errno == ENODEV;
    };
    if (!refused(aliased))
      _exit(30); // differently named alias leaked real hardware.
    if (!refused(chained))
      _exit(31); // chained alias leaked real hardware.
    int ok = open(lookalike.c_str(), O_RDONLY | O_CLOEXEC);
    if (ok < 0)
      _exit(32); // a node merely NAMED like KFD must not be refused.
    close(ok);
    _exit(0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0)
      << "30=named alias leaked, 31=chained alias leaked, 32=false positive on a "
      << "non-GPU node whose name resembles KFD";
  EXPECT_EQ(close(kfd), 0);
}

// The case that used to deadlock: fork racing threads that are inside interposed
// calls. Under the PID gate the child touches nothing inherited, so it must always
// reach exec. The alarm in the child is the liveness assertion.
TEST(InterposerForkTest, MultithreadedForkExecAlwaysReachesExec) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));

  std::atomic<bool> stop{false};
  std::vector<std::thread> churn;
  for (int i = 0; i < 4; ++i) {
    churn.emplace_back([&] {
      while (!stop.load(std::memory_order_acquire)) {
        int d = dup(kfd);
        if (d >= 0)
          close(d);
      }
    });
  }
  for (int i = 0; i < 2; ++i) {
    churn.emplace_back([&] {
      while (!stop.load(std::memory_order_acquire)) {
        struct stat st {};
        stat("/sys/class/kfd/kfd/topology/nodes/0/properties", &st);
      }
    });
  }

  for (int round = 0; round < 24; ++round) {
    const pid_t child = fork();
    ASSERT_GE(child, 0) << "fork failed: " << std::strerror(errno);
    if (child == 0) {
      alarm(10);
      execl("/bin/true", "true", static_cast<char *>(nullptr));
      _exit(127); // exec failed
    }
    int status = 0;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status))
        << "round " << round << ": child never reached exec (a fork racing an "
        << "interposed call must not be able to block on inherited state)";
    EXPECT_EQ(WEXITSTATUS(status), 0) << "round " << round;
  }

  stop.store(true, std::memory_order_release);
  for (auto &t : churn)
    t.join();
  EXPECT_EQ(close(kfd), 0);
}

// system() and popen() fork internally without going through our exported fork(),
// so they exercise the same child path via a route we do not control.
TEST(InterposerForkTest, SystemAndPopenWorkFromAMultithreadedParent) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));

  std::atomic<bool> stop{false};
  std::vector<std::thread> churn;
  for (int i = 0; i < 4; ++i) {
    churn.emplace_back([&] {
      while (!stop.load(std::memory_order_acquire)) {
        int d = dup(kfd);
        if (d >= 0)
          close(d);
      }
    });
  }

  for (int round = 0; round < 8; ++round) {
    EXPECT_EQ(system("/bin/true"), 0) << "round " << round;
    FILE *pipe = popen("/bin/echo rocjitsu", "r");
    ASSERT_NE(pipe, nullptr) << "round " << round;
    char buf[64] = {};
    EXPECT_NE(fgets(buf, sizeof(buf), pipe), nullptr);
    EXPECT_STREQ(buf, "rocjitsu\n");
    EXPECT_EQ(pclose(pipe), 0) << "round " << round;
  }

  stop.store(true, std::memory_order_release);
  for (auto &t : churn)
    t.join();
  EXPECT_EQ(close(kfd), 0);
}

// posix_spawn uses vfork/CLONE_VFORK internally, where the child shares the parent's
// address space until exec. The gate must not mutate anything in that window.
TEST(InterposerForkTest, PosixSpawnFromAMultithreadedParent) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));

  std::atomic<bool> stop{false};
  std::vector<std::thread> churn;
  for (int i = 0; i < 4; ++i) {
    churn.emplace_back([&] {
      while (!stop.load(std::memory_order_acquire)) {
        int d = dup(kfd);
        if (d >= 0)
          close(d);
      }
    });
  }

  for (int round = 0; round < 8; ++round) {
    pid_t spawned = -1;
    char prog[] = "/bin/true";
    char *const argv[] = {prog, nullptr};
    ASSERT_EQ(posix_spawn(&spawned, prog, nullptr, nullptr, argv, environ), 0) << "round " << round;
    int status = 0;
    ASSERT_EQ(waitpid(spawned, &status, 0), spawned);
    EXPECT_TRUE(WIFEXITED(status)) << "round " << round;
    EXPECT_EQ(WEXITSTATUS(status), 0) << "round " << round;
  }

  stop.store(true, std::memory_order_release);
  for (auto &t : churn)
    t.join();
  // The parent's own KFD routing must be untouched by all that spawning.
  EXPECT_TRUE(kfd_version_ok(kfd));
  EXPECT_EQ(close(kfd), 0);
}

// RETIRED: this exercised fork-WITHOUT-exec in local mode -- the child re-opened
// KFD/DRM and ran GEM + syncobj work with no intervening exec. That contradicts the
// contract local mode can actually honour (see InterposerForkTest above): the
// simulator lives in this address space and its driver locks cannot be drained by an
// atfork handler. The capability is only supportable where the state is NOT in the
// forking address space, i.e. daemon mode -- and re-homing it there needs its own
// design first, because a daemon-mode child still inherits the client-side context,
// RemoteDriver, fd maps and shared_ptr control blocks. Being out-of-process removes
// the simulator from the child, not the proxy. Tracked separately; do not simply
// move this test.

TEST(InterposerSyncobjTest, ConcurrentVmUpdatesShareTimeline) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  drm_syncobj_create create{};
  ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &create), 0);
  constexpr size_t kBoSize = 0x1000;
  const std::array<uint64_t, 2> vas = {0x1000000000ULL, 0x1000100000ULL};
  std::array<int, 2> dmabufs = {make_sized_memfd(kBoSize), make_sized_memfd(kBoSize)};
  ASSERT_GE(dmabufs[0], 0);
  ASSERT_GE(dmabufs[1], 0);
  std::array<uint32_t, 2> gem_handles{};
  ASSERT_TRUE(prime_import(drm, dmabufs[0], &gem_handles[0]));
  ASSERT_TRUE(prime_import(drm, dmabufs[1], &gem_handles[1]));

  const unsigned long gem_va = DRM_AMDGPU_GEM_VA_request();
  std::atomic<uint64_t> next_point{0};
  std::array<int, 2> results = {-1, -1};
  std::barrier start(3);
  auto submit = [&](size_t index) {
    drm_amdgpu_gem_va map{};
    map.handle = gem_handles[index];
    map.operation = AMDGPU_VA_OP_MAP;
    map.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
    map.va_address = vas[index];
    map.map_size = kBoSize;
    map.vm_timeline_point = next_point.fetch_add(1) + 1;
    map.vm_timeline_syncobj_out = create.handle;
    start.arrive_and_wait();
    results[index] = ioctl(drm, gem_va, &map);
  };
  std::thread first(submit, 0);
  std::thread second(submit, 1);
  start.arrive_and_wait();
  first.join();
  second.join();
  EXPECT_EQ(results[0], 0);
  EXPECT_EQ(results[1], 0);

  uint32_t handles[] = {create.handle};
  uint64_t points[] = {2};
  drm_syncobj_timeline_wait wait{};
  wait.handles = reinterpret_cast<uintptr_t>(handles);
  wait.points = reinterpret_cast<uintptr_t>(points);
  wait.count_handles = 1;
  wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), 0);

  for (size_t i = 0; i < gem_handles.size(); ++i) {
    drm_amdgpu_gem_va unmap{};
    unmap.handle = gem_handles[i];
    unmap.operation = AMDGPU_VA_OP_UNMAP;
    unmap.va_address = vas[i];
    unmap.map_size = kBoSize;
    EXPECT_EQ(ioctl(drm, gem_va, &unmap), 0);
    EXPECT_EQ(gem_close(drm, gem_handles[i]), 0);
    EXPECT_EQ(close(dmabufs[i]), 0);
  }
  drm_syncobj_destroy destroy{};
  destroy.handle = create.handle;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerSyncobjTest, OutOfOrderTimelinePointPreservesWatermark) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  drm_syncobj_create create{};
  ASSERT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &create), 0);
  constexpr size_t kBoSize = 0x1000;
  constexpr uint64_t kVa = 0x1000000000ULL;
  int dmabuf = make_sized_memfd(kBoSize);
  ASSERT_GE(dmabuf, 0);
  uint32_t gem_handle = 0;
  ASSERT_TRUE(prime_import(drm, dmabuf, &gem_handle));

  const unsigned long gem_va = DRM_AMDGPU_GEM_VA_request();
  drm_amdgpu_gem_va map{};
  map.handle = gem_handle;
  map.operation = AMDGPU_VA_OP_MAP;
  map.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  map.va_address = kVa;
  map.map_size = kBoSize;
  map.vm_timeline_point = 2;
  map.vm_timeline_syncobj_out = create.handle;
  ASSERT_EQ(ioctl(drm, gem_va, &map), 0);

  drm_amdgpu_gem_va unmap{};
  unmap.handle = gem_handle;
  unmap.operation = AMDGPU_VA_OP_UNMAP;
  unmap.va_address = kVa;
  unmap.map_size = kBoSize;
  unmap.vm_timeline_point = 1;
  unmap.vm_timeline_syncobj_out = create.handle;
  EXPECT_EQ(ioctl(drm, gem_va, &unmap), 0);

  uint32_t handles[] = {create.handle};
  uint64_t points[] = {2};
  drm_syncobj_timeline_wait wait{};
  wait.handles = reinterpret_cast<uintptr_t>(handles);
  wait.points = reinterpret_cast<uintptr_t>(points);
  wait.count_handles = 1;
  wait.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait), 0);

  drm_syncobj_destroy destroy{};
  destroy.handle = create.handle;
  EXPECT_EQ(ioctl(drm, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy), 0);
  EXPECT_EQ(gem_close(drm, gem_handle), 0);
  EXPECT_EQ(close(dmabuf), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerGemTest, GemNamespaceIsPrivateButSharedByDuplicates) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int owner = open_drm_render();
  int foreign = open_drm_render();
  if (owner < 0 || foreign < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";
  int owner_dup = dup(owner);
  ASSERT_GE(owner_dup, 0);

  constexpr size_t kBoSize = 0x1000;
  constexpr uint64_t kVa = 0x1000000000ULL;
  int dmabuf = make_sized_memfd(kBoSize);
  ASSERT_GE(dmabuf, 0);
  uint32_t gem_handle = 0;
  ASSERT_TRUE(prime_import(owner, dmabuf, &gem_handle));

  const unsigned long gem_va = DRM_AMDGPU_GEM_VA_request();
  drm_amdgpu_gem_va update{};
  update.handle = gem_handle;
  update.operation = AMDGPU_VA_OP_MAP;
  update.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  update.va_address = kVa;
  update.map_size = kBoSize;

  EXPECT_EQ(ioctl(foreign, gem_va, &update), -1);
  EXPECT_EQ(errno, ENOENT);
  update.operation = AMDGPU_VA_OP_REPLACE;
  EXPECT_EQ(ioctl(foreign, gem_va, &update), -1);
  EXPECT_EQ(errno, ENOENT);

  // A duplicate resolves to the same DRM file and can use the imported handle.
  update.operation = AMDGPU_VA_OP_MAP;
  ASSERT_EQ(ioctl(owner_dup, gem_va, &update), 0);

  update.operation = AMDGPU_VA_OP_UNMAP;
  EXPECT_EQ(ioctl(foreign, gem_va, &update), -1);
  EXPECT_EQ(errno, ENOENT);
  ASSERT_EQ(ioctl(owner, gem_va, &update), 0);

  update.operation = AMDGPU_VA_OP_MAP;
  ASSERT_EQ(ioctl(owner, gem_va, &update), 0);
  drm_amdgpu_gem_va clear{};
  clear.operation = AMDGPU_VA_OP_CLEAR;
  clear.va_address = kVa;
  clear.map_size = kBoSize;
  EXPECT_EQ(ioctl(foreign, gem_va, &clear), -1);
  EXPECT_EQ(errno, EINVAL);
  update.operation = AMDGPU_VA_OP_UNMAP;
  ASSERT_EQ(ioctl(owner, gem_va, &update), 0)
      << "a foreign CLEAR must not remove the owner's mapping";

  update.operation = AMDGPU_VA_OP_MAP;
  ASSERT_EQ(ioctl(owner, gem_va, &update), 0);
  EXPECT_EQ(gem_close(foreign, gem_handle), -1);
  EXPECT_EQ(errno, ENOENT);
  update.operation = AMDGPU_VA_OP_UNMAP;
  ASSERT_EQ(ioctl(owner, gem_va, &update), 0)
      << "a foreign GEM_CLOSE must not destroy the owner's handle";

  EXPECT_EQ(gem_close(owner_dup, gem_handle), 0);
  EXPECT_EQ(close(dmabuf), 0);
  EXPECT_EQ(close(owner_dup), 0);
  EXPECT_EQ(close(foreign), 0);
  EXPECT_EQ(close(owner), 0);
  EXPECT_EQ(close(kfd), 0);
}

TEST(InterposerGemTest, IndependentDrmFilesRejectOverlappingMappings) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int first_drm = open_drm_render();
  int second_drm = open_drm_render();
  if (first_drm < 0 || second_drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  constexpr size_t kBoSize = 0x1000;
  constexpr uint64_t kVa = 0x1000000000ULL;
  int first_dmabuf = make_sized_memfd(kBoSize);
  int second_dmabuf = make_sized_memfd(kBoSize);
  ASSERT_GE(first_dmabuf, 0);
  ASSERT_GE(second_dmabuf, 0);
  uint32_t first_handle = 0;
  uint32_t second_handle = 0;
  ASSERT_TRUE(prime_import(first_drm, first_dmabuf, &first_handle));
  ASSERT_TRUE(prime_import(second_drm, second_dmabuf, &second_handle));

  const unsigned long gem_va = DRM_AMDGPU_GEM_VA_request();
  drm_amdgpu_gem_va first_map{};
  first_map.handle = first_handle;
  first_map.operation = AMDGPU_VA_OP_MAP;
  first_map.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  first_map.va_address = kVa;
  first_map.map_size = kBoSize;
  ASSERT_EQ(ioctl(first_drm, gem_va, &first_map), 0);

  drm_amdgpu_gem_va second_map = first_map;
  second_map.handle = second_handle;
  EXPECT_EQ(ioctl(second_drm, gem_va, &second_map), -1);
  EXPECT_EQ(errno, EINVAL);
  second_map.operation = AMDGPU_VA_OP_REPLACE;
  EXPECT_EQ(ioctl(second_drm, gem_va, &second_map), -1);
  EXPECT_EQ(errno, EINVAL);

  drm_amdgpu_gem_va first_unmap = first_map;
  first_unmap.operation = AMDGPU_VA_OP_UNMAP;
  first_unmap.flags = 0;
  ASSERT_EQ(ioctl(first_drm, gem_va, &first_unmap), 0)
      << "rejected foreign updates must leave the first mapping intact";

  // Once the first file releases the shared VA, the second file may claim it.
  second_map.operation = AMDGPU_VA_OP_MAP;
  ASSERT_EQ(ioctl(second_drm, gem_va, &second_map), 0);
  drm_amdgpu_gem_va second_unmap = second_map;
  second_unmap.operation = AMDGPU_VA_OP_UNMAP;
  second_unmap.flags = 0;
  EXPECT_EQ(ioctl(second_drm, gem_va, &second_unmap), 0);

  EXPECT_EQ(gem_close(second_drm, second_handle), 0);
  EXPECT_EQ(gem_close(first_drm, first_handle), 0);
  EXPECT_EQ(close(second_dmabuf), 0);
  EXPECT_EQ(close(first_dmabuf), 0);
  EXPECT_EQ(close(second_drm), 0);
  EXPECT_EQ(close(first_drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

// A dmabuf export fd number that is closed and then recycled by a second export
// must resolve to a DISTINCT, stable GEM handle — never one derived from the fd
// number. Two concurrently-live BOs whose export fds happened to reuse the same
// integer must keep independent handles and independent GPU mappings, and closing
// one handle must not disturb the other. This pins the fix for the old
// handle = dmabuf_fd + 1 scheme, under which a recycled fd tore down a live BO.
TEST(InterposerGemTest, ReusedDmabufFdMintsDistinctHandles) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));

  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  constexpr size_t kBoSize = 0x1000;
  const unsigned long gem_va = DRM_AMDGPU_GEM_VA_request();
  const uint64_t va_a = 0x1000000000ULL;
  const uint64_t va_b = 0x1000100000ULL;

  // First BO: import and map it (the export fd must stay open across MAP, which
  // lazily mmaps the backing pages — mirrors ROCr, which closes the export fd only
  // AFTER access setup). Then close the export fd so its number becomes free.
  int dmabuf_a = make_sized_memfd(kBoSize);
  ASSERT_GE(dmabuf_a, 0);
  uint32_t handle_a = 0;
  ASSERT_TRUE(prime_import(drm, dmabuf_a, &handle_a));
  ASSERT_NE(handle_a, 0u);

  drm_amdgpu_gem_va map_a{};
  map_a.handle = handle_a;
  map_a.operation = AMDGPU_VA_OP_MAP;
  map_a.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  map_a.va_address = va_a;
  map_a.map_size = kBoSize;
  ASSERT_EQ(ioctl(drm, gem_va, &map_a), 0);

  const int reused_number = dmabuf_a;
  ASSERT_EQ(close(dmabuf_a), 0); // A's mapping stays live; the handle owns it now.

  // Second BO: force a fresh memfd onto the SAME fd number A's export used, then
  // import + map it. Under the old handle = dmabuf_fd + 1 scheme this PRIME would
  // collide with A's still-live handle and tear down A's BO; with stable handles it
  // must mint a distinct handle and leave A untouched.
  int tmp = make_sized_memfd(kBoSize);
  ASSERT_GE(tmp, 0);
  int dmabuf_b = reused_number;
  // If make_sized_memfd already recycled the freed number for `tmp`, it is already
  // the reused number — dup2 onto itself then close would leave it closed, so just
  // use it directly. Otherwise move it onto the reused number.
  if (tmp != dmabuf_b) {
    ASSERT_EQ(dup2(tmp, dmabuf_b), dmabuf_b);
    ASSERT_EQ(close(tmp), 0);
  }
  uint32_t handle_b = 0;
  ASSERT_TRUE(prime_import(drm, dmabuf_b, &handle_b));
  ASSERT_NE(handle_b, 0u);
  EXPECT_NE(handle_a, handle_b)
      << "a recycled dmabuf fd number must not collide with a live handle";

  drm_amdgpu_gem_va map_b{};
  map_b.handle = handle_b;
  map_b.operation = AMDGPU_VA_OP_MAP;
  map_b.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  map_b.va_address = va_b;
  map_b.map_size = kBoSize;
  ASSERT_EQ(ioctl(drm, gem_va, &map_b), 0);
  close(dmabuf_b); // B's mapping stays live via its handle.

  // A must still be fully live despite B reusing its export fd number: A's own
  // UNMAP through A's handle must still succeed (proving B's PRIME did not tear
  // down A's mapping).
  drm_amdgpu_gem_va unmap_a{};
  unmap_a.handle = handle_a;
  unmap_a.operation = AMDGPU_VA_OP_UNMAP;
  unmap_a.va_address = va_a;
  unmap_a.map_size = kBoSize;
  EXPECT_EQ(ioctl(drm, gem_va, &unmap_a), 0)
      << "reusing A's export fd number for B must not tear down A's mapping";

  // Closing A's handle must not disturb B: B's UNMAP through its own handle must
  // still succeed afterward.
  EXPECT_EQ(gem_close(drm, handle_a), 0);
  drm_amdgpu_gem_va unmap_b{};
  unmap_b.handle = handle_b;
  unmap_b.operation = AMDGPU_VA_OP_UNMAP;
  unmap_b.va_address = va_b;
  unmap_b.map_size = kBoSize;
  EXPECT_EQ(ioctl(drm, gem_va, &unmap_b), 0)
      << "closing the recycled-fd sibling handle must not tear down this BO's mapping";

  EXPECT_EQ(gem_close(drm, handle_b), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

// UNMAP with a handle that does not own the exact range must fail rather than
// tear down another handle's PTEs or report a phantom success.
TEST(InterposerGemTest, UnmapWithWrongHandleFails) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  constexpr size_t kBoSize = 0x1000;
  int dmabuf_a = make_sized_memfd(kBoSize);
  ASSERT_GE(dmabuf_a, 0);
  int dmabuf_b = make_sized_memfd(kBoSize);
  ASSERT_GE(dmabuf_b, 0);
  uint32_t handle_a = 0, handle_b = 0;
  ASSERT_TRUE(prime_import(drm, dmabuf_a, &handle_a));
  ASSERT_TRUE(prime_import(drm, dmabuf_b, &handle_b));

  const unsigned long gem_va = DRM_AMDGPU_GEM_VA_request();
  const uint64_t va_a = 0x1000000000ULL;
  drm_amdgpu_gem_va map_a{};
  map_a.handle = handle_a;
  map_a.operation = AMDGPU_VA_OP_MAP;
  map_a.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  map_a.va_address = va_a;
  map_a.map_size = kBoSize;
  ASSERT_EQ(ioctl(drm, gem_va, &map_a), 0);

  // UNMAP of A's range through B's handle must fail (B does not own it) and leave
  // A's mapping intact, so A's own UNMAP then succeeds.
  drm_amdgpu_gem_va bad_unmap{};
  bad_unmap.handle = handle_b;
  bad_unmap.operation = AMDGPU_VA_OP_UNMAP;
  bad_unmap.va_address = va_a;
  bad_unmap.map_size = kBoSize;
  EXPECT_EQ(ioctl(drm, gem_va, &bad_unmap), -1);

  drm_amdgpu_gem_va good_unmap{};
  good_unmap.handle = handle_a;
  good_unmap.operation = AMDGPU_VA_OP_UNMAP;
  good_unmap.va_address = va_a;
  good_unmap.map_size = kBoSize;
  EXPECT_EQ(ioctl(drm, gem_va, &good_unmap), 0)
      << "A's mapping must survive a wrong-handle UNMAP attempt";

  EXPECT_EQ(gem_close(drm, handle_a), 0);
  EXPECT_EQ(gem_close(drm, handle_b), 0);
  EXPECT_EQ(close(dmabuf_a), 0);
  EXPECT_EQ(close(dmabuf_b), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

// PRIME_FD_TO_HANDLE dups the export fd internally, so the caller may close its
// export fd BEFORE GEM_VA MAP and the deferred lazy backing mmap must still succeed
// (the handle owns a private dup of the dmabuf that outlives the caller's fd).
TEST(InterposerGemTest, MapSucceedsAfterExportFdClosed) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  constexpr size_t kBoSize = 0x1000;
  int dmabuf = make_sized_memfd(kBoSize);
  ASSERT_GE(dmabuf, 0);
  uint32_t handle = 0;
  ASSERT_TRUE(prime_import(drm, dmabuf, &handle));
  ASSERT_NE(handle, 0u);

  // Close the export fd BEFORE mapping. Under the old scheme (store the raw fd for a
  // later lazy mmap) this would either fail or map an unrelated recycled fd. Force a
  // recycle of the number so a lingering raw-fd dependency would be caught.
  const int reused_number = dmabuf;
  ASSERT_EQ(close(dmabuf), 0);
  int filler = make_sized_memfd(kBoSize);
  ASSERT_GE(filler, 0);
  if (filler != reused_number) {
    ASSERT_EQ(dup2(filler, reused_number), reused_number);
    ASSERT_EQ(close(filler), 0);
  }

  const unsigned long gem_va = DRM_AMDGPU_GEM_VA_request();
  const uint64_t va = 0x1000000000ULL;
  drm_amdgpu_gem_va map{};
  map.handle = handle;
  map.operation = AMDGPU_VA_OP_MAP;
  map.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  map.va_address = va;
  map.map_size = kBoSize;
  EXPECT_EQ(ioctl(drm, gem_va, &map), 0)
      << "GEM_VA MAP must succeed via the handle's private dmabuf dup after the "
         "caller closed (and recycled) its export fd";

  drm_amdgpu_gem_va unmap{};
  unmap.handle = handle;
  unmap.operation = AMDGPU_VA_OP_UNMAP;
  unmap.va_address = va;
  unmap.map_size = kBoSize;
  EXPECT_EQ(ioctl(drm, gem_va, &unmap), 0);
  EXPECT_EQ(gem_close(drm, handle), 0);
  EXPECT_EQ(close(reused_number), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

// AMDGPU_VA_OP_REPLACE at a VA that overlaps an existing mapping of a DIFFERENT size
// must evict the old mapping (by its own extent), not silently install overlapping
// PTEs. After a REPLACE, the old owner's stale range must be gone: its handle's UNMAP
// of the original range must fail, and no double-unmap can occur.
TEST(InterposerGemTest, ReplaceEvictsOverlappingDifferentSizeRange) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  constexpr size_t kBigBo = 0x4000;
  constexpr size_t kSmallBo = 0x1000;
  const unsigned long gem_va = DRM_AMDGPU_GEM_VA_request();
  const uint64_t va = 0x1000000000ULL;

  int dmabuf_a = make_sized_memfd(kBigBo);
  ASSERT_GE(dmabuf_a, 0);
  int dmabuf_b = make_sized_memfd(kSmallBo);
  ASSERT_GE(dmabuf_b, 0);
  uint32_t handle_a = 0, handle_b = 0;
  ASSERT_TRUE(prime_import(drm, dmabuf_a, &handle_a));
  ASSERT_TRUE(prime_import(drm, dmabuf_b, &handle_b));

  // A maps a large range at va.
  drm_amdgpu_gem_va map_a{};
  map_a.handle = handle_a;
  map_a.operation = AMDGPU_VA_OP_MAP;
  map_a.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  map_a.va_address = va;
  map_a.map_size = kBigBo;
  ASSERT_EQ(ioctl(drm, gem_va, &map_a), 0);

  // B REPLACEs a SMALLER range at the same base va. This overlaps A's range but is
  // not identical; it must still evict A's mapping.
  drm_amdgpu_gem_va replace_b{};
  replace_b.handle = handle_b;
  replace_b.operation = AMDGPU_VA_OP_REPLACE;
  replace_b.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  replace_b.va_address = va;
  replace_b.map_size = kSmallBo;
  EXPECT_EQ(ioctl(drm, gem_va, &replace_b), 0)
      << "REPLACE overlapping a different-size range must succeed and evict it";

  // A's original range is gone: A's UNMAP of it must now fail (the record was evicted
  // by the overlap-aware REPLACE, so A cannot double-unmap B's new PTEs).
  drm_amdgpu_gem_va unmap_a{};
  unmap_a.handle = handle_a;
  unmap_a.operation = AMDGPU_VA_OP_UNMAP;
  unmap_a.va_address = va;
  unmap_a.map_size = kBigBo;
  EXPECT_EQ(ioctl(drm, gem_va, &unmap_a), -1)
      << "A's overlapping range must have been evicted by B's REPLACE";

  // B's new range is live and its UNMAP succeeds exactly once.
  drm_amdgpu_gem_va unmap_b{};
  unmap_b.handle = handle_b;
  unmap_b.operation = AMDGPU_VA_OP_UNMAP;
  unmap_b.va_address = va;
  unmap_b.map_size = kSmallBo;
  EXPECT_EQ(ioctl(drm, gem_va, &unmap_b), 0);

  EXPECT_EQ(gem_close(drm, handle_a), 0);
  EXPECT_EQ(gem_close(drm, handle_b), 0);
  EXPECT_EQ(close(dmabuf_a), 0);
  EXPECT_EQ(close(dmabuf_b), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

// After B REPLACEs A's range, closing A must NOT tear down B's PTEs: the REPLACE
// transferred ownership of the VA to B, so A's teardown has nothing to unmap there
// and B's mapping stays live (its own UNMAP still succeeds afterward).
TEST(InterposerGemTest, ReplaceTransfersOwnershipAcrossOldOwnerClose) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  constexpr size_t kBoSize = 0x1000;
  const unsigned long gem_va = DRM_AMDGPU_GEM_VA_request();
  const uint64_t va = 0x1000000000ULL;

  int dmabuf_a = make_sized_memfd(kBoSize);
  ASSERT_GE(dmabuf_a, 0);
  int dmabuf_b = make_sized_memfd(kBoSize);
  ASSERT_GE(dmabuf_b, 0);
  uint32_t handle_a = 0, handle_b = 0;
  ASSERT_TRUE(prime_import(drm, dmabuf_a, &handle_a));
  ASSERT_TRUE(prime_import(drm, dmabuf_b, &handle_b));

  drm_amdgpu_gem_va map_a{};
  map_a.handle = handle_a;
  map_a.operation = AMDGPU_VA_OP_MAP;
  map_a.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  map_a.va_address = va;
  map_a.map_size = kBoSize;
  ASSERT_EQ(ioctl(drm, gem_va, &map_a), 0);

  drm_amdgpu_gem_va replace_b = map_a;
  replace_b.handle = handle_b;
  replace_b.operation = AMDGPU_VA_OP_REPLACE;
  ASSERT_EQ(ioctl(drm, gem_va, &replace_b), 0);

  // Ownership of the VA transferred to B: A must no longer own the range, so A's
  // UNMAP of it fails. This is the load-bearing assertion — gem_va_unmap() reports
  // success purely on process existence, not PTE presence, so without this negative
  // check the later "B's UNMAP succeeds" alone could pass even if REPLACE had failed
  // to evict A's record (A's close would then quietly tear down the shared PTE while
  // B's bookkeeping stayed intact).
  drm_amdgpu_gem_va unmap_a{};
  unmap_a.handle = handle_a;
  unmap_a.operation = AMDGPU_VA_OP_UNMAP;
  unmap_a.va_address = va;
  unmap_a.map_size = kBoSize;
  EXPECT_EQ(ioctl(drm, gem_va, &unmap_a), -1)
      << "A must not own the range after REPLACE transferred it to B";

  // Close A entirely. Its GEM_CLOSE reap must not unmap va — B owns it now.
  EXPECT_EQ(gem_close(drm, handle_a), 0);
  EXPECT_EQ(close(dmabuf_a), 0);

  // B's mapping is still live: its UNMAP through its own handle succeeds.
  drm_amdgpu_gem_va unmap_b{};
  unmap_b.handle = handle_b;
  unmap_b.operation = AMDGPU_VA_OP_UNMAP;
  unmap_b.va_address = va;
  unmap_b.map_size = kBoSize;
  EXPECT_EQ(ioctl(drm, gem_va, &unmap_b), 0)
      << "B's mapping must survive A's close after REPLACE transferred ownership";

  EXPECT_EQ(gem_close(drm, handle_b), 0);
  EXPECT_EQ(close(dmabuf_b), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}

// AMDGPU_VA_OP_CLEAR tears down a range's PTEs and updates the owning handle's
// bookkeeping, so a later GEM_CLOSE of that handle does not double-unmap the range.
// A UNMAP of the cleared range must fail (it is gone), and GEM_CLOSE must still
// succeed cleanly.
TEST(InterposerGemTest, ClearUpdatesOwnerBookkeeping) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  ASSERT_TRUE(kfd_version_ok(kfd));
  int drm = open_drm_render();
  if (drm < 0)
    GTEST_SKIP() << "synthetic DRM render node unavailable in this configuration";

  constexpr size_t kBoSize = 0x1000;
  const unsigned long gem_va = DRM_AMDGPU_GEM_VA_request();
  const uint64_t va = 0x1000000000ULL;

  int dmabuf = make_sized_memfd(kBoSize);
  ASSERT_GE(dmabuf, 0);
  uint32_t handle = 0;
  ASSERT_TRUE(prime_import(drm, dmabuf, &handle));

  drm_amdgpu_gem_va map{};
  map.handle = handle;
  map.operation = AMDGPU_VA_OP_MAP;
  map.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  map.va_address = va;
  map.map_size = kBoSize;
  ASSERT_EQ(ioctl(drm, gem_va, &map), 0);

  // CLEAR is handle-agnostic; it uses handle 0 and tears down whatever owns the VA.
  drm_amdgpu_gem_va clear{};
  clear.handle = 0;
  clear.operation = AMDGPU_VA_OP_CLEAR;
  clear.va_address = va;
  clear.map_size = kBoSize;
  EXPECT_EQ(ioctl(drm, gem_va, &clear), 0) << "CLEAR of a mapped range must succeed";

  // The range is gone from the owner's bookkeeping: an UNMAP now fails.
  drm_amdgpu_gem_va unmap{};
  unmap.handle = handle;
  unmap.operation = AMDGPU_VA_OP_UNMAP;
  unmap.va_address = va;
  unmap.map_size = kBoSize;
  EXPECT_EQ(ioctl(drm, gem_va, &unmap), -1) << "CLEAR must have removed the range record";

  // GEM_CLOSE must not double-unmap the already-cleared range.
  EXPECT_EQ(gem_close(drm, handle), 0);
  EXPECT_EQ(close(dmabuf), 0);
  EXPECT_EQ(close(drm), 0);
  EXPECT_EQ(close(kfd), 0);
}
