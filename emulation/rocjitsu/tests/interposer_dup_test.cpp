// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file interposer_dup_test.cpp
/// @brief LD_PRELOAD regression tests for the interposer's KFD fd/backend
///        bookkeeping across dup / dup2 / dup3 / fcntl(F_DUPFD).
///
/// @details These run with librocjitsu.so preloaded (see the ENVIRONMENT set in
/// tests/CMakeLists.txt) so that open("/dev/kfd") is serviced by the simulated
/// KFD driver. AMDKFD_IOC_GET_VERSION is used purely as a routing probe: it
/// succeeds (returns 0 and fills the version) only when the fd is routed to a
/// KFD backend, and fails when the fd falls through to the real (non-KFD)
/// descriptor. The SimulatedDriver-level unit tests (KfdIoctlTest) cannot catch
/// these because they exercise the driver object directly, bypassing the fd
/// tracking that lives in the interposer.

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "drm/amdgpu_drm.h"
#include "drm/drm.h"
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

namespace {

// Issue AMDKFD_IOC_GET_VERSION on fd. Returns true if the fd routed to a KFD
// backend (ioctl succeeded and reported the expected major version).
bool kfd_version_ok(int fd) {
  kfd_ioctl_get_version_args args{};
  int rc = ioctl(fd, AMDKFD_IOC_GET_VERSION, &args);
  return rc == 0 && args.major_version == KFD_IOCTL_MAJOR_VERSION;
}

int open_kfd() { return open("/dev/kfd", O_RDWR | O_CLOEXEC); }

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

namespace {

// Open the synthetic DRM render node the interposer exposes for the simulated GPU
// (render minor 128 in the KMD test configs). Requires the KFD driver to be up, so
// callers open /dev/kfd first.
int open_drm_render() { return open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC); }

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

} // namespace

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
