// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file interposer.cpp
/// @brief LD_PRELOAD interposer that redirects KFD syscalls to rocjitsu KFD drivers.
///
/// @details Intercepts open, close, ioctl, mmap, munmap, and filesystem access
/// to route /dev/kfd operations and sysfs topology reads through one of two
/// strategies. Normal simulation creates a VM and uses SimulatedKfd to own all
/// visible GPU discovery and queue execution. DBT guest mode uses GuestKfd to
/// append one synthetic guest GPU over either real KFD hardware or an existing
/// SimulatedKfd target. The HSA tools hook maps guest-agent API calls to that
/// execution agent and translates guest code objects before loading them. All
/// mutable state is consolidated in InterposerContext.

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/config/dbt_guest_config.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/guest_kfd.h"
#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "rocjitsu/kmd/linux/linux_kfd.h"
#include "rocjitsu/kmd/linux/remote_driver.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/kmd/linux/sysfs.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"
#include "rocjitsu/vm/plugins/profiled_execution_plugin_group.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"

RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
// Vendored kernel DRM/amdgpu UAPI (MIT). Provides the real drm_version,
// drm_amdgpu_info, drm_amdgpu_info_device, drm_amdgpu_info_vram_gtt, and
// drm_amdgpu_memory_info structs so the interposer services the amdgpu DRM
// ioctl ABI directly. These are kernel ABI, not libdrm library types, so this
// keeps the interposer independent of libdrm.
#include "amdgpu_drm.h"
#include "drm.h"
RJ_DIAGNOSTIC_POP

#include "util/dynamic_loader.h"
#include "util/log.h"

#include <atomic>
#include <cassert>
#include <cerrno>
#include <charconv>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <exception>
#include <fcntl.h>
#include <linux/memfd.h>
#include <memory>
#include <mutex>
#include <optional>
#include <signal.h>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" rocjitsu::ExecutionPlugin *createKernelLoggingPlugin();
extern "C" rocjitsu::ExecutionPlugin *createRaceDetectorPlugin();

using rocjitsu::GuestKfd;
using rocjitsu::LinuxKfd;
using rocjitsu::RemoteDriver;
using rocjitsu::SimulatedKfd;
using rocjitsu::Sysfs;

static int connect_to_daemon() {
  auto path = rocjitsu::rpc_default_socket_path();
  int sock = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (sock < 0)
    return -1;
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  path.copy(addr.sun_path, sizeof(addr.sun_path) - 1);
  if (connect(sock, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
    rocjitsu::libc_passthrough().close(sock);
    return -1;
  }
  return sock;
}

namespace {

/// @brief Convert a kernel-style driver ioctl result into the libc ioctl(2)
/// return/`errno` contract.
///
/// @param r Driver result: `>= 0` on success, `-errno` on failure.
/// @returns @p r unchanged when non-negative; otherwise `-1` with `errno` set to
///          `-r`.
int kfd_ioctl_ret(int r) {
  if (r < 0) {
    errno = -r;
    return -1;
  }
  return r;
}

/// @brief Return the child-process rocjitsu config path.
///
/// @details The launcher writes the config path to the shared runtime file for
/// both local simulation and DBT guest mode.
std::optional<std::string> child_config_path() {
  auto cfg_file = rocjitsu::rpc_default_config_file_path();
  char cfg_buf[4096]{};
  auto &real = rocjitsu::libc_passthrough();
  int cfg_fd = real.openat(AT_FDCWD, cfg_file.c_str(), O_RDONLY, 0);
  if (cfg_fd < 0)
    return std::nullopt;

  auto n = real.read(cfg_fd, cfg_buf, sizeof(cfg_buf) - 1);
  real.close(cfg_fd);
  if (n <= 0)
    return std::nullopt;

  while (n > 0 && (cfg_buf[n - 1] == '\n' || cfg_buf[n - 1] == '\r'))
    cfg_buf[--n] = '\0';
  return std::string(cfg_buf);
}

void *raw_mmap_syscall(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  // syscall(2) is the libc wrapper, not a raw inline syscall instruction: on
  // kernel errors it returns -1 and sets errno. For mmap(2), success returns
  // the mapped address; for munmap(2), success returns exactly 0.
  long rc = syscall(SYS_mmap, addr, length, prot, flags, fd, offset);
  if (rc == -1)
    return MAP_FAILED;
  return reinterpret_cast<void *>(static_cast<uintptr_t>(rc));
}

int raw_munmap_syscall(void *addr, size_t length) {
  long rc = syscall(SYS_munmap, addr, length);
  assert(rc == 0 || rc == -1);
  return static_cast<int>(rc);
}

void rj_sigsegv_handler(int, siginfo_t *, void *) {
  signal(SIGSEGV, SIG_DFL);
  raise(SIGSEGV);
}

__attribute__((constructor)) void rj_install_signal_handler() {
  struct sigaction sa {};
  sa.sa_sigaction = rj_sigsegv_handler;
  sa.sa_flags = SA_SIGINFO;
  sigaction(SIGSEGV, &sa, nullptr);
}

/// @brief All mutable interposer state.
class InterposerContext {
public:
  /// @brief Which backend holds the open reference for a tracked KFD dup fd.
  enum class DupBackend : uint8_t { Local, Remote };

  static inline thread_local bool in_construction = false;
  static rocjitsu::LibcPassthrough &real() { return rocjitsu::libc_passthrough(); }
  static InterposerContext &ctx;

  static void init() {
    new (storage_) InterposerContext();
    real().resolve();
  }

  /// @brief Reset interposer state in a forked child process.
  /// @details After fork(), the child inherits the parent's address space but
  /// the engine thread is dead (only the calling thread survives). Mutexes may
  /// be locked by threads that no longer exist. We reinitialize everything so
  /// the next open("/dev/kfd") creates a fresh connection.
  void reset_after_fork() {
    active_driver_.store(nullptr, std::memory_order_release);
    rj_vm_ = nullptr;
    if (guest_driver_)
      guest_driver_->reset_after_fork();
    guest_driver_.reset();
    // Drop the child's reference to the inherited RemoteDriver. Storing nullptr
    // releases this shared_ptr; if it was the last reference in the child, the
    // child's ~RemoteDriver runs and closes only the child's inherited (dup'd)
    // fds — it does not send RPC_CLOSE and cannot invalidate the parent's live
    // connection, so this is safe in a forked child.
    //
    // Unlike the plain mutexes below, remote_ is NOT reinitialized via
    // placement-new: re-constructing a live object over itself ends the current
    // object's lifetime without running its destructor, which is UB for a
    // non-trivial type like std::atomic<std::shared_ptr<>> (it leaks/By-passes the
    // control-block bookkeeping). A plain store(nullptr) is the correct way to
    // release it.
    //
    // Caveat: std::atomic<std::shared_ptr<>> may serialize on a libstdc++-internal
    // pool spinlock. If a parent thread was mid remote_.load()/store() at fork(),
    // the child inherits that lock held-by-a-dead-thread and this store() could
    // block. reset_after_fork() therefore assumes no in-flight remote_ atomic
    // access at the fork point (the common fork-then-exec / single-threaded-fork
    // case) — placement-new would not sidestep this deadlock either (still UB),
    // and leaving remote_ non-null would wrongly reuse the parent's connection.
    remote_.store(nullptr, std::memory_order_relaxed);
    remote_kfd_fd_.store(-1, std::memory_order_relaxed);
    remote_open_refs_.store(0, std::memory_order_relaxed);
    new (&init_mutex_) std::mutex();
    new (&fd_mutex_) std::mutex();
    new (&remote_mutex_) std::mutex();
    sysfs_fds_.clear();
    drm_fds_.clear();
    kfd_dup_fds_.clear();
    in_construction = false;
  }

  LinuxKfd *driver() { return active_driver_.load(std::memory_order_acquire); }
  int driver_fd() {
    auto *d = driver();
    return d ? d->fd() : -1;
  }
  bool initialized() const {
    return active_driver_.load(std::memory_order_acquire) != nullptr ||
           remote_.load(std::memory_order_acquire) != nullptr;
  }

  /// @brief Take a lifetime-extending snapshot of the active remote driver.
  /// @details The returned shared_ptr keeps the RemoteDriver alive for as long as
  /// the caller holds it, even if a concurrent teardown_remote() clears remote_.
  std::shared_ptr<RemoteDriver> remote() { return remote_.load(std::memory_order_acquire); }

  int remote_kfd_fd() const { return remote_kfd_fd_.load(std::memory_order_acquire); }

  std::shared_ptr<RemoteDriver> remote_lookup(int fd) {
    auto active_remote = remote_.load(std::memory_order_acquire);
    return (fd >= 0 && fd == remote_kfd_fd_.load(std::memory_order_acquire) && active_remote)
               ? active_remote
               : nullptr;
  }

  // No lock needed: the snapshot keeps the RemoteDriver alive, and its handshake
  // metadata (topology/drm paths, gpu_info) is immutable after open() — close()
  // does not clear it — so this read never races a concurrent teardown. Callers
  // that need BOTH the topology and drm paths together must take a single
  // remote() snapshot and read both from it (see redirect_sysfs_path), so the two
  // paths can't come from different RemoteDrivers across a teardown/reconnect.
  std::string remote_drm_path() {
    auto active_remote = remote_.load(std::memory_order_acquire);
    return active_remote ? std::string(active_remote->drm_path()) : std::string{};
  }

  /// @brief Retain one remote open reference if a remote connection is live.
  /// @details Combined check-and-retain under remote_mutex_ so a concurrent
  /// last-release+teardown cannot slip between "is a remote live?" and the
  /// increment and resurrect a torn-down connection. Returns true if a reference
  /// was added (i.e. a remote is active). Mirrors the local path, which does the
  /// same check-and-retain under process_mutex_.
  bool retain_remote_open_if_active() {
    std::lock_guard lock(remote_mutex_);
    if (!remote_.load(std::memory_order_acquire))
      return false;
    remote_open_refs_.fetch_add(1, std::memory_order_acq_rel);
    return true;
  }

  /// @brief Drop one remote open reference, tearing down the connection on the
  /// last release.
  /// @details On the final release this sends RPC_CLOSE to the daemon (via
  /// RemoteDriver::close()) so the daemon frees this client's process state,
  /// rather than leaking it until socket disconnect at process exit. The whole
  /// decrement-and-maybe-teardown runs under remote_mutex_ so it is serialized
  /// against retain and get_or_create_remote; retain can never observe a live
  /// remote and add a reference in the window where this is tearing one down.
  void release_remote_open() {
    std::shared_ptr<RemoteDriver> dead;
    {
      std::lock_guard lock(remote_mutex_);
      // Tolerate a spurious release after teardown already reset the count to 0:
      // teardown_locked() clears kfd_dup_fds_ and zeroes the count together, so a
      // dup close that lost the race can still land here. Only decrement a live
      // reference.
      int prev = remote_open_refs_.load(std::memory_order_acquire);
      if (prev <= 0)
        return;
      remote_open_refs_.store(prev - 1, std::memory_order_release);
      if (prev == 1)
        dead = teardown_remote_locked();
    }
    // Perform the graceful RPC_CLOSE OUTSIDE remote_mutex_. The atomic shared_ptr
    // already guarantees the driver's lifetime, so the (blocking) RPC shutdown
    // does not need the lock; holding remote_mutex_ across a blocking send/recv
    // would stall every other remote_mutex_ user behind an arbitrary-length RPC.
    if (dead)
      dead->close();
  }

  /// @brief Result of get_or_create_remote(): the live driver plus the primary
  /// KFD fd number captured under remote_mutex_.
  /// @details Returning the fd snapshot (rather than making the caller re-read
  /// remote_kfd_fd() after the lock drops) closes a race where a concurrent
  /// dup2/dup3 clears remote_kfd_fd_ between the helper returning and the caller
  /// reading it, making open("/dev/kfd") return -1 or a reused non-KFD fd.
  struct RemoteOpenResult {
    std::shared_ptr<RemoteDriver> driver;
    int fd = -1;
    explicit operator bool() const { return static_cast<bool>(driver); }
  };

  RemoteOpenResult get_or_create_remote() {
    std::lock_guard lock(remote_mutex_);
    auto active_remote = remote_.load(std::memory_order_acquire);
    if (active_remote) {
      // The connection is already live: retain a reference and reuse it. Never
      // re-open() a live RemoteDriver — that would re-handshake and leak a fresh
      // socket/fd. remote_mutex_ is already held here, so increment directly
      // (serialized with release).
      remote_open_refs_.fetch_add(1, std::memory_order_acq_rel);
      // If the cached primary fd number was lost (e.g. dup2 overwrote it while
      // other refs kept the connection alive), mint a fresh synthetic primary fd
      // WITHOUT reconnecting so open("/dev/kfd") still returns a valid fd.
      int fd = remote_kfd_fd_.load(std::memory_order_acquire);
      if (fd < 0) {
        fd = active_remote->reissue_synthetic_kfd_fd();
        if (fd < 0) {
          // Roll back the reference taken above. remote_mutex_ is held, so
          // decrement directly rather than via release_remote_open() (which would
          // re-lock and deadlock).
          remote_open_refs_.fetch_sub(1, std::memory_order_acq_rel);
          return {};
        }
        remote_kfd_fd_.store(fd, std::memory_order_release);
      }
      // Return the fd captured under the lock so a concurrent invalidation
      // cannot clear it before the caller uses it.
      return {active_remote, fd};
    }
    int sock = connect_to_daemon();
    if (sock < 0)
      return {};
    // Build the driver and open its KFD connection BEFORE publishing remote_.
    // Publishing early (then returning on open()<0) would leave remote_ non-null
    // with remote_kfd_fd_ == -1, so initialized() would report success and gate
    // out local-VM creation / a later remote retry. Only a fully-open driver is
    // ever visible to lock-free readers.
    if (!active_remote)
      active_remote = std::make_shared<RemoteDriver>(sock);
    int fd = active_remote->open();
    if (fd < 0)
      return {};
    // Publish the fd and open refcount BEFORE remote_. initialized() and the
    // lock-free readers gate on remote_, so making the remote_ release-store the
    // LAST step guarantees any thread that observes a non-null remote_ also sees
    // a valid remote_kfd_fd_ and a nonzero open refcount — never a half-published
    // remote whose fd/ref state has not landed yet.
    remote_kfd_fd_.store(fd, std::memory_order_release);
    // The primary remote KFD fd holds the first open reference.
    remote_open_refs_.store(1, std::memory_order_release);
    remote_.store(active_remote, std::memory_order_release);
    return {active_remote, fd};
  }

  LinuxKfd *lookup(int fd) {
    auto *d = driver();
    return (d && fd >= 0 && fd == d->fd()) ? d : nullptr;
  }

  bool owns_fd(int fd) {
    auto *d = driver();
    return d && d->owns_fd(fd);
  }

  std::string redirect(const char *path) {
    auto *d = driver();
    return d ? d->redirect_sysfs_path(path) : std::string{};
  }

  std::string redirect_sysfs_path(const char *path) {
    if (!path)
      return {};

    std::string_view sv(path);
    if (!sv.starts_with("/sys/class/drm") && !sv.starts_with("/sys/devices/virtual/kfd") &&
        !sv.starts_with("/sys/class/kfd"))
      return {};

    // initialized() already covers a live remote (remote_ != nullptr), which is
    // the stable signal independent of the primary fd number; only create a local
    // VM when nothing is initialized yet.
    if (!initialized())
      get_or_create();

    // Read topology and drm paths from ONE remote snapshot so a concurrent
    // teardown/reconnect between the two reads cannot combine a topology path from
    // one RemoteDriver with a drm path from another (or an empty one). The
    // metadata is immutable-after-open, so a single live snapshot is consistent.
    if (auto remote = remote_.load(std::memory_order_acquire)) {
      std::string remote_topology(remote->topology_path());
      if (!remote_topology.empty()) {
        std::string redirected = LinuxKfd::redirect_sysfs_root_path(
            path, remote_topology, std::string(remote->drm_path()));
        if (!redirected.empty())
          return redirected;
      }
    }

    return redirect(path);
  }

  bool is_kfd_primary(int fd) {
    return fd == driver_fd() || fd == remote_kfd_fd_.load(std::memory_order_acquire);
  }

  bool is_kfd_dup(int fd) {
    std::lock_guard lock(fd_mutex_);
    return kfd_dup_fds_.count(fd) != 0;
  }

  bool is_kfd_tracked(int fd) { return is_kfd_primary(fd) || is_kfd_dup(fd); }

  void track_open_fd(int fd) {
    if (fd < 0 || is_kfd_primary(fd))
      return;
    std::lock_guard lock(fd_mutex_);
    // GuestKfd::open() already retained one driver (local) reference before
    // returning this app-facing dup fd. Track it as a Local dup so ioctl/mmap/
    // close route through the driver without incrementing the reference count a
    // second time (unlike commit_dup(), which consumes a fresh reservation).
    kfd_dup_fds_.emplace(fd, DupBackend::Local);
  }

  /// @brief Resolve which backend a tracked KFD fd belongs to.
  /// @details Returns the backend of the local/remote primary fd, or the backend
  /// recorded for a tracked dup, or nullopt if the fd is not a tracked KFD fd.
  /// A dup must inherit the SOURCE fd's backend rather than guessing from global
  /// state: when local mode was created first and the daemon later becomes
  /// active, both driver() and remote_ are non-null, so guessing would mis-tag a
  /// dup of the remote fd as Local and release the wrong backend on close.
  std::optional<DupBackend> kfd_backend_of(int fd) {
    if (fd < 0)
      return std::nullopt;
    if (fd == driver_fd())
      return DupBackend::Local;
    if (fd == remote_kfd_fd_.load(std::memory_order_acquire))
      return DupBackend::Remote;
    std::lock_guard lock(fd_mutex_);
    auto it = kfd_dup_fds_.find(fd);
    if (it != kfd_dup_fds_.end())
      return it->second;
    return std::nullopt;
  }

  /// @brief Retain one open reference on a specific backend.
  /// @returns true if a reference was acquired; false if that backend is no
  /// longer active (local VM gone, or remote torn down). Mirrors release_backend.
  bool retain_backend(DupBackend backend) {
    if (backend == DupBackend::Local) {
      auto *d = driver();
      // retain_local_open() reports false if the local process was already torn
      // down (a racing last-close), so a dup of a dying local fd is not falsely
      // treated as retained.
      return d && d->retain_local_open();
    }
    return retain_remote_open_if_active();
  }

  /// @brief Release one open reference on the backend recorded for a dup.
  /// @details Local dups release the local process refcount; remote dups release
  /// the daemon connection refcount. release_remote_open() is a no-op when no
  /// remote reference is live, so a release that races a teardown is harmless.
  void release_backend(DupBackend backend) {
    if (backend == DupBackend::Local) {
      if (auto *d = driver())
        d->close();
    } else {
      release_remote_open();
    }
  }

  /// @brief Reserve (retain) a reference on the backend that owns @p src_fd,
  /// before duplicating it, so the backend cannot be torn down by a racing
  /// last-close between the dup syscall and tracking the new fd.
  /// @details Acquires the open reference on the source fd's backend FIRST; the
  /// caller then performs the dup syscall and, on success, hands the reserved
  /// backend to commit_dup() (which records the new fd tagged with that same
  /// backend). Reserving before the syscall keeps the invariant "every entry in
  /// kfd_dup_fds_ holds exactly one reference on its recorded backend" and passing
  /// the source backend through (rather than rediscovering it from global state)
  /// prevents mis-tagging a remote-fd dup as Local when both backends are active.
  /// @returns The reserved backend, or nullopt if @p src_fd is not a tracked KFD
  /// fd or its backend is no longer active. On a non-nullopt return the caller
  /// MUST consume the reservation exactly once: commit_dup() on syscall success,
  /// or release_backend() on syscall failure.
  [[nodiscard]] std::optional<DupBackend> reserve_dup_backend(int src_fd) {
    auto backend = kfd_backend_of(src_fd);
    if (!backend)
      return std::nullopt;
    if (!retain_backend(*backend))
      return std::nullopt; // backend went away before we could reserve.
    return backend;
  }

  /// @brief Record a dup fd whose backend reference was already reserved via
  /// reserve_dup_backend(). Consumes exactly that one reserved reference.
  void commit_dup(int fd, DupBackend backend) {
    if (fd < 0 || is_kfd_primary(fd)) {
      // Cannot track this as a dup (invalid, or the number aliases a primary).
      // Release the reserved reference to stay balanced.
      release_backend(backend);
      return;
    }
    bool inserted = false;
    {
      std::lock_guard lock(fd_mutex_);
      inserted = kfd_dup_fds_.emplace(fd, backend).second;
    }
    if (!inserted)
      // fd already tracked (dup returned a recycled number we still hold): undo
      // the reserved reference so the count stays balanced.
      release_backend(backend);
  }

  void untrack_dup(int fd) {
    if (fd < 0)
      return;
    DupBackend backend;
    bool was_tracked = false;
    {
      std::lock_guard lock(fd_mutex_);
      auto it = kfd_dup_fds_.find(fd);
      if (it != kfd_dup_fds_.end()) {
        backend = it->second;
        kfd_dup_fds_.erase(it);
        was_tracked = true;
      }
    }
    if (was_tracked)
      release_backend(backend);
  }

  /// @brief Drop all KFD identity/references for an fd number that dup2/dup3 is
  /// about to reuse (the syscall already atomically closed whatever it was).
  /// @details Covers three cases so the reused number cannot keep routing to a
  /// stale backend:
  ///   - tracked KFD dup: release its recorded backend and erase its entry;
  ///   - remote primary: clear remote_kfd_fd_ and drop its open reference;
  ///   - local primary: forget the primary fd number and drop its open reference.
  /// The primary paths mirror close()'s handling of the primary fd, minus the
  /// real close() (dup2/dup3 already replaced the descriptor).
  void invalidate_overwritten_kfd_fd(int fd) {
    if (fd < 0)
      return;
    // Remote primary? Compare-and-clear remote_kfd_fd_ under remote_mutex_ so the
    // "is this the primary?" test and the clear are atomic with respect to
    // get_or_create_remote() (which retains/reissues the fd under the same lock).
    // Without the lock a racing open could observe the old fd number after this
    // cleared it, and hand back a stale/invalidated descriptor.
    if (invalidate_remote_primary(fd))
      return;
    // Local primary? Let invalidate_primary_fd() decide under the driver's own
    // lock rather than pre-filtering on an unlocked fd() load: fd() can change
    // between the check and the lock (a concurrent open()/re-mint), which could
    // skip invalidating the overwritten primary and leave the reused number
    // misclassified as KFD. The under-lock compare-and-clear is authoritative,
    // and its result tells us whether to drop an open reference:
    //   - kClearedDropRef: the primary held one counted open reference (e.g.
    //     SimulatedKfd) — drop it via close();
    //   - kClearedKeepRefs: the classification was cleared but the primary fd is
    //     internal and NOT counted in the open-reference bookkeeping (e.g.
    //     GuestKfd's hidden real fd, kept alive by app dups) — do NOT close();
    //   - kNotPrimary: fall through to dup tracking.
    if (auto *d = driver()) {
      switch (d->invalidate_primary_fd(fd)) {
      case LinuxKfd::PrimaryInvalidation::kClearedDropRef:
        d->close(); // drop the primary open reference
        return;
      case LinuxKfd::PrimaryInvalidation::kClearedKeepRefs:
        return; // classification cleared; no counted reference to drop
      case LinuxKfd::PrimaryInvalidation::kNotPrimary:
        break; // fall through to dup handling
      }
    }
    // Otherwise, a tracked dup (or nothing).
    untrack_dup(fd);
  }

  /// @brief If @p fd is the remote primary, clear it and drop its open reference.
  /// @returns true if @p fd matched the remote primary (handled here); false so
  /// the caller falls through to local-primary / dup handling.
  /// @details The compare-and-clear and the reference drop run under
  /// remote_mutex_, serialized with get_or_create_remote()'s retain/reissue. On
  /// the last reference the connection is torn down and the (blocking) RPC_CLOSE
  /// is run OUTSIDE the lock, mirroring release_remote_open().
  bool invalidate_remote_primary(int fd) {
    std::shared_ptr<RemoteDriver> dead;
    {
      std::lock_guard lock(remote_mutex_);
      int expected = fd;
      if (!remote_kfd_fd_.compare_exchange_strong(expected, -1, std::memory_order_acq_rel))
        return false; // not the remote primary
      // The number is being reused; it no longer names the synthetic remote KFD
      // fd. Drop the primary's open reference (inlined from release_remote_open()
      // because we already hold remote_mutex_). If other refs keep the connection
      // alive, remote_ stays non-null with remote_kfd_fd_ == -1; routing then uses
      // ctx.remote() (not the fd number) and a later open("/dev/kfd") re-mints a
      // fresh primary fd via get_or_create_remote() without reconnecting.
      int prev = remote_open_refs_.load(std::memory_order_acquire);
      if (prev > 0) {
        remote_open_refs_.store(prev - 1, std::memory_order_release);
        if (prev == 1)
          dead = teardown_remote_locked();
      }
    }
    if (dead)
      dead->close();
    return true;
  }

  void clear_dups() {
    std::vector<DupBackend> released;
    {
      std::lock_guard lock(fd_mutex_);
      released.reserve(kfd_dup_fds_.size());
      for (auto &[fd, backend] : kfd_dup_fds_)
        released.push_back(backend);
      kfd_dup_fds_.clear();
    }
    // Drop the references the cleared dups were holding, each on the backend it
    // was tracked against. Used when a fresh open() rebinds the local process;
    // the just-opened reference is preserved because clear_dups runs before any
    // new dups are tracked.
    for (DupBackend backend : released)
      release_backend(backend);
  }

  /// @brief Tear down the remote connection state. Caller MUST hold remote_mutex_.
  /// @details Only invoked from release_remote_open() on the last reference, which
  /// already holds remote_mutex_. Keeping the lock across the decrement, the
  /// pointer/refcount reset, and the fd/dup cleanup makes the whole "last release
  /// destroys" decision atomic with respect to retain_remote_open_if_active() and
  /// get_or_create_remote(). Any remaining Remote-tagged dup fds refer to the
  /// now-closed synthetic fd; erasing them makes their later close()/ioctl fall
  /// through to the real syscall instead of a dead RPC connection.
  /// @returns The extracted RemoteDriver so the caller can run the (blocking)
  /// RPC_CLOSE shutdown OUTSIDE remote_mutex_; the shared_ptr keeps it alive.
  [[nodiscard]] std::shared_ptr<RemoteDriver> teardown_remote_locked() {
    auto active_remote = remote_.load(std::memory_order_acquire);
    if (!active_remote)
      return nullptr;
    // Clear the published pointer first so no new reader can pick this driver up.
    // The RemoteDriver is destroyed by the last shared_ptr: if a racing reader
    // still holds a snapshot mid-ioctl/mmap, destruction (and socket close in
    // ~RemoteDriver) is deferred until it releases, so there is no use-after-free.
    remote_.store(nullptr, std::memory_order_release);
    // Reset the refcount to 0 under the lock. A concurrent
    // retain_remote_open_if_active() serializes on remote_mutex_ and, seeing
    // remote_ == nullptr, refuses to add a reference, so it cannot resurrect this
    // torn-down connection.
    remote_open_refs_.store(0, std::memory_order_release);
    int fd = remote_kfd_fd_.exchange(-1, std::memory_order_acq_rel);
    if (fd >= 0)
      InterposerContext::real().close(fd);
    // Erase ONLY the Remote-tagged dups. kfd_dup_fds_ can hold Local entries too
    // (local + daemon mode simultaneously active); clearing the whole map would
    // drop a live local dup's tracking without releasing its local open
    // reference, leaking it and breaking its later close. The remote refcount is
    // zero here, so every remaining Remote entry has already had its reference
    // released via untrack_dup()/clear_dups(); we only drop stale tracking.
    {
      std::lock_guard fd_lock(fd_mutex_);
      std::erase_if(kfd_dup_fds_,
                    [](const auto &entry) { return entry.second == DupBackend::Remote; });
    }
    return active_remote;
  }

  void track_sysfs(int fd, const std::string &path) {
    std::lock_guard lock(fd_mutex_);
    sysfs_fds_[fd] = path;
  }

  std::string lookup_sysfs(int fd) {
    std::lock_guard lock(fd_mutex_);
    auto it = sysfs_fds_.find(fd);
    return (it != sysfs_fds_.end()) ? it->second : std::string{};
  }

  void untrack_sysfs(int fd) {
    std::lock_guard lock(fd_mutex_);
    sysfs_fds_.erase(fd);
  }

  void track_drm(int fd, uint32_t render_minor = 128) {
    std::lock_guard lock(fd_mutex_);
    drm_fds_[fd] = render_minor;
  }

  bool is_drm(int fd) {
    std::lock_guard lock(fd_mutex_);
    return drm_fds_.count(fd) != 0;
  }

  uint32_t drm_render_minor(int fd) {
    std::lock_guard lock(fd_mutex_);
    auto it = drm_fds_.find(fd);
    return (it != drm_fds_.end()) ? it->second : 128;
  }

  bool untrack_drm(int fd) {
    std::lock_guard lock(fd_mutex_);
    return drm_fds_.erase(fd) != 0;
  }

  bool create_local_vm(const std::string &config_path) {
    rj_vm_t *created_vm = nullptr;
    if (rj_vm_create(config_path.c_str(), RJ_VM_MODE_LOCAL, &created_vm) !=
        ROCJITSU_STATUS_SUCCESS) {
      util::Logger::debug_print("rocjitsu: failed to create VM from ", config_path);
      return false;
    }
    rj_vm_ = created_vm;
    if (!rj_vm_->vm || !rj_vm_->vm->driver() || rj_vm_->vm->driver()->fd() < 0) {
      util::Logger::debug_print("rocjitsu: local VM did not acquire a KFD open");
      rj_vm_destroy(rj_vm_);
      rj_vm_ = nullptr;
      return false;
    }

    if (rj_vm_->soc) {
      std::shared_ptr<rocjitsu::ExecutionPluginGroup> pg;
      if (std::getenv("RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP"))
        pg = std::make_shared<rocjitsu::ProfiledExecutionPluginGroup>();
      else
        pg = std::make_shared<rocjitsu::ExecutionPluginGroup>();

      std::string sinks_str = "stderr";
      if (const char *s = std::getenv("RJ_SINKS"))
        sinks_str = s;
      std::istringstream ss(sinks_str);
      std::string token;
      while (std::getline(ss, token, ',')) {
        if (token == "stderr")
          pg->add_sink(&rocjitsu::StderrSink::instance());
        else if (token == "stdout")
          pg->add_sink(&rocjitsu::StdoutSink::instance());
        else if (token == "file") {
          const char *dir = std::getenv("RJ_SINK_DIR");
          if (dir)
            pg->set_sink_dir(dir);
        }
      }

      if (const char *rj_log = std::getenv("RJ_LOG"); rj_log && std::string(rj_log) == "1") {
        pg->add(std::unique_ptr<rocjitsu::ExecutionPlugin>(createKernelLoggingPlugin()));
        fprintf(stderr, "[rocjitsu] Logging enabled (RJ_LOG)\n");
      }

      if (const char *race = std::getenv("RJ_RACE"); race && std::string(race) == "1") {
        pg->add(std::unique_ptr<rocjitsu::ExecutionPlugin>(createRaceDetectorPlugin()));
        fprintf(stderr, "[rocjitsu] Race detection enabled (RJ_RACE)\n");
      }
      rj_vm_->soc->set_plugin_group(pg);
    }
    return true;
  }

  void destroy_local_vm() {
    if (!rj_vm_)
      return;
    rj_vm_destroy(rj_vm_);
    rj_vm_ = nullptr;
  }

  void start_local_vm() {
    assert(rj_vm_ != nullptr);
    std::thread([vm = rj_vm_]() { rj_vm_run(vm, nullptr); }).detach();
  }

  LinuxKfd *get_or_create() {
    std::lock_guard lock(init_mutex_);
    if (active_driver_.load(std::memory_order_acquire) == nullptr) {
      in_construction = true;
      std::optional<std::string> cfg_path = child_config_path();
      if (!cfg_path) {
        util::Logger::debug_print("rocjitsu: no child config path");
        in_construction = false;
        return nullptr;
      }

      try {
        auto dbt_guest = rocjitsu::config::load_dbt_guest_config_from_file(*cfg_path);
        if (dbt_guest.enabled) {
          LinuxKfd *execution_driver = nullptr;
          const bool simulator_backend =
              dbt_guest.host.backend == rocjitsu::config::DbtExecutionBackend::Simulator;
          if (simulator_backend) {
            const std::string host_config_path = rocjitsu::config::resolve_dbt_host_config_path(
                *cfg_path, dbt_guest.host.simulator_config_path);
            if (!create_local_vm(host_config_path)) {
              in_construction = false;
              return nullptr;
            }
            execution_driver = rj_vm_->vm->driver();
            rocjitsu::config::validate_dbt_simulator_device_limits(dbt_guest,
                                                                   rj_vm_->loaded.device);
          }

          auto guest_driver = std::make_unique<GuestKfd>(std::move(dbt_guest), execution_driver);
          if (!guest_driver->prepare_for_discovery()) {
            // GuestKfd owns the simulator's bootstrap open, so destroy it while
            // the VM and execution driver are still alive.
            guest_driver.reset();
            destroy_local_vm();
            in_construction = false;
            return nullptr;
          }
          auto *driver = guest_driver.get();
          guest_driver_ = std::move(guest_driver);
          active_driver_.store(driver, std::memory_order_release);
          if (simulator_backend)
            start_local_vm();
          in_construction = false;
          return driver;
        }
      } catch (const std::exception &e) {
        util::Logger::debug_print("rocjitsu: failed to load child config: ", e.what());
        destroy_local_vm();
        in_construction = false;
        return nullptr;
      }
      if (!create_local_vm(*cfg_path)) {
        in_construction = false;
        return nullptr;
      }

      // Publish the fully-constructed, not-yet-running driver BEFORE starting the
      // engine thread: the release-store of active_driver_ pairs with acquire
      // loads in driver()/initialized(), so any reader that observes the driver
      // also observes all of the setup above (config, plugin group, sinks).
      // Starting rj_vm_run() before the store would let the detached engine
      // thread begin mutating the VM before it is published.
      LinuxKfd *driver = rj_vm_->vm->driver();
      active_driver_.store(driver, std::memory_order_release);
      start_local_vm();
      in_construction = false;
    }
    return driver();
  }

  static int fopen_flags_from_mode(const char *mode) {
    bool plus = std::strchr(mode, '+') != nullptr;
    switch (mode[0]) {
    case 'w':
      return (plus ? O_RDWR : O_WRONLY) | O_CREAT | O_TRUNC;
    case 'a':
      return (plus ? O_RDWR : O_WRONLY) | O_CREAT | O_APPEND;
    default:
      return plus ? O_RDWR : O_RDONLY;
    }
  }

private:
  rj_vm_t *rj_vm_ = nullptr;
  std::unique_ptr<GuestKfd> guest_driver_;
  std::atomic<LinuxKfd *> active_driver_{nullptr};
  /// @brief Active daemon-mode remote driver, or nullptr in local mode.
  /// @details Held as an atomic shared_ptr so lock-free readers (`remote()`,
  /// `remote_lookup()`, `initialized()`, the AMDKFD ioctl fallback, the mmap
  /// path) each take a lifetime-extending snapshot: a racing `teardown_remote()`
  /// that stores nullptr cannot free the object while another thread still holds
  /// a snapshot in `remote->ioctl()`/`remote->mmap()`. The object is destroyed by
  /// the last shared_ptr, not by a manual delete, so there is no use-after-free.
  /// `remote_mutex_` still serializes the compound create+open and clear
  /// sequences; the atomic makes the pointer swap data-race-free.
  std::atomic<std::shared_ptr<RemoteDriver>> remote_{nullptr};
  std::atomic<int> remote_kfd_fd_{-1};
  /// @brief Open-reference count for the remote (daemon-mode) KFD connection.
  /// @details The primary remote fd and every dup of it each hold one
  /// reference; the RPC connection is torn down only when the last reference is
  /// released. Mirrors SimulatedKfd's local open refcount for daemon mode.
  std::atomic<int> remote_open_refs_{0};

  std::mutex init_mutex_;
  std::mutex fd_mutex_;
  std::mutex remote_mutex_;
  std::unordered_map<int, std::string> sysfs_fds_;
  std::unordered_map<int, uint32_t> drm_fds_;
  /// @brief Tracked KFD dup fds → the backend that holds their open reference.
  /// @details Each dup of a KFD fd retains one open reference. The backend is
  /// captured at track time so untrack releases the SAME backend even if the
  /// active backend changed (local↔remote) or was torn down in between; a dup is
  /// recorded only if a reference was actually acquired, so track/untrack stay
  /// balanced and can never over-release or resurrect the wrong connection.
  std::unordered_map<int, DupBackend> kfd_dup_fds_;

  alignas(16) static uint8_t storage_[];
};

// Storage for the singleton is never destructed. Using aligned raw storage
// avoids __cxa_finalize destroying the object while the detached engine
// thread is still running.
alignas(16) uint8_t InterposerContext::storage_[sizeof(InterposerContext)];
InterposerContext &InterposerContext::ctx =
    *reinterpret_cast<InterposerContext *>(InterposerContext::storage_);

__attribute__((constructor)) static void init_interposer() { InterposerContext::init(); }

} // namespace

extern "C" {

static std::string redirect_sysfs_path(const char *path);
static std::string redirect_sys_dev_char(const char *path);
static std::optional<Sysfs::GpuInfo> interposer_gpu_info(uint32_t render_minor);

struct SyntheticDrmOpenResult {
  bool handled = false;
  int fd = -1;
};

bool parse_render_minor_suffix(const char *first, const char *last, uint32_t *render_minor) {
  uint32_t parsed = 0;
  auto result = std::from_chars(first, last, parsed);
  if (result.ec != std::errc{} || result.ptr != last)
    return false;
  *render_minor = parsed;
  return true;
}

bool render_minor_from_drm_node_path(const char *raw_path, const char *drm_base,
                                     uint32_t *render_minor) {
  std::string_view path(raw_path);
  static constexpr std::string_view kRealRenderPrefix = "/dev/dri/renderD";
  if (path.starts_with(kRealRenderPrefix))
    return parse_render_minor_suffix(path.data() + kRealRenderPrefix.size(),
                                     path.data() + path.size(), render_minor);

  if (!drm_base || drm_base[0] == '\0')
    return false;

  std::string redirected_render_prefix = std::string(drm_base) + "/dev_dri/renderD";
  if (!path.starts_with(redirected_render_prefix))
    return false;
  return parse_render_minor_suffix(path.data() + redirected_render_prefix.size(),
                                   path.data() + path.size(), render_minor);
}

static SyntheticDrmOpenResult open_synthetic_drm_fd(const char *path) {
  if (!path)
    return {};

  std::string_view path_view(path);
  if (!path_view.starts_with("/dev/dri/renderD") &&
      path_view.find("/dev_dri/renderD") == std::string_view::npos)
    return {};

  if (!InterposerContext::ctx.initialized())
    InterposerContext::ctx.get_or_create();

  // Snapshot the remote once; a live snapshot (not remote_kfd_fd() >= 0) is the
  // stable signal that a daemon connection can service this render node, even if
  // a dup2/dup3 cleared the primary fd number while other refs keep it alive.
  auto remote = InterposerContext::ctx.remote();
  if (InterposerContext::ctx.driver_fd() < 0 && !remote)
    return {};

  std::string drm_base;
  if (auto *drv = InterposerContext::ctx.driver())
    drm_base = drv->drm_path();
  else
    drm_base = InterposerContext::ctx.remote_drm_path();

  // HIP/libdrm may open the generated dev_dri node after following redirected
  // sysfs metadata instead of opening the literal /dev/dri/renderD* path.
  // Treat both path forms as the same synthetic DRM render node.
  uint32_t render_minor = 0;
  if (!render_minor_from_drm_node_path(path, drm_base.c_str(), &render_minor))
    return {};

  auto *drv = InterposerContext::ctx.driver();
  const bool local_handles_render = drv && drv->handles_drm_render_minor(render_minor);
  const bool remote_handles_render = static_cast<bool>(remote);
  if (!local_handles_render && !remote_handles_render)
    return {};

  auto raw_drm_fd = InterposerContext::real().memfd_create("rocjitsu_drm", MFD_CLOEXEC);
  if (raw_drm_fd < 0)
    return {true, -1};

  // Use real().fcntl, not the unqualified fcntl: this TU defines the interposed
  // fcntl with external linkage, so an unqualified call would re-enter our own
  // hook (reserve_dup_backend/untrack_dup, fd_mutex_) needlessly.
  int high_fd = InterposerContext::real().fcntl(raw_drm_fd, F_DUPFD_CLOEXEC, 512);
  int saved_errno = errno;
  InterposerContext::real().close(raw_drm_fd);
  if (high_fd < 0) {
    errno = saved_errno;
    return {true, -1};
  }

  InterposerContext::ctx.track_drm(high_fd, render_minor);
  return {true, high_fd};
}

RJ_INTERPOSER_EXPORT int open(const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }

  assert(InterposerContext::real().ready());
  auto *volatile p = path;
  if (!p || InterposerContext::in_construction)
    return InterposerContext::real().openat(AT_FDCWD, path, flags, mode);

  if (auto drm_fd = open_synthetic_drm_fd(path); drm_fd.handled)
    return drm_fd.fd;

  if (std::strcmp(path, "/dev/kfd") == 0) {
    // Use the fd captured under remote_mutex_ (not a fresh remote_kfd_fd() read),
    // so a concurrent dup2/dup3 that invalidates the primary fd cannot make this
    // return -1 or a reused non-KFD descriptor.
    if (auto remote = InterposerContext::ctx.get_or_create_remote())
      return remote.fd;

    auto *drv = InterposerContext::ctx.get_or_create();
    if (!drv) {
      errno = ENODEV;
      return -1;
    }
    int kfd_fd = drv->open();
    if (kfd_fd < 0)
      return kfd_fd;
    if (kfd_fd != drv->fd())
      InterposerContext::ctx.track_open_fd(kfd_fd);
    if (!drv->owns_fd(drv->fd()))
      InterposerContext::ctx.clear_dups();
    return kfd_fd;
  }

  std::string redirected = InterposerContext::ctx.redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (!redirected.empty()) {
    int fd = InterposerContext::real().openat(AT_FDCWD, redirected.c_str(), flags, mode);
    if (fd >= 0)
      InterposerContext::ctx.track_sysfs(fd, redirected);
    return fd;
  }

  return InterposerContext::real().openat(AT_FDCWD, path, flags, mode);
}

RJ_INTERPOSER_EXPORT int open64(const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }
  return open(path, flags, mode);
}

RJ_INTERPOSER_EXPORT int __open_2(const char *path, int oflag) { return open(path, oflag, 0); }

RJ_INTERPOSER_EXPORT int __open64_2(const char *path, int oflag) { return open(path, oflag, 0); }

RJ_INTERPOSER_EXPORT int __openat_2(int dirfd, const char *path, int oflag) {
  return openat(dirfd, path, oflag, 0);
}

RJ_INTERPOSER_EXPORT int __openat64_2(int dirfd, const char *path, int oflag) {
  return openat(dirfd, path, oflag, 0);
}

RJ_INTERPOSER_EXPORT int openat(int dirfd, const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }

  auto *volatile p_at = path;
  if (!p_at)
    return InterposerContext::real().openat(dirfd, path, flags, mode);
  if (InterposerContext::in_construction)
    return InterposerContext::real().openat(dirfd, path, flags, mode);

  if (path[0] == '/') {
    if (auto drm_fd = open_synthetic_drm_fd(path); drm_fd.handled)
      return drm_fd.fd;

    std::string redirected = InterposerContext::ctx.redirect_sysfs_path(path);
    if (redirected.empty())
      redirected = redirect_sys_dev_char(path);
    if (!redirected.empty()) {
      int fd = InterposerContext::real().openat(AT_FDCWD, redirected.c_str(), flags, mode);
      if (fd >= 0)
        InterposerContext::ctx.track_sysfs(fd, redirected);
      return fd;
    }
  } else if (dirfd != AT_FDCWD) {
    auto dir_path = InterposerContext::ctx.lookup_sysfs(dirfd);
    if (!dir_path.empty()) {
      std::string full = dir_path + "/" + path;
      int fd = InterposerContext::real().openat(AT_FDCWD, full.c_str(), flags, mode);
      if (fd >= 0)
        InterposerContext::ctx.track_sysfs(fd, full);
      return fd;
    }
  }

  return InterposerContext::real().openat(dirfd, path, flags, mode);
}

RJ_INTERPOSER_EXPORT int openat64(int dirfd, const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }
  return openat(dirfd, path, flags, mode);
}

RJ_INTERPOSER_EXPORT int close(int fd) {
  assert(InterposerContext::real().ready());
  if (InterposerContext::ctx.remote_lookup(fd)) {
    // Closing the primary remote KFD fd drops one open reference; the synthetic
    // fd and RPC connection are torn down only when the last reference is
    // released (teardown_remote), mirroring local-mode primary close which also
    // defers teardown to the last reference.
    InterposerContext::ctx.release_remote_open();
    return 0;
  }
  InterposerContext::ctx.untrack_sysfs(fd);
  if (InterposerContext::ctx.untrack_drm(fd)) {
    InterposerContext::real().close(fd);
    return 0;
  }
  if (InterposerContext::ctx.is_kfd_dup(fd)) {
    InterposerContext::ctx.untrack_dup(fd);
    return static_cast<int>(InterposerContext::real().close(fd));
  }
  if (auto *drv = InterposerContext::ctx.lookup(fd)) {
    drv->close();
    return 0;
  }
  if (InterposerContext::ctx.owns_fd(fd))
    return 0;
  return static_cast<int>(InterposerContext::real().close(fd));
}

__attribute__((destructor(101))) void rj_interposer_shutdown() {}

RJ_INTERPOSER_EXPORT int ioctl(int fd, unsigned long request, ...) {
  assert(InterposerContext::real().ready());
  va_list ap;
  va_start(ap, request);
  void *arg = va_arg(ap, void *);
  va_end(ap);

  constexpr unsigned kDrmIoctlType = 'd';
  constexpr unsigned kDrmIoctlNrVersion = 0x00;
  constexpr unsigned kDrmIoctlNrAmdgpuInfo = DRM_COMMAND_BASE + DRM_AMDGPU_INFO;
  constexpr unsigned kDrmIoctlNrPrimeFdToHandle = 0x2e;

  if (InterposerContext::ctx.is_drm(fd)) {
    unsigned nr = _IOC_NR(request);
    unsigned type = _IOC_TYPE(request);
    if (type == kDrmIoctlType && nr == kDrmIoctlNrVersion && arg) {
      auto *ver = static_cast<drm_version *>(arg);
      ver->version_major = 3;
      ver->version_minor = 57;
      ver->version_patchlevel = 0;
      static constexpr const char drv_name[] = "amdgpu";
      constexpr size_t kNameStrLen = sizeof(drv_name) - 1;
      // Mirror the kernel's drm_version contract: copy at most the caller's
      // advertised buffer length, and only write the NUL terminator when the
      // buffer has room for it. A caller that sized name to exactly the queried
      // length must not get a terminator written one byte past the end.
      if (ver->name && ver->name_len > 0) {
        size_t copy = ver->name_len < kNameStrLen ? ver->name_len : kNameStrLen;
        std::memcpy(ver->name, drv_name, copy);
        if (ver->name_len > kNameStrLen)
          ver->name[kNameStrLen] = '\0';
      }
      ver->name_len = kNameStrLen;
      if (ver->date && ver->date_len > 0)
        ver->date[0] = '\0';
      ver->date_len = 1;
      if (ver->desc && ver->desc_len > 0)
        ver->desc[0] = '\0';
      ver->desc_len = 1;
      return 0;
    }
    if (type == kDrmIoctlType && nr == kDrmIoctlNrPrimeFdToHandle && arg) {
      struct drm_prime_handle {
        uint32_t handle;
        uint32_t flags;
        int32_t fd;
      };
      auto *prime = static_cast<drm_prime_handle *>(arg);
      if (prime->fd < 0) {
        errno = EINVAL;
        return -1;
      }
      prime->handle = static_cast<uint32_t>(prime->fd) + 1u;
      return 0;
    }
    if (type == kDrmIoctlType && nr == kDrmIoctlNrAmdgpuInfo && arg) {
      // Service the AMDGPU_INFO queries that real libdrm_amdgpu issues during
      // amdgpu_device_initialize / amdgpu_query_gpu_info_init. Answering these
      // at the ioctl layer lets real libdrm run unmodified (no library shim).
      // The init cascade (amdgpu_gpu_info.c) requires, in order:
      //   ACCEL_WORKING (must be nonzero or init aborts), DEV_INFO,
      //   READ_MMR_REG (gb_addr_cfg is mandatory for all families),
      //   VRAM_GTT, MEMORY. Failures (-1) abort device init.
      auto *info = static_cast<drm_amdgpu_info *>(arg);
      auto gpu_info = interposer_gpu_info(InterposerContext::ctx.drm_render_minor(fd));
      if (!gpu_info) {
        errno = ENODEV;
        return -1;
      }
      const Sysfs::GpuInfo *gpu = &*gpu_info;
      auto *out = info->return_pointer ? reinterpret_cast<void *>(info->return_pointer) : nullptr;
      if (!out || info->return_size == 0)
        return 0;
      std::memset(out, 0, info->return_size);

      switch (info->query) {
      case AMDGPU_INFO_ACCEL_WORKING: {
        if (info->return_size >= sizeof(uint32_t))
          *static_cast<uint32_t *>(out) = 1u;
        return 0;
      }
      case AMDGPU_INFO_READ_MMR_REG: {
        // rocjitsu does not model raster/tiling MMRs. libdrm only stores the
        // returned words (never validates them), so zero-fill `count` u32s is
        // sufficient for both the AI short path and the pre-AI cascade.
        return 0; // buffer already zeroed
      }
      case AMDGPU_INFO_VRAM_GTT: {
        if (info->return_size >= sizeof(drm_amdgpu_info_vram_gtt)) {
          auto *vg = static_cast<drm_amdgpu_info_vram_gtt *>(out);
          vg->vram_size = gpu->local_mem_size;
          vg->vram_cpu_accessible_size = gpu->local_mem_size;
          vg->gtt_size = gpu->local_mem_size;
        }
        return 0;
      }
      case AMDGPU_INFO_MEMORY: {
        if (info->return_size >= sizeof(drm_amdgpu_memory_info)) {
          auto *m = static_cast<drm_amdgpu_memory_info *>(out);
          m->vram.total_heap_size = gpu->local_mem_size;
          m->vram.usable_heap_size = gpu->local_mem_size;
          m->vram.max_allocation = gpu->local_mem_size;
          m->cpu_accessible_vram = m->vram;
          m->gtt = m->vram;
        }
        return 0;
      }
      case AMDGPU_INFO_DEV_INFO: {
        if (info->return_size >= sizeof(drm_amdgpu_info_device)) {
          auto *dev = static_cast<drm_amdgpu_info_device *>(out);
          dev->device_id = gpu->device_id;
          dev->chip_rev = gpu->revision_id;
          dev->external_rev = rocjitsu::kmd::external_rev_id_for_gfx_target_version(
              gpu->gfx_target_version, gpu->revision_id);
          dev->pci_rev = gpu->pci_revision_id;
          dev->family = gpu->family_id;
          dev->num_shader_engines = rocjitsu::kmd::drm_shader_engine_count(
              gpu->num_shader_engines, gpu->num_shader_arrays_per_engine);
          dev->num_shader_arrays_per_engine = gpu->num_shader_arrays_per_engine;
          dev->gpu_counter_freq = 100000;
          dev->max_engine_clock = gpu->max_engine_clk_fcompute;
          dev->max_memory_clock = gpu->mem_clk_max;
          dev->wave_front_size = gpu->wave_front_size;
          dev->num_cu_per_sh = gpu->num_cu_per_sh;
          dev->num_hw_gfx_contexts =
              rocjitsu::kmd::num_hw_gfx_contexts_for_gfx_target_version(gpu->gfx_target_version);
          dev->vram_type = gpu->vram_type;
          dev->vram_bit_width = gpu->mem_width;
          dev->cu_active_number =
              rocjitsu::kmd::drm_cu_active_number(gpu->num_shader_engines, gpu->num_cu_per_sh);
          // VA aperture — libdrm's VA manager (amdgpu_vamgr_init) needs a sane
          // range. Mirror the KFD GPUVM aperture used elsewhere.
          dev->virtual_address_offset = 0x200000;       // 2 MiB
          dev->virtual_address_max = 0x800000000000ULL; // 47-bit canonical
          dev->virtual_address_alignment = 0x1000;      // 4 KiB
          dev->pte_fragment_size = 0x200000;            // 2 MiB
          dev->gart_page_size = 0x1000;                 // 4 KiB
          dev->high_va_offset = 0xffff800000000000ULL;
          dev->high_va_max = 0xffffffffffffffffULL;
        }
        return 0;
      }
      default:
        // Unhandled query: succeed with zero-filled buffer. libdrm tolerates
        // zeros for the optional queries (FW_VERSION, sensors, etc.).
        return 0;
      }
    }
    errno = EINVAL;
    return -1;
  }

  if (auto remote = InterposerContext::ctx.remote_lookup(fd))
    return kfd_ioctl_ret(remote->ioctl(request, arg));
  // Dispatch a tracked KFD fd by its RECORDED backend, not by "remote if any
  // remote is live". In mixed local+daemon mode a Local-tagged dup must route to
  // the local driver and a Remote-tagged dup to the remote connection; guessing
  // remote would silently switch a local dup's backend.
  if (auto backend = InterposerContext::ctx.kfd_backend_of(fd)) {
    if (*backend == InterposerContext::DupBackend::Remote) {
      // kfd_backend_of() already established this fd is Remote-backed, so route
      // via the remote snapshot directly rather than remote_lookup(remote_kfd_fd_):
      // the primary fd number may have been invalidated/reused while a remote
      // shared_ptr snapshot is still live, and this dup still belongs to it.
      if (auto remote = InterposerContext::ctx.remote())
        return kfd_ioctl_ret(remote->ioctl(request, arg));
    } else if (auto *drv = InterposerContext::ctx.driver()) {
      return kfd_ioctl_ret(drv->ioctl(request, arg));
    }
  }
  // Late-ioctl safety net: an AMDKFD ('K') ioctl may arrive on a tracked KFD fd
  // whose primary remote handle changed underneath it (e.g. a close/dup race in
  // daemon mode). Forward only AMDKFD-typed ioctls, and only on fds whose backend
  // is KNOWN to be Remote. Capture the backend once (a single locked lookup) and
  // require it to hold a Remote value: this both excludes Local-backed fds (a
  // Local dup whose driver() was transiently null above belongs to the local
  // driver) and refuses to route when the backend is unknown/nullopt (e.g.
  // tracking removed concurrently), so a type-'K' ioctl is never guessed onto the
  // remote connection.
  if (_IOC_TYPE(request) == AMDKFD_IOCTL_BASE) {
    auto backend = InterposerContext::ctx.kfd_backend_of(fd);
    if (backend == InterposerContext::DupBackend::Remote) {
      if (auto remote = InterposerContext::ctx.remote())
        return kfd_ioctl_ret(remote->ioctl(request, arg));
    }
  }

  // Every tracked KFD primary/dup is routed above by its recorded backend
  // (remote_lookup / kfd_backend_of). Deliberately NO local-driver fallback for
  // tracked dups here: a Remote-tagged dup observed during a remote teardown
  // window (remote_ cleared but its kfd_dup_fds_ entry not yet erased) must fall
  // through to the real ioctl, not be misrouted to a live local driver in mixed
  // local+daemon mode.
  return InterposerContext::real().ioctl(fd, request, arg);
}

RJ_INTERPOSER_EXPORT int dup(int oldfd) {
  assert(InterposerContext::real().ready());
  // Reserve the source backend's open reference BEFORE the syscall so a racing
  // last-close cannot tear the backend down between the dup and tracking the new
  // fd (which would leave a valid KFD dup untracked / unreferenced).
  auto reserved = InterposerContext::ctx.reserve_dup_backend(oldfd);
  int rc = InterposerContext::real().dup(oldfd);
  if (rc < 0) {
    // Syscall failed: roll back the reservation.
    if (reserved)
      InterposerContext::ctx.release_backend(*reserved);
    return rc;
  }
  if (reserved)
    InterposerContext::ctx.commit_dup(rc, *reserved);
  else
    InterposerContext::ctx.untrack_dup(rc);
  return rc;
}

namespace {
// Reconcile interposer tracking after a successful dup2/dup3(oldfd -> newfd),
// consuming a backend reservation taken (before the syscall) for oldfd.
// dup2/dup3 atomically CLOSE newfd before installing the duplicate, so any
// tracking/reference newfd previously held must be dropped before the
// replacement is tracked:
//   - stale sysfs/DRM tracking for newfd is removed;
//   - if newfd was a tracked KFD dup, its reference is released and its stale
//     kfd_dup_fds_ entry erased (otherwise commit_dup would see the stale entry
//     and release the newly-reserved backend, leaving the OLD backend recorded);
//   - if newfd was a PRIMARY KFD fd (local or remote), that primary identity is
//     invalidated and its reference released, so the reused fd number no longer
//     routes to the old backend.
// Then the replacement is recorded on the reserved source backend.
void reconcile_dup_target(int newfd, std::optional<InterposerContext::DupBackend> reserved) {
  InterposerContext::ctx.untrack_sysfs(newfd);
  InterposerContext::ctx.untrack_drm(newfd);
  InterposerContext::ctx.invalidate_overwritten_kfd_fd(newfd);
  if (reserved)
    InterposerContext::ctx.commit_dup(newfd, *reserved);
}
} // namespace

RJ_INTERPOSER_EXPORT int dup2(int oldfd, int newfd) {
  assert(InterposerContext::real().ready());
  // dup2(fd, fd) is a POSIX no-op that leaves the descriptor live; mutating
  // tracking would drop a still-open ref. Forward without touching tracking.
  if (oldfd == newfd)
    return InterposerContext::real().dup2(oldfd, newfd);
  // Reserve the source backend before the syscall (see dup()).
  auto reserved = InterposerContext::ctx.reserve_dup_backend(oldfd);
  int rc = InterposerContext::real().dup2(oldfd, newfd);
  if (rc < 0) {
    if (reserved)
      InterposerContext::ctx.release_backend(*reserved);
    return rc;
  }
  reconcile_dup_target(rc, reserved);
  return rc;
}

#ifdef SYS_dup3
RJ_INTERPOSER_EXPORT int dup3(int oldfd, int newfd, int flags) {
  assert(InterposerContext::real().ready());
  // dup3(fd, fd, ...) is required to fail with EINVAL without altering the
  // descriptor; do not mutate tracking before the syscall confirms that.
  auto reserved = InterposerContext::ctx.reserve_dup_backend(oldfd);
  int rc = InterposerContext::real().dup3(oldfd, newfd, flags);
  if (rc < 0) {
    if (reserved)
      InterposerContext::ctx.release_backend(*reserved);
    return rc;
  }
  reconcile_dup_target(rc, reserved);
  return rc;
}
#endif

namespace {
enum class FcntlArgKind { None, Int, Ptr };

FcntlArgKind fcntl_arg_kind(int cmd) {
  switch (cmd) {
  case F_DUPFD:
  case F_DUPFD_CLOEXEC:
  case F_SETFD:
  case F_SETFL:
  case F_SETOWN:
  case F_SETSIG:
  case F_SETLEASE:
  case F_NOTIFY:
  case F_SETPIPE_SZ:
  case F_ADD_SEALS:
    return FcntlArgKind::Int;
#ifdef F_SETLK
  case F_SETLK:
  case F_SETLKW:
#endif
#if defined(F_SETLK64) && (!defined(F_SETLK) || F_SETLK64 != F_SETLK)
  case F_SETLK64:
  case F_SETLKW64:
#endif
  case F_GETLK:
#if defined(F_GETLK64) && (!defined(F_GETLK) || F_GETLK64 != F_GETLK)
  case F_GETLK64:
#endif
#ifdef F_GETOWNER_UIDS
  case F_GETOWNER_UIDS:
#endif
#ifdef F_GET_RW_HINT
  case F_GET_RW_HINT:
#endif
#ifdef F_SET_RW_HINT
  case F_SET_RW_HINT:
#endif
#ifdef F_GET_FILE_RW_HINT
  case F_GET_FILE_RW_HINT:
#endif
#ifdef F_SET_FILE_RW_HINT
  case F_SET_FILE_RW_HINT:
#endif
    return FcntlArgKind::Ptr;
#ifdef F_SETOWN_EX
  case F_SETOWN_EX:
    return FcntlArgKind::Ptr;
#endif
#ifdef F_GETOWN_EX
  case F_GETOWN_EX:
    return FcntlArgKind::Ptr;
#endif
  default:
    return FcntlArgKind::None;
  }
}
} // namespace

namespace {
// Shared implementation for fcntl / fcntl64. The variadic third argument is
// extracted by the public entry points (which can't forward a va_list) and
// passed here already resolved. Both fcntl and fcntl64 share the same kernel
// ABI, so InterposerContext::real().fcntl services both.
int fcntl_impl(int fd, int cmd, void *ptr_arg, int int_arg) {
  FcntlArgKind kind = fcntl_arg_kind(cmd);
  // F_DUPFD/F_DUPFD_CLOEXEC create a new fd like dup(); reserve the source
  // backend BEFORE the syscall so a racing last-close cannot tear it down
  // between the dup and tracking the new fd. F_DUPFD does not overwrite an
  // existing fd, so no primary/dup reconciliation of a target is needed.
  const bool is_dupfd = (cmd == F_DUPFD || cmd == F_DUPFD_CLOEXEC);
  std::optional<InterposerContext::DupBackend> reserved;
  if (is_dupfd)
    reserved = InterposerContext::ctx.reserve_dup_backend(fd);

  long rc = 0;
  switch (kind) {
  case FcntlArgKind::Int:
    rc = InterposerContext::real().fcntl(fd, cmd, int_arg);
    break;
  case FcntlArgKind::Ptr:
    rc = InterposerContext::real().fcntl(fd, cmd, ptr_arg);
    break;
  case FcntlArgKind::None:
  default:
    rc = InterposerContext::real().fcntl(fd, cmd, 0L);
    break;
  }

  if (is_dupfd) {
    if (rc < 0) {
      // Syscall failed: roll back the reservation.
      if (reserved)
        InterposerContext::ctx.release_backend(*reserved);
    } else {
      if (reserved)
        InterposerContext::ctx.commit_dup(static_cast<int>(rc), *reserved);
      else
        InterposerContext::ctx.untrack_dup(static_cast<int>(rc));
      // Propagate DRM render-node tracking across dup so ioctls on the duped fd
      // are still recognized. libdrm's amdgpu_device_initialize duplicates the
      // render fd (via fcntl64 F_DUPFD_CLOEXEC) and issues all AMDGPU_INFO ioctls
      // on the copy.
      if (InterposerContext::ctx.is_drm(fd))
        InterposerContext::ctx.track_drm(static_cast<int>(rc),
                                         InterposerContext::ctx.drm_render_minor(fd));
    }
  }
  return static_cast<int>(rc);
}
} // namespace

RJ_INTERPOSER_EXPORT int fcntl(int fd, int cmd, ...) {
  assert(InterposerContext::real().ready());
  va_list ap;
  va_start(ap, cmd);
  FcntlArgKind kind = fcntl_arg_kind(cmd);
  void *ptr_arg = nullptr;
  int int_arg = 0;
  if (kind == FcntlArgKind::Ptr)
    ptr_arg = va_arg(ap, void *);
  else if (kind == FcntlArgKind::Int)
    int_arg = va_arg(ap, int);
  va_end(ap);
  return fcntl_impl(fd, cmd, ptr_arg, int_arg);
}

// libdrm_amdgpu imports fcntl64@GLIBC_2.28 (not fcntl), so it must be
// interposed separately or libdrm's F_DUPFD_CLOEXEC on the render fd bypasses
// our dup tracking and subsequent ioctls land on an untracked fd.
RJ_INTERPOSER_EXPORT int fcntl64(int fd, int cmd, ...) {
  assert(InterposerContext::real().ready());
  va_list ap;
  va_start(ap, cmd);
  FcntlArgKind kind = fcntl_arg_kind(cmd);
  void *ptr_arg = nullptr;
  int int_arg = 0;
  if (kind == FcntlArgKind::Ptr)
    ptr_arg = va_arg(ap, void *);
  else if (kind == FcntlArgKind::Int)
    int_arg = va_arg(ap, int);
  va_end(ap);
  return fcntl_impl(fd, cmd, ptr_arg, int_arg);
}

RJ_INTERPOSER_EXPORT void *mmap(void *addr, size_t length, int prot, int flags, int fd,
                                off_t offset) {
  if (!InterposerContext::real().ready() || !InterposerContext::real().mmap)
    return raw_mmap_syscall(addr, length, prot, flags, fd, offset);

  assert(InterposerContext::real().ready());
  if (auto remote = InterposerContext::ctx.remote_lookup(fd))
    return remote->mmap(addr, length, prot, flags, offset);

  if (auto *drv = InterposerContext::ctx.lookup(fd))
    return drv->mmap(addr, length, prot, flags, offset);

  // Dispatch a tracked KFD dup by its RECORDED backend (as ioctl() does), not by
  // "remote if any remote is live"; otherwise a Local-tagged dup would misroute
  // to the remote in mixed local+daemon mode. For Remote, route via the remote
  // snapshot directly so routing still works if the primary fd number changed.
  if (auto backend = InterposerContext::ctx.kfd_backend_of(fd)) {
    if (*backend == InterposerContext::DupBackend::Remote) {
      if (auto remote = InterposerContext::ctx.remote())
        return remote->mmap(addr, length, prot, flags, offset);
    } else if (auto *drv = InterposerContext::ctx.driver()) {
      return drv->mmap(addr, length, prot, flags, offset);
    }
  }

  if (InterposerContext::ctx.is_drm(fd)) {
    // Route via the remote snapshot (not remote_lookup(remote_kfd_fd())): a
    // dup2/dup3 may have cleared the primary fd number while the connection is
    // still live via other refs, and this DRM fd still belongs to that remote.
    if (auto remote = InterposerContext::ctx.remote())
      return remote->mmap(addr, length, prot, flags, offset);
    if (auto *drv = InterposerContext::ctx.driver())
      return drv->mmap(addr, length, prot, flags, offset);
  }

  if (fd < 0 && (flags & MAP_FIXED) && prot != PROT_NONE && addr) {
    int memfd_out = -1;
    off_t memfd_offset = 0;
    auto remote_memfd = InterposerContext::ctx.remote();
    if (remote_memfd) {
      auto lookup = remote_memfd->find_memfd_for_addr(addr, length, &memfd_out, &memfd_offset);
      if (lookup == RemoteDriver::MemfdLookup::kDupFailed) {
        // A daemon-shared range covered this address but we could not obtain a
        // descriptor for it. Falling back to an anonymous mapping would silently
        // detach it from the shared memory, so fail the mmap instead. Preserve
        // the errno set by the failed dup (EMFILE/ENFILE/...) rather than
        // clobbering it, so the caller sees the real failure cause.
        return MAP_FAILED;
      }
      if (lookup == RemoteDriver::MemfdLookup::kFound) {
        // memfd_out is a caller-owned dup; close it once we are done. Its
        // validity is independent of a concurrent RemoteDriver teardown/close.
        auto total = static_cast<off_t>(length) + memfd_offset;
        [[maybe_unused]] auto ft_rc = ftruncate(memfd_out, total);
        fallocate(memfd_out, 0, memfd_offset, static_cast<off_t>(length));
        auto *raw = InterposerContext::real().mmap(
            addr, length, prot, (flags & ~MAP_ANONYMOUS) | MAP_SHARED, memfd_out, memfd_offset);
        // Preserve the mmap errno across close() (which may set its own).
        int mmap_errno = errno;
        InterposerContext::real().close(memfd_out);
        if (raw != MAP_FAILED) {
#ifdef MADV_POPULATE_WRITE
          InterposerContext::real().madvise(raw, length, MADV_POPULATE_WRITE);
#endif
          return raw;
        }
        // A daemon-shared range matched but the shared mapping failed. Fail
        // closed with the real errno rather than falling through to an anonymous
        // MAP_FIXED mapping, which would silently detach this GPUVM address from
        // the daemon's shared memory (same invariant as kDupFailed above).
        errno = mmap_errno;
        return MAP_FAILED;
      }
    }
  }
  return InterposerContext::real().mmap(addr, length, prot, flags, fd, offset);
}

RJ_INTERPOSER_EXPORT int mprotect(void *addr, size_t length, int prot) {
  assert(InterposerContext::real().ready());
  auto *drv = InterposerContext::ctx.driver();
  if (drv && drv->is_doorbell_range(addr, length)) {
    errno = EPERM;
    return -1;
  }
  return InterposerContext::real().mprotect(addr, length, prot);
}

RJ_INTERPOSER_EXPORT int madvise(void *addr, size_t length, int advice) {
  assert(InterposerContext::real().ready());
  if ((advice == MADV_HUGEPAGE || advice == MADV_DONTFORK) &&
      reinterpret_cast<uintptr_t>(addr) >= 0x1000000000ULL)
    return 0;
  return InterposerContext::real().madvise(addr, length, advice);
}

RJ_INTERPOSER_EXPORT int munmap(void *addr, size_t length) {
  if (!InterposerContext::real().ready() || !InterposerContext::real().munmap)
    return raw_munmap_syscall(addr, length);

  assert(InterposerContext::real().ready());
  // Address-based unmap: try the live remote snapshot regardless of whether the
  // primary fd number is currently valid (a dup2/dup3 may have cleared it while
  // the connection stays alive via other refs).
  if (auto remote = InterposerContext::ctx.remote()) {
    int ret = remote->munmap(addr, length);
    if (ret != -ENOENT)
      return ret;
  }
  auto *drv = InterposerContext::ctx.driver();
  if (drv) {
    int ret = drv->munmap(addr, length);
    if (ret != -ENOENT)
      return ret;
  }
  return InterposerContext::real().munmap(addr, length);
}

} // extern "C"

extern "C" {

// -- fopen / freopen interposition (sysfs redirect) --

RJ_INTERPOSER_EXPORT FILE *fopen(const char *path, const char *mode) {
  if (!InterposerContext::real().ready()) {
    auto fn = util::lookup_symbol<FILE *(*)(const char *, const char *)>(RTLD_NEXT, "fopen");
    return fn ? fn(path, mode) : nullptr;
  }
  if (!path || !mode)
    return nullptr;

  const char *actual = path;
  std::string redirected;
  if (!InterposerContext::in_construction) {
    redirected = InterposerContext::ctx.redirect_sysfs_path(path);
    if (redirected.empty())
      redirected = redirect_sys_dev_char(path);
    if (!redirected.empty())
      actual = redirected.c_str();
  }

  int fd = InterposerContext::real().openat(AT_FDCWD, actual,
                                            InterposerContext::fopen_flags_from_mode(mode), 0644);
  if (fd < 0)
    return nullptr;
  return fdopen(fd, mode);
}

RJ_INTERPOSER_EXPORT FILE *fopen64(const char *path, const char *mode) { return fopen(path, mode); }

RJ_INTERPOSER_EXPORT FILE *freopen(const char *path, const char *mode, FILE *stream) {
  if (!path || !mode)
    return nullptr;
  RJ_DIAGNOSTIC_PUSH
  RJ_DIAGNOSTIC_IGNORE_NONNULL_COMPARE
  if (stream)
    ::fclose(stream);
  RJ_DIAGNOSTIC_POP
  return fopen(path, mode);
}

RJ_INTERPOSER_EXPORT FILE *freopen64(const char *path, const char *mode, FILE *stream) {
  return freopen(path, mode, stream);
}

// -- stat/lstat/access interposition --

static std::string redirect_sysfs_path(const char *path) {
  if (!path || !InterposerContext::real().ready() || InterposerContext::in_construction)
    return {};
  return InterposerContext::ctx.redirect_sysfs_path(path);
}

static std::string redirect_sys_dev_char(const char *path) {
  if (!path || !InterposerContext::real().ready() || InterposerContext::in_construction)
    return {};
  std::string_view sv(path);
  constexpr std::string_view prefix = "/sys/dev/char/";
  if (!sv.starts_with(prefix))
    return {};

  auto rest = sv.substr(prefix.size());
  auto colon = rest.find(':');
  if (colon == std::string_view::npos)
    return {};

  uint32_t major_num = 0, minor_num = 0;
  if (std::from_chars(rest.data(), rest.data() + colon, major_num).ec != std::errc{} ||
      major_num != 226)
    return {};

  auto after_colon = rest.substr(colon + 1);
  auto slash_pos = after_colon.find('/');
  auto minor_end = (slash_pos != std::string_view::npos) ? after_colon.data() + slash_pos
                                                         : after_colon.data() + after_colon.size();
  if (std::from_chars(after_colon.data(), minor_end, minor_num).ec != std::errc{})
    return {};

  std::string drm_base;
  auto *drv = InterposerContext::ctx.driver();
  if (drv) {
    auto direct = drv->redirect_sysfs_path(path);
    if (!direct.empty())
      return direct;
    drm_base = drv->drm_path();
  } else {
    drm_base = InterposerContext::ctx.remote_drm_path();
  }
  if (drm_base.empty())
    return {};

  std::string entry = (minor_num >= 128) ? "renderD" + std::to_string(minor_num)
                                         : "card" + std::to_string(minor_num);
  std::string suffix;
  if (slash_pos != std::string_view::npos)
    suffix = std::string(after_colon.substr(slash_pos));

  return drm_base + "/" + entry + suffix;
}

// Return the GPU metadata BY VALUE. Returning a raw pointer into the local
// topology or the remote driver would dangle: the remote snapshot below is
// destroyed when this function returns, so a concurrent teardown could free the
// object while the caller still dereferenced the pointer. A copy is cheap and
// severs that lifetime dependency.
static std::optional<Sysfs::GpuInfo> interposer_gpu_info(uint32_t render_minor) {
  auto *drv = InterposerContext::ctx.driver();
  if (drv) {
    if (const Sysfs::GpuInfo *info = drv->gpu_info_for_render_minor(render_minor))
      return *info;
    return std::nullopt;
  }
  if (auto remote = InterposerContext::ctx.remote()) {
    if (const Sysfs::GpuInfo *info = remote->gpu_info())
      return *info;
  }
  return std::nullopt;
}

static std::string redirect_dev_dri(const char *path) {
  if (!path || !InterposerContext::real().ready() || InterposerContext::in_construction)
    return {};
  std::string_view sv(path);
  // Redirect both the /dev/dri directory and individual node files
  // (/dev/dri/renderD<minor>, /dev/dri/card<n>) into our synthetic dev_dri
  // tree. libdrm's drmGetMinorType probes node existence with access() on these
  // exact paths to classify an fd as a render node; without per-node redirect
  // the probe hits the real host (where extra GPUs don't exist) and fails,
  // breaking amdgpu_device_initialize's amdgpu_get_auth on multi-GPU configs.
  constexpr std::string_view kDevDri = "/dev/dri/";
  bool is_dir = (sv == "/dev/dri" || sv == "/dev/dri/");
  bool is_node = sv.starts_with(kDevDri) && (sv.substr(kDevDri.size()).starts_with("renderD") ||
                                             sv.substr(kDevDri.size()).starts_with("card"));
  if (!is_dir && !is_node)
    return {};
  std::string drm_base;
  auto *drv = InterposerContext::ctx.driver();
  if (drv) {
    auto direct = drv->redirect_sysfs_path(path);
    if (!direct.empty())
      return direct;
    drm_base = drv->drm_path();
  } else {
    drm_base = InterposerContext::ctx.remote_drm_path();
  }
  if (drm_base.empty())
    return {};
  if (is_dir)
    return drm_base + "/dev_dri";
  return drm_base + "/dev_dri/" + std::string(sv.substr(kDevDri.size()));
}

RJ_INTERPOSER_EXPORT int stat(const char *path, struct stat *buf) {
  if (!InterposerContext::real().ready()) {
    auto fn = util::lookup_symbol<int (*)(const char *, struct stat *)>(RTLD_NEXT, "stat");
    return fn ? fn(path, buf) : -1;
  }
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  if (!redirected.empty())
    return InterposerContext::real().stat(redirected.c_str(), buf);
  return InterposerContext::real().stat(path, buf);
}

RJ_INTERPOSER_EXPORT int lstat(const char *path, struct stat *buf) {
  if (!InterposerContext::real().ready()) {
    auto fn = util::lookup_symbol<int (*)(const char *, struct stat *)>(RTLD_NEXT, "lstat");
    return fn ? fn(path, buf) : -1;
  }
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  if (!redirected.empty())
    return InterposerContext::real().lstat(redirected.c_str(), buf);
  return InterposerContext::real().lstat(path, buf);
}

RJ_INTERPOSER_EXPORT int access(const char *path, int mode) {
  if (!InterposerContext::real().ready()) {
    auto fn = util::lookup_symbol<int (*)(const char *, int)>(RTLD_NEXT, "access");
    return fn ? fn(path, mode) : -1;
  }
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  if (!redirected.empty())
    return InterposerContext::real().access(redirected.c_str(), mode);
  return InterposerContext::real().access(path, mode);
}

// -- opendir interposition --

RJ_INTERPOSER_EXPORT DIR *opendir(const char *name) {
  if (!InterposerContext::real().ready()) {
    auto fn = util::lookup_symbol<DIR *(*)(const char *)>(RTLD_NEXT, "opendir");
    return fn ? fn(name) : nullptr;
  }
  auto *volatile p_od = name;
  if (!p_od) {
    errno = EINVAL;
    return nullptr;
  }
  if (!InterposerContext::in_construction) {
    std::string redirected = InterposerContext::ctx.redirect_sysfs_path(name);
    if (redirected.empty())
      redirected = redirect_sys_dev_char(name);
    if (redirected.empty())
      redirected = redirect_dev_dri(name);
    if (!redirected.empty())
      return InterposerContext::real().opendir(redirected.c_str());
  }
  return InterposerContext::real().opendir(name);
}

// -- fstat interposition (DRM memfd → synthetic st_rdev) --

RJ_INTERPOSER_EXPORT int fstat(int fd, struct stat *buf) {
  if (!InterposerContext::real().ready()) {
    auto fn = util::lookup_symbol<int (*)(int, struct stat *)>(RTLD_NEXT, "fstat");
    return fn ? fn(fd, buf) : -1;
  }
  int rc = InterposerContext::real().fstat_fn(fd, buf);
  if (rc == 0 && InterposerContext::ctx.is_drm(fd)) {
    uint32_t render_minor = InterposerContext::ctx.drm_render_minor(fd);
    buf->st_rdev = makedev(226, render_minor);
    buf->st_mode = (buf->st_mode & ~S_IFMT) | S_IFCHR;
  }
  return rc;
}

RJ_INTERPOSER_EXPORT int fstat64(int fd, struct stat64 *buf) {
  using fstat64_fn_t = int (*)(int, struct stat64 *);
  static fstat64_fn_t real_fstat64 = util::lookup_symbol<fstat64_fn_t>(RTLD_NEXT, "fstat64");
  if (!real_fstat64)
    return -1;
  int rc = real_fstat64(fd, buf);
  if (rc == 0 && InterposerContext::real().ready() && InterposerContext::ctx.is_drm(fd)) {
    uint32_t render_minor = InterposerContext::ctx.drm_render_minor(fd);
    buf->st_rdev = makedev(226, render_minor);
    buf->st_mode = (buf->st_mode & ~S_IFMT) | S_IFCHR;
  }
  return rc;
}

RJ_INTERPOSER_EXPORT int __fxstat(int ver, int fd, struct stat *buf) {
  using fxstat_fn_t = int (*)(int, int, struct stat *);
  static fxstat_fn_t real_fxstat = util::lookup_symbol<fxstat_fn_t>(RTLD_NEXT, "__fxstat");
  if (!real_fxstat)
    return -1;
  int rc = real_fxstat(ver, fd, buf);
  if (rc == 0 && InterposerContext::real().ready() && InterposerContext::ctx.is_drm(fd)) {
    uint32_t render_minor = InterposerContext::ctx.drm_render_minor(fd);
    buf->st_rdev = makedev(226, render_minor);
    buf->st_mode = (buf->st_mode & ~S_IFMT) | S_IFCHR;
  }
  return rc;
}

RJ_INTERPOSER_EXPORT int __fxstat64(int ver, int fd, struct stat64 *buf) {
  using fxstat64_fn_t = int (*)(int, int, struct stat64 *);
  static fxstat64_fn_t real_fxstat64 = util::lookup_symbol<fxstat64_fn_t>(RTLD_NEXT, "__fxstat64");
  if (!real_fxstat64)
    return -1;
  int rc = real_fxstat64(ver, fd, buf);
  if (rc == 0 && InterposerContext::real().ready() && InterposerContext::ctx.is_drm(fd)) {
    uint32_t render_minor = InterposerContext::ctx.drm_render_minor(fd);
    buf->st_rdev = makedev(226, render_minor);
    buf->st_mode = (buf->st_mode & ~S_IFMT) | S_IFCHR;
  }
  return rc;
}

// -- readlink interposition (redirect /sys/dev/char/) --

RJ_INTERPOSER_EXPORT ssize_t readlink(const char *path, char *buf, size_t bufsiz) {
  if (!InterposerContext::real().ready()) {
    auto fn = util::lookup_symbol<ssize_t (*)(const char *, char *, size_t)>(RTLD_NEXT, "readlink");
    return fn ? fn(path, buf, bufsiz) : -1;
  }
  auto redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_sysfs_path(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return InterposerContext::real().readlink_fn(actual, buf, bufsiz);
}

// -- stat64/lstat64 interposition (distinct from stat on glibc 2.33+) --

RJ_INTERPOSER_EXPORT int stat64(const char *path, struct stat64 *buf) {
  using stat64_fn_t = int (*)(const char *, struct stat64 *);
  static stat64_fn_t real_stat64 = util::lookup_symbol<stat64_fn_t>(RTLD_NEXT, "stat64");
  if (!real_stat64)
    return -1;
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return real_stat64(actual, buf);
}

RJ_INTERPOSER_EXPORT int lstat64(const char *path, struct stat64 *buf) {
  using lstat64_fn_t = int (*)(const char *, struct stat64 *);
  static lstat64_fn_t real_lstat64 = util::lookup_symbol<lstat64_fn_t>(RTLD_NEXT, "lstat64");
  if (!real_lstat64)
    return -1;
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return real_lstat64(actual, buf);
}

RJ_INTERPOSER_EXPORT int __xstat(int ver, const char *path, struct stat *buf) {
  using xstat_fn_t = int (*)(int, const char *, struct stat *);
  static xstat_fn_t real_xstat = util::lookup_symbol<xstat_fn_t>(RTLD_NEXT, "__xstat");
  if (!real_xstat)
    return -1;
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return real_xstat(ver, actual, buf);
}

RJ_INTERPOSER_EXPORT int __xstat64(int ver, const char *path, struct stat64 *buf) {
  using xstat64_fn_t = int (*)(int, const char *, struct stat64 *);
  static xstat64_fn_t real_xstat64 = util::lookup_symbol<xstat64_fn_t>(RTLD_NEXT, "__xstat64");
  if (!real_xstat64)
    return -1;
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return real_xstat64(ver, actual, buf);
}

RJ_INTERPOSER_EXPORT int __lxstat(int ver, const char *path, struct stat *buf) {
  using lxstat_fn_t = int (*)(int, const char *, struct stat *);
  static lxstat_fn_t real_lxstat = util::lookup_symbol<lxstat_fn_t>(RTLD_NEXT, "__lxstat");
  if (!real_lxstat)
    return -1;
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return real_lxstat(ver, actual, buf);
}

RJ_INTERPOSER_EXPORT int __lxstat64(int ver, const char *path, struct stat64 *buf) {
  using lxstat64_fn_t = int (*)(int, const char *, struct stat64 *);
  static lxstat64_fn_t real_lxstat64 = util::lookup_symbol<lxstat64_fn_t>(RTLD_NEXT, "__lxstat64");
  if (!real_lxstat64)
    return -1;
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return real_lxstat64(ver, actual, buf);
}

RJ_INTERPOSER_EXPORT pid_t fork() {
  assert(InterposerContext::real().ready());
  pid_t pid = InterposerContext::real().fork();
  if (pid == 0)
    InterposerContext::ctx.reset_after_fork();
  return pid;
}

} // extern "C"
