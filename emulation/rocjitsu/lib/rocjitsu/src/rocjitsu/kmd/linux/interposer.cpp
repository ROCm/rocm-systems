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
#include "rocjitsu/vm/plugins/plugin_loader.h"
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

#include <algorithm>
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
#include <pthread.h>
#include <shared_mutex>
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
#include <system_error>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

using rocjitsu::GuestKfd;
using rocjitsu::LinuxKfd;
using rocjitsu::RemoteDriver;
using rocjitsu::SimulatedKfd;
using rocjitsu::Sysfs;

namespace {

/// @brief Attempt to connect @p sock to the AF_UNIX socket at @p path.
/// @returns A connected socket fd on success; -1 with errno set on failure.
/// @details Creates a FRESH socket for each attempt: POSIX leaves a stream
/// socket's state unspecified after a failed connect(), so a socket must not be
/// reused for a second connect(). Fails with ENAMETOOLONG rather than silently
/// truncating a path that does not fit sun_path, so a too-long runtime dir cannot
/// connect to an unintended (truncated) socket endpoint.
int try_connect(const std::string &path) {
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (path.size() >= sizeof(addr.sun_path)) {
    errno = ENAMETOOLONG;
    return -1;
  }
  std::memcpy(addr.sun_path, path.c_str(), path.size());
  int sock = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (sock < 0)
    return -1;
  if (connect(sock, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
    int saved = errno;
    rocjitsu::libc_passthrough().close(sock);
    errno = saved;
    return -1;
  }
  return sock;
}

/// @brief Connect to the daemon for this invocation's per-PID runtime directory.
/// @details Connects to <runtime_dir>/daemon.sock. Only when that per-PID
/// directory does not exist (attach / daemon-only clients that share the
/// well-known location) does it fall back to rpc_default_socket_path(). The
/// fallback is gated on dir-absence rather than connect-failure so a daemon-mode
/// app is never silently cross-connected to an unrelated daemon at the shared
/// well-known socket if its own daemon's socket is transiently unavailable.
int connect_to_daemon(const std::string &runtime_dir) {
  int sock = try_connect(runtime_dir + "/daemon.sock");
  if (sock >= 0)
    return sock;
  // Preserve the per-PID connect() failure reason across the access() probe below:
  // access() overwrites errno on error and leaves it unspecified on success
  // (POSIX), so without saving it a caller that reaches the final `return -1`
  // would see an unrelated errno instead of the real connect failure.
  const int connect_errno = errno;

  // Fall back to the well-known socket only for invocations that never created a
  // per-PID directory (attach / daemon-only). access() goes through the real libc
  // so it does not re-enter this interposer's own path hooks. Gate strictly on the
  // dir genuinely not existing (ENOENT / a non-directory component, ENOTDIR): any
  // other error (EACCES/EPERM, transient IO) must NOT trigger the fallback, or a
  // daemon-mode client whose own dir is momentarily inaccessible could be silently
  // cross-connected to an unrelated daemon at the shared socket.
  const bool dir_absent = rocjitsu::libc_passthrough().access(runtime_dir.c_str(), F_OK) != 0 &&
                          (errno == ENOENT || errno == ENOTDIR);
  if (dir_absent)
    return try_connect(rocjitsu::rpc_default_socket_path());
  errno = connect_errno;
  return -1;
}

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

/// @brief Read the child-process rocjitsu config path from @p cfg_file.
///
/// @details The launcher writes the config path to a runtime file (per-PID
/// invocation directory) that the interposer reads back for both local
/// simulation and DBT guest mode.
std::optional<std::string> child_config_path(const std::string &cfg_file) {
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
    // Resolve the per-invocation runtime directory once here, in the library
    // constructor: this runs single-threaded before any app code (and thus before
    // any app fork). Writing it once here keeps invocation_runtime_dir() an
    // immutable, lock-free read and closes two hazards a lazy resolve would have:
    // (1) a data race — the accessor is reached from paths holding different locks
    // (remote_mutex_ vs init_mutex_); (2) an empty-at-fork window — a child the app
    // forks inherits this populated string and reconnects to the parent's daemon,
    // instead of recomputing rpc_invocation_runtime_dir(child_pid) and missing it.
    //
    // Prefer the dir the launcher exported before execvp: every descendant
    // (including grandchildren spawned through wrappers like ctest, whose PID
    // differs from the launcher's) inherits the exact directory holding
    // config_path/daemon.sock. Fall back to this process's PID-scoped default for
    // attach mode, where no launcher set the variable.
    // Assigned before resolve() (which flips real().ready() true, the gate every
    // interposed entry point checks) so no reader can observe an empty value.
    // Treat an unset OR empty $ROCJITSU_INVOCATION_DIR as "no launcher dir": an
    // empty value would otherwise leave invocation_runtime_dir_ empty, so
    // connect_to_daemon() would target "/daemon.sock" and access("") would probe
    // the CWD — either mis-gating the fallback or connecting to an unintended
    // socket. Fall back to this process's PID-scoped default in that case.
    const char *dir = getenv(rocjitsu::kRpcInvocationDirEnv);
    if (dir && *dir)
      ctx.invocation_runtime_dir_ = dir;
    else
      ctx.invocation_runtime_dir_ = rocjitsu::rpc_invocation_runtime_dir(getpid());
    // Reset child state on ANY glibc fork-family primitive, not just the
    // interposed fork() symbol. system()/popen()/posix_spawn() and libraries that
    // call fork() through a path that doesn't bind to our exported fork() would
    // otherwise leave the child with mutexes locked-by-a-dead-thread and a live
    // remote_ aliasing the parent's daemon connection — the next interposed
    // open()/ioctl()/close() in that child would then deadlock or corrupt the
    // parent's connection. pthread_atfork's child handler runs inside libc fork,
    // covering every fork that goes through glibc. (vfork/posix_spawn children run
    // no atfork handlers by design, but they may only exec/_exit, so there is no
    // interposer state for them to corrupt.) reset_after_fork() is idempotent. It is
    // NOT strictly async-signal-safe — container clear()/destructors call free() and it
    // closes the child's dmabuf-dup fds — so it relies on the standard fork-then-exec /
    // single-threaded-fork assumption (the same one the remote_ handling documents
    // below); a multithreaded fork from a signal handler is out of scope.
    pthread_atfork(nullptr, nullptr, &InterposerContext::atfork_child);
    real().resolve();
  }

  /// @brief pthread_atfork child handler: reset interposer state in the child.
  static void atfork_child() { ctx.reset_after_fork(); }

  /// @brief Reset interposer state in a forked child process.
  /// @details After fork(), the child inherits the parent's address space but
  /// the engine thread is dead (only the calling thread survives). We DISCARD all
  /// inherited driver/connection/bookkeeping state so the next open("/dev/kfd")
  /// creates a fresh connection. The mutexes are deliberately left as-is (see the
  /// body): under the supported single-threaded-fork model they are provably
  /// unlocked, and reconstructing them would be undefined behavior.
  void reset_after_fork() {
    active_driver_.store(nullptr, std::memory_order_release);
    // fork() copies the parent std::thread object (so it looks joinable) but does
    // NOT reproduce the engine thread in the child. Release the owning handle
    // WITHOUT join()/detach(): both would operate on a thread that does not exist
    // here (join deadlocks, detach is UB on a non-existent thread). The abandoned
    // heap allocation belongs to the inherited parent VM the child also drops just
    // below; the child never runs a local VM until a fresh open() rebuilds one.
    // Intentionally leak the unique_ptr's std::thread object: it names a thread
    // that does not exist in this child, so it must be neither joined nor detached,
    // and the tiny allocation dies with the abandoned parent VM state.
    static_cast<void>(local_vm_thread_.release());
    rj_vm_ = nullptr;
    shutdown_requested_ = false;
    if (guest_driver_)
      guest_driver_->reset_after_fork();
    guest_driver_.reset();
    // Drop the child's reference to the inherited RemoteDriver. Storing nullptr
    // releases this shared_ptr; if it was the last reference in the child, the
    // child's ~RemoteDriver runs and closes only the child's inherited (dup'd)
    // fds — it does not send RPC_CLOSE and cannot invalidate the parent's live
    // connection, so this is safe in a forked child.
    //
    // remote_ is released with a plain store(nullptr): ending a live
    // std::atomic<std::shared_ptr<>>'s lifetime without running its destructor
    // would bypass the control-block bookkeeping. store(nullptr) drops the child's
    // reference correctly (the child's ~RemoteDriver closes only the child's dup'd
    // fds; it sends no RPC_CLOSE and cannot disturb the parent's connection).
    //
    // Caveat: std::atomic<std::shared_ptr<>> may serialize on a libstdc++-internal
    // pool spinlock. If a parent thread was mid remote_.load()/store() at fork(),
    // the child inherits that lock held-by-a-dead-thread and this store() could
    // block. That is the same single-threaded-fork invariant relied on throughout
    // this handler (below): a genuinely multithreaded fork is out of scope.
    remote_.store(nullptr, std::memory_order_relaxed);
    remote_kfd_fd_.store(-1, std::memory_order_relaxed);
    remote_open_refs_.store(0, std::memory_order_relaxed);
    // The interposer mutexes (init_mutex_/fd_mutex_/remote_mutex_) and the
    // driver_lifetime_ latch are intentionally left UNTOUCHED here — not
    // reconstructed. Under the supported fork-then-exec / single-threaded-fork
    // model (see init() above), the only thread that survives into the child is the
    // one that called fork(); it was executing fork(), not inside any interposer
    // critical section (no lock is held across a fork-capable call such as
    // system()/the engine join), so every interposer lock is provably UNLOCKED and
    // is inherited ready to use. Placement-new over a live std::mutex /
    // std::shared_mutex would end the object's lifetime WITHOUT running its
    // destructor — undefined behavior ([basic.life]) — and would not repair the
    // data a held lock protected anyway, so it fixes nothing the invariant does not
    // already guarantee. A genuinely multithreaded fork is out of scope: no
    // reinitialization here could make the inherited bookkeeping
    // (kfd_dup_fds_/gem_entries_/remote_) self-consistent, which is exactly why
    // POSIX deems "reinitialize the mutex in the child" unsatisfactory.
    sysfs_fds_.clear();
    drm_fds_.clear();
    kfd_dup_fds_.clear();
    // Drop GEM bookkeeping WITHOUT munmapping cpu_ptr or unmapping PTEs: those host
    // pages and the parent's page table belong to the parent process and the child
    // must not touch them. The child re-initializes a fresh driver; any GEM mappings
    // it needs are re-created via EXPORT/PRIME/GEM_VA. (Deliberate, like the remote_
    // and mutex handling above — not an oversight.)
    //
    // DO close each entry's private dmabuf dup though: it was created with
    // F_DUPFD_CLOEXEC (closes on exec, NOT on fork), so a fork-without-exec child
    // inherits these descriptors and clearing the map without closing them leaks a
    // child-local fd per live GEM handle. The fd is the child's own copy — closing it
    // touches no parent state. (real().close() is not async-signal-safe, but this runs
    // under the same fork-then-exec / single-threaded-fork assumption as the remote_
    // handling above.)
    for (auto &[handle, gem] : gem_entries_)
      if (gem.owns_dmabuf_fd && gem.dmabuf_fd >= 0)
        real().close(gem.dmabuf_fd);
    gem_entries_.clear();
    // Also drop the transient EXPORT_DMABUF fd->flags handoffs: they key on the
    // parent's dmabuf fd numbers, so a child that reuses one of those fd numbers
    // before a fresh EXPORT could otherwise fold a stale parent MTYPE hint into its
    // own PRIME import.
    pending_gem_flags_.clear();
    in_construction = false;
  }

  LinuxKfd *driver() { return active_driver_.load(std::memory_order_acquire); }

  /// @brief Pin the active driver object's lifetime for the duration of a call.
  /// @details Returns a shared lock on driver_lifetime_. Hold it across any
  /// sequence that dereferences driver() so a concurrent teardown cannot free the
  /// owning VM/GuestKfd mid-call. Acquire ONCE at a top-level libc interposer
  /// entry point (the lock is non-reentrant); internal helpers it calls must not
  /// re-acquire it. See driver_lifetime_.
  [[nodiscard]] std::shared_lock<std::shared_mutex> pin_driver_lifetime() {
    return std::shared_lock<std::shared_mutex>(driver_lifetime_);
  }
  /// @brief True if the active driver is the local SimulatedKfd (not remote/guest).
  bool driver_is_simulated() { return dynamic_cast<SimulatedKfd *>(driver()) != nullptr; }
  /// @brief The active local driver's primary fd, or -1 if none.
  /// @details Dereferences the driver, so callers must hold a driver_lifetime_ pin
  /// (or init_mutex_) — the internal classification helpers is_kfd_primary()/
  /// kfd_backend_of() that call this run only under a caller-held pin.
  int driver_fd() {
    auto *d = driver();
    return d ? d->fd() : -1;
  }
  bool initialized() const {
    return active_driver_.load(std::memory_order_acquire) != nullptr ||
           remote_.load(std::memory_order_acquire) != nullptr;
  }

  /// @brief Record a process-exit shutdown request from the DSO finalizer.
  /// @details As an LD_PRELOAD library, librocjitsu.so can be finalized before HIP
  /// and ROCR. If they still hold app-facing KFD descriptors, this request stays
  /// pending and the VM is kept alive; their later close paths complete it. If the
  /// simulator is already idle, this tears the VM down now.
  void request_local_vm_shutdown() {
    std::lock_guard lock(init_mutex_);
    shutdown_requested_ = true;
    shutdown_local_vm_if_idle_locked();
  }

  /// @brief Release one app-facing local KFD reference (the close/dup teardown path).
  /// @details Drops the driver's counted reference and, if a process-exit shutdown
  /// is pending and the simulator is now idle, tears the VM down — all under
  /// init_mutex_. The close() and the destroy MUST be in one critical section: the
  /// active driver is a raw pointer owned by state a concurrent
  /// request_local_vm_shutdown()/release_local_open() can free, so dropping the
  /// lock between close() and the idle check would let another thread destroy the
  /// VM and leave this thread calling into freed memory. init_mutex_ is only ever
  /// taken by the construction/shutdown paths (never by the engine or ioctl paths),
  /// so holding it across the driver close() introduces no lock-order inversion.
  void release_local_open() {
    std::lock_guard lock(init_mutex_);
    if (auto *active_driver = driver())
      active_driver->close();
    if (shutdown_requested_)
      shutdown_local_vm_if_idle_locked();
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

  /// @brief The per-invocation runtime directory for this process image.
  /// @details Populated once in init() before any thread or app fork, so this is
  /// a lock-free immutable read. A forked app child inherits the parent's value
  /// (reset_after_fork() intentionally does not clear it) and thus reconnects to
  /// the same daemon rather than recomputing a dir under its own PID.
  const std::string &invocation_runtime_dir() const { return invocation_runtime_dir_; }

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
  /// was added (i.e. a remote is active). Mirrors the local path, where
  /// reserve_dup_backend() holds the shared driver_lifetime_ pin across the
  /// classify+retain so a concurrent teardown cannot free the driver mid-retain.
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
    int sock = connect_to_daemon(invocation_runtime_dir());
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

    // Pin the local driver's lifetime across redirect() (which derefs driver()).
    // Acquired AFTER get_or_create() released init_mutex_, honoring the lock order
    // init_mutex_ -> driver_lifetime_ (never shared-pin -> init_mutex_).
    auto driver_lifetime = pin_driver_lifetime();
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
  /// @details The Local branch dereferences the driver; the caller
  /// (reserve_dup_backend) holds the driver_lifetime_ pin across this call, so a
  /// concurrent teardown cannot free the driver between the load and
  /// retain_local_open(). retain_local_open() reports false if the local process
  /// was already torn down (a racing last-close). Do NOT pin here: std::shared_mutex
  /// is non-recursive and the caller already holds the pin.
  bool retain_backend(DupBackend backend) {
    if (backend == DupBackend::Local) {
      auto *d = driver();
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
      release_local_open();
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
    // Pin the driver's lifetime across the whole classify+retain: kfd_backend_of()
    // -> driver_fd() and retain_backend()'s Local branch both dereference the local
    // driver, so a concurrent teardown must not free it mid-call. A single pin
    // covers both (shared_mutex is non-recursive, so retain_backend must not
    // re-pin). Takes no init_mutex_, so no nesting/inversion.
    auto driver_lifetime = pin_driver_lifetime();
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
    // is_kfd_primary() -> driver_fd() dereferences the local driver, so pin its
    // lifetime for the check and DROP the pin before release_backend() (which can
    // reach init_mutex_ via release_local_open()): never shared-pin -> init_mutex_.
    bool aliases_primary = false;
    if (fd < 0) {
      aliases_primary = true;
    } else {
      auto driver_lifetime = pin_driver_lifetime();
      aliases_primary = is_kfd_primary(fd);
    }
    if (aliases_primary) {
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
    // Pin the driver's lifetime only across the invalidate_primary_fd() deref,
    // then DROP the pin before release_local_open() (which takes init_mutex_):
    // holding driver_lifetime_ across an init_mutex_ acquisition would invert the
    // teardown order (init_mutex_ -> driver_lifetime_) and could deadlock.
    LinuxKfd::PrimaryInvalidation invalidation = LinuxKfd::PrimaryInvalidation::kNotPrimary;
    {
      auto driver_lifetime = pin_driver_lifetime();
      if (auto *d = driver())
        invalidation = d->invalidate_primary_fd(fd);
    }
    switch (invalidation) {
    case LinuxKfd::PrimaryInvalidation::kClearedDropRef:
      release_local_open(); // drop the primary open reference
      return;
    case LinuxKfd::PrimaryInvalidation::kClearedKeepRefs:
      return; // classification cleared; no counted reference to drop
    case LinuxKfd::PrimaryInvalidation::kNotPrimary:
      break; // fall through to dup handling
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

    std::string config_json = read_file_passthrough(config_path.c_str());
    if (rj_vm_load_plugins(rj_vm_, config_json.c_str(), nullptr) != ROCJITSU_STATUS_SUCCESS) {
      util::Logger::debug_print("rocjitsu: failed to configure execution plugins");
      rj_vm_destroy(rj_vm_);
      rj_vm_ = nullptr;
      return false;
    }
    return true;
  }

  /// @brief Publish a fully-ready local driver. Caller holds init_mutex_.
  /// @details The sole site that stores a non-null active_driver_. Done under the
  /// exclusive lifetime latch for symmetry with teardown and so the release-store
  /// is ordered against readers' shared-pinned acquire loads. Only call after the
  /// engine has started successfully (see get_or_create) so no reader can observe a
  /// half-started device.
  void publish_local_driver_locked(LinuxKfd *driver) {
    std::unique_lock<std::shared_mutex> lifetime(driver_lifetime_);
    active_driver_.store(driver, std::memory_order_release);
  }

  /// @brief Centralized local-backend teardown. Caller holds init_mutex_.
  /// @details The ONE path that unpublishes the driver, stops the engine, and frees
  /// the local VM AND/OR the guest driver. Handles a simulator VM (rj_vm_ set), a
  /// hardware-backed guest (guest_driver_ set, rj_vm_ null), or the simulator-backed
  /// guest (both set). Ordering follows the lifetime plan so no latch is held across
  /// the engine join and no fast-path deref can touch freed state:
  ///   1. Unpublish under the EXCLUSIVE lifetime latch (store active_driver_=null),
  ///      draining every in-flight shared pin. After this, no new deref can observe
  ///      the driver (readers re-load driver() under their pin and see null).
  ///   2. Release the latch and join the engine thread. Not holding the latch (or,
  ///      by the lock hierarchy, taking init_mutex_ from the engine) across the join
  ///      avoids any deadlock if engine shutdown ever re-enters an interposed path.
  ///   3. Free the owners. Safe without the latch now: the driver is unpublished, so
  ///      no shared pin can still reach it, and construction (get_or_create) is
  ///      serialized by init_mutex_ which the caller holds. guest_driver_ is reset
  ///      before rj_vm_destroy() because a simulator-backed guest owns the VM's
  ///      bootstrap open.
  void destroy_local_vm() {
    // Nothing published and no owner to free (also the guest-only reset() re-entry).
    if (!rj_vm_ && !guest_driver_)
      return;
    {
      std::unique_lock<std::shared_mutex> lifetime(driver_lifetime_);
      active_driver_.store(nullptr, std::memory_order_release);
    }
    // request_exit() unblocks run()'s idle wait so the join is prompt. Done with no
    // lifetime latch held (see step 2 above). Only a simulator VM has an engine
    // thread; a hardware-backed guest has none.
    if (local_vm_thread_) {
      rj_vm_request_exit(rj_vm_, "interposer VM destruction");
      local_vm_thread_->join();
      local_vm_thread_.reset();
    }
    guest_driver_.reset();
    if (rj_vm_) {
      rj_vm_destroy(rj_vm_);
      rj_vm_ = nullptr;
    }
  }

  /// @brief Launch the local VM engine on an owned (joinable) thread and wait for
  /// component startup to complete before returning.
  /// @returns false if the engine thread could not be created OR a component's
  /// startup() failed (caller unwinds).
  /// @details The thread is owned (not detached) so process teardown can join it
  /// before destroying VM/driver state it reaches. wait_until_started() closes the
  /// window where a short-lived caller could publish and then tear down a device
  /// whose components are still in startup() — the race that faulted the engine in
  /// CommandProcessor::startup() against C++ finalizers. If startup() threw, the
  /// engine already returned an error and stopped, so join the thread and report
  /// failure rather than leaving a published-but-dead device.
  [[nodiscard]] bool start_local_vm() {
    assert(rj_vm_ != nullptr);
    assert(!local_vm_thread_);
    try {
      // make_unique/std::thread can throw system_error (resource exhaustion) or
      // bad_alloc; treat any failure to launch as a start failure so the caller
      // unwinds and never publishes a device with no engine behind it.
      local_vm_thread_ = std::make_unique<std::thread>([vm = rj_vm_]() { rj_vm_run(vm, nullptr); });
    } catch (const std::exception &e) {
      util::Logger::debug_print("rocjitsu: failed to start local VM engine thread: ", e.what());
      return false;
    }
    if (!rj_vm_->engine->wait_until_started()) {
      // A component's startup() threw; the engine thread already returned. Join it
      // so the caller can safely destroy the VM on the unwind path.
      local_vm_thread_->join();
      local_vm_thread_.reset();
      util::Logger::debug_print("rocjitsu: local VM engine failed during component startup");
      return false;
    }
    return true;
  }

  /// @brief A GEM buffer object synthesized from a prime (dmabuf) fd.
  /// @details The native DRM emulation has no real GEM objects. EXPORT_DMABUF
  /// hands userspace a dmabuf fd whose KFD allocation flags determine the GPU PTE
  /// MTYPE; PRIME_FD_TO_HANDLE then mints a STABLE, monotonically-increasing GEM
  /// handle (never derived from the fd number). The entry is keyed by that handle,
  /// which owns the mapping's lifetime: it carries the flags from export through to
  /// GEM_VA (where it lazily mmaps the backing pages used to install the GPU page
  /// table) and lives until DRM_IOCTL_GEM_CLOSE (or until its owning DRM file
  /// closes). It deliberately does NOT die when the transient dmabuf export fd is
  /// closed — ROCr closes that fd immediately after GEM_VA returns, while the GPU
  /// mapping must stay live for the caller. Because handles are not fd-derived, a
  /// recycled dmabuf fd number can never resolve to a still-live handle and tear
  /// down an unrelated BO. `dmabuf_fd` is a PRIVATE dup taken at PRIME time and held
  /// only for the lazy backing mmap, so the backing stays valid even if the caller
  /// closes the export fd before GEM_VA (the fd number cannot be recycled out from
  /// under us); `drm_fd` scopes the handle to its DRM file so a file close reaps any
  /// handles the caller never GEM_CLOSE'd.
  /// installed_vas holds the GPU VA ranges mapped from this BO so teardown can
  /// remove the page-table entries before munmapping cpu_ptr.
  ///
  /// `owner` records the SimulatedKfd whose page table actually holds these PTEs,
  /// captured at map time. The local driver may be absent when GEM_CLOSE arrives
  /// (e.g. a remote/DBT-guest backend is active, or — in a forked child — before the
  /// driver is recreated), so teardown unmaps through `owner` only while it is still
  /// the active driver, and skips the unmap otherwise (that page table is gone).
  /// The local driver is a process-lifetime singleton (see get_or_create), so
  /// `owner` never points at a freed-and-replaced driver.
  struct GemMapping {
    uint64_t va_address = 0;
    uint64_t map_size = 0;
    bool operator==(const GemMapping &) const = default;
    /// @brief True if this VA interval intersects @p other. Half-open [va, va+size).
    /// map_size is already bounds-checked non-zero and non-overflowing in gem_map().
    [[nodiscard]] bool overlaps(const GemMapping &other) const {
      return va_address < other.va_address + other.map_size &&
             other.va_address < va_address + map_size;
    }
  };

  struct GemEntry {
    int dmabuf_fd = -1;          ///< Private dup of the backing dmabuf fd for the lazy mmap.
    bool owns_dmabuf_fd = false; ///< True if dmabuf_fd is our dup (close at teardown).
    int drm_fd = -1;             ///< Owning DRM file; the entry is reaped when it closes.
    uint64_t size = 0;
    uint32_t alloc_flags = 0;
    void *cpu_ptr = nullptr;
    SimulatedKfd *owner = nullptr;
    std::vector<GemMapping> installed_vas;
  };

  /// @brief Record KFD alloc flags for an exported dmabuf fd (at EXPORT_DMABUF).
  /// @details The flags determine the GPU PTE MTYPE when the fd is later mapped via
  /// GEM_VA and must be captured at export time because the underlying allocation
  /// may be freed before the map. This is only a TRANSIENT fd→flags association,
  /// consumed by the next PRIME_FD_TO_HANDLE on the same fd (which folds the flags
  /// into a stable-handle GemEntry). To keep the fd key from going stale — a dmabuf
  /// fd closed without a PRIME, then recycled by the kernel for an unrelated file —
  /// drop_pending_gem_flags(fd) clears the record at close(fd), so a reused fd
  /// number can never inherit a previous export's MTYPE.
  void track_gem_flags(int dmabuf_fd, uint32_t alloc_flags) {
    std::lock_guard lock(fd_mutex_);
    pending_gem_flags_[dmabuf_fd] = alloc_flags;
  }

  /// @brief Drop any transient EXPORT_DMABUF flags recorded for @p fd (at close(fd)).
  /// @details Called from the close() hook for every fd. Cheap no-op when @p fd is
  /// not a pending dmabuf export. Prevents a closed-without-PRIME export fd from
  /// leaving a stale flag that a later PRIME on the recycled fd number would apply.
  void drop_pending_gem_flags(int fd) {
    std::lock_guard lock(fd_mutex_);
    pending_gem_flags_.erase(fd);
  }

  /// @brief Mint a stable GEM handle for a prime-imported dmabuf (PRIME_FD_TO_HANDLE).
  /// @details Allocates a fresh monotonically-increasing handle (never fd-derived),
  /// consumes the transient EXPORT_DMABUF flags for @p dmabuf_fd (defaulting to 0 if
  /// PRIME arrives without a preceding EXPORT), records @p size and the owning DRM
  /// file, and returns the handle. Handle 0 is never minted, so callers may treat 0
  /// as "no handle".
  /// @returns The stable GEM handle (>= 1).
  uint32_t prime_import(int dmabuf_fd, int drm_fd, uint64_t size) {
    std::lock_guard lock(fd_mutex_);
    uint32_t alloc_flags = 0;
    if (auto it = pending_gem_flags_.find(dmabuf_fd); it != pending_gem_flags_.end()) {
      alloc_flags = it->second;
      pending_gem_flags_.erase(it);
    }
    // Pin the backing to the HANDLE's lifetime by dup'ing the dmabuf fd now, rather
    // than storing the caller's fd number for a later lazy mmap. ROCr closes the
    // export fd right after GEM_VA returns, but nothing in the DRM ABI forbids a
    // client from closing it between PRIME and a deferred GEM_VA MAP; the fd number
    // could then be recycled and the lazy mmap in gem_map() would map an unrelated
    // file. The dup keeps the same dmabuf open under a private fd until the handle is
    // torn down. Falls back to the raw fd if dup fails (best effort; the common
    // fd-still-open case is unaffected).
    int backing_fd = InterposerContext::real().fcntl(dmabuf_fd, F_DUPFD_CLOEXEC, 0);
    // Mint the next free handle. Skip 0 ("no handle") and any handle still live, so a
    // uint32 wrap after a very long-lived process cannot silently overwrite an
    // in-use entry (which would detach its PTEs from a future GEM_CLOSE).
    uint32_t handle = next_gem_handle_++;
    while (handle == 0 || gem_entries_.count(handle) != 0)
      handle = next_gem_handle_++;
    GemEntry &gem = gem_entries_[handle];
    gem = {};
    gem.dmabuf_fd = (backing_fd >= 0) ? backing_fd : dmabuf_fd;
    gem.owns_dmabuf_fd = (backing_fd >= 0);
    gem.drm_fd = drm_fd;
    gem.size = size;
    gem.alloc_flags = alloc_flags;
    return handle;
  }

  /// @brief Install (or replace) a GEM_VA range in the GPU page table for @p handle.
  /// @details Runs entirely under fd_mutex_ and performs BOTH the bookkeeping AND
  /// the page-table install (drv->gem_va_map) atomically, so a concurrent GEM_CLOSE
  /// (untrack_gem, also under fd_mutex_) can never interleave between recording the
  /// range and installing the PTEs — which would otherwise leave PTEs pointing into
  /// a munmapped cpu_ptr with no entry left to tear them down. It lazily mmaps the
  /// dmabuf fd's backing pages the first time (the fd is still open at GEM_VA time),
  /// bounds-checks the request against the BO size, records the range, and installs
  /// the PTEs. The lock order fd_mutex_ -> driver page-table lock matches
  /// teardown_gem_entry_locked, so there is no inversion.
  /// @param replace When true (AMDGPU_VA_OP_REPLACE), first evict any existing range
  ///   that OVERLAPS {va_address, map_size} — from whatever handle currently owns it —
  ///   so the old owner's bookkeeping does not later tear down the replacement's PTEs.
  ///   When false (AMDGPU_VA_OP_MAP), an overlapping pre-existing range is a conflict.
  /// @retval true the range was installed.
  /// @retval false unknown handle, out-of-bounds request, failed mmap, no driver, or
  ///   (MAP only) the range is already mapped.
  [[nodiscard]] bool gem_map(uint32_t handle, uint64_t va_address, uint64_t offset_in_bo,
                             uint64_t map_size, bool replace) {
    std::lock_guard lock(fd_mutex_);
    auto *drv = dynamic_cast<SimulatedKfd *>(driver());
    if (!drv)
      return false;
    auto it = gem_entries_.find(handle);
    if (it == gem_entries_.end())
      return false;
    GemEntry &gem = it->second;
    // Bound the request within the BO without letting offset_in_bo + map_size
    // overflow (both are caller-controlled __u64 from the UAPI struct): a wrap
    // would defeat a naive sum-vs-size check and install PTEs pointing outside
    // the mmap. Reject a zero-size BO, a zero-size map, and any range past the end.
    if (gem.size == 0 || map_size == 0 || offset_in_bo > gem.size ||
        map_size > gem.size - offset_in_bo)
      return false;
    const GemMapping range{va_address, map_size};
    // Handle the target VA range's current occupant. REPLACE evicts any current
    // holder (possibly a different handle) so its records cannot later unmap the new
    // PTEs. Plain MAP treats an existing range as a conflict rather than silently
    // double-mapping over another handle's PTEs.
    if (replace) {
      if (!evict_range_locked(drv, range, /*allow_missing=*/true))
        return false;
    } else if (range_is_mapped_locked(range)) {
      return false;
    }
    if (!gem.cpu_ptr) {
      void *p = InterposerContext::real().mmap(nullptr, gem.size, PROT_READ | PROT_WRITE,
                                               MAP_SHARED, gem.dmabuf_fd, 0);
      if (p == MAP_FAILED)
        return false;
      gem.cpu_ptr = p;
    }
    // Record which SimulatedKfd's page table receives these PTEs so GEM_CLOSE (or a
    // DRM-file-close reap) unmaps through the driver that installed them, never a
    // replacement one. Set the owner on the first mapping and keep it: all ranges of
    // one BO install into the same driver's page table. The local driver is a
    // process-lifetime singleton (created once, never destroyed except in the fork
    // child, which also clears gem_entries_), so the owner never dangles and every
    // subsequent map observes the same driver.
    if (gem.installed_vas.empty())
      gem.owner = drv;
    else
      assert(gem.owner == drv && "GEM ranges of one BO must share one owning driver");
    void *host = static_cast<uint8_t *>(gem.cpu_ptr) + offset_in_bo;
    // Install the PTEs while still holding fd_mutex_ so the range record and the
    // page-table state stay consistent against a concurrent teardown. gem_va_map
    // only returns false if the local process vanished mid-call; treat that as a
    // failed map (do not record the range) so GEM_VA reports the error rather than a
    // phantom success.
    if (!drv->gem_va_map(va_address, host, map_size, gem.alloc_flags))
      return false;
    gem.installed_vas.push_back(range);
    return true;
  }

  /// @brief Remove a GEM_VA range from the GPU page table for @p handle (UNMAP).
  /// @details Performs the page-table unmap AND the bookkeeping erase atomically
  /// under fd_mutex_ (same rationale as gem_map). Validates that @p handle actually
  /// owns the exact {va_address, map_size} range before mutating the page table, so
  /// an UNMAP with a wrong handle or range cannot tear down PTEs the handle does not
  /// own and cannot report success for a no-op.
  /// @retval true the range was owned by @p handle and has been unmapped.
  /// @retval false no driver, unknown handle, or @p handle does not own that range.
  [[nodiscard]] bool gem_unmap(uint32_t handle, uint64_t va_address, uint64_t map_size) {
    std::lock_guard lock(fd_mutex_);
    auto *drv = dynamic_cast<SimulatedKfd *>(driver());
    if (!drv)
      return false;
    auto it = gem_entries_.find(handle);
    if (it == gem_entries_.end())
      return false;
    GemEntry &gem = it->second;
    const GemMapping range{va_address, map_size};
    if (std::find(gem.installed_vas.begin(), gem.installed_vas.end(), range) ==
        gem.installed_vas.end())
      return false; // This handle does not own the exact range — do not touch PTEs.
    if (!drv->gem_va_unmap(va_address, map_size))
      return false;
    std::erase(gem.installed_vas, range);
    return true;
  }

  /// @brief Clear a GEM_VA range from the GPU page table (CLEAR).
  /// @details CLEAR is handle-agnostic: it tears down every recorded range that
  /// OVERLAPS {va_address, map_size} wherever it is currently recorded, updating the
  /// owning entries' bookkeeping so a later GEM_CLOSE does not double-unmap them.
  /// Fails if no recorded range overlaps (so the ioctl reports EINVAL instead of a
  /// phantom clear).
  /// @retval true at least one overlapping range was found and cleared.
  /// @retval false no driver, or no recorded range overlaps.
  [[nodiscard]] bool gem_clear(uint64_t va_address, uint64_t map_size) {
    std::lock_guard lock(fd_mutex_);
    auto *drv = dynamic_cast<SimulatedKfd *>(driver());
    if (!drv)
      return false;
    return evict_range_locked(drv, GemMapping{va_address, map_size}, /*allow_missing=*/false);
  }

  /// @brief Reap every GEM handle owned by a closing DRM file (at its close()).
  /// @details A well-behaved caller GEM_CLOSEs each handle, but a crash or leak can
  /// leave handles live; the DRM file close is their backstop, mirroring the kernel
  /// dropping a drm_file's GEM objects. Tears each entry's PTEs + host mmap down
  /// under fd_mutex_ before erasing, so no state escapes the lock.
  void reap_gem_for_drm_fd(int drm_fd) {
    // teardown_gem_entry_locked() derefs the active driver; pin its lifetime
    // (order driver_lifetime_ -> fd_mutex_, consistent with the other GEM paths).
    auto driver_lifetime = pin_driver_lifetime();
    std::lock_guard lock(fd_mutex_);
    for (auto it = gem_entries_.begin(); it != gem_entries_.end();) {
      if (it->second.drm_fd == drm_fd) {
        teardown_gem_entry_locked(it->second);
        it = gem_entries_.erase(it);
      } else {
        ++it;
      }
    }
  }

  /// @brief Drop the GEM entry for a closing GEM handle (at DRM_IOCTL_GEM_CLOSE).
  /// @details The handle owns the mapping lifetime, so this is the point where the
  /// BO is truly gone. Tears down the page-table entries (through the owning driver)
  /// and munmaps the host mapping entirely under fd_mutex_, so no GemEntry pointer
  /// or driver pointer escapes the lock and a concurrent GEM_VA cannot race the
  /// teardown.
  void untrack_gem(uint32_t handle) {
    // teardown_gem_entry_locked() derefs the active driver; pin its lifetime
    // (order driver_lifetime_ -> fd_mutex_, consistent with the other GEM paths).
    auto driver_lifetime = pin_driver_lifetime();
    std::lock_guard lock(fd_mutex_);
    auto it = gem_entries_.find(handle);
    if (it == gem_entries_.end())
      return;
    teardown_gem_entry_locked(it->second);
    gem_entries_.erase(it);
  }

  LinuxKfd *get_or_create() {
    std::lock_guard lock(init_mutex_);
    if (active_driver_.load(std::memory_order_acquire) == nullptr) {
      // Fail closed once process-exit shutdown has been requested: the local VM is
      // a one-shot per process. Recreating it after the finalizer ran would start
      // an engine no later close would ever retire (the DSO finalizer fires once),
      // reintroducing the live-engine-at-teardown condition this change removes —
      // and would leave stale generation-bound state (e.g. GemEntry::owner) from
      // the destroyed VM. A late sysfs/DRM probe therefore gets no local driver.
      if (shutdown_requested_) {
        util::Logger::debug_print("rocjitsu: local VM creation refused after shutdown request");
        return nullptr;
      }
      in_construction = true;
      // Config-path discovery mirrors load_dbt_guest_config_from_runtime_config()'s
      // reader precedence exactly, probing tiers in order and using the first whose
      // config_path handoff actually exists:
      //   1. the per-invocation directory (the launcher writes config_path there and
      //      exports $ROCJITSU_INVOCATION_DIR); invocation_runtime_dir() already
      //      collapses to the per-PID default when that env var is unset.
      //   2. this process's PID-scoped default — reached only when the env var is set
      //      but its config_path is absent/stale, matching the reader's tier 2 (a
      //      no-op duplicate of tier 1 when the env var is unset).
      //   3. the well-known $ROCJITSU_RUNTIME_DIR/config_path for a bare LD_PRELOAD
      //      client that sets no invocation dir. Without this fallback such a client's
      //      config is never found and hsa_init fails with OUT_OF_RESOURCES.
      // Probing tier 2 as well as tier 1 keeps the interposer's view consistent with
      // the reader's: an env var pointing at a dir without config_path must still find
      // a valid per-PID handoff instead of skipping straight to the well-known path.
      std::vector<std::string> cfg_candidates;
      cfg_candidates.push_back(invocation_runtime_dir() + "/config_path");
      cfg_candidates.push_back(rocjitsu::rpc_invocation_config_file_path(getpid()));
      cfg_candidates.push_back(rocjitsu::rpc_default_config_file_path());
      std::optional<std::string> cfg_path;
      std::string tried_last;
      for (const auto &candidate : cfg_candidates) {
        if (candidate == tried_last)
          continue; // Skip a duplicate tier (e.g. env unset collapses 1 and 2).
        tried_last = candidate;
        cfg_path = child_config_path(candidate);
        if (cfg_path)
          break;
      }
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
          // Start the engine and wait for readiness BEFORE publishing, so no
          // concurrent interposed call can observe a driver whose components are
          // still starting (or whose startup will fail). On failure nothing was
          // published; destroy_local_vm() unpublishes (no-op) + frees, and
          // guest_driver_ is reset first because it owns the simulator bootstrap
          // open.
          if (simulator_backend && !start_local_vm()) {
            guest_driver_.reset();
            destroy_local_vm();
            in_construction = false;
            return nullptr;
          }
          // Publish the fully-ready driver under the exclusive lifetime latch.
          publish_local_driver_locked(driver);
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

      // Start the engine and wait for component startup to SUCCEED before
      // publishing active_driver_. Publishing earlier would let a concurrent
      // interposed call observe a half-started (or about-to-fail) device — the
      // race this branch removes. On failure nothing was published; destroy_local_vm
      // frees the VM.
      LinuxKfd *driver = rj_vm_->vm->driver();
      if (!start_local_vm()) {
        destroy_local_vm();
        in_construction = false;
        return nullptr;
      }
      // Publish the fully-ready driver under the exclusive lifetime latch. The
      // release-store pairs with acquire loads in driver()/initialized() so any
      // reader that observes the driver also observes all setup above.
      publish_local_driver_locked(driver);
      in_construction = false;
    }
    return driver();
  }

  /// @brief Read an entire file via the libc passthrough (real openat/read/
  /// close), retrying on EINTR.
  /// @details Used during driver construction where re-entering our interposed
  /// I/O path could recurse. Read errors (other than EINTR) are reported and
  /// yield an empty string, distinct from a successfully read empty file.
  static std::string read_file_passthrough(const char *path) {
    if (!real().ready())
      return {};
    int fd;
    do {
      fd = real().openat(AT_FDCWD, path, O_RDONLY, 0);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0)
      return {};
    std::string out;
    char buf[4096] = {};
    for (;;) {
      ssize_t n = real().read(fd, buf, sizeof(buf));
      if (n < 0) {
        if (errno == EINTR)
          continue;
        util::Logger::warn("rocjitsu: read error on '", path, "': ", std::strerror(errno));
        real().close(fd);
        return {};
      }
      if (n == 0)
        break; // EOF.
      out.append(buf, static_cast<size_t>(n));
    }
    real().close(fd);
    return out;
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
  /// @brief True if the published local driver holds no app-facing KFD reference.
  /// @details Caller holds init_mutex_ (and may additionally hold the exclusive
  /// lifetime latch). SimulatedKfd carries one internal bootstrap open on top of
  /// app-facing opens, so a refcount of one is idle; GuestKfd counts only
  /// app-facing references, so zero is idle. A null or non-ref-tracking driver is
  /// treated as idle.
  bool local_driver_is_idle_locked() {
    LinuxKfd *active_driver = active_driver_.load(std::memory_order_acquire);
    if (auto *simulated_driver = dynamic_cast<SimulatedKfd *>(active_driver))
      return simulated_driver->local_open_ref_count() <= 1;
    if (auto *guest = dynamic_cast<GuestKfd *>(active_driver))
      return guest->local_open_ref_count() == 0;
    return true;
  }

  /// @brief Stop and destroy the local backend iff no app-facing KFD reference
  /// remains. Handles both a simulator VM and a hardware-backed guest.
  /// @details Caller holds init_mutex_. Runs in four phases so that a blocking
  /// driver call can never deadlock teardown and a racing retain can never resurrect
  /// a torn-down backend:
  ///
  ///   1. Preliminary idle check under init_mutex_ ONLY (no lifetime latch). If an
  ///      app-facing reference remains, return WITHOUT taking the exclusive latch.
  ///      This is what keeps the DSO finalizer from deadlocking: while references
  ///      are held a concurrent AMDKFD_IOC_WAIT_EVENTS holds a shared pin
  ///      indefinitely, so taking the exclusive latch here would block the finalizer
  ///      forever — and the later HIP/ROCR finalizers that would close those
  ///      references (ending the wait) cannot run until this one returns.
  ///   2. If idle, wake any blocking driver call (begin_local_shutdown, e.g. an
  ///      indefinite WAIT_EVENTS) so it returns and drops its shared pin. This is
  ///      non-destructive and MUST run BEFORE the exclusive acquire, or the latch
  ///      would wait on a pin whose call only ends once the driver is closed (which
  ///      happens after the latch).
  ///   3. Acquire the exclusive latch (now drains promptly) and RE-CHECK idle for
  ///      the retain-vs-teardown TOCTOU, then unpublish. A dup/fcntl(F_DUPFD*) that
  ///      retained a reference after phase 1 is caught here: retain_backend takes a
  ///      shared pin, so it either completed before this acquire (the recheck sees
  ///      it) or blocks until after unpublication (retain_local_open() then fails).
  ///      If the recheck fails (the driver is still live) teardown is aborted, and
  ///      the phase-2 wake is rolled back via end_local_shutdown() so the surviving
  ///      consumer's blocking calls are not left permanently failing.
  ///   4. Driver unpublished; destroy_local_vm() joins the engine and frees with NO
  ///      latch held.
  void shutdown_local_vm_if_idle_locked() {
    // Hardware DBT mode leaves rj_vm_ == nullptr but still publishes guest_driver_,
    // so a guest-only backend must not be skipped here.
    if (!rj_vm_ && !guest_driver_)
      return;

    // Phase 1: preliminary idle check with only init_mutex_ held.
    if (!local_driver_is_idle_locked())
      return;

    // Phase 2: wake blocking calls so their shared pins drain.
    auto *woken_driver = active_driver_.load(std::memory_order_acquire);
    if (woken_driver != nullptr)
      woken_driver->begin_local_shutdown();

    // Phase 3: exclusive latch, recheck idle (TOCTOU), unpublish.
    {
      std::unique_lock<std::shared_mutex> lifetime(driver_lifetime_);
      if (!local_driver_is_idle_locked()) {
        // A racing dup/retain kept the driver live after phase 1, so abort teardown
        // -- but the phase-2 wake left it in a "closing" state that would wrongly
        // fail the surviving consumer's blocking calls (e.g. WAIT_EVENTS -> -EBADF).
        // Roll that back so the still-live driver keeps serving.
        if (woken_driver != nullptr)
          woken_driver->end_local_shutdown();
        return;
      }
      active_driver_.store(nullptr, std::memory_order_release);
    }

    // Phase 4: unpublished; finish stop/join/free with no latch held (the join must
    // not run under the lifetime latch — see destroy_local_vm). destroy_local_vm
    // re-runs an unpublish under the latch, which is a harmless no-op now.
    destroy_local_vm();
  }

  rj_vm_t *rj_vm_ = nullptr;
  /// @brief Owner of the local VM engine loop; null when no local VM is running.
  /// @details Deliberately unique_ptr<std::thread>, NOT std::jthread: a jthread
  /// destructor always request_stop()+join()s, which would deadlock in the fork
  /// child (the thread does not exist there, so join hangs). reset_after_fork()
  /// must abandon the handle with neither join nor detach; a raw-owned std::thread
  /// lets it .release() the object. Do not switch this to jthread.
  std::unique_ptr<std::thread> local_vm_thread_;
  /// @brief Set once the interposer DSO finalizer has requested process-exit
  /// shutdown; the VM is torn down as soon as it is also idle.
  bool shutdown_requested_ = false;
  std::unique_ptr<GuestKfd> guest_driver_;
  std::atomic<LinuxKfd *> active_driver_{nullptr};
  /// @brief Reader/writer latch guarding the LIFETIME of the object behind
  /// active_driver_ (a SimulatedKfd owned by rj_vm_, or a GuestKfd owned by
  /// guest_driver_ — two owners freed at two sites).
  /// @details A raw active_driver_ pointer has no refcount, so a fast-path deref
  /// (ioctl/mmap/munmap/…) could use it while shutdown_local_vm_if_idle_locked()
  /// freed the owning VM/GuestKfd on another thread. Fast paths take a SHARED lock
  /// for the whole operation (they don't contend with each other, only with
  /// teardown). Teardown does NOT hold the exclusive latch across the frees: it is
  /// an "unpublish and drain, then destroy" sequence. Under the EXCLUSIVE latch it
  /// only stores active_driver_ = nullptr and, by acquiring the latch, waits for
  /// every in-flight shared pin to drain; it then RELEASES the latch before joining
  /// the engine thread and running guest_driver_.reset() / rj_vm_destroy(). Freeing
  /// without the latch is safe because the driver is already unpublished (no new
  /// deref can observe it) and construction is serialized by init_mutex_; keeping
  /// the join off the latch also avoids deadlock if engine shutdown re-enters an
  /// interposed path. The lock is non-reentrant, so it is acquired ONLY at the
  /// top-level libc interposer entry points; the internal driver()/lookup()/
  /// kfd_backend_of() helpers stay lock-free and run under the caller's held guard.
  std::shared_mutex driver_lifetime_;
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
  /// @brief Imported GEM buffer objects, keyed by a stable, minted GEM handle.
  /// @details The handle (never fd-derived) owns the mapping lifetime; entries live
  /// from PRIME_FD_TO_HANDLE until DRM_IOCTL_GEM_CLOSE, or until the owning DRM file
  /// closes (reap_gem_for_drm_fd). Because handles are not recycled with fd numbers,
  /// a reused dmabuf fd can never collide with a still-live BO.
  std::unordered_map<uint32_t, GemEntry> gem_entries_;
  /// @brief Next stable GEM handle to mint. Starts at 1 so 0 means "no handle".
  uint32_t next_gem_handle_ = 1;
  /// @brief Transient EXPORT_DMABUF fd→alloc_flags association awaiting the next
  /// PRIME_FD_TO_HANDLE on the same fd, which folds the flags into a GemEntry and
  /// erases the pending record. Short-lived, so keying by (recyclable) fd is safe.
  std::unordered_map<int, uint32_t> pending_gem_flags_;

  /// @brief Tear down a GEM entry's GPU PTEs and host mapping. Caller holds
  /// fd_mutex_. Removes page-table ranges through the driver that installed them
  /// (owner), BEFORE munmapping cpu_ptr, so the page table never holds pointers
  /// into freed host memory. If that driver is no longer the active one, its page
  /// table is already gone with it, so the PTE removal is skipped (never applied to
  /// a different, replacement driver).
  void teardown_gem_entry_locked(GemEntry &gem) {
    if (gem.owner && gem.owner == dynamic_cast<SimulatedKfd *>(driver())) {
      // The owning driver is still active; remove its PTEs. gem_va_unmap only fails
      // if the local process already vanished, in which case the page table is gone
      // and there is nothing to remove — either way the range must not remain
      // recorded, so the return value is intentionally not actionable here.
      for (const auto &r : gem.installed_vas)
        (void)gem.owner->gem_va_unmap(r.va_address, r.map_size);
    }
    if (gem.cpu_ptr && gem.size)
      InterposerContext::real().munmap(gem.cpu_ptr, gem.size);
    gem.cpu_ptr = nullptr;
    gem.installed_vas.clear();
    // Release our private dup of the backing dmabuf (taken in prime_import) now that
    // no lazy mmap can reference it. Close through the passthrough table so we don't
    // re-enter our own close() hook and its GEM/dup bookkeeping.
    if (gem.owns_dmabuf_fd && gem.dmabuf_fd >= 0)
      InterposerContext::real().close(gem.dmabuf_fd);
    gem.dmabuf_fd = -1;
    gem.owns_dmabuf_fd = false;
  }

  /// @brief Evict every recorded range that OVERLAPS @p range, across all handles.
  /// @details Caller holds fd_mutex_ and passes the live simulated @p drv. Used by
  /// GEM_VA REPLACE (evict prior mappings before installing the new one) and CLEAR
  /// (handle-agnostic teardown). Matching is by interval intersection, not exact
  /// equality: a REPLACE at a VA previously mapped with a DIFFERENT size (or a
  /// sub/super-range) must still evict the old mapping, otherwise its stale
  /// bookkeeping would later double-unmap or leak the new PTEs. Each overlapping
  /// range is unmapped by its OWN {va_address, map_size} extent (not @p range's) so
  /// the page-table removal matches what was installed, then dropped from its entry's
  /// bookkeeping. The host mmap is left intact — the owning handle still exists and
  /// other ranges may reference it; it is munmapped only at GEM_CLOSE / reap.
  /// @param allow_missing When true, no overlap is a success (a MAP onto a free VA has
  ///   nothing to evict); when false (REPLACE/CLEAR), no overlap is a failure so the
  ///   ioctl reports EINVAL.
  /// @retval true nothing overlapped (allow_missing) or all overlaps were evicted.
  /// @retval false nothing overlapped (only when !allow_missing) or an unmap failed.
  /// @note Not rolled back on a mid-loop unmap failure: already-evicted ranges stay
  ///   evicted. This is safe because gem_va_unmap() only fails when the local process
  ///   has already vanished (SimulatedKfd::gem_va_unmap), i.e. its page table is being
  ///   torn down anyway, so a partially-evicted state is never observed by a live GPU.
  [[nodiscard]] bool evict_range_locked(SimulatedKfd *drv, const GemMapping &range,
                                        bool allow_missing) {
    bool evicted_any = false;
    for (auto &[handle, gem] : gem_entries_) {
      for (auto vit = gem.installed_vas.begin(); vit != gem.installed_vas.end();) {
        if (!vit->overlaps(range)) {
          ++vit;
          continue;
        }
        if (!drv->gem_va_unmap(vit->va_address, vit->map_size))
          return false; // process gone; page table already being destroyed (see @note)
        vit = gem.installed_vas.erase(vit);
        evicted_any = true;
      }
    }
    return evicted_any || allow_missing;
  }

  /// @brief Whether any GEM entry owns a range OVERLAPPING @p range. Caller holds
  /// fd_mutex_. Used by plain MAP to reject a map that would collide with (not just
  /// exactly duplicate) an existing range's PTEs.
  [[nodiscard]] bool range_is_mapped_locked(const GemMapping &range) const {
    for (const auto &[handle, gem] : gem_entries_)
      for (const auto &existing : gem.installed_vas)
        if (existing.overlaps(range))
          return true;
    return false;
  }

  std::string invocation_runtime_dir_;

  alignas(16) static uint8_t storage_[];
};

// Storage for the singleton is explicitly shut down (rj_interposer_shutdown →
// request_local_vm_shutdown, which stops and joins the engine once idle) but never
// destructed. Aligned raw storage keeps interposed libc entry points from reaching
// a destroyed context during late process teardown, while the phase-aware shutdown
// still gives the local VM engine thread a bounded, joined lifetime.
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

  // Pin the local driver's lifetime across the driver_fd()/drm_path()/
  // handles_drm_render_minor() reads below. Acquired AFTER get_or_create() above
  // (which takes init_mutex_) to keep the single lock order init_mutex_ ->
  // driver_lifetime_.
  auto driver_lifetime = InterposerContext::ctx.pin_driver_lifetime();

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

    if (!InterposerContext::ctx.get_or_create()) {
      errno = ENODEV;
      return -1;
    }
    // Do NOT carry the raw get_or_create() pointer across the now-released
    // init_mutex_. Re-read driver() under a shared lifetime pin so a racing
    // pending-shutdown close cannot free it between here and the open()/fd() calls.
    int kfd_fd = -1;
    bool clear_dups_needed = false;
    {
      auto driver_lifetime = InterposerContext::ctx.pin_driver_lifetime();
      auto *drv = InterposerContext::ctx.driver();
      if (!drv) {
        errno = ENODEV;
        return -1;
      }
      kfd_fd = drv->open();
      if (kfd_fd < 0)
        return kfd_fd;
      if (kfd_fd != drv->fd())
        InterposerContext::ctx.track_open_fd(kfd_fd);
      clear_dups_needed = !drv->owns_fd(drv->fd());
    }
    // clear_dups() routes through release_local_open() (init_mutex_), so it must run
    // AFTER dropping the shared pin (never shared-pin -> init_mutex_).
    if (clear_dups_needed)
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
  // Drop any transient EXPORT_DMABUF flags for this fd: a dmabuf export fd closed
  // before a PRIME_FD_TO_HANDLE would otherwise leave a stale fd→flags record that a
  // later PRIME on the recycled fd number could misapply as the wrong PTE MTYPE.
  // No-op for non-dmabuf fds.
  InterposerContext::ctx.drop_pending_gem_flags(fd);
  // NOTE: a GEM/dmabuf mapping is NOT torn down when a transient dmabuf EXPORT fd
  // closes. ROCr closes that fd immediately after VMemorySetAccessPerHandle()
  // returns, while the GPU mapping must stay live for the caller. GEM state is keyed
  // by a stable GEM handle and released on DRM_IOCTL_GEM_CLOSE (see the ioctl
  // handler). Closing the DRM FILE itself, however, is the true backstop: reap any
  // handles still open on it (mirroring the kernel dropping a drm_file's GEM
  // objects) so a leaked/never-GEM_CLOSE'd handle cannot outlive its DRM file.
  if (InterposerContext::ctx.untrack_drm(fd)) {
    InterposerContext::ctx.reap_gem_for_drm_fd(fd);
    InterposerContext::real().close(fd);
    return 0;
  }
  if (InterposerContext::ctx.is_kfd_dup(fd)) {
    InterposerContext::ctx.untrack_dup(fd);
    return static_cast<int>(InterposerContext::real().close(fd));
  }
  // Classify the fd against the local driver under a lifetime pin (lookup()/
  // owns_fd() deref the driver), then DROP the pin before release_local_open()
  // (which takes init_mutex_) to preserve the init_mutex_ -> driver_lifetime_
  // order and avoid deadlock.
  bool is_primary_local_fd = false;
  bool is_owned_fd = false;
  {
    auto driver_lifetime = InterposerContext::ctx.pin_driver_lifetime();
    is_primary_local_fd = InterposerContext::ctx.lookup(fd) != nullptr;
    if (!is_primary_local_fd)
      is_owned_fd = InterposerContext::ctx.owns_fd(fd);
  }
  if (is_primary_local_fd) {
    // Route through release_local_open() (not a bare driver close) so the last
    // app-facing close also completes a pending process-exit shutdown, letting the
    // local VM be torn down once no KFD reference remains.
    InterposerContext::ctx.release_local_open();
    return 0;
  }
  if (is_owned_fd)
    return 0;
  return static_cast<int>(InterposerContext::real().close(fd));
}

// Runs at library finalization. As an LD_PRELOAD lib we may finalize before HIP
// and ROCR, which can still hold KFD descriptors and issue an implicit
// hsa_shut_down() from THEIR finalizers. So only RECORD the shutdown request here;
// the VM is destroyed later, when the final KFD client closes and the simulator is
// idle (via release_local_open()). Destroying it now would make HIP's later scratch
// release hang against a gone driver.
//
// This deferred, reference-counted teardown is what makes cross-library ordering
// safe — NOT the destructor priority. GCC destructor priorities only order
// .fini_array entries WITHIN this DSO; the KFD-closing finalizers that matter live
// in other DSOs (HIP/ROCR) and are ordered relative to librocjitsu.so by dynamic
// finalization order (an LD_PRELOAD lib finalizes before its dependents), which no
// attribute here can change. Since this finalizer is now idempotent and only
// records a request, its priority is immaterial; default priority is used simply
// because the old destructor(101) special-casing is no longer needed.
__attribute__((destructor)) void rj_interposer_shutdown() {
  InterposerContext::ctx.request_local_vm_shutdown();
}

RJ_INTERPOSER_EXPORT int ioctl(int fd, unsigned long request, ...) {
  assert(InterposerContext::real().ready());
  va_list ap;
  va_start(ap, request);
  void *arg = va_arg(ap, void *);
  va_end(ap);

  constexpr unsigned kDrmIoctlType = 'd';
  constexpr unsigned kDrmIoctlNrVersion = 0x00;
  constexpr unsigned kDrmIoctlNrGemClose = _IOC_NR(DRM_IOCTL_GEM_CLOSE);
  constexpr unsigned kDrmIoctlNrAmdgpuInfo = DRM_COMMAND_BASE + DRM_AMDGPU_INFO;
  constexpr unsigned kDrmIoctlNrGemVa = DRM_COMMAND_BASE + DRM_AMDGPU_GEM_VA;
  constexpr unsigned kDrmIoctlNrPrimeFdToHandle = _IOC_NR(DRM_IOCTL_PRIME_FD_TO_HANDLE);

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
      auto *prime = static_cast<drm_prime_handle *>(arg);
      if (prime->fd < 0) {
        errno = EINVAL;
        return -1;
      }

      // Size the BO so GEM_VA can map the dmabuf into the GPU page table. The dmabuf
      // fd is mmap-able (it dups the allocation's backing memfd). Use the real fstat
      // (not the interposed one) so we don't re-enter our own hook.
      uint64_t sz = 0;
      struct stat st {};
      if (InterposerContext::real().fstat_fn(prime->fd, &st) == 0 && st.st_size > 0) {
        sz = static_cast<uint64_t>(st.st_size);
      } else {
        // No size means a later GEM_VA MAP on this handle will fail (EINVAL);
        // log here so that EINVAL is traceable to its real cause rather than
        // looking like a bad map request.
        util::Logger::warn("PRIME_FD_TO_HANDLE: dmabuf fd=", prime->fd,
                           " has no usable size (fstat st_size<=0); GEM_VA maps for the minted "
                           "handle will fail");
      }
      // Mint a stable handle (independent of the fd number) scoped to this DRM file,
      // folding in the alloc flags captured at EXPORT_DMABUF. The caller closes the
      // export fd right after access setup; the handle — not the fd — owns the BO.
      prime->handle = InterposerContext::ctx.prime_import(prime->fd, fd, sz);
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
    if (type == kDrmIoctlType && nr == kDrmIoctlNrGemClose && arg) {
      // GEM_CLOSE releases a GEM handle — the true end of the imported BO's
      // lifetime (the transient dmabuf export fd was closed long ago). untrack_gem
      // removes the GPU page-table ranges (through the driver that installed them)
      // BEFORE munmapping the host pages, entirely under fd_mutex_ so no state
      // escapes the lock.
      auto *gc = static_cast<drm_gem_close *>(arg);
      InterposerContext::ctx.untrack_gem(gc->handle);
      return 0;
    }
    if (type == kDrmIoctlType && nr == kDrmIoctlNrGemVa && arg) {
      // GEM_VA installs (or tears down) a GPU virtual mapping for a prime-
      // imported buffer. HSA's vmem path (hsa_amd_vmem_map) lowers to this via
      // amdgpu_bo_va_op; IREE's ring allocator triple-maps one BO at adjacent
      // VAs. We map by GEM handle, lazily mmap the backing pages, and
      // install/remove them in the GPU page table.
      auto *va = static_cast<drm_amdgpu_gem_va *>(arg);
      // GEM_VA page-table installation only applies to the local simulated driver;
      // there is no such path for a remote (daemon) or DBT-guest backend. Report
      // failure rather than a phantom success so userspace does not record a mapping
      // that was never installed and later fault on the GPU page table. gem_map/
      // gem_unmap do the bookkeeping AND the page-table mutation atomically under
      // fd_mutex_ (so a concurrent GEM_CLOSE cannot leave dangling PTEs); they
      // return false on no-driver / unknown handle / out-of-bounds / failed mmap.
      // Pin the local driver's lifetime across the whole GEM_VA op: the check and
      // gem_map/gem_unmap/gem_clear (which deref the local driver under fd_mutex_)
      // must all see a live driver. Order is driver_lifetime_ -> fd_mutex_, which
      // does not invert against teardown (init_mutex_ -> driver_lifetime_).
      auto driver_lifetime = InterposerContext::ctx.pin_driver_lifetime();
      if (!InterposerContext::ctx.driver_is_simulated()) {
        errno = ENODEV;
        return -1;
      }
      bool ok = false;
      switch (va->operation) {
      case AMDGPU_VA_OP_MAP:
      case AMDGPU_VA_OP_REPLACE:
        // REPLACE evicts any prior occupant of the VA range (from whatever handle
        // owns it) before installing the new mapping, so closing the old handle
        // cannot later unmap the replacement. MAP rejects a range already in use.
        ok = InterposerContext::ctx.gem_map(va->handle, va->va_address, va->offset_in_bo,
                                            va->map_size,
                                            /*replace=*/va->operation == AMDGPU_VA_OP_REPLACE);
        break;
      case AMDGPU_VA_OP_UNMAP:
        // UNMAP requires the supplied handle to own the exact range.
        ok = InterposerContext::ctx.gem_unmap(va->handle, va->va_address, va->map_size);
        break;
      case AMDGPU_VA_OP_CLEAR:
        // CLEAR is handle-agnostic: tear down the exact range wherever it lives.
        ok = InterposerContext::ctx.gem_clear(va->va_address, va->map_size);
        break;
      default:
        // Unknown GEM_VA operation — do not claim it succeeded.
        errno = EINVAL;
        return -1;
      }
      if (!ok) {
        errno = EINVAL;
        return -1;
      }
      return 0;
    }
    // Only DRM command ioctls (nr >= DRM_COMMAND_BASE) carry an AMDGPU-relative
    // command number; core DRM ioctls (nr < DRM_COMMAND_BASE) would underflow and
    // log a nonsense "AMDGPU cmd", so report the raw nr for those.
    if (nr >= DRM_COMMAND_BASE) {
      util::Logger::warn("DRM ioctl rejected: nr=0x", std::hex, nr, " (AMDGPU cmd 0x",
                         nr - DRM_COMMAND_BASE, std::dec, ") fd=", fd);
    } else {
      util::Logger::warn("DRM ioctl rejected: nr=0x", std::hex, nr, std::dec,
                         " (core DRM) fd=", fd);
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
  {
    // Pin the driver's lifetime across the classification and the LOCAL dispatch:
    // the classifier kfd_backend_of() -> driver_fd() dereferences the local driver,
    // so a concurrent teardown could free it between the load and the fd() read if
    // the pin were taken only on the Local branch. std::shared_mutex is
    // non-recursive, so this single pin covers classification + the local branch (no
    // nested pin inside). The block takes no init_mutex_; it takes fd_mutex_
    // (track_gem_flags) only under this pin, matching the driver_lifetime_ ->
    // fd_mutex_ order.
    //
    // The Remote branch must NOT dispatch under the local pin: remote->ioctl() can
    // block indefinitely (e.g. a daemon-side WAIT_EVENTS), which would hold the
    // local pin open and stall teardown of an idle LOCAL VM on an unrelated remote
    // operation. So on the Remote branch, capture the remote shared_ptr snapshot,
    // DROP the pin, then dispatch — the snapshot keeps the remote alive on its own.
    std::shared_ptr<RemoteDriver> remote_dispatch;
    {
      auto lifetime = InterposerContext::ctx.pin_driver_lifetime();
      if (auto backend = InterposerContext::ctx.kfd_backend_of(fd)) {
        if (*backend == InterposerContext::DupBackend::Remote) {
          // kfd_backend_of() already established this fd is Remote-backed, so route
          // via the remote snapshot directly rather than remote_lookup(remote_kfd_fd_):
          // the primary fd number may have been invalidated/reused while a remote
          // shared_ptr snapshot is still live, and this dup still belongs to it.
          remote_dispatch = InterposerContext::ctx.remote();
        } else if (auto *drv = InterposerContext::ctx.driver()) {
          int rc = drv->ioctl(request, arg);
          // Capture the KFD allocation flags for a freshly exported dmabuf fd. The
          // flags determine the GPU PTE MTYPE when the fd is later mapped via GEM_VA,
          // and must be recorded now because the allocation may be freed first. Only
          // the local simulated driver exports dmabufs this path can later map.
          if (rc == 0 && request == AMDKFD_IOC_EXPORT_DMABUF && arg) {
            if (auto *sim = dynamic_cast<SimulatedKfd *>(drv)) {
              auto *export_args = static_cast<kfd_ioctl_export_dmabuf_args *>(arg);
              // alloc_flags_for_handle locks the process alloc mutex internally, so the
              // interposer does not reach into driver-private per-process state.
              InterposerContext::ctx.track_gem_flags(
                  static_cast<int>(export_args->dmabuf_fd),
                  sim->alloc_flags_for_handle(export_args->handle));
            }
          }
          return kfd_ioctl_ret(rc);
        }
      }
    }
    // Pin dropped: dispatch the remote op on the lifetime-extending snapshot.
    if (remote_dispatch)
      return kfd_ioctl_ret(remote_dispatch->ioctl(request, arg));
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
    // kfd_backend_of() -> driver_fd() dereferences the local driver, so pin its
    // lifetime for the classification (the routed remote->ioctl is separately
    // lifetime-safe via its shared_ptr snapshot).
    InterposerContext::DupBackend backend;
    bool have_backend = false;
    {
      auto driver_lifetime = InterposerContext::ctx.pin_driver_lifetime();
      if (auto b = InterposerContext::ctx.kfd_backend_of(fd)) {
        backend = *b;
        have_backend = true;
      }
    }
    if (have_backend && backend == InterposerContext::DupBackend::Remote) {
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
  // dup2/dup3 atomically close whatever newfd was, bypassing the close() hook, so
  // every per-fd cleanup close() performs must be mirrored here. Drop any transient
  // EXPORT_DMABUF flags for newfd: a dmabuf export fd overwritten before a
  // PRIME_FD_TO_HANDLE would otherwise leave a stale fd→flags record that a later
  // PRIME on the recycled fd number could misapply as the wrong PTE MTYPE. No-op for
  // non-dmabuf fds.
  InterposerContext::ctx.drop_pending_gem_flags(newfd);
  // If newfd was a DRM render fd still owning live GEM handles, reap them here just as
  // close() does (untrack_drm + reap_gem_for_drm_fd) — otherwise those GemEntry
  // objects (keyed by drm_fd == newfd) leak their PTEs, host mmap, and dup'd dmabuf
  // fd, and a later commit_dup could re-tag the same number as a KFD dup.
  if (InterposerContext::ctx.untrack_drm(newfd))
    InterposerContext::ctx.reap_gem_for_drm_fd(newfd);
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

  // Classify + dispatch the LOCAL branches under a shared lifetime pin so a
  // concurrent teardown cannot free the VM/GuestKfd mid-mmap. The pin scopes ONLY
  // the local derefs (lookup/kfd_backend_of/is_drm and the local driver mmap): a
  // REMOTE mmap performs a synchronous RPC that can block indefinitely on a
  // stalled daemon, so holding the local pin across it would stall teardown of an
  // otherwise-idle local VM (mirrors the ioctl() path). For a Remote-backed fd,
  // capture a lifetime-extending remote snapshot here, DROP the pin, then dispatch.
  std::shared_ptr<RemoteDriver> remote_mmap;
  {
    auto driver_lifetime = InterposerContext::ctx.pin_driver_lifetime();

    if (auto *drv = InterposerContext::ctx.lookup(fd))
      return drv->mmap(addr, length, prot, flags, offset);

    // Dispatch a tracked KFD dup by its RECORDED backend (as ioctl() does), not by
    // "remote if any remote is live"; otherwise a Local-tagged dup would misroute
    // to the remote in mixed local+daemon mode. For Remote, route via the remote
    // snapshot directly so routing still works if the primary fd number changed.
    if (auto backend = InterposerContext::ctx.kfd_backend_of(fd)) {
      if (*backend == InterposerContext::DupBackend::Remote) {
        remote_mmap = InterposerContext::ctx.remote();
      } else if (auto *drv = InterposerContext::ctx.driver()) {
        return drv->mmap(addr, length, prot, flags, offset);
      }
    }

    if (!remote_mmap && InterposerContext::ctx.is_drm(fd)) {
      // Route via the remote snapshot (not remote_lookup(remote_kfd_fd())): a
      // dup2/dup3 may have cleared the primary fd number while the connection is
      // still live via other refs, and this DRM fd still belongs to that remote.
      remote_mmap = InterposerContext::ctx.remote();
      if (!remote_mmap) {
        if (auto *drv = InterposerContext::ctx.driver())
          return drv->mmap(addr, length, prot, flags, offset);
      }
    }
  }
  // Pin dropped: dispatch the remote op on the lifetime-extending snapshot so a
  // stalled daemon cannot block local-VM teardown.
  if (remote_mmap)
    return remote_mmap->mmap(addr, length, prot, flags, offset);

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
  {
    auto driver_lifetime = InterposerContext::ctx.pin_driver_lifetime();
    auto *drv = InterposerContext::ctx.driver();
    if (drv && drv->is_doorbell_range(addr, length)) {
      errno = EPERM;
      return -1;
    }
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
  {
    auto driver_lifetime = InterposerContext::ctx.pin_driver_lifetime();
    if (auto *drv = InterposerContext::ctx.driver()) {
      int ret = drv->munmap(addr, length);
      if (ret != -ENOENT)
        return ret;
    }
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
  {
    // Pin the local driver's lifetime across the redirect_sysfs_path()/drm_path()
    // reads (both return owned strings, so no pointer escapes the pin).
    auto driver_lifetime = InterposerContext::ctx.pin_driver_lifetime();
    auto *drv = InterposerContext::ctx.driver();
    if (drv) {
      auto direct = drv->redirect_sysfs_path(path);
      if (!direct.empty())
        return direct;
      drm_base = drv->drm_path();
    } else {
      drm_base = InterposerContext::ctx.remote_drm_path();
    }
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
  {
    // Pin the local driver's lifetime while reading its GpuInfo; the result is
    // returned by value so nothing points into the driver after the pin drops.
    auto driver_lifetime = InterposerContext::ctx.pin_driver_lifetime();
    if (auto *drv = InterposerContext::ctx.driver()) {
      if (const Sysfs::GpuInfo *info = drv->gpu_info_for_render_minor(render_minor))
        return *info;
      return std::nullopt;
    }
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
  {
    // Pin the local driver's lifetime across the redirect_sysfs_path()/drm_path()
    // reads (both return owned strings, so no pointer escapes the pin).
    auto driver_lifetime = InterposerContext::ctx.pin_driver_lifetime();
    auto *drv = InterposerContext::ctx.driver();
    if (drv) {
      auto direct = drv->redirect_sysfs_path(path);
      if (!direct.empty())
        return direct;
      drm_base = drv->drm_path();
    } else {
      drm_base = InterposerContext::ctx.remote_drm_path();
    }
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

// fork() is intentionally NOT interposed: the child reset is registered via
// pthread_atfork() in InterposerContext::init(), which libc runs for every
// fork-family primitive that goes through glibc (fork/system/popen), not just an
// interposed fork() symbol. A passthrough wrapper here would double-run the reset.

} // extern "C"
