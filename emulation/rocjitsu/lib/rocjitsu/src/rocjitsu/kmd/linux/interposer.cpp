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
#include "rocjitsu/kmd/linux/host_mapping_lock.h"
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
#include <libdrm/amdgpu_drm.h>
#include <libdrm/drm.h>
RJ_DIAGNOSTIC_POP

#include "util/dynamic_loader.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <exception>
#include <fcntl.h>
#include <limits>
#include <linux/futex.h>
#include <linux/memfd.h>
#include <memory>
#include <mutex>
#include <optional>
#include <pthread.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/uio.h>
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

static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "futex-backed state requires a lock-free uint32_t atomic");

/// @brief Sleep while @p word still equals @p expected, until @p deadline or a signal.
long futex_wait_until(std::atomic<uint32_t> &word, uint32_t expected, const timespec *deadline) {
  return syscall(SYS_futex, &word, FUTEX_WAIT_BITSET_PRIVATE, expected, deadline, nullptr,
                 FUTEX_BITSET_MATCH_ANY);
}

/// @brief Wake every waiter sleeping on @p word.
void futex_wake_all(std::atomic<uint32_t> &word) {
  (void)syscall(SYS_futex, &word, FUTEX_WAKE_PRIVATE, std::numeric_limits<int>::max(), nullptr,
                nullptr, 0);
}

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

bool is_proc_maps_path(const char *path) {
  if (!path || !std::string_view(path).starts_with("/proc/"))
    return false;
  const std::string_view name(path);
  return name.ends_with("/maps") || name.ends_with("/smaps");
}

int open_proc_maps_snapshot(const char *path, int flags) {
  if (!is_proc_maps_path(path) || (flags & O_ACCMODE) != O_RDONLY)
    return -1;

  auto &libc = rocjitsu::libc_passthrough();
  int source = libc.openat(AT_FDCWD, path, flags, 0);
  if (source < 0)
    return -1;
  std::string contents;
  std::array<char, 16384> buffer{};
  while (true) {
    ssize_t count = libc.read(source, buffer.data(), buffer.size());
    if (count <= 0)
      break;
    contents.append(buffer.data(), static_cast<size_t>(count));
  }
  int read_errno = errno;
  libc.close(source);
  if (contents.find("/memfd:rocjitsu_remote_kfd (deleted)") == std::string::npos) {
    errno = read_errno;
    return -1;
  }

  constexpr std::string_view marker = "/memfd:rocjitsu_remote_kfd (deleted)";
  size_t position = 0;
  while ((position = contents.find(marker, position)) != std::string::npos) {
    contents.replace(position, marker.size(), "/dev/kfd");
    position += sizeof("/dev/kfd") - 1;
  }

  int snapshot = libc.memfd_create("rocjitsu_proc_maps", MFD_CLOEXEC);
  if (snapshot < 0)
    return -1;
  if (libc.write(snapshot, contents.data(), contents.size()) !=
          static_cast<ssize_t>(contents.size()) ||
      lseek(snapshot, 0, SEEK_SET) < 0) {
    int saved_errno = errno;
    libc.close(snapshot);
    errno = saved_errno;
    return -1;
  }
  return snapshot;
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

/// @brief Read the child-process rocjitsu config handoff from @p cfg_file.
///
/// @details The first line is the config path. DBT launches include the resolved
/// host KFD gpu_id on the second line.
std::optional<rocjitsu::config::DbtRuntimeConfigHandoff>
child_config_handoff(const std::string &cfg_file) {
  constexpr size_t kMaxHandoffSize = 64 * 1024;
  constexpr size_t kReadChunkSize = 4096;
  auto &real = rocjitsu::libc_passthrough();
  int cfg_fd = real.openat(AT_FDCWD, cfg_file.c_str(), O_RDONLY, 0);
  if (cfg_fd < 0)
    return std::nullopt;

  std::string contents;
  contents.reserve(kReadChunkSize);
  while (contents.size() < kMaxHandoffSize) {
    char buffer[kReadChunkSize];
    const size_t remaining = kMaxHandoffSize - contents.size();
    ssize_t bytes_read = 0;
    do {
      bytes_read = real.read(cfg_fd, buffer, std::min(sizeof(buffer), remaining));
    } while (bytes_read < 0 && errno == EINTR);
    if (bytes_read < 0) {
      real.close(cfg_fd);
      return std::nullopt;
    }
    if (bytes_read == 0)
      break;
    contents.append(buffer, static_cast<size_t>(bytes_read));
  }

  char extra = 0;
  ssize_t extra_bytes = 0;
  do {
    extra_bytes = real.read(cfg_fd, &extra, 1);
  } while (extra_bytes < 0 && errno == EINTR);
  real.close(cfg_fd);
  if (contents.empty() || extra_bytes != 0)
    return std::nullopt;

  return rocjitsu::config::parse_dbt_runtime_config_handoff(contents);
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

void *raw_mremap_syscall(void *old_address, size_t old_size, size_t new_size, int flags,
                         void *new_address) {
  long rc = syscall(SYS_mremap, old_address, old_size, new_size, flags, new_address);
  if (rc < 0)
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

  /// @brief Hold in_construction for a scope, restoring it on every exit path.
  ///
  /// @details Clearing the flag by hand is a fail-OPEN hazard rather than a leak:
  /// while it is set this thread bypasses path redirection entirely, so a throw that
  /// skipped a clear would leave the thread opening the HOST's real device nodes for
  /// the rest of its life -- the exact outcome the child gate exists to prevent.
  /// Restores the previous value rather than clearing, so a nested construction
  /// cannot end the outer one's scope early.
  class ConstructionScope {
  public:
    ConstructionScope() : previous_(in_construction) { in_construction = true; }
    ~ConstructionScope() { in_construction = previous_; }
    ConstructionScope(const ConstructionScope &) = delete;
    ConstructionScope &operator=(const ConstructionScope &) = delete;

  private:
    bool previous_;
  };

  static rocjitsu::LibcPassthrough &real() { return rocjitsu::libc_passthrough(); }
  static InterposerContext &ctx;

  static void init() {
    new (storage_) InterposerContext();
    ctx.owner_pid_ = getpid();
    // Record the HOST KFD device identity once, here, while single-threaded and
    // before real().resolve() flips the gate. A forked child compares a resolved
    // open target against this to decide whether it is about to touch real GPU
    // hardware. Captured as an immutable dev_t rather than a pathname because only
    // identity survives symlinks, chained aliases, dirfd-relative spellings and
    // bind mounts; a name comparison does not. Absent host KFD leaves it 0, which
    // matches nothing.
    {
      struct stat kfd_st {};
      if (::stat("/dev/kfd", &kfd_st) == 0 && S_ISCHR(kfd_st.st_mode))
        ctx.host_kfd_rdev_ = kfd_st.st_rdev;
    }
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
    // NO pthread_atfork handlers, deliberately. rocJITsu's local mode keeps the
    // simulator's state -- the VM, its engine thread, the driver objects and their
    // private mutexes -- inside this address space, and a driver call holds those
    // private locks for its whole duration, sometimes across a blocking wait. A
    // prepare handler therefore cannot drain them: waiting on a lock held by a
    // thread parked in an indefinite WAIT_EVENTS would hang fork() itself. The
    // allocator-style "lock everything in prepare" pattern only works when no lock
    // is ever held across a blocking call, which is false here.
    //
    // So local mode takes the same contract as every comparable in-process GPU
    // runtime (CUDA, HSA/ROCr) and as ThreadSanitizer: rocJITsu services are
    // UNAVAILABLE between fork/vfork and exec. Enforcement is owner_pid_, compared
    // at the top of every interposed entry point before any inherited lock,
    // container, driver or shared_ptr is touched, so a child can never reach state a
    // vanished parent thread was mutating. That is a structural guarantee rather
    // than a narrowed race, and it needs no child-side cleanup at all.
    real().resolve();
  }

  std::shared_ptr<LinuxKfd> driver() const {
    std::lock_guard lock(publish_mutex_);
    return active_driver_;
  }
  void publish_driver(std::shared_ptr<LinuxKfd> driver) {
    // Let the displaced driver die AFTER the lock is dropped. If the swap ever drops
    // the last reference, its deleter runs -- for the guest driver that is engine
    // exit, thread join and VM destroy -- and publish_mutex_ is the innermost lock,
    // taken by every reader. Today the callers always keep a second owner alive
    // across the swap, but that is an invariant of the call sites rather than of
    // this function, and it is cheaper to not depend on it.
    std::shared_ptr<LinuxKfd> displaced;
    {
      std::lock_guard lock(publish_mutex_);
      displaced = std::exchange(active_driver_, std::move(driver));
    }
  }
  /// @brief Publish (or clear) the active remote driver.
  /// @details Caller MUST hold remote_mutex_. That is what makes the compound
  /// create+open and teardown sequences atomic with respect to every reader that
  /// takes remote_mutex_; publish_mutex_ alone would only make the pointer swap
  /// itself safe. Both call sites -- get_or_create_remote() and
  /// teardown_remote_locked() -- hold it.
  void publish_remote(std::shared_ptr<RemoteDriver> remote) {
    std::shared_ptr<RemoteDriver> displaced;
    {
      std::lock_guard lock(publish_mutex_);
      displaced = std::exchange(remote_, std::move(remote));
    }
    // Destroyed here, outside publish_mutex_, for the reason publish_driver() gives.
  }

  /// @brief True if the active driver is the local SimulatedKfd (not remote/guest).
  bool driver_is_simulated() { return dynamic_cast<SimulatedKfd *>(driver().get()) != nullptr; }
  /// @brief The active local driver's primary fd, or -1 if none.
  int driver_fd() {
    auto d = driver();
    return d ? d->fd() : -1;
  }
  bool initialized() const { return driver() != nullptr || remote() != nullptr; }

  /// @brief Record a process-exit shutdown request from the DSO finalizer.
  /// @details As an LD_PRELOAD library, librocjitsu.so can be finalized before HIP
  /// and ROCR. If they still hold app-facing KFD descriptors, this request stays
  /// pending and the VM is kept alive; their later close paths complete it. If the
  /// simulator is already idle, this tears the VM down now.
  void request_local_vm_shutdown() {
    // Declared before the lock so it is destroyed AFTER the lock is dropped; see
    // DoomedLocalVm.
    DoomedLocalVm doomed;
    {
      std::lock_guard lock(init_mutex_);
      shutdown_requested_ = true;
      doomed = shutdown_local_vm_if_idle_locked();
    }
  }

  /// @brief Create-if-needed and open the local backend as ONE atomic operation.
  /// @details Holds init_mutex_ across backend creation, driver->open(), and the fd
  /// bookkeeping. That is required, not merely tidy: the shutdown decision runs
  /// under this same lock, so splitting create from open would let the finalizer
  /// observe the driver at its baseline reference count and retire it in the window
  /// after get_or_create() returns and before the application's reference exists —
  /// handing the caller an fd whose backend is already being destroyed. The remote
  /// path (get_or_create_remote) is combined for the same reason.
  /// @returns The app-visible fd, or -1 with errno set.
  struct LocalOpen {
    int fd = -1;
    bool clear_dups_needed = false;
  };
  LocalOpen open_local() {
    std::lock_guard lock(init_mutex_);
    auto drv = get_or_create_locked();
    if (!drv) {
      errno = ENODEV;
      return {};
    }
    LocalOpen result;
    result.fd = drv->open();
    if (result.fd < 0)
      return result;
    if (result.fd != drv->fd())
      track_open_fd(result.fd);
    result.clear_dups_needed = !drv->owns_fd(drv->fd());
    return result;
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
    // Declared before the lock so it is destroyed AFTER the lock is dropped; see
    // DoomedLocalVm.
    DoomedLocalVm doomed;
    {
      std::lock_guard lock(init_mutex_);
      auto active_driver = driver();
      // Cancel BEFORE the close, not after. This close may be the last app reference,
      // and the driver's own close() then tears the process down -- after which
      // begin_local_shutdown() has nothing left to find. A thread parked in
      // WAIT_EVENTS would be woken by the destructive path with -EBADF instead of the
      // benign timeout its caller re-polls on, and a thread parked on the debugger
      // handshake would not be released at all and would hold its driver snapshot for
      // the whole handshake deadline. Gated on shutdown_requested_, which is only ever
      // set and never cleared, so an ordinary close is untouched: cancelling early is
      // sound exactly when the process is already on its way out.
      if (active_driver && shutdown_requested_)
        active_driver->begin_local_shutdown();
      if (active_driver)
        active_driver->close();
      if (shutdown_requested_)
        doomed = shutdown_local_vm_if_idle_locked();
    }
  }

  /// @brief Take a lifetime-extending snapshot of the active remote driver.
  /// @details The returned shared_ptr keeps the RemoteDriver alive for as long as
  /// the caller holds it, even if a concurrent teardown_remote() clears remote_.
  std::shared_ptr<RemoteDriver> remote() const {
    std::lock_guard lock(publish_mutex_);
    return remote_;
  }

  int remote_kfd_fd() const { return remote_kfd_fd_.load(std::memory_order_acquire); }

  std::shared_ptr<RemoteDriver> remote_lookup(int fd) {
    auto active_remote = remote();
    return (fd >= 0 && fd == remote_kfd_fd_.load(std::memory_order_acquire) && active_remote)
               ? active_remote
               : nullptr;
  }

  /// @brief The per-invocation runtime directory for this process image.
  /// @details Populated once in init() before any thread or app fork, so this is
  /// a lock-free immutable read. A forked app child inherits the parent's value
  /// (the child never mutates it) and thus would reconnect to
  /// the same daemon rather than recomputing a dir under its own PID.
  const std::string &invocation_runtime_dir() const { return invocation_runtime_dir_; }

  bool owned_by_current_process() const { return owner_pid_ == getpid(); }
  /// @brief st_rdev of the host's real KFD, or 0 when the host has none.
  [[nodiscard]] dev_t host_kfd_rdev() const { return host_kfd_rdev_; }

  // No lock needed: the snapshot keeps the RemoteDriver alive, and its handshake
  // metadata (topology/drm paths, gpu_info) is immutable after open() — close()
  // does not clear it — so this read never races a concurrent teardown. Callers
  // that need BOTH the topology and drm paths together must take a single
  // remote() snapshot and read both from it (see redirect_sysfs_path), so the two
  // paths can't come from different RemoteDrivers across a teardown/reconnect.
  std::string remote_drm_path() {
    auto active_remote = remote();
    return active_remote ? std::string(active_remote->drm_path()) : std::string{};
  }

  /// @brief Retain one remote open reference if a remote connection is live.
  /// @details Combined check-and-retain under remote_mutex_ so a concurrent
  /// last-release+teardown cannot slip between "is a remote live?" and the
  /// increment and resurrect a torn-down connection. Returns true if a reference
  /// was added (i.e. a remote is active). Mirrors the local path, where
  /// reserve_dup_backend() runs the classify+retain under init_mutex_, so a
  /// concurrent teardown cannot interleave with it.
  bool retain_remote_open_if_active() {
    std::lock_guard lock(remote_mutex_);
    if (!remote())
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
    auto active_remote = remote();
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
    publish_remote(active_remote);
    return {active_remote, fd};
  }

  /// @brief The active local driver if @p fd is its primary fd, else null.
  /// @details Returns a snapshot, so the caller's dereference is lifetime-safe on
  /// its own — no separate guard to remember.
  std::shared_ptr<LinuxKfd> lookup(int fd) {
    auto d = driver();
    return (d && fd >= 0 && fd == d->fd()) ? d : nullptr;
  }

  bool owns_fd(int fd) {
    auto d = driver();
    return d && d->owns_fd(fd);
  }

  std::string redirect(const char *path) {
    auto d = driver();
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
    if (auto active_remote = remote()) {
      std::string remote_topology(active_remote->topology_path());
      if (!remote_topology.empty()) {
        std::string redirected = LinuxKfd::redirect_sysfs_root_path(
            path, remote_topology, std::string(active_remote->drm_path()));
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
  /// (reserve_dup_backend) runs under init_mutex_ across this call, so a
  /// concurrent teardown cannot free the driver between the load and
  /// retain_local_open(). retain_local_open() reports false if the local process
  /// was already torn down (a racing last-close).
  bool retain_backend(DupBackend backend) {
    if (backend == DupBackend::Local) {
      auto d = driver();
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
    // Classify once WITHOUT init_mutex_ first, purely to get out of the way. Every
    // dup/dup2/dup3/fcntl(F_DUPFD*) in the process reaches here, on any descriptor,
    // and init_mutex_ is held for the whole of VM bring-up and teardown -- so taking
    // it unconditionally would queue an unrelated socket dup behind a simulator
    // lifecycle, while that dup holds drm_fd_lifecycle_mutex_ and thereby blocks
    // every close() too. kfd_backend_of() is safe to call unlocked: it snapshots the
    // driver internally and takes only the fd table's own lock.
    if (!kfd_backend_of(src_fd))
      return std::nullopt;

    // Now the authoritative classify+retain, under init_mutex_ -- the SAME lock
    // shutdown_local_vm_if_idle_locked() makes its idle decision under. That mutual
    // exclusion is what removes the retain-vs-teardown race at the root: this either
    // completes first (and teardown's idle check observes the new reference) or runs
    // after teardown unpublished (and finds no driver, failing closed). Neither side
    // needs to re-check the other or undo anything. The unlocked probe above is only
    // a filter; it is deliberately re-done here rather than trusted.
    std::lock_guard lock(init_mutex_);
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
    // Classify first, then release_backend(): the classification only reads the
    // driver, while release_backend() may drop the last reference and free it.
    bool aliases_primary = false;
    if (fd < 0) {
      aliases_primary = true;
    } else {
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
    // The driver snapshot keeps the object alive across invalidate_primary_fd();
    // it is released before release_local_open() only so the object can actually be
    // freed there when this was the last reference.
    LinuxKfd::PrimaryInvalidation invalidation = LinuxKfd::PrimaryInvalidation::kNotPrimary;
    {
      if (auto d = driver())
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
    auto active_remote = remote();
    if (!active_remote)
      return nullptr;
    // Clear the published pointer first so no new reader can pick this driver up.
    // The RemoteDriver is destroyed by the last shared_ptr: if a racing reader
    // still holds a snapshot mid-ioctl/mmap, destruction (and socket close in
    // ~RemoteDriver) is deferred until it releases, so there is no use-after-free.
    publish_remote(nullptr);
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

  struct SyncobjEntry {
    bool has_fence = false;
    uint64_t submitted_point = 0;
    uint64_t signaled_point = 0;
  };

  /// @brief Move-only operational reference to the backend serving a DRM file.
  /// @details Distinct from a plain shared_ptr, which only prevents deallocation:
  /// a GuestKfd whose last KFD reference is dropped clears readiness and its
  /// synthetic mappings while the object still exists, and a RemoteDriver's RPC
  /// connection is closed on its last reference. A DRM file outlives neither, so it
  /// holds an actual open reference for its whole life. Releasing a LOCAL lease
  /// routes through release_local_open(), the same path a KFD close takes, which is
  /// what lets a final DRM close complete a pending process-exit shutdown.
  class DrmBackendLease {
  public:
    DrmBackendLease() : context_(nullptr) {}
    DrmBackendLease(InterposerContext *context, std::shared_ptr<LinuxKfd> local)
        : context_(context), local_(std::move(local)) {}
    DrmBackendLease(InterposerContext *context, std::shared_ptr<RemoteDriver> remote)
        : context_(context), remote_(std::move(remote)) {}

    DrmBackendLease(DrmBackendLease &&other) noexcept : context_(nullptr) {
      *this = std::move(other);
    }
    DrmBackendLease &operator=(DrmBackendLease &&other) noexcept {
      if (this != &other) {
        release();
        context_ = std::exchange(other.context_, nullptr);
        local_ = std::move(other.local_);
        other.local_.reset();
        remote_ = std::move(other.remote_);
        other.remote_.reset();
      }
      return *this;
    }
    DrmBackendLease(const DrmBackendLease &) = delete;
    DrmBackendLease &operator=(const DrmBackendLease &) = delete;
    ~DrmBackendLease() { release(); }

    /// @brief The leased local driver, or null when this lease is remote/absent.
    /// @details This is the file's ROUTING AUTHORITY, not merely a lifetime pin.
    /// The lease is an OPERATIONAL reference -- it keeps the backend open, not just
    /// allocated -- and it is fixed for the file's whole life. Routing through it
    /// means a file's GEM/VM work always reaches the backend that owns its page
    /// table, decided once at creation, instead of being re-derived on each call
    /// from a published pointer that teardown can change underneath it.
    [[nodiscard]] const std::shared_ptr<LinuxKfd> &local() const { return local_; }
    /// @brief The leased remote driver, or null when this lease is local/absent.
    [[nodiscard]] const std::shared_ptr<RemoteDriver> &remote() const { return remote_; }
    [[nodiscard]] bool is_remote() const { return remote_ != nullptr; }
    explicit operator bool() const { return local_ != nullptr || remote_ != nullptr; }

    /// @brief Drop the reference. MUST NOT run with an interposer lock held: the
    /// local path takes init_mutex_ and can complete a pending teardown.
    void release() {
      if (!context_)
        return;
      auto *context = std::exchange(context_, nullptr);
      const bool was_remote = remote_ != nullptr;
      local_.reset();
      remote_.reset();
      if (was_remote)
        context->release_remote_open();
      else
        context->release_local_open();
    }

  private:
    InterposerContext *context_;
    std::shared_ptr<LinuxKfd> local_;
    std::shared_ptr<RemoteDriver> remote_;
  };

  /// @brief Take an operational lease on whichever backend serves @p render_minor.
  /// @details The retain and the shutdown decision both run under init_mutex_ (local)
  /// or remote_mutex_ (remote), so a finalizer cannot retire the backend between the
  /// "who handles this node?" question and the reference that answers it.
  /// @returns An engaged lease, or a disengaged one if no backend can serve it.
  DrmBackendLease acquire_drm_backend_lease(uint32_t render_minor) {
    {
      std::lock_guard lock(init_mutex_);
      auto drv = driver();
      if (drv && drv->handles_drm_render_minor(render_minor) && drv->retain_local_open())
        return DrmBackendLease(this, std::move(drv));
    }
    {
      std::lock_guard lock(remote_mutex_);
      // Through the accessor, not the field: publish_mutex_ is the innermost lock
      // and every other reader goes this way, so a raw read here would be the one
      // access a reviewer has to re-derive the safety of.
      if (auto active_remote = remote()) {
        remote_open_refs_.fetch_add(1, std::memory_order_acq_rel);
        return DrmBackendLease(this, std::move(active_remote));
      }
    }
    return {};
  }

  struct DrmFileState {
    uint64_t id = 0;
    uint32_t render_minor = 128;
    /// @brief Operational reference to the backend this DRM file belongs to.
    /// @details A synthetic render fd owns GEM mappings whose teardown
    /// (reap_gem_for_drm_file -> gem_va_unmap) must reach the driver that installed
    /// the page-table entries, and DRM ioctls served from this file need that
    /// backend to still be OPEN, not merely allocated. The lease therefore holds a
    /// real open reference -- the SAME one a KFD descriptor holds -- so DRM files
    /// and KFD fds are counted by one mechanism and the shutdown policy needs no
    /// DRM-specific bookkeeping.
    DrmBackendLease backend_lease;
    // Guarded by InterposerContext::fd_mutex_. Reservations and tracked fds both
    // contribute one reference, so the state and its namespaces remain live while
    // an ioctl or duplication operation holds a token.
    size_t open_fds = 0;
    uint32_t next_syncobj_handle = 1;
    std::unordered_map<uint32_t, std::shared_ptr<SyncobjEntry>> syncobj_entries;
    // Timeline activity on one DRM file must not wake waiters belonging to every
    // other open DRM file in the process. Waiters sleep on this generation with
    // futex(2), which preserves EINTR instead of transparently restarting after a
    // signal as std::condition_variable does.
    std::atomic<uint32_t> syncobj_generation{0};
  };

  using DrmFileToken = std::shared_ptr<DrmFileState>;

  /// @brief Serialize synthetic DRM fd-table mutations with kernel fd operations.
  /// @details The lock may cover only metadata operations and calls through the
  /// real-libc passthrough table. Backend teardown, RPC, GEM cleanup, and any
  /// interposed libc entry point must run after this lock is released.
  ///
  /// LOCK ORDER: init_mutex_ must NEVER be acquired while this lock is held. The
  /// reverse edge exists and cannot be removed -- driver code runs under init_mutex_
  /// (open_local, release_local_open) and can reach the interposed close() hook,
  /// which takes this lock for every descriptor -- so acquiring them in this order
  /// anywhere would close an ABBA cycle between a dup and a close. That is why the
  /// dup paths reserve their backend, which takes init_mutex_, BEFORE this scope.
  std::unique_lock<std::mutex> lock_drm_fd_lifecycle() {
    return std::unique_lock(drm_fd_lifecycle_mutex_);
  }

  /// @brief What a final DRM-file release still owes, handed back to the caller.
  /// @details The GEM entries must be reaped while the backend is STILL open, and
  /// the lease must then be released with no interposer lock held (its local path
  /// takes init_mutex_ and can complete a pending teardown). Returning both makes
  /// that order structural instead of something each of the four release sites has
  /// to reproduce: reap `file_id`, then let `lease` destruct.
  struct DrmFinalRelease {
    std::optional<uint64_t> file_id;
    DrmBackendLease lease;
  };

  /// @brief Drop one fd or reservation reference while fd_mutex_ is held.
  /// @returns The identity and lease of the file when its final reference went.
  [[nodiscard]] DrmFinalRelease release_drm_file_locked(const DrmFileToken &state) {
    if (!state)
      return {};
    if (state->open_fds == 0) {
      util::Logger::warn("DRM file refcount underflow, id=", state->id);
      return {};
    }
    --state->open_fds;
    if (state->open_fds != 0)
      return {};
    return {state->id, std::move(state->backend_lease)};
  }

  struct DrmUntrackResult {
    bool tracked = false;
    DrmFinalRelease release;
  };

  /// @brief Finish a final DRM release: reap GEM, then drop the backend lease.
  /// @details Must run with NO interposer lock held.
  void complete_drm_release(DrmFinalRelease release) {
    if (release.file_id)
      reap_gem_for_drm_file(*release.file_id);
    // release.lease destructs here, dropping the open reference. For a local
    // backend that routes through release_local_open(), which completes a pending
    // process-exit shutdown exactly as the last KFD close would.
  }

  /// @brief Track a newly opened synthetic DRM fd.
  /// @returns The identity of a stale displaced DRM file whose final reference
  /// was removed, if any. The caller reaps it after releasing the lifecycle lock.
  /// @param lease Taken by reference and moved in LAST, after every operation that can
  /// throw. Owning it here across an allocation failure would destroy it during the
  /// unwind -- while the CALLER still holds the fd-lifecycle lock, since callee objects
  /// die first -- and releasing a lease reaches release_local_open() and so init_mutex_,
  /// which is the one order lock_drm_fd_lifecycle() forbids. Leaving it with the caller
  /// means a failure releases it at the caller's scope exit, after that lock is gone.
  [[nodiscard]] DrmFinalRelease track_drm(int fd, uint32_t render_minor, DrmBackendLease &lease) {
    auto state = std::make_shared<DrmFileState>();
    state->render_minor = render_minor;
    state->open_fds = 1;
    DrmFinalRelease displaced;
    {
      std::lock_guard lock(fd_mutex_);
      state->id = next_drm_file_id_++;
      if (auto stale = drm_fds_.find(fd); stale != drm_fds_.end()) {
        displaced = release_drm_file_locked(stale->second);
        state->backend_lease = std::move(lease); // noexcept, after the throwing work
        stale->second = std::move(state);
      } else {
        auto [it, inserted] = drm_fds_.emplace(fd, state); // may throw; lease not ours yet
        static_cast<void>(inserted);
        it->second->backend_lease = std::move(lease);
      }
    }
    return displaced;
  }

  /// @brief True if @p file's leased backend is a local SimulatedKfd.
  /// @details Routing question, so it consults the FILE's immutable lease rather
  /// than the mutable published driver. GEM/VM work belongs to the page table of the
  /// backend that owns the file.
  bool drm_file_is_simulated(const DrmFileToken &file) {
    if (!file)
      return false;
    const auto &local = file->backend_lease.local();
    return dynamic_cast<SimulatedKfd *>(local.get()) != nullptr;
  }

  /// @brief The SimulatedKfd serving @p file, or null.
  SimulatedKfd *drm_file_simulated(const DrmFileToken &file) {
    if (!file)
      return nullptr;
    return dynamic_cast<SimulatedKfd *>(file->backend_lease.local().get());
  }

  /// @brief Pin the DRM file currently tracked at @p fd.
  /// @details The fd-table lifecycle lock is required when this reservation must
  /// stay paired with a following kernel fd operation such as dup(). An ioctl may
  /// call this directly: fd_mutex_ makes the lookup and increment atomic with
  /// untracking, and the returned shared state remains valid if close races later.
  DrmFileToken reserve_drm_file(int fd) {
    std::lock_guard lock(fd_mutex_);
    auto it = drm_fds_.find(fd);
    if (it == drm_fds_.end())
      return nullptr;
    ++it->second->open_fds;
    return it->second;
  }

  /// @brief Release a token obtained from reserve_drm_file().
  /// @details Final GEM cleanup runs outside fd_mutex_ and any lifecycle lock.
  void release_drm_file_reservation(const DrmFileToken &state) {
    if (!state)
      return;
    DrmFinalRelease release;
    {
      std::lock_guard lock(fd_mutex_);
      release = release_drm_file_locked(state);
    }
    // A reservation CAN be the final reference (a racing close dropped the fd's own
    // reference first), so this must run the same completion as close(): reap, then
    // drop the lease, which re-evaluates a pending shutdown.
    complete_drm_release(std::move(release));
  }

  /// @brief Scoped ioctl reservation that preserves the ioctl's final errno.
  class DrmFileReservation {
  public:
    DrmFileReservation(InterposerContext &context, DrmFileToken file)
        : context_(&context), file_(std::move(file)) {}
    DrmFileReservation(const DrmFileReservation &) = delete;
    DrmFileReservation &operator=(const DrmFileReservation &) = delete;
    DrmFileReservation(DrmFileReservation &&other) noexcept
        : context_(std::exchange(other.context_, nullptr)), file_(std::move(other.file_)) {}
    DrmFileReservation &operator=(DrmFileReservation &&) = delete;
    ~DrmFileReservation() {
      if (!context_)
        return;
      const int saved_errno = errno;
      context_->release_drm_file_reservation(file_);
      errno = saved_errno;
    }

    [[nodiscard]] const DrmFileToken &file() const { return file_; }

  private:
    InterposerContext *context_;
    DrmFileToken file_;
  };

  /// @brief Capture a stable DRM-file identity for one ioctl invocation.
  [[nodiscard]] DrmFileReservation reserve_drm_file_for_ioctl(int fd) {
    return DrmFileReservation(*this, reserve_drm_file(fd));
  }

  /// @brief Reconcile DRM tracking after a successful descriptor duplication.
  /// @details Replaces stale target tracking even when @p state is null, so an
  /// ordinary duplicate cannot inherit a recycled synthetic DRM identity. A
  /// non-null state consumes the source reservation as the new fd's reference.
  /// @returns The identity of a displaced DRM file whose final reference was
  /// removed, if any. The caller performs GEM cleanup after releasing the
  /// lifecycle lock.
  [[nodiscard]] DrmFinalRelease commit_drm_dup(int fd, const DrmFileToken &state) {
    if (fd < 0)
      return {};
    DrmFinalRelease displaced_release;
    {
      std::lock_guard lock(fd_mutex_);
      if (auto stale = drm_fds_.find(fd); stale != drm_fds_.end()) {
        auto displaced = std::move(stale->second);
        if (state)
          stale->second = state;
        else
          drm_fds_.erase(stale);
        displaced_release = release_drm_file_locked(displaced);
      } else if (state) {
        drm_fds_.emplace(fd, state);
      }
    }
    return displaced_release;
  }

  bool is_drm(int fd) {
    std::lock_guard lock(fd_mutex_);
    return drm_fds_.count(fd) != 0;
  }

  static uint32_t drm_render_minor(const DrmFileToken &file) {
    return file ? file->render_minor : 128;
  }

  uint32_t drm_render_minor(int fd) {
    std::lock_guard lock(fd_mutex_);
    auto it = drm_fds_.find(fd);
    return (it != drm_fds_.end()) ? it->second->render_minor : 128;
  }

  /// @brief Remove @p fd from DRM tracking while preserving outstanding tokens.
  DrmUntrackResult untrack_drm(int fd) {
    std::lock_guard lock(fd_mutex_);
    auto it = drm_fds_.find(fd);
    if (it == drm_fds_.end())
      return {};
    auto state = std::move(it->second);
    drm_fds_.erase(it);
    return {.tracked = true, .release = release_drm_file_locked(state)};
  }

  /// @brief Allocate a syncobj handle in @p file's private namespace.
  int create_syncobj(const DrmFileToken &file, uint32_t flags, uint32_t *handle) {
    if (!handle || (flags & ~DRM_SYNCOBJ_CREATE_SIGNALED) != 0)
      return -EINVAL;
    if (!file)
      return -EBADF;
    std::lock_guard lock(fd_mutex_);
    auto &state = *file;
    if (state.syncobj_entries.size() == std::numeric_limits<uint32_t>::max())
      return -ENOSPC;
    uint32_t candidate = state.next_syncobj_handle++;
    while (candidate == 0 || state.syncobj_entries.count(candidate) != 0)
      candidate = state.next_syncobj_handle++;
    try {
      auto entry = std::make_shared<SyncobjEntry>();
      entry->has_fence = (flags & DRM_SYNCOBJ_CREATE_SIGNALED) != 0;
      state.syncobj_entries.emplace(candidate, std::move(entry));
    } catch (const std::bad_alloc &) {
      return -ENOMEM;
    } catch (const std::length_error &) {
      return -ENOMEM;
    }
    *handle = candidate;
    return 0;
  }

  /// @brief Remove a syncobj handle from @p file's private namespace.
  int destroy_syncobj(const DrmFileToken &file, uint32_t handle) {
    if (!file)
      return -EBADF;
    std::lock_guard lock(fd_mutex_);
    if (file->syncobj_entries.erase(handle) == 0)
      return -EINVAL;
    return 0;
  }

  /// @brief Copy a caller-owned array without faulting inside the interposer.
  template <typename T>
  static int snapshot_user_array(uint64_t address, uint32_t count, std::vector<T> &snapshot) {
    try {
      snapshot.resize(count);
    } catch (const std::bad_alloc &) {
      return -ENOMEM;
    } catch (const std::length_error &) {
      return -ENOMEM;
    }
    const size_t bytes = static_cast<size_t>(count) * sizeof(T);
    iovec local{snapshot.data(), bytes};
    iovec remote{reinterpret_cast<void *>(static_cast<uintptr_t>(address)), bytes};
    long copied;
    do {
      copied = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    } while (copied < 0 && errno == EINTR);
    return copied == static_cast<long>(bytes) ? 0 : -EFAULT;
  }

  /// @brief Wait for one or all requested points in a DRM-file namespace.
  /// @details Waiters use a per-file generation futex. Producers update the
  /// points under fd_mutex_, increment the generation with release ordering, and
  /// wake all sleepers; waiters recheck state under the same mutex after waking.
  int wait_syncobj_timeline(const DrmFileToken &file, drm_syncobj_timeline_wait *wait) {
    if (!wait)
      return -EINVAL;
    const drm_syncobj_timeline_wait request = *wait;
    constexpr uint32_t kSupportedFlags =
        DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT |
        DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_DEADLINE;
    if ((request.flags & ~kSupportedFlags) != 0)
      return -EINVAL;
    if (request.count_handles == 0)
      return 0;
    std::vector<uint32_t> handles;
    std::vector<uint64_t> points;
    if (int rc = snapshot_user_array(request.handles, request.count_handles, handles); rc != 0)
      return rc;
    if (int rc = snapshot_user_array(request.points, request.count_handles, points); rc != 0)
      return rc;
    if (!file)
      return -EBADF;
    std::vector<std::shared_ptr<SyncobjEntry>> entries;
    try {
      entries.reserve(request.count_handles);
    } catch (const std::bad_alloc &) {
      return -ENOMEM;
    } catch (const std::length_error &) {
      return -ENOMEM;
    }
    std::unique_lock lock(fd_mutex_);
    for (uint32_t i = 0; i < request.count_handles; ++i) {
      auto it = file->syncobj_entries.find(handles[i]);
      if (it == file->syncobj_entries.end())
        return -ENOENT;
      entries.push_back(it->second);
    }
    const bool wait_all = (request.flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL) != 0;
    const bool wait_for_submit = (request.flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT) != 0;
    const bool wait_available = (request.flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE) != 0;
    // The kernel validates fence availability once when the wait begins. A later
    // unrelated wakeup must not turn a valid pending wait into EINVAL.
    if (!wait_for_submit && !wait_available) {
      for (uint32_t i = 0; i < request.count_handles; ++i) {
        const auto &entry = *entries[i];
        const bool submitted =
            entry.has_fence && (points[i] == 0 || entry.submitted_point >= points[i]);
        if (!submitted)
          return -EINVAL;
      }
    }
    const auto evaluate = [&]() -> std::pair<int, uint32_t> {
      bool all_ready = true;
      bool any_ready = false;
      uint32_t first_ready = 0;
      for (uint32_t i = 0; i < request.count_handles; ++i) {
        const auto &entry = *entries[i];
        const bool submitted =
            entry.has_fence && (points[i] == 0 || entry.submitted_point >= points[i]);
        const bool ready = wait_available ? submitted
                                          : entry.has_fence && (points[i] == 0 ||
                                                                entry.signaled_point >= points[i]);
        all_ready = all_ready && ready;
        if (!wait_all && ready && !any_ready) {
          first_ready = i;
          any_ready = true;
        }
      }
      return {(wait_all ? all_ready : any_ready) ? 1 : 0, first_ready};
    };

    while (true) {
      auto [state, first_ready] = evaluate();
      if (state < 0)
        return state;
      if (state > 0) {
        wait->first_signaled = wait_all ? std::numeric_limits<uint32_t>::max() : first_ready;
        return 0;
      }

      timespec now{};
      if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return -errno;
      const int64_t now_ns = static_cast<int64_t>(now.tv_sec) * 1'000'000'000LL + now.tv_nsec;
      if (request.timeout_nsec <= now_ns)
        return -ETIME;
      const uint32_t generation = file->syncobj_generation.load(std::memory_order_relaxed);
      lock.unlock();
      const timespec deadline{static_cast<time_t>(request.timeout_nsec / 1'000'000'000LL),
                              static_cast<long>(request.timeout_nsec % 1'000'000'000LL)};
      // This user-space model cannot recover which signal interrupted futex(2),
      // so it cannot apply the kernel ioctl layer's SA_RESTART decision. Surface
      // EINTR consistently and let callers retry; the focused regression pins
      // that documented difference from the in-kernel DRM implementation.
      const long rc = futex_wait_until(file->syncobj_generation, generation, &deadline);
      const int wait_errno = errno;
      lock.lock();
      if (rc == 0 || wait_errno == EAGAIN)
        continue;
      if (wait_errno == ETIMEDOUT)
        return -ETIME;
      if (wait_errno == EINTR)
        return -EINTR;
      return -wait_errno;
    }
  }

  /// @brief Publish a completed binary or timeline fence under fd_mutex_.
  static void signal_syncobj_locked(const std::shared_ptr<SyncobjEntry> &entry, uint64_t point) {
    if (!entry)
      return;
    entry->has_fence = true;
    if (point == 0) {
      entry->submitted_point = 0;
      entry->signaled_point = 0;
      return;
    }
    entry->submitted_point = std::max(entry->submitted_point, point);
    entry->signaled_point = std::max(entry->signaled_point, point);
  }

  /// @brief Resolve one handle while fd_mutex_ is held.
  static std::shared_ptr<SyncobjEntry> lookup_syncobj_locked(const DrmFileToken &file,
                                                             uint32_t handle) {
    if (!file || handle == 0)
      return nullptr;
    auto entry = file->syncobj_entries.find(handle);
    return entry != file->syncobj_entries.end() ? entry->second : nullptr;
  }

  /// @brief Advance the per-file generation and wake every futex waiter.
  static void notify_syncobj_waiters(const DrmFileToken &file) {
    if (!file)
      return;
    file->syncobj_generation.fetch_add(1, std::memory_order_release);
    futex_wake_all(file->syncobj_generation);
  }

  bool create_local_vm(const std::string &config_path) {
    rj_vm_t *created_vm = nullptr;
    if (rj_vm_create(config_path.c_str(), RJ_VM_MODE_LOCAL, &created_vm) !=
        ROCJITSU_STATUS_SUCCESS) {
      util::Logger::debug_print("rocjitsu: failed to create VM from ", config_path);
      return false;
    }
    // The deleter IS the teardown sequence: stop the engine, join its thread, free
    // the VM. Nothing else may destroy a VM, so the order cannot be got wrong at a
    // call site and cannot be skipped on an unwind path. It runs when the last
    // reference goes away — which is why it must never run ON the engine thread
    // (it would join itself); the engine never takes a driver snapshot, and
    // assert_not_engine_thread() enforces that.
    local_vm_ = std::shared_ptr<rj_vm_t>(created_vm, [this, owner = getpid()](rj_vm_t *vm) {
      if (!vm)
        return;
      if (getpid() != owner) {
        // A forked child inherited this reference. The engine thread does not exist
        // here, so joining it would hang, and the VM belongs to the parent, so
        // destroying it would free state the parent still owns. Abandon it.
        //
        // A child never reaches this deleter under the fork-then-exec contract
        // (owner_pid_ gates every entry point), but the guard is kept because the
        // cost is one comparison and the failure mode -- a child joining a thread
        // that does not exist -- is an unrecoverable hang.
        return;
      }
      assert_not_engine_thread();
      if (local_vm_thread_) {
        rj_vm_request_exit(vm, "interposer VM destruction");
        local_vm_thread_->join();
        local_vm_thread_.reset();
      }
      rj_vm_destroy(vm);
    });
    // soc is validated alongside vm because rj_vm_run() rejects a null soc BEFORE
    // entering engine->run(); catching it here fails the open with ENODEV instead
    // of relying on the engine thread's readiness backstop.
    if (!local_vm_->vm || !local_vm_->soc || !local_vm_->vm->driver() ||
        local_vm_->vm->driver()->fd() < 0) {
      util::Logger::debug_print("rocjitsu: local VM did not acquire a KFD open");
      local_vm_.reset();
      return false;
    }

    std::string config_json = read_file_passthrough(config_path.c_str());
    if (rj_vm_load_plugins(local_vm_.get(), config_json.c_str(), nullptr) !=
        ROCJITSU_STATUS_SUCCESS) {
      util::Logger::debug_print("rocjitsu: failed to configure execution plugins");
      local_vm_.reset();
      return false;
    }
    return true;
  }

  /// @brief Abort if called on the local VM engine thread.
  /// @details The VM deleter joins that thread, so running it there would deadlock
  /// on self-join. This holds because the engine never takes a driver snapshot; the
  /// assert is what keeps it true as the engine grows.
  void assert_not_engine_thread() const {
    assert((!local_vm_thread_ || local_vm_thread_->get_id() != std::this_thread::get_id()) &&
           "local VM teardown must not run on the engine thread (its deleter joins it)");
  }

  /// @brief Publish a fully-ready local driver. Caller holds init_mutex_.
  /// @details The sole site that stores a non-null active_driver_. Only call after
  /// the engine has started successfully (see get_or_create) so no reader can
  /// observe a half-started device.
  void publish_local_driver_locked(std::shared_ptr<LinuxKfd> driver) {
    // Capture the driver's own internal references (the local VM's bootstrap open
    // for SimulatedKfd; none for GuestKfd) as the "idle" watermark. Nothing
    // app-facing can exist yet: the driver is not published, so no interposed path
    // can have reached it. See local_driver_is_idle_locked().
    local_open_ref_baseline_ = driver ? driver->local_open_ref_count() : 0;
    publish_driver(std::move(driver));
  }

  /// @brief The published SimulatedKfd as an ALIASING snapshot on local_vm_.
  /// @details Shares the VM's control block while pointing at the driver the VM
  /// owns by unique_ptr, so the driver is kept alive by, and cannot outlive, its
  /// VM — without VirtualMachine having to change its ownership at all.
  std::shared_ptr<LinuxKfd> simulated_driver_alias() const {
    return std::shared_ptr<LinuxKfd>(local_vm_, local_vm_->vm->driver());
  }

  /// @brief The owners a local-VM teardown drops, moved out so they can be destroyed
  /// OUTSIDE init_mutex_.
  /// @details Their destruction runs the deleters: the guest's first, because it
  /// captures local_vm_, then the VM's engine stop/join/destroy. None of that may
  /// happen under init_mutex_, which the fd paths wait on. Ordering does not come
  /// from the declaration order of these members -- it comes from the guest deleter
  /// holding its own reference to the VM.
  struct DoomedLocalVm {
    std::shared_ptr<GuestKfd> guest;
    std::shared_ptr<rj_vm_t> vm;
  };

  /// @brief Release the interposer's own references to the local backend.
  /// @details Caller holds init_mutex_. This does NOT free anything directly: it
  /// unpublishes and drops our references, and whoever holds the last one runs the
  /// deleters (guest first, because its deleter captures local_vm_, then the VM's
  /// engine stop/join/destroy). A reader still inside a call keeps its snapshot
  /// alive and frees on its way out, so there is no window in which an in-flight
  /// dereference can touch freed state.
  void destroy_local_vm() {
    publish_driver(nullptr);
    guest_driver_.reset();
    local_vm_.reset();
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
    assert(local_vm_ != nullptr);
    assert(!local_vm_thread_);
    try {
      // make_unique/std::thread can throw system_error (resource exhaustion) or
      // bad_alloc; treat any failure to launch as a start failure so the caller
      // unwinds and never publishes a device with no engine behind it.
      // Latch readiness from the THREAD, not just from inside engine->run(): the
      // wait below happens under init_mutex_, so any rj_vm_run() path that returns
      // without reaching run() (e.g. its own argument validation) would otherwise
      // strand this wait forever AND hold init_mutex_ forever — deadlocking every
      // later interposed open/close and the DSO finalizer itself. The engine's own
      // in-band latch wins; this only fires if nothing latched.
      local_vm_thread_ = std::make_unique<std::thread>([vm = local_vm_.get()]() {
        rj_vm_run(vm, nullptr);
        vm->engine->latch_startup_if_unlatched(/*failed=*/true);
      });
    } catch (const std::exception &e) {
      util::Logger::debug_print("rocjitsu: failed to start local VM engine thread: ", e.what());
      return false;
    }
    if (!local_vm_->engine->wait_until_started()) {
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
  /// The local driver is created at most once per process and is never recreated
  /// after a shutdown request (see get_or_create), so `owner` can never come to
  /// point at a DIFFERENT driver that reused the address.
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
    uint64_t drm_file_id = 0;    ///< Owning DRM file; the entry is reaped on its last close.
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
  uint32_t prime_import(int dmabuf_fd, const DrmFileToken &drm_file, uint64_t size) {
    std::lock_guard lock(fd_mutex_);
    if (!drm_file)
      return 0;
    if (gem_entries_.size() == std::numeric_limits<uint32_t>::max())
      return 0;
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
    gem.drm_file_id = drm_file->id;
    gem.size = size;
    gem.alloc_flags = alloc_flags;
    return handle;
  }

  /// @brief Install (or replace) a GEM_VA range in the GPU page table for @p handle.
  /// @details Runs entirely under fd_mutex_ and performs BOTH the bookkeeping AND
  /// the page-table install (drv->gem_va_map) and output timeline signal atomically,
  /// so a concurrent GEM_CLOSE
  /// (untrack_gem, also under fd_mutex_) can never interleave between recording the
  /// range and installing the PTEs — which would otherwise leave PTEs pointing into
  /// a munmapped cpu_ptr with no entry left to tear them down. It lazily mmaps the
  /// dmabuf fd's backing pages the first time (the fd is still open at GEM_VA time),
  /// bounds-checks the request against the BO size, records the range, installs
  /// the PTEs, and publishes the output timeline point. The lock order
  /// fd_mutex_ -> driver page-table lock matches
  /// teardown_gem_entry_locked, so there is no inversion.
  /// @param replace When true (AMDGPU_VA_OP_REPLACE), reject overlap with another
  ///   DRM file before evicting an existing range in the calling DRM-file namespace.
  ///   The namespaces share one simulated GPU page table, so neither operation may
  ///   overwrite another file's PTEs. When false (AMDGPU_VA_OP_MAP), any overlapping
  ///   pre-existing range is a conflict.
  /// @param publish_timeline Whether this update publishes its output timeline.
  ///   AMDGPU_VM_DELAY_UPDATE suppresses publication, matching the kernel contract.
  /// @returns Zero when the range was installed, `-ENOENT` for an unknown handle in
  ///   this DRM-file namespace, or another negative errno for invalid requests.
  [[nodiscard]] int gem_map(const DrmFileToken &file, uint32_t handle, uint64_t va_address,
                            uint64_t offset_in_bo, uint64_t map_size, bool replace,
                            bool publish_timeline = true, uint32_t timeline_handle = 0,
                            uint64_t timeline_point = 0) {
    std::unique_lock lock(fd_mutex_);
    if (!file)
      return -EBADF;
    auto timeline = lookup_syncobj_locked(file, timeline_handle);
    if (timeline_handle != 0 && !timeline)
      return -ENOENT;
    auto *drv = drm_file_simulated(file);
    if (!drv)
      return -ENODEV;
    auto it = gem_entries_.find(handle);
    if (it == gem_entries_.end() || it->second.drm_file_id != file->id)
      return -ENOENT;
    GemEntry &gem = it->second;
    // Bound the request within the BO without letting offset_in_bo + map_size
    // overflow (both are caller-controlled __u64 from the UAPI struct): a wrap
    // would defeat a naive sum-vs-size check and install PTEs pointing outside
    // the mmap. Reject a zero-size BO, a zero-size map, and any range past the end.
    if (gem.size == 0 || map_size == 0 || offset_in_bo > gem.size ||
        map_size > gem.size - offset_in_bo)
      return -EINVAL;
    const GemMapping range{va_address, map_size};
    // Handle the target VA range's current occupant. Independent DRM files have
    // private handle namespaces but currently share one simulated GPU page table,
    // so a foreign mapping must never be overwritten. REPLACE may evict only a
    // current holder from this DRM file; plain MAP rejects any current holder.
    const bool foreign_overlap = range_is_mapped_by_other_drm_file_locked(file->id, range);
    if (foreign_overlap) {
      // TODO: Give each independent DRM file its own simulated GPUVM. Until then,
      // permitting this overlap would overwrite the single shared page table and
      // let either file tear down the other's PTEs. Make the model limitation
      // visible instead of silently accepting or rejecting the update.
      util::Logger::warn("DRM files currently share one simulated GPUVM; rejecting overlapping "
                         "GEM_VA range at address ",
                         va_address);
      return -EINVAL;
    }
    if (replace) {
      if (!evict_range_locked(drv, file->id, range, /*allow_missing=*/true))
        return -EINVAL;
    } else if (range_is_mapped_locked(range)) {
      return -EINVAL;
    }
    if (!gem.cpu_ptr) {
      void *p = InterposerContext::real().mmap(nullptr, gem.size, PROT_READ | PROT_WRITE,
                                               MAP_SHARED, gem.dmabuf_fd, 0);
      if (p == MAP_FAILED)
        return -EINVAL;
      gem.cpu_ptr = p;
    }
    // Record which SimulatedKfd's page table receives these PTEs so GEM_CLOSE (or a
    // DRM-file-close reap) unmaps through the driver that installed them, never a
    // replacement one. Set the owner on the first mapping and keep it: all ranges of
    // one BO install into the same driver's page table. The local driver is created
    // at most once per process and never recreated after a shutdown request,
    // so the owner can never come to name a different driver; teardown additionally
    // compares it against the live driver before unmapping.
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
      return -EINVAL;
    gem.installed_vas.push_back(range);
    if (publish_timeline)
      signal_syncobj_locked(timeline, timeline_point);
    lock.unlock();
    if (publish_timeline && timeline)
      notify_syncobj_waiters(file);
    return 0;
  }

  /// @brief Remove a GEM_VA range from the GPU page table for @p handle (UNMAP).
  /// @details Performs the page-table unmap AND the bookkeeping erase atomically
  /// under fd_mutex_ (same rationale as gem_map). Validates that @p handle actually
  /// owns the exact {va_address, map_size} range before mutating the page table, so
  /// an UNMAP with a wrong handle or range cannot tear down PTEs the handle does not
  /// own and cannot report success for a no-op.
  /// @returns Zero when the range was unmapped, `-ENOENT` for an unknown handle in
  ///   this DRM-file namespace, or another negative errno for invalid requests.
  [[nodiscard]] int gem_unmap(const DrmFileToken &file, uint32_t handle, uint64_t va_address,
                              uint64_t map_size, bool publish_timeline = true,
                              uint32_t timeline_handle = 0, uint64_t timeline_point = 0) {
    std::unique_lock lock(fd_mutex_);
    if (!file)
      return -EBADF;
    auto timeline = lookup_syncobj_locked(file, timeline_handle);
    if (timeline_handle != 0 && !timeline)
      return -ENOENT;
    auto *drv = drm_file_simulated(file);
    if (!drv)
      return -ENODEV;
    auto it = gem_entries_.find(handle);
    if (it == gem_entries_.end() || it->second.drm_file_id != file->id)
      return -ENOENT;
    GemEntry &gem = it->second;
    const GemMapping range{va_address, map_size};
    if (std::find(gem.installed_vas.begin(), gem.installed_vas.end(), range) ==
        gem.installed_vas.end())
      return -EINVAL; // This handle does not own the exact range — do not touch PTEs.
    if (!drv->gem_va_unmap(va_address, map_size))
      return -EINVAL;
    std::erase(gem.installed_vas, range);
    if (publish_timeline)
      signal_syncobj_locked(timeline, timeline_point);
    lock.unlock();
    if (publish_timeline && timeline)
      notify_syncobj_waiters(file);
    return 0;
  }

  /// @brief Clear a GEM_VA range from the GPU page table (CLEAR).
  /// @details CLEAR is handle-agnostic within one DRM file: it tears down every
  /// recorded range that OVERLAPS {va_address, map_size} in the calling file's
  /// namespace, updating the owning entries' bookkeeping so a later GEM_CLOSE does
  /// not double-unmap them.
  /// Fails if no recorded range overlaps (so the ioctl reports EINVAL instead of a
  /// phantom clear).
  /// @returns Zero when an overlapping range in the calling DRM-file namespace was
  ///   cleared, or a negative errno when no such range exists or the request fails.
  [[nodiscard]] int gem_clear(const DrmFileToken &file, uint64_t va_address, uint64_t map_size,
                              bool publish_timeline = true, uint32_t timeline_handle = 0,
                              uint64_t timeline_point = 0) {
    std::unique_lock lock(fd_mutex_);
    if (!file)
      return -EBADF;
    auto timeline = lookup_syncobj_locked(file, timeline_handle);
    if (timeline_handle != 0 && !timeline)
      return -ENOENT;
    auto *drv = drm_file_simulated(file);
    if (!drv)
      return -ENODEV;
    if (!evict_range_locked(drv, file->id, GemMapping{va_address, map_size},
                            /*allow_missing=*/false))
      return -EINVAL;
    if (publish_timeline)
      signal_syncobj_locked(timeline, timeline_point);
    lock.unlock();
    if (publish_timeline && timeline)
      notify_syncobj_waiters(file);
    return 0;
  }

  /// @brief Reap every GEM handle owned by a closing DRM file (at its close()).
  /// @details A well-behaved caller GEM_CLOSEs each handle, but a crash or leak can
  /// leave handles live; the DRM file close is their backstop, mirroring the kernel
  /// dropping a drm_file's GEM objects. Tears each entry's PTEs + host mmap down
  /// under fd_mutex_ before erasing, so no state escapes the lock.
  void reap_gem_for_drm_file(uint64_t drm_file_id) {
    std::lock_guard lock(fd_mutex_);
    for (auto it = gem_entries_.begin(); it != gem_entries_.end();) {
      if (it->second.drm_file_id == drm_file_id) {
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
  int untrack_gem(const DrmFileToken &file, uint32_t handle) {
    std::lock_guard lock(fd_mutex_);
    if (!file)
      return -EBADF;
    auto it = gem_entries_.find(handle);
    if (it == gem_entries_.end() || it->second.drm_file_id != file->id)
      return -ENOENT;
    teardown_gem_entry_locked(it->second);
    gem_entries_.erase(it);
    return 0;
  }

  std::shared_ptr<LinuxKfd> get_or_create() {
    std::lock_guard lock(init_mutex_);
    return get_or_create_locked();
  }

  /// @brief get_or_create() with init_mutex_ already held by the caller.
  /// @details Split out so open_local() can hold one lock across creation AND the
  /// application's open(), which is what keeps the shutdown policy from retiring a
  /// backend in between.
  std::shared_ptr<LinuxKfd> get_or_create_locked() {
    if (driver() == nullptr) {
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
      ConstructionScope construction_scope;
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
      std::optional<rocjitsu::config::DbtRuntimeConfigHandoff> handoff;
      std::string tried_last;
      for (const auto &candidate : cfg_candidates) {
        if (candidate == tried_last)
          continue; // Skip a duplicate tier (e.g. env unset collapses 1 and 2).
        tried_last = candidate;
        handoff = child_config_handoff(candidate);
        if (handoff)
          break;
      }
      if (!handoff) {
        util::Logger::debug_print("rocjitsu: no child config path");
        return nullptr;
      }

      try {
        auto dbt_guest = rocjitsu::config::load_dbt_guest_config_from_handoff(*handoff);
        if (dbt_guest.enabled) {
          LinuxKfd *execution_driver = nullptr;
          const bool simulator_backend =
              dbt_guest.host.backend == rocjitsu::config::DbtExecutionBackend::Simulator;
          if (simulator_backend) {
            const std::string host_config_path = rocjitsu::config::resolve_dbt_host_config_path(
                handoff->config_path, dbt_guest.host.simulator_config_path);
            if (!create_local_vm(host_config_path)) {
              return nullptr;
            }
            execution_driver = local_vm_->vm->driver();
            rocjitsu::config::validate_dbt_simulator_device_limits(dbt_guest,
                                                                   local_vm_->loaded.device);
          }

          // The deleter captures local_vm_, so the guest is destroyed BEFORE the VM
          // whose bootstrap open it holds. That ordering is now a property of the
          // object graph rather than a rule two unwind paths have to follow.
          auto guest_driver =
              std::shared_ptr<GuestKfd>(new GuestKfd(std::move(dbt_guest), execution_driver),
                                        [vm = local_vm_](GuestKfd *g) { delete g; });
          if (!guest_driver->prepare_for_discovery()) {
            guest_driver.reset();
            destroy_local_vm();
            return nullptr;
          }
          auto driver = std::static_pointer_cast<LinuxKfd>(guest_driver);
          guest_driver_ = std::move(guest_driver);
          // Start the engine and wait for readiness BEFORE publishing, so no
          // concurrent interposed call can observe a driver whose components are
          // still starting (or whose startup will fail). On failure nothing was
          // published; destroy_local_vm() unpublishes (no-op) + frees, and
          // guest_driver_ is reset first because it owns the simulator bootstrap
          // open.
          if (simulator_backend && !start_local_vm()) {
            driver.reset();
            destroy_local_vm();
            return nullptr;
          }
          publish_local_driver_locked(driver);
          return driver;
        }

        if (!create_local_vm(handoff->config_path)) {
          return nullptr;
        }
      } catch (const std::exception &e) {
        // This is where a broken runtime handoff lands, including an enabled DBT config
        // with no resolved host gpu_id. Failing closed leaves the process with a null
        // driver and an opaque downstream failure, so the reason must be audible in a
        // default build the way the hook layer's equivalent refusal already is.
        util::Logger::warn("rocjitsu: failed to load child config: ", e.what());
        destroy_local_vm();
        return nullptr;
      }

      // Start the engine and wait for component startup to SUCCEED before
      // publishing active_driver_. Publishing earlier would let a concurrent
      // interposed call observe a half-started (or about-to-fail) device — the
      // race this branch removes. On failure nothing was published; destroy_local_vm
      // frees the VM.
      if (!start_local_vm()) {
        destroy_local_vm();
        return nullptr;
      }
      // The release-store pairs with acquire loads in driver()/initialized() so any
      // reader that observes the driver also observes all setup above.
      publish_local_driver_locked(simulated_driver_alias());
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
  /// @brief PID of the process that constructed this context. Set once in init()
  /// and NEVER mutated, including in a forked child: a child must be able to detect
  /// that it does not own this state, and a writable marker would be a mutation of
  /// the parent's memory in the vfork window.
  pid_t owner_pid_ = 0;
  /// @brief st_rdev of the host's real /dev/kfd, or 0 if none exists.
  /// @details Immutable after init(). Used only by the forked-child classifier.
  dev_t host_kfd_rdev_ = 0;

  /// @brief True if the published local driver holds no app-facing KFD reference.
  /// @details Caller holds init_mutex_. This is a POLICY question (has the
  /// application finished with the device?), deliberately separate from the
  /// lifetime question (is anyone mid-call?), which active_driver_'s shared_ptr
  /// answers on its own. "Idle" means the driver's reference count is back to the
  /// baseline captured at publication, i.e. every reference the APPLICATION added
  /// is gone. Comparing against a captured baseline rather than a per-class
  /// constant keeps this backend-agnostic and keeps the VM-level fact "the local VM
  /// holds one bootstrap open" out of SimulatedKfd. A null or non-ref-tracking
  /// driver reports 0 and is idle.
  bool local_driver_is_idle_locked() {
    auto active_driver = driver();
    if (!active_driver)
      return true;
    // Synthetic DRM files are counted here too: each holds a real open reference on
    // the backend (DrmBackendLease), the same one a KFD descriptor holds, so this
    // single comparison covers both and needs no DRM-specific term.
    return active_driver->local_open_ref_count() <= local_open_ref_baseline_;
  }

  /// @brief Tear the local backend down iff a shutdown was requested and no
  /// app-facing KFD reference remains. Caller holds init_mutex_.
  /// @details Two steps, both under init_mutex_, and that is the whole algorithm:
  ///
  ///   1. If not idle, return. The DSO finalizer reaches here while HIP/ROCr may
  ///      still hold KFD descriptors; their later closes call back in through
  ///      release_local_open() and complete the request then.
  ///   2. If idle, release any parked blocking call (begin_local_shutdown) so its
  ///      caller returns and drops its driver snapshot, then unpublish and drop our
  ///      references. Whoever holds the last snapshot runs the deleters.
  ///
  /// There is no post-decision re-check and nothing to roll back. retain_local_open()
  /// also runs under init_mutex_, so a racing dup either completes before this call
  /// (and the idle check sees its reference) or after it (and finds no published
  /// driver, failing closed). The decision is made once, under one lock.
  [[nodiscard]] DoomedLocalVm shutdown_local_vm_if_idle_locked() {
    DoomedLocalVm doomed;
    // Hardware DBT mode leaves local_vm_ == nullptr but still publishes
    // guest_driver_, so a guest-only backend must not be skipped here.
    if (!local_vm_ && !guest_driver_)
      return doomed;
    if (!local_driver_is_idle_locked())
      return doomed;

    // Release any parked blocking call so its caller returns and drops the driver
    // snapshot that would otherwise keep the object alive past this point. Only the
    // driver can do this; closing an fd does not cancel an in-flight kernel wait.
    if (auto active_driver = driver())
      active_driver->begin_local_shutdown();

    // Unpublish here, but hand the OWNERS back so they die after the caller drops
    // init_mutex_. Destroying them runs the VM deleter -- engine exit, thread join,
    // rj_vm_destroy -- and doing that under init_mutex_ makes every dup() of a
    // tracked KFD fd wait out the whole teardown while itself holding
    // drm_fd_lifecycle_mutex_, which blocks close() process-wide. Recreation in the
    // gap is not a concern: both callers set shutdown_requested_ under this same
    // lock first, and creation is refused while it is set.
    publish_driver(nullptr);
    doomed.guest = std::move(guest_driver_);
    doomed.vm = std::move(local_vm_);
    guest_driver_.reset();
    local_vm_.reset();
    return doomed;
  }

  /// @brief Owner of the local VM, or null when no local VM exists.
  /// @details A shared_ptr whose DELETER performs the entire engine teardown:
  /// rj_vm_request_exit(), join the engine thread, rj_vm_destroy(). Teardown is
  /// therefore not a sequence anyone has to remember to run in order — it is what
  /// happens when the last reference goes away.
  std::shared_ptr<rj_vm_t> local_vm_;
  /// @brief Owner of the local VM engine loop; null when no local VM is running.
  /// @details Deliberately unique_ptr<std::thread>, NOT std::jthread: a jthread
  /// destructor always request_stop()+join()s, which would deadlock in the fork
  /// child (the thread does not exist there, so join hangs). A raw-owned
  /// std::thread keeps that abandonment expressible. Do not switch this to
  /// jthread.
  std::unique_ptr<std::thread> local_vm_thread_;
  /// @brief Set once the interposer DSO finalizer has requested process-exit
  /// shutdown; the VM is torn down as soon as it is also idle.
  bool shutdown_requested_ = false;
  /// @brief Reference count the published driver had before any application
  /// reference existed. Written by publish_local_driver_locked(), read by
  /// local_driver_is_idle_locked(); both under init_mutex_.
  uint32_t local_open_ref_baseline_ = 0;
  /// @brief Owner of the hardware/guest DBT driver, or null when unused.
  /// @details Its deleter captures local_vm_, so the guest is destroyed BEFORE the
  /// VM it borrows the simulator bootstrap open from. That ordering is expressed by
  /// the capture rather than by a comment two call sites have to honour.
  std::shared_ptr<GuestKfd> guest_driver_;
  /// @brief The published local driver, or null when none is active.
  /// @details Published under publish_mutex_, exactly like remote_ below, so a
  /// reader simply takes a lifetime-extending snapshot: whoever holds one keeps the
  /// object alive,
  /// and the last release destroys it. That removes the need for any lifetime lock
  /// on this pointer, and with it the whole class of "did every dereference remember
  /// to take the pin, exactly once, and never above init_mutex_?" bugs.
  ///
  /// For the simulator this is an ALIASING shared_ptr built on local_vm_: it shares
  /// the VM's control block but points at vm->driver(), so the SimulatedKfd cannot
  /// outlive, and does not need to be separately owned by, the VM that contains it.
  /// VirtualMachine keeps its plain unique_ptr and is untouched. For DBT the guest
  /// driver is published directly.
  std::shared_ptr<LinuxKfd> active_driver_;
  /// @brief Active daemon-mode remote driver, or nullptr in local mode.
  /// @details Read and written ONLY through `remote()` / `publish_remote()`, which
  /// hold `publish_mutex_`. Readers (`remote()`, `remote_lookup()`, `initialized()`,
  /// the AMDKFD ioctl fallback, the mmap path) each take a lifetime-extending
  /// snapshot: a racing `teardown_remote()` that publishes nullptr cannot free the
  /// object while another thread still holds a snapshot in
  /// `remote->ioctl()`/`remote->mmap()`. The object is destroyed by the last
  /// shared_ptr, not by a manual delete, so there is no use-after-free.
  ///
  /// `remote_mutex_` is the OUTER lock: both publishers (`get_or_create_remote()`
  /// and `teardown_remote_locked()`) hold it across the whole create+open or
  /// clear sequence, so a reader that holds it -- as
  /// `acquire_drm_backend_lease()` does -- is serialized against every write, not
  /// merely against a torn pointer.
  std::shared_ptr<RemoteDriver> remote_;
  /// @brief Leaf mutex guarding the two published backend pointers.
  /// @details Deliberately a mutex WE own rather than std::atomic<std::shared_ptr>.
  /// libstdc++ implements that with a process-wide pool of spinlocks keyed by
  /// address. Owning the lock keeps the publication path's synchronization visible
  /// and auditable here rather than hidden in a library-internal pool. Innermost in
  /// the lock order, held only for a pointer copy.
  mutable std::mutex publish_mutex_;
  std::atomic<int> remote_kfd_fd_{-1};
  /// @brief Open-reference count for the remote (daemon-mode) KFD connection.
  /// @details The primary remote fd and every dup of it each hold one
  /// reference; the RPC connection is torn down only when the last reference is
  /// released. Mirrors SimulatedKfd's local open refcount for daemon mode.
  std::atomic<int> remote_open_refs_{0};

  std::mutex init_mutex_;
  std::mutex fd_mutex_;
  /// Serializes synthetic DRM fd-table operations with tracking updates. The
  /// kernel makes close and duplication atomic with respect to the process fd
  /// table; this lock extends that atomicity through the interposer's drm_fds_
  /// bookkeeping so a recycled integer fd cannot acquire an older DRM file's
  /// namespaces. Its scope is deliberately limited to source lookup plus the
  /// kernel fd operation and DRM tracking update. Backend teardown, RPC, GEM
  /// cleanup, and other interposed libc calls must run after it is released.
  std::mutex drm_fd_lifecycle_mutex_;
  std::mutex remote_mutex_;
  std::unordered_map<int, std::string> sysfs_fds_;
  std::unordered_map<int, DrmFileToken> drm_fds_;
  uint64_t next_drm_file_id_ = 1;
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
  /// closes (reap_gem_for_drm_file). Because handles are not recycled with fd numbers,
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
    if (gem.owner && gem.owner == dynamic_cast<SimulatedKfd *>(driver().get())) {
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

  /// @brief Evict every recorded range that OVERLAPS @p range in one DRM file.
  /// @details Caller holds fd_mutex_ and passes the live simulated @p drv and stable
  /// DRM-file identity. Used by GEM_VA REPLACE (evict prior mappings before installing
  /// the new one) and CLEAR (handle-agnostic teardown). Matching is by interval
  /// intersection, not exact equality: a REPLACE at a VA previously mapped with a
  /// DIFFERENT size (or a sub/super-range) must still evict the old mapping, otherwise
  /// its stale bookkeeping would later double-unmap or leak the new PTEs. Each
  /// overlapping range is unmapped by its OWN {va_address, map_size} extent (not @p
  /// range's) so the page-table removal matches what was installed, then dropped from
  /// its entry's bookkeeping. The host mmap is left intact — the owning handle still
  /// exists and other ranges may reference it; it is munmapped only at GEM_CLOSE /
  /// reap.
  /// @param allow_missing When true, no overlap is a success (a MAP onto a free VA has
  ///   nothing to evict); when false (REPLACE/CLEAR), no overlap is a failure so the
  ///   ioctl reports EINVAL.
  /// @retval true nothing overlapped (allow_missing) or all overlaps were evicted.
  /// @retval false nothing overlapped (only when !allow_missing) or an unmap failed.
  /// @note Not rolled back on a mid-loop unmap failure: already-evicted ranges stay
  ///   evicted. This is safe because gem_va_unmap() only fails when the local process
  ///   has already vanished (SimulatedKfd::gem_va_unmap), i.e. its page table is being
  ///   torn down anyway, so a partially-evicted state is never observed by a live GPU.
  [[nodiscard]] bool evict_range_locked(SimulatedKfd *drv, uint64_t drm_file_id,
                                        const GemMapping &range, bool allow_missing) {
    bool evicted_any = false;
    for (auto &[handle, gem] : gem_entries_) {
      if (gem.drm_file_id != drm_file_id)
        continue;
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

  /// @brief Whether any GEM entry owns a range OVERLAPPING @p range in the shared
  /// simulated GPU page table. Caller holds fd_mutex_. Used by plain MAP to reject
  /// a map that would collide with an existing range's PTEs, regardless of which
  /// DRM file owns it.
  [[nodiscard]] bool range_is_mapped_locked(const GemMapping &range) const {
    for (const auto &[handle, gem] : gem_entries_) {
      for (const auto &existing : gem.installed_vas)
        if (existing.overlaps(range))
          return true;
    }
    return false;
  }

  /// @brief Whether a GEM entry owned by a DRM file other than @p drm_file_id
  /// overlaps @p range in the shared simulated GPU page table. Caller holds
  /// fd_mutex_. Used by REPLACE to reject a foreign collision before evicting any
  /// mapping owned by the caller.
  [[nodiscard]] bool range_is_mapped_by_other_drm_file_locked(uint64_t drm_file_id,
                                                              const GemMapping &range) const {
    for (const auto &[handle, gem] : gem_entries_) {
      if (gem.drm_file_id == drm_file_id)
        continue;
      for (const auto &existing : gem.installed_vas)
        if (existing.overlaps(range))
          return true;
    }
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

namespace {
/// @brief Run a bare mapping syscall with the layout held still.
///
/// @details An emulated atomic holds a raw pointer across a writability check
/// and the modify, so the layout must not change under it. Only the bare
/// syscall is covered: the driver's own mapping paths re-enter the memory model
/// and take its VMID lock, and the emulated atomic reaches this lock while
/// already holding that one, so widening the region would invert them.
/// @pre The caller owns the interposer state. A forked child must not reach
/// this: the lock is inherited, its holder may not have survived the fork, and
/// the process contract forbids touching inherited state before exec.
template <typename F> auto with_host_mapping_held(F &&syscall) {
  auto lock = rocjitsu::host_mapping_lock().lock_exclusive();
  return syscall();
}
} // namespace

extern "C" {

static std::string redirect_sysfs_path(const char *path);
static std::string redirect_sys_dev_char(const char *path);
static std::optional<Sysfs::GpuInfo> interposer_gpu_info(uint32_t render_minor);
static std::optional<Sysfs::GpuInfo>
interposer_gpu_info_for(const InterposerContext::DrmFileToken &file);

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

  // Snapshot the driver for the driver_fd()/drm_path()/handles_drm_render_minor()
  // reads below.
  uint32_t render_minor = 0;
  {
    if (InterposerContext::ctx.driver_fd() < 0 && !remote)
      return {};

    std::string drm_base;
    if (auto drv = InterposerContext::ctx.driver())
      drm_base = drv->drm_path();
    else
      drm_base = InterposerContext::ctx.remote_drm_path();

    // HIP/libdrm may open the generated dev_dri node after following redirected
    // sysfs metadata instead of opening the literal /dev/dri/renderD* path.
    // Treat both path forms as the same synthetic DRM render node.
    if (!render_minor_from_drm_node_path(path, drm_base.c_str(), &render_minor))
      return {};
  }

  // Acquire the operational lease BEFORE creating the fd. The retain and the
  // shutdown decision share init_mutex_, so the backend cannot be retired between
  // "who serves this node?" and the reference that answers it — the window the raw
  // snapshot left open.
  auto backend_lease = InterposerContext::ctx.acquire_drm_backend_lease(render_minor);
  if (!backend_lease)
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

  InterposerContext::DrmFinalRelease displaced;
  try {
    auto drm_lifecycle = InterposerContext::ctx.lock_drm_fd_lifecycle();
    // backend_lease stays OURS across this call: on a throw it is released at this
    // function's scope exit, by which point drm_lifecycle is gone. See track_drm().
    displaced = InterposerContext::ctx.track_drm(high_fd, render_minor, backend_lease);
  } catch (const std::exception &) {
    // Not just bad_alloc: unique_lock's constructor can throw system_error, and this
    // is an extern "C" entry point -- letting anything escape into a C caller frame
    // is undefined.
    const int saved_errno = ENOMEM;
    InterposerContext::real().close(high_fd);
    errno = saved_errno;
    return {true, -1};
  }
  InterposerContext::ctx.complete_drm_release(std::move(displaced));
  return {true, high_fd};
}

/// @brief True if @p st describes a device rocJITsu emulates.
/// @details Pure identity: the host KFD's recorded st_rdev, or DRM's fixed major.
/// No pathname inspection at all -- a name can lie (a chained alias, a dirfd-relative
/// spelling, or an unrelated node whose name merely contains "kfd"), a dev_t cannot.
[[nodiscard]] inline bool rj_is_gpu_device_stat(const struct stat &st) {
  if (!S_ISCHR(st.st_mode))
    return false;
  constexpr unsigned kDrmMajor = 226;
  if (major(st.st_rdev) == kDrmMajor)
    return true;
  const dev_t host_kfd = InterposerContext::ctx.host_kfd_rdev();
  return host_kfd != 0 && st.st_rdev == host_kfd;
}

/// @brief Append @p src_len bytes of @p src to @p out, bounded.
/// @returns False if the result would not fit, leaving @p out unusable.
[[nodiscard]] inline bool rj_path_append(char *out, size_t out_size, size_t *len, const char *src,
                                         size_t src_len) {
  if (*len + src_len + 1 > out_size)
    return false;
  std::memcpy(out + *len, src, src_len);
  *len += src_len;
  out[*len] = '\0';
  return true;
}

/// @brief Resolve an open-like (dirfd, path) pair to an absolute path.
/// @details Fixed buffers and no allocation. A child forked from a multithreaded
/// parent can inherit a held malloc arena lock, so this is as barred from allocating
/// as it is from taking a mutex; that also rules out snprintf's %s machinery, hence
/// the hand-built descriptor path.
/// @returns False when the pair cannot be resolved into @p out_size bytes.
[[nodiscard]] inline bool rj_child_absolute_path(int dirfd, const char *path, char *out,
                                                 size_t out_size) {
  if (out_size == 0)
    return false;
  out[0] = '\0';
  size_t len = 0;

  if (path[0] != '/') {
    if (dirfd == AT_FDCWD) {
      if (!::getcwd(out, out_size))
        return false;
      len = std::strlen(out);
    } else if (dirfd >= 0) {
      char link[sizeof("/proc/self/fd/") + 20];
      static constexpr char kPrefix[] = "/proc/self/fd/";
      size_t n = sizeof(kPrefix) - 1;
      std::memcpy(link, kPrefix, n);
      char digits[20];
      size_t d = 0;
      unsigned v = static_cast<unsigned>(dirfd);
      do {
        digits[d++] = static_cast<char>('0' + v % 10);
        v /= 10;
      } while (v > 0 && d < sizeof(digits));
      while (d > 0)
        link[n++] = digits[--d];
      link[n] = '\0';

      auto &real = InterposerContext::real();
      if (!real.readlink_fn)
        return false;
      const ssize_t got = real.readlink_fn(link, out, out_size - 1);
      if (got <= 0)
        return false;
      out[got] = '\0';
      len = static_cast<size_t>(got);
    } else {
      return false;
    }
    if (!rj_path_append(out, out_size, &len, "/", 1))
      return false;
  }
  return rj_path_append(out, out_size, &len, path, std::strlen(path));
}

/// @brief Collapse repeated separators, "." and ".." in place, lexically.
/// @details Lexical is exactly right for the one caller: it runs only when the
/// target does not exist, so there is no final component left for a symlink to
/// redirect. @p p must be absolute.
inline void rj_path_normalize(char *p) {
  p[0] = '/';
  char *out = p + 1;
  const char *in = p + 1;
  while (*in == '/')
    ++in;
  while (*in) {
    const char *start = in;
    while (*in && *in != '/')
      ++in;
    const size_t seg = static_cast<size_t>(in - start);
    if (seg == 1 && start[0] == '.') {
      // "." names the directory already written.
    } else if (seg == 2 && start[0] == '.' && start[1] == '.') {
      if (out > p + 1) {
        while (out > p + 1 && out[-1] != '/')
          --out;
        if (out > p + 1)
          --out; // and off the separator that preceded it
      }
    } else {
      // The separator goes BEFORE the segment, never after it. Writing a trailing
      // one would land on the NUL the read cursor is about to reach, and the
      // separator skip below would then walk `in` off the end of the string.
      if (out > p + 1)
        *out++ = '/';
      std::memmove(out, start, seg);
      out += seg;
    }
    while (*in == '/')
      ++in;
  }
  *out = '\0';
}

/// @brief True if @p p is an endpoint rocJITsu serves from the emulator.
/// @details Mirrors what the PARENT matches on: open() compares "/dev/kfd" exactly
/// and open_synthetic_drm_fd() matches the "/dev/dri/renderD" prefix. Both are
/// spellings, not device identities, so both hold whether or not the host has a
/// matching node. @p p must already be normalized.
[[nodiscard]] inline bool rj_path_names_emulated_endpoint(const char *p) {
  if (std::strcmp(p, "/dev/kfd") == 0)
    return true;
  static constexpr char kRenderPrefix[] = "/dev/dri/renderD";
  constexpr size_t kRenderPrefixLen = sizeof(kRenderPrefix) - 1;
  if (std::strncmp(p, kRenderPrefix, kRenderPrefixLen) != 0 || p[kRenderPrefixLen] == '\0')
    return false;
  for (const char *d = p + kRenderPrefixLen; *d; ++d)
    if (*d < '0' || *d > '9')
      return false;
  return true;
}

/// @brief True if an open-like call would land on a GPU device node.
/// @details Used ONLY by the forked-child gate, so it must classify without touching
/// any inherited interposer state -- no mutexes, no containers, no driver pointers.
///
/// Identity FIRST: it stat()s the resolved target and compares the device
/// major/minor, so "/dev//kfd", "/dev/./kfd", a symlink, a cwd-relative path and
/// openat(dirfd, "kfd", ...) are all caught, where a string compare catches only the
/// literal form. The stat goes through the pass-through table, never our own hook.
///
/// Spelling SECOND, and only when there is no node to identify. Identity is the
/// stronger test, but it needs something to stat, and a host with no GPU device nodes
/// -- CI, and any pure-emulation deployment -- has nothing to offer it. The parent
/// still serves those endpoints from the emulator there, because it matches on
/// spelling, so the child has to recognize the same spellings or its contract would
/// silently depend on the box having hardware.
///
/// Fails CLOSED: if the target cannot be classified it is treated as a GPU endpoint,
/// because the cost of a false positive (a child gets ENODEV on some unrelated node)
/// is far below the cost of a false negative (a child silently acquires real
/// hardware). Character devices only, so ordinary files are never refused.
[[nodiscard]] inline bool rj_child_open_hits_gpu_device(int dirfd, const char *path) {
  // Not a classification failure: there is no target to classify. open(nullptr) is
  // the caller's own bug and libc answers it with EFAULT, which this path has to
  // reproduce rather than convert into ENODEV.
  if (!path)
    return false;
  auto &real = InterposerContext::real();
  // Unresolved pass-through table: nothing here can classify anything, so this IS
  // the "cannot be classified" case and fails closed like the rest. Refusing is
  // also the only safe answer available -- the non-refused path forwards through
  // real.openat, which is one of the symbols missing here.
  if (!real.fstat_fn || !real.openat || !real.close)
    return true;

  // O_PATH resolves symlink chains and relative components against dirfd WITHOUT
  // opening the device, so classification has no side effects on real hardware.
  int probe = real.openat(dirfd, path, O_PATH | O_CLOEXEC, 0);
  if (probe < 0) {
    // No node there to identify -- but the interposer synthesizes GPU endpoints by
    // SPELLING (open() compares "/dev/kfd" exactly; open_synthetic_drm_fd() matches
    // the render-node prefix), so the parent would have served this open from the
    // emulator regardless of what the host has under /dev. Identity classification
    // is inert on such a host: host_kfd_rdev() stays 0 with nothing to record, and
    // there is no node to stat. Falling through would hand the child a plain ENOENT
    // for an endpoint its parent holds open, making the contract depend on whether
    // the box happens to have hardware -- so match the parent's own rule instead.
    char resolved[PATH_MAX];
    // Unresolvable pair (readlink failed, or the joined path exceeds PATH_MAX):
    // the spelling cannot be compared either, so both classifiers have now come up
    // empty and the documented fail-closed rule applies.
    if (!rj_child_absolute_path(dirfd, path, resolved, sizeof(resolved)))
      return true;
    rj_path_normalize(resolved);
    return rj_path_names_emulated_endpoint(resolved);
  }
  struct stat st {};
  const int rc = real.fstat_fn(probe, &st);
  real.close(probe);
  if (rc != 0)
    return true; // Unclassifiable: fail closed.
  return rj_is_gpu_device_stat(st);
}

/// @brief True if @p fd currently refers to a real GPU device.
/// @details Lock-free and state-free: one passthrough fstat, no interposer mutex,
/// container or pointer. That is what makes it usable from a forked child, where
/// every inherited lock is suspect.
[[nodiscard]] inline bool rj_fd_is_real_gpu(int fd) {
  if (fd < 0)
    return false;
  auto &real = InterposerContext::real();
  if (!real.fstat_fn)
    return false;
  struct stat st {};
  if (real.fstat_fn(fd, &st) != 0)
    return false;
  return rj_is_gpu_device_stat(st);
}

/// @brief Refuse an operation a forked child aimed at an inherited real GPU fd.
/// @details Hardware-backed GuestKfd holds a real /dev/kfd descriptor, and fork()
/// copies the whole descriptor table, so making the APPLICATION-facing fd synthetic
/// does not stop a child from finding the real one (e.g. by walking /proc/self/fd)
/// and using or laundering it across exec. Child pass-through paths therefore check
/// the descriptor's identity, not just the path used to obtain it.
///
/// This is a COOPERATIVE, API-level contract, not a security boundary: it defends
/// against a child that inherits hardware by accident, which is the realistic
/// failure, but interposition cannot stop a child that issues raw syscalls or
/// receives a descriptor over a unix socket. A hard guarantee requires hardware KFD
/// ownership to live outside the process (daemon mode), which is tracked separately.
[[nodiscard]] inline bool rj_child_refuses_fd(int fd) {
  if (!rj_fd_is_real_gpu(fd))
    return false;
  errno = ENODEV;
  return true;
}

/// @brief Pass an open through to libc, then VERIFY what was actually opened.
/// @details Closes the classify-then-reopen window: the O_PATH probe and the real
/// open resolve the path twice, so a target swapped in between could slip a real GPU
/// node past classification. Re-checking the descriptor we actually got makes the
/// decision depend on the object opened rather than on the name resolved earlier. A
/// GPU node that arrives this way is closed immediately, so the child never retains
/// a usable real-hardware descriptor.
[[nodiscard]] inline int rj_child_open_verified(int dirfd, const char *path, int flags,
                                                mode_t mode) {
  auto &real = InterposerContext::real();
  int fd = real.openat(dirfd, path, flags, mode);
  if (fd < 0)
    return fd;
  struct stat st {};
  if (real.fstat_fn && real.fstat_fn(fd, &st) == 0 && rj_is_gpu_device_stat(st)) {
    real.close(fd);
    errno = ENODEV;
    return -1;
  }
  return fd;
}

/// @brief Handle an open-like call to a GPU endpoint from a forked child.
/// @details Deliberately FAILS instead of passing through to libc. Passing through
/// would open the HOST's real /dev/kfd or render node when one exists, silently
/// moving the child off the emulator and onto real hardware -- a far worse outcome
/// than an error, and one that could corrupt real GPU state. ENODEV matches what the
/// interposer already returns when no backend can serve a request.
[[nodiscard]] inline bool rj_child_must_refuse_open(int dirfd, const char *path, int *out_errno) {
  if (!rj_child_open_hits_gpu_device(dirfd, path))
    return false;
  *out_errno = ENODEV;
  return true;
}

/// @brief True when this process owns the interposer state and may touch it.
/// @details False in a process that inherited this address space through fork/vfork
/// and has not yet exec'd. rocJITsu registers NO atfork handlers (see
/// InterposerContext::init()), so such a child may hold locks whose owners no longer
/// exist and containers a vanished thread was mid-mutation on. Every interposed entry
/// point must therefore test this BEFORE in_construction, any driver or remote
/// snapshot, fd classification, path redirection, logging that reaches context state,
/// or any mutex -- i.e. before touching anything inherited at all.
///
/// Callers must already have confirmed real().ready(); this deliberately does not
/// re-test it, because the not-ready early path differs per hook (raw syscall or
/// dlsym) and must run first.
[[nodiscard]] inline bool rj_owns_interposer_state() {
  return InterposerContext::ctx.owned_by_current_process();
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
  // ready() first, as openat() and every other entry point does. owner_pid_ is zero
  // until init() runs, so before the constructor rj_owns_interposer_state() is false
  // for the OWNER too -- taking the child branch there would fail closed on an
  // unresolved passthrough table and turn every open() in that window, GPU endpoint
  // or not, into ENODEV.
  if (InterposerContext::real().ready() && !rj_owns_interposer_state()) {
    int refuse_errno = 0;
    if (rj_child_must_refuse_open(AT_FDCWD, path, &refuse_errno)) {
      errno = refuse_errno;
      return -1;
    }
    return rj_child_open_verified(AT_FDCWD, path, flags, mode);
  }
  auto *volatile p = path;
  if (!p || InterposerContext::in_construction)
    return InterposerContext::real().openat(AT_FDCWD, path, flags, mode);

  if (int snapshot = open_proc_maps_snapshot(path, flags); snapshot >= 0)
    return snapshot;

  if (auto drm_fd = open_synthetic_drm_fd(path); drm_fd.handled)
    return drm_fd.fd;

  if (std::strcmp(path, "/dev/kfd") == 0) {
    // Use the fd captured under remote_mutex_ (not a fresh remote_kfd_fd() read),
    // so a concurrent dup2/dup3 that invalidates the primary fd cannot make this
    // return -1 or a reused non-KFD descriptor.
    if (auto remote = InterposerContext::ctx.get_or_create_remote())
      return remote.fd;

    // ONE combined operation under init_mutex_: creating the backend and taking the
    // application's reference on it must not be separable, or the shutdown policy
    // can retire the backend in between and hand back a dead fd.
    auto opened = InterposerContext::ctx.open_local();
    if (opened.fd < 0)
      return opened.fd;
    // clear_dups() routes through release_local_open() (init_mutex_), so it must run
    // after open_local() has released it.
    if (opened.clear_dups_needed)
      InterposerContext::ctx.clear_dups();
    return opened.fd;
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

  if (InterposerContext::real().ready() && !rj_owns_interposer_state()) {
    int refuse_errno = 0;
    if (rj_child_must_refuse_open(dirfd, path, &refuse_errno)) {
      errno = refuse_errno;
      return -1;
    }
    return rj_child_open_verified(dirfd, path, flags, mode);
  }
  auto *volatile p_at = path;
  if (!p_at)
    return InterposerContext::real().openat(dirfd, path, flags, mode);
  if (InterposerContext::in_construction)
    return InterposerContext::real().openat(dirfd, path, flags, mode);

  if (path[0] == '/') {
    if (int snapshot = open_proc_maps_snapshot(path, flags); snapshot >= 0)
      return snapshot;

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
  // A vfork child shares the parent's address space until exec/_exit but has a
  // separate descriptor table. Descriptor cleanup in that window must close the
  // child's fd without clearing the parent's KFD/DRM bookkeeping.
  if (!InterposerContext::ctx.owned_by_current_process())
    return static_cast<int>(InterposerContext::real().close(fd));
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
  InterposerContext::DrmUntrackResult drm_close;
  {
    auto drm_lifecycle = InterposerContext::ctx.lock_drm_fd_lifecycle();
    drm_close = InterposerContext::ctx.untrack_drm(fd);
    if (drm_close.tracked)
      InterposerContext::real().close(fd);
  }
  if (drm_close.tracked) {
    // Reaps GEM while the backend is still open, then drops the lease. Releasing a
    // local lease routes through release_local_open(), so a final DRM close can
    // complete a pending process-exit shutdown exactly as a final KFD close does.
    InterposerContext::ctx.complete_drm_release(std::move(drm_close.release));
    return 0;
  }
  if (InterposerContext::ctx.is_kfd_dup(fd)) {
    InterposerContext::ctx.untrack_dup(fd);
    return static_cast<int>(InterposerContext::real().close(fd));
  }
  // Classify the fd against the local driver, then release the snapshot before
  // release_local_open() so that call can actually free the driver when this was
  // the last reference.
  bool is_primary_local_fd = false;
  bool is_owned_fd = false;
  {
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
  // PID gate FIRST, before anything else. A forked child that calls exit() instead
  // of exec() runs this finalizer too, and request_local_vm_shutdown() takes
  // init_mutex_ -- inherited, possibly locked by a parent thread that does not exist
  // here. The parent owns this VM's teardown; a child must never attempt it.
  if (!InterposerContext::ctx.owned_by_current_process())
    return;
  InterposerContext::ctx.request_local_vm_shutdown();
}

RJ_INTERPOSER_EXPORT int ioctl(int fd, unsigned long request, ...) {
  assert(InterposerContext::real().ready());
  va_list ap;
  va_start(ap, request);
  void *arg = va_arg(ap, void *);
  va_end(ap);
  if (!rj_owns_interposer_state()) {
    if (rj_child_refuses_fd(fd))
      return -1;
    return InterposerContext::real().ioctl(fd, request, arg);
  }

  constexpr unsigned kDrmIoctlType = 'd';
  constexpr unsigned kDrmIoctlNrVersion = 0x00;
  constexpr unsigned kDrmIoctlNrGemClose = _IOC_NR(DRM_IOCTL_GEM_CLOSE);
  constexpr unsigned kDrmIoctlNrAmdgpuInfo = DRM_COMMAND_BASE + DRM_AMDGPU_INFO;
  constexpr unsigned kDrmIoctlNrGemVa = DRM_COMMAND_BASE + DRM_AMDGPU_GEM_VA;
  constexpr unsigned kDrmIoctlNrPrimeFdToHandle = _IOC_NR(DRM_IOCTL_PRIME_FD_TO_HANDLE);
  constexpr unsigned kDrmIoctlNrSyncobjCreate = _IOC_NR(DRM_IOCTL_SYNCOBJ_CREATE);
  constexpr unsigned kDrmIoctlNrSyncobjDestroy = _IOC_NR(DRM_IOCTL_SYNCOBJ_DESTROY);
  constexpr unsigned kDrmIoctlNrSyncobjTimelineWait = _IOC_NR(DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT);

  auto drm_file_reservation = InterposerContext::ctx.reserve_drm_file_for_ioctl(fd);
  const auto &drm_file = drm_file_reservation.file();
  if (drm_file) {
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
      prime->handle = InterposerContext::ctx.prime_import(prime->fd, drm_file, sz);
      if (prime->handle == 0) {
        errno = EBADF;
        return -1;
      }
      return 0;
    }
    if (type == kDrmIoctlType && nr == kDrmIoctlNrSyncobjCreate && arg) {
      auto *create = static_cast<drm_syncobj_create *>(arg);
      return kfd_ioctl_ret(
          InterposerContext::ctx.create_syncobj(drm_file, create->flags, &create->handle));
    }
    if (type == kDrmIoctlType && nr == kDrmIoctlNrSyncobjDestroy && arg) {
      auto *destroy = static_cast<drm_syncobj_destroy *>(arg);
      if (destroy->pad != 0) {
        errno = EINVAL;
        return -1;
      }
      return kfd_ioctl_ret(InterposerContext::ctx.destroy_syncobj(drm_file, destroy->handle));
    }
    if (type == kDrmIoctlType && nr == kDrmIoctlNrSyncobjTimelineWait && arg) {
      return kfd_ioctl_ret(InterposerContext::ctx.wait_syncobj_timeline(
          drm_file, static_cast<drm_syncobj_timeline_wait *>(arg)));
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
      // Answer from the FILE's own backend rather than re-deriving one from the
      // published pointer. The file's lease already names the backend that owns this
      // render node, so this cannot consult a driver that does not.
      auto gpu_info = interposer_gpu_info_for(drm_file);
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
        drm_amdgpu_info_device dev{};
        dev.device_id = gpu->device_id;
        dev.chip_rev = gpu->revision_id;
        dev.external_rev = rocjitsu::kmd::external_rev_id_for_gfx_target_version(
            gpu->gfx_target_version, gpu->revision_id);
        dev.pci_rev = gpu->pci_revision_id;
        dev.family = gpu->family_id;
        // libdrm reports shader engines, which GpuInfo already stores directly;
        // round-trip the derived KFD array_count to keep the two views pinned to
        // one definition.
        dev.num_shader_engines = rocjitsu::kmd::drm_shader_engine_count(
            gpu->array_count_per_xcc(), gpu->num_shader_arrays_per_engine);
        dev.num_shader_arrays_per_engine = gpu->num_shader_arrays_per_engine;
        dev.gpu_counter_freq = 100000;
        dev.max_engine_clock = gpu->max_engine_clk_fcompute;
        dev.max_memory_clock = gpu->mem_clk_max;
        dev.wave_front_size = gpu->wave_front_size;
        dev.num_cu_per_sh = gpu->num_cu_per_sh;
        dev.num_hw_gfx_contexts =
            rocjitsu::kmd::num_hw_gfx_contexts_for_gfx_target_version(gpu->gfx_target_version);
        dev.vram_type = gpu->vram_type;
        dev.vram_bit_width = gpu->mem_width;
        dev.cu_active_number =
            rocjitsu::kmd::drm_cu_active_number(gpu->simd_count, gpu->simd_per_cu);
        // VA aperture — libdrm's VA manager (amdgpu_vamgr_init) needs a sane
        // range. Mirror the KFD GPUVM aperture used elsewhere.
        dev.virtual_address_offset = 0x200000;       // 2 MiB
        dev.virtual_address_max = 0x800000000000ULL; // 47-bit canonical
        dev.virtual_address_alignment = 0x1000;      // 4 KiB
        dev.pte_fragment_size = 0x200000;            // 2 MiB
        dev.gart_page_size = 0x1000;                 // 4 KiB
        dev.high_va_offset = 0xffff800000000000ULL;
        dev.high_va_max = 0xffffffffffffffffULL;

        // Older libdrm headers use a shorter trailing struct. The kernel ABI
        // returns the prefix that fits instead of withholding every field.
        std::memcpy(out, &dev, std::min<size_t>(info->return_size, sizeof(dev)));
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
      return kfd_ioctl_ret(InterposerContext::ctx.untrack_gem(drm_file, gc->handle));
    }
    if (type == kDrmIoctlType && nr == kDrmIoctlNrGemVa && arg) {
      // GEM_VA installs (or tears down) a GPU virtual mapping for a prime-
      // imported buffer. HSA's vmem path (hsa_amd_vmem_map) lowers to this via
      // amdgpu_bo_va_op; IREE's ring allocator triple-maps one BO at adjacent
      // VAs. We map by GEM handle, lazily mmap the backing pages, and
      // install/remove them in the GPU page table.
      auto *va = static_cast<drm_amdgpu_gem_va *>(arg);
      // The kernel waits synchronously for every input fence before touching the
      // VM. This model has no input-fence lookup/wait implementation, so reject a
      // nonzero authoritative count instead of applying the update before its
      // dependencies. The pointer is unused and ignored when the count is zero.
      if (va->num_syncobj_handles != 0) {
        errno = EINVAL;
        return -1;
      }
      // Ask THIS FILE's backend, not the globally published one. A remote DRM file
      // must not install page-table entries into an unrelated local simulator just
      // because one happens to be published.
      if (!InterposerContext::ctx.drm_file_is_simulated(drm_file)) {
        errno = ENODEV;
        return -1;
      }
      const bool publish_timeline = (va->flags & AMDGPU_VM_DELAY_UPDATE) == 0;
      int result = 0;
      switch (va->operation) {
      case AMDGPU_VA_OP_MAP:
      case AMDGPU_VA_OP_REPLACE:
        // REPLACE evicts any prior occupant of the VA range in this DRM-file
        // namespace before installing the new mapping, so closing the old handle
        // cannot later unmap the replacement. MAP rejects a range already in use.
        result = InterposerContext::ctx.gem_map(
            drm_file, va->handle, va->va_address, va->offset_in_bo, va->map_size,
            /*replace=*/va->operation == AMDGPU_VA_OP_REPLACE, publish_timeline,
            va->vm_timeline_syncobj_out, va->vm_timeline_point);
        break;
      case AMDGPU_VA_OP_UNMAP:
        // UNMAP requires the supplied handle to own the exact range.
        result = InterposerContext::ctx.gem_unmap(
            drm_file, va->handle, va->va_address, va->map_size, publish_timeline,
            va->vm_timeline_syncobj_out, va->vm_timeline_point);
        break;
      case AMDGPU_VA_OP_CLEAR:
        // CLEAR is handle-agnostic within the calling DRM-file namespace.
        result = InterposerContext::ctx.gem_clear(drm_file, va->va_address, va->map_size,
                                                  publish_timeline, va->vm_timeline_syncobj_out,
                                                  va->vm_timeline_point);
        break;
      default:
        // Unknown GEM_VA operation — do not claim it succeeded.
        errno = EINVAL;
        return -1;
      }
      if (result != 0)
        return kfd_ioctl_ret(result);
      return 0;
    }
    // The AMDGPU command range starts at DRM_COMMAND_BASE, while generic DRM
    // requests occupy ranges on both sides of it (timeline syncobj requests, for
    // example, start at 0xbf). Only label requests in the range defined by this
    // vendored AMDGPU UAPI as driver-relative commands.
    if (nr >= DRM_COMMAND_BASE && nr < DRM_COMMAND_END) {
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
    // Lifetime here rests on shared_ptr snapshots, not on a lock: publication swaps
    // the pointer under publish_mutex_ and a reader copies it, so holding a copy is
    // what keeps a driver alive. Two separate snapshots are involved, and neither
    // spans the other:
    //   - the classification, kfd_backend_of() -> driver_fd(), dereferences the local
    //     driver but takes its OWN snapshot internally, so it is safe on its own and
    //     needs nothing held around it here;
    //   - the Local branch takes its own snapshot (drv) and dispatches under it.
    // This block takes no init_mutex_; it takes fd_mutex_ (track_gem_flags) only
    // under the Local snapshot, matching the fd_mutex_ order.
    //
    // The Remote branch must NOT dispatch while holding a local snapshot:
    // remote->ioctl() can block indefinitely (e.g. a daemon-side WAIT_EVENTS), which
    // would keep an idle LOCAL VM alive and stall its teardown on an unrelated remote
    // operation. So it only CAPTURES the remote snapshot here and dispatches after
    // the classification scope has closed -- that snapshot keeps the remote alive by
    // itself.
    std::shared_ptr<RemoteDriver> remote_dispatch;
    {
      if (auto backend = InterposerContext::ctx.kfd_backend_of(fd)) {
        if (*backend == InterposerContext::DupBackend::Remote) {
          // kfd_backend_of() already established this fd is Remote-backed, so route
          // via the remote snapshot directly rather than remote_lookup(remote_kfd_fd_):
          // the primary fd number may have been invalidated/reused while a remote
          // shared_ptr snapshot is still live, and this dup still belongs to it.
          remote_dispatch = InterposerContext::ctx.remote();
        } else if (auto drv = InterposerContext::ctx.driver()) {
          int rc = drv->ioctl(request, arg);
          // Capture the KFD allocation flags for a freshly exported dmabuf fd. The
          // flags determine the GPU PTE MTYPE when the fd is later mapped via GEM_VA,
          // and must be recorded now because the allocation may be freed first. Only
          // the local simulated driver exports dmabufs this path can later map.
          if (rc == 0 && request == AMDKFD_IOC_EXPORT_DMABUF && arg) {
            if (auto *sim = dynamic_cast<SimulatedKfd *>(drv.get())) {
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
    // Classification done and no local snapshot held: dispatch the remote op on the
    // lifetime-extending snapshot captured above.
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
    // kfd_backend_of() -> driver_fd() dereferences the local driver, but snapshots it
    // internally, so the classification is lifetime-safe without holding anything
    // here; the routed remote->ioctl is separately safe via its own snapshot.
    InterposerContext::DupBackend backend;
    bool have_backend = false;
    {
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
  if (!rj_owns_interposer_state()) {
    if (rj_child_refuses_fd(oldfd))
      return -1;
    return InterposerContext::real().dup(oldfd);
  }
  std::optional<InterposerContext::DupBackend> reserved;
  InterposerContext::DrmFileToken drm_file;
  InterposerContext::DrmFinalRelease drm_release;
  int rc;
  // Reserve the backend BEFORE the DRM lifecycle lock. reserve_dup_backend() takes
  // init_mutex_ for a tracked KFD fd, while driver code running under init_mutex_
  // reaches the interposed close() hook, which takes drm_fd_lifecycle_mutex_ --
  // acquiring them in the other order here would close an ABBA cycle between a dup
  // and a close. Only "reserve before the syscall" is load-bearing, and it still
  // holds; the failure path below releases the reservation outside the lock too.
  reserved = InterposerContext::ctx.reserve_dup_backend(oldfd);
  {
    auto drm_lifecycle = InterposerContext::ctx.lock_drm_fd_lifecycle();
    // Keep the source kernel fd and its DRM identity paired until dup() and the
    // tracking update complete. Backend and GEM cleanup run after this scope.
    drm_file = InterposerContext::ctx.reserve_drm_file(oldfd);
    rc = InterposerContext::real().dup(oldfd);
    if (rc >= 0)
      drm_release = InterposerContext::ctx.commit_drm_dup(rc, drm_file);
  }
  if (rc < 0) {
    const int saved_errno = errno;
    if (reserved)
      InterposerContext::ctx.release_backend(*reserved);
    InterposerContext::ctx.release_drm_file_reservation(drm_file);
    errno = saved_errno;
    return rc;
  }
  if (reserved)
    InterposerContext::ctx.commit_dup(rc, *reserved);
  else
    InterposerContext::ctx.untrack_dup(rc);
  InterposerContext::ctx.complete_drm_release(std::move(drm_release));
  return rc;
}

namespace {
// Reconcile interposer tracking after a successful dup2/dup3(oldfd -> newfd),
// consuming a backend reservation taken (before the syscall) for oldfd.
// dup2/dup3 atomically CLOSE newfd before installing the duplicate, so any
// tracking/reference newfd previously held must be dropped before the
// replacement is tracked:
//   - stale sysfs tracking for newfd is removed;
//   - if newfd was a tracked KFD dup, its reference is released and its stale
//     kfd_dup_fds_ entry erased (otherwise commit_dup would see the stale entry
//     and release the newly-reserved backend, leaving the OLD backend recorded);
//   - if newfd was a PRIMARY KFD fd (local or remote), that primary identity is
//     invalidated and its reference released, so the reused fd number no longer
//     routes to the old backend.
// Then the replacement is recorded on the reserved source backend.
void reconcile_dup_target(int newfd, std::optional<InterposerContext::DupBackend> reserved,
                          InterposerContext::DrmFinalRelease overwritten_release,
                          InterposerContext::DrmFinalRelease displaced_release) {
  InterposerContext::ctx.untrack_sysfs(newfd);
  // dup2/dup3 atomically close whatever newfd was, bypassing the close() hook, so
  // every per-fd cleanup close() performs must be mirrored here. Drop any transient
  // EXPORT_DMABUF flags for newfd: a dmabuf export fd overwritten before a
  // PRIME_FD_TO_HANDLE would otherwise leave a stale fd→flags record that a later
  // PRIME on the recycled fd number could misapply as the wrong PTE MTYPE. No-op for
  // non-dmabuf fds.
  InterposerContext::ctx.drop_pending_gem_flags(newfd);
  InterposerContext::ctx.complete_drm_release(std::move(overwritten_release));
  InterposerContext::ctx.complete_drm_release(std::move(displaced_release));
  InterposerContext::ctx.invalidate_overwritten_kfd_fd(newfd);
  if (reserved)
    InterposerContext::ctx.commit_dup(newfd, *reserved);
}
} // namespace

RJ_INTERPOSER_EXPORT int dup2(int oldfd, int newfd) {
  assert(InterposerContext::real().ready());
  if (!rj_owns_interposer_state()) {
    if (rj_child_refuses_fd(oldfd))
      return -1;
    return static_cast<int>(syscall(SYS_dup2, oldfd, newfd));
  }
  // dup2(fd, fd) is a POSIX no-op that leaves the descriptor live; mutating
  // tracking would drop a still-open ref. Forward without touching tracking.
  if (oldfd == newfd)
    return InterposerContext::real().dup2(oldfd, newfd);
  std::optional<InterposerContext::DupBackend> reserved;
  InterposerContext::DrmFileToken drm_file;
  InterposerContext::DrmFinalRelease overwritten_release;
  InterposerContext::DrmFinalRelease displaced_release;
  int rc;
  // Reserve the backend BEFORE the DRM lifecycle lock. reserve_dup_backend() takes
  // init_mutex_ for a tracked KFD fd, while driver code running under init_mutex_
  // reaches the interposed close() hook, which takes drm_fd_lifecycle_mutex_ --
  // acquiring them in the other order here would close an ABBA cycle between a dup
  // and a close. Only "reserve before the syscall" is load-bearing, and it still
  // holds; the failure path below releases the reservation outside the lock too.
  reserved = InterposerContext::ctx.reserve_dup_backend(oldfd);
  {
    auto drm_lifecycle = InterposerContext::ctx.lock_drm_fd_lifecycle();
    drm_file = InterposerContext::ctx.reserve_drm_file(oldfd);
    rc = InterposerContext::real().dup2(oldfd, newfd);
    if (rc >= 0) {
      auto drm_close = InterposerContext::ctx.untrack_drm(rc);
      overwritten_release = std::move(drm_close.release);
      displaced_release = InterposerContext::ctx.commit_drm_dup(rc, drm_file);
    }
  }
  if (rc < 0) {
    const int saved_errno = errno;
    if (reserved)
      InterposerContext::ctx.release_backend(*reserved);
    InterposerContext::ctx.release_drm_file_reservation(drm_file);
    errno = saved_errno;
    return rc;
  }
  reconcile_dup_target(rc, reserved, std::move(overwritten_release), std::move(displaced_release));
  return rc;
}

#ifdef SYS_dup3
RJ_INTERPOSER_EXPORT int dup3(int oldfd, int newfd, int flags) {
  assert(InterposerContext::real().ready());
  if (!rj_owns_interposer_state()) {
    if (rj_child_refuses_fd(oldfd))
      return -1;
    return static_cast<int>(syscall(SYS_dup3, oldfd, newfd, flags));
  }
  // dup3(fd, fd, ...) is required to fail with EINVAL without altering the
  // descriptor; do not mutate tracking before the syscall confirms that.
  std::optional<InterposerContext::DupBackend> reserved;
  InterposerContext::DrmFileToken drm_file;
  InterposerContext::DrmFinalRelease overwritten_release;
  InterposerContext::DrmFinalRelease displaced_release;
  int rc;
  // Reserve the backend BEFORE the DRM lifecycle lock. reserve_dup_backend() takes
  // init_mutex_ for a tracked KFD fd, while driver code running under init_mutex_
  // reaches the interposed close() hook, which takes drm_fd_lifecycle_mutex_ --
  // acquiring them in the other order here would close an ABBA cycle between a dup
  // and a close. Only "reserve before the syscall" is load-bearing, and it still
  // holds; the failure path below releases the reservation outside the lock too.
  reserved = InterposerContext::ctx.reserve_dup_backend(oldfd);
  {
    auto drm_lifecycle = InterposerContext::ctx.lock_drm_fd_lifecycle();
    drm_file = InterposerContext::ctx.reserve_drm_file(oldfd);
    rc = InterposerContext::real().dup3(oldfd, newfd, flags);
    if (rc >= 0) {
      auto drm_close = InterposerContext::ctx.untrack_drm(rc);
      overwritten_release = std::move(drm_close.release);
      displaced_release = InterposerContext::ctx.commit_drm_dup(rc, drm_file);
    }
  }
  if (rc < 0) {
    const int saved_errno = errno;
    if (reserved)
      InterposerContext::ctx.release_backend(*reserved);
    InterposerContext::ctx.release_drm_file_reservation(drm_file);
    errno = saved_errno;
    return rc;
  }
  reconcile_dup_target(rc, reserved, std::move(overwritten_release), std::move(displaced_release));
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
  // Gate here rather than in fcntl()/fcntl64(): both funnel through this, and the
  // F_DUPFD paths below touch inherited dup and DRM tracking.
  if (!rj_owns_interposer_state()) {
    // Two ways to launder hardware authority past exec, not one: duplicating the
    // descriptor, OR clearing FD_CLOEXEC on the descriptor already held, which needs
    // no duplication at all. Ordinary fcntl queries on an inherited fd are harmless
    // and must keep working, so only these are gated.
    const bool duplicates = (cmd == F_DUPFD || cmd == F_DUPFD_CLOEXEC);
    const bool launders_across_exec = (cmd == F_SETFD && (int_arg & FD_CLOEXEC) == 0);
    if ((duplicates || launders_across_exec) && rj_child_refuses_fd(fd))
      return -1;
    if (kind == FcntlArgKind::Ptr)
      return InterposerContext::real().fcntl(fd, cmd, ptr_arg);
    if (kind == FcntlArgKind::Int)
      return InterposerContext::real().fcntl(fd, cmd, int_arg);
    return InterposerContext::real().fcntl(fd, cmd);
  }
  // F_DUPFD/F_DUPFD_CLOEXEC create a new fd like dup(); reserve the source
  // backend BEFORE the syscall so a racing last-close cannot tear it down
  // between the dup and tracking the new fd. F_DUPFD does not overwrite an
  // existing fd, so no primary/dup reconciliation of a target is needed.
  const bool is_dupfd = (cmd == F_DUPFD || cmd == F_DUPFD_CLOEXEC);
  std::optional<InterposerContext::DupBackend> reserved;
  InterposerContext::DrmFileToken drm_file;
  const auto invoke = [&]() -> long {
    switch (kind) {
    case FcntlArgKind::Int:
      return InterposerContext::real().fcntl(fd, cmd, int_arg);
    case FcntlArgKind::Ptr:
      return InterposerContext::real().fcntl(fd, cmd, ptr_arg);
    case FcntlArgKind::None:
    default:
      return InterposerContext::real().fcntl(fd, cmd, 0L);
    }
  };

  InterposerContext::DrmFinalRelease drm_release;
  long rc;
  // Reserve the backend BEFORE the DRM lifecycle lock. reserve_dup_backend() takes
  // init_mutex_ for a tracked KFD fd, while driver code running under init_mutex_
  // reaches the interposed close() hook, which takes drm_fd_lifecycle_mutex_ --
  // acquiring them in the other order here would close an ABBA cycle between a dup
  // and a close. Only "reserve before the syscall" is load-bearing, and it still
  // holds; the failure path below releases the reservation outside the lock too.
  if (is_dupfd)
    reserved = InterposerContext::ctx.reserve_dup_backend(fd);
  if (is_dupfd) {
    auto drm_lifecycle = InterposerContext::ctx.lock_drm_fd_lifecycle();
    drm_file = InterposerContext::ctx.reserve_drm_file(fd);
    rc = invoke();
    if (rc >= 0)
      drm_release = InterposerContext::ctx.commit_drm_dup(static_cast<int>(rc), drm_file);
  } else {
    rc = invoke();
  }

  if (is_dupfd) {
    if (rc < 0) {
      const int saved_errno = errno;
      if (reserved)
        InterposerContext::ctx.release_backend(*reserved);
      InterposerContext::ctx.release_drm_file_reservation(drm_file);
      errno = saved_errno;
    } else {
      if (reserved)
        InterposerContext::ctx.commit_dup(static_cast<int>(rc), *reserved);
      else
        InterposerContext::ctx.untrack_dup(static_cast<int>(rc));
      InterposerContext::ctx.complete_drm_release(std::move(drm_release));
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

// libdrm and any translation unit built with _FILE_OFFSET_BITS=64 bind mmap64, not
// mmap, exactly as they bind fcntl64 rather than fcntl. Exporting only mmap would let
// those calls bypass the child real-GPU refusal, local/remote KFD routing, per-file
// DRM lease routing, and doorbell/daemon shared-memory handling. Both symbols share
// one implementation so the two can never drift.
static void *mmap_impl(void *addr, size_t length, int prot, int flags, int fd, off_t offset);

RJ_INTERPOSER_EXPORT void *mmap(void *addr, size_t length, int prot, int flags, int fd,
                                off_t offset) {
  return mmap_impl(addr, length, prot, flags, fd, offset);
}

RJ_INTERPOSER_EXPORT void *mmap64(void *addr, size_t length, int prot, int flags, int fd,
                                  off64_t offset) {
  // off_t is 64-bit on the supported target, so this is a width-preserving forward.
  // The static_assert makes a narrower off_t a build failure rather than a silent
  // offset truncation.
  static_assert(sizeof(off_t) == sizeof(off64_t),
                "mmap64 would truncate its offset: off_t is narrower than off64_t");
  return mmap_impl(addr, length, prot, flags, fd, static_cast<off_t>(offset));
}

static void *mmap_impl(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  if (!InterposerContext::real().ready() || !InterposerContext::real().mmap)
    return raw_mmap_syscall(addr, length, prot, flags, fd, offset);
  if (!rj_owns_interposer_state()) {
    if (rj_child_refuses_fd(fd))
      return MAP_FAILED;
    // A forked child owns none of this state, and the mapping lock is inherited
    // like any other: its holder may not exist here. Straight to libc.
    return InterposerContext::real().mmap(addr, length, prot, flags, fd, offset);
  }

  assert(InterposerContext::real().ready());
  if (auto remote = InterposerContext::ctx.remote_lookup(fd))
    return remote->mmap(addr, length, prot, flags, offset);

  // Classify and dispatch the LOCAL branches on a driver snapshot, which keeps the
  // VM/GuestKfd alive for the duration. For a Remote-backed fd, capture a
  // lifetime-extending remote snapshot and dispatch that instead.
  std::shared_ptr<RemoteDriver> remote_mmap;
  {

    if (auto drv = InterposerContext::ctx.lookup(fd))
      return drv->mmap(addr, length, prot, flags, offset);

    // Dispatch a tracked KFD dup by its RECORDED backend (as ioctl() does), not by
    // "remote if any remote is live", so routing follows the reference the dup
    // actually holds. For Remote, route via the remote snapshot directly so routing
    // still works if the primary fd number changed.
    if (auto backend = InterposerContext::ctx.kfd_backend_of(fd)) {
      if (*backend == InterposerContext::DupBackend::Remote) {
        remote_mmap = InterposerContext::ctx.remote();
      } else if (auto drv = InterposerContext::ctx.driver()) {
        return drv->mmap(addr, length, prot, flags, offset);
      }
    }

    // A tracked DRM fd routes by the backend ITS FILE leased, not by "remote if any
    // remote is live" -- the file's lease is the authority on which backend owns it.
    // The reservation also keeps that lease alive and the file's identity pinned
    // across the dispatch, so a racing close cannot retire the backend or detach the
    // file mid-call.
    if (!remote_mmap) {
      if (auto drm_file = InterposerContext::ctx.reserve_drm_file(fd)) {
        InterposerContext::DrmFileReservation held(InterposerContext::ctx, drm_file);
        const auto &lease = drm_file->backend_lease;
        // BOTH branches dispatch inside the reservation. Holding a shared_ptr keeps
        // the object allocated but not OPERATIONAL: if the reservation were released
        // first and a racing final close made it the last reference, the lease would
        // run release_remote_open(), RemoteDriver::close() would drop the RPC
        // socket, and this mmap would then execute against a closed connection.
        // Keeping the lease alive across the call is the whole point of it.
        if (const auto &local = lease.local())
          return local->mmap(addr, length, prot, flags, offset);
        if (const auto &remote = lease.remote())
          return remote->mmap(addr, length, prot, flags, offset);
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
        auto *raw = with_host_mapping_held([&] {
          return InterposerContext::real().mmap(
              addr, length, prot, (flags & ~MAP_ANONYMOUS) | MAP_SHARED, memfd_out, memfd_offset);
        });
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
  if ((flags & MAP_FIXED) && addr) {
    if (auto driver = InterposerContext::ctx.driver())
      return driver->mmap_replacing_client_doorbell_views(addr, length, prot, flags, fd, offset);
  }
  return with_host_mapping_held(
      [&] { return InterposerContext::real().mmap(addr, length, prot, flags, fd, offset); });
}

RJ_INTERPOSER_EXPORT int mprotect(void *addr, size_t length, int prot) {
  assert(InterposerContext::real().ready());
  if (!rj_owns_interposer_state())
    return InterposerContext::real().mprotect(addr, length, prot);
  {
    auto drv = InterposerContext::ctx.driver();
    if (drv && drv->is_doorbell_range(addr, length)) {
      errno = EPERM;
      return -1;
    }
  }
  return with_host_mapping_held(
      [&] { return InterposerContext::real().mprotect(addr, length, prot); });
}

RJ_INTERPOSER_EXPORT int madvise(void *addr, size_t length, int advice) {
  assert(InterposerContext::real().ready());
  if (!rj_owns_interposer_state())
    return InterposerContext::real().madvise(addr, length, advice);
  const bool high_gpu_address = reinterpret_cast<uintptr_t>(addr) >= 0x1000000000ULL;
  if (advice == MADV_HUGEPAGE && high_gpu_address)
    return 0;
  if (advice == MADV_DONTFORK && high_gpu_address) {
    if (InterposerContext::ctx.driver_is_simulated())
      return 0;
  }
  return InterposerContext::real().madvise(addr, length, advice);
}

/// @brief mremap moves or resizes an existing mapping, so it can pull an
/// identity page out from under an in-flight emulated atomic exactly as munmap
/// and mprotect can. rocjitsu does not redirect it -- no driver mapping is
/// reachable through it -- but it still has to be serialized.
RJ_INTERPOSER_EXPORT void *mremap(void *old_address, size_t old_size, size_t new_size, int flags,
                                  ...) {
  void *new_address = nullptr;
  if (flags & MREMAP_FIXED) {
    va_list ap;
    va_start(ap, flags);
    new_address = va_arg(ap, void *);
    va_end(ap);
  }
  if (!InterposerContext::real().ready() || !InterposerContext::real().mremap)
    return raw_mremap_syscall(old_address, old_size, new_size, flags, new_address);
  if (!rj_owns_interposer_state())
    return InterposerContext::real().mremap(old_address, old_size, new_size, flags, new_address);
  return with_host_mapping_held([&] {
    return InterposerContext::real().mremap(old_address, old_size, new_size, flags, new_address);
  });
}

RJ_INTERPOSER_EXPORT int munmap(void *addr, size_t length) {
  if (!InterposerContext::real().ready() || !InterposerContext::real().munmap)
    return raw_munmap_syscall(addr, length);
  if (!rj_owns_interposer_state())
    return InterposerContext::real().munmap(addr, length);

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
    if (auto drv = InterposerContext::ctx.driver()) {
      int ret = drv->munmap(addr, length);
      if (ret != -ENOENT)
        return ret;
    }
  }
  return with_host_mapping_held([&] { return InterposerContext::real().munmap(addr, length); });
}

} // extern "C"

extern "C" {

// -- fopen / freopen interposition (sysfs redirect) --

RJ_INTERPOSER_EXPORT FILE *fopen(const char *path, const char *mode) {
  if (!InterposerContext::real().ready()) {
    auto fn = util::lookup_symbol<FILE *(*)(const char *, const char *)>(RTLD_NEXT, "fopen");
    return fn ? fn(path, mode) : nullptr;
  }
  if (!rj_owns_interposer_state()) {
    int refuse_errno = 0;
    if (rj_child_must_refuse_open(AT_FDCWD, path, &refuse_errno)) {
      errno = refuse_errno;
      return nullptr;
    }
    // Call libc's fopen and POST-validate, rather than re-deriving open flags and
    // using fdopen. Re-deriving loses semantics only libc's own mode parser has: the
    // 0666 creation mode for "w"/"a", "x" (exclusive -- must fail EEXIST rather than
    // truncate), and "e" (FD_CLOEXEC). Validating the descriptor afterwards closes
    // the same path-swap window without reimplementing any of that.
    FILE *stream = InterposerContext::real().fopen(path, mode);
    if (!stream)
      return nullptr;
    struct stat st {};
    const int fd = fileno(stream);
    if (fd >= 0 && InterposerContext::real().fstat_fn &&
        InterposerContext::real().fstat_fn(fd, &st) == 0 && rj_is_gpu_device_stat(st)) {
      fclose(stream);
      errno = ENODEV;
      return nullptr;
    }
    return stream;
  }
  if (!path || !mode)
    return nullptr;

  int open_flags = InterposerContext::fopen_flags_from_mode(mode);
  if (int snapshot = open_proc_maps_snapshot(path, open_flags); snapshot >= 0)
    return fdopen(snapshot, mode);

  const char *actual = path;
  std::string redirected;
  if (!InterposerContext::in_construction) {
    redirected = InterposerContext::ctx.redirect_sysfs_path(path);
    if (redirected.empty())
      redirected = redirect_sys_dev_char(path);
    if (!redirected.empty())
      actual = redirected.c_str();
  }

  int fd = InterposerContext::real().openat(AT_FDCWD, actual, open_flags, 0644);
  if (fd < 0)
    return nullptr;
  return fdopen(fd, mode);
}

RJ_INTERPOSER_EXPORT FILE *fopen64(const char *path, const char *mode) { return fopen(path, mode); }

RJ_INTERPOSER_EXPORT FILE *freopen(const char *path, const char *mode, FILE *stream) {
  if (!path || !mode)
    return nullptr;
  if (InterposerContext::real().ready() && !rj_owns_interposer_state()) {
    int refuse_errno = 0;
    if (rj_child_must_refuse_open(AT_FDCWD, path, &refuse_errno)) {
      // POSIX freopen closes the original stream FIRST and does so even when opening
      // the replacement fails, so refusing must not leave the caller's descriptor
      // live -- it would otherwise outlive a failed freopen and survive the exec that
      // follows. fclose can set errno itself, so restore ours afterwards. Guarded
      // because the owner path below treats a null stream as reachable too.
      RJ_DIAGNOSTIC_PUSH
      RJ_DIAGNOSTIC_IGNORE_NONNULL_COMPARE
      if (stream)
        fclose(stream);
      RJ_DIAGNOSTIC_POP
      errno = refuse_errno;
      return nullptr;
    }
    // freopen cannot be expressed as fdopen of a pre-validated descriptor -- it must
    // rebind the caller's stream -- so validate AFTER the fact instead, closing the
    // classify/reopen window from the other side. A swapped symlink that lands a GPU
    // device here is closed before the stream is returned.
    FILE *reopened = InterposerContext::real().freopen(path, mode, stream);
    if (!reopened)
      return nullptr;
    struct stat st {};
    const int fd = fileno(reopened);
    if (fd >= 0 && InterposerContext::real().fstat_fn &&
        InterposerContext::real().fstat_fn(fd, &st) == 0 && rj_is_gpu_device_stat(st)) {
      fclose(reopened);
      errno = ENODEV;
      return nullptr;
    }
    return reopened;
  }
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
    auto drv = InterposerContext::ctx.driver();
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
/// @brief GPU metadata for a DRM file, answered by ITS backend.
/// @details The file's lease is immutable for its whole life, so this cannot drift
/// onto another backend the way a global lookup can. Falls back to the global helper
/// only for a file with no lease (which cannot happen for a synthetic node, but
/// keeps this total).
static std::optional<Sysfs::GpuInfo>
interposer_gpu_info_for(const InterposerContext::DrmFileToken &file) {
  if (!file)
    return std::nullopt;
  const uint32_t render_minor = InterposerContext::ctx.drm_render_minor(file);
  const auto &lease = file->backend_lease;
  if (const auto &local = lease.local()) {
    if (const Sysfs::GpuInfo *info = local->gpu_info_for_render_minor(render_minor))
      return *info;
    return std::nullopt;
  }
  if (const auto &remote = lease.remote()) {
    if (const Sysfs::GpuInfo *info = remote->gpu_info())
      return *info;
    return std::nullopt;
  }
  return interposer_gpu_info(render_minor);
}

static std::optional<Sysfs::GpuInfo> interposer_gpu_info(uint32_t render_minor) {
  {
    if (auto drv = InterposerContext::ctx.driver()) {
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
    auto drv = InterposerContext::ctx.driver();
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
  if (!rj_owns_interposer_state())
    return InterposerContext::real().stat(path, buf);
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
  if (!rj_owns_interposer_state())
    return InterposerContext::real().lstat(path, buf);
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
  if (!rj_owns_interposer_state())
    return InterposerContext::real().access(path, mode);
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
  if (!rj_owns_interposer_state())
    return InterposerContext::real().opendir(name);
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
  if (!rj_owns_interposer_state())
    return InterposerContext::real().fstat_fn(fd, buf);
  int rc = InterposerContext::real().fstat_fn(fd, buf);
  if (rc == 0 && InterposerContext::ctx.is_drm(fd)) {
    uint32_t render_minor = InterposerContext::ctx.drm_render_minor(fd);
    buf->st_rdev = makedev(226, render_minor);
    buf->st_mode = (buf->st_mode & ~S_IFMT) | S_IFCHR;
  }
  return rc;
}

RJ_INTERPOSER_EXPORT int fstat64(int fd, struct stat64 *buf) {
  auto &real = InterposerContext::real();
  if (!real.fstat64_fn) {
    // The alias table is resolved eagerly in init(); a call that lands before
    // that has no passthrough to use. Fail with an errno rather than -1 alone,
    // which would leave the caller reading a stale one.
    errno = ENOSYS;
    return -1;
  }
  int rc = real.fstat64_fn(fd, reinterpret_cast<void *>(buf));
  // Gate BEFORE is_drm(), which takes fd_mutex_. No lazy static above this point:
  // its initialization guard could be inherited mid-init by a forked child.
  if (rc != 0 || !real.ready() || !rj_owns_interposer_state())
    return rc;
  if (InterposerContext::ctx.is_drm(fd)) {
    uint32_t render_minor = InterposerContext::ctx.drm_render_minor(fd);
    buf->st_rdev = makedev(226, render_minor);
    buf->st_mode = (buf->st_mode & ~S_IFMT) | S_IFCHR;
  }
  return rc;
}

RJ_INTERPOSER_EXPORT int __fxstat(int ver, int fd, struct stat *buf) {
  auto &real = InterposerContext::real();
  if (!real.fxstat_fn) {
    // The alias table is resolved eagerly in init(); a call that lands before
    // that has no passthrough to use. Fail with an errno rather than -1 alone,
    // which would leave the caller reading a stale one.
    errno = ENOSYS;
    return -1;
  }
  int rc = real.fxstat_fn(ver, fd, buf);
  // Gate BEFORE is_drm(), which takes fd_mutex_. No lazy static above this point:
  // its initialization guard could be inherited mid-init by a forked child.
  if (rc != 0 || !real.ready() || !rj_owns_interposer_state())
    return rc;
  if (InterposerContext::ctx.is_drm(fd)) {
    uint32_t render_minor = InterposerContext::ctx.drm_render_minor(fd);
    buf->st_rdev = makedev(226, render_minor);
    buf->st_mode = (buf->st_mode & ~S_IFMT) | S_IFCHR;
  }
  return rc;
}

RJ_INTERPOSER_EXPORT int __fxstat64(int ver, int fd, struct stat64 *buf) {
  auto &real = InterposerContext::real();
  if (!real.fxstat64_fn) {
    // The alias table is resolved eagerly in init(); a call that lands before
    // that has no passthrough to use. Fail with an errno rather than -1 alone,
    // which would leave the caller reading a stale one.
    errno = ENOSYS;
    return -1;
  }
  int rc = real.fxstat64_fn(ver, fd, reinterpret_cast<void *>(buf));
  // Gate BEFORE is_drm(), which takes fd_mutex_. No lazy static above this point:
  // its initialization guard could be inherited mid-init by a forked child.
  if (rc != 0 || !real.ready() || !rj_owns_interposer_state())
    return rc;
  if (InterposerContext::ctx.is_drm(fd)) {
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
  if (!rj_owns_interposer_state())
    return InterposerContext::real().readlink_fn(path, buf, bufsiz);
  auto redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_sysfs_path(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return InterposerContext::real().readlink_fn(actual, buf, bufsiz);
}

// -- stat64/lstat64 interposition (distinct from stat on glibc 2.33+) --

RJ_INTERPOSER_EXPORT int stat64(const char *path, struct stat64 *buf) {
  auto &real = InterposerContext::real();
  if (!real.stat64_fn) {
    // The alias table is resolved eagerly in init(); a call that lands before
    // that has no passthrough to use. Fail with an errno rather than -1 alone,
    // which would leave the caller reading a stale one.
    errno = ENOSYS;
    return -1;
  }
  // Gate BEFORE redirect_sysfs_path(), which reaches published driver and fd state.
  // No lazy static above this point (see LibcPassthrough for why).
  if (!real.ready() || !rj_owns_interposer_state())
    return real.stat64_fn(path, reinterpret_cast<void *>(buf));
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  if (redirected.empty())
    return real.stat64_fn(path, reinterpret_cast<void *>(buf));
  return real.stat64_fn(redirected.c_str(), reinterpret_cast<void *>(buf));
}

RJ_INTERPOSER_EXPORT int lstat64(const char *path, struct stat64 *buf) {
  auto &real = InterposerContext::real();
  if (!real.lstat64_fn) {
    // The alias table is resolved eagerly in init(); a call that lands before
    // that has no passthrough to use. Fail with an errno rather than -1 alone,
    // which would leave the caller reading a stale one.
    errno = ENOSYS;
    return -1;
  }
  // Gate BEFORE redirect_sysfs_path(), which reaches published driver and fd state.
  // No lazy static above this point (see LibcPassthrough for why).
  if (!real.ready() || !rj_owns_interposer_state())
    return real.lstat64_fn(path, reinterpret_cast<void *>(buf));
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  if (redirected.empty())
    return real.lstat64_fn(path, reinterpret_cast<void *>(buf));
  return real.lstat64_fn(redirected.c_str(), reinterpret_cast<void *>(buf));
}

RJ_INTERPOSER_EXPORT int __xstat(int ver, const char *path, struct stat *buf) {
  auto &real = InterposerContext::real();
  if (!real.xstat_fn) {
    // The alias table is resolved eagerly in init(); a call that lands before
    // that has no passthrough to use. Fail with an errno rather than -1 alone,
    // which would leave the caller reading a stale one.
    errno = ENOSYS;
    return -1;
  }
  // Gate BEFORE redirect_sysfs_path(), which reaches published driver and fd state.
  // No lazy static above this point (see LibcPassthrough for why).
  if (!real.ready() || !rj_owns_interposer_state())
    return real.xstat_fn(ver, path, buf);
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  if (redirected.empty())
    return real.xstat_fn(ver, path, buf);
  return real.xstat_fn(ver, redirected.c_str(), buf);
}

RJ_INTERPOSER_EXPORT int __xstat64(int ver, const char *path, struct stat64 *buf) {
  auto &real = InterposerContext::real();
  if (!real.xstat64_fn) {
    // The alias table is resolved eagerly in init(); a call that lands before
    // that has no passthrough to use. Fail with an errno rather than -1 alone,
    // which would leave the caller reading a stale one.
    errno = ENOSYS;
    return -1;
  }
  // Gate BEFORE redirect_sysfs_path(), which reaches published driver and fd state.
  // No lazy static above this point (see LibcPassthrough for why).
  if (!real.ready() || !rj_owns_interposer_state())
    return real.xstat64_fn(ver, path, reinterpret_cast<void *>(buf));
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  if (redirected.empty())
    return real.xstat64_fn(ver, path, reinterpret_cast<void *>(buf));
  return real.xstat64_fn(ver, redirected.c_str(), reinterpret_cast<void *>(buf));
}

RJ_INTERPOSER_EXPORT int __lxstat(int ver, const char *path, struct stat *buf) {
  auto &real = InterposerContext::real();
  if (!real.lxstat_fn) {
    // The alias table is resolved eagerly in init(); a call that lands before
    // that has no passthrough to use. Fail with an errno rather than -1 alone,
    // which would leave the caller reading a stale one.
    errno = ENOSYS;
    return -1;
  }
  // Gate BEFORE redirect_sysfs_path(), which reaches published driver and fd state.
  // No lazy static above this point (see LibcPassthrough for why).
  if (!real.ready() || !rj_owns_interposer_state())
    return real.lxstat_fn(ver, path, buf);
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  if (redirected.empty())
    return real.lxstat_fn(ver, path, buf);
  return real.lxstat_fn(ver, redirected.c_str(), buf);
}

RJ_INTERPOSER_EXPORT int __lxstat64(int ver, const char *path, struct stat64 *buf) {
  auto &real = InterposerContext::real();
  if (!real.lxstat64_fn) {
    // The alias table is resolved eagerly in init(); a call that lands before
    // that has no passthrough to use. Fail with an errno rather than -1 alone,
    // which would leave the caller reading a stale one.
    errno = ENOSYS;
    return -1;
  }
  // Gate BEFORE redirect_sysfs_path(), which reaches published driver and fd state.
  // No lazy static above this point (see LibcPassthrough for why).
  if (!real.ready() || !rj_owns_interposer_state())
    return real.lxstat64_fn(ver, path, reinterpret_cast<void *>(buf));
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  if (redirected.empty())
    return real.lxstat64_fn(ver, path, reinterpret_cast<void *>(buf));
  return real.lxstat64_fn(ver, redirected.c_str(), reinterpret_cast<void *>(buf));
}

// fork() is intentionally NOT interposed, and needs no wrapper: the child performs
// no rocJITsu work at all. Every interposed entry point compares owner_pid_ first
// and passes a child straight through to libc, which covers fork-family primitives
// that never bind our symbols anyway (system/popen/posix_spawn). See
// InterposerContext::init() for why local mode is fork-then-exec only.

} // extern "C"
