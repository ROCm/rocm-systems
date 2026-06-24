// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file interposer.cpp
/// @brief LD_PRELOAD interposer that redirects KFD syscalls to the simulated driver.
///
/// @details Intercepts open, close, ioctl, mmap, munmap, and fopen to route
/// /dev/kfd operations and sysfs topology reads through SimulatedDriver.
/// All mutable state is consolidated in InterposerContext.

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/remote_driver.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/kmd/linux/simulated_driver.h"
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
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <linux/memfd.h>
#include <mutex>
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
#include <unordered_set>

extern "C" rocjitsu::ExecutionPlugin *createKernelLoggingPlugin();
extern "C" rocjitsu::ExecutionPlugin *createRaceDetectorPlugin();

using rocjitsu::RemoteDriver;
using rocjitsu::SimulatedDriver;
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
    syscall(SYS_close, sock);
    return -1;
  }
  return sock;
}

namespace {

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

/// @brief Real libc function pointers resolved via dlsym(RTLD_NEXT).
/// @details Holds the original libc implementations that our LD_PRELOAD
/// interposer shadows. Resolved once at constructor time via resolve().
class LibcPassthrough {
public:
  int (*openat)(int, const char *, int, ...) = nullptr;
  int (*close)(int) = nullptr;
  int (*ioctl)(int, unsigned long, ...) = nullptr;
  void *(*mmap)(void *, size_t, int, int, int, off_t) = nullptr;
  int (*munmap)(void *, size_t) = nullptr;
  int (*mprotect)(void *, size_t, int) = nullptr;
  int (*madvise)(void *, size_t, int) = nullptr;
  int (*dup)(int) = nullptr;
  int (*dup2)(int, int) = nullptr;
  int (*dup3)(int, int, int) = nullptr;
  int (*fcntl)(int, int, ...) = nullptr;
  FILE *(*fopen)(const char *, const char *) = nullptr;
  FILE *(*freopen)(const char *, const char *, FILE *) = nullptr;
  DIR *(*opendir)(const char *) = nullptr;
  int (*stat)(const char *, struct stat *) = nullptr;
  int (*lstat)(const char *, struct stat *) = nullptr;
  int (*access)(const char *, int) = nullptr;
  int (*fstat_fn)(int, struct stat *) = nullptr;
  ssize_t (*readlink_fn)(const char *, char *, size_t) = nullptr;
  pid_t (*fork)() = nullptr;

  bool ready() const { return initialized_; }

  void resolve() {
    auto *handle = RTLD_NEXT;
    openat = util::lookup_symbol<decltype(openat)>(handle, "openat");
    close = util::lookup_symbol<decltype(close)>(handle, "close");
    ioctl = util::lookup_symbol<decltype(ioctl)>(handle, "ioctl");
    mmap = util::lookup_symbol<decltype(mmap)>(handle, "mmap");
    munmap = util::lookup_symbol<decltype(munmap)>(handle, "munmap");
    mprotect = util::lookup_symbol<decltype(mprotect)>(handle, "mprotect");
    madvise = util::lookup_symbol<decltype(madvise)>(handle, "madvise");
    dup = util::lookup_symbol<decltype(dup)>(handle, "dup");
    dup2 = util::lookup_symbol<decltype(dup2)>(handle, "dup2");
    dup3 = util::lookup_symbol<decltype(dup3)>(handle, "dup3");
    fcntl = util::lookup_symbol<decltype(fcntl)>(handle, "fcntl");
    fopen = util::lookup_symbol<decltype(fopen)>(handle, "fopen");
    freopen = util::lookup_symbol<decltype(freopen)>(handle, "freopen");
    opendir = util::lookup_symbol<decltype(opendir)>(handle, "opendir");
    stat = util::lookup_symbol<decltype(stat)>(handle, "stat");
    lstat = util::lookup_symbol<decltype(lstat)>(handle, "lstat");
    access = util::lookup_symbol<decltype(access)>(handle, "access");
    fstat_fn = util::lookup_symbol<decltype(fstat_fn)>(handle, "fstat");
    readlink_fn = util::lookup_symbol<decltype(readlink_fn)>(handle, "readlink");
    fork = util::lookup_symbol<decltype(fork)>(handle, "fork");
    assert(openat && close && ioctl && mmap && munmap && mprotect && madvise);
    assert(dup && dup2 && fcntl && fopen && freopen && opendir && fork);
    assert(stat && lstat && access && fstat_fn && readlink_fn);
    initialized_ = true;
  }

private:
  bool initialized_ = false;
};

/// @brief All mutable interposer state.
class InterposerContext {
public:
  static inline std::atomic<bool> in_construction{false};
  static inline LibcPassthrough real{};
  static InterposerContext &ctx;

  static void init() {
    new (storage_) InterposerContext();
    real.resolve();
  }

  /// @brief Reset interposer state in a forked child process.
  /// @details After fork(), the child inherits the parent's address space but
  /// the engine thread is dead (only the calling thread survives). Mutexes may
  /// be locked by threads that no longer exist. We reinitialize everything so
  /// the next open("/dev/kfd") creates a fresh connection.
  void reset_after_fork() {
    rj_vm_ = nullptr;
    remote_ = nullptr;
    remote_kfd_fd_.store(-1, std::memory_order_relaxed);
    remote_open_refs_.store(0, std::memory_order_relaxed);
    new (&init_mutex_) std::mutex();
    new (&fd_mutex_) std::mutex();
    new (&remote_mutex_) std::mutex();
    sysfs_fds_.clear();
    drm_fds_.clear();
    handle_to_drm_fd_.clear();
    kfd_dup_fds_.clear();
    in_construction = false;
  }

  SimulatedDriver *driver() { return rj_vm_ ? rj_vm_->vm->driver() : nullptr; }
  int driver_fd() {
    auto *d = driver();
    return d ? d->fd() : -1;
  }
  bool initialized() const { return rj_vm_ != nullptr || remote_ != nullptr; }

  std::unique_lock<std::mutex> lock_remote() { return std::unique_lock(remote_mutex_); }

  RemoteDriver *remote() { return remote_; }

  int remote_kfd_fd() const { return remote_kfd_fd_.load(std::memory_order_acquire); }

  RemoteDriver *remote_lookup(int fd) {
    return (fd >= 0 && fd == remote_kfd_fd_.load(std::memory_order_acquire) && remote_) ? remote_
                                                                                        : nullptr;
  }

  std::string remote_topology_path() {
    std::lock_guard lock(remote_mutex_);
    return remote_ ? std::string(remote_->topology_path()) : std::string{};
  }

  std::string remote_drm_path() {
    std::lock_guard lock(remote_mutex_);
    return remote_ ? std::string(remote_->drm_path()) : std::string{};
  }

  /// @brief True when a daemon-mode (remote) KFD connection is open.
  bool is_remote_mode() const { return remote_open_refs_.load(std::memory_order_acquire) > 0; }

  /// @brief Add one open reference for a remote (daemon-mode) KFD fd.
  /// @details Each live remote KFD fd (the primary plus every dup) holds one
  /// reference; the RPC connection is torn down only when the last reference is
  /// dropped. Mirrors SimulatedDriver's local open refcount for the daemon path.
  void retain_remote_open() { remote_open_refs_.fetch_add(1, std::memory_order_acq_rel); }

  /// @brief Drop one remote open reference, tearing down the connection on the
  /// last release.
  /// @details On the final release this sends RPC_CLOSE to the daemon (via
  /// RemoteDriver::close()) so the daemon frees this client's process state,
  /// rather than leaking it until socket disconnect at process exit.
  void release_remote_open() {
    int prev = remote_open_refs_.fetch_sub(1, std::memory_order_acq_rel);
    assert(prev > 0 && "remote open refcount underflow");
    if (prev == 1)
      teardown_remote();
  }

  RemoteDriver *get_or_create_remote() {
    std::lock_guard lock(remote_mutex_);
    if (remote_ && remote_kfd_fd_.load(std::memory_order_acquire) >= 0) {
      // Re-open of an already-connected daemon: each open holds one reference,
      // mirroring SimulatedDriver::open() retaining the local process.
      retain_remote_open();
      return remote_;
    }
    int sock = connect_to_daemon();
    if (sock < 0)
      return nullptr;
    if (!remote_)
      remote_ = new RemoteDriver(sock);
    int fd = remote_->open();
    if (fd < 0)
      return nullptr;
    remote_kfd_fd_.store(fd, std::memory_order_release);
    // The primary remote KFD fd holds the first open reference.
    remote_open_refs_.store(1, std::memory_order_release);
    return remote_;
  }

  SimulatedDriver *lookup(int fd) {
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

  bool is_kfd_primary(int fd) {
    return fd == driver_fd() || fd == remote_kfd_fd_.load(std::memory_order_acquire);
  }

  bool is_kfd_dup(int fd) {
    std::lock_guard lock(fd_mutex_);
    return kfd_dup_fds_.count(fd) != 0;
  }

  bool is_kfd_tracked(int fd) { return is_kfd_primary(fd) || is_kfd_dup(fd); }

  void track_dup(int fd) {
    if (fd < 0 || is_kfd_primary(fd))
      return;
    bool newly_tracked = false;
    {
      std::lock_guard lock(fd_mutex_);
      newly_tracked = kfd_dup_fds_.insert(fd).second;
    }
    if (!newly_tracked)
      return;
    // Each live KFD fd (primary + every dup) holds one open reference, so the
    // process/connection is torn down only when the last fd closes. Retain on
    // whichever backend is active (local SimulatedDriver or remote daemon RPC).
    if (auto *d = driver())
      d->retain_local_open();
    else if (is_remote_mode())
      retain_remote_open();
  }

  void untrack_dup(int fd) {
    if (fd < 0)
      return;
    bool was_tracked = false;
    {
      std::lock_guard lock(fd_mutex_);
      was_tracked = kfd_dup_fds_.erase(fd) != 0;
    }
    if (!was_tracked)
      return;
    if (auto *d = driver())
      d->close();
    else if (is_remote_mode())
      release_remote_open();
  }

  void clear_dups() {
    size_t released = 0;
    {
      std::lock_guard lock(fd_mutex_);
      released = kfd_dup_fds_.size();
      kfd_dup_fds_.clear();
    }
    // Drop the references the cleared dups were holding. Used when a fresh
    // open() rebinds the local process; the just-opened reference is preserved
    // because clear_dups runs before any new dups are tracked. Remote teardown
    // releases its own references explicitly and never routes through here.
    if (auto *d = driver())
      for (size_t i = 0; i < released; ++i)
        d->close();
  }

  /// @brief Tear down the remote (daemon) connection: send RPC_CLOSE, close the
  /// synthetic primary fd, and drop daemon-redirect topology paths.
  /// @details Invoked on the last remote open reference release. Any remaining
  /// dup-tracked fds refer to the now-closed synthetic fd; clear the set so
  /// their subsequent close()/ioctl calls fall through to the real syscall
  /// instead of being misrouted to a dead RPC connection.
  void teardown_remote() {
    std::lock_guard lock(remote_mutex_);
    if (!remote_)
      return;
    remote_->close();
    delete remote_;
    remote_ = nullptr;
    int fd = remote_kfd_fd_.exchange(-1, std::memory_order_acq_rel);
    if (fd >= 0)
      InterposerContext::real.close(fd);
    std::lock_guard fd_lock(fd_mutex_);
    kfd_dup_fds_.clear();
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

  int drm_fd_for_handle(void *handle) {
    std::lock_guard lock(fd_mutex_);
    auto it = handle_to_drm_fd_.find(handle);
    return (it != handle_to_drm_fd_.end()) ? it->second : -1;
  }

  void track_drm_handle(void *handle, int fd) {
    std::lock_guard lock(fd_mutex_);
    handle_to_drm_fd_[handle] = fd;
  }

  int untrack_drm_handle(void *handle) {
    std::lock_guard lock(fd_mutex_);
    auto it = handle_to_drm_fd_.find(handle);
    if (it == handle_to_drm_fd_.end())
      return -1;
    int fd = it->second;
    handle_to_drm_fd_.erase(it);
    return fd;
  }

  SimulatedDriver *get_or_create() {
    std::lock_guard lock(init_mutex_);
    if (!rj_vm_) {
      in_construction = true;
      auto cfg_file = rocjitsu::rpc_default_config_file_path();
      char cfg_buf[4096]{};
      int cfg_fd = static_cast<int>(syscall(SYS_openat, AT_FDCWD, cfg_file.c_str(), O_RDONLY, 0));
      if (cfg_fd < 0) {
        util::Logger::debug_print("rocjitsu: no config file at ", cfg_file);
        in_construction = false;
        return nullptr;
      }
      auto n = syscall(SYS_read, cfg_fd, cfg_buf, sizeof(cfg_buf) - 1);
      syscall(SYS_close, cfg_fd);
      if (n <= 0) {
        in_construction = false;
        return nullptr;
      }
      while (n > 0 && (cfg_buf[n - 1] == '\n' || cfg_buf[n - 1] == '\r'))
        cfg_buf[--n] = '\0';
      if (rj_vm_create(cfg_buf, RJ_VM_MODE_LOCAL, &rj_vm_) != ROCJITSU_STATUS_SUCCESS) {
        util::Logger::debug_print("rocjitsu: failed to create VM");
        in_construction = false;
        return nullptr;
      }

      // Set up execution plugins based on environment variables.
      if (rj_vm_->soc) {
        std::shared_ptr<rocjitsu::ExecutionPluginGroup> pg;
        if (std::getenv("RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP"))
          pg = std::make_shared<rocjitsu::ProfiledExecutionPluginGroup>();
        else
          pg = std::make_shared<rocjitsu::ExecutionPluginGroup>();

        std::string sinks_str = "stderr";
        if (const char *s = std::getenv("RJ_SINKS"))
          sinks_str = s;
        {
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

      std::thread([vm = rj_vm_]() { rj_vm_run(vm, nullptr); }).detach();
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
  RemoteDriver *remote_ = nullptr;
  std::atomic<int> remote_kfd_fd_{-1};
  /// @brief Open-reference count for the remote (daemon-mode) KFD connection.
  /// @details The primary remote fd and every dup of it each hold one
  /// reference; the RPC connection is torn down only when the last reference is
  /// released. Mirrors SimulatedDriver's local open refcount for daemon mode.
  std::atomic<int> remote_open_refs_{0};

  std::mutex init_mutex_;
  std::mutex fd_mutex_;
  std::mutex remote_mutex_;
  std::unordered_map<int, std::string> sysfs_fds_;
  std::unordered_map<int, uint32_t> drm_fds_;
  std::unordered_map<void *, int> handle_to_drm_fd_;
  std::unordered_set<int> kfd_dup_fds_;

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
static const Sysfs::GpuInfo *interposer_gpu_info();

struct SyntheticDrmOpenResult {
  bool handled = false;
  int fd = -1;
};

static SyntheticDrmOpenResult open_synthetic_drm_fd(const char *path) {
  static constexpr std::string_view kRenderPrefix = "/dev/dri/renderD";
  if (!path || !std::string_view(path).starts_with(kRenderPrefix))
    return {};

  if (!InterposerContext::ctx.remote_lookup(InterposerContext::ctx.remote_kfd_fd()) &&
      !InterposerContext::ctx.initialized())
    InterposerContext::ctx.get_or_create();

  if (InterposerContext::ctx.driver_fd() < 0 && InterposerContext::ctx.remote_kfd_fd() < 0)
    return {};

  uint32_t render_minor = 128;
  auto minor_str = std::string_view(path).substr(kRenderPrefix.size());
  if (!minor_str.empty()) {
    uint32_t parsed_minor = 0;
    auto *first = minor_str.data();
    auto *last = first + minor_str.size();
    if (auto result = std::from_chars(first, last, parsed_minor);
        result.ec == std::errc{} && result.ptr == last)
      render_minor = parsed_minor;
  }

  auto raw_drm_fd = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_drm", MFD_CLOEXEC));
  if (raw_drm_fd < 0)
    return {true, -1};

  int high_fd = fcntl(raw_drm_fd, F_DUPFD_CLOEXEC, 512);
  int saved_errno = errno;
  InterposerContext::real.close(raw_drm_fd);
  if (high_fd < 0) {
    errno = saved_errno;
    return {true, -1};
  }

  InterposerContext::ctx.track_drm(high_fd, render_minor);
  return {true, high_fd};
}

int open(const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }

  assert(InterposerContext::real.ready());
  auto *volatile p = path;
  if (!p || InterposerContext::in_construction)
    return static_cast<int>(syscall(SYS_openat, AT_FDCWD, path, flags, mode));

  if (auto drm_fd = open_synthetic_drm_fd(path); drm_fd.handled)
    return drm_fd.fd;

  if (std::strcmp(path, "/dev/kfd") == 0) {
    if (InterposerContext::ctx.get_or_create_remote())
      return InterposerContext::ctx.remote_kfd_fd();

    auto *drv = InterposerContext::ctx.get_or_create();
    if (!drv) {
      errno = ENODEV;
      return -1;
    }
    drv->open();
    InterposerContext::ctx.clear_dups();
    return InterposerContext::ctx.driver_fd();
  }

  if (std::string_view(path).starts_with("/sys/class/drm") ||
      std::string_view(path).starts_with("/sys/devices/virtual/kfd") ||
      std::string_view(path).starts_with("/sys/class/kfd")) {
    if (!InterposerContext::ctx.remote_lookup(InterposerContext::ctx.remote_kfd_fd()) &&
        !InterposerContext::ctx.initialized())
      InterposerContext::ctx.get_or_create();
  }
  std::string redirected;
  auto remote_topo = InterposerContext::ctx.remote_topology_path();
  if (!remote_topo.empty()) {
    std::string_view sv(path);
    constexpr const char *kfd_prefix = "/sys/devices/virtual/kfd/kfd/topology";
    constexpr const char *kfd_alt = "/sys/class/kfd/kfd/topology";
    constexpr const char *drm_prefix = "/sys/class/drm";
    if (sv.starts_with(kfd_prefix))
      redirected = remote_topo + std::string(sv.substr(std::strlen(kfd_prefix)));
    else if (sv.starts_with(kfd_alt))
      redirected = remote_topo + std::string(sv.substr(std::strlen(kfd_alt)));
    else if (sv.starts_with(drm_prefix)) {
      auto remote_drm = InterposerContext::ctx.remote_drm_path();
      if (!remote_drm.empty())
        redirected = remote_drm + std::string(sv.substr(std::strlen(drm_prefix)));
    }
  }
  if (redirected.empty())
    redirected = InterposerContext::ctx.redirect(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (!redirected.empty()) {
    int fd = InterposerContext::real.openat(AT_FDCWD, redirected.c_str(), flags, mode);
    if (fd >= 0)
      InterposerContext::ctx.track_sysfs(fd, redirected);
    return fd;
  }

  return InterposerContext::real.openat(AT_FDCWD, path, flags, mode);
}

int open64(const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }
  return open(path, flags, mode);
}

int __open_2(const char *path, int oflag) { return open(path, oflag, 0); }

int __open64_2(const char *path, int oflag) { return open(path, oflag, 0); }

int __openat_2(int dirfd, const char *path, int oflag) { return openat(dirfd, path, oflag, 0); }

int __openat64_2(int dirfd, const char *path, int oflag) { return openat(dirfd, path, oflag, 0); }

int openat(int dirfd, const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }

  auto *volatile p_at = path;
  if (!p_at)
    return InterposerContext::real.openat(dirfd, path, flags, mode);
  if (InterposerContext::in_construction)
    return InterposerContext::real.openat(dirfd, path, flags, mode);

  if (path[0] == '/') {
    if (auto drm_fd = open_synthetic_drm_fd(path); drm_fd.handled)
      return drm_fd.fd;

    if (std::string_view(path).starts_with("/sys/class/drm") ||
        std::string_view(path).starts_with("/sys/devices/virtual/kfd") ||
        std::string_view(path).starts_with("/sys/class/kfd")) {
      if (!InterposerContext::ctx.remote_lookup(InterposerContext::ctx.remote_kfd_fd()) &&
          !InterposerContext::ctx.initialized())
        InterposerContext::ctx.get_or_create();
    }
    std::string redirected;
    auto remote_topo_at = InterposerContext::ctx.remote_topology_path();
    if (!remote_topo_at.empty()) {
      std::string_view sv(path);
      constexpr const char *kfd_p = "/sys/devices/virtual/kfd/kfd/topology";
      constexpr const char *kfd_a = "/sys/class/kfd/kfd/topology";
      constexpr const char *drm_p = "/sys/class/drm";
      if (sv.starts_with(kfd_p))
        redirected = remote_topo_at + std::string(sv.substr(std::strlen(kfd_p)));
      else if (sv.starts_with(kfd_a))
        redirected = remote_topo_at + std::string(sv.substr(std::strlen(kfd_a)));
      else if (sv.starts_with(drm_p)) {
        auto remote_drm_at = InterposerContext::ctx.remote_drm_path();
        if (!remote_drm_at.empty())
          redirected = remote_drm_at + std::string(sv.substr(std::strlen(drm_p)));
      }
    }
    if (redirected.empty())
      redirected = InterposerContext::ctx.redirect(path);
    if (redirected.empty())
      redirected = redirect_sys_dev_char(path);
    if (!redirected.empty()) {
      int fd = InterposerContext::real.openat(AT_FDCWD, redirected.c_str(), flags, mode);
      if (fd >= 0)
        InterposerContext::ctx.track_sysfs(fd, redirected);
      return fd;
    }
  } else if (dirfd != AT_FDCWD) {
    auto dir_path = InterposerContext::ctx.lookup_sysfs(dirfd);
    if (!dir_path.empty()) {
      std::string full = dir_path + "/" + path;
      int fd = InterposerContext::real.openat(AT_FDCWD, full.c_str(), flags, mode);
      if (fd >= 0)
        InterposerContext::ctx.track_sysfs(fd, full);
      return fd;
    }
  }

  return InterposerContext::real.openat(dirfd, path, flags, mode);
}

int openat64(int dirfd, const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }
  return openat(dirfd, path, flags, mode);
}

int close(int fd) {
  assert(InterposerContext::real.ready());
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
    InterposerContext::real.close(fd);
    return 0;
  }
  if (InterposerContext::ctx.is_kfd_dup(fd)) {
    InterposerContext::ctx.untrack_dup(fd);
    return static_cast<int>(InterposerContext::real.close(fd));
  }
  if (auto *drv = InterposerContext::ctx.lookup(fd)) {
    drv->close();
    return 0;
  }
  if (InterposerContext::ctx.owns_fd(fd))
    return 0;
  return static_cast<int>(InterposerContext::real.close(fd));
}

__attribute__((destructor(101))) void rj_interposer_shutdown() {}

int ioctl(int fd, unsigned long request, ...) {
  assert(InterposerContext::real.ready());
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
      if (ver->name && ver->name_len >= sizeof(drv_name) - 1)
        std::memcpy(ver->name, drv_name, sizeof(drv_name));
      ver->name_len = sizeof(drv_name) - 1;
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
      auto *gpu = interposer_gpu_info();
      if (!gpu) {
        errno = ENODEV;
        return -1;
      }
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

  if (auto *remote = InterposerContext::ctx.remote_lookup(fd))
    return remote->ioctl(request, arg);
  if (InterposerContext::ctx.is_kfd_dup(fd)) {
    if (auto *remote = InterposerContext::ctx.remote_lookup(InterposerContext::ctx.remote_kfd_fd()))
      return remote->ioctl(request, arg);
  }
  // Late-ioctl safety net: an AMDKFD ('K') ioctl may arrive on a tracked KFD fd
  // whose primary remote handle changed underneath it (e.g. a close/dup race in
  // daemon mode). Forward only AMDKFD-typed ioctls, and only on fds we already
  // track as KFD (primary or dup), so an arbitrary unrelated fd carrying a
  // type-'K' ioctl is never misrouted to the remote KFD driver.
  if (_IOC_TYPE(request) == AMDKFD_IOCTL_BASE && InterposerContext::ctx.is_kfd_tracked(fd)) {
    if (auto *remote = InterposerContext::ctx.remote())
      return remote->ioctl(request, arg);
  }

  auto *drv = InterposerContext::ctx.lookup(fd);
  if (!drv && InterposerContext::ctx.is_kfd_dup(fd))
    drv = InterposerContext::ctx.driver();
  if (drv)
    return drv->ioctl(request, arg);

  return InterposerContext::real.ioctl(fd, request, arg);
}

int dup(int oldfd) {
  assert(InterposerContext::real.ready());
  int rc = InterposerContext::real.dup(oldfd);
  if (rc >= 0) {
    if (InterposerContext::ctx.is_kfd_tracked(oldfd))
      InterposerContext::ctx.track_dup(rc);
    else
      InterposerContext::ctx.untrack_dup(rc);
  }
  return rc;
}

int dup2(int oldfd, int newfd) {
  assert(InterposerContext::real.ready());
  InterposerContext::ctx.untrack_sysfs(newfd);
  InterposerContext::ctx.untrack_drm(newfd);
  InterposerContext::ctx.untrack_dup(newfd);
  int rc = InterposerContext::real.dup2(oldfd, newfd);
  if (rc >= 0) {
    if (InterposerContext::ctx.is_kfd_tracked(oldfd))
      InterposerContext::ctx.track_dup(rc);
    else
      InterposerContext::ctx.untrack_dup(rc);
  }
  return rc;
}

#ifdef SYS_dup3
int dup3(int oldfd, int newfd, int flags) {
  assert(InterposerContext::real.ready());
  InterposerContext::ctx.untrack_sysfs(newfd);
  InterposerContext::ctx.untrack_drm(newfd);
  InterposerContext::ctx.untrack_dup(newfd);
  int rc = InterposerContext::real.dup3(oldfd, newfd, flags);
  if (rc >= 0) {
    if (InterposerContext::ctx.is_kfd_tracked(oldfd))
      InterposerContext::ctx.track_dup(rc);
    else
      InterposerContext::ctx.untrack_dup(rc);
  }
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
// ABI, so InterposerContext::real.fcntl services both.
int fcntl_impl(int fd, int cmd, void *ptr_arg, int int_arg) {
  FcntlArgKind kind = fcntl_arg_kind(cmd);
  long rc = 0;
  switch (kind) {
  case FcntlArgKind::Int:
    rc = InterposerContext::real.fcntl(fd, cmd, int_arg);
    break;
  case FcntlArgKind::Ptr:
    rc = InterposerContext::real.fcntl(fd, cmd, ptr_arg);
    break;
  case FcntlArgKind::None:
  default:
    rc = InterposerContext::real.fcntl(fd, cmd, 0L);
    break;
  }

  if (rc >= 0 && (cmd == F_DUPFD || cmd == F_DUPFD_CLOEXEC)) {
    if (InterposerContext::ctx.is_kfd_tracked(fd))
      InterposerContext::ctx.track_dup(static_cast<int>(rc));
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
  return static_cast<int>(rc);
}
} // namespace

int fcntl(int fd, int cmd, ...) {
  assert(InterposerContext::real.ready());
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
int fcntl64(int fd, int cmd, ...) {
  assert(InterposerContext::real.ready());
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

void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  assert(InterposerContext::real.ready());
  if (auto *remote = InterposerContext::ctx.remote_lookup(fd))
    return remote->mmap(addr, length, prot, flags, offset);

  if (auto *drv = InterposerContext::ctx.lookup(fd))
    return drv->mmap(addr, length, prot, flags, offset);

  if (InterposerContext::ctx.is_drm(fd)) {
    if (auto *remote = InterposerContext::ctx.remote_lookup(InterposerContext::ctx.remote_kfd_fd()))
      return remote->mmap(addr, length, prot, flags, offset);
    if (auto *drv = InterposerContext::ctx.driver())
      return drv->mmap(addr, length, prot, flags, offset);
  }

  if (fd < 0 && (flags & MAP_FIXED) && prot != PROT_NONE && addr) {
    int memfd_out = -1;
    off_t memfd_offset = 0;
    if (InterposerContext::ctx.remote() && InterposerContext::ctx.remote()->find_memfd_for_addr(
                                               addr, length, &memfd_out, &memfd_offset)) {
      auto total = static_cast<off_t>(length) + memfd_offset;
      [[maybe_unused]] auto ft_rc = ftruncate(memfd_out, total);
      fallocate(memfd_out, 0, memfd_offset, static_cast<off_t>(length));
      auto *raw = InterposerContext::real.mmap(
          addr, length, prot, (flags & ~MAP_ANONYMOUS) | MAP_SHARED, memfd_out, memfd_offset);
      if (raw != MAP_FAILED) {
#ifdef MADV_POPULATE_WRITE
        InterposerContext::real.madvise(raw, length, MADV_POPULATE_WRITE);
#endif
        return raw;
      }
    }
  }
  return InterposerContext::real.mmap(addr, length, prot, flags, fd, offset);
}

int mprotect(void *addr, size_t length, int prot) {
  assert(InterposerContext::real.ready());
  auto *drv = InterposerContext::ctx.driver();
  if (drv && drv->is_doorbell_range(addr, length)) {
    errno = EPERM;
    return -1;
  }
  return InterposerContext::real.mprotect(addr, length, prot);
}

int madvise(void *addr, size_t length, int advice) {
  assert(InterposerContext::real.ready());
  if ((advice == MADV_HUGEPAGE || advice == MADV_DONTFORK) &&
      reinterpret_cast<uintptr_t>(addr) >= 0x1000000000ULL)
    return 0;
  return InterposerContext::real.madvise(addr, length, advice);
}

int munmap(void *addr, size_t length) {
  assert(InterposerContext::real.ready());
  if (auto *remote = InterposerContext::ctx.remote_lookup(InterposerContext::ctx.remote_kfd_fd())) {
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
  return InterposerContext::real.munmap(addr, length);
}

} // extern "C"

extern "C" {

// -- fopen / freopen interposition (sysfs redirect) --

FILE *fopen(const char *path, const char *mode) {
  if (!InterposerContext::real.ready()) {
    auto fn = util::lookup_symbol<FILE *(*)(const char *, const char *)>(RTLD_NEXT, "fopen");
    return fn ? fn(path, mode) : nullptr;
  }
  if (!path || !mode)
    return nullptr;

  const char *actual = path;
  std::string redirected;
  if (!InterposerContext::in_construction) {
    if (std::string_view(path).starts_with("/sys/class/drm") ||
        std::string_view(path).starts_with("/sys/devices/virtual/kfd") ||
        std::string_view(path).starts_with("/sys/class/kfd")) {
      if (!InterposerContext::ctx.remote_lookup(InterposerContext::ctx.remote_kfd_fd()) &&
          !InterposerContext::ctx.initialized())
        InterposerContext::ctx.get_or_create();
    }
    auto remote_topo_fp = InterposerContext::ctx.remote_topology_path();
    if (!remote_topo_fp.empty()) {
      std::string_view sv(path);
      constexpr const char *kp = "/sys/devices/virtual/kfd/kfd/topology";
      constexpr const char *ka = "/sys/class/kfd/kfd/topology";
      constexpr const char *dp = "/sys/class/drm";
      if (sv.starts_with(kp))
        redirected = remote_topo_fp + std::string(sv.substr(std::strlen(kp)));
      else if (sv.starts_with(ka))
        redirected = remote_topo_fp + std::string(sv.substr(std::strlen(ka)));
      else if (sv.starts_with(dp)) {
        auto remote_drm_fp = InterposerContext::ctx.remote_drm_path();
        if (!remote_drm_fp.empty())
          redirected = remote_drm_fp + std::string(sv.substr(std::strlen(dp)));
      }
    }
    if (redirected.empty())
      redirected = InterposerContext::ctx.redirect(path);
    if (redirected.empty())
      redirected = redirect_sys_dev_char(path);
    if (!redirected.empty())
      actual = redirected.c_str();
  }

  int fd = InterposerContext::real.openat(AT_FDCWD, actual,
                                          InterposerContext::fopen_flags_from_mode(mode), 0644);
  if (fd < 0)
    return nullptr;
  return fdopen(fd, mode);
}

FILE *fopen64(const char *path, const char *mode) { return fopen(path, mode); }

FILE *freopen(const char *path, const char *mode, FILE *stream) {
  if (!path || !mode)
    return nullptr;
  RJ_DIAGNOSTIC_PUSH
  RJ_DIAGNOSTIC_IGNORE_NONNULL_COMPARE
  if (stream)
    ::fclose(stream);
  RJ_DIAGNOSTIC_POP
  return fopen(path, mode);
}

FILE *freopen64(const char *path, const char *mode, FILE *stream) {
  return freopen(path, mode, stream);
}

// -- stat/lstat/access interposition --

static std::string redirect_sysfs_path(const char *path) {
  if (!path || !InterposerContext::real.ready() || InterposerContext::in_construction)
    return {};
  std::string_view sv(path);
  if (!sv.starts_with("/sys/class/drm") && !sv.starts_with("/sys/devices/virtual/kfd") &&
      !sv.starts_with("/sys/class/kfd"))
    return {};
  if (!InterposerContext::ctx.remote_lookup(InterposerContext::ctx.remote_kfd_fd()) &&
      !InterposerContext::ctx.initialized())
    InterposerContext::ctx.get_or_create();
  auto remote_topo = InterposerContext::ctx.remote_topology_path();
  if (!remote_topo.empty()) {
    constexpr const char *kp = "/sys/devices/virtual/kfd/kfd/topology";
    constexpr const char *ka = "/sys/class/kfd/kfd/topology";
    constexpr const char *dp = "/sys/class/drm";
    if (sv.starts_with(kp))
      return remote_topo + std::string(sv.substr(std::strlen(kp)));
    if (sv.starts_with(ka))
      return remote_topo + std::string(sv.substr(std::strlen(ka)));
    if (sv.starts_with(dp)) {
      auto remote_drm = InterposerContext::ctx.remote_drm_path();
      if (!remote_drm.empty())
        return remote_drm + std::string(sv.substr(std::strlen(dp)));
    }
  }
  auto fallback = InterposerContext::ctx.redirect(path);
  return fallback;
}

static std::string redirect_sys_dev_char(const char *path) {
  if (!path || !InterposerContext::real.ready() || InterposerContext::in_construction)
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
  if (drv)
    drm_base = drv->topology().drm_path();
  else
    drm_base = InterposerContext::ctx.remote_drm_path();
  if (drm_base.empty())
    return {};

  std::string entry = (minor_num >= 128) ? "renderD" + std::to_string(minor_num)
                                         : "card" + std::to_string(minor_num);
  std::string suffix;
  if (slash_pos != std::string_view::npos)
    suffix = std::string(after_colon.substr(slash_pos));

  return drm_base + "/" + entry + suffix;
}

static const Sysfs::GpuInfo *interposer_gpu_info() {
  auto *drv = InterposerContext::ctx.driver();
  if (drv)
    return &drv->topology().gpu_info();
  if (auto *remote = InterposerContext::ctx.remote_lookup(InterposerContext::ctx.remote_kfd_fd()))
    return remote->gpu_info();
  return nullptr;
}

static std::string redirect_dev_dri(const char *path) {
  if (!path || !InterposerContext::real.ready() || InterposerContext::in_construction)
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
  if (drv)
    drm_base = drv->topology().drm_path();
  else
    drm_base = InterposerContext::ctx.remote_drm_path();
  if (drm_base.empty())
    return {};
  if (is_dir)
    return drm_base + "/dev_dri";
  return drm_base + "/dev_dri/" + std::string(sv.substr(kDevDri.size()));
}

int stat(const char *path, struct stat *buf) {
  if (!InterposerContext::real.ready()) {
    auto fn = util::lookup_symbol<int (*)(const char *, struct stat *)>(RTLD_NEXT, "stat");
    return fn ? fn(path, buf) : -1;
  }
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  if (!redirected.empty())
    return InterposerContext::real.stat(redirected.c_str(), buf);
  return InterposerContext::real.stat(path, buf);
}

int lstat(const char *path, struct stat *buf) {
  if (!InterposerContext::real.ready()) {
    auto fn = util::lookup_symbol<int (*)(const char *, struct stat *)>(RTLD_NEXT, "lstat");
    return fn ? fn(path, buf) : -1;
  }
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  if (!redirected.empty())
    return InterposerContext::real.lstat(redirected.c_str(), buf);
  return InterposerContext::real.lstat(path, buf);
}

int access(const char *path, int mode) {
  if (!InterposerContext::real.ready()) {
    auto fn = util::lookup_symbol<int (*)(const char *, int)>(RTLD_NEXT, "access");
    return fn ? fn(path, mode) : -1;
  }
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_dev_dri(path);
  if (!redirected.empty())
    return InterposerContext::real.access(redirected.c_str(), mode);
  return InterposerContext::real.access(path, mode);
}

// -- opendir interposition --

DIR *opendir(const char *name) {
  if (!InterposerContext::real.ready()) {
    auto fn = util::lookup_symbol<DIR *(*)(const char *)>(RTLD_NEXT, "opendir");
    return fn ? fn(name) : nullptr;
  }
  auto *volatile p_od = name;
  if (!p_od) {
    errno = EINVAL;
    return nullptr;
  }
  if (!InterposerContext::in_construction) {
    if (std::string_view(name).starts_with("/sys/class/drm") ||
        std::string_view(name).starts_with("/sys/devices/virtual/kfd") ||
        std::string_view(name).starts_with("/sys/class/kfd")) {
      if (!InterposerContext::ctx.remote_lookup(InterposerContext::ctx.remote_kfd_fd()) &&
          !InterposerContext::ctx.initialized())
        InterposerContext::ctx.get_or_create();
    }
    std::string redirected;
    auto remote_topo_od = InterposerContext::ctx.remote_topology_path();
    if (!remote_topo_od.empty()) {
      std::string_view sv(name);
      constexpr const char *kp = "/sys/devices/virtual/kfd/kfd/topology";
      constexpr const char *ka = "/sys/class/kfd/kfd/topology";
      constexpr const char *dp = "/sys/class/drm";
      if (sv.starts_with(kp))
        redirected = remote_topo_od + std::string(sv.substr(std::strlen(kp)));
      else if (sv.starts_with(ka))
        redirected = remote_topo_od + std::string(sv.substr(std::strlen(ka)));
      else if (sv.starts_with(dp)) {
        auto remote_drm_od = InterposerContext::ctx.remote_drm_path();
        if (!remote_drm_od.empty())
          redirected = remote_drm_od + std::string(sv.substr(std::strlen(dp)));
      }
    }
    if (redirected.empty())
      redirected = InterposerContext::ctx.redirect(name);
    if (redirected.empty())
      redirected = redirect_sys_dev_char(name);
    if (redirected.empty())
      redirected = redirect_dev_dri(name);
    if (!redirected.empty())
      return InterposerContext::real.opendir(redirected.c_str());
  }
  return InterposerContext::real.opendir(name);
}

// -- fstat interposition (DRM memfd → synthetic st_rdev) --

int fstat(int fd, struct stat *buf) {
  if (!InterposerContext::real.ready()) {
    auto fn = util::lookup_symbol<int (*)(int, struct stat *)>(RTLD_NEXT, "fstat");
    return fn ? fn(fd, buf) : -1;
  }
  int rc = InterposerContext::real.fstat_fn(fd, buf);
  if (rc == 0 && InterposerContext::ctx.is_drm(fd)) {
    uint32_t render_minor = InterposerContext::ctx.drm_render_minor(fd);
    buf->st_rdev = makedev(226, render_minor);
    buf->st_mode = (buf->st_mode & ~S_IFMT) | S_IFCHR;
  }
  return rc;
}

int fstat64(int fd, struct stat64 *buf) {
  using fstat64_fn_t = int (*)(int, struct stat64 *);
  static fstat64_fn_t real_fstat64 = util::lookup_symbol<fstat64_fn_t>(RTLD_NEXT, "fstat64");
  if (!real_fstat64)
    return -1;
  int rc = real_fstat64(fd, buf);
  if (rc == 0 && InterposerContext::real.ready() && InterposerContext::ctx.is_drm(fd)) {
    uint32_t render_minor = InterposerContext::ctx.drm_render_minor(fd);
    buf->st_rdev = makedev(226, render_minor);
    buf->st_mode = (buf->st_mode & ~S_IFMT) | S_IFCHR;
  }
  return rc;
}

int __fxstat(int ver, int fd, struct stat *buf) {
  using fxstat_fn_t = int (*)(int, int, struct stat *);
  static fxstat_fn_t real_fxstat = util::lookup_symbol<fxstat_fn_t>(RTLD_NEXT, "__fxstat");
  if (!real_fxstat)
    return -1;
  int rc = real_fxstat(ver, fd, buf);
  if (rc == 0 && InterposerContext::real.ready() && InterposerContext::ctx.is_drm(fd)) {
    uint32_t render_minor = InterposerContext::ctx.drm_render_minor(fd);
    buf->st_rdev = makedev(226, render_minor);
    buf->st_mode = (buf->st_mode & ~S_IFMT) | S_IFCHR;
  }
  return rc;
}

int __fxstat64(int ver, int fd, struct stat64 *buf) {
  using fxstat64_fn_t = int (*)(int, int, struct stat64 *);
  static fxstat64_fn_t real_fxstat64 = util::lookup_symbol<fxstat64_fn_t>(RTLD_NEXT, "__fxstat64");
  if (!real_fxstat64)
    return -1;
  int rc = real_fxstat64(ver, fd, buf);
  if (rc == 0 && InterposerContext::real.ready() && InterposerContext::ctx.is_drm(fd)) {
    uint32_t render_minor = InterposerContext::ctx.drm_render_minor(fd);
    buf->st_rdev = makedev(226, render_minor);
    buf->st_mode = (buf->st_mode & ~S_IFMT) | S_IFCHR;
  }
  return rc;
}

// -- readlink interposition (redirect /sys/dev/char/) --

ssize_t readlink(const char *path, char *buf, size_t bufsiz) {
  if (!InterposerContext::real.ready()) {
    auto fn = util::lookup_symbol<ssize_t (*)(const char *, char *, size_t)>(RTLD_NEXT, "readlink");
    return fn ? fn(path, buf, bufsiz) : -1;
  }
  auto redirected = redirect_sys_dev_char(path);
  if (redirected.empty())
    redirected = redirect_sysfs_path(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return InterposerContext::real.readlink_fn(actual, buf, bufsiz);
}

// -- stat64/lstat64 interposition (distinct from stat on glibc 2.33+) --

int stat64(const char *path, struct stat64 *buf) {
  using stat64_fn_t = int (*)(const char *, struct stat64 *);
  static stat64_fn_t real_stat64 = util::lookup_symbol<stat64_fn_t>(RTLD_NEXT, "stat64");
  if (!real_stat64)
    return -1;
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return real_stat64(actual, buf);
}

int lstat64(const char *path, struct stat64 *buf) {
  using lstat64_fn_t = int (*)(const char *, struct stat64 *);
  static lstat64_fn_t real_lstat64 = util::lookup_symbol<lstat64_fn_t>(RTLD_NEXT, "lstat64");
  if (!real_lstat64)
    return -1;
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return real_lstat64(actual, buf);
}

int __xstat(int ver, const char *path, struct stat *buf) {
  using xstat_fn_t = int (*)(int, const char *, struct stat *);
  static xstat_fn_t real_xstat = util::lookup_symbol<xstat_fn_t>(RTLD_NEXT, "__xstat");
  if (!real_xstat)
    return -1;
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return real_xstat(ver, actual, buf);
}

int __xstat64(int ver, const char *path, struct stat64 *buf) {
  using xstat64_fn_t = int (*)(int, const char *, struct stat64 *);
  static xstat64_fn_t real_xstat64 = util::lookup_symbol<xstat64_fn_t>(RTLD_NEXT, "__xstat64");
  if (!real_xstat64)
    return -1;
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return real_xstat64(ver, actual, buf);
}

int __lxstat(int ver, const char *path, struct stat *buf) {
  using lxstat_fn_t = int (*)(int, const char *, struct stat *);
  static lxstat_fn_t real_lxstat = util::lookup_symbol<lxstat_fn_t>(RTLD_NEXT, "__lxstat");
  if (!real_lxstat)
    return -1;
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return real_lxstat(ver, actual, buf);
}

int __lxstat64(int ver, const char *path, struct stat64 *buf) {
  using lxstat64_fn_t = int (*)(int, const char *, struct stat64 *);
  static lxstat64_fn_t real_lxstat64 = util::lookup_symbol<lxstat64_fn_t>(RTLD_NEXT, "__lxstat64");
  if (!real_lxstat64)
    return -1;
  auto redirected = redirect_sysfs_path(path);
  if (redirected.empty())
    redirected = redirect_sys_dev_char(path);
  const char *actual = redirected.empty() ? path : redirected.c_str();
  return real_lxstat64(ver, actual, buf);
}

pid_t fork() {
  assert(InterposerContext::real.ready());
  pid_t pid = InterposerContext::real.fork();
  if (pid == 0)
    InterposerContext::ctx.reset_after_fork();
  return pid;
}

} // extern "C"
