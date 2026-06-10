// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file interposer.cpp
/// @brief LD_PRELOAD interposer that redirects KFD syscalls to the simulated driver.
///
/// @details Intercepts open, close, ioctl, mmap, munmap, and fopen to route
/// /dev/kfd operations and sysfs topology reads through SimulatedDriver.
/// All mutable state is consolidated in InterposerContext.

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/kmd/linux/remote_driver.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/kmd/linux/simulated_driver.h"
#include "rocjitsu/kmd/linux/sysfs.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"
#include "rocjitsu/vm/plugins/profiled_execution_plugin_group.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"

#include "util/dynamic_loader.h"
#include "util/log.h"

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
    remote_kfd_fd_ = -1;
    new (&init_mutex_) std::mutex();
    new (&fd_mutex_) std::mutex();
    sysfs_fds_.clear();
    drm_fds_.clear();
    kfd_dup_fds_.clear();
    in_construction = false;
  }

  SimulatedDriver *driver() { return rj_vm_ ? rj_vm_->vm->driver() : nullptr; }
  int driver_fd() {
    auto *d = driver();
    return d ? d->fd() : -1;
  }
  bool initialized() const { return rj_vm_ != nullptr || remote_ != nullptr; }

  /// @brief Get the remote driver instance, or nullptr if not connected.
  RemoteDriver *remote() { return remote_; }

  /// @brief Get the synthetic KFD fd for the remote driver.
  /// @retval >=0 Valid fd when connected to a daemon.
  /// @retval -1 Not connected.
  int remote_kfd_fd() const { return remote_kfd_fd_; }

  /// @brief Look up the remote driver by its KFD fd.
  /// @retval non-null If fd matches the remote KFD fd.
  /// @retval nullptr If fd doesn't match or no remote driver exists.
  RemoteDriver *remote_lookup(int fd) {
    return (fd >= 0 && fd == remote_kfd_fd_ && remote_) ? remote_ : nullptr;
  }

  /// @brief Get the daemon's sysfs topology directory path.
  /// @returns The topology path string, or empty if not connected.
  std::string remote_topology_path() {
    return remote_ ? std::string(remote_->topology_path()) : std::string{};
  }

  /// @brief Get the daemon's DRM sysfs directory path.
  /// @returns The DRM path string, or empty if not connected.
  std::string remote_drm_path() {
    return remote_ ? std::string(remote_->drm_path()) : std::string{};
  }

  /// @brief Connect to the daemon and perform the RPC handshake.
  /// @details Tries to connect to the daemon socket. If successful, creates
  /// a RemoteDriver, performs the handshake, and caches the instance.
  /// @retval non-null Connected remote driver.
  /// @retval nullptr No daemon running or handshake failed.
  RemoteDriver *get_or_create_remote() {
    if (remote_ && remote_kfd_fd_ >= 0)
      return remote_;
    int sock = connect_to_daemon();
    if (sock < 0)
      return nullptr;
    if (!remote_)
      remote_ = new RemoteDriver(sock);
    int fd = remote_->open();
    if (fd < 0)
      return nullptr;
    remote_kfd_fd_ = fd;
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

  bool is_kfd_primary(int fd) { return fd == driver_fd() || fd == remote_kfd_fd_; }

  bool is_kfd_dup(int fd) {
    std::lock_guard lock(fd_mutex_);
    return kfd_dup_fds_.count(fd) != 0;
  }

  bool is_kfd_tracked(int fd) { return is_kfd_primary(fd) || is_kfd_dup(fd); }

  void track_dup(int fd) {
    if (fd < 0 || is_kfd_primary(fd))
      return;
    std::lock_guard lock(fd_mutex_);
    kfd_dup_fds_.insert(fd);
  }

  void untrack_dup(int fd) {
    if (fd < 0)
      return;
    std::lock_guard lock(fd_mutex_);
    kfd_dup_fds_.erase(fd);
  }

  void clear_dups() {
    std::lock_guard lock(fd_mutex_);
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
  int remote_kfd_fd_ = -1;

  std::mutex init_mutex_;
  std::mutex fd_mutex_;
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

static void *(*real_dlsym_fn)(void *, const char *) = nullptr;

__attribute__((constructor(101))) static void resolve_real_dlsym() {
  real_dlsym_fn =
      reinterpret_cast<decltype(real_dlsym_fn)>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34"));
  if (!real_dlsym_fn)
    real_dlsym_fn =
        reinterpret_cast<decltype(real_dlsym_fn)>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));
  if (!real_dlsym_fn) {
    fprintf(stderr, "rocjitsu: failed to resolve real dlsym\n");
    abort();
  }
}

__attribute__((constructor)) static void init_interposer() { InterposerContext::init(); }

} // namespace

extern "C" {

static std::string redirect_sysfs_path(const char *path);
static std::string redirect_sys_dev_char(const char *path);
static const Sysfs::GpuInfo *interposer_gpu_info();

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

  if (std::string_view(path).starts_with("/dev/dri/renderD")) {
    if (!InterposerContext::ctx.remote_lookup(InterposerContext::ctx.remote_kfd_fd()) &&
        !InterposerContext::ctx.initialized())
      InterposerContext::ctx.get_or_create();
  }
  if (std::string_view(path).starts_with("/dev/dri/renderD") &&
      (InterposerContext::ctx.driver_fd() >= 0 || InterposerContext::ctx.remote_kfd_fd() >= 0)) {
    uint32_t render_minor = 128;
    auto minor_str = std::string_view(path).substr(16);
    if (!minor_str.empty())
      render_minor = static_cast<uint32_t>(std::atoi(minor_str.data()));
    auto raw_drm_fd = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_drm", MFD_CLOEXEC));
    if (raw_drm_fd >= 0) {
      int high_fd = fcntl(raw_drm_fd, F_DUPFD_CLOEXEC, 512);
      InterposerContext::real.close(raw_drm_fd);
      if (high_fd >= 0) {
        InterposerContext::ctx.track_drm(high_fd, render_minor);
        return high_fd;
      }
    }
    errno = EMFILE;
    return -1;
  }

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

  if (path[0] == '/') {
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
  if (auto *remote = InterposerContext::ctx.remote_lookup(fd))
    return remote->close();
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
    int rc = drv->close();
    InterposerContext::ctx.clear_dups();
    return rc;
  }
  if (InterposerContext::ctx.owns_fd(fd))
    return 0;
  return static_cast<int>(InterposerContext::real.close(fd));
}

int ioctl(int fd, unsigned long request, ...) {
  assert(InterposerContext::real.ready());
  va_list ap;
  va_start(ap, request);
  void *arg = va_arg(ap, void *);
  va_end(ap);

  constexpr unsigned kDrmIoctlType = 'd';
  constexpr unsigned kDrmIoctlNrVersion = 0x00;
  constexpr unsigned kDrmIoctlNrAmdgpuInfo = 0x45;
  constexpr unsigned kAmdgpuInfoDevInfo = 0x16;

  if (InterposerContext::ctx.is_drm(fd)) {
    unsigned nr = _IOC_NR(request);
    unsigned type = _IOC_TYPE(request);
    if (type == kDrmIoctlType && nr == kDrmIoctlNrVersion && arg) {
      struct drm_version {
        int version_major, version_minor, version_patchlevel;
        size_t name_len;
        char *name;
        size_t date_len;
        char *date;
        size_t desc_len;
        char *desc;
      };
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
    if (type == kDrmIoctlType && nr == kDrmIoctlNrAmdgpuInfo && arg) {
      struct drm_amdgpu_info {
        uint64_t return_pointer;
        uint32_t return_size;
        uint32_t query;
        uint64_t pad;
      };
      auto *info = static_cast<drm_amdgpu_info *>(arg);
      if (info->return_pointer && info->return_size > 0) {
        auto *out = reinterpret_cast<void *>(info->return_pointer);
        std::memset(out, 0, info->return_size);
        if (info->query == kAmdgpuInfoDevInfo) {
          struct drm_amdgpu_info_device {
            uint32_t device_id;
            uint32_t chip_rev;
            uint32_t external_rev;
            uint32_t pci_rev;
            uint32_t family;
            uint32_t num_shader_engines;
            uint32_t num_shader_arrays_per_engine;
            uint32_t gpu_counter_freq;
            uint64_t max_engine_clock;
            uint64_t max_memory_clock;
            uint32_t cu_active_number;
            uint32_t cu_ao_mask;
            uint32_t cu_bitmap[4][4];
            uint32_t enabled_rb_pipes_mask;
            uint32_t num_rb_pipes;
            uint32_t num_hw_gfx_contexts;
            uint32_t pad0;
            uint64_t ids_flags;
            uint64_t virtual_address_offset;
            uint64_t virtual_address_max;
            uint32_t virtual_address_alignment;
            uint32_t pte_fragment_size;
            uint32_t gart_page_size;
            uint32_t ce_ram_size;
            uint32_t vram_type;
            uint32_t vram_bit_width;
            uint32_t vce_harvest_config;
            uint32_t gc_double_offchip_lds_buf;
            uint64_t prim_buf_gpu_addr;
            uint64_t pos_buf_gpu_addr;
            uint64_t cntl_sb_buf_gpu_addr;
            uint64_t param_buf_gpu_addr;
            uint32_t wave_front_size;
            uint32_t num_shader_visible_vgprs;
            uint32_t num_cu_per_sh;
            uint32_t num_tcc_blocks;
            uint32_t gs_vgt_table_depth;
            uint32_t gs_prim_buffer_depth;
            uint32_t max_gs_waves_per_vgt;
            uint32_t pad1;
            uint64_t cu_ao_bitmap[4][4];
            uint64_t high_va_offset;
            uint64_t high_va_max;
            uint32_t pa_sc_tile_steering_override;
            uint64_t tcc_disabled_mask;
          };
          if (info->return_size >= sizeof(drm_amdgpu_info_device)) {
            auto *dev = static_cast<drm_amdgpu_info_device *>(out);
            auto *gpu = interposer_gpu_info();
            if (gpu) {
              dev->device_id = gpu->device_id;
              dev->family = gpu->family_id;
              dev->num_shader_engines = gpu->num_shader_engines;
              dev->num_shader_arrays_per_engine = gpu->num_shader_arrays_per_engine;
              dev->wave_front_size = gpu->wave_front_size;
              dev->num_cu_per_sh = gpu->num_cu_per_sh;
              dev->vram_type = 3; // HBM
              dev->vram_bit_width = gpu->mem_width;
              dev->cu_active_number =
                  gpu->num_shader_engines * gpu->num_shader_arrays_per_engine * gpu->num_cu_per_sh;
            }
          }
        }
      }
      return 0;
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

int fcntl(int fd, int cmd, ...) {
  assert(InterposerContext::real.ready());
  va_list ap;
  va_start(ap, cmd);
  FcntlArgKind kind = fcntl_arg_kind(cmd);
  long rc = 0;
  switch (kind) {
  case FcntlArgKind::Int: {
    int arg = va_arg(ap, int);
    rc = InterposerContext::real.fcntl(fd, cmd, arg);
    break;
  }
  case FcntlArgKind::Ptr: {
    void *arg = va_arg(ap, void *);
    rc = InterposerContext::real.fcntl(fd, cmd, arg);
    break;
  }
  case FcntlArgKind::None:
  default:
    rc = InterposerContext::real.fcntl(fd, cmd, 0L);
    break;
  }
  va_end(ap);

  if (rc >= 0 && (cmd == F_DUPFD || cmd == F_DUPFD_CLOEXEC)) {
    if (InterposerContext::ctx.is_kfd_tracked(fd))
      InterposerContext::ctx.track_dup(static_cast<int>(rc));
    else
      InterposerContext::ctx.untrack_dup(static_cast<int>(rc));
  }
  return static_cast<int>(rc);
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

// -- libdrm interposition --

int amdgpu_device_initialize(int /*fd*/, uint32_t *major_version, uint32_t *minor_version,
                             void **device_handle) {
  if (InterposerContext::ctx.driver_fd() < 0 && InterposerContext::ctx.remote_kfd_fd() < 0)
    return -1;
  *major_version = 3;
  *minor_version = 57;
  static int dummy_handle = 1;
  *device_handle = &dummy_handle;
  return 0;
}

int amdgpu_device_initialize2(int fd, bool /*deduplicate_device*/, uint32_t *major_version,
                              uint32_t *minor_version, void **device_handle) {
  return amdgpu_device_initialize(fd, major_version, minor_version, device_handle);
}

int amdgpu_device_deinitialize(void * /*device_handle*/) { return 0; }

int amdgpu_device_get_fd(void * /*device_handle*/) {
  int fd = InterposerContext::ctx.remote_kfd_fd();
  return fd >= 0 ? fd : InterposerContext::ctx.driver_fd();
}

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
  return nullptr;
}

static std::string redirect_dev_dri(const char *path) {
  if (!path || !InterposerContext::real.ready() || InterposerContext::in_construction)
    return {};
  std::string_view sv(path);
  if (sv != "/dev/dri" && sv != "/dev/dri/")
    return {};
  std::string drm_base;
  auto *drv = InterposerContext::ctx.driver();
  if (drv)
    drm_base = drv->topology().drm_path();
  else
    drm_base = InterposerContext::ctx.remote_drm_path();
  if (drm_base.empty())
    return {};
  return drm_base + "/dev_dri";
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

// -- DRM device enumeration (direct PLT linkage consumers) --

struct drmPciBusInfo {
  uint16_t domain;
  uint8_t bus;
  uint8_t dev;
  uint8_t func;
};

struct drmPciDeviceInfo {
  uint16_t vendor_id;
  uint16_t device_id;
  uint16_t subvendor_id;
  uint16_t subdevice_id;
  uint8_t revision_id;
};

struct drmDevice {
  char **nodes;
  int available_nodes;
  int bustype;
  union {
    drmPciBusInfo *pci;
  } businfo;
  union {
    drmPciDeviceInfo *pci;
  } deviceinfo;
};

static constexpr int DRM_NODE_PRIMARY = 0;
static constexpr int DRM_NODE_RENDER = 2;
static constexpr int DRM_BUS_PCI = 0;

static std::unordered_set<void *> our_drm_allocs;
static std::mutex drm_alloc_mutex;

static drmDevice *alloc_drm_device(const Sysfs::GpuInfo &gpu, uint32_t card_idx) {
  uint32_t render_minor = gpu.drm_render_minor;
  char primary_path[64], render_path[64];
  snprintf(primary_path, sizeof(primary_path), "/dev/dri/card%u", card_idx);
  snprintf(render_path, sizeof(render_path), "/dev/dri/renderD%u", render_minor);

  size_t primary_len = strlen(primary_path) + 1;
  size_t render_len = strlen(render_path) + 1;
  constexpr int kMaxNodeTypes = 3;

  size_t alloc_size = sizeof(drmDevice) + kMaxNodeTypes * sizeof(char *) + primary_len +
                      render_len + sizeof(drmPciBusInfo) + sizeof(drmPciDeviceInfo);
  auto *mem = static_cast<uint8_t *>(calloc(1, alloc_size));
  if (!mem)
    return nullptr;

  auto *dev = reinterpret_cast<drmDevice *>(mem);
  auto *nodes = reinterpret_cast<char **>(mem + sizeof(drmDevice));
  auto *str_buf = reinterpret_cast<char *>(nodes + kMaxNodeTypes);
  auto *bus = reinterpret_cast<drmPciBusInfo *>(str_buf + primary_len + render_len);
  auto *pci_dev = reinterpret_cast<drmPciDeviceInfo *>(bus + 1);

  memcpy(str_buf, primary_path, primary_len);
  memcpy(str_buf + primary_len, render_path, render_len);

  nodes[DRM_NODE_PRIMARY] = str_buf;
  nodes[DRM_NODE_RENDER] = str_buf + primary_len;
  nodes[1] = nullptr;

  uint32_t bus_num = (gpu.location_id >> 8) & 0xFF;
  uint32_t dev_num = (gpu.location_id >> 3) & 0x1F;
  uint32_t func_num = gpu.location_id & 0x7;

  bus->domain = static_cast<uint16_t>(gpu.domain);
  bus->bus = static_cast<uint8_t>(bus_num);
  bus->dev = static_cast<uint8_t>(dev_num);
  bus->func = static_cast<uint8_t>(func_num);

  pci_dev->vendor_id = static_cast<uint16_t>(gpu.vendor_id);
  pci_dev->device_id = static_cast<uint16_t>(gpu.device_id);
  pci_dev->subvendor_id = static_cast<uint16_t>(gpu.vendor_id);
  pci_dev->subdevice_id = static_cast<uint16_t>(gpu.device_id);
  pci_dev->revision_id = 0;

  dev->nodes = nodes;
  dev->available_nodes = (1 << DRM_NODE_PRIMARY) | (1 << DRM_NODE_RENDER);
  dev->bustype = DRM_BUS_PCI;
  dev->businfo.pci = bus;
  dev->deviceinfo.pci = pci_dev;

  {
    std::lock_guard lock(drm_alloc_mutex);
    our_drm_allocs.insert(mem);
  }

  return dev;
}

void drmFreeDevice(drmDevice **device) {
  if (!device || !*device)
    return;
  bool ours;
  {
    std::lock_guard lock(drm_alloc_mutex);
    ours = our_drm_allocs.erase(*device) != 0;
  }
  if (ours) {
    free(*device);
    *device = nullptr;
    return;
  }
  using fn_t = void (*)(drmDevice **);
  static fn_t real_fn = util::lookup_symbol<fn_t>(RTLD_NEXT, "drmFreeDevice");
  if (real_fn)
    real_fn(device);
}

void drmFreeDevices(drmDevice **devices, int count) {
  for (int i = 0; i < count; ++i) {
    if (devices[i])
      drmFreeDevice(&devices[i]);
  }
}

int drmGetDevice(int fd, drmDevice **device) {
  if (!device)
    return -EINVAL;
  *device = nullptr;
  if (!InterposerContext::real.ready() || !InterposerContext::ctx.is_drm(fd))
    return -ENODEV;
  auto *gpu = interposer_gpu_info();
  if (!gpu)
    return -ENODEV;
  *device = alloc_drm_device(*gpu, 0);
  return *device ? 0 : -ENOMEM;
}

int drmGetDevice2(int fd, uint32_t /*flags*/, drmDevice **device) {
  return drmGetDevice(fd, device);
}

int drmGetDevices(drmDevice **devices, int max_devices) {
  if (!devices || max_devices <= 0)
    return 0;
  auto *gpu = interposer_gpu_info();
  if (!gpu)
    return 0;
  devices[0] = alloc_drm_device(*gpu, 0);
  return devices[0] ? 1 : 0;
}

int drmGetDevices2(uint32_t /*flags*/, drmDevice **devices, int max_devices) {
  return drmGetDevices(devices, max_devices);
}

// Intercept dlsym so that dlsym-based consumers (amdsmi) get our DRM
// device query implementations instead of libdrm's.  Without this,
// libdrm's internal drmGetDeviceFromDevId runs on our synthetic memfds
// and produces a corrupted drmDevice that crashes in drmFreeDevice.
// drmFreeDevice/drmFreeDevices are intentionally excluded: our own
// drmFreeDevice fallback calls dlsym(RTLD_NEXT, "drmFreeDevice") and
// including it here would cause infinite recursion.
void *dlsym(void *handle, const char *symbol) {
  if (InterposerContext::real.ready()) {
    static const std::unordered_map<std::string_view, void *> overrides = {
        {"drmGetDevice", reinterpret_cast<void *>(&drmGetDevice)},
        {"drmGetDevice2", reinterpret_cast<void *>(&drmGetDevice2)},
        {"drmGetDevices", reinterpret_cast<void *>(&drmGetDevices)},
        {"drmGetDevices2", reinterpret_cast<void *>(&drmGetDevices2)},
    };
    auto it = overrides.find(symbol);
    if (it != overrides.end())
      return it->second;
  }
  if (real_dlsym_fn)
    return real_dlsym_fn(handle, symbol);
  // Called before resolve_real_dlsym constructor runs (e.g., during
  // dynamic linker symbol resolution at library load time).  Resolve
  // the real dlsym now and forward.
  resolve_real_dlsym();
  return real_dlsym_fn(handle, symbol);
}

pid_t fork() {
  assert(InterposerContext::real.ready());
  pid_t pid = InterposerContext::real.fork();
  if (pid == 0)
    InterposerContext::ctx.reset_after_fork();
  return pid;
}

} // extern "C"
