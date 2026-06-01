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
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"

#include "util/dynamic_loader.h"
#include "util/log.h"

#include <cassert>
#include <cerrno>
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
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

using rocjitsu::RemoteDriver;
using rocjitsu::SimulatedDriver;

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
    fork = util::lookup_symbol<decltype(fork)>(handle, "fork");
    assert(openat && close && ioctl && mmap && munmap && mprotect && madvise);
    assert(dup && dup2 && fcntl && fopen && freopen && opendir && fork);
    assert(stat && lstat && access);
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

__attribute__((constructor)) static void init_interposer() { InterposerContext::init(); }

} // namespace

extern "C" {

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

  constexpr unsigned long kDrmIoctlVersion = 0xc0406400;
  constexpr unsigned long kDrmIoctlAmdgpuInfo = 0x40186445;

  if (InterposerContext::ctx.is_drm(fd)) {
    if (request == kDrmIoctlVersion && arg) {
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
    if (request == kDrmIoctlAmdgpuInfo && arg) {
      struct drm_amdgpu_info {
        uint64_t return_pointer;
        uint32_t return_size;
        uint32_t query;
        uint64_t pad;
      };
      auto *info = static_cast<drm_amdgpu_info *>(arg);
      if (info->return_pointer && info->return_size > 0)
        std::memset(reinterpret_cast<void *>(info->return_pointer), 0, info->return_size);
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
        InterposerContext::real.madvise(raw, length, MADV_POPULATE_WRITE);
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

int stat(const char *path, struct stat *buf) {
  if (!InterposerContext::real.ready()) {
    auto fn = util::lookup_symbol<int (*)(const char *, struct stat *)>(RTLD_NEXT, "stat");
    return fn ? fn(path, buf) : -1;
  }
  auto redirected = redirect_sysfs_path(path);
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
    if (!redirected.empty())
      return InterposerContext::real.opendir(redirected.c_str());
  }
  return InterposerContext::real.opendir(name);
}

pid_t fork() {
  assert(InterposerContext::real.ready());
  pid_t pid = InterposerContext::real.fork();
  if (pid == 0)
    InterposerContext::ctx.reset_after_fork();
  return pid;
}

} // extern "C"
