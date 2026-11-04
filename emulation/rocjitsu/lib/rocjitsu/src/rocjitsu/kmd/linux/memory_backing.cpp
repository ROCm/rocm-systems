// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file memory_backing.cpp
/// @brief Client for shared memory backing, exported tokens, and metadata files.

#include "rocjitsu/kmd/linux/memory_backing.h"

#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "rocjitsu/kmd/linux/memory_keeper.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <mutex>
#include <new>
#include <poll.h>
#include <sched.h>
#include <string>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <sys/xattr.h>
#include <unistd.h>

#if defined(__SANITIZE_ADDRESS__)
#define RJ_MEMORY_BACKING_WITH_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define RJ_MEMORY_BACKING_WITH_ASAN 1
#endif
#endif

#if defined(RJ_MEMORY_BACKING_WITH_ASAN)
#include <sanitizer/common_interface_defs.h>
#endif

namespace rocjitsu {
namespace {

bool identify(int fd, kmd::KeeperTokenIdentity &identity) {
  const int saved_errno = errno;
  sockaddr_un address{};
  socklen_t length = sizeof(address);
  bool found = host_call(libc_passthrough().getsockname, ::getsockname, fd,
                         reinterpret_cast<sockaddr *>(&address), &length) == 0 &&
               length == offsetof(sockaddr_un, sun_path) + 1 + sizeof(identity) &&
               address.sun_family == AF_UNIX && address.sun_path[0] == 0;
  if (found) {
    std::memcpy(&identity, address.sun_path + 1, sizeof(identity));
    found = identity.magic == kmd::kKeeperTokenMagic && identity.size > 0 && identity.shmid >= -1 &&
            (identity.mode == O_RDONLY || identity.mode == O_RDWR);
  }
  errno = saved_errno;
  return found;
}

// A token retains RAM only while its keeper serves the peer endpoint. Reject
// stale handles explicitly even if another mapping still retains the segment.
int check_token_liveness(int fd) {
  pollfd peer{fd, POLLRDHUP, 0};
  int result;
  do {
    result = host_call(libc_passthrough().poll, ::poll, &peer, 1, 0);
  } while (result < 0 && errno == EINTR);
  if (result < 0)
    return -1;
  if (peer.revents & POLLNVAL) {
    errno = EBADF;
    return -1;
  }
  if (peer.revents & (POLLHUP | POLLRDHUP | POLLERR)) {
    errno = ESTALE;
    return -1;
  }
  return 0;
}

int receive_result(int socket, bool expect_descriptor, bool *disconnected) {
  int descriptor = -1;
  int response = EIO;
  const long received =
      kmd::receive_keeper_descriptor(socket, &response, sizeof(response), descriptor);
  if (disconnected && (received == 0 || (received < 0 && (errno == EPIPE || errno == ECONNRESET))))
    *disconnected = true;
  const int error = received < 0 ? errno : (received == sizeof(response) ? response : EIO);
  if (error || (descriptor >= 0) != expect_descriptor) {
    host_close_fd(descriptor);
    errno = error ? error : EIO;
    return -1;
  }
  return expect_descriptor ? descriptor : 0;
}

int request_reopen(int socket, int mode,
                   kmd::KeeperOperation operation = kmd::KeeperOperation::Backing) {
  int reply[2];
  if (host_call(libc_passthrough().socketpair, ::socketpair, AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC,
                0, reply) != 0)
    return -1;
  kmd::KeeperRequest request{0, mode, operation, {}};
  const int result = kmd::send_keeper_descriptor(socket, &request, sizeof(request), reply[1]);
  int error = result == 0 ? 0 : errno;
  host_close_fd(reply[1]);
  int descriptor = -1;
  if (result == 0) {
    descriptor = receive_result(reply[0], true, nullptr);
    error = descriptor < 0 ? errno : 0;
  }
  host_close_fd(reply[0]);
  if (error) {
    host_close_fd(descriptor);
    errno = error;
    return -1;
  }
  return descriptor;
}

int request_allocation_flags(int socket, kmd::KeeperOperation operation, uint32_t *flags,
                             bool *present) {
  int reply[2];
  if (host_call(libc_passthrough().socketpair, ::socketpair, AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC,
                0, reply) != 0)
    return -1;
  kmd::KeeperRequest request{};
  request.operation = operation;
  request.size = operation == kmd::KeeperOperation::SetAllocationFlags ? *flags : 0;
  const int sent = kmd::send_keeper_descriptor(socket, &request, sizeof(request), reply[1]);
  const int send_error = errno;
  host_close_fd(reply[1]);
  if (sent != 0) {
    host_close_fd(reply[0]);
    errno = send_error;
    return -1;
  }
  kmd::KeeperAllocationFlagsReply response{};
  int unexpected_fd = -1;
  const long received =
      kmd::receive_keeper_descriptor(reply[0], &response, sizeof(response), unexpected_fd);
  const int receive_error = errno;
  host_close_fd(unexpected_fd);
  host_close_fd(reply[0]);
  if (received != sizeof(response) || unexpected_fd >= 0 || response.error != 0) {
    errno = received < 0 ? receive_error : response.error ? response.error : EPROTO;
    return -1;
  }
  if (response.present > 1) {
    errno = EPROTO;
    return -1;
  }
  *flags = response.flags;
  *present = response.present != 0;
  return 0;
}

struct AllocationFlagsXattr {
  uint32_t version = 1;
  uint32_t flags = 0;
};

std::string keeper_path() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<void *>(&create_memory_backing), &info) && info.dli_fname) {
    std::error_code error;
    const auto library = std::filesystem::canonical(info.dli_fname, error);
    if (!error) {
      // The same object is linked into the interposer, CLI, and unit tests.
      // Resolve each supported installed layout relative to that object.
      for (const char *relative : {RJ_MEMORY_KEEPER_INSTALL_PATH, RJ_MEMORY_KEEPER_BIN_INSTALL_PATH,
                                   RJ_MEMORY_KEEPER_TEST_INSTALL_PATH}) {
        const auto installed = library.parent_path() / relative;
        if (host_call(libc_passthrough().access, ::access, installed.c_str(), X_OK) == 0)
          return installed.string();
      }
      // Match actual build output directories, not arbitrary descendants: an
      // install staging prefix may itself live beneath the build directory.
      for (const char *directory : {RJ_MEMORY_KEEPER_BUILD_LIBDIR, RJ_MEMORY_KEEPER_BUILD_BINDIR,
                                    RJ_MEMORY_KEEPER_BUILD_TESTDIR}) {
        if (!*directory)
          continue;
        const auto build = std::filesystem::canonical(directory, error);
        if (!error && library.parent_path() == build)
          return RJ_MEMORY_KEEPER_BUILD_PATH;
      }
    }
  }
  // An installed object must not silently use a helper from another build.
  return {};
}

// The bootstrap and its child use only async-signal-safe libc operations. They
// must not enter the C++ allocator, application atfork hooks, or stdio locks.
[[noreturn]] void startup_failed(int channel, int error) {
  kmd::send_keeper_startup_reply(channel, error);
  host_call(libc_passthrough().exit_immediately, ::_exit, 1);
  __builtin_unreachable();
}

// NOFILE may have been lowered below descriptors inherited from the application.
// Enumerate the actual table without libc directory streams after the raw clone.
bool close_inherited_fds() {
  if (host_close_range(static_cast<unsigned>(kmd::kKeeperFirstUnreservedFd), UINT_MAX, 0u) == 0)
    return true;
  const int directory = host_call(libc_passthrough().openat, ::openat, AT_FDCWD, "/proc/self/fd",
                                  O_RDONLY | O_DIRECTORY | O_CLOEXEC, 0);
  if (directory < 0)
    return false;
  struct DirectoryEntry {
    uint64_t inode;
    int64_t offset;
    unsigned short record_size;
    unsigned char type;
    char name[1];
  };
  alignas(DirectoryEntry) char buffer[4096];
  int error = 0;
  while (!error) {
    const long bytes = host_getdents64(directory, buffer, sizeof(buffer));
    if (bytes < 0 && errno == EINTR)
      continue;
    if (bytes <= 0) {
      error = bytes < 0 ? errno : 0;
      break;
    }
    for (size_t position = 0; position < static_cast<size_t>(bytes);) {
      constexpr size_t name_offset = offsetof(DirectoryEntry, name);
      if (static_cast<size_t>(bytes) - position < name_offset + 1) {
        error = EIO;
        break;
      }
      const auto *entry = reinterpret_cast<const DirectoryEntry *>(buffer + position);
      if (entry->record_size <= name_offset || entry->record_size > bytes - position) {
        error = EIO;
        break;
      }
      const char *name = buffer + position + name_offset;
      unsigned number = 0;
      size_t index = 0;
      for (; index < entry->record_size - name_offset && name[index] >= '0' && name[index] <= '9';
           ++index) {
        const unsigned digit = name[index] - '0';
        if (number > (INT_MAX - digit) / 10)
          break;
        number = number * 10 + digit;
      }
      if (index && index < entry->record_size - name_offset && name[index] == '\0' &&
          number >= kmd::kKeeperFirstUnreservedFd && number != static_cast<unsigned>(directory))
        host_close_fd(static_cast<int>(number));
      position += entry->record_size;
    }
  }
  host_close_fd(directory);
  errno = error;
  return error == 0;
}

[[noreturn]] void exec_keeper(const char *listener_name, int lease, int startup, const char *path,
                              const char *owner) {
  // Preserve the incoming descriptors before moving them into fixed slots.
  const int saved_lease = host_call(libc_passthrough().fcntl, ::fcntl, lease, F_DUPFD_CLOEXEC,
                                    kmd::kKeeperFirstUnreservedFd);
  if (saved_lease < 0)
    startup_failed(startup, errno);
  const int saved_startup = host_call(libc_passthrough().fcntl, ::fcntl, startup, F_DUPFD_CLOEXEC,
                                      kmd::kKeeperFirstUnreservedFd);
  if (saved_startup < 0)
    startup_failed(startup, errno);
  if (host_call(libc_passthrough().dup3, ::dup3, saved_lease, kmd::kKeeperLeaseFd, 0) < 0 ||
      host_call(libc_passthrough().dup3, ::dup3, saved_startup, kmd::kKeeperStartupFd, 0) < 0)
    startup_failed(saved_startup, errno);
  for (int fd = 0; fd < kmd::kKeeperLeaseFd; ++fd)
    host_close_fd(fd);
  if (!close_inherited_fds())
    startup_failed(kmd::kKeeperStartupFd, errno);
  char *arguments[] = {const_cast<char *>(path), const_cast<char *>(owner),
                       const_cast<char *>(listener_name), nullptr};
  // The helper links its C++ runtime statically. Application preload and loader
  // configuration must not initialize a simulator inside the helper.
  // TSan's optional read-only shadow optimization writes a temporary file
  // before main. An unavailable temp directory skips that optimization while
  // retaining instrumentation, including with an inherited hard file limit of zero.
  char no_temporary_files[] = "TMPDIR=/dev/null";
  char *environment[] = {no_temporary_files, nullptr};
  host_call(libc_passthrough().execve, ::execve, path, arguments, environment);
  startup_failed(kmd::kKeeperStartupFd, errno);
}

constexpr size_t kBootstrapStackSize = 256 * 1024;

int clone_keeper(int (*entry)(void *), void *stack, int flags, void *argument) {
#if defined(RJ_MEMORY_BACKING_WITH_ASAN)
  // clone copies ASan's thread state but changes the child's stack. Each child
  // completes this switch at entry; the parent cancels its copy of the switch.
  void *fake_stack = nullptr;
  __sanitizer_start_switch_fiber(&fake_stack, stack, kBootstrapStackSize);
#endif
  const int child = host_call(libc_passthrough().clone, ::clone, entry,
                              static_cast<char *>(stack) + kBootstrapStackSize, flags, argument);
  const int error = errno;
#if defined(RJ_MEMORY_BACKING_WITH_ASAN)
  const void *original_stack = nullptr;
  size_t original_size = 0;
  __sanitizer_finish_switch_fiber(nullptr, &original_stack, &original_size);
  __sanitizer_start_switch_fiber(nullptr, original_stack, original_size);
  __sanitizer_finish_switch_fiber(fake_stack, nullptr, nullptr);
#endif
  errno = error;
  return child;
}

struct KeeperBootstrap {
  const char *listener_name;
  int lease;
  int startup;
  int owner_startup;
  const char *path;
  const char *owner;
  void *child_stack;
};

int enter_keeper(void *argument) {
#if defined(RJ_MEMORY_BACKING_WITH_ASAN)
  __sanitizer_finish_switch_fiber(nullptr, nullptr, nullptr);
#endif
  const auto &bootstrap = *static_cast<KeeperBootstrap *>(argument);
  exec_keeper(bootstrap.listener_name, bootstrap.lease, bootstrap.startup, bootstrap.path,
              bootstrap.owner);
}

int detach_keeper(void *argument) {
#if defined(RJ_MEMORY_BACKING_WITH_ASAN)
  __sanitizer_finish_switch_fiber(nullptr, nullptr, nullptr);
#endif
  const auto &bootstrap = *static_cast<KeeperBootstrap *>(argument);
  host_close_fd(bootstrap.owner_startup);
  if (host_call(libc_passthrough().setsid, ::setsid) < 0)
    startup_failed(bootstrap.startup, errno);
  if (clone_keeper(enter_keeper, bootstrap.child_stack, CLONE_UNTRACED | SIGCHLD, argument) < 0)
    startup_failed(bootstrap.startup, errno);
  return 0;
}

std::atomic<pid_t> warned_subreaper{};

void finish_private_request(int socket) {
  // A forked child can retain this socket. Signal EOF to the keeper even if
  // that child keeps its descriptor open.
  const int error = errno;
  ::shutdown(socket, SHUT_WR);
  errno = error;
}

class Keeper {
public:
  bool owns_current_process() const {
    return owner_ == getpid() && user_ == geteuid() && group_ == getegid();
  }

  template <typename Function> int visit_locked(bool &retry, Function &&function) {
    std::lock_guard lock(mutex_);
    // A caller may have obtained this pointer just before a credential change
    // replaced it. Never restart an unreachable helper after retirement.
    retry = retired_;
    if (retry)
      return -1;
    return function(*this);
  }

  int prepare() {
    if (has_lease()) {
      if (probe_listener() == 0)
        return 0;
      if (errno != ECONNREFUSED)
        return -1;
      release_lease();
    }
    return start();
  }

  template <typename Function> int request(Function &&function) {
    for (unsigned attempt = 0; attempt < 2; ++attempt) {
      if (!has_lease() && start() != 0)
        return -1;
      const int socket = connect_socket();
      int result = -1;
      bool disconnected = false;
      if (socket >= 0 && host_call(libc_passthrough().fcntl, ::fcntl, socket, F_SETFL, 0) == 0)
        result = function(socket, disconnected);
      const int error = errno;
      host_close_fd(socket);
      if (attempt != 0 ||
          (socket < 0 ? error != ECONNREFUSED : !disconnected || !helper_disconnected())) {
        errno = error;
        return result;
      }
      release_lease();
    }
    __builtin_unreachable();
  }

  bool retire() {
    if (owner_ != getpid()) {
      // The lease mapping is MADV_DONTFORK. Neither its address nor the mutex
      // inherited from a vanished parent thread belongs to this child.
      return false;
    }
    std::lock_guard lock(mutex_);
    retired_ = true;
    release_lease();
    return true;
  }

private:
  static uint64_t process_inode(int fd) {
    struct stat info {};
    return host_call(libc_passthrough().fstat_fn, ::fstat, fd, &info) == 0 ? info.st_ino : 0;
  }

  bool helper_has_exited() const {
    const int process = host_pidfd_open(helper_pid_, 0);
    if (process < 0)
      return errno == ESRCH;
    pollfd exited{process, POLLIN, 0};
    const bool dead = host_call(libc_passthrough().poll, ::poll, &exited, 1, 0) > 0 &&
                      (exited.revents & (POLLIN | POLLHUP));
    const uint64_t inode = helper_inode_ ? process_inode(process) : 0;
    host_close_fd(process);
    return dead || (inode && inode != helper_inode_);
  }

  int connect_socket() const {
    const int socket = host_call(libc_passthrough().socket, ::socket, AF_UNIX,
                                 SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (socket < 0)
      return -1;
    if (host_call(libc_passthrough().connect, ::connect, socket,
                  reinterpret_cast<const sockaddr *>(&address_), length_) == 0) {
      // Anyone in this network namespace can bind the name after the keeper exits.
      // Authenticate the listener before sending it any backing descriptors.
      ucred peer{};
      socklen_t size = sizeof(peer);
      if (host_call(libc_passthrough().getsockopt, ::getsockopt, socket, SOL_SOCKET, SO_PEERCRED,
                    &peer, &size) == 0) {
        if (size == sizeof(peer) && peer.pid == helper_pid_ && peer.uid == user_ &&
            peer.gid == group_) {
          if (authenticate_peer(socket))
            return socket;
        } else {
          errno = ECONNREFUSED;
        }
      }
      finish_private_request(socket);
    }
    const int error = errno;
    host_close_fd(socket);
    // A full queue or an interrupted identity check can mean either a busy
    // keeper or a new listener at the old name. Retain a live helper's lease.
    const bool check_lifetime = error == EAGAIN || error == EPIPE || error == ECONNRESET;
    errno = check_lifetime && helper_has_exited() ? ECONNREFUSED : error;
    return -1;
  }

  bool authenticate_peer(int socket) const {
    // A recycled PID can have the same numeric credentials. A 64-bit pidfs
    // inode identifies the process that sent the trusted startup reply.
    if (helper_inode_) {
      const int process = host_pidfd_open(helper_pid_, 0);
      if (process >= 0) {
        const uint64_t inode = process_inode(process);
        host_close_fd(process);
        if (inode) {
          if (inode == helper_inode_)
            return true;
          errno = ECONNREFUSED;
          return false;
        }
      }
    }

    // Older kernels lack unique pidfd inodes, and the caller can run out of
    // descriptors. Ask for the launch cookie before sending any backing files.
    const kmd::KeeperRequest identify{0, O_RDWR, kmd::KeeperOperation::Identify, {}};
    if (kmd::send_keeper_descriptor(socket, &identify, sizeof(identify), -1) != 0)
      return false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    if (!poll_until(socket, POLLIN, deadline)) {
      errno = EAGAIN;
      return false;
    }
    kmd::KeeperCookie proof{};
    int descriptor = -1;
    const long bytes =
        kmd::receive_keeper_descriptor(socket, &proof, sizeof(proof), descriptor, MSG_DONTWAIT);
    host_close_fd(descriptor);
    if (bytes <= 0) {
      if (bytes == 0)
        errno = EPIPE;
      else if (errno == EPROTO)
        errno = ECONNREFUSED;
      return false;
    }
    if (bytes == sizeof(proof) && descriptor < 0 && proof == cookie_)
      return true;
    errno = ECONNREFUSED;
    return false;
  }

  int probe_listener() const {
    const int socket = connect_socket();
    if (socket < 0)
      return -1;
    finish_private_request(socket);
    host_close_fd(socket);
    return 0;
  }

  static short poll_until(int fd, short events, std::chrono::steady_clock::time_point deadline) {
    pollfd state{fd, events, 0};
    for (;;) {
      const auto remaining =
          std::chrono::ceil<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
      if (remaining.count() <= 0)
        return 0;
      const int ready = host_call(libc_passthrough().poll, ::poll, &state, 1,
                                  static_cast<int>(remaining.count()));
      if (ready >= 0 || errno != EINTR)
        return ready > 0 ? state.revents : 0;
    }
  }

  bool helper_disconnected() const {
    const int socket = connect_socket();
    if (socket < 0)
      return errno == ECONNREFUSED;
    // A live keeper replies to an empty batch. A stopped keeper can keep its
    // lease even when it cannot reply within the deadline.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    kmd::KeeperRequest request{0, O_RDWR, kmd::KeeperOperation::FlushFiles, {}};
    bool closed;
    if (kmd::send_keeper_descriptor(socket, &request, sizeof(request), -1) != 0)
      closed = errno == EPIPE || errno == ECONNRESET;
    else
      closed = poll_until(socket, POLLIN | POLLRDHUP, deadline) & (POLLHUP | POLLRDHUP | POLLERR);
    finish_private_request(socket);
    host_close_fd(socket);
    if (!closed)
      return false;

    // The helper can reject a private channel under memory pressure. During
    // exit it can close the channel before releasing the listener. Give it the
    // remainder of the probe budget to exit, then check the listener itself.
    const int process = host_pidfd_open(helper_pid_, 0);
    if (process >= 0) {
      poll_until(process, POLLIN, deadline);
      host_close_fd(process);
    }
    return probe_listener() < 0 && errno == ECONNREFUSED;
  }

  void release_lease() {
    if (lease_mapping_ != MAP_FAILED)
      host_call(libc_passthrough().munmap, ::munmap, lease_mapping_, lease_mapping_size_);
    lease_mapping_ = MAP_FAILED;
  }

  bool has_lease() const { return lease_mapping_ != MAP_FAILED; }

  int start() {
    const std::string path = keeper_path();
    const int result = launch(path.c_str());
    if (result != 0) {
      const int error = errno;
      release_lease();
      if (!warned_) {
        util::Logger::warn("Cannot start memory keeper '", path, "': ", std::strerror(error));
        warned_ = true;
      }
      errno = error;
    }
    return result;
  }

  int launch(const char *path) {
    if (host_call(libc_passthrough().access, ::access, path, X_OK) != 0)
      return -1;
    int subreaper = 0;
    host_call(libc_passthrough().prctl, ::prctl, PR_GET_CHILD_SUBREAPER, &subreaper, 0L, 0L, 0L);
    if ((subreaper || owner_ == 1 || getppid() == 1) &&
        warned_subreaper.exchange(owner_, std::memory_order_relaxed) != owner_)
      util::Logger::warn("Memory keeper child-wait isolation is unsupported for child "
                         "subreapers and PID-1 owners. PID 1 or the adopting subreaper must "
                         "reap exited keepers with waitpid(..., WNOHANG), including after "
                         "application children exit; use a reaping init such as tini in "
                         "containers. Do not block waiting for live keepers");
    uint64_t nonce = 0;
    long random_bytes;
    do {
      random_bytes = host_call(libc_passthrough().getrandom, ::getrandom, &nonce, sizeof(nonce), 0);
    } while (random_bytes < 0 && errno == EINTR);
    if (random_bytes != sizeof(nonce)) {
      if (random_bytes >= 0)
        errno = EIO;
      return -1;
    }
    address_.sun_family = AF_UNIX;
    const int name_length = std::snprintf(address_.sun_path + 1, sizeof(address_.sun_path) - 1,
                                          "rocjitsu-memory-%d-%016lx", owner_, nonce);
    length_ = offsetof(sockaddr_un, sun_path) + 1 + name_length;
    int leases[2];
    if (host_call(libc_passthrough().socketpair, ::socketpair, AF_UNIX,
                  SOCK_SEQPACKET | SOCK_CLOEXEC, 0, leases) != 0)
      return -1;
    // SCM_CREDENTIALS on the inherited startup channel identifies the ready helper
    // in the owner's PID namespace, even after the bootstrap has detached it.
    const int pass_credentials = 1;
    if (host_call(libc_passthrough().setsockopt, ::setsockopt, leases[0], SOL_SOCKET, SO_PASSCRED,
                  &pass_credentials, sizeof(pass_credentials)) != 0) {
      const int error = errno;
      host_close_fd(leases[0]);
      host_close_fd(leases[1]);
      errno = error;
      return -1;
    }
    const int lease_fd =
        host_call(libc_passthrough().memfd_create, ::memfd_create, "rocjitsu-owner", MFD_CLOEXEC);
    struct flock lock {};
    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_SET;
    int keeper_lease_fd = -1;
    if (lease_fd >= 0 &&
        host_call(libc_passthrough().fcntl, ::fcntl, lease_fd, F_OFD_SETLK, &lock) == 0) {
      lease_mapping_size_ = static_cast<size_t>(sysconf(_SC_PAGESIZE));
      // A PROT_NONE mapping pins the locked open-file description without sizing
      // or accessing the empty memfd. No long-lived fd number can be replaced by
      // application close_range/dup2 while retirement is releasing the lease.
      lease_mapping_ = host_call(libc_passthrough().mmap, ::mmap, nullptr, lease_mapping_size_,
                                 PROT_NONE, MAP_SHARED, lease_fd, 0);
      if (lease_mapping_ != MAP_FAILED &&
          host_call(libc_passthrough().madvise, ::madvise, lease_mapping_, lease_mapping_size_,
                    MADV_DONTFORK) == 0) {
        char path[64];
        std::snprintf(path, sizeof(path), "/proc/self/fd/%d", lease_fd);
        // dup would share the owner's OFD lock; the watcher needs a fresh open.
        keeper_lease_fd =
            host_call(libc_passthrough().openat, ::openat, AT_FDCWD, path, O_RDWR | O_CLOEXEC, 0);
      }
    }
    const int lease_error = errno;
    host_close_fd(lease_fd);
    if (keeper_lease_fd < 0) {
      host_close_fd(leases[0]);
      host_close_fd(leases[1]);
      errno = lease_error;
      return -1;
    }
    char owner_text[32];
    std::snprintf(owner_text, sizeof(owner_text), "%d", owner_);
    // The bootstrap never execs, so ordinary waits cannot reap it. Its helper
    // is detached unless the owner is a child subreaper (diagnosed above).
    // libc clone requires separate child stacks. Neither clone shares VM, so
    // the parent can release its stack copies immediately after the first clone.
    void *stacks =
        host_call(libc_passthrough().mmap, ::mmap, nullptr, 2 * kBootstrapStackSize,
                  PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
    if (stacks == MAP_FAILED) {
      const int error = errno;
      host_close_fd(keeper_lease_fd);
      host_close_fd(leases[0]);
      host_close_fd(leases[1]);
      errno = error;
      return -1;
    }
    KeeperBootstrap arguments{address_.sun_path + 1,
                              keeper_lease_fd,
                              leases[1],
                              leases[0],
                              path,
                              owner_text,
                              static_cast<char *>(stacks) + kBootstrapStackSize};
    const pid_t bootstrap = clone_keeper(detach_keeper, stacks, CLONE_UNTRACED, &arguments);
    const int error = errno;
    host_call(libc_passthrough().munmap, ::munmap, stacks, 2 * kBootstrapStackSize);
    host_close_fd(keeper_lease_fd);
    host_close_fd(leases[1]);
    if (bootstrap < 0) {
      host_close_fd(leases[0]);
      errno = error;
      return -1;
    }
    while (host_call(libc_passthrough().waitpid, ::waitpid, bootstrap, nullptr, __WCLONE) < 0 &&
           errno == EINTR) {
    }
    pollfd startup{leases[0], POLLIN, 0};
    long ready;
    do {
      ready = host_call(libc_passthrough().poll, ::poll, &startup, 1, 10000);
    } while (ready < 0 && errno == EINTR);
    if (ready <= 0) {
      host_close_fd(leases[0]);
      if (ready == 0)
        errno = ETIMEDOUT;
      return -1;
    }
    kmd::KeeperStartupReply reply{};
    int descriptor = -1;
    pid_t sender_pid = -1;
    const long received = kmd::receive_keeper_descriptor(leases[0], &reply, sizeof(reply),
                                                         descriptor, 0, nullptr, &sender_pid);
    int startup_error = reply.error;
    if (received < 0)
      startup_error = errno;
    else if (received == 0)
      startup_error = EIO;
    else if (received != sizeof(reply) || reply.version != kmd::kKeeperProtocolVersion ||
             descriptor >= 0 || sender_pid <= 0)
      startup_error = EPROTO;
    host_close_fd(leases[0]);
    host_close_fd(descriptor);
    if (startup_error) {
      errno = startup_error;
      return -1;
    }
    helper_pid_ = sender_pid;
    helper_inode_ = reply.process_inode;
    cookie_ = reply.cookie;
    return 0;
  }

  kmd::KeeperCookie cookie_{};
  uint64_t helper_inode_ = 0;
  pid_t helper_pid_ = -1;
  const pid_t owner_ = getpid();
  const uid_t user_ = geteuid();
  const gid_t group_ = getegid();
  std::mutex mutex_;
  sockaddr_un address_{};
  socklen_t length_ = 0;
  void *lease_mapping_ = MAP_FAILED;
  size_t lease_mapping_size_ = 0;
  bool warned_ = false;
  bool retired_ = false;
};

// Interposer builds retain the manager for the process lifetime; explicit hosts
// release it when unloading. The keeper watches its owner pidfd and lease, then
// exits once externally held tokens are gone.
// A forked child never locks a mutex inherited from the parent's threads.
std::atomic<Keeper *> keeper{};

#if defined(RJ_DISABLE_INTERPOSER)
__attribute__((destructor)) void release_keeper() {
  // Explicit hosts retire all callers before unloading. A forked child must
  // not destroy a mutex inherited from a vanished parent thread.
  auto *current = keeper.exchange(nullptr, std::memory_order_relaxed);
  if (current && current->retire())
    delete current;
}
#endif

Keeper *get_keeper() {
  auto *current = keeper.load(std::memory_order_acquire);
  if (!current || !current->owns_current_process()) {
    auto *candidate = new (std::nothrow) Keeper;
    if (!candidate) {
      errno = ENOMEM;
      return nullptr;
    }
    while (!keeper.compare_exchange_weak(current, candidate, std::memory_order_acq_rel)) {
      if (current && current->owns_current_process()) {
        delete candidate;
        return current;
      }
    }
    if (current)
      current->retire();
    current = candidate;
  }
  return current;
}

template <typename Function> int with_keeper(Function &&function) {
  for (;;) {
    auto *current = get_keeper();
    if (!current)
      return -1;
    bool retry = false;
    const int result = current->visit_locked(retry, function);
    if (!retry)
      return result;
  }
}

template <typename Function> int with_keeper_connection(Function &&function) {
  return with_keeper([&](Keeper &current) { return current.request(function); });
}

int allocate_backing(size_t size, int sized_file = -1,
                     kmd::KeeperOperation operation = kmd::KeeperOperation::Backing,
                     const char *name = kMemoryBackingDefaultName) {
  kmd::KeeperRequest request{size, O_RDWR, operation, {}};
  std::snprintf(request.name, sizeof(request.name), "%s", name);
  return with_keeper_connection([&](int socket, bool &disconnected) {
    if (kmd::send_keeper_descriptor(socket, &request, sizeof(request), sized_file) != 0) {
      disconnected = errno == EPIPE || errno == ECONNRESET;
      finish_private_request(socket);
      return -1;
    }
    return receive_result(socket, sized_file < 0, &disconnected);
  });
}

int populate_mapping(int fd, const void *data, size_t size) {
  // Even overwriting an already sized file with write() checks RLIMIT_FSIZE.
  // These private setup mappings have never been published to a simulated GPU.
  void *view = host_call(libc_passthrough().mmap, ::mmap, nullptr, size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
  if (view == MAP_FAILED)
    return -1;
  std::memcpy(reinterpret_cast<void *>(view), data, size);
  host_call(libc_passthrough().munmap, ::munmap, view, size);
  return 0;
}

int open_backing_file(const BackingFile &file, int flags = O_TRUNC) {
  return host_call(libc_passthrough().openat, ::openat, AT_FDCWD, file.path.c_str(),
                   O_RDWR | O_CREAT | O_CLOEXEC | flags, 0666);
}

// Unfinished entries use this sentinel; completed entries hold zero or an errno.
constexpr int kPendingFile = -1;

int send_backing_files(int socket, bool &disconnected, std::span<BackingFile> files,
                       size_t &attempted_files) {
  std::array<BackingFile *, kmd::kKeeperMaxBatchFiles> pending{};
  size_t count = 0;
  auto flush = [&] {
    if (!count)
      return 0;
    kmd::KeeperRequest request{count, O_RDWR, kmd::KeeperOperation::FlushFiles, {}};
    // Inline writes produce no replies until this flush. Blocking sends
    // therefore cannot deadlock on a full reply queue, even with small socket
    // buffers. The helper never blocks on an application reply socket.
    if (kmd::send_keeper_descriptor(socket, &request, sizeof(request), -1, nullptr, 0, 0) != 0) {
      disconnected = errno == EPIPE || errno == ECONNRESET;
      return -1;
    }
    kmd::KeeperFileResults results;
    int descriptor = -1;
    const long received =
        kmd::receive_keeper_descriptor(socket, &results, sizeof(results), descriptor);
    disconnected = received == 0 || (received < 0 && (errno == EPIPE || errno == ECONNRESET));
    if (received != sizeof(results) || descriptor >= 0 || results.count != count) {
      const int error = received < 0 ? errno : EPROTO;
      host_close_fd(descriptor);
      errno = error;
      return -1;
    }
    for (size_t file_index = 0; file_index < count; ++file_index)
      pending[file_index]->error = results.errors[file_index];
    count = 0;
    return 0;
  };
  for (size_t file_index = 0; file_index < files.size(); ++file_index) {
    BackingFile &file = files[file_index];
    // A replacement helper can inherit lower limits. Keep acknowledged
    // files intact and retry only requests without a result.
    if (file.error != kPendingFile)
      continue;
    const size_t size = file.contents.size();
    const bool inline_file = size <= kmd::kKeeperInlineFileSize;
    if (!inline_file && flush() != 0)
      return -1;
    // A new inode cannot alias pending writes. Complete those writes before
    // truncating an existing path, including its symlink or hard-link aliases.
    int fd = open_backing_file(file, count ? O_EXCL : O_TRUNC);
    if (fd < 0 && errno == EEXIST && count) {
      if (flush() != 0)
        return -1;
      fd = open_backing_file(file);
    }
    if (fd < 0 && errno == EMFILE) {
      // The batch socket may occupy the last descriptor. Finish earlier writes
      // before releasing it so the caller can open the remaining paths in order.
      if (flush() == 0)
        errno = EMFILE;
      return -1;
    }
    attempted_files = std::max(attempted_files, file_index + 1);
    if (fd < 0) {
      file.error = errno;
      continue;
    }
    if (size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
      file.error = EFBIG;
      host_close_fd(fd);
      continue;
    }
    kmd::KeeperRequest request{size, O_RDWR, kmd::KeeperOperation::BatchFile, {}};
    auto send_file = [&] {
      return kmd::send_keeper_descriptor(socket, &request, sizeof(request), fd,
                                         file.contents.data(), inline_file ? size : 0, 0);
    };
    int sent = send_file();
    // Descriptors queued on a Unix socket still count against RLIMIT_NOFILE.
    // Finish the outstanding writes before retrying with an empty queue.
    if (sent < 0 && errno == ETOOMANYREFS && count) {
      if (flush() != 0) {
        const int error = errno;
        host_close_fd(fd);
        errno = error;
        return -1;
      }
      sent = send_file();
    }
    int error = sent < 0 ? errno : 0;
    if (sent < 0)
      disconnected = error == EPIPE || error == ECONNRESET;
    else if (!inline_file && (receive_result(socket, false, &disconnected) < 0 ||
                              populate_mapping(fd, file.contents.data(), size) < 0))
      error = errno;
    host_close_fd(fd);
    if (disconnected) {
      errno = error;
      return -1;
    }
    if (error || !inline_file) {
      file.error = error;
      continue;
    }
    pending[count++] = &file;
    if (count == pending.size() && flush() != 0)
      return -1;
  }
  return flush();
}

} // namespace

int prepare_memory_keeper() {
  return with_keeper([](Keeper &current) { return current.prepare(); });
}

int create_memory_backing(size_t size, bool preallocate, const char *name) {
  if (!size || size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
    errno = EINVAL;
    return -1;
  }
  const char *force_token = std::getenv("RJ_TEST_FORCE_TOKEN_BACKING");
  const auto operation = force_token && std::strcmp(force_token, "1") == 0
                             ? kmd::KeeperOperation::TokenBacking
                         : preallocate ? kmd::KeeperOperation::PreallocatedBacking
                                       : kmd::KeeperOperation::Backing;
  return allocate_backing(size, -1, operation, name);
}

int populate_backing_file(int fd, const void *data, size_t size) {
  if (!size)
    return 0;
  if (size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
    errno = EFBIG;
    return -1;
  }
  // The caller owns the descriptor and may have opened it with O_APPEND.
  // Linux pwrite would append even with offset zero; a mapping ignores that flag.
  if (allocate_backing(size, fd, kmd::KeeperOperation::SizeFile) < 0)
    return -1;
  return populate_mapping(fd, data, size);
}

void create_backing_files(std::span<BackingFile> files) {
  while (!files.empty()) {
    BackingFile &first = files.front();
    if (first.contents.empty()) {
      // A local truncate needs only one descriptor. Complete preceding writes
      // first in case another path in this batch names the same inode.
      const int fd = open_backing_file(first);
      first.error = fd < 0 ? errno : 0;
      host_close_fd(fd);
      files = files.subspan(1);
      continue;
    }
    const auto empty = std::find_if(files.begin(), files.end(),
                                    [](const BackingFile &file) { return file.contents.empty(); });
    const std::span<BackingFile> batch = files.first(empty - files.begin());
    for (BackingFile &file : batch)
      file.error = kPendingFile;
    size_t attempted_files = 0;
    const int result = with_keeper_connection([&](int socket, bool &disconnected) {
      const int status = send_backing_files(socket, disconnected, batch, attempted_files);
      finish_private_request(socket);
      return status;
    });
    if (result < 0) {
      const int error = errno;
      for (BackingFile &file : batch.first(attempted_files))
        if (file.error == kPendingFile)
          file.error = error;
      // Paths normally open before population. Preserve their creation, truncation
      // and individual open errors even when the keeper connection fails first.
      for (BackingFile &file : batch.subspan(attempted_files)) {
        const int fd = open_backing_file(file);
        file.error = fd < 0 ? errno : error;
        if (fd >= 0 && error == EMFILE)
          file.error =
              populate_backing_file(fd, file.contents.data(), file.contents.size()) < 0 ? errno : 0;
        host_close_fd(fd);
      }
    }
    files = files.subspan(batch.size());
  }
}

bool is_memory_backing_token(int fd) {
  kmd::KeeperTokenIdentity identity;
  return identify(fd, identity);
}

bool kmd::is_device_memory_backing(int fd) {
  kmd::KeeperTokenIdentity identity;
  if (identify(fd, identity))
    return identity.device_backing;
  const int saved_errno = errno;
  std::array<char, 64> path{};
  std::array<char, 256> target{};
  std::snprintf(path.data(), path.size(), "/proc/self/fd/%d", fd);
  const ssize_t length = host_call(libc_passthrough().readlink_fn, ::readlink, path.data(),
                                   target.data(), target.size());
  errno = saved_errno;
  if (length <= 0)
    return false;
  const std::string_view name(target.data(), static_cast<size_t>(length));
  constexpr std::string_view kMemfdPrefix = "/memfd:";
  return name.starts_with(kMemfdPrefix) &&
         name.substr(kMemfdPrefix.size()).starts_with(kDeviceBackingNamePrefix);
}

int stat_memory_backing(int fd, struct stat *result) {
  if (host_call(libc_passthrough().fstat_fn, ::fstat, fd, result) != 0)
    return -1;
  if (S_ISSOCK(result->st_mode))
    stat_memory_backing_token(fd, result);
  return 0;
}

bool stat_memory_backing_token(int fd, struct stat *result) {
  kmd::KeeperTokenIdentity identity;
  if (!identify(fd, identity))
    return false;
  result->st_dev = identity.shmid < 0 ? identity.ipc_namespace : 0;
  result->st_ino = identity.object;
  result->st_mode = S_IFREG | 0600;
  result->st_size = identity.size;
  return true;
}

bool seek_memory_backing_token(int fd, off_t offset, int whence, off_t &result) {
  kmd::KeeperTokenIdentity identity;
  if (!identify(fd, identity))
    return false;
  result = -1;
  if (offset != 0 || (whence != SEEK_SET && whence != SEEK_END))
    errno = EINVAL;
  else
    result = whence == SEEK_END ? static_cast<off_t>(identity.size) : 0;
  return true;
}

int fcntl_memory_backing(int fd, int command, uintptr_t argument) {
  kmd::KeeperTokenIdentity identity;
  if (identify(fd, identity)) {
    switch (command) {
    case F_GETFL:
      return identity.mode;
    case F_GET_SEALS:
      if (identity.shmid < 0)
        break;
      return F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL;
    case F_ADD_SEALS:
      if (identity.shmid < 0)
        break;
      errno = EPERM;
      return -1;
    }
    if (identity.shmid < 0 && (command == F_GET_SEALS || command == F_ADD_SEALS)) {
      if (check_token_liveness(fd) != 0)
        return -1;
      const int file = request_reopen(fd, identity.mode, kmd::KeeperOperation::OpenFile);
      if (file < 0)
        return -1;
      const int result = host_call(libc_passthrough().fcntl, ::fcntl, file, command, argument);
      const int error = errno;
      host_close_fd(file);
      errno = error;
      return result;
    }
  }
  return host_call(libc_passthrough().fcntl, ::fcntl, fd, command, argument);
}

int reopen_memory_backing(int fd, int access_mode) {
  kmd::KeeperTokenIdentity identity;
  if (identify(fd, identity)) {
    if (check_token_liveness(fd) != 0)
      return -1;
    if (identity.shmid < 0 && (access_mode & O_ACCMODE) != identity.mode) {
      const int source = request_reopen(fd, identity.mode, kmd::KeeperOperation::OpenFile);
      if (source < 0)
        return -1;
      char path[64];
      std::snprintf(path, sizeof(path), "/proc/self/fd/%d", source);
      int reopened = host_call(libc_passthrough().openat, ::openat, AT_FDCWD, path, access_mode, 0);
      const int open_error = errno;
      host_close_fd(source);
      if (reopened < 0) {
        errno = open_error;
        return -1;
      }
      uint32_t policy = 0;
      if (get_memory_backing_alloc_flags(fd, &policy) != 1 ||
          retain_memory_backing_export_policy(reopened, policy) != 0) {
        const int error = errno;
        host_close_fd(reopened);
        errno = error ? error : EPROTO;
        return -1;
      }
      return reopened;
    }
    const int reopened = request_reopen(fd, access_mode & O_ACCMODE);
    if (reopened < 0) {
      const int error = errno;
      if (check_token_liveness(fd) == 0)
        errno = error;
      return -1;
    }
    if (reopened >= 0 && !(access_mode & O_CLOEXEC))
      host_call(libc_passthrough().fcntl, ::fcntl, reopened, F_SETFD, 0);
    return reopened;
  }
  char path[64];
  std::snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
  return host_call(libc_passthrough().openat, ::openat, AT_FDCWD, path, access_mode, 0);
}

int get_memory_backing_alloc_flags(int fd, uint32_t *flags) {
  if (!flags) {
    errno = EINVAL;
    return -1;
  }
  kmd::KeeperTokenIdentity identity;
  if (identify(fd, identity)) {
    if (check_token_liveness(fd) != 0)
      return -1;
    bool present = false;
    return request_allocation_flags(fd, kmd::KeeperOperation::GetAllocationFlags, flags,
                                    &present) == 0
               ? static_cast<int>(present)
               : -1;
  }
  AllocationFlagsXattr policy{};
  const ssize_t size = ::fgetxattr(fd, kmd::kAllocationFlagsXattr, &policy, sizeof(policy));
  if (size < 0 && (errno == ENODATA || errno == EOPNOTSUPP))
    return 0;
  if (size < 0)
    return -1;
  if (size != sizeof(policy) || policy.version != 1) {
    errno = EPROTO;
    return -1;
  }
  *flags = policy.flags;
  return 1;
}

int set_memory_backing_alloc_flags(int fd, uint32_t flags) {
  kmd::KeeperTokenIdentity identity;
  if (identify(fd, identity)) {
    if (check_token_liveness(fd) != 0)
      return -1;
    bool present = false;
    return request_allocation_flags(fd, kmd::KeeperOperation::SetAllocationFlags, &flags, &present);
  }
  const AllocationFlagsXattr policy{1, flags};
  if (::fsetxattr(fd, kmd::kAllocationFlagsXattr, &policy, sizeof(policy), XATTR_CREATE) == 0)
    return 0;
  if (errno != EEXIST)
    return -1;
  uint32_t existing = 0;
  const int status = get_memory_backing_alloc_flags(fd, &existing);
  if (status == 1 && existing == flags)
    return 0;
  if (status >= 0)
    errno = EEXIST;
  return -1;
}

int retain_memory_backing_export_policy(int &fd, uint32_t flags) {
  if (set_memory_backing_alloc_flags(fd, flags) == 0)
    return 0;
  if (errno != EOPNOTSUPP || is_memory_backing_token(fd))
    return -1;
  const int file_flags = host_call(libc_passthrough().fcntl, ::fcntl, fd, F_GETFL);
  const int descriptor_flags = host_call(libc_passthrough().fcntl, ::fcntl, fd, F_GETFD);
  if (file_flags < 0 || descriptor_flags < 0)
    return -1;
  const int mode = file_flags & O_ACCMODE;
  if (mode != O_RDONLY && mode != O_RDWR) {
    errno = EINVAL;
    return -1;
  }
  kmd::KeeperRequest request{flags, mode, kmd::KeeperOperation::WrapFile, {}};
  request.name[0] = kmd::is_device_memory_backing(fd) ? 1 : 0;
  const int replacement = with_keeper_connection([&](int socket, bool &disconnected) {
    if (kmd::send_keeper_descriptor(socket, &request, sizeof(request), fd) != 0) {
      disconnected = errno == EPIPE || errno == ECONNRESET;
      finish_private_request(socket);
      return -1;
    }
    return receive_result(socket, true, &disconnected);
  });
  if (replacement < 0)
    return -1;
  if (!(descriptor_flags & FD_CLOEXEC) &&
      host_call(libc_passthrough().fcntl, ::fcntl, replacement, F_SETFD, 0) != 0) {
    const int error = errno;
    host_close_fd(replacement);
    errno = error;
    return -1;
  }
  host_close_fd(fd);
  fd = replacement;
  return 0;
}

int check_memory_backing_size(int fd, size_t size) {
  struct stat info {};
  if (stat_memory_backing(fd, &info) != 0)
    return -1;
  if (info.st_size >= 0 && size <= static_cast<uint64_t>(info.st_size))
    return 0;
  errno = EINVAL;
  return -1;
}

void *map_memory_backing(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  kmd::KeeperTokenIdentity identity;
  if (fd < 0 || (flags & MAP_ANONYMOUS) || !identify(fd, identity)) {
    auto mmap_fn = libc_passthrough().mmap;
    return mmap_fn ? mmap_fn(addr, length, prot, flags, fd, offset)
                   : ::mmap(addr, length, prot, flags, fd, offset);
  }
  if (check_token_liveness(fd) != 0)
    return MAP_FAILED;
  // A shmid can name unrelated memory in another IPC namespace. Reject that
  // transfer explicitly instead of attaching an accidentally matching id.
  if (identity.shmid >= 0 &&
      (!identity.ipc_namespace || identity.ipc_namespace != kmd::keeper_ipc_namespace())) {
    errno = EXDEV;
    return MAP_FAILED;
  }
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  if (!length || offset < 0 || static_cast<uint64_t>(offset) % page ||
      length > SIZE_MAX - (page - 1) || identity.size > SIZE_MAX - (page - 1) ||
      (flags & MAP_TYPE) != MAP_SHARED || (flags & ~kMemoryBackingMappingFlags)) {
    errno = EINVAL;
    return MAP_FAILED;
  }
  const size_t mapped_size = (length + page - 1) & ~(page - 1);
  const size_t region_size = (identity.size + page - 1) & ~(page - 1);
  if (static_cast<uint64_t>(offset) >= region_size || mapped_size > region_size - offset) {
    errno = EINVAL;
    return MAP_FAILED;
  }
  if (identity.mode == O_RDONLY && (prot & PROT_WRITE)) {
    errno = EACCES;
    return MAP_FAILED;
  }
  if ((flags & (MAP_FIXED | MAP_FIXED_NOREPLACE)) &&
      (reinterpret_cast<uintptr_t>(addr) % page ||
       reinterpret_cast<uintptr_t>(addr) > UINTPTR_MAX - mapped_size)) {
    errno = EINVAL;
    return MAP_FAILED;
  }
  if (identity.shmid < 0) {
    const int file = request_reopen(fd, identity.mode, kmd::KeeperOperation::OpenFile);
    if (file < 0)
      return MAP_FAILED;
    void *mapped =
        host_call(libc_passthrough().mmap, ::mmap, addr, length, prot, flags, file, offset);
    const int error = errno;
    host_close_fd(file);
    errno = error;
    return mapped;
  }
  int placement_flags = flags & MAP_FIXED_NOREPLACE;
#ifdef MAP_32BIT
  if (!(flags & MAP_FIXED))
    placement_flags |= flags & MAP_32BIT;
#endif
  const bool reserve_target = placement_flags != 0;
  if (reserve_target) {
    void *reservation = host_call(libc_passthrough().mmap, ::mmap, addr, mapped_size, PROT_NONE,
                                  MAP_PRIVATE | MAP_ANONYMOUS | placement_flags, -1, 0);
    if (reservation == MAP_FAILED)
      return MAP_FAILED;
    addr = reinterpret_cast<void *>(reservation);
    flags |= MAP_FIXED;
  }
  // An unmapped MAP_FIXED target may itself be selected by shmat. Keep
  // overlapping temporary attachments until another attachment lies outside
  // the target. At most two disjoint regions of this size can overlap it.
  void *attachments[3]{};
  unsigned count = 0;
  void *source = nullptr;
  do {
    source =
        reinterpret_cast<void *>(host_call(libc_passthrough().shmat, ::shmat, identity.shmid,
                                           nullptr, identity.mode == O_RDONLY ? SHM_RDONLY : 0));
    if (source == reinterpret_cast<void *>(-1)) {
      const int error = errno;
      for (unsigned index = 0; index < count; ++index)
        host_call(libc_passthrough().shmdt, ::shmdt, attachments[index]);
      if (reserve_target)
        host_call(libc_passthrough().munmap, ::munmap, addr, mapped_size);
      if (check_token_liveness(fd) == 0)
        errno = error;
      return MAP_FAILED;
    }
    attachments[count++] = source;
    const auto target = reinterpret_cast<uintptr_t>(addr);
    const auto start = reinterpret_cast<uintptr_t>(source);
    if (!(flags & MAP_FIXED) || target >= start + region_size || start >= target + mapped_size)
      break;
  } while (count < 3);
  // Drop overlapping attachments before replacing the target. Detaching their
  // old addresses afterwards could detach the newly installed alias instead.
  for (unsigned index = 0; index + 1 < count; ++index)
    host_call(libc_passthrough().shmdt, ::shmdt, attachments[index]);
  void *view = static_cast<char *>(source) + offset;
  void *mapped = MAP_FAILED;
  if (host_call(libc_passthrough().mprotect, ::mprotect, view, mapped_size, prot) == 0)
    mapped = host_call(libc_passthrough().mremap, ::mremap, view, 0, mapped_size,
                       MREMAP_MAYMOVE | ((flags & MAP_FIXED) ? MREMAP_FIXED : 0), addr);
  const int error = errno;
  host_call(libc_passthrough().shmdt, ::shmdt, source);
  if (mapped == MAP_FAILED && reserve_target)
    host_call(libc_passthrough().munmap, ::munmap, addr, mapped_size);
  errno = error;
  return mapped;
}

} // namespace rocjitsu
