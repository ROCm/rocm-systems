// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/memory_backing.h"

#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "util/log.h"

#include <atomic>
#include <cerrno>
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
#include <pthread.h>
#include <sched.h>
#include <string>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
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

// The standalone CLI/tests may call these before the interposer resolves its
// table. Prefer its resolved entry point when present, otherwise use libc.
template <typename Function, typename Fallback, typename... Args>
decltype(auto) host_call(Function resolved, Fallback fallback, Args... args) {
  return (resolved ? resolved : fallback)(args...);
}

void close_fd(int fd) {
  if (fd >= 0)
    host_call(libc_passthrough().close, ::close, fd);
}

constexpr uint64_t kMagic = 0x314d41525553544a;
struct Identity {
  uint64_t magic = kMagic;
  uint64_t size = 0;
  uint64_t object = 0;
  uint64_t token = 0;
  uint64_t ipc_namespace = 0;
  int32_t shmid = -1;
  int32_t mode = O_RDWR;
};

uint64_t ipc_namespace() {
  struct stat info {};
  if (host_call(libc_passthrough().stat, ::stat, "/proc/self/ns/ipc", &info) != 0)
    return 0;
  return info.st_ino;
}

bool identify(int fd, Identity &identity) {
  const int saved_errno = errno;
  sockaddr_un address{};
  socklen_t length = sizeof(address);
  bool found = host_call(libc_passthrough().getsockname, ::getsockname, fd,
                         reinterpret_cast<sockaddr *>(&address), &length) == 0 &&
               length == offsetof(sockaddr_un, sun_path) + 1 + sizeof(identity) &&
               address.sun_family == AF_UNIX && address.sun_path[0] == 0;
  if (found) {
    std::memcpy(&identity, address.sun_path + 1, sizeof(identity));
    found = identity.magic == kMagic && identity.size > 0 && identity.shmid >= 0 &&
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

enum class Operation : uint32_t { Backing, SizeFile, PreallocatedBacking, TokenBacking };

struct Request {
  uint64_t size;
  int32_t mode;
  Operation operation;
};

int send_descriptor(int socket, const void *data, size_t size, int fd) {
  iovec buffer{const_cast<void *>(data), size};
  alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))]{};
  msghdr message{};
  message.msg_iov = &buffer;
  message.msg_iovlen = 1;
  if (fd >= 0) {
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    auto *header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(fd));
    std::memcpy(CMSG_DATA(header), &fd, sizeof(fd));
  }
  long sent;
  do {
    sent = host_call(libc_passthrough().sendmsg, ::sendmsg, socket, &message,
                     MSG_NOSIGNAL | MSG_DONTWAIT);
  } while (sent < 0 && errno == EINTR);
  if (sent == static_cast<long>(size))
    return 0;
  if (sent >= 0)
    errno = EIO;
  return -1;
}

long receive_descriptor(int socket, void *data, size_t size, int &fd, int flags = 0) {
  iovec buffer{data, size};
  alignas(cmsghdr) char control[CMSG_SPACE(4 * sizeof(int))]{};
  msghdr message{};
  message.msg_iov = &buffer;
  message.msg_iovlen = 1;
  message.msg_control = control;
  message.msg_controllen = sizeof(control);
  long received;
  do {
    received = host_call(libc_passthrough().recvmsg, ::recvmsg, socket, &message,
                         flags | MSG_CMSG_CLOEXEC);
  } while (received < 0 && errno == EINTR);
  fd = -1;
  unsigned descriptors = 0;
  if (received <= 0)
    return received;
  for (auto *header = CMSG_FIRSTHDR(&message); header; header = CMSG_NXTHDR(&message, header)) {
    if (header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS)
      continue;
    const auto count = (header->cmsg_len - CMSG_LEN(0)) / sizeof(int);
    for (size_t index = 0; index < count; ++index) {
      int incoming;
      std::memcpy(&incoming, CMSG_DATA(header) + index * sizeof(int), sizeof(incoming));
      if (++descriptors == 1)
        fd = incoming;
      else
        close_fd(incoming);
    }
  }
  if (descriptors > 1 || (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC))) {
    close_fd(fd);
    fd = -1;
    errno = EPROTO;
    return -1;
  }
  return received;
}

struct Region {
  Identity identity;
  void *mapping;
  size_t tokens;
};

struct Token {
  Region *region;
  int socket;
};

template <typename T> T *allocate_record() {
  auto result = host_call(libc_passthrough().mmap, ::mmap, nullptr, sizeof(T),
                          PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (result == MAP_FAILED)
    return nullptr;
  return new (reinterpret_cast<void *>(result)) T{};
}

template <typename T> void free_record(T *record) {
  host_call(libc_passthrough().munmap, ::munmap, record, sizeof(T));
}

void release_region(Region *region) {
  host_call(libc_passthrough().shmdt, ::shmdt, region->mapping);
  free_record(region);
}

Region *create_region(size_t size) {
  auto *region = allocate_record<Region>();
  if (!region)
    return nullptr;
  const int shmid = host_call(libc_passthrough().shmget, ::shmget, IPC_PRIVATE, size,
                              IPC_CREAT | SHM_NORESERVE | 0600);
  if (shmid < 0) {
    const int error = errno;
    free_record(region);
    errno = error;
    return nullptr;
  }
  region->identity.size = size;
  region->identity.ipc_namespace = ipc_namespace();
  region->identity.shmid = shmid;
  region->mapping =
      reinterpret_cast<void *>(host_call(libc_passthrough().shmat, ::shmat, shmid, nullptr, 0));
  const int attach_error = errno;
  // The keeper attachment, and later client mappings, own the segment. Even a
  // killed keeper leaves no named IPC object behind.
  const int removed = host_call(libc_passthrough().shmctl, ::shmctl, shmid, IPC_RMID, nullptr);
  if (removed != 0 && region->mapping != reinterpret_cast<void *>(-1)) {
    const int error = errno;
    release_region(region);
    errno = error;
    return nullptr;
  }
  if (region->mapping == reinterpret_cast<void *>(-1)) {
    free_record(region);
    errno = attach_error;
    return nullptr;
  }
  long random_bytes;
  do {
    random_bytes = host_call(libc_passthrough().getrandom, ::getrandom, &region->identity.object,
                             sizeof(uint64_t), 0);
  } while (random_bytes < 0 && errno == EINTR);
  if (random_bytes != sizeof(uint64_t)) {
    const int error = random_bytes < 0 ? errno : EIO;
    release_region(region);
    errno = error;
    return nullptr;
  }
  return region;
}

// The keeper has its own file limit. Prefer ordinary exportable files when
// that limit permits; unlike the application, it never runs application code
// that can lower the limit concurrently with sizing a file. An inherited hard
// limit is handled by shared RAM below, without changing application limits.
int create_file(size_t size, bool preallocate) {
  rlimit limit{};
  if (host_call(libc_passthrough().prlimit, ::prlimit, 0, RLIMIT_FSIZE, nullptr, &limit) != 0)
    return -1;
  if (size > limit.rlim_cur) {
    errno = EFBIG;
    return -1;
  }
  const int fd = host_call(libc_passthrough().memfd_create, ::memfd_create, "rocjitsu_backing",
                           MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (fd < 0)
    return -1;
  if (host_call(libc_passthrough().ftruncate, ::ftruncate, fd, size) != 0 ||
      host_call(libc_passthrough().fcntl, ::fcntl, fd, F_ADD_SEALS,
                F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) != 0) {
    const int error = errno;
    close_fd(fd);
    errno = error;
    return -1;
  }
  if (preallocate)
    host_call(libc_passthrough().fallocate, ::fallocate, fd, 0, 0, size);
  return fd;
}

int create_token(int epoll, Region *region, int mode, uint64_t serial) {
  auto *token = allocate_record<Token>();
  if (!token)
    return -1;
  int sockets[2];
  if (host_call(libc_passthrough().socketpair, ::socketpair, AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC,
                0, sockets) != 0) {
    const int error = errno;
    free_record(token);
    errno = error;
    return -1;
  }
  Identity identity = region->identity;
  identity.mode = mode;
  identity.token = serial;
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path + 1, &identity, sizeof(identity));
  token->socket = sockets[0];
  token->region = region;
  epoll_event event{};
  event.events = EPOLLIN | EPOLLRDHUP;
  event.data.ptr = token;
  if (host_call(libc_passthrough().bind, ::bind, sockets[1],
                reinterpret_cast<const sockaddr *>(&address),
                offsetof(sockaddr_un, sun_path) + 1 + sizeof(identity)) != 0 ||
      host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_ADD, sockets[0],
                &event) != 0) {
    const int error = errno;
    close_fd(sockets[0]);
    close_fd(sockets[1]);
    free_record(token);
    errno = error;
    return -1;
  }
  ++region->tokens;
  return sockets[1];
}

struct LeaseWatch {
  int file;
  int event;
};

void *watch_lease(void *argument) {
  const auto watch = *static_cast<const LeaseWatch *>(argument);
  struct flock lock {};
  lock.l_type = F_WRLCK;
  lock.l_whence = SEEK_SET;
  // POSIX record locks belong to a process, not its open-file description.
  // Forked children do not retain this lock when the owner closes it on exec.
  while (host_call(libc_passthrough().fcntl, ::fcntl, watch.file, F_SETLKW, &lock) < 0 &&
         errno == EINTR) {
  }
  const uint64_t ready = 1;
  host_call(libc_passthrough().write, ::write, watch.event, &ready, sizeof(ready));
  return nullptr;
}

// Runs in a fresh executable, with no inherited application mappings or locks.
[[noreturn]] void keep_regions(int listener, int lease, int startup, pid_t owner) {
  host_call(libc_passthrough().prctl, ::prctl, PR_SET_NAME, "rj-memory", 0L, 0L, 0L);
  // Avoid invoking application handlers in the keeper. The kernel closes its
  // sockets and drops its marked-for-removal attachments on termination.
  struct sigaction action {};
  action.sa_handler = SIG_DFL;
  sigemptyset(&action.sa_mask);
  for (int signal = 1; signal < NSIG; ++signal)
    if (signal != SIGKILL && signal != SIGSTOP)
      sigaction(signal, &action, nullptr);
  // This process only sizes internal backing files. Ignore its own size-limit
  // signal so even an external prlimit racing create_file falls back to RAM.
  action.sa_handler = SIG_IGN;
  sigaction(SIGXFSZ, &action, nullptr);
  sigset_t mask;
  sigemptyset(&mask);
  sigprocmask(SIG_SETMASK, &mask, nullptr);

  // Only the helper changes its limits. A low application soft limit need not
  // constrain metadata files or the number of simultaneously retained tokens.
  for (const auto resource : {RLIMIT_FSIZE, RLIMIT_NOFILE}) {
    rlimit limit{};
    if (host_call(libc_passthrough().prlimit, ::prlimit, 0, resource, nullptr, &limit) != 0)
      host_call(libc_passthrough().exit_immediately, ::_exit, 1);
    limit.rlim_cur = limit.rlim_max;
    if (host_call(libc_passthrough().prlimit, ::prlimit, 0, resource, &limit, nullptr) != 0)
      host_call(libc_passthrough().exit_immediately, ::_exit, 1);
  }

  const int parent = host_pidfd_open(owner, 0);
  const int epoll = host_call(libc_passthrough().epoll_create1, ::epoll_create1, EPOLL_CLOEXEC);
  const int lease_eventfd = host_call(libc_passthrough().eventfd, ::eventfd, 0, EFD_CLOEXEC);
  Token listener_event{}, parent_event{}, lease_event{};
  epoll_event event{};
  event.events = EPOLLIN;
  event.data.ptr = &listener_event;
  if (parent < 0 || epoll < 0 || lease_eventfd < 0 ||
      host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_ADD, listener,
                &event) != 0)
    host_call(libc_passthrough().exit_immediately, ::_exit, 1);
  event.data.ptr = &parent_event;
  if (host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_ADD, parent, &event) !=
      0)
    host_call(libc_passthrough().exit_immediately, ::_exit, 1);
  event.data.ptr = &lease_event;
  if (host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_ADD, lease_eventfd,
                &event) != 0)
    host_call(libc_passthrough().exit_immediately, ::_exit, 1);
  LeaseWatch watch{lease, lease_eventfd};
  pthread_t watcher;
  const int started = pthread_create(&watcher, nullptr, watch_lease, &watch);
  if (started != 0) {
    const int error = -started;
    send_descriptor(startup, &error, sizeof(error), -1);
    host_call(libc_passthrough().exit_immediately, ::_exit, 1);
  }
  pthread_detach(watcher);
  const int ready = getpid();
  if (send_descriptor(startup, &ready, sizeof(ready), -1) != 0)
    host_call(libc_passthrough().exit_immediately, ::_exit, 1);
  close_fd(startup);
  size_t tokens = 0;
  uint64_t serial = 0;
  auto retire = [&](Token *token) {
    host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_DEL, token->socket,
              nullptr);
    close_fd(token->socket);
    if (token->region) {
      if (--token->region->tokens == 0)
        release_region(token->region);
      --tokens;
    }
    free_record(token);
  };
  bool owner_alive = true;
  while (owner_alive || tokens != 0) {
    epoll_event events[32];
    const int count = host_call(libc_passthrough().epoll_wait, ::epoll_wait, epoll, events, 32, -1);
    if (count < 0) {
      if (errno == EINTR)
        continue;
      break;
    }
    for (int index = 0; index < count; ++index) {
      auto *token = static_cast<Token *>(events[index].data.ptr);
      if (token == &parent_event || token == &lease_event) {
        if (!owner_alive)
          continue;
        owner_alive = false;
        host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_DEL, parent, nullptr);
        host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_DEL, listener,
                  nullptr);
        host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_DEL, lease_eventfd,
                  nullptr);
        close_fd(parent);
        close_fd(listener);
        continue;
      }
      if (token == &listener_event) {
        if (!owner_alive)
          continue;
        const int socket = host_call(libc_passthrough().accept4, ::accept4, listener, nullptr,
                                     nullptr, SOCK_CLOEXEC);
        if (socket < 0)
          continue;
        ucred peer{};
        socklen_t peer_size = sizeof(peer);
        if (host_call(libc_passthrough().getsockopt, ::getsockopt, socket, SOL_SOCKET, SO_PEERCRED,
                      &peer, &peer_size) != 0 ||
            peer.pid != owner || peer.uid != geteuid() || peer.gid != getegid()) {
          close_fd(socket);
          continue;
        }
        auto *connection = allocate_record<Token>();
        if (!connection) {
          close_fd(socket);
          continue;
        }
        connection->socket = socket;
        event.events = EPOLLIN | EPOLLRDHUP;
        event.data.ptr = connection;
        if (host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_ADD, socket,
                      &event) != 0) {
          close_fd(socket);
          free_record(connection);
        }
        continue;
      }
      const int socket = token->socket;
      Request request{};
      int reply = -1;
      const long received =
          receive_descriptor(socket, &request, sizeof(request), reply, MSG_DONTWAIT);
      if (received == 0 || (received < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
        retire(token);
        continue;
      }
      if (received != sizeof(request) || reply < 0) {
        close_fd(reply);
        continue;
      }
      int result = EINVAL;
      int descriptor = -1;
      if (!token->region && request.operation == Operation::SizeFile) {
        char byte = 0;
        if (receive_descriptor(reply, &byte, 1, descriptor, MSG_DONTWAIT) == 1 && descriptor >= 0)
          result =
              host_call(libc_passthrough().ftruncate, ::ftruncate, descriptor, request.size) == 0
                  ? 0
                  : errno;
      } else if ((request.operation == Operation::Backing ||
                  request.operation == Operation::PreallocatedBacking ||
                  request.operation == Operation::TokenBacking) &&
                 (request.mode == O_RDONLY || request.mode == O_RDWR) &&
                 (token->region || (request.size && request.size <= SIZE_MAX))) {
        if (!token->region) {
          if (request.operation == Operation::TokenBacking) {
            result = EFBIG;
          } else {
            descriptor =
                create_file(request.size, request.operation == Operation::PreallocatedBacking);
            result = descriptor >= 0 ? 0 : errno;
          }
        }
        if (token->region || result == EFBIG) {
          Region *region = token->region ? token->region : create_region(request.size);
          if (region) {
            descriptor = create_token(epoll, region, request.mode, ++serial);
            result = descriptor >= 0 ? 0 : errno;
            if (descriptor >= 0)
              ++tokens;
            else if (!token->region)
              release_region(region);
          } else {
            result = errno;
          }
        }
      }
      send_descriptor(reply, &result, sizeof(result), descriptor);
      close_fd(descriptor);
      close_fd(reply);
      if (!token->region)
        retire(token);
    }
  }
  host_call(libc_passthrough().exit_immediately, ::_exit, 0);
  __builtin_unreachable();
}

int request_backing(int socket, size_t size, int mode, int sized_file = -1,
                    bool *disconnected = nullptr, Operation operation = Operation::Backing) {
  int reply[2];
  if (host_call(libc_passthrough().socketpair, ::socketpair, AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC,
                0, reply) != 0)
    return -1;
  const Request request{size, mode, sized_file >= 0 ? Operation::SizeFile : operation};
  int result = 0;
  if (sized_file >= 0) {
    const char byte = 0;
    result = send_descriptor(reply[0], &byte, 1, sized_file);
  }
  if (result == 0)
    result = send_descriptor(socket, &request, sizeof(request), reply[1]);
  int error = result == 0 ? 0 : errno;
  if (disconnected && (error == EPIPE || error == ECONNRESET))
    *disconnected = true;
  close_fd(reply[1]);
  int descriptor = -1;
  if (result == 0) {
    int response = EIO;
    const long received = receive_descriptor(reply[0], &response, sizeof(response), descriptor);
    if (disconnected && received != sizeof(response))
      *disconnected = true;
    error = received < 0 ? errno : (received == sizeof(response) ? response : EIO);
    if (error == 0 && descriptor < 0)
      error = EIO;
  }
  close_fd(reply[0]);
  if (error) {
    close_fd(descriptor);
    errno = error;
    return -1;
  }
  return descriptor;
}

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
  const int result = -error;
  send_descriptor(channel, &result, sizeof(result), -1);
  host_call(libc_passthrough().exit_immediately, ::_exit, 1);
  __builtin_unreachable();
}

// NOFILE may have been lowered below descriptors inherited from the application.
// Enumerate the actual table without libc directory streams after the raw clone.
bool close_inherited_fds() {
  if (host_close_range(static_cast<unsigned>(kMemoryKeeperFirstUnreservedFd), UINT_MAX, 0u) == 0)
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
          number >= kMemoryKeeperFirstUnreservedFd && number != static_cast<unsigned>(directory))
        close_fd(static_cast<int>(number));
      position += entry->record_size;
    }
  }
  close_fd(directory);
  errno = error;
  return error == 0;
}

[[noreturn]] void exec_keeper(int control, int lease, int startup, const char *path,
                              const char *owner) {
  // Preserve the incoming descriptors before moving them into fixed slots.
  const int saved_lease = host_call(libc_passthrough().fcntl, ::fcntl, lease, F_DUPFD_CLOEXEC,
                                    kMemoryKeeperFirstUnreservedFd);
  const int saved_startup = host_call(libc_passthrough().fcntl, ::fcntl, startup, F_DUPFD_CLOEXEC,
                                      kMemoryKeeperFirstUnreservedFd);
  if (saved_lease < 0 || saved_startup < 0)
    startup_failed(startup, errno);
  if (control != kMemoryKeeperListenerFd) {
    if (host_call(libc_passthrough().dup3, ::dup3, control, kMemoryKeeperListenerFd, 0) < 0)
      startup_failed(saved_startup, errno);
  } else {
    host_call(libc_passthrough().fcntl, ::fcntl, kMemoryKeeperListenerFd, F_SETFD, 0);
  }
  if (host_call(libc_passthrough().dup3, ::dup3, saved_lease, kMemoryKeeperLeaseFd, 0) < 0 ||
      host_call(libc_passthrough().dup3, ::dup3, saved_startup, kMemoryKeeperStartupFd, 0) < 0)
    startup_failed(saved_startup, errno);
  for (int fd = 0; fd < kMemoryKeeperListenerFd; ++fd)
    close_fd(fd);
  if (!close_inherited_fds())
    startup_failed(kMemoryKeeperStartupFd, errno);
  char *arguments[] = {const_cast<char *>(path), const_cast<char *>(owner), nullptr};
  // The helper links its C++ runtime statically. Application preload and loader
  // configuration must not initialize a simulator inside the helper.
  // TSan's optional read-only shadow optimization writes a temporary file
  // before main. An unavailable temp directory skips that optimization while
  // retaining instrumentation, including with an inherited hard file limit of zero.
  char no_temporary_files[] = "TMPDIR=/dev/null";
  char *environment[] = {no_temporary_files, nullptr};
  host_call(libc_passthrough().execve, ::execve, path, arguments, environment);
  startup_failed(kMemoryKeeperStartupFd, errno);
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
  int listener;
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
  exec_keeper(bootstrap.listener, bootstrap.lease, bootstrap.startup, bootstrap.path,
              bootstrap.owner);
}

int detach_keeper(void *argument) {
#if defined(RJ_MEMORY_BACKING_WITH_ASAN)
  __sanitizer_finish_switch_fiber(nullptr, nullptr, nullptr);
#endif
  const auto &bootstrap = *static_cast<KeeperBootstrap *>(argument);
  close_fd(bootstrap.owner_startup);
  if (host_call(libc_passthrough().setsid, ::setsid) < 0)
    startup_failed(bootstrap.startup, errno);
  if (clone_keeper(enter_keeper, bootstrap.child_stack, CLONE_UNTRACED | SIGCHLD, argument) < 0)
    startup_failed(bootstrap.startup, errno);
  return 0;
}

std::atomic<pid_t> warned_subreaper{};

class Keeper {
public:
  const pid_t owner = getpid();
  const uid_t user = geteuid();
  const gid_t group = getegid();

  bool owns_current_process() const {
    return owner == getpid() && user == geteuid() && group == getegid();
  }

  int allocate(size_t size, int sized_file, bool &retry, Operation operation) {
    std::lock_guard lock(mutex_);
    // A caller may have obtained this pointer just before a credential change
    // replaced it. Never restart an unreachable helper after retirement.
    retry = retired_;
    if (retry)
      return -1;
    for (unsigned attempt = 0; attempt < 2; ++attempt) {
      if (!has_lease()) {
        release_lease();
        if (start() != 0)
          return -1;
      }
      const int socket =
          host_call(libc_passthrough().socket, ::socket, AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
      if (socket < 0)
        return -1;
      int descriptor = -1;
      bool disconnected = false;
      if (host_call(libc_passthrough().connect, ::connect, socket,
                    reinterpret_cast<const sockaddr *>(&address_), length_) == 0)
        descriptor = request_backing(socket, size, O_RDWR, sized_file, &disconnected, operation);
      else
        disconnected = errno == ECONNREFUSED;
      const int error = errno;
      close_fd(socket);
      if (descriptor >= 0 || !disconnected || attempt != 0) {
        errno = error;
        return descriptor;
      }
      // Retry only when the helper transport was lost. Resource errors such as
      // EMFILE must not forget a live helper and leak its process.
      release_lease();
    }
    __builtin_unreachable();
  }

  void retire() {
    if (owner != getpid()) {
      // A forked child must not lock a mutex held by a vanished parent thread.
      release_lease();
      return;
    }
    std::lock_guard lock(mutex_);
    retired_ = true;
    release_lease();
  }

private:
  void release_lease() {
    // close_range/dup2 may have closed or replaced the application's lease.
    // Never close an unrelated descriptor that reused its number.
    if (has_lease())
      close_fd(lease_);
    lease_ = -1;
    lease_inode_ = 0;
  }

  bool has_lease() const {
    struct stat info {};
    return lease_ >= 0 && lease_inode_ &&
           host_call(libc_passthrough().fstat_fn, ::fstat, lease_, &info) == 0 &&
           info.st_ino == lease_inode_ && info.st_dev == lease_device_;
  }

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
    if ((subreaper || owner == 1 || getppid() == 1) &&
        warned_subreaper.exchange(owner, std::memory_order_relaxed) != owner)
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
                                          "rocjitsu-memory-%d-%016lx", owner, nonce);
    length_ = offsetof(sockaddr_un, sun_path) + 1 + name_length;
    const int listener =
        host_call(libc_passthrough().socket, ::socket, AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (listener < 0)
      return -1;
    int leases[2] = {-1, -1};
    if (host_call(libc_passthrough().bind, ::bind, listener,
                  reinterpret_cast<const sockaddr *>(&address_), length_) != 0 ||
        host_call(libc_passthrough().listen, ::listen, listener, SOMAXCONN) != 0 ||
        host_call(libc_passthrough().socketpair, ::socketpair, AF_UNIX,
                  SOCK_SEQPACKET | SOCK_CLOEXEC, 0, leases) != 0) {
      const int error = errno;
      close_fd(listener);
      errno = error;
      return -1;
    }
    lease_ =
        host_call(libc_passthrough().memfd_create, ::memfd_create, "rocjitsu-owner", MFD_CLOEXEC);
    struct flock lock {};
    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_SET;
    struct stat info {};
    if (lease_ < 0 || host_call(libc_passthrough().fstat_fn, ::fstat, lease_, &info) != 0 ||
        host_call(libc_passthrough().fcntl, ::fcntl, lease_, F_SETLK, &lock) != 0) {
      const int error = errno;
      close_fd(lease_);
      lease_ = -1;
      close_fd(leases[0]);
      close_fd(leases[1]);
      close_fd(listener);
      errno = error;
      return -1;
    }
    lease_inode_ = info.st_ino;
    lease_device_ = info.st_dev;
    char owner_text[32];
    std::snprintf(owner_text, sizeof(owner_text), "%d", owner);
    // The bootstrap never execs, so ordinary waits cannot reap it. Its helper
    // is detached unless the owner is a child subreaper (diagnosed above).
    // libc clone requires separate child stacks. Neither clone shares VM, so
    // the parent can release its stack copies immediately after the first clone.
    void *stacks =
        host_call(libc_passthrough().mmap, ::mmap, nullptr, 2 * kBootstrapStackSize,
                  PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
    if (stacks == MAP_FAILED) {
      const int error = errno;
      close_fd(listener);
      close_fd(leases[0]);
      close_fd(leases[1]);
      errno = error;
      return -1;
    }
    KeeperBootstrap arguments{listener,
                              lease_,
                              leases[1],
                              leases[0],
                              path,
                              owner_text,
                              static_cast<char *>(stacks) + kBootstrapStackSize};
    const pid_t bootstrap = clone_keeper(detach_keeper, stacks, CLONE_UNTRACED, &arguments);
    const int error = errno;
    host_call(libc_passthrough().munmap, ::munmap, stacks, 2 * kBootstrapStackSize);
    close_fd(listener);
    close_fd(leases[1]);
    if (bootstrap < 0) {
      close_fd(leases[0]);
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
      close_fd(leases[0]);
      if (ready == 0)
        errno = ETIMEDOUT;
      return -1;
    }
    int response = 0, descriptor = -1;
    const long received = receive_descriptor(leases[0], &response, sizeof(response), descriptor);
    close_fd(leases[0]);
    close_fd(descriptor);
    if (received != sizeof(response) || response <= 0) {
      errno = response < 0 ? -response : EIO;
      return -1;
    }
    return 0;
  }

  std::mutex mutex_;
  sockaddr_un address_{};
  socklen_t length_ = 0;
  int lease_ = -1;
  ino_t lease_inode_ = 0;
  dev_t lease_device_ = 0;
  bool warned_ = false;
  bool retired_ = false;
};

// Intentionally process-lifetime: the keeper watches its owner pidfd and an
// exec-sensitive lease, then exits once externally held tokens are gone.
// A forked child never locks a mutex inherited from the parent's threads.
std::atomic<Keeper *> keeper{};

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

int allocate_backing(size_t size, int sized_file = -1, Operation operation = Operation::Backing) {
  for (;;) {
    auto *current = get_keeper();
    if (!current)
      return -1;
    bool retry = false;
    const int result = current->allocate(size, sized_file, retry, operation);
    if (!retry)
      return result;
  }
}

} // namespace

[[noreturn]] void run_memory_keeper(int listener, int lease, int startup, pid_t owner) {
  keep_regions(listener, lease, startup, owner);
}

int create_memory_backing(size_t size, bool preallocate) {
  if (!size || size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
    errno = EINVAL;
    return -1;
  }
  const char *force_token = std::getenv("RJ_TEST_FORCE_TOKEN_BACKING");
  const auto operation = force_token && std::strcmp(force_token, "1") == 0 ? Operation::TokenBacking
                         : preallocate ? Operation::PreallocatedBacking
                                       : Operation::Backing;
  return allocate_backing(size, -1, operation);
}

int populate_backing_file(int fd, const void *data, size_t size) {
  if (!size)
    return 0;
  if (size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
    errno = EFBIG;
    return -1;
  }
  const int resized = allocate_backing(size, fd);
  if (resized < 0)
    return -1;
  close_fd(resized);
  // Even overwriting an already sized file with write() checks RLIMIT_FSIZE.
  // Populate through its shared mapping instead; these are private setup
  // mappings that have never been published to a simulated GPU.
  void *view = host_call(libc_passthrough().mmap, ::mmap, nullptr, size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
  if (view == MAP_FAILED)
    return -1;
  std::memcpy(reinterpret_cast<void *>(view), data, size);
  host_call(libc_passthrough().munmap, ::munmap, view, size);
  return 0;
}

bool is_memory_backing_token(int fd) {
  Identity identity;
  return identify(fd, identity);
}

int stat_memory_backing(int fd, struct stat *result) {
  if (host_call(libc_passthrough().fstat_fn, ::fstat, fd, result) != 0)
    return -1;
  if (S_ISSOCK(result->st_mode))
    stat_memory_backing_token(fd, result);
  return 0;
}

bool stat_memory_backing_token(int fd, struct stat *result) {
  Identity identity;
  if (!identify(fd, identity))
    return false;
  result->st_dev = 0;
  result->st_ino = identity.object;
  result->st_mode = S_IFREG | 0600;
  result->st_size = identity.size;
  return true;
}

bool seek_memory_backing_token(int fd, off_t offset, int whence, off_t &result) {
  Identity identity;
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
  Identity identity;
  if (identify(fd, identity)) {
    switch (command) {
    case F_GETFL:
      return identity.mode;
    case F_GET_SEALS:
      return F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL;
    case F_ADD_SEALS:
      errno = EPERM;
      return -1;
    }
  }
  return host_call(libc_passthrough().fcntl, ::fcntl, fd, command, argument);
}

int reopen_memory_backing(int fd, int access_mode) {
  Identity identity;
  if (identify(fd, identity)) {
    if (check_token_liveness(fd) != 0)
      return -1;
    const int reopened = request_backing(fd, 0, access_mode & O_ACCMODE);
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

int ensure_memory_backing_size(int fd, size_t size) {
  struct stat info {};
  if (stat_memory_backing(fd, &info) != 0)
    return -1;
  if (info.st_size >= 0 && size <= static_cast<uint64_t>(info.st_size))
    return 0;
  errno = EINVAL;
  return -1;
}

void *map_memory_backing(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  Identity identity;
  if (fd < 0 || (flags & MAP_ANONYMOUS) || !identify(fd, identity)) {
    auto mmap_fn = libc_passthrough().mmap;
    return mmap_fn ? mmap_fn(addr, length, prot, flags, fd, offset)
                   : ::mmap(addr, length, prot, flags, fd, offset);
  }
  if (check_token_liveness(fd) != 0)
    return MAP_FAILED;
  // A shmid can name unrelated memory in another IPC namespace. Reject that
  // transfer explicitly instead of attaching an accidentally matching id.
  if (!identity.ipc_namespace || identity.ipc_namespace != ipc_namespace()) {
    errno = EXDEV;
    return MAP_FAILED;
  }
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  if (!length || offset < 0 || static_cast<uint64_t>(offset) % page ||
      length > SIZE_MAX - (page - 1) || identity.size > SIZE_MAX - (page - 1) ||
      (flags & MAP_TYPE) != MAP_SHARED) {
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
