// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file memory_keeper.cpp
/// @brief Server loop for the isolated memory keeper.

#include "rocjitsu/kmd/linux/memory_keeper.h"

#include "rocjitsu/kmd/linux/device_backing.h"
#include "rocjitsu/kmd/linux/libc_passthrough.h"

#include <cerrno>
#include <csignal>
#include <cstring>
#include <new>
#include <pthread.h>
#include <string_view>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/queue.h>
#include <sys/resource.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/statfs.h>
#include <sys/un.h>
#include <unistd.h>

namespace rocjitsu::kmd {
namespace {

struct Region {
  KeeperTokenIdentity identity;
  void *mapping;
  size_t tokens;
};

struct Channel {
  Region *region;
  int socket;
  KeeperFileResults files;
  LIST_ENTRY(Channel) links;
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

Region *create_region(size_t size, bool device_backing) {
  auto *region = allocate_record<Region>();
  if (!region)
    return nullptr;
  int shmid = -1;
  for (uint32_t slot = 0; slot < kDeviceTokenSlotCount; ++slot) {
    const key_t key =
        device_backing ? static_cast<key_t>(kDeviceTokenKeyPrefix | slot) : IPC_PRIVATE;
    shmid = host_call(libc_passthrough().shmget, ::shmget, key, size,
                      IPC_CREAT | IPC_EXCL | SHM_NORESERVE | 0600);
    if (shmid >= 0 || errno != EEXIST || !device_backing)
      break;
  }
  // The marker is best effort; namespace-wide key contention must not reject
  // an allocation when anonymous shared-memory capacity is still available.
  if (device_backing && shmid < 0 && errno == EEXIST)
    shmid = host_call(libc_passthrough().shmget, ::shmget, IPC_PRIVATE, size,
                      IPC_CREAT | IPC_EXCL | SHM_NORESERVE | 0600);
  if (shmid < 0) {
    const int error = errno;
    free_record(region);
    errno = error;
    return nullptr;
  }
  region->identity.size = size;
  region->identity.device_backing = device_backing;
  region->identity.ipc_namespace = keeper_ipc_namespace();
  region->identity.shmid = shmid;
  region->mapping =
      reinterpret_cast<void *>(host_call(libc_passthrough().shmat, ::shmat, shmid, nullptr, 0));
  const int attach_error = errno;
  // After IPC_RMID the keeper attachment and client mappings own the segment.
  // A keeper killed between shmget and IPC_RMID can still leave a named segment.
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
int create_file(size_t size, bool preallocate, const char *name) {
  rlimit limit{};
  if (host_call(libc_passthrough().prlimit, ::prlimit, 0, RLIMIT_FSIZE, nullptr, &limit) != 0)
    return -1;
  if (size > limit.rlim_cur) {
    errno = EFBIG;
    return -1;
  }
  const int fd = host_call(libc_passthrough().memfd_create, ::memfd_create, name,
                           MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (fd < 0)
    return -1;
  if (host_call(libc_passthrough().ftruncate, ::ftruncate, fd, size) != 0 ||
      host_call(libc_passthrough().fcntl, ::fcntl, fd, F_ADD_SEALS,
                F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) != 0) {
    const int error = errno;
    host_close_fd(fd);
    errno = error;
    return -1;
  }
  if (preallocate)
    host_call(libc_passthrough().fallocate, ::fallocate, fd, 0, 0, size);
  return fd;
}

int create_token(int epoll, Region *region, int mode, uint64_t serial) {
  auto *token = allocate_record<Channel>();
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
  KeeperTokenIdentity identity = region->identity;
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
    host_close_fd(sockets[0]);
    host_close_fd(sockets[1]);
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
  // The owner's non-inherited mapping retains a different open-file description.
  // Unmapping it, including on exec or exit, releases its lock.
  while (host_call(libc_passthrough().fcntl, ::fcntl, watch.file, F_OFD_SETLKW, &lock) < 0 &&
         errno == EINTR) {
  }
  const uint64_t ready = 1;
  host_call(libc_passthrough().write, ::write, watch.event, &ready, sizeof(ready));
  return nullptr;
}

} // namespace

// Runs in a fresh executable, with no inherited application mappings or locks.
[[noreturn]] void run_memory_keeper(pid_t owner, const char *listener_name) {
  constexpr int lease = kKeeperLeaseFd;
  constexpr int startup = kKeeperStartupFd;
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

  auto fail_startup = [&](int error) {
    send_keeper_startup_reply(startup, error);
    host_call(libc_passthrough().exit_immediately, ::_exit, 1);
  };
  // Only the helper changes its limits. A low application soft limit need not
  // constrain metadata files or the number of simultaneously retained tokens.
  for (const auto resource : {RLIMIT_FSIZE, RLIMIT_NOFILE}) {
    rlimit limit{};
    if (host_call(libc_passthrough().prlimit, ::prlimit, 0, resource, nullptr, &limit) != 0)
      fail_startup(errno);
    limit.rlim_cur = limit.rlim_max;
    if (host_call(libc_passthrough().prlimit, ::prlimit, 0, resource, &limit, nullptr) != 0)
      fail_startup(errno);
  }

  // Create the listener after exec so an application fork can never retain
  // it past the helper's lifetime.
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  const size_t length = std::strlen(listener_name);
  if (length == 0 || length >= sizeof(address.sun_path))
    fail_startup(EINVAL);
  std::memcpy(address.sun_path + 1, listener_name, length);
  const int listener =
      host_call(libc_passthrough().socket, ::socket, AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (listener < 0 ||
      host_call(libc_passthrough().bind, ::bind, listener, reinterpret_cast<sockaddr *>(&address),
                offsetof(sockaddr_un, sun_path) + 1 + length) != 0 ||
      host_call(libc_passthrough().listen, ::listen, listener, SOMAXCONN) != 0)
    fail_startup(errno);
  const int parent = host_pidfd_open(owner, 0);
  if (parent < 0)
    fail_startup(errno);
  const int epoll = host_call(libc_passthrough().epoll_create1, ::epoll_create1, EPOLL_CLOEXEC);
  if (epoll < 0)
    fail_startup(errno);
  const int lease_eventfd = host_call(libc_passthrough().eventfd, ::eventfd, 0, EFD_CLOEXEC);
  if (lease_eventfd < 0)
    fail_startup(errno);
  Channel listener_event{}, parent_event{}, lease_event{};
  epoll_event event{};
  event.events = EPOLLIN;
  event.data.ptr = &listener_event;
  if (host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_ADD, listener,
                &event) != 0)
    fail_startup(errno);
  event.data.ptr = &parent_event;
  if (host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_ADD, parent, &event) !=
      0)
    fail_startup(errno);
  event.data.ptr = &lease_event;
  if (host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_ADD, lease_eventfd,
                &event) != 0)
    fail_startup(errno);
  LeaseWatch watch{lease, lease_eventfd};
  pthread_t watcher;
  const int started = pthread_create(&watcher, nullptr, watch_lease, &watch);
  if (started != 0)
    fail_startup(started);
  pthread_detach(watcher);
  // Only the inherited startup channel and the authenticated owner receive this cookie.
  KeeperCookie cookie;
  ssize_t bytes;
  do {
    bytes = host_call(libc_passthrough().getrandom, ::getrandom, cookie.data(), sizeof(cookie), 0);
  } while (bytes < 0 && errno == EINTR);
  if (bytes != sizeof(cookie))
    fail_startup(bytes < 0 ? errno : EIO);
  // Linux pidfs on 64-bit kernels allocates a monotonically increasing inode
  // for each process. Old anonymous pidfds share an inode, and 32-bit pidfs
  // may recycle inode numbers. Use the launch cookie in those configurations.
  uint64_t process_inode = 0;
  if constexpr (sizeof(long) == 8) {
    constexpr long kPidfsMagic = 0x50494446; // PID_FS_MAGIC in Linux 6.9 and later.
    const int self = host_pidfd_open(getpid(), 0);
    struct statfs fs {};
    struct stat info {};
    if (self >= 0 && fstatfs(self, &fs) == 0 && fs.f_type == kPidfsMagic &&
        host_call(libc_passthrough().fstat_fn, ::fstat, self, &info) == 0)
      process_inode = info.st_ino;
    host_close_fd(self);
  }
  if (send_keeper_startup_reply(startup, 0, cookie, process_inode) != 0)
    host_call(libc_passthrough().exit_immediately, ::_exit, 1);
  host_close_fd(startup);
  size_t tokens = 0;
  uint64_t serial = 0;
  LIST_HEAD(PrivateChannels, Channel) private_channels{};
  auto retire = [&](Channel *channel) {
    host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_DEL, channel->socket,
              nullptr);
    host_close_fd(channel->socket);
    if (channel->region) {
      if (--channel->region->tokens == 0)
        release_region(channel->region);
      --tokens;
    } else {
      LIST_REMOVE(channel, links);
    }
    free_record(channel);
  };
  // A request is usually queued before accept completes. Serve it immediately
  // without allocating a connection record or another trip through epoll. A
  // connected client that has not sent yet is registered below, never waited on.
  bool owner_alive = true;
  auto serve = [&](Channel &channel) {
    const int socket = channel.socket;
    Region *region = channel.region;
    if (!region && !owner_alive)
      return false;
    KeeperFilePacket packet;
    KeeperRequest &request = packet.request;
    int incoming = -1;
    bool missing_descriptor = false;
    const long received = receive_keeper_descriptor(socket, &packet, sizeof(packet), incoming,
                                                    MSG_DONTWAIT, &missing_descriptor);
    if (received == 0 || (received < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
      return false;
    }
    if (received < 0)
      return true;
    // Allocation connections are private to a single request, so their
    // responses can use the connection itself. Exported tokens can be shared
    // across processes and threads; those still supply a private reply socket.
    const bool batch_file = !region && received >= static_cast<long>(sizeof(request)) &&
                            request.operation == KeeperOperation::BatchFile;
    const bool inline_file = batch_file && request.size <= kKeeperInlineFileSize;
    const size_t expected = sizeof(request) + (inline_file ? request.size : 0);
    if (received != static_cast<long>(expected) || (region && incoming < 0)) {
      host_close_fd(incoming);
      return region != nullptr;
    }
    if (!region && request.operation == KeeperOperation::Identify) {
      host_close_fd(incoming);
      return incoming < 0 && !missing_descriptor && channel.files.count == 0 &&
             send_keeper_descriptor(socket, &cookie, sizeof(cookie), -1) == 0;
    }
    if (!region && request.operation == KeeperOperation::FlushFiles) {
      const bool valid = !missing_descriptor && incoming < 0 && request.size == channel.files.count;
      host_close_fd(incoming);
      if (!valid)
        return false;
      const bool sent =
          send_keeper_descriptor(socket, &channel.files, sizeof(channel.files), -1) == 0;
      channel.files.count = 0;
      return sent;
    }
    if (!region && (channel.files.count == channel.files.errors.size() ||
                    (channel.files.count && !inline_file))) {
      host_close_fd(incoming);
      return false;
    }
    const int reply = region ? incoming : socket;
    int result = EINVAL;
    int descriptor = -1;
    if (missing_descriptor) {
      result = EMFILE;
    } else if (!region && (request.operation == KeeperOperation::SizeFile || batch_file)) {
      if (incoming >= 0) {
        result = host_call(libc_passthrough().ftruncate, ::ftruncate, incoming, request.size) == 0
                     ? 0
                     : errno;
        // The helper's file limit permits this write. pwrite keeps the
        // caller's shared open-file offset unchanged, including on errors.
        size_t written = 0;
        while (result == 0 && inline_file && written < request.size) {
          const ssize_t count =
              ::pwrite(incoming, packet.data + written, request.size - written, written);
          if (count < 0 && errno == EINTR)
            continue;
          if (count <= 0)
            result = count < 0 ? errno : EIO;
          else
            written += count;
        }
      }
    } else if ((request.operation == KeeperOperation::Backing ||
                request.operation == KeeperOperation::PreallocatedBacking ||
                request.operation == KeeperOperation::TokenBacking) &&
               (request.mode == O_RDONLY || request.mode == O_RDWR) &&
               (region || (request.size && request.size <= SIZE_MAX &&
                           std::memchr(request.name, '\0', sizeof(request.name))))) {
      if (!region) {
        if (request.operation == KeeperOperation::TokenBacking) {
          result = EFBIG;
        } else {
          descriptor =
              create_file(request.size, request.operation == KeeperOperation::PreallocatedBacking,
                          request.name);
          result = descriptor >= 0 ? 0 : errno;
        }
      }
      if (region || result == EFBIG) {
        Region *backing =
            region ? region
                   : create_region(
                         request.size,
                         std::string_view(request.name).starts_with(kDeviceBackingNamePrefix));
        if (backing) {
          descriptor = create_token(epoll, backing, request.mode, ++serial);
          result = descriptor >= 0 ? 0 : errno;
          if (descriptor >= 0)
            ++tokens;
          else if (!region)
            release_region(backing);
        } else {
          result = errno;
        }
      }
    }
    if (inline_file) {
      channel.files.errors[channel.files.count++] = result;
      host_close_fd(incoming);
      return true;
    }
    const bool sent = send_keeper_descriptor(reply, &result, sizeof(result), descriptor) == 0;
    host_close_fd(descriptor);
    host_close_fd(incoming);
    return region != nullptr || (sent && batch_file);
  };
  while (owner_alive || tokens != 0) {
    epoll_event events[32];
    const int count = host_call(libc_passthrough().epoll_wait, ::epoll_wait, epoll, events, 32, -1);
    if (count < 0) {
      if (errno == EINTR)
        continue;
      break;
    }
    for (int index = 0; index < count; ++index) {
      auto *channel = static_cast<Channel *>(events[index].data.ptr);
      if (channel == &parent_event || channel == &lease_event) {
        if (!owner_alive)
          continue;
        owner_alive = false;
        host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_DEL, parent, nullptr);
        host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_DEL, listener,
                  nullptr);
        host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_DEL, lease_eventfd,
                  nullptr);
        host_close_fd(parent);
        host_close_fd(listener);
        continue;
      }
      if (channel == &listener_event) {
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
          host_close_fd(socket);
          continue;
        }
        Channel accepted{};
        accepted.socket = socket;
        if (!serve(accepted)) {
          host_close_fd(socket);
          continue;
        }
        auto *connection = allocate_record<Channel>();
        if (!connection) {
          host_close_fd(socket);
          continue;
        }
        *connection = accepted;
        event.events = EPOLLIN | EPOLLRDHUP;
        event.data.ptr = connection;
        if (host_call(libc_passthrough().epoll_ctl, ::epoll_ctl, epoll, EPOLL_CTL_ADD, socket,
                      &event) != 0) {
          host_close_fd(socket);
          free_record(connection);
        } else {
          LIST_INSERT_HEAD(&private_channels, connection, links);
        }
        continue;
      }
      if (!serve(*channel))
        retire(channel);
    }
    // epoll_wait may have returned private channels alongside the owner event.
    // Finish the batch before freeing any remaining channel records.
    if (!owner_alive)
      while (Channel *channel = LIST_FIRST(&private_channels))
        retire(channel);
  }
  host_call(libc_passthrough().exit_immediately, ::_exit, 0);
  __builtin_unreachable();
}

} // namespace rocjitsu::kmd
