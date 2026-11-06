// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file memory_keeper_protocol.cpp
/// @brief Sequenced-packet transport for keeper requests and replies.

#include "rocjitsu/kmd/linux/memory_keeper.h"

#include "rocjitsu/kmd/linux/libc_passthrough.h"

#include <cerrno>
#include <cstring>
#include <sys/socket.h>
#include <sys/stat.h>

namespace rocjitsu::kmd {

uint64_t keeper_ipc_namespace() {
  struct stat info {};
  if (host_call(libc_passthrough().stat, ::stat, "/proc/self/ns/ipc", &info) != 0)
    return 0;
  return info.st_ino;
}

int send_keeper_descriptor(int socket, const void *data, size_t size, int fd, const void *payload,
                           size_t payload_size, int flags) {
  iovec buffers[] = {{const_cast<void *>(data), size}, {const_cast<void *>(payload), payload_size}};
  alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))]{};
  msghdr message{};
  message.msg_iov = buffers;
  message.msg_iovlen = payload_size ? 2 : 1;
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
    sent = host_call(libc_passthrough().sendmsg, ::sendmsg, socket, &message, MSG_NOSIGNAL | flags);
  } while (sent < 0 && errno == EINTR);
  if (sent == static_cast<long>(size + payload_size))
    return 0;
  if (sent >= 0)
    errno = EIO;
  return -1;
}

int send_keeper_startup_reply(int socket, int error, const KeeperCookie &cookie,
                              uint64_t process_inode) {
  const KeeperStartupReply reply{kKeeperProtocolVersion, error, cookie, process_inode};
  return send_keeper_descriptor(socket, &reply, sizeof(reply), -1);
}

long receive_keeper_descriptor(int socket, void *data, size_t size, int &fd, int flags,
                               bool *missing_descriptor, pid_t *sender_pid) {
  iovec buffer{data, size};
  alignas(cmsghdr) char control[CMSG_SPACE(sizeof(ucred)) + CMSG_SPACE(sizeof(int))]{};
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
  if (sender_pid)
    *sender_pid = -1;
  if (missing_descriptor)
    *missing_descriptor = false;
  unsigned descriptors = 0;
  if (received < 0)
    return received;
  for (auto *header = CMSG_FIRSTHDR(&message); header; header = CMSG_NXTHDR(&message, header)) {
    if (header->cmsg_level != SOL_SOCKET)
      continue;
    if (sender_pid && header->cmsg_type == SCM_CREDENTIALS &&
        header->cmsg_len == CMSG_LEN(sizeof(ucred))) {
      ucred credentials{};
      std::memcpy(&credentials, CMSG_DATA(header), sizeof(credentials));
      *sender_pid = credentials.pid;
    }
    if (header->cmsg_type != SCM_RIGHTS)
      continue;
    const auto count = (header->cmsg_len - CMSG_LEN(0)) / sizeof(int);
    for (size_t index = 0; index < count; ++index) {
      int incoming;
      std::memcpy(&incoming, CMSG_DATA(header) + index * sizeof(int), sizeof(incoming));
      if (++descriptors == 1)
        fd = incoming;
      else
        host_close_fd(incoming);
    }
  }
  // A full receiving fd table drops SCM_RIGHTS but preserves the request bytes.
  if (missing_descriptor && received > 0 && !descriptors && (message.msg_flags & MSG_CTRUNC) &&
      !(message.msg_flags & MSG_TRUNC)) {
    *missing_descriptor = true;
    return received;
  }
  // Even an empty sequenced packet can install SCM_RIGHTS descriptors.
  if (descriptors > 1 || (!received && fd >= 0) || (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC))) {
    host_close_fd(fd);
    fd = -1;
    errno = EPROTO;
    return -1;
  }
  return received;
}

} // namespace rocjitsu::kmd
