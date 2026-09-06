// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "vfio_user_client.h"

// libvfio-user's headers pull in <stdint.h>, <sys/uio.h> and others before
// opening their own linkage block, so those are included first here, outside the
// wrapper, rather than being dragged into C linkage by it. The C++ spelling is
// used where there is one; the rest are POSIX interfaces with no C++ header.
//
// The wrapper is then only around libvfio-user's own declarations. v0.8 guards
// them itself, which makes this nest harmlessly, but releases before it do not,
// and a consumer building against one of those otherwise has to work around the
// name mangling from outside the project.
#include <cstdint>

#include <sys/queue.h>
#include <sys/uio.h>
#include <syslog.h>
#include <unistd.h>

extern "C" {
#include <libvfio-user.h>
#include <vfio-user.h>
}

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <format>

namespace rocjitsu::test {
namespace {

/// @brief The fixed part of an unmap request.
///
/// @details The wire struct ends in a flexible array for the dirty-page bitmap,
/// which C++ will not let us instantiate, and the tests never ask for one.
struct DmaUnmapRequest {
  uint32_t argsz;
  uint32_t flags;
  uint64_t addr;
  uint64_t size;
} __attribute__((packed));
static_assert(sizeof(DmaUnmapRequest) == 24, "unmap request must match the wire layout");

/// @brief Largest reply the tests exchange, which bounds a bad msg_size.
constexpr uint32_t kMaxReplyBytes = 64 * 1024;

bool read_exactly(int socket, void *into, std::size_t length) {
  auto *cursor = static_cast<std::byte *>(into);
  while (length > 0) {
    const ssize_t got = ::read(socket, cursor, length);
    if (got <= 0) {
      return false;
    }
    cursor += got;
    length -= static_cast<std::size_t>(got);
  }
  return true;
}

} // namespace

VfioUserClient::~VfioUserClient() { close_socket(); }

bool VfioUserClient::send(uint16_t command, std::span<const std::byte> payload, int fd) {
  vfio_user_header header{};
  header.msg_id = next_message_id_++;
  header.cmd = command;
  header.msg_size = static_cast<uint32_t>(sizeof(header) + payload.size());
  header.flags = VFIO_USER_F_TYPE_COMMAND;

  iovec parts[2] = {
      {.iov_base = &header, .iov_len = sizeof(header)},
      {.iov_base = const_cast<std::byte *>(payload.data()), .iov_len = payload.size()}};
  msghdr message{};
  message.msg_iov = parts;
  message.msg_iovlen = payload.empty() ? 1 : 2;

  // A shared window is only usable by the server if the descriptor backing it
  // travels with the message.
  alignas(cmsghdr) std::array<std::byte, CMSG_SPACE(sizeof(int))> control{};
  if (fd >= 0) {
    message.msg_control = control.data();
    message.msg_controllen = control.size();
    cmsghdr *descriptor = CMSG_FIRSTHDR(&message);
    descriptor->cmsg_level = SOL_SOCKET;
    descriptor->cmsg_type = SCM_RIGHTS;
    descriptor->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(descriptor), &fd, sizeof(fd));
  }

  // SOCK_STREAM may accept less than the whole message, which would leave the
  // server reading a header whose msg_size never arrives -- framing corruption
  // that looks like a protocol bug rather than a short write. Only the first
  // sendmsg carries the descriptor; the remainder is a plain byte stream.
  ssize_t sent = ::sendmsg(socket_, &message, 0);
  if (sent < 0) {
    return false;
  }
  const auto total = static_cast<std::size_t>(header.msg_size);
  auto done = static_cast<std::size_t>(sent);
  while (done < total) {
    const std::byte *rest = nullptr;
    std::size_t remaining = 0;
    if (done < sizeof(header)) {
      rest = reinterpret_cast<const std::byte *>(&header) + done;
      remaining = sizeof(header) - done;
    } else {
      rest = payload.data() + (done - sizeof(header));
      remaining = total - done;
    }
    const ssize_t more = ::send(socket_, rest, remaining, 0);
    if (more <= 0) {
      return false;
    }
    done += static_cast<std::size_t>(more);
  }
  return done == total;
}

bool VfioUserClient::receive(std::vector<std::byte> &payload) {
  vfio_user_header header{};
  if (!read_exactly(socket_, &header, sizeof(header))) {
    return false;
  }
  if (header.msg_size < sizeof(header) || header.msg_size > kMaxReplyBytes) {
    return false;
  }

  payload.resize(header.msg_size - sizeof(header));
  if (!payload.empty() && !read_exactly(socket_, payload.data(), payload.size())) {
    return false;
  }
  // The server reports a refused access by flagging the reply, which is how a
  // test tells "the device said no" from "the transport broke".
  return (header.flags & VFIO_USER_F_ERROR) == 0;
}

bool VfioUserClient::request(uint16_t command, std::span<const std::byte> payload, int fd,
                             std::vector<std::byte> &reply) {
  return send(command, payload, fd) && receive(reply);
}

bool VfioUserClient::connect(const std::string &socket_path) {
  // Refused rather than truncated: snprintf would silently shorten an overlong
  // path and then connect to a DIFFERENT socket, which presents as the server
  // never answering.
  sockaddr_un address{};
  if (socket_path.size() >= sizeof(address.sun_path)) {
    return false;
  }

  close_socket();
  socket_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (socket_ < 0) {
    return false;
  }

  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
  if (::connect(socket_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
    // Callers retry while the server reaches its accept loop, so a descriptor
    // left behind here is one leaked per attempt.
    close_socket();
    return false;
  }

  // The version exchange carries each side's limits as JSON. The server replies
  // with its own, which the tests do not need beyond the handshake succeeding.
  const std::string capabilities =
      std::format(R"({{"capabilities":{{"max_msg_fds":{},"max_data_xfer_size":{}}}}})", 8,
                  VFIO_USER_DEFAULT_MAX_DATA_XFER_SIZE);
  std::vector<std::byte> payload(sizeof(vfio_user_version) + capabilities.size() + 1);
  auto *version = reinterpret_cast<vfio_user_version *>(payload.data());
  version->major = LIB_VFIO_USER_MAJOR;
  version->minor = LIB_VFIO_USER_MINOR;
  std::memcpy(payload.data() + sizeof(vfio_user_version), capabilities.c_str(),
              capabilities.size() + 1);

  std::vector<std::byte> reply;
  if (!request(VFIO_USER_VERSION, payload, -1, reply)) {
    close_socket();
    return false;
  }
  return true;
}

void VfioUserClient::close_socket() {
  if (socket_ >= 0) {
    ::close(socket_);
    socket_ = -1;
  }
}

bool VfioUserClient::device_info(uint32_t &region_count, uint32_t &irq_count) {
  vfio_user_device_info request_body{};
  request_body.argsz = sizeof(request_body);

  std::vector<std::byte> reply;
  if (!request(VFIO_USER_DEVICE_GET_INFO,
               {reinterpret_cast<const std::byte *>(&request_body), sizeof(request_body)}, -1,
               reply) ||
      reply.size() < sizeof(vfio_user_device_info)) {
    return false;
  }

  const auto *info = reinterpret_cast<const vfio_user_device_info *>(reply.data());
  region_count = info->num_regions;
  irq_count = info->num_irqs;
  return true;
}

bool VfioUserClient::irq_info(uint32_t index, uint32_t &count) {
  vfio_irq_info request_body{};
  request_body.argsz = sizeof(request_body);
  request_body.index = index;

  std::vector<std::byte> reply;
  if (!request(VFIO_USER_DEVICE_GET_IRQ_INFO,
               {reinterpret_cast<const std::byte *>(&request_body), sizeof(request_body)}, -1,
               reply) ||
      reply.size() < sizeof(vfio_irq_info)) {
    return false;
  }

  count = reinterpret_cast<const vfio_irq_info *>(reply.data())->count;
  return true;
}

bool VfioUserClient::arm_irq(uint32_t index, int fd) {
  // The wire struct ends in a flexible array, which C++ will not let us nest or
  // instantiate, so the payload is built as bytes. One vector needs one
  // descriptor, and it rides in the message's control data rather than here.
  std::vector<std::byte> payload(sizeof(vfio_irq_set));
  auto *set = reinterpret_cast<vfio_irq_set *>(payload.data());
  set->argsz = static_cast<uint32_t>(payload.size());
  set->flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER;
  set->index = index;
  set->start = 0;
  set->count = 1;

  std::vector<std::byte> reply;
  return request(VFIO_USER_DEVICE_SET_IRQS, payload, fd, reply);
}

bool VfioUserClient::region_info(uint32_t region, uint64_t &size, uint32_t &flags) {
  vfio_region_info request_body{};
  request_body.argsz = sizeof(request_body);
  request_body.index = region;

  std::vector<std::byte> reply;
  if (!request(VFIO_USER_DEVICE_GET_REGION_INFO,
               {reinterpret_cast<const std::byte *>(&request_body), sizeof(request_body)}, -1,
               reply) ||
      reply.size() < sizeof(vfio_region_info)) {
    return false;
  }

  const auto *info = reinterpret_cast<const vfio_region_info *>(reply.data());
  size = info->size;
  flags = info->flags;
  return true;
}

bool VfioUserClient::region_read(uint32_t region, uint64_t offset, std::span<std::byte> into) {
  std::vector<std::byte> payload(sizeof(vfio_user_region_access));
  auto *access = reinterpret_cast<vfio_user_region_access *>(payload.data());
  access->offset = offset;
  access->region = region;
  access->count = static_cast<uint32_t>(into.size());

  std::vector<std::byte> reply;
  if (!request(VFIO_USER_REGION_READ, payload, -1, reply) ||
      reply.size() < sizeof(vfio_user_region_access) + into.size()) {
    return false;
  }
  std::memcpy(into.data(), reply.data() + sizeof(vfio_user_region_access), into.size());
  return true;
}

bool VfioUserClient::region_write(uint32_t region, uint64_t offset,
                                  std::span<const std::byte> from) {
  std::vector<std::byte> payload(sizeof(vfio_user_region_access) + from.size());
  auto *access = reinterpret_cast<vfio_user_region_access *>(payload.data());
  access->offset = offset;
  access->region = region;
  access->count = static_cast<uint32_t>(from.size());
  std::memcpy(payload.data() + sizeof(vfio_user_region_access), from.data(), from.size());

  std::vector<std::byte> reply;
  return request(VFIO_USER_REGION_WRITE, payload, -1, reply);
}

bool VfioUserClient::dma_map(uint64_t iova, uint64_t size, int fd, uint64_t fd_offset,
                             DmaProtection protection) {
  vfio_user_dma_map request_body{};
  request_body.argsz = sizeof(request_body);
  request_body.flags = 0;
  if (protection == DmaProtection::ReadOnly || protection == DmaProtection::ReadWrite) {
    request_body.flags |= VFIO_USER_F_DMA_REGION_READ;
  }
  if (protection == DmaProtection::WriteOnly || protection == DmaProtection::ReadWrite) {
    request_body.flags |= VFIO_USER_F_DMA_REGION_WRITE;
  }
  if (fd >= 0) {
    request_body.flags |= VFIO_USER_F_DMA_REGION_MMAP;
  }
  request_body.offset = fd_offset;
  request_body.addr = iova;
  request_body.size = size;

  std::vector<std::byte> reply;
  return request(VFIO_USER_DMA_MAP,
                 {reinterpret_cast<const std::byte *>(&request_body), sizeof(request_body)}, fd,
                 reply);
}

bool VfioUserClient::dma_unmap(uint64_t iova, uint64_t size) {
  DmaUnmapRequest request_body{};
  request_body.argsz = sizeof(request_body);
  request_body.addr = iova;
  request_body.size = size;

  std::vector<std::byte> reply;
  return request(VFIO_USER_DMA_UNMAP,
                 {reinterpret_cast<const std::byte *>(&request_body), sizeof(request_body)}, -1,
                 reply);
}

} // namespace rocjitsu::test
