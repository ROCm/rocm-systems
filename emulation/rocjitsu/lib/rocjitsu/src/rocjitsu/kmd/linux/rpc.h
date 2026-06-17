// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_RPC_H_
#define ROCJITSU_KMD_LINUX_RPC_H_

/// @file rpc.h
/// @brief RPC format for the rocjitsu daemon ↔ client Unix socket protocol.
///
/// @details All messages use a fixed 16-byte RpcHeader followed by an
/// opcode-specific payload. For RPC_IOCTL, the payload contains an
/// RpcIoctlRequest header followed by the raw ioctl args. Embedded pointer
/// arrays are inlined after the args. File descriptors (memfds for GPU
/// memory) are passed via SCM_RIGHTS ancillary messages.

#include "rocjitsu/vm/rj_vm.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

namespace rocjitsu {

/// @brief RPC operation codes.
enum RpcOpcode : uint16_t {
  RPC_HANDSHAKE, ///< Initial handshake — response carries topology path.
  RPC_OPEN,      ///< Open the KFD device.
  RPC_CLOSE,     ///< Close the client connection.
  RPC_MMAP,      ///< Map GPU memory — payload is RpcMmapRequest.
  RPC_MUNMAP,    ///< Unmap GPU memory — payload is RpcMunmapRequest.
  RPC_IOCTL,     ///< KFD ioctl passthrough — payload is RpcIoctlRequest + args.
};

/// @brief Unified RPC message header (16 bytes, fixed size).
/// @details Used for both requests and responses. For requests, `result` is 0.
/// For responses, `result` carries the return code from the daemon.
struct RpcHeader {
  uint16_t opcode;        ///< RpcOpcode value.
  uint16_t reserved;      ///< Reserved for future use (must be 0).
  uint32_t request_id;    ///< Correlates request to response.
  uint32_t payload_bytes; ///< Size of the payload in bytes following this header.
  int32_t result; ///< 0 for requests; return code for responses (negative errno on failure).
};

/// @brief RPC protocol version. Increment when making breaking changes.
inline constexpr uint32_t kRpcProtocolVersion = 3;

/// @brief Fixed-size GPU metadata sent during daemon handshake.
using RpcGpuInfo = ::rj_vm_gpu_info_t;

/// @brief Handshake response payload (sent after RPC_HANDSHAKE).
struct RpcHandshakeResponse {
  uint32_t version;           ///< Protocol version (kRpcProtocolVersion).
  uint32_t gpu_id;            ///< KFD gpu_id for the simulated device.
  uint32_t topology_path_len; ///< Length of the topology path string that follows.
  uint32_t drm_path_len;      ///< Length of the DRM sysfs path string that follows.
  RpcGpuInfo gpu_info;        ///< Device metadata for client-side DRM/libdrm emulation.
};

/// @brief Ioctl request payload (when opcode == RPC_IOCTL).
/// @details Followed by `args_bytes` of raw ioctl arg data. For ioctls with
/// embedded pointers, the pointed-to arrays are inlined after the args.
struct RpcIoctlRequest {
  uint32_t ioctl_cmd;  ///< AMDKFD_IOC_* ioctl number.
  uint32_t args_bytes; ///< Size in bytes of the ioctl args (and any inlined arrays) that follow.
};

/// @brief mmap request payload (when opcode == RPC_MMAP).
struct RpcMmapRequest {
  uint64_t addr;   ///< Requested address (may be MAP_FIXED).
  uint64_t length; ///< Length in bytes to map.
  int32_t prot;    ///< Memory protection flags (PROT_READ, PROT_WRITE, etc.).
  int32_t flags;   ///< Mapping flags (MAP_SHARED, MAP_FIXED, etc.).
  int64_t offset;  ///< KFD mmap offset encoding (type + gpu_id + handle).
};

/// @brief mmap response payload (when opcode == RPC_MMAP).
struct RpcMmapResponse {
  uint64_t mapped_addr; ///< Address the daemon mapped at (informational).
};

/// @brief munmap request payload (when opcode == RPC_MUNMAP).
struct RpcMunmapRequest {
  uint64_t addr;   ///< Address of the mapping to unmap.
  uint64_t length; ///< Length in bytes to unmap.
};

static_assert(sizeof(RpcHeader) == 16);
static_assert(sizeof(RpcGpuInfo) == 312);
static_assert(sizeof(RpcHandshakeResponse) == 328);
static_assert(sizeof(RpcIoctlRequest) == 8);
static_assert(sizeof(RpcMmapRequest) == 32);
static_assert(sizeof(RpcMmapResponse) == 8);
static_assert(sizeof(RpcMunmapRequest) == 16);

/// @brief Send a message with optional ancillary file descriptors.
/// @param sock Connected Unix domain socket fd.
/// @param data Pointer to the message data to send.
/// @param data_len Size of the message data in bytes.
/// @param fds Array of file descriptors to pass via SCM_RIGHTS (may be nullptr).
/// @param num_fds Number of file descriptors in the fds array.
/// @retval >0 Number of bytes sent.
/// @retval -1 Send failed; errno is set by sendmsg(2).
inline constexpr size_t kRpcMaxFds = 4;

inline ssize_t rpc_send_msg(int sock, const void *data, size_t data_len, const int *fds = nullptr,
                            size_t num_fds = 0) {
  if (num_fds > kRpcMaxFds) {
    errno = EINVAL;
    return -1;
  }
  iovec iov{const_cast<void *>(data), data_len};
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  union {
    cmsghdr align;
    char buf[CMSG_SPACE(sizeof(int) * kRpcMaxFds)];
  } cmsg_buf{};

  if (fds && num_fds > 0) {
    msg.msg_control = cmsg_buf.buf;
    msg.msg_controllen = CMSG_SPACE(num_fds * sizeof(int));
    auto *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(num_fds * sizeof(int));
    std::memcpy(CMSG_DATA(cmsg), fds, num_fds * sizeof(int));
  }

  return sendmsg(sock, &msg, MSG_NOSIGNAL);
}

/// @brief Receive a message with optional ancillary file descriptors.
/// @param sock Connected Unix domain socket fd.
/// @param data Buffer to receive the message data into.
/// @param data_len Size of the receive buffer in bytes.
/// @param[out] fds Buffer for received file descriptors (may be nullptr).
/// @param[in,out] num_fds On input: capacity of the fds buffer. On output:
/// number of file descriptors actually received.
/// @retval >0 Number of bytes received.
/// @retval 0 Peer closed the connection.
/// @retval -1 Receive failed; errno is set by recvmsg(2).
inline ssize_t rpc_recv_msg(int sock, void *data, size_t data_len, int *fds = nullptr,
                            size_t *num_fds = nullptr) {
  iovec iov{data, data_len};
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  union {
    cmsghdr align;
    char buf[CMSG_SPACE(sizeof(int) * kRpcMaxFds)];
  } cmsg_buf{};

  if (fds && num_fds && *num_fds > 0) {
    msg.msg_control = cmsg_buf.buf;
    msg.msg_controllen = sizeof(cmsg_buf.buf);
  }

  ssize_t bytes_received = recvmsg(sock, &msg, MSG_CMSG_CLOEXEC | MSG_WAITALL);
  if (bytes_received <= 0) {
    if (num_fds)
      *num_fds = 0;
    return bytes_received;
  }

  size_t fd_count = 0;
  if (fds && num_fds) {
    size_t capacity = *num_fds;
    for (auto *cmsg = CMSG_FIRSTHDR(&msg); cmsg; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
      if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
        size_t ancillary_bytes = cmsg->cmsg_len - CMSG_LEN(0);
        size_t count = ancillary_bytes / sizeof(int);
        auto *received = reinterpret_cast<int *>(CMSG_DATA(cmsg));
        for (size_t i = 0; i < count; ++i) {
          if (fd_count < capacity)
            fds[fd_count] = received[i];
          else
            ::close(received[i]);
          ++fd_count;
        }
      }
    }
    *num_fds = std::min(fd_count, capacity);
  }

  return bytes_received;
}

/// @brief Read exactly `total_bytes` bytes from a socket, handling partial reads.
/// @param sock Connected Unix domain socket fd.
/// @param buffer Buffer to read into.
/// @param total_bytes Number of bytes to read.
/// @retval true All bytes were read successfully.
/// @retval false Connection closed or recv(2) returned an error (errno is set).
inline bool rpc_recv_exact(int sock, void *buffer, size_t total_bytes) {
  auto *cursor = static_cast<uint8_t *>(buffer);
  size_t remaining = total_bytes;
  while (remaining > 0) {
    ssize_t bytes_read = recv(sock, cursor, remaining, 0);
    if (bytes_read < 0 && errno == EINTR)
      continue;
    if (bytes_read <= 0)
      return false;
    cursor += bytes_read;
    remaining -= static_cast<size_t>(bytes_read);
  }
  return true;
}

/// @brief Send exactly `total_bytes` bytes to a socket, handling partial writes.
/// @param sock Connected Unix domain socket fd.
/// @param buffer Buffer to send from.
/// @param total_bytes Number of bytes to send.
/// @retval true All bytes were sent successfully.
/// @retval false Connection closed or send(2) returned an error (errno is set).
inline bool rpc_send_exact(int sock, const void *buffer, size_t total_bytes) {
  auto *cursor = static_cast<const uint8_t *>(buffer);
  size_t remaining = total_bytes;
  while (remaining > 0) {
    ssize_t bytes_sent = send(sock, cursor, remaining, MSG_NOSIGNAL);
    if (bytes_sent < 0 && errno == EINTR)
      continue;
    if (bytes_sent <= 0)
      return false;
    cursor += bytes_sent;
    remaining -= static_cast<size_t>(bytes_sent);
  }
  return true;
}

/// @brief Per-user runtime directory for rocjitsu state files.
/// @details Checks $ROCJITSU_RUNTIME_DIR first (used by test infrastructure
/// for per-process isolation), then $XDG_RUNTIME_DIR/rocjitsu, falling back
/// to /tmp/rocjitsu-<uid>.
inline std::string rpc_default_runtime_dir() {
  if (const char *rj = getenv("ROCJITSU_RUNTIME_DIR"))
    return rj;
  if (const char *xdg = getenv("XDG_RUNTIME_DIR"))
    return std::string(xdg) + "/rocjitsu";
  return "/tmp/rocjitsu-" + std::to_string(getuid());
}

/// @brief Default daemon socket path for the current user.
inline std::string rpc_default_socket_path() { return rpc_default_runtime_dir() + "/daemon.sock"; }

/// @brief Path to the config file written by the CLI for interposer discovery.
inline std::string rpc_default_config_file_path() {
  return rpc_default_runtime_dir() + "/config_path";
}

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_RPC_H_
