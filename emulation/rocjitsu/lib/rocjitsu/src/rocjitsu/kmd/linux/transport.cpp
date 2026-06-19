// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/transport.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include <filesystem>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

namespace rocjitsu {

UnixTransport::UnixTransport(int fd) : fd_(fd) {}

UnixTransport::~UnixTransport() { close(); }

std::unique_ptr<UnixTransport> UnixTransport::listen(const std::string &endpoint) {
  int sock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock < 0)
    return nullptr;

  std::filesystem::create_directories(std::filesystem::path(endpoint).parent_path());
  unlink(endpoint.c_str());

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  endpoint.copy(addr.sun_path, sizeof(addr.sun_path) - 1);

  if (bind(sock, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
      ::listen(sock, 16) != 0) {
    ::close(sock);
    return nullptr;
  }

  return std::make_unique<UnixTransport>(sock);
}

std::unique_ptr<UnixTransport> UnixTransport::accept() {
  int client = ::accept(fd_, nullptr, nullptr);
  if (client < 0)
    return nullptr;
  return std::make_unique<UnixTransport>(client);
}

std::unique_ptr<UnixTransport> UnixTransport::connect(const std::string &endpoint) {
  int sock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock < 0)
    return nullptr;

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  endpoint.copy(addr.sun_path, sizeof(addr.sun_path) - 1);

  if (::connect(sock, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
    syscall(SYS_close, sock);
    return nullptr;
  }

  return std::make_unique<UnixTransport>(sock);
}

bool UnixTransport::send(const void *data, size_t len) { return rpc_send_exact(fd_, data, len); }

bool UnixTransport::recv(void *data, size_t len) { return rpc_recv_exact(fd_, data, len); }

ssize_t UnixTransport::send_with_handle(const void *data, size_t len, int handle) {
  return rpc_send_msg(fd_, data, len, &handle, 1);
}

ssize_t UnixTransport::recv_with_handle(void *data, size_t len, int *handle_out,
                                        size_t *num_handles) {
  return rpc_recv_msg(fd_, data, len, handle_out, num_handles);
}

void UnixTransport::close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

} // namespace rocjitsu
