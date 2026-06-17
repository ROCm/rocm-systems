// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file transport.h
/// @brief Abstract RPC transport for daemon ↔ client communication.

#ifndef ROCJITSU_KMD_LINUX_TRANSPORT_H_
#define ROCJITSU_KMD_LINUX_TRANSPORT_H_

#include <cstddef>
#include <memory>
#include <string>
#include <sys/types.h>

namespace rocjitsu {

/// @brief Abstract byte-stream transport with optional shared-memory handle passing.
class Transport {
public:
  virtual ~Transport() = default;

  virtual bool send(const void *data, size_t len) = 0;
  virtual bool recv(void *data, size_t len) = 0;

  virtual ssize_t send_with_handle(const void *data, size_t len, int handle) = 0;
  virtual ssize_t recv_with_handle(void *data, size_t len, int *handle_out,
                                   size_t *num_handles) = 0;

  virtual void close() = 0;
};

/// @brief Unix domain socket transport with SCM_RIGHTS handle passing.
class UnixTransport : public Transport {
public:
  [[nodiscard]] static std::unique_ptr<UnixTransport> listen(const std::string &endpoint);
  [[nodiscard]] std::unique_ptr<UnixTransport> accept();
  [[nodiscard]] static std::unique_ptr<UnixTransport> connect(const std::string &endpoint);

  explicit UnixTransport(int fd);
  ~UnixTransport() override;

  bool send(const void *data, size_t len) override;
  bool recv(void *data, size_t len) override;
  ssize_t send_with_handle(const void *data, size_t len, int handle) override;
  ssize_t recv_with_handle(void *data, size_t len, int *handle_out, size_t *num_handles) override;
  void close() override;

  [[nodiscard]] int fd() const { return fd_; }

private:
  int fd_ = -1;
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_TRANSPORT_H_
