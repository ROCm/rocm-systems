// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rccl_tcp_bootstrap.h
/// @brief Provide a bounded multi-rank TCP rendezvous for RCCL tests.
/// @details Rank 0 holds one connection per other rank. Integers on the wire
/// use network byte order. Connections persist across cases so an old run or
/// another test filter cannot supply an NCCL ID.

#pragma once

#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace rocjitsu::test::rccl_tcp_bootstrap {

enum class Role { Listen, Connect };

struct Config {
  Role role;
  std::string endpoint;
  std::string run_id;
  int rank;
  int world_size;
  uint32_t seed;
  size_t id_size;
  std::chrono::milliseconds timeout;
};

namespace detail {

using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;
constexpr uint32_t kMagic = 0x524a4253; // RJBS
constexpr uint16_t kVersion = 1;
constexpr uint32_t kMaxPayload = 4096;
constexpr size_t kRunIdLength = 32;
constexpr size_t kMaxCaseName = 1024;

enum class FrameType : uint16_t { Hello = 1, Case = 2, Id = 3, Ack = 4 };

class Fd {
public:
  explicit Fd(int value = -1) : value_(value) {}
  ~Fd() { reset(); }
  Fd(const Fd &) = delete;
  Fd &operator=(const Fd &) = delete;
  Fd(Fd &&other) noexcept : value_(std::exchange(other.value_, -1)) {}
  Fd &operator=(Fd &&other) noexcept {
    if (this != &other) {
      reset();
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  int get() const { return value_; }
  void reset() {
    if (value_ >= 0)
      (void)::close(value_);
    value_ = -1;
  }

private:
  int value_;
};

[[noreturn]] inline void fail_errno(std::string_view action) {
  throw std::system_error(errno, std::generic_category(), std::string(action));
}

inline void set_nonblocking(int fd) {
  int flags = ::fcntl(fd, F_GETFL);
  if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
    fail_errno("set TCP bootstrap socket nonblocking");
  flags = ::fcntl(fd, F_GETFD);
  if (flags < 0 || ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) < 0)
    fail_errno("set TCP bootstrap socket close-on-exec");
}

inline void wait_ready(int fd, short events, Deadline deadline) {
  for (;;) {
    const auto remaining = deadline - Clock::now();
    if (remaining <= Clock::duration::zero())
      throw std::runtime_error("TCP bootstrap timeout");
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        remaining + std::chrono::milliseconds(1))
                        .count();
    pollfd pfd{fd, events, 0};
    const int rc = ::poll(&pfd, 1, static_cast<int>(std::min<int64_t>(ms, INT_MAX)));
    if (rc < 0) {
      if (errno == EINTR)
        continue;
      fail_errno("poll TCP bootstrap socket");
    }
    if (rc == 0)
      continue;
    if (pfd.revents & POLLNVAL)
      throw std::runtime_error("TCP bootstrap socket is invalid");
    // Let accept, connect, send, or recv report the specific peer error.
    if (pfd.revents & (events | POLLERR | POLLHUP))
      return;
  }
}

inline void write_exact(int fd, const void *data, size_t size, Deadline deadline) {
  const auto *bytes = static_cast<const uint8_t *>(data);
  size_t done = 0;
  while (done < size) {
    if (Clock::now() >= deadline)
      throw std::runtime_error("TCP bootstrap timeout");
    const ssize_t n = ::send(fd, bytes + done, size - done, MSG_NOSIGNAL);
    if (n > 0) {
      done += static_cast<size_t>(n);
    } else if (n < 0 && errno == EINTR) {
      continue;
    } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      wait_ready(fd, POLLOUT, deadline);
    } else if (n == 0) {
      throw std::runtime_error("TCP bootstrap peer disconnected during write");
    } else {
      fail_errno("send TCP bootstrap frame");
    }
  }
}

inline void read_exact(int fd, void *data, size_t size, Deadline deadline) {
  auto *bytes = static_cast<uint8_t *>(data);
  size_t done = 0;
  while (done < size) {
    if (Clock::now() >= deadline)
      throw std::runtime_error("TCP bootstrap timeout");
    const ssize_t n = ::recv(fd, bytes + done, size - done, 0);
    if (n > 0) {
      done += static_cast<size_t>(n);
    } else if (n == 0) {
      throw std::runtime_error("TCP bootstrap peer disconnected during read");
    } else if (errno == EINTR) {
      continue;
    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
      wait_ready(fd, POLLIN, deadline);
    } else {
      fail_errno("receive TCP bootstrap frame");
    }
  }
}

inline void append_u16(std::vector<uint8_t> &bytes, uint16_t value) {
  bytes.push_back(static_cast<uint8_t>(value >> 8));
  bytes.push_back(static_cast<uint8_t>(value));
}

inline void append_u32(std::vector<uint8_t> &bytes, uint32_t value) {
  bytes.push_back(static_cast<uint8_t>(value >> 24));
  bytes.push_back(static_cast<uint8_t>(value >> 16));
  bytes.push_back(static_cast<uint8_t>(value >> 8));
  bytes.push_back(static_cast<uint8_t>(value));
}

class Reader {
public:
  explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {}
  uint16_t u16() {
    require(2);
    const uint16_t value = (static_cast<uint16_t>(bytes_[offset_]) << 8) | bytes_[offset_ + 1];
    offset_ += 2;
    return value;
  }
  uint32_t u32() {
    require(4);
    const uint32_t value = (static_cast<uint32_t>(bytes_[offset_]) << 24) |
                           (static_cast<uint32_t>(bytes_[offset_ + 1]) << 16) |
                           (static_cast<uint32_t>(bytes_[offset_ + 2]) << 8) | bytes_[offset_ + 3];
    offset_ += 4;
    return value;
  }
  std::string string(size_t size) {
    require(size);
    if (size == 0)
      return {};
    std::string value(reinterpret_cast<const char *>(bytes_.data() + offset_), size);
    offset_ += size;
    return value;
  }
  void finish() const {
    if (offset_ != bytes_.size())
      throw std::runtime_error("TCP bootstrap frame has trailing bytes");
  }

private:
  void require(size_t size) const {
    if (size > bytes_.size() - offset_)
      throw std::runtime_error("TCP bootstrap frame is truncated");
  }
  std::span<const uint8_t> bytes_;
  size_t offset_ = 0;
};

inline void write_frame(int fd, FrameType type, std::span<const uint8_t> payload,
                        Deadline deadline) {
  if (payload.size() > kMaxPayload)
    throw std::invalid_argument("TCP bootstrap frame is too large");
  std::vector<uint8_t> header;
  header.reserve(12);
  append_u32(header, kMagic);
  append_u16(header, kVersion);
  append_u16(header, static_cast<uint16_t>(type));
  append_u32(header, static_cast<uint32_t>(payload.size()));
  write_exact(fd, header.data(), header.size(), deadline);
  write_exact(fd, payload.data(), payload.size(), deadline);
}

inline std::vector<uint8_t> read_frame(int fd, FrameType expected, Deadline deadline) {
  std::array<uint8_t, 12> header{};
  read_exact(fd, header.data(), header.size(), deadline);
  Reader reader(header);
  const uint32_t magic = reader.u32();
  const uint16_t version = reader.u16();
  const uint16_t type = reader.u16();
  const uint32_t size = reader.u32();
  if (magic != kMagic || version != kVersion)
    throw std::runtime_error("TCP bootstrap magic or version mismatch");
  if (type != static_cast<uint16_t>(expected))
    throw std::runtime_error("TCP bootstrap unexpected frame type");
  if (size > kMaxPayload)
    throw std::runtime_error("TCP bootstrap frame exceeds size limit");
  std::vector<uint8_t> payload(size);
  read_exact(fd, payload.data(), payload.size(), deadline);
  return payload;
}

inline void validate_config(const Config &config) {
  if ((config.role != Role::Listen && config.role != Role::Connect) || config.world_size < 2 ||
      config.world_size > 64 || config.rank < 0 || config.rank >= config.world_size ||
      (config.role == Role::Listen && config.rank != 0) ||
      (config.role == Role::Connect && config.rank == 0))
    throw std::invalid_argument(
        "TCP bootstrap requires listener rank 0 and distinct connector ranks");
  if (config.run_id.size() != kRunIdLength ||
      !std::all_of(config.run_id.begin(), config.run_id.end(), [](unsigned char ch) {
        return (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F') || (ch >= '0' && ch <= '9');
      }))
    throw std::invalid_argument("TCP bootstrap run ID must be exactly 32 ASCII hex digits");
  if (config.id_size == 0 || config.id_size > kMaxPayload)
    throw std::invalid_argument("TCP bootstrap NCCL ID size is invalid");
  if (config.timeout.count() < 1 || config.timeout.count() > 600000)
    throw std::invalid_argument("TCP bootstrap timeout must be 1-600000 ms");
}

inline sockaddr_in parse_endpoint(std::string_view endpoint) {
  const size_t colon = endpoint.find(':');
  if (colon == std::string_view::npos || endpoint.find(':', colon + 1) != std::string_view::npos)
    throw std::invalid_argument("TCP bootstrap endpoint must be IPv4:PORT");
  const std::string ip(endpoint.substr(0, colon));
  const auto port_text = endpoint.substr(colon + 1);
  uint32_t port = 0;
  const auto [end, error] =
      std::from_chars(port_text.data(), port_text.data() + port_text.size(), port);
  if (error != std::errc{} || end != port_text.data() + port_text.size() || port == 0 ||
      port > 65535)
    throw std::invalid_argument("TCP bootstrap endpoint port must be 1-65535");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(port));
  if (::inet_pton(AF_INET, ip.c_str(), &address.sin_addr) != 1)
    throw std::invalid_argument("TCP bootstrap endpoint requires numeric IPv4");
  return address;
}

inline Fd open_socket(const Config &config, Deadline deadline) {
  const sockaddr_in address = parse_endpoint(config.endpoint);
  if (config.role == Role::Connect) {
    // A guest may start before its peer has bound the listener. Retry refused
    // connections within the same deadline, using a new socket each time.
    for (;;) {
      if (Clock::now() >= deadline)
        throw std::runtime_error("TCP bootstrap timeout");
      Fd socket(::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0));
      if (socket.get() < 0)
        fail_errno("create TCP bootstrap socket");
      int error = 0;
      if (::connect(socket.get(), reinterpret_cast<const sockaddr *>(&address), sizeof(address)) <
          0) {
        error = errno;
        if (error == EINPROGRESS || error == EINTR) {
          wait_ready(socket.get(), POLLOUT, deadline);
          socklen_t size = sizeof(error);
          if (::getsockopt(socket.get(), SOL_SOCKET, SO_ERROR, &error, &size) < 0)
            fail_errno("inspect TCP bootstrap connection");
        }
      }
      if (error == 0)
        return socket;
      if (error != ECONNREFUSED && error != EHOSTUNREACH && error != ENETUNREACH)
        throw std::system_error(error, std::generic_category(), "connect TCP bootstrap socket");
      const auto remaining = deadline - Clock::now();
      if (remaining <= Clock::duration::zero())
        throw std::runtime_error("TCP bootstrap timeout");
      const auto delay = std::min(
          remaining, std::chrono::duration_cast<Clock::duration>(std::chrono::milliseconds(50)));
      const int ms =
          static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(delay).count());
      if (::poll(nullptr, 0, std::max(ms, 1)) < 0 && errno != EINTR)
        fail_errno("wait to retry TCP bootstrap connection");
    }
  }

  Fd socket(::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0));
  if (socket.get() < 0)
    fail_errno("create TCP bootstrap socket");
  const int reuse = 1;
  if (::setsockopt(socket.get(), SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0)
    fail_errno("configure TCP bootstrap listener");
  if (::bind(socket.get(), reinterpret_cast<const sockaddr *>(&address), sizeof(address)) < 0)
    fail_errno("bind TCP bootstrap listener");
  if (::listen(socket.get(), config.world_size - 1) < 0)
    fail_errno("listen for TCP bootstrap peers");
  return socket;
}

inline Fd accept_peer(int listener, Deadline deadline) {
  for (;;) {
    if (Clock::now() >= deadline)
      throw std::runtime_error("TCP bootstrap timeout");
    const int peer = ::accept4(listener, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (peer >= 0)
      return Fd(peer);
    if (errno == EINTR)
      continue;
    if (errno != EAGAIN && errno != EWOULDBLOCK)
      fail_errno("accept TCP bootstrap peer");
    wait_ready(listener, POLLIN, deadline);
  }
}

} // namespace detail

class Session {
public:
  // Open a numeric IPv4 endpoint, then exchange a versioned HELLO. The
  // deadline covers connection establishment and the entire HELLO exchange.
  static Session Open(Config config) {
    detail::validate_config(config);
    const auto deadline = detail::Clock::now() + config.timeout;
    auto socket = detail::open_socket(config, deadline);
    Session session(std::move(config));
    if (session.config_.role == Role::Connect) {
      const int peer_rank = session.hello(socket.get(), deadline);
      session.sockets_[peer_rank] = std::move(socket);
    } else {
      for (int connection = 1; connection < session.config_.world_size; ++connection) {
        auto peer = detail::accept_peer(socket.get(), deadline);
        const int peer_rank = session.hello(peer.get(), deadline);
        if (session.sockets_[peer_rank].get() >= 0)
          throw std::runtime_error("TCP bootstrap duplicate connector rank");
        session.sockets_[peer_rank] = std::move(peer);
      }
    }
    return session;
  }

  // Takes ownership of an already connected socket. Useful for transport
  // adapters and host-only protocol tests; the same HELLO checks still apply.
  static Session FromConnectedFd(int fd, Config config) {
    detail::Fd socket(fd);
    if (socket.get() < 0)
      throw std::invalid_argument("TCP bootstrap socket is invalid");
    detail::validate_config(config);
    if (config.world_size != 2)
      throw std::invalid_argument("one connected socket requires a two-rank bootstrap");
    detail::set_nonblocking(socket.get());
    const auto deadline = detail::Clock::now() + config.timeout;
    Session session(std::move(config));
    const int peer_rank = session.hello(socket.get(), deadline);
    session.sockets_[peer_rank] = std::move(socket);
    return session;
  }

  Session(Session &&) noexcept = default;
  Session &operator=(Session &&) noexcept = default;
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  // Every rank announces the full suite.name and ordinal before rank 0 sends
  // the exact NCCL ID bytes. Every connector acknowledges before rank 0 returns.
  void ExchangeCase(std::string_view case_name, uint32_t ordinal, void *id) {
    if (case_name.empty() || case_name.size() > detail::kMaxCaseName || !id)
      throw std::invalid_argument("TCP bootstrap case name or ID buffer is invalid");
    const auto deadline = detail::Clock::now() + config_.timeout;
    try {
      std::vector<uint8_t> payload;
      detail::append_u32(payload, ordinal);
      detail::append_u16(payload, static_cast<uint16_t>(case_name.size()));
      payload.insert(payload.end(), case_name.begin(), case_name.end());
      if (config_.rank == 0) {
        for (int peer_rank = 1; peer_rank < config_.world_size; ++peer_rank)
          detail::write_frame(connected_socket(peer_rank), detail::FrameType::Case, payload,
                              deadline);
        for (int peer_rank = 1; peer_rank < config_.world_size; ++peer_rank)
          read_case(connected_socket(peer_rank), case_name, ordinal, deadline);
        const auto *bytes = static_cast<const uint8_t *>(id);
        for (int peer_rank = 1; peer_rank < config_.world_size; ++peer_rank)
          detail::write_frame(connected_socket(peer_rank), detail::FrameType::Id,
                              std::span(bytes, config_.id_size), deadline);
        for (int peer_rank = 1; peer_rank < config_.world_size; ++peer_rank) {
          auto ack =
              detail::read_frame(connected_socket(peer_rank), detail::FrameType::Ack, deadline);
          detail::Reader ack_reader(ack);
          if (ack_reader.u32() != ordinal)
            throw std::runtime_error("TCP bootstrap acknowledgement ordinal mismatch");
          ack_reader.finish();
        }
      } else {
        const int fd = connected_socket(0);
        detail::write_frame(fd, detail::FrameType::Case, payload, deadline);
        read_case(fd, case_name, ordinal, deadline);
        auto id_bytes = detail::read_frame(fd, detail::FrameType::Id, deadline);
        if (id_bytes.size() != config_.id_size)
          throw std::runtime_error("TCP bootstrap NCCL ID size mismatch");
        std::memcpy(id, id_bytes.data(), id_bytes.size());
        std::vector<uint8_t> ack;
        detail::append_u32(ack, ordinal);
        detail::write_frame(fd, detail::FrameType::Ack, ack, deadline);
      }
    } catch (...) {
      for (auto &socket : sockets_)
        socket.reset();
      throw;
    }
  }

private:
  explicit Session(Config config)
      : sockets_(static_cast<size_t>(config.world_size)), config_(std::move(config)) {}

  int connected_socket(int rank) const {
    const int fd = sockets_[rank].get();
    if (fd < 0)
      throw std::runtime_error("TCP bootstrap session is closed");
    return fd;
  }

  static void read_case(int fd, std::string_view case_name, uint32_t ordinal,
                        detail::Deadline deadline) {
    auto peer_case = detail::read_frame(fd, detail::FrameType::Case, deadline);
    detail::Reader reader(peer_case);
    const uint32_t peer_ordinal = reader.u32();
    const auto peer_name = reader.string(reader.u16());
    reader.finish();
    if (peer_ordinal != ordinal || peer_name != case_name)
      throw std::runtime_error("TCP bootstrap test case or ordinal mismatch");
  }

  int hello(int fd, detail::Deadline deadline) const {
    std::vector<uint8_t> payload;
    detail::append_u16(payload, static_cast<uint16_t>(config_.run_id.size()));
    payload.insert(payload.end(), config_.run_id.begin(), config_.run_id.end());
    detail::append_u16(payload, static_cast<uint16_t>(config_.rank));
    detail::append_u16(payload, static_cast<uint16_t>(config_.world_size));
    detail::append_u32(payload, config_.seed);
    detail::append_u32(payload, static_cast<uint32_t>(config_.id_size));
    detail::write_frame(fd, detail::FrameType::Hello, payload, deadline);
    auto peer_hello = detail::read_frame(fd, detail::FrameType::Hello, deadline);
    detail::Reader reader(peer_hello);
    const auto peer_run_id = reader.string(reader.u16());
    const auto peer_rank = reader.u16();
    const auto peer_world_size = reader.u16();
    const auto peer_seed = reader.u32();
    const auto peer_id_size = reader.u32();
    reader.finish();
    const bool valid_peer =
        config_.rank == 0 ? peer_rank > 0 && peer_rank < config_.world_size : peer_rank == 0;
    if (peer_run_id != config_.run_id || !valid_peer || peer_world_size != config_.world_size ||
        peer_seed != config_.seed || peer_id_size != config_.id_size)
      throw std::runtime_error("TCP bootstrap HELLO run, rank, world, seed, or ID size mismatch");
    return peer_rank;
  }

  std::vector<detail::Fd> sockets_;
  Config config_;
};

} // namespace rocjitsu::test::rccl_tcp_bootstrap
