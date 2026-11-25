// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rccl_tcp_bootstrap.h"

#include <gtest/gtest.h>

#include <sys/socket.h>

#include <array>
#include <chrono>
#include <functional>
#include <future>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace tcp_bootstrap = rocjitsu::test::rccl_tcp_bootstrap;

using namespace std::chrono_literals;
using tcp_bootstrap::Config;
using tcp_bootstrap::Role;
using tcp_bootstrap::Session;
using tcp_bootstrap::detail::Clock;
using tcp_bootstrap::detail::Fd;
constexpr char kRunId[] = "0123456789abcdef0123456789abcdef";

std::array<int, 2> socket_pair() {
  std::array<int, 2> fds{};
  if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds.data()) < 0) {
    ADD_FAILURE() << "cannot create bootstrap test socket pair: " << std::strerror(errno);
    std::abort();
  }
  return fds;
}

Config config(Role role, std::string run_id = kRunId, uint32_t seed = 42,
              std::chrono::milliseconds timeout = 2s) {
  return Config{role, "", std::move(run_id), role == Role::Listen ? 0 : 1, 2, seed, 128, timeout};
}

template <typename T> T require_value(util::FailureOr<T> result) {
  if (result.failed()) {
    ADD_FAILURE() << "unexpected bootstrap setup failure";
    std::abort();
  }
  return std::move(result.value());
}

template <typename Action> std::string failure_message(Action action) {
  util::StringDiagnostic diagnostic;
  const auto result = action(diagnostic.emitter());
  EXPECT_TRUE(result.failed());
  return diagnostic.message();
}

std::vector<uint8_t> hello_payload(uint16_t rank, uint16_t world_size, uint32_t seed,
                                   uint32_t id_size) {
  std::vector<uint8_t> payload;
  tcp_bootstrap::detail::append_u16(payload, sizeof(kRunId) - 1);
  payload.insert(payload.end(), std::begin(kRunId), std::end(kRunId) - 1);
  tcp_bootstrap::detail::append_u16(payload, rank);
  tcp_bootstrap::detail::append_u16(payload, world_size);
  tcp_bootstrap::detail::append_u32(payload, seed);
  tcp_bootstrap::detail::append_u32(payload, id_size);
  return payload;
}

std::string failure_from_raw_hello(const std::function<void(int)> &respond) {
  const auto fds = socket_pair();
  auto peer = std::async(std::launch::async, [fd = fds[1], respond] {
    Fd socket(fd);
    ASSERT_TRUE(tcp_bootstrap::detail::set_nonblocking(socket.get()).succeeded());
    ASSERT_TRUE(tcp_bootstrap::detail::read_frame(
                    socket.get(), tcp_bootstrap::detail::FrameType::Hello, Clock::now() + 2s)
                    .succeeded());
    respond(socket.get());
  });
  const auto error = failure_message([&](util::DiagnosticEmitter diagnostic) {
    return Session::FromConnectedFd(fds[0], config(Role::Listen), diagnostic);
  });
  peer.get();
  return error;
}

TEST(RcclTcpBootstrapTest, ExchangesExactIdsAcrossTwoCases) {
  const auto fds = socket_pair();
  auto peer = std::async(std::launch::async, [fd = fds[1]] {
    auto session = require_value(Session::FromConnectedFd(fd, config(Role::Connect)));
    for (uint32_t ordinal = 0; ordinal < 2; ++ordinal) {
      std::array<uint8_t, 128> id{};
      const std::string name = ordinal == 0 ? "RcclTest.AllReduce" : "RcclTest.Broadcast";
      ASSERT_TRUE(session.ExchangeCase(name, ordinal, id.data()).succeeded());
      for (size_t i = 0; i < id.size(); ++i) {
        if (id[i] != static_cast<uint8_t>(i + ordinal)) {
          FAIL() << "rank 1 received an incorrect NCCL ID byte";
        }
      }
    }
  });

  auto session = require_value(Session::FromConnectedFd(fds[0], config(Role::Listen)));
  for (uint32_t ordinal = 0; ordinal < 2; ++ordinal) {
    std::array<uint8_t, 128> id{};
    for (size_t i = 0; i < id.size(); ++i) {
      id[i] = static_cast<uint8_t>(i + ordinal);
    }
    const std::string name = ordinal == 0 ? "RcclTest.AllReduce" : "RcclTest.Broadcast";
    ASSERT_TRUE(session.ExchangeCase(name, ordinal, id.data()).succeeded());
  }
  peer.get();
}

TEST(RcclTcpBootstrapTest, ConnectsWhenListenerStartsLater) {
  Fd reservation(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
  ASSERT_GE(reservation.get(), 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  ASSERT_EQ(
      ::bind(reservation.get(), reinterpret_cast<const sockaddr *>(&address), sizeof(address)), 0);
  socklen_t address_size = sizeof(address);
  ASSERT_EQ(::getsockname(reservation.get(), reinterpret_cast<sockaddr *>(&address), &address_size),
            0);
  const std::string endpoint = "127.0.0.1:" + std::to_string(ntohs(address.sin_port));
  reservation.reset();

  auto connector = std::async(std::launch::async, [&] {
    auto peer_config = config(Role::Connect);
    peer_config.endpoint = endpoint;
    auto session = require_value(Session::Open(peer_config));
    std::array<uint8_t, 128> id{};
    EXPECT_TRUE(session.ExchangeCase("RcclTest.AllReduce", 0, id.data()).succeeded());
    return id;
  });
  std::this_thread::sleep_for(60ms);
  auto listener_config = config(Role::Listen);
  listener_config.endpoint = endpoint;
  auto session = require_value(Session::Open(listener_config));
  std::array<uint8_t, 128> id{};
  id.fill(0xa5);
  EXPECT_TRUE(session.ExchangeCase("RcclTest.AllReduce", 0, id.data()).succeeded());
  EXPECT_EQ(connector.get(), id);
}

TEST(RcclTcpBootstrapTest, ExchangesExactIdsAcrossEightRanksAndCases) {
  Fd reservation(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
  ASSERT_GE(reservation.get(), 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  ASSERT_EQ(
      ::bind(reservation.get(), reinterpret_cast<const sockaddr *>(&address), sizeof(address)), 0);
  socklen_t address_size = sizeof(address);
  ASSERT_EQ(::getsockname(reservation.get(), reinterpret_cast<sockaddr *>(&address), &address_size),
            0);
  const std::string endpoint = "127.0.0.1:" + std::to_string(ntohs(address.sin_port));
  reservation.reset();

  std::vector<std::future<void>> connectors;
  for (int rank = 7; rank >= 1; --rank) {
    connectors.push_back(std::async(std::launch::async, [endpoint, rank] {
      auto session = require_value(
          Session::Open(Config{Role::Connect, endpoint, kRunId, rank, 8, 42, 128, 5s}));
      for (uint32_t ordinal = 0; ordinal < 2; ++ordinal) {
        std::array<uint8_t, 128> id{};
        const std::string name = ordinal == 0 ? "RcclTest.AllReduce" : "RcclTest.Broadcast";
        ASSERT_TRUE(session.ExchangeCase(name, ordinal, id.data()).succeeded());
        for (size_t i = 0; i < id.size(); ++i) {
          if (id[i] != static_cast<uint8_t>(i + ordinal)) {
            FAIL() << "connector received an incorrect NCCL ID byte";
          }
        }
      }
    }));
  }
  auto listener =
      require_value(Session::Open(Config{Role::Listen, endpoint, kRunId, 0, 8, 42, 128, 5s}));
  for (uint32_t ordinal = 0; ordinal < 2; ++ordinal) {
    std::array<uint8_t, 128> id{};
    for (size_t i = 0; i < id.size(); ++i) {
      id[i] = static_cast<uint8_t>(i + ordinal);
    }
    const std::string name = ordinal == 0 ? "RcclTest.AllReduce" : "RcclTest.Broadcast";
    ASSERT_TRUE(listener.ExchangeCase(name, ordinal, id.data()).succeeded());
  }
  for (auto &connector : connectors) {
    connector.get();
  }
}

TEST(RcclTcpBootstrapTest, HandlesFragmentedReadsAndPartialWrites) {
  const auto fds = socket_pair();
  Fd receiver(fds[0]);
  Fd sender(fds[1]);
  ASSERT_TRUE(tcp_bootstrap::detail::set_nonblocking(receiver.get()).succeeded());
  ASSERT_TRUE(tcp_bootstrap::detail::set_nonblocking(sender.get()).succeeded());

  std::array<uint8_t, 128> input{};
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<uint8_t>(i);
  }
  auto fragmented_writer = std::async(std::launch::async, [&] {
    for (uint8_t byte : input) {
      ASSERT_TRUE(tcp_bootstrap::detail::write_exact(sender.get(), &byte, 1, Clock::now() + 2s)
                      .succeeded());
      std::this_thread::sleep_for(50us);
    }
  });
  std::array<uint8_t, 128> received{};
  ASSERT_TRUE(tcp_bootstrap::detail::read_exact(receiver.get(), received.data(), received.size(),
                                                Clock::now() + 2s)
                  .succeeded());
  fragmented_writer.get();
  EXPECT_EQ(received, input);

  const int small_buffer = 1024;
  ASSERT_EQ(::setsockopt(sender.get(), SOL_SOCKET, SO_SNDBUF, &small_buffer, sizeof(small_buffer)),
            0);
  std::vector<uint8_t> large_input(256 * 1024);
  for (size_t i = 0; i < large_input.size(); ++i) {
    large_input[i] = static_cast<uint8_t>(i);
  }
  auto bulk_writer = std::async(std::launch::async, [&] {
    ASSERT_TRUE(tcp_bootstrap::detail::write_exact(sender.get(), large_input.data(),
                                                   large_input.size(), Clock::now() + 5s)
                    .succeeded());
  });
  std::vector<uint8_t> large_received(large_input.size());
  ASSERT_TRUE(tcp_bootstrap::detail::read_exact(receiver.get(), large_received.data(),
                                                large_received.size(), Clock::now() + 5s)
                  .succeeded());
  bulk_writer.get();
  EXPECT_EQ(large_received, large_input);
}

TEST(RcclTcpBootstrapTest, RejectsRunAndCaseMismatch) {
  {
    const auto fds = socket_pair();
    auto peer = std::async(std::launch::async, [fd = fds[1]] {
      return failure_message([&](util::DiagnosticEmitter diagnostic) {
        return Session::FromConnectedFd(
            fd, config(Role::Connect, "fedcba9876543210fedcba9876543210"), diagnostic);
      });
    });
    const auto error = failure_message([&](util::DiagnosticEmitter diagnostic) {
      return Session::FromConnectedFd(fds[0], config(Role::Listen), diagnostic);
    });
    EXPECT_NE(error.find("HELLO"), std::string::npos);
    EXPECT_NE(peer.get().find("HELLO"), std::string::npos);
  }

  {
    const auto fds = socket_pair();
    auto peer = std::async(std::launch::async, [fd = fds[1]] {
      auto session = require_value(Session::FromConnectedFd(fd, config(Role::Connect)));
      std::array<uint8_t, 128> id{};
      return failure_message([&](util::DiagnosticEmitter diagnostic) {
        return session.ExchangeCase("RcclTest.Broadcast", 0, id.data(), diagnostic);
      });
    });
    auto session = require_value(Session::FromConnectedFd(fds[0], config(Role::Listen)));
    std::array<uint8_t, 128> id{};
    const auto error = failure_message([&](util::DiagnosticEmitter diagnostic) {
      return session.ExchangeCase("RcclTest.AllReduce", 0, id.data(), diagnostic);
    });
    EXPECT_NE(error.find("case or ordinal mismatch"), std::string::npos);
    EXPECT_NE(peer.get().find("case or ordinal mismatch"), std::string::npos);
  }

  {
    const auto fds = socket_pair();
    auto peer = std::async(std::launch::async, [fd = fds[1]] {
      auto session = require_value(Session::FromConnectedFd(fd, config(Role::Connect)));
      std::array<uint8_t, 128> id{};
      return failure_message([&](util::DiagnosticEmitter diagnostic) {
        return session.ExchangeCase("RcclTest.AllReduce", 1, id.data(), diagnostic);
      });
    });
    auto session = require_value(Session::FromConnectedFd(fds[0], config(Role::Listen)));
    std::array<uint8_t, 128> id{};
    const auto error = failure_message([&](util::DiagnosticEmitter diagnostic) {
      return session.ExchangeCase("RcclTest.AllReduce", 0, id.data(), diagnostic);
    });
    EXPECT_NE(error.find("case or ordinal mismatch"), std::string::npos);
    EXPECT_NE(peer.get().find("case or ordinal mismatch"), std::string::npos);
  }
}

TEST(RcclTcpBootstrapTest, RejectsHelloSeedRankAndIdSizeMismatch) {
  const auto check_peer_config = [](Config peer_config) {
    const auto fds = socket_pair();
    auto peer = std::async(std::launch::async, [fd = fds[1], peer_config] {
      return failure_message([&](util::DiagnosticEmitter diagnostic) {
        return Session::FromConnectedFd(fd, peer_config, diagnostic);
      });
    });
    const auto error = failure_message([&](util::DiagnosticEmitter diagnostic) {
      return Session::FromConnectedFd(fds[0], config(Role::Listen), diagnostic);
    });
    EXPECT_NE(error.find("HELLO"), std::string::npos);
    EXPECT_NE(peer.get().find("HELLO"), std::string::npos);
  };

  check_peer_config(config(Role::Connect, kRunId, 43));
  auto wrong_id_size = config(Role::Connect);
  wrong_id_size.id_size = 64;
  check_peer_config(wrong_id_size);

  const auto wrong_rank = failure_from_raw_hello([](int fd) {
    const auto payload = hello_payload(0, 2, 42, 128);
    ASSERT_TRUE(tcp_bootstrap::detail::write_frame(fd, tcp_bootstrap::detail::FrameType::Hello,
                                                   payload, Clock::now() + 2s)
                    .succeeded());
  });
  EXPECT_NE(wrong_rank.find("HELLO"), std::string::npos);
}

TEST(RcclTcpBootstrapTest, RejectsTruncatedAndMalformedFrames) {
  const auto truncated = failure_from_raw_hello([](int fd) {
    std::vector<uint8_t> header;
    tcp_bootstrap::detail::append_u32(header, tcp_bootstrap::detail::kMagic);
    tcp_bootstrap::detail::append_u16(header, tcp_bootstrap::detail::kVersion);
    tcp_bootstrap::detail::append_u16(
        header, static_cast<uint16_t>(tcp_bootstrap::detail::FrameType::Hello));
    tcp_bootstrap::detail::append_u32(header, 10);
    ASSERT_TRUE(
        tcp_bootstrap::detail::write_exact(fd, header.data(), header.size(), Clock::now() + 2s)
            .succeeded());
    const std::array<uint8_t, 3> partial_payload{0, 1, 2};
    ASSERT_TRUE(tcp_bootstrap::detail::write_exact(fd, partial_payload.data(),
                                                   partial_payload.size(), Clock::now() + 2s)
                    .succeeded());
  });
  EXPECT_NE(truncated.find("disconnected"), std::string::npos);

  const auto malformed = failure_from_raw_hello([](int fd) {
    const std::vector<uint8_t> payload{0};
    ASSERT_TRUE(tcp_bootstrap::detail::write_frame(fd, tcp_bootstrap::detail::FrameType::Hello,
                                                   payload, Clock::now() + 2s)
                    .succeeded());
  });
  EXPECT_NE(malformed.find("truncated"), std::string::npos);

  const auto oversized = failure_from_raw_hello([](int fd) {
    std::vector<uint8_t> header;
    tcp_bootstrap::detail::append_u32(header, tcp_bootstrap::detail::kMagic);
    tcp_bootstrap::detail::append_u16(header, tcp_bootstrap::detail::kVersion);
    tcp_bootstrap::detail::append_u16(
        header, static_cast<uint16_t>(tcp_bootstrap::detail::FrameType::Hello));
    tcp_bootstrap::detail::append_u32(header, tcp_bootstrap::detail::kMaxPayload + 1);
    ASSERT_TRUE(
        tcp_bootstrap::detail::write_exact(fd, header.data(), header.size(), Clock::now() + 2s)
            .succeeded());
  });
  EXPECT_NE(oversized.find("size limit"), std::string::npos);
}

TEST(RcclTcpBootstrapTest, BoundsDisconnectAndTimeout) {
  {
    const auto fds = socket_pair();
    auto peer = std::async(std::launch::async, [fd = fds[1]] {
      (void)require_value(Session::FromConnectedFd(fd, config(Role::Connect)));
    });
    auto session = require_value(Session::FromConnectedFd(fds[0], config(Role::Listen)));
    peer.get(); // The peer closes the socket before the first case.
    std::array<uint8_t, 128> id{};
    EXPECT_FALSE(failure_message([&](util::DiagnosticEmitter diagnostic) {
                   return session.ExchangeCase("RcclTest.AllReduce", 0, id.data(), diagnostic);
                 }).empty());
  }

  {
    const auto fds = socket_pair();
    Fd silent_peer(fds[1]);
    const auto start = Clock::now();
    const auto error = failure_message([&](util::DiagnosticEmitter diagnostic) {
      return Session::FromConnectedFd(
          fds[0], config(Role::Listen, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 42, 40ms), diagnostic);
    });
    EXPECT_NE(error.find("timeout"), std::string::npos);
    EXPECT_LT(Clock::now() - start, 1s);
  }
}

TEST(RcclTcpBootstrapTest, RejectsInvalidEndpointsAndConfigurations) {
  EXPECT_TRUE(tcp_bootstrap::detail::parse_endpoint("localhost:1234").failed());
  EXPECT_TRUE(tcp_bootstrap::detail::parse_endpoint("127.0.0.1:0").failed());
  EXPECT_TRUE(tcp_bootstrap::detail::parse_endpoint("127.0.0.1:65536").failed());
  EXPECT_TRUE(tcp_bootstrap::detail::parse_endpoint("127.0.0.1:1234").succeeded());
  auto invalid = config(Role::Listen);
  invalid.rank = 1;
  EXPECT_TRUE(tcp_bootstrap::detail::validate_config(invalid).failed());
  invalid = config(Role::Connect, "0123456789abcdef0123456789abcdeg");
  EXPECT_TRUE(tcp_bootstrap::detail::validate_config(invalid).failed());
  invalid = config(Role::Connect, "0123456789abcdef0123456789abcde");
  EXPECT_TRUE(tcp_bootstrap::detail::validate_config(invalid).failed());
  invalid = config(Role::Connect, "0123456789abcdef0123456789abcdef0");
  EXPECT_TRUE(tcp_bootstrap::detail::validate_config(invalid).failed());
  invalid = config(Role::Connect, "ABCDEF0123456789ABCDEF0123456789");
  EXPECT_TRUE(tcp_bootstrap::detail::validate_config(invalid).succeeded());
}

} // namespace
