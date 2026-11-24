/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef RAS_CLIENT_HELPERS_HPP
#define RAS_CLIENT_HELPERS_HPP

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

#include <gtest/gtest.h>

namespace RcclUnitTesting {

// Picks a likely-free loopback TCP port: binds to port 0, reads the kernel assignment and closes the probe socket.
// Returns 0 on failure. The caller must bind with SO_REUSEADDR (the RAS listener does) so that the closed probe
// does not block it, and accept a small race window in which something else on the host takes the port.
inline int pickFreePort() {
  const int s = ::socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) return 0;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t len = sizeof(addr);
  int port = 0;
  if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0 &&
      ::getsockname(s, reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
    port = ntohs(addr.sin_port);
  }
  ::close(s);
  return port;
}

// Sends one command to the RAS client listener on 127.0.0.1:port, optionally after "SET FORMAT <format>", and
// returns everything the server writes until it closes the connection. Returns "" if the connection or the format
// switch fails. A receive error or a receive that waits longer than recvTimeout is a test failure.
inline std::string rasRequest(int port, const std::string& command, const std::string& format = "",
                              std::chrono::seconds recvTimeout = std::chrono::seconds(30)) {
  int sock = -1;
  for (int attempt = 0; attempt < 50 && sock < 0; ++attempt) {
    const int s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return {};
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      sock = s;
    } else {
      ::close(s);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  if (sock < 0) return {};
  timeval tv{static_cast<time_t>(recvTimeout.count()), 0};
  ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  char buf[4096];
  if (!format.empty()) {
    const std::string setFormat = "SET FORMAT " + format + "\n";
    ::send(sock, setFormat.c_str(), setFormat.size(), 0);
    const ssize_t n = ::recv(sock, buf, sizeof(buf), 0);
    if (n <= 0 || std::string(buf, n).find("OK") == std::string::npos) {
      ::close(sock);
      return {};
    }
  }
  const std::string line = command + "\n";
  ::send(sock, line.c_str(), line.size(), 0);
  std::string body;
  ssize_t n;
  while ((n = ::recv(sock, buf, sizeof(buf), 0)) > 0) body.append(buf, n);
  if (n < 0) {
    ADD_FAILURE() << "RAS client request \"" << command << "\": recv failed before the server closed the "
                  << "connection (" << strerror(errno) << ", timeout " << recvTimeout.count() << " s)";
  }
  ::close(sock);
  return body;
}

}  // namespace RcclUnitTesting

#endif  // RAS_CLIENT_HELPERS_HPP
