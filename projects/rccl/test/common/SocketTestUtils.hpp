/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once

#include <netinet/in.h>

#include "socket.h"

namespace RcclUnitTesting {
namespace SocketTestUtils {

inline constexpr double kCpuBusyThreshold = 0.5;

inline union ncclSocketAddress LoopbackAddr() {
  union ncclSocketAddress addr = {};
  addr.sin.sin_family = AF_INET;
  addr.sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin.sin_port = 0;
  return addr;
}

inline double timespecDiffMs(const struct timespec& start, const struct timespec& end) {
  return (end.tv_sec - start.tv_sec) * 1e3 +
         (end.tv_nsec - start.tv_nsec) / 1e6;
}

} // namespace SocketTestUtils
} // namespace RcclUnitTesting
