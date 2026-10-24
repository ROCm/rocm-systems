// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "rocjitsu/kmd/linux/memory_backing.h"

#include <charconv>
#include <cstring>

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  pid_t owner = 0;
  const char *end = argv[1] + std::strlen(argv[1]);
  auto parsed = std::from_chars(argv[1], end, owner);
  if (parsed.ec != std::errc{} || parsed.ptr != end || owner <= 0)
    return 1;
  using namespace rocjitsu;
  libc_passthrough().resolve();
  run_memory_keeper(kMemoryKeeperListenerFd, kMemoryKeeperLeaseFd, kMemoryKeeperStartupFd, owner);
}
