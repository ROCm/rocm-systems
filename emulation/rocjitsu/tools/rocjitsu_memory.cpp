// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
  rocjitsu::memory_backing::run_keeper(3, 4, 5, owner);
}
