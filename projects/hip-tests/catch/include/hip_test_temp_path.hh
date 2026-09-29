/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hip_test_filesystem.hh"

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

#ifdef _WIN32
#include <process.h>
#define HIP_TEST_GETPID _getpid
#else
#include <unistd.h>
#define HIP_TEST_GETPID getpid
#endif

// Naming for temporary test artifacts. The temp directory is shared, so names
// must be unguessable: a predictable path can be pre-seeded with a symlink.
namespace hip_test {

// 64 bits from std::random_device as 16 hex chars. The pid keeps the name
// unique if random_device degenerates to a fixed sequence, which the standard
// permits; live pids in a namespace are distinct, clock readings are not.
inline std::string RandomTag() {
  std::random_device dev;
  uint64_t value = (static_cast<uint64_t>(dev()) << 32) ^ dev();
  value ^= static_cast<uint64_t>(HIP_TEST_GETPID());
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(value));
  return std::string(buf);
}

// <temp dir>/<stem>-<tag><extension>.
inline std::string TempPath(const std::string& stem, const std::string& extension) {
  return (fs::temp_directory_path() / (stem + "-" + RandomTag() + extension)).string();
}

}  // namespace hip_test
