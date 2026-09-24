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

// Naming for temporary artifacts written by tests. Tests write under the temp
// directory because an installed test tree may be read-only, and that directory
// is shared, so the name has to be both unique and unguessable: a predictable
// path can be pre-seeded with a symlink that the subsequent open follows.
namespace hip_test {

// 16 hex characters: 64 bits from std::random_device, mixed with the pid.
//
// The pid is what keeps the name unique if std::random_device degenerates.
// The standard allows it to return a fixed sequence when the implementation
// cannot do better, and at least one toolchain does exactly that. Live pids
// within a namespace are distinct by definition, so concurrent processes still
// get distinct names in that case, which a clock reading would not guarantee.
inline std::string RandomTag() {
  std::random_device dev;
  uint64_t value = (static_cast<uint64_t>(dev()) << 32) ^ dev();
  value ^= static_cast<uint64_t>(HIP_TEST_GETPID());
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(value));
  return std::string(buf);
}

// Path under the temp directory as <stem>-<tag><extension>. The caller supplies
// the stem so each test keeps its own recognisable prefix.
inline std::string TempPath(const std::string& stem, const std::string& extension) {
  return (fs::temp_directory_path() / (stem + "-" + RandomTag() + extension)).string();
}

}  // namespace hip_test
