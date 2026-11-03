/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once

#include <gtest/gtest.h>

#include <climits>
#include <cstdlib>
#include <string>
#include <unistd.h>

namespace RcclUnitTesting {

// Topology XML fixtures from tools/topo_expl/models. The build and the install both stage them in
// this directory beside the test binary, so the tests need no source tree at run time.
// RCCL_TEST_TOPO_DIR overrides the directory.
constexpr const char* kTopoFixtureDirName = "rccl-test-topo-models";

inline std::string topoFixtureDir() {
  const char* overrideDir = std::getenv("RCCL_TEST_TOPO_DIR");
  if (overrideDir != nullptr && overrideDir[0] != '\0') return overrideDir;

  char exe[PATH_MAX];
  const ssize_t len = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  if (len <= 0) return kTopoFixtureDirName;
  const std::string exePath(exe, static_cast<size_t>(len));
  return exePath.substr(0, exePath.find_last_of('/') + 1) + kTopoFixtureDirName;
}

inline std::string topoFixturePath(const char* name) { return topoFixtureDir() + "/" + name; }

// A missing fixture is a broken test setup, never a reason to skip.
inline ::testing::AssertionResult topoFixtureReadable(const std::string& path) {
  if (access(path.c_str(), R_OK) == 0) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << "topology fixture not readable: " << path << " (the build stages "
         << kTopoFixtureDirName << "/ beside the test binary; set RCCL_TEST_TOPO_DIR to override)";
}

}  // namespace RcclUnitTesting
