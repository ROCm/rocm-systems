/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ProfilerPluginClose: a profiler plugin that exposes no usable interface must be
// unloaded when ncclProfilerPluginInit() rejects it (src/plugin/profiler.cc, fail:
// path), not left mapped for the lifetime of the process.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {
namespace {

// MAX_STR_LEN in src/plugin/plugin_open.cc: NCCL_PROFILER_PLUGIN is copied into a
// buffer of this size, so a longer path is silently truncated and never loads.
constexpr size_t kPluginPathLimit = 255;

class ScopedTempFile {
 public:
  explicit ScopedTempFile(const char* pathTemplate) : path_(pathTemplate) {
    int fd = mkstemp(path_.data());
    if (fd >= 0) {
      valid_ = true;
      close(fd);
    }
  }

  ~ScopedTempFile() {
    if (valid_) unlink(path_.c_str());
  }

  ScopedTempFile(const ScopedTempFile&) = delete;
  ScopedTempFile& operator=(const ScopedTempFile&) = delete;

  bool valid() const { return valid_; }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  bool valid_ = false;
};

bool fileHasContent(const std::string& path) {
  std::ifstream f(path);
  return f.peek() != std::ifstream::traits_type::eof();
}

std::string executableDir() {
  char buf[4096] = {0};
  ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (len <= 0) return "";

  std::string path(buf, static_cast<size_t>(len));
  size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? "" : path.substr(0, slash);
}

// The stub is installed next to the test binary, which is the only location that
// holds for both a build tree and an installed package. The build-tree path baked
// in by CMake is kept as a fallback for anything that runs the binary in place.
std::vector<std::string> stubCandidates() {
  std::vector<std::string> candidates;
  if (std::string dir = executableDir(); !dir.empty())
    candidates.push_back(dir + "/" + RCCL_TEST_PROFILER_STUB_NAME);
  candidates.push_back(RCCL_TEST_PROFILER_STUB_BUILD_PATH);
  return candidates;
}

// Returns the reason this host cannot run the test, or "" when it can.
// GTEST_SKIP() must be issued by the caller: it expands to a bare return and
// would otherwise only leave this helper, letting the test body run on.
std::string gpuSkipReason() {
  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1)
    return "requires at least one GPU";
  return "";
}

} // namespace

// Isolated because the outcome of the plugin probe is latched in a process global:
// once a plugin has been rejected, later communicators skip the load entirely.
TEST(ProfilerPluginClose, UnusablePluginIsUnloaded) {
  RUN_ISOLATED_TEST("ProfilerPluginClose.UnusablePluginIsUnloaded", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    std::string stubPath;
    std::string tried;
    for (const std::string& candidate : stubCandidates()) {
      tried += "\n  " + candidate;
      if (access(candidate.c_str(), R_OK) == 0) {
        stubPath = candidate;
        break;
      }
    }
    ASSERT_FALSE(stubPath.empty()) << "stub profiler plugin not found, tried:" << tried;
    if (stubPath.size() >= kPluginPathLimit)
      GTEST_SKIP() << "stub path exceeds the " << kPluginPathLimit
                   << " byte plugin path limit and would never be loaded: " << stubPath;

    ScopedTempFile loadMarker("/tmp/rccl_profiler_stub_load_XXXXXX");
    ASSERT_TRUE(loadMarker.valid()) << "could not create the plugin load marker";
    ASSERT_EQ(setenv("RCCL_TEST_PROFILER_STUB_LOAD_FILE", loadMarker.path().c_str(), 1), 0);
    ASSERT_EQ(setenv("NCCL_PROFILER_PLUGIN", stubPath.c_str(), 1), 0);

    // A plugin RCCL cannot use must not stop a communicator from coming up.
    ncclUniqueId id;
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
    ncclComm_t comm = nullptr;
    ASSERT_EQ(ncclCommInitRank(&comm, 1, id, 0), ncclSuccess);
    ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);

    // Without this the check below would also pass for a plugin that was never
    // opened, leaving the test green while exercising nothing.
    ASSERT_TRUE(fileHasContent(loadMarker.path()))
        << "profiler plugin " << stubPath << " was never loaded";

    // RTLD_NOLOAD loads nothing and returns non-NULL only while the object is
    // mapped; it still takes a reference, so drop it before asserting.
    void* handle = dlopen(stubPath.c_str(), RTLD_NOLOAD | RTLD_LAZY);
    if (handle != nullptr) dlclose(handle);
    EXPECT_EQ(handle, nullptr)
        << "profiler plugin " << stubPath << " is still mapped after comm create/destroy";
  });
}

} // namespace RcclUnitTesting
