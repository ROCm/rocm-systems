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
#include <filesystem>
#include <fstream>
#include <string>

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

// The stub is built and installed beside the test binary, so it is always one step
// away from wherever this executable is running from -- build tree or install.
std::string stubPluginPath() {
  std::error_code ec;
  std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec) return "";
  return (exe.parent_path() / RCCL_TEST_PROFILER_STUB_NAME).string();
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

    const std::string stubPath = stubPluginPath();
    ASSERT_FALSE(stubPath.empty()) << "could not resolve the test binary's own directory";
    ASSERT_EQ(access(stubPath.c_str(), R_OK), 0)
        << "stub profiler plugin missing: " << stubPath;
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
