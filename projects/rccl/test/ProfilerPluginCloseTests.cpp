/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ProfilerPluginClose: a profiler plugin that exposes no usable interface must be
// unloaded when ncclProfilerPluginInit() rejects it (src/plugin/profiler.cc, fail:
// path), not left mapped for the lifetime of the process.
//
// DebugAttn: RCCL logs the stub's rejection at ATTN, because NCCL_PROFILER_PLUGIN asked for it by name.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {

// Re-reads NCCL_DEBUG* on the next log line. Exported from src/debug.cc.
extern "C" void ncclResetDebugInitInternal();

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

// In the build tree the stub sits beside the test binary; in an install it is in
// the library directory. Check both so the test runs from either layout.
std::string stubPluginPath() {
  std::error_code ec;
  std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec) return "";
  const std::filesystem::path dir = exe.parent_path();
  const std::filesystem::path candidates[] = {
    dir / RCCL_TEST_PROFILER_STUB_NAME,                                           // build tree
    dir.parent_path() / RCCL_TEST_INSTALL_LIBDIR / RCCL_TEST_PROFILER_STUB_NAME,  // install tree
  };
  for (const std::filesystem::path& candidate : candidates) {
    std::error_code exists_ec;
    if (std::filesystem::exists(candidate, exists_ec)) return candidate.string();
  }
  return "";
}

// Call before RUN_ISOLATED_TEST: a GTEST_SKIP inside the isolated child is scored as a pass.
std::string stubSkipReason() {
  const std::string stubPath = stubPluginPath();
  if (stubPath.size() >= kPluginPathLimit) {
    return "stub path exceeds the " + std::to_string(kPluginPathLimit) + " byte plugin path limit: " + stubPath;
  }
  return "";
}

// Brings a 1-rank comm up and down with the stub requested and returns in @p log what RCCL wrote.
void logStubRejection(const char* debug, const char* debugLevels, std::string* log) {
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  const std::string stubPath = stubPluginPath();
  ASSERT_FALSE(stubPath.empty()) << "could not resolve the test binary's own directory";
  ASSERT_EQ(access(stubPath.c_str(), R_OK), 0) << "stub profiler plugin missing: " << stubPath;
  ScopedTempFile logFile("/tmp/rccl_debug_attn_XXXXXX");
  ASSERT_TRUE(logFile.valid()) << "could not create the debug log file";

  // An inherited setting could add levels or filter out the INIT subsystem.
  for (const char* var : {"NCCL_DEBUG_LEVELS", "NCCL_DEBUG_SUBSYS", "NCCL_WARN_ENABLE_DEBUG_INFO"}) {
    ASSERT_EQ(unsetenv(var), 0);
  }
  if (debugLevels) {
    ASSERT_EQ(setenv("NCCL_DEBUG_LEVELS", debugLevels, 1), 0);
  }
  ASSERT_EQ(setenv("NCCL_DEBUG", debug, 1), 0);
  ASSERT_EQ(setenv("NCCL_DEBUG_FILE", logFile.path().c_str(), 1), 0);
  ASSERT_EQ(setenv("NCCL_PROFILER_PLUGIN", stubPath.c_str(), 1), 0);
  ncclResetDebugInitInternal();

  ncclUniqueId id;
  ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
  ncclComm_t comm = nullptr;
  ASSERT_EQ(ncclCommInitRank(&comm, 1, id, 0), ncclSuccess);
  ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);

  // RCCL line-buffers NCCL_DEBUG_FILE, so every line it logged is on disk.
  std::ifstream f(logFile.path());
  log->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// True if @p log has the line RCCL writes when it rejects the stub, and that line
// contains @p tag.
bool loggedRejection(const std::string& log, const std::string& tag) {
  std::istringstream lines(log);
  for (std::string line; std::getline(lines, line);) {
    if (line.find("External profiler plugin") != std::string::npos &&
        line.find("is unsupported") != std::string::npos && line.find(tag) != std::string::npos) {
      return true;
    }
  }
  return false;
}

} // namespace

// Isolated because the outcome of the plugin probe is latched in a process global:
// once a plugin has been rejected, later communicators skip the load entirely.
TEST(ProfilerPluginClose, UnusablePluginIsUnloaded) {
  if (auto reason = stubSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
  RUN_ISOLATED_TEST("ProfilerPluginClose.UnusablePluginIsUnloaded", []() {
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    const std::string stubPath = stubPluginPath();
    ASSERT_FALSE(stubPath.empty())
        << "stub profiler plugin not found beside the test binary or in ../"
           RCCL_TEST_INSTALL_LIBDIR " "
           "(or /proc/self/exe could not be resolved)";
    ASSERT_EQ(access(stubPath.c_str(), R_OK), 0)
        << "stub profiler plugin missing: " << stubPath;

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

// NCCL_DEBUG=ATTN shows ATTN messages on top of WARN, but not INFO. Releases before
// NCCL 2.32 do not recognize the value and log nothing.
TEST(DebugAttn, RejectedProfilerPluginLoggedAtAttn) {
  if (auto reason = stubSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
  RUN_ISOLATED_TEST("DebugAttn.RejectedProfilerPluginLoggedAtAttn", []() {
    std::string log;
    logStubRejection("ATTN", nullptr, &log);
    if (::testing::Test::HasFatalFailure()) return;
    EXPECT_TRUE(loggedRejection(log, "NCCL ATTN")) << "NCCL_DEBUG=ATTN log:\n" << log;
    EXPECT_EQ(log.find("NCCL INFO"), std::string::npos) << "NCCL_DEBUG=ATTN enabled INFO:\n" << log;
  });
}

// NCCL_DEBUG_LEVELS adds single levels to NCCL_DEBUG, here ATTN to a WARN baseline.
TEST(DebugAttn, DebugLevelsAddsAttnToWarn) {
  if (auto reason = stubSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
  RUN_ISOLATED_TEST("DebugAttn.DebugLevelsAddsAttnToWarn", []() {
    std::string log;
    logStubRejection("WARN", "ATTN", &log);
    if (::testing::Test::HasFatalFailure()) return;
    EXPECT_TRUE(loggedRejection(log, "NCCL ATTN")) << "NCCL_DEBUG=WARN NCCL_DEBUG_LEVELS=ATTN log:\n" << log;
  });
}

// Control: INFO always shows the rejection, so a failure above is about the level.
TEST(DebugAttn, RejectedProfilerPluginLoggedAtInfo) {
  if (auto reason = stubSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
  RUN_ISOLATED_TEST("DebugAttn.RejectedProfilerPluginLoggedAtInfo", []() {
    std::string log;
    logStubRejection("INFO", nullptr, &log);
    if (::testing::Test::HasFatalFailure()) return;
    EXPECT_TRUE(loggedRejection(log, "")) << "NCCL_DEBUG=INFO log:\n" << log;
    // Pins the prefix that RejectedProfilerPluginLoggedAtAttn expects to be absent.
    EXPECT_NE(log.find("NCCL INFO"), std::string::npos) << "NCCL_DEBUG=INFO logged no INFO line:\n" << log;
  });
}

} // namespace RcclUnitTesting
