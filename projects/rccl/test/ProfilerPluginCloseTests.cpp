/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ProfilerPluginClose: a profiler plugin that exposes no usable interface must be
// unloaded when ncclProfilerPluginInit() rejects it (src/plugin/profiler.cc, fail:
// path), not left mapped for the lifetime of the process.
//
// DebugAttn: rejecting a plugin the user named in NCCL_PROFILER_PLUGIN is a fallback
// worth seeing without INFO, so NCCL 2.32 logs it at the ATTN level. The same stub
// is the cheapest way to make RCCL emit an ATTN line.

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

// gpuSkipReason(), plus a stub path RCCL would truncate and so never load.
std::string attnSkipReason() {
  if (auto reason = gpuSkipReason(); !reason.empty()) return reason;
  if (stubPluginPath().size() >= kPluginPathLimit)
    return "stub path exceeds the " + std::to_string(kPluginPathLimit) + " byte plugin path limit";
  return "";
}

// Brings a single-rank communicator up and down with the stub requested, so RCCL
// rejects it, and returns in @p log what RCCL wrote at the given NCCL_DEBUG and
// NCCL_DEBUG_LEVELS (nullptr leaves NCCL_DEBUG_LEVELS unset).
void logStubRejection(const char* debug, const char* debugLevels, std::string* log) {
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  const std::string stubPath = stubPluginPath();
  ASSERT_EQ(access(stubPath.c_str(), R_OK), 0) << "stub profiler plugin missing: " << stubPath;
  ScopedTempFile logFile("/tmp/rccl_debug_attn_XXXXXX");
  ASSERT_TRUE(logFile.valid()) << "could not create the debug log file";

  // An inherited setting could add levels or filter out the INIT subsystem.
  for (const char* var : {"NCCL_DEBUG_LEVELS", "NCCL_DEBUG_SUBSYS", "NCCL_WARN_ENABLE_DEBUG_INFO"})
    ASSERT_EQ(unsetenv(var), 0);
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
        line.find("is unsupported") != std::string::npos && line.find(tag) != std::string::npos)
      return true;
  }
  return false;
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

// NCCL_DEBUG=ATTN shows ATTN messages on top of WARN, but not INFO. Releases before
// NCCL 2.32 do not recognize the value and log nothing.
TEST(DebugAttn, RejectedProfilerPluginLoggedAtAttn) {
  RUN_ISOLATED_TEST("DebugAttn.RejectedProfilerPluginLoggedAtAttn", []() {
    if (auto reason = attnSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    std::string log;
    logStubRejection("ATTN", nullptr, &log);
    if (::testing::Test::HasFatalFailure()) return;
    EXPECT_TRUE(loggedRejection(log, "NCCL ATTN")) << "NCCL_DEBUG=ATTN log:\n" << log;
    EXPECT_EQ(log.find("NCCL INFO"), std::string::npos) << "NCCL_DEBUG=ATTN enabled INFO:\n" << log;
  });
}

// NCCL_DEBUG_LEVELS adds single levels to NCCL_DEBUG, here ATTN to a WARN baseline.
TEST(DebugAttn, DebugLevelsAddsAttnToWarn) {
  RUN_ISOLATED_TEST("DebugAttn.DebugLevelsAddsAttnToWarn", []() {
    if (auto reason = attnSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    std::string log;
    logStubRejection("WARN", "ATTN", &log);
    if (::testing::Test::HasFatalFailure()) return;
    EXPECT_TRUE(loggedRejection(log, "NCCL ATTN")) << "NCCL_DEBUG=WARN NCCL_DEBUG_LEVELS=ATTN log:\n" << log;
  });
}

// Control: INFO shows the rejection whichever level logs it, so the stub is loaded
// and rejected and the debug file is written, and the tests above can only fail on
// the level itself.
TEST(DebugAttn, RejectedProfilerPluginLoggedAtInfo) {
  RUN_ISOLATED_TEST("DebugAttn.RejectedProfilerPluginLoggedAtInfo", []() {
    if (auto reason = attnSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    std::string log;
    logStubRejection("INFO", nullptr, &log);
    if (::testing::Test::HasFatalFailure()) return;
    EXPECT_TRUE(loggedRejection(log, "")) << "NCCL_DEBUG=INFO log:\n" << log;
  });
}

} // namespace RcclUnitTesting
