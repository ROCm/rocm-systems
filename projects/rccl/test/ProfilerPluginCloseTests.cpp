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
#include <string>

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {
namespace {

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

    const char* stubPath = RCCL_TEST_PROFILER_STUB_PATH;
    ASSERT_EQ(access(stubPath, R_OK), 0) << "stub profiler plugin missing: " << stubPath;
    ASSERT_EQ(setenv("NCCL_PROFILER_PLUGIN", stubPath, 1), 0);

    // A plugin RCCL cannot use must not stop a communicator from coming up.
    ncclUniqueId id;
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
    ncclComm_t comm = nullptr;
    ASSERT_EQ(ncclCommInitRank(&comm, 1, id, 0), ncclSuccess);
    ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);

    // RTLD_NOLOAD loads nothing and returns non-NULL only while the object is
    // mapped; it still takes a reference, so drop it before asserting.
    void* handle = dlopen(stubPath, RTLD_NOLOAD | RTLD_LAZY);
    if (handle != nullptr) dlclose(handle);
    EXPECT_EQ(handle, nullptr)
        << "profiler plugin " << stubPath << " is still mapped after comm create/destroy";
  });
}

} // namespace RcclUnitTesting
