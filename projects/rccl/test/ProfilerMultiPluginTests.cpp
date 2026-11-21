/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ProfilerMultiPlugin: NCCL_PROFILER_PLUGIN takes a comma-separated list, every plugin in it
// is loaded and initialized, and each one only gets the events its own mask asks for.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>

#include <filesystem>
#include <string>

#include "common/ProcessIsolatedTestRunner.hpp"
#include "nccl_common.h"
#include "nccl_profiler.h"

namespace RcclUnitTesting {

// Isolated because the profiler plugins are loaded once per process, at the first communicator.
TEST(ProfilerMultiPlugin, EachListedPluginGetsItsOwnEvents) {
  RUN_ISOLATED_TEST("ProfilerMultiPlugin.EachListedPluginGetsItsOwnEvents", []() {
    if (int n = 0; hipGetDeviceCount(&n) != hipSuccess || n < 2) GTEST_SKIP() << "requires at least 2 GPUs";
    std::error_code ec;
    const auto src = std::filesystem::read_symlink("/proc/self/exe", ec).parent_path() / RCCL_TEST_PROFILER_MULTI_NAME;
    ASSERT_FALSE(ec) << "could not resolve the test binary's own directory";

    // Two files, so dlopen hands out two handles with separate state; one path twice is deduplicated.
    constexpr int kPlugins = 2;
    const int masks[kPlugins] = {ncclProfileCollApi, ncclProfileGroupApi};
    std::string paths[kPlugins];
    void* libs[kPlugins] = {};
    std::string list;
    for (int i = 0; i < kPlugins; ++i) {
      char tmpl[] = "/tmp/rccl_profiler_multi_XXXXXX";
      int fd = mkstemp(tmpl);
      ASSERT_GE(fd, 0);
      close(fd);
      paths[i] = tmpl;
      ASSERT_TRUE(std::filesystem::copy_file(src, paths[i], std::filesystem::copy_options::overwrite_existing, ec))
          << src << ": " << ec.message();
      libs[i] = dlopen(paths[i].c_str(), RTLD_NOW | RTLD_LOCAL);
      ASSERT_NE(libs[i], nullptr) << dlerror();
      reinterpret_cast<void (*)(int)>(dlsym(libs[i], "rcclMultiProfSetMask"))(masks[i]);
      list += (i ? "," : "") + paths[i];
    }
    ASSERT_EQ(setenv("NCCL_PROFILER_PLUGIN", list.c_str(), 1), 0);

    constexpr int kRanks = 2;
    constexpr size_t kCount = 1024;
    ncclComm_t comms[kRanks];
    hipStream_t streams[kRanks];
    float* bufs[kRanks];
    ASSERT_EQ(ncclCommInitAll(comms, kRanks, nullptr), ncclSuccess);
    for (int r = 0; r < kRanks; ++r) {
      ASSERT_EQ(hipSetDevice(r), hipSuccess);
      ASSERT_EQ(hipStreamCreate(&streams[r]), hipSuccess);
      ASSERT_EQ(hipMalloc(&bufs[r], kCount * sizeof(float)), hipSuccess);
      ASSERT_EQ(hipMemset(bufs[r], 0, kCount * sizeof(float)), hipSuccess);
    }
    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for (int r = 0; r < kRanks; ++r)
      ASSERT_EQ(ncclAllReduce(bufs[r], bufs[r], kCount, ncclFloat, ncclSum, comms[r], streams[r]), ncclSuccess);
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
    for (int r = 0; r < kRanks; ++r) ASSERT_EQ(hipStreamSynchronize(streams[r]), hipSuccess);

    int inits[kPlugins];
    uint64_t seen[kPlugins];
    for (int i = 0; i < kPlugins; ++i) {
      inits[i] = reinterpret_cast<int (*)()>(dlsym(libs[i], "rcclMultiProfInitCount"))();
      seen[i] = reinterpret_cast<uint64_t (*)()>(dlsym(libs[i], "rcclMultiProfSeenTypes"))();
    }
    for (int r = 0; r < kRanks; ++r) {
      ASSERT_EQ(ncclCommDestroy(comms[r]), ncclSuccess);
      ASSERT_EQ(hipStreamDestroy(streams[r]), hipSuccess);
      ASSERT_EQ(hipFree(bufs[r]), hipSuccess);
    }
    for (int i = 0; i < kPlugins; ++i) {
      dlclose(libs[i]);
      unlink(paths[i].c_str());
    }

    for (int i = 0; i < kPlugins; ++i) EXPECT_GE(inits[i], kRanks) << "init() calls on plugin " << i;
    EXPECT_TRUE(seen[0] & ncclProfileCollApi) << "plugin 0 (CollApi mask) saw event types 0x" << std::hex << seen[0];
    EXPECT_TRUE(seen[1] & ncclProfileGroupApi) << "plugin 1 (GroupApi mask) saw event types 0x" << std::hex << seen[1];
    // CollApi does not gate GroupApi: plugin 1 must not get the CollApi events plugin 0 asked for.
    EXPECT_FALSE(seen[1] & ncclProfileCollApi) << "plugin 1 (GroupApi mask) saw event types 0x" << std::hex << seen[1];
  });
}

} // namespace RcclUnitTesting
