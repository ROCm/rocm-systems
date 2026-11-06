/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/profiler/profiler_v8.cc: v8 is the native interface, so the plugin's own
// vtable is used as is and no shim is involved. The plugin is a FakeLib entry (see profiler-test.cc's ncclOsDlsym).

#include <gtest/gtest.h>

#include <map>
#include <string>

#include "comm.h"
#include "nccl_profiler.h"

#include PROFILER_V8_CC_PATH

namespace {

using FakeLib = std::map<std::string, void*>;

TEST(ProfilerV8Microtest, GetProfiler_ReturnsThePluginsOwnVtableAsApiAndTarget) {
  ncclProfiler_v8_t v8{"FakeV8"};
  FakeLib lib{{"ncclProfiler_v8", &v8}};
  void* target = nullptr;
  EXPECT_EQ(getNcclProfiler_v8(&lib, &target), &v8);
  EXPECT_EQ(target, &v8);

  FakeLib older{{"ncclProfiler_v7", &v8}};
  target = &v8;
  EXPECT_EQ(getNcclProfiler_v8(&older, &target), nullptr);
  EXPECT_EQ(target, nullptr);
}

}  // namespace
