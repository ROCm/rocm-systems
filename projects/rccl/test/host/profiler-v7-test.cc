/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/profiler/profiler_v7.cc: v7 is the native interface, so the plugin's own
// vtable is used as is. The plugin is a FakeLib entry (see profiler-test.cc's ncclOsDlsym).

#include <gtest/gtest.h>

#include <map>
#include <string>

#include "comm.h"
#include "nccl_profiler.h"

#include PROFILER_V7_CC_PATH

namespace {

using FakeLib = std::map<std::string, void*>;

TEST(ProfilerV7Microtest, GetProfiler_ReturnsThePluginsOwnVtable) {
  ncclProfiler_v7_t v7{"FakeV7"};
  FakeLib lib{{"ncclProfiler_v7", &v7}};
  EXPECT_EQ(getNcclProfiler_v7(&lib), &v7);

  FakeLib older{{"ncclProfiler_v6", &v7}};
  EXPECT_EQ(getNcclProfiler_v7(&older), nullptr);
}

}  // namespace
