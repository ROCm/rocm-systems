// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_shutdown_lifecycle_test.cpp
/// @brief on_shutdown is delivered once across create/load/run/destroy.

#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

#ifndef PLUGIN_LOADER_FIXTURE_DIR
#error "PLUGIN_LOADER_FIXTURE_DIR must be defined"
#endif

namespace {

int count_events(const std::string &trace, const char *event) {
  const std::string needle = std::string("good:") + event + "\n";
  int count = 0;
  for (std::size_t pos = 0; (pos = trace.find(needle, pos)) != std::string::npos;
       pos += needle.size())
    ++count;
  return count;
}

std::string read_file(const std::string &path) {
  std::ifstream input(path);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

} // namespace

TEST(PluginGroupLifecycle, ShutdownRunsOnceAcrossRunAndDestroy) {
  const std::string config = std::string(CONFIG_DIR) + "/gfx950_mi355x.json";
  rj_vm_t *raw = nullptr;
  ASSERT_EQ(rj_vm_create(config.c_str(), RJ_VM_MODE_DEFAULT, &raw), ROCJITSU_STATUS_SUCCESS);
  struct Guard {
    rj_vm_t *vm = nullptr;
    ~Guard() {
      if (vm)
        rj_vm_destroy(vm);
    }
  } guard{raw};

  rocjitsu::test::ScopedTempFile trace_file("plugin-shutdown-lifecycle-");
  ASSERT_EQ(setenv("ROCJITSU_PLUGIN_TEST_TRACE", trace_file.path().c_str(), 1), 0);
  struct UnsetTrace {
    ~UnsetTrace() { unsetenv("ROCJITSU_PLUGIN_TEST_TRACE"); }
  } unset_trace;

  ASSERT_EQ(rj_vm_load_plugins(guard.vm, R"({"plugins":{"good":{}}})", PLUGIN_LOADER_FIXTURE_DIR),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_TRUE(guard.vm->plugin_group_active.load(std::memory_order_acquire));

  rj_status_t run_status = ROCJITSU_STATUS_ERROR;
  std::thread engine([&] { run_status = rj_vm_run(guard.vm, nullptr); });
  rj_vm_request_exit(guard.vm, "plugin shutdown lifecycle test");
  engine.join();

  EXPECT_EQ(run_status, ROCJITSU_STATUS_SUCCESS);
  EXPECT_FALSE(guard.vm->plugin_group_active.load(std::memory_order_acquire));

  rj_vm_t *vm = guard.vm;
  guard.vm = nullptr;
  rj_vm_destroy(vm);

  const std::string events = read_file(trace_file.path());
  EXPECT_EQ(count_events(events, "shutdown"), 1) << events;
}
