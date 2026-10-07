// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_shutdown_lifecycle_test.cpp
/// @brief on_shutdown is delivered once across create/load/run/destroy.

#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "rocjitsu/vm/soc.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef PLUGIN_LOADER_FIXTURE_DIR
#error "PLUGIN_LOADER_FIXTURE_DIR must be defined"
#endif

namespace {

using rocjitsu::ExecutionPlugin;
using rocjitsu::ExecutionPluginGroup;
using rocjitsu::PluginSinkConfig;

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

struct VmGuard {
  rj_vm_t *vm = nullptr;
  ~VmGuard() {
    if (vm)
      rj_vm_destroy(vm);
  }
};

/// Throws from onShutdown(), standing in for a plugin whose report write fails.
class ThrowingShutdownPlugin final : public ExecutionPlugin {
public:
  ThrowingShutdownPlugin() : ExecutionPlugin("throws_on_shutdown") {}
  void onShutdown() override { throw std::runtime_error("report write failed"); }
};

class CountingShutdownPlugin final : public ExecutionPlugin {
public:
  explicit CountingShutdownPlugin(int &count) : ExecutionPlugin("counts_shutdown"), count_(count) {}
  void onShutdown() override { ++count_; }

private:
  int &count_;
};

/// A throwing plugin ahead of a counting one, as alphabetical load order can arrange.
std::shared_ptr<ExecutionPluginGroup> make_throwing_group(int &shutdown_count) {
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  EXPECT_TRUE(group->add(std::make_unique<ThrowingShutdownPlugin>()));
  EXPECT_TRUE(group->add(std::make_unique<CountingShutdownPlugin>(shutdown_count)));
  return group;
}

} // namespace

TEST(PluginGroupLifecycle, ShutdownRunsOnceAcrossRunAndDestroy) {
  const std::string config = std::string(CONFIG_DIR) + "/gfx950_mi355x.json";
  rj_vm_t *raw = nullptr;
  ASSERT_EQ(rj_vm_create(config.c_str(), RJ_VM_MODE_DEFAULT, &raw), ROCJITSU_STATUS_SUCCESS);
  VmGuard guard{raw};

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

TEST(PluginGroupLifecycle, ShutdownReachesEveryPluginAndNamesTheOneThatThrew) {
  int shutdown_count = 0;
  auto group = make_throwing_group(shutdown_count);

  const std::vector<std::string> failures = group->onShutdown();

  EXPECT_EQ(shutdown_count, 1);
  ASSERT_EQ(failures.size(), 1u);
  EXPECT_EQ(failures[0], "throws_on_shutdown: report write failed");
}

// rj_vm_run() delivers on_shutdown on the local VM engine thread, where an
// escaping exception would call std::terminate and take the process down.
TEST(PluginGroupLifecycle, ThrowingShutdownOnEngineThreadIsContained) {
  // Declared before the guard: the plugin holds a reference that the VM's destroy may use.
  int shutdown_count = 0;
  const std::string config = std::string(CONFIG_DIR) + "/gfx950_mi355x.json";
  rj_vm_t *raw = nullptr;
  ASSERT_EQ(rj_vm_create(config.c_str(), RJ_VM_MODE_DEFAULT, &raw), ROCJITSU_STATUS_SUCCESS);
  VmGuard guard{raw};

  guard.vm->soc->set_plugin_group(make_throwing_group(shutdown_count));
  guard.vm->plugin_group_active.store(true, std::memory_order_release);

  rj_status_t run_status = ROCJITSU_STATUS_ERROR;
  std::thread engine([&] { run_status = rj_vm_run(guard.vm, nullptr); });
  rj_vm_request_exit(guard.vm, "throwing plugin shutdown test");
  engine.join();

  EXPECT_EQ(run_status, ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(shutdown_count, 1);
  EXPECT_FALSE(guard.vm->plugin_group_active.load(std::memory_order_acquire));

  rj_vm_t *vm = guard.vm;
  guard.vm = nullptr;
  rj_vm_destroy(vm);
  EXPECT_EQ(shutdown_count, 1);
}
