// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/rj_vm.h"

#include <gtest/gtest.h>

#include <array>
#include <csignal>
#include <dlfcn.h>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

namespace {

template <typename Function> Function load_function(void *library, const char *name) {
  auto function = reinterpret_cast<Function>(dlsym(library, name));
  EXPECT_NE(function, nullptr) << name << ": " << dlerror();
  return function;
}

void expect_same_action(const struct sigaction &actual, const struct sigaction &expected) {
  EXPECT_EQ(actual.sa_sigaction, expected.sa_sigaction);
  EXPECT_EQ(actual.sa_flags, expected.sa_flags);
  for (int signal_number = 1; signal_number < NSIG; ++signal_number)
    EXPECT_EQ(sigismember(&actual.sa_mask, signal_number),
              sigismember(&expected.sa_mask, signal_number));
}

size_t keeper_lease_count() {
  std::ifstream mappings("/proc/self/maps");
  EXPECT_TRUE(mappings.is_open());
  size_t count = 0;
  for (std::string line; std::getline(mappings, line);)
    if (line.find("/memfd:rocjitsu-owner") != std::string::npos)
      ++count;
  EXPECT_TRUE(mappings.eof());
  return count;
}

constexpr const char *kConfig = R"({
  "num_threads":1,"cpu_dispatch_threads":1,
  "vm":{"arch":"cdna5","gpu":{"device":{
    "gpu_id":1250,"gfx_target_version":120500,"unique_id":1250,
    "num_sdma_engines":0,"local_mem_size":1048576,"simd_count":16,
    "num_shader_engines":1,"num_shader_arrays_per_engine":1,"num_cu_per_sh":4}}},
  "topology":{"root":{"name":"soc","type":"soc","children":[
    {"name":"vram","type":"gpu_memory"},
    {"name":"xcd0","type":"xcd","children":[
      {"name":"l2","type":"l2_cache"},
      {"name":"cp","type":"command_processor"},
      {"name":"se0","type":"shader_engine","children":[
        {"name":"cu[0:4]","type":"compute_unit"}]}]}]}}
})";

TEST(VmEmbeddingApi, UnloadsAfterExplicitLifecycleWithoutChangingHostActions) {
  const size_t initial_leases = keeper_lease_count();
  constexpr std::array signals{SIGSEGV, SIGBUS, SIGINT, SIGTERM};
  std::array<struct sigaction, signals.size()> actions{};
  for (size_t index = 0; index < signals.size(); ++index)
    ASSERT_EQ(sigaction(signals[index], nullptr, &actions[index]), 0);
  ASSERT_EQ(dlsym(RTLD_DEFAULT, "rj_vm_create_from_string"), nullptr);
  for (unsigned cycle = 0; cycle < 2; ++cycle) {
    SCOPED_TRACE(cycle);
    const auto check_actions = [&] {
      for (size_t index = 0; index < signals.size(); ++index) {
        struct sigaction action {};
        ASSERT_EQ(sigaction(signals[index], nullptr, &action), 0);
        expect_same_action(action, actions[index]);
      }
    };
    std::unique_ptr<void, decltype(&dlclose)> library(
        dlopen(RJ_VM_API_LIBRARY, RTLD_NOW | RTLD_LOCAL), dlclose);
    ASSERT_NE(library, nullptr) << dlerror();
    ASSERT_EQ(dlsym(RTLD_DEFAULT, "rj_vm_create_from_string"), nullptr);
    ASSERT_NO_FATAL_FAILURE(check_actions());
    const auto create = load_function<decltype(&rj_vm_create_from_string)>(
        library.get(), "rj_vm_create_from_string");
    const auto destroy = load_function<decltype(&rj_vm_destroy)>(library.get(), "rj_vm_destroy");
    const auto run = load_function<decltype(&rj_vm_run)>(library.get(), "rj_vm_run");
    const auto request_exit =
        load_function<decltype(&rj_vm_request_exit)>(library.get(), "rj_vm_request_exit");
    const auto open_device =
        load_function<decltype(&rj_vm_device_open)>(library.get(), "rj_vm_device_open");
    const auto close_device =
        load_function<decltype(&rj_vm_device_close)>(library.get(), "rj_vm_device_close");
    ASSERT_FALSE(HasFailure());

    rj_vm_t *raw = nullptr;
    ASSERT_EQ(create(kConfig, RJ_VM_MODE_LOCAL, &raw), ROCJITSU_STATUS_SUCCESS);
    std::unique_ptr<rj_vm_t, decltype(destroy)> vm(raw, destroy);
    uint32_t process_id = 0;
    ASSERT_EQ(open_device(vm.get(), 0, &process_id), ROCJITSU_STATUS_SUCCESS);
    ASSERT_NO_FATAL_FAILURE(check_actions());
    rj_status_t run_status = ROCJITSU_STATUS_ERROR;
    std::thread runner([&] { run_status = run(vm.get(), nullptr); });
    // Exit requests are latched even when they race component startup.
    request_exit(vm.get(), "embedding lifecycle complete");
    runner.join();
    EXPECT_EQ(run_status, ROCJITSU_STATUS_SUCCESS);
    EXPECT_EQ(close_device(vm.get(), process_id), ROCJITSU_STATUS_SUCCESS);
    vm.reset();
    EXPECT_EQ(keeper_lease_count(), initial_leases + 1);
    ASSERT_NO_FATAL_FAILURE(check_actions());
    EXPECT_EQ(dlclose(library.release()), 0);
    EXPECT_EQ(keeper_lease_count(), initial_leases);
    std::unique_ptr<void, decltype(&dlclose)> retained(
        dlopen(RJ_VM_API_LIBRARY, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD), dlclose);
    EXPECT_EQ(retained, nullptr) << "library retained after VM and worker teardown";
    ASSERT_NO_FATAL_FAILURE(check_actions());
  }
}

} // namespace
