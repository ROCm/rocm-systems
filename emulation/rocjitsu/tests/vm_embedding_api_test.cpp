// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/rj_vm.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_hsa_queue.h"
#include "uapi/linux/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <array>
#include <csignal>
#include <cstddef>
#include <dlfcn.h>
#include <fstream>
#include <memory>
#include <new>
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

void exercise_lifecycle(bool create_queue) {
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
    const auto destroy =
        load_function<decltype(&rj_vm_destroy_checked)>(library.get(), "rj_vm_destroy_checked");
    const auto run = load_function<decltype(&rj_vm_run)>(library.get(), "rj_vm_run");
    const auto request_exit =
        load_function<decltype(&rj_vm_request_exit)>(library.get(), "rj_vm_request_exit");
    const auto open_device =
        load_function<decltype(&rj_vm_device_open)>(library.get(), "rj_vm_device_open");
    const auto close_device =
        load_function<decltype(&rj_vm_device_close)>(library.get(), "rj_vm_device_close");
    const auto execute =
        load_function<decltype(&rj_vm_execute_as)>(library.get(), "rj_vm_execute_as");
    ASSERT_FALSE(::testing::Test::HasFailure());

    alignas(4096) std::array<std::byte, 4096> descriptor_storage{};
    static_assert(sizeof(amd_queue_v2_t) <= sizeof(descriptor_storage));
    auto &descriptor = *new (descriptor_storage.data()) amd_queue_v2_t{};
    alignas(4096) std::array<hsa_barrier_and_packet_t, 64> ring{};
    for (auto &packet : ring)
      packet.header = HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
    descriptor.hsa_queue.base_address = ring.data();
    descriptor.hsa_queue.size = ring.size();
    descriptor.hsa_queue.type = HSA_QUEUE_TYPE_MULTI;
    descriptor.hsa_queue.features = HSA_QUEUE_FEATURE_KERNEL_DISPATCH;
    rj_vm_t *raw = nullptr;
    ASSERT_EQ(create(kConfig, RJ_VM_MODE_LOCAL, &raw), ROCJITSU_STATUS_SUCCESS);
    std::unique_ptr<rj_vm_t, decltype(destroy)> vm(raw, destroy);
    uint32_t process_id = 0;
    ASSERT_EQ(open_device(vm.get(), 0, &process_id), ROCJITSU_STATUS_SUCCESS);
    ASSERT_NO_FATAL_FAILURE(check_actions());
    const auto command = [&](uint32_t opcode, auto &arguments) {
      rj_vm_cmd_t command{};
      command.cmd = opcode;
      command.buf = &arguments;
      command.buf_size = sizeof(arguments);
      command.shared_handle = command.in_handle = command.in_mem_handle = command.in_proc_handle =
          -1;
      ASSERT_EQ(execute(vm.get(), process_id, &command), ROCJITSU_STATUS_SUCCESS);
      ASSERT_EQ(command.result, 0);
    };
    kfd_ioctl_alloc_memory_of_gpu_args allocation{};
    kfd_ioctl_create_queue_args queue{};
    if (create_queue) {
      allocation.va_addr = reinterpret_cast<uintptr_t>(&descriptor);
      allocation.mmap_offset = allocation.va_addr;
      allocation.size = descriptor_storage.size();
      allocation.gpu_id = 1250;
      allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
      ASSERT_NO_FATAL_FAILURE(command(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, allocation));
      struct {
        // KFD command header followed by its inline device list.
        kfd_ioctl_map_memory_to_gpu_args arguments{};
        // GPU receiving access to the queue descriptor.
        uint32_t gpu_id = 1250;
      } mapping;
      mapping.arguments.handle = allocation.handle;
      mapping.arguments.n_devices = 1;
      ASSERT_NO_FATAL_FAILURE(command(AMDKFD_IOC_MAP_MEMORY_TO_GPU, mapping));
      ASSERT_EQ(mapping.arguments.n_success, 1u);
      queue.ring_base_address = reinterpret_cast<uintptr_t>(ring.data());
      queue.ring_size = sizeof(ring);
      queue.read_pointer_address = reinterpret_cast<uintptr_t>(&descriptor.read_dispatch_id);
      queue.write_pointer_address = reinterpret_cast<uintptr_t>(&descriptor.write_dispatch_id);
      queue.gpu_id = allocation.gpu_id;
      queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
      queue.queue_percentage = 100;
      ASSERT_NO_FATAL_FAILURE(command(AMDKFD_IOC_CREATE_QUEUE, queue));
      // CDNA5 registration translates and updates this word on the command thread.
      ASSERT_NE(descriptor.caps & AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM, 0u);
    }
    rj_status_t run_status = ROCJITSU_STATUS_ERROR;
    std::thread runner([&] { run_status = run(vm.get(), nullptr); });
    // Exit requests are latched even when they race component startup.
    request_exit(vm.get(), "embedding lifecycle complete");
    runner.join();
    EXPECT_EQ(run_status, ROCJITSU_STATUS_SUCCESS);
    if (create_queue) {
      kfd_ioctl_destroy_queue_args destroy_queue{};
      destroy_queue.queue_id = queue.queue_id;
      ASSERT_NO_FATAL_FAILURE(command(AMDKFD_IOC_DESTROY_QUEUE, destroy_queue));
      struct {
        // KFD command header followed by its inline device list.
        kfd_ioctl_unmap_memory_from_gpu_args arguments{};
        // GPU whose descriptor access is retired before freeing the backing.
        uint32_t gpu_id = 1250;
      } unmapping;
      unmapping.arguments.handle = allocation.handle;
      unmapping.arguments.n_devices = 1;
      ASSERT_NO_FATAL_FAILURE(command(AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, unmapping));
      ASSERT_EQ(unmapping.arguments.n_success, 1u);
      kfd_ioctl_free_memory_of_gpu_args free_memory{};
      free_memory.handle = allocation.handle;
      ASSERT_NO_FATAL_FAILURE(command(AMDKFD_IOC_FREE_MEMORY_OF_GPU, free_memory));
    }
    EXPECT_EQ(close_device(vm.get(), process_id), ROCJITSU_STATUS_SUCCESS);
    ASSERT_EQ(destroy(vm.get()), ROCJITSU_STATUS_SUCCESS);
    (void)vm.release();
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

TEST(VmEmbeddingApi, UnloadsAfterExplicitLifecycleWithoutChangingHostActions) {
  exercise_lifecycle(false);
}

TEST(VmEmbeddingApi, UnloadsAfterCommandTranslationsOnALiveHostThread) { exercise_lifecycle(true); }

} // namespace
