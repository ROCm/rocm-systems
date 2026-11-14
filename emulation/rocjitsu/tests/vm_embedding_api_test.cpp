// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/rj_vm.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/kernel_descriptor_abi.h"
#include "rocjitsu/code/rj_code.h"
#include "scoped_temp.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_hsa_queue.h"
#include "hsa/amd_hsa_signal.h"
#include "uapi/linux/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <csignal>
#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <vector>

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

enum class Lifecycle {
  CreateDestroy,
  Paths,
  CreateQueue,
  StepBarrier,
  RunBarrier,
  StepKernel,
  RunKernel
};

void exercise_lifecycle(Lifecycle lifecycle) {
  const bool create_queue = lifecycle != Lifecycle::CreateDestroy && lifecycle != Lifecycle::Paths;
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
    const auto step = load_function<decltype(&rj_vm_step)>(library.get(), "rj_vm_step");
    const auto topology_path =
        load_function<decltype(&rj_vm_topology_path)>(library.get(), "rj_vm_topology_path");
    const auto drm_path = load_function<decltype(&rj_vm_drm_path)>(library.get(), "rj_vm_drm_path");
    const auto request_exit =
        load_function<decltype(&rj_vm_request_exit)>(library.get(), "rj_vm_request_exit");
    const auto open_device =
        load_function<decltype(&rj_vm_device_open)>(library.get(), "rj_vm_device_open");
    const auto close_device =
        load_function<decltype(&rj_vm_device_close)>(library.get(), "rj_vm_device_close");
    const auto execute =
        load_function<decltype(&rj_vm_execute_as)>(library.get(), "rj_vm_execute_as");
    const auto map_device =
        load_function<decltype(&rj_vm_device_map_as)>(library.get(), "rj_vm_device_map_as");
    const auto unmap_device =
        load_function<decltype(&rj_vm_device_unmap_as)>(library.get(), "rj_vm_device_unmap_as");
    ASSERT_FALSE(::testing::Test::HasFailure());

    struct alignas(64) Kernel {
      // The code-object ABI record consumed by the dispatch packet.
      rocjitsu::amdhsa::kernel_descriptor_t descriptor{};
      // CDNA5 s_endpgm, so completion proves a CU execution batch ran.
      uint32_t instruction = 0xBFB00000u;
    };
    struct alignas(4096) Storage {
      // KFD queue descriptor, including producer and consumer cursors.
      amd_queue_v2_t descriptor{};
      // Producer-owned AQL ring exposed through the same USERPTR allocation.
      alignas(64) std::array<hsa_barrier_and_packet_t, 64> ring{};
      // Host-visible completion reached through the GPU's registered mapping.
      amd_signal_t completion{};
      // Descriptor and instruction storage for the kernel lifecycle cases.
      Kernel kernel;
    } storage;
    auto &descriptor = storage.descriptor;
    auto &ring = storage.ring;
    auto &completion = storage.completion;
    auto &kernel = storage.kernel;
    for (auto &packet : ring)
      packet.header = HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
    descriptor.hsa_queue.base_address = ring.data();
    descriptor.hsa_queue.size = ring.size();
    descriptor.hsa_queue.type = HSA_QUEUE_TYPE_MULTI;
    descriptor.hsa_queue.features = HSA_QUEUE_FEATURE_KERNEL_DISPATCH;
    rj_vm_t *raw = nullptr;
    ASSERT_EQ(create(kConfig, RJ_VM_MODE_LOCAL, &raw), ROCJITSU_STATUS_SUCCESS);
    std::unique_ptr<rj_vm_t, decltype(destroy)> vm(raw, destroy);
    if (lifecycle == Lifecycle::Paths) {
      const char *path = nullptr;
      ASSERT_EQ(topology_path(vm.get(), &path), ROCJITSU_STATUS_SUCCESS);
      ASSERT_NE(path, nullptr);
      EXPECT_NE(*path, '\0');
      ASSERT_EQ(drm_path(vm.get(), &path), ROCJITSU_STATUS_SUCCESS);
      ASSERT_NE(path, nullptr);
      EXPECT_NE(*path, '\0');
    }
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
      allocation.size = sizeof(storage);
      allocation.gpu_id = 1250;
      allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                         KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;
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
    rj_vm_map_t doorbell{};
    const bool execute_packet = create_queue && lifecycle != Lifecycle::CreateQueue;
    completion.kind = AMD_SIGNAL_KIND_USER;
    completion.value = 1;
    auto completed = [&] {
      return std::atomic_ref<int64_t>(const_cast<int64_t &>(completion.value))
                 .load(std::memory_order_acquire) == 0;
    };
    kernel.descriptor.kernel_code_entry_byte_offset = offsetof(Kernel, instruction);
    if (execute_packet) {
      doorbell.length = 4096;
      doorbell.offset = queue.doorbell_offset & ~uint64_t{4095};
      doorbell.prot = PROT_READ | PROT_WRITE;
      doorbell.flags = MAP_SHARED;
      ASSERT_EQ(map_device(vm.get(), process_id, &doorbell), ROCJITSU_STATUS_SUCCESS);
      ASSERT_EQ(doorbell.map_errno, 0);
      ASSERT_NE(doorbell.mapped_addr, reinterpret_cast<uint64_t>(MAP_FAILED));
      const bool dispatch_kernel =
          lifecycle == Lifecycle::StepKernel || lifecycle == Lifecycle::RunKernel;
      uint16_t header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
      if (dispatch_kernel) {
        hsa_kernel_dispatch_packet_t dispatch{};
        dispatch.header = HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
        dispatch.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
        dispatch.workgroup_size_x = dispatch.workgroup_size_y = dispatch.workgroup_size_z = 1;
        dispatch.grid_size_x = dispatch.grid_size_y = dispatch.grid_size_z = 1;
        dispatch.kernel_object = reinterpret_cast<uint64_t>(&kernel.descriptor);
        dispatch.completion_signal.handle = reinterpret_cast<uint64_t>(&completion);
        std::memcpy(ring.data(), &dispatch, sizeof(dispatch));
        header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
      } else {
        ring[0].completion_signal.handle = reinterpret_cast<uint64_t>(&completion);
      }
      std::atomic_ref<uint16_t>(ring[0].header).store(header, std::memory_order_release);
      std::atomic_ref<uint64_t>(const_cast<uint64_t &>(descriptor.write_dispatch_id))
          .store(1, std::memory_order_release);
      auto *doorbell_word =
          reinterpret_cast<uint64_t *>(doorbell.mapped_addr + (queue.doorbell_offset & 4095));
      std::atomic_ref<uint64_t>(*doorbell_word).store(0, std::memory_order_release);
      if (lifecycle == Lifecycle::StepBarrier || lifecycle == Lifecycle::StepKernel) {
        while (!completed())
          ASSERT_EQ(step(vm.get(), nullptr), ROCJITSU_STATUS_SUCCESS);
      } else {
        std::jthread stopper([&](std::stop_token stop) {
          while (!stop.stop_requested() && !completed())
            std::this_thread::yield();
          if (completed())
            request_exit(vm.get(), "embedding packet completed");
        });
        EXPECT_EQ(run(vm.get(), nullptr), ROCJITSU_STATUS_SUCCESS);
        stopper.request_stop();
      }
      EXPECT_TRUE(completed());
      EXPECT_EQ(
          std::atomic_ref<uint64_t>(const_cast<uint64_t &>(descriptor.read_dispatch_id)).load(),
          1u);
    } else {
      rj_status_t run_status = ROCJITSU_STATUS_ERROR;
      std::thread runner([&] { run_status = run(vm.get(), nullptr); });
      // Exit requests are latched even when they race component startup.
      request_exit(vm.get(), "embedding lifecycle complete");
      runner.join();
      EXPECT_EQ(run_status, ROCJITSU_STATUS_SUCCESS);
    }
    if (create_queue) {
      kfd_ioctl_destroy_queue_args destroy_queue{};
      destroy_queue.queue_id = queue.queue_id;
      ASSERT_NO_FATAL_FAILURE(command(AMDKFD_IOC_DESTROY_QUEUE, destroy_queue));
      if (doorbell.length) {
        rj_vm_unmap_t unmap{doorbell.mapped_addr, doorbell.length};
        EXPECT_EQ(unmap_device(vm.get(), process_id, &unmap), ROCJITSU_STATUS_SUCCESS);
      }
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
  exercise_lifecycle(Lifecycle::CreateDestroy);
}

TEST(VmEmbeddingApi, UnloadsAfterCommandTranslationsOnALiveHostThread) {
  exercise_lifecycle(Lifecycle::CreateQueue);
}

TEST(VmEmbeddingApi, UnloadsAfterPathQueriesOnALiveHostThread) {
  exercise_lifecycle(Lifecycle::Paths);
}

TEST(VmEmbeddingApi, UnloadsAfterSteppingABarrierOnALiveHostThread) {
  exercise_lifecycle(Lifecycle::StepBarrier);
}

TEST(VmEmbeddingApi, UnloadsAfterRunningABarrierOnALiveHostThread) {
  exercise_lifecycle(Lifecycle::RunBarrier);
}

TEST(VmEmbeddingApi, UnloadsAfterSteppingAKernelOnALiveHostThread) {
  exercise_lifecycle(Lifecycle::StepKernel);
}

TEST(VmEmbeddingApi, UnloadsAfterRunningAKernelOnALiveHostThread) {
  exercise_lifecycle(Lifecycle::RunKernel);
}

void exercise_code_inspection(bool basic_blocks) {
  using namespace rocjitsu;
  constexpr std::array<uint32_t, 2> text_words{0xBF810000u, 0xBF800000u};
  constexpr char section_names[] = "\0.text\0.shstrtab\0";
  constexpr size_t text_offset = sizeof(Elf64_Ehdr);
  constexpr size_t names_offset = text_offset + sizeof(text_words);
  constexpr size_t sections_offset = (names_offset + sizeof(section_names) + 7) & ~size_t{7};
  std::array<Elf64_Shdr, 3> sections{};
  sections[1].sh_name = 1;
  sections[1].sh_type = SHT_PROGBITS;
  sections[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  sections[1].sh_offset = text_offset;
  sections[1].sh_size = sizeof(text_words);
  sections[1].sh_addralign = sizeof(uint32_t);
  sections[2].sh_name = 7;
  sections[2].sh_type = SHT_STRTAB;
  sections[2].sh_offset = names_offset;
  sections[2].sh_size = sizeof(section_names);
  sections[2].sh_addralign = 1;
  Elf64_Ehdr header{};
  std::memcpy(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  header.e_ident[EI_CLASS] = ELFCLASS64;
  header.e_ident[EI_DATA] = ELFDATA2LSB;
  header.e_ident[EI_VERSION] = 1;
  header.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  header.e_type = ET_REL;
  header.e_machine = EM_AMDGPU;
  header.e_version = 1;
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX942;
  header.e_ehsize = sizeof(header);
  header.e_shoff = sections_offset;
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shnum = sections.size();
  header.e_shstrndx = 2;
  std::vector<uint8_t> image(sections_offset + sizeof(sections));
  std::memcpy(image.data(), &header, sizeof(header));
  std::memcpy(image.data() + text_offset, text_words.data(), sizeof(text_words));
  std::memcpy(image.data() + names_offset, section_names, sizeof(section_names));
  std::memcpy(image.data() + sections_offset, sections.data(), sizeof(sections));
  test::ScopedTempFile file;
  ASSERT_TRUE(file.initialize("rj-code-embedding-"));
  ASSERT_TRUE(file.write({reinterpret_cast<const char *>(image.data()), image.size()}));

  const size_t initial_leases = keeper_lease_count();
  for (unsigned cycle = 0; cycle < 2; ++cycle) {
    SCOPED_TRACE(cycle);
    std::unique_ptr<void, decltype(&dlclose)> library(
        dlopen(RJ_VM_API_LIBRARY, RTLD_NOW | RTLD_LOCAL), dlclose);
    ASSERT_NE(library, nullptr) << dlerror();
    const auto create = load_function<decltype(&rj_code_executable_create)>(
        library.get(), "rj_code_executable_create");
    const auto destroy = load_function<decltype(&rj_code_executable_destroy)>(
        library.get(), "rj_code_executable_destroy");
    const auto get_object = load_function<decltype(&rj_code_executable_get_code_object)>(
        library.get(), "rj_code_executable_get_code_object");
    const auto destroy_object =
        load_function<decltype(&rj_code_object_destroy)>(library.get(), "rj_code_object_destroy");
    const auto release_object =
        load_function<decltype(&rj_code_object_release)>(library.get(), "rj_code_object_release");
    const auto create_instructions = load_function<decltype(&rj_code_inst_list_create)>(
        library.get(), "rj_code_inst_list_create");
    const auto destroy_instructions = load_function<decltype(&rj_code_inst_list_destroy)>(
        library.get(), "rj_code_inst_list_destroy");
    const auto create_blocks = load_function<decltype(&rj_code_basic_block_list_create)>(
        library.get(), "rj_code_basic_block_list_create");
    const auto destroy_blocks = load_function<decltype(&rj_code_basic_block_list_destroy)>(
        library.get(), "rj_code_basic_block_list_destroy");
    ASSERT_FALSE(::testing::Test::HasFailure());
    rj_code_executable_t *raw_executable = nullptr;
    ASSERT_EQ(create(file.path().c_str(), &raw_executable), ROCJITSU_STATUS_SUCCESS);
    std::unique_ptr<rj_code_executable_t, decltype(destroy)> executable(raw_executable, destroy);
    rj_code_object_t *raw_object = nullptr;
    ASSERT_EQ(get_object(executable.get(), ROCJITSU_CODE_TARGET_GFX942, 0, &raw_object),
              ROCJITSU_STATUS_SUCCESS);
    const auto delete_object = [&](rj_code_object_t *object) {
      destroy_object(object);
      release_object(object);
    };
    std::unique_ptr<rj_code_object_t, decltype(delete_object)> object(raw_object, delete_object);
    executable.reset();
    if (basic_blocks) {
      rj_code_basic_block_list_t *blocks = nullptr;
      ASSERT_EQ(create_blocks(object.get(), ROCJITSU_CODE_TARGET_GFX942, &blocks),
                ROCJITSU_STATUS_SUCCESS);
      ASSERT_NE(blocks, nullptr);
      destroy_blocks(blocks);
    } else {
      rj_code_inst_list_t *instructions = nullptr;
      ASSERT_EQ(create_instructions(object.get(), ROCJITSU_CODE_TARGET_GFX942, &instructions),
                ROCJITSU_STATUS_SUCCESS);
      ASSERT_NE(instructions, nullptr);
      destroy_instructions(instructions);
    }
    object.reset();
    EXPECT_EQ(dlclose(library.release()), 0);
    EXPECT_EQ(keeper_lease_count(), initial_leases);
    std::unique_ptr<void, decltype(&dlclose)> retained(
        dlopen(RJ_VM_API_LIBRARY, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD), dlclose);
    EXPECT_EQ(retained, nullptr) << "library retained after code inspection";
  }
}

TEST(VmEmbeddingApi, UnloadsAfterInstructionInspectionOnALiveHostThread) {
  exercise_code_inspection(false);
}

TEST(VmEmbeddingApi, UnloadsAfterBasicBlockInspectionOnALiveHostThread) {
  exercise_code_inspection(true);
}

} // namespace
