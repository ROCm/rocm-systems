// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file simulated_kfd_test.cpp
/// @brief Tests for SimulatedKfd creation, open/close, and topology generation.

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/guest_kfd.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include "embedded_schema.h"
#include "simdojo/sim/simulation.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace {

const std::string CONFIG_PATH = std::string(CONFIG_DIR) + "/gfx950_cdna4.json";

struct TestVM {
  rocjitsu::config::LoadedConfig loaded;
  std::unique_ptr<simdojo::SimulationEngine> engine;

  rocjitsu::SimulatedKfd *driver() {
    auto *vm = dynamic_cast<rocjitsu::VirtualMachine *>(engine->topology().root());
    return vm ? vm->driver() : nullptr;
  }
};

TestVM create_test_vm() {
  TestVM t;
  t.loaded = rocjitsu::config::load_config(CONFIG_PATH.c_str(), rocjitsu::kEmbeddedSchema);
  auto *soc = t.loaded.soc();

  t.loaded.engine_config.max_ticks = 0;
  t.loaded.engine_config.await_primaries = true;
  t.engine = std::make_unique<simdojo::SimulationEngine>(t.loaded.engine_config);

  auto root = t.loaded.take_root();
  root.release();
  auto vm = std::make_unique<rocjitsu::VirtualMachine>(std::unique_ptr<rocjitsu::SoC>(soc));
  vm->driver()->setup_topology(t.loaded.device, soc->num_xcds());

  t.engine->topology().set_root(std::move(vm));
  t.loaded.wire_links(t.engine->topology());
  soc->wire_backing(t.engine->topology());
  t.engine->create();
  t.engine->register_as_primary();

  return t;
}

class SimulatedKfdTest : public ::testing::Test {
protected:
  void SetUp() override { setenv("RJ_CONFIG", CONFIG_PATH.c_str(), 1); }
};

TEST_F(SimulatedKfdTest, CreateDefault) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
}

TEST_F(SimulatedKfdTest, OpenAndClose) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);

  int fd = t.driver()->open();
  EXPECT_GE(fd, 0);

  int ret = t.driver()->close();
  EXPECT_EQ(ret, 0);
}

TEST_F(SimulatedKfdTest, PermanentVramBackingSurvivesCpuMapLifecycle) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  auto *vm = dynamic_cast<rocjitsu::VirtualMachine *>(t.engine->topology().root());
  ASSERT_NE(vm, nullptr);
  auto *memory = vm->memory();
  ASSERT_NE(memory, nullptr);

  ASSERT_GE(t.driver()->open(), 0);
  uint32_t process_id = t.driver()->local_process_id();
  auto process = t.driver()->find_process(process_id);
  ASSERT_NE(process, nullptr);

  constexpr size_t kPageSize = rocjitsu::KfdProcess::kPageSize;
  void *reservation = ::mmap(nullptr, kPageSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(reservation, MAP_FAILED);

  kfd_ioctl_alloc_memory_of_gpu_args alloc{};
  alloc.va_addr = reinterpret_cast<uint64_t>(reservation);
  alloc.size = kPageSize;
  alloc.gpu_id = t.driver()->gpu_id();
  alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc), 0);

  {
    auto backing = process->resolve_backing(alloc.va_addr, alloc.size);
    ASSERT_TRUE(backing.has_value());
    EXPECT_FALSE(backing->gpu_accessible);
  }
  EXPECT_EQ(memory->resolve_host_ptr(alloc.va_addr, process_id), nullptr);

  uint32_t gpu_id = alloc.gpu_id;
  kfd_ioctl_map_memory_to_gpu_args map{};
  map.handle = alloc.handle;
  map.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
  map.n_devices = 1;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map), 0);
  ASSERT_EQ(map.n_success, 1u);
  EXPECT_NE(memory->resolve_host_ptr(alloc.va_addr, process_id), nullptr);
  EXPECT_TRUE(process->page_table_.empty());

  constexpr uint32_t kGpuValue = 0xABCDEF01u;
  constexpr uint32_t kCpuValue = 0x10203040u;
  memory->write32(alloc.va_addr, kGpuValue, process_id);

  void *cpu_mapping =
      t.driver()->mmap(reservation, kPageSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                       static_cast<off_t>(alloc.mmap_offset));
  ASSERT_EQ(cpu_mapping, reservation);
  EXPECT_EQ(*static_cast<uint32_t *>(cpu_mapping), kGpuValue);
  *static_cast<uint32_t *>(cpu_mapping) = kCpuValue;
  EXPECT_EQ(memory->read32(alloc.va_addr, process_id), kCpuValue);

  ASSERT_EQ(t.driver()->munmap(cpu_mapping, kPageSize), 0);
  EXPECT_EQ(memory->read32(alloc.va_addr, process_id), kCpuValue);

  cpu_mapping = t.driver()->mmap(reservation, kPageSize, PROT_READ | PROT_WRITE,
                                 MAP_SHARED | MAP_FIXED, static_cast<off_t>(alloc.mmap_offset));
  ASSERT_EQ(cpu_mapping, reservation);
  EXPECT_EQ(*static_cast<uint32_t *>(cpu_mapping), kCpuValue);
  ASSERT_EQ(t.driver()->munmap(cpu_mapping, kPageSize), 0);

  kfd_ioctl_unmap_memory_from_gpu_args unmap{};
  unmap.handle = alloc.handle;
  unmap.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
  unmap.n_devices = 1;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmap), 0);
  EXPECT_EQ(memory->resolve_host_ptr(alloc.va_addr, process_id), nullptr);

  kfd_ioctl_free_memory_of_gpu_args free_args{};
  free_args.handle = alloc.handle;
  EXPECT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args), 0);
  EXPECT_EQ(t.driver()->close(), 0);
}

TEST_F(SimulatedKfdTest, PermanentVramBackingTracksPartialAndCrossRangeCpuUnmaps) {
  TestVM t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  ASSERT_GE(t.driver()->open(), 0);
  const uint32_t process_id = t.driver()->local_process_id();
  std::shared_ptr<rocjitsu::KfdProcess> process = t.driver()->find_process(process_id);
  ASSERT_NE(process, nullptr);

  constexpr size_t kPageSize = rocjitsu::KfdProcess::kPageSize;
  constexpr size_t kAllocationSize = 3 * kPageSize;
  constexpr size_t kMappingLength = 2 * kPageSize + 1;
  void *reservation =
      ::mmap(nullptr, kAllocationSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(reservation, MAP_FAILED);

  kfd_ioctl_alloc_memory_of_gpu_args alloc{};
  alloc.va_addr = reinterpret_cast<uint64_t>(reservation);
  alloc.size = kAllocationSize;
  alloc.gpu_id = t.driver()->gpu_id();
  alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc), 0);

  void *mapping = t.driver()->mmap(reservation, kMappingLength, PROT_READ | PROT_WRITE,
                                   MAP_SHARED | MAP_FIXED, static_cast<off_t>(alloc.mmap_offset));
  ASSERT_EQ(mapping, reservation);
  const uintptr_t begin = reinterpret_cast<uintptr_t>(mapping);
  {
    std::lock_guard<std::mutex> lock(process->alloc_mutex_);
    ASSERT_EQ(process->cpu_mappings_.size(), 1u);
    EXPECT_EQ(process->cpu_mappings_.begin()->first, begin);
    EXPECT_EQ(process->cpu_mappings_.begin()->second, begin + kAllocationSize);
  }

  void *middle_page = static_cast<uint8_t *>(mapping) + kPageSize;
  ASSERT_EQ(t.driver()->munmap(middle_page, kPageSize), 0);
  {
    std::lock_guard<std::mutex> lock(process->alloc_mutex_);
    ASSERT_EQ(process->cpu_mappings_.size(), 2u);
    std::map<uintptr_t, uintptr_t>::const_iterator first = process->cpu_mappings_.begin();
    EXPECT_EQ(first->first, begin);
    EXPECT_EQ(first->second, begin + kPageSize);
    std::map<uintptr_t, uintptr_t>::const_iterator second = std::next(first);
    EXPECT_EQ(second->first, begin + 2 * kPageSize);
    EXPECT_EQ(second->second, begin + kAllocationSize);
  }

  ASSERT_EQ(t.driver()->munmap(mapping, kAllocationSize), 0);
  {
    std::lock_guard<std::mutex> lock(process->alloc_mutex_);
    EXPECT_TRUE(process->cpu_mappings_.empty());
  }

  mapping = t.driver()->mmap(reservation, kMappingLength, PROT_READ | PROT_WRITE,
                             MAP_SHARED | MAP_FIXED, static_cast<off_t>(alloc.mmap_offset));
  ASSERT_EQ(mapping, reservation);
  ASSERT_EQ(t.driver()->munmap(middle_page, kPageSize), 0);
  ASSERT_EQ(t.driver()->munmap(middle_page, 2 * kPageSize), 0);
  {
    std::lock_guard<std::mutex> lock(process->alloc_mutex_);
    ASSERT_EQ(process->cpu_mappings_.size(), 1u);
    EXPECT_EQ(process->cpu_mappings_.begin()->first, begin);
    EXPECT_EQ(process->cpu_mappings_.begin()->second, begin + kPageSize);
  }

  ASSERT_EQ(t.driver()->munmap(mapping, 1), 0);
  {
    std::lock_guard<std::mutex> lock(process->alloc_mutex_);
    EXPECT_TRUE(process->cpu_mappings_.empty());
  }

  kfd_ioctl_free_memory_of_gpu_args free_args{};
  free_args.handle = alloc.handle;
  EXPECT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args), 0);
  EXPECT_EQ(t.driver()->close(), 0);
}

TEST_F(SimulatedKfdTest, PermanentVramBackingImportsAcrossPartialMappings) {
  TestVM t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  rocjitsu::VirtualMachine *vm =
      dynamic_cast<rocjitsu::VirtualMachine *>(t.engine->topology().root());
  ASSERT_NE(vm, nullptr);
  rocjitsu::amdgpu::GpuMemory *memory = vm->memory();
  ASSERT_NE(memory, nullptr);
  ASSERT_GE(t.driver()->open(), 0);
  const uint32_t process_id = t.driver()->local_process_id();

  constexpr size_t kPageSize = rocjitsu::KfdProcess::kPageSize;
  constexpr size_t kAllocationSize = 3 * kPageSize;
  uint8_t *reservation = static_cast<uint8_t *>(
      ::mmap(nullptr, kAllocationSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(reservation, MAP_FAILED);
  *reinterpret_cast<uint32_t *>(reservation) = 0x11111111u;
  *reinterpret_cast<uint32_t *>(reservation + kPageSize) = 0x22222222u;
  *reinterpret_cast<uint32_t *>(reservation + 2 * kPageSize) = 0x33333333u;

  kfd_ioctl_alloc_memory_of_gpu_args alloc{};
  alloc.va_addr = reinterpret_cast<uint64_t>(reservation);
  alloc.size = kAllocationSize;
  alloc.gpu_id = t.driver()->gpu_id();
  alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc), 0);

  ASSERT_EQ(t.driver()->mmap(reservation, kPageSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                             static_cast<off_t>(alloc.mmap_offset)),
            reservation);
  ASSERT_EQ(t.driver()->mmap(reservation, kAllocationSize, PROT_READ | PROT_WRITE,
                             MAP_SHARED | MAP_FIXED, static_cast<off_t>(alloc.mmap_offset)),
            reservation);

  uint32_t gpu_id = alloc.gpu_id;
  kfd_ioctl_map_memory_to_gpu_args map{};
  map.handle = alloc.handle;
  map.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
  map.n_devices = 1;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map), 0);
  EXPECT_EQ(memory->read32(alloc.va_addr, process_id), 0x11111111u);
  EXPECT_EQ(memory->read32(alloc.va_addr + kPageSize, process_id), 0x22222222u);
  EXPECT_EQ(memory->read32(alloc.va_addr + 2 * kPageSize, process_id), 0x33333333u);

  ASSERT_EQ(t.driver()->munmap(reservation, kAllocationSize), 0);
  kfd_ioctl_free_memory_of_gpu_args free_args{};
  free_args.handle = alloc.handle;
  EXPECT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args), 0);
  EXPECT_EQ(t.driver()->close(), 0);
}

TEST_F(SimulatedKfdTest, GpuUsePermanentlyMakesVramBackingAuthoritative) {
  TestVM t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  rocjitsu::VirtualMachine *vm =
      dynamic_cast<rocjitsu::VirtualMachine *>(t.engine->topology().root());
  ASSERT_NE(vm, nullptr);
  rocjitsu::amdgpu::GpuMemory *memory = vm->memory();
  ASSERT_NE(memory, nullptr);
  ASSERT_GE(t.driver()->open(), 0);
  const uint32_t process_id = t.driver()->local_process_id();

  constexpr size_t kPageSize = rocjitsu::KfdProcess::kPageSize;
  uint32_t *reservation = static_cast<uint32_t *>(
      ::mmap(nullptr, kPageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(reservation, MAP_FAILED);
  *reservation = 0x11111111u;

  kfd_ioctl_alloc_memory_of_gpu_args alloc{};
  alloc.va_addr = reinterpret_cast<uint64_t>(reservation);
  alloc.size = kPageSize;
  alloc.gpu_id = t.driver()->gpu_id();
  alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc), 0);

  uint32_t gpu_id = alloc.gpu_id;
  kfd_ioctl_map_memory_to_gpu_args map{};
  map.handle = alloc.handle;
  map.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
  map.n_devices = 1;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map), 0);
  memory->write32(alloc.va_addr, 0x22222222u, process_id);

  kfd_ioctl_unmap_memory_from_gpu_args unmap{};
  unmap.handle = alloc.handle;
  unmap.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
  unmap.n_devices = 1;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmap), 0);

  void *mapping = t.driver()->mmap(reservation, kPageSize, PROT_READ | PROT_WRITE,
                                   MAP_SHARED | MAP_FIXED, static_cast<off_t>(alloc.mmap_offset));
  ASSERT_EQ(mapping, reservation);
  EXPECT_EQ(*static_cast<uint32_t *>(mapping), 0x22222222u);

  ASSERT_EQ(t.driver()->munmap(mapping, kPageSize), 0);
  kfd_ioctl_free_memory_of_gpu_args free_args{};
  free_args.handle = alloc.handle;
  EXPECT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args), 0);
  EXPECT_EQ(t.driver()->close(), 0);
}

TEST_F(SimulatedKfdTest, IpcExportUpdatesPermanentBackingMtype) {
  TestVM t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  rocjitsu::VirtualMachine *vm =
      dynamic_cast<rocjitsu::VirtualMachine *>(t.engine->topology().root());
  ASSERT_NE(vm, nullptr);
  rocjitsu::amdgpu::GpuMemory *memory = vm->memory();
  ASSERT_NE(memory, nullptr);
  ASSERT_GE(t.driver()->open(), 0);
  const uint32_t process_id = t.driver()->local_process_id();

  constexpr uint64_t kGpuVa = 0x7100000000ULL;
  kfd_ioctl_alloc_memory_of_gpu_args alloc{};
  alloc.va_addr = kGpuVa;
  alloc.size = rocjitsu::KfdProcess::kPageSize;
  alloc.gpu_id = t.driver()->gpu_id();
  alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc), 0);

  uint32_t gpu_id = alloc.gpu_id;
  kfd_ioctl_map_memory_to_gpu_args map{};
  map.handle = alloc.handle;
  map.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
  map.n_devices = 1;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map), 0);
  ASSERT_EQ(memory->pte_mtype(kGpuVa, process_id), rocjitsu::amdgpu::Mtype::RW);

  kfd_ioctl_ipc_export_handle_args export_args{};
  export_args.handle = alloc.handle;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_IPC_EXPORT_HANDLE, &export_args), 0);
  EXPECT_EQ(memory->pte_mtype(kGpuVa, process_id), rocjitsu::amdgpu::Mtype::CC);

  kfd_ioctl_free_memory_of_gpu_args free_args{};
  free_args.handle = alloc.handle;
  EXPECT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args), 0);
  EXPECT_EQ(t.driver()->close(), 0);
}

TEST_F(SimulatedKfdTest, PermanentVramBackingIsIsolatedAndReclaimedByProcess) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  auto *vm = dynamic_cast<rocjitsu::VirtualMachine *>(t.engine->topology().root());
  ASSERT_NE(vm, nullptr);
  auto *memory = vm->memory();
  ASSERT_NE(memory, nullptr);

  uint32_t first_process = t.driver()->open_process(/*client_pid=*/0);
  uint32_t second_process = t.driver()->open_process(/*client_pid=*/0);
  ASSERT_NE(first_process, second_process);

  constexpr uint64_t kGpuVa = 0x7000000000ULL;
  constexpr uint64_t kSize = rocjitsu::KfdProcess::kPageSize;
  auto allocate_vram = [&](uint32_t process_id) {
    kfd_ioctl_alloc_memory_of_gpu_args alloc{};
    alloc.va_addr = kGpuVa;
    alloc.size = kSize;
    alloc.gpu_id = t.driver()->gpu_id();
    alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
    EXPECT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc), 0);

    uint32_t gpu_id = alloc.gpu_id;
    kfd_ioctl_map_memory_to_gpu_args map{};
    map.handle = alloc.handle;
    map.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
    map.n_devices = 1;
    EXPECT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map), 0);
    EXPECT_EQ(map.n_success, 1u);
    return alloc.handle;
  };
  auto free_vram = [&](uint32_t process_id, uint64_t handle) {
    uint32_t gpu_id = t.driver()->gpu_id();
    kfd_ioctl_unmap_memory_from_gpu_args unmap{};
    unmap.handle = handle;
    unmap.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
    unmap.n_devices = 1;
    EXPECT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmap), 0);

    kfd_ioctl_free_memory_of_gpu_args free_args{};
    free_args.handle = handle;
    EXPECT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args), 0);
  };

  uint64_t first_handle = allocate_vram(first_process);
  uint64_t second_handle = allocate_vram(second_process);
  ASSERT_NE(first_handle, 0u);
  ASSERT_NE(second_handle, 0u);

  memory->write32(kGpuVa, 0x11111111u, first_process);
  memory->write32(kGpuVa, 0x22222222u, second_process);
  EXPECT_EQ(memory->read32(kGpuVa, first_process), 0x11111111u);
  EXPECT_EQ(memory->read32(kGpuVa, second_process), 0x22222222u);

  free_vram(first_process, first_handle);
  EXPECT_EQ(memory->read32(kGpuVa, second_process), 0x22222222u);

  first_handle = allocate_vram(first_process);
  ASSERT_NE(first_handle, 0u);
  EXPECT_EQ(memory->read32(kGpuVa, first_process), 0u);

  free_vram(first_process, first_handle);
  free_vram(second_process, second_handle);
  EXPECT_EQ(t.driver()->close(first_process), 0);
  EXPECT_EQ(t.driver()->close(second_process), 0);
}

TEST_F(SimulatedKfdTest, PermanentVramBackingDoesNotPrefaultLargeAllocations) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  ASSERT_GE(t.driver()->open(), 0);
  uint32_t process_id = t.driver()->local_process_id();
  auto process = t.driver()->find_process(process_id);
  ASSERT_NE(process, nullptr);

  constexpr uint64_t kGpuVa = 0x6000000000ULL;
  constexpr uint64_t kAllocationSize = 1ULL << 30;
  kfd_ioctl_alloc_memory_of_gpu_args alloc{};
  alloc.va_addr = kGpuVa;
  alloc.size = kAllocationSize;
  alloc.gpu_id = t.driver()->gpu_id();
  alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc), 0);

  {
    std::lock_guard<std::mutex> lock(process->alloc_mutex_);
    auto allocation = process->allocations_.find(alloc.handle);
    ASSERT_NE(allocation, process->allocations_.end());
    ASSERT_TRUE(allocation->second.permanent_backing);
    ASSERT_EQ(allocation->second.backing_size, kAllocationSize);

    const size_t page_count = kAllocationSize / rocjitsu::KfdProcess::kPageSize;
    std::vector<uint8_t> residency(page_count);
    ASSERT_EQ(
        ::mincore(allocation->second.host_ptr, allocation->second.backing_size, residency.data()),
        0);
    size_t resident_pages = 0;
    for (uint8_t state : residency)
      resident_pages += state & 1;
    EXPECT_EQ(resident_pages, 0u);
  }

  kfd_ioctl_free_memory_of_gpu_args free_args{};
  free_args.handle = alloc.handle;
  EXPECT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args), 0);
  EXPECT_EQ(t.driver()->close(), 0);
}

TEST_F(SimulatedKfdTest, LargePermanentBackingUsesOneRangeInsteadOfPerPageEntries) {
  rocjitsu::KfdProcess process(/*process_id=*/9);
  constexpr uint64_t kBaseVa = 0x8000000000ULL;
  constexpr uint64_t kLargeAllocationSize = 1ULL << 40;
  constexpr uint64_t kPage = rocjitsu::KfdProcess::kPageSize;
  constexpr uintptr_t kHostBase = 0x1000000000ULL;

  {
    std::lock_guard<std::mutex> lock(process.alloc_mutex_);
    rocjitsu::KfdProcess::GpuAllocation allocation{};
    allocation.handle = 1;
    allocation.gpu_va = kBaseVa;
    allocation.size = kLargeAllocationSize;
    allocation.backing_size = kLargeAllocationSize;
    allocation.host_ptr = reinterpret_cast<void *>(kHostBase);
    allocation.permanent_backing = true;
    allocation.mapped_to_gpu = true;
    process.allocations_[allocation.handle] = allocation;
    process.refresh_backing_ranges_locked();
  }

  auto expect_resolved = [&](uint64_t gpu_va) {
    auto backing = process.resolve_backing(gpu_va, kPage);
    ASSERT_TRUE(backing.has_value());
    ASSERT_TRUE(backing->gpu_accessible);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(backing->address), kHostBase + (gpu_va - kBaseVa));
    EXPECT_EQ(backing->range_size, kLargeAllocationSize);
  };
  expect_resolved(kBaseVa);
  expect_resolved(kBaseVa + (kLargeAllocationSize / 2));
  expect_resolved(kBaseVa + kLargeAllocationSize - kPage);
  EXPECT_FALSE(process.resolve_backing(kBaseVa - kPage, kPage).has_value());
  EXPECT_FALSE(process.resolve_backing(kBaseVa + kLargeAllocationSize, kPage).has_value());
  EXPECT_TRUE(process.page_table_.empty());
}

TEST_F(SimulatedKfdTest, GuestDiscoveryOpenIsReleasedOnLastClose) {
  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create(CONFIG_PATH.c_str(), RJ_VM_MODE_LOCAL, &raw_vm), ROCJITSU_STATUS_SUCCESS);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);
  ASSERT_NE(vm, nullptr);
  ASSERT_NE(vm->vm, nullptr);
  auto *execution_driver = vm->vm->driver();
  ASSERT_NE(execution_driver, nullptr);
  ASSERT_EQ(execution_driver->local_open_ref_count(), 1u)
      << "RJ_VM_MODE_LOCAL must provide the bootstrap open adopted by GuestKfd";

  rocjitsu::config::DbtGuestConfig config;
  config.enabled = true;
  config.guest_isa = "gfx950";
  config.host.isa = "gfx950";
  config.host.gpu_id = execution_driver->gpu_id();
  config.host.backend = rocjitsu::config::DbtExecutionBackend::Simulator;
  config.guest_device = vm->loaded.device;
  config.guest_device.gpu_id += 1;
  config.guest_device.drm_render_minor += 1;

  rocjitsu::GuestKfd guest(std::move(config), execution_driver);
  for (int cycle = 0; cycle < 2; ++cycle) {
    ASSERT_TRUE(guest.prepare_for_discovery());

    const int app_fd = guest.open();
    ASSERT_GE(app_fd, 0);
    EXPECT_EQ(execution_driver->local_open_ref_count(), 1u)
        << "application open must reuse the discovery-owned reference";

    EXPECT_EQ(::close(app_fd), 0);
    EXPECT_EQ(guest.close(), 0);
    EXPECT_EQ(execution_driver->local_open_ref_count(), 0u)
        << "last guest close must release the simulated process";
  }
}

TEST_F(SimulatedKfdTest, GuestOpenSurvivesExecutionPrimaryOverwrite) {
  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create(CONFIG_PATH.c_str(), RJ_VM_MODE_LOCAL, &raw_vm), ROCJITSU_STATUS_SUCCESS);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);
  ASSERT_NE(vm, nullptr);
  ASSERT_NE(vm->vm, nullptr);
  auto *execution_driver = vm->vm->driver();
  ASSERT_NE(execution_driver, nullptr);
  ASSERT_EQ(execution_driver->local_open_ref_count(), 1u);

  rocjitsu::config::DbtGuestConfig config;
  config.enabled = true;
  config.guest_isa = "gfx950";
  config.host.isa = "gfx950";
  config.host.gpu_id = execution_driver->gpu_id();
  config.host.backend = rocjitsu::config::DbtExecutionBackend::Simulator;
  config.guest_device = vm->loaded.device;
  config.guest_device.gpu_id += 1;
  config.guest_device.drm_render_minor += 1;

  rocjitsu::GuestKfd guest(std::move(config), execution_driver);
  ASSERT_TRUE(guest.prepare_for_discovery());
  const int app_fd = guest.open();
  ASSERT_GE(app_fd, 0);
  ASSERT_EQ(execution_driver->local_open_ref_count(), 1u);
  const uint32_t process_id = execution_driver->local_process_id();

  const int hidden_fd = execution_driver->fd();
  ASSERT_GE(hidden_fd, 0);
  int pipefd[2];
  ASSERT_EQ(::pipe(pipefd), 0);
  ASSERT_EQ(::dup2(pipefd[0], hidden_fd), hidden_fd);
  ASSERT_EQ(::close(pipefd[0]), 0);

  EXPECT_EQ(guest.invalidate_primary_fd(hidden_fd),
            rocjitsu::LinuxKfd::PrimaryInvalidation::kClearedKeepRefs);
  EXPECT_EQ(execution_driver->fd(), -1);
  EXPECT_EQ(execution_driver->local_open_ref_count(), 1u)
      << "hidden-fd overwrite must not release GuestKfd's backend open";
  EXPECT_EQ(execution_driver->local_process_id(), process_id);

  kfd_ioctl_get_version_args version{};
  EXPECT_EQ(guest.ioctl(AMDKFD_IOC_GET_VERSION, &version), 0)
      << "the surviving app open must keep the simulated process usable";

  const int reopened_fd = guest.open();
  ASSERT_GE(reopened_fd, 0);
  EXPECT_EQ(execution_driver->local_open_ref_count(), 1u)
      << "re-minting the simulator primary must not leak another backend open";
  EXPECT_EQ(execution_driver->local_process_id(), process_id);

  EXPECT_EQ(::close(reopened_fd), 0);
  EXPECT_EQ(guest.close(), 0);
  EXPECT_EQ(execution_driver->local_open_ref_count(), 1u);
  EXPECT_EQ(::close(app_fd), 0);
  EXPECT_EQ(guest.close(), 0);
  EXPECT_EQ(execution_driver->local_open_ref_count(), 0u);

  EXPECT_EQ(::close(hidden_fd), 0);
  EXPECT_EQ(::close(pipefd[1]), 0);
}

TEST_F(SimulatedKfdTest, TopologyDirectoryExists) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);

  const auto &path = t.driver()->topology().path();
  EXPECT_FALSE(path.empty());
  EXPECT_TRUE(std::filesystem::exists(path));
  EXPECT_TRUE(std::filesystem::exists(path + "/generation_id"));
  EXPECT_TRUE(std::filesystem::exists(path + "/nodes/0/properties"));
  EXPECT_TRUE(std::filesystem::exists(path + "/nodes/1/properties"));
}

// Regression test for the close()-vs-in-flight-ioctl teardown race. ioctl()
// snapshots the KfdProcess shared_ptr WITHOUT retaining an open reference, so a
// concurrent close() can erase and tear down the process while other threads are
// dispatching ioctls against the same process id. The hardening (op_mutex_ held
// across all teardown + an is_closing() guard taken right after op_mutex_ in
// dispatch_ioctl) must ensure: (a) no crash / use-after-free, and (b) an ioctl
// that loses the race returns a clean error (-ESRCH) instead of operating on a
// dismantled process. This drives that race in-process, many times, under TSan/
// ASan in the sanitizer CI job.
TEST_F(SimulatedKfdTest, ConcurrentIoctlAndCloseIsRaceFree) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  auto *drv = t.driver();

  constexpr int kRounds = 200;
  constexpr int kIoctlThreads = 4;

  for (int round = 0; round < kRounds; ++round) {
    uint32_t pid = drv->open_process(/*client_pid=*/0);
    ASSERT_NE(pid, 0u);

    const uint32_t gpu_id = drv->gpu_id();
    std::atomic<bool> go{false};
    std::atomic<int> bad{0};
    std::vector<std::thread> workers;
    workers.reserve(kIoctlThreads);
    for (int i = 0; i < kIoctlThreads; ++i) {
      workers.emplace_back([&, pid] {
        while (!go.load(std::memory_order_acquire))
          ;
        // Accepted outcomes for every ioctl below: success (process still live),
        // or a clean -ESRCH/-EINVAL once close() has torn it down. Any other value
        // (or a crash / ASAN-TSAN report) means teardown overlapped a live ioctl.
        auto ok = [](int rc) { return rc == 0 || rc == -ESRCH || rc == -EINVAL; };
        for (int k = 0; k < 50; ++k) {
          // Stateless routing probe.
          kfd_ioctl_get_version_args ver{};
          if (!ok(drv->ioctl(pid, AMDKFD_IOC_GET_VERSION, &ver)))
            bad.fetch_add(1, std::memory_order_relaxed);

          // State-touching ioctls that read/write alloc_mutex_-guarded
          // allocations_, so close()'s teardown can actually overlap a handler
          // dereferencing per-process state (not just the op_mutex_/is_closing
          // gate). alloc then free the returned handle.
          kfd_ioctl_alloc_memory_of_gpu_args alloc{};
          alloc.va_addr = 0x100000000ULL + static_cast<uint64_t>(k) * 0x1000ULL;
          alloc.size = 0x1000;
          alloc.gpu_id = gpu_id;
          alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
          int arc = drv->ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc);
          if (!ok(arc))
            bad.fetch_add(1, std::memory_order_relaxed);
          if (arc == 0) {
            kfd_ioctl_free_memory_of_gpu_args freed{};
            freed.handle = alloc.handle;
            if (!ok(drv->ioctl(pid, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &freed)))
              bad.fetch_add(1, std::memory_order_relaxed);
          }
        }
      });
    }

    go.store(true, std::memory_order_release);
    // Race close() against the in-flight ioctls.
    int cret = drv->close(pid);
    EXPECT_EQ(cret, 0);
    for (auto &w : workers)
      w.join();
    EXPECT_EQ(bad.load(), 0) << "round " << round << ": ioctl saw invalid state during close";
  }
}

} // namespace
