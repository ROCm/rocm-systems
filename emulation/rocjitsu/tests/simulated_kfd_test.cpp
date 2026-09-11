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
#include "rocjitsu/kmd/linux/host_mapping_lock.h"
#include "simdojo/sim/simulation.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <shared_mutex>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace {

const std::string CONFIG_PATH = std::string(CONFIG_DIR) + "/gfx950_mi355x.json";

struct TestVM {
  rocjitsu::config::LoadedConfig loaded;
  std::unique_ptr<simdojo::SimulationEngine> engine;

  rocjitsu::SimulatedKfd *driver() {
    auto *vm = dynamic_cast<rocjitsu::VirtualMachine *>(engine->topology().root());
    return vm ? vm->driver() : nullptr;
  }

  rocjitsu::amdgpu::GpuMemory *memory(uint32_t index = 0) {
    auto *vm = dynamic_cast<rocjitsu::VirtualMachine *>(engine->topology().root());
    auto *soc = vm ? vm->soc(index) : nullptr;
    return soc ? soc->memory() : nullptr;
  }

  uint32_t num_socs() {
    auto *vm = dynamic_cast<rocjitsu::VirtualMachine *>(engine->topology().root());
    return vm ? vm->num_socs() : 0;
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

TEST_F(SimulatedKfdTest, DoorbellClientRemapKeepsDriverAliasStableWhenOffsetRecycles) {
  auto t = create_test_vm();
  auto *driver = t.driver();
  ASSERT_NE(driver, nullptr);
  ASSERT_GE(driver->open(), 0);

  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(host_page_size, 0);
  const size_t doorbell_page_size = static_cast<size_t>(host_page_size);
  const off_t doorbell_mmap_offset = static_cast<off_t>(
      rocjitsu::KFD_MMAP_TYPE_DOORBELL | rocjitsu::kfd_mmap_gpu_id(driver->gpu_id()));

  void *client_page = driver->mmap(nullptr, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                                   doorbell_mmap_offset);
  ASSERT_NE(client_page, MAP_FAILED);

  alignas(4096) std::array<uint8_t, 4096> ring{};
  alignas(uint64_t) uint64_t read_pointer = 0;
  alignas(uint64_t) uint64_t write_pointer = 0;

  kfd_ioctl_create_queue_args first_queue{};
  first_queue.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  first_queue.ring_size = ring.size();
  first_queue.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointer);
  first_queue.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointer);
  first_queue.gpu_id = driver->gpu_id();
  first_queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE;
  ASSERT_EQ(driver->ioctl(AMDKFD_IOC_CREATE_QUEUE, &first_queue), 0);

  kfd_ioctl_destroy_queue_args destroy_first{};
  destroy_first.queue_id = first_queue.queue_id;
  ASSERT_EQ(driver->ioctl(AMDKFD_IOC_DESTROY_QUEUE, &destroy_first), 0);

  // Leave a non-sentinel value in the freed slot, then exercise the driver's
  // re-mmap path. The backing and monitor alias must survive this client-view
  // replacement rather than being recreated or orphaned.
  auto *doorbell_slot = reinterpret_cast<uint64_t *>(
      static_cast<char *>(client_page) + (first_queue.doorbell_offset % doorbell_page_size));
  std::atomic_ref<uint64_t>(*doorbell_slot).store(0, std::memory_order_release);
  ASSERT_EQ(driver->mmap(client_page, doorbell_page_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_FIXED, doorbell_mmap_offset),
            client_page);

  // Replace the client address with an inaccessible anonymous mapping, exactly
  // as a runtime remap can replace the view returned by KFD. Driver-side queue
  // initialization must never dereference this address.
  ASSERT_EQ(::mmap(client_page, doorbell_page_size, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0),
            client_page);

  kfd_ioctl_create_queue_args second_queue{};
  second_queue.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  second_queue.ring_size = ring.size();
  second_queue.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointer);
  second_queue.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointer);
  second_queue.gpu_id = driver->gpu_id();
  second_queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE;
  ASSERT_EQ(driver->ioctl(AMDKFD_IOC_CREATE_QUEUE, &second_queue), 0);
  EXPECT_EQ(second_queue.doorbell_offset, first_queue.doorbell_offset);

  // Restore the client view and verify that recycled-slot initialization went
  // through the stable alias into the same canonical backing.
  ASSERT_EQ(driver->mmap(client_page, doorbell_page_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_FIXED, doorbell_mmap_offset),
            client_page);
  doorbell_slot = reinterpret_cast<uint64_t *>(static_cast<char *>(client_page) +
                                               (second_queue.doorbell_offset % doorbell_page_size));
  EXPECT_EQ(std::atomic_ref<uint64_t>(*doorbell_slot).load(std::memory_order_acquire),
            ~uint64_t(0));

  kfd_ioctl_destroy_queue_args destroy_second{};
  destroy_second.queue_id = second_queue.queue_id;
  EXPECT_EQ(driver->ioctl(AMDKFD_IOC_DESTROY_QUEUE, &destroy_second), 0);
  EXPECT_EQ(driver->close(), 0);
}

TEST_F(SimulatedKfdTest, AdditionalDoorbellMmapKeepsEarlierClientViewLive) {
  auto t = create_test_vm();
  auto *driver = t.driver();
  ASSERT_NE(driver, nullptr);
  ASSERT_GE(driver->open(), 0);

  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(host_page_size, 0);
  const size_t doorbell_page_size = static_cast<size_t>(host_page_size);
  const off_t doorbell_mmap_offset = static_cast<off_t>(
      rocjitsu::KFD_MMAP_TYPE_DOORBELL | rocjitsu::kfd_mmap_gpu_id(driver->gpu_id()));

  void *first = driver->mmap(nullptr, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                             doorbell_mmap_offset);
  ASSERT_NE(first, MAP_FAILED);
  void *second = driver->mmap(nullptr, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                              doorbell_mmap_offset);
  ASSERT_NE(second, MAP_FAILED);
  ASSERT_NE(second, first);

  auto *first_slot = static_cast<uint64_t *>(first);
  auto *second_slot = static_cast<uint64_t *>(second);
  std::atomic_ref<uint64_t>(*first_slot).store(0x123456789abcdef0ULL, std::memory_order_release);
  EXPECT_EQ(std::atomic_ref<uint64_t>(*second_slot).load(std::memory_order_acquire),
            0x123456789abcdef0ULL);
  EXPECT_EQ(driver->close(), 0);
}

/// @brief Block until a mapping change is genuinely stuck behind the lock.
/// @details Sleeping and observing that nothing happened proves nothing: an
/// implementation that takes no lock passes that check whenever the machine is
/// busy enough to leave the worker unscheduled. Waiting for the blocked-writer
/// count instead makes the block a fact, and an implementation that never
/// takes the lock fails here rather than passing by accident.
[[nodiscard]] bool wait_for_blocked_mapping_change() {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (rocjitsu::host_mapping_lock().blocked_writers() == 0) {
    if (std::chrono::steady_clock::now() > deadline)
      return false;
    std::this_thread::yield();
  }
  return true;
}

/// @brief Backing an allocation must not replace a page an access is holding.
/// @details A MAP_FIXED replacement over a client range is the same hazard as
/// the application's own mmap: an emulated atomic establishes that a page is
/// writable and then stores through it, so a mapping change in between stores
/// into whatever replaced the page. The interposer takes this lock around the
/// application's mapping calls; the driver's own MAP_FIXED paths are reached
/// without passing through it, so they have to take it themselves. Standing in
/// for an in-flight atomic by holding the lock shared here.
TEST_F(SimulatedKfdTest, FixedAllocationMmapWaitsForInFlightHostAccesses) {
  auto t = create_test_vm();
  auto *driver = t.driver();
  ASSERT_NE(driver, nullptr);
  ASSERT_GE(driver->open(), 0);

  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(host_page_size, 0);
  const size_t page_size = static_cast<size_t>(host_page_size);

  void *target =
      ::mmap(nullptr, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(target, MAP_FAILED);

  kfd_ioctl_alloc_memory_of_gpu_args alloc{};
  alloc.va_addr = reinterpret_cast<uint64_t>(target);
  alloc.size = page_size;
  alloc.gpu_id = driver->gpu_id();
  alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(driver->ioctl(driver->local_process_id(), AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc), 0);

  std::atomic<bool> completed{false};
  auto held = rocjitsu::host_mapping_lock().lock_shared();
  std::thread mapper([&] {
    driver->mmap(target, page_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                 static_cast<off_t>(alloc.mmap_offset));
    completed.store(true, std::memory_order_release);
  });

  const bool blocked = wait_for_blocked_mapping_change();
  EXPECT_TRUE(blocked) << "the driver never took the mapping lock for this operation";
  EXPECT_FALSE(!blocked || completed.load(std::memory_order_acquire))
      << "the driver replaced a mapping while an access held a pointer into it";

  held.unlock();
  mapper.join();
  EXPECT_TRUE(completed.load(std::memory_order_acquire));

  EXPECT_EQ(driver->close(), 0);
  ::munmap(target, page_size);
}

/// @brief Tearing an allocation down must not pull a page out from under one.
/// @details Sharper than the replacement case: teardown drops the GPU page-table
/// entry first, so a concurrent access misses the page table, falls through to
/// the identity path, and validates the very host page this is about to remove.
/// Routed through the driver rather than the interposer, so the interposer's own
/// lock never sees it.
TEST_F(SimulatedKfdTest, AllocationMunmapWaitsForInFlightHostAccesses) {
  auto t = create_test_vm();
  auto *driver = t.driver();
  ASSERT_NE(driver, nullptr);
  ASSERT_GE(driver->open(), 0);

  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(host_page_size, 0);
  const size_t page_size = static_cast<size_t>(host_page_size);

  void *target =
      ::mmap(nullptr, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(target, MAP_FAILED);

  kfd_ioctl_alloc_memory_of_gpu_args alloc{};
  alloc.va_addr = reinterpret_cast<uint64_t>(target);
  alloc.size = page_size;
  alloc.gpu_id = driver->gpu_id();
  alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(driver->ioctl(driver->local_process_id(), AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc), 0);
  ASSERT_EQ(driver->mmap(target, page_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                         static_cast<off_t>(alloc.mmap_offset)),
            target);

  std::atomic<bool> completed{false};
  auto held = rocjitsu::host_mapping_lock().lock_shared();
  std::thread unmapper([&] {
    driver->munmap(target, page_size);
    completed.store(true, std::memory_order_release);
  });

  const bool blocked = wait_for_blocked_mapping_change();
  EXPECT_TRUE(blocked) << "the driver never took the mapping lock for this operation";
  EXPECT_FALSE(!blocked || completed.load(std::memory_order_acquire))
      << "the driver withdrew a mapping while an access held a pointer into it";

  held.unlock();
  unmapper.join();
  EXPECT_TRUE(completed.load(std::memory_order_acquire));

  EXPECT_EQ(driver->close(), 0);
}

/// @brief A doorbell MAP_FIXED must not replace a page an access is holding.
/// @details The interposer hands a MAP_FIXED mmap straight to the driver, so
/// this replacement never passes through the interposer's lock either.
TEST_F(SimulatedKfdTest, FixedDoorbellMmapWaitsForInFlightHostAccesses) {
  auto t = create_test_vm();
  auto *driver = t.driver();
  ASSERT_NE(driver, nullptr);
  ASSERT_GE(driver->open(), 0);

  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(host_page_size, 0);
  const size_t doorbell_page_size = static_cast<size_t>(host_page_size);
  const off_t doorbell_mmap_offset = static_cast<off_t>(
      rocjitsu::KFD_MMAP_TYPE_DOORBELL | rocjitsu::kfd_mmap_gpu_id(driver->gpu_id()));

  void *target = ::mmap(nullptr, doorbell_page_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(target, MAP_FAILED);
  ASSERT_EQ(::munmap(target, doorbell_page_size), 0);

  std::atomic<bool> completed{false};
  auto held = rocjitsu::host_mapping_lock().lock_shared();
  std::thread mapper([&] {
    driver->force_next_doorbell_monitor_mmap_at_for_testing(target);
    driver->mmap(target, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                 doorbell_mmap_offset);
    completed.store(true, std::memory_order_release);
  });

  const bool blocked = wait_for_blocked_mapping_change();
  EXPECT_TRUE(blocked) << "the driver never took the mapping lock for this operation";
  EXPECT_FALSE(!blocked || completed.load(std::memory_order_acquire))
      << "the driver replaced a doorbell page while an access held a pointer into it";

  held.unlock();
  mapper.join();
  EXPECT_TRUE(completed.load(std::memory_order_acquire));

  EXPECT_EQ(driver->close(), 0);
}

TEST_F(SimulatedKfdTest, FirstFixedDoorbellMmapKeepsMonitorAliasPrivate) {
  auto t = create_test_vm();
  auto *driver = t.driver();
  ASSERT_NE(driver, nullptr);
  ASSERT_GE(driver->open(), 0);

  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(host_page_size, 0);
  const size_t doorbell_page_size = static_cast<size_t>(host_page_size);
  const off_t doorbell_mmap_offset = static_cast<off_t>(
      rocjitsu::KFD_MMAP_TYPE_DOORBELL | rocjitsu::kfd_mmap_gpu_id(driver->gpu_id()));

  void *target = ::mmap(nullptr, doorbell_page_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(target, MAP_FAILED);
  ASSERT_EQ(::munmap(target, doorbell_page_size), 0);

  driver->force_next_doorbell_monitor_mmap_at_for_testing(target);
  ASSERT_EQ(driver->mmap(target, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                         doorbell_mmap_offset),
            target);
  auto process = driver->find_process(driver->local_process_id());
  ASSERT_NE(process, nullptr);
  void *monitor = process->gpu(0).doorbell_monitor_page;
  ASSERT_NE(monitor, nullptr);
  EXPECT_NE(monitor, target);

  EXPECT_EQ(driver->mmap(target, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                         doorbell_mmap_offset),
            target);
  EXPECT_EQ(process->gpu(0).doorbell_monitor_page, monitor);
  EXPECT_EQ(driver->close(), 0);
}

TEST_F(SimulatedKfdTest, FailedMonitorMmapPreservesFixedTarget) {
  auto t = create_test_vm();
  auto *driver = t.driver();
  ASSERT_NE(driver, nullptr);
  ASSERT_GE(driver->open(), 0);

  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(host_page_size, 0);
  const size_t doorbell_page_size = static_cast<size_t>(host_page_size);
  const off_t doorbell_mmap_offset = static_cast<off_t>(
      rocjitsu::KFD_MMAP_TYPE_DOORBELL | rocjitsu::kfd_mmap_gpu_id(driver->gpu_id()));

  void *target = ::mmap(nullptr, doorbell_page_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(target, MAP_FAILED);
  auto *bytes = static_cast<uint8_t *>(target);
  bytes[0] = 0x5a;
  bytes[doorbell_page_size - 1] = 0xa5;

  driver->fail_next_doorbell_monitor_mmap_for_testing();
  errno = 0;
  EXPECT_EQ(driver->mmap(target, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                         doorbell_mmap_offset),
            MAP_FAILED);
  EXPECT_EQ(errno, ENOMEM);

  unsigned char residency = 0;
  EXPECT_EQ(::mincore(target, doorbell_page_size, &residency), 0);
  EXPECT_EQ(bytes[0], 0x5a);
  EXPECT_EQ(bytes[doorbell_page_size - 1], 0xa5);
  bytes[0] = 0x3c;

  auto process = driver->find_process(driver->local_process_id());
  ASSERT_NE(process, nullptr);
  EXPECT_EQ(process->gpu(0).doorbell_monitor_page, nullptr);
  EXPECT_EQ(process->gpu(0).doorbell_memfd, -1);
  EXPECT_TRUE(process->gpu(0).doorbell_views.empty());

  EXPECT_EQ(driver->close(), 0);
  EXPECT_EQ(bytes[0], 0x3c);
  EXPECT_EQ(::munmap(target, doorbell_page_size), 0);
}

TEST_F(SimulatedKfdTest, DoorbellMonitorRejectsOverlappingMunmapWhileQueueIsLive) {
  auto t = create_test_vm();
  auto *driver = t.driver();
  ASSERT_NE(driver, nullptr);
  ASSERT_GE(driver->open(), 0);

  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(host_page_size, 0);
  const size_t doorbell_page_size = 2 * static_cast<size_t>(host_page_size);
  const off_t doorbell_mmap_offset = static_cast<off_t>(
      rocjitsu::KFD_MMAP_TYPE_DOORBELL | rocjitsu::kfd_mmap_gpu_id(driver->gpu_id()));
  ASSERT_NE(driver->mmap(nullptr, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                         doorbell_mmap_offset),
            MAP_FAILED);

  alignas(4096) std::array<uint8_t, 4096> ring{};
  alignas(uint64_t) uint64_t read_pointer = 0;
  alignas(uint64_t) uint64_t write_pointer = 0;
  kfd_ioctl_create_queue_args queue{};
  queue.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  queue.ring_size = ring.size();
  queue.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointer);
  queue.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointer);
  queue.gpu_id = driver->gpu_id();
  queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE;
  ASSERT_EQ(driver->ioctl(AMDKFD_IOC_CREATE_QUEUE, &queue), 0);

  auto process = driver->find_process(driver->local_process_id());
  ASSERT_NE(process, nullptr);
  void *monitor = process->gpu(0).doorbell_monitor_page;
  ASSERT_NE(monitor, nullptr);

  errno = 0;
  EXPECT_EQ(driver->munmap(monitor, doorbell_page_size), -1);
  EXPECT_EQ(errno, EPERM);

  errno = 0;
  EXPECT_EQ(driver->munmap(static_cast<char *>(monitor) + host_page_size,
                           static_cast<size_t>(host_page_size)),
            -1);
  EXPECT_EQ(errno, EPERM);

  const auto containing_base = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(monitor) -
                                                        static_cast<size_t>(host_page_size));
  errno = 0;
  EXPECT_EQ(
      driver->munmap(containing_base, doorbell_page_size + 2 * static_cast<size_t>(host_page_size)),
      -1);
  EXPECT_EQ(errno, EPERM);
  EXPECT_EQ(process->gpu(0).doorbell_monitor_page, monitor);
  EXPECT_TRUE(driver->is_doorbell_range(monitor, doorbell_page_size));

  kfd_ioctl_destroy_queue_args destroy{};
  destroy.queue_id = queue.queue_id;
  EXPECT_EQ(driver->ioctl(AMDKFD_IOC_DESTROY_QUEUE, &destroy), 0);
  EXPECT_EQ(driver->close(), 0);
}

TEST_F(SimulatedKfdTest, FixedAnonymousMmapRetiresOnlyReplacedDoorbellView) {
  auto t = create_test_vm();
  auto *driver = t.driver();
  ASSERT_NE(driver, nullptr);
  ASSERT_GE(driver->open(), 0);

  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(host_page_size, 0);
  const size_t doorbell_page_size = static_cast<size_t>(host_page_size);
  const off_t doorbell_mmap_offset = static_cast<off_t>(
      rocjitsu::KFD_MMAP_TYPE_DOORBELL | rocjitsu::kfd_mmap_gpu_id(driver->gpu_id()));

  void *first = driver->mmap(nullptr, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                             doorbell_mmap_offset);
  ASSERT_NE(first, MAP_FAILED);
  void *second = driver->mmap(nullptr, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                              doorbell_mmap_offset);
  ASSERT_NE(second, MAP_FAILED);
  ASSERT_NE(second, first);

  errno = 0;
  EXPECT_EQ(driver->mmap_replacing_client_doorbell_views(
                first, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, -1, 0),
            MAP_FAILED);
  EXPECT_EQ(errno, EBADF);
  EXPECT_TRUE(driver->is_doorbell_range(first, doorbell_page_size));
  EXPECT_TRUE(driver->is_doorbell_range(second, doorbell_page_size));

  ASSERT_EQ(driver->mmap_replacing_client_doorbell_views(
                first, doorbell_page_size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0),
            first);
  EXPECT_FALSE(driver->is_doorbell_range(first, doorbell_page_size));
  EXPECT_TRUE(driver->is_doorbell_range(second, doorbell_page_size));
  EXPECT_EQ(::mprotect(first, doorbell_page_size, PROT_READ), 0);
  EXPECT_EQ(::mprotect(first, doorbell_page_size, PROT_READ | PROT_WRITE), 0);
  EXPECT_EQ(driver->munmap(first, doorbell_page_size), -ENOENT);
  EXPECT_EQ(::munmap(first, doorbell_page_size), 0);

  ASSERT_EQ(driver->mmap_replacing_client_doorbell_views(
                first, doorbell_page_size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0),
            first);
  *static_cast<uint8_t *>(first) = 0x5a;
  EXPECT_EQ(driver->close(), 0);

  unsigned char residency = 0;
  ASSERT_EQ(::mincore(first, doorbell_page_size, &residency), 0)
      << "closing KFD must not unmap the anonymous replacement";
  EXPECT_EQ(*static_cast<uint8_t *>(first), 0x5a);
  EXPECT_EQ(::munmap(first, doorbell_page_size), 0);
}

TEST_F(SimulatedKfdTest, PartialFixedMmapKeepsMultiPageDoorbellView) {
  auto t = create_test_vm();
  auto *driver = t.driver();
  ASSERT_NE(driver, nullptr);
  ASSERT_GE(driver->open(), 0);

  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(host_page_size, 0);
  const size_t doorbell_page_size = 2 * static_cast<size_t>(host_page_size);
  const off_t doorbell_mmap_offset = static_cast<off_t>(
      rocjitsu::KFD_MMAP_TYPE_DOORBELL | rocjitsu::kfd_mmap_gpu_id(driver->gpu_id()));
  void *client = driver->mmap(nullptr, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                              doorbell_mmap_offset);
  ASSERT_NE(client, MAP_FAILED);

  auto *first_page = static_cast<uint8_t *>(client);
  auto *second_page = first_page + host_page_size;
  *first_page = 0x5a;
  *second_page = 0xa5;

  errno = 0;
  EXPECT_EQ(driver->mmap_replacing_client_doorbell_views(
                second_page, static_cast<size_t>(host_page_size), PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0),
            MAP_FAILED);
  EXPECT_EQ(errno, EINVAL);
  EXPECT_TRUE(driver->is_doorbell_range(client, doorbell_page_size));
  EXPECT_EQ(*first_page, 0x5a);
  EXPECT_EQ(*second_page, 0xa5);
  EXPECT_EQ(driver->close(), 0);
}

TEST_F(SimulatedKfdTest, GuestFixedMmapRetiresSimulatedDoorbellView) {
  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create(CONFIG_PATH.c_str(), RJ_VM_MODE_LOCAL, &raw_vm), ROCJITSU_STATUS_SUCCESS);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);
  ASSERT_NE(vm, nullptr);
  ASSERT_NE(vm->vm, nullptr);
  auto *execution_driver = vm->vm->driver();
  ASSERT_NE(execution_driver, nullptr);

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

  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(host_page_size, 0);
  const size_t doorbell_page_size = static_cast<size_t>(host_page_size);
  const off_t doorbell_mmap_offset = static_cast<off_t>(
      rocjitsu::KFD_MMAP_TYPE_DOORBELL | rocjitsu::kfd_mmap_gpu_id(execution_driver->gpu_id()));
  void *client = guest.mmap(nullptr, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                            doorbell_mmap_offset);
  ASSERT_NE(client, MAP_FAILED);

  ASSERT_EQ(
      guest.mmap_replacing_client_doorbell_views(client, doorbell_page_size, PROT_READ | PROT_WRITE,
                                                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0),
      client);
  EXPECT_FALSE(guest.is_doorbell_range(client, doorbell_page_size));
  EXPECT_EQ(::mprotect(client, doorbell_page_size, PROT_READ), 0);
  EXPECT_EQ(::mprotect(client, doorbell_page_size, PROT_READ | PROT_WRITE), 0);
  *static_cast<uint8_t *>(client) = 0x5a;

  EXPECT_EQ(::close(app_fd), 0);
  EXPECT_EQ(guest.close(), 0);
  unsigned char residency = 0;
  ASSERT_EQ(::mincore(client, doorbell_page_size, &residency), 0)
      << "guest close must not unmap the anonymous replacement";
  EXPECT_EQ(*static_cast<uint8_t *>(client), 0x5a);
  EXPECT_EQ(::munmap(client, doorbell_page_size), 0);
}

TEST_F(SimulatedKfdTest, ProcessAddressedDoorbellMmapPublishesCanonicalMemfd) {
  auto t = create_test_vm();
  auto *driver = t.driver();
  ASSERT_NE(driver, nullptr);
  const uint32_t process_id = driver->open_process();
  ASSERT_NE(process_id, 0u);

  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(host_page_size, 0);
  const size_t doorbell_page_size = static_cast<size_t>(host_page_size);
  const off_t doorbell_mmap_offset = static_cast<off_t>(
      rocjitsu::KFD_MMAP_TYPE_DOORBELL | rocjitsu::kfd_mmap_gpu_id(driver->gpu_id()));

  ASSERT_NE(driver->mmap(process_id, nullptr, doorbell_page_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, doorbell_mmap_offset),
            MAP_FAILED);
  EXPECT_GE(driver->get_mmap_memfd(process_id, doorbell_mmap_offset), 0)
      << "daemon fd passing must use the canonical backing created by mmap";
  EXPECT_EQ(driver->close(process_id), 0);
}

TEST_F(SimulatedKfdTest, UnknownDoorbellGpuDoesNotUseCanonicalBacking) {
  auto t = create_test_vm();
  auto *driver = t.driver();
  ASSERT_NE(driver, nullptr);
  ASSERT_GE(driver->open(), 0);

  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(host_page_size, 0);
  const size_t doorbell_page_size = static_cast<size_t>(host_page_size);
  const off_t valid_offset = static_cast<off_t>(rocjitsu::KFD_MMAP_TYPE_DOORBELL |
                                                rocjitsu::kfd_mmap_gpu_id(driver->gpu_id()));
  ASSERT_NE(
      driver->mmap(nullptr, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED, valid_offset),
      MAP_FAILED);
  ASSERT_GE(driver->get_mmap_memfd(valid_offset), 0);

  const uint32_t unknown_gpu_id = driver->gpu_id() + 1;
  const off_t unknown_offset = static_cast<off_t>(rocjitsu::KFD_MMAP_TYPE_DOORBELL |
                                                  rocjitsu::kfd_mmap_gpu_id(unknown_gpu_id));
  EXPECT_EQ(driver->get_mmap_memfd(unknown_offset), -1);
  errno = 0;
  EXPECT_EQ(
      driver->mmap(nullptr, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED, unknown_offset),
      MAP_FAILED);
  EXPECT_EQ(errno, EINVAL);
  EXPECT_EQ(driver->close(), 0);
}

TEST_F(SimulatedKfdTest, FixedDoorbellMmapCannotReplaceMonitorAlias) {
  auto t = create_test_vm();
  auto *driver = t.driver();
  ASSERT_NE(driver, nullptr);
  ASSERT_GE(driver->open(), 0);

  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(host_page_size, 0);
  const size_t doorbell_page_size = static_cast<size_t>(host_page_size);
  const off_t doorbell_mmap_offset = static_cast<off_t>(
      rocjitsu::KFD_MMAP_TYPE_DOORBELL | rocjitsu::kfd_mmap_gpu_id(driver->gpu_id()));
  ASSERT_NE(driver->mmap(nullptr, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                         doorbell_mmap_offset),
            MAP_FAILED);

  auto process = driver->find_process(driver->local_process_id());
  ASSERT_NE(process, nullptr);
  void *monitor = process->gpu(0).doorbell_monitor_page;
  ASSERT_NE(monitor, nullptr);
  EXPECT_TRUE(driver->is_doorbell_range(monitor, doorbell_page_size));

  errno = 0;
  EXPECT_EQ(driver->mmap(monitor, doorbell_page_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_FIXED, doorbell_mmap_offset),
            MAP_FAILED);
  EXPECT_EQ(errno, EINVAL);
  EXPECT_EQ(process->gpu(0).doorbell_monitor_page, monitor);
  EXPECT_EQ(driver->close(), 0);
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
  // Each open() yields a DISTINCT descriptor, as real KFD does -- duplicated from
  // the synthetic backing, never from a real device.
  EXPECT_NE(reopened_fd, app_fd);
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

// A caller-visible WAIT_EVENTS on the hardware path is served as MANY bounded ioctls.
// That is invisible to the caller until a final close lands between two slices: the
// connection the wait had already started on would be gone, and the next slice would
// fail ENODEV -- an operation revoked partway, which a native blocking ioctl is never
// subject to, since the kernel holds a file reference for the whole syscall. A lease
// held for the caller-visible duration makes that close DEFER instead.
//
// Driving the lease directly rather than through a real sliced wait is deliberate: what
// it guards is backend-independent, so the contract is provable here with no device and
// no timing race. The observable is the simulator backend's open reference, which the
// final close is what normally releases.
TEST_F(SimulatedKfdTest, FinalCloseUnderAWaitLeaseDefersUntilTheLeaseDrains) {
  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create(CONFIG_PATH.c_str(), RJ_VM_MODE_LOCAL, &raw_vm), ROCJITSU_STATUS_SUCCESS);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);
  ASSERT_NE(vm->vm, nullptr);
  auto *execution_driver = vm->vm->driver();
  ASSERT_NE(execution_driver, nullptr);

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

  ASSERT_TRUE(rocjitsu::GuestKfdTestAccess::acquire_wait_lease(guest));

  EXPECT_EQ(::close(app_fd), 0);
  EXPECT_EQ(guest.close(), 0);
  EXPECT_EQ(execution_driver->local_open_ref_count(), 1u)
      << "the final close must not tear the connection out from under a live wait";

  // Still usable, which is the point: the wait would have issued its next slice here.
  kfd_ioctl_get_version_args version{};
  EXPECT_EQ(guest.ioctl(AMDKFD_IOC_GET_VERSION, &version), 0);

  rocjitsu::GuestKfdTestAccess::release_wait_lease(guest);
  EXPECT_EQ(execution_driver->local_open_ref_count(), 0u)
      << "the draining lease must perform the close that close() handed off";

  // The ioctl above republished readiness against the connection the lease was holding
  // open, at zero open references. A drain that closed the descriptor but left that flag
  // set would wedge the driver: ensure_ready() short-circuits on it, so nothing would
  // ever reopen and every later call would find no host fd. This is what proves the
  // drain runs the WHOLE close transition rather than only the descriptor close, and it
  // has to hold for a hardware guest, which -- unlike the simulator backend -- has no
  // second open() path that would repair the state by accident.
  EXPECT_EQ(guest.fd(), -1) << "the drain must have retired the connection";
  EXPECT_TRUE(guest.prepare_for_discovery());
  EXPECT_GE(guest.fd(), 0)
      << "the drain left stale readiness behind: discovery short-circuits on the flag, "
         "so the connection is never reacquired and every later call fails ENODEV";
}

// The production path: a real guest.ioctl(WAIT_EVENTS) racing a real final close. The
// tests above drive the lease directly, so they prove what the lease does but not that
// dispatch actually takes one -- and the simulator backend used to reach the wait through
// a route that took no lease at all, so a final close released the backend, marked its
// event state closing, and the already-running wait came back EBADF.
TEST_F(SimulatedKfdTest, DispatchedWaitSurvivesAConcurrentFinalClose) {
  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create(CONFIG_PATH.c_str(), RJ_VM_MODE_LOCAL, &raw_vm), ROCJITSU_STATUS_SUCCESS);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);
  ASSERT_NE(vm->vm, nullptr);
  auto *execution_driver = vm->vm->driver();
  ASSERT_NE(execution_driver, nullptr);

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

  auto proc = execution_driver->find_process(execution_driver->local_process_id());
  ASSERT_NE(proc, nullptr);
  constexpr size_t kSlots = 64;
  std::vector<uint64_t> page(kSlots, 0);
  proc->event_state_.adopt_page(page.data(), page.size() * sizeof(uint64_t));

  kfd_ioctl_create_event_args create{};
  ASSERT_EQ(guest.ioctl(AMDKFD_IOC_CREATE_EVENT, &create), 0);

  // INDEFINITE, so the waiter's registration cannot lapse under a descheduled test thread
  // and the wait cannot quietly expire before the close -- either of which would let this
  // pass without the deferral ever being exercised. It also sharpens the close assertion:
  // a close that waited for this wait would never return at all.
  kfd_event_data event{};
  event.event_id = create.event_id;
  kfd_ioctl_wait_events_args wait_args{};
  wait_args.events_ptr = reinterpret_cast<uint64_t>(&event);
  wait_args.num_events = 1;
  wait_args.wait_for_all = 1;
  wait_args.timeout = 0xFFFFFFFFu;

  int wait_rc = 0;
  std::thread waiter([&] { wait_rc = guest.ioctl(AMDKFD_IOC_WAIT_EVENTS, &wait_args); });

  // Synchronize on the waiter being REGISTERED, not on a flag it sets before entering the
  // ioctl: a descheduled waiter would let the close land first, the wait would then be
  // correctly refused, and the race under test would never run.
  bool registered = false;
  const auto registered_by = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < registered_by) {
    if (proc->event_state_.waiter_count(create.event_id) != 0) {
      registered = true;
      break;
    }
    std::this_thread::yield();
  }

  // Close on its OWN thread, with a bounded wait for it to finish. The defect this test
  // exists to catch is a close that waits for the lease instead of handing off, and the
  // wait it would be waiting on is indefinite -- so doing the close inline would never
  // reach the wake below and would hang the whole suite rather than fail one test. The
  // controlling thread keeps an independent recovery wake in hand instead.
  std::atomic<bool> close_done{false};
  uint32_t refs_during_wait = 0;
  int app_close_rc = -1;
  int guest_close_rc = -1;
  std::thread closer;
  bool closed_before_wake = false;
  if (registered) {
    closer = std::thread([&] {
      app_close_rc = ::close(app_fd);
      guest_close_rc = guest.close();
      refs_during_wait = execution_driver->local_open_ref_count();
      close_done.store(true, std::memory_order_release);
    });
    const auto close_by = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!close_done.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < close_by)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    closed_before_wake = close_done.load(std::memory_order_acquire);
  }

  // Wake and join BEFORE any fatal assertion: a joinable std::thread destroyed on the way
  // out of a failing test terminates the binary, hiding the failure it should report.
  // This is also the recovery wake that bounds a close which blocked -- but only one that
  // blocked with the driver mutex RELEASED. A close that blocked while holding it stalls
  // this wake too, and releasing the wait's lease needs that same mutex, so nothing
  // in-process can recover. CMakeLists registers this test with its own CTest timeout for
  // exactly that case.
  guest.begin_local_shutdown();
  if (closer.joinable())
    closer.join();
  waiter.join();

  ASSERT_TRUE(registered) << "the waiter never registered";
  EXPECT_TRUE(closed_before_wake)
      << "close() blocked behind the running wait; it must hand its close off, not wait";
  EXPECT_EQ(app_close_rc, 0);
  EXPECT_EQ(guest_close_rc, 0);
  EXPECT_EQ(refs_during_wait, 1u)
      << "the final close must not release the backend under a running wait";
  EXPECT_EQ(wait_rc, 0) << "the dispatched wait was revoked mid-flight: " << wait_rc;
  EXPECT_EQ(wait_args.wait_result, static_cast<uint32_t>(KFD_IOC_WAIT_RESULT_TIMEOUT));
  EXPECT_EQ(execution_driver->local_open_ref_count(), 0u)
      << "the draining wait must perform the close that close() handed off";

  // A Driver returns the KERNEL convention, -errno; the interposer performs the single
  // conversion to libc's -1/errno. Returning a bare -1 from here would be read as -EPERM,
  // which is how an interrupted hardware wait reached the runtime as EPERM and defeated
  // its EINTR retry. Reached here with no reference left, which is the one wait failure
  // that needs no device to provoke.
  kfd_ioctl_wait_events_args orphaned = wait_args;
  EXPECT_EQ(guest.ioctl(AMDKFD_IOC_WAIT_EVENTS, &orphaned), -ENODEV)
      << "a failed wait must report -errno, not libc's -1";
}

// A reopen that lands while a deferred close is still pending ADOPTS the connection.
// Without cancelling the deferral the draining wait would then close a connection the
// new epoch had already started using -- the deferral turning into the very revocation
// it exists to prevent.
TEST_F(SimulatedKfdTest, ReopenDuringADeferredCloseCancelsTheDeferral) {
  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create(CONFIG_PATH.c_str(), RJ_VM_MODE_LOCAL, &raw_vm), ROCJITSU_STATUS_SUCCESS);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);
  ASSERT_NE(vm->vm, nullptr);
  auto *execution_driver = vm->vm->driver();
  ASSERT_NE(execution_driver, nullptr);

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

  ASSERT_TRUE(rocjitsu::GuestKfdTestAccess::acquire_wait_lease(guest));
  EXPECT_EQ(::close(app_fd), 0);
  EXPECT_EQ(guest.close(), 0);

  const int reopened_fd = guest.open();
  ASSERT_GE(reopened_fd, 0);
  EXPECT_EQ(execution_driver->local_open_ref_count(), 1u);

  rocjitsu::GuestKfdTestAccess::release_wait_lease(guest);
  EXPECT_EQ(execution_driver->local_open_ref_count(), 1u)
      << "a drained lease must not close a connection the reopen adopted";

  kfd_ioctl_get_version_args version{};
  EXPECT_EQ(guest.ioctl(AMDKFD_IOC_GET_VERSION, &version), 0)
      << "the adopted connection must still serve the new epoch";

  EXPECT_EQ(::close(reopened_fd), 0);
  EXPECT_EQ(guest.close(), 0);
  EXPECT_EQ(execution_driver->local_open_ref_count(), 0u);
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

// begin_local_shutdown() releases parked waiters so their callers drop the driver
// snapshot that keeps the object alive. It must do ONLY that: an earlier version
// also marked the driver closing and poisoned every event-page slot with
// KFD_SIGNAL_EVENT_LIMIT, which turned a live consumer's ioctls into -EBADF and
// destroyed the ages it was polling. This asserts the wake disturbs neither the
// page nor the closing flag, so a driver that survives it keeps serving.
// A GPU access that cannot be translated must reach the runtime as a memory
// violation. Hardware raises a VM fault and KFD reports it on the process's
// KFD_IOC_EVENT_MEMORY event; without that report the simulator either
// dereferences the address -- corrupting the host -- or swallows it, and the
// failure resurfaces somewhere unrelated with nothing tying it to its cause.
TEST_F(SimulatedKfdTest, UnresolvedGpuAddressRaisesMemoryExceptionEvent) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  auto *drv = t.driver();
  auto *memory = t.memory();
  ASSERT_NE(memory, nullptr);

  ASSERT_GE(drv->open(), 0);
  const uint32_t process_id = drv->local_process_id();
  auto proc = drv->find_process(process_id);
  ASSERT_NE(proc, nullptr);

  constexpr size_t kSlots = 64;
  std::vector<uint64_t> page(kSlots, 0);
  proc->event_state_.adopt_page(page.data(), page.size() * sizeof(uint64_t));

  kfd_ioctl_create_event_args create{};
  create.event_type = KFD_IOC_EVENT_MEMORY;
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_CREATE_EVENT, &create), 0);

  // A page reserved but never made accessible: what a runtime VA aperture looks
  // like, and where an address invented upstream most often lands.
  void *raw =
      mmap(nullptr, rocjitsu::KfdProcess::kPageSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(raw, MAP_FAILED);
  const uint64_t faulting_va = reinterpret_cast<uint64_t>(raw) + 0x40;

  memory->read32(faulting_va, process_id);

  kfd_event_data ev{};
  ev.event_id = create.event_id;
  kfd_ioctl_wait_events_args wait{};
  wait.events_ptr = reinterpret_cast<uint64_t>(&ev);
  wait.num_events = 1;
  wait.wait_for_all = 1;
  wait.timeout = 0;
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_WAIT_EVENTS, &wait), 0);

  EXPECT_EQ(wait.wait_result, KFD_IOC_WAIT_RESULT_COMPLETE)
      << "the memory-exception event must be signaled by the rejected access";
  EXPECT_EQ(ev.memory_exception_data.va, faulting_va)
      << "the report must name the address that faulted";
  EXPECT_EQ(ev.memory_exception_data.failure.NotPresent, 1u);
  EXPECT_EQ(ev.memory_exception_data.failure.ReadOnly, 0u);

  munmap(raw, rocjitsu::KfdProcess::kPageSize);
  EXPECT_EQ(drv->close(), 0);
}

// A fault has to name the device it happened on. The runtime maps gpu_id to a
// node and surfaces it as the faulting node, so reporting every device's
// violation as the first one is misattribution the caller can see, not just
// untidy metadata.
TEST_F(SimulatedKfdTest, MemoryExceptionNamesTheFaultingGpu) {
  const std::string config = std::string(CONFIG_DIR) + "/gfx950_mi355x_kmd_2gpu.json";
  rj_vm_t *vm = nullptr;
  ASSERT_EQ(rj_vm_create(config.c_str(), RJ_VM_MODE_LOCAL, &vm), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(vm, nullptr);
  ASSERT_GE(vm->vm->num_socs(), 2u) << "this regression needs a second device to misattribute";

  auto *drv = dynamic_cast<rocjitsu::SimulatedKfd *>(vm->vm->driver());
  ASSERT_NE(drv, nullptr);
  auto *second_memory = vm->vm->soc(1)->memory();
  ASSERT_NE(second_memory, nullptr);

  const uint32_t process_id = drv->local_process_id();
  auto proc = drv->find_process(process_id);
  ASSERT_NE(proc, nullptr);

  constexpr size_t kSlots = 64;
  std::vector<uint64_t> page(kSlots, 0);
  proc->event_state_.adopt_page(page.data(), page.size() * sizeof(uint64_t));

  kfd_ioctl_create_event_args create{};
  create.event_type = KFD_IOC_EVENT_MEMORY;
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_CREATE_EVENT, &create), 0);

  void *raw =
      mmap(nullptr, rocjitsu::KfdProcess::kPageSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(raw, MAP_FAILED);

  // Fault the SECOND device only.
  second_memory->read32(reinterpret_cast<uint64_t>(raw), process_id);

  kfd_event_data ev{};
  ev.event_id = create.event_id;
  kfd_ioctl_wait_events_args wait{};
  wait.events_ptr = reinterpret_cast<uint64_t>(&ev);
  wait.num_events = 1;
  wait.wait_for_all = 1;
  wait.timeout = 0;
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_WAIT_EVENTS, &wait), 0);
  ASSERT_EQ(wait.wait_result, KFD_IOC_WAIT_RESULT_COMPLETE);
  EXPECT_EQ(ev.memory_exception_data.gpu_id, drv->gpu_id_at(1))
      << "a fault on the second device must not be reported against the first";
  EXPECT_NE(drv->gpu_id_at(1), drv->gpu_id_at(0))
      << "the devices must differ or this asserts nothing";

  munmap(raw, rocjitsu::KfdProcess::kPageSize);
  rj_vm_destroy(vm);
}

// A page that is readable but refuses the write is a protection fault, and the
// KFD ABI reports it in a different field than a missing page. The runtime
// reads the two separately, so collapsing them misstates the cause.
TEST_F(SimulatedKfdTest, ReadOnlyPageReportsReadOnlyRatherThanNotPresent) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  auto *drv = t.driver();
  auto *memory = t.memory();
  ASSERT_NE(memory, nullptr);

  ASSERT_GE(drv->open(), 0);
  const uint32_t process_id = drv->local_process_id();
  auto proc = drv->find_process(process_id);
  ASSERT_NE(proc, nullptr);

  constexpr size_t kSlots = 64;
  std::vector<uint64_t> page(kSlots, 0);
  proc->event_state_.adopt_page(page.data(), page.size() * sizeof(uint64_t));

  kfd_ioctl_create_event_args create{};
  create.event_type = KFD_IOC_EVENT_MEMORY;
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_CREATE_EVENT, &create), 0);

  void *raw =
      mmap(nullptr, rocjitsu::KfdProcess::kPageSize, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(raw, MAP_FAILED);
  const uint64_t addr = reinterpret_cast<uint64_t>(raw);

  // Reading is fine; only the write violates the protection.
  memory->read32(addr, process_id);
  memory->write32(addr, 0x1234u, process_id);

  kfd_event_data ev{};
  ev.event_id = create.event_id;
  kfd_ioctl_wait_events_args wait{};
  wait.events_ptr = reinterpret_cast<uint64_t>(&ev);
  wait.num_events = 1;
  wait.wait_for_all = 1;
  wait.timeout = 0;
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_WAIT_EVENTS, &wait), 0);
  ASSERT_EQ(wait.wait_result, KFD_IOC_WAIT_RESULT_COMPLETE);
  EXPECT_EQ(ev.memory_exception_data.failure.ReadOnly, 1u);
  EXPECT_EQ(ev.memory_exception_data.failure.NotPresent, 0u);

  munmap(raw, rocjitsu::KfdProcess::kPageSize);
  EXPECT_EQ(drv->close(), 0);
}

// An atomic against a mapped page that is not writable must name the cause the
// mapping actually has. A PROT_NONE reservation -- the shape the runtime leaves
// around an aperture -- is absent to the GPU, so reporting it as a protection
// violation would send the runtime looking for a permission problem on memory
// that is not mapped at all.
TEST_F(SimulatedKfdTest, MappedReservationReportsNotPresentRatherThanReadOnly) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  auto *drv = t.driver();
  auto *memory = t.memory();
  ASSERT_NE(memory, nullptr);

  ASSERT_GE(drv->open(), 0);
  const uint32_t process_id = drv->local_process_id();
  auto proc = drv->find_process(process_id);
  ASSERT_NE(proc, nullptr);

  constexpr size_t kSlots = 64;
  std::vector<uint64_t> page(kSlots, 0);
  proc->event_state_.adopt_page(page.data(), page.size() * sizeof(uint64_t));

  kfd_ioctl_create_event_args create{};
  create.event_type = KFD_IOC_EVENT_MEMORY;
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_CREATE_EVENT, &create), 0);

  void *raw =
      mmap(nullptr, rocjitsu::KfdProcess::kPageSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(raw, MAP_FAILED);

  // Reached through a page-table entry at a GPU address that is nothing like a
  // host pointer, so the mapped path is the only one that can service it: an
  // identity fallback would answer a different question and let the mapped
  // classification regress unnoticed.
  constexpr uint64_t kGpuVa = 0x8000'0000'0000ULL;
  proc->map_pages(kGpuVa, raw, rocjitsu::KfdProcess::kPageSize);
  ASSERT_TRUE(memory->is_mapped(kGpuVa, process_id))
      << "the atomic must resolve through the page table, not by identity";

  EXPECT_EQ(memory->atomic_store(kGpuVa, sizeof(uint64_t), 1, process_id),
            rocjitsu::amdgpu::AccessOutcome::Faulted);

  kfd_event_data ev{};
  ev.event_id = create.event_id;
  kfd_ioctl_wait_events_args wait{};
  wait.events_ptr = reinterpret_cast<uint64_t>(&ev);
  wait.num_events = 1;
  wait.wait_for_all = 1;
  wait.timeout = 0;
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_WAIT_EVENTS, &wait), 0);
  ASSERT_EQ(wait.wait_result, KFD_IOC_WAIT_RESULT_COMPLETE);
  EXPECT_EQ(ev.memory_exception_data.va, kGpuVa);
  EXPECT_EQ(ev.memory_exception_data.failure.NotPresent, 1u);
  EXPECT_EQ(ev.memory_exception_data.failure.ReadOnly, 0u);

  munmap(raw, rocjitsu::KfdProcess::kPageSize);
  EXPECT_EQ(drv->close(), 0);
}

// The reporting path must stay quiet for addresses that do resolve, or every
// local-mode workload would raise faults for its ordinary pageable pointers.
TEST_F(SimulatedKfdTest, ResolvableAddressRaisesNoMemoryException) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  auto *drv = t.driver();
  auto *memory = t.memory();
  ASSERT_NE(memory, nullptr);

  ASSERT_GE(drv->open(), 0);
  const uint32_t process_id = drv->local_process_id();
  auto proc = drv->find_process(process_id);
  ASSERT_NE(proc, nullptr);

  constexpr size_t kSlots = 64;
  std::vector<uint64_t> page(kSlots, 0);
  proc->event_state_.adopt_page(page.data(), page.size() * sizeof(uint64_t));

  kfd_ioctl_create_event_args create{};
  create.event_type = KFD_IOC_EVENT_MEMORY;
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_CREATE_EVENT, &create), 0);

  void *raw = mmap(nullptr, rocjitsu::KfdProcess::kPageSize, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(raw, MAP_FAILED);
  const uint64_t addr = reinterpret_cast<uint64_t>(raw) + 0x40;

  constexpr uint32_t kValue = 0x5eeded;
  memory->write32(addr, kValue, process_id);
  EXPECT_EQ(memory->read32(addr, process_id), kValue);

  kfd_event_data ev{};
  ev.event_id = create.event_id;
  kfd_ioctl_wait_events_args wait{};
  wait.events_ptr = reinterpret_cast<uint64_t>(&ev);
  wait.num_events = 1;
  wait.wait_for_all = 1;
  wait.timeout = 0;
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_WAIT_EVENTS, &wait), 0);
  EXPECT_EQ(wait.wait_result, KFD_IOC_WAIT_RESULT_TIMEOUT)
      << "a translation that succeeded must not report a violation";

  munmap(raw, rocjitsu::KfdProcess::kPageSize);
  EXPECT_EQ(drv->close(), 0);
}

TEST_F(SimulatedKfdTest, BeginLocalShutdownLeavesSignaledEventPageIntact) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  auto *drv = t.driver();

  ASSERT_GE(drv->open(), 0);
  auto proc = drv->find_process(drv->local_process_id());
  ASSERT_NE(proc, nullptr);

  // Provide a real event page and adopt it, mirroring the CREATE_EVENT mmap path.
  constexpr size_t kSlots = 64;
  std::vector<uint64_t> page(kSlots, 0);
  proc->event_state_.adopt_page(page.data(), page.size() * sizeof(uint64_t));

  // Create a signal event and signal it, so its page slot holds a real (non-zero,
  // non-sentinel) age.
  kfd_ioctl_create_event_args create{};
  create.event_type = 0; // signal event
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_CREATE_EVENT, &create), 0);

  kfd_ioctl_set_event_args set{};
  set.event_id = create.event_id;
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_SET_EVENT, &set), 0);

  const uint64_t signaled_age = page[create.event_id];
  ASSERT_NE(signaled_age, 0u);
  ASSERT_NE(signaled_age, static_cast<uint64_t>(KFD_SIGNAL_EVENT_LIMIT));

  drv->begin_local_shutdown();
  EXPECT_EQ(page[create.event_id], signaled_age)
      << "begin_local_shutdown must not disturb the event page";
  EXPECT_FALSE(proc->event_state_.is_closing())
      << "begin_local_shutdown must not mark the driver closing: that turns a live "
         "consumer's ioctls into -EBADF/-ESRCH and cannot be undone faithfully";

  // A wait issued after the wake is released rather than failed: it reports a
  // benign timeout (rc 0), never -EBADF. That distinction is why the wake must not
  // set closing_ — an in-flight caller must be able to unwind normally while
  // teardown drains, and -EBADF is not a legal answer from a still-published driver.
  kfd_event_data ev{};
  ev.event_id = create.event_id;
  kfd_ioctl_wait_events_args wait{};
  wait.events_ptr = reinterpret_cast<uint64_t>(&ev);
  wait.num_events = 1;
  wait.wait_for_all = 1;
  wait.timeout = 0;
  EXPECT_EQ(drv->ioctl(AMDKFD_IOC_WAIT_EVENTS, &wait), 0)
      << "a wait after the wake must not fail the caller";

  EXPECT_EQ(drv->close(), 0);
}

// An AUTO-RESET event that was signaled while a waiter was parked is the state the
// old speculative-shutdown design lost: signaling advances and publishes the age but
// deliberately leaves signaled == false, so a rollback that rebuilt the page from
// `signaled` events alone replaced a real pending completion with the unsignaled
// sentinel. The wake must therefore leave the page alone even in this state. There is
// no rollback any more, but the property is what makes that safe, so pin it.
TEST_F(SimulatedKfdTest, WakeDoesNotDisturbPendingAutoResetEventPage) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  auto *drv = t.driver();
  ASSERT_GE(drv->open(), 0);
  auto proc = drv->find_process(drv->local_process_id());
  ASSERT_NE(proc, nullptr);

  constexpr size_t kSlots = 64;
  std::vector<uint64_t> page(kSlots, 0);
  proc->event_state_.adopt_page(page.data(), page.size() * sizeof(uint64_t));

  kfd_ioctl_create_event_args create{};
  create.event_type = 0;
  create.auto_reset = 1;
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_CREATE_EVENT, &create), 0);

  // Register a waiter deterministically: poll the event's waiter count rather than
  // sleeping, so the SET_EVENT below is guaranteed to take the "waiters present"
  // auto-reset path (which advances and publishes the age while deliberately
  // leaving signaled == false).
  std::atomic<int> wait_rc{-1};
  std::atomic<uint32_t> wait_result{0};
  std::thread waiter([&] {
    kfd_event_data ev{};
    ev.event_id = create.event_id;
    kfd_ioctl_wait_events_args wait{};
    wait.events_ptr = reinterpret_cast<uint64_t>(&ev);
    wait.num_events = 1;
    wait.wait_for_all = 1;
    wait.timeout = 0xFFFFFFFFu;
    wait_rc.store(drv->ioctl(AMDKFD_IOC_WAIT_EVENTS, &wait), std::memory_order_release);
    wait_result.store(wait.wait_result, std::memory_order_release);
  });
  while (proc->event_state_.waiter_count(create.event_id) == 0)
    std::this_thread::yield();

  kfd_ioctl_set_event_args set{};
  set.event_id = create.event_id;
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_SET_EVENT, &set), 0);
  const uint64_t pending_age =
      std::atomic_ref<uint64_t>(page[create.event_id]).load(std::memory_order_acquire);
  ASSERT_NE(pending_age, 0u);
  ASSERT_NE(pending_age, static_cast<uint64_t>(KFD_SIGNAL_EVENT_LIMIT));

  drv->begin_local_shutdown();
  waiter.join();

  // The slot is written with atomic_ref by the driver, so read it the same way.
  EXPECT_EQ(std::atomic_ref<uint64_t>(page[create.event_id]).load(std::memory_order_acquire),
            pending_age)
      << "the wake must not overwrite a pending auto-reset completion whose event is "
         "not flagged signaled; that is the lost signal the old page rollback caused";
  EXPECT_EQ(wait_rc.load(std::memory_order_acquire), 0)
      << "the released waiter must not fail its caller";

  EXPECT_EQ(drv->close(), 0);
}

// The wake exists to release an INDEFINITE WAIT_EVENTS so its caller drops the
// driver snapshot that would otherwise keep the object alive forever. Assert that
// directly: a thread parked with an infinite timeout must be released, and it must
// come back as a benign timeout (rc 0) rather than -EBADF, which is not a legal
// answer for a driver that is still serving.
TEST_F(SimulatedKfdTest, BeginLocalShutdownReleasesIndefiniteWaitAsTimeout) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  auto *drv = t.driver();

  ASSERT_GE(drv->open(), 0);
  auto proc = drv->find_process(drv->local_process_id());
  ASSERT_NE(proc, nullptr);

  constexpr size_t kSlots = 64;
  std::vector<uint64_t> page(kSlots, 0);
  proc->event_state_.adopt_page(page.data(), page.size() * sizeof(uint64_t));

  kfd_ioctl_create_event_args create{};
  create.event_type = 0;
  ASSERT_EQ(drv->ioctl(AMDKFD_IOC_CREATE_EVENT, &create), 0);

  std::atomic<bool> parked{false};
  int wait_rc = -1;
  uint32_t wait_result = 0;
  std::thread waiter([&] {
    kfd_event_data ev{};
    ev.event_id = create.event_id;
    kfd_ioctl_wait_events_args wait{};
    wait.events_ptr = reinterpret_cast<uint64_t>(&ev);
    wait.num_events = 1;
    wait.wait_for_all = 1;
    wait.timeout = 0xFFFFFFFFu; // indefinite
    parked.store(true, std::memory_order_release);
    wait_rc = drv->ioctl(AMDKFD_IOC_WAIT_EVENTS, &wait);
    wait_result = wait.wait_result;
  });

  while (!parked.load(std::memory_order_acquire))
    std::this_thread::yield();
  // Give the waiter a moment to actually park inside the condition variable.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  drv->begin_local_shutdown();
  waiter.join();

  EXPECT_EQ(wait_rc, 0) << "a cancelled wait must not fail the caller's ioctl";
  EXPECT_EQ(wait_result, static_cast<uint32_t>(KFD_IOC_WAIT_RESULT_TIMEOUT))
      << "a cancelled wait must report a benign timeout, not a completion";

  EXPECT_EQ(drv->close(), 0);
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

// GuestKfd's hardware path cannot wake a blocking kernel WAIT_EVENTS, so it forwards
// the wait as short, cancellable polls instead. That loop is the contract the
// interposer's phase-2 teardown wake depends on for a hardware-backed guest, but
// reaching it through GuestKfd::ioctl() needs a real /dev/kfd. These drive the
// algorithm directly with an injected poll, covering all five of its outcomes.

// A zero-timeout poll is already prompt, so it must be forwarded verbatim: exactly
// one poll, with the caller's timeout untouched (no kPollMs slice substituted).
TEST(GuestKfdWaitPollLoopTest, ZeroTimeoutForwardsUnchanged) {
  std::atomic<bool> cancelled{false};
  kfd_ioctl_wait_events_args args{};
  args.timeout = 0;

  int polls = 0;
  uint32_t observed_timeout = 0xDEADBEEF;
  int rc = rocjitsu::GuestKfdTestAccess::wait_events_poll_loop(args, cancelled, [&] {
    ++polls;
    observed_timeout = args.timeout;
    args.wait_result = KFD_IOC_WAIT_RESULT_TIMEOUT;
    return 0;
  });

  EXPECT_EQ(rc, 0);
  EXPECT_EQ(polls, 1) << "a zero-timeout poll must not be split into slices";
  EXPECT_EQ(observed_timeout, 0u) << "a zero-timeout poll must be forwarded verbatim";
  EXPECT_EQ(args.timeout, 0u);
}

// Cancellation (begin_local_shutdown) must end the loop with a benign timeout and
// leave the caller's timeout field as they passed it. The caller is unwinding
// through the re-poll path it already has for a timeout -- teardown is committed by
// the time this fires, so the re-poll finds the driver gone rather than resuming.
TEST(GuestKfdWaitPollLoopTest, CancellationReportsTimeoutAndRestoresTheTimeout) {
  std::atomic<bool> cancelled{false};
  kfd_ioctl_wait_events_args args{};
  args.timeout = 0xFFFFFFFFu; // indefinite
  args.wait_result = KFD_IOC_WAIT_RESULT_COMPLETE;

  int polls = 0;
  int rc = rocjitsu::GuestKfdTestAccess::wait_events_poll_loop(args, cancelled, [&] {
    ++polls;
    EXPECT_NE(args.timeout, 0xFFFFFFFFu) << "each kernel poll must use the short slice";
    cancelled.store(true, std::memory_order_release);
    args.wait_result = KFD_IOC_WAIT_RESULT_TIMEOUT;
    return 0;
  });

  EXPECT_EQ(rc, 0) << "cancellation must not fail the caller's ioctl";
  EXPECT_EQ(args.wait_result, static_cast<uint32_t>(KFD_IOC_WAIT_RESULT_TIMEOUT));
  EXPECT_EQ(args.timeout, 0xFFFFFFFFu) << "the caller's timeout must be restored";
  EXPECT_EQ(polls, 1);
}

// An indefinite wait must keep polling across timeouts and return as soon as an
// event completes, never expiring on its own.
TEST(GuestKfdWaitPollLoopTest, IndefiniteWaitPollsUntilCompletion) {
  std::atomic<bool> cancelled{false};
  kfd_ioctl_wait_events_args args{};
  args.timeout = 0xFFFFFFFFu;

  int polls = 0;
  int rc = rocjitsu::GuestKfdTestAccess::wait_events_poll_loop(args, cancelled, [&] {
    ++polls;
    args.wait_result = (polls < 4) ? KFD_IOC_WAIT_RESULT_TIMEOUT : KFD_IOC_WAIT_RESULT_COMPLETE;
    return 0;
  });

  EXPECT_EQ(rc, 0);
  EXPECT_EQ(polls, 4) << "an indefinite wait must re-poll until an event completes";
  EXPECT_EQ(args.wait_result, static_cast<uint32_t>(KFD_IOC_WAIT_RESULT_COMPLETE));
  EXPECT_EQ(args.timeout, 0xFFFFFFFFu);
}

// A finite wait whose events never fire must expire on its own deadline (not spin
// forever), reporting the caller-visible timeout and restoring the timeout field.
TEST(GuestKfdWaitPollLoopTest, FiniteDeadlineExpiresWithTimeout) {
  std::atomic<bool> cancelled{false};
  kfd_ioctl_wait_events_args args{};
  args.timeout = 20; // ms

  int polls = 0;
  const auto started = std::chrono::steady_clock::now();
  int rc = rocjitsu::GuestKfdTestAccess::wait_events_poll_loop(args, cancelled, [&] {
    ++polls;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    args.wait_result = KFD_IOC_WAIT_RESULT_TIMEOUT;
    return 0;
  });
  const auto elapsed = std::chrono::steady_clock::now() - started;

  EXPECT_EQ(rc, 0);
  EXPECT_EQ(args.wait_result, static_cast<uint32_t>(KFD_IOC_WAIT_RESULT_TIMEOUT));
  EXPECT_EQ(args.timeout, 20u) << "the caller's timeout must be restored";
  EXPECT_GE(polls, 1);
  EXPECT_GE(elapsed, std::chrono::milliseconds(20)) << "must not return before its deadline";
}

// A failing kernel poll must propagate immediately and must never leave the internal
// slice value in the returned args. This covers the indefinite case, which reports the
// caller's original timeout back; a finite wait instead reports what is LEFT of the
// budget, so an EINTR retry continues it rather than restarting it.
TEST(GuestKfdWaitPollLoopTest, PollFailurePropagatesWithTheTimeoutRestored) {
  std::atomic<bool> cancelled{false};
  kfd_ioctl_wait_events_args args{};
  args.timeout = 0xFFFFFFFFu;

  int polls = 0;
  int rc = rocjitsu::GuestKfdTestAccess::wait_events_poll_loop(args, cancelled, [&] {
    ++polls;
    return -1;
  });

  EXPECT_EQ(rc, -1);
  EXPECT_EQ(polls, 1);
  EXPECT_EQ(args.timeout, 0xFFFFFFFFu);
}

// The restart convention for a FINITE wait: a failed poll reports what is left of the
// caller's budget, never the original. Restoring the original would let a caller that
// retries on EINTR restart the wait each time and never reach its deadline.
TEST(GuestKfdWaitPollLoopTest, PollFailureOnAFiniteWaitReportsTheRemainingBudget) {
  std::atomic<bool> cancelled{false};
  constexpr uint32_t kBudgetMs = 500;
  kfd_ioctl_wait_events_args args{};
  args.timeout = kBudgetMs;

  int polls = 0;
  int rc = rocjitsu::GuestKfdTestAccess::wait_events_poll_loop(args, cancelled, [&] {
    ++polls;
    // Burn a little of the budget so the remainder is strictly smaller, without
    // getting anywhere near the deadline.
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return -1;
  });

  EXPECT_EQ(rc, -1);
  EXPECT_EQ(polls, 1);
  EXPECT_LT(args.timeout, kBudgetMs) << "a failed finite poll must not restart the budget";
  EXPECT_GT(args.timeout, 0u) << "most of the budget was still left";
}

} // namespace
