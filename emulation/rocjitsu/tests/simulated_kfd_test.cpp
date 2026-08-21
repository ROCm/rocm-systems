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

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
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
