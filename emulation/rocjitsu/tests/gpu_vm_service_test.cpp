// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/command_processor_queue_backend.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/queue_service.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

class ByteAddressSpace final : public AddressSpaceTranslator, public PhysicalMemoryAccess {
public:
  explicit ByteAddressSpace(uint8_t value) : bytes_(4096, static_cast<std::byte>(value)) {}

  VmTranslationResult translate(uint64_t address, std::size_t size,
                                VmAccessKind /*access*/) const override {
    if (size == 0 || address > bytes_.size() || size > bytes_.size() - address)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation = {.domain = VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = bytes_.size() - address,
                        .mtype = Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }

  VmAccessOutcome read(VmMemoryDomain domain, uint64_t address,
                       std::span<std::byte> bytes) override {
    if (domain != VmMemoryDomain::System)
      return VmAccessOutcome::Malformed;
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::copy_n(bytes_.begin() + static_cast<ptrdiff_t>(address), bytes.size(), bytes.begin());
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(VmMemoryDomain domain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    if (domain != VmMemoryDomain::System)
      return VmAccessOutcome::Malformed;
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::copy(bytes.begin(), bytes.end(), bytes_.begin() + static_cast<ptrdiff_t>(address));
    return VmAccessOutcome::Complete;
  }

private:
  std::vector<std::byte> bytes_;
};

AddressSpaceHandle register_byte_address_space(GpuVm &gpu_vm, uint32_t vmid, uint8_t value) {
  auto access = std::make_shared<ByteAddressSpace>(value);
  return gpu_vm.register_translated(vmid, access, access);
}

QueueCreateInfo queue_info(AddressSpaceHandle address_space, CommandProcessor &owner,
                           uint32_t queue_id) {
  return {.address_space = address_space,
          .backend = make_command_processor_queue_backend(owner),
          .process_id = 7,
          .queue_id = queue_id,
          .ring_size = 4096};
}

enum class BlockedQueueOperation : uint8_t { Attach, Update, Doorbell };

class BlockingQueueBackend final : public QueueBackend {
public:
  explicit BlockingQueueBackend(BlockedQueueOperation blocked_operation)
      : blocked_operation_(blocked_operation) {}

  void attach(const QueueCreateInfo &) override {
    std::unique_lock lock(mutex_);
    ++attach_calls_;
    if (blocked_operation_ == BlockedQueueOperation::Attach)
      block_locked(lock);
  }

  void rollback_attach(const QueueCreateInfo &) noexcept override {
    std::lock_guard lock(mutex_);
    ++rollback_calls_;
  }

  void update(uint32_t, uint32_t, uint64_t, uint32_t, uint32_t) override {
    block(BlockedQueueOperation::Update);
  }

  QueueDoorbellDisposition notify_doorbell(uint32_t, uint32_t, uint64_t) override {
    block(BlockedQueueOperation::Doorbell);
    return QueueDoorbellDisposition::Complete;
  }

  void detach(uint32_t, uint32_t) noexcept override {
    {
      std::lock_guard lock(mutex_);
      ++detach_calls_;
      condition_.notify_all();
    }
    if (detach_observer_)
      detach_observer_();
  }

  void set_detach_observer(std::function<void()> observer) {
    detach_observer_ = std::move(observer);
  }

  bool wait_until_callback_is_blocked() {
    std::unique_lock lock(mutex_);
    return condition_.wait_for(lock, std::chrono::seconds(5),
                               [this]() { return callback_is_blocked_; });
  }

  void release_callback() {
    std::lock_guard lock(mutex_);
    release_callback_ = true;
    condition_.notify_all();
  }

  uint32_t attach_calls() const {
    std::lock_guard lock(mutex_);
    return attach_calls_;
  }

  uint32_t detach_calls() const {
    std::lock_guard lock(mutex_);
    return detach_calls_;
  }

  uint32_t rollback_calls() const {
    std::lock_guard lock(mutex_);
    return rollback_calls_;
  }

private:
  void block(BlockedQueueOperation operation) {
    if (operation != blocked_operation_)
      return;
    std::unique_lock lock(mutex_);
    block_locked(lock);
  }

  void block_locked(std::unique_lock<std::mutex> &lock) {
    callback_is_blocked_ = true;
    condition_.notify_all();
    condition_.wait(lock, [this]() { return release_callback_; });
  }

  const BlockedQueueOperation blocked_operation_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  uint32_t attach_calls_ = 0;
  uint32_t rollback_calls_ = 0;
  uint32_t detach_calls_ = 0;
  bool callback_is_blocked_ = false;
  bool release_callback_ = false;
  std::function<void()> detach_observer_;
};

class MutatingThrowQueueBackend final : public QueueBackend {
public:
  void attach(const QueueCreateInfo &) override {
    std::lock_guard lock(mutex_);
    attached_ = true;
    throw std::runtime_error("injected attach failure after mutation");
  }

  void rollback_attach(const QueueCreateInfo &) noexcept override {
    std::lock_guard lock(mutex_);
    attached_ = false;
    ++rollback_calls_;
  }

  void update(uint32_t, uint32_t, uint64_t, uint32_t, uint32_t) override {}
  QueueDoorbellDisposition notify_doorbell(uint32_t, uint32_t, uint64_t) override {
    return QueueDoorbellDisposition::Complete;
  }

  void detach(uint32_t, uint32_t) noexcept override {
    std::lock_guard lock(mutex_);
    ++detach_calls_;
  }

  bool attached() const {
    std::lock_guard lock(mutex_);
    return attached_;
  }

  uint32_t rollback_calls() const {
    std::lock_guard lock(mutex_);
    return rollback_calls_;
  }

  uint32_t detach_calls() const {
    std::lock_guard lock(mutex_);
    return detach_calls_;
  }

private:
  mutable std::mutex mutex_;
  bool attached_ = false;
  uint32_t rollback_calls_ = 0;
  uint32_t detach_calls_ = 0;
};

QueueCreateInfo queue_info(AddressSpaceHandle address_space,
                           const std::shared_ptr<QueueBackend> &backend, uint32_t queue_id) {
  return {.address_space = address_space,
          .backend = backend,
          .process_id = 7,
          .queue_id = queue_id,
          .ring_size = 4096};
}

bool wait_until_queue_is_revoked(const QueueService &queues, QueueHandle handle) {
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (queues.contains(handle) && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  return !queues.contains(handle);
}

bool wait_until_create_admission_closes(const QueueService &queues) {
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (queues.accepting_creates_for_test() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  return !queues.accepting_creates_for_test();
}

TEST(GpuVmService, TwoQueuesRetainOneAddressSpace) {
  GpuMemory memory("memory");
  GpuVm gpu_vm(&memory);
  QueueService queues(gpu_vm);
  CommandProcessor command_processor("cp");
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x2a);

  QueueHandle first = queues.create(queue_info(address_space, command_processor, 1));
  QueueHandle second = queues.create(queue_info(address_space, command_processor, 2));

  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(command_processor.queue_address_space_for_test(1, 7), address_space);
  EXPECT_EQ(command_processor.queue_address_space_for_test(2, 7), address_space);
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 2u);
  EXPECT_FALSE(gpu_vm.unregister_address_space(address_space));

  EXPECT_TRUE(queues.destroy(first));
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 1u);
  std::array<std::byte, 1> value{};
  EXPECT_EQ(gpu_vm.read(address_space, 0, value), VmAccessOutcome::Complete);
  EXPECT_EQ(std::to_integer<uint8_t>(value[0]), 0x2a);

  EXPECT_TRUE(queues.destroy(second));
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
  EXPECT_FALSE(gpu_vm.lookup(address_space));
}

TEST(GpuVmService, QueueCannotRetainOneAddressSpaceAndRouteThroughAnotherVmid) {
  GpuMemory memory("memory");
  GpuVm gpu_vm(&memory);
  QueueService queues(gpu_vm);
  CommandProcessor command_processor("cp");
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x2a);
  QueueCreateInfo mismatched = queue_info(address_space, command_processor, 1);
  mismatched.process_id = 8;

  EXPECT_FALSE(queues.create(mismatched));
  EXPECT_EQ(command_processor.registered_queue_count_for_test(), 0u);
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(CommandProcessorDoorbell, TransportNotificationDoesNotWaitForQueueExecutionLock) {
  CommandProcessor command_processor("cp");
  HwQueue queue{
      .address_space = {}, .process_id = 7, .queue_id = 1, .ring_size = 4096, .last_doorbell = 4};
  const uint64_t registration = command_processor.register_queue(queue);

  std::promise<void> locked;
  std::promise<void> release;
  std::shared_future<void> release_future = release.get_future().share();
  std::future<void> holder = std::async(std::launch::async, [&]() {
    command_processor.with_queue_lock_for_test([&]() {
      locked.set_value();
      release_future.wait();
    });
  });
  locked.get_future().wait();

  std::future<void> notification = std::async(
      std::launch::async, [&]() { command_processor.notify_queue_doorbell(registration, 8); });
  EXPECT_EQ(notification.wait_for(std::chrono::milliseconds(50)), std::future_status::ready);
  notification.get();

  release.set_value();
  holder.get();
  command_processor.drain_doorbell_inbox_for_test();
  EXPECT_EQ(command_processor.queue_last_doorbell_for_test(registration), 8u);
  command_processor.unregister_queue(queue.queue_id, queue.process_id);
}

TEST(CommandProcessorDoorbell, StaleRegistrationCannotNotifyReusedQueueIdentity) {
  CommandProcessor command_processor("cp");
  HwQueue queue{
      .address_space = {}, .process_id = 7, .queue_id = 1, .ring_size = 4096, .last_doorbell = 4};
  const uint64_t stale = command_processor.register_queue(queue);
  command_processor.unregister_queue(queue.queue_id, queue.process_id);

  const uint64_t current = command_processor.register_queue(queue);
  ASSERT_NE(current, stale);
  command_processor.notify_queue_doorbell(stale, 12);
  command_processor.drain_doorbell_inbox_for_test();
  EXPECT_EQ(command_processor.queue_last_doorbell_for_test(current), 4u);
  command_processor.unregister_queue(queue.queue_id, queue.process_id);
}

TEST(CommandProcessorVmFault, FaultsEveryFanoutReplicaAndDropsPresentAndFutureWork) {
  CommandProcessor owner("owner");
  CommandProcessor peer("peer");
  owner.set_xcd_topology(0, {&owner, &peer});
  peer.set_xcd_topology(1, {&owner, &peer});

  constexpr uint32_t kProcessId = 7;
  constexpr uint32_t kQueueId = 41;
  constexpr uint32_t kDispatchId = 19;
  HwQueue queue{.address_space = {},
                .process_id = kProcessId,
                .queue_id = kQueueId,
                .ring_size = 4096,
                .xcd_fanout = true};
  (void)owner.register_queue(queue);

  auto grid = std::make_shared<GridCompletion>();
  grid->grid_wgs = 2;
  DispatchEntry owner_entry{};
  owner_entry.dispatch_id = kDispatchId;
  owner_entry.queue_id = kQueueId;
  owner_entry.process_id = kProcessId;
  owner_entry.kind = DispatchPacketKind::Kernel;
  owner_entry.total_wgs = 1;
  owner_entry.grid_completion = grid;
  DispatchEntry peer_entry = owner_entry;
  peer_entry.fanout_peer = true;

  owner.accept_fanout_shard(std::move(owner_entry));
  peer.accept_fanout_shard(std::move(peer_entry));
  owner.drain_fanout_inbox_for_test();
  peer.drain_fanout_inbox_for_test();
  ASSERT_TRUE(owner.has_dispatch_for_test(kQueueId, kProcessId, kDispatchId));
  ASSERT_TRUE(peer.has_dispatch_for_test(kQueueId, kProcessId, kDispatchId));

  // A fault can originate on any shard. The callback broadcasts without ever
  // holding two CP queue locks at once.
  peer.notify_dispatch_vm_fault(kQueueId, kProcessId, kDispatchId, VmAccessOutcome::Faulted);
  owner.drain_fanout_inbox_for_test();
  EXPECT_TRUE(grid->faulted());
  EXPECT_TRUE(owner.queue_faulted_for_test(kQueueId, kProcessId));
  EXPECT_TRUE(peer.queue_faulted_for_test(kQueueId, kProcessId));
  EXPECT_FALSE(owner.has_dispatch_for_test(kQueueId, kProcessId, kDispatchId));
  EXPECT_FALSE(peer.has_dispatch_for_test(kQueueId, kProcessId, kDispatchId));

  // Re-reporting the same terminal fault is idempotent, and a shard already in
  // flight after the fault cannot reintroduce work behind the stopped queue.
  peer.notify_dispatch_vm_fault(kQueueId, kProcessId, kDispatchId, VmAccessOutcome::Faulted);
  owner.drain_fanout_inbox_for_test();
  DispatchEntry late{};
  late.dispatch_id = kDispatchId + 1;
  late.queue_id = kQueueId;
  late.process_id = kProcessId;
  late.kind = DispatchPacketKind::Kernel;
  late.total_wgs = 1;
  peer.accept_fanout_shard(std::move(late));
  peer.drain_fanout_inbox_for_test();
  EXPECT_FALSE(peer.has_dispatch_for_test(kQueueId, kProcessId, kDispatchId + 1));

  owner.unregister_queue(kQueueId, kProcessId);
}

TEST(CommandProcessorVmFault, RejectsNonterminalOrMismatchedNotifications) {
  CommandProcessor command_processor("cp");
  constexpr uint32_t kProcessId = 7;
  constexpr uint32_t kQueueId = 42;
  constexpr uint32_t kDispatchId = 23;
  (void)command_processor.register_queue(
      {.address_space = {}, .process_id = kProcessId, .queue_id = kQueueId, .ring_size = 4096});

  DispatchEntry entry{};
  entry.dispatch_id = kDispatchId;
  entry.queue_id = kQueueId;
  entry.process_id = kProcessId;
  entry.kind = DispatchPacketKind::Kernel;
  entry.total_wgs = 1;
  command_processor.accept_fanout_shard(std::move(entry));
  command_processor.drain_fanout_inbox_for_test();

  command_processor.notify_dispatch_vm_fault(kQueueId, kProcessId, kDispatchId + 1,
                                             VmAccessOutcome::Faulted);
  command_processor.notify_dispatch_vm_fault(kQueueId, kProcessId, kDispatchId,
                                             VmAccessOutcome::Unavailable);
  EXPECT_FALSE(command_processor.queue_faulted_for_test(kQueueId, kProcessId));
  EXPECT_TRUE(command_processor.has_dispatch_for_test(kQueueId, kProcessId, kDispatchId));

  command_processor.unregister_queue(kQueueId, kProcessId);
}

TEST(GpuVmService, ReusedSlotsRejectStaleHandles) {
  GpuMemory memory("memory");
  GpuVm gpu_vm(&memory);
  QueueService queues(gpu_vm);
  CommandProcessor command_processor("cp");
  AddressSpaceHandle old_address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  QueueHandle old_queue = queues.create(queue_info(old_address_space, command_processor, 1));

  ASSERT_TRUE(queues.destroy(old_queue));
  ASSERT_TRUE(gpu_vm.unregister_address_space(old_address_space));
  AddressSpaceHandle new_address_space = register_byte_address_space(gpu_vm, 8, 0x22);
  QueueCreateInfo new_queue_info = queue_info(new_address_space, command_processor, 2);
  new_queue_info.process_id = 8;
  QueueHandle new_queue = queues.create(new_queue_info);

  EXPECT_EQ(new_address_space.slot, old_address_space.slot);
  EXPECT_NE(new_address_space.generation, old_address_space.generation);
  EXPECT_EQ(new_queue.slot, old_queue.slot);
  EXPECT_NE(new_queue.generation, old_queue.generation);
  EXPECT_FALSE(queues.notify_doorbell(old_queue, 1));
  EXPECT_FALSE(queues.destroy(old_queue));
  EXPECT_FALSE(gpu_vm.lookup(old_address_space));
  EXPECT_TRUE(queues.contains(new_queue));

  EXPECT_TRUE(queues.destroy(new_queue));
  EXPECT_TRUE(gpu_vm.unregister_address_space(new_address_space));
}

TEST(GpuVmService, ReplacingABindingPreservesIdentityAndAdvancesItsEpoch) {
  GpuMemory memory("memory");
  GpuVm gpu_vm(&memory);
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  ASSERT_TRUE(address_space);
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  const uint64_t old_epoch = gpu_vm.lookup(address_space)->translation_epoch;

  auto replacement = std::make_shared<ByteAddressSpace>(0x22);
  EXPECT_TRUE(gpu_vm.replace_translated(address_space, replacement, replacement));

  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->translation_epoch, old_epoch + 1);
  std::array<std::byte, 1> value{};
  EXPECT_EQ(gpu_vm.read(address_space, 0, value), VmAccessOutcome::Complete);
  EXPECT_EQ(std::to_integer<uint8_t>(value[0]), 0x22);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, AccessSnapshotRetainsOneBindingAndNamespacesItsTranslationEpoch) {
  GpuMemory memory("memory");
  GpuVm gpu_vm(&memory);
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  ASSERT_TRUE(address_space);

  const std::optional<GpuVmAccess> old_access = gpu_vm.snapshot(address_space);
  ASSERT_TRUE(old_access);
  const VmCacheNamespace old_namespace = old_access->cache_namespace();
  EXPECT_EQ(old_namespace.address_space, address_space);

  auto replacement = std::make_shared<ByteAddressSpace>(0x22);
  ASSERT_TRUE(gpu_vm.replace_translated(address_space, replacement, replacement));
  const std::optional<GpuVmAccess> new_access = gpu_vm.snapshot(address_space);
  ASSERT_TRUE(new_access);
  EXPECT_NE(new_access->cache_namespace(), old_namespace);
  EXPECT_EQ(new_access->cache_namespace().address_space, address_space);

  std::array<std::byte, 1> value{};
  EXPECT_EQ(old_access->read(0, value), VmAccessOutcome::Complete);
  EXPECT_EQ(std::to_integer<uint8_t>(value[0]), 0x11);
  EXPECT_EQ(new_access->read(0, value), VmAccessOutcome::Complete);
  EXPECT_EQ(std::to_integer<uint8_t>(value[0]), 0x22);

  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
  EXPECT_FALSE(gpu_vm.snapshot(address_space));
  EXPECT_EQ(old_access->read(0, value), VmAccessOutcome::Complete);
  EXPECT_EQ(std::to_integer<uint8_t>(value[0]), 0x11);
}

TEST(GpuVmService, ClearingGartPreservesItsIdentityAndUnrelatedAddressSpaces) {
  GpuMemory memory("memory");
  GpuVm gpu_vm(&memory);
  const AddressSpaceHandle gart = gpu_vm.initialize_gart_address_space();
  const AddressSpaceHandle process = register_byte_address_space(gpu_vm, 7, 0x11);
  ASSERT_TRUE(gart);
  ASSERT_TRUE(process);

  auto physical = std::make_shared<ByteAddressSpace>(0x22);
  ASSERT_TRUE(gpu_vm.publish_gart(
      {.page_table_base = 0x1000, .aperture_start = 0x100000000, .aperture_end = 0x100000fff},
      physical));
  const std::optional<AddressSpaceInfo> published = gpu_vm.lookup(gart);
  ASSERT_TRUE(published);
  ASSERT_TRUE(published->ready);

  ASSERT_TRUE(gpu_vm.retain_queue(gart));
  EXPECT_FALSE(gpu_vm.clear_gart_binding()) << "a live queue lost the GART binding it retained";
  ASSERT_TRUE(gpu_vm.release_queue(gart));
  ASSERT_TRUE(gpu_vm.clear_gart_binding());

  const std::optional<AddressSpaceInfo> cleared = gpu_vm.lookup(gart);
  ASSERT_TRUE(cleared);
  EXPECT_FALSE(cleared->ready);
  EXPECT_EQ(cleared->translation_epoch, published->translation_epoch + 1);
  EXPECT_EQ(gpu_vm.gart_address_space(), gart);
  std::array<std::byte, 1> value{};
  EXPECT_EQ(gpu_vm.read(gart, 0x100000000, value), VmAccessOutcome::Unavailable);
  EXPECT_EQ(gpu_vm.find_vmid(7), process);
  EXPECT_TRUE(gpu_vm.lookup(process));
  EXPECT_EQ(gpu_vm.active_address_spaces(), 2u);

  ASSERT_TRUE(gpu_vm.publish_gart(
      {.page_table_base = 0x2000, .aperture_start = 0x100000000, .aperture_end = 0x100000fff},
      physical));
  EXPECT_EQ(gpu_vm.gart_address_space(), gart);
  ASSERT_TRUE(gpu_vm.lookup(gart));
  EXPECT_TRUE(gpu_vm.lookup(gart)->ready);
  EXPECT_TRUE(gpu_vm.unregister_address_space(process));
}

TEST(GpuVmService, ReusedAddressSpaceSlotHasANewCacheNamespace) {
  GpuMemory memory("memory");
  GpuVm gpu_vm(&memory);
  const AddressSpaceHandle old_handle = register_byte_address_space(gpu_vm, 7, 0x11);
  ASSERT_TRUE(old_handle);
  const std::optional<GpuVmAccess> old_access = gpu_vm.snapshot(old_handle);
  ASSERT_TRUE(old_access);
  const VmCacheNamespace old_namespace = old_access->cache_namespace();

  ASSERT_TRUE(gpu_vm.unregister_address_space(old_handle));
  const AddressSpaceHandle new_handle = register_byte_address_space(gpu_vm, 8, 0x22);
  ASSERT_TRUE(new_handle);
  const std::optional<GpuVmAccess> new_access = gpu_vm.snapshot(new_handle);
  ASSERT_TRUE(new_access);

  EXPECT_EQ(new_handle.slot, old_handle.slot);
  EXPECT_NE(new_access->cache_namespace(), old_namespace);
  EXPECT_FALSE(gpu_vm.snapshot(old_handle));
  EXPECT_TRUE(gpu_vm.unregister_address_space(new_handle));
}

TEST(GpuVmService, ResetInvalidatesQueueAndAddressSpaceHandles) {
  GpuMemory memory("memory");
  GpuVm gpu_vm(&memory);
  QueueService queues(gpu_vm);
  CommandProcessor command_processor("cp");
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  QueueHandle queue = queues.create(queue_info(address_space, command_processor, 1));
  const uint64_t old_queue_epoch = queues.reset_epoch();
  const uint64_t old_vm_epoch = gpu_vm.reset_epoch();

  ASSERT_TRUE(queues.reset());

  EXPECT_EQ(queues.active_queues(), 0u);
  EXPECT_EQ(gpu_vm.active_address_spaces(), 0u);
  EXPECT_EQ(queues.reset_epoch(), old_queue_epoch + 1);
  EXPECT_EQ(gpu_vm.reset_epoch(), old_vm_epoch + 1);
  EXPECT_FALSE(queues.notify_doorbell(queue, 1));
  EXPECT_FALSE(gpu_vm.lookup(address_space));
}

TEST(GpuVmService, DestroyRevokesHandleAndDrainsAdmittedUpdateBeforeDetach) {
  GpuMemory memory("memory");
  GpuVm gpu_vm(&memory);
  QueueService queues(gpu_vm);
  auto backend = std::make_shared<BlockingQueueBackend>(BlockedQueueOperation::Update);
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  QueueHandle queue = queues.create(queue_info(address_space, backend, 1));
  ASSERT_TRUE(queue);
  std::atomic<bool> detach_saw_revoked_handle = false;
  std::atomic<bool> detach_saw_retained_address_space = false;
  backend->set_detach_observer([&]() {
    detach_saw_revoked_handle.store(!queues.contains(queue), std::memory_order_relaxed);
    const std::optional<AddressSpaceInfo> info = gpu_vm.lookup(address_space);
    detach_saw_retained_address_space.store(info && info->queue_references == 1,
                                            std::memory_order_relaxed);
  });

  std::future<bool> update =
      std::async(std::launch::async, [&]() { return queues.update(queue, 0x1000, 4096, 100); });
  if (!backend->wait_until_callback_is_blocked()) {
    backend->release_callback();
    (void)update.get();
    FAIL() << "the update backend callback was not admitted";
  }
  std::future<bool> destroy =
      std::async(std::launch::async, [&]() { return queues.destroy(queue); });

  if (!wait_until_queue_is_revoked(queues, queue)) {
    backend->release_callback();
    (void)update.get();
    (void)destroy.get();
    FAIL() << "destroy did not revoke the queue handle";
  }
  EXPECT_EQ(destroy.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  EXPECT_EQ(backend->detach_calls(), 0u);
  EXPECT_FALSE(queues.update(queue, 0x2000, 4096, 100));

  backend->release_callback();
  EXPECT_TRUE(update.get());
  EXPECT_TRUE(destroy.get());
  EXPECT_EQ(backend->detach_calls(), 1u);
  EXPECT_TRUE(detach_saw_revoked_handle.load(std::memory_order_relaxed));
  EXPECT_TRUE(detach_saw_retained_address_space.load(std::memory_order_relaxed));
  EXPECT_FALSE(queues.destroy(queue));
  EXPECT_EQ(backend->detach_calls(), 1u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, ResetClosesAdmissionAndDrainsAdmittedDoorbellBeforeDetach) {
  GpuMemory memory("memory");
  GpuVm gpu_vm(&memory);
  QueueService queues(gpu_vm);
  auto backend = std::make_shared<BlockingQueueBackend>(BlockedQueueOperation::Doorbell);
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  QueueHandle queue = queues.create(queue_info(address_space, backend, 1));
  ASSERT_TRUE(queue);

  std::future<bool> doorbell =
      std::async(std::launch::async, [&]() { return queues.notify_doorbell(queue, 7); });
  if (!backend->wait_until_callback_is_blocked()) {
    backend->release_callback();
    (void)doorbell.get();
    FAIL() << "the doorbell backend callback was not admitted";
  }
  std::future<bool> reset = std::async(std::launch::async, [&]() { return queues.reset(); });

  if (!wait_until_queue_is_revoked(queues, queue)) {
    backend->release_callback();
    (void)doorbell.get();
    reset.get();
    FAIL() << "reset did not revoke the queue handle";
  }
  EXPECT_EQ(reset.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  EXPECT_EQ(backend->detach_calls(), 0u);
  EXPECT_FALSE(queues.create(queue_info(address_space, backend, 2)));
  EXPECT_EQ(backend->attach_calls(), 1u);

  backend->release_callback();
  EXPECT_TRUE(doorbell.get());
  EXPECT_TRUE(reset.get());
  EXPECT_EQ(backend->detach_calls(), 1u);
  EXPECT_EQ(queues.active_queues(), 0u);
  EXPECT_FALSE(queues.notify_doorbell(queue, 8));
  EXPECT_FALSE(gpu_vm.lookup(address_space));
}

TEST(GpuVmService, FailedMutatingAttachRollsBackBeforeReleasingTheVmReference) {
  GpuMemory memory("memory");
  GpuVm gpu_vm(&memory);
  QueueService queues(gpu_vm);
  auto backend = std::make_shared<MutatingThrowQueueBackend>();
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);

  EXPECT_THROW((void)queues.create(queue_info(address_space, backend, 1)), std::runtime_error);

  EXPECT_FALSE(backend->attached());
  EXPECT_EQ(backend->rollback_calls(), 1u);
  EXPECT_EQ(backend->detach_calls(), 0u);
  EXPECT_EQ(queues.active_queues(), 0u);
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, DuplicateBackendKeyLeavesTheOriginalQueueUsable) {
  GpuMemory memory("memory");
  GpuVm gpu_vm(&memory);
  QueueService queues(gpu_vm);
  CommandProcessor command_processor("cp");
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  const QueueCreateInfo info = queue_info(address_space, command_processor, 1);

  const QueueHandle original = queues.create(info);
  ASSERT_TRUE(original);
  EXPECT_THROW((void)queues.create(info), std::runtime_error);

  EXPECT_TRUE(queues.contains(original));
  EXPECT_EQ(queues.active_queues(), 1u);
  EXPECT_EQ(command_processor.registered_queue_count_for_test(), 1u);
  EXPECT_TRUE(queues.update(original, 0x2000, 8192, 75));
  EXPECT_TRUE(queues.destroy(original));
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, ResetPublishesItsEpochAfterBlockedCreateRollsBack) {
  GpuMemory memory("memory");
  GpuVm gpu_vm(&memory);
  QueueService queues(gpu_vm);
  auto backend = std::make_shared<BlockingQueueBackend>(BlockedQueueOperation::Attach);
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  const uint64_t old_epoch = queues.reset_epoch();

  std::future<QueueHandle> create = std::async(
      std::launch::async, [&]() { return queues.create(queue_info(address_space, backend, 1)); });
  if (!backend->wait_until_callback_is_blocked()) {
    backend->release_callback();
    (void)create.get();
    FAIL() << "the attach backend callback did not block";
  }
  std::future<bool> reset = std::async(std::launch::async, [&]() { return queues.reset(); });
  if (!wait_until_create_admission_closes(queues)) {
    backend->release_callback();
    (void)create.get();
    (void)reset.get();
    FAIL() << "reset did not close queue admission";
  }

  EXPECT_EQ(queues.reset_epoch(), old_epoch);
  EXPECT_EQ(queues.active_queues(), 0u);
  EXPECT_EQ(reset.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

  backend->release_callback();
  EXPECT_FALSE(create.get());
  EXPECT_TRUE(reset.get());
  EXPECT_EQ(backend->attach_calls(), 1u);
  EXPECT_EQ(backend->rollback_calls(), 0u);
  EXPECT_EQ(backend->detach_calls(), 1u);
  EXPECT_EQ(queues.reset_epoch(), old_epoch + 1);
  EXPECT_EQ(queues.active_queues(), 0u);
  EXPECT_FALSE(gpu_vm.lookup(address_space));
}

TEST(CommandProcessorQueueRegistration, FailedFanoutRollsBackOnlyNewReplicas) {
  CommandProcessor owner("owner");
  CommandProcessor first_peer("first-peer");
  CommandProcessor rejecting_peer("rejecting-peer");
  owner.set_xcd_topology(0, {&owner, &first_peer, &rejecting_peer});

  HwQueue existing{};
  existing.process_id = 7;
  existing.queue_id = 11;
  rejecting_peer.register_queue(existing);

  HwQueue fanout = existing;
  fanout.xcd_fanout = true;
  EXPECT_THROW(owner.register_queue(fanout), std::runtime_error);

  EXPECT_EQ(owner.registered_queue_count_for_test(), 0u);
  EXPECT_EQ(first_peer.registered_queue_count_for_test(), 0u);
  EXPECT_EQ(rejecting_peer.registered_queue_count_for_test(), 1u);
  rejecting_peer.unregister_queue(existing.queue_id, existing.process_id);
}

} // namespace
} // namespace rocjitsu::amdgpu
