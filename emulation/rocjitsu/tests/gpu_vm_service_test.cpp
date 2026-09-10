// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/aql_packet_types.h"
#include "rocjitsu/vm/amdgpu/aql_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/soc.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace rocjitsu::amdgpu {

class GpuQueueRegistryTestAccess {
public:
  static void fail_next_registry_allocation(GpuQueueRegistry &registry) {
    std::lock_guard lock(registry.mutex_);
    registry.fail_next_registry_allocation_ = true;
  }
};

class CommandProcessorCloseTestAccess {
public:
  static bool fetch_first_queue(CommandProcessor &command_processor) {
    std::lock_guard<std::recursive_mutex> lock(command_processor.hw_queue_mutex_);
    if (command_processor.aql_queues_.size() != 1)
      return false;
    command_processor.fetch_from_queue(command_processor.aql_queues_.front(), 0);
    return true;
  }

  static bool retain_cursor_publication(CommandProcessor &command_processor, GpuVmAccess access,
                                        uint64_t cursor) {
    std::lock_guard<std::recursive_mutex> lock(command_processor.hw_queue_mutex_);
    if (command_processor.aql_queues_.size() != 1)
      return false;
    command_processor.aql_queues_.front().read_pointer_journal.retire(cursor, std::move(access));
    return true;
  }

  static bool retry_cursor_publication(CommandProcessor &command_processor) {
    std::lock_guard<std::recursive_mutex> lock(command_processor.hw_queue_mutex_);
    if (command_processor.aql_queues_.size() != 1)
      return false;
    AqlQueueRecord &queue = command_processor.aql_queues_.front();
    queue.faulted = true;
    command_processor.fetch_from_queue(queue, 0);
    return !queue.read_pointer_journal.publication_pending();
  }

  static bool retain_idle_completion_publication(CommandProcessor &command_processor) {
    std::lock_guard<std::recursive_mutex> lock(command_processor.hw_queue_mutex_);
    if (command_processor.aql_queues_.size() != 1)
      return false;
    AqlQueueRecord &queue = command_processor.aql_queues_.front();
    queue.idle_publication.phase = QueueIdlePublicationPhase::StoreMailbox;
    queue.publication_retry_pending = true;
    return true;
  }

  static bool clear_idle_completion_publication(CommandProcessor &command_processor) {
    std::lock_guard<std::recursive_mutex> lock(command_processor.hw_queue_mutex_);
    if (command_processor.aql_queues_.size() != 1)
      return false;
    AqlQueueRecord &queue = command_processor.aql_queues_.front();
    queue.idle_publication.reset();
    queue.publication_retry_pending = false;
    return true;
  }

  static bool mark_publication_faulted(CommandProcessor &command_processor) {
    std::lock_guard<std::recursive_mutex> lock(command_processor.hw_queue_mutex_);
    if (command_processor.aql_queues_.size() != 1)
      return false;
    command_processor.aql_queues_.front().publication_faulted = true;
    return true;
  }
};

namespace {

TEST(IdentityAddressSpaceTranslator, ProducesPageBoundedReadWriteExecuteTranslations) {
  IdentityAddressSpaceTranslator translator;
  constexpr uint64_t kAddress = IdentityAddressSpaceTranslator::kPageSize - 7;

  for (const VmAccessKind access :
       {VmAccessKind::Read, VmAccessKind::Write, VmAccessKind::Execute, VmAccessKind::Atomic}) {
    const VmTranslationResult translated = translator.translate(kAddress, 16, access);
    ASSERT_TRUE(translated);
    EXPECT_EQ(translated.translation.domain, VmMemoryDomain::Local);
    EXPECT_EQ(translated.translation.address, kAddress);
    EXPECT_EQ(translated.translation.contiguous_bytes, 7u);
    EXPECT_TRUE(translated.translation.permissions.allows(access));
  }

  EXPECT_EQ(translator.translate(0, 0, VmAccessKind::Read).outcome, VmAccessOutcome::Malformed);
  EXPECT_EQ(
      translator.translate(std::numeric_limits<uint64_t>::max(), 2, VmAccessKind::Write).outcome,
      VmAccessOutcome::Malformed);
}

TEST(GpuMemoryPhysicalAccess, SupportsTranslatedDataAndAtomicOperations) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  auto translator = std::make_shared<IdentityAddressSpaceTranslator>();
  auto physical = std::make_shared<GpuMemoryPhysicalAccess>(memory);
  const AddressSpaceHandle address_space =
      gpu_vm.register_unrouted_address_space(0, translator, physical);
  ASSERT_TRUE(address_space);

  const std::array<std::byte, 8> initial = {std::byte{0x08}, std::byte{0x07}, std::byte{0x06},
                                            std::byte{0x05}, std::byte{0x04}, std::byte{0x03},
                                            std::byte{0x02}, std::byte{0x01}};
  ASSERT_EQ(gpu_vm.write(address_space, 0x1ffcu, initial), VmAccessOutcome::Complete);

  std::array<std::byte, 8> readback{};
  ASSERT_EQ(gpu_vm.read(address_space, 0x1ffcu, readback), VmAccessOutcome::Complete);
  EXPECT_EQ(readback, initial);

  ASSERT_EQ(gpu_vm.atomic_store(address_space, 0x2008, sizeof(uint64_t), 0x1122334455667788ULL),
            VmAccessOutcome::Complete);
  const AtomicLoadResult loaded = gpu_vm.atomic_load(address_space, 0x2008, sizeof(uint64_t));
  ASSERT_EQ(loaded.outcome, VmAccessOutcome::Complete);
  EXPECT_EQ(loaded.value, 0x1122334455667788ULL);

  const AtomicCompareExchangeResult exchanged = gpu_vm.compare_exchange(
      address_space, 0x2008, sizeof(uint64_t), 0x1122334455667788ULL, 0xaabbccddeeff0011ULL);
  EXPECT_EQ(exchanged.outcome, VmAccessOutcome::Complete);
  EXPECT_EQ(exchanged.observed, 0x1122334455667788ULL);
  EXPECT_TRUE(exchanged.exchanged);
  EXPECT_EQ(gpu_vm.atomic_load(address_space, 0x2008, sizeof(uint64_t)).value,
            0xaabbccddeeff0011ULL);

  std::array<std::byte, 2> invalid{};
  EXPECT_EQ(physical->read(VmMemoryDomain::System, 0, invalid), VmAccessOutcome::Malformed);
  EXPECT_EQ(physical->write(VmMemoryDomain::Local, std::numeric_limits<uint64_t>::max(), invalid),
            VmAccessOutcome::Malformed);
  EXPECT_EQ(physical->atomic_load(VmMemoryDomain::Local, 3, sizeof(uint32_t)).outcome,
            VmAccessOutcome::Malformed);
  EXPECT_EQ(physical->resolve_host_pointer(VmMemoryDomain::Local, 0, 1), nullptr);
}

TEST(GpuVmService, BindingLeasePreventsUnregisterAndResetUntilItsOwnerReleasesIt) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  const AddressSpaceHandle address_space =
      gpu_vm.register_unrouted_address_space(11, std::make_shared<IdentityAddressSpaceTranslator>(),
                                             std::make_shared<GpuMemoryPhysicalAccess>(memory));
  ASSERT_TRUE(address_space);

  std::optional<GpuVmBindingLease> lease = gpu_vm.retain_binding(address_space);
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->handle(), address_space);
  EXPECT_EQ(lease->info().vmid, 11u);
  EXPECT_FALSE(gpu_vm.unregister_address_space(address_space));
  EXPECT_FALSE(gpu_vm.reset());

  lease.reset();
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, UnroutedIdentityBindingCoexistsWithRoutedVmidZero) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  const AddressSpaceHandle gart = gpu_vm.initialize_gart_address_space();
  auto translator = std::make_shared<IdentityAddressSpaceTranslator>();
  auto physical = std::make_shared<GpuMemoryPhysicalAccess>(memory);
  const AddressSpaceHandle internal =
      gpu_vm.register_unrouted_address_space(0, translator, physical);
  ASSERT_TRUE(gart);
  ASSERT_TRUE(internal);
  EXPECT_NE(internal, gart);
  EXPECT_EQ(gpu_vm.active_address_spaces(), 2u);
  ASSERT_TRUE(gpu_vm.find_vmid(0));
  EXPECT_EQ(*gpu_vm.find_vmid(0), gart);

  const std::array<std::byte, 4> value = {std::byte{0x12}, std::byte{0x34}, std::byte{0x56},
                                          std::byte{0x78}};
  const std::optional<GpuVmAccess> internal_access = gpu_vm.snapshot(internal);
  ASSERT_TRUE(internal_access);
  EXPECT_EQ(gpu_vm.write(internal, 0x4000, value), VmAccessOutcome::Complete);
  std::array<std::byte, 4> readback{};
  EXPECT_EQ(internal_access->read(0x4000, readback), VmAccessOutcome::Complete);
  EXPECT_EQ(readback, value);

  ASSERT_TRUE(gpu_vm.unregister_address_space(internal));
  EXPECT_EQ(gpu_vm.active_address_spaces(), 1u);
  ASSERT_TRUE(gpu_vm.find_vmid(0));
  EXPECT_EQ(*gpu_vm.find_vmid(0), gart);
  EXPECT_TRUE(gpu_vm.snapshot(gart));
  EXPECT_FALSE(gpu_vm.snapshot(gart)->info().ready);
  readback.fill(std::byte{0});
  EXPECT_EQ(internal_access->read(0x4000, readback), VmAccessOutcome::Complete);
  EXPECT_EQ(readback, value);

  EXPECT_TRUE(gpu_vm.reset());
  EXPECT_EQ(gpu_vm.active_address_spaces(), 0u);
  EXPECT_FALSE(gpu_vm.find_vmid(0));
}

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

  VmAccessOutcome atomic_store(VmMemoryDomain domain, uint64_t address, uint32_t width,
                               uint64_t value) override {
    if (domain != VmMemoryDomain::System || (width != 4 && width != 8) || address > bytes_.size() ||
        width > bytes_.size() - address) {
      return VmAccessOutcome::Faulted;
    }
    atomic_store_calls_.fetch_add(1, std::memory_order_relaxed);
    std::memcpy(bytes_.data() + address, &value, width);
    return VmAccessOutcome::Complete;
  }

  [[nodiscard]] std::size_t atomic_store_calls() const {
    return atomic_store_calls_.load(std::memory_order_relaxed);
  }

  void reset_atomic_store_calls() { atomic_store_calls_.store(0, std::memory_order_relaxed); }

private:
  std::vector<std::byte> bytes_;
  std::atomic<std::size_t> atomic_store_calls_ = 0;
};

AddressSpaceHandle register_byte_address_space(GpuVm &gpu_vm, uint32_t vmid, uint8_t value) {
  auto access = std::make_shared<ByteAddressSpace>(value);
  return gpu_vm.register_translated(vmid, access, access);
}

QueueRegistrationRequest queue_request(AddressSpaceHandle address_space,
                                       std::shared_ptr<QueueBindingFactory> binding_factory,
                                       uint32_t queue_id) {
  QueueRegistrationRequest request{};
  request.identity = {.address_space = address_space, .process_id = 7, .queue_id = queue_id};
  request.ring = {.base_address = 0x100,
                  .size_bytes = 64,
                  .consumer_pointer_address = 0x80,
                  .producer_pointer_address = 0x88};
  request.binding_factory = std::move(binding_factory);
  request.type = QueueType::Compute;
  request.packet_format = QueuePacketFormat::Aql;
  return request;
}

QueueRegistrationRequest queue_request(AddressSpaceHandle address_space, CommandProcessor &owner,
                                       uint32_t queue_id) {
  return queue_request(address_space, make_aql_queue_binding_factory(owner), queue_id);
}

enum class BlockedQueueOperation : uint8_t {
  None,
  CreateBinding,
  PrepareClose,
  Reconfigure,
  Submit,
  DestroyBinding,
};

class RecordingQueueBindingFactory final : public QueueBindingFactory {
private:
  struct BindingState {
    QueueIdentity identity;
    uint32_t reconfigure_calls = 0;
    uint32_t submit_calls = 0;
    uint32_t destruction_calls = 0;
  };

  struct SharedState {
    explicit SharedState(BlockedQueueOperation blocked_operation_value)
        : blocked_operation(blocked_operation_value) {}

    void block(BlockedQueueOperation operation) {
      if (operation != blocked_operation)
        return;
      std::unique_lock lock(mutex);
      callback_is_blocked = true;
      condition.notify_all();
      condition.wait(lock, [this]() { return release_callback; });
    }

    const BlockedQueueOperation blocked_operation;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::vector<std::shared_ptr<BindingState>> bindings;
    uint32_t create_binding_calls = 0;
    uint32_t rejected_creation_rollbacks = 0;
    uint32_t prepare_close_calls = 0;
    bool reject_next_creation = false;
    QueuePrepareCloseStatus close_status = QueuePrepareCloseStatus::Ready;
    bool transient_state = false;
    bool callback_is_blocked = false;
    bool release_callback = false;
    std::function<void()> destruction_observer;
  };

  class Binding final : public QueueBinding {
  public:
    Binding(std::shared_ptr<SharedState> shared, std::shared_ptr<BindingState> state)
        : shared_(std::move(shared)), state_(std::move(state)) {}

    ~Binding() override {
      {
        std::lock_guard lock(shared_->mutex);
        ++state_->destruction_calls;
        shared_->condition.notify_all();
      }
      shared_->block(BlockedQueueOperation::DestroyBinding);
      try {
        std::lock_guard lock(shared_->mutex);
        if (shared_->destruction_observer)
          shared_->destruction_observer();
      } catch (...) {
      }
    }

    QueuePrepareCloseStatus prepare_close() noexcept override {
      {
        std::lock_guard lock(shared_->mutex);
        ++shared_->prepare_close_calls;
      }
      shared_->block(BlockedQueueOperation::PrepareClose);
      std::lock_guard lock(shared_->mutex);
      return shared_->close_status;
    }

    QueueReconfigureStatus reconfigure(const QueueReconfigureRequest &) override {
      {
        std::lock_guard lock(shared_->mutex);
        ++state_->reconfigure_calls;
      }
      shared_->block(BlockedQueueOperation::Reconfigure);
      return QueueReconfigureStatus::Applied;
    }

    QueueSubmissionStatus submit_producer(uint64_t) override {
      {
        std::lock_guard lock(shared_->mutex);
        ++state_->submit_calls;
      }
      shared_->block(BlockedQueueOperation::Submit);
      return QueueSubmissionStatus::Accepted;
    }

  private:
    std::shared_ptr<SharedState> shared_;
    std::shared_ptr<BindingState> state_;
  };

public:
  explicit RecordingQueueBindingFactory(BlockedQueueOperation blocked_operation)
      : state_(std::make_shared<SharedState>(blocked_operation)) {}

  QueueBindingCreateResult create_binding(const QueueRegistrationRequest &request) override {
    {
      std::lock_guard lock(state_->mutex);
      ++state_->create_binding_calls;
      if (state_->reject_next_creation) {
        state_->reject_next_creation = false;
        state_->transient_state = true;
        state_->transient_state = false;
        ++state_->rejected_creation_rollbacks;
        return {};
      }
    }
    state_->block(BlockedQueueOperation::CreateBinding);
    auto binding_state = std::make_shared<BindingState>();
    binding_state->identity = request.identity;
    {
      std::lock_guard lock(state_->mutex);
      state_->bindings.push_back(binding_state);
    }
    return {.status = QueueBindingCreateStatus::Bound,
            .binding = std::make_unique<Binding>(state_, std::move(binding_state))};
  }

  void reject_next_creation() {
    std::lock_guard lock(state_->mutex);
    state_->reject_next_creation = true;
  }

  void set_destruction_observer(std::function<void()> observer) {
    std::lock_guard lock(state_->mutex);
    state_->destruction_observer = std::move(observer);
  }

  void set_close_allowed(bool allowed) {
    std::lock_guard lock(state_->mutex);
    state_->close_status = allowed ? QueuePrepareCloseStatus::Ready : QueuePrepareCloseStatus::Busy;
  }

  void set_close_status(QueuePrepareCloseStatus status) {
    std::lock_guard lock(state_->mutex);
    state_->close_status = status;
  }

  bool wait_until_callback_is_blocked() {
    std::unique_lock lock(state_->mutex);
    return state_->condition.wait_for(lock, std::chrono::seconds(5),
                                      [this]() { return state_->callback_is_blocked; });
  }

  void release_callback() {
    std::lock_guard lock(state_->mutex);
    state_->release_callback = true;
    state_->condition.notify_all();
  }

  uint32_t create_binding_calls() const {
    std::lock_guard lock(state_->mutex);
    return state_->create_binding_calls;
  }

  uint32_t prepare_close_calls() const {
    std::lock_guard lock(state_->mutex);
    return state_->prepare_close_calls;
  }

  uint32_t binding_count() const {
    std::lock_guard lock(state_->mutex);
    return static_cast<uint32_t>(state_->bindings.size());
  }

  QueueIdentity identity(uint32_t binding_index) const {
    std::lock_guard lock(state_->mutex);
    return state_->bindings.at(binding_index)->identity;
  }

  uint32_t reconfigure_calls(uint32_t binding_index) const {
    std::lock_guard lock(state_->mutex);
    return state_->bindings.at(binding_index)->reconfigure_calls;
  }

  uint32_t submit_calls(uint32_t binding_index) const {
    std::lock_guard lock(state_->mutex);
    return state_->bindings.at(binding_index)->submit_calls;
  }

  uint32_t destruction_calls(uint32_t binding_index) const {
    std::lock_guard lock(state_->mutex);
    return state_->bindings.at(binding_index)->destruction_calls;
  }

  uint32_t total_destruction_calls() const {
    std::lock_guard lock(state_->mutex);
    uint32_t total = 0;
    for (const std::shared_ptr<BindingState> &binding : state_->bindings)
      total += binding->destruction_calls;
    return total;
  }

  bool transient_state() const {
    std::lock_guard lock(state_->mutex);
    return state_->transient_state;
  }

  uint32_t rejected_creation_rollbacks() const {
    std::lock_guard lock(state_->mutex);
    return state_->rejected_creation_rollbacks;
  }

private:
  std::shared_ptr<SharedState> state_;
};

bool wait_until_queue_is_revoked(const GpuQueueRegistry &queues, QueueHandle handle) {
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (queues.contains(handle) && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  return !queues.contains(handle);
}

bool wait_until_registration_admission_closes(const GpuQueueRegistry &queues) {
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (queues.accepting_registrations_for_test() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  return !queues.accepting_registrations_for_test();
}

TEST(GpuVmService, TwoQueuesRetainOneAddressSpace) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  CommandProcessor command_processor("cp");
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x2a);
  command_processor.set_gpu_vm(&gpu_vm);
  const std::shared_ptr<QueueBindingFactory> binding_factory =
      make_aql_queue_binding_factory(command_processor);

  QueueHandle first = queues.register_queue(queue_request(address_space, binding_factory, 1));
  QueueHandle second = queues.register_queue(queue_request(address_space, binding_factory, 2));

  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(command_processor.queue_address_space_for_test(1, 7), address_space);
  EXPECT_EQ(command_processor.queue_address_space_for_test(2, 7), address_space);
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 2u);
  EXPECT_FALSE(gpu_vm.unregister_address_space(address_space));

  EXPECT_TRUE(queues.unregister_queue(first));
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 1u);
  std::array<std::byte, 1> value{};
  EXPECT_EQ(gpu_vm.read(address_space, 0, value), VmAccessOutcome::Complete);
  EXPECT_EQ(std::to_integer<uint8_t>(value[0]), 0x2a);

  EXPECT_TRUE(queues.unregister_queue(second));
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
  EXPECT_FALSE(gpu_vm.lookup(address_space));
}

TEST(GpuVmService, DestructorDestroysBindingsAndReleasesAddressSpaces) {
  GpuVm gpu_vm;
  const AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x2a);
  ASSERT_TRUE(address_space);
  auto binding_factory =
      std::make_shared<RecordingQueueBindingFactory>(BlockedQueueOperation::Reconfigure);

  {
    GpuQueueRegistry queues(gpu_vm);
    ASSERT_TRUE(queues.register_queue(queue_request(address_space, binding_factory, 1)));
    ASSERT_TRUE(gpu_vm.lookup(address_space));
    EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 1u);
  }

  EXPECT_EQ(binding_factory->destruction_calls(0), 1u);
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, QueueCannotRetainOneAddressSpaceAndRouteThroughAnotherVmid) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  CommandProcessor command_processor("cp");
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x2a);
  QueueRegistrationRequest mismatched = queue_request(address_space, command_processor, 1);
  mismatched.identity.process_id = 8;

  EXPECT_FALSE(queues.register_queue(mismatched));
  EXPECT_EQ(command_processor.registered_queue_count_for_test(), 0u);
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(CommandProcessorDoorbell, TransportNotificationDoesNotWaitForQueueExecutionLock) {
  GpuVm gpu_vm;
  CommandProcessor command_processor("cp");
  const AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  command_processor.set_gpu_vm(&gpu_vm);
  AqlQueueConfig queue{.address_space = address_space,
                       .process_id = 7,
                       .queue_id = 1,
                       .ring_base_va = 0x100,
                       .ring_size = 4096,
                       .read_ptr_va = 0x80,
                       .write_ptr_va = 0x88,
                       .last_doorbell = 4};
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
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(CommandProcessorDoorbell, StaleRegistrationCannotNotifyReusedQueueIdentity) {
  GpuVm gpu_vm;
  CommandProcessor command_processor("cp");
  const AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  command_processor.set_gpu_vm(&gpu_vm);
  AqlQueueConfig queue{.address_space = address_space,
                       .process_id = 7,
                       .queue_id = 1,
                       .ring_base_va = 0x100,
                       .ring_size = 4096,
                       .read_ptr_va = 0x80,
                       .write_ptr_va = 0x88,
                       .last_doorbell = 4};
  const uint64_t stale = command_processor.register_queue(queue);
  command_processor.unregister_queue(queue.queue_id, queue.process_id);

  const uint64_t current = command_processor.register_queue(queue);
  ASSERT_NE(current, stale);
  command_processor.notify_queue_doorbell(stale, 12);
  command_processor.drain_doorbell_inbox_for_test();
  EXPECT_EQ(command_processor.queue_last_doorbell_for_test(current), 4u);
  command_processor.unregister_queue(queue.queue_id, queue.process_id);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(CommandProcessorPacketFetch, UnretiredPacketDoesNotRepublishUnchangedCursor) {
  GpuVm gpu_vm;
  auto access = std::make_shared<ByteAddressSpace>(0);
  const AddressSpaceHandle address_space = gpu_vm.register_translated(7, access, access);
  ASSERT_TRUE(address_space);

  constexpr uint64_t kReadPointerAddress = 0x80;
  constexpr uint64_t kWritePointerAddress = 0x88;
  constexpr uint64_t kRingAddress = 0x100;
  ASSERT_EQ(gpu_vm.atomic_store(address_space, kReadPointerAddress, sizeof(uint64_t), 0),
            VmAccessOutcome::Complete);
  ASSERT_EQ(gpu_vm.atomic_store(address_space, kWritePointerAddress, sizeof(uint64_t), 1),
            VmAccessOutcome::Complete);

  hsa_kernel_dispatch_packet_t unsupported{};
  unsupported.header = 7;
  ASSERT_EQ(gpu_vm.write(address_space, kRingAddress,
                         std::as_bytes(std::span(&unsupported, std::size_t{1}))),
            VmAccessOutcome::Complete);

  CommandProcessor command_processor("cp");
  command_processor.set_gpu_vm(&gpu_vm);
  const uint64_t registration = command_processor.register_queue({
      .address_space = address_space,
      .process_id = 7,
      .queue_id = 1,
      .ring_base_va = kRingAddress,
      .ring_size = static_cast<uint32_t>(kAqlPacketBytes),
      .read_ptr_va = kReadPointerAddress,
      .write_ptr_va = kWritePointerAddress,
      .doorbell_va = 0x90,
      .doorbell_mode = QueueDoorbellMode::VmPolled,
  });
  ASSERT_NE(registration, 0u);

  access->reset_atomic_store_calls();
  ASSERT_TRUE(CommandProcessorCloseTestAccess::fetch_first_queue(command_processor));

  EXPECT_EQ(access->atomic_store_calls(), 0u);
  EXPECT_TRUE(command_processor.queue_faulted_for_test(1, 7));

  EXPECT_TRUE(command_processor.unregister_queue_registration(registration));
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(CommandProcessorVmFault, FaultsEveryFanoutReplicaAndDropsPresentAndFutureWork) {
  GpuVm gpu_vm;
  CommandProcessor owner("owner");
  CommandProcessor peer("peer");
  const AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  owner.set_gpu_vm(&gpu_vm);
  peer.set_gpu_vm(&gpu_vm);
  owner.set_xcd_topology(0, {&owner, &peer});
  peer.set_xcd_topology(1, {&owner, &peer});

  constexpr uint32_t kProcessId = 7;
  constexpr uint32_t kQueueId = 41;
  constexpr uint32_t kDispatchId = 19;
  AqlQueueConfig queue{.address_space = address_space,
                       .process_id = kProcessId,
                       .queue_id = kQueueId,
                       .ring_base_va = 0x100,
                       .ring_size = 4096,
                       .read_ptr_va = 0x80,
                       .write_ptr_va = 0x88,
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
  peer.unregister_queue(kQueueId, kProcessId);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(CommandProcessorVmFault, RejectsNonterminalOrMismatchedNotifications) {
  GpuVm gpu_vm;
  CommandProcessor command_processor("cp");
  constexpr uint32_t kProcessId = 7;
  constexpr uint32_t kQueueId = 42;
  constexpr uint32_t kDispatchId = 23;
  const AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, kProcessId, 0x11);
  command_processor.set_gpu_vm(&gpu_vm);
  (void)command_processor.register_queue({.address_space = address_space,
                                          .process_id = kProcessId,
                                          .queue_id = kQueueId,
                                          .ring_base_va = 0x100,
                                          .ring_size = 4096,
                                          .read_ptr_va = 0x80,
                                          .write_ptr_va = 0x88});

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
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, ReusedSlotsRejectStaleHandles) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  CommandProcessor command_processor("cp");
  AddressSpaceHandle old_address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  command_processor.set_gpu_vm(&gpu_vm);
  QueueHandle old_queue =
      queues.register_queue(queue_request(old_address_space, command_processor, 1));

  ASSERT_TRUE(queues.unregister_queue(old_queue));
  ASSERT_TRUE(gpu_vm.unregister_address_space(old_address_space));
  AddressSpaceHandle new_address_space = register_byte_address_space(gpu_vm, 8, 0x22);
  QueueRegistrationRequest new_queue_request =
      queue_request(new_address_space, command_processor, 2);
  new_queue_request.identity.process_id = 8;
  QueueHandle new_queue = queues.register_queue(new_queue_request);

  EXPECT_EQ(new_address_space.slot, old_address_space.slot);
  EXPECT_NE(new_address_space.generation, old_address_space.generation);
  EXPECT_EQ(new_queue.slot, old_queue.slot);
  EXPECT_NE(new_queue.generation, old_queue.generation);
  EXPECT_FALSE(queues.submit_producer(old_queue, 1).found);
  EXPECT_FALSE(queues.unregister_queue(old_queue));
  EXPECT_FALSE(gpu_vm.lookup(old_address_space));
  EXPECT_TRUE(queues.contains(new_queue));

  EXPECT_TRUE(queues.unregister_queue(new_queue));
  EXPECT_TRUE(gpu_vm.unregister_address_space(new_address_space));
}

TEST(GpuVmService, ReplacingABindingPreservesIdentityAndAdvancesItsEpoch) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
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
  GpuVm gpu_vm;
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
  GpuVm gpu_vm;
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
  GpuVm gpu_vm;
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

TEST(GpuVmService, CloseAllInvalidatesQueueHandlesWithoutResettingAddressSpaces) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  CommandProcessor command_processor("cp");
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  command_processor.set_gpu_vm(&gpu_vm);
  QueueHandle queue = queues.register_queue(queue_request(address_space, command_processor, 1));
  const uint64_t old_queue_epoch = queues.lifecycle_epoch();
  const uint64_t old_vm_epoch = gpu_vm.reset_epoch();

  queues.close_all();

  EXPECT_EQ(queues.active_queues(), 0u);
  EXPECT_EQ(gpu_vm.active_address_spaces(), 1u);
  EXPECT_EQ(queues.lifecycle_epoch(), old_queue_epoch + 1);
  EXPECT_EQ(gpu_vm.reset_epoch(), old_vm_epoch);
  EXPECT_FALSE(queues.submit_producer(queue, 1).found);
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, UnregisterDrainsAdmittedReconfigureBeforeDestroyingBinding) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  auto binding_factory =
      std::make_shared<RecordingQueueBindingFactory>(BlockedQueueOperation::Reconfigure);
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  QueueHandle queue = queues.register_queue(queue_request(address_space, binding_factory, 1));
  ASSERT_TRUE(queue);
  std::atomic<bool> destruction_saw_revoked_handle = false;
  std::atomic<bool> destruction_saw_retained_address_space = false;
  binding_factory->set_destruction_observer([&]() {
    destruction_saw_revoked_handle.store(!queues.contains(queue), std::memory_order_relaxed);
    const std::optional<AddressSpaceInfo> info = gpu_vm.lookup(address_space);
    destruction_saw_retained_address_space.store(info && info->queue_references == 1,
                                                 std::memory_order_relaxed);
  });

  std::future<QueueReconfigureResult> reconfigure = std::async(std::launch::async, [&]() {
    return queues.reconfigure_queue(
        queue,
        {.ring_base_address = 0x1000, .ring_size_bytes = 4096, .scheduling_percentage = 100});
  });
  if (!binding_factory->wait_until_callback_is_blocked()) {
    binding_factory->release_callback();
    (void)reconfigure.get();
    FAIL() << "the reconfigure binding callback was not admitted";
  }
  std::future<QueueCloseResult> unregister =
      std::async(std::launch::async, [&]() { return queues.unregister_queue(queue); });

  if (!wait_until_queue_is_revoked(queues, queue)) {
    binding_factory->release_callback();
    (void)reconfigure.get();
    (void)unregister.get();
    FAIL() << "unregister did not revoke the queue handle";
  }
  EXPECT_EQ(unregister.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  EXPECT_EQ(binding_factory->destruction_calls(0), 0u);
  EXPECT_FALSE(queues
                   .reconfigure_queue(queue, {.ring_base_address = 0x2000,
                                              .ring_size_bytes = 4096,
                                              .scheduling_percentage = 100})
                   .found);

  binding_factory->release_callback();
  const QueueReconfigureResult result = reconfigure.get();
  EXPECT_TRUE(result.found);
  EXPECT_EQ(result.status, QueueReconfigureStatus::Applied);
  EXPECT_TRUE(unregister.get());
  EXPECT_EQ(binding_factory->destruction_calls(0), 1u);
  EXPECT_TRUE(destruction_saw_revoked_handle.load(std::memory_order_relaxed));
  EXPECT_TRUE(destruction_saw_retained_address_space.load(std::memory_order_relaxed));
  EXPECT_FALSE(queues.unregister_queue(queue));
  EXPECT_EQ(binding_factory->destruction_calls(0), 1u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, RejectedPreparedCloseKeepsTheSameQueueHandleLive) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  auto binding_factory =
      std::make_shared<RecordingQueueBindingFactory>(BlockedQueueOperation::None);
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  QueueHandle queue = queues.register_queue(queue_request(address_space, binding_factory, 1));
  ASSERT_TRUE(queue);
  binding_factory->set_close_allowed(false);

  EXPECT_FALSE(queues.unregister_queue(queue));
  EXPECT_TRUE(queues.contains(queue));
  EXPECT_EQ(queues.active_queues(), 1u);
  EXPECT_EQ(binding_factory->prepare_close_calls(), 1u);
  EXPECT_EQ(binding_factory->destruction_calls(0), 0u);
  EXPECT_TRUE(queues.submit_producer(queue, 1).found);
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 1u);

  binding_factory->set_close_allowed(true);
  EXPECT_TRUE(queues.unregister_queue(queue));
  EXPECT_FALSE(queues.contains(queue));
  EXPECT_EQ(binding_factory->prepare_close_calls(), 2u);
  EXPECT_EQ(binding_factory->destruction_calls(0), 1u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, ForceCloseBypassesPreparationAndDestroysTheBindingOnce) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  auto binding_factory =
      std::make_shared<RecordingQueueBindingFactory>(BlockedQueueOperation::None);
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  QueueHandle queue = queues.register_queue(queue_request(address_space, binding_factory, 1));
  ASSERT_TRUE(queue);
  binding_factory->set_close_status(QueuePrepareCloseStatus::Faulted);

  const QueueCloseResult closed = queues.unregister_queue(queue, QueueCloseMode::ForceCancel);

  EXPECT_TRUE(closed);
  EXPECT_EQ(closed.status, QueueCloseStatus::Closed);
  EXPECT_EQ(binding_factory->prepare_close_calls(), 0u);
  EXPECT_EQ(binding_factory->destruction_calls(0), 1u);
  EXPECT_FALSE(queues.contains(queue));
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, CloseAllWaitsForRejectedGracefulCloseThenForceClosesOnce) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  auto binding_factory =
      std::make_shared<RecordingQueueBindingFactory>(BlockedQueueOperation::PrepareClose);
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  QueueHandle queue = queues.register_queue(queue_request(address_space, binding_factory, 1));
  ASSERT_TRUE(queue);
  binding_factory->set_close_allowed(false);

  std::future<QueueCloseResult> graceful =
      std::async(std::launch::async, [&]() { return queues.unregister_queue(queue); });
  if (!binding_factory->wait_until_callback_is_blocked()) {
    binding_factory->release_callback();
    (void)graceful.get();
    FAIL() << "prepare_close was not admitted";
  }
  std::future<void> force_all = std::async(std::launch::async, [&]() { queues.close_all(); });
  EXPECT_EQ(force_all.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

  binding_factory->release_callback();
  const QueueCloseResult rejected = graceful.get();
  EXPECT_EQ(rejected.status, QueueCloseStatus::Busy);
  force_all.get();

  EXPECT_EQ(binding_factory->prepare_close_calls(), 1u);
  EXPECT_EQ(binding_factory->destruction_calls(0), 1u);
  EXPECT_FALSE(queues.contains(queue));
  EXPECT_EQ(queues.active_queues(), 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, CloseAllDrainsAdmittedSubmissionBeforeDestroyingBinding) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  auto binding_factory =
      std::make_shared<RecordingQueueBindingFactory>(BlockedQueueOperation::Submit);
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  QueueHandle queue = queues.register_queue(queue_request(address_space, binding_factory, 1));
  ASSERT_TRUE(queue);

  std::future<QueueSubmissionResult> submission =
      std::async(std::launch::async, [&]() { return queues.submit_producer(queue, 7); });
  if (!binding_factory->wait_until_callback_is_blocked()) {
    binding_factory->release_callback();
    (void)submission.get();
    FAIL() << "the producer binding callback was not admitted";
  }
  std::future<void> close_all = std::async(std::launch::async, [&]() { queues.close_all(); });

  if (!wait_until_queue_is_revoked(queues, queue)) {
    binding_factory->release_callback();
    (void)submission.get();
    close_all.get();
    FAIL() << "close_all did not revoke the queue handle";
  }
  EXPECT_EQ(close_all.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  EXPECT_EQ(binding_factory->destruction_calls(0), 0u);
  EXPECT_FALSE(queues.register_queue(queue_request(address_space, binding_factory, 2)));
  EXPECT_EQ(binding_factory->create_binding_calls(), 1u);

  binding_factory->release_callback();
  const QueueSubmissionResult result = submission.get();
  EXPECT_TRUE(result.found);
  EXPECT_EQ(result.status, QueueSubmissionStatus::Accepted);
  close_all.get();
  EXPECT_EQ(binding_factory->destruction_calls(0), 1u);
  EXPECT_EQ(queues.active_queues(), 0u);
  EXPECT_FALSE(queues.submit_producer(queue, 8).found);
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, RejectedMutatingBindingCreationRollsBackBeforeReleasingTheVmReference) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  auto binding_factory =
      std::make_shared<RecordingQueueBindingFactory>(BlockedQueueOperation::None);
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  binding_factory->reject_next_creation();

  EXPECT_FALSE(queues.register_queue(queue_request(address_space, binding_factory, 1)));

  EXPECT_FALSE(binding_factory->transient_state());
  EXPECT_EQ(binding_factory->rejected_creation_rollbacks(), 1u);
  EXPECT_EQ(binding_factory->binding_count(), 0u);
  EXPECT_EQ(binding_factory->total_destruction_calls(), 0u);
  EXPECT_EQ(queues.active_queues(), 0u);
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, FailedSecondAqlBindingCreationLeavesTheOriginalQueueUsable) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  CommandProcessor command_processor("cp");
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  command_processor.set_gpu_vm(&gpu_vm);
  const std::shared_ptr<QueueBindingFactory> binding_factory =
      make_aql_queue_binding_factory(command_processor);
  const QueueRegistrationRequest request = queue_request(address_space, binding_factory, 1);

  const QueueHandle original = queues.register_queue(request);
  ASSERT_TRUE(original);
  EXPECT_FALSE(queues.register_queue(request));

  EXPECT_TRUE(queues.contains(original));
  EXPECT_EQ(queues.active_queues(), 1u);
  EXPECT_EQ(command_processor.registered_queue_count_for_test(), 1u);
  const QueueReconfigureResult reconfigure = queues.reconfigure_queue(
      original,
      {.ring_base_address = 0x2000, .ring_size_bytes = 8192, .scheduling_percentage = 75});
  EXPECT_TRUE(reconfigure.found);
  EXPECT_EQ(reconfigure.status, QueueReconfigureStatus::Applied);
  EXPECT_TRUE(queues.unregister_queue(original));
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, AqlBindingFactoryRejectsInvalidRingLayoutsBeforeRegistration) {
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  CommandProcessor command_processor("cp");
  const AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  const std::shared_ptr<QueueBindingFactory> binding_factory =
      make_aql_queue_binding_factory(command_processor);
  const QueueRingLayout valid = queue_request(address_space, binding_factory, 1).ring;

  std::array<QueueRingLayout, 9> invalid{};
  invalid.fill(valid);
  invalid[0].base_address = 0;
  invalid[1].base_address += 1;
  invalid[2].size_bytes = 0;
  invalid[3].size_bytes = static_cast<uint32_t>(kAqlPacketBytes - 1);
  invalid[4].size_bytes = static_cast<uint32_t>(kAqlPacketBytes + 1);
  invalid[5].consumer_pointer_address = 0;
  invalid[6].consumer_pointer_address += 1;
  invalid[7].producer_pointer_address = 0;
  invalid[8].producer_pointer_address += 1;

  uint32_t queue_id = 1;
  for (const QueueRingLayout &layout : invalid) {
    SCOPED_TRACE(queue_id);
    QueueRegistrationRequest request = queue_request(address_space, binding_factory, queue_id++);
    request.ring = layout;
    EXPECT_FALSE(queues.register_queue(request));
    EXPECT_EQ(command_processor.registered_queue_count_for_test(), 0u);
  }

  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(CommandProcessorQueueRegistration, RejectsInvalidAqlRingLayoutsAtDirectBoundary) {
  GpuVm gpu_vm;
  CommandProcessor command_processor("cp");
  const AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  command_processor.set_gpu_vm(&gpu_vm);
  AqlQueueConfig valid{.address_space = address_space,
                       .process_id = 7,
                       .queue_id = 1,
                       .ring_base_va = 0x100,
                       .ring_size = 64,
                       .read_ptr_va = 0x80,
                       .write_ptr_va = 0x88};

  std::array<AqlQueueConfig, 10> invalid{};
  invalid.fill(valid);
  invalid[0].ring_base_va = 0;
  invalid[1].ring_base_va += 1;
  invalid[2].ring_size = 0;
  invalid[3].ring_size = static_cast<uint32_t>(kAqlPacketBytes - 1);
  invalid[4].ring_size = static_cast<uint32_t>(kAqlPacketBytes + 1);
  invalid[5].read_ptr_va = 0;
  invalid[6].read_ptr_va += 1;
  invalid[7].write_ptr_va = 0;
  invalid[8].write_ptr_va += 1;
  invalid[9].doorbell_mode = QueueDoorbellMode::VmPolled;

  for (AqlQueueConfig &config : invalid)
    EXPECT_EQ(command_processor.register_queue(std::move(config)), 0u);
  EXPECT_EQ(command_processor.registered_queue_count_for_test(), 0u);

  const uint64_t registration_id = command_processor.register_queue(valid);
  ASSERT_NE(registration_id, 0u);
  EXPECT_FALSE(command_processor.update_queue_registration(registration_id, 0, 64, 100));
  EXPECT_FALSE(command_processor.update_queue_registration(registration_id, 0x101, 64, 100));
  EXPECT_FALSE(command_processor.update_queue_registration(registration_id, 0x200, 0, 100));
  EXPECT_FALSE(command_processor.update_queue_registration(registration_id, 0x200, 63, 100));
  EXPECT_FALSE(command_processor.update_queue_registration(registration_id, 0x200, 65, 100));
  EXPECT_TRUE(command_processor.update_queue_registration(registration_id, 0x200, 64, 100));
  EXPECT_TRUE(command_processor.unregister_queue_registration(registration_id));
}

TEST(CommandProcessorQueueRegistration, RequiresAnAddressSpaceAndUsesTheConfiguredInternalOne) {
  AqlQueueConfig queue{.address_space = {},
                       .process_id = 0,
                       .queue_id = 1,
                       .ring_base_va = 0x100,
                       .ring_size = 64,
                       .read_ptr_va = 0x80,
                       .write_ptr_va = 0x88};
  CommandProcessor unbound("unbound");
  EXPECT_EQ(unbound.register_queue(queue), 0u);

  GpuMemory memory("memory");
  GpuVm gpu_vm;
  auto backing = std::make_shared<GpuMemoryPhysicalAccess>(memory);
  const AddressSpaceHandle internal = gpu_vm.register_unrouted_address_space(
      0, std::make_shared<IdentityAddressSpaceTranslator>(), backing, {}, true);
  ASSERT_TRUE(internal);

  CommandProcessor command_processor("cp");
  command_processor.set_gpu_vm(&gpu_vm, internal);
  const uint64_t registration_id = command_processor.register_queue(queue);
  ASSERT_NE(registration_id, 0u);
  EXPECT_EQ(command_processor.queue_address_space_for_test(queue.queue_id, queue.process_id),
            internal);
  EXPECT_FALSE(gpu_vm.unregister_address_space(internal));
  EXPECT_TRUE(command_processor.unregister_queue_registration(registration_id));
  EXPECT_TRUE(gpu_vm.unregister_address_space(internal));
}

TEST(SoCInternalAddressSpace, RefusesBackingReplacementWhileAnotherAddressSpaceExists) {
  GpuMemory original_memory("original");
  GpuMemory replacement_memory("replacement");
  GpuMemory external_memory("external");
  rocjitsu::SoC soc("soc", &original_memory);
  const AddressSpaceHandle original_internal = soc.internal_address_space();
  ASSERT_TRUE(original_internal);

  const AddressSpaceHandle external = soc.gpu_vm().register_unrouted_address_space(
      7, std::make_shared<IdentityAddressSpaceTranslator>(),
      std::make_shared<GpuMemoryPhysicalAccess>(external_memory));
  ASSERT_TRUE(external);

  EXPECT_FALSE(soc.set_memory(&replacement_memory));
  EXPECT_EQ(soc.memory(), &original_memory);
  EXPECT_EQ(soc.internal_address_space(), original_internal);
  EXPECT_TRUE(soc.gpu_vm().lookup(original_internal));

  ASSERT_TRUE(soc.gpu_vm().unregister_address_space(external));
  EXPECT_FALSE(soc.set_memory(&replacement_memory));
  EXPECT_EQ(soc.memory(), &original_memory);
  EXPECT_EQ(soc.internal_address_space(), original_internal);
}

TEST(SoCInternalAddressSpace, RefusesBackingReplacementWhileAnInternalAqlQueueExists) {
  GpuMemory original_memory("original");
  GpuMemory replacement_memory("replacement");
  CommandProcessor command_processor("cp");
  Xcd xcd("xcd");
  xcd.set_command_processor(&command_processor);
  rocjitsu::SoC soc("soc", &original_memory);
  soc.add_xcd(&xcd);

  const AddressSpaceHandle original_internal = soc.internal_address_space();
  ASSERT_TRUE(original_internal);
  EXPECT_EQ(command_processor.default_address_space(), original_internal);

  const uint64_t registration_id = command_processor.register_queue({
      .address_space = {},
      .process_id = 0,
      .queue_id = 1,
      .ring_base_va = 0x100,
      .ring_size = static_cast<uint32_t>(kAqlPacketBytes),
      .read_ptr_va = 0x80,
      .write_ptr_va = 0x88,
  });
  ASSERT_NE(registration_id, 0u);
  EXPECT_TRUE(command_processor.has_registered_queues());

  EXPECT_FALSE(soc.set_memory(&replacement_memory));
  EXPECT_EQ(soc.memory(), &original_memory);
  EXPECT_EQ(soc.internal_address_space(), original_internal);

  ASSERT_TRUE(command_processor.unregister_queue_registration(registration_id));
  EXPECT_FALSE(command_processor.has_registered_queues());
  EXPECT_FALSE(soc.set_memory(&replacement_memory));
  EXPECT_EQ(soc.memory(), &original_memory);
  EXPECT_EQ(command_processor.default_address_space(), original_internal);
}

TEST(SoCInternalAddressSpace, ReinstallsTheSameBackingAfterAnExplicitVmReset) {
  GpuMemory memory("memory");
  CommandProcessor command_processor("cp");
  Xcd xcd("xcd");
  xcd.set_command_processor(&command_processor);
  rocjitsu::SoC soc("soc", &memory);
  soc.add_xcd(&xcd);

  const AddressSpaceHandle original_internal = soc.internal_address_space();
  ASSERT_TRUE(original_internal);
  ASSERT_TRUE(soc.gpu_vm().reset());
  EXPECT_FALSE(soc.gpu_vm().lookup(original_internal));

  ASSERT_TRUE(soc.set_memory(&memory));
  const AddressSpaceHandle replacement_internal = soc.internal_address_space();
  EXPECT_TRUE(replacement_internal);
  EXPECT_NE(replacement_internal, original_internal);
  EXPECT_EQ(command_processor.default_address_space(), replacement_internal);
}

TEST(GpuVmService, AqlReconfigureRejectsInvalidRingGeometry) {
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  CommandProcessor command_processor("cp");
  const AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  command_processor.set_gpu_vm(&gpu_vm);
  const std::shared_ptr<QueueBindingFactory> binding_factory =
      make_aql_queue_binding_factory(command_processor);
  const QueueHandle queue = queues.register_queue(queue_request(address_space, binding_factory, 1));
  ASSERT_TRUE(queue);

  constexpr std::array invalid{
      QueueReconfigureRequest{
          .ring_base_address = 0, .ring_size_bytes = 64, .scheduling_percentage = 100},
      QueueReconfigureRequest{
          .ring_base_address = 0x101, .ring_size_bytes = 64, .scheduling_percentage = 100},
      QueueReconfigureRequest{
          .ring_base_address = 0x200, .ring_size_bytes = 0, .scheduling_percentage = 100},
      QueueReconfigureRequest{
          .ring_base_address = 0x200, .ring_size_bytes = 63, .scheduling_percentage = 100},
      QueueReconfigureRequest{
          .ring_base_address = 0x200, .ring_size_bytes = 65, .scheduling_percentage = 100},
  };
  for (const QueueReconfigureRequest request : invalid) {
    const QueueReconfigureResult result = queues.reconfigure_queue(queue, request);
    EXPECT_TRUE(result.found);
    EXPECT_EQ(result.status, QueueReconfigureStatus::Invalid);
  }

  const QueueReconfigureResult disabled = queues.reconfigure_queue(
      queue, {.ring_base_address = 0x200, .ring_size_bytes = 64, .scheduling_percentage = 0});
  EXPECT_TRUE(disabled.found);
  EXPECT_EQ(disabled.status, QueueReconfigureStatus::Disabled);

  EXPECT_TRUE(queues.unregister_queue(queue));
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, GracefulAqlCloseWaitsForRetiredCursorPublication) {
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  CommandProcessor command_processor("cp");
  const AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  command_processor.set_gpu_vm(&gpu_vm);
  const QueueHandle queue =
      queues.register_queue(queue_request(address_space, command_processor, 1));
  ASSERT_TRUE(queue);
  std::optional<GpuVmAccess> access = gpu_vm.snapshot(address_space);
  ASSERT_TRUE(access);
  ASSERT_TRUE(CommandProcessorCloseTestAccess::retain_cursor_publication(command_processor,
                                                                         std::move(*access), 1));

  const QueueCloseResult blocked = queues.unregister_queue(queue);
  EXPECT_TRUE(blocked.found);
  EXPECT_EQ(blocked.status, QueueCloseStatus::Busy);
  EXPECT_TRUE(queues.contains(queue));
  EXPECT_EQ(command_processor.registered_queue_count_for_test(), 1u);

  ASSERT_TRUE(CommandProcessorCloseTestAccess::retry_cursor_publication(command_processor));
  EXPECT_TRUE(queues.unregister_queue(queue));
  EXPECT_EQ(command_processor.registered_queue_count_for_test(), 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, GracefulAqlCloseWaitsForCompletionPublicationRetry) {
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  CommandProcessor command_processor("cp");
  const AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  command_processor.set_gpu_vm(&gpu_vm);
  const QueueHandle queue =
      queues.register_queue(queue_request(address_space, command_processor, 1));
  ASSERT_TRUE(queue);
  ASSERT_TRUE(
      CommandProcessorCloseTestAccess::retain_idle_completion_publication(command_processor));

  const QueueCloseResult blocked = queues.unregister_queue(queue);
  EXPECT_TRUE(blocked.found);
  EXPECT_EQ(blocked.status, QueueCloseStatus::Busy);
  EXPECT_TRUE(queues.contains(queue));
  EXPECT_EQ(command_processor.registered_queue_count_for_test(), 1u);

  ASSERT_TRUE(
      CommandProcessorCloseTestAccess::clear_idle_completion_publication(command_processor));
  EXPECT_TRUE(queues.unregister_queue(queue));
  EXPECT_EQ(command_processor.registered_queue_count_for_test(), 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, TerminalAqlPublicationFailureRequiresForceCancel) {
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  CommandProcessor command_processor("cp");
  const AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  command_processor.set_gpu_vm(&gpu_vm);
  const QueueHandle queue =
      queues.register_queue(queue_request(address_space, command_processor, 1));
  ASSERT_TRUE(queue);
  ASSERT_TRUE(CommandProcessorCloseTestAccess::mark_publication_faulted(command_processor));

  const QueueCloseResult faulted = queues.unregister_queue(queue);
  EXPECT_TRUE(faulted.found);
  EXPECT_EQ(faulted.status, QueueCloseStatus::Faulted);
  EXPECT_TRUE(queues.contains(queue));
  EXPECT_EQ(command_processor.registered_queue_count_for_test(), 1u);

  const QueueCloseResult canceled = queues.unregister_queue(queue, QueueCloseMode::ForceCancel);
  EXPECT_EQ(canceled.status, QueueCloseStatus::Closed);
  EXPECT_EQ(command_processor.registered_queue_count_for_test(), 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, StaleAqlBindingCannotChangeOrRemoveReusedQueueIdentity) {
  GpuVm gpu_vm;
  CommandProcessor command_processor("cp");
  const AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  command_processor.set_gpu_vm(&gpu_vm);
  const std::shared_ptr<QueueBindingFactory> binding_factory =
      make_aql_queue_binding_factory(command_processor);
  const QueueRegistrationRequest request = queue_request(address_space, binding_factory, 1);

  QueueBindingCreateResult stale = binding_factory->create_binding(request);
  ASSERT_TRUE(stale);
  command_processor.unregister_queue(request.identity.queue_id, request.identity.process_id);

  QueueBindingCreateResult current = binding_factory->create_binding(request);
  ASSERT_TRUE(current);
  ASSERT_EQ(command_processor.registered_queue_count_for_test(), 1u);
  EXPECT_FALSE(command_processor.queue_runtime_suspended_for_test(request.identity.queue_id,
                                                                  request.identity.process_id));

  EXPECT_EQ(stale.binding->reconfigure(
                {.ring_base_address = 0x2000, .ring_size_bytes = 4096, .scheduling_percentage = 0}),
            QueueReconfigureStatus::Stale);
  EXPECT_FALSE(command_processor.queue_runtime_suspended_for_test(request.identity.queue_id,
                                                                  request.identity.process_id));

  stale.binding.reset();
  EXPECT_EQ(command_processor.registered_queue_count_for_test(), 1u);

  current.binding.reset();
  EXPECT_EQ(command_processor.registered_queue_count_for_test(), 0u);
}

TEST(GpuVmService, ReusableBindingFactoryKeepsIndependentBindingsAcrossARejectedCreation) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  auto binding_factory =
      std::make_shared<RecordingQueueBindingFactory>(BlockedQueueOperation::None);
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);

  const QueueHandle first = queues.register_queue(queue_request(address_space, binding_factory, 1));
  ASSERT_TRUE(first);
  binding_factory->reject_next_creation();
  EXPECT_FALSE(queues.register_queue(queue_request(address_space, binding_factory, 99)));
  const QueueHandle second =
      queues.register_queue(queue_request(address_space, binding_factory, 2));
  ASSERT_TRUE(second);
  EXPECT_EQ(binding_factory->create_binding_calls(), 3u);
  ASSERT_EQ(binding_factory->binding_count(), 2u);
  EXPECT_EQ(binding_factory->identity(0).queue_id, 1u);
  EXPECT_EQ(binding_factory->identity(1).queue_id, 2u);

  const QueueSubmissionResult submission = queues.submit_producer(first, 11);
  const QueueReconfigureResult reconfigure = queues.reconfigure_queue(
      second, {.ring_base_address = 0x2000, .ring_size_bytes = 4096, .scheduling_percentage = 100});
  EXPECT_TRUE(submission.found);
  EXPECT_EQ(submission.status, QueueSubmissionStatus::Accepted);
  EXPECT_TRUE(reconfigure.found);
  EXPECT_EQ(reconfigure.status, QueueReconfigureStatus::Applied);
  EXPECT_EQ(binding_factory->submit_calls(0), 1u);
  EXPECT_EQ(binding_factory->submit_calls(1), 0u);
  EXPECT_EQ(binding_factory->reconfigure_calls(0), 0u);
  EXPECT_EQ(binding_factory->reconfigure_calls(1), 1u);

  EXPECT_TRUE(queues.unregister_queue(first));
  queues.close_all();
  EXPECT_EQ(binding_factory->destruction_calls(0), 1u);
  EXPECT_EQ(binding_factory->destruction_calls(1), 1u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, CloseAllPublishesItsEpochAfterBlockedRegistrationRollsBack) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  auto binding_factory =
      std::make_shared<RecordingQueueBindingFactory>(BlockedQueueOperation::CreateBinding);
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  const uint64_t old_epoch = queues.lifecycle_epoch();

  std::future<QueueHandle> registration = std::async(std::launch::async, [&]() {
    return queues.register_queue(queue_request(address_space, binding_factory, 1));
  });
  if (!binding_factory->wait_until_callback_is_blocked()) {
    binding_factory->release_callback();
    (void)registration.get();
    FAIL() << "the binding factory callback did not block";
  }
  std::future<void> close_all = std::async(std::launch::async, [&]() { queues.close_all(); });
  if (!wait_until_registration_admission_closes(queues)) {
    binding_factory->release_callback();
    (void)registration.get();
    close_all.get();
    FAIL() << "close_all did not close queue admission";
  }

  EXPECT_EQ(queues.lifecycle_epoch(), old_epoch);
  EXPECT_EQ(queues.active_queues(), 0u);
  EXPECT_EQ(close_all.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

  binding_factory->release_callback();
  EXPECT_FALSE(registration.get());
  close_all.get();
  EXPECT_EQ(binding_factory->create_binding_calls(), 1u);
  EXPECT_EQ(binding_factory->binding_count(), 1u);
  EXPECT_EQ(binding_factory->destruction_calls(0), 1u);
  EXPECT_EQ(queues.lifecycle_epoch(), old_epoch + 1);
  EXPECT_EQ(queues.active_queues(), 0u);
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(GpuVmService, CloseAllWaitsForFailedRegistrationCleanupWithoutResettingVm) {
  GpuMemory memory("memory");
  GpuVm gpu_vm;
  GpuQueueRegistry queues(gpu_vm);
  auto binding_factory =
      std::make_shared<RecordingQueueBindingFactory>(BlockedQueueOperation::DestroyBinding);
  AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  const uint64_t old_queue_epoch = queues.lifecycle_epoch();
  const uint64_t old_vm_epoch = gpu_vm.reset_epoch();

  GpuQueueRegistryTestAccess::fail_next_registry_allocation(queues);
  std::future<QueueHandle> registration = std::async(std::launch::async, [&]() {
    return queues.register_queue(queue_request(address_space, binding_factory, 1));
  });
  if (!binding_factory->wait_until_callback_is_blocked()) {
    binding_factory->release_callback();
    try {
      (void)registration.get();
    } catch (...) {
    }
    FAIL() << "failed registration did not enter binding cleanup";
  }

  std::future<void> close_all = std::async(std::launch::async, [&]() { queues.close_all(); });
  if (!wait_until_registration_admission_closes(queues)) {
    binding_factory->release_callback();
    try {
      (void)registration.get();
    } catch (...) {
    }
    close_all.get();
    FAIL() << "close_all did not close queue admission";
  }

  EXPECT_EQ(close_all.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 1u);
  EXPECT_EQ(binding_factory->destruction_calls(0), 1u);

  binding_factory->release_callback();
  EXPECT_THROW((void)registration.get(), std::bad_alloc);
  close_all.get();
  EXPECT_EQ(binding_factory->destruction_calls(0), 1u);
  EXPECT_EQ(queues.lifecycle_epoch(), old_queue_epoch + 1);
  EXPECT_EQ(gpu_vm.reset_epoch(), old_vm_epoch);
  ASSERT_TRUE(gpu_vm.lookup(address_space));
  EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(CommandProcessorQueueRegistration, FailedFanoutRollsBackOnlyNewReplicas) {
  GpuVm gpu_vm;
  const AddressSpaceHandle address_space = register_byte_address_space(gpu_vm, 7, 0x11);
  CommandProcessor owner("owner");
  CommandProcessor first_peer("first-peer");
  CommandProcessor rejecting_peer("rejecting-peer");
  owner.set_gpu_vm(&gpu_vm);
  first_peer.set_gpu_vm(&gpu_vm);
  rejecting_peer.set_gpu_vm(&gpu_vm);
  owner.set_xcd_topology(0, {&owner, &first_peer, &rejecting_peer});

  AqlQueueConfig existing{};
  existing.address_space = address_space;
  existing.process_id = 7;
  existing.queue_id = 11;
  existing.ring_base_va = 0x100;
  existing.ring_size = 4096;
  existing.read_ptr_va = 0x80;
  existing.write_ptr_va = 0x88;
  rejecting_peer.register_queue(existing);

  AqlQueueConfig fanout = existing;
  fanout.xcd_fanout = true;
  EXPECT_EQ(owner.register_queue(fanout), 0u);

  EXPECT_EQ(owner.registered_queue_count_for_test(), 0u);
  EXPECT_EQ(first_peer.registered_queue_count_for_test(), 0u);
  EXPECT_EQ(rejecting_peer.registered_queue_count_for_test(), 1u);
  rejecting_peer.unregister_queue(existing.queue_id, existing.process_id);
}

} // namespace
} // namespace rocjitsu::amdgpu
