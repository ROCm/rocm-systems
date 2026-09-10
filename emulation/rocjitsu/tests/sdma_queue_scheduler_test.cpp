// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_scheduler.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

using namespace std::chrono_literals;

template <typename Predicate>
bool wait_until(Predicate predicate, std::chrono::milliseconds timeout = 1s) {
  const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(50us);
  }
  return true;
}

class ScopedMapping {
public:
  ScopedMapping() {
    void *mapping = mmap(nullptr, GpuMemory::PAGE_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    data_ = mapping == MAP_FAILED ? nullptr : static_cast<uint8_t *>(mapping);
  }
  ScopedMapping(const ScopedMapping &) = delete;
  ScopedMapping &operator=(const ScopedMapping &) = delete;
  ~ScopedMapping() {
    if (data_ != nullptr)
      munmap(data_, GpuMemory::PAGE_SIZE);
  }

  uint8_t *data() const { return data_; }

private:
  uint8_t *data_ = nullptr;
};

class EngineMemory final : public AddressSpaceTranslator, public PhysicalMemoryAccess {
public:
  explicit EngineMemory(std::size_t size = 0x1000) : bytes_(size) {}

  VmTranslationResult translate(uint64_t address, std::size_t size, VmAccessKind) const override {
    std::lock_guard lock(mutex_);
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

  VmAccessOutcome read(VmMemoryDomain, uint64_t address, std::span<std::byte> bytes) override {
    std::unique_lock lock(mutex_);
    ++read_attempts_;
    condition_.notify_all();
    if (block_reads_) {
      condition_.wait(lock, [this]() { return !block_reads_ || permitted_reads_ != 0; });
      if (block_reads_)
        --permitted_reads_;
    }
    if (!read_outcomes_.empty()) {
      const VmAccessOutcome outcome = read_outcomes_.front();
      read_outcomes_.pop_front();
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
    }
    if (reads_unavailable_)
      return VmAccessOutcome::Unavailable;
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(address), bytes.size(), bytes.begin());
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(VmMemoryDomain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    std::lock_guard lock(mutex_);
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::copy(bytes.begin(), bytes.end(), bytes_.begin() + static_cast<std::ptrdiff_t>(address));
    write_addresses_.push_back(address);
    return VmAccessOutcome::Complete;
  }

  AtomicLoadResult atomic_load(VmMemoryDomain, uint64_t address, uint32_t width) override {
    std::lock_guard lock(mutex_);
    if ((width != 4 && width != 8) || address > bytes_.size() || width > bytes_.size() - address)
      return {.outcome = VmAccessOutcome::Faulted};
    uint64_t value = 0;
    std::memcpy(&value, bytes_.data() + address, width);
    return {.outcome = VmAccessOutcome::Complete, .value = value};
  }

  VmAccessOutcome atomic_store(VmMemoryDomain, uint64_t address, uint32_t width,
                               uint64_t value) override {
    std::lock_guard lock(mutex_);
    if ((width != 4 && width != 8) || address > bytes_.size() || width > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::memcpy(bytes_.data() + address, &value, width);
    return VmAccessOutcome::Complete;
  }

  template <typename Value> void store(uint64_t address, const Value &value) {
    std::lock_guard lock(mutex_);
    std::memcpy(bytes_.data() + address, &value, sizeof(value));
  }

  template <typename Value> Value load(uint64_t address) const {
    std::lock_guard lock(mutex_);
    Value value{};
    std::memcpy(&value, bytes_.data() + address, sizeof(value));
    return value;
  }

  void return_next_read(VmAccessOutcome outcome) {
    std::lock_guard lock(mutex_);
    read_outcomes_.push_back(outcome);
  }

  void set_reads_unavailable(bool unavailable) {
    std::lock_guard lock(mutex_);
    reads_unavailable_ = unavailable;
  }

  uint32_t read_attempts() const {
    std::lock_guard lock(mutex_);
    return read_attempts_;
  }

  void block_reads() {
    std::lock_guard lock(mutex_);
    block_reads_ = true;
  }

  void unblock_reads() {
    std::lock_guard lock(mutex_);
    block_reads_ = false;
    condition_.notify_all();
  }

  bool wait_for_read_attempts(uint32_t attempts) {
    std::unique_lock lock(mutex_);
    return condition_.wait_for(lock, 1s, [this, attempts]() { return read_attempts_ >= attempts; });
  }

  void permit_one_read() {
    std::lock_guard lock(mutex_);
    ++permitted_reads_;
    condition_.notify_all();
  }

  std::vector<uint64_t> write_addresses() const {
    std::lock_guard lock(mutex_);
    return write_addresses_;
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::vector<std::byte> bytes_;
  std::deque<VmAccessOutcome> read_outcomes_;
  std::vector<uint64_t> write_addresses_;
  uint32_t read_attempts_ = 0;
  uint32_t permitted_reads_ = 0;
  bool block_reads_ = false;
  bool reads_unavailable_ = false;
};

class SdmaQueueSchedulerTest : public ::testing::Test {
protected:
  static constexpr uint32_t kProcessId = 7;
  static constexpr uint32_t kQueueId = 11;
  static constexpr uint64_t kReadPointer = 0x80;
  static constexpr uint64_t kRing = 0x100;
  static constexpr uint32_t kRingBytes = 64;

  SdmaQueueSchedulerTest() : memory_(std::make_shared<EngineMemory>()), scheduler_(gpu_vm_) {
    address_space_ = gpu_vm_.register_translated(kProcessId, memory_, memory_);
    if (!address_space_)
      throw std::runtime_error("cannot register SDMA scheduler test address space");
    memory_->store<uint64_t>(kReadPointer, 0);
  }

  ~SdmaQueueSchedulerTest() override {
    // An ASSERT after block_reads() may leave the worker parked in the synthetic
    // backing store. Always release it before shutdown so a test failure cannot
    // turn fixture teardown into a permanent join.
    memory_->unblock_reads();
    scheduler_.shutdown();
    (void)gpu_vm_.unregister_address_space(address_space_);
  }

  SdmaQueueConfig queue_info(uint32_t queue_id = kQueueId, uint64_t ring = kRing,
                             uint64_t read_pointer = kReadPointer,
                             uint32_t ring_bytes = kRingBytes) const {
    return {.ring = {.address_space = address_space_,
                     .ring_base = ring,
                     .ring_bytes = ring_bytes,
                     .read_pointer_address = read_pointer,
                     .initial_cursor = 0},
            .context = {.process_id = kProcessId, .queue_id = queue_id, .engine_id = 0}};
  }

  QueueRegistrationRequest registration_info() const {
    return {.identity = {.address_space = address_space_,
                         .interrupt_sink = {},
                         .process_id = kProcessId,
                         .queue_id = kQueueId},
            .ring = {.base_address = kRing,
                     .size_bytes = kRingBytes,
                     .consumer_pointer_address = kReadPointer},
            .doorbell = {},
            .binding_factory = {},
            .initial_consumer_cursor = 0,
            .type = QueueType::Sdma,
            .packet_format = QueuePacketFormat::Sdma};
  }

  std::shared_ptr<EngineMemory> memory_;
  GpuVm gpu_vm_;
  SdmaQueueScheduler scheduler_;
  AddressSpaceHandle address_space_;
};

TEST_F(SdmaQueueSchedulerTest, ServicesExplicitNotificationAndShutsDownCleanly) {
  memory_->store<uint32_t>(kRing, 0);
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);
  EXPECT_EQ(scheduler_.active_queues(), 1u);
  EXPECT_TRUE(scheduler_.contains(queue));
  EXPECT_FALSE(scheduler_.set_packet_dialect(SdmaPacketDialect::Gfx1250));

  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  EXPECT_TRUE(
      wait_until([this]() { return memory_->load<uint64_t>(kReadPointer) == sizeof(uint32_t); }));
  EXPECT_TRUE(scheduler_.detach(queue));
  EXPECT_EQ(scheduler_.active_queues(), 0u);
  EXPECT_TRUE(scheduler_.set_packet_dialect(SdmaPacketDialect::Gfx1250));
  EXPECT_EQ(scheduler_.packet_dialect(), SdmaPacketDialect::Gfx1250);

  scheduler_.shutdown();
  EXPECT_FALSE(scheduler_.attach(queue_info(), {}));
}

TEST_F(SdmaQueueSchedulerTest, RetainsItsAddressSpaceUntilDetach) {
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);

  EXPECT_FALSE(gpu_vm_.unregister_address_space(address_space_));
  EXPECT_TRUE(scheduler_.detach(queue));
  EXPECT_TRUE(gpu_vm_.unregister_address_space(address_space_));
}

TEST_F(SdmaQueueSchedulerTest, GpuQueueRegistryDestructorDetachesScheduledQueue) {
  {
    GpuQueueRegistry queues(gpu_vm_);
    QueueRegistrationRequest info = registration_info();
    info.binding_factory = make_sdma_queue_binding_factory(scheduler_);
    ASSERT_TRUE(queues.register_queue(info));
    EXPECT_EQ(scheduler_.active_queues(), 1u);
    ASSERT_TRUE(gpu_vm_.lookup(address_space_));
    EXPECT_EQ(gpu_vm_.lookup(address_space_)->queue_references, 1u);
  }

  EXPECT_EQ(scheduler_.active_queues(), 0u);
  ASSERT_TRUE(gpu_vm_.lookup(address_space_));
  EXPECT_EQ(gpu_vm_.lookup(address_space_)->queue_references, 0u);
}

TEST_F(SdmaQueueSchedulerTest, BindingFactoryTranslatesOnlyMinimalSdmaContext) {
  std::optional<SdmaQueueContext> callback_context;
  std::optional<SdmaQueueContext> observer_context;
  auto factory = std::make_shared<SdmaQueueBindingFactory>(
      scheduler_,
      [&](const SdmaQueueContext &context) {
        callback_context = context;
        return SdmaPacketCallbacks{};
      },
      [&](const SdmaQueueContext &context) {
        observer_context = context;
        return SdmaQueueProgressObserver{};
      });

  GpuQueueRegistry queues(gpu_vm_);
  QueueRegistrationRequest info = registration_info();
  info.engine_id = 3;
  info.binding_factory = factory;
  const QueueHandle queue = queues.register_queue(info);
  ASSERT_TRUE(queue);
  ASSERT_TRUE(callback_context);
  ASSERT_TRUE(observer_context);
  EXPECT_EQ(callback_context->process_id, kProcessId);
  EXPECT_EQ(callback_context->queue_id, kQueueId);
  EXPECT_EQ(callback_context->engine_id, 3u);
  EXPECT_EQ(*observer_context, *callback_context);

  EXPECT_EQ(queues.unregister_queue(queue, QueueCloseMode::ForceCancel).status,
            QueueCloseStatus::Closed);
}

TEST_F(SdmaQueueSchedulerTest, NotificationsReturnImmediatelyAndRetainTheLargestProducer) {
  memory_->store<uint32_t>(kRing, 0);
  memory_->store<uint32_t>(kRing + sizeof(uint32_t), 0);
  memory_->block_reads();
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  ASSERT_TRUE(memory_->wait_for_read_attempts(1));
  EXPECT_EQ(scheduler_.notify(queue, 2 * sizeof(uint32_t)), QueueSubmissionStatus::Accepted);

  memory_->permit_one_read();
  ASSERT_TRUE(memory_->wait_for_read_attempts(2));
  memory_->permit_one_read();
  EXPECT_TRUE(wait_until(
      [this]() { return memory_->load<uint64_t>(kReadPointer) == 2 * sizeof(uint32_t); }));
  EXPECT_TRUE(scheduler_.detach(queue));
}

TEST_F(SdmaQueueSchedulerTest, BlockedServiceIsRetriedWithoutFrontendResubmission) {
  memory_->store<uint32_t>(kRing, 0);
  memory_->store<uint32_t>(kRing + sizeof(uint32_t), 0);
  memory_->return_next_read(VmAccessOutcome::Unavailable);
  memory_->block_reads();
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  ASSERT_TRUE(memory_->wait_for_read_attempts(1));
  EXPECT_EQ(scheduler_.notify(queue, 2 * sizeof(uint32_t)), QueueSubmissionStatus::Accepted);

  memory_->permit_one_read();
  ASSERT_TRUE(memory_->wait_for_read_attempts(2));
  memory_->permit_one_read();
  ASSERT_TRUE(memory_->wait_for_read_attempts(3));
  memory_->permit_one_read();

  EXPECT_TRUE(wait_until(
      [this]() { return memory_->load<uint64_t>(kReadPointer) == 2 * sizeof(uint32_t); }));
  EXPECT_TRUE(scheduler_.detach(queue));
}

TEST_F(SdmaQueueSchedulerTest, ConcurrentSameProducerNotificationsBothComplete) {
  memory_->store<uint32_t>(kRing, 0);
  memory_->block_reads();
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  ASSERT_TRUE(memory_->wait_for_read_attempts(1));
  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);

  memory_->permit_one_read();
  EXPECT_TRUE(
      wait_until([this]() { return memory_->load<uint64_t>(kReadPointer) == sizeof(uint32_t); }));
  EXPECT_TRUE(scheduler_.detach(queue));
}

TEST_F(SdmaQueueSchedulerTest, ConcurrentOlderProducerDoesNotRegressTheServiceTarget) {
  memory_->store<uint32_t>(kRing, 0);
  memory_->store<uint32_t>(kRing + sizeof(uint32_t), 0);
  memory_->block_reads();
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler_.notify(queue, 2 * sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  ASSERT_TRUE(memory_->wait_for_read_attempts(1));
  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);

  memory_->permit_one_read();
  memory_->permit_one_read();
  EXPECT_TRUE(wait_until(
      [this]() { return memory_->load<uint64_t>(kReadPointer) == 2 * sizeof(uint32_t); }));
  EXPECT_TRUE(scheduler_.detach(queue));
}

TEST_F(SdmaQueueSchedulerTest, TerminalServiceIsReportedAfterAcceptedSubmission) {
  memory_->store<uint32_t>(kRing, 0);
  memory_->store<uint32_t>(kRing + sizeof(uint32_t), 0);
  memory_->return_next_read(VmAccessOutcome::Faulted);
  memory_->block_reads();
  std::mutex progress_mutex;
  std::condition_variable progress_condition;
  bool terminal = false;
  const SdmaQueueScheduler::Handle queue =
      scheduler_.attach(queue_info(), {}, [&](const SdmaQueueProgress &progress) {
        const std::lock_guard lock(progress_mutex);
        terminal = progress.terminal;
        progress_condition.notify_all();
      });
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  ASSERT_TRUE(memory_->wait_for_read_attempts(1));
  EXPECT_EQ(scheduler_.notify(queue, 2 * sizeof(uint32_t)), QueueSubmissionStatus::Accepted);

  memory_->permit_one_read();
  {
    std::unique_lock lock(progress_mutex);
    EXPECT_TRUE(progress_condition.wait_for(lock, 1s, [&]() { return terminal; }));
  }
  EXPECT_EQ(scheduler_.notify(queue, 2 * sizeof(uint32_t)), QueueSubmissionStatus::Faulted);
  EXPECT_TRUE(scheduler_.detach(queue));
}

TEST_F(SdmaQueueSchedulerTest, PollsHostDoorbellWithoutFrontendNotification) {
  memory_->store<uint32_t>(kRing, 0);
  alignas(8) uint64_t doorbell = std::numeric_limits<uint64_t>::max();
  SdmaQueueConfig info = queue_info();
  info.doorbell_base = &doorbell;
  info.last_doorbell = std::numeric_limits<uint64_t>::max();
  info.host_accessible = true;
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(info, {});
  ASSERT_TRUE(queue);

  std::atomic_ref<uint64_t>(doorbell).store(sizeof(uint32_t), std::memory_order_release);
  EXPECT_TRUE(wait_until([this]() {
    return memory_->load<uint64_t>(kReadPointer) == sizeof(uint32_t);
  })) << "the SDMA worker did not observe the host doorbell";
  EXPECT_TRUE(scheduler_.detach(queue));
}

TEST_F(SdmaQueueSchedulerTest, DoorbellRemapDoesNotWaitForActivePacketService) {
  memory_->store<uint32_t>(kRing, 0);
  memory_->store<uint32_t>(kRing + sizeof(uint32_t), 0);
  memory_->block_reads();
  alignas(8) uint64_t original_doorbell = sizeof(uint32_t);
  alignas(8) uint64_t replacement_doorbell = 0;
  SdmaQueueConfig info = queue_info();
  info.doorbell_base = &original_doorbell;
  info.last_doorbell = sizeof(uint32_t);
  info.host_accessible = true;
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(info, {});
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  ASSERT_TRUE(memory_->wait_for_read_attempts(1));

  std::future<void> remap = std::async(std::launch::async, [this, &replacement_doorbell]() {
    scheduler_.set_process_doorbell_base(kProcessId, &replacement_doorbell);
  });
  EXPECT_EQ(remap.wait_for(1s), std::future_status::ready)
      << "doorbell remapping waited for unrelated packet execution";
  remap.get();

  std::atomic_ref<uint64_t>(replacement_doorbell)
      .store(2 * sizeof(uint32_t), std::memory_order_release);
  memory_->permit_one_read();
  ASSERT_TRUE(memory_->wait_for_read_attempts(2));
  memory_->permit_one_read();
  EXPECT_TRUE(wait_until(
      [this]() { return memory_->load<uint64_t>(kReadPointer) == 2 * sizeof(uint32_t); }));
  EXPECT_TRUE(scheduler_.detach(queue));
}

TEST_F(SdmaQueueSchedulerTest, DisableSerializesBeforeAConcurrentNotification) {
  memory_->store<uint32_t>(kRing, 0);
  memory_->store<uint32_t>(kRing + sizeof(uint32_t), 0);
  memory_->block_reads();
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  ASSERT_TRUE(memory_->wait_for_read_attempts(1));

  std::promise<void> update_started;
  std::future<void> update_started_future = update_started.get_future();
  std::future<QueueReconfigureStatus> disabled =
      std::async(std::launch::async, [this, queue, &update_started]() {
        update_started.set_value();
        return scheduler_.update(queue, 0, 0, 0);
      });
  ASSERT_EQ(update_started_future.wait_for(1s), std::future_status::ready);
  EXPECT_EQ(disabled.wait_for(20ms), std::future_status::timeout);

  EXPECT_EQ(scheduler_.notify(queue, 2 * sizeof(uint32_t)), QueueSubmissionStatus::Accepted);

  memory_->permit_one_read();
  ASSERT_EQ(disabled.wait_for(1s), std::future_status::ready);
  EXPECT_EQ(disabled.get(), QueueReconfigureStatus::Disabled);
  std::this_thread::sleep_for(1ms);
  EXPECT_EQ(memory_->read_attempts(), 1u);
  EXPECT_TRUE(scheduler_.detach(queue));
}

TEST_F(SdmaQueueSchedulerTest, RoundRobinServicePreventsOneQueueFromMonopolizingTheWorker) {
  constexpr uint64_t kSecondReadPointer = 0x88;
  constexpr uint64_t kSecondRing = 0x400;
  constexpr uint64_t kFirstFenceTarget = 0x700;
  constexpr uint64_t kSecondFenceTarget = 0x708;
  constexpr uint32_t kNopPacketsBeforeFence = 64;
  constexpr uint32_t kFenceDwords = 4;

  for (uint32_t packet_index = 0; packet_index < kNopPacketsBeforeFence; ++packet_index)
    memory_->store<uint32_t>(kRing + packet_index * sizeof(uint32_t), 0);
  const std::array<uint32_t, kFenceDwords> first_fence = {
      5, static_cast<uint32_t>(kFirstFenceTarget), 0, 0x11111111};
  memory_->store(kRing + kNopPacketsBeforeFence * sizeof(uint32_t), first_fence);
  const std::array<uint32_t, kFenceDwords> second_fence = {
      5, static_cast<uint32_t>(kSecondFenceTarget), 0, 0x22222222};
  memory_->store(kSecondRing, second_fence);
  memory_->store<uint64_t>(kSecondReadPointer, 0);
  memory_->block_reads();
  alignas(8) uint64_t second_doorbell = 0;

  const SdmaQueueScheduler::Handle first =
      scheduler_.attach(queue_info(kQueueId, kRing, kReadPointer, 512), {});
  SdmaQueueConfig second_info = queue_info(kQueueId + 1, kSecondRing, kSecondReadPointer);
  second_info.doorbell_base = &second_doorbell;
  second_info.host_accessible = true;
  const SdmaQueueScheduler::Handle second = scheduler_.attach(second_info, {});
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);

  std::future<QueueSubmissionStatus> first_notification =
      std::async(std::launch::async, [this, first]() {
        return scheduler_.notify(first, (kNopPacketsBeforeFence + kFenceDwords) * sizeof(uint32_t));
      });
  ASSERT_TRUE(memory_->wait_for_read_attempts(1));
  std::atomic_ref<uint64_t>(second_doorbell)
      .store(kFenceDwords * sizeof(uint32_t), std::memory_order_release);

  memory_->unblock_reads();
  ASSERT_EQ(first_notification.wait_for(1s), std::future_status::ready);
  EXPECT_EQ(first_notification.get(), QueueSubmissionStatus::Accepted);
  EXPECT_TRUE(wait_until([this]() {
    return memory_->load<uint64_t>(kSecondReadPointer) == kFenceDwords * sizeof(uint32_t);
  }));
  EXPECT_TRUE(wait_until([this]() {
    return memory_->load<uint64_t>(kReadPointer) ==
           (kNopPacketsBeforeFence + kFenceDwords) * sizeof(uint32_t);
  }));

  const std::vector<uint64_t> writes = memory_->write_addresses();
  ASSERT_EQ(writes.size(), 2u);
  EXPECT_EQ(writes[0], kSecondFenceTarget);
  EXPECT_EQ(writes[1], kFirstFenceTarget);
  EXPECT_TRUE(scheduler_.detach(first));
  EXPECT_TRUE(scheduler_.detach(second));
}

TEST_F(SdmaQueueSchedulerTest, RetriesUnavailablePacketWithoutSecondDoorbell) {
  memory_->store<uint32_t>(kRing, 0);
  memory_->return_next_read(VmAccessOutcome::Unavailable);
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);

  (void)scheduler_.notify(queue, sizeof(uint32_t));
  EXPECT_TRUE(
      wait_until([this]() { return memory_->load<uint64_t>(kReadPointer) == sizeof(uint32_t); }));
  EXPECT_GE(memory_->read_attempts(), 2u);
  EXPECT_TRUE(scheduler_.detach(queue));
}

TEST_F(SdmaQueueSchedulerTest, GracefulDetachPreservesBlockedPacketStateUntilItDrains) {
  memory_->store<uint32_t>(kRing, 0);
  memory_->set_reads_unavailable(true);
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  ASSERT_TRUE(wait_until([this]() { return memory_->read_attempts() != 0; }));
  EXPECT_EQ(scheduler_.prepare_detach(queue), QueuePrepareCloseStatus::Busy);
  EXPECT_TRUE(scheduler_.contains(queue));

  memory_->set_reads_unavailable(false);
  ASSERT_TRUE(
      wait_until([this]() { return memory_->load<uint64_t>(kReadPointer) == sizeof(uint32_t); }));
  EXPECT_EQ(scheduler_.prepare_detach(queue), QueuePrepareCloseStatus::Ready);
  EXPECT_FALSE(scheduler_.contains(queue));
}

TEST_F(SdmaQueueSchedulerTest, DisableRejectsBlockedPacketStateSoGracefulDetachCanDrain) {
  memory_->store<uint32_t>(kRing, 0);
  memory_->set_reads_unavailable(true);
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  ASSERT_TRUE(wait_until([this]() { return memory_->read_attempts() != 0; }));
  ASSERT_EQ(scheduler_.update(queue, 0, 0, 0), QueueReconfigureStatus::Busy);
  EXPECT_EQ(scheduler_.prepare_detach(queue), QueuePrepareCloseStatus::Busy);
  EXPECT_TRUE(scheduler_.contains(queue));

  memory_->set_reads_unavailable(false);
  ASSERT_TRUE(
      wait_until([this]() { return memory_->load<uint64_t>(kReadPointer) == sizeof(uint32_t); }));
  EXPECT_EQ(scheduler_.prepare_detach(queue), QueuePrepareCloseStatus::Ready);
  EXPECT_FALSE(scheduler_.contains(queue));
}

TEST_F(SdmaQueueSchedulerTest, DetachWaitsForActiveServiceAndRejectsLaterNotifications) {
  memory_->store<uint32_t>(kRing, 0);
  memory_->block_reads();
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  ASSERT_TRUE(memory_->wait_for_read_attempts(1));
  std::future<bool> detached =
      std::async(std::launch::async, [this, queue]() { return scheduler_.detach(queue); });
  EXPECT_EQ(detached.wait_for(20ms), std::future_status::timeout);

  memory_->permit_one_read();
  ASSERT_EQ(detached.wait_for(1s), std::future_status::ready);
  EXPECT_TRUE(detached.get());

  const uint32_t reads_after_detach = memory_->read_attempts();
  EXPECT_EQ(scheduler_.notify(queue, 2 * sizeof(uint32_t)), QueueSubmissionStatus::Faulted);
  std::this_thread::sleep_for(1ms);
  EXPECT_EQ(memory_->read_attempts(), reads_after_detach);
}

TEST_F(SdmaQueueSchedulerTest, RebindsAfterBlockedServiceTurnQuiesces) {
  constexpr uint64_t kReplacementRing = 0x200;
  memory_->store<uint32_t>(kRing, 0);
  memory_->store<uint32_t>(kReplacementRing, 0);
  memory_->set_reads_unavailable(true);
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  EXPECT_EQ(scheduler_.update(queue, kReplacementRing, kRingBytes, 100),
            QueueReconfigureStatus::Applied);

  memory_->set_reads_unavailable(false);
  EXPECT_TRUE(
      wait_until([this]() { return memory_->load<uint64_t>(kReadPointer) == sizeof(uint32_t); }));
  EXPECT_TRUE(scheduler_.detach(queue));
}

TEST_F(SdmaQueueSchedulerTest, ProgressObserverRunsOutsideMutexButInsideDetachQuiescence) {
  memory_->store<uint32_t>(kRing, 0);
  std::mutex observer_mutex;
  std::condition_variable observer_condition;
  bool entered = false;
  bool release = false;
  const SdmaQueueScheduler::Handle queue =
      scheduler_.attach(queue_info(), {}, [&](const SdmaQueueProgress &) {
        std::unique_lock lock(observer_mutex);
        entered = true;
        observer_condition.notify_all();
        observer_condition.wait(lock, [&]() { return release; });
      });
  ASSERT_TRUE(queue);
  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  {
    std::unique_lock lock(observer_mutex);
    ASSERT_TRUE(observer_condition.wait_for(lock, 1s, [&]() { return entered; }));
  }

  // The observer does not hold the scheduler mutex, so a newer target can be
  // retained while the frontend is still publishing the previous progress.
  EXPECT_EQ(scheduler_.notify(queue, 2 * sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  std::future<bool> detached =
      std::async(std::launch::async, [this, queue]() { return scheduler_.detach(queue); });
  EXPECT_EQ(detached.wait_for(20ms), std::future_status::timeout);
  {
    const std::lock_guard lock(observer_mutex);
    release = true;
  }
  observer_condition.notify_all();
  ASSERT_EQ(detached.wait_for(1s), std::future_status::ready);
  EXPECT_TRUE(detached.get());
}

TEST_F(SdmaQueueSchedulerTest, TerminalStateIsVisibleBeforeObserverNotification) {
  memory_->store<uint32_t>(kRing, 0xff);
  std::mutex observer_mutex;
  std::condition_variable observer_condition;
  bool entered = false;
  bool release = false;
  const SdmaQueueScheduler::Handle queue =
      scheduler_.attach(queue_info(), {}, [&](const SdmaQueueProgress &progress) {
        if (!progress.terminal)
          return;
        std::unique_lock lock(observer_mutex);
        entered = true;
        observer_condition.notify_all();
        observer_condition.wait(lock, [&]() { return release; });
      });
  ASSERT_TRUE(queue);
  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  {
    std::unique_lock lock(observer_mutex);
    ASSERT_TRUE(observer_condition.wait_for(lock, 1s, [&]() { return entered; }));
  }

  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Faulted);
  {
    const std::lock_guard lock(observer_mutex);
    release = true;
  }
  observer_condition.notify_all();
  EXPECT_TRUE(scheduler_.detach(queue));
}

TEST_F(SdmaQueueSchedulerTest, RejectsInvalidAdmissionWithoutThrowing) {
  SdmaQueueConfig invalid = queue_info();
  invalid.ring.ring_bytes = 0;
  EXPECT_FALSE(scheduler_.attach(invalid, {}));
}

TEST_F(SdmaQueueSchedulerTest, RejectsStaleHandleAfterSlotReuse) {
  const SdmaQueueScheduler::Handle stale = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(stale);
  ASSERT_TRUE(scheduler_.detach(stale));

  SdmaQueueConfig replacement_info = queue_info();
  replacement_info.context.queue_id = kQueueId + 1;
  const SdmaQueueScheduler::Handle replacement = scheduler_.attach(replacement_info, {});
  ASSERT_TRUE(replacement);
  EXPECT_NE(stale, replacement);
  EXPECT_FALSE(scheduler_.contains(stale));
  EXPECT_EQ(scheduler_.notify(stale, sizeof(uint32_t)), QueueSubmissionStatus::Faulted);
  EXPECT_FALSE(scheduler_.detach(stale));
  EXPECT_TRUE(scheduler_.contains(replacement));
  EXPECT_TRUE(scheduler_.detach(replacement));
}

TEST_F(SdmaQueueSchedulerTest, LatchesTerminalQueueFailure) {
  memory_->store<uint32_t>(kRing, 0xff);
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  EXPECT_TRUE(wait_until([&]() {
    return scheduler_.notify(queue, sizeof(uint32_t)) == QueueSubmissionStatus::Faulted;
  }));
  memory_->store<uint32_t>(kRing, 0);
  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Faulted);
  EXPECT_EQ(memory_->load<uint64_t>(kReadPointer), 0u);
  EXPECT_TRUE(scheduler_.detach(queue));
}

TEST_F(SdmaQueueSchedulerTest, GracefulDetachClosesTerminalWithoutRetirement) {
  memory_->store<uint32_t>(kRing, 0xff);
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler_.notify(queue, sizeof(uint32_t)), QueueSubmissionStatus::Accepted);
  ASSERT_TRUE(wait_until([&]() {
    return scheduler_.notify(queue, sizeof(uint32_t)) == QueueSubmissionStatus::Faulted;
  }));
  EXPECT_EQ(memory_->load<uint64_t>(kReadPointer), 0u);
  EXPECT_EQ(scheduler_.prepare_detach(queue), QueuePrepareCloseStatus::Ready);
  EXPECT_FALSE(scheduler_.contains(queue));
}

TEST_F(SdmaQueueSchedulerTest, GracefulDetachClosesRetiredTerminalAfterCursorPublication) {
  constexpr uint64_t kBadSource = 0x2000;
  constexpr uint64_t kDestination = 0x300;
  const std::array<uint32_t, 7> copy = {
      1, 3, 0, static_cast<uint32_t>(kBadSource), 0, static_cast<uint32_t>(kDestination), 0};
  memory_->store(kRing, copy);
  const SdmaQueueScheduler::Handle queue = scheduler_.attach(queue_info(), {});
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler_.notify(queue, sizeof(copy)), QueueSubmissionStatus::Accepted);
  ASSERT_TRUE(wait_until([&]() {
    return memory_->load<uint64_t>(kReadPointer) == sizeof(copy) &&
           scheduler_.notify(queue, sizeof(copy)) == QueueSubmissionStatus::Faulted;
  }));
  EXPECT_EQ(scheduler_.prepare_detach(queue), QueuePrepareCloseStatus::Ready);
  EXPECT_FALSE(scheduler_.contains(queue));
}

TEST(SdmaQueueSchedulerCacheMaintenanceTest, FailedWritebackFaultsQueueBeforePacketWrite) {
  constexpr uint32_t kProcessId = 7;
  constexpr uint32_t kQueueId = 11;
  constexpr uint64_t kReadPointer = 0x80;
  constexpr uint64_t kRing = 0x100;
  constexpr uint64_t kFenceTarget = 0x300;
  constexpr uint64_t kDirtyAddress = 0x160000;

  auto queue_memory = std::make_shared<EngineMemory>();
  GpuVm gpu_vm;
  const AddressSpaceHandle queue_address_space =
      gpu_vm.register_translated(kProcessId, queue_memory, queue_memory);
  ASSERT_TRUE(queue_address_space);
  queue_memory->store<uint64_t>(kReadPointer, 0);
  const std::array<uint32_t, 4> fence = {5, static_cast<uint32_t>(kFenceTarget), 0, 0x11223344};
  queue_memory->store(kRing, fence);

  ScopedMapping mapping;
  ASSERT_NE(mapping.data(), nullptr);
  KfdProcess legacy_process(kProcessId);
  legacy_process.map_pages(kDirtyAddress, mapping.data(), GpuMemory::PAGE_SIZE);
  GpuMemory legacy_memory("legacy_memory");
  GpuVm legacy_gpu_vm;
  LegacyGpuVmAdapter legacy_vm(legacy_gpu_vm, &legacy_memory);
  const AddressSpaceHandle legacy_address_space = legacy_vm.register_address_space(
      kProcessId, &legacy_process.page_table_, &legacy_process.page_table_mutex_,
      legacy_process.page_table_generation());
  ASSERT_TRUE(legacy_address_space);

  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  l2.set_backing_memory(&legacy_memory);
  l2.set_legacy_maintenance_memory(&legacy_memory);
  l2.set_legacy_maintenance_vm(&legacy_gpu_vm);
  l2.set_gpu_vm(&legacy_gpu_vm);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_line{};
  dirty_line.fill(0x5c);
  l2.writeback_line(kDirtyAddress, dirty_line.data(), Mtype::RW, kProcessId);
  l2.set_backing_memory(nullptr);
  ASSERT_EQ(mprotect(mapping.data(), GpuMemory::PAGE_SIZE, PROT_READ), 0);

  SdmaQueueScheduler scheduler(gpu_vm, coherence);
  SdmaPacketCallbacks callbacks;
  callbacks.acquire_cache_maintenance = [coherence](SdmaCacheOperation operation) {
    return coherence->acquire_cache_maintenance(operation);
  };
  const SdmaQueueScheduler::Handle queue = scheduler.attach(
      {.ring = {.address_space = queue_address_space,
                .ring_base = kRing,
                .ring_bytes = 64,
                .read_pointer_address = kReadPointer,
                .initial_cursor = 0},
       .context = {.process_id = kProcessId, .queue_id = kQueueId, .engine_id = 0}},
      std::move(callbacks));
  ASSERT_TRUE(queue);

  EXPECT_EQ(scheduler.notify(queue, fence.size() * sizeof(uint32_t)),
            QueueSubmissionStatus::Accepted);
  EXPECT_TRUE(wait_until([&]() {
    return scheduler.notify(queue, fence.size() * sizeof(uint32_t)) ==
           QueueSubmissionStatus::Faulted;
  }));
  EXPECT_EQ(queue_memory->load<uint32_t>(kFenceTarget), 0u);
  EXPECT_EQ(queue_memory->load<uint64_t>(kReadPointer), 0u);

  EXPECT_TRUE(scheduler.detach(queue));
  scheduler.shutdown();
  ASSERT_EQ(mprotect(mapping.data(), GpuMemory::PAGE_SIZE, PROT_READ | PROT_WRITE), 0);
  {
    [[maybe_unused]] DeviceCacheMaintenanceLease maintenance =
        coherence->acquire_cache_maintenance(DeviceCacheOperation::WritebackInvalidate);
  }
  EXPECT_EQ(mapping.data()[0], 0x5cu);
  EXPECT_TRUE(legacy_vm.unregister_vmid(kProcessId));
  EXPECT_TRUE(gpu_vm.unregister_address_space(queue_address_space));
}

} // namespace
} // namespace rocjitsu::amdgpu
