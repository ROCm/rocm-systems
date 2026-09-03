// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/completion_tracker.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_hsa_queue.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

class CompletionRetryMemory final : public AddressSpaceTranslator, public PhysicalMemoryAccess {
public:
  enum class Operation { Read, AtomicStore, CompareExchange };

  VmTranslationResult translate(uint64_t address, std::size_t size, VmAccessKind) const override {
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
    if (fail_once(Operation::Read, address))
      return failure_outcome_;
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::copy_n(bytes_.begin() + static_cast<ptrdiff_t>(address), bytes.size(), bytes.begin());
    ++successful_reads_[address];
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(VmMemoryDomain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::copy(bytes.begin(), bytes.end(), bytes_.begin() + static_cast<ptrdiff_t>(address));
    return VmAccessOutcome::Complete;
  }

  AtomicLoadResult atomic_load(VmMemoryDomain, uint64_t address, uint32_t width) override {
    if ((width != 4 && width != 8) || address > bytes_.size() || width > bytes_.size() - address)
      return {.outcome = VmAccessOutcome::Faulted, .value = 0};
    uint64_t value = 0;
    std::memcpy(&value, bytes_.data() + address, width);
    return {.outcome = VmAccessOutcome::Complete, .value = value};
  }

  VmAccessOutcome atomic_store(VmMemoryDomain, uint64_t address, uint32_t width,
                               uint64_t value) override {
    if (fail_once(Operation::AtomicStore, address))
      return failure_outcome_;
    if ((width != 4 && width != 8) || address > bytes_.size() || width > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::memcpy(bytes_.data() + address, &value, width);
    ++successful_stores_[address];
    return VmAccessOutcome::Complete;
  }

  AtomicCompareExchangeResult compare_exchange(VmMemoryDomain, uint64_t address, uint32_t width,
                                               uint64_t expected, uint64_t desired) override {
    if (fail_once(Operation::CompareExchange, address))
      return {.outcome = failure_outcome_, .observed = 0, .exchanged = false};
    if ((width != 4 && width != 8) || address > bytes_.size() || width > bytes_.size() - address)
      return {.outcome = VmAccessOutcome::Faulted, .observed = 0, .exchanged = false};
    uint64_t observed = 0;
    std::memcpy(&observed, bytes_.data() + address, width);
    const bool exchanged = observed == expected;
    if (exchanged) {
      std::memcpy(bytes_.data() + address, &desired, width);
      ++successful_compare_exchanges_;
    }
    return {.outcome = VmAccessOutcome::Complete, .observed = observed, .exchanged = exchanged};
  }

  template <typename T> void store(uint64_t address, T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::memcpy(bytes_.data() + address, &value, sizeof(value));
  }

  template <typename T> T load(uint64_t address) const {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    std::memcpy(&value, bytes_.data() + address, sizeof(value));
    return value;
  }

  void fail_next(Operation operation, uint64_t address,
                 VmAccessOutcome outcome = VmAccessOutcome::Unavailable) {
    failure_operation_ = operation;
    failure_address_ = address;
    failure_outcome_ = outcome;
    failure_armed_ = true;
  }

  uint32_t successful_stores(uint64_t address) const {
    const auto found = successful_stores_.find(address);
    return found == successful_stores_.end() ? 0 : found->second;
  }
  uint32_t successful_compare_exchanges() const { return successful_compare_exchanges_; }

private:
  bool fail_once(Operation operation, uint64_t address) {
    if (!failure_armed_ || operation != failure_operation_ || address != failure_address_)
      return false;
    failure_armed_ = false;
    return true;
  }

  std::array<std::byte, 4096> bytes_{};
  std::unordered_map<uint64_t, uint32_t> successful_reads_;
  std::unordered_map<uint64_t, uint32_t> successful_stores_;
  Operation failure_operation_ = Operation::Read;
  uint64_t failure_address_ = std::numeric_limits<uint64_t>::max();
  VmAccessOutcome failure_outcome_ = VmAccessOutcome::Unavailable;
  bool failure_armed_ = false;
  uint32_t successful_compare_exchanges_ = 0;
};

class CompletionEndCounter final : public ExecutionPlugin {
public:
  CompletionEndCounter() : ExecutionPlugin("completion_end_counter") {}
  void onAmdgpuDispatchExecutionEnd(uint32_t) override { ++calls; }
  uint32_t calls = 0;
};

struct CompletionFixture {
  static constexpr uint32_t kProcessId = 19;
  static constexpr uint32_t kQueueId = 23;
  static constexpr uint64_t kSignal = 0x100;
  static constexpr uint64_t kMailbox = 0x300;
  static constexpr uint32_t kEventId = 31;
  static constexpr uint32_t kValueOffset = 8;
  static constexpr uint32_t kMailboxOffset = 16;
  static constexpr uint32_t kEventOffset = 24;
  static constexpr uint32_t kStartOffset = 32;
  static constexpr uint32_t kEndOffset = 40;

  CompletionFixture() : tracker(nullptr, &vm, cus) {
    address_space = vm.register_translated(kProcessId, memory, memory);
    memory->store(kSignal + kValueOffset, uint64_t{1});
    memory->store(kSignal + kMailboxOffset, kMailbox);
    memory->store(kSignal + kEventOffset, kEventId);
    subscription = InterruptSubscription([this](uint32_t process_id, uint32_t) {
      EXPECT_EQ(process_id, kProcessId);
      ++interrupts;
    });

    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto plugin = std::make_unique<CompletionEndCounter>();
    end_counter = plugin.get();
    EXPECT_TRUE(group->add(std::move(plugin)));
    tracker.set_plugin_group(std::move(group));
    tracker.set_dispatch_retired_callback([this](const DispatchEntry &) { ++retired; });

    queues.resize(1);
    queues.front().fanout_replica = true;
    DispatchEntry entry{};
    entry.dispatch_id = 7;
    entry.queue_id = kQueueId;
    entry.address_space = address_space;
    entry.interrupt_sink = subscription.sink();
    entry.process_id = kProcessId;
    entry.total_wgs = 1;
    entry.completed_wgs = 1;
    entry.completion_signal = kSignal;
    entry.kind = DispatchPacketKind::Kernel;
    queues.front().push_entry(std::move(entry));
  }

  GpuVm vm;
  std::shared_ptr<CompletionRetryMemory> memory = std::make_shared<CompletionRetryMemory>();
  AddressSpaceHandle address_space;
  std::vector<ComputeUnitCore *> cus;
  CompletionTracker tracker;
  InterruptSubscription subscription;
  CompletionEndCounter *end_counter = nullptr;
  std::vector<HwQueueState> queues;
  uint32_t retired = 0;
  uint32_t interrupts = 0;
};

void expect_retry_then_exactly_once(CompletionRetryMemory::Operation operation, uint64_t address) {
  CompletionFixture fixture;
  fixture.memory->fail_next(operation, address);

  const CompletionDrainResult first = fixture.tracker.drain_completions(fixture.queues);
  EXPECT_TRUE(first.retry_pending);
  EXPECT_FALSE(first.terminal_fault);
  ASSERT_EQ(fixture.queues.front().entries.size(), 1u);
  EXPECT_EQ(fixture.retired, 0u);
  EXPECT_EQ(fixture.interrupts, 0u);
  ASSERT_NE(fixture.end_counter, nullptr);
  EXPECT_EQ(fixture.end_counter->calls, 1u);

  const CompletionDrainResult second = fixture.tracker.drain_completions(fixture.queues);
  EXPECT_FALSE(second.retry_pending);
  EXPECT_FALSE(second.terminal_fault);
  EXPECT_TRUE(fixture.queues.front().entries.empty());
  EXPECT_EQ(fixture.retired, 1u);
  EXPECT_EQ(fixture.interrupts, 1u);
  EXPECT_EQ(fixture.end_counter->calls, 1u);
  EXPECT_EQ(
      fixture.memory->load<uint64_t>(CompletionFixture::kSignal + CompletionFixture::kValueOffset),
      0u);
  EXPECT_EQ(fixture.memory->successful_compare_exchanges(), 1u);
  EXPECT_EQ(fixture.memory->successful_stores(CompletionFixture::kMailbox), 1u);
  EXPECT_EQ(fixture.memory->successful_stores(CompletionFixture::kSignal +
                                              CompletionFixture::kStartOffset),
            1u);
  EXPECT_EQ(
      fixture.memory->successful_stores(CompletionFixture::kSignal + CompletionFixture::kEndOffset),
      1u);
}

TEST(CompletionTrackerTest, RetriesUnavailableStartTimestampBeforeSignalDecrement) {
  expect_retry_then_exactly_once(CompletionRetryMemory::Operation::AtomicStore,
                                 CompletionFixture::kSignal + CompletionFixture::kStartOffset);
}

TEST(CompletionTrackerTest, RetriesUnavailableSignalCompareExchange) {
  expect_retry_then_exactly_once(CompletionRetryMemory::Operation::CompareExchange,
                                 CompletionFixture::kSignal + CompletionFixture::kValueOffset);
}

TEST(CompletionTrackerTest, RetriesUnavailableMetadataReadWithoutRepeatingEarlierStages) {
  expect_retry_then_exactly_once(CompletionRetryMemory::Operation::Read,
                                 CompletionFixture::kSignal + CompletionFixture::kEventOffset);
}

TEST(CompletionTrackerTest, RetriesUnavailableMailboxAfterExactlyOneSignalDecrement) {
  expect_retry_then_exactly_once(CompletionRetryMemory::Operation::AtomicStore,
                                 CompletionFixture::kMailbox);
}

TEST(CompletionTrackerTest, RetryKeepsTheAddressSpaceSnapshotThatStartedPublication) {
  CompletionFixture fixture;
  fixture.memory->fail_next(CompletionRetryMemory::Operation::CompareExchange,
                            CompletionFixture::kSignal + CompletionFixture::kValueOffset);

  const CompletionDrainResult first = fixture.tracker.drain_completions(fixture.queues);
  ASSERT_TRUE(first.retry_pending);

  auto replacement = std::make_shared<CompletionRetryMemory>();
  replacement->store(CompletionFixture::kSignal + CompletionFixture::kValueOffset, uint64_t{1});
  replacement->store(CompletionFixture::kSignal + CompletionFixture::kMailboxOffset,
                     CompletionFixture::kMailbox);
  replacement->store(CompletionFixture::kSignal + CompletionFixture::kEventOffset,
                     CompletionFixture::kEventId);
  ASSERT_TRUE(fixture.vm.replace_translated(fixture.address_space, replacement, replacement));

  const CompletionDrainResult second = fixture.tracker.drain_completions(fixture.queues);
  EXPECT_FALSE(second.retry_pending);
  EXPECT_FALSE(second.terminal_fault);
  EXPECT_EQ(
      fixture.memory->load<uint64_t>(CompletionFixture::kSignal + CompletionFixture::kValueOffset),
      0u);
  EXPECT_EQ(
      replacement->load<uint64_t>(CompletionFixture::kSignal + CompletionFixture::kValueOffset),
      1u);
  EXPECT_EQ(fixture.memory->successful_stores(CompletionFixture::kMailbox), 1u);
  EXPECT_EQ(replacement->successful_stores(CompletionFixture::kMailbox), 0u);
}

TEST(CompletionTrackerTest, UnavailableQueueDoesNotStarveIndependentQueueRetirement) {
  constexpr uint64_t kSecondSignal = 0x400;
  constexpr uint64_t kSecondMailbox = 0x480;
  constexpr uint32_t kSecondEvent = 37;

  CompletionFixture fixture;
  fixture.memory->store(kSecondSignal + CompletionFixture::kValueOffset, uint64_t{1});
  fixture.memory->store(kSecondSignal + CompletionFixture::kMailboxOffset, kSecondMailbox);
  fixture.memory->store(kSecondSignal + CompletionFixture::kEventOffset, kSecondEvent);

  HwQueueState second_queue{};
  second_queue.fanout_replica = true;
  DispatchEntry second_entry{};
  second_entry.dispatch_id = 8;
  second_entry.queue_id = CompletionFixture::kQueueId + 1;
  second_entry.address_space = fixture.address_space;
  second_entry.interrupt_sink = fixture.subscription.sink();
  second_entry.process_id = CompletionFixture::kProcessId;
  second_entry.total_wgs = 1;
  second_entry.completed_wgs = 1;
  second_entry.completion_signal = kSecondSignal;
  second_entry.kind = DispatchPacketKind::Kernel;
  second_queue.push_entry(std::move(second_entry));
  fixture.queues.push_back(std::move(second_queue));
  fixture.memory->fail_next(CompletionRetryMemory::Operation::AtomicStore,
                            CompletionFixture::kSignal + CompletionFixture::kStartOffset);

  const CompletionDrainResult first = fixture.tracker.drain_completions(fixture.queues);
  EXPECT_TRUE(first.retry_pending);
  ASSERT_EQ(fixture.queues.front().entries.size(), 1u);
  EXPECT_TRUE(fixture.queues.back().entries.empty());
  EXPECT_EQ(
      fixture.memory->load<uint64_t>(CompletionFixture::kSignal + CompletionFixture::kValueOffset),
      1u);
  EXPECT_EQ(fixture.memory->load<uint64_t>(kSecondSignal + CompletionFixture::kValueOffset), 0u);
  EXPECT_EQ(fixture.interrupts, 1u);
  EXPECT_EQ(fixture.retired, 1u);

  const CompletionDrainResult second = fixture.tracker.drain_completions(fixture.queues);
  EXPECT_FALSE(second.retry_pending);
  EXPECT_TRUE(fixture.queues.front().entries.empty());
  EXPECT_EQ(fixture.interrupts, 2u);
  EXPECT_EQ(fixture.retired, 2u);
}

TEST(CompletionTrackerTest, ReportsTerminalPublicationFaultWithoutRetiringEntry) {
  CompletionFixture fixture;
  fixture.memory->fail_next(CompletionRetryMemory::Operation::CompareExchange,
                            CompletionFixture::kSignal + CompletionFixture::kValueOffset,
                            VmAccessOutcome::Faulted);

  const CompletionDrainResult result = fixture.tracker.drain_completions(fixture.queues);
  ASSERT_TRUE(result.terminal_fault);
  EXPECT_EQ(result.terminal_fault->queue_id, CompletionFixture::kQueueId);
  EXPECT_EQ(result.terminal_fault->dispatch_id, 7u);
  EXPECT_EQ(result.terminal_fault->outcome, VmAccessOutcome::Faulted);
  ASSERT_EQ(fixture.queues.front().entries.size(), 1u);
  EXPECT_TRUE(fixture.queues.front().entries.front().terminal_faulted);
  EXPECT_EQ(fixture.retired, 0u);
  EXPECT_EQ(fixture.interrupts, 0u);
}

TEST(CompletionTrackerTest, QueueIdlePublicationSurvivesUnavailableMailboxStore) {
  constexpr uint64_t kQueueDescriptor = 0x500;
  constexpr uint64_t kIdleSignal = 0x200;
  constexpr uint64_t kIdleMailbox = 0x380;
  constexpr uint32_t kIdleEvent = 41;

  CompletionFixture fixture;
  fixture.queues.front().fanout_replica = false;
  fixture.queues.front().address_space = fixture.address_space;
  fixture.queues.front().interrupt_sink = fixture.subscription.sink();
  fixture.queues.front().queue_desc_va = kQueueDescriptor;
  fixture.queues.front().entries.front().completion_signal = 0;
  fixture.memory->store(kQueueDescriptor + offsetof(amd_queue_t, queue_inactive_signal),
                        kIdleSignal);
  fixture.memory->store(kIdleSignal + CompletionFixture::kValueOffset, uint64_t{0});
  fixture.memory->store(kIdleSignal + CompletionFixture::kMailboxOffset, kIdleMailbox);
  fixture.memory->store(kIdleSignal + CompletionFixture::kEventOffset, kIdleEvent);
  fixture.subscription.reset();
  fixture.subscription = InterruptSubscription([&fixture](uint32_t process_id, uint32_t event_id) {
    EXPECT_EQ(process_id, CompletionFixture::kProcessId);
    if (event_id == kIdleEvent || event_id == 0)
      ++fixture.interrupts;
  });
  fixture.queues.front().interrupt_sink = fixture.subscription.sink();
  fixture.memory->fail_next(CompletionRetryMemory::Operation::AtomicStore, kIdleMailbox);

  const CompletionDrainResult first = fixture.tracker.drain_completions(fixture.queues);
  EXPECT_TRUE(first.retry_pending);
  EXPECT_TRUE(fixture.queues.front().entries.empty());
  EXPECT_TRUE(fixture.queues.front().idle_publication.active());
  EXPECT_FALSE(fixture.tracker.all_complete(fixture.queues));
  EXPECT_EQ(fixture.memory->load<uint64_t>(kIdleSignal + CompletionFixture::kValueOffset), 0x10u);
  EXPECT_EQ(fixture.interrupts, 0u);

  const CompletionDrainResult second = fixture.tracker.drain_completions(fixture.queues);
  EXPECT_FALSE(second.retry_pending);
  EXPECT_FALSE(second.terminal_fault);
  EXPECT_FALSE(fixture.queues.front().idle_publication.active());
  EXPECT_TRUE(fixture.tracker.all_complete(fixture.queues));
  EXPECT_EQ(fixture.memory->successful_stores(kIdleMailbox), 1u);
  EXPECT_EQ(fixture.interrupts, 2u);
}

TEST(CompletionTrackerTest, NewQueueGenerationFinishesCommittedIdleRetry) {
  constexpr uint64_t kQueueDescriptor = 0x500;
  constexpr uint64_t kIdleSignal = 0x200;
  constexpr uint64_t kIdleMailbox = 0x380;
  constexpr uint32_t kIdleEvent = 41;

  CompletionFixture fixture;
  auto &queue = fixture.queues.front();
  queue.fanout_replica = false;
  queue.address_space = fixture.address_space;
  queue.interrupt_sink = fixture.subscription.sink();
  queue.queue_desc_va = kQueueDescriptor;
  queue.entries.front().completion_signal = 0;
  fixture.memory->store(kQueueDescriptor + offsetof(amd_queue_t, queue_inactive_signal),
                        kIdleSignal);
  fixture.memory->store(kIdleSignal + CompletionFixture::kValueOffset, uint64_t{0});
  fixture.memory->store(kIdleSignal + CompletionFixture::kMailboxOffset, kIdleMailbox);
  fixture.memory->store(kIdleSignal + CompletionFixture::kEventOffset, kIdleEvent);
  fixture.subscription.reset();
  fixture.subscription =
      InterruptSubscription([&fixture](uint32_t, uint32_t) { ++fixture.interrupts; });
  queue.interrupt_sink = fixture.subscription.sink();
  fixture.memory->fail_next(CompletionRetryMemory::Operation::AtomicStore, kIdleMailbox);

  const CompletionDrainResult first = fixture.tracker.drain_completions(fixture.queues);
  ASSERT_TRUE(first.retry_pending);
  ASSERT_TRUE(queue.entries.empty());
  ASSERT_TRUE(queue.idle_publication.active());
  EXPECT_EQ(queue.idle_publication.phase, QueueIdlePublicationPhase::StoreMailbox);
  EXPECT_EQ(fixture.memory->load<uint64_t>(kIdleSignal + CompletionFixture::kValueOffset), 0x10u);
  EXPECT_EQ(fixture.memory->successful_compare_exchanges(), 1u);

  DispatchEntry next{};
  next.dispatch_id = 8;
  next.queue_id = CompletionFixture::kQueueId;
  next.address_space = fixture.address_space;
  next.interrupt_sink = fixture.subscription.sink();
  next.process_id = CompletionFixture::kProcessId;
  next.total_wgs = 1;
  next.kind = DispatchPacketKind::Kernel;
  queue.push_entry(std::move(next));

  const CompletionDrainResult while_active = fixture.tracker.drain_completions(fixture.queues);
  EXPECT_FALSE(while_active.retry_pending);
  EXPECT_FALSE(while_active.terminal_fault);
  ASSERT_EQ(queue.entries.size(), 1u);
  EXPECT_FALSE(queue.idle_publication.active());
  EXPECT_FALSE(fixture.tracker.all_complete(fixture.queues));
  EXPECT_EQ(fixture.memory->load<uint64_t>(kIdleSignal + CompletionFixture::kValueOffset), 0x10u);
  EXPECT_EQ(fixture.memory->load<uint64_t>(kIdleMailbox), kIdleEvent);
  EXPECT_EQ(fixture.memory->successful_compare_exchanges(), 1u);
  EXPECT_EQ(fixture.interrupts, 2u);
  EXPECT_EQ(fixture.memory->successful_stores(kIdleMailbox), 1u);

  queue.entries.front().completed_wgs = 1;
  const CompletionDrainResult final = fixture.tracker.drain_completions(fixture.queues);
  EXPECT_FALSE(final.retry_pending);
  EXPECT_FALSE(final.terminal_fault);
  EXPECT_TRUE(queue.entries.empty());
  EXPECT_FALSE(queue.idle_publication.active());
  EXPECT_EQ(fixture.memory->successful_stores(kIdleMailbox), 2u);
  EXPECT_EQ(fixture.memory->successful_compare_exchanges(), 1u);
  EXPECT_EQ(fixture.interrupts, 4u);
}

TEST(CompletionTrackerTest, NewQueueGenerationCancelsUncommittedIdleRetry) {
  constexpr uint64_t kQueueDescriptor = 0x500;
  constexpr uint64_t kIdleSignal = 0x200;
  constexpr uint64_t kIdleMailbox = 0x380;

  CompletionFixture fixture;
  auto &queue = fixture.queues.front();
  queue.fanout_replica = false;
  queue.address_space = fixture.address_space;
  queue.interrupt_sink = fixture.subscription.sink();
  queue.queue_desc_va = kQueueDescriptor;
  queue.entries.front().completion_signal = 0;
  fixture.memory->store(kQueueDescriptor + offsetof(amd_queue_t, queue_inactive_signal),
                        kIdleSignal);
  fixture.memory->store(kIdleSignal + CompletionFixture::kValueOffset, uint64_t{0});
  fixture.memory->store(kIdleSignal + CompletionFixture::kMailboxOffset, kIdleMailbox);
  fixture.memory->store(kIdleSignal + CompletionFixture::kEventOffset, uint32_t{41});
  fixture.memory->fail_next(CompletionRetryMemory::Operation::CompareExchange,
                            kIdleSignal + CompletionFixture::kValueOffset);

  const CompletionDrainResult first = fixture.tracker.drain_completions(fixture.queues);
  ASSERT_TRUE(first.retry_pending);
  ASSERT_TRUE(queue.entries.empty());
  ASSERT_TRUE(queue.idle_publication.active());
  EXPECT_EQ(queue.idle_publication.phase, QueueIdlePublicationPhase::StoreIdleStatus);
  EXPECT_EQ(fixture.memory->load<uint64_t>(kIdleSignal + CompletionFixture::kValueOffset), 0u);
  EXPECT_EQ(fixture.memory->successful_compare_exchanges(), 0u);

  DispatchEntry next{};
  next.dispatch_id = 8;
  next.queue_id = CompletionFixture::kQueueId;
  next.address_space = fixture.address_space;
  next.interrupt_sink = fixture.subscription.sink();
  next.process_id = CompletionFixture::kProcessId;
  next.total_wgs = 1;
  next.kind = DispatchPacketKind::Kernel;
  queue.push_entry(std::move(next));

  const CompletionDrainResult canceled = fixture.tracker.drain_completions(fixture.queues);
  EXPECT_FALSE(canceled.retry_pending);
  EXPECT_FALSE(canceled.terminal_fault);
  ASSERT_EQ(queue.entries.size(), 1u);
  EXPECT_FALSE(queue.idle_publication.active());
  EXPECT_EQ(fixture.memory->load<uint64_t>(kIdleSignal + CompletionFixture::kValueOffset), 0u);
  EXPECT_EQ(fixture.memory->successful_stores(kIdleMailbox), 0u);
  EXPECT_EQ(fixture.interrupts, 0u);

  queue.entries.front().completed_wgs = 1;
  const CompletionDrainResult final = fixture.tracker.drain_completions(fixture.queues);
  EXPECT_FALSE(final.retry_pending);
  EXPECT_FALSE(final.terminal_fault);
  EXPECT_TRUE(queue.entries.empty());
  EXPECT_FALSE(queue.idle_publication.active());
  EXPECT_EQ(fixture.memory->load<uint64_t>(kIdleSignal + CompletionFixture::kValueOffset), 0x10u);
  EXPECT_EQ(fixture.memory->successful_compare_exchanges(), 1u);
  EXPECT_EQ(fixture.memory->successful_stores(kIdleMailbox), 1u);
  EXPECT_EQ(fixture.interrupts, 2u);
}

} // namespace
