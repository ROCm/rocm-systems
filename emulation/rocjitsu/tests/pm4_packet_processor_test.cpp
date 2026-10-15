// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pm4/pm4_packet_processor.h"

#include "legacy_gpu_memory_fixture.h"

#include "rocjitsu/vm/amdgpu/dispatch_entry.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

class FlatMemory final : public AddressSpaceTranslator, public PhysicalMemoryAccess {
public:
  explicit FlatMemory(size_t size = 0x4000) : bytes_(size) {}

  VmTranslationResult translate(uint64_t address, std::size_t size, VmAccessKind) const override {
    if (size == 0 || address > bytes_.size() || size > bytes_.size() - address)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation = {.domain = VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = bytes_.size() - address,
                        .mtype = Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = false}}};
  }

  VmAccessOutcome read(VmMemoryDomain, uint64_t address, std::span<std::byte> bytes) override {
    if (unavailable_read_ && address == *unavailable_read_)
      return VmAccessOutcome::Unavailable;
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::ranges::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(address), bytes.size(),
                        bytes.begin());
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(VmMemoryDomain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::ranges::copy(bytes, bytes_.begin() + static_cast<std::ptrdiff_t>(address));
    return VmAccessOutcome::Complete;
  }

  AtomicLoadResult atomic_load(VmMemoryDomain, uint64_t address, uint32_t width) override {
    atomic_loads.emplace_back(address, width);
    if (unavailable_read_ && address == *unavailable_read_)
      return {.outcome = VmAccessOutcome::Unavailable};
    if ((width != 4 && width != 8) || address % width != 0 || address > bytes_.size() ||
        width > bytes_.size() - address)
      return {.outcome = VmAccessOutcome::Malformed};
    uint64_t value = 0;
    std::memcpy(&value, bytes_.data() + address, width);
    return {.outcome = VmAccessOutcome::Complete, .value = value};
  }

  VmAccessOutcome atomic_store(VmMemoryDomain, uint64_t address, uint32_t width,
                               uint64_t value) override {
    if ((width != 4 && width != 8) || address % width != 0 || address > bytes_.size() ||
        width > bytes_.size() - address)
      return VmAccessOutcome::Malformed;
    if (store_outcome_once_ && address == store_outcome_once_->first) {
      const VmAccessOutcome outcome = store_outcome_once_->second;
      store_outcome_once_.reset();
      return outcome;
    }
    if (unavailable_store_once_ && address == *unavailable_store_once_) {
      unavailable_store_once_.reset();
      return VmAccessOutcome::Unavailable;
    }
    std::memcpy(bytes_.data() + address, &value, width);
    return VmAccessOutcome::Complete;
  }

  AtomicCompareExchangeResult compare_exchange(VmMemoryDomain, uint64_t, uint32_t, uint64_t,
                                               uint64_t) override {
    return {.outcome = VmAccessOutcome::Malformed};
  }

  template <typename T> void store(uint64_t address, T value) {
    std::memcpy(bytes_.data() + address, &value, sizeof(value));
  }

  template <typename T> T load(uint64_t address) const {
    T value{};
    std::memcpy(&value, bytes_.data() + address, sizeof(value));
    return value;
  }

  void make_read_unavailable(uint64_t address) { unavailable_read_ = address; }
  void make_reads_available() { unavailable_read_.reset(); }
  void make_store_unavailable_once(uint64_t address) { unavailable_store_once_ = address; }
  void make_store_faulted_once(uint64_t address) {
    store_outcome_once_ = std::pair{address, VmAccessOutcome::Faulted};
  }

  std::vector<std::pair<uint64_t, uint32_t>> atomic_loads;

private:
  std::vector<std::byte> bytes_;
  std::optional<uint64_t> unavailable_read_;
  std::optional<uint64_t> unavailable_store_once_;
  std::optional<std::pair<uint64_t, VmAccessOutcome>> store_outcome_once_;
};

enum class Pm4TestStatus { Ready, Blocked, Faulted };

class Pm4PacketProcessorTest : public ::testing::Test {
protected:
  static constexpr uint64_t kRing = 0x100;
  static constexpr uint64_t kReadPointer = 0x80;
  static constexpr uint32_t kOutput = 0x800;

  Pm4PacketProcessorTest() : memory(std::make_shared<FlatMemory>()) {
    address_space = vm.register_translated(1, memory, memory);
    context.arch = ROCJITSU_CODE_ARCH_RDNA4;
    context.retry = [this] {
      queue.command_retry_pending = true;
      status = Pm4TestStatus::Blocked;
    };
    context.fault_queue = [this] {
      queue.faulted = true;
      status = Pm4TestStatus::Faulted;
    };
    context.wake = [] {};
    context.flush_caches = [] {};
    context.dispatch = [this](const std::array<uint32_t, 4> &dimensions) {
      dispatches.push_back(dimensions);
      queue.dispatches.push_entry({});
    };
  }

  void submit(std::span<const uint32_t> words, bool ring = false, uint64_t initial_cursor = 0,
              uint32_t ring_dwords = 64) {
    queue.address_space = address_space;
    queue.packet_format = QueuePacketFormat::Pm4;
    queue.submission_queue = !ring;
    queue.read_ptr_va = kReadPointer;
    queue.read_pointer_journal = ConsumerCursorJournal(kReadPointer, initial_cursor, 4);
    for (size_t word_index = 0; word_index < words.size(); ++word_index) {
      const uint64_t offset = ring ? (initial_cursor + word_index) % ring_dwords : word_index;
      memory->store<uint32_t>(kRing + offset * 4, words[word_index]);
    }
    Pm4Submission submission;
    submission.buffers.push_back({.address = ring ? initial_cursor * 4 : kRing,
                                  .dwords = static_cast<uint32_t>(words.size()),
                                  .ring_base = ring ? kRing : 0,
                                  .ring_bytes = ring ? ring_dwords * 4 : 0});
    submission.complete = [this](bool success) { completed = success; };
    queue.commands.submissions.push_back(std::move(submission));
  }

  Pm4TestStatus service() {
    // Model CP resuming the queue on its next scheduling turn.
    queue.command_retry_pending = false;
    status = Pm4TestStatus::Ready;
    process_pm4_packets(queue, &vm, context);
    return status;
  }

  std::shared_ptr<FlatMemory> memory;
  GpuVm vm;
  AddressSpaceHandle address_space;
  ComputeQueueRecord queue;
  Pm4ExecutionContext context;
  std::vector<std::array<uint32_t, 4>> dispatches;
  bool completed = false;
  Pm4TestStatus status = Pm4TestStatus::Ready;
};

TEST_F(Pm4PacketProcessorTest, ProcessesRegisterWriteAcrossNativeRingWrap) {
  const std::array words{0xc0017900u, 0x40u, 0xdeadbeefu, 0xffff1000u};
  submit(words, true, 3, 4);
  uint32_t writes = 0;
  queue.packet_callbacks.write_uconfig_register = [&](uint64_t reg, uint32_t value) {
    EXPECT_EQ(reg, 0xc040u);
    EXPECT_EQ(value, 0xdeadbeefu);
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  };
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(writes, 1u);
  EXPECT_TRUE(completed);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 7u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
}

TEST_F(Pm4PacketProcessorTest, MaskedPredExecRetainsSkippedCursorAcrossPublicationRetry) {
  context.arch = ROCJITSU_CODE_ARCH_CDNA3;
  context.xcc_id = 0;
  const std::array words{0xc0002300u, (1u << 25) | 5u, 0xc0033700u, 5u << 8, kOutput, 0u, 99u};
  submit(words, true, 6, 8);
  memory->make_store_unavailable_once(kReadPointer);
  EXPECT_EQ(service(), Pm4TestStatus::Blocked);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 13u);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
  EXPECT_TRUE(queue.read_pointer_journal.publication_pending());
  EXPECT_FALSE(completed);
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 5u);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
  EXPECT_TRUE(completed);
}

TEST_F(Pm4PacketProcessorTest, CondExecRetainsSkippedCursorAcrossPublicationRetry) {
  constexpr uint32_t kPredicate = kOutput + 16;
  const std::array words{0xc0032200u, kPredicate, 0u,      0u, 5u,
                         0xc0033700u, 5u << 8,    kOutput, 0u, 99u};
  memory->store(kPredicate, uint32_t{0});
  submit(words, true, 14, 16);
  memory->make_store_unavailable_once(kReadPointer);
  EXPECT_EQ(service(), Pm4TestStatus::Blocked);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 24u);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
  EXPECT_TRUE(queue.read_pointer_journal.publication_pending());
  EXPECT_FALSE(completed);
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 8u);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
  EXPECT_TRUE(completed);
}

TEST_F(Pm4PacketProcessorTest, PublishesCommittedEffectBeforeFetchingAnotherPacket) {
  const std::array words{0xc0017900u, 0x40u, 7u, 0xffff1000u};
  submit(words, true);
  uint32_t writes = 0;
  queue.packet_callbacks.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  };
  memory->make_store_unavailable_once(kReadPointer);
  EXPECT_EQ(service(), Pm4TestStatus::Blocked);
  EXPECT_EQ(writes, 1u);
  EXPECT_TRUE(queue.read_pointer_journal.publication_pending());
  EXPECT_FALSE(completed);
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 4u);
  EXPECT_TRUE(completed);
}

TEST_F(Pm4PacketProcessorTest, RetainsTerminalPublicationFailureForCpCancellation) {
  const std::array words{0xc0017900u, 0x40u, 7u};
  submit(words, true);
  uint32_t writes = 0;
  queue.packet_callbacks.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  };
  memory->make_store_faulted_once(kReadPointer);
  EXPECT_EQ(service(), Pm4TestStatus::Faulted);
  EXPECT_EQ(writes, 1u);
  EXPECT_TRUE(queue.publication_faulted);
  EXPECT_TRUE(queue.read_pointer_journal.publication_pending());
  EXPECT_FALSE(completed);
}

TEST_F(Pm4PacketProcessorTest, RetriesRegisterBackpressureWithoutAdvancingThePacket) {
  const std::array words{0xc0017900u, 0x40u, 7u};
  submit(words, true);
  bool ready = false;
  uint32_t writes = 0;
  queue.packet_callbacks.write_uconfig_register = [&](uint64_t, uint32_t) {
    if (!ready)
      return Pm4RegisterWriteStatus::Blocked;
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  };
  EXPECT_EQ(service(), Pm4TestStatus::Blocked);
  EXPECT_EQ(writes, 0u);
  EXPECT_EQ(queue.commands.submissions.front().buffers.front().address, 0u);
  ready = true;
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
}

TEST_F(Pm4PacketProcessorTest, RetriesUnavailableFetchWithoutApplyingAnEffect) {
  const std::array words{0xc0017900u, 0x40u, 7u};
  submit(words, true);
  uint32_t writes = 0;
  queue.packet_callbacks.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  };
  memory->make_read_unavailable(kRing + 4);
  EXPECT_EQ(service(), Pm4TestStatus::Blocked);
  EXPECT_EQ(writes, 0u);
  EXPECT_EQ(queue.commands.submissions.front().buffers.front().address, 0u);
}

TEST_F(Pm4PacketProcessorTest, RevokedTransactionBecomesTerminalInsteadOfBlocked) {
  const std::array words{0xffff1000u};
  submit(words, true);
  memory->make_read_unavailable(kRing);
  EXPECT_EQ(service(), Pm4TestStatus::Blocked);
  ASSERT_TRUE(queue.command_access);
  ASSERT_TRUE(vm.invalidate(address_space));

  EXPECT_EQ(service(), Pm4TestStatus::Faulted);
  EXPECT_TRUE(queue.faulted);
  EXPECT_FALSE(queue.command_retry_pending);
  EXPECT_FALSE(completed);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 0u);
}

TEST_F(Pm4PacketProcessorTest, RejectsTruncatedPacketBeforeCallingItsHandler) {
  const std::array words{0xc0017900u, 0x40u};
  submit(words, true);
  uint32_t writes = 0;
  queue.packet_callbacks.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  };
  EXPECT_EQ(service(), Pm4TestStatus::Faulted);
  EXPECT_EQ(writes, 0u);
  EXPECT_FALSE(completed);
}

TEST_F(Pm4PacketProcessorTest, RetiresSupportedNopEncodings) {
  const std::array words{0x80000000u, 0xc0001000u, 0u, 0xffff1000u};
  submit(words, true);
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 4u);
  EXPECT_TRUE(completed);
}

TEST_F(Pm4PacketProcessorTest, YieldsAtTheNativePacketBudgetAndResumesWithoutReplay) {
  const std::vector<uint32_t> words(300, 0xffff1000u);
  submit(words, true, 0, 512);
  EXPECT_EQ(service(), Pm4TestStatus::Blocked);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 256u);
  EXPECT_FALSE(completed);
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 300u);
  EXPECT_TRUE(completed);
}

TEST_F(Pm4PacketProcessorTest, RejectsInvalidRegisterOffsetsBeforeApplyingAnEffect) {
  const std::array words{0xc0017900u, 0x10040u, 7u};
  submit(words, true);
  uint32_t writes = 0;
  queue.packet_callbacks.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  };
  EXPECT_EQ(service(), Pm4TestStatus::Faulted);
  EXPECT_EQ(writes, 0u);
}

TEST_F(Pm4PacketProcessorTest, ReturnsFromAnIndirectBufferWithPersistentRegisterState) {
  const std::array child{0xc0017600u, kPm4ComputeNumThreadY, 8u};
  for (size_t word_index = 0; word_index < child.size(); ++word_index)
    memory->store<uint32_t>(0x400 + word_index * 4, child[word_index]);
  const std::array words{0xc0017600u, kPm4ComputeNumThreadX, 4u, 0xc0023f00u, 0x400u, 0u, 3u,
                         0xc0017600u, kPm4ComputeNumThreadZ, 2u};
  submit(words);
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(queue.commands.sh_registers[kPm4ComputeNumThreadX], 4u);
  EXPECT_EQ(queue.commands.sh_registers[kPm4ComputeNumThreadY], 8u);
  EXPECT_EQ(queue.commands.sh_registers[kPm4ComputeNumThreadZ], 2u);
  EXPECT_TRUE(completed);
}

TEST_F(Pm4PacketProcessorTest, WaitBlocksLaterMemoryEffectsUntilItsConditionPasses) {
  const std::array words{0xc0053c00u, 0x13u,       kOutput, 0u,          1u, ~0u,
                         0u,          0xc0033700u, 0x100u,  kOutput + 4, 0u, 0xdeadbeefu};
  submit(words);
  EXPECT_EQ(service(), Pm4TestStatus::Blocked);
  EXPECT_EQ(memory->load<uint32_t>(kOutput + 4), 0u);
  EXPECT_EQ(queue.commands.submissions.front().buffers.front().address, kRing);
  memory->store<uint32_t>(kOutput, 1);
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(memory->load<uint32_t>(kOutput + 4), 0xdeadbeefu);
  EXPECT_TRUE(completed);
}

TEST_F(Pm4PacketProcessorTest, WideWaitUsesMaskedHighBitsAndRetriesUnavailableMemory) {
  constexpr uint64_t reference = 0x8000000000000010ull;
  constexpr uint64_t mask = 0xffff00000000ffffull;
  const std::array<uint64_t, 6> blocked{reference + (1ull << 48), reference + (1ull << 48),
                                        reference + (1ull << 48), reference,
                                        reference - (1ull << 48), reference};
  const std::array<uint64_t, 6> ready{
      reference - (1ull << 48), reference, reference,
      reference + (1ull << 48), reference, reference + (1ull << 48)};
  for (uint32_t function = 1; function <= 6; ++function) {
    SCOPED_TRACE(function);
    const std::array words{0xc0079300u,
                           function | 0x10u,
                           kOutput,
                           0u,
                           uint32_t(reference),
                           uint32_t(reference >> 32),
                           uint32_t(mask),
                           uint32_t(mask >> 32),
                           0u,
                           0xc0033700u,
                           0x100u,
                           kOutput + 16,
                           0u,
                           function};
    memory->atomic_loads.clear();
    memory->store<uint64_t>(kOutput, blocked[function - 1] | 0x0000123456780000ull);
    submit(words);
    EXPECT_EQ(service(), Pm4TestStatus::Blocked);
    ASSERT_EQ(memory->atomic_loads.size(), 1u);
    EXPECT_EQ(memory->atomic_loads.back(), (std::pair<uint64_t, uint32_t>{kOutput, 8}));
    EXPECT_EQ(memory->load<uint32_t>(kOutput + 16), function - 1);
    EXPECT_EQ(queue.commands.submissions.front().buffers.front().address, kRing);
    memory->store<uint64_t>(kOutput, ready[function - 1] | 0x0000fedcba980000ull);
    memory->make_read_unavailable(kOutput);
    EXPECT_EQ(service(), Pm4TestStatus::Blocked);
    EXPECT_EQ(memory->load<uint32_t>(kOutput + 16), function - 1);
    memory->make_reads_available();
    EXPECT_EQ(service(), Pm4TestStatus::Ready);
    EXPECT_EQ(memory->load<uint32_t>(kOutput + 16), function);
    EXPECT_EQ(memory->load<uint64_t>(kOutput + 8), 0u);
  }
}

TEST_F(Pm4PacketProcessorTest, WideWaitFaultPreventsLaterMemoryEffects) {
  for (uint32_t operand : {kOutput + 4, 0x8000u}) {
    SCOPED_TRACE(operand);
    completed = false;
    memory->store<uint32_t>(kOutput + 16, 0);
    queue.faulted = false;
    queue.commands.submissions.clear();
    queue.command_access.reset();
    const std::array words{0xc0079300u, 0x13u, operand,     0u,     0u,           0u, ~0u,
                           ~0u,         0u,    0xc0033700u, 0x100u, kOutput + 16, 0u, 99u};
    submit(words);
    EXPECT_EQ(service(), Pm4TestStatus::Faulted);
    EXPECT_TRUE(queue.faulted);
    EXPECT_FALSE(completed);
    EXPECT_EQ(memory->load<uint32_t>(kOutput + 16), 0u);
  }
}

TEST_F(Pm4PacketProcessorTest, ConcurrentWideWaitCannotPassOnMixedHalves) {
  LegacyPageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  test::LegacyGpuMemoryFixture host_memory("memory");
  host_memory.set_passthrough(true);
  host_memory.register_process(7, &page_table, &page_table_mutex);
  GpuVm &gpu_vm = host_memory.gpu_vm();
  const std::optional<AddressSpaceHandle> handle = gpu_vm.find_vmid(7);
  ASSERT_TRUE(handle);
  constexpr uint64_t first = 0x1111111122222222ull;
  constexpr uint64_t second = 0x3333333344444444ull;
  constexpr uint64_t mixed = 0x1111111144444444ull;
  alignas(8) uint64_t operand = first;
  uint32_t output = 0;
  const uint64_t operand_address = reinterpret_cast<uint64_t>(&operand);
  const uint64_t output_address = reinterpret_cast<uint64_t>(&output);
  const std::array words{0xc0079300u,
                         0x13u,
                         uint32_t(operand_address),
                         uint32_t(operand_address >> 32),
                         uint32_t(mixed),
                         uint32_t(mixed >> 32),
                         ~0u,
                         ~0u,
                         0u,
                         0xc0033700u,
                         0x100u,
                         uint32_t(output_address),
                         uint32_t(output_address >> 32),
                         0xdeadbeefu};
  queue.address_space = *handle;
  queue.process_id = 7;
  queue.packet_format = QueuePacketFormat::Pm4;
  queue.submission_queue = true;
  Pm4Submission submission;
  const uint64_t commands = reinterpret_cast<uint64_t>(words.data());
  submission.buffers.push_back({.address = commands, .dwords = uint32_t(words.size())});
  submission.complete = [this](bool success) { completed = success; };
  queue.commands.submissions.push_back(std::move(submission));
  auto service_host = [&] {
    queue.command_retry_pending = false;
    status = Pm4TestStatus::Ready;
    process_pm4_packets(queue, &gpu_vm, context);
    return status;
  };
  std::atomic_ref<uint64_t> published(operand);
  std::atomic<uint32_t> stores = 0;
  std::jthread producer([&](std::stop_token stop) {
    while (!stop.stop_requested()) {
      published.store(first, std::memory_order_release);
      published.store(second, std::memory_order_release);
      stores.fetch_add(1, std::memory_order_relaxed);
    }
  });
  // This flag provides no acquire edge for the wait operand.
  while (stores.load(std::memory_order_relaxed) == 0)
    std::this_thread::yield();
  for (uint32_t poll = 0; poll < 10000; ++poll) {
    ASSERT_EQ(service_host(), Pm4TestStatus::Blocked);
    ASSERT_EQ(output, 0u);
    ASSERT_FALSE(completed);
    ASSERT_EQ(queue.commands.submissions.front().buffers.front().address, commands);
  }
  producer.request_stop();
  producer.join();
  published.store(mixed, std::memory_order_release);
  EXPECT_EQ(service_host(), Pm4TestStatus::Ready);
  EXPECT_EQ(output, 0xdeadbeefu);
  EXPECT_TRUE(completed);
  EXPECT_FALSE(queue.faulted);
}

TEST_F(Pm4PacketProcessorTest, ReleaseInterruptFollowsWriteAndUsesContextId) {
  uint32_t calls = 0;
  uint64_t expected = 0;
  InterruptSubscription subscription([&](uint32_t process_id, uint32_t event_id) {
    ++calls;
    EXPECT_EQ(process_id, 42u);
    EXPECT_EQ(event_id, 73u);
    EXPECT_EQ(memory->load<uint64_t>(kOutput), expected);
  });
  queue.process_id = 42;
  queue.interrupt_sink = subscription.sink();
  for (uint32_t interrupt : {0u, 1u, 2u, 3u}) {
    SCOPED_TRACE(interrupt);
    calls = 0;
    memory->store<uint64_t>(kOutput, 0);
    const uint32_t selection = interrupt == 1 ? 0 : 2;
    expected = selection == 0 ? 0 : 0x123456789abcdef0ull;
    const std::array words{0xc0064900u, 0u, (selection << 29) | (interrupt << 24),
                           kOutput,     0u, 0x9abcdef0u,
                           0x12345678u, 73u};
    submit(words);
    EXPECT_EQ(service(), Pm4TestStatus::Ready);
    EXPECT_EQ(memory->load<uint64_t>(kOutput), expected);
    EXPECT_EQ(calls, interrupt == 1 || interrupt == 2 ? 1u : 0u);
    EXPECT_EQ(service(), Pm4TestStatus::Ready);
    EXPECT_EQ(calls, interrupt == 1 || interrupt == 2 ? 1u : 0u);
  }
}

TEST_F(Pm4PacketProcessorTest, UnsupportedReleaseInterruptHasNoEffects) {
  uint32_t calls = 0;
  InterruptSubscription subscription([&](uint32_t, uint32_t) { ++calls; });
  queue.interrupt_sink = subscription.sink();
  for (uint32_t interrupt : {4u, 5u, 6u, 7u}) {
    SCOPED_TRACE(interrupt);
    queue.faulted = false;
    queue.commands.submissions.clear();
    completed = false;
    memory->store<uint64_t>(kOutput, 0);
    const std::array words{0xc0064900u, 0u,           (2u << 29) | (interrupt << 24),
                           kOutput,     0u,           1u,
                           0u,          73u,          0xc0033700u,
                           0x100u,      kOutput + 16, 0u,
                           99u};
    submit(words);
    EXPECT_EQ(service(), Pm4TestStatus::Faulted);
    EXPECT_EQ(memory->load<uint64_t>(kOutput), 0u);
    EXPECT_EQ(memory->load<uint32_t>(kOutput + 16), 0u);
    EXPECT_EQ(calls, 0u);
    EXPECT_FALSE(completed);
  }
}

TEST_F(Pm4PacketProcessorTest, Gfx12WaitAndReleaseRespectDirectionalPermissions) {
  memory = std::make_shared<FlatMemory>(0xa000);
  constexpr uint64_t valid = 1, readable = 1u << 5, writable = 1u << 6;
  constexpr uint64_t leaf = uint64_t{1} << 63;
  for (uint64_t table = 0x1000; table < 0x5000; table += 0x1000)
    memory->store<uint64_t>(table, (table + 0x1000) | valid);
  memory->store<uint64_t>(0x5000, leaf | valid | readable | writable);
  memory->store<uint64_t>(0x5040, 0x8000 | leaf | valid | readable);
  memory->store<uint64_t>(0x5048, 0x9000 | leaf | valid | writable);
  auto translator = std::make_shared<Gfx12PageTableTranslator>(memory, 0x1000);
  std::vector<VmAccessKind> faults;
  address_space = vm.register_address_space(
      2, translator, memory, [&](uint64_t, VmAccessKind kind) { faults.push_back(kind); });
  const std::optional<GpuVmAccess> access = vm.snapshot(address_space);
  ASSERT_TRUE(access);
  constexpr uint64_t value = 0x123456789abcdef0ull;
  for (bool wide : {false, true}) {
    SCOPED_TRACE(wide);
    memory->store<uint64_t>(0x8000, value);
    memory->store<uint64_t>(0x9000, 0);
    std::vector<uint32_t> words{wide ? 0xc0079300u : 0xc0053c00u, 0x13u, 0x8000u, 0u,
                                uint32_t(value)};
    if (wide)
      words.push_back(uint32_t(value >> 32));
    words.push_back(UINT32_MAX);
    if (wide)
      words.push_back(UINT32_MAX);
    words.push_back(0);
    words.insert(words.end(), {0xc0064900u, 0u, (wide ? 2u : 1u) << 29, 0x9000u, 0u,
                               uint32_t(value), uint32_t(value >> 32), 0u});
    submit(words);
    EXPECT_EQ(service(), Pm4TestStatus::Ready);
    EXPECT_EQ(memory->load<uint64_t>(0x9000), wide ? value : uint32_t(value));
    EXPECT_TRUE(faults.empty());
    const uint32_t width = wide ? sizeof(uint64_t) : sizeof(uint32_t);
    EXPECT_EQ(access->atomic_load(0x9000, width).outcome, VmAccessOutcome::Faulted);
    EXPECT_EQ(access->atomic_store(0x8000, width, 0), VmAccessOutcome::Faulted);
    EXPECT_EQ(access->compare_exchange(0x8000, width, value, 0).outcome, VmAccessOutcome::Faulted);
    EXPECT_EQ(access->compare_exchange(0x9000, width, value, 0).outcome, VmAccessOutcome::Faulted);
    EXPECT_EQ(faults, (std::vector{VmAccessKind::Read, VmAccessKind::Write, VmAccessKind::Atomic,
                                   VmAccessKind::Atomic}));
    faults.clear();
  }
}

TEST_F(Pm4PacketProcessorTest, ConcurrentWaitAcquiresNarrowPublicationAndWideReleasePacket) {
  LegacyPageTable page_table;
  util::DistributedSharedMutex page_table_mutex;
  test::LegacyGpuMemoryFixture host_memory("memory");
  host_memory.set_passthrough(true);
  host_memory.register_process(7, &page_table, &page_table_mutex);
  GpuVm &gpu_vm = host_memory.gpu_vm();
  const std::optional<AddressSpaceHandle> handle = gpu_vm.find_vmid(7);
  ASSERT_TRUE(handle);
  for (bool wide : {false, true}) {
    SCOPED_TRACE(wide);
    uint64_t wide_operand = 0;
    uint32_t narrow_operand = 0, payload = 0, output = 0;
    const uint64_t operand_address = wide ? reinterpret_cast<uint64_t>(&wide_operand)
                                          : reinterpret_cast<uint64_t>(&narrow_operand);
    const uint64_t payload_address = reinterpret_cast<uint64_t>(&payload);
    const uint64_t output_address = reinterpret_cast<uint64_t>(&output);
    constexpr uint32_t expected = 0x12345678;
    std::vector<uint32_t> words{wide ? 0xc0079300u : 0xc0053c00u, 0x13u, uint32_t(operand_address),
                                uint32_t(operand_address >> 32), 1u};
    if (wide)
      words.push_back(0);
    words.push_back(UINT32_MAX);
    if (wide)
      words.push_back(UINT32_MAX);
    words.push_back(0);
    // COPY_DATA reads ordinary payload only after the acquire wait passes.
    words.insert(words.end(), {0xc0044000u, 1u | (1u << 8), uint32_t(payload_address),
                               uint32_t(payload_address >> 32), uint32_t(output_address),
                               uint32_t(output_address >> 32)});
    queue = ComputeQueueRecord{};
    queue.address_space = *handle;
    queue.packet_format = QueuePacketFormat::Pm4;
    queue.submission_queue = true;
    Pm4Submission submission;
    submission.buffers.push_back(
        {.address = reinterpret_cast<uint64_t>(words.data()), .dwords = uint32_t(words.size())});
    queue.commands.submissions.push_back(std::move(submission));
    const std::array release{
        0xc0064900u, 0u, 2u << 29, uint32_t(operand_address), uint32_t(operand_address >> 32),
        1u,          0u, 0u};
    ComputeQueueRecord producer_queue;
    producer_queue.address_space = *handle;
    producer_queue.packet_format = QueuePacketFormat::Pm4;
    producer_queue.submission_queue = true;
    Pm4Submission producer_submission;
    producer_submission.buffers.push_back({.address = reinterpret_cast<uint64_t>(release.data()),
                                           .dwords = uint32_t(release.size())});
    producer_queue.commands.submissions.push_back(std::move(producer_submission));
    Pm4ExecutionContext producer_context{};
    producer_context.flush_caches = [] {};
    producer_context.wake = [] {};
    producer_context.fault_queue = [&] { producer_queue.faulted = true; };
    producer_context.retry = [&] { producer_queue.command_retry_pending = true; };
    // Snapshot creation takes an exclusive VM lock: do it before publication
    // so that lock cannot substitute for the operand's acquire/release edge.
    queue.command_access = gpu_vm.snapshot_pinned(*handle);
    producer_queue.command_access = gpu_vm.snapshot_pinned(*handle);
    ASSERT_TRUE(queue.command_access);
    ASSERT_TRUE(producer_queue.command_access);
    std::jthread producer([&] {
      payload = expected;
      if (wide)
        process_pm4_packets(producer_queue, &gpu_vm, producer_context);
      else
        std::atomic_ref<uint32_t>(narrow_operand).store(1, std::memory_order_release);
    });
    for (uint32_t poll = 0; poll < 10000 && !queue.commands.submissions.empty(); ++poll) {
      queue.command_retry_pending = false;
      process_pm4_packets(queue, &gpu_vm, context);
      std::this_thread::yield();
    }
    // No join or auxiliary acquire may hide the wait/release synchronization edge.
    EXPECT_EQ(output, expected);
    EXPECT_TRUE(queue.commands.submissions.empty());
    EXPECT_FALSE(queue.faulted);
    producer.join();
    EXPECT_FALSE(producer_queue.faulted);
    EXPECT_FALSE(producer_queue.command_retry_pending);
  }
}

TEST_F(Pm4PacketProcessorTest, DispatchPausesInterpretationUntilCpRetiresIt) {
  const std::array words{0xc0031500u, 2u,     3u,      4u, 1u,
                         0xc0033700u, 0x100u, kOutput, 0u, 0xdeadbeefu};
  submit(words);
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  ASSERT_EQ(dispatches.size(), 1u);
  EXPECT_EQ(dispatches.front(), (std::array<uint32_t, 4>{2, 3, 4, 1}));
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
  EXPECT_FALSE(completed);
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
  queue.dispatches.entries.clear();
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0xdeadbeefu);
  EXPECT_TRUE(completed);
}

TEST_F(Pm4PacketProcessorTest, QueueCancellationKeepsThePacketExceptionScope) {
  const std::array words{0xc0017900u, 0x40u, 7u};
  submit(words, true);
  queue.packet_callbacks.write_uconfig_register = [](uint64_t, uint32_t) {
    return Pm4RegisterWriteStatus::Rejected;
  };
  uint32_t cancellations = 0;
  context.fault_queue = [&] {
    if (++cancellations == 1)
      throw std::runtime_error("cancellation callback failed once");
  };
  EXPECT_NO_THROW(service());
  EXPECT_EQ(cancellations, 2u);
  EXPECT_FALSE(completed);
}

TEST_F(Pm4PacketProcessorTest, AlreadyFailedSubmissionCancellationRemainsOutsideTheTryBlock) {
  const std::array words{0xffff1000u};
  submit(words);
  queue.commands.submissions.front().failure->failed.store(true);
  uint32_t cancellations = 0;
  context.fault_queue = [&] {
    ++cancellations;
    throw std::runtime_error("cancellation callback failed");
  };
  EXPECT_THROW(service(), std::runtime_error);
  EXPECT_EQ(cancellations, 1u);
}

TEST_F(Pm4PacketProcessorTest, CancellationFromTheCatchBlockIsNotCaughtAgain) {
  const std::array words{0xc000ff00u, 0u};
  submit(words);
  uint32_t cancellations = 0;
  context.fault_queue = [&] {
    ++cancellations;
    throw std::runtime_error("cancellation callback failed");
  };
  EXPECT_THROW(service(), std::runtime_error);
  EXPECT_EQ(cancellations, 1u);
}

TEST(Pm4ProducerCursorTest, NormalizesWrappedPointersAndRejectsOverruns) {
  EXPECT_EQ(normalize_pm4_producer_cursor(2, 3, 4), 6u);
  EXPECT_EQ(normalize_pm4_producer_cursor(6, 3, 4), 6u);
  EXPECT_FALSE(normalize_pm4_producer_cursor(8, 3, 4));
  EXPECT_FALSE(normalize_pm4_producer_cursor(0, 0, 0));
  EXPECT_FALSE(normalize_pm4_producer_cursor(0, UINT64_MAX, 4));
}

} // namespace
} // namespace rocjitsu::amdgpu
