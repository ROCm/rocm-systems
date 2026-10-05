// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pm4/pm4_packet_processor.h"

#include "rocjitsu/vm/amdgpu/dispatch_entry.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

class FlatMemory final : public AddressSpaceTranslator, public PhysicalMemoryAccess {
public:
  FlatMemory() : bytes_(0x4000) {}

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
  void make_store_unavailable_once(uint64_t address) { unavailable_store_once_ = address; }
  void make_store_faulted_once(uint64_t address) {
    store_outcome_once_ = std::pair{address, VmAccessOutcome::Faulted};
  }

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
