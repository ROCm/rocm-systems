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
    context.wake = [] {};
    context.flush_caches = [] {};
    context.dispatch = [this](const std::array<uint32_t, 4> &dimensions, util::DiagnosticEmitter) {
      dispatches.push_back(dimensions);
      queue.dispatches.push_entry({});
      return util::Result::success();
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

TEST_F(Pm4PacketProcessorTest, ReturnedDispatchFailureDoesNotRetireNativeRingPacket) {
  const std::array words{0xc0031500u, 2u, 3u, 4u, 1u};
  constexpr uint64_t kInitialCursor = 3;
  memory->store<uint32_t>(kReadPointer, kInitialCursor);
  submit(words, true, kInitialCursor);
  uint32_t dispatch_attempts = 0;
  context.dispatch = [&](const std::array<uint32_t, 4> &dimensions,
                         util::DiagnosticEmitter emit_error) -> util::Result {
    EXPECT_EQ(dimensions, (std::array<uint32_t, 4>{2, 3, 4, 1}));
    ++dispatch_attempts;
    return emit_error.emit() << "invalid test launch";
  };

  EXPECT_EQ(service(), Pm4TestStatus::Faulted);
  EXPECT_EQ(dispatch_attempts, 1u);
  EXPECT_TRUE(queue.faulted);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), kInitialCursor);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), kInitialCursor);
  EXPECT_FALSE(completed);
}

TEST_F(Pm4PacketProcessorTest, ReturnedDispatchFailureCancellationIsNotCaughtAgain) {
  const std::array words{0xc0031500u, 2u, 3u, 4u, 1u};
  submit(words);
  context.dispatch = [](const std::array<uint32_t, 4> &,
                        util::DiagnosticEmitter emit_error) -> util::Result {
    return emit_error.emit() << "invalid test launch";
  };
  uint32_t cancellations = 0;
  context.fault_queue = [&] {
    ++cancellations;
    throw std::runtime_error("cancellation callback failed");
  };
  EXPECT_THROW(service(), std::runtime_error);
  EXPECT_EQ(cancellations, 1u);
  EXPECT_FALSE(completed);
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

// Each case reaches a distinct admission check before its packet effect.
struct RejectedPm4Packet {
  const char *name;
  std::vector<uint32_t> words;
  const char *reason;
  bool graphics = false;
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4;
};

class Pm4AdmissionTest : public Pm4PacketProcessorTest,
                         public ::testing::WithParamInterface<RejectedPm4Packet> {};

TEST_P(Pm4AdmissionTest, FaultsWithoutRetirementOrSuccessCompletion) {
  const auto &packet = GetParam();
  constexpr uint64_t kInitialCursor = 3;
  memory->store<uint32_t>(kReadPointer, kInitialCursor);
  memory->store<uint32_t>(kOutput, 0x12345678);
  submit(packet.words, true, kInitialCursor);
  queue.commands.submissions.front().graphics_engine = packet.graphics;
  context.arch = packet.arch;
  uint32_t callbacks = 0;
  queue.packet_callbacks.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++callbacks;
    return Pm4RegisterWriteStatus::Complete;
  };
  testing::internal::CaptureStderr();
  EXPECT_NO_THROW(EXPECT_EQ(service(), Pm4TestStatus::Faulted));
  // A faulted queue must neither replay its packet nor repeat the diagnostic.
  EXPECT_NO_THROW(service());
  const auto diagnostic = testing::internal::GetCapturedStderr();
  const auto reason_offset = diagnostic.find(packet.reason);
  EXPECT_NE(reason_offset, std::string::npos) << diagnostic;
  if (reason_offset != std::string::npos)
    EXPECT_EQ(diagnostic.find(packet.reason, reason_offset + 1), std::string::npos) << diagnostic;
  EXPECT_TRUE(queue.faulted);
  EXPECT_FALSE(completed);
  EXPECT_TRUE(dispatches.empty());
  EXPECT_EQ(callbacks, 0u);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), kInitialCursor);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), kInitialCursor);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0x12345678u);
  EXPECT_TRUE(std::ranges::all_of(queue.commands.sh_registers, [](auto value) { return !value; }));
  EXPECT_TRUE(
      std::ranges::all_of(queue.commands.context_registers, [](auto value) { return !value; }));
  EXPECT_TRUE(
      std::ranges::all_of(queue.commands.uconfig_registers, [](auto value) { return !value; }));
}

INSTANTIATE_TEST_SUITE_P(
    RejectedPackets, Pm4AdmissionTest,
    ::testing::Values(
        RejectedPm4Packet{"Type", {0u, 0u}, "invalid PM4 packet size or type"},
        RejectedPm4Packet{"Count", {0xc0017900u, 0u}, "invalid PM4 packet size or type"},
        RejectedPm4Packet{"RequireSize", {0xc0001500u, 0u}, "expects 4 payload words"},
        RejectedPm4Packet{"Opcode", {0xc000ff00u, 0u}, "unsupported PM4 opcode"},
        RejectedPm4Packet{"GraphicsEngine", {0xc0012800u, 0u, 0u}, "graphics state packet"},
        RejectedPm4Packet{"ClearStateMode", {0xc0001200u, 4u}, "unsupported CLEAR_STATE", true},
        RejectedPm4Packet{"ClearStatePop", {0xc0001200u, 2u}, "pop without a saved context", true},
        RejectedPm4Packet{"PredExecControl", {0xc0002300u, 0u}, "unsupported PRED_EXEC"},
        RejectedPm4Packet{"PredExecExtent",
                          {0xc0002300u, 1u},
                          "PRED_EXEC exceeds",
                          false,
                          ROCJITSU_CODE_ARCH_CDNA3},
        RejectedPm4Packet{"CondExecExtent", {0xc0032200u, 0x800u, 0u, 0u, 1u}, "invalid COND_EXEC"},
        RejectedPm4Packet{"PredicateControl",
                          {0xc0022000u, 1u << 16, 0u, 0u},
                          "unsupported SET_PREDICATION",
                          true},
        RejectedPm4Packet{"PredicateAlignment",
                          {0xc0022000u, 3u << 16, 1u, 0u},
                          "unaligned SET_PREDICATION",
                          true},
        RejectedPm4Packet{"BaseIndex", {0xc0021100u, 2u, 0u, 0u}, "unsupported SET_BASE", true},
        RejectedPm4Packet{"WritePayload", {0xc0023700u, 0u, 0x800u, 0u}, "invalid WRITE_DATA"},
        RejectedPm4Packet{
            "WriteDestination", {0xc0033700u, 0u, 0x800u, 0u, 99u}, "unsupported WRITE_DATA"},
        RejectedPm4Packet{"AtomicControl",
                          {0xc0071e00u, 0x80u, 0x800u, 0u, 0u, 0u, 0u, 0u, 0u},
                          "unsupported ATOMIC_MEM control"},
        RejectedPm4Packet{"AtomicOperation",
                          {0xc0071e00u, 0u, 0x800u, 0u, 0u, 0u, 0u, 0u, 0u},
                          "unsupported ATOMIC_MEM operation"},
        RejectedPm4Packet{"ReleaseSource",
                          {0xc0064900u, 0u, 4u << 29, 0x800u, 0u, 99u, 0u, 0u},
                          "unsupported RELEASE_MEM"},
        RejectedPm4Packet{"CopyDestination",
                          {0xc0044000u, 5u, 99u, 0u, 0x800u, 0u},
                          "unsupported COPY_DATA destination"},
        RejectedPm4Packet{"CopySource",
                          {0xc0044000u, 0x103u, 0u, 0u, 0x800u, 0u},
                          "unsupported COPY_DATA source"},
        RejectedPm4Packet{"CopyRegister",
                          {0xc0044000u, 0x100u, 0x3000u, 0u, 0x800u, 0u},
                          "register outside modeled apertures",
                          true},
        RejectedPm4Packet{
            "WaitOperation", {0xc0053c00u, 0u, 0x800u, 0u, 0u, 0u, 0u}, "unsupported WAIT_REG_MEM"},
        RejectedPm4Packet{"WaitComparison",
                          {0xc0053c00u, 0x17u, 0x800u, 0u, 0u, 0u, 0u},
                          "invalid WAIT_REG_MEM comparison"},
        RejectedPm4Packet{
            "ShadowPayload", {0xc0035f00u, 1u, 0u, 0u, 1u}, "invalid shadow register load payload"},
        RejectedPm4Packet{"ShadowRange",
                          {0xc0035f00u, 0x800u, 0u, 0x400u, 1u},
                          "invalid shadow register load range"},
        RejectedPm4Packet{"IndexedLoadRange",
                          {0xc0036300u, 0x800u, 0u, 0x3ffu, 2u},
                          "unsupported register load range"},
        RejectedPm4Packet{"ShaderRange", {0xc0017600u, 0x400u, 99u}, "invalid shader register"},
        RejectedPm4Packet{
            "ContextRmw", {0xc0025100u, 0x2000u, ~0u, 99u}, "invalid CONTEXT_REG_RMW", true},
        RejectedPm4Packet{"PrimeControl", {0xc0035d00u, 0u, 1u, 0u, 1u}, "invalid PRIME_UTCL2"},
        RejectedPm4Packet{
            "ShaderPairPayload", {0xc000ba00u, 0u}, "invalid SET_SH_REG_PAIRS payload"},
        RejectedPm4Packet{
            "ShaderPairRegister", {0xc001ba00u, 0x400u, 99u}, "invalid SET_SH_REG_PAIRS register"},
        RejectedPm4Packet{"ContextPairRegister",
                          {0xc001b800u, 0x2000u, 99u},
                          "graphics register outside aperture",
                          true},
        RejectedPm4Packet{"PackedPayload",
                          {0xc003b900u, 3u, 0u, 99u, 99u},
                          "invalid packed graphics register payload",
                          true},
        RejectedPm4Packet{"PackedRegister",
                          {0xc003b900u, 2u, 0x2000u, 99u, 99u},
                          "graphics register outside aperture",
                          true},
        RejectedPm4Packet{"GraphicsIndex",
                          {0xc0016900u, 1u << 28, 99u},
                          "unsupported graphics register index",
                          true},
        RejectedPm4Packet{"NativePairs", {0xc001be00u, 0u, 99u}, "unsupported on a native ring"},
        RejectedPm4Packet{"Event", {0xc0004600u, 63u}, "unsupported PM4 EVENT_WRITE"},
        RejectedPm4Packet{"Streamout",
                          {0xc004c300u, 0x800u, 0u, 4u, 0x800u, 0u},
                          "unsupported PM4 streamout statistics query",
                          true},
        RejectedPm4Packet{"DmaDestination",
                          {0xc0055000u, 1u << 20, 0u, 0u, 0x800u, 0u, 4u},
                          "unsupported PM4 DMA destination"},
        RejectedPm4Packet{"DmaSource",
                          {0xc0055000u, 1u << 29, 0u, 0u, 0x800u, 0u, 4u},
                          "unsupported PM4 DMA source"},
        RejectedPm4Packet{"IndirectAlignment",
                          {0xc0023f00u, 1u, 0u, 1u},
                          "invalid PM4 indirect buffer address or extent"},
        RejectedPm4Packet{"IndirectOverflow",
                          {0xc0023f00u, 0xfffffffcu, ~0u, 1u},
                          "invalid PM4 indirect buffer address or extent"},
        RejectedPm4Packet{
            "InterleavedEngine", {0xc003a700u, 1u, 1u, 1u, 1u}, "interleaved dispatch requires"}),
    [](const auto &info) { return info.param.name; });

class Pm4EffectFailureTest : public Pm4PacketProcessorTest,
                             public ::testing::WithParamInterface<RejectedPm4Packet> {};

TEST_P(Pm4EffectFailureTest, StopsBeforeDependentEffectsAndRetirement) {
  const auto &packet = GetParam();
  auto words = packet.words;
  words.insert(words.end(), {0xc0033700u, 5u << 8, kOutput + 4, 0u, 99u});
  submit(words, true);
  memory->store<uint32_t>(kOutput, 0x12345678);
  uint32_t cancellations = 0;
  context.fault_queue = [&] {
    ++cancellations;
    queue.faulted = true;
    status = Pm4TestStatus::Faulted;
  };
  testing::internal::CaptureStderr();
  EXPECT_EQ(service(), Pm4TestStatus::Faulted);
  service();
  const auto diagnostic = testing::internal::GetCapturedStderr();
  const std::string reason = std::string("PM4 queue failed: ") + packet.reason;
  const auto reason_offset = diagnostic.find(reason);
  EXPECT_NE(reason_offset, std::string::npos) << diagnostic;
  if (reason_offset != std::string::npos)
    EXPECT_EQ(diagnostic.find(reason, reason_offset + 1), std::string::npos) << diagnostic;
  EXPECT_EQ(cancellations, 1u);
  EXPECT_FALSE(queue.command_retry_pending);
  EXPECT_FALSE(completed);
  EXPECT_TRUE(dispatches.empty());
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 0u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0x12345678u);
  EXPECT_EQ(memory->load<uint32_t>(kOutput + 4), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    FailedReads, Pm4EffectFailureTest,
    ::testing::Values(RejectedPm4Packet{"CopySource",
                                        {0xc0044000u, 0x101u, 0x4000u, 0u, 0x800u, 0u},
                                        "PM4 COPY_DATA read failed"},
                      RejectedPm4Packet{"DmaSource",
                                        {0xc0055000u, 0u, 0x4000u, 0u, 0x800u, 0u, 4u},
                                        "PM4 DMA copy failed"},
                      RejectedPm4Packet{"WaitSource",
                                        {0xc0053c00u, 0x13u, 0x4000u, 0u, 1u, ~0u, 0u},
                                        "PM4 WAIT_REG_MEM read failed"},
                      RejectedPm4Packet{"IndirectArguments",
                                        {0xc0021600u, 0x4000u, 0u, 1u},
                                        "PM4 DISPATCH_INDIRECT read failed"}),
    [](const auto &info) { return info.param.name; });

TEST_F(Pm4PacketProcessorTest, ShadowLoadFailureRetainsOnlyEarlierRanges) {
  const std::array words{0xc0055f00u, 0x3ffcu,     0u,      0u,      1u, 1u,
                         1u,          0xc0033700u, 5u << 8, kOutput, 0u, 99u};
  memory->store<uint32_t>(0x3ffc, 0x12345678);
  queue.commands.sh_registers[1] = 0xabcdef;
  submit(words, true);
  EXPECT_EQ(service(), Pm4TestStatus::Faulted);
  EXPECT_EQ(queue.commands.sh_registers[0], 0x12345678u);
  EXPECT_EQ(queue.commands.sh_registers[1], 0xabcdefu);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 0u);
  EXPECT_FALSE(completed);
}

TEST_F(Pm4PacketProcessorTest, FaultedHeaderFetchDoesNotExecuteOrRetire) {
  const std::array words{0xffff1000u};
  submit(words, true);
  queue.commands.submissions.front().buffers.front().ring_base = 0x4000;
  EXPECT_EQ(service(), Pm4TestStatus::Faulted);
  EXPECT_FALSE(queue.command_retry_pending);
  EXPECT_EQ(queue.commands.submissions.front().buffers.front().address, 0u);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 0u);
  EXPECT_FALSE(completed);
}

TEST_F(Pm4PacketProcessorTest, FaultedPayloadFetchDoesNotDispatchOrAdvance) {
  const std::array words{0xc0031500u, 2u, 3u, 4u, 1u};
  submit(words, true);
  auto &ib = queue.commands.submissions.front().buffers.front();
  ib.ring_base = 0x3ffc;
  memory->store<uint32_t>(ib.ring_base, words[0]);
  EXPECT_EQ(service(), Pm4TestStatus::Faulted);
  EXPECT_FALSE(queue.command_retry_pending);
  EXPECT_TRUE(dispatches.empty());
  EXPECT_EQ(ib.address, 0u);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 0u);
  EXPECT_FALSE(completed);
}

TEST_F(Pm4PacketProcessorTest, AdmissionCancellationRunsOnceOutsideExceptionContainment) {
  const std::array words{0xc0001500u, 0u}; // DISPATCH_DIRECT requires four payload words.
  submit(words, true);
  uint32_t cancellations = 0;
  context.fault_queue = [&] {
    ++cancellations;
    throw std::runtime_error("cancellation callback failed");
  };
  EXPECT_THROW(service(), std::runtime_error);
  EXPECT_EQ(cancellations, 1u);
  EXPECT_FALSE(completed);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 0u);
}

TEST_F(Pm4PacketProcessorTest, RegisterPairFailurePreservesValidPrefixWithoutRetiring) {
  const std::array words{0xc003b800u, 0u, 99u, 0x2000u, 100u};
  submit(words, true);
  queue.commands.submissions.front().graphics_engine = true;
  EXPECT_NO_THROW(EXPECT_EQ(service(), Pm4TestStatus::Faulted));
  EXPECT_EQ(queue.commands.context_registers[0], 99u);
  EXPECT_EQ(queue.commands.context_registers[1], 0u);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 0u);
  EXPECT_FALSE(completed);
}

TEST_F(Pm4PacketProcessorTest, RejectsCommandExtentOverflowBeforeFetching) {
  const std::array words{0xffff1000u};
  submit(words, true);
  queue.commands.submissions.front().buffers.front().address = UINT64_MAX - 3;
  EXPECT_NO_THROW(EXPECT_EQ(service(), Pm4TestStatus::Faulted));
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 0u);
  EXPECT_FALSE(completed);
}

TEST_F(Pm4PacketProcessorTest, RejectsIndirectDispatchAddressOverflowBeforeDispatch) {
  const std::array words{0xc0011600u, 4u, 1u};
  submit(words, true);
  queue.commands.submissions.front().graphics_engine = true;
  queue.commands.indirect_base = UINT64_MAX - 3;
  EXPECT_NO_THROW(EXPECT_EQ(service(), Pm4TestStatus::Faulted));
  EXPECT_TRUE(dispatches.empty());
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 0u);
  EXPECT_FALSE(completed);
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
