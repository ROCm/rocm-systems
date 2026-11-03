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
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

class FlatMemory final : public AddressSpaceTranslator, public PhysicalMemoryAccess {
public:
  static constexpr uint64_t kSize = 0x4000;
  FlatMemory() : bytes_(kSize) {}

  VmTranslationResult translate(uint64_t address, std::size_t size,
                                VmAccessKind access) const override {
    for (const uint64_t alias : aliases) {
      if (address >= alias && address - alias < bytes_.size()) {
        address -= alias;
        break;
      }
    }
    if (size == 0 || address > bytes_.size() || size > bytes_.size() - address)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    const VmPermissions permissions = protected_address_ && address == *protected_address_
                                          ? protected_permissions_
                                          : VmPermissions{.readable = true, .writable = true};
    if (!permissions.allows(access))
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    return {.outcome = VmAccessOutcome::Complete,
            .translation = {.domain = VmMemoryDomain::System,
                            .address = address,
                            .contiguous_bytes = bytes_.size() - address,
                            .mtype = Mtype::RW,
                            .permissions = permissions}};
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
    atomic_load_addresses_.push_back(address);
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
    atomic_store_addresses_.push_back(address);
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
  void protect(uint64_t address, VmPermissions permissions) {
    protected_address_ = address;
    protected_permissions_ = permissions;
  }
  const std::vector<uint64_t> &atomic_load_addresses() const { return atomic_load_addresses_; }
  const std::vector<uint64_t> &atomic_store_addresses() const { return atomic_store_addresses_; }
  void make_store_unavailable_once(uint64_t address) { unavailable_store_once_ = address; }
  void make_store_faulted_once(uint64_t address) {
    store_outcome_once_ = std::pair{address, VmAccessOutcome::Faulted};
  }
  std::vector<uint64_t> aliases;

private:
  // Optional operand whose translation enforces directional access permissions.
  std::optional<uint64_t> protected_address_;
  // Permissions for that operand; other fixture memory remains readable and writable.
  VmPermissions protected_permissions_;
  // Atomic observations distinguish synchronization packets from byte copies.
  std::vector<uint64_t> atomic_load_addresses_;
  // Atomic publications include payloads and the native consumer cursor.
  std::vector<uint64_t> atomic_store_addresses_;
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
    context.flush_caches = [] { return VmAccessOutcome::Complete; };
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

TEST_F(Pm4PacketProcessorTest, LoadsHighRegisterAddressesInTheQueueAddressSpace) {
  constexpr uint64_t raw = 0x0000800012340000;
  constexpr uint64_t canonical = 0xffff800012340000;
  constexpr uint64_t wide = 0x0001800056780000;
  constexpr uint32_t lo = static_cast<uint32_t>(raw), hi = raw >> 32;
  constexpr uint32_t wide_lo = static_cast<uint32_t>(wide + 12), wide_hi = wide >> 32;
  const std::array<uint32_t, 30> packets{
      0xc0035e00, lo,      hi,      1, 1, // LOAD_UCONFIG_REG
      0xc0035f00, lo,      hi,      2, 1, // LOAD_SH_REG
      0xc0036100, lo,      hi,      3, 1, // LOAD_CONTEXT_REG
      0xc0036300, lo,      hi,      4, 1, // LOAD_SH_REG_INDEX
      0xc0039f00, lo,      hi,      5, 1, // LOAD_CONTEXT_REG_INDEX
      0xc0036300, wide_lo, wide_hi, 6, 1, // Full-width indexed source.
  };
  for (bool native_ring : {false, true}) {
    SCOPED_TRACE(native_ring);
    memory->aliases = {native_ring ? raw : canonical, wide};
    const uint32_t value = native_ring ? 40 : 20;
    for (uint32_t i = 0; i < 4; ++i)
      memory->store(i * 4, value + i);
    submit(packets, native_ring);
    queue.commands.submissions.front().graphics_engine = true;
    ASSERT_EQ(service(), Pm4TestStatus::Ready);
    EXPECT_EQ(queue.commands.uconfig_registers[1], value + 1);
    EXPECT_EQ(queue.commands.sh_registers[2], value + 2);
    EXPECT_EQ(queue.commands.context_registers[3], value + 3);
    EXPECT_EQ(queue.commands.sh_registers[4], value);
    EXPECT_EQ(queue.commands.context_registers[5], value);
    EXPECT_EQ(queue.commands.sh_registers[6], value + 3);
  }
}

TEST_F(Pm4PacketProcessorTest, RegisterLoadReadsAcrossTheDrmVaSplit) {
  constexpr uint64_t split = uint64_t{1} << 47;
  memory->aliases = {split - FlatMemory::kSize, 0xffff800000000000};
  memory->store<uint32_t>(FlatMemory::kSize - 4, 55);
  memory->store<uint32_t>(0, 66);
  // The indexed load straddles bit 47. The shadow load adds a register
  // offset that puts its first source dword in the upper half.
  const std::array<uint32_t, 10> packets{0xc0036300, 0xfffffffc, 0x7fff, 2, 2,
                                         0xc0035f00, 0xfffffffc, 0x7fff, 1, 1};
  submit(packets);
  ASSERT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(queue.commands.sh_registers[1], 66u);
  EXPECT_EQ(queue.commands.sh_registers[2], 55u);
  EXPECT_EQ(queue.commands.sh_registers[3], 66u);
}

TEST_F(Pm4PacketProcessorTest, ShadowLoadRejectsOffsetPastThe48BitAperture) {
  // A register offset of one dword would carry the encoded base into bit 48.
  memory->aliases = {uint64_t{1} << 48};
  const std::array<uint32_t, 5> packet{0xc0035f00, 0xfffffffc, 0xffff, 1, 1};
  submit(packet);
  EXPECT_EQ(service(), Pm4TestStatus::Faulted);
}

class Pm4PredExecTest : public Pm4PacketProcessorTest,
                        public ::testing::WithParamInterface<rj_code_arch_t> {};

TEST_P(Pm4PredExecTest, MaskedPredExecRetainsSkippedCursorAcrossPublicationRetry) {
  context.arch = GetParam();
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

TEST_P(Pm4PredExecTest, SelectedPredExecExecutesOnHighestVirtualXcc) {
  context.arch = GetParam();
  context.xcc_id = 7;
  const std::array words{0xc0002300u, (1u << 31) | 5u, 0xc0033700u, 5u << 8, kOutput, 0u, 99u};
  submit(words);
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 99u);
  EXPECT_TRUE(completed);
}

TEST_P(Pm4PredExecTest, RejectsReservedPredExecBitsBeforeEffects) {
  context.arch = GetParam();
  for (uint32_t bit = 14; bit < 24; ++bit) {
    SCOPED_TRACE(bit);
    queue.faulted = false;
    queue.commands.submissions.clear();
    const std::array words{
        0xc0002300u, (1u << bit) | (1u << 24) | 5u, 0xc0033700u, 5u << 8, kOutput, 0u, 99u};
    submit(words);
    EXPECT_EQ(service(), Pm4TestStatus::Faulted);
    EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
    EXPECT_FALSE(completed);
  }
}

TEST_P(Pm4PredExecTest, RejectsPredExecXccOutsideMask) {
  context.arch = GetParam();
  context.xcc_id = 8;
  const std::array words{0xc0002300u, 0xff000005u, 0xc0033700u, 5u << 8, kOutput, 0u, 99u};
  submit(words);
  EXPECT_EQ(service(), Pm4TestStatus::Faulted);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
  EXPECT_FALSE(completed);
}

TEST_P(Pm4PredExecTest, RejectsPredExecBeyondBufferForSelectedAndMaskedXcc) {
  context.arch = GetParam();
  for (uint32_t mask : {1u << 24, 1u << 25}) {
    SCOPED_TRACE(mask);
    queue.faulted = false;
    queue.commands.submissions.clear();
    const std::array words{0xc0002300u, mask | 6u, 0xc0033700u, 5u << 8, kOutput, 0u, 99u};
    submit(words);
    EXPECT_EQ(service(), Pm4TestStatus::Faulted);
    EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
    EXPECT_FALSE(completed);
  }
}

INSTANTIATE_TEST_SUITE_P(Cdna, Pm4PredExecTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4,
                                           ROCJITSU_CODE_ARCH_CDNA5));

TEST_F(Pm4PacketProcessorTest, RejectsPredExecOnOtherArchitectures) {
  for (const auto arch :
       {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_RDNA1,
        ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
        ROCJITSU_CODE_ARCH_RDNA4}) {
    SCOPED_TRACE(arch);
    context.arch = arch;
    queue.faulted = false;
    queue.commands.submissions.clear();
    const std::array words{0xc0002300u, (1u << 24) | 5u, 0xc0033700u, 5u << 8, kOutput, 0u, 99u};
    submit(words);
    EXPECT_EQ(service(), Pm4TestStatus::Faulted);
    EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
    EXPECT_FALSE(completed);
  }
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
  EXPECT_EQ(std::ranges::count(memory->atomic_load_addresses(), kOutput), 1);
  EXPECT_EQ(memory->load<uint32_t>(kOutput + 4), 0u);
  EXPECT_EQ(queue.commands.submissions.front().buffers.front().address, kRing);
  memory->store<uint32_t>(kOutput, 1);
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(std::ranges::count(memory->atomic_load_addresses(), kOutput), 2);
  EXPECT_EQ(memory->load<uint32_t>(kOutput + 4), 0xdeadbeefu);
  EXPECT_TRUE(completed);
}

TEST_F(Pm4PacketProcessorTest, ReleaseInterruptObservesPayloadBeforeABlockedTail) {
  for (uint32_t selection : {1u, 2u}) {
    SCOPED_TRACE(selection);
    const uint32_t gate = kOutput + 16;
    memory->store<uint64_t>(kOutput, 0);
    memory->store<uint32_t>(gate, 0);
    const std::array words{0xc0064900u, 0x528u, (selection << 29) | (2u << 24),
                           kOutput,     0u,     0x12345678u,
                           0x87654321u, 47u,    0xc0053c00u,
                           0x13u,       gate,   0u,
                           1u,          ~0u,    0u};
    queue.process_id = 19;
    uint32_t interrupts = 0;
    InterruptSubscription subscription([&](uint32_t process, uint32_t event) {
      EXPECT_EQ(process, 19u);
      EXPECT_EQ(event, 47u);
      EXPECT_EQ(memory->load<uint64_t>(kOutput),
                selection == 1 ? UINT64_C(0x12345678) : UINT64_C(0x8765432112345678));
      EXPECT_FALSE(completed);
      ++interrupts;
    });
    queue.interrupt_sink = subscription.sink();
    completed = false;
    const auto previous_stores = std::ranges::count(memory->atomic_store_addresses(), kOutput);
    submit(words);
    EXPECT_EQ(service(), Pm4TestStatus::Blocked);
    EXPECT_EQ(std::ranges::count(memory->atomic_store_addresses(), kOutput), previous_stores + 1);
    EXPECT_EQ(interrupts, 1u);
    EXPECT_FALSE(completed);
    memory->store<uint32_t>(gate, 1);
    EXPECT_EQ(service(), Pm4TestStatus::Ready);
    EXPECT_EQ(interrupts, 1u);
    EXPECT_TRUE(completed);
  }
}

TEST_F(Pm4PacketProcessorTest, WaitReadsReadOnlyMemory) {
  memory->store<uint32_t>(kOutput, 1);
  memory->protect(kOutput, {.readable = true});
  const std::array words{0xc0053c00u, 0x13u, kOutput, 0u, 1u, ~0u, 0u};
  submit(words);
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_TRUE(completed);
}

TEST_F(Pm4PacketProcessorTest, ReleaseWritesWriteOnlyMemory) {
  memory->protect(kOutput, {.writable = true});
  const std::array words{0xc0064900u, 0x528u, 1u << 29, kOutput, 0u, 31u, 0u, 0u};
  submit(words);
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 31u);
  EXPECT_TRUE(completed);
}

TEST_F(Pm4PacketProcessorTest, UnavailableWaitLoadRetainsThePacketUntilMemoryIsAvailable) {
  const std::array words{0xc0053c00u, 0x13u, kOutput, 0u, 1u, ~0u, 0u};
  memory->store<uint32_t>(kOutput, 1);
  memory->make_read_unavailable(kOutput);
  submit(words, true);
  EXPECT_EQ(service(), Pm4TestStatus::Blocked);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 0u);
  EXPECT_FALSE(completed);
  memory->make_reads_available();
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(std::ranges::count(memory->atomic_load_addresses(), kOutput), 2);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), words.size());
  EXPECT_TRUE(completed);
}

TEST_F(Pm4PacketProcessorTest, NonReturningExchangeUsesAtomicStoreAtBothWidths) {
  for (uint32_t control : {7u, 0x27u}) {
    SCOPED_TRACE(control);
    memory->store<uint64_t>(kOutput, 0);
    const std::array words{0xc0071e00u, control, kOutput, 0u, 0x12345678u, 0x87654321u, 0u, 0u, 0u};
    submit(words);
    EXPECT_EQ(service(), Pm4TestStatus::Ready);
    EXPECT_EQ(memory->load<uint64_t>(kOutput),
              control == 7 ? UINT64_C(0x12345678) : UINT64_C(0x8765432112345678));
  }
  EXPECT_EQ(std::ranges::count(memory->atomic_store_addresses(), kOutput), 2);
}

TEST_F(Pm4PacketProcessorTest, ReleaseWithoutInterruptStillPublishesData) {
  uint32_t interrupts = 0;
  InterruptSubscription subscription([&](uint32_t, uint32_t) { ++interrupts; });
  queue.interrupt_sink = subscription.sink();
  for (uint32_t control : {0u, 3u}) {
    SCOPED_TRACE(control);
    memory->store<uint32_t>(kOutput, 0);
    const std::array words{0xc0064900u, 0x528u, (1u << 29) | (control << 24), kOutput, 0u, 31u,
                           0u,          47u};
    submit(words);
    EXPECT_EQ(service(), Pm4TestStatus::Ready);
    EXPECT_EQ(memory->load<uint32_t>(kOutput), 31u);
    EXPECT_EQ(interrupts, 0u);
  }
}

TEST_F(Pm4PacketProcessorTest, ReleaseInterruptWithoutDataDoesNotAccessTheAddress) {
  uint32_t interrupts = 0;
  queue.process_id = 19;
  InterruptSubscription subscription([&](uint32_t process, uint32_t event) {
    EXPECT_EQ(process, 19u);
    EXPECT_EQ(event, 47u);
    ++interrupts;
  });
  queue.interrupt_sink = subscription.sink();
  for (uint32_t control : {1u, 4u}) {
    SCOPED_TRACE(control);
    const std::array words{0xc0064900u, 0x528u, control << 24, ~0u, ~0u, 0u, 0u, 47u};
    submit(words);
    EXPECT_EQ(service(), Pm4TestStatus::Ready);
  }
  EXPECT_EQ(interrupts, 2u);
}

TEST_F(Pm4PacketProcessorTest, FailedReleasePublicationDoesNotNotify) {
  uint32_t interrupts = 0;
  InterruptSubscription subscription([&](uint32_t, uint32_t) { ++interrupts; });
  queue.interrupt_sink = subscription.sink();
  const std::array words{0xc0064900u, 0x528u, (1u << 29) | (2u << 24), 0x4000u, 0u, 31u, 0u, 47u};
  submit(words);
  EXPECT_EQ(service(), Pm4TestStatus::Faulted);
  EXPECT_EQ(interrupts, 0u);
}

TEST_F(Pm4PacketProcessorTest, ReleaseInterruptIsNotReplayedAfterCursorPublicationRetry) {
  uint32_t interrupts = 0;
  InterruptSubscription subscription([&](uint32_t, uint32_t) { ++interrupts; });
  queue.interrupt_sink = subscription.sink();
  const std::array words{0xc0064900u, 0x528u, (1u << 29) | (2u << 24), kOutput, 0u, 31u, 0u, 47u};
  submit(words, true);
  memory->make_store_unavailable_once(kReadPointer);
  EXPECT_EQ(service(), Pm4TestStatus::Blocked);
  EXPECT_EQ(interrupts, 1u);
  EXPECT_TRUE(queue.read_pointer_journal.publication_pending());
  EXPECT_EQ(service(), Pm4TestStatus::Ready);
  EXPECT_EQ(interrupts, 1u);
  EXPECT_TRUE(completed);
}

TEST_F(Pm4PacketProcessorTest, RejectsConditionalReleaseInterruptsBeforeEffects) {
  uint32_t interrupts = 0;
  InterruptSubscription subscription([&](uint32_t, uint32_t) { ++interrupts; });
  queue.interrupt_sink = subscription.sink();
  for (uint32_t control : {5u, 6u, 7u}) {
    SCOPED_TRACE(control);
    queue.faulted = false;
    queue.commands.submissions.clear();
    const std::array words{0xc0064900u, 0x528u, (1u << 29) | (control << 24), kOutput, 0u, 31u,
                           0u,          47u};
    submit(words);
    EXPECT_EQ(service(), Pm4TestStatus::Faulted);
    EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
    EXPECT_EQ(interrupts, 0u);
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

TEST_F(Pm4PacketProcessorTest, FailedDependencyDoesNotExecuteOrRetirePacket) {
  const std::array words{0xc0033700u, 0x100u, kOutput, 0u, 0xdeadbeefu};
  submit(words, true);
  queue.commands.submissions.front().ready =
      [](util::DiagnosticEmitter emit_error) -> util::FailureOr<bool> {
    return emit_error.emit() << "PM4 submission dependency failed";
  };
  testing::internal::CaptureStderr();
  EXPECT_NO_THROW(EXPECT_EQ(service(), Pm4TestStatus::Faulted));
  EXPECT_EQ(testing::internal::GetCapturedStderr(),
            "[rj warn] PM4 queue failed: PM4 submission dependency failed\n");
  EXPECT_FALSE(queue.command_retry_pending);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 0u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);
  EXPECT_FALSE(completed);
}

TEST_F(Pm4PacketProcessorTest, FlushFailurePreservesEarlierPacketEffectsAndRetirement) {
  const std::array words{0xc0033700u, 0x100u, kOutput,     0u, 0x12345678u,
                         0xc0033700u, 0x100u, kOutput + 4, 0u, 0xdeadbeefu};
  submit(words, true);
  uint32_t flushes = 0;
  context.flush_caches = [&] {
    return ++flushes == 1 ? VmAccessOutcome::Complete : VmAccessOutcome::Faulted;
  };
  EXPECT_EQ(service(), Pm4TestStatus::Faulted);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0x12345678u);
  EXPECT_EQ(memory->load<uint32_t>(kOutput + 4), 0u);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 5u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 5u);
  EXPECT_FALSE(completed);
}

class Pm4CacheFlushFailureTest : public Pm4PacketProcessorTest,
                                 public ::testing::WithParamInterface<VmAccessOutcome> {};

TEST_P(Pm4CacheFlushFailureTest, PreventsPacketEffectAndRetirement) {
  const std::array words{0xc0033700u, 0x100u, kOutput, 0u, 0xdeadbeefu};
  submit(words, true);
  uint32_t flushes = 0;
  context.flush_caches = [&] {
    ++flushes;
    return GetParam();
  };
  EXPECT_NO_THROW(EXPECT_EQ(service(), Pm4TestStatus::Faulted));
  EXPECT_EQ(flushes, 1u);
  EXPECT_FALSE(queue.command_retry_pending);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 0u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);
  EXPECT_FALSE(completed);
}

TEST_P(Pm4CacheFlushFailureTest, PreventsSuccessfulStreamCompletion) {
  const std::array words{0xffff1000u};
  submit(words);
  context.flush_caches = [] { return GetParam(); };
  EXPECT_NO_THROW(EXPECT_EQ(service(), Pm4TestStatus::Faulted));
  EXPECT_FALSE(queue.command_retry_pending);
  EXPECT_FALSE(completed);
  EXPECT_EQ(queue.commands.submissions.size(), 1u);
}

INSTANTIATE_TEST_SUITE_P(NonComplete, Pm4CacheFlushFailureTest,
                         ::testing::Values(VmAccessOutcome::Unavailable, VmAccessOutcome::Faulted,
                                           VmAccessOutcome::Malformed));

TEST_F(Pm4PacketProcessorTest, PropagatesDispatchCallbackExceptionAfterCommittedWrite) {
  const std::array words{0xc0033700u, 0x100u, kOutput, 0u, 0xdeadbeefu,
                         0xc0031500u, 1u,     1u,      1u, 1u};
  submit(words, true);
  uint32_t faults = 0, completions = 0;
  context.fault_queue = [&] { ++faults; };
  queue.commands.submissions.front().complete = [&](bool) { ++completions; };
  context.dispatch = [](const std::array<uint32_t, 4> &, util::DiagnosticEmitter) -> util::Result {
    throw std::runtime_error("dispatch callback");
  };
  EXPECT_THROW(service(), std::runtime_error);
  EXPECT_EQ(faults, 0u);
  EXPECT_EQ(completions, 0u);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0xdeadbeefu);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 5u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 5u);
}

struct RejectedPm4Packet {
  const char *name;
  std::vector<uint32_t> words;
  const char *reason;
};

class Pm4AdmissionTest : public Pm4PacketProcessorTest,
                         public ::testing::WithParamInterface<RejectedPm4Packet> {};

TEST_P(Pm4AdmissionTest, FaultsWithoutRetirementOrSuccessCompletion) {
  const auto &packet = GetParam();
  constexpr uint64_t kInitialCursor = 3;
  memory->store<uint32_t>(kReadPointer, kInitialCursor);
  memory->store<uint32_t>(kOutput, 0x12345678);
  submit(packet.words, true, kInitialCursor);
  uint32_t callbacks = 0;
  queue.packet_callbacks.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++callbacks;
    return Pm4RegisterWriteStatus::Complete;
  };
  testing::internal::CaptureStderr();
  EXPECT_NO_THROW(EXPECT_EQ(service(), Pm4TestStatus::Faulted));
  // A faulted queue must neither replay its packet nor repeat the diagnostic.
  EXPECT_NO_THROW(service());
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  const std::string::size_type reason_offset = diagnostic.find(packet.reason);
  EXPECT_NE(reason_offset, std::string::npos) << diagnostic;
  if (reason_offset != std::string::npos) {
    EXPECT_EQ(diagnostic.find(packet.reason, reason_offset + 1), std::string::npos) << diagnostic;
  }
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
        RejectedPm4Packet{"RequireSize", {0xc0001500u, 0u}, "expects 4 payload words"},
        RejectedPm4Packet{"IndirectSwap",
                          {0xc0023f00u, 0x401u, 0u, 1u},
                          "unsupported PM4 indirect buffer swap function"},
        RejectedPm4Packet{"Opcode", {0xc000ff00u, 0u}, "unsupported PM4 opcode"},
        RejectedPm4Packet{
            "WriteDestination", {0xc0033700u, 0u, 0x800u, 0u, 99u}, "unsupported WRITE_DATA"}),
    [](const auto &info) { return info.param.name; });

class Pm4EffectFailureTest : public Pm4PacketProcessorTest,
                             public ::testing::WithParamInterface<RejectedPm4Packet> {};

TEST_P(Pm4EffectFailureTest, StopsBeforeDependentEffectsAndRetirement) {
  const auto &packet = GetParam();
  std::vector<uint32_t> words = packet.words;
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
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  const std::string reason = std::string("PM4 queue failed: ") + packet.reason;
  const std::string::size_type reason_offset = diagnostic.find(reason);
  EXPECT_NE(reason_offset, std::string::npos) << diagnostic;
  if (reason_offset != std::string::npos) {
    EXPECT_EQ(diagnostic.find(reason, reason_offset + 1), std::string::npos) << diagnostic;
  }
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
  Pm4IndirectBuffer &ib = queue.commands.submissions.front().buffers.front();
  ib.ring_base = 0x3ffc;
  memory->store<uint32_t>(ib.ring_base, words[0]);
  EXPECT_EQ(service(), Pm4TestStatus::Faulted);
  EXPECT_FALSE(queue.command_retry_pending);
  EXPECT_TRUE(dispatches.empty());
  EXPECT_EQ(ib.address, 0u);
  EXPECT_EQ(queue.read_pointer_journal.cursor(), 0u);
  EXPECT_FALSE(completed);
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
