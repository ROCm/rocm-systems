// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/sdma_executor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

class RetryMemory final : public AddressSpaceTranslator, public PhysicalMemoryAccess {
public:
  RetryMemory() : bytes_(0x10000) {}

  VmTranslationResult translate(uint64_t address, std::size_t size, VmAccessKind) const override {
    if (size == 0 || address > bytes_.size() || size > bytes_.size() - address)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    constexpr uint64_t kSegment = 16;
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation = {.domain = VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = kSegment - (address % kSegment),
                        .mtype = Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }

  VmAccessOutcome read(VmMemoryDomain, uint64_t address, std::span<std::byte> bytes) override {
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(address), bytes.size(), bytes.begin());
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(VmMemoryDomain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    ++write_calls_[address];
    if (address == unavailable_write_address_ && !unavailable_write_returned_) {
      unavailable_write_returned_ = true;
      return VmAccessOutcome::Unavailable;
    }
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::copy(bytes.begin(), bytes.end(), bytes_.begin() + static_cast<std::ptrdiff_t>(address));
    return VmAccessOutcome::Complete;
  }

  AtomicLoadResult atomic_load(VmMemoryDomain, uint64_t address, uint32_t width) override {
    if (width != 4 && width != 8)
      return {.outcome = VmAccessOutcome::Malformed};
    uint64_t value = 0;
    std::memcpy(&value, bytes_.data() + address, width);
    return {.outcome = VmAccessOutcome::Complete, .value = value};
  }

  VmAccessOutcome atomic_store(VmMemoryDomain, uint64_t address, uint32_t width,
                               uint64_t value) override {
    if (address == unavailable_atomic_store_address_ && !unavailable_atomic_store_returned_) {
      unavailable_atomic_store_returned_ = true;
      return VmAccessOutcome::Unavailable;
    }
    if ((width != 4 && width != 8) || address > bytes_.size() || width > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::memcpy(bytes_.data() + address, &value, width);
    return VmAccessOutcome::Complete;
  }

  AtomicCompareExchangeResult compare_exchange(VmMemoryDomain, uint64_t address, uint32_t width,
                                               uint64_t expected, uint64_t desired) override {
    if (width != 8)
      return {.outcome = VmAccessOutcome::Malformed};
    const uint64_t observed = load<uint64_t>(address);
    if (observed != expected)
      return {.outcome = VmAccessOutcome::Complete, .observed = observed, .exchanged = false};
    store(address, desired);
    ++successful_compare_exchanges_;
    return {.outcome = VmAccessOutcome::Complete, .observed = observed, .exchanged = true};
  }

  template <typename T> void store(uint64_t address, T value) {
    std::memcpy(bytes_.data() + address, &value, sizeof(value));
  }

  template <typename T> T load(uint64_t address) const {
    T value{};
    std::memcpy(&value, bytes_.data() + address, sizeof(value));
    return value;
  }

  void set_unavailable_write(uint64_t address) { unavailable_write_address_ = address; }
  void set_unavailable_atomic_store(uint64_t address) {
    unavailable_atomic_store_address_ = address;
  }
  uint32_t write_calls(uint64_t address) const {
    const auto found = write_calls_.find(address);
    return found == write_calls_.end() ? 0 : found->second;
  }
  uint32_t successful_compare_exchanges() const { return successful_compare_exchanges_; }

private:
  std::vector<std::byte> bytes_;
  std::map<uint64_t, uint32_t> write_calls_;
  uint64_t unavailable_write_address_ = UINT64_MAX;
  uint64_t unavailable_atomic_store_address_ = UINT64_MAX;
  bool unavailable_write_returned_ = false;
  bool unavailable_atomic_store_returned_ = false;
  uint32_t successful_compare_exchanges_ = 0;
};

struct ExecutorFixture {
  ExecutorFixture() : memory(std::make_shared<RetryMemory>()), vm(nullptr) {
    handle = vm.register_translated(7, memory, memory);
    access = vm.snapshot(handle);
  }
  std::shared_ptr<RetryMemory> memory;
  GpuVm vm;
  AddressSpaceHandle handle;
  std::optional<GpuVmAccess> access;
};

std::array<uint32_t, 7> copy_packet(uint64_t source, uint64_t destination, uint32_t bytes) {
  return {1,
          bytes - 1,
          0,
          static_cast<uint32_t>(source),
          static_cast<uint32_t>(source >> 32),
          static_cast<uint32_t>(destination),
          static_cast<uint32_t>(destination >> 32)};
}

TEST(SdmaExecutorTest, CopyResumesAtFirstUncommittedPhysicalSpan) {
  ExecutorFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kSource = 0x100;
  constexpr uint64_t kDestination = 0x200;
  for (uint32_t index = 0; index < 32; ++index)
    fixture.memory->store<uint8_t>(kSource + index, static_cast<uint8_t>(index + 1));
  fixture.memory->set_unavailable_write(kDestination + 16);

  SdmaExecutor executor(SdmaPacketDialect::Gfx1250);
  const auto packet = copy_packet(kSource, kDestination, 32);
  const SdmaExecutionResult first = executor.start(packet, *fixture.access);

  EXPECT_EQ(first.outcome, SdmaExecutionOutcome::Unavailable);
  EXPECT_EQ(first.packet_dwords, packet.size());
  EXPECT_TRUE(first.operation_committed);
  EXPECT_FALSE(first.completion_published);
  EXPECT_FALSE(first.retire_packet);
  EXPECT_TRUE(executor.pending());
  EXPECT_EQ(fixture.memory->write_calls(kDestination), 1u);

  const SdmaExecutionResult resumed = executor.resume();
  EXPECT_EQ(resumed.outcome, SdmaExecutionOutcome::Complete);
  EXPECT_TRUE(resumed.operation_committed);
  EXPECT_TRUE(resumed.completion_published);
  EXPECT_TRUE(resumed.retire_packet);
  EXPECT_FALSE(executor.pending());
  EXPECT_EQ(fixture.memory->write_calls(kDestination), 1u);
  for (uint32_t index = 0; index < 32; ++index)
    EXPECT_EQ(fixture.memory->load<uint8_t>(kDestination + index), static_cast<uint8_t>(index + 1));
}

TEST(SdmaExecutorTest, SignalPublicationRetryDoesNotReplayCommittedAtomic) {
  ExecutorFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kSignalValue = 0x1408;
  constexpr uint64_t kMailbox = 0x1500;
  fixture.memory->store<uint64_t>(kSignalValue, 8);
  fixture.memory->store<uint64_t>(kSignalValue + 8, kMailbox);
  fixture.memory->store<uint32_t>(kSignalValue + 16, 37);
  fixture.memory->set_unavailable_atomic_store(kMailbox);
  uint32_t interrupts = 0;
  SdmaExecutor executor(SdmaPacketDialect::Gfx1250, {.poll_register = {},
                                                     .write_register = {},
                                                     .deliver_interrupt =
                                                         [&](uint32_t event) {
                                                           EXPECT_EQ(event, 37u);
                                                           ++interrupts;
                                                           return VmAccessOutcome::Complete;
                                                         },
                                                     .maintain_caches = {},
                                                     .timestamp = {}});
  const std::array<uint32_t, 8> packet = {
      10u | (47u << 25), static_cast<uint32_t>(kSignalValue), 0, UINT32_MAX, UINT32_MAX, 0, 0, 0};

  const SdmaExecutionResult first = executor.start(packet, *fixture.access);
  EXPECT_EQ(first.outcome, SdmaExecutionOutcome::Unavailable);
  EXPECT_TRUE(first.operation_committed);
  EXPECT_FALSE(first.completion_published);
  EXPECT_EQ(fixture.memory->load<uint64_t>(kSignalValue), 7u);
  EXPECT_EQ(fixture.memory->successful_compare_exchanges(), 1u);
  EXPECT_EQ(interrupts, 0u);

  const SdmaExecutionResult resumed = executor.resume();
  EXPECT_EQ(resumed.outcome, SdmaExecutionOutcome::Complete);
  EXPECT_EQ(fixture.memory->load<uint64_t>(kSignalValue), 7u);
  EXPECT_EQ(fixture.memory->load<uint64_t>(kMailbox), 37u);
  EXPECT_EQ(fixture.memory->successful_compare_exchanges(), 1u);
  EXPECT_EQ(interrupts, 1u);
}

TEST(SdmaExecutorTest, IndirectBufferRetainsNestedWriteProgress) {
  ExecutorFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kIndirect = 0x600;
  constexpr uint64_t kDestination = 0x700;
  const std::array<uint32_t, 12> indirect = {
      2,          static_cast<uint32_t>(kDestination),
      0,          7,
      0x04030201, 0x08070605,
      0x0c0b0a09, 0x100f0e0d,
      0x14131211, 0x18171615,
      0x1c1b1a19, 0x201f1e1d,
  };
  for (std::size_t index = 0; index < indirect.size(); ++index)
    fixture.memory->store<uint32_t>(kIndirect + index * 4, indirect[index]);
  fixture.memory->set_unavailable_write(kDestination + 16);
  const std::array<uint32_t, 6> packet = {
      4, static_cast<uint32_t>(kIndirect), 0, static_cast<uint32_t>(indirect.size()), 0, 0};
  SdmaExecutor executor(SdmaPacketDialect::Gfx1250);

  const SdmaExecutionResult first = executor.start(packet, *fixture.access);
  EXPECT_EQ(first.outcome, SdmaExecutionOutcome::Unavailable);
  EXPECT_TRUE(first.operation_committed);
  EXPECT_EQ(fixture.memory->write_calls(kDestination), 1u);
  const SdmaExecutionResult resumed = executor.resume();
  EXPECT_EQ(resumed.outcome, SdmaExecutionOutcome::Complete);
  EXPECT_EQ(resumed.packet_dwords, packet.size());
  EXPECT_EQ(fixture.memory->write_calls(kDestination), 1u);
  for (uint32_t index = 0; index < 32; ++index)
    EXPECT_EQ(fixture.memory->load<uint8_t>(kDestination + index), static_cast<uint8_t>(index + 1));
}

TEST(SdmaExecutorTest, MemoryPollRefreshesValueAfterUnsatisfiedPredicate) {
  ExecutorFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kPollAddress = 0x900;
  fixture.memory->store<uint64_t>(kPollAddress, 0);
  const std::array<uint32_t, 8> packet = {
      8u | (5u << 8) | (3u << 28),
      static_cast<uint32_t>(kPollAddress),
      static_cast<uint32_t>(kPollAddress >> 32),
      1,
      0,
      UINT32_MAX,
      UINT32_MAX,
      0,
  };
  SdmaExecutor executor(SdmaPacketDialect::Gfx1250);

  const SdmaExecutionResult first = executor.start(packet, *fixture.access);
  EXPECT_EQ(first.outcome, SdmaExecutionOutcome::Unavailable);
  EXPECT_TRUE(executor.pending());

  fixture.memory->store<uint64_t>(kPollAddress, 1);
  const SdmaExecutionResult resumed = executor.resume();
  EXPECT_EQ(resumed.outcome, SdmaExecutionOutcome::Complete);
  EXPECT_TRUE(resumed.retire_packet);
  EXPECT_FALSE(executor.pending());
}

TEST(SdmaExecutorTest, ReportsMalformedPacketWithoutCollapsingItIntoFault) {
  ExecutorFixture fixture;
  ASSERT_TRUE(fixture.access);
  SdmaExecutor executor(SdmaPacketDialect::Gfx1250);
  const std::array<uint32_t, 1> unsupported = {0xff};

  const SdmaExecutionResult result = executor.start(unsupported, *fixture.access);

  EXPECT_EQ(result.outcome, SdmaExecutionOutcome::Malformed);
  EXPECT_FALSE(result.retire_packet);
  EXPECT_FALSE(executor.pending());
}

} // namespace
} // namespace rocjitsu::amdgpu
