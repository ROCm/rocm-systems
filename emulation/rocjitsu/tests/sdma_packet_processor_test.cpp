// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/sdma_packet_processor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

static_assert(PacketProcessor<SdmaPacketProcessor>);
static_assert(
    std::same_as<decltype(std::declval<SdmaPacketProcessRequest>().access), const GpuVmAccess &>);
static_assert(std::same_as<decltype(std::declval<SdmaPacketProcessRequest>().continuation),
                           SdmaPacketContinuation &>);

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

class PacketProcessorFixture {
public:
  PacketProcessorFixture() : memory(std::make_shared<RetryMemory>()) {
    handle = vm.register_translated(7, memory, memory);
    access = vm.snapshot(handle);
  }
  std::shared_ptr<RetryMemory> memory;
  GpuVm vm;
  AddressSpaceHandle handle;
  std::optional<GpuVmAccess> access;
  SdmaPacketContinuation continuation;
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

TEST(SdmaPacketProcessorTest, CopyResumesAtFirstUncommittedPhysicalSpan) {
  PacketProcessorFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kSource = 0x100;
  constexpr uint64_t kDestination = 0x200;
  for (uint32_t index = 0; index < 32; ++index)
    fixture.memory->store<uint8_t>(kSource + index, static_cast<uint8_t>(index + 1));
  fixture.memory->set_unavailable_write(kDestination + 16);

  SdmaPacketProcessor processor(SdmaPacketDialect::Gfx1250);
  const auto packet = copy_packet(kSource, kDestination, 32);
  const SdmaPacketProcessResult first = processor.process({.available_dwords = packet,
                                                           .access = *fixture.access,
                                                           .continuation = fixture.continuation});

  EXPECT_EQ(first.packet.status, PacketProcessStatus::Blocked);
  EXPECT_EQ(first.packet.retirement_bytes, packet.size() * sizeof(uint32_t));
  EXPECT_EQ(first.packet.retirement, PacketRetirement::Hold);
  EXPECT_TRUE(first.operation_committed);
  EXPECT_FALSE(first.completion_published);
  EXPECT_TRUE(fixture.continuation.pending());
  EXPECT_EQ(fixture.memory->write_calls(kDestination), 1u);

  const SdmaPacketProcessResult resumed = processor.process(
      {.available_dwords = {}, .access = *fixture.access, .continuation = fixture.continuation});
  EXPECT_EQ(resumed.packet.status, PacketProcessStatus::Complete);
  EXPECT_EQ(resumed.packet.retirement, PacketRetirement::Retire);
  EXPECT_TRUE(resumed.operation_committed);
  EXPECT_TRUE(resumed.completion_published);
  EXPECT_FALSE(fixture.continuation.pending());
  EXPECT_EQ(fixture.memory->write_calls(kDestination), 1u);
  for (uint32_t index = 0; index < 32; ++index)
    EXPECT_EQ(fixture.memory->load<uint8_t>(kDestination + index), static_cast<uint8_t>(index + 1));
}

TEST(SdmaPacketProcessorTest, SignalPublicationRetryDoesNotReplayCommittedAtomic) {
  PacketProcessorFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kSignalValue = 0x1408;
  constexpr uint64_t kMailbox = 0x1500;
  fixture.memory->store<uint64_t>(kSignalValue, 8);
  fixture.memory->store<uint64_t>(kSignalValue + 8, kMailbox);
  fixture.memory->store<uint32_t>(kSignalValue + 16, 37);
  fixture.memory->set_unavailable_atomic_store(kMailbox);
  uint32_t interrupts = 0;
  SdmaPacketProcessor processor(SdmaPacketDialect::Gfx1250, {.read_register = {},
                                                             .write_register = {},
                                                             .deliver_interrupt =
                                                                 [&](uint32_t event) {
                                                                   EXPECT_EQ(event, 37u);
                                                                   ++interrupts;
                                                                   return VmAccessOutcome::Complete;
                                                                 },
                                                             .acquire_cache_maintenance = {},
                                                             .timestamp = {}});
  const std::array<uint32_t, 8> packet = {
      10u | (47u << 25), static_cast<uint32_t>(kSignalValue), 0, UINT32_MAX, UINT32_MAX, 0, 0, 0};

  const SdmaPacketProcessResult first = processor.process({.available_dwords = packet,
                                                           .access = *fixture.access,
                                                           .continuation = fixture.continuation});
  EXPECT_EQ(first.packet.status, PacketProcessStatus::Blocked);
  EXPECT_TRUE(first.operation_committed);
  EXPECT_FALSE(first.completion_published);
  EXPECT_EQ(fixture.memory->load<uint64_t>(kSignalValue), 7u);
  EXPECT_EQ(fixture.memory->successful_compare_exchanges(), 1u);
  EXPECT_EQ(interrupts, 0u);

  const SdmaPacketProcessResult resumed = processor.process(
      {.available_dwords = {}, .access = *fixture.access, .continuation = fixture.continuation});
  EXPECT_EQ(resumed.packet.status, PacketProcessStatus::Complete);
  EXPECT_EQ(fixture.memory->load<uint64_t>(kSignalValue), 7u);
  EXPECT_EQ(fixture.memory->load<uint64_t>(kMailbox), 37u);
  EXPECT_EQ(fixture.memory->successful_compare_exchanges(), 1u);
  EXPECT_EQ(interrupts, 1u);
}

TEST(SdmaPacketProcessorTest, CacheLeaseIsReleasedOnUnavailableAndReacquiredOnResume) {
  PacketProcessorFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kDestination = 0x280;
  fixture.memory->set_unavailable_write(kDestination);

  bool lease_active = false;
  uint32_t acquired = 0;
  uint32_t released = 0;
  SdmaPacketCallbacks callbacks;
  callbacks.acquire_cache_maintenance = [&](SdmaCacheOperation operation) {
    EXPECT_EQ(operation, SdmaCacheOperation::WritebackInvalidate);
    EXPECT_FALSE(lease_active);
    lease_active = true;
    ++acquired;
    return SdmaCacheLease([&] {
      EXPECT_TRUE(lease_active);
      lease_active = false;
      ++released;
    });
  };
  const std::array<uint32_t, 5> packet = {11, static_cast<uint32_t>(kDestination), 0, 0x11223344,
                                          3};
  SdmaPacketProcessor processor(SdmaPacketDialect::Gfx1250, std::move(callbacks));

  const SdmaPacketProcessResult first = processor.process({.available_dwords = packet,
                                                           .access = *fixture.access,
                                                           .continuation = fixture.continuation});
  EXPECT_EQ(first.packet.status, PacketProcessStatus::Blocked);
  EXPECT_FALSE(lease_active);
  EXPECT_EQ(acquired, 1u);
  EXPECT_EQ(released, 1u);

  const SdmaPacketProcessResult resumed = processor.process(
      {.available_dwords = {}, .access = *fixture.access, .continuation = fixture.continuation});
  EXPECT_EQ(resumed.packet.status, PacketProcessStatus::Complete);
  EXPECT_FALSE(lease_active);
  EXPECT_EQ(acquired, 2u);
  EXPECT_EQ(released, 2u);
  EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination), 0x11223344u);
}

TEST(SdmaPacketProcessorTest, IndirectBufferRetainsNestedWriteProgress) {
  PacketProcessorFixture fixture;
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
  SdmaPacketProcessor processor(SdmaPacketDialect::Gfx1250);

  const SdmaPacketProcessResult first = processor.process({.available_dwords = packet,
                                                           .access = *fixture.access,
                                                           .continuation = fixture.continuation});
  EXPECT_EQ(first.packet.status, PacketProcessStatus::Blocked);
  EXPECT_TRUE(first.operation_committed);
  EXPECT_EQ(fixture.memory->write_calls(kDestination), 1u);
  const SdmaPacketProcessResult resumed = processor.process(
      {.available_dwords = {}, .access = *fixture.access, .continuation = fixture.continuation});
  EXPECT_EQ(resumed.packet.status, PacketProcessStatus::Complete);
  EXPECT_EQ(resumed.packet.retirement_bytes, packet.size() * sizeof(uint32_t));
  EXPECT_EQ(fixture.memory->write_calls(kDestination), 1u);
  for (uint32_t index = 0; index < 32; ++index)
    EXPECT_EQ(fixture.memory->load<uint8_t>(kDestination + index), static_cast<uint8_t>(index + 1));
}

TEST(SdmaPacketProcessorTest, MemoryPollRefreshesValueAfterUnsatisfiedPredicate) {
  PacketProcessorFixture fixture;
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
  SdmaPacketProcessor processor(SdmaPacketDialect::Gfx1250);

  const SdmaPacketProcessResult first = processor.process({.available_dwords = packet,
                                                           .access = *fixture.access,
                                                           .continuation = fixture.continuation});
  EXPECT_EQ(first.packet.status, PacketProcessStatus::Blocked);
  EXPECT_TRUE(fixture.continuation.pending());

  fixture.memory->store<uint64_t>(kPollAddress, 1);
  const SdmaPacketProcessResult resumed = processor.process(
      {.available_dwords = {}, .access = *fixture.access, .continuation = fixture.continuation});
  EXPECT_EQ(resumed.packet.status, PacketProcessStatus::Complete);
  EXPECT_EQ(resumed.packet.retirement, PacketRetirement::Retire);
  EXPECT_FALSE(fixture.continuation.pending());
}

TEST(SdmaPacketProcessorTest, ReportsBoundedInputNeededWithoutRetainingAPartialPacket) {
  PacketProcessorFixture fixture;
  ASSERT_TRUE(fixture.access);
  SdmaPacketProcessor processor(SdmaPacketDialect::Gfx1250);

  const SdmaPacketProcessResult empty = processor.process(
      {.available_dwords = {}, .access = *fixture.access, .continuation = fixture.continuation});
  EXPECT_EQ(empty.packet.status, PacketProcessStatus::NeedInput);
  EXPECT_EQ(empty.packet.required_bytes, sizeof(uint32_t));
  EXPECT_FALSE(fixture.continuation.pending());

  const std::array<uint32_t, 1> fence_header = {5};
  const SdmaPacketProcessResult fixed = processor.process({.available_dwords = fence_header,
                                                           .access = *fixture.access,
                                                           .continuation = fixture.continuation});
  EXPECT_EQ(fixed.packet.status, PacketProcessStatus::NeedInput);
  EXPECT_EQ(fixed.packet.required_bytes, 4 * sizeof(uint32_t));
  EXPECT_FALSE(fixture.continuation.pending());

  const std::array<uint32_t, 4> write_prefix = {2, 0x200, 0, 3};
  const SdmaPacketProcessResult variable =
      processor.process({.available_dwords = write_prefix,
                         .access = *fixture.access,
                         .continuation = fixture.continuation});
  EXPECT_EQ(variable.packet.status, PacketProcessStatus::NeedInput);
  EXPECT_EQ(variable.packet.required_bytes, 8 * sizeof(uint32_t));
  EXPECT_FALSE(fixture.continuation.pending());

  const std::array<uint32_t, 1> unsupported_atomic = {10};
  const SdmaPacketProcessResult malformed =
      processor.process({.available_dwords = unsupported_atomic,
                         .access = *fixture.access,
                         .continuation = fixture.continuation});
  EXPECT_EQ(malformed.packet.status, PacketProcessStatus::Malformed);
  EXPECT_EQ(malformed.packet.required_bytes, 0u);
  EXPECT_FALSE(fixture.continuation.pending());
}

TEST(SdmaPacketProcessorTest, ConditionalReportsSkippedRetirementFromItsHeaderOnly) {
  PacketProcessorFixture fixture;
  ASSERT_TRUE(fixture.access);
  SdmaPacketProcessor processor(SdmaPacketDialect::Gfx1250);
  const std::array<uint32_t, 5> conditional = {9, 0x300, 0, 1, 7};

  const SdmaPacketProcessResult result = processor.process({.available_dwords = conditional,
                                                            .access = *fixture.access,
                                                            .continuation = fixture.continuation});

  EXPECT_EQ(result.packet.status, PacketProcessStatus::Complete);
  EXPECT_EQ(result.packet.retirement, PacketRetirement::Retire);
  EXPECT_EQ(result.packet.retirement_bytes, 12 * sizeof(uint32_t));
  EXPECT_FALSE(fixture.continuation.pending());
}

TEST(SdmaPacketProcessorTest, ProcessesOnlyTheDecodedPacketExtentFromALargeSuffix) {
  PacketProcessorFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kPollAddress = 0x900;
  fixture.memory->store<uint64_t>(kPollAddress, 0);

  std::vector<uint32_t> unread_suffix(4096, 0xff);
  const std::array<uint32_t, 8> poll = {
      8u | (5u << 8) | (3u << 28),
      static_cast<uint32_t>(kPollAddress),
      0,
      1,
      0,
      UINT32_MAX,
      UINT32_MAX,
      0,
  };
  std::copy(poll.begin(), poll.end(), unread_suffix.begin());
  SdmaPacketProcessor processor(SdmaPacketDialect::Gfx1250);

  const SdmaPacketProcessResult first = processor.process({.available_dwords = unread_suffix,
                                                           .access = *fixture.access,
                                                           .continuation = fixture.continuation});
  EXPECT_EQ(first.packet.status, PacketProcessStatus::Blocked);
  EXPECT_EQ(first.packet.retirement_bytes, poll.size() * sizeof(uint32_t));
  EXPECT_TRUE(fixture.continuation.pending());

  fixture.memory->store<uint64_t>(kPollAddress, 1);
  const SdmaPacketProcessResult resumed = processor.process(
      {.available_dwords = {}, .access = *fixture.access, .continuation = fixture.continuation});
  EXPECT_EQ(resumed.packet.status, PacketProcessStatus::Complete);
  EXPECT_EQ(resumed.packet.retirement_bytes, poll.size() * sizeof(uint32_t));
  EXPECT_FALSE(fixture.continuation.pending());
}

TEST(SdmaPacketProcessorTest, ReportsMalformedPacketWithoutCollapsingItIntoFault) {
  PacketProcessorFixture fixture;
  ASSERT_TRUE(fixture.access);
  SdmaPacketProcessor processor(SdmaPacketDialect::Gfx1250);
  const std::array<uint32_t, 1> unsupported = {0xff};

  const SdmaPacketProcessResult result = processor.process({.available_dwords = unsupported,
                                                            .access = *fixture.access,
                                                            .continuation = fixture.continuation});

  EXPECT_EQ(result.packet.status, PacketProcessStatus::Malformed);
  EXPECT_EQ(result.packet.retirement, PacketRetirement::Hold);
  EXPECT_FALSE(fixture.continuation.pending());
}

} // namespace
} // namespace rocjitsu::amdgpu
