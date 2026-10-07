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
  explicit RetryMemory(std::size_t size = 0x10000, uint64_t segment_size = 16)
      : bytes_(size), segment_size_(segment_size) {}

  VmTranslationResult translate(uint64_t address, std::size_t size, VmAccessKind) const override {
    if (size == 0 || address > bytes_.size() || size > bytes_.size() - address)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation = {.domain = VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = segment_size_ - (address % segment_size_),
                        .mtype = Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }

  VmAccessOutcome read(VmMemoryDomain, uint64_t address, std::span<std::byte> bytes) override {
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::ranges::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(address), bytes.size(),
                        bytes.begin());
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
    std::ranges::copy(bytes, bytes_.begin() + static_cast<std::ptrdiff_t>(address));
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
  uint64_t segment_size_;
  std::map<uint64_t, uint32_t> write_calls_;
  uint64_t unavailable_write_address_ = UINT64_MAX;
  uint64_t unavailable_atomic_store_address_ = UINT64_MAX;
  bool unavailable_write_returned_ = false;
  bool unavailable_atomic_store_returned_ = false;
  uint32_t successful_compare_exchanges_ = 0;
};

class PacketProcessorFixture {
public:
  explicit PacketProcessorFixture(std::size_t size = 0x10000, uint64_t segment_size = 16)
      : memory(std::make_shared<RetryMemory>(size, segment_size)) {
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

TEST(SdmaPacketProcessorTest, ExtendedCountCopiesBeyondFourMiB) {
  constexpr uint64_t kSource = 0x1000;
  constexpr uint64_t kDestination = 0x500000;
  constexpr uint32_t kBytes = 0x400010;
  for (const auto dialect :
       {SdmaPacketDialect::Legacy, SdmaPacketDialect::LegacyExtendedCount,
        SdmaPacketDialect::Gfx11Plus, SdmaPacketDialect::Gfx1250, SdmaPacketDialect::Rdna4}) {
    SCOPED_TRACE(static_cast<int>(dialect));
    PacketProcessorFixture fixture(kDestination + kBytes, 0x1000);
    ASSERT_TRUE(fixture.access);
    // Check both ends of the transfer and the first byte beyond the legacy limit.
    fixture.memory->store<uint32_t>(kSource, 0x12345678);
    fixture.memory->store<uint32_t>(kSource + 0x400000, 0x87654321);
    fixture.memory->store<uint32_t>(kSource + kBytes - 4, 0xabcdef01);
    SdmaPacketProcessor processor(dialect);
    auto packet = copy_packet(kSource, kDestination, kBytes);
    packet[1] |= 0xc0000000; // Reserved high bits must not extend the transfer.
    const SdmaPacketProcessResult result =
        processor.process({.available_dwords = packet,
                           .access = *fixture.access,
                           .continuation = fixture.continuation});
    ASSERT_EQ(result.packet.status, PacketProcessStatus::Complete);
    EXPECT_FALSE(fixture.continuation.pending());
    EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination), 0x12345678u);
    if (dialect == SdmaPacketDialect::Legacy) {
      EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination + 0x400000), 0u);
      EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination + kBytes - 4), 0u);
    } else {
      EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination + 0x400000), 0x87654321u);
      EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination + kBytes - 4), 0xabcdef01u);
    }
  }
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
  std::ranges::copy(poll, unread_suffix.begin());
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

std::array<uint32_t, 13> linear_rect_packet(bool gfx12_rect, uint64_t source, uint64_t destination,
                                            uint32_t element, uint32_t rect_x, uint32_t rect_y,
                                            uint32_t rect_z, uint32_t src_pitch_bytes,
                                            uint32_t dst_pitch_bytes, uint32_t src_slice_bytes = 0,
                                            uint32_t dst_slice_bytes = 0, uint32_t src_off_x = 0,
                                            uint32_t dst_off_x = 0, uint32_t src_off_z = 0,
                                            uint32_t dst_off_z = 0) {
  const uint32_t element_bytes = 1u << element;
  const uint32_t src_pitch_elements = src_pitch_bytes / element_bytes;
  const uint32_t dst_pitch_elements = dst_pitch_bytes / element_bytes;
  std::array<uint32_t, 13> packet{};
  packet[0] = 1u | (4u << 8) | (element << 29);
  packet[1] = static_cast<uint32_t>(source);
  packet[2] = static_cast<uint32_t>(source >> 32);
  packet[3] = src_off_x;
  packet[6] = static_cast<uint32_t>(destination);
  packet[7] = static_cast<uint32_t>(destination >> 32);
  packet[8] = dst_off_x;
  const uint32_t pitch_shift = gfx12_rect ? 16u : 13u;
  packet[4] = src_off_z | ((src_pitch_elements - 1u) << pitch_shift);
  packet[9] = dst_off_z | ((dst_pitch_elements - 1u) << pitch_shift);
  if (src_slice_bytes != 0 || dst_slice_bytes != 0) {
    packet[5] = src_slice_bytes / element_bytes - 1u;
    packet[10] = dst_slice_bytes / element_bytes - 1u;
  }
  packet[11] = (rect_x - 1u) | ((rect_y - 1u) << 16);
  packet[12] = rect_z - 1u;
  return packet;
}

TEST(SdmaPacketProcessorTest, LinearRectCopiesPitchedRowsAndSkipsTheGap) {
  const struct {
    bool gfx12_rect;
    SdmaPacketDialect dialect;
  } cases[] = {
      {false, SdmaPacketDialect::LegacyExtendedCount},
      {true, SdmaPacketDialect::Gfx1250},
      {true, SdmaPacketDialect::Rdna4},
  };
  for (const auto &test_case : cases) {
    PacketProcessorFixture fixture;
    ASSERT_TRUE(fixture.access);
    constexpr uint64_t kSource = 0x1000;
    constexpr uint64_t kDestination = 0x1800;
    fixture.memory->store<uint64_t>(kSource, 0x1122334455667788ull);
    fixture.memory->store<uint64_t>(kSource + 8, 0xaabbccddeeff0011ull);
    for (uint32_t i = 0; i < 32; ++i)
      fixture.memory->store<uint8_t>(kDestination + i, 0x5a);

    const std::array<uint32_t, 13> packet = linear_rect_packet(
        test_case.gfx12_rect, kSource, kDestination, /*element=*/0, /*rect_x=*/4, /*rect_y=*/2,
        /*rect_z=*/1, /*src_pitch_bytes=*/8, /*dst_pitch_bytes=*/16);
    SdmaPacketProcessor processor(test_case.dialect);
    const SdmaPacketProcessResult result =
        processor.process({.available_dwords = packet,
                           .access = *fixture.access,
                           .continuation = fixture.continuation});

    EXPECT_EQ(result.packet.status, PacketProcessStatus::Complete)
        << static_cast<int>(test_case.dialect);
    EXPECT_EQ(result.packet.retirement_bytes, packet.size() * sizeof(uint32_t));
    EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination), 0x55667788u);
    EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination + 16), 0xeeff0011u);
    EXPECT_EQ(fixture.memory->load<uint8_t>(kDestination + 4), 0x5a);
    EXPECT_EQ(fixture.memory->load<uint8_t>(kDestination + 15), 0x5a);
  }
}

TEST(SdmaPacketProcessorTest, LinearRectHonorsElementSizeAndSlicePitch) {
  PacketProcessorFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kSource = 0x2000;
  constexpr uint64_t kDestination = 0x3000;
  for (uint32_t i = 0; i < 128; ++i)
    fixture.memory->store<uint8_t>(kSource + i, static_cast<uint8_t>(i + 1));

  // 16-byte elements, two rows, then a second slice. Matches the dword-aligned
  // rectangle hipMemcpy2D submits, plus a slice so the Z stride is exercised.
  const std::array<uint32_t, 13> packet =
      linear_rect_packet(false, kSource, kDestination, /*element=*/4, /*rect_x=*/2, /*rect_y=*/2,
                         /*rect_z=*/2, /*src_pitch_bytes=*/32, /*dst_pitch_bytes=*/32,
                         /*src_slice_bytes=*/64, /*dst_slice_bytes=*/64);
  SdmaPacketProcessor processor(SdmaPacketDialect::LegacyExtendedCount);
  const SdmaPacketProcessResult result = processor.process({.available_dwords = packet,
                                                            .access = *fixture.access,
                                                            .continuation = fixture.continuation});

  EXPECT_EQ(result.packet.status, PacketProcessStatus::Complete);
  EXPECT_EQ(result.packet.retirement_bytes, 13u * sizeof(uint32_t));
  for (const uint64_t offset : {uint64_t{0}, uint64_t{32}, uint64_t{64}, uint64_t{96}}) {
    EXPECT_EQ(fixture.memory->load<uint64_t>(kDestination + offset),
              fixture.memory->load<uint64_t>(kSource + offset));
    EXPECT_EQ(fixture.memory->load<uint64_t>(kDestination + offset + 8),
              fixture.memory->load<uint64_t>(kSource + offset + 8));
  }
}

TEST(SdmaPacketProcessorTest, LinearRectUsesZOriginWhenTheCopyIsOneSlice) {
  const struct {
    bool gfx12_rect;
    SdmaPacketDialect dialect;
  } cases[] = {
      {false, SdmaPacketDialect::LegacyExtendedCount},
      {true, SdmaPacketDialect::Gfx1250},
      {true, SdmaPacketDialect::Rdna4},
  };
  for (const auto &test_case : cases) {
    PacketProcessorFixture fixture;
    ASSERT_TRUE(fixture.access);
    constexpr uint64_t kSource = 0x1000;
    constexpr uint64_t kDestination = 0x1800;
    fixture.memory->store<uint32_t>(kSource, 0x11111111u);
    fixture.memory->store<uint32_t>(kSource + 32, 0x22222222u);
    fixture.memory->store<uint32_t>(kDestination, 0);

    const std::array<uint32_t, 13> packet =
        linear_rect_packet(test_case.gfx12_rect, kSource, kDestination, /*element=*/0,
                           /*rect_x=*/4, /*rect_y=*/1, /*rect_z=*/1, /*src_pitch_bytes=*/4,
                           /*dst_pitch_bytes=*/4, /*src_slice_bytes=*/32, /*dst_slice_bytes=*/32,
                           /*src_off_x=*/0, /*dst_off_x=*/0, /*src_off_z=*/1, /*dst_off_z=*/0);
    SdmaPacketProcessor processor(test_case.dialect);
    const SdmaPacketProcessResult result =
        processor.process({.available_dwords = packet,
                           .access = *fixture.access,
                           .continuation = fixture.continuation});

    EXPECT_EQ(result.packet.status, PacketProcessStatus::Complete)
        << static_cast<int>(test_case.dialect);
    EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination), 0x22222222u)
        << static_cast<int>(test_case.dialect);
  }
}

TEST(SdmaPacketProcessorTest, LinearRectAcceptsTheGfx1250RuntimeFields) {
  PacketProcessorFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kSource = 0x1000;
  constexpr uint64_t kDestination = 0x1800;
  fixture.memory->store<uint64_t>(kSource, 0x1122334455667788ull);
  fixture.memory->store<uint64_t>(kSource + 8, 0xaabbccddeeff0011ull);
  for (uint32_t i = 0; i < 16; ++i)
    fixture.memory->store<uint8_t>(kDestination + i, 0x5a);

  std::array<uint32_t, 13> packet = linear_rect_packet(
      true, kSource, kDestination, /*element=*/0, /*rect_x=*/4, /*rect_y=*/1, /*rect_z=*/1,
      /*src_pitch_bytes=*/16, /*dst_pitch_bytes=*/16, /*src_slice_bytes=*/0,
      /*dst_slice_bytes=*/0, /*src_off_x=*/8, /*dst_off_x=*/4);
  // ROCr sets HEADER.npd and system scope for both sides on gfx1250. The scope
  // fields share bit positions with the pre-GFX12 endian swap.
  constexpr uint32_t kSystemScope = 3;
  packet[0] |= 1u << 28;
  packet[12] |= (kSystemScope << 18) | (kSystemScope << 26);

  SdmaPacketProcessor processor(SdmaPacketDialect::Gfx1250);
  const SdmaPacketProcessResult result = processor.process({.available_dwords = packet,
                                                            .access = *fixture.access,
                                                            .continuation = fixture.continuation});

  EXPECT_EQ(result.packet.status, PacketProcessStatus::Complete);
  EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination + 4), 0xeeff0011u);
  EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination), 0x5a5a5a5au);
  EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination + 8), 0x5a5a5a5au);
}

TEST(SdmaPacketProcessorTest, LinearRectRejectsEndianSwap) {
  PacketProcessorFixture fixture;
  ASSERT_TRUE(fixture.access);
  std::array<uint32_t, 13> packet = linear_rect_packet(false, 0x1000, 0x1800, 0, 4, 1, 1, 4, 4);
  packet[12] |= 1u << 16;

  SdmaPacketProcessor processor(SdmaPacketDialect::LegacyExtendedCount);
  const SdmaPacketProcessResult result = processor.process({.available_dwords = packet,
                                                            .access = *fixture.access,
                                                            .continuation = fixture.continuation});
  EXPECT_EQ(result.packet.status, PacketProcessStatus::Malformed);
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
