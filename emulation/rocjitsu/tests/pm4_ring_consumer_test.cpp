// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pm4_ring_consumer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

class FlatMemory final : public AddressSpaceTranslator, public PhysicalMemoryAccess {
public:
  FlatMemory() : bytes_(0x1000) {}

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
    std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(address), bytes.size(), bytes.begin());
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(VmMemoryDomain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::copy(bytes.begin(), bytes.end(), bytes_.begin() + static_cast<std::ptrdiff_t>(address));
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

private:
  std::vector<std::byte> bytes_;
  std::optional<uint64_t> unavailable_read_;
  std::optional<uint64_t> unavailable_store_once_;
};

class RingConsumerFixture {
public:
  RingConsumerFixture() : memory(std::make_shared<FlatMemory>()) {
    handle = vm.register_translated(1, memory, memory);
    access = vm.snapshot(handle);
  }

  std::shared_ptr<FlatMemory> memory;
  GpuVm vm;
  AddressSpaceHandle handle;
  std::optional<GpuVmAccess> access;
};

TEST(Pm4RingConsumerTest, ProcessesTheBoundedSupportedSubsetAcrossRingWrap) {
  RingConsumerFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kRing = 0x100;
  constexpr uint64_t kRingDwords = 4;
  fixture.memory->store<uint32_t>(kRing + 3 * sizeof(uint32_t), 0xc0017900);
  fixture.memory->store<uint32_t>(kRing + 0 * sizeof(uint32_t), 0x40);
  fixture.memory->store<uint32_t>(kRing + 1 * sizeof(uint32_t), 0xdeadbeef);
  fixture.memory->store<uint32_t>(kRing + 2 * sizeof(uint32_t), 0xffff1000);

  uint64_t register_dword = 0;
  uint32_t register_value = 0;
  Pm4PacketProcessor packet_processor({.write_uconfig_register = [&](uint64_t reg, uint32_t value) {
    register_dword = reg;
    register_value = value;
    return Pm4RegisterWriteStatus::Complete;
  }});
  Pm4RingConsumer consumer(packet_processor, fixture.vm, fixture.handle, kRing,
                           kRingDwords * sizeof(uint32_t), 0x80, 3);
  const Pm4RingResult result = consumer.consume(7);

  EXPECT_EQ(result.status, Pm4RingStatus::Complete);
  EXPECT_EQ(result.read_pointer, 7u);
  EXPECT_EQ(register_dword, 0xc040u);
  EXPECT_EQ(register_value, 0xdeadbeefu);
}

TEST(Pm4RingConsumerTest, StopsAtThePacketBudgetAndResumesAtTheNextPacket) {
  RingConsumerFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kRing = 0x100;
  constexpr uint64_t kReadPointer = 0x80;
  fixture.memory->store<uint32_t>(kRing, 0xffff1000);
  fixture.memory->store<uint32_t>(kRing + sizeof(uint32_t), 0xffff1000);

  Pm4PacketProcessor packet_processor;
  Pm4RingConsumer consumer(packet_processor, fixture.vm, fixture.handle, kRing,
                           4 * sizeof(uint32_t), kReadPointer, 0);

  const Pm4RingResult first = consumer.consume(2, 1);
  EXPECT_EQ(first.status, Pm4RingStatus::Complete);
  EXPECT_EQ(first.read_pointer, 1u);
  EXPECT_EQ(fixture.memory->load<uint32_t>(kReadPointer), 1u);

  const Pm4RingResult second = consumer.consume(2, 1);
  EXPECT_EQ(second.status, Pm4RingStatus::Complete);
  EXPECT_EQ(second.read_pointer, 2u);
  EXPECT_EQ(fixture.memory->load<uint32_t>(kReadPointer), 2u);
}

TEST(Pm4RingConsumerTest, LeavesTheUnsupportedPacketAtTheReadPointer) {
  RingConsumerFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kRing = 0x100;
  fixture.memory->store<uint32_t>(kRing, 0xc0002000);

  Pm4PacketProcessor packet_processor;
  Pm4RingConsumer consumer(packet_processor, fixture.vm, fixture.handle, kRing,
                           4 * sizeof(uint32_t), 0x80, 0);
  const Pm4RingResult result = consumer.consume(1);

  EXPECT_EQ(result.status, Pm4RingStatus::UnsupportedPacket);
  EXPECT_EQ(result.read_pointer, 0u);
  EXPECT_EQ(result.packet_header, 0xc0002000u);
}

TEST(Pm4RingConsumerTest, PreservesUnavailableAsARetryableOutcome) {
  RingConsumerFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kRing = 0x100;
  fixture.memory->make_read_unavailable(kRing);

  Pm4PacketProcessor packet_processor;
  Pm4RingConsumer consumer(packet_processor, fixture.vm, fixture.handle, kRing,
                           4 * sizeof(uint32_t), 0x80, 0);
  const Pm4RingResult result = consumer.consume(1);

  EXPECT_EQ(result.status, Pm4RingStatus::Blocked);
  EXPECT_EQ(result.read_pointer, 0u);
}

TEST(Pm4RingConsumerTest, RejectsARingWhoseAddressRangeWraps) {
  RingConsumerFixture fixture;
  ASSERT_TRUE(fixture.access);

  Pm4PacketProcessor packet_processor;
  Pm4RingConsumer consumer(packet_processor, fixture.vm, fixture.handle, UINT64_MAX - 7,
                           4 * sizeof(uint32_t), 0x80, 0);
  const Pm4RingResult result = consumer.consume(1);

  EXPECT_EQ(result.status, Pm4RingStatus::Malformed);
  EXPECT_EQ(result.read_pointer, 0u);
}

TEST(Pm4RingConsumerTest, PreservesTerminalPacketAcrossCursorPublicationRetry) {
  RingConsumerFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kRing = 0x100;
  constexpr uint64_t kReadPointer = 0x80;
  constexpr uint32_t kUnsupportedHeader = 0xc0002000;
  fixture.memory->store<uint32_t>(kRing, 0xc0017900);
  fixture.memory->store<uint32_t>(kRing + sizeof(uint32_t), 0x40);
  fixture.memory->store<uint32_t>(kRing + 2 * sizeof(uint32_t), 0xdeadbeef);
  fixture.memory->store<uint32_t>(kRing + 3 * sizeof(uint32_t), kUnsupportedHeader);
  fixture.memory->make_store_unavailable_once(kReadPointer);

  uint32_t register_writes = 0;
  Pm4PacketProcessor packet_processor({
      .write_uconfig_register =
          [&](uint64_t, uint32_t) {
            ++register_writes;
            return Pm4RegisterWriteStatus::Complete;
          },
  });
  Pm4RingConsumer consumer(packet_processor, fixture.vm, fixture.handle, kRing,
                           8 * sizeof(uint32_t), kReadPointer, 0);

  const Pm4RingResult blocked = consumer.consume(4);
  EXPECT_EQ(blocked.status, Pm4RingStatus::Blocked);
  EXPECT_EQ(blocked.read_pointer, 3u);
  EXPECT_EQ(register_writes, 1u);

  const Pm4RingResult terminal = consumer.consume(4);
  EXPECT_EQ(terminal.status, Pm4RingStatus::UnsupportedPacket);
  EXPECT_EQ(terminal.read_pointer, 3u);
  EXPECT_EQ(terminal.packet_header, kUnsupportedHeader);
  EXPECT_EQ(terminal.register_dword, 0u);
  EXPECT_EQ(register_writes, 1u);
  EXPECT_FALSE(consumer.in_flight());
}

} // namespace
} // namespace rocjitsu::amdgpu
