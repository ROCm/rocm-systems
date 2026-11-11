// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/sdma_ring_consumer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

class QueueMemory final : public AddressSpaceTranslator, public PhysicalMemoryAccess {
public:
  explicit QueueMemory(std::size_t size = 0x1000) : bytes_(size) {}

  VmTranslationResult translate(uint64_t address, std::size_t size, VmAccessKind) const override {
    if (size == 0 || address > bytes_.size() || size > bytes_.size() - address)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    constexpr uint64_t kSegmentBytes = 16;
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation = {.domain = VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = kSegmentBytes - (address % kSegmentBytes),
                        .mtype = Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }

  VmAccessOutcome read(VmMemoryDomain, uint64_t address, std::span<std::byte> bytes) override {
    ++read_attempts_[address];
    read_bytes_[address] += bytes.size();
    if (const VmAccessOutcome outcome = next(read_outcomes_, address);
        outcome != VmAccessOutcome::Complete)
      return outcome;
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::ranges::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(address), bytes.size(),
                        bytes.begin());
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(VmMemoryDomain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    if (const VmAccessOutcome outcome = next(write_outcomes_, address);
        outcome != VmAccessOutcome::Complete)
      return outcome;
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::ranges::copy(bytes, bytes_.begin() + static_cast<std::ptrdiff_t>(address));
    return VmAccessOutcome::Complete;
  }

  AtomicLoadResult atomic_load(VmMemoryDomain, uint64_t address, uint32_t width) override {
    ++atomic_load_attempts_[address];
    if (const VmAccessOutcome outcome = next(atomic_load_outcomes_, address);
        outcome != VmAccessOutcome::Complete)
      return {.outcome = outcome};
    if ((width != 4 && width != 8) || address > bytes_.size() || width > bytes_.size() - address)
      return {.outcome = VmAccessOutcome::Faulted};
    uint64_t value = 0;
    std::memcpy(&value, bytes_.data() + address, width);
    return {.outcome = VmAccessOutcome::Complete, .value = value};
  }

  VmAccessOutcome atomic_store(VmMemoryDomain, uint64_t address, uint32_t width,
                               uint64_t value) override {
    ++atomic_store_attempts_[address];
    if (const VmAccessOutcome outcome = next(atomic_store_outcomes_, address);
        outcome != VmAccessOutcome::Complete)
      return outcome;
    if ((width != 4 && width != 8) || address > bytes_.size() || width > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::memcpy(bytes_.data() + address, &value, width);
    return VmAccessOutcome::Complete;
  }

  AtomicCompareExchangeResult compare_exchange(VmMemoryDomain domain, uint64_t address,
                                               uint32_t width, uint64_t expected,
                                               uint64_t desired) override {
    const AtomicLoadResult loaded = atomic_load(domain, address, width);
    if (loaded.outcome != VmAccessOutcome::Complete)
      return {.outcome = loaded.outcome};
    if (loaded.value != expected)
      return {.outcome = VmAccessOutcome::Complete, .observed = loaded.value, .exchanged = false};
    const VmAccessOutcome outcome = atomic_store(domain, address, width, desired);
    return {.outcome = outcome,
            .observed = loaded.value,
            .exchanged = outcome == VmAccessOutcome::Complete};
  }

  template <typename T> void store(uint64_t address, const T &value) {
    std::memcpy(bytes_.data() + address, &value, sizeof(value));
  }

  template <typename T> T load(uint64_t address) const {
    T value{};
    std::memcpy(&value, bytes_.data() + address, sizeof(value));
    return value;
  }

  void return_next_read(uint64_t address, VmAccessOutcome outcome) {
    read_outcomes_[address].push_back(outcome);
  }
  void return_next_write(uint64_t address, VmAccessOutcome outcome) {
    write_outcomes_[address].push_back(outcome);
  }
  void return_next_atomic_load(uint64_t address, VmAccessOutcome outcome) {
    atomic_load_outcomes_[address].push_back(outcome);
  }
  void return_next_atomic_store(uint64_t address, VmAccessOutcome outcome) {
    atomic_store_outcomes_[address].push_back(outcome);
  }
  uint32_t read_attempts(uint64_t address) const { return attempts(read_attempts_, address); }
  std::size_t read_bytes(uint64_t begin, uint64_t end) const {
    std::size_t bytes = 0;
    for (const auto &[address, count] : read_bytes_) {
      if (address >= begin && address < end)
        bytes += count;
    }
    return bytes;
  }
  uint32_t atomic_load_attempts(uint64_t address) const {
    return attempts(atomic_load_attempts_, address);
  }
  uint32_t atomic_store_attempts(uint64_t address) const {
    return attempts(atomic_store_attempts_, address);
  }

private:
  static VmAccessOutcome next(std::map<uint64_t, std::deque<VmAccessOutcome>> &outcomes,
                              uint64_t address) {
    std::map<uint64_t, std::deque<VmAccessOutcome>>::iterator found = outcomes.find(address);
    if (found == outcomes.end() || found->second.empty())
      return VmAccessOutcome::Complete;
    const VmAccessOutcome outcome = found->second.front();
    found->second.pop_front();
    return outcome;
  }

  static uint32_t attempts(const std::map<uint64_t, uint32_t> &values, uint64_t address) {
    const std::map<uint64_t, uint32_t>::const_iterator found = values.find(address);
    return found == values.end() ? 0 : found->second;
  }

  std::vector<std::byte> bytes_;
  std::map<uint64_t, std::deque<VmAccessOutcome>> read_outcomes_;
  std::map<uint64_t, std::deque<VmAccessOutcome>> write_outcomes_;
  std::map<uint64_t, std::deque<VmAccessOutcome>> atomic_load_outcomes_;
  std::map<uint64_t, std::deque<VmAccessOutcome>> atomic_store_outcomes_;
  std::map<uint64_t, uint32_t> read_attempts_;
  std::map<uint64_t, std::size_t> read_bytes_;
  std::map<uint64_t, uint32_t> atomic_load_attempts_;
  std::map<uint64_t, uint32_t> atomic_store_attempts_;
};

class RingConsumerFixture {
public:
  static constexpr uint64_t kRing = 0x100;
  static constexpr uint64_t kReadPointer = 0x80;

  explicit RingConsumerFixture(std::size_t size = 0x1000)
      : memory(std::make_shared<QueueMemory>(size)) {
    address_space = vm.register_translated(7, memory, memory);
  }

  SdmaRingConsumer make_consumer(std::optional<uint64_t> initial = uint64_t{0},
                                 uint64_t ring_bytes = 64, SdmaPacketCallbacks callbacks = {}) {
    return SdmaRingConsumer(vm,
                            {.address_space = address_space,
                             .ring_base = kRing,
                             .ring_bytes = ring_bytes,
                             .read_pointer_address = kReadPointer,
                             .initial_cursor = initial},
                            SdmaPacketDialect::Gfx1250, std::move(callbacks));
  }

  std::shared_ptr<QueueMemory> memory;
  GpuVm vm;
  AddressSpaceHandle address_space;
};

TEST(SdmaRingConsumerTest, SwapExchangesBothRangesAndRetiresTheFollowingFence) {
  for (const SdmaPacketDialect dialect :
       {SdmaPacketDialect::LegacyExtendedCount, SdmaPacketDialect::Gfx1250}) {
    SCOPED_TRACE(static_cast<int>(dialect));
    RingConsumerFixture fixture;
    constexpr uint64_t kAddressA = 0x400;
    constexpr uint64_t kAddressB = 0x500;
    constexpr uint64_t kFence = 0x600;
    for (uint32_t index = 0; index < 64; ++index) {
      fixture.memory->store<uint8_t>(kAddressA + index, index + 1);
      fixture.memory->store<uint8_t>(kAddressB + index, 255 - index);
    }
    // ROCr's seven-dword COPY_LINEAR_SWAP followed by its completion fence.
    const std::array<uint32_t, 11> packets = {0x901, 63, 0,      kAddressA, 0,         kAddressB,
                                              0,     5,  kFence, 0,         0x12345678};
    fixture.memory->store(RingConsumerFixture::kRing, packets);
    SdmaRingConsumer consumer(fixture.vm,
                              {.address_space = fixture.address_space,
                               .ring_base = RingConsumerFixture::kRing,
                               .ring_bytes = 64,
                               .read_pointer_address = RingConsumerFixture::kReadPointer,
                               .initial_cursor = 0},
                              dialect);

    ASSERT_EQ(consumer.service(sizeof(packets)), SdmaRingStatus::Idle);
    EXPECT_EQ(consumer.cursor(), sizeof(packets));
    EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer), sizeof(packets));
    EXPECT_EQ(fixture.memory->load<uint32_t>(kFence), 0x12345678u);
    for (uint32_t index = 0; index < 64; ++index) {
      EXPECT_EQ(fixture.memory->load<uint8_t>(kAddressA + index), 255 - index);
      EXPECT_EQ(fixture.memory->load<uint8_t>(kAddressB + index), index + 1);
    }
  }
}

TEST(SdmaRingConsumerTest, MalformedPacketWarnsOnceWithItsHeaderAndRing) {
  RingConsumerFixture fixture;
  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing, 0x00007f01);
  SdmaRingConsumer consumer = fixture.make_consumer();
  testing::internal::CaptureStderr();
  const SdmaRingStatus first = consumer.service(4);
  const SdmaRingStatus second = consumer.service(4);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_EQ(first, SdmaRingStatus::Malformed);
  EXPECT_EQ(second, SdmaRingStatus::Malformed);
  EXPECT_NE(log.find("[rj warn]"), std::string::npos);
  EXPECT_NE(log.find("ring=0x100"), std::string::npos);
  EXPECT_NE(log.find("header=0x00007f01"), std::string::npos);
  EXPECT_NE(log.find("opcode=1 sub-opcode=127"), std::string::npos);
  EXPECT_EQ(std::ranges::count(log, '\n'), 1);
  EXPECT_EQ(consumer.cursor(), 0u);
}

TEST(SdmaRingConsumerTest, SwapBroadcastFlagAtPublishedRingTailReportsItsHeader) {
  for (const SdmaPacketDialect dialect :
       {SdmaPacketDialect::LegacyExtendedCount, SdmaPacketDialect::Gfx1250}) {
    SCOPED_TRACE(static_cast<int>(dialect));
    RingConsumerFixture fixture;
    const uint32_t header = dialect == SdmaPacketDialect::Gfx1250 ? 0x08000901 : 0x10000901;
    // Publish only the seven swap dwords. Unrelated trailing words must not
    // determine whether the decoder can identify the unsupported flag.
    const std::array<uint32_t, 7> packet = {header, 63, 0, 0x400, 0, 0x500, 0};
    fixture.memory->store(RingConsumerFixture::kRing, packet);
    fixture.memory->store<uint64_t>(0x400, 123);
    fixture.memory->store<uint64_t>(0x500, 456);
    SdmaRingConsumer consumer(fixture.vm,
                              {.address_space = fixture.address_space,
                               .ring_base = RingConsumerFixture::kRing,
                               .ring_bytes = 64,
                               .read_pointer_address = RingConsumerFixture::kReadPointer,
                               .initial_cursor = 0},
                              dialect);
    testing::internal::CaptureStderr();
    const SdmaRingStatus first = consumer.service(sizeof(packet));
    const SdmaRingStatus second = consumer.service(sizeof(packet));
    const std::string log = testing::internal::GetCapturedStderr();

    EXPECT_EQ(first, SdmaRingStatus::Malformed);
    EXPECT_EQ(second, SdmaRingStatus::Malformed);
    EXPECT_NE(log.find("root_cursor=0x0"), std::string::npos);
    EXPECT_NE(
        log.find(dialect == SdmaPacketDialect::Gfx1250 ? "header=0x08000901" : "header=0x10000901"),
        std::string::npos);
    EXPECT_EQ(std::ranges::count(log, '\n'), 1);
    EXPECT_EQ(consumer.cursor(), 0u);
    EXPECT_EQ(fixture.memory->load<uint64_t>(0x400), 123u);
    EXPECT_EQ(fixture.memory->load<uint64_t>(0x500), 456u);
  }
}

TEST(SdmaRingConsumerTest, SwapWaitSignalBlocksThenRetiresTheFollowingFence) {
  RingConsumerFixture fixture(0x2000);
  constexpr uint64_t kAddressA = 0x400;
  constexpr uint64_t kAddressB = 0x500;
  constexpr uint64_t kWaitAddress = 0x1600;
  constexpr uint64_t kSignalAddress = 0x1700;
  constexpr uint64_t kFenceAddress = 0x800;
  fixture.memory->store<uint64_t>(kAddressA, 0x1122334455667788);
  fixture.memory->store<uint64_t>(kAddressB, 0xaabbccddeeff0011);
  fixture.memory->store<uint64_t>(kWaitAddress, 1);
  fixture.memory->store<uint64_t>(kSignalAddress, 1);
  const std::array<uint32_t, 23> packets = {0xc0000901,
                                            3u | (3u << 18),
                                            kWaitAddress,
                                            0,
                                            0,
                                            0,
                                            UINT32_MAX,
                                            UINT32_MAX,
                                            63,
                                            (3u << 18) | (3u << 26),
                                            kAddressA,
                                            0,
                                            kAddressB,
                                            0,
                                            0x70u | (3u << 18),
                                            kSignalAddress,
                                            0,
                                            1,
                                            0,
                                            5,
                                            kFenceAddress,
                                            0,
                                            0x12345678};
  fixture.memory->store(RingConsumerFixture::kRing, packets);
  SdmaRingConsumer consumer = fixture.make_consumer(0, 128);

  EXPECT_EQ(consumer.service(sizeof(packets)), SdmaRingStatus::Blocked);
  EXPECT_EQ(consumer.cursor(), 0u);
  EXPECT_EQ(fixture.memory->load<uint64_t>(kAddressA), 0x1122334455667788u);
  EXPECT_EQ(fixture.memory->load<uint64_t>(kSignalAddress), 1u);
  EXPECT_EQ(fixture.memory->load<uint32_t>(kFenceAddress), 0u);
  fixture.memory->store<uint64_t>(kWaitAddress, 0);
  ASSERT_EQ(consumer.service(sizeof(packets)), SdmaRingStatus::Idle);
  EXPECT_EQ(consumer.cursor(), sizeof(packets));
  EXPECT_EQ(fixture.memory->load<uint64_t>(kAddressA), 0xaabbccddeeff0011u);
  EXPECT_EQ(fixture.memory->load<uint64_t>(kAddressB), 0x1122334455667788u);
  EXPECT_EQ(fixture.memory->load<uint64_t>(kSignalAddress), 0u);
  EXPECT_EQ(fixture.memory->load<uint32_t>(kFenceAddress), 0x12345678u);
}

TEST(SdmaRingConsumerTest, ProducerFailureDoesNotBlameTheBufferedPacket) {
  RingConsumerFixture fixture(0x2000);
  constexpr uint64_t kWaitAddress = 0x1600;
  constexpr uint64_t kAddressA = 0x400;
  constexpr uint64_t kAddressB = 0x500;
  fixture.memory->store<uint64_t>(kWaitAddress, 1);
  const std::array<uint32_t, 14> packet = {0x40000901,
                                           3,
                                           static_cast<uint32_t>(kWaitAddress),
                                           0,
                                           0,
                                           0,
                                           UINT32_MAX,
                                           UINT32_MAX,
                                           63,
                                           0,
                                           static_cast<uint32_t>(kAddressA),
                                           0,
                                           static_cast<uint32_t>(kAddressB),
                                           0};
  fixture.memory->store(RingConsumerFixture::kRing, packet);
  SdmaRingConsumer consumer = fixture.make_consumer();
  ASSERT_EQ(consumer.service(sizeof(packet)), SdmaRingStatus::Blocked);

  testing::internal::CaptureStderr();
  const SdmaRingStatus result = consumer.service(4);
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_EQ(result, SdmaRingStatus::Malformed);
  EXPECT_NE(log.find("malformed SDMA ring: ring=0x100 cursor=0x0"), std::string::npos);
  EXPECT_EQ(log.find("header="), std::string::npos);
  EXPECT_EQ(std::ranges::count(log, '\n'), 1);
}

TEST(SdmaRingConsumerTest, NestedMalformedHeaderSurvivesRetirementPublicationRetry) {
  for (const bool retry : {false, true}) {
    for (const VmAccessOutcome publication :
         {VmAccessOutcome::Complete, VmAccessOutcome::Malformed, VmAccessOutcome::Faulted}) {
      SCOPED_TRACE(retry);
      SCOPED_TRACE(static_cast<int>(publication));
      RingConsumerFixture fixture;
      constexpr uint64_t kIndirect = 0x400;
      constexpr uint64_t kDestination = 0x600;
      const std::array<uint32_t, 6> nested = {
          2, static_cast<uint32_t>(kDestination), 0, 0, 0x12345678, 0x00007f01};
      const std::array<uint32_t, 6> packet = {
          4, static_cast<uint32_t>(kIndirect), 0, nested.size(), 0, 0};
      fixture.memory->store(kIndirect, nested);
      fixture.memory->store(RingConsumerFixture::kRing, packet);
      if (retry)
        fixture.memory->return_next_atomic_store(RingConsumerFixture::kReadPointer,
                                                 VmAccessOutcome::Unavailable);
      fixture.memory->return_next_atomic_store(RingConsumerFixture::kReadPointer, publication);
      SdmaRingConsumer consumer = fixture.make_consumer();

      testing::internal::CaptureStderr();
      if (retry) {
        const SdmaRingStatus blocked = consumer.service(sizeof(packet));
        EXPECT_EQ(blocked, SdmaRingStatus::Blocked);
      }
      const SdmaRingStatus first = consumer.service(sizeof(packet));
      const SdmaRingStatus second = consumer.service(sizeof(packet));
      const SdmaRingStatus third = consumer.service(sizeof(packet));
      const std::string log = testing::internal::GetCapturedStderr();
      const SdmaRingStatus terminal = publication == VmAccessOutcome::Faulted
                                          ? SdmaRingStatus::Faulted
                                          : SdmaRingStatus::Malformed;
      EXPECT_EQ(first, terminal);
      EXPECT_EQ(second, terminal);
      EXPECT_EQ(third, terminal);
      EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination), 0x12345678u);
      EXPECT_EQ(consumer.cursor(), sizeof(packet));
      EXPECT_EQ(fixture.memory->atomic_store_attempts(RingConsumerFixture::kReadPointer),
                retry ? 2u : 1u);
      EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer),
                publication == VmAccessOutcome::Complete ? sizeof(packet) : 0u);
      EXPECT_NE(log.find("malformed SDMA packet (root or indirect)"), std::string::npos);
      EXPECT_NE(log.find("root_cursor=0x0 header=0x00007f01 opcode=1 sub-opcode=127"),
                std::string::npos);
      EXPECT_EQ(std::ranges::count(log, '\n'), 1);
    }
  }
}

TEST(ConsumerCursorJournalTest, PublishesThroughTheRetirementSnapshot) {
  RingConsumerFixture fixture;
  std::optional<GpuVmAccess> retirement_access = fixture.vm.snapshot_pinned(fixture.address_space);
  ASSERT_TRUE(retirement_access);

  ConsumerCursorJournal journal(RingConsumerFixture::kReadPointer, 0);
  journal.retire(4, 4, *retirement_access);

  std::shared_ptr<QueueMemory> replacement = std::make_shared<QueueMemory>();
  replacement->store<uint64_t>(RingConsumerFixture::kReadPointer, 99);
  ASSERT_TRUE(fixture.vm.replace_translated(fixture.address_space, replacement, replacement));

  EXPECT_EQ(journal.publish(), VmAccessOutcome::Complete);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer), 4u);
  EXPECT_EQ(replacement->load<uint64_t>(RingConsumerFixture::kReadPointer), 99u);
  EXPECT_FALSE(journal.publication_pending());
}

TEST(ConsumerCursorJournalTest, SupportsThirtyTwoBitHardwareCursors) {
  RingConsumerFixture fixture;
  fixture.memory->store<uint64_t>(RingConsumerFixture::kReadPointer, 0xfeedface00000001ULL);
  std::optional<GpuVmAccess> access = fixture.vm.snapshot(fixture.address_space);
  ASSERT_TRUE(access);

  ConsumerCursorJournal journal(RingConsumerFixture::kReadPointer, std::nullopt, sizeof(uint32_t));
  EXPECT_EQ(journal.initialize(*access), VmAccessOutcome::Complete);
  EXPECT_EQ(journal.cursor(), 1u);

  journal.retire(7, 7, *access);
  EXPECT_EQ(journal.publish(), VmAccessOutcome::Complete);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer),
            0xfeedface00000007ULL);
}

TEST(SdmaRingConsumerTest, InitialCursorOverridesMemoryAndAbsentCursorUsesAtomicLoad) {
  RingConsumerFixture fixture;
  fixture.memory->store<uint64_t>(RingConsumerFixture::kReadPointer, 0);
  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing + 4, 0);
  SdmaRingConsumer supplied = fixture.make_consumer(4);

  EXPECT_EQ(supplied.service(8), SdmaRingStatus::Idle);
  EXPECT_EQ(supplied.cursor(), 8u);
  EXPECT_EQ(fixture.memory->atomic_load_attempts(RingConsumerFixture::kReadPointer), 0u);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer), 8u);

  fixture.memory->store<uint64_t>(RingConsumerFixture::kReadPointer, 8);
  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing + 8, 0);
  SdmaRingConsumer loaded = fixture.make_consumer(std::nullopt);
  EXPECT_EQ(loaded.service(12), SdmaRingStatus::Idle);
  EXPECT_EQ(loaded.cursor(), 12u);
  EXPECT_EQ(fixture.memory->atomic_load_attempts(RingConsumerFixture::kReadPointer), 1u);
}

TEST(SdmaRingConsumerTest, FetchesAcrossRingWrapWithoutReplayingCompletedSegment) {
  RingConsumerFixture fixture;
  constexpr uint64_t kDestination = 0x300;
  const std::array<uint32_t, 4> fence = {5, static_cast<uint32_t>(kDestination), 0, 0x12345678};
  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing + 12, fence[0]);
  fixture.memory->store(RingConsumerFixture::kRing,
                        std::array<uint32_t, 3>{fence[1], fence[2], fence[3]});
  fixture.memory->return_next_read(RingConsumerFixture::kRing, VmAccessOutcome::Unavailable);
  SdmaRingConsumer consumer = fixture.make_consumer(12, 16);

  EXPECT_EQ(consumer.service(28), SdmaRingStatus::Blocked);
  EXPECT_EQ(fixture.memory->read_attempts(RingConsumerFixture::kRing + 12), 1u);
  EXPECT_EQ(consumer.service(28), SdmaRingStatus::Idle);
  EXPECT_EQ(fixture.memory->read_attempts(RingConsumerFixture::kRing + 12), 1u);
  EXPECT_EQ(fixture.memory->read_attempts(RingConsumerFixture::kRing), 2u);
  EXPECT_EQ(consumer.cursor(), 28u);
  EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination), fence[3]);
}

TEST(SdmaRingConsumerTest, FetchesOnlyTheHeadPacketBeforeYieldingAtTheBudget) {
  RingConsumerFixture fixture;
  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing, 0);
  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing + sizeof(uint32_t), 0);
  SdmaRingConsumer consumer = fixture.make_consumer(0);

  EXPECT_EQ(consumer.service(2 * sizeof(uint32_t), 1), SdmaRingStatus::Runnable);
  EXPECT_EQ(consumer.cursor(), sizeof(uint32_t));
  EXPECT_EQ(fixture.memory->read_bytes(RingConsumerFixture::kRing, RingConsumerFixture::kRing + 64),
            sizeof(uint32_t));
  EXPECT_EQ(fixture.memory->read_attempts(RingConsumerFixture::kRing + sizeof(uint32_t)), 0u);

  EXPECT_EQ(consumer.service(2 * sizeof(uint32_t), 1), SdmaRingStatus::Idle);
  EXPECT_EQ(fixture.memory->read_bytes(RingConsumerFixture::kRing, RingConsumerFixture::kRing + 64),
            2 * sizeof(uint32_t));
}

TEST(SdmaRingConsumerTest, RetiresTheHeadBeforeAnUnavailableLaterPacket) {
  RingConsumerFixture fixture;
  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing + 12, 0);
  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing + 16, 0);
  fixture.memory->return_next_read(RingConsumerFixture::kRing + 16, VmAccessOutcome::Unavailable);
  SdmaRingConsumer consumer = fixture.make_consumer(12);

  EXPECT_EQ(consumer.service(20), SdmaRingStatus::Blocked);
  EXPECT_EQ(consumer.cursor(), 16u);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer), 16u);
  EXPECT_EQ(fixture.memory->read_attempts(RingConsumerFixture::kRing + 12), 1u);
  EXPECT_EQ(fixture.memory->read_attempts(RingConsumerFixture::kRing + 16), 1u);

  EXPECT_EQ(consumer.service(20), SdmaRingStatus::Idle);
  EXPECT_EQ(consumer.cursor(), 20u);
  EXPECT_EQ(fixture.memory->read_attempts(RingConsumerFixture::kRing + 12), 1u);
  EXPECT_EQ(fixture.memory->read_attempts(RingConsumerFixture::kRing + 16), 2u);
}

TEST(SdmaRingConsumerTest, RejectsAnIncompletePublishedPacketWithoutRefetchingItsPrefix) {
  RingConsumerFixture fixture;
  constexpr uint64_t kDestination = 0x300;
  const std::array<uint32_t, 4> fence = {5, static_cast<uint32_t>(kDestination), 0, 0x12345678};
  fixture.memory->store(RingConsumerFixture::kRing, fence);
  SdmaRingConsumer consumer = fixture.make_consumer(0);

  EXPECT_EQ(consumer.service(sizeof(uint32_t)), SdmaRingStatus::Malformed);
  EXPECT_EQ(consumer.cursor(), 0u);
  EXPECT_EQ(fixture.memory->read_attempts(RingConsumerFixture::kRing), 1u);
  EXPECT_EQ(consumer.service(sizeof(fence)), SdmaRingStatus::Malformed);
  EXPECT_EQ(fixture.memory->read_attempts(RingConsumerFixture::kRing), 1u);
  EXPECT_EQ(fixture.memory->load<uint32_t>(kDestination), 0u);
}

TEST(SdmaRingConsumerTest, ConditionalRetirementSkipsTheReportedStreamExtent) {
  RingConsumerFixture fixture;
  constexpr uint64_t kCondition = 0x300;
  fixture.memory->store<uint32_t>(kCondition, 0);
  const std::array<uint32_t, 7> conditional = {
      9, static_cast<uint32_t>(kCondition), 0, 1, 2, 0xff, 0xff};
  fixture.memory->store(RingConsumerFixture::kRing, conditional);
  SdmaRingConsumer consumer = fixture.make_consumer(0);

  EXPECT_EQ(consumer.service(sizeof(conditional)), SdmaRingStatus::Idle);
  EXPECT_EQ(consumer.cursor(), sizeof(conditional));
  EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer), sizeof(conditional));
}

TEST(SdmaRingConsumerTest,
     ConditionalRetirementWaitsForTheSkippedExtentWithoutReevaluatingTheCondition) {
  RingConsumerFixture fixture;
  constexpr uint64_t kCondition = 0x300;
  fixture.memory->store<uint32_t>(kCondition, 0);
  const std::array<uint32_t, 5> conditional = {9, static_cast<uint32_t>(kCondition), 0, 1, 7};
  fixture.memory->store(RingConsumerFixture::kRing, conditional);
  SdmaRingConsumer consumer = fixture.make_consumer(0);

  EXPECT_EQ(consumer.service(sizeof(conditional)), SdmaRingStatus::Blocked);
  EXPECT_EQ(consumer.cursor(), 0u);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer), 0u);
  EXPECT_EQ(fixture.memory->read_attempts(kCondition), 1u);

  fixture.memory->store<uint32_t>(kCondition, 1);
  constexpr uint64_t kRetiredBytes = 12 * sizeof(uint32_t);
  EXPECT_EQ(consumer.service(kRetiredBytes), SdmaRingStatus::Idle);
  EXPECT_EQ(consumer.cursor(), kRetiredBytes);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer), kRetiredBytes);
  EXPECT_EQ(fixture.memory->read_attempts(kCondition), 1u);
}

TEST(SdmaRingConsumerTest, RejectsAConditionalRetirementExtentLargerThanTheRing) {
  RingConsumerFixture fixture;
  constexpr uint64_t kCondition = 0x300;
  fixture.memory->store<uint32_t>(kCondition, 0);
  const std::array<uint32_t, 5> conditional = {9, static_cast<uint32_t>(kCondition), 0, 1, 20};
  fixture.memory->store(RingConsumerFixture::kRing, conditional);
  SdmaRingConsumer consumer = fixture.make_consumer(0);

  EXPECT_EQ(consumer.service(sizeof(conditional)), SdmaRingStatus::Malformed);
  EXPECT_EQ(consumer.cursor(), 0u);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer), 0u);
}

TEST(SdmaRingConsumerTest, ReportsRunnableWhenThePacketBudgetLeavesWork) {
  RingConsumerFixture fixture;
  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing, 0);
  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing + sizeof(uint32_t), 0);
  SdmaRingConsumer consumer = fixture.make_consumer(0);

  EXPECT_EQ(consumer.service(2 * sizeof(uint32_t), 1), SdmaRingStatus::Runnable);
  EXPECT_EQ(consumer.cursor(), sizeof(uint32_t));
  EXPECT_EQ(consumer.service(2 * sizeof(uint32_t), 1), SdmaRingStatus::Idle);
  EXPECT_EQ(consumer.cursor(), 2 * sizeof(uint32_t));
}

TEST(SdmaRingConsumerTest, RetriesBlockedInitialLoadFetchPacketProcessingAndPublication) {
  RingConsumerFixture fixture;
  fixture.memory->store<uint64_t>(RingConsumerFixture::kReadPointer, 0);
  fixture.memory->return_next_atomic_load(RingConsumerFixture::kReadPointer,
                                          VmAccessOutcome::Unavailable);
  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing, 0);
  fixture.memory->return_next_read(RingConsumerFixture::kRing, VmAccessOutcome::Unavailable);
  fixture.memory->return_next_atomic_store(RingConsumerFixture::kReadPointer,
                                           VmAccessOutcome::Unavailable);
  SdmaRingConsumer consumer = fixture.make_consumer(std::nullopt);

  EXPECT_EQ(consumer.service(4), SdmaRingStatus::Blocked);
  EXPECT_EQ(consumer.service(4), SdmaRingStatus::Blocked);
  EXPECT_EQ(consumer.service(4), SdmaRingStatus::Blocked);
  EXPECT_EQ(consumer.cursor(), 4u) << "retired cursor is visible while publication retries";
  EXPECT_EQ(consumer.service(4), SdmaRingStatus::Idle);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer), 4u);

  constexpr uint64_t kPoll = 0x300;
  const std::array<uint32_t, 8> poll = {8u | (5u << 8) | (3u << 28),
                                        static_cast<uint32_t>(kPoll),
                                        0,
                                        1,
                                        0,
                                        UINT32_MAX,
                                        UINT32_MAX,
                                        0};
  fixture.memory->store(RingConsumerFixture::kRing + 16, poll);
  fixture.memory->store<uint64_t>(kPoll, 1);
  fixture.memory->return_next_atomic_load(kPoll, VmAccessOutcome::Unavailable);
  SdmaRingConsumer processor_retry = fixture.make_consumer(16);
  EXPECT_EQ(processor_retry.service(48), SdmaRingStatus::Blocked);
  EXPECT_EQ(processor_retry.cursor(), 16u);
  EXPECT_EQ(processor_retry.service(48), SdmaRingStatus::Idle);
  EXPECT_EQ(processor_retry.cursor(), 48u);
}

TEST(SdmaRingConsumerTest, RevokedTransactionBecomesTerminalInsteadOfBlocked) {
  RingConsumerFixture fixture;
  fixture.memory->store<uint64_t>(RingConsumerFixture::kReadPointer, 0);
  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing, 0);
  fixture.memory->return_next_atomic_load(RingConsumerFixture::kReadPointer,
                                          VmAccessOutcome::Unavailable);
  SdmaRingConsumer consumer = fixture.make_consumer(std::nullopt);

  EXPECT_EQ(consumer.service(sizeof(uint32_t)), SdmaRingStatus::Blocked);
  ASSERT_TRUE(fixture.vm.invalidate(fixture.address_space));

  EXPECT_EQ(consumer.service(sizeof(uint32_t)), SdmaRingStatus::Faulted);
  ASSERT_TRUE(consumer.terminal());
  EXPECT_EQ(*consumer.terminal(), SdmaRingStatus::Faulted);
  EXPECT_FALSE(consumer.in_flight());
}

TEST(SdmaRingConsumerTest, CacheMaintenanceFailureIsTerminalAndRetiresOnlyCommittedPrefixes) {
  for (bool committed_prefix : {false, true}) {
    SCOPED_TRACE(committed_prefix);
    RingConsumerFixture fixture;
    constexpr uint64_t kSource = 0x400;
    constexpr uint64_t kDestination = 0x500;
    for (uint32_t index = 0; index < 32; ++index)
      fixture.memory->store<uint8_t>(kSource + index, static_cast<uint8_t>(index + 1));
    const std::array<uint32_t, 7> copy = {
        1, 31, 0, static_cast<uint32_t>(kSource), 0, static_cast<uint32_t>(kDestination), 0};
    fixture.memory->store(RingConsumerFixture::kRing, copy);
    uint32_t maintenance_calls = 0;
    SdmaPacketCallbacks callbacks;
    callbacks.acquire_cache_maintenance =
        [&](SdmaCacheOperation) -> util::FailureOr<SdmaCacheLease> {
      ++maintenance_calls;
      if (!committed_prefix || maintenance_calls == 2)
        return util::Result::failure();
      return SdmaCacheLease{};
    };
    if (committed_prefix)
      fixture.memory->return_next_write(kDestination + 16, VmAccessOutcome::Unavailable);
    SdmaRingConsumer consumer = fixture.make_consumer(0, 64, std::move(callbacks));
    if (committed_prefix) {
      ASSERT_EQ(consumer.service(sizeof(copy)), SdmaRingStatus::Blocked);
      EXPECT_TRUE(consumer.in_flight());
      EXPECT_EQ(consumer.cursor(), 0u);
      EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer), 0u);
    }
    EXPECT_EQ(consumer.service(sizeof(copy)), SdmaRingStatus::Faulted);
    EXPECT_EQ(consumer.cursor(), committed_prefix ? sizeof(copy) : 0u);
    EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer),
              committed_prefix ? sizeof(copy) : 0u);
    EXPECT_FALSE(consumer.in_flight());
    ASSERT_TRUE(consumer.terminal());
    EXPECT_EQ(*consumer.terminal(), SdmaRingStatus::Faulted);
    EXPECT_EQ(consumer.service(sizeof(copy)), SdmaRingStatus::Faulted);
    EXPECT_EQ(maintenance_calls, committed_prefix ? 2u : 1u);
    for (uint32_t index = 0; index < 32; ++index)
      EXPECT_EQ(fixture.memory->load<uint8_t>(kDestination + index),
                committed_prefix && index < 16 ? static_cast<uint8_t>(index + 1) : 0);
  }
}

TEST(SdmaRingConsumerTest, PublishesRetiredPacketBeforeLatchingTerminalOutcome) {
  RingConsumerFixture fixture;
  constexpr uint64_t kBadSource = 0x2000;
  constexpr uint64_t kDestination = 0x300;
  const std::array<uint32_t, 7> copy = {
      1, 3, 0, static_cast<uint32_t>(kBadSource), 0, static_cast<uint32_t>(kDestination), 0};
  fixture.memory->store(RingConsumerFixture::kRing, copy);
  fixture.memory->return_next_atomic_store(RingConsumerFixture::kReadPointer,
                                           VmAccessOutcome::Unavailable);
  SdmaRingConsumer consumer = fixture.make_consumer(0);

  EXPECT_EQ(consumer.service(sizeof(copy)), SdmaRingStatus::Blocked);
  EXPECT_EQ(consumer.cursor(), sizeof(copy));
  EXPECT_FALSE(consumer.terminal());
  EXPECT_EQ(consumer.service(sizeof(copy)), SdmaRingStatus::Faulted);
  ASSERT_TRUE(consumer.terminal());
  EXPECT_EQ(*consumer.terminal(), SdmaRingStatus::Faulted);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer), sizeof(copy));
  EXPECT_FALSE(consumer.in_flight());
  EXPECT_EQ(consumer.service(sizeof(copy)), SdmaRingStatus::Faulted);
}

TEST(SdmaRingConsumerTest, DistinguishesMalformedConfigurationBoundsAndExecutionFromVmFaults) {
  RingConsumerFixture fixture;
  SdmaRingConsumer misaligned(fixture.vm,
                              {.address_space = fixture.address_space,
                               .ring_base = RingConsumerFixture::kRing,
                               .ring_bytes = 64,
                               .read_pointer_address = RingConsumerFixture::kReadPointer + 4,
                               .initial_cursor = 0},
                              SdmaPacketDialect::Gfx1250);
  EXPECT_EQ(misaligned.service(0), SdmaRingStatus::Malformed);
  EXPECT_EQ(misaligned.service(0), SdmaRingStatus::Malformed);

  SdmaRingConsumer bounds = fixture.make_consumer(0, 16);
  EXPECT_EQ(bounds.service(20), SdmaRingStatus::Malformed);

  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing, 0xff);
  SdmaRingConsumer bad_packet = fixture.make_consumer(0);
  EXPECT_EQ(bad_packet.service(4), SdmaRingStatus::Malformed);
  EXPECT_FALSE(bad_packet.in_flight());

  SdmaRingConsumer missing_ring(fixture.vm,
                                {.address_space = fixture.address_space,
                                 .ring_base = 0x2000,
                                 .ring_bytes = 64,
                                 .read_pointer_address = RingConsumerFixture::kReadPointer,
                                 .initial_cursor = 0},
                                SdmaPacketDialect::Gfx1250);
  EXPECT_EQ(missing_ring.service(4), SdmaRingStatus::Faulted);
}

TEST(SdmaRingConsumerTest, RootReplacementDoesNotChangeAnInFlightBatchSnapshot) {
  RingConsumerFixture fixture;
  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing, 0);
  fixture.memory->return_next_atomic_store(RingConsumerFixture::kReadPointer,
                                           VmAccessOutcome::Unavailable);
  SdmaRingConsumer consumer = fixture.make_consumer(0);

  EXPECT_EQ(consumer.service(4), SdmaRingStatus::Blocked);
  std::shared_ptr<QueueMemory> replacement = std::make_shared<QueueMemory>();
  replacement->store<uint64_t>(RingConsumerFixture::kReadPointer, 99);
  ASSERT_TRUE(fixture.vm.replace_translated(fixture.address_space, replacement, replacement));

  EXPECT_EQ(consumer.service(4), SdmaRingStatus::Idle);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer), 4u);
  EXPECT_EQ(replacement->load<uint64_t>(RingConsumerFixture::kReadPointer), 99u);
}

TEST(SdmaRingConsumerTest, SemanticContinuationUsesTheConsumerOwnedBatchSnapshot) {
  RingConsumerFixture fixture;
  constexpr uint64_t kPoll = 0x300;
  const std::array<uint32_t, 8> poll = {8u | (5u << 8) | (3u << 28),
                                        static_cast<uint32_t>(kPoll),
                                        0,
                                        1,
                                        0,
                                        UINT32_MAX,
                                        UINT32_MAX,
                                        0};
  fixture.memory->store(RingConsumerFixture::kRing, poll);
  fixture.memory->store<uint64_t>(kPoll, 0);
  SdmaRingConsumer consumer = fixture.make_consumer(0);

  EXPECT_EQ(consumer.service(sizeof(poll)), SdmaRingStatus::Blocked);
  EXPECT_TRUE(consumer.in_flight());
  const std::size_t ring_read_bytes =
      fixture.memory->read_bytes(RingConsumerFixture::kRing, RingConsumerFixture::kRing + 64);

  std::shared_ptr<QueueMemory> replacement = std::make_shared<QueueMemory>();
  replacement->store<uint64_t>(RingConsumerFixture::kReadPointer, 99);
  replacement->store<uint64_t>(kPoll, 1);
  ASSERT_TRUE(fixture.vm.replace_translated(fixture.address_space, replacement, replacement));
  fixture.memory->store<uint64_t>(kPoll, 1);

  EXPECT_EQ(consumer.service(sizeof(poll)), SdmaRingStatus::Idle);
  EXPECT_EQ(fixture.memory->read_bytes(RingConsumerFixture::kRing, RingConsumerFixture::kRing + 64),
            ring_read_bytes);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RingConsumerFixture::kReadPointer), sizeof(poll));
  EXPECT_EQ(replacement->load<uint64_t>(RingConsumerFixture::kReadPointer), 99u);
  EXPECT_FALSE(consumer.in_flight());
}

TEST(SdmaRingConsumerTest, ReconfigureIsIdleOnlyAndResetDiscardsRetryAndTerminalState) {
  RingConsumerFixture fixture;
  fixture.memory->store<uint32_t>(RingConsumerFixture::kRing, 0);
  fixture.memory->return_next_read(RingConsumerFixture::kRing, VmAccessOutcome::Unavailable);
  SdmaRingConsumer consumer = fixture.make_consumer(0);
  const SdmaRingConfig replacement{.address_space = fixture.address_space,
                                   .ring_base = RingConsumerFixture::kRing + 0x40,
                                   .ring_bytes = 32,
                                   .read_pointer_address = RingConsumerFixture::kReadPointer,
                                   .initial_cursor = 8};

  EXPECT_EQ(consumer.service(4), SdmaRingStatus::Blocked);
  EXPECT_TRUE(consumer.in_flight());
  EXPECT_FALSE(consumer.reconfigure(replacement));
  EXPECT_EQ(consumer.config().ring_base, RingConsumerFixture::kRing);

  consumer.reset();
  EXPECT_FALSE(consumer.in_flight());
  EXPECT_TRUE(consumer.reconfigure(replacement));
  EXPECT_EQ(consumer.cursor(), 8u);
  fixture.memory->store<uint32_t>(replacement.ring_base + 8, 0xff);
  EXPECT_EQ(consumer.service(12), SdmaRingStatus::Malformed);
  ASSERT_TRUE(consumer.terminal());
  consumer.reset();
  EXPECT_FALSE(consumer.terminal());
  EXPECT_FALSE(consumer.in_flight());
}

} // namespace
} // namespace rocjitsu::amdgpu
