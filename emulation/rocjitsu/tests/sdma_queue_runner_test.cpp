// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/sdma_queue_runner.h"

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
    if (const auto outcome = next(read_outcomes_, address); outcome != VmAccessOutcome::Complete)
      return outcome;
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
    ++atomic_load_attempts_[address];
    if (const auto outcome = next(atomic_load_outcomes_, address);
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
    if (const auto outcome = next(atomic_store_outcomes_, address);
        outcome != VmAccessOutcome::Complete)
      return outcome;
    if ((width != 4 && width != 8) || address > bytes_.size() || width > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::memcpy(bytes_.data() + address, &value, width);
    return VmAccessOutcome::Complete;
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
  void return_next_atomic_load(uint64_t address, VmAccessOutcome outcome) {
    atomic_load_outcomes_[address].push_back(outcome);
  }
  void return_next_atomic_store(uint64_t address, VmAccessOutcome outcome) {
    atomic_store_outcomes_[address].push_back(outcome);
  }
  uint32_t read_attempts(uint64_t address) const { return attempts(read_attempts_, address); }
  uint32_t atomic_load_attempts(uint64_t address) const {
    return attempts(atomic_load_attempts_, address);
  }
  uint32_t atomic_store_attempts(uint64_t address) const {
    return attempts(atomic_store_attempts_, address);
  }

private:
  static VmAccessOutcome next(std::map<uint64_t, std::deque<VmAccessOutcome>> &outcomes,
                              uint64_t address) {
    auto found = outcomes.find(address);
    if (found == outcomes.end() || found->second.empty())
      return VmAccessOutcome::Complete;
    const VmAccessOutcome outcome = found->second.front();
    found->second.pop_front();
    return outcome;
  }

  static uint32_t attempts(const std::map<uint64_t, uint32_t> &values, uint64_t address) {
    const auto found = values.find(address);
    return found == values.end() ? 0 : found->second;
  }

  std::vector<std::byte> bytes_;
  std::map<uint64_t, std::deque<VmAccessOutcome>> read_outcomes_;
  std::map<uint64_t, std::deque<VmAccessOutcome>> atomic_load_outcomes_;
  std::map<uint64_t, std::deque<VmAccessOutcome>> atomic_store_outcomes_;
  std::map<uint64_t, uint32_t> read_attempts_;
  std::map<uint64_t, uint32_t> atomic_load_attempts_;
  std::map<uint64_t, uint32_t> atomic_store_attempts_;
};

struct RunnerFixture {
  static constexpr uint64_t kRing = 0x100;
  static constexpr uint64_t kReadPointer = 0x80;

  RunnerFixture() : memory(std::make_shared<QueueMemory>()) {
    address_space = vm.register_translated(7, memory, memory);
  }

  SdmaQueueRunner make_runner(std::optional<uint64_t> initial = uint64_t{0},
                              uint64_t ring_bytes = 64, SdmaExecutorCallbacks callbacks = {}) {
    return SdmaQueueRunner(vm,
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

TEST(SdmaQueueRunnerTest, InitialCursorOverridesMemoryAndAbsentCursorUsesAtomicLoad) {
  RunnerFixture fixture;
  fixture.memory->store<uint64_t>(RunnerFixture::kReadPointer, 0);
  fixture.memory->store<uint32_t>(RunnerFixture::kRing + 4, 0);
  auto supplied = fixture.make_runner(4);

  EXPECT_EQ(supplied.service(8), SdmaQueueServiceOutcome::Drained);
  EXPECT_EQ(supplied.cursor(), 8u);
  EXPECT_EQ(fixture.memory->atomic_load_attempts(RunnerFixture::kReadPointer), 0u);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RunnerFixture::kReadPointer), 8u);

  fixture.memory->store<uint64_t>(RunnerFixture::kReadPointer, 8);
  fixture.memory->store<uint32_t>(RunnerFixture::kRing + 8, 0);
  auto loaded = fixture.make_runner(std::nullopt);
  EXPECT_EQ(loaded.service(12), SdmaQueueServiceOutcome::Drained);
  EXPECT_EQ(loaded.cursor(), 12u);
  EXPECT_EQ(fixture.memory->atomic_load_attempts(RunnerFixture::kReadPointer), 1u);
}

TEST(SdmaQueueRunnerTest, FetchesAcrossRingWrapWithoutReplayingCompletedSegment) {
  RunnerFixture fixture;
  fixture.memory->store<uint32_t>(RunnerFixture::kRing + 12, 0);
  fixture.memory->store<uint32_t>(RunnerFixture::kRing, 0);
  fixture.memory->return_next_read(RunnerFixture::kRing, VmAccessOutcome::Unavailable);
  auto runner = fixture.make_runner(12, 16);

  EXPECT_EQ(runner.service(20), SdmaQueueServiceOutcome::Unavailable);
  EXPECT_EQ(fixture.memory->read_attempts(RunnerFixture::kRing + 12), 1u);
  EXPECT_EQ(runner.service(20), SdmaQueueServiceOutcome::Drained);
  EXPECT_EQ(fixture.memory->read_attempts(RunnerFixture::kRing + 12), 1u);
  EXPECT_EQ(fixture.memory->read_attempts(RunnerFixture::kRing), 2u);
  EXPECT_EQ(runner.cursor(), 20u);
}

TEST(SdmaQueueRunnerTest, RetriesUnavailableInitialLoadFetchExecutorAndPublication) {
  RunnerFixture fixture;
  fixture.memory->store<uint64_t>(RunnerFixture::kReadPointer, 0);
  fixture.memory->return_next_atomic_load(RunnerFixture::kReadPointer,
                                          VmAccessOutcome::Unavailable);
  fixture.memory->store<uint32_t>(RunnerFixture::kRing, 0);
  fixture.memory->return_next_read(RunnerFixture::kRing, VmAccessOutcome::Unavailable);
  fixture.memory->return_next_atomic_store(RunnerFixture::kReadPointer,
                                           VmAccessOutcome::Unavailable);
  auto runner = fixture.make_runner(std::nullopt);

  EXPECT_EQ(runner.service(4), SdmaQueueServiceOutcome::Unavailable);
  EXPECT_EQ(runner.service(4), SdmaQueueServiceOutcome::Unavailable);
  EXPECT_EQ(runner.service(4), SdmaQueueServiceOutcome::Unavailable);
  EXPECT_EQ(runner.cursor(), 4u) << "retired cursor is visible while publication retries";
  EXPECT_EQ(runner.service(4), SdmaQueueServiceOutcome::Drained);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RunnerFixture::kReadPointer), 4u);

  constexpr uint64_t kPoll = 0x300;
  const std::array<uint32_t, 8> poll = {8u | (5u << 8) | (3u << 28),
                                        static_cast<uint32_t>(kPoll),
                                        0,
                                        1,
                                        0,
                                        UINT32_MAX,
                                        UINT32_MAX,
                                        0};
  fixture.memory->store(RunnerFixture::kRing + 16, poll);
  fixture.memory->store<uint64_t>(kPoll, 1);
  fixture.memory->return_next_atomic_load(kPoll, VmAccessOutcome::Unavailable);
  auto executor_retry = fixture.make_runner(16);
  EXPECT_EQ(executor_retry.service(48), SdmaQueueServiceOutcome::Unavailable);
  EXPECT_EQ(executor_retry.cursor(), 16u);
  EXPECT_EQ(executor_retry.service(48), SdmaQueueServiceOutcome::Drained);
  EXPECT_EQ(executor_retry.cursor(), 48u);
}

TEST(SdmaQueueRunnerTest, PublishesRetiredPacketBeforeLatchingTerminalOutcome) {
  RunnerFixture fixture;
  constexpr uint64_t kBadSource = 0x2000;
  constexpr uint64_t kDestination = 0x300;
  const std::array<uint32_t, 7> copy = {
      1, 3, 0, static_cast<uint32_t>(kBadSource), 0, static_cast<uint32_t>(kDestination), 0};
  fixture.memory->store(RunnerFixture::kRing, copy);
  fixture.memory->return_next_atomic_store(RunnerFixture::kReadPointer,
                                           VmAccessOutcome::Unavailable);
  auto runner = fixture.make_runner(0);

  EXPECT_EQ(runner.service(sizeof(copy)), SdmaQueueServiceOutcome::Unavailable);
  EXPECT_EQ(runner.cursor(), sizeof(copy));
  EXPECT_FALSE(runner.terminal());
  EXPECT_EQ(runner.service(sizeof(copy)), SdmaQueueServiceOutcome::Faulted);
  ASSERT_TRUE(runner.terminal());
  EXPECT_EQ(*runner.terminal(), SdmaQueueServiceOutcome::Faulted);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RunnerFixture::kReadPointer), sizeof(copy));
  EXPECT_EQ(runner.service(sizeof(copy)), SdmaQueueServiceOutcome::Faulted);
}

TEST(SdmaQueueRunnerTest, DistinguishesMalformedConfigurationBoundsAndExecutionFromVmFaults) {
  RunnerFixture fixture;
  auto misaligned = SdmaQueueRunner(fixture.vm,
                                    {.address_space = fixture.address_space,
                                     .ring_base = RunnerFixture::kRing,
                                     .ring_bytes = 64,
                                     .read_pointer_address = RunnerFixture::kReadPointer + 4,
                                     .initial_cursor = 0},
                                    SdmaPacketDialect::Gfx1250);
  EXPECT_EQ(misaligned.service(0), SdmaQueueServiceOutcome::Malformed);
  EXPECT_EQ(misaligned.service(0), SdmaQueueServiceOutcome::Malformed);

  auto bounds = fixture.make_runner(0, 16);
  EXPECT_EQ(bounds.service(20), SdmaQueueServiceOutcome::Malformed);

  fixture.memory->store<uint32_t>(RunnerFixture::kRing, 0xff);
  auto bad_packet = fixture.make_runner(0);
  EXPECT_EQ(bad_packet.service(4), SdmaQueueServiceOutcome::Malformed);

  auto missing_ring = SdmaQueueRunner(fixture.vm,
                                      {.address_space = fixture.address_space,
                                       .ring_base = 0x2000,
                                       .ring_bytes = 64,
                                       .read_pointer_address = RunnerFixture::kReadPointer,
                                       .initial_cursor = 0},
                                      SdmaPacketDialect::Gfx1250);
  EXPECT_EQ(missing_ring.service(4), SdmaQueueServiceOutcome::Faulted);
}

TEST(SdmaQueueRunnerTest, RootReplacementDoesNotChangeAnInFlightBatchSnapshot) {
  RunnerFixture fixture;
  fixture.memory->store<uint32_t>(RunnerFixture::kRing, 0);
  fixture.memory->return_next_atomic_store(RunnerFixture::kReadPointer,
                                           VmAccessOutcome::Unavailable);
  auto runner = fixture.make_runner(0);

  EXPECT_EQ(runner.service(4), SdmaQueueServiceOutcome::Unavailable);
  auto replacement = std::make_shared<QueueMemory>();
  replacement->store<uint64_t>(RunnerFixture::kReadPointer, 99);
  ASSERT_TRUE(fixture.vm.replace_translated(fixture.address_space, replacement, replacement));

  EXPECT_EQ(runner.service(4), SdmaQueueServiceOutcome::Drained);
  EXPECT_EQ(fixture.memory->load<uint64_t>(RunnerFixture::kReadPointer), 4u);
  EXPECT_EQ(replacement->load<uint64_t>(RunnerFixture::kReadPointer), 99u);
}

TEST(SdmaQueueRunnerTest, ReconfigureIsIdleOnlyAndResetDiscardsRetryAndTerminalState) {
  RunnerFixture fixture;
  fixture.memory->store<uint32_t>(RunnerFixture::kRing, 0);
  fixture.memory->return_next_read(RunnerFixture::kRing, VmAccessOutcome::Unavailable);
  auto runner = fixture.make_runner(0);
  const SdmaQueueRunnerConfig replacement{.address_space = fixture.address_space,
                                          .ring_base = RunnerFixture::kRing + 0x40,
                                          .ring_bytes = 32,
                                          .read_pointer_address = RunnerFixture::kReadPointer,
                                          .initial_cursor = 8};

  EXPECT_EQ(runner.service(4), SdmaQueueServiceOutcome::Unavailable);
  EXPECT_TRUE(runner.in_flight());
  EXPECT_FALSE(runner.reconfigure(replacement));
  EXPECT_EQ(runner.config().ring_base, RunnerFixture::kRing);

  runner.reset();
  EXPECT_FALSE(runner.in_flight());
  EXPECT_TRUE(runner.reconfigure(replacement));
  EXPECT_EQ(runner.cursor(), 8u);
  fixture.memory->store<uint32_t>(replacement.ring_base + 8, 0xff);
  EXPECT_EQ(runner.service(12), SdmaQueueServiceOutcome::Malformed);
  ASSERT_TRUE(runner.terminal());
  runner.reset();
  EXPECT_FALSE(runner.terminal());
  EXPECT_FALSE(runner.in_flight());
}

} // namespace
} // namespace rocjitsu::amdgpu
