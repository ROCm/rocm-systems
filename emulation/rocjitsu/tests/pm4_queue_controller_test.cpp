// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/pm4_queue_controller.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

class Pm4QueueMemory final : public AddressSpaceTranslator, public PhysicalMemoryAccess {
public:
  Pm4QueueMemory() : bytes_(0x1000) {}

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
        width > bytes_.size() - address) {
      return {.outcome = VmAccessOutcome::Malformed};
    }
    uint64_t value = 0;
    std::memcpy(&value, bytes_.data() + address, width);
    return {.outcome = VmAccessOutcome::Complete, .value = value};
  }

  VmAccessOutcome atomic_store(VmMemoryDomain, uint64_t address, uint32_t width,
                               uint64_t value) override {
    if (unavailable_store_ && address == *unavailable_store_)
      return VmAccessOutcome::Unavailable;
    if (faulted_store_ && address == *faulted_store_)
      return VmAccessOutcome::Faulted;
    if ((width != 4 && width != 8) || address % width != 0 || address > bytes_.size() ||
        width > bytes_.size() - address) {
      return VmAccessOutcome::Malformed;
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
  void make_store_unavailable(uint64_t address) { unavailable_store_ = address; }
  void make_store_faulted(uint64_t address) { faulted_store_ = address; }
  void make_available() {
    unavailable_read_.reset();
    unavailable_store_.reset();
    faulted_store_.reset();
  }

private:
  std::vector<std::byte> bytes_;
  std::optional<uint64_t> unavailable_read_;
  std::optional<uint64_t> unavailable_store_;
  std::optional<uint64_t> faulted_store_;
};

class Pm4QueueControllerTest : public ::testing::Test {
protected:
  void SetUp() override {
    memory = std::make_shared<Pm4QueueMemory>();
    address_space = gpu_vm.register_translated(1, memory, memory);
    ASSERT_TRUE(address_space);
    memory->store<uint32_t>(kReadPointer, 0);
  }

  Pm4QueueController::RegistrationId attach(Pm4PacketCallbacks callbacks, uint64_t ring = kRing,
                                            uint64_t read_pointer = kReadPointer,
                                            uint32_t ring_bytes = kRingBytes) {
    return controller.attach({.address_space = address_space,
                              .ring_base = ring,
                              .ring_size_bytes = ring_bytes,
                              .consumer_pointer_address = read_pointer,
                              .initial_consumer_cursor = std::nullopt,
                              .packet_callbacks = std::move(callbacks)});
  }

  static constexpr uint64_t kRing = 0x100;
  static constexpr uint32_t kRingBytes = 64;
  static constexpr uint64_t kReadPointer = 0x200;

  GpuVm gpu_vm;
  std::shared_ptr<Pm4QueueMemory> memory;
  AddressSpaceHandle address_space;
  Pm4QueueController controller{gpu_vm};
};

TEST_F(Pm4QueueControllerTest, SubmissionDefersEffectsUntilTheCpServicesTheQueue) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  uint32_t writes = 0;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);

  EXPECT_EQ(controller.notify(registration, 3), QueueSubmissionStatus::Accepted);
  EXPECT_EQ(writes, 0u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);

  EXPECT_FALSE(controller.service());
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
}

TEST_F(Pm4QueueControllerTest, BoundsEachQueueServiceTurn) {
  constexpr uint64_t kLargeRing = 0x400;
  constexpr uint64_t kLargeReadPointer = 0xe00;
  constexpr uint32_t kPacketCount = 257;
  constexpr uint32_t kLargeRingBytes = 512 * sizeof(uint32_t);
  for (uint32_t packet = 0; packet < kPacketCount; ++packet)
    memory->store<uint32_t>(kLargeRing + packet * sizeof(uint32_t), 0xffff1000);
  memory->store<uint32_t>(kLargeReadPointer, 0);

  const auto registration = attach({}, kLargeRing, kLargeReadPointer, kLargeRingBytes);
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(controller.notify(registration, kPacketCount), QueueSubmissionStatus::Accepted);

  EXPECT_TRUE(controller.service());
  EXPECT_EQ(memory->load<uint32_t>(kLargeReadPointer), 256u);

  EXPECT_FALSE(controller.service());
  EXPECT_EQ(memory->load<uint32_t>(kLargeReadPointer), kPacketCount);
}

TEST_F(Pm4QueueControllerTest, RetainsItsAddressSpaceUntilDetach) {
  const Pm4QueueController::RegistrationId registration = attach({});
  ASSERT_NE(registration, 0u);

  EXPECT_FALSE(gpu_vm.unregister_address_space(address_space));
  EXPECT_TRUE(controller.detach(registration));
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST_F(Pm4QueueControllerTest, CursorPublicationRetryDoesNotReplayThePacketEffect) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  uint32_t writes = 0;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(controller.notify(registration, 3), QueueSubmissionStatus::Accepted);
  memory->make_store_unavailable(kReadPointer);

  EXPECT_TRUE(controller.service());
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);

  memory->make_available();
  EXPECT_FALSE(controller.service());
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
}

TEST_F(Pm4QueueControllerTest, RootReplacementDoesNotMovePendingCursorPublication) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  uint32_t writes = 0;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(controller.notify(registration, 3), QueueSubmissionStatus::Accepted);
  memory->make_store_unavailable(kReadPointer);

  EXPECT_TRUE(controller.service());
  EXPECT_EQ(writes, 1u);

  auto replacement = std::make_shared<Pm4QueueMemory>();
  replacement->store<uint32_t>(kReadPointer, 99);
  ASSERT_TRUE(gpu_vm.replace_translated(address_space, replacement, replacement));
  memory->make_available();

  EXPECT_FALSE(controller.service());
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
  EXPECT_EQ(replacement->load<uint32_t>(kReadPointer), 99u);
}

TEST_F(Pm4QueueControllerTest, RootReplacementDoesNotMoveBlockedPacketFetch) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  uint32_t written_value = 0;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t value) {
    written_value = value;
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(controller.notify(registration, 3), QueueSubmissionStatus::Accepted);
  memory->make_read_unavailable(kRing);

  EXPECT_TRUE(controller.service());
  EXPECT_EQ(written_value, 0u);

  auto replacement = std::make_shared<Pm4QueueMemory>();
  replacement->store<uint32_t>(kRing, 0xc0017900);
  replacement->store<uint32_t>(kRing + 4, 0x40);
  replacement->store<uint32_t>(kRing + 8, 0x12345678);
  replacement->store<uint32_t>(kReadPointer, 99);
  ASSERT_TRUE(gpu_vm.replace_translated(address_space, replacement, replacement));
  memory->make_available();

  EXPECT_FALSE(controller.service());
  EXPECT_EQ(written_value, 0xdeadbeefu);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
  EXPECT_EQ(replacement->load<uint32_t>(kReadPointer), 99u);
}

TEST_F(Pm4QueueControllerTest, LaterDoorbellUsesFreshSnapshotAfterBlockedTransaction) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0x11111111);
  std::vector<uint32_t> values;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t value) {
    values.push_back(value);
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(controller.notify(registration, 3), QueueSubmissionStatus::Accepted);
  memory->make_store_unavailable(kReadPointer);

  EXPECT_TRUE(controller.service());
  ASSERT_EQ(values.size(), 1u);
  EXPECT_EQ(values[0], 0x11111111u);

  auto replacement = std::make_shared<Pm4QueueMemory>();
  replacement->store<uint32_t>(kRing + 12, 0xc0017900);
  replacement->store<uint32_t>(kRing + 16, 0x40);
  replacement->store<uint32_t>(kRing + 20, 0x22222222);
  replacement->store<uint32_t>(kReadPointer, 3);
  ASSERT_TRUE(gpu_vm.replace_translated(address_space, replacement, replacement));
  ASSERT_EQ(controller.notify(registration, 6), QueueSubmissionStatus::Accepted);
  memory->make_available();

  EXPECT_TRUE(controller.service());
  ASSERT_EQ(values.size(), 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
  EXPECT_EQ(replacement->load<uint32_t>(kReadPointer), 3u);

  EXPECT_FALSE(controller.service());
  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(values[1], 0x22222222u);
  EXPECT_EQ(replacement->load<uint32_t>(kReadPointer), 6u);
}

TEST_F(Pm4QueueControllerTest, PublishesEarlierProgressBeforeRetryingALaterFetch) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  memory->store<uint32_t>(kRing + 12, 0xffff1000);
  uint32_t writes = 0;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(controller.notify(registration, 4), QueueSubmissionStatus::Accepted);
  memory->make_read_unavailable(kRing + 12);

  EXPECT_TRUE(controller.service());
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);

  memory->make_available();
  EXPECT_FALSE(controller.service());
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 4u);
}

TEST_F(Pm4QueueControllerTest, RejectsInvalidRingsAtRegistration) {
  EXPECT_EQ(controller.attach({.address_space = address_space,
                               .ring_base = kRing + 1,
                               .ring_size_bytes = kRingBytes,
                               .consumer_pointer_address = kReadPointer,
                               .initial_consumer_cursor = std::nullopt,
                               .packet_callbacks = {}}),
            0u);
  EXPECT_EQ(controller.attach({.address_space = address_space,
                               .ring_base = kRing,
                               .ring_size_bytes = kRingBytes - 1,
                               .consumer_pointer_address = kReadPointer,
                               .initial_consumer_cursor = std::nullopt,
                               .packet_callbacks = {}}),
            0u);
}

TEST_F(Pm4QueueControllerTest, BlockedQueuesRetryIndependently) {
  constexpr uint64_t kOtherRing = 0x300;
  constexpr uint64_t kOtherReadPointer = 0x380;
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 1);
  memory->store<uint32_t>(kOtherRing, 0xc0017900);
  memory->store<uint32_t>(kOtherRing + 4, 0x40);
  memory->store<uint32_t>(kOtherRing + 8, 2);
  memory->store<uint32_t>(kOtherReadPointer, 0);

  bool first_blocked = true;
  bool second_blocked = true;
  uint32_t first_attempts = 0;
  uint32_t second_attempts = 0;
  const auto first = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++first_attempts;
    return first_blocked ? Pm4RegisterWriteStatus::Blocked : Pm4RegisterWriteStatus::Complete;
  }});
  const auto second = attach({.write_uconfig_register =
                                  [&](uint64_t, uint32_t) {
                                    ++second_attempts;
                                    return second_blocked ? Pm4RegisterWriteStatus::Blocked
                                                          : Pm4RegisterWriteStatus::Complete;
                                  }},
                             kOtherRing, kOtherReadPointer);
  ASSERT_NE(first, 0u);
  ASSERT_NE(second, 0u);
  ASSERT_EQ(controller.notify(first, 3), QueueSubmissionStatus::Accepted);
  ASSERT_EQ(controller.notify(second, 3), QueueSubmissionStatus::Accepted);

  EXPECT_TRUE(controller.service());
  EXPECT_EQ(first_attempts, 1u);
  EXPECT_EQ(second_attempts, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);
  EXPECT_EQ(memory->load<uint32_t>(kOtherReadPointer), 0u);

  first_blocked = false;
  second_blocked = false;
  EXPECT_FALSE(controller.service());
  EXPECT_EQ(first_attempts, 2u);
  EXPECT_EQ(second_attempts, 2u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
  EXPECT_EQ(memory->load<uint32_t>(kOtherReadPointer), 3u);
}

TEST_F(Pm4QueueControllerTest, DetachWaitsForActiveServiceWithoutHoldingControllerMutex) {
  using namespace std::chrono_literals;
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  std::promise<void> callback_entered;
  std::promise<void> release_callback;
  std::shared_future<void> release = release_callback.get_future().share();
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    callback_entered.set_value();
    release.wait();
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(controller.notify(registration, 3), QueueSubmissionStatus::Accepted);

  auto service = std::async(std::launch::async, [&] { return controller.service(); });
  ASSERT_EQ(callback_entered.get_future().wait_for(1s), std::future_status::ready);
  auto detach = std::async(std::launch::async, [&] { return controller.detach(registration); });

  EXPECT_EQ(detach.wait_for(20ms), std::future_status::timeout);
  EXPECT_EQ(controller.active_queues(), 1u);
  release_callback.set_value();
  EXPECT_FALSE(service.get());
  EXPECT_TRUE(detach.get());
}

TEST_F(Pm4QueueControllerTest, ReconfigureWaitsForActiveService) {
  using namespace std::chrono_literals;
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  std::promise<void> callback_entered;
  std::promise<void> release_callback;
  std::shared_future<void> release = release_callback.get_future().share();
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    callback_entered.set_value();
    release.wait();
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(controller.notify(registration, 3), QueueSubmissionStatus::Accepted);

  auto service = std::async(std::launch::async, [&] { return controller.service(); });
  ASSERT_EQ(callback_entered.get_future().wait_for(1s), std::future_status::ready);
  auto reconfigure = std::async(std::launch::async, [&] {
    return controller.update(
        registration,
        {.ring_base_address = kRing, .ring_size_bytes = kRingBytes, .scheduling_percentage = 100});
  });

  EXPECT_EQ(reconfigure.wait_for(20ms), std::future_status::timeout);
  release_callback.set_value();
  EXPECT_FALSE(service.get());
  EXPECT_EQ(reconfigure.get(), QueueReconfigureStatus::Applied);
}

TEST_F(Pm4QueueControllerTest, GracefulDetachWaitsForPendingCursorPublication) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  uint32_t writes = 0;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(controller.notify(registration, 3), QueueSubmissionStatus::Accepted);
  memory->make_store_unavailable(kReadPointer);

  EXPECT_TRUE(controller.service());
  EXPECT_EQ(controller.prepare_detach(registration), QueuePrepareCloseStatus::Busy);
  EXPECT_EQ(controller.active_queues(), 1u);
  EXPECT_EQ(writes, 1u);

  memory->make_available();
  EXPECT_FALSE(controller.service());
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
  EXPECT_EQ(controller.prepare_detach(registration), QueuePrepareCloseStatus::Ready);
  EXPECT_EQ(controller.active_queues(), 0u);
}

TEST_F(Pm4QueueControllerTest, GracefulDetachCancelsBlockedPacketBeforeAnyEffect) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  uint32_t attempts = 0;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++attempts;
    return Pm4RegisterWriteStatus::Blocked;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(controller.notify(registration, 3), QueueSubmissionStatus::Accepted);

  EXPECT_TRUE(controller.service());
  EXPECT_EQ(attempts, 1u);
  EXPECT_EQ(controller.prepare_detach(registration), QueuePrepareCloseStatus::Ready);
  EXPECT_EQ(controller.active_queues(), 0u);
}

TEST_F(Pm4QueueControllerTest, GracefulDetachReportsFaultedPublicationWithoutRetryingForever) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  const auto registration = attach({.write_uconfig_register = [](uint64_t, uint32_t) {
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(controller.notify(registration, 3), QueueSubmissionStatus::Accepted);
  memory->make_store_faulted(kReadPointer);

  EXPECT_FALSE(controller.service());
  EXPECT_EQ(controller.prepare_detach(registration), QueuePrepareCloseStatus::Faulted);
  EXPECT_EQ(controller.active_queues(), 1u);
  EXPECT_TRUE(controller.detach(registration));
  EXPECT_EQ(controller.active_queues(), 0u);
}

TEST_F(Pm4QueueControllerTest, TerminalQueueCanBeDisabledAndReconfigured) {
  constexpr uint64_t kReplacementRing = 0x300;
  memory->store<uint32_t>(kRing, 0xffff1000);
  const auto registration = attach({});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(controller.notify(registration, 1), QueueSubmissionStatus::Accepted);
  EXPECT_FALSE(controller.service());

  EXPECT_EQ(
      controller.update(registration,
                        {.ring_base_address = 0, .ring_size_bytes = 0, .scheduling_percentage = 0}),
      QueueReconfigureStatus::Disabled);
  memory->store<uint32_t>(kReplacementRing, 0xffff1000);
  EXPECT_EQ(controller.update(registration, {.ring_base_address = kReplacementRing,
                                             .ring_size_bytes = kRingBytes,
                                             .scheduling_percentage = 100}),
            QueueReconfigureStatus::Applied);
  EXPECT_EQ(controller.notify(registration, 1), QueueSubmissionStatus::Accepted);
  EXPECT_FALSE(controller.service());
  EXPECT_EQ(controller.prepare_detach(registration), QueuePrepareCloseStatus::Ready);
}

TEST(Pm4QueueBindingTest, GracefulRegistryRemovalKeepsHandleUntilPublicationCompletes) {
  constexpr uint64_t kRing = 0x100;
  constexpr uint64_t kReadPointer = 0x200;
  auto memory = std::make_shared<Pm4QueueMemory>();
  GpuVm gpu_vm;
  const AddressSpaceHandle address_space = gpu_vm.register_translated(1, memory, memory);
  ASSERT_TRUE(address_space);
  CommandProcessor command_processor("cp");
  command_processor.set_gpu_vm(&gpu_vm);
  GpuQueueRegistry registry(gpu_vm);
  uint32_t writes = 0;
  const std::shared_ptr<QueueBindingFactory> binding_factory =
      command_processor.make_pm4_queue_binding_factory(
          {.write_uconfig_register = [&](uint64_t, uint32_t) {
            ++writes;
            return Pm4RegisterWriteStatus::Complete;
          }});
  const QueueHandle queue = registry.register_queue({
      .identity = {.address_space = address_space, .process_id = 1, .queue_id = 7},
      .ring = {.base_address = kRing, .size_bytes = 64, .consumer_pointer_address = kReadPointer},
      .doorbell = {},
      .binding_factory = binding_factory,
      .type = QueueType::Compute,
      .packet_format = QueuePacketFormat::Pm4,
  });
  ASSERT_TRUE(queue);
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  memory->make_store_unavailable(kReadPointer);

  const QueueSubmissionResult blocked = registry.submit_producer(queue, 3);
  ASSERT_TRUE(blocked.found);
  EXPECT_EQ(blocked.status, QueueSubmissionStatus::Retry);
  EXPECT_EQ(writes, 1u);
  EXPECT_FALSE(registry.unregister_queue(queue));
  EXPECT_TRUE(registry.contains(queue));

  memory->make_available();
  const QueueSubmissionResult completed = registry.submit_producer(queue, 3);
  ASSERT_TRUE(completed.found);
  EXPECT_EQ(completed.status, QueueSubmissionStatus::Accepted);
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
  EXPECT_TRUE(registry.unregister_queue(queue));
  EXPECT_FALSE(registry.contains(queue));
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

} // namespace
} // namespace rocjitsu::amdgpu
