// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/mes_engine.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

namespace {

class RejectingComputeBindings final : public rocjitsu::amdgpu::ComputeQueueBindingProvider {
public:
  std::optional<rocjitsu::amdgpu::ComputeQueueBindingPlan> make_aql(uint32_t) override {
    return std::nullopt;
  }

  std::optional<rocjitsu::amdgpu::ComputeQueueBindingPlan>
  make_pm4(uint32_t, rocjitsu::amdgpu::Pm4PacketCallbacks) override {
    return std::nullopt;
  }
};

class MesTestMemory final : public rocjitsu::amdgpu::PhysicalMemoryAccess {
public:
  rocjitsu::amdgpu::VmAccessOutcome read(rocjitsu::amdgpu::VmMemoryDomain, uint64_t address,
                                         std::span<std::byte> bytes) override {
    if (!contains(address, bytes.size()))
      return rocjitsu::amdgpu::VmAccessOutcome::Faulted;
    std::memcpy(bytes.data(), bytes_.data() + address, bytes.size());
    return rocjitsu::amdgpu::VmAccessOutcome::Complete;
  }

  rocjitsu::amdgpu::VmAccessOutcome write(rocjitsu::amdgpu::VmMemoryDomain, uint64_t address,
                                          std::span<const std::byte> bytes) override {
    if (unavailable_write_ && *unavailable_write_ == address) {
      unavailable_write_.reset();
      return rocjitsu::amdgpu::VmAccessOutcome::Unavailable;
    }
    if (!contains(address, bytes.size()))
      return rocjitsu::amdgpu::VmAccessOutcome::Faulted;
    std::memcpy(bytes_.data() + address, bytes.data(), bytes.size());
    return rocjitsu::amdgpu::VmAccessOutcome::Complete;
  }

  template <typename T> void store(uint64_t address, T value) {
    const auto bytes = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
    ASSERT_TRUE(contains(address, bytes.size()));
    std::memcpy(bytes_.data() + address, bytes.data(), bytes.size());
  }

  template <typename T> T load(uint64_t address) const {
    std::array<std::byte, sizeof(T)> bytes{};
    EXPECT_TRUE(contains(address, bytes.size()));
    std::memcpy(bytes.data(), bytes_.data() + address, bytes.size());
    return std::bit_cast<T>(bytes);
  }

  void make_next_write_unavailable(uint64_t address) { unavailable_write_ = address; }

private:
  bool contains(uint64_t address, std::size_t size) const {
    return address <= bytes_.size() && size <= bytes_.size() - address;
  }

  std::vector<std::byte> bytes_ = std::vector<std::byte>(0x4000);
  std::optional<uint64_t> unavailable_write_;
};

class MesEngineStateTest : public ::testing::Test {
protected:
  static constexpr uint64_t kPageTable = 0x1000;
  static constexpr uint64_t kPhysicalPage = 0x2000;
  static constexpr uint64_t kGart = 0x1'0000'0000;
  static constexpr uint64_t kRing = kGart;
  static constexpr uint64_t kReadPointer = kGart + 0x700;
  static constexpr uint64_t kCompletion = kGart + 0x708;
  static constexpr uint64_t kMqd = kGart + 0x800;
  static constexpr uint64_t kSchedulerRing = kGart + 0xc00;
  static constexpr uint64_t kSchedulerReadPointer = kGart + 0xd00;
  static constexpr uint64_t kSchedulerDoorbell = 0x58;
  static constexpr uint64_t kMesDoorbell = 0x60;
  static constexpr uint32_t kFrameDwords = 64;

  MesEngineStateTest() : queue_registry_(gpu_vm_), engine_(gpu_vm_, queue_registry_, bindings_) {
    memory_->store<uint64_t>(kPageTable, kPhysicalPage | 0x3);
    EXPECT_TRUE(gpu_vm_.initialize_gart_address_space());
    EXPECT_TRUE(gpu_vm_.publish_gart(
        {.page_table_base = kPageTable, .aperture_start = kGart, .aperture_end = kGart + 0xfff},
        memory_));
    EXPECT_TRUE(engine_.attach_frontend("test", nullptr, {}, {}));
  }

  uint64_t physical(uint64_t gpu_address) const { return kPhysicalPage + gpu_address - kGart; }

  template <typename T> void store(uint64_t gpu_address, T value) {
    memory_->store<T>(physical(gpu_address), value);
  }

  template <typename T> T load(uint64_t gpu_address) const {
    return memory_->load<T>(physical(gpu_address));
  }

  void write_add_frame(uint64_t frame, uint64_t completion_value) {
    store<uint32_t>(frame, 0x00040021);
    store<uint64_t>(frame + 20 * sizeof(uint32_t), kMqd);
    store<uint32_t>(frame + 28 * sizeof(uint32_t), 3);
    store<uint64_t>(frame + 38 * sizeof(uint32_t), kCompletion);
    store<uint64_t>(frame + 40 * sizeof(uint32_t), completion_value);

    store<uint32_t>(kMqd + 136 * sizeof(uint32_t), static_cast<uint32_t>(kSchedulerRing >> 8));
    store<uint32_t>(kMqd + 137 * sizeof(uint32_t), static_cast<uint32_t>(kSchedulerRing >> 40));
    store<uint32_t>(kMqd + 139 * sizeof(uint32_t), static_cast<uint32_t>(kSchedulerReadPointer));
    store<uint32_t>(kMqd + 140 * sizeof(uint32_t),
                    static_cast<uint32_t>(kSchedulerReadPointer >> 32));
    store<uint32_t>(kMqd + 143 * sizeof(uint32_t), static_cast<uint32_t>(kSchedulerDoorbell));
    store<uint32_t>(kMqd + 145 * sizeof(uint32_t), 9);
  }

  void write_remove_frame(uint64_t frame, uint64_t completion_value) {
    store<uint32_t>(frame, 0x00040031);
    store<uint32_t>(frame + sizeof(uint32_t),
                    static_cast<uint32_t>(kSchedulerDoorbell / sizeof(uint32_t)));
    store<uint64_t>(frame + 6 * sizeof(uint32_t), kCompletion);
    store<uint64_t>(frame + 8 * sizeof(uint32_t), completion_value);
  }

  rocjitsu::amdgpu::MesDoorbellDisposition notify(uint64_t write_pointer) {
    const rocjitsu::amdgpu::MesKernelQueue queue{.ring_base = kRing,
                                                 .read_pointer_address = kReadPointer,
                                                 .write_pointer_address = 0,
                                                 .ring_dwords = 256,
                                                 .doorbell_offset = kMesDoorbell,
                                                 .active = true};
    const rocjitsu::amdgpu::MesDoorbellContext context([memory = memory_]() { return memory; },
                                                       0x1234, 7);
    return engine_.notify_doorbell(kMesDoorbell, write_pointer, queue, context);
  }

  std::shared_ptr<MesTestMemory> memory_ = std::make_shared<MesTestMemory>();
  rocjitsu::amdgpu::GpuVm gpu_vm_;
  rocjitsu::amdgpu::GpuQueueRegistry queue_registry_;
  RejectingComputeBindings bindings_;
  rocjitsu::amdgpu::MesEngine engine_;
};

TEST(MesEngineTest, FrontendAttachDetachAndResetAreExplicit) {
  rocjitsu::amdgpu::GpuVm gpu_vm;
  rocjitsu::amdgpu::GpuQueueRegistry queue_registry(gpu_vm);
  RejectingComputeBindings compute_bindings;
  rocjitsu::amdgpu::MesEngine engine(gpu_vm, queue_registry, compute_bindings);

  EXPECT_TRUE(engine.attach_frontend("first", nullptr, {}, {}));
  EXPECT_FALSE(engine.attach_frontend("second", nullptr, {}, {}));
  EXPECT_TRUE(engine.reset());
  EXPECT_TRUE(engine.detach_frontend());
  EXPECT_TRUE(engine.attach_frontend("replacement", nullptr, {}, {}));
  EXPECT_TRUE(engine.detach_frontend());
}

TEST_F(MesEngineStateTest, AddRemoveAndDetachFollowCommittedQueueState) {
  write_add_frame(kRing, 1);
  EXPECT_EQ(notify(kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Complete);
  EXPECT_EQ(engine_.active_queues(), 1u);
  EXPECT_EQ(load<uint64_t>(kCompletion), 1u);
  EXPECT_EQ(load<uint64_t>(kReadPointer), kFrameDwords);
  EXPECT_FALSE(engine_.detach_frontend());

  write_remove_frame(kRing + kFrameDwords * sizeof(uint32_t), 2);
  EXPECT_EQ(notify(2 * kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Complete);
  EXPECT_EQ(engine_.active_queues(), 0u);
  EXPECT_EQ(load<uint64_t>(kCompletion), 2u);
  EXPECT_EQ(load<uint64_t>(kReadPointer), 2 * kFrameDwords);
  EXPECT_TRUE(engine_.detach_frontend());
}

TEST_F(MesEngineStateTest, RemovePublicationRetryDoesNotReplayTheSemantic) {
  write_add_frame(kRing, 1);
  ASSERT_EQ(notify(kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Complete);
  ASSERT_EQ(engine_.active_queues(), 1u);

  write_remove_frame(kRing + kFrameDwords * sizeof(uint32_t), 2);
  memory_->make_next_write_unavailable(physical(kCompletion));
  EXPECT_EQ(notify(2 * kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Retry);
  EXPECT_EQ(engine_.active_queues(), 0u);
  EXPECT_EQ(load<uint64_t>(kCompletion), 1u);
  EXPECT_EQ(load<uint64_t>(kReadPointer), kFrameDwords);

  EXPECT_EQ(notify(2 * kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Complete);
  EXPECT_EQ(engine_.active_queues(), 0u);
  EXPECT_EQ(load<uint64_t>(kCompletion), 2u);
  EXPECT_EQ(load<uint64_t>(kReadPointer), 2 * kFrameDwords);
  EXPECT_TRUE(engine_.detach_frontend());
}

TEST_F(MesEngineStateTest, ResetClearsLiveQueuesBeforeFrontendDetach) {
  write_add_frame(kRing, 1);
  ASSERT_EQ(notify(kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Complete);
  ASSERT_EQ(engine_.active_queues(), 1u);

  EXPECT_TRUE(engine_.reset());
  EXPECT_EQ(engine_.active_queues(), 0u);
  EXPECT_TRUE(engine_.detach_frontend());
}

} // namespace
