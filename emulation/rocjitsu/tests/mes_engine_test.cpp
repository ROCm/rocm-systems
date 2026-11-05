// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/mes_engine.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_scheduler.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_hsa_queue.h"
RJ_DIAGNOSTIC_POP

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

struct RecordingBindingState {
  std::vector<rocjitsu::amdgpu::QueueRegistrationRequest> requests;
  std::vector<uint64_t> submissions;
  uint32_t destructions = 0;
  bool reject_next = false;
};

class RecordingQueueBinding final : public rocjitsu::amdgpu::QueueBinding {
public:
  explicit RecordingQueueBinding(std::shared_ptr<RecordingBindingState> state)
      : state_(std::move(state)) {}
  ~RecordingQueueBinding() override { ++state_->destructions; }

  rocjitsu::amdgpu::QueueReconfigureStatus
  reconfigure(const rocjitsu::amdgpu::QueueReconfigureRequest &) override {
    return rocjitsu::amdgpu::QueueReconfigureStatus::Applied;
  }

  rocjitsu::amdgpu::QueueSubmissionStatus submit_producer(uint64_t producer_value) override {
    state_->submissions.push_back(producer_value);
    return rocjitsu::amdgpu::QueueSubmissionStatus::Accepted;
  }

private:
  std::shared_ptr<RecordingBindingState> state_;
};

class RecordingQueueBindingFactory final : public rocjitsu::amdgpu::QueueBindingFactory {
public:
  explicit RecordingQueueBindingFactory(std::shared_ptr<RecordingBindingState> state)
      : state_(std::move(state)) {}

  rocjitsu::amdgpu::QueueBindingCreateResult
  create_binding(const rocjitsu::amdgpu::QueueRegistrationRequest &request) override {
    rocjitsu::amdgpu::QueueRegistrationRequest recorded = request;
    recorded.binding_factory.reset();
    state_->requests.push_back(std::move(recorded));
    if (state_->reject_next) {
      state_->reject_next = false;
      return {};
    }
    return {.status = rocjitsu::amdgpu::QueueBindingCreateStatus::Bound,
            .binding = std::make_unique<RecordingQueueBinding>(state_)};
  }

private:
  std::shared_ptr<RecordingBindingState> state_;
};

class RecordingComputeBindings final : public rocjitsu::amdgpu::ComputeQueueBindingProvider {
public:
  RecordingComputeBindings()
      : aql_factory(std::make_shared<RecordingQueueBindingFactory>(aql)),
        pm4_factory(std::make_shared<RecordingQueueBindingFactory>(pm4)) {}

  std::optional<rocjitsu::amdgpu::ComputeQueueBindingPlan> make_aql(uint32_t ordinal) override {
    aql_ordinals.push_back(ordinal);
    return rocjitsu::amdgpu::ComputeQueueBindingPlan{.factory = aql_factory, .xcd_fanout = true};
  }

  std::optional<rocjitsu::amdgpu::ComputeQueueBindingPlan>
  make_pm4(uint32_t ordinal, rocjitsu::amdgpu::Pm4PacketCallbacks) override {
    pm4_ordinals.push_back(ordinal);
    return rocjitsu::amdgpu::ComputeQueueBindingPlan{.factory = pm4_factory, .xcd_fanout = false};
  }

  std::shared_ptr<RecordingBindingState> aql = std::make_shared<RecordingBindingState>();
  std::shared_ptr<RecordingBindingState> pm4 = std::make_shared<RecordingBindingState>();
  std::shared_ptr<RecordingQueueBindingFactory> aql_factory;
  std::shared_ptr<RecordingQueueBindingFactory> pm4_factory;
  std::vector<uint32_t> aql_ordinals;
  std::vector<uint32_t> pm4_ordinals;
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
  static constexpr uint64_t kProcessPageTable = 0x3000;
  static constexpr uint64_t kProcessContext = 0x123000;
  static constexpr uint64_t kComputeRing = 0x10000;
  static constexpr uint64_t kComputeReadPointer = 0x11000;
  static constexpr uint64_t kQueueDescriptor = 0x12000;
  static constexpr uint64_t kComputeWritePointer =
      kQueueDescriptor + offsetof(amd_queue_t, write_dispatch_id);
  static constexpr uint64_t kComputeDoorbell = 0x100;
  static constexpr uint32_t kFrameDwords = 64;

  MesEngineStateTest()
      : sdma_scheduler_(gpu_vm_), queue_registry_(gpu_vm_),
        sdma_factory_(rocjitsu::amdgpu::make_sdma_queue_binding_factory(sdma_scheduler_)),
        engine_(gpu_vm_, queue_registry_, bindings_) {
    constexpr uint64_t kReadable = uint64_t{1} << 5;
    constexpr uint64_t kWriteable = uint64_t{1} << 6;
    constexpr uint64_t kIsPte = uint64_t{1} << 63;
    memory_->store<uint64_t>(kPageTable, kPhysicalPage | 0x3 | kReadable | kWriteable | kIsPte);
    EXPECT_TRUE(gpu_vm_.initialize_gart_address_space());
    EXPECT_TRUE(gpu_vm_.publish_gart(
        {.page_table_base = kPageTable, .aperture_start = kGart, .aperture_end = kGart + 0xfff},
        memory_));
    EXPECT_TRUE(engine_.attach_frontend(
        "test", sdma_factory_, {},
        rocjitsu::amdgpu::MesFrontendCallbacks(
            {},
            [this](uint64_t read_pointer, uint64_t write_pointer) {
              published_pointers_.emplace_back(read_pointer, write_pointer);
            },
            [this](uint32_t offset, rocjitsu::amdgpu::QueuePacketFormat format) {
              resolved_doorbells_.emplace_back(offset, format);
              rocjitsu::amdgpu::QueueDoorbellBinding binding = frontend_doorbell_;
              binding.offset = offset;
              return binding;
            })));
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

  void write_invalidate_tlbs_frame(uint64_t frame, uint16_t process_id, uint64_t completion_value) {
    store<uint32_t>(frame, 0x00040141);
    store<uint64_t>(frame + 2 * sizeof(uint32_t), kCompletion);
    store<uint64_t>(frame + 4 * sizeof(uint32_t), completion_value);
    store<uint32_t>(frame + 6 * sizeof(uint32_t), static_cast<uint32_t>(process_id) << 16);
  }

  void write_compute_add_frame(uint64_t frame, bool aql, uint64_t completion_value) {
    store<uint32_t>(frame, 0x00040021);
    store<uint32_t>(frame + sizeof(uint32_t), 7);
    store<uint64_t>(frame + 2 * sizeof(uint32_t), kProcessPageTable);
    store<uint64_t>(frame + 10 * sizeof(uint32_t), kProcessContext);
    store<uint64_t>(frame + 20 * sizeof(uint32_t), kMqd);
    store<uint32_t>(frame + 28 * sizeof(uint32_t), 1);
    store<uint64_t>(frame + 38 * sizeof(uint32_t), kCompletion);
    store<uint64_t>(frame + 40 * sizeof(uint32_t), completion_value);

    store<uint32_t>(kMqd + 136 * sizeof(uint32_t), static_cast<uint32_t>(kComputeRing >> 8));
    store<uint32_t>(kMqd + 137 * sizeof(uint32_t), static_cast<uint32_t>(kComputeRing >> 40));
    store<uint32_t>(kMqd + 139 * sizeof(uint32_t), static_cast<uint32_t>(kComputeReadPointer));
    store<uint32_t>(kMqd + 140 * sizeof(uint32_t),
                    static_cast<uint32_t>(kComputeReadPointer >> 32));
    store<uint32_t>(kMqd + 141 * sizeof(uint32_t), static_cast<uint32_t>(kComputeWritePointer));
    store<uint32_t>(kMqd + 142 * sizeof(uint32_t),
                    static_cast<uint32_t>(kComputeWritePointer >> 32));
    store<uint32_t>(kMqd + 143 * sizeof(uint32_t), static_cast<uint32_t>(kComputeDoorbell));
    store<uint32_t>(kMqd + 145 * sizeof(uint32_t), 3 | (aql ? 0x08000000 : 0));
    store<uint32_t>(kMqd + 181 * sizeof(uint32_t), aql ? 1 : 0);
  }

  void write_sdma_add_frame(uint64_t frame, uint64_t completion_value) {
    constexpr uint64_t kSdmaRing = 0x13000;
    constexpr uint64_t kSdmaReadPointer = 0x14000;
    constexpr uint64_t kSdmaWritePointer = 0x14008;
    constexpr uint64_t kSdmaDoorbell = 0x180;
    store<uint32_t>(frame, 0x00040021);
    store<uint32_t>(frame + sizeof(uint32_t), 7);
    store<uint64_t>(frame + 2 * sizeof(uint32_t), kProcessPageTable);
    store<uint64_t>(frame + 10 * sizeof(uint32_t), kProcessContext);
    store<uint64_t>(frame + 20 * sizeof(uint32_t), kMqd);
    store<uint32_t>(frame + 18 * sizeof(uint32_t), kSdmaDoorbell / sizeof(uint32_t));
    store<uint64_t>(frame + 22 * sizeof(uint32_t), kSdmaWritePointer);
    store<uint32_t>(frame + 28 * sizeof(uint32_t), 2);
    store<uint32_t>(frame + 30 * sizeof(uint32_t), 16);
    store<uint64_t>(frame + 38 * sizeof(uint32_t), kCompletion);
    store<uint64_t>(frame + 40 * sizeof(uint32_t), completion_value);

    store<uint32_t>(kMqd, 8);
    store<uint32_t>(kMqd + sizeof(uint32_t), static_cast<uint32_t>(kSdmaRing >> 8));
    store<uint32_t>(kMqd + 2 * sizeof(uint32_t), static_cast<uint32_t>(kSdmaRing >> 40));
    store<uint64_t>(kMqd + 3 * sizeof(uint32_t), 0);
    store<uint64_t>(kMqd + 7 * sizeof(uint32_t), kSdmaReadPointer);
    store<uint32_t>(kMqd + 17 * sizeof(uint32_t), static_cast<uint32_t>(kSdmaDoorbell));
    store<uint64_t>(kMqd + 24 * sizeof(uint32_t), kSdmaWritePointer);
    store<uint32_t>(kMqd + 126 * sizeof(uint32_t), 2);
    store<uint32_t>(kMqd + 127 * sizeof(uint32_t), 17);
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
  rocjitsu::amdgpu::SdmaQueueScheduler sdma_scheduler_;
  rocjitsu::amdgpu::GpuQueueRegistry queue_registry_;
  RecordingComputeBindings bindings_;
  std::shared_ptr<rocjitsu::amdgpu::SdmaQueueBindingFactory> sdma_factory_;
  rocjitsu::amdgpu::MesEngine engine_;
  rocjitsu::amdgpu::QueueDoorbellBinding frontend_doorbell_;
  std::vector<std::pair<uint32_t, rocjitsu::amdgpu::QueuePacketFormat>> resolved_doorbells_;
  std::vector<std::pair<uint64_t, uint64_t>> published_pointers_;
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

TEST_F(MesEngineStateTest, ComputeQueuesUseFrontendPlacementDoorbellsAndLifecycle) {
  alignas(8) uint64_t doorbell = 0;
  frontend_doorbell_ = {.mode = rocjitsu::amdgpu::QueueDoorbellMode::HostPolled,
                        .host_base = &doorbell,
                        .last_value = 4};
  write_compute_add_frame(kRing, true, 1);
  memory_->make_next_write_unavailable(physical(kCompletion));

  EXPECT_EQ(notify(kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Retry);
  ASSERT_EQ(bindings_.aql->requests.size(), 1u);
  EXPECT_EQ(engine_.active_queues(), 1u);
  EXPECT_EQ(notify(kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Complete);
  EXPECT_EQ(bindings_.aql->requests.size(), 1u) << "publication retry replayed ADD_QUEUE";

  const rocjitsu::amdgpu::QueueRegistrationRequest &aql = bindings_.aql->requests.front();
  EXPECT_EQ(aql.abi, rocjitsu::amdgpu::QueueAbi::KfdAql);
  EXPECT_EQ(aql.queue_descriptor_address, kQueueDescriptor);
  EXPECT_TRUE(aql.xcd_fanout);
  EXPECT_EQ(aql.doorbell.mode, rocjitsu::amdgpu::QueueDoorbellMode::HostPolled);
  EXPECT_EQ(aql.doorbell.host_base, &doorbell);
  EXPECT_EQ(aql.doorbell.last_value, 4u);
  EXPECT_EQ(resolved_doorbells_.back(), (std::pair{static_cast<uint32_t>(kComputeDoorbell),
                                                   rocjitsu::amdgpu::QueuePacketFormat::Aql}));

  EXPECT_EQ(engine_.notify_doorbell(kComputeDoorbell, 9, std::nullopt,
                                    rocjitsu::amdgpu::MesDoorbellContext({}, 0x1234, 7)),
            rocjitsu::amdgpu::MesDoorbellDisposition::Complete);
  EXPECT_EQ(bindings_.aql->submissions, (std::vector<uint64_t>{9}));

  write_remove_frame(kRing + kFrameDwords * sizeof(uint32_t), 2);
  store<uint32_t>(kRing + kFrameDwords * sizeof(uint32_t) + sizeof(uint32_t),
                  kComputeDoorbell / sizeof(uint32_t));
  EXPECT_EQ(notify(2 * kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Complete);
  EXPECT_EQ(bindings_.aql->destructions, 1u);
  EXPECT_EQ(engine_.active_queues(), 0u);

  write_compute_add_frame(kRing + 2 * kFrameDwords * sizeof(uint32_t), false, 3);
  EXPECT_EQ(notify(3 * kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Complete);
  ASSERT_EQ(bindings_.pm4->requests.size(), 1u);
  EXPECT_EQ(bindings_.pm4->requests.front().packet_format,
            rocjitsu::amdgpu::QueuePacketFormat::Pm4);
  EXPECT_FALSE(bindings_.pm4->requests.front().xcd_fanout);
  EXPECT_EQ(bindings_.aql_ordinals, (std::vector<uint32_t>{0}));
  EXPECT_EQ(bindings_.pm4_ordinals, (std::vector<uint32_t>{1}));

  write_remove_frame(kRing + 3 * kFrameDwords * sizeof(uint32_t), 4);
  store<uint32_t>(kRing + 3 * kFrameDwords * sizeof(uint32_t) + sizeof(uint32_t),
                  kComputeDoorbell / sizeof(uint32_t));
  EXPECT_EQ(notify(4 * kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Complete);
  EXPECT_EQ(bindings_.pm4->destructions, 1u);
}

TEST_F(MesEngineStateTest, PasidDoesNotClaimTheSameNumericVmidRoute) {
  const auto routed = gpu_vm_.register_address_space(
      7, std::make_shared<rocjitsu::amdgpu::IdentityAddressSpaceTranslator>(), memory_);
  ASSERT_TRUE(routed);
  ASSERT_EQ(gpu_vm_.find_vmid(7), routed);

  write_compute_add_frame(kRing, true, 1);
  EXPECT_EQ(notify(kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Complete);
  EXPECT_EQ(engine_.active_queues(), 1u);
  EXPECT_EQ(gpu_vm_.find_vmid(7), routed) << "the MES PASID displaced a frontend-owned VMID route";

  ASSERT_EQ(bindings_.aql->requests.size(), 1u);
  const auto mes = bindings_.aql->requests.front().identity.address_space;
  ASSERT_TRUE(gpu_vm_.lookup(mes));
  ASSERT_TRUE(gpu_vm_.lookup(routed));
  const uint64_t mes_epoch = gpu_vm_.lookup(mes)->translation_epoch;
  const uint64_t routed_epoch = gpu_vm_.lookup(routed)->translation_epoch;

  write_invalidate_tlbs_frame(kRing + kFrameDwords * sizeof(uint32_t), 7, 2);
  EXPECT_EQ(notify(2 * kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Complete);
  ASSERT_TRUE(gpu_vm_.lookup(mes));
  ASSERT_TRUE(gpu_vm_.lookup(routed));
  EXPECT_EQ(gpu_vm_.lookup(mes)->translation_epoch, mes_epoch + 1);
  EXPECT_EQ(gpu_vm_.lookup(routed)->translation_epoch, routed_epoch);

  EXPECT_TRUE(engine_.reset());
  EXPECT_EQ(gpu_vm_.find_vmid(7), routed);
  EXPECT_TRUE(gpu_vm_.unregister_address_space(routed));
}

TEST_F(MesEngineStateTest, MalformedSdmaMqdIsRejectedBeforeQueueCommit) {
  write_sdma_add_frame(kRing, 1);
  store<uint64_t>(kMqd + 7 * sizeof(uint32_t), 0);

  EXPECT_EQ(notify(kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Faulted);
  EXPECT_EQ(engine_.active_queues(), 0u);
  EXPECT_EQ(sdma_scheduler_.active_queues(), 0u);
  EXPECT_EQ(load<uint64_t>(kCompletion), 0u);
  EXPECT_EQ(load<uint64_t>(kReadPointer), 0u);
}

TEST_F(MesEngineStateTest, SdmaAddDoorbellAndRemoveUseTheSharedQueueRegistry) {
  constexpr uint64_t kSdmaDoorbell = 0x180;
  write_sdma_add_frame(kRing, 1);

  EXPECT_EQ(notify(kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Complete);
  EXPECT_EQ(engine_.active_queues(), 1u);
  EXPECT_EQ(sdma_scheduler_.active_queues(), 1u);
  EXPECT_EQ(resolved_doorbells_.back(), (std::pair{static_cast<uint32_t>(kSdmaDoorbell),
                                                   rocjitsu::amdgpu::QueuePacketFormat::Sdma}));
  EXPECT_EQ(engine_.notify_doorbell(kSdmaDoorbell, 0, std::nullopt,
                                    rocjitsu::amdgpu::MesDoorbellContext({}, 0x1234, 7)),
            rocjitsu::amdgpu::MesDoorbellDisposition::Complete);

  write_remove_frame(kRing + kFrameDwords * sizeof(uint32_t), 2);
  store<uint32_t>(kRing + kFrameDwords * sizeof(uint32_t) + sizeof(uint32_t),
                  kSdmaDoorbell / sizeof(uint32_t));
  EXPECT_EQ(notify(2 * kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Complete);
  EXPECT_EQ(sdma_scheduler_.active_queues(), 0u);
  EXPECT_EQ(engine_.active_queues(), 0u);
}

TEST_F(MesEngineStateTest, FailedComputeRegistrationAndGfxQueueDoNotCommit) {
  bindings_.aql->reject_next = true;
  write_compute_add_frame(kRing, true, 1);
  EXPECT_EQ(notify(kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Faulted);
  EXPECT_EQ(bindings_.aql->requests.size(), 1u);
  EXPECT_EQ(engine_.active_queues(), 0u);
  EXPECT_EQ(load<uint64_t>(kReadPointer), 0u);

  store<uint32_t>(kRing + 28 * sizeof(uint32_t), 0);
  EXPECT_EQ(notify(kFrameDwords), rocjitsu::amdgpu::MesDoorbellDisposition::Faulted);
  EXPECT_EQ(engine_.active_queues(), 0u);
  EXPECT_EQ(load<uint64_t>(kReadPointer), 0u);
}

} // namespace
