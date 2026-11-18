// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/mes_engine.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_scheduler.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <span>
#include <unistd.h>

namespace {
thread_local bool trace_allocations = false;
thread_local std::size_t allocation_count = 0;
thread_local std::size_t fail_allocation = 0;
thread_local void *last_allocation = nullptr;
thread_local bool injected = false;

class AllocationTrace {
public:
  explicit AllocationTrace(std::size_t failure_index) {
    allocation_count = 0;
    fail_allocation = failure_index;
    injected = false;
    last_allocation = nullptr;
    trace_allocations = true;
  }
  ~AllocationTrace() { trace_allocations = false; }
  AllocationTrace(const AllocationTrace &) = delete;
  AllocationTrace &operator=(const AllocationTrace &) = delete;
};
} // namespace

// Isolated replacement, including nothrow new so ASan sees matching malloc/free.
void *operator new(std::size_t size) {
  if (trace_allocations && ++allocation_count == fail_allocation) {
    trace_allocations = false;
    injected = true;
    throw std::bad_alloc();
  }
  for (;;) {
    if (void *memory = std::malloc(size == 0 ? 1 : size)) {
      if (trace_allocations)
        last_allocation = memory;
      return memory;
    }
    const std::new_handler handler = std::get_new_handler();
    if (!handler)
      throw std::bad_alloc();
    handler();
  }
}

// Keep free out of new/delete callers to avoid GCC's mismatched-new-delete warning.
RJ_NOINLINE void operator delete(void *memory) noexcept { std::free(memory); }
RJ_NOINLINE void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void *operator new(std::size_t size, const std::nothrow_t &) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}
RJ_NOINLINE void operator delete(void *memory, const std::nothrow_t &) noexcept {
  ::operator delete(memory);
}

namespace rocjitsu::amdgpu {
namespace {

class QueueMemory final : public AddressSpaceTranslator, public PhysicalMemoryAccess {
public:
  VmTranslationResult translate(uint64_t address, std::size_t size, VmAccessKind) const override {
    return {.outcome = VmAccessOutcome::Complete,
            .translation = {.domain = VmMemoryDomain::System,
                            .address = address,
                            .contiguous_bytes = size,
                            .mtype = Mtype::RW,
                            .permissions = {.readable = true, .writable = true}}};
  }
  VmAccessOutcome read(VmMemoryDomain, uint64_t, std::span<std::byte> bytes) override {
    std::ranges::fill(bytes, std::byte{0});
    return VmAccessOutcome::Complete;
  }
  VmAccessOutcome write(VmMemoryDomain, uint64_t, std::span<const std::byte>) override {
    return VmAccessOutcome::Complete;
  }
};

class QueueRegistrationDeathTest : public ::testing::TestWithParam<QueueType> {
protected:
  void SetUp() override {
    auto memory = std::make_shared<QueueMemory>();
    address_space_ = gpu_vm_.register_translated(7, memory, memory);
    ASSERT_TRUE(address_space_);
    processor_.set_gpu_vm(&gpu_vm_);
    factory_ = GetParam() == QueueType::Compute ? make_compute_queue_binding_factory(processor_)
                                                : make_sdma_queue_binding_factory(scheduler_);
    request_.identity.address_space = address_space_;
    request_.identity.process_id = 7;
    request_.identity.queue_id = 11;
    request_.ring = {.base_address = 0x100, .size_bytes = 64, .consumer_pointer_address = 0x80};
    request_.binding_factory = factory_;
    request_.type = GetParam();
    request_.packet_format =
        GetParam() == QueueType::Compute ? QueuePacketFormat::Pm4 : QueuePacketFormat::Sdma;
  }

  GpuVm gpu_vm_;
  CommandProcessor processor_{"allocation-test"};
  SdmaQueueScheduler scheduler_{gpu_vm_};
  AddressSpaceHandle address_space_;
  std::shared_ptr<QueueBindingFactory> factory_;
  QueueRegistrationRequest request_;
};

TEST_P(QueueRegistrationDeathTest, BindingAllocationFailureCannotReachRecoveringAdapter) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  size_t binding_allocation = 0;
  {
    QueueBindingCreateResult warm = factory_->create_binding(request_);
    ASSERT_TRUE(warm);
  }
  {
    QueueBindingCreateResult baseline;
    {
      const AllocationTrace trace(0);
      baseline = factory_->create_binding(request_);
    }
    ASSERT_TRUE(baseline);
    ASSERT_EQ(last_allocation, baseline.binding.get());
    binding_allocation = allocation_count;
  }
  // Fail the ownership allocation after the processor has attached the queue.
  {
    const size_t failure_index = binding_allocation;
    EXPECT_EXIT(
        {
          std::set_terminate([] { _exit(injected ? 77 : 78); });
          const AllocationTrace trace(failure_index);
          try {
            (void)factory_->create_binding(request_);
          } catch (...) {
            _exit(79);
          }
          _exit(80);
        },
        ::testing::ExitedWithCode(77), "");
  }
}

TEST_P(QueueRegistrationDeathTest, ForeignCallbackAllocationExceptionStillPropagates) {
  if (GetParam() != QueueType::Sdma)
    GTEST_SKIP() << "only SDMA resolves foreign factories during binding creation";
  for (bool progress : {false, true}) {
    SdmaCallbackFactory callbacks =
        progress ? SdmaCallbackFactory{}
                 : [](const SdmaQueueContext &) -> SdmaPacketCallbacks { throw std::bad_alloc(); };
    SdmaProgressObserverFactory observer = progress
        ? [](const SdmaQueueContext &) -> SdmaQueueProgressObserver { throw std::bad_alloc(); }
    : SdmaProgressObserverFactory{};
    SdmaQueueBindingFactory factory(scheduler_, std::move(callbacks), std::move(observer));
    EXPECT_THROW((void)factory.create_binding(request_), std::bad_alloc);
    EXPECT_EQ(scheduler_.active_queues(), 0u);
  }
}

class MesPublicationMemory final : public PhysicalMemoryAccess {
public:
  VmAccessOutcome read(VmMemoryDomain, uint64_t address, std::span<std::byte> bytes) override {
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::memcpy(bytes.data(), bytes_.data() + address, bytes.size());
    return VmAccessOutcome::Complete;
  }
  VmAccessOutcome write(VmMemoryDomain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    if (address == 0x2708)
      publication_allocation_count = allocation_count;
    std::memcpy(bytes_.data() + address, bytes.data(), bytes.size());
    return VmAccessOutcome::Complete;
  }
  template <typename Value> void store(uint64_t address, Value value) {
    std::memcpy(bytes_.data() + address, &value, sizeof(value));
  }

  template <typename Value> Value load(uint64_t address) const {
    Value value;
    std::memcpy(&value, bytes_.data() + address, sizeof(value));
    return value;
  }

  std::size_t publication_allocation_count = 0;

private:
  std::array<std::byte, 0x4000> bytes_{};
};

class MesPublicationBindings final : public ComputeQueueBindingProvider {
public:
  explicit MesPublicationBindings(CommandProcessor &processor)
      : factory_(make_compute_queue_binding_factory(processor)) {}
  std::optional<ComputeQueueBindingPlan> make_aql(uint32_t) override { return std::nullopt; }
  std::optional<ComputeQueueBindingPlan> make_pm4(uint32_t ordinal, Pm4PacketCallbacks) override {
    last_ordinal = ordinal;
    return ComputeQueueBindingPlan{.factory = factory_};
  }

  uint32_t last_ordinal = UINT32_MAX;

private:
  std::shared_ptr<QueueBindingFactory> factory_;
};

class MesPublicationAllocationTest : public ::testing::Test {
protected:
  class State {
  public:
    State() {
      processor.set_gpu_vm(&gpu_vm);
      memory->store<uint64_t>(0x1000, 0x2000 | 0x63 | (uint64_t{1} << 63));
      EXPECT_TRUE(gpu_vm.initialize_gart_address_space());
      EXPECT_TRUE(gpu_vm.publish_gart(
          {.page_table_base = 0x1000, .aperture_start = kGart, .aperture_end = kGart + 0xfff},
          memory));
      EXPECT_TRUE(engine.attach_frontend("MES allocation test", nullptr, {}, {}));
      // One PM4 ADD_QUEUE frame with an MQD in the GART mapping.
      memory->store<uint32_t>(0x2000, 0x00040021);
      memory->store<uint32_t>(0x2000 + sizeof(uint32_t), 7);
      memory->store<uint64_t>(0x2000 + 2 * sizeof(uint32_t), 0x3000);
      memory->store<uint64_t>(0x2000 + 20 * sizeof(uint32_t), kGart + 0x800);
      memory->store<uint32_t>(0x2000 + 28 * sizeof(uint32_t), 1);
      memory->store<uint64_t>(0x2000 + 38 * sizeof(uint32_t), kGart + 0x708);
      memory->store<uint64_t>(0x2000 + 40 * sizeof(uint32_t), 1);
      memory->store<uint32_t>(0x2800 + 136 * sizeof(uint32_t), 0x100);
      memory->store<uint32_t>(0x2800 + 139 * sizeof(uint32_t), 0x11000);
      memory->store<uint32_t>(0x2800 + 141 * sizeof(uint32_t), 0x12000);
      memory->store<uint32_t>(0x2800 + 143 * sizeof(uint32_t), 0x100);
      memory->store<uint32_t>(0x2800 + 145 * sizeof(uint32_t), 9);
    }
    ~State() { (void)engine.teardown_queues(); }

    MesDoorbellDisposition notify() {
      const MesKernelQueue queue{.ring_base = kGart,
                                 .read_pointer_address = kGart + 0x700,
                                 .ring_dwords = 256,
                                 .doorbell_offset = 0x60,
                                 .active = true};
      const MesDoorbellContext context(
          [this] {
            preparation_allocation_count = allocation_count;
            return memory;
          },
          0x1234, 7);
      return engine.notify_doorbell(0x60, 64, queue, context);
    }

    std::size_t preparation_allocation_count = 0;
    std::shared_ptr<MesPublicationMemory> memory = std::make_shared<MesPublicationMemory>();
    GpuVm gpu_vm;
    CommandProcessor processor{"MES allocation test"};
    GpuQueueRegistry registry{gpu_vm};
    MesPublicationBindings bindings{processor};
    MesEngine engine{gpu_vm, registry, bindings};

  private:
    static constexpr uint64_t kGart = 0x1'0000'0000;
  };
};

TEST_F(MesPublicationAllocationTest, AllocationFailurePreservesQueueStateAndAllowsRetry) {
  {
    State warm;
    ASSERT_EQ(warm.notify(), MesDoorbellDisposition::Complete);
  }
  for (bool preparation : {true, false}) {
    std::size_t failure_index = 0;
    {
      State baseline;
      MesDoorbellDisposition result;
      {
        const AllocationTrace trace(0);
        result = baseline.notify();
      }
      ASSERT_EQ(result, MesDoorbellDisposition::Complete);
      ASSERT_EQ(baseline.registry.active_queues(), 1u);
      ASSERT_EQ(baseline.engine.active_queues(), 1u);
      // Cover bookkeeping preparation and the last allocation before publication.
      failure_index = preparation ? baseline.preparation_allocation_count
                                  : baseline.memory->publication_allocation_count;
      ASSERT_GT(failure_index, 0u);
    }
    State state;
    {
      const AllocationTrace trace(failure_index);
      EXPECT_THROW((void)state.notify(), std::bad_alloc);
    }
    ASSERT_TRUE(injected);
    EXPECT_EQ(state.registry.active_queues(), 0u);
    EXPECT_EQ(state.engine.active_queues(), 0u);
    EXPECT_EQ(state.engine.active_address_spaces(), 0u);
    EXPECT_EQ(state.memory->load<uint64_t>(0x2708), 0u);
    EXPECT_EQ(state.memory->load<uint64_t>(0x2700), 0u);
    EXPECT_EQ(state.notify(), MesDoorbellDisposition::Complete);
    EXPECT_EQ(state.memory->load<uint64_t>(0x2708), 1u);
    EXPECT_EQ(state.memory->load<uint64_t>(0x2700), 64u);
    EXPECT_EQ(state.registry.active_queues(), 1u);
    EXPECT_EQ(state.engine.active_queues(), 1u);
    EXPECT_EQ(state.bindings.last_ordinal, 0u);
  }
}

INSTANTIATE_TEST_SUITE_P(QueueTypes, QueueRegistrationDeathTest,
                         ::testing::Values(QueueType::Compute, QueueType::Sdma));

} // namespace
} // namespace rocjitsu::amdgpu
