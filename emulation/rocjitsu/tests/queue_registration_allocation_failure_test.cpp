// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_scheduler.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <new>
#include <span>

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

struct BindingObservation {
  bool destroyed = false;
  uint32_t queue_references_at_destruction = 0;
};

class ObservedBinding final : public QueueBinding {
public:
  ObservedBinding(std::unique_ptr<QueueBinding> binding, GpuVm &gpu_vm,
                  AddressSpaceHandle address_space, BindingObservation &observation)
      : binding_(std::move(binding)), gpu_vm_(gpu_vm), address_space_(address_space),
        observation_(observation) {}
  ~ObservedBinding() override {
    binding_.reset();
    observation_.destroyed = true;
    const std::optional<AddressSpaceInfo> info = gpu_vm_.lookup(address_space_);
    observation_.queue_references_at_destruction = info ? info->queue_references : 0;
  }
  QueuePrepareCloseStatus prepare_close() noexcept override { return binding_->prepare_close(); }
  QueueReconfigureStatus reconfigure(const QueueReconfigureRequest &request) override {
    return binding_->reconfigure(request);
  }
  QueueSubmissionStatus submit_producer(uint64_t cursor) override {
    return binding_->submit_producer(cursor);
  }

private:
  std::unique_ptr<QueueBinding> binding_;
  GpuVm &gpu_vm_;
  AddressSpaceHandle address_space_;
  BindingObservation &observation_;
};

class AllocationFactory final : public QueueBindingFactory {
public:
  AllocationFactory(std::shared_ptr<QueueBindingFactory> factory, GpuVm &gpu_vm,
                    BindingObservation &observation)
      : factory_(std::move(factory)), gpu_vm_(gpu_vm), observation_(observation) {}

  QueueBindingCreateResult create_binding(const QueueRegistrationRequest &request) override {
    QueueBindingCreateResult result;
    {
      const AllocationTrace trace(failure_index);
      result = factory_->create_binding(request);
    }
    allocations = allocation_count;
    final_allocation_is_binding = last_allocation == result.binding.get();
    if (fail_record && result) {
      result.binding = std::make_unique<ObservedBinding>(
          std::move(result.binding), gpu_vm_, request.identity.address_space, observation_);
      // No allocation intervenes between returning this binding and make_shared<QueueRecord>.
      allocation_count = 0;
      fail_allocation = 1;
      injected = false;
      trace_allocations = true;
    }
    return result;
  }

  std::size_t failure_index = 0;
  std::size_t allocations = 0;
  bool final_allocation_is_binding = false;
  bool fail_record = false;

private:
  std::shared_ptr<QueueBindingFactory> factory_;
  GpuVm &gpu_vm_;
  BindingObservation &observation_;
};

class QueueRegistrationAllocationTest : public ::testing::TestWithParam<QueueType> {
protected:
  class Queues {
  public:
    explicit Queues(QueueType type) : scheduler(gpu_vm), registry(gpu_vm), type_(type) {}

    void initialize() {
      auto memory = std::make_shared<QueueMemory>();
      address_space = gpu_vm.register_translated(7, memory, memory);
      ASSERT_TRUE(address_space);
      processor.set_gpu_vm(&gpu_vm);
      const std::shared_ptr<QueueBindingFactory> factory =
          type_ == QueueType::Compute ? make_compute_queue_binding_factory(processor)
                                      : make_sdma_queue_binding_factory(scheduler);
      allocation_factory = std::make_shared<AllocationFactory>(factory, gpu_vm, observation);
      request = {
          .identity = {.address_space = address_space, .process_id = 7, .queue_id = 11},
          .ring = {.base_address = 0x100, .size_bytes = 64, .consumer_pointer_address = 0x80},
          .doorbell = {},
          .binding_factory = allocation_factory,
          .type = type_,
          .packet_format =
              type_ == QueueType::Compute ? QueuePacketFormat::Pm4 : QueuePacketFormat::Sdma};
    }

    std::size_t owner_queues() const {
      return type_ == QueueType::Compute ? processor.registered_pm4_queue_count_for_test()
                                         : scheduler.active_queues();
    }

    void check_cleanup_and_retry() {
      EXPECT_EQ(owner_queues(), 0u);
      EXPECT_EQ(registry.active_queues(), 0u);
      ASSERT_TRUE(gpu_vm.lookup(address_space));
      EXPECT_EQ(gpu_vm.lookup(address_space)->queue_references, 0u);
      allocation_factory->failure_index = 0;
      allocation_factory->fail_record = false;
      const QueueHandle replacement = registry.register_queue(request);
      ASSERT_TRUE(replacement);
      EXPECT_EQ(owner_queues(), 1u);
      ASSERT_TRUE(registry.unregister_queue(replacement, QueueCloseMode::ForceCancel));
      EXPECT_EQ(owner_queues(), 0u);
      // A leaked execution-owner lease from the failed attempt would still block this.
      EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
    }

    GpuVm gpu_vm;
    CommandProcessor processor{"allocation-test"};
    SdmaQueueScheduler scheduler;
    BindingObservation observation;
    GpuQueueRegistry registry;
    AddressSpaceHandle address_space;
    std::shared_ptr<AllocationFactory> allocation_factory;
    QueueRegistrationRequest request;

  private:
    QueueType type_;
  };
};

TEST_P(QueueRegistrationAllocationTest, BindingAllocationFailureReleasesExecutionOwner) {
  std::size_t binding_allocation = 0;
  {
    Queues baseline(GetParam());
    ASSERT_NO_FATAL_FAILURE(baseline.initialize());
    const QueueHandle handle = baseline.registry.register_queue(baseline.request);
    ASSERT_TRUE(handle);
    ASSERT_EQ(baseline.owner_queues(), 1u);
    // Identify the actual make_unique allocation, without depending on private type sizes.
    ASSERT_TRUE(baseline.allocation_factory->final_allocation_is_binding);
    binding_allocation = baseline.allocation_factory->allocations;
    ASSERT_GT(binding_allocation, 0u);
    ASSERT_TRUE(baseline.registry.unregister_queue(handle, QueueCloseMode::ForceCancel));
  }

  // A fresh execution owner has the same allocation sequence as the baseline.
  Queues queues(GetParam());
  ASSERT_NO_FATAL_FAILURE(queues.initialize());
  queues.allocation_factory->failure_index = binding_allocation;
  EXPECT_THROW((void)queues.registry.register_queue(queues.request), std::bad_alloc);
  trace_allocations = false;
  ASSERT_TRUE(injected);
  EXPECT_EQ(allocation_count, binding_allocation);
  ASSERT_NO_FATAL_FAILURE(queues.check_cleanup_and_retry());
}

TEST_P(QueueRegistrationAllocationTest, RecordAllocationDestroysBindingBeforeReleasingVm) {
  Queues queues(GetParam());
  ASSERT_NO_FATAL_FAILURE(queues.initialize());
  queues.allocation_factory->fail_record = true;
  EXPECT_THROW((void)queues.registry.register_queue(queues.request), std::bad_alloc);
  trace_allocations = false;
  ASSERT_TRUE(injected);
  EXPECT_EQ(allocation_count, 1u);
  EXPECT_TRUE(queues.observation.destroyed);
  EXPECT_EQ(queues.observation.queue_references_at_destruction, 1u);
  ASSERT_NO_FATAL_FAILURE(queues.check_cleanup_and_retry());
}

INSTANTIATE_TEST_SUITE_P(ComputeAndSdma, QueueRegistrationAllocationTest,
                         ::testing::Values(QueueType::Compute, QueueType::Sdma));

} // namespace
} // namespace rocjitsu::amdgpu
