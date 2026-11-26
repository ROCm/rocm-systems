// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/kmd/linux/linux_kfd.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_scheduler.h"
#include "rocjitsu/vm/rj_vm.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <span>
#include <sys/mman.h>
#include <unistd.h>

namespace {
thread_local bool trace_allocations = false;
thread_local std::size_t allocation_count = 0;
thread_local std::size_t fail_allocation = 0;
thread_local const volatile unsigned char *doorbell_page = nullptr;
thread_local void *last_allocation = nullptr;
thread_local bool injected = false;

class AllocationTrace {
public:
  explicit AllocationTrace(std::size_t failure_index,
                           const volatile unsigned char *mapped_page = nullptr) {
    allocation_count = 0;
    fail_allocation = failure_index;
    doorbell_page = mapped_page;
    injected = false;
    last_allocation = nullptr;
    trace_allocations = true;
  }
  ~AllocationTrace() {
    trace_allocations = false;
    doorbell_page = nullptr;
  }
  AllocationTrace(const AllocationTrace &) = delete;
  AllocationTrace &operator=(const AllocationTrace &) = delete;
};
} // namespace

// Isolated replacement, including nothrow new so ASan sees matching malloc/free.
void *operator new(std::size_t size) {
  if (trace_allocations && (!doorbell_page || *doorbell_page == 0xff) &&
      ++allocation_count == fail_allocation) {
    trace_allocations = false;
    injected = true;
    throw std::bad_alloc();
  }
  for (;;) {
    if (void *memory = std::malloc(size == 0 ? 1 : size)) {
      if (trace_allocations) {
        last_allocation = memory;
      }
      return memory;
    }
    const std::new_handler handler = std::get_new_handler();
    if (!handler) {
      throw std::bad_alloc();
    }
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
  if (GetParam() != QueueType::Sdma) {
    GTEST_SKIP() << "only SDMA resolves foreign factories during binding creation";
  }
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

INSTANTIATE_TEST_SUITE_P(QueueTypes, QueueRegistrationDeathTest,
                         ::testing::Values(QueueType::Compute, QueueType::Sdma));

} // namespace
} // namespace rocjitsu::amdgpu

namespace rocjitsu {
namespace {

TEST(KfdMmapAllocationDeathTest, FixedDoorbellReplacementTerminatesThroughLocalVmApi) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        constexpr size_t kPage = 4096;
        constexpr uint32_t kGpuId = 38144;
        rj_vm_t *vm = nullptr;
        if (rj_vm_create(RJ_KFD_MMAP_TEST_CONFIG, RJ_VM_MODE_LOCAL, &vm) !=
            ROCJITSU_STATUS_SUCCESS) {
          _exit(1);
        }
        void *target =
            ::mmap(nullptr, kPage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (target == MAP_FAILED) {
          _exit(2);
        }
        rj_vm_map_t map{};
        map.addr = reinterpret_cast<uint64_t>(target);
        map.length = kPage;
        map.offset = static_cast<int64_t>(KFD_MMAP_TYPE_DOORBELL | kfd_mmap_gpu_id(kGpuId));
        map.prot = PROT_READ | PROT_WRITE;
        map.flags = MAP_SHARED | MAP_FIXED;

        std::set_terminate([] { _exit(injected ? 77 : 78); });
        // Fail the first allocation after MAP_FIXED replaces the zero-filled
        // page with doorbell backing, which KFD initializes to 0xff.
        const AllocationTrace trace(1, static_cast<unsigned char *>(target));
        try {
          (void)rj_vm_device_map(vm, &map);
        } catch (const std::bad_alloc &) {
          _exit(79);
        }
        _exit(80);
      },
      ::testing::ExitedWithCode(77), "");
}

} // namespace
} // namespace rocjitsu
