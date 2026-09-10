// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"

#include "rocjitsu/isa/arch/amdgpu/shared/tensor_dma.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::test::cdna5;

class TestMemoryInstruction final : public Instruction {
public:
  explicit TestMemoryInstruction(std::unique_ptr<DynamicInstState> state)
      : Instruction("test_mem", nullptr) {
    flags_ |= MEMORY_OP;
    set_data(std::move(state));
  }
};

class MemoryLifecyclePlugin final : public ExecutionPlugin {
public:
  MemoryLifecyclePlugin() : ExecutionPlugin("vm-memory-lifecycle") {}

  void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &inst,
                                        amdgpu::Wavefront &) override {
    if (inst.mnemonic() == kMnemonic)
      ++before_count;
  }

  void onAmdgpuAfterExecuteInstruction(uint64_t, const Instruction &inst,
                                       amdgpu::Wavefront &) override {
    if (inst.mnemonic() == kMnemonic)
      ++after_count;
  }

  void onAmdgpuRouteMemoryInstruction(const Instruction &inst, amdgpu::Wavefront &) override {
    if (inst.mnemonic() == kMnemonic)
      ++route_count;
  }

  static constexpr std::string_view kMnemonic = "global_load_b32";
  uint32_t before_count = 0;
  uint32_t after_count = 0;
  uint32_t route_count = 0;
};

class TestExternalAddressSpace final : public amdgpu::AddressSpaceTranslator,
                                       public amdgpu::PhysicalMemoryAccess {
public:
  struct AccessRecord {
    uint64_t address = 0;
    std::size_t size = 0;
  };

  explicit TestExternalAddressSpace(std::size_t size = 64 * 1024) : bytes_(size) {}

  amdgpu::VmTranslationResult translate(uint64_t address, std::size_t size,
                                        amdgpu::VmAccessKind access) const override {
    if (size == 0 || address > bytes_.size() || size > bytes_.size() - address)
      return {.outcome = amdgpu::VmAccessOutcome::Faulted, .translation = {}};
    if (access == amdgpu::VmAccessKind::Atomic && !atomic_allowed_)
      return {.outcome = amdgpu::VmAccessOutcome::Faulted, .translation = {}};
    uint64_t contiguous_bytes = bytes_.size() - address;
    if (translation_granule != 0) {
      const uint64_t granule_remaining = translation_granule - address % translation_granule;
      contiguous_bytes = std::min(contiguous_bytes, granule_remaining);
    }
    return {
        .outcome = amdgpu::VmAccessOutcome::Complete,
        .translation = {.domain = amdgpu::VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = contiguous_bytes,
                        .mtype = amdgpu::Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }

  amdgpu::VmAccessOutcome read(amdgpu::VmMemoryDomain domain, uint64_t address,
                               std::span<std::byte> bytes) override {
    std::lock_guard lock(mutex_);
    ++read_calls;
    last_read = {.address = address, .size = bytes.size()};
    read_history.push_back(last_read);
    if (domain == amdgpu::VmMemoryDomain::Compatibility || !contains(address, bytes.size()))
      return amdgpu::VmAccessOutcome::Faulted;
    if (unavailable_read_call == read_calls) {
      unavailable_read_call = 0;
      return amdgpu::VmAccessOutcome::Unavailable;
    }
    std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(address), bytes.size(), bytes.begin());
    return amdgpu::VmAccessOutcome::Complete;
  }

  amdgpu::VmAccessOutcome write(amdgpu::VmMemoryDomain domain, uint64_t address,
                                std::span<const std::byte> bytes) override {
    std::lock_guard lock(mutex_);
    ++write_calls;
    last_write = {.address = address, .size = bytes.size()};
    write_history.push_back(last_write);
    if (domain == amdgpu::VmMemoryDomain::Compatibility || !contains(address, bytes.size()))
      return amdgpu::VmAccessOutcome::Faulted;
    if (unavailable_write_call == write_calls) {
      unavailable_write_call = 0;
      return amdgpu::VmAccessOutcome::Unavailable;
    }
    std::copy(bytes.begin(), bytes.end(), bytes_.begin() + static_cast<std::ptrdiff_t>(address));
    return amdgpu::VmAccessOutcome::Complete;
  }

  amdgpu::AtomicLoadResult atomic_load(amdgpu::VmMemoryDomain domain, uint64_t address,
                                       uint32_t width) override {
    std::lock_guard lock(mutex_);
    ++atomic_load_calls;
    if (unavailable_atomic_load_call == atomic_load_calls) {
      unavailable_atomic_load_call = 0;
      return {.outcome = amdgpu::VmAccessOutcome::Unavailable};
    }
    if (next_atomic_load_outcome != amdgpu::VmAccessOutcome::Complete) {
      const amdgpu::VmAccessOutcome outcome = next_atomic_load_outcome;
      next_atomic_load_outcome = amdgpu::VmAccessOutcome::Complete;
      return {.outcome = outcome};
    }
    if (!valid_atomic(domain, address, width))
      return {.outcome = amdgpu::VmAccessOutcome::Faulted};
    uint64_t value = 0;
    std::memcpy(&value, bytes_.data() + address, width);
    return {.outcome = amdgpu::VmAccessOutcome::Complete, .value = value};
  }

  amdgpu::VmAccessOutcome atomic_store(amdgpu::VmMemoryDomain domain, uint64_t address,
                                       uint32_t width, uint64_t value) override {
    std::lock_guard lock(mutex_);
    ++atomic_store_calls;
    if (unavailable_atomic_store_address && address == *unavailable_atomic_store_address) {
      ++unavailable_atomic_store_hits;
      return amdgpu::VmAccessOutcome::Unavailable;
    }
    if (!valid_atomic(domain, address, width))
      return amdgpu::VmAccessOutcome::Faulted;
    std::memcpy(bytes_.data() + address, &value, width);
    return amdgpu::VmAccessOutcome::Complete;
  }

  amdgpu::AtomicCompareExchangeResult compare_exchange(amdgpu::VmMemoryDomain domain,
                                                       uint64_t address, uint32_t width,
                                                       uint64_t expected,
                                                       uint64_t desired) override {
    std::lock_guard lock(mutex_);
    ++compare_exchange_calls;
    if (!valid_atomic(domain, address, width))
      return {.outcome = amdgpu::VmAccessOutcome::Faulted};
    uint64_t observed = 0;
    std::memcpy(&observed, bytes_.data() + address, width);
    const uint64_t mask = width == sizeof(uint64_t) ? UINT64_MAX : UINT32_MAX;
    if (forced_compare_exchange_misses != 0) {
      --forced_compare_exchange_misses;
      observed = (observed + 1) & mask;
      std::memcpy(bytes_.data() + address, &observed, width);
      return {
          .outcome = amdgpu::VmAccessOutcome::Complete, .observed = observed, .exchanged = false};
    }
    const bool exchanged = (observed & mask) == (expected & mask);
    if (exchanged)
      std::memcpy(bytes_.data() + address, &desired, width);
    return {
        .outcome = amdgpu::VmAccessOutcome::Complete, .observed = observed, .exchanged = exchanged};
  }

  template <typename T> void store(uint64_t address, T value) {
    std::lock_guard lock(mutex_);
    ASSERT_TRUE(contains(address, sizeof(value)));
    std::memcpy(bytes_.data() + address, &value, sizeof(value));
  }

  template <typename T> T load(uint64_t address) const {
    std::lock_guard lock(mutex_);
    T value{};
    EXPECT_TRUE(contains(address, sizeof(value)));
    if (contains(address, sizeof(value)))
      std::memcpy(&value, bytes_.data() + address, sizeof(value));
    return value;
  }

  bool atomic_allowed_ = true;
  amdgpu::VmAccessOutcome next_atomic_load_outcome = amdgpu::VmAccessOutcome::Complete;
  uint64_t translation_granule = 0;
  uint32_t unavailable_read_call = 0;
  uint32_t unavailable_write_call = 0;
  uint32_t unavailable_atomic_load_call = 0;
  uint32_t read_calls = 0;
  uint32_t write_calls = 0;
  uint32_t atomic_load_calls = 0;
  uint32_t atomic_store_calls = 0;
  uint32_t unavailable_atomic_store_hits = 0;
  uint32_t compare_exchange_calls = 0;
  uint32_t forced_compare_exchange_misses = 0;
  std::optional<uint64_t> unavailable_atomic_store_address;
  AccessRecord last_read;
  AccessRecord last_write;
  std::vector<AccessRecord> read_history;
  std::vector<AccessRecord> write_history;

private:
  bool contains(uint64_t address, std::size_t size) const {
    return address <= bytes_.size() && size <= bytes_.size() - address;
  }

  bool valid_atomic(amdgpu::VmMemoryDomain domain, uint64_t address, uint32_t width) const {
    return domain == amdgpu::VmMemoryDomain::System &&
           (width == sizeof(uint32_t) || width == sizeof(uint64_t)) && address % width == 0 &&
           contains(address, width);
  }

  mutable std::mutex mutex_;
  std::vector<std::byte> bytes_;
};

class TranslatedPipelineContext {
public:
  Gfx1250Sim sim;
  std::shared_ptr<TestExternalAddressSpace> external = std::make_shared<TestExternalAddressSpace>();
  amdgpu::ComputeUnitCore *cu = sim.cu();
  amdgpu::Wavefront *wf = nullptr;
  amdgpu::AddressSpaceHandle address_space;

  TranslatedPipelineContext() {
    address_space = sim.soc->gpu_vm().register_translated(77, external, external);
    cu->set_gpu_vm(&sim.soc->gpu_vm());
    wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    if (wf != nullptr) {
      wf->set_process_id(77);
      wf->set_address_space(address_space);
      wf->set_exec(0x3u);
    }
  }
};

TEST(GpuVmPipeline, TranslatedScalarLoadAndStoreUseExternalBacking) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);

  constexpr uint64_t kAddress = 0x100;
  constexpr uint32_t kExternalValue = 0x12345678u;
  constexpr uint32_t kLegacyValue = 0xdeadbeefu;
  constexpr uint32_t kStoredValue = 0xa5a55a5au;
  constexpr uint32_t kDestination = 4;
  context.external->store(kAddress, kExternalValue);
  context.sim.memory->write32(kAddress, kLegacyValue);

  auto load = std::make_unique<amdgpu::ScalarMemState>();
  load->addr = kAddress;
  load->dst_register = {amdgpu::ScalarRegisterStorage::SGPR, kDestination, 1};
  load->num_dwords = 1;
  load->is_load = true;
  amdgpu::ScalarMemPipeline pipeline(&context.cu->l1_scalar());
  EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(load)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(context.cu->read_sgpr_storage(context.wf->sgpr_alloc().base + kDestination),
            kExternalValue);

  auto store = std::make_unique<amdgpu::ScalarMemState>();
  store->addr = kAddress;
  store->num_dwords = 1;
  store->is_load = false;
  store->store_data[0] = kStoredValue;
  EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(store)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);

  EXPECT_EQ(context.external->load<uint32_t>(kAddress), kStoredValue);
  EXPECT_EQ(context.sim.memory->read32(kAddress), kLegacyValue);
  EXPECT_EQ(context.external->read_calls, 1u);
  EXPECT_EQ(context.external->write_calls, 1u);
}

TEST(GpuVmPipeline, ScratchWithoutBackingIsATerminalFault) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);
  ASSERT_EQ(context.wf->scratch_base(), 0u);

  auto store = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  store->elem_size = sizeof(uint8_t);
  store->num_elems = 1;
  store->is_load = false;
  store->wf_size = context.wf->wf_size();
  store->exec_mask = 1;
  store->lane_mask = 1;
  store->scratch_swizzle = true;
  store->requires_scratch_backing = true;
  store->scratch_lane_mask = 1;
  store->scratch_addr_stride = context.wf->wf_size() * sizeof(uint32_t);
  store->store_data.resize(context.wf->wf_size());

  amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(store)), *context.wf),
            amdgpu::VmAccessOutcome::Faulted);
  EXPECT_TRUE(pipeline.empty());
  EXPECT_TRUE(context.wf->wait_counters().empty());
  EXPECT_NE(context.wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(context.external->read_calls, 0u);
  EXPECT_EQ(context.external->write_calls, 0u);
}

TEST(GpuVmPipeline, ScratchRetryWithRevokedBackingReportsOneFault) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);

  constexpr uint64_t kScratchBase = 0x100;
  context.wf->set_scratch_base(kScratchBase);
  context.external->unavailable_write_call = 1;

  auto store = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  store->elem_size = sizeof(uint8_t);
  store->num_elems = 1;
  store->is_load = false;
  store->wf_size = context.wf->wf_size();
  store->exec_mask = 1;
  store->lane_mask = 1;
  store->scratch_swizzle = true;
  store->requires_scratch_backing = true;
  store->scratch_lane_mask = 1;
  store->scratch_addr_stride = context.wf->wf_size() * sizeof(uint32_t);
  store->per_lane_addr[0] = kScratchBase;
  store->store_data.resize(context.wf->wf_size());

  amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
  uint32_t fault_count = 0;
  amdgpu::VmAccessOutcome fault_outcome = amdgpu::VmAccessOutcome::Complete;
  pipeline.set_fault_handler([&](amdgpu::Wavefront &, amdgpu::VmAccessOutcome outcome) {
    ++fault_count;
    fault_outcome = outcome;
  });

  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(store)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
  ASSERT_FALSE(context.wf->wait_counters().empty());

  context.wf->set_scratch_base(0);
  pipeline.tick();

  EXPECT_TRUE(pipeline.empty());
  EXPECT_TRUE(context.wf->wait_counters().empty());
  EXPECT_EQ(fault_count, 1u);
  EXPECT_EQ(fault_outcome, amdgpu::VmAccessOutcome::Faulted);
}

TEST(GpuVmPipeline, CompletionRetryOnOneQueueDoesNotStarveAnotherQueue) {
  Gfx1250Sim sim;
  auto external = std::make_shared<TestExternalAddressSpace>();
  constexpr uint32_t kProcessId = 77;
  const amdgpu::AddressSpaceHandle address_space =
      sim.soc->gpu_vm().register_translated(kProcessId, external, external);
  ASSERT_TRUE(address_space);

  constexpr uint64_t kRingAddress = 0x1000;
  constexpr uint64_t kReadPointerAddress = 0x2000;
  constexpr uint64_t kWritePointerAddress = 0x2008;
  constexpr uint64_t kDoorbellAddress = 0x2010;
  constexpr uint64_t kSignalAddress = 0x3000;
  constexpr uint64_t kKernelAddress = 0x5000;
  constexpr uint64_t kStartTimestampOffset = 32;
  constexpr uint32_t kStalledQueueId = 1;
  constexpr uint32_t kRunnableQueueId = 2;
  constexpr uint32_t kEndpgm = S_ENDPGM_GFX12;

  ASSERT_EQ(sim.write_kernel(kKernelAddress, &kEndpgm, 1), kKernelAddress);
  std::array<std::byte, sizeof(rocr::llvm::amdhsa::kernel_descriptor_t) + sizeof(kEndpgm)>
      kernel_image{};
  sim.memory->read_block(
      kKernelAddress,
      std::span<uint8_t>(reinterpret_cast<uint8_t *>(kernel_image.data()), kernel_image.size()));
  ASSERT_EQ(external->write(amdgpu::VmMemoryDomain::System, kKernelAddress, kernel_image),
            amdgpu::VmAccessOutcome::Complete);

  hsa_kernel_dispatch_packet_t stalled_packet{};
  stalled_packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  stalled_packet.setup = 1;
  stalled_packet.workgroup_size_x = 32;
  stalled_packet.workgroup_size_y = 1;
  stalled_packet.workgroup_size_z = 1;
  stalled_packet.grid_size_x = 32;
  stalled_packet.grid_size_y = 1;
  stalled_packet.grid_size_z = 1;
  stalled_packet.kernel_object = kKernelAddress;
  stalled_packet.completion_signal.handle = kSignalAddress;
  external->store(kRingAddress, stalled_packet);
  external->store(kReadPointerAddress, uint64_t{0});
  external->store(kWritePointerAddress, uint64_t{1});
  external->store(kDoorbellAddress, uint64_t{1});
  external->store(kSignalAddress + 8, uint64_t{1});

  amdgpu::AqlQueueConfig stalled_queue{};
  stalled_queue.address_space = address_space;
  stalled_queue.process_id = kProcessId;
  stalled_queue.queue_id = kStalledQueueId;
  stalled_queue.ring_base_va = kRingAddress;
  stalled_queue.ring_size = 64;
  stalled_queue.read_ptr_va = kReadPointerAddress;
  stalled_queue.write_ptr_va = kWritePointerAddress;
  stalled_queue.doorbell_va = kDoorbellAddress;
  stalled_queue.doorbell_mode = amdgpu::QueueDoorbellMode::VmPolled;
  sim.cp()->register_queue(std::move(stalled_queue));

  external->unavailable_atomic_store_address = kSignalAddress + kStartTimestampOffset;
  sim.engine->schedule_event_now(sim.cp()->doorbell_event());
  for (uint32_t step = 0; step < 1000 && external->unavailable_atomic_store_hits == 0; ++step)
    ASSERT_TRUE(sim.engine->step());
  ASSERT_GT(external->unavailable_atomic_store_hits, 0u);
  ASSERT_EQ(sim.cp()->dispatched_count(), 1u);

  constexpr uint64_t kRunnableRingAddress = 0xF1000000;
  test::AqlQueue runnable_queue(sim.memory, sim.cp(), kRunnableRingAddress,
                                test::AqlQueue::DEFAULT_RING_SIZE, kRunnableRingAddress + 0x10000,
                                kRunnableRingAddress + 0x10008, kRunnableRingAddress + 0x10010,
                                false, kRunnableQueueId);
  const uint64_t runnable_kernel = sim.write_kernel(0xF2000000, &kEndpgm, 1);
  runnable_queue.dispatch(runnable_kernel, 32, 32);

  for (uint32_t step = 0; step < 1000 && sim.cp()->dispatched_count() != 2; ++step)
    ASSERT_TRUE(sim.engine->step());
  ASSERT_EQ(sim.cp()->dispatched_count(), 2u);
  for (uint32_t step = 0; step < 1000 && sim.cp()->has_dispatch_for_test(kRunnableQueueId, 0, 2);
       ++step)
    ASSERT_TRUE(sim.engine->step());

  EXPECT_EQ(sim.cp()->accepted_entry_count_for_test(kRunnableQueueId, 0), 1u);
  EXPECT_FALSE(sim.cp()->has_dispatch_for_test(kRunnableQueueId, 0, 2));
  EXPECT_EQ(sim.memory->read64(kRunnableRingAddress + 0x10000), 1u);
  EXPECT_GT(external->unavailable_atomic_store_hits, 1u);
}

TEST(GpuVmPipeline, UnavailableDataRetryDoesNotReplayInstructionPluginCallbacks) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);

  auto plugin_group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto plugin = std::make_unique<MemoryLifecyclePlugin>();
  MemoryLifecyclePlugin *events = plugin.get();
  ASSERT_TRUE(plugin_group->add(std::move(plugin)));
  context.sim.soc->set_plugin_group(plugin_group);

  constexpr uint64_t kCodeAddress = 0x1000;
  constexpr uint64_t kDataAddress = 0x2000;
  constexpr uint32_t kValue = 0x12345678u;
  constexpr uint32_t kDestination = 1;
  constexpr auto load = cdna5::build_vglobal(
      cdna5::kGlobalLoadB32Vglobal, {.saddr = 0, .vdst = kDestination, .vaddr = 0, .ioffset = 0});
  constexpr uint32_t kSNopGfx12 = 0xBF800000u;
  const std::array<uint32_t, 4> code = {load[0], load[1], load[2], kSNopGfx12};
  context.external->store(kCodeAddress, code);
  context.external->store(kDataAddress, kValue);

  context.wf->pc = kCodeAddress;
  context.wf->set_exec(1);
  context.cu->write_sgpr(context.wf->sgpr_alloc().base, static_cast<uint32_t>(kDataAddress));
  context.cu->write_sgpr(context.wf->sgpr_alloc().base + 1,
                         static_cast<uint32_t>(kDataAddress >> 32));
  context.cu->write_vgpr(context.wf->vgpr_alloc().base, 0, 0);
  context.external->unavailable_read_call = 2;

  EXPECT_TRUE(context.cu->step());
  EXPECT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(events->before_count, 1u);
  EXPECT_EQ(events->after_count, 1u);
  EXPECT_EQ(events->route_count, 1u);

  static_cast<void>(context.cu->step());
  EXPECT_EQ(events->before_count, 1u);
  EXPECT_EQ(events->after_count, 1u);
  EXPECT_EQ(events->route_count, 1u);
  EXPECT_EQ(context.external->read_calls, 3u);
  EXPECT_EQ(context.cu->read_vgpr_storage(context.wf->vgpr_alloc().base + kDestination, 0), kValue);
}

TEST(GpuVmPipeline, TranslatedScalarReadAndWriteResumeWithoutReplayingCompletedChunks) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);
  context.external->translation_granule = sizeof(uint32_t);

  constexpr uint64_t kLoadAddress = 0x180;
  constexpr std::array<uint32_t, 2> kLoaded = {0x11223344u, 0x55667788u};
  constexpr uint32_t kDestination = 8;
  context.external->store(kLoadAddress, kLoaded[0]);
  context.external->store(kLoadAddress + sizeof(uint32_t), kLoaded[1]);
  context.external->unavailable_read_call = 2;

  auto load = std::make_unique<amdgpu::ScalarMemState>();
  load->addr = kLoadAddress;
  load->dst_register = {amdgpu::ScalarRegisterStorage::SGPR, kDestination, 2};
  load->num_dwords = 2;
  load->is_load = true;
  amdgpu::ScalarMemPipeline pipeline(&context.cu->l1_scalar());
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(load)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(context.cu->read_sgpr_storage(context.wf->sgpr_alloc().base + kDestination), 0u);

  pipeline.tick();

  EXPECT_EQ(context.wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_TRUE(context.wf->wait_counters().empty());
  EXPECT_EQ(context.cu->read_sgpr_storage(context.wf->sgpr_alloc().base + kDestination),
            kLoaded[0]);
  EXPECT_EQ(context.cu->read_sgpr_storage(context.wf->sgpr_alloc().base + kDestination + 1),
            kLoaded[1]);
  ASSERT_EQ(context.external->read_history.size(), 3u);
  EXPECT_EQ(context.external->read_history[0].address, kLoadAddress);
  EXPECT_EQ(context.external->read_history[1].address, kLoadAddress + sizeof(uint32_t));
  EXPECT_EQ(context.external->read_history[2].address, kLoadAddress + sizeof(uint32_t));

  constexpr uint64_t kStoreAddress = 0x1c0;
  constexpr std::array<uint32_t, 2> kStored = {0xa1b2c3d4u, 0xe5f60718u};
  context.external->unavailable_write_call = 2;
  auto store = std::make_unique<amdgpu::ScalarMemState>();
  store->addr = kStoreAddress;
  store->num_dwords = 2;
  store->is_load = false;
  std::copy(kStored.begin(), kStored.end(), store->store_data);
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(store)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);

  pipeline.tick();

  EXPECT_EQ(context.wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_TRUE(context.wf->wait_counters().empty());
  EXPECT_EQ(context.external->load<uint32_t>(kStoreAddress), kStored[0]);
  EXPECT_EQ(context.external->load<uint32_t>(kStoreAddress + sizeof(uint32_t)), kStored[1]);
  ASSERT_EQ(context.external->write_history.size(), 3u);
  EXPECT_EQ(context.external->write_history[0].address, kStoreAddress);
  EXPECT_EQ(context.external->write_history[1].address, kStoreAddress + sizeof(uint32_t));
  EXPECT_EQ(context.external->write_history[2].address, kStoreAddress + sizeof(uint32_t));
}

TEST(GpuVmPipeline, DeferredOperationCannotWriteBackIntoReusedWaveSlot) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);

  constexpr uint64_t kAddress = 0x1f0;
  constexpr uint32_t kExternalValue = 0x12345678u;
  constexpr uint32_t kDestination = 10;
  constexpr uint32_t kSentinel = 0xa5a55a5au;
  context.external->store(kAddress, kExternalValue);
  context.external->unavailable_read_call = 1;

  auto load = std::make_unique<amdgpu::ScalarMemState>();
  load->addr = kAddress;
  load->dst_register = {amdgpu::ScalarRegisterStorage::SGPR, kDestination, 1};
  load->num_dwords = 1;
  load->is_load = true;
  amdgpu::ScalarMemPipeline pipeline(&context.cu->l1_scalar());
  const uint64_t old_generation = context.wf->dispatch_generation();
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(load)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);

  context.cu->free_wavefront_resources(*context.wf);
  amdgpu::Wavefront *replacement = context.cu->dispatch_wf(1, 0, kGfx1250ScalarSlots, 32);
  ASSERT_EQ(replacement, context.wf);
  ASSERT_NE(replacement->dispatch_generation(), old_generation);
  replacement->set_process_id(77);
  replacement->set_address_space(context.address_space);
  context.cu->write_sgpr(replacement->sgpr_alloc().base + kDestination, kSentinel);

  pipeline.tick();

  EXPECT_TRUE(pipeline.empty());
  EXPECT_EQ(context.cu->read_sgpr_storage(replacement->sgpr_alloc().base + kDestination),
            kSentinel);
  EXPECT_TRUE(replacement->wait_counters().empty());
  EXPECT_EQ(replacement->state(), amdgpu::WfState::RUNNING);
}

TEST(GpuVmPipeline, UnreadyGartRemainsOnTranslatedPathUntilPublication) {
  Gfx1250Sim sim;
  auto external = std::make_shared<TestExternalAddressSpace>();
  amdgpu::ComputeUnitCore *cu = sim.cu();
  const amdgpu::AddressSpaceHandle address_space =
      sim.soc->gpu_vm().initialize_gart_address_space();
  ASSERT_TRUE(address_space);
  const auto initial_info = sim.soc->gpu_vm().lookup(address_space);
  ASSERT_TRUE(initial_info);
  EXPECT_FALSE(initial_info->legacy_cache_compatible);
  EXPECT_FALSE(initial_info->ready);
  cu->set_gpu_vm(&sim.soc->gpu_vm());
  amdgpu::Wavefront *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_address_space(address_space);

  constexpr uint64_t kPageTable = 0x1000;
  constexpr uint64_t kAperture = 0x10000;
  constexpr uint64_t kPhysicalPage = 0x9000;
  constexpr uint64_t kValidSystemPte = (uint64_t{1} << 0) | (uint64_t{1} << 1);
  constexpr uint32_t kExternalValue = 0x12345678u;
  constexpr uint32_t kLegacyValue = 0xdeadbeefu;
  constexpr uint32_t kSentinel = 0xa5a55a5au;
  constexpr uint32_t kDestination = 12;
  sim.memory->write32(kAperture, kLegacyValue);
  cu->write_sgpr(wf->sgpr_alloc().base + kDestination, kSentinel);

  auto load = std::make_unique<amdgpu::ScalarMemState>();
  load->addr = kAperture;
  load->dst_register = {amdgpu::ScalarRegisterStorage::SGPR, kDestination, 1};
  load->num_dwords = 1;
  load->is_load = true;
  amdgpu::ScalarMemPipeline pipeline(&cu->l1_scalar());
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(load)), *wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(cu->read_sgpr_storage(wf->sgpr_alloc().base + kDestination), kSentinel);
  EXPECT_EQ(external->read_calls, 0u);

  external->store(kPageTable, kPhysicalPage | kValidSystemPte);
  external->store(kPhysicalPage, kExternalValue);
  ASSERT_TRUE(sim.soc->gpu_vm().publish_gart({.page_table_base = kPageTable,
                                              .aperture_start = kAperture,
                                              .aperture_end = kAperture + 0xfff},
                                             external));
  pipeline.tick();

  EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_TRUE(wf->wait_counters().empty());
  EXPECT_EQ(cu->read_sgpr_storage(wf->sgpr_alloc().base + kDestination), kExternalValue);
  EXPECT_EQ(sim.memory->read32(kAperture), kLegacyValue);
}

TEST(GpuVmPipeline, TranslatedVectorLoadAndStoreCoalesceInExternalBacking) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);

  constexpr uint64_t kAddress = 0x200;
  constexpr uint32_t kDestination = 8;
  constexpr std::array<uint32_t, 4> kLoaded = {0x10111213u, 0x20212223u, 0x30313233u, 0x40414243u};
  constexpr std::array<uint32_t, 4> kStored = {0xa1a2a3a4u, 0xb1b2b3b4u, 0xc1c2c3c4u, 0xd1d2d3d4u};
  for (uint32_t index = 0; index < kLoaded.size(); ++index)
    context.external->store(kAddress + index * sizeof(uint32_t), kLoaded[index]);

  auto load = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  load->elem_size = sizeof(uint32_t);
  load->num_elems = 2;
  load->is_load = true;
  load->wf_size = context.wf->wf_size();
  load->exec_mask = 0x3;
  load->lane_mask = 0x3;
  load->dst_reg_base = context.wf->vgpr_alloc().base + kDestination;
  load->per_lane_addr[0] = kAddress;
  load->per_lane_addr[1] = kAddress + 2 * sizeof(uint32_t);
  amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
  EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(load)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);

  for (uint32_t lane = 0; lane < 2; ++lane)
    for (uint32_t element = 0; element < 2; ++element)
      EXPECT_EQ(context.cu->read_vgpr(context.wf->vgpr_alloc().base + kDestination + element, lane),
                kLoaded[lane * 2 + element]);
  EXPECT_EQ(context.external->read_calls, 1u);
  EXPECT_EQ(context.external->last_read.address, kAddress);
  EXPECT_EQ(context.external->last_read.size, sizeof(kLoaded));

  auto store = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  store->elem_size = sizeof(uint32_t);
  store->num_elems = 2;
  store->is_load = false;
  store->wf_size = context.wf->wf_size();
  store->exec_mask = 0x3;
  store->lane_mask = 0x3;
  store->per_lane_addr[0] = kAddress;
  store->per_lane_addr[1] = kAddress + 2 * sizeof(uint32_t);
  store->store_data.resize(context.wf->wf_size() * 2 * sizeof(uint32_t));
  std::memcpy(store->store_data.data(), kStored.data(), sizeof(kStored));
  EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(store)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);

  for (uint32_t index = 0; index < kStored.size(); ++index)
    EXPECT_EQ(context.external->load<uint32_t>(kAddress + index * sizeof(uint32_t)),
              kStored[index]);
  EXPECT_EQ(context.external->write_calls, 1u);
  EXPECT_EQ(context.external->last_write.address, kAddress);
  EXPECT_EQ(context.external->last_write.size, sizeof(kStored));
}

TEST(GpuVmPipeline, TranslatedVectorStoreRetrySkipsCompletedRequests) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);

  constexpr uint64_t kFirstAddress = 0x240;
  constexpr uint64_t kSecondAddress = 0x340;
  constexpr std::array<uint32_t, 2> kStored = {0x10203040u, 0x50607080u};
  context.external->unavailable_write_call = 2;

  auto store = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  store->elem_size = sizeof(uint32_t);
  store->num_elems = 1;
  store->is_load = false;
  store->wf_size = context.wf->wf_size();
  store->exec_mask = 0x3;
  store->lane_mask = 0x3;
  store->per_lane_addr[0] = kFirstAddress;
  store->per_lane_addr[1] = kSecondAddress;
  store->store_data.resize(context.wf->wf_size() * sizeof(uint32_t));
  std::memcpy(store->store_data.data(), kStored.data(), sizeof(kStored));

  amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(store)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(context.external->load<uint32_t>(kFirstAddress), kStored[0]);
  EXPECT_EQ(context.external->load<uint32_t>(kSecondAddress), 0u);

  pipeline.tick();

  EXPECT_EQ(context.wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_TRUE(context.wf->wait_counters().empty());
  EXPECT_EQ(context.external->load<uint32_t>(kFirstAddress), kStored[0]);
  EXPECT_EQ(context.external->load<uint32_t>(kSecondAddress), kStored[1]);
  ASSERT_EQ(context.external->write_history.size(), 3u);
  EXPECT_EQ(context.external->write_history[0].address, kFirstAddress);
  EXPECT_EQ(context.external->write_history[1].address, kSecondAddress);
  EXPECT_EQ(context.external->write_history[2].address, kSecondAddress);
}

TEST(GpuVmPipeline, TranslatedAtomicsUseStrongBackingOperationsForBothWidths) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);

  struct AtomicCase {
    amdgpu::AtomicOp operation;
    uint32_t width;
    uint64_t address;
    uint64_t initial;
    uint64_t source;
    uint64_t compare;
    uint64_t expected;
  };
  constexpr std::array<AtomicCase, 4> kCases = {{
      {amdgpu::AtomicOp::ADD, 4, 0x300, 7, 5, 0, 12},
      {amdgpu::AtomicOp::ADD, 8, 0x308, 0x1'0000'0000ULL, 9, 0, 0x1'0000'0009ULL},
      {amdgpu::AtomicOp::CMPSWAP, 4, 0x310, 0x11223344, 0xaabbccdd, 0x11223344, 0xaabbccdd},
      {amdgpu::AtomicOp::CMPSWAP, 8, 0x318, 0x1122334455667788ULL, 0xaabbccddeeff0011ULL,
       0x1122334455667788ULL, 0xaabbccddeeff0011ULL},
  }};

  amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
  for (uint32_t index = 0; index < kCases.size(); ++index) {
    const AtomicCase &test = kCases[index];
    if (test.width == sizeof(uint32_t))
      context.external->store(test.address, static_cast<uint32_t>(test.initial));
    else
      context.external->store(test.address, test.initial);

    auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
    state->elem_size = test.width;
    state->num_elems = 1;
    state->is_load = true;
    state->atomic_op = test.operation;
    state->wf_size = context.wf->wf_size();
    state->exec_mask = 1;
    state->lane_mask = 1;
    state->dst_reg_base = context.wf->vgpr_alloc().base + 12 + index * 2;
    state->per_lane_addr[0] = test.address;
    const uint32_t source_stride =
        test.operation == amdgpu::AtomicOp::CMPSWAP ? test.width * 2 : test.width;
    state->store_data.resize(context.wf->wf_size() * source_stride);
    std::memcpy(state->store_data.data(), &test.source, test.width);
    if (test.operation == amdgpu::AtomicOp::CMPSWAP)
      std::memcpy(state->store_data.data() + test.width, &test.compare, test.width);

    EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(state)), *context.wf),
              amdgpu::VmAccessOutcome::Complete);
    if (test.width == sizeof(uint32_t)) {
      EXPECT_EQ(context.external->load<uint32_t>(test.address),
                static_cast<uint32_t>(test.expected));
      EXPECT_EQ(context.cu->read_vgpr(context.wf->vgpr_alloc().base + 12 + index * 2, 0),
                static_cast<uint32_t>(test.initial));
    } else {
      EXPECT_EQ(context.external->load<uint64_t>(test.address), test.expected);
      const uint64_t returned =
          context.cu->read_vgpr(context.wf->vgpr_alloc().base + 12 + index * 2, 0) |
          (static_cast<uint64_t>(
               context.cu->read_vgpr(context.wf->vgpr_alloc().base + 13 + index * 2, 0))
           << 32);
      EXPECT_EQ(returned, test.initial);
    }
  }

  EXPECT_EQ(context.external->read_calls, 0u);
  EXPECT_EQ(context.external->write_calls, 0u);
  EXPECT_EQ(context.external->atomic_load_calls, kCases.size());
  EXPECT_EQ(context.external->compare_exchange_calls, kCases.size());
}

TEST(GpuVmPipeline, UnavailableAtomicDoesNotCompleteArchitecturalState) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);

  constexpr uint64_t kAddress = 0x400;
  constexpr uint32_t kInitial = 10;
  constexpr uint32_t kSource = 7;
  constexpr uint32_t kDestination = 24;
  constexpr uint32_t kSentinel = 0xfeedface;
  context.external->store(kAddress, kInitial);
  context.external->next_atomic_load_outcome = amdgpu::VmAccessOutcome::Unavailable;
  context.cu->write_vgpr(context.wf->vgpr_alloc().base + kDestination, 0, kSentinel);

  auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  state->elem_size = sizeof(uint32_t);
  state->num_elems = 1;
  state->is_load = true;
  state->atomic_op = amdgpu::AtomicOp::ADD;
  state->wf_size = context.wf->wf_size();
  state->exec_mask = 1;
  state->lane_mask = 1;
  state->dst_reg_base = context.wf->vgpr_alloc().base + kDestination;
  state->per_lane_addr[0] = kAddress;
  state->store_data.resize(context.wf->wf_size() * sizeof(uint32_t));
  std::memcpy(state->store_data.data(), &kSource, sizeof(kSource));

  amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
  EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(state)), *context.wf),
            amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_EQ(context.external->load<uint32_t>(kAddress), kInitial);
  EXPECT_EQ(context.cu->read_vgpr(context.wf->vgpr_alloc().base + kDestination, 0), kSentinel);
  EXPECT_EQ(context.external->compare_exchange_calls, 0u);
  EXPECT_TRUE(context.wf->wait_counters().empty());
}

TEST(GpuVmPipeline, TranslatedAtomicRetrySkipsCompletedLanes) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);

  constexpr uint64_t kFirstAddress = 0x420;
  constexpr uint64_t kSecondAddress = 0x428;
  constexpr std::array<uint32_t, 2> kInitial = {10, 20};
  constexpr std::array<uint32_t, 2> kSource = {1, 2};
  constexpr uint32_t kDestination = 30;
  context.external->store(kFirstAddress, kInitial[0]);
  context.external->store(kSecondAddress, kInitial[1]);
  context.external->unavailable_atomic_load_call = 2;

  auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  state->elem_size = sizeof(uint32_t);
  state->num_elems = 1;
  state->is_load = true;
  state->atomic_op = amdgpu::AtomicOp::ADD;
  state->wf_size = context.wf->wf_size();
  state->exec_mask = 0x3;
  state->lane_mask = 0x3;
  state->dst_reg_base = context.wf->vgpr_alloc().base + kDestination;
  state->per_lane_addr[0] = kFirstAddress;
  state->per_lane_addr[1] = kSecondAddress;
  state->store_data.resize(context.wf->wf_size() * sizeof(uint32_t));
  std::memcpy(state->store_data.data(), kSource.data(), sizeof(kSource));

  amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(state)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(context.external->load<uint32_t>(kFirstAddress), kInitial[0] + kSource[0]);
  EXPECT_EQ(context.external->load<uint32_t>(kSecondAddress), kInitial[1]);

  pipeline.tick();

  EXPECT_EQ(context.wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_TRUE(context.wf->wait_counters().empty());
  EXPECT_EQ(context.external->load<uint32_t>(kFirstAddress), kInitial[0] + kSource[0]);
  EXPECT_EQ(context.external->load<uint32_t>(kSecondAddress), kInitial[1] + kSource[1]);
  EXPECT_EQ(context.external->atomic_load_calls, 3u);
  EXPECT_EQ(context.external->compare_exchange_calls, 2u);
}

TEST(GpuVmPipeline, TranslatedTensorLoadRetriesBeforeCommittingLdsOrBarrier) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);
  context.wf->set_lds_base(context.cu->allocate_lds(256));

  constexpr uint64_t kGlobalAddress = 0x500;
  constexpr uint32_t kBarrierAddress = 64;
  constexpr std::array<uint32_t, 2> kExternal = {0x11223344u, 0x55667788u};
  constexpr std::array<uint32_t, 2> kLegacy = {0xdeadbeefu, 0xcafef00du};
  constexpr uint32_t kLdsSentinel = 0xa5a55a5au;
  for (uint32_t index = 0; index < kExternal.size(); ++index) {
    context.external->store(kGlobalAddress + index * sizeof(uint32_t), kExternal[index]);
    context.sim.memory->write32(kGlobalAddress + index * sizeof(uint32_t), kLegacy[index]);
    context.wf->lds().write32(context.wf->lds_base() + index * sizeof(uint32_t), kLdsSentinel);
  }
  context.wf->lds().write64(context.wf->lds_base() + kBarrierAddress, 0);
  context.external->unavailable_read_call = 2;

  amdgpu::tensor_dma_detail::TensorDmaDescriptor descriptor;
  descriptor.elem_size = sizeof(uint32_t);
  descriptor.atomic_barrier = true;
  descriptor.atomic_barrier_addr = kBarrierAddress;
  auto state = std::make_unique<amdgpu::tensor_dma_detail::TensorDmaState>(descriptor, false);
  state->access = context.wf->snapshot_vm_access();
  ASSERT_TRUE(state->access);
  for (uint32_t index = 0; index < kExternal.size(); ++index) {
    state->elements.push_back(
        {.global_address = kGlobalAddress + index * sizeof(uint32_t),
         .lds_address = context.wf->lds_base() + index * static_cast<uint32_t>(sizeof(uint32_t)),
         .in_bounds = true});
  }

  amdgpu::TensorDmaPipeline pipeline;
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(state)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(context.wf->lds().read32(context.wf->lds_base()), kLdsSentinel);
  EXPECT_EQ(context.wf->lds().read32(context.wf->lds_base() + sizeof(uint32_t)), kLdsSentinel);
  EXPECT_EQ(context.wf->lds().read64(context.wf->lds_base() + kBarrierAddress), 0u);

  pipeline.tick();

  EXPECT_EQ(context.wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_TRUE(context.wf->wait_counters().empty());
  EXPECT_EQ(context.wf->lds().read32(context.wf->lds_base()), kExternal[0]);
  EXPECT_EQ(context.wf->lds().read32(context.wf->lds_base() + sizeof(uint32_t)), kExternal[1]);
  EXPECT_EQ(context.sim.memory->read32(kGlobalAddress), kLegacy[0]);
  const uint64_t barrier = context.wf->lds().read64(context.wf->lds_base() + kBarrierAddress);
  EXPECT_EQ(barrier, amdgpu::kLdsBarrierCellPhaseMask << amdgpu::kLdsBarrierCellPhaseShift);
  EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(barrier));
  ASSERT_EQ(context.external->read_history.size(), 3u);
  EXPECT_EQ(context.external->read_history[0].address, kGlobalAddress);
  EXPECT_EQ(context.external->read_history[1].address, kGlobalAddress + sizeof(uint32_t));
  EXPECT_EQ(context.external->read_history[2].address, kGlobalAddress + sizeof(uint32_t));
}

TEST(GpuVmPipeline, TranslatedTensorStoreRetriesWithoutReplayingOrResnapshottingLds) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);
  context.wf->set_lds_base(context.cu->allocate_lds(256));

  constexpr uint64_t kGlobalAddress = 0x600;
  constexpr uint32_t kBarrierAddress = 64;
  constexpr std::array<uint32_t, 2> kStored = {0x10203040u, 0x50607080u};
  context.wf->lds().write64(context.wf->lds_base() + kBarrierAddress, 0);
  context.external->unavailable_write_call = 2;

  amdgpu::tensor_dma_detail::TensorDmaDescriptor descriptor;
  descriptor.elem_size = sizeof(uint32_t);
  descriptor.atomic_barrier = true;
  descriptor.atomic_barrier_addr = kBarrierAddress;
  auto state = std::make_unique<amdgpu::tensor_dma_detail::TensorDmaState>(descriptor, true);
  state->access = context.wf->snapshot_vm_access();
  ASSERT_TRUE(state->access);
  for (uint32_t index = 0; index < kStored.size(); ++index) {
    amdgpu::tensor_dma_detail::TensorDmaTransferElement element{
        .global_address = kGlobalAddress + index * sizeof(uint32_t),
        .lds_address = context.wf->lds_base() + index * static_cast<uint32_t>(sizeof(uint32_t)),
        .in_bounds = true};
    std::memcpy(element.bytes.data(), &kStored[index], sizeof(uint32_t));
    state->elements.push_back(element);
  }

  amdgpu::TensorDmaPipeline pipeline;
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(state)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(context.external->load<uint32_t>(kGlobalAddress), kStored[0]);
  EXPECT_EQ(context.external->load<uint32_t>(kGlobalAddress + sizeof(uint32_t)), 0u);
  context.wf->lds().write32(context.wf->lds_base(), 0xffffffffu);
  context.wf->lds().write32(context.wf->lds_base() + sizeof(uint32_t), 0xffffffffu);

  pipeline.tick();

  EXPECT_EQ(context.wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_TRUE(context.wf->wait_counters().empty());
  EXPECT_EQ(context.external->load<uint32_t>(kGlobalAddress), kStored[0]);
  EXPECT_EQ(context.external->load<uint32_t>(kGlobalAddress + sizeof(uint32_t)), kStored[1]);
  const uint64_t barrier = context.wf->lds().read64(context.wf->lds_base() + kBarrierAddress);
  EXPECT_EQ(barrier, amdgpu::kLdsBarrierCellPhaseMask << amdgpu::kLdsBarrierCellPhaseShift);
  EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(barrier));
  ASSERT_EQ(context.external->write_history.size(), 3u);
  EXPECT_EQ(context.external->write_history[0].address, kGlobalAddress);
  EXPECT_EQ(context.external->write_history[1].address, kGlobalAddress + sizeof(uint32_t));
  EXPECT_EQ(context.external->write_history[2].address, kGlobalAddress + sizeof(uint32_t));
}

TEST(GpuVmPipeline, TranslatedAtomicRetriesCompareExchangeContention) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);

  constexpr uint64_t kAddress = 0x408;
  constexpr uint64_t kInitial = 10;
  constexpr uint64_t kSource = 5;
  constexpr uint32_t kDestination = 26;
  context.external->store(kAddress, kInitial);
  context.external->forced_compare_exchange_misses = 1;

  auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  state->elem_size = sizeof(uint64_t);
  state->num_elems = 1;
  state->is_load = true;
  state->atomic_op = amdgpu::AtomicOp::ADD;
  state->wf_size = context.wf->wf_size();
  state->exec_mask = 1;
  state->lane_mask = 1;
  state->dst_reg_base = context.wf->vgpr_alloc().base + kDestination;
  state->per_lane_addr[0] = kAddress;
  state->store_data.resize(context.wf->wf_size() * sizeof(uint64_t));
  std::memcpy(state->store_data.data(), &kSource, sizeof(kSource));

  amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
  EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(state)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(context.external->load<uint64_t>(kAddress), kInitial + 1 + kSource);
  const uint64_t returned = context.cu->read_vgpr(context.wf->vgpr_alloc().base + kDestination, 0) |
                            (static_cast<uint64_t>(context.cu->read_vgpr(
                                 context.wf->vgpr_alloc().base + kDestination + 1, 0))
                             << 32);
  EXPECT_EQ(returned, kInitial + 1);
  EXPECT_EQ(context.external->atomic_load_calls, 1u);
  EXPECT_EQ(context.external->compare_exchange_calls, 2u);
  EXPECT_EQ(context.external->read_calls, 0u);
  EXPECT_EQ(context.external->write_calls, 0u);
}

TEST(GpuVmPipeline, AtomicPermissionFaultDoesNotReachPhysicalBacking) {
  TranslatedPipelineContext context;
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);

  constexpr uint64_t kAddress = 0x418;
  constexpr uint32_t kInitial = 10;
  constexpr uint32_t kSource = 7;
  constexpr uint32_t kDestination = 28;
  constexpr uint32_t kSentinel = 0x1234abcd;
  context.external->store(kAddress, kInitial);
  context.external->atomic_allowed_ = false;
  context.cu->write_vgpr(context.wf->vgpr_alloc().base + kDestination, 0, kSentinel);

  auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  state->elem_size = sizeof(uint32_t);
  state->num_elems = 1;
  state->is_load = true;
  state->atomic_op = amdgpu::AtomicOp::ADD;
  state->wf_size = context.wf->wf_size();
  state->exec_mask = 1;
  state->lane_mask = 1;
  state->dst_reg_base = context.wf->vgpr_alloc().base + kDestination;
  state->per_lane_addr[0] = kAddress;
  state->store_data.resize(context.wf->wf_size() * sizeof(uint32_t));
  std::memcpy(state->store_data.data(), &kSource, sizeof(kSource));

  amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
  EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(state)), *context.wf),
            amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(context.external->load<uint32_t>(kAddress), kInitial);
  EXPECT_EQ(context.cu->read_vgpr(context.wf->vgpr_alloc().base + kDestination, 0), kSentinel);
  EXPECT_EQ(context.external->atomic_load_calls, 0u);
  EXPECT_EQ(context.external->compare_exchange_calls, 0u);
  EXPECT_TRUE(context.wf->wait_counters().empty());
}

} // namespace
