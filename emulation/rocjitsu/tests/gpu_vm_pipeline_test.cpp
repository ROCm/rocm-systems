// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"

#include "rocjitsu/isa/arch/amdgpu/shared/tensor_dma.h"
#include "rocjitsu/kmd/linux/host_mapping_lock.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/image_metadata.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unistd.h>
#include <utility>
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
    std::ranges::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(address), bytes.size(),
                        bytes.begin());
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
    std::ranges::copy(bytes, bytes_.begin() + static_cast<std::ptrdiff_t>(address));
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

  amdgpu::VmAccessOutcome atomic_modify(amdgpu::VmMemoryDomain domain, uint64_t address,
                                        uint32_t width, const AtomicMutation &mutation) override {
    std::lock_guard lock(mutex_);
    ++atomic_modify_calls;
    if (atomic_modify_calls == unavailable_atomic_modify_call)
      return amdgpu::VmAccessOutcome::Unavailable;
    if (!valid_atomic(domain, address, width))
      return amdgpu::VmAccessOutcome::Faulted;
    mutation(std::span(bytes_).subspan(address, width));
    return amdgpu::VmAccessOutcome::Complete;
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

  std::unique_ptr<amdgpu::VmRamLeaseRequest>
  prepare_ram_lease(amdgpu::PhysicalMemoryAccess &,
                    std::span<const amdgpu::VmRamRange>) const override {
    ++prepare_ram_calls;
    return nullptr;
  }

  mutable uint32_t prepare_ram_calls = 0;
  bool atomic_allowed_ = true;
  amdgpu::VmAccessOutcome next_atomic_load_outcome = amdgpu::VmAccessOutcome::Complete;
  uint64_t translation_granule = 0;
  uint32_t unavailable_read_call = 0;
  uint32_t unavailable_write_call = 0;
  uint32_t unavailable_atomic_load_call = 0;
  uint32_t read_calls = 0;
  uint32_t write_calls = 0;
  uint32_t atomic_load_calls = 0;
  uint32_t atomic_modify_calls = 0;
  uint32_t unavailable_atomic_modify_call = 0;
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
           (width == sizeof(uint16_t) || width == sizeof(uint32_t) || width == sizeof(uint64_t)) &&
           address % width == 0 && contains(address, width);
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

  explicit TranslatedPipelineContext(bool legacy_cache_compatible = false) {
    address_space = sim.soc->gpu_vm().register_address_space(77, external, external, {},
                                                             legacy_cache_compatible);
    cu->set_gpu_vm(&sim.soc->gpu_vm());
    wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    if (wf != nullptr) {
      wf->set_process_id(77);
      wf->set_address_space(address_space);
      wf->set_exec(0x3u);
    }
  }
};

class InstructionVmPlugin final : public ExecutionPlugin {
public:
  InstructionVmPlugin() : ExecutionPlugin("instruction-vm") {}
  void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &,
                                        amdgpu::Wavefront &wf) override {
    if (before_instruction)
      before_instruction(wf);
  }
  void onAmdgpuAfterExecuteInstruction(uint64_t, const Instruction &inst,
                                       amdgpu::Wavefront &wf) override {
    if (inst.mnemonic() == "s_nop")
      ++nops;
    if (after_instruction)
      after_instruction(wf);
  }
  uint32_t nops = 0;
  std::function<void(amdgpu::Wavefront &)> before_instruction;
  std::function<void(amdgpu::Wavefront &)> after_instruction;
};

TEST(GpuVmPipeline, NestedStepPreservesOuterInstructionSnapshot) {
  TranslatedPipelineContext context;
  ASSERT_NE(context.wf, nullptr);
  constexpr uint64_t kCode = 0x1000, kData = 0x2000;
  constexpr uint32_t kValue = 0x12345678, kDestination = 1;
  constexpr auto load = cdna5::build_vglobal(
      cdna5::kGlobalLoadB32Vglobal, {.saddr = 0, .vdst = kDestination, .vaddr = 0, .ioffset = 0});
  context.external->store(kCode, std::array{load[0], load[1], load[2], S_ENDPGM_GFX12});
  context.external->store(kData, kValue);
  // A nested fetch fault must not replace the snapshot used to probe the
  // outer instruction's data access after its plugin callback returns.
  auto nested_backing = std::make_shared<TestExternalAddressSpace>(kData);
  const auto nested_handle =
      context.sim.soc->gpu_vm().register_translated(88, nested_backing, nested_backing);
  auto *nested = context.cu->dispatch_wf(1, kData, kGfx1250ScalarSlots, 32);
  ASSERT_NE(nested, nullptr);
  nested->set_dispatch_id(1);
  nested->set_process_id(88);
  nested->set_address_space(nested_handle);
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto plugin = std::make_unique<InstructionVmPlugin>();
  bool entered = false;
  plugin->before_instruction = [&](amdgpu::Wavefront &wf) {
    if (&wf != context.wf || std::exchange(entered, true))
      return;
    wf.set_debug_halted(true);
    static_cast<void>(context.cu->step());
    wf.set_debug_halted(false);
  };
  plugin->after_instruction = [&](amdgpu::Wavefront &wf) {
    if (&wf == context.wf)
      context.cu->request_functional_yield();
  };
  ASSERT_TRUE(group->add(std::move(plugin)));
  context.sim.soc->set_plugin_group(group);
  uint32_t faults = 0;
  context.cu->set_debug_active(true);
  context.cu->set_memory_violation_handler([&](amdgpu::Wavefront &wf, uint64_t, bool) {
    if (&wf != context.wf)
      return false;
    ++faults;
    return true;
  });
  context.wf->pc = kCode;
  context.wf->set_exec(1);
  context.cu->write_sgpr(context.wf->sgpr_alloc().base, kData);
  context.cu->write_sgpr(context.wf->sgpr_alloc().base + 1, 0);
  context.cu->write_vgpr(context.wf->vgpr_alloc().base, 0, 0);
  context.cu->set_functional_quantum(1);
  static_cast<void>(context.cu->run_quantum());
  EXPECT_TRUE(entered);
  EXPECT_EQ(faults, 0u);
  EXPECT_EQ(context.cu->read_vgpr_storage(context.wf->vgpr_alloc().base + kDestination, 0), kValue);
}

TEST(GpuVmPipeline, QuantumInstructionFetchTracksAddressSpaceChanges) {
  enum class Change { ReplaceRoot, Invalidate, Unregister, ExplicitHandle, NumericVmid };
  for (Change change : {Change::ReplaceRoot, Change::Invalidate, Change::Unregister,
                        Change::ExplicitHandle, Change::NumericVmid}) {
    SCOPED_TRACE(static_cast<int>(change));
    TranslatedPipelineContext context;
    ASSERT_NE(context.wf, nullptr);
    auto &vm = context.sim.soc->gpu_vm();
    constexpr uint64_t kCode = 0x1000;
    constexpr uint32_t kNop = 0xBF800000u;
    context.external->store(kCode, std::array{kNop, kNop, kNop, S_ENDPGM_GFX12});
    auto replacement = std::make_shared<TestExternalAddressSpace>();
    replacement->store(kCode, std::array{kNop, S_ENDPGM_GFX12, kNop, kNop});
    // The unrouted handle deliberately shares the routed binding's VMID.
    const auto other = vm.register_unrouted_address_space(77, replacement, replacement);
    ASSERT_TRUE(other);
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto plugin = std::make_unique<InstructionVmPlugin>();
    auto *events = plugin.get();
    bool changed = false;
    events->after_instruction = [&](amdgpu::Wavefront &wf) {
      if (std::exchange(changed, true))
        return;
      switch (change) {
      case Change::ReplaceRoot:
        EXPECT_TRUE(vm.replace_translated(context.address_space, replacement, replacement));
        break;
      case Change::Invalidate:
        context.external->store(kCode + 4, S_ENDPGM_GFX12);
        EXPECT_TRUE(vm.invalidate(context.address_space));
        break;
      case Change::Unregister:
        EXPECT_TRUE(vm.unregister_address_space(context.address_space));
        break;
      case Change::ExplicitHandle:
        wf.set_address_space(other);
        break;
      case Change::NumericVmid:
        // Start on the unrouted handle, then resolve the same numeric VMID.
        context.external->store(kCode + 4, S_ENDPGM_GFX12);
        wf.set_address_space({});
        break;
      }
    };
    ASSERT_TRUE(group->add(std::move(plugin)));
    context.sim.soc->set_plugin_group(group);
    context.wf->pc = kCode;
    if (change == Change::NumericVmid) {
      replacement->store(kCode, std::array{kNop, kNop, kNop, S_ENDPGM_GFX12});
      context.wf->set_address_space(other);
    }
    context.cu->set_functional_quantum(16);
    EXPECT_TRUE(context.cu->run_quantum().ran);
    EXPECT_EQ(events->nops, 1u);
    EXPECT_FALSE(context.cu->has_active_wfs());
  }
}

TEST(GpuVmPipeline, QuantumReleasesInstructionSnapshotOnReturnAndException) {
  for (bool throw_from_plugin : {false, true}) {
    SCOPED_TRACE(throw_from_plugin);
    TranslatedPipelineContext context;
    ASSERT_NE(context.wf, nullptr);
    constexpr uint64_t kCode = 0x1000;
    context.external->store(kCode, std::array<uint32_t, 4>{0xBF800000u, S_ENDPGM_GFX12, 0, 0});
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto plugin = std::make_unique<InstructionVmPlugin>();
    auto *events = plugin.get();
    events->after_instruction = [&](amdgpu::Wavefront &) {
      if (throw_from_plugin)
        throw std::runtime_error("instruction callback");
    };
    ASSERT_TRUE(group->add(std::move(plugin)));
    context.sim.soc->set_plugin_group(group);
    context.wf->pc = kCode;
    context.cu->set_functional_quantum(1);
    if (throw_from_plugin)
      EXPECT_THROW(context.cu->run_quantum(), std::runtime_error);
    else
      EXPECT_TRUE(context.cu->run_quantum().ran);

    std::weak_ptr<TestExternalAddressSpace> old_backing = context.external;
    auto &vm = context.sim.soc->gpu_vm();
    ASSERT_TRUE(vm.unregister_address_space(context.address_space));
    context.external.reset();
    EXPECT_TRUE(old_backing.expired());

    auto replacement = std::make_shared<TestExternalAddressSpace>();
    replacement->store(kCode, std::array<uint32_t, 4>{S_ENDPGM_GFX12, 0, 0, 0});
    context.wf->set_address_space(vm.register_translated(77, replacement, replacement));
    context.wf->pc = kCode;
    events->after_instruction = {};
    static_cast<void>(context.cu->step());
    EXPECT_FALSE(context.cu->has_active_wfs());
  }
}

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

TEST(GpuVmPipeline, ScalarPrivateBatchRequiresUnobservedQuantumAndFullMask) {
  constexpr uint32_t vmid = 79;
  constexpr uint64_t address = 0x4000;
  Gfx1250Sim sim;
  rocjitsu::KfdProcess process(vmid);
  std::array<uint32_t, 8> input{1, 2, 3, 4, 5, 6, 7, 8};
  process.map_pages(address, input.data(), sizeof(input), amdgpu::Mtype::UC,
                    amdgpu::LegacyHostExtentOwner::DriverSealedRam);
  amdgpu::LegacyGpuVmAdapter adapter(sim.soc->gpu_vm(), sim.memory);
  const auto handle = adapter.register_address_space(
      vmid, {.page_table = &process.page_table_,
             .page_table_mutex = &process.page_table_mutex_,
             .page_table_generation = process.page_table_generation(),
             .request_mutex = process.page_table_request_mutex(),
             .mutation_epoch = process.page_table_mutation_epoch(),
             .page_table_cache_state = process.page_table_cache_state()});
  ASSERT_TRUE(handle);
  auto *cu = sim.cu();
  cu->set_gpu_vm(&sim.soc->gpu_vm());
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  ASSERT_NE(cu->l2(), nullptr);
  wf->set_process_id(vmid);
  wf->set_address_space(handle);
  auto observed = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(observed->add(std::make_unique<MemoryLifecyclePlugin>()));
  amdgpu::ScalarMemPipeline pipeline(&cu->l1_scalar());
  struct Case {
    bool quantum, plugin, debug;
    uint16_t mask;
    uint64_t expected_reads;
  };
  // Admit once, then refuse each independent gate with the others permissive.
  constexpr std::array cases{Case{true, false, false, 0xffff, 1},
                             Case{false, false, false, 0xffff, 8},
                             Case{true, true, false, 0xffff, 8}, Case{true, false, true, 0xffff, 8},
                             Case{true, false, false, 0xf, 4}};
  for (const auto &test : cases) {
    SCOPED_TRACE(::testing::Message()
                 << test.quantum << '/' << test.plugin << '/' << test.debug << '/' << test.mask);
    std::optional<amdgpu::GpuVmAccessBatchGuard> guard;
    if (test.quantum)
      guard.emplace();
    cu->set_plugin_group(test.plugin ? observed : nullptr);
    cu->set_debug_active(test.debug);
    auto load = std::make_unique<amdgpu::ScalarMemState>();
    load->addr = address;
    load->dst_register = {amdgpu::ScalarRegisterStorage::SGPR, 8, 8};
    load->num_dwords = 8;
    load->is_load = true;
    load->load_dword_mask = test.mask;
    const uint64_t before = cu->l2()->backing_read_transactions();
    ASSERT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(load)), *wf),
              amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(cu->l2()->backing_read_transactions() - before, test.expected_reads);
    for (uint32_t i = 0; i < input.size(); ++i)
      EXPECT_EQ(cu->read_sgpr_storage(wf->sgpr_alloc().base + 8 + i),
                test.mask & (1u << i) ? input[i] : 0u);
  }
  cu->set_debug_active(false);
  cu->set_plugin_group(nullptr);
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
  std::ranges::copy(kStored, store->store_data);
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
  constexpr uint64_t kValidSystemPte =
      (uint64_t{1} << 0) | (uint64_t{1} << 1) | (uint64_t{1} << 5) | (uint64_t{1} << 63);
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
    bool source_nan_first = false;
  };
  constexpr std::array<AtomicCase, 8> kCases = {{
      {amdgpu::AtomicOp::ADD, 4, 0x300, 7, 5, 0, 12},
      {amdgpu::AtomicOp::ADD, 8, 0x308, 0x1'0000'0000ULL, 9, 0, 0x1'0000'0009ULL},
      {amdgpu::AtomicOp::CMPSWAP, 4, 0x310, 0x11223344, 0xaabbccdd, 0x11223344, 0xaabbccdd},
      {amdgpu::AtomicOp::CMPSWAP, 8, 0x318, 0x1122334455667788ULL, 0xaabbccddeeff0011ULL,
       0x1122334455667788ULL, 0xaabbccddeeff0011ULL},
      {amdgpu::AtomicOp::PK_ADD_F16, 4, 0x320, 0x3c004000, 0x42004400, 0, 0x44004600},
      {amdgpu::AtomicOp::PK_ADD_BF16, 4, 0x324, 0x3f804000, 0x40404080, 0, 0x408040c0},
      {amdgpu::AtomicOp::FADD, 4, 0x328, 0x7f800002, 0xff800004, 0, 0x7fc00002},
      {amdgpu::AtomicOp::FADD, 4, 0x32c, 0x7f800002, 0xff800004, 0, 0xffc00004, true},
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
    state->atomic_source_nan_first = test.source_nan_first;
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

TEST(GpuVmPipeline, LegacyAtomicBatchPreservesLaneResultsOnRetry) {
  for (uint64_t mask : {uint64_t{0}, uint64_t{0xb}}) {
    for (bool retry : {false, true}) {
      SCOPED_TRACE(mask);
      SCOPED_TRACE(retry);
      TranslatedPipelineContext context(/*legacy_cache_compatible=*/true);
      ASSERT_NE(context.wf, nullptr);
      constexpr uint64_t kAddress = 0x420;
      constexpr uint32_t kInitial = 10, kDestination = 30, kSentinel = 0xdeadbeef;
      constexpr std::array<uint32_t, 4> kSource = {1, 2, 3, 4};
      context.external->store(kAddress, kInitial);
      context.external->unavailable_atomic_modify_call = retry ? 2 : 0;
      context.wf->set_exec(mask);
      auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
      state->elem_size = sizeof(uint32_t);
      state->num_elems = 1;
      state->is_load = true;
      state->atomic_op = amdgpu::AtomicOp::ADD;
      state->wf_size = context.wf->wf_size();
      state->exec_mask = mask;
      state->lane_mask = mask;
      state->dst_reg_base = context.wf->vgpr_alloc().base + kDestination;
      state->store_data.resize(state->wf_size * sizeof(uint32_t));
      std::memcpy(state->store_data.data(), kSource.data(), sizeof(kSource));
      for (uint32_t lane = 0; lane < kSource.size(); ++lane) {
        state->per_lane_addr[lane] = kAddress;
        context.cu->write_vgpr(state->dst_reg_base, lane, kSentinel);
      }
      const auto &coherence = context.cu->l2()->coherence_domain();
      const auto epoch = coherence->current_epoch();
      amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
      EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(state)), *context.wf),
                amdgpu::VmAccessOutcome::Complete);
      if (mask && retry) {
        ASSERT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
        EXPECT_EQ(context.external->load<uint32_t>(kAddress), kInitial + kSource[0]);
        pipeline.tick();
      }
      EXPECT_TRUE(context.wf->wait_counters().empty());
      EXPECT_EQ(coherence->current_epoch() - epoch, mask ? (retry ? 2u : 1u) : 0u);
      uint32_t expected = kInitial;
      for (uint32_t lane = 0; lane < kSource.size(); ++lane) {
        const bool active = (mask & (uint64_t{1} << lane)) != 0;
        EXPECT_EQ(context.cu->read_vgpr_storage(context.wf->vgpr_alloc().base + kDestination, lane),
                  active ? expected : kSentinel);
        if (active)
          expected += kSource[lane];
      }
      EXPECT_EQ(context.external->load<uint32_t>(kAddress), expected);
      EXPECT_EQ(context.external->atomic_modify_calls, mask ? (retry ? 4u : 3u) : 0u);
    }
  }
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

namespace {

class MetadataInitiationPipeline final : public amdgpu::GlobalMemPipeline {
public:
  using GlobalMemPipeline::GlobalMemPipeline;
  using GlobalMemPipeline::initiate_access;
};

struct ShaderMetadataContext {
  static constexpr uint64_t metadata_base = 0x100000, pixel_base = 0x200000;
  static constexpr auto sealed = amdgpu::LegacyHostExtentOwner::DriverSealedRam;
  struct Reporter final : amdgpu::MemoryFaultReporter {
    std::vector<std::pair<uint64_t, amdgpu::MemoryFaultCause>> faults;
    std::function<void(uint64_t)> callback;
    void report_memory_fault(uint32_t, uint64_t address, amdgpu::MemoryFaultCause cause) override {
      faults.emplace_back(address, cause);
      if (callback)
        callback(address);
    }
  } reporter;
  Gfx1250Sim sim;
  KfdProcess process{79};
  std::vector<uint8_t> metadata = std::vector<uint8_t>(131072, 0xff);
  std::vector<uint8_t> pixels = std::vector<uint8_t>(262144);
  amdgpu::LegacyGpuVmAdapter adapter{sim.soc->gpu_vm(), sim.memory};
  amdgpu::ComputeUnitCore *cu = sim.cu();
  amdgpu::Wavefront *wf = nullptr;
  amdgpu::AddressSpaceHandle handle;
  uint32_t width = 24, height = 8, bytes = 4, swizzle = 27;
  bool depth = false;

  ShaderMetadataContext() {
    process.map_pages(metadata_base, metadata.data(), metadata.size(), amdgpu::Mtype::RW, sealed);
    process.map_pages(pixel_base, pixels.data(), pixels.size(), amdgpu::Mtype::RW, sealed);
    handle = adapter.register_address_space(
        79, {.page_table = &process.page_table_,
             .page_table_mutex = &process.page_table_mutex_,
             .page_table_generation = process.page_table_generation(),
             .request_mutex = process.page_table_request_mutex(),
             .mutation_epoch = process.page_table_mutation_epoch(),
             .page_table_cache_state = process.page_table_cache_state(),
             .fault_reporter = &reporter});
    cu->set_gpu_vm(&sim.soc->gpu_vm());
    wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    if (!handle || !wf)
      throw std::runtime_error("metadata test wave setup failed");
    wf->set_process_id(79);
    wf->set_address_space(handle);
    reset();
  }

  void reset() {
    std::ranges::fill(metadata, 0xff);
    for (size_t i = 0; i < pixels.size(); ++i)
      pixels[i] = (i * 17 + 3) & 255;
    reporter.faults.clear();
    wf->clear_instruction_execution_error();
  }

  uint64_t address(uint32_t x) const {
    return *amdgpu::gfx11_image_address(pixel_base, x, 0, width, bytes, swizzle);
  }
  uint64_t key_address(uint32_t x) const {
    return *amdgpu::gfx11_metadata_address(metadata_base, x, 0, width, height, bytes, swizzle,
                                           depth);
  }
  void key(uint32_t x, uint32_t value) {
    const size_t offset = key_address(x) - metadata_base;
    std::memcpy(metadata.data() + offset, &value, depth ? 4 : 1);
  }
  std::unique_ptr<amdgpu::VectorMemState> state(bool load, bool sampling = false) const {
    auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
    d->elem_size = bytes;
    d->num_elems = 1;
    d->is_load = load;
    d->wf_size = wf->wf_size();
    d->exec_mask = d->lane_mask = 0x25;
    d->image_metadata = std::make_unique<amdgpu::ImageMetadataAccess>();
    auto &image = *d->image_metadata;
    image.base = pixel_base;
    image.metadata = metadata_base;
    image.slice_size = 65536;
    image.width = width;
    image.height = height;
    image.swizzle = swizzle;
    image.depth = depth;
    for (const auto &[lane, x] : {std::pair{0u, 1u}, std::pair{2u, 9u}, std::pair{5u, 17u}}) {
      image.coordinates[lane] = x;
      d->per_lane_addr[lane] = address(x);
    }
    if (!load) {
      d->store_data.resize(size_t{d->wf_size} * bytes);
      for (size_t i = 0; i < d->store_data.size(); ++i)
        d->store_data[i] = 0x40 + (i & 63);
    }
    if (sampling) {
      d->image_sample = std::make_unique<amdgpu::ImageSampleAccess>();
      d->image_sample->tap_count = 2;
      for (uint32_t tap = 0; tap < 2; ++tap) {
        auto &t = d->image_sample->taps[tap];
        t.lane_mask = tap ? 0x24 : 0x25;
        t.coordinates = image.coordinates;
        t.addresses = d->per_lane_addr;
      }
      // A later tap repeats an earlier block and texel; it remains a live read.
      d->image_sample->taps[1].coordinates[2] = 1;
      d->image_sample->taps[1].addresses[2] = address(1);
    }
    return d;
  }

  struct Result {
    amdgpu::VmAccessOutcome outcome;
    std::vector<uint8_t> response;
    uint64_t lane_mask;
    amdgpu::InstructionExecutionError error;
    int round, flags, error_number;
    bool operator==(const Result &) const = default;
  };
  Result run(std::unique_ptr<amdgpu::VectorMemState> d, bool original) {
    auto *data = d.get();
    TestMemoryInstruction inst(std::move(d));
    MetadataInitiationPipeline pipeline(&cu->l1_vector(), cu->l2());
    cu->set_debug_active(original);
    fenv_t saved;
    std::fegetenv(&saved);
    std::fesetround(FE_UPWARD);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_OVERFLOW | FE_INEXACT);
    errno = EILSEQ;
    const auto outcome = pipeline.initiate_access(inst, *wf);
    const int error_number = errno;
    const int flags = std::fetestexcept(FE_ALL_EXCEPT), round = std::fegetround();
    std::fesetenv(&saved);
    cu->set_debug_active(false);
    return {outcome, data->response_data, data->lane_mask, wf->instruction_execution_error(), round,
            flags,   error_number};
  }
};

TEST(GpuVmPipeline, MetadataRamKeepsDccTapLaneAndClearStoreOrder) {
  ShaderMetadataContext original, candidate;
  for (const uint32_t bytes : {1u, 2u, 4u, 8u, 16u}) {
    original.bytes = candidate.bytes = bytes;
    for (const bool load : {false, true}) {
      for (const uint32_t key : {0xffu, 0u, 1u, 2u, 4u, 6u, 8u, 10u}) {
        SCOPED_TRACE(::testing::Message() << bytes << '/' << load << '/' << key);
        original.reset();
        candidate.reset();
        for (uint32_t x : {1u, 9u, 17u}) {
          original.key(x, key);
          candidate.key(x, key);
        }
        const auto expected = original.run(original.state(load, load), true);
        const auto actual = candidate.run(candidate.state(load, load), false);
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(candidate.metadata, original.metadata);
        EXPECT_EQ(candidate.pixels, original.pixels);
        EXPECT_EQ(candidate.reporter.faults, original.reporter.faults);
      }
    }
  }
}

TEST(GpuVmPipeline, MetadataRamKeepsHtileEndpointsAndLaterUnsupportedKeyPrefix) {
  ShaderMetadataContext original, candidate;
  original.depth = candidate.depth = true;
  original.swizzle = candidate.swizzle = 24;
  for (const uint32_t key : {0xffffffffu, 0u, 0xfffffff0u, 0x3f80000fu}) {
    for (const bool load : {false, true}) {
      SCOPED_TRACE(::testing::Message() << key << '/' << load);
      original.reset();
      candidate.reset();
      original.key(1, key);
      candidate.key(1, key);
      original.key(9, 3); // unsupported second block, after the first block's effects
      candidate.key(9, 3);
      const auto expected = original.run(original.state(load), true);
      const auto actual = candidate.run(candidate.state(load), false);
      EXPECT_EQ(actual, expected);
      EXPECT_EQ(actual.error, amdgpu::InstructionExecutionError::UnsupportedOperandValue);
      EXPECT_EQ(candidate.metadata, original.metadata);
      EXPECT_EQ(candidate.pixels, original.pixels);
      EXPECT_EQ(candidate.reporter.faults, original.reporter.faults);
    }
  }
}

TEST(GpuVmPipeline, MetadataRamRefusalKeepsPageAndSubextentFaultPrefixes) {
  for (uint32_t variant = 0; variant < 5; ++variant) {
    for (const bool load : {false, true}) {
      SCOPED_TRACE(::testing::Message() << variant << '/' << load);
      ShaderMetadataContext original, candidate;
      original.width = candidate.width = 256;
      for (auto *context : {&original, &candidate}) {
        context->key(1, 0);
        context->key(128, 0);
        const uint64_t late = context->address(128);
        auto &pte = context->process.page_table_.at(late >> KfdProcess::kPageShift);
        if (variant == 0)
          context->process.page_table_.erase(late >> KfdProcess::kPageShift);
        else if (variant == 1) {
          const uint32_t split = (late & 4095) + 2;
          auto *host_page = context->pixels.data() + (late & ~uint64_t{4095}) - context->pixel_base;
          pte.host_extents = {{host_page, split, 0, context->sealed},
                              {host_page + split, 4096 - split, split, context->sealed}};
        } else if (variant == 2)
          pte.host_extents.push_back(pte.host_extents.front());
        else if (variant == 3)
          pte.host_extents.front().owner = amdgpu::LegacyHostExtentOwner::Application;
        else
          context->process.page_table_.erase((context->pixel_base + 126976) >>
                                             KfdProcess::kPageShift);
      }
      auto reference = original.state(load), draft = candidate.state(load);
      for (auto *d : {reference.get(), draft.get()}) {
        d->exec_mask = d->lane_mask = 0x5;
        d->image_metadata->coordinates[2] = 128;
        d->per_lane_addr[2] = original.address(128);
      }
      const auto expected = original.run(std::move(reference), true);
      const auto actual = candidate.run(std::move(draft), false);
      EXPECT_EQ(actual, expected);
      EXPECT_EQ(candidate.metadata, original.metadata);
      EXPECT_EQ(candidate.pixels, original.pixels);
      EXPECT_EQ(candidate.reporter.faults, original.reporter.faults);
      if (variant < 3) {
        EXPECT_EQ(actual.error, amdgpu::InstructionExecutionError::UnsupportedOperandValue);
        if (variant != 0) {
          EXPECT_FALSE(candidate.reporter.faults.empty());
        }
      } else {
        EXPECT_EQ(actual.error, amdgpu::InstructionExecutionError::None);
      }
    }
  }
}

} // namespace

namespace {
TEST(GpuVmPipeline, MetadataRamDefaultBackendKeepsExactReadOrder) {
  TranslatedPipelineContext context;
  constexpr uint64_t pixel = 0x8000, metadata = 0x4000;
  auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  state->elem_size = 4;
  state->num_elems = 1;
  state->is_load = true;
  state->wf_size = context.wf->wf_size();
  state->exec_mask = state->lane_mask = 5;
  state->image_metadata = std::make_unique<amdgpu::ImageMetadataAccess>();
  auto &image = *state->image_metadata;
  image.base = pixel;
  image.metadata = metadata;
  image.width = image.height = 8;
  image.swizzle = 27;
  std::array<uint64_t, 2> keys{}, addresses{};
  for (size_t i = 0; i < 2; ++i) {
    const uint32_t lane = i * 2, x = i + 1;
    image.coordinates[lane] = x;
    keys[i] = *amdgpu::gfx11_metadata_address(metadata, x, 0, 8, 8, 4, 27, false);
    addresses[i] = *amdgpu::gfx11_image_address(pixel, x, 0, 8, 4, 27);
    state->per_lane_addr[lane] = addresses[i];
    context.external->store<uint8_t>(keys[i], 0xff);
    context.external->store<uint32_t>(addresses[i], 0x12340000 + i);
  }
  auto *data = state.get();
  TestMemoryInstruction inst(std::move(state));
  MetadataInitiationPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
  EXPECT_EQ(pipeline.initiate_access(inst, *context.wf), amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(context.external->read_history.size(), 4u);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(context.external->read_history[2 * i].address, keys[i]);
    EXPECT_EQ(context.external->read_history[2 * i].size, 1u);
    EXPECT_EQ(context.external->read_history[2 * i + 1].address, addresses[i]);
    EXPECT_EQ(context.external->read_history[2 * i + 1].size, 4u);
    uint32_t result;
    std::memcpy(&result, data->response_data.data() + 2 * i * 4, 4);
    EXPECT_EQ(result, 0x12340000 + i);
  }
  EXPECT_EQ(context.external->prepare_ram_calls, 0u);
  EXPECT_EQ(context.external->write_calls, 0u);
}
} // namespace

namespace {
TEST(GpuVmPipeline, MetadataRamSelectedSwizzleBlocksContainEveryCorePixel) {
  // Independently execute the scalar materialization cores, including key-1
  // reference reads and clipped edge clears, and check every resulting pixel
  // access against the algebraic block used by the instruction planner.
  struct CheckedMemory {
    amdgpu::VmRamRange pixels;
    uint64_t key_address;
    uint32_t key, key_bytes;
    mutable bool contained = true;
    mutable size_t pixel_reads = 0, pixel_writes = 0, key_reads = 0, key_writes = 0;
    bool contains(uint64_t address, size_t size) const {
      return address >= pixels.address && size <= pixels.size &&
             address - pixels.address <= pixels.size - size;
    }
    amdgpu::VmAccessOutcome read(uint64_t address, std::span<std::byte> out) const {
      if (address == key_address) {
        contained &= out.size() == key_bytes;
        std::memcpy(out.data(), &key, out.size());
        ++key_reads;
      } else {
        contained &= contains(address, out.size());
        std::ranges::fill(out, std::byte{0x42});
        ++pixel_reads;
      }
      return amdgpu::VmAccessOutcome::Complete;
    }
    amdgpu::VmAccessOutcome write(uint64_t address, std::span<const std::byte> in) const {
      if (address == key_address) {
        contained &= in.size() == key_bytes;
        ++key_writes;
      } else {
        contained &= contains(address, in.size());
        ++pixel_writes;
      }
      return amdgpu::VmAccessOutcome::Complete;
    }
  };
  for (const bool depth : {false, true}) {
    for (const uint32_t bytes : {1u, 2u, 4u, 8u, 16u}) {
      if (depth && bytes != 4)
        continue;
      for (const uint32_t swizzle : depth ? std::array{24u, 28u} : std::array{27u, 31u}) {
        const uint32_t block_log2 = amdgpu::image_block_log2(false, swizzle);
        const uint64_t block_size = uint64_t{1} << block_log2;
        for (const uint32_t width : {7u, 129u, 1023u, 4096u}) {
          const uint32_t height = width / 2 + 1;
          const auto mip = amdgpu::image_mip_layout(false, swizzle, bytes, width, height, 1, 0);
          ASSERT_TRUE(mip);
          for (const uint32_t layer : {0u, 1u, 3u, 17u}) {
            if (depth && layer)
              continue;
            for (const bool pipe : {false, true}) {
              if (depth && !pipe)
                continue; // HTILE has no pipe-alignment variant.
              constexpr uint64_t base = 0x1003400, metadata = 0x20000100;
              const uint64_t layer_base =
                  amdgpu::image_layer_base(false, base, mip->slice_size, layer, bytes, swizzle);
              for (const auto &[x, y] : {std::pair{0u, 0u}, std::pair{width - 1, height - 1},
                                         std::pair{width / 2, height / 2}}) {
                const uint64_t block = amdgpu::image_metadata_detail::pixel_swizzle_block_offset(
                    x, y, width, bytes, swizzle);
                const auto key_address = amdgpu::gfx11_metadata_address(
                    metadata, x, y, width, height, bytes, swizzle, depth, depth || pipe, layer);
                ASSERT_TRUE(key_address);
                for (const uint32_t key :
                     depth ? std::array{0u, 0xfffffff0u} : std::array{0u, 1u}) {
                  SCOPED_TRACE(::testing::Message()
                               << depth << '/' << bytes << '/' << swizzle << '/' << width << '/'
                               << layer << '/' << pipe << '/' << x << '/' << y << '/' << key);
                  CheckedMemory memory{{(layer_base & ~(block_size - 1)) + block, block_size},
                                       *key_address,
                                       key,
                                       depth ? 4u : 1u};
                  const char *error =
                      depth ? amdgpu::image_metadata_detail::materialize_gfx11_htile(
                                  memory, base, metadata, x, y, width, height, bytes, swizzle)
                            : amdgpu::image_metadata_detail::materialize_gfx11_dcc(
                                  memory, base, metadata, x, y, width, height, bytes, swizzle, pipe,
                                  layer, mip->slice_size);
                  EXPECT_EQ(error, nullptr);
                  EXPECT_TRUE(memory.contained);
                  EXPECT_EQ(memory.key_reads, 1u);
                  EXPECT_EQ(memory.key_writes, 1u);
                  EXPECT_GT(memory.pixel_writes, 0u);
                  EXPECT_EQ(memory.pixel_reads, (!depth && key == 1) ? 1u : 0u);
                }
              }
            }
          }
        }
      }
    }
  }
}
} // namespace

namespace {
TEST(GpuVmPipeline, MetadataRamBoundsOriginalDirectTexelsSeparatelyFromClearCoordinates) {
  ShaderMetadataContext original, candidate;
  original.width = candidate.width = 512;
  original.height = candidate.height = 128;
  for (const bool load : {false, true}) {
    original.reset();
    candidate.reset();
    original.key(1, 0);
    candidate.key(1, 0);
    auto reference = original.state(load), draft = candidate.state(load);
    // A manually supplied direct address can name another swizzle block. It
    // must join the envelope even though clear coordinates remain unchanged.
    reference->per_lane_addr[2] = original.address(400);
    draft->per_lane_addr[2] = candidate.address(400);
    const auto expected = original.run(std::move(reference), true);
    const auto actual = candidate.run(std::move(draft), false);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(actual.error, amdgpu::InstructionExecutionError::None);
    EXPECT_EQ(candidate.metadata, original.metadata);
    EXPECT_EQ(candidate.pixels, original.pixels);
    EXPECT_EQ(candidate.reporter.faults, original.reporter.faults);
  }
}
} // namespace

namespace {
TEST(GpuVmPipeline, MetadataAddressCacheMatchesLayoutsLayersAndPartialBlocks) {
  for (const bool depth : {false, true}) {
    for (const uint32_t bytes : {1u, 2u, 4u, 8u, 16u}) {
      if (depth && bytes > 4)
        continue;
      for (const uint32_t swizzle : {depth ? 24u : 27u, depth ? 28u : 31u}) {
        for (const bool pipe : {false, true}) {
          for (const auto &dimensions : {std::pair{37u, 19u}, std::pair{2053u, 1031u}}) {
            const auto [width, height] = dimensions;
            constexpr uint64_t metadata = 0x1234500;
            amdgpu::image_metadata_detail::MetadataAddressCache cache(metadata, width, height,
                                                                      bytes, swizzle, depth, pipe);
            const uint32_t bits = depth ? 6 : 8 - std::countr_zero(bytes);
            const uint32_t bw = 1u << ((bits + 1) / 2), bh = 1u << (bits / 2);
            for (const uint32_t layer : {0u, 1u, 5u, 0u}) {
              for (const uint32_t block_y : {0u, bh, height - 1}) {
                for (const uint32_t block_x : {0u, bw, width - 1}) {
                  for (uint32_t dy = 0; dy < bh; ++dy) {
                    for (uint32_t dx = 0; dx < bw; ++dx) {
                      const uint32_t x = block_x + dx, y = block_y + dy;
                      EXPECT_EQ(cache.lookup(x, y, layer),
                                amdgpu::gfx11_metadata_address(metadata, x, y, width, height, bytes,
                                                               swizzle, depth, pipe, layer))
                          << depth << '/' << bytes << '/' << swizzle << '/' << pipe << '/' << layer
                          << '/' << x << '/' << y;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  for (const uint32_t bytes : {0u, 3u, 32u}) {
    amdgpu::image_metadata_detail::MetadataAddressCache cache(0, 8, 8, bytes, 27, false, true);
    EXPECT_EQ(cache.lookup(0, 0), std::nullopt);
    EXPECT_EQ(cache.lookup(1, 1), std::nullopt);
  }
}

TEST(GpuVmPipeline, MetadataAddressCacheReadsChangedKeysAndKeepsErrorPrefixes) {
  struct Event {
    bool write;
    uint64_t address;
    size_t size;
    std::array<std::byte, 16> value{};
    bool operator==(const Event &) const = default;
  };
  struct Memory {
    uint64_t key_address = 0;
    mutable uint32_t key = 0;
    size_t fail_at = SIZE_MAX;
    mutable std::vector<Event> events;
    amdgpu::VmAccessOutcome read(uint64_t address, std::span<std::byte> bytes) const {
      if (address == key_address)
        std::memcpy(bytes.data(), &key, bytes.size());
      else
        for (size_t i = 0; i < bytes.size(); ++i)
          bytes[i] = std::byte((address + i) & 255);
      Event event{false, address, bytes.size()};
      std::copy(bytes.begin(), bytes.end(), event.value.begin());
      events.push_back(event);
      return events.size() == fail_at ? amdgpu::VmAccessOutcome::Faulted
                                      : amdgpu::VmAccessOutcome::Complete;
    }
    amdgpu::VmAccessOutcome write(uint64_t address, std::span<const std::byte> bytes) const {
      Event event{true, address, bytes.size()};
      std::copy(bytes.begin(), bytes.end(), event.value.begin());
      events.push_back(event);
      if (events.size() == fail_at)
        return amdgpu::VmAccessOutcome::Faulted;
      if (address == key_address)
        std::memcpy(&key, bytes.data(), bytes.size());
      return amdgpu::VmAccessOutcome::Complete;
    }
  };
  // Change the key between successive coordinates in one cached block. No
  // expanded-key or successful-materialization result may be reused.
  for (const bool depth : {false, true}) {
    for (const uint32_t bytes : {2u, 4u}) {
      const uint32_t swizzle = depth ? 24 : 27;
      constexpr uint64_t base = 0x80000, metadata = 0x40000;
      constexpr uint32_t width = 37, height = 19;
      const auto mip = amdgpu::image_mip_layout(false, swizzle, bytes, width, height, 1, 0);
      ASSERT_TRUE(mip);
      for (const size_t failure : {SIZE_MAX, size_t{1}, size_t{2}, size_t{5}}) {
        amdgpu::image_metadata_detail::MetadataAddressCache cache(metadata, width, height, bytes,
                                                                  swizzle, depth, true);
        for (const uint32_t key : {0xffu, 0u, 1u, 2u, 4u, 6u, 8u, 10u, 0x80u}) {
          for (const uint32_t x : {1u, 2u, 3u, 17u, 2u}) {
            SCOPED_TRACE(::testing::Message()
                         << depth << '/' << bytes << '/' << failure << '/' << key << '/' << x);
            const uint32_t layer = depth ? 0 : 1;
            const auto address = cache.lookup(x, 1, layer);
            ASSERT_TRUE(address);
            Memory original, cached;
            original.key_address = cached.key_address = *address;
            original.key = cached.key = key;
            original.fail_at = cached.fail_at = failure;
            const char *expected =
                depth ? amdgpu::image_metadata_detail::materialize_gfx11_htile(
                            original, base, metadata, x, 1, width, height, bytes, swizzle)
                      : amdgpu::image_metadata_detail::materialize_gfx11_dcc(
                            original, base, metadata, x, 1, width, height, bytes, swizzle, true,
                            layer, mip->slice_size);
            const char *actual =
                depth ? amdgpu::image_metadata_detail::materialize_gfx11_htile_at_address(
                            cached, base, *address, x, 1, width, height, bytes, swizzle)
                      : amdgpu::image_metadata_detail::materialize_gfx11_dcc_at_address(
                            cached, base, *address, x, 1, width, height, bytes, swizzle, layer,
                            mip->slice_size);
            EXPECT_EQ(std::string_view(actual ? actual : ""),
                      std::string_view(expected ? expected : ""));
            EXPECT_EQ(cached.key, original.key);
            EXPECT_EQ(cached.events, original.events);
            ASSERT_FALSE(cached.events.empty());
            EXPECT_FALSE(cached.events.front().write);
            EXPECT_EQ(cached.events.front().address, *address);
          }
        }
      }
    }
  }
}
} // namespace

namespace {
// A test-only seam after the real guards have been released. Production keeps
// request storage alive until the device boundary ends and has no such callback.
class MetadataReleaseHook final : public amdgpu::VmRamLeaseRequest {
public:
  explicit MetadataReleaseHook(std::unique_ptr<amdgpu::VmRamLeaseRequest> request)
      : request_(std::move(request)) {}
  bool try_acquire() override { return request_->try_acquire(); }
  std::span<std::byte> bytes(size_t index) const override {
    if (throw_bytes)
      throw std::runtime_error("test span failure");
    return request_->bytes(index);
  }
  void release() override {
    ++release_calls;
    if (releases)
      return;
    request_->release();
    ++releases;
    if (after_release)
      after_release();
  }
  unsigned releases = 0, release_calls = 0;
  bool throw_bytes = false;
  std::function<void()> after_release;

private:
  std::unique_ptr<amdgpu::VmRamLeaseRequest> request_;
};

TEST(GpuVmPipeline, MetadataRamExpandedKeysStayInAdmittedSpans) {
  ShaderMetadataContext context;
  const auto access = context.wf->snapshot_vm_access();
  ASSERT_TRUE(access);
  const std::array<amdgpu::VmRamRange, 2> ranges{
      {{context.metadata_base, 16384}, {context.address(1), 4}}};
  auto prepared = access->prepare_ram_lease(ranges);
  ASSERT_TRUE(prepared);
  MetadataReleaseHook request(std::move(prepared));
  ASSERT_TRUE(request.try_acquire());
  {
    const amdgpu::image_metadata_detail::FallbackRamAccess ram(*access, ranges, request);
    std::array<std::byte, 4> value{};
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
      ASSERT_EQ(amdgpu::image_metadata_detail::materialize_gfx11_dcc(
                    ram, context.pixel_base, context.metadata_base, 1, 0, context.width,
                    context.height, 4, context.swizzle),
                nullptr);
      ASSERT_EQ(ram.read(context.address(1), value), amdgpu::VmAccessOutcome::Complete);
      ASSERT_EQ(ram.write(context.address(1), value), amdgpu::VmAccessOutcome::Complete);
      EXPECT_TRUE(ram.has_ram());
      EXPECT_EQ(request.releases, 0u);
    }
    EXPECT_TRUE(context.reporter.faults.empty());
  }
  EXPECT_EQ(request.release_calls, 1u);
}

TEST(GpuVmPipeline, MetadataRamOwnerReleasesExactlyOnceOnEveryExit) {
  for (unsigned exit = 0; exit < 5; ++exit) {
    SCOPED_TRACE(exit);
    ShaderMetadataContext context;
    const auto access = context.wf->snapshot_vm_access();
    ASSERT_TRUE(access);
    const uint64_t address = context.address(1);
    const std::array<amdgpu::VmRamRange, 2> ranges{{{context.metadata_base, 16384}, {address, 4}}};
    auto prepared = access->prepare_ram_lease(ranges);
    ASSERT_TRUE(prepared);
    MetadataReleaseHook request(std::move(prepared));
    ASSERT_TRUE(request.try_acquire());
    request.throw_bytes = exit == 4;
    try {
      [&] {
        const amdgpu::image_metadata_detail::FallbackRamAccess ram(*access, ranges, request);
        if (exit == 1)
          return; // Initial alias refusal, before any materializer operation.
        std::array<std::byte, 4> value{};
        EXPECT_EQ(ram.read(address + (exit == 3 ? 4 : 0), value),
                  amdgpu::VmAccessOutcome::Complete);
        if (exit >= 2)
          throw std::runtime_error("test transfer failure");
      }();
    } catch (const std::runtime_error &) {
      EXPECT_GE(exit, 2u);
    }
    EXPECT_EQ(request.release_calls, 1u);
    EXPECT_EQ(request.releases, 1u);
  }
}

TEST(GpuVmPipeline, MetadataRamTransitionKeepsWholeSpanAndUsesReplacementMapping) {
  ASSERT_EXIT(
      ([] {
        alarm(20);
        ShaderMetadataContext context;
        const auto access = context.wf->snapshot_vm_access();
        if (!access)
          _exit(1);
        const uint64_t address = context.address(1);
        const std::array<amdgpu::VmRamRange, 2> ranges{
            {{context.metadata_base, 16384}, {address, 4}}};
        auto prepared = access->prepare_ram_lease(ranges);
        if (!prepared)
          _exit(2);
        MetadataReleaseHook request(std::move(prepared));
        if (!request.try_acquire())
          _exit(3);
        const amdgpu::image_metadata_detail::FallbackRamAccess ram(*access, ranges, request);
        const std::array<std::byte, 4> first{std::byte{1}, std::byte{2}, std::byte{3},
                                             std::byte{4}};
        if (ram.write(address, first) != amdgpu::VmAccessOutcome::Complete)
          _exit(4);
        const auto initial = context.pixels;
        std::array<uint8_t, 4096> replacement{};
        request.after_release = [&] {
          context.process.map_pages(address & ~uint64_t{4095}, replacement.data(),
                                    replacement.size(), amdgpu::Mtype::RW, context.sealed);
          errno = ERANGE; // Guard cleanup must not replace the caller's errno.
        };
        errno = EILSEQ;
        // Two bytes overlap the former lease. The whole write must happen only
        // against the replacement mapping, with no old-span prefix copied.
        if (ram.write(address + 2, first) != amdgpu::VmAccessOutcome::Complete || ram.has_ram() ||
            request.releases != 1 || errno != EILSEQ || context.pixels != initial)
          _exit(5);
        const auto offset = address & 4095;
        if (std::memcmp(replacement.data() + offset + 2, first.data(), first.size()) != 0)
          _exit(6);
        // A subsequent read aliases the former RAM storage as its destination.
        // It must use the current mapping and must never resurrect old spans.
        auto destination = std::as_writable_bytes(std::span(context.pixels))
                               .subspan(address - context.pixel_base, 4);
        if (ram.read(address, destination) != amdgpu::VmAccessOutcome::Complete ||
            std::memcmp(destination.data(), replacement.data() + offset, 4) != 0 ||
            request.releases != 1)
          _exit(7);
        _exit(0);
      }()),
      ::testing::ExitedWithCode(0), "");
}

TEST(GpuVmPipeline, MetadataRamTransitionRevalidatesRevokedSnapshot) {
  ASSERT_EXIT(
      ([] {
        alarm(20);
        ShaderMetadataContext context;
        const auto access = context.wf->snapshot_vm_access();
        if (!access)
          _exit(1);
        const uint64_t address = context.address(1);
        const std::array<amdgpu::VmRamRange, 2> ranges{
            {{context.metadata_base, 16384}, {address, 4}}};
        auto prepared = access->prepare_ram_lease(ranges);
        if (!prepared)
          _exit(2);
        MetadataReleaseHook request(std::move(prepared));
        if (!request.try_acquire())
          _exit(3);
        const amdgpu::image_metadata_detail::FallbackRamAccess ram(*access, ranges, request);
        std::array<std::byte, 4> value{};
        if (ram.read(address, value) != amdgpu::VmAccessOutcome::Complete)
          _exit(4);
        request.after_release = [&] {
          if (!context.sim.soc->gpu_vm().invalidate(context.handle))
            _exit(5);
        };
        const auto initial = context.pixels;
        if (ram.write(address + 4, value) != amdgpu::VmAccessOutcome::Unavailable ||
            ram.has_ram() || ram.read(address, value) != amdgpu::VmAccessOutcome::Unavailable ||
            request.releases != 1 || context.pixels != initial)
          _exit(6);
        _exit(0);
      }()),
      ::testing::ExitedWithCode(0), "");
}

TEST(GpuVmPipeline, MetadataRamLateClearFaultKeepsPrefixAndCallbackErrno) {
  ASSERT_EXIT(
      ([] {
        alarm(30);
        for (unsigned variant = 0; variant < 3; ++variant) {
          for (bool load : {false, true}) {
            ShaderMetadataContext original, candidate;
            std::array<std::unique_ptr<amdgpu::VectorMemState>, 2> states;
            unsigned index = 0;
            for (auto *context : {&original, &candidate}) {
              context->width = 256;
              context->depth = variant == 2;
              context->swizzle = context->depth ? 24 : 27;
              const uint32_t first_x = variant == 1 ? 17 : 1;
              const uint32_t last_x = variant == 1 ? 1 : 128;
              context->key(first_x, 0xffffffff);
              context->key(last_x, variant == 1 ? 1 : 0);
              auto state = context->state(load);
              state->exec_mask = state->lane_mask = 5;
              state->image_metadata->coordinates[0] = first_x;
              state->per_lane_addr[0] = context->address(first_x);
              state->image_metadata->coordinates[2] = last_x;
              state->per_lane_addr[2] = context->address(last_x);
              const uint64_t begin = std::min(context->address(first_x), context->address(last_x));
              const uint64_t end =
                  std::max(context->address(first_x), context->address(last_x)) + 4;
              // Key-1's clear reference lies below the direct envelope; a zero
              // clear's second store lies above it. Keep one strict extent for
              // all directly selected bytes, leaving only the clear access bad.
              const uint64_t page = (variant == 1 ? begin : end - 1) & ~uint64_t{4095};
              auto &extent = context->process.page_table_.at(page >> 12).host_extents.front();
              if (variant == 1) {
                extent.host_ptr += begin - page;
                extent.gpu_page_offset = begin - page;
                extent.host_backed_bytes = 4096 - extent.gpu_page_offset;
              } else {
                extent.host_backed_bytes = end - page;
              }
              const auto access = context->wf->snapshot_vm_access();
              if (!access)
                _exit(1);
              const std::array<amdgpu::VmRamRange, 2> ranges{
                  {{context->metadata_base, context->depth ? 131072u : 16384u},
                   {begin, end - begin}}};
              auto request = access->prepare_ram_lease(ranges);
              if (!request || !request->try_acquire())
                _exit(2);
              request->release();
              context->reporter.callback = [context, page](uint64_t) {
                // Both extra guards must be gone before ordinary fault
                // publication. Repairing a mapping is supported callback reentry.
                context->process.map_pages(page,
                                           context->pixels.data() + page - context->pixel_base,
                                           4096, amdgpu::Mtype::RW, context->sealed);
                auto mapping = host_mapping_lock().lock_exclusive();
                errno = EDOM;
              };
              states[index++] = std::move(state);
            }
            const auto expected = original.run(std::move(states[0]), true);
            const auto actual = candidate.run(std::move(states[1]), false);
            if (actual != expected || actual.error_number != EDOM ||
                actual.error != amdgpu::InstructionExecutionError::UnsupportedOperandValue ||
                candidate.pixels != original.pixels || candidate.metadata != original.metadata ||
                candidate.reporter.faults != original.reporter.faults ||
                candidate.reporter.faults.size() != 1)
              _exit(3);
            if (!load) {
              const auto first = candidate.address(variant == 1 ? 17 : 1) - candidate.pixel_base;
              const std::array<uint8_t, 4> stored{0x40, 0x41, 0x42, 0x43};
              if (std::memcmp(candidate.pixels.data() + first, stored.data(), stored.size()) != 0)
                _exit(4);
            }
          }
        }
        _exit(0);
      }()),
      ::testing::ExitedWithCode(0), "");
}
} // namespace

namespace {
TEST(GpuVmPipeline, MetadataRamTransitionPreservesCrossPagePartialTransfer) {
  for (bool load : {false, true}) {
    SCOPED_TRACE(load);
    ShaderMetadataContext original, candidate;
    constexpr uint64_t address = ShaderMetadataContext::pixel_base + 4094;
    original.process.unmap_pages(ShaderMetadataContext::pixel_base + 4096, 4096);
    candidate.process.unmap_pages(ShaderMetadataContext::pixel_base + 4096, 4096);
    const auto ordinary = original.wf->snapshot_vm_access();
    const auto access = candidate.wf->snapshot_vm_access();
    ASSERT_TRUE(ordinary);
    ASSERT_TRUE(access);
    const std::array<amdgpu::VmRamRange, 2> ranges{
        {{candidate.metadata_base, 16384}, {address, 2}}};
    std::array<std::byte, 4> expected{std::byte{0xa1}, std::byte{0xb2}, std::byte{0xc3},
                                      std::byte{0xd4}};
    auto actual = expected;
    const auto reference_outcome =
        load ? ordinary->read(address, expected) : ordinary->write(address, expected);
    auto prepared = access->prepare_ram_lease(ranges);
    ASSERT_TRUE(prepared);
    MetadataReleaseHook request(std::move(prepared));
    ASSERT_TRUE(request.try_acquire());
    const amdgpu::image_metadata_detail::FallbackRamAccess ram(*access, ranges, request);
    const auto actual_outcome = load ? ram.read(address, actual) : ram.write(address, actual);
    EXPECT_EQ(actual_outcome, reference_outcome);
    EXPECT_EQ(actual_outcome, amdgpu::VmAccessOutcome::Unavailable);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(candidate.pixels, original.pixels);
    EXPECT_EQ(candidate.reporter.faults, original.reporter.faults);
    EXPECT_FALSE(ram.has_ram());
    EXPECT_EQ(request.releases, 1u);
    if (load) {
      EXPECT_EQ(actual[0], std::byte(candidate.pixels[4094]));
      EXPECT_EQ(actual[1], std::byte(candidate.pixels[4095]));
      EXPECT_EQ(actual[2], std::byte{0xc3});
      EXPECT_EQ(actual[3], std::byte{0xd4});
    } else {
      EXPECT_EQ(candidate.pixels[4094], 0xa1);
      EXPECT_EQ(candidate.pixels[4095], 0xb2);
    }
  }
}
} // namespace

namespace {
class MetadataBatchPipeline final : public amdgpu::GlobalMemPipeline {
public:
  using GlobalMemPipeline::GlobalMemPipeline;
  unsigned initiations = 0;
  struct Completion {
    std::vector<uint8_t> response;
    uint64_t mask;
    amdgpu::InstructionExecutionError error;
    int rounding, flags, error_number;
    bool operator==(const Completion &) const = default;
  };
  std::vector<Completion> completed;
  std::function<void(Instruction &, amdgpu::Wavefront &)> on_initiate;
  std::function<void(amdgpu::Wavefront &)> on_complete;

protected:
  amdgpu::VmAccessOutcome initiate_access(Instruction &inst, amdgpu::Wavefront &wf) override {
    ++initiations;
    if (on_initiate)
      on_initiate(inst, wf);
    return GlobalMemPipeline::initiate_access(inst, wf);
  }
  amdgpu::MemoryAccessCompletion complete_access(Instruction &inst, amdgpu::Wavefront &wf,
                                                 amdgpu::MemoryAccessDeferredCompletion) override {
    const auto &d = *inst.data_as<amdgpu::VectorMemState>();
    completed.push_back({d.response_data, d.lane_mask, wf.instruction_execution_error(),
                         std::fegetround(), std::fetestexcept(FE_ALL_EXCEPT), errno});
    if (on_complete)
      on_complete(wf);
    return amdgpu::MemoryAccessCompletion::Complete;
  }
};

amdgpu::Wavefront *metadata_second_wave(ShaderMetadataContext &context) {
  auto *wf = context.cu->dispatch_wf(1, 0, kGfx1250ScalarSlots, 32);
  if (wf) {
    wf->set_process_id(79);
    wf->set_address_space(context.handle);
  }
  return wf;
}

TEST(GpuVmPipeline, MetadataStepBatchKeepsClearKeysPrefixAndHostState) {
  for (bool depth : {false, true}) {
    for (uint32_t key : {0u, 1u, 2u, 4u, 6u, 8u, 10u, 0xffu, 0xfffffff0u}) {
      // Other nonzero HTILE low nibbles take the same unsupported-key branch.
      if (depth && key != 0 && key != 1 && key != 0xff && key != 0xfffffff0)
        continue;
      for (bool malformed_late : {false, true}) {
        SCOPED_TRACE(::testing::Message() << depth << '/' << key << '/' << malformed_late);
        ShaderMetadataContext original, candidate;
        original.depth = candidate.depth = depth;
        original.swizzle = candidate.swizzle = depth ? 24 : 27;
        for (auto *context : {&original, &candidate}) {
          context->cu->set_debug_active(false);
          context->cu->set_plugin_group(std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{}));
          context->key(1, key);
          if (malformed_late)
            context->key(9, depth ? 3 : 0x80);
        }
        auto *old_second = metadata_second_wave(original);
        auto *new_second = metadata_second_wave(candidate);
        ASSERT_NE(old_second, nullptr);
        ASSERT_NE(new_second, nullptr);
        MetadataBatchPipeline serial(&original.cu->l1_vector(), original.cu->l2());
        MetadataBatchPipeline batch(&candidate.cu->l1_vector(), candidate.cu->l2());
        batch.on_complete = [&](amdgpu::Wavefront &) {
          // Architectural completion runs after all batch cache and mapping guards.
          auto boundary = candidate.cu->l2()->coherence_domain()->try_acquire_clean_boundary();
          EXPECT_EQ(boundary.outcome(), amdgpu::VmAccessOutcome::Complete);
          std::unique_lock request(*candidate.process.page_table_request_mutex());
        };
        fenv_t saved;
        ASSERT_EQ(std::fegetenv(&saved), 0);
        std::fesetround(FE_DOWNWARD);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_OVERFLOW | FE_INEXACT);
        errno = EILSEQ;
        for (auto *wf : {original.wf, old_second}) {
          ASSERT_EQ(
              serial.issue_deferred(new TestMemoryInstruction(original.state(true, true)), *wf),
              amdgpu::VmAccessOutcome::Complete);
        }
        const auto epoch = candidate.cu->l2()->coherence_domain()->current_epoch();
        {
          amdgpu::GlobalMemPipeline::StepBatch scope(batch, true);
          for (auto *wf : {candidate.wf, new_second}) {
            ASSERT_EQ(
                batch.issue_deferred(new TestMemoryInstruction(candidate.state(true, true)), *wf),
                amdgpu::VmAccessOutcome::Complete);
            EXPECT_EQ(wf->state(), amdgpu::WfState::VM_RETRY);
            EXPECT_FALSE(wf->wait_counters().empty());
          }
          EXPECT_TRUE(batch.completed.empty());
          scope.finish();
        }
        std::fesetenv(&saved);
        EXPECT_EQ(batch.initiations, 2u); // no ordinary retry: both full RAM proofs succeeded
        EXPECT_EQ(batch.completed, serial.completed);
        EXPECT_EQ(candidate.metadata, original.metadata);
        EXPECT_EQ(candidate.pixels, original.pixels);
        EXPECT_TRUE(candidate.reporter.faults.empty());
        EXPECT_EQ(candidate.cu->l2()->coherence_domain()->current_epoch(), epoch + 2);
        EXPECT_TRUE(batch.empty());
        for (auto *wf : {candidate.wf, new_second}) {
          EXPECT_TRUE(wf->wait_counters().empty());
          EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
        }
      }
    }
  }
}

TEST(GpuVmPipeline, MetadataStepBatchRefusalPreservesOrdinaryFaultAndMappingReentry) {
  ShaderMetadataContext original, candidate;
  for (auto *context : {&original, &candidate}) {
    context->cu->set_debug_active(false);
    context->cu->set_plugin_group(std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{}));
    context->key(1, 0);
    const auto address = context->address(17);
    auto &pte = context->process.page_table_.at(address >> KfdProcess::kPageShift);
    auto *page = context->pixels.data() + (address & ~uint64_t{4095}) - context->pixel_base;
    const uint32_t split = (address & 4095) + 2;
    pte.host_extents = {{page, split, 0, context->sealed},
                        {page + split, 4096 - split, split, context->sealed}};
    context->reporter.callback = [context](uint64_t) {
      // Ordinary fault reporting must not retain the candidate's PTE/mapping guards.
      std::unique_lock request(*context->process.page_table_request_mutex());
      std::unique_lock table(context->process.page_table_mutex_);
      errno = EDOM;
    };
  }
  auto *old_second = metadata_second_wave(original);
  auto *new_second = metadata_second_wave(candidate);
  ASSERT_NE(old_second, nullptr);
  ASSERT_NE(new_second, nullptr);
  MetadataBatchPipeline serial(&original.cu->l1_vector(), original.cu->l2());
  MetadataBatchPipeline batch(&candidate.cu->l1_vector(), candidate.cu->l2());
  for (auto *wf : {original.wf, old_second}) {
    ASSERT_EQ(serial.issue_deferred(new TestMemoryInstruction(original.state(true)), *wf),
              amdgpu::VmAccessOutcome::Complete);
  }
  {
    amdgpu::GlobalMemPipeline::StepBatch scope(batch, true);
    for (auto *wf : {candidate.wf, new_second}) {
      ASSERT_EQ(batch.issue_deferred(new TestMemoryInstruction(candidate.state(true)), *wf),
                amdgpu::VmAccessOutcome::Complete);
    }
    scope.finish();
  }
  EXPECT_EQ(batch.initiations, 4u);
  EXPECT_EQ(batch.completed, serial.completed);
  EXPECT_EQ(candidate.metadata, original.metadata);
  EXPECT_EQ(candidate.pixels, original.pixels);
  EXPECT_EQ(candidate.reporter.faults, original.reporter.faults);
  EXPECT_FALSE(candidate.reporter.faults.empty());
  EXPECT_TRUE(batch.empty());
}

TEST(GpuVmPipeline, MetadataStepBatchRetiresOnceAndCancelsBeforeSlotReuse) {
  ShaderMetadataContext context;
  context.cu->set_debug_active(false);
  context.cu->set_plugin_group(std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{}));
  auto *second = metadata_second_wave(context);
  ASSERT_NE(second, nullptr);
  MetadataBatchPipeline batch(&context.cu->l1_vector(), context.cu->l2());
  const auto generation = second->dispatch_generation();
  {
    amdgpu::GlobalMemPipeline::StepBatch scope(batch, true);
    for (auto *wf : {context.wf, second}) {
      ASSERT_EQ(batch.issue_deferred(new TestMemoryInstruction(context.state(true)), *wf),
                amdgpu::VmAccessOutcome::Complete);
    }
    batch.cancel(*second);
    context.cu->free_wavefront_resources(*second);
    auto *replacement = metadata_second_wave(context);
    ASSERT_EQ(replacement, second);
    EXPECT_NE(replacement->dispatch_generation(), generation);
    batch.on_complete = [&](amdgpu::Wavefront &) {
      // Both the cache boundary and the request ownership must be gone here.
      auto boundary = context.cu->l2()->coherence_domain()->try_acquire_clean_boundary();
      EXPECT_EQ(boundary.outcome(), amdgpu::VmAccessOutcome::Complete);
      std::unique_lock request(*context.process.page_table_request_mutex());
      batch.cancel(*replacement);
    };
    scope.finish();
  }
  ASSERT_EQ(batch.completed.size(), 1u);
  EXPECT_TRUE(batch.empty());
  EXPECT_TRUE(context.wf->wait_counters().empty());
  EXPECT_TRUE(second->wait_counters().empty());
  batch.tick();
  EXPECT_EQ(batch.completed.size(), 1u);
}

TEST(GpuVmPipeline, MetadataStepBatchUnwindLeavesOrdinaryRetryAndObserverGate) {
  ShaderMetadataContext context;
  context.cu->set_debug_active(false);
  context.cu->set_plugin_group(std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{}));
  auto *second = metadata_second_wave(context);
  ASSERT_NE(second, nullptr);
  MetadataBatchPipeline batch(&context.cu->l1_vector(), context.cu->l2());
  try {
    amdgpu::GlobalMemPipeline::StepBatch scope(batch, true);
    for (auto *wf : {context.wf, second}) {
      ASSERT_EQ(batch.issue_deferred(new TestMemoryInstruction(context.state(true)), *wf),
                amdgpu::VmAccessOutcome::Complete);
    }
    throw std::runtime_error("test step interruption");
  } catch (const std::runtime_error &) {
  }
  EXPECT_TRUE(batch.completed.empty());
  batch.tick();
  EXPECT_EQ(batch.initiations, 4u);
  ASSERT_EQ(batch.completed.size(), 2u);
  EXPECT_TRUE(batch.empty());
  context.cu->set_debug_active(true);
  {
    amdgpu::GlobalMemPipeline::StepBatch scope(batch, true);
    ASSERT_EQ(batch.issue_deferred(new TestMemoryInstruction(context.state(true)), *context.wf),
              amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(context.wf->state(), amdgpu::WfState::RUNNING);
    EXPECT_EQ(batch.completed.size(), 3u);
    scope.finish();
  }
  context.cu->set_debug_active(false);
  context.cu->set_plugin_group(std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{}));
  auto plugins = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(plugins->add(std::make_unique<MemoryLifecyclePlugin>()));
  context.cu->set_plugin_group(plugins);
  {
    amdgpu::GlobalMemPipeline::StepBatch scope(batch, true);
    ASSERT_EQ(batch.issue_deferred(new TestMemoryInstruction(context.state(true)), *context.wf),
              amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(batch.completed.size(), 4u);
    scope.finish();
  }
  EXPECT_EQ(batch.initiations, 6u);
}

TEST(GpuVmPipeline, MetadataStepBatchDisabledInnerScopeRestoresCollectionAfterUnwind) {
  for (bool interrupt : {false, true}) {
    SCOPED_TRACE(interrupt);
    ShaderMetadataContext context;
    context.cu->set_debug_active(false);
    context.cu->set_plugin_group(std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{}));
    auto *second = metadata_second_wave(context);
    ASSERT_NE(second, nullptr);
    MetadataBatchPipeline batch(&context.cu->l1_vector(), context.cu->l2());
    {
      amdgpu::GlobalMemPipeline::StepBatch outer(batch, true);
      try {
        amdgpu::GlobalMemPipeline::StepBatch inner(batch, false);
        ASSERT_EQ(batch.issue_deferred(new TestMemoryInstruction(context.state(true)), *context.wf),
                  amdgpu::VmAccessOutcome::Complete);
        EXPECT_EQ(batch.completed.size(), 1u);
        EXPECT_EQ(context.wf->state(), amdgpu::WfState::RUNNING);
        EXPECT_TRUE(batch.empty());
        if (interrupt)
          throw std::runtime_error("interrupt disabled inner scope");
        inner.finish();
      } catch (const std::runtime_error &) {
      }
      ASSERT_EQ(batch.issue_deferred(new TestMemoryInstruction(context.state(true)), *second),
                amdgpu::VmAccessOutcome::Complete);
      EXPECT_EQ(batch.completed.size(), 1u);
      EXPECT_EQ(second->state(), amdgpu::WfState::VM_RETRY);
      outer.finish();
    }
    EXPECT_EQ(batch.completed.size(), 2u);
    ASSERT_EQ(batch.issue_deferred(new TestMemoryInstruction(context.state(true)), *context.wf),
              amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(batch.completed.size(), 3u);
    EXPECT_TRUE(batch.empty());
    EXPECT_TRUE(context.wf->wait_counters().empty());
    EXPECT_TRUE(second->wait_counters().empty());
  }
}

TEST(GpuVmPipeline, MetadataStepBatchEmptyFinishDisablesCollectionBeforeLaterIssue) {
  for (bool enabled : {false, true}) {
    SCOPED_TRACE(enabled);
    ShaderMetadataContext context;
    context.cu->set_debug_active(false);
    context.cu->set_plugin_group(std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{}));
    MetadataBatchPipeline batch(&context.cu->l1_vector(), context.cu->l2());
    {
      amdgpu::GlobalMemPipeline::StepBatch scope(batch, enabled);
      scope.finish();
      ASSERT_EQ(batch.issue_deferred(new TestMemoryInstruction(context.state(true)), *context.wf),
                amdgpu::VmAccessOutcome::Complete);
      EXPECT_EQ(batch.initiations, 1u);
      EXPECT_EQ(batch.completed.size(), 1u);
      EXPECT_EQ(context.wf->state(), amdgpu::WfState::RUNNING);
      EXPECT_TRUE(batch.empty());
    }
    EXPECT_TRUE(context.wf->wait_counters().empty());
  }
}

TEST(GpuVmPipeline, MetadataStepBatchNestedNondeferredIssueRestoresOuterMarker) {
  for (bool interrupt : {false, true}) {
    SCOPED_TRACE(interrupt);
    ShaderMetadataContext context;
    context.cu->set_debug_active(false);
    context.cu->set_plugin_group(std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{}));
    auto *second = metadata_second_wave(context);
    ASSERT_NE(second, nullptr);
    MetadataBatchPipeline batch(&context.cu->l1_vector(), context.cu->l2());
    bool nested = false;
    batch.on_initiate = [&](Instruction &, amdgpu::Wavefront &wf) {
      if (&wf == second && interrupt)
        throw std::runtime_error("interrupt nested initiation");
      if (&wf != context.wf || nested)
        return;
      nested = true;
      if (interrupt) {
        // The injected exception precedes the transport and ownership handoff.
        // Restore its acquired token explicitly; this test checks the scoped
        // instruction marker, not a new exception-cleanup contract for issue().
        auto instruction = std::make_unique<TestMemoryInstruction>(context.state(true));
        EXPECT_THROW(batch.issue(instruction.get(), *second), std::runtime_error);
        second->release_wait_counter(amdgpu::WaitCounterType::VMCNT);
        EXPECT_TRUE(batch.completed.empty());
      } else {
        EXPECT_EQ(batch.issue(new TestMemoryInstruction(context.state(true)), *second),
                  amdgpu::VmAccessOutcome::Complete);
        EXPECT_EQ(batch.completed.size(), 1u);
        EXPECT_EQ(second->state(), amdgpu::WfState::RUNNING);
      }
    };
    {
      amdgpu::GlobalMemPipeline::StepBatch scope(batch, true);
      ASSERT_EQ(batch.issue_deferred(new TestMemoryInstruction(context.state(true)), *context.wf),
                amdgpu::VmAccessOutcome::Complete);
      EXPECT_TRUE(nested);
      EXPECT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
      EXPECT_EQ(batch.completed.size(), interrupt ? 0u : 1u);
      scope.finish();
    }
    EXPECT_EQ(batch.initiations, 3u);
    EXPECT_EQ(batch.completed.size(), interrupt ? 1u : 2u);
    EXPECT_TRUE(batch.empty());
    EXPECT_TRUE(context.wf->wait_counters().empty());
    EXPECT_TRUE(second->wait_counters().empty());
  }
}

class MetadataPreparationBackend final : public amdgpu::AddressSpaceTranslator,
                                         public amdgpu::PhysicalMemoryAccess {
public:
  explicit MetadataPreparationBackend(amdgpu::GpuVmAccess access)
      : AddressSpaceTranslator(true), access_(std::move(access)) {}
  amdgpu::VmTranslationResult translate(uint64_t address, size_t size,
                                        amdgpu::VmAccessKind) const override {
    return {
        .outcome = amdgpu::VmAccessOutcome::Complete,
        .translation = {.domain = amdgpu::VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = size,
                        .mtype = amdgpu::Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }
  amdgpu::VmAccessOutcome read(amdgpu::VmMemoryDomain, uint64_t address,
                               std::span<std::byte> bytes) override {
    return access_.read(address, bytes);
  }
  amdgpu::VmAccessOutcome write(amdgpu::VmMemoryDomain, uint64_t address,
                                std::span<const std::byte> bytes) override {
    return access_.write(address, bytes);
  }
  std::unique_ptr<amdgpu::VmRamLeaseRequest>
  prepare_ram_lease(amdgpu::PhysicalMemoryAccess &,
                    std::span<const amdgpu::VmRamRange> ranges) const override {
    ++preparations;
    if (prepare_hook)
      prepare_hook();
    auto request = access_.prepare_ram_lease(ranges);
    if (!request)
      return nullptr;
    struct Request final : amdgpu::VmRamLeaseRequest {
      std::unique_ptr<amdgpu::VmRamLeaseRequest> inner;
      const MetadataPreparationBackend &owner;
      Request(std::unique_ptr<amdgpu::VmRamLeaseRequest> request,
              const MetadataPreparationBackend &backend)
          : inner(std::move(request)), owner(backend) {}
      ~Request() override {
        inner.reset();
        ++owner.destructions;
        errno = E2BIG;
      }
      std::span<std::byte> bytes(size_t index) const override { return inner->bytes(index); }
      bool try_acquire() override { return inner->try_acquire(); }
      void release() override { inner->release(); }
    };
    return std::make_unique<Request>(std::move(request), *this);
  }
  mutable unsigned preparations = 0, destructions = 0;
  std::function<void()> prepare_hook;

private:
  amdgpu::GpuVmAccess access_;
};

TEST(GpuVmPipeline, MetadataStepBatchPreparationCanCancelAnyQueuedWaveAndReuseItsSlot) {
  // Cancel the current, an already prepared, or a later instruction. cancel()
  // rebuilds the entire queue even when the affected wave is not the current one.
  for (const auto &[cancel_index, trigger] :
       {std::pair{0u, 1u}, std::pair{0u, 2u}, std::pair{2u, 1u}}) {
    SCOPED_TRACE(::testing::Message() << cancel_index << '/' << trigger);
    ShaderMetadataContext context;
    context.cu->set_debug_active(false);
    context.cu->set_plugin_group(std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{}));
    const auto access = context.wf->snapshot_vm_access();
    ASSERT_TRUE(access);
    auto backend = std::make_shared<MetadataPreparationBackend>(*access);
    const auto handle =
        context.sim.soc->gpu_vm().register_unrouted_address_space(79, backend, backend, {}, true);
    ASSERT_TRUE(handle);
    auto *second = metadata_second_wave(context);
    auto *third = metadata_second_wave(context);
    ASSERT_NE(second, nullptr);
    ASSERT_NE(third, nullptr);
    const std::array waves{context.wf, second, third};
    for (auto *wf : waves)
      wf->set_address_space(handle);
    MetadataBatchPipeline batch(&context.cu->l1_vector(), context.cu->l2());
    amdgpu::Wavefront *replacement = nullptr;
    bool cancelled = false;
    backend->prepare_hook = [&] {
      if (cancelled || backend->preparations != trigger)
        return;
      cancelled = true;
      auto *victim = waves[cancel_index];
      const auto generation = victim->dispatch_generation();
      batch.cancel(*victim);
      context.cu->free_wavefront_resources(*victim);
      replacement = metadata_second_wave(context);
      ASSERT_EQ(replacement, victim);
      ASSERT_NE(replacement->dispatch_generation(), generation);
      replacement->set_address_space(handle);
      // New work issued reentrantly does not join the snapshotted step batch.
      batch.defer_unavailable(new TestMemoryInstruction(context.state(true)), *replacement);
      errno = ERANGE;
    };
    batch.on_complete = [&](amdgpu::Wavefront &wf) {
      EXPECT_NE(&wf, replacement);
      errno = EDOM;
    };
    {
      amdgpu::GlobalMemPipeline::StepBatch scope(batch, true);
      for (auto *wf : waves) {
        ASSERT_EQ(batch.issue_deferred(new TestMemoryInstruction(context.state(true)), *wf),
                  amdgpu::VmAccessOutcome::Complete);
      }
      errno = EILSEQ;
      scope.finish();
      EXPECT_EQ(errno, EDOM); // completion wins over prepared-storage cleanup
    }
    ASSERT_TRUE(cancelled);
    ASSERT_EQ(batch.completed.size(), 2u);
    EXPECT_EQ(batch.completed.front().error_number, EILSEQ);
    EXPECT_EQ(batch.completed.back().error_number, EDOM);
    EXPECT_EQ(backend->destructions, backend->preparations);
    EXPECT_EQ(batch.initiations, 3u); // the surviving prepared requests did not replay
    ASSERT_NE(replacement, nullptr);
    EXPECT_EQ(replacement->state(), amdgpu::WfState::VM_RETRY);
    EXPECT_FALSE(replacement->wait_counters().empty());
    EXPECT_FALSE(batch.empty());
    batch.on_complete = {};
    batch.tick();
    EXPECT_EQ(batch.completed.size(), 3u);
    EXPECT_TRUE(batch.empty());
    EXPECT_EQ(replacement->state(), amdgpu::WfState::RUNNING);
    EXPECT_TRUE(replacement->wait_counters().empty());
  }
}

TEST(GpuVmPipeline, MetadataStepBatchPreparationFailureLeavesOwnedRetryWithoutEffects) {
  ShaderMetadataContext context;
  context.cu->set_debug_active(false);
  context.cu->set_plugin_group(std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{}));
  context.key(1, 0);
  const auto before_pixels = context.pixels, before_metadata = context.metadata;
  const auto access = context.wf->snapshot_vm_access();
  ASSERT_TRUE(access);
  auto backend = std::make_shared<MetadataPreparationBackend>(*access);
  const auto handle =
      context.sim.soc->gpu_vm().register_unrouted_address_space(79, backend, backend, {}, true);
  ASSERT_TRUE(handle);
  auto *second = metadata_second_wave(context);
  ASSERT_NE(second, nullptr);
  context.wf->set_address_space(handle);
  second->set_address_space(handle);
  MetadataBatchPipeline batch(&context.cu->l1_vector(), context.cu->l2());
  backend->prepare_hook = [&] {
    if (backend->preparations == 2)
      throw std::bad_alloc();
  };
  bool failed = false;
  try {
    amdgpu::GlobalMemPipeline::StepBatch scope(batch, true);
    for (auto *wf : {context.wf, second}) {
      ASSERT_EQ(batch.issue_deferred(new TestMemoryInstruction(context.state(true)), *wf),
                amdgpu::VmAccessOutcome::Complete);
    }
    errno = EILSEQ;
    scope.finish();
  } catch (const std::bad_alloc &) {
    failed = true;
  }
  EXPECT_TRUE(failed);
  EXPECT_EQ(errno, EILSEQ);
  EXPECT_EQ(context.pixels, before_pixels);
  EXPECT_EQ(context.metadata, before_metadata);
  EXPECT_TRUE(batch.completed.empty());
  EXPECT_FALSE(batch.empty());
  backend->prepare_hook = {};
  batch.tick();
  EXPECT_EQ(batch.completed.size(), 2u);
  EXPECT_TRUE(batch.empty());
  for (auto *wf : {context.wf, second}) {
    EXPECT_TRUE(wf->wait_counters().empty());
    EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
  }
}

TEST(GpuVmPipeline, MetadataStepBatchKeepsArchitecturalWritebackAndWaitTokens) {
  ShaderMetadataContext context;
  context.cu->set_debug_active(false);
  context.cu->set_plugin_group(std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{}));
  auto *second = metadata_second_wave(context);
  ASSERT_NE(second, nullptr);
  amdgpu::GlobalMemPipeline batch(&context.cu->l1_vector(), context.cu->l2());
  constexpr uint32_t sentinel = 0xdeadbeef;
  {
    amdgpu::GlobalMemPipeline::StepBatch scope(batch, true);
    for (auto *wf : {context.wf, second}) {
      const auto destination = wf->vgpr_alloc().base + 1;
      for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
        context.cu->write_vgpr(destination, lane, sentinel);
      auto d = context.state(true);
      d->dst_reg_base = destination;
      ASSERT_EQ(batch.issue_deferred(new TestMemoryInstruction(std::move(d)), *wf),
                amdgpu::VmAccessOutcome::Complete);
      EXPECT_EQ(context.cu->read_vgpr(destination, 0), sentinel);
      EXPECT_FALSE(wf->wait_counters().empty());
      // The issue caller advances exactly once when the pipeline retains ownership.
      wf->pc += 8;
    }
    scope.finish();
  }
  EXPECT_TRUE(batch.empty());
  for (auto *wf : {context.wf, second}) {
    uint32_t expected;
    std::memcpy(&expected, context.pixels.data() + context.address(1) - context.pixel_base, 4);
    EXPECT_EQ(context.cu->read_vgpr(wf->vgpr_alloc().base + 1, 0), expected);
    EXPECT_EQ(context.cu->read_vgpr(wf->vgpr_alloc().base + 1, 1), sentinel);
    EXPECT_EQ(wf->pc, 8u);
    EXPECT_TRUE(wf->wait_counters().empty());
    EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
  }
}
TEST(GpuVmPipeline, MetadataStepBatchKeepsQuantumProgressAndIndependentYield) {
  for (bool sleep_peer : {false, true}) {
    SCOPED_TRACE(sleep_peer);
    ShaderMetadataContext context;
    amdgpu::ComputeUnitCore::Config config{.arch = ROCJITSU_CODE_ARCH_RDNA3,
                                           .num_wf_slots = 4,
                                           .sgprs_per_wf = 106,
                                           .vgprs_per_wf = 256,
                                           .lds_size_kb = 64,
                                           .functional_quantum = 3};
    auto cu = amdgpu::ComputeUnitCore::create("metadata-batch-cu", config, context.sim.memory,
                                              context.cu->l2());
    ASSERT_TRUE(cu);
    cu->set_gpu_vm(&context.sim.soc->gpu_vm());
    cu->set_pool_driven(true);
    std::array<uint32_t, 1024> code;
    code.fill(rdna3::build_sopp(rdna3::kSNopSopp)[0]);
    const auto load = rdna3::build_mimg(rdna3::kImageLoadMimg,
                                        {.dim = 1, .dmask = 1, .vaddr = 0, .vdata = 4, .srsrc = 2});
    std::ranges::copy(load, code.begin());
    code[64] = rdna3::build_sopp(rdna3::kSSleepSopp, {.simm16 = 1})[0];
    constexpr uint64_t code_base = 0x400000;
    context.process.map_pages(code_base, code.data(), sizeof(code), amdgpu::Mtype::RW,
                              context.sealed);
    const std::array<uint32_t, 8> resource{uint32_t(context.pixel_base >> 8),
                                           (20u << 20) | (((context.width - 1) & 3u) << 30),
                                           ((context.width - 1) >> 2) |
                                               ((context.height - 1) << 14),
                                           (9u << 28) | (context.swizzle << 20) | 0xfac,
                                           0,
                                           0,
                                           (1u << 21) | (1u << 19),
                                           uint32_t(context.metadata_base >> 16)};
    std::array<amdgpu::Wavefront *, 2> waves{};
    for (uint32_t index = 0; index < waves.size(); ++index) {
      auto *wf = cu->dispatch_wf(index, code_base, 32, 32, 32);
      ASSERT_NE(wf, nullptr);
      waves[index] = wf;
      wf->set_process_id(79);
      wf->set_address_space(context.handle);
      wf->set_exec(1);
      for (uint32_t i = 0; i < resource.size(); ++i)
        wf->debug_write_sgpr(8 + i, resource[i]);
      wf->debug_write_vgpr(0, 0, 1);
      wf->debug_write_vgpr(1, 0, 0);
      wf->debug_write_vgpr(4, 0, 0xdeadbeef);
    }
    if (sleep_peer) {
      auto *wf = cu->dispatch_wf(2, code_base + 256, 32, 32, 32);
      ASSERT_NE(wf, nullptr);
      wf->set_process_id(79);
      wf->set_address_space(context.handle);
    }
    const auto result = cu->run_quantum();
    EXPECT_TRUE(result.ran);
    EXPECT_EQ(result.yielded, sleep_peer);
    EXPECT_EQ(result.iterations, sleep_peer ? 1u : 3u);
    for (auto *wf : waves) {
      uint32_t expected;
      std::memcpy(&expected, context.pixels.data() + context.address(1) - context.pixel_base, 4);
      EXPECT_EQ(wf->debug_read_vgpr(4, 0), expected);
      EXPECT_EQ(wf->pc, code_base + (sleep_peer ? 8 : 16));
      EXPECT_TRUE(wf->wait_counters().empty());
      EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
      EXPECT_FALSE(wf->instruction_execution_failed());
    }
  }
}

} // namespace
