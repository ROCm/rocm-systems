// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "async_vm_test_support.h"
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
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
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
        .translation = {.domain = translation_domain,
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

  bool atomic_allowed_ = true;
  amdgpu::VmMemoryDomain translation_domain = amdgpu::VmMemoryDomain::System;
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
  amdgpu::ComputeUnitCore *cu = nullptr;
  amdgpu::Wavefront *wf = nullptr;
  amdgpu::AddressSpaceHandle address_space;

  void initialize(bool legacy_cache_compatible = false) {
    ASSERT_NO_FATAL_FAILURE(sim.initialize());
    cu = sim.cu();
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
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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
    ASSERT_NO_FATAL_FAILURE(context.initialize());
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
    ASSERT_NO_FATAL_FAILURE(context.initialize());
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

TEST(GpuVmPipeline, QuantumRetainsAdmissionSnapshotAcrossReturnAndException) {
  for (bool throw_from_plugin : {false, true}) {
    SCOPED_TRACE(throw_from_plugin);
    TranslatedPipelineContext context;
    ASSERT_NO_FATAL_FAILURE(context.initialize());
    ASSERT_NE(context.wf, nullptr);
    constexpr uint64_t kCode = 0x1000;
    constexpr uint32_t kNop = 0xBF800000u;
    context.external->store(kCode, std::array{kNop, kNop, S_ENDPGM_GFX12, 0u});
    auto &vm = context.sim.soc->gpu_vm();
    {
      auto admitted = vm.snapshot_pinned(context.address_space);
      ASSERT_TRUE(admitted);
      context.wf->set_vm_access(std::make_shared<amdgpu::GpuVmAccess>(std::move(*admitted)));
    }
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

    auto replacement = std::make_shared<TestExternalAddressSpace>();
    replacement->store(kCode, std::array{kNop, S_ENDPGM_GFX12, 0u, 0u});
    ASSERT_TRUE(vm.replace_translated(context.address_space, replacement, replacement));
    std::weak_ptr<TestExternalAddressSpace> old_backing = context.external;
    context.external.reset();
    EXPECT_FALSE(old_backing.expired());

    // Resume after the callback, including when it threw before the PC advance.
    events->after_instruction = {};
    context.wf->pc = kCode + sizeof(uint32_t);
    context.cu->set_functional_quantum(4);
    EXPECT_TRUE(context.cu->run_quantum().ran);
    EXPECT_EQ(events->nops, 2u);
    EXPECT_FALSE(context.cu->has_active_wfs());
    EXPECT_TRUE(old_backing.expired());
  }
}

TEST(GpuVmPipeline, WaveUsesAdmissionSnapshotAcrossRootReplacement) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);

  constexpr uint64_t kAddress = 0x100;
  constexpr uint32_t kOriginal = 0x11223344;
  constexpr uint32_t kReplacement = 0xaabbccdd;
  context.external->store(kAddress, kOriginal);

  std::optional<amdgpu::GpuVmAccess> admitted =
      context.sim.soc->gpu_vm().snapshot_pinned(context.address_space);
  ASSERT_TRUE(admitted);
  context.wf->set_vm_access(std::make_shared<amdgpu::GpuVmAccess>(std::move(*admitted)));

  auto replacement = std::make_shared<TestExternalAddressSpace>();
  replacement->store(kAddress, kReplacement);
  ASSERT_TRUE(context.sim.soc->gpu_vm().replace_translated(context.address_space, replacement,
                                                           replacement));

  std::array<uint8_t, sizeof(uint32_t)> bytes{};
  EXPECT_EQ(context.wf->read_gpu_memory(kAddress, bytes), amdgpu::VmAccessOutcome::Complete);
  uint32_t observed = 0;
  std::memcpy(&observed, bytes.data(), sizeof(observed));
  EXPECT_EQ(observed, kOriginal);
}

TEST(GpuVmPipeline, RevokedWaveCannotFetchFromReplacementRoot) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);
  constexpr uint64_t kCode = 0x1000;
  constexpr uint32_t kNop = 0xBF800000u;
  context.external->store(kCode, std::array{kNop, S_ENDPGM_GFX12, 0u, 0u});

  auto &vm = context.sim.soc->gpu_vm();
  auto admitted = vm.snapshot_pinned(context.address_space);
  ASSERT_TRUE(admitted);
  context.wf->set_vm_access(std::make_shared<amdgpu::GpuVmAccess>(std::move(*admitted)));
  context.wf->pc = kCode;
  ASSERT_TRUE(vm.invalidate(context.address_space));

  auto replacement = std::make_shared<TestExternalAddressSpace>();
  replacement->store(kCode, std::array{kNop, S_ENDPGM_GFX12, 0u, 0u});
  ASSERT_TRUE(vm.replace_translated(context.address_space, replacement, replacement));
  for (uint32_t step = 0; step < 4 && !context.cu->is_idle(); ++step)
    static_cast<void>(context.cu->step());

  EXPECT_TRUE(context.cu->is_idle());
  EXPECT_EQ(replacement->read_calls, 0u);
}

TEST(GpuVmPipeline, WaveAdmissionSnapshotRevocationTerminatesAfterUnregister) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);

  std::optional<amdgpu::GpuVmAccess> admitted =
      context.sim.soc->gpu_vm().snapshot_pinned(context.address_space);
  ASSERT_TRUE(admitted);
  context.wf->set_vm_access(std::make_shared<amdgpu::GpuVmAccess>(std::move(*admitted)));
  ASSERT_TRUE(context.sim.soc->gpu_vm().unregister_address_space(context.address_space));

  std::array<uint8_t, sizeof(uint32_t)> bytes{};
  EXPECT_EQ(context.wf->read_gpu_memory(0x100, bytes), amdgpu::VmAccessOutcome::Revoked);
  for (uint32_t step = 0; step < 4 && !context.cu->is_idle(); ++step)
    static_cast<void>(context.cu->step());
  EXPECT_TRUE(context.cu->is_idle());
  EXPECT_EQ(context.wf->vm_access(), nullptr);
}

TEST(GpuVmPipeline, TranslatedScalarLoadAndStoreUseExternalBacking) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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

TEST(GpuVmPipeline, DdsStepReportsUnsupportedInstructionAndPcBeforeFaulting) {
  for (bool mixed : {false, true}) {
    for (bool load : {false, true}) {
      for (uint32_t dds_lane : {0u, 1u}) {
        SCOPED_TRACE(testing::Message()
                     << "mixed=" << mixed << " load=" << load << " DDS lane=" << dds_lane);
        TranslatedPipelineContext context;
        ASSERT_NO_FATAL_FAILURE(context.initialize());
        ASSERT_NE(context.wf, nullptr);
        constexpr uint64_t shared_base = uint64_t{1} << 48;
        constexpr uint64_t pc = 0x1000;
        context.wf->set_apertures(shared_base, shared_base + UINT32_MAX, 0, 0);
        context.wf->pc = pc;
        context.wf->set_exec(mixed ? 3 : uint64_t{1} << dds_lane);
        const auto words = cdna5::build_vflat(
            load ? cdna5::kFlatLoadB32Vflat : cdna5::kFlatStoreB32Vflat,
            {.saddr = 124, .vdst = uint8_t(load ? 16 : 0), .vsrc = 8, .vaddr = 0});
        context.external->store(pc, std::array{words[0], words[1], words[2], S_ENDPGM_GFX12});
        const uint32_t vb = context.wf->vgpr_alloc().base;
        for (uint32_t lane = 0; lane < 2; ++lane) {
          const uint64_t address = lane == dds_lane ? shared_base + 0xffffffc0 : 0x2000;
          context.cu->write_vgpr(vb, lane, static_cast<uint32_t>(address));
          context.cu->write_vgpr(vb + 1, lane, static_cast<uint32_t>(address >> 32));
          context.cu->write_vgpr(vb + 8, lane, 42);
        }
        testing::internal::CaptureStderr();
        static_cast<void>(context.cu->step());
        const std::string diagnostic = testing::internal::GetCapturedStderr();
        const std::string prefix = "unsupported FLAT DDS execution: ";
        const auto warning = diagnostic.find(prefix);
        EXPECT_NE(warning, std::string::npos) << diagnostic;
        EXPECT_EQ(diagnostic.find(prefix, warning == std::string::npos ? 0 : warning + 1),
                  std::string::npos)
            << diagnostic;
        EXPECT_NE(diagnostic.find(load ? "flat_load_b32" : "flat_store_b32"), std::string::npos);
        EXPECT_NE(diagnostic.find("pc=0x1000"), std::string::npos);
        EXPECT_NE(diagnostic.find("DDS backing is not implemented"), std::string::npos);
        EXPECT_TRUE(context.wf->is_halted());
        EXPECT_TRUE(context.wf->wait_counters().empty());
        EXPECT_EQ(context.external->write_calls, 0u);
        for (const auto &read : context.external->read_history)
          EXPECT_NE(read.address, 0x2000u);
      }
    }
  }
}

TEST(GpuVmPipeline, NonlegacySystemReadsProbeBeforeBackingAndAfterStampedCompletion) {
  amdgpu::test::AsyncVmHarness owner_sim;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto memory =
      std::make_shared<amdgpu::test::AsyncVmMemory>(*owner_sim.owner, 0x400, 0x100, 0x140);
  memory->backing->read_freshness = std::make_shared<CanonicalLineFreshness>();
  memory->backing->read_backing_id = 17;
  const auto address_space = sim.soc->gpu_vm().register_address_space(77, memory, memory);
  ASSERT_TRUE(address_space);
  auto *cu = sim.cu();
  cu->set_gpu_vm(&sim.soc->gpu_vm());
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(77);
  wf->set_address_space(address_space);
  wf->set_exec(1);

  constexpr uint64_t kScalarAddress = 0x100, kVectorAddress = 0x120;
  constexpr uint32_t kScalarValue = 0x11223344u, kVectorValue = 0x55667788u;
  memory->store(kScalarAddress, std::as_bytes(std::span(&kScalarValue, 1)));
  memory->store(kVectorAddress, std::as_bytes(std::span(&kVectorValue, 1)));
  std::vector<bool> scalar_stamps, vector_stamps;
  const auto probe = [&](uint64_t wanted, uint32_t prior_effects, std::vector<bool> &seen) {
    return [&, wanted, prior_effects](uint32_t process_id, const amdgpu::GpuVmAccess &access,
                                      uint64_t address, std::span<std::byte> bytes,
                                      const amdgpu::VmCompletedRead *completed) {
      EXPECT_EQ(process_id, 77u);
      EXPECT_EQ(address, wanted);
      EXPECT_EQ(bytes.size(), sizeof(uint32_t));
      EXPECT_FALSE(access.revoked());
      const auto *stamp = completed ? &completed->stamp : nullptr;
      EXPECT_EQ(memory->backing->read_effects.load(), prior_effects + (stamp ? 1u : 0u))
          << "each probe must bracket its own backing read";
      if (stamp) {
        EXPECT_TRUE(*stamp);
        EXPECT_TRUE(stamp->covers(memory->backing->read_backing_id, address, bytes.size()));
      }
      seen.push_back(stamp != nullptr);
      return false;
    };
  };

  amdgpu::ScalarMemPipeline scalar(&cu->l1_scalar());
  scalar.set_peer_cache_read_handler(probe(kScalarAddress, 0, scalar_stamps));
  auto scalar_load = std::make_unique<amdgpu::ScalarMemState>();
  scalar_load->addr = kScalarAddress;
  scalar_load->dst_register = {amdgpu::ScalarRegisterStorage::SGPR, 8, 1};
  scalar_load->num_dwords = 1;
  scalar_load->is_load = true;
  EXPECT_EQ(scalar.issue_deferred(new TestMemoryInstruction(std::move(scalar_load)), *wf),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(scalar_stamps, (std::vector<bool>{false}));
  EXPECT_EQ(memory->backing->read_effects, 0u);
  (void)owner_sim.engine.step();
  scalar.tick();
  EXPECT_TRUE(scalar.empty());
  EXPECT_EQ(scalar_stamps, (std::vector<bool>{false, true}));
  EXPECT_EQ(cu->read_sgpr_storage(wf->sgpr_alloc().base + 8), kScalarValue);

  amdgpu::GlobalMemPipeline vector(&cu->l1_vector(), cu->l2());
  vector.set_peer_cache_read_handler(probe(kVectorAddress, 1, vector_stamps));
  auto vector_load = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  vector_load->elem_size = sizeof(uint32_t);
  vector_load->num_elems = 1;
  vector_load->is_load = true;
  vector_load->wf_size = wf->wf_size();
  vector_load->exec_mask = 1;
  vector_load->lane_mask = 1;
  vector_load->dst_reg_base = wf->vgpr_alloc().base + 8;
  vector_load->per_lane_addr[0] = kVectorAddress;
  EXPECT_EQ(vector.issue_deferred(new TestMemoryInstruction(std::move(vector_load)), *wf),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(vector_stamps, (std::vector<bool>{false}));
  (void)owner_sim.engine.step();
  vector.tick();
  EXPECT_TRUE(vector.empty());
  EXPECT_EQ(vector_stamps, (std::vector<bool>{false, true}));
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + 8, 0), kVectorValue);

  const auto access = sim.soc->gpu_vm().snapshot(address_space);
  ASSERT_TRUE(access);
  std::array<uint8_t, sizeof(uint32_t)> cached{};
  EXPECT_FALSE(cu->l1_scalar().try_read_peer_cached(kScalarAddress, cached, 77, 1,
                                                    access->cache_identity()));
  EXPECT_FALSE(cu->l1_vector().try_read_peer_cached(kVectorAddress, cached, 77, 1,
                                                    access->cache_identity()));
  EXPECT_FALSE(
      cu->l2()->try_read_peer_cached(kScalarAddress, cached, 77, 1, access->cache_identity()));
  EXPECT_FALSE(
      cu->l2()->try_read_peer_cached(kVectorAddress, cached, 77, 1, access->cache_identity()));
}

TEST(GpuVmPipeline, NonlegacyLocalReadSkipsSystemCacheProbe) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
  ASSERT_NE(context.wf, nullptr);
  context.external->translation_domain = amdgpu::VmMemoryDomain::Local;
  constexpr uint64_t kAddress = 0x180;
  constexpr uint32_t kValue = 0x12345678u;
  context.external->store(kAddress, kValue);
  uint32_t probes = 0;
  amdgpu::ScalarMemPipeline pipeline(&context.cu->l1_scalar());
  pipeline.set_peer_cache_read_handler([&](uint32_t, const amdgpu::GpuVmAccess &, uint64_t,
                                           std::span<std::byte>, const amdgpu::VmCompletedRead *) {
    ++probes;
    return false;
  });
  auto load = std::make_unique<amdgpu::ScalarMemState>();
  load->addr = kAddress;
  load->dst_register = {amdgpu::ScalarRegisterStorage::SGPR, 9, 1};
  load->num_dwords = 1;
  load->is_load = true;
  EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(load)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(context.cu->read_sgpr_storage(context.wf->sgpr_alloc().base + 9), kValue);
  EXPECT_EQ(context.external->read_calls, 1u);
  EXPECT_EQ(probes, 0u);
}

TEST(GpuVmPipeline, ScratchWithoutBackingIsATerminalFault) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
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
  std::array<std::byte, sizeof(rocjitsu::amdhsa::kernel_descriptor_t) + sizeof(kEndpgm)>
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

  amdgpu::ComputeQueueConfig stalled_queue{};
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
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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

TEST(GpuVmPipeline, RevokedDataRetryTerminatesInsteadOfSpinning) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);

  std::optional<amdgpu::GpuVmAccess> admitted =
      context.sim.soc->gpu_vm().snapshot_pinned(context.address_space);
  ASSERT_TRUE(admitted);
  context.wf->set_vm_access(std::make_shared<amdgpu::GpuVmAccess>(std::move(*admitted)));

  constexpr uint64_t kAddress = 0x2000;
  constexpr uint32_t kDestination = 1;
  context.external->store(kAddress, 0x12345678u);
  context.external->unavailable_read_call = 1;

  auto load = std::make_unique<amdgpu::ScalarMemState>();
  load->addr = kAddress;
  load->dst_register = {amdgpu::ScalarRegisterStorage::SGPR, kDestination, 1};
  load->num_dwords = 1;
  load->is_load = true;

  amdgpu::ScalarMemPipeline pipeline(&context.cu->l1_scalar());
  uint32_t fault_count = 0;
  amdgpu::VmAccessOutcome fault_outcome = amdgpu::VmAccessOutcome::Complete;
  pipeline.set_fault_handler([&](amdgpu::Wavefront &, amdgpu::VmAccessOutcome outcome) {
    ++fault_count;
    fault_outcome = outcome;
  });

  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(load)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
  ASSERT_TRUE(context.sim.soc->gpu_vm().invalidate(context.address_space));

  pipeline.tick();

  EXPECT_TRUE(pipeline.empty());
  EXPECT_TRUE(context.wf->wait_counters().empty());
  EXPECT_EQ(fault_count, 1u);
  EXPECT_EQ(fault_outcome, amdgpu::VmAccessOutcome::Revoked);
}

TEST(GpuVmPipeline, TranslatedScalarReadAndWriteResumeWithoutReplayingCompletedChunks) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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

TEST(GpuVmPipeline, MaskedScalarStoreRetryPreservesCompletedWordsAndPartialWord) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);
  context.external->translation_granule = sizeof(uint16_t);

  constexpr uint64_t kAddress = 0x1c0;
  constexpr uint32_t kSentinel = 0x55555555;
  constexpr std::array<uint32_t, 4> kStored = {0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00};
  for (uint32_t index = 0; index < kStored.size(); ++index)
    context.external->store(kAddress + index * sizeof(uint32_t), kSentinel);
  // Complete the first DWORD and half the second before backing memory stalls.
  context.external->unavailable_write_call = 4;
  auto store = std::make_unique<amdgpu::ScalarMemState>();
  store->addr = kAddress;
  store->num_dwords = kStored.size();
  store->is_load = false;
  store->load_dword_mask = 0x7;
  std::ranges::copy(kStored, store->store_data);
  amdgpu::ScalarMemPipeline pipeline(&context.cu->l1_scalar());
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(store)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_FALSE(context.wf->wait_counters().empty());
  EXPECT_EQ(context.external->load<uint32_t>(kAddress), kStored[0]);
  EXPECT_EQ(context.external->load<uint32_t>(kAddress + 4), 0x55557788u);

  // A peer can change already-written bytes before retry. Replaying them would
  // overwrite its updates, so check both completed-word and within-word progress.
  context.external->store(kAddress, uint32_t{0x12345678});
  context.external->store(kAddress + 4, uint16_t{0xabcd});
  pipeline.tick();

  EXPECT_TRUE(pipeline.empty());
  EXPECT_EQ(context.wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_TRUE(context.wf->wait_counters().empty());
  EXPECT_EQ(context.external->load<uint32_t>(kAddress), 0x12345678u);
  EXPECT_EQ(context.external->load<uint32_t>(kAddress + 4), 0x5566abcdu);
  EXPECT_EQ(context.external->load<uint32_t>(kAddress + 8), kStored[2]);
  EXPECT_EQ(context.external->load<uint32_t>(kAddress + 12), kSentinel);
  constexpr std::array<uint32_t, 7> kExpectedOffsets = {0, 2, 4, 6, 6, 8, 10};
  ASSERT_EQ(context.external->write_history.size(), kExpectedOffsets.size());
  for (uint32_t index = 0; index < kExpectedOffsets.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(context.external->write_history[index].address, kAddress + kExpectedOffsets[index]);
    EXPECT_EQ(context.external->write_history[index].size, sizeof(uint16_t));
  }
}

TEST(GpuVmPipeline, FullyMaskedScalarStoreDoesNotAcquireRevokedAddressSpace) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);
  ASSERT_TRUE(context.sim.soc->gpu_vm().invalidate(context.address_space));

  auto store = std::make_unique<amdgpu::ScalarMemState>();
  store->addr = 0x1c0;
  store->num_dwords = 4;
  store->is_load = false;
  store->load_dword_mask = 0;
  amdgpu::ScalarMemPipeline pipeline(&context.cu->l1_scalar());
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(store)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_TRUE(pipeline.empty());
  EXPECT_EQ(context.wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_TRUE(context.wf->wait_counters().empty());
  EXPECT_EQ(context.external->write_calls, 0u);
}

TEST(GpuVmPipeline, OwnedOwnerEventRetiresScalarLoadAndStoreOnce) {
  amdgpu::test::AsyncVmHarness owner_sim;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto memory =
      std::make_shared<amdgpu::test::AsyncVmMemory>(*owner_sim.owner, 0x400, 0x104, 0x108);
  const auto address_space = sim.soc->gpu_vm().register_translated(77, memory, memory);
  auto *cu = sim.cu();
  cu->set_gpu_vm(&sim.soc->gpu_vm());
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(77);
  wf->set_address_space(address_space);
  wf->set_exec(0x3u);

  constexpr uint32_t kDestination = 8;
  const std::array<uint32_t, 2> loaded{0x11223344u, 0x55667788u};
  memory->store(0x100, std::as_bytes(std::span(loaded)));
  amdgpu::ScalarMemPipeline pipeline(&cu->l1_scalar());
  auto load = std::make_unique<amdgpu::ScalarMemState>();
  load->addr = 0x100;
  load->dst_register = {amdgpu::ScalarRegisterStorage::SGPR, kDestination, 2};
  load->num_dwords = 2;
  load->is_load = true;
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(load)), *wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(memory->backing->posts, 1u);
  EXPECT_EQ(memory->backing->read_effects, 0u);
  pipeline.tick();
  EXPECT_EQ(wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(memory->backing->posts, 1u);
  (void)owner_sim.engine.step();
  pipeline.tick();
  EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_TRUE(wf->wait_counters().empty());
  EXPECT_EQ(cu->read_sgpr_storage(wf->sgpr_alloc().base + kDestination), loaded[0]);
  EXPECT_EQ(cu->read_sgpr_storage(wf->sgpr_alloc().base + kDestination + 1), loaded[1]);
  EXPECT_EQ(memory->backing->read_effects, 1u);

  const std::array<uint32_t, 2> stored{0xa1b2c3d4u, 0xe5f60718u};
  auto store = std::make_unique<amdgpu::ScalarMemState>();
  store->addr = 0x100;
  store->num_dwords = 2;
  store->is_load = false;
  std::ranges::copy(stored, store->store_data);
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(store)), *wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(memory->backing->posts, 2u);
  EXPECT_EQ(memory->backing->write_effects, 0u);
  (void)owner_sim.engine.step();
  pipeline.tick();
  EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_TRUE(wf->wait_counters().empty());
  EXPECT_EQ(memory->load(0x100, sizeof(stored)),
            std::vector<std::byte>(std::as_bytes(std::span(stored)).begin(),
                                   std::as_bytes(std::span(stored)).end()));
  EXPECT_EQ(memory->backing->write_effects, 1u);
  EXPECT_EQ(memory->backing->posts, 2u);
}

TEST(GpuVmPipeline, MaskedScalarPeerStoreRetainsDeferredTransferToken) {
  amdgpu::test::AsyncVmHarness owner_sim;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto memory =
      std::make_shared<amdgpu::test::AsyncVmMemory>(*owner_sim.owner, 0x400, 0x100, 0x104);
  const auto address_space = sim.soc->gpu_vm().register_address_space(77, memory, memory, {}, true);
  ASSERT_TRUE(address_space);
  auto *cu = sim.cu();
  cu->set_gpu_vm(&sim.soc->gpu_vm());
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(77);
  wf->set_address_space(address_space);

  constexpr uint32_t kValue = 0x11223344u;
  auto store = std::make_unique<amdgpu::ScalarMemState>();
  store->addr = 0x100;
  store->num_dwords = 4;
  store->is_load = false;
  store->load_dword_mask = 1;
  store->store_data[0] = kValue;
  amdgpu::ScalarMemPipeline pipeline(&cu->l1_scalar());
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(store)), *wf),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(memory->backing->posts, 1u);
  EXPECT_EQ(memory->backing->write_effects, 0u);

  ASSERT_TRUE(owner_sim.engine.step());
  pipeline.tick();
  EXPECT_TRUE(pipeline.empty());
  EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_TRUE(wf->wait_counters().empty());
  EXPECT_EQ(memory->backing->posts, 1u);
  EXPECT_EQ(memory->backing->write_effects, 1u);
  EXPECT_EQ(memory->load(0x100, sizeof(kValue)),
            std::vector<std::byte>(std::as_bytes(std::span(&kValue, 1)).begin(),
                                   std::as_bytes(std::span(&kValue, 1)).end()));
  EXPECT_EQ(memory->load(0x104, sizeof(kValue)), std::vector<std::byte>(sizeof(kValue)));
}

TEST(GpuVmPipeline, MixedScalarPeerDataKeepsLocalCachePath) {
  amdgpu::test::AsyncVmHarness owner_sim;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  // Put the local word and deferred peer word on separate cache lines. A
  // requester-local fill must remain inside its own complete cache line.
  constexpr uint64_t kMixedAddress = 0x1fc;
  auto memory =
      std::make_shared<amdgpu::test::AsyncVmMemory>(*owner_sim.owner, 0x400, 0x200, 0x204);
  const auto address_space = sim.soc->gpu_vm().register_address_space(77, memory, memory, {}, true);
  ASSERT_TRUE(address_space);
  auto *cu = sim.cu();
  cu->set_gpu_vm(&sim.soc->gpu_vm());
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(77);
  wf->set_address_space(address_space);

  constexpr std::array<uint32_t, 2> kPeerWords{0x11223344u, 0x55667788u};
  memory->store(kMixedAddress, std::as_bytes(std::span(kPeerWords)));
  sim.memory->write32(kMixedAddress, 0xdeadbeefu);
  sim.memory->write32(kMixedAddress + 4, 0xdeadbeefu);

  amdgpu::ScalarMemPipeline pipeline(&cu->l1_scalar());
  auto peer_load = std::make_unique<amdgpu::ScalarMemState>();
  peer_load->addr = kMixedAddress;
  peer_load->dst_register = {amdgpu::ScalarRegisterStorage::SGPR, 8, 2};
  peer_load->num_dwords = 2;
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(peer_load)), *wf),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(memory->backing->posts, 1u);
  (void)owner_sim.engine.step();
  pipeline.tick();
  EXPECT_TRUE(pipeline.empty());
  EXPECT_EQ(cu->read_sgpr_storage(wf->sgpr_alloc().base + 8), kPeerWords[0]);
  EXPECT_EQ(cu->read_sgpr_storage(wf->sgpr_alloc().base + 9), kPeerWords[1]);
  EXPECT_EQ(memory->backing->read_effects, 1u);

  constexpr uint32_t kLocalValue = 0xa1b2c3d4u;
  constexpr uint32_t kChangedBackingValue = 0x12345678u;
  constexpr uint64_t kLocalAddress = 0x180;
  ASSERT_EQ(cu->l1_scalar().store(kLocalAddress, 1, &kLocalValue, wf->process_id()),
            amdgpu::VmAccessOutcome::Complete);
  memory->store(kLocalAddress, std::as_bytes(std::span(&kLocalValue, 1)));
  auto local_load = std::make_unique<amdgpu::ScalarMemState>();
  local_load->addr = kLocalAddress;
  local_load->dst_register = {amdgpu::ScalarRegisterStorage::SGPR, 10, 1};
  local_load->num_dwords = 1;
  EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(local_load)), *wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(cu->read_sgpr_storage(wf->sgpr_alloc().base + 10), kLocalValue);
  memory->store(kLocalAddress, std::as_bytes(std::span(&kChangedBackingValue, 1)));
  auto cached_load = std::make_unique<amdgpu::ScalarMemState>();
  cached_load->addr = kLocalAddress;
  cached_load->dst_register = {amdgpu::ScalarRegisterStorage::SGPR, 11, 1};
  cached_load->num_dwords = 1;
  EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(cached_load)), *wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(cu->read_sgpr_storage(wf->sgpr_alloc().base + 11), kLocalValue);
  EXPECT_EQ(memory->backing->posts, 1u);
}

TEST(GpuVmPipeline, LegacyVectorPeerLaneUsesOneDeferredOwnerEffect) {
  amdgpu::test::AsyncVmHarness owner_sim;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto memory =
      std::make_shared<amdgpu::test::AsyncVmMemory>(*owner_sim.owner, 0x400, 0x104, 0x108);
  const auto address_space = sim.soc->gpu_vm().register_address_space(77, memory, memory, {}, true);
  ASSERT_TRUE(address_space);
  auto *cu = sim.cu();
  cu->set_gpu_vm(&sim.soc->gpu_vm());
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(77);
  wf->set_address_space(address_space);
  wf->set_exec(0x7u);

  constexpr std::array<uint32_t, 3> kStored{0x10203040u, 0x50607080u, 0x90a0b0c0u};
  auto store = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  store->elem_size = sizeof(uint32_t);
  store->num_elems = 1;
  store->is_load = false;
  store->wf_size = wf->wf_size();
  store->exec_mask = 0x7u;
  store->lane_mask = 0x7u;
  store->per_lane_addr[0] = 0x204;
  store->per_lane_addr[1] = 0x200;
  store->per_lane_addr[2] = 0x104;
  store->store_data.resize(store->wf_size * sizeof(uint32_t));
  std::memcpy(store->store_data.data(), kStored.data(), sizeof(kStored));
  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(store)), *wf),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(memory->backing->posts, 1u);
  EXPECT_EQ(memory->backing->write_effects, 0u);
  (void)owner_sim.engine.step();
  pipeline.tick();
  EXPECT_TRUE(pipeline.empty());
  EXPECT_EQ(memory->backing->write_effects, 1u);
  EXPECT_EQ(memory->backing->posts, 1u);
  const std::array<uint32_t, 2> address_order{kStored[1], kStored[0]};
  EXPECT_EQ(memory->load(0x200, sizeof(address_order)),
            std::vector<std::byte>(std::as_bytes(std::span(address_order)).begin(),
                                   std::as_bytes(std::span(address_order)).end()));
  EXPECT_EQ(memory->load(0x104, sizeof(kStored[2])),
            std::vector<std::byte>(std::as_bytes(std::span(&kStored[2], 1)).begin(),
                                   std::as_bytes(std::span(&kStored[2], 1)).end()));

  constexpr uint32_t kLocalValue = 0xaabbccddu;
  constexpr uint32_t kChangedBackingValue = 0x12345678u;
  constexpr uint64_t kLocalAddress = 0x180;
  sim.memory->write32(kLocalAddress, kStored[0]);
  memory->store(kLocalAddress, std::as_bytes(std::span(&kLocalValue, 1)));
  auto local_load = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  local_load->elem_size = sizeof(uint32_t);
  local_load->num_elems = 1;
  local_load->is_load = true;
  local_load->wf_size = wf->wf_size();
  local_load->exec_mask = 1;
  local_load->lane_mask = 1;
  local_load->dst_reg_base = wf->vgpr_alloc().base + 8;
  local_load->per_lane_addr[0] = kLocalAddress;
  EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(local_load)), *wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + 8, 0), kLocalValue);
  memory->store(kLocalAddress, std::as_bytes(std::span(&kChangedBackingValue, 1)));
  auto cached_load = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  cached_load->elem_size = sizeof(uint32_t);
  cached_load->num_elems = 1;
  cached_load->is_load = true;
  cached_load->wf_size = wf->wf_size();
  cached_load->exec_mask = 1;
  cached_load->lane_mask = 1;
  cached_load->dst_reg_base = wf->vgpr_alloc().base + 9;
  cached_load->per_lane_addr[0] = kLocalAddress;
  EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(cached_load)), *wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + 9, 0), kLocalValue);
  EXPECT_EQ(memory->backing->posts, 1u);

  auto mixed_load = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  mixed_load->elem_size = sizeof(uint32_t);
  mixed_load->num_elems = 1;
  mixed_load->is_load = true;
  mixed_load->wf_size = wf->wf_size();
  mixed_load->exec_mask = 0x7u;
  mixed_load->lane_mask = 0x7u;
  mixed_load->dst_reg_base = wf->vgpr_alloc().base + 10;
  mixed_load->per_lane_addr[0] = 0x204;
  mixed_load->per_lane_addr[1] = 0x200;
  mixed_load->per_lane_addr[2] = 0x104;
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(mixed_load)), *wf),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(wf->state(), amdgpu::WfState::VM_RETRY);
  (void)owner_sim.engine.step();
  pipeline.tick();
  EXPECT_TRUE(pipeline.empty());
  EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + 10, 0), kStored[0]);
  EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + 10, 1), kStored[1]);
  EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + 10, 2), kStored[2]);
  EXPECT_EQ(memory->backing->posts, 2u);
}

TEST(GpuVmPipeline, LegacyVectorAndAtomicIngressRetryBeforeAnyLocalEffect) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize(/*legacy_cache_compatible=*/true));
  ASSERT_NE(context.wf, nullptr);
  constexpr uint64_t kFirst = 0x300;
  constexpr uint64_t kSecond = 0x380;
  constexpr uint32_t kFirstInitial = 7, kSecondInitial = 9;
  constexpr uint32_t kFirstStore = 31, kSecondStore = 37;
  context.external->store(kFirst, kFirstInitial);
  context.external->store(kSecond, kSecondInitial);
  bool open = false;
  std::vector<amdgpu::MemoryIngressSpan> observed;
  amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
  pipeline.set_ingress_handler([&](uint32_t process_id, const amdgpu::GpuVmAccess &,
                                   std::span<const amdgpu::MemoryIngressSpan> spans) {
    EXPECT_EQ(process_id, 77u);
    observed.assign(spans.begin(), spans.end());
    return amdgpu::MemoryIngressAdmission{.outcome = open ? amdgpu::VmAccessOutcome::Complete
                                                          : amdgpu::VmAccessOutcome::Unavailable,
                                          .permit = {}};
  });

  auto store = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  store->elem_size = sizeof(uint32_t);
  store->num_elems = 1;
  store->is_load = false;
  store->mtype = amdgpu::Mtype::UC;
  store->wf_size = context.wf->wf_size();
  store->exec_mask = 3;
  store->lane_mask = 3;
  store->per_lane_addr[0] = kFirst;
  store->per_lane_addr[1] = kSecond;
  store->store_data.resize(store->wf_size * sizeof(uint32_t));
  std::memcpy(store->store_data.data(), &kFirstStore, sizeof(kFirstStore));
  std::memcpy(store->store_data.data() + sizeof(uint32_t), &kSecondStore, sizeof(kSecondStore));
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(store)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
  ASSERT_EQ(observed.size(), 2u);
  EXPECT_EQ(observed[0].address, kFirst);
  EXPECT_EQ(observed[1].address, kSecond);
  EXPECT_EQ(observed[0].access, amdgpu::VmAccessKind::Write);
  EXPECT_EQ(context.external->load<uint32_t>(kFirst), kFirstInitial);
  EXPECT_EQ(context.external->load<uint32_t>(kSecond), kSecondInitial);
  open = true;
  pipeline.tick();
  EXPECT_TRUE(pipeline.empty());
  EXPECT_EQ(context.external->load<uint32_t>(kFirst), kFirstStore);
  EXPECT_EQ(context.external->load<uint32_t>(kSecond), kSecondStore);

  auto atomic = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  atomic->elem_size = sizeof(uint32_t);
  atomic->num_elems = 1;
  atomic->is_load = true;
  atomic->atomic_op = amdgpu::AtomicOp::ADD;
  atomic->wf_size = context.wf->wf_size();
  atomic->exec_mask = 1;
  atomic->lane_mask = 1;
  atomic->per_lane_addr[0] = kFirst;
  atomic->dst_reg_base = context.wf->vgpr_alloc().base + 8;
  atomic->store_data.resize(atomic->wf_size * sizeof(uint32_t));
  constexpr uint32_t kIncrement = 5;
  std::memcpy(atomic->store_data.data(), &kIncrement, sizeof(kIncrement));
  open = false;
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(atomic)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
  ASSERT_EQ(observed.size(), 1u);
  EXPECT_EQ(observed[0].address, kFirst);
  EXPECT_EQ(observed[0].access, amdgpu::VmAccessKind::Atomic);
  EXPECT_EQ(context.external->load<uint32_t>(kFirst), kFirstStore);
  open = true;
  pipeline.tick();
  EXPECT_TRUE(pipeline.empty());
  EXPECT_EQ(context.external->load<uint32_t>(kFirst), kFirstStore + kIncrement);

  // Legacy callers can identify their VM only by process ID. The installed
  // handler must still see the span before that wave reaches a local cache.
  context.wf->set_address_space({});
  auto vmid_only = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  vmid_only->elem_size = sizeof(uint32_t);
  vmid_only->num_elems = 1;
  vmid_only->is_load = false;
  vmid_only->mtype = amdgpu::Mtype::UC;
  vmid_only->wf_size = context.wf->wf_size();
  vmid_only->exec_mask = 1;
  vmid_only->lane_mask = 1;
  vmid_only->per_lane_addr[0] = kSecond;
  vmid_only->store_data.resize(vmid_only->wf_size * sizeof(uint32_t));
  constexpr uint32_t kVmidOnlyStore = 41;
  std::memcpy(vmid_only->store_data.data(), &kVmidOnlyStore, sizeof(kVmidOnlyStore));
  open = false;
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(vmid_only)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
  ASSERT_EQ(observed.size(), 1u);
  EXPECT_EQ(observed[0].address, kSecond);
  EXPECT_EQ(context.external->load<uint32_t>(kSecond), kSecondStore);
  open = true;
  pipeline.tick();
  EXPECT_EQ(context.external->load<uint32_t>(kSecond), kVmidOnlyStore);
}

TEST(GpuVmPipeline, LegacyPeerAtomicFailsBeforeTheCacheOrOwnerEffect) {
  amdgpu::test::AsyncVmHarness owner_sim;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  auto memory =
      std::make_shared<amdgpu::test::AsyncVmMemory>(*owner_sim.owner, 0x400, 0x104, 0x108);
  const auto address_space = sim.soc->gpu_vm().register_address_space(77, memory, memory, {}, true);
  ASSERT_TRUE(address_space);
  auto *cu = sim.cu();
  cu->set_gpu_vm(&sim.soc->gpu_vm());
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(77);
  wf->set_address_space(address_space);
  wf->set_exec(1);
  constexpr uint32_t kInitial = 7;
  memory->store(0x104, std::as_bytes(std::span(&kInitial, 1)));
  sim.memory->write32(0x104, kInitial);

  auto atomic = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  atomic->elem_size = sizeof(uint32_t);
  atomic->num_elems = 1;
  atomic->is_load = true;
  atomic->atomic_op = amdgpu::AtomicOp::ADD;
  atomic->wf_size = wf->wf_size();
  atomic->exec_mask = 1;
  atomic->lane_mask = 1;
  atomic->per_lane_addr[0] = 0x104;
  atomic->store_data.resize(atomic->wf_size * sizeof(uint32_t));
  constexpr uint32_t kSource = 3;
  std::memcpy(atomic->store_data.data(), &kSource, sizeof(kSource));
  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  EXPECT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(atomic)), *wf),
            amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(memory->backing->posts, 0u);
  EXPECT_EQ(memory->backing->write_effects, 0u);
  EXPECT_EQ(sim.memory->read32(0x104), kInitial);
}

TEST(GpuVmPipeline, DeferredOperationCannotWriteBackIntoReusedWaveSlot) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
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
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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

TEST(GpuVmPipeline, TransactionPolicyPreservesBatchingAndSplitRetryProgress) {
  using Policy = amdgpu::MemoryTransactionPolicy;
  for (auto policy : {Policy{}, Policy::unqualified_64()}) {
    for (auto [elements, lanes, offset] :
         {std::array<uint32_t, 3>{4, 8, 16}, {4, 16, 16}, {3, 16, 0}, {4, 8, 4}}) {
      const uint32_t width = elements * sizeof(uint32_t);
      const uint32_t bytes = lanes * width;
      for (bool reverse : {false, true}) {
        for (bool is_load : {false, true}) {
          SCOPED_TRACE(std::format("granule={} bytes={} reverse={} load={}", policy.bytes(), bytes,
                                   reverse, is_load));
          TranslatedPipelineContext context;
          ASSERT_NO_FATAL_FAILURE(context.initialize());
          EXPECT_EQ(context.cu->l1_vector().transaction_policy().bytes(), 128u);
          EXPECT_EQ(context.sim.soc->memory()->transaction_policy().bytes(), 128u);
          // Offset the record to cover partial first/last transactions too.
          const uint64_t address = 0x200 + offset;
          constexpr uint32_t kDestination = 8;
          std::vector<uint32_t> values(bytes / sizeof(uint32_t));
          for (uint32_t i = 0; i < values.size(); ++i) {
            values[i] = 0x12340000 + i;
            context.external->store(address + i * sizeof(uint32_t), is_load ? values[i] : 0u);
          }
          auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
          state->elem_size = sizeof(uint32_t);
          state->num_elems = elements;
          state->is_load = is_load;
          state->wf_size = context.wf->wf_size();
          state->exec_mask = (uint64_t{1} << lanes) - 1;
          state->lane_mask = state->exec_mask;
          state->dst_reg_base = context.wf->vgpr_alloc().base + kDestination;
          state->store_data.resize(state->wf_size * width);
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            const uint32_t index = reverse ? lanes - 1 - lane : lane;
            state->per_lane_addr[lane] = address + index * width;
            if (!is_load)
              std::memcpy(state->store_data.data() + lane * width, values.data() + index * elements,
                          width);
          }
          if (is_load)
            context.external->unavailable_read_call = 2;
          else
            context.external->unavailable_write_call = 2;
          amdgpu::L1VectorCache l1(context.cu->l2(), policy);
          amdgpu::GlobalMemPipeline pipeline(&l1, context.cu->l2());
          ASSERT_EQ(
              pipeline.issue_deferred(new TestMemoryInstruction(std::move(state)), *context.wf),
              amdgpu::VmAccessOutcome::Complete);
          const bool split_requests = reverse || policy.bytes() < Policy{}.bytes();
          EXPECT_EQ(context.wf->state(),
                    split_requests ? amdgpu::WfState::VM_RETRY : amdgpu::WfState::RUNNING);
          // Reordered runs visit the transaction containing the first active
          // lane first. Change a completed word to detect accidental replay.
          const uint64_t first_address =
              reverse
                  ? std::max(address, (address + bytes - width) / policy.bytes() * policy.bytes())
                  : address;
          if (split_requests) {
            context.external->store(first_address, uint32_t{0xabcdef01});
            pipeline.tick();
          }
          ASSERT_EQ(context.wf->state(), amdgpu::WfState::RUNNING);
          if (is_load) {
            for (uint32_t lane = 0; lane < lanes; ++lane) {
              const uint32_t index = reverse ? lanes - 1 - lane : lane;
              for (uint32_t word = 0; word < elements; ++word)
                EXPECT_EQ(context.cu->read_vgpr(context.wf->vgpr_alloc().base + kDestination + word,
                                                lane),
                          values[index * elements + word]);
            }
          } else {
            for (uint32_t i = 0; i < values.size(); ++i)
              EXPECT_EQ(context.external->load<uint32_t>(address + i * sizeof(uint32_t)),
                        split_requests && address + i * sizeof(uint32_t) == first_address
                            ? 0xabcdef01
                            : values[i]);
          }
          const auto &history =
              is_load ? context.external->read_history : context.external->write_history;
          if (!split_requests) {
            ASSERT_EQ(history.size(), 1u);
            EXPECT_EQ(history.front().address, address);
            EXPECT_EQ(history.front().size, bytes);
            continue;
          }
          const uint32_t first = std::min<uint64_t>(
              address + bytes - first_address, policy.bytes() - first_address % policy.bytes());
          ASSERT_EQ(history.size(),
                    (address % policy.bytes() + bytes + policy.bytes() - 1) / policy.bytes() + 1);
          EXPECT_EQ(history.front().address, first_address);
          EXPECT_EQ(history.front().size, first);
          EXPECT_EQ(history[2].address, history[1].address);
          for (const auto &request : history) {
            EXPECT_LE(request.size, policy.bytes());
            EXPECT_EQ(request.address / policy.bytes(),
                      (request.address + request.size - 1) / policy.bytes());
          }
        }
      }
    }
  }
}

TEST(GpuVmPipeline, PermutedTranslatedVectorTransfersPreserveSnapshotsAndRetryProgress) {
  for (uint32_t num_lanes : {4u, 8u}) {
    for (bool is_load : {true, false}) {
      TranslatedPipelineContext context;
      ASSERT_NO_FATAL_FAILURE(context.initialize());
      constexpr uint64_t kAddress = 0x200;
      constexpr uint32_t kDestination = 8;
      constexpr uint32_t kElements = 4;
      std::vector<uint32_t> values(num_lanes * kElements);
      for (uint32_t i = 0; i < values.size(); ++i) {
        values[i] = 0x12340000 + i;
        context.external->store(kAddress + i * sizeof(uint32_t), is_load ? values[i] : 0u);
      }
      auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
      state->elem_size = sizeof(uint32_t);
      state->num_elems = kElements;
      state->is_load = is_load;
      state->wf_size = context.wf->wf_size();
      state->exec_mask = (1u << num_lanes) - 1;
      state->lane_mask = state->exec_mask;
      state->dst_reg_base = context.wf->vgpr_alloc().base + kDestination;
      state->store_data.resize(state->wf_size * kElements * sizeof(uint32_t));
      for (uint32_t lane = 0; lane < num_lanes; ++lane) {
        state->per_lane_addr[lane] =
            kAddress + (num_lanes - 1 - lane) * kElements * sizeof(uint32_t);
        if (!is_load)
          std::memcpy(state->store_data.data() + lane * kElements * sizeof(uint32_t),
                      values.data() + (num_lanes - 1 - lane) * kElements,
                      kElements * sizeof(uint32_t));
      }
      const uint32_t half_bytes = values.size() * sizeof(uint32_t) / 2;
      // Retry halfway through a reordered transaction. The first half must
      // not be reloaded or republished when the second translation span resumes.
      context.external->translation_granule = half_bytes;
      if (is_load)
        context.external->unavailable_read_call = 2;
      else
        context.external->unavailable_write_call = 2;
      amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
      ASSERT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(state)), *context.wf),
                amdgpu::VmAccessOutcome::Complete);
      ASSERT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
      // Change the completed span externally to detect replay on either path.
      context.external->store(kAddress, uint32_t{0xabcdef01});
      pipeline.tick();
      ASSERT_EQ(context.wf->state(), amdgpu::WfState::RUNNING);
      if (is_load) {
        for (uint32_t lane = 0; lane < num_lanes; ++lane)
          for (uint32_t elem = 0; elem < kElements; ++elem)
            EXPECT_EQ(
                context.cu->read_vgpr(context.wf->vgpr_alloc().base + kDestination + elem, lane),
                values[(num_lanes - 1 - lane) * kElements + elem]);
      } else {
        for (uint32_t i = 0; i < values.size(); ++i)
          EXPECT_EQ(context.external->load<uint32_t>(kAddress + i * sizeof(uint32_t)),
                    i == 0 ? 0xabcdef01 : values[i]);
      }
      const auto &history =
          is_load ? context.external->read_history : context.external->write_history;
      ASSERT_EQ(history.size(), 3u);
      EXPECT_EQ(history[0].address, kAddress);
      EXPECT_EQ(history[1].address, kAddress + half_bytes);
      EXPECT_EQ(history[2].address, kAddress + half_bytes);
      for (const auto &request : history)
        EXPECT_EQ(request.size, half_bytes);
    }
  }
}

TEST(GpuVmPipeline, PermutedTranslatedLanesUseOnePhysicalRequest) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
  constexpr uint64_t kAddress = 0x200;
  constexpr uint32_t kDestination = 8;
  for (uint32_t lane = 0; lane < 32; ++lane)
    context.external->store(kAddress + lane * sizeof(uint32_t), 0x1000u + lane);
  auto load = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  load->elem_size = sizeof(uint32_t);
  load->num_elems = 1;
  load->is_load = true;
  load->wf_size = context.wf->wf_size();
  load->exec_mask = 0xffffffff;
  load->lane_mask = 0xffffffff;
  load->dst_reg_base = context.wf->vgpr_alloc().base + kDestination;
  for (uint32_t lane = 0; lane < 32; ++lane)
    load->per_lane_addr[lane] = kAddress + (lane ^ 1) * sizeof(uint32_t);
  amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
  ASSERT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(load)), *context.wf),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(context.external->read_calls, 1u);
  EXPECT_EQ(context.external->last_read.address, kAddress);
  EXPECT_EQ(context.external->last_read.size, 32 * sizeof(uint32_t));
  for (uint32_t lane = 0; lane < 32; ++lane)
    EXPECT_EQ(context.cu->read_vgpr(context.wf->vgpr_alloc().base + kDestination, lane),
              0x1000u + (lane ^ 1));
}

TEST(GpuVmPipeline, ReorderedTranslatedTransactionsRespectBoundariesAndOverlaps) {
  constexpr uint32_t kDestination = 8;
  for (const auto &addresses : {std::array<uint64_t, 4>{0x284, 0x280, 0x27c, 0x278},
                                std::array<uint64_t, 4>{0x240, 0x240, 0x244, 0x248}}) {
    TranslatedPipelineContext context;
    ASSERT_NO_FATAL_FAILURE(context.initialize());
    std::array<uint32_t, 4> values{1, 2, 3, 4};
    amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
    for (bool is_load : {false, true}) {
      auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
      state->elem_size = sizeof(uint32_t);
      state->num_elems = 1;
      state->is_load = is_load;
      state->wf_size = context.wf->wf_size();
      state->exec_mask = 0xf;
      state->lane_mask = 0xf;
      state->dst_reg_base = context.wf->vgpr_alloc().base + kDestination;
      state->store_data.resize(state->wf_size * sizeof(uint32_t));
      std::memcpy(state->store_data.data(), values.data(), sizeof(values));
      std::ranges::copy(addresses, state->per_lane_addr.begin());
      ASSERT_EQ(pipeline.issue(new TestMemoryInstruction(std::move(state)), *context.wf),
                amdgpu::VmAccessOutcome::Complete);
    }
    for (uint32_t lane = 0; lane < addresses.size(); ++lane) {
      const uint32_t expected = addresses[0] == addresses[1] && lane == 0 ? 2 : values[lane];
      EXPECT_EQ(context.external->load<uint32_t>(addresses[lane]), expected);
      EXPECT_EQ(context.cu->read_vgpr(context.wf->vgpr_alloc().base + kDestination, lane),
                expected);
    }
    EXPECT_EQ(context.external->write_calls, 2u);
    EXPECT_EQ(context.external->read_calls, 2u);
    if (addresses[0] != addresses[1]) {
      EXPECT_EQ(context.external->write_history[0].address, 0x280u);
      EXPECT_EQ(context.external->write_history[1].address, 0x278u);
      EXPECT_EQ(context.external->write_history[0].size, 2 * sizeof(uint32_t));
      EXPECT_EQ(context.external->write_history[1].size, 2 * sizeof(uint32_t));
    }
  }
}

TEST(GpuVmPipeline, ClippedReorderedRetryPreservesGapsAndCompletedBytes) {
  constexpr uint64_t kAddress = 0x200;
  constexpr uint32_t kDestination = 8;
  constexpr uint32_t kSentinel = 0xcccccccc;
  const std::array<uint32_t, 6> kSource{10, 11, 20, 21, 30, 31};
  const std::array<uint32_t, 8> kExpected{20, 21, 10, 11, kSentinel, kSentinel, 30, kSentinel};
  for (bool load : {false, true}) {
    for (uint32_t retry_call : {2u, 4u}) {
      SCOPED_TRACE(load);
      SCOPED_TRACE(retry_call);
      TranslatedPipelineContext context;
      ASSERT_NO_FATAL_FAILURE(context.initialize());
      for (unsigned i = 0; i < kExpected.size(); ++i)
        context.external->store(kAddress + i * sizeof(uint32_t), load ? kExpected[i] : kSentinel);
      auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
      state->elem_size = sizeof(uint32_t);
      state->num_elems = 2;
      state->is_load = load;
      state->wf_size = context.wf->wf_size();
      state->exec_mask = 7;
      state->lane_mask = 7;
      state->element_lane_masks.assign(2, 7);
      state->element_lane_masks[1] = 3;
      state->dst_reg_base = context.wf->vgpr_alloc().base + kDestination;
      state->per_lane_addr[0] = kAddress + 8;
      state->per_lane_addr[1] = kAddress;
      state->per_lane_addr[2] = kAddress + 24;
      state->store_data.resize(state->wf_size * 2 * sizeof(uint32_t));
      std::memcpy(state->store_data.data(), kSource.data(), sizeof(kSource));
      context.external->translation_granule = 6;
      if (load)
        context.external->unavailable_read_call = retry_call;
      else
        context.external->unavailable_write_call = retry_call;
      amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
      ASSERT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(state)), *context.wf),
                amdgpu::VmAccessOutcome::Complete);
      ASSERT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
      context.external->store(kAddress, uint32_t{0xabcdef01});
      pipeline.tick();
      ASSERT_EQ(context.wf->state(), amdgpu::WfState::RUNNING);
      ASSERT_TRUE(context.wf->wait_counters().empty());
      for (unsigned i = 0; i < kExpected.size(); ++i)
        EXPECT_EQ(context.external->load<uint32_t>(kAddress + i * sizeof(uint32_t)),
                  i == 0 ? 0xabcdef01 : kExpected[i]);
      if (load) {
        for (unsigned lane = 0; lane < 3; ++lane)
          for (unsigned element = 0; element < 2; ++element)
            EXPECT_EQ(
                context.cu->read_vgpr(context.wf->vgpr_alloc().base + kDestination + element, lane),
                lane == 2 && element == 1 ? 0 : kSource[lane * 2 + element]);
      }
      const auto &history = load ? context.external->read_history : context.external->write_history;
      unsigned prefix_calls = 0;
      for (const auto &request : history) {
        prefix_calls += request.address == kAddress;
        EXPECT_TRUE(
            (request.address >= kAddress && request.address + request.size <= kAddress + 16) ||
            (request.address >= kAddress + 24 && request.address + request.size <= kAddress + 28));
      }
      EXPECT_EQ(prefix_calls, 1u);
    }
  }
}

TEST(GpuVmPipeline, TranslatedVectorStoreRetrySkipsCompletedRequests) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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
      TranslatedPipelineContext context;
      ASSERT_NO_FATAL_FAILURE(context.initialize(/*legacy_cache_compatible=*/true));
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
      const uint64_t reads = context.cu->l2()->backing_read_transactions();
      const uint64_t writes = context.cu->l2()->backing_write_transactions();
      amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
      EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(state)), *context.wf),
                amdgpu::VmAccessOutcome::Complete);
      if (mask && retry) {
        ASSERT_EQ(context.wf->state(), amdgpu::WfState::VM_RETRY);
        EXPECT_EQ(context.cu->l2()->backing_read_transactions() - reads, 2u);
        EXPECT_EQ(context.cu->l2()->backing_write_transactions() - writes, 2u);
        EXPECT_EQ(context.external->load<uint32_t>(kAddress), kInitial + kSource[0]);
        // Retire the first attempt's snapshot after lane 0 has committed. The
        // retry must acquire a new generation without repeating that lane.
        ASSERT_TRUE(context.sim.soc->gpu_vm().invalidate(context.address_space));
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
      const uint32_t attempts = mask ? (retry ? 4u : 3u) : 0u;
      EXPECT_EQ(context.external->atomic_modify_calls, attempts);
      EXPECT_EQ(context.cu->l2()->backing_read_transactions() - reads, attempts);
      EXPECT_EQ(context.cu->l2()->backing_write_transactions() - writes, attempts);
    }
  }
}

TEST(GpuVmPipeline, TranslatedAtomicRetrySkipsCompletedLanes) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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

TEST(GpuVmPipeline, MixedFlatAtomicRetryPreservesLdsAndCompletedGlobalLanes) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
  ASSERT_TRUE(context.address_space);
  ASSERT_NE(context.wf, nullptr);
  auto &wf = *context.wf;
  wf.set_exec(0x7);
  wf.set_lds_base(context.cu->allocate_lds(256));
  constexpr uint32_t lds_offset = 64;
  constexpr std::array<uint32_t, 3> initial = {10, 20, 30};
  constexpr std::array<uint32_t, 3> source = {1, 2, 3};
  context.external->store(0x100, initial[0]);
  context.external->store(0x200, initial[2]);
  wf.lds().write32(wf.lds_base() + lds_offset, initial[1]);
  context.external->unavailable_atomic_load_call = 2;

  auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  state->elem_size = sizeof(uint32_t);
  state->num_elems = 1;
  state->is_load = true;
  state->atomic_op = amdgpu::AtomicOp::ADD;
  state->wf_size = wf.wf_size();
  state->exec_mask = state->lane_mask = 0x7;
  state->flat_local_lane_mask = 0x2;
  state->dst_reg_base = wf.vgpr_alloc().base + 4;
  state->per_lane_addr[0] = 0x100;
  state->per_lane_addr[1] = wf.lds_base() + lds_offset;
  state->per_lane_addr[2] = 0x200;
  state->store_data.resize(wf.wf_size() * sizeof(uint32_t));
  std::memcpy(state->store_data.data(), source.data(), sizeof(source));

  amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
  ASSERT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(state)), wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(wf.state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(context.external->load<uint32_t>(0x100), initial[0] + source[0]);
  EXPECT_EQ(wf.lds().read32(wf.lds_base() + lds_offset), initial[1]);
  EXPECT_EQ(context.external->load<uint32_t>(0x200), initial[2]);

  pipeline.tick();
  EXPECT_EQ(wf.state(), amdgpu::WfState::RUNNING);
  EXPECT_TRUE(wf.wait_counters().empty());
  EXPECT_EQ(context.external->load<uint32_t>(0x100), initial[0] + source[0]);
  EXPECT_EQ(wf.lds().read32(wf.lds_base() + lds_offset), initial[1] + source[1]);
  EXPECT_EQ(context.external->load<uint32_t>(0x200), initial[2] + source[2]);
  for (uint32_t lane = 0; lane < initial.size(); ++lane)
    EXPECT_EQ(context.cu->read_vgpr(wf.vgpr_alloc().base + 4, lane), initial[lane]);
  EXPECT_EQ(context.external->atomic_load_calls, 3u);
  EXPECT_EQ(context.external->compare_exchange_calls, 2u);
}

TEST(GpuVmPipeline, MixedFlatLegacyAtomicRetryRefreshesVmSnapshot) {
  for (uint32_t local_lane : {0u, 1u}) {
    SCOPED_TRACE(local_lane);
    const uint32_t first_global_lane = 1 - local_lane;
    TranslatedPipelineContext context;
    ASSERT_NO_FATAL_FAILURE(context.initialize(/*legacy_cache_compatible=*/true));
    ASSERT_TRUE(context.address_space);
    ASSERT_NE(context.wf, nullptr);
    auto &wf = *context.wf;
    wf.set_exec(0x7);
    wf.set_lds_base(context.cu->allocate_lds(256));
    constexpr uint32_t lds_offset = 64;
    constexpr std::array<uint32_t, 3> initial = {10, 20, 30};
    constexpr std::array<uint32_t, 3> source = {1, 2, 3};
    context.external->store(0x100, initial[first_global_lane]);
    context.external->store(0x200, initial[2]);
    wf.lds().write32(wf.lds_base() + lds_offset, initial[local_lane]);
    context.external->unavailable_atomic_modify_call = 2;

    auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
    state->elem_size = sizeof(uint32_t);
    state->num_elems = 1;
    state->is_load = true;
    state->atomic_op = amdgpu::AtomicOp::ADD;
    state->wf_size = wf.wf_size();
    state->exec_mask = state->lane_mask = 0x7;
    state->flat_local_lane_mask = uint64_t{1} << local_lane;
    state->dst_reg_base = wf.vgpr_alloc().base + 4;
    state->per_lane_addr[first_global_lane] = 0x100;
    state->per_lane_addr[local_lane] = wf.lds_base() + lds_offset;
    state->per_lane_addr[2] = 0x200;
    state->store_data.resize(wf.wf_size() * sizeof(uint32_t));
    std::memcpy(state->store_data.data(), source.data(), sizeof(source));

    amdgpu::GlobalMemPipeline pipeline(&context.cu->l1_vector(), context.cu->l2());
    ASSERT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(state)), wf),
              amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(wf.state(), amdgpu::WfState::VM_RETRY);
    EXPECT_EQ(context.external->load<uint32_t>(0x100),
              initial[first_global_lane] + source[first_global_lane]);
    EXPECT_EQ(wf.lds().read32(wf.lds_base() + lds_offset), initial[local_lane]);
    EXPECT_EQ(context.external->load<uint32_t>(0x200), initial[2]);

    // The first global lane has committed. Refresh the VM snapshot on retry,
    // preserving that lane's result and deferring the LDS atomic until success.
    ASSERT_TRUE(context.sim.soc->gpu_vm().invalidate(context.address_space));
    pipeline.tick();
    EXPECT_EQ(wf.state(), amdgpu::WfState::RUNNING);
    EXPECT_TRUE(wf.wait_counters().empty());
    EXPECT_EQ(context.external->load<uint32_t>(0x100),
              initial[first_global_lane] + source[first_global_lane]);
    EXPECT_EQ(wf.lds().read32(wf.lds_base() + lds_offset),
              initial[local_lane] + source[local_lane]);
    EXPECT_EQ(context.external->load<uint32_t>(0x200), initial[2] + source[2]);
    for (uint32_t lane = 0; lane < initial.size(); ++lane)
      EXPECT_EQ(context.cu->read_vgpr(wf.vgpr_alloc().base + 4, lane), initial[lane]);
    EXPECT_EQ(context.external->atomic_modify_calls, 3u);
  }
}

TEST(GpuVmPipeline, TranslatedTensorLoadRetriesBeforeCommittingLdsOrBarrier) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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

TEST(GpuVmPipeline, DeferredTensorLoadCommitsLdsAfterPeerReadEffects) {
  amdgpu::test::AsyncVmHarness owner_sim;
  Gfx1250Sim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize());
  constexpr uint64_t kGlobalAddress = 0x100;
  constexpr uint32_t kBarrierAddress = 64;
  constexpr std::array<uint32_t, 2> kValues{0x11223344u, 0x55667788u};
  constexpr uint32_t kLdsSentinel = 0xa5a55a5au;
  auto memory = std::make_shared<amdgpu::test::AsyncVmMemory>(
      *owner_sim.owner, 0x400, kGlobalAddress, kGlobalAddress + sizeof(kValues));
  memory->store(kGlobalAddress, std::as_bytes(std::span(kValues)));
  const auto address_space = sim.soc->gpu_vm().register_translated(77, memory, memory);
  ASSERT_TRUE(address_space);
  auto *cu = sim.cu();
  cu->set_gpu_vm(&sim.soc->gpu_vm());
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(77);
  wf->set_address_space(address_space);
  wf->set_lds_base(cu->allocate_lds(256));
  for (uint32_t index = 0; index < kValues.size(); ++index)
    wf->lds().write32(wf->lds_base() + index * sizeof(uint32_t), kLdsSentinel);
  wf->lds().write64(wf->lds_base() + kBarrierAddress, 0);

  amdgpu::tensor_dma_detail::TensorDmaDescriptor descriptor;
  descriptor.elem_size = sizeof(uint32_t);
  descriptor.atomic_barrier = true;
  descriptor.atomic_barrier_addr = kBarrierAddress;
  auto state = std::make_unique<amdgpu::tensor_dma_detail::TensorDmaState>(descriptor, false);
  state->access = wf->snapshot_vm_access();
  ASSERT_TRUE(state->access);
  for (uint32_t index = 0; index < kValues.size(); ++index) {
    state->elements.push_back(
        {.global_address = kGlobalAddress + index * sizeof(uint32_t),
         .lds_address = wf->lds_base() + index * static_cast<uint32_t>(sizeof(uint32_t)),
         .in_bounds = true});
  }

  amdgpu::TensorDmaPipeline pipeline;
  EXPECT_EQ(pipeline.issue_deferred(new TestMemoryInstruction(std::move(state)), *wf),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(wf->state(), amdgpu::WfState::VM_RETRY);
  EXPECT_EQ(memory->backing->posts, 1u);
  EXPECT_EQ(memory->backing->read_effects, 0u);
  pipeline.tick();
  EXPECT_EQ(memory->backing->posts, 1u);
  for (uint32_t index = 0; index < kValues.size(); ++index) {
    EXPECT_TRUE(owner_sim.engine.step());
    pipeline.tick();
    EXPECT_EQ(wf->lds().read32(wf->lds_base() + index * sizeof(uint32_t)),
              index + 1 == kValues.size() ? kValues[index] : kLdsSentinel);
  }
  EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_TRUE(wf->wait_counters().empty());
  EXPECT_EQ(wf->lds().read32(wf->lds_base()), kValues[0]);
  EXPECT_EQ(wf->lds().read32(wf->lds_base() + sizeof(uint32_t)), kValues[1]);
  EXPECT_EQ(memory->backing->posts, kValues.size());
  EXPECT_EQ(memory->backing->read_effects, kValues.size());
  EXPECT_NE(wf->lds().read64(wf->lds_base() + kBarrierAddress), 0u);
}

TEST(GpuVmPipeline, TranslatedTensorStoreRetriesWithoutReplayingOrResnapshottingLds) {
  TranslatedPipelineContext context;
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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
  ASSERT_NO_FATAL_FAILURE(context.initialize());
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
