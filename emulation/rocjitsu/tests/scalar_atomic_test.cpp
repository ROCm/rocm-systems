// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <tuple>
#include <vector>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

// Match translated vector atomics: this backing intentionally has no atomic_modify override.
class ScalarAtomicBacking : public PhysicalMemoryAccess {
public:
  explicit ScalarAtomicBacking(GpuMemory &memory) : backing_(memory) {}

  VmAccessOutcome read(VmMemoryDomain domain, uint64_t address,
                       std::span<std::byte> bytes) override {
    return backing_.read(domain, address, bytes);
  }
  VmAccessOutcome write(VmMemoryDomain domain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    return backing_.write(domain, address, bytes);
  }
  AtomicLoadResult atomic_load(VmMemoryDomain domain, uint64_t address, uint32_t width) override {
    ++load_attempts;
    if (outcome != VmAccessOutcome::Complete)
      return {.outcome = outcome};
    return backing_.atomic_load(domain, address, width);
  }
  AtomicCompareExchangeResult compare_exchange(VmMemoryDomain domain, uint64_t address,
                                               uint32_t width, uint64_t expected,
                                               uint64_t desired) override {
    ++update_attempts;
    if (outcome != VmAccessOutcome::Complete)
      return {.outcome = outcome};
    if (exchange_outcome != VmAccessOutcome::Complete)
      return {.outcome = exchange_outcome};
    const auto result = backing_.compare_exchange(domain, address, width, expected, desired);
    if (result.outcome == VmAccessOutcome::Complete && result.exchanged)
      ++mutations;
    return result;
  }

  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  VmAccessOutcome exchange_outcome = VmAccessOutcome::Complete;
  unsigned load_attempts = 0;
  unsigned update_attempts = 0;
  unsigned mutations = 0;

protected:
  GpuMemoryPhysicalAccess backing_;
};

class CachedScalarAtomicBacking final : public ScalarAtomicBacking {
public:
  using ScalarAtomicBacking::ScalarAtomicBacking;

  VmAccessOutcome atomic_modify(VmMemoryDomain domain, uint64_t address, uint32_t width,
                                const AtomicMutation &mutation) override {
    ++update_attempts;
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    return backing_.atomic_modify(domain, address, width, [&](std::span<std::byte> bytes) {
      ++mutations;
      mutation(bytes);
    });
  }
};

enum class MemoryMode { Direct, CachedVm, TranslatedVm };

class ScalarAtomicTest : public ::testing::TestWithParam<std::tuple<rj_code_arch_t, MemoryMode>> {
protected:
  static constexpr uint64_t kAddress = 0x1000;
  static constexpr uint64_t kProgram = 0x100000;

  void SetUp() override {
    ComputeUnitCore::Config config{};
    config.arch = std::get<0>(GetParam());
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 16;
    config.lds_size_kb = 64;
    config.functional_quantum = 1;
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
    l2.set_backing_memory(&memory);
    cu = ComputeUnitCore::create("scalar_atomic_cu", config, &memory, &l2);
    decoder = Decoder::create(config.arch);
    ASSERT_NE(cu, nullptr);
    ASSERT_NE(decoder, nullptr);
    wf = cu->dispatch_wf(0, kProgram, 104, 16);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(0);
    pipeline = std::make_unique<ScalarMemPipeline>(&cu->l1_scalar());
    if (memory_mode() != MemoryMode::Direct) {
      if (memory_mode() == MemoryMode::CachedVm)
        backing = std::make_shared<CachedScalarAtomicBacking>(memory);
      else
        backing = std::make_shared<ScalarAtomicBacking>(memory);
      const auto space =
          vm.register_address_space(7, std::make_shared<IdentityAddressSpaceTranslator>(), backing,
                                    {}, memory_mode() == MemoryMode::CachedVm);
      ASSERT_TRUE(space);
      l2.set_gpu_vm(&vm);
      cu->set_gpu_vm(&vm);
      wf->set_process_id(7);
      wf->set_address_space(space);
    }
    write_sgpr(0, kAddress);
    write_sgpr(1, 0);
  }

  void TearDown() override {
    if (wf)
      wf->halt();
  }

  MemoryMode memory_mode() const { return std::get<1>(GetParam()); }

  void write_sgpr(unsigned reg, uint32_t value) {
    cu->write_sgpr(wf->sgpr_alloc().base + reg, value);
  }
  uint32_t read_sgpr(unsigned reg) { return cu->read_sgpr(wf->sgpr_alloc().base + reg); }

  std::unique_ptr<Instruction> prepare(cdna3::SmemBuilderFields fields = {
                                           .sdata = 4, .glc = 1, .imm = 1}) {
    const auto words = cdna3::build_smem(cdna3::kSAtomicDecSmem, fields);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    if (!inst)
      return nullptr;
    if (cu->execute_instruction(inst.get(), *wf).failed()) {
      ADD_FAILURE() << "s_atomic_dec execution failed";
      return nullptr;
    }
    EXPECT_TRUE(inst->is_memory_op());
    return inst;
  }

  GpuMemory memory{"scalar_atomic_memory"};
  GpuVm vm;
  L2Cache l2{"scalar_atomic_l2"};
  std::shared_ptr<ScalarAtomicBacking> backing;
  std::unique_ptr<ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  std::unique_ptr<ScalarMemPipeline> pipeline;
  Wavefront *wf = nullptr;
};

INSTANTIATE_TEST_SUITE_P(
    Cdna, ScalarAtomicTest,
    ::testing::Combine(::testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4),
                       ::testing::Values(MemoryMode::Direct, MemoryMode::CachedVm,
                                         MemoryMode::TranslatedVm)));

TEST_P(ScalarAtomicTest, DecrementWrapAndOptionalReturn) {
  // MI300 ISA, S_ATOMIC_DEC: zero and values above DATA wrap to DATA.
  constexpr std::array<std::array<uint32_t, 3>, 10> cases{{{0, 0, 0},
                                                           {1, 0, 0},
                                                           {0xffffffff, 0, 0},
                                                           {0, 7, 7},
                                                           {1, 7, 0},
                                                           {7, 7, 6},
                                                           {8, 7, 7},
                                                           {0xffffffff, 7, 7},
                                                           {0, 0xffffffff, 0xffffffff},
                                                           {0xffffffff, 0xffffffff, 0xfffffffe}}};
  for (const auto &[old, limit, expected] : cases) {
    for (uint8_t glc : {0, 1}) {
      SCOPED_TRACE(testing::Message() << "old=" << old << " limit=" << limit << " glc=" << +glc);
      memory.write32(kAddress, old);
      memory.write32(kAddress + 4, 0xdeadbeef);
      write_sgpr(4, limit);
      write_sgpr(5, 0x12345678);
      wf->write_scc(1);
      auto inst = prepare({.sdata = 4, .glc = glc, .imm = 1});
      ASSERT_NE(inst, nullptr);
      EXPECT_EQ(memory.read32(kAddress), old);
      EXPECT_EQ(read_sgpr(4), limit);
      ASSERT_EQ(pipeline->issue(inst.release(), *wf), VmAccessOutcome::Complete);
      EXPECT_EQ(memory.read32(kAddress), expected);
      EXPECT_EQ(read_sgpr(4), glc ? old : limit);
      EXPECT_EQ(memory.read32(kAddress + 4), 0xdeadbeef);
      EXPECT_EQ(read_sgpr(5), 0x12345678);
      EXPECT_EQ(wf->read_scc(), 1u);
      EXPECT_EQ(wf->exec(), 0u);
      EXPECT_TRUE(wf->wait_counters().empty());
    }
  }
}

TEST_P(ScalarAtomicTest, ImmediateRegisterAndCombinedOffsets) {
  const std::array fields{
      cdna3::SmemBuilderFields{.sbase = 3, .sdata = 4, .glc = 1, .imm = 1, .offset = 12},
      // LLVM accepts SGPR offsets, but their hardware behavior is unverified.
      // CDNA1-4 manuals specify immediate/M0 atomic offsets (MI300 Table 38).
      cdna3::SmemBuilderFields{.sbase = 3, .sdata = 4, .glc = 1, .offset = 8},
      // Likewise, this immediate + SGPR form covers LLVM/simulator compatibility.
      cdna3::SmemBuilderFields{.sbase = 3,
                               .sdata = 4,
                               .soffset_en = 1,
                               .glc = 1,
                               .imm = 1,
                               .offset = 0x1ffffc,
                               .soffset = 8},
      cdna3::SmemBuilderFields{.sbase = 3,
                               .sdata = 4,
                               .soffset_en = 1,
                               .glc = 1,
                               .imm = 1,
                               .offset = 8,
                               .soffset = 124}};
  for (size_t form = 0; form < fields.size(); ++form) {
    SCOPED_TRACE(form);
    // Low address bits are ignored separately for base and each offset.
    write_sgpr(6, kAddress + 3);
    write_sgpr(7, 1);
    write_sgpr(8, form == 2 ? 19 : 15);
    wf->set_m0(7);
    write_sgpr(4, 9);
    const uint64_t address = (uint64_t{1} << 32) + kAddress + 12;
    memory.write32(address, 4);
    auto inst = prepare(fields[form]);
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(pipeline->issue(inst.release(), *wf), VmAccessOutcome::Complete);
    EXPECT_EQ(memory.read32(address), 3u);
    EXPECT_EQ(read_sgpr(4), 4u);
  }
}

TEST_P(ScalarAtomicTest, SourceAndAddressAreCapturedBeforeReturn) {
  for (bool aliases_base : {false, true}) {
    write_sgpr(0, kAddress);
    write_sgpr(4, 16);
    const uint64_t address = kAddress + (aliases_base ? 0 : 16);
    memory.write32(address, 4);
    // The SGPR-offset alias case covers an LLVM-accepted encoding whose hardware
    // behavior is unverified, as in ImmediateRegisterAndCombinedOffsets above.
    auto inst = prepare(aliases_base ? cdna3::SmemBuilderFields{.sdata = 0, .glc = 1, .imm = 1}
                                     : cdna3::SmemBuilderFields{.sdata = 4, .glc = 1, .offset = 4});
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(pipeline->issue(inst.release(), *wf), VmAccessOutcome::Complete);
    EXPECT_EQ(memory.read32(address), 3u);
    EXPECT_EQ(read_sgpr(aliases_base ? 0 : 4), 4u);
  }
}

TEST_P(ScalarAtomicTest, NamedScalarRegisterCanSupplyAndReceiveData) {
  memory.write32(kAddress, 4);
  wf->set_vcc(0x1234567800000009ull);
  auto inst = prepare({.sdata = 106, .glc = 1, .imm = 1});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(pipeline->issue(inst.release(), *wf), VmAccessOutcome::Complete);
  EXPECT_EQ(memory.read32(kAddress), 3u);
  EXPECT_EQ(wf->vcc(), 0x1234567800000004ull);
}

TEST_P(ScalarAtomicTest, AtomicPublishesCachedWritesAndInvalidatesScalarCopies) {
  if (memory_mode() == MemoryMode::TranslatedVm)
    GTEST_SKIP() << "Translated accesses bypass the legacy cache hierarchy";
  memory.write32(kAddress, 8);
  uint32_t cached = 0;
  ASSERT_EQ(cu->l1_scalar().load(kAddress, 1, &cached, wf->process_id()),
            VmAccessOutcome::Complete);
  ASSERT_EQ(cached, 8u);
  const std::array<uint32_t, 2> dirty{4, 0x12345678};
  ASSERT_EQ(l2.write(kAddress, reinterpret_cast<const uint8_t *>(dirty.data()), sizeof(dirty),
                     Mtype::RW, wf->process_id()),
            VmAccessOutcome::Complete);
  write_sgpr(4, 9);
  auto inst = prepare();
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(pipeline->issue(inst.release(), *wf), VmAccessOutcome::Complete);
  EXPECT_EQ(read_sgpr(4), 4u);
  EXPECT_EQ(memory.read32(kAddress), 3u);
  EXPECT_EQ(memory.read32(kAddress + 4), dirty[1]);
  ASSERT_EQ(cu->l1_scalar().load(kAddress, 1, &cached, wf->process_id()),
            VmAccessOutcome::Complete);
  EXPECT_EQ(cached, 3u);
}

TEST_P(ScalarAtomicTest, UnavailableAtomicRetainsCounterAndUpdatesExactlyOnce) {
  if (!backing)
    GTEST_SKIP() << "Retry injection requires a VM backing";
  for (uint8_t glc : {0, 1}) {
    memory.write32(kAddress, 4);
    write_sgpr(4, 9);
    backing->load_attempts = 0;
    backing->update_attempts = 0;
    backing->mutations = 0;
    backing->outcome = VmAccessOutcome::Unavailable;
    auto inst = prepare({.sdata = 4, .glc = glc, .imm = 1});
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(pipeline->issue_deferred(inst.release(), *wf), VmAccessOutcome::Complete);
    EXPECT_EQ(wf->wait_counters().lgkmcnt, 1u);
    EXPECT_EQ(wf->wait_counters().vmcnt, 0u);
    EXPECT_EQ(read_sgpr(4), 9u);
    EXPECT_EQ(memory.read32(kAddress), 4u);
    pipeline->tick();
    EXPECT_EQ(wf->wait_counters().lgkmcnt, 1u);
    EXPECT_EQ(backing->mutations, 0u);
    backing->outcome = VmAccessOutcome::Complete;
    pipeline->tick();
    EXPECT_TRUE(wf->wait_counters().empty());
    if (memory_mode() == MemoryMode::TranslatedVm) {
      EXPECT_EQ(backing->load_attempts, 3u);
      EXPECT_EQ(backing->update_attempts, 1u);
    } else {
      EXPECT_EQ(backing->update_attempts, 3u);
    }
    EXPECT_EQ(backing->mutations, 1u);
    EXPECT_EQ(read_sgpr(4), glc ? 4u : 9u);
    EXPECT_EQ(memory.read32(kAddress), 3u);
  }
}

TEST_P(ScalarAtomicTest, FailedAtomicDoesNotWriteBackOrLeakCounter) {
  if (!backing)
    GTEST_SKIP() << "Fault injection requires a VM backing";
  memory.write32(kAddress, 4);
  write_sgpr(4, 9);
  backing->outcome = VmAccessOutcome::Faulted;
  auto inst = prepare();
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(pipeline->issue(inst.release(), *wf), VmAccessOutcome::Faulted);
  EXPECT_TRUE(wf->wait_counters().empty());
  EXPECT_EQ(backing->mutations, 0u);
  EXPECT_EQ(read_sgpr(4), 9u);
  EXPECT_EQ(memory.read32(kAddress), 4u);
}

TEST_P(ScalarAtomicTest, UnavailableCompareExchangeRetriesWithCurrentMemoryValue) {
  if (memory_mode() != MemoryMode::TranslatedVm)
    GTEST_SKIP() << "Compare/exchange retries require a translated backing";
  for (uint8_t glc : {0, 1}) {
    memory.write32(kAddress, 4);
    write_sgpr(4, 9);
    backing->load_attempts = 0;
    backing->update_attempts = 0;
    backing->mutations = 0;
    backing->exchange_outcome = VmAccessOutcome::Unavailable;
    auto inst = prepare({.sdata = 4, .glc = glc, .imm = 1});
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(pipeline->issue_deferred(inst.release(), *wf), VmAccessOutcome::Complete);
    pipeline->tick();
    EXPECT_EQ(wf->wait_counters().lgkmcnt, 1u);
    EXPECT_EQ(backing->load_attempts, 1u);
    EXPECT_EQ(backing->mutations, 0u);
    EXPECT_EQ(read_sgpr(4), 9u);
    EXPECT_EQ(memory.read32(kAddress), 4u);

    // Another writer changes memory after our load. The resumed CAS must retry
    // with its observed value: zero wraps to nine, and GLC must return zero.
    memory.write32(kAddress, 0);
    backing->exchange_outcome = VmAccessOutcome::Complete;
    pipeline->tick();
    EXPECT_TRUE(wf->wait_counters().empty());
    EXPECT_EQ(backing->load_attempts, 1u);
    EXPECT_EQ(backing->update_attempts, 4u);
    EXPECT_EQ(backing->mutations, 1u);
    EXPECT_EQ(read_sgpr(4), glc ? 0u : 9u);
    EXPECT_EQ(memory.read32(kAddress), 9u);

    pipeline->tick();
    EXPECT_EQ(backing->mutations, 1u);
    EXPECT_EQ(memory.read32(kAddress), 9u);
  }
}

TEST_P(ScalarAtomicTest, FailedCompareExchangeDoesNotWriteBackOrLeakCounter) {
  if (memory_mode() != MemoryMode::TranslatedVm)
    GTEST_SKIP() << "Compare/exchange faults require a translated backing";
  for (uint8_t glc : {0, 1}) {
    memory.write32(kAddress, 4);
    write_sgpr(4, 9);
    backing->load_attempts = 0;
    backing->update_attempts = 0;
    backing->exchange_outcome = VmAccessOutcome::Faulted;
    auto inst = prepare({.sdata = 4, .glc = glc, .imm = 1});
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(pipeline->issue(inst.release(), *wf), VmAccessOutcome::Faulted);
    EXPECT_TRUE(wf->wait_counters().empty());
    EXPECT_EQ(backing->load_attempts, 1u);
    EXPECT_EQ(backing->update_attempts, 1u);
    EXPECT_EQ(backing->mutations, 0u);
    EXPECT_EQ(read_sgpr(4), 9u);
    EXPECT_EQ(memory.read32(kAddress), 4u);
  }
}

TEST_P(ScalarAtomicTest, ReturnedRegisterRequiresLgkmWait) {
  for (bool waited : {false, true}) {
    for (uint8_t glc : {0, 1}) {
      const uint64_t program = kProgram + (waited * 2 + glc) * 0x100;
      wf->pc = program;
      wf->ensure_memory_wait_scoreboard().clear();
      const uint64_t before = cu->memory_wait_diagnostic_count();
      write_sgpr(4, 9);
      memory.write32(kAddress, 4);
      const auto atomic =
          cdna3::build_smem(cdna3::kSAtomicDecSmem, {.sdata = 4, .glc = glc, .imm = 1});
      std::vector<uint32_t> words(atomic.begin(), atomic.end());
      // lgkmcnt(0), leaving vmcnt/expcnt unconstrained.
      words.push_back(waited ? 0xbf8c007fu : 0xbf800000u);
      words.push_back(0xbe850004u); // s_mov_b32 s5, s4
      words.insert(words.end(), 4, 0xbf800000u);
      for (size_t i = 0; i < words.size(); ++i)
        memory.write32(program + i * 4, words[i]);
      for (unsigned i = 0; i < 3; ++i)
        (void)cu->step();
      ASSERT_FALSE(wf->is_halted());
      EXPECT_EQ(wf->pc, program + 16);
      EXPECT_EQ(read_sgpr(5), glc ? 4u : 9u);
      EXPECT_EQ(cu->memory_wait_diagnostic_count() - before, glc && !waited ? 1u : 0u);
    }
  }
}

TEST_P(ScalarAtomicTest, ReportsScalarReadModifyWriteToPlugins) {
  class Observer final : public ExecutionPlugin {
  public:
    Observer() : ExecutionPlugin("scalar_atomic_observer") {}
    bool observes_memory_routing() const override { return true; }
    void onAmdgpuMemoryAccessRouted(const MemoryAccessObservation &access) override {
      EXPECT_EQ(access.mnemonic, "s_atomic_dec");
      EXPECT_EQ(access.route, MemoryRoute::SCALAR);
      EXPECT_EQ(access.atomic_op, AtomicOp::DEC);
      EXPECT_EQ(access.wait_counter, WaitCounterType::LGKMCNT);
      EXPECT_EQ(access.element_size_bytes, 4u);
      EXPECT_EQ(access.elements_per_lane, 1u);
      EXPECT_EQ(access.request_lane_mask, 1u);
      ASSERT_EQ(access.addresses.size(), 1u);
      EXPECT_EQ(access.addresses[0], kAddress);
      returns.push_back(access.is_load);
    }
    std::vector<bool> returns;
  };
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<Observer>();
  auto *record = observer.get();
  ASSERT_TRUE(group->add(std::move(observer)));
  cu->set_plugin_group(group);
  for (uint8_t glc : {0, 1}) {
    write_sgpr(4, 9);
    memory.write32(kAddress, 4);
    const uint64_t program = kProgram + glc * 0x100;
    wf->pc = program;
    const auto words =
        cdna3::build_smem(cdna3::kSAtomicDecSmem, {.sdata = 4, .glc = glc, .imm = 1});
    memory.write32(program, words[0]);
    memory.write32(program + 4, words[1]);
    memory.write32(program + 8, 0xbf800000);
    memory.write32(program + 12, 0xbf800000);
    (void)cu->step();
    ASSERT_FALSE(wf->is_halted());
    EXPECT_EQ(wf->pc, program + 8);
  }
  EXPECT_EQ(record->returns, (std::vector<bool>{false, true}));
}
} // namespace
