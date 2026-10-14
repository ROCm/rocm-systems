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
#include <string_view>
#include <tuple>
#include <utility>
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
  void write_sgprs(unsigned reg, uint32_t dwords, uint64_t value) {
    for (uint32_t i = 0; i < dwords; ++i)
      write_sgpr(reg + i, static_cast<uint32_t>(value >> (32 * i)));
  }
  uint64_t read_sgprs(unsigned reg, uint32_t dwords) {
    uint64_t value = 0;
    for (uint32_t i = 0; i < dwords; ++i)
      value |= uint64_t{read_sgpr(reg + i)} << (32 * i);
    return value;
  }
  void write_memory(uint64_t address, uint32_t dwords, uint64_t value) {
    if (dwords == 1)
      memory.write32(address, static_cast<uint32_t>(value));
    else
      memory.write64(address, value);
  }
  uint64_t read_memory(uint64_t address, uint32_t dwords) {
    return dwords == 1 ? memory.read32(address) : memory.read64(address);
  }

  std::unique_ptr<Instruction>
  prepare(cdna3::SmemBuilderFields fields = {.sdata = 4, .glc = 1, .imm = 1},
          uint16_t opcode = cdna3::kSAtomicDecSmem) {
    const auto words = cdna3::build_smem(opcode, fields);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    if (!inst)
      return nullptr;
    if (cu->execute_instruction(inst.get(), *wf).failed()) {
      ADD_FAILURE() << inst->mnemonic() << " execution failed";
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

struct ReadModifyWriteCase {
  uint16_t opcode;
  uint32_t dwords;
  uint64_t old;
  uint64_t data;
  uint64_t expected;
};

TEST_P(ScalarAtomicTest, ReadModifyWriteAndOptionalReturn) {
  constexpr std::array<ReadModifyWriteCase, 15> cases{{
      // MI300 ISA, S_ATOMIC_DEC: zero and values above DATA wrap to DATA.
      {cdna3::kSAtomicDecSmem, 1, 0, 0, 0},
      {cdna3::kSAtomicDecSmem, 1, 1, 0, 0},
      {cdna3::kSAtomicDecSmem, 1, 0xffffffff, 0, 0},
      {cdna3::kSAtomicDecSmem, 1, 0, 7, 7},
      {cdna3::kSAtomicDecSmem, 1, 1, 7, 0},
      {cdna3::kSAtomicDecSmem, 1, 7, 7, 6},
      {cdna3::kSAtomicDecSmem, 1, 8, 7, 7},
      {cdna3::kSAtomicDecSmem, 1, 0xffffffff, 7, 7},
      {cdna3::kSAtomicDecSmem, 1, 0, 0xffffffff, 0xffffffff},
      {cdna3::kSAtomicDecSmem, 1, 0xffffffff, 0xffffffff, 0xfffffffe},
      // S_ATOMIC_DEC_X2 borrows across dwords and compares all 64 bits.
      {cdna3::kSAtomicDecX2Smem, 2, 0x1'0000'0000, ~uint64_t{0}, 0xffff'ffff},
      {cdna3::kSAtomicDecX2Smem, 2, 0x1'0000'0000, 0x1'0000'0000, 0xffff'ffff},
      {cdna3::kSAtomicDecX2Smem, 2, 0x2'0000'0001, 0x1'0000'0005, 0x1'0000'0005},
      {cdna3::kSAtomicDecX2Smem, 2, 0, 0x1'2345'6789, 0x1'2345'6789},
      {cdna3::kSAtomicDecX2Smem, 2, ~uint64_t{0}, ~uint64_t{0}, ~uint64_t{1}},
  }};
  for (const auto &[opcode, dwords, old, data, expected] : cases) {
    for (uint8_t glc : {0, 1}) {
      SCOPED_TRACE(testing::Message()
                   << "opcode=" << opcode << " old=" << old << " data=" << data << " glc=" << +glc);
      const uint64_t guard_address = kAddress + dwords * 4;
      const unsigned guard_sgpr = 4 + dwords;
      write_memory(kAddress, dwords, old);
      memory.write32(guard_address, 0xdeadbeef);
      write_sgprs(4, dwords, data);
      write_sgpr(guard_sgpr, 0x12345678);
      wf->write_scc(1);
      auto inst = prepare({.sdata = 4, .glc = glc, .imm = 1}, opcode);
      ASSERT_NE(inst, nullptr);
      EXPECT_EQ(read_memory(kAddress, dwords), old);
      EXPECT_EQ(read_sgprs(4, dwords), data);
      ASSERT_EQ(pipeline->issue(inst.release(), *wf), VmAccessOutcome::Complete);
      EXPECT_EQ(read_memory(kAddress, dwords), expected);
      EXPECT_EQ(read_sgprs(4, dwords), glc ? old : data);
      EXPECT_EQ(memory.read32(guard_address), 0xdeadbeef);
      EXPECT_EQ(read_sgpr(guard_sgpr), 0x12345678);
      EXPECT_EQ(wf->read_scc(), 1u);
      EXPECT_EQ(wf->exec(), 0u);
      EXPECT_TRUE(wf->wait_counters().empty());
    }
  }
}

TEST_P(ScalarAtomicTest, ImmediateRegisterAndCombinedOffsets) {
  const std::array fields{
      cdna3::SmemBuilderFields{.sbase = 3, .sdata = 4, .glc = 1, .imm = 1, .offset = 12},
      cdna3::SmemBuilderFields{.sbase = 3, .sdata = 4, .glc = 1, .offset = 8},
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

constexpr std::array<std::pair<uint16_t, uint32_t>, 2> kDecrementForms{
    {{cdna3::kSAtomicDecSmem, 1}, {cdna3::kSAtomicDecX2Smem, 2}}};

TEST_P(ScalarAtomicTest, UnavailableAtomicRetainsCounterAndUpdatesExactlyOnce) {
  if (!backing)
    GTEST_SKIP() << "Retry injection requires a VM backing";
  for (const auto &[opcode, dwords] : kDecrementForms) {
    for (uint8_t glc : {0, 1}) {
      SCOPED_TRACE(testing::Message() << "opcode=" << opcode << " glc=" << +glc);
      write_memory(kAddress, dwords, 4);
      write_sgprs(4, dwords, 9);
      backing->load_attempts = 0;
      backing->update_attempts = 0;
      backing->mutations = 0;
      backing->outcome = VmAccessOutcome::Unavailable;
      auto inst = prepare({.sdata = 4, .glc = glc, .imm = 1}, opcode);
      ASSERT_NE(inst, nullptr);
      ASSERT_EQ(pipeline->issue_deferred(inst.release(), *wf), VmAccessOutcome::Complete);
      // Two-dword scalar atomics hold two LGKM tokens, like two-dword scalar loads.
      EXPECT_EQ(wf->wait_counters().lgkmcnt, dwords);
      EXPECT_EQ(wf->wait_counters().vmcnt, 0u);
      EXPECT_EQ(read_sgprs(4, dwords), 9u);
      EXPECT_EQ(read_memory(kAddress, dwords), 4u);
      pipeline->tick();
      EXPECT_EQ(wf->wait_counters().lgkmcnt, dwords);
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
      EXPECT_EQ(read_sgprs(4, dwords), glc ? 4u : 9u);
      EXPECT_EQ(read_memory(kAddress, dwords), 3u);
    }
  }
}

TEST_P(ScalarAtomicTest, MisalignedTwoDwordAtomicFaultsWithoutUpdate) {
  // On gfx90a, S_ATOMIC_*_X2 at an address that is 4 mod 8 raises a memory
  // violation and leaves both neighboring qwords unchanged.
  memory.write64(kAddress, 0x5'0000'0005);
  memory.write64(kAddress + 8, 0x5'0000'0005);
  write_sgprs(4, 2, 9);
  auto inst = prepare({.sdata = 4, .glc = 1, .imm = 1, .offset = 4}, cdna3::kSAtomicDecX2Smem);
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(pipeline->issue(inst.release(), *wf), VmAccessOutcome::Malformed);
  EXPECT_TRUE(wf->wait_counters().empty());
  EXPECT_EQ(memory.read64(kAddress), 0x5'0000'0005u);
  EXPECT_EQ(memory.read64(kAddress + 8), 0x5'0000'0005u);
  EXPECT_EQ(read_sgprs(4, 2), 9u);
  if (backing) {
    EXPECT_EQ(backing->load_attempts, 0u);
    EXPECT_EQ(backing->update_attempts, 0u);
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
  for (const auto &[opcode, dwords] : kDecrementForms) {
    for (uint8_t glc : {0, 1}) {
      SCOPED_TRACE(testing::Message() << "opcode=" << opcode << " glc=" << +glc);
      write_memory(kAddress, dwords, 4);
      write_sgprs(4, dwords, 9);
      backing->load_attempts = 0;
      backing->update_attempts = 0;
      backing->mutations = 0;
      backing->exchange_outcome = VmAccessOutcome::Unavailable;
      auto inst = prepare({.sdata = 4, .glc = glc, .imm = 1}, opcode);
      ASSERT_NE(inst, nullptr);
      ASSERT_EQ(pipeline->issue_deferred(inst.release(), *wf), VmAccessOutcome::Complete);
      pipeline->tick();
      EXPECT_EQ(wf->wait_counters().lgkmcnt, dwords);
      EXPECT_EQ(backing->load_attempts, 1u);
      EXPECT_EQ(backing->mutations, 0u);
      EXPECT_EQ(read_sgprs(4, dwords), 9u);
      EXPECT_EQ(read_memory(kAddress, dwords), 4u);

      // Another writer changes memory after our load. The resumed CAS must retry
      // with its observed value: zero wraps to nine, and GLC must return zero.
      write_memory(kAddress, dwords, 0);
      backing->exchange_outcome = VmAccessOutcome::Complete;
      pipeline->tick();
      EXPECT_TRUE(wf->wait_counters().empty());
      EXPECT_EQ(backing->load_attempts, 1u);
      EXPECT_EQ(backing->update_attempts, 4u);
      EXPECT_EQ(backing->mutations, 1u);
      EXPECT_EQ(read_sgprs(4, dwords), glc ? 0u : 9u);
      EXPECT_EQ(read_memory(kAddress, dwords), 9u);

      pipeline->tick();
      EXPECT_EQ(backing->mutations, 1u);
      EXPECT_EQ(read_memory(kAddress, dwords), 9u);
    }
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
  for (const auto &[opcode, dwords] : kDecrementForms) {
    for (bool waited : {false, true}) {
      for (uint8_t glc : {0, 1}) {
        SCOPED_TRACE(testing::Message()
                     << "opcode=" << opcode << " waited=" << waited << " glc=" << +glc);
        const uint64_t program = kProgram + ((dwords - 1) * 4 + waited * 2 + glc) * 0x100;
        wf->pc = program;
        wf->ensure_memory_wait_scoreboard().clear();
        const uint64_t before = cu->memory_wait_diagnostic_count();
        write_sgprs(4, dwords, 0x9'0000'0009);
        write_memory(kAddress, dwords, 0x7'0000'0004);
        const auto atomic = cdna3::build_smem(opcode, {.sdata = 4, .glc = glc, .imm = 1});
        std::vector<uint32_t> words(atomic.begin(), atomic.end());
        // lgkmcnt(0), leaving vmcnt/expcnt unconstrained.
        words.push_back(waited ? 0xbf8c007fu : 0xbf800000u);
        // s_mov_b32 from the last returned dword, so X2 checks its high dword.
        const uint32_t last = 4 + dwords - 1;
        words.push_back(0xbe800000u | ((last + 1) << 16) | last);
        words.insert(words.end(), 4, 0xbf800000u);
        for (size_t i = 0; i < words.size(); ++i)
          memory.write32(program + i * 4, words[i]);
        for (unsigned i = 0; i < 3; ++i)
          (void)cu->step();
        ASSERT_FALSE(wf->is_halted());
        EXPECT_EQ(wf->pc, program + 16);
        EXPECT_EQ(read_sgpr(last + 1), glc ? (dwords == 1 ? 4u : 7u) : 9u);
        EXPECT_EQ(cu->memory_wait_diagnostic_count() - before, glc && !waited ? 1u : 0u);
      }
    }
  }
}

TEST_P(ScalarAtomicTest, ReportsScalarReadModifyWriteToPlugins) {
  class Observer final : public ExecutionPlugin {
  public:
    Observer() : ExecutionPlugin("scalar_atomic_observer") {}
    bool observes_memory_routing() const override { return true; }
    void onAmdgpuMemoryAccessRouted(const MemoryAccessObservation &access) override {
      EXPECT_EQ(access.route, MemoryRoute::SCALAR);
      EXPECT_EQ(access.atomic_op, AtomicOp::DEC);
      EXPECT_EQ(access.wait_counter, WaitCounterType::LGKMCNT);
      EXPECT_EQ(access.element_size_bytes, 4u);
      EXPECT_EQ(access.request_lane_mask, 1u);
      ASSERT_EQ(access.addresses.size(), 1u);
      EXPECT_EQ(access.addresses[0], kAddress);
      observed.emplace_back(access.mnemonic, access.elements_per_lane, access.is_load);
    }
    std::vector<std::tuple<std::string_view, uint32_t, bool>> observed;
  };
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<Observer>();
  auto *record = observer.get();
  ASSERT_TRUE(group->add(std::move(observer)));
  cu->set_plugin_group(group);
  for (const auto &[opcode, dwords] : kDecrementForms) {
    for (uint8_t glc : {0, 1}) {
      write_sgprs(4, dwords, 9);
      write_memory(kAddress, dwords, 4);
      const uint64_t program = kProgram + ((dwords - 1) * 2 + glc) * 0x100;
      wf->pc = program;
      const auto words = cdna3::build_smem(opcode, {.sdata = 4, .glc = glc, .imm = 1});
      memory.write32(program, words[0]);
      memory.write32(program + 4, words[1]);
      memory.write32(program + 8, 0xbf800000);
      memory.write32(program + 12, 0xbf800000);
      (void)cu->step();
      ASSERT_FALSE(wf->is_halted());
      EXPECT_EQ(wf->pc, program + 8);
    }
  }
  EXPECT_EQ(record->observed, (std::vector<std::tuple<std::string_view, uint32_t, bool>>{
                                  {"s_atomic_dec", 1, false},
                                  {"s_atomic_dec", 1, true},
                                  {"s_atomic_dec_x2", 2, false},
                                  {"s_atomic_dec_x2", 2, true},
                              }));
}
} // namespace
