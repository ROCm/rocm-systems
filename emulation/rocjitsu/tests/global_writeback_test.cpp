// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "decode_test_util.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

namespace rocjitsu {
namespace {
using namespace amdgpu;

class GlobalWriteback : public testing::TestWithParam<rj_code_arch_t> {
protected:
  void SetUp() override {
    l2.set_backing_memory(&memory);
    ComputeUnitCore::Config config{};
    config.arch = GetParam();
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 32;
    config.vgprs_per_wf = 16;
    ASSERT_NO_FATAL_FAILURE(
        test::initialize_compute_unit(cu, "writeback_cu", config, &memory, &l2));
    wave = cu->dispatch_wf(0, pc, 32, 16, 32);
    ASSERT_TRUE(wave);
    wave->set_exec(0);
  }
  void TearDown() override {
    if (wave)
      wave->halt();
  }
  // Both VGLOBAL encodings have no explicit operands. SCOPE occupies bits 19:18 of word 1.
  static std::array<uint32_t, 3> words(uint32_t scope) { return {0xee0b007c, scope << 18, 0}; }

  void dirty(uint32_t addr, uint8_t value) {
    std::array<uint8_t, L2Cache::LINE_SIZE> line{};
    line[0] = value;
    ASSERT_EQ(l2.writeback_line(addr, line.data(), 0, 1, Mtype::RW), VmAccessOutcome::Complete);
  }

  void issue(uint32_t scope) {
    const auto encoding = words(scope);
    for (auto word : encoding) {
      memory.write32(pc, word);
      pc += 4;
    }
    cu->instruction_cache().invalidate_all();
    cu->step();
    ASSERT_EQ(wave->pc, pc);
    EXPECT_FALSE(wave->is_halted());
    EXPECT_TRUE(wave->wait_counters().empty());
  }

  static constexpr uint32_t address = 0x4000;
  uint64_t pc = 0x1000;
  GpuMemory memory{"writeback_ram"};
  L2Cache l2{"writeback_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  Wavefront *wave = nullptr;
};

TEST_P(GlobalWriteback, ScopePromotesDirtyBytesAndKeepsTheLineResident) {
  // CDNA5 may partition the device into multiple XCDs with separate L2s.
  std::optional<L2Cache> peer;
  if (GetParam() == ROCJITSU_CODE_ARCH_CDNA5) {
    peer.emplace("peer_l2", l2.coherence_domain());
    peer->set_backing_memory(&memory);
  }
  constexpr auto peer_address = address + 8 * L2Cache::LINE_SIZE;
  if (peer) {
    std::array<uint8_t, L2Cache::LINE_SIZE> line{};
    line[0] = 0x59;
    ASSERT_EQ(peer->writeback_line(peer_address, line.data(), 0, 1, Mtype::RW),
              VmAccessOutcome::Complete);
  }
  const uint8_t value = 0xa7;
  for (uint32_t scope = 0; scope < 4; ++scope) {
    SCOPED_TRACE(scope);
    const auto addr = address + scope * L2Cache::LINE_SIZE;
    ASSERT_NO_FATAL_FAILURE(dirty(addr, value));
    memory.write32(addr, 0x44332211);
    ASSERT_NO_FATAL_FAILURE(issue(scope));
    const bool published = scope == 3 || (scope == 2 && peer.has_value());
    EXPECT_EQ(memory.read32(addr), published ? 0x443322a7u : 0x44332211u);
    uint8_t cached = 0;
    EXPECT_TRUE(l2.try_read_peer_cached(addr, {&cached, 1}, 0, 0, 0));
    EXPECT_EQ(cached, value);
  }
  auto writes = l2.backing_write_transactions();
  ASSERT_NO_FATAL_FAILURE(issue(3));
  EXPECT_EQ(l2.backing_write_transactions(), writes);
  EXPECT_EQ(memory.read32(peer_address), 0u);
}

class HeldWriteback : public GlobalMemPipeline {
public:
  using GlobalMemPipeline::GlobalMemPipeline;
  bool held = true;
  unsigned attempts = 0;

protected:
  VmAccessOutcome initiate_access(Instruction &inst, Wavefront &wave) override {
    ++attempts;
    EXPECT_EQ(wave.wait_counters().vscnt, 1u);
    return held ? VmAccessOutcome::Unavailable : GlobalMemPipeline::initiate_access(inst, wave);
  }
};

TEST_P(GlobalWriteback, RetryKeepsOneStoreCounterUntilAcknowledged) {
  auto decoder = Decoder::create(GetParam());
  auto encoding = words(3);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, encoding.data()));
  ASSERT_TRUE(inst);
  ASSERT_TRUE(inst->is_memory_op());
  ASSERT_EQ(inst->num_src_operands(), 0);
  ASSERT_EQ(inst->num_dst_operands(), 0);
  auto *info = inst->amdgpu_memory_issue_info();
  ASSERT_TRUE(info);
  EXPECT_FALSE(info->exec_masked);
  ASSERT_EQ(info->counter_obligations().size(), 1u);
  EXPECT_EQ(info->counter_obligations()[0].wait_counter_type(), WaitCounterType::STORECNT);
  ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
  ASSERT_TRUE(inst->data());
  const uint8_t value = 0x81;
  ASSERT_NO_FATAL_FAILURE(dirty(address, value));
  HeldWriteback pipe(&cu->l1_vector(), &l2);
  ASSERT_EQ(pipe.issue_deferred(inst, *wave), VmAccessOutcome::Complete);
  EXPECT_FALSE(inst);
  EXPECT_EQ(wave->state(), WfState::VM_RETRY);
  pipe.tick();
  EXPECT_EQ(wave->wait_counters().vscnt, 1u);
  EXPECT_EQ(memory.read32(address), 0u);
  pipe.held = false;
  pipe.tick();
  EXPECT_EQ(pipe.attempts, 3u);
  EXPECT_EQ(memory.read32(address), value);
  EXPECT_TRUE(wave->wait_counters().empty());
  EXPECT_TRUE(pipe.empty());
}

INSTANTIATE_TEST_SUITE_P(Gfx12, GlobalWriteback,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5));
} // namespace
} // namespace rocjitsu
