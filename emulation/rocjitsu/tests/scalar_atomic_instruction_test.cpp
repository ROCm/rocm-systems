// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <array>
#include <memory>

namespace rocjitsu {
namespace {
class ScalarAtomicInstruction : public testing::TestWithParam<rj_code_arch_t> {
protected:
  void SetUp() override {
    l2_.set_backing_memory(&memory_);
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = GetParam();
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 102;
    config.vgprs_per_wf = 16;
    config.lds_size_kb = 64;
    cu_ = amdgpu::ComputeUnitCore::create("scalar_atomic", config, &memory_, &l2_);
    wave_ = cu_->dispatch_wf(0, 0, 102, 16);
    ASSERT_NE(wave_, nullptr);
    // Scalar memory instructions execute even with no active vector lanes.
    wave_->set_exec(0);
    decoder_ = Decoder::create(GetParam());
  }
  void TearDown() override {
    if (wave_)
      wave_->halt();
  }
  void put(unsigned reg, uint32_t value) { cu_->write_sgpr(wave_->sgpr_alloc().base + reg, value); }
  uint32_t get(unsigned reg) { return cu_->read_sgpr(wave_->sgpr_alloc().base + reg); }
  void base(uint32_t records = 0xffffffff, uint32_t stride = 0) {
    put(0, 0x4000);
    put(1, stride << 16);
    put(2, records);
    put(3, 0);
  }
  void issue(uint32_t opcode, bool returning, uint32_t offset = 0) {
    std::array<uint32_t, 2> words{0xc0000000u | (opcode << 18) | (1u << 17) |
                                      (uint32_t{returning} << 16) | (32u << 6),
                                  offset};
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder_, words.data()));
    ASSERT_NE(instruction, nullptr);
    SCOPED_TRACE(instruction->mnemonic());
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
    ASSERT_NE(instruction->data(), nullptr);
    ASSERT_EQ(pipeline_.issue(instruction.release(), *wave_), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(wave_->wait_counters().lgkmcnt, 0u);
  }
  amdgpu::GpuMemory memory_{"atomic_memory"};
  amdgpu::L2Cache l2_{"atomic_l2"};
  amdgpu::L1ScalarCache l1_{&l2_};
  amdgpu::ScalarMemPipeline pipeline_{&l1_};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_;
  amdgpu::Wavefront *wave_ = nullptr;
};

TEST_P(ScalarAtomicInstruction, AllIntegerOperationsWithAndWithoutReturn) {
  // SWAP, CMPSWAP, ADD, SUB, SMIN, UMIN, SMAX, UMAX, AND, OR, XOR, INC, DEC.
  constexpr std::array<uint32_t, 13> expected{3, 3, 10, 4, 3, 3, 7, 7, 3, 7, 4, 0, 3};
  base();
  for (bool buffer : {false, true})
    for (bool wide : {false, true})
      for (bool returning : {false, true})
        for (unsigned op = 0; op < expected.size(); ++op) {
          SCOPED_TRACE(testing::Message() << "op=" << op << " buffer=" << buffer << " wide=" << wide
                                          << " returning=" << returning);
          memory_.write32(0x4000, 7);
          memory_.write32(0x4004, 0);
          put(32, 3);
          put(33, wide ? 0 : 7);
          put(34, 7);
          put(35, 0);
          ASSERT_NO_FATAL_FAILURE(issue((buffer ? 64 : 128) + (wide ? 32 : 0) + op, returning));
          EXPECT_EQ(memory_.read32(0x4000), expected[op]);
          EXPECT_EQ(memory_.read32(0x4004), 0u);
          EXPECT_EQ(get(32), returning ? 7u : 3u);
          EXPECT_EQ(get(33), wide ? 0u : 7u);
          EXPECT_EQ(get(34), 7u);
          EXPECT_EQ(get(35), 0u);
        }
}

TEST_P(ScalarAtomicInstruction, WideCarryAndFailedComparePreserveComparisonSource) {
  base();
  memory_.write32(0x4000, 0xffffffff);
  memory_.write32(0x4004, 1);
  put(32, 2);
  put(33, 0);
  ASSERT_NO_FATAL_FAILURE(issue(162, true)); // S_ATOMIC_ADD_X2.
  EXPECT_EQ(memory_.read32(0x4000), 1u);
  EXPECT_EQ(memory_.read32(0x4004), 2u);
  EXPECT_EQ(get(32), 0xffffffffu);
  EXPECT_EQ(get(33), 1u);
  put(32, 42);
  put(33, 0);
  put(34, 1);
  put(35, 3);
  ASSERT_NO_FATAL_FAILURE(issue(161, true)); // Comparison differs in the high DWORD.
  EXPECT_EQ(memory_.read32(0x4000), 1u);
  EXPECT_EQ(memory_.read32(0x4004), 2u);
  EXPECT_EQ(get(32), 1u);
  EXPECT_EQ(get(33), 2u);
  EXPECT_EQ(get(34), 1u);
  EXPECT_EQ(get(35), 3u);
}

TEST_P(ScalarAtomicInstruction, BufferStrideControlsBoundsWithoutChangingAddress) {
  base(2, 8); // 16-byte bound; stride is not an address contribution.
  for (unsigned i = 0; i < 5; ++i)
    memory_.write32(0x4000 + i * 4, 10 + i);
  put(32, 100);
  ASSERT_NO_FATAL_FAILURE(issue(66, true, 12));
  EXPECT_EQ(get(32), 13u);
  EXPECT_EQ(memory_.read32(0x400c), 113u);
  put(32, 100);
  ASSERT_NO_FATAL_FAILURE(issue(66, true, 16));
  EXPECT_EQ(get(32), 0u);
  EXPECT_EQ(memory_.read32(0x4010), 14u);
  base(12);
  put(32, 99);
  put(33, 0);
  ASSERT_NO_FATAL_FAILURE(issue(96, true, 8)); // X2 straddles the descriptor bound.
  EXPECT_EQ(get(32), 0u);
  EXPECT_EQ(get(33), 0u);
  EXPECT_EQ(memory_.read32(0x4008), 12u);
}

TEST_P(ScalarAtomicInstruction, BufferStoresRespectPerDwordBounds) {
  base(3, 4);
  for (unsigned i = 0; i < 4; ++i) {
    memory_.write32(0x4000 + i * 4, 10);
    put(32 + i, 100 + i);
  }
  ASSERT_NO_FATAL_FAILURE(issue(26, true)); // S_BUFFER_STORE_DWORDX4.
  EXPECT_EQ(memory_.read32(0x4000), 100u);
  EXPECT_EQ(memory_.read32(0x4004), 101u);
  EXPECT_EQ(memory_.read32(0x4008), 102u);
  EXPECT_EQ(memory_.read32(0x400c), 10u);
}

TEST_P(ScalarAtomicInstruction, DiscardInvalidatesOnlyAddressedScalarLines) {
  base();
  constexpr uint32_t old = 17, changed = 23;
  for (unsigned lines : {1u, 2u}) {
    l1_.invalidate_all();
    for (unsigned i = 0; i < 3; ++i) {
      ASSERT_EQ(l2_.write(0x4000 + i * 64, reinterpret_cast<const uint8_t *>(&old), 4),
                amdgpu::VmAccessOutcome::Complete);
      uint32_t result = 0;
      ASSERT_EQ(l1_.load(0x4000 + i * 64, 1, &result), amdgpu::VmAccessOutcome::Complete);
      ASSERT_EQ(result, old);
      ASSERT_EQ(l2_.write(0x4000 + i * 64, reinterpret_cast<const uint8_t *>(&changed), 4),
                amdgpu::VmAccessOutcome::Complete);
    }
    ASSERT_NO_FATAL_FAILURE(issue(lines == 1 ? 40 : 41, false, 60));
    for (unsigned i = 0; i < 3; ++i) {
      uint32_t result = 0;
      ASSERT_EQ(l1_.load(0x4000 + i * 64, 1, &result), amdgpu::VmAccessOutcome::Complete);
      EXPECT_EQ(result, i < lines ? changed : old);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Cdna, ScalarAtomicInstruction,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4));
} // namespace
} // namespace rocjitsu
