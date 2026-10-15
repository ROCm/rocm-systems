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
class ScalarBufferStore : public testing::TestWithParam<rj_code_arch_t> {
protected:
  void SetUp() override {
    l2_.set_backing_memory(&memory_);
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = GetParam();
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 102;
    config.vgprs_per_wf = 16;
    config.lds_size_kb = 64;
    cu_ = amdgpu::ComputeUnitCore::create("scalar_buffer_store", config, &memory_, &l2_);
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
  amdgpu::GpuMemory memory_{"scalar_store_memory"};
  amdgpu::L2Cache l2_{"scalar_store_l2"};
  amdgpu::L1ScalarCache l1_{&l2_};
  amdgpu::ScalarMemPipeline pipeline_{&l1_};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_;
  amdgpu::Wavefront *wave_ = nullptr;
};

TEST_P(ScalarBufferStore, BufferStoresRespectPerDwordBounds) {
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

INSTANTIATE_TEST_SUITE_P(Cdna, ScalarBufferStore,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4));
} // namespace
} // namespace rocjitsu
