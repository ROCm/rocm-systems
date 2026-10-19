// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>

namespace rocjitsu {
namespace {
class BufferLdsTransfer : public testing::TestWithParam<rj_code_arch_t> {
protected:
  void SetUp() override {
    l2_.set_backing_memory(&memory_);
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = GetParam();
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 102;
    config.vgprs_per_wf = 16;
    config.lds_size_kb = 64;
    cu_ = amdgpu::ComputeUnitCore::create("lds_transfer", config, &memory_, &l2_);
    decoder_ = Decoder::create(GetParam());
    wave_ = cu_->dispatch_wf(0, 0, 102, 16);
    ASSERT_NE(wave_, nullptr);
    wave_->set_lds_base(128);
    wave_->set_lds_size(4096);
    pipeline_ = std::make_unique<amdgpu::GlobalMemPipeline>(&cu_->l1_vector(), &l2_);
  }
  void TearDown() override {
    if (wave_)
      wave_->halt();
  }
  void scalar(unsigned reg, uint32_t value) {
    cu_->write_sgpr(wave_->sgpr_alloc().base + reg, value);
  }
  void vector(unsigned reg, unsigned lane, uint32_t value) {
    cu_->write_vgpr(wave_->vgpr_alloc().base + reg, lane, value);
  }
  void issue(std::array<uint32_t, 2> words) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder_, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wave_).succeeded());
    auto *state = inst->data_as<amdgpu::VectorMemState>();
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->wait_counter_type, amdgpu::WaitCounterType::VMCNT);
    ASSERT_EQ(pipeline_->issue(inst.release(), *wave_), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(wave_->wait_counters().vmcnt, 0u);
    EXPECT_EQ(wave_->wait_counters().lgkmcnt, 0u);
  }
  void clear_lds() {
    for (uint32_t a = 0; a < 4608; a += 4)
      wave_->lds().write32(a, 0xcccccccc);
  }
  amdgpu::GpuMemory memory_{"lds_transfer_memory"};
  amdgpu::L2Cache l2_{"lds_transfer_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_;
  std::unique_ptr<amdgpu::GlobalMemPipeline> pipeline_;
  amdgpu::Wavefront *wave_ = nullptr;
};

TEST_P(BufferLdsTransfer, BufferNarrowAndX3TransfersUseDwordLaneLayout) {
  for (unsigned opcode :
       {cdna4::kBufferLoadSbyteMubuf,
        cdna4::kBufferLoadDwordx3Mubuf}) { // BUFFER_LOAD_SBYTE / BUFFER_LOAD_DWORDX3.
    clear_lds();
    wave_->set_exec(0x25);
    wave_->set_m0(0xfffc0040);
    scalar(0, 0x4000);
    scalar(1, 0);
    scalar(2, 4096);
    scalar(3, 0);
    for (unsigned lane = 0; lane < 64; ++lane) {
      vector(2, lane, lane * 16);
      for (unsigned component = 0; component < 4; ++component)
        memory_.write32(0x4000 + lane * 16 + component * 4, 0x76548081 + component);
    }
    auto words = cdna4::build_mubuf(
        opcode, {.offen = 1, .lds = 1, .vaddr = 2, .vdata = 8, .srsrc = 0, .soffset = 128});
    ASSERT_NO_FATAL_FAILURE(issue(words));
    for (unsigned lane = 0; lane < 64; ++lane) {
      const uint32_t stride = opcode == cdna4::kBufferLoadSbyteMubuf ? 4 : 16;
      EXPECT_EQ(wave_->lds().read32(128 + 64 + lane * stride),
                0x25u & (uint64_t{1} << lane)
                    ? (opcode == cdna4::kBufferLoadSbyteMubuf ? 0xffffff81u : 0x76548081u)
                    : 0xccccccccu);
      if (opcode == cdna4::kBufferLoadDwordx3Mubuf) {
        EXPECT_EQ(wave_->lds().read32(128 + 64 + lane * 16 + 12), 0xccccccccu);
      }
    }
  }
}

TEST_P(BufferLdsTransfer, RawAndFormatBufferLoadsShareM0WidthAndAllocationBounds) {
  for (auto opcode : {cdna4::kBufferLoadDwordMubuf, cdna4::kBufferLoadFormatXMubuf})
    for (uint32_t m0 : {0x10040u, 0xfffc0042u}) {
      SCOPED_TRACE(testing::Message() << "opcode=" << opcode << " m0=" << m0);
      clear_lds();
      wave_->set_exec(7);
      wave_->set_m0(m0);
      wave_->set_lds_size(74);
      scalar(0, 0x4000);
      scalar(1, 0);
      scalar(2, 4096);
      scalar(3, 4 | (7 << 12) | (4 << 15)); // FLOAT32, select X.
      for (unsigned lane = 0; lane < 3; ++lane) {
        vector(2, lane, lane * 4);
        vector(8, lane, 0x11111111);
        memory_.write32(0x4000 + lane * 4, 0x3f800000 + lane);
      }
      auto words = cdna4::build_mubuf(
          opcode, {.offen = 1, .lds = 1, .vaddr = 2, .vdata = 8, .srsrc = 0, .soffset = 128});
      ASSERT_NO_FATAL_FAILURE(issue(words));
      const unsigned offset = m0 & (GetParam() == ROCJITSU_CODE_ARCH_CDNA4 ? 0x3ffff : 0xffff);
      for (unsigned lane = 0; lane < 3; ++lane) {
        if (offset < 4096) {
          EXPECT_EQ(wave_->lds().read32(128 + offset + lane * 4),
                    offset + lane * 4 + 4 <= 74 ? 0x3f800000u + lane : 0xccccccccu);
        }
        EXPECT_EQ(cu_->read_vgpr(wave_->vgpr_alloc().base + 8, lane), 0x11111111u);
      }
      if (offset >= 4096) {
        EXPECT_EQ(wave_->lds().read32(128 + 64), 0xccccccccu);
      }
    }
}

INSTANTIATE_TEST_SUITE_P(Cdna, BufferLdsTransfer,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4));
} // namespace
} // namespace rocjitsu
