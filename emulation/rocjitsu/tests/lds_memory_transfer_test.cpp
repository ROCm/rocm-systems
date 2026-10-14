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
#include <string_view>

namespace rocjitsu {
namespace {
class LdsMemoryTransfer : public testing::TestWithParam<rj_code_arch_t> {
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
    wave_->set_scratch_base(0x8000);
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
  std::array<uint32_t, 2> flat(unsigned opcode, unsigned segment, int offset) {
    auto words = cdna4::build_flat(opcode, {.offset = uint16_t(offset & 0xfff),
                                            .lds = uint8_t(segment == 1),
                                            .seg = uint8_t(segment),
                                            .addr = 2,
                                            .saddr = 0,
                                            .vdst = 8});
    words[0] |= uint32_t(offset & 0x1000); // Signed offset bit absent from the generic builder.
    return words;
  }
  void issue(std::array<uint32_t, 2> words, bool scratch = false) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder_, words.data()));
    ASSERT_NE(inst, nullptr);
    if (scratch) {
      EXPECT_TRUE(std::string_view(inst->mnemonic()).starts_with("scratch_load_lds_"));
    }
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

TEST_P(LdsMemoryTransfer, GlobalAndScratchExtendNarrowLoadsAndRespectExec) {
  if (GetParam() < ROCJITSU_CODE_ARCH_CDNA3)
    GTEST_SKIP() << "Dedicated LDS opcodes start at CDNA3";
  constexpr std::array<uint32_t, 5> expected{0x81, 0xffffff81, 0x8081, 0xffff8081, 0x76548081};
  for (unsigned segment : {1u, 2u})
    for (int offset : {-8, 0, 12})
      for (unsigned op = 0; op < expected.size(); ++op)
        for (uint64_t mask : {uint64_t{0}, uint64_t{0x8000000000000025}}) {
          SCOPED_TRACE(testing::Message() << "segment=" << segment << " offset=" << offset
                                          << " op=" << op << " mask=" << mask);
          clear_lds();
          wave_->set_exec(mask);
          wave_->set_m0(0xfffc0040); // High bits are excluded from the LDS offset.
          scalar(0, segment == 1 ? 8 : 0x4000);
          scalar(1, 0);
          for (unsigned lane = 0; lane < 64; ++lane) {
            const uint32_t vaddr = 32 + lane * 4;
            vector(2, lane, vaddr);
            vector(8, lane, 0x12345678);
            const uint64_t private_offset = 8 + vaddr + offset;
            const uint64_t address = segment == 1 ? 0x8000 + (private_offset / 4) * 256 + lane * 4
                                                  : 0x4000 + vaddr + offset;
            memory_.write32(address, 0x76548081);
          }
          l2_.invalidate_all();
          cu_->l1_vector().invalidate_all();
          ASSERT_NO_FATAL_FAILURE(issue(flat(38 + op, segment, offset), segment == 1));
          for (unsigned lane = 0; lane < 64; ++lane) {
            EXPECT_EQ(wave_->lds().read32(128 + 64 + offset + lane * 4),
                      mask & (uint64_t{1} << lane) ? expected[op] : 0xccccccccu);
            EXPECT_EQ(cu_->read_vgpr(wave_->vgpr_alloc().base + 8, lane), 0x12345678u);
          }
          EXPECT_EQ(wave_->lds().read32(128 + 64 + offset - 4), 0xccccccccu);
          EXPECT_EQ(wave_->lds().read32(128 + 64 + offset + 256), 0xccccccccu);
        }
}

TEST_P(LdsMemoryTransfer, WideGlobalLoadsLeaveFourthDwordUntouchedForX3) {
  if (GetParam() < ROCJITSU_CODE_ARCH_CDNA3)
    GTEST_SKIP() << "Dedicated LDS opcodes start at CDNA3";
  if (GetParam() != ROCJITSU_CODE_ARCH_CDNA4)
    GTEST_SKIP() << "Wide global-to-LDS forms start with CDNA4";
  for (unsigned words : {3u, 4u}) {
    clear_lds();
    wave_->set_exec(0x25);
    wave_->set_m0(64);
    scalar(0, 0x4000);
    scalar(1, 0);
    for (unsigned lane = 0; lane < 64; ++lane) {
      vector(2, lane, lane * 16);
      for (unsigned component = 0; component < 4; ++component)
        memory_.write32(0x4000 + lane * 16 + component * 4, 0x12340000 + lane * 16 + component);
    }
    ASSERT_NO_FATAL_FAILURE(issue(flat(words == 3 ? 126 : 125, 2, 0)));
    for (unsigned lane = 0; lane < 64; ++lane)
      for (unsigned component = 0; component < 4; ++component)
        EXPECT_EQ(wave_->lds().read32(128 + 64 + lane * 16 + component * 4),
                  (0x25u & (uint64_t{1} << lane)) && component < words
                      ? 0x12340000 + lane * 16 + component
                      : 0xccccccccu);
  }
}

TEST_P(LdsMemoryTransfer, AllocationBoundsDropWholeLanesAndInvalidSegmentsReject) {
  if (GetParam() < ROCJITSU_CODE_ARCH_CDNA3)
    GTEST_SKIP() << "Dedicated LDS opcodes start at CDNA3";
  clear_lds();
  wave_->set_exec(7);
  wave_->set_lds_size(72);
  wave_->set_m0(64);
  scalar(0, 0x4000);
  scalar(1, 0);
  for (unsigned lane = 0; lane < 3; ++lane) {
    vector(2, lane, lane * 4);
    memory_.write32(0x4000 + lane * 4, 10 + lane);
  }
  ASSERT_NO_FATAL_FAILURE(issue(flat(42, 2, 0)));
  EXPECT_EQ(wave_->lds().read32(128 + 64), 10u);
  EXPECT_EQ(wave_->lds().read32(128 + 68), 11u);
  EXPECT_EQ(wave_->lds().read32(128 + 72), 0xccccccccu);
  for (unsigned segment : {0u, 3u}) {
    auto words = flat(42, segment, 0);
    EXPECT_TRUE(decode_fails(*decoder_, words.data()));
  }
  if (GetParam() == ROCJITSU_CODE_ARCH_CDNA4) {
    auto words = flat(126, 1, 0);
    EXPECT_TRUE(decode_fails(*decoder_, words.data()));
  }
}

TEST_P(LdsMemoryTransfer, BufferNarrowAndX3TransfersUseDwordLaneLayout) {
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

TEST_P(LdsMemoryTransfer, RawAndFormatBufferLoadsShareM0WidthAndAllocationBounds) {
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

TEST_P(LdsMemoryTransfer, OlderFlatLdsBitRedirectsGlobalAndScratchLoads) {
  if (GetParam() >= ROCJITSU_CODE_ARCH_CDNA3)
    GTEST_SKIP() << "LDS bit forms are CDNA1/2";
  for (unsigned segment : {1u, 2u}) {
    clear_lds();
    wave_->set_exec(3);
    wave_->set_m0(0xffff0040); // CDNA1/2 use only the low 16-bit byte offset.
    scalar(0, segment == 1 ? 8 : 0x4000);
    scalar(1, 0);
    for (unsigned lane = 0; lane < 2; ++lane) {
      vector(2, lane, lane * 4);
      vector(8, lane, 0x11111111);
      const uint64_t address =
          segment == 1 ? 0x8000 + (8 + lane * 4 + 4) / 4 * 256 + lane * 4 : 0x4000 + lane * 4 + 4;
      memory_.write32(address, 0x55667788 + lane);
    }
    auto words = flat(20, segment, 4); // Pre-gfx940 GLOBAL/SCRATCH_LOAD_DWORD.
    words[0] |= 1u << 13;
    ASSERT_NO_FATAL_FAILURE(issue(words));
    for (unsigned lane = 0; lane < 2; ++lane) {
      EXPECT_EQ(wave_->lds().read32(128 + 64 + 4 + lane * 4), 0x55667788u + lane);
      EXPECT_EQ(cu_->read_vgpr(wave_->vgpr_alloc().base + 8, lane), 0x11111111u);
    }
  }
}

TEST_P(LdsMemoryTransfer, OlderLdsBitRejectsFlatAndUnsupportedWidths) {
  if (GetParam() >= ROCJITSU_CODE_ARCH_CDNA3)
    GTEST_SKIP() << "LDS bit forms are CDNA1/2";
  for (unsigned op : {16u, 17u, 18u, 19u, 20u, 21u, 22u, 23u, 36u})
    for (unsigned segment : {0u, 1u, 2u}) {
      SCOPED_TRACE(testing::Message() << "opcode=" << op << " segment=" << segment);
      auto words = flat(op, segment, 0);
      words[0] |= 1u << 13;
      EXPECT_EQ(decode_fails(*decoder_, words.data()), segment == 0 || op > 20);
    }
}

INSTANTIATE_TEST_SUITE_P(Cdna, LdsMemoryTransfer,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4));
} // namespace
} // namespace rocjitsu
