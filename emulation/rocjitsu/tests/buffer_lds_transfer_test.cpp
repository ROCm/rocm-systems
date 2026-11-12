// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
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
constexpr uint32_t kGfx9DataFormat32 = 4u << 15;

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
    ASSERT_NO_FATAL_FAILURE(
        rocjitsu::test::initialize_compute_unit(cu_, "lds_transfer", config, &memory_, &l2_));
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

TEST_P(BufferLdsTransfer, Cdna2InvalidatePreservesDirtyLocalMemory) {
  if (GetParam() != ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP();
  constexpr uint64_t address = 0x4000;
  memory_.write32(address, 1);
  std::array<uint8_t, amdgpu::L2Cache::LINE_SIZE> line{};
  line[0] = 2;
  ASSERT_EQ(l2_.writeback_line(address, line.data(), 0, 4, amdgpu::Mtype::RW, wave_->process_id()),
            amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(memory_.read32(address), 1u);
  const auto words = cdna2::build_mubuf(cdna2::kBufferInvl2);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder_, words.data()));
  ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wave_).succeeded());
  EXPECT_EQ(memory_.read32(address), 1u);
  ASSERT_EQ(l2_.flush_all(), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(memory_.read32(address), 2u);
}

TEST_P(BufferLdsTransfer, Cdna2StoreLdsUsesM0OffsetAndCapturedActiveLanes) {
  if (GetParam() != ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP();
  const auto missing_lds = cdna2::build_mubuf(cdna2::kBufferStoreLdsDword);
  EXPECT_TRUE(decoder_->decode_window(missing_lds).failed());

  const auto vector_offset =
      cdna2::build_mubuf(cdna2::kBufferStoreLdsDword, {.offen = 1, .lds = 1});
  auto decoded = decoder_->decode_window(vector_offset);
  ASSERT_TRUE(decoded.succeeded());
  ASSERT_TRUE(cu_->execute_instruction(decoded.value().get(), *wave_).failed());
  EXPECT_EQ(wave_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::InvalidInstructionEncoding);
  wave_->clear_instruction_execution_error();
  wave_->set_m0(8);
  wave_->set_exec(7);
  wave_->set_lds_size(20); // Third source word is beyond this workgroup.
  scalar(0, 0x4000);
  scalar(1, 4u << 16);
  scalar(2, 256);
  scalar(3, 1u << 23); // add TID, linear stride of four bytes
  for (unsigned i = 0; i < 3; ++i) {
    wave_->lds().write32(wave_->lds_base() + 12 + 4 * i, 0x11000000u + i);
    memory_.write32(0x4004 + 4 * i, 0xccccccccu);
  }
  ASSERT_NO_FATAL_FAILURE(issue(
      cdna2::build_mubuf(cdna2::kBufferStoreLdsDword, {.offset = 4, .lds = 1, .soffset = 128})));
  cu_->l1_vector().flush_all();
  l2_.flush_all();
  EXPECT_EQ(memory_.read32(0x4004), 0x11000000u);
  EXPECT_EQ(memory_.read32(0x4008), 0x11000001u);
  EXPECT_EQ(memory_.read32(0x400c), 0xccccccccu);
}

TEST(BufferLdsTransferDecode, Gfx950RestrictsLdsBitToDocumentedLoadOpcodes) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  for (auto op : {cdna4::kBufferLoadFormatX, cdna4::kBufferLoadUbyte, cdna4::kBufferLoadSbyte,
                  cdna4::kBufferLoadUshort, cdna4::kBufferLoadSshort, cdna4::kBufferLoadDword,
                  cdna4::kBufferLoadDwordx3, cdna4::kBufferLoadDwordx4}) {
    const auto words = cdna4::build_mubuf(op, {.lds = 1, .soffset = 128});
    EXPECT_FALSE(decode_fails(*decoder, words.data())) << op;
  }
  for (auto op : {cdna4::kBufferLoadDwordx2, cdna4::kBufferLoadFormatXy, cdna4::kBufferLoadUbyteD16,
                  cdna4::kBufferStoreDword, cdna4::kBufferAtomicAdd}) {
    for (uint8_t lds : {0, 1}) {
      const auto words = cdna4::build_mubuf(op, {.lds = lds, .soffset = 128});
      EXPECT_EQ(decode_fails(*decoder, words.data()), lds != 0) << op;
    }
  }
}

TEST(BufferLdsDecode, Gfx10RejectsUnsupportedLdsLoads) {
  for (const auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    auto decoder = Decoder::create(arch);
    SCOPED_TRACE(arch);
    const auto valid = rdna1::build_mubuf(rdna1::kBufferLoadFormatXMubuf, {.lds = 1});
    EXPECT_TRUE(decoder->decode(valid.data()).succeeded());
    for (const auto words : {
             rdna1::build_mubuf(rdna1::kBufferLoadDwordx3Mubuf, {.lds = 1}),
             rdna1::build_mubuf(rdna1::kBufferLoadFormatD16XMubuf, {.lds = 1, .opm = 1}),
             rdna1::build_mubuf(rdna1::kBufferLoadDwordMubuf, {.lds = 1, .tfe = 1}),
         }) {
      EXPECT_TRUE(decoder->decode(words.data()).failed());
    }
  }
}

TEST_P(BufferLdsTransfer, BufferNarrowAndX3TransfersUseDwordLaneLayout) {
  for (unsigned opcode :
       {cdna4::kBufferLoadSbyteMubuf,
        cdna4::kBufferLoadDwordx3Mubuf}) { // BUFFER_LOAD_SBYTE / BUFFER_LOAD_DWORDX3.
    if (GetParam() != ROCJITSU_CODE_ARCH_CDNA4 && opcode == cdna4::kBufferLoadDwordx3Mubuf) {
      const auto words = cdna4::build_mubuf(opcode, {.lds = 1});
      EXPECT_TRUE(decoder_->decode(words.data()).failed());
      continue;
    }
    clear_lds();
    wave_->set_exec(0x25);
    wave_->set_m0(0xfffc0040);
    scalar(0, 0x4000);
    scalar(1, 0);
    scalar(2, 4096);
    scalar(3, kGfx9DataFormat32);
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

TEST(BufferLdsDecode, Mi100RequiresOneOfSixLoadsOrTheDedicatedStore) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA1);
  ASSERT_NE(decoder, nullptr);
  // LLVM gfx908 bytes for BUFFER_LOAD_FORMAT_X and BUFFER_STORE_LDS_DWORD
  // with s[4:7] and s0. Both require TFE clear.
  for (uint32_t lo : {0xe0010000u, 0xe0f50000u}) {
    std::array<uint32_t, 2> words{lo, 0x00010000u};
    EXPECT_TRUE(decoder->decode(words.data()).succeeded());
    words[1] |= 1u << 23;
    EXPECT_TRUE(decoder->decode(words.data()).failed());
  }
  for (uint32_t lo : {0xe0f40000u, 0xe0550000u, 0xe0050000u, 0xe0810000u}) {
    // STORE_LDS_DWORD without LDS, then LDS forms of DWORDX2, FORMAT_XY, UBYTE_D16.
    std::array<uint32_t, 2> words{lo, 0x00010000u};
    EXPECT_TRUE(decoder->decode(words.data()).failed());
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
      if (GetParam() == ROCJITSU_CODE_ARCH_CDNA2) {
        EXPECT_EQ(wave_->trapsts() & (1u << 8), 0u);
      }
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

TEST_P(BufferLdsTransfer, UnboundRawLoadsStoresAndAtomicsDoNotAccessMemory) {
  constexpr uint32_t sentinel = 0x76543210;
  // A nonzero mapped base and permissive bounds distinguish unbound behavior
  // from null-pointer handling and the ordinary range check.
  for (bool bound : {false, true}) {
    SCOPED_TRACE(bound);
    for (auto opcode : {cdna4::kBufferLoadDwordMubuf, cdna4::kBufferStoreDwordMubuf,
                        cdna4::kBufferAtomicAddMubuf}) {
      SCOPED_TRACE(opcode);
      wave_->set_exec(1);
      scalar(0, 0x4000);
      scalar(1, 0);
      scalar(2, ~0u);
      scalar(3, bound ? kGfx9DataFormat32 : 0);
      memory_.write32(0x4000, sentinel);
      cu_->l1_vector().invalidate_all();
      l2_.invalidate_all();
      vector(8, 0, 3);
      const auto words =
          cdna4::build_mubuf(opcode, {.sc0 = 1, .vdata = 8, .srsrc = 0, .soffset = 128});
      ASSERT_NO_FATAL_FAILURE(issue(words));
      const bool store = opcode == cdna4::kBufferStoreDwordMubuf;
      const bool atomic = opcode == cdna4::kBufferAtomicAddMubuf;
      EXPECT_EQ(cu_->read_vgpr(wave_->vgpr_alloc().base + 8, 0),
                store ? 3u : (bound ? sentinel : 0u));
      EXPECT_EQ(memory_.read32(0x4000), bound && store    ? 3u
                                        : bound && atomic ? sentinel + 3
                                                          : sentinel);
    }
  }
}

TEST_P(BufferLdsTransfer, UnboundRawLdsLoadsWriteZerosWithoutReadingMemory) {
  constexpr uint32_t sentinel = 0x76543210;
  for (bool bound : {false, true}) {
    SCOPED_TRACE(bound);
    clear_lds();
    wave_->set_exec(1);
    wave_->set_m0(64);
    scalar(0, 0x4000);
    scalar(1, 0);
    scalar(2, ~0u);
    scalar(3, bound ? kGfx9DataFormat32 : 0);
    memory_.write32(0x4000, sentinel);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    vector(8, 0, 3);
    const auto words = cdna4::build_mubuf(cdna4::kBufferLoadDwordMubuf,
                                          {.lds = 1, .vdata = 8, .srsrc = 0, .soffset = 128});
    ASSERT_NO_FATAL_FAILURE(issue(words));
    EXPECT_EQ(wave_->lds().read32(128 + 64), bound ? sentinel : 0u);
    EXPECT_EQ(wave_->lds().read32(128 + 68), 0xccccccccu);
    EXPECT_EQ(cu_->read_vgpr(wave_->vgpr_alloc().base + 8, 0), 3u);
  }
}

INSTANTIATE_TEST_SUITE_P(Cdna, BufferLdsTransfer,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4));
} // namespace
} // namespace rocjitsu
