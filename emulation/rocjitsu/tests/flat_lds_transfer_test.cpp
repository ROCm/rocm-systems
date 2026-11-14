// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "decode_test_util.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <array>
#include <gtest/gtest.h>
#include <memory>

namespace rocjitsu {
namespace {
class FlatLdsTransfer : public testing::TestWithParam<std::pair<rj_code_arch_t, unsigned>> {
protected:
  bool modern() const {
    return GetParam().first == ROCJITSU_CODE_ARCH_CDNA3 ||
           GetParam().first == ROCJITSU_CODE_ARCH_CDNA4;
  }
  bool rdna() const {
    return GetParam().first == ROCJITSU_CODE_ARCH_RDNA1 ||
           GetParam().first == ROCJITSU_CODE_ARCH_RDNA2;
  }
  unsigned vector_saddr() const { return rdna() ? 125 : 127; }
  void SetUp() override {
    l2.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = GetParam().first;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 102;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    ASSERT_NO_FATAL_FAILURE(
        rocjitsu::test::initialize_compute_unit(cu, "flat_lds", cfg, &memory, &l2));
    wave = cu->dispatch_wf(0, 0, 102, 16, GetParam().second);
    ASSERT_NE(wave, nullptr);
    wave->set_lds_base(128);
    wave->set_lds_size(1024);
    wave->set_scratch_base(0x10000);
    wave->set_scratch_lane_size(64);
    decoder = Decoder::create(GetParam().first);
    pipeline = std::make_unique<amdgpu::GlobalMemPipeline>(&cu->l1_vector(), &l2);
  }
  void TearDown() override {
    if (wave)
      wave->halt();
  }
  void scalar(unsigned reg, uint32_t value) {
    cu->write_sgpr(wave->sgpr_alloc().base + reg, value);
  }
  void vector(unsigned reg, unsigned lane, uint32_t value) {
    cu->write_vgpr(wave->vgpr_alloc().base + reg, lane, value);
  }
  std::array<uint32_t, 2> encoding(unsigned kind, unsigned segment, bool vaddr, unsigned saddr,
                                   int offset = 0, bool lds = true) {
    const unsigned op = (modern() && lds ? 38 : rdna() ? 8 : 16) + kind;
    return encoding_for_opcode(op, segment, vaddr, saddr, offset, lds);
  }
  std::array<uint32_t, 2> encoding_for_opcode(unsigned op, unsigned segment, bool vaddr,
                                              unsigned saddr, int offset = 0, bool lds = true,
                                              unsigned vdst = 8) {
    const unsigned bit13 = modern() ? vaddr : lds;
    return {0xdc000000u | (op << 18) | (segment << 14) | (bit13 << 13) |
                (static_cast<unsigned>(offset) & (rdna() ? 0xfff : 0x1fff)),
            2u | (saddr << 16) | (vdst << 24)};
  }
  std::array<uint32_t, 2> global_wide_lds_encoding(unsigned count, unsigned saddr, int offset,
                                                   unsigned vdst = 8) {
    return encoding_for_opcode(count == 3 ? cdna4::kGlobalLoadLdsDwordx3Flat
                                          : cdna4::kGlobalLoadLdsDwordx4Flat,
                               2, false, saddr, offset, true, vdst);
  }
  void clear() {
    for (unsigned a = 0; a < 1408; a += 4)
      wave->lds().write32(a, 0xcccccccc);
    for (unsigned lane = 0; lane < wave->wf_size(); ++lane)
      vector(8, lane, 0xdeadbeef);
  }
  void issue(std::array<uint32_t, 2> words, bool lds = true) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    if (lds) {
      EXPECT_EQ(inst->num_dst_operands(), 0);
      InstDefUse effects(*inst, 64);
      EXPECT_TRUE(effects.uses.contains({RegClass::M0, 0, 1}));
      EXPECT_FALSE(effects.uses.contains({RegClass::VGPR, 255, 1}));
    }
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
    auto *state = inst->data_as<amdgpu::VectorMemState>();
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->lds_dst, lds);
    EXPECT_EQ(state->wait_counter_type, amdgpu::WaitCounterType::VMCNT);
    ASSERT_EQ(pipeline->issue(inst.release(), *wave), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(wave->wait_counters().vmcnt, 0);
    EXPECT_EQ(wave->wait_counters().lgkmcnt, 0);
  }
  amdgpu::GpuMemory memory{"flat_memory"};
  amdgpu::L2Cache l2{"flat_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  std::unique_ptr<amdgpu::GlobalMemPipeline> pipeline;
  amdgpu::Wavefront *wave = nullptr;
};

TEST_P(FlatLdsTransfer, ScratchWidthsAndAddressModesPreserveInactiveLanesAndVgprs) {
  for (unsigned kind = 0; kind < 5; ++kind)
    for (unsigned mode = 0;
         mode < (modern() || GetParam().first == ROCJITSU_CODE_ARCH_RDNA2 ? 4u : 2u); ++mode) {
      if (!modern() && mode == 2)
        continue; // Legacy encodings have no combined form.
      SCOPED_TRACE(testing::Message() << "kind=" << kind << " mode=" << mode);
      clear();
      wave->set_exec(0x25);
      wave->set_m0(0xfffc0043); // Ignore high bits and align M0 down to a dword.
      const bool has_vector = mode == 0 || mode == 2;
      const bool has_scalar = mode == 1 || mode == 2;
      scalar(4, 8);
      for (unsigned lane = 0; lane < wave->wf_size(); ++lane) {
        const unsigned subword = kind < 2 ? lane % 4 : kind < 4 ? (lane % 2) * 2 : 0;
        vector(2, lane, has_vector ? 4 + subword : 0xffffffff);
        for (unsigned slot = 0; slot < 8; ++slot)
          memory.write32(0x10000 + slot * wave->wf_size() * 4 + lane * 4,
                         0x89878583u + lane * 0x01010101u + slot * 0x02020202u);
      }
      ASSERT_NO_FATAL_FAILURE(issue(encoding(kind, 1, has_vector,
                                             has_scalar   ? 4
                                             : has_vector ? vector_saddr()
                                                          : 127,
                                             4)));
      for (unsigned lane = 0; lane < wave->wf_size(); ++lane) {
        const unsigned subword = has_vector ? (kind < 2   ? lane % 4
                                               : kind < 4 ? (lane % 2) * 2
                                                          : 0)
                                            : 0;
        const unsigned slot = (has_vector ? 1 : 0) + (has_scalar ? 2 : 0) + 1;
        uint32_t expected =
            (0x89878583u + lane * 0x01010101u + slot * 0x02020202u) >> (subword * 8);
        if (kind == 0)
          expected &= 0xff;
        if (kind == 1)
          expected = static_cast<int8_t>(expected);
        if (kind == 2)
          expected &= 0xffff;
        if (kind == 3)
          expected = static_cast<int16_t>(expected);
        EXPECT_EQ(wave->lds().read32(128 + 68 + lane * 4),
                  (0x25 & (uint64_t{1} << lane)) ? expected : 0xcccccccc);
        EXPECT_EQ(cu->read_vgpr(wave->vgpr_alloc().base + 8, lane), 0xdeadbeef);
      }
    }
}

TEST_P(FlatLdsTransfer, GlobalLoadsToLdsAdvertiseVmemObligation) {
  if (!modern())
    GTEST_SKIP() << "Tests specific to modern architectures.";
  for (unsigned kind = 0; kind < 5; ++kind) {
    SCOPED_TRACE(testing::Message() << "kind=" << kind);
    const auto words = encoding(kind, 2, false, vector_saddr());
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    const auto *issue_info = inst->amdgpu_memory_issue_info();
    ASSERT_NE(issue_info, nullptr);
    const auto obligations = issue_info->counter_obligations();
    ASSERT_EQ(obligations.size(), 1u);
    EXPECT_EQ(obligations[0].wait_counter_type(), amdgpu::WaitCounterType::VMCNT);
    EXPECT_EQ(obligations[0].completion_class(), amdgpu::MemoryCompletionClass::VMEM);
  }
}

TEST_P(FlatLdsTransfer, GlobalLoadsToLdsPreserveInactiveLanesAndVgprs) {
  if (!modern())
    GTEST_SKIP() << "Tests specific to modern architectures.";
  const std::array<uint32_t, 5> expected{0x81, 0xffffff81, 0x8081, 0xffff8081, 0x76548081};
  const int base_offset = 64;
  const uint64_t exec_mask = 0x25;
  for (unsigned kind = 0; kind < 5; ++kind) {
    for (bool scalar_base : {false, true}) {
      for (int offset : {-4, 0, 4}) {
        SCOPED_TRACE(testing::Message()
                     << "kind=" << kind << " scalar_base=" << scalar_base << " offset=" << offset);
        clear();
        wave->set_exec(exec_mask);
        wave->set_m0(base_offset);
        scalar(4, 0x20000);
        scalar(5, 0);
        for (unsigned lane = 0; lane < wave->wf_size(); ++lane) {
          const int source_slot = base_offset + static_cast<int>(lane) * 4;
          const int address_before_offset = source_slot - offset;

          vector(2, lane,
                 scalar_base ? static_cast<uint32_t>(address_before_offset)
                             : 0x20000u + static_cast<uint32_t>(address_before_offset));
          vector(3, lane, 0);
          memory.write32(0x20000u + static_cast<uint32_t>(source_slot), 0x76548081u + lane);
        }

        ASSERT_NO_FATAL_FAILURE(
            issue(encoding(kind, 2, false, scalar_base ? 4 : vector_saddr(), offset)));
        for (unsigned lane = 0; lane < wave->wf_size(); ++lane) {
          const unsigned destination =
              static_cast<unsigned>(128 + 64 + offset + static_cast<int>(lane) * 4);
          const bool active = (exec_mask & (uint64_t{1} << lane)) != 0;

          EXPECT_EQ(wave->lds().read32(destination), active ? expected[kind] + lane : 0xccccccccu)
              << "lane=" << lane;

          EXPECT_EQ(cu->read_vgpr(wave->vgpr_alloc().base + 8, lane), 0xdeadbeefu)
              << "lane=" << lane;
        }
      }
    }
  }
}

TEST_P(FlatLdsTransfer, Cdna3RejectsCdna4WideGlobalLoadEncodings) {
  if (GetParam().first != ROCJITSU_CODE_ARCH_CDNA3)
    GTEST_SKIP() << "Checks the CDNA3 decoder boundary.";
  for (unsigned dwords : {3u, 4u}) {
    auto words = global_wide_lds_encoding(dwords, 127, 0);
    EXPECT_TRUE(decode_fails(*decoder, words.data()));
  }
}

TEST_P(FlatLdsTransfer, CoincidentDsWritesIgnoreData1) {
  wave->set_lds_size(512);
  wave->set_exec(1);
  vector(0, 0, 0);
  vector(2, 0, 0x12345678);
  vector(3, 0, 0x90abcdef);
  auto &board = wave->ensure_memory_wait_scoreboard();
  unsigned hazards = 0;
  const auto ds_words = [&](uint16_t opcode, uint8_t offset, uint8_t data1) {
    return rdna()
               ? rdna1::build_ds(
                     opcode, {.offset1 = offset, .addr = 0, .data0 = 2, .data1 = data1, .vdst = 10})
               : cdna3::build_ds(
                     opcode,
                     {.offset1 = offset, .addr = 0, .data0 = 2, .data1 = data1, .vdst = 10});
  };
  for (uint16_t opcode :
       {cdna3::kDsWrite2B32, cdna3::kDsWrite2st64B32, cdna3::kDsWrite2B64, cdna3::kDsWrite2st64B64,
        cdna3::kDsWrxchg2RtnB32, cdna3::kDsWrxchg2st64RtnB32, cdna3::kDsWrxchg2RtnB64,
        cdna3::kDsWrxchg2st64RtnB64}) {
    SCOPED_TRACE(opcode);
    for (uint8_t offset : {0, 1}) {
      auto words = ds_words(opcode, offset, 8);
      std::unique_ptr<Instruction> store(decode_valid(*decoder, words.data()));
      ASSERT_NE(store, nullptr);
      board.clear();
      const auto load = amdgpu::WaitCounterKind::Load;
      board.add({board.issue(load), 0x100, 1, {RegClass::VGPR, 8, 1}, load, 0xf});
      hazards = 0;
      board.bind(8, &hazards, [](void *context, const amdgpu::MemoryWaitScoreboard::Hazard &) {
        ++*static_cast<unsigned *>(context);
      });
      board.check_instruction(*store, *wave);
      EXPECT_EQ(hazards, offset != 0 ? 1u : 0u);
    }
    board.clear();
    // A coincident store must also execute when DATA1 is outside the VGPR allocation.
    auto words = ds_words(opcode, 0, 254);
    std::unique_ptr<Instruction> store(decode_valid(*decoder, words.data()));
    ASSERT_NE(store, nullptr);
    ASSERT_TRUE(cu->execute_instruction(store.get(), *wave).succeeded());
    amdgpu::LocalMemPipeline local;
    ASSERT_EQ(local.issue(store.release(), *wave), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(wave->lds().read32(wave->lds_base()), 0x12345678u);
  }
}

TEST_P(FlatLdsTransfer, Cdna3DualAddressAccessesRespectTheWorkgroupAllocation) {
  if (GetParam().first != ROCJITSU_CODE_ARCH_CDNA3)
    GTEST_SKIP();
  ASSERT_EQ(cu->allocate_lds(1), 0u);
  const uint32_t allocation = cu->allocate_lds(1);
  ASSERT_EQ(allocation, 512u);
  wave->set_lds_base(allocation);
  wave->set_lds_size(512);
  wave->set_m0(0xffffffffu);
  wave->set_exec(1);
  vector(0, 0, 508);
  vector(1, 0, 0x11223344);
  vector(2, 0, 0xaabbccdd);
  const uint32_t last = allocation + 508;
  wave->lds().write32(last + 4, 0xdeadbeef);
  amdgpu::LocalMemPipeline local;
  auto issue_ds = [&](uint16_t op, uint8_t offset1) {
    const auto words =
        cdna3::build_ds(op, {.offset1 = offset1, .addr = 0, .data0 = 1, .data1 = 2, .vdst = 6});
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
    ASSERT_EQ(local.issue(inst.release(), *wave), amdgpu::VmAccessOutcome::Complete);
  };
  ASSERT_NO_FATAL_FAILURE(issue_ds(cdna3::kDsWrite2B32, 0));
  EXPECT_EQ(wave->lds().read32(last), 0x11223344u);
  ASSERT_NO_FATAL_FAILURE(issue_ds(cdna3::kDsWrite2B32, 1));
  EXPECT_EQ(wave->lds().read32(last + 4), 0xdeadbeefu);
  ASSERT_NO_FATAL_FAILURE(issue_ds(cdna3::kDsRead2B32, 1));
  EXPECT_EQ(cu->read_vgpr(wave->vgpr_alloc().base + 6, 0), 0x11223344u);
  EXPECT_EQ(cu->read_vgpr(wave->vgpr_alloc().base + 7, 0), 0u);
  ASSERT_NO_FATAL_FAILURE(issue_ds(cdna3::kDsWrxchg2RtnB32, 0));
  EXPECT_EQ(wave->lds().read32(last), 0x11223344u);
  EXPECT_EQ(cu->read_vgpr(wave->vgpr_alloc().base + 7, 0), 0x11223344u);
}

TEST_P(FlatLdsTransfer, LegacyGlobalWidthsAndScalarOrVectorBase) {
  if (modern())
    GTEST_SKIP() << "Modern GLOBAL LDS transfers use separate opcodes.";
  const std::array<uint32_t, 5> expected{0x81, 0xffffff81, 0x8081, 0xffff8081, 0x76548081};
  for (unsigned kind = 0; kind < 5; ++kind)
    for (bool scalar_base : {false, true}) {
      clear();
      wave->set_exec(0x25);
      wave->set_m0(68);
      scalar(4, 0x20000);
      scalar(5, 0);
      for (unsigned lane = 0; lane < wave->wf_size(); ++lane) {
        vector(2, lane, (scalar_base ? 0 : 0x20000) + lane * 4 + 4);
        vector(3, lane, 0);
        memory.write32(0x20000 + lane * 4, 0x76548081);
      }
      ASSERT_NO_FATAL_FAILURE(issue(encoding(kind, 2, true, scalar_base ? 4 : vector_saddr(), -4)));
      for (unsigned lane = 0; lane < wave->wf_size(); ++lane)
        EXPECT_EQ(wave->lds().read32(128 + 64 + lane * 4),
                  (0x25 & (uint64_t{1} << lane)) ? expected[kind] : 0xcccccccc);
    }
}

TEST_P(FlatLdsTransfer, DestinationBoundsAndNegativeOffsetsDoNotEscapeAllocation) {
  for (int offset : {-4, 4}) {
    clear();
    wave->set_exec(7);
    wave->set_lds_size(12);
    wave->set_m0(0);
    scalar(4, 8);
    for (unsigned lane = 0; lane < wave->wf_size(); ++lane)
      memory.write32(0x10000 + (8 + offset) / 4 * wave->wf_size() * 4 + lane * 4, 0x12345678);
    ASSERT_NO_FATAL_FAILURE(issue(encoding(4, 1, false, 4, offset)));
    for (unsigned a = 0; a < 160; a += 4) {
      const bool written = offset < 0 ? a == 128 || a == 132 : a == 132 || a == 136;
      EXPECT_EQ(wave->lds().read32(a), written ? 0x12345678 : 0xcccccccc) << a;
    }
  }
}

TEST_P(FlatLdsTransfer, EmptyExecDoesNotTouchLdsOrVgprs) {
  clear();
  wave->set_exec(0);
  wave->set_m0(0);
  ASSERT_NO_FATAL_FAILURE(issue(encoding(4, 1, true, 127)));
  EXPECT_EQ(wave->lds().read32(128), 0xcccccccc);
}

TEST_P(FlatLdsTransfer, Cdna2ScratchChecksLogicalBytesBeforeSwizzling) {
  if (GetParam().first != ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP();
  clear();
  wave->set_exec(3);
  wave->set_scratch_lane_size(4);
  vector(2, 0, 0);
  vector(2, 1, 4);
  memory.write32(0x10000, 0xabcdef01);
  memory.write32(0x10000 + 256 + 4, 0xfeedbeef);
  ASSERT_NO_FATAL_FAILURE(issue(encoding(4, 1, true, 127, 0, false), false));
  EXPECT_EQ(cu->read_vgpr(wave->vgpr_alloc().base + 8, 0), 0xabcdef01u);
  EXPECT_EQ(cu->read_vgpr(wave->vgpr_alloc().base + 8, 1), 0u);
  // The dedicated SCRATCH instruction bypasses generic FLAT aperture error checking.
  EXPECT_EQ(wave->trapsts() & (1u << 8), 0u);
}

TEST_P(FlatLdsTransfer, OrdinaryScratchLoadsUseSameAddressSelection) {
  for (bool has_vector : {false, true}) {
    clear();
    wave->set_exec(1);
    scalar(4, 8);
    vector(2, 0, 4);
    memory.write32(0x10000 + (has_vector ? 1 : 2) * wave->wf_size() * 4, has_vector ? 123 : 456);
    ASSERT_NO_FATAL_FAILURE(
        issue(encoding(4, 1, has_vector, has_vector ? vector_saddr() : 4, 0, false), false));
    EXPECT_EQ(cu->read_vgpr(wave->vgpr_alloc().base + 8, 0), has_vector ? 123 : 456);
  }
}

TEST_P(FlatLdsTransfer, UnusedRegistersDoNotBecomeMemoryDependencies) {
  clear();
  wave->set_exec(1);
  scalar(4, 0);
  wave->set_m0(0);
  memory.write32(0x10000, 123);
  auto words = encoding(4, 1, false, 4);
  words[1] = (words[1] & 0x00ffff00u) | 255u | (255u << 24);
  ASSERT_NO_FATAL_FAILURE(issue(words));
  EXPECT_EQ(wave->lds().read32(128), 123);
}

TEST_P(FlatLdsTransfer, M0WidthAndPartialDwordBounds) {
  clear();
  wave->set_exec(7);
  wave->set_lds_size(74);
  wave->set_m0(0x10040);
  scalar(4, 0);
  for (unsigned lane = 0; lane < 3; ++lane)
    memory.write32(0x10000 + lane * 4, 123);
  ASSERT_NO_FATAL_FAILURE(issue(encoding(4, 1, false, 4)));
  EXPECT_EQ(wave->lds().read32(128 + 64), modern() ? 0xcccccccc : 123);
  EXPECT_EQ(wave->lds().read32(128 + 68), modern() ? 0xcccccccc : 123);
  EXPECT_EQ(wave->lds().read32(128 + 72), 0xcccccccc);
}

TEST_P(FlatLdsTransfer, SegmentDispatchAndInvalidLegacyDestinations) {
  for (unsigned kind = 0; kind < 5; ++kind) {
    for (unsigned segment : {0u, 3u}) {
      auto words = encoding(kind, segment, true, 127);
      EXPECT_TRUE(decode_fails(*decoder, words.data()));
    }
    if (modern()) {
      auto words = encoding(kind, 2, false, 127);
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      ASSERT_NE(inst, nullptr);
      EXPECT_TRUE(inst->mnemonic().starts_with("global_load_lds_"));
    }
  }
  if (!modern()) {
    // Ordinary DWORDX2 loads cannot use the legacy LDS destination bit.
    auto words = encoding(5, 1, true, 127);
    EXPECT_TRUE(decode_fails(*decoder, words.data()));
  }
}

TEST_P(FlatLdsTransfer, RdnaD16LdsWritesFullDwordsWithoutDestinationDependencies) {
  if (!rdna())
    GTEST_SKIP() << "This encoding is specific to RDNA1/2.";
  // LLVM opcode order: UBYTE, UBYTE_HI, SBYTE, SBYTE_HI, SHORT, SHORT_HI.
  for (unsigned opcode = 32; opcode <= 37; ++opcode)
    for (unsigned segment : {1u, 2u}) {
      SCOPED_TRACE(testing::Message() << "opcode=" << opcode << " segment=" << segment);
      clear();
      wave->set_exec(5);
      wave->set_m0(0);
      for (unsigned lane : {0u, 2u}) {
        vector(2, lane, segment == 1 ? 2 : 0x20002 + lane * 4);
        vector(3, lane, 0);
        memory.write32((segment == 1 ? 0x10000 : 0x20000) + lane * 4, 0xa7832190);
      }
      auto words = encoding(0, segment, true, vector_saddr());
      words[0] = (words[0] & ~(0x7fu << 18)) | (opcode << 18);
      // A nonexistent destination must neither be read nor be written.
      words[1] = (words[1] & 0x00ffffff) | (255u << 24);
      ASSERT_NO_FATAL_FAILURE(issue(words));
      const uint32_t expected = opcode < 34 ? 0x83 : opcode < 36 ? 0xffffff83 : 0xa783;
      for (unsigned lane : {0u, 2u})
        EXPECT_EQ(wave->lds().read32(128 + lane * 4), expected);
    }
}

TEST_P(FlatLdsTransfer, RdnaLlvmEncodingsAndUnsignedGlobalOffsets) {
  if (!rdna())
    GTEST_SKIP() << "This encoding is specific to RDNA1/2.";
  clear();
  wave->set_exec(1);
  wave->set_m0(0);
  vector(2, 0, 4);
  memory.write32(0x10000 + 2 * wave->wf_size() * 4, 0x11223344);
  ASSERT_NO_FATAL_FAILURE(issue({0xdc306004, 0x007d0002}));
  EXPECT_EQ(wave->lds().read32(132), 0x11223344);
  ASSERT_NO_FATAL_FAILURE(issue({0xdc304004, 0x087d0002}, false));
  EXPECT_EQ(cu->read_vgpr(wave->vgpr_alloc().base + 8, 0), 0x11223344);
  if (GetParam().first == ROCJITSU_CODE_ARCH_RDNA2) {
    memory.write32(0x10000 + wave->wf_size() * 4, 0x55667788);
    vector(0, 0, 0xffffffff);
    ASSERT_NO_FATAL_FAILURE(issue({0xdc304004, 0x087f0000}, false));
    EXPECT_EQ(cu->read_vgpr(wave->vgpr_alloc().base + 8, 0), 0x55667788);
  }
  for (bool lds : {false, true})
    for (bool scalar_base : {false, true}) {
      vector(2, 0, 0x80000000);
      vector(3, 0, 2);
      scalar(4, 0x10000);
      scalar(5, 1);
      const uint64_t address = scalar_base ? 0x180010000ULL : 0x280000000ULL;
      memory.write32(address, 0x31415926);
      ASSERT_NO_FATAL_FAILURE(
          issue(encoding(4, 2, true, scalar_base ? 4 : vector_saddr(), 0, lds), lds));
      auto words = encoding(4, 2, true, scalar_base ? 4 : vector_saddr(), 0, lds);
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      InstDefUse effects(*inst, 64);
      EXPECT_EQ(effects.uses.contains({RegClass::M0, 0, 1}), lds);
      EXPECT_EQ(lds ? wave->lds().read32(128) : cu->read_vgpr(wave->vgpr_alloc().base + 8, 0),
                0x31415926);
    }
}

TEST_P(FlatLdsTransfer, RdnaScratchScalarOffsetIsUnsigned) {
  if (!rdna())
    GTEST_SKIP() << "CDNA retains its existing scalar-offset arithmetic.";
  wave->set_exec(1);
  wave->set_scratch_lane_size(0x80000004);
  scalar(4, 0x80000000);
  auto words = encoding(4, 1, false, 4);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
  const auto *state = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->per_lane_addr[0], 0x10000 + uint64_t{0x80000000} * wave->wf_size());
}

TEST_P(FlatLdsTransfer, GlobalWideLoadsPreservePaddingInactiveLanesAndVgprs) {
  if (GetParam().first != ROCJITSU_CODE_ARCH_CDNA4)
    GTEST_SKIP() << "Wide GLOBAL LDS loads require CDNA4.";
  constexpr uint64_t base = 0x123400020000;
  for (unsigned count : {3u, 4u})
    for (bool scalar_base : {false, true})
      for (int offset : {-16, -4, 0, 4, 16})
        for (uint64_t exec : {uint64_t{0}, uint64_t{0x25}, uint64_t{0x8000000000000025}})
          for (unsigned vdst : {8u, 255u}) {
            SCOPED_TRACE(testing::Message() << count << ',' << scalar_base << ',' << offset << ','
                                            << exec << ',' << vdst);
            clear();
            cu->flush_all();
            wave->set_lds_size(2048);
            wave->set_exec(exec);
            wave->set_m0(0xfffc0043);
            scalar(4, static_cast<uint32_t>(base));
            scalar(5, base >> 32);
            for (unsigned lane = 0; lane < wave->wf_size(); ++lane) {
              const uint64_t address = base + 0x80000000u + lane * 32;
              vector(2, lane, scalar_base ? 0x80000000u + lane * 32 : address);
              vector(3, lane, address >> 32);
              for (unsigned word = 0; word < 4; ++word) {
                vector(8 + word, lane, 0xdeadbeef);
                if (word < count)
                  memory.write32(address + offset + word * 4, 0x12340000 + lane * 16 + word);
              }
            }
            ASSERT_NO_FATAL_FAILURE(
                issue(global_wide_lds_encoding(count, scalar_base ? 4 : 127, offset, vdst)));
            EXPECT_EQ(wave->exec(), exec);
            for (unsigned address = 0; address < 1408; address += 4) {
              const int relative = static_cast<int>(address) - (128 + 64 + offset);
              const unsigned lane = relative / 16;
              const unsigned word = (relative % 16) / 4;
              const bool written =
                  relative >= 0 && lane < 64 && word < count && (exec & (uint64_t{1} << lane));
              EXPECT_EQ(wave->lds().read32(address),
                        written ? 0x12340000 + lane * 16 + word : 0xcccccccc)
                  << address;
            }
            for (unsigned lane = 0; lane < wave->wf_size(); ++lane)
              for (unsigned word = 0; word < 4; ++word)
                EXPECT_EQ(cu->read_vgpr(wave->vgpr_alloc().base + 8 + word, lane), 0xdeadbeef);
          }
}

TEST_P(FlatLdsTransfer, GlobalWideLoadsRespectM0WidthAndWholePayloadBounds) {
  if (GetParam().first != ROCJITSU_CODE_ARCH_CDNA4)
    GTEST_SKIP() << "Wide GLOBAL LDS loads require CDNA4.";
  for (unsigned count : {3u, 4u})
    for (uint32_t m0 : {0u, 3u, 0x10003u, 0xfffc0003u})
      for (int offset : {-16, 0, 16})
        // Include exact-fit and one-byte-short boundaries for the first two lanes.
        for (unsigned allocation_size : {11u, 12u, 15u, 16u, 27u, 28u, 31u, 32u, 44u}) {
          SCOPED_TRACE(testing::Message()
                       << count << ',' << m0 << ',' << offset << ',' << allocation_size);
          clear();
          cu->flush_all();
          wave->set_lds_size(allocation_size);
          wave->set_exec(7);
          wave->set_m0(m0);
          for (unsigned lane = 0; lane < 3; ++lane) {
            vector(2, lane, 0x20000 + lane * 32);
            vector(3, lane, 0);
            for (unsigned word = 0; word < count; ++word)
              memory.write32(0x20000 + lane * 32 + offset + word * 4, 0xabc000 + lane * 16 + word);
          }
          ASSERT_NO_FATAL_FAILURE(issue(global_wide_lds_encoding(count, 127, offset)));
          for (unsigned address = 0; address < 256; address += 4) {
            uint32_t expected = 0xcccccccc;
            for (unsigned lane = 0; lane < 3; ++lane) {
              const int relative =
                  static_cast<int>(m0 & 0x3fffcu) + offset + static_cast<int>(lane) * 16;
              if (relative < 0 || relative + count * 4 > allocation_size)
                continue;
              for (unsigned word = 0; word < count; ++word)
                if (address == 128 + relative + word * 4)
                  expected = 0xabc000 + lane * 16 + word;
            }
            EXPECT_EQ(wave->lds().read32(address), expected) << address;
          }
        }
}

TEST_P(FlatLdsTransfer, GlobalWideLoadsApplyThirteenBitOffsetToLds) {
  if (GetParam().first != ROCJITSU_CODE_ARCH_CDNA4)
    GTEST_SKIP() << "Wide GLOBAL LDS loads require CDNA4.";
  // Offset 2048 sets bit 11, so it becomes -2048 if only 12 bits are sign-extended.
  constexpr int offset = 2048;
  constexpr unsigned destination = 128 + offset;
  for (unsigned count : {3u, 4u}) {
    SCOPED_TRACE(count);
    clear();
    for (unsigned word = 0; word < 4; ++word)
      wave->lds().write32(destination + word * 4, 0xcccccccc);
    cu->flush_all();
    wave->set_lds_size(4096);
    wave->set_exec(1);
    wave->set_m0(0);
    vector(2, 0, 0x20000);
    vector(3, 0, 0);
    for (unsigned word = 0; word < count; ++word)
      memory.write32(0x20000 + offset + word * 4, 0xabc000 + word);
    ASSERT_NO_FATAL_FAILURE(issue(global_wide_lds_encoding(count, 127, offset)));
    for (unsigned word = 0; word < 4; ++word)
      EXPECT_EQ(wave->lds().read32(destination + word * 4),
                word < count ? 0xabc000 + word : 0xcccccccc)
          << word;
  }
}

INSTANTIATE_TEST_SUITE_P(Architectures, FlatLdsTransfer,
                         testing::Values(std::pair{ROCJITSU_CODE_ARCH_CDNA1, 64u},
                                         std::pair{ROCJITSU_CODE_ARCH_CDNA2, 64u},
                                         std::pair{ROCJITSU_CODE_ARCH_CDNA3, 64u},
                                         std::pair{ROCJITSU_CODE_ARCH_CDNA4, 64u},
                                         std::pair{ROCJITSU_CODE_ARCH_RDNA1, 32u},
                                         std::pair{ROCJITSU_CODE_ARCH_RDNA1, 64u},
                                         std::pair{ROCJITSU_CODE_ARCH_RDNA2, 32u},
                                         std::pair{ROCJITSU_CODE_ARCH_RDNA2, 64u}));
} // namespace
} // namespace rocjitsu
