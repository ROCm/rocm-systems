// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
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
    cu = amdgpu::ComputeUnitCore::create("flat_lds", cfg, &memory, &l2);
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
    const unsigned bit13 = modern() ? vaddr : lds;
    return {0xdc000000u | (op << 18) | (segment << 14) | (bit13 << 13) |
                (static_cast<unsigned>(offset) & (rdna() ? 0xfff : 0x1fff)),
            2u | (saddr << 16) | (8u << 24)};
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
      InstDefUse effects(*inst);
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
      // Preserve the GLOBAL sibling's decoding. Its execution remains
      // unimplemented and is covered by a separate change (PR #13218).
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
      InstDefUse effects(*inst);
      EXPECT_EQ(effects.uses.contains({RegClass::M0, 0, 1}), lds);
      EXPECT_EQ(lds ? wave->lds().read32(128) : cu->read_vgpr(wave->vgpr_alloc().base + 8, 0),
                0x31415926);
    }
}

TEST_P(FlatLdsTransfer, RdnaScratchScalarOffsetIsUnsigned) {
  if (!rdna())
    GTEST_SKIP() << "CDNA retains its existing scalar-offset arithmetic.";
  wave->set_exec(1);
  scalar(4, 0x80000000);
  auto words = encoding(4, 1, false, 4);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
  const auto *state = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->per_lane_addr[0], 0x10000 + uint64_t{0x80000000} * wave->wf_size());
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
