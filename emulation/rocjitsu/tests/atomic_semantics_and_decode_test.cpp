// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna3_5_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna3_to_rdna4.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
using namespace rocjitsu;
namespace {
class InstructionPolicyMachine {
public:
  explicit InstructionPolicyMachine(rj_code_arch_t arch)
      : memory("policy_memory"), cache("policy_cache") {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cache.set_backing_memory(&memory);
    cu = amdgpu::ComputeUnitCore::create("instruction_policy", config, &memory, &cache);
    decoder = Decoder::create(arch);
    wave = cu->dispatch_wf(0, 0, 106, 256);
    wave->set_exec(1);
    base = wave->vgpr_alloc().base;
  }
  ~InstructionPolicyMachine() { wave->halt(); }
  template <size_t N>
  uint32_t run(const std::array<uint32_t, N> &words, uint32_t a, uint32_t b = 0, uint32_t c = 0,
               uint32_t mode = 0xf0, bool memory_op = false, bool buffer = false) {
    wave->set_mode_raw(mode);
    cu->write_vgpr(base, 0, a);
    cu->write_vgpr(base + 1, 0, b);
    cu->write_vgpr(base + 2, 0, c);
    cu->write_vgpr(base + 6, 0, 0xfacebeef);
    std::array<uint32_t, 4> padded{};
    std::copy(words.begin(), words.end(), padded.begin());
    DecodeResult decoded = decoder->decode(padded.data());
    if (decoded.failed())
      throw std::runtime_error("Instruction encoding rejected by decoder");
    std::unique_ptr<Instruction> instruction(std::move(decoded).value());
    cu->execute_instruction(instruction.get(), *wave);
    if (memory_op) {
      amdgpu::L1VectorCache l1(&cache);
      amdgpu::GlobalMemPipeline pipeline(&l1, &cache);
      pipeline.issue(instruction.release(), *wave);
      l1.flush_all();
      cache.flush_all();
    }
    return cu->read_vgpr(base + (buffer ? 0 : 6), 0);
  }
  template <size_t N>
  uint32_t global(const std::array<uint32_t, N> &words, uint32_t old, uint32_t src, bool buffer) {
    memory.write32(0x1000, old);
    const uint32_t scalar = wave->sgpr_alloc().base;
    cu->write_sgpr(scalar + 4, 0x1000);
    cu->write_sgpr(scalar + 5, 0);
    cu->write_sgpr(scalar + 6, 4096);
    cu->write_sgpr(scalar + 7, 0);
    cu->write_vgpr(base + 4, 0, 0);
    cu->write_vgpr(base + 5, 0, 0);
    const uint32_t returned = run(words, src, 0, 0, 0xf0, true, buffer);
    EXPECT_EQ(returned, old);
    return memory.read32(0x1000);
  }
  amdgpu::GpuMemory memory;
  amdgpu::L2Cache cache;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wave = nullptr;
  uint32_t base = 0;
};
void witness(const char *name, rj_code_arch_t arch, uint32_t got, uint32_t expected) {
  EXPECT_EQ(got, expected) << name << " arch=" << unsigned(arch);
}
TEST(AtomicSemanticsAndDecode, ConditionalAndClampedMemorySubtract) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA4}) {
    InstructionPolicyMachine machine(arch);
    for (uint16_t opcode : {uint16_t{55}, uint16_t{80}}) {
      const char *name = opcode == 55 ? "sub_clamp" : "cond_sub";
      for (uint32_t old : {2u, 5u, 7u}) {
        const uint32_t expected = old >= 5 ? old - 5 : (opcode == 55 ? 0 : old);
        const std::array<uint32_t, 3> buffer = cdna5::build_vbuffer(
            opcode,
            {.soffset = 124, .vdata = 0, .rsrc = 4, .scope = 2, .th = 1, .offen = 1, .vaddr = 4});
        witness(name, arch, machine.global(buffer, old, 5, true), expected);
        const std::array<uint32_t, 3> global = cdna5::build_vglobal(
            opcode, {.saddr = 4, .vdst = 6, .scope = 2, .th = 1, .vsrc = 0, .vaddr = 4});
        witness(name, arch, machine.global(global, old, 5, false), expected);
      }
    }
  }
}
TEST(AtomicSemanticsAndDecode, ConditionalAndClampedLegacyCsub) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    InstructionPolicyMachine machine(arch);
    std::array<uint32_t, 2> buffer{}, global{};
    if (arch == ROCJITSU_CODE_ARCH_RDNA2) {
      buffer =
          rdna2::build_mubuf(52, {.glc = 1, .vaddr = 4, .vdata = 0, .srsrc = 1, .soffset = 128});
      global =
          rdna2::build_flat(52, {.seg = 2, .glc = 1, .addr = 4, .data = 0, .saddr = 4, .vdst = 6});
    } else {
      buffer =
          rdna3::build_mubuf(55, {.glc = 1, .vaddr = 4, .vdata = 0, .srsrc = 1, .soffset = 128});
      global =
          rdna3::build_flat(55, {.glc = 1, .seg = 2, .addr = 4, .data = 0, .saddr = 4, .vdst = 6});
    }
    witness("legacy_csub_buffer", arch, machine.global(buffer, 2, 5, true), 0);
    ASSERT_FALSE(machine.decoder->decode(global.data()).failed());
    witness("legacy_csub_global", arch, machine.global(global, 2, 5, false), 0);
  }
}
TEST(AtomicSemanticsAndDecode, LegacyCsubLegalizesToClampedSubtract) {
  const std::array<uint32_t, 2> source =
      rdna3::build_flat(rdna3::kGlobalAtomicCsubU32Flat,
                        {.glc = 1, .seg = 2, .addr = 4, .data = 0, .saddr = 4, .vdst = 6});
  for (bool rdna35 : {false, true}) {
    const InstructionLegalization *entry =
        rdna35 ? lookup(kLegalization_rdna3_5_to_rdna4, source[0] >> 23,
                        rdna3::kGlobalAtomicCsubU32Flat)
               : lookup(kLegalization_rdna3_to_rdna4, source[0] >> 23,
                        rdna3::kGlobalAtomicCsubU32Flat);
    ASSERT_NE(entry, nullptr);
    ASSERT_EQ(entry->target_opcode, rdna4::kGlobalAtomicSubClampU32Vglobal);
    InstructionPolicyMachine machine(ROCJITSU_CODE_ARCH_RDNA4);
    const std::array<uint32_t, 3> target = rdna4::build_vglobal(
        entry->target_opcode, {.saddr = 4, .vdst = 6, .scope = 2, .th = 1, .vsrc = 0, .vaddr = 4});
    EXPECT_EQ(machine.global(target, 2, 5, false), 0u);
    EXPECT_EQ(machine.global(target, 7, 5, false), 2u);
  }
}

TEST(AtomicSemanticsAndDecode, TranslatedCompareStorePreservesOperandOrder) {
  for (uint16_t opcode : {cdna4::kDsCmpstB32Ds, cdna4::kDsCmpstRtnB32Ds, cdna4::kDsCmpstB64Ds,
                          cdna4::kDsCmpstRtnB64Ds}) {
    const std::array<uint32_t, 2> source =
        cdna4::build_ds(opcode, {.addr = 4, .data0 = 0, .data1 = 2, .vdst = 6});
    const TranslationResult translated =
        cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_DS, source[0], source[1], 0, opcode);
    ASSERT_EQ(translated.word_count, 2);
    for (bool equal : {false, true}) {
      InstructionPolicyMachine machine(ROCJITSU_CODE_ARCH_RDNA4);
      machine.wave->lds().write32(0x100, equal ? 2 : 3);
      machine.wave->lds().write32(0x104, 0);
      machine.cu->write_vgpr(machine.base, 0, 2);
      machine.cu->write_vgpr(machine.base + 1, 0, 0);
      machine.cu->write_vgpr(machine.base + 2, 0, 5);
      machine.cu->write_vgpr(machine.base + 3, 0, 0);
      machine.cu->write_vgpr(machine.base + 4, 0, 0x100);
      DecodeResult decoded = machine.decoder->decode(translated.words);
      ASSERT_FALSE(decoded.failed());
      std::unique_ptr<Instruction> instruction(std::move(decoded).value());
      machine.cu->execute_instruction(instruction.get(), *machine.wave);
      amdgpu::LocalMemPipeline pipeline;
      pipeline.issue(instruction.release(), *machine.wave);
      EXPECT_EQ(machine.wave->lds().read32(0x100), equal ? 5u : 3u);
      EXPECT_EQ(machine.wave->lds().read32(0x104), 0u);
      if (opcode == cdna4::kDsCmpstRtnB32Ds || opcode == cdna4::kDsCmpstRtnB64Ds) {
        EXPECT_EQ(machine.cu->read_vgpr(machine.base + 6, 0), equal ? 2u : 3u);
      }
    }
  }
}

TEST(AtomicSemanticsAndDecode, ConditionalAndClampedDsSubtract) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA4}) {
    InstructionPolicyMachine machine(arch);
    for (uint16_t opcode : {uint16_t{152}, uint16_t{153}, uint16_t{168}, uint16_t{169}}) {
      machine.wave->lds().write32(0x100, 2);
      machine.cu->write_vgpr(machine.base + 4, 0, 0x100);
      const std::array<uint32_t, 2> words =
          cdna5::build_vds(opcode, {.addr = 4, .data0 = 0, .vdst = 6});
      machine.wave->set_mode_raw(0xf0);
      machine.cu->write_vgpr(machine.base, 0, 5);
      machine.cu->write_vgpr(machine.base + 6, 0, 0xbad);
      std::unique_ptr<Instruction> instruction(machine.decoder->decode(words.data()).value());
      machine.cu->execute_instruction(instruction.get(), *machine.wave);
      amdgpu::LocalMemPipeline pipeline;
      pipeline.issue(instruction.release(), *machine.wave);
      witness("ds_conditional_clamped_sub", arch, machine.wave->lds().read32(0x100),
              (opcode & 1) ? 0 : 2);
      if (opcode >= 168) {
        EXPECT_EQ(machine.cu->read_vgpr(machine.base + 6, 0), 2u);
      }
    }
  }
}

TEST(AtomicSemanticsAndDecode, GlobalOnlyPackedHalfAndScalarFloatAtomics) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2}) {
    InstructionPolicyMachine machine(arch);
    // Both opcodes are GLOBAL-only on these targets; FLAT and SCRATCH must reject them.
    for (uint16_t opcode : {uint16_t{77}, uint16_t{78}}) {
      for (uint8_t segment : {uint8_t{0}, uint8_t{1}, uint8_t{2}, uint8_t{3}}) {
        const std::array<uint32_t, 2> words = cdna1::build_flat(
            opcode, {.seg = segment, .glc = 1, .addr = 4, .data = 0, .saddr = 4, .vdst = 6});
        EXPECT_EQ(machine.decoder->decode(words.data()).failed(), segment != 2);
        if (segment != 2)
          continue;
        if (opcode == 78) {
          EXPECT_EQ(machine.global(words, 0x3c004000, 0x42004400, false), 0x44004600u);
          EXPECT_EQ(machine.global(words, 0x00010001, 0x00010001, false), 0x00020002u);
          EXPECT_EQ(machine.global(words, 0x7bff7bff, 0x7bff7bff, false), 0x7c007c00u);
        } else {
          EXPECT_EQ(machine.global(words, 1, 1, false), 0u);
        }
      }
    }
  }
}

TEST(AtomicSemanticsAndDecode, GlobalOnlyCsubRejectsOtherSegments) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    InstructionPolicyMachine machine(arch);
    for (uint8_t segment : {uint8_t{0}, uint8_t{1}, uint8_t{2}, uint8_t{3}}) {
      const std::array<uint32_t, 2> words =
          arch == ROCJITSU_CODE_ARCH_RDNA2
              ? rdna2::build_flat(
                    52, {.seg = segment, .glc = 1, .addr = 4, .data = 0, .saddr = 4, .vdst = 6})
              : rdna3::build_flat(
                    55, {.glc = 1, .seg = segment, .addr = 4, .data = 0, .saddr = 4, .vdst = 6});
      EXPECT_EQ(machine.decoder->decode(words.data()).failed(), segment != 2);
    }
  }
}

TEST(AtomicSemanticsAndDecode, LegacyGlobalAddtidUsesScalarAddressAndSignedOffset) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    InstructionPolicyMachine machine(arch);
    const uint32_t scalar = machine.wave->sgpr_alloc().base;
    machine.cu->write_sgpr(scalar + 4, 0x1004);
    machine.cu->write_sgpr(scalar + 5, 0);
    machine.memory.write32(0x1000, 0x12345678);
    const std::array<uint32_t, 2> load =
        arch == ROCJITSU_CODE_ARCH_RDNA2
            ? rdna2::build_flat(22, {.offset = 0x1ffc, .seg = 2, .saddr = 4, .vdst = 6})
            : rdna3::build_flat(40, {.offset = 0x1ffc, .seg = 2, .saddr = 4, .vdst = 6});
    EXPECT_EQ(machine.run(load, 0, 0, 0, 0xf0, true), 0x12345678u);
    const std::array<uint32_t, 2> store =
        arch == ROCJITSU_CODE_ARCH_RDNA2
            ? rdna2::build_flat(23, {.offset = 0x1ffc, .seg = 2, .data = 0, .saddr = 4})
            : rdna3::build_flat(41, {.offset = 0x1ffc, .seg = 2, .data = 0, .saddr = 4});
    machine.run(store, 0xabcdef01, 0, 0, 0xf0, true);
    EXPECT_EQ(machine.memory.read32(0x1000), 0xabcdef01u);
    machine.cu->write_sgpr(scalar + 4, 0x2000);
    machine.memory.write32(0x2000, 0x11223344);
    const std::array<uint32_t, 2> boundary_load =
        arch == ROCJITSU_CODE_ARCH_RDNA2
            ? rdna2::build_flat(22, {.offset = 0x1000, .seg = 2, .saddr = 4, .vdst = 6})
            : rdna3::build_flat(40, {.offset = 0x1000, .seg = 2, .saddr = 4, .vdst = 6});
    EXPECT_EQ(machine.run(boundary_load, 0, 0, 0, 0xf0, true),
              arch == ROCJITSU_CODE_ARCH_RDNA2 ? 0x11223344u : 0xabcdef01u);
  }
}

TEST(AtomicSemanticsAndDecode, PackedAtomicsAcrossOlderTargets) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                              ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    InstructionPolicyMachine machine(arch);
    const std::array<uint32_t, 2> buffer =
        cdna1::build_mubuf(78, {.glc = 1, .vaddr = 4, .vdata = 0, .srsrc = 1, .soffset = 128});
    EXPECT_EQ(machine.global(buffer, 0x3c004000, 0x42004400, true), 0x44004600u);
    EXPECT_EQ(machine.global(buffer, 0x00010001, 0x00010001, true), 0x00020002u);
    const int saved = std::fegetround();
    std::fesetround(FE_UPWARD);
    EXPECT_EQ(machine.global(buffer, 0x3c003c00, 0x10001000, true), 0x3c003c00u);
    std::fesetround(saved);
    if (arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2)
      continue;
    for (bool bf16 : {false, true}) {
      const std::array<uint32_t, 2> global = cdna3::build_flat(
          bf16 ? 82 : 78, {.seg = 2, .sc0 = 1, .addr = 4, .data = 0, .saddr = 4, .vdst = 6});
      const uint32_t one_two = bf16 ? 0x3f804000u : 0x3c004000u;
      const uint32_t three_four = bf16 ? 0x40404080u : 0x42004400u;
      const uint32_t four_six = bf16 ? 0x408040c0u : 0x44004600u;
      EXPECT_EQ(machine.global(global, one_two, three_four, false), four_six);
      EXPECT_EQ(machine.global(global, 0x00010001, 0x00010001, false), 0x00020002u);
    }
  }
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4}) {
    InstructionPolicyMachine machine(arch);
    for (bool bf16 : {false, true}) {
      const std::array<uint32_t, 2> words =
          arch == ROCJITSU_CODE_ARCH_RDNA4
              ? rdna4::build_vds(bf16 ? 171 : 170, {.addr = 4, .data0 = 0, .vdst = 6})
              : cdna3::build_ds(bf16 ? 184 : 183, {.addr = 4, .data0 = 0, .vdst = 6});
      for (uint32_t mode : {0u, 0xf0u}) {
        machine.wave->set_mode_raw(mode);
        machine.wave->lds().write32(0x100, 0x00010001);
        machine.cu->write_vgpr(machine.base + 4, 0, 0x100);
        machine.cu->write_vgpr(machine.base, 0, 0x00010001);
        DecodeResult decoded = machine.decoder->decode(words.data());
        ASSERT_FALSE(decoded.failed());
        std::unique_ptr<Instruction> instruction(std::move(decoded).value());
        machine.cu->execute_instruction(instruction.get(), *machine.wave);
        amdgpu::LocalMemPipeline pipeline;
        pipeline.issue(instruction.release(), *machine.wave);
        EXPECT_EQ(machine.cu->read_vgpr(machine.base + 6, 0), 0x00010001u);
        EXPECT_EQ(machine.wave->lds().read32(0x100), mode ? 0x00020002u : 0u);
      }
    }
  }
}
} // namespace

namespace {
class AtomicPolicyExecutionTest : public testing::TestWithParam<rj_code_arch_t> {
protected:
  void SetUp() override {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = GetParam();
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cache_.set_backing_memory(&memory_);
    compute_unit_ = amdgpu::ComputeUnitCore::create("atomic_policy", config, &memory_, &cache_);
    decoder_ = Decoder::create(GetParam());
    wave_ = compute_unit_->dispatch_wf(0, 0, 106, 256);
    wave_->set_exec(1);
    base_ = wave_->vgpr_alloc().base;
  }

  void TearDown() override { wave_->halt(); }

  bool modern_policy() const {
    return GetParam() == ROCJITSU_CODE_ARCH_CDNA5 || GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  }

  bool compare_first() const {
    return GetParam() == ROCJITSU_CODE_ARCH_CDNA1 || GetParam() == ROCJITSU_CODE_ARCH_CDNA2 ||
           GetParam() == ROCJITSU_CODE_ARCH_CDNA3 || GetParam() == ROCJITSU_CODE_ARCH_CDNA4 ||
           GetParam() == ROCJITSU_CODE_ARCH_RDNA1 || GetParam() == ROCJITSU_CODE_ARCH_RDNA2;
  }

  std::array<uint32_t, 2> ds_words(uint16_t opcode) const {
    switch (GetParam()) {
    case ROCJITSU_CODE_ARCH_CDNA1:
      return cdna1::build_ds(opcode, {.addr = 4, .data0 = 0, .data1 = 2, .vdst = 6});
    case ROCJITSU_CODE_ARCH_CDNA2:
      return cdna2::build_ds(opcode, {.addr = 4, .data0 = 0, .data1 = 2, .vdst = 6});
    case ROCJITSU_CODE_ARCH_CDNA3:
      return cdna3::build_ds(opcode, {.addr = 4, .data0 = 0, .data1 = 2, .vdst = 6});
    case ROCJITSU_CODE_ARCH_CDNA4:
      return cdna4::build_ds(opcode, {.addr = 4, .data0 = 0, .data1 = 2, .vdst = 6});
    case ROCJITSU_CODE_ARCH_RDNA1:
      return rdna1::build_ds(opcode, {.addr = 4, .data0 = 0, .data1 = 2, .vdst = 6});
    case ROCJITSU_CODE_ARCH_RDNA2:
      return rdna2::build_ds(opcode, {.addr = 4, .data0 = 0, .data1 = 2, .vdst = 6});
    case ROCJITSU_CODE_ARCH_RDNA3:
      return rdna3::build_ds(opcode, {.addr = 4, .data0 = 0, .data1 = 2, .vdst = 6});
    case ROCJITSU_CODE_ARCH_RDNA3_5:
      return rdna3_5::build_ds(opcode, {.addr = 4, .data0 = 0, .data1 = 2, .vdst = 6});
    case ROCJITSU_CODE_ARCH_RDNA4:
      return rdna4::build_vds(opcode, {.addr = 4, .data0 = 0, .data1 = 2, .vdst = 6});
    case ROCJITSU_CODE_ARCH_CDNA5:
      return cdna5::build_vds(opcode, {.addr = 4, .data0 = 0, .data1 = 2, .vdst = 6});
    default:
      ADD_FAILURE() << "Unexpected architecture";
      return {};
    }
  }

  uint16_t ds_add_return_opcode() const {
    switch (GetParam()) {
    case ROCJITSU_CODE_ARCH_CDNA1:
      return cdna1::kDsAddRtnF32Ds;
    case ROCJITSU_CODE_ARCH_CDNA2:
      return cdna2::kDsAddRtnF32Ds;
    case ROCJITSU_CODE_ARCH_CDNA3:
      return cdna3::kDsAddRtnF32Ds;
    case ROCJITSU_CODE_ARCH_CDNA4:
      return cdna4::kDsAddRtnF32Ds;
    case ROCJITSU_CODE_ARCH_CDNA5:
      return cdna5::kDsAddRtnF32Vds;
    case ROCJITSU_CODE_ARCH_RDNA1:
      return rdna1::kDsAddRtnF32Ds;
    case ROCJITSU_CODE_ARCH_RDNA2:
      return rdna2::kDsAddRtnF32Ds;
    case ROCJITSU_CODE_ARCH_RDNA3:
      return rdna3::kDsAddRtnF32Ds;
    case ROCJITSU_CODE_ARCH_RDNA3_5:
      return rdna3_5::kDsAddRtnF32Ds;
    case ROCJITSU_CODE_ARCH_RDNA4:
      return rdna4::kDsAddRtnF32Vds;
    default:
      return 0;
    }
  }

  uint64_t execute_ds(uint16_t opcode, uint64_t old_bits, uint64_t source_bits,
                      uint64_t compare_bits, uint32_t mode, uint32_t width = 4,
                      uint32_t address = 0x100) {
    constexpr uint32_t kAddress = 0x100;
    wave_->set_mode_raw(mode);
    if (width == 8)
      wave_->lds().write64(kAddress, old_bits);
    else
      wave_->lds().write32(kAddress, static_cast<uint32_t>(old_bits));
    const bool comparison = opcode == cdna4::kDsCmpstB32Ds || opcode == cdna4::kDsCmpstF32Ds ||
                            opcode == cdna4::kDsCmpstRtnB32Ds ||
                            opcode == cdna4::kDsCmpstRtnF32Ds || opcode == cdna4::kDsCmpstB64Ds ||
                            opcode == cdna4::kDsCmpstF64Ds || opcode == cdna4::kDsCmpstRtnB64Ds ||
                            opcode == cdna4::kDsCmpstRtnF64Ds;
    const bool reversed = comparison && compare_first();
    const uint64_t data0 = reversed ? compare_bits : source_bits;
    const uint64_t data1 = reversed ? source_bits : compare_bits;
    compute_unit_->write_vgpr(base_, 0, static_cast<uint32_t>(data0));
    compute_unit_->write_vgpr(base_ + 1, 0, static_cast<uint32_t>(data0 >> 32));
    compute_unit_->write_vgpr(base_ + 2, 0, static_cast<uint32_t>(data1));
    compute_unit_->write_vgpr(base_ + 3, 0, static_cast<uint32_t>(data1 >> 32));
    compute_unit_->write_vgpr(base_ + 4, 0, address);
    compute_unit_->write_vgpr(base_ + 6, 0, 0xbad);
    compute_unit_->write_vgpr(base_ + 7, 0, 0xbad);
    const std::array<uint32_t, 2> words = ds_words(opcode);
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder_, words.data()));
    if (!instruction) {
      ADD_FAILURE() << "Invalid DS opcode " << opcode;
      return 0;
    }
    compute_unit_->execute_instruction(instruction.get(), *wave_);
    if (!instruction->data()) {
      ADD_FAILURE() << "No memory request for " << instruction->mnemonic();
      return old_bits;
    }
    const bool returns = instruction->data_as<amdgpu::VectorMemState>()->is_load;
    // Requests must snapshot MODE during execution, before deferred memory work.
    wave_->set_mode_raw(mode ^ 0xf0);
    amdgpu::LocalMemPipeline pipeline;
    pipeline.issue(instruction.release(), *wave_);
    if (returns) {
      EXPECT_EQ(compute_unit_->read_vgpr(base_ + 6, 0), static_cast<uint32_t>(old_bits));
      if (width == 8) {
        EXPECT_EQ(compute_unit_->read_vgpr(base_ + 7, 0), static_cast<uint32_t>(old_bits >> 32));
      }
    }
    return width == 8 ? wave_->lds().read64(kAddress) : wave_->lds().read32(kAddress);
  }

  std::array<uint32_t, 3> memory_words(bool buffer, bool compare = false) const {
    std::array<uint32_t, 2> words{};
    switch (GetParam()) {
    case ROCJITSU_CODE_ARCH_CDNA1:
      if (buffer)
        words = cdna1::build_mubuf(cdna1::kBufferAtomicAddF32Mubuf,
                                   {.glc = 1, .vaddr = 4, .vdata = 0, .srsrc = 1, .soffset = 128});
      else
        words =
            cdna1::build_flat(cdna1::kGlobalAtomicAddF32Flat,
                              {.seg = 2, .glc = 1, .addr = 4, .data = 0, .saddr = 4, .vdst = 6});
      break;
    case ROCJITSU_CODE_ARCH_CDNA2:
      if (buffer)
        words = cdna2::build_mubuf(cdna2::kBufferAtomicAddF32Mubuf,
                                   {.glc = 1, .vaddr = 4, .vdata = 0, .srsrc = 1, .soffset = 128});
      else
        words =
            cdna2::build_flat(cdna2::kGlobalAtomicAddF32Flat,
                              {.seg = 2, .glc = 1, .addr = 4, .data = 0, .saddr = 4, .vdst = 6});
      break;
    case ROCJITSU_CODE_ARCH_CDNA3:
      if (buffer)
        words = cdna3::build_mubuf(cdna3::kBufferAtomicAddF32Mubuf,
                                   {.sc0 = 1, .vaddr = 4, .vdata = 0, .srsrc = 1, .soffset = 128});
      else
        words =
            cdna3::build_flat(cdna3::kFlatAtomicAddF32Flat,
                              {.seg = 2, .sc0 = 1, .addr = 4, .data = 0, .saddr = 4, .vdst = 6});
      break;
    case ROCJITSU_CODE_ARCH_CDNA4:
      if (buffer)
        words = cdna4::build_mubuf(cdna4::kBufferAtomicAddF32Mubuf,
                                   {.sc0 = 1, .vaddr = 4, .vdata = 0, .srsrc = 1, .soffset = 128});
      else
        words =
            cdna4::build_flat(cdna4::kFlatAtomicAddF32Flat,
                              {.seg = 2, .sc0 = 1, .addr = 4, .data = 0, .saddr = 4, .vdst = 6});
      break;
    case ROCJITSU_CODE_ARCH_RDNA1:
      if (buffer)
        words = rdna1::build_mubuf(rdna1::kBufferAtomicFcmpswapMubuf,
                                   {.glc = 1, .vaddr = 4, .vdata = 0, .srsrc = 1, .soffset = 128});
      else
        words =
            rdna1::build_flat(rdna1::kFlatAtomicFcmpswapFlat,
                              {.seg = 2, .glc = 1, .addr = 4, .data = 0, .saddr = 4, .vdst = 6});
      break;
    case ROCJITSU_CODE_ARCH_RDNA2:
      if (buffer)
        words = rdna2::build_mubuf(rdna2::kBufferAtomicFcmpswapMubuf,
                                   {.glc = 1, .vaddr = 4, .vdata = 0, .srsrc = 1, .soffset = 128});
      else
        words =
            rdna2::build_flat(rdna2::kFlatAtomicFcmpswapFlat,
                              {.seg = 2, .glc = 1, .addr = 4, .data = 0, .saddr = 4, .vdst = 6});
      break;
    case ROCJITSU_CODE_ARCH_RDNA3:
      if (buffer)
        words = rdna3::build_mubuf(
            (compare ? rdna3::kBufferAtomicCmpswapF32Mubuf : rdna3::kBufferAtomicAddF32Mubuf),
            {.glc = 1, .vaddr = 4, .vdata = 0, .srsrc = 1, .soffset = 128});
      else
        words = rdna3::build_flat(
            (compare ? rdna3::kFlatAtomicCmpswapF32Flat : rdna3::kFlatAtomicAddF32Flat),
            {.glc = 1, .seg = 2, .addr = 4, .data = 0, .saddr = 4, .vdst = 6});
      break;
    case ROCJITSU_CODE_ARCH_RDNA3_5:
      if (buffer)
        words = rdna3_5::build_mubuf(
            (compare ? rdna3_5::kBufferAtomicCmpswapF32Mubuf : rdna3_5::kBufferAtomicAddF32Mubuf),
            {.glc = 1, .vaddr = 4, .vdata = 0, .srsrc = 1, .soffset = 128});
      else
        words = rdna3_5::build_flat(
            (compare ? rdna3_5::kFlatAtomicCmpswapF32Flat : rdna3_5::kFlatAtomicAddF32Flat),
            {.glc = 1, .seg = 2, .addr = 4, .data = 0, .saddr = 4, .vdst = 6});
      break;
    case ROCJITSU_CODE_ARCH_RDNA4:
      if (buffer)
        return rdna4::build_vbuffer(
            rdna4::kBufferAtomicAddF32Vbuffer,
            {.soffset = 124, .vdata = 0, .rsrc = 4, .scope = 2, .th = 1, .offen = 1, .vaddr = 4});
      return rdna4::build_vglobal(
          rdna4::kGlobalAtomicAddF32Vglobal,
          {.saddr = 4, .vdst = 6, .scope = 2, .th = 1, .vsrc = 0, .vaddr = 4});
    case ROCJITSU_CODE_ARCH_CDNA5:
      if (buffer)
        return cdna5::build_vbuffer(
            cdna5::kBufferAtomicAddF32Vbuffer,
            {.soffset = 124, .vdata = 0, .rsrc = 4, .scope = 2, .th = 1, .offen = 1, .vaddr = 4});
      return cdna5::build_vglobal(
          cdna5::kGlobalAtomicAddF32Vglobal,
          {.saddr = 4, .vdst = 6, .scope = 2, .th = 1, .vsrc = 0, .vaddr = 4});
    default:
      ADD_FAILURE() << "Unexpected memory architecture";
      break;
    }
    return {words[0], words[1], 0};
  }

  uint32_t execute_memory(bool buffer, uint32_t old_bits, uint32_t source_bits, uint32_t mode,
                          bool comparison = false, uint32_t compare_bits = 0) {
    constexpr uint32_t kAddress = 0x1000;
    memory_.write32(kAddress, old_bits);
    wave_->set_mode_raw(mode);
    const uint32_t scalar_base = wave_->sgpr_alloc().base;
    compute_unit_->write_sgpr(scalar_base + 4, kAddress);
    compute_unit_->write_sgpr(scalar_base + 5, 0);
    compute_unit_->write_sgpr(scalar_base + 6, 4096);
    compute_unit_->write_sgpr(scalar_base + 7, 0);
    compute_unit_->write_vgpr(base_, 0, source_bits);
    compute_unit_->write_vgpr(base_ + 1, 0, compare_bits);
    compute_unit_->write_vgpr(base_ + 4, 0, 0);
    compute_unit_->write_vgpr(base_ + 5, 0, 0);
    compute_unit_->write_vgpr(base_ + 6, 0, 0xbad);
    const std::array<uint32_t, 3> words = memory_words(buffer, comparison);
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder_, words.data()));
    if (!instruction) {
      ADD_FAILURE() << "Invalid memory atomic";
      return old_bits;
    }
    compute_unit_->execute_instruction(instruction.get(), *wave_);
    if (!instruction->data()) {
      ADD_FAILURE() << "No request for " << instruction->mnemonic();
      return old_bits;
    }
    wave_->set_mode_raw(mode ^ 0xf0);
    amdgpu::GlobalMemPipeline pipeline(nullptr, &cache_);
    pipeline.issue(instruction.release(), *wave_);
    EXPECT_EQ(compute_unit_->read_vgpr(base_ + (buffer ? 0 : 6), 0), old_bits);
    return memory_.read32(kAddress);
  }

  amdgpu::GpuMemory memory_{"atomic_policy_memory"};
  amdgpu::L2Cache cache_{"atomic_policy_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> compute_unit_;
  std::unique_ptr<Decoder> decoder_;
  amdgpu::Wavefront *wave_ = nullptr;
  uint32_t base_ = 0;
};

TEST_P(AtomicPolicyExecutionTest, DsScalarDenormalModesAndRounding) {
  for (uint16_t opcode : {uint16_t{21}, ds_add_return_opcode()}) {
    for (uint32_t denorm_mode = 0; denorm_mode < 4; ++denorm_mode) {
      const uint32_t mode = denorm_mode << 4;
      EXPECT_EQ(execute_ds(opcode, 1, 1, 0, mode), denorm_mode == 3 ? 2u : 0u);
      EXPECT_EQ(execute_ds(opcode, 0x00800001, 0x80800000, 0, mode), (denorm_mode & 2) ? 1u : 0u);
    }
    for (int host_round : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
      const int saved_round = std::fegetround();
      ASSERT_EQ(std::fesetround(host_round), 0);
      const uint64_t result = execute_ds(opcode, 0x3f800000, 0x33800000, 0, 0xff);
      const int restored_round = std::fegetround();
      std::fesetround(saved_round);
      EXPECT_EQ(result, 0x3f800000u);
      EXPECT_EQ(restored_round, host_round);
    }
  }
  EXPECT_EQ(execute_ds(18, 1, 0x3f800000, 0, 0), modern_policy() ? 0u : 1u);
  EXPECT_EQ(execute_ds(18, 0, 0x80000000, 0, 0xf0), 0x80000000u);
  EXPECT_EQ(execute_ds(19, 0x80000000, 0, 0, 0xf0), 0u);
  EXPECT_EQ(execute_ds(21, 0x7fc00002, 0x7fc00004, 0, 0), 0x7fc00002u);
  EXPECT_EQ(execute_ds(21, 0x7f800000, 0xff800000, 0, 0), 0xffc00000u);
  EXPECT_EQ(execute_ds(21, 0xff800000, 0x7f800000, 0, 0), 0xffc00000u);
  for (uint16_t opcode : {18, 19}) {
    EXPECT_EQ(execute_ds(opcode, 0x7f800002, 0x3f800000, 0, 0),
              modern_policy() ? 0x3f800000u : 0x7fc00002u);
    EXPECT_EQ(execute_ds(opcode, 0x3f800000, 0x7f800004, 0, 0),
              modern_policy() ? 0x3f800000u : 0x7fc00004u);
    EXPECT_EQ(execute_ds(opcode, 0x7fc00002, 0x3f800000, 0, 0), 0x3f800000u);
    EXPECT_EQ(execute_ds(opcode, 0x3f800000, 0x7fc00004, 0, 0), 0x3f800000u);
    EXPECT_EQ(execute_ds(opcode, 0x7fc00002, 0x7fc00004, 0, 0),
              modern_policy() ? 0x7fc00004u : 0x7fc00002u);
    EXPECT_EQ(execute_ds(opcode, 0x7fc00002, 0x7f800004, 0, 0), 0x7fc00004u);
  }
}

TEST_P(AtomicPolicyExecutionTest, DsMinMaxIgnoreOutputDenormalControl) {
  for (uint16_t opcode : {18, 19, 82, 83}) {
    const bool wide = opcode >= 64;
    const uint32_t width = wide ? 8 : 4;
    const uint64_t sign = wide ? 0x8000000000000000ULL : 0x80000000ULL;
    const uint64_t nan = wide ? 0x7ff8000000000002ULL : 0x7fc00002ULL;
    const bool minimum = (opcode & 1) == 0;
    const uint64_t selected = minimum ? 1 : 2;
    // MIN/MAX only use the input-denormal control, including NaN selection.
    for (uint32_t mode : {0x50u, 0xf0u}) {
      EXPECT_EQ(execute_ds(opcode, 1, 2, 0, mode, width), selected);
      EXPECT_EQ(execute_ds(opcode, sign | 1, sign | 2, 0, mode, width), sign | (minimum ? 2 : 1));
      EXPECT_EQ(execute_ds(opcode, 1, nan, 0, mode, width), 1u);
      EXPECT_EQ(execute_ds(opcode, nan, sign | 1, 0, mode, width), sign | 1);
    }
  }
}

TEST_P(AtomicPolicyExecutionTest, DsMinMaxBothNanPayloads) {
  for (uint16_t opcode : {uint16_t{18}, uint16_t{19}, uint16_t{82}, uint16_t{83}}) {
    const bool wide = opcode >= 64;
    const uint32_t width = wide ? 8 : 4;
    const uint64_t quiet = wide ? 0x8000000000000ULL : 0x400000ULL;
    const uint64_t old_nan = wide ? 0x7ff0000000000002ULL : 0x7f800002ULL;
    const uint64_t source_nan = wide ? 0xfff0000000000004ULL : 0xff800004ULL;
    for (bool quiet_old : {false, true}) {
      for (bool quiet_source : {false, true}) {
        const uint64_t old_bits = old_nan | (quiet_old ? quiet : 0);
        const uint64_t source_bits = source_nan | (quiet_source ? quiet : 0);
        const uint64_t selected =
            modern_policy() || (quiet_old && !quiet_source) ? source_bits : old_bits;
        EXPECT_EQ(execute_ds(opcode, old_bits, source_bits, 0, 0xf0, width), selected | quiet);
      }
    }
  }
}

TEST_P(AtomicPolicyExecutionTest, DsConditionalExchangeUsesData0SignBits) {
  // Both halves have independent enables; DATA1 is deliberately unrelated.
  constexpr uint64_t kOld = 0x1234567800000002ULL;
  constexpr uint64_t kUnrelatedData1 = 0xfedcba9876543210ULL;
  for (uint32_t enables = 0; enables < 4; ++enables) {
    const uint64_t source = 0x0000000500000007ULL | ((enables & 1u) ? 0x80000000ULL : 0) |
                            ((enables & 2u) ? 0x8000000000000000ULL : 0);
    const uint64_t expected = ((enables & 1u) ? 7ULL : 2ULL) |
                              ((enables & 2u) ? 0x0000000500000000ULL : 0x1234567800000000ULL);
    EXPECT_EQ(execute_ds(cdna4::kDsCondxchg32RtnB64Ds, kOld, source, kUnrelatedData1, 0, 8,
                         /*address=*/0x10107),
              expected);
  }
}

TEST_P(AtomicPolicyExecutionTest, DsIntegerCompareStoreOrderAndReturn) {
  for (uint16_t opcode : {16, 48, 80, 112}) {
    const uint32_t width = opcode >= 80 ? 8 : 4;
    const uint64_t old_bits = width == 8 ? 0x1234567800000002ULL : 2;
    const uint64_t replacement = width == 8 ? 0x9876543200000005ULL : 5;
    EXPECT_EQ(execute_ds(opcode, old_bits, replacement, old_bits, 0, width), replacement);
    EXPECT_EQ(execute_ds(opcode, old_bits, replacement, old_bits + 1, 0, width), old_bits);
  }
}

TEST_P(AtomicPolicyExecutionTest, DsFloatCompareStoreEqualityAndDenormals) {
  if (modern_policy())
    GTEST_SKIP() << "This ISA exposes integer compare-store only";
  for (uint16_t opcode : {17, 49, 81, 113}) {
    const uint32_t width = opcode >= 81 ? 8 : 4;
    const uint64_t one = width == 8 ? 0x3ff0000000000000ULL : 0x3f800000;
    const uint64_t negative_zero = width == 8 ? 0x8000000000000000ULL : 0x80000000;
    const uint64_t nan = width == 8 ? 0x7ff8000000000002ULL : 0x7fc00002;
    EXPECT_EQ(execute_ds(opcode, one, 0, one, 0xf0, width), 0u);
    EXPECT_EQ(execute_ds(opcode, 0, one, negative_zero, 0xf0, width), one);
    EXPECT_EQ(execute_ds(opcode, nan, one, nan, 0xf0, width), nan);
    EXPECT_EQ(execute_ds(opcode, 1, one, 0, 0, width), one);
    EXPECT_EQ(execute_ds(opcode, 1, one, 0, 0xf0, width), 1u);
    EXPECT_EQ(execute_ds(opcode, one, 1, one, 0, width), 0u);
    // Compare-store flushes its result using the input control, including a failed comparison.
    EXPECT_EQ(execute_ds(opcode, 1, one, one, 0, width), 0u);
    EXPECT_EQ(execute_ds(opcode, negative_zero | 1, one, one, 0, width), negative_zero);
    EXPECT_EQ(execute_ds(opcode, 1, one, one, 0x50, width), 1u);
  }
}

TEST_P(AtomicPolicyExecutionTest, MemoryAddDenormalsAndHostRounding) {
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA1 || GetParam() == ROCJITSU_CODE_ARCH_RDNA2)
    GTEST_SKIP() << "No F32 atomic add on this ISA";
  for (bool buffer : {false, true}) {
    for (uint32_t mode : {0u, 0xffu}) {
      EXPECT_EQ(execute_memory(buffer, 1, 1, mode), modern_policy() ? 2u : 0u);
      EXPECT_EQ(execute_memory(buffer, 0x00800001, 0x80800000, mode), 1u);
      for (int host_round : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
        const int saved_round = std::fegetround();
        ASSERT_EQ(std::fesetround(host_round), 0);
        const uint32_t result = execute_memory(buffer, 0x3f800000, 0x33800000, mode);
        const int restored_round = std::fegetround();
        std::fesetround(saved_round);
        EXPECT_EQ(result, 0x3f800000u);
        EXPECT_EQ(restored_round, host_round);
      }
    }
  }
}

TEST_P(AtomicPolicyExecutionTest, BufferFloatCompareSwapDefinesOnlyReturnedElement) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA1 && GetParam() != ROCJITSU_CODE_ARCH_RDNA2 &&
      GetParam() != ROCJITSU_CODE_ARCH_RDNA3 && GetParam() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "No floating buffer compare-swap on this ISA";
  for (uint32_t width : {1u, 2u}) {
    if (width == 2 && GetParam() != ROCJITSU_CODE_ARCH_RDNA1 &&
        GetParam() != ROCJITSU_CODE_ARCH_RDNA2)
      continue;
    std::array<uint32_t, 3> words = memory_words(/*buffer=*/true, /*compare=*/true);
    if (width == 2) {
      const std::array<uint32_t, 2> wide_words =
          GetParam() == ROCJITSU_CODE_ARCH_RDNA1
              ? rdna1::build_mubuf(rdna1::kBufferAtomicFcmpswapX2Mubuf, {.glc = 1})
              : rdna2::build_mubuf(rdna2::kBufferAtomicFcmpswapX2Mubuf, {.glc = 1});
      std::copy(wide_words.begin(), wide_words.end(), words.begin());
    }
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder_, words.data()));
    ASSERT_NE(instruction, nullptr);
    ASSERT_EQ(instruction->num_dst_operands(), 2);
    EXPECT_EQ(instruction->dst_operand(1)->size_bits(), width * 32);
    const InstDefUse def_use(*instruction);
    EXPECT_TRUE(
        def_use.uses.contains(RegisterRef{RegClass::VGPR, 0, static_cast<uint8_t>(width * 2)}));
    EXPECT_TRUE(def_use.defs.contains(RegisterRef{RegClass::VGPR, 0, static_cast<uint8_t>(width)}));
    // The comparison half remains live through the atomic return.
    EXPECT_FALSE(
        def_use.defs.contains(RegisterRef{RegClass::VGPR, static_cast<uint16_t>(width), 1}));
  }
}

TEST_P(AtomicPolicyExecutionTest, MemoryFloatCompareStore) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA1 && GetParam() != ROCJITSU_CODE_ARCH_RDNA2 &&
      GetParam() != ROCJITSU_CODE_ARCH_RDNA3 && GetParam() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "This witness covers RDNA1 through RDNA3.5 memory compare-swap";
  for (bool buffer : {false, true}) {
    EXPECT_EQ(execute_memory(buffer, 0, 0x3f800000, 0xf0, true, 0x80000000), 0x3f800000u);
    EXPECT_EQ(execute_memory(buffer, 0x7fc00002, 0x3f800000, 0xf0, true, 0x7fc00002), 0x7fc00002u);
    EXPECT_EQ(execute_memory(buffer, 1, 0x3f800000, 0, true, 0), 0x3f800000u);
  }
}

TEST_P(AtomicPolicyExecutionTest, DsF64AddPreservesDenormalsAndRoundsToNearest) {
  if (GetParam() != ROCJITSU_CODE_ARCH_CDNA2 && GetParam() != ROCJITSU_CODE_ARCH_CDNA3 &&
      GetParam() != ROCJITSU_CODE_ARCH_CDNA4 && GetParam() != ROCJITSU_CODE_ARCH_CDNA5)
    GTEST_SKIP() << "No F64 LDS add on this ISA";
  const uint16_t add_opcode = modern_policy() ? cdna5::kDsAddF64Vds : cdna4::kDsAddF64Ds;
  for (uint16_t opcode : {add_opcode, static_cast<uint16_t>(add_opcode + 32)}) {
    EXPECT_EQ(execute_ds(opcode, 1, 1, 0, 0, 8), 2u);
    EXPECT_EQ(execute_ds(opcode, 0x7ff0000000000000ULL, 0xfff0000000000000ULL, 0, 0, 8),
              0xfff8000000000000ULL);
    EXPECT_EQ(execute_ds(opcode, 0xfff0000000000000ULL, 0x7ff0000000000000ULL, 0, 0, 8),
              0xfff8000000000000ULL);
    const int saved_round = std::fegetround();
    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
    const uint64_t result =
        execute_ds(opcode, 0x3ff0000000000000ULL, 0x3ca0000000000000ULL, 0, 0, 8);
    std::fesetround(saved_round);
    EXPECT_EQ(result, 0x3ff0000000000000ULL);
  }
}

INSTANTIATE_TEST_SUITE_P(AllAmdgpu, AtomicPolicyExecutionTest,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4,
                                         ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA1,
                                         ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
                                         ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4));
} // namespace
