// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file translate_test.cpp
/// @brief CPU-only unit tests for the DBT translation pipeline.
///
/// Tests encoding correctness, legalization table integrity, and structural
/// properties of translated code objects — without requiring a GPU. Covers:
///   - Coherency bit remapping (GFX940→GFX12, GFX9→GFX12)
///   - Encoding field preservation across SOP1/SOP2/SOPP/SMEM/VOP3 formats
///   - Decode-encode round-trip for CDNA4→RDNA4
///   - Legalization table lookup and zero-ILLEGAL invariant across all ISA pairs
///   - Waitcnt decode/encode (GFX9 monolithic → GFX12 split counters)
///   - E2E binary translation: vector_add code object → translated ELF
///     with valid RDNA4 instructions, correct ELF flags, no GFX9 waitcnt
///
/// These tests complement the hardware tests in hsa_translate_test.cpp which
/// verify correctness on real RDNA4 GPUs.

#include "rocjitsu/code/dbt/encoding_translator.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/encoding_fields.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_cdna2.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna1.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna2.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna3_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna3_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna3_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_rdna2.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna2_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna2_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna3_5_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna3_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna3_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna4_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

TEST(CoherencyRemap, Gfx940ToGfx12AgentScope) {
  auto coh = remap_gfx940_to_gfx12({1, 0, 0});
  EXPECT_EQ(coh.scope, 1);
  EXPECT_EQ(coh.th, 0);
}

TEST(CoherencyRemap, Gfx940ToGfx12SystemScope) {
  auto coh = remap_gfx940_to_gfx12({1, 1, 0});
  EXPECT_EQ(coh.scope, 3);
  EXPECT_EQ(coh.th, 0);
}

TEST(CoherencyRemap, Gfx940ToGfx12NonTemporal) {
  auto coh = remap_gfx940_to_gfx12({0, 0, 1});
  EXPECT_EQ(coh.scope, 0);
  EXPECT_EQ(coh.th, 3);
}

TEST(CoherencyRemap, Gfx9GlcToGfx12) {
  auto coh_glc1 = remap_gfx9_to_gfx12({1});
  EXPECT_EQ(coh_glc1.scope, 2);
  EXPECT_EQ(coh_glc1.th, 0);

  auto coh_glc0 = remap_gfx9_to_gfx12({0});
  EXPECT_EQ(coh_glc0.scope, 0);
  EXPECT_EQ(coh_glc0.th, 0);
}

TEST(EncodingTranslator, Sop1PreservesRegisters) {
  cdna4::Sop1MachineInst src{};
  src.ssrc0 = 42;
  src.sdst = 17;
  src.op = 3;
  src.encoding = 0x17D;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOP1, w0, 0, 0, 5);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop1MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 42);
  EXPECT_EQ(dst.sdst, 17);
  EXPECT_EQ(dst.op, 5);
  EXPECT_EQ(dst.encoding, 0x17D);
}

TEST(EncodingTranslator, Sop2PreservesRegisters) {
  cdna4::Sop2MachineInst src{};
  src.ssrc0 = 10;
  src.ssrc1 = 20;
  src.sdst = 30;
  src.op = 7;
  src.encoding = 0x2;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOP2, w0, 0, 0, 7);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop2MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 10);
  EXPECT_EQ(dst.ssrc1, 20);
  EXPECT_EQ(dst.sdst, 30);
  EXPECT_EQ(dst.op, 7);
}

TEST(InstructionBuilder, Sop2SetsEncodingPrefix) {
  const uint32_t word = build_s_lshl_b32(1, 2, 3, ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_EQ((word >> 30) & 0x3u, 0x2u);
}

TEST(EncodingTranslator, SoppPreservesSimm16) {
  cdna4::SoppMachineInst src{};
  src.simm16 = 0xABCD;
  src.op = 12;
  src.encoding = 0x17F;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOPP, w0, 0, 0, 12);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::SoppMachineInst>(result.words[0]);
  EXPECT_EQ(dst.simm16, 0xABCD);
  EXPECT_EQ(dst.op, 12);
}

TEST(EncodingTranslator, SmemRemapsCoherency) {
  cdna4::SmemMachineInst src{};
  src.sbase = 5;
  src.sdata = 3;
  src.glc = 1;
  src.nv = 0;
  src.op = 0;
  src.offset = 0x100;
  src.soffset = 0x7F;
  src.encoding = 0x3D;
  uint32_t words[2];
  std::memcpy(words, &src, sizeof(src));

  auto result =
      cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SMEM, words[0], words[1], 0, 0);

  ASSERT_EQ(result.word_count, 2);
  rdna4::SmemMachineInst dst{};
  std::memcpy(&dst, result.words, sizeof(dst));
  EXPECT_EQ(dst.sbase, 5);
  EXPECT_EQ(dst.sdata, 3);
  EXPECT_EQ(dst.scope, 2);
  EXPECT_EQ(dst.th, 0);
  EXPECT_EQ(dst.nv, 0);
  EXPECT_EQ(dst.soffset, 0x7C); // CDNA4 null (0x7F) → RDNA4 null (0x7C)
}

TEST(EncodingTranslator, Vop3PreservesModifiers) {
  cdna4::Vop3MachineInst src{};
  src.vdst = 10;
  src.src0 = 100;
  src.src1 = 200;
  src.src2 = 50;
  src.clamp = 1;
  src.omod = 2;
  src.neg = 5;
  src.abs = 3;
  src.op = 100;
  src.encoding = 0x35;
  uint32_t words[2];
  std::memcpy(words, &src, sizeof(src));

  auto result =
      cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_VOP3, words[0], words[1], 0, 100);

  ASSERT_EQ(result.word_count, 2);
  rdna4::Vop3MachineInst dst{};
  std::memcpy(&dst, result.words, sizeof(dst));
  EXPECT_EQ(dst.vdst, 10);
  EXPECT_EQ(dst.src0, 100);
  EXPECT_EQ(dst.src1, 200);
  EXPECT_EQ(dst.src2, 50);
  EXPECT_EQ(dst.clamp, 1);
  EXPECT_EQ(dst.omod, 2);
  EXPECT_EQ(dst.neg, 5);
  EXPECT_EQ(dst.abs, 3);
}

TEST(EncodingTranslator, UnknownEncodingReturnsEmpty) {
  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(0xFFFF, 0, 0, 0, 0);
  EXPECT_EQ(result.word_count, 0);
}

TEST(EncodingTranslator, DecodeEncodeRoundTrip) {
  cdna4::Sop1MachineInst src{};
  src.ssrc0 = 55;
  src.sdst = 33;
  src.op = 4;
  src.encoding = 0x17D;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto fields = cdna4_to_rdna4::decode_sop1_cdna4(w0);
  EXPECT_EQ(fields.ssrc0, 55u);
  EXPECT_EQ(fields.sdst, 33u);
  EXPECT_EQ(fields.op, 4u);

  auto result = cdna4_to_rdna4::encode_sop1_rdna4(fields, 4);
  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop1MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 55);
  EXPECT_EQ(dst.sdst, 33);
  EXPECT_EQ(dst.op, 4);
}

TEST(LegalizationLookup, FindsKnownInstruction) {
  const auto *entry = lookup(kLegalization_cdna4_to_rdna4, 0, 0);
  EXPECT_NE(entry, nullptr);
  if (entry) {
    EXPECT_NE(entry->action, Action::Illegal);
  }
}

TEST(LegalizationLookup, ReturnsNullForUnknown) {
  const auto *entry = lookup(kLegalization_cdna4_to_rdna4, 0xFFFF, 0xFFFF);
  EXPECT_EQ(entry, nullptr);
}

TEST(LegalizationTable, NoIllegalEntries_Cdna4ToRdna4) {
  for (const auto &e : kLegalization_cdna4_to_rdna4) {
    EXPECT_NE(e.action, Action::Illegal)
        << "ILLEGAL at encoding_id=" << e.src_encoding_id << " opcode=" << e.src_opcode;
  }
}

#define CHECK_NO_ILLEGAL(pair)                                                                     \
  TEST(LegalizationTable, NoIllegalEntries_##pair) {                                               \
    for (const auto &e : kLegalization_##pair) {                                                   \
      EXPECT_NE(e.action, Action::Illegal)                                                         \
          << "ILLEGAL at encoding_id=" << e.src_encoding_id << " opcode=" << e.src_opcode;         \
    }                                                                                              \
    EXPECT_GT(std::size(kLegalization_##pair), 0u) << "table is empty";                            \
  }

CHECK_NO_ILLEGAL(cdna1_to_cdna2)
CHECK_NO_ILLEGAL(cdna1_to_cdna3)
CHECK_NO_ILLEGAL(cdna1_to_cdna4)
CHECK_NO_ILLEGAL(cdna1_to_rdna1)
CHECK_NO_ILLEGAL(cdna1_to_rdna2)
CHECK_NO_ILLEGAL(cdna1_to_rdna3)
CHECK_NO_ILLEGAL(cdna1_to_rdna4)
CHECK_NO_ILLEGAL(cdna2_to_cdna3)
CHECK_NO_ILLEGAL(cdna2_to_cdna4)
CHECK_NO_ILLEGAL(cdna2_to_rdna3)
CHECK_NO_ILLEGAL(cdna2_to_rdna4)
CHECK_NO_ILLEGAL(cdna3_to_cdna4)
CHECK_NO_ILLEGAL(cdna3_to_rdna3)
CHECK_NO_ILLEGAL(cdna3_to_rdna4)
CHECK_NO_ILLEGAL(cdna4_to_rdna3)
CHECK_NO_ILLEGAL(rdna1_to_cdna3)
CHECK_NO_ILLEGAL(rdna1_to_cdna4)
CHECK_NO_ILLEGAL(rdna1_to_rdna2)
CHECK_NO_ILLEGAL(rdna1_to_rdna3)
CHECK_NO_ILLEGAL(rdna1_to_rdna4)
CHECK_NO_ILLEGAL(rdna2_to_rdna3)
CHECK_NO_ILLEGAL(rdna2_to_rdna4)
CHECK_NO_ILLEGAL(rdna3_5_to_rdna4)
CHECK_NO_ILLEGAL(rdna3_to_cdna4)
CHECK_NO_ILLEGAL(rdna3_to_rdna4)
CHECK_NO_ILLEGAL(rdna4_to_cdna4)

#undef CHECK_NO_ILLEGAL

} // namespace
} // namespace rocjitsu

// --- WaitcntTranslator tests ---
#include "rocjitsu/code/dbt/semantic_translator.h"

using rocjitsu::decode_waitcnt_gfx9;
using rocjitsu::encode_waitcnt_gfx12;
using rocjitsu::WaitcntValues;

TEST(WaitcntTranslator, DecodeVmcnt0) {
  auto v = decode_waitcnt_gfx9(0x0000);
  EXPECT_EQ(v.vmcnt, 0);
  EXPECT_EQ(v.lgkmcnt, 0);
  EXPECT_EQ(v.expcnt, 0);
}

TEST(WaitcntTranslator, DecodeAllRelaxed) {
  auto v = decode_waitcnt_gfx9(0xCF7F);
  EXPECT_EQ(v.vmcnt, 0x3F);
  EXPECT_EQ(v.lgkmcnt, 0x0F);
  EXPECT_EQ(v.expcnt, 0x07);
}

TEST(WaitcntTranslator, DecodeVmcnt15Lgkm0) {
  uint16_t simm16 = 0x000F;
  auto v = decode_waitcnt_gfx9(simm16);
  EXPECT_EQ(v.vmcnt, 15);
  EXPECT_EQ(v.lgkmcnt, 0);
  EXPECT_EQ(v.expcnt, 0);
}

TEST(WaitcntTranslator, EncodeAllZeroProducesMultipleWords) {
  WaitcntValues v{0, 0, 0};
  auto words = encode_waitcnt_gfx12(v);
  EXPECT_GE(words.size(), 3u);
}

TEST(WaitcntTranslator, EncodeAllRelaxedProducesNop) {
  WaitcntValues v{0x3F, 0x0F, 0x07};
  auto words = encode_waitcnt_gfx12(v);
  ASSERT_EQ(words.size(), 1u);
  uint8_t op = (words[0] >> 16) & 0x7F;
  EXPECT_EQ(op, 0);
}

TEST(WaitcntTranslator, EncodeVmcnt0EmitsLoadcntAndStorecnt) {
  WaitcntValues v{0, 0x0F, 0x07};
  auto words = encode_waitcnt_gfx12(v);
  EXPECT_GE(words.size(), 2u);

  bool has_loadcnt = false;
  bool has_storecnt_dscnt = false;
  for (auto w : words) {
    uint8_t op = (w >> 16) & 0x7F;
    if (op == 64)
      has_loadcnt = true;
    if (op == 73)
      has_storecnt_dscnt = true;
  }
  EXPECT_TRUE(has_loadcnt);
  EXPECT_TRUE(has_storecnt_dscnt);
}

// --- End-to-end BinaryTranslator integration tests ---
#ifdef HAS_DEVICE_KERNELS

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

namespace {

std::string kernel_path(const char *name) { return std::string(KERNEL_DIR) + "/" + name + ".o"; }

struct MutableKernelDescriptorImage {
  std::vector<uint8_t> image;
  uint64_t text_offset = 0;
  uint64_t text_size = 0;
  uint64_t kd_file_off = 0;
  bool valid = false;
};

MutableKernelDescriptorImage mutable_vector_add_descriptor(rj_code_arch_t guest_arch,
                                                           rj_code_arch_t host_arch) {
  MutableKernelDescriptorImage fixture;
  rocjitsu::Executable exec(kernel_path("vector_add"));
  if (!exec.is_valid() || exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950) == 0)
    return fixture;

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  if (co == nullptr || co->text_sections().empty())
    return fixture;

  fixture.image.resize(co->image_size());
  std::memcpy(fixture.image.data(), co->image_data(), fixture.image.size());
  fixture.text_offset = co->text_sections()[0]->sectionOffset();
  fixture.text_size = co->text_sections()[0]->size();

  rocjitsu::KernelDescriptorTranslator translator(guest_arch, host_arch);
  const auto translations =
      translator.translate_image(fixture.image, fixture.text_offset, fixture.text_size,
                                 rocjitsu::KernelDescriptorTranslationOptions{});
  if (translations.empty())
    return fixture;

  fixture.kd_file_off = translations[0].descriptor_file_offset;
  fixture.valid =
      fixture.kd_file_off + sizeof(rocr::llvm::amdhsa::kernel_descriptor_t) <= fixture.image.size();
  return fixture;
}

rocr::llvm::amdhsa::kernel_descriptor_t *
mutable_kernel_descriptor(MutableKernelDescriptorImage &fixture) {
  return reinterpret_cast<rocr::llvm::amdhsa::kernel_descriptor_t *>(fixture.image.data() +
                                                                     fixture.kd_file_off);
}

std::vector<rocjitsu::KdTranslation>
translate_mutable_descriptor(const MutableKernelDescriptorImage &fixture, rj_code_arch_t guest_arch,
                             rj_code_arch_t host_arch) {
  rocjitsu::KernelDescriptorTranslator translator(guest_arch, host_arch);
  return translator.translate_image(fixture.image, fixture.text_offset, fixture.text_size,
                                    rocjitsu::KernelDescriptorTranslationOptions{});
}

} // namespace

using rocjitsu::BinaryTranslator;
using rocjitsu::Decoder;
using rocjitsu::Executable;

TEST(BinaryTranslatorE2E, TranslateVectorAddCdna4ToRdna4) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load vector_add.o";
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(*co);

  EXPECT_FALSE(result.elf_bytes.empty()) << "Translation produced empty ELF";
  EXPECT_EQ(result.host_arch, ROCJITSU_CODE_ARCH_RDNA4);

  // Verify ELF machine flags contain GFX1200.
  ASSERT_GE(result.elf_bytes.size(), 48u);
  uint32_t e_flags = 0;
  std::memcpy(&e_flags, result.elf_bytes.data() + 48, sizeof(e_flags));
  constexpr uint32_t kEfAmdgpuMachGfx1200 = 0x48;
  EXPECT_EQ(e_flags & 0xFF, kEfAmdgpuMachGfx1200)
      << "ELF e_flags should contain GFX1200 machine type";
}

TEST(KernelDescriptorTranslator, Cdna4ToRdna4MaterializesWorkgroupIdsFromTtmpGridPayload) {
  using namespace rocr::llvm::amdhsa;

  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  std::vector<uint8_t> image(co->image_size());
  std::memcpy(image.data(), co->image_data(), image.size());

  ASSERT_FALSE(co->text_sections().empty());
  const auto *original_text = co->text_sections()[0];
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_RDNA4);
  const auto original_translations =
      translator.translate_image(image, original_text->sectionOffset(), original_text->size(),
                                 rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_FALSE(original_translations.empty());
  const uint64_t kd_file_off = original_translations[0].descriptor_file_offset;
  ASSERT_LE(kd_file_off + sizeof(kernel_descriptor_t), image.size());

  auto *kd = reinterpret_cast<kernel_descriptor_t *>(image.data() + kd_file_off);
  kd->compute_pgm_rsrc2 = 0;
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 12);
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1);
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y, 1);
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z, 1);

  rocjitsu::AmdGpuCodeObject mutated(image.data(), image.size());
  ASSERT_TRUE(mutated.is_valid());
  ASSERT_FALSE(mutated.text_sections().empty());
  const auto *text = mutated.text_sections()[0];

  const auto translations = translator.translate_image(
      image, text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  const auto translated = std::find_if(translations.begin(), translations.end(),
                                       [kd_file_off](const auto &translation) {
                                         return translation.descriptor_file_offset == kd_file_off;
                                       });
  ASSERT_NE(translated, translations.end());

  constexpr uint16_t ttmp_base = 108;
  const uint16_t shift16 = rocjitsu::scalar_positive_inline_u32(16);
  const std::vector<uint32_t> expected = {
      rocjitsu::build_s_mov_b32(12, ttmp_base + 9, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_mov_b32(13, ttmp_base + 7, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_lshl_b32(13, 13, shift16, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_lshr_b32(13, 13, shift16, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_lshr_b32(14, ttmp_base + 7, shift16, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
  };
  EXPECT_EQ(translated->prologue_words, expected)
      << "CDNA workgroup_id SGPRs must be rebuilt from RDNA4 TTMP9 and packed TTMP7";
}

TEST(KernelDescriptorTranslator, Cdna4ToRdna4MaterializesXOnlyWorkgroupId) {
  using namespace rocr::llvm::amdhsa;

  auto fixture = mutable_vector_add_descriptor(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(fixture.valid);

  auto *kd = mutable_kernel_descriptor(fixture);
  kd->compute_pgm_rsrc2 = 0;
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 12);
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1);

  const auto translations =
      translate_mutable_descriptor(fixture, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  const auto translated =
      std::find_if(translations.begin(), translations.end(), [&fixture](const auto &translation) {
        return translation.descriptor_file_offset == fixture.kd_file_off;
      });
  ASSERT_NE(translated, translations.end());

  constexpr uint16_t ttmp_base = 108;
  const std::vector<uint32_t> expected = {
      rocjitsu::build_s_mov_b32(12, ttmp_base + 9, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
  };
  EXPECT_EQ(translated->prologue_words, expected);
}

TEST(KernelDescriptorTranslator, Cdna4ToRdna4SkipsPrologueWhenNoWorkgroupIdsAreEnabled) {
  using namespace rocr::llvm::amdhsa;

  auto fixture = mutable_vector_add_descriptor(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(fixture.valid);

  auto *kd = mutable_kernel_descriptor(fixture);
  kd->compute_pgm_rsrc2 = 0;
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 12);
  const int64_t original_entry = kd->kernel_code_entry_byte_offset;

  const auto translations =
      translate_mutable_descriptor(fixture, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  const auto translated =
      std::find_if(translations.begin(), translations.end(), [&fixture](const auto &translation) {
        return translation.descriptor_file_offset == fixture.kd_file_off;
      });
  ASSERT_NE(translated, translations.end());
  EXPECT_TRUE(translated->prologue_words.empty());

  rocjitsu::AmdGpuCodeObject mutated(fixture.image.data(), fixture.image.size());
  ASSERT_TRUE(mutated.is_valid());
  rocjitsu::CodeObjectPatcher patcher(mutated);
  EXPECT_TRUE(patcher.apply_kernel_descriptor_translation(*translated, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_TRUE(patcher.cave_body().empty());

  const auto patched_image = patcher.emit();
  const auto *patched_kd =
      reinterpret_cast<const kernel_descriptor_t *>(patched_image.data() + fixture.kd_file_off);
  EXPECT_EQ(patched_kd->kernel_code_entry_byte_offset, original_entry);
}

TEST(KernelDescriptorTranslator, CdnaAccVgprExpansionGrowsUnifiedVgprAllocationForRdna4) {
  using namespace rocr::llvm::amdhsa;

  for (rj_code_arch_t guest_arch :
       {ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    auto fixture = mutable_vector_add_descriptor(guest_arch, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(fixture.valid);

    auto *kd = mutable_kernel_descriptor(fixture);
    kd->compute_pgm_rsrc1 = 0;
    kd->compute_pgm_rsrc3 = 0;
    AMDHSA_BITS_SET(kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 7);
    AMDHSA_BITS_SET(kd->compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 15);

    const auto translations =
        translate_mutable_descriptor(fixture, guest_arch, ROCJITSU_CODE_ARCH_RDNA4);
    const auto translated =
        std::find_if(translations.begin(), translations.end(), [&fixture](const auto &translation) {
          return translation.descriptor_file_offset == fixture.kd_file_off;
        });
    ASSERT_NE(translated, translations.end());
    EXPECT_EQ(translated->accvgpr_base, 64u);
    EXPECT_EQ(translated->guest_vgpr_count, 64u);
    EXPECT_EQ(translated->target_vgpr_count, 128u);
    EXPECT_EQ(translated->target_vgpr_granulated, 31u);
  }
}

TEST(KernelDescriptorTranslator, RdnaWave64UsesAmdhsaDescriptorVgprEncoding) {
  using namespace rocr::llvm::amdhsa;

  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);
  ASSERT_FALSE(co->text_sections().empty());

  std::vector<uint8_t> image(co->image_size());
  std::memcpy(image.data(), co->image_data(), image.size());

  const auto *text = co->text_sections()[0];
  rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_RDNA4);
  const auto parsed = parser.translate_image(image, text->sectionOffset(), text->size(),
                                             rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_FALSE(parsed.empty());
  const uint64_t kd_file_off = parsed[0].descriptor_file_offset;
  ASSERT_LE(kd_file_off + sizeof(kernel_descriptor_t), image.size());

  auto *kd = reinterpret_cast<kernel_descriptor_t *>(image.data() + kd_file_off);
  kd->compute_pgm_rsrc1 = 0;
  kd->kernel_code_properties = 0;
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 31);

  // Use CDNA1 as the guest so the source descriptor field means exactly
  // 128 VGPRs without AccVGPR remapping. The target assertions below check the
  // AMDHSA descriptor encoding for RDNA Wave64, not the ISA manual's physical
  // allocation block size. AMDHSA GFX10-GFX12 Wave64 encodes 128 VGPRs as
  // ceil(128 / 4) - 1 == 31; using the RDNA3/RDNA4 physical block size of 8
  // would incorrectly produce 15.
  for (rj_code_arch_t host_arch :
       {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
        ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA1, host_arch);
    const auto translations = translator.translate_image(
        image, text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});
    const auto translated = std::find_if(translations.begin(), translations.end(),
                                         [kd_file_off](const auto &translation) {
                                           return translation.descriptor_file_offset == kd_file_off;
                                         });
    ASSERT_NE(translated, translations.end());
    EXPECT_EQ(translated->target_wave_size, 64);
    EXPECT_EQ(translated->target_vgpr_count, 128u);
    EXPECT_EQ(translated->target_vgpr_granulated, 31u);
  }
}

TEST(BinaryTranslatorE2E, DescriptorPrologueRedirectsEntryWithoutOverwritingOriginalEntry) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  const auto *original_image = reinterpret_cast<const uint8_t *>(co->image_data());
  ASSERT_FALSE(co->text_sections().empty());
  const auto *original_text = co->text_sections()[0];
  rocjitsu::KernelDescriptorTranslator original_parser(ROCJITSU_CODE_ARCH_CDNA4,
                                                       ROCJITSU_CODE_ARCH_RDNA4);
  const auto original_infos = original_parser.translate_image(
      {original_image, co->image_size()}, original_text->sectionOffset(), original_text->size(),
      rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_FALSE(original_infos.empty());
  const rocjitsu::KdTranslation *original_info = &original_infos[0];
  const uint64_t kd_file_off = original_info->descriptor_file_offset;

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated_co(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated_co.is_valid());
  ASSERT_FALSE(translated_co.text_sections().empty());
  const auto *translated_text = translated_co.text_sections()[0];
  const auto *translated_image = reinterpret_cast<const uint8_t *>(translated_co.image_data());
  rocjitsu::KernelDescriptorTranslator translated_parser(ROCJITSU_CODE_ARCH_RDNA4,
                                                         ROCJITSU_CODE_ARCH_RDNA4);
  const auto translated_infos = translated_parser.translate_image(
      {translated_image, translated_co.image_size()}, translated_text->sectionOffset(),
      translated_text->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  const auto translated_info = std::find_if(
      translated_infos.begin(), translated_infos.end(),
      [kd_file_off](const auto &info) { return info.descriptor_file_offset == kd_file_off; });
  ASSERT_NE(translated_info, translated_infos.end());

  EXPECT_GT(translated_info->entry_text_offset, original_info->entry_text_offset)
      << "CDNA4 workgroup-id SGPRs must be materialized from RDNA4's TTMP launch payload";
  EXPECT_GE(translated_info->guest_vgpr_count, 128u)
      << "DBT semantic lowerings need conservative temporary VGPR headroom in the descriptor";

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  ASSERT_FALSE(translated_co.text_sections().empty());
  const auto *text = translated_co.text_sections()[0];
  const auto *words = reinterpret_cast<const uint32_t *>(text->data());

  ASSERT_EQ(original_info->entry_text_offset % sizeof(uint32_t), 0u);
  ASSERT_EQ(translated_info->entry_text_offset % sizeof(uint32_t), 0u);
  ASSERT_LT(original_info->entry_text_offset, text->size());
  ASSERT_LT(translated_info->entry_text_offset, text->size());

  std::unique_ptr<rocjitsu::Instruction> original_entry(
      decoder->decode(&words[original_info->entry_text_offset / sizeof(uint32_t)]));
  ASSERT_NE(original_entry, nullptr);
  EXPECT_NE(std::string_view(original_entry->mnemonic()), "s_branch")
      << "Original kernel entry should not be replaced by a prologue branch stub";

  std::unique_ptr<rocjitsu::Instruction> redirected_entry(
      decoder->decode(&words[translated_info->entry_text_offset / sizeof(uint32_t)]));
  ASSERT_NE(redirected_entry, nullptr);
  EXPECT_EQ(std::string_view(redirected_entry->mnemonic()), "s_mov_b32")
      << "Redirected kernel entry should begin with the descriptor ABI prologue";
}

TEST(CodeObjectPatcher, KernelEntryPrologueIs256ByteAligned) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  rocjitsu::CodeObjectPatcher patcher(*co);
  ASSERT_FALSE(co->text_sections().empty());
  const auto *image = reinterpret_cast<const uint8_t *>(co->image_data());
  const auto *text = co->text_sections()[0];
  rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  const auto infos =
      parser.translate_image({image, co->image_size()}, text->sectionOffset(), text->size(),
                             rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_FALSE(infos.empty());

  patcher.set_cave_start(infos[0].entry_text_offset + sizeof(uint32_t));
  const std::array<uint32_t, 1> prologue = {rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4)};
  const auto prologue_entry = patcher.append_kernel_entry_prologue(
      infos[0].entry_text_offset, prologue, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(prologue_entry.has_value());

  EXPECT_EQ(*prologue_entry % 256, infos[0].entry_text_offset % 256)
      << "Kernel descriptor entry points are hardware launch addresses; the cave prologue must "
         "preserve the original entry's .text-relative alignment residue";
}

TEST(CodeObjectPatcher, KernelEntryPrologueRejectsOutOfRangeBranch) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  rocjitsu::CodeObjectPatcher patcher(*co);
  ASSERT_FALSE(co->text_sections().empty());
  const auto *image = reinterpret_cast<const uint8_t *>(co->image_data());
  const auto *text = co->text_sections()[0];
  rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  const auto infos =
      parser.translate_image({image, co->image_size()}, text->sectionOffset(), text->size(),
                             rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_FALSE(infos.empty());

  patcher.set_cave_start(infos[0].entry_text_offset + 200000);
  const std::array<uint32_t, 1> prologue = {rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4)};
  const auto prologue_entry = patcher.append_kernel_entry_prologue(
      infos[0].entry_text_offset, prologue, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_FALSE(prologue_entry.has_value());
  EXPECT_EQ(patcher.cave_body_size(), 0u);
}

TEST(CodeObjectPatcher, RejectsOutOfRangeKernelDescriptorUpdates) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  rocjitsu::CodeObjectPatcher patcher(*co);
  std::array<uint8_t, sizeof(rocr::llvm::amdhsa::kernel_descriptor_t)> descriptor{};
  const uint64_t image_size = static_cast<uint64_t>(co->image_size());

  EXPECT_FALSE(patcher.patch_kernel_descriptor(image_size, descriptor));
  EXPECT_FALSE(patcher.patch_kernel_descriptor(image_size - 1, descriptor));

  rocjitsu::KdTranslation invalid;
  invalid.descriptor_file_offset = image_size;
  EXPECT_FALSE(patcher.apply_kernel_descriptor_translation(invalid, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_TRUE(patcher.cave_body().empty());
}

TEST(BinaryTranslatorE2E, CaveOverflowLeavesCodeObjectUnchanged) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);
  ASSERT_FALSE(co->text_sections().empty());

  std::vector<uint8_t> image(co->image_size());
  std::memcpy(image.data(), co->image_data(), image.size());

  const auto *text = co->text_sections()[0];
  uint8_t *text_bytes = image.data() + text->sectionOffset();
  const size_t word_count = text->size() / sizeof(uint32_t);
  const uint32_t nop = rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);
  const uint32_t filler = rocjitsu::build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4);

  size_t overwritten_padding_words = 0;
  for (size_t i = word_count; i > 0; --i) {
    uint32_t word = 0;
    std::memcpy(&word, text_bytes + (i - 1) * sizeof(uint32_t), sizeof(word));
    if (word != nop)
      break;
    std::memcpy(text_bytes + (i - 1) * sizeof(uint32_t), &filler, sizeof(filler));
    ++overwritten_padding_words;
  }
  ASSERT_GT(overwritten_padding_words, 0u);

  rocjitsu::AmdGpuCodeObject no_padding(image.data(), image.size());
  ASSERT_TRUE(no_padding.is_valid());

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(no_padding);

  EXPECT_EQ(result.elf_bytes, image);
  const bool warned =
      std::any_of(result.warnings.begin(), result.warnings.end(), [](const std::string &warning) {
        return warning.find("cave body") != std::string::npos &&
               warning.find("leaving code object unchanged") != std::string::npos;
      });
  EXPECT_TRUE(warned);
}

TEST(BinaryTranslatorE2E, OutputDecodesAsValidRdna4) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());

  // Construct an RDNA4 code object from the translated ELF bytes.
  rocjitsu::AmdGpuCodeObject translated_co(result.elf_bytes.data(), result.elf_bytes.size());

  // Decode every instruction with the RDNA4 decoder.
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);

  int decode_failures = 0;
  int inst_count = 0;
  for (const auto *sec : translated_co.text_sections()) {
    const auto *data = reinterpret_cast<const uint32_t *>(sec->data());
    const size_t words = sec->size() / sizeof(uint32_t);
    size_t pc = 0;
    while (pc < words) {
      try {
        std::unique_ptr<rocjitsu::Instruction> inst(decoder->decode(&data[pc]));
        if (!inst) {
          ++decode_failures;
          ++pc;
          continue;
        }
        pc += inst->size() / 4;
        ++inst_count;
      } catch (const std::exception &e) {
        std::cerr << "  decode fail at 0x" << std::hex << pc * 4 << " word=0x" << data[pc] << ": "
                  << e.what() << "\n";
        ++decode_failures;
        ++pc;
      }
    }
  }
  EXPECT_GT(inst_count, 0) << "Text section should contain instructions";
  EXPECT_EQ(decode_failures, 0) << decode_failures << " instructions failed to decode as RDNA4";
}

TEST(BinaryTranslatorE2E, NoGfx9WaitcntInOutput) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated_co(result.elf_bytes.data(), result.elf_bytes.size());
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);

  for (const auto *sec : translated_co.text_sections()) {
    const auto *data = reinterpret_cast<const uint32_t *>(sec->data());
    const size_t words = sec->size() / sizeof(uint32_t);
    size_t pc = 0;
    while (pc < words) {
      try {
        std::unique_ptr<rocjitsu::Instruction> inst(decoder->decode(&data[pc]));
        if (!inst) {
          ++pc;
          continue;
        }
        EXPECT_NE(std::string_view(inst->mnemonic()), "s_waitcnt")
            << "GFX9 s_waitcnt found in translated output at offset 0x" << std::hex << pc * 4;
        pc += inst->size() / 4;
      } catch (...) {
        ++pc;
      }
    }
  }
}

TEST(BinaryTranslatorE2E, TextSizesMatch) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  const size_t original_text_size =
      co->text_sections().empty() ? 0 : co->text_sections()[0]->size();

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated_co(result.elf_bytes.data(), result.elf_bytes.size());
  const size_t translated_text_size =
      translated_co.text_sections().empty() ? 0 : translated_co.text_sections()[0]->size();

  // The translated .text must be at least as large as the original
  // (code caves may grow it, but individual instructions never shift).
  EXPECT_GE(translated_text_size, original_text_size);
}

TEST(BinaryTranslatorE2E, WriteTranslatedElfToFile) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());

  // Write a GFX1201 variant (same ISA, different MACH flag).
  auto elf_1201 = result.elf_bytes;
  // Patch e_flags: clear low 8 bits (MACH), set GFX1201 = 0x4E.
  uint32_t e_flags = 0;
  std::memcpy(&e_flags, elf_1201.data() + 48, 4);
  e_flags = (e_flags & ~0xFFu) | 0x4E;
  std::memcpy(elf_1201.data() + 48, &e_flags, 4);

  const char *out_path = "/tmp/vector_add_gfx1201.co";
  FILE *f = fopen(out_path, "wb");
  ASSERT_NE(f, nullptr);
  fwrite(elf_1201.data(), 1, elf_1201.size(), f);
  fclose(f);
  printf("  Wrote translated ELF to %s (%zu bytes)\n", out_path, elf_1201.size());
}

TEST(BinaryTranslatorE2E, DumpTranslation) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto dump = [](const char *label, const uint8_t *text, size_t size, rj_code_arch_t arch) {
    auto dec = Decoder::create(arch);
    if (!dec)
      return;
    const auto *data = reinterpret_cast<const uint32_t *>(text);
    size_t words = size / 4, pc = 0;
    printf("\n--- %s (%zu bytes, %zu words) ---\n", label, size, words);
    while (pc < words) {
      try {
        std::unique_ptr<rocjitsu::Instruction> inst(dec->decode(&data[pc]));
        if (!inst) {
          printf("  0x%04zx: ???\n", pc * 4);
          ++pc;
          continue;
        }
        printf("  0x%04zx: %-45s [", pc * 4, inst->disassemble().c_str());
        for (int i = 0; i < inst->size() / 4; i++)
          printf("%s%08X", i ? " " : "", data[pc + i]);
        printf("]\n");
        pc += inst->size() / 4;
      } catch (...) {
        printf("  0x%04zx: [decode error] 0x%08X\n", pc * 4, data[pc]);
        ++pc;
      }
    }
  };

  for (const auto *sec : co->text_sections())
    dump("CDNA4 source", reinterpret_cast<const uint8_t *>(sec->data()), sec->size(),
         ROCJITSU_CODE_ARCH_CDNA4);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  for (const auto *sec : translated.text_sections())
    dump("RDNA4 translated", reinterpret_cast<const uint8_t *>(sec->data()), sec->size(),
         ROCJITSU_CODE_ARCH_RDNA4);

  if (!result.warnings.empty()) {
    printf("\n--- Warnings (%zu) ---\n", result.warnings.size());
    for (const auto &w : result.warnings)
      printf("  %s\n", w.c_str());
  }
}

#endif // HAS_DEVICE_KERNELS
