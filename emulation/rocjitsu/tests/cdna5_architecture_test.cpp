// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"

namespace {

using namespace rocjitsu;
using namespace rocjitsu::test::cdna5;

template <typename T> void append_bytes(std::vector<uint8_t> &bytes, const T &value) {
  auto *src = reinterpret_cast<const uint8_t *>(&value);
  bytes.insert(bytes.end(), src, src + sizeof(T));
}

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

std::vector<uint8_t> make_minimal_gfx1250_elf() {
  constexpr uint8_t text[] = {0x00, 0x00, 0xB0, 0xBF};
  constexpr char shstrtab[] = "\0.text\0.shstrtab\0";
  constexpr uint32_t text_name = 1;
  constexpr uint32_t shstrtab_name = 7;

  std::vector<uint8_t> image(sizeof(Elf64_Ehdr), 0);
  const size_t text_offset = image.size();
  image.insert(image.end(), std::begin(text), std::end(text));
  const size_t shstrtab_offset = image.size();
  image.insert(image.end(), std::begin(shstrtab), std::end(shstrtab));
  image.resize(align_up(image.size(), alignof(Elf64_Shdr)), 0);
  const size_t shoff = image.size();

  Elf64_Shdr null_shdr{};
  append_bytes(image, null_shdr);

  Elf64_Shdr text_shdr{};
  text_shdr.sh_name = text_name;
  text_shdr.sh_type = SHT_PROGBITS;
  text_shdr.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  text_shdr.sh_offset = text_offset;
  text_shdr.sh_size = sizeof(text);
  text_shdr.sh_addralign = alignof(uint32_t);
  append_bytes(image, text_shdr);

  Elf64_Shdr shstrtab_shdr{};
  shstrtab_shdr.sh_name = shstrtab_name;
  shstrtab_shdr.sh_type = SHT_STRTAB;
  shstrtab_shdr.sh_offset = shstrtab_offset;
  shstrtab_shdr.sh_size = sizeof(shstrtab);
  shstrtab_shdr.sh_addralign = 1;
  append_bytes(image, shstrtab_shdr);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_DATA] = 1;
  ehdr.e_ident[EI_VERSION] = 1;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  ehdr.e_type = ET_REL;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = 3;
  ehdr.e_shstrndx = 2;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));
  return image;
}

TEST(Gfx1250ConfigTest, ConfigLoadsTopology) {
  auto loaded = config::load_config(kGfx1250ConfigPath, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  ASSERT_NE(soc, nullptr);
  EXPECT_EQ(soc->arch(), ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(config::parse_arch("cdna5"), ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_STREQ(config::arch_to_string(ROCJITSU_CODE_ARCH_CDNA5), "cdna5");

  EXPECT_TRUE(loaded.device.present);
  EXPECT_EQ(loaded.device.gfx_target_version, 120500u);
  EXPECT_EQ(loaded.device.marketing_name, "AMD Instinct MI455X");
  EXPECT_EQ(loaded.device.simd_count, 1024u);
  EXPECT_EQ(loaded.device.max_waves_per_simd, kGfx1250MaxWavesPerSimd);
  EXPECT_EQ(loaded.device.num_shader_engines, 2u);
  EXPECT_EQ(loaded.device.num_shader_arrays_per_engine, 2u);
  EXPECT_EQ(loaded.device.num_cu_per_sh, 8u);
  EXPECT_EQ(loaded.device.simd_per_cu, kGfx1250SimdsPerCu);
  EXPECT_EQ(loaded.device.wave_front_size, 32u);
  EXPECT_EQ(loaded.device.local_mem_size, kGfx1250HbmBytes);
  EXPECT_EQ(loaded.device.lds_size_kb, kGfx1250LdsSizeKb);
  EXPECT_EQ(loaded.device.mem_width, kGfx1250HbmWidthBits);
  EXPECT_EQ(loaded.device.l1_size_kb, kGfx1250VectorCacheSizeKb);
  EXPECT_EQ(loaded.device.l2_size_kb, kGfx1250L2SizeKb);

  EXPECT_EQ(soc->num_xcds(), 8u);
  EXPECT_EQ(soc->num_iods(), 2u);
  EXPECT_EQ(soc->iod(0)->req_ports().size(), 6u);
  EXPECT_EQ(soc->iod(1)->req_ports().size(), 6u);
  EXPECT_EQ(soc->xcd(0)->num_shader_engines(), 2u);
  EXPECT_EQ(soc->xcd(0)->shader_engine(0)->num_compute_units(), 16u);
  // num_shader_engines is the shader-engine count itself, not the shader-array
  // count that has to be divided down: KFD's node_props.array_count is derived
  // from it as engines * arrays_per_engine, which is the product libhsakmt and
  // rocdbgapi invert to recover the engine count. A shader engine still holds
  // arrays_per_engine * num_cu_per_sh compute units.
  EXPECT_EQ(loaded.device.num_shader_engines, soc->xcd(0)->num_shader_engines());
  EXPECT_EQ(loaded.device.num_shader_arrays_per_engine * loaded.device.num_cu_per_sh,
            soc->xcd(0)->shader_engine(0)->num_compute_units());
  EXPECT_EQ(soc->num_xcds() * soc->xcd(0)->num_shader_engines() *
                soc->xcd(0)->shader_engine(0)->num_compute_units() * loaded.device.simd_per_cu,
            loaded.device.simd_count);
  auto *cu = soc->xcd(0)->shader_engine(0)->compute_unit(0);
  ASSERT_NE(cu, nullptr);
  EXPECT_EQ(cu->wf_size(), 32u);
  EXPECT_EQ(cu->config().num_wf_slots, kGfx1250WaveSlotsPerCu);
  EXPECT_EQ(cu->config().sgprs_per_wf, kGfx1250ScalarSlots);
  EXPECT_EQ(cu->config().vgprs_per_wf, kGfx1250Wave32VgprAllocation);
  EXPECT_EQ(cu->config().lds_size_kb, kGfx1250LdsSizeKb);
  EXPECT_EQ(soc->xcd(0)->command_processor()->vgpr_granularity(), kGfx1250VgprEncodingGranule);
  EXPECT_EQ(soc->xcd(0)->command_processor()->sdma_packet_dialect(),
            amdgpu::SdmaPacketDialect::Gfx1250);
}

TEST(Gfx1250CodeObjectTest, MachineFlagMapsToTarget) {
  auto image = make_minimal_gfx1250_elf();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  EXPECT_EQ(co.target_id(), ROCJITSU_CODE_TARGET_GFX1250);
  ASSERT_EQ(co.text_sections().size(), 1u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  const auto *text = co.text_sections()[0];
  auto *words = reinterpret_cast<const uint32_t *>(text->data());
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "s_endpgm");
}

TEST(Gfx1250DecodeTest, SMovB64Literal64ConsumesThreeDwords) {
  const uint32_t words[] = {
      0xBEB801FEu, // s_mov_b64 s[56:57], literal64
      0xFFFFFF80u,
      0xFFFFFFFFu,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "s_mov_b64");
  EXPECT_EQ(inst->size(), sizeof(words));
  ASSERT_NE(inst->raw_encoding(), nullptr);
  EXPECT_EQ(inst->raw_encoding()[0], words[0]);
  EXPECT_EQ(inst->raw_encoding()[1], words[1]);
  EXPECT_EQ(inst->raw_encoding()[2], words[2]);
}

TEST(Gfx1250DecodeTest, ScalarSourceRejectsReservedSelector) {
  const uint32_t words[] = {
      0x8C9000E2u, // s_or_b64 s[16:17], reserved selector 226, s0
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(words)), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, Vop3LiteralConsumesThreeDwords) {
  const uint32_t words[] = {
      0xD6570001u, // v_and_or_b32 v1, 0xf8, v1, v2
      0x040A02FFu,
      0x000000F8u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_and_or_b32");
  EXPECT_EQ(inst->size(), sizeof(words));
}

TEST(Gfx1250DecodeTest, Vop3RejectsLiteral64Selector) {
  const uint32_t words[] = {
      0xD5D50000u, // v_sqrt_f16 v0, reserved literal64 selector
      0x000000FEu,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(words)), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, Vop1RejectsUnsupportedLiteral32WithoutExtensionWord) {
  const auto words = cdna5::build_vop1(cdna5::kVReadfirstlaneB32Vop1, {.src0 = 255});

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(words.data())), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, Vop2RejectsUnsupportedLiteral32WithoutExtensionWord) {
  const auto words = cdna5::build_vop2(cdna5::kVFmamkF64Vop2, {.src0 = 255});

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(words.data())), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, Vop2RejectsUnsupportedLiteral64WithoutExtensionWords) {
  const auto words = cdna5::build_vop2(cdna5::kVFmamkF32Vop2, {.src0 = 254});

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(words.data())), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, SaluRejectsMixedLiteralWidths) {
  const uint32_t words[] = {
      0xBF5DFFFEu, // s_cmp_neq_f16 literal64, literal32
      0x00000000u,
      0x00000000u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(words)), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, SendmsgRtnSelectorsAreNotLiterals) {
  const uint32_t words[] = {
      0xBE804CFFu, // s_sendmsg_rtn_b32 s0, 255
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "s_sendmsg_rtn_b32");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_EQ(inst->src_operand(0)->name(), "255");
}

TEST(Gfx1250DecodeTest, VopdRejectsLiteral64Selector) {
  const uint32_t words[] = {
      0xCA52FFFFu,
      0xFFFFFCFEu,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(words)), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, Vop3RejectsDppWithLiteral) {
  const uint32_t words[] = {
      0xD6290B00u, // v_min3_num_f32 with src0:DPP and src2:literal
      0x83FF00FAu,
      0x00001500u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(words)), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, Vop3RejectsInvalidScalarDestination) {
  const uint32_t words[] = {
      0xD41B10FFu, // v_cmp_ngt_f32 with reserved scalar destination 255
      0x000000FAu,
      0x00000000u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(words)), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, Vop3RejectsInvalidVgprSource) {
  const uint32_t words[] = {
      0xD7600000u, // v_readlane_b32 s0, invalid, null
      0x0000F8D7u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(words)), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, Vop3ReadlaneValidatesLaneSelector) {
  const uint32_t valid_words[] = {
      0xD7600000u, // v_readlane_b32 s0, v215, 1
      0x000103D7u,
  };
  const uint32_t invalid_words[] = {
      0xD7600000u, // v_readlane_b32 with reserved lane selector 491
      0x0003D7D7u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> valid(decoder->decode(valid_words));
  ASSERT_NE(valid, nullptr);
  EXPECT_EQ(valid->disassemble(), "v_readlane_b32 s0, v215, 1");
  EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(invalid_words)), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, Vop3CmpxValidatesExecDestination) {
  const uint32_t valid_words[] = {
      0xD4CD007Eu, // v_cmpx_ne_u32 exec, v0, v1
      0x00020300u,
  };
  const uint32_t invalid_words[] = {
      0xD4CD00F4u, // v_cmpx_ne_u32 with reserved EXEC destination 244
      0x00020300u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> valid(decoder->decode(valid_words));
  ASSERT_NE(valid, nullptr);
  EXPECT_EQ(valid->disassemble(), "v_cmpx_ne_u32 exec, v0, v1");
  EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(invalid_words)), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, Vop3SdstLiteralConsumesThreeDwords) {
  const uint32_t words[] = {
      0xD7020001u, // v_subrev_co_u32 v1, s0, 0x60, s12
      0x020018FFu,
      0x00000060u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_subrev_co_u32");
  EXPECT_EQ(inst->size(), sizeof(words));
}

TEST(Gfx1250DecodeTest, VFmamkF64ImpliedLiteralConsumesThreeDwords) {
  const uint32_t words[] = {
      0x46040504u, // v_fmamk_f64 v[2:3], v[4:5], -30.0, v[2:3]
      0x00000000u, 0xC1F00000u,
      0x7E042B02u, // v_cvt_u32_f64_e32 v2, v[2:3]
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> fmamk(decoder->decode(words));
  ASSERT_NE(fmamk, nullptr);
  EXPECT_EQ(fmamk->mnemonic(), "v_fmamk_f64_e32");
  EXPECT_EQ(fmamk->size(), 3 * sizeof(uint32_t));

  std::unique_ptr<Instruction> next(decoder->decode(words + 3));
  ASSERT_NE(next, nullptr);
  EXPECT_EQ(next->mnemonic(), "v_cvt_u32_f64_e32");
}

TEST(Gfx1250DecodeTest, SWaitXcntHasWaitcntMetadata) {
  const uint32_t words[] = {
      0xBFC50000u, // s_wait_xcnt 0
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "s_wait_xcnt");
  EXPECT_TRUE(inst->is_waitcnt());
  EXPECT_EQ(inst->disassemble(), "s_wait_xcnt 0");
}

TEST(Gfx1250DecodeTest, BufferOffenUsesSingleVaddrRegister) {
  const uint32_t words[] = {
      0xC405C07Cu, // buffer_load_b128 v[32:35], v7, s[4:7], s0 offen
      0x40800820u,
      0x00000007u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "buffer_load_b128");
  ASSERT_EQ(inst->num_dst_operands(), 1);
  ASSERT_EQ(inst->num_src_operands(), 4);

  const Operand *vdst = inst->dst_operand(0);
  ASSERT_NE(vdst, nullptr);
  EXPECT_FALSE(vdst->is_fieldless());
  EXPECT_EQ(vdst->name(), "v[32:35]");

  const Operand *vaddr = inst->src_operand(0);
  ASSERT_NE(vaddr, nullptr);
  EXPECT_FALSE(vaddr->is_fieldless());
  EXPECT_EQ(vaddr->size_bits(), 32);
  ASSERT_TRUE(vaddr->to_register_ref().has_value());
  EXPECT_EQ(*vaddr->to_register_ref(), (RegisterRef{RegClass::VGPR, 7, 1}));

  const Operand *gpumem = inst->src_operand(3);
  ASSERT_NE(gpumem, nullptr);
  EXPECT_TRUE(gpumem->is_fieldless());
  EXPECT_EQ(gpumem->size_bits(), 128);
  EXPECT_FALSE(gpumem->to_register_ref().has_value());
  // End-to-end: the decoded memory pseudo-operand is inert through the normal
  // accessors, driven by the capability flags the generated ctor applies.
  EXPECT_FALSE(gpumem->reads_value());
  EXPECT_FALSE(gpumem->is_writable());
  EXPECT_FALSE(gpumem->is_vgpr());

  EXPECT_EQ(inst->disassemble(), "buffer_load_b128 v[32:35], v7, s[4:7], NULL offen");
}

TEST(Gfx1250DecodeTest, BufferWithoutIdxenOffenDoesNotExposeVaddrRegister) {
  const uint32_t words[] = {
      0xC405C07Cu, // buffer_load_b128 v[32:35], s[4:7], s0
      0x00800820u,
      0x00000007u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "buffer_load_b128");
  ASSERT_EQ(inst->num_dst_operands(), 1);
  ASSERT_EQ(inst->num_src_operands(), 4);

  const Operand *vaddr = inst->src_operand(0);
  ASSERT_NE(vaddr, nullptr);
  EXPECT_FALSE(vaddr->is_fieldless());
  EXPECT_EQ(vaddr->size_bits(), 0);
  EXPECT_FALSE(vaddr->to_register_ref().has_value());

  const Operand *gpumem = inst->src_operand(3);
  ASSERT_NE(gpumem, nullptr);
  EXPECT_TRUE(gpumem->is_fieldless());
  EXPECT_EQ(gpumem->size_bits(), 128);
  EXPECT_FALSE(gpumem->to_register_ref().has_value());

  EXPECT_EQ(inst->disassemble(), "buffer_load_b128 v[32:35], s[4:7], NULL");
}

TEST(Gfx1250DecodeTest, WmmaF8f6f4UsesMatrixFormatOperandWidths) {
  const uint32_t words[] = {
      0xCC336010u,
      0x04421100u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(), "v_wmma_f32_16x16x128_f8f6f4 v[16:23], v[0:7], v[8:15], v[16:23] "
                                 "matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4");
}

TEST(Gfx1250DecodeTest, WmmaScaleF8f6f4ConsumesVop3px2Pair) {
  const uint32_t words[] = {
      0xCC350000u,
      0x02020900u,
      0xCC330006u,
      0x02026912u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_wmma_scale_f32_16x16x128_f8f6f4");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_EQ(inst->disassemble(),
            "v_wmma_scale_f32_16x16x128_f8f6f4 v[6:13], v[18:33], v[52:67], 0, v0, v4");
}

TEST(Gfx1250DecodeTest, WmmaScalePairRejectsInvalidEmbeddedSourceSelectors) {
  constexpr uint32_t invalid_src0_selectors[] = {255u, 250u, 233u, 234u};
  for (const uint32_t embedded_src0 : invalid_src0_selectors) {
    SCOPED_TRACE(embedded_src0);
    const uint32_t words[] = {
        0xCC350000u,
        0x20020700u,
        0xCC330000u,
        0xD600D400u | embedded_src0,
    };

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(words)), util::InvalidInst);
  }
}

TEST(Gfx1250DecodeTest, WmmaScale16F8f6f4ConsumesVop3px2Pair) {
  const uint32_t words[] = {
      0xCC3A0000u,
      0x0202391Au,
      0xCC336012u,
      0x02021502u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_wmma_scale16_f32_16x16x128_f8f6f4");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_EQ(inst->disassemble(),
            "v_wmma_scale16_f32_16x16x128_f8f6f4 v[18:25], v[2:9], v[10:17], 0, "
            "v[26:27], v[28:29] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4");
}

TEST(Gfx1250DecodeTest, WmmaScaleF4_32x16x128ConsumesVop3px2Pair) {
  const uint32_t words[] = {
      0xCC350000u,
      0x02025328u,
      0xCC884000u,
      0x1A024110u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_wmma_scale_f32_32x16x128_f4");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_EQ(inst->disassemble(),
            "v_wmma_scale_f32_32x16x128_f4 v[0:15], v[16:31], v[32:39], 0, v40, v41");
}

TEST(Gfx1250DecodeTest, WmmaScalePrefixRejectsNonWmmaSuffix) {
  const uint32_t words[] = {
      0xCC350000u,
      0x02020900u,
      0xCC340006u,
      0x02026912u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_THROW(static_cast<void>(decoder->decode(words)), util::InvalidInst);
}

TEST(Gfx1250ExecutionTest, WmmaRegularScaleInlineZeroMatchesNeutralScalarSources) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = sim.dispatch_scratch_wf();
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xffffffffu);

  constexpr uint16_t kVgprEncoding = 256;
  constexpr auto scalar_prefix =
      cdna5::build_vop3p(0x35, {.src0 = 0, .src1 = 1, .src2 = kVgprEncoding});
  constexpr auto inline_prefix =
      cdna5::build_vop3p(0x35, {.src0 = 128, .src1 = 128, .src2 = kVgprEncoding});
  constexpr auto scalar_matrix = cdna5::build_vop3p(
      cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
      {.vdst = 32, .src0 = kVgprEncoding, .src1 = kVgprEncoding + 16, .src2 = kVgprEncoding + 32});
  constexpr auto inline_matrix = cdna5::build_vop3p(
      cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
      {.vdst = 40, .src0 = kVgprEncoding, .src1 = kVgprEncoding + 16, .src2 = kVgprEncoding + 40});
  const std::array<uint32_t, 4> scalar_words = {scalar_prefix[0], scalar_prefix[1],
                                                scalar_matrix[0], scalar_matrix[1]};
  const std::array<uint32_t, 4> inline_words = {inline_prefix[0], inline_prefix[1],
                                                inline_matrix[0], inline_matrix[1]};

  write_wave_sgpr(*cu, *wf, 0, 0x7f7f7f7fu);
  write_wave_sgpr(*cu, *wf, 1, 0x7f7f7f7fu);
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  for (uint32_t reg = 0; reg < 32; ++reg)
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      cu->write_vgpr(vgpr_base + reg, lane, 0x38383838u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> scalar_inst(decoder->decode(scalar_words.data()));
  std::unique_ptr<Instruction> inline_inst(decoder->decode(inline_words.data()));
  ASSERT_NE(scalar_inst, nullptr);
  ASSERT_NE(inline_inst, nullptr);
  cu->execute_instruction(scalar_inst.get(), *wf);
  cu->execute_instruction(inline_inst.get(), *wf);

  for (uint32_t reg = 0; reg < 8; ++reg)
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      const uint32_t scalar_result = cu->read_vgpr(vgpr_base + 32 + reg, lane);
      EXPECT_EQ(scalar_result, std::bit_cast<uint32_t>(128.0f))
          << "reg " << reg << ", lane " << lane;
      EXPECT_EQ(cu->read_vgpr(vgpr_base + 40 + reg, lane), scalar_result)
          << "reg " << reg << ", lane " << lane;
    }
}

TEST(Gfx1250DecodeTest, SwmmacPrintsIndexKeyAndReuseModifiers) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  const uint32_t index_key_words[] = {
      0xCC65081Au,
      0x1C8E0112u,
  };
  std::unique_ptr<Instruction> index_key_inst(decoder->decode(index_key_words));
  ASSERT_NE(index_key_inst, nullptr);
  EXPECT_EQ(index_key_inst->disassemble(),
            "v_swmmac_f32_16x16x64_f16 v[26:33], v[18:25], v[0:15], v35 index_key:1");

  const uint32_t matrix_a_reuse_words[] = {
      0xCC65201Au,
      0x1C8E0112u,
  };
  std::unique_ptr<Instruction> matrix_a_reuse_inst(decoder->decode(matrix_a_reuse_words));
  ASSERT_NE(matrix_a_reuse_inst, nullptr);
  EXPECT_EQ(matrix_a_reuse_inst->disassemble(),
            "v_swmmac_f32_16x16x64_f16 v[26:33], v[18:25], v[0:15], v35 matrix_a_reuse");

  const uint32_t matrix_b_reuse_words[] = {
      0xCC65401Au,
      0x1C8E0112u,
  };
  std::unique_ptr<Instruction> matrix_b_reuse_inst(decoder->decode(matrix_b_reuse_words));
  ASSERT_NE(matrix_b_reuse_inst, nullptr);
  EXPECT_EQ(matrix_b_reuse_inst->disassemble(),
            "v_swmmac_f32_16x16x64_f16 v[26:33], v[18:25], v[0:15], v35 matrix_b_reuse");
}

TEST(Gfx1250DecodeTest, VopdXyConsumesTwoDwords) {
  const uint32_t words[] = {
      0xCA500501u, // v_dual_cndmask_b32 v2, v1, v2 :: v_dual_mov_b32 v1, 0
      0x02000080u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_dual_cndmask_b32 :: v_dual_mov_b32");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_NE(inst->disassemble().find("v_dual_cndmask_b32"), std::string::npos);
  EXPECT_NE(inst->disassemble().find("v_dual_mov_b32"), std::string::npos);
}

TEST(Gfx1250DecodeTest, Vopd3ConsumesThreeDwords) {
  const uint32_t words[] = {
      0xCF455083u, // v_dual_lshlrev_b32 v1, 3, v0 :: v_dual_lshrrev_b32 v10, 6, v0
      0x00000086u,
      0x0A000001u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_dual_lshlrev_b32 :: v_dual_lshrrev_b32");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_NE(inst->disassemble().find("v_dual_lshlrev_b32"), std::string::npos);
  EXPECT_NE(inst->disassemble().find("v_dual_lshrrev_b32"), std::string::npos);
}

TEST(Gfx1250DecodeTest, Vopd3RejectsSrcX0Literal32Selector) {
  const uint32_t words[] = {
      0xCF4550FFu,
      0x00000086u,
      0x0A000001u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(words)), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, Vopd3RejectsSrcY0Literal32Selector) {
  const uint32_t words[] = {
      0xCF455083u,
      0x000000FFu,
      0x0A000001u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_THROW(std::unique_ptr<Instruction>(decoder->decode(words)), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, VopdLiteralConsumesThreeDwords) {
  const uint32_t words[] = {
      0xC8D006FFu, // v_dual_mul_f32 v4, 0x4f7ffffe, v3 :: v_dual_mov_b32 v3, 0
      0x04020080u,
      0x4F7FFFFEu,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_dual_mul_f32 :: v_dual_mov_b32");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_NE(inst->disassemble().find("0x4f7ffffe"), std::string::npos);
}

TEST(Gfx1250DecodeTest, VopdSourceOperandsFollowPrintedSlots) {
  const uint32_t words[] = {
      0xCF448082u, // v_dual_lshlrev_b32 v17, 2, v9 :: v_dual_mov_b32 v9, s11
      0x0009000Bu,
      0x09000011u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_dual_lshlrev_b32 :: v_dual_mov_b32");
  EXPECT_EQ(inst->num_src_operands(), 3);
  ASSERT_NE(inst->src_operand(2), nullptr);
  EXPECT_EQ(inst->src_operand(2)->name(), "s11");
  ASSERT_TRUE(inst->src_operand(2)->to_register_ref().has_value());
  EXPECT_EQ(*inst->src_operand(2)->to_register_ref(), (RegisterRef{RegClass::SGPR, 11, 1}));
}

TEST(Gfx1250DecodeTest, VopdRejectsInvalidOpcodes) {
  const std::array<std::array<uint32_t, 3>, 2> words = {{
      // Opcode 18 is valid in the Y slot, but not the X slot.
      {0xCF000000u | (18u << 18) | (3u << 12), 0, 0},
      // Opcode 32 is valid in the X slot, but not the Y slot.
      {0xCF000000u | (3u << 18) | (32u << 12), 0, 0},
  }};

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  for (const auto &encoding : words)
    EXPECT_THROW(static_cast<void>(decoder->decode(encoding.data())), util::InvalidInst);
}

TEST(Gfx1250DecodeTest, PublicDecoderReportsInvalidVopdEncoding) {
  const uint32_t words[] = {
      (0x32u << 26) | (12u << 22) | (8u << 17), // Opcode 12 is not an X op.
      0,
  };

  rj_code_decoder_t *decoder = nullptr;
  ASSERT_EQ(rj_code_decoder_create(ROCJITSU_CODE_ARCH_CDNA5, &decoder), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(decoder, nullptr);

  auto *inst = reinterpret_cast<rj_code_inst_t *>(static_cast<uintptr_t>(1));
  EXPECT_EQ(rj_code_decoder_decode(decoder, words, &inst), ROCJITSU_STATUS_ERROR);
  EXPECT_EQ(inst, nullptr);
  rj_code_decoder_destroy(decoder);
}

TEST(Gfx1250DecodeTest, Vopd3RejectsOverlappingDestinations) {
  const std::array<std::array<uint32_t, 3>, 2> words = {{
      make_vopd3_pair({.op = VopdOp::CndmaskB32, .src0 = 0, .src1 = 1, .src2 = 2, .dst = 10},
                      {.op = VopdOp::MulF32, .src0 = 0, .src1 = 1, .src2 = 2, .dst = 10}),
      make_vopd3_pair({.op = VopdOp::FmaF64, .src0 = 0, .src1 = 1, .src2 = 2, .dst = 10},
                      {.op = VopdOp::MulF32, .src0 = 0, .src1 = 1, .src2 = 2, .dst = 11}),
  }};

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  for (const auto &encoding : words)
    EXPECT_THROW(static_cast<void>(decoder->decode(encoding.data())), util::InvalidInst);
}

} // namespace
