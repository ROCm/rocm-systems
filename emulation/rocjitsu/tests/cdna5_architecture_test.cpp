// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "util/data_types.h"

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

std::array<uint32_t, 4> build_scaled_wmma_words(uint8_t prefix_op, uint32_t matrix_a_fmt,
                                                uint32_t matrix_b_fmt, uint32_t scale_a_fmt,
                                                uint32_t scale_b_fmt, uint16_t scale_src0,
                                                uint16_t scale_src1, uint8_t vdst = 64) {
  constexpr uint16_t kVgprEncoding = 256;
  auto prefix = cdna5::build_vop3p(prefix_op, {.neg_hi = static_cast<uint8_t>(scale_b_fmt),
                                               .src0 = scale_src0,
                                               .src1 = scale_src1,
                                               .src2 = 256,
                                               .neg = static_cast<uint8_t>(scale_a_fmt)});
  auto matrix = cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
                                   {.vdst = vdst,
                                    .opsel = static_cast<uint8_t>(matrix_a_fmt),
                                    .src0 = kVgprEncoding,
                                    .src1 = kVgprEncoding + 32,
                                    .src2 = static_cast<uint16_t>(kVgprEncoding + vdst),
                                    .opsel_hi = static_cast<uint8_t>(matrix_b_fmt & 0x3u)});
  matrix[0] |= (matrix_b_fmt >> 2u) << 14u;
  return {prefix[0], prefix[1], matrix[0], matrix[1]};
}

uint32_t wmma_format_bits(uint32_t fmt) {
  if (fmt == 4)
    return 4;
  if (fmt == 2 || fmt == 3)
    return 6;
  return 8;
}

uint8_t encode_wmma_one(uint32_t fmt) {
  switch (fmt) {
  case 0:
    return util::f32_to_fp8_e4m3_rne(1.0f);
  case 1:
    return util::f32_to_bf8_e5m2_rne(1.0f);
  case 2:
    return util::f32_to_fp6_e2m3_rne(1.0f);
  case 3:
    return util::f32_to_bf6_e3m2_rne(1.0f);
  case 4:
    return util::f32_to_fp4_e2m1_rne(1.0f);
  default:
    return 0;
  }
}

void write_wmma_packed(amdgpu::ComputeUnitCore &cu, uint32_t base, const amdgpu::InputLoc &loc,
                       uint8_t value) {
  const uint32_t mask = (1u << loc.data_bits) - 1u;
  const uint32_t bits = static_cast<uint32_t>(value) & mask;
  const uint32_t first_bits = std::min(loc.data_bits, 32u - loc.bit_offset);
  const uint32_t first_mask = ((1u << first_bits) - 1u) << loc.bit_offset;
  uint32_t word = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  word = (word & ~first_mask) | ((bits << loc.bit_offset) & first_mask);
  cu.write_vgpr(base + loc.vgpr_offset, loc.lane, word);
  if (first_bits != loc.data_bits) {
    const uint32_t remaining = loc.data_bits - first_bits;
    const uint32_t second_mask = (1u << remaining) - 1u;
    word = cu.read_vgpr(base + loc.vgpr_offset + 1, loc.lane);
    word = (word & ~second_mask) | ((bits >> first_bits) & second_mask);
    cu.write_vgpr(base + loc.vgpr_offset + 1, loc.lane, word);
  }
}

uint8_t encode_wmma_value(uint32_t fmt, float value) {
  switch (fmt) {
  case 0:
    return util::f32_to_fp8_e4m3_rne(value);
  case 1:
    return util::f32_to_bf8_e5m2_rne(value);
  case 2:
    return util::f32_to_fp6_e2m3_rne(value);
  case 3:
    return util::f32_to_bf6_e3m2_rne(value);
  case 4:
    return util::f32_to_fp4_e2m1_rne(value);
  default:
    return 0;
  }
}

float decode_wmma_value(uint32_t fmt, uint8_t value) {
  switch (fmt) {
  case 0:
    return util::fp8_e4m3_to_f32(value);
  case 1:
    return util::bf8_e5m2_to_f32(value);
  case 2:
    return util::fp6_e2m3_to_f32(value);
  case 3:
    return util::bf6_e3m2_to_f32(value);
  case 4:
    return util::fp4_e2m1_to_f32(value);
  default:
    return 0.0f;
  }
}

// Independent transcription of the CDNA5 block-scaled WMMA tables. Keeping the
// physical writer separate from the execution helper makes this an integration
// check rather than seeding data through the implementation under test.
amdgpu::InputLoc manual_block_scaled_wmma_loc(uint32_t dim, uint32_t index, uint32_t k,
                                              uint32_t data_bits) {
  const uint32_t slice = dim == 32 ? index / 16u : 0u;
  const uint32_t index_in_slice = index % 16u;
  const uint32_t block_elems = data_bits == 8 ? 16u : 32u;
  const uint32_t lane = index_in_slice + 16u * ((k / block_elems) & 1u);
  const uint32_t slot = (k / (2u * block_elems)) * block_elems + (k % block_elems);
  const uint32_t bit = slot * data_bits;
  const uint32_t bit_in_word = bit % 32u;
  const uint32_t sub_element = (32u % data_bits == 0) ? bit_in_word / data_bits : 0u;
  return {bit / 32u + 8u * slice, lane, sub_element, bit_in_word, data_bits};
}

std::array<uint32_t, 4> build_scaled_wmma_execution_words(
    uint32_t M, bool scale16, uint32_t matrix_a_fmt, uint32_t matrix_b_fmt, uint16_t scale_src0,
    uint16_t scale_src1, uint32_t scale_a_select, uint32_t scale_b_select, uint8_t vdst = 64) {
  constexpr uint16_t kVgprEncoding = 256;
  auto prefix =
      cdna5::build_vop3p(scale16 ? 0x3a : 0x35, {.opsel = static_cast<uint8_t>(scale_a_select),
                                                 .src0 = scale_src0,
                                                 .src1 = scale_src1,
                                                 .src2 = kVgprEncoding,
                                                 .opsel_hi = static_cast<uint8_t>(scale_b_select)});
  const uint16_t matrix_op =
      M == 32 ? cdna5::kVWmmaF3232x16x128F4Vop3p : cdna5::kVWmmaF3216x16x128F8f6f4Vop3p;
  auto matrix =
      cdna5::build_vop3p(matrix_op, {.vdst = vdst,
                                     .opsel = static_cast<uint8_t>(matrix_a_fmt),
                                     .src0 = kVgprEncoding,
                                     .src1 = kVgprEncoding + 32,
                                     .src2 = 128,
                                     .opsel_hi = static_cast<uint8_t>(matrix_b_fmt & 0x3u)});
  if (M == 32)
    matrix[0] |= 1u << 14u;
  else
    matrix[0] |= (matrix_b_fmt >> 2u) << 14u;
  return {prefix[0], prefix[1], matrix[0], matrix[1]};
}

struct ForceScalarGuard {
  bool old = util::force_scalar();
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(old); }
};

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
  EXPECT_EQ(loaded.device.device_id, 30145u);
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
  EXPECT_TRUE(cu->sram_ecc());
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
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
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
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
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
  EXPECT_TRUE(decode_fails(*decoder, words));
}

TEST(Gfx1250DecodeTest, Vop3LiteralConsumesThreeDwords) {
  const uint32_t words[] = {
      0xD6570001u, // v_and_or_b32 v1, 0xf8, v1, v2
      0x040A02FFu,
      0x000000F8u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
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
  EXPECT_TRUE(decode_fails(*decoder, words));
}

TEST(Gfx1250DecodeTest, Vop1RejectsUnsupportedLiteral32WithoutExtensionWord) {
  const auto words = cdna5::build_vop1(cdna5::kVReadfirstlaneB32Vop1, {.src0 = 255});

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decode_fails(*decoder, words.data()));
}

TEST(Gfx1250DecodeTest, Vop2RejectsUnsupportedLiteral32WithoutExtensionWord) {
  const auto words = cdna5::build_vop2(cdna5::kVFmamkF64Vop2, {.src0 = 255});

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decode_fails(*decoder, words.data()));
}

TEST(Gfx1250DecodeTest, Vop2RejectsUnsupportedLiteral64WithoutExtensionWords) {
  const auto words = cdna5::build_vop2(cdna5::kVFmamkF32Vop2, {.src0 = 254});

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decode_fails(*decoder, words.data()));
}

TEST(Gfx1250DecodeTest, SaluRejectsMixedLiteralWidths) {
  const uint32_t words[] = {
      0xBF5DFFFEu, // s_cmp_neq_f16 literal64, literal32
      0x00000000u,
      0x00000000u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decode_fails(*decoder, words));
}

TEST(Gfx1250DecodeTest, SendmsgRtnSelectorsAreNotLiterals) {
  const uint32_t words[] = {
      0xBE804CFFu, // s_sendmsg_rtn_b32 s0, 255
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
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
  EXPECT_TRUE(decode_fails(*decoder, words));
}

TEST(Gfx1250DecodeTest, Vop3RejectsDppWithLiteral) {
  const uint32_t words[] = {
      0xD6290B00u, // v_min3_num_f32 with src0:DPP and src2:literal
      0x83FF00FAu,
      0x00001500u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decode_fails(*decoder, words));
}

TEST(Gfx1250DecodeTest, Vop3RejectsInvalidScalarDestination) {
  const uint32_t words[] = {
      0xD41B10FFu, // v_cmp_ngt_f32 with reserved scalar destination 255
      0x000000FAu,
      0x00000000u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decode_fails(*decoder, words));
}

TEST(Gfx1250DecodeTest, Vop3RejectsInvalidVgprSource) {
  const uint32_t words[] = {
      0xD7600000u, // v_readlane_b32 s0, invalid, null
      0x0000F8D7u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decode_fails(*decoder, words));
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
  std::unique_ptr<Instruction> valid(decode_valid(*decoder, valid_words));
  ASSERT_NE(valid, nullptr);
  EXPECT_EQ(valid->disassemble(), "v_readlane_b32 s0, v215, 1");
  EXPECT_TRUE(decode_fails(*decoder, invalid_words));
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
  std::unique_ptr<Instruction> valid(decode_valid(*decoder, valid_words));
  ASSERT_NE(valid, nullptr);
  EXPECT_EQ(valid->disassemble(), "v_cmpx_ne_u32 exec, v0, v1");
  EXPECT_TRUE(decode_fails(*decoder, invalid_words));
}

TEST(Gfx1250DecodeTest, Vop3SdstLiteralConsumesThreeDwords) {
  const uint32_t words[] = {
      0xD7020001u, // v_subrev_co_u32 v1, s0, 0x60, s12
      0x020018FFu,
      0x00000060u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
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
  std::unique_ptr<Instruction> fmamk(decode_valid(*decoder, words));
  ASSERT_NE(fmamk, nullptr);
  EXPECT_EQ(fmamk->mnemonic(), "v_fmamk_f64_e32");
  EXPECT_EQ(fmamk->size(), 3 * sizeof(uint32_t));

  std::unique_ptr<Instruction> next(decode_valid(*decoder, words + 3));
  ASSERT_NE(next, nullptr);
  EXPECT_EQ(next->mnemonic(), "v_cvt_u32_f64_e32");
}

TEST(Gfx1250DecodeTest, SWaitXcntHasWaitcntMetadata) {
  const uint32_t words[] = {
      0xBFC50000u, // s_wait_xcnt 0
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
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
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
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
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
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
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(), "v_wmma_f32_16x16x128_f8f6f4 v[16:23], v[0:7], v[8:15], v[16:23] "
                                 "matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4");
}

TEST(Gfx1250DecodeTest, WmmaScaleF8f6f4ConsumesVop3px2Pair) {
  const uint32_t words[] = {
      0xCC350000u,
      0x04020900u,
      0xCC330006u,
      0x02026912u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_wmma_scale_f32_16x16x128_f8f6f4");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_EQ(inst->disassemble(),
            "v_wmma_scale_f32_16x16x128_f8f6f4 v[6:13], v[18:33], v[52:67], 0, v0, v4");
}

TEST(Gfx1250DecodeTest, WmmaScaleAcceptsLlvmAndIsaFixedSrc2Encodings) {
  auto isa_words = build_scaled_wmma_words(0x35, 0, 0, 0, 0, 128, 128);
  auto llvm_words = isa_words;
  constexpr uint32_t kSrc2Mask = 0x1ffu << 18;
  llvm_words[1] = (llvm_words[1] & ~kSrc2Mask) | (0x080u << 18);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> isa_inst(decode_valid(*decoder, isa_words.data()));
  std::unique_ptr<Instruction> llvm_inst(decode_valid(*decoder, llvm_words.data()));
  ASSERT_NE(isa_inst, nullptr);
  ASSERT_NE(llvm_inst, nullptr);
  EXPECT_EQ(llvm_inst->mnemonic(), isa_inst->mnemonic());
  EXPECT_EQ(llvm_inst->size(), isa_inst->size());
  EXPECT_EQ(llvm_inst->num_src_operands(), isa_inst->num_src_operands());
  EXPECT_EQ(llvm_inst->num_dst_operands(), isa_inst->num_dst_operands());
  EXPECT_EQ(llvm_inst->disassemble(), isa_inst->disassemble());
}

TEST(Gfx1250DecodeTest, WmmaScalePairRejectsInvalidEmbeddedSourceSelectors) {
  constexpr uint32_t invalid_src0_selectors[] = {255u, 250u, 233u, 234u};
  for (const uint32_t embedded_src0 : invalid_src0_selectors) {
    SCOPED_TRACE(embedded_src0);
    const uint32_t words[] = {
        0xCC350000u,
        0x24020700u,
        0xCC330000u,
        0xD600D400u | embedded_src0,
    };

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    EXPECT_TRUE(decode_fails(*decoder, words));
  }
}

TEST(Gfx1250DecodeTest, WmmaScalePairRejectsInvalidFixedAndReservedFields) {
  struct InvalidField {
    const char *name;
    size_t word;
    uint32_t bits;
  };
  constexpr std::array invalid_fields = {
      InvalidField{"prefix_vdst", 0, 1u << 0},
      InvalidField{"prefix_neg_hi_reserved", 0, 1u << 10},
      InvalidField{"prefix_opsel_reserved", 0, 1u << 12},
      InvalidField{"prefix_clamp", 0, 1u << 15},
      InvalidField{"prefix_opsel_hi_reserved", 1, 1u << 28},
      InvalidField{"prefix_neg_reserved", 1, 1u << 31},
      InvalidField{"matrix_neg_hi_reserved_0", 2, 1u << 8},
      InvalidField{"matrix_neg_hi_reserved_1", 2, 1u << 9},
      InvalidField{"matrix_clamp", 2, 1u << 15},
      InvalidField{"matrix_neg_reserved_0", 3, 1u << 29},
      InvalidField{"matrix_neg_reserved_1", 3, 1u << 30},
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  for (const auto &field : invalid_fields) {
    SCOPED_TRACE(field.name);
    auto words = build_scaled_wmma_words(0x35, 0, 0, 0, 0, 128, 128);
    words[field.word] |= field.bits;
    EXPECT_TRUE(decode_fails(*decoder, words.data()));
  }

  for (uint32_t fixed_src2 : {0u, 255u, 511u}) {
    SCOPED_TRACE(::testing::Message() << "prefix_src2=" << fixed_src2);
    auto words = build_scaled_wmma_words(0x35, 0, 0, 0, 0, 128, 128);
    constexpr uint32_t kSrc2Mask = 0x1ffu << 18;
    words[1] = (words[1] & ~kSrc2Mask) | (fixed_src2 << 18);
    EXPECT_TRUE(decode_fails(*decoder, words.data()));
  }
}

TEST(Gfx1250DecodeTest, WmmaScale16F8f6f4ConsumesVop3px2Pair) {
  const uint32_t words[] = {
      0xCC3A0000u,
      0x0402391Au,
      0xCC336012u,
      0x02021502u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_wmma_scale16_f32_16x16x128_f8f6f4");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_EQ(inst->disassemble(),
            "v_wmma_scale16_f32_16x16x128_f8f6f4 v[18:25], v[2:9], v[10:17], 0, "
            "v[26:27], v[28:29] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4");
}

TEST(Gfx1250DecodeTest, WmmaScale16ScalarSourcesUseSingleSgprs) {
  const auto words = build_scaled_wmma_words(0x3a, 0, 0, 0, 0, 0, 2);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(),
            "v_wmma_scale16_f32_16x16x128_f8f6f4 v[64:71], v[0:15], v[32:47], v[64:71], "
            "s0, s2");
}

TEST(Gfx1250DecodeTest, WmmaScale16InlineScaleSourcesRemainInline) {
  const auto words = build_scaled_wmma_words(0x3a, 0, 0, 0, 0, 128, 128);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(),
            "v_wmma_scale16_f32_16x16x128_f8f6f4 v[64:71], v[0:15], v[32:47], v[64:71], "
            "0, 0");
}

TEST(Gfx1250DecodeTest, WmmaScaleF4_32x16x128ConsumesVop3px2Pair) {
  const uint32_t words[] = {
      0xCC350000u,
      0x04025328u,
      0xCC884000u,
      0x1A024110u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_wmma_scale_f32_32x16x128_f4");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_EQ(inst->disassemble(),
            "v_wmma_scale_f32_32x16x128_f4 v[0:15], v[16:31], v[32:39], 0, v40, v41");
}

TEST(Gfx1250DecodeTest, WmmaScalePrefixRejectsNonWmmaSuffix) {
  const uint32_t words[] = {
      0xCC350000u,
      0x04020900u,
      0xCC340006u,
      0x02026912u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decode_fails(*decoder, words));
}

TEST(Gfx1250DecodeTest, WmmaScaleEnforcesMatrixAndScaleFormatLegality) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  for (uint32_t matrix_a_fmt = 0; matrix_a_fmt <= 4; ++matrix_a_fmt) {
    for (uint32_t matrix_b_fmt = 0; matrix_b_fmt <= 4; ++matrix_b_fmt) {
      for (uint32_t scale_a_fmt = 0; scale_a_fmt <= 2; ++scale_a_fmt) {
        for (uint32_t scale_b_fmt = 0; scale_b_fmt <= 2; ++scale_b_fmt) {
          SCOPED_TRACE(::testing::Message()
                       << "matrix_a_fmt=" << matrix_a_fmt << " matrix_b_fmt=" << matrix_b_fmt
                       << " scale_a_fmt=" << scale_a_fmt << " scale_b_fmt=" << scale_b_fmt);
          const bool both_e8 = scale_a_fmt == 0 && scale_b_fmt == 0;
          const bool non_e8_only_on_f4 =
              (scale_a_fmt == 0 || matrix_a_fmt == 4) && (scale_b_fmt == 0 || matrix_b_fmt == 4);
          const bool matching_f4_scales =
              matrix_a_fmt != 4 || matrix_b_fmt != 4 || scale_a_fmt == scale_b_fmt;
          const bool legal = both_e8 || (non_e8_only_on_f4 && matching_f4_scales);
          const auto words = build_scaled_wmma_words(0x35, matrix_a_fmt, matrix_b_fmt, scale_a_fmt,
                                                     scale_b_fmt, 128, 128);
          if (legal)
            EXPECT_FALSE(decode_fails(*decoder, words.data()));
          else
            EXPECT_TRUE(decode_fails(*decoder, words.data()));
        }
      }
    }
  }
}

TEST(Gfx1250DecodeTest, WmmaScaleRejectsIllegalScaleSources) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  for (uint16_t selector : {106u, 127u, 129u, 255u}) {
    SCOPED_TRACE(selector);
    const auto invalid = build_scaled_wmma_words(0x35, 0, 0, 0, 0, selector, 128);
    EXPECT_TRUE(decode_fails(*decoder, invalid.data()));
  }

  for (uint16_t selector : {1u, 128u, 256u, 510u}) {
    SCOPED_TRACE(::testing::Message() << "legal Scale16 selector=" << selector);
    const auto valid = build_scaled_wmma_words(0x3a, 0, 0, 0, 0, selector, 128);
    EXPECT_FALSE(decode_fails(*decoder, valid.data()));
  }
  for (uint16_t selector : {257u, 509u, 511u}) {
    SCOPED_TRACE(::testing::Message() << "illegal Scale16 selector=" << selector);
    const auto invalid = build_scaled_wmma_words(0x3a, 0, 0, 0, 0, selector, 128);
    EXPECT_TRUE(decode_fails(*decoder, invalid.data()));
  }
}

TEST(Gfx1250ExecutionTest, WmmaScaleExecutesAllMatrixFormatPairs) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = sim.dispatch_scratch_wf();
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xffffffffu);
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  ForceScalarGuard scalar_guard;

  for (bool force_scalar : {true, false}) {
    util::set_force_scalar_for_testing(force_scalar);
    for (uint32_t matrix_a_fmt = 0; matrix_a_fmt <= 4; ++matrix_a_fmt) {
      for (uint32_t matrix_b_fmt = 0; matrix_b_fmt <= 4; ++matrix_b_fmt) {
        SCOPED_TRACE(::testing::Message() << "force_scalar=" << force_scalar << " matrix_a_fmt="
                                          << matrix_a_fmt << " matrix_b_fmt=" << matrix_b_fmt);
        for (uint32_t reg = 0; reg < 72; ++reg)
          for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
            cu->write_vgpr(vgpr_base + reg, lane, 0);

        const uint32_t a_bits = wmma_format_bits(matrix_a_fmt);
        const uint32_t b_bits = wmma_format_bits(matrix_b_fmt);
        const uint8_t a_one = encode_wmma_one(matrix_a_fmt);
        const uint8_t b_one = encode_wmma_one(matrix_b_fmt);
        for (uint32_t row = 0; row < 16; ++row)
          for (uint32_t k = 0; k < 128; ++k)
            write_wmma_packed(*cu, vgpr_base,
                              amdgpu::wmma_a_input_loc(16, 128, row, k, a_bits, b_bits), a_one);
        for (uint32_t col = 0; col < 16; ++col)
          for (uint32_t k = 0; k < 128; ++k)
            write_wmma_packed(*cu, vgpr_base + 32,
                              amdgpu::wmma_b_input_loc(16, 128, col, k, a_bits, b_bits), b_one);

        const auto words =
            build_scaled_wmma_words(0x35, matrix_a_fmt, matrix_b_fmt, 0, 0, 128, 128);
        std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
        ASSERT_NE(inst, nullptr);
        cu->execute_instruction(inst.get(), *wf);
        for (uint32_t reg = 0; reg < 8; ++reg)
          for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
            EXPECT_EQ(cu->read_vgpr(vgpr_base + 64 + reg, lane), std::bit_cast<uint32_t>(128.0f));
      }
    }
  }
}

TEST(Gfx1250ExecutionTest, WmmaRegularScaleInlineZeroMatchesNeutralScalarSources) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = sim.dispatch_scratch_wf();
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xffffffffu);

  constexpr uint16_t kVgprEncoding = 256;
  constexpr auto scalar_prefix = cdna5::build_vop3p(0x35, {.src0 = 0, .src1 = 1, .src2 = 256});
  constexpr auto inline_prefix = cdna5::build_vop3p(0x35, {.src0 = 128, .src1 = 128, .src2 = 256});
  constexpr auto scalar_matrix = cdna5::build_vop3p(
      cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
      {.vdst = 32, .src0 = kVgprEncoding, .src1 = kVgprEncoding + 16, .src2 = kVgprEncoding + 32});
  constexpr auto inline_matrix = cdna5::build_vop3p(
      cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
      {.vdst = 40, .src0 = kVgprEncoding, .src1 = kVgprEncoding + 16, .src2 = kVgprEncoding + 40});
  constexpr auto llvm_matrix = cdna5::build_vop3p(
      cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
      {.vdst = 48, .src0 = kVgprEncoding, .src1 = kVgprEncoding + 16, .src2 = kVgprEncoding + 48});
  const std::array<uint32_t, 4> scalar_words = {scalar_prefix[0], scalar_prefix[1],
                                                scalar_matrix[0], scalar_matrix[1]};
  const std::array<uint32_t, 4> inline_words = {inline_prefix[0], inline_prefix[1],
                                                inline_matrix[0], inline_matrix[1]};
  std::array<uint32_t, 4> llvm_words = {inline_prefix[0], inline_prefix[1], llvm_matrix[0],
                                        llvm_matrix[1]};
  constexpr uint32_t kSrc2Mask = 0x1ffu << 18;
  llvm_words[1] = (llvm_words[1] & ~kSrc2Mask) | (0x080u << 18);

  write_wave_sgpr(*cu, *wf, 0, 0x807e007fu);
  write_wave_sgpr(*cu, *wf, 1, 0x0102037fu);
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  for (uint32_t reg = 0; reg < 32; ++reg)
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      cu->write_vgpr(vgpr_base + reg, lane, 0x38383838u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> scalar_inst(decode_valid(*decoder, scalar_words.data()));
  std::unique_ptr<Instruction> inline_inst(decode_valid(*decoder, inline_words.data()));
  std::unique_ptr<Instruction> llvm_inst(decode_valid(*decoder, llvm_words.data()));
  ASSERT_NE(scalar_inst, nullptr);
  ASSERT_NE(inline_inst, nullptr);
  ASSERT_NE(llvm_inst, nullptr);
  cu->execute_instruction(scalar_inst.get(), *wf);
  cu->execute_instruction(inline_inst.get(), *wf);
  cu->execute_instruction(llvm_inst.get(), *wf);

  for (uint32_t reg = 0; reg < 8; ++reg)
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      const uint32_t scalar_result = cu->read_vgpr(vgpr_base + 32 + reg, lane);
      EXPECT_EQ(scalar_result, std::bit_cast<uint32_t>(128.0f))
          << "reg " << reg << ", lane " << lane;
      EXPECT_EQ(cu->read_vgpr(vgpr_base + 40 + reg, lane), scalar_result)
          << "reg " << reg << ", lane " << lane;
      EXPECT_EQ(cu->read_vgpr(vgpr_base + 48 + reg, lane), scalar_result)
          << "reg " << reg << ", lane " << lane;
    }
}

TEST(Gfx1250ExecutionTest, WmmaScale16InlineZeroAndSgprUseNeutralScaleForEveryBlock) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = sim.dispatch_scratch_wf();
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xffffffffu);
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  for (uint32_t reg = 0; reg < 48; ++reg)
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      cu->write_vgpr(vgpr_base + reg, lane, 0x38383838u);
  write_wave_sgpr(*cu, *wf, 0, 0x0102037fu);
  write_wave_sgpr(*cu, *wf, 1, 0x11223344u);
  write_wave_sgpr(*cu, *wf, 2, 0xa0b0c07fu);
  write_wave_sgpr(*cu, *wf, 3, 0x55667788u);

  const auto scalar_words = build_scaled_wmma_words(0x3a, 0, 0, 0, 0, 0, 2, 64);
  const auto inline_words = build_scaled_wmma_words(0x3a, 0, 0, 0, 0, 128, 128, 72);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> scalar_inst(decode_valid(*decoder, scalar_words.data()));
  std::unique_ptr<Instruction> inline_inst(decode_valid(*decoder, inline_words.data()));
  ASSERT_NE(scalar_inst, nullptr);
  ASSERT_NE(inline_inst, nullptr);
  cu->execute_instruction(scalar_inst.get(), *wf);
  cu->execute_instruction(inline_inst.get(), *wf);

  for (uint32_t reg = 0; reg < 8; ++reg)
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      EXPECT_EQ(cu->read_vgpr(vgpr_base + 64 + reg, lane), std::bit_cast<uint32_t>(128.0f));
      EXPECT_EQ(cu->read_vgpr(vgpr_base + 72 + reg, lane), std::bit_cast<uint32_t>(128.0f));
    }
}

TEST(Gfx1250ExecutionTest, WmmaScaleDecodesE5m3ForFp4Operand) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = sim.dispatch_scratch_wf();
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xffffffffu);
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  for (uint32_t row = 0; row < 16; ++row)
    for (uint32_t k = 0; k < 128; ++k)
      write_wmma_packed(*cu, vgpr_base, amdgpu::wmma_a_input_loc(16, 128, row, k, 4, 8),
                        util::f32_to_fp4_e2m1_rne(1.0f));
  for (uint32_t col = 0; col < 16; ++col)
    for (uint32_t k = 0; k < 128; ++k)
      write_wmma_packed(*cu, vgpr_base + 32, amdgpu::wmma_b_input_loc(16, 128, col, k, 4, 8),
                        util::f32_to_fp8_e4m3_rne(1.0f));
  write_wave_sgpr(*cu, *wf, 0, 0x78u);
  write_wave_sgpr(*cu, *wf, 1, 0x7fu);

  const auto words = build_scaled_wmma_words(0x35, 4, 0, 1, 0, 0, 1);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  cu->execute_instruction(inst.get(), *wf);
  for (uint32_t reg = 0; reg < 8; ++reg)
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      EXPECT_EQ(cu->read_vgpr(vgpr_base + 64 + reg, lane), std::bit_cast<uint32_t>(128.0f));
}

TEST(Gfx1250ExecutionTest, WmmaNonE8ScalesApplyAfterEachBlockDot) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = sim.dispatch_scratch_wf();
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xffffffffu);
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  ForceScalarGuard scalar_guard;

  struct ScaleCase {
    uint32_t format;
    uint8_t encoded;
    float value;
  };
  const std::array scale_cases = {
      ScaleCase{1, util::f32_to_fp8_e5m3_rne(1.5f), 1.5f},
      ScaleCase{2, util::f32_to_fp8_e4m3_rne(0.75f), 0.75f},
  };
  constexpr float kAccumulator = 16777216.0f;

  for (bool force_scalar : {true, false}) {
    util::set_force_scalar_for_testing(force_scalar);
    for (bool scale16 : {false, true}) {
      for (const auto &scale : scale_cases) {
        SCOPED_TRACE(::testing::Message() << "force_scalar=" << force_scalar << " scale16="
                                          << scale16 << " scale_format=" << scale.format);
        for (uint32_t reg = 0; reg < 80; ++reg)
          for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
            cu->write_vgpr(vgpr_base + reg, lane, 0);
        for (uint32_t row = 0; row < 16; ++row)
          for (uint32_t k = 0; k < 128; ++k)
            write_wmma_packed(*cu, vgpr_base, amdgpu::wmma_a_input_loc(16, 128, row, k, 4, 8),
                              util::f32_to_fp4_e2m1_rne(0.5f));
        for (uint32_t col = 0; col < 16; ++col)
          for (uint32_t k = 0; k < 128; ++k)
            write_wmma_packed(*cu, vgpr_base + 32, amdgpu::wmma_b_input_loc(16, 128, col, k, 4, 8),
                              util::f32_to_fp8_e4m3_rne(1.0f));
        for (uint32_t reg = 0; reg < 8; ++reg)
          for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
            cu->write_vgpr(vgpr_base + 64 + reg, lane, std::bit_cast<uint32_t>(kAccumulator));
        write_wave_sgpr(*cu, *wf, 0, scale.encoded);
        write_wave_sgpr(*cu, *wf, 1, 0x7fu);

        const auto words =
            build_scaled_wmma_words(scale16 ? 0x3a : 0x35, 4, 0, scale.format, 0, 0, 1);
        std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
        ASSERT_NE(inst, nullptr);
        cu->execute_instruction(inst.get(), *wf);

        float expected = kAccumulator;
        const uint32_t block_size = scale16 ? 16 : 32;
        for (uint32_t block = 0; block < 128 / block_size; ++block) {
          float block_dot = 0.0f;
          for (uint32_t k = 0; k < block_size; ++k)
            block_dot = std::fma(0.5f, 1.0f, block_dot);
          block_dot *= scale.value;
          expected += block_dot;
        }
        for (uint32_t reg = 0; reg < 8; ++reg)
          for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
            EXPECT_EQ(cu->read_vgpr(vgpr_base + 64 + reg, lane), std::bit_cast<uint32_t>(expected));
      }
    }
  }
}

TEST(Gfx1250ExecutionTest, WmmaScaledExecutionMatchesManualLayoutAndDistinctBlockScales) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = sim.dispatch_scratch_wf();
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xffffffffu);
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  ForceScalarGuard scalar_guard;

  constexpr uint32_t kMatrixA = 0;
  constexpr uint32_t kMatrixB = 32;
  constexpr uint32_t kOutput = 64;
  constexpr uint32_t kScaleA = 80;
  constexpr uint32_t kScaleB = 82;
  constexpr std::array<float, 4> matrix_values = {-1.0f, -0.5f, 0.5f, 1.5f};

  struct TestCase {
    uint32_t M;
    uint32_t matrix_a_fmt;
    uint32_t matrix_b_fmt;
    bool scale16;
    uint32_t scale_a_select;
    uint32_t scale_b_select;
  };
  constexpr TestCase cases[] = {
      {16, 0, 4, false, 1, 0}, // FP8 x FP4, 32 values per scale.
      {16, 4, 0, true, 0, 1},  // FP4 x FP8, 16 values per scale.
      {16, 2, 3, false, 1, 1}, // FP6 x BF6, including cross-VGPR packed values.
      {32, 4, 4, true, 0, 1},  // Two-slice FP4 A matrix and 64-bit scale words.
  };

  const auto matrix_a_raw = [&](const TestCase &tc, uint32_t row, uint32_t k) {
    const float value = matrix_values[(3u * row + 5u * k + k / 16u) % matrix_values.size()];
    return encode_wmma_value(tc.matrix_a_fmt, value);
  };
  const auto matrix_b_raw = [&](const TestCase &tc, uint32_t col, uint32_t k) {
    const float value = matrix_values[(5u * col + 3u * k + k / 32u + 1u) % matrix_values.size()];
    return encode_wmma_value(tc.matrix_b_fmt, value);
  };
  const auto scale_a_raw = [](uint32_t row, uint32_t block) {
    return static_cast<uint8_t>(0x7eu + ((row + 2u * block) % 3u));
  };
  const auto scale_b_raw = [](uint32_t col, uint32_t block) {
    return static_cast<uint8_t>(0x7eu + ((2u * col + block + 1u) % 3u));
  };

  for (const auto &tc : cases) {
    SCOPED_TRACE(::testing::Message() << "M=" << tc.M << " a_fmt=" << tc.matrix_a_fmt
                                      << " b_fmt=" << tc.matrix_b_fmt << " scale16=" << tc.scale16);
    const uint32_t N = 16;
    const uint32_t K = 128;
    const uint32_t a_bits = wmma_format_bits(tc.matrix_a_fmt);
    const uint32_t b_bits = wmma_format_bits(tc.matrix_b_fmt);
    const uint32_t scale_blocks = tc.scale16 ? 8u : 4u;

    auto seed_inputs_and_scales = [&]() {
      for (uint32_t reg = 0; reg < 84; ++reg)
        for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
          cu->write_vgpr(vgpr_base + reg, lane, 0);

      for (uint32_t row = 0; row < tc.M; ++row)
        for (uint32_t k = 0; k < K; ++k)
          write_wmma_packed(*cu, vgpr_base + kMatrixA,
                            manual_block_scaled_wmma_loc(tc.M, row, k, a_bits),
                            matrix_a_raw(tc, row, k));
      for (uint32_t col = 0; col < N; ++col)
        for (uint32_t k = 0; k < K; ++k)
          write_wmma_packed(*cu, vgpr_base + kMatrixB,
                            manual_block_scaled_wmma_loc(N, col, k, b_bits),
                            matrix_b_raw(tc, col, k));

      for (uint32_t row = 0; row < tc.M; ++row) {
        uint64_t word = 0;
        for (uint32_t block = 0; block < scale_blocks; ++block)
          word |= static_cast<uint64_t>(scale_a_raw(row, block)) << (8u * block);
        const uint32_t lane = tc.M == 32 ? row : row + 16u * tc.scale_a_select;
        cu->write_vgpr(vgpr_base + kScaleA, lane, static_cast<uint32_t>(word));
        if (tc.scale16)
          cu->write_vgpr(vgpr_base + kScaleA + 1, lane, static_cast<uint32_t>(word >> 32u));
      }
      for (uint32_t col = 0; col < N; ++col) {
        uint64_t word = 0;
        for (uint32_t block = 0; block < scale_blocks; ++block)
          word |= static_cast<uint64_t>(scale_b_raw(col, block)) << (8u * block);
        const uint32_t lane = col + 16u * tc.scale_b_select;
        cu->write_vgpr(vgpr_base + kScaleB, lane, static_cast<uint32_t>(word));
        if (tc.scale16)
          cu->write_vgpr(vgpr_base + kScaleB + 1, lane, static_cast<uint32_t>(word >> 32u));
      }
    };

    const auto words = build_scaled_wmma_execution_words(
        tc.M, tc.scale16, tc.matrix_a_fmt, tc.matrix_b_fmt, 256 + kScaleA, 256 + kScaleB,
        tc.scale_a_select, tc.scale_b_select, kOutput);
    auto run = [&](bool force_scalar) {
      util::set_force_scalar_for_testing(force_scalar);
      seed_inputs_and_scales();
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      EXPECT_NE(inst, nullptr);
      if (!inst)
        return std::vector<uint32_t>{};
      cu->execute_instruction(inst.get(), *wf);
      std::vector<uint32_t> output;
      output.reserve(tc.M * N);
      for (uint32_t row = 0; row < tc.M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          const auto loc = amdgpu::wmma_output_loc_32(tc.M, N, row, col);
          output.push_back(cu->read_vgpr(vgpr_base + kOutput + loc.reg, loc.lane));
        }
      return output;
    };

    const auto scalar = run(true);
    const auto simd = run(false);
    ASSERT_EQ(scalar.size(), tc.M * N);
    ASSERT_EQ(simd, scalar);

    const uint32_t block_size = tc.scale16 ? 16u : 32u;
    for (uint32_t row = 0; row < tc.M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        float expected = 0.0f;
        for (uint32_t block = 0; block < scale_blocks; ++block) {
          float block_sum = 0.0f;
          for (uint32_t k = block * block_size; k < (block + 1u) * block_size; ++k) {
            const float a = decode_wmma_value(tc.matrix_a_fmt, matrix_a_raw(tc, row, k));
            const float b = decode_wmma_value(tc.matrix_b_fmt, matrix_b_raw(tc, col, k));
            block_sum = std::fma(a, b, block_sum);
          }
          block_sum *= util::e8m0_to_f32(scale_a_raw(row, block));
          block_sum *= util::e8m0_to_f32(scale_b_raw(col, block));
          expected += block_sum;
        }
        EXPECT_EQ(scalar[row * N + col], std::bit_cast<uint32_t>(expected))
            << "row=" << row << " col=" << col;
      }
    }
  }
}

TEST(Gfx1250DecodeTest, SwmmacPrintsIndexKeyAndReuseModifiers) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  const uint32_t index_key_words[] = {
      0xCC65081Au,
      0x1C8E0112u,
  };
  std::unique_ptr<Instruction> index_key_inst(decode_valid(*decoder, index_key_words));
  ASSERT_NE(index_key_inst, nullptr);
  EXPECT_EQ(index_key_inst->disassemble(),
            "v_swmmac_f32_16x16x64_f16 v[26:33], v[18:25], v[0:15], v35 index_key:1");

  const uint32_t matrix_a_reuse_words[] = {
      0xCC65201Au,
      0x1C8E0112u,
  };
  std::unique_ptr<Instruction> matrix_a_reuse_inst(decode_valid(*decoder, matrix_a_reuse_words));
  ASSERT_NE(matrix_a_reuse_inst, nullptr);
  EXPECT_EQ(matrix_a_reuse_inst->disassemble(),
            "v_swmmac_f32_16x16x64_f16 v[26:33], v[18:25], v[0:15], v35 matrix_a_reuse");

  const uint32_t matrix_b_reuse_words[] = {
      0xCC65401Au,
      0x1C8E0112u,
  };
  std::unique_ptr<Instruction> matrix_b_reuse_inst(decode_valid(*decoder, matrix_b_reuse_words));
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
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
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
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
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
  EXPECT_TRUE(decode_fails(*decoder, words));
}

TEST(Gfx1250DecodeTest, Vopd3RejectsSrcY0Literal32Selector) {
  const uint32_t words[] = {
      0xCF455083u,
      0x000000FFu,
      0x0A000001u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decode_fails(*decoder, words));
}

TEST(Gfx1250DecodeTest, VopdLiteralConsumesThreeDwords) {
  const uint32_t words[] = {
      0xC8D006FFu, // v_dual_mul_f32 v4, 0x4f7ffffe, v3 :: v_dual_mov_b32 v3, 0
      0x04020080u,
      0x4F7FFFFEu,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
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
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
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
    EXPECT_TRUE(decode_fails(*decoder, encoding.data()));
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
    EXPECT_TRUE(decode_fails(*decoder, encoding.data()));
}

} // namespace
