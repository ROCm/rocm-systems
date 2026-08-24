// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file pseudo_scalar_exec_test.cpp
/// @brief Cross-architecture pseudo-scalar transcendental execution tests.

#include "decode_test_util.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/pseudo_scalar.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/except.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace {

using namespace rocjitsu;

using BaseEncodingWords = std::array<uint32_t, 2>;
using InstructionWords = std::array<uint32_t, 3>;
using EncodingLookup = std::optional<BaseEncodingWords> (*)(std::string_view);

template <typename Entry, size_t N>
std::optional<BaseEncodingWords> find_test_encoding(const Entry (&encodings)[N],
                                                    std::string_view mnemonic) {
  for (const auto &encoding : encodings) {
    if (encoding.mnemonic == mnemonic)
      return encoding.words;
  }
  return std::nullopt;
}

std::optional<BaseEncodingWords> rdna4_encoding(std::string_view mnemonic) {
  return find_test_encoding(rdna4::test_data::ENCODINGS, mnemonic);
}

std::optional<BaseEncodingWords> gfx1250_encoding(std::string_view mnemonic) {
  return find_test_encoding(cdna5::test_data::ENCODINGS, mnemonic);
}

struct PseudoScalarProfile {
  rj_code_arch_t arch;
  std::string_view name;
  EncodingLookup find_encoding;
  uint32_t setreg_op;
  uint32_t setreg_imm_op;
};

constexpr std::array<PseudoScalarProfile, 2> kProfiles{{
    {ROCJITSU_CODE_ARCH_RDNA4, "rdna4", rdna4_encoding, rdna4::kSSetregB32Sopk,
     rdna4::kSSetregImm32B32Sopk},
    {ROCJITSU_CODE_ARCH_CDNA5, "gfx1250", gfx1250_encoding, cdna5::kSSetregB32Sopk,
     cdna5::kSSetregImm32B32Sopk},
}};

struct PseudoScalarCase {
  std::string_view mnemonic;
  uint32_t source;
  uint32_t expected;
  bool set_source_opsel = false;
};

constexpr uint32_t f32_bits(float value) { return std::bit_cast<uint32_t>(value); }

constexpr std::array<PseudoScalarCase, 10> kCases{{
    {"v_s_exp_f32", f32_bits(0.0f), f32_bits(1.0f)},
    {"v_s_log_f32", f32_bits(1.0f), f32_bits(0.0f)},
    {"v_s_rcp_f32", f32_bits(2.0f), f32_bits(0.5f)},
    {"v_s_rsq_f32", f32_bits(4.0f), f32_bits(0.5f)},
    {"v_s_sqrt_f32", f32_bits(4.0f), f32_bits(2.0f)},
    {"v_s_exp_f16", 0xCAFE0000u, 0x00003C00u, true},
    {"v_s_log_f16", 0xCAFE3C00u, 0x00000000u, true},
    {"v_s_rcp_f16", 0xCAFE4000u, 0x00003800u, true},
    {"v_s_rsq_f16", 0xCAFE4400u, 0x00003800u, true},
    {"v_s_sqrt_f16", 0xCAFE4400u, 0x00004000u, true},
}};

struct PseudoScalarParam {
  const PseudoScalarProfile *profile = nullptr;
  const PseudoScalarCase *test_case = nullptr;
};

constexpr auto make_parameters() {
  std::array<PseudoScalarParam, kProfiles.size() * kCases.size()> parameters{};
  std::size_t index = 0;
  for (const auto &profile : kProfiles)
    for (const auto &test_case : kCases)
      parameters[index++] = {&profile, &test_case};
  return parameters;
}

constexpr auto kParameters = make_parameters();

struct Vop3EncodingOptions {
  bool source_opsel = false;
  bool abs_src0 = false;
  bool neg_src0 = false;
  bool clamp = false;
  uint32_t omod = 0;
  bool use_literal = false;
  uint32_t literal = 0;
};

InstructionWords encode_vop3(const PseudoScalarProfile &profile, std::string_view mnemonic,
                             uint32_t destination_sgpr, uint32_t source_sgpr,
                             Vop3EncodingOptions options = {}) {
  const auto base_encoding = profile.find_encoding(mnemonic);
  if (!base_encoding) {
    ADD_FAILURE() << profile.name << " missing generated encoding for " << mnemonic;
    return {};
  }

  InstructionWords words{(*base_encoding)[0], (*base_encoding)[1], options.literal};
  words[0] = (words[0] & ~0xFFu) | (destination_sgpr & 0xFFu);
  words[1] = (words[1] & ~0x1FFu) | (options.use_literal ? 255u : (source_sgpr & 0x1FFu));
  if (options.source_opsel)
    words[0] |= 1u << 11;
  if (options.abs_src0)
    words[0] |= 1u << 8;
  if (options.neg_src0)
    words[1] |= 1u << 29;
  if (options.clamp)
    words[0] |= 1u << 15;
  words[1] |= (options.omod & 0x3u) << 27;
  return words;
}

constexpr uint32_t encode_hwreg(uint32_t id, uint32_t offset = 0, uint32_t size = 32) {
  return (id & 0x3fu) | ((offset & 0x1fu) << 6) | (((size - 1u) & 0x1fu) << 11);
}

constexpr BaseEncodingWords encode_sopk(uint32_t op, uint32_t sdst, uint32_t simm16,
                                        uint32_t literal = 0) {
  return {(0xbu << 28) | ((op & 0x1fu) << 23) | ((sdst & 0x7fu) << 16) | (simm16 & 0xffffu),
          literal};
}

struct PseudoScalarSpecialCase {
  std::string_view name;
  std::string_view mnemonic;
  uint32_t source;
  uint32_t expected;
  std::string_view expected_operand_name;
  Vop3EncodingOptions encoding;
  uint32_t mode = 0;
};

constexpr std::array<PseudoScalarSpecialCase, 85> kSpecialCases{{
    {"literal_f32",
     "v_s_sqrt_f32",
     0,
     f32_bits(2.0f),
     "0x40800000",
     {.use_literal = true, .literal = f32_bits(4.0f)}},
    {"literal_f16_low_half",
     "v_s_sqrt_f16",
     0,
     0x00004000u,
     "0x4400",
     {.source_opsel = true, .use_literal = true, .literal = 0xCAFE4400u}},
    {"abs_modifier", "v_s_sqrt_f32", f32_bits(-4.0f), f32_bits(2.0f), "", {.abs_src0 = true}},
    {"neg_modifier", "v_s_rcp_f32", f32_bits(2.0f), f32_bits(-0.5f), "", {.neg_src0 = true}},
    {"omod_modifier", "v_s_sqrt_f32", f32_bits(4.0f), f32_bits(1.0f), "", {.omod = 3}},
    {"clamp_modifier", "v_s_sqrt_f32", f32_bits(4.0f), f32_bits(1.0f), "", {.clamp = true}},
    {"fp16_ovfl_mode",
     "v_s_exp_f16",
     0xCAFE4C00u,
     0x00007BFFu,
     "",
     {.source_opsel = true},
     amdgpu::Wavefront::FP16_OVFL_BIT},
    {"f32_input_denorm_flush", "v_s_log_f32", 0x00000001u, 0xFF800000u, "", {}},
    {"f32_input_denorm_allow", "v_s_log_f32", 0x00000001u, f32_bits(-149.0f), "", {}, 1u << 4},
    {"f32_output_denorm_allow", "v_s_exp_f32", f32_bits(-149.0f), 0x00000001u, "", {}, 1u << 5},
    {"f32_ignores_f16_output_denorm_mode",
     "v_s_exp_f32",
     f32_bits(-149.0f),
     0x00000000u,
     "",
     {},
     1u << 7},
    {"f16_input_denorm_allow",
     "v_s_log_f16",
     0xCAFE0001u,
     0x0000CE00u,
     "",
     {.source_opsel = true},
     1u << 6},
    {"f16_output_denorm_allow",
     "v_s_exp_f16",
     0xCAFECE00u,
     0x00000001u,
     "",
     {.source_opsel = true},
     1u << 7},
    {"f32_round_toward_positive", "v_s_exp_f32", f32_bits(0.5f), 0x3FB504F4u, "", {}, 1u},
    {"f16_round_toward_positive",
     "v_s_exp_f16",
     0xCAFE3800u,
     0x00003DA9u,
     "",
     {.source_opsel = true},
     1u << 2},
    {"f32_round_toward_negative", "v_s_sqrt_f32", f32_bits(5.0f), 0x400F1BBCu, "", {}, 2u},
    {"f32_round_toward_zero", "v_s_sqrt_f32", f32_bits(5.0f), 0x400F1BBCu, "", {}, 3u},
    {"f16_round_toward_negative",
     "v_s_sqrt_f16",
     0xCAFE4200u,
     0x00003EEDu,
     "",
     {.source_opsel = true},
     2u << 2},
    {"f16_round_toward_zero",
     "v_s_sqrt_f16",
     0xCAFE4200u,
     0x00003EEDu,
     "",
     {.source_opsel = true},
     3u << 2},
    {"f32_negative_round_toward_negative", "v_s_rcp_f32", f32_bits(-3.0f), 0xBEAAAAABu, "", {}, 2u},
    {"f32_negative_round_toward_zero", "v_s_rcp_f32", f32_bits(-3.0f), 0xBEAAAAAAu, "", {}, 3u},
    {"f16_negative_round_toward_negative",
     "v_s_rcp_f16",
     0xCAFEC200u,
     0x0000B556u,
     "",
     {.source_opsel = true},
     2u << 2},
    {"f16_negative_round_toward_zero",
     "v_s_rcp_f16",
     0xCAFEC200u,
     0x0000B555u,
     "",
     {.source_opsel = true},
     3u << 2},
    {"f32_finite_overflow_round_toward_negative",
     "v_s_exp_f32",
     f32_bits(1024.0f),
     0x7F7FFFFFu,
     "",
     {},
     2u},
    {"f32_finite_overflow_round_toward_zero",
     "v_s_exp_f32",
     f32_bits(2000.0f),
     0x7F7FFFFFu,
     "",
     {},
     3u},
    {"f32_finite_overflow_round_to_nearest", "v_s_exp_f32", f32_bits(1024.0f), 0x7F800000u, "", {}},
    {"f32_finite_overflow_round_toward_positive",
     "v_s_exp_f32",
     f32_bits(1024.0f),
     0x7F800000u,
     "",
     {},
     1u},
    {"f32_finite_underflow_round_toward_positive",
     "v_s_exp_f32",
     f32_bits(-2000.0f),
     0x00000001u,
     "",
     {},
     1u | (1u << 5)},
    {"f32_finite_underflow_round_toward_negative",
     "v_s_exp_f32",
     f32_bits(-2000.0f),
     0x00000000u,
     "",
     {},
     2u | (1u << 5)},
    {"f32_finite_underflow_round_to_nearest",
     "v_s_exp_f32",
     f32_bits(-2000.0f),
     0x00000000u,
     "",
     {},
     1u << 5},
    {"f32_finite_underflow_round_toward_zero",
     "v_s_exp_f32",
     f32_bits(-2000.0f),
     0x00000000u,
     "",
     {},
     3u | (1u << 5)},
    {"f32_destination_overflow_round_to_nearest",
     "v_s_exp_f32",
     f32_bits(128.0f),
     0x7F800000u,
     "",
     {}},
    {"f32_destination_overflow_round_toward_positive",
     "v_s_exp_f32",
     f32_bits(128.0f),
     0x7F800000u,
     "",
     {},
     1u},
    {"f32_destination_overflow_round_toward_negative",
     "v_s_exp_f32",
     f32_bits(128.0f),
     0x7F7FFFFFu,
     "",
     {},
     2u},
    {"f32_destination_overflow_round_toward_zero",
     "v_s_exp_f32",
     f32_bits(128.0f),
     0x7F7FFFFFu,
     "",
     {},
     3u},
    {"f32_destination_underflow_round_to_nearest",
     "v_s_exp_f32",
     f32_bits(-150.0f),
     0x00000000u,
     "",
     {},
     1u << 5},
    {"f32_destination_underflow_round_toward_positive",
     "v_s_exp_f32",
     f32_bits(-150.0f),
     0x00000001u,
     "",
     {},
     1u | (1u << 5)},
    {"f32_destination_underflow_round_toward_negative",
     "v_s_exp_f32",
     f32_bits(-150.0f),
     0x00000000u,
     "",
     {},
     2u | (1u << 5)},
    {"f32_destination_underflow_round_toward_zero",
     "v_s_exp_f32",
     f32_bits(-150.0f),
     0x00000000u,
     "",
     {},
     3u | (1u << 5)},
    {"f16_destination_overflow_round_to_nearest",
     "v_s_exp_f16",
     0xCAFE4C00u,
     0x00007C00u,
     "",
     {.source_opsel = true}},
    {"f16_destination_overflow_round_toward_positive",
     "v_s_exp_f16",
     0xCAFE4C00u,
     0x00007C00u,
     "",
     {.source_opsel = true},
     1u << 2},
    {"f16_destination_overflow_round_toward_negative",
     "v_s_exp_f16",
     0xCAFE4C00u,
     0x00007BFFu,
     "",
     {.source_opsel = true},
     2u << 2},
    {"f16_destination_overflow_round_toward_zero",
     "v_s_exp_f16",
     0xCAFE4C00u,
     0x00007BFFu,
     "",
     {.source_opsel = true},
     3u << 2},
    {"f16_destination_underflow_round_to_nearest",
     "v_s_exp_f16",
     0xCAFECE40u,
     0x00000000u,
     "",
     {.source_opsel = true},
     1u << 7},
    {"f16_destination_underflow_round_toward_positive",
     "v_s_exp_f16",
     0xCAFECE40u,
     0x00000001u,
     "",
     {.source_opsel = true},
     (1u << 2) | (1u << 7)},
    {"f16_destination_underflow_round_toward_negative",
     "v_s_exp_f16",
     0xCAFECE40u,
     0x00000000u,
     "",
     {.source_opsel = true},
     (2u << 2) | (1u << 7)},
    {"f16_destination_underflow_round_toward_zero",
     "v_s_exp_f16",
     0xCAFECE40u,
     0x00000000u,
     "",
     {.source_opsel = true},
     (3u << 2) | (1u << 7)},
    {"f32_rcp_omod_destination_overflow_round_to_nearest",
     "v_s_rcp_f32",
     0x00800000u,
     0x7F800000u,
     "",
     {.omod = 2}},
    {"f32_rcp_omod_destination_overflow_round_toward_positive",
     "v_s_rcp_f32",
     0x00800000u,
     0x7F800000u,
     "",
     {.omod = 2},
     1u},
    {"f32_rcp_omod_destination_overflow_round_toward_negative",
     "v_s_rcp_f32",
     0x00800000u,
     0x7F7FFFFFu,
     "",
     {.omod = 2},
     2u},
    {"f32_rcp_omod_destination_overflow_round_toward_zero",
     "v_s_rcp_f32",
     0x00800000u,
     0x7F7FFFFFu,
     "",
     {.omod = 2},
     3u},
    {"f16_rcp_omod_destination_overflow_round_to_nearest",
     "v_s_rcp_f16",
     0xCAFE0400u,
     0x00007C00u,
     "",
     {.source_opsel = true, .omod = 2}},
    {"f16_rcp_omod_destination_overflow_round_toward_positive",
     "v_s_rcp_f16",
     0xCAFE0400u,
     0x00007C00u,
     "",
     {.source_opsel = true, .omod = 2},
     1u << 2},
    {"f16_rcp_omod_destination_overflow_round_toward_negative",
     "v_s_rcp_f16",
     0xCAFE0400u,
     0x00007BFFu,
     "",
     {.source_opsel = true, .omod = 2},
     2u << 2},
    {"f16_rcp_omod_destination_overflow_round_toward_zero",
     "v_s_rcp_f16",
     0xCAFE0400u,
     0x00007BFFu,
     "",
     {.source_opsel = true, .omod = 2},
     3u << 2},
    {"f32_true_positive_infinity", "v_s_exp_f32", 0x7F800000u, 0x7F800000u, "", {}, 3u},
    {"f32_true_negative_infinity", "v_s_exp_f32", 0xFF800000u, 0x00000000u, "", {}, 1u | (1u << 5)},
    {"f32_divide_by_zero", "v_s_rcp_f32", 0x00000000u, 0x7F800000u, "", {}},
    {"f32_negative_divide_by_zero", "v_s_rcp_f32", 0x80000000u, 0xFF800000u, "", {}},
    {"f32_log_negative_domain", "v_s_log_f32", f32_bits(-1.0f), 0xFFC00000u, "", {}},
    {"f32_rsq_negative_domain", "v_s_rsq_f32", f32_bits(-1.0f), 0xFFC00000u, "", {}},
    {"f32_sqrt_negative_domain", "v_s_sqrt_f32", f32_bits(-1.0f), 0xFFC00000u, "", {}},
    {"f16_log_negative_domain",
     "v_s_log_f16",
     0xCAFEBC00u,
     0x0000FE00u,
     "",
     {.source_opsel = true}},
    {"f16_rsq_negative_domain",
     "v_s_rsq_f16",
     0xCAFEBC00u,
     0x0000FE00u,
     "",
     {.source_opsel = true}},
    {"f16_sqrt_negative_domain",
     "v_s_sqrt_f16",
     0xCAFEBC00u,
     0x0000FE00u,
     "",
     {.source_opsel = true}},
    {"f16_finite_overflow_round_toward_zero",
     "v_s_exp_f16",
     0xCAFE7BFFu,
     0x00007BFFu,
     "",
     {.source_opsel = true},
     3u << 2},
    {"f16_finite_overflow_round_to_nearest",
     "v_s_exp_f16",
     0xCAFE7BFFu,
     0x00007C00u,
     "",
     {.source_opsel = true}},
    {"f16_finite_overflow_round_toward_positive",
     "v_s_exp_f16",
     0xCAFE7BFFu,
     0x00007C00u,
     "",
     {.source_opsel = true},
     1u << 2},
    {"f16_finite_overflow_round_toward_negative",
     "v_s_exp_f16",
     0xCAFE7BFFu,
     0x00007BFFu,
     "",
     {.source_opsel = true},
     2u << 2},
    {"f16_finite_overflow_fp16_ovfl",
     "v_s_exp_f16",
     0xCAFE7BFFu,
     0x00007BFFu,
     "",
     {.source_opsel = true},
     amdgpu::Wavefront::FP16_OVFL_BIT},
    {"f16_finite_underflow_round_toward_positive",
     "v_s_exp_f16",
     0xCAFEFBFFu,
     0x00000001u,
     "",
     {.source_opsel = true},
     (1u << 2) | (1u << 7)},
    {"f16_finite_underflow_round_to_nearest",
     "v_s_exp_f16",
     0xCAFEFBFFu,
     0x00000000u,
     "",
     {.source_opsel = true},
     1u << 7},
    {"f16_finite_underflow_round_toward_negative",
     "v_s_exp_f16",
     0xCAFEFBFFu,
     0x00000000u,
     "",
     {.source_opsel = true},
     (2u << 2) | (1u << 7)},
    {"f16_finite_underflow_round_toward_zero",
     "v_s_exp_f16",
     0xCAFEFBFFu,
     0x00000000u,
     "",
     {.source_opsel = true},
     (3u << 2) | (1u << 7)},
    {"f16_true_positive_infinity_with_fp16_ovfl",
     "v_s_exp_f16",
     0xCAFE7C00u,
     0x00007C00u,
     "",
     {.source_opsel = true},
     amdgpu::Wavefront::FP16_OVFL_BIT | (3u << 2)},
    {"f16_divide_by_zero_with_fp16_ovfl",
     "v_s_rcp_f16",
     0xCAFE0000u,
     0x00007C00u,
     "",
     {.source_opsel = true},
     amdgpu::Wavefront::FP16_OVFL_BIT},
    {"f32_omod_flushes_output_denorm",
     "v_s_exp_f32",
     f32_bits(-149.0f),
     0x00000000u,
     "",
     {.omod = 1},
     1u << 5},
    {"f32_omod_normalizes_negative_zero",
     "v_s_sqrt_f32",
     0x80000000u,
     0x00000000u,
     "",
     {.omod = 1}},
    {"f32_clamp_maps_nan_to_zero",
     "v_s_sqrt_f32",
     f32_bits(-1.0f),
     0x00000000u,
     "",
     {.clamp = true}},
    {"f16_omod_flushes_output_denorm",
     "v_s_exp_f16",
     0xCAFECE00u,
     0x00000000u,
     "",
     {.source_opsel = true, .omod = 1},
     1u << 7},
    {"f16_clamp_maps_nan_to_zero",
     "v_s_sqrt_f16",
     0xCAFEBC00u,
     0x00000000u,
     "",
     {.source_opsel = true, .clamp = true}},
    {"f32_clamp_normalizes_negative_zero",
     "v_s_sqrt_f32",
     0x80000000u,
     0x00000000u,
     "",
     {.clamp = true}},
    {"f32_unclamped_preserves_negative_zero", "v_s_sqrt_f32", 0x80000000u, 0x80000000u, "", {}},
    {"f16_clamp_normalizes_negative_zero",
     "v_s_sqrt_f16",
     0xCAFE8000u,
     0x00000000u,
     "",
     {.source_opsel = true, .clamp = true}},
    {"f16_unclamped_preserves_negative_zero",
     "v_s_sqrt_f16",
     0xCAFE8000u,
     0x00008000u,
     "",
     {.source_opsel = true}},
}};

struct PseudoScalarSpecialParam {
  const PseudoScalarProfile *profile = nullptr;
  const PseudoScalarSpecialCase *test_case = nullptr;
};

constexpr auto make_special_parameters() {
  std::array<PseudoScalarSpecialParam, kProfiles.size() * kSpecialCases.size()> parameters{};
  std::size_t index = 0;
  for (const auto &profile : kProfiles)
    for (const auto &test_case : kSpecialCases)
      parameters[index++] = {&profile, &test_case};
  return parameters;
}

constexpr auto kSpecialParameters = make_special_parameters();

class PseudoScalarFixture {
public:
  explicit PseudoScalarFixture(const PseudoScalarProfile &profile)
      : profile(profile), gpu_memory(std::string(profile.name) + "_pseudo_scalar_memory"),
        l2(std::string(profile.name) + "_pseudo_scalar_l2") {
    config.arch = profile.arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    compute_unit =
        amdgpu::ComputeUnitCore::create(std::string(profile.name), config, &gpu_memory, &l2);
    decoder = Decoder::create(profile.arch);
    if (compute_unit)
      wavefront = compute_unit->dispatch_wf(0, 0, config.sgprs_per_wf, config.vgprs_per_wf);
  }

  ~PseudoScalarFixture() {
    if (wavefront && !wavefront->is_halted())
      wavefront->halt();
  }

  bool ready() const {
    return compute_unit != nullptr && decoder != nullptr && wavefront != nullptr;
  }

  uint32_t sgpr_base() const { return wavefront->sgpr_alloc().base; }

  const PseudoScalarProfile &profile;
  amdgpu::GpuMemory gpu_memory;
  amdgpu::L2Cache l2;
  amdgpu::ComputeUnitCore::Config config{};
  std::unique_ptr<amdgpu::ComputeUnitCore> compute_unit;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wavefront = nullptr;
};

class PseudoScalarExecTest : public ::testing::TestWithParam<PseudoScalarParam> {};
class PseudoScalarSpecialExecTest : public ::testing::TestWithParam<PseudoScalarSpecialParam> {};

std::string pseudo_scalar_param_name(const ::testing::TestParamInfo<PseudoScalarParam> &info) {
  return std::string(info.param.profile->name) + "_" + std::string(info.param.test_case->mnemonic);
}

std::string
pseudo_scalar_special_param_name(const ::testing::TestParamInfo<PseudoScalarSpecialParam> &info) {
  return std::string(info.param.profile->name) + "_" + std::string(info.param.test_case->name);
}

TEST_P(PseudoScalarExecTest, ExecutesAtExecZeroWithDestinationOpselZeroAndUsesLowF16Half) {
  constexpr uint32_t kSourceSgpr = 0;
  constexpr uint32_t kDestinationSgpr = 4;
  constexpr uint32_t kDestinationSentinel = 0xDEADBEEFu;
  constexpr uint32_t kDestinationOpselBit = 1u << 14;

  const PseudoScalarProfile &profile = *GetParam().profile;
  const PseudoScalarCase &test_case = *GetParam().test_case;
  PseudoScalarFixture fixture(profile);
  ASSERT_TRUE(fixture.ready()) << profile.name;

  const InstructionWords words =
      encode_vop3(profile, test_case.mnemonic, kDestinationSgpr, kSourceSgpr,
                  {.source_opsel = test_case.set_source_opsel});
  ASSERT_EQ(words[0] & kDestinationOpselBit, 0u) << "OPSEL[3] must select the low destination half";
  std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
  ASSERT_NE(instruction, nullptr) << profile.name << " " << test_case.mnemonic;
  ASSERT_EQ(std::string_view(instruction->mnemonic()), test_case.mnemonic) << profile.name;
  EXPECT_EQ(instruction->size(), 8u);

  fixture.compute_unit->write_sgpr(fixture.sgpr_base() + kSourceSgpr, test_case.source);
  fixture.compute_unit->write_sgpr(fixture.sgpr_base() + kDestinationSgpr, kDestinationSentinel);
  fixture.wavefront->set_exec(0);

  fixture.compute_unit->execute_instruction(instruction.get(), *fixture.wavefront);

  EXPECT_EQ(fixture.compute_unit->read_sgpr(fixture.sgpr_base() + kDestinationSgpr),
            test_case.expected);
  EXPECT_EQ(fixture.wavefront->exec(), 0u);
}

TEST_P(PseudoScalarExecTest, ExecutesWithVccAsSourceAndDestination) {
  constexpr std::array<uint32_t, 2> kVccSelectors{{106u, 107u}};
  constexpr uint32_t kOtherHalfSentinel = 0xDEADBEEFu;

  const PseudoScalarProfile &profile = *GetParam().profile;
  const PseudoScalarCase &test_case = *GetParam().test_case;
  for (const uint32_t selector : kVccSelectors) {
    SCOPED_TRACE(selector);
    PseudoScalarFixture fixture(profile);
    ASSERT_TRUE(fixture.ready()) << profile.name;

    const InstructionWords words = encode_vop3(profile, test_case.mnemonic, selector, selector,
                                               {.source_opsel = test_case.set_source_opsel});
    std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
    ASSERT_NE(instruction, nullptr);

    const uint64_t source = selector == kVccSelectors[0]
                                ? (uint64_t{kOtherHalfSentinel} << 32) | test_case.source
                                : (uint64_t{test_case.source} << 32) | kOtherHalfSentinel;
    const uint64_t expected = selector == kVccSelectors[0]
                                  ? (uint64_t{kOtherHalfSentinel} << 32) | test_case.expected
                                  : (uint64_t{test_case.expected} << 32) | kOtherHalfSentinel;
    fixture.wavefront->set_vcc_raw(source);
    fixture.compute_unit->execute_instruction(instruction.get(), *fixture.wavefront);
    EXPECT_EQ(fixture.wavefront->vcc(), expected);
  }
}

TEST(PseudoScalarDecodeTest, RejectsOutOfClassDestinations) {
  constexpr std::array<uint32_t, 4> kInvalidSelectors{{126u, 127u, 128u, 255u}};
  constexpr uint32_t kSourceSgpr = 0;

  for (const PseudoScalarProfile &profile : kProfiles) {
    SCOPED_TRACE(profile.name);
    std::unique_ptr<Decoder> decoder = Decoder::create(profile.arch);
    ASSERT_NE(decoder, nullptr);
    for (const PseudoScalarCase &test_case : kCases) {
      SCOPED_TRACE(test_case.mnemonic);
      for (const uint32_t selector : kInvalidSelectors) {
        SCOPED_TRACE(selector);
        const InstructionWords words =
            encode_vop3(profile, test_case.mnemonic, selector, kSourceSgpr);
        EXPECT_TRUE(decode_fails(*decoder, words.data()));
      }
    }
  }
}

TEST(PseudoScalarDecodeTest, RejectsDppAndDpp8Sources) {
  constexpr uint32_t kDestinationSgpr = 4;
  constexpr std::array<uint32_t, 2> kUnsupportedSources{{amdgpu::SRC_DPP, amdgpu::SRC_DPP8_FI_0}};

  for (const PseudoScalarProfile &profile : kProfiles) {
    std::unique_ptr<Decoder> decoder = Decoder::create(profile.arch);
    ASSERT_NE(decoder, nullptr) << profile.name;
    for (const uint32_t source : kUnsupportedSources) {
      const InstructionWords words = encode_vop3(profile, "v_s_sqrt_f32", kDestinationSgpr, source);
      EXPECT_TRUE(decode_fails(*decoder, words.data())) << profile.name << " source=" << source;
    }
  }
}

TEST(PseudoScalarHelperTest, HandlesExplicitSpecialCasesWithoutHostInvalidOrDivideByZero) {
  using amdgpu::pseudo_scalar::Operation;

  fenv_t saved_environment{};
  ASSERT_EQ(std::fegetenv(&saved_environment), 0);
  ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
  const uint32_t f32_rcp_zero =
      amdgpu::pseudo_scalar::execute_f32(Operation::RCP, 0.0f, false, false, 0, 3, 0, false);
  const uint32_t f32_rcp_negative_zero =
      amdgpu::pseudo_scalar::execute_f32(Operation::RCP, -0.0f, false, false, 0, 3, 0, false);
  const uint32_t f32_sqrt_negative =
      amdgpu::pseudo_scalar::execute_f32(Operation::SQRT, -1.0f, false, false, 0, 3, 0, false);
  const uint32_t f32_rsq_negative =
      amdgpu::pseudo_scalar::execute_f32(Operation::RSQ, -1.0f, false, false, 0, 3, 0, false);
  const uint32_t f32_log_zero =
      amdgpu::pseudo_scalar::execute_f32(Operation::LOG2, 0.0f, false, false, 0, 3, 0, false);
  const uint32_t f32_log_negative =
      amdgpu::pseudo_scalar::execute_f32(Operation::LOG2, -1.0f, false, false, 0, 3, 0, false);
  const uint32_t f32_exp_positive_infinity = amdgpu::pseudo_scalar::execute_f32(
      Operation::EXP2, std::bit_cast<float>(0x7F800000u), false, false, 0, 3, 0, false);
  const uint32_t f32_exp_negative_infinity = amdgpu::pseudo_scalar::execute_f32(
      Operation::EXP2, std::bit_cast<float>(0xFF800000u), false, false, 0, 3, 0, false);
  const uint32_t f32_signaling_nan = amdgpu::pseudo_scalar::execute_f32(
      Operation::SQRT, std::bit_cast<float>(0x7FA00001u), false, false, 0, 3, 0, false);
  const uint32_t f16_rcp_zero = amdgpu::pseudo_scalar::execute_f16(
      Operation::RCP, util::f16_to_f32(0x0000u), false, false, 0, 3, 0, false, true);
  const uint32_t f16_sqrt_negative = amdgpu::pseudo_scalar::execute_f16(
      Operation::SQRT, util::f16_to_f32(0xBC00u), false, false, 0, 3, 0, false, false);
  const uint32_t f16_rsq_negative = amdgpu::pseudo_scalar::execute_f16(
      Operation::RSQ, util::f16_to_f32(0xBC00u), false, false, 0, 3, 0, false, false);
  const uint32_t f16_log_negative = amdgpu::pseudo_scalar::execute_f16(
      Operation::LOG2, util::f16_to_f32(0xBC00u), false, false, 0, 3, 0, false, false);
  const uint32_t f16_signaling_nan = amdgpu::pseudo_scalar::execute_f16(
      Operation::SQRT, util::f16_to_f32(0x7D01u), false, false, 0, 3, 0, false, false);
  const int leaked_exceptions = std::fetestexcept(FE_INVALID | FE_DIVBYZERO);
  const int restore_result = std::fesetenv(&saved_environment);

  EXPECT_EQ(restore_result, 0);
  EXPECT_EQ(leaked_exceptions, 0);
  EXPECT_EQ(f32_rcp_zero, 0x7F800000u);
  EXPECT_EQ(f32_rcp_negative_zero, 0xFF800000u);
  EXPECT_EQ(f32_sqrt_negative, 0xFFC00000u);
  EXPECT_EQ(f32_rsq_negative, 0xFFC00000u);
  EXPECT_EQ(f32_log_zero, 0xFF800000u);
  EXPECT_EQ(f32_log_negative, 0xFFC00000u);
  EXPECT_EQ(f32_exp_positive_infinity, 0x7F800000u);
  EXPECT_EQ(f32_exp_negative_infinity, 0x00000000u);
  EXPECT_TRUE(std::isnan(std::bit_cast<float>(f32_signaling_nan)));
  EXPECT_NE(f32_signaling_nan & 0x00400000u, 0u);
  EXPECT_EQ(f16_rcp_zero, 0x00007C00u);
  EXPECT_EQ(f16_sqrt_negative, 0x0000FE00u);
  EXPECT_EQ(f16_rsq_negative, 0x0000FE00u);
  EXPECT_EQ(f16_log_negative, 0x0000FE00u);
  EXPECT_EQ(f16_signaling_nan & 0x7C00u, 0x7C00u);
  EXPECT_NE(f16_signaling_nan & 0x03FFu, 0u);
  EXPECT_NE(f16_signaling_nan & 0x0200u, 0u);
}

TEST(PseudoScalarHelperTest, PreservesAndFlushesSignedDenormals) {
  using amdgpu::pseudo_scalar::Operation;

  const float f32_negative_minimum = std::bit_cast<float>(0x80000001u);
  const float f32_negative_maximum = std::bit_cast<float>(0xFF7FFFFFu);
  EXPECT_EQ(amdgpu::pseudo_scalar::execute_f32(Operation::SQRT, f32_negative_minimum, false, false,
                                               0, 0, 0, false),
            0x80000000u);
  const uint32_t f32_allowed_negative_input = amdgpu::pseudo_scalar::execute_f32(
      Operation::SQRT, f32_negative_minimum, false, false, 0, 1, 0, false);
  EXPECT_TRUE(std::isnan(std::bit_cast<float>(f32_allowed_negative_input)));
  EXPECT_EQ(amdgpu::pseudo_scalar::execute_f32(Operation::RCP, f32_negative_maximum, false, false,
                                               0, 2, 0, false),
            0x80200000u);
  EXPECT_EQ(amdgpu::pseudo_scalar::execute_f32(Operation::RCP, f32_negative_maximum, false, false,
                                               0, 0, 0, false),
            0x80000000u);

  const float f16_negative_minimum = util::f16_to_f32(0x8001u);
  const float f16_negative_maximum = util::f16_to_f32(0xFBFFu);
  EXPECT_EQ(amdgpu::pseudo_scalar::execute_f16(Operation::SQRT, f16_negative_minimum, false, false,
                                               0, 0, 0, false, false),
            0x00008000u);
  const uint32_t f16_allowed_negative_input = amdgpu::pseudo_scalar::execute_f16(
      Operation::SQRT, f16_negative_minimum, false, false, 0, 1, 0, false, false);
  EXPECT_EQ(f16_allowed_negative_input & 0x7C00u, 0x7C00u);
  EXPECT_NE(f16_allowed_negative_input & 0x03FFu, 0u);
  EXPECT_EQ(amdgpu::pseudo_scalar::execute_f16(Operation::RCP, f16_negative_maximum, false, false,
                                               0, 2, 0, false, false),
            0x00008100u);
  EXPECT_EQ(amdgpu::pseudo_scalar::execute_f16(Operation::RCP, f16_negative_maximum, false, false,
                                               0, 0, 0, false, false),
            0x00008000u);
}

INSTANTIATE_TEST_SUITE_P(AllProfilesAndInstructions, PseudoScalarExecTest,
                         ::testing::ValuesIn(kParameters), pseudo_scalar_param_name);

TEST_P(PseudoScalarSpecialExecTest, CoversLiteralModifierAndModeBehavior) {
  constexpr uint32_t kSourceSgpr = 0;
  constexpr uint32_t kDestinationSgpr = 4;
  constexpr uint32_t kDestinationSentinel = 0xDEADBEEFu;

  const PseudoScalarProfile &profile = *GetParam().profile;
  const PseudoScalarSpecialCase &test_case = *GetParam().test_case;
  PseudoScalarFixture fixture(profile);
  ASSERT_TRUE(fixture.ready()) << profile.name;

  const InstructionWords words =
      encode_vop3(profile, test_case.mnemonic, kDestinationSgpr, kSourceSgpr, test_case.encoding);
  std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words.data()));
  ASSERT_NE(instruction, nullptr) << profile.name << " " << test_case.mnemonic;
  ASSERT_EQ(std::string_view(instruction->mnemonic()), test_case.mnemonic) << profile.name;
  EXPECT_EQ(instruction->size(), test_case.encoding.use_literal ? 12u : 8u);

  if (!test_case.expected_operand_name.empty()) {
    ASSERT_EQ(instruction->num_src_operands(), 1);
    ASSERT_NE(instruction->src_operand(0), nullptr);
    EXPECT_EQ(instruction->src_operand(0)->name(), test_case.expected_operand_name);
  }

  fixture.compute_unit->write_sgpr(fixture.sgpr_base() + kSourceSgpr, test_case.source);
  fixture.compute_unit->write_sgpr(fixture.sgpr_base() + kDestinationSgpr, kDestinationSentinel);
  fixture.wavefront->set_mode_raw(test_case.mode);
  fixture.wavefront->set_exec(0);

  fixture.compute_unit->execute_instruction(instruction.get(), *fixture.wavefront);

  EXPECT_EQ(fixture.compute_unit->read_sgpr(fixture.sgpr_base() + kDestinationSgpr),
            test_case.expected);
  EXPECT_EQ(fixture.wavefront->exec(), 0u);
  EXPECT_EQ(fixture.wavefront->mode_raw(), test_case.mode);
}

INSTANTIATE_TEST_SUITE_P(AllProfilesAndSpecialCases, PseudoScalarSpecialExecTest,
                         ::testing::ValuesIn(kSpecialParameters), pseudo_scalar_special_param_name);

TEST(PseudoScalarModeIntegrationTest, SetregInstructionsUpdateModesConsumedByPseudoScalars) {
  constexpr uint32_t kSourceSgpr = 0;
  constexpr uint32_t kModeSgpr = 2;
  constexpr uint32_t kDestinationSgpr = 4;
  constexpr uint32_t kModeHwreg = 1;

  for (const PseudoScalarProfile &profile : kProfiles) {
    SCOPED_TRACE(profile.name);
    PseudoScalarFixture fixture(profile);
    ASSERT_TRUE(fixture.ready());

    const BaseEncodingWords set_round_words =
        encode_sopk(profile.setreg_op, kModeSgpr, encode_hwreg(kModeHwreg));
    std::unique_ptr<Instruction> set_round(decode_valid(*fixture.decoder, set_round_words.data()));
    ASSERT_NE(set_round, nullptr);
    ASSERT_EQ(std::string_view(set_round->mnemonic()), "s_setreg_b32");

    const InstructionWords exp_words =
        encode_vop3(profile, "v_s_exp_f32", kDestinationSgpr, kSourceSgpr);
    std::unique_ptr<Instruction> exp(decode_valid(*fixture.decoder, exp_words.data()));
    ASSERT_NE(exp, nullptr);

    fixture.wavefront->set_mode_raw(0);
    fixture.compute_unit->write_sgpr(fixture.sgpr_base() + kModeSgpr, 1u);
    fixture.compute_unit->write_sgpr(fixture.sgpr_base() + kSourceSgpr, f32_bits(0.5f));
    fixture.compute_unit->execute_instruction(set_round.get(), *fixture.wavefront);
    fixture.compute_unit->execute_instruction(exp.get(), *fixture.wavefront);
    EXPECT_EQ(fixture.wavefront->mode_raw(), 1u);
    EXPECT_EQ(fixture.compute_unit->read_sgpr(fixture.sgpr_base() + kDestinationSgpr), 0x3FB504F4u);

    const BaseEncodingWords set_denorm_words =
        encode_sopk(profile.setreg_imm_op, 0, encode_hwreg(kModeHwreg, 6, 2), 1u);
    std::unique_ptr<Instruction> set_denorm(
        decode_valid(*fixture.decoder, set_denorm_words.data()));
    ASSERT_NE(set_denorm, nullptr);
    ASSERT_EQ(std::string_view(set_denorm->mnemonic()), "s_setreg_imm32_b32");

    const InstructionWords log_words =
        encode_vop3(profile, "v_s_log_f16", kDestinationSgpr, kSourceSgpr, {.source_opsel = true});
    std::unique_ptr<Instruction> log(decode_valid(*fixture.decoder, log_words.data()));
    ASSERT_NE(log, nullptr);

    fixture.wavefront->set_mode_raw(0);
    fixture.compute_unit->write_sgpr(fixture.sgpr_base() + kSourceSgpr, 0xCAFE0001u);
    fixture.compute_unit->execute_instruction(set_denorm.get(), *fixture.wavefront);
    fixture.compute_unit->execute_instruction(log.get(), *fixture.wavefront);
    EXPECT_EQ(fixture.wavefront->mode_raw(), 1u << 6);
    EXPECT_EQ(fixture.compute_unit->read_sgpr(fixture.sgpr_base() + kDestinationSgpr), 0x0000CE00u);
  }
}

} // namespace
