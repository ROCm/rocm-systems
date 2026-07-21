// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file scalar_scc_test.cpp
/// @brief Cross-architecture scalar SCC execution and preservation tests.

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"

#include "rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/test_encodings.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace rocjitsu;

using EncodingWords = std::array<uint32_t, 2>;
using EncodingLookup = std::optional<EncodingWords> (*)(std::string_view);

enum class WrexecDialect {
  Andn,
  AndNot,
};

template <typename Entry, size_t N>
std::optional<EncodingWords> find_test_encoding(const Entry (&encodings)[N],
                                                std::string_view mnemonic) {
  for (const auto &encoding : encodings) {
    if (encoding.mnemonic == mnemonic)
      return encoding.words;
  }
  return std::nullopt;
}

std::optional<EncodingWords> cdna1_encoding(std::string_view mnemonic) {
  return find_test_encoding(cdna1::test_data::ENCODINGS, mnemonic);
}

std::optional<EncodingWords> cdna2_encoding(std::string_view mnemonic) {
  return find_test_encoding(cdna2::test_data::ENCODINGS, mnemonic);
}

std::optional<EncodingWords> cdna3_encoding(std::string_view mnemonic) {
  return find_test_encoding(cdna3::test_data::ENCODINGS, mnemonic);
}

std::optional<EncodingWords> cdna4_encoding(std::string_view mnemonic) {
  return find_test_encoding(cdna4::test_data::ENCODINGS, mnemonic);
}

std::optional<EncodingWords> rdna1_encoding(std::string_view mnemonic) {
  return find_test_encoding(rdna1::test_data::ENCODINGS, mnemonic);
}

std::optional<EncodingWords> rdna2_encoding(std::string_view mnemonic) {
  return find_test_encoding(rdna2::test_data::ENCODINGS, mnemonic);
}

std::optional<EncodingWords> rdna3_encoding(std::string_view mnemonic) {
  return find_test_encoding(rdna3::test_data::ENCODINGS, mnemonic);
}

std::optional<EncodingWords> rdna3_5_encoding(std::string_view mnemonic) {
  return find_test_encoding(rdna3_5::test_data::ENCODINGS, mnemonic);
}

std::optional<EncodingWords> rdna4_encoding(std::string_view mnemonic) {
  return find_test_encoding(rdna4::test_data::ENCODINGS, mnemonic);
}

std::optional<EncodingWords> gfx1250_encoding(std::string_view mnemonic) {
  return find_test_encoding(gfx1250::test_data::ENCODINGS, mnemonic);
}

class ScalarSccProfile {
public:
  rj_code_arch_t arch;
  std::string_view name;
  EncodingLookup find_encoding;
  WrexecDialect wrexec_dialect;
  bool has_wrexec_b32;
};

const std::array<ScalarSccProfile, 10> kScalarSccProfiles{{
    {ROCJITSU_CODE_ARCH_CDNA1, "cdna1", cdna1_encoding, WrexecDialect::Andn, false},
    {ROCJITSU_CODE_ARCH_CDNA2, "cdna2", cdna2_encoding, WrexecDialect::Andn, false},
    {ROCJITSU_CODE_ARCH_CDNA3, "cdna3", cdna3_encoding, WrexecDialect::Andn, false},
    {ROCJITSU_CODE_ARCH_CDNA4, "cdna4", cdna4_encoding, WrexecDialect::Andn, false},
    {ROCJITSU_CODE_ARCH_RDNA1, "rdna1", rdna1_encoding, WrexecDialect::Andn, true},
    {ROCJITSU_CODE_ARCH_RDNA2, "rdna2", rdna2_encoding, WrexecDialect::Andn, true},
    {ROCJITSU_CODE_ARCH_RDNA3, "rdna3", rdna3_encoding, WrexecDialect::AndNot, true},
    {ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", rdna3_5_encoding, WrexecDialect::AndNot, true},
    {ROCJITSU_CODE_ARCH_RDNA4, "rdna4", rdna4_encoding, WrexecDialect::AndNot, true},
    {ROCJITSU_CODE_ARCH_GFX1250, "gfx1250", gfx1250_encoding, WrexecDialect::AndNot, true},
}};

EncodingWords require_encoding(const ScalarSccProfile &profile, std::string_view mnemonic) {
  auto words = profile.find_encoding(mnemonic);
  if (!words) {
    ADD_FAILURE() << profile.name << " missing generated encoding for " << mnemonic;
    return {};
  }
  return *words;
}

std::optional<std::string_view> first_supported(const ScalarSccProfile &profile,
                                                std::initializer_list<std::string_view> mnemonics) {
  for (const auto mnemonic : mnemonics) {
    if (profile.find_encoding(mnemonic))
      return mnemonic;
  }
  return std::nullopt;
}

std::string_view require_first_supported(const ScalarSccProfile &profile,
                                         std::initializer_list<std::string_view> mnemonics,
                                         std::string_view instruction_family) {
  auto mnemonic = first_supported(profile, mnemonics);
  if (!mnemonic) {
    ADD_FAILURE() << profile.name << " has no encoding for " << instruction_family;
    return {};
  }
  return *mnemonic;
}

EncodingWords encode_sop1(const ScalarSccProfile &profile, std::string_view mnemonic, uint32_t sdst,
                          uint32_t ssrc0) {
  auto words = require_encoding(profile, mnemonic);
  constexpr uint32_t kOperandMask = 0xFFu | (0x7Fu << 16);
  words[0] = (words[0] & ~kOperandMask) | (ssrc0 & 0xFFu) | ((sdst & 0x7Fu) << 16);
  return words;
}

EncodingWords encode_sop2(const ScalarSccProfile &profile, std::string_view mnemonic, uint32_t sdst,
                          uint32_t ssrc0, uint32_t ssrc1) {
  auto words = require_encoding(profile, mnemonic);
  constexpr uint32_t kOperandMask = 0xFFFFu | (0x7Fu << 16);
  words[0] = (words[0] & ~kOperandMask) | (ssrc0 & 0xFFu) | ((ssrc1 & 0xFFu) << 8) |
             ((sdst & 0x7Fu) << 16);
  return words;
}

EncodingWords encode_sopc(const ScalarSccProfile &profile, std::string_view mnemonic,
                          uint32_t ssrc0, uint32_t ssrc1) {
  auto words = require_encoding(profile, mnemonic);
  words[0] = (words[0] & ~0xFFFFu) | (ssrc0 & 0xFFu) | ((ssrc1 & 0xFFu) << 8);
  return words;
}

EncodingWords encode_sopk(const ScalarSccProfile &profile, std::string_view mnemonic, uint32_t sdst,
                          uint32_t simm16) {
  auto words = require_encoding(profile, mnemonic);
  constexpr uint32_t kOperandMask = 0xFFFFu | (0x7Fu << 16);
  words[0] = (words[0] & ~kOperandMask) | (simm16 & 0xFFFFu) | ((sdst & 0x7Fu) << 16);
  return words;
}

class ScalarSccFixture {
public:
  ScalarSccFixture(const ScalarSccProfile &profile, std::string_view test_name)
      : profile(profile),
        gpu_mem(std::string(profile.name) + "_" + std::string(test_name) + "_mem"),
        l2(std::string(profile.name) + "_" + std::string(test_name) + "_l2") {
    config.arch = profile.arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create(std::string(profile.name), config, &gpu_mem, &l2);
    decoder = Decoder::create(profile.arch);
    if (cu)
      wf = cu->dispatch_wf(0, 0, config.sgprs_per_wf, config.vgprs_per_wf);
  }

  ~ScalarSccFixture() {
    // A wave frees its SGPR/VGPR blocks when it halts (s_endpgm-equivalent
    // termination), so halt the resident wave to release its resources rather
    // than a CU-wide reset.
    if (wf && !wf->is_halted())
      wf->halt();
  }

  bool ready() const { return cu != nullptr && decoder != nullptr && wf != nullptr; }

  std::unique_ptr<Instruction> decode(const EncodingWords &words,
                                      std::string_view expected_mnemonic) {
    std::unique_ptr<Instruction> inst(decoder->decode(words.data()));
    if (inst) {
      EXPECT_EQ(std::string_view(inst->mnemonic()), expected_mnemonic) << profile.name;
    }
    return inst;
  }

  void expect_scc_consumer(bool expected_scc, std::string_view context) {
    constexpr uint32_t kCmovSentinel = 0xDEADBEEFu;
    const auto cmov_words = encode_sopk(profile, "s_cmovk_i32", /*sdst=*/8, /*simm16=*/0x1234u);
    auto cmov = decode(cmov_words, "s_cmovk_i32");
    ASSERT_NE(cmov, nullptr) << profile.name;
    EXPECT_EQ(wf->read_scc(), expected_scc) << profile.name << " " << context;
    cu->write_sgpr(sgpr_base() + 8, kCmovSentinel);
    cu->execute_instruction(cmov.get(), *wf);
    EXPECT_EQ(cu->read_sgpr(sgpr_base() + 8), expected_scc ? 0x1234u : kCmovSentinel)
        << profile.name << " " << context;
  }

  uint32_t sgpr_base() const { return wf->sgpr_alloc().base; }

  const ScalarSccProfile &profile;
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  amdgpu::ComputeUnitCore::Config config{};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;
};

class WrexecPair {
public:
  WrexecDialect dialect;
  std::string_view first;
  std::string_view second;
  bool is_b64;
};

constexpr std::array<WrexecPair, 4> kWrexecPairs{{
    {WrexecDialect::Andn, "s_andn1_wrexec_b32", "s_andn2_wrexec_b32", false},
    {WrexecDialect::Andn, "s_andn1_wrexec_b64", "s_andn2_wrexec_b64", true},
    {WrexecDialect::AndNot, "s_and_not0_wrexec_b32", "s_and_not1_wrexec_b32", false},
    {WrexecDialect::AndNot, "s_and_not0_wrexec_b64", "s_and_not1_wrexec_b64", true},
}};

bool profile_expects_wrexec_pair(const ScalarSccProfile &profile, const WrexecPair &pair) {
  return profile.wrexec_dialect == pair.dialect && (pair.is_b64 || profile.has_wrexec_b32);
}

bool expect_wrexec_availability(const ScalarSccProfile &profile, const WrexecPair &pair) {
  const bool expected = profile_expects_wrexec_pair(profile, pair);
  const bool first_available = profile.find_encoding(pair.first).has_value();
  const bool second_available = profile.find_encoding(pair.second).has_value();
  EXPECT_EQ(first_available, expected) << profile.name << " " << pair.first;
  EXPECT_EQ(second_available, expected) << profile.name << " " << pair.second;
  return first_available && second_available;
}

void expect_wrexec_def_use(const ScalarSccProfile &profile, const WrexecPair &pair) {
  auto decoder = Decoder::create(profile.arch);
  ASSERT_NE(decoder, nullptr) << profile.name;

  constexpr uint32_t kSourceSgpr = 0;
  constexpr uint32_t kDestSgpr = 4;
  const uint8_t width = pair.is_b64 ? 2 : 1;
  for (const auto mnemonic : {pair.first, pair.second}) {
    const auto words = encode_sop1(profile, mnemonic, kDestSgpr, kSourceSgpr);
    std::unique_ptr<Instruction> inst(decoder->decode(words.data()));
    ASSERT_NE(inst, nullptr) << profile.name << " " << mnemonic;
    ASSERT_EQ(std::string_view(inst->mnemonic()), mnemonic) << profile.name;

    const RegisterRef source{RegClass::SGPR, kSourceSgpr, width};
    const RegisterRef destination{RegClass::SGPR, kDestSgpr, width};
    ASSERT_EQ(inst->num_src_operands(), 1) << profile.name << " " << mnemonic;
    ASSERT_EQ(inst->num_dst_operands(), 1) << profile.name << " " << mnemonic;
    EXPECT_EQ(inst->src_operand(0)->to_register_ref(), source) << profile.name << " " << mnemonic;
    EXPECT_EQ(inst->dst_operand(0)->to_register_ref(), destination)
        << profile.name << " " << mnemonic;

    const InstDefUse def_use(*inst);
    EXPECT_TRUE(def_use.uses.contains(source)) << profile.name << " " << mnemonic;
    EXPECT_FALSE(def_use.uses.contains(destination)) << profile.name << " " << mnemonic;
    EXPECT_TRUE(def_use.defs.contains(destination)) << profile.name << " " << mnemonic;
    EXPECT_FALSE(def_use.defs.contains(source)) << profile.name << " " << mnemonic;
    EXPECT_EQ(def_use.uses.size(), static_cast<size_t>(width)) << profile.name << " " << mnemonic;
    EXPECT_EQ(def_use.defs.size(), static_cast<size_t>(width)) << profile.name << " " << mnemonic;
  }
}

void run_wrexec_scc_cases(const ScalarSccProfile &profile, const WrexecPair &pair) {
  ScalarSccFixture fixture(profile, "wrexec_scc");
  ASSERT_TRUE(fixture.ready()) << profile.name;

  class Case {
  public:
    std::string_view mnemonic;
    uint64_t old_exec;
    uint64_t src;
    uint64_t expected;
  };

  const uint64_t mask = pair.is_b64 ? ~uint64_t{0} : uint64_t{0xFFFFFFFFu};
  const std::array<Case, 4> cases{{
      {pair.first, 0x00FF00FF00FF00FFULL, 0x000F000F000F000FULL, 0x00F000F000F000F0ULL & mask},
      {pair.first, 0x00FF00FF00FF00FFULL, ~uint64_t{0}, 0u},
      {pair.second, 0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL & mask},
      {pair.second, ~uint64_t{0}, 0xF0F0F0F0F0F0F0F0ULL, 0u},
  }};

  constexpr uint32_t kSourceSgpr = 0;
  constexpr uint32_t kDestSgpr = 4;
  constexpr uint32_t kSourceHighSentinel = 0xA5A55A5Au;
  constexpr uint32_t kDestHighSentinel = 0x5A5AA5A5u;
  constexpr uint32_t kExecHighSentinel = 0xC3C33C3Cu;
  const uint32_t sb = fixture.sgpr_base();
  for (const auto &test_case : cases) {
    const auto words = encode_sop1(profile, test_case.mnemonic, kDestSgpr, kSourceSgpr);
    auto inst = fixture.decode(words, test_case.mnemonic);
    ASSERT_NE(inst, nullptr) << profile.name;

    fixture.cu->write_sgpr(sb + kSourceSgpr, static_cast<uint32_t>(test_case.src));
    fixture.cu->write_sgpr(sb + kSourceSgpr + 1, pair.is_b64
                                                     ? static_cast<uint32_t>(test_case.src >> 32)
                                                     : kSourceHighSentinel);
    fixture.cu->write_sgpr(sb + kDestSgpr, 0xBAD0D57u);
    fixture.cu->write_sgpr(sb + kDestSgpr + 1, kDestHighSentinel);
    const uint64_t raw_exec = pair.is_b64 ? test_case.old_exec
                                          : (static_cast<uint64_t>(kExecHighSentinel) << 32) |
                                                (test_case.old_exec & 0xFFFFFFFFULL);
    fixture.wf->set_exec_raw(raw_exec);
    const bool expected_scc = test_case.expected != 0;
    fixture.wf->write_scc(!expected_scc);
    fixture.cu->execute_instruction(inst.get(), *fixture.wf);

    uint64_t actual = fixture.cu->read_sgpr(sb + kDestSgpr);
    if (pair.is_b64) {
      actual |= static_cast<uint64_t>(fixture.cu->read_sgpr(sb + kDestSgpr + 1)) << 32;
      EXPECT_EQ(fixture.wf->exec_raw(), test_case.expected)
          << profile.name << " " << test_case.mnemonic;
    } else {
      EXPECT_EQ(fixture.cu->read_sgpr(sb + kSourceSgpr + 1), kSourceHighSentinel)
          << profile.name << " " << test_case.mnemonic;
      EXPECT_EQ(fixture.cu->read_sgpr(sb + kDestSgpr + 1), kDestHighSentinel)
          << profile.name << " " << test_case.mnemonic;
      EXPECT_EQ(static_cast<uint32_t>(fixture.wf->exec_raw() >> 32), kExecHighSentinel)
          << profile.name << " " << test_case.mnemonic;
    }
    EXPECT_EQ(actual, test_case.expected) << profile.name << " " << test_case.mnemonic;
    const uint64_t lane_mask = fixture.wf->wf_size() == 64 ? ~uint64_t{0} : uint64_t{0xFFFFFFFFu};
    EXPECT_EQ(fixture.wf->exec(), test_case.expected & lane_mask)
        << profile.name << " " << test_case.mnemonic;
    fixture.expect_scc_consumer(expected_scc, test_case.mnemonic);
  }
}

TEST(ScalarSccTest, WrexecWritesDestinationExecAndScc) {
  for (const auto &profile : kScalarSccProfiles) {
    for (const auto &candidate : kWrexecPairs) {
      const bool expected = profile_expects_wrexec_pair(profile, candidate);
      const bool available = expect_wrexec_availability(profile, candidate);
      if (expected && available)
        run_wrexec_scc_cases(profile, candidate);
    }
  }
}

TEST(ScalarSccTest, WrexecRegisterDefUseMatchesArchitectureMatrix) {
  for (const auto &profile : kScalarSccProfiles) {
    for (const auto &candidate : kWrexecPairs) {
      const bool expected = profile_expects_wrexec_pair(profile, candidate);
      const bool available = expect_wrexec_availability(profile, candidate);
      if (expected && available)
        expect_wrexec_def_use(profile, candidate);
    }
  }
}

enum class SaveexecOp {
  And,
  Or,
  Xor,
  ExecAndNotSrc,
  ExecOrNotSrc,
  SrcAndNotExec,
  SrcOrNotExec,
  Nand,
  Nor,
  Xnor,
};

class SaveexecVariant {
public:
  std::string_view mnemonic_stem;
  SaveexecOp operation;
};

constexpr std::array<SaveexecVariant, 14> kSaveexecVariants{{
    {"s_and_saveexec", SaveexecOp::And},
    {"s_or_saveexec", SaveexecOp::Or},
    {"s_xor_saveexec", SaveexecOp::Xor},
    {"s_andn1_saveexec", SaveexecOp::ExecAndNotSrc},
    {"s_orn1_saveexec", SaveexecOp::ExecOrNotSrc},
    {"s_andn2_saveexec", SaveexecOp::SrcAndNotExec},
    {"s_orn2_saveexec", SaveexecOp::SrcOrNotExec},
    {"s_and_not0_saveexec", SaveexecOp::ExecAndNotSrc},
    {"s_or_not0_saveexec", SaveexecOp::ExecOrNotSrc},
    {"s_and_not1_saveexec", SaveexecOp::SrcAndNotExec},
    {"s_or_not1_saveexec", SaveexecOp::SrcOrNotExec},
    {"s_nand_saveexec", SaveexecOp::Nand},
    {"s_nor_saveexec", SaveexecOp::Nor},
    {"s_xnor_saveexec", SaveexecOp::Xnor},
}};

uint64_t evaluate_saveexec(SaveexecOp operation, uint64_t old_exec, uint64_t src) {
  switch (operation) {
  case SaveexecOp::And:
    return old_exec & src;
  case SaveexecOp::Or:
    return old_exec | src;
  case SaveexecOp::Xor:
    return old_exec ^ src;
  case SaveexecOp::ExecAndNotSrc:
    return old_exec & ~src;
  case SaveexecOp::ExecOrNotSrc:
    return old_exec | ~src;
  case SaveexecOp::SrcAndNotExec:
    return src & ~old_exec;
  case SaveexecOp::SrcOrNotExec:
    return src | ~old_exec;
  case SaveexecOp::Nand:
    return ~(old_exec & src);
  case SaveexecOp::Nor:
    return ~(old_exec | src);
  case SaveexecOp::Xnor:
    return ~(old_exec ^ src);
  }
  return 0;
}

std::pair<uint64_t, uint64_t> zero_result_saveexec_inputs(SaveexecOp operation) {
  constexpr uint64_t kAll = ~uint64_t{0};
  switch (operation) {
  case SaveexecOp::And:
  case SaveexecOp::Or:
    return {0, 0};
  case SaveexecOp::Xor:
  case SaveexecOp::ExecAndNotSrc:
  case SaveexecOp::SrcAndNotExec:
  case SaveexecOp::Nand:
    return {kAll, kAll};
  case SaveexecOp::ExecOrNotSrc:
    return {0, kAll};
  case SaveexecOp::SrcOrNotExec:
  case SaveexecOp::Nor:
    return {kAll, 0};
  case SaveexecOp::Xnor:
    return {0, kAll};
  }
  return {0, 0};
}

void run_saveexec_case(ScalarSccFixture &fixture, const ScalarSccProfile &profile,
                       const SaveexecVariant &variant, bool is_b64, uint64_t old_exec, uint64_t src,
                       uint32_t sdst = 4) {
  constexpr uint32_t kHighSentinel = 0xA5A55A5Au;
  const uint64_t width_mask = is_b64 ? ~uint64_t{0} : uint64_t{0xFFFFFFFFu};
  old_exec &= width_mask;
  src &= width_mask;
  const uint64_t expected = evaluate_saveexec(variant.operation, old_exec, src) & width_mask;
  const std::string mnemonic = std::string(variant.mnemonic_stem) + (is_b64 ? "_b64" : "_b32");
  const auto words = encode_sop1(profile, mnemonic, sdst, /*ssrc0=*/0);
  auto inst = fixture.decode(words, mnemonic);
  ASSERT_NE(inst, nullptr) << profile.name;

  const uint32_t sb = fixture.sgpr_base();
  fixture.cu->write_sgpr(sb, static_cast<uint32_t>(src));
  if (is_b64)
    fixture.cu->write_sgpr(sb + 1, static_cast<uint32_t>(src >> 32));
  else
    fixture.cu->write_sgpr(sb + sdst + 1, kHighSentinel);

  const uint64_t initial_raw_exec =
      is_b64 ? old_exec : (static_cast<uint64_t>(kHighSentinel) << 32) | old_exec;
  fixture.wf->set_exec_raw(initial_raw_exec);
  fixture.wf->write_scc(expected == 0);
  fixture.cu->execute_instruction(inst.get(), *fixture.wf);

  if (is_b64) {
    const uint64_t saved = fixture.cu->read_sgpr(sb + sdst) |
                           (static_cast<uint64_t>(fixture.cu->read_sgpr(sb + sdst + 1)) << 32);
    EXPECT_EQ(saved, old_exec) << profile.name << " " << mnemonic;
    EXPECT_EQ(fixture.wf->exec_raw(), expected) << profile.name << " " << mnemonic;
  } else {
    EXPECT_EQ(fixture.cu->read_sgpr(sb + sdst), static_cast<uint32_t>(old_exec))
        << profile.name << " " << mnemonic;
    EXPECT_EQ(fixture.cu->read_sgpr(sb + sdst + 1), kHighSentinel)
        << profile.name << " " << mnemonic;
    EXPECT_EQ(fixture.wf->exec_raw(), (static_cast<uint64_t>(kHighSentinel) << 32) | expected)
        << profile.name << " " << mnemonic;
  }

  const uint64_t lane_mask = fixture.wf->wf_size() == 64 ? ~uint64_t{0} : uint64_t{0xFFFFFFFFu};
  EXPECT_EQ(fixture.wf->exec(), expected & lane_mask) << profile.name << " " << mnemonic;
  fixture.expect_scc_consumer(expected != 0, mnemonic);
}

TEST(ScalarSccTest, SaveexecSupportsAliasedSourceAndDestination) {
  constexpr uint64_t kOldExec = 0x0F0FF0F00FF00FF0ULL;
  constexpr uint64_t kSource = 0x3333CCCC5555AAAAULL;
  const auto &variant = kSaveexecVariants.front();
  size_t b32_coverage = 0;
  size_t b64_coverage = 0;

  for (const auto &profile : kScalarSccProfiles) {
    ScalarSccFixture fixture(profile, "saveexec_alias");
    ASSERT_TRUE(fixture.ready()) << profile.name;

    for (const bool is_b64 : {false, true}) {
      const std::string mnemonic = std::string(variant.mnemonic_stem) + (is_b64 ? "_b64" : "_b32");
      if (!profile.find_encoding(mnemonic))
        continue;

      run_saveexec_case(fixture, profile, variant, is_b64, kOldExec, kSource, /*sdst=*/0);
      ++(is_b64 ? b64_coverage : b32_coverage);
    }
  }

  EXPECT_GT(b32_coverage, 0u);
  EXPECT_GT(b64_coverage, 0u);
}

TEST(ScalarSccTest, SaveexecBooleanMatrixPreservesExecAndWritesScc) {
  constexpr uint64_t kDistinguishingOldExec = 0x0F0FF0F00FF00FF0ULL;
  constexpr uint64_t kDistinguishingSource = 0x3333CCCC5555AAAAULL;
  std::array<size_t, kSaveexecVariants.size()> b32_coverage{};
  std::array<size_t, kSaveexecVariants.size()> b64_coverage{};

  for (const auto &profile : kScalarSccProfiles) {
    ScalarSccFixture fixture(profile, "saveexec_boolean_scc");
    ASSERT_TRUE(fixture.ready()) << profile.name;

    for (size_t index = 0; index < kSaveexecVariants.size(); ++index) {
      const auto &variant = kSaveexecVariants[index];
      for (const bool is_b64 : {false, true}) {
        const std::string mnemonic =
            std::string(variant.mnemonic_stem) + (is_b64 ? "_b64" : "_b32");
        if (!profile.find_encoding(mnemonic))
          continue;

        const uint64_t width_mask = is_b64 ? ~uint64_t{0} : uint64_t{0xFFFFFFFFu};
        ASSERT_NE(
            evaluate_saveexec(variant.operation, kDistinguishingOldExec, kDistinguishingSource) &
                width_mask,
            0u)
            << mnemonic;
        run_saveexec_case(fixture, profile, variant, is_b64, kDistinguishingOldExec,
                          kDistinguishingSource);

        const auto [zero_old_exec, zero_source] = zero_result_saveexec_inputs(variant.operation);
        ASSERT_EQ(evaluate_saveexec(variant.operation, zero_old_exec, zero_source) & width_mask, 0u)
            << mnemonic;
        run_saveexec_case(fixture, profile, variant, is_b64, zero_old_exec, zero_source);

        ++(is_b64 ? b64_coverage[index] : b32_coverage[index]);
      }
    }
  }

  for (size_t index = 0; index < kSaveexecVariants.size(); ++index) {
    EXPECT_GT(b32_coverage[index], 0u) << kSaveexecVariants[index].mnemonic_stem;
    EXPECT_GT(b64_coverage[index], 0u) << kSaveexecVariants[index].mnemonic_stem;
  }
}

std::string_view addk_mnemonic(const ScalarSccProfile &profile) {
  return require_first_supported(profile, {"s_addk_i32", "s_addk_co_i32"}, "ADDK");
}

void run_addk_scc_cases(const ScalarSccProfile &profile) {
  ScalarSccFixture fixture(profile, "addk_scc");
  ASSERT_TRUE(fixture.ready()) << profile.name;

  class Case {
  public:
    uint32_t initial;
    uint16_t immediate;
    uint32_t expected;
    bool expected_scc;
  };
  constexpr std::array<Case, 4> cases{{
      {0x7FFFFFFFu, 1u, 0x80000000u, true},
      {0xFFFFFFFFu, 1u, 0u, false},
      {0x80000000u, 0xFFFFu, 0x7FFFFFFFu, true},
      {5u, 0xFFFDu, 2u, false},
  }};

  const auto mnemonic = addk_mnemonic(profile);
  const uint32_t sb = fixture.sgpr_base();
  for (const auto &test_case : cases) {
    const auto words = encode_sopk(profile, mnemonic, /*sdst=*/4, test_case.immediate);
    auto inst = fixture.decode(words, mnemonic);
    ASSERT_NE(inst, nullptr) << profile.name;

    fixture.cu->write_sgpr(sb + 4, test_case.initial);
    fixture.wf->write_scc(!test_case.expected_scc);
    fixture.cu->execute_instruction(inst.get(), *fixture.wf);
    EXPECT_EQ(fixture.cu->read_sgpr(sb + 4), test_case.expected) << profile.name;
    fixture.expect_scc_consumer(test_case.expected_scc, mnemonic);
  }
}

TEST(ScalarSccTest, AddkUsesSignedOverflowAndFeedsSccConsumer) {
  for (const auto &profile : kScalarSccProfiles)
    run_addk_scc_cases(profile);
}

void run_mulk_cases(const ScalarSccProfile &profile) {
  ScalarSccFixture fixture(profile, "mulk");
  ASSERT_TRUE(fixture.ready()) << profile.name;

  class Case {
  public:
    uint32_t initial;
    uint16_t immediate;
    uint32_t expected;
  };
  constexpr std::array<Case, 4> cases{{
      {7u, 3u, 21u},
      {7u, 0xFFFDu, 0xFFFFFFEBu},
      {0x12345678u, 0u, 0u},
      {0x80000001u, 2u, 2u},
  }};

  const uint32_t sb = fixture.sgpr_base();
  for (const auto &test_case : cases) {
    const auto words = encode_sopk(profile, "s_mulk_i32", /*sdst=*/4, test_case.immediate);
    auto inst = fixture.decode(words, "s_mulk_i32");
    ASSERT_NE(inst, nullptr) << profile.name;

    for (const bool initial_scc : {false, true}) {
      fixture.cu->write_sgpr(sb + 4, test_case.initial);
      fixture.wf->write_scc(initial_scc);
      fixture.cu->execute_instruction(inst.get(), *fixture.wf);
      EXPECT_EQ(fixture.cu->read_sgpr(sb + 4), test_case.expected) << profile.name;
      fixture.expect_scc_consumer(initial_scc, "s_mulk_i32");
    }
  }
}

TEST(ScalarSccTest, MulkUsesOldDestinationAndSignedImmediate) {
  for (const auto &profile : kScalarSccProfiles)
    run_mulk_cases(profile);
}

TEST(ScalarSccTest, AddkAndMulkRegisterDestinationRead) {
  for (const auto &profile : kScalarSccProfiles) {
    auto decoder = Decoder::create(profile.arch);
    ASSERT_NE(decoder, nullptr) << profile.name;
    const std::array mnemonics{addk_mnemonic(profile), std::string_view{"s_mulk_i32"}};
    for (const auto mnemonic : mnemonics) {
      const auto words = encode_sopk(profile, mnemonic, /*sdst=*/4, /*simm16=*/1);
      std::unique_ptr<Instruction> inst(decoder->decode(words.data()));
      ASSERT_NE(inst, nullptr) << profile.name << " " << mnemonic;
      ASSERT_EQ(std::string_view(inst->mnemonic()), mnemonic) << profile.name;

      ASSERT_EQ(inst->num_src_operands(), 2) << profile.name << " " << mnemonic;
      ASSERT_EQ(inst->num_dst_operands(), 1) << profile.name << " " << mnemonic;
      EXPECT_EQ(inst->src_operand(0)->to_register_ref(), (RegisterRef{RegClass::SGPR, 4, 1}))
          << profile.name << " " << mnemonic;
      EXPECT_EQ(inst->src_operand(0), inst->dst_operand(0)) << profile.name << " " << mnemonic;
      EXPECT_EQ(inst->src_operand(0)->name(), "s4") << profile.name << " " << mnemonic;
      EXPECT_FALSE(inst->src_operand(1)->to_register_ref()) << profile.name << " " << mnemonic;
      EXPECT_EQ(inst->src_operand(1)->encoding_value(), 1) << profile.name << " " << mnemonic;
      EXPECT_EQ(inst->src_operand(1)->name(), "1") << profile.name << " " << mnemonic;
      EXPECT_EQ(inst->dst_operand(0)->to_register_ref(), (RegisterRef{RegClass::SGPR, 4, 1}))
          << profile.name << " " << mnemonic;

      const InstDefUse def_use(*inst);
      EXPECT_TRUE(def_use.uses.contains(RegisterRef{RegClass::SGPR, 4, 1}))
          << profile.name << " " << mnemonic;
      EXPECT_TRUE(def_use.defs.contains(RegisterRef{RegClass::SGPR, 4, 1}))
          << profile.name << " " << mnemonic;
      EXPECT_EQ(def_use.uses.size(), 1u) << profile.name << " " << mnemonic;
      EXPECT_EQ(def_use.defs.size(), 1u) << profile.name << " " << mnemonic;
    }
  }
}

void run_sop2_scc_cases(const ScalarSccProfile &profile) {
  ScalarSccFixture fixture(profile, "sop2_scc");
  ASSERT_TRUE(fixture.ready()) << profile.name;

  const auto add_u32 =
      require_first_supported(profile, {"s_add_u32", "s_add_co_u32"}, "unsigned ADD");
  const auto sub_u32 =
      require_first_supported(profile, {"s_sub_u32", "s_sub_co_u32"}, "unsigned SUB");
  const auto add_i32 =
      require_first_supported(profile, {"s_add_i32", "s_add_co_i32"}, "signed ADD");
  const auto sub_i32 =
      require_first_supported(profile, {"s_sub_i32", "s_sub_co_i32"}, "signed SUB");

  class Case {
  public:
    std::string_view mnemonic;
    uint32_t lhs;
    uint32_t rhs;
    uint32_t expected_result;
    bool expected_scc;
  };
  const std::array<Case, 18> cases{{
      {add_u32, 0xFFFFFFFFu, 1u, 0u, true},
      {add_u32, 7u, 5u, 12u, false},
      {sub_u32, 0u, 1u, 0xFFFFFFFFu, true},
      {sub_u32, 7u, 5u, 2u, false},
      {add_i32, 0x7FFFFFFFu, 1u, 0x80000000u, true},
      {add_i32, 0xFFFFFFFFu, 1u, 0u, false},
      {sub_i32, 0x80000000u, 1u, 0x7FFFFFFFu, true},
      {sub_i32, 5u, 3u, 2u, false},
      {"s_and_b32", 0xF3u, 0x0Fu, 3u, true},
      {"s_and_b32", 0xF0u, 0x0Fu, 0u, false},
      {"s_min_i32", 0xFFFFFFFCu, 3u, 0xFFFFFFFCu, true},
      {"s_min_i32", 4u, 0xFFFFFFFEu, 0xFFFFFFFEu, false},
      {"s_max_i32", 7u, 3u, 7u, true},
      {"s_max_i32", 0xFFFFFFFCu, 3u, 3u, false},
      {"s_max_i32", 0x80000000u, 0x80000000u, 0x80000000u, false},
      {"s_max_u32", 7u, 3u, 7u, true},
      {"s_max_u32", 3u, 7u, 7u, false},
      {"s_max_u32", 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, false},
  }};

  const uint32_t sb = fixture.sgpr_base();
  for (const auto &test_case : cases) {
    const auto words =
        encode_sop2(profile, test_case.mnemonic, /*sdst=*/2, /*ssrc0=*/0, /*ssrc1=*/1);
    auto inst = fixture.decode(words, test_case.mnemonic);
    ASSERT_NE(inst, nullptr) << profile.name;

    fixture.cu->write_sgpr(sb, test_case.lhs);
    fixture.cu->write_sgpr(sb + 1, test_case.rhs);
    fixture.wf->write_scc(!test_case.expected_scc);
    fixture.cu->execute_instruction(inst.get(), *fixture.wf);
    EXPECT_EQ(fixture.cu->read_sgpr(sb + 2), test_case.expected_result)
        << profile.name << " " << test_case.mnemonic;
    fixture.expect_scc_consumer(test_case.expected_scc, test_case.mnemonic);
  }
}

TEST(ScalarSccTest, Sop2SccModesFeedSccConsumer) {
  for (const auto &profile : kScalarSccProfiles)
    run_sop2_scc_cases(profile);
}

void run_sop2_carry_input_cases(const ScalarSccProfile &profile) {
  ScalarSccFixture fixture(profile, "sop2_carry_input_scc");
  ASSERT_TRUE(fixture.ready()) << profile.name;

  const auto addc =
      require_first_supported(profile, {"s_addc_u32", "s_addc_co_u32", "s_add_co_ci_u32"}, "ADDC");
  const auto subb =
      require_first_supported(profile, {"s_subb_u32", "s_subb_co_u32", "s_sub_co_ci_u32"}, "SUBB");
  ASSERT_FALSE(addc.empty()) << profile.name;
  ASSERT_FALSE(subb.empty()) << profile.name;

  class Case {
  public:
    std::string_view mnemonic;
    uint32_t lhs;
    uint32_t rhs;
    bool input_scc;
    uint32_t expected_result;
    bool expected_scc;
  };
  const std::array<Case, 8> cases{{
      {addc, 1u, 2u, false, 3u, false},
      {addc, 1u, 2u, true, 4u, false},
      {addc, 0xFFFFFFFFu, 1u, false, 0u, true},
      {addc, 0xFFFFFFFFu, 0u, true, 0u, true},
      {subb, 5u, 3u, false, 2u, false},
      {subb, 5u, 3u, true, 1u, false},
      {subb, 0u, 1u, false, 0xFFFFFFFFu, true},
      {subb, 0u, 0u, true, 0xFFFFFFFFu, true},
  }};

  const uint32_t sb = fixture.sgpr_base();
  for (const auto &test_case : cases) {
    const auto words =
        encode_sop2(profile, test_case.mnemonic, /*sdst=*/2, /*ssrc0=*/0, /*ssrc1=*/1);
    auto inst = fixture.decode(words, test_case.mnemonic);
    ASSERT_NE(inst, nullptr) << profile.name;

    fixture.cu->write_sgpr(sb, test_case.lhs);
    fixture.cu->write_sgpr(sb + 1, test_case.rhs);
    fixture.wf->write_scc(test_case.input_scc);
    fixture.cu->execute_instruction(inst.get(), *fixture.wf);
    EXPECT_EQ(fixture.cu->read_sgpr(sb + 2), test_case.expected_result)
        << profile.name << " " << test_case.mnemonic;
    fixture.expect_scc_consumer(test_case.expected_scc, test_case.mnemonic);
  }
}

TEST(ScalarSccTest, Sop2CarryInputReadsAndWritesScc) {
  for (const auto &profile : kScalarSccProfiles)
    run_sop2_carry_input_cases(profile);
}

void run_sopc_sopk_compare_scc_cases(const ScalarSccProfile &profile) {
  ScalarSccFixture fixture(profile, "scalar_compare_scc");
  ASSERT_TRUE(fixture.ready()) << profile.name;

  class SopcCase {
  public:
    std::string_view mnemonic;
    uint32_t lhs;
    uint32_t rhs;
    bool expected_scc;
  };
  constexpr std::array<SopcCase, 4> sopc_cases{{
      {"s_cmp_eq_i32", 5u, 5u, true},
      {"s_cmp_eq_i32", 5u, 6u, false},
      {"s_bitcmp0_b32", 0u, 31u, true},
      {"s_bitcmp0_b32", 0x80000000u, 31u, false},
  }};

  const uint32_t sb = fixture.sgpr_base();
  for (const auto &test_case : sopc_cases) {
    const auto words = encode_sopc(profile, test_case.mnemonic, /*ssrc0=*/0, /*ssrc1=*/1);
    auto inst = fixture.decode(words, test_case.mnemonic);
    ASSERT_NE(inst, nullptr) << profile.name;

    fixture.cu->write_sgpr(sb, test_case.lhs);
    fixture.cu->write_sgpr(sb + 1, test_case.rhs);
    fixture.wf->write_scc(!test_case.expected_scc);
    fixture.cu->execute_instruction(inst.get(), *fixture.wf);
    fixture.expect_scc_consumer(test_case.expected_scc, test_case.mnemonic);
  }

  if (!profile.find_encoding("s_cmpk_eq_i32"))
    return;

  class SopkCase {
  public:
    uint32_t src;
    uint16_t immediate;
    bool expected_scc;
  };
  constexpr std::array<SopkCase, 2> sopk_cases{{
      {0xFFFFFFFFu, 0xFFFFu, true},
      {0x0000FFFFu, 0xFFFFu, false},
  }};
  for (const auto &test_case : sopk_cases) {
    const auto words = encode_sopk(profile, "s_cmpk_eq_i32", /*sdst=*/4, test_case.immediate);
    auto inst = fixture.decode(words, "s_cmpk_eq_i32");
    ASSERT_NE(inst, nullptr) << profile.name;

    fixture.cu->write_sgpr(sb + 4, test_case.src);
    fixture.wf->write_scc(!test_case.expected_scc);
    fixture.cu->execute_instruction(inst.get(), *fixture.wf);
    fixture.expect_scc_consumer(test_case.expected_scc, "s_cmpk_eq_i32");
  }
}

TEST(ScalarSccTest, SopcAndSopkCompareFeedSccConsumer) {
  for (const auto &profile : kScalarSccProfiles)
    run_sopc_sopk_compare_scc_cases(profile);
}

class ScalarUnarySccCase {
public:
  std::string_view mnemonic;
  uint64_t src;
  uint64_t expected;
  bool result_is_64_bit = false;
};

template <size_t N>
void run_scalar_unary_preserves_scc(const ScalarSccProfile &profile,
                                    const std::array<ScalarUnarySccCase, N> &cases,
                                    std::string_view test_name) {
  bool has_supported_case = false;
  for (const auto &test_case : cases)
    has_supported_case |= profile.find_encoding(test_case.mnemonic).has_value();
  if (!has_supported_case)
    return;

  ScalarSccFixture fixture(profile, test_name);
  ASSERT_TRUE(fixture.ready()) << profile.name;
  const uint32_t sb = fixture.sgpr_base();
  constexpr uint32_t kHighDestinationSentinel = 0xA5A55A5Au;

  for (const auto &test_case : cases) {
    if (!profile.find_encoding(test_case.mnemonic))
      continue;
    const auto words = encode_sop1(profile, test_case.mnemonic, /*sdst=*/2, /*ssrc0=*/0);
    for (bool initial_scc : std::array<bool, 2>{false, true}) {
      auto inst = fixture.decode(words, test_case.mnemonic);
      ASSERT_NE(inst, nullptr) << profile.name;

      fixture.cu->write_sgpr(sb, static_cast<uint32_t>(test_case.src));
      fixture.cu->write_sgpr(sb + 1, static_cast<uint32_t>(test_case.src >> 32));
      fixture.cu->write_sgpr(sb + 2, 0u);
      fixture.cu->write_sgpr(sb + 3, kHighDestinationSentinel);
      fixture.wf->write_scc(initial_scc);
      fixture.cu->execute_instruction(inst.get(), *fixture.wf);

      uint64_t actual = fixture.cu->read_sgpr(sb + 2);
      if (test_case.result_is_64_bit) {
        actual |= static_cast<uint64_t>(fixture.cu->read_sgpr(sb + 3)) << 32;
      } else {
        EXPECT_EQ(fixture.cu->read_sgpr(sb + 3), kHighDestinationSentinel)
            << profile.name << " " << test_case.mnemonic;
      }
      EXPECT_EQ(actual, test_case.expected) << profile.name << " " << test_case.mnemonic;
      EXPECT_EQ(fixture.wf->read_scc(), initial_scc)
          << profile.name << " " << test_case.mnemonic << " initial_scc=" << initial_scc;
    }
  }
}

TEST(ScalarSccTest, ScalarCvtPreservesScc) {
  const uint32_t one_f32 = std::bit_cast<uint32_t>(1.0f);
  const uint32_t one_f16 = util::f32_to_f16(1.0f);
  const uint32_t qnan_f32 = 0x7FC00000u;
  const uint32_t i32_overflow_f32 = std::bit_cast<uint32_t>(2147483648.0f);
  const uint32_t below_i32_min_f32 = std::bit_cast<uint32_t>(-2147483904.0f);
  const uint32_t u32_overflow_f32 = std::bit_cast<uint32_t>(4294967296.0f);

  const std::array<ScalarUnarySccCase, 13> cases{{
      {"s_cvt_f32_i32", 1u, one_f32},
      {"s_cvt_f32_u32", 1u, one_f32},
      {"s_cvt_i32_f32", one_f32, 1u},
      {"s_cvt_i32_f32", qnan_f32, 0u},
      {"s_cvt_i32_f32", i32_overflow_f32, 0x7FFFFFFFu},
      {"s_cvt_i32_f32", below_i32_min_f32, 0x80000000u},
      {"s_cvt_u32_f32", one_f32, 1u},
      {"s_cvt_u32_f32", qnan_f32, 0u},
      {"s_cvt_u32_f32", std::bit_cast<uint32_t>(-1.0f), 0u},
      {"s_cvt_u32_f32", u32_overflow_f32, 0xFFFFFFFFu},
      {"s_cvt_f16_f32", one_f32, one_f16},
      {"s_cvt_f32_f16", one_f16, one_f32},
      {"s_cvt_hi_f32_f16", one_f16 << 16, one_f32},
  }};

  for (const auto &profile : kScalarSccProfiles)
    run_scalar_unary_preserves_scc(profile, cases, "scalar_cvt_scc");
}

TEST(ScalarSccTest, ScalarScanPreservesScc) {
  constexpr std::array<ScalarUnarySccCase, 18> cases{{
      {"s_ff0_i32_b32", 0x0000000FULL, 4u},
      {"s_ff0_i32_b64", 0x00000000FFFFFFFFULL, 32u},
      {"s_ff1_i32_b32", 0x00000100ULL, 8u},
      {"s_ff1_i32_b64", 0x0000000100000000ULL, 32u},
      {"s_flbit_i32_b32", 0x00F00000ULL, 8u},
      {"s_flbit_i32_b64", 0x0000000100000000ULL, 31u},
      {"s_flbit_i32", 0x40000000ULL, 1u},
      {"s_flbit_i32_i64", 0x0000000100000000ULL, 31u},
      {"s_ctz_i32_b32", 0x00000100ULL, 8u},
      {"s_ctz_i32_b64", 0x0000000100000000ULL, 32u},
      {"s_clz_i32_u32", 0x00F00000ULL, 8u},
      {"s_clz_i32_u64", 0x0000000100000000ULL, 31u},
      {"s_cls_i32", 0x40000000ULL, 1u},
      {"s_cls_i32", 0u, 0xFFFFFFFFu},
      {"s_cls_i32", 0xFFFFFFFFULL, 0xFFFFFFFFu},
      {"s_cls_i32_i64", 0x0000010080000000ULL, 23u},
      {"s_cls_i32_i64", 0u, 0xFFFFFFFFu},
      {"s_cls_i32_i64", 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFu},
  }};

  for (const auto &profile : kScalarSccProfiles)
    run_scalar_unary_preserves_scc(profile, cases, "scalar_scan_scc");
}

TEST(ScalarSccTest, ScalarMiscUnaryPreservesScc) {
  constexpr std::array<ScalarUnarySccCase, 10> cases{{
      {"s_brev_b32", 1u, 0x80000000u},
      {"s_brev_b64", 1u, 0x8000000000000000ULL, true},
      {"s_ceil_f32", 0x3FC00000u, 0x40000000u},
      {"s_floor_f32", 0x3FC00000u, 0x3F800000u},
      {"s_trunc_f32", 0x3FC00000u, 0x3F800000u},
      {"s_rndne_f32", 0x3FC00000u, 0x40000000u},
      {"s_ceil_f16", 0x3E00u, 0x4000u},
      {"s_floor_f16", 0x3E00u, 0x3C00u},
      {"s_trunc_f16", 0x3E00u, 0x3C00u},
      {"s_rndne_f16", 0x3E00u, 0x4000u},
  }};

  for (const auto &profile : kScalarSccProfiles)
    run_scalar_unary_preserves_scc(profile, cases, "scalar_misc_unary_scc");
}

void run_scalar_scan_carry_chain(const ScalarSccProfile &profile) {
  const auto scan_mnemonic = first_supported(profile, {"s_ff1_i32_b32", "s_ctz_i32_b32"});
  const auto add_mnemonic = first_supported(profile, {"s_add_u32", "s_add_co_u32"});
  const auto addc_mnemonic =
      first_supported(profile, {"s_addc_u32", "s_addc_co_u32", "s_add_co_ci_u32"});
  ASSERT_TRUE(scan_mnemonic.has_value()) << profile.name;
  ASSERT_TRUE(add_mnemonic.has_value()) << profile.name;
  ASSERT_TRUE(addc_mnemonic.has_value()) << profile.name;

  ScalarSccFixture fixture(profile, "scalar_scan_carry");
  ASSERT_TRUE(fixture.ready()) << profile.name;
  const uint32_t sb = fixture.sgpr_base();
  const auto add_words = encode_sop2(profile, *add_mnemonic, /*sdst=*/0, /*ssrc0=*/4, /*ssrc1=*/0);
  const auto scan_words = encode_sop1(profile, *scan_mnemonic, /*sdst=*/10, /*ssrc0=*/12);
  const auto addc_words =
      encode_sop2(profile, *addc_mnemonic, /*sdst=*/1, /*ssrc0=*/5, /*ssrc1=*/1);

  auto add = fixture.decode(add_words, *add_mnemonic);
  auto scan = fixture.decode(scan_words, *scan_mnemonic);
  auto addc = fixture.decode(addc_words, *addc_mnemonic);
  ASSERT_NE(add, nullptr);
  ASSERT_NE(scan, nullptr);
  ASSERT_NE(addc, nullptr);

  fixture.cu->write_sgpr(sb, 1u);
  fixture.cu->write_sgpr(sb + 1, 0u);
  fixture.cu->write_sgpr(sb + 4, 0xFFFFFFFFu);
  fixture.cu->write_sgpr(sb + 5, 0x54u);
  fixture.cu->write_sgpr(sb + 10, 0u);
  fixture.cu->write_sgpr(sb + 12, 1u);
  fixture.wf->write_scc(false);

  fixture.cu->execute_instruction(add.get(), *fixture.wf);
  ASSERT_EQ(fixture.cu->read_sgpr(sb), 0u);
  ASSERT_TRUE(fixture.wf->read_scc());

  fixture.cu->execute_instruction(scan.get(), *fixture.wf);
  ASSERT_EQ(fixture.cu->read_sgpr(sb + 10), 0u);
  ASSERT_TRUE(fixture.wf->read_scc());

  fixture.cu->execute_instruction(addc.get(), *fixture.wf);
  EXPECT_EQ(fixture.cu->read_sgpr(sb + 1), 0x55u);
  EXPECT_FALSE(fixture.wf->read_scc());
}

TEST(ScalarSccTest, ScalarScanDoesNotBreakAddcCarryChain) {
  for (const auto &profile : kScalarSccProfiles)
    run_scalar_scan_carry_chain(profile);
}

} // namespace
