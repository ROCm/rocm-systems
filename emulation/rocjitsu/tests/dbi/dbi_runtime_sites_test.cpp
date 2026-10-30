// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Site selection for the DBI runtime hook, on synthetic code objects: the
// decisions that leave an object unpatched, and which instruction each block's
// site lands on.

#include "rocjitsu/hooks/dbi_runtime_sites.h"

#include "dbi_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "scalar_builders.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::hooks {
namespace {

using test::kMovV2Zero;
using test::kMovV3V2;

constexpr const char *kProbeSymbol = "rj_test_probe";

struct SitesTarget {
  const char *name;
  uint32_t e_flags;
  rj_code_arch_t arch;
};

const SitesTarget kTargets[] = {
    {"Gfx942", EF_AMDGPU_MACH_AMDGCN_GFX942, ROCJITSU_CODE_ARCH_CDNA3},
    {"Gfx950", EF_AMDGPU_MACH_AMDGCN_GFX950, ROCJITSU_CODE_ARCH_CDNA4},
    {"Gfx1200", EF_AMDGPU_MACH_AMDGCN_GFX1200, ROCJITSU_CODE_ARCH_RDNA4},
};
const SitesTarget &kGfx950 = kTargets[1];
const SitesTarget &kGfx1200 = kTargets[2];

// Three blocks: 0x00 [v_mov, s_cbranch_scc0 -> 0x0c], 0x08 [s_endpgm],
// 0x0c [s_getpc_b64, v_mov, v_mov, s_endpgm]. The middle block has nothing to
// anchor at; the last cannot anchor at its first instruction and has two
// candidates after it.
std::vector<uint32_t> three_block_text(const SitesTarget &target) {
  return {kMovV2Zero,
          test::build_s_cbranch_scc(/*scc1=*/false, /*offset_dwords=*/1, target.arch),
          build_s_endpgm(target.arch),
          build_s_getpc_b64(/*s[0:1]=*/0, target.arch),
          kMovV3V2,
          kMovV2Zero,
          build_s_endpgm(target.arch)};
}

std::vector<uint8_t> kernel(const SitesTarget &target, const std::vector<uint32_t> &text,
                            uint32_t granulated_sgpr_count = 3, uint64_t entry_text_offset = 0,
                            bool wave32 = false) {
  return test::make_amdgpu_kernel_elf(
      text, /*private_bytes=*/0, granulated_sgpr_count, target.e_flags,
      /*granulated_vgpr_count=*/0, /*accum_offset=*/0, /*unterminated_kd_name=*/false,
      /*wrap_section_header_table=*/false, /*wrap_symtab_range=*/false,
      /*kd_crosses_section=*/false, wave32, /*kernarg_segment_ptr=*/false,
      /*kernarg_size=*/0, entry_text_offset);
}

std::vector<uint8_t> probe(const SitesTarget &target,
                           const std::vector<uint32_t> &leading_words = {}) {
  return test::make_amdgpu_probe_elf(kProbeSymbol,
                                     {build_s_setpc_b64(/*s[30:31]=*/30, target.arch)},
                                     target.e_flags, leading_words);
}

// A function stub, then the zero words a linker pads the next function's
// 256-byte alignment with, which do not decode on RDNA4.
std::vector<uint32_t> stub_and_zero_padding(rj_code_arch_t arch) {
  std::vector<uint32_t> words(0x100 / sizeof(uint32_t), 0u);
  words.front() = build_s_setpc_b64(/*s[30:31]=*/30, arch);
  return words;
}

DbiInstrumentation instrument(const std::vector<uint8_t> &guest,
                              const std::vector<uint8_t> &probe_elf) {
  const AmdGpuCodeObject probe_object(probe_elf.data(), probe_elf.size());
  EXPECT_TRUE(probe_object.is_valid());
  return instrument_each_block(guest, DbiProbe{&probe_object, kProbeSymbol});
}

std::vector<uint32_t> text_words(const AmdGpuCodeObject &object) {
  const Section *text = object.text_sections().front();
  std::vector<uint32_t> words(text->size() / sizeof(uint32_t));
  std::memcpy(words.data(), text->data(), words.size() * sizeof(uint32_t));
  return words;
}

// The Instrumentor refused the object as @p decision, with an error naming
// @p reason.
void expect_refused(const DbiInstrumentation &result, DbiLoadDecision decision,
                    std::string_view reason) {
  EXPECT_EQ(result.decision, decision);
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find(reason), std::string::npos) << result.errors.front();
}

class DbiRuntimeSitesTargetTest : public ::testing::TestWithParam<SitesTarget> {};

TEST_P(DbiRuntimeSitesTargetTest, CallsTheProbeAtTheFirstAnchorOfEachBlock) {
  const SitesTarget &target = GetParam();
  const std::vector<uint32_t> original = three_block_text(target);
  const DbiInstrumentation result = instrument(kernel(target, original), probe(target));
  ASSERT_EQ(result.decision, DbiLoadDecision::Instrumented)
      << (result.errors.empty() ? std::string{} : result.errors.front());
  EXPECT_EQ(result.sites, 2u);
  EXPECT_EQ(result.blocks_without_site, 1u);

  const AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> words = text_words(patched);
  ASSERT_GT(words.size(), original.size());
  // A site's anchor becomes an s_branch into the cave; every other word stays.
  constexpr uint32_t kSoppOpcodeMask = 0xFFFF0000u;
  const uint32_t branch_opcode = build_s_branch(0, target.arch) & kSoppOpcodeMask;
  for (size_t index = 0; index < original.size(); ++index) {
    if (index == 0 || index == 4)
      EXPECT_EQ(words[index] & kSoppOpcodeMask, branch_opcode) << "word " << index;
    else
      EXPECT_EQ(words[index], original[index]) << "word " << index;
  }
  // The cave holds the copied probe body, so the sites call the probe rather
  // than taking the inline nop.
  const std::vector<uint32_t> cave(words.begin() + static_cast<ptrdiff_t>(original.size()),
                                   words.end());
  EXPECT_NE(std::find(cave.begin(), cave.end(), build_s_setpc_b64(30, target.arch)), cave.end());
}

INSTANTIATE_TEST_SUITE_P(AllDbiTargets, DbiRuntimeSitesTargetTest, ::testing::ValuesIn(kTargets),
                         [](const ::testing::TestParamInfo<SitesTarget> &info) {
                           return std::string(info.param.name);
                         });

TEST(DbiRuntimeSites, RejectsBytesThatAreNotACodeObject) {
  const std::vector<uint8_t> garbage(64, 0x5a);
  const DbiInstrumentation result = instrument(garbage, probe(kGfx950));
  EXPECT_EQ(result.decision, DbiLoadDecision::NotAmdGpuObject);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(DbiRuntimeSites, LeavesUnvalidatedArchUnpatched) {
  const SitesTarget gfx90a{"Gfx90a", EF_AMDGPU_MACH_AMDGCN_GFX90A, ROCJITSU_CODE_ARCH_CDNA2};
  const DbiInstrumentation result =
      instrument(kernel(gfx90a, {kMovV2Zero, build_s_endpgm(gfx90a.arch)}), probe(gfx90a));
  EXPECT_EQ(result.decision, DbiLoadDecision::UnsupportedArch);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(DbiRuntimeSites, LeavesObjectUnpatchedWhenProbeTargetDiffers) {
  // gfx1200 and gfx1201 are both RDNA4; the Instrumentor still refuses the pair.
  const SitesTarget gfx1201{"Gfx1201", EF_AMDGPU_MACH_AMDGCN_GFX1201, ROCJITSU_CODE_ARCH_RDNA4};
  for (const DbiInstrumentation &result :
       {instrument(kernel(gfx1201, three_block_text(gfx1201)), probe(kGfx1200)),
        instrument(kernel(kGfx950, three_block_text(kGfx950)), probe(kGfx1200))}) {
    expect_refused(result, DbiLoadDecision::PatchFailed, "concrete target");
  }
}

TEST(DbiRuntimeSites, LeavesObjectWithoutKernelUnpatched) {
  // A probe object is a code object with a function and no kernel descriptor.
  const std::vector<uint8_t> probe_elf = probe(kGfx950);
  const DbiInstrumentation result = instrument(probe_elf, probe_elf);
  EXPECT_EQ(result.decision, DbiLoadDecision::NotSingleKernel);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(DbiRuntimeSites, LeavesObjectWithTwoKernelsUnpatched) {
  const DbiInstrumentation result =
      instrument(test::make_gfx950_two_kernel_elf(three_block_text(kGfx950), /*private_bytes=*/0),
                 probe(kGfx950));
  EXPECT_EQ(result.decision, DbiLoadDecision::NotSingleKernel);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(DbiRuntimeSites, InstrumentsAWave32Kernel) {
  // CallsTheProbeAtTheFirstAnchorOfEachBlock instruments the same kernel in Wave64.
  const DbiInstrumentation result =
      instrument(kernel(kGfx1200, three_block_text(kGfx1200), /*granulated_sgpr_count=*/3,
                        /*entry_text_offset=*/0, /*wave32=*/true),
                 probe(kGfx1200));
  ASSERT_EQ(result.decision, DbiLoadDecision::Instrumented)
      << (result.errors.empty() ? std::string{} : result.errors.front());
  EXPECT_EQ(result.sites, 2u);
}

TEST(DbiRuntimeSites, LeavesKernelThatMakesACallUnpatched) {
  // 0x00 s_call_b64 s[30:31] -> 0x0c, 0x04 v_mov, 0x08 s_endpgm,
  // 0x0c v_mov, 0x10 s_setpc_b64 s[30:31]. Every block but the s_call_b64's has
  // an anchor.
  const rj_code_arch_t arch = kGfx950.arch;
  const std::vector<uint32_t> text = {build_s_call_b64(/*s[30:31]=*/30, /*offset_dwords=*/2, arch),
                                      kMovV2Zero, build_s_endpgm(arch), kMovV3V2,
                                      build_s_setpc_b64(/*s[30:31]=*/30, arch)};
  const DbiInstrumentation result = instrument(kernel(kGfx950, text), probe(kGfx950));
  expect_refused(result, DbiLoadDecision::UnsupportedKernel, "makes a call");
}

TEST(DbiRuntimeSites, LeavesKernelWithAnUnresolvedJumpUnpatched) {
  const rj_code_arch_t arch = kGfx950.arch;
  const std::vector<uint32_t> text = {kMovV2Zero, build_s_setpc_b64(/*s[4:5]=*/4, arch)};
  const DbiInstrumentation result = instrument(kernel(kGfx950, text), probe(kGfx950));
  expect_refused(result, DbiLoadDecision::UnsupportedKernel, "unresolved successor");
}

TEST(DbiRuntimeSites, LeavesKernelThatBranchesOutOfTextUnpatched) {
  const rj_code_arch_t arch = kGfx950.arch;
  const std::vector<uint32_t> text = {kMovV2Zero, build_s_branch(/*offset_dwords=*/0x7fff, arch),
                                      build_s_endpgm(arch)};
  const DbiInstrumentation result = instrument(kernel(kGfx950, text), probe(kGfx950));
  expect_refused(result, DbiLoadDecision::UnsupportedKernel, "unresolved successor");
}

TEST(DbiRuntimeSites, LeavesDeadCodeAlone) {
  // 0x00 v_mov, 0x04 s_endpgm: the kernel. 0x08 v_mov, 0x0c s_setpc_b64
  // s[30:31]: a function nothing calls. Its return is unresolved control flow,
  // which only matters if the kernel can reach it.
  const rj_code_arch_t arch = kGfx950.arch;
  const std::vector<uint32_t> text = {kMovV2Zero, build_s_endpgm(arch), kMovV3V2,
                                      build_s_setpc_b64(/*s[30:31]=*/30, arch)};
  const DbiInstrumentation result = instrument(kernel(kGfx950, text), probe(kGfx950));
  ASSERT_EQ(result.decision, DbiLoadDecision::Instrumented)
      << (result.errors.empty() ? std::string{} : result.errors.front());
  EXPECT_EQ(result.sites, 1u);

  const AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> words = text_words(patched);
  ASSERT_GE(words.size(), text.size());
  EXPECT_EQ(words[2], text[2]);
  EXPECT_EQ(words[3], text[3]);
}

TEST(DbiRuntimeSites, PlacesTheSiteAtAnEntryPastPadding) {
  // Two s_nop of padding fall into the entry at 8 with no terminator between.
  const rj_code_arch_t arch = kGfx950.arch;
  const std::vector<uint32_t> text = {build_s_nop(0, arch), build_s_nop(0, arch), kMovV2Zero,
                                      build_s_endpgm(arch)};
  const DbiInstrumentation result = instrument(
      kernel(kGfx950, text, /*granulated_sgpr_count=*/3, /*entry_text_offset=*/8), probe(kGfx950));
  ASSERT_EQ(result.decision, DbiLoadDecision::Instrumented)
      << (result.errors.empty() ? std::string{} : result.errors.front());
  EXPECT_EQ(result.sites, 1u);

  const AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> words = text_words(patched);
  ASSERT_GT(words.size(), text.size());
  EXPECT_EQ(words[0], text[0]);
  EXPECT_EQ(words[1], text[1]);
  EXPECT_NE(words[2], text[2]);
  EXPECT_EQ(words[3], text[3]);
}

TEST(DbiRuntimeSites, InstrumentsAnRdna4KernelBehindAStubAndZeroPadding) {
  const rj_code_arch_t arch = kGfx1200.arch;
  std::vector<uint32_t> text = stub_and_zero_padding(arch);
  const uint64_t entry = text.size() * sizeof(uint32_t);
  text.push_back(kMovV2Zero);
  text.push_back(build_s_endpgm(arch));
  const DbiInstrumentation result =
      instrument(kernel(kGfx1200, text, /*granulated_sgpr_count=*/3, entry), probe(kGfx1200));
  ASSERT_EQ(result.decision, DbiLoadDecision::Instrumented)
      << (result.errors.empty() ? std::string{} : result.errors.front());
  EXPECT_EQ(result.sites, 1u);

  const AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> words = text_words(patched);
  ASSERT_GT(words.size(), text.size());
  EXPECT_TRUE(std::equal(text.begin(), text.begin() + entry / sizeof(uint32_t), words.begin()));
}

TEST(DbiRuntimeSites, CallsAnRdna4ProbeBehindAStubAndZeroPadding) {
  const DbiInstrumentation result =
      instrument(kernel(kGfx1200, three_block_text(kGfx1200)),
                 probe(kGfx1200, stub_and_zero_padding(kGfx1200.arch)));
  ASSERT_EQ(result.decision, DbiLoadDecision::Instrumented)
      << (result.errors.empty() ? std::string{} : result.errors.front());
  EXPECT_EQ(result.sites, 2u);
}

TEST(DbiRuntimeSites, LeavesObjectWithNothingToAnchorUnpatched) {
  const DbiInstrumentation result =
      instrument(kernel(kGfx950, {build_s_endpgm(kGfx950.arch)}), probe(kGfx950));
  EXPECT_EQ(result.decision, DbiLoadDecision::NoSites);
  EXPECT_EQ(result.blocks_without_site, 1u);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(DbiRuntimeSites, ReportsInstrumentorRefusal) {
  // 16 SGPRs: the probe's link pair s[30:31] is outside the allocation.
  const DbiInstrumentation result = instrument(
      kernel(kGfx950, three_block_text(kGfx950), /*granulated_sgpr_count=*/1), probe(kGfx950));
  expect_refused(result, DbiLoadDecision::PatchFailed, "return-link pair");
}

TEST(DbiRuntimeSites, ReportsAKernelThatDoesNotDecode) {
  // RDNA4 has no encoding for a zero word, and this one is reachable from the
  // entry, so the kernel's blocks cannot be built.
  const DbiInstrumentation result = instrument(
      kernel(kGfx1200, {kMovV2Zero, 0u, build_s_endpgm(kGfx1200.arch)}), probe(kGfx1200));
  expect_refused(result, DbiLoadDecision::PatchFailed, "Invalid instruction opcode");
}

TEST(DbiRuntimeSites, NamesEveryDecision) {
  for (size_t index = 0; index < kDbiLoadDecisionCount; ++index)
    EXPECT_STRNE(dbi_load_decision_name(static_cast<DbiLoadDecision>(index)), "unknown") << index;
  EXPECT_STREQ(dbi_load_decision_name(DbiLoadDecision::Count), "unknown");
}

} // namespace
} // namespace rocjitsu::hooks
