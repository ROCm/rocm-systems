// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbi_replacement_sim_test.cpp
/// @brief Execute replacement/probe composition with replacement-derived liveness.
/// @details Memory outputs distinguish the requested replacement, probe clobbers,
/// and the omitted original instruction. DbiSim synthesizes its dispatch descriptor;
/// these tests exercise simulated instruction and spill semantics, not hardware
/// resource allocation or physical GPU execution.

#include "../amdhsa_metadata_test_util.h"
#include "../dbi_test_util.h"
#include "dbi_sim.h"
#include "global_memory_builders.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/spill_builders.h"
#include "rocjitsu/code/builders/vector_builders.h"
#include "rocjitsu/code/patch/instrumentor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

struct ReplacementSimTarget {
  std::string_view sim_arch;
  rj_code_arch_t arch;
  uint32_t machine;
  uint32_t wave_size;
};

constexpr ReplacementSimTarget kCdna3 = {.sim_arch = "cdna3",
                                         .arch = ROCJITSU_CODE_ARCH_CDNA3,
                                         .machine = EF_AMDGPU_MACH_AMDGCN_GFX942,
                                         .wave_size = 64};
constexpr ReplacementSimTarget kCdna4 = {.sim_arch = "cdna4",
                                         .arch = ROCJITSU_CODE_ARCH_CDNA4,
                                         .machine = EF_AMDGPU_MACH_AMDGCN_GFX950,
                                         .wave_size = 64};
constexpr ReplacementSimTarget kRdna4 = {.sim_arch = "rdna4",
                                         .arch = ROCJITSU_CODE_ARCH_RDNA4,
                                         .machine = EF_AMDGPU_MACH_AMDGCN_GFX1200,
                                         .wave_size = 32};

constexpr uint64_t kOutputAddress = 0x80000;
constexpr uint32_t kInitialPrivateBytes = 64;
constexpr uint32_t kReplacementInput = 7;
constexpr uint32_t kOriginalInput = 11;
constexpr uint32_t kProbeClobber = 99;
constexpr uint32_t kUnwritten = 0xc0ffee11;

// DbiSim's CP initializes v0 with the lane's local work-item ID. Two additions
// produce a byte offset for a distinct output dword per active lane.
std::optional<uint32_t> double_v0(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_vop2(cdna3::kVAddU32Vop2, {.src0 = 256, .vsrc1 = 0, .vdst = 0}).front();
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::build_vop2(cdna4::kVAddU32Vop2, {.src0 = 256, .vsrc1 = 0, .vdst = 0}).front();
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::build_vop2(rdna4::kVAddNcU32Vop2, {.src0 = 256, .vsrc1 = 0, .vdst = 0}).front();
  default:
    return std::nullopt;
  }
}

std::vector<uint8_t> instruction_bytes(std::span<const uint32_t> words) {
  std::vector<uint8_t> bytes(words.size_bytes());
  if (!bytes.empty())
    std::memcpy(bytes.data(), words.data(), bytes.size());
  return bytes;
}

class DbiReplacementSim : public ::testing::TestWithParam<ReplacementSimTarget> {
protected:
  void SetUp() override {
    const ReplacementSimTarget &target = GetParam();
    const util::FailureOr<uint32_t> from_v1 = build_v_mov_b32_src(3, 257, target.arch);
    const util::FailureOr<uint32_t> from_v2 = build_v_mov_b32_src(3, 258, target.arch);
    ASSERT_TRUE(from_v1.succeeded());
    ASSERT_TRUE(from_v2.succeeded());
    copy_v3_v1_ = from_v1.value();
    copy_v3_v2_ = from_v2.value();
    const std::optional<uint32_t> doubled = double_v0(target.arch);
    ASSERT_TRUE(doubled);
    const util::FailureOr<uint32_t> address_low = build_s_mov_b32(2, 255, target.arch);
    const util::FailureOr<uint32_t> address_high = build_s_mov_b32(3, 128, target.arch);
    const util::FailureOr<uint32_t> probe_return = build_s_setpc_b64(30, target.arch);
    ASSERT_TRUE(address_low.succeeded());
    ASSERT_TRUE(address_high.succeeded());
    ASSERT_TRUE(probe_return.succeeded());
    setup_ = {address_low.value(),
              static_cast<uint32_t>(kOutputAddress),
              address_high.value(),
              *doubled,
              *doubled,
              test::make_mov_vgpr_inline(1, kReplacementInput),
              test::make_mov_vgpr_inline(2, kOriginalInput)};
    const util::FailureOr<std::array<uint32_t, 2>> clobber =
        build_v_mov_b32_imm(1, kProbeClobber, target.arch);
    ASSERT_TRUE(clobber.succeeded());
    probe_words_.assign(clobber.value().begin(), clobber.value().end());
    probe_words_.push_back(probe_return.value());
  }

  void finish_kernel(std::vector<uint32_t> &words) const {
    const util::FailureOr<std::vector<uint32_t>> store =
        test::build_global_store_dword(3, 0, 2, 0, GetParam().arch);
    ASSERT_TRUE(store.succeeded());
    words.insert(words.end(), store.value().begin(), store.value().end());
    const util::FailureOr<uint32_t> terminator = build_s_endpgm(GetParam().arch);
    ASSERT_TRUE(terminator.succeeded());
    words.push_back(terminator.value());
  }

  void make_image(const std::vector<uint32_t> &words, std::vector<uint8_t> &image) const {
    const ReplacementSimTarget &target = GetParam();
    image = test::make_amdgpu_kernel_elf(words, kInitialPrivateBytes, 3, target.machine, 0, 0,
                                         false, false, false, false, target.wave_size == 32);
    image = test::with_descriptor_metadata(image, target.arch);
    ASSERT_FALSE(image.empty());
  }

  InstrumentationPoint replacement(uint64_t offset, uint32_t original,
                                   std::span<const uint32_t> body) const {
    const std::array<uint32_t, 1> original_words = {original};
    return {.anchor_offset = offset,
            .kind = InstrumentationKind::ReplaceWithInst,
            .probe_symbol = {},
            .probe_args = {},
            .replacement =
                InstructionReplacement{.expected_bytes = instruction_bytes(original_words),
                                       .replacement_bytes = instruction_bytes(body)}};
  }

  InstrumentationPoint probe_at(uint64_t offset, const AmdGpuCodeObject &probe) const {
    return {.anchor_offset = offset,
            .kind = InstrumentationKind::BeforeInst,
            .probe_obj = &probe,
            .probe_symbol = "replacement_probe",
            .probe_args = {}};
  }

  void expect_memory(const std::vector<uint32_t> &words, uint32_t private_bytes,
                     uint32_t expected) const {
    const ReplacementSimTarget &target = GetParam();
    test::DbiSim simulator;
    ASSERT_NO_FATAL_FAILURE(simulator.initialize(target.sim_arch, target.wave_size));
    if (target.wave_size == 32)
      simulator.set_wave32_dispatch();
    const std::vector<uint32_t> unwritten(target.wave_size, kUnwritten);
    simulator.write_memory(kOutputAddress, {reinterpret_cast<const uint8_t *>(unwritten.data()),
                                            unwritten.size() * sizeof(uint32_t)});
    const std::optional<std::vector<uint8_t>> output = simulator.run_and_read_memory(
        words, private_bytes, kOutputAddress, unwritten.size() * sizeof(uint32_t));
    ASSERT_TRUE(output) << "replacement kernel did not halt";
    ASSERT_EQ(output->size(), unwritten.size() * sizeof(uint32_t));
    for (uint32_t lane = 0; lane < target.wave_size; ++lane) {
      uint32_t observed = 0;
      std::memcpy(&observed, output->data() + lane * sizeof(uint32_t), sizeof(observed));
      EXPECT_EQ(observed, expected) << "lane " << lane;
    }
  }

  void without_v1_restore(std::vector<uint32_t> &words) const {
    const util::FailureOr<std::vector<uint32_t>> load =
        build_scratch_load_dword(1, kInitialPrivateBytes, GetParam().arch);
    ASSERT_TRUE(load.succeeded());
    const std::vector<uint32_t>::iterator restore =
        std::ranges::search(words, load.value()).begin();
    ASSERT_NE(restore, words.end()) << "replacement-only input v1 was not preserved";
    const util::FailureOr<uint32_t> nop = build_s_nop(0, GetParam().arch);
    ASSERT_TRUE(nop.succeeded());
    for (size_t index = 0; index < load.value().size(); ++index)
      *(restore + static_cast<std::ptrdiff_t>(index)) = nop.value();
  }

  std::vector<uint32_t> setup_;
  std::vector<uint32_t> probe_words_;
  uint32_t copy_v3_v1_ = 0;
  uint32_t copy_v3_v2_ = 0;
};

TEST_P(DbiReplacementSim, SameSiteProbePreservesTheReplacementInputInEitherQueueOrder) {
  std::vector<uint32_t> words = setup_;
  const uint64_t anchor = words.size() * sizeof(uint32_t);
  words.push_back(copy_v3_v2_);
  finish_kernel(words);
  ASSERT_NO_FATAL_FAILURE(expect_memory(words, kInitialPrivateBytes, kOriginalInput));
  std::vector<uint8_t> image;
  ASSERT_NO_FATAL_FAILURE(make_image(words, image));
  const AmdGpuCodeObject source(image.data(), image.size());
  const std::vector<uint8_t> probe_image =
      test::make_amdgpu_probe_elf("replacement_probe", probe_words_, GetParam().machine);
  const AmdGpuCodeObject probe(probe_image.data(), probe_image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_TRUE(probe.is_valid());
  const std::array<uint32_t, 1> body = {copy_v3_v1_};
  for (int request_shape = 0; request_shape < 3; ++request_shape) {
    SCOPED_TRACE(request_shape);
    Instrumentor instrumentor(source, GetParam().arch);
    InstrumentationPoint mutation = replacement(anchor, copy_v3_v2_, body);
    InstrumentationPoint observation = probe_at(anchor, probe);
    if (request_shape == 0) {
      mutation.probe_obj = &probe;
      mutation.probe_symbol = "replacement_probe";
      instrumentor.add_point(std::move(mutation));
    } else if (request_shape == 1) {
      instrumentor.add_point(std::move(observation));
      instrumentor.add_point(std::move(mutation));
    } else {
      instrumentor.add_point(std::move(mutation));
      instrumentor.add_point(std::move(observation));
    }
    const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    ASSERT_EQ(result.patches.size(), 1u);
    ASSERT_TRUE(result.patches.front().is_probe_call);
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    const std::vector<uint32_t> patched = test::section_words(candidate, ".text");
    const uint32_t scratch = test::patched_private_segment_size(candidate);
    EXPECT_GT(scratch, kInitialPrivateBytes);
    ASSERT_NO_FATAL_FAILURE(expect_memory(patched, scratch, kReplacementInput));
    // A stale liveness model regards v1 as dead because the old instruction
    // reads v2. Removing the actual restore must expose the probe's 99 instead.
    std::vector<uint32_t> missing_restore = patched;
    ASSERT_NO_FATAL_FAILURE(without_v1_restore(missing_restore));
    ASSERT_NO_FATAL_FAILURE(expect_memory(missing_restore, scratch, kProbeClobber));
    EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
  }
}

TEST_P(DbiReplacementSim, RemovingALaterDefinitionChangesLivenessAtAnEarlierProbe) {
  std::vector<uint32_t> words = setup_;
  const uint64_t probe_offset = words.size() * sizeof(uint32_t);
  const util::FailureOr<uint32_t> nop = build_s_nop(0, GetParam().arch);
  ASSERT_TRUE(nop.succeeded());
  words.push_back(nop.value());
  const uint64_t definition_offset = words.size() * sizeof(uint32_t);
  const uint32_t definition = test::make_mov_vgpr_inline(1, kOriginalInput);
  words.push_back(definition);
  words.push_back(copy_v3_v1_);
  finish_kernel(words);
  ASSERT_NO_FATAL_FAILURE(expect_memory(words, kInitialPrivateBytes, kOriginalInput));
  std::vector<uint8_t> image;
  ASSERT_NO_FATAL_FAILURE(make_image(words, image));
  const AmdGpuCodeObject source(image.data(), image.size());
  const std::vector<uint8_t> probe_image =
      test::make_amdgpu_probe_elf("replacement_probe", probe_words_, GetParam().machine);
  const AmdGpuCodeObject probe(probe_image.data(), probe_image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_TRUE(probe.is_valid());
  Instrumentor instrumentor(source, GetParam().arch);
  instrumentor.add_point(probe_at(probe_offset, probe));
  instrumentor.add_point(replacement(definition_offset, definition, {}));
  const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(candidate.is_valid());
  const std::vector<uint32_t> patched = test::section_words(candidate, ".text");
  const uint32_t scratch = test::patched_private_segment_size(candidate);
  EXPECT_GT(scratch, kInitialPrivateBytes);
  ASSERT_NO_FATAL_FAILURE(expect_memory(patched, scratch, kReplacementInput));
  std::vector<uint32_t> missing_restore = patched;
  ASSERT_NO_FATAL_FAILURE(without_v1_restore(missing_restore));
  ASSERT_NO_FATAL_FAILURE(expect_memory(missing_restore, scratch, kProbeClobber));
  // Reinstalling only the deleted definition must restore the original 11,
  // independently proving that the empty replacement removed its effect.
  std::vector<uint32_t> replayed = patched;
  ASSERT_LT(definition_offset / sizeof(uint32_t), replayed.size());
  replayed[definition_offset / sizeof(uint32_t)] = definition;
  ASSERT_NO_FATAL_FAILURE(expect_memory(replayed, scratch, kOriginalInput));
  EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
}

TEST_P(DbiReplacementSim, EmptyReplacementRemovesTheOriginalEffect) {
  std::vector<uint32_t> words = setup_;
  const uint64_t anchor = words.size() * sizeof(uint32_t);
  const uint32_t definition = test::make_mov_vgpr_inline(1, kOriginalInput);
  words.push_back(definition);
  words.push_back(copy_v3_v1_);
  finish_kernel(words);
  ASSERT_NO_FATAL_FAILURE(expect_memory(words, kInitialPrivateBytes, kOriginalInput));
  std::vector<uint8_t> image;
  ASSERT_NO_FATAL_FAILURE(make_image(words, image));
  const AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  Instrumentor instrumentor(source, GetParam().arch);
  instrumentor.add_point(replacement(anchor, definition, {}));
  const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  ASSERT_EQ(result.sites.size(), 1u);
  EXPECT_EQ(result.sites.front().guest_size, 0u);
  const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(candidate.is_valid());
  const std::vector<uint32_t> patched = test::section_words(candidate, ".text");
  const uint32_t scratch = test::patched_private_segment_size(candidate);
  ASSERT_NO_FATAL_FAILURE(expect_memory(patched, scratch, kReplacementInput));
  std::vector<uint32_t> replayed = patched;
  ASSERT_LT(anchor / sizeof(uint32_t), replayed.size());
  replayed[anchor / sizeof(uint32_t)] = definition;
  ASSERT_NO_FATAL_FAILURE(expect_memory(replayed, scratch, kOriginalInput));
  EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
}

INSTANTIATE_TEST_SUITE_P(Targets, DbiReplacementSim, ::testing::Values(kCdna3, kCdna4, kRdna4),
                         [](const ::testing::TestParamInfo<ReplacementSimTarget> &info) {
                           return std::string(info.param.sim_arch);
                         });

} // namespace
} // namespace rocjitsu
