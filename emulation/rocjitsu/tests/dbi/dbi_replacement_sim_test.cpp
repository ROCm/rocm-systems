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
#include "rocjitsu/code/kernel_descriptor_scan.h"
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
  rocjitsu::IsaFamily arch;
  uint32_t machine;
  uint32_t wave_size;
};

constexpr ReplacementSimTarget kCdna3 = {.sim_arch = "cdna3",
                                         .arch = rocjitsu::IsaFamily::Cdna3,
                                         .machine = EF_AMDGPU_MACH_AMDGCN_GFX942,
                                         .wave_size = 64};
constexpr ReplacementSimTarget kCdna4 = {.sim_arch = "cdna4",
                                         .arch = rocjitsu::IsaFamily::Cdna4,
                                         .machine = EF_AMDGPU_MACH_AMDGCN_GFX950,
                                         .wave_size = 64};
constexpr ReplacementSimTarget kRdna4 = {.sim_arch = "rdna4",
                                         .arch = rocjitsu::IsaFamily::Rdna4,
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
std::optional<uint32_t> double_v0(rocjitsu::IsaFamily arch) {
  switch (arch) {
  case rocjitsu::IsaFamily::Cdna3:
    return cdna3::build_vop2(cdna3::kVAddU32Vop2, {.src0 = 256, .vsrc1 = 0, .vdst = 0}).front();
  case rocjitsu::IsaFamily::Cdna4:
    return cdna4::build_vop2(cdna4::kVAddU32Vop2, {.src0 = 256, .vsrc1 = 0, .vdst = 0}).front();
  case rocjitsu::IsaFamily::Rdna4:
    return rdna4::build_vop2(rdna4::kVAddNcU32Vop2, {.src0 = 256, .vsrc1 = 0, .vdst = 0}).front();
  default:
    return std::nullopt;
  }
}

std::vector<uint8_t> instruction_bytes(std::span<const uint32_t> words) {
  std::vector<uint8_t> bytes(words.size_bytes());
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), words.data(), bytes.size());
  }
  return bytes;
}

class DbiReplacementSim : public ::testing::TestWithParam<ReplacementSimTarget> {
protected:
  enum class LoggerOutput { PerLane, Shared };
  enum class ObservationDefinition { Full, Partial, Unknown, Absent };
  struct ObservationProgram {
    std::vector<uint8_t> image;
    uint64_t early_offset = 0;
    uint64_t observer_offset = 0;
  };

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

  void make_probe_images(LoggerOutput output, std::vector<uint8_t> &logger_image,
                         std::vector<uint8_t> &clobber_image) const {
    const rocjitsu::IsaFamily arch = GetParam().arch;
    const util::FailureOr<uint32_t> address_low = build_s_mov_b32(2, 255, arch);
    const util::FailureOr<uint32_t> address_high = build_s_mov_b32(3, 128, arch);
    const util::FailureOr<uint32_t> ret = build_s_setpc_b64(30, arch);
    ASSERT_TRUE(address_low.succeeded());
    ASSERT_TRUE(address_high.succeeded());
    ASSERT_TRUE(ret.succeeded());
    std::vector<uint32_t> words = {address_low.value(), static_cast<uint32_t>(kOutputAddress),
                                   address_high.value()};
    if (output == LoggerOutput::Shared) {
      // One v1 -> v0 argument avoids a cycle or a v1 clobber at the observer.
      const util::FailureOr<uint32_t> zero_offset =
          build_v_mov_b32_src(2, scalar_positive_inline_u32(0), arch);
      ASSERT_TRUE(zero_offset.succeeded());
      words.push_back(zero_offset.value());
    }
    const util::FailureOr<std::vector<uint32_t>> store =
        test::build_global_store_dword(0, output == LoggerOutput::PerLane ? 1 : 2, 2, 0, arch);
    ASSERT_TRUE(store.succeeded());
    words.insert(words.end(), store.value().begin(), store.value().end());
    words.push_back(ret.value());
    logger_image = test::make_amdgpu_probe_elf("replacement_probe", words, GetParam().machine);
    clobber_image =
        test::make_amdgpu_probe_elf("replacement_probe", probe_words_, GetParam().machine);
  }

  void make_observation_program(ObservationDefinition definition, uint32_t private_bytes,
                                ObservationProgram &program, bool interior_entry = false) const {
    const rocjitsu::IsaFamily arch = GetParam().arch;
    const util::FailureOr<uint32_t> nop = build_s_nop(0, arch);
    const util::FailureOr<uint32_t> end = build_s_endpgm(arch);
    ASSERT_TRUE(nop.succeeded());
    ASSERT_TRUE(end.succeeded());
    std::vector<uint32_t> words = setup_;
    program.early_offset = words.size() * sizeof(uint32_t);
    words.push_back(nop.value());
    if (definition == ObservationDefinition::Full || definition == ObservationDefinition::Partial) {
      const util::FailureOr<uint16_t> exec = scalar_operand_exec_lo(arch);
      const util::FailureOr<uint16_t> all_ones = scalar_inline_neg_one(arch);
      ASSERT_TRUE(exec.succeeded());
      ASSERT_TRUE(all_ones.succeeded());
      const util::FailureOr<uint32_t> set_exec =
          build_s_mov_b64(exec.value(),
                          definition == ObservationDefinition::Full ? all_ones.value()
                                                                    : scalar_positive_inline_u32(1),
                          arch);
      ASSERT_TRUE(set_exec.succeeded());
      words.push_back(set_exec.value());
    }
    if (definition != ObservationDefinition::Absent) {
      words.push_back(test::make_mov_vgpr_inline(1, kReplacementInput));
    }
    program.observer_offset = words.size() * sizeof(uint32_t);
    words.push_back(nop.value());
    words.push_back(end.value());
    make_image(words, program.image, private_bytes, interior_entry ? program.early_offset : 0);
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

  void make_image(const std::vector<uint32_t> &words, std::vector<uint8_t> &image,
                  uint32_t private_bytes = kInitialPrivateBytes, uint64_t entry_offset = 0) const {
    const ReplacementSimTarget &target = GetParam();
    image = test::make_amdgpu_kernel_elf(
        words, private_bytes, test::kCallableProbeGranulatedSgprCount, target.machine, 0, 0, false,
        false, false, false, target.wave_size == 32, false, 0, entry_offset);
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

  void expect_memory(const std::vector<uint32_t> &words, uint32_t private_bytes, uint32_t expected,
                     uint32_t output_dwords = 0, uint64_t entry_offset = 0) const {
    const ReplacementSimTarget &target = GetParam();
    test::DbiSim simulator;
    ASSERT_NO_FATAL_FAILURE(simulator.initialize(target.sim_arch, target.wave_size));
    simulator.set_entry_offset(entry_offset);
    if (target.wave_size == 32) {
      simulator.set_wave32_dispatch();
    }
    const uint32_t output_count = output_dwords == 0 ? target.wave_size : output_dwords;
    const std::vector<uint32_t> unwritten(output_count, kUnwritten);
    simulator.write_memory(kOutputAddress, {reinterpret_cast<const uint8_t *>(unwritten.data()),
                                            unwritten.size() * sizeof(uint32_t)});
    const std::optional<std::vector<uint8_t>> output = simulator.run_and_read_memory(
        words, private_bytes, kOutputAddress, unwritten.size() * sizeof(uint32_t));
    ASSERT_TRUE(output) << "replacement kernel did not halt";
    ASSERT_EQ(output->size(), unwritten.size() * sizeof(uint32_t));
    for (uint32_t lane = 0; lane < output_count; ++lane) {
      uint32_t observed = 0;
      std::memcpy(&observed, output->data() + lane * sizeof(uint32_t), sizeof(observed));
      EXPECT_EQ(observed, expected) << "lane " << lane;
    }
  }

  void without_v1_restore(std::span<uint32_t> words,
                          uint32_t private_bytes = kInitialPrivateBytes + sizeof(uint32_t),
                          bool verify_store = false) const {
    for (uint32_t offset = kInitialPrivateBytes; offset < private_bytes;
         offset += sizeof(uint32_t)) {
      const util::FailureOr<std::vector<uint32_t>> load =
          build_scratch_load_dword(1, offset, GetParam().arch);
      ASSERT_TRUE(load.succeeded());
      const std::span<uint32_t>::iterator restore =
          std::ranges::search(words, load.value()).begin();
      if (restore == words.end()) {
        continue;
      }
      if (verify_store) {
        const util::FailureOr<std::vector<uint32_t>> store =
            build_scratch_store_dword(1, offset, GetParam().arch);
        ASSERT_TRUE(store.succeeded());
        ASSERT_NE(std::ranges::search(words, store.value()).begin(), words.end());
      }
      const util::FailureOr<uint32_t> nop = build_s_nop(0, GetParam().arch);
      ASSERT_TRUE(nop.succeeded());
      std::fill_n(restore, load.value().size(), nop.value());
      return;
    }
    FAIL() << "v1 scratch preservation was not found in the selected instruction range";
  }

  void without_site_v1_restore(std::vector<uint32_t> &words, const InstrumentedSite &site,
                               uint32_t private_bytes) const {
    ASSERT_EQ(site.instrumentation_offset % sizeof(uint32_t), 0u);
    ASSERT_EQ(site.instrumentation_size % sizeof(uint32_t), 0u);
    const size_t begin = site.instrumentation_offset / sizeof(uint32_t);
    const size_t size = site.instrumentation_size / sizeof(uint32_t);
    ASSERT_LE(begin + size, words.size());
    ASSERT_NO_FATAL_FAILURE(
        without_v1_restore(std::span(words).subspan(begin, size), private_bytes, true));
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
  ASSERT_NO_FATAL_FAILURE(finish_kernel(words));
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
  ASSERT_NO_FATAL_FAILURE(finish_kernel(words));
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

TEST_P(DbiReplacementSim, LaterProbeArgumentsSurviveDeletingTheirOnlyGuestRead) {
  const ReplacementSimTarget &target = GetParam();
  const util::FailureOr<uint32_t> nop = build_s_nop(0, target.arch);
  const util::FailureOr<uint32_t> end = build_s_endpgm(target.arch);
  const util::FailureOr<uint32_t> read_offset = build_v_mov_b32_src(3, 256, target.arch);
  ASSERT_TRUE(nop.succeeded());
  ASSERT_TRUE(end.succeeded());
  ASSERT_TRUE(read_offset.succeeded());
  // The observer receives the value in v0 and the lane's byte offset in v1.
  // Only the probe writes output, so guest execution cannot hide a bad capture.
  std::vector<uint8_t> logger_image;
  std::vector<uint8_t> clobber_image;
  ASSERT_NO_FATAL_FAILURE(make_probe_images(LoggerOutput::PerLane, logger_image, clobber_image));
  const AmdGpuCodeObject logger(logger_image.data(), logger_image.size());
  const AmdGpuCodeObject clobber(clobber_image.data(), clobber_image.size());
  ASSERT_TRUE(logger.is_valid());
  ASSERT_TRUE(clobber.is_valid());
  constexpr std::array<std::string_view, 5> cases = {
      "retained guest read", "guest redefinition", "attached deletion",
      "replacement queued before observer", "observer queued before replacement"};
  for (size_t request_shape = 0; request_shape < cases.size(); ++request_shape) {
    SCOPED_TRACE(cases[request_shape]);
    const bool retained = request_shape == 0;
    const bool redefined = request_shape == 1;
    std::vector<uint32_t> words = setup_;
    const uint64_t early_offset = words.size() * sizeof(uint32_t);
    words.push_back(nop.value());
    if (redefined) {
      words.push_back(test::make_mov_vgpr_inline(1, kOriginalInput));
    }
    const uint64_t observer_offset = words.size() * sizeof(uint32_t);
    words.push_back(copy_v3_v1_);
    // Keep the address argument live in the guest to isolate the value argument.
    words.push_back(read_offset.value());
    words.push_back(end.value());
    std::vector<uint8_t> image;
    ASSERT_NO_FATAL_FAILURE(make_image(words, image));
    const std::vector<uint8_t> pristine = image;
    const AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    Instrumentor instrumentor(source, target.arch);
    InstrumentationPoint early = probe_at(early_offset, clobber);
    InstrumentationPoint observer = probe_at(observer_offset, logger);
    observer.probe_args = {probe_arg_vgpr(1), probe_arg_vgpr(0)};
    InstrumentationPoint mutation = replacement(observer_offset, copy_v3_v1_, {});
    if (retained) {
      instrumentor.add_point(std::move(early));
      instrumentor.add_point(std::move(observer));
    } else if (request_shape <= 2) {
      mutation.probe_obj = &logger;
      mutation.probe_symbol = "replacement_probe";
      mutation.probe_args = observer.probe_args;
      instrumentor.add_point(std::move(early));
      instrumentor.add_point(std::move(mutation));
    } else {
      if (request_shape == 3) {
        instrumentor.add_point(std::move(mutation));
        instrumentor.add_point(std::move(observer));
      } else {
        instrumentor.add_point(std::move(observer));
        instrumentor.add_point(std::move(mutation));
      }
      instrumentor.add_point(std::move(early));
    }
    const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    ASSERT_EQ(result.sites.size(), 2u);
    ASSERT_EQ(result.patches.size(), 2u);
    EXPECT_TRUE(std::ranges::all_of(result.patches, &InstrumentationPatch::is_probe_call));
    ASSERT_EQ(source.image_size(), pristine.size());
    EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    const std::vector<uint32_t> patched = test::section_words(candidate, ".text");
    const uint32_t scratch = test::patched_private_segment_size(candidate);
    ASSERT_NO_FATAL_FAILURE(
        expect_memory(patched, scratch, redefined ? kOriginalInput : kReplacementInput));
    if (!redefined) {
      const std::vector<InstrumentedSite>::const_iterator early_site =
          std::ranges::find(result.sites, early_offset, &InstrumentedSite::original_offset);
      ASSERT_NE(early_site, result.sites.end());
      std::vector<uint32_t> missing_restore = patched;
      ASSERT_NO_FATAL_FAILURE(without_site_v1_restore(missing_restore, *early_site, scratch));
      ASSERT_NO_FATAL_FAILURE(expect_memory(missing_restore, scratch, kProbeClobber));
    }
  }
}

TEST_P(DbiReplacementSim, ObservationLivenessRespectsDefinitionMasksAndScratchAvailability) {
  std::vector<uint8_t> logger_image;
  std::vector<uint8_t> clobber_image;
  ASSERT_NO_FATAL_FAILURE(make_probe_images(LoggerOutput::Shared, logger_image, clobber_image));
  const AmdGpuCodeObject logger(logger_image.data(), logger_image.size());
  const AmdGpuCodeObject clobber(clobber_image.data(), clobber_image.size());
  ASSERT_TRUE(logger.is_valid());
  ASSERT_TRUE(clobber.is_valid());
  struct Case {
    ObservationDefinition definition;
    uint32_t private_bytes;
  };
  constexpr std::array<Case, 4> cases = {
      Case{.definition = ObservationDefinition::Full, .private_bytes = 0},
      Case{.definition = ObservationDefinition::Full, .private_bytes = kInitialPrivateBytes},
      Case{.definition = ObservationDefinition::Partial, .private_bytes = 0},
      Case{.definition = ObservationDefinition::Unknown, .private_bytes = 0}};
  for (const Case &test_case : cases) {
    SCOPED_TRACE(static_cast<int>(test_case.definition));
    SCOPED_TRACE(test_case.private_bytes);
    ObservationProgram program;
    ASSERT_NO_FATAL_FAILURE(
        make_observation_program(test_case.definition, test_case.private_bytes, program));
    const std::vector<uint8_t> pristine = program.image;
    const AmdGpuCodeObject source(program.image.data(), program.image.size());
    ASSERT_TRUE(source.is_valid());
    Instrumentor instrumentor(source, GetParam().arch);
    instrumentor.add_point(probe_at(program.early_offset, clobber));
    InstrumentationPoint observer = probe_at(program.observer_offset, logger);
    observer.probe_args = {probe_arg_vgpr(1)};
    instrumentor.add_point(std::move(observer));
    const Instrumentor::ValidationResult validated = instrumentor.validate_points();
    ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
    ASSERT_EQ(validated.sites.size(), 2u);
    const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
    ASSERT_EQ(source.image_size(), pristine.size());
    EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
    if (test_case.definition != ObservationDefinition::Full) {
      EXPECT_TRUE(std::ranges::any_of(result.errors, [](const std::string &error) {
        return error.find("fixed scratch to spill into") != std::string::npos;
      }));
      EXPECT_TRUE(result.elf_bytes.empty());
      EXPECT_TRUE(result.sites.empty());
      continue;
    }
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    ASSERT_EQ(result.sites.size(), 2u);
    ASSERT_EQ(result.patches.size(), 2u);
    EXPECT_TRUE(std::ranges::all_of(result.sites, [](const InstrumentedSite &site) {
      return site.kind == InstrumentationKind::BeforeInst;
    }));
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    const uint32_t scratch = test::patched_private_segment_size(candidate);
    EXPECT_EQ(scratch, test_case.private_bytes);
    ASSERT_NO_FATAL_FAILURE(
        expect_memory(test::section_words(candidate, ".text"), scratch, kReplacementInput, 1));
  }
}

TEST_P(DbiReplacementSim, AddingALaterObserverInvalidatesCachedLiveness) {
  ObservationProgram program;
  ASSERT_NO_FATAL_FAILURE(
      make_observation_program(ObservationDefinition::Absent, kInitialPrivateBytes, program));
  const std::vector<uint8_t> pristine = program.image;
  const AmdGpuCodeObject source(program.image.data(), program.image.size());
  std::vector<uint8_t> logger_image;
  std::vector<uint8_t> clobber_image;
  ASSERT_NO_FATAL_FAILURE(make_probe_images(LoggerOutput::Shared, logger_image, clobber_image));
  const AmdGpuCodeObject logger(logger_image.data(), logger_image.size());
  const AmdGpuCodeObject clobber(clobber_image.data(), clobber_image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_TRUE(logger.is_valid());
  ASSERT_TRUE(clobber.is_valid());
  Instrumentor instrumentor(source, GetParam().arch);
  instrumentor.add_point(probe_at(program.early_offset, clobber));
  // No replacement or later guest definition can hide a stale analysis cache.
  const Instrumentor::ValidationResult warmed = instrumentor.validate_points();
  ASSERT_TRUE(warmed.errors.empty()) << warmed.errors.front();
  ASSERT_EQ(warmed.sites.size(), 1u);
  InstrumentationPoint observer = probe_at(program.observer_offset, logger);
  observer.probe_args = {probe_arg_vgpr(1)};
  instrumentor.add_point(std::move(observer));
  const Instrumentor::ValidationResult validated = instrumentor.validate_points();
  ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
  ASSERT_EQ(validated.sites.size(), 2u);
  const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  ASSERT_EQ(result.sites.size(), 2u);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_TRUE(std::ranges::all_of(result.sites, [](const InstrumentedSite &site) {
    return site.kind == InstrumentationKind::BeforeInst;
  }));
  ASSERT_EQ(source.image_size(), pristine.size());
  EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
  const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(candidate.is_valid());
  const uint32_t scratch = test::patched_private_segment_size(candidate);
  EXPECT_GT(scratch, kInitialPrivateBytes);
  const std::vector<uint32_t> patched = test::section_words(candidate, ".text");
  ASSERT_NO_FATAL_FAILURE(expect_memory(patched, scratch, kReplacementInput, 1));
  const std::vector<InstrumentedSite>::const_iterator early_site =
      std::ranges::find(result.sites, program.early_offset, &InstrumentedSite::original_offset);
  ASSERT_NE(early_site, result.sites.end());
  std::vector<uint32_t> missing_restore = patched;
  ASSERT_NO_FATAL_FAILURE(without_site_v1_restore(missing_restore, *early_site, scratch));
  ASSERT_NO_FATAL_FAILURE(expect_memory(missing_restore, scratch, kProbeClobber, 1));
}

class DbiVskipReplacementSim : public DbiReplacementSim {};

TEST_P(DbiVskipReplacementSim, SuppressedGuestDefinitionsPreserveTheEarlierProbeInput) {
  const rocjitsu::IsaFamily arch = GetParam().arch;
  const util::FailureOr<uint32_t> nop = build_s_nop(0, arch);
  const util::FailureOr<uint32_t> end = build_s_endpgm(arch);
  const util::FailureOr<uint32_t> set_exec = build_s_mov_b64(126, 193, arch);
  const util::FailureOr<uint32_t> set_s4 =
      build_s_mov_b32(4, scalar_positive_inline_u32(kReplacementInput), arch);
  const util::FailureOr<uint32_t> copy_s4 = build_v_mov_b32_src(1, 4, arch);
  const util::FailureOr<uint32_t> clobber_s4 = build_s_mov_b32(4, 255, arch);
  const util::FailureOr<uint32_t> ret = build_s_setpc_b64(30, arch);
  ASSERT_TRUE(nop.succeeded());
  ASSERT_TRUE(end.succeeded());
  ASSERT_TRUE(set_exec.succeeded());
  ASSERT_TRUE(set_s4.succeeded());
  ASSERT_TRUE(copy_s4.succeeded());
  ASSERT_TRUE(clobber_s4.succeeded());
  ASSERT_TRUE(ret.succeeded());
  enum class VskipWrite { Set, Clear, ImmediateMode, RegisterMode };
  struct Case {
    VskipWrite write;
    bool earlier_probe;
    bool scalar_destination;
  };
  constexpr std::array<Case, 6> cases = {
      Case{.write = VskipWrite::Set, .earlier_probe = true, .scalar_destination = false},
      Case{.write = VskipWrite::Clear, .earlier_probe = true, .scalar_destination = false},
      Case{.write = VskipWrite::Set, .earlier_probe = false, .scalar_destination = false},
      Case{.write = VskipWrite::ImmediateMode, .earlier_probe = true, .scalar_destination = false},
      Case{.write = VskipWrite::RegisterMode, .earlier_probe = true, .scalar_destination = false},
      Case{.write = VskipWrite::Set, .earlier_probe = true, .scalar_destination = true}};
  for (const Case &test_case : cases) {
    SCOPED_TRACE(static_cast<int>(test_case.write));
    SCOPED_TRACE(test_case.earlier_probe);
    SCOPED_TRACE(test_case.scalar_destination);
    std::vector<uint8_t> logger_image;
    std::vector<uint8_t> clobber_image;
    ASSERT_NO_FATAL_FAILURE(make_probe_images(LoggerOutput::Shared, logger_image, clobber_image));
    if (test_case.scalar_destination) {
      clobber_image = test::make_amdgpu_probe_elf("replacement_probe",
                                                  {clobber_s4.value(), kProbeClobber, ret.value()},
                                                  GetParam().machine);
    }
    const AmdGpuCodeObject logger(logger_image.data(), logger_image.size());
    const AmdGpuCodeObject clobber(clobber_image.data(), clobber_image.size());
    ASSERT_TRUE(logger.is_valid());
    ASSERT_TRUE(clobber.is_valid());
    std::vector<uint32_t> words = setup_;
    if (test_case.scalar_destination) {
      words.push_back(set_s4.value());
    }
    const uint64_t early_offset = words.size() * sizeof(uint32_t);
    words.push_back(nop.value());
    words.push_back(set_exec.value());
    switch (test_case.write) {
    case VskipWrite::Set:
      words.push_back(0xbf108081u); // s_setvskip 1, 0
      break;
    case VskipWrite::Clear:
      words.push_back(0xbf108080u); // s_setvskip 0, 0
      break;
    case VskipWrite::ImmediateMode:
      words.insert(words.end(), {0xba000701u, 1u}); // s_setreg_imm32_b32 MODE[28], 1
      words.push_back(0xbf800001u);                 // s_nop 1: MODE write wait states
      break;
    case VskipWrite::RegisterMode: {
      const util::FailureOr<uint32_t> set_s7 =
          build_s_mov_b32(7, scalar_positive_inline_u32(1), arch);
      ASSERT_TRUE(set_s7.succeeded());
      words.insert(words.end(), {set_s7.value(), 0xb9070701u, 0xbf800001u});
      // s_setreg_b32 MODE[28], s7; s_nop 1
      break;
    }
    }
    // VSKIP suppresses vector encodings with either a VGPR or an SGPR result.
    words.push_back(test_case.scalar_destination ? 0x7e080502u // v_readfirstlane_b32 s4, v2
                                                 : test::make_mov_vgpr_inline(1, kOriginalInput));
    words.push_back(0xbf108080u); // Both callbacks execute with VSKIP clear.
    if (test_case.scalar_destination) {
      words.push_back(copy_s4.value());
    }
    const uint64_t observer_offset = words.size() * sizeof(uint32_t);
    words.push_back(nop.value());
    words.push_back(end.value());
    std::vector<uint8_t> image;
    ASSERT_NO_FATAL_FAILURE(make_image(words, image));
    const std::vector<uint8_t> pristine = image;
    const AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    Instrumentor instrumentor(source, arch);
    if (test_case.earlier_probe) {
      instrumentor.add_point(probe_at(early_offset, clobber));
    }
    InstrumentationPoint observer = replacement(observer_offset, nop.value(), {});
    observer.probe_obj = &logger;
    observer.probe_symbol = "replacement_probe";
    observer.probe_args = {probe_arg_vgpr(1)};
    instrumentor.add_point(std::move(observer));
    const Instrumentor::ValidationResult validated = instrumentor.validate_points();
    ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
    const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    EXPECT_EQ(result.sites.size(), test_case.earlier_probe ? 2u : 1u);
    ASSERT_EQ(source.image_size(), pristine.size());
    EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    const uint32_t expected =
        test_case.write == VskipWrite::Clear ? kOriginalInput : kReplacementInput;
    ASSERT_NO_FATAL_FAILURE(expect_memory(test::section_words(candidate, ".text"),
                                          test::patched_private_segment_size(candidate), expected,
                                          1));
  }
}

TEST_P(DbiVskipReplacementSim, CallableSiteRequiresVskipClearAtItsActualBoundary) {
  const util::FailureOr<uint32_t> nop = build_s_nop(0, GetParam().arch);
  const util::FailureOr<uint32_t> end = build_s_endpgm(GetParam().arch);
  ASSERT_TRUE(nop.succeeded());
  ASSERT_TRUE(end.succeeded());
  std::vector<uint8_t> logger_image;
  std::vector<uint8_t> clobber_image;
  ASSERT_NO_FATAL_FAILURE(make_probe_images(LoggerOutput::Shared, logger_image, clobber_image));
  const AmdGpuCodeObject logger(logger_image.data(), logger_image.size());
  ASSERT_TRUE(logger.is_valid());
  struct Case {
    std::vector<uint32_t> prefix;
    bool accepted;
  };
  const std::array<Case, 4> cases = {
      Case{.prefix = {0xbf108081u}, .accepted = false},              // s_setvskip 1, 0
      Case{.prefix = {0xb9070701u, 0xbf800001u}, .accepted = false}, // Unknown MODE[28].
      Case{.prefix = {0xbf108081u, 0xbf108080u}, .accepted = true},
      // A generic immediate MODE write proves VSKIP clear before the callback.
      Case{.prefix = {0xbf108081u, 0xba000701u, 0u, 0xbf800001u}, .accepted = true}};
  for (const Case &test_case : cases) {
    for (bool replacement_site : {false, true}) {
      SCOPED_TRACE(test_case.accepted);
      SCOPED_TRACE(test_case.prefix.size());
      SCOPED_TRACE(replacement_site);
      std::vector<uint32_t> words = setup_;
      words.insert(words.end(), test_case.prefix.begin(), test_case.prefix.end());
      const uint64_t anchor = words.size() * sizeof(uint32_t);
      words.insert(words.end(), {nop.value(), 0xbf108080u, end.value()});
      std::vector<uint8_t> image;
      ASSERT_NO_FATAL_FAILURE(make_image(words, image));
      const std::vector<uint8_t> pristine = image;
      const AmdGpuCodeObject source(image.data(), image.size());
      ASSERT_TRUE(source.is_valid());
      Instrumentor instrumentor(source, GetParam().arch);
      InstrumentationPoint point =
          replacement_site ? replacement(anchor, nop.value(), {}) : probe_at(anchor, logger);
      point.probe_obj = &logger;
      point.probe_symbol = "replacement_probe";
      point.probe_args = {probe_arg_vgpr(1)};
      instrumentor.add_point(std::move(point));
      const Instrumentor::ValidationResult validated = instrumentor.validate_points();
      if (test_case.accepted) {
        ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
        ASSERT_EQ(validated.sites.size(), 1u);
      } else {
        ASSERT_FALSE(validated.errors.empty());
        EXPECT_NE(validated.errors.front().find("probe observation may inherit enabled VSKIP"),
                  std::string::npos)
            << validated.errors.front();
        EXPECT_TRUE(validated.sites.empty());
      }
      const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
      ASSERT_EQ(source.image_size(), pristine.size());
      EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
      if (!test_case.accepted) {
        EXPECT_TRUE(result.elf_bytes.empty());
        EXPECT_TRUE(result.sites.empty());
        EXPECT_TRUE(result.patches.empty());
        ASSERT_FALSE(result.errors.empty());
        EXPECT_NE(result.errors.front().find("VSKIP"), std::string::npos);
        continue;
      }
      ASSERT_TRUE(result.errors.empty()) << result.errors.front();
      ASSERT_EQ(result.sites.size(), 1u);
      const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
      ASSERT_TRUE(candidate.is_valid());
      ASSERT_NO_FATAL_FAILURE(expect_memory(test::section_words(candidate, ".text"),
                                            test::patched_private_segment_size(candidate),
                                            kReplacementInput, 1));
    }
  }
}

TEST_P(DbiVskipReplacementSim, LocalClearDoesNotProveCallableStateAcrossAnOpaqueTransfer) {
  const rocjitsu::IsaFamily arch = GetParam().arch;
  const util::FailureOr<uint32_t> transfer = build_s_setpc_b64(8, arch);
  const util::FailureOr<uint32_t> nop = build_s_nop(0, arch);
  const util::FailureOr<uint32_t> end = build_s_endpgm(arch);
  ASSERT_TRUE(transfer.succeeded());
  ASSERT_TRUE(nop.succeeded());
  ASSERT_TRUE(end.succeeded());
  std::vector<uint8_t> logger_image;
  std::vector<uint8_t> clobber_image;
  ASSERT_NO_FATAL_FAILURE(make_probe_images(LoggerOutput::Shared, logger_image, clobber_image));
  const AmdGpuCodeObject logger(logger_image.data(), logger_image.size());
  ASSERT_TRUE(logger.is_valid());
  for (bool enable_vskip : {false, true}) {
    SCOPED_TRACE(enable_vskip);
    std::vector<uint32_t> words = setup_;
    if (enable_vskip) {
      words.push_back(0xbf108081u);
    }
    words.insert(words.end(), {transfer.value(), 0xbf108080u});
    // The unresolved transfer can enter at this callback, bypassing the local
    // clear. Its decoded block alone therefore cannot prove the incoming state.
    const uint64_t anchor = words.size() * sizeof(uint32_t);
    words.insert(words.end(), {nop.value(), end.value()});
    std::vector<uint8_t> image;
    ASSERT_NO_FATAL_FAILURE(make_image(words, image));
    const std::vector<uint8_t> pristine = image;
    const AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    Instrumentor instrumentor(source, arch);
    InstrumentationPoint point = probe_at(anchor, logger);
    point.probe_args = {probe_arg_vgpr(1)};
    instrumentor.add_point(std::move(point));
    const Instrumentor::ValidationResult validated = instrumentor.validate_points();
    ASSERT_FALSE(validated.errors.empty());
    EXPECT_NE(validated.errors.front().find("probe observation may inherit enabled GPR indexing"),
              std::string::npos)
        << validated.errors.front();
    EXPECT_TRUE(validated.sites.empty());
    const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
    EXPECT_TRUE(result.elf_bytes.empty());
    EXPECT_TRUE(result.sites.empty());
    EXPECT_TRUE(result.patches.empty());
    ASSERT_FALSE(result.errors.empty());
    EXPECT_NE(result.errors.front().find("probe observation may inherit enabled GPR indexing"),
              std::string::npos)
        << result.errors.front();
    ASSERT_EQ(source.image_size(), pristine.size());
    EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
  }
}

TEST_P(DbiVskipReplacementSim, KernelEntryUsesDispatchStateDespiteAVskipEnabledBackedge) {
  std::vector<uint32_t> words = setup_;
  words.push_back(0xbf108081u); // s_setvskip 1, 0
  const int16_t backedge = -static_cast<int16_t>(words.size() + 1);
  // s_cbranch_execz entry: present in the CFG but not taken by this full wave.
  words.push_back(0xbf880000u | static_cast<uint16_t>(backedge));
  words.push_back(0xbf108080u);
  const util::FailureOr<uint32_t> end = build_s_endpgm(GetParam().arch);
  ASSERT_TRUE(end.succeeded());
  words.push_back(end.value());
  std::vector<uint8_t> image;
  ASSERT_NO_FATAL_FAILURE(make_image(words, image));
  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject source(image.data(), image.size());
  std::vector<uint8_t> logger_image;
  std::vector<uint8_t> clobber_image;
  ASSERT_NO_FATAL_FAILURE(make_probe_images(LoggerOutput::Shared, logger_image, clobber_image));
  const AmdGpuCodeObject probe(logger_image.data(), logger_image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_TRUE(probe.is_valid());
  Instrumentor instrumentor(source, GetParam().arch);
  InstrumentationPoint point = probe_at(0, probe);
  point.kind = InstrumentationKind::KernelEntry;
  point.probe_args = {probe_arg_imm(kProbeClobber)};
  instrumentor.add_point(std::move(point));
  const Instrumentor::ValidationResult validated = instrumentor.validate_points();
  ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
  const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  ASSERT_EQ(result.sites.size(), 1u);
  ASSERT_EQ(source.image_size(), pristine.size());
  EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
  const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(candidate.is_valid());
  const std::optional<uint64_t> entry = test::patched_entry_text_offset(candidate);
  ASSERT_TRUE(entry.has_value());
  ASSERT_NO_FATAL_FAILURE(expect_memory(test::section_words(candidate, ".text"),
                                        test::patched_private_segment_size(candidate),
                                        kProbeClobber, 1, *entry));
}

class DbiPreloadReplacementSim : public DbiReplacementSim {};

TEST_P(DbiPreloadReplacementSim, FirmwareEntryDoesNotInheritExecFromTheSkippedPrefix) {
  const ReplacementSimTarget &target = GetParam();
  const util::FailureOr<uint32_t> nop = build_s_nop(0, target.arch);
  const util::FailureOr<uint32_t> full_exec = build_s_mov_b64(126, 193, target.arch);
  const util::FailureOr<uint32_t> copy_lane = build_v_mov_b32_src(3, 4, target.arch);
  ASSERT_TRUE(nop.succeeded());
  ASSERT_TRUE(full_exec.succeeded());
  ASSERT_TRUE(copy_lane.succeeded());
  const std::array<uint32_t, 2> write_lane =
      target.arch == rocjitsu::IsaFamily::Cdna3
          ? cdna3::build_vop3(cdna3::kVWritelaneB32Vop3,
                              {.vdst = 1,
                               .src0 = scalar_positive_inline_u32(kReplacementInput),
                               .src1 = scalar_positive_inline_u32(63)})
          : cdna4::build_vop3(cdna4::kVWritelaneB32Vop3,
                              {.vdst = 1,
                               .src0 = scalar_positive_inline_u32(kReplacementInput),
                               .src1 = scalar_positive_inline_u32(63)});
  const std::array<uint32_t, 2> read_lane =
      target.arch == rocjitsu::IsaFamily::Cdna3
          ? cdna3::build_vop3(cdna3::kVReadlaneB32Vop3,
                              {.vdst = 4, .src0 = 257, .src1 = scalar_positive_inline_u32(63)})
          : cdna4::build_vop3(cdna4::kVReadlaneB32Vop3,
                              {.vdst = 4, .src0 = 257, .src1 = scalar_positive_inline_u32(63)});
  std::vector<uint8_t> logger_image;
  std::vector<uint8_t> clobber_image;
  ASSERT_NO_FATAL_FAILURE(make_probe_images(LoggerOutput::Shared, logger_image, clobber_image));
  const AmdGpuCodeObject clobber(clobber_image.data(), clobber_image.size());
  ASSERT_TRUE(clobber.is_valid());
  struct Case {
    bool preload;
    bool replace_prefix;
    uint32_t private_bytes;
  };
  constexpr std::array<Case, 4> cases = {
      Case{.preload = true, .replace_prefix = false, .private_bytes = kInitialPrivateBytes},
      Case{.preload = true, .replace_prefix = true, .private_bytes = kInitialPrivateBytes},
      Case{.preload = true, .replace_prefix = true, .private_bytes = 0},
      Case{.preload = false, .replace_prefix = false, .private_bytes = 0}};
  for (const Case &test_case : cases) {
    SCOPED_TRACE(test_case.preload);
    SCOPED_TRACE(test_case.replace_prefix);
    SCOPED_TRACE(test_case.private_bytes);
    std::vector<uint32_t> words(256 / sizeof(uint32_t), nop.value());
    words.front() = test_case.replace_prefix ? nop.value() : full_exec.value();
    words.insert(words.end(), setup_.begin(), setup_.end());
    // v_writelane defines lane 63 even when the one-work-item dispatch leaves it
    // inactive. Only the descriptor entry executes the full-EXEC prefix.
    words.insert(words.end(), write_lane.begin(), write_lane.end());
    const uint64_t anchor = words.size() * sizeof(uint32_t);
    words.push_back(nop.value());
    words.push_back(test::make_mov_vgpr_inline(1, kOriginalInput));
    words.insert(words.end(), read_lane.begin(), read_lane.end());
    words.push_back(copy_lane.value());
    ASSERT_NO_FATAL_FAILURE(finish_kernel(words));
    std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
        words, test_case.private_bytes, test::kCallableProbeGranulatedSgprCount, target.machine);
    if (test_case.preload) {
      const AmdGpuCodeObject original(image.data(), image.size());
      const Section &text = *original.text_sections().front();
      const std::vector<KernelDescriptorInfo> kernels =
          scan_kernel_descriptors(image, text.sectionOffset(), text.size());
      ASSERT_EQ(kernels.size(), 1u);
      amdhsa::kernel_descriptor_t descriptor = kernels.front().descriptor;
      descriptor.kernarg_size = sizeof(uint32_t);
      RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, amdhsa::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT,
                         1);
      RJ_AMDHSA_BITS_SET(descriptor.kernarg_preload, amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 1);
      std::memcpy(image.data() + kernels.front().descriptor_file_offset, &descriptor,
                  sizeof(descriptor));
    }
    image = test::with_descriptor_metadata(image, target.arch);
    ASSERT_FALSE(image.empty());
    const std::vector<uint8_t> pristine = image;
    const AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    Instrumentor instrumentor(source, target.arch);
    if (test_case.replace_prefix) {
      const std::array<uint32_t, 1> replacement_words = {full_exec.value()};
      instrumentor.add_point(replacement(0, nop.value(), replacement_words));
    }
    InstrumentationPoint point = probe_at(anchor, clobber);
    point.force_full_exec = true;
    instrumentor.add_point(std::move(point));
    const Instrumentor::ValidationResult validated = instrumentor.validate_points();
    ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
    const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
    ASSERT_EQ(source.image_size(), pristine.size());
    EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
    if (test_case.preload && test_case.private_bytes == 0) {
      EXPECT_TRUE(result.elf_bytes.empty());
      EXPECT_TRUE(result.sites.empty());
      EXPECT_TRUE(result.patches.empty());
      EXPECT_TRUE(std::ranges::any_of(result.errors, [](const std::string &error) {
        return error.find("fixed scratch to spill into") != std::string::npos;
      }));
      continue;
    }
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    EXPECT_EQ(result.sites.size(), test_case.replace_prefix ? 2u : 1u);
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    test::DbiSim simulator;
    // The descriptor remains Wave64. A one-work-item dispatch supplies partial
    // EXEC; selecting +256 models the preloading firmware's alternate entry.
    ASSERT_NO_FATAL_FAILURE(simulator.initialize(target.sim_arch, 1));
    simulator.set_entry_offset(test_case.preload ? 256 : 0);
    const std::optional<std::vector<uint8_t>> output = simulator.run_and_read_memory(
        test::section_words(candidate, ".text"), test::patched_private_segment_size(candidate),
        kOutputAddress, sizeof(uint32_t));
    ASSERT_TRUE(output) << "replacement kernel did not halt";
    ASSERT_EQ(output->size(), sizeof(uint32_t));
    uint32_t observed = 0;
    std::memcpy(&observed, output->data(), sizeof(observed));
    EXPECT_EQ(observed, test_case.preload ? kReplacementInput : kOriginalInput);
  }
}

TEST_P(DbiReplacementSim, InteriorDescriptorEntrySeparatesExecAndVskipProofs) {
  ObservationProgram program;
  ASSERT_NO_FATAL_FAILURE(make_observation_program(ObservationDefinition::Full, 0, program, true));
  const std::vector<uint8_t> pristine = program.image;
  const AmdGpuCodeObject source(program.image.data(), program.image.size());
  std::vector<uint8_t> logger_image;
  std::vector<uint8_t> clobber_image;
  ASSERT_NO_FATAL_FAILURE(make_probe_images(LoggerOutput::Shared, logger_image, clobber_image));
  const AmdGpuCodeObject logger(logger_image.data(), logger_image.size());
  const AmdGpuCodeObject clobber(clobber_image.data(), clobber_image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_TRUE(logger.is_valid());
  ASSERT_TRUE(clobber.is_valid());
  Instrumentor instrumentor(source, GetParam().arch);
  instrumentor.add_point(probe_at(program.early_offset, clobber));
  InstrumentationPoint observer = probe_at(program.observer_offset, logger);
  observer.probe_args = {probe_arg_vgpr(1)};
  instrumentor.add_point(std::move(observer));
  const Instrumentor::ValidationResult validated = instrumentor.validate_points();
  ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
  ASSERT_EQ(validated.sites.size(), 2u);
  const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
  // Descriptor entries are now block boundaries, so the full EXEC write after
  // this entry can discharge the observation's old value. CDNA still merges
  // unknown VSKIP from the decoded prefix and conservatively requires a spill.
  if (GetParam().arch == rocjitsu::IsaFamily::Rdna4) {
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    ASSERT_EQ(result.sites.size(), 2u);
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    const uint32_t scratch = test::patched_private_segment_size(candidate);
    EXPECT_EQ(scratch, 0u);
    ASSERT_NO_FATAL_FAILURE(expect_memory(test::section_words(candidate, ".text"), scratch,
                                          kReplacementInput, 1, program.early_offset));
  } else {
    EXPECT_TRUE(std::ranges::any_of(result.errors, [](const std::string &error) {
      return error.find("fixed scratch to spill into") != std::string::npos;
    }));
    EXPECT_TRUE(result.elf_bytes.empty());
    EXPECT_TRUE(result.sites.empty());
  }
  ASSERT_EQ(source.image_size(), pristine.size());
  EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
}

TEST_P(DbiReplacementSim, EmptyReplacementRemovesTheOriginalEffect) {
  std::vector<uint32_t> words = setup_;
  const uint64_t anchor = words.size() * sizeof(uint32_t);
  const uint32_t definition = test::make_mov_vgpr_inline(1, kOriginalInput);
  words.push_back(definition);
  words.push_back(copy_v3_v1_);
  ASSERT_NO_FATAL_FAILURE(finish_kernel(words));
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

class DbiVectorRelativeSim : public DbiReplacementSim {
protected:
  struct RelativeCase {
    std::vector<uint8_t> image;
    std::vector<uint8_t> probe;
    uint64_t anchor = 0;
    uint32_t relative = 0;
  };

  void make_relative_case(RelativeCase &value) const {
    const rocjitsu::IsaFamily arch = GetParam().arch;
    ASSERT_EQ(arch, rocjitsu::IsaFamily::Rdna4);
    const util::FailureOr<uint16_t> m0 = scalar_operand_m0(arch);
    ASSERT_TRUE(m0.succeeded());
    const util::FailureOr<uint32_t> set_m0 =
        build_s_mov_b32(m0.value(), scalar_positive_inline_u32(2), arch);
    const util::FailureOr<uint32_t> clear_v2 =
        build_v_mov_b32_src(2, scalar_positive_inline_u32(0), arch);
    const util::FailureOr<uint32_t> ret = build_s_setpc_b64(30, arch);
    ASSERT_TRUE(set_m0.succeeded());
    ASSERT_TRUE(clear_v2.succeeded());
    ASSERT_TRUE(ret.succeeded());
    value.relative = rdna4::build_vop1(rdna4::kVMovrelsB32Vop1, {.src0 = 256, .vdst = 3})[0];
    std::vector<uint32_t> words = setup_;
    words.push_back(set_m0.value());
    value.anchor = words.size() * sizeof(uint32_t);
    words.push_back(copy_v3_v1_);
    ASSERT_NO_FATAL_FAILURE(finish_kernel(words));
    make_image(words, value.image);
    value.probe =
        test::make_gfx1200_probe_elf("replacement_probe", {clear_v2.value(), ret.value()});
  }

  void expect_source_unchanged(const AmdGpuCodeObject &source,
                               const std::vector<uint8_t> &image) const {
    ASSERT_EQ(source.image_size(), image.size());
    EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
  }

  void expect_emitted_memory(const InstrumentedCodeObjectDebug &result, uint32_t expected) const {
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    expect_memory(test::section_words(candidate, ".text"),
                  test::patched_private_segment_size(candidate), expected);
  }
};

TEST_P(DbiVectorRelativeSim, PureRelativeAndDirectProbeControlsRemainExecutable) {
  RelativeCase value;
  ASSERT_NO_FATAL_FAILURE(make_relative_case(value));
  const AmdGpuCodeObject source(value.image.data(), value.image.size());
  const AmdGpuCodeObject probe(value.probe.data(), value.probe.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_TRUE(probe.is_valid());

  const std::array<uint32_t, 1> relative = {value.relative};
  Instrumentor pure(source, GetParam().arch);
  pure.add_point(replacement(value.anchor, copy_v3_v1_, relative));
  const InstrumentedCodeObjectDebug pure_result = pure.patch_with_debug_summaries();
  ASSERT_NO_FATAL_FAILURE(expect_emitted_memory(pure_result, 11));
  ASSERT_NO_FATAL_FAILURE(expect_source_unchanged(source, value.image));

  const std::array<uint32_t, 1> direct = {copy_v3_v2_};
  Instrumentor ordinary(source, GetParam().arch);
  InstrumentationPoint direct_point = replacement(value.anchor, copy_v3_v1_, direct);
  direct_point.probe_obj = &probe;
  direct_point.probe_symbol = "replacement_probe";
  ordinary.add_point(direct_point);
  const InstrumentedCodeObjectDebug direct_result = ordinary.patch_with_debug_summaries();
  ASSERT_NO_FATAL_FAILURE(expect_emitted_memory(direct_result, 11));
  ASSERT_NO_FATAL_FAILURE(expect_source_unchanged(source, value.image));
}

TEST_P(DbiVectorRelativeSim, CallableProbeRejectsReplacementIntroducingHiddenVgprRead) {
  RelativeCase value;
  ASSERT_NO_FATAL_FAILURE(make_relative_case(value));
  const AmdGpuCodeObject source(value.image.data(), value.image.size());
  const AmdGpuCodeObject probe(value.probe.data(), value.probe.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_TRUE(probe.is_valid());
  const std::array<uint32_t, 1> relative = {value.relative};
  for (int shape = 0; shape < 3; ++shape) {
    SCOPED_TRACE(shape);
    Instrumentor instrumentor(source, GetParam().arch);
    InstrumentationPoint mutation = replacement(value.anchor, copy_v3_v1_, relative);
    const InstrumentationPoint observation = probe_at(value.anchor, probe);
    if (shape == 0) {
      mutation.probe_obj = &probe;
      mutation.probe_symbol = "replacement_probe";
      instrumentor.add_point(mutation);
    } else {
      instrumentor.add_point(shape == 1 ? mutation : observation);
      instrumentor.add_point(shape == 1 ? observation : mutation);
    }
    const Instrumentor::ValidationResult validation = instrumentor.validate_points();
    EXPECT_FALSE(validation.errors.empty());
    EXPECT_TRUE(validation.sites.empty());
    const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
    EXPECT_FALSE(result.errors.empty());
    EXPECT_TRUE(result.elf_bytes.empty());
    EXPECT_TRUE(result.sites.empty());
    EXPECT_TRUE(result.patches.empty());
    if (!result.errors.empty()) {
      EXPECT_NE(result.errors.front().find("VGPR"), std::string::npos);
    }
    ASSERT_NO_FATAL_FAILURE(expect_source_unchanged(source, value.image));
  }
}

TEST_P(DbiReplacementSim, AfterObservationUsesTheValueDefinedByItsAnchor) {
  const ReplacementSimTarget &target = GetParam();
  const util::FailureOr<uint32_t> nop = build_s_nop(0, target.arch);
  const util::FailureOr<uint32_t> end = build_s_endpgm(target.arch);
  const util::FailureOr<uint16_t> exec = scalar_operand_exec_lo(target.arch);
  const util::FailureOr<uint16_t> all_ones = scalar_inline_neg_one(target.arch);
  ASSERT_TRUE(nop.succeeded() && end.succeeded() && exec.succeeded() && all_ones.succeeded());
  std::vector<uint8_t> logger_image;
  std::vector<uint8_t> clobber_image;
  ASSERT_NO_FATAL_FAILURE(make_probe_images(LoggerOutput::Shared, logger_image, clobber_image));
  const AmdGpuCodeObject logger(logger_image.data(), logger_image.size());
  const AmdGpuCodeObject clobber(clobber_image.data(), clobber_image.size());
  ASSERT_TRUE(logger.is_valid() && clobber.is_valid());
  struct Case {
    InstrumentationKind kind;
    bool partial_exec;
    uint32_t private_bytes;
    std::optional<uint32_t> expected;
  };
  const std::array<Case, 5> cases = {Case{.kind = InstrumentationKind::AfterInst,
                                          .partial_exec = false,
                                          .private_bytes = 0,
                                          .expected = kOriginalInput},
                                     Case{.kind = InstrumentationKind::BeforeInst,
                                          .partial_exec = false,
                                          .private_bytes = 0,
                                          .expected = std::nullopt},
                                     Case{.kind = InstrumentationKind::BeforeInst,
                                          .partial_exec = false,
                                          .private_bytes = kInitialPrivateBytes,
                                          .expected = kReplacementInput},
                                     Case{.kind = InstrumentationKind::AfterInst,
                                          .partial_exec = true,
                                          .private_bytes = 0,
                                          .expected = std::nullopt},
                                     Case{.kind = InstrumentationKind::AfterInst,
                                          .partial_exec = true,
                                          .private_bytes = kInitialPrivateBytes,
                                          .expected = kOriginalInput}};
  for (const Case &test_case : cases) {
    SCOPED_TRACE(static_cast<int>(test_case.kind));
    SCOPED_TRACE(test_case.partial_exec);
    SCOPED_TRACE(test_case.private_bytes);
    const util::FailureOr<uint32_t> set_exec = build_s_mov_b64(
        exec.value(), test_case.partial_exec ? scalar_positive_inline_u32(1) : all_ones.value(),
        target.arch);
    ASSERT_TRUE(set_exec.succeeded());
    std::vector<uint32_t> words = setup_;
    words.push_back(set_exec.value());
    const uint64_t early_offset = words.size() * sizeof(uint32_t);
    words.push_back(nop.value());
    const uint64_t definition_offset = words.size() * sizeof(uint32_t);
    words.push_back(test::make_mov_vgpr_inline(1, kOriginalInput));
    words.push_back(end.value());
    std::vector<uint8_t> image;
    ASSERT_NO_FATAL_FAILURE(make_image(words, image, test_case.private_bytes));
    const std::vector<uint8_t> pristine = image;
    const AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    Instrumentor instrumentor(source, target.arch);
    InstrumentationPoint early = probe_at(early_offset, clobber);
    early.force_full_exec = true;
    instrumentor.add_point(std::move(early));
    InstrumentationPoint observer = probe_at(definition_offset, logger);
    observer.kind = test_case.kind;
    observer.probe_args = {probe_arg_vgpr(1)};
    instrumentor.add_point(std::move(observer));
    const Instrumentor::ValidationResult validation = instrumentor.validate_points();
    ASSERT_TRUE(validation.errors.empty()) << validation.errors.front();
    const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
    ASSERT_EQ(source.image_size(), pristine.size());
    EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
    if (!test_case.expected) {
      EXPECT_TRUE(std::ranges::any_of(result.errors, [](const std::string &error) {
        return error.find("fixed scratch to spill into") != std::string::npos;
      }));
      EXPECT_TRUE(result.elf_bytes.empty());
      EXPECT_TRUE(result.sites.empty());
      EXPECT_TRUE(result.patches.empty());
      continue;
    }
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    ASSERT_EQ(result.sites.size(), 2u);
    const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(candidate.is_valid());
    const uint32_t scratch = test::patched_private_segment_size(candidate);
    if (test_case.private_bytes == 0) {
      EXPECT_EQ(scratch, 0u);
    }
    ASSERT_NO_FATAL_FAILURE(
        expect_memory(test::section_words(candidate, ".text"), scratch, *test_case.expected, 1));
  }
}

INSTANTIATE_TEST_SUITE_P(Gfx1200, DbiVectorRelativeSim, ::testing::Values(kRdna4));

INSTANTIATE_TEST_SUITE_P(Cdna, DbiPreloadReplacementSim, ::testing::Values(kCdna3, kCdna4),
                         [](const ::testing::TestParamInfo<ReplacementSimTarget> &info) {
                           return std::string(info.param.sim_arch);
                         });

INSTANTIATE_TEST_SUITE_P(Cdna, DbiVskipReplacementSim, ::testing::Values(kCdna3, kCdna4),
                         [](const ::testing::TestParamInfo<ReplacementSimTarget> &info) {
                           return std::string(info.param.sim_arch);
                         });

INSTANTIATE_TEST_SUITE_P(Targets, DbiReplacementSim, ::testing::Values(kCdna3, kCdna4, kRdna4),
                         [](const ::testing::TestParamInfo<ReplacementSimTarget> &info) {
                           return std::string(info.param.sim_arch);
                         });

} // namespace
} // namespace rocjitsu
