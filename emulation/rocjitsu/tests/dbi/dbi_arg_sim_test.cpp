// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbi_arg_sim_test.cpp
/// @brief Simulator end-to-end for DBI probe arguments on CDNA3 (gfx942,
/// wave64), CDNA4 (gfx950, wave64), and RDNA4 (gfx1200, wave32).
///
/// The static tests (tests/patch/instrumentor_test.cpp,
/// tests/patch/trampoline_builder_test.cpp) prove the patched ELF *contains* the
/// v_mov_b32 argument writes ahead of the call. This test proves both halves of
/// the contract hold at runtime: the probe reads the value the caller asked for,
/// and the guest's own use of that register is unharmed.
///
/// The second half is the one worth stating. Arguments arrive in VGPRs from v0,
/// which the instrumented kernel is also free to be using -- so the argument
/// write lands on a live register and the envelope has to spill it. A probe
/// argument is therefore not just "put a value in v0"; it is that plus the
/// save/restore that keeps the guest from noticing.
///
/// Kernel (entry at .text offset 0):
///   v_mov_b32 v0, K0  ; offset 0: guest value into v0
///   v_mov_b32 v1, v0  ; offset 4: ANCHOR -- reads v0 into v1 (v0 live here)
///   s_endpgm          ; offset 8
/// Probe: { v_mov_b32 v3, v0 ; s_setpc_b64 s[30:31] } -- publishes its first
/// argument in v3.
///
/// Read back per lane: v3 must be the argument the site passed, and v1 must be
/// K0, the value the guest put in v0 before the anchor.

#include "../amdhsa_metadata_test_util.h"
#include "../cdna5_sim_test_common.h"
#include "../dbi_test_util.h"
#include "dbi_sim.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/exec_state.h"
#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/smem_builders.h"
#include "rocjitsu/code/builders/vector_builders.h"
#include "rocjitsu/code/kernel_descriptor_abi.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

using test::kMovV1V0;
using test::kMovV3V0;

// The guest's value in v0, an inline constant (1..64) so the kernel's move is one
// word and the anchor lands at .text offset 4.
constexpr uint32_t kGuestValue = 7;
// The argument the site passes, distinct from kGuestValue so the two are
// telling apart in a register read. Deliberately a value no guest instruction
// here would produce, since the negative control finds the argument write by
// searching the patched text for it.
constexpr uint32_t kArgSentinel = 0xA5A5A5A5u;

// Per-arch knobs: the sim/config arch string, the code arch (for builders), the
// ELF machine flag, and the wavefront size.
struct ArgSimArch {
  const char *sim_arch;
  rj_code_arch_t arch;
  uint32_t e_flags;
  uint32_t wave_size;
};

inline constexpr ArgSimArch kCdna3ArgArch{"cdna3", ROCJITSU_CODE_ARCH_CDNA3,
                                          EF_AMDGPU_MACH_AMDGCN_GFX942, /*wave_size=*/64};
inline constexpr ArgSimArch kCdna4ArgArch{"cdna4", ROCJITSU_CODE_ARCH_CDNA4,
                                          EF_AMDGPU_MACH_AMDGCN_GFX950, /*wave_size=*/64};
inline constexpr ArgSimArch kRdna4ArgArch{"rdna4", ROCJITSU_CODE_ARCH_RDNA4,
                                          EF_AMDGPU_MACH_AMDGCN_GFX1200, /*wave_size=*/32};

// Patches a kernel so a probe taking one argument is called at an anchor where
// the argument register v0 is live, then runs it.
//
// VGPR budget, a patch-time constraint only: make_amdgpu_kernel_elf defaults to
// 8 unified VGPRs with the AGPR window at v4, so the kernel owns v0..v3 as
// ordinary VGPRs, and one argument in v0 plus the probe's v3 both fit. A fixture
// passing more than four argument dwords must raise granulated_vgpr_count, or
// the orchestrator's argument-ownership gate rejects the site in SetUp and the
// test fails for the wrong reason. Note that gate covers the *argument* VGPRs
// only -- nothing checks a probe body's own clobbers against the bound, so a
// probe scratch register at v4 or above is accepted here rather than rejected.
// Execution is unaffected either way: DbiSim dispatches its own descriptor with
// the full register file (see dbi_sim.h).
class DbiArgSimBase : public ::testing::Test {
protected:
  explicit DbiArgSimBase(const ArgSimArch &a) : a_(a) {}

  void SetUp() override {
    util::FailureOr<uint32_t> maybe_s_endpgm_3 = build_s_endpgm(a_.arch);
    ASSERT_TRUE(maybe_s_endpgm_3.succeeded());
    const uint32_t endpgm = maybe_s_endpgm_3.value();
    util::FailureOr<uint32_t> maybe_s_setpc_b64_2 = build_s_setpc_b64(/*s[30:31]=*/30, a_.arch);
    ASSERT_TRUE(maybe_s_setpc_b64_2.succeeded());
    const uint32_t setpc = maybe_s_setpc_b64_2.value();
    // v_mov v0, K0 ; v_mov v1, v0 (ANCHOR at offset 4, v0 live) ; s_endpgm.
    const std::vector<uint8_t> target = test::with_descriptor_metadata(
        test::make_amdgpu_kernel_elf({test::make_mov_vgpr_inline(0, kGuestValue), kMovV1V0, endpgm},
                                     /*private_bytes=*/64,
                                     /*granulated_sgpr_count=*/3, a_.e_flags),
        a_.arch);
    auto probe = test::make_amdgpu_probe_elf("rj_test_arg_probe", {kMovV3V0, setpc}, a_.e_flags);

    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    ASSERT_TRUE(obj.is_valid());
    ASSERT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, a_.arch);
    InstrumentationPoint pt;
    pt.anchor_offset = 4; // v_mov_b32 v1, v0 -> reads v0 (v0 live at the anchor).
    pt.probe_obj = &probe_obj;
    pt.probe_symbol = "rj_test_arg_probe";
    pt.probe_args.push_back(probe_arg_imm(kArgSentinel));
    instr.add_point(pt);

    auto result = instr.patch_with_debug_summaries();
    ASSERT_TRUE(result.errors.empty())
        << (result.errors.empty() ? std::string{} : result.errors.front());
    ASSERT_EQ(result.patches.size(), 1u);
    ASSERT_TRUE(result.patches[0].is_probe_call);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    patched_text_ = test::section_words(patched, ".text");
    ASSERT_FALSE(patched_text_.empty());
    patched_scratch_ = test::patched_private_segment_size(patched);
  }

  // Both halves of the contract, read out of one execution: the probe sees the
  // argument the site passed, and the guest's v0 survives being used to carry
  // it. One dispatch rather than two so the two assertions describe the same run.
  void expect_argument_arrives_and_guest_register_survives() {
    EXPECT_EQ(patched_scratch_, 68u)
        << "descriptor scratch must grow to spill the live argument VGPR";
    test::DbiSim sim;
    ASSERT_NO_FATAL_FAILURE(sim.initialize(a_.sim_arch, a_.wave_size));
    const std::vector<std::vector<uint32_t>> regs =
        sim.run_and_read_vgprs(patched_text_, patched_scratch_, {/*v3=*/3, /*v1=*/1});
    const std::vector<uint32_t> &v3 = regs[0];
    const std::vector<uint32_t> &v1 = regs[1];
    ASSERT_EQ(v3.size(), a_.wave_size) << "kernel did not run to completion (no dispatched wave)";
    ASSERT_EQ(v1.size(), a_.wave_size);

    for (uint32_t lane = 0; lane < a_.wave_size; ++lane) {
      EXPECT_EQ(v3[lane], kArgSentinel)
          << "lane " << lane << ": probe did not receive its argument";
      EXPECT_EQ(v1[lane], kGuestValue)
          << "lane " << lane << ": the guest's v0 did not survive carrying the argument";
    }
  }

  // Negative control: prove the argument write is what the probe reads, not
  // whatever the guest happened to leave in v0. Nop the two v_mov_b32 v0, literal
  // words and the probe should read the guest's value instead.
  void expect_without_the_argument_write_probe_reads_the_guest_value() {
    const auto maybe_arg_write = build_v_mov_b32_imm(0, kArgSentinel, a_.arch);
    ASSERT_TRUE(maybe_arg_write.succeeded());
    const auto arg_write = maybe_arg_write.value();
    util::FailureOr<uint32_t> maybe_s_nop_1 = build_s_nop(0, a_.arch);
    ASSERT_TRUE(maybe_s_nop_1.succeeded());
    const uint32_t nop = maybe_s_nop_1.value();

    std::vector<uint32_t> sabotaged = patched_text_;
    auto it = std::ranges::search(sabotaged, arg_write).begin();
    ASSERT_NE(it, sabotaged.end()) << "argument materialization not found in the patched text";
    for (size_t i = 0; i < arg_write.size(); ++i)
      *(it + static_cast<std::ptrdiff_t>(i)) = nop;

    test::DbiSim broken;
    ASSERT_NO_FATAL_FAILURE(broken.initialize(a_.sim_arch, a_.wave_size));
    const std::vector<uint32_t> v3 =
        broken.run_and_read_vgpr(sabotaged, patched_scratch_, /*reg=*/3);
    ASSERT_EQ(v3.size(), a_.wave_size);
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v3[lane], kGuestValue)
          << "lane " << lane << ": without the argument write the probe must read the guest's v0";
  }

  ArgSimArch a_;
  std::vector<uint32_t> patched_text_;
  uint32_t patched_scratch_ = 0;
};

class DbiCdna3ArgSimFixture : public DbiArgSimBase {
protected:
  DbiCdna3ArgSimFixture() : DbiArgSimBase(kCdna3ArgArch) {}
};
class DbiCdna4ArgSimFixture : public DbiArgSimBase {
protected:
  DbiCdna4ArgSimFixture() : DbiArgSimBase(kCdna4ArgArch) {}
};
class DbiRdna4ArgSimFixture : public DbiArgSimBase {
protected:
  DbiRdna4ArgSimFixture() : DbiArgSimBase(kRdna4ArgArch) {}
};

TEST_F(DbiCdna3ArgSimFixture, ArgumentArrivesAndGuestRegisterSurvives) {
  expect_argument_arrives_and_guest_register_survives();
}
TEST_F(DbiCdna3ArgSimFixture, WithoutTheArgumentWriteProbeReadsTheGuestValue) {
  expect_without_the_argument_write_probe_reads_the_guest_value();
}
TEST_F(DbiCdna4ArgSimFixture, ArgumentArrivesAndGuestRegisterSurvives) {
  expect_argument_arrives_and_guest_register_survives();
}
TEST_F(DbiCdna4ArgSimFixture, WithoutTheArgumentWriteProbeReadsTheGuestValue) {
  expect_without_the_argument_write_probe_reads_the_guest_value();
}
TEST_F(DbiRdna4ArgSimFixture, ArgumentArrivesAndGuestRegisterSurvives) {
  expect_argument_arrives_and_guest_register_survives();
}
TEST_F(DbiRdna4ArgSimFixture, WithoutTheArgumentWriteProbeReadsTheGuestValue) {
  expect_without_the_argument_write_probe_reads_the_guest_value();
}

class DbiGuestArgumentSim : public ::testing::TestWithParam<ArgSimArch> {};

TEST_P(DbiGuestArgumentSim, OverlappingArgumentsCaptureGuestValuesBeforeOverwritingThem) {
  const ArgSimArch &target = GetParam();
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(target.arch);
  const util::FailureOr<uint32_t> setpc = build_s_setpc_b64(30, target.arch);
  const util::FailureOr<uint32_t> nop = build_s_nop(0, target.arch);
  ASSERT_TRUE(endpgm.succeeded());
  ASSERT_TRUE(setpc.succeeded());
  ASSERT_TRUE(nop.succeeded());
  const util::FailureOr<uint32_t> guest_move = build_v_mov_b32_src(2, 256, target.arch);
  const util::FailureOr<uint32_t> probe_first = build_v_mov_b32_src(3, 256, target.arch);
  const util::FailureOr<uint32_t> probe_second = build_v_mov_b32_src(4, 257, target.arch);
  ASSERT_TRUE(guest_move.succeeded());
  ASSERT_TRUE(probe_first.succeeded());
  ASSERT_TRUE(probe_second.succeeded());
  const std::vector<uint8_t> image = test::with_descriptor_metadata(
      test::make_amdgpu_kernel_elf(
          {test::make_mov_vgpr_inline(0, 7), test::make_mov_vgpr_inline(1, 13), guest_move.value(),
           endpgm.value()},
          /*private_bytes=*/64, /*granulated_sgpr_count=*/3, target.e_flags,
          /*granulated_vgpr_count=*/3, /*accum_offset=*/1),
      target.arch);
  const std::vector<uint8_t> body = test::make_amdgpu_probe_elf(
      "guest_arguments", {probe_first.value(), probe_second.value(), setpc.value()},
      target.e_flags);
  AmdGpuCodeObject object(image.data(), image.size());
  AmdGpuCodeObject probe(body.data(), body.size());
  Instrumentor instrumentor(object, target.arch);
  InstrumentationPoint point;
  point.anchor_offset = 8;
  point.probe_obj = &probe;
  point.probe_symbol = "guest_arguments";
  point.probe_args = {probe_arg_vgpr(1), probe_arg_vgpr(0), probe_arg_imm(99)};
  instrumentor.add_point(point);
  const InstrumentedCodeObject result = instrumentor.patch();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  const std::vector<uint32_t> text = test::section_words(patched, ".text");
  const uint32_t scratch = test::patched_private_segment_size(patched);
  test::DbiSim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize(target.sim_arch, target.wave_size));
  const std::vector<std::vector<uint32_t>> registers =
      sim.run_and_read_vgprs(text, scratch, {3, 4, 0, 1, 2});
  const std::array<uint32_t, 5> expected{13, 7, 7, 13, 7};
  for (size_t index = 0; index < expected.size(); ++index) {
    ASSERT_EQ(registers[index].size(), target.wave_size);
    for (uint32_t lane = 0; lane < target.wave_size; ++lane)
      EXPECT_EQ(registers[index][lane], expected[index])
          << "register " << index << " lane " << lane;
  }
  // Removing the cycle's source capture must corrupt the observed argument.
  std::vector<uint32_t> broken = text;
  const util::FailureOr<uint32_t> capture_word = build_v_mov_b32_src(3, 257, target.arch);
  ASSERT_TRUE(capture_word.succeeded());
  const uint32_t capture = capture_word.value();
  const std::vector<uint32_t>::iterator found = std::find(broken.begin(), broken.end(), capture);
  ASSERT_NE(found, broken.end());
  *found = nop.value();
  test::DbiSim control;
  ASSERT_NO_FATAL_FAILURE(control.initialize(target.sim_arch, target.wave_size));
  const std::vector<uint32_t> observed = control.run_and_read_vgpr(broken, scratch, 3);
  ASSERT_EQ(observed.size(), target.wave_size);
  EXPECT_NE(observed.front(), 13u);
}

INSTANTIATE_TEST_SUITE_P(Targets, DbiGuestArgumentSim,
                         ::testing::Values(kCdna3ArgArch, kCdna4ArgArch, kRdna4ArgArch),
                         [](const ::testing::TestParamInfo<ArgSimArch> &info) {
                           return std::string(info.param.sim_arch);
                         });

TEST(DbiFixtureSetupTest, MalformedConfigRecordsFatalFailure) {
  EXPECT_FATAL_FAILURE(
      {
        test::DbiSim sim;
        sim.initialize("\"", 64);
      },
      "Failed to parse JSON config:");
}

TEST(DbiAfterInstSim, ProbeObservesTheResultAndPreservesLiveGuestDefinitions) {
  constexpr uint32_t kOldValue = 3;
  constexpr uint32_t kAnchorResult = 7;
  constexpr uint32_t kCallbackClobber = 11;
  for (const ArgSimArch arch : {kCdna3ArgArch, kCdna4ArgArch, kRdna4ArgArch}) {
    SCOPED_TRACE(arch.sim_arch);
    const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch.arch);
    const util::FailureOr<uint32_t> setpc = build_s_setpc_b64(30, arch.arch);
    ASSERT_TRUE(endpgm.succeeded());
    ASSERT_TRUE(setpc.succeeded());
    for (const InstrumentationKind kind :
         {InstrumentationKind::BeforeInst, InstrumentationKind::AfterInst}) {
      const util::FailureOr<uint32_t> guest_move = build_v_mov_b32_src(2, 257, arch.arch);
      ASSERT_TRUE(guest_move.succeeded());
      const std::vector<uint32_t> guest = {test::make_mov_vgpr_inline(1, kOldValue),
                                           test::make_mov_vgpr_inline(0, kAnchorResult), kMovV1V0,
                                           guest_move.value(), endpgm.value()};
      const bool wave32 = arch.arch == ROCJITSU_CODE_ARCH_RDNA4;
      const std::vector<uint8_t> image =
          wave32 ? test::make_gfx1200_wave32_kernel_elf(guest, 64, 3)
                 : test::make_amdgpu_kernel_elf(guest, 64, 3, arch.e_flags);
      const std::vector<uint8_t> target = test::with_descriptor_metadata(image, arch.arch);
      const std::vector<uint8_t> probe = test::make_amdgpu_probe_elf(
          "after_result",
          {kMovV3V0, test::make_mov_vgpr_inline(1, kCallbackClobber), setpc.value()}, arch.e_flags);
      const AmdGpuCodeObject object(target.data(), target.size());
      const AmdGpuCodeObject body(probe.data(), probe.size());
      Instrumentor instrumentor(object, arch.arch);
      const InstrumentationPoint point = {.anchor_offset = 8,
                                          .kind = kind,
                                          .probe_obj = &body,
                                          .probe_symbol = "after_result",
                                          .probe_args = {probe_arg_vgpr(1)}};
      instrumentor.add_point(point);
      const InstrumentedCodeObject result = instrumentor.patch();
      ASSERT_TRUE(result.errors.empty()) << result.errors.front();
      ASSERT_EQ(result.sites.size(), 1u);
      const InstrumentedSite &site = result.sites.front();
      const AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
      const std::vector<uint32_t> text = test::section_words(patched, ".text");
      EXPECT_GT(site.instrumentation_size, 0u);
      if (kind == InstrumentationKind::AfterInst)
        EXPECT_EQ(site.instrumentation_offset, site.guest_offset + site.guest_size);
      else
        EXPECT_EQ(site.guest_offset, site.instrumentation_offset + site.instrumentation_size);
      const uint64_t resume_offset = kind == InstrumentationKind::AfterInst
                                         ? site.instrumentation_offset + site.instrumentation_size
                                         : site.guest_offset + site.guest_size;
      ASSERT_EQ(resume_offset + sizeof(uint32_t), text.size() * sizeof(uint32_t));
      const std::optional<int16_t> resume_delta =
          compute_sopp_branch_simm16(resume_offset, point.anchor_offset + site.guest_size);
      ASSERT_TRUE(resume_delta);
      const util::FailureOr<uint32_t> resume = build_s_branch(*resume_delta, arch.arch);
      ASSERT_TRUE(resume.succeeded());
      EXPECT_EQ(text.back(), resume.value());
      test::DbiSim sim;
      ASSERT_NO_FATAL_FAILURE(sim.initialize(arch.sim_arch, arch.wave_size));
      if (wave32)
        sim.set_wave32_dispatch();
      const std::vector<std::vector<uint32_t>> values =
          sim.run_and_read_vgprs(text, test::patched_private_segment_size(patched), {3, 2});
      ASSERT_EQ(values.size(), 2u);
      ASSERT_EQ(values[0].size(), arch.wave_size);
      ASSERT_EQ(values[1].size(), arch.wave_size);
      for (uint32_t lane = 0; lane < arch.wave_size; ++lane) {
        EXPECT_EQ(values[0][lane],
                  kind == InstrumentationKind::AfterInst ? kAnchorResult : kOldValue);
        EXPECT_EQ(values[1][lane], kAnchorResult);
      }
    }
  }
}

TEST(DbiAfterInstSim, RejectsGprIndexEnableWhileBeforeInstPreservesTheDisplacedRegister) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr util::FailureOr<uint32_t> nop = build_s_nop(0, kArch);
  static_assert(nop.succeeded());
  constexpr util::FailureOr<uint32_t> endpgm = build_s_endpgm(kArch);
  constexpr util::FailureOr<uint32_t> setpc = build_s_setpc_b64(30, kArch);
  static_assert(endpgm.succeeded() && setpc.succeeded());
  constexpr util::FailureOr<uint32_t> index_displacement =
      build_s_mov_b32(5, scalar_positive_inline_u32(4), kArch);
  static_assert(index_displacement.succeeded());
  constexpr uint32_t kDisplacedV7Value = 9;
  constexpr uint32_t kProbeValue = 5;
  const uint32_t enable = cdna4::build_sopc(cdna4::kSSetGprIdxOnSopc, {.ssrc0 = 5, .ssrc1 = 8})[0];
  const uint32_t disable = cdna4::build_sopp(cdna4::kSSetGprIdxOffSopp)[0];
  const util::FailureOr<uint32_t> readback = build_v_mov_b32_src(2, 256 + 7, kArch);
  ASSERT_TRUE(readback.succeeded());
  const std::vector<uint8_t> image = test::with_descriptor_metadata(
      test::make_gfx950_kernel_elf({test::make_mov_vgpr_inline(7, kDisplacedV7Value),
                                    index_displacement.value(), enable, nop.value(), disable,
                                    readback.value(), endpgm.value()},
                                   /*private_bytes=*/64, /*granulated_sgpr_count=*/3,
                                   /*granulated_vgpr_count=*/3, /*accum_offset=*/1),
      kArch);
  const std::vector<uint8_t> probe = test::make_gfx950_probe_elf(
      "index_clobber", {test::make_mov_vgpr_inline(3, kProbeValue), setpc.value()});
  const AmdGpuCodeObject object(image.data(), image.size());
  const AmdGpuCodeObject body(probe.data(), probe.size());
  for (const uint64_t anchor_offset : {8u, 12u}) {
    SCOPED_TRACE(anchor_offset);
    for (const InstrumentationKind kind :
         {InstrumentationKind::BeforeInst, InstrumentationKind::AfterInst}) {
      Instrumentor instrumentor(object, kArch);
      const InstrumentationPoint point = {.anchor_offset = anchor_offset,
                                          .kind = kind,
                                          .probe_obj = &body,
                                          .probe_symbol = "index_clobber",
                                          .probe_args = {}};
      instrumentor.add_point(point);
      const InstrumentedCodeObject result = instrumentor.patch();
      if (kind == InstrumentationKind::AfterInst || anchor_offset == 12) {
        ASSERT_EQ(result.errors.size(), 1u);
        EXPECT_NE(result.errors.front().find("GPR indexing"), std::string::npos);
        EXPECT_TRUE(result.elf_bytes.empty());
        EXPECT_TRUE(result.sites.empty());
        continue;
      }
      ASSERT_TRUE(result.errors.empty()) << result.errors.front();
      const AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
      test::DbiSim sim;
      ASSERT_NO_FATAL_FAILURE(sim.initialize("cdna4", 64));
      const std::vector<uint32_t> values = sim.run_and_read_vgpr(
          test::section_words(patched, ".text"), test::patched_private_segment_size(patched), 2);
      ASSERT_EQ(values.size(), 64u);
      for (const uint32_t value : values)
        EXPECT_EQ(value, kDisplacedV7Value);
    }
  }
}

class BarrierMarkerObserver final : public ExecutionPlugin {
public:
  BarrierMarkerObserver() : ExecutionPlugin("dbi_barrier_marker") {}

  void onAmdgpuAfterExecuteInstruction(uint64_t, const Instruction &inst,
                                       amdgpu::Wavefront &wave) override {
    if (inst.mnemonic() == "s_barrier_wait" && wave.state() == amdgpu::WfState::BARRIER)
      ++blocked_waits;
  }

  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> waves, BarrierKind kind) override {
    EXPECT_EQ(kind, BarrierKind::Workgroup);
    std::vector<uint32_t> markers;
    for (const amdgpu::Wavefront *wave : waves)
      markers.push_back(wave->debug_read_vgpr(3, 0));
    resolutions.push_back(std::move(markers));
  }

  uint32_t blocked_waits = 0;
  std::vector<std::vector<uint32_t>> resolutions;
};

TEST(DbiAfterInstSim, TwoWaveBarrierWaitObservesCompletionAfterTheRelocatedWaitReturns) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr uint32_t kMarker = 5;
  constexpr uint16_t kDelayInstructions = 32;
  constexpr uint32_t kSignal = cdna5::build_sop1(cdna5::kSBarrierSignalSop1, {.ssrc0 = 193})[0];
  constexpr uint32_t kWait = cdna5::build_sopp(cdna5::kSBarrierWaitSopp, {.simm16 = 0xffff})[0];
  constexpr util::FailureOr<uint32_t> setpc = build_s_setpc_b64(30, kArch);
  static_assert(setpc.succeeded());
  std::vector<uint32_t> guest = {
      test::make_mov_vgpr_inline(3, 0),
      cdna5::build_vop1(cdna5::kVReadfirstlaneB32Vop1, {.src0 = 256, .vdst = 4})[0],
      cdna5::build_sopc(cdna5::kSCmpEqU32Sopc, {.ssrc0 = 4, .ssrc1 = 128})[0],
      cdna5::build_sopp(cdna5::kSCbranchScc1Sopp, {.simm16 = kDelayInstructions})[0]};
  guest.insert(guest.end(), kDelayInstructions, cdna5::build_sopp(cdna5::kSNopSopp)[0]);
  guest.push_back(kSignal);
  const uint64_t wait_offset = guest.size() * sizeof(uint32_t);
  guest.push_back(kWait);
  guest.push_back(cdna5::build_sopp(cdna5::kSEndpgmSopp)[0]);
  const std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
      guest, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1250, 3, 0, false, false, false, false, true);
  const std::vector<uint8_t> body = test::make_amdgpu_probe_elf(
      "barrier_marker", {test::make_mov_vgpr_inline(3, kMarker), setpc.value()},
      EF_AMDGPU_MACH_AMDGCN_GFX1250);
  const AmdGpuCodeObject object(image.data(), image.size());
  const AmdGpuCodeObject probe(body.data(), body.size());
  for (const InstrumentationKind kind :
       {InstrumentationKind::BeforeInst, InstrumentationKind::AfterInst}) {
    SCOPED_TRACE(static_cast<int>(kind));
    Instrumentor instrumentor(object, kArch);
    const InstrumentationPoint point = {.anchor_offset = wait_offset,
                                        .kind = kind,
                                        .probe_obj = &probe,
                                        .probe_symbol = "barrier_marker",
                                        .probe_args = {}};
    instrumentor.add_point(point);
    const InstrumentedCodeObject result = instrumentor.patch();
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    ASSERT_EQ(result.sites.size(), 1u);
    const AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    EXPECT_EQ(test::patched_private_segment_size(patched), 0u);
    const std::vector<uint32_t> text = test::section_words(patched, ".text");
    const InstrumentedSite &site = result.sites.front();
    ASSERT_LT(site.guest_offset / sizeof(uint32_t), text.size());
    EXPECT_EQ(site.guest_size, sizeof(uint32_t));
    EXPECT_EQ(text[site.guest_offset / sizeof(uint32_t)], kWait);
    test::cdna5::Gfx1250Sim sim;
    ASSERT_NO_FATAL_FAILURE(sim.initialize());
    auto observer = std::make_unique<BarrierMarkerObserver>();
    BarrierMarkerObserver *record = observer.get();
    ASSERT_TRUE(sim.plugin_group->add(std::move(observer)));
    sim.soc->set_plugin_group(sim.plugin_group);
    uint32_t properties = 0;
    RJ_AMDHSA_BITS_SET(properties, amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 1);
    const uint64_t kernel = sim.write_kernel(0x10000, text.data(), text.size(), 104, 64, 0, false,
                                             false, false, properties);
    test::AqlQueue queue(sim.memory, sim.cp());
    queue.dispatch(kernel, 64, 64);
    test::cdna5::step_until_xcd_halted(sim);
    ASSERT_EQ(record->resolutions.size(), 1u);
    ASSERT_EQ(record->resolutions.front().size(), 2u);
    EXPECT_GE(record->blocked_waits, 1u);
    EXPECT_EQ(std::ranges::count(record->resolutions.front(), kMarker),
              kind == InstrumentationKind::BeforeInst ? 1 : 0);
    ASSERT_EQ(sim.snapshot->snapshots().size(), 2u);
    for (const test::WavefrontSnapshot &wave : sim.snapshot->snapshots())
      EXPECT_EQ(wave.vgpr(3, 0), kMarker);
  }
}

TEST(DbiAfterInstSim, VskipEnableRejectsAfterInstWhileBeforeInstPreservesScalarState) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr util::FailureOr<uint32_t> result =
      build_s_mov_b32(4, scalar_positive_inline_u32(7), kArch);
  constexpr util::FailureOr<uint32_t> clobber =
      build_s_mov_b32(4, scalar_positive_inline_u32(11), kArch);
  constexpr util::FailureOr<uint32_t> setpc = build_s_setpc_b64(30, kArch);
  static_assert(result.succeeded() && clobber.succeeded() && setpc.succeeded());
  const util::FailureOr<uint32_t> readback = build_v_mov_b32_src(2, 4, kArch);
  ASSERT_TRUE(readback.succeeded());
  const std::vector<uint8_t> image = test::with_descriptor_metadata(
      test::make_gfx950_kernel_elf(
          {result.value(),
           cdna4::build_sopc(cdna4::kSSetvskipSopc, {.ssrc0 = 129, .ssrc1 = 128})[0],
           cdna4::build_sopc(cdna4::kSSetvskipSopc, {.ssrc0 = 128, .ssrc1 = 128})[0],
           readback.value(), cdna4::build_sopp(cdna4::kSEndpgmSopp)[0]},
          64),
      kArch);
  const std::vector<uint8_t> body =
      test::make_gfx950_probe_elf("vskip", {clobber.value(), setpc.value()});
  const AmdGpuCodeObject object(image.data(), image.size());
  const AmdGpuCodeObject probe(body.data(), body.size());
  for (const InstrumentationKind kind :
       {InstrumentationKind::BeforeInst, InstrumentationKind::AfterInst}) {
    Instrumentor instrumentor(object, kArch);
    const InstrumentationPoint point = {.anchor_offset = 4,
                                        .kind = kind,
                                        .probe_obj = &probe,
                                        .probe_symbol = "vskip",
                                        .probe_args = {},
                                        .force_full_exec = true};
    instrumentor.add_point(point);
    const InstrumentedCodeObject patched = instrumentor.patch();
    if (kind == InstrumentationKind::AfterInst) {
      ASSERT_EQ(patched.errors.size(), 1u);
      EXPECT_NE(patched.errors.front().find("VSKIP"), std::string::npos);
      continue;
    }
    ASSERT_TRUE(patched.errors.empty()) << patched.errors.front();
    const AmdGpuCodeObject executable(patched.elf_bytes.data(), patched.elf_bytes.size());
    test::DbiSim sim;
    ASSERT_NO_FATAL_FAILURE(sim.initialize("cdna4", 64));
    const std::vector<uint32_t> values =
        sim.run_and_read_vgpr(test::section_words(executable, ".text"),
                              test::patched_private_segment_size(executable), 2);
    ASSERT_EQ(values.size(), 64u);
    for (const uint32_t value : values)
      EXPECT_EQ(value, 7u);
  }
}

TEST(DbiAfterInstSim, UnknownIncomingVskipRejectsInstructionAndExitObservations) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint64_t kObservationOffset = 36;
  constexpr uint64_t kExitOffset = 48;
  constexpr uint32_t kGuestScalar = 7;
  constexpr uint32_t kProbeMarker = 5;
  const util::FailureOr<std::array<uint32_t, 2>> load = build_s_load_dwordx2(4, 0, 0, kArch);
  const util::FailureOr<uint32_t> guest_scalar =
      build_s_mov_b32(8, scalar_positive_inline_u32(kGuestScalar), kArch);
  const util::FailureOr<uint32_t> clobber =
      build_s_mov_b32(8, scalar_positive_inline_u32(11), kArch);
  const util::FailureOr<uint32_t> jump = build_s_setpc_b64(4, kArch);
  const util::FailureOr<uint32_t> direct_jump = build_s_branch(0, kArch);
  const util::FailureOr<uint32_t> readback = build_v_mov_b32_src(2, 8, kArch);
  const util::FailureOr<uint32_t> ret = build_s_setpc_b64(30, kArch);
  ASSERT_TRUE(load.succeeded() && guest_scalar.succeeded() && clobber.succeeded() &&
              jump.succeeded() && direct_jump.succeeded() && readback.succeeded() &&
              ret.succeeded());
  const uint32_t clear = cdna4::build_sopc(cdna4::kSSetvskipSopc, {.ssrc0 = 128, .ssrc1 = 128})[0];
  const std::vector<uint8_t> body =
      test::make_gfx950_probe_elf("incoming_vskip", {test::make_mov_vgpr_inline(3, kProbeMarker),
                                                     clobber.value(), ret.value()});
  const AmdGpuCodeObject probe(body.data(), body.size());
  // DbiSim places the code immediately after its descriptor at address 0x1000.
  // Loading the destination from kernargs keeps it unknown to static discovery.
  const uint64_t target = 0x1000 + sizeof(amdhsa::kernel_descriptor_t) + kObservationOffset;
  std::vector<uint8_t> kernarg(sizeof(target));
  std::memcpy(kernarg.data(), &target, sizeof(target));
  for (const bool enables_vskip : {false, true}) {
    SCOPED_TRACE(enables_vskip);
    const std::vector<uint32_t> guest = {
        guest_scalar.value(), test::make_mov_vgpr_inline(3, 0), load.value()[0], load.value()[1],
        cdna4::build_sopp(cdna4::kSWaitcntSopp, {.simm16 = 0})[0],
        cdna4::build_sopc(cdna4::kSCmpEqU32Sopc, {.ssrc0 = 128, .ssrc1 = 129})[0],
        // The visible safe edge reaches the observation, but SCC=0 takes the
        // other path at runtime: set VSKIP, then jump through the loaded target.
        cdna4::build_sopp(cdna4::kSCbranchScc1Sopp, {.simm16 = 2})[0],
        enables_vskip ? cdna4::build_sopc(cdna4::kSSetvskipSopc, {.ssrc0 = 129, .ssrc1 = 128})[0]
                      : clear,
        enables_vskip ? jump.value() : direct_jump.value(), cdna4::build_sopp(cdna4::kSNopSopp)[0],
        clear, readback.value(), cdna4::build_sopp(cdna4::kSEndpgmSopp)[0]};
    test::DbiSim baseline;
    ASSERT_NO_FATAL_FAILURE(baseline.initialize("cdna4", 64));
    baseline.set_kernarg(kernarg);
    const std::vector<std::vector<uint32_t>> original =
        baseline.run_and_read_vgprs(guest, 64, {2, 3});
    ASSERT_EQ(original.size(), 2u);
    EXPECT_EQ(original[0], std::vector<uint32_t>(64, kGuestScalar));
    EXPECT_EQ(original[1], std::vector<uint32_t>(64, 0));

    const std::vector<uint8_t> image =
        test::with_descriptor_metadata(test::make_gfx950_kernel_elf(guest, 64), kArch);
    const AmdGpuCodeObject object(image.data(), image.size());
    for (const InstrumentationKind kind :
         {InstrumentationKind::BeforeInst, InstrumentationKind::AfterInst,
          InstrumentationKind::KernelExit}) {
      SCOPED_TRACE(static_cast<int>(kind));
      // The clear-only execution control uses a resolved direct edge so it
      // does not require a separate proof for arbitrary indirect destinations.
      if (!enables_vskip && kind != InstrumentationKind::BeforeInst)
        continue;
      Instrumentor instrumentor(object, kArch);
      instrumentor.add_point({.anchor_offset = kind == InstrumentationKind::KernelExit
                                                   ? kExitOffset
                                                   : kObservationOffset,
                              .kind = kind,
                              .probe_obj = &probe,
                              .probe_symbol = "incoming_vskip",
                              .probe_args = {}});
      const InstrumentedCodeObject result = instrumentor.patch();
      if (enables_vskip) {
        EXPECT_EQ(result.errors.size(), 1u);
        if (!result.errors.empty()) {
          EXPECT_NE(result.errors.front().find("may inherit enabled"), std::string::npos)
              << result.errors.front();
        }
        EXPECT_TRUE(result.elf_bytes.empty());
        EXPECT_TRUE(result.sites.empty());
        continue;
      }
      ASSERT_TRUE(result.errors.empty()) << result.errors.front();
      const AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
      EXPECT_GT(test::patched_private_segment_size(patched), 64u)
          << "the live probe-clobbered s8 must exercise the scalar spill restore";
      test::DbiSim sim;
      ASSERT_NO_FATAL_FAILURE(sim.initialize("cdna4", 64));
      sim.set_kernarg(kernarg);
      const std::vector<std::vector<uint32_t>> values =
          sim.run_and_read_vgprs(test::section_words(patched, ".text"),
                                 test::patched_private_segment_size(patched), {2, 3});
      ASSERT_EQ(values.size(), 2u);
      EXPECT_EQ(values[0], std::vector<uint32_t>(64, kGuestScalar));
      EXPECT_EQ(values[1], std::vector<uint32_t>(64, kProbeMarker));
    }
  }
}

TEST(DbiAfterInstSim, RejectsVskipEnablingProbeAndPreservesScalarStateWithClearOnlyProbe) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr util::FailureOr<uint32_t> result =
      build_s_mov_b32(4, scalar_positive_inline_u32(7), kArch);
  constexpr util::FailureOr<uint32_t> clobber =
      build_s_mov_b32(4, scalar_positive_inline_u32(11), kArch);
  constexpr util::FailureOr<uint32_t> setpc = build_s_setpc_b64(30, kArch);
  static_assert(result.succeeded() && clobber.succeeded() && setpc.succeeded());
  const uint32_t clear = cdna4::build_sopc(cdna4::kSSetvskipSopc, {.ssrc0 = 128, .ssrc1 = 128})[0];
  const util::FailureOr<uint32_t> readback = build_v_mov_b32_src(2, 4, kArch);
  ASSERT_TRUE(readback.succeeded());
  const std::vector<uint8_t> image = test::with_descriptor_metadata(
      test::make_gfx950_kernel_elf(
          {result.value(), clear, readback.value(), cdna4::build_sopp(cdna4::kSEndpgmSopp)[0]}, 64),
      kArch);
  const AmdGpuCodeObject object(image.data(), image.size());
  for (const bool enables : {false, true}) {
    SCOPED_TRACE(enables);
    const uint32_t write_vskip = cdna4::build_sopc(
        cdna4::kSSetvskipSopc, {.ssrc0 = static_cast<uint8_t>(128 + enables), .ssrc1 = 128})[0];
    const std::vector<uint8_t> body = test::make_gfx950_probe_elf(
        "vskip_result_clobber", {clobber.value(), write_vskip, setpc.value()});
    const AmdGpuCodeObject probe(body.data(), body.size());
    Instrumentor instrumentor(object, kArch);
    const InstrumentationPoint point = {.kind = InstrumentationKind::AfterInst,
                                        .probe_obj = &probe,
                                        .probe_symbol = "vskip_result_clobber",
                                        .probe_args = {},
                                        .force_full_exec = true};
    instrumentor.add_point(point);
    const InstrumentedCodeObject patched = instrumentor.patch();
    if (enables && !patched.errors.empty()) {
      ASSERT_EQ(patched.errors.size(), 1u);
      EXPECT_NE(patched.errors.front().find("VSKIP"), std::string::npos);
      EXPECT_TRUE(patched.elf_bytes.empty());
      EXPECT_TRUE(patched.sites.empty());
      continue;
    }
    ASSERT_TRUE(patched.errors.empty()) << patched.errors.front();
    const AmdGpuCodeObject executable(patched.elf_bytes.data(), patched.elf_bytes.size());
    test::DbiSim sim;
    ASSERT_NO_FATAL_FAILURE(sim.initialize("cdna4", 64));
    const std::vector<uint32_t> values =
        sim.run_and_read_vgpr(test::section_words(executable, ".text"),
                              test::patched_private_segment_size(executable), 2);
    // If admission regresses, execute the unsafe image too: the guest clears
    // VSKIP before reading s4, exposing an SGPR restore suppressed by the probe.
    EXPECT_EQ(values, std::vector<uint32_t>(64, 7));
    EXPECT_FALSE(enables) << "a probe that enables VSKIP must be rejected before emission";
  }
}

TEST(DbiAfterInstSim, PreservesANewScalarDefinitionWithoutArguments) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint16_t kResultSgpr = 4;
  constexpr util::FailureOr<uint32_t> endpgm = build_s_endpgm(kArch);
  static_assert(endpgm.succeeded());
  constexpr util::FailureOr<uint32_t> setpc = build_s_setpc_b64(30, kArch);
  static_assert(setpc.succeeded());
  constexpr util::FailureOr<uint32_t> guest_result =
      build_s_mov_b32(kResultSgpr, scalar_positive_inline_u32(7), kArch);
  static_assert(guest_result.succeeded());
  constexpr util::FailureOr<uint32_t> clobber =
      build_s_mov_b32(kResultSgpr, scalar_positive_inline_u32(11), kArch);
  static_assert(clobber.succeeded());
  const util::FailureOr<uint32_t> readback = build_v_mov_b32_src(2, kResultSgpr, kArch);
  ASSERT_TRUE(readback.succeeded());
  const std::vector<uint8_t> image = test::with_descriptor_metadata(
      test::make_gfx950_kernel_elf({guest_result.value(), readback.value(), endpgm.value()}, 64),
      kArch);
  const std::vector<uint8_t> probe =
      test::make_gfx950_probe_elf("clobber_result", {clobber.value(), setpc.value()});
  const AmdGpuCodeObject object(image.data(), image.size());
  const AmdGpuCodeObject body(probe.data(), probe.size());

  // A full-width scalar definition kills the old value without relying on
  // EXEC analysis. With no arguments, only AfterInst's definitions union can
  // make this new result live for preservation around the callback.
  const std::unique_ptr<Decoder> decoder = Decoder::create(kArch);
  ASSERT_NE(decoder, nullptr);
  const FailureOr<std::vector<std::unique_ptr<BasicBlock>>> blocks =
      BasicBlock::build(object, *decoder, kArch);
  ASSERT_TRUE(blocks.succeeded());
  ASSERT_EQ(blocks.value().size(), 1u);
  const std::array<BasicBlock *, 1> scope = {blocks.value().front().get()};
  const LivenessAnalysis liveness(scope, 64);
  const Instruction &anchor = *scope.front()->instructions().begin();
  const FailureOr<bool> live_before =
      liveness.is_live_before(anchor, {RegClass::SGPR, kResultSgpr, 1});
  ASSERT_TRUE(live_before.succeeded());
  ASSERT_FALSE(live_before.value());

  Instrumentor instrumentor(object, kArch);
  const InstrumentationPoint point = {.kind = InstrumentationKind::AfterInst,
                                      .probe_obj = &body,
                                      .probe_symbol = "clobber_result",
                                      .probe_args = {}};
  instrumentor.add_point(point);
  const InstrumentedCodeObject result = instrumentor.patch();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  const AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  test::DbiSim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize("cdna4", 64));
  const std::vector<uint32_t> values = sim.run_and_read_vgpr(
      test::section_words(patched, ".text"), test::patched_private_segment_size(patched), 2);
  ASSERT_EQ(values.size(), 64u);
  for (const uint32_t value : values)
    EXPECT_EQ(value, 7u);
}

TEST(DbiAfterInstSim, ObservesAndRestoresTheExecMaskProducedByTheAnchor) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr util::FailureOr<uint16_t> exec_lo = scalar_operand_exec_lo(kArch);
  static_assert(exec_lo.succeeded());
  constexpr uint16_t kExecLo = exec_lo.value();
  constexpr util::FailureOr<uint32_t> endpgm = build_s_endpgm(kArch);
  static_assert(endpgm.succeeded());
  constexpr util::FailureOr<uint32_t> setpc = build_s_setpc_b64(30, kArch);
  static_assert(setpc.succeeded());
  constexpr util::FailureOr<uint32_t> guest_mask =
      build_s_mov_b64(kExecLo, scalar_positive_inline_u32(3), kArch);
  static_assert(guest_mask.succeeded());
  constexpr util::FailureOr<uint32_t> clobber_mask =
      build_s_mov_b64(kExecLo, scalar_positive_inline_u32(0), kArch);
  static_assert(clobber_mask.succeeded());
  const std::vector<uint8_t> image = test::with_descriptor_metadata(
      test::make_gfx950_kernel_elf({test::make_mov_vgpr_inline(2, 0), guest_mask.value(),
                                    test::make_mov_vgpr_inline(2, 7), endpgm.value()},
                                   64),
      kArch);
  const std::vector<uint8_t> probe =
      test::make_gfx950_probe_elf("observe_mask", {kMovV3V0, clobber_mask.value(), setpc.value()});
  const AmdGpuCodeObject object(image.data(), image.size());
  const AmdGpuCodeObject body(probe.data(), probe.size());
  Instrumentor instrumentor(object, kArch);
  const InstrumentationPoint point = {.anchor_offset = 4,
                                      .kind = InstrumentationKind::AfterInst,
                                      .probe_obj = &body,
                                      .probe_symbol = "observe_mask",
                                      .probe_args = {{.source = ProbeArgSource::AnchorExecLo}},
                                      .force_full_exec = true};
  instrumentor.add_point(point);
  const InstrumentedCodeObject result = instrumentor.patch();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  const AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  test::DbiSim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize("cdna4", 64));
  const std::vector<std::vector<uint32_t>> values = sim.run_and_read_vgprs(
      test::section_words(patched, ".text"), test::patched_private_segment_size(patched), {3, 2});
  ASSERT_EQ(values.size(), 2u);
  ASSERT_EQ(values[0].size(), 64u);
  ASSERT_EQ(values[1].size(), 64u);
  for (uint32_t lane = 0; lane < 64; ++lane) {
    EXPECT_EQ(values[0][lane], 3u);
    EXPECT_EQ(values[1][lane], lane < 2 ? 7u : 0u);
  }
}

} // namespace
} // namespace rocjitsu
