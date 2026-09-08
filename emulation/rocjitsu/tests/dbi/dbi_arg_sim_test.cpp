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

#include "../dbi_test_util.h"
#include "dbi_sim.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/vector_builders.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/rj_code.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
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
    const uint32_t endpgm = build_s_endpgm(a_.arch);
    const uint32_t setpc = build_s_setpc_b64(/*s[30:31]=*/30, a_.arch);
    // v_mov v0, K0 ; v_mov v1, v0 (ANCHOR at offset 4, v0 live) ; s_endpgm.
    auto target = test::make_amdgpu_kernel_elf(
        {test::make_mov_vgpr_inline(0, kGuestValue), kMovV1V0, endpgm}, /*private_bytes=*/64,
        /*granulated_sgpr_count=*/3, a_.e_flags);
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
    pt.probe_args = {kArgSentinel};
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
    test::DbiSim sim(a_.sim_arch, a_.wave_size);
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
    const auto arg_write = build_v_mov_b32_imm(0, kArgSentinel, a_.arch);
    const uint32_t nop = build_s_nop(0, a_.arch);

    std::vector<uint32_t> sabotaged = patched_text_;
    auto it = std::search(sabotaged.begin(), sabotaged.end(), arg_write.begin(), arg_write.end());
    ASSERT_NE(it, sabotaged.end()) << "argument materialization not found in the patched text";
    for (size_t i = 0; i < arg_write.size(); ++i)
      *(it + static_cast<std::ptrdiff_t>(i)) = nop;

    test::DbiSim broken(a_.sim_arch, a_.wave_size);
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

} // namespace
} // namespace rocjitsu
