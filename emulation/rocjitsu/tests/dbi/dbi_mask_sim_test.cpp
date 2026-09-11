// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbi_mask_sim_test.cpp
/// @brief Simulator end-to-end for DBI probe mask policy on CDNA3 (gfx942,
/// wave64), CDNA4 (gfx950, wave64), and RDNA4 (gfx1200, wave32).
///
/// Two capabilities, both only observable at a site whose EXEC is *not* full:
/// running the probe with every lane enabled, and handing it the mask the guest
/// was running under. The static tests
/// (tests/patch/trampoline_builder_test.cpp) prove the envelope reaches the call
/// with the window open and that the mask argument reads the saved pair. This
/// proves what the lanes actually do.
///
/// Every kernel here has the same shape: some setup under the launch mask, then
/// `s_mov_b64 exec, 3` to narrow to lanes 0 and 1, then the anchor, then
/// `s_endpgm`. v3 is the observation register, written under the full launch
/// mask and dead at the anchor, so the envelope does not spill it and whatever
/// the probe leaves there survives to the halt snapshot. Lanes the probe did not
/// run on still read their setup value.
///
/// The masked and full-exec cases are each other's control. They are run in two
/// shapes, because the envelope differs: with no arguments and nothing to spill
/// the full-mask window never opens at all, so the masked probe simply inherits
/// an EXEC nobody touched; with a live argument VGPR the window opens, the
/// argument is written in every lane, and the masked case additionally exercises
/// the anchor-mask restore that force_full_exec is defined by skipping.

#include "../dbi_test_util.h"
#include "dbi_sim.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/rj_code.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace rocjitsu {
namespace {

using test::kMovV1V0;
using test::kMovV3V0;

// Lanes active at the anchor. An inline scalar constant so the narrowing s_mov
// is one word; small enough that most of the wave is inactive on both wave
// sizes. Chosen odd-shaped rather than 1 so exec_hi is distinguishable from
// exec_lo when both are passed.
constexpr uint32_t kAnchorMask = 3;
// What the probe leaves in v3. Distinct from every value the guest writes, so
// "the probe ran on this lane" is readable per lane.
constexpr uint32_t kProbeSentinel = 42;
// What the kernel leaves in v3 before the anchor: the value an unreached lane
// still shows.
constexpr uint32_t kUntouched = 0;
// The guest's own value in v0, live across the anchor in the argument shapes.
constexpr uint32_t kGuestValue = 7;
// Pre-set in v2 so a probe writing exec_hi (which is 0 on a wave64 mask of 3)
// is distinguishable from a probe that never ran.
constexpr uint32_t kV2Preset = 9;

// v_mov_b32 v2, v1. Same VOP1 layout as dbi_test_util.h's kMovV3V2: vdst in bits
// [24:17], op 1 in [16:9], src0 257 (v1) in [8:0]. Local because the shared
// header has no v2 <- v1 form and its nearest sibling, kMovV5V1, names a
// register this fixture's descriptor puts in the AGPR window.
constexpr uint32_t kMovV2V1 = 0x7E040301u;

struct MaskSimArch {
  const char *sim_arch;
  rj_code_arch_t arch;
  uint32_t e_flags;
  uint32_t wave_size;
  // Descriptor wave size must match what DbiSim dispatches, or the patch-time
  // gates reason about a different kernel than the one that executes. Only RDNA
  // has a choice to make; CDNA is wave64 by construction.
  bool wave32;
};

inline constexpr MaskSimArch kCdna3MaskArch{"cdna3", ROCJITSU_CODE_ARCH_CDNA3,
                                            EF_AMDGPU_MACH_AMDGCN_GFX942, /*wave_size=*/64,
                                            /*wave32=*/false};
inline constexpr MaskSimArch kCdna4MaskArch{"cdna4", ROCJITSU_CODE_ARCH_CDNA4,
                                            EF_AMDGPU_MACH_AMDGCN_GFX950, /*wave_size=*/64,
                                            /*wave32=*/false};
inline constexpr MaskSimArch kRdna4MaskArch{"rdna4", ROCJITSU_CODE_ARCH_RDNA4,
                                            EF_AMDGPU_MACH_AMDGCN_GFX1200, /*wave_size=*/32,
                                            /*wave32=*/true};

class DbiMaskSimBase : public ::testing::Test {
protected:
  explicit DbiMaskSimBase(const MaskSimArch &a) : a_(a) {}

  // Kernel: @p setup under the launch mask, the narrowing s_mov, the anchor, and
  // s_endpgm. The anchor is the word after the narrow, so its offset follows
  // from the setup length.
  [[nodiscard]] std::vector<uint8_t> make_target(const std::vector<uint32_t> &setup,
                                                 uint32_t anchor_word) {
    std::vector<uint32_t> text = setup;
    text.push_back(build_s_mov_b64(scalar_operand_exec_lo(a_.arch),
                                   scalar_positive_inline_u32(kAnchorMask), a_.arch));
    anchor_offset_ = static_cast<uint64_t>(text.size()) * sizeof(uint32_t);
    text.push_back(anchor_word);
    text.push_back(build_s_endpgm(a_.arch));
    return test::make_amdgpu_kernel_elf(text, /*private_bytes=*/64, /*granulated_sgpr_count=*/3,
                                        a_.e_flags, /*granulated_vgpr_count=*/0,
                                        /*accum_offset=*/0, /*unterminated_kd_name=*/false,
                                        /*wrap_section_header_table=*/false,
                                        /*wrap_symtab_range=*/false, /*kd_crosses_section=*/false,
                                        a_.wave32);
  }

  // Patch the anchor with @p probe_body and run. Returns the per-lane values of
  // @p regs.
  std::vector<std::vector<uint32_t>> run_patched(const std::vector<uint8_t> &target,
                                                 const std::vector<uint32_t> &probe_body,
                                                 const std::vector<ProbeArgValue> &args,
                                                 bool full_exec,
                                                 const std::vector<uint32_t> &regs) {
    auto probe = test::make_amdgpu_probe_elf("rj_test_mask_probe", probe_body, a_.e_flags);
    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    EXPECT_TRUE(obj.is_valid());
    EXPECT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, a_.arch);
    InstrumentationPoint pt;
    pt.anchor_offset = anchor_offset_;
    pt.probe_obj = &probe_obj;
    pt.probe_symbol = "rj_test_mask_probe";
    pt.probe_args = args;
    pt.force_full_exec = full_exec;
    instr.add_point(pt);

    auto result = instr.patch_with_debug_summaries();
    EXPECT_TRUE(result.errors.empty())
        << (result.errors.empty() ? std::string{} : result.errors.front());
    if (!result.errors.empty())
      return std::vector<std::vector<uint32_t>>(regs.size());

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    EXPECT_TRUE(patched.is_valid());
    patched_text_ = test::section_words(patched, ".text");
    patched_scratch_ = test::patched_private_segment_size(patched);
    EXPECT_FALSE(patched_text_.empty());

    test::DbiSim sim(a_.sim_arch, a_.wave_size);
    return sim.run_and_read_vgprs(patched_text_, patched_scratch_, regs);
  }

  // The no-argument kernel: v3 initialized, nothing live across the anchor.
  [[nodiscard]] std::vector<uint8_t> bare_target() {
    return make_target({test::make_mov_vgpr_inline(3, kUntouched)},
                       test::make_mov_vgpr_inline(1, 5));
  }

  // The argument kernel: v0 holds a guest value the anchor reads, so the
  // argument write lands on a live register and the envelope has to spill it.
  [[nodiscard]] std::vector<uint8_t> live_argument_target() {
    return make_target(
        {test::make_mov_vgpr_inline(3, kUntouched), test::make_mov_vgpr_inline(0, kGuestValue)},
        kMovV1V0);
  }

  // Publishes an inline sentinel in v3; reads nothing, so it takes no arguments.
  [[nodiscard]] std::vector<uint32_t> sentinel_probe() const {
    return {test::make_mov_vgpr_inline(3, kProbeSentinel),
            build_s_setpc_b64(/*s[30:31]=*/30, a_.arch)};
  }

  // Republishes its first argument in v3.
  [[nodiscard]] std::vector<uint32_t> echo_probe() const {
    return {kMovV3V0, build_s_setpc_b64(/*s[30:31]=*/30, a_.arch)};
  }

  // Widened before the shift: the wave64 fixtures run lanes up to 63, and
  // shifting a uint32_t by 32 or more is undefined. On x86 the count masks to 5
  // bits, so lane 32 would read as lane 0 and an inactive lane could report
  // active.
  [[nodiscard]] bool lane_active(uint32_t lane) const {
    return (uint64_t{kAnchorMask} >> lane) & 1u;
  }

  //============================================================================
  // No arguments, nothing to spill: the envelope emits no EXEC toggles at all,
  // so the probe simply inherits the mask the guest had.
  //============================================================================

  void expect_masked_probe_reaches_only_active_lanes() {
    const std::vector<uint32_t> v3 =
        run_patched(bare_target(), sentinel_probe(), {}, /*full_exec=*/false, {3}).front();
    ASSERT_EQ(v3.size(), a_.wave_size) << "kernel did not run to completion (no dispatched wave)";
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v3[lane], lane_active(lane) ? kProbeSentinel : kUntouched)
          << "lane " << lane << ": a masked probe must run exactly on the anchor's lanes";
  }

  void expect_full_exec_probe_reaches_every_lane() {
    const std::vector<uint32_t> v3 =
        run_patched(bare_target(), sentinel_probe(), {}, /*full_exec=*/true, {3}).front();
    ASSERT_EQ(v3.size(), a_.wave_size) << "kernel did not run to completion (no dispatched wave)";
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v3[lane], kProbeSentinel)
          << "lane " << lane << ": a full-exec probe must run on every lane";
  }

  //============================================================================
  // One argument in a live v0: the window opens, the argument is written in
  // every lane, and v0 spills. This is the shape a real logging probe takes.
  //============================================================================

  // Masked, so the envelope emits the anchor-mask restore between the argument
  // writes and the call. That restore is precisely what force_full_exec omits,
  // so this is the case the policy is defined against.
  void expect_masked_probe_with_a_live_argument_restores_the_anchor_mask() {
    const auto regs = run_patched(live_argument_target(), echo_probe(),
                                  {probe_arg_imm(kProbeSentinel)}, /*full_exec=*/false, {3, 1});
    ASSERT_EQ(regs[0].size(), a_.wave_size) << "kernel did not run to completion";
    EXPECT_EQ(patched_scratch_, 68u) << "descriptor scratch must grow to spill the live v0";
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane) {
      EXPECT_EQ(regs[0][lane], lane_active(lane) ? kProbeSentinel : kUntouched)
          << "lane " << lane << ": the probe must have run under the restored anchor mask";
      if (lane_active(lane)) {
        EXPECT_EQ(regs[1][lane], kGuestValue)
            << "lane " << lane << ": the guest's v0 did not survive carrying the argument";
      }
    }
  }

  // The same site under force_full_exec: the probe now reaches every lane, and
  // the spill still round-trips, so the guest is unharmed either way.
  void expect_full_exec_probe_with_a_live_argument_spills_and_reaches_every_lane() {
    const auto regs = run_patched(live_argument_target(), echo_probe(),
                                  {probe_arg_imm(kProbeSentinel)}, /*full_exec=*/true, {3, 1});
    ASSERT_EQ(regs[0].size(), a_.wave_size) << "kernel did not run to completion";
    EXPECT_EQ(patched_scratch_, 68u) << "descriptor scratch must grow to spill the live v0";
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane) {
      EXPECT_EQ(regs[0][lane], kProbeSentinel)
          << "lane " << lane << ": a full-exec probe must receive its argument in every lane";
      if (lane_active(lane)) {
        EXPECT_EQ(regs[1][lane], kGuestValue)
            << "lane " << lane << ": the guest's v0 did not survive carrying the argument";
      }
    }
  }

  //============================================================================
  // The mask as a value.
  //============================================================================

  // Run under force_full_exec so every lane's v3 carries it: the argument write
  // already runs under EXEC = -1, so a lane reading kUntouched would mean the
  // value never arrived rather than that the lane was inactive.
  void expect_probe_receives_the_anchor_mask() {
    const std::vector<uint32_t> v3 =
        run_patched(bare_target(), echo_probe(), {{ProbeArgSource::AnchorExecLo, 0}},
                    /*full_exec=*/true, {3})
            .front();
    ASSERT_EQ(v3.size(), a_.wave_size) << "kernel did not run to completion";
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v3[lane], kAnchorMask)
          << "lane " << lane << ": probe did not receive the guest's EXEC mask";
  }

  // Wave64 only: the high dword is rejected outright on a Wave32 kernel. v2 is
  // preset to a nonzero value so receiving exec_hi (0 for this mask) is
  // distinguishable from the probe never having written it.
  void expect_probe_receives_both_mask_dwords() {
    ASSERT_EQ(a_.wave_size, 64u) << "exec_hi is meaningful only on a wave64 kernel";
    auto target = make_target(
        {test::make_mov_vgpr_inline(3, kUntouched), test::make_mov_vgpr_inline(2, kV2Preset)},
        test::make_mov_vgpr_inline(1, 5));
    // Publish v0 (exec_lo) in v3 and v1 (exec_hi) in v2.
    const std::vector<uint32_t> probe{kMovV3V0, kMovV2V1,
                                      build_s_setpc_b64(/*s[30:31]=*/30, a_.arch)};
    const auto regs = run_patched(
        target, probe, {{ProbeArgSource::AnchorExecLo, 0}, {ProbeArgSource::AnchorExecHi, 0}},
        /*full_exec=*/true, {3, 2});
    ASSERT_EQ(regs[0].size(), a_.wave_size) << "kernel did not run to completion";
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane) {
      EXPECT_EQ(regs[0][lane], kAnchorMask) << "lane " << lane << ": wrong exec_lo";
      EXPECT_EQ(regs[1][lane], 0u) << "lane " << lane << ": wrong exec_hi";
    }
  }

  // Wave32 only, and the counterpart of the case above: with no high EXEC dword
  // to deliver, the request is rejected at patch time. Asserting it here is what
  // makes this fixture's Wave32 descriptor observable -- without it the flag
  // would be decorative, and the suite would plan against a Wave64 kernel while
  // dispatching 32 lanes.
  void expect_anchor_exec_high_dword_is_rejected() {
    ASSERT_EQ(a_.wave_size, 32u) << "exec_hi is rejected only on a wave32 kernel";
    auto target = bare_target();
    auto probe = test::make_amdgpu_probe_elf("rj_test_mask_probe", echo_probe(), a_.e_flags);
    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());

    Instrumentor instr(obj, a_.arch);
    InstrumentationPoint pt;
    pt.anchor_offset = anchor_offset_;
    pt.probe_obj = &probe_obj;
    pt.probe_symbol = "rj_test_mask_probe";
    pt.probe_args = {{ProbeArgSource::AnchorExecHi, 0}};
    instr.add_point(pt);

    auto result = instr.patch_with_debug_summaries();
    ASSERT_FALSE(result.errors.empty()) << "a Wave32 kernel has no high EXEC dword to pass";
    EXPECT_NE(result.errors.front().find("Wave32"), std::string::npos) << result.errors.front();
  }

  //============================================================================
  // Negative control.
  //============================================================================

  // Nop the widen that opens the full-mask window and the full-exec probe stops
  // reaching the inactive lanes, which is the answer the masked case gives.
  // Proves the extra lanes come from that instruction and not from the wave
  // happening to be fully active.
  void expect_without_the_widen_full_exec_reaches_only_active_lanes() {
    ASSERT_NO_FATAL_FAILURE(
        (void)run_patched(bare_target(), sentinel_probe(), {}, /*full_exec=*/true, {3}));
    ASSERT_FALSE(patched_text_.empty());

    const uint32_t widen =
        build_s_mov_b64(scalar_operand_exec_lo(a_.arch), scalar_inline_neg_one(a_.arch), a_.arch);
    std::vector<uint32_t> sabotaged = patched_text_;
    auto it = std::find(sabotaged.begin(), sabotaged.end(), widen);
    ASSERT_NE(it, sabotaged.end()) << "full-mask widen not found in the patched text";
    *it = build_s_nop(0, a_.arch);

    test::DbiSim broken(a_.sim_arch, a_.wave_size);
    const std::vector<uint32_t> v3 = broken.run_and_read_vgpr(sabotaged, patched_scratch_,
                                                              /*reg=*/3);
    ASSERT_EQ(v3.size(), a_.wave_size);
    for (uint32_t lane = 0; lane < a_.wave_size; ++lane)
      EXPECT_EQ(v3[lane], lane_active(lane) ? kProbeSentinel : kUntouched)
          << "lane " << lane << ": without the widen the probe must stay under the anchor mask";
  }

  MaskSimArch a_;
  uint64_t anchor_offset_ = 0;
  std::vector<uint32_t> patched_text_;
  uint32_t patched_scratch_ = 0;
};

class DbiCdna3MaskSimFixture : public DbiMaskSimBase {
protected:
  DbiCdna3MaskSimFixture() : DbiMaskSimBase(kCdna3MaskArch) {}
};
class DbiCdna4MaskSimFixture : public DbiMaskSimBase {
protected:
  DbiCdna4MaskSimFixture() : DbiMaskSimBase(kCdna4MaskArch) {}
};
class DbiRdna4MaskSimFixture : public DbiMaskSimBase {
protected:
  DbiRdna4MaskSimFixture() : DbiMaskSimBase(kRdna4MaskArch) {}
};

TEST_F(DbiCdna3MaskSimFixture, MaskedProbeReachesOnlyActiveLanes) {
  expect_masked_probe_reaches_only_active_lanes();
}
TEST_F(DbiCdna3MaskSimFixture, FullExecProbeReachesEveryLane) {
  expect_full_exec_probe_reaches_every_lane();
}
TEST_F(DbiCdna3MaskSimFixture, MaskedProbeWithALiveArgumentRestoresTheAnchorMask) {
  expect_masked_probe_with_a_live_argument_restores_the_anchor_mask();
}
TEST_F(DbiCdna3MaskSimFixture, FullExecProbeWithALiveArgumentSpillsAndReachesEveryLane) {
  expect_full_exec_probe_with_a_live_argument_spills_and_reaches_every_lane();
}
TEST_F(DbiCdna3MaskSimFixture, ProbeReceivesTheAnchorMask) {
  expect_probe_receives_the_anchor_mask();
}
TEST_F(DbiCdna3MaskSimFixture, WithoutTheWidenFullExecReachesOnlyActiveLanes) {
  expect_without_the_widen_full_exec_reaches_only_active_lanes();
}

TEST_F(DbiCdna4MaskSimFixture, MaskedProbeReachesOnlyActiveLanes) {
  expect_masked_probe_reaches_only_active_lanes();
}
TEST_F(DbiCdna4MaskSimFixture, FullExecProbeReachesEveryLane) {
  expect_full_exec_probe_reaches_every_lane();
}
TEST_F(DbiCdna4MaskSimFixture, MaskedProbeWithALiveArgumentRestoresTheAnchorMask) {
  expect_masked_probe_with_a_live_argument_restores_the_anchor_mask();
}
TEST_F(DbiCdna4MaskSimFixture, FullExecProbeWithALiveArgumentSpillsAndReachesEveryLane) {
  expect_full_exec_probe_with_a_live_argument_spills_and_reaches_every_lane();
}
TEST_F(DbiCdna4MaskSimFixture, ProbeReceivesTheAnchorMask) {
  expect_probe_receives_the_anchor_mask();
}
TEST_F(DbiCdna4MaskSimFixture, WithoutTheWidenFullExecReachesOnlyActiveLanes) {
  expect_without_the_widen_full_exec_reaches_only_active_lanes();
}

TEST_F(DbiRdna4MaskSimFixture, MaskedProbeReachesOnlyActiveLanes) {
  expect_masked_probe_reaches_only_active_lanes();
}
TEST_F(DbiRdna4MaskSimFixture, FullExecProbeReachesEveryLane) {
  expect_full_exec_probe_reaches_every_lane();
}
TEST_F(DbiRdna4MaskSimFixture, MaskedProbeWithALiveArgumentRestoresTheAnchorMask) {
  expect_masked_probe_with_a_live_argument_restores_the_anchor_mask();
}
TEST_F(DbiRdna4MaskSimFixture, FullExecProbeWithALiveArgumentSpillsAndReachesEveryLane) {
  expect_full_exec_probe_with_a_live_argument_spills_and_reaches_every_lane();
}
TEST_F(DbiRdna4MaskSimFixture, ProbeReceivesTheAnchorMask) {
  expect_probe_receives_the_anchor_mask();
}
TEST_F(DbiRdna4MaskSimFixture, WithoutTheWidenFullExecReachesOnlyActiveLanes) {
  expect_without_the_widen_full_exec_reaches_only_active_lanes();
}

// Wave64 only. A Wave32 kernel has no high EXEC dword, and the orchestrator
// rejects the request rather than delivering one (covered in
// tests/patch/instrumentor_test.cpp).
TEST_F(DbiCdna3MaskSimFixture, ProbeReceivesBothMaskDwords) {
  expect_probe_receives_both_mask_dwords();
}
TEST_F(DbiCdna4MaskSimFixture, ProbeReceivesBothMaskDwords) {
  expect_probe_receives_both_mask_dwords();
}

// Wave32 only, for the same reason inverted.
TEST_F(DbiRdna4MaskSimFixture, AnchorExecHighDwordIsRejected) {
  expect_anchor_exec_high_dword_is_rejected();
}

} // namespace
} // namespace rocjitsu
