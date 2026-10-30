// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbi_wave32_sim_test.cpp
/// @brief Simulator end-to-end for DBI liveness on an RDNA4 (gfx1200) kernel
/// that executes Wave32.
///
/// A Wave32 compare writes one SGPR of the pair its encoding names. In the
/// kernel's own code, the kernel saves EXEC in s1 and narrows it, the anchor
/// follows, then a compare into s0, then EXEC is restored from s1 and every
/// lane writes v3. Were the compare taken to write s[0:1], s1 would be dead at
/// the anchor and free for the call envelope's temporaries; the restore would
/// then bring back the envelope's value, and the lanes it leaves off would
/// never write v3. That takes the envelope choosing s[0:1], the lowest dead
/// pair; the Wave64-planned negative control fails if it stops doing so. In the
/// probe, a compare into s2 must leave the kernel's s3 alone without a spill.

#include "../dbi_test_util.h"
#include "dbi_sim.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace rocjitsu {
namespace {

constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
constexpr uint32_t kWaveSize = 32;
// Lanes active at the anchor: a strict subset, so a lost EXEC restore shows.
constexpr uint32_t kAnchorMask = 3;
// The value every lane must end up with in v3.
constexpr uint32_t kWritten = 42;

// Patch @p text with @p probe_body at @p anchor_offset, its descriptor
// declaring Wave32 or Wave64, then run it Wave32 either way. Returns v3 per
// lane; empty when patching failed or no wave halted.
std::vector<uint32_t> patch_and_run(const std::vector<uint32_t> &text, uint64_t anchor_offset,
                                    const std::vector<uint32_t> &probe_body,
                                    bool descriptor_wave32) {
  const auto target = descriptor_wave32
                          ? test::make_gfx1200_wave32_kernel_elf(text, /*private_bytes=*/0)
                          : test::make_gfx1200_kernel_elf(text, /*private_bytes=*/0);
  const auto probe = test::make_gfx1200_probe_elf("rj_test_probe", probe_body);
  AmdGpuCodeObject obj(target.data(), target.size());
  AmdGpuCodeObject probe_obj(probe.data(), probe.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_TRUE(probe_obj.is_valid());

  Instrumentor instr(obj, kArch);
  InstrumentationPoint pt;
  pt.anchor_offset = anchor_offset;
  pt.probe_obj = &probe_obj;
  pt.probe_symbol = "rj_test_probe";
  instr.add_point(pt);
  auto result = instr.patch_with_debug_summaries();
  EXPECT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front());
  if (!result.errors.empty())
    return {};
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  EXPECT_TRUE(patched.is_valid());

  test::DbiSim sim;
  sim.initialize("rdna4", kWaveSize);
  if (testing::Test::HasFatalFailure())
    return {};
  sim.set_wave32_dispatch();
  return sim.run_and_read_vgpr(test::section_words(patched, ".text"),
                               test::patched_private_segment_size(patched), /*reg=*/3);
}

// The kernel's saved-EXEC sequence, with a no-op probe at the anchor.
std::vector<uint32_t> run_saved_exec_kernel(bool descriptor_wave32) {
  constexpr uint16_t kExecLo = scalar_operand_exec_lo(kArch);
  constexpr auto compare =
      rdna4::build_vop3(rdna4::kVCmpEqU32Vop3, {.vdst = 0, .src0 = 128, .src1 = 256 + 7});
  std::vector<uint32_t> text{
      test::make_mov_vgpr_inline(3, 0),
      build_s_mov_b32(/*sdst=*/1, kExecLo, kArch),
      build_s_mov_b32(kExecLo, scalar_positive_inline_u32(kAnchorMask), kArch),
  };
  const uint64_t anchor_offset = text.size() * sizeof(uint32_t);
  text.insert(text.end(), {
                              test::make_mov_vgpr_inline(1, 5),
                              compare[0],
                              compare[1],
                              build_s_mov_b32(kExecLo, /*ssrc0=*/1, kArch),
                              test::make_mov_vgpr_inline(3, kWritten),
                              build_s_endpgm(kArch),
                          });
  return patch_and_run(text, anchor_offset, {build_s_setpc_b64(/*s[30:31]=*/30, kArch)},
                       descriptor_wave32);
}

TEST(DbiWave32Sim, CompareIntoS0KeepsSavedExecInS1) {
  const std::vector<uint32_t> v3 = run_saved_exec_kernel(/*descriptor_wave32=*/true);
  ASSERT_EQ(v3.size(), kWaveSize) << "kernel did not run to completion (no dispatched wave)";
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    EXPECT_EQ(v3[lane], kWritten) << "lane " << lane << " did not get its EXEC bit back";
}

// Negative control: planned as Wave64, liveness takes the compare to write
// s[0:1], so s1 is free at the anchor and the restore loses the saved EXEC.
TEST(DbiWave32Sim, PlannedAsWave64LosesSavedExec) {
  const std::vector<uint32_t> v3 = run_saved_exec_kernel(/*descriptor_wave32=*/false);
  ASSERT_EQ(v3.size(), kWaveSize) << "kernel did not run to completion (no dispatched wave)";
  EXPECT_TRUE(std::ranges::any_of(v3, [](uint32_t value) { return value != kWritten; }))
      << "the envelope left s1 alone even though liveness reported it dead";
}

// The kernel holds kWritten in s3 across the call and has no scratch. The
// probe's compare into s2 is planned as clobbering only s2, so the call needs
// no spill; s3 then reaches v3 only if the compare really writes s2 alone.
TEST(DbiWave32Sim, ProbeCompareIntoS2KeepsKernelValueInS3) {
  const std::vector<uint32_t> text{
      build_s_mov_b32(/*sdst=*/3, scalar_positive_inline_u32(kWritten), kArch),
      test::make_mov_vgpr_inline(1, 5),
      rdna4::build_vop1(rdna4::kVMovB32Vop1, {.src0 = 3, .vdst = 3})[0],
      build_s_endpgm(kArch),
  };
  constexpr auto compare =
      rdna4::build_vop3(rdna4::kVCmpEqU32Vop3, {.vdst = 2, .src0 = 128, .src1 = 128});
  const std::vector<uint32_t> v3 =
      patch_and_run(text, /*anchor_offset=*/sizeof(uint32_t),
                    {compare[0], compare[1], build_s_setpc_b64(/*s[30:31]=*/30, kArch)},
                    /*descriptor_wave32=*/true);
  ASSERT_EQ(v3.size(), kWaveSize) << "kernel did not run to completion (no dispatched wave)";
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    EXPECT_EQ(v3[lane], kWritten) << "lane " << lane << " lost the kernel's s3";
}

} // namespace
} // namespace rocjitsu
