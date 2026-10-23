// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/branch_stack.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

using Program = std::vector<uint32_t>;

constexpr uint64_t kCodeBase = 0x100000;
constexpr uint32_t kSNop = 0xBF800000u;     // s_nop 0
constexpr uint32_t kModeWait = 0xBF800001u; // s_nop 1: two wait states between S_SETREGs of MODE

constexpr uint64_t kTarget = 0x200000;
constexpr uint64_t kNextPc = kCodeBase + 4;
// Seeded into stack entries to show a uniform fork leaves them alone.
constexpr uint64_t kStackSentinel = 0xDEADBEEFCAFEF00Dull;

class CbranchForkJoinTest : public ::testing::TestWithParam<rj_code_arch_t> {
protected:
  void SetUp() override {
    ComputeUnitCore::Config config{};
    config.arch = GetParam();
    // Common configuration for all tests.
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 256;
    config.functional_quantum = 1;
    cu = ComputeUnitCore::create("cbranch_fork_join_cu", config, &memory, &l2);
    ASSERT_NE(cu, nullptr);
    wf = cu->dispatch_wf(0, kCodeBase, 104, 256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(~uint64_t{0});
  }

  void TearDown() override {
    if (wf && !wf->is_halted())
      wf->halt();
  }

  void write_program(Program program) {
    program.insert(program.end(), 4, kSNop); // pad to instruction fetch window
    for (size_t i = 0; i < program.size(); ++i)
      memory.write32(kCodeBase + i * sizeof(uint32_t), program[i]);
  }

  /// Execute one instruction that falls through to the next one.
  void step(unsigned bytes = 4) {
    const uint64_t previous = wf->pc;
    (void)cu->step();
    EXPECT_FALSE(wf->is_halted());
    EXPECT_EQ(wf->pc, previous + bytes);
  }

  uint32_t sgpr(uint32_t index) { return cu->read_sgpr(wf->sgpr_alloc().base + index); }
  void set_sgpr(uint32_t index, uint32_t value) {
    cu->write_sgpr(wf->sgpr_alloc().base + index, value);
  }
  uint64_t sgpr64(uint32_t index) {
    return sgpr(index) | (static_cast<uint64_t>(sgpr(index + 1)) << 32);
  }
  void set_sgpr64(uint32_t index, uint64_t value) {
    set_sgpr(index, static_cast<uint32_t>(value));
    set_sgpr(index + 1, static_cast<uint32_t>(value >> 32));
  }

  GpuMemory memory{"cbranch_fork_join_memory"};
  L2Cache l2{"cbranch_fork_join_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  Wavefront *wf = nullptr;
};

TEST_P(CbranchForkJoinTest, CspSetterUpdatesOnlyTheCspBits) {
  constexpr uint32_t kOtherBits = ~Wavefront::CSP_MASK;
  wf->set_mode_raw(kOtherBits);

  for (uint32_t csp = 0; csp <= 7; ++csp) {
    wf->set_cond_stack_ptr(csp);
    EXPECT_EQ(wf->cond_stack_ptr(), csp);
    EXPECT_EQ(wf->mode_raw(), kOtherBits | (csp << Wavefront::CSP_SHIFT));
  }

  wf->set_cond_stack_ptr(8); // overflow; only last 3 bits are used
  EXPECT_EQ(wf->cond_stack_ptr(), 0u);
  EXPECT_EQ(wf->mode_raw(), kOtherBits);

  wf->set_cond_stack_ptr(0xFFFFFFF9u); // also, overflow
  EXPECT_EQ(wf->cond_stack_ptr(), 1u);
  EXPECT_EQ(wf->mode_raw(), kOtherBits | (1u << Wavefront::CSP_SHIFT));
}

TEST_P(CbranchForkJoinTest, Getreg24ReadsCsp) {
  write_program({
      0xB8981741u, // read MODE.CSP into s24
  });
  wf->set_cond_stack_ptr(5);
  step();

  EXPECT_EQ(sgpr(24), 5u);
}

TEST_P(CbranchForkJoinTest, SetregWritesCsp) {
  write_program({
      0xB9071741u, // write 7 into MODE.CSP
      kModeWait,
      0xBA001741u, // write 4 into MODE.CSP
      4u,          // literal
  });
  constexpr uint32_t kOtherBits = 0x000003C5u; // arbitrary non-CSP MODE bits; for testing
  wf->set_mode_raw(kOtherBits);
  set_sgpr(7, 3);

  step();
  EXPECT_EQ(wf->cond_stack_ptr(), 3u);
  EXPECT_EQ(wf->mode_raw() & ~Wavefront::CSP_MASK, kOtherBits);

  step();
  step(8);
  EXPECT_EQ(wf->cond_stack_ptr(), 4u);
  EXPECT_EQ(wf->mode_raw() & ~Wavefront::CSP_MASK, kOtherBits);
}

TEST_P(CbranchForkJoinTest, SetregWritesOnlyTheSelectedCspBits) {
  write_program({
      0xB9071741u, // write 7 into MODE.CSP
      kModeWait,
      0xB9070741u, // write 7 into MODE.CSP[0]
  });
  constexpr uint32_t kOtherBits = 0x000003C5u;
  wf->set_mode_raw(kOtherBits);

  set_sgpr(7, 0b1001); // should become 001 in MODE.CSP
  step();
  EXPECT_EQ(wf->cond_stack_ptr(), 1u);
  EXPECT_EQ(wf->mode_raw() & ~Wavefront::CSP_MASK, kOtherBits);

  // A 1-bit write at offset 29 replaces only CSP bit 0: 0b110 becomes 0b111.
  wf->set_cond_stack_ptr(0b110);
  set_sgpr(7, 0xFFFFFFF9u); // bit 0 set, bits 1-2 clear, everything above set
  step();
  step();
  EXPECT_EQ(wf->cond_stack_ptr(), 7u);
  EXPECT_EQ(wf->mode_raw() & ~Wavefront::CSP_MASK, kOtherBits);
}

TEST_P(CbranchForkJoinTest, ReusedSlotStartsWithEmptyForkStack) {
  wf->set_cond_stack_ptr(6);
  wf->halt();
  wf = cu->dispatch_wf(1, kCodeBase, 104, 256);
  ASSERT_NE(wf, nullptr);

  EXPECT_EQ(wf->cond_stack_ptr(), 0u);
}

TEST_P(CbranchForkJoinTest, BranchStackEntriesStartAtTheWavesFirstSgpr) {
  for (uint32_t n = 0; n <= 7; ++n)
    EXPECT_EQ(branch_stack_entry_sgpr(*wf, n), wf->sgpr_alloc().base + 4 * n);
}

TEST_P(CbranchForkJoinTest, ForkWithEveryActiveLanePassingJumpsWithoutPushing) {
  set_sgpr64(0, kStackSentinel);
  set_sgpr64(2, kStackSentinel);
  EXPECT_EQ(cbranch_fork(*wf, ~uint64_t{0}, kTarget, kNextPc), kTarget);
  EXPECT_EQ(wf->exec(), ~uint64_t{0});
  EXPECT_EQ(wf->cond_stack_ptr(), 0u);
  EXPECT_EQ(sgpr64(0), kStackSentinel);
  EXPECT_EQ(sgpr64(2), kStackSentinel);
}

TEST_P(CbranchForkJoinTest, ForkWithEveryActiveLaneFailingFallsThroughWithoutPushing) {
  set_sgpr64(0, kStackSentinel);
  set_sgpr64(2, kStackSentinel);
  EXPECT_EQ(cbranch_fork(*wf, 0, kTarget, kNextPc), kNextPc);
  EXPECT_EQ(wf->exec(), ~uint64_t{0});
  EXPECT_EQ(wf->cond_stack_ptr(), 0u);
  EXPECT_EQ(sgpr64(0), kStackSentinel);
  EXPECT_EQ(sgpr64(2), kStackSentinel);
}

TEST_P(CbranchForkJoinTest, ForkWithEmptyExecIsUniformAndJumps) {
  // With EXEC = 0 the pass mask equals EXEC, so the fork takes the branch.
  wf->set_exec(0);
  set_sgpr64(0, kStackSentinel);
  EXPECT_EQ(cbranch_fork(*wf, 0x15, kTarget, kNextPc), kTarget);
  EXPECT_EQ(wf->exec(), 0u);
  EXPECT_EQ(wf->cond_stack_ptr(), 0u);
  EXPECT_EQ(sgpr64(0), kStackSentinel);
}

TEST_P(CbranchForkJoinTest, ForkRunsTheSmallerPassSideAndPushesTheFailSide) {
  constexpr uint64_t kPass = 0x15; // 3 lanes pass, 61 fail
  EXPECT_EQ(cbranch_fork(*wf, kPass, kTarget, kNextPc), kTarget);
  EXPECT_EQ(wf->exec(), kPass);
  EXPECT_EQ(wf->cond_stack_ptr(), 1u);
  EXPECT_EQ(sgpr64(0), ~kPass);
  EXPECT_EQ(sgpr64(2), kNextPc);
}

TEST_P(CbranchForkJoinTest, ForkRunsTheSmallerFailSideAndPushesThePassSide) {
  constexpr uint64_t kFail = 0x15; // 3 lanes fail, 61 pass
  EXPECT_EQ(cbranch_fork(*wf, ~kFail, kTarget, kNextPc), kNextPc);
  EXPECT_EQ(wf->exec(), kFail);
  EXPECT_EQ(wf->cond_stack_ptr(), 1u);
  EXPECT_EQ(sgpr64(0), ~kFail);
  EXPECT_EQ(sgpr64(2), kTarget);
}

TEST_P(CbranchForkJoinTest, ForkTieRunsThePassSideFirst) {
  constexpr uint64_t kPass = 0x00000000FFFFFFFFull; // 32 lanes each way
  EXPECT_EQ(cbranch_fork(*wf, kPass, kTarget, kNextPc), kTarget);
  EXPECT_EQ(wf->exec(), kPass);
  EXPECT_EQ(wf->cond_stack_ptr(), 1u);
  EXPECT_EQ(sgpr64(0), ~kPass);
  EXPECT_EQ(sgpr64(2), kNextPc);
}

TEST_P(CbranchForkJoinTest, ForkIgnoresCompareBitsOfInactiveLanes) {
  // Only lanes 0-7 are active. The compare mask also sets lanes 8-63; they must
  // not reach either side. That leaves pass = 0x0F and fail = 0xF0, a tie.
  wf->set_exec(0xFF);
  EXPECT_EQ(cbranch_fork(*wf, 0xFFFFFFFFFFFFFF0Full, kTarget, kNextPc), kTarget);
  EXPECT_EQ(wf->exec(), 0x0Fu);
  EXPECT_EQ(wf->cond_stack_ptr(), 1u);
  EXPECT_EQ(sgpr64(0), 0xF0u);
  EXPECT_EQ(sgpr64(2), kNextPc);
}

TEST_P(CbranchForkJoinTest, JoinFirstArrivalResumesTheSavedPath) {
  // State left by a divergent fork that started at CSP 0.
  wf->set_cond_stack_ptr(1);
  wf->set_exec(0x15);
  set_sgpr64(0, ~uint64_t{0x15});
  set_sgpr64(2, kTarget);
  EXPECT_EQ(cbranch_join(*wf, /*saved_csp=*/0, kNextPc), kTarget);
  EXPECT_EQ(wf->exec(), ~uint64_t{0x15});
  EXPECT_EQ(wf->cond_stack_ptr(), 0u);
}

TEST_P(CbranchForkJoinTest, JoinSecondArrivalFallsThrough) {
  wf->set_cond_stack_ptr(2);
  wf->set_exec(0x15);
  EXPECT_EQ(cbranch_join(*wf, /*saved_csp=*/2, kNextPc), kNextPc);
  EXPECT_EQ(wf->exec(), 0x15u);
  EXPECT_EQ(wf->cond_stack_ptr(), 2u);
}

TEST_P(CbranchForkJoinTest, NestedForksPopTheInnerEntryBeforeTheOuter) {
  constexpr uint64_t kInnerTarget = 0x300000;
  constexpr uint64_t kInnerNextPc = 0x300004;
  constexpr uint64_t kAfterInnerJoin = 0x300100;
  constexpr uint64_t kAfterOuterJoin = 0x300200;

  // Outer fork: lanes 0-3 pass and run first; lanes 4-63 wait in entry 0.
  const uint32_t outer_saved = wf->cond_stack_ptr();
  EXPECT_EQ(outer_saved, 0u);
  EXPECT_EQ(cbranch_fork(*wf, 0x0F, kTarget, kNextPc), kTarget);
  EXPECT_EQ(wf->exec(), 0x0Fu);

  // Inner fork inside the outer pass side: lane 0 passes, lanes 1-3 wait in entry 1.
  const uint32_t inner_saved = wf->cond_stack_ptr();
  EXPECT_EQ(inner_saved, 1u);
  EXPECT_EQ(cbranch_fork(*wf, 0x01, kInnerTarget, kInnerNextPc), kInnerTarget);
  EXPECT_EQ(wf->exec(), 0x01u);
  EXPECT_EQ(wf->cond_stack_ptr(), 2u);
  EXPECT_EQ(sgpr64(4), 0x0Eu);
  EXPECT_EQ(sgpr64(6), kInnerNextPc);

  EXPECT_EQ(cbranch_join(*wf, inner_saved, kAfterInnerJoin), kInnerNextPc);
  EXPECT_EQ(wf->exec(), 0x0Eu);
  EXPECT_EQ(wf->cond_stack_ptr(), 1u);

  EXPECT_EQ(cbranch_join(*wf, inner_saved, kAfterInnerJoin), kAfterInnerJoin);
  EXPECT_EQ(wf->cond_stack_ptr(), 1u);
  EXPECT_EQ(sgpr64(0), ~uint64_t{0x0F});
  EXPECT_EQ(sgpr64(2), kNextPc);

  EXPECT_EQ(cbranch_join(*wf, outer_saved, kAfterOuterJoin), kNextPc);
  EXPECT_EQ(wf->exec(), ~uint64_t{0x0F});
  EXPECT_EQ(wf->cond_stack_ptr(), 0u);

  EXPECT_EQ(cbranch_join(*wf, outer_saved, kAfterOuterJoin), kAfterOuterJoin);
  EXPECT_EQ(wf->cond_stack_ptr(), 0u);
}

TEST_P(CbranchForkJoinTest, ForkAtFullDepthWritesEntrySevenAndWrapsCsp) {
  // The CDNA3/4 pseudocode increments CSP as a 3-bit field (WAVE_MODE.CSP += 3'1U),
  // so a push at CSP 7 writes s[28..31] and leaves CSP at 0. No valid program
  // gets here; this pins the modelled behaviour.
  wf->set_cond_stack_ptr(7);
  EXPECT_EQ(cbranch_fork(*wf, 0x15, kTarget, kNextPc), kTarget);
  EXPECT_EQ(wf->cond_stack_ptr(), 0u);
  EXPECT_EQ(sgpr64(28), ~uint64_t{0x15});
  EXPECT_EQ(sgpr64(30), kNextPc);
}

// MODE.CSP and the fork/join instructions exist only on GFX9-derived ISAs.
INSTANTIATE_TEST_SUITE_P(Cdna, CbranchForkJoinTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                           ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4));

} // namespace
