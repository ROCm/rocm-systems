// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/shared/branch_stack.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
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

// If/else through the branch stack. Offsets are bytes from kCodeBase.
constexpr uint32_t kIForkVccThen = 0xB86A0002u; // s_cbranch_i_fork vcc, THEN
constexpr uint32_t kGForkVccS36 = 0x9480246Au;  // s_cbranch_g_fork vcc, s[36:37]
constexpr uint64_t kForkOffset = 0x08;
constexpr uint64_t kElseOffset = 0x0C;
constexpr uint64_t kThenOffset = 0x14;
constexpr uint64_t kJoinOffset = 0x18;
constexpr uint64_t kEndOffset = 0x24;
constexpr uint32_t kThenValue = 1;
constexpr uint32_t kElseValue = 2;
constexpr uint32_t kAfterJoinValue = 7;

// Per lane:
//   v0 = (vcc lane set) ? 1 : 2;
//   v1 = 7;
//
// As assembly:
//         s_getreg_b32      s34, hwreg(HW_REG_MODE, 29, 3)  ; saved CSP
//         s_mov_b64         s[32:33], exec                  ; saved EXEC
//         <fork>            vcc, THEN
//   ELSE: v_mov_b32         v0, 2
//         s_branch          JOIN
//   THEN: v_mov_b32         v0, 1
//   JOIN: s_cbranch_join    s34
//         s_mov_b64         exec, s[32:33]                  ; join leaves the last side's EXEC
//         v_mov_b32         v1, 7
Program if_else_program(uint32_t fork) {
  return {
      0xB8A21741u, // 0x00 s_getreg_b32 s34, hwreg(HW_REG_MODE, 29, 3)
      0xBEA0017Eu, // 0x04 s_mov_b64 s[32:33], exec
      fork,        // 0x08 fork to THEN
      0x7E000282u, // 0x0C ELSE: v_mov_b32 v0, 2
      0xBF820001u, // 0x10 s_branch JOIN
      0x7E000281u, // 0x14 THEN: v_mov_b32 v0, 1
      0xBE802E22u, // 0x18 JOIN: s_cbranch_join s34
      0xBEFE0120u, // 0x1C s_mov_b64 exec, s[32:33]
      0x7E020287u, // 0x20 v_mov_b32 v1, 7
  };
}

class CbranchForkJoinTest : public ::testing::TestWithParam<rj_code_arch_t> {
protected:
  void SetUp() override {
    ComputeUnitCore::Config config{};
    config.arch = GetParam();
    // Common configuration for all tests.
    config.num_wf_slots = 2;
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

  /// Execute without fetching at the deliberately high PC used by target-width tests.
  void execute(uint32_t word) {
    auto decoder = Decoder::create(GetParam());
    ASSERT_NE(decoder, nullptr);
    const std::array<uint32_t, 4> words{word, 0, 0, 0};
    auto decoded = decoder->decode(words.data());
    ASSERT_FALSE(decoded.failed());
    Instruction &inst = *decoded.value();
    ASSERT_EQ(inst.size(), 4u);
    ASSERT_TRUE(cu->execute_instruction(&inst, *wf).succeeded());
    wf->pc += inst.size(); // Match the normal CU post-execution increment.
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
  uint32_t vgpr(uint32_t index, uint32_t lane) {
    return cu->read_vgpr(wf->vgpr_alloc().base + index, lane);
  }
  void fill_vgpr(uint32_t index, uint32_t value) {
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      cu->write_vgpr(wf->vgpr_alloc().base + index, lane, value);
  }

  /// Run until the PC reaches @p end_offset; returns the offset of each executed instruction.
  std::vector<uint64_t> run_until(uint64_t end_offset, size_t max_steps = 32) {
    std::vector<uint64_t> trace;
    while (wf->pc != kCodeBase + end_offset && trace.size() < max_steps) {
      trace.push_back(wf->pc - kCodeBase);
      (void)cu->step();
      if (wf->is_halted())
        break;
    }
    EXPECT_FALSE(wf->is_halted());
    EXPECT_EQ(wf->pc, kCodeBase + end_offset);
    return trace;
  }

  /// Let the wave execute scalar loads from identity-mapped memory.
  void enable_scalar_loads() {
    l2.set_backing_memory(&memory);
    cu->set_gpu_vm(&vm);
    auto space = vm.register_address_space(0, std::make_shared<IdentityAddressSpaceTranslator>(),
                                           std::make_shared<GpuMemoryPhysicalAccess>(memory));
    ASSERT_TRUE(space);
    wf->set_address_space(space);
  }

  GpuMemory memory{"cbranch_fork_join_memory"};
  L2Cache l2{"cbranch_fork_join_l2"};
  GpuVm vm;
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
    EXPECT_EQ(branch_stack_entry_physical_sgpr(*wf, n), wf->sgpr_alloc().base + 4 * n);
}

TEST_P(CbranchForkJoinTest, ForkAndJoinUseTheSecondWavesPhysicalStack) {
  Wavefront *first_wave = wf;
  for (uint32_t index = 0; index < 32; ++index)
    set_sgpr(index, 0x12340000u + index);

  wf = cu->dispatch_wf(1, kCodeBase, 104, 256);
  ASSERT_NE(wf, nullptr);
  ASSERT_NE(wf->sgpr_alloc().base, 0u);
  for (uint32_t index = 0; index < 32; ++index)
    set_sgpr(index, 0x56780000u + index);
  wf->set_cond_stack_ptr(3);

  const auto push = fork_stack_access(*wf, 0x15);
  ASSERT_TRUE(push);
  EXPECT_EQ(push->first_wave_sgpr, 12u);
  EXPECT_TRUE(push->write);
  EXPECT_EQ(branch_stack_entry_physical_sgpr(*wf, 3), wf->sgpr_alloc().base + 12);
  EXPECT_EQ(cbranch_fork(*wf, 0x15, kTarget, kNextPc), kTarget);
  EXPECT_EQ(wf->cond_stack_ptr(), 4u);
  EXPECT_EQ(sgpr64(12), ~uint64_t{0x15});
  EXPECT_EQ(sgpr64(14), kNextPc);

  const auto pop = join_stack_access(*wf, 3);
  ASSERT_TRUE(pop);
  EXPECT_EQ(pop->first_wave_sgpr, 12u);
  EXPECT_FALSE(pop->write);
  EXPECT_EQ(cbranch_join(*wf, 3, kTarget), kNextPc);
  EXPECT_EQ(wf->exec(), ~uint64_t{0x15});
  EXPECT_EQ(wf->cond_stack_ptr(), 3u);
  for (uint32_t index = 0; index < 32; ++index) {
    EXPECT_EQ(cu->read_sgpr(first_wave->sgpr_alloc().base + index), 0x12340000u + index);
    if (index < 12 || index >= 16) {
      EXPECT_EQ(sgpr(index), 0x56780000u + index);
    }
  }
  EXPECT_EQ(cu->isa_diagnostic_count(), 0u);
  first_wave->halt();
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
  EXPECT_EQ(cu->isa_diagnostic_count(), 1u);
}

TEST_P(CbranchForkJoinTest, IForkRunsTheSmallerPassSideThenTheSavedFailSide) {
  constexpr uint64_t kPass = 0x15; // 3 lanes pass, 61 fail
  write_program(if_else_program(kIForkVccThen));
  wf->set_vcc(kPass);
  fill_vgpr(0, 0);

  EXPECT_EQ(run_until(kEndOffset),
            (std::vector<uint64_t>{0x00, 0x04, kForkOffset, kThenOffset, kJoinOffset, kElseOffset,
                                   0x10, kJoinOffset, 0x1C, 0x20}));
  for (uint32_t lane = 0; lane < 64; ++lane) {
    const bool passed = (kPass >> lane) & 1;
    EXPECT_EQ(vgpr(0, lane), passed ? kThenValue : kElseValue) << "lane " << lane;
    EXPECT_EQ(vgpr(1, lane), kAfterJoinValue) << "lane " << lane;
  }
  EXPECT_EQ(wf->exec(), ~uint64_t{0});
  EXPECT_EQ(wf->cond_stack_ptr(), 0u);
}

TEST_P(CbranchForkJoinTest, IForkRunsTheSmallerFailSideFirst) {
  constexpr uint64_t kFail = 0x15; // 3 lanes fail, 61 pass
  write_program(if_else_program(kIForkVccThen));
  wf->set_vcc(~kFail);
  fill_vgpr(0, 0);

  EXPECT_EQ(run_until(kEndOffset),
            (std::vector<uint64_t>{0x00, 0x04, kForkOffset, kElseOffset, 0x10, kJoinOffset,
                                   kThenOffset, kJoinOffset, 0x1C, 0x20}));
  for (uint32_t lane = 0; lane < 64; ++lane) {
    const bool failed = (kFail >> lane) & 1;
    EXPECT_EQ(vgpr(0, lane), failed ? kElseValue : kThenValue) << "lane " << lane;
    EXPECT_EQ(vgpr(1, lane), kAfterJoinValue) << "lane " << lane;
  }
  EXPECT_EQ(wf->exec(), ~uint64_t{0});
  EXPECT_EQ(wf->cond_stack_ptr(), 0u);
}

TEST_P(CbranchForkJoinTest, UniformIForkSkipsTheStackAndTheElseSide) {
  write_program(if_else_program(kIForkVccThen));
  wf->set_vcc(~uint64_t{0});
  fill_vgpr(0, 0);
  set_sgpr64(0, kStackSentinel);

  EXPECT_EQ(run_until(kEndOffset),
            (std::vector<uint64_t>{0x00, 0x04, kForkOffset, kThenOffset, kJoinOffset, 0x1C, 0x20}));
  for (uint32_t lane = 0; lane < 64; ++lane)
    EXPECT_EQ(vgpr(0, lane), kThenValue) << "lane " << lane;
  EXPECT_EQ(sgpr64(0), kStackSentinel);
  EXPECT_EQ(wf->cond_stack_ptr(), 0u);
}

TEST_P(CbranchForkJoinTest, GForkTakesItsTargetFromAnSgprPair) {
  constexpr uint64_t kPass = 0x15;
  write_program(if_else_program(kGForkVccS36));
  wf->set_vcc(kPass);
  set_sgpr64(36, kCodeBase + kThenOffset);
  fill_vgpr(0, 0);

  EXPECT_EQ(run_until(kEndOffset),
            (std::vector<uint64_t>{0x00, 0x04, kForkOffset, kThenOffset, kJoinOffset, kElseOffset,
                                   0x10, kJoinOffset, 0x1C, 0x20}));
  for (uint32_t lane = 0; lane < 64; ++lane) {
    const bool passed = (kPass >> lane) & 1;
    EXPECT_EQ(vgpr(0, lane), passed ? kThenValue : kElseValue) << "lane " << lane;
  }
  EXPECT_EQ(wf->exec(), ~uint64_t{0});
  EXPECT_EQ(wf->cond_stack_ptr(), 0u);
}

TEST_P(CbranchForkJoinTest, GetpcRoundTripsThroughGForkWithLoadBias) {
  constexpr uint64_t kHighPc = 0x0000FFFF12345000ULL;
  constexpr uint64_t kExport = 0xFFFFFFFF12345004ULL;
  constexpr uint64_t kForkPc = 0x2000;
  constexpr uint64_t kFewLanes = 0x15;
  for (uint64_t bias : {uint64_t{0}, uint64_t{0x00007FFF00000000ULL}}) {
    SCOPED_TRACE(::testing::Message() << "bias=" << std::hex << bias);
    wf->set_code_load_bias(bias);
    for (uint64_t compare : {~uint64_t{0}, uint64_t{0}, kFewLanes, ~kFewLanes}) {
      SCOPED_TRACE(::testing::Message() << "compare=" << std::hex << compare);
      wf->set_exec(~uint64_t{0});
      wf->set_cond_stack_ptr(0);
      set_sgpr64(0, kStackSentinel);
      set_sgpr64(2, kStackSentinel);
      wf->pc = kHighPc;
      ASSERT_NO_FATAL_FAILURE(execute(build_s_getpc_b64(36, GetParam())));
      ASSERT_EQ(sgpr64(36), kExport);

      wf->pc = kForkPc;
      wf->set_vcc(compare);
      ASSERT_NO_FATAL_FAILURE(execute(kGForkVccS36));
      const bool pass_runs_first = compare == ~uint64_t{0} || compare == kFewLanes;
      EXPECT_EQ(wf->pc, pass_runs_first ? kHighPc + 4 : kForkPc + 4);
      EXPECT_EQ(sgpr64(36), kExport);
      if (compare == 0 || compare == ~uint64_t{0}) {
        EXPECT_EQ(wf->exec(), ~uint64_t{0});
        EXPECT_EQ(wf->cond_stack_ptr(), 0u);
        EXPECT_EQ(sgpr64(0), kStackSentinel);
        EXPECT_EQ(sgpr64(2), kStackSentinel);
      } else {
        EXPECT_EQ(wf->exec(), kFewLanes);
        EXPECT_EQ(wf->cond_stack_ptr(), 1u);
        EXPECT_EQ(sgpr64(0), ~kFewLanes);
        EXPECT_EQ(sgpr64(2), pass_runs_first ? kForkPc + 4 : kHighPc + 4);
        set_sgpr(34, 0);                               // Saved CSP for the matching join.
        ASSERT_NO_FATAL_FAILURE(execute(0xBE802E22u)); // s_cbranch_join s34
        EXPECT_EQ(wf->pc, pass_runs_first ? kForkPc + 4 : kHighPc + 4);
        EXPECT_EQ(wf->exec(), ~kFewLanes);
        EXPECT_EQ(wf->cond_stack_ptr(), 0u);
      }
    }
  }
}

TEST_P(CbranchForkJoinTest, GForkPreservesLinkedPositiveRel32Targets) {
  constexpr uint64_t kPositiveTag = 0x0001FFFF00001000ULL;
  for (uint64_t bias : {uint64_t{0}, uint64_t{0x00007FFF00000000ULL}}) {
    SCOPED_TRACE(::testing::Message() << "bias=" << std::hex << bias);
    wf->set_code_load_bias(bias);
    wf->set_exec(~uint64_t{0});
    wf->set_vcc(~uint64_t{0});
    wf->pc = kCodeBase;
    set_sgpr64(36, kPositiveTag);
    ASSERT_NO_FATAL_FAILURE(execute(kGForkVccS36));
    EXPECT_EQ(wf->pc, bias ? bias + 0x1000 : 0x0000FFFF00001000ULL);
    EXPECT_EQ(sgpr64(36), kPositiveTag);
    EXPECT_EQ(wf->cond_stack_ptr(), 0u);
  }
}

TEST_P(CbranchForkJoinTest, ForkAndJoinCarryControlFlowMetadata) {
  auto decoder = Decoder::create(GetParam());
  ASSERT_NE(decoder, nullptr);
  constexpr uint64_t kCfgFlags = BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL;
  const auto check = [&](uint32_t word, uint64_t cfg_flag, std::optional<int64_t> offset,
                         bool is_fork) {
    const std::array<uint32_t, 4> words{word, 0, 0, 0};
    auto decoded = decoder->decode(words.data());
    ASSERT_FALSE(decoded.failed()) << std::hex << word;
    const Instruction &inst = *decoded.value();
    SCOPED_TRACE(inst.mnemonic());
    EXPECT_EQ(inst.flags() & kCfgFlags, cfg_flag);
    EXPECT_NE(inst.flags() & IGNORES_EXEC, 0u);
    EXPECT_NE(inst.flags() & WRITES_EXEC, 0u);
    EXPECT_NE(inst.flags() & BRANCH_STACK, 0u);
    EXPECT_EQ(inst.branch_offset_bytes(), offset);

    // A fork may write any stack entry, so its defs must not kill; a join may
    // read any of them.
    EXPECT_EQ((inst.flags() & PREDICATED_DEF) != 0, is_fork);
    RegisterSet defs;
    RegisterSet uses;
    inst.implicit_defs(defs);
    inst.implicit_uses(uses);
    const RegisterSet &stack = is_fork ? defs : uses;
    EXPECT_TRUE(stack.contains(RegisterRef{RegClass::SGPR, 0, 1}));
    EXPECT_TRUE(stack.contains(RegisterRef{RegClass::SGPR, 31, 1}));
    EXPECT_FALSE(stack.contains(RegisterRef{RegClass::SGPR, 32, 1}));
    EXPECT_FALSE((is_fork ? uses : defs).contains(RegisterRef{RegClass::SGPR, 0, 1}));
  };
  check(kIForkVccThen, COND_BRANCH, int64_t{8}, /*is_fork=*/true);
  check(kGForkVccS36, INDIRECT_BRANCH, std::nullopt, /*is_fork=*/true);
  check(0xBE802E22u, INDIRECT_BRANCH, std::nullopt, /*is_fork=*/false); // s_cbranch_join s34
}

TEST_P(CbranchForkJoinTest, NestedIForksUnwindInnerThenOuter) {
  // Per lane:
  //   if (vcc) { v0 = s[38:39] lane set ? 4 : 3; } else { v0 = 2; }
  //   v1 = 7;
  write_program({
      // Outer block: fork on vcc.
      0xB8A21741u, // 0x00             s_getreg_b32     s34, hwreg(HW_REG_MODE, 29, 3) ; CSP
      0xBEA0017Eu, // 0x04             s_mov_b64        s[32:33], exec                 ; EXEC
      0xB86A0002u, // 0x08             s_cbranch_i_fork vcc, OUTER_THEN
      0x7E000282u, // 0x0C OUTER_ELSE: v_mov_b32        v0, 2
      0xBF820008u, // 0x10             s_branch         OUTER_JOIN

      // Inner block, nested in OUTER_THEN: fork on s[38:39].
      0xB8A31741u, // 0x14     OUTER_THEN: s_getreg_b32     s35, hwreg(HW_REG_MODE, 29, 3) ; CSP
      0xBEA4017Eu, // 0x18                 s_mov_b64        s[36:37], exec                 ; EXEC
      0xB8260002u, // 0x1C                 s_cbranch_i_fork s[38:39], INNER_THEN
      0x7E000283u, // 0x20     INNER_ELSE: v_mov_b32        v0, 3
      0xBF820001u, // 0x24                 s_branch         INNER_JOIN
      0x7E000284u, // 0x28     INNER_THEN: v_mov_b32        v0, 4
      0xBE802E23u, // 0x2C     INNER_JOIN: s_cbranch_join   s35
      0xBEFE0124u, // 0x30                 s_mov_b64        exec, s[36:37]                 ; EXEC

      // Back in the outer block.
      0xBE802E22u, // 0x34 OUTER_JOIN: s_cbranch_join   s34
      0xBEFE0120u, // 0x38             s_mov_b64        exec, s[32:33]                 ; EXEC
      0x7E020287u, // 0x3C             v_mov_b32        v1, 7
  });
  wf->set_vcc(0x0F);    // lanes 0-3 take the outer THEN side
  set_sgpr64(38, 0x01); // lane 0 takes the inner THEN side
  fill_vgpr(0, 0);

  EXPECT_EQ(run_until(0x40),
            (std::vector<uint64_t>{0x00, 0x04, 0x08, 0x14, 0x18, 0x1C, 0x28, 0x2C, 0x20, 0x24, 0x2C,
                                   0x30, 0x34, 0x0C, 0x10, 0x34, 0x38, 0x3C}));
  for (uint32_t lane = 0; lane < 64; ++lane) {
    const uint32_t expected = lane == 0 ? 4u : lane < 4 ? 3u : 2u;
    EXPECT_EQ(vgpr(0, lane), expected) << "lane " << lane;
    EXPECT_EQ(vgpr(1, lane), kAfterJoinValue) << "lane " << lane;
  }
  EXPECT_EQ(wf->exec(), ~uint64_t{0});
  EXPECT_EQ(wf->cond_stack_ptr(), 0u);
  EXPECT_EQ(cu->isa_diagnostic_count(), 0u);
}

TEST_P(CbranchForkJoinTest, DebugStopRestoresCsp) {
  wf->set_cond_stack_ptr(3);
  const auto saved = wf->debug_stop_state();
  wf->set_cond_stack_ptr(0);
  wf->restore_debug_stop_state(saved);
  EXPECT_EQ(wf->cond_stack_ptr(), 3u);
}

TEST_P(CbranchForkJoinTest, ValidStackUseReportsNoDiagnostics) {
  // The sixth and deepest documented entry, s[20:23], then its join.
  wf->set_cond_stack_ptr(5);
  EXPECT_EQ(cbranch_fork(*wf, 0x15, kTarget, kNextPc), kTarget);
  EXPECT_EQ(cbranch_join(*wf, /*saved_csp=*/5, kNextPc), kNextPc);
  EXPECT_EQ(wf->cond_stack_ptr(), 5u);
  EXPECT_EQ(cu->isa_diagnostic_count(), 0u);
}

TEST_P(CbranchForkJoinTest, ForkPastSixEntriesReportsUndefinedBehavior) {
  wf->set_cond_stack_ptr(6);
  EXPECT_EQ(cbranch_fork(*wf, 0x15, kTarget, kNextPc), kTarget);
  EXPECT_EQ(wf->cond_stack_ptr(), 7u);
  EXPECT_EQ(cu->isa_diagnostic_count(), 1u);
}

TEST_P(CbranchForkJoinTest, JoinBelowSavedCspReportsUndefinedBehavior) {
  // CSP 0 against a saved 2: no entry of the matching fork is left. The pop
  // still happens and wraps CSP to 7, as the 3-bit field does.
  EXPECT_EQ(wf->cond_stack_ptr(), 0u);
  (void)cbranch_join(*wf, /*saved_csp=*/2, kNextPc);
  EXPECT_EQ(wf->cond_stack_ptr(), 7u);
  EXPECT_EQ(cu->isa_diagnostic_count(), 1u);
}

TEST_P(CbranchForkJoinTest, StackEntryOutsideSgprAllocationReportsUndefinedBehavior) {
  // A 16-SGPR wave owns s[0:15]; entry 4 would be s[16:19].
  wf->halt();
  wf = cu->dispatch_wf(1, kCodeBase, 16, 256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(~uint64_t{0});
  wf->set_cond_stack_ptr(4);

  (void)cbranch_fork(*wf, 0x15, kTarget, kNextPc);
  EXPECT_EQ(cu->isa_diagnostic_count(), 1u);
  (void)cbranch_join(*wf, /*saved_csp=*/4, kNextPc);
  EXPECT_EQ(cu->isa_diagnostic_count(), 2u);
}

// Each program below is: a scalar load whose result is still pending, the fork
// or join at 0x08, then (where present) a read of the load's destination at 0x0C.
constexpr uint32_t kLoadS20 = 0xC0020514u;     // s_load_dword s20, s[40:41], 0x0
constexpr uint32_t kLoadS14 = 0xC0020394u;     // s_load_dword s14, s[40:41], 0x0
constexpr uint32_t kReadS2 = 0xBE950002u;      // s_mov_b32 s21, s2
constexpr uint32_t kLoadS2 = 0xC0020094u;      // s_load_dword s2, s[40:41], 0x0
constexpr uint32_t kReadS20 = 0xBE950014u;     // s_mov_b32 s21, s20
constexpr uint32_t kForkVccNext = 0xB86A0000u; // s_cbranch_i_fork vcc, 0 (both sides at 0x0C)
constexpr uint32_t kJoinS34 = 0xBE802E22u;     // s_cbranch_join s34
constexpr uint64_t kLoadAddress = 0x300000;
constexpr uint64_t kAfterStackOp = kCodeBase + 0x0C;

TEST_P(CbranchForkJoinTest, WaitScoreboardSeesOnlyThePoppedJoinEntry) {
  // The join pops entry 0, s[0:3]; s20 is not part of it.
  write_program({kLoadS20, 0u, kJoinS34, kReadS20});
  ASSERT_NO_FATAL_FAILURE(enable_scalar_loads());
  set_sgpr64(40, kLoadAddress);
  wf->set_cond_stack_ptr(1);
  set_sgpr(34, 0);
  set_sgpr64(0, ~uint64_t{0});
  set_sgpr64(2, kAfterStackOp);

  (void)cu->step();
  (void)cu->step();
  EXPECT_EQ(wf->pc, kAfterStackOp);
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 0u);
  // The pending s20 result was not consumed, so the real unwaited read reports.
  (void)cu->step();
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 1u);
}

TEST_P(CbranchForkJoinTest, WaitScoreboardSeesNoStackAccessForAUniformFork) {
  write_program({kLoadS20, 0u, kForkVccNext, kReadS20});
  ASSERT_NO_FATAL_FAILURE(enable_scalar_loads());
  set_sgpr64(40, kLoadAddress);
  wf->set_vcc(~uint64_t{0});

  (void)cu->step();
  (void)cu->step();
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 0u);
  (void)cu->step();
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 1u);
}

TEST_P(CbranchForkJoinTest, WaitScoreboardSeesOnlyTheEntryADivergentForkPushes) {
  // The fork pushes entry 0, s[0:3]; s20 is not part of it.
  write_program({kLoadS20, 0u, kForkVccNext, kReadS20});
  ASSERT_NO_FATAL_FAILURE(enable_scalar_loads());
  set_sgpr64(40, kLoadAddress);
  wf->set_vcc(0x15);

  (void)cu->step();
  (void)cu->step();
  EXPECT_EQ(wf->cond_stack_ptr(), 1u);
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 0u);
  (void)cu->step();
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 1u);
}

TEST_P(CbranchForkJoinTest, WaitScoreboardReportsAForkOverwritingAPendingEntry) {
  write_program({kLoadS2, 0u, kForkVccNext});
  ASSERT_NO_FATAL_FAILURE(enable_scalar_loads());
  set_sgpr64(40, kLoadAddress);
  wf->set_vcc(0x15);

  (void)cu->step();
  (void)cu->step();
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 1u);
}

TEST_P(CbranchForkJoinTest, WaitScoreboardReportsAJoinReadingAPendingEntry) {
  write_program({kLoadS2, 0u, kJoinS34});
  ASSERT_NO_FATAL_FAILURE(enable_scalar_loads());
  set_sgpr64(40, kLoadAddress);
  // s2 holds the low half of the resume PC, so give the load the same value.
  memory.write32(kLoadAddress, static_cast<uint32_t>(kAfterStackOp));
  wf->set_cond_stack_ptr(1);
  set_sgpr(34, 0);
  set_sgpr64(0, ~uint64_t{0});
  set_sgpr64(2, kAfterStackOp);

  (void)cu->step();
  (void)cu->step();
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 1u);
}

TEST_P(CbranchForkJoinTest, WaitScoreboardReportsAForkOverwritingADeeperPendingEntry) {
  // CSP 3 selects s[12:15], including the pending s14 result.
  write_program({kLoadS14, 0u, kForkVccNext});
  ASSERT_NO_FATAL_FAILURE(enable_scalar_loads());
  set_sgpr64(40, kLoadAddress);
  wf->set_cond_stack_ptr(3);
  wf->set_vcc(0x15);

  (void)cu->step();
  (void)cu->step();
  EXPECT_EQ(wf->pc, kAfterStackOp);
  EXPECT_EQ(wf->cond_stack_ptr(), 4u);
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 1u);
}

TEST_P(CbranchForkJoinTest, WaitScoreboardReportsAJoinReadingADeeperPendingEntry) {
  // CSP 4 pops s[12:15], including the pending resume PC in s14.
  write_program({kLoadS14, 0u, kJoinS34});
  ASSERT_NO_FATAL_FAILURE(enable_scalar_loads());
  set_sgpr64(40, kLoadAddress);
  memory.write32(kLoadAddress, static_cast<uint32_t>(kAfterStackOp));
  wf->set_cond_stack_ptr(4);
  set_sgpr(34, 3);
  set_sgpr64(12, ~uint64_t{0});
  set_sgpr64(14, kAfterStackOp);

  (void)cu->step();
  (void)cu->step();
  EXPECT_EQ(wf->pc, kAfterStackOp);
  EXPECT_EQ(wf->cond_stack_ptr(), 3u);
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 1u);
}

TEST_P(CbranchForkJoinTest, WaitScoreboardSeesNoStackAccessForAFallthroughJoin) {
  // Entry 0 contains a pending s2 result, but equal CSPs mean there is no pop.
  write_program({kLoadS2, 0u, kJoinS34, kReadS2});
  ASSERT_NO_FATAL_FAILURE(enable_scalar_loads());
  set_sgpr64(40, kLoadAddress);
  wf->set_cond_stack_ptr(1);
  set_sgpr(34, 1);

  (void)cu->step();
  (void)cu->step();
  EXPECT_EQ(wf->pc, kAfterStackOp);
  EXPECT_EQ(wf->cond_stack_ptr(), 1u);
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 0u);
  (void)cu->step();
  EXPECT_EQ(cu->memory_wait_diagnostic_count(), 1u);
}

// MODE.CSP and the fork/join instructions exist only on CDNA1-4.
INSTANTIATE_TEST_SUITE_P(Cdna, CbranchForkJoinTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                           ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4));

} // namespace
