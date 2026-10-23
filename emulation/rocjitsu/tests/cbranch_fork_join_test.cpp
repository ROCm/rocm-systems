// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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

// MODE.CSP and the fork/join instructions exist only on GFX9-derived ISAs.
INSTANTIATE_TEST_SUITE_P(Cdna, CbranchForkJoinTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                           ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4));

} // namespace
