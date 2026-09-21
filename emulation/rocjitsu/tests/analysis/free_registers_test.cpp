// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Direct coverage for the free-register search shared by LivenessAnalysis and
// the DBI trampoline builder. Its callers reach it with different bounds
// (encoding width, a kernel's own .sgpr_count, the cross-family allocatable
// limit), so the bound handling is what these tests pin down.

#include "rocjitsu/code/analysis/free_registers.h"
#include "rocjitsu/isa/register_set.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace rocjitsu {
namespace {

[[nodiscard]] RegisterSet sgprs(std::initializer_list<uint16_t> indices) {
  RegisterSet set;
  for (uint16_t index : indices)
    set.expand(RegisterRef{RegClass::SGPR, index, 1});
  return set;
}

TEST(FreeRegisters, FindsLowestQualifyingBase) {
  EXPECT_EQ(find_free_run(sgprs({0, 1}), RegClass::SGPR, 1, 0, 1, 16), 2);
  EXPECT_EQ(find_free_run(RegisterSet{}, RegClass::SGPR, 1, 0, 1, 16), 0);
}

TEST(FreeRegisters, RunMustBeWhollyFree) {
  // s2 blocks the pair based at 2; the search continues rather than reporting a
  // partially free run.
  EXPECT_EQ(find_free_run(sgprs({0, 2}), RegClass::SGPR, 2, 0, 1, 16), 3);
}

TEST(FreeRegisters, SearchStartIsRoundedUpToAlignment) {
  // An odd search_start with an even alignment must not yield an odd base.
  EXPECT_EQ(find_free_run(RegisterSet{}, RegClass::SGPR, 2, /*search_start=*/3,
                          /*base_alignment=*/2, 16),
            4);
}

TEST(FreeRegisters, BoundIsExclusiveAndCoversTheWholeRun) {
  // A pair based at 14 occupies s14 and s15, so it fits under bound 16 but not 15.
  const RegisterSet occupied = sgprs({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
  EXPECT_EQ(find_free_run(occupied, RegClass::SGPR, 2, 0, 2, 16), 14);
  EXPECT_EQ(find_free_run(occupied, RegClass::SGPR, 2, 0, 2, 15), std::nullopt)
      << "the run's last register must also be below the bound";
}

TEST(FreeRegisters, ZeroBoundFindsNothing) {
  EXPECT_EQ(find_free_run(RegisterSet{}, RegClass::SGPR, 1, 0, 1, 0), std::nullopt);
}

TEST(FreeRegisters, SearchStartPastBoundFindsNothing) {
  EXPECT_EQ(find_free_run(RegisterSet{}, RegClass::SGPR, 1, /*search_start=*/32,
                          /*base_alignment=*/1, 16),
            std::nullopt);
}

TEST(FreeRegisters, FullSetFindsNothing) {
  RegisterSet all;
  for (uint16_t i = 0; i < 16; ++i)
    all.expand(RegisterRef{RegClass::SGPR, i, 1});
  EXPECT_EQ(find_free_run(all, RegClass::SGPR, 1, 0, 1, 16), std::nullopt);
}

TEST(FreeRegisters, SgprPairIsEvenAligned) {
  // s1 alone is free below s2, but a pair may not start there.
  EXPECT_EQ(find_free_sgpr(sgprs({0}), 16), 1);
  EXPECT_EQ(find_free_sgpr_pair(sgprs({0}), 16), 2);
}

TEST(FreeRegisters, SgprHelpersHonorSearchStart) {
  EXPECT_EQ(find_free_sgpr(RegisterSet{}, 16, /*search_start=*/5), 5);
  EXPECT_EQ(find_free_sgpr_pair(RegisterSet{}, 16, /*search_start=*/5), 6);
}

TEST(FreeRegisters, SgprHelpersRespectTheBound) {
  const RegisterSet occupied = sgprs({0, 1, 2, 3});
  EXPECT_EQ(find_free_sgpr(occupied, 4), std::nullopt);
  EXPECT_EQ(find_free_sgpr(occupied, 5), 4);
  EXPECT_EQ(find_free_sgpr_pair(occupied, 5), std::nullopt)
      << "s[4:5] needs bound 6; bound 5 admits only s4";
  EXPECT_EQ(find_free_sgpr_pair(occupied, 6), 4);
}

TEST(FreeRegisters, VgprRunsAreSearchedIndependently) {
  RegisterSet set;
  set.expand(RegisterRef{RegClass::VGPR, 0, 1});
  EXPECT_EQ(find_free_run(set, RegClass::VGPR, 1, 0, 1, 8), 1);
  EXPECT_EQ(find_free_run(set, RegClass::SGPR, 1, 0, 1, 8), 0);
}

TEST(FreeRegisters, AlignedRunSkipsMisalignedGaps) {
  // s[2:3] is free but a 4-aligned quad must start at 4.
  const RegisterSet occupied = sgprs({0, 1, 4});
  EXPECT_EQ(find_free_run(occupied, RegClass::SGPR, 2, 0, /*base_alignment=*/4, 16), 8);
}

} // namespace
} // namespace rocjitsu
