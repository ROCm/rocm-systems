// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Loader fallback tests: a bogus primary soname followed by a valid candidate
// must still resolve. No GPU required.

#include <gtest/gtest.h>

#include <vector>

#include "amd_smi/impl/amd_smi_lib_loader.h"
#include "unit_fixtures.h"

namespace {

TEST_F(SystemUnit, LibLoaderFirstCandidateWins) {
  amd::smi::AMDSmiLibraryLoader loader;
  EXPECT_EQ(loader.load(std::vector<const char*>{"libm.so.6", "does_not_exist.so"}),
            AMDSMI_STATUS_SUCCESS);
}

TEST_F(SystemUnit, LibLoaderFallsBackWhenPrimaryMissing) {
  amd::smi::AMDSmiLibraryLoader loader;
  EXPECT_EQ(loader.load(std::vector<const char*>{"does_not_exist.so.9", "libm.so.6"}),
            AMDSMI_STATUS_SUCCESS);
}

TEST_F(SystemUnit, LibLoaderAllCandidatesMissingFails) {
  amd::smi::AMDSmiLibraryLoader loader;
  EXPECT_EQ(loader.load(std::vector<const char*>{"nope_a.so.9", "nope_b.so.9"}),
            AMDSMI_STATUS_FAIL_LOAD_MODULE);
}

// When a candidate is already loaded in the process, the loader must still keep
// a usable handle so load_symbol() resolves.
TEST_F(SystemUnit, LibLoaderAlreadyLoadedResolvesSymbol) {
  amd::smi::AMDSmiLibraryLoader keep_open;
  ASSERT_EQ(keep_open.load("libm.so.6"), AMDSMI_STATUS_SUCCESS);

  amd::smi::AMDSmiLibraryLoader loader;
  ASSERT_EQ(loader.load(std::vector<const char*>{"libm.so.6"}), AMDSMI_STATUS_SUCCESS);
  double (*cos_fn)(double) = nullptr;
  EXPECT_EQ(loader.load_symbol(&cos_fn, "cos"), AMDSMI_STATUS_SUCCESS);
  EXPECT_NE(cos_fn, nullptr);
}

}  // namespace
