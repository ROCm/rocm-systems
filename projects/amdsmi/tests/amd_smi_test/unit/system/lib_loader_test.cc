// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Loader fallback tests: a bogus primary soname followed by a valid candidate
// must still resolve. No GPU required.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_lib_loader.h"

namespace {

TEST(SystemUnit, LibLoaderFirstCandidateWins) {
  amd::smi::AMDSmiLibraryLoader loader;
  EXPECT_EQ(loader.load(std::vector<const char*>{"libm.so.6", "does_not_exist.so"}),
            AMDSMI_STATUS_SUCCESS);
}

TEST(SystemUnit, LibLoaderFallsBackWhenPrimaryMissing) {
  amd::smi::AMDSmiLibraryLoader loader;
  EXPECT_EQ(loader.load(std::vector<const char*>{"does_not_exist.so.9", "libm.so.6"}),
            AMDSMI_STATUS_SUCCESS);
}

TEST(SystemUnit, LibLoaderAllCandidatesMissingFails) {
  amd::smi::AMDSmiLibraryLoader loader;
  testing::internal::CaptureStderr();
  amdsmi_status_t status = loader.load(std::vector<const char*>{"nope_a.so.9", "nope_b.so.9"});
  std::string logged = testing::internal::GetCapturedStderr();
  EXPECT_EQ(status, AMDSMI_STATUS_FAIL_LOAD_MODULE);
  EXPECT_FALSE(logged.empty());
}

// log_on_failure=false must still report the same failure status as the
// logging path above, and must not write to stderr.
TEST(SystemUnit, LibLoaderAllCandidatesMissingFailsSilently) {
  amd::smi::AMDSmiLibraryLoader loader;
  testing::internal::CaptureStderr();
  amdsmi_status_t status = loader.load(std::vector<const char*>{"nope_a.so.9", "nope_b.so.9"},
                                       /*log_on_failure=*/false);
  std::string logged = testing::internal::GetCapturedStderr();
  EXPECT_EQ(status, AMDSMI_STATUS_FAIL_LOAD_MODULE);
  EXPECT_TRUE(logged.empty());
}

TEST(SystemUnit, LibLoaderLoadSymbolMissingFailsWithLog) {
  amd::smi::AMDSmiLibraryLoader loader;
  ASSERT_EQ(loader.load(std::vector<const char*>{"libm.so.6"}), AMDSMI_STATUS_SUCCESS);
  void (*missing_fn)() = nullptr;
  testing::internal::CaptureStderr();
  amdsmi_status_t status = loader.load_symbol(&missing_fn, "does_not_exist_symbol");
  std::string logged = testing::internal::GetCapturedStderr();
  EXPECT_EQ(status, AMDSMI_STATUS_FAIL_LOAD_SYMBOL);
  EXPECT_FALSE(logged.empty());
}

// log_on_failure=false must still report the same failure status as the
// logging path above, and must not write to stderr.
TEST(SystemUnit, LibLoaderLoadSymbolMissingFailsSilently) {
  amd::smi::AMDSmiLibraryLoader loader;
  ASSERT_EQ(loader.load(std::vector<const char*>{"libm.so.6"}), AMDSMI_STATUS_SUCCESS);
  void (*missing_fn)() = nullptr;
  testing::internal::CaptureStderr();
  amdsmi_status_t status =
      loader.load_symbol(&missing_fn, "does_not_exist_symbol", /*log_on_failure=*/false);
  std::string logged = testing::internal::GetCapturedStderr();
  EXPECT_EQ(status, AMDSMI_STATUS_FAIL_LOAD_SYMBOL);
  EXPECT_TRUE(logged.empty());
}

// When a candidate is already loaded in the process, the loader must still keep
// a usable handle so load_symbol() resolves.
TEST(SystemUnit, LibLoaderAlreadyLoadedResolvesSymbol) {
  amd::smi::AMDSmiLibraryLoader keep_open;
  ASSERT_EQ(keep_open.load("libm.so.6"), AMDSMI_STATUS_SUCCESS);

  amd::smi::AMDSmiLibraryLoader loader;
  ASSERT_EQ(loader.load(std::vector<const char*>{"libm.so.6"}), AMDSMI_STATUS_SUCCESS);
  double (*cos_fn)(double) = nullptr;
  EXPECT_EQ(loader.load_symbol(&cos_fn, "cos"), AMDSMI_STATUS_SUCCESS);
  EXPECT_NE(cos_fn, nullptr);
}

}  // namespace
