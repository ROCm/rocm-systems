// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Hardware-free unit tests for the fwupd UMA carveout adapter.
//
// The two entry-point tests exercise the real fwupd_* functions: on any CI host
// (no APU carveout firmware, and usually no fwupd daemon or D-Bus system bus)
// they report NOT_SUPPORTED; on affected hardware they resolve the setting, so
// those outcomes are accepted and sanity-checked rather than hard-failed.
//
// The remaining tests cover the pure parsing/mapping helpers directly with
// synthetic settings, so option indexing, current-value detection, the
// "unknown" sentinel, count clamping and description truncation are all verified
// without a daemon, GPU, or root.

#include "fwupd_carveout.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "fwupd_carveout_internal.h"

using amd::smi::detail::BiosSetting;
using amd::smi::detail::FindCarveout;
using amd::smi::detail::PopulateCarveoutInfo;
using amd::smi::detail::ValidateCarveoutWrite;

namespace {

BiosSetting MakeCarveout(std::string id, std::string current, std::vector<std::string> values) {
  BiosSetting s;
  s.id = std::move(id);
  s.name = "Dedicated Video Memory";
  s.current = std::move(current);
  s.values = std::move(values);
  return s;
}

}  // namespace

TEST(GpuUnit, FwupdCarveoutGetReportsNotSupportedOrSaneInfo) {
  amdsmi_uma_carveout_info_t info{};
  const amdsmi_status_t ret = amd::smi::fwupd_get_carveout_info(&info);
  if (ret == AMDSMI_STATUS_SUCCESS) {
    EXPECT_GT(info.num_options, 0u);
    EXPECT_LE(info.num_options, static_cast<uint32_t>(AMDSMI_MAX_CARVEOUT_OPTIONS));
    EXPECT_LE(info.current_index, info.num_options);
  } else {
    EXPECT_EQ(ret, AMDSMI_STATUS_NOT_SUPPORTED);
  }
}

TEST(GpuUnit, FwupdCarveoutSetReportsNotSupportedWithoutMutating) {
  setenv("AMDSMI_DRY_RUN", "1", 1);
  const amdsmi_status_t ret = amd::smi::fwupd_set_carveout(0);
  unsetenv("AMDSMI_DRY_RUN");
  EXPECT_TRUE(ret == AMDSMI_STATUS_NOT_SUPPORTED || ret == AMDSMI_STATUS_SUCCESS ||
              ret == AMDSMI_STATUS_NO_PERM)
      << "unexpected status: " << ret;
}

// --- Pure parsing/mapping helpers (no D-Bus, no hardware) --------------------

TEST(GpuUnit, FwupdCarveoutFindPrefersAmdIdOverHp) {
  const std::vector<BiosSetting> settings = {
      MakeCarveout("com.hp-bioscfg.Dedicated_Graphics_Memory", "4 GB", {"512 MB", "4 GB"}),
      MakeCarveout("com.amd-gpu.uma_carveout", "(1 GB)", {"Minimum (512 MB)", "(1 GB)"}),
  };
  const BiosSetting* c = FindCarveout(settings);
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c->id, "com.amd-gpu.uma_carveout");
}

TEST(GpuUnit, FwupdCarveoutFindMatchesIdCaseInsensitively) {
  const std::vector<BiosSetting> settings = {
      MakeCarveout("COM.AMD-GPU.UMA_CARVEOUT", "(1 GB)", {"(1 GB)"}),
  };
  EXPECT_NE(FindCarveout(settings), nullptr);
}

TEST(GpuUnit, FwupdCarveoutFindReturnsNullWhenNoCarveout) {
  const std::vector<BiosSetting> settings = {
      MakeCarveout("com.example.unrelated", "x", {"x", "y"}),
  };
  EXPECT_EQ(FindCarveout(settings), nullptr);
}

TEST(GpuUnit, FwupdCarveoutPopulateIndexesOptionsAndCurrent) {
  const BiosSetting s = MakeCarveout("com.amd-gpu.uma_carveout", "(2 GB)",
                                     {"Minimum (512 MB)", "(1 GB)", "(2 GB)", "High (16 GB)"});
  amdsmi_uma_carveout_info_t info{};
  ASSERT_EQ(PopulateCarveoutInfo(s, &info), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(info.num_options, 4u);
  for (uint32_t i = 0; i < info.num_options; ++i) {
    EXPECT_EQ(info.options[i].index, i);
  }
  EXPECT_STREQ(info.options[0].description, "Minimum (512 MB)");
  EXPECT_STREQ(info.options[2].description, "(2 GB)");
  EXPECT_EQ(info.current_index, 2u);  // "(2 GB)" is index 2
}

TEST(GpuUnit, FwupdCarveoutPopulateMarksCurrentUnknownWhenRedacted) {
  // An unprivileged reader gets the option list but an empty current value.
  const BiosSetting s = MakeCarveout("com.amd-gpu.uma_carveout", "", {"a", "b", "c"});
  amdsmi_uma_carveout_info_t info{};
  ASSERT_EQ(PopulateCarveoutInfo(s, &info), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(info.num_options, 3u);
  EXPECT_EQ(info.current_index, info.num_options);  // sentinel: unknown
}

TEST(GpuUnit, FwupdCarveoutPopulateReportsNotSupportedForEmptyOptions) {
  const BiosSetting s = MakeCarveout("com.amd-gpu.uma_carveout", "", {});
  amdsmi_uma_carveout_info_t info{};
  EXPECT_EQ(PopulateCarveoutInfo(s, &info), AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, FwupdCarveoutPopulateClampsOptionCountToMax) {
  std::vector<std::string> many;
  for (int i = 0; i < AMDSMI_MAX_CARVEOUT_OPTIONS + 5; ++i) {
    many.push_back("opt" + std::to_string(i));
  }
  const BiosSetting s = MakeCarveout("com.amd-gpu.uma_carveout", "", std::move(many));
  amdsmi_uma_carveout_info_t info{};
  ASSERT_EQ(PopulateCarveoutInfo(s, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.num_options, static_cast<uint32_t>(AMDSMI_MAX_CARVEOUT_OPTIONS));
}

TEST(GpuUnit, FwupdCarveoutPopulateTruncatesLongDescription) {
  const std::string long_desc(AMDSMI_MAX_STRING_LENGTH + 50, 'x');
  const BiosSetting s = MakeCarveout("com.amd-gpu.uma_carveout", "", {long_desc});
  amdsmi_uma_carveout_info_t info{};
  ASSERT_EQ(PopulateCarveoutInfo(s, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(std::string(info.options[0].description).size(),
            static_cast<size_t>(AMDSMI_MAX_STRING_LENGTH - 1));
  EXPECT_EQ(info.options[0].description[AMDSMI_MAX_STRING_LENGTH - 1], '\0');
}

TEST(GpuUnit, FwupdCarveoutValidateRejectsReadOnly) {
  BiosSetting s = MakeCarveout("com.amd-gpu.uma_carveout", "(1 GB)", {"(1 GB)", "(2 GB)"});
  s.read_only = true;
  EXPECT_EQ(ValidateCarveoutWrite(s, 0), AMDSMI_STATUS_NO_PERM);
}

TEST(GpuUnit, FwupdCarveoutValidateRejectsOutOfRangeIndex) {
  const BiosSetting s = MakeCarveout("com.amd-gpu.uma_carveout", "", {"a", "b"});
  EXPECT_EQ(ValidateCarveoutWrite(s, 2), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(ValidateCarveoutWrite(s, 99u), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, FwupdCarveoutValidateRejectsEmptyName) {
  BiosSetting s = MakeCarveout("com.amd-gpu.uma_carveout", "", {"a"});
  s.name = "";
  EXPECT_EQ(ValidateCarveoutWrite(s, 0), AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, FwupdCarveoutValidateAcceptsWritableInRange) {
  const BiosSetting s = MakeCarveout("com.amd-gpu.uma_carveout", "(1 GB)", {"(1 GB)", "(2 GB)"});
  EXPECT_EQ(ValidateCarveoutWrite(s, 1), AMDSMI_STATUS_SUCCESS);
}
