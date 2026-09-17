// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for the pp_power_profile_mode parser behind
// rsmi_dev_power_profile_presets_get() / amdsmi_get_gpu_power_profile_presets().
// Driven over captured sysfs text -- no GPU required. Covers the transposed
// SMU 13.0.x layout (gfx1102) that previously aborted amd-smi, plus the classic
// per-line layout with single, absent, and multiple current markers.

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi.h"

namespace amd::smi {
// Library-local test seam defined in src/rocm_smi.cc; reached through the static
// archive. Keep this signature in sync with the definition.
rsmi_status_t ParsePowerProfileMode(const std::vector<std::string>& lines,
                                    rsmi_power_profile_status_t* status,
                                    std::map<rsmi_power_profile_preset_masks_t, uint32_t>* ind_map);
}  // namespace amd::smi

namespace {

// Splits a raw pp_power_profile_mode blob into lines, matching how the library
// reads the sysfs file.
std::vector<std::string> Lines(const std::string& blob) {
  std::vector<std::string> out;
  std::istringstream fs(blob);
  std::string line;
  while (std::getline(fs, line)) {
    out.push_back(line);
  }
  return out;
}

// gfx1102 (Navi33, SMU 13.0.x): every profile and the '*' current marker are on
// the first line, followed by one row per tunable parameter. Captured verbatim
// from a failing Zuul run (trimmed to a few parameter rows).
constexpr char kTransposedGfx1102[] =
    "                    0 BOOTUP_DEFAULT* 1 3D_FULL_SCREEN  2 POWER_SAVING    3 "
    "VIDEO           4 VR              5 COMPUTE         6 CUSTOM          7 "
    "WINDOW_3D\n"
    "Gfx_ActiveHystLimit 0  0  0  0  0  0  0  0\n"
    "Gfx_IdleHystLimit   0  2  0  0  1  1  0  2\n"
    "Fclk_BoosterFreq    0  0  0  0  0  0  0  0\n";

// Classic per-line layout: text header, then one profile per line with the '*'
// current marker on the active profile's own line.
constexpr char kClassicSingleMarker[] =
    "NUM        MODE_NAME     BUSY_SET_POINT FPS UseRlcBusy\n"
    "  0   BOOTUP_DEFAULT:    0 0 1\n"
    "  1   3D_FULL_SCREEN:    0 0 1\n"
    "  2   POWER_SAVING*:     0 0 1\n"
    "  3          VIDEO:      0 0 1\n";

// Classic layout where the driver marks no profile current.
constexpr char kClassicNoMarker[] =
    "NUM        MODE_NAME\n"
    "  0   BOOTUP_DEFAULT:\n"
    "  1   3D_FULL_SCREEN:\n";

// Classic layout with more than one '*' marker -- the first must win.
constexpr char kClassicMultiMarker[] =
    "NUM        MODE_NAME\n"
    "  0   BOOTUP_DEFAULT*:\n"
    "  1   3D_FULL_SCREEN:\n"
    "  2   POWER_SAVING*:\n";

constexpr rsmi_bit_field_t kSevenKnown =
    RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT | RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK |
    RSMI_PWR_PROF_PRST_POWER_SAVING_MASK | RSMI_PWR_PROF_PRST_VIDEO_MASK |
    RSMI_PWR_PROF_PRST_VR_MASK | RSMI_PWR_PROF_PRST_COMPUTE_MASK | RSMI_PWR_PROF_PRST_CUSTOM_MASK;

// The regression: the transposed layout previously left current INVALID and
// aborted. It must now report BOOTUP_DEFAULT current with all listed profiles.
TEST(PowerProfileParse, TransposedGfx1102ReadsCurrentAndAvailable) {
  rsmi_power_profile_status_t status{};
  std::map<rsmi_power_profile_preset_masks_t, uint32_t> ind_map;
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(Lines(kTransposedGfx1102), &status, &ind_map),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT);
  // All 8 listed profiles are counted; WINDOW_3D has no rsmi mask, so the
  // available bitfield holds the other 7.
  EXPECT_EQ(status.num_profiles, 8u);
  EXPECT_EQ(status.available_profiles, kSevenKnown);
  EXPECT_EQ(ind_map[RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT], 0u);
  EXPECT_EQ(ind_map[RSMI_PWR_PROF_PRST_COMPUTE_MASK], 5u);
}

// Same transposed layout without the '*': current unknown, profiles still
// parsed, and it must not abort.
TEST(PowerProfileParse, TransposedNoMarkerReportsUnknown) {
  std::string blob = kTransposedGfx1102;
  blob.erase(blob.find('*'), 1);
  rsmi_power_profile_status_t status{};
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(Lines(blob), &status, nullptr), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_INVALID);
  EXPECT_EQ(status.available_profiles, kSevenKnown);
}

TEST(PowerProfileParse, ClassicSingleMarker) {
  rsmi_power_profile_status_t status{};
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(Lines(kClassicSingleMarker), &status, nullptr),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_POWER_SAVING_MASK);
  EXPECT_EQ(status.available_profiles,
            RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT | RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK |
                RSMI_PWR_PROF_PRST_POWER_SAVING_MASK | RSMI_PWR_PROF_PRST_VIDEO_MASK);
}

TEST(PowerProfileParse, ClassicNoMarkerReportsUnknownWithoutAbort) {
  rsmi_power_profile_status_t status{};
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(Lines(kClassicNoMarker), &status, nullptr),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_INVALID);
  EXPECT_EQ(status.available_profiles,
            RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT | RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK);
}

TEST(PowerProfileParse, ClassicMultipleMarkersKeepFirst) {
  rsmi_power_profile_status_t status{};
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(Lines(kClassicMultiMarker), &status, nullptr),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT);
}

}  // namespace
