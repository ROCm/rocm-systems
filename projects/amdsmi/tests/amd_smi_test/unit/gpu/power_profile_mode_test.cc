// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for the pp_power_profile_mode parser behind
// rsmi_dev_power_profile_presets_get() / amdsmi_get_gpu_power_profile_presets().
// Driven over real sysfs captures -- no GPU required at test time. The transposed
// SMU 13.0.7 fixtures were captured verbatim from a Navi 33 (gfx1102, device
// 0x7480, host navi33-4) with each named profile forced current in turn; the
// classic fixture is a verbatim Navi 21 (gfx1030, 0x73bf) capture.

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi.h"

namespace amd::smi {
rsmi_status_t ParsePowerProfileMode(const std::vector<std::string>& lines,
                                    rsmi_power_profile_status_t* status,
                                    std::map<rsmi_power_profile_preset_masks_t, uint32_t>* ind_map);
}  // namespace amd::smi

namespace {

std::vector<std::string> Lines(const std::string& blob) {
  std::vector<std::string> out;
  std::istringstream fs(blob);
  std::string line;
  while (std::getline(fs, line)) {
    out.push_back(line);
  }
  return out;
}

// The seven presets modelled before WINDOW_3D was added (used by the classic
// Navi 21 layout, which does not list WINDOW_3D).
constexpr rsmi_bit_field_t kSevenKnown =
    RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT | RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK |
    RSMI_PWR_PROF_PRST_POWER_SAVING_MASK | RSMI_PWR_PROF_PRST_VIDEO_MASK |
    RSMI_PWR_PROF_PRST_VR_MASK | RSMI_PWR_PROF_PRST_COMPUTE_MASK | RSMI_PWR_PROF_PRST_CUSTOM_MASK;

// All eight profiles the gfx1102 transposed table lists, now that WINDOW_3D has
// an rsmi preset.
constexpr rsmi_bit_field_t kEightKnown = kSevenKnown | RSMI_PWR_PROF_PRST_WINDOW_3D_MASK;

// gfx1102 (Navi 33, SMU 13.0.7) transposed layout: every profile and the '*'
// current marker are on the first line ("%d %-14s%s"), followed by one row per
// tunable parameter. BOOTUP_DEFAULT is exactly 14 chars, so its '*' abuts the
// name. Captured with BOOTUP_DEFAULT current.
constexpr char kGfx1102BootupCurrent[] =
    "                              0 BOOTUP_DEFAULT* 1 3D_FULL_SCREEN  2 POWER_SAVING    3 VIDEO   "
    "        4 VR              5 COMPUTE         6 CUSTOM          7 WINDOW_3D       \n"
    "Gfx_ActiveHystLimit           0                 0                 0                 0         "
    "        0                 0                 0                 0                 \n"
    "Gfx_IdleHystLimit             0                 2                 0                 0         "
    "        1                 1                 0                 2                 \n";

// Same card with COMPUTE current. COMPUTE is 7 chars, so %-14s left-justifies it
// and the driver prints "COMPUTE       * " -- the '*' is its own whitespace token.
constexpr char kGfx1102ComputeCurrent[] =
    "                              0 BOOTUP_DEFAULT  1 3D_FULL_SCREEN  2 POWER_SAVING    3 VIDEO   "
    "        4 VR              5 COMPUTE       * 6 CUSTOM          7 WINDOW_3D       \n"
    "Gfx_ActiveHystLimit           0                 0                 0                 0         "
    "        0                 0                 0                 0                 \n"
    "Gfx_IdleHystLimit             0                 2                 0                 0         "
    "        1                 1                 0                 2                 \n";

// Same card with VIDEO current (5 chars) -- another separated-marker case.
constexpr char kGfx1102VideoCurrent[] =
    "                              0 BOOTUP_DEFAULT  1 3D_FULL_SCREEN  2 POWER_SAVING    3 VIDEO   "
    "      * 4 VR              5 COMPUTE         6 CUSTOM          7 WINDOW_3D       \n"
    "Gfx_ActiveHystLimit           0                 0                 0                 0         "
    "        0                 0                 0                 0                 \n"
    "Gfx_IdleHystLimit             0                 2                 0                 0         "
    "        1                 1                 0                 2                 \n";

// Same card with WINDOW_3D current. WINDOW_3D is now a modelled preset (0x80),
// so it is reported as current and listed in available_profiles.
constexpr char kGfx1102Window3dCurrent[] =
    "                              0 BOOTUP_DEFAULT  1 3D_FULL_SCREEN  2 POWER_SAVING    3 VIDEO   "
    "        4 VR              5 COMPUTE         6 CUSTOM          7 WINDOW_3D     * \n"
    "Gfx_ActiveHystLimit           0                 0                 0                 0         "
    "        0                 0                 0                 0                 \n"
    "Gfx_IdleHystLimit             0                 2                 0                 0         "
    "        1                 1                 0                 2                 \n";

TEST(GpuUnit, PowerProfileTransposedBootupDefaultCurrent) {
  rsmi_power_profile_status_t status{};
  std::map<rsmi_power_profile_preset_masks_t, uint32_t> ind_map;
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(Lines(kGfx1102BootupCurrent), &status, &ind_map),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT);
  EXPECT_EQ(status.num_profiles, 8u);
  EXPECT_EQ(status.available_profiles, kEightKnown);
  EXPECT_EQ(ind_map[RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT], 0u);
  EXPECT_EQ(ind_map[RSMI_PWR_PROF_PRST_COMPUTE_MASK], 5u);
  EXPECT_EQ(ind_map[RSMI_PWR_PROF_PRST_WINDOW_3D_MASK], 7u);
}

// Short current-profile name -> the '*' is its own token. COMPUTE must be current,
// the driver index kept, and the stray marker must not be counted.
TEST(GpuUnit, PowerProfileTransposedComputeCurrentSeparatedMarker) {
  rsmi_power_profile_status_t status{};
  std::map<rsmi_power_profile_preset_masks_t, uint32_t> ind_map;
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(Lines(kGfx1102ComputeCurrent), &status, &ind_map),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_COMPUTE_MASK);
  EXPECT_EQ(status.num_profiles, 8u);
  EXPECT_EQ(status.available_profiles, kEightKnown);
  EXPECT_EQ(ind_map[RSMI_PWR_PROF_PRST_COMPUTE_MASK], 5u);
}

TEST(GpuUnit, PowerProfileTransposedVideoCurrentSeparatedMarker) {
  rsmi_power_profile_status_t status{};
  std::map<rsmi_power_profile_preset_masks_t, uint32_t> ind_map;
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(Lines(kGfx1102VideoCurrent), &status, &ind_map),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_VIDEO_MASK);
  EXPECT_EQ(status.num_profiles, 8u);
  EXPECT_EQ(status.available_profiles, kEightKnown);
  EXPECT_EQ(ind_map[RSMI_PWR_PROF_PRST_VIDEO_MASK], 3u);
}

// WINDOW_3D current: now a modelled preset, so it is reported (not INVALID) and
// its separated '*' marker is attributed to it, with the driver index kept.
TEST(GpuUnit, PowerProfileTransposedWindow3dCurrent) {
  rsmi_power_profile_status_t status{};
  std::map<rsmi_power_profile_preset_masks_t, uint32_t> ind_map;
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(Lines(kGfx1102Window3dCurrent), &status, &ind_map),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_WINDOW_3D_MASK);
  EXPECT_EQ(status.num_profiles, 8u);
  EXPECT_EQ(status.available_profiles, kEightKnown);
  EXPECT_EQ(ind_map[RSMI_PWR_PROF_PRST_WINDOW_3D_MASK], 7u);
}

// No profile marked current (defensive path the driver never produces): current
// unknown, profiles still parsed, no abort.
TEST(GpuUnit, PowerProfileTransposedNoMarkerReportsUnknown) {
  std::string blob = kGfx1102BootupCurrent;
  blob.erase(blob.find('*'), 1);
  rsmi_power_profile_status_t status{};
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(Lines(blob), &status, nullptr), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_INVALID);
  EXPECT_EQ(status.num_profiles, 8u);
  EXPECT_EQ(status.available_profiles, kEightKnown);
}

// Classic per-line layout: text header, then one profile per line with the '*'
// current marker on the active profile's own line.
constexpr char kClassicSingleMarker[] =
    "NUM        MODE_NAME     BUSY_SET_POINT FPS UseRlcBusy\n"
    "  0   BOOTUP_DEFAULT:    0 0 1\n"
    "  1   3D_FULL_SCREEN:    0 0 1\n"
    "  2   POWER_SAVING*:     0 0 1\n"
    "  3          VIDEO:      0 0 1\n";

constexpr char kClassicNoMarker[] =
    "NUM        MODE_NAME\n"
    "  0   BOOTUP_DEFAULT:\n"
    "  1   3D_FULL_SCREEN:\n";

constexpr char kClassicMultiMarker[] =
    "NUM        MODE_NAME\n"
    "  0   BOOTUP_DEFAULT*:\n"
    "  1   3D_FULL_SCREEN:\n"
    "  2   POWER_SAVING*:\n";

// Real Navi 21 (Sienna Cichlid, 0x73bf) capture: PROFILE_INDEX(NAME) header, seven
// "%2d %14s%s:" profile lines (BOOTUP_DEFAULT current), each followed by three
// CLOCK_TYPE detail rows. Navi 21 does not expose WINDOW_3D.
constexpr char kClassicNavi21RealCapture[] =
    "PROFILE_INDEX(NAME) CLOCK_TYPE(NAME) FPS MinFreqType MinActiveFreqType MinActiveFreq "
    "BoosterFreqType BoosterFreq PD_Data_limit_c PD_Data_error_coeff PD_Data_error_rate_coeff\n"
    " 0 BOOTUP_DEFAULT*:\n"
    "                    0(       GFXCLK)       0       5       1       0       4     800 4587520  "
    "-65536       0\n"
    "                    1(       SOCCLK)       0       5       1       0       1       0 3276800  "
    "-65536   -6553\n"
    "                    2(       MEMCLK)       0       5       1       0       4     800  327680  "
    "-65536       0\n"
    " 1 3D_FULL_SCREEN :\n"
    "                    0(       GFXCLK)       0       5       0    1600       4     650 5242880  "
    " -3276       0\n"
    "                    1(       SOCCLK)       0       5       1       0       1       0  655360  "
    "-65536   -6553\n"
    "                    2(       MEMCLK)       0       5       4     850       4     800  327680  "
    "-65536       0\n"
    " 2   POWER_SAVING :\n"
    "                    0(       GFXCLK)       0       5       1       0       3       0 5898240  "
    "-65536       0\n"
    "                    1(       SOCCLK)       0       5       1       0       1       0 3407872  "
    "-65536   -6553\n"
    "                    2(       MEMCLK)       0       5       1       0       3       0 1966080  "
    "-65536       0\n"
    " 3          VIDEO :\n"
    "                    0(       GFXCLK)       0       5       1       0       4     500 4587520  "
    "-65536       0\n"
    "                    1(       SOCCLK)       0       5       1       0       1       0 3473408  "
    "-65536   -6553\n"
    "                    2(       MEMCLK)       0       5       1       0       4     500 1966080  "
    "-65536       0\n"
    " 4             VR :\n"
    "                    0(       GFXCLK)       0       5       4    1000       1       0 3276800  "
    "     0       0\n"
    "                    1(       SOCCLK)       0       5       1       0       1       0  655360  "
    "-65536   -6553\n"
    "                    2(       MEMCLK)       0       5       1       0       4     800  327680  "
    "-65536       0\n"
    " 5        COMPUTE :\n"
    "                    0(       GFXCLK)       0       5       1       0       4     800 4587520  "
    "-65536       0\n"
    "                    1(       SOCCLK)       0       5       1       0       1       0 3276800  "
    "-65536   -6553\n"
    "                    2(       MEMCLK)       0       5       1       0       4     800  327680  "
    "-65536       0\n"
    " 6         CUSTOM :\n"
    "                    0(       GFXCLK)       0       5       1       0       4     800 4587520  "
    "-65536       0\n"
    "                    1(       SOCCLK)       0       5       1       0       1       0 3276800  "
    "-65536   -6553\n"
    "                    2(       MEMCLK)       0       5       1       0       4     800  327680  "
    "-65536       0\n";

TEST(GpuUnit, PowerProfileClassicSingleMarker) {
  rsmi_power_profile_status_t status{};
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(Lines(kClassicSingleMarker), &status, nullptr),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_POWER_SAVING_MASK);
  EXPECT_EQ(status.available_profiles,
            RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT | RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK |
                RSMI_PWR_PROF_PRST_POWER_SAVING_MASK | RSMI_PWR_PROF_PRST_VIDEO_MASK);
}

TEST(GpuUnit, PowerProfileClassicNoMarkerReportsUnknownWithoutAbort) {
  rsmi_power_profile_status_t status{};
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(Lines(kClassicNoMarker), &status, nullptr),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_INVALID);
  EXPECT_EQ(status.available_profiles,
            RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT | RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK);
}

TEST(GpuUnit, PowerProfileClassicMultipleMarkersKeepLast) {
  // The driver marks exactly one profile current; a table with several markers
  // is malformed. The classic path keeps the last marker seen, matching this
  // path's long-standing release behavior (unchanged here).
  rsmi_power_profile_status_t status{};
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(Lines(kClassicMultiMarker), &status, nullptr),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_POWER_SAVING_MASK);
}

TEST(GpuUnit, PowerProfileClassicNavi21RealCapture) {
  const std::vector<std::string> lines = Lines(kClassicNavi21RealCapture);
  rsmi_power_profile_status_t status{};
  std::map<rsmi_power_profile_preset_masks_t, uint32_t> ind_map;
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(lines, &status, &ind_map), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT);
  EXPECT_EQ(status.available_profiles, kSevenKnown);
  EXPECT_EQ(ind_map[RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT], 0u);
  EXPECT_EQ(ind_map[RSMI_PWR_PROF_PRST_CUSTOM_MASK], 6u);
  EXPECT_EQ(status.num_profiles, static_cast<uint32_t>(lines.size() - 1));
}

// A first line that only starts with digits but is not a contiguous 0..N-1
// index run is not the transposed layout; is_transposed_power_profile_mode()
// must fall through to the classic parser rather than misread it. Here the
// first line puts two indices in a row with no name between them, so it is
// treated as a classic header and the two following rows are parsed per-line.
constexpr char kDigitFirstNonMonotonic[] =
    "0 1 5 9 spurious digit-first header\n"
    "  0   BOOTUP_DEFAULT:    0 0 1\n"
    "  1   COMPUTE*:          0 0 1\n";

TEST(GpuUnit, PowerProfileDigitFirstButNonMonotonicIsClassic) {
  rsmi_power_profile_status_t status{};
  EXPECT_EQ(amd::smi::ParsePowerProfileMode(Lines(kDigitFirstNonMonotonic), &status, nullptr),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(status.num_profiles, 2u);
  EXPECT_EQ(status.current, RSMI_PWR_PROF_PRST_COMPUTE_MASK);
  EXPECT_EQ(status.available_profiles,
            RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT | RSMI_PWR_PROF_PRST_COMPUTE_MASK);
}

}  // namespace
