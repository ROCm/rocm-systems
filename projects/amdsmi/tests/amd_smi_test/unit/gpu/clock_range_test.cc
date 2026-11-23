// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Hardware-free clock tests. OdClkRange* and DpmRanges* drive the
// pp_od_clk_voltage / pp_dpm_* parsers over in-memory streams. DpmFrequency*,
// DpmClockInfo* and DpmMixed* drive amdsmi_get_clk_freq / amdsmi_get_clock_info
// on fake sysfs GPUs; add new behavior cases there.
//
// MI45x pp_od_clk_voltage has no OD_FCLK section (kOdNoFclk), so FCLK min/max
// must fall back to pp_dpm_fclk instead of reporting a max of 0 MHz.

#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_clk_testing.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "rocm_smi/rocm_smi_gpu_metrics.h"
#include "test_fixture_utils.h"

namespace {

using amd::smi::AMDSmiDrm;
using amd::smi::AMDSmiGPUDevice;
using amd::smi::testing::ScopedOverride;
using amd::smi::testing::ScopedProcessorRegistration;

// pp_od_clk_voltage as seen on MI45x: SCLK and MCLK sections plus an OD_RANGE
// footer, but no OD_FCLK section.
constexpr char kOdNoFclk[] =
    "OD_SCLK:\n"
    "0: 500Mhz\n"
    "1: 2100Mhz\n"
    "OD_MCLK:\n"
    "0: 900Mhz\n"
    "1: 1200Mhz\n"
    "OD_RANGE:\n"
    "SCLK:     500Mhz        2100Mhz\n"
    "MCLK:     900Mhz        1200Mhz\n";

// Same layout with an explicit OD_FCLK section present.
constexpr char kOdWithFclk[] =
    "OD_SCLK:\n"
    "0: 500Mhz\n"
    "1: 2100Mhz\n"
    "OD_FCLK:\n"
    "0: 1000Mhz\n"
    "1: 1100Mhz\n";

// Bare GFXCLK/MCLK/FCLK aliases in place of the OD_* headers, as some GPUs emit.
constexpr char kOdAliasHeaders[] =
    "GFXCLK:\n"
    "0: 500Mhz\n"
    "1: 2100Mhz\n"
    "MCLK:\n"
    "0: 900Mhz\n"
    "1: 1200Mhz\n"
    "FCLK:\n"
    "0: 1000Mhz\n"
    "1: 1100Mhz\n";

// OD_FCLK present but every level reads 0 -- no usable range.
constexpr char kOdFclkAllZero[] =
    "OD_FCLK:\n"
    "0: 0Mhz\n"
    "1: 0Mhz\n";

// OD_FCLK with a non-conforming line between two valid levels.
constexpr char kOdFclkGarbageLine[] =
    "OD_FCLK:\n"
    "0: 1000Mhz\n"
    "not-a-level\n"
    "1: 1100Mhz\n";

// "Old Format" layout: OD_FCLK is immediately followed by OD_VDDC_CURVE, whose
// "idx: freq volt" lines resemble level lines and must not fold into the range.
constexpr char kOdFclkThenCurve[] =
    "OD_SCLK:\n"
    "0: 500Mhz\n"
    "1: 2100Mhz\n"
    "OD_FCLK:\n"
    "0: 1000Mhz\n"
    "1: 1100Mhz\n"
    "OD_VDDC_CURVE:\n"
    "0: 500Mhz 700mV\n"
    "1: 1354Mhz 860mV\n"
    "2: 2000Mhz 1150mV\n";

// Two OD_FCLK sections: levels from every occurrence merge into one range.
constexpr char kOdFclkDuplicate[] =
    "OD_FCLK:\n"
    "0: 1000Mhz\n"
    "1: 1100Mhz\n"
    "OD_SCLK:\n"
    "0: 500Mhz\n"
    "OD_FCLK:\n"
    "0: 900Mhz\n"
    "1: 2000Mhz\n";

// A single-level (locked-clock) section: min == max.
constexpr char kOdFclkSingleLevel[] =
    "OD_FCLK:\n"
    "0: 1500Mhz\n";

TEST(GpuUnit, OdClkRangeReadsSclkSection) {
  std::istringstream od(kOdNoFclk);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_GFX, &max, &min));
  EXPECT_EQ(max, 2100u);
  EXPECT_EQ(min, 500u);
}

TEST(GpuUnit, OdClkRangeReadsMclkSection) {
  std::istringstream od(kOdNoFclk);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_MEM, &max, &min));
  EXPECT_EQ(max, 1200u);
  EXPECT_EQ(min, 900u);
}

// The regression: with no OD_FCLK section the parser reports "absent" (false)
// and leaves the out-params untouched, so the caller derives FCLK min/max from
// pp_dpm_fclk rather than reporting 0.
TEST(GpuUnit, OdClkRangeFallsBackWhenFclkSectionMissing) {
  std::istringstream od(kOdNoFclk);
  unsigned int max = 4242;
  unsigned int min = 4242;
  EXPECT_FALSE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 4242u);
  EXPECT_EQ(min, 4242u);
}

// With OD_FCLK present the range is read directly, no fallback.
TEST(GpuUnit, OdClkRangeReadsFclkSectionWhenPresent) {
  std::istringstream od(kOdWithFclk);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 1100u);
  EXPECT_EQ(min, 1000u);
}

// A missing/empty pp_od_clk_voltage yields no section -> fall back.
TEST(GpuUnit, OdClkRangeFallsBackOnEmptyStream) {
  std::istringstream od("");
  unsigned int max = 7;
  unsigned int min = 7;
  EXPECT_FALSE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_GFX, &max, &min));
}

// Domains that have no overdrive section (e.g. SOC) are never OD-backed.
TEST(GpuUnit, OdClkRangeRejectsNonOdDomain) {
  std::istringstream od(kOdWithFclk);
  unsigned int max = 7;
  unsigned int min = 7;
  EXPECT_FALSE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_SOC, &max, &min));
}

// The bare GFXCLK/MCLK/FCLK aliases are accepted just like the OD_* headers.
TEST(GpuUnit, OdClkRangeReadsAliasHeaders) {
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  std::istringstream gfx(kOdAliasHeaders);
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(gfx, AMDSMI_CLK_TYPE_GFX, &max, &min));
  EXPECT_EQ(max, 2100u);
  EXPECT_EQ(min, 500u);

  max = 0;
  min = UINT_MAX;
  std::istringstream mem(kOdAliasHeaders);
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(mem, AMDSMI_CLK_TYPE_MEM, &max, &min));
  EXPECT_EQ(max, 1200u);
  EXPECT_EQ(min, 900u);

  max = 0;
  min = UINT_MAX;
  std::istringstream df(kOdAliasHeaders);
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(df, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 1100u);
  EXPECT_EQ(min, 1000u);
}

// Section present but every level parses to 0 -> reported absent so the caller
// still falls back to pp_dpm_*.
TEST(GpuUnit, OdClkRangeFallsBackWhenAllLevelsZero) {
  std::istringstream od(kOdFclkAllZero);
  unsigned int max = 555;
  unsigned int min = 555;
  EXPECT_FALSE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 555u);
  EXPECT_EQ(min, 555u);
}

// A non-conforming line inside the section is skipped, not treated as its end,
// so levels on both sides still contribute to the range.
TEST(GpuUnit, OdClkRangeSkipsMalformedLineWithinSection) {
  std::istringstream od(kOdFclkGarbageLine);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 1100u);
  EXPECT_EQ(min, 1000u);
}

// An unrecognized section header (OD_VDDC_CURVE) ends FCLK parsing, so its curve
// lines are not mistaken for FCLK levels and do not inflate the max.
TEST(GpuUnit, OdClkRangeStopsAtUnrecognizedSectionHeader) {
  std::istringstream od(kOdFclkThenCurve);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 1100u);
  EXPECT_EQ(min, 1000u);
}

// Repeated section headers merge: levels from every OD_FCLK occurrence feed one
// range. No live sysfs file repeats a header; this pins the contract.
TEST(GpuUnit, OdClkRangeMergesRepeatedSectionHeaders) {
  std::istringstream od(kOdFclkDuplicate);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 2000u);
  EXPECT_EQ(min, 900u);
}

// A single-level (locked-clock) section reports min == max.
TEST(GpuUnit, OdClkRangeReadsSingleLevelSection) {
  std::istringstream od(kOdFclkSingleLevel);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 1500u);
  EXPECT_EQ(min, 1500u);
}

// --- smi_amdgpu_parse_dpm_ranges(): the pp_dpm_* fold + bounds guard ---

// A pp_dpm_sclk stream with dpm levels but no "S:" sleep line: the sleep freq
// keeps its UINT_MAX "unavailable" sentinel, which must NOT be rejected as out
// of bounds. Regression: a prior guard failed the whole call on that sentinel,
// so amdsmi_get_clock_info() returned INPUT_OUT_OF_BOUNDS on every GPU without a
// deep-sleep level.
constexpr char kDpmSclkNoSleep[] =
    "0: 500Mhz\n"
    "1: 1500Mhz *\n"
    "2: 2100Mhz\n";

TEST(GpuUnit, DpmRangesUnavailableSleepSentinelNotRejected) {
  std::istringstream dpm(kDpmSclkNoSleep);
  // GFX min/max come from the OD range (500..2100); there is no "S:" line.
  SmiAmdgpuOdClkRange od_range{true, 2100u, 500u};
  SmiAmdgpuClkRanges ranges;
  EXPECT_EQ(smi_amdgpu_parse_dpm_ranges(dpm, od_range, ranges), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(ranges.max_freq, 2100);
  EXPECT_EQ(ranges.min_freq, 500);
  // Unset sleep state surfaces as the unavailable marker (UINT_MAX -> -1 as int).
  EXPECT_EQ(ranges.sleep_state_freq, -1);
}

// An empty pp_dpm_* stream with no overdrive range: min keeps its UINT_MAX
// "unavailable" sentinel and surfaces as -1 rather than failing the call.
TEST(GpuUnit, DpmRangesEmptyStreamKeepsUnavailableMin) {
  std::istringstream dpm("");
  SmiAmdgpuOdClkRange od_range;  // not present -> derive from the (absent) levels
  SmiAmdgpuClkRanges ranges;
  EXPECT_EQ(smi_amdgpu_parse_dpm_ranges(dpm, od_range, ranges), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(ranges.max_freq, 0);
  EXPECT_EQ(ranges.min_freq, -1);
  EXPECT_EQ(ranges.sleep_state_freq, -1);
}

// --- Public clock APIs on fake sysfs GPUs ---

constexpr char kDpmFclkValid[] = "0: 1300Mhz\n1: 1550Mhz *\n2: 1800Mhz\n";

// Blank or whitespace-only lines: malformed driver data (the Raphael iGPU
// returns "\n").
constexpr const char* kDpmFclkBlank[] = {"\n", " \t\r\n", "\n\n", "\n0: 1300Mhz\n",
                                         "0: 1300Mhz\n\n"};

const char* OdLabel(const char* od) { return od == kOdWithFclk ? "kOdWithFclk" : "kOdNoFclk"; }

// amdsmi_init without discovery, a fake sysfs tree behind the DRM root override,
// and registered GPUs card0, card1, ... in the order AddGpu() is called.
class FakeClockHost {
 public:
  FakeClockHost()
      : root_guard_(amd::smi::test_hooks::SysfsDrmRoot, amd::smi::test_hooks::SetSysfsDrmRoot,
                    tree_.root()) {}

  // Release in reverse: each ScopedRocmSmiDevice truncates RocmSMI's device
  // vector to its own index.
  ~FakeClockHost() {
    while (!gpus_.empty()) gpus_.pop_back();
  }

  // Adds the next card with valid gpu_metrics (amdsmi_get_clock_info reads them
  // even for FCLK) and pp_od_clk_voltage `od`, but no pp_dpm_fclk.
  amdsmi_processor_handle AddGpu(const char* od) {
    const size_t gpu = gpus_.size();
    amd::smi::AMDGpuMetrics_v10_t metrics{};
    metrics.m_common_header = {sizeof(metrics), 1, 0};
    WriteDeviceFile(gpu, "gpu_metrics",
                    std::string(reinterpret_cast<const char*>(&metrics), sizeof(metrics)));
    SetOd(gpu, od);
    gpus_.push_back(std::make_unique<Gpu>(tree_.root(), Card(gpu), drm_));
    return gpus_.back()->registration.handle();
  }

  void SetOd(size_t gpu, const char* od) { WriteDeviceFile(gpu, "pp_od_clk_voltage", od); }
  void SetDpmFclk(size_t gpu, const char* dpm) { WriteDeviceFile(gpu, "pp_dpm_fclk", dpm); }

 private:
  struct Gpu {
    Gpu(const std::string& root, const std::string& card, AMDSmiDrm& drm)
        : rocm(root + "/" + card),
          device(rocm.index(), card, amdsmi_bdf_t{}, drm),
          registration(&device) {}
    ScopedRocmSmiDevice rocm;
    AMDSmiGPUDevice device;
    ScopedProcessorRegistration registration;
  };

  static std::string Card(size_t gpu) { return "card" + std::to_string(gpu); }

  void WriteDeviceFile(size_t gpu, const std::string& name, const std::string& contents) {
    tree_.WriteFile(Card(gpu) + "/device/" + name, contents);
  }

  ScopedAmdsmiInit init_;
  FakeSysfsTree tree_;
  ScopedOverride<std::string> root_guard_;
  AMDSmiDrm drm_;
  std::vector<std::unique_ptr<Gpu>> gpus_;
};

TEST(GpuUnit, DpmFrequencyMissingTableNotSupported) {
  FakeClockHost host;
  const auto gpu = host.AddGpu(kOdNoFclk);
  amdsmi_frequencies_t frequencies{};
  EXPECT_EQ(amdsmi_get_clk_freq(gpu, AMDSMI_CLK_TYPE_DF, &frequencies),
            AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, DpmFrequencyEmptyTablesPreserveDataError) {
  FakeClockHost host;
  const auto gpu = host.AddGpu(kOdNoFclk);
  for (const char* contents : {"", "\n", " \t\r\n", "\n\n"}) {
    SCOPED_TRACE(::testing::PrintToString(std::string(contents)));
    host.SetDpmFclk(0, "0: 1300Mhz *\n");
    amdsmi_frequencies_t frequencies{};
    ASSERT_EQ(amdsmi_get_clk_freq(gpu, AMDSMI_CLK_TYPE_DF, &frequencies), AMDSMI_STATUS_SUCCESS);
    ASSERT_EQ(frequencies.num_supported, 1u);

    host.SetDpmFclk(0, contents);
    EXPECT_EQ(amdsmi_get_clk_freq(gpu, AMDSMI_CLK_TYPE_DF, &frequencies),
              AMDSMI_STATUS_UNEXPECTED_DATA);
    EXPECT_EQ(frequencies.num_supported, 0u);
  }
}

TEST(GpuUnit, DpmFrequencyValidTableStillReportsLevels) {
  FakeClockHost host;
  const auto gpu = host.AddGpu(kOdNoFclk);
  host.SetDpmFclk(0, kDpmFclkValid);
  amdsmi_frequencies_t frequencies{};
  ASSERT_EQ(amdsmi_get_clk_freq(gpu, AMDSMI_CLK_TYPE_DF, &frequencies), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(frequencies.num_supported, 3u);
  EXPECT_EQ(frequencies.current, 1u);
  EXPECT_EQ(frequencies.frequency[0], 1300000000u);
  EXPECT_EQ(frequencies.frequency[1], 1550000000u);
  EXPECT_EQ(frequencies.frequency[2], 1800000000u);
}

// A valid OD_FCLK range must not turn a blank pp_dpm_fclk into SUCCESS. Clock
// info only: rsmi asserts on the leading-blank row in Debug builds.
TEST(GpuUnit, DpmClockInfoBlankTablesReportIo) {
  FakeClockHost host;
  const auto gpu = host.AddGpu(kOdNoFclk);
  for (const char* od : {kOdNoFclk, kOdWithFclk}) {
    host.SetOd(0, od);
    for (const char* dpm : kDpmFclkBlank) {
      SCOPED_TRACE(::testing::Message() << OdLabel(od) << ", pp_dpm_fclk "
                                        << ::testing::PrintToString(std::string(dpm)));
      host.SetDpmFclk(0, dpm);
      amdsmi_clk_info_t info{};
      EXPECT_EQ(amdsmi_get_clock_info(gpu, AMDSMI_CLK_TYPE_DF, &info), AMDSMI_STATUS_IO);
    }
  }
}

// Unlike "\n", a zero-byte pp_dpm_fclk has always returned SUCCESS: the OD_FCLK
// range when present, otherwise max 0 and an unavailable min.
TEST(GpuUnit, DpmClockInfoZeroByteTableKeepsExistingResult) {
  FakeClockHost host;
  const auto gpu = host.AddGpu(kOdNoFclk);
  host.SetDpmFclk(0, "");
  amdsmi_clk_info_t info{};
  ASSERT_EQ(amdsmi_get_clock_info(gpu, AMDSMI_CLK_TYPE_DF, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.max_clk, 0u);
  EXPECT_EQ(info.min_clk, UINT32_MAX);

  host.SetOd(0, kOdWithFclk);
  ASSERT_EQ(amdsmi_get_clock_info(gpu, AMDSMI_CLK_TYPE_DF, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.max_clk, 1100u);
  EXPECT_EQ(info.min_clk, 1000u);
}

TEST(GpuUnit, DpmMixedDevicesPreservePublicApiErrors) {
  FakeClockHost host;
  const auto dgpu = host.AddGpu(kOdNoFclk);
  const auto igpu = host.AddGpu(kOdNoFclk);
  host.SetDpmFclk(0, kDpmFclkValid);
  host.SetDpmFclk(1, "\n");

  amdsmi_frequencies_t frequencies{};
  ASSERT_EQ(amdsmi_get_clk_freq(dgpu, AMDSMI_CLK_TYPE_DF, &frequencies), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(frequencies.num_supported, 3u);
  EXPECT_EQ(frequencies.current, 1u);
  amdsmi_clk_info_t info{};
  ASSERT_EQ(amdsmi_get_clock_info(dgpu, AMDSMI_CLK_TYPE_DF, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.min_clk, 1300u);
  EXPECT_EQ(info.max_clk, 1800u);

  // These distinct errors capture current newline-only behavior, not a
  // prescribed cross-API status contract.
  for (const char* od : {kOdNoFclk, kOdWithFclk}) {
    SCOPED_TRACE(OdLabel(od));
    host.SetOd(1, od);
    EXPECT_EQ(amdsmi_get_clk_freq(igpu, AMDSMI_CLK_TYPE_DF, &frequencies),
              AMDSMI_STATUS_UNEXPECTED_DATA);
    EXPECT_EQ(frequencies.num_supported, 0u);
    EXPECT_EQ(amdsmi_get_clock_info(igpu, AMDSMI_CLK_TYPE_DF, &info), AMDSMI_STATUS_IO);
  }
}

// A genuinely out-of-range value (> INT_MAX, not the sentinel) is still
// rejected, preserving the protection added in #9549.
constexpr char kDpmSclkHugeFreq[] = "0: 3000000000Mhz\n";  // 3e9 > INT_MAX

TEST(GpuUnit, DpmRangesRealOutOfRangeStillRejected) {
  std::istringstream dpm(kDpmSclkHugeFreq);
  SmiAmdgpuOdClkRange od_range;  // not present -> derive min/max from the level
  SmiAmdgpuClkRanges ranges;
  EXPECT_EQ(smi_amdgpu_parse_dpm_ranges(dpm, od_range, ranges), AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS);
}

// With an "S:" sleep line present the deep-sleep frequency is parsed and returned.
constexpr char kDpmSclkWithSleep[] =
    "S: 100Mhz\n"
    "0: 500Mhz\n"
    "1: 2100Mhz *\n";

TEST(GpuUnit, DpmRangesReadsSleepStateWhenPresent) {
  std::istringstream dpm(kDpmSclkWithSleep);
  SmiAmdgpuOdClkRange od_range{true, 2100u, 500u};
  SmiAmdgpuClkRanges ranges;
  EXPECT_EQ(smi_amdgpu_parse_dpm_ranges(dpm, od_range, ranges), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(ranges.sleep_state_freq, 100);
}
// A real (non-sentinel) sleep value above INT_MAX is rejected too: the guard
// covers the "S:" sleep line, not just the dpm levels.
constexpr char kDpmSclkHugeSleep[] =
    "S: 3000000000Mhz\n"
    "0: 500Mhz\n";

TEST(GpuUnit, DpmRangesHugeSleepValueRejected) {
  std::istringstream dpm(kDpmSclkHugeSleep);
  SmiAmdgpuOdClkRange od_range;  // not present -> derive min/max from the level
  SmiAmdgpuClkRanges ranges;
  EXPECT_EQ(smi_amdgpu_parse_dpm_ranges(dpm, od_range, ranges), AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS);
}

}  // namespace
